# =============================================================================
# backend/app/retriever.py — Multi-Stage Knowledge Graph Retrieval Pipeline
# =============================================================================

import re
from typing import List, Optional, Dict, Set

from .config import logger
from .embeddings import get_embedding
from .query_parser import (
    _sanitize_lucene_query,
    _extract_path_hints,
    _extract_blame_hints,
    _extract_impact_subjects,
    _extract_field_hints,
    _extract_repo_hints_from_query,
    extract_query_intent,
    QueryIntent,
)


def _build_context_block(
    records: list[dict],
    source: str,
    allow_no_code: bool = False,
    global_seen: set[str] | None = None,
) -> str:
    """Format a list of retrieval records into clearly labelled plain-text sections."""
    if not records:
        return ""

    lines: list[str] = [f"[Source: {source}]"]
    seen_ids: set[str] = set()

    for rec in records:
        name     = rec.get("name") or rec.get("func_name") or "<unknown>"
        filepath = rec.get("filepath") or rec.get("path") or ""
        code     = (rec.get("code") or "").strip()
        uid      = f"{filepath}::{name}"

        if uid in seen_ids or (global_seen is not None and uid in global_seen):
            continue
        if not code and not allow_no_code:
            continue
        seen_ids.add(uid)
        if global_seen is not None:
            global_seen.add(uid)

        lines.append(f"\n{'─'*60}")
        label = rec.get("seed_label") or rec.get("type") or rec.get("consumer_type") or "Function"
        if label in ["Module", "Repository", "Package"]:
            lines.append(f"{label:<8} : {name}")
            if filepath and filepath != "<unknown path>" and filepath != name:
                lines.append(f"File     : {filepath}")
        elif label in ["Type", "Variable", "Directive", "File"]:
            lines.append(f"{label:<8} : {name}")
            lines.append(f"File     : {filepath or '<unknown path>'}")
        else:
            lines.append(f"Function : {name}")
            lines.append(f"File     : {filepath or '<unknown path>'}")
        repo = rec.get("repo") or rec.get("repository") or ""
        if repo:
            lines.append(f"Repo     : {repo}")
        rel = rec.get("rel_type") or ""
        if rel:
            lines.append(f"Relation : {rel}")
        if code:
            lines.append("Code     :")
            lines.append(code)
        else:
            lines.append("Code     : <not stored — structural reference only>")

        for conn in rec.get("connected") or []:
            c_name = conn.get("name") or "<unknown>"
            c_path = conn.get("path") or conn.get("filepath") or ""
            c_code = (conn.get("code") or "").strip()
            c_uid  = f"{c_path}::{c_name}"
            if c_uid in seen_ids or (global_seen is not None and c_uid in global_seen) or not c_code:
                continue
            seen_ids.add(c_uid)
            if global_seen is not None:
                global_seen.add(c_uid)
            lines.append(f"\n  ↳ Called/Dep: {c_name}  [{c_path}]")
            lines.append(f"  {c_code.replace(chr(10), chr(10)+'  ')}")

    return "\n".join(lines)


def _extract_relevant_lines(code: str, term: str, context: int = 6) -> str:
    """Return the full function body with lines that contain `term` marked with '>>>'."""
    if not code or not term:
        return code or ""

    term_lower = term.lower()
    all_lines  = code.splitlines()
    hit_set    = {i for i, ln in enumerate(all_lines) if term_lower in ln.lower()}

    if not hit_set:
        return code

    marked = []
    for i, ln in enumerate(all_lines):
        prefix = ">>> " if i in hit_set else "    "
        marked.append(f"{prefix}{ln}")
    return "\n".join(marked)


def _fetch_subject_definition(
    driver,
    subject: str,
    selected_repos: Optional[List[str]] = None,
) -> Optional[dict]:
    """Look up the definition of the named subject in the graph."""
    repo_filter = "AND (fn.repo IN $selected_repos OR fn.full_name IN $selected_repos OR fn.name IN $selected_repos)" if selected_repos else ""

    defn_cypher = f"""
    MATCH (fn)
    WHERE (fn:Function OR fn:File OR fn:Type OR fn:Variable OR fn:Directive OR fn:Module OR fn:Repository)
      AND (
          toLower(coalesce(fn.name, '')) CONTAINS toLower($subject)
       OR toLower(coalesce(fn.full_name, '')) CONTAINS toLower($subject)
       OR (fn.code IS NOT NULL AND toLower(fn.code) CONTAINS toLower($subject))
      )
      {repo_filter}
    RETURN
      coalesce(fn.name, fn.full_name, fn.path, '') AS name,
      coalesce(fn.filepath, fn.path, '')     AS filepath,
      coalesce(fn.repo, fn.full_name, '')    AS repo,
      coalesce(fn.code, '')                  AS code,
      'SUBJECT_DEFINITION'                   AS rel_type,
      labels(fn)[0]                          AS seed_label
    ORDER BY case when toLower(coalesce(fn.name, fn.full_name, '')) = toLower($subject) then 0 else 1 end, size(coalesce(fn.code, '')) DESC
    LIMIT 3
    """
    try:
        with driver.session() as session:
            rows = session.run(
                defn_cypher,
                subject=subject.lower(),
                selected_repos=selected_repos or [],
            ).data()
        if rows:
            logger.info(
                "[IMPACT] Subject definition for %r found: %d candidate(s).",
                subject, len(rows),
            )
            best = max(rows, key=lambda r: len(r.get("code") or ""))
            
            # ─── GENERIC STRUCT TRUNCATION STEP INJECTED HERE ───
            raw_code = best.get("code", "")
            lines = raw_code.splitlines()
            if len(lines) > 50:
                # Crop the context size dynamically to protect the LLM's token attention span
                best["code"] = "\n".join(lines[:40]) + f"\n\n... [TRUNCATED: {len(lines)-40} structural lines omitted for context optimization. See local file mapping for full implementation details] ..."
            # ───────────────────────────────────────────────────────

            best["connected"] = []
            return best
    except Exception as exc:
        logger.warning("[IMPACT] Subject definition lookup for %r failed: %s", subject, exc)
    return None


def _build_blame_context_block(records: list[dict], global_seen: set[str] | None = None) -> str:
    """Format blame records into a clearly labelled plain-text section for the LLM."""
    if not records:
        return ""

    lines: list[str] = ["[Source: blame-traversal]"]
    seen: set[str] = set()

    for rec in records:
        author      = rec.get("author") or "<unknown>"
        sha         = rec.get("commit_sha") or "<no sha>"
        msg         = (rec.get("commit_msg") or "").strip().splitlines()[0]
        date        = rec.get("committed_at") or ""
        func_name   = rec.get("func_name") or ""
        filepath    = rec.get("filepath") or "<unknown path>"

        uid = f"{sha}::{filepath}::{func_name}"
        if uid in seen or (global_seen is not None and uid in global_seen):
            continue
        seen.add(uid)
        if global_seen is not None:
            global_seen.add(uid)

        lines.append(f"\n{'─'*60}")
        lines.append(f"Author     : {author}")
        lines.append(f"Commit SHA : {sha}")
        lines.append(f"Date       : {date}")
        lines.append(f"Message    : {msg}")
        if func_name:
            lines.append(f"Function   : {func_name}")
        lines.append(f"File       : {filepath}")

    return "\n".join(lines)


def _run_impact_traversal(
    driver,
    subjects: List[str],
    selected_repos: Optional[List[str]] = None,
) -> list[dict]:
    """Traverse the graph to find every File and Function that depends on/imports/calls the subjects."""
    results: list[dict] = []
    seen_ids: set[str] = set()

    for subject in subjects:
        subject_lower = subject.lower()

        seed_cypher = """
        MATCH (seed)
        WHERE (
            seed:Repository OR seed:Module OR seed:File OR seed:Function OR seed:Type OR seed:Variable OR seed:Directive
        )
        AND NOT toLower(coalesce(seed.path, seed.filepath, '')) =~ '.*\\.(md|txt|json|yaml|yml|html|css|rst|toml)$'
        AND (
            toLower(coalesce(seed.name, ''))     CONTAINS $subject
            OR toLower(coalesce(seed.path, ''))  CONTAINS $subject
            OR toLower(coalesce(seed.full_name, '')) CONTAINS $subject
        )
        RETURN
          elementId(seed)                                         AS seed_id,
          labels(seed)[0]                                  AS seed_label,
          coalesce(seed.name, seed.path, seed.full_name)   AS seed_name,
          coalesce(seed.path, seed.filepath, seed.full_name, '') AS seed_path
        LIMIT 20
        """
        try:
            with driver.session() as session:
                seed_rows = session.run(seed_cypher, subject=subject_lower).data()
        except Exception as exc:
            logger.warning("[IMPACT] Seed query for %r failed: %s", subject, exc)
            seed_rows = []

        logger.info(
            "[IMPACT] Subject=%r → %d seed node(s) found.", subject, len(seed_rows)
        )

        if not seed_rows:
            fallback_cypher = """
            MATCH (f:File)
            WHERE toLower(f.path) CONTAINS $subject
              AND NOT toLower(f.path) =~ '.*\\.(md|txt|json|yaml|yml|html|css|rst|toml)$'
            RETURN
              elementId(f)     AS seed_id,
              'File'    AS seed_label,
              f.path    AS seed_name,
              f.path    AS seed_path
            LIMIT 10
            """
            try:
                with driver.session() as session:
                    seed_rows = session.run(fallback_cypher, subject=subject_lower).data()
                logger.info(
                    "[IMPACT] Fallback file-path search %r → %d result(s).",
                    subject, len(seed_rows),
                )
            except Exception as exc:
                logger.warning("[IMPACT] Fallback query for %r failed: %s", subject, exc)

        for seed in seed_rows:
            seed_id   = seed["seed_id"]
            seed_name = seed["seed_name"]
            seed_path = seed["seed_path"]

            repo_filter = (
                "AND (consumer.repo IN $selected_repos OR consumer.full_name IN $selected_repos OR consumer.name IN $selected_repos)"
                if selected_repos else ""
            )
            consumer_cypher = f"""
            MATCH (consumer)-[rel:DEPENDS_ON|CALLS|USES_REPO|IMPLEMENTS|WRAPS|PRODUCES|LIFECYCLE_HOOK|REGISTERS_WITH|DISPATCHES_TO|FORWARDS_TO|EMBEDS|DECLARES_METHOD|DECLARES|DECLARES_TYPE|DECLARES_VAR|HAS_DIRECTIVE]->(seed)
            WHERE elementId(seed) = $seed_id
              AND (consumer:File OR consumer:Function OR consumer:Module OR consumer:Type OR consumer:Repository)
              {repo_filter}
            WITH coalesce(consumer.repo, 'local / unscoped') AS repository,
                 type(rel) AS dependency_type,
                 labels(consumer)[0] AS consumer_type,
                 consumer
            RETURN repository,
                   dependency_type,
                   consumer_type,
                   count(consumer) AS total_impacted_items,
                   collect(DISTINCT case 
                       when consumer:Function then coalesce(consumer.filepath, consumer.path, '') + ' :: ' + consumer.name 
                       else coalesce(consumer.path, consumer.filepath, '') 
                   end)[0..25] AS top_sample_locations
            ORDER BY total_impacted_items DESC
            """
            try:
                with driver.session() as session:
                    consumer_rows = session.run(
                        consumer_cypher,
                        seed_id=seed_id,
                        selected_repos=selected_repos or [],
                    ).data()
                logger.info(
                    "[IMPACT] Seed id=%s (%s) → %d aggregated repo group(s).",
                    seed_id, seed_name, len(consumer_rows),
                )
            except Exception as exc:
                logger.warning("[IMPACT] Consumer aggregation traversal for seed id=%s failed: %s", seed_id, exc)
                consumer_rows = []

            seed_uid = f"{seed_path}::{seed_name}"
            if seed_uid not in seen_ids:
                seen_ids.add(seed_uid)
                results.append({
                    "name":     seed_name,
                    "filepath": seed_path,
                    "repo":     "",
                    "rel_type": "SUBJECT (being moved/refactored)",
                    "code":     "",
                    "connected": [],
                })

            for row in consumer_rows:
                repo_name = row.get("repository") or "unknown"
                dep_type = row.get("dependency_type") or "DEPENDS_ON"
                cons_type = row.get("consumer_type") or "Node"
                count_items = row.get("total_impacted_items") or 0
                samples = row.get("top_sample_locations") or []
                
                uid = f"agg::{repo_name}::{dep_type}::{seed_name}"
                if uid not in seen_ids:
                    seen_ids.add(uid)
                    sample_str = "\n".join(f"  • {loc}" for loc in samples)
                    omitted_cnt = count_items - len(samples)
                    omitted_note = f"\n  ... and {omitted_cnt} additional dependent items not shown in this summary." if omitted_cnt > 0 else ""
                    results.append({
                        "name":     f"Repo: {repo_name} ({count_items} impacted {cons_type}s)",
                        "filepath": f"Aggregated Impact Summary | Repo: {repo_name}",
                        "repo":     repo_name,
                        "rel_type": f"AGGREGATED_{dep_type} ({count_items} items affected)",
                        "code":     f"The graph reports exactly {count_items} dependent items in repository '{repo_name}'.\nRetrieved sample locations ({len(samples)} of {count_items} shown):\n{sample_str}{omitted_note}",
                        "connected": [],
                    })

            # ── Cross-Repo Multi-Hop Traversal ─────────────────────────────────
            multihop_cypher = f"""
            MATCH (entry:Function)-[:CALLS|DISPATCHES_TO|FORWARDS_TO*1..3]->(callee:Function)-[:CALLS|DEPENDS_ON|IMPLEMENTS|WRAPS|PRODUCES|LIFECYCLE_HOOK|REGISTERS_WITH|DISPATCHES_TO|FORWARDS_TO|EMBEDS|DECLARES_METHOD]->(seed)
            WHERE elementId(seed) = $seed_id
              AND (
                  coalesce(entry.repo, '') <> coalesce(seed.repo, '') 
                  OR entry.entry_point = true 
                  OR entry.name IN ['main', 'init', 'execute', 'run', 'ServeHTTP']
                  OR entry.name STARTS WITH 'Test'
              )
              {repo_filter.replace('consumer.', 'entry.')}
            RETURN coalesce(entry.repo, 'local') AS repository,
                   coalesce(entry.filepath, entry.path, '') AS entrypoint_file,
                   entry.name AS entrypoint_func,
                   count(*) AS call_paths
            ORDER BY call_paths DESC
            LIMIT 15
            """
            try:
                with driver.session() as session:
                    multihop_rows = session.run(
                        multihop_cypher,
                        seed_id=seed_id,
                        selected_repos=selected_repos or [],
                    ).data()
                logger.info(
                    "[IMPACT] Seed id=%s (%s) → %d multi-hop entrypoint(s).",
                    seed_id, seed_name, len(multihop_rows),
                )
            except Exception as exc:
                logger.warning("[IMPACT] Multi-hop traversal for seed id=%s failed: %s", seed_id, exc)
                multihop_rows = []

            for row in multihop_rows:
                repo_name = row.get("repository") or "unknown"
                efile = row.get("entrypoint_file") or ""
                efunc = row.get("entrypoint_func") or "<unknown>"
                paths_cnt = row.get("call_paths") or 1
                
                uid = f"mh::{repo_name}::{efile}::{efunc}"
                if uid not in seen_ids:
                    seen_ids.add(uid)
                    results.append({
                        "name":     efunc,
                        "filepath": efile,
                        "repo":     repo_name,
                        "rel_type": f"MULTI_HOP_ENTRYPOINT (via {paths_cnt} transitive call path(s))",
                        "code":     f"Critical entrypoint or top-level workflow in repository '{repo_name}' that transitively depends on '{seed_name}'.",
                        "connected": [],
                    })

            reverse_cypher = f"""
            MATCH (seed)-[rel:DEPENDS_ON|CALLS|IMPLEMENTS|WRAPS|PRODUCES|LIFECYCLE_HOOK|REGISTERS_WITH|DISPATCHES_TO|FORWARDS_TO|EMBEDS|DECLARES_METHOD]->(dep)
            WHERE elementId(seed) = $seed_id
              AND (dep:File OR dep:Function OR dep:Module OR dep:Type OR dep:Repository)
              {repo_filter.replace('consumer.', 'dep.')}
            RETURN
              dep.name                                         AS name,
              coalesce(dep.filepath, dep.path, '')             AS filepath,
              coalesce(dep.repo, '')                           AS repo,
              type(rel)                                        AS rel_type_raw,
              coalesce(dep.code, '')                           AS code
            LIMIT 20
            """
            try:
                with driver.session() as session:
                    dep_rows = session.run(
                        reverse_cypher,
                        seed_id=seed_id,
                        selected_repos=selected_repos or [],
                    ).data()
                logger.info(
                    "[IMPACT] Seed id=%s outbound deps → %d node(s).",
                    seed_id, len(dep_rows),
                )
            except Exception as exc:
                logger.warning("[IMPACT] Reverse traversal for seed id=%s failed: %s", seed_id, exc)
                dep_rows = []

            for row in dep_rows:
                uid = f"{row.get('filepath')}::{row.get('name')}"
                if uid not in seen_ids:
                    seen_ids.add(uid)
                    results.append({
                        "name":     row.get("name") or "<unknown>",
                        "filepath": row.get("filepath") or "",
                        "repo":     row.get("repo") or "",
                        "rel_type": "OUTBOUND_" + (row.get("rel_type_raw") or "EDGE"),
                        "code":     row.get("code") or "",
                        "connected": [],
                    })

    return results


def _run_structural_traversal(
    driver,
    subjects: List[str],
    target_repos: Optional[List[str]] = None,
    source_repos: Optional[List[str]] = None,
) -> list[dict]:
    """Find structs/types that implement interfaces defined in `subjects`.

    Strategy:
    1. Locate Interface/Type nodes whose name or repo path contains any subject term.
    2. Traverse IMPLEMENTS edges outward to find implementing structs.
    3. Optionally filter implementors by target_repos.
    4. For each implementor, fetch the DECLARES_METHOD edges to surface the
       specific methods that satisfy the interface contract.
    5. As a fallback, text-grep Function nodes whose code contains both a
       subject interface name and a method receiver pattern.
    """
    results: list[dict] = []
    seen_ids: set[str] = set()

    for subject in subjects:
        subject_lower = subject.lower()

        # ── Step 1: locate interface / type seed nodes ────────────────────────
        iface_cypher = """
        MATCH (iface)
        WHERE (iface:Type OR iface:Function)
          AND (
              toLower(coalesce(iface.name, ''))      CONTAINS $subject
           OR toLower(coalesce(iface.full_name, '')) CONTAINS $subject
           OR toLower(coalesce(iface.path, ''))      CONTAINS $subject
          )
        OPTIONAL MATCH (iface)-[:DECLARES_METHOD]->(meth:Function)
        WITH iface, collect(DISTINCT {name: meth.name, code: coalesce(meth.code, '')})[..20] AS iface_methods
        RETURN
          elementId(iface)                                           AS iface_id,
          coalesce(iface.name, iface.full_name, iface.path)         AS iface_name,
          coalesce(iface.kind, labels(iface)[0])                    AS iface_kind,
          coalesce(iface.code, '')                                   AS iface_code,
          coalesce(iface.filepath, iface.path, iface.full_name, '') AS iface_path,
          coalesce(iface.repo, '')                                   AS iface_repo,
          coalesce(iface.method_names, [])                          AS iface_method_names,
          iface_methods
        LIMIT 30
        """
        try:
            with driver.session() as session:
                iface_rows = session.run(iface_cypher, subject=subject_lower).data()
        except Exception as exc:
            logger.warning("[STRUCTURAL] Interface seed query for %r failed: %s", subject, exc)
            iface_rows = []

        logger.info(
            "[STRUCTURAL] subject=%r → %d interface/type seed(s).", subject, len(iface_rows)
        )

        # Always emit the interface definition itself as context.
        # C2: when code is absent, synthesise a contract block from method_names /
        # materialised DECLARES_METHOD Function nodes so the LLM is never blind.
        for iface in iface_rows:
            uid = f"iface::{iface.get('iface_path')}::{iface.get('iface_name')}"
            if uid not in seen_ids:
                seen_ids.add(uid)

                raw_code      = (iface.get("iface_code") or "").strip()
                method_names  = iface.get("iface_method_names") or []
                iface_methods = iface.get("iface_methods") or []

                # Build a synthetic interface contract block if code is missing
                if not raw_code:
                    # Prefer full signatures from materialised Function nodes
                    sigs = [
                        m["code"].strip()
                        for m in iface_methods
                        if m.get("name") and m.get("code", "").strip()
                    ]
                    if not sigs and method_names:
                        sigs = [f"    {mn}(...)" for mn in method_names]
                    if sigs:
                        iface_name = iface.get("iface_name") or subject
                        raw_code = (
                            f"type {iface_name} interface {{\n"
                            + "\n".join(f"    {s}" if not s.startswith("    ") else s for s in sigs)
                            + "\n}"
                        )

                # Build connected list from materialised DECLARES_METHOD methods
                connected = [
                    {"name": m.get("name"), "path": iface.get("iface_path"), "code": m.get("code")}
                    for m in iface_methods
                    if m.get("name") and m.get("code", "").strip()
                ]

                results.append({
                    "name":      iface.get("iface_name") or "<unknown>",
                    "filepath":  iface.get("iface_path") or "",
                    "repo":      iface.get("iface_repo") or "",
                    "rel_type":  "INTERFACE_DEFINITION",
                    "code":      raw_code,
                    "connected": connected,
                })

        # ── Step 2 & 3: find implementors via IMPLEMENTS edges ─────────────────
        target_repo_filter = (
            "AND (impl.repo IN $target_repos OR impl.full_name IN $target_repos OR impl.name IN $target_repos)"
            if target_repos else ""
        )
        for iface in iface_rows:
            iface_id = iface["iface_id"]
            iface_name = iface.get("iface_name") or subject

            impl_cypher = f"""
            MATCH (impl:Type)-[:IMPLEMENTS]->(iface)
            WHERE elementId(iface) = $iface_id
              {target_repo_filter}
            OPTIONAL MATCH (impl)-[:DECLARES_METHOD]->(meth:Function)
            WITH impl,
                 collect(DISTINCT {{
                   name: meth.name,
                   path: coalesce(meth.filepath, meth.path, ''),
                   code: coalesce(meth.code, '')
                 }})[..12] AS declared_methods
            RETURN
              impl.name                                   AS name,
              coalesce(impl.filepath, impl.path, '')      AS filepath,
              coalesce(impl.repo, '')                     AS repo,
              impl.kind                                   AS kind,
              coalesce(impl.code, '')                     AS code,
              impl.method_names                           AS impl_method_names,
              declared_methods
            ORDER BY size(coalesce(impl.code, '')) DESC
            LIMIT 20
            """
            try:
                with driver.session() as session:
                    impl_rows = session.run(
                        impl_cypher,
                        iface_id=iface_id,
                        target_repos=target_repos or [],
                    ).data()
                logger.info(
                    "[STRUCTURAL] iface=%r → %d implementor(s).", iface_name, len(impl_rows)
                )
            except Exception as exc:
                logger.warning("[STRUCTURAL] Implementor query for iface %r failed: %s", iface_name, exc)
                impl_rows = []

            for row in impl_rows:
                uid = f"impl::{row.get('filepath')}::{row.get('name')}"
                if uid not in seen_ids:
                    seen_ids.add(uid)
                    connected = [
                        {"name": m.get("name"), "path": m.get("path"), "code": m.get("code")}
                        for m in (row.get("declared_methods") or [])
                        if m.get("name")
                    ]
                    results.append({
                        "name":      row.get("name") or "<unknown>",
                        "filepath":  row.get("filepath") or "",
                        "repo":      row.get("repo") or "",
                        "rel_type":  f"IMPLEMENTS_{iface_name.upper()}",
                        "code":      row.get("code") or "",
                        "connected": connected,
                    })

        # ── Step 4: Text/code-grep fallback ───────────────────────────────────
        # Search for Function nodes whose code mentions the subject (interface name)
        # and that live in the target repos — catches cases where IMPLEMENTS edges
        # were not explicitly ingested but the code clearly references the interface.
        repo_grep_filter = (
            "AND (fn.repo IN $target_repos OR fn.full_name IN $target_repos)"
            if target_repos else ""
        )
        grep_cypher = f"""
        MATCH (fn:Function)
        WHERE fn.code IS NOT NULL
          AND toLower(fn.code) CONTAINS toLower($subject)
          {repo_grep_filter}
        RETURN
          fn.name                                AS name,
          coalesce(fn.filepath, fn.path, '')     AS filepath,
          coalesce(fn.repo, '')                  AS repo,
          fn.code                                AS code
        ORDER BY size(fn.code) DESC
        LIMIT 15
        """
        try:
            with driver.session() as session:
                grep_rows = session.run(
                    grep_cypher,
                    subject=subject_lower,
                    target_repos=target_repos or [],
                ).data()
            logger.info(
                "[STRUCTURAL] code-grep for subject=%r → %d function(s).", subject, len(grep_rows)
            )
        except Exception as exc:
            logger.warning("[STRUCTURAL] code-grep for %r failed: %s", subject, exc)
            grep_rows = []

        for row in grep_rows:
            uid = f"grep::{row.get('filepath')}::{row.get('name')}"
            if uid not in seen_ids:
                seen_ids.add(uid)
                results.append({
                    "name":      row.get("name") or "<unknown>",
                    "filepath":  row.get("filepath") or "",
                    "repo":      row.get("repo") or "",
                    "rel_type":  "CODE_REFERENCES_INTERFACE",
                    "code":      _extract_relevant_lines(row.get("code") or "", subject),
                    "connected": [],
                })

        # ── Step 5: Type node grep — look for structs in target repos that ─────
        # declare methods whose names match the known cobra interface methods.
        type_grep_cypher = f"""
        MATCH (t:Type)
        WHERE t.code IS NOT NULL
          AND toLower(t.code) CONTAINS toLower($subject)
          {repo_grep_filter.replace('fn.', 't.')}
        OPTIONAL MATCH (t)-[:DECLARES_METHOD|EMBEDS]->(m:Function)
        WITH t, collect(DISTINCT {{
          name: m.name,
          path: coalesce(m.filepath, m.path, ''),
          code: coalesce(m.code, '')
        }})[..8] AS methods
        RETURN
          coalesce(t.name, t.full_name)          AS name,
          coalesce(t.filepath, t.repo, '')       AS filepath,
          coalesce(t.repo, '')                   AS repo,
          coalesce(t.kind, 'Type')               AS kind,
          coalesce(t.code, '')                   AS code,
          methods
        ORDER BY size(coalesce(t.code, '')) DESC
        LIMIT 10
        """
        try:
            with driver.session() as session:
                type_rows = session.run(
                    type_grep_cypher,
                    subject=subject_lower,
                    target_repos=target_repos or [],
                ).data()
            logger.info(
                "[STRUCTURAL] type-grep for subject=%r → %d type(s).", subject, len(type_rows)
            )
        except Exception as exc:
            logger.warning("[STRUCTURAL] type-grep for %r failed: %s", subject, exc)
            type_rows = []

        for row in type_rows:
            uid = f"type_grep::{row.get('filepath')}::{row.get('name')}"
            if uid not in seen_ids:
                seen_ids.add(uid)
                connected = [
                    {"name": m.get("name"), "path": m.get("path"), "code": m.get("code")}
                    for m in (row.get("methods") or [])
                    if m.get("name")
                ]
                results.append({
                    "name":      row.get("name") or "<unknown>",
                    "filepath":  row.get("filepath") or "",
                    "repo":      row.get("repo") or "",
                    "rel_type":  "TYPE_REFERENCES_INTERFACE",
                    "code":      row.get("code") or "",
                    "connected": connected,
                })

    return results


def retrieve_code_context(
    user_query: str,
    driver,
    google_api_key: str,
    selected_repos: Optional[List[str]] = None,
    top_k: int = 5,
) -> tuple[str, Optional[QueryIntent]]:
    """Multi-stage retrieval pipeline: Stage 0a Structural → Stage 0 Impact → Stage 1 Vector → Stage 2 Path Boost → Stage 3 Commit → Stage 4 Blame."""
    logger.info("[RETRIEVE] ── New retrieval ────────────────────────────")
    logger.info("[RETRIEVE] Query        : %r", user_query[:200])
    logger.info("[RETRIEVE] Selected repos: %s", selected_repos or "<all>")
    logger.info("[RETRIEVE] top_k        : %d", top_k)

    query_vector  = get_embedding(user_query, google_api_key)
    path_hints    = _extract_path_hints(user_query)
    intent        = extract_query_intent(user_query, google_api_key)
    context_parts: list[str] = []
    retrieval_path_tags: list[str] = []
    global_seen_ids: set[str] = set()
    vector_rows:   list[dict] = []

    if intent and (intent.wants_impact or intent.wants_blame or intent.wants_commit_files or intent.wants_recency or intent.wants_structural or intent.subjects):
        intent_summary_lines = [
            "[Source: user-query-intent | LLM Intent Classification]",
            f"Primary Intent : Impact={intent.wants_impact} | Blame={intent.wants_blame} | Commit={intent.wants_commit_files} | Recency={intent.wants_recency} | Structural={intent.wants_structural}",
            f"Target Subjects: {intent.subjects or 'none explicitly detected'}",
            f"Modified Symbols/Keys: {intent.field_hints or 'none detected'}",
            f"Target Repos   : {intent.repo_hints or 'all / unscoped'}",
        ]
        context_parts.append("\n".join(intent_summary_lines))
        logger.info("[RETRIEVE] Prepended [Source: user-query-intent] metadata block.")

    # ── Stage 0a: Structural / Interface-satisfaction traversal ──────────────
    _is_structural_query = intent.wants_structural
    if _is_structural_query:
        structural_subjects = intent.subjects or _extract_impact_subjects(user_query)
        structural_repo_hints = intent.repo_hints or _extract_repo_hints_from_query(user_query)
        _effective_repos_s0a = selected_repos or []

        # Separate source repos (where interfaces are defined) from target repos
        # (where implementing structs live). If the query mentions two distinct
        # repos like "spf13/cobra" and "gohugoio/hugo", we treat the first as
        # source and the second as target. Otherwise apply all hints to both.
        _source_repos: Optional[List[str]] = None
        _target_repos: Optional[List[str]] = None
        if structural_repo_hints and len(structural_repo_hints) >= 2:
            _source_repos = [structural_repo_hints[0]]
            _target_repos = structural_repo_hints[1:]
        elif structural_repo_hints:
            _target_repos = structural_repo_hints

        _target_repos = _effective_repos_s0a or _target_repos

        logger.info(
            "[RETRIEVE] Stage 0a — structural traversal triggered. "
            "subjects=%s  source_repos=%s  target_repos=%s",
            structural_subjects, _source_repos, _target_repos,
        )

        if structural_subjects:
            structural_rows = _run_structural_traversal(
                driver,
                structural_subjects,
                target_repos=_target_repos or None,
                source_repos=_source_repos or None,
            )
            logger.info(
                "[RETRIEVE] Stage 0a returned %d structural record(s).", len(structural_rows)
            )
            if structural_rows:
                struct_block = _build_context_block(
                    structural_rows, "structural-interface", allow_no_code=True, global_seen=global_seen_ids
                )
                if struct_block:
                    context_parts.append(struct_block)
                    retrieval_path_tags.append("structural")
        else:
            logger.warning(
                "[RETRIEVE] ⚠️  Stage 0a: structural query detected but no subjects extracted."
            )
    else:
        logger.info("[RETRIEVE] Stage 0a skipped — not a structural/interface query.")

    # ── Stage 0: Dependency-impact traversal ─────────────────────────────────
    _query_lower_s0 = user_query.lower()
    _is_impact_query = intent.wants_impact
    if _is_impact_query:
        impact_subjects  = intent.subjects or _extract_impact_subjects(user_query)
        field_hints_s0   = intent.field_hints or _extract_field_hints(user_query)
        if field_hints_s0:
            _expanded_fh = []
            for h in field_hints_s0:
                _expanded_fh.append(h)
                if "." in h:
                    _expanded_fh.append(h.split(".")[-1])
            field_hints_s0 = list(dict.fromkeys(_expanded_fh))
        query_repo_hints = intent.repo_hints or _extract_repo_hints_from_query(user_query)

        _effective_repos = selected_repos or []
        _repo_hint_filter: Optional[List[str]] = None
        if query_repo_hints and not _effective_repos:
            _repo_hint_filter = query_repo_hints

        logger.info(
            "[RETRIEVE] Stage 0 — impact traversal triggered. "
            "subjects=%s  field_hints=%s  repo_hints=%s",
            impact_subjects, field_hints_s0, query_repo_hints,
        )
        if impact_subjects:
            impact_rows = _run_impact_traversal(
                driver, impact_subjects,
                selected_repos=_effective_repos or None,
            )

            _primary_subject = next(
                (s for s in impact_subjects if len(s) > 4), impact_subjects[0]
            ) if impact_subjects else None
            if _primary_subject:
                defn_record = _fetch_subject_definition(
                    driver, _primary_subject,
                    selected_repos=_effective_repos or None,
                )
                if defn_record:
                    logger.info(
                        "[RETRIEVE] Stage 0 — prepending subject definition for %r.",
                        _primary_subject,
                    )
                    impact_rows = [defn_record] + impact_rows

            if _repo_hint_filter and impact_rows:
                filtered = []
                for row in impact_rows:
                    row_repo = (row.get("repo") or "").lower()
                    row_path = (row.get("filepath") or "").lower()
                    if (
                        row.get("rel_type", "").startswith("SUBJECT")
                        or any(hint in row_repo or hint in row_path for hint in _repo_hint_filter)
                    ):
                        filtered.append(row)
                logger.info(
                    "[RETRIEVE] Stage 0 repo-hint post-filter: %d → %d row(s) (hints=%s).",
                    len(impact_rows), len(filtered), _repo_hint_filter,
                )
                impact_rows = filtered

            if field_hints_s0:
                for field in field_hints_s0:
                    repo_filter_fg = (
                        "AND (fn.repo IN $selected_repos OR fn.full_name IN $selected_repos OR fn.name IN $selected_repos)"
                        if _effective_repos else ""
                    )
                    if any(w in _query_lower_s0 for w in ["signature", "param", "arg", "return"]):
                        field_rel_label = f"REFERENCES_SIGNATURE_{field.upper()}"
                    elif any(w in _query_lower_s0 for w in ["method", "func", "function"]):
                        field_rel_label = f"REFERENCES_METHOD_{field.upper()}"
                    elif any(w in _query_lower_s0 for w in ["key", "config", "setting"]):
                        field_rel_label = f"REFERENCES_KEY_{field.upper()}"
                    else:
                        field_rel_label = f"REFERENCES_FIELD_{field.upper()}"
                    # Change the query variable to match the global 'fn' filter token
                    field_grep_cypher = f"""
                    MATCH (fn)
                    WHERE (fn:Function OR fn:Type)
                    AND fn.code IS NOT NULL
                    AND (
                        toLower(fn.code) CONTAINS toLower($field_name)
                        OR (fn:Type AND $field_name IN fn.field_names)
                    )
                    {repo_filter_fg}
                    RETURN
                    fn.name                                  AS name,
                    coalesce(fn.filepath, fn.path, '')       AS filepath,
                    coalesce(fn.repo, '')                    AS repo,
                    fn.code                                  AS code,
                    labels(fn)[0]                            AS seed_label
                    LIMIT 20
                    """
                    try:
                        with driver.session() as session:
                            field_rows = session.run(
                                field_grep_cypher,
                                field_name=field,
                                selected_repos=_effective_repos,
                            ).data()
                        
                        for r in field_rows:
                            r["rel_type"] = field_rel_label
                            r.setdefault("connected", [])
                            if r.get("code"):
                                r["code"] = _extract_relevant_lines(r["code"], field)
                        impact_rows.extend(field_rows)
                    except Exception as exc:
                        logger.warning("[RETRIEVE] Stage 0 symbol-grep '%s' failed: %s", field, exc)

            logger.info(
                "[RETRIEVE] Stage 0 returned %d impact record(s) (post-filter).", len(impact_rows)
            )
            if impact_rows:
                impact_block = _build_context_block(impact_rows, "dependency-impact", allow_no_code=True, global_seen=global_seen_ids)
                if impact_block:
                    context_parts.append(impact_block)
                    retrieval_path_tags.append("impact")
        else:
            logger.warning(
                "[RETRIEVE] ⚠️  Stage 0: impact query detected but no subjects extracted."
            )
    else:
        logger.info("[RETRIEVE] Stage 0 skipped — no impact/migration keywords in query.")

    # ── Stage 1: Vector search ────────────────────────────────────────────────
    if query_vector:
        repo_clause = (
            "AND (node.repo IN $selected_repos OR node.full_name IN $selected_repos OR node.name IN $selected_repos)"
            if selected_repos else ""
        )

        # 1a: Function-level semantic search (code_embeddings)
        vector_cypher = f"""
        CALL db.index.vector.queryNodes('code_embeddings', $top_k, $query_vector)
        YIELD node, score
        WITH node, score
        WHERE node.code IS NOT NULL {repo_clause}
        OPTIONAL MATCH (node)-[:CALLS|DEPENDS_ON|IMPLEMENTS|WRAPS|PRODUCES|LIFECYCLE_HOOK|REGISTERS_WITH|DISPATCHES_TO|FORWARDS_TO|EMBEDS|DECLARES_METHOD]->(connected)
        WHERE (connected:Function OR connected:File OR connected:Type)
          AND connected.code IS NOT NULL
          {"AND (connected.repo IN $selected_repos OR connected.full_name IN $selected_repos OR connected.name IN $selected_repos)" if selected_repos else ""}
        WITH node,
             score,
             collect(DISTINCT {{
               name:     connected.name,
               path:     coalesce(connected.filepath, connected.path),
               code:     connected.code
             }})[..4] AS connected_nodes
        RETURN
          node.name                                          AS name,
          coalesce(node.filepath, node.path)                 AS filepath,
          node.code                                          AS code,
          connected_nodes                                    AS connected,
          score
        ORDER BY score DESC
        LIMIT $top_k
        """

        # 1b: Module-level semantic search (module_embeddings).
        # Finds package/module nodes whose name embedding is semantically close to
        # the query — important for questions like "code related to color rendering"
        # that refer to packages rather than individual functions.
        module_vector_cypher = f"""
        CALL db.index.vector.queryNodes('module_embeddings', $mod_top_k, $query_vector)
        YIELD node, score
        WITH node, score
        WHERE node.embedding IS NOT NULL {
            "AND (node.repo IN $selected_repos OR node.name IN $selected_repos)" if selected_repos else ""
        }
        OPTIONAL MATCH (f:File)-[:DEPENDS_ON]->(node)
        {
            "WHERE f.repo IN $selected_repos" if selected_repos else ""
        }
        WITH node, score,
             collect(DISTINCT f.path)[..6] AS dependent_files
        RETURN
          node.name                   AS name,
          coalesce(node.repo, '')     AS filepath,
          '(module: ' + node.name + ')\\nDepended on by:\\n' + reduce(s='', p IN dependent_files | s + '  ' + p + '\\n') AS code,
          []                          AS connected,
          score
        ORDER BY score DESC
        LIMIT $mod_top_k
        """

        # 1c: type search
        type_vector_cypher = f"""
        CALL db.index.vector.queryNodes('type_embeddings', $mod_top_k, $query_vector)
        YIELD node, score
        WITH node, score
        WHERE node.code IS NOT NULL {
            "AND (node.repo IN $selected_repos OR node.name IN $selected_repos)" if selected_repos else ""
        }
        OPTIONAL MATCH (node)-[:EMBEDS|IMPLEMENTS|DECLARES_METHOD]->(connected)
        WHERE (connected:Function OR connected:Type OR connected:Module)
        WITH node, score,
             collect(DISTINCT {{
               name: connected.name,
               path: coalesce(connected.filepath, ''),
               code: coalesce(connected.code, '')
             }})[..4] AS connected_nodes
        RETURN
          node.name                              AS name,
          coalesce(node.filepath, node.repo, '') AS filepath,
          '(type: ' + node.name + ' [' + coalesce(node.kind, '') + '])\\n' + coalesce(node.code, '') AS code,
          connected_nodes                        AS connected,
          score
        ORDER BY score DESC
        LIMIT $mod_top_k
        """

        # 1d: var search
        var_vector_cypher = f"""
        CALL db.index.vector.queryNodes('var_embeddings', $mod_top_k, $query_vector)
        YIELD node, score
        WITH node, score
        WHERE node.code IS NOT NULL {
            "AND (node.repo IN $selected_repos OR node.name IN $selected_repos)" if selected_repos else ""
        }
        RETURN
          node.name                              AS name,
          coalesce(node.filepath, node.repo, '') AS filepath,
          '(var: ' + node.name + ' [' + coalesce(node.kind, '') + '])\\n' + coalesce(node.code, '') AS code,
          []                                     AS connected,
          score
        ORDER BY score DESC
        LIMIT $mod_top_k
        """

        logger.info(
            "[RETRIEVE] Stage 1 — vector search (top_k=%d, repo_filter=%s) …",
            top_k, bool(selected_repos),
        )
        try:
            with driver.session() as session:
                # 1a: function search
                rows = session.run(
                    vector_cypher,
                    query_vector=query_vector,
                    top_k=top_k,
                    selected_repos=selected_repos or [],
                ).data()

                # 1b: module search (half of top_k, minimum 3)
                mod_top_k = max(3, top_k // 2)
                mod_rows = session.run(
                    module_vector_cypher,
                    query_vector=query_vector,
                    mod_top_k=mod_top_k,
                    selected_repos=selected_repos or [],
                ).data()

                # 1c: type search
                try:
                    type_rows = session.run(
                        type_vector_cypher,
                        query_vector=query_vector,
                        mod_top_k=mod_top_k,
                        selected_repos=selected_repos or [],
                    ).data()
                except Exception:
                    type_rows = []

                # 1d: var search
                try:
                    var_rows = session.run(
                        var_vector_cypher,
                        query_vector=query_vector,
                        mod_top_k=mod_top_k,
                        selected_repos=selected_repos or [],
                    ).data()
                except Exception:
                    var_rows = []

            logger.info(
                "[RETRIEVE] Stage 1 returned %d func + %d mod + %d type + %d var record(s).",
                len(rows), len(mod_rows), len(type_rows), len(var_rows),
            )
            if rows:
                vector_rows = rows
                context_parts.append(_build_context_block(rows, "vector-search", global_seen=global_seen_ids))
                retrieval_path_tags.append("vector")
            if mod_rows:
                vector_rows.extend(mod_rows)
                context_parts.append(_build_context_block(mod_rows, "module-vector", global_seen=global_seen_ids))
                retrieval_path_tags.append("module-vector")
            if type_rows:
                vector_rows.extend(type_rows)
                context_parts.append(_build_context_block(type_rows, "type-vector", global_seen=global_seen_ids))
                retrieval_path_tags.append("type-vector")
            if var_rows:
                vector_rows.extend(var_rows)
                context_parts.append(_build_context_block(var_rows, "var-vector", global_seen=global_seen_ids))
                retrieval_path_tags.append("var-vector")
        except Exception as exc:
            logger.error("[RETRIEVE] ❌ Stage 1 vector search FAILED: %s", exc)

    else:
        logger.warning(
            "[RETRIEVE] ⚠️  No query vector — skipping Stage 1 entirely."
        )

    # ── Stage 2: Path-hint keyword boost ─────────────────────────────────────
    if path_hints:
        logger.info(
            "[RETRIEVE] Stage 2 — path-hint boost for: %s", path_hints
        )
        hint_rows: list[dict] = []
        for hint in path_hints:
            repo_filter_clause = (
                "AND (fn.repo IN $selected_repos OR fn.full_name IN $selected_repos OR fn.name IN $selected_repos)"
                if selected_repos else ""
            )
            hint_cypher = f"""
            MATCH (fn)
            WHERE (fn:Function OR fn:Type OR fn:Variable)
              AND toLower(fn.filepath) CONTAINS toLower($hint)
              AND fn.code IS NOT NULL
              {repo_filter_clause}
            RETURN
              fn.name     AS name,
              fn.filepath AS filepath,
              fn.code     AS code,
              []          AS connected
            LIMIT 8
            """
            try:
                with driver.session() as session:
                    rows = session.run(
                        hint_cypher,
                        hint=hint,
                        selected_repos=selected_repos or [],
                    ).data()
                logger.info(
                    "[RETRIEVE] Stage 2 hint=%r → %d function(s) found.",
                    hint, len(rows),
                )
                hint_rows.extend(rows)
            except Exception as exc:
                logger.warning(
                    "[RETRIEVE] Stage 2 hint=%r failed: %s", hint, exc
                )

        if hint_rows:
            context_parts.append(_build_context_block(hint_rows, "path-hint-boost", global_seen=global_seen_ids))
            retrieval_path_tags.append("path-hint")
        else:
            logger.warning(
                "[RETRIEVE] ⚠️  Stage 2 path-hints produced 0 results."
            )
    else:
        logger.info("[RETRIEVE] Stage 2 skipped — no path hints in query.")

    # ── Stage 3: Commit retrieval ──────────────────────────────────────────────
    _query_lower = user_query.lower()
    _needs_commit_context = (
        not context_parts
        or intent.wants_commit_files
    )
    _needs_recency_sort = intent.wants_recency

    lucene_query = _sanitize_lucene_query(user_query)
    fulltext_cypher = """
    CALL db.index.fulltext.queryNodes('commit_summaries', $search_query)
    YIELD node, score
    RETURN
      coalesce(node.sha, node.id, toString(elementId(node))) AS name,
      coalesce(node.repo, '')                          AS filepath,
      coalesce(
        node.summary_text,
        node.message,
        node.diff_text,
        ''
      )                                               AS code,
      []                                              AS connected
    LIMIT 10
    """

    if _needs_commit_context:
        if not context_parts:
            logger.warning(
                "[RETRIEVE] ⚠️  Stages 1+2 both empty — will use commit context."
            )
        else:
            logger.info(
                "[RETRIEVE] Stage 3 — augmenting with commit context."
            )

        if _needs_recency_sort:
            _repo_matches = re.findall(r'\b[\w\-]+/[\w\-]+\b', user_query)
            _repo_hint = _repo_matches[0] if _repo_matches else None

            _limit_match = re.search(r'\b(\d+)\b', user_query)
            _commit_limit = min(int(_limit_match.group(1)), 20) if _limit_match else 10

            logger.info(
                "[RETRIEVE] Stage 3a — recency query: repo_hint=%r, limit=%d",
                _repo_hint, _commit_limit,
            )

            if _repo_hint:
                recency_cypher = """
                MATCH (c:Commit)-[:BELONGS_TO]->(r:Repository {full_name: $repo_hint})
                OPTIONAL MATCH (u:User)-[:AUTHORED]->(c)
                RETURN
                  c.sha                          AS name,
                  r.full_name                    AS filepath,
                  coalesce(c.message, c.summary_text, c.diff_text, '') AS code,
                  []                             AS connected
                ORDER BY c.timestamp DESC
                LIMIT $commit_limit
                """
                params = {"repo_hint": _repo_hint, "commit_limit": _commit_limit}
            else:
                recency_cypher = """
                MATCH (c:Commit)-[:BELONGS_TO]->(r:Repository)
                RETURN
                  c.sha       AS name,
                  r.full_name AS filepath,
                  coalesce(c.message, c.summary_text, c.diff_text, '') AS code,
                  []          AS connected
                ORDER BY c.timestamp DESC
                LIMIT $commit_limit
                """
                params = {"commit_limit": _commit_limit}
                if selected_repos:
                    recency_cypher = """
                    MATCH (c:Commit)-[:BELONGS_TO]->(r:Repository)
                    WHERE r.full_name IN $selected_repos
                    RETURN
                      c.sha       AS name,
                      r.full_name AS filepath,
                      coalesce(c.message, c.summary_text, c.diff_text, '') AS code,
                      []          AS connected
                    ORDER BY c.timestamp DESC
                    LIMIT $commit_limit
                    """
                    params["selected_repos"] = selected_repos

            try:
                with driver.session() as session:
                    rows = session.run(recency_cypher, **params).data()
                logger.info(
                    "[RETRIEVE] Stage 3a recency → %d commit(s).", len(rows)
                )
                if rows:
                    context_parts.append(
                        _build_context_block(rows, "recent-commits", global_seen=global_seen_ids)
                    )
                    retrieval_path_tags.append("recent-commits")
            except Exception as exc:
                logger.error("[RETRIEVE] ❌ Stage 3a recency FAILED: %s", exc)

        logger.info(
            "[RETRIEVE] Stage 3b — fulltext: sanitized=%r (original=%r)",
            lucene_query, user_query,
        )
        fulltext_cypher_local = fulltext_cypher
        try:
            with driver.session() as session:
                rows = session.run(
                    fulltext_cypher_local, search_query=lucene_query
                ).data()
            logger.info("[RETRIEVE] Stage 3b fulltext → %d record(s).", len(rows))
            if rows:
                context_parts.append(
                    _build_context_block(rows, "fulltext-commits", global_seen=global_seen_ids)
                )
                retrieval_path_tags.append("fulltext")
        except Exception as exc:
            logger.error("[RETRIEVE] ❌ Stage 3b fulltext FAILED: %s", exc)

    # ── Stage 3c: Commit-SHA → Files/Functions lookup ─────────────────────────
    _SHA_PATTERN = re.compile(r'\b([0-9a-f]{7,40})\b', re.IGNORECASE)
    _COMMIT_FILE_INTENT = re.compile(
        r'\b(what|which)\b.{0,30}\b(file|files|changed|modified|touched|affect)\b'
        r'|\b(file|files)\b.{0,20}\b(commit|sha|hash)\b'
        r'|\bcommit\b.{0,20}\b(change|changed|modify|modified|touch|touched|affect)\b'
        r'|\bshow\b.{0,20}\b(file|files)\b.{0,20}\bcommit\b',
        re.IGNORECASE,
    )
    _sha_matches   = _SHA_PATTERN.findall(user_query)
    _has_sha       = bool(_sha_matches)
    _has_file_intent = bool(_COMMIT_FILE_INTENT.search(user_query))

    if _has_sha or intent.wants_commit_files or (_has_file_intent and "commit" in _query_lower):
        logger.info(
            "[RETRIEVE] Stage 3c — commit-file lookup triggered. "
            "sha_matches=%s  has_file_intent=%s  wants_commit_files=%s",
            _sha_matches, _has_file_intent, intent.wants_commit_files,
        )

        commit_file_rows: list[dict] = []

        def _lookup_commit_files(sha_hint: str) -> list[dict]:
            rows_out: list[dict] = []

            repo_filter_cf = (
                "WHERE r.full_name IN $selected_repos OR c.repo IN $selected_repos" if selected_repos else ""
            )
            file_cypher = f"""
            MATCH (c:Commit)-[:MODIFIED]->(f:File)
            WHERE toLower(c.sha) CONTAINS toLower($sha_hint)
            OPTIONAL MATCH (c)-[:BELONGS_TO]->(r:Repository)
            {repo_filter_cf}
            RETURN
              f.path                                         AS name,
              f.path                                         AS filepath,
              coalesce(r.full_name, c.repo, '')              AS repo,
              'FILE_CHANGED_BY_COMMIT'                       AS rel_type,
              coalesce(f.code, '')                           AS code,
              c.sha                                          AS commit_sha,
              coalesce(c.message, c.summary_text, '')        AS commit_msg
            ORDER BY f.path
            LIMIT 50
            """
            try:
                with driver.session() as session:
                    result = session.run(
                        file_cypher,
                        sha_hint=sha_hint,
                        selected_repos=selected_repos or [],
                    ).data()
                logger.info(
                    "[RETRIEVE] Stage 3c sha=%r → %d file node(s).",
                    sha_hint, len(result),
                )
                rows_out.extend(result)
            except Exception as exc:
                logger.warning("[RETRIEVE] Stage 3c file lookup sha=%r failed: %s", sha_hint, exc)

            fn_cypher = f"""
            MATCH (c:Commit)-[:MODIFIED]->(fn:Function)
            WHERE toLower(c.sha) CONTAINS toLower($sha_hint)
            OPTIONAL MATCH (c)-[:BELONGS_TO]->(r:Repository)
            {repo_filter_cf}
            RETURN
              fn.name                                        AS name,
              coalesce(fn.filepath, fn.path, '')             AS filepath,
              coalesce(r.full_name, c.repo, '')              AS repo,
              'FUNCTION_CHANGED_BY_COMMIT'                   AS rel_type,
              coalesce(fn.code, '')                          AS code,
              c.sha                                          AS commit_sha,
              coalesce(c.message, c.summary_text, '')        AS commit_msg
            ORDER BY fn.filepath, fn.name
            LIMIT 50
            """
            try:
                with driver.session() as session:
                    result = session.run(
                        fn_cypher,
                        sha_hint=sha_hint,
                        selected_repos=selected_repos or [],
                    ).data()
                logger.info(
                    "[RETRIEVE] Stage 3c sha=%r → %d function node(s).",
                    sha_hint, len(result),
                )
                rows_out.extend(result)
            except Exception as exc:
                logger.warning("[RETRIEVE] Stage 3c fn lookup sha=%r failed: %s", sha_hint, exc)

            meta_cypher = """
            MATCH (c:Commit)
            WHERE toLower(c.sha) CONTAINS toLower($sha_hint)
            OPTIONAL MATCH (u:User)-[:AUTHORED]->(c)
            OPTIONAL MATCH (c)-[:BELONGS_TO]->(r:Repository)
            RETURN
              coalesce(c.sha, '')                            AS name,
              coalesce(r.full_name, c.repo, '')              AS filepath,
              coalesce(c.message, c.summary_text, c.diff_text, '') AS code,
              ''                                             AS rel_type,
              coalesce(u.login, '')                          AS author,
              coalesce(r.full_name, '')                      AS repo,
              c.timestamp                                    AS committed_at
            LIMIT 5
            """
            try:
                with driver.session() as session:
                    meta_result = session.run(meta_cypher, sha_hint=sha_hint).data()
                logger.info(
                    "[RETRIEVE] Stage 3c sha=%r → %d commit meta record(s).",
                    sha_hint, len(meta_result),
                )
                rows_out = meta_result + rows_out
            except Exception as exc:
                logger.warning("[RETRIEVE] Stage 3c meta lookup sha=%r failed: %s", sha_hint, exc)

            return rows_out

        if _sha_matches:
            seen_shas: set[str] = set()
            for sha in _sha_matches:
                if sha.lower() not in seen_shas:
                    seen_shas.add(sha.lower())
                    commit_file_rows.extend(_lookup_commit_files(sha))
        else:
            logger.info(
                "[RETRIEVE] Stage 3c — no SHA in query; re-using fulltext commit results."
            )
            try:
                with driver.session() as session:
                    ft_rows = session.run(
                        fulltext_cypher, search_query=lucene_query
                    ).data()
                for ft_row in ft_rows[:3]:
                    sha_candidate = ft_row.get("name", "")
                    if sha_candidate and len(sha_candidate) >= 7:
                        commit_file_rows.extend(_lookup_commit_files(sha_candidate[:12]))
            except Exception as exc:
                logger.warning("[RETRIEVE] Stage 3c fallback fulltext→MODIFIED failed: %s", exc)

        if commit_file_rows:
            block = _build_context_block(commit_file_rows, "commit-file-lookup", allow_no_code=True, global_seen=global_seen_ids)
            if block:
                context_parts.append(block)
                retrieval_path_tags.append("commit-files")
            logger.info(
                "[RETRIEVE] Stage 3c — %d commit-file record(s) added to context.",
                len(commit_file_rows),
            )
        else:
            logger.warning(
                "[RETRIEVE] ⚠️  Stage 3c returned 0 results."
            )
    else:
        logger.info("[RETRIEVE] Stage 3c skipped — no commit SHA or file-intent in query.")

    # ── Stage 3d: File modification frequency / churn lookup ──────────────────
    if _needs_commit_context and not _has_sha:
        logger.info("[RETRIEVE] Stage 3d — file modification frequency / churn check.")
        repo_filter_mod = (
            "WHERE (f.repo IN $selected_repos OR f.full_name IN $selected_repos OR f.name IN $selected_repos OR c.repo IN $selected_repos)"
            if selected_repos else ""
        )
        freq_cypher = f"""
        MATCH (c:Commit)-[:MODIFIED]->(f:File)
        {repo_filter_mod}
        WITH f, count(c) AS mods
        RETURN
          f.path AS name,
          f.path AS filepath,
          coalesce(f.repo, '') AS repo,
          'FILE_MODIFICATION_FREQUENCY' AS rel_type,
          'Modified in ' + toString(mods) + ' commit(s).' AS code
        ORDER BY mods DESC
        LIMIT 15
        """
        try:
            with driver.session() as session:
                freq_rows = session.run(
                    freq_cypher,
                    selected_repos=selected_repos or [],
                ).data()
            logger.info("[RETRIEVE] Stage 3d frequency → %d file(s).", len(freq_rows))
            if freq_rows:
                block = _build_context_block(freq_rows, "commit-file-lookup", allow_no_code=True, global_seen=global_seen_ids)
                if block:
                    context_parts.append(block)
                    if "commit-files" not in retrieval_path_tags:
                        retrieval_path_tags.append("commit-files")
        except Exception as exc:
            logger.warning("[RETRIEVE] Stage 3d frequency failed: %s", exc)

    # ── Stage 4: Blame traversal ───────────────────────────────────────────────
    _is_blame_query = intent.wants_blame

    if _is_blame_query:
        logger.info("[RETRIEVE] Stage 4 — blame traversal triggered.")

        semantic_filepaths = list(dict.fromkeys(
            r["filepath"] for r in vector_rows if r.get("filepath")
        ))

        blame_hints  = _extract_blame_hints(user_query)
        intent_subs  = intent.subjects if (intent and intent.subjects) else []
        fallback_hints = list(dict.fromkeys(blame_hints["func_hints"] + blame_hints["file_hints"] + intent_subs))

        logger.info(
            "[RETRIEVE] Stage 4 semantic_filepaths=%s  fallback_hints=%s",
            semantic_filepaths, fallback_hints,
        )

        blame_rows: list[dict] = []

        for filepath in semantic_filepaths:
            if selected_repos:
                blame_fp_cypher = """
                MATCH (u:User)-[:AUTHORED]->(c:Commit)-[:MODIFIED]->(f:File)
                WHERE f.path = $filepath
                MATCH (c)-[:BELONGS_TO]->(r:Repository)
                WHERE r.full_name IN $selected_repos
                RETURN
                  u.login       AS author,
                  c.sha         AS commit_sha,
                  coalesce(c.message, c.summary_text, '') AS commit_msg,
                  c.timestamp   AS committed_at,
                  ''            AS func_name,
                  f.path        AS filepath
                ORDER BY c.timestamp DESC
                LIMIT 5
                """
            else:
                blame_fp_cypher = """
                MATCH (u:User)-[:AUTHORED]->(c:Commit)-[:MODIFIED]->(f:File)
                WHERE f.path = $filepath
                RETURN
                  u.login       AS author,
                  c.sha         AS commit_sha,
                  coalesce(c.message, c.summary_text, '') AS commit_msg,
                  c.timestamp   AS committed_at,
                  ''            AS func_name,
                  f.path        AS filepath
                ORDER BY c.timestamp DESC
                LIMIT 5
                """
            try:
                with driver.session() as session:
                    rows = session.run(
                        blame_fp_cypher,
                        filepath=filepath,
                        selected_repos=selected_repos or [],
                    ).data()
                logger.info(
                    "[RETRIEVE] Stage 4a semantic filepath=%r → %d record(s).", filepath, len(rows)
                )
                blame_rows.extend(rows)
            except Exception as exc:
                logger.warning("[RETRIEVE] Stage 4a filepath=%r failed: %s", filepath, exc)

            if selected_repos:
                blame_fn_cypher = """
                MATCH (u:User)-[:AUTHORED]->(c:Commit)-[:MODIFIED]->(fn:Function)
                WHERE fn.filepath = $filepath
                MATCH (c)-[:BELONGS_TO]->(r:Repository)
                WHERE r.full_name IN $selected_repos
                RETURN
                  u.login       AS author,
                  c.sha         AS commit_sha,
                  coalesce(c.message, c.summary_text, '') AS commit_msg,
                  c.timestamp   AS committed_at,
                  fn.name       AS func_name,
                  fn.filepath   AS filepath
                ORDER BY c.timestamp DESC
                LIMIT 5
                """
            else:
                blame_fn_cypher = """
                MATCH (u:User)-[:AUTHORED]->(c:Commit)-[:MODIFIED]->(fn:Function)
                WHERE fn.filepath = $filepath
                RETURN
                  u.login       AS author,
                  c.sha         AS commit_sha,
                  coalesce(c.message, c.summary_text, '') AS commit_msg,
                  c.timestamp   AS committed_at,
                  fn.name       AS func_name,
                  fn.filepath   AS filepath
                ORDER BY c.timestamp DESC
                LIMIT 5
                """
            try:
                with driver.session() as session:
                    rows = session.run(
                        blame_fn_cypher,
                        filepath=filepath,
                        selected_repos=selected_repos or [],
                    ).data()
                logger.info(
                    "[RETRIEVE] Stage 4a func-level filepath=%r → %d record(s).", filepath, len(rows)
                )
                blame_rows.extend(rows)
            except Exception as exc:
                logger.warning("[RETRIEVE] Stage 4a func filepath=%r failed: %s", filepath, exc)

        for hint in fallback_hints:
            if selected_repos:
                blame_hint_cypher = """
                MATCH (u:User)-[:AUTHORED]->(c:Commit)-[:BELONGS_TO]->(r:Repository)
                WHERE r.full_name IN $selected_repos
                OPTIONAL MATCH (c)-[:MODIFIED]->(target)
                WITH u, c, target
                WHERE toLower(c.message) CONTAINS toLower($hint)
                   OR toLower(coalesce(c.summary_text, '')) CONTAINS toLower($hint)
                   OR toLower(coalesce(target.path, target.filepath, '')) CONTAINS toLower($hint)
                   OR (target:Function AND toLower(target.name) CONTAINS toLower($hint))
                RETURN
                  u.login       AS author,
                  c.sha         AS commit_sha,
                  coalesce(c.message, c.summary_text, '') AS commit_msg,
                  c.timestamp   AS committed_at,
                  CASE WHEN target:Function THEN target.name ELSE '' END AS func_name,
                  coalesce(target.filepath, target.path, '') AS filepath
                ORDER BY c.timestamp DESC
                LIMIT 5
                """
            else:
                blame_hint_cypher = """
                MATCH (u:User)-[:AUTHORED]->(c:Commit)
                OPTIONAL MATCH (c)-[:MODIFIED]->(target)
                WITH u, c, target
                WHERE toLower(c.message) CONTAINS toLower($hint)
                   OR toLower(coalesce(c.summary_text, '')) CONTAINS toLower($hint)
                   OR toLower(coalesce(target.path, target.filepath, '')) CONTAINS toLower($hint)
                   OR (target:Function AND toLower(target.name) CONTAINS toLower($hint))
                RETURN
                  u.login       AS author,
                  c.sha         AS commit_sha,
                  coalesce(c.message, c.summary_text, '') AS commit_msg,
                  c.timestamp   AS committed_at,
                  CASE WHEN target:Function THEN target.name ELSE '' END AS func_name,
                  coalesce(target.filepath, target.path, '') AS filepath
                ORDER BY c.timestamp DESC
                LIMIT 5
                """
            try:
                with driver.session() as session:
                    rows = session.run(
                        blame_hint_cypher,
                        hint=hint,
                        selected_repos=selected_repos or [],
                    ).data()
                logger.info(
                    "[RETRIEVE] Stage 4b fallback hint=%r → %d record(s).", hint, len(rows)
                )
                blame_rows.extend(rows)
            except Exception as exc:
                logger.warning("[RETRIEVE] Stage 4b hint=%r failed: %s", hint, exc)

        if not semantic_filepaths and not fallback_hints:
            logger.info("[RETRIEVE] Stage 4c — no hints; fetching top authors overall.")
            repo_clause = "WHERE r.full_name IN $selected_repos" if selected_repos else ""
            top_authors_cypher = f"""
            MATCH (u:User)-[:AUTHORED]->(c:Commit)-[:BELONGS_TO]->(r:Repository)
            {repo_clause}
            RETURN
              u.login   AS author,
              count(c)  AS commit_count
            ORDER BY commit_count DESC
            LIMIT 10
            """
            try:
                with driver.session() as session:
                    rows = session.run(
                        top_authors_cypher,
                        selected_repos=selected_repos or [],
                    ).data()
                logger.info("[RETRIEVE] Stage 4c top-authors → %d record(s).", len(rows))
                blame_rows.extend([
                    {
                        "author":       r.get("author"),
                        "commit_sha":   f"{r.get('commit_count')} commits total",
                        "commit_msg":   "",
                        "committed_at": "",
                        "func_name":    "",
                        "filepath":     "(all files)",
                    }
                    for r in rows
                ])
            except Exception as exc:
                logger.warning("[RETRIEVE] Stage 4c failed: %s", exc)

        if blame_rows:
            context_parts.append(_build_blame_context_block(blame_rows, global_seen=global_seen_ids))
            retrieval_path_tags.append("blame")
            logger.info("[RETRIEVE] Stage 4 — %d blame record(s) added to context.", len(blame_rows))
        else:
            logger.warning(
                "[RETRIEVE] ⚠️  Stage 4 blame traversal returned 0 results."
            )
    else:
        logger.info("[RETRIEVE] Stage 4 skipped — no blame keywords in query.")

    # ── Final assembly ────────────────────────────────────────────────────────
    if not context_parts:
        logger.error(
            "[RETRIEVE] ❌ ALL stages returned 0 results. "
            "Check: (1) Function nodes have .code set, "
            "(2) 'code_embeddings' vector index exists, "
            "(3) 'commit_summaries' fulltext index exists."
        )
        return "No relevant context found in the knowledge graph.", intent

    full_context = "\n\n".join(context_parts)
    retrieval_path = "+".join(retrieval_path_tags) or "none"
    logger.info(
        "[RETRIEVE] ✅ path=%-20s | context_chars=%d",
        retrieval_path, len(full_context),
    )
    logger.debug("[RETRIEVE] Context preview (first 3000 chars):\n%s", full_context[:3000])
    return full_context, intent
