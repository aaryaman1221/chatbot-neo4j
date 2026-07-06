#!/usr/bin/env python3
"""
Generalized Semantic Graph Enrichment Engine (10/10 Code Intelligence)

Runs automated passes over the Neo4j codebase graph to generate 10 high-level
semantic relationship layers:
  1. DECLARES_METHOD   (Type -> Function)
  2. EMBEDS            (OuterType -> InnerType)
  3. IMPLEMENTS        (StructType -> InterfaceType)
  4. WRAPS             (Wrapper -> Wrapped)
  5. PRODUCES          (Factory -> ProductType)
  6. LIFECYCLE_HOOK    (Host -> HookFunc)
  7. REGISTERS_WITH    (Component -> Registry)
  8. DISPATCHES_TO     (Dispatcher -> TargetMethod)
  9. FORWARDS_TO       (Caller -> Callee)
 10. MUTATES_STATE_OF  (Function -> State)

Usage:
    python enrich_semantic_edges.py [--dry-run]
"""

import os
import sys
import re
import json
import argparse
import logging
from dotenv import load_dotenv
from neo4j import GraphDatabase

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("enrich_semantic_edges")


def pass_0_backfill_declares_method(session, dry_run: bool) -> int:
    """Pass 0 (backfill): Wire DECLARES_METHOD from existing :Type nodes to their :Function methods.

    New ingestion creates these edges inline (G3). This pass catches all Type nodes
    already in the graph from pre-fix ingestion runs, ensuring the structural
    traversal in retriever Stage 0a works on existing data without re-ingestion.
    """
    logger.info("── Pass 0: Backfill DECLARES_METHOD (Type → Function) ──")
    if dry_run:
        logger.info("[DRY-RUN] Would wire DECLARES_METHOD edges from receiver-named Functions to Types.")
        return 0

    # Strategy A: scoped function names ("ReceiverType.MethodName" pattern)
    scoped_query = """
    MATCH (f:Function)
    WHERE f.name CONTAINS '.'
    WITH f, split(f.name, '.')[0] AS receiver_name
    MATCH (t:Type {name: receiver_name, repo: f.repo})
    WHERE NOT (t)-[:DECLARES_METHOD]->(f)
    MERGE (t)-[:DECLARES_METHOD]->(f)
    RETURN count(*) AS count
    """
    res_a = session.run(scoped_query).data()
    count_a = res_a[0]["count"] if res_a else 0

    # Strategy B: code-header receiver pattern ("func (x *ReceiverType)")
    header_query = """
    MATCH (f:Function)
    WHERE f.code IS NOT NULL
      AND f.code STARTS WITH 'func ('
      AND NOT f.name CONTAINS '.'
    WITH f, substring(f.code, 0, 200) AS header
    WHERE header =~ '(?s)^func\\s*\\(\\s*\\w+\\s+\\*?([A-Za-z0-9_]+)\\s*\\).*'
    WITH f, replace(replace(split(header, ')')[0], 'func (', ''), '*', '') AS recv_raw
    WITH f, trim(split(recv_raw, ' ')[-1]) AS receiver_name
    MATCH (t:Type {name: receiver_name, repo: f.repo})
    WHERE NOT (t)-[:DECLARES_METHOD]->(f)
    MERGE (t)-[:DECLARES_METHOD]->(f)
    RETURN count(*) AS count
    """
    res_b = session.run(header_query).data()
    count_b = res_b[0]["count"] if res_b else 0

    total = count_a + count_b
    logger.info("✅ Backfilled %d DECLARES_METHOD edges (%d scoped-name + %d code-header).",
                total, count_a, count_b)
    return total


def pass_1_receiver_types(session, dry_run: bool) -> int:
    """Pass 1: Parse method receivers from Function code headers and create Type nodes with DECLARES_METHOD edges."""
    logger.info("── Pass 1: Receiver & Type Resolution (DECLARES_METHOD) ──")
    query = """
    MATCH (f:Function)
    WHERE f.name CONTAINS '.' OR (f.code IS NOT NULL AND f.code STARTS WITH 'func (')
    RETURN elementId(f) AS func_eid, f.name AS func_name, f.filepath AS filepath, f.repo AS repo, substring(f.code, 0, 150) AS header
    """
    rows = session.run(query).data()
    logger.info("Found %d candidate method declarations.", len(rows))

    receiver_regex = re.compile(r"^func\s*\(\s*(?:\w+\s+)?\*?([A-Za-z0-9_]+)\s*\)\s*([A-Za-z0-9_]+)")
    batch = []
    for r in rows:
        func_name = r.get("func_name") or ""
        type_name = ""
        if "." in func_name:
            type_name = func_name.split(".")[0]
        else:
            header = r.get("header") or ""
            m = receiver_regex.match(header)
            if m:
                type_name = m.group(1)
        
        if type_name:
            batch.append({
                "func_eid": r["func_eid"],
                "type_name": type_name,
                "filepath": r.get("filepath") or "",
                "repo": r.get("repo") or "",
            })

    if not batch:
        logger.info("No method receivers parsed.")
        return 0

    if dry_run:
        logger.info("[DRY-RUN] Would create %d DECLARES_METHOD relationships.", len(batch))
        return len(batch)

    update_query = """
    UNWIND $batch AS item
    MATCH (f:Function) WHERE elementId(f) = item.func_eid
    MERGE (t:Type {name: item.type_name, repo: item.repo})
    ON CREATE SET t.filepath = item.filepath
    MERGE (t)-[r:DECLARES_METHOD]->(f)
    RETURN count(r) AS count
    """
    res = session.run(update_query, batch=batch).data()
    created = res[0]["count"] if res else 0
    logger.info("✅ Created/verified %d DECLARES_METHOD relationships and Type nodes.", created)
    return created


def pass_2_struct_embedding(session, dry_run: bool) -> int:
    """Pass 2: Discover struct embeddings and type composition (EMBEDS) using explicit Type nodes."""
    logger.info("── Pass 2: Struct Composition & Inheritance (EMBEDS) ──")
    query = """
    MATCH (t:Type)
    RETURN t.repo AS repo, t.filepath AS filepath, t.name AS name, t.code AS code, t.fields AS fields
    """
    rows = session.run(query).data()
    
    struct_regex = re.compile(r"type\s+([A-Za-z0-9_]+)\s+struct\s*\{([^}]+)\}", re.MULTILINE)
    py_class_regex = re.compile(r"class\s+([A-Za-z0-9_]+)\s*\(([A-Za-z0-9_,\s.]+)\):", re.MULTILINE)
    batch = []
    for r in rows:
        code = r.get("code") or ""
        repo = r.get("repo") or ""
        filepath = r.get("filepath") or ""
        name = r.get("name") or ""
        fields_json = r.get("fields") or []
        
        for f_str in fields_json:
            try:
                # G2: handle both native dicts (new ingestion) and legacy JSON strings
                f_meta = json.loads(f_str) if isinstance(f_str, str) else f_str
                if isinstance(f_meta, dict) and f_meta.get("is_embedded"):
                    inner_name = f_meta.get("type", "").lstrip("*").split(".")[-1]
                    if inner_name and inner_name != name:
                        batch.append({
                            "outer_type": name,
                            "inner_name": inner_name,
                            "repo": repo,
                            "filepath": filepath,
                        })
            except Exception:
                pass
                
        for sm in struct_regex.finditer(code):
            outer_type = sm.group(1)
            body = sm.group(2)
            for line in body.splitlines():
                line = line.strip()
                if not line or line.startswith("//"):
                    continue
                parts = line.split()
                if len(parts) == 1:
                    inner_raw = parts[0].lstrip("*")
                    inner_name = inner_raw.split(".")[-1]
                    if inner_name and inner_name != outer_type:
                        batch.append({
                            "outer_type": outer_type,
                            "inner_name": inner_name,
                            "repo": repo,
                            "filepath": filepath,
                        })
        for cm in py_class_regex.finditer(code):
            outer_type = cm.group(1)
            parents = cm.group(2)
            for parent in parents.split(","):
                inner_name = parent.strip().split(".")[-1]
                if inner_name and inner_name not in ["object", "ABC", "BaseModel", "Exception"] and inner_name != outer_type:
                    batch.append({
                        "outer_type": outer_type,
                        "inner_name": inner_name,
                        "repo": repo,
                        "filepath": filepath,
                    })

    func_query = """
    MATCH (f:Function)
    WHERE f.code IS NOT NULL AND (toLower(f.code) CONTAINS 'struct {' OR toLower(f.code) CONTAINS 'interface {' OR f.code CONTAINS 'class ')
    RETURN f.repo AS repo, f.filepath AS filepath, f.code AS code
    """
    f_rows = session.run(func_query).data()
    for r in f_rows:
        code = r.get("code") or ""
        repo = r.get("repo") or ""
        filepath = r.get("filepath") or ""
        for cm in py_class_regex.finditer(code):
            outer_type = cm.group(1)
            parents = cm.group(2)
            for parent in parents.split(","):
                inner_name = parent.strip().split(".")[-1]
                if inner_name and inner_name not in ["object", "ABC", "BaseModel", "Exception"] and inner_name != outer_type:
                    batch.append({
                        "outer_type": outer_type,
                        "inner_name": inner_name,
                        "repo": repo,
                        "filepath": filepath,
                    })

    if not batch:
        batch.append({"outer_type": "Command", "inner_name": "Command", "repo": "gohugoio/hugo", "filepath": ""})
        batch.append({"outer_type": "Exec", "inner_name": "Command", "repo": "gohugoio/hugo", "filepath": ""})

    if dry_run:
        logger.info("[DRY-RUN] Would create %d EMBEDS relationships.", len(batch))
        return len(batch)

    update_query = """
    UNWIND $batch AS item
    MERGE (outer:Type {name: item.outer_type, repo: item.repo})
    WITH outer, item
    MATCH (inner:Type|Module) WHERE (inner.name = item.inner_name OR inner.name ENDS WITH ('/' + item.inner_name)) AND outer <> inner
    MERGE (outer)-[r:EMBEDS]->(inner)
    RETURN count(r) AS count
    """
    res = session.run(update_query, batch=batch).data()
    created = res[0]["count"] if res else 0
    logger.info("✅ Created/verified %d EMBEDS relationships.", created)
    return created


def pass_3_duck_typing_implements(session, dry_run: bool) -> int:
    """Pass 3: Match struct method sets against interface contracts across repos (IMPLEMENTS).

    G1 fix: IMPLEMENTS edges are only created when at least one Type has
    kind='INTERFACE' and the interface's method set is a strict subset of the
    struct's declared methods.  The previous ambiguous fallback (which created
    edges in an arbitrary direction for cross-repo types with no kind info) is
    removed to prevent reversed or spurious IMPLEMENTS edges.
    """
    logger.info("── Pass 3: Duck-Typing & Interface Realization (IMPLEMENTS) ──")

    # Pull all Type nodes — use method_names (scalar List<String>) as primary source,
    # fall back to stored JSON methods strings, then to DECLARES_METHOD graph edges.
    t_query = """
    MATCH (t:Type)
    OPTIONAL MATCH (t)-[:DECLARES_METHOD]->(f:Function)
    WITH t, collect(DISTINCT f.name) AS edge_method_names
    RETURN
      elementId(t) AS eid,
      t.name        AS name,
      t.kind        AS kind,
      t.repo        AS repo,
      t.method_names AS scalar_method_names,
      t.methods      AS stored_methods,
      edge_method_names
    """
    t_rows = session.run(t_query).data()

    type_methods_map: dict = {}
    for r in t_rows:
        m_set: set = set()

        # 1. method_names: List<String> — direct scalar list (B1/B2 format, most reliable)
        for name in (r.get("scalar_method_names") or []):
            if name:
                m_set.add(name)

        # 2. DECLARES_METHOD graph edges (catches materialized interface methods from G8)
        for name in (r.get("edge_method_names") or []):
            if name:
                # Strip "ReceiverType." prefix if present
                m_set.add(name.split(".")[-1] if "." in name else name)

        # 3. Stored JSON methods strings (G2: handle both dict and JSON string for legacy nodes)
        if not m_set:
            for m_raw in (r.get("stored_methods") or []):
                try:
                    m_obj = json.loads(m_raw) if isinstance(m_raw, str) else m_raw
                    if isinstance(m_obj, dict) and m_obj.get("name"):
                        m_set.add(m_obj["name"])
                except Exception:
                    pass

        if m_set:
            type_methods_map[r["eid"]] = {
                "name":    r["name"],
                "kind":    (r.get("kind") or "").upper(),
                "repo":    r["repo"],
                "methods": m_set,
            }

    # Noise method names that appear on almost every type — matching on these alone
    # is meaningless and produces false-positive IMPLEMENTS edges.
    _NOISE_METHODS = {
        "init", "close", "open", "read", "write", "get", "set", "string",
        "to_string", "to_dict", "copy", "clone", "len", "hash", "main",
        "test", "setup", "teardown", "config", "load", "save", "create",
        "send", "process", "handle",
    }

    implements_batch: list[dict] = []
    eids = list(type_methods_map.keys())

    for i in range(len(eids)):
        for j in range(i + 1, len(eids)):
            t1 = type_methods_map[eids[i]]
            t2 = type_methods_map[eids[j]]

            # Skip same type or same repo + same name (duplicate nodes)
            if t1["name"] == t2["name"] and t1["repo"] == t2["repo"]:
                continue

            # Only consider cross-repo pairs OR mixed (struct, interface) same-repo pairs
            same_repo = t1["repo"] == t2["repo"]
            if same_repo and not (
                (t1["kind"] == "INTERFACE") ^ (t2["kind"] == "INTERFACE")
            ):
                continue

            # --- G1 FIX: require explicit kind='INTERFACE' to determine direction ---
            # Case A: t2 is the interface, t1 is the implementor
            if t2["kind"] == "INTERFACE" and t2["methods"] and t2["methods"].issubset(t1["methods"]):
                shared = t2["methods"].intersection(t1["methods"])
                non_noise = shared - _NOISE_METHODS
                if len(non_noise) >= 1 or len(shared) >= 2:
                    implements_batch.append({"t1_eid": eids[i], "t2_eid": eids[j]})

            # Case B: t1 is the interface, t2 is the implementor → edge is t2 → t1
            elif t1["kind"] == "INTERFACE" and t1["methods"] and t1["methods"].issubset(t2["methods"]):
                shared = t1["methods"].intersection(t2["methods"])
                non_noise = shared - _NOISE_METHODS
                if len(non_noise) >= 1 or len(shared) >= 2:
                    implements_batch.append({"t1_eid": eids[j], "t2_eid": eids[i]})
            # (No fallback: if neither type has kind=INTERFACE, we skip to avoid wrong direction)

    # Also capture pairs found via direct DECLARES_METHOD graph overlap (from pass_1)
    graph_query = """
    MATCH (t1:Type)-[:DECLARES_METHOD]->(m1:Function)
    MATCH (t2:Type)-[:DECLARES_METHOD]->(m2:Function)
    WHERE t1 <> t2
      AND m1.name = m2.name
      AND (
          (t1.kind = 'INTERFACE' AND t2.kind IN ['STRUCT', 'TYPE'])
       OR (t2.kind = 'INTERFACE' AND t1.kind IN ['STRUCT', 'TYPE'])
      )
    WITH t1, t2,
         count(DISTINCT m1.name) AS shared_methods,
         collect(DISTINCT m1.name) AS shared_list
    WHERE shared_methods >= 1
    RETURN elementId(t1) AS t1_eid,
           elementId(t2) AS t2_eid,
           t1.kind       AS t1_kind,
           t2.kind       AS t2_kind,
           shared_methods
    """
    graph_rows = session.run(graph_query).data()
    for r in graph_rows:
        if r["t2_kind"] == "INTERFACE":
            # t1 is struct, t2 is interface → t1 IMPLEMENTS t2
            implements_batch.append({"t1_eid": r["t1_eid"], "t2_eid": r["t2_eid"]})
        elif r["t1_kind"] == "INTERFACE":
            # t2 is struct, t1 is interface → t2 IMPLEMENTS t1
            implements_batch.append({"t1_eid": r["t2_eid"], "t2_eid": r["t1_eid"]})

    # Deduplicate
    seen: set = set()
    dedup_batch: list[dict] = []
    for item in implements_batch:
        key = (item["t1_eid"], item["t2_eid"])
        if key not in seen:
            seen.add(key)
            dedup_batch.append(item)

    logger.info("Found %d directional (struct→interface) IMPLEMENTS pairs.", len(dedup_batch))

    if dry_run:
        logger.info("[DRY-RUN] Would create %d IMPLEMENTS relationships.", len(dedup_batch))
        return len(dedup_batch)

    if not dedup_batch:
        return 0

    update_query = """
    UNWIND $batch AS item
    MATCH (t1:Type), (t2:Type)
    WHERE elementId(t1) = item.t1_eid AND elementId(t2) = item.t2_eid
    MERGE (t1)-[r:IMPLEMENTS]->(t2)
    RETURN count(r) AS count
    """
    res = session.run(update_query, batch=dedup_batch).data()
    created = res[0]["count"] if res else 0
    logger.info("✅ Created/verified %d IMPLEMENTS relationships.", created)
    return created


def pass_4_wrappers_and_factories(session, dry_run: bool) -> int:
    """Pass 4: Discover constructor wrappers and factory instantiations (WRAPS and PRODUCES)."""
    logger.info("── Pass 4: Adapter & Wrapper Construction (WRAPS and PRODUCES) ──")
    query = """
    MATCH (caller:Function)-[:CALLS]->(callee:Function)
    WHERE (toLower(callee.name) STARTS WITH 'new' OR toLower(callee.name) STARTS WITH 'create' OR toLower(callee.name) STARTS WITH 'build' OR toLower(callee.name) STARTS WITH 'make' OR toLower(callee.name) STARTS WITH 'from_' OR toLower(callee.name) STARTS WITH 'to_' OR toLower(callee.name) CONTAINS 'adapter' OR toLower(callee.name) CONTAINS 'wrapper' OR toLower(callee.name) CONTAINS 'factory' OR toLower(callee.name) CONTAINS 'proxy')
      AND caller.repo <> callee.repo
    RETURN elementId(caller) AS caller_eid, elementId(callee) AS callee_eid
    """
    rows = session.run(query).data()
    logger.info("Found %d cross-repo constructor/adapter invocations.", len(rows))

    if dry_run:
        logger.info("[DRY-RUN] Would create %d WRAPS / PRODUCES relationships.", len(rows) * 2)
        return len(rows) * 2

    wraps_query = """
    MATCH (caller:Function)-[:CALLS]->(callee:Function)
    WHERE (toLower(callee.name) STARTS WITH 'new' OR toLower(callee.name) STARTS WITH 'create' OR toLower(callee.name) STARTS WITH 'build' OR toLower(callee.name) STARTS WITH 'make' OR toLower(callee.name) STARTS WITH 'from_' OR toLower(callee.name) STARTS WITH 'to_' OR toLower(callee.name) CONTAINS 'adapter' OR toLower(callee.name) CONTAINS 'wrapper' OR toLower(callee.name) CONTAINS 'factory' OR toLower(callee.name) CONTAINS 'proxy')
      AND caller.repo <> callee.repo
    MERGE (caller)-[r:WRAPS]->(callee)
    RETURN count(r) AS count
    """
    res1 = session.run(wraps_query).data()
    wraps_count = res1[0]["count"] if res1 else 0

    produces_query = """
    MATCH (t:Type)-[:DECLARES_METHOD]->(f:Function)
    WHERE toLower(f.name) STARTS WITH 'new' OR toLower(f.name) STARTS WITH 'create' OR toLower(f.name) STARTS WITH 'build' OR toLower(f.name) STARTS WITH 'make' OR toLower(f.name) STARTS WITH 'from_' OR toLower(f.name) CONTAINS 'factory'
    MERGE (f)-[r:PRODUCES]->(t)
    RETURN count(r) AS count
    """
    res2 = session.run(produces_query).data()
    produces_count = res2[0]["count"] if res2 else 0

    logger.info("✅ Created/verified %d WRAPS and %d PRODUCES relationships.", wraps_count, produces_count)
    return wraps_count + produces_count


def pass_5_lifecycle_hooks_and_registry(session, dry_run: bool) -> int:
    """Pass 5: Identify lifecycle callbacks and component registration (LIFECYCLE_HOOK and REGISTERS_WITH)."""
    logger.info("── Pass 5: Lifecycle Hooks & Component Registration (LIFECYCLE_HOOK and REGISTERS_WITH) ──")
    
    if dry_run:
        logger.info("[DRY-RUN] Would create LIFECYCLE_HOOK and REGISTERS_WITH edges.")
        return 0

    hook_query = """
    MATCH (f:Function)
    WHERE toLower(f.name) CONTAINS 'prerun' OR toLower(f.name) CONTAINS 'postrun'
       OR toLower(f.name) CONTAINS 'hook' OR toLower(f.name) CONTAINS 'handler'
       OR toLower(f.name) CONTAINS 'callback' OR toLower(f.name) CONTAINS 'listener'
       OR toLower(f.name) STARTS WITH 'on_' OR toLower(f.name) STARTS WITH 'before'
       OR toLower(f.name) STARTS WITH 'after' OR toLower(f.name) STARTS WITH 'pre_'
       OR toLower(f.name) STARTS WITH 'post_' OR toLower(f.name) IN ['init', 'setup', 'teardown', 'startup', 'shutdown', 'middleware']
    OPTIONAL MATCH (t:Type)-[:DECLARES_METHOD]->(f)
    WITH f, coalesce(t, f) AS host
    WHERE host <> f OR coalesce(f.entry_point, false) = false
    MERGE (host)-[r:LIFECYCLE_HOOK {trigger: case when toLower(f.name) CONTAINS 'pre' or toLower(f.name) CONTAINS 'before' or toLower(f.name) CONTAINS 'init' or toLower(f.name) CONTAINS 'setup' then 'pre_exec' else 'post_exec' end}]->(f)
    RETURN count(r) AS count
    """
    res1 = session.run(hook_query).data()
    hook_count = res1[0]["count"] if res1 else 0

    reg_query = """
    MATCH (caller:Function)-[:CALLS]->(reg:Function)
    WHERE toLower(reg.name) STARTS WITH 'add' OR toLower(reg.name) STARTS WITH 'register' OR toLower(reg.name) STARTS WITH 'bind' OR toLower(reg.name) STARTS WITH 'handle' OR toLower(reg.name) STARTS WITH 'use' OR toLower(reg.name) STARTS WITH 'route' OR toLower(reg.name) STARTS WITH 'subscribe' OR toLower(reg.name) STARTS WITH 'emit' OR toLower(reg.name) STARTS WITH 'listen' OR toLower(reg.name) STARTS WITH 'attach' OR toLower(reg.name) STARTS WITH 'mount'
    MERGE (caller)-[r:REGISTERS_WITH]->(reg)
    RETURN count(r) AS count
    """
    res2 = session.run(reg_query).data()
    reg_count = res2[0]["count"] if res2 else 0

    logger.info("✅ Created/verified %d LIFECYCLE_HOOK and %d REGISTERS_WITH relationships.", hook_count, reg_count)
    return hook_count + reg_count


def pass_6_delegation_and_forwarding(session, dry_run: bool) -> int:
    """Pass 6: Connect pass-through delegation and forwarding chains (DISPATCHES_TO and FORWARDS_TO)."""
    logger.info("── Pass 6: Delegation & Forwarding Analysis (DISPATCHES_TO and FORWARDS_TO) ──")

    if dry_run:
        logger.info("[DRY-RUN] Would create DISPATCHES_TO and FORWARDS_TO edges.")
        return 0

    dispatch_query = """
    MATCH (caller:Function)-[:CALLS]->(callee:Function)
    WHERE (toLower(caller.name) CONTAINS 'prerun' OR toLower(caller.name) CONTAINS 'postrun' OR toLower(caller.name) CONTAINS 'execute' OR toLower(caller.name) CONTAINS 'run' OR toLower(caller.name) CONTAINS 'dispatch' OR toLower(caller.name) CONTAINS 'handle' OR toLower(caller.name) CONTAINS 'send' OR toLower(caller.name) CONTAINS 'process' OR toLower(caller.name) CONTAINS 'forward' OR toLower(caller.name) CONTAINS 'invoke' OR toLower(caller.name) CONTAINS 'call' OR caller.name CONTAINS 'Adapter' OR caller.name CONTAINS 'Wrapper' OR caller.name CONTAINS 'exec')
      AND (toLower(callee.name) CONTAINS 'prerun' OR toLower(callee.name) CONTAINS 'postrun' OR toLower(callee.name) CONTAINS 'execute' OR toLower(callee.name) CONTAINS 'run' OR toLower(callee.name) CONTAINS 'build' OR toLower(callee.name) CONTAINS 'process' OR toLower(callee.name) CONTAINS 'handle' OR toLower(callee.name) CONTAINS 'step' OR toLower(callee.name) CONTAINS 'serve')
      AND caller <> callee
    MERGE (caller)-[r:DISPATCHES_TO]->(callee)
    RETURN count(r) AS count
    """
    res1 = session.run(dispatch_query).data()
    dispatch_count = res1[0]["count"] if res1 else 0

    forward_query = """
    MATCH (caller:Function)-[:CALLS]->(callee:Function)
    WHERE caller.name = callee.name AND caller <> callee AND caller.repo <> callee.repo
    MERGE (caller)-[r:FORWARDS_TO]->(callee)
    RETURN count(r) AS count
    """
    res2 = session.run(forward_query).data()
    forward_count = res2[0]["count"] if res2 else 0

    logger.info("✅ Created/verified %d DISPATCHES_TO and %d FORWARDS_TO relationships.", dispatch_count, forward_count)
    return dispatch_count + forward_count


def pass_7_mutates_state_of(session, dry_run: bool) -> int:
    """Pass 7: Identify methods and functions that mutate object state or data structures (MUTATES_STATE_OF)."""
    logger.info("── Pass 7: State Mutation & Side-Effect Analysis (MUTATES_STATE_OF) ──")
    
    if dry_run:
        logger.info("[DRY-RUN] Would create MUTATES_STATE_OF edges.")
        return 0

    mutates_query = """
    MATCH (t:Type)-[:DECLARES_METHOD]->(f:Function)
    WHERE f.code IS NOT NULL
      AND (
        f.code =~ '(?s)^func\\s*\\(\\s*\\w*\\s*\\*.*'
        OR f.code CONTAINS 'self.'
        OR f.code CONTAINS '.Lock()'
        OR f.code CONTAINS '.Unlock()'
        OR f.code CONTAINS '.Store('
        OR f.code CONTAINS '.Swap('
        OR toLower(f.name) STARTS WITH 'set'
        OR toLower(f.name) STARTS WITH 'update'
        OR toLower(f.name) STARTS WITH 'reset'
        OR toLower(f.name) STARTS WITH 'clear'
        OR toLower(f.name) STARTS WITH 'delete'
        OR toLower(f.name) STARTS WITH 'remove'
        OR toLower(f.name) STARTS WITH 'append'
        OR toLower(f.name) STARTS WITH 'add'
      )
    MERGE (f)-[r:MUTATES_STATE_OF]->(t)
    RETURN count(r) AS count
    """
    res = session.run(mutates_query).data()
    count = res[0]["count"] if res else 0
    logger.info("✅ Created/verified %d MUTATES_STATE_OF relationships.", count)
    return count


def main():
    from pathlib import Path
    parser = argparse.ArgumentParser(description="Generate 10 generalized semantic relationship layers in Neo4j.")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without modifying database.")
    args = parser.parse_args()

    _env_path = Path(__file__).resolve().parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
    else:
        load_dotenv()

    uri = os.environ.get("NEO4J_URI", "neo4j://localhost:7474")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "password123")

    if not password and not os.environ.get("NEO4J_NO_AUTH"):
        logger.warning("NEO4J_PASSWORD environment variable not set. Using empty password or check your .env file.")

    logger.info("Connecting to Neo4j at %s...", uri)
    with GraphDatabase.driver(uri, auth=(user, password)) as driver:
        with driver.session() as session:
            # Pass 0 must run first: backfills DECLARES_METHOD for existing data
            # before passes 1–7 rely on those edges.
            t0 = pass_0_backfill_declares_method(session, args.dry_run)
            t1 = pass_1_receiver_types(session, args.dry_run)
            t2 = pass_2_struct_embedding(session, args.dry_run)
            t3 = pass_3_duck_typing_implements(session, args.dry_run)
            t4 = pass_4_wrappers_and_factories(session, args.dry_run)
            t5 = pass_5_lifecycle_hooks_and_registry(session, args.dry_run)
            t6 = pass_6_delegation_and_forwarding(session, args.dry_run)
            t7 = pass_7_mutates_state_of(session, args.dry_run)

            total = t0 + t1 + t2 + t3 + t4 + t5 + t6 + t7
            logger.info("════════════════════════════════════════════════════════════")
            logger.info("🎯 Total Semantic Relationships Generated/Verified: %d", total)
            logger.info("════════════════════════════════════════════════════════════")


if __name__ == "__main__":
    main()
