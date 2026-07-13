#!/usr/bin/env python3
# =============================================================================
# ingest/pipeline.py — Pipeline Orchestration and Graph Database Bootstrap
# =============================================================================

import base64
import json
import time
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Optional

import requests
from tqdm import tqdm
from github import Auth, Github, GithubException
from neo4j import GraphDatabase

from .config import (
    logger,
    GITHUB_API_BASE,
    SOURCE_EXTENSIONS,
    SOURCE_FILENAMES,
    ENTRY_POINT_NAMES,
    UTILITY_DIRS,
    _GRAPH_FLUSH_BATCH,
    _EMBED_FUNC_BATCH_SIZE,
    _GO_TEST_SUFFIXES,
)
from .queries import (
    CYPHER_CREATE_REPO,
    CYPHER_INGEST_FUNCTION,
    CYPHER_INGEST_CALLS,
    CYPHER_MODIFIED_FUNCTION,
    CYPHER_INGEST_COMMIT,
    CYPHER_INGEST_FILE,
    CYPHER_INGEST_DEPENDENCY,
    CYPHER_REMOVE_DEPENDENCY,
    CYPHER_TREE_FILE,
    CYPHER_TREE_DIR,
    CYPHER_LINK_REPO_DEPENDENCY,
    CYPHER_LINK_MODULE_TO_REPO,
    CYPHER_LINK_XREPO_CALLS,
    CYPHER_LINK_XREPO_CALLS_TRANSITIVE,
    CYPHER_VECTOR_INDEX,
    CYPHER_CODE_VECTOR_INDEX,
    CYPHER_MODULE_VECTOR_INDEX,
    CYPHER_FULLTEXT_INDEX,
    CYPHER_UPSERT_STATUS,
    CYPHER_CONSTRAINT_FILE_REPO,
    CYPHER_CONSTRAINT_FUNC_ID,
    CYPHER_INDEX_FUNC_NAME_REPO,
    CYPHER_CLEANUP_UNSCOPED_FILES,
    CYPHER_CLEANUP_UNSCOPED_DIRS,
    CYPHER_CLEANUP_UNSCOPED_FUNCS,
    CYPHER_MARK_FILE_SCANNED,
    CYPHER_INGEST_TYPE,
    CYPHER_INGEST_TYPE_EMBEDDING,
    CYPHER_INGEST_VARIABLE,
    CYPHER_INGEST_DIRECTIVE,
    CYPHER_TYPE_VECTOR_INDEX,
    CYPHER_VAR_VECTOR_INDEX,
    CYPHER_CONSTRAINT_TYPE_ID,
    CYPHER_CONSTRAINT_VAR_ID,
    CYPHER_INDEX_TYPE_NAME_REPO,
    CYPHER_INDEX_TYPE_KIND,
    CYPHER_CONSTRAINT_DIRECTIVE_UNIQUE,
    CYPHER_INGEST_IFACE_METHOD,
    CYPHER_LINK_DECLARES_METHOD,
    CYPHER_RELINK_UNRESOLVED_CALLS,
    CYPHER_PATCH_EMPTY_CODE_STUBS,
    CYPHER_FIND_EMPTY_CODE_FILES,
    CYPHER_UNMARK_FILES_FOR_RESCAN,
    CYPHER_RELINK_CALLS_BY_CODE_GREP,
)
from .parser import (
    _is_noise_file,
    parse_python_ast,
    parse_go_ast,
    extract_imports_from_source,
    extract_temporal_dependencies,
    get_modified_functions,
)
from .ai_service import (
    build_compact_diff,
    render_diff_text,
    _heuristic_summary,
    summarize_with_llm,
    get_embeddings_batch,
)
from .github_client import (
    _github_headers,
    _fetch_json,
    _fetch_all_commits_async,
)


def _write_commit_to_graph(
    driver,
    *,
    repo_full_name: str,
    commit_sha: str,
    modified_files: list,
    dependencies: dict,
    actor_login: str = "unknown",
    committed_at: str = "",
    summary_text: str = "",
    diff_text: str = "",
    commit_message: str = "",
    commit_url: str = "",
):
    with driver.session() as session:
        session.run(
            CYPHER_INGEST_COMMIT,
            repo_full_name=repo_full_name,
            actor_login=actor_login or "unknown",
            commit_sha=commit_sha,
            committed_at=committed_at,
            summary_text=summary_text,
            diff_text=diff_text,
            commit_message=commit_message,
            commit_url=commit_url,
        )
        for filepath in modified_files:
            session.run(CYPHER_INGEST_FILE, repo_full_name=repo_full_name, commit_sha=commit_sha, filepath=filepath)
            
        for source, _, target in dependencies.get("added", []):
            session.run(CYPHER_INGEST_DEPENDENCY, filepath=source, target_module=target, repo_full_name=repo_full_name, commit_sha=commit_sha, embedding=None)
            
        for source, _, target in dependencies.get("removed", []):
            session.run(CYPHER_REMOVE_DEPENDENCY, filepath=source, target_module=target, repo_full_name=repo_full_name, commit_sha=commit_sha)


def resolve_cross_repo_edges(driver, parent_repo: str, helper_repo: str):
    helper_repo_name = helper_repo.split("/")[-1]
    with driver.session() as session:
        session.run(CYPHER_LINK_REPO_DEPENDENCY, parent_repo=parent_repo, helper_repo=helper_repo, helper_repo_name=helper_repo_name)
        session.run(CYPHER_LINK_MODULE_TO_REPO)
        session.run(CYPHER_LINK_XREPO_CALLS, parent_repo=parent_repo, helper_repo=helper_repo, helper_prefix=helper_repo_name)
        session.run(CYPHER_LINK_XREPO_CALLS_TRANSITIVE, parent_repo=parent_repo, helper_repo=helper_repo)


def _set_status(driver, repo_full_name: str, status: str, detail: str = "", commits_processed: int = 0, files_scanned: int = 0):
    try:
        with driver.session() as session:
            session.run(
                CYPHER_UPSERT_STATUS,
                repo_full_name=repo_full_name,
                status=status,
                detail=detail,
                commits_processed=commits_processed,
                files_scanned=files_scanned,
                updated_at=datetime.now(timezone.utc).isoformat(),
            )
    except Exception as exc:
        logger.warning("Could not persist bootstrap status: %s", exc)


def phase1_scan_repo_tree(driver, repo_full_name: str, github_token: str) -> list:
    logger.info("Phase 1 — Fetching repository file tree for %s …", repo_full_name)
    gh = Github(auth=Auth.Token(github_token))
    repo = gh.get_repo(repo_full_name)
    tree = repo.get_git_tree(repo.default_branch, recursive=True).tree

    source_files = []
    with tqdm(tree, desc="  Scanning file tree", unit="item", ncols=88, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]") as pbar:
        with driver.driver.session() if hasattr(driver, 'driver') else driver.session() as session:
            for item in pbar:
                path = item.path
                if _is_noise_file(path): continue

                parent_path = path.rsplit("/", 1)[0] if "/" in path else ""
                if item.type == "tree":
                    dir_basename = path.rsplit("/", 1)[-1].lower()
                    session.run(CYPHER_TREE_DIR, repo_full_name=repo_full_name, child_path=path, parent_path=parent_path, utility=(dir_basename in UTILITY_DIRS))
                elif item.type == "blob":
                    name_no_ext = path.rsplit("/", 1)[-1].rsplit(".", 1)[0].lower()
                    ext = ("." + path.rsplit(".", 1)[1].lower()) if "." in path else ""
                    session.run(CYPHER_TREE_FILE, repo_full_name=repo_full_name, child_path=path, parent_path=parent_path, entry_point=(name_no_ext in ENTRY_POINT_NAMES))
                    if ext in SOURCE_EXTENSIONS or path.rsplit("/", 1)[-1].lower() in SOURCE_FILENAMES:
                        source_files.append(path)
    return source_files


def phase2_scan_file_contents(driver, repo_full_name: str, file_paths: list, github_token: str, google_api_key: str) -> int:
    if not file_paths: return 0
    logger.info("Phase 2 — Ingesting syntax maps for %d components …", len(file_paths))
    scanned = 0
    import_batch_params = []

    def _flush_imports():
        if not import_batch_params: return
        unique_modules = list({target for _, target, _ in import_batch_params})
        embeddings_map = dict(zip(unique_modules, get_embeddings_batch(unique_modules, google_api_key)))
        with driver.session() as sess:
            for filepath, target_module, repo_fn in import_batch_params:
                sess.run(CYPHER_INGEST_DEPENDENCY, filepath=filepath, target_module=target_module, repo_full_name=repo_fn, commit_sha="initial_scan", embedding=embeddings_map.get(target_module))
        import_batch_params.clear()

    with tqdm(file_paths, desc="  Syntax ingestion loop", unit="file", ncols=88) as pbar:
        for path in pbar:
            try:
                url = f"{GITHUB_API_BASE}/repos/{repo_full_name}/contents/{path}"
                data = _fetch_json(url, github_token=github_token)
                content_b64 = data.get("content")
                if not content_b64: continue

                source_code = base64.b64decode(content_b64).decode("utf-8", errors="replace")
                
                # Buffer import streams
                for source, _, target in extract_imports_from_source(path, source_code):
                    import_batch_params.append((source, target, repo_full_name))

                if path.endswith(".go"):
                    ast_data = parse_go_ast(path, source_code)
                elif path.endswith(".py"):
                    ast_data = parse_python_ast(path, source_code)
                else:
                    continue

                # --- Bulk Embed Subroutines ---
                funcs = ast_data.get("functions", [])
                func_embeds = []
                for i in range(0, len(funcs), _EMBED_FUNC_BATCH_SIZE):
                    func_embeds.extend(get_embeddings_batch([f"{f['name']}\n{f.get('code', '')}" for f in funcs[i:i+_EMBED_FUNC_BATCH_SIZE]], google_api_key))

                types = ast_data.get("types", [])
                type_embeds = []
                for i in range(0, len(types), _EMBED_FUNC_BATCH_SIZE):
                    type_embeds.extend(get_embeddings_batch([f"{t['name']}\n{t.get('code', '')}" for t in types[i:i+_EMBED_FUNC_BATCH_SIZE]], google_api_key))

                vars_list = ast_data.get("variables", [])
                var_embeds = []
                for i in range(0, len(vars_list), _EMBED_FUNC_BATCH_SIZE):
                    var_embeds.extend(get_embeddings_batch([f"{v['name']}\n{v.get('code', '')}" for v in vars_list[i:i+_EMBED_FUNC_BATCH_SIZE]], google_api_key))

                # --- Write Atomic Transaction Block per Component ---
                with driver.session() as sess:
                    with sess.begin_transaction() as tx:
                        test_functions = {f["name"] for f in funcs if f.get("is_test")}

                        for func, emb in zip(funcs, func_embeds):
                            qcalls = sorted(list({c[2] or c[1] for c in ast_data.get("calls", []) if c[0] == func["name"] and (len(c) > 1 and c[1])}))
                            tx.run(
                                CYPHER_INGEST_FUNCTION,
                                repo_full_name=repo_full_name, filepath=path, func_id=f"{repo_full_name}::{func['id']}",
                                func_name=func["name"], func_code=func.get("code", ""), embedding=emb or None, qualified_calls=qcalls,
                                is_exported=func.get("is_exported", False), is_pointer_receiver=func.get("is_pointer_receiver", False),
                                channels_sent=func.get("channels_sent", []), channels_received=func.get("channels_received", []),
                                return_types=func.get("return_types", []), is_test=func.get("is_test", False),
                                accepts_context=func.get("accepts_context", False), lock_sequence=func.get("lock_sequence", []),
                                propagated_errors=func.get("propagated_errors", [])
                            )

                        for call in ast_data.get("calls", []):
                            caller_name = call[0]
                            if caller_name in test_functions:
                                continue
                            tx.run(
                                CYPHER_INGEST_CALLS,
                                caller_id=f"{repo_full_name}::{path}::{call[0]}",
                                callee_name=call[1],
                                callee_qualified=call[2] if len(call) > 2 else None,
                                caller_filepath=path,
                                repo_full_name=repo_full_name,
                                call_type=call[3] if len(call) > 3 else "SYNC",
                                propagates_context=call[4] if len(call) > 4 else False,
                                creates_cancellation_scope=call[5] if len(call) > 5 else False
                            )

                        for t, emb in zip(types, type_embeds):
                            raw_f, raw_m = t.get("fields", []), t.get("methods", [])
                            tx.run(
                                CYPHER_INGEST_TYPE,
                                repo_full_name=repo_full_name, filepath=path, type_id=f"{repo_full_name}::{t['id']}",
                                type_name=t["name"], kind=t["kind"], code=t.get("code", ""),
                                fields=[json.dumps(f) for f in raw_f], field_names=[f["name"] for f in raw_f if isinstance(f, dict) and f.get("name")],
                                methods=[json.dumps(m) for m in raw_m], method_names=[m["name"] for m in raw_m if isinstance(m, dict) and m.get("name")],
                                tags=t.get("tags", []), embedding=emb or None, is_exported=t.get("is_exported", False),
                                tag_mappings=json.dumps(t.get("tag_mappings", {})),
                                type_parameters=json.dumps(t.get("type_parameters", []))
                            )
                            for emb_n in t.get("embedded_types", []):
                                tx.run(CYPHER_INGEST_TYPE_EMBEDDING, outer_id=f"{repo_full_name}::{t['id']}", repo_full_name=repo_full_name, inner_name=emb_n)

                            if t.get("kind") == "INTERFACE":
                                for m in raw_m:
                                    m_name = m.get("name") or ""
                                    if not m_name: continue
                                    scoped_m = f"{t['name']}.{m_name}"
                                    tx.run(CYPHER_INGEST_IFACE_METHOD, repo_full_name=repo_full_name, filepath=path, type_id=f"{repo_full_name}::{t['id']}", method_id=f"{repo_full_name}::{path}::{scoped_m}", method_name=scoped_m, signature=m.get("signature") or "", is_exported=m_name[0].isupper())

                        for v, emb in zip(vars_list, var_embeds):
                            tx.run(
                                CYPHER_INGEST_VARIABLE,
                                repo_full_name=repo_full_name, filepath=path, var_id=f"{repo_full_name}::{v['id']}",
                                var_name=v["name"], kind=v["kind"], code=v.get("code", ""), embedding=emb or None,
                                is_exported=v.get("is_exported", False),
                                chan_elem_type=v.get("chan_elem_type", "")
                            )

                        for d in ast_data.get("directives", []):
                            tx.run(CYPHER_INGEST_DIRECTIVE, filepath=path, repo_full_name=repo_full_name, directive=d["directive"], args=d["args"])

                        tx.run(CYPHER_LINK_DECLARES_METHOD, filepath=path, repo_full_name=repo_full_name)
                        tx.run(CYPHER_MARK_FILE_SCANNED, filepath=path, repo_full_name=repo_full_name)

                scanned += 1
                if scanned % _GRAPH_FLUSH_BATCH == 0: _flush_imports()
                time.sleep(0.05)

            except requests.exceptions.HTTPError as exc:
                if exc.response is not None and exc.response.status_code == 403: time.sleep(60)
            except Exception as exc:
                tqdm.write(f"[WARN] Ingestion anomaly on {path}: {exc}")

    _flush_imports()
    return scanned


def phase2_5_relink_calls(
    driver,
    repo_full_name: str,
    github_token: str = "",
    google_api_key: str = "",
) -> tuple[int, int, int]:
    """Phase 2.5 — Three-pass post-scan repair.

    Runs three passes after phase2_scan_file_contents has finished:

    Pass A — Re-scan files whose Function nodes have empty/null code.
        The root cause: CYPHER_MODIFIED_FUNCTION creates a bare stub with only
        an `id` property.  Phase 2's CYPHER_INGEST_FUNCTION fills it in via
        ON MATCH SET — but only if Phase 2 SUCCESSFULLY PARSED that file.  If
        the file's parse raised an exception that was caught silently, or if
        the file was already marked content_scanned=true from a prior partial
        run, Phase 2 skips it and the stub stays empty forever.
        Fix: find those filepaths, unmark them, and re-run phase2 for them.
        Requires github_token (network call to fetch file contents).

    Pass B — Re-link CALLS edges via stored qualified_calls.
        CYPHER_INGEST_CALLS runs once per call at parse time.  If the callee
        node didn't exist yet, the edge is silently dropped.  This pass
        re-attempts resolution for all entries in caller.qualified_calls.

    Pass C — Code-grep fallback for same-package calls.
        For functions with code but empty/null qualified_calls, text-match the
        callee name inside the caller's source across same-directory files.

    Returns (files_rescanned, edges_from_B, edges_from_C).
    """
    # ── Legacy/Existing Stub Migration ────────────────────────────────────────
    try:
        with driver.session() as session:
            migrate_query = """
            MATCH (f:Function)
            WHERE f.id STARTS WITH $repo_full_name + '::'
              AND (f.repo IS NULL OR f.filepath IS NULL)
              AND f.id CONTAINS '::'
            WITH f, split(f.id, '::') AS parts
            WHERE size(parts) >= 3
            SET f.repo = parts[0],
                f.filepath = parts[1],
                f.name = parts[2]
            RETURN count(f) AS migrated_count
            """
            res = session.run(migrate_query, repo_full_name=repo_full_name).single()
            migrated_cnt = res["migrated_count"] if res else 0
            if migrated_cnt > 0:
                logger.info("Phase 2.5 — Migrated %d legacy stubs with repo/filepath.", migrated_cnt)
    except Exception as exc:
        logger.warning("Phase 2.5 — Legacy stubs migration failed (non-fatal): %s", exc)

    logger.info("Phase 2.5 — Pass A: finding Function nodes with empty code …")
    files_rescanned = 0
    try:
        with driver.session() as session:
            rows = session.run(CYPHER_FIND_EMPTY_CODE_FILES, repo_full_name=repo_full_name).data()
        empty_files = [r["filepath"] for r in rows]
    except Exception as exc:
        logger.warning("Phase 2.5 — Pass A: query failed: %s", exc)
        empty_files = []

    if not empty_files:
        logger.info("Phase 2.5 — Pass A: no empty-code stubs found. Graph is clean.")
    elif not github_token:
        logger.warning(
            "Phase 2.5 — Pass A: %d file(s) with empty-code stubs detected but "
            "no GITHUB_TOKEN provided — skipping re-scan. "
            "Run with GITHUB_TOKEN set to fix these stubs: %s",
            len(empty_files), empty_files[:5],
        )
    else:
        logger.info(
            "Phase 2.5 — Pass A: %d file(s) need re-scanning: %s",
            len(empty_files), empty_files[:5],
        )
        try:
            with driver.session() as session:
                session.run(CYPHER_UNMARK_FILES_FOR_RESCAN, repo_full_name=repo_full_name, paths=empty_files)
            logger.info("Phase 2.5 — Pass A: unmarked %d file(s) for re-scan.", len(empty_files))
            files_rescanned = phase2_scan_file_contents(
                driver, repo_full_name, empty_files, github_token, google_api_key
            )
            logger.info("Phase 2.5 — Pass A complete: re-scanned %d file(s).", files_rescanned)
        except Exception as exc:
            logger.warning("Phase 2.5 — Pass A: re-scan failed (non-fatal): %s", exc)

    # ── Pass B: re-link via qualified_calls ───────────────────────────────────
    logger.info("Phase 2.5 — Pass B: re-linking unresolved CALLS via qualified_calls …")
    edges_b = 0
    try:
        with driver.session() as session:
            result = session.run(CYPHER_RELINK_UNRESOLVED_CALLS, repo_full_name=repo_full_name)
            summary = result.single()
            edges_b = summary["edges_created"] if summary else 0
        logger.info("Phase 2.5 — Pass B complete: %d new CALLS edge(s).", edges_b)
    except Exception as exc:
        logger.warning("Phase 2.5 — Pass B failed (non-fatal): %s", exc)

    # ── Pass C: code-grep fallback for same-package calls ────────────────────
    logger.info("Phase 2.5 — Pass C: code-grep same-package call fallback …")
    edges_c = 0
    try:
        with driver.session() as session:
            result = session.run(CYPHER_RELINK_CALLS_BY_CODE_GREP, repo_full_name=repo_full_name)
            summary = result.single()
            edges_c = summary["edges_created"] if summary else 0
        logger.info("Phase 2.5 — Pass C complete: %d new CALLS edge(s) via code-grep.", edges_c)
    except Exception as exc:
        logger.warning("Phase 2.5 — Pass C failed (non-fatal): %s", exc)

    logger.info(
        "Phase 2.5 summary — files_rescanned=%d  edges_B=%d  edges_C=%d",
        files_rescanned, edges_b, edges_c,
    )
    return files_rescanned, edges_b, edges_c


def phase3_backfill_commits(driver, repo_full_name: str, github_token: str, google_api_key: str, max_commits: int = 200, deep_scan_days: int = 7, skip_llm: bool = False, force_llm_update: bool = False) -> int:
    logger.info("Phase 3 — Chronological commit trace synchronization…")
    processed = 0
    page, per_page = 1, 100
    commit_shas = []

    while len(commit_shas) < max_commits:
        try:
            url = f"{GITHUB_API_BASE}/repos/{repo_full_name}/commits"
            resp = requests.get(url, headers=_github_headers(github_token), params={"per_page": per_page, "page": page}, timeout=30)
            resp.raise_for_status()
            page_data = resp.json()
            if not page_data: break
            for c in page_data:
                if len(commit_shas) >= max_commits: break
                if c.get("sha"): commit_shas.append(c["sha"])
            if len(page_data) < per_page: break
            page += 1
            time.sleep(0.2)
        except Exception:
            break

    with driver.session() as session:
        existing_shas = {record["sha"] for record in session.run("MATCH (c:Commit) RETURN c.sha AS sha")}

    if not force_llm_update:
        commit_shas = [sha for sha in commit_shas if sha not in existing_shas]
    if not commit_shas: return 0

    cutoff_date = datetime.now(timezone.utc) - timedelta(days=deep_scan_days)
    commit_data = asyncio.run(_fetch_all_commits_async(repo_full_name, commit_shas, github_token))

    with tqdm(commit_shas, desc="  Commit backfill", unit="commit", ncols=88) as pbar:
        for sha in pbar:
            try:
                payload, files = commit_data.get(sha, ({}, []))
                if not payload: continue
                compact_files = build_compact_diff(files)

                c_info = payload.get("commit", {})
                actor = (payload.get("author") or {}).get("login") or (c_info.get("author") or {}).get("name") or "unknown"
                msg = c_info.get("message", "")
                c_date = c_info.get("author", {}).get("date", "")

                summary = _heuristic_summary(repo_full_name, compact_files, msg, actor_login=actor) if skip_llm else summarize_with_llm(repo_full_name=repo_full_name, actor_login=actor, commit_msg=msg, compact_files=compact_files, raw_diff=render_diff_text(compact_files), google_api_key=google_api_key)

                _write_commit_to_graph(driver, repo_full_name=repo_full_name, commit_sha=sha, modified_files=[item["filename"] for item in compact_files], dependencies=extract_temporal_dependencies(compact_files), actor_login=actor, committed_at=c_date, summary_text=summary, diff_text=render_diff_text(compact_files), commit_message=msg, commit_url=payload.get("html_url", ""))

                if c_date and datetime.strptime(c_date, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc) >= cutoff_date:
                    for item in compact_files:
                        fp = item["filename"]
                        try:
                            raw_data = _fetch_json(f"{GITHUB_API_BASE}/repos/{repo_full_name}/contents/{fp}?ref={sha}", github_token=github_token)
                            code = base64.b64decode(raw_data["content"]).decode("utf-8")
                            h_ast = parse_go_ast(fp, code) if fp.endswith(".go") else parse_python_ast(fp, code)
                            for f_id in get_modified_functions(item.get("patch", ""), fp, h_ast):
                                func_name = f_id.split("::")[-1]
                                with driver.session() as sess:
                                    sess.run(
                                        CYPHER_MODIFIED_FUNCTION,
                                        commit_sha=sha,
                                        func_id=f"{repo_full_name}::{f_id}",
                                        func_name=func_name,
                                        filepath=fp,
                                        repo_full_name=repo_full_name,
                                    )
                        except Exception:
                            pass
                processed += 1
            except Exception:
                continue
    return processed


def get_unprocessed_files(driver, repo_full_name: str, source_files: list) -> list:
    with driver.session() as session:
        scanned = {r["path"] for r in session.run("MATCH (f:File {repo: $repo, content_scanned: true}) RETURN f.path AS path", repo=repo_full_name)}
    return [f for f in source_files if f not in scanned]


def bootstrap(repo_full_name: str, github_token: str, google_api_key: str, neo4j_uri: str, neo4j_user: str, neo4j_password: str, max_commits: int = 200, skip_llm: bool = False, force_llm_update: bool = False):
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    try: driver.verify_connectivity()
    except Exception as exc: raise SystemExit(1) from exc

    gh = Github(auth=Auth.Token(github_token))
    repo = gh.get_repo(repo_full_name)

    with driver.session() as session:
        session.run(CYPHER_CREATE_REPO, repo_full_name=repo_full_name, repo_name=repo.name, repo_url=repo.html_url)

    for cypher, _ in [
        (CYPHER_VECTOR_INDEX, ""), (CYPHER_CODE_VECTOR_INDEX, ""), (CYPHER_MODULE_VECTOR_INDEX, ""),
        (CYPHER_TYPE_VECTOR_INDEX, ""), (CYPHER_VAR_VECTOR_INDEX, ""), (CYPHER_FULLTEXT_INDEX, ""),
        (CYPHER_CONSTRAINT_FILE_REPO, ""), (CYPHER_CONSTRAINT_FUNC_ID, ""), (CYPHER_CONSTRAINT_TYPE_ID, ""),
        (CYPHER_CONSTRAINT_VAR_ID, ""), (CYPHER_CONSTRAINT_DIRECTIVE_UNIQUE, ""), (CYPHER_INDEX_FUNC_NAME_REPO, ""),
        (CYPHER_INDEX_TYPE_NAME_REPO, ""), (CYPHER_INDEX_TYPE_KIND, "")
    ]:
        try:
            with driver.session() as session: session.run(cypher)
        except Exception: pass

    for cypher in [CYPHER_CLEANUP_UNSCOPED_FILES, CYPHER_CLEANUP_UNSCOPED_DIRS, CYPHER_CLEANUP_UNSCOPED_FUNCS]:
        try:
            with driver.session() as session: session.run(cypher)
        except Exception: pass

    _set_status(driver, repo_full_name, "in_progress", "Running Phase 1 tree resolution…")
    src_files = phase1_scan_repo_tree(driver, repo_full_name, github_token)
    
    unscanned = get_unprocessed_files(driver, repo_full_name, src_files)
    scanned_cnt = phase2_scan_file_contents(driver, repo_full_name, unscanned, github_token, google_api_key)

    _set_status(driver, repo_full_name, "in_progress", "Running Phase 2.5 CALLS re-resolution…")
    files_rescanned, edges_b, edges_c = phase2_5_relink_calls(
        driver, repo_full_name, github_token=github_token, google_api_key=google_api_key
    )
    logger.info(
        "Phase 2.5 summary — files_rescanned=%d  edges_B=%d  edges_C=%d",
        files_rescanned, edges_b, edges_c,
    )

    commits_cnt = phase3_backfill_commits(driver, repo_full_name, github_token, google_api_key, max_commits, skip_llm=skip_llm, force_llm_update=force_llm_update)
    _set_status(driver, repo_full_name, "completed", "Processing finalized.", commits_processed=commits_cnt, files_scanned=scanned_cnt)
    driver.close()