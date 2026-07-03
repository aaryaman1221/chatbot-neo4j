# =============================================================================
# ingest/pipeline.py — Pipeline Orchestration and Graph Database Bootstrap
# =============================================================================

import base64
import time
import asyncio
from datetime import datetime, timedelta, timezone
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
    CYPHER_VECTOR_INDEX,
    CYPHER_CODE_VECTOR_INDEX,
    CYPHER_MODULE_VECTOR_INDEX,
    CYPHER_FULLTEXT_INDEX,
    CYPHER_UPSERT_STATUS,
    CYPHER_CONSTRAINT_FILE_REPO,
    CYPHER_CONSTRAINT_FUNC_ID,
    CYPHER_CLEANUP_UNSCOPED_FILES,
    CYPHER_CLEANUP_UNSCOPED_DIRS,
    CYPHER_CLEANUP_UNSCOPED_FUNCS,
    CYPHER_MARK_FILE_SCANNED,
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
    get_embedding,
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
            session.run(
                CYPHER_INGEST_FILE,
                repo_full_name=repo_full_name,
                commit_sha=commit_sha,
                filepath=filepath,
            )
        # --- Temporal dependency upserts (added imports) ---
        for source, _rel, target in dependencies.get("added", []):
            session.run(
                CYPHER_INGEST_DEPENDENCY,
                filepath=source,
                target_module=target,
                repo_full_name=repo_full_name,
                commit_sha=commit_sha,
                embedding=None,
            )
        # --- Temporal dependency soft-deletes (removed imports) ---
        for source, _rel, target in dependencies.get("removed", []):
            session.run(
                CYPHER_REMOVE_DEPENDENCY,
                filepath=source,
                target_module=target,
                repo_full_name=repo_full_name,
                commit_sha=commit_sha,
            )


def resolve_cross_repo_edges(driver, parent_repo: str, helper_repo: str):
    helper_repo_name = helper_repo.split("/")[-1]
    with driver.session() as session:
        result = session.run(
            CYPHER_LINK_REPO_DEPENDENCY,
            parent_repo=parent_repo,
            helper_repo=helper_repo,
            helper_repo_name=helper_repo_name,
        )
        summary = result.consume()
        logger.info(
            "Cross-repo edges: %d USES_REPO relationships created (%s → %s)",
            summary.counters.relationships_created,
            parent_repo,
            helper_repo,
        )
        res_mod = session.run(CYPHER_LINK_MODULE_TO_REPO)
        sum_mod = res_mod.consume()
        logger.info("Module to Repository links created: %d", sum_mod.counters.relationships_created)

        res_calls = session.run(
            CYPHER_LINK_XREPO_CALLS,
            parent_repo=parent_repo,
            helper_repo=helper_repo,
            helper_prefix=helper_repo_name,
        )
        sum_calls = res_calls.consume()
        logger.info(
            "Cross-repo CALLS (direct & wrapper modules) created (%s → %s): %d",
            parent_repo,
            helper_repo,
            sum_calls.counters.relationships_created,
        )


def _set_status(driver, repo_full_name: str, status: str, detail: str = "",
                commits_processed: int = 0, files_scanned: int = 0):
    try:
        with driver.session() as session:
            session.run(
                CYPHER_UPSERT_STATUS,
                repo_full_name=repo_full_name,
                status=status,
                detail=detail,
                commits_processed=commits_processed,
                files_scanned=files_scanned,
                updated_at=datetime.utcnow().isoformat(),
            )
    except Exception as exc:
        logger.warning("Could not persist bootstrap status: %s", exc)


def phase1_scan_repo_tree(driver, repo_full_name: str, github_token: str) -> list:
    logger.info("Phase 1 — Fetching repository file tree for %s …", repo_full_name)

    auth = Auth.Token(github_token)
    gh = Github(auth=auth)
    repo = gh.get_repo(repo_full_name)
    default_branch = repo.default_branch
    tree = repo.get_git_tree(default_branch, recursive=True).tree

    source_files = []
    dirs_seen = set()

    with tqdm(tree, desc="  Scanning file tree", unit="item", ncols=88,
               bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]") as pbar:
        with driver.session() as session:
            for item in pbar:
                path = item.path
                item_type = item.type 
                pbar.set_postfix_str(path[-50:] if len(path) > 50 else path, refresh=False)

                if _is_noise_file(path):
                    continue

                if item_type == "tree":
                    parent_path = path.rsplit("/", 1)[0] if "/" in path else ""
                    dir_basename = path.rsplit("/", 1)[-1].lower()
                    is_utility = dir_basename in UTILITY_DIRS
                    session.run(
                        CYPHER_TREE_DIR,
                        repo_full_name=repo_full_name,
                        child_path=path,
                        parent_path=parent_path,
                        utility=is_utility,
                    )
                    dirs_seen.add(path)

                elif item_type == "blob":
                    parent_path = path.rsplit("/", 1)[0] if "/" in path else ""
                    name_no_ext = path.rsplit("/", 1)[-1].rsplit(".", 1)[0].lower()
                    is_entry = name_no_ext in ENTRY_POINT_NAMES
                    ext = ("." + path.rsplit(".", 1)[1].lower()) if "." in path else ""
                    session.run(
                        CYPHER_TREE_FILE,
                        repo_full_name=repo_full_name,
                        child_path=path,
                        parent_path=parent_path,
                        entry_point=is_entry,
                    )
                    fname_lower = path.rsplit("/", 1)[-1].lower()
                    if ext in SOURCE_EXTENSIONS or fname_lower in SOURCE_FILENAMES:
                        source_files.append(path)

    logger.info(
        "Phase 1 complete — %d source files, %d directories indexed.",
        len(source_files), len(dirs_seen),
    )
    return source_files


def phase2_scan_file_contents(
    driver,
    repo_full_name: str,
    file_paths: list,
    github_token: str,
    google_api_key: str,
) -> int:
    if not file_paths:
        return 0

    logger.info("Phase 2 — Scanning %d source files for import dependencies …", len(file_paths))
    scanned = 0
    batch_params: list[tuple] = []

    def _flush_batch():
        if not batch_params:
            return
        unique_modules = list({target for _, target, _ in batch_params})
        batch_vectors = get_embeddings_batch(unique_modules, google_api_key)
        embeddings_map: dict = dict(zip(unique_modules, batch_vectors))

        with driver.session() as sess:
            for filepath, target_module, repo_fn in batch_params:
                emb = embeddings_map.get(target_module) or None
                sess.run(
                    CYPHER_INGEST_DEPENDENCY,
                    filepath=filepath,
                    target_module=target_module,
                    repo_full_name=repo_fn,
                    commit_sha="initial_scan",
                    embedding=emb,
                )
        batch_params.clear()

    with tqdm(file_paths, desc="  Dependency scan", unit="file", ncols=88,
               bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:
        for path in pbar:
            pbar.set_postfix_str(path[-50:] if len(path) > 50 else path, refresh=False)
            try:
                url = f"{GITHUB_API_BASE}/repos/{repo_full_name}/contents/{path}"
                data = _fetch_json(url, github_token=github_token)
                content_b64 = data.get("content")
                if not content_b64:
                    continue

                source_code = base64.b64decode(content_b64).decode("utf-8", errors="replace")
                
                deps = extract_imports_from_source(path, source_code)
                for source, _rel, target in deps:
                    batch_params.append((source, target, repo_full_name))

                if path.endswith(".go"):
                    ast_data = parse_go_ast(path, source_code)
                else:
                    ast_data = parse_python_ast(path, source_code)
                with driver.session() as sess:
                    for func in ast_data["functions"]:
                        prefixed_id = f"{repo_full_name}::{func['id']}"
                        embed_text = f"{func['name']}\n{func.get('code', '')}"
                        qual_calls_set = set()
                        for c in ast_data["calls"]:
                            if c[0] == func["name"]:
                                if len(c) > 2 and c[2]:
                                    qual_calls_set.add(c[2])
                                if len(c) > 1 and c[1]:
                                    qual_calls_set.add(c[1])
                        func_qual_calls = sorted(list(qual_calls_set))
                        sess.run(
                            CYPHER_INGEST_FUNCTION,
                            repo_full_name=repo_full_name,
                            filepath=path,
                            func_id=prefixed_id,
                            func_name=func["name"],
                            func_code=func.get("code", ""),
                            embedding=get_embedding(embed_text, google_api_key),
                            qualified_calls=func_qual_calls,
                        )
                    for call_record in ast_data["calls"]:
                        caller = call_record[0]
                        callee_bare = call_record[1]
                        caller_id = f"{repo_full_name}::{path}::{caller}"
                        sess.run(
                            CYPHER_INGEST_CALLS,
                            caller_id=caller_id,
                            callee_name=callee_bare,
                            caller_filepath=path,
                            repo_full_name=repo_full_name,
                        )
                    sess.run(
                        CYPHER_MARK_FILE_SCANNED,
                        filepath=path,
                        repo_full_name=repo_full_name,
                    )

                scanned += 1
                if scanned % _GRAPH_FLUSH_BATCH == 0:
                    _flush_batch()

                time.sleep(0.1)

            except requests.exceptions.HTTPError as exc:
                if exc.response is not None and exc.response.status_code == 403:
                    tqdm.write(f"[WARN] Rate-limit on {path}, sleeping 60s …")
                    time.sleep(60)
                else:
                    tqdm.write(f"[WARN] HTTP error scanning {path}: {exc}")
            except Exception as exc:
                tqdm.write(f"[WARN] Error scanning {path}: {exc}")

    _flush_batch()
    logger.info("Phase 2 complete — %d/%d files scanned.", scanned, len(file_paths))
    return scanned


def phase3_backfill_commits(
    driver,
    repo_full_name: str,
    github_token: str,
    google_api_key: str,
    max_commits: int = 200,
    deep_scan_days: int = 7,
    skip_llm: bool = False,
    force_llm_update: bool = False,
) -> int:
    logger.info(
        "Phase 3 — Backfilling up to %d commits for %s …", max_commits, repo_full_name
    )

    processed = 0
    page = 1
    per_page = 100
    commit_shas: list[str] = []

    print("\n  Collecting commit list …")
    while len(commit_shas) < max_commits:
        try:
            url = f"{GITHUB_API_BASE}/repos/{repo_full_name}/commits"
            resp = requests.get(
                url,
                headers=_github_headers(github_token),
                params={"per_page": per_page, "page": page},
                timeout=30,
            )
            resp.raise_for_status()
            page_data = resp.json()
            if not page_data:
                break
            for c in page_data:
                if len(commit_shas) >= max_commits:
                    break
                sha = c.get("sha")
                if sha:
                    commit_shas.append(sha)
            if len(page_data) < per_page:
                break
            page += 1
            time.sleep(0.3)
        except Exception as exc:
            logger.warning("Error fetching commit page %d: %s", page, exc)
            break

    print("\n  Checking Neo4j for existing commits to save tokens …")
    existing_shas = set()
    try:
        with driver.session() as session:
            result = session.run("MATCH (c:Commit) RETURN c.sha AS sha")
            existing_shas = {record["sha"] for record in result}
    except Exception as exc:
        logger.warning("Could not fetch existing commits: %s", exc)

    if force_llm_update and existing_shas:
        logger.info(
            "  FORCE_LLM_UPDATE=true — re-processing all %d commits through the LLM "
            "to overwrite heuristic summaries (skipping none).",
            len(commit_shas),
        )
    else:
        new_commit_shas = [sha for sha in commit_shas if sha not in existing_shas]
        skipped = len(commit_shas) - len(new_commit_shas)
        logger.info("  Skipping %d already ingested commits.", skipped)
        logger.info("  Proceeding with %d new commits.", len(new_commit_shas))
        commit_shas = new_commit_shas

        if not commit_shas:
            logger.info("  No new commits to process. Exiting Phase 3 early.")
            return 0

    cutoff_date = datetime.now(timezone.utc) - timedelta(days=deep_scan_days)
    print("\n  Fetching commit diffs concurrently…")
    commit_data = asyncio.run(
        _fetch_all_commits_async(repo_full_name, commit_shas, github_token)
    )

    with tqdm(commit_shas, desc="  Backfilling commits", unit="commit", ncols=88,
               bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
        for sha in pbar:
            pbar.set_postfix_str(sha[:8], refresh=False)
            try:
                commit_payload, files = commit_data.get(sha, ({}, []))
                if not commit_payload:
                    continue
                compact_files = build_compact_diff(files)

                commit_info = commit_payload.get("commit", {})
                actor_login = (
                    (commit_payload.get("author") or {}).get("login")
                    or (commit_info.get("author") or {}).get("name")
                    or "unknown"
                )
                commit_msg  = commit_info.get("message", "")
                commit_date = commit_info.get("author", {}).get("date", "")
                commit_url  = commit_payload.get("html_url", "")

                modified_files_list = [item["filename"] for item in compact_files]
                dependencies = extract_temporal_dependencies(compact_files)
                raw_diff = render_diff_text(compact_files)

                is_deep_scan = False
                if commit_date:
                    commit_dt = datetime.strptime(commit_date, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
                    is_deep_scan = commit_dt >= cutoff_date

                summary_text = (
                    _heuristic_summary(repo_full_name, compact_files, commit_msg)
                    if skip_llm
                    else summarize_with_llm(
                        repo_full_name=repo_full_name,
                        actor_login=actor_login,
                        commit_msg=commit_msg,
                        compact_files=compact_files,
                        raw_diff=raw_diff,
                        google_api_key=google_api_key,
                    )
                )

                _write_commit_to_graph(
                    driver,
                    repo_full_name=repo_full_name,
                    commit_sha=sha,
                    modified_files=modified_files_list,
                    dependencies=dependencies,
                    actor_login=actor_login,
                    committed_at=commit_date,
                    summary_text=summary_text,
                    diff_text=raw_diff,
                    commit_message=commit_msg,
                    commit_url=commit_url,
                )

                if is_deep_scan:
                    for item in compact_files:
                        filepath = item["filename"]
                        patch = item.get("patch", "")
                        try:
                            raw_url = f"{GITHUB_API_BASE}/repos/{repo_full_name}/contents/{filepath}?ref={sha}"
                            raw_data = _fetch_json(raw_url, github_token=github_token)
                            raw_code = base64.b64decode(raw_data["content"]).decode("utf-8")
                            if filepath.endswith(".go"):
                                historical_ast = parse_go_ast(filepath, raw_code)
                            else:
                                historical_ast = parse_python_ast(filepath, raw_code)
                            mod_funcs = get_modified_functions(patch, filepath, historical_ast)
                            with driver.session() as sess:
                                for func_id in mod_funcs:
                                    prefixed_id = f"{repo_full_name}::{func_id}"
                                    sess.run(CYPHER_MODIFIED_FUNCTION,
                                             commit_sha=sha, func_id=prefixed_id)
                        except Exception as exc:
                            logger.debug(
                                "Deep-scan function attribution failed for %s@%s: %s",
                                filepath, sha[:8], exc,
                            )

                processed += 1
                time.sleep(0.5)

            except requests.exceptions.HTTPError as exc:
                if exc.response is not None and exc.response.status_code == 403:
                    tqdm.write(f"[WARN] Rate-limit hit on commit {sha[:8]}, sleeping 60s …")
                    time.sleep(60)
                else:
                    tqdm.write(f"[WARN] HTTP error on commit {sha[:8]}: {exc}")
            except Exception as exc:
                tqdm.write(f"[WARN] Failed to process commit {sha[:8]}: {exc}")
                continue

    logger.info("Phase 3 complete — %d/%d commits ingested.", processed, len(commit_shas))
    return processed


def get_unprocessed_files(driver, repo_full_name: str, source_files: list) -> list:
    already_scanned: set[str] = set()
    try:
        with driver.session() as session:
            result = session.run(
                "MATCH (f:File {repo: $repo, content_scanned: true}) RETURN f.path AS path",
                repo=repo_full_name,
            )
            already_scanned = {record["path"] for record in result}
    except Exception as exc:
        logger.warning("Could not fetch already-scanned files: %s", exc)

    unprocessed = [f for f in source_files if f not in already_scanned]
    skipped = len(source_files) - len(unprocessed)
    if skipped:
        logger.info(
            "  Resuming Phase 2 — skipping %d already-scanned file(s), %d remaining.",
            skipped, len(unprocessed),
        )
    return unprocessed


def bootstrap(
    repo_full_name: str,
    github_token: str,
    google_api_key: str,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    max_commits: int = 200,
    skip_llm: bool = False,
    force_llm_update: bool = False,
):
    print("=" * 70)
    print(f"  GraphRAG Ingest Pipeline — {repo_full_name}")
    print("=" * 70)

    logger.info("Connecting to Neo4j at %s …", neo4j_uri)
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    try:
        driver.verify_connectivity()
        logger.info("Neo4j connection established ✓")
    except Exception as exc:
        logger.error("Cannot connect to Neo4j: %s", exc)
        raise SystemExit(1) from exc

    auth = Auth.Token(github_token)
    gh = Github(auth=auth)
    try:
        repo = gh.get_repo(repo_full_name)
    except GithubException as exc:
        logger.error("GitHub error: %s", exc)
        raise SystemExit(1) from exc

    with driver.session() as session:
        session.run(
            CYPHER_CREATE_REPO,
            repo_full_name=repo_full_name,
            repo_name=repo.name,
            repo_url=repo.html_url,
        )

    logger.info("Creating Neo4j indexes and constraints …")
    for cypher, label in [
        (CYPHER_VECTOR_INDEX,         "vector index issue_embeddings"),
        (CYPHER_CODE_VECTOR_INDEX,    "vector index code_embeddings"),
        (CYPHER_MODULE_VECTOR_INDEX,  "vector index module_embeddings"),
        (CYPHER_FULLTEXT_INDEX,       "fulltext index commit_summaries"),
        (CYPHER_CONSTRAINT_FILE_REPO, "composite uniqueness: File(path, repo)"),
        (CYPHER_CONSTRAINT_FUNC_ID,   "uniqueness: Function(id)"),
    ]:
        try:
            with driver.session() as session:
                session.run(cypher)
            logger.info("  ✓ %s", label)
        except Exception as exc:
            logger.warning("  Index/constraint note (%s): %s", label, exc)

    logger.info("Pre-flight orphan cleanup …")
    for cypher, label in [
        (CYPHER_CLEANUP_UNSCOPED_FILES, "unscoped File nodes"),
        (CYPHER_CLEANUP_UNSCOPED_DIRS,  "unscoped Directory nodes"),
        (CYPHER_CLEANUP_UNSCOPED_FUNCS, "unscoped Function nodes"),
    ]:
        try:
            with driver.session() as session:
                result  = session.run(cypher)
                deleted = result.consume().counters.nodes_deleted
                if deleted:
                    logger.info("  ✓ Cleaned up %d %s", deleted, label)
        except Exception as exc:
            logger.warning("  Cleanup note (%s): %s", label, exc)

    _set_status(driver, repo_full_name, "in_progress", "Starting Phase 1 …")

    print()
    source_files = phase1_scan_repo_tree(driver, repo_full_name, github_token)
    _set_status(driver, repo_full_name, "in_progress",
                f"Phase 1 done — {len(source_files)} source files indexed.",
                files_scanned=len(source_files))

    print()
    unscanned_files = get_unprocessed_files(driver, repo_full_name, source_files)
    files_scanned = phase2_scan_file_contents(
        driver, repo_full_name, unscanned_files, github_token, google_api_key
    )
    _set_status(driver, repo_full_name, "in_progress",
                f"Phase 2 done — {files_scanned} files scanned for imports.",
                files_scanned=files_scanned)

    print()
    commits_processed = phase3_backfill_commits(
        driver, repo_full_name, github_token, google_api_key, max_commits,
        skip_llm=skip_llm,
        force_llm_update=force_llm_update,
    )

    _set_status(
        driver,
        repo_full_name,
        status="completed",
        detail=(
            f"Bootstrap complete — {files_scanned} files scanned, "
            f"{commits_processed} commits ingested."
        ),
        commits_processed=commits_processed,
        files_scanned=files_scanned,
    )

    driver.close()

    print()
    print("=" * 70)
    print(f"  ✅  Bootstrap COMPLETE for {repo_full_name}")
    print(f"      Files scanned  : {files_scanned}")
    print(f"      Commits stored : {commits_processed}")
    print("=" * 70)
