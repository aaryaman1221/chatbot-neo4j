#!/usr/bin/env python3
"""
Migration script to clean up ambiguous CALLS edges and establish cross-repo links
without requiring a full repository re-ingestion.

Usage:
    python migrate_calls_edges.py [--dry-run]
"""

import os
import sys
import argparse
import logging
from dotenv import load_dotenv
import json
import zipfile
import io
import requests
from neo4j import GraphDatabase

from ingest.queries import (
    CYPHER_LINK_REPO_DEPENDENCY,
    CYPHER_LINK_MODULE_TO_REPO,
    CYPHER_LINK_XREPO_CALLS,
)
from ingest.ai_service import get_embeddings_batch
from ingest.parser import parse_go_ast, parse_python_ast
from ingest.queries import (
    CYPHER_INGEST_TYPE,
    CYPHER_INGEST_TYPE_EMBEDDING,
    CYPHER_INGEST_VARIABLE,
    CYPHER_INGEST_DIRECTIVE,
    CYPHER_INGEST_IFACE_METHOD,
    CYPHER_LINK_DECLARES_METHOD,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("migrate_calls_edges")

CYPHER_CLEANUP_AMBIGUOUS_CALLS = """
MATCH (caller:Function)-[r:CALLS]->(callee:Function)
WHERE caller.repo = callee.repo
  AND caller.filepath <> callee.filepath
  AND NOT coalesce(r.cross_repo, false)
WITH caller, r, callee
MATCH (local:Function {name: callee.name, filepath: caller.filepath, repo: caller.repo})
WHERE local <> callee
DELETE r
RETURN count(*) AS deleted_edges
"""

CYPHER_CLEANUP_FALSE_BARE_CALLS = """
MATCH (caller:Function)-[r:CALLS]->(callee:Function)
WHERE caller.repo = callee.repo
  AND caller.filepath <> callee.filepath
  AND split(caller.filepath, '/')[0..-1] <> split(callee.filepath, '/')[0..-1]
  AND NOT coalesce(r.cross_repo, false)
  AND NOT callee.name CONTAINS '.'
  AND (
       callee.name IN ['New', 'Run', 'Execute', 'Close', 'Open', 'Init', 'String', 'Read', 'Write', 'Update', 'Focus', 'Blur', 'Blink', 'Start', 'Stop', 'Reset', 'Clear', 'Add', 'Remove', 'Delete', 'Get', 'Set', 'List', 'Find', 'Check', 'Verify', 'Validate', 'Parse', 'Format', 'Print', 'Println', 'Error', 'Fatal', 'Panic', 'Log', 'Debug', 'Info', 'Warn', 'Main', 'Test', 'Setup', 'Teardown', 'Config', 'Load', 'Save', 'Create', 'Send', 'Process', 'Handle']
    OR size(callee.name) < 5
    OR NOT EXISTS {
       MATCH (caller_file:File {path: caller.filepath, repo: caller.repo})-[:DEPENDS_ON]->(m:Module)
       WHERE toLower(callee.filepath) CONTAINS toLower(m.name) OR toLower(m.name) ENDS WITH toLower(split(callee.filepath, '/')[0])
    }
  )
DELETE r
RETURN count(*) AS deleted_edges
"""

CYPHER_CLEANUP_FALSE_XREPO_CALLS = """
MATCH (caller:Function)-[r:CALLS {cross_repo: true}]->(callee:Function)
WHERE NOT EXISTS {
  MATCH (f:File {repo: caller.repo})-[:USES_REPO]->(r_helper:Repository {full_name: callee.repo})
}
DELETE r
RETURN count(*) AS deleted_edges
"""

CYPHER_GET_REPOS = """
MATCH (r:Repository)
RETURN r.full_name AS full_name
"""

def _process_and_ingest_ast(session, repo_full_name: str, file_items: list, api_key: str):
    logger.info("Parsing AST for %d files in %s...", len(file_items), repo_full_name)
    all_types = []
    all_vars = []
    all_directives = []
    for rel_path, code_str in file_items:
        try:
            ast_data = parse_go_ast(rel_path, code_str)
            for t in ast_data.get("types", []):
                all_types.append((t, rel_path))
            for v in ast_data.get("variables", []):
                all_vars.append((v, rel_path))
            for d in ast_data.get("directives", []):
                all_directives.append((d, rel_path))
        except Exception as exc:
            logger.debug("Error parsing %s: %s", rel_path, exc)

    logger.info("Extracted %d types, %d variables, %d directives. Generating embeddings in bulk...", len(all_types), len(all_vars), len(all_directives))
    
    type_embeds = []
    if all_types and api_key:
        type_texts = [f"{t['name']}\n{t.get('code', '')}" for t, _ in all_types]
        for i in range(0, len(type_texts), 64):
            chunk = type_texts[i : i + 64]
            type_embeds.extend(get_embeddings_batch(chunk, api_key))
    else:
        type_embeds = [None] * len(all_types)

    var_embeds = []
    if all_vars and api_key:
        var_texts = [f"{v['name']}\n{v.get('code', '')}" for v, _ in all_vars]
        for i in range(0, len(var_texts), 64):
            chunk = var_texts[i : i + 64]
            var_embeds.extend(get_embeddings_batch(chunk, api_key))
    else:
        var_embeds = [None] * len(all_vars)

    logger.info("Writing nodes and embeddings to Neo4j...")
    for (t, rel_path), emb in zip(all_types, type_embeds):
        tid = f"{repo_full_name}::{t['id']}"
        raw_fields = t.get("fields", [])
        raw_methods = t.get("methods", [])
        fields_json = [json.dumps(f) for f in raw_fields]
        methods_json = [json.dumps(m) for m in raw_methods]
        field_names = [f["name"] for f in raw_fields if isinstance(f, dict) and f.get("name")]
        method_names = [m["name"] for m in raw_methods if isinstance(m, dict) and m.get("name")]
        session.run(
            CYPHER_INGEST_TYPE,
            repo_full_name=repo_full_name,
            filepath=rel_path,
            type_id=tid,
            type_name=t["name"],
            kind=t["kind"],
            code=t.get("code", ""),
            fields=fields_json,
            field_names=field_names,
            methods=methods_json,
            method_names=method_names,
            tags=t.get("tags", []),
            embedding=emb or None,
            is_exported=t.get("is_exported", False),
        )
        for embed_name in t.get("embedded_types", []):
            session.run(
                CYPHER_INGEST_TYPE_EMBEDDING,
                outer_id=tid,
                repo_full_name=repo_full_name,
                inner_name=embed_name,
            )
        if t.get("kind") == "INTERFACE":
            for method in raw_methods:
                m_name = method.get("name") or ""
                m_sig = method.get("signature") or ""
                if not m_name:
                    continue
                scoped_method_name = f"{t['name']}.{m_name}"
                method_id = f"{repo_full_name}::{rel_path}::{scoped_method_name}"
                session.run(
                    CYPHER_INGEST_IFACE_METHOD,
                    repo_full_name=repo_full_name,
                    filepath=rel_path,
                    type_id=tid,
                    method_id=method_id,
                    method_name=scoped_method_name,
                    signature=m_sig,
                    is_exported=m_name[0].isupper() if m_name else False,
                )

    for (v, rel_path), emb in zip(all_vars, var_embeds):
        vid = f"{repo_full_name}::{v['id']}"
        session.run(
            CYPHER_INGEST_VARIABLE,
            repo_full_name=repo_full_name,
            filepath=rel_path,
            var_id=vid,
            var_name=v["name"],
            kind=v["kind"],
            code=v.get("code", ""),
            embedding=emb or None,
            is_exported=v.get("is_exported", False),
        )

    for d, rel_path in all_directives:
        session.run(
            CYPHER_INGEST_DIRECTIVE,
            filepath=rel_path,
            repo_full_name=repo_full_name,
            directive=d["directive"],
            args=d["args"],
        )
    unique_paths = sorted(list({rel_path for _, rel_path in file_items}))
    for rel_path in unique_paths:
        session.run(
            CYPHER_LINK_DECLARES_METHOD,
            filepath=rel_path,
            repo_full_name=repo_full_name,
        )
    logger.info("✅ Completed bulk AST ingestion for %s.", repo_full_name)

def main():
    from pathlib import Path
    parser = argparse.ArgumentParser(description="Clean up ambiguous CALLS edges and migrate cross-repo links.")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without modifying the database.")
    parser.add_argument("--scan-local", type=str, help="Scan a local repository directory for zero-reingestion Type/Var/Directive extraction.")
    parser.add_argument("--scan-github", type=str, help="Scan remote GitHub repositories directly (comma-separated, e.g. gohugoio/hugo,spf13/cobra) via zipball for zero-reingestion Type/Var/Directive extraction.")
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
            logger.info("0. Re-populating AST qualified_calls, call_type, and Go concurrency metadata from stored source code...")
            if not args.dry_run:
                rows = session.run("MATCH (f:Function) WHERE f.code IS NOT NULL RETURN f.id AS id, f.filepath AS fp, f.code AS code, f.repo AS repo").data()
                logger.info("Found %d Function nodes to scan for call expressions and metadata...", len(rows))
                batch_updates = []
                call_edge_updates = []
                for r in rows:
                    fp = r["fp"] or ""
                    code = r["code"] or ""
                    if fp.endswith(".go"):
                        ast = parse_go_ast(fp, code)
                    elif fp.endswith(".py"):
                        ast = parse_python_ast(fp, code)
                    else:
                        continue
                    qcalls_set = set()
                    for c in ast.get("calls", []):
                        if len(c) > 2 and c[2]:
                            qcalls_set.add(c[2])
                        if len(c) > 1 and c[1]:
                            qcalls_set.add(c[1])
                        if len(c) > 3 and c[3] != "SYNC":
                            call_edge_updates.append({
                                "caller_id": r["id"],
                                "callee_bare": c[1],
                                "call_type": c[3]
                            })
                    qcalls = sorted(list(qcalls_set))
                    funcs_list = ast.get("functions", [])
                    meta = funcs_list[0] if funcs_list else {}
                    batch_updates.append({
                        "id": r["id"],
                        "qcalls": qcalls,
                        "is_exported": meta.get("is_exported", False),
                        "is_pointer_receiver": meta.get("is_pointer_receiver", False),
                        "channels_sent": meta.get("channels_sent", []),
                        "channels_received": meta.get("channels_received", []),
                    })
                if batch_updates:
                    session.run("""
                        UNWIND $batch AS item
                        MATCH (f:Function {id: item.id})
                        SET f.qualified_calls = item.qcalls,
                            f.is_exported = coalesce(item.is_exported, false),
                            f.is_pointer_receiver = coalesce(item.is_pointer_receiver, false),
                            f.channels_sent = item.channels_sent,
                            f.channels_received = item.channels_received
                    """, batch=batch_updates)
                if call_edge_updates:
                    session.run("""
                        UNWIND $batch AS item
                        MATCH (caller:Function {id: item.caller_id})-[r:CALLS]->(callee:Function)
                        WHERE callee.name = item.callee_bare OR callee.id ENDS WITH ("::" + item.callee_bare)
                        SET r.call_type = item.call_type
                    """, batch=call_edge_updates)
                logger.info("✅ Re-populated AST metadata on %d Function nodes and updated %d call edges.", len(batch_updates), len(call_edge_updates))
            else:
                logger.info("[DRY RUN] Would scan and re-populate AST metadata.")

            logger.info("0.5 Re-populating Type metadata (field_names, method_names) and interface methods from stored code/fields/methods...")
            if not args.dry_run:
                type_rows = session.run("MATCH (t:Type) RETURN elementId(t) AS eid, t.name AS name, t.kind AS kind, t.code AS code, t.fields AS fields, t.methods AS methods, t.repo AS repo, t.filepath AS fp").data()
                logger.info("Found %d Type nodes to check...", len(type_rows))
                type_batch = []
                iface_methods_batch = []
                for tr in type_rows:
                    f_names = []
                    for f_str in (tr["fields"] or []):
                        try:
                            f_obj = json.loads(f_str) if isinstance(f_str, str) else f_str
                            if isinstance(f_obj, dict) and f_obj.get("name"):
                                f_names.append(f_obj["name"])
                        except Exception:
                            pass
                    m_names = []
                    raw_methods = []
                    for m_str in (tr["methods"] or []):
                        try:
                            m_obj = json.loads(m_str) if isinstance(m_str, str) else m_str
                            if isinstance(m_obj, dict):
                                raw_methods.append(m_obj)
                                if m_obj.get("name"):
                                    m_names.append(m_obj["name"])
                        except Exception:
                            pass
                    if tr["kind"] == "INTERFACE" and not raw_methods and tr["code"]:
                        fp_dummy = tr["fp"] if (tr["fp"] and tr["fp"].endswith(".go")) else "dummy.go"
                        ast_dummy = parse_go_ast(fp_dummy, tr["code"])
                        for td in ast_dummy.get("types", []):
                            if td.get("name") == tr["name"] and td.get("methods"):
                                raw_methods = td["methods"]
                                m_names = [m["name"] for m in raw_methods if m.get("name")]
                                break
                    type_batch.append({
                        "eid": tr["eid"],
                        "field_names": f_names,
                        "method_names": m_names,
                        "methods": [json.dumps(m) for m in raw_methods] if raw_methods else (tr["methods"] or []),
                    })
                    if tr["kind"] == "INTERFACE" and raw_methods:
                        for m in raw_methods:
                            m_name = m.get("name") or ""
                            m_sig = m.get("signature") or ""
                            if not m_name:
                                continue
                            scoped_method_name = f"{tr['name']}.{m_name}"
                            method_id = f"{tr['repo']}::{tr['fp']}::{scoped_method_name}"
                            iface_methods_batch.append({
                                "repo_full_name": tr["repo"],
                                "filepath": tr["fp"] or "<unknown>",
                                "type_id": f"{tr['repo']}::{tr['fp']}::{tr['name']}",
                                "method_id": method_id,
                                "method_name": scoped_method_name,
                                "signature": m_sig,
                                "is_exported": m_name[0].isupper() if m_name else False,
                            })
                if type_batch:
                    session.run("""
                        UNWIND $batch AS item
                        MATCH (t:Type)
                        WHERE elementId(t) = item.eid
                        SET t.field_names = item.field_names,
                            t.method_names = item.method_names,
                            t.methods = item.methods
                    """, batch=type_batch)
                    logger.info("✅ Re-populated field_names and method_names on %d Type nodes.", len(type_batch))
                if iface_methods_batch:
                    for im in iface_methods_batch:
                        if im["repo_full_name"] and im["filepath"]:
                            session.run(
                                CYPHER_INGEST_IFACE_METHOD,
                                repo_full_name=im["repo_full_name"],
                                filepath=im["filepath"],
                                type_id=im["type_id"],
                                method_id=im["method_id"],
                                method_name=im["method_name"],
                                signature=im["signature"],
                                is_exported=im["is_exported"],
                            )
                    logger.info("✅ Materialized %d interface methods as Function nodes.", len(iface_methods_batch))
                repos_res = session.run("MATCH (r:Repository) RETURN r.full_name AS repo").data()
                for rr in repos_res:
                    session.run(
                        """
                        MATCH (f:Function)
                        WHERE f.repo = $repo AND f.name CONTAINS '.'
                        WITH f, split(f.name, '.')[0] AS receiver_name
                        MATCH (t:Type {name: receiver_name, repo: $repo})
                        MERGE (t)-[:DECLARES_METHOD]->(f)
                        """,
                        repo=rr["repo"],
                    )
                logger.info("✅ Re-linked DECLARES_METHOD edges across all repositories.")
            else:
                logger.info("[DRY RUN] Would re-populate Type metadata and interface methods.")

            if args.scan_local and not args.dry_run:
                logger.info("Scanning local filesystem directory %s for zero-reingestion Type/Var extraction...", args.scan_local)
                local_path = Path(args.scan_local).resolve()
                if not local_path.exists() or not local_path.is_dir():
                    logger.error("Local directory %s does not exist or is not a directory.", args.scan_local)
                else:
                    repo_full_name = local_path.name
                    git_config = local_path / ".git" / "config"
                    if git_config.exists():
                        try:
                            with open(git_config, "r", encoding="utf8", errors="ignore") as gf:
                                for gline in gf:
                                    if "url = " in gline and "github.com" in gline:
                                        parts = gline.strip().split("github.com/")[-1].split(".git")[0]
                                        if parts:
                                            repo_full_name = parts
                                            break
                        except Exception:
                            pass
                    logger.info("Using repository name: %s for local scan.", repo_full_name)
                    api_key = os.environ.get("GEMINI_API_KEY", os.environ.get("GOOGLE_API_KEY", ""))
                    go_files = list(local_path.rglob("*.go"))
                    logger.info("Found %d .go files in %s", len(go_files), local_path)
                    file_items = []
                    for gf in go_files:
                        try:
                            rel_path = str(gf.relative_to(local_path))
                            code_str = gf.read_text(encoding="utf8", errors="replace")
                            file_items.append((rel_path, code_str))
                        except Exception as exc:
                            logger.debug("Error reading local file %s: %s", gf, exc)
                    _process_and_ingest_ast(session, repo_full_name, file_items, api_key)

            if args.scan_github and not args.dry_run:
                repos_to_scan = [r.strip() for r in args.scan_github.split(",") if r.strip()]
                api_key = os.environ.get("GEMINI_API_KEY", os.environ.get("GOOGLE_API_KEY", ""))
                token = os.environ.get("GITHUB_TOKEN", "")
                headers = {"Authorization": f"Bearer {token}"} if token else {}
                for repo_full_name in repos_to_scan:
                    logger.info("Downloading GitHub zipball for %s...", repo_full_name)
                    url = f"https://api.github.com/repos/{repo_full_name}/zipball"
                    try:
                        resp = requests.get(url, headers=headers, timeout=120)
                        if resp.status_code != 200:
                            logger.error("Failed to download zipball for %s: HTTP %d", repo_full_name, resp.status_code)
                            continue
                        zf = zipfile.ZipFile(io.BytesIO(resp.content))
                        go_files = [f for f in zf.namelist() if f.endswith(".go") and not f.endswith("_test.go") and not "/vendor/" in f]
                        logger.info("Found %d non-test .go files in %s zipball.", len(go_files), repo_full_name)
                        file_items = []
                        for f in go_files:
                            try:
                                rel_path = f.split("/", 1)[1] if "/" in f else f
                                code_str = zf.read(f).decode("utf8", errors="replace")
                                file_items.append((rel_path, code_str))
                            except Exception as exc:
                                logger.debug("Error reading zip file entry %s: %s", f, exc)
                        _process_and_ingest_ast(session, repo_full_name, file_items, api_key)
                    except Exception as exc:
                        logger.error("Error downloading/processing zipball for %s: %s", repo_full_name, exc)

            if args.dry_run:
                logger.info("[DRY RUN] Would clean up ambiguous CALLS edges where local callee exists or generic name across files.")
            else:
                logger.info("1. Cleaning up ambiguous CALLS edges where a file-local callee exists...")
                res = session.run(CYPHER_CLEANUP_AMBIGUOUS_CALLS)
                record = res.single()
                deleted = record["deleted_edges"] if record else 0
                logger.info("✅ Deleted %d ambiguous intra-repo CALLS edges.", deleted)

                logger.info("1b. Cleaning up false bare-name intra-repo CALLS edges for generic methods...")
                res_bare = session.run(CYPHER_CLEANUP_FALSE_BARE_CALLS)
                rec_bare = res_bare.single()
                deleted_bare = rec_bare["deleted_edges"] if rec_bare else 0
                logger.info("✅ Deleted %d false bare-name intra-repo CALLS edges.", deleted_bare)

                logger.info("1c. Cleaning up false cross-repo CALLS edges without USES_REPO relationship...")
                res_xrepo = session.run(CYPHER_CLEANUP_FALSE_XREPO_CALLS)
                rec_xrepo = res_xrepo.single()
                deleted_xrepo = rec_xrepo["deleted_edges"] if rec_xrepo else 0
                logger.info("✅ Deleted %d false cross-repo CALLS edges.", deleted_xrepo)

            logger.info("2. Creating Module-to-Repository representation links...")
            if not args.dry_run:
                res_mod = session.run(CYPHER_LINK_MODULE_TO_REPO)
                sum_mod = res_mod.consume()
                logger.info("✅ Module -> Repository edges created: %d", sum_mod.counters.relationships_created)
            else:
                logger.info("[DRY RUN] Would run CYPHER_LINK_MODULE_TO_REPO.")

            logger.info("3. Re-running cross-repo linking for all repository pairs in the graph...")
            repos = [r["full_name"] for r in session.run(CYPHER_GET_REPOS) if r["full_name"]]
            logger.info("Found %d repositories: %s", len(repos), repos)

            for parent in repos:
                for helper in repos:
                    if parent == helper:
                        continue
                    helper_name = helper.split("/")[-1]
                    logger.info("Checking cross-repo links: %s -> %s (prefix: %s)", parent, helper, helper_name)
                    if not args.dry_run:
                        res_use = session.run(
                            CYPHER_LINK_REPO_DEPENDENCY,
                            parent_repo=parent,
                            helper_repo=helper,
                            helper_repo_name=helper_name,
                        )
                        sum_use = res_use.consume()
                        res_calls = session.run(
                            CYPHER_LINK_XREPO_CALLS,
                            parent_repo=parent,
                            helper_repo=helper,
                            helper_prefix=helper_name,
                        )
                        sum_calls = res_calls.consume()
                        if sum_use.counters.relationships_created > 0 or sum_calls.counters.relationships_created > 0:
                            logger.info(
                                "   Linked %s -> %s: %d USES_REPO, %d cross-repo CALLS",
                                parent, helper,
                                sum_use.counters.relationships_created,
                                sum_calls.counters.relationships_created,
                            )
                    else:
                        logger.info("[DRY RUN] Would link %s -> %s", parent, helper)

    logger.info("🎉 Migration script completed successfully.")

if __name__ == "__main__":
    main()
