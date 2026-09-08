#!/usr/bin/env python3
"""
Migration script to clean up ambiguous CALLS edges and establish cross-repo links
without requiring a full repository re-ingestion. Optimized for Go structural patterns.

Usage:
    python migrate_calls_edges.py [--dry-run]
"""

import os
import sys
import argparse
import logging
import json
import zipfile
import io
import requests
from pathlib import Path
from dotenv import load_dotenv
from neo4j import GraphDatabase

# This script lives in maintenance/ — make the repo root importable so `ingest.*` resolves.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ingest.queries import (
    CYPHER_LINK_REPO_DEPENDENCY,
    CYPHER_LINK_MODULE_TO_REPO,
    CYPHER_LINK_XREPO_CALLS,
    CYPHER_INGEST_FUNCTION,
    CYPHER_INGEST_CALLS,
    CYPHER_INGEST_TYPE,
    CYPHER_INGEST_TYPE_EMBEDDING,
    CYPHER_INGEST_VARIABLE,
    CYPHER_INGEST_DIRECTIVE,
    CYPHER_INGEST_IFACE_METHOD,
    CYPHER_LINK_DECLARES_METHOD,
    CYPHER_MARK_FILE_SCANNED,
)
from ingest.ai_service import get_embeddings_batch
from ingest.parser import parse_go_ast, parse_python_ast

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("migrate_calls_edges")

# --- Optimized Cypher Queries ---

# Fix: Go files in the same directory share a package block. Don't drop intra-package calls.
CYPHER_CLEANUP_AMBIGUOUS_CALLS = """
MATCH (caller:Function)-[r:CALLS]->(callee:Function)
WHERE caller.repo = callee.repo
  AND caller.filepath <> callee.filepath
  AND split(caller.filepath, '/')[0..-1] <> split(callee.filepath, '/')[0..-1]
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
       callee.name IN ['New', 'Run', 'Execute', 'Close', 'Open', 'Init', 'String', 'Read', 'Write', 'Update', 'Start', 'Stop', 'Reset', 'Clear', 'Add', 'Remove', 'Delete', 'Get', 'Set', 'List', 'Find', 'Check', 'Verify', 'Validate', 'Parse', 'Format', 'Log', 'Main', 'Test']
    OR size(callee.name) < 4
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

# New: Go Structural Polymorphism Mapper (Implicit Interface Resolution)
CYPHER_COMPUTE_INTERFACE_IMPLEMENTATIONS = """
MATCH (t:Type {kind: "STRUCT"}), (i:Type {kind: "INTERFACE"})
WHERE t.repo = i.repo AND t.method_names IS NOT NULL AND i.method_names IS NOT NULL
  AND all(mName IN i.method_names WHERE mName IN t.method_names)
MERGE (t)-[r:IMPLEMENTS]->(i)
RETURN count(r) AS structural_bindings
"""

CYPHER_GET_REPOS = """
MATCH (r:Repository)
RETURN r.full_name AS full_name
"""


def _process_and_ingest_ast(session, repo_full_name: str, file_items: list, api_key: str):
    logger.info("Parsing AST for %d files in %s...", len(file_items), repo_full_name)
    all_types, all_vars, all_directives, all_funcs, all_calls = [], [], [], [], []

    for rel_path, code_str in file_items:
        try:
            if rel_path.endswith(".go"):
                ast_data = parse_go_ast(rel_path, code_str)
                for t in ast_data.get("types", []):
                    all_types.append((t, rel_path))
                for v in ast_data.get("variables", []):
                    all_vars.append((v, rel_path))
                for d in ast_data.get("directives", []):
                    all_directives.append((d, rel_path))
                test_functions = {f["name"] for f in ast_data.get("functions", []) if f.get("is_test")}
                for f in ast_data.get("functions", []):
                    all_funcs.append((f, rel_path))
                for c in ast_data.get("calls", []):
                    if c[0] in test_functions:
                        continue
                    all_calls.append((c, rel_path))
        except Exception as exc:
            logger.debug("Error parsing %s: %s", rel_path, exc)

    logger.info(
        "Extracted %d functions, %d types, %d variables. Processing embeddings...",
        len(all_funcs), len(all_types), len(all_vars),
    )

    func_embeds = []
    if all_funcs and api_key:
        func_texts = [f"{f['name']}\n{f.get('code', '')}" for f, _ in all_funcs]
        for i in range(0, len(func_texts), 64):
            func_embeds.extend(get_embeddings_batch(func_texts[i : i + 64], api_key))
    else:
        func_embeds = [None] * len(all_funcs)

    for (f, rel_path), emb in zip(all_funcs, func_embeds):
        func_id = f"{repo_full_name}::{f['id']}"
        qcalls = sorted(list({
            c[2] or c[1] for c, c_path in all_calls
            if c_path == rel_path and c[0] == f["name"] and (len(c) > 1 and c[1])
        }))
        session.run(
            CYPHER_INGEST_FUNCTION,
            repo_full_name=repo_full_name, filepath=rel_path, func_id=func_id,
            func_name=f["name"], func_code=f.get("code", ""), embedding=emb or None, qualified_calls=qcalls,
            is_exported=f.get("is_exported", False), is_pointer_receiver=f.get("is_pointer_receiver", False),
            channels_sent=f.get("channels_sent", []), channels_received=f.get("channels_received", []),
            return_types=f.get("return_types", []), is_test=f.get("is_test", False),
            accepts_context=f.get("accepts_context", False), lock_sequence=f.get("lock_sequence", []),
            propagated_errors=f.get("propagated_errors", []),
        )

    for call, rel_path in all_calls:
        session.run(
            CYPHER_INGEST_CALLS,
            caller_id=f"{repo_full_name}::{rel_path}::{call[0]}",
            callee_name=call[1],
            callee_qualified=call[2] if len(call) > 2 else None,
            caller_filepath=rel_path,
            repo_full_name=repo_full_name,
            call_type=call[3] if len(call) > 3 else "SYNC",
            propagates_context=call[4] if len(call) > 4 else False,
            creates_cancellation_scope=call[5] if len(call) > 5 else False,
        )

    type_embeds = []
    if all_types and api_key:
        type_texts = [f"{t['name']}\n{t.get('code', '')}" for t, _ in all_types]
        for i in range(0, len(type_texts), 64):
            type_embeds.extend(get_embeddings_batch(type_texts[i : i + 64], api_key))
    else:
        type_embeds = [None] * len(all_types)

    for (t, rel_path), emb in zip(all_types, type_embeds):
        tid = f"{repo_full_name}::{t['id']}"
        raw_fields = t.get("fields", [])
        raw_methods = t.get("methods", [])
        
        session.run(
            CYPHER_INGEST_TYPE,
            repo_full_name=repo_full_name,
            filepath=rel_path,
            type_id=tid,
            type_name=t["name"],
            kind=t["kind"],
            code=t.get("code", ""),
            fields=[json.dumps(f) for f in raw_fields],
            field_names=[f["name"] for f in raw_fields if isinstance(f, dict) and f.get("name")],
            methods=[json.dumps(m) for m in raw_methods],
            method_names=[m["name"] for m in raw_methods if isinstance(m, dict) and m.get("name")],
            tags=t.get("tags", []),
            embedding=emb or None,
            is_exported=t.get("is_exported", False),
            tag_mappings=json.dumps(t.get("tag_mappings", {})),
            type_parameters=json.dumps(t.get("type_parameters", [])),
        )
        
        for embed_name in t.get("embedded_types", []):
            session.run(CYPHER_INGEST_TYPE_EMBEDDING, outer_id=tid, repo_full_name=repo_full_name, inner_name=embed_name)

        if t.get("kind") == "INTERFACE":
            for method in raw_methods:
                m_name = method.get("name") or ""
                if not m_name:
                    continue
                scoped_name = f"{t['name']}.{m_name}"
                session.run(
                    CYPHER_INGEST_IFACE_METHOD,
                    repo_full_name=repo_full_name,
                    filepath=rel_path,
                    type_id=tid,
                    method_id=f"{repo_full_name}::{rel_path}::{scoped_name}",
                    method_name=scoped_name,
                    signature=method.get("signature") or "",
                    is_exported=m_name[0].isupper(),
                )

    for rel_path, _ in file_items:
        if rel_path.endswith(".go"):
            session.run(CYPHER_MARK_FILE_SCANNED, filepath=rel_path, repo_full_name=repo_full_name)

    logger.info(
        "Ingested %d functions, %d calls, %d types, %d variables, %d directives for %s.",
        len(all_funcs), len(all_calls), len(all_types), len(all_vars), len(all_directives), repo_full_name,
    )


def main():
    parser = argparse.ArgumentParser(description="Clean up ambiguous Go CALLS edges and resolve type ecosystems.")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without modifying the database.")
    parser.add_argument("--scan-local", type=str, help="Scan a local directory for schema elements.")
    parser.add_argument("--scan-github", type=str, help="Scan remote repositories via GitHub zipball stream.")
    args = parser.parse_args()

    _env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(_env_path if _env_path.exists() else None)

    uri = os.environ.get("NEO4J_URI", "neo4j://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "password123")

    logger.info("Connecting to Neo4j instance at %s...", uri)
    with GraphDatabase.driver(uri, auth=(user, password)) as driver:
        with driver.session() as session:
            # --- Pass 0: AST Scan Local/GitHub ---
            if not args.dry_run:
                if args.scan_local:
                    logger.info("Scanning local directory %s...", args.scan_local)
                    local_path = Path(args.scan_local)
                    if not local_path.exists():
                        logger.error("Local path %s does not exist.", local_path)
                        sys.exit(1)
                    
                    file_items = []
                    for p in local_path.rglob("*.go"):
                        if p.is_file():
                            try:
                                rel_p = str(p.relative_to(local_path))
                                file_items.append((rel_p, p.read_text(errors="ignore")))
                            except Exception as e:
                                logger.debug("Failed to read local file %s: %s", p, e)
                    
                    repo_name = os.environ.get("TARGET_REPO", "local/repo")
                    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY") or ""
                    _process_and_ingest_ast(session, repo_name, file_items, api_key)
                    logger.info("Local scan and ingestion of AST completed.")

                if args.scan_github:
                    logger.info("Scanning remote GitHub repository %s...", args.scan_github)
                    github_token = os.environ.get("GITHUB_TOKEN", "")
                    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY") or ""
                    headers = {}
                    if github_token:
                        headers["Authorization"] = f"token {github_token}"
                    url = f"https://api.github.com/repos/{args.scan_github}/zipball"
                    logger.info("Downloading zipball from %s...", url)
                    try:
                        r = requests.get(url, headers=headers)
                        r.raise_for_status()
                        z = zipfile.ZipFile(io.BytesIO(r.content))
                        file_items = []
                        for name in z.namelist():
                            parts = name.split("/", 1)
                            if len(parts) > 1 and parts[1].endswith(".go"):
                                try:
                                    code_str = z.read(name).decode("utf-8", errors="ignore")
                                    file_items.append((parts[1], code_str))
                                except Exception as e:
                                    logger.debug("Failed to decode zip entry %s: %s", name, e)
                        _process_and_ingest_ast(session, args.scan_github, file_items, api_key)
                        logger.info("GitHub scan and ingestion of AST completed.")
                    except Exception as exc:
                        logger.error("Failed to scan GitHub repo %s: %s", args.scan_github, exc)
                        sys.exit(1)

            # --- Pass 1: Structural Function Re-scan ---
            if not args.dry_run:
                logger.info("Syncing AST functional definitions from stored text nodes...")
                rows = session.run("MATCH (f:Function) WHERE f.code IS NOT NULL RETURN f.id AS id, f.filepath AS fp, f.code AS code, f.repo AS repo").data()
                batch_updates, call_edge_updates = [], []
                
                for r in rows:
                    fp = r["fp"] or ""
                    if not fp.endswith(".go"):
                        continue
                    ast = parse_go_ast(fp, r["code"] or "")
                    
                    qcalls_set = {c[2] for c in ast.get("calls", []) if len(c) > 2 and c[2]} | \
                                 {c[1] for c in ast.get("calls", []) if len(c) > 1 and c[1]}
                    
                    for c in ast.get("calls", []):
                        if len(c) > 3 and c[3] != "SYNC":
                            call_edge_updates.append({"caller_id": r["id"], "callee_bare": c[1], "call_type": c[3]})
                            
                    funcs_list = ast.get("functions", [])
                    meta = funcs_list[0] if funcs_list else {}
                    batch_updates.append({
                        "id": r["id"],
                        "qcalls": sorted(list(qcalls_set)),
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
            
            # --- Pass 2: Type Engine & Implicit Interfaces ---
            if not args.dry_run:
                logger.info("Computing implicit structural polymorph links (Go interfaces)...")
                iface_res = session.run(CYPHER_COMPUTE_INTERFACE_IMPLEMENTATIONS).single()
                logger.info("✅ Established %d dynamic IMPLEMENTS paths.", iface_res["structural_bindings"] if iface_res else 0)

            # --- Pass 3: Safe Go-aware Edge Pruning ---
            if not args.dry_run:
                logger.info("Executing isolated graph pruning passes...")
                del_ambig = session.run(CYPHER_CLEANUP_AMBIGUOUS_CALLS).single()["deleted_edges"]
                del_bare = session.run(CYPHER_CLEANUP_FALSE_BARE_CALLS).single()["deleted_edges"]
                logger.info("✅ Graph edge refinement summary: Removed %d ambiguous and %d bare invalid CALLS.", del_ambig, del_bare)
            else:
                logger.info("[DRY RUN] Skipping edge validation queries.")

            # --- Pass 4: System Multi-Repo Linking ---
            if not args.dry_run:
                session.run(CYPHER_LINK_MODULE_TO_REPO)
                repos = [r["full_name"] for r in session.run(CYPHER_GET_REPOS) if r["full_name"]]
                for parent in repos:
                    for helper in repos:
                        if parent == helper:
                            continue
                        h_name = helper.split("/")[-1]
                        session.run(CYPHER_LINK_REPO_DEPENDENCY, parent_repo=parent, helper_repo=helper, helper_repo_name=h_name)
                        session.run(CYPHER_LINK_XREPO_CALLS, parent_repo=parent, helper_repo=helper, helper_prefix=h_name)

    logger.info("🎉 Go Migration Engine routine completed successfully.")

if __name__ == "__main__":
    main()