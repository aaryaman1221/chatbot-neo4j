#!/usr/bin/env python3
# =============================================================================
# patch_function_names.py — Zero-Token In-Place Graph Restoration
# =============================================================================
#
# PURPOSE:
#   Restores chopped Go function names and code blocks in an existing Neo4j knowledge
#   graph without re-running embedding generation or LLM summaries.
#
#   Why this is needed:
#   An earlier UTF-8 byte-slicing bug in tree-sitter Go AST parsing caused function
#   names in files with multi-byte headers (like copyright symbols) to be stripped of
#   their first 1-2 characters (e.g., 'newServerCommand' -> 'wServerCommand()').
#
#   How this works:
#   1. Fetches the repository tarball from GitHub in a single HTTP request (0 LLM tokens).
#   2. Parses all Go files in memory using the corrected AST parser.
#   3. Matches existing database function nodes against the cleanly parsed AST names.
#   4. Executes in-place Cypher updates to fix fn.name, fn.id, and fn.code while
#      preserving all existing embeddings, call graphs, and commit blame edges!
#
# USAGE:
#   python3 patch_function_names.py
#
# =============================================================================

import os
import sys
import io
import time
import tarfile
import requests
from dotenv import load_dotenv
from neo4j import GraphDatabase
from tqdm import tqdm

from ingest.config import logger, GITHUB_API_BASE
from ingest.parser import parse_go_ast, parse_python_ast

CYPHER_GET_SOURCE_FILES = """
MATCH (fn:Function)
WHERE fn.repo = $repo AND (fn.filepath ENDS WITH '.go' OR fn.filepath ENDS WITH '.py')
RETURN DISTINCT fn.filepath AS filepath
"""

CYPHER_GET_FILE_FUNCS = """
MATCH (fn:Function)
WHERE fn.repo = $repo AND fn.filepath = $filepath
RETURN fn.name AS old_name, fn.id AS old_id, fn.code AS code
"""

CYPHER_PATCH_FUNCTION = """
MATCH (fn:Function {repo: $repo, filepath: $filepath, name: $old_name})
OPTIONAL MATCH (existing:Function {id: $new_id})
WITH fn, existing
SET fn.name = $correct_name,
    fn.code = $correct_code,
    fn.id   = CASE WHEN existing IS NULL THEN $new_id ELSE $new_id + '_' + toString(id(fn)) END
RETURN count(fn) AS updated
"""


def patch_graph():
    from pathlib import Path
    _env_path = Path(__file__).resolve().parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
    else:
        load_dotenv()
    
    github_token = os.environ.get("GITHUB_TOKEN", "")
    neo4j_uri    = os.environ.get("NEO4J_URI", "neo4j://localhost:7687")
    neo4j_user   = os.environ.get("NEO4J_USER", "neo4j")
    neo4j_pass   = os.environ.get("NEO4J_PASSWORD", "")
    target_repo  = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("TARGET_REPO", "gohugoio/hugo")

    if not neo4j_pass and not os.environ.get("NEO4J_NO_AUTH"):
        print("[WARN] NEO4J_PASSWORD not set. Using empty password or check your .env file.")

    print(f"[*] Connecting to Neo4j at {neo4j_uri} for repo '{target_repo}'...")
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_pass))

    with driver.session() as session:
        file_rows = session.run(CYPHER_GET_SOURCE_FILES, repo=target_repo).data()
    
    db_paths = [r["filepath"] for r in file_rows]
    print(f"[*] Found {len(db_paths)} distinct Go/Python files with functions in graph.")

    if not db_paths:
        print("[!] No Go or Python files found to patch.")
        driver.close()
        return

    print(f"[*] Downloading tarball for '{target_repo}' from GitHub API...")
    url = f"{GITHUB_API_BASE}/repos/{target_repo}/tarball"
    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"

    start_time = time.time()
    resp = requests.get(url, headers=headers, timeout=60)
    resp.raise_for_status()
    print(f"[*] Download complete in {time.time() - start_time:.1f}s ({len(resp.content)/1024/1024:.2f} MB).")

    print("[*] Extracting source files from tarball in memory...")
    tar_files = {}
    with tarfile.open(fileobj=io.BytesIO(resp.content), mode="r:gz") as tar:
        for member in tar.getmembers():
            if not (member.name.endswith(".go") or member.name.endswith(".py")) or member.isdir():
                continue
            rel_path = member.name.split("/", 1)[-1]
            if rel_path in db_paths:
                raw_bytes = tar.extractfile(member).read()
                tar_files[rel_path] = raw_bytes.decode("utf-8", errors="replace")

    print(f"[*] Loaded {len(tar_files)} matching source files from archive.")

    total_patched = 0
    total_checked = 0

    with driver.session() as session:
        with tqdm(db_paths, desc="Patching functions", unit="file", ncols=88) as pbar:
            for filepath in pbar:
                pbar.set_postfix_str(filepath[-40:] if len(filepath) > 40 else filepath, refresh=False)
                source_code = tar_files.get(filepath)
                if not source_code:
                    continue

                if filepath.endswith(".go"):
                    ast_data = parse_go_ast(filepath, source_code)
                else:
                    ast_data = parse_python_ast(filepath, source_code)
                correct_funcs = {f["name"]: f for f in ast_data.get("functions", [])}

                db_funcs = session.run(CYPHER_GET_FILE_FUNCS, repo=target_repo, filepath=filepath).data()
                
                for r in db_funcs:
                    total_checked += 1
                    old_name = r["old_name"]
                    old_code = r.get("code") or ""
                    clean_old = old_name.split("(")[0]

                    for corr_name, corr_data in correct_funcs.items():
                        # Match if name matches or if old bare name is the suffix of scoped name (e.g. Close -> Client.Close)
                        is_match = (corr_name == old_name) or (corr_name.split(".")[-1] == clean_old) or (clean_old and corr_name.endswith(clean_old) and len(clean_old) >= 2)
                        if is_match:
                            # Verify code match if available to distinguish identically named methods in same file
                            if old_code and corr_data.get("code") and (old_code[:40] != corr_data["code"][:40]) and (old_name != corr_name):
                                continue
                            if old_name != corr_name or old_code != corr_data.get("code", ""):
                                new_id = f"{target_repo}::{filepath}::{corr_name}"
                                session.run(
                                    CYPHER_PATCH_FUNCTION,
                                    repo=target_repo,
                                    filepath=filepath,
                                    old_name=old_name,
                                    correct_name=corr_name,
                                    new_id=new_id,
                                    correct_code=corr_data.get("code", ""),
                                )
                                total_patched += 1
                            break

    driver.close()
    print(f"\n[DONE] Checked {total_checked} functions. Successfully patched {total_patched} function names and IDs in-place!")
    print("[*] All vector embeddings, call graphs, and commit blame relationships were preserved with zero token cost.")


if __name__ == "__main__":
    patch_graph()
