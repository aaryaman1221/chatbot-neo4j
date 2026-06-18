#!/usr/bin/env python3
# =============================================================================
# backend_ingest.py — CLI Data Pipeline for GitHub → Neo4j GraphRAG Bootstrap
# =============================================================================
#
# PURPOSE:
#   A pure Python command-line tool to bootstrap a GitHub repository into a
#   Neo4j knowledge graph.  Zero Streamlit dependencies.
#
# USAGE:
#   python backend_ingest.py
#
# ENVIRONMENT VARIABLES (create a .env file or export them):
#   GITHUB_TOKEN      — GitHub Personal Access Token (classic or fine-grained)
#   GOOGLE_API_KEY    — Google Gemini API key (from aistudio.google.com)
#   NEO4J_URI         — Neo4j Bolt/Neo4j URI  (default: neo4j://localhost:7687)
#   NEO4J_USER        — Neo4j username         (default: neo4j)
#   NEO4J_PASSWORD    — Neo4j password
#   TARGET_REPO       — "owner/repo" to ingest  (default: neo4j/neo4j-graphrag-python)
#   MAX_COMMITS       — Maximum commits to backfill (default: 200)
#
# DOCKER (quick Neo4j setup):
#   docker run --name neo4j-graphrag \
#     -p 7474:7474 -p 7687:7687 \
#     -e NEO4J_AUTH=neo4j/password123 \
#     -e NEO4J_PLUGINS='["apoc"]' \
#     -e NEO4J_apoc_export_file_enabled=true \
#     -e NEO4J_apoc_import_file_enabled=true \
#     -e NEO4J_dbms_security_procedures_unrestricted=apoc.* \
#     -e NEO4J_dbms_memory_heap_initial__size=512m \
#     -e NEO4J_dbms_memory_heap_max__size=1G \
#     neo4j:5.20
#
# REQUIREMENTS:
#   pip install PyGithub neo4j google-genai tqdm requests python-dotenv
#
# =============================================================================

import base64
import json
import logging
import os
import re
import time
import ast
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests
from dotenv import load_dotenv
from github import Auth, Github, GithubException
from neo4j import GraphDatabase
from tqdm import tqdm


# ── Google Gemini (google-genai SDK) ─────────────────────────────────────────
try:
    from google import genai as google_genai
    from google.genai import types as genai_types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    print("[WARN] google-genai not installed — LLM summaries will use heuristics.")

# =============================================================================
#  LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("backend_ingest")

# =============================================================================
#  CONFIGURATION CONSTANTS
# =============================================================================

GITHUB_API_BASE = "https://api.github.com"

# File/directory filters
SOURCE_EXTENSIONS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".java", ".rs",
    ".rb", ".php", ".c", ".cpp", ".h", ".hpp", ".cs", ".swift",
    ".kt", ".scala", ".r", ".R", ".vue", ".svelte",
}

SOURCE_FILENAMES = {"go.mod", "go.work"}

ENTRY_POINT_NAMES = {
    "main", "index", "app", "server", "__main__", "manage",
    "wsgi", "asgi", "cli", "entrypoint",
}

UTILITY_DIRS = {
    "utils", "util", "lib", "libs", "common", "shared",
    "helpers", "helper", "core", "pkg", "internal",
}

IGNORED_DIRECTORIES = {
    "node_modules", "venv", ".venv", "env", "dist", "build", ".next", ".git",
}

IGNORED_FILENAMES = {
    ".ds_store", "cargo.lock", "gemfile.lock", "package-lock.json",
    "poetry.lock", "pnpm-lock.yaml", "yarn.lock",
}

IGNORED_SUFFIXES = (
    ".bin", ".dll", ".exe", ".gif", ".gz", ".ico", ".jpeg", ".jpg",
    ".lock", ".mp3", ".mp4", ".pdf", ".png", ".so", ".svg", ".tar",
    ".tgz", ".ttf", ".woff", ".woff2", ".zip",
)

# Neo4j batch flush size
_GRAPH_FLUSH_BATCH = 50


# =============================================================================
#  NEO4J CYPHER TEMPLATES  (shared schema with frontend_chat.py)
# =============================================================================

CYPHER_CREATE_REPO = """
MERGE (repo:Repository {full_name: $repo_full_name})
  ON CREATE SET repo.name = $repo_name, repo.url = $repo_url
"""

CYPHER_INGEST_FUNCTION = """
MERGE (file:File {path: $filepath})
MERGE (func:Function {id: $func_id})
  ON CREATE SET 
    func.name = $func_name, 
    func.filepath = $filepath, 
    func.code = $func_code
MERGE (file)-[:DECLARES]->(func)
"""

CYPHER_INGEST_CALLS = """
MERGE (caller:Function {id: $caller_id})
MERGE (callee:Function {name: $callee_name})
MERGE (caller)-[:CALLS]->(callee)
"""

CYPHER_MODIFIED_FUNCTION = """
MERGE (commit:Commit {sha: $commit_sha})
MERGE (func:Function {id: $func_id})
MERGE (commit)-[:MODIFIED]->(func)
"""

CYPHER_INGEST_COMMIT = """
MERGE (repo:Repository {full_name: $repo_full_name})
MERGE (author:User {login: $actor_login})
MERGE (commit:Commit {sha: $commit_sha})
  ON CREATE SET
    commit.timestamp    = $committed_at,
    commit.summary_text = $summary_text,
    commit.diff_text    = $diff_text,
    commit.message      = $commit_message,
    commit.url          = $commit_url
MERGE (author)-[:AUTHORED]->(commit)
MERGE (commit)-[:BELONGS_TO]->(repo)
"""

CYPHER_INGEST_FILE = """
MERGE (repo:Repository {full_name: $repo_full_name})
MERGE (file:File {path: $filepath})
  ON CREATE SET file.repo = $repo_full_name
MERGE (commit:Commit {sha: $commit_sha})
MERGE (commit)-[:MODIFIED]->(file)
"""

CYPHER_INGEST_DEPENDENCY = """
MERGE (file:File {path: $filepath})
MERGE (module:Module {name: $target_module})
MERGE (file)-[:DEPENDS_ON]->(module)
"""

CYPHER_TREE_FILE = """
MERGE (repo:Repository {full_name: $repo_full_name})
MERGE (file:File {path: $child_path})
  ON CREATE SET file.repo = $repo_full_name,
                file.entry_point = $entry_point
WITH file
OPTIONAL MATCH (parent:Directory {path: $parent_path})
FOREACH (_ IN CASE WHEN parent IS NOT NULL THEN [1] ELSE [] END |
  MERGE (parent)-[:CONTAINS]->(file)
)
"""

CYPHER_TREE_DIR = """
MERGE (repo:Repository {full_name: $repo_full_name})
MERGE (dir:Directory {path: $child_path})
  ON CREATE SET dir.repo = $repo_full_name,
                dir.utility = $utility
WITH dir
OPTIONAL MATCH (parent:Directory {path: $parent_path})
FOREACH (_ IN CASE WHEN parent IS NOT NULL THEN [1] ELSE [] END |
  MERGE (parent)-[:CONTAINS]->(dir)
)
"""

# Vector index on Issue.embedding (3 072 dims — gemini-embedding-2-preview)
CYPHER_VECTOR_INDEX = """
CREATE VECTOR INDEX issue_embeddings IF NOT EXISTS
FOR (i:Issue) ON (i.embedding)
OPTIONS {
  indexConfig: {
    `vector.dimensions`: 3072,
    `vector.similarity_function`: 'cosine'
  }
}
"""

# Full-text index for commit-level RAG retrieval
CYPHER_FULLTEXT_INDEX = """
CREATE FULLTEXT INDEX commit_summaries IF NOT EXISTS
FOR (c:Commit) ON EACH [c.summary_text, c.diff_text, c.message]
"""

# Bootstrap status tracking node
CYPHER_UPSERT_STATUS = """
MERGE (bs:BootstrapStatus {repo: $repo_full_name})
SET bs.status            = $status,
    bs.detail            = $detail,
    bs.commits_processed = $commits_processed,
    bs.files_scanned     = $files_scanned,
    bs.updated_at        = $updated_at
"""


# =============================================================================
#  FILE FILTER UTILITIES
# =============================================================================

def _is_noise_file(filename: str) -> bool:
    """Return True if this file should be skipped during ingestion."""
    lower_path = filename.lower().replace("\\", "/")
    path_parts = set(lower_path.split("/"))
    if path_parts.intersection(IGNORED_DIRECTORIES):
        return True
    name = lower_path.rsplit("/", 1)[-1]
    if name in IGNORED_FILENAMES:
        return True
    return name.endswith(IGNORED_SUFFIXES)


# =============================================================================
#  DEPENDENCY EXTRACTION UTILITIES
# =============================================================================

def parse_python_ast(filepath: str, source_code: str) -> dict:
    """
    Parses a Python file to extract functions, internal calls, and raw source code.
    """
    if not filepath.endswith(".py") or not source_code:
        return {"functions": [], "calls": []}

    functions = []
    calls = []
    
    try:
        tree = ast.parse(source_code)
        
        # 1. Extract Function Boundaries and Code
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Safely extract the raw code snippet
                func_code = ast.get_source_segment(source_code, node) or ""
                
                functions.append({
                    "name": node.name,
                    "id": f"{filepath}::{node.name}",
                    "start": node.lineno,
                    "end": node.end_lineno,
                    "code": func_code
                })
                
                # 2. Extract Function Calls
                for child in ast.walk(node):
                    if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                        calls.append((node.name, child.func.id))
                        
    except Exception:
        # Broad catch to ignore any files with bad syntax that can't be parsed
        pass
        
    return {"functions": functions, "calls": calls}

def _append_dependency(dependencies: list, filepath: str, target_module: str):
    """Append a dependency edge, deduplicating by (file, module) pair."""
    if not target_module:
        return
    edge = (filepath, "DEPENDS_ON", target_module)
    if edge not in dependencies:
        dependencies.append(edge)


def _go_file_kind(filepath: str) -> Optional[str]:
    name = filepath.lower().replace("\\", "/").rsplit("/", 1)[-1]
    if name == "go.mod":    return "go_mod"
    if name == "go.work":   return "go_work"
    if name.endswith(".go"): return "go_source"
    return None


def _extract_go_dependencies(filepath: str, lines: list, patch_mode: bool = False) -> list:
    """Parse Go import / require / use / replace statements."""
    dependencies = []
    in_import_block = in_require_block = in_use_block = in_replace_block = False

    for raw_line in lines:
        line = raw_line.rstrip("\n")
        if patch_mode:
            if not line or line[0] not in {"+", " "}:
                continue
            line = line[1:].lstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            continue

        if stripped.startswith("import ("):
            in_import_block = True; continue
        if in_import_block:
            if stripped.startswith(")"):
                in_import_block = False; continue
            m = re.search(r'["`]\s*([^"`]+?)\s*["`]', stripped)
            if m: _append_dependency(dependencies, filepath, m.group(1))
            continue
        if stripped.startswith("import "):
            m = re.search(r'["`]\s*([^"`]+?)\s*["`]', stripped)
            if m: _append_dependency(dependencies, filepath, m.group(1))
            continue

        if stripped.startswith("require ("):
            in_require_block = True; continue
        if stripped.startswith("use ("):
            in_use_block = True; continue
        if stripped.startswith("replace ("):
            in_replace_block = True; continue

        if in_require_block:
            if stripped.startswith(")"):
                in_require_block = False; continue
            parts = stripped.split()
            if len(parts) >= 2: _append_dependency(dependencies, filepath, parts[0])
            continue

        if in_use_block:
            if stripped.startswith(")"):
                in_use_block = False; continue
            _append_dependency(dependencies, filepath, stripped)
            continue

        if in_replace_block:
            if stripped.startswith(")"):
                in_replace_block = False; continue
            if "=>" in stripped:
                left, right = [p.strip() for p in stripped.split("=>", 1)]
                _append_dependency(dependencies, filepath, left.split()[0] if left else "")
                _append_dependency(dependencies, filepath, right.split()[0] if right else "")
            continue

        # single-line forms
        if stripped.startswith("require "):
            parts = stripped.split()
            if len(parts) >= 3: _append_dependency(dependencies, filepath, parts[1])
        if stripped.startswith("use "):
            parts = stripped.split(None, 1)
            if len(parts) == 2: _append_dependency(dependencies, filepath, parts[1].strip())
        if stripped.startswith("replace ") and "=>" in stripped:
            body = stripped[len("replace "):].strip()
            left, right = [p.strip() for p in body.split("=>", 1)]
            _append_dependency(dependencies, filepath, left.split()[0] if left else "")
            _append_dependency(dependencies, filepath, right.split()[0] if right else "")

    return dependencies


def _extract_generic_dependencies(filepath: str, lines: list, patch_mode: bool = False) -> list:
    """Parse Python / JS / TS / Ruby / C++ / Rust import statements."""
    dependencies = []
    patterns = [
        r"^\s*from\s+([a-zA-Z0-9_./@+-]+)\s+import",
        r"^\s*import\s+([a-zA-Z0-9_./@+-]+)",
        r"from\s+['\"]([^'\"]+)['\"]",
        r"require\(['\"]([^'\"]+)['\"]\)",
        r"#include\s*[<\"]([^>\"]+)[>\"]",
        r"use\s+([a-zA-Z0-9_:]+)",
    ]
    for raw_line in lines:
        line = raw_line.rstrip("\n")
        if patch_mode:
            if not line or line[0] not in {"+", " "}:
                continue
            line = line[1:].lstrip()
        stripped = line.strip()
        if not stripped or (stripped.startswith("#") and not stripped.startswith("#include")):
            continue
        for pattern in patterns:
            m = re.search(pattern, stripped)
            if m:
                _append_dependency(dependencies, filepath, m.group(1))
    return dependencies


def extract_imports_from_source(filepath: str, source_code: str) -> list:
    """Extract import/require statements from full source text (not a diff patch)."""
    kind = _go_file_kind(filepath)
    lines = source_code.split("\n")
    if kind in {"go_mod", "go_work", "go_source"}:
        return _extract_go_dependencies(filepath, lines, patch_mode=False)
    return _extract_generic_dependencies(filepath, lines, patch_mode=False)


def extract_file_dependencies(compact_files: list) -> list:
    """Extract dependencies from diff patches of a list of changed files."""
    dependencies = []
    for item in compact_files:
        source_file = item.get("filename")
        patch = item.get("patch", "")
        if not patch or not source_file:
            continue
        kind = _go_file_kind(source_file)
        lines = patch.split("\n")
        if kind in {"go_mod", "go_work", "go_source"}:
            new_deps = _extract_go_dependencies(source_file, lines, patch_mode=True)
        else:
            new_deps = _extract_generic_dependencies(source_file, lines, patch_mode=True)
        for dep in new_deps:
            if dep not in dependencies:
                dependencies.append(dep)
    return dependencies


# =============================================================================
#  DIFF UTILITIES
# =============================================================================

def _truncate(text: str, limit: int = 5000) -> str:
    if not text:
        return ""
    return text if len(text) <= limit else f"{text[:limit]}\n... [truncated]"


def build_compact_diff(files: list, max_files: int = 12) -> list:
    """Filter noise files, truncate patches, and rank by change size."""
    filtered = []
    for item in files:
        filename = item.get("filename") or item.get("path") or "unknown"
        if _is_noise_file(filename):
            continue
        patch = item.get("patch")
        if not patch:
            continue
        filtered.append({
            "filename": filename,
            "status":    item.get("status", "modified"),
            "additions": item.get("additions", 0),
            "deletions": item.get("deletions", 0),
            "changes":   item.get("changes", 0),
            "patch":     _truncate(patch, 4000),
        })
    filtered.sort(key=lambda x: x.get("changes", 0), reverse=True)
    return filtered[:max_files]


def render_diff_text(files: list) -> str:
    """Convert compact diff list into a single human-readable block."""
    sections = []
    for item in files:
        sections.append("\n".join([
            f"File: {item['filename']}",
            f"Status: {item.get('status', 'modified')}",
            f"Additions: {item.get('additions', 0)}",
            f"Deletions: {item.get('deletions', 0)}",
            "Patch:",
            item.get("patch", ""),
        ]))
    return "\n\n".join(sections)


# =============================================================================
#  LLM SUMMARIZATION
# =============================================================================

def _heuristic_summary(repo_full_name: str, compact_files: list, commit_msg: str) -> str:
    """Fallback summary when no LLM is available."""
    if not compact_files:
        return f"Commit in {repo_full_name}: {commit_msg[:200]}"
    parts = [
        f"{item['filename']} ({item.get('status','modified')}, "
        f"+{item.get('additions',0)}/-{item.get('deletions',0)})"
        for item in compact_files[:5]
    ]
    extra = f" and {len(compact_files) - 5} more" if len(compact_files) > 5 else ""
    return (
        f"Commit in {repo_full_name}: {commit_msg[:100]}\n"
        f"Key files: {', '.join(parts)}{extra}."
    )


def summarize_with_llm(
    repo_full_name: str,
    actor_login: str,
    commit_msg: str,
    compact_files: list,
    raw_diff: str,
    google_api_key: Optional[str] = None,
) -> str:
    """
    Generate an LLM summary of a commit diff using google-genai.

    Falls back to a REST call, then to a heuristic summary if both fail.
    """
    if not google_api_key:
        return _heuristic_summary(repo_full_name, compact_files, commit_msg)

    file_overview = "\n".join(
        f"- {item['filename']} ({item.get('status','modified')} "
        f"| +{item.get('additions',0)} / -{item.get('deletions',0)})"
        for item in compact_files
    ) or "- No text patches available."

    system_instruction = (
        "You are an expert senior developer reviewing code changes. "
        "Explain what changed and why it matters in plain English. "
        "Do not recite code lines. Focus on behavior, architecture, bug fixes, and risks. "
        "Keep it to at most 3 short paragraphs."
    )

    user_prompt = (
        f"Repository: {repo_full_name}\n"
        f"Actor: {actor_login or 'unknown'}\n"
        f"Commit: {commit_msg[:200]}\n\n"
        f"Files changed:\n{file_overview}\n\n"
        f"Diff:\n{_truncate(raw_diff, 20000)}"
    )

    # ── Attempt 1: google-genai SDK ───────────────────────────────────────────
    if GENAI_AVAILABLE:
        try:
            client = google_genai.Client(api_key=google_api_key)
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=user_prompt,
                config=genai_types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.2,
                    max_output_tokens=1500,
                ),
            )
            text = response.text.strip() if response.text else ""
            if text:
                return text
        except Exception as exc:
            logger.debug("google-genai SDK failed: %s", exc)

    # ── Attempt 2: REST fallback ──────────────────────────────────────────────
    model = "gemini-2.0-flash"
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={google_api_key}"
    )
    payload = {
        "systemInstruction": {"parts": [{"text": system_instruction}]},
        "contents":          [{"parts": [{"text": user_prompt}]}],
        "generationConfig":  {"temperature": 0.2, "maxOutputTokens": 1500},
    }
    try:
        for attempt in range(5):
            resp = requests.post(
                url,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=60,
            )
            if resp.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            resp.raise_for_status()
            break
        data = resp.json()
        text = (
            data.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
            .strip()
        )
        if text:
            return text
    except Exception as exc:
        logger.debug("REST LLM fallback failed: %s", exc)

    return _heuristic_summary(repo_full_name, compact_files, commit_msg)


# =============================================================================
#  GITHUB API HELPERS
# =============================================================================

def _github_headers(github_token: str) -> dict:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"
    return headers


def _fetch_json(url: str, params=None, github_token: str = "") -> dict:
    resp = requests.get(url, headers=_github_headers(github_token), params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _fetch_commit_files(repo_full_name: str, sha: str, github_token: str):
    url = f"{GITHUB_API_BASE}/repos/{repo_full_name}/commits/{sha}"
    payload = _fetch_json(url, github_token=github_token)
    return payload, payload.get("files", []) or []


# =============================================================================
#  NEO4J GRAPH WRITE HELPERS
# =============================================================================

def _write_commit_to_graph(
    driver,
    *,
    repo_full_name: str,
    commit_sha: str,
    modified_files: list,
    dependencies: list,
    actor_login: str = "unknown",
    committed_at: str = "",
    summary_text: str = "",
    diff_text: str = "",
    commit_message: str = "",
    commit_url: str = "",
):
    """Write a single Commit node plus its edges to Neo4j."""
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
        for source, _rel, target in dependencies:
            session.run(
                CYPHER_INGEST_DEPENDENCY,
                filepath=source,
                target_module=target,
            )


def _set_status(driver, repo_full_name: str, status: str, detail: str = "",
                commits_processed: int = 0, files_scanned: int = 0):
    """Persist bootstrap progress as a BootstrapStatus node in Neo4j."""
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


# =============================================================================
#  PHASE 1 — REPOSITORY FILE TREE SCAN
# =============================================================================

def phase1_scan_repo_tree(driver, repo_full_name: str, github_token: str) -> list:
    """
    Fetch the full file tree from GitHub via PyGithub and write
    Directory / File nodes + CONTAINS edges into Neo4j.

    Returns the list of source-file paths suitable for full-content scanning.
    """
    logger.info("Phase 1 — Fetching repository file tree for %s …", repo_full_name)

    auth = Auth.Token(github_token)
    gh = Github(auth=auth)
    repo = gh.get_repo(repo_full_name)
    default_branch = repo.default_branch
    tree = repo.get_git_tree(default_branch, recursive=True).tree

    source_files = []
    dirs_seen = set()

    # Wrap the tree iterator with tqdm for real-time terminal progress
    with tqdm(tree, desc="  Scanning file tree", unit="item", ncols=88,
               bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]") as pbar:
        with driver.session() as session:
            for item in pbar:
                path = item.path
                item_type = item.type  # "blob" or "tree"
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

def get_modified_functions(patch_text: str, filepath: str, ast_data:dict)->list:
    """Reads a Git patch, extracts modified line numbers, and maps them to functions."""
    if not patch_text or not ast_data.get("functions"):
        return[]

    modified_lines = set()

    # Extract line numbers from git patch headers
    for line in patch_text.split("\n"):
        if line.startsWith("@@"):
            # Example header: @@ -40,5 +40,8 @@
            match = re.search(r'\+(\d+)(?:,\d+)? @@', line)
            if match:
                start_line = int(match.group(1))
                line_count = int(match.group(2) or 1)
                for i in range(start_line, start_line + line_count):
                    modified_lines.add(i)
    
    # Cross-reference with AST function boundaries
    modified_funcs = set()
    for func in ast_data["functions"]:
        func_range = set(range(func["start"], func["end"] + 1))
        if modified_lines.intersection(func_range):
            modified_funcs.add(func["id"])

    return list(modified_funcs)


# =============================================================================
#  PHASE 2 — FILE CONTENT / DEPENDENCY SCAN
# =============================================================================

def phase2_scan_file_contents(
    driver,
    repo_full_name: str,
    file_paths: list,
    github_token: str,
) -> int:
    if not file_paths:
        return 0

    logger.info("Phase 2 — Scanning %d source files for import dependencies …", len(file_paths))
    scanned = 0
    batch_params: list[tuple] = []

    def _flush_batch():
        if not batch_params:
            return
        with driver.session() as sess:
            for filepath, target_module in batch_params:
                sess.run(CYPHER_INGEST_DEPENDENCY, filepath=filepath, target_module=target_module)
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
                
                # 1. Dependency Extraction
                deps = extract_imports_from_source(path, source_code)
                for source, _rel, target in deps:
                    batch_params.append((source, target))

                # 2. NEW: AST Extraction and DB Write
                ast_data = parse_python_ast(path, source_code)
                with driver.session() as sess:
                    # Write Functions
                    for func in ast_data["functions"]:
                        sess.run(CYPHER_INGEST_FUNCTION, 
                                 filepath=path, 
                                 func_id=func["id"], 
                                 func_name=func["name"],
                                 func_code=func.get("code", "")) # <--- FIXED: Safely accessing the dict
                                 
                    # Write Calls
                    for caller, callee in ast_data["calls"]:
                        caller_id = f"{path}::{caller}"
                        sess.run(CYPHER_INGEST_CALLS, 
                                 caller_id=caller_id, callee_name=callee)

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


# =============================================================================
#  PHASE 3 — COMMIT HISTORY BACKFILL
# =============================================================================

def phase3_backfill_commits(
    driver,
    repo_full_name: str,
    github_token: str,
    google_api_key: str,
    max_commits: int = 200,
    deep_scan_days: int = 7,
) -> int:
    """
    Walk the repository's commit history via the GitHub REST API, generate an
    LLM summary for each commit diff, and write Commit / User / File nodes
    and AUTHORED / MODIFIED / DEPENDS_ON edges to Neo4j.

    Returns the total number of commits processed.
    """
    logger.info(
        "Phase 3 — Backfilling up to %d commits for %s …", max_commits, repo_full_name
    )

    processed = 0
    page = 1
    per_page = 100
    commit_shas: list[str] = []

    # ── Step 1: collect commit SHAs up to the max ────────────────────────────
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

    total = len(commit_shas)
    logger.info("  Found %d commits to process.", total)

    # ── Step 2: process each commit with a tqdm progress bar ─────────────────
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=deep_scan_days)
    with tqdm(commit_shas, desc="  Backfilling commits", unit="commit", ncols=88,
               bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
        for sha in pbar:
            pbar.set_postfix_str(sha[:8], refresh=False)
            try:
                commit_payload, files = _fetch_commit_files(repo_full_name, sha, github_token)
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

                commit_dt = datetime.strptime(commit_date, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
                is_deep_scan = commit_dt >= cutoff_date

                modified_files_list = [item["filename"] for item in compact_files]
                dependencies = extract_file_dependencies(compact_files)
                raw_diff = render_diff_text(compact_files)

                is_deep_scan = False
                if commit_date:
                    commit_dt = datetime.strptime(commit_date, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
                    is_deep_scan = commit_dt >= cutoff_date

                summary_text = summarize_with_llm(
                    repo_full_name=repo_full_name,
                    actor_login=actor_login,
                    commit_msg=commit_msg,
                    compact_files=compact_files,
                    raw_diff=raw_diff,
                    google_api_key=google_api_key,
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

                    # Parse AST and map the diff
                    historical_ast = parse_python_ast(filepath, raw_code)
                    mod_funcs = get_modified_functions(patch, filepath, historical_ast)

                    with driver.session() as sess:
                        for func_id in mod_funcs:
                            sess.run(CYPHER_MODIFIED_FUNCTION,
                                     commit_sha = sha, func_id=func_id)
                
                except Excepton as e:
                    # Skip the file was deleted or cannot be parsed
                    pass

                processed += 1
                time.sleep(0.5)  # GitHub rate-limit courtesy pause

            except requests.exceptions.HTTPError as exc:
                if exc.response is not None and exc.response.status_code == 403:
                    tqdm.write(f"[WARN] Rate-limit hit on commit {sha[:8]}, sleeping 60s …")
                    time.sleep(60)
                else:
                    tqdm.write(f"[WARN] HTTP error on commit {sha[:8]}: {exc}")
            except Exception as exc:
                tqdm.write(f"[WARN] Failed to process commit {sha[:8]}: {exc}")
                continue

    logger.info("Phase 3 complete — %d/%d commits ingested.", processed, total)
    return processed


# =============================================================================
#  MAIN BOOTSTRAP PIPELINE
# =============================================================================

def bootstrap(
    repo_full_name: str,
    github_token: str,
    google_api_key: str,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    max_commits: int = 200,
):
    """
    Full end-to-end bootstrap pipeline:
      1. Connect to Neo4j and create indexes
      2. Phase 1 — Scan repository file tree  → Directory / File nodes
      3. Phase 2 — Scan file contents         → DEPENDS_ON Module edges
      4. Phase 3 — Backfill commit history    → Commit / User nodes + summaries
    """
    print("=" * 70)
    print(f"  GraphRAG Ingest Pipeline — {repo_full_name}")
    print("=" * 70)

    # ── Connect to Neo4j ──────────────────────────────────────────────────────
    logger.info("Connecting to Neo4j at %s …", neo4j_uri)
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    try:
        driver.verify_connectivity()
        logger.info("Neo4j connection established ✓")
    except Exception as exc:
        logger.error("Cannot connect to Neo4j: %s", exc)
        raise SystemExit(1) from exc

    # ── Create repository node ────────────────────────────────────────────────
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

    # ── Create indexes ────────────────────────────────────────────────────────
    logger.info("Creating Neo4j indexes …")
    for cypher, label in [
        (CYPHER_VECTOR_INDEX,   "vector index issue_embeddings"),
        (CYPHER_FULLTEXT_INDEX, "fulltext index commit_summaries"),
    ]:
        try:
            with driver.session() as session:
                session.run(cypher)
            logger.info("  ✓ %s", label)
        except Exception as exc:
            logger.warning("  Index note (%s): %s", label, exc)

    _set_status(driver, repo_full_name, "in_progress", "Starting Phase 1 …")

    # ── Phase 1: File tree ────────────────────────────────────────────────────
    print()
    source_files = phase1_scan_repo_tree(driver, repo_full_name, github_token)
    _set_status(driver, repo_full_name, "in_progress",
                f"Phase 1 done — {len(source_files)} source files indexed.",
                files_scanned=len(source_files))

    # ── Phase 2: File content / dependency scan ───────────────────────────────
    print()
    files_scanned = phase2_scan_file_contents(
        driver, repo_full_name, source_files, github_token
    )
    _set_status(driver, repo_full_name, "in_progress",
                f"Phase 2 done — {files_scanned} files scanned for imports.",
                files_scanned=files_scanned)

    # ── Phase 3: Commit backfill ──────────────────────────────────────────────
    print()
    commits_processed = phase3_backfill_commits(
        driver, repo_full_name, github_token, google_api_key, max_commits
    )

    # ── Finalise ──────────────────────────────────────────────────────────────
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


# =============================================================================
#  ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Load .env if present
    load_dotenv()

    # ── Read configuration from environment ───────────────────────────────────
    GITHUB_TOKEN   = os.environ.get("GITHUB_TOKEN", "")
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
    NEO4J_URI      = os.environ.get("NEO4J_URI",      "neo4j://localhost:7687")
    NEO4J_USER     = os.environ.get("NEO4J_USER",     "neo4j")
    NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "")
    TARGET_REPO    = os.environ.get("TARGET_REPO",    "neo4j/neo4j-graphrag-python")
    MAX_COMMITS    = int(os.environ.get("MAX_COMMITS", "200"))

    # ── Validate required variables ───────────────────────────────────────────
    missing = []
    if not GITHUB_TOKEN:   missing.append("GITHUB_TOKEN")
    if not NEO4J_PASSWORD: missing.append("NEO4J_PASSWORD")

    if missing:
        print(f"[ERROR] Missing required environment variables: {', '.join(missing)}")
        print("        Create a .env file or export them before running.")
        raise SystemExit(1)

    if not GOOGLE_API_KEY:
        print("[WARN] GOOGLE_API_KEY not set — LLM summaries will use heuristics.")

    # ── Run the pipeline ──────────────────────────────────────────────────────
    bootstrap(
        repo_full_name=TARGET_REPO,
        github_token=GITHUB_TOKEN,
        google_api_key=GOOGLE_API_KEY,
        neo4j_uri=NEO4J_URI,
        neo4j_user=NEO4J_USER,
        neo4j_password=NEO4J_PASSWORD,
        max_commits=MAX_COMMITS,
    )
