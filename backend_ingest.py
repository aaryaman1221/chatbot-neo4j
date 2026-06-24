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
#   FORCE_LLM_UPDATE  — If "true", re-process all commits through the LLM even if
#                       they already exist in Neo4j (overwrites heuristic summaries)
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

import logging
import os
import re
import time
import ast
import httpx
import asyncio
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests
from dotenv import load_dotenv
from github import Auth, Github, GithubException
from neo4j import GraphDatabase
from tqdm import tqdm

# ── AST Parsing Dependencies ───────────────────────────────────────────────

try:
    import tree_sitter_go as tsgo
    from tree_sitter import Language, Parser
    
    GO_LANGUAGE = Language(tsgo.language())
    go_parser = Parser(GO_LANGUAGE)
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    print("[WARN] tree-sitter or tree-sitter-go not installed. Go function parsing will be skipped.")

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

_GRAPH_FLUSH_BATCH = 50


# =============================================================================
#  NEO4J CYPHER TEMPLATES
# =============================================================================

CYPHER_CREATE_REPO = """
MERGE (repo:Repository {full_name: $repo_full_name})
  ON CREATE SET repo.name = $repo_name, repo.url = $repo_url
"""

CYPHER_INGEST_FUNCTION = """
MERGE (repo:Repository {full_name: $repo_full_name})
MERGE (file:File {path: $filepath, repo: $repo_full_name})
MERGE (func:Function {id: $func_id})
  ON CREATE SET
    func.name     = $func_name,
    func.filepath = $filepath,
    func.repo     = $repo_full_name,
    func.code     = $func_code
MERGE (file)-[:DECLARES]->(func)
MERGE (repo)-[:DECLARES]->(func)
"""

CYPHER_INGEST_CALLS = """
MATCH (caller:Function {id: $caller_id})
OPTIONAL MATCH (callee:Function {name: $callee_name, repo: $repo_full_name})
WITH caller, callee WHERE callee IS NOT NULL
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
SET
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
MERGE (file:File {path: $filepath, repo: $repo_full_name})
  ON CREATE SET file.repo = $repo_full_name
MERGE (commit:Commit {sha: $commit_sha})
MERGE (commit)-[:MODIFIED]->(file)
MERGE (repo)-[:CONTAINS_FILE]->(file)
"""

CYPHER_INGEST_DEPENDENCY = """
MERGE (file:File {path: $filepath, repo: $repo_full_name})
MERGE (module:Module {name: $target_module})
MERGE (file)-[r:DEPENDS_ON]->(module)
  ON CREATE SET r.added_in_commit = $commit_sha, r.is_active = true
  ON MATCH  SET r.is_active = true
"""

CYPHER_REMOVE_DEPENDENCY = """
MATCH (file:File {path: $filepath, repo: $repo_full_name})-[r:DEPENDS_ON]->(module:Module {name: $target_module})
SET r.is_active = false, r.deleted_in_commit = $commit_sha
"""

CYPHER_TREE_FILE = """
MERGE (repo:Repository {full_name: $repo_full_name})
MERGE (file:File {path: $child_path, repo: $repo_full_name})
  ON CREATE SET file.repo        = $repo_full_name,
                file.entry_point = $entry_point
MERGE (repo)-[:CONTAINS_FILE]->(file)
WITH file, repo
OPTIONAL MATCH (parent:Directory {path: $parent_path, repo: $repo_full_name})
FOREACH (_ IN CASE WHEN parent IS NOT NULL THEN [1] ELSE [] END |
  MERGE (parent)-[:CONTAINS]->(file)
)
"""

CYPHER_TREE_DIR = """
MERGE (repo:Repository {full_name: $repo_full_name})
MERGE (dir:Directory {path: $child_path, repo: $repo_full_name})
  ON CREATE SET dir.repo    = $repo_full_name,
                dir.utility = $utility
MERGE (repo)-[:CONTAINS_DIR]->(dir)
WITH dir, repo
OPTIONAL MATCH (parent:Directory {path: $parent_path, repo: $repo_full_name})
FOREACH (_ IN CASE WHEN parent IS NOT NULL THEN [1] ELSE [] END |
  MERGE (parent)-[:CONTAINS]->(dir)
)
"""

CYPHER_LINK_REPO_DEPENDENCY = """
MATCH (f:File {repo: $parent_repo})-[:DEPENDS_ON]->(m:Module)
WHERE toLower(m.name) CONTAINS toLower($helper_repo_name)
MERGE (r:Repository {full_name: $helper_repo})
MERGE (f)-[:USES_REPO]->(r)
"""

CYPHER_CROSS_REPO_IMPACT = """
MATCH (helperCommit:Commit {sha: $commit_sha})-[:MODIFIED]->(changed)
WHERE changed:Function OR changed:File
WITH collect(changed) AS changedNodes
MATCH (parentFile:File {repo: $parent_repo})-[:USES_REPO]->(helperRepo:Repository {full_name: $helper_repo})
OPTIONAL MATCH (parentFunc:Function {repo: $parent_repo})-[:CALLS]->(helperFunc:Function)
WHERE helperFunc IN changedNodes
RETURN DISTINCT
  parentFile.path      AS affected_file,
  parentFunc.name      AS affected_function,
  parentFunc.filepath  AS affected_in_file
ORDER BY affected_file
"""

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

CYPHER_FULLTEXT_INDEX = """
CREATE FULLTEXT INDEX commit_summaries IF NOT EXISTS
FOR (c:Commit) ON EACH [c.summary_text, c.diff_text, c.message]
"""

CYPHER_UPSERT_STATUS = """
MERGE (bs:BootstrapStatus {repo: $repo_full_name})
SET bs.status            = $status,
    bs.detail            = $detail,
    bs.commits_processed = $commits_processed,
    bs.files_scanned     = $files_scanned,
    bs.updated_at        = $updated_at
"""

CYPHER_CONSTRAINT_FILE_REPO = """
CREATE CONSTRAINT file_repo_unique IF NOT EXISTS
FOR (f:File) REQUIRE (f.path, f.repo) IS UNIQUE
"""

CYPHER_CONSTRAINT_FUNC_ID = """
CREATE CONSTRAINT func_id_unique IF NOT EXISTS
FOR (f:Function) REQUIRE f.id IS UNIQUE
"""

CYPHER_CLEANUP_UNSCOPED_FILES = "MATCH (f:File)      WHERE f.repo IS NULL DETACH DELETE f"
CYPHER_CLEANUP_UNSCOPED_DIRS  = "MATCH (d:Directory) WHERE d.repo IS NULL DETACH DELETE d"
CYPHER_CLEANUP_UNSCOPED_FUNCS = "MATCH (f:Function)  WHERE f.repo IS NULL DETACH DELETE f"

CYPHER_MARK_FILE_SCANNED = "MATCH (file:File {path: $filepath, repo: $repo_full_name}) SET file.content_scanned = true"


def _is_noise_file(filename: str) -> bool:
    lower_path = filename.lower().replace("\\", "/")
    path_parts = set(lower_path.split("/"))
    if path_parts.intersection(IGNORED_DIRECTORIES):
        return True
    name = lower_path.rsplit("/", 1)[-1]
    if name in IGNORED_FILENAMES:
        return True
    return name.endswith(IGNORED_SUFFIXES)


def parse_python_ast(filepath: str, source_code: str) -> dict:
    if not filepath.endswith(".py") or not source_code:
        return {"functions": [], "calls": []}

    functions = []
    calls = []
    
    try:
        tree = ast.parse(source_code)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_code = ast.get_source_segment(source_code, node) or ""
                functions.append({
                    "name": node.name,
                    "id": f"{filepath}::{node.name}",
                    "start": node.lineno,
                    "end": node.end_lineno,
                    "code": func_code
                })
                for child in ast.walk(node):
                    if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                        calls.append((node.name, child.func.id))
    except Exception:
        pass
        
    return {"functions": functions, "calls": calls}

def parse_go_ast(filepath: str, source_code: str) -> dict:
    if not TREE_SITTER_AVAILABLE or not filepath.endswith(".go") or not source_code:
        return {"functions": [], "calls": []}

    try:
        tree = go_parser.parse(bytes(source_code, "utf8"))
    except Exception as exc:
        logger.debug("Failed to parse Go AST for %s: %s", filepath, exc)
        return {"functions": [], "calls": []}

    functions = []
    calls = []

    def get_text(node):
        return source_code[node.start_byte:node.end_byte]

    def walk(node, current_func=None):
        new_func = current_func
        
        # 1. Identify Function and Method Declarations
        if node.type in ['function_declaration', 'method_declaration']:
            name_node = None
            for child in node.children:
                if child.type in ['identifier', 'field_identifier']:
                    name_node = child
                    break
            
            if name_node:
                name = get_text(name_node)
                new_func = name
                functions.append({
                    "name": name,
                    "id": f"{filepath}::{name}",
                    # Tree-sitter lines are 0-indexed, we convert to 1-indexed
                    "start": node.start_point[0] + 1, 
                    "end": node.end_point[0] + 1,
                    "code": get_text(node)
                })

        # 2. Identify Function Calls within a function
        elif node.type == 'call_expression' and current_func:
            func_node = node.children[0]
            callee_name = None
            
            if func_node.type == 'identifier': # e.g., foo()
                callee_name = get_text(func_node)
            elif func_node.type == 'selector_expression': # e.g., pkg.foo() or obj.foo()
                for child in func_node.children:
                    if child.type == 'field_identifier':
                        callee_name = get_text(child)
                        break
            
            if callee_name:
                calls.append((current_func, callee_name))

        # Recurse through children
        for child in node.children:
            walk(child, new_func)

    walk(tree.root_node)
    return {"functions": functions, "calls": calls}


def _append_dependency(dependencies: list, filepath: str, target_module: str):
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
    kind = _go_file_kind(filepath)
    lines = source_code.split("\n")
    if kind in {"go_mod", "go_work", "go_source"}:
        return _extract_go_dependencies(filepath, lines, patch_mode=False)
    return _extract_generic_dependencies(filepath, lines, patch_mode=False)


def extract_file_dependencies(compact_files: list) -> list:
    """Legacy additive-only extraction. Kept for backwards compatibility."""
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


def extract_temporal_dependencies(compact_files: list) -> dict:
    """Split a git diff into added and removed import dependencies.

    For each changed file in *compact_files* the patch lines are partitioned
    into two buckets:

    * **additions** – lines that start with ``+`` (but are NOT the ``+++``
      hunk header).  The ``+`` prefix is stripped before parsing.
    * **deletions** – lines that start with ``-`` (but are NOT the ``---``
      hunk header).  The ``-`` prefix is stripped before parsing.

    Context lines (starting with a space) and hunk headers are ignored.

    Returns::

        {
            "added":   [(source_filepath, "DEPENDS_ON", target_module), ...],
            "removed": [(source_filepath, "DEPENDS_ON", target_module), ...],
        }
    """
    added: list = []
    removed: list = []

    for item in compact_files:
        source_file = item.get("filename")
        patch = item.get("patch", "")
        if not patch or not source_file:
            continue

        kind = _go_file_kind(source_file)
        extractor = (
            _extract_go_dependencies
            if kind in {"go_mod", "go_work", "go_source"}
            else _extract_generic_dependencies
        )

        addition_lines: list[str] = []
        deletion_lines: list[str] = []

        for raw_line in patch.split("\n"):
            # Skip hunk headers (+++ / ---) and context lines
            if raw_line.startswith("+") and not raw_line.startswith("+++"):
                addition_lines.append(raw_line[1:])  # strip the leading +
            elif raw_line.startswith("-") and not raw_line.startswith("---"):
                deletion_lines.append(raw_line[1:])  # strip the leading -

        for dep in extractor(source_file, addition_lines, patch_mode=False):
            if dep not in added:
                added.append(dep)

        for dep in extractor(source_file, deletion_lines, patch_mode=False):
            if dep not in removed:
                removed.append(dep)

    return {"added": added, "removed": removed}


def _truncate(text: str, limit: int = 5000) -> str:
    if not text:
        return ""
    return text if len(text) <= limit else f"{text[:limit]}\n... [truncated]"


def build_compact_diff(files: list, max_files: int = 12) -> list:
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


def _heuristic_summary(repo_full_name: str, compact_files: list, commit_msg: str) -> str:
    logging.info("heuristic summary used")
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

    if GENAI_AVAILABLE:
        try:
            client = google_genai.Client(api_key=google_api_key)
            response = client.models.generate_content(
                model="gemini-3.5-flash",
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

    model = "gemini-3.5-flash"
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


async def _fetch_commit_async(
    client: httpx.AsyncClient,
    repo_full_name: str,
    sha: str,
    semaphore: asyncio.Semaphore,
) -> tuple[str, dict, list]:
    async with semaphore:
        try:
            url = f"{GITHUB_API_BASE}/repos/{repo_full_name}/commits/{sha}"
            resp = await client.get(url, timeout=30)
            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", 60))
                await asyncio.sleep(retry_after)
                resp = await client.get(url, timeout=30)
            resp.raise_for_status()
            payload = resp.json()
            return sha, payload, payload.get("files", []) or []
        except Exception as exc:
            logger.warning("Async fetch failed for commit %s: %s", sha[:8], exc)
            return sha, {}, []


async def _fetch_all_commits_async(
    repo_full_name: str,
    commit_shas: list,
    github_token: str,
    max_concurrent: int = 8,
) -> dict:
    semaphore = asyncio.Semaphore(max_concurrent)
    headers = _github_headers(github_token)

    async with httpx.AsyncClient(headers=headers) as client:
        tasks = [
            _fetch_commit_async(client, repo_full_name, sha, semaphore)
            for sha in commit_shas
        ]
        results = await asyncio.gather(*tasks)

    return {sha: (payload, files) for sha, payload, files in results}


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
    """Write a single commit and its temporal dependency changes to Neo4j.

    *dependencies* must be a dict with the shape returned by
    :func:`extract_temporal_dependencies`::

        {
            "added":   [(source_filepath, rel, target_module), ...],
            "removed": [(source_filepath, rel, target_module), ...],
        }

    Added edges are MERGEd and marked ``is_active = true``.  Removed edges are
    matched and soft-deleted by setting ``is_active = false`` plus
    ``deleted_in_commit``.
    """
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

def get_modified_functions(patch_text: str, filepath: str, ast_data:dict)->list:
    if not patch_text or not ast_data.get("functions"):
        return[]

    modified_lines = set()

    for line in patch_text.split("\n"):
        if line.startswith("@@"):
            match = re.search(r'\+(\d+)(?:,(\d+))? @@', line)
            if match:
                start_line = int(match.group(1))
                line_count = int(match.group(2) or 1)
                for i in range(start_line, start_line + line_count):
                    modified_lines.add(i)
    
    modified_funcs = set()
    for func in ast_data["functions"]:
        func_range = set(range(func["start"], func["end"] + 1))
        if modified_lines.intersection(func_range):
            modified_funcs.add(func["id"])

    return list(modified_funcs)


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
            for filepath, target_module, repo_fn in batch_params:
                sess.run(
                    CYPHER_INGEST_DEPENDENCY,
                    filepath=filepath,
                    target_module=target_module,
                    repo_full_name=repo_fn,
                    # Phase 2 scans HEAD without a specific commit reference;
                    # use a sentinel so the temporal property is still populated.
                    commit_sha="initial_scan",
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
                        sess.run(
                            CYPHER_INGEST_FUNCTION,
                            repo_full_name=repo_full_name,
                            filepath=path,
                            func_id=prefixed_id,
                            func_name=func["name"],
                            func_code=func.get("code", ""),
                        )
                    for caller, callee in ast_data["calls"]:
                        caller_id = f"{repo_full_name}::{path}::{caller}"
                        sess.run(
                            CYPHER_INGEST_CALLS,
                            caller_id=caller_id,
                            callee_name=callee,
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

    # --- INCREMENTAL INGESTION ---
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
    # --------------------------------

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

                commit_dt = datetime.strptime(commit_date, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
                is_deep_scan = commit_dt >= cutoff_date

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
                        except Exception:
                            pass

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
    """Return only the files that have not yet been fully scanned in Phase 2."""
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
        driver, repo_full_name, unscanned_files, github_token
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

if __name__ == "__main__":
    load_dotenv()

    GITHUB_TOKEN   = os.environ.get("GITHUB_TOKEN", "")
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
    NEO4J_URI      = os.environ.get("NEO4J_URI",      "neo4j://localhost:7687")
    NEO4J_USER     = os.environ.get("NEO4J_USER",     "neo4j")
    NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "")
    TARGET_REPO       = os.environ.get("TARGET_REPO",       "neo4j/neo4j-graphrag-python")
    MAX_COMMITS       = int(os.environ.get("MAX_COMMITS", "200"))
    SKIP_LLM          = os.environ.get("SKIP_LLM",          "false").lower() == "true"
    FORCE_LLM_UPDATE  = os.environ.get("FORCE_LLM_UPDATE",  "false").lower() == "true"

    missing = []
    if not GITHUB_TOKEN:   missing.append("GITHUB_TOKEN")
    if not NEO4J_PASSWORD: missing.append("NEO4J_PASSWORD")

    if missing:
        print(f"[ERROR] Missing required environment variables: {', '.join(missing)}")
        print("        Create a .env file or export them before running.")
        raise SystemExit(1)

    if not GOOGLE_API_KEY:
        print("[WARN] GOOGLE_API_KEY not set — LLM summaries will use heuristics.")

    bootstrap(
        repo_full_name=TARGET_REPO,
        github_token=GITHUB_TOKEN,
        google_api_key=GOOGLE_API_KEY,
        neo4j_uri=NEO4J_URI,
        neo4j_user=NEO4J_USER,
        neo4j_password=NEO4J_PASSWORD,
        max_commits=MAX_COMMITS,
        skip_llm=SKIP_LLM,
        force_llm_update=FORCE_LLM_UPDATE,
    )