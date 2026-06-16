# =============================================================================
# UNIFIED GRAPHRAG EXPLORER: Neo4j + GitHub Issues + Commits + Google Gemini
# =============================================================================
#
# DOCKER SETUP — spin up Neo4j with APOC plugin:
#
#   docker run \
#     --name neo4j-graphrag \
#     -p 7474:7474 -p 7687:7687 \
#     -e NEO4J_AUTH=neo4j/password123 \
#     -e NEO4J_PLUGINS='["apoc"]' \
#     -e NEO4J_apoc_export_file_enabled=true \
#     -e NEO4J_apoc_import_file_enabled=true \
#     -e NEO4J_apoc_import_file_use__neo4j__config=true \
#     -e NEO4J_dbms_security_procedures_unrestricted=apoc.* \
#     -e NEO4J_dbms_memory_heap_initial__size=512m \
#     -e NEO4J_dbms_memory_heap_max__size=1G \
#     neo4j:5.20
#
# Then open the Neo4j Browser at: http://localhost:7474
# Default credentials: neo4j / password123
#
# =============================================================================
# REQUIREMENTS — save as requirements.txt:
#
#   streamlit>=1.35.0
#   PyGithub>=2.3.0
#   neo4j>=5.20.0
#   neo4j-graphrag[google-genai]>=1.1.0
#   google-genai>=0.8.0
#   langchain-google-genai>=1.0.0
#   requests>=2.31.0
#
# Install with:
#   pip install -r requirements.txt
#
# Run with:
#   streamlit run app.py
# =============================================================================

import base64
import json
import logging
import re
import threading
import time
import traceback
from datetime import datetime
from typing import Optional

import requests
import streamlit as st
from github import Github, GithubException
from neo4j import GraphDatabase, exceptions as neo4j_exc

# ── Neo4j GraphRAG imports ────────────────────────────────────────────────────
from neo4j_graphrag.retrievers import VectorRetriever
from neo4j_graphrag.generation import GraphRAG

# ── Google Gemini imports ─────────────────────────────────────────────────────
try:
    from google import genai as google_genai
    from google.genai import types as genai_types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

# ── LangChain Gemini embeddings (for VectorRetriever compatibility) ───────────
try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from langchain_google_genai import ChatGoogleGenerativeAI
    LANGCHAIN_GENAI_AVAILABLE = True
except ImportError:
    LANGCHAIN_GENAI_AVAILABLE = False

logger = logging.getLogger(__name__)

GITHUB_API_BASE = "https://api.github.com"
BOOTSTRAP_MAX_COMMITS = 200  # Configurable cap for the Streamlit UI


# =============================================================================
#  PAGE CONFIG & GLOBAL STYLES
# =============================================================================

st.set_page_config(
    page_title="GraphRAG Explorer — Neo4j + Gemini",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* ── Google Font ─────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* ── Dark gradient background ────────────────────────────── */
    .stApp {
        background: linear-gradient(135deg, #0d0d1a 0%, #0a1628 50%, #0d0d1a 100%);
        min-height: 100vh;
    }

    /* ── Sidebar ─────────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1923 0%, #0d1520 100%);
        border-right: 1px solid rgba(0, 212, 255, 0.15);
    }
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #00d4ff;
    }

    /* ── Main title ──────────────────────────────────────────── */
    .hero-title {
        font-size: 2.6rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00d4ff 0%, #7b61ff 50%, #ff6b6b 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0.2rem;
        line-height: 1.2;
    }
    .hero-sub {
        text-align: center;
        color: rgba(255,255,255,0.5);
        font-size: 0.95rem;
        margin-bottom: 2rem;
        font-weight: 300;
        letter-spacing: 0.5px;
    }

    /* ── Info / metric cards ─────────────────────────────────── */
    .metric-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(0, 212, 255, 0.2);
        border-radius: 12px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
    }
    .metric-card .label {
        font-size: 0.72rem;
        color: rgba(255,255,255,0.45);
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.25rem;
    }
    .metric-card .value {
        font-size: 1.4rem;
        font-weight: 600;
        color: #00d4ff;
    }

    /* ── Pipeline badge ──────────────────────────────────────── */
    .pipeline-badge {
        display: inline-block;
        background: linear-gradient(90deg, #7b61ff22, #00d4ff22);
        border: 1px solid rgba(123, 97, 255, 0.4);
        border-radius: 20px;
        padding: 0.2rem 0.75rem;
        font-size: 0.72rem;
        color: #7b61ff;
        font-weight: 500;
        letter-spacing: 0.5px;
        margin-right: 0.5rem;
    }

    /* ── Chat messages ───────────────────────────────────────── */
    [data-testid="stChatMessage"] {
        background: rgba(255,255,255,0.03) !important;
        border: 1px solid rgba(255,255,255,0.07) !important;
        border-radius: 14px !important;
        margin-bottom: 0.75rem !important;
    }
    [data-testid="stChatMessage"][data-testid*="user"] {
        border-color: rgba(0, 212, 255, 0.2) !important;
    }
    [data-testid="stChatMessage"][data-testid*="assistant"] {
        border-color: rgba(123, 97, 255, 0.2) !important;
    }

    /* ── Buttons ──────────────────────────────────────────────── */
    .stButton > button {
        background: linear-gradient(135deg, #00d4ff, #7b61ff) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        letter-spacing: 0.3px !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.2) !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(0, 212, 255, 0.35) !important;
    }

    /* ── Input fields ─────────────────────────────────────────── */
    .stTextInput > div > div > input,
    .stTextInput > div > div > input:focus {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        border-radius: 8px !important;
        color: #e2e8f0 !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: rgba(0, 212, 255, 0.5) !important;
        box-shadow: 0 0 0 2px rgba(0, 212, 255, 0.1) !important;
    }

    /* ── Divider glow ─────────────────────────────────────────── */
    hr {
        border: none !important;
        height: 1px !important;
        background: linear-gradient(90deg, transparent, rgba(0,212,255,0.4), transparent) !important;
        margin: 1.5rem 0 !important;
    }

    /* ── Code / monospace ─────────────────────────────────────── */
    code {
        background: rgba(0, 212, 255, 0.08) !important;
        color: #00d4ff !important;
        border-radius: 4px !important;
        padding: 0.1rem 0.35rem !important;
        font-size: 0.85em !important;
    }

    /* ── Status indicators ────────────────────────────────────── */
    .status-dot {
        display: inline-block;
        width: 8px; height: 8px;
        border-radius: 50%;
        margin-right: 6px;
        vertical-align: middle;
    }
    .status-dot.green  { background: #00e676; box-shadow: 0 0 6px #00e676; }
    .status-dot.red    { background: #ff5252; box-shadow: 0 0 6px #ff5252; }
    .status-dot.yellow { background: #ffd740; box-shadow: 0 0 6px #ffd740; }
    .status-dot.blue   { background: #448aff; box-shadow: 0 0 6px #448aff; }

    /* ── Section headers ──────────────────────────────────────── */
    .section-header {
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: rgba(255,255,255,0.35);
        margin: 1.25rem 0 0.6rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
#  SESSION STATE INITIALISATION
# =============================================================================

def _init_session():
    defaults = {
        "messages": [],
        "neo4j_driver": None,
        "ingestion_done": False,
        "issue_count": 0,
        "repo_name": "",
        # Bootstrap tracking
        "bootstrap_status": "idle",   # idle | running | completed | failed
        "bootstrap_detail": "",
        "bootstrap_commits": 0,
        "bootstrap_files": 0,
        "bootstrap_thread": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

_init_session()


# =============================================================================
#  FILE FILTER CONSTANTS  (shared by ingestion + bootstrap)
# =============================================================================

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

# Bootstrap batch size for Neo4j writes
_GRAPH_FLUSH_BATCH = 50


# =============================================================================
#  NEO4J CYPHER TEMPLATES
# =============================================================================

INGEST_ISSUE_CYPHER = """
MERGE (repo:Repository {full_name: $repo_full_name})
  ON CREATE SET repo.name = $repo_name, repo.url = $repo_url

MERGE (owner:User {login: $owner_login})
  ON CREATE SET owner.url = $owner_url
MERGE (owner)-[:OWNS]->(repo)

MERGE (opener:User {login: $opener_login})
  ON CREATE SET opener.url = $opener_url

MERGE (issue:Issue {id: $issue_id})
  ON CREATE SET
    issue.number    = $issue_number,
    issue.title     = $issue_title,
    issue.body      = $issue_body,
    issue.state     = $issue_state,
    issue.url       = $issue_url,
    issue.labels    = $issue_labels,
    issue.created   = $issue_created
SET issue.embedding = $embedding

MERGE (repo)-[:HAS_ISSUE]->(issue)
MERGE (opener)-[:OPENED]->(issue)
"""

INGEST_COMMIT_CYPHER = """
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

INGEST_FILE_CYPHER = """
MERGE (repo:Repository {full_name: $repo_full_name})
MERGE (file:File {path: $filepath})
  ON CREATE SET file.repo = $repo_full_name
MERGE (commit:Commit {sha: $commit_sha})
MERGE (commit)-[:MODIFIED]->(file)
"""

INGEST_DEPENDENCY_CYPHER = """
MERGE (file:File {path: $filepath})
MERGE (module:Module {name: $target_module})
MERGE (file)-[:DEPENDS_ON]->(module)
"""

INGEST_TREE_FILE_CYPHER = """
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

INGEST_TREE_DIR_CYPHER = """
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

INGEST_REVIEWED_CYPHER = """
MERGE (commit:Commit {sha: $commit_sha})
  ON CREATE SET commit.repo = $repo_full_name
MERGE (file:File {path: $filepath})
  ON CREATE SET file.repo = $repo_full_name
MERGE (commit)-[:REVIEWED]->(file)
"""

# Vector index covers both Issue embeddings and Commit summary_text (full-text)
VECTOR_INDEX_ISSUE_CYPHER = """
CREATE VECTOR INDEX issue_embeddings IF NOT EXISTS
FOR (i:Issue) ON (i.embedding)
OPTIONS {
  indexConfig: {
    `vector.dimensions`: 3072,
    `vector.similarity_function`: 'cosine'
  }
}
"""

# Full-text index on Commit summary/diff so the RAG can search code context
FULLTEXT_COMMIT_CYPHER = """
CREATE FULLTEXT INDEX commit_summaries IF NOT EXISTS
FOR (c:Commit) ON EACH [c.summary_text, c.diff_text, c.message]
"""


# =============================================================================
#  HELPER UTILITIES
# =============================================================================

def _status_dot(color: str) -> str:
    return f'<span class="status-dot {color}"></span>'


def _check_neo4j(uri: str, user: str, pwd: str) -> tuple[bool, str]:
    try:
        drv = GraphDatabase.driver(uri, auth=(user, pwd))
        drv.verify_connectivity()
        drv.close()
        return True, "Connected"
    except Exception as exc:
        return False, str(exc)


def _get_neo4j_driver(uri: str, user: str, pwd: str):
    if st.session_state.neo4j_driver is None:
        st.session_state.neo4j_driver = GraphDatabase.driver(uri, auth=(user, pwd))
    return st.session_state.neo4j_driver


def _embed_text(text: str, api_key: str, model: str = "models/gemini-embedding-2-preview") -> list[float]:
    """Generate an embedding vector using Google GenAI."""
    client = google_genai.Client(api_key=api_key)
    response = client.models.embed_content(model=model, contents=text)
    embedding = response.embeddings[0].values
    return list(embedding) if embedding else []


def _safe_body(issue) -> str:
    """Return a non-None issue body, truncated to 4 000 chars for embedding."""
    body = issue.body or ""
    title = issue.title or ""
    combined = f"Title: {title}\n\n{body}"
    return combined[:4000]


def _truncate(text, limit=5000):
    if not text:
        return ""
    return text if len(text) <= limit else f"{text[:limit]}\n... [truncated]"


# =============================================================================
#  FILE FILTER & DEPENDENCY EXTRACTION UTILITIES
#  (ported from github_monitor.py — no NetworkX, pure Python)
# =============================================================================

def _is_noise_file(filename: str) -> bool:
    lower_path = filename.lower().replace("\\", "/")
    path_parts = set(lower_path.split("/"))
    if path_parts.intersection(IGNORED_DIRECTORIES):
        return True
    name = lower_path.rsplit("/", 1)[-1]
    if name in IGNORED_FILENAMES:
        return True
    return name.endswith(IGNORED_SUFFIXES)


def _append_dependency(dependencies: list, filepath: str, target_module: str):
    """Append a dependency edge once per file/module pair."""
    if not target_module:
        return
    edge = (filepath, "DEPENDS_ON", target_module)
    if edge not in dependencies:
        dependencies.append(edge)


def _go_file_kind(filepath: str) -> Optional[str]:
    name = filepath.lower().replace("\\", "/").rsplit("/", 1)[-1]
    if name == "go.mod":
        return "go_mod"
    if name == "go.work":
        return "go_work"
    if name.endswith(".go"):
        return "go_source"
    return None


def _extract_go_dependencies_from_lines(filepath: str, lines: list, patch_mode: bool = False) -> list:
    """Extract Go import/require targets from either source text or patch text."""
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
            in_import_block = True
            continue

        if in_import_block:
            if stripped.startswith(")"):
                in_import_block = False
                continue
            match = re.search(r'["`]\s*([^"`]+?)\s*["`]', stripped)
            if match:
                _append_dependency(dependencies, filepath, match.group(1))
            continue

        if stripped.startswith("import "):
            match = re.search(r'["`]\s*([^"`]+?)\s*["`]', stripped)
            if match:
                _append_dependency(dependencies, filepath, match.group(1))
            continue

        if stripped.startswith("require ("):
            in_require_block = True
            continue
        if stripped.startswith("use ("):
            in_use_block = True
            continue
        if stripped.startswith("replace ("):
            in_replace_block = True
            continue

        if in_require_block:
            if stripped.startswith(")"):
                in_require_block = False
                continue
            parts = stripped.split()
            if len(parts) >= 2:
                _append_dependency(dependencies, filepath, parts[0])
            continue

        if in_use_block:
            if stripped.startswith(")"):
                in_use_block = False
                continue
            _append_dependency(dependencies, filepath, stripped)
            continue

        if in_replace_block:
            if stripped.startswith(")"):
                in_replace_block = False
                continue
            if "=>" in stripped:
                left, right = [p.strip() for p in stripped.split("=>", 1)]
                left = left.split()[0] if left else ""
                right = right.split()[0] if right else ""
                _append_dependency(dependencies, filepath, left)
                _append_dependency(dependencies, filepath, right)
            continue

        if stripped.startswith("require "):
            parts = stripped.split()
            if len(parts) >= 3:
                _append_dependency(dependencies, filepath, parts[1])
        if stripped.startswith("use "):
            parts = stripped.split(None, 1)
            if len(parts) == 2:
                _append_dependency(dependencies, filepath, parts[1].strip())
        if stripped.startswith("replace ") and "=>" in stripped:
            body = stripped[len("replace "):].strip()
            left, right = [p.strip() for p in body.split("=>", 1)]
            left = left.split()[0] if left else ""
            right = right.split()[0] if right else ""
            _append_dependency(dependencies, filepath, left)
            _append_dependency(dependencies, filepath, right)

    return dependencies


def _extract_generic_dependencies_from_lines(filepath: str, lines: list, patch_mode: bool = False) -> list:
    """Extract imports for non-Go languages from line-oriented text."""
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
            match = re.search(pattern, stripped)
            if match:
                _append_dependency(dependencies, filepath, match.group(1))

    return dependencies


def extract_file_dependencies(compact_files: list) -> list:
    """Scans code diff patches to extract import/require statements."""
    dependencies = []
    for item in compact_files:
        source_file = item.get("filename")
        patch = item.get("patch", "")
        if not patch or not source_file:
            continue
        kind = _go_file_kind(source_file)
        lines = patch.split("\n")
        if kind in {"go_mod", "go_work", "go_source"}:
            dependencies.extend(
                dep for dep in _extract_go_dependencies_from_lines(source_file, lines, patch_mode=True)
                if dep not in dependencies
            )
        else:
            dependencies.extend(
                dep for dep in _extract_generic_dependencies_from_lines(source_file, lines, patch_mode=True)
                if dep not in dependencies
            )
    return dependencies


def extract_imports_from_source(filepath: str, source_code: str) -> list:
    """Parses the full source code of a file for import/require statements.

    Unlike extract_file_dependencies() which only reads diff patches,
    this scans every line to catch pre-existing imports.
    """
    kind = _go_file_kind(filepath)
    lines = source_code.split("\n")
    if kind in {"go_mod", "go_work", "go_source"}:
        return _extract_go_dependencies_from_lines(filepath, lines, patch_mode=False)
    return _extract_generic_dependencies_from_lines(filepath, lines, patch_mode=False)


# =============================================================================
#  DIFF BUILDING & LLM SUMMARIZATION UTILITIES
#  (ported from github_monitor.py — uses google-genai client; requests fallback)
# =============================================================================

def build_compact_diff(files: list, max_files: int = 12) -> list:
    """Filter, truncate, and rank file diffs for LLM consumption."""
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
            "status": item.get("status", "modified"),
            "additions": item.get("additions", 0),
            "deletions": item.get("deletions", 0),
            "changes": item.get("changes", 0),
            "patch": _truncate(patch, 4000),
        })
    filtered.sort(key=lambda x: x.get("changes", 0), reverse=True)
    return filtered[:max_files]


def render_diff_text(files: list) -> str:
    """Convert compact_diff dicts into a single human-readable text block."""
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


def _heuristic_summary(repo_full_name: str, event_type: str, compact_files: list, meta: dict) -> str:
    """Fallback summary when no LLM is available."""
    if not compact_files:
        return f"{event_type.title()} event in {repo_full_name}. No text patches available."
    parts = []
    for item in compact_files[:5]:
        parts.append(
            f"{item['filename']} ({item.get('status','modified')}, "
            f"+{item.get('additions',0)}/-{item.get('deletions',0)})"
        )
    extra = f" and {len(compact_files) - 5} more file(s)" if len(compact_files) > 5 else ""
    action = meta.get("action")
    action_part = f" action={action}" if action else ""
    return (
        f"{event_type.title()} event in {repo_full_name}{action_part}. "
        f"Key files touched: {', '.join(parts)}{extra}. "
        "This likely changes behavior in the listed areas and should be reviewed for impact and regressions."
    )


def summarize_with_llm(
    repo_full_name: str,
    event_type: str,
    actor_login: str,
    meta: dict,
    compact_files: list,
    raw_diff: str,
    google_api_key: Optional[str] = None,
) -> str:
    """Generate an LLM summary of a commit diff.

    Attempts to use the google-genai client (preferred).
    Falls back to a direct requests call, then to a heuristic summary.
    The api_key is taken from the parameter; if None, it attempts
    st.session_state (safe to call from background threads only when
    session_state was populated before the thread started).
    """
    api_key = google_api_key
    if not api_key:
        api_key = st.session_state.get("_google_api_key_snapshot")

    file_overview = "\n".join(
        f"- {item['filename']} ({item.get('status','modified')} "
        f"| +{item.get('additions',0)} / -{item.get('deletions',0)})"
        for item in compact_files
    ) or "- No text patches were available."

    user_prompt = (
        f"Repository: {repo_full_name}\n"
        f"Event: {event_type}\n"
        f"Actor: {actor_login or 'unknown'}\n"
        f"Metadata: {json.dumps(meta, ensure_ascii=True, default=str)}\n\n"
        f"Files changed:\n{file_overview}\n\n"
        f"Diff:\n{_truncate(raw_diff, 20000)}"
    )

    system_instruction = (
        "You are an expert senior developer reviewing code changes. "
        "Explain what changed and why it matters in plain English. "
        "Do not recite code lines. Focus on behavior, architecture, bug fixes, and risks. "
        "Keep it to at most 3 short paragraphs."
    )

    if not api_key:
        return _heuristic_summary(repo_full_name, event_type, compact_files, meta)

    # ── Try google-genai SDK first ────────────────────────────────────────────
    if GENAI_AVAILABLE:
        try:
            client = google_genai.Client(api_key=api_key)
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
            return text or _heuristic_summary(repo_full_name, event_type, compact_files, meta)
        except Exception:
            pass  # fall through to requests

    # ── Requests fallback ─────────────────────────────────────────────────────
    model = "gemini-2.0-flash"
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )
    payload = {
        "systemInstruction": {"parts": [{"text": system_instruction}]},
        "contents": [{"parts": [{"text": user_prompt}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 1500},
    }
    try:
        for attempt in range(5):
            resp = requests.post(url, headers={"Content-Type": "application/json"},
                                 json=payload, timeout=60)
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
        return text or _heuristic_summary(repo_full_name, event_type, compact_files, meta)
    except Exception:
        return _heuristic_summary(repo_full_name, event_type, compact_files, meta)


# =============================================================================
#  GITHUB API HELPERS  (raw requests, token from snapshot)
# =============================================================================

def _github_headers(github_token: Optional[str] = None) -> dict:
    token = github_token or st.session_state.get("_github_token_snapshot", "")
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _fetch_json(url: str, params=None, github_token: Optional[str] = None) -> dict:
    resp = requests.get(url, headers=_github_headers(github_token), params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _fetch_commit_files(repo_full_name: str, sha: str, github_token: Optional[str] = None):
    url = f"{GITHUB_API_BASE}/repos/{repo_full_name}/commits/{sha}"
    payload = _fetch_json(url, github_token=github_token)
    return payload, payload.get("files", []) or []


# =============================================================================
#  NEO4J GRAPH MUTATION HELPERS
# =============================================================================

def update_neo4j_knowledge_graph(
    driver,
    repo_full_name: str,
    commit_sha: str,
    modified_files: list,
    dependencies: list,
    actor_login: Optional[str] = None,
    committed_at: Optional[str] = None,
    summary_text: str = "",
    diff_text: str = "",
    commit_message: str = "",
    commit_url: str = "",
):
    """Write a Commit node + its MODIFIED File edges and DEPENDS_ON Module edges to Neo4j.

    This is the Neo4j equivalent of the old update_knowledge_graph() that
    mutated a NetworkX DiGraph backed by a JSON file on disk.
    """
    actor_login = actor_login or "unknown_author"
    committed_at = committed_at or ""

    with driver.session() as session:
        # 1. Create Commit node, User (author) node, and relationships
        session.run(
            INGEST_COMMIT_CYPHER,
            repo_full_name=repo_full_name,
            actor_login=actor_login,
            commit_sha=commit_sha,
            committed_at=committed_at,
            summary_text=summary_text,
            diff_text=diff_text,
            commit_message=commit_message,
            commit_url=commit_url,
        )

        # 2. Link Commit → Modified Files
        for filepath in modified_files:
            session.run(
                INGEST_FILE_CYPHER,
                repo_full_name=repo_full_name,
                commit_sha=commit_sha,
                filepath=filepath,
            )

        # 3. Module dependency edges  (file)-[:DEPENDS_ON]->(module)
        for source, _rel, target in dependencies:
            session.run(
                INGEST_DEPENDENCY_CYPHER,
                filepath=source,
                target_module=target,
            )


def scan_neo4j_repo_tree(
    driver,
    github_token: str,
    repo_full_name: str,
    status_callback=None,
) -> list:
    """Fetch the full file tree from GitHub and build the Directory/File hierarchy in Neo4j.

    Neo4j replacement for the old scan_repo_tree() that used NetworkX.
    Returns the list of source-file paths suitable for full-content scanning.
    """
    gh = Github(github_token)
    repo = gh.get_repo(repo_full_name)
    default_branch = repo.default_branch
    tree = repo.get_git_tree(default_branch, recursive=True).tree

    source_files = []
    dirs_seen = set()

    with driver.session() as session:
        for item in tree:
            path = item.path
            item_type = item.type  # "blob" or "tree"

            if _is_noise_file(path):
                continue

            if item_type == "tree":
                parent_path = path.rsplit("/", 1)[0] if "/" in path else ""
                dir_basename = path.rsplit("/", 1)[-1].lower()
                is_utility = dir_basename in UTILITY_DIRS
                session.run(
                    INGEST_TREE_DIR_CYPHER,
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
                    INGEST_TREE_FILE_CYPHER,
                    repo_full_name=repo_full_name,
                    child_path=path,
                    parent_path=parent_path,
                    entry_point=is_entry,
                )

                # Collect scannable source files
                fname_lower = path.rsplit("/", 1)[-1].lower()
                if ext in SOURCE_EXTENSIONS or fname_lower in SOURCE_FILENAMES:
                    source_files.append(path)

    if status_callback:
        status_callback(f"File tree indexed: {len(source_files)} source files, {len(dirs_seen)} directories.")

    return source_files


def scan_neo4j_file_contents(
    driver,
    repo_full_name: str,
    file_paths: list,
    github_token: str,
    status_callback=None,
) -> int:
    """Fetch raw content for each source file and write DEPENDS_ON edges to Neo4j.

    Neo4j replacement for the old scan_file_contents() that used NetworkX batches.
    Returns the number of files successfully scanned.
    """
    if not file_paths:
        return 0

    scanned = 0
    batch_params: list[tuple] = []   # (filepath, target_module) pairs

    def _flush_batch():
        if not batch_params:
            return
        with driver.session() as sess:
            for filepath, target_module in batch_params:
                sess.run(INGEST_DEPENDENCY_CYPHER, filepath=filepath, target_module=target_module)
        batch_params.clear()

    for path in file_paths:
        try:
            url = f"{GITHUB_API_BASE}/repos/{repo_full_name}/contents/{path}"
            data = _fetch_json(url, github_token=github_token)

            content_b64 = data.get("content")
            if not content_b64:
                continue

            source_code = base64.b64decode(content_b64).decode("utf-8", errors="replace")
            deps = extract_imports_from_source(path, source_code)

            for source, _rel, target in deps:
                batch_params.append((source, target))

            scanned += 1

            if scanned % _GRAPH_FLUSH_BATCH == 0:
                _flush_batch()
                if status_callback:
                    status_callback(f"File content scan: {scanned}/{len(file_paths)} files processed…")

            time.sleep(0.1)  # gentle rate-limit buffer

        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 403:
                logger.warning("Rate limit hit scanning %s, sleeping 60s", path)
                time.sleep(60)
            else:
                logger.warning("HTTP error scanning %s: %s", path, e)
        except Exception as e:
            logger.warning("Error scanning file %s: %s", path, e)

    _flush_batch()
    return scanned


# =============================================================================
#  BOOTSTRAP STATUS  (stored in Neo4j as a BootstrapStatus node)
# =============================================================================

BOOTSTRAP_STATUS_CYPHER = """
MERGE (bs:BootstrapStatus {repo: $repo_full_name})
SET bs.status            = $status,
    bs.detail            = $detail,
    bs.commits_processed = $commits_processed,
    bs.files_scanned     = $files_scanned,
    bs.updated_at        = $updated_at
"""

GET_BOOTSTRAP_STATUS_CYPHER = """
MATCH (bs:BootstrapStatus {repo: $repo_full_name})
RETURN bs
"""


def _set_bootstrap_status_neo4j(
    driver,
    repo_full_name: str,
    status: str,
    detail: str = "",
    commits_processed: int = 0,
    files_scanned: int = 0,
):
    """Persist bootstrap progress as a Neo4j node (replaces SQLite table)."""
    try:
        with driver.session() as session:
            session.run(
                BOOTSTRAP_STATUS_CYPHER,
                repo_full_name=repo_full_name,
                status=status,
                detail=detail,
                commits_processed=commits_processed,
                files_scanned=files_scanned,
                updated_at=datetime.utcnow().isoformat(),
            )
    except Exception as exc:
        logger.warning("Could not persist bootstrap status to Neo4j: %s", exc)


def _get_bootstrap_status_neo4j(driver, repo_full_name: str) -> Optional[dict]:
    try:
        with driver.session() as session:
            result = session.run(GET_BOOTSTRAP_STATUS_CYPHER, repo_full_name=repo_full_name)
            record = result.single()
            if record:
                return dict(record["bs"])
    except Exception:
        pass
    return None


# =============================================================================
#  BOOTSTRAP PIPELINE  (background thread)
# =============================================================================

def _bootstrap_worker(
    driver,
    repo_full_name: str,
    github_token: str,
    google_api_key: str,
    max_commits: int,
):
    """Run the full bootstrap pipeline in a daemon thread.

    Phases:
      1. Scan repo file tree → File/Directory nodes in Neo4j
      2. Scan file contents  → DEPENDS_ON Module edges
      3. Backfill commit history → Commit nodes with summaries stored on the node
    """
    def _update(status, detail="", commits=0, files=0):
        # Update both Neo4j and Streamlit session state (thread-safe write)
        _set_bootstrap_status_neo4j(driver, repo_full_name, status, detail, commits, files)
        st.session_state["bootstrap_status"] = status
        st.session_state["bootstrap_detail"] = detail
        st.session_state["bootstrap_commits"] = commits
        st.session_state["bootstrap_files"] = files

    try:
        _update("in_progress", "Phase 1/3: Scanning repository file tree…")

        # ── Phase 1: File tree ────────────────────────────────────────────────
        source_files = scan_neo4j_repo_tree(
            driver, github_token, repo_full_name,
            status_callback=lambda msg: _update("in_progress", msg),
        )
        files_count = len(source_files)
        _update("in_progress", f"Phase 2/3: Scanning {files_count} source files for imports…",
                files=files_count)

        # ── Phase 2: File content / dependency scan ───────────────────────────
        files_scanned = scan_neo4j_file_contents(
            driver, repo_full_name, source_files, github_token,
            status_callback=lambda msg: _update("in_progress", msg, files=files_count),
        )
        _update("in_progress", f"Phase 3/3: Backfilling commit history (max {max_commits})…",
                files=files_scanned)

        # ── Phase 3: Commit backfill ──────────────────────────────────────────
        processed = 0
        page = 1
        per_page = 100

        while processed < max_commits:
            try:
                url = f"{GITHUB_API_BASE}/repos/{repo_full_name}/commits"
                resp = requests.get(
                    url,
                    headers=_github_headers(github_token),
                    params={"per_page": per_page, "page": page},
                    timeout=30,
                )
                resp.raise_for_status()
                commits = resp.json()
                if not commits:
                    break

                for commit_data in commits:
                    if processed >= max_commits:
                        break
                    sha = commit_data.get("sha")
                    if not sha:
                        continue

                    try:
                        commit_payload, files = _fetch_commit_files(
                            repo_full_name, sha, github_token
                        )
                        compact_files = build_compact_diff(files)

                        commit_info = commit_data.get("commit", {})
                        actor_login = (
                            (commit_data.get("author") or {}).get("login")
                            or (commit_info.get("author") or {}).get("name")
                            or "unknown"
                        )
                        commit_msg = commit_info.get("message", "")
                        commit_date = commit_info.get("author", {}).get("date", "")
                        commit_url = commit_data.get("html_url", "")

                        modified_files_list = [item["filename"] for item in compact_files]
                        dependencies = extract_file_dependencies(compact_files)
                        raw_diff = render_diff_text(compact_files)
                        meta = {"event_type": "push", "commit_message": commit_msg, "backfill": True}

                        # Summarize — summary_text stored on Commit node (not SQLite)
                        summary_text = summarize_with_llm(
                            repo_full_name, "push", actor_login, meta,
                            compact_files, raw_diff, google_api_key=google_api_key,
                        )

                        update_neo4j_knowledge_graph(
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

                        processed += 1

                        if processed % 25 == 0:
                            _update(
                                "in_progress",
                                f"Backfilling commits: {processed}/{max_commits}",
                                commits=processed,
                                files=files_scanned,
                            )

                        time.sleep(0.5)  # rate-limit buffer

                    except Exception as e:
                        logger.warning("Error backfilling commit %s: %s", sha[:8], e)
                        continue

                # Rate-limit header awareness
                remaining = resp.headers.get("X-RateLimit-Remaining")
                if remaining and int(remaining) < 10:
                    reset_time = int(resp.headers.get("X-RateLimit-Reset", 0))
                    sleep_for = max(reset_time - int(time.time()), 60)
                    time.sleep(sleep_for)

                if len(commits) < per_page:
                    break
                page += 1
                time.sleep(0.5)

            except requests.exceptions.HTTPError as e:
                if e.response is not None and e.response.status_code == 403:
                    time.sleep(60)
                else:
                    logger.error("HTTP error during backfill: %s", e)
                    break
            except Exception as e:
                logger.error("Unexpected error during backfill: %s", e)
                break

        _update(
            "completed",
            f"Done: {files_scanned} files scanned, {processed} commits backfilled.",
            commits=processed,
            files=files_scanned,
        )
        logger.info(
            "Bootstrap COMPLETE for %s | %d files, %d commits",
            repo_full_name, files_scanned, processed,
        )

    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {exc}"
        logger.exception("Bootstrap FAILED for %s: %s", repo_full_name, exc)
        _update("failed", error_msg)


def enqueue_bootstrap(
    driver,
    repo_full_name: str,
    github_token: str,
    google_api_key: str,
    max_commits: int = BOOTSTRAP_MAX_COMMITS,
):
    """Launch the bootstrap pipeline in a non-blocking daemon thread.

    Returns the thread so the caller can monitor it if needed.
    """
    # Snapshot API keys into session state so the worker thread can read them
    st.session_state["_github_token_snapshot"] = github_token
    st.session_state["_google_api_key_snapshot"] = google_api_key

    existing = _get_bootstrap_status_neo4j(driver, repo_full_name)
    if existing and existing.get("status") == "completed":
        st.session_state["bootstrap_status"] = "completed"
        st.session_state["bootstrap_detail"] = existing.get("detail", "Already completed.")
        st.session_state["bootstrap_commits"] = existing.get("commits_processed", 0)
        st.session_state["bootstrap_files"] = existing.get("files_scanned", 0)
        return None

    thread = threading.Thread(
        target=_bootstrap_worker,
        args=(driver, repo_full_name, github_token, google_api_key, max_commits),
        daemon=True,
        name=f"bootstrap-{repo_full_name}",
    )
    st.session_state["bootstrap_thread"] = thread
    st.session_state["bootstrap_status"] = "running"
    st.session_state["bootstrap_detail"] = "Starting…"
    thread.start()
    logger.info("Bootstrap thread started for %s", repo_full_name)
    return thread


# =============================================================================
#  INGESTION PIPELINE  (issues + initial codebase tree)
# =============================================================================

def run_ingestion(
    github_token: str,
    google_api_key: str,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_pwd: str,
    repo_full_name: str,
    n_issues: int = 15,
):
    """Full ingestion pipeline: GitHub Issues → embeddings → Neo4j."""
    progress = st.progress(0, text="🔌 Connecting to GitHub…")

    # 1 ── GitHub connection ────────────────────────────────────────────────
    try:
        gh = Github(github_token)
        repo = gh.get_repo(repo_full_name)
        owner = repo.owner
    except GithubException as exc:
        st.error(f"❌ GitHub error: {exc.data.get('message', str(exc))}")
        return False
    except Exception as exc:
        st.error(f"❌ GitHub connection failed: {exc}")
        return False

    progress.progress(10, text="📋 Fetching issues…")

    # 2 ── Fetch issues ─────────────────────────────────────────────────────
    try:
        open_issues = list(
            repo.get_issues(state="open", sort="created", direction="desc")[:n_issues]
        )
    except Exception as exc:
        st.error(f"❌ Failed to fetch issues: {exc}")
        return False

    if not open_issues:
        st.warning("⚠️ No open issues found for this repository.")
        return False

    actual_count = len(open_issues)
    progress.progress(20, text=f"✅ Fetched {actual_count} issues. Connecting to Neo4j…")

    # 3 ── Neo4j connection ─────────────────────────────────────────────────
    try:
        driver = _get_neo4j_driver(neo4j_uri, neo4j_user, neo4j_pwd)
        driver.verify_connectivity()
    except Exception as exc:
        st.error(f"❌ Neo4j connection failed: {exc}")
        return False

    progress.progress(30, text="🔧 Creating indexes…")

    # 4 ── Create vector + fulltext indexes ────────────────────────────────
    try:
        with driver.session() as session:
            session.run(VECTOR_INDEX_ISSUE_CYPHER)
    except Exception as exc:
        st.warning(f"⚠️ Vector index note: {exc}")

    try:
        with driver.session() as session:
            session.run(FULLTEXT_COMMIT_CYPHER)
    except Exception as exc:
        st.warning(f"⚠️ Fulltext commit index note: {exc}")

    progress.progress(35, text="🌳 Building repository file tree in Neo4j…")

    # 5 ── Build File Tree ─────────────────────────────────────────────────
    try:
        scannable_files = scan_neo4j_repo_tree(driver, github_token, repo_full_name)
        st.toast(f"Indexed {len(scannable_files)} source files into the graph!", icon="🌳")
    except Exception as exc:
        st.warning(f"⚠️ Codebase tree ingestion failed: {exc}")

    # 6 ── Embed & ingest each issue ───────────────────────────────────────
    embed_status = st.empty()
    failed_embeds = []

    for idx, issue in enumerate(open_issues):
        pct = 40 + int((idx / actual_count) * 55)
        progress.progress(pct, text=f"🧠 Embedding issue {idx + 1}/{actual_count}: #{issue.number}…")
        embed_status.caption(f"**#{issue.number}** — {issue.title[:80]}…")

        try:
            body_text = _safe_body(issue)
            embedding = _embed_text(body_text, google_api_key)
        except Exception as exc:
            failed_embeds.append(issue.number)
            embedding = []
            st.warning(f"⚠️ Embedding failed for issue #{issue.number}: {exc}")

        try:
            labels = [lbl.name for lbl in issue.labels]
            with driver.session() as session:
                session.run(
                    INGEST_ISSUE_CYPHER,
                    repo_full_name=repo.full_name,
                    repo_name=repo.name,
                    repo_url=repo.html_url,
                    owner_login=owner.login,
                    owner_url=owner.html_url,
                    opener_login=issue.user.login,
                    opener_url=issue.user.html_url,
                    issue_id=issue.id,
                    issue_number=issue.number,
                    issue_title=issue.title,
                    issue_body=(issue.body or "")[:2000],
                    issue_state=issue.state,
                    issue_url=issue.html_url,
                    issue_labels=labels,
                    issue_created=str(issue.created_at),
                    embedding=embedding,
                )
        except Exception as exc:
            st.error(f"❌ Neo4j write failed for issue #{issue.number}: {exc}")
            return False

        time.sleep(0.1)

    embed_status.empty()
    progress.progress(97, text="🏗️ Finalising graph schema…")
    time.sleep(0.3)
    progress.progress(100, text="✅ Ingestion complete!")
    time.sleep(0.4)
    progress.empty()

    st.session_state.ingestion_done = True
    st.session_state.issue_count = actual_count
    st.session_state.repo_name = repo_full_name

    if failed_embeds:
        st.warning(
            f"⚠️ Embeddings skipped for issues: {failed_embeds}. "
            "Those nodes exist in the graph but won't appear in vector search."
        )

    return True


# =============================================================================
#  GRAPHRAG QUERY PIPELINE
# =============================================================================

class GeminiEmbedderWrapper:
    """Thin wrapper so google-genai embeddings conform to the
    neo4j_graphrag Embedder interface (embed_query method)."""

    def __init__(self, api_key: str, model: str = "models/gemini-embedding-2-preview"):
        self.api_key = api_key
        self.model = model
        self._client = google_genai.Client(api_key=api_key)

    def embed_query(self, text: str) -> list[float]:
        response = self._client.models.embed_content(
            model=self.model, contents=text
        )
        embedding = response.embeddings[0].values
        return list(embedding) if embedding else []


def build_rag_pipeline(
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_pwd: str,
    google_api_key: str,
    index_name: str = "issue_embeddings",
):
    """Instantiate VectorRetriever + GraphRAG pipeline.

    The retriever searches over Issue embeddings. The LLM also receives
    Commit summary context via the graph traversal in the Cypher query
    embedded in the return_properties configuration.
    """
    driver = _get_neo4j_driver(neo4j_uri, neo4j_user, neo4j_pwd)
    embedder = GeminiEmbedderWrapper(api_key=google_api_key)

    retriever = VectorRetriever(
        driver=driver,
        index_name=index_name,
        embedder=embedder,
        # Return Issue properties + relevant commit summaries via a traversal
        return_properties=[
            "number", "title", "body", "state", "url", "labels", "created"
        ],
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=google_api_key,
        temperature=0.2,
    )
    rag = GraphRAG(retriever=retriever, llm=llm)
    return rag


def _fetch_commit_context(driver, question: str, limit: int = 3) -> str:
    """Pull the most relevant commit summaries from Neo4j via fulltext search
    to augment the RAG answer with code-change context."""
    try:
        with driver.session() as session:
            result = session.run(
                """
                CALL db.index.fulltext.queryNodes('commit_summaries', $query)
                YIELD node, score
                WHERE score > 0
                RETURN node.sha AS sha, node.summary_text AS summary,
                       node.message AS message, node.timestamp AS ts
                ORDER BY score DESC
                LIMIT $limit
                """,
                query=question,
                limit=limit,
            )
            rows = result.data()
            if not rows:
                return ""
            parts = []
            for row in rows:
                sha_short = (row.get("sha") or "")[:8]
                msg = (row.get("message") or "").split("\n")[0][:100]
                summary = (row.get("summary") or "").strip()
                ts = row.get("ts", "")
                parts.append(
                    f"Commit {sha_short} ({ts}): {msg}\nSummary: {summary}"
                )
            return "\n\n---\n\n".join(parts)
    except Exception:
        return ""


def query_graphrag(
    question: str,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_pwd: str,
    google_api_key: str,
    top_k: int = 5,
) -> str:
    """Run the GraphRAG pipeline and return an answer string.

    Retrieval is two-pronged:
      1. VectorRetriever: cosine similarity on Issue embeddings.
      2. Fulltext search on Commit.summary_text / Commit.diff_text.
    Both contexts are merged into the prompt for the final LLM call.
    """
    rag = build_rag_pipeline(neo4j_uri, neo4j_user, neo4j_pwd, google_api_key)
    driver = _get_neo4j_driver(neo4j_uri, neo4j_user, neo4j_pwd)

    # Pull supplementary commit context
    commit_context = _fetch_commit_context(driver, question)

    # Build an augmented question that includes commit context if available
    augmented_question = question
    if commit_context:
        augmented_question = (
            f"{question}\n\n"
            f"[Additional context from recent code commits:]\n{commit_context}"
        )

    result = rag.search(
        query_text=augmented_question,
        retriever_config={"top_k": top_k},
        return_context=False,
    )
    return result.answer


# =============================================================================
#  SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding: 0.5rem 0 1rem 0;">
            <div style="font-size:2.4rem;">🕸️</div>
            <div style="font-size:1.1rem; font-weight:700; color:#00d4ff; letter-spacing:0.5px;">GraphRAG Explorer</div>
            <div style="font-size:0.72rem; color:rgba(255,255,255,0.4); margin-top:2px;">Neo4j · GitHub · Gemini</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-header">🔑 API Keys</div>', unsafe_allow_html=True)
    github_token = st.text_input(
        "GitHub Personal Access Token",
        type="password",
        placeholder="ghp_xxxxxxxxxxxxxxxxxxxx",
        help="Create at github.com → Settings → Developer settings → PAT",
    )
    google_api_key = st.text_input(
        "Google Gemini API Key",
        type="password",
        placeholder="Enter key…",
        help="Create at aistudio.google.com",
    )

    st.markdown('<div class="section-header">🗄️ Neo4j Connection</div>', unsafe_allow_html=True)
    neo4j_uri = st.text_input("Neo4j URI", value="neo4j://localhost:7687")
    neo4j_user = st.text_input("Username", value="neo4j")
    neo4j_pwd = st.text_input("Password", type="password", placeholder="password123")

    # ── Connectivity status ────────────────────────────────────────────────
    if neo4j_uri and neo4j_user and neo4j_pwd:
        ok, msg = _check_neo4j(neo4j_uri, neo4j_user, neo4j_pwd)
        dot = _status_dot("green") if ok else _status_dot("red")
        st.markdown(
            f'<div style="font-size:0.78rem; color:rgba(255,255,255,0.6); margin-top:4px;">'
            f'{dot}Neo4j: {"Connected" if ok else "Unreachable"}</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-header">📦 Target Repository</div>', unsafe_allow_html=True)
    repo_input = st.text_input(
        "GitHub Repository",
        value="neo4j/neo4j-graphrag-python",
        placeholder="owner/repo",
        help="Format: owner/repository-name",
    )
    n_issues = st.slider("Issues to ingest", min_value=5, max_value=30, value=15, step=5)
    max_commits = st.slider(
        "Max commits to backfill",
        min_value=25, max_value=500, value=BOOTSTRAP_MAX_COMMITS, step=25,
        help="Number of commits the bootstrap pipeline will process. Higher = more graph context, but takes longer.",
    )

    st.markdown("---")

    # ── Ingest Issues button ───────────────────────────────────────────────
    ingest_clicked = st.button("⚡ Ingest Repository Issues", use_container_width=True)

    if ingest_clicked:
        missing = []
        if not github_token:   missing.append("GitHub Token")
        if not google_api_key: missing.append("Google API Key")
        if not neo4j_uri:      missing.append("Neo4j URI")
        if not neo4j_user:     missing.append("Neo4j Username")
        if not neo4j_pwd:      missing.append("Neo4j Password")
        if not repo_input:     missing.append("Repository")

        if missing:
            st.error(f"⚠️ Missing fields: {', '.join(missing)}")
        else:
            success = run_ingestion(
                github_token=github_token,
                google_api_key=google_api_key,
                neo4j_uri=neo4j_uri,
                neo4j_user=neo4j_user,
                neo4j_pwd=neo4j_pwd,
                repo_full_name=repo_input,
                n_issues=n_issues,
            )
            if success:
                st.success(
                    f"🎉 Ingested **{st.session_state.issue_count}** issues from "
                    f"`{repo_input}` into Neo4j!"
                )

    # ── Bootstrap Codebase Graph button ───────────────────────────────────
    st.markdown("---")
    bootstrap_clicked = st.button("⚡ Bootstrap Codebase Graph", use_container_width=True)

    if bootstrap_clicked:
        missing = []
        if not github_token:   missing.append("GitHub Token")
        if not google_api_key: missing.append("Google API Key")
        if not neo4j_uri:      missing.append("Neo4j URI")
        if not neo4j_user:     missing.append("Neo4j Username")
        if not neo4j_pwd:      missing.append("Neo4j Password")
        if not repo_input:     missing.append("Repository")

        if missing:
            st.error(f"⚠️ Missing fields: {', '.join(missing)}")
        elif st.session_state.bootstrap_status == "running":
            st.warning("⏳ Bootstrap is already running in the background.")
        else:
            try:
                driver = _get_neo4j_driver(neo4j_uri, neo4j_user, neo4j_pwd)
                driver.verify_connectivity()
                enqueue_bootstrap(
                    driver=driver,
                    repo_full_name=repo_input,
                    github_token=github_token,
                    google_api_key=google_api_key,
                    max_commits=max_commits,
                )
                st.toast(
                    "🚀 Bootstrap started in the background! "
                    "The codebase graph will populate while you chat.",
                    icon="🌳",
                )
            except Exception as exc:
                st.error(f"❌ Could not start bootstrap: {exc}")

    # ── Bootstrap status indicator ─────────────────────────────────────────
    bstatus = st.session_state.bootstrap_status
    bdetail = st.session_state.bootstrap_detail
    bcommits = st.session_state.bootstrap_commits
    bfiles = st.session_state.bootstrap_files

    if bstatus != "idle":
        color_map = {
            "running": "blue", "completed": "green",
            "failed": "red", "idle": "yellow",
        }
        dot = _status_dot(color_map.get(bstatus, "yellow"))
        label_map = {
            "running": "⏳ Bootstrap running…",
            "completed": "✅ Bootstrap complete",
            "failed": "❌ Bootstrap failed",
        }
        label = label_map.get(bstatus, bstatus.title())
        st.markdown(
            f'<div class="metric-card" style="margin-top:8px;">'
            f'<div class="label">Codebase Bootstrap</div>'
            f'<div style="font-size:0.85rem;color:rgba(255,255,255,0.75);padding-top:2px;">'
            f'{dot}{label}</div>'
            f'<div style="font-size:0.7rem;color:rgba(255,255,255,0.4);margin-top:4px;">'
            f'{bdetail[:120] if bdetail else ""}</div>'
            f'<div style="font-size:0.7rem;color:rgba(255,255,255,0.35);margin-top:2px;">'
            f'Files: {bfiles} &nbsp;·&nbsp; Commits: {bcommits}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Issue ingestion status summary ────────────────────────────────────
    st.markdown("---")
    if st.session_state.ingestion_done:
        dot = _status_dot("green")
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="label">Graph Status</div>'
            f'<div class="value">{st.session_state.issue_count} Issues</div>'
            f'<div style="font-size:0.75rem;color:rgba(255,255,255,0.45);margin-top:4px;">'
            f'{dot}{st.session_state.repo_name}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        dot = _status_dot("yellow")
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="label">Graph Status</div>'
            f'<div style="font-size:0.85rem;color:rgba(255,255,255,0.5);padding-top:4px;">'
            f'{dot}No data ingested yet</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Pipeline info ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        """
        <div style="font-size:0.7rem; color:rgba(255,255,255,0.35); line-height:1.8;">
            <div style="margin-bottom:6px; font-weight:600; color:rgba(255,255,255,0.5);">
                Pipeline
            </div>
            <span class="pipeline-badge">PyGithub</span>Issues + Commits<br>
            <span class="pipeline-badge">Gemini</span>Embeddings + Summaries<br>
            <span class="pipeline-badge">Neo4j</span>Graph Store<br>
            <span class="pipeline-badge">VectorRAG</span>Issue Retrieval<br>
            <span class="pipeline-badge">FullText</span>Commit Search<br>
            <span class="pipeline-badge">Gemini 2.5</span>Generation
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
#  MAIN PANEL
# =============================================================================

st.markdown(
    '<h1 class="hero-title">🕸️ GraphRAG Explorer</h1>'
    '<p class="hero-sub">Query GitHub repository issues <em>and</em> commit history '
    'using a unified Neo4j knowledge graph powered by Google Gemini</p>',
    unsafe_allow_html=True,
)

# ── Architecture overview ──────────────────────────────────────────────────────
with st.expander("📐 Architecture Overview", expanded=False):
    cols = st.columns(6)
    steps = [
        ("1️⃣", "GitHub API", "Fetch issues & full commit history via PyGithub + REST"),
        ("2️⃣", "Gemini Embed", "Generate 3072-dim vectors for each issue body"),
        ("3️⃣", "Gemini Summarize", "LLM summaries of commit diffs stored on Commit nodes"),
        ("4️⃣", "Neo4j Graph", "Repo·Issue·User·Commit·File·Directory·Module nodes"),
        ("5️⃣", "VectorRetriever", "Cosine search on issue_embeddings + fulltext on commit_summaries"),
        ("6️⃣", "Gemini 2.5 Flash", "Augmented generation with Issues + Commit context"),
    ]
    for col, (num, title, desc) in zip(cols, steps):
        with col:
            st.markdown(
                f"""
                <div class="metric-card" style="text-align:center; min-height:120px;">
                    <div style="font-size:1.6rem;">{num}</div>
                    <div style="font-size:0.8rem; font-weight:600; color:#00d4ff; margin:4px 0;">{title}</div>
                    <div style="font-size:0.7rem; color:rgba(255,255,255,0.45);">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

st.markdown("---")

# ── Readiness check banner ─────────────────────────────────────────────────────
if not st.session_state.ingestion_done:
    st.info(
        "👈 **Get started**: Fill in your API keys and Neo4j credentials in the sidebar, "
        "then click **⚡ Ingest Repository Issues** to populate the knowledge graph. "
        "Optionally, click **⚡ Bootstrap Codebase Graph** to also backfill commits and "
        "file dependencies in the background. Once ingestion completes, ask any question below!"
    )
else:
    bstatus_label = ""
    if st.session_state.bootstrap_status == "running":
        bstatus_label = " &nbsp;·&nbsp; ⏳ Codebase bootstrap running in background"
    elif st.session_state.bootstrap_status == "completed":
        bstatus_label = (
            f" &nbsp;·&nbsp; 🌳 {st.session_state.bootstrap_commits} commits "
            f"+ {st.session_state.bootstrap_files} files indexed"
        )

    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, rgba(0,212,255,0.08), rgba(123,97,255,0.08));
            border: 1px solid rgba(0,212,255,0.25);
            border-radius: 12px;
            padding: 0.85rem 1.25rem;
            margin-bottom: 1.25rem;
            font-size: 0.88rem;
            color: rgba(255,255,255,0.8);
        ">
            ✅ Knowledge graph ready &nbsp;·&nbsp;
            <strong>{st.session_state.issue_count} issues</strong>
            from <code>{st.session_state.repo_name}</code> indexed
            {bstatus_label}
            &nbsp;·&nbsp; Ask anything below ↓
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Suggested questions ────────────────────────────────────────────────────────
if st.session_state.ingestion_done and not st.session_state.messages:
    st.markdown(
        '<div style="font-size:0.78rem; color:rgba(255,255,255,0.4); margin-bottom:8px;">💡 Try asking…</div>',
        unsafe_allow_html=True,
    )
    suggestion_cols = st.columns(3)
    suggestions = [
        "What are the recent bug reports?",
        "Summarise the most critical open issues",
        "Which issues mention performance problems?",
        "What files were changed most recently?",
        "What feature requests are open?",
        "Which commits touched authentication code?",
    ]
    for i, suggestion in enumerate(suggestions):
        with suggestion_cols[i % 3]:
            if st.button(suggestion, key=f"sugg_{i}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": suggestion})
                st.rerun()

# ── Chat history ───────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑‍💻" if msg["role"] == "user" else "🤖"):
        st.markdown(msg["content"])

# ── Chat input ─────────────────────────────────────────────────────────────────
user_input = st.chat_input(
    "Ask a question about the repository issues or codebase…",
    disabled=not st.session_state.ingestion_done,
)

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(user_input)

    if not google_api_key:
        answer = "❌ Google API Key is missing. Please enter it in the sidebar."
        with st.chat_message("assistant", avatar="🤖"):
            st.error(answer)
    elif not neo4j_uri or not neo4j_user or not neo4j_pwd:
        answer = "❌ Neo4j credentials are incomplete. Please check the sidebar."
        with st.chat_message("assistant", avatar="🤖"):
            st.error(answer)
    else:
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("🔍 Searching knowledge graph and generating answer…"):
                try:
                    answer = query_graphrag(
                        question=user_input,
                        neo4j_uri=neo4j_uri,
                        neo4j_user=neo4j_user,
                        neo4j_pwd=neo4j_pwd,
                        google_api_key=google_api_key,
                        top_k=5,
                    )
                except neo4j_exc.ServiceUnavailable:
                    answer = (
                        "❌ **Neo4j is unreachable.** "
                        "Ensure Docker is running and the container is healthy:\n\n"
                        "```bash\ndocker ps\ndocker logs neo4j-graphrag\n```"
                    )
                except neo4j_exc.AuthError:
                    answer = "❌ **Neo4j authentication failed.** Check your username and password."
                except Exception as exc:
                    tb = traceback.format_exc()
                    answer = (
                        f"❌ **An error occurred during retrieval:**\n\n"
                        f"```\n{type(exc).__name__}: {exc}\n```\n\n"
                        f"<details><summary>Full traceback</summary>\n\n"
                        f"```\n{tb}\n```\n\n</details>"
                    )
            st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})

# ── Footer ──────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="
        text-align: center;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(255,255,255,0.06);
        font-size: 0.7rem;
        color: rgba(255,255,255,0.2);
        letter-spacing: 0.5px;
    ">
        GraphRAG Explorer &nbsp;·&nbsp; Neo4j + PyGithub + Google Gemini &nbsp;·&nbsp;
        <a href="https://neo4j.com/docs/neo4j-graphrag-python/current/" target="_blank"
           style="color:rgba(0,212,255,0.5); text-decoration:none;">Docs</a>
    </div>
    """,
    unsafe_allow_html=True,
)