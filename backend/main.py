import json
import logging
import os
import traceback
from functools import lru_cache
from typing import Optional, List

from fastapi import FastAPI, HTTPException, Header, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from dotenv import load_dotenv

load_dotenv()

from neo4j import GraphDatabase, exceptions as neo4j_exc

try:
    from google import genai as google_genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
import requests

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, _LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("graphrag")

app = FastAPI(title="GraphRAG API")

# Default covers the Vite dev server and common alt port.
_ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",")
    if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------

class ChatRequest(BaseModel):
    query: str
    selected_repos: Optional[List[str]] = None
    top_k: int = 5

class ConnectionRequest(BaseModel):
    uri: str
    user: str
    password: str

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def get_embedding(text: str, api_key: str) -> list:
    """Return a float vector for *text*. Logs the retrieval path taken."""
    if not text or not api_key:
        logger.warning("[EMBED] Skipped — text or api_key is empty.")
        return []

    if GENAI_AVAILABLE:
        logger.debug("[EMBED] Trying google-genai SDK (model=gemini-embedding-2) …")
        try:
            client = google_genai.Client(api_key=api_key)
            resp = client.models.embed_content(model="gemini-embedding-2", contents=text)
            vec = list(resp.embeddings[0].values)
            logger.info("[EMBED] ✅ google-genai SDK — vector dim=%d", len(vec))
            return vec
        except Exception as exc:
            logger.warning("[EMBED] ❌ google-genai SDK failed (%s) — trying REST fallback.", exc)
    else:
        logger.debug("[EMBED] google-genai SDK not available — going straight to REST.")

    # REST fallback
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2:embedContent?key={api_key}"
    payload = {"model": "models/gemini-embedding-2", "content": {"parts": [{"text": text}]}}
    try:
        logger.debug("[EMBED] Trying REST endpoint …")
        resp = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=30)
        resp.raise_for_status()
        vec = resp.json().get("embedding", {}).get("values", [])
        if vec:
            logger.info("[EMBED] ✅ REST fallback succeeded — vector dim=%d", len(vec))
        else:
            logger.warning("[EMBED] ❌ REST fallback returned empty embedding.")
        return vec
    except Exception as exc:
        logger.error("[EMBED] ❌ REST fallback also failed: %s", exc)
        return []

def _check_neo4j(uri: str, user: str, pwd: str) -> tuple[bool, str]:
    try:
        drv = GraphDatabase.driver(uri, auth=(user, pwd))
        drv.verify_connectivity()
        drv.close()
        return True, "Connected"
    except Exception as exc:
        return False, str(exc)

def _query_graph_stats(driver) -> dict:
    stats = {
        "nodes": 0, "relationships": 0,
        "commits": 0, "files": 0,
        "issues": 0, "repositories": 0,
        "modules": 0, "users": 0,
    }
    queries = {
        "nodes":         "MATCH (n) RETURN count(n) AS c",
        "relationships": "MATCH ()-[r]->() RETURN count(r) AS c",
        "commits":       "MATCH (c:Commit) RETURN count(c) AS c",
        "files":         "MATCH (f:File) RETURN count(f) AS c",
        "issues":        "MATCH (i:Issue) RETURN count(i) AS c",
        "repositories":  "MATCH (r:Repository) RETURN count(r) AS c",
        "modules":       "MATCH (m:Module) RETURN count(m) AS c",
        "users":         "MATCH (u:User) RETURN count(u) AS c",
    }
    try:
        with driver.session() as session:
            for key, cypher in queries.items():
                result = session.run(cypher)
                record = result.single()
                if record:
                    stats[key] = record["c"]
    except Exception as exc:
        logger.warning("Could not fetch graph stats: %s", exc)
    return stats

def _fetch_available_repos(driver) -> list:
    try:
        with driver.session() as session:
            result = session.run(
                "MATCH (r:Repository) RETURN r.full_name AS full_name ORDER BY r.full_name"
            )
            return [rec["full_name"] for rec in result if rec["full_name"]]
    except Exception as exc:
        logger.warning("Could not fetch available repos: %s", exc)
        return []

# -----------------------------------------------------------------------------
# GraphRAG Core — helpers
# -----------------------------------------------------------------------------

# Common Go/Python component path segments that indicate a specific file target.
_COMPONENT_KEYWORDS = [
    "help", "textinput", "textarea", "list", "paginator", "progress",
    "spinner", "viewport", "filepicker", "table", "cursor", "key",
    "styles", "style",
]

def _sanitize_lucene_query(query: str) -> str:
    """
    Strip or escape characters that Lucene's classic query parser treats as
    special syntax.  The most common offenders in repo queries are:
      /  — opens a regex literal (causes TokenMgrError on bare prefix like /hugo)
      ?  — single-char wildcard
      *  — multi-char wildcard at unexpected positions
      ~  — fuzzy / proximity operator
      ^  — boosting operator

    We replace each with a space so the remaining terms are still searched.
    """
    import re
    # Replace Lucene special chars with a space
    sanitized = re.sub(r'[/\\\?\*~\^\[\]{}()!]', ' ', query)
    # Collapse multiple spaces
    sanitized = re.sub(r'\s+', ' ', sanitized).strip()
    return sanitized or query  # never return empty string


def _extract_path_hints(query: str) -> List[str]:
    """
    Pull file-path fragments out of the user query.

    Recognises patterns like:
      - bubbles/help   bubbles/textinput
      - help.go        textinput.go
      - the help component

    Returns a deduplicated list of lowercase path segments to match against
    Function.filepath in the graph.
    """
    import re
    hints: list[str] = []

    # Explicit path separators: "bubbles/help", "charmbracelet/lipgloss"
    for m in re.finditer(r'[\w\-]+/[\w\-]+', query):
        segment = m.group().split("/")[-1].lower()
        hints.append(segment)

    # Bare ".go" filenames: "help.go", "textinput.go"
    for m in re.finditer(r'\b([\w\-]+)\.go\b', query, re.IGNORECASE):
        hints.append(m.group(1).lower())

    # Component keywords that appear as standalone words
    q_lower = query.lower()
    for kw in _COMPONENT_KEYWORDS:
        if re.search(rf'\b{re.escape(kw)}\b', q_lower):
            hints.append(kw)

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for h in hints:
        if h not in seen:
            seen.add(h)
            unique.append(h)
    return unique


# Regex that detects blame-style intent even when adverbs sit between
# 'who' and the verb (e.g. "who last changed", "who recently modified").
import re as _re_blame
_BLAME_PATTERN = _re_blame.compile(
    r'\bwho\b.{0,25}\b(added|wrote|changed|modified|introduced|created|implemented|broke|touched|owns|did)\b'
    r'|\bwhich commit\b'
    r'|\bwhen was\b'
    r'|\bwhen did\b'
    r'|\bgit blame\b'
    r'|\bblame\b'
    r'|\bwho is responsible\b'
    r'|\bwho owns\b',
    _re_blame.IGNORECASE,
)


def _extract_blame_hints(query: str) -> dict:
    """
    Extract function name, file name, and any quoted identifier from the query
    to drive the blame Cypher lookup.

    Returns a dict with keys:
      - 'func_hints': list of possible function/symbol names
      - 'file_hints': list of possible file path fragments
    """
    import re
    func_hints: list[str] = []
    file_hints: list[str] = []

    # Quoted identifiers: 'process_payment', "handleLogin", `myFunc`
    for m in re.finditer(r'[\'"`]([\w_]+)[\'"`]', query):
        func_hints.append(m.group(1))

    # CamelCase or snake_case words that look like function/class names
    for m in re.finditer(r'\b([a-z][a-z0-9]*(?:_[a-z0-9]+)+|[A-Z][a-zA-Z0-9]{2,})\b', query):
        word = m.group(1)
        # Skip common English filler words
        if word.lower() not in {
            "who", "what", "when", "where", "which", "how", "the", "this", "that",
            "was", "were", "has", "have", "did", "does", "added", "changed",
            "introduced", "modified", "created", "wrote",
        }:
            func_hints.append(word)

    # File extensions: auth.py, payment.go, user_service.ts
    for m in re.finditer(r'\b([\w_\-]+\.(?:py|go|ts|js|java|cpp|c|rb|rs|kt))\b', query, re.IGNORECASE):
        file_hints.append(m.group(1))
        func_hints.append(m.group(1).rsplit(".", 1)[0])  # stem as func hint too

    # Path-style fragments: auth/login, services/payment
    for m in re.finditer(r'\b([\w\-]+/[\w\-]+)\b', query):
        file_hints.append(m.group(1))

    # Deduplicate
    func_hints = list(dict.fromkeys(h for h in func_hints if len(h) > 2))
    file_hints = list(dict.fromkeys(file_hints))
    return {"func_hints": func_hints, "file_hints": file_hints}


def _build_context_block(records: list[dict], source: str, allow_no_code: bool = False) -> str:
    """
    Format a list of retrieval records into clearly labelled plain-text sections.

    Each record is expected to have at minimum:
      - 'name' / 'func_name' — the function or symbol name
      - 'filepath'           — source file path
      - 'code'               — source code body (may be absent for structural/impact records)

    Connected sub-nodes (if present under 'connected') are appended as
    sub-sections underneath their seed.

    When allow_no_code=True (e.g. impact analysis), records with no code body are still
    included as structural evidence (name + filepath only).
    """
    if not records:
        return ""

    lines: list[str] = [f"[Source: {source}]"]
    seen_ids: set[str] = set()

    for rec in records:
        name     = rec.get("name") or rec.get("func_name") or "<unknown>"
        filepath = rec.get("filepath") or rec.get("path") or "<unknown path>"
        code     = (rec.get("code") or "").strip()
        uid      = f"{filepath}::{name}"

        if uid in seen_ids:
            continue
        # Drop code-less records unless caller explicitly allows structural-only entries
        if not code and not allow_no_code:
            continue
        seen_ids.add(uid)

        lines.append(f"\n{'─'*60}")
        lines.append(f"Function : {name}")
        lines.append(f"File     : {filepath}")
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

        # Connected nodes returned by the expanded vector query
        for conn in rec.get("connected") or []:
            c_name = conn.get("name") or "<unknown>"
            c_path = conn.get("path") or conn.get("filepath") or ""
            c_code = (conn.get("code") or "").strip()
            c_uid  = f"{c_path}::{c_name}"
            if c_uid in seen_ids or not c_code:
                continue
            seen_ids.add(c_uid)
            lines.append(f"\n  ↳ Called/Dep: {c_name}  [{c_path}]")
            lines.append(f"  {c_code.replace(chr(10), chr(10)+'  ')}")

    return "\n".join(lines)


def _build_blame_context_block(records: list[dict]) -> str:
    """
    Format blame records (User → Commit → Function/File) into a clearly
    labelled plain-text section for the LLM.

    Each record is expected to have:
      - 'author'       : User.login
      - 'commit_sha'   : Commit.sha
      - 'commit_msg'   : Commit.message
      - 'committed_at' : Commit.timestamp
      - 'func_name'    : Function.name  (may be None for file-level hits)
      - 'filepath'     : Function.filepath or File.path
    """
    if not records:
        return ""

    lines: list[str] = ["[Source: blame-traversal]"]
    seen: set[str] = set()

    for rec in records:
        author      = rec.get("author") or "<unknown>"
        sha         = rec.get("commit_sha") or "<no sha>"
        msg         = (rec.get("commit_msg") or "").strip().splitlines()[0]  # first line only
        date        = rec.get("committed_at") or ""
        func_name   = rec.get("func_name") or ""
        filepath    = rec.get("filepath") or "<unknown path>"

        uid = f"{sha}::{filepath}::{func_name}"
        if uid in seen:
            continue
        seen.add(uid)

        lines.append(f"\n{'─'*60}")
        lines.append(f"Author     : {author}")
        lines.append(f"Commit SHA : {sha}")
        lines.append(f"Date       : {date}")
        lines.append(f"Message    : {msg}")
        if func_name:
            lines.append(f"Function   : {func_name}")
        lines.append(f"File       : {filepath}")

    return "\n".join(lines)


# -----------------------------------------------------------------------------
# GraphRAG Core — impact analysis helpers
# -----------------------------------------------------------------------------

# Keywords that signal a cross-package / migration impact question.
_IMPACT_KEYWORDS = (
    # movement / migration
    "move", "moved", "moving",
    "migrate", "migrated", "migration",
    "extract", "extracted",
    # modification / removal
    "update", "updated", "updating",
    "remove", "removed", "removing",
    "delete", "deleted", "deleting",
    "rename", "renamed", "renaming",
    "change", "changed", "changing",
    "modify", "modified", "modifying",
    "drop", "dropped", "dropping",
    "deprecate", "deprecated",
    # impact / dependency
    "affect", "affected", "affects",
    "impact", "impacts",
    "depend", "depends", "dependency", "dependencies",
    "import", "imports",
    "refactor", "split", "separate",
    "what breaks", "what will break", "what changes",
    "breaking", "breaks",
)

import re as _re_impact


def _extract_impact_subjects(query: str) -> List[str]:
    """
    Extract candidate package/module/symbol names from an impact query.

    Looks for:
      - Quoted strings: 'BlockStorageResourceData', "Event"
      - CamelCase identifiers (structs, types): BlockStorageResourceData
      - snake_case identifiers: resource_quota, orbiter-metering
      - Words following "package", "module", "struct", "class", "type" etc.
      - Names following modification verbs (remove/update/rename/delete X)
    Returns a deduplicated list of lowercase candidates.
    """
    candidates: list[str] = []

    # Quoted identifiers  e.g. 'BlockStorageResourceData' or `Event`
    for m in _re_impact.finditer(r'[\'"`]([\w_\-\.]+)[\'"`]', query):
        candidates.append(m.group(1).lower())

    # Words after type/struct/class/package/module/field keywords
    for m in _re_impact.finditer(
        r'\b(?:package|module|library|pkg|component|resource|struct|class|type|field|property|attribute)\s+([\w_\-\.]+)',
        query, _re_impact.IGNORECASE
    ):
        candidates.append(m.group(1).lower())

    # Names immediately following modification verbs
    # e.g. "update BlockStorageResourceData", "removing Event", "rename OldName"
    for m in _re_impact.finditer(
        r'\b(?:update|modify|remove|rename|delete|drop|deprecate|change|adding|removing)\s+([\w_\-\.]+)',
        query, _re_impact.IGNORECASE
    ):
        cand = m.group(1).lower()
        if len(cand) > 2:
            candidates.append(cand)

    # CamelCase identifiers — structs, types, classes (e.g. BlockStorageResourceData)
    for m in _re_impact.finditer(r'\b([A-Z][a-z]+(?:[A-Z][a-z0-9]*)+)\b', query):
        candidates.append(m.group(1).lower())

    # snake_case or hyphen-case (e.g. resource_quota, orbiter-metering)
    for m in _re_impact.finditer(r'\b([a-z][a-z0-9]+(?:[_\-][a-z0-9]+)+)\b', query):
        candidates.append(m.group(1).lower())

    # Also pick up bare words next to move/extract/migrate/split
    for m in _re_impact.finditer(
        r'\b(?:move|extract|migrate|split)\s+([\w_\-\.]+)',
        query, _re_impact.IGNORECASE
    ):
        cand = m.group(1).lower()
        if len(cand) > 2:
            candidates.append(cand)

    # Deduplicate preserving order, skip very short tokens and common stopwords
    _STOPWORDS = {
        "the", "from", "into", "what", "will", "get", "this", "that",
        "to", "in", "of", "a", "an", "and", "or", "if", "is", "are",
        "for", "at", "by", "be", "it", "on", "as", "do", "how",
        "field", "repo", "repository", "package", "struct", "class",
    }
    seen: set[str] = set()
    unique: list[str] = []
    for c in candidates:
        if c not in seen and c not in _STOPWORDS and len(c) > 2:
            seen.add(c)
            unique.append(c)
    return unique


def _extract_field_hints(query: str) -> List[str]:
    """
    Extract specific field/property/method names that the user says they are
    removing, renaming, or modifying — e.g. "removing Event Field" → ["event"].

    These are used for a targeted code-grep search inside consumer function bodies
    to pinpoint exactly which callers reference that field.
    """
    field_hints: list[str] = []

    # Patterns like "removing X field", "remove the X field", "delete X property"
    for m in _re_impact.finditer(
        r'\b(?:remov(?:e|ing)|delet(?:e|ing)|dropp?(?:ing)?|renam(?:e|ing)|updat(?:e|ing))\s+'
        r'(?:the\s+)?([A-Za-z_][\w_]*)\s+(?:field|property|attribute|column|param|parameter|arg|argument)',
        query, _re_impact.IGNORECASE
    ):
        field_hints.append(m.group(1).lower())

    # Patterns like "X field" or "field X" when near a modification verb in same sentence
    for m in _re_impact.finditer(
        r'\b([A-Z][a-z]+|[a-z][a-z0-9]+)\s+[Ff]ield\b',
        query
    ):
        word = m.group(1).lower()
        if word not in {"the", "a", "this", "that", "some", "any"}:
            field_hints.append(word)

    return list(dict.fromkeys(field_hints))  # deduplicate


def _extract_repo_hints_from_query(query: str) -> List[str]:
    """
    Extract repository names mentioned in the query.
    Recognises patterns like:
      - "orbiter-metering repo"
      - "the orbiter repo"
      - "owner/repo-name"
    Returns repo name fragments (lowercase) to use as additional graph filters.
    """
    repo_hints: list[str] = []

    # "owner/repo" pattern
    for m in _re_impact.finditer(r'\b([\w\-]+/[\w\-]+)\b', query):
        repo_hints.append(m.group(1).lower())

    # "<name> repo" or "<name> repository"
    for m in _re_impact.finditer(
        r'\b([\w][\w\-]+)\s+(?:repo|repository|codebase|service|module)\b',
        query, _re_impact.IGNORECASE
    ):
        candidate = m.group(1).lower()
        if len(candidate) > 2:
            repo_hints.append(candidate)

    return list(dict.fromkeys(repo_hints))


def _run_impact_traversal(
    driver,
    subjects: List[str],
    selected_repos: Optional[List[str]] = None,
) -> list[dict]:
    """
    Given a list of package/symbol names, traverse the existing graph to find
    every File and Function that depends on / imports / calls any of them.

    Works entirely on the existing graph — no re-ingestion required.

    Traversal order (broadest to narrowest):
      1. Exact or partial name match on Module / Package nodes
      2. Exact or partial filepath/path match on File nodes
      3. Walk outgoing DEPENDS_ON / IMPORTS / CALLS / DEFINED_IN / BELONGS_TO
         edges from those seed nodes to find all consumers.
      4. Return consumer File/Function records annotated with the relationship
         type and the repo they belong to.
    """
    results: list[dict] = []
    seen_ids: set[str] = set()

    for subject in subjects:
        subject_lower = subject.lower()

        # ── Strategy A: find any node whose name/path CONTAINS the subject ────
        #    Labels we search: Module, Package, File, Function
        #    We use toLower() for case-insensitive matching.
        seed_cypher = """
        MATCH (seed)
        WHERE (
            seed:Module OR seed:Package OR seed:File OR seed:Function
        )
        AND (
            toLower(coalesce(seed.name, ''))     CONTAINS $subject
            OR toLower(coalesce(seed.path, ''))  CONTAINS $subject
            OR toLower(coalesce(seed.full_name, '')) CONTAINS $subject
        )
        RETURN
          id(seed)                                         AS seed_id,
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
            # ── Strategy B: fulltext match on file paths even without Module nodes ──
            # Falls back to searching File nodes whose path contains the subject.
            fallback_cypher = """
            MATCH (f:File)
            WHERE toLower(f.path) CONTAINS $subject
            RETURN
              id(f)     AS seed_id,
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

            # ── Traverse: find consumers of this seed node ──────────────────
            # Consumers are nodes that have an incoming DEPENDS_ON / IMPORTS /
            # CALLS / USES edge pointing TO the seed, or that are connected via
            # outgoing edges from the seed to the callers.
            #
            # We use a variable-length path of depth 1-2 so we catch both
            # direct deps and one-hop transitive deps without exploding.
            repo_filter = (
                "AND consumer.repo IN $selected_repos"
                if selected_repos else ""
            )
            consumer_cypher = f"""
            MATCH (consumer)-[rel:DEPENDS_ON|IMPORTS|CALLS|USES|INCLUDES]->(seed)
            WHERE id(seed) = $seed_id
              AND (consumer:File OR consumer:Function OR consumer:Module)
              {repo_filter}
            RETURN
              consumer.name                                       AS name,
              coalesce(consumer.filepath, consumer.path, '')      AS filepath,
              coalesce(consumer.repo, '')                         AS repo,
              type(rel)                                           AS rel_type,
              coalesce(consumer.code, '')                         AS code
            LIMIT 30
            """
            try:
                with driver.session() as session:
                    consumer_rows = session.run(
                        consumer_cypher,
                        seed_id=seed_id,
                        selected_repos=selected_repos or [],
                    ).data()
                logger.info(
                    "[IMPACT] Seed id=%d (%s) → %d consumer(s).",
                    seed_id, seed_name, len(consumer_rows),
                )
            except Exception as exc:
                logger.warning("[IMPACT] Consumer traversal for seed id=%d failed: %s", seed_id, exc)
                consumer_rows = []

            # Also include the seed itself as a reference point
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
                uid = f"{row.get('filepath')}::{row.get('name')}"
                if uid not in seen_ids:
                    seen_ids.add(uid)
                    results.append({
                        "name":     row.get("name") or "<unknown>",
                        "filepath": row.get("filepath") or "",
                        "repo":     row.get("repo") or "",
                        "rel_type": row.get("rel_type") or "",
                        "code":     row.get("code") or "",
                        "connected": [],
                    })

            # ── Reverse direction: edges FROM seed to things it calls/uses ──
            # This helps answer "what does resourcequota itself depend on?"
            reverse_cypher = f"""
            MATCH (seed)-[rel:DEPENDS_ON|IMPORTS|CALLS|USES]->(dep)
            WHERE id(seed) = $seed_id
              AND (dep:File OR dep:Function OR dep:Module)
              {repo_filter}
            RETURN
              dep.name                                         AS name,
              coalesce(dep.filepath, dep.path, '')             AS filepath,
              coalesce(dep.repo, '')                           AS repo,
              'OUTBOUND_' + type(rel)                          AS rel_type,
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
                    "[IMPACT] Seed id=%d outbound deps → %d node(s).",
                    seed_id, len(dep_rows),
                )
            except Exception as exc:
                logger.warning("[IMPACT] Reverse traversal for seed id=%d failed: %s", seed_id, exc)
                dep_rows = []

            for row in dep_rows:
                uid = f"{row.get('filepath')}::{row.get('name')}"
                if uid not in seen_ids:
                    seen_ids.add(uid)
                    results.append({
                        "name":     row.get("name") or "<unknown>",
                        "filepath": row.get("filepath") or "",
                        "repo":     row.get("repo") or "",
                        "rel_type": row.get("rel_type") or "",
                        "code":     row.get("code") or "",
                        "connected": [],
                    })

    return results


# -----------------------------------------------------------------------------
# GraphRAG Core — main retrieval
# -----------------------------------------------------------------------------

def retrieve_code_context(
    user_query: str,
    driver,
    google_api_key: str,
    selected_repos: Optional[List[str]] = None,
    top_k: int = 5,
) -> str:
    """
    Multi-stage retrieval pipeline:
      Stage 0 — Dependency-impact traversal (move/affect/depend/import queries).
                 Finds all consumers/dependents of a named package or module
                 using the existing graph edges — no re-ingestion required.
      Stage 1 — Vector search (semantic — returns seed + connected code bodies).
                 Results are also reused by Stage 4 for blame resolution.
      Stage 2 — Path-hint keyword boost (direct file-path lookup for named components).
      Stage 3 — Fulltext commit-summary fallback (commit/history keywords).
      Stage 4 — Blame traversal (who added/changed — uses Stage 1 filepaths directly).

    Logs every decision point so you can see exactly which path was taken.
    """
    logger.info("[RETRIEVE] ── New retrieval ────────────────────────────")
    logger.info("[RETRIEVE] Query        : %r", user_query[:200])
    logger.info("[RETRIEVE] Selected repos: %s", selected_repos or "<all>")
    logger.info("[RETRIEVE] top_k        : %d", top_k)

    query_vector  = get_embedding(user_query, google_api_key)
    path_hints    = _extract_path_hints(user_query)
    context_parts: list[str] = []
    retrieval_path_tags: list[str] = []
    vector_rows:   list[dict] = []   # kept for Stage 4 semantic blame resolution

    # ── Stage 0: Dependency-impact traversal ─────────────────────────────────
    # Triggers when the query contains impact/migration/dependency keywords.
    # Uses existing graph edges — requires NO re-ingestion.
    _query_lower_s0 = user_query.lower()
    _is_impact_query = any(kw in _query_lower_s0 for kw in _IMPACT_KEYWORDS)
    if _is_impact_query:
        impact_subjects  = _extract_impact_subjects(user_query)
        field_hints_s0   = _extract_field_hints(user_query)
        query_repo_hints = _extract_repo_hints_from_query(user_query)

        # If the user didn't select repos in the UI but mentioned one in the query,
        # use those repo-name fragments as a soft filter for Stage 0.
        _effective_repos = selected_repos or []
        _repo_hint_filter: Optional[List[str]] = None
        if query_repo_hints and not _effective_repos:
            _repo_hint_filter = query_repo_hints  # e.g. ["orbiter-metering"]

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

            # ── If repo hints found in query, post-filter impact rows ──────
            # Keeps rows whose repo field CONTAINS any of the repo hint fragments.
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

            # ── Field-targeted code-grep ────────────────────────────────────
            # If the user mentioned a specific field being removed/renamed,
            # find every Function whose stored code contains that field name.
            # This surfaces exactly which callers will break.
            if field_hints_s0:
                for field in field_hints_s0:
                    repo_filter_fg = (
                        "AND fn.repo IN $selected_repos"
                        if _effective_repos else ""
                    )
                    field_grep_cypher = f"""
                    MATCH (fn:Function)
                    WHERE fn.code IS NOT NULL
                      AND toLower(fn.code) CONTAINS toLower($field_name)
                      {repo_filter_fg}
                    RETURN
                      fn.name                                AS name,
                      coalesce(fn.filepath, fn.path, '')     AS filepath,
                      coalesce(fn.repo, '')                  AS repo,
                      'REFERENCES_FIELD_' + $field_name      AS rel_type,
                      fn.code                                AS code
                    LIMIT 20
                    """
                    try:
                        with driver.session() as session:
                            field_rows = session.run(
                                field_grep_cypher,
                                field_name=field,
                                selected_repos=_effective_repos,
                            ).data()
                        logger.info(
                            "[RETRIEVE] Stage 0 field-grep '%s' → %d function(s) reference it.",
                            field, len(field_rows),
                        )
                        # Post-filter by repo hint if needed
                        if _repo_hint_filter:
                            field_rows = [
                                r for r in field_rows
                                if any(
                                    hint in (r.get("repo") or "").lower()
                                    or hint in (r.get("filepath") or "").lower()
                                    for hint in _repo_hint_filter
                                )
                            ]
                        impact_rows.extend(field_rows)
                    except Exception as exc:
                        logger.warning(
                            "[RETRIEVE] Stage 0 field-grep '%s' failed: %s", field, exc
                        )

            logger.info(
                "[RETRIEVE] Stage 0 returned %d impact record(s) (post-filter).", len(impact_rows)
            )
            if impact_rows:
                impact_block = _build_context_block(impact_rows, "dependency-impact", allow_no_code=True)
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
    # Fixed query:
    #   • Repo filter is in WITH clause, not after YIELD (avoids cross-repo pollution)
    #   • Returns DISTINCT seed + up to 4 connected nodes each WITH their code bodies
    #   • No more structural_path label-only strings — real code is returned
    if query_vector:
        repo_clause = (
            "AND node.repo IN $selected_repos"
            if selected_repos else ""
        )
        vector_cypher = f"""
        CALL db.index.vector.queryNodes('code_embeddings', $top_k, $query_vector)
        YIELD node, score
        WITH node, score
        WHERE node.code IS NOT NULL {repo_clause}
        OPTIONAL MATCH (node)-[:CALLS|DEPENDS_ON]->(connected)
        WHERE (connected:Function OR connected:File)
          AND connected.code IS NOT NULL
          {"AND connected.repo IN $selected_repos" if selected_repos else ""}
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
        logger.info(
            "[RETRIEVE] Stage 1 — vector search (top_k=%d, repo_filter=%s) …",
            top_k, bool(selected_repos),
        )
        try:
            with driver.session() as session:
                rows = session.run(
                    vector_cypher,
                    query_vector=query_vector,
                    top_k=top_k,
                    selected_repos=selected_repos or [],
                ).data()
            logger.info(
                "[RETRIEVE] Stage 1 returned %d seed record(s)  "
                "(each may carry up to 4 connected nodes).", len(rows),
            )
            if rows:
                vector_rows = rows          # expose to Stage 4
                context_parts.append(_build_context_block(rows, "vector-search"))
                retrieval_path_tags.append("vector")
        except Exception as exc:
            logger.error("[RETRIEVE] ❌ Stage 1 vector search FAILED: %s", exc)
    else:
        logger.warning(
            "[RETRIEVE] ⚠️  No query vector — skipping Stage 1 entirely."
        )

    # ── Stage 2: Path-hint keyword boost ─────────────────────────────────────
    # For any file-path fragments found in the query (e.g. "help", "textinput"),
    # directly fetch all Function nodes whose filepath contains that segment.
    # This catches downstream consumers that vector similarity misses.
    if path_hints:
        logger.info(
            "[RETRIEVE] Stage 2 — path-hint boost for: %s", path_hints
        )
        hint_rows: list[dict] = []
        for hint in path_hints:
            repo_filter_clause = (
                "AND fn.repo IN $selected_repos"
                if selected_repos else ""
            )
            hint_cypher = f"""
            MATCH (fn:Function)
            WHERE toLower(fn.filepath) CONTAINS toLower($hint)
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
            context_parts.append(_build_context_block(hint_rows, "path-hint-boost"))
            retrieval_path_tags.append("path-hint")
        else:
            logger.warning(
                "[RETRIEVE] ⚠️  Stage 2 path-hints produced 0 results. "
                "Likely cause: these files were not AST-parsed during ingest "
                "(run backend_ingest.py phase 2 for the relevant repos)."
            )
    else:
        logger.info("[RETRIEVE] Stage 2 skipped — no path hints in query.")

    # ── Stage 3: Commit retrieval ──────────────────────────────────────────────
    # Triggers on commit/history-related keywords in the query.
    # Stage 3a: Direct ORDER BY timestamp DESC — for "last N / recent" questions.
    # Stage 3b: Fulltext search — for broader commit-keyword augmentation.
    _COMMIT_KEYWORDS = (
        "commit", "commits", "merge", "push", "pr", "pull request",
        "issue", "bug", "fix", "release", "tag", "author", "contributor",
        "last", "recent", "latest", "history",
    )
    _RECENCY_KEYWORDS = ("last", "recent", "latest", "newest", "history")
    _query_lower = user_query.lower()
    _needs_commit_context = (
        not context_parts
        or any(kw in _query_lower for kw in _COMMIT_KEYWORDS)
    )
    _needs_recency_sort = any(kw in _query_lower for kw in _RECENCY_KEYWORDS)

    if _needs_commit_context:
        if not context_parts:
            logger.warning(
                "[RETRIEVE] ⚠️  Stages 1+2 both empty — will use commit context."
            )
        else:
            logger.info(
                "[RETRIEVE] Stage 3 — augmenting with commit context "
                "(query contains commit/history keywords)."
            )

        # ── Stage 3a: Recency-sorted direct Cypher ────────────────────────────
        # Extracts "owner/repo" pattern from the query to filter by repo.
        # Returns commits ordered by timestamp DESC so "last N" is correct.
        if _needs_recency_sort:
            import re as _re
            # Pull "owner/repo" patterns (e.g. "gohugoio/hugo") from the query
            _repo_matches = _re.findall(r'\b[\w\-]+/[\w\-]+\b', user_query)
            _repo_hint = _repo_matches[0] if _repo_matches else None

            # Also extract a numeric limit (e.g. "last 5" → 5)
            _limit_match = _re.search(r'\b(\d+)\b', user_query)
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
                # No specific repo — fetch most recent commits across all repos
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
                        _build_context_block(rows, "recent-commits")
                    )
                    retrieval_path_tags.append("recent-commits")
            except Exception as exc:
                logger.error("[RETRIEVE] ❌ Stage 3a recency FAILED: %s", exc)

        # ── Stage 3b: Fulltext search ──────────────────────────────────────────
        lucene_query = _sanitize_lucene_query(user_query)
        logger.info(
            "[RETRIEVE] Stage 3b — fulltext: sanitized=%r (original=%r)",
            lucene_query, user_query,
        )
        fulltext_cypher = """
        CALL db.index.fulltext.queryNodes('commit_summaries', $search_query)
        YIELD node, score
        RETURN
          coalesce(node.sha, node.id, toString(id(node))) AS name,
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
        try:
            with driver.session() as session:
                rows = session.run(
                    fulltext_cypher, search_query=lucene_query
                ).data()
            logger.info("[RETRIEVE] Stage 3b fulltext → %d record(s).", len(rows))
            if rows:
                context_parts.append(
                    _build_context_block(rows, "fulltext-commits")
                )
                retrieval_path_tags.append("fulltext")
        except Exception as exc:
            logger.error("[RETRIEVE] ❌ Stage 3b fulltext FAILED: %s", exc)

    # ── Stage 3c: Commit-SHA → Files/Functions lookup ─────────────────────────
    # Triggers when the query contains a hex SHA (≥7 chars) OR commit-file
    # intent keywords ("what files did commit X change / touch / modify").
    # Walks (Commit)-[:MODIFIED]->(File|Function) directly — no fulltext needed.
    import re as _re_sha
    _SHA_PATTERN = _re_sha.compile(r'\b([0-9a-f]{7,40})\b', _re_sha.IGNORECASE)
    _COMMIT_FILE_INTENT = _re_sha.compile(
        r'\b(what|which)\b.{0,30}\b(file|files|changed|modified|touched|affect)\b'
        r'|\b(file|files)\b.{0,20}\b(commit|sha|hash)\b'
        r'|\bcommit\b.{0,20}\b(change|changed|modify|modified|touch|touched|affect)\b'
        r'|\bshow\b.{0,20}\b(file|files)\b.{0,20}\bcommit\b',
        _re_sha.IGNORECASE,
    )
    _sha_matches   = _SHA_PATTERN.findall(user_query)
    _has_sha       = bool(_sha_matches)
    _has_file_intent = bool(_COMMIT_FILE_INTENT.search(user_query))

    if _has_sha or (_has_file_intent and "commit" in _query_lower):
        logger.info(
            "[RETRIEVE] Stage 3c — commit-file lookup triggered. "
            "sha_matches=%s  has_file_intent=%s",
            _sha_matches, _has_file_intent,
        )

        commit_file_rows: list[dict] = []

        # Helper: run both File-level and Function-level MODIFIED traversal for a SHA hint
        def _lookup_commit_files(sha_hint: str) -> list[dict]:
            rows_out: list[dict] = []

            # ── File nodes changed by this commit ──────────────────────────
            repo_filter_cf = (
                "AND r.full_name IN $selected_repos" if selected_repos else ""
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

            # ── Function nodes changed by this commit ──────────────────────
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

            # ── Commit metadata itself ─────────────────────────────────────
            # Always include the commit node so the LLM sees the full message.
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
                # Prepend meta so LLM sees commit context first
                rows_out = meta_result + rows_out
            except Exception as exc:
                logger.warning("[RETRIEVE] Stage 3c meta lookup sha=%r failed: %s", sha_hint, exc)

            return rows_out

        if _sha_matches:
            # Deduplicate SHAs (a query might mention the same SHA twice)
            seen_shas: set[str] = set()
            for sha in _sha_matches:
                if sha.lower() not in seen_shas:
                    seen_shas.add(sha.lower())
                    commit_file_rows.extend(_lookup_commit_files(sha))
        else:
            # No explicit SHA but clear file-intent — try fulltext to find relevant commits,
            # then traverse their MODIFIED edges.
            logger.info(
                "[RETRIEVE] Stage 3c — no SHA in query; will re-use fulltext commit results "
                "and traverse their MODIFIED edges."
            )
            # Re-run fulltext only if not already done above
            try:
                with driver.session() as session:
                    ft_rows = session.run(
                        fulltext_cypher, search_query=lucene_query
                    ).data()
                for ft_row in ft_rows[:3]:  # limit to top 3 commits to avoid explosion
                    sha_candidate = ft_row.get("name", "")
                    if sha_candidate and len(sha_candidate) >= 7:
                        commit_file_rows.extend(_lookup_commit_files(sha_candidate[:12]))
            except Exception as exc:
                logger.warning("[RETRIEVE] Stage 3c fallback fulltext→MODIFIED failed: %s", exc)

        if commit_file_rows:
            block = _build_context_block(commit_file_rows, "commit-file-lookup", allow_no_code=True)
            if block:
                context_parts.append(block)
                retrieval_path_tags.append("commit-files")
            logger.info(
                "[RETRIEVE] Stage 3c — %d commit-file record(s) added to context.",
                len(commit_file_rows),
            )
        else:
            logger.warning(
                "[RETRIEVE] ⚠️  Stage 3c returned 0 results. "
                "Check that (Commit)-[:MODIFIED]->(File|Function) edges exist in the graph. "
                "If not, re-run backend_ingest.py diff/file-linking phase."
            )
    else:
        logger.info("[RETRIEVE] Stage 3c skipped — no commit SHA or file-intent in query.")

    # ── Stage 4: Blame traversal ───────────────────────────────────────────────
    # Triggers on "who added / who changed / which commit / blame" style queries.
    # Walks the (User)-[:AUTHORED]->(Commit)-[:MODIFIED]->(Function|File) chain
    # and returns author + commit metadata alongside the code artefact.
    _is_blame_query = bool(_BLAME_PATTERN.search(user_query))

    if _is_blame_query:
        logger.info("[RETRIEVE] Stage 4 — blame traversal triggered.")

        # ── Semantic resolution: use exact filepaths from Stage 1 vector results
        # This is far more reliable than NLP hint extraction from free text.
        semantic_filepaths = list(dict.fromkeys(
            r["filepath"] for r in vector_rows if r.get("filepath")
        ))

        # ── Fallback: NLP hint extraction (only when vector search returned nothing)
        fallback_hints: list[str] = []
        if not semantic_filepaths:
            logger.info("[RETRIEVE] Stage 4 — no vector results, falling back to NLP hints.")
            blame_hints  = _extract_blame_hints(user_query)
            fallback_hints = blame_hints["func_hints"] + blame_hints["file_hints"]

        logger.info(
            "[RETRIEVE] Stage 4 semantic_filepaths=%s  fallback_hints=%s",
            semantic_filepaths, fallback_hints,
        )

        blame_rows: list[dict] = []

        # 4a — Semantic path: query by exact filepaths from Stage 1
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

            # Also try Function-level blame for the same filepath
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

        # 4b — Fallback: NLP hint-based lookup (only runs when Stage 1 had no results)
        for hint in fallback_hints:
            if selected_repos:
                blame_hint_cypher = """
                MATCH (u:User)-[:AUTHORED]->(c:Commit)-[:MODIFIED]->(f:File)
                WHERE toLower(f.path) CONTAINS toLower($hint)
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
                blame_hint_cypher = """
                MATCH (u:User)-[:AUTHORED]->(c:Commit)-[:MODIFIED]->(f:File)
                WHERE toLower(f.path) CONTAINS toLower($hint)
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

        # 4c — fallback: no semantic paths and no NLP hints → return top authors overall
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
                # Reformat to match blame schema
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
            context_parts.append(_build_blame_context_block(blame_rows))
            retrieval_path_tags.append("blame")
            logger.info("[RETRIEVE] Stage 4 — %d blame record(s) added to context.", len(blame_rows))
        else:
            logger.warning(
                "[RETRIEVE] ⚠️  Stage 4 blame traversal returned 0 results. "
                "Check that (User)-[:AUTHORED]->(Commit)-[:MODIFIED]->(Function/File) "
                "edges were created during ingest."
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
        return "No relevant context found in the knowledge graph."

    full_context = "\n\n".join(context_parts)
    retrieval_path = "+".join(retrieval_path_tags) or "none"
    logger.info(
        "[RETRIEVE] ✅ path=%-20s | context_chars=%d",
        retrieval_path, len(full_context),
    )
    logger.debug("[RETRIEVE] Context preview (first 3000 chars):\n%s", full_context[:3000])
    return full_context

def answer_question_hybrid(
    user_input: str,
    llm,
    driver,
    google_api_key: str,
    selected_repos: Optional[List[str]] = None,
    top_k: int = 5,
) -> dict:
    graph_context = retrieve_code_context(
        user_input, driver, google_api_key, selected_repos, top_k
    )

    context_is_empty = graph_context.strip() == "No relevant context found in the knowledge graph."

    system_prompt = f"""You are a senior software engineering assistant with deep knowledge of codebases, 
git history, issues, and repository structure. You have access to a knowledge graph that stores 
code functions, commits, files, issues, and repository metadata.

The context below is structured as labelled sections. Each section starts with a [Source: ...] tag.

Sections from "vector-search" or "path-hint-boost" contain function/code blocks:
  Function : <name>
  File     : <filepath>
  Code     : <source code>
  ↳ Called/Dep: <name> [<file>]   (optional — connected functions)

Sections from "fulltext-fallback" contain commit or issue records:
  Function : <sha or id>
  File     : <repo>
  Code     : <commit message / summary / diff>

Sections from "blame-traversal" contain git-blame style records linking authors to code changes:
  Author     : <GitHub login of the person who made the commit>
  Commit SHA : <the exact commit SHA>
  Date       : <ISO timestamp of the commit>
  Message    : <first line of the commit message>
  Function   : <name of the function that was modified>  (may be absent for file-level hits)
  File       : <file path that was changed>

Sections from "commit-file-lookup" contain the files and functions directly changed by a specific commit:
  Function : <file path OR function name that was changed>
  File     : <file path>
  Repo     : <repository the file belongs to>
  Relation : FILE_CHANGED_BY_COMMIT | FUNCTION_CHANGED_BY_COMMIT
  Code     : <source code if stored, or "<not stored — structural reference only>">
  The first record(s) in this section are the commit metadata itself (SHA, message, author).

Sections from "dependency-impact" contain graph-traversal results showing what depends on a package/module/struct/field:
  Function : <name of the consumer file, function, or module>
  File     : <filepath of the consumer>
  Repo     : <repository the consumer belongs to>  (may be absent)
  Relation : one of:
    - IMPORTS / DEPENDS_ON / CALLS / USES         → this consumer directly depends on the subject
    - OUTBOUND_DEPENDS_ON / OUTBOUND_CALLS         → the subject depends on this node
    - SUBJECT (being moved/refactored)             → this IS the struct/package being changed
    - REFERENCES_FIELD_<name>                      → this function's code contains the field name
                                                     being removed/renamed (highest-risk callers)
  Code     : <source code if available, or "<not stored — structural reference only>">

CONTEXT:
{graph_context}

Instructions:
- Answer ONLY using the information shown in the context above. Never invent evidence.
- For questions about commits, list them with their SHA, repo, and summary.
- For questions about code, write like a senior engineer doing a code review explanation.
- For blame / ownership questions (who added / who changed / which commit):
    * Lead with a clear statement of the author's name and commit SHA.
    * Include the date and the commit message so the manager has full context.
    * If multiple commits touched the same code, list them in reverse-chronological order.
    * Use the heading "### Ownership & Blame Summary" instead of the generic summary.
- For impact / migration / dependency questions (move package / remove field / what gets affected / what imports this):
    * Use the heading "### Migration Impact Analysis".
    * List the SUBJECT node first (the thing being moved or modified).
    * If Relation is REFERENCES_FIELD_<name>: these are the highest-risk callers — list them FIRST under
      "#### 🔴 Direct Field References (will break immediately)" and quote the line(s) of code that use the field.
    * Then list IMPORTS / DEPENDS_ON / CALLS consumers under "#### 🟡 Structural Dependents".
    * Group all results by repository — cross-repo dependencies are especially dangerous.
    * Note any entries where Code is "<not stored>" — these need manual verification.
    * Close with "### Recommended Action Items" listing concrete steps (update field references, bump version, run tests, etc.).
- For commit-file questions (what files did commit X change / touch / modify):
    * Use the heading "### Commit Change Summary".
    * Start with the commit metadata (SHA, author, timestamp, message) from the first record(s) in "commit-file-lookup".
    * Then list all changed FILES in a bullet list (Relation: FILE_CHANGED_BY_COMMIT).
    * Then list all changed FUNCTIONS in a separate bullet list (Relation: FUNCTION_CHANGED_BY_COMMIT), grouped by file.
    * If Code is "<not stored>" for some entries, still list the file/function name — its presence in the graph confirms it was modified.
    * If 0 file records were found but a commit was found, say the commit was recorded but no file-level MODIFIED edges exist yet.
- Start with a "### Summary" section (or the appropriate specialized heading for blame/impact/commit-file queries).
- Include "### Impact Analysis" ONLY when you can point to specific code that changes.
- End with "### Evidence" listing every function/commit SHA and file/repo you referenced, as bullet points.
- If the context truly contains no information relevant to the question (not just no code), say:
  "The graph does not contain sufficient data to answer this question. The relevant data may not have been ingested yet."
"""

    logger.info("[LLM] ── Invoking LLM ─────────────────────────────────")
    logger.info("[LLM] Context empty   : %s", context_is_empty)
    logger.info("[LLM] System prompt   : %d chars", len(system_prompt))
    logger.info("[LLM] User input      : %r", user_input[:200])
    logger.debug("[LLM] Full system prompt:\n%s", system_prompt[:4000])

    if context_is_empty:
        logger.warning(
            "[LLM] ⚠️  LLM is being called with EMPTY context. "
            "Response will be generic/unhelpful. Fix the retrieval pipeline above."
        )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_input)
    ]

    try:
        response = llm.invoke(messages)
        logger.info("[LLM] ✅ LLM responded — response_chars=%d", len(response.content))
        logger.debug("[LLM] Response content:\n%s", response.content[:2000])
        return {
            "answer": response.content,
            "usage": response.usage_metadata if hasattr(response, "usage_metadata") else None
        }
    except Exception as exc:
        logger.error("[LLM] ❌ LLM invocation failed: %s", exc)
        return {
            "answer": f"❌ Agent Error: {exc}",
            "usage": None
        }

# -----------------------------------------------------------------------------
# API Endpoints
# -----------------------------------------------------------------------------

@lru_cache(maxsize=16)
def _get_cached_driver(uri: str, user: str, password: str):
    """Return a long-lived, pooled Neo4j driver. Reused across requests with the same credentials."""
    logger.info("Creating new pooled Neo4j driver for %s", uri)
    return GraphDatabase.driver(uri, auth=(user, password))


def get_neo4j_driver(uri, user, password):
    if not all([uri, user, password]):
        raise HTTPException(status_code=400, detail="Neo4j credentials are required in headers")
    return _get_cached_driver(uri, user, password)

@app.get("/api/health")
def health_check():
    """Liveness probe — returns 200 OK when the server is up."""
    return {"status": "ok"}


@app.get("/api/stats")
def get_stats(
    x_neo4j_uri: str = Header(...),
    x_neo4j_user: str = Header(...),
    x_neo4j_password: str = Header(...)
):
    try:
        drv = get_neo4j_driver(x_neo4j_uri, x_neo4j_user, x_neo4j_password)
        return _query_graph_stats(drv)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/repos")
def get_repos(
    x_neo4j_uri: str = Header(...),
    x_neo4j_user: str = Header(...),
    x_neo4j_password: str = Header(...)
):
    try:
        drv = get_neo4j_driver(x_neo4j_uri, x_neo4j_user, x_neo4j_password)
        return {"repos": _fetch_available_repos(drv)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/check-connection")
def check_connection(req: ConnectionRequest):
    ok, msg = _check_neo4j(req.uri, req.user, req.password)
    if not ok:
        raise HTTPException(status_code=400, detail=msg)
    return {"status": "ok", "message": msg}

@app.post("/api/chat")
def chat(
    req: ChatRequest,
    x_neo4j_uri: str = Header(...),
    x_neo4j_user: str = Header(...),
    x_neo4j_password: str = Header(...),
):
    import time as _time
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise HTTPException(status_code=500, detail="GOOGLE_API_KEY not found in environment.")

    logger.info(
        "[CHAT] ════════════════════════════════════════════════════════"
    )
    logger.info("[CHAT] Incoming query   : %r", req.query[:200])
    logger.info("[CHAT] Selected repos   : %s", req.selected_repos or "<all>")
    logger.info("[CHAT] top_k            : %d", req.top_k)

    _t0 = _time.perf_counter()
    try:
        drv = get_neo4j_driver(x_neo4j_uri, x_neo4j_user, x_neo4j_password)
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=google_api_key, temperature=0.0)

        result = answer_question_hybrid(
            user_input=req.query,
            llm=llm,
            driver=drv,
            google_api_key=google_api_key,
            selected_repos=req.selected_repos,
            top_k=req.top_k,
        )

        answer = result["answer"]
        usage = result["usage"]

        elapsed = _time.perf_counter() - _t0
        logger.info("[CHAT] ✅ Done in %.2fs — answer_chars=%d", elapsed, len(answer))
        return {"answer": answer, "usage": usage}
    except Exception as e:
        elapsed = _time.perf_counter() - _t0
        logger.error("[CHAT] ❌ Failed after %.2fs: %s", elapsed, e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
