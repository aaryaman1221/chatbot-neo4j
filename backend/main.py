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


def _build_context_block(records: list[dict], source: str) -> str:
    """
    Format a list of retrieval records into clearly labelled plain-text sections.

    Each record is expected to have at minimum:
      - 'name' / 'func_name' — the function or symbol name
      - 'filepath'           — source file path
      - 'code'               — source code body

    Connected sub-nodes (if present under 'connected') are appended as
    sub-sections underneath their seed.
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

        if uid in seen_ids or not code:
            continue
        seen_ids.add(uid)

        lines.append(f"\n{'─'*60}")
        lines.append(f"Function : {name}")
        lines.append(f"File     : {filepath}")
        lines.append("Code     :")
        lines.append(code)

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
    3-stage retrieval pipeline:
      Stage 1 — Vector search (fixed Cypher: returns seed + connected code bodies).
      Stage 2 — Path-hint keyword boost (direct file-path lookup for components
                 explicitly named in the query — bypasses vector ranking).
      Stage 3 — Fulltext commit-summary fallback (only if stages 1+2 both fail).

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

CONTEXT:
{graph_context}

Instructions:
- Answer ONLY using the information shown in the context above. Never invent evidence.
- For questions about commits, list them with their SHA, repo, and summary.
- For questions about code, write like a senior engineer doing a code review explanation.
- Start with a "### Summary" section.
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
