# =============================================================================
# backend/app/config.py — Configuration, Logging, Models & Database Pooling
# =============================================================================

import logging
import os
from pathlib import Path
from functools import lru_cache
from typing import Optional, List

try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
    else:
        load_dotenv()
except ImportError:
    pass

from fastapi import HTTPException
from pydantic import BaseModel
from neo4j import GraphDatabase

def get_google_api_key() -> str:
    """Return Google/Gemini API key checking multiple standard environment variable names."""
    return (
        os.getenv("GOOGLE_API_KEY")
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_GENAI_API_KEY")
        or os.getenv("GENAI_API_KEY")
        or ""
    )

# ── Google GenAI SDK Availability ────────────────────────────────────────────
try:
    from google import genai as google_genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

# ── Logging Setup ────────────────────────────────────────────────────────────
_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, _LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("graphrag")

# ── CORS Configuration ───────────────────────────────────────────────────────
ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",")
    if o.strip()
]

# ── Neo4j Connection Defaults (server-side fallback) ─────────────────────────
# When the frontend omits the X-Neo4j-* headers — e.g. the deployed site, where
# credentials are injected from Key Vault — the API falls back to these env vars.
NEO4J_URI = os.getenv("NEO4J_URI", "")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_PRECONFIGURED = bool(NEO4J_URI and NEO4J_PASSWORD)

# ── Pydantic Request Models ──────────────────────────────────────────────────
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    query: str
    selected_repos: Optional[List[str]] = None
    top_k: int = 5
    chat_history: Optional[List[ChatMessage]] = None


class ConnectionRequest(BaseModel):
    uri: Optional[str] = None
    user: Optional[str] = None
    password: Optional[str] = None

# ── Neo4j Database Helpers & Connection Pooling ──────────────────────────────
@lru_cache(maxsize=16)
def _get_cached_driver(uri: str, user: str, password: str):
    """Return a long-lived, pooled Neo4j driver. Reused across requests with the same credentials."""
    logger.info("Creating new pooled Neo4j driver for %s", uri)
    return GraphDatabase.driver(uri, auth=(user, password))


def resolve_neo4j_creds(uri: Optional[str], user: Optional[str], password: Optional[str]):
    """Merge caller-supplied Neo4j credentials with the server-side env defaults.

    Each field falls back to its ``NEO4J_*`` environment value when the caller
    passes nothing (blank header or missing request-body field). This lets the
    deployed site work with zero user input while local dev can still override
    per request via the X-Neo4j-* headers.
    """
    resolved_uri = (uri or "").strip() or NEO4J_URI
    resolved_user = (user or "").strip() or NEO4J_USER
    resolved_password = (password or "") or NEO4J_PASSWORD
    return resolved_uri, resolved_user, resolved_password


def get_neo4j_driver(uri: str, user: str, password: str):
    uri, user, password = resolve_neo4j_creds(uri, user, password)
    if not all([uri, user, password]):
        raise HTTPException(
            status_code=400,
            detail="Neo4j credentials are required (via X-Neo4j-* headers or server configuration)",
        )
    return _get_cached_driver(uri, user, password)


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
        "types": 0, "variables": 0, "directives": 0,
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
        "types":         "MATCH (t:Type) RETURN count(t) AS c",
        "variables":     "MATCH (v:Variable) RETURN count(v) AS c",
        "directives":    "MATCH (d:Directive) RETURN count(d) AS c",
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
