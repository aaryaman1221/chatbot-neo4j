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

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="GraphRAG API")

# Configurable allowed origins — set ALLOWED_ORIGINS=http://a.com,http://b.com in .env
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
    if not text or not api_key: return []
    if GENAI_AVAILABLE:
        try:
            client = google_genai.Client(api_key=api_key)
            resp = client.models.embed_content(model="gemini-embedding-2", contents=text)
            return list(resp.embeddings[0].values)
        except Exception as exc:
            logger.debug("google-genai embed failed: %s", exc)

    # REST fallback
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2:embedContent?key={api_key}"
    payload = {"model": "models/gemini-embedding-2", "content": {"parts": [{"text": text}]}}
    try:
        resp = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=30)
        resp.raise_for_status()
        return resp.json().get("embedding", {}).get("values", [])
    except Exception as exc:
        logger.debug("REST embedding failed: %s", exc)
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
# GraphRAG Core
# -----------------------------------------------------------------------------

def retrieve_code_context(
    user_query: str,
    driver,
    google_api_key: str,
    selected_repos: Optional[List[str]] = None,
    top_k: int = 5,
) -> str:
    query_vector = get_embedding(user_query, google_api_key)
    results = []

    if query_vector:
        repo_filter = "WHERE node.repo IN $selected_repos" if selected_repos else ""
        vector_cypher = f"""
        CALL db.index.vector.queryNodes('code_embeddings', $top_k, $query_vector)
        YIELD node, score
        {repo_filter}
        MATCH path = (node)-[:CALLS|DEPENDS_ON*1..2]-(connected)
        RETURN [n IN nodes(path) | labels(n)[0] + ' ' + coalesce(n.name, n.path, '')] AS structural_path,
               node.code AS source_code
        LIMIT 20
        """
        try:
            with driver.session() as session:
                results = session.run(
                    vector_cypher,
                    query_vector=query_vector,
                    top_k=top_k,
                    selected_repos=selected_repos or [],
                ).data()
        except Exception as exc:
            logger.error("Vector search failed: %s", exc)

    # Fulltext fallback — triggered when nodes have no embeddings yet or vector returns nothing
    if not results:
        logger.info("Vector search returned no results — falling back to fulltext index.")
        fulltext_cypher = """
        CALL db.index.fulltext.queryNodes('commit_summaries', $search_query)
        YIELD node, score
        RETURN node.summary_text AS structural_path, node.diff_text AS source_code
        LIMIT 10
        """
        try:
            with driver.session() as session:
                results = session.run(fulltext_cypher, search_query=user_query).data()
        except Exception as exc:
            logger.warning("Fulltext fallback also failed: %s", exc)

    if not results:
        return "No relevant context found in the knowledge graph."

    return json.dumps(results, indent=2)

def answer_question_hybrid(
    user_input: str,
    llm,
    driver,
    google_api_key: str,
    selected_repos: Optional[List[str]] = None,
    top_k: int = 5,
) -> str:
    graph_context = retrieve_code_context(
        user_input, driver, google_api_key, selected_repos, top_k
    )

    system_prompt = f"""
    You are a senior backend architect analyzing a codebase.

    Here is the relevant sub-graph architecture retrieved from the database:
    {graph_context}

    Answer the user's question based ONLY on this architecture.
    Write like a senior engineer explaining a codebase.
    Start with a "### Summary" section.
    Include a "### Impact Analysis" ONLY if there are tangible impacts. DO NOT list empty impacts.
    End with a "### Evidence" section listing the specific files, functions, or modules referenced in your answer as bullet points.
    Never invent evidence.
    If the database results are empty, strictly reply that you do not have the data.
    """

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_input)
    ]

    try:
        response = llm.invoke(messages)
        return response.content
    except Exception as exc:
        return f"❌ Agent Error: {exc}"

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
    x_neo4j_password: str = Header(...)
):
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise HTTPException(status_code=500, detail="GOOGLE_API_KEY not found in environment.")

    try:
        drv = get_neo4j_driver(x_neo4j_uri, x_neo4j_user, x_neo4j_password)
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=google_api_key, temperature=0.0)

        answer = answer_question_hybrid(
            user_input=req.query,
            llm=llm,
            driver=drv,
            google_api_key=google_api_key,
            selected_repos=req.selected_repos,
            top_k=req.top_k,
        )

        return {"answer": answer}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
