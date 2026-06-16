#!/usr/bin/env python3
# =============================================================================
# frontend_chat.py — Streamlit GraphRAG Chat UI
# =============================================================================
#
# PURPOSE:
#   A clean, lightweight Streamlit chatbot that queries a pre-built Neo4j
#   GraphRAG knowledge graph.  Does NO data ingestion — run backend_ingest.py
#   first to populate the database.
#
# RUN:
#   streamlit run frontend_chat.py
#
# REQUIREMENTS:
#   pip install streamlit neo4j neo4j-graphrag[google-genai] \
#               google-genai langchain-google-genai
#
# SHARED SCHEMA (must match backend_ingest.py):
#   Nodes   : Repository · File · Directory · Module · Commit · User · Issue
#   Indexes : issue_embeddings (vector) · commit_summaries (fulltext)
#
# =============================================================================

import logging
import traceback
from typing import Optional

import streamlit as st
from neo4j import GraphDatabase, exceptions as neo4j_exc
from langchain_core.prompts import PromptTemplate

# ── Neo4j GraphRAG ────────────────────────────────────────────────────────────
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_google_genai import ChatGoogleGenerativeAI

# ── Google Gemini (google-genai) ──────────────────────────────────────────────
try:
    from google import genai as google_genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

# ── LangChain Gemini bridge ───────────────────────────────────────────────────
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    LANGCHAIN_GENAI_AVAILABLE = True
except ImportError:
    LANGCHAIN_GENAI_AVAILABLE = False

logger = logging.getLogger(__name__)


# =============================================================================
#  PAGE CONFIG & GLOBAL STYLES
# =============================================================================

st.set_page_config(
    page_title="GraphRAG Chat — Neo4j + Gemini",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* ── Background ──────────────────────────────────────── */
    .stApp {
        background: linear-gradient(135deg, #0d0d1a 0%, #0a1628 50%, #0d0d1a 100%);
        min-height: 100vh;
    }

    /* ── Sidebar ─────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1923 0%, #0d1520 100%);
        border-right: 1px solid rgba(0, 212, 255, 0.15);
    }
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #00d4ff;
    }

    /* ── Hero title ──────────────────────────────────────── */
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

    /* ── Metric cards ────────────────────────────────────── */
    .metric-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(0, 212, 255, 0.2);
        border-radius: 12px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
        transition: border-color 0.2s ease;
    }
    .metric-card:hover {
        border-color: rgba(0, 212, 255, 0.4);
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
    .metric-card .sub {
        font-size: 0.72rem;
        color: rgba(255,255,255,0.35);
        margin-top: 0.2rem;
    }

    /* ── Status dot ──────────────────────────────────────── */
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

    /* ── Pipeline badge ──────────────────────────────────── */
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

    /* ── Section header ──────────────────────────────────── */
    .section-header {
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: rgba(255,255,255,0.35);
        margin: 1.25rem 0 0.6rem 0;
    }

    /* ── Chat messages ───────────────────────────────────── */
    [data-testid="stChatMessage"] {
        background: rgba(255,255,255,0.03) !important;
        border: 1px solid rgba(255,255,255,0.07) !important;
        border-radius: 14px !important;
        margin-bottom: 0.75rem !important;
        transition: border-color 0.2s ease !important;
    }
    [data-testid="stChatMessage"]:hover {
        border-color: rgba(255,255,255,0.12) !important;
    }

    /* ── Buttons ─────────────────────────────────────────── */
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

    /* ── Inputs ──────────────────────────────────────────── */
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

    /* ── Divider glow ────────────────────────────────────── */
    hr {
        border: none !important;
        height: 1px !important;
        background: linear-gradient(90deg, transparent, rgba(0,212,255,0.4), transparent) !important;
        margin: 1.5rem 0 !important;
    }

    /* ── Code ────────────────────────────────────────────── */
    code {
        background: rgba(0, 212, 255, 0.08) !important;
        color: #00d4ff !important;
        border-radius: 4px !important;
        padding: 0.1rem 0.35rem !important;
        font-size: 0.85em !important;
    }

    /* ── Graph stats grid ────────────────────────────────── */
    .stats-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.6rem;
        margin-bottom: 1rem;
    }
    .stat-item {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(0, 212, 255, 0.15);
        border-radius: 8px;
        padding: 0.6rem 0.8rem;
        text-align: center;
    }
    .stat-item .stat-value {
        font-size: 1.2rem;
        font-weight: 700;
        color: #00d4ff;
        display: block;
    }
    .stat-item .stat-label {
        font-size: 0.65rem;
        color: rgba(255,255,255,0.4);
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }

    /* ── Suggestion chips ────────────────────────────────── */
    .suggestion-chip {
        display: inline-block;
        background: rgba(123,97,255,0.12);
        border: 1px solid rgba(123,97,255,0.3);
        border-radius: 20px;
        padding: 0.3rem 0.8rem;
        font-size: 0.78rem;
        color: rgba(255,255,255,0.7);
        cursor: pointer;
        margin: 0.2rem;
        transition: all 0.2s ease;
    }
    .suggestion-chip:hover {
        background: rgba(123,97,255,0.25);
        border-color: rgba(123,97,255,0.6);
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
#  SESSION STATE
# =============================================================================

def _init_session():
    defaults = {
        "messages":     [],
        "neo4j_driver": None,
        "graph_stats":  None,   # cached { nodes: int, rels: int, ... }
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

_init_session()


# =============================================================================
#  NEO4J HELPERS
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


def _get_driver(uri: str, user: str, pwd: str):
    """Return a cached Neo4j driver (creates one if needed)."""
    if st.session_state.neo4j_driver is None:
        st.session_state.neo4j_driver = GraphDatabase.driver(uri, auth=(user, pwd))
    return st.session_state.neo4j_driver


def _close_driver():
    if st.session_state.neo4j_driver is not None:
        try:
            st.session_state.neo4j_driver.close()
        except Exception:
            pass
        st.session_state.neo4j_driver = None


def _query_graph_stats(driver) -> dict:
    """
    Return graph statistics from Neo4j:
      nodes, relationships, commits, files, issues, repositories
    Gracefully returns zeros on any error.
    """
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


def _fetch_commit_context(driver, question: str, limit: int = 3) -> str:
    """Full-text search on commit_summaries index for additional RAG context."""
    try:
        with driver.session() as session:
            result = session.run(
                """
                CALL db.index.fulltext.queryNodes('commit_summaries', $query)
                YIELD node, score
                WHERE score > 0
                RETURN node.sha       AS sha,
                       node.summary_text AS summary,
                       node.message   AS message,
                       node.timestamp AS ts
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
                msg       = (row.get("message") or "").split("\n")[0][:100]
                summary   = (row.get("summary") or "").strip()
                ts        = row.get("ts", "")
                parts.append(f"Commit {sha_short} ({ts}): {msg}\nSummary: {summary}")
            return "\n\n---\n\n".join(parts)
    except Exception:
        return ""


# =============================================================================
#  CYPHER QA PIPELINE
# =============================================================================

def query_graph_cypher(
    question: str,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_pwd: str,
    google_api_key: str,
) -> str:
    """Translates natural language to Cypher, executes it, and returns the LLM's answer."""

    CYPHER_GENERATION_TEMPLATE = """Task: Generate a Cypher statement to query a graph database.
    Instructions:
    1. Use ONLY the provided relationship types and properties in the schema.
    2. Ensure your Cypher is syntactically valid for Neo4j v5.
    3. CRITICAL: A 'WHERE' clause must immediately follow a 'MATCH', 'OPTIONAL MATCH', 'YIELD', or 'WITH' clause. Never place 'WHERE' randomly.
    4. Do not include any explanations, apologies, or markdown formatting (like ```cypher). Return ONLY the raw query string.

    Schema:
    {schema}

    The question is:
    {question}"""

    cypher_prompt = PromptTemplate(
        template=CYPHER_GENERATION_TEMPLATE,
        input_variables=["schema", "question"]
    )
    
    # 1. Connect directly to the Neo4j Graph to read the schema
    graph = Neo4jGraph(
        url=neo4j_uri,
        username=neo4j_user,
        password=neo4j_pwd
    )

    # 2. Initialize the Gemini LLM
    # Temperature is 0.0 because generating Cypher requires strict deterministic logic
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=google_api_key,
        temperature=0.0 
    )

    # 3. Create the Cypher QA Chain
    chain = GraphCypherQAChain.from_llm(
        cypher_llm=llm,       
        qa_llm=llm,           
        graph=graph,          
        verbose=True,         # Prints the generated Cypher to your terminal!
        cypher_prompt=cypher_prompt,
        allow_dangerous_requests=True, 
        validate_cypher=True  
    )

    # 4. Execute the chain
    try:
        response = chain.invoke({"query": question})
        return response.get("result", "I couldn't formulate an answer based on the database results.")
    except ValueError as exc:
        if "No tools" in str(exc) or "OutputParserException" in str(exc):
            return "❌ **Query Generation Failed:** The LLM couldn't map that question to the database schema."
        raise exc
    except Exception as exc:
        return f"❌ **Query Execution Failed:**\n```\n{str(exc)}\n```"


# =============================================================================
#  SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding: 0.5rem 0 1rem 0;">
            <div style="font-size:2.4rem;">🕸️</div>
            <div style="font-size:1.1rem; font-weight:700; color:#00d4ff; letter-spacing:0.5px;">GraphRAG Chat</div>
            <div style="font-size:0.72rem; color:rgba(255,255,255,0.4); margin-top:2px;">Neo4j · Gemini · Query-Only</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Neo4j credentials ─────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🗄️ Neo4j Connection</div>', unsafe_allow_html=True)
    neo4j_uri  = st.text_input("Neo4j URI",  value="neo4j://localhost:7687", key="uri")
    neo4j_user = st.text_input("Username",   value="neo4j",                   key="usr")
    neo4j_pwd  = st.text_input("Password",   type="password", placeholder="password123", key="pwd")

    # ── Gemini API key ────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🔑 Google Gemini</div>', unsafe_allow_html=True)
    google_api_key = st.text_input(
        "Gemini API Key",
        type="password",
        placeholder="Enter key…",
        help="Get yours at aistudio.google.com",
        key="gkey",
    )

    # ── Connection status ─────────────────────────────────────────────────────
    if neo4j_uri and neo4j_user and neo4j_pwd:
        ok, msg = _check_neo4j(neo4j_uri, neo4j_user, neo4j_pwd)
        dot = _status_dot("green") if ok else _status_dot("red")
        status_label = "Connected" if ok else "Unreachable"
        st.markdown(
            f'<div style="font-size:0.78rem; color:rgba(255,255,255,0.6); margin-top:4px;">'
            f'{dot}Neo4j: {status_label}</div>',
            unsafe_allow_html=True,
        )

        # ── Load graph stats on connect ────────────────────────────────────────
        if ok and st.session_state.graph_stats is None:
            try:
                drv = _get_driver(neo4j_uri, neo4j_user, neo4j_pwd)
                st.session_state.graph_stats = _query_graph_stats(drv)
            except Exception:
                st.session_state.graph_stats = {}

    # ── Refresh stats button ──────────────────────────────────────────────────
    if st.button("🔄 Refresh Graph Stats", use_container_width=True, key="refresh_stats"):
        if neo4j_uri and neo4j_user and neo4j_pwd:
            try:
                # Force a new driver on refresh
                _close_driver()
                drv = _get_driver(neo4j_uri, neo4j_user, neo4j_pwd)
                st.session_state.graph_stats = _query_graph_stats(drv)
                st.toast("Graph stats refreshed!", icon="✅")
            except Exception as exc:
                st.error(f"Could not refresh stats: {exc}")

    # ── Graph metrics display ─────────────────────────────────────────────────
    stats = st.session_state.graph_stats
    if stats:
        st.markdown('<div class="section-header">📊 Graph Metrics</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="stats-grid">
              <div class="stat-item">
                <span class="stat-value">{stats.get('nodes', 0):,}</span>
                <span class="stat-label">Nodes</span>
              </div>
              <div class="stat-item">
                <span class="stat-value">{stats.get('relationships', 0):,}</span>
                <span class="stat-label">Edges</span>
              </div>
              <div class="stat-item">
                <span class="stat-value">{stats.get('commits', 0):,}</span>
                <span class="stat-label">Commits</span>
              </div>
              <div class="stat-item">
                <span class="stat-value">{stats.get('files', 0):,}</span>
                <span class="stat-label">Files</span>
              </div>
              <div class="stat-item">
                <span class="stat-value">{stats.get('issues', 0):,}</span>
                <span class="stat-label">Issues</span>
              </div>
              <div class="stat-item">
                <span class="stat-value">{stats.get('modules', 0):,}</span>
                <span class="stat-label">Modules</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── RAG settings ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">⚙️ RAG Settings</div>', unsafe_allow_html=True)
    top_k = st.slider("Top-K retrieval results", min_value=1, max_value=15, value=5, step=1)

    st.markdown("---")

    # ── Clear chat button ─────────────────────────────────────────────────────
    if st.button("🗑️ Clear Chat History", use_container_width=True, key="clear_chat"):
        st.session_state.messages = []
        st.rerun()

    # ── Pipeline info ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        """
        <div style="font-size:0.7rem; color:rgba(255,255,255,0.35); line-height:1.8;">
            <div style="margin-bottom:6px; font-weight:600; color:rgba(255,255,255,0.5);">Pipeline</div>
            <span class="pipeline-badge">VectorRAG</span>Issue similarity<br>
            <span class="pipeline-badge">FullText</span>Commit search<br>
            <span class="pipeline-badge">Gemini 2.5</span>Answer generation<br>
            <br>
            <div style="color:rgba(255,255,255,0.25); font-size:0.65rem; margin-top:4px;">
                Run <code>backend_ingest.py</code> to populate the graph before chatting.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
#  MAIN PANEL
# =============================================================================

st.markdown(
    '<h1 class="hero-title">🕸️ GraphRAG Chat</h1>'
    '<p class="hero-sub">Query your Neo4j knowledge graph with natural language, '
    'powered by Google Gemini</p>',
    unsafe_allow_html=True,
)

# ── Architecture overview ─────────────────────────────────────────────────────
with st.expander("📐 Pipeline Overview", expanded=False):
    cols = st.columns(5)
    steps = [
        ("1️⃣", "Neo4j Graph",       "Pre-built by backend_ingest.py"),
        ("2️⃣", "VectorRetriever",   "Cosine search on Issue.embedding"),
        ("3️⃣", "Fulltext Search",   "Commit summaries & diff context"),
        ("4️⃣", "Context Merge",     "Issues + Commits fused into prompt"),
        ("5️⃣", "Gemini 2.5 Flash",  "Generates grounded answer"),
    ]
    for col, (num, title, desc) in zip(cols, steps):
        with col:
            st.markdown(
                f"""
                <div class="metric-card" style="text-align:center; min-height:110px;">
                    <div style="font-size:1.5rem;">{num}</div>
                    <div style="font-size:0.8rem; font-weight:600; color:#00d4ff; margin:4px 0;">{title}</div>
                    <div style="font-size:0.7rem; color:rgba(255,255,255,0.45);">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

st.markdown("---")

# ── Readiness / status banner ─────────────────────────────────────────────────
stats = st.session_state.graph_stats or {}
_nodes = stats.get("nodes", 0)
_rels  = stats.get("relationships", 0)
_ready = neo4j_uri and neo4j_user and neo4j_pwd and google_api_key and _nodes > 0

if not (neo4j_uri and neo4j_user and neo4j_pwd):
    st.info(
        "👈 **Get started**: Enter your Neo4j URI, username, and password in the sidebar. "
        "Make sure `backend_ingest.py` has been run first to populate the graph."
    )
elif not google_api_key:
    st.warning("⚠️ Enter your **Gemini API key** in the sidebar to enable answer generation.")
elif _nodes == 0:
    st.warning(
        "⚠️ The Neo4j graph appears to be **empty**. "
        "Run `backend_ingest.py` first to bootstrap the knowledge graph:\n\n"
        "```bash\npython backend_ingest.py\n```"
    )
else:
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
            <strong>{_nodes:,} nodes</strong> &nbsp;·&nbsp;
            <strong>{_rels:,} relationships</strong>
            &nbsp;·&nbsp; Ask anything below ↓
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Suggested questions ───────────────────────────────────────────────────────
if _ready and not st.session_state.messages:
    st.markdown(
        '<div style="font-size:0.78rem; color:rgba(255,255,255,0.4); margin-bottom:8px;">💡 Try asking…</div>',
        unsafe_allow_html=True,
    )
    suggestions = [
        "What are the most critical open issues?",
        "Summarise recent bug reports",
        "Which issues mention performance problems?",
        "What files were changed most frequently?",
        "Which commits touched authentication code?",
        "What feature requests are open?",
    ]
    suggestion_cols = st.columns(3)
    for i, suggestion in enumerate(suggestions):
        with suggestion_cols[i % 3]:
            if st.button(suggestion, key=f"sugg_{i}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": suggestion})
                st.rerun()

# ── Chat history ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    avatar = "🧑‍💻" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# ── Chat input ────────────────────────────────────────────────────────────────
chat_placeholder = (
    "Ask a question about the repository issues or codebase…"
    if _ready else
    "Configure Neo4j and Gemini credentials in the sidebar first…"
)

user_input = st.chat_input(chat_placeholder, disabled=not _ready)

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(user_input)

    # ── Validate credentials before querying ──────────────────────────────────
    answer: Optional[str] = None

    if not google_api_key:
        answer = "❌ **Gemini API Key** is missing. Please enter it in the sidebar."
        with st.chat_message("assistant", avatar="🤖"):
            st.error(answer)

    elif not (neo4j_uri and neo4j_user and neo4j_pwd):
        answer = "❌ **Neo4j credentials** are incomplete. Please check the sidebar."
        with st.chat_message("assistant", avatar="🤖"):
            st.error(answer)

    else:
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("🔍 Searching knowledge graph and generating answer…"):
                try:
                    answer = query_graph_cypher(
                        question=user_input,
                        neo4j_uri=neo4j_uri,
                        neo4j_user=neo4j_user,
                        neo4j_pwd=neo4j_pwd,
                        google_api_key=google_api_key,
                    )
                except neo4j_exc.ServiceUnavailable:
                    answer = (
                        "❌ **Neo4j is unreachable.** Ensure Docker is running:\n\n"
                        "```bash\ndocker ps\ndocker logs neo4j-graphrag\n```"
                    )
                except neo4j_exc.AuthError:
                    answer = "❌ **Neo4j authentication failed.** Check your username and password."
                except ImportError as exc:
                    answer = (
                        f"❌ **Missing dependency:** {exc}\n\n"
                        "Run: `pip install google-genai langchain-google-genai`"
                    )
                except Exception as exc:
                    tb = traceback.format_exc()
                    answer = (
                        f"❌ **Retrieval error:**\n\n"
                        f"```\n{type(exc).__name__}: {exc}\n```\n\n"
                        f"<details><summary>Full traceback</summary>\n\n"
                        f"```\n{tb}\n```\n\n</details>"
                    )

            st.markdown(answer)

    if answer is not None:
        st.session_state.messages.append({"role": "assistant", "content": answer})

# ── Footer ────────────────────────────────────────────────────────────────────
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
        GraphRAG Chat &nbsp;·&nbsp; Neo4j + Google Gemini &nbsp;·&nbsp;
        <a href="https://neo4j.com/docs/neo4j-graphrag-python/current/" target="_blank"
           style="color:rgba(0,212,255,0.5); text-decoration:none;">Docs</a>
    </div>
    """,
    unsafe_allow_html=True,
)
