#!/usr/bin/env python3
# =============================================================================
# frontend_chat.py — Streamlit GraphRAG Chat UI
# =============================================================================

import logging
import os
import traceback
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

import streamlit as st
from neo4j import GraphDatabase, exceptions as neo4j_exc
from langchain_core.prompts import PromptTemplate

from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_google_genai import ChatGoogleGenerativeAI

try:
    from google import genai as google_genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    LANGCHAIN_GENAI_AVAILABLE = True
except ImportError:
    LANGCHAIN_GENAI_AVAILABLE = False

logger = logging.getLogger(__name__)

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

    .stApp {
        background: linear-gradient(135deg, #0d0d1a 0%, #0a1628 50%, #0d0d1a 100%);
        min-height: 100vh;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1923 0%, #0d1520 100%);
        border-right: 1px solid rgba(0, 212, 255, 0.15);
    }
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #00d4ff;
    }

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

    .section-header {
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: rgba(255,255,255,0.35);
        margin: 1.25rem 0 0.6rem 0;
    }

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

    hr {
        border: none !important;
        height: 1px !important;
        background: linear-gradient(90deg, transparent, rgba(0,212,255,0.4), transparent) !important;
        margin: 1.5rem 0 !important;
    }

    code {
        background: rgba(0, 212, 255, 0.08) !important;
        color: #00d4ff !important;
        border-radius: 4px !important;
        padding: 0.1rem 0.35rem !important;
        font-size: 0.85em !important;
    }

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

def _init_session():
    defaults = {
        "messages":        [],
        "neo4j_driver":    None,
        "graph_stats":     None,
        "available_repos": [],
        "selected_repos":  [],
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

_init_session()

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

def _fetch_commit_context(
    driver,
    question: str,
    selected_repos: Optional[list] = None,
    limit: int = 3,
    deep_scan_days: int = 30,
) -> str:
    try:
        from datetime import datetime, timezone, timedelta
        cutoff_iso = (
            (datetime.now(timezone.utc) - timedelta(days=deep_scan_days)).strftime("%Y-%m-%dT%H:%M:%SZ")
            if deep_scan_days > 0 else None
        )
        use_repo_filter = bool(selected_repos)

        with driver.session() as session:
            if use_repo_filter and cutoff_iso:
                result = session.run(
                    """
                    CALL db.index.fulltext.queryNodes('commit_summaries', $query)
                    YIELD node, score
                    WHERE score > 0 AND node.timestamp >= $cutoff
                    WITH node, score
                    MATCH (node)-[:BELONGS_TO]->(r:Repository)
                    WHERE r.full_name IN $selected_repos
                    RETURN node.sha          AS sha,
                           node.summary_text AS summary,
                           node.message      AS message,
                           node.timestamp    AS ts
                    ORDER BY score DESC
                    LIMIT $limit
                    """,
                    query=question,
                    cutoff=cutoff_iso,
                    selected_repos=selected_repos,
                    limit=limit,
                )
            elif use_repo_filter:
                result = session.run(
                    """
                    CALL db.index.fulltext.queryNodes('commit_summaries', $query)
                    YIELD node, score
                    WHERE score > 0
                    WITH node, score
                    MATCH (node)-[:BELONGS_TO]->(r:Repository)
                    WHERE r.full_name IN $selected_repos
                    RETURN node.sha          AS sha,
                           node.summary_text AS summary,
                           node.message      AS message,
                           node.timestamp    AS ts
                    ORDER BY score DESC
                    LIMIT $limit
                    """,
                    query=question,
                    selected_repos=selected_repos,
                    limit=limit,
                )
            elif cutoff_iso:
                result = session.run(
                    """
                    CALL db.index.fulltext.queryNodes('commit_summaries', $query)
                    YIELD node, score
                    WHERE score > 0 AND node.timestamp >= $cutoff
                    RETURN node.sha          AS sha,
                           node.summary_text AS summary,
                           node.message      AS message,
                           node.timestamp    AS ts
                    ORDER BY score DESC
                    LIMIT $limit
                    """,
                    query=question,
                    cutoff=cutoff_iso,
                    limit=limit,
                )
            else:
                result = session.run(
                    """
                    CALL db.index.fulltext.queryNodes('commit_summaries', $query)
                    YIELD node, score
                    WHERE score > 0
                    RETURN node.sha          AS sha,
                           node.summary_text AS summary,
                           node.message      AS message,
                           node.timestamp    AS ts
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

def analyze_cross_repo_impact(
    driver,
    helper_repo: str,
    parent_repo: str,
    commit_sha: str,
    google_api_key: str,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_pwd: str,
) -> str:
    affected_rows = []
    try:
        with driver.session() as session:
            result = session.run(
                """
                MATCH (helperCommit:Commit {sha: $commit_sha})-[:MODIFIED]->(changed)
                WHERE changed:Function OR changed:File
                WITH collect(changed) AS changedNodes
                MATCH (parentFile:File {repo: $parent_repo})-[:USES_REPO]->
                      (helperRepo:Repository {full_name: $helper_repo})
                OPTIONAL MATCH (parentFunc:Function)-[:CALLS]->(helperFunc:Function)
                WHERE parentFunc.repo = $parent_repo AND helperFunc IN changedNodes
                RETURN DISTINCT
                    parentFile.path      AS affected_file,
                    parentFunc.name      AS affected_function,
                    parentFunc.code      AS affected_code
                ORDER BY affected_file
                """,
                commit_sha=commit_sha,
                parent_repo=parent_repo,
                helper_repo=helper_repo,
            )
            affected_rows = result.data()
    except Exception as exc:
        return f"❌ **Graph query failed:** {exc}"

    if not affected_rows:
        return (
            f"✅ No direct impact detected in `{parent_repo}` "
            f"from commit `{commit_sha[:8]}` in `{helper_repo}`. "
            "Either the cross-repo edges haven't been resolved yet "
            "(run `resolve_cross_repo_edges` in the backend), "
            "or this commit doesn't touch anything the parent imports."
        )

    context_lines = [f"Helper repo: {helper_repo}", f"Parent repo: {parent_repo}",
                     f"Commit: {commit_sha[:8]}", "", "Affected parent-repo code:"]
    for row in affected_rows:
        context_lines.append(f"\nFile: {row.get('affected_file', '?')}")
        if row.get("affected_function"):
            context_lines.append(f"Function: {row['affected_function']}")
        if row.get("affected_code"):
            snippet = (row["affected_code"] or "")[:800]
            context_lines.append(f"Code:\n{snippet}")
    context = "\n".join(context_lines)

    commit_summary = ""
    try:
        with driver.session() as session:
            rec = session.run(
                "MATCH (c:Commit {sha: $sha}) RETURN c.summary_text AS s, c.message AS m",
                sha=commit_sha,
            ).single()
            if rec:
                commit_summary = rec.get("s") or rec.get("m") or ""
    except Exception:
        pass

    prompt = (
        f"A commit was made to the helper library `{helper_repo}`.\n"
        f"Commit summary: {commit_summary[:400]}\n\n"
        f"The following code in the parent repo `{parent_repo}` directly "
        f"depends on what changed:\n\n{context}\n\n"
        "For each affected file and function:\n"
        "1. Explain what might break and why.\n"
        "2. Suggest the minimal code change needed to fix or adapt it.\n"
        "Be specific and concise. Do not repeat the full code back."
    )

    from langchain_google_genai import ChatGoogleGenerativeAI
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=google_api_key,
        temperature=0.2,
    )
    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as exc:
        return f"❌ **LLM call failed:** {exc}"

def query_graph_cypher(
    question: str,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_pwd: str,
    google_api_key: str,
    selected_repos: Optional[list] = None,
    deep_scan_days: int = 30,
) -> str:
    graph = Neo4jGraph(
        url=neo4j_uri,
        username=neo4j_user,
        password=neo4j_pwd
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=google_api_key,
        temperature=0.0 
    )

    from datetime import datetime, timezone, timedelta as _td
    if deep_scan_days > 0:
        _cutoff_str = (datetime.now(timezone.utc) - _td(days=deep_scan_days)).strftime("%Y-%m-%dT%H:%M:%SZ")
        _time_hint = (
            f"6. RETRIEVAL DEPTH: The user set a {deep_scan_days}-day window. "
            f"When querying Commit nodes by recency, prefer filtering with "
            f"`WHERE c.timestamp >= '{_cutoff_str}'` unless the question asks for older history."
        )
    else:
        _time_hint = "6. RETRIEVAL DEPTH: No time limit — search the full commit history."

    if selected_repos:
        _repo_list = "[" + ", ".join(f'"{r}"' for r in selected_repos) + "]"
        _repo_hint = (
            f"7. REPO SCOPE & CROSS-REPO: The user has selected {len(selected_repos)} repositories: {_repo_list}. "
            f"When filtering File, Directory, or Function nodes, use `WHERE n.repo IN {_repo_list}`. "
            f"CRITICAL: `Module` nodes do NOT have a `repo` property. To find shared dependencies, you must traverse "
            f"from Files to Modules (e.g., `(f:File)-[:DEPENDS_ON]->(m:Module)`)."
        )
    else:
        _repo_hint = (
            "7. REPO SCOPE: All repositories are in scope. "
            "No repository filtering is needed — search the full graph."
        )

    CYPHER_GENERATION_TEMPLATE = (
        "Task: Generate a Cypher statement to query a graph database.\n"
        "    Instructions:\n"
        "    1. Use ONLY the provided relationship types and properties explicitly listed in the schema. Do NOT hallucinate or invent properties.\n"
        "    2. Ensure your Cypher is syntactically valid for Neo4j v5.\n"
        "    3. CRITICAL: A 'WHERE' clause must immediately follow a 'MATCH', 'OPTIONAL MATCH', 'YIELD', or 'WITH' clause. Never place 'WHERE' randomly.\n"
        "    4. CRITICAL: When filtering by ANY string property (file paths, modules, authors, or repo names), ALWAYS use case-insensitive `CONTAINS`. Example: `WHERE toLower(f.repo) CONTAINS toLower('tqdm')`. NEVER use exact `=` matching for strings.\n"
        "    5. Do not include any explanations, apologies, or markdown formatting (like ```cypher). Return ONLY the raw query string.\n"
        f"    {_time_hint}\n"
        f"    {_repo_hint}\n"
        "    8. RETURN CLAUSES: When asked about relationships (dependencies, calls, imports), return the connected node properties. Only return relationship properties if they actually exist in the provided schema.\n"
        "    9. SYMBOL & BLAST-RADIUS ANALYSIS: The graph tracks module-level edges (`File-[:DEPENDS_ON]->Module`, `Function-[:CALLS]->Function`), NOT individual imported names "
        "(e.g. there is no node for `init()` or `Fore` themselves — only for the `colorama` module and the functions whose code happens to mention it). "
        "For 'what breaks if symbol/function X changes', 'where would a bug in module Y surface', or any other symbol-level or blast-radius question, use this general two-hop pattern: "
        "(a) match the dependency edge from the target module/file, (b) walk to the functions declared in or calling from the dependent file, and fetch their `code` "
        "so the symbol-level filtering (does this function actually reference `init` or `Fore`?) happens by reading source text, not by the graph schema. "
        "Module-import template: `MATCH (f:File)-[:DEPENDS_ON]->(m:Module), (f)-[:DECLARES]->(fn:Function) WHERE toLower(m.name) CONTAINS toLower('<module>') RETURN f.path, fn.name, fn.code`. "
        "Function-call template: `MATCH (caller:Function)-[:CALLS]->(callee:Function) WHERE toLower(callee.name) CONTAINS toLower('<symbol>') RETURN caller.name, caller.code, callee.name`. "
        "Always prefer returning `fn.code` / `caller.code` over names alone when the question asks 'where', 'how', or 'what would break' — the answer step needs the source text to reason about specific symbols.\n"
        "    10. SPARSE-PROPERTY FALLBACK: Some properties (e.g. a commit's summary vs. its raw message) may only be populated on a subset of nodes. When a question could be answered by either of two known alternate properties, "
        "use `coalesce()` across them (e.g. `coalesce(c.summary_text, c.message)`) instead of querying only one and risking an empty result. "
        "Do not let a narrow property choice cause a false 'no data' answer when a broader, still-schema-valid query would have found it.\n\n"
        "    Schema:\n"
        "    {schema}\n\n"
        "    The question is:\n"
        "    {question}"
    )

    cypher_prompt = PromptTemplate(
        template=CYPHER_GENERATION_TEMPLATE,
        input_variables=["schema", "question"],
    )

    QA_GENERATION_TEMPLATE = """You are a senior software engineering assistant.
    Use the following information retrieved from the codebase graph database to answer the user's question.
    
    IMPORTANT INSTRUCTIONS:
    1. The `Database Results` provided below are the direct output of a strict database query designed to answer the user's exact question. 
    2. TRUST THE DATA: If the user asks "what files depend on X" and the results yield a list of files, you must confidently state that those files depend on X. 
    3. Do NOT claim you lack information just because the results only contain names/paths instead of full source code or explicit relationship labels.
    4. If the database results are completely empty (`[]`), only then say you don't have the data to answer.

    Database Results:
    {context}

    User Question:
    {question}"""

    qa_prompt = PromptTemplate(
        template=QA_GENERATION_TEMPLATE,
        input_variables=["context", "question"]
    )

    chain = GraphCypherQAChain.from_llm(
        cypher_llm=llm,       
        qa_llm=llm,           
        graph=graph,          
        verbose=True,         
        cypher_prompt=cypher_prompt,  
        qa_prompt=qa_prompt,          
        allow_dangerous_requests=True, 
        validate_cypher=True  
    )

    try:
        response = chain.invoke({"query": question})
        return response.get("result", "I couldn't formulate an answer based on the database results.")
    except ValueError as exc:
        if "No tools" in str(exc) or "OutputParserException" in str(exc):
            return "❌ **Query Generation Failed:** The LLM couldn't map that question to the database schema."
        raise exc
    except Exception as exc:
        return f"❌ **Query Execution Failed:**\n```\n{str(exc)}\n```"

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

    st.markdown('<div class="section-header">🗄️ Neo4j Connection</div>', unsafe_allow_html=True)
    neo4j_uri  = st.text_input("Neo4j URI",  value="neo4j://localhost:7687", key="uri")
    neo4j_user = st.text_input("Username",   value="neo4j",                   key="usr")
    neo4j_pwd  = st.text_input("Password",   type="password", placeholder="password123", key="pwd")

    st.markdown('<div class="section-header">🔑 Google Gemini</div>', unsafe_allow_html=True)
    google_api_key = os.getenv("GOOGLE_API_KEY", "")
    if google_api_key:
        st.markdown(
            f'{_status_dot("green")}<span style="font-size:0.82rem; color:rgba(255,255,255,0.75);">'
            'API key loaded from <code>.env</code></span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'{_status_dot("red")}<span style="font-size:0.82rem; color:rgba(255,100,100,0.9);">'
            '<code>GOOGLE_API_KEY</code> not found in <code>.env</code></span>',
            unsafe_allow_html=True,
        )

    if neo4j_uri and neo4j_user and neo4j_pwd:
        ok, msg = _check_neo4j(neo4j_uri, neo4j_user, neo4j_pwd)
        dot = _status_dot("green") if ok else _status_dot("red")
        status_label = "Connected" if ok else "Unreachable"
        st.markdown(
            f'<div style="font-size:0.78rem; color:rgba(255,255,255,0.6); margin-top:4px;">'
            f'{dot}Neo4j: {status_label}</div>',
            unsafe_allow_html=True,
        )

        if ok and st.session_state.graph_stats is None:
            try:
                drv = _get_driver(neo4j_uri, neo4j_user, neo4j_pwd)
                st.session_state.graph_stats = _query_graph_stats(drv)
            except Exception:
                st.session_state.graph_stats = {}

    if st.button("🔄 Refresh Graph Stats", use_container_width=True, key="refresh_stats"):
        if neo4j_uri and neo4j_user and neo4j_pwd:
            try:
                _close_driver()
                drv = _get_driver(neo4j_uri, neo4j_user, neo4j_pwd)
                st.session_state.graph_stats = _query_graph_stats(drv)
                new_repos = _fetch_available_repos(drv)
                st.session_state.available_repos = new_repos
                existing = set(st.session_state.selected_repos)
                for r in new_repos:
                    if r not in existing:
                        st.session_state.selected_repos.append(r)
                st.toast("Graph stats and repository list refreshed!", icon="✅")
            except Exception as exc:
                st.error(f"Could not refresh stats: {exc}")

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

    st.markdown('<div class="section-header">📦 Active Repositories</div>', unsafe_allow_html=True)
    if neo4j_uri and neo4j_user and neo4j_pwd:
        if not st.session_state.available_repos:
            try:
                drv = _get_driver(neo4j_uri, neo4j_user, neo4j_pwd)
                st.session_state.available_repos = _fetch_available_repos(drv)
                if not st.session_state.selected_repos:
                    st.session_state.selected_repos = list(st.session_state.available_repos)
            except Exception:
                pass

        available_repos = st.session_state.available_repos
        if available_repos:
            _col_all, _col_none = st.columns(2)
            with _col_all:
                if st.button("All", key="repo_sel_all", use_container_width=True):
                    st.session_state.selected_repos = list(available_repos)
                    st.rerun()
            with _col_none:
                if st.button("None", key="repo_clr_all", use_container_width=True):
                    st.session_state.selected_repos = []
                    st.rerun()

            selected_repos_widget = st.multiselect(
                label="Repositories in scope",
                options=available_repos,
                default=st.session_state.selected_repos,
                key="repo_multiselect",
                label_visibility="collapsed",
                placeholder="Empty = all repositories in scope",
            )
            st.session_state.selected_repos = selected_repos_widget

            n_sel, n_total = len(selected_repos_widget), len(available_repos)
            if n_sel == 0 or n_sel == n_total:
                _scope_txt   = "🌐 All repositories in scope"
                _scope_color = "#00d4ff"
            else:
                _scope_txt   = f"🎯 {n_sel} / {n_total} repos selected"
                _scope_color = "#7b61ff"
            st.markdown(
                f'<div style="font-size:0.73rem; color:{_scope_color}; '
                f'margin-top:-2px; margin-bottom:6px;">{_scope_txt}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="font-size:0.73rem; color:rgba(255,255,255,0.3);">'  
                'No repos found. Run <code>backend_ingest.py</code> first.</div>',
                unsafe_allow_html=True,
            )

    st.markdown('<div class="section-header">⚙️ RAG Settings</div>', unsafe_allow_html=True)
    top_k = st.slider("Top-K retrieval results", min_value=1, max_value=15, value=5, step=1)

    deep_scan_days = st.slider(
        "Retrieval depth (days)",
        min_value=0,
        max_value=365,
        value=30,
        step=1,
        help=(
            "Limits commit context to the last N days.\n\n"
            "**0** = no limit (search all history)\n"
            "**7** = last week only (fast / recent)\n"
            "**365** = full year of commits"
        ),
        key="deep_scan_days",
    )
    if deep_scan_days == 0:
        _depth_label = "🌐 All history"
    elif deep_scan_days <= 7:
        _depth_label = f"⚡ Last {deep_scan_days}d — recent only"
    elif deep_scan_days <= 30:
        _depth_label = f"📅 Last {deep_scan_days}d — balanced"
    elif deep_scan_days <= 90:
        _depth_label = f"🗂️ Last {deep_scan_days}d — broad"
    else:
        _depth_label = f"📜 Last {deep_scan_days}d — deep history"
    st.markdown(
        f'<div style="font-size:0.75rem; color:rgba(255,255,255,0.45); margin-top:-6px;">'
        f'{_depth_label}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown(
        """
        <div style="font-size:0.7rem; color:rgba(255,255,255,0.35); line-height:1.8;">
            <div style="margin-bottom:6px; font-weight:600; color:rgba(255,255,255,0.5);">Pipeline</div>
            <span class="pipeline-badge">VectorRAG</span>Issue similarity<br>
            <span class="pipeline-badge">FullText</span>Commit search<br>
            <span class="pipeline-badge">Gemini 2.5</span>Answer generation<br>
            <br>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    '<h1 class="hero-title">🕸️ GraphRAG Chat</h1>'
    '<p class="hero-sub">Query your Neo4j knowledge graph with natural language, '
    'powered by Google Gemini</p>',
    unsafe_allow_html=True,
)

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

with st.expander("🔍 Cross-Repo Impact Analysis", expanded=False):
    st.markdown(
        "Paste a **commit SHA** from the helper repo to find what breaks in the parent repo.",
        unsafe_allow_html=False,
    )
    col_helper, col_parent = st.columns(2)
    with col_helper:
        impact_helper = st.text_input("Helper repo (owner/repo)", key="impact_helper",
                                      placeholder="acme/mylib")
    with col_parent:
        impact_parent = st.text_input("Parent repo (owner/repo)", key="impact_parent",
                                      placeholder="acme/product")
    impact_sha = st.text_input("Commit SHA from helper repo", key="impact_sha",
                                placeholder="abc1234...")

    if st.button("Analyze Impact", key="btn_impact"):
        if not (impact_helper and impact_parent and impact_sha):
            st.warning("Fill in both repo names and the commit SHA.")
        elif not google_api_key:
            st.error("Gemini API key required.")
        else:
            with st.spinner("Tracing impact across repos…"):
                try:
                    drv = _get_driver(neo4j_uri, neo4j_user, neo4j_pwd)
                    impact_result = analyze_cross_repo_impact(
                        driver=drv,
                        helper_repo=impact_helper,
                        parent_repo=impact_parent,
                        commit_sha=impact_sha,
                        google_api_key=google_api_key,
                        neo4j_uri=neo4j_uri,
                        neo4j_user=neo4j_user,
                        neo4j_pwd=neo4j_pwd,
                    )
                    st.markdown(impact_result)
                except Exception as exc:
                    st.error(f"Error: {exc}")

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
    st.warning("⚠️ **Gemini API key** not found. Set `GOOGLE_API_KEY` in your `.env` file and restart the app.")
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

if _ready and not st.session_state.messages:
    st.markdown(
        '<div style="font-size:0.78rem; color:rgba(255,255,255,0.4); margin-bottom:8px;">💡 Try asking…</div>',
        unsafe_allow_html=True,
    )
    suggestions = [
        "What are the most critical open issues?",
        "Summarise recent bug reports",
        "What files were changed most frequently?",
        "Which commits touched authentication code?",
        "Which parent-repo files import from the helper repo?",
        "What functions in the parent repo call helper repo functions?",
    ]
    suggestion_cols = st.columns(3)
    for i, suggestion in enumerate(suggestions):
        with suggestion_cols[i % 3]:
            if st.button(suggestion, key=f"sugg_{i}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": suggestion})
                st.rerun()

for msg in st.session_state.messages:
    avatar = "🧑‍💻" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

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

    answer: Optional[str] = None

    if not google_api_key:
        answer = "❌ **Gemini API Key** is missing. Set `GOOGLE_API_KEY` in your `.env` file and restart the app."
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
                    _sel = st.session_state.get("selected_repos") or []
                    _available = st.session_state.get("available_repos") or []
                    _active_repos = _sel if (_sel and len(_sel) < len(_available)) else None

                    answer = query_graph_cypher(
                        question=user_input,
                        neo4j_uri=neo4j_uri,
                        neo4j_user=neo4j_user,
                        neo4j_pwd=neo4j_pwd,
                        google_api_key=google_api_key,
                        selected_repos=_active_repos,
                        deep_scan_days=deep_scan_days,
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