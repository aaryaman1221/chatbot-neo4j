#!/usr/bin/env python3
# =============================================================================
# frontend_chat.py — Streamlit GraphRAG Chat UI
# =============================================================================

import json
import logging
import os
import traceback
from typing import Optional

from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool

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


def extract_evidence_count(text: str) -> int:
    """Counts the bullet points in the LLM's Evidence section to drive the confidence score."""
    if "## Evidence" in text:
        # Split the text at the Evidence header and count the bullets in that section
        evidence_section = text.split("## Evidence")[-1]
        return evidence_section.count("- ") + evidence_section.count("* ")
    return 0


# =============================================================================
# Intent Classification — runs BEFORE the agent, decouples routing from keywords
# =============================================================================

# Canonical intent labels the classifier must choose from.
INTENT_LABELS = [
    "DEPENDENCY_TRAVERSAL",  # who calls/uses/imports X; what depends on X
    "BUG_SURFACE_ANALYSIS",  # where a bug/change in X would propagate / surface
    "BLAST_RADIUS",          # what breaks if X changes; downstream impact
    "COMMIT_SEARCH",         # git history, why was X changed, recent commits
    "GENERAL_GRAPH_QUERY",   # architecture overview, file locations, open-ended
]

_INTENT_SYSTEM = """\
You are a query-intent classifier for a code-repository knowledge graph.
Classify the user's question into EXACTLY ONE of these intents:

  DEPENDENCY_TRAVERSAL  — asks which files/functions/components call, use,
                          import, or depend on a module, function, or symbol.
                          Signals: "what calls X", "which functions use X",
                          "what imports X", "depends on X", "references X",
                          "uses X", "invokes X", cross-repo usage questions.

  BUG_SURFACE_ANALYSIS  — asks where a bug or defect in X would show up,
                          propagate, or be visible in the codebase.
                          Signals: "where would a bug surface", "where does X
                          affect", "if X is broken what fails", "rendering bug",
                          "error in X would appear in".

  BLAST_RADIUS          — asks what breaks / is at risk if X is changed,
                          removed, or refactored. Forward-impact questions.
                          Signals: "blast radius", "what breaks if", "impact of
                          changing", "downstream effects", "what would break".

  COMMIT_SEARCH         — asks about git history, commit messages, why/when a
                          change was made, recent fixes, PR history.
                          Signals: "commit", "git log", "when was X changed",
                          "who changed", "recent fix", "history of".

  GENERAL_GRAPH_QUERY   — everything else: architecture overviews, file
                          locations, listing nodes, open-ended questions.

Also extract these three fields. Read the rules carefully:

  module  — the primary library / package being asked ABOUT (the dependency
             being analysed). Must be a bare package/repo name, lowercase.
             Examples: "lipgloss", "tqdm", "colorama".
             Empty string if none.

  symbol  — a SPECIFIC, LITERAL function name, type name, or method name that
             appears as an identifier in source code. Must look like valid code:
             "Color", "NewStyle", "Render", "Init", "Fore.RED".
             CRITICAL: if the user describes behaviour in plain English
             (e.g. "color rendering logic", "authentication flow", "error
             handling") that is NOT a symbol — set symbol to "".
             Only populate symbol when you can point to an actual identifier.

  repo    — the repository the question is scoped TO (the consumer/caller side).
             Must be the TOP-LEVEL repository name only — never a sub-path.
             Examples: "bubbles", "tqdm", "faker".
             If the user says "bubbles/progress", repo = "bubbles" (the repo
             name); the sub-component "progress" is a path hint, not a repo.
             Empty string if none / same as module.

  path_hint — an optional sub-directory or component path within repo that
              the user mentioned (e.g. "progress" from "bubbles/progress").
              Empty string if none.

Respond with ONLY a JSON object, no markdown, no explanation:
{"intent": "<LABEL>", "module": "<str>", "symbol": "<str>", "repo": "<str>", "path_hint": "<str>"}
"""

def classify_intent(question: str, llm) -> dict:
    """
    Call the LLM once to classify the question into a canonical intent and
    extract key entities. Falls back to GENERAL_GRAPH_QUERY on any error.

    Returns a dict: {intent, module, symbol, repo, path_hint}
    """
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        response = llm.invoke([
            SystemMessage(content=_INTENT_SYSTEM),
            HumanMessage(content=question),
        ])
        raw = response.content
        # Strip accidental markdown fences
        raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        parsed = json.loads(raw)
        intent = parsed.get("intent", "GENERAL_GRAPH_QUERY")
        if intent not in INTENT_LABELS:
            intent = "GENERAL_GRAPH_QUERY"
        return {
            "intent":    intent,
            "module":    parsed.get("module", ""),
            "symbol":    parsed.get("symbol", ""),
            "repo":      parsed.get("repo", ""),
            "path_hint": parsed.get("path_hint", ""),
        }
    except Exception as exc:
        logger.warning("Intent classification failed (%s), falling back to GENERAL_GRAPH_QUERY", exc)
        return {"intent": "GENERAL_GRAPH_QUERY", "module": "", "symbol": "", "repo": "", "path_hint": ""}


# =============================================================================
# GraphRAGToolkit — Structured Cypher tools, no keyword-based routing
# =============================================================================

class GraphRAGToolkit:
    """Encapsulates Neo4j driver + GraphCypherQAChain.
    Tools now have precise Cypher for every intent; routing is done by the
    intent classifier, not by LLM keyword matching on docstrings."""

    def __init__(self, driver, cypher_qa_chain):
        self.driver = driver
        self.cypher_qa_chain = cypher_qa_chain

    # ------------------------------------------------------------------
    # Direct Cypher tools (called by create_and_run_agent after intent
    # classification, so docstrings describe the graph pattern, not the
    # user's vocabulary — the agent never has to guess which to call).
    # ------------------------------------------------------------------

    def run_dependency_traversal(self, module: str, symbol: str = "", repo: str = "", path_hint: str = "") -> str:
        """
        Two-pass approach:
          Pass 1 — Function-[:CALLS]->Function for direct call-graph edges
                   (only when a real code symbol is given).
          Pass 2 — File-[:DEPENDS_ON]->Module + File-[:DECLARES]->Function,
                   filtered by repo name on f.repo, and path_hint on f.path.
                   Symbol filter applies only when symbol looks like a code
                   identifier (short, no spaces).
        Returns the union of both passes.
        """
        # Only use symbol as a source-text filter when it's a plausible
        # code identifier — short and contains no spaces.
        code_symbol = symbol if (symbol and len(symbol) < 40 and " " not in symbol) else ""

        results = []

        # Pass 1: direct Function-[:CALLS]->Function edges (needs a real symbol)
        if code_symbol:
            cypher_fn_calls = """
            MATCH (caller:Function)-[:CALLS]->(callee:Function)
            WHERE toLower(callee.name) CONTAINS toLower($symbol)
            OPTIONAL MATCH (caller_file:File)-[:DECLARES]->(caller)
            WITH caller_file, caller, callee
            WHERE $repo = "" OR toLower(coalesce(caller_file.repo, "")) CONTAINS toLower($repo)
            WITH caller_file, caller, callee
            WHERE $path_hint = "" OR toLower(coalesce(caller_file.path, "")) CONTAINS toLower($path_hint)
            RETURN
                coalesce(caller_file.path, "<unknown file>") AS file,
                caller.name  AS caller_fn,
                callee.name  AS callee_fn,
                caller.code  AS code
            LIMIT 30
            """
            try:
                with self.driver.session() as session:
                    rows = session.run(cypher_fn_calls, {"symbol": code_symbol, "repo": repo, "path_hint": path_hint}).data()
                    results.extend(rows)
            except Exception as exc:
                logger.warning("Pass-1 (CALLS) query failed: %s", exc)

        # Pass 2: module-import-level callers
        if module:
            cypher_module_import = """
            MATCH (f:File)-[r:DEPENDS_ON]->(m:Module)
            WHERE r.is_active = true
              AND toLower(m.name) CONTAINS toLower($module)
            WITH f, m
            WHERE $repo = "" OR toLower(coalesce(f.repo, "")) CONTAINS toLower($repo)
            WITH f, m
            WHERE $path_hint = "" OR toLower(f.path) CONTAINS toLower($path_hint)
            MATCH (f)-[:DECLARES]->(fn:Function)
            WITH f, m, fn
            WHERE $symbol = "" OR fn.code CONTAINS $symbol
            RETURN
                f.path    AS file,
                fn.name   AS caller_fn,
                ""        AS callee_fn,
                fn.code   AS code,
                m.name    AS module_name
            LIMIT 30
            """
            try:
                with self.driver.session() as session:
                    rows = session.run(cypher_module_import, {"module": module, "symbol": code_symbol, "repo": repo, "path_hint": path_hint}).data()
                    results.extend(rows)
            except Exception as exc:
                logger.warning("Pass-2 (DEPENDS_ON) query failed: %s", exc)

        if not results:
            return (
                f"No callers found for"
                + (f" symbol '{symbol}'" if symbol else "")
                + (f" in module '{module}'" if module else "")
                + (f" scoped to repo '{repo}'" if repo else "")
                + (f" path '{path_hint}'" if path_hint else "")
                + "."
            )
        return json.dumps(results, default=str, indent=2)

    def run_blast_radius(self, module: str, symbol: str = "", repo: str = "", path_hint: str = "") -> str:
        """
        Forward-impact analysis: which files/functions are at risk if `module`
        (or a specific `symbol` within it) changes. Uses DEPENDS_ON + DECLARES
        with source-text filtering only when symbol is a real code identifier.
        """
        code_symbol = symbol if (symbol and len(symbol) < 40 and " " not in symbol) else ""

        cypher_statement = """
        MATCH (f:File)-[r:DEPENDS_ON]->(m:Module)
        WHERE r.is_active = true
          AND toLower(m.name) CONTAINS toLower($module)
        WITH f, m
        WHERE $repo = "" OR toLower(coalesce(f.repo, "")) CONTAINS toLower($repo)
        WITH f, m
        WHERE $path_hint = "" OR toLower(f.path) CONTAINS toLower($path_hint)
        OPTIONAL MATCH (f)-[:DECLARES]->(fn:Function)
        WITH f, m, fn
        WHERE $symbol = "" OR fn.code CONTAINS $symbol
        WITH f, m, collect(fn.name) AS funcs
        WHERE size(funcs) > 0 OR $symbol = ""
        RETURN
            f.path   AS file,
            m.name   AS module_name,
            funcs    AS at_risk_functions
        LIMIT 40
        """
        try:
            with self.driver.session() as session:
                results = session.run(cypher_statement, {"module": module, "symbol": code_symbol, "repo": repo, "path_hint": path_hint}).data()
            if not results:
                return (
                    f"No blast-radius results for module '{module}'"
                    + (f" / symbol '{symbol}'" if symbol else "")
                    + "."
                )
            return json.dumps(results, default=str, indent=2)
        except Exception as exc:
            return f"Error running blast-radius query: {exc}"

    def run_bug_surface_analysis(self, module: str, symbol: str = "", repo: str = "", path_hint: str = "") -> str:
        """
        Surface analysis: finds all files that import the module, then for each
        finds the functions that reference the symbol (or all functions if no
        real symbol), returning fn.code for the QA LLM to reason about.
        Wider net than blast-radius — no grouping/aggregation.
        """
        code_symbol = symbol if (symbol and len(symbol) < 40 and " " not in symbol) else ""

        cypher_statement = """
        MATCH (f:File)-[r:DEPENDS_ON]->(m:Module)
        WHERE r.is_active = true
          AND toLower(m.name) CONTAINS toLower($module)
        WITH f, m
        WHERE $repo = "" OR toLower(coalesce(f.repo, "")) CONTAINS toLower($repo)
        WITH f, m
        WHERE $path_hint = "" OR toLower(f.path) CONTAINS toLower($path_hint)
        MATCH (f)-[:DECLARES]->(fn:Function)
        WITH f, m, fn
        WHERE $symbol = "" OR fn.code CONTAINS $symbol
        RETURN
            f.path   AS file,
            m.name   AS source_module,
            fn.name  AS function_name,
            fn.code  AS function_code
        ORDER BY f.path, fn.name
        LIMIT 40
        """
        try:
            with self.driver.session() as session:
                results = session.run(cypher_statement, {"module": module, "symbol": code_symbol, "repo": repo, "path_hint": path_hint}).data()
            if not results:
                return (
                    f"No surface-analysis results for module '{module}'"
                    + (f" / symbol '{symbol}'" if symbol else "")
                    + "."
                )
            return json.dumps(results, default=str, indent=2)
        except Exception as exc:
            return f"Error running bug-surface analysis: {exc}"

    def run_commit_search(self, query: str) -> str:
        cypher_statement = """
        CALL db.index.fulltext.queryNodes('commit_summaries', $query)
        YIELD node, score
        WHERE score > 0
        RETURN node.sha       AS sha,
               coalesce(node.summary_text, node.message) AS message,
               node.timestamp AS timestamp
        ORDER BY score DESC
        LIMIT 8
        """
        try:
            with self.driver.session() as session:
                results = session.run(cypher_statement, {"query": query}).data()
            if not results:
                return "No matching commits found."
            return json.dumps(results, default=str, indent=2)
        except Exception as exc:
            return f"Error searching commit history: {exc}"

    def run_generic_query(self, question: str) -> str:
        try:
            response = self.cypher_qa_chain.invoke({"query": question})
            return response.get("result", "No result returned from the graph.")
        except Exception as exc:
            return f"Error querying knowledge graph: {exc}"

    def get_tools(self) -> list:
        """Expose LangChain @tool wrappers for the agent.
        Docstrings here describe graph patterns, not user vocabulary — routing
        is handled upstream by classify_intent() before the agent is invoked."""
        driver = self.driver
        cypher_qa_chain = self.cypher_qa_chain
        toolkit = self

        @tool
        def analyze_blast_radius(target_module: str, symbol: str = "", repo: str = "", path_hint: str = "") -> str:
            """Forward-impact analysis: given a module (and optional symbol),
            return all files and functions that depend on it and would be broken
            if it changed. Use when intent is BLAST_RADIUS."""
            return toolkit.run_blast_radius(module=target_module, symbol=symbol, repo=repo, path_hint=path_hint)

        @tool
        def find_callers_and_dependents(target_module: str, symbol: str = "", repo: str = "", path_hint: str = "") -> str:
            """Find all files and functions that call, import, or reference the
            given module or symbol. Covers both Function-[:CALLS]->Function edges
            and File-[:DEPENDS_ON]->Module edges. Use when intent is
            DEPENDENCY_TRAVERSAL."""
            return toolkit.run_dependency_traversal(module=target_module, symbol=symbol, repo=repo, path_hint=path_hint)

        @tool
        def surface_bug_impact(target_module: str, symbol: str = "", repo: str = "", path_hint: str = "") -> str:
            """Return every file and function that imports a module and references
            a given symbol, including their source code, so the LLM can reason
            about where a bug in that module/symbol would surface. Use when intent
            is BUG_SURFACE_ANALYSIS."""
            return toolkit.run_bug_surface_analysis(module=target_module, symbol=symbol, repo=repo, path_hint=path_hint)

        @tool
        def search_commit_history(semantic_query: str) -> str:
            """Search the fulltext index over commit messages and summaries.
            Accepts ONLY semantic_query. Do not pass target_module, repo, or
            any other argument to this tool."""
            return toolkit.run_commit_search(query=semantic_query)

        @tool
        def generic_graph_query(question: str) -> str:
            """General-purpose graph query via GraphCypherQAChain.
            Accepts ONLY question. Do not pass any other arguments."""
            return toolkit.run_generic_query(question=question)

        return [
            analyze_blast_radius,
            find_callers_and_dependents,
            surface_bug_impact,
            search_commit_history,
            generic_graph_query,
        ]


# Intent → tool name mapping used by create_and_run_agent to prime the agent
# with the correct tool before it reasons, eliminating keyword-based guessing.
_INTENT_TO_TOOL = {
    "DEPENDENCY_TRAVERSAL": "find_callers_and_dependents",
    "BUG_SURFACE_ANALYSIS": "surface_bug_impact",
    "BLAST_RADIUS":         "analyze_blast_radius",
    "COMMIT_SEARCH":        "search_commit_history",
    "GENERAL_GRAPH_QUERY":  "generic_graph_query",
}


def create_and_run_agent(
    user_input: str,
    llm,
    toolkit: GraphRAGToolkit,
    intent_data: Optional[dict] = None,
) -> str:
    """
    1. Classify intent and extract entities — skipped when intent_data is
       pre-computed by route_and_answer() (saves one LLM call).
    2. Build a system prompt that names the correct tool and passes the
       extracted entities, so the agent never has to guess from vocabulary.
    3. Run the agent.

    Returns the agent's final answer string.
    """
    # ── Step 1: classify (skip if already done upstream) ──────────────────────
    if intent_data is None:
        intent_data = classify_intent(user_input, llm)
    intent      = intent_data["intent"]
    module      = intent_data["module"]
    symbol      = intent_data["symbol"]
    repo        = intent_data["repo"]
    path_hint   = intent_data["path_hint"]
    tool_name   = _INTENT_TO_TOOL.get(intent, "generic_graph_query")

    logger.info("Intent: %s | module=%s | symbol=%s | repo=%s | path_hint=%s → tool=%s",
                intent, module, symbol, repo, path_hint, tool_name)

    # ── Step 2: build a per-tool-aware system prompt ─────────────────────────
    # Each intent gets a call instruction that names ONLY the arguments that
    # tool actually accepts, preventing kwargs bleed across tools.
    if intent == "COMMIT_SEARCH":
        # Derive a compact search phrase from the raw question — don't pass
        # repo/module/symbol as args since the tool only takes semantic_query.
        call_instruction = (
            f"Call `search_commit_history` with:\n"
            f"  semantic_query: a concise phrase capturing the key concepts "
            f"from the user's question (e.g. 'lipgloss API migration', "
            f"'color rendering fix'). Do NOT pass any other arguments."
        )
    elif intent == "GENERAL_GRAPH_QUERY":
        call_instruction = (
            f"Call `generic_graph_query` with:\n"
            f"  question: the user's full question verbatim. "
            f"Do NOT pass any other arguments."
        )
    else:
        # Graph traversal tools: build the argument block from extracted entities
        arg_lines = [f"  target_module: \"{module}\"" if module else "  target_module: \"\""]
        if symbol:
            arg_lines.append(f"  symbol: \"{symbol}\"  # literal code identifier")
        else:
            arg_lines.append("  symbol: \"\"  # omit — no specific identifier extracted")
        if repo:
            arg_lines.append(f"  repo: \"{repo}\"")
        if path_hint:
            arg_lines.append(f"  path_hint: \"{path_hint}\"")
        arg_block = "\n".join(arg_lines)
        call_instruction = (
            f"Call `{tool_name}` with these exact arguments:\n{arg_block}\n"
            f"Do NOT add extra arguments. Do NOT change these values."
        )

    system_msg = (
        "You are a senior software architect analyzing a repository knowledge graph.\n\n"
        f"The user's question has been classified as: **{intent}**.\n\n"
        f"{call_instruction}\n\n"
        "After the tool returns results, synthesise a clear answer with:\n"
        "  ### Summary\n"
        "  ### Impact Analysis  (only if there are real impacts)\n"
        "  ### Evidence  (bullet list of specific files / functions / commits)\n"
        "Never invent evidence. If results are empty, say so."
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_msg),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # ── Step 3: run agent ─────────────────────────────────────────────────────
    tools    = toolkit.get_tools()
    agent    = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    result   = executor.invoke({"input": user_input})

    content = result.get("output", "The agent did not produce a final answer.")

    # Handle Google GenAI mixed-list outputs
    if isinstance(content, list):
        parsed_text = []
        for block in content:
            if isinstance(block, str):
                parsed_text.append(block)
            elif isinstance(block, dict) and "text" in block:
                parsed_text.append(block["text"])
        return "".join(parsed_text)

    return str(content)

def _build_cypher_chain(
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_pwd: str,
    llm,
    selected_repos: Optional[list] = None,
    top_k: int = 5,
):
    """Build and return a GraphCypherQAChain with the shared prompt templates.
    Single source of truth — replaces the two duplicate chain constructions
    that previously existed in query_graph_cypher() and the inline block."""
    graph = Neo4jGraph(url=neo4j_uri, username=neo4j_user, password=neo4j_pwd)

    if selected_repos:
        _repo_list = "[" + ", ".join(f'"{r}"' for r in selected_repos) + "]"
        _repo_hint = (
            f"6. REPO SCOPE & CROSS-REPO: The user has selected {len(selected_repos)} repositories: {_repo_list}. "
            f"When filtering File, Directory, or Function nodes, use `WHERE n.repo IN {_repo_list}`. "
            f"CRITICAL: `Module` nodes do NOT have a `repo` property. To find shared dependencies, you must traverse "
            f"from Files to Modules (e.g., `(f:File)-[r:DEPENDS_ON]->(m:Module)`)."
        )
    else:
        _repo_hint = (
            "6. REPO SCOPE: All repositories are in scope. "
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
        f"    {_repo_hint}\n"
        "    7. RETURN CLAUSES: Always return meaningful context. If asked \"Which files depend on X?\", do NOT just return `f.path`. Return `f.path`, `m.name`, and any relevant function names.\n"
        "    8. SYMBOL & BLAST-RADIUS ANALYSIS: The graph tracks module-level edges (`File-[:DEPENDS_ON]->Module`, `Function-[:CALLS]->Function`), NOT individual imported names "
        "For 'what breaks if symbol/function X changes', 'where would a bug in module Y surface', or any other symbol-level or blast-radius question, use this general two-hop pattern: "
        "(a) match the dependency edge from the target module/file, (b) walk to the functions declared in or calling from the dependent file, and fetch their `code` "
        "so the symbol-level filtering (does this function actually reference `init` or `Fore`?) happens by reading source text, not by the graph schema. "
        "Module-import template: `MATCH (f:File)-[r:DEPENDS_ON]->(m:Module), (f)-[:DECLARES]->(fn:Function) WHERE r.is_active = true AND toLower(m.name) CONTAINS toLower('<module>') RETURN f.path, fn.name, fn.code`. "
        "Function-call template: `MATCH (caller:Function)-[:CALLS]->(callee:Function) WHERE toLower(callee.name) CONTAINS toLower('<symbol>') RETURN caller.name, caller.code, callee.name`. "
        "Always prefer returning `fn.code` / `caller.code` over names alone when the question asks 'where', 'how', or 'what would break' — the answer step needs the source text to reason about specific symbols.\n"
        "    9. SPARSE-PROPERTY FALLBACK: Some properties (e.g. a commit's summary vs. its raw message) may only be populated on a subset of nodes. When a question could be answered by either of two known alternate properties, "
        "use `coalesce()` across them (e.g. `coalesce(c.summary_text, c.message)`) instead of querying only one and risking an empty result. "
        "Do not let a narrow property choice cause a false 'no data' answer when a broader, still-schema-valid query would have found it.\n"
        "    10. TEMPORAL FILTERING: The graph tracks historical code changes. Relationships like `DEPENDS_ON` have an `is_active` boolean property. By default, you MUST append `WHERE r.is_active = true` to your queries to ensure you only return current, active codebase architecture. ONLY omit this filter if the user explicitly asks about 'deleted', 'historical', or 'removed' code.\n\n"
        "    11. COMMIT SEARCHES: When searching for concepts in commit history, DO NOT use `CONTAINS`. You MUST use the fulltext index: `CALL db.index.fulltext.queryNodes('commit_summaries', '<search terms>') YIELD node AS c, score ...` "
        "    Schema:\n"
        "    {schema}\n\n"
        "    The question is:\n"
        "    {question}"
    )

    cypher_prompt = PromptTemplate(
        template=CYPHER_GENERATION_TEMPLATE,
        input_variables=["schema", "question"],
    )

    QA_GENERATION_TEMPLATE = """
    You are a senior software architect performing repository analysis.

    Database Results:
    {context}

    User Question:
    {question}

    Instructions for Formatting & Interpretation:
    - CRITICAL GRAPH READING RULE: If the Database Results contain file paths (e.g., `f.path`, `file`) alongside module names (e.g., `m.name`, `module`), it strictly means the FILE depends on (imports) the MODULE. Do not incorrectly claim the files "belong" to the module.
    - REPOSITORY RECOGNITION: The user might use shorthand like "bubbles". If the Database Results return file paths matching the query, assume those files belong to the requested repository. Do not claim you lack data just because the exact repository name isn't appended to every file string.
    - Write like a senior engineer explaining a codebase.
    - Start with a "### Summary" section.
    - Include a "### Impact Analysis" ONLY if there are tangible impacts. DO NOT list empty impacts.
    - Use `###` (H3) or inline bold text (e.g., `**Summary:**`) for all section headers — do NOT use `##` (H2) headers.
    - End with a "### Evidence" section listing the specific files, functions, modules, or commits referenced in your answer as bullet points.
    - Never invent evidence.
    - If the database results are truly empty, strictly reply that you do not have the data.
    """

    qa_prompt = PromptTemplate(
        template=QA_GENERATION_TEMPLATE,
        input_variables=["context", "question"]
    )

    return GraphCypherQAChain.from_llm(
        cypher_llm=llm,
        qa_llm=llm,
        graph=graph,
        verbose=True,
        cypher_prompt=cypher_prompt,
        qa_prompt=qa_prompt,
        allow_dangerous_requests=True,
        validate_cypher=True,
        top_k=top_k,
    )

def route_and_answer(
    user_input: str,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_pwd: str,
    google_api_key: str,
    driver,
    selected_repos: Optional[list] = None,
    top_k: int = 5,
) -> str:
    """
    Single-pass token-efficient router.

    Flow:
      1. classify_intent()  — 1 cheap LLM call (~150 input tokens, JSON only).
      2a. GENERAL_GRAPH_QUERY → GraphCypherQAChain directly (no agent overhead).
           Total: 2 LLM calls (Cypher gen + QA answer).
      2b. Specific intent    → Agent with pre-wired intent_data (no re-classify).
           Total: 2-3 LLM calls (agent + tool round-trip).

    Replaces the old dual-path that always ran both the QA chain AND the agent
    (4 LLM calls, with the QA result silently discarded).
    """
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=google_api_key,
        temperature=0.0,
    )

    # ── Step 1: classify once ─────────────────────────────────────────────────
    intent_data = classify_intent(user_input, llm)
    intent = intent_data["intent"]
    logger.info("route_and_answer: intent=%s | query=%.80s", intent, user_input)

    # ── Step 2a: simple path — QA chain only ─────────────────────────────────
    if intent == "GENERAL_GRAPH_QUERY":
        chain = _build_cypher_chain(
            neo4j_uri=neo4j_uri,
            neo4j_user=neo4j_user,
            neo4j_pwd=neo4j_pwd,
            llm=llm,
            selected_repos=selected_repos,
            top_k=top_k,
        )
        try:
            response = chain.invoke({"query": user_input})
            answer = response.get("result", "")
            if not answer:
                return "I couldn't find relevant data in the knowledge graph for that query."
            # Compute confidence (available for future display use)
            calculate_confidence(
                evidence_count=extract_evidence_count(answer),
                repo_count=len(selected_repos) if selected_repos else 1,
                exact_matches=1,
            )
            return answer
        except ValueError as exc:
            if "No tools" in str(exc) or "OutputParserException" in str(exc):
                return "❌ **Query Generation Failed:** The LLM couldn't map that question to the database schema."
            raise
        except Exception as exc:
            return f"❌ **Query Execution Failed:**\n```\n{str(exc)}\n```"

    # ── Step 2b: complex path — agent with pre-wired intent ───────────────────
    chain = _build_cypher_chain(
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_user,
        neo4j_pwd=neo4j_pwd,
        llm=llm,
        selected_repos=selected_repos,
        top_k=top_k,
    )
    toolkit = GraphRAGToolkit(driver=driver, cypher_qa_chain=chain)
    return create_and_run_agent(
        user_input=user_input,
        llm=llm,
        toolkit=toolkit,
        intent_data=intent_data,   # ← skips a second classify_intent() call
    )

def calculate_confidence(
        evidence_count: int,
        repo_count: int,
        exact_matches: int
) -> float:
    
    score = 0

    score += min(evidence_count * 10, 40)
    score += min(repo_count * 10, 30)
    score += min(exact_matches * 5, 30)

    return min(score, 100)

with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding: 0.5rem 0 1rem 0;">
            <div style="font-size:2.4rem;">🕸️</div>
            <div style="font-size:1.1rem; font-weight:700; color:#00d4ff; letter-spacing:0.5px;">GraphRAG Chat</div>
            <div style="font-size:0.72rem; color:rgba(255,255,255,0.4); margin-top:2px;">Neo4j · Gemini · Query-Only</div>
        </div>
        """, # <--- ADD THIS COMMA
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
    top_k = st.slider(
        "Top-K retrieval results",
        min_value=1,
        max_value=15,
        value=5,
        step=1,
        help="Controls how many relevant graph items are retrieved for context. A higher number gives the AI more background knowledge but consumes more API tokens.",
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
            with st.spinner("🔍 Searching knowledge graph…"):
                try:
                    _sel = st.session_state.get("selected_repos") or []
                    _available = st.session_state.get("available_repos") or []
                    _active_repos = _sel if (_sel and len(_sel) < len(_available)) else None
                    drv = _get_driver(neo4j_uri, neo4j_user, neo4j_pwd)

                    # ── Single-pass router: classify once, then route ──────────
                    # Replaces the old dual-path (QA chain + agent = 4 LLM calls)
                    # with classify_intent() → QA chain OR agent (2-3 LLM calls).
                    answer = route_and_answer(
                        user_input=user_input,
                        neo4j_uri=neo4j_uri,
                        neo4j_user=neo4j_user,
                        neo4j_pwd=neo4j_pwd,
                        google_api_key=google_api_key,
                        driver=drv,
                        selected_repos=_active_repos,
                        top_k=top_k,
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