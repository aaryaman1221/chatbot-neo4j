import json
import logging
import os
import traceback
from typing import Optional, List

from fastapi import FastAPI, HTTPException, Header, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from dotenv import load_dotenv

load_dotenv()

from neo4j import GraphDatabase, exceptions as neo4j_exc

# LangChain / Google GenAI imports
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="GraphRAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
# Intent Classification & GraphRAG Toolkit
# -----------------------------------------------------------------------------

INTENT_LABELS = [
    "DEPENDENCY_TRAVERSAL",
    "BUG_SURFACE_ANALYSIS",
    "BLAST_RADIUS",
    "COMMIT_SEARCH",
    "GENERAL_GRAPH_QUERY",
]

_INTENT_SYSTEM = """\
You are a query-intent classifier for a code-repository knowledge graph.
Classify the user's question into EXACTLY ONE of these intents:

  DEPENDENCY_TRAVERSAL  — asks which files/functions/components call, use,
                          import, or depend on a module, function, or symbol.

  BUG_SURFACE_ANALYSIS  — asks where a bug or defect in X would show up,
                          propagate, or be visible in the codebase.

  BLAST_RADIUS          — asks what breaks / is at risk if X is changed,
                          removed, or refactored. Forward-impact questions.

  COMMIT_SEARCH         — asks about git history, commit messages, why/when a
                          change was made, recent fixes, PR history.

  GENERAL_GRAPH_QUERY   — everything else: architecture overviews, file
                          locations, listing nodes, open-ended questions.

Also extract these three fields: module, symbol, repo, path_hint.
Respond with ONLY a JSON object:
{"intent": "<LABEL>", "module": "<str>", "symbol": "<str>", "repo": "<str>", "path_hint": "<str>"}
"""

def classify_intent(question: str, llm) -> dict:
    try:
        response = llm.invoke([
            SystemMessage(content=_INTENT_SYSTEM),
            HumanMessage(content=question),
        ])
        raw = response.content.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
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

class GraphRAGToolkit:
    def __init__(self, driver, cypher_qa_chain):
        self.driver = driver
        self.cypher_qa_chain = cypher_qa_chain

    def run_dependency_traversal(self, module: str, symbol: str = "", repo: str = "", path_hint: str = "") -> str:
        code_symbol = symbol if (symbol and len(symbol) < 40 and " " not in symbol) else ""
        results = []

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
                logger.warning("Pass-1 query failed: %s", exc)

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
                logger.warning("Pass-2 query failed: %s", exc)

        if not results:
            return "No callers found."
        return json.dumps(results, default=str, indent=2)

    def run_blast_radius(self, module: str, symbol: str = "", repo: str = "", path_hint: str = "") -> str:
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
                return "No blast-radius results."
            return json.dumps(results, default=str, indent=2)
        except Exception as exc:
            return f"Error: {exc}"

    def run_bug_surface_analysis(self, module: str, symbol: str = "", repo: str = "", path_hint: str = "") -> str:
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
                return "No surface-analysis results."
            return json.dumps(results, default=str, indent=2)
        except Exception as exc:
            return f"Error: {exc}"

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
            return f"Error: {exc}"

    def run_generic_query(self, question: str) -> str:
        try:
            response = self.cypher_qa_chain.invoke({"query": question})
            return response.get("result", "No result returned.")
        except Exception as exc:
            return f"Error: {exc}"

    def get_tools(self) -> list:
        toolkit = self
        @tool
        def analyze_blast_radius(target_module: str, symbol: str = "", repo: str = "", path_hint: str = "") -> str:
            """Forward-impact analysis: given a module."""
            return toolkit.run_blast_radius(module=target_module, symbol=symbol, repo=repo, path_hint=path_hint)
        @tool
        def find_callers_and_dependents(target_module: str, symbol: str = "", repo: str = "", path_hint: str = "") -> str:
            """Find all files and functions that call."""
            return toolkit.run_dependency_traversal(module=target_module, symbol=symbol, repo=repo, path_hint=path_hint)
        @tool
        def surface_bug_impact(target_module: str, symbol: str = "", repo: str = "", path_hint: str = "") -> str:
            """Return every file and function that imports a module."""
            return toolkit.run_bug_surface_analysis(module=target_module, symbol=symbol, repo=repo, path_hint=path_hint)
        @tool
        def search_commit_history(semantic_query: str) -> str:
            """Search commit messages."""
            return toolkit.run_commit_search(query=semantic_query)
        @tool
        def generic_graph_query(question: str) -> str:
            """General graph query."""
            return toolkit.run_generic_query(question=question)
        return [analyze_blast_radius, find_callers_and_dependents, surface_bug_impact, search_commit_history, generic_graph_query]

_INTENT_TO_TOOL = {
    "DEPENDENCY_TRAVERSAL": "find_callers_and_dependents",
    "BUG_SURFACE_ANALYSIS": "surface_bug_impact",
    "BLAST_RADIUS":         "analyze_blast_radius",
    "COMMIT_SEARCH":        "search_commit_history",
    "GENERAL_GRAPH_QUERY":  "generic_graph_query",
}

def _build_cypher_chain(neo4j_uri, neo4j_user, neo4j_pwd, llm, selected_repos=None, top_k=5):
    graph = Neo4jGraph(url=neo4j_uri, username=neo4j_user, password=neo4j_pwd)
    _repo_hint = ""
    if selected_repos:
        _repo_list = "[" + ", ".join(f'"{r}"' for r in selected_repos) + "]"
        _repo_hint = f"WHERE n.repo IN {_repo_list}"
    
    cypher_prompt = PromptTemplate(
        template="Task: Generate Cypher.\nSchema: {schema}\nQuestion: {question}\n" + _repo_hint,
        input_variables=["schema", "question"],
    )
    qa_prompt = PromptTemplate(
        template="Answer based on context: {context}\nQuestion: {question}",
        input_variables=["context", "question"]
    )
    return GraphCypherQAChain.from_llm(
        cypher_llm=llm, qa_llm=llm, graph=graph, verbose=True,
        cypher_prompt=cypher_prompt, qa_prompt=qa_prompt,
        allow_dangerous_requests=True, validate_cypher=True, top_k=top_k,
    )

def route_and_answer(user_input, neo4j_uri, neo4j_user, neo4j_pwd, google_api_key, driver, selected_repos, top_k):
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=google_api_key, temperature=0.0)
    intent_data = classify_intent(user_input, llm)
    intent = intent_data["intent"]

    if intent == "GENERAL_GRAPH_QUERY":
        chain = _build_cypher_chain(neo4j_uri, neo4j_user, neo4j_pwd, llm, selected_repos, top_k)
        try:
            return chain.invoke({"query": user_input}).get("result", "No result.")
        except Exception as exc:
            return f"❌ Query Failed: {exc}"

    chain = _build_cypher_chain(neo4j_uri, neo4j_user, neo4j_pwd, llm, selected_repos, top_k)
    toolkit = GraphRAGToolkit(driver=driver, cypher_qa_chain=chain)
    
    tool_name = _INTENT_TO_TOOL.get(intent, "generic_graph_query")
    system_msg = (
        f"You are a senior architect. Intent: {intent}.\n"
        f"Use `{tool_name}` to get context, then answer."
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_msg),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    tools = toolkit.get_tools()
    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    try:
        res = executor.invoke({"input": user_input})
        out = res.get("output", "")
        if isinstance(out, list):
            return "".join([b.get("text", "") if isinstance(b, dict) else str(b) for b in out])
        return str(out)
    except Exception as exc:
        return f"❌ Agent Error: {exc}"

# -----------------------------------------------------------------------------
# API Endpoints
# -----------------------------------------------------------------------------

def get_neo4j_driver(uri, user, password):
    if not all([uri, user, password]):
        raise HTTPException(status_code=400, detail="Neo4j credentials are required in headers")
    return GraphDatabase.driver(uri, auth=(user, password))

@app.get("/api/stats")
def get_stats(
    x_neo4j_uri: str = Header(...),
    x_neo4j_user: str = Header(...),
    x_neo4j_password: str = Header(...)
):
    try:
        drv = get_neo4j_driver(x_neo4j_uri, x_neo4j_user, x_neo4j_password)
        stats = _query_graph_stats(drv)
        drv.close()
        return stats
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
        repos = _fetch_available_repos(drv)
        drv.close()
        return {"repos": repos}
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
        answer = route_and_answer(
            user_input=req.query,
            neo4j_uri=x_neo4j_uri,
            neo4j_user=x_neo4j_user,
            neo4j_pwd=x_neo4j_password,
            google_api_key=google_api_key,
            driver=drv,
            selected_repos=req.selected_repos,
            top_k=req.top_k,
        )
        drv.close()
        return {"answer": answer}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
