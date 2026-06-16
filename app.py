# =============================================================================
# LOCAL GRAPHRAG DEMO: Neo4j + GitHub Issues + Google Gemini AI
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
#
# Install with:
#   pip install -r requirements.txt
#
# Run with:
#   streamlit run app.py
# =============================================================================

import time
import traceback
from typing import Optional

import streamlit as st
from github import Github, GithubException
from neo4j import GraphDatabase, exceptions as neo4j_exc

# ── Neo4j GraphRAG imports ───────────────────────────────────────────────────
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
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

_init_session()


# =============================================================================
#  HELPER UTILITIES
# =============================================================================

def _status_dot(color: str) -> str:
    return f'<span class="status-dot {color}"></span>'


def _check_neo4j(uri: str, user: str, pwd: str) -> tuple[bool, str]:
    """Attempt a quick connectivity check against Neo4j."""
    try:
        drv = GraphDatabase.driver(uri, auth=(user, pwd))
        drv.verify_connectivity()
        return True, "Connected"
    except Exception as exc:
        return False, str(exc)


def _get_neo4j_driver(uri: str, user: str, pwd: str):
    if st.session_state.neo4j_driver is None:
        st.session_state.neo4j_driver = GraphDatabase.driver(uri, auth=(user, pwd))
    return st.session_state.neo4j_driver


def _embed_text(text: str, api_key: str, model: str = "models/gemini-embedding-2-preview") -> list[float]:
    """Generate an embedding vector using Google GenAI.

    Note: google-genai 1.x requires `contents` to be a list.
    Returns the first embedding's float values.
    """
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


# =============================================================================
#  INGESTION PIPELINE
# =============================================================================
# File extensions to scan for full-file dependency analysis
SOURCE_EXTENSIONS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".java", ".rs",
    ".rb", ".php", ".c", ".cpp", ".h", ".hpp", ".cs", ".swift",
    ".kt", ".scala", ".r", ".R", ".vue", ".svelte",
}

# Some source-like files do not have a traditional extension.
SOURCE_FILENAMES = {
    "go.mod",
    "go.work",
}

# Patterns for identifying entry-point files
ENTRY_POINT_NAMES = {
    "main", "index", "app", "server", "__main__", "manage",
    "wsgi", "asgi", "cli", "entrypoint",
}

# Directory names that indicate shared utilities
UTILITY_DIRS = {
    "utils", "util", "lib", "libs", "common", "shared",
    "helpers", "helper", "core", "pkg", "internal",
}

#  All IGNORED methods to remove unnecessary files from initial bootstrapping process
IGNORED_DIRECTORIES = {
    "node_modules", "venv", ".venv", "env", "dist", "build", ".next", ".git"
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

INGEST_CYPHER = """
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
  ON CREATE SET commit.timestamp = $committed_at

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

INGEST_TREE_CYPHER = """
MERGE (repo:Repository {full_name: $repo_full_name})
MERGE (parent_dir:Directory {path: $parent_path})
  ON CREATE SET parent_dir.repo = $repo_full_name

MERGE (child:Item {path: $child_path}) // Item can be labeled as File or Directory later
  ON CREATE SET child.repo = $repo_full_name

MERGE (parent_dir)-[:CONTAINS]->(child)
"""

VECTOR_INDEX_CYPHER = """
CREATE VECTOR INDEX issue_embeddings IF NOT EXISTS
FOR (i:Issue) ON (i.embedding)
OPTIONS {
  indexConfig: {
    `vector.dimensions`: 3072,
    `vector.similarity_function`: 'cosine'
  }
}
"""

def update_neo4j_knowledge_graph(driver, repo_full_name, commit_sha, modified_files, dependencies, actor_login, committed_at):
    """
    Replaces update_knowledge_graph(). Integrates commits, modified files, 
    and dependencies into the Neo4j database.
    """
    actor_login = actor_login or "unknown_author"
    committed_at = committed_at or ""

    with driver.session() as session:
        # 1. Create Commit and Author
        session.run(
            INGEST_COMMIT_CYPHER,
            repo_full_name=repo_full_name,
            actor_login=actor_login,
            commit_sha=commit_sha,
            committed_at=committed_at
        )

        # 2. Link Commit to Modified Files
        for filepath in modified_files:
            session.run(
                INGEST_FILE_CYPHER,
                repo_full_name=repo_full_name,
                commit_sha=commit_sha,
                filepath=filepath
            )

        # 3. Map Module Dependencies
        for source, rel, target in dependencies:
            # Note: `rel` is ignored in this exact cypher to standardise on DEPENDS_ON, 
            # but you can dynamically set the relationship type if needed.
            session.run(
                INGEST_DEPENDENCY_CYPHER,
                filepath=source,
                target_module=target
            )

def _is_noise_file(filename):
    lower_path = filename.lower().replace("\\", "/")
    
    # 1. Check directories (do this first, as it catches entire folders of noise quickly)
    path_parts = set(lower_path.split("/"))
    if path_parts.intersection(IGNORED_DIRECTORIES):
        return True

    # Extract just the filename for the next checks
    name = lower_path.rsplit("/", 1)[-1]
    
    # 2. Check exact filenames
    if name in IGNORED_FILENAMES:
        return True
        
    # 3. Check suffixes
    return name.endswith(IGNORED_SUFFIXES)



def scan_neo4j_repo_tree(driver, github_token, repo_full_name):
    """
    Replaces scan_repo_tree(). Fetches the full file tree from GitHub and 
    builds the directory/file hierarchy in Neo4j.
    """
    gh = Github(github_token)
    repo = gh.get_repo(repo_full_name)
    default_branch = repo.default_branch
    
    # Fetch the recursive tree
    tree = repo.get_git_tree(default_branch, recursive=True).tree
    source_files = []

    with driver.session() as session:
        for item in tree:
            path = item.path
            item_type = item.type  # "blob" (file) or "tree" (directory)

            # Bring over your _is_noise_file and IGNORED_DIRECTORIES logic here
            if _is_noise_file(path):
                continue

            # Assign specific labels based on the item type
            if item_type == "tree":
                session.run(f"MATCH (n {{path: '{path}'}}) SET n:Directory")
            elif item_type == "blob":
                session.run(f"MATCH (n {{path: '{path}'}}) SET n:File")
                # Collect for deep scanning later
                if any(path.endswith(ext) for ext in SOURCE_EXTENSIONS): 
                    source_files.append(path)

            # Link to parent directory
            if "/" in path:
                parent_path = path.rsplit("/", 1)[0]
                session.run(
                    INGEST_TREE_CYPHER,
                    repo_full_name=repo_full_name,
                    parent_path=parent_path,
                    child_path=path
                )
                
    return source_files


def run_ingestion(
    github_token: str,
    google_api_key: str,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_pwd: str,
    repo_full_name: str,
    n_issues: int = 15,
):
    """Full ingestion pipeline: GitHub → embeddings → Neo4j."""
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
        open_issues = list(repo.get_issues(state="open", sort="created", direction="desc")[:n_issues])
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

    progress.progress(30, text="🔧 Creating vector index…")

    # 4 ── Create vector index ──────────────────────────────────────────────
    try:
        with driver.session() as session:
            session.run(VECTOR_INDEX_CYPHER)
    except Exception as exc:
        # Index may already exist with different config — warn but continue
        st.warning(f"⚠️ Vector index note: {exc}")

    progress.progress(40, text="🌳 Building repository file tree in Neo4j…")

    # 4.5 ── Build File Tree & Codebase Graph ──────────────────────────────
    try:
        # Build the structural tree
        scannable_files = scan_neo4j_repo_tree(driver, github_token, repo_full_name)
        
        # (Optional) Implement the deep dependency scan here using scan_file_contents logic
        # For a full backfill, you would iterate over repo.get_commits() and call 
        # update_neo4j_knowledge_graph() for each commit.
        
        st.toast(f"Indexed {len(scannable_files)} source files into the graph!", icon="🌳")
    except Exception as exc:
        st.warning(f"⚠️ Codebase tree ingestion failed: {exc}")

    # 5 ── Embed & ingest each issue ───────────────────────────────────────
    embed_status = st.empty()
    failed_embeds = []

    for idx, issue in enumerate(open_issues):
        pct = 35 + int((idx / actual_count) * 60)
        progress.progress(pct, text=f"🧠 Embedding issue {idx + 1}/{actual_count}: #{issue.number}…")
        embed_status.caption(f"**#{issue.number}** — {issue.title[:80]}…")

        try:
            body_text = _safe_body(issue)
            embedding = _embed_text(body_text, google_api_key)
        except Exception as exc:
            failed_embeds.append(issue.number)
            embedding = []   # store empty; still ingest the node
            st.warning(f"⚠️ Embedding failed for issue #{issue.number}: {exc}")

        try:
            labels = [lbl.name for lbl in issue.labels]
            with driver.session() as session:
                session.run(
                    INGEST_CYPHER,
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

        time.sleep(0.1)   # gentle rate-limit buffer

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
        st.warning(f"⚠️ Embeddings skipped for issues: {failed_embeds}. "
                   "Those nodes exist in the graph but won't appear in vector search.")

    return True


# =============================================================================
#  GRAPHRAG QUERY PIPELINE
# =============================================================================

class GeminiEmbedderWrapper:
    """
    Thin wrapper so google-genai embeddings conform to the
    neo4j_graphrag Embedder interface (embed_query method).

    neo4j_graphrag calls embedder.embed_query(text) for retrieval.
    google-genai 1.x requires contents to be a list.
    """
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
    """Instantiate VectorRetriever + GraphRAG pipeline."""
    driver = _get_neo4j_driver(neo4j_uri, neo4j_user, neo4j_pwd)
    embedder = GeminiEmbedderWrapper(api_key=google_api_key)

    retriever = VectorRetriever(
        driver=driver,
        index_name=index_name,
        embedder=embedder,
        return_properties=["number", "title", "body", "state", "url", "labels", "created"],
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=google_api_key,
        temperature=0.2
    )
    rag = GraphRAG(retriever=retriever, llm=llm)
    return rag


def query_graphrag(
    question: str,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_pwd: str,
    google_api_key: str,
    top_k: int = 5,
) -> str:
    """Run the GraphRAG pipeline and return an answer string."""
    rag = build_rag_pipeline(neo4j_uri, neo4j_user, neo4j_pwd, google_api_key)
    result = rag.search(
        query_text=question,
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

    st.markdown("---")

    # ── Ingest button ──────────────────────────────────────────────────────
    ingest_clicked = st.button("⚡ Ingest Repository", use_container_width=True)

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

    # ── Status summary ─────────────────────────────────────────────────────
    if st.session_state.ingestion_done:
        st.markdown("---")
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
            <span class="pipeline-badge">PyGithub</span>Issues<br>
            <span class="pipeline-badge">Gemini</span>Embeddings<br>
            <span class="pipeline-badge">Neo4j</span>Graph Store<br>
            <span class="pipeline-badge">VectorRAG</span>Retrieval<br>
            <span class="pipeline-badge">Gemini 2.5</span>Generation
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
#  MAIN PANEL
# =============================================================================

# ── Hero header ───────────────────────────────────────────────────────────────
st.markdown(
    '<h1 class="hero-title">🕸️ GraphRAG Explorer</h1>'
    '<p class="hero-sub">Query GitHub repository issues using knowledge graph + vector retrieval powered by Google Gemini</p>',
    unsafe_allow_html=True,
)

# ── Architecture overview ──────────────────────────────────────────────────────
with st.expander("📐 Architecture Overview", expanded=False):
    cols = st.columns(5)
    steps = [
        ("1️⃣", "GitHub API", "Fetch open issues via PyGithub"),
        ("2️⃣", "Gemini Embed", "Generate 3072-dim vectors for each issue body"),
        ("3️⃣", "Neo4j Graph", "Store nodes (Repo, Issue, User) + relationships"),
        ("4️⃣", "VectorRetriever", "Cosine similarity search on issue_embeddings index"),
        ("5️⃣", "Gemini 2.5 Flash", "Augmented generation with retrieved context"),
    ]
    for col, (num, title, desc) in zip(cols, steps):
        with col:
            st.markdown(
                f"""
                <div class="metric-card" style="text-align:center; min-height:110px;">
                    <div style="font-size:1.6rem;">{num}</div>
                    <div style="font-size:0.8rem; font-weight:600; color:#00d4ff; margin:4px 0;">{title}</div>
                    <div style="font-size:0.7rem; color:rgba(255,255,255,0.45);">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

st.markdown("---")

# ── Readiness check banner ──────────────────────────────────────────────────
if not st.session_state.ingestion_done:
    st.info(
        "👈 **Get started**: Fill in your API keys and Neo4j credentials in the sidebar, "
        "then click **⚡ Ingest Repository** to populate the knowledge graph. "
        "Once ingestion completes, ask any question below!"
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
            ✅ Knowledge graph ready &nbsp;·&nbsp; <strong>{st.session_state.issue_count} issues</strong>
            from <code>{st.session_state.repo_name}</code> indexed &nbsp;·&nbsp;
            Ask anything below ↓
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Suggested questions ──────────────────────────────────────────────────────
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
        "Are there any issues related to documentation?",
        "What feature requests are open?",
        "List issues related to authentication or security",
    ]
    for i, suggestion in enumerate(suggestions):
        with suggestion_cols[i % 3]:
            if st.button(suggestion, key=f"sugg_{i}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": suggestion})
                st.rerun()

# ── Chat history ─────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑‍💻" if msg["role"] == "user" else "🤖"):
        st.markdown(msg["content"])

# ── Chat input ────────────────────────────────────────────────────────────────
user_input = st.chat_input(
    "Ask a question about the repository issues…",
    disabled=not st.session_state.ingestion_done,
)

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(user_input)

    # ── Validate required credentials ─────────────────────────────────────
    if not google_api_key:
        with st.chat_message("assistant", avatar="🤖"):
            st.error("❌ Google API Key is missing. Please enter it in the sidebar.")
        st.session_state.messages.append({
            "role": "assistant",
            "content": "❌ Google API Key is missing. Please enter it in the sidebar.",
        })
    elif not neo4j_uri or not neo4j_user or not neo4j_pwd:
        with st.chat_message("assistant", avatar="🤖"):
            st.error("❌ Neo4j credentials are incomplete. Please check the sidebar.")
        st.session_state.messages.append({
            "role": "assistant",
            "content": "❌ Neo4j credentials are incomplete. Please check the sidebar.",
        })
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

# ── Footer ─────────────────────────────────────────────────────────────────────
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
