# GraphRAG Explorer: Neo4j + Gemini

A powerful Python-based application that ingests GitHub repository data into a Neo4j knowledge graph and provides a highly interactive Streamlit chat interface. It leverages an Agentic Workflow using LangChain and Google's Gemini models to answer complex queries about your codebase.

## Features

- **Backend Ingestion Pipeline** (`backend_ingest.py`): 
  - Pure Python CLI to bootstrap a GitHub repository into a Neo4j graph.
  - Parses code structure using Tree-sitter (AST). *(Note: Currently supports only Python and Go code)*
  - Uses Google Gemini for LLM-powered summarization of code and commits.
  - Models repositories, files, functions, commits, and their dependencies in Neo4j.
- **Frontend Agentic Chat** (`frontend_chat.py`):
  - Modern, responsive Streamlit UI with a rich aesthetic.
  - Powered by a LangChain Tool-Calling Agent.
  - Uses Neo4j Vector Retrieval and GraphCypherQAChain to dynamically answer questions about the repository structure, commit history, and code logic.

## Prerequisites

- Python 3.9+
- Docker (for easily running Neo4j)
- A GitHub Personal Access Token
- A Google Gemini API Key

## Installation

1. **Clone the repository** (if you haven't already):
   ```bash
   git clone <your-repo-url>
   cd neo4j-testing
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Environment Variables**:
   Create a `.env` file in the root directory with the following variables:
   ```env
   GITHUB_TOKEN=your_github_personal_access_token
   GOOGLE_API_KEY=your_gemini_api_key
   NEO4J_URI=neo4j://localhost:7687
   NEO4J_USER=neo4j
   NEO4J_PASSWORD=your_neo4j_password
   TARGET_REPO=owner/repo # e.g., neo4j/neo4j-graphrag-python
   MAX_COMMITS=200
   FORCE_LLM_UPDATE=false
   ```

## Usage

### 1. Start Neo4j Database

You can easily start a Neo4j instance with the APOC plugin using Docker:

```bash
docker run --name neo4j-graphrag \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/your_neo4j_password \
  -e NEO4J_PLUGINS='["apoc"]' \
  -e NEO4J_apoc_export_file_enabled=true \
  -e NEO4J_apoc_import_file_enabled=true \
  -e NEO4J_dbms_security_procedures_unrestricted=apoc.* \
  neo4j:5.20
```

### 2. Run the Backend Ingestion

Ingest the target GitHub repository into your Neo4j graph database. This step will extract files, AST structures, and commits, enriching them with LLM-generated summaries.

```bash
python backend_ingest.py
```

### 3. Start the Frontend Chat UI

Once the data is ingested, start the Streamlit application to interactively explore and chat with your codebase.

```bash
streamlit run frontend_chat.py
```

## Architecture Overview

- **Graph Model**: The graph models `Repository`, `File`, `Function`, `Commit`, `User`, and `Module` nodes, with rich relationships such as `DECLARES`, `CALLS`, `MODIFIED`, `AUTHORED`, and `DEPENDS_ON`.
- **LLM Agent**: The UI utilizes a `create_tool_calling_agent` from LangChain, equipping it with specific tools to perform impact analysis, search commit history, and execute generic Cypher queries against the graph.

## Token Optimization & Graph Retrieval

Unlike standard vector DB-based retrieval that relies solely on text chunk similarity, this application leverages **GraphRAG** via Neo4j. This results in significant token optimization:
- **Targeted Context**: Instead of padding the prompt with entire files or disjointed text chunks, the system retrieves precise subgraphs (e.g., a function, its direct dependencies, and recent commits).
- **Reduced Token Costs**: By filtering out irrelevant code and returning highly structured context, the LLM prompt remains small and dense with useful information, drastically reducing token usage per query.

## GraphRAG Explorer vs. GitHub Copilot CLI

While standard tools like GitHub Copilot CLI excel at autocomplete and local file edits, GraphRAG Explorer is built for deep, repository-scale analytical queries with several key advantages:

1. **Much Less Token Usage**: Copilot often sends large portions of your active workspace to the LLM context. GraphRAG queries the Neo4j database first, extracting only the exact relationships and feeding minimal, precise context to Gemini.
2. **Reliable Cross-Repo Analysis**: Copilot's context window struggles with deep dependencies spanning multiple files or repositories. GraphRAG naturally connects these entities via explicit graph edges (`DEPENDS_ON`, `CALLS`), enabling highly accurate cross-repo and cross-module reasoning.
3. **Commit History Integration**: GraphRAG models git history natively. You can ask *why* a piece of code changed or *who* introduced a bug, leveraging `MODIFIED` and `AUTHORED` relationships.
4. **Transparent Reasoning**: The retrieved context is based on explicit graph traversal. You can verify exactly which files, functions, and commits were used to generate the answer, minimizing black-box hallucinations.

## Dependencies

Major dependencies include:
- `neo4j` and `neo4j-graphrag`: For graph database interaction.
- `google-genai` and `langchain-google-genai`: For Gemini LLM capabilities.
- `langchain` and `langchain-neo4j`: For the agentic workflow and Cypher QA chains.
- `streamlit`: For the frontend UI.
- `PyGithub`: For fetching repository data.
- `tree-sitter` (optional but recommended): For precise code parsing.
