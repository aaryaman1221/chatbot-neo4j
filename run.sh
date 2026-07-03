#!/usr/bin/env bash
# =============================================================================
# run.sh — One-Click Setup & Startup for GraphRAG Bootstrap
# =============================================================================
set -e

echo "======================================================================"
echo "         GraphRAG Ingestion Pipeline — One-Click Setup"
echo "======================================================================"
echo

# 1. Check if .env file exists
if [ ! -f .env ]; then
    echo "[SETUP] No .env file found. Creating template .env..."
    cat <<EOF > .env
# GitHub Personal Access Token (Required)
GITHUB_TOKEN=your_github_token_here

# Google Gemini API Key (Required for LLM summarization and embeddings)
GOOGLE_API_KEY=your_gemini_api_key_here

# Neo4j Database Credentials
NEO4J_URI=neo4j://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password123

# Target Repository & Pipeline Options
TARGET_REPO=neo4j/neo4j-graphrag-python
MAX_COMMITS=200
SKIP_LLM=false
FORCE_LLM_UPDATE=false
EOF
    echo "[SETUP] ⚠️  Created .env template."
    echo "[SETUP] 👉 Please edit .env with your GITHUB_TOKEN and GOOGLE_API_KEY, then re-run ./run.sh!"
    exit 1
fi

# 2. Check and start Neo4j Docker container if needed
if command -v docker >/dev/null 2>&1; then
    if docker info >/dev/null 2>&1; then
        if ! docker ps | grep -q neo4j-graphrag; then
            if docker ps -a | grep -q neo4j-graphrag; then
                echo "[DOCKER] Starting existing neo4j-graphrag container..."
                docker start neo4j-graphrag
            else
                echo "[DOCKER] Starting new Neo4j Docker container (neo4j-graphrag)..."
                docker run -d --name neo4j-graphrag \
                  -p 7474:7474 -p 7687:7687 \
                  -e NEO4J_AUTH=neo4j/password123 \
                  -e NEO4J_PLUGINS='["apoc"]' \
                  -e NEO4J_apoc_export_file_enabled=true \
                  -e NEO4J_apoc_import_file_enabled=true \
                  -e NEO4J_dbms_security_procedures_unrestricted=apoc.* \
                  -e NEO4J_dbms_memory_heap_initial__size=512m \
                  -e NEO4J_dbms_memory_heap_max__size=1G \
                  neo4j:5.20
                echo "[DOCKER] Waiting 15 seconds for Neo4j to initialize..."
                sleep 15
            fi
        else
            echo "[DOCKER] Neo4j container is already running ✓"
        fi
    else
        echo "[WARN] Docker daemon is not running. Assuming Neo4j is running externally."
    fi
else
    echo "[WARN] Docker is not installed. Assuming Neo4j is running externally."
fi

echo
echo "[SETUP] Verifying Python requirements..."
pip install -q -r requirements.txt

echo
echo "[RUN] Starting ingestion pipeline..."
python3 backend_ingest.py "$@"
