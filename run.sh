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

# 2. Start the local Neo4j graph via docker compose (skip if NEO4J_URI is remote,
#    e.g. an AuraDB neo4j+s:// endpoint, or if Docker is unavailable).
NEO4J_URI_VALUE="$(grep -E '^NEO4J_URI=' .env | head -1 | cut -d= -f2-)"
case "$NEO4J_URI_VALUE" in
    neo4j+s://*|neo4j+ssc://*|bolt+s://*)
        echo "[DOCKER] NEO4J_URI points at a managed/remote graph — skipping local Neo4j."
        ;;
    *)
        if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
            echo "[DOCKER] Ensuring local Neo4j is up (docker compose up -d neo4j)..."
            docker compose up -d neo4j
            echo "[DOCKER] Waiting for Neo4j to become healthy..."
            for _ in $(seq 1 30); do
                status="$(docker inspect -f '{{.State.Health.Status}}' graphrag-neo4j 2>/dev/null || echo starting)"
                [ "$status" = "healthy" ] && { echo "[DOCKER] Neo4j is healthy ✓"; break; }
                sleep 2
            done
        else
            echo "[WARN] Docker not available. Assuming Neo4j is running externally (check NEO4J_URI)."
        fi
        ;;
esac

echo
echo "[SETUP] Verifying Python requirements..."
pip install -q -r requirements.txt

echo
echo "[RUN] Starting ingestion pipeline..."
python3 backend_ingest.py "$@"
