#!/usr/bin/env python3
# =============================================================================
# backend_ingest.py — CLI Wrapper for GitHub → Neo4j GraphRAG Bootstrap
# =============================================================================
#
# PURPOSE:
#   A lightweight CLI entry point to bootstrap a GitHub repository into a
#   Neo4j knowledge graph using the modular `ingest` package.
#
# =============================================================================

import os
import sys
from dotenv import load_dotenv
from neo4j import GraphDatabase

# Re-exported for backward compatibility with migrate_calls_edges.py and other scripts
from ingest.queries import (
    CYPHER_LINK_REPO_DEPENDENCY,
    CYPHER_LINK_MODULE_TO_REPO,
    CYPHER_LINK_XREPO_CALLS,
)
from ingest.pipeline import bootstrap, resolve_cross_repo_edges


if __name__ == "__main__":
    load_dotenv()

    if "--link-repos" in sys.argv:
        NEO4J_URI      = os.environ.get("NEO4J_URI",      "neo4j://localhost:7687")
        NEO4J_USER     = os.environ.get("NEO4J_USER",     "neo4j")
        NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "")
        TARGET_REPO    = os.environ.get("TARGET_REPO",    "")
        HELPER_REPOS   = [r.strip() for r in os.environ.get("HELPER_REPOS", "").split(",") if r.strip()]
        
        idx = sys.argv.index("--link-repos")
        args = sys.argv[idx + 1:]
        if len(args) >= 2:
            parent = args[0]
            helpers = args[1:]
        elif TARGET_REPO and HELPER_REPOS:
            parent = TARGET_REPO
            helpers = HELPER_REPOS
        else:
            print("[ERROR] --link-repos requires <parent_repo> <helper_repo1>... or TARGET_REPO and HELPER_REPOS env vars.")
            sys.exit(1)
            
        print(f"Linking cross-repo edges for parent '{parent}' with helpers: {helpers}")
        with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)) as driver:
            for h in helpers:
                resolve_cross_repo_edges(driver, parent, h)
        print("✅ Cross-repo linking COMPLETE.")
        sys.exit(0)

    GITHUB_TOKEN   = os.environ.get("GITHUB_TOKEN", "")
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
    NEO4J_URI      = os.environ.get("NEO4J_URI",      "neo4j://localhost:7687")
    NEO4J_USER     = os.environ.get("NEO4J_USER",     "neo4j")
    NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "")
    TARGET_REPO       = os.environ.get("TARGET_REPO",       "neo4j/neo4j-graphrag-python")
    MAX_COMMITS       = int(os.environ.get("MAX_COMMITS", "200"))
    SKIP_LLM          = os.environ.get("SKIP_LLM",          "false").lower() == "true"
    FORCE_LLM_UPDATE  = os.environ.get("FORCE_LLM_UPDATE",  "false").lower() == "true"

    missing = []
    if not GITHUB_TOKEN:   missing.append("GITHUB_TOKEN")
    if not NEO4J_PASSWORD: missing.append("NEO4J_PASSWORD")

    if missing:
        print(f"[ERROR] Missing required environment variables: {', '.join(missing)}")
        print("        Create a .env file or export them before running.")
        raise SystemExit(1)

    if not GOOGLE_API_KEY:
        print("[WARN] GOOGLE_API_KEY not set — LLM summaries will use heuristics.")

    bootstrap(
        repo_full_name=TARGET_REPO,
        github_token=GITHUB_TOKEN,
        google_api_key=GOOGLE_API_KEY,
        neo4j_uri=NEO4J_URI,
        neo4j_user=NEO4J_USER,
        neo4j_password=NEO4J_PASSWORD,
        max_commits=MAX_COMMITS,
        skip_llm=SKIP_LLM,
        force_llm_update=FORCE_LLM_UPDATE,
    )