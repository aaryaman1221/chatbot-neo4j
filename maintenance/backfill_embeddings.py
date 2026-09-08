#!/usr/bin/env python3
# =============================================================================
# backfill_embeddings.py — Backfill missing Neo4j vector embeddings
# =============================================================================

import os
import sys
import time
from dotenv import load_dotenv
from neo4j import GraphDatabase
from tqdm import tqdm

# This script lives in maintenance/ — make the repo root importable so `ingest.*` resolves.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from ingest.ai_service import get_embeddings_batch

def main():
    load_dotenv(os.path.join(_REPO_ROOT, ".env"))

    NEO4J_URI      = os.environ.get("NEO4J_URI",      "neo4j://localhost:7687")
    NEO4J_USER     = os.environ.get("NEO4J_USER",     "neo4j")
    NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "password123")
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY") or ""

    if not GOOGLE_API_KEY:
        print("[ERROR] GOOGLE_API_KEY not found in environment or .env file.")
        sys.exit(1)

    print("Connecting to Neo4j...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        driver.verify_connectivity()
    except Exception as e:
        print(f"[ERROR] Could not connect to Neo4j: {e}")
        sys.exit(1)

    BATCH_SIZE = 30

    # 1. Backfill Type nodes
    print("\nFetching Type nodes missing embeddings...")
    with driver.session() as session:
        types = session.run("""
        MATCH (t:Type)
        WHERE t.embedding IS NULL AND t.code IS NOT NULL AND t.code <> ''
        RETURN t.id AS id, t.name AS name, t.code AS code
        """).data()

    total_types = len(types)
    print(f"Found {total_types} Type node(s) missing embeddings.")
    if total_types > 0:
        with tqdm(total=total_types, desc="Embedding Types") as pbar:
            for idx in range(0, total_types, BATCH_SIZE):
                batch = types[idx:idx + BATCH_SIZE]
                texts = [f"{t['name']}\n{t['code']}" for t in batch]
                
                try:
                    embeds = get_embeddings_batch(texts, GOOGLE_API_KEY)
                    
                    # Write embeddings back to database
                    update_data = []
                    for t, emb in zip(batch, embeds):
                        if emb:
                            update_data.append({"id": t["id"], "embedding": emb})
                            
                    if update_data:
                        with driver.session() as session:
                            session.run("""
                            UNWIND $data AS item
                            MATCH (t:Type {id: item.id})
                            SET t.embedding = item.embedding
                            """, data=update_data)
                            
                    pbar.update(len(batch))
                    time.sleep(6.0)
                except Exception as e:
                    print(f"\n[WARN] Failed to embed type batch: {e}")
                    pbar.update(len(batch))

    # 2. Backfill Function nodes
    print("\nFetching Function nodes missing embeddings...")
    with driver.session() as session:
        funcs = session.run("""
        MATCH (f:Function)
        WHERE f.embedding IS NULL AND f.code IS NOT NULL AND f.code <> ''
        RETURN f.id AS id, f.name AS name, f.code AS code
        """).data()

    total_funcs = len(funcs)
    print(f"Found {total_funcs} Function node(s) missing embeddings.")
    if total_funcs > 0:
        with tqdm(total=total_funcs, desc="Embedding Functions") as pbar:
            for idx in range(0, total_funcs, BATCH_SIZE):
                batch = funcs[idx:idx + BATCH_SIZE]
                texts = [f"{f['name']}\n{f['code']}" for f in batch]
                
                try:
                    embeds = get_embeddings_batch(texts, GOOGLE_API_KEY)
                    
                    update_data = []
                    for f, emb in zip(batch, embeds):
                        if emb:
                            update_data.append({"id": f["id"], "embedding": emb})
                            
                    if update_data:
                        with driver.session() as session:
                            session.run("""
                            UNWIND $data AS item
                            MATCH (f:Function {id: item.id})
                            SET f.embedding = item.embedding
                            """, data=update_data)
                            
                    pbar.update(len(batch))
                    time.sleep(6.0)
                except Exception as e:
                    print(f"\n[WARN] Failed to embed function batch: {e}")
                    pbar.update(len(batch))

    print("\n✅ Embedding backfill complete!")
    driver.close()

if __name__ == "__main__":
    main()
