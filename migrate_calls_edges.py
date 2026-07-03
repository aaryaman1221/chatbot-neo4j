#!/usr/bin/env python3
"""
Migration script to clean up ambiguous CALLS edges and establish cross-repo links
without requiring a full repository re-ingestion.

Usage:
    python migrate_calls_edges.py [--dry-run]
"""

import os
import sys
import argparse
import logging
from dotenv import load_dotenv
from neo4j import GraphDatabase

from backend_ingest import (
    CYPHER_LINK_REPO_DEPENDENCY,
    CYPHER_LINK_MODULE_TO_REPO,
    CYPHER_LINK_XREPO_CALLS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("migrate_calls_edges")

CYPHER_CLEANUP_AMBIGUOUS_CALLS = """
MATCH (caller:Function)-[r:CALLS]->(callee:Function)
WHERE caller.repo = callee.repo
  AND caller.filepath <> callee.filepath
  AND NOT coalesce(r.cross_repo, false)
WITH caller, r, callee
MATCH (local:Function {name: callee.name, filepath: caller.filepath, repo: caller.repo})
WHERE local <> callee
DELETE r
RETURN count(*) AS deleted_edges
"""

CYPHER_GET_REPOS = """
MATCH (r:Repository)
RETURN r.full_name AS full_name
"""

def main():
    parser = argparse.ArgumentParser(description="Clean up ambiguous CALLS edges and migrate cross-repo links.")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without modifying the database.")
    args = parser.parse_args()

    load_dotenv()
    uri = os.environ.get("NEO4J_URI", "neo4j://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "")

    if not password:
        logger.error("NEO4J_PASSWORD environment variable not set. Please check your .env file.")
        sys.exit(1)

    logger.info("Connecting to Neo4j at %s...", uri)
    with GraphDatabase.driver(uri, auth=(user, password)) as driver:
        with driver.session() as session:
            if args.dry_run:
                logger.info("[DRY RUN] Would clean up ambiguous CALLS edges where local callee exists.")
            else:
                logger.info("1. Cleaning up ambiguous CALLS edges where a file-local callee exists...")
                res = session.run(CYPHER_CLEANUP_AMBIGUOUS_CALLS)
                record = res.single()
                deleted = record["deleted_edges"] if record else 0
                logger.info("✅ Deleted %d ambiguous intra-repo CALLS edges.", deleted)

            logger.info("2. Creating Module-to-Repository representation links...")
            if not args.dry_run:
                res_mod = session.run(CYPHER_LINK_MODULE_TO_REPO)
                sum_mod = res_mod.consume()
                logger.info("✅ Module -> Repository edges created: %d", sum_mod.counters.relationships_created)
            else:
                logger.info("[DRY RUN] Would run CYPHER_LINK_MODULE_TO_REPO.")

            logger.info("3. Re-running cross-repo linking for all repository pairs in the graph...")
            repos = [r["full_name"] for r in session.run(CYPHER_GET_REPOS) if r["full_name"]]
            logger.info("Found %d repositories: %s", len(repos), repos)

            for parent in repos:
                for helper in repos:
                    if parent == helper:
                        continue
                    helper_name = helper.split("/")[-1]
                    logger.info("Checking cross-repo links: %s -> %s (prefix: %s)", parent, helper, helper_name)
                    if not args.dry_run:
                        res_use = session.run(
                            CYPHER_LINK_REPO_DEPENDENCY,
                            parent_repo=parent,
                            helper_repo=helper,
                            helper_repo_name=helper_name,
                        )
                        sum_use = res_use.consume()
                        res_calls = session.run(
                            CYPHER_LINK_XREPO_CALLS,
                            parent_repo=parent,
                            helper_repo=helper,
                            helper_prefix=helper_name,
                        )
                        sum_calls = res_calls.consume()
                        if sum_use.counters.relationships_created > 0 or sum_calls.counters.relationships_created > 0:
                            logger.info(
                                "   Linked %s -> %s: %d USES_REPO, %d cross-repo CALLS",
                                parent, helper,
                                sum_use.counters.relationships_created,
                                sum_calls.counters.relationships_created,
                            )
                    else:
                        logger.info("[DRY RUN] Would link %s -> %s", parent, helper)

    logger.info("🎉 Migration script completed successfully.")

if __name__ == "__main__":
    main()
