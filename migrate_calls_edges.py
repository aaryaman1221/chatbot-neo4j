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
from ingest.parser import parse_go_ast, parse_python_ast

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

CYPHER_CLEANUP_FALSE_BARE_CALLS = """
MATCH (caller:Function)-[r:CALLS]->(callee:Function)
WHERE caller.repo = callee.repo
  AND caller.filepath <> callee.filepath
  AND split(caller.filepath, '/')[0..-1] <> split(callee.filepath, '/')[0..-1]
  AND NOT coalesce(r.cross_repo, false)
  AND NOT callee.name CONTAINS '.'
  AND (
       callee.name IN ['New', 'Run', 'Execute', 'Close', 'Open', 'Init', 'String', 'Read', 'Write', 'Update', 'Focus', 'Blur', 'Blink', 'Start', 'Stop', 'Reset', 'Clear', 'Add', 'Remove', 'Delete', 'Get', 'Set', 'List', 'Find', 'Check', 'Verify', 'Validate', 'Parse', 'Format', 'Print', 'Println', 'Error', 'Fatal', 'Panic', 'Log', 'Debug', 'Info', 'Warn', 'Main', 'Test', 'Setup', 'Teardown', 'Config', 'Load', 'Save', 'Create', 'Send', 'Process', 'Handle']
    OR size(callee.name) < 5
    OR NOT EXISTS {
       MATCH (caller_file:File {path: caller.filepath, repo: caller.repo})-[:DEPENDS_ON]->(m:Module)
       WHERE toLower(callee.filepath) CONTAINS toLower(m.name) OR toLower(m.name) ENDS WITH toLower(split(callee.filepath, '/')[0])
    }
  )
DELETE r
RETURN count(*) AS deleted_edges
"""

CYPHER_CLEANUP_FALSE_XREPO_CALLS = """
MATCH (caller:Function)-[r:CALLS {cross_repo: true}]->(callee:Function)
WHERE NOT EXISTS {
  MATCH (f:File {repo: caller.repo})-[:USES_REPO]->(r_helper:Repository {full_name: callee.repo})
}
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
            logger.info("0. Re-populating AST qualified_calls on Function nodes from stored source code...")
            if not args.dry_run:
                rows = session.run("MATCH (f:Function) WHERE f.qualified_calls IS NULL AND f.code IS NOT NULL RETURN f.id AS id, f.filepath AS fp, f.code AS code").data()
                logger.info("Found %d Function nodes to scan for call expressions...", len(rows))
                batch_updates = []
                for r in rows:
                    fp = r["fp"] or ""
                    code = r["code"] or ""
                    if fp.endswith(".go"):
                        ast = parse_go_ast(fp, code)
                    elif fp.endswith(".py"):
                        ast = parse_python_ast(fp, code)
                    else:
                        continue
                    qcalls_set = set()
                    for c in ast.get("calls", []):
                        if len(c) > 2 and c[2]:
                            qcalls_set.add(c[2])
                        if len(c) > 1 and c[1]:
                            qcalls_set.add(c[1])
                    qcalls = sorted(list(qcalls_set))
                    if qcalls:
                        batch_updates.append({"id": r["id"], "qcalls": qcalls})
                if batch_updates:
                    session.run("""
                        UNWIND $batch AS item
                        MATCH (f:Function {id: item.id})
                        SET f.qualified_calls = item.qcalls
                    """, batch=batch_updates)
                logger.info("✅ Re-populated qualified_calls on %d Function nodes.", len(batch_updates))
            else:
                logger.info("[DRY RUN] Would scan and re-populate AST qualified_calls.")

            if args.dry_run:
                logger.info("[DRY RUN] Would clean up ambiguous CALLS edges where local callee exists or generic name across files.")
            else:
                logger.info("1. Cleaning up ambiguous CALLS edges where a file-local callee exists...")
                res = session.run(CYPHER_CLEANUP_AMBIGUOUS_CALLS)
                record = res.single()
                deleted = record["deleted_edges"] if record else 0
                logger.info("✅ Deleted %d ambiguous intra-repo CALLS edges.", deleted)

                logger.info("1b. Cleaning up false bare-name intra-repo CALLS edges for generic methods...")
                res_bare = session.run(CYPHER_CLEANUP_FALSE_BARE_CALLS)
                rec_bare = res_bare.single()
                deleted_bare = rec_bare["deleted_edges"] if rec_bare else 0
                logger.info("✅ Deleted %d false bare-name intra-repo CALLS edges.", deleted_bare)

                logger.info("1c. Cleaning up false cross-repo CALLS edges without USES_REPO relationship...")
                res_xrepo = session.run(CYPHER_CLEANUP_FALSE_XREPO_CALLS)
                rec_xrepo = res_xrepo.single()
                deleted_xrepo = rec_xrepo["deleted_edges"] if rec_xrepo else 0
                logger.info("✅ Deleted %d false cross-repo CALLS edges.", deleted_xrepo)

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
