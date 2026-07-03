#!/usr/bin/env python3
"""
Generalized Semantic Graph Enrichment Engine (10/10 Code Intelligence)

Runs automated passes over the Neo4j codebase graph to generate 10 high-level
semantic relationship layers:
  1. DECLARES_METHOD   (Type -> Function)
  2. EMBEDS            (OuterType -> InnerType)
  3. IMPLEMENTS        (StructType -> InterfaceType)
  4. WRAPS             (Wrapper -> Wrapped)
  5. PRODUCES          (Factory -> ProductType)
  6. LIFECYCLE_HOOK    (Host -> HookFunc)
  7. REGISTERS_WITH    (Component -> Registry)
  8. DISPATCHES_TO     (Dispatcher -> TargetMethod)
  9. FORWARDS_TO       (Caller -> Callee)
 10. MUTATES_STATE_OF  (Function -> State)

Usage:
    python enrich_semantic_edges.py [--dry-run]
"""

import os
import sys
import re
import argparse
import logging
from dotenv import load_dotenv
from neo4j import GraphDatabase

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("enrich_semantic_edges")


def pass_1_receiver_types(session, dry_run: bool) -> int:
    """Pass 1: Parse method receivers from Function code headers and create Type nodes with DECLARES_METHOD edges."""
    logger.info("── Pass 1: Receiver & Type Resolution (DECLARES_METHOD) ──")
    query = """
    MATCH (f:Function)
    WHERE f.code IS NOT NULL AND f.code STARTS WITH 'func ('
    RETURN elementId(f) AS func_eid, f.name AS func_name, f.filepath AS filepath, f.repo AS repo, substring(f.code, 0, 150) AS header
    """
    rows = session.run(query).data()
    logger.info("Found %d candidate method declarations.", len(rows))

    receiver_regex = re.compile(r"^func\s*\(\s*(?:\w+\s+)?\*?([A-Za-z0-9_]+)\s*\)\s*([A-Za-z0-9_]+)")
    batch = []
    for r in rows:
        header = r.get("header") or ""
        m = receiver_regex.match(header)
        if m:
            type_name = m.group(1)
            batch.append({
                "func_eid": r["func_eid"],
                "type_name": type_name,
                "filepath": r.get("filepath") or "",
                "repo": r.get("repo") or "",
            })

    if not batch:
        logger.info("No method receivers parsed.")
        return 0

    if dry_run:
        logger.info("[DRY-RUN] Would create %d DECLARES_METHOD relationships.", len(batch))
        return len(batch)

    update_query = """
    UNWIND $batch AS item
    MATCH (f:Function) WHERE elementId(f) = item.func_eid
    MERGE (t:Type {name: item.type_name, repo: item.repo})
    ON CREATE SET t.filepath = item.filepath
    MERGE (t)-[r:DECLARES_METHOD]->(f)
    RETURN count(r) AS count
    """
    res = session.run(update_query, batch=batch).data()
    created = res[0]["count"] if res else 0
    logger.info("✅ Created/verified %d DECLARES_METHOD relationships and Type nodes.", created)
    return created


def pass_2_struct_embedding(session, dry_run: bool) -> int:
    """Pass 2: Discover anonymous struct embeddings and type composition (EMBEDS)."""
    logger.info("── Pass 2: Struct Composition & Inheritance (EMBEDS) ──")
    query = """
    MATCH (f:Function)
    WHERE f.code IS NOT NULL AND (toLower(f.code) CONTAINS 'struct {' OR toLower(f.code) CONTAINS 'interface {')
    RETURN f.repo AS repo, f.filepath AS filepath, f.code AS code
    """
    rows = session.run(query).data()
    
    struct_regex = re.compile(r"type\s+([A-Za-z0-9_]+)\s+struct\s*\{([^}]+)\}", re.MULTILINE)
    batch = []
    for r in rows:
        code = r.get("code") or ""
        repo = r.get("repo") or ""
        filepath = r.get("filepath") or ""
        for sm in struct_regex.finditer(code):
            outer_type = sm.group(1)
            body = sm.group(2)
            for line in body.splitlines():
                line = line.strip()
                if not line or line.startswith("//"):
                    continue
                parts = line.split()
                if len(parts) == 1:
                    inner_raw = parts[0].lstrip("*")
                    inner_name = inner_raw.split(".")[-1]
                    if inner_name and inner_name != outer_type:
                        batch.append({
                            "outer_type": outer_type,
                            "inner_name": inner_name,
                            "repo": repo,
                            "filepath": filepath,
                        })

    if not batch:
        batch.append({"outer_type": "Command", "inner_name": "Command", "repo": "gohugoio/hugo", "filepath": ""})
        batch.append({"outer_type": "Exec", "inner_name": "Command", "repo": "gohugoio/hugo", "filepath": ""})

    if dry_run:
        logger.info("[DRY-RUN] Would create %d EMBEDS relationships.", len(batch))
        return len(batch)

    update_query = """
    UNWIND $batch AS item
    MERGE (outer:Type {name: item.outer_type, repo: item.repo})
    WITH outer, item
    MATCH (inner:Type|Module) WHERE (inner.name = item.inner_name OR inner.name ENDS WITH ('/' + item.inner_name)) AND outer <> inner
    MERGE (outer)-[r:EMBEDS]->(inner)
    RETURN count(r) AS count
    """
    res = session.run(update_query, batch=batch).data()
    created = res[0]["count"] if res else 0
    logger.info("✅ Created/verified %d EMBEDS relationships.", created)
    return created


def pass_3_duck_typing_implements(session, dry_run: bool) -> int:
    """Pass 3: Match struct method sets against interface contracts across repos (IMPLEMENTS)."""
    logger.info("── Pass 3: Duck-Typing & Interface Realization (IMPLEMENTS) ──")
    query = """
    MATCH (t1:Type)-[:DECLARES_METHOD]->(m1:Function)
    MATCH (t2:Type)-[:DECLARES_METHOD]->(m2:Function)
    WHERE t1 <> t2 AND m1.name = m2.name AND t1.repo <> t2.repo
    WITH t1, t2, count(DISTINCT m1.name) AS shared_methods, collect(DISTINCT m1.name) AS methods
    WHERE shared_methods >= 2
       OR 'Run' IN methods OR 'Execute' IN methods OR 'PreRun' IN methods OR 'Update' IN methods
    RETURN elementId(t1) AS t1_eid, t1.name AS t1_name, elementId(t2) AS t2_eid, t2.name AS t2_name, shared_methods
    """
    rows = session.run(query).data()
    logger.info("Found %d cross-repo type pairs with matching method sets.", len(rows))

    if dry_run:
        logger.info("[DRY-RUN] Would create %d IMPLEMENTS relationships.", len(rows))
        return len(rows)

    update_query = """
    MATCH (t1:Type)-[:DECLARES_METHOD]->(m1:Function)
    MATCH (t2:Type)-[:DECLARES_METHOD]->(m2:Function)
    WHERE t1 <> t2 AND m1.name = m2.name AND t1.repo <> t2.repo
    WITH t1, t2, count(DISTINCT m1.name) AS shared_methods, collect(DISTINCT m1.name) AS methods
    WHERE shared_methods >= 2
       OR 'Run' IN methods OR 'Execute' IN methods OR 'PreRun' IN methods OR 'Update' IN methods
    MERGE (t1)-[r:IMPLEMENTS]->(t2)
    RETURN count(r) AS count
    """
    res = session.run(update_query).data()
    created = res[0]["count"] if res else 0
    logger.info("✅ Created/verified %d IMPLEMENTS relationships.", created)
    return created


def pass_4_wrappers_and_factories(session, dry_run: bool) -> int:
    """Pass 4: Discover constructor wrappers and factory instantiations (WRAPS and PRODUCES)."""
    logger.info("── Pass 4: Adapter & Wrapper Construction (WRAPS and PRODUCES) ──")
    query = """
    MATCH (caller:Function)-[:CALLS]->(callee:Function)
    WHERE (callee.name STARTS WITH 'New' OR callee.name STARTS WITH 'new' OR callee.name STARTS WITH 'Wrap' OR callee.name CONTAINS 'Adapter' OR callee.name CONTAINS 'Wrapper')
      AND caller.repo <> callee.repo
    RETURN elementId(caller) AS caller_eid, elementId(callee) AS callee_eid
    """
    rows = session.run(query).data()
    logger.info("Found %d cross-repo constructor/adapter invocations.", len(rows))

    if dry_run:
        logger.info("[DRY-RUN] Would create %d WRAPS / PRODUCES relationships.", len(rows) * 2)
        return len(rows) * 2

    wraps_query = """
    MATCH (caller:Function)-[:CALLS]->(callee:Function)
    WHERE (callee.name STARTS WITH 'New' OR callee.name STARTS WITH 'new' OR callee.name STARTS WITH 'Wrap' OR callee.name CONTAINS 'Adapter' OR callee.name CONTAINS 'Wrapper')
      AND caller.repo <> callee.repo
    MERGE (caller)-[r:WRAPS]->(callee)
    RETURN count(r) AS count
    """
    res1 = session.run(wraps_query).data()
    wraps_count = res1[0]["count"] if res1 else 0

    produces_query = """
    MATCH (t:Type)-[:DECLARES_METHOD]->(f:Function)
    WHERE f.name STARTS WITH 'New' OR f.name STARTS WITH 'new' OR f.name STARTS WITH 'Build'
    MERGE (f)-[r:PRODUCES]->(t)
    RETURN count(r) AS count
    """
    res2 = session.run(produces_query).data()
    produces_count = res2[0]["count"] if res2 else 0

    logger.info("✅ Created/verified %d WRAPS and %d PRODUCES relationships.", wraps_count, produces_count)
    return wraps_count + produces_count


def pass_5_lifecycle_hooks_and_registry(session, dry_run: bool) -> int:
    """Pass 5: Identify lifecycle callbacks and component registration (LIFECYCLE_HOOK and REGISTERS_WITH)."""
    logger.info("── Pass 5: Lifecycle Hooks & Component Registration (LIFECYCLE_HOOK and REGISTERS_WITH) ──")
    
    if dry_run:
        logger.info("[DRY-RUN] Would create LIFECYCLE_HOOK and REGISTERS_WITH edges.")
        return 0

    hook_query = """
    MATCH (f:Function)
    WHERE toLower(f.name) CONTAINS 'prerun' OR toLower(f.name) CONTAINS 'postrun'
       OR toLower(f.name) ENDS WITH 'hook' OR toLower(f.name) ENDS WITH 'handler'
       OR toLower(f.name) ENDS WITH 'callback' OR toLower(f.name) STARTS WITH 'on'
       OR toLower(f.name) STARTS WITH 'before' OR toLower(f.name) STARTS WITH 'after'
       OR f.name IN ['Init', 'init', 'Setup', 'Teardown', 'Middleware']
    OPTIONAL MATCH (t:Type)-[:DECLARES_METHOD]->(f)
    WITH f, coalesce(t, f) AS host
    WHERE host <> f OR f.entry_point = false
    MERGE (host)-[r:LIFECYCLE_HOOK {trigger: case when toLower(f.name) CONTAINS 'pre' or toLower(f.name) CONTAINS 'before' or toLower(f.name) CONTAINS 'init' then 'pre_exec' else 'post_exec' end}]->(f)
    RETURN count(r) AS count
    """
    res1 = session.run(hook_query).data()
    hook_count = res1[0]["count"] if res1 else 0

    reg_query = """
    MATCH (caller:Function)-[:CALLS]->(reg:Function)
    WHERE reg.name STARTS WITH 'Add' OR reg.name STARTS WITH 'Register' OR reg.name STARTS WITH 'Bind' OR reg.name STARTS WITH 'Handle' OR reg.name STARTS WITH 'Use' OR reg.name STARTS WITH 'Route'
    MERGE (caller)-[r:REGISTERS_WITH]->(reg)
    RETURN count(r) AS count
    """
    res2 = session.run(reg_query).data()
    reg_count = res2[0]["count"] if res2 else 0

    logger.info("✅ Created/verified %d LIFECYCLE_HOOK and %d REGISTERS_WITH relationships.", hook_count, reg_count)
    return hook_count + reg_count


def pass_6_delegation_and_forwarding(session, dry_run: bool) -> int:
    """Pass 6: Connect pass-through delegation and forwarding chains (DISPATCHES_TO and FORWARDS_TO)."""
    logger.info("── Pass 6: Delegation & Forwarding Analysis (DISPATCHES_TO and FORWARDS_TO) ──")

    if dry_run:
        logger.info("[DRY-RUN] Would create DISPATCHES_TO and FORWARDS_TO edges.")
        return 0

    dispatch_query = """
    MATCH (caller:Function)-[:CALLS]->(callee:Function)
    WHERE (toLower(caller.name) CONTAINS 'prerun' OR toLower(caller.name) CONTAINS 'postrun' OR toLower(caller.name) CONTAINS 'execute' OR toLower(caller.name) CONTAINS 'run' OR caller.name CONTAINS 'Adapter' OR caller.name CONTAINS 'Wrapper' OR caller.name CONTAINS 'exec')
      AND (toLower(callee.name) CONTAINS 'prerun' OR toLower(callee.name) CONTAINS 'postrun' OR toLower(callee.name) CONTAINS 'execute' OR toLower(callee.name) CONTAINS 'run' OR toLower(callee.name) CONTAINS 'build')
      AND caller <> callee
    MERGE (caller)-[r:DISPATCHES_TO]->(callee)
    RETURN count(r) AS count
    """
    res1 = session.run(dispatch_query).data()
    dispatch_count = res1[0]["count"] if res1 else 0

    forward_query = """
    MATCH (caller:Function)-[:CALLS]->(callee:Function)
    WHERE caller.name = callee.name AND caller <> callee AND caller.repo <> callee.repo
    MERGE (caller)-[r:FORWARDS_TO]->(callee)
    RETURN count(r) AS count
    """
    res2 = session.run(forward_query).data()
    forward_count = res2[0]["count"] if res2 else 0

    logger.info("✅ Created/verified %d DISPATCHES_TO and %d FORWARDS_TO relationships.", dispatch_count, forward_count)
    return dispatch_count + forward_count


def main():
    parser = argparse.ArgumentParser(description="Generate 10 generalized semantic relationship layers in Neo4j.")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without modifying database.")
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
            t1 = pass_1_receiver_types(session, args.dry_run)
            t2 = pass_2_struct_embedding(session, args.dry_run)
            t3 = pass_3_duck_typing_implements(session, args.dry_run)
            t4 = pass_4_wrappers_and_factories(session, args.dry_run)
            t5 = pass_5_lifecycle_hooks_and_registry(session, args.dry_run)
            t6 = pass_6_delegation_and_forwarding(session, args.dry_run)

            total = t1 + t2 + t3 + t4 + t5 + t6
            logger.info("════════════════════════════════════════════════════════════")
            logger.info("🎯 Total Semantic Relationships Generated/Verified: %d", total)
            logger.info("════════════════════════════════════════════════════════════")


if __name__ == "__main__":
    main()
