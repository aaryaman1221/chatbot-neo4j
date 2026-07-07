#!/usr/bin/env python3
"""
Generalized Semantic Graph Enrichment Engine (Optimized Production Grade)

Runs automated passes over the Neo4j codebase graph to generate high-level
semantic relationship layers while enforcing multi-repo dependency boundaries.
"""

import os
import sys
import re
import json
import argparse
import logging
from pathlib import Path
from dotenv import load_dotenv
from neo4j import GraphDatabase

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("enrich_semantic_edges")


def pass_0_backfill_declares_method(session, dry_run: bool) -> int:
    """Pass 0: Wire DECLARES_METHOD from existing :Type nodes to their methods."""
    logger.info("── Pass 0: Backfill DECLARES_METHOD (Type → Function) ──")
    if dry_run:
        logger.info("[DRY-RUN] Would backfill DECLARES_METHOD edges.")
        return 0

    # Strategy A: Explicit scoped naming configurations ("ReceiverType.MethodName")
    scoped_query = """
    MATCH (f:Function) WHERE f.name CONTAINS '.'
    WITH f, split(f.name, '.')[0] AS receiver_name
    MATCH (t:Type {name: receiver_name, repo: f.repo})
    WHERE NOT (t)-[:DECLARES_METHOD]->(f)
    MERGE (t)-[:DECLARES_METHOD]->(f)
    RETURN count(*) AS count
    """
    count_a = session.run(scoped_query).single()["count"] or 0

    # Strategy B: Go structural headers handling implicit pointer/value receivers or omitted labels
    header_query = """
    MATCH (f:Function)
    WHERE f.code IS NOT NULL AND f.code STARTS WITH 'func (' AND NOT f.name CONTAINS '.'
    WITH f, substring(f.code, 0, 200) AS header
    WHERE header =~ '(?s)^func\\s*\\(\\s*(\\*?[A-Za-z0-9_]+|\\w+\\s+\\*?[A-Za-z0-9_]+)\\s*\\).*'
    WITH f, split(split(header, ')')[0], '(')[1] AS raw_receiver
    WITH f, replace(replace(raw_receiver, '*', ''), ' ', '') AS cleaned_receiver
    MATCH (t:Type {name: cleaned_receiver, repo: f.repo})
    WHERE NOT (t)-[:DECLARES_METHOD]->(f)
    MERGE (t)-[:DECLARES_METHOD]->(f)
    RETURN count(*) AS count
    """
    count_b = session.run(header_query).single()["count"] or 0

    total = count_a + count_b
    logger.info("✅ Backfilled %d DECLARES_METHOD edges.", total)
    return total


def pass_1_receiver_types(session, dry_run: bool) -> int:
    """Pass 1: Parse method receivers from raw code blocks safely."""
    logger.info("── Pass 1: Receiver Type Extraction ──")
    query = """
    MATCH (f:Function)
    WHERE f.name CONTAINS '.' OR (f.code IS NOT NULL AND f.code STARTS WITH 'func (')
    RETURN elementId(f) AS func_eid, f.name AS func_name, f.filepath AS filepath, f.repo AS repo, substring(f.code, 0, 150) AS header
    """
    rows = session.run(query).data()

    # Supports: (r *Repo), (*Repo), (Repo)
    receiver_regex = re.compile(r"^func\s*\(\s*(?:\w+\s+)?\*?([A-Za-z0-9_]+)\s*\)")
    batch = []
    for r in rows:
        func_name = r.get("func_name") or ""
        type_name = func_name.split(".")[0] if "." in func_name else ""
        
        if not type_name:
            m = receiver_regex.match(r.get("header") or "")
            if m:
                type_name = m.group(1)
        
        if type_name:
            batch.append({
                "func_eid": r["func_eid"],
                "type_name": type_name,
                "filepath": r.get("filepath") or "",
                "repo": r.get("repo") or "",
            })

    if not batch or dry_run:
        return len(batch)

    update_query = """
    UNWIND $batch AS item
    MATCH (f:Function) WHERE elementId(f) = item.func_eid
    MERGE (t:Type {name: item.type_name, repo: item.repo})
    ON CREATE SET t.filepath = item.filepath
    MERGE (t)-[r:DECLARES_METHOD]->(f)
    RETURN count(r) AS count
    """
    return session.run(update_query, batch=batch).single()["count"] or 0


def pass_2_struct_embedding(session, dry_run: bool) -> int:
    """Pass 2: Trace structural composition inheritance (EMBEDS)."""
    logger.info("── Pass 2: Composition & Composition Analysis (EMBEDS) ──")
    query = "MATCH (t:Type) RETURN t.repo AS repo, t.filepath AS filepath, t.name AS name, t.code AS code, t.fields AS fields"
    rows = session.run(query).data()
    
    struct_regex = re.compile(r"type\s+([A-Za-z0-9_]+)\s+struct\s*\{([^}]+)\}", re.MULTILINE)
    py_class_regex = re.compile(r"class\s+([A-Za-z0-9_]+)\s*\(([A-Za-z0-9_,\s.]+)\):", re.MULTILINE)
    batch = []

    for r in rows:
        name, repo, code = r["name"], r["repo"], r["code"] or ""
        
        for f_str in (r["fields"] or []):
            try:
                meta = json.loads(f_str) if isinstance(f_str, str) else f_str
                if isinstance(meta, dict) and meta.get("is_embedded"):
                    inner = meta.get("type", "").lstrip("*").split(".")[-1]
                    if inner and inner != name:
                        batch.append({"outer_type": name, "inner_name": inner, "repo": repo})
            except Exception:
                pass
                
        for sm in struct_regex.finditer(code):
            outer, body = sm.group(1), sm.group(2)
            for line in body.splitlines():
                parts = line.strip().split()
                if len(parts) == 1 and not parts[0].startswith("//"):
                    inner = parts[0].lstrip("*").split(".")[-1]
                    if inner and inner != outer:
                        batch.append({"outer_type": outer, "inner_name": inner, "repo": repo})

        for cm in py_class_regex.finditer(code):
            outer, parents = cm.group(1), cm.group(2)
            for p in parents.split(","):
                inner = p.strip().split(".")[-1]
                if inner and inner not in ["object", "ABC", "BaseModel"] and inner != outer:
                    batch.append({"outer_type": outer, "inner_name": inner, "repo": repo})

    if not batch or dry_run:
        return len(batch)

    update_query = """
    UNWIND $batch AS item
    MATCH (outer:Type {name: item.outer_type, repo: item.repo})
    MATCH (inner:Type) WHERE inner.repo = item.repo AND inner.name = item.inner_name AND outer <> inner
    MERGE (outer)-[r:EMBEDS]->(inner)
    RETURN count(r) AS count
    """
    return session.run(update_query, batch=batch).single()["count"] or 0


def pass_3_duck_typing_implements(session, dry_run: bool) -> int:
    """Pass 3: Connect Type structures to structural Interfaces (IMPLEMENTS)."""
    logger.info("── Pass 3: Duck-Typing Contracts (IMPLEMENTS) ──")
    
    t_query = """
    MATCH (t:Type)
    OPTIONAL MATCH (t)-[:DECLARES_METHOD]->(f:Function)
    WITH t, collect(DISTINCT f.name) AS edge_methods
    RETURN elementId(t) AS eid, t.name AS name, t.kind AS kind, t.repo AS repo, t.method_names AS scalar_methods, edge_methods
    """
    t_rows = session.run(t_query).data()
    type_methods_map = {}

    for r in t_rows:
        m_set = set(r.get("scalar_methods") or [])
        for m in r.get("edge_methods") or []:
            m_set.add(m.split(".")[-1])
            
        if m_set:
            type_methods_map[r["eid"]] = {
                "name": r["name"], "kind": (r.get("kind") or "").upper(), "repo": r["repo"], "methods": m_set
            }

    batch = []
    eids = list(type_methods_map.keys())
    for i in range(len(eids)):
        for j in range(i + 1, len(eids)):
            t1, t2 = type_methods_map[eids[i]], type_methods_map[eids[j]]
            
            # Skip duplicate nodes or exact matches
            if t1["name"] == t2["name"] and t1["repo"] == t2["repo"]:
                continue
                
            # Allow same-repo combinations OR valid cross-repo structural matches
            # but reject pairs where neither node is explicitly marked as an INTERFACE
            if not ((t1["kind"] == "INTERFACE") or (t2["kind"] == "INTERFACE")):
                continue

            if t2["kind"] == "INTERFACE" and t2["methods"].issubset(t1["methods"]):
                batch.append({"t1_eid": eids[i], "t2_eid": eids[j]})
            elif t1["kind"] == "INTERFACE" and t1["methods"].issubset(t2["methods"]):
                batch.append({"t1_eid": eids[j], "t2_eid": eids[i]})

    if not batch or dry_run:
        return len(batch)

    update_query = """
    UNWIND $batch AS item
    MATCH (t1:Type), (t2:Type) WHERE elementId(t1) = item.t1_eid AND elementId(t2) = item.t2_eid
    MERGE (t1)-[r:IMPLEMENTS]->(t2)
    RETURN count(r) AS count
    """
    return session.run(update_query, batch=batch).single()["count"] or 0


def pass_4_wrappers_and_factories(session, dry_run: bool) -> int:
    """Pass 4: Discover object wrappers and lifecycle initializers."""
    logger.info("── Pass 4: Wrapper Construction & Factories ──")
    if dry_run: return 0

    # Guarded cross-repo query checking explicit USES_REPO boundaries to eliminate noise matches
    wraps_query = """
    MATCH (caller:Function)-[:CALLS]->(callee:Function)
    WHERE caller.repo <> callee.repo
      AND (toLower(callee.name) STARTS WITH 'new' OR toLower(callee.name) CONTAINS 'adapter' OR toLower(callee.name) CONTAINS 'wrapper')
      AND EXISTS { MATCH (f:File {repo: caller.repo})-[:USES_REPO]->(:Repository {full_name: callee.repo}) }
    MERGE (caller)-[r:WRAPS]->(callee)
    RETURN count(r) AS count
    """
    wraps_count = session.run(wraps_query).single()["count"] or 0

    produces_query = """
    MATCH (t:Type)-[:DECLARES_METHOD]->(f:Function)
    WHERE toLower(f.name) STARTS WITH 'new' OR toLower(f.name) STARTS WITH 'create'
    MERGE (f)-[r:PRODUCES]->(t)
    RETURN count(r) AS count
    """
    produces_count = session.run(produces_query).single()["count"] or 0
    return wraps_count + produces_count


def pass_5_lifecycle_hooks_and_registry(session, dry_run: bool) -> int:
    """Pass 5: Identify lifecycle management and system callbacks."""
    logger.info("── Pass 5: System Hooks & Registries ──")
    if dry_run: return 0

    hook_query = """
    MATCH (f:Function)
    WHERE toLower(f.name) CONTAINS 'hook' OR toLower(f.name) CONTAINS 'handler' OR toLower(f.name) STARTS WITH 'on_'
    OPTIONAL MATCH (t:Type)-[:DECLARES_METHOD]->(f)
    WITH f, coalesce(t, f) AS host
    WHERE host <> f
    MERGE (host)-[r:LIFECYCLE_HOOK]->(f)
    RETURN count(r) AS count
    """
    hooks = session.run(hook_query).single()["count"] or 0

    reg_query = """
    MATCH (caller:Function)-[:CALLS]->(reg:Function)
    WHERE toLower(reg.name) STARTS WITH 'register' OR toLower(reg.name) STARTS WITH 'add'
    MERGE (caller)-[r:REGISTERS_WITH]->(reg)
    RETURN count(r) AS count
    """
    registries = session.run(reg_query).single()["count"] or 0
    return hooks + registries


def pass_6_delegation_and_forwarding(session, dry_run: bool) -> int:
    """Pass 6: Track proxy layers and forwarding chains."""
    logger.info("── Pass 6: Forwarding & Proxies ──")
    if dry_run: return 0

    dispatch_query = """
    MATCH (caller:Function)-[:CALLS]->(callee:Function)
    WHERE toLower(caller.name) CONTAINS 'dispatch' AND caller <> callee
    MERGE (caller)-[r:DISPATCHES_TO]->(callee)
    RETURN count(r) AS count
    """
    dispatches = session.run(dispatch_query).single()["count"] or 0

    forward_query = """
    MATCH (caller:Function)-[:CALLS]->(callee:Function)
    WHERE caller.name = callee.name AND caller.repo <> callee.repo
      AND EXISTS { MATCH (f:File {repo: caller.repo})-[:USES_REPO]->(:Repository {full_name: callee.repo}) }
    MERGE (caller)-[r:FORWARDS_TO]->(callee)
    RETURN count(r) AS count
    """
    forwards = session.run(forward_query).single()["count"] or 0
    return dispatches + forwards


def pass_7_mutates_state_of(session, dry_run: bool) -> int:
    """Pass 7: Verify pointer mutation blocks and variable updates."""
    logger.info("── Pass 7: Mutative Side-Effect Analysis ──")
    if dry_run: return 0

    mutates_query = """
    MATCH (t:Type)-[:DECLARES_METHOD]->(f:Function)
    WHERE f.is_pointer_receiver = true 
       OR f.code CONTAINS 'self.' 
       OR toLower(f.name) STARTS WITH 'set'
       OR toLower(f.name) STARTS WITH 'update'
    MERGE (f)-[r:MUTATES_STATE_OF]->(t)
    RETURN count(r) AS count
    """
    return session.run(mutates_query).single()["count"] or 0


def pass_8_return_types(session, dry_run: bool) -> int:
    """Pass 8: Link functions to their return types."""
    logger.info("── Pass 8: Return Types (Function → Type) ──")
    if dry_run: return 0

    query = """
    MATCH (f:Function) WHERE f.return_types IS NOT NULL AND size(f.return_types) > 0
    UNWIND f.return_types AS ret_type
    MATCH (t:Type {name: ret_type, repo: f.repo})
    MERGE (f)-[r:RETURNS]->(t)
    RETURN count(r) AS count
    """
    return session.run(query).single()["count"] or 0


def pass_9_test_coverage(session, dry_run: bool) -> int:
    """Pass 9: Connect test functions to target functions they verify."""
    logger.info("── Pass 9: Test Coverage (TestXxx → target functions) ──")
    if dry_run: return 0

    query_direct = """
    MATCH (test:Function {is_test: true})
    WHERE test.name STARTS WITH 'Test'
    WITH test, substring(test.name, 4) AS target_name
    MATCH (target:Function {name: target_name, repo: test.repo, is_test: false})
    MERGE (test)-[r:TESTS]->(target)
    RETURN count(r) AS count
    """
    count_direct = session.run(query_direct).single()["count"] or 0

    query_method = """
    MATCH (test:Function {is_test: true})
    WHERE test.name STARTS WITH 'Test' AND test.name CONTAINS '_'
    WITH test, substring(test.name, 4) AS raw_name
    WITH test, split(raw_name, '_')[0] AS struct_name, split(raw_name, '_')[1] AS method_name
    MATCH (target:Function {name: struct_name + '.' + method_name, repo: test.repo, is_test: false})
    MERGE (test)-[r:TESTS]->(target)
    RETURN count(r) AS count
    """
    count_method = session.run(query_method).single()["count"] or 0

    return count_direct + count_method


def pass_10_error_paths(session, dry_run: bool) -> int:
    """Pass 10: Track proxy layers and error propagation chains."""
    logger.info("── Pass 10: Error Path Tracing ──")
    if dry_run: return 0

    query = """
    MATCH (caller:Function) WHERE caller.propagated_errors IS NOT NULL AND size(caller.propagated_errors) > 0
    UNWIND caller.propagated_errors AS callee_name
    MATCH (callee:Function {name: callee_name, repo: caller.repo})
    WHERE caller <> callee
    MERGE (caller)-[r:PROPAGATES_ERROR]->(callee)
    RETURN count(r) AS count
    """
    return session.run(query).single()["count"] or 0


def pass_11_generics_constraints(session, dry_run: bool) -> int:
    """Pass 11: Link generic type parameters to their constraints."""
    logger.info("── Pass 11: Generic Type Constraints ──")
    if dry_run: return 0

    builtin_constraints = {"any", "comparable", "error", "string", "int", "int8", "int16", "int32", "int64",
                           "uint", "uint8", "uint16", "uint32", "uint64", "float32", "float64", "bool", "byte", "rune"}

    type_query = """
    MATCH (t:Type) WHERE t.type_parameters IS NOT NULL AND t.type_parameters <> '[]'
    RETURN elementId(t) AS eid, t.repo AS repo, t.type_parameters AS tp
    """
    rows_types = session.run(type_query).data()

    func_query = """
    MATCH (f:Function) WHERE f.type_parameters IS NOT NULL AND f.type_parameters <> '[]'
    RETURN elementId(f) AS eid, f.repo AS repo, f.type_parameters AS tp
    """
    rows_funcs = session.run(func_query).data()

    batch = []
    for r in rows_types + rows_funcs:
        tp_str = r.get("tp") or "[]"
        try:
            tp_list = json.loads(tp_str) if isinstance(tp_str, str) else tp_str
            if isinstance(tp_list, list):
                for tp in tp_list:
                    constraint = tp.get("constraint") or ""
                    clean_constraint = constraint.lstrip("*").split(".")[-1].strip()
                    if clean_constraint and clean_constraint not in builtin_constraints:
                        batch.append({
                            "source_eid": r["eid"],
                            "constraint_name": clean_constraint,
                            "repo": r["repo"]
                        })
        except Exception:
            pass

    if not batch:
        return 0

    update_query = """
    UNWIND $batch AS item
    MATCH (src) WHERE elementId(src) = item.source_eid
    MATCH (constraint:Type {name: item.constraint_name, repo: item.repo})
    WHERE src <> constraint
    MERGE (src)-[r:CONSTRAINED_BY]->(constraint)
    RETURN count(r) AS count
    """
    return session.run(update_query, batch=batch).single()["count"] or 0


def pass_12_channel_elements(session, dry_run: bool) -> int:
    """Pass 12: Link typed channels to their element types."""
    logger.info("── Pass 12: Channel Element Types (Variable → Type) ──")
    if dry_run: return 0

    query = """
    MATCH (v:Variable {kind: "CHAN"}) 
    WHERE v.chan_elem_type IS NOT NULL AND v.chan_elem_type <> ""
    MATCH (t:Type {name: v.chan_elem_type, repo: v.repo})
    MERGE (v)-[r:CHAN_ELEM_TYPE]->(t)
    RETURN count(r) AS count
    """
    return session.run(query).single()["count"] or 0


def pass_13_go_generate(session, dry_run: bool) -> int:
    """Pass 13: Resolve go:generate directives to output types or files."""
    logger.info("── Pass 13: go:generate Resolution ──")
    if dry_run: return 0

    query = """
    MATCH (d:Directive {directive: "go:generate"})
    RETURN elementId(d) AS eid, d.filepath AS filepath, d.repo AS repo, d.args AS args
    """
    rows = session.run(query).data()
    batch_files = []
    batch_types = []

    for r in rows:
        args = r.get("args") or ""
        filepath = r.get("filepath") or ""
        repo = r.get("repo") or ""

        m_type = re.search(r'-type[ =]([A-Za-z0-9_]+)', args)
        if m_type:
            type_name = m_type.group(1)
            batch_types.append({
                "dir_eid": r["eid"],
                "type_name": type_name,
                "repo": repo
            })
            gen_file = f"{type_name.lower()}_string.go"
            parent_dir = filepath.rsplit("/", 1)[0] if "/" in filepath else ""
            target_path = f"{parent_dir}/{gen_file}" if parent_dir else gen_file
            batch_files.append({
                "dir_eid": r["eid"],
                "target_path": target_path,
                "repo": repo
            })

        m_dest = re.search(r'-destination[ =]([A-Za-z0-9_./-]+)', args)
        if m_dest:
            dest_val = m_dest.group(1)
            parent_dir = filepath.rsplit("/", 1)[0] if "/" in filepath else ""
            if parent_dir and not dest_val.startswith("/"):
                target_path = os.path.normpath(f"{parent_dir}/{dest_val}").replace("\\", "/")
            else:
                target_path = os.path.normpath(dest_val).replace("\\", "/")
            batch_files.append({
                "dir_eid": r["eid"],
                "target_path": target_path,
                "repo": repo
            })

    count = 0
    if batch_types:
        type_query = """
        UNWIND $batch AS item
        MATCH (d:Directive) WHERE elementId(d) = item.dir_eid
        MATCH (t:Type {name: item.type_name, repo: item.repo})
        MERGE (d)-[r:GENERATES_TYPE]->(t)
        RETURN count(r) AS count
        """
        count += session.run(type_query, batch=batch_types).single()["count"] or 0

    if batch_files:
        file_query = """
        UNWIND $batch AS item
        MATCH (d:Directive) WHERE elementId(d) = item.dir_eid
        MATCH (f:File {path: item.target_path, repo: item.repo})
        MERGE (d)-[r:GENERATES_FILE]->(f)
        RETURN count(r) AS count
        """
        count += session.run(file_query, batch=batch_files).single()["count"] or 0

    return count


def main():
    parser = argparse.ArgumentParser(description="Generate semantic abstractions over code architecture graphs.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    _env_path = Path(__file__).resolve().parent / ".env"
    load_dotenv(_env_path if _env_path.exists() else None)

    uri = os.environ.get("NEO4J_URI", "neo4j://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "password123")

    with GraphDatabase.driver(uri, auth=(user, password)) as driver:
        with driver.session() as session:
            t0 = pass_0_backfill_declares_method(session, args.dry_run)
            t1 = pass_1_receiver_types(session, args.dry_run)
            t2 = pass_2_struct_embedding(session, args.dry_run)
            t3 = pass_3_duck_typing_implements(session, args.dry_run)
            t4 = pass_4_wrappers_and_factories(session, args.dry_run)
            t5 = pass_5_lifecycle_hooks_and_registry(session, args.dry_run)
            t6 = pass_6_delegation_and_forwarding(session, args.dry_run)
            t7 = pass_7_mutates_state_of(session, args.dry_run)
            t8 = pass_8_return_types(session, args.dry_run)
            t9 = pass_9_test_coverage(session, args.dry_run)
            t10 = pass_10_error_paths(session, args.dry_run)
            t11 = pass_11_generics_constraints(session, args.dry_run)
            t12 = pass_12_channel_elements(session, args.dry_run)
            t13 = pass_13_go_generate(session, args.dry_run)

            logger.info("🎯 Total Relational Edges Managed: %d", (t0+t1+t2+t3+t4+t5+t6+t7+t8+t9+t10+t11+t12+t13))


if __name__ == "__main__":
    main()