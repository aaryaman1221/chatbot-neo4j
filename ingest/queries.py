# =============================================================================
# ingest/queries.py — Neo4j Cypher Query Templates & Indexes (Hardened)
# =============================================================================

CYPHER_CREATE_REPO = """
MERGE (repo:Repository {full_name: $repo_full_name})
  ON CREATE SET repo.name = $repo_name, repo.url = $repo_url
"""

CYPHER_INGEST_FUNCTION = """
MERGE (repo:Repository {full_name: $repo_full_name})
MERGE (file:File {path: $filepath, repo: $repo_full_name})
MERGE (func:Function {id: $func_id})
  ON CREATE SET
    func.name               = $func_name,
    func.filepath           = $filepath,
    func.repo               = $repo_full_name,
    func.code               = $func_code,
    func.embedding          = $embedding,
    func.qualified_calls    = $qualified_calls,
    func.is_exported        = coalesce($is_exported, false),
    func.is_pointer_receiver = coalesce($is_pointer_receiver, false),
    func.channels_sent      = $channels_sent,
    func.channels_received  = $channels_received,
    func.return_types       = $return_types,
    func.is_test            = coalesce($is_test, false),
    func.accepts_context    = coalesce($accepts_context, false),
    func.lock_sequence      = $lock_sequence,
    func.propagated_errors  = $propagated_errors
  ON MATCH SET
    func.name               = $func_name,
    func.filepath           = $filepath,
    func.repo               = $repo_full_name,
    func.code               = $func_code,
    func.embedding          = coalesce($embedding, func.embedding),
    func.qualified_calls    = $qualified_calls,
    func.is_exported        = coalesce($is_exported, false),
    func.is_pointer_receiver = coalesce($is_pointer_receiver, false),
    func.channels_sent      = $channels_sent,
    func.channels_received  = $channels_received,
    func.return_types       = $return_types,
    func.is_test            = coalesce($is_test, false),
    func.accepts_context    = coalesce($accepts_context, false),
    func.lock_sequence      = $lock_sequence,
    func.propagated_errors  = $propagated_errors
MERGE (file)-[:DECLARES]->(func)
MERGE (repo)-[:DECLARES]->(func)
"""

CYPHER_INGEST_CALLS = """
MATCH (caller:Function {id: $caller_id})

// 1. Try file-local match first (same exact component)
OPTIONAL MATCH (local:Function {name: $callee_name, filepath: $caller_filepath, repo: $repo_full_name})

// 2. Try package/directory-local match second (Go package-scoped evaluation)
OPTIONAL MATCH (pkg_local:Function {name: $callee_name, repo: $repo_full_name})
WHERE pkg_local.filepath <> $caller_filepath
  AND split(pkg_local.filepath, '/')[0..-1] = split($caller_filepath, '/')[0..-1]

// 3. Precise Qualified Match: Leverage the pipeline's qualified package string if available
OPTIONAL MATCH (qual_match:Function {repo: $repo_full_name})
WHERE $callee_qualified IS NOT NULL 
  AND (
    qual_match.name = $callee_qualified
    OR (qual_match.name = split($callee_qualified, '.')[-1] AND toLower(qual_match.filepath) CONTAINS toLower(split($callee_qualified, '.')[0]))
  )

// 4. Strategic Repository Fallback Heuristics
OPTIONAL MATCH (repo_wide:Function {name: $callee_name, repo: $repo_full_name})
WHERE repo_wide.filepath <> $caller_filepath
  AND split(repo_wide.filepath, '/')[0..-1] <> split($caller_filepath, '/')[0..-1]
  AND $callee_qualified IS NULL
  AND (
       $callee_name CONTAINS '.'
    OR EXISTS { 
         MATCH (caller_file:File {path: $caller_filepath, repo: $repo_full_name})-[:DEPENDS_ON]->(m:Module) 
         WHERE toLower(repo_wide.filepath) CONTAINS toLower(m.name) 
            OR toLower(m.name) ENDS WITH toLower(split(repo_wide.filepath, '/')[0]) 
       }
    OR (NOT $callee_name IN ['New', 'Run', 'Execute', 'Close', 'Open', 'Init', 'String', 'Read', 'Write', 'Update', 'Start', 'Stop', 'Reset'] AND size($callee_name) >= 5)
  )

WITH caller, coalesce(local, pkg_local, qual_match, repo_wide) AS callee
WHERE callee IS NOT NULL AND caller <> callee
MERGE (caller)-[r:CALLS]->(callee)
  ON CREATE SET r.call_type = coalesce($call_type, 'SYNC'),
                r.propagates_context = coalesce($propagates_context, false),
                r.creates_cancellation_scope = coalesce($creates_cancellation_scope, false)
  ON MATCH  SET r.call_type = coalesce($call_type, r.call_type, 'SYNC'),
                r.propagates_context = coalesce($propagates_context, r.propagates_context, false),
                r.creates_cancellation_scope = coalesce($creates_cancellation_scope, r.creates_cancellation_scope, false)
"""

CYPHER_INGEST_TYPE = """
MERGE (repo:Repository {full_name: $repo_full_name})
MERGE (file:File {path: $filepath, repo: $repo_full_name})
MERGE (type:Type {id: $type_id})
  ON CREATE SET
    type.name         = $type_name,
    type.kind         = $kind,
    type.filepath     = $filepath,
    type.repo         = $repo_full_name,
    type.code         = $code,
    type.fields       = $fields,
    type.field_names  = $field_names,
    type.methods      = $methods,
    type.method_names = $method_names,
    type.tags         = $tags,
    type.embedding    = $embedding,
    type.is_exported  = coalesce($is_exported, false),
    type.tag_mappings = $tag_mappings,
    type.type_parameters = $type_parameters
  ON MATCH SET
    type.name         = $type_name,
    type.kind         = $kind,
    type.filepath     = $filepath,
    type.repo         = $repo_full_name,
    type.code         = $code,
    type.fields       = $fields,
    type.field_names  = $field_names,
    type.methods      = $methods,
    type.method_names = $method_names,
    type.tags         = $tags,
    type.embedding    = coalesce($embedding, type.embedding),
    type.is_exported  = coalesce($is_exported, false),
    type.tag_mappings = $tag_mappings,
    type.type_parameters = $type_parameters
MERGE (file)-[:DECLARES_TYPE]->(type)
MERGE (repo)-[:DECLARES_TYPE]->(type)
"""

CYPHER_INGEST_TYPE_EMBEDDING = """
MATCH (outer:Type {id: $outer_id})
MATCH (inner:Type {repo: $repo_full_name})
WHERE inner.name = $inner_name OR inner.id ENDS WITH ("::" + $inner_name)
MERGE (outer)-[:EMBEDS]->(inner)
"""

CYPHER_INGEST_VARIABLE = """
MERGE (repo:Repository {full_name: $repo_full_name})
MERGE (file:File {path: $filepath, repo: $repo_full_name})
MERGE (var:Variable {id: $var_id})
  ON CREATE SET
    var.name        = $var_name,
    var.kind        = $kind,
    var.filepath    = $filepath,
    var.repo        = $repo_full_name,
    var.code        = $code,
    var.embedding   = $embedding,
    var.is_exported = coalesce($is_exported, false),
    var.chan_elem_type = $chan_elem_type
  ON MATCH SET
    var.name        = $var_name,
    var.kind        = $kind,
    var.filepath    = $filepath,
    var.repo         = $repo_full_name,
    var.code        = $code,
    var.embedding   = coalesce($embedding, var.embedding),
    var.is_exported = coalesce($is_exported, false),
    var.chan_elem_type = $chan_elem_type
MERGE (file)-[:DECLARES_VAR]->(var)
MERGE (repo)-[:DECLARES_VAR]->(var)
"""

CYPHER_INGEST_DIRECTIVE = """
MERGE (file:File {path: $filepath, repo: $repo_full_name})
MERGE (dir:Directive {filepath: $filepath, repo: $repo_full_name, directive: $directive, args: $args})
MERGE (file)-[:HAS_DIRECTIVE]->(dir)
"""

CYPHER_MODIFIED_FUNCTION = """
MERGE (commit:Commit {sha: $commit_sha})
MERGE (func:Function {id: $func_id})
  ON CREATE SET
    func.name         = $func_name,
    func.filepath     = $filepath,
    func.repo         = $repo_full_name
MERGE (commit)-[:MODIFIED]->(func)
"""

CYPHER_INGEST_COMMIT = """
MERGE (repo:Repository {full_name: $repo_full_name})
MERGE (author:User {login: $actor_login})
MERGE (commit:Commit {sha: $commit_sha})
SET
    commit.timestamp    = $committed_at,
    commit.summary_text = $summary_text,
    commit.diff_text    = $diff_text,
    commit.message      = $commit_message,
    commit.url          = $commit_url
MERGE (author)-[:AUTHORED]->(commit)
MERGE (commit)-[:BELONGS_TO]->(repo)
"""

CYPHER_INGEST_FILE = """
MERGE (repo:Repository {full_name: $repo_full_name})
MERGE (file:File {path: $filepath, repo: $repo_full_name})
  ON CREATE SET file.repo = $repo_full_name
MERGE (commit:Commit {sha: $commit_sha})
MERGE (commit)-[:MODIFIED]->(file)
MERGE (repo)-[:CONTAINS_FILE]->(file)
"""

CYPHER_INGEST_DEPENDENCY = """
MERGE (file:File {path: $filepath, repo: $repo_full_name})
MERGE (module:Module {name: $target_module})
  ON CREATE SET module.embedding = coalesce(module.embedding, $embedding)
MERGE (file)-[r:DEPENDS_ON]->(module)
  ON CREATE SET r.added_in_commit = $commit_sha, r.is_active = true
  ON MATCH  SET r.is_active = true
"""

CYPHER_REMOVE_DEPENDENCY = """
MATCH (file:File {path: $filepath, repo: $repo_full_name})-[r:DEPENDS_ON]->(module:Module {name: $target_module})
SET r.is_active = false, r.deleted_in_commit = $commit_sha
"""

CYPHER_TREE_FILE = """
MERGE (repo:Repository {full_name: $repo_full_name})
MERGE (file:File {path: $child_path, repo: $repo_full_name})
  ON CREATE SET file.repo        = $repo_full_name,
                file.entry_point = $entry_point
  ON MATCH  SET file.entry_point = coalesce(file.entry_point, $entry_point)
MERGE (repo)-[:CONTAINS_FILE]->(file)
WITH file, repo
OPTIONAL MATCH (parent:Directory {path: $parent_path, repo: $repo_full_name})
FOREACH (_ IN CASE WHEN parent IS NOT NULL THEN [1] ELSE [] END |
  MERGE (parent)-[:CONTAINS]->(file)
)
"""

CYPHER_TREE_DIR = """
MERGE (repo:Repository {full_name: $repo_full_name})
MERGE (dir:Directory {path: $child_path, repo: $repo_full_name})
  ON CREATE SET dir.repo    = $repo_full_name,
                dir.utility = $utility
MERGE (repo)-[:CONTAINS_DIR]->(dir)
MERGE (repo)-[:CONTAINS_DIR]->(dir)
WITH dir, repo
OPTIONAL MATCH (parent:Directory {path: $parent_path, repo: $repo_full_name})
FOREACH (_ IN CASE WHEN parent IS NOT NULL THEN [1] ELSE [] END |
  MERGE (parent)-[:CONTAINS]->(dir)
)
"""

CYPHER_LINK_REPO_DEPENDENCY = """
MATCH (f:File {repo: $parent_repo})-[:DEPENDS_ON]->(m:Module)
WHERE toLower(m.name) CONTAINS toLower($helper_repo_name)
MERGE (r:Repository {full_name: $helper_repo})
MERGE (f)-[:USES_REPO]->(r)
"""

CYPHER_LINK_MODULE_TO_REPO = """
MATCH (m:Module)
MATCH (r:Repository)
WHERE toLower(m.name) ENDS WITH "/" + toLower(r.name)
   OR toLower(m.name) = toLower(r.name)
   OR toLower(m.name) = toLower(r.full_name)
   OR toLower(m.name) ENDS WITH "/" + toLower(r.full_name)
MERGE (m)-[:REPRESENTS]->(r)
"""

CYPHER_LINK_XREPO_CALLS = """
MATCH (parentFile:File {repo: $parent_repo})-[:USES_REPO]->(helper_repo:Repository {full_name: $helper_repo})
WITH DISTINCT helper_repo
MATCH (caller:Function {repo: $parent_repo})
WHERE caller.qualified_calls IS NOT NULL AND size(caller.qualified_calls) > 0
UNWIND caller.qualified_calls AS qcall
WITH caller, qcall, split(qcall, '.') AS parts, helper_repo
WHERE size(parts) >= 2
WITH caller, qcall, parts[0] AS prefix, parts[-1] AS func_name, helper_repo
OPTIONAL MATCH (m:Module)-[:REPRESENTS]->(helper_repo)
WHERE toLower(m.name) ENDS WITH "/" + toLower(prefix)
   OR toLower(m.name) = toLower(prefix)
   OR toLower(m.name) ENDS WITH "/" + toLower($helper_prefix)
WITH caller, qcall, prefix, func_name, helper_repo, m
WHERE (toLower(prefix) = toLower($helper_prefix) OR m IS NOT NULL)
MATCH (callee:Function {repo: $helper_repo})
WHERE (callee.name = func_name)
   OR (func_name IN ['New', 'NewCommand', 'Init', 'Execute', 'Run']
       AND callee.name IN ['Command', 'NewCommand', 'Execute', 'RunE', 'Run'])
MERGE (caller)-[:CALLS {cross_repo: true}]->(callee)
"""

CYPHER_LINK_XREPO_CALLS_TRANSITIVE = """
MATCH (A:Function {repo: $parent_repo})-[:CALLS]->(B:Function {repo: $parent_repo})
MATCH (B)-[:CALLS {cross_repo: true}]->(C:Function {repo: $helper_repo})
WHERE NOT (A)-[:CALLS {cross_repo: true}]->(C)
MERGE (A)-[:CALLS {cross_repo: true, via: B.name}]->(C)
"""

CYPHER_CROSS_REPO_IMPACT = """
MATCH (helperCommit:Commit {sha: $commit_sha})-[:MODIFIED]->(changed)
WHERE changed:Function OR changed:File
WITH collect(changed) AS changedNodes
MATCH (parentFile:File {repo: $parent_repo})-[:USES_REPO]->(helperRepo:Repository {full_name: $helper_repo})
OPTIONAL MATCH (parentFunc:Function {repo: $parent_repo})-[:CALLS {cross_repo: true}]->(helperFunc:Function)
WHERE helperFunc IN changedNodes
WITH changedNodes, collect(DISTINCT {
  affected_file: parentFile.path,
  affected_function: parentFunc.name,
  affected_in_file: parentFunc.filepath,
  impact_type: 'CALLS'
}) AS call_impacts
OPTIONAL MATCH (changedFile:File)-[:DECLARES]->(changedFunc:Function)
WHERE changedFile IN changedNodes OR changedFunc IN changedNodes
WITH changedNodes, call_impacts, collect(DISTINCT changedFile.path) AS changedPaths
OPTIONAL MATCH (parentFile2:File {repo: $parent_repo})-[:DEPENDS_ON]->(m:Module)-[:REPRESENTS]->(helperRepo2:Repository {full_name: $helper_repo})
WITH call_impacts, collect(DISTINCT {
  affected_file: parentFile2.path,
  affected_function: null,
  affected_in_file: parentFile2.path,
  impact_type: 'DEPENDS_ON'
}) AS dep_impacts
UNWIND (call_impacts + dep_impacts) AS impact
WHERE impact.affected_file IS NOT NULL
RETURN DISTINCT
  impact.affected_file AS affected_file,
  impact.affected_function AS affected_function,
  impact.affected_in_file AS affected_in_file,
  impact.impact_type AS impact_type
ORDER BY affected_file
"""

# --- Indexes & Constraints Block ---

CYPHER_VECTOR_INDEX = """
CREATE VECTOR INDEX issue_embeddings IF NOT EXISTS
FOR (i:Issue) ON (i.embedding)
OPTIONS {
  indexConfig: {
    `vector.dimensions`: 3072,
    `vector.similarity_function`: 'cosine'
  }
}
"""

CYPHER_CODE_VECTOR_INDEX = """
CREATE VECTOR INDEX code_embeddings IF NOT EXISTS
FOR (n:Function) ON (n.embedding)
OPTIONS {
  indexConfig: {
    `vector.dimensions`: 3072,
    `vector.similarity_function`: 'cosine'
  }
}
"""

CYPHER_MODULE_VECTOR_INDEX = """
CREATE VECTOR INDEX module_embeddings IF NOT EXISTS
FOR (m:Module) ON (m.embedding)
OPTIONS {
  indexConfig: {
    `vector.dimensions`: 3072,
    `vector.similarity_function`: 'cosine'
  }
}
"""

CYPHER_TYPE_VECTOR_INDEX = """
CREATE VECTOR INDEX type_embeddings IF NOT EXISTS
FOR (t:Type) ON (t.embedding)
OPTIONS {
  indexConfig: {
    `vector.dimensions`: 3072,
    `vector.similarity_function`: 'cosine'
  }
}
"""

CYPHER_VAR_VECTOR_INDEX = """
CREATE VECTOR INDEX var_embeddings IF NOT EXISTS
FOR (v:Variable) ON (v.embedding)
OPTIONS {
  indexConfig: {
    `vector.dimensions`: 3072,
    `vector.similarity_function`: 'cosine'
  }
}
"""

CYPHER_FULLTEXT_INDEX = """
CREATE FULLTEXT INDEX commit_summaries IF NOT EXISTS
FOR (c:Commit) ON EACH [c.summary_text, c.diff_text, c.message]
"""

CYPHER_UPSERT_STATUS = """
MERGE (bs:BootstrapStatus {repo: $repo_full_name})
SET bs.status            = $status,
    bs.detail            = $detail,
    bs.commits_processed = $commits_processed,
    bs.files_scanned     = $files_scanned,
    bs.updated_at        = $updated_at
"""

CYPHER_CONSTRAINT_FILE_REPO = """
CREATE CONSTRAINT file_repo_unique IF NOT EXISTS
FOR (f:File) REQUIRE (f.path, f.repo) IS UNIQUE
"""

CYPHER_CONSTRAINT_FUNC_ID = """
CREATE CONSTRAINT func_id_unique IF NOT EXISTS
FOR (f:Function) REQUIRE f.id IS UNIQUE
"""

CYPHER_CONSTRAINT_TYPE_ID = """
CREATE CONSTRAINT type_id_unique IF NOT EXISTS
FOR (t:Type) REQUIRE t.id IS UNIQUE
"""

CYPHER_CONSTRAINT_VAR_ID = """
CREATE CONSTRAINT var_id_unique IF NOT EXISTS
FOR (v:Variable) REQUIRE v.id IS UNIQUE
"""

CYPHER_CONSTRAINT_DIRECTIVE_UNIQUE = """
CREATE CONSTRAINT directive_unique IF NOT EXISTS
FOR (d:Directive) REQUIRE (d.filepath, d.repo, d.directive, d.args) IS UNIQUE
"""

CYPHER_INDEX_FUNC_NAME_REPO = """
CREATE INDEX func_name_repo IF NOT EXISTS FOR (f:Function) ON (f.name, f.repo)
"""

CYPHER_INDEX_TYPE_NAME_REPO = """
CREATE INDEX type_name_repo IF NOT EXISTS FOR (t:Type) ON (t.name, t.repo)
"""

CYPHER_INDEX_TYPE_KIND = """
CREATE INDEX type_kind IF NOT EXISTS FOR (t:Type) ON (t.kind)
"""

CYPHER_CLEANUP_UNSCOPED_FILES = "MATCH (f:File)      WHERE f.repo IS NULL DETACH DELETE f"
CYPHER_CLEANUP_UNSCOPED_DIRS  = "MATCH (d:Directory) WHERE d.repo IS NULL DETACH DELETE d"
CYPHER_CLEANUP_UNSCOPED_FUNCS = "MATCH (f:Function)  WHERE f.repo IS NULL DETACH DELETE f"

CYPHER_MARK_FILE_SCANNED = "MATCH (file:File {path: $filepath, repo: $repo_full_name}) SET file.content_scanned = true"

CYPHER_INGEST_IFACE_METHOD = """
MERGE (repo:Repository {full_name: $repo_full_name})
MERGE (file:File {path: $filepath, repo: $repo_full_name})
MERGE (f:Function {id: $method_id})
  ON CREATE SET
    f.name               = $method_name,
    f.filepath           = $filepath,
    f.repo               = $repo_full_name,
    f.code               = $signature,
    f.is_exported        = $is_exported,
    f.is_pointer_receiver = false,
    f.channels_sent      = [],
    f.channels_received  = [],
    f.qualified_calls    = []
  ON MATCH SET
    f.name               = $method_name,
    f.code               = coalesce($signature, f.code),
    f.is_exported        = $is_exported
MERGE (file)-[:DECLARES]->(f)
MERGE (repo)-[:DECLARES]->(f)
WITH f
MATCH (t:Type {id: $type_id})
MERGE (t)-[:DECLARES_METHOD]->(f)
"""

CYPHER_LINK_DECLARES_METHOD = """
MATCH (f:Function)
WHERE f.filepath = $filepath
  AND f.repo     = $repo_full_name
  AND f.name     CONTAINS '.'
WITH f, split(f.name, '.')[0] AS receiver_name
MATCH (t:Type {name: receiver_name, repo: $repo_full_name})
MERGE (t)-[:DECLARES_METHOD]->(f)
"""

# ──────────────────────────────────────────────────────────────────────────────
# Post-scan CALLS re-resolution pass
# ──────────────────────────────────────────────────────────────────────────────

# Re-attempts to create CALLS edges for any Function whose qualified_calls list
# contains names that could not be resolved at initial ingest time (because the
# callee's file hadn't been parsed yet when the caller's file was processed).
#
# Resolution priority (mirrors CYPHER_INGEST_CALLS):
#   1. File-local (same filepath, same repo)
#   2. Package-local (same directory, same repo)
#   3. Qualified last-segment match (uses stored qualified call string)
#   4. Repo-wide heuristic (cross-directory, non-generic names >= 5 chars)
#
# We scope to $repo_full_name so a single driver call handles one repo at a
# time, keeping transaction size predictable.
CYPHER_RELINK_UNRESOLVED_CALLS = """
MATCH (caller:Function {repo: $repo_full_name})
WHERE caller.qualified_calls IS NOT NULL
  AND size(caller.qualified_calls) > 0
UNWIND caller.qualified_calls AS qcall
WITH caller, qcall,
     split(qcall, '.')[-1]  AS callee_bare,
     split(qcall, '.')[0]   AS callee_prefix

// 1. File-local
OPTIONAL MATCH (local:Function {name: callee_bare, filepath: caller.filepath, repo: $repo_full_name})

// 2. Package-local (same directory, different file)
OPTIONAL MATCH (pkg_local:Function {name: callee_bare, repo: $repo_full_name})
WHERE pkg_local.filepath <> caller.filepath
  AND split(pkg_local.filepath, '/')[0..-1] = split(caller.filepath, '/')[0..-1]

// 3. Qualified last-segment match (cross-directory, package-prefixed call)
OPTIONAL MATCH (qual_match:Function {repo: $repo_full_name})
WHERE callee_prefix <> callee_bare        // only if there actually is a prefix
  AND qual_match.name = callee_bare
  AND toLower(qual_match.filepath) CONTAINS toLower(callee_prefix)

// 4. Repo-wide heuristic for unambiguous names
OPTIONAL MATCH (repo_wide:Function {name: callee_bare, repo: $repo_full_name})
WHERE repo_wide.filepath <> caller.filepath
  AND split(repo_wide.filepath, '/')[0..-1] <> split(caller.filepath, '/')[0..-1]
  AND callee_prefix = callee_bare          // no package prefix → unqualified call
  AND NOT callee_bare IN ['New', 'Run', 'Execute', 'Close', 'Open', 'Init',
                           'String', 'Read', 'Write', 'Update', 'Start', 'Stop', 'Reset']
  AND size(callee_bare) >= 5

WITH caller, coalesce(local, pkg_local, qual_match, repo_wide) AS callee
WHERE callee IS NOT NULL
  AND caller <> callee
  AND NOT (caller)-[:CALLS]->(callee)   // skip already-linked pairs
MERGE (caller)-[r:CALLS]->(callee)
  ON CREATE SET r.call_type = 'SYNC', r.relinked = true
RETURN count(r) AS edges_created
"""

# Patches Function stub nodes that were created by CYPHER_MODIFIED_FUNCTION
# (commit ingestion writes MERGE (func:Function {id: $func_id}) with NO code)
# before their source file was processed by phase2_scan_file_contents.
#
# After phase2 runs, any stub whose source file is now marked content_scanned
# but still has an empty/null code property should be back-filled from the
# fully-parsed sibling Function node that phase2 created under the same id.
# Because CYPHER_INGEST_FUNCTION uses MERGE on id and sets code in ON MATCH,
# phase2 already overwrites stubs — so this query is a safety net that catches
# the edge case where a stub's id was synthesised differently by the commit
# path vs the scan path.
#
# We match stubs by:
#   - code IS NULL or code = ''
#   - their filepath exists as a File with content_scanned = true
#   - a sibling Function in the same filepath + same name exists with real code
CYPHER_PATCH_EMPTY_CODE_STUBS = """
MATCH (stub:Function {repo: $repo_full_name})
WHERE (stub.code IS NULL OR stub.code = '')
  AND stub.filepath IS NOT NULL
MATCH (scanned:File {path: stub.filepath, repo: $repo_full_name, content_scanned: true})
MATCH (donor:Function {name: stub.name, filepath: stub.filepath, repo: $repo_full_name})
WHERE donor <> stub
  AND donor.code IS NOT NULL
  AND donor.code <> ''
SET stub.code               = donor.code,
    stub.embedding          = coalesce(stub.embedding, donor.embedding),
    stub.qualified_calls    = coalesce(stub.qualified_calls, donor.qualified_calls),
    stub.return_types       = coalesce(stub.return_types, donor.return_types),
    stub.accepts_context    = coalesce(stub.accepts_context, donor.accepts_context),
    stub.lock_sequence      = coalesce(stub.lock_sequence, donor.lock_sequence),
    stub.channels_sent      = coalesce(stub.channels_sent, donor.channels_sent),
    stub.channels_received  = coalesce(stub.channels_received, donor.channels_received),
    stub.propagated_errors  = coalesce(stub.propagated_errors, donor.propagated_errors),
    stub.is_exported        = coalesce(stub.is_exported, donor.is_exported),
    stub.is_pointer_receiver = coalesce(stub.is_pointer_receiver, donor.is_pointer_receiver),
    stub.is_test            = coalesce(stub.is_test, donor.is_test)
RETURN count(stub) AS stubs_patched
"""

# ──────────────────────────────────────────────────────────────────────────────
# Pass A — Step 1: Find distinct filepaths that contain Function nodes with
# empty or null .code. These files need to be re-fetched from GitHub and
# re-parsed by phase2_scan_file_contents.
# ──────────────────────────────────────────────────────────────────────────────
CYPHER_FIND_EMPTY_CODE_FILES = """
MATCH (f:Function {repo: $repo_full_name})
WHERE (f.code IS NULL OR f.code = '')
  AND f.filepath IS NOT NULL
  AND f.filepath <> ''
RETURN DISTINCT f.filepath AS filepath
ORDER BY filepath
"""

# ──────────────────────────────────────────────────────────────────────────────
# Pass A — Step 2: Unmark those files so phase2_scan_file_contents will
# re-process them. REMOVE removes the property entirely; phase2 re-sets it
# to true after successful scanning via CYPHER_MARK_FILE_SCANNED.
# ──────────────────────────────────────────────────────────────────────────────
CYPHER_UNMARK_FILES_FOR_RESCAN = """
MATCH (f:File {repo: $repo_full_name})
WHERE f.path IN $paths
REMOVE f.content_scanned
RETURN count(f) AS files_unmarked
"""

# ──────────────────────────────────────────────────────────────────────────────
# Pass C — Code-grep fallback for same-package calls.
# ──────────────────────────────────────────────────────────────────────────────
CYPHER_RELINK_CALLS_BY_CODE_GREP = """
MATCH (caller:Function {repo: $repo_full_name})
WHERE caller.code IS NOT NULL
  AND size(caller.code) > 20
MATCH (callee:Function {repo: $repo_full_name})
WHERE callee.code IS NOT NULL
  AND caller <> callee
  AND callee.filepath <> caller.filepath
  AND split(callee.filepath, '/')[0..-1] = split(caller.filepath, '/')[0..-1]
  AND caller.code CONTAINS callee.name
  AND size(callee.name) >= 6
  AND NOT callee.name IN ['String', 'Error', 'Close', 'Write', 'Read', 'Reset',
                           'Marshal', 'Unmarshal', 'Format', 'Append', 'Encode', 'Decode']
  AND NOT (caller)-[:CALLS]->(callee)
MERGE (caller)-[r:CALLS]->(callee)
  ON CREATE SET r.call_type = 'SYNC', r.relinked = true, r.via_grep = true
RETURN count(r) AS edges_created
"""