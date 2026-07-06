# =============================================================================
# ingest/queries.py — Neo4j Cypher Query Templates & Indexes
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
    func.channels_received  = $channels_received
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
    func.channels_received  = $channels_received
MERGE (file)-[:DECLARES]->(func)
MERGE (repo)-[:DECLARES]->(func)
"""

CYPHER_INGEST_CALLS = """
MATCH (caller:Function {id: $caller_id})
// 1. Try file-local match first (same file as the caller)
OPTIONAL MATCH (local:Function {name: $callee_name, filepath: $caller_filepath, repo: $repo_full_name})
// 2. Try package/directory-local match second (same parent directory in the same repo)
OPTIONAL MATCH (pkg_local:Function {name: $callee_name, repo: $repo_full_name})
WHERE pkg_local.filepath <> $caller_filepath
  AND split(pkg_local.filepath, '/')[0..-1] = split($caller_filepath, '/')[0..-1]
// 3. Fall back to repo-wide match ONLY if callee is scoped (e.g. Receiver.Method),
// or if caller explicitly imports/depends on callee's module/path, or if callee is specific
OPTIONAL MATCH (repo_wide:Function {name: $callee_name, repo: $repo_full_name})
WHERE repo_wide.filepath <> $caller_filepath
  AND split(repo_wide.filepath, '/')[0..-1] <> split($caller_filepath, '/')[0..-1]
  AND (
       $callee_name CONTAINS '.'
    OR EXISTS { MATCH (caller_file:File {path: $caller_filepath, repo: $repo_full_name})-[:DEPENDS_ON]->(m:Module) WHERE toLower(repo_wide.filepath) CONTAINS toLower(m.name) OR toLower(m.name) ENDS WITH toLower(split(repo_wide.filepath, '/')[0]) }
    OR (NOT $callee_name IN ['New', 'Run', 'Execute', 'Close', 'Open', 'Init', 'String', 'Read', 'Write', 'Update', 'Focus', 'Blur', 'Blink', 'Start', 'Stop', 'Reset', 'Clear', 'Add', 'Remove', 'Delete', 'Get', 'Set', 'List', 'Find', 'Check', 'Verify', 'Validate', 'Parse', 'Format', 'Print', 'Println', 'Error', 'Fatal', 'Panic', 'Log', 'Debug', 'Info', 'Warn', 'Main', 'Test', 'Setup', 'Teardown', 'Config', 'Load', 'Save', 'Create', 'Send', 'Process', 'Handle'] AND size($callee_name) >= 5)
  )
WITH caller, coalesce(local, pkg_local, repo_wide) AS callee
WHERE callee IS NOT NULL
MERGE (caller)-[r:CALLS]->(callee)
  ON CREATE SET r.call_type = coalesce($call_type, 'SYNC')
  ON MATCH  SET r.call_type = coalesce($call_type, r.call_type, 'SYNC')
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
    type.is_exported  = coalesce($is_exported, false)
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
    type.is_exported  = coalesce($is_exported, false)
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
    var.is_exported = coalesce($is_exported, false)
  ON MATCH SET
    var.name        = $var_name,
    var.kind        = $kind,
    var.filepath    = $filepath,
    var.repo        = $repo_full_name,
    var.code        = $code,
    var.embedding   = coalesce($embedding, var.embedding),
    var.is_exported = coalesce($is_exported, false)
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
// Require the module path to end with /repo-name or equal the full_name exactly.
// This avoids spurious edges where a substring like "bubbles" matches unrelated modules.
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
// Match by module suffix — handles aliased imports (e.g. `import lg "github.com/charmbracelet/lipgloss"`)
// where the in-code prefix ("lg") differs from the repo/package name ("lipgloss").
OPTIONAL MATCH (m:Module)-[:REPRESENTS]->(helper_repo)
WHERE toLower(m.name) ENDS WITH "/" + toLower(prefix)
   OR toLower(m.name) = toLower(prefix)
   OR toLower(m.name) ENDS WITH "/" + toLower($helper_prefix)
WITH caller, qcall, prefix, func_name, helper_repo, m
WHERE toLower(prefix) = toLower($helper_prefix)
   OR m IS NOT NULL
MATCH (callee:Function {repo: $helper_repo})
WHERE (callee.name = func_name)
   OR (func_name IN ['New', 'NewCommand', 'Init', 'Execute', 'Run']
       AND callee.name IN ['Command', 'NewCommand', 'Execute', 'RunE', 'Run'])
MERGE (caller)-[:CALLS {cross_repo: true}]->(callee)
"""

# Second-pass: capture transitive cross-repo calls where parent-repo function A
# calls intermediate function B (same repo), and B directly calls helper-repo function C.
# This covers 2-hop chains like: bubbles::progress::Render -> bubbles::util::renderBar -> lipgloss::NewStyle
# without requiring A to have lipgloss in its own qualified_calls.
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
// Path 1: Direct cross-repo CALLS
MATCH (parentFile:File {repo: $parent_repo})-[:USES_REPO]->(helperRepo:Repository {full_name: $helper_repo})
OPTIONAL MATCH (parentFunc:Function {repo: $parent_repo})-[:CALLS {cross_repo: true}]->(helperFunc:Function)
WHERE helperFunc IN changedNodes
WITH changedNodes, collect(DISTINCT {
  affected_file: parentFile.path,
  affected_function: parentFunc.name,
  affected_in_file: parentFunc.filepath,
  impact_type: 'CALLS'
}) AS call_impacts
// Path 2: Module-level dependency impact
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

CYPHER_CLEANUP_UNSCOPED_FILES = "MATCH (f:File)      WHERE f.repo IS NULL DETACH DELETE f"
CYPHER_CLEANUP_UNSCOPED_DIRS  = "MATCH (d:Directory) WHERE d.repo IS NULL DETACH DELETE d"
CYPHER_CLEANUP_UNSCOPED_FUNCS = "MATCH (f:Function)  WHERE f.repo IS NULL DETACH DELETE f"

CYPHER_MARK_FILE_SCANNED = "MATCH (file:File {path: $filepath, repo: $repo_full_name}) SET file.content_scanned = true"

# Performance index: enables fast repo-wide name lookups in CYPHER_INGEST_CALLS.
# Without this, every call-edge write triggers a full Function label scan.
CYPHER_INDEX_FUNC_NAME_REPO = """
CREATE INDEX func_name_repo IF NOT EXISTS FOR (f:Function) ON (f.name, f.repo)
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

CYPHER_CONSTRAINT_TYPE_ID = """
CREATE CONSTRAINT type_id_unique IF NOT EXISTS
FOR (t:Type) REQUIRE t.id IS UNIQUE
"""

CYPHER_CONSTRAINT_VAR_ID = """
CREATE CONSTRAINT var_id_unique IF NOT EXISTS
FOR (v:Variable) REQUIRE v.id IS UNIQUE
"""

CYPHER_INDEX_TYPE_NAME_REPO = """
CREATE INDEX type_name_repo IF NOT EXISTS FOR (t:Type) ON (t.name, t.repo)
"""

# G4 — fast filtering on Type.kind ("STRUCT" / "INTERFACE" / "TYPE")
CYPHER_INDEX_TYPE_KIND = """
CREATE INDEX type_kind IF NOT EXISTS FOR (t:Type) ON (t.kind)
"""

# M2 — prevent duplicate Directive nodes when the same file is re-ingested
CYPHER_CONSTRAINT_DIRECTIVE_UNIQUE = """
CREATE CONSTRAINT directive_unique IF NOT EXISTS
FOR (d:Directive) REQUIRE (d.filepath, d.repo, d.directive, d.args) IS UNIQUE
"""

# G8 — materialize an interface method signature as a real :Function node so
# DECLARES_METHOD edges carry traversable code and the LLM can surface them.
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

# G3 — link all methods in a just-ingested file to their receiver Type nodes
# inline, without waiting for the enrichment script to run pass_1.
CYPHER_LINK_DECLARES_METHOD = """
MATCH (f:Function)
WHERE f.filepath = $filepath
  AND f.repo     = $repo_full_name
  AND f.name     CONTAINS '.'
WITH f, split(f.name, '.')[0] AS receiver_name
MATCH (t:Type {name: receiver_name, repo: $repo_full_name})
MERGE (t)-[:DECLARES_METHOD]->(f)
"""
