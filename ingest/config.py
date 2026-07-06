# =============================================================================
# ingest/config.py — Configuration Constants and Logging Setup
# =============================================================================

import logging

# ── Logging Setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("backend_ingest")

# ── GitHub & Pipeline Constants ──────────────────────────────────────────────
GITHUB_API_BASE = "https://api.github.com"

SOURCE_EXTENSIONS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".java", ".rs",
    ".rb", ".php", ".c", ".cpp", ".h", ".hpp", ".cs", ".swift",
    ".kt", ".scala", ".r", ".R", ".vue", ".svelte",
}

SOURCE_FILENAMES = {"go.mod", "go.work"}

ENTRY_POINT_NAMES = {
    "main", "index", "app", "server", "__main__", "manage",
    "wsgi", "asgi", "cli", "entrypoint",
}

UTILITY_DIRS = {
    "utils", "util", "lib", "libs", "common", "shared",
    "helpers", "helper", "core", "pkg", "internal",
}

IGNORED_DIRECTORIES = {
    "node_modules", "venv", ".venv", "env", "dist", "build", ".next", ".git",
}

IGNORED_FILENAMES = {
    ".ds_store", "cargo.lock", "gemfile.lock", "package-lock.json",
    "poetry.lock", "pnpm-lock.yaml", "yarn.lock",
    # go.sum contains cryptographic hashes, not import paths — scanning it
    # injects thousands of spurious Module nodes into the graph.
    "go.sum",
}

IGNORED_SUFFIXES = (
    ".bin", ".dll", ".exe", ".gif", ".gz", ".ico", ".jpeg", ".jpg",
    ".lock", ".mp3", ".mp4", ".pdf", ".png", ".so", ".svg", ".tar",
    ".tgz", ".ttf", ".woff", ".woff2", ".zip",
)

_GRAPH_FLUSH_BATCH = 50

# Maximum number of function texts sent to get_embeddings_batch() in one call.
# Gemini embed_content accepts up to 100 items per batch request.
# Large generated Go files (e.g. mocks, protobuf) can have 200+ functions;
# this cap splits them into safe chunks.
_EMBED_FUNC_BATCH_SIZE = 80

# Go test file suffixes to skip during AST/call-graph scanning.
# _test.go functions (TestXxx, BenchmarkXxx, FuzzXxx) pollute the call graph
# with test-only edges that have nothing to do with the library's public API.
_GO_TEST_SUFFIXES = ("_test.go",)
