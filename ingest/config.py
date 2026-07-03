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
}

IGNORED_SUFFIXES = (
    ".bin", ".dll", ".exe", ".gif", ".gz", ".ico", ".jpeg", ".jpg",
    ".lock", ".mp3", ".mp4", ".pdf", ".png", ".so", ".svg", ".tar",
    ".tgz", ".ttf", ".woff", ".woff2", ".zip",
)

_GRAPH_FLUSH_BATCH = 50
