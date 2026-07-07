# =============================================================================
# backend/app/query_parser.py — Query Sanitization, Intent & Hint Extraction
# =============================================================================

import os
import re
from functools import lru_cache
from typing import List, Dict, Optional
from pydantic import BaseModel, Field

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    _LANGCHAIN_GENAI_AVAILABLE = True
except ImportError:
    _LANGCHAIN_GENAI_AVAILABLE = False

try:
    from .config import logger
except (ImportError, ValueError):
    import logging
    logger = logging.getLogger("graphrag")

# Common Go/Python component path segments that indicate a specific file target.
_COMPONENT_KEYWORDS = [
    "help", "textinput", "textarea", "list", "paginator", "progress",
    "spinner", "viewport", "filepicker", "table", "cursor", "key",
    "styles", "style",
]

# Keywords that signal a cross-package / migration impact question.
_IMPACT_KEYWORDS = (
    "move", "moved", "moving", "migrate", "migrated", "migration",
    "extract", "extracted", "remove", "removed", "removing",
    "delete", "deleted", "deleting", "rename", "renamed", "renaming",
    "drop", "dropped", "dropping", "deprecate", "deprecated",
    "affect", "affected", "affects", "impact", "impacts",
    "depend", "depends", "dependency", "dependencies", "import", "imports",
    "refactor", "split", "separate", "what breaks", "what will break", "what changes",
    "if i change", "if i modify", "if we change", "if we modify", "breaking", "breaks",
)

# New: Heuristic indicators for structural typing queries
_STRUCTURAL_KEYWORDS = (
    "implement", "implements", "implementing", "satisfy", "satisfies", "satisfied",
    "duck typing", "duck-typing", "interface contract", "method set", "method-set",
    "struct embedding", "type hierarchy", "hierarchy", "composition", "declares"
)

_BLAME_PATTERN = re.compile(
    r'\bwho\b.{0,25}\b(added|wrote|changed|modified|introduced|created|implemented|broke|touched|owns|did)\b'
    r'|\bwhich commit\b'
    r'|\bwhen was\b'
    r'|\bwhen did\b'
    r'|\bgit blame\b'
    r'|\bblame\b'
    r'|\bwho is responsible\b'
    r'|\bwho owns\b',
    re.IGNORECASE,
)

_COMMIT_KEYWORDS = (
    "commit", "commits", "merge", "push", "pr", "pull request",
    "issue", "bug", "fix", "release", "tag", "author", "contributor",
    "last", "recent", "latest", "history",
    "change", "changed", "changes", "changing",
    "modify", "modified", "modifying",
    "update", "updated", "updating",
    "frequently", "often", "churn", "when was",
)
_RECENCY_KEYWORDS = ("last", "recent", "latest", "newest", "history")


_CONCURRENCY_KEYWORDS = (
    "goroutine", "channel", "chan", "mutex", "lock", "waitgroup", "context", "defer", "concurrency", "go routine", "sync.mutex", "sync.rwmutex", "sync.waitgroup"
)
_TEST_COVERAGE_KEYWORDS = ("test", "testing", "tests", "coverage", "covered by tests", "test coverage")
_DIRECTIVE_KEYWORDS = ("directive", "directives", "go:generate", "go:build", "go:embed", "build tag", "build tags", "compiler tag", "compiler tags", "embed asset", "embed assets")
_SCHEMA_KEYWORDS = ("schema", "struct tag", "struct tags", "tag mapping", "tag mappings", "json:", "bson:", "db:", "omitempty", "db tag", "json tag", "bson tag", "database table", "database mapping", "serialize", "serialization")


def _sanitize_lucene_query(query: str) -> str:
    sanitized = re.sub(r'[/\\\?\*~\^\[\]{}()!]', ' ', query)
    sanitized = re.sub(r'\s+', ' ', sanitized).strip()
    return sanitized or query


def _extract_path_hints(query: str) -> List[str]:
    hints: list[str] = []
    for m in re.finditer(r'[\w\-]+/[\w\-]+', query):
        hints.append(m.group().split("/")[-1].lower())
    for m in re.finditer(r'\b([\w\-]+)\.go\b', query, re.IGNORECASE):
        hints.append(m.group(1).lower())
    q_lower = query.lower()
    for kw in _COMPONENT_KEYWORDS:
        if re.search(rf'\b{re.escape(kw)}\b', q_lower):
            hints.append(kw)
    return list(dict.fromkeys(hints))


def _extract_blame_hints(query: str) -> Dict[str, List[str]]:
    func_hints: list[str] = []
    file_hints: list[str] = []

    for m in re.finditer(r'[\'"`]([\w_]+)[\'"`]', query):
        func_hints.append(m.group(1))
    for m in re.finditer(r'\b([a-z][a-z0-9]*(?:_[a-z0-9]+)+|[A-Z][a-zA-Z0-9]{2,})\b', query):
        word = m.group(1)
        if word.lower() not in {
            "who", "what", "when", "where", "which", "how", "the", "this", "that",
            "was", "were", "has", "have", "did", "does", "added", "changed",
            "introduced", "modified", "created", "wrote",
        }:
            func_hints.append(word)
    for m in re.finditer(r'\b([\w_\-]+\.(?:py|go|ts|js|java|cpp|c|rb|rs|kt))\b', query, re.IGNORECASE):
        file_hints.append(m.group(1))
        func_hints.append(m.group(1).rsplit(".", 1)[0])
    for m in re.finditer(r'\b([\w\-]+/[\w\-]+)\b', query):
        file_hints.append(m.group(1))

    return {
        "func_hints": list(dict.fromkeys(h for h in func_hints if len(h) > 2)),
        "file_hints": list(dict.fromkeys(file_hints))
    }


def _extract_impact_subjects(query: str) -> List[str]:
    query = re.sub(r'\(\s*\)', '', query)
    candidates: list[str] = []

    for m in re.finditer(r'[\'"`]([\w_\-\.]+)[\'"`]', query):
        candidates.append(m.group(1).lower())
    for m in re.finditer(
        r'\b(package|module|library|pkg|component|resource|struct|class|type|interface|goroutine|channel|chan|var|variable|defer|directive|field|property|attribute)\s+([\w_\-\.]+)',
        query, re.IGNORECASE
    ):
        candidates.append(m.group(2).lower())
    for m in re.finditer(
        r'\b(update|modify|remove|rename|delete|drop|deprecate|change|adding|removing|move|extract|migrate|split)\s+([\w_\-\.]+)',
        query, re.IGNORECASE
    ):
        cand = m.group(2).lower()
        if len(cand) > 2:
            candidates.append(cand)
    for m in re.finditer(r'\b([A-Z][a-z]+(?:[A-Z][a-z0-9]*)+)\b', query):
        candidates.append(m.group(1).lower())
    for m in re.finditer(r'\b([a-z][a-z0-9]+(?:[_\-][a-z0-9]+)+)\b', query):
        candidates.append(m.group(1).lower())

    _STOPWORDS = {
        "the", "from", "into", "what", "will", "get", "this", "that", "to", "in", "of", 
        "a", "an", "and", "or", "if", "is", "are", "for", "at", "by", "be", "it", "on", 
        "as", "do", "how", "field", "repo", "repository", "package", "struct", "class",
    }
    return list(dict.fromkeys(c for c in candidates if c not in _STOPWORDS and len(c) > 2))


def _extract_field_hints(query: str) -> List[str]:
    field_hints: list[str] = []
    for m in re.finditer(
        r'\b(?:remov(?:e|ing)|delet(?:e|ing)|dropp?(?:ing)?|renam(?:e|ing)|updat(?:e|ing))\s+'
        r'(?:the\s+)?([A-Za-z_][\w_.]*)\s+(?:(?:interface|function|struct|config|method|field|property|attribute|param|parameter|arg|argument|return|setting|tag|chan|channel|goroutine|var|variable|defer)\s+)*(?:field|property|attribute|column|param|parameter|arg|argument|method|function|signature|key|config|setting|tag|chan|channel|goroutine|var|variable|defer)',
        query, re.IGNORECASE
    ):
        field_hints.append(m.group(1).lower())

    for m in re.finditer(
        r'\b([A-Z][a-z]+|[a-z][a-z0-9]+)\s+(?:(?:interface|function|struct|config|method|field|property|attribute|param|parameter|arg|argument|return|setting|tag|chan|channel|goroutine|var|variable|defer)\s+)*(?:[Ff]ield|[Mm]ethod|[Ff]unction|[Ss]ignature|[Kk]ey|[Pp]roperty|[Pp]arameter|[A]rgument|[S]etting|[Tt]ag|[Cc]hannel|[Gg]oroutine|[Vv]ariable|[Dd]efer)\b',
        query
    ):
        word = m.group(1).lower()
        if word not in {"the", "a", "this", "that", "some", "any", "interface", "function", "struct", "config", "method", "field", "property", "attribute", "param", "parameter", "arg", "argument", "return", "setting", "tag", "chan", "channel", "goroutine", "var", "variable", "defer"}:
            field_hints.append(word)

    expanded_hints = []
    for h in field_hints:
        expanded_hints.append(h)
        if "." in h:
            expanded_hints.append(h.split(".")[-1])
    return list(dict.fromkeys(expanded_hints))


def _extract_repo_hints_from_query(query: str) -> List[str]:
    repo_hints: list[str] = []
    for m in re.finditer(r'\b([\w\-]+/[\w\-]+)\b', query):
        repo_hints.append(m.group(1).lower())
    for m in re.finditer(r'\b([\w][\w\-]+)\s+(?:repo|repository|codebase|service|module)\b', query, re.IGNORECASE):
        candidate = m.group(1).lower()
        if len(candidate) > 2:
            repo_hints.append(candidate)
    return list(dict.fromkeys(repo_hints))


class QueryIntent(BaseModel):
    wants_impact: bool = Field(default=False, description="True if the user is asking what depends on, imports, calls, or would break from changing something.")
    wants_blame: bool = Field(default=False, description="True if the user is asking who wrote, changed, modified, or owns some code.")
    wants_commit_files: bool = Field(default=False, description="True if asking about commit history, file modification frequency, churn, or what a specific commit touched.")
    wants_recency: bool = Field(default=False, description="True if asking for recent/latest/last N commits or commit history.")
    wants_structural: bool = Field(default=False, description="True if the user is asking about structural/type relationships: which structs implement an interface, duck typing, method sets, etc.")
    wants_concurrency: bool = Field(default=False, description="True if the user is asking about concurrency, goroutines, channels, mutexes, locks, waitgroups, context, or deferred logic.")
    wants_test_coverage: bool = Field(default=False, description="True if asking about test coverage, test functions, which tests run/cover a function, or untested code.")
    wants_directive: bool = Field(default=False, description="True if asking about compiler directives, build tags, go:generate, go:build, or go:embed.")
    wants_schema: bool = Field(default=False, description="True if asking about database mappings, struct tags, BSON, JSON field tags, or serialization schemas.")

    subjects: List[str] = Field(default_factory=list, description="Literal package, module, or symbol names the user is asking about.")
    field_hints: List[str] = Field(default_factory=list, description="Specific struct fields, properties, function signatures, methods, or config keys mentioned.")
    repo_hints: List[str] = Field(default_factory=list, description="Repository names mentioned.")
    file_hints: List[str] = Field(default_factory=list, description="File name fragments mentioned.")


@lru_cache(maxsize=512)
def _cached_llm_intent(query: str, api_key: str) -> Optional[QueryIntent]:
    if not _LANGCHAIN_GENAI_AVAILABLE or not api_key:
        return None
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key, temperature=0.0)
        structured_llm = llm.with_structured_output(QueryIntent)
        intent = structured_llm.invoke(
            "Classify this code-search query and extract literal entity names.\n\n"
            "IMPORTANT classification rules:\n"
            "- Set wants_structural=True (NOT wants_impact) when the query is about type/interface relationships.\n"
            "- Set wants_impact=True ONLY when the user asks what BREAKS, DEPENDS ON, or IMPORTS something.\n"
            "- Set wants_concurrency=True when the query is about concurrency, goroutines, channels, mutexes, locks, waitgroups, context, or deferred logic.\n"
            "- Set wants_test_coverage=True when the query is about test coverage, test functions, which tests cover a struct/function, or untested code.\n"
            "- Set wants_directive=True when the query is about compiler directives, build tags, go:generate, go:build, or go:embed.\n"
            "- Set wants_schema=True when the query is about database mappings, struct tags, BSON, JSON field tags, or serialization schemas.\n"
            f"Query: {query}"
        )
        if isinstance(intent, QueryIntent): return intent
        elif isinstance(intent, dict): return QueryIntent(**intent)
    except Exception as exc:
        logger.warning("[QUERY_PARSER] ⚠️ LLM intent extraction failed: %s", exc)
    return None


def extract_query_intent(query: str, google_api_key: Optional[str] = None) -> QueryIntent:
    api_key = google_api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or ""
    intent = None
    if api_key:
        intent = _cached_llm_intent(query, api_key)
    
    if intent is not None:
        logger.info(
            "[QUERY_PARSER] ✅ LLM intent extracted: impact=%s blame=%s commit=%s recency=%s structural=%s concurrency=%s test_coverage=%s directive=%s schema=%s subjects=%s",
            intent.wants_impact, intent.wants_blame, intent.wants_commit_files, intent.wants_recency, intent.wants_structural,
            intent.wants_concurrency, intent.wants_test_coverage, intent.wants_directive, intent.wants_schema, intent.subjects,
        )
        return intent

    # --- Robust Heuristic Fallback Pipeline ---
    logger.info("[QUERY_PARSER] Using fallback regex/keyword intent extraction for query: %r", query[:100])
    q_lower = query.lower()
    
    wants_impact = any(kw in q_lower for kw in _IMPACT_KEYWORDS)
    wants_blame = bool(_BLAME_PATTERN.search(query))
    wants_commit = any(kw in q_lower for kw in _COMMIT_KEYWORDS)
    wants_recency = any(kw in q_lower for kw in _RECENCY_KEYWORDS)
    wants_structural = any(kw in q_lower for kw in _STRUCTURAL_KEYWORDS)
    wants_concurrency = any(kw in q_lower for kw in _CONCURRENCY_KEYWORDS)
    wants_test_coverage = any(kw in q_lower for kw in _TEST_COVERAGE_KEYWORDS)
    wants_directive = any(kw in q_lower for kw in _DIRECTIVE_KEYWORDS)
    wants_schema = any(kw in q_lower for kw in _SCHEMA_KEYWORDS)
    
    # Fix: Backfill subjects for structural lookups if impact keywords aren't present
    subjects = _extract_impact_subjects(query) if (wants_impact or wants_structural or wants_concurrency or wants_test_coverage or wants_directive or wants_schema) else []
    field_hints = _extract_field_hints(query)
    repo_hints = _extract_repo_hints_from_query(query)
    file_hints = _extract_path_hints(query)
    
    return QueryIntent(
        wants_impact=wants_impact,
        wants_blame=wants_blame,
        wants_commit_files=wants_commit,
        wants_recency=wants_recency,
        wants_structural=wants_structural,
        wants_concurrency=wants_concurrency,
        wants_test_coverage=wants_test_coverage,
        wants_directive=wants_directive,
        wants_schema=wants_schema,
        subjects=subjects,
        field_hints=field_hints,
        repo_hints=repo_hints,
        file_hints=file_hints,
    )