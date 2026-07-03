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
    # movement / migration
    "move", "moved", "moving",
    "migrate", "migrated", "migration",
    "extract", "extracted",
    # removal / deprecation
    "remove", "removed", "removing",
    "delete", "deleted", "deleting",
    "rename", "renamed", "renaming",
    "drop", "dropped", "dropping",
    "deprecate", "deprecated",
    # impact / dependency
    "affect", "affected", "affects",
    "impact", "impacts",
    "depend", "depends", "dependency", "dependencies",
    "import", "imports",
    "refactor", "split", "separate",
    "what breaks", "what will break", "what changes",
    "if i change", "if i modify", "if we change", "if we modify",
    "breaking", "breaks",
)

# Regex that detects blame-style intent even when adverbs sit between
# 'who' and the verb (e.g. "who last changed", "who recently modified").
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
    # modification & churn terms
    "change", "changed", "changes", "changing",
    "modify", "modified", "modifying",
    "update", "updated", "updating",
    "frequently", "often", "churn", "when was",
)
_RECENCY_KEYWORDS = ("last", "recent", "latest", "newest", "history")


def _sanitize_lucene_query(query: str) -> str:
    """
    Strip or escape characters that Lucene's classic query parser treats as
    special syntax.
    """
    sanitized = re.sub(r'[/\\\?\*~\^\[\]{}()!]', ' ', query)
    sanitized = re.sub(r'\s+', ' ', sanitized).strip()
    return sanitized or query


def _extract_path_hints(query: str) -> List[str]:
    """Pull file-path fragments out of the user query."""
    hints: list[str] = []

    for m in re.finditer(r'[\w\-]+/[\w\-]+', query):
        segment = m.group().split("/")[-1].lower()
        hints.append(segment)

    for m in re.finditer(r'\b([\w\-]+)\.go\b', query, re.IGNORECASE):
        hints.append(m.group(1).lower())

    q_lower = query.lower()
    for kw in _COMPONENT_KEYWORDS:
        if re.search(rf'\b{re.escape(kw)}\b', q_lower):
            hints.append(kw)

    seen: set[str] = set()
    unique: list[str] = []
    for h in hints:
        if h not in seen:
            seen.add(h)
            unique.append(h)
    return unique


def _extract_blame_hints(query: str) -> Dict[str, List[str]]:
    """
    Extract function name, file name, and any quoted identifier from the query
    to drive the blame Cypher lookup.
    """
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

    func_hints = list(dict.fromkeys(h for h in func_hints if len(h) > 2))
    file_hints = list(dict.fromkeys(file_hints))
    return {"func_hints": func_hints, "file_hints": file_hints}


def _extract_impact_subjects(query: str) -> List[str]:
    """Extract candidate package/module/symbol names from an impact query."""
    candidates: list[str] = []

    for m in re.finditer(r'[\'"`]([\w_\-\.]+)[\'"`]', query):
        candidates.append(m.group(1).lower())

    for m in re.finditer(
        r'\b(?:package|module|library|pkg|component|resource|struct|class|type|field|property|attribute)\s+([\w_\-\.]+)',
        query, re.IGNORECASE
    ):
        candidates.append(m.group(1).lower())

    for m in re.finditer(
        r'\b(?:update|modify|remove|rename|delete|drop|deprecate|change|adding|removing)\s+([\w_\-\.]+)',
        query, re.IGNORECASE
    ):
        cand = m.group(1).lower()
        if len(cand) > 2:
            candidates.append(cand)

    for m in re.finditer(r'\b([A-Z][a-z]+(?:[A-Z][a-z0-9]*)+)\b', query):
        candidates.append(m.group(1).lower())

    for m in re.finditer(r'\b([a-z][a-z0-9]+(?:[_\-][a-z0-9]+)+)\b', query):
        candidates.append(m.group(1).lower())

    for m in re.finditer(
        r'\b(?:move|extract|migrate|split)\s+([\w_\-\.]+)',
        query, re.IGNORECASE
    ):
        cand = m.group(1).lower()
        if len(cand) > 2:
            candidates.append(cand)

    _STOPWORDS = {
        "the", "from", "into", "what", "will", "get", "this", "that",
        "to", "in", "of", "a", "an", "and", "or", "if", "is", "are",
        "for", "at", "by", "be", "it", "on", "as", "do", "how",
        "field", "repo", "repository", "package", "struct", "class",
    }
    seen: set[str] = set()
    unique: list[str] = []
    for c in candidates:
        if c not in seen and c not in _STOPWORDS and len(c) > 2:
            seen.add(c)
            unique.append(c)
    return unique


def _extract_field_hints(query: str) -> List[str]:
    """Extract specific field/property/method names that the user mentions removing/modifying."""
    field_hints: list[str] = []

    for m in re.finditer(
        r'\b(?:remov(?:e|ing)|delet(?:e|ing)|dropp?(?:ing)?|renam(?:e|ing)|updat(?:e|ing))\s+'
        r'(?:the\s+)?([A-Za-z_][\w_]*)\s+(?:field|property|attribute|column|param|parameter|arg|argument)',
        query, re.IGNORECASE
    ):
        field_hints.append(m.group(1).lower())

    for m in re.finditer(
        r'\b([A-Z][a-z]+|[a-z][a-z0-9]+)\s+[Ff]ield\b',
        query
    ):
        word = m.group(1).lower()
        if word not in {"the", "a", "this", "that", "some", "any"}:
            field_hints.append(word)

    return list(dict.fromkeys(field_hints))


def _extract_repo_hints_from_query(query: str) -> List[str]:
    """Extract repository names mentioned in the query."""
    repo_hints: list[str] = []

    for m in re.finditer(r'\b([\w\-]+/[\w\-]+)\b', query):
        repo_hints.append(m.group(1).lower())

    for m in re.finditer(
        r'\b([\w][\w\-]+)\s+(?:repo|repository|codebase|service|module)\b',
        query, re.IGNORECASE
    ):
        candidate = m.group(1).lower()
        if len(candidate) > 2:
            repo_hints.append(candidate)

    return list(dict.fromkeys(repo_hints))


# ── Structured Intent Schema & Extraction ────────────────────────────────────

class QueryIntent(BaseModel):
    wants_impact: bool = Field(default=False, description="True if the user is asking what depends on, imports, calls, or would break from changing something")
    wants_blame: bool = Field(default=False, description="True if the user is asking who wrote, changed, modified, or owns some code, or historical author responsibility")
    wants_commit_files: bool = Field(default=False, description="True if asking about commit history, file modification frequency, churn, most frequently changed files, what files/functions changed over time, or what files/functions a specific commit touched")
    wants_recency: bool = Field(default=False, description="True if asking for recent/latest/last N commits, commit history, or most frequently modified/changed files over time")

    subjects: List[str] = Field(default_factory=list, description="Literal package, module, or symbol names the user is asking about (e.g. 'spf13/cobra', 'QueryIntent'), exactly as they'd appear as identifiers — do not invent or expand names not implied by the query")
    field_hints: List[str] = Field(default_factory=list, description="Specific struct fields/properties mentioned as being removed/changed")
    repo_hints: List[str] = Field(default_factory=list, description="Repository names mentioned, e.g. 'gohugoio/hugo'")
    file_hints: List[str] = Field(default_factory=list, description="File name fragments mentioned")


@lru_cache(maxsize=512)
def _cached_llm_intent(query: str, api_key: str) -> Optional[QueryIntent]:
    if not _LANGCHAIN_GENAI_AVAILABLE or not api_key:
        return None
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=0.0,
        )
        structured_llm = llm.with_structured_output(QueryIntent)
        intent = structured_llm.invoke(
            f"Classify this code-search query and extract literal entity names "
            f"(package/module/symbol/repo/file names as written — do not normalize "
            f"or guess canonical forms).\n\nQuery: {query}"
        )
        if isinstance(intent, QueryIntent):
            return intent
        elif isinstance(intent, dict):
            return QueryIntent(**intent)
    except Exception as exc:
        logger.warning("[QUERY_PARSER] ⚠️ LLM intent extraction failed: %s", exc)
    return None


def extract_query_intent(query: str, google_api_key: Optional[str] = None) -> QueryIntent:
    """
    Extract structured intent and literal entity mentions from a code-search query.
    Uses Gemini structured output with LRU caching, falling back to regex/keyword heuristics on failure.
    """
    api_key = google_api_key or os.getenv("GOOGLE_API_KEY") or ""
    intent = None
    if api_key:
        intent = _cached_llm_intent(query, api_key)
    
    if intent is not None:
        logger.info(
            "[QUERY_PARSER] ✅ LLM intent extracted: impact=%s blame=%s commit=%s recency=%s subjects=%s",
            intent.wants_impact, intent.wants_blame, intent.wants_commit_files, intent.wants_recency, intent.subjects,
        )
        return intent

    # Fallback to legacy regex/keyword extraction
    logger.info("[QUERY_PARSER] Using fallback regex/keyword intent extraction for query: %r", query[:100])
    q_lower = query.lower()
    
    wants_impact = any(kw in q_lower for kw in _IMPACT_KEYWORDS)
    wants_blame = bool(_BLAME_PATTERN.search(query))
    wants_commit = any(kw in q_lower for kw in _COMMIT_KEYWORDS)
    wants_recency = any(kw in q_lower for kw in _RECENCY_KEYWORDS)
    
    subjects = _extract_impact_subjects(query) if wants_impact else []
    field_hints = _extract_field_hints(query)
    repo_hints = _extract_repo_hints_from_query(query)
    file_hints = _extract_path_hints(query)
    
    return QueryIntent(
        wants_impact=wants_impact,
        wants_blame=wants_blame,
        wants_commit_files=wants_commit,
        wants_recency=wants_recency,
        subjects=subjects,
        field_hints=field_hints,
        repo_hints=repo_hints,
        file_hints=file_hints,
    )

