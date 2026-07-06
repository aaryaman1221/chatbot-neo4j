# =============================================================================
# ingest/ai_service.py — Gemini LLM Summarization and Embeddings
# =============================================================================

import re
import time
import requests
from typing import Optional

from .config import logger
from .parser import _is_noise_file

# ── Google Gemini (google-genai SDK) ─────────────────────────────────────────
try:
    from google import genai as google_genai
    from google.genai import types as genai_types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    logger.warning("google-genai not installed — LLM summaries will use heuristics.")


def _truncate(text: str, limit: int = 5000) -> str:
    if not text:
        return ""
    return text if len(text) <= limit else f"{text[:limit]}\n... [truncated]"


def build_compact_diff(files: list, max_files: int = 12) -> list:
    filtered = []
    for item in files:
        filename = item.get("filename") or item.get("path") or "unknown"
        if _is_noise_file(filename):
            continue
        patch = item.get("patch")
        if not patch:
            continue
        filtered.append({
            "filename": filename,
            "status":    item.get("status", "modified"),
            "additions": item.get("additions", 0),
            "deletions": item.get("deletions", 0),
            "changes":   item.get("changes", 0),
            "patch":     _truncate(patch, 4000),
        })
    filtered.sort(key=lambda x: x.get("changes", 0), reverse=True)
    return filtered[:max_files]


def render_diff_text(files: list) -> str:
    sections = []
    for item in files:
        sections.append("\n".join([
            f"File: {item['filename']}",
            f"Status: {item.get('status', 'modified')}",
            f"Additions: {item.get('additions', 0)}",
            f"Deletions: {item.get('deletions', 0)}",
            "Patch:",
            item.get("patch", ""),
        ]))
    return "\n\n".join(sections)


def _extract_patch_symbols(patch: str) -> list:
    if not patch:
        return []
    symbols = set()
    for line in patch.split("\n"):
        if line.startswith("@@"):
            parts = line.split("@@", 2)
            if len(parts) >= 3:
                ctx = parts[2].strip()
                m = re.search(r'(?:(?:async\s+)?def|func|class|function|interface|type|struct)\s+(?:\([^)]+\)\s+)?([a-zA-Z0-9_]+)', ctx)
                if m:
                    symbols.add(m.group(1))
                elif ctx and not ctx.startswith(("#", "//", "/*", "*", "-", "+")):
                    m2 = re.search(r'([a-zA-Z0-9_]+)\s*(?:\(|:=|=|\{)', ctx)
                    if m2 and m2.group(1) not in ("if", "for", "while", "return", "switch", "case", "else", "elif"):
                        symbols.add(m2.group(1))
        elif line.startswith(("+", "-")) and not line.startswith(("+++", "---")):
            content = line[1:].strip()
            m = re.search(r'^(?:export\s+)?(?:async\s+)?(?:def|func|class|function|interface|type|struct)\s+(?:\([^)]+\)\s+)?([a-zA-Z0-9_]+)', content)
            if m and m.group(1) not in ("if", "for", "while", "return", "switch", "case", "else", "elif", "import", "from", "package", "var", "const", "let"):
                symbols.add(m.group(1))
    return sorted(list(symbols))


def _heuristic_summary(
    repo_full_name: str,
    compact_files: list,
    commit_msg: str,
    actor_login: Optional[str] = None,
) -> str:
    logger.info("heuristic summary used")
    author_part = f" by {actor_login}" if actor_login and actor_login != "unknown" else ""
    header = f"Commit in {repo_full_name}{author_part}: {commit_msg[:200].strip()}"
    if not compact_files:
        return header

    total_add = sum(item.get("additions", 0) for item in compact_files)
    total_del = sum(item.get("deletions", 0) for item in compact_files)
    stats_line = f"Stats: {len(compact_files)} files changed (+{total_add} / -{total_del})"

    file_lines = []
    for item in compact_files[:10]:
        filename = item.get("filename") or item.get("path") or "unknown"
        status = item.get("status", "modified")
        adds = item.get("additions", 0)
        dels = item.get("deletions", 0)
        line_str = f"- {filename} ({status}, +{adds}/-{dels})"
        symbols = _extract_patch_symbols(item.get("patch", ""))
        if symbols:
            line_str += f"\n  ↳ Modified symbols/functions: {', '.join(symbols[:8])}" if len(symbols) <= 8 else f"\n  ↳ Modified symbols/functions: {', '.join(symbols[:8])} (+{len(symbols)-8} more)"
        file_lines.append(line_str)

    if len(compact_files) > 10:
        file_lines.append(f"... and {len(compact_files) - 10} more files.")

    return f"{header}\n\n{stats_line}\n\nKey changes:\n" + "\n".join(file_lines)


def summarize_with_llm(
    repo_full_name: str,
    actor_login: str,
    commit_msg: str,
    compact_files: list,
    raw_diff: str,
    google_api_key: Optional[str] = None,
) -> str:
    if not google_api_key:
        return _heuristic_summary(repo_full_name, compact_files, commit_msg, actor_login=actor_login)

    file_overview = "\n".join(
        f"- {item['filename']} ({item.get('status','modified')} "
        f"| +{item.get('additions',0)} / -{item.get('deletions',0)})"
        for item in compact_files
    ) or "- No text patches available."

    system_instruction = (
        "You are an expert senior developer reviewing code changes. "
        "Explain what changed and why it matters in plain English. "
        "Do not recite code lines. Focus on behavior, architecture, bug fixes, and risks. "
        "Keep it to at most 3 short paragraphs."
    )

    user_prompt = (
        f"Repository: {repo_full_name}\n"
        f"Actor: {actor_login or 'unknown'}\n"
        f"Commit: {commit_msg[:200]}\n\n"
        f"Files changed:\n{file_overview}\n\n"
        f"Diff:\n{_truncate(raw_diff, 20000)}"
    )

    if GENAI_AVAILABLE:
        try:
            client = google_genai.Client(api_key=google_api_key, http_options={'timeout': 60.0})
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=user_prompt,
                config=genai_types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.2,
                    max_output_tokens=1500,
                ),
            )
            text = response.text.strip() if response.text else ""
            if text:
                return text
        except Exception as exc:
            logger.debug("google-genai SDK failed: %s", exc)

    model = "gemini-2.5-flash"
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={google_api_key}"
    )
    payload = {
        "systemInstruction": {"parts": [{"text": system_instruction}]},
        "contents":          [{"parts": [{"text": user_prompt}]}],
        "generationConfig":  {"temperature": 0.2, "maxOutputTokens": 1500},
    }
    try:
        for attempt in range(5):
            resp = requests.post(
                url,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=60,
            )
            if resp.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            resp.raise_for_status()
            break
        data = resp.json()
        text = (
            data.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
            .strip()
        )
        if text:
            return text
    except Exception as exc:
        logger.debug("REST LLM fallback failed: %s", exc)

    return _heuristic_summary(repo_full_name, compact_files, commit_msg, actor_login=actor_login)


def get_embedding(text: str, api_key: str) -> list:
    if not text or not api_key:
        return []
    if GENAI_AVAILABLE:
        try:
            client = google_genai.Client(api_key=api_key, http_options={'timeout': 60.0})
            resp = client.models.embed_content(model="gemini-embedding-2", contents=text)
            return list(resp.embeddings[0].values)
        except Exception as exc:
            logger.debug("google-genai embed failed: %s", exc)
    return []


def get_embeddings_batch(texts: list, api_key: str) -> list:
    """Embed a list of texts in a single Gemini API call."""
    if not texts or not api_key:
        return [[] for _ in texts]
    if GENAI_AVAILABLE:
        try:
            client = google_genai.Client(api_key=api_key, http_options={'timeout': 60.0})
            resp = client.models.embed_content(model="gemini-embedding-2", contents=texts)
            return [list(emb.values) for emb in resp.embeddings]
        except Exception as exc:
            logger.debug("Batch embed failed — falling back to individual calls: %s", exc)
    return [get_embedding(t, api_key) for t in texts]
