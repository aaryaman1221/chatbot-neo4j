# =============================================================================
# backend/app/embeddings.py — Google Gemini Vector Embedding Generation
# =============================================================================

import requests
from .config import logger, GENAI_AVAILABLE

try:
    from google import genai as google_genai
except ImportError:
    pass


def get_embedding(text: str, api_key: str) -> list:
    """Return a float vector for *text*. Logs the retrieval path taken."""
    if not text or not api_key:
        logger.warning("[EMBED] Skipped — text or api_key is empty.")
        return []

    if GENAI_AVAILABLE:
        logger.debug("[EMBED] Trying google-genai SDK (model=gemini-embedding-2) …")
        try:
            client = google_genai.Client(api_key=api_key)
            resp = client.models.embed_content(model="gemini-embedding-2", contents=text)
            vec = list(resp.embeddings[0].values)
            logger.info("[EMBED] ✅ google-genai SDK — vector dim=%d", len(vec))
            return vec
        except Exception as exc:
            logger.warning("[EMBED] ❌ google-genai SDK failed (%s) — trying REST fallback.", exc)
    else:
        logger.debug("[EMBED] google-genai SDK not available — going straight to REST.")

    # REST fallback
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2:embedContent?key={api_key}"
    payload = {"model": "models/gemini-embedding-2", "content": {"parts": [{"text": text}]}}
    try:
        logger.debug("[EMBED] Trying REST endpoint …")
        resp = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=30)
        resp.raise_for_status()
        vec = resp.json().get("embedding", {}).get("values", [])
        if vec:
            logger.info("[EMBED] ✅ REST fallback succeeded — vector dim=%d", len(vec))
        else:
            logger.warning("[EMBED] ❌ REST fallback returned empty embedding.")
        return vec
    except Exception as exc:
        logger.error("[EMBED] ❌ REST fallback also failed: %s", exc)
        return []
