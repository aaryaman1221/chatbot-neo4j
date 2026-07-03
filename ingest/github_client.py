# =============================================================================
# ingest/github_client.py — GitHub API Client & Async Commit Fetching
# =============================================================================

import time
import asyncio
import httpx
import requests

from .config import logger, GITHUB_API_BASE


def _github_headers(github_token: str) -> dict:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"
    return headers


def _fetch_json(url: str, params=None, github_token: str = "") -> dict:
    """GET a JSON endpoint with exponential back-off for transient errors."""
    last_exc: Exception = RuntimeError(f"Failed after 5 attempts: {url}")
    for attempt in range(5):
        try:
            resp = requests.get(
                url, headers=_github_headers(github_token), params=params, timeout=30
            )
            if resp.status_code in (429, 500, 502, 503):
                wait = int(resp.headers.get("Retry-After", 2 ** attempt))
                logger.debug(
                    "HTTP %s — retrying in %ds (attempt %d/5): %s",
                    resp.status_code, wait, attempt + 1, url,
                )
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.Timeout as exc:
            last_exc = exc
            logger.debug("Timeout (attempt %d/5): %s", attempt + 1, url)
            time.sleep(2 ** attempt)
    raise last_exc


async def _fetch_commit_async(
    client: httpx.AsyncClient,
    repo_full_name: str,
    sha: str,
    semaphore: asyncio.Semaphore,
) -> tuple[str, dict, list]:
    async with semaphore:
        try:
            url = f"{GITHUB_API_BASE}/repos/{repo_full_name}/commits/{sha}"
            resp = await client.get(url, timeout=30)
            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", 60))
                await asyncio.sleep(retry_after)
                resp = await client.get(url, timeout=30)
            resp.raise_for_status()
            payload = resp.json()
            return sha, payload, payload.get("files", []) or []
        except Exception as exc:
            logger.warning("Async fetch failed for commit %s: %s", sha[:8], exc)
            return sha, {}, []


async def _fetch_all_commits_async(
    repo_full_name: str,
    commit_shas: list,
    github_token: str,
    max_concurrent: int = 8,
) -> dict:
    semaphore = asyncio.Semaphore(max_concurrent)
    headers = _github_headers(github_token)

    async with httpx.AsyncClient(headers=headers) as client:
        tasks = [
            _fetch_commit_async(client, repo_full_name, sha, semaphore)
            for sha in commit_shas
        ]
        results = await asyncio.gather(*tasks)

    return {sha: (payload, files) for sha, payload, files in results}
