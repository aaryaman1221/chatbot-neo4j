# maintenance/

One-off scripts that repair or upgrade an **existing** Neo4j graph in place —
you don't need them for a fresh ingest (`backend_ingest.py` handles that). Run
them from the repo root with the same `.env` the ingest pipeline uses:

```bash
python maintenance/<script>.py [--dry-run]
```

| Script | When to run it |
|---|---|
| `enrich_semantic_edges.py` | After ingesting repos — adds the high-level semantic relationship layers (`DECLARES_METHOD`, `IMPLEMENTS`, `WRAPS`, …) and cross-repo boundaries. |
| `migrate_calls_edges.py` | Repos ingested before the call-edge fixes — cleans ambiguous `CALLS` edges and establishes cross-repo symbol links without re-ingesting. |
| `patch_function_names.py` | Go function names look truncated (an old UTF-8 byte-slicing bug) — restores `name`/`id`/`code` from a fresh GitHub tarball, keeping embeddings and blame edges. |
| `backfill_embeddings.py` | Some nodes are missing vector embeddings because an earlier run timed out — fills only the gaps. |

They need the full dependency set (`requirements.txt`), not `backend/requirements-api.txt`.
