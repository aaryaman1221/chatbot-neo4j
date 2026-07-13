# =============================================================================
# backend/app/routes.py — FastAPI API Router & Endpoint Definitions
# =============================================================================

import os
import traceback
import time as _time
import uuid

from fastapi import APIRouter, Header, HTTPException
from langchain_google_genai import ChatGoogleGenerativeAI

from .config import (
    logger,
    ChatRequest,
    ConnectionRequest,
    get_neo4j_driver,
    _check_neo4j,
    _query_graph_stats,
    _fetch_available_repos,
    get_google_api_key,
)
from .llm_service import answer_question_hybrid

api_router = APIRouter()
SERVER_INSTANCE_ID = str(uuid.uuid4())
logger.info("[STARTUP] Generated SERVER_INSTANCE_ID = %s", SERVER_INSTANCE_ID)


@api_router.get("/api/health")
def health_check():
    """Liveness probe — returns 200 OK when the server is up."""
    return {"status": "ok"}


@api_router.get("/api/stats")
def get_stats(
    x_neo4j_uri: str = Header(...),
    x_neo4j_user: str = Header(...),
    x_neo4j_password: str = Header(...)
):
    try:
        drv = get_neo4j_driver(x_neo4j_uri, x_neo4j_user, x_neo4j_password)
        return _query_graph_stats(drv)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/api/repos")
def get_repos(
    x_neo4j_uri: str = Header(...),
    x_neo4j_user: str = Header(...),
    x_neo4j_password: str = Header(...)
):
    try:
        drv = get_neo4j_driver(x_neo4j_uri, x_neo4j_user, x_neo4j_password)
        return {"repos": _fetch_available_repos(drv)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/api/check-connection")
def check_connection(req: ConnectionRequest):
    ok, msg = _check_neo4j(req.uri, req.user, req.password)
    if not ok:
        raise HTTPException(status_code=400, detail=msg)
    return {"status": "ok", "message": msg}


@api_router.post("/api/chat")
def chat(
    req: ChatRequest,
    x_neo4j_uri: str = Header(...),
    x_neo4j_user: str = Header(...),
    x_neo4j_password: str = Header(...),
):
    google_api_key = get_google_api_key()
    if not google_api_key:
        raise HTTPException(status_code=500, detail="Google/Gemini API key not found. Please set GOOGLE_API_KEY or GEMINI_API_KEY in your .env file or environment.")

    logger.info(
        "[CHAT] ════════════════════════════════════════════════════════"
    )
    logger.info("[CHAT] Incoming query   : %r", req.query[:200])
    logger.info("[CHAT] Selected repos   : %s", req.selected_repos or "<all>")
    logger.info("[CHAT] top_k            : %d", req.top_k)

    _t0 = _time.perf_counter()
    try:
        drv = get_neo4j_driver(x_neo4j_uri, x_neo4j_user, x_neo4j_password)
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=google_api_key, temperature=0.0)

        result = answer_question_hybrid(
            user_input=req.query,
            llm=llm,
            driver=drv,
            google_api_key=google_api_key,
            selected_repos=req.selected_repos,
            top_k=req.top_k,
            chat_history=req.chat_history,
        )

        answer = result["answer"]
        usage = result["usage"]
        needs_clarification = result.get("needs_clarification", False)

        elapsed = _time.perf_counter() - _t0
        logger.info("[CHAT] ✅ Done in %.2fs — answer_chars=%d, clarification=%s", elapsed, len(answer), needs_clarification)
        return {
            "answer": answer,
            "usage": usage,
            "needs_clarification": needs_clarification,
            "instance_id": SERVER_INSTANCE_ID
        }
    except Exception as e:
        elapsed = _time.perf_counter() - _t0
        logger.error("[CHAT] ❌ Failed after %.2fs: %s", elapsed, e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
