# =============================================================================
# backend/main.py — GraphRAG FastAPI Entry Point
# =============================================================================
"""
Lightweight entry point for the GraphRAG backend REST API.
This file initializes the FastAPI app instance, configures CORS middleware,
and registers all API routes defined in the modular `app/` package.

To start the server locally:
    cd backend
    uvicorn main:app --reload
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import ALLOWED_ORIGINS, logger
from app.routes import api_router

app = FastAPI(title="GraphRAG API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)

logger.info("GraphRAG FastAPI application initialized (modular architecture).")
