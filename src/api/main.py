"""Telecom GenAI Platform - Main API Application"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app

from src.api.routes import router
from src.monitoring.metrics import REQUEST_COUNT, startup_metrics
from src.monitoring.logging import setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown."""
    setup_logging()
    startup_metrics()
    logging.info("Telecom GenAI Platform starting up")
    yield
    logging.info("Telecom GenAI Platform shutting down")


app = FastAPI(
    title="Telecom GenAI Platform",
    description="Enterprise GenAI integration for telecom automation",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Prometheus metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Include API routes
app.include_router(router, prefix="/api/v1")


@app.get("/health")
async def health_check():
    """System health check endpoint."""
    return {
        "status": "healthy",
        "service": "telecom-genai-platform",
        "version": "0.1.0",
    }
