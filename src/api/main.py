"""Application factory with explicit runtime ownership and admission controls."""

import asyncio
import secrets
import time
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from src.api.routes import router
from src.monitoring.logging import setup_logging
from src.monitoring.metrics import AUTH_FAILURES
from src.monitoring.tracing import configure_tracing
from src.runtime import Runtime
from src.settings import AppSettings
from src.version import VERSION


def create_app(settings: AppSettings | None = None, runtime: Runtime | None = None):
    if settings is None:
        load_dotenv()
    config = settings or AppSettings.load()
    owned_runtime = runtime or Runtime(config)

    @asynccontextmanager
    async def lifespan(application):
        setup_logging()
        await owned_runtime.start()
        try:
            yield
        finally:
            await owned_runtime.close()

    application = FastAPI(title="Telecom GenAI Platform", version=VERSION, lifespan=lifespan)
    application.state.runtime = owned_runtime
    application.state.settings = config
    application.include_router(router, prefix="/api/v1")
    static_dir = Path(__file__).parent / "static"
    application.mount("/static", StaticFiles(directory=static_dir), name="static")
    application.add_middleware(
        CORSMiddleware, allow_origins=config.cors_origins,
        allow_credentials=False, allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Authorization", "Content-Type"],
    )
    window_start = time.monotonic()
    requests = 0

    @application.middleware("http")
    async def controls(request: Request, call_next):
        nonlocal window_start, requests
        path = request.url.path
        protected = config.app_mode == "live" and path not in ("/health", "/ready")
        if config.app_mode == "demo" and path in (
            "/api/v1/chat", "/api/v1/ingest", "/api/v1/ingest/upload",
        ):
            return JSONResponse({"detail": "Endpoint requires APP_MODE=live"}, status_code=503)
        if config.app_mode == "live" and (path == "/demo" or path.startswith("/api/v1/demo")):
            return JSONResponse({"detail": "Demo is disabled in live mode"}, status_code=404)
        if protected:
            authorization = request.headers.get("authorization", "")
            token = authorization[7:] if authorization.startswith("Bearer ") else ""
            expected = config.admin_token if path.startswith("/api/v1/ingest") else config.api_token
            if not expected.get_secret_value() or not secrets.compare_digest(token.encode(), expected.get_secret_value().encode()):
                AUTH_FAILURES.inc()
                return JSONResponse({"detail": "Authentication required"}, status_code=401)
        if request.method in ("POST", "DELETE"):
            now = time.monotonic()
            if now - window_start >= 60:
                window_start, requests = now, 0
            if requests >= config.requests_per_minute:
                return JSONResponse({"detail": "Request quota exceeded"}, status_code=429, headers={"Retry-After": "60"})
            requests += 1
            content = bytearray()
            async for chunk in request.stream():
                content.extend(chunk)
                if len(content) > config.max_upload_bytes:
                    return JSONResponse({"detail": "Request body too large"}, status_code=413)
            request._body = bytes(content)
        try:
            response = await call_next(request)
        except Exception:
            return JSONResponse({"detail": "Processing unavailable"}, status_code=503)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store"
        return response

    @application.get("/health")
    async def health():
        return {"status": "healthy", "version": VERSION, "mode": config.app_mode}

    @application.get("/ready")
    async def ready():
        try:
            async with asyncio.timeout(7):
                available = await owned_runtime.readiness()
        except Exception:
            available = False
        return JSONResponse({"ready": available, "mode": config.app_mode}, status_code=200 if available else 503)

    @application.get("/demo", response_class=FileResponse)
    async def demo():
        return FileResponse(static_dir / "demo.html")

    @application.get("/metrics")
    async def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    configure_tracing(application)
    return application


app = create_app()
