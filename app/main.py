import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.models.loader import models_registry
from app.api.routes import api_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Loads Mamba, Transformer, and SSDLite detector ONCE at startup.
    Ensures zero per-frame model loading overhead during real-time inference.
    """
    models_registry.load_all_models()
    yield

app = FastAPI(
    title="Drishti AI - Assistive Vision Navigation",
    description="Real-time multimodal navigation and obstacle guidance for the visually impaired.",
    version="1.0.0",
    lifespan=lifespan
)

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

class NoCacheStaticMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response: Response = await call_next(request)
        if request.url.path.endswith((".html", ".css", ".js")) or request.url.path == "/":
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response

# Cross-Origin Resource Sharing
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(NoCacheStaticMiddleware)

# Mount API routers (Health, WebSocket)
app.include_router(api_router)

# Mount Frontend static assets
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)
