from fastapi import APIRouter
from app.api.health import router as health_router
from app.api.websocket import router as ws_router

api_router = APIRouter()
api_router.include_router(health_router, tags=["Health"])
api_router.include_router(ws_router, tags=["WebSocket"])
