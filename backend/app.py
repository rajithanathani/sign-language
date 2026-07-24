"""
Sign Language Recognition - FastAPI Application Entrypoint

This module initializes the FastAPI application instance, configures CORS middleware,
mounts API routers (/api/v1/health and /api/v1/predict), and serves as the server execution entrypoint.
"""

import sys
from pathlib import Path
from typing import Dict, Any

# Ensure project root is in sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

try:
    from backend.config import settings
    from backend.api.health import router as health_router
    from backend.api.predict import router as predict_router
except ModuleNotFoundError:
    from config import settings
    from api.health import router as health_router
    from api.predict import router as predict_router

# Instantiate FastAPI Application
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="Production REST API for Sign Language Alphabet Recognition using Deep Learning CNN and OpenCV Skeleton Synthesis.",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS Middleware for Frontend Access
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API Routers
app.include_router(health_router, prefix=settings.API_V1_STR)
app.include_router(predict_router, prefix=settings.API_V1_STR)


@app.on_event("startup")
async def startup_event():
    """Trigger model loading on startup after Uvicorn binds port."""
    try:
        from backend.services.predictor import predictor_service
        predictor_service._load_model()
    except Exception as e:
        print(f"Startup model load notification: {e}")


@app.get("/", summary="Root Welcome Endpoint")
async def root() -> Dict[str, Any]:
    """
    Root API endpoint returning service metadata and interactive OpenAPI documentation link.
    """
    return {
        "app_name": settings.APP_NAME,
        "version": settings.VERSION,
        "docs_url": "/docs",
        "health_check": f"{settings.API_V1_STR}/health",
        "predict_endpoint": f"{settings.API_V1_STR}/predict",
        "message": "Welcome to the Sign Language Alphabet Recognition API!"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, reload=True)
