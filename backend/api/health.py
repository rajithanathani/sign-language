"""
Sign Language Recognition - Health Check API Router

This module defines health check endpoints for verifying API service uptime,
FastAPI status, and TensorFlow model readiness.
"""

import sys
from pathlib import Path
from typing import Dict, Any

# Ensure project root is in sys.path
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from fastapi import APIRouter, status

try:
    from backend.config import settings
    from backend.services.predictor import predictor_service
except ModuleNotFoundError:
    from config import settings
    from services.predictor import predictor_service

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    status_code=status.HTTP_200_OK,
    response_model=Dict[str, Any],
    summary="System Health & Model Readiness Probe"
)
async def health_check() -> Dict[str, Any]:
    """
    Check operational status of FastAPI backend server and neural network model engine.

    Returns:
        JSON response with health status, version, and model initialization boolean flag.
    """
    model_ready = predictor_service.is_model_loaded()
    return {
        "status": "healthy",
        "app_name": settings.APP_NAME,
        "version": settings.VERSION,
        "model_loaded": model_ready,
        "message": "Sign Language Recognition API is operational."
    }
