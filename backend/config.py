"""
Sign Language Recognition - Backend Configuration Module

This module defines system configuration settings for the FastAPI application,
including server metadata, model file paths, CORS allowed origins, confidence thresholds,
and skeleton rendering constants.
"""

import os
from pathlib import Path
from typing import List, Tuple
from pydantic import BaseModel, Field

# Base Directory of Project Root
BASE_DIR: Path = Path(__file__).resolve().parent.parent

# Load environment variables from .env if present
try:
    from dotenv import load_dotenv
    env_file = BASE_DIR / ".env"
    if env_file.exists():
        load_dotenv(dotenv_path=env_file)
except ImportError:
    pass


class Settings(BaseModel):
    """
    Application Settings Model backed by Pydantic schema validation.
    """
    APP_NAME: str = Field(default_factory=lambda: os.getenv("APP_NAME", "Sign Language Alphabet Recognition API"))
    VERSION: str = Field(default_factory=lambda: os.getenv("VERSION", "1.0.0"))
    API_V1_STR: str = "/api/v1"
    DEBUG: bool = Field(default_factory=lambda: os.getenv("DEBUG", "False").lower() == "true")
    
    # Path to trained Keras model artifact
    MODEL_PATH: str = Field(
        default_factory=lambda: os.getenv("MODEL_PATH", str(BASE_DIR / "models" / "best_model.keras"))
    )
    
    # Image Input Dimension for CNN Model
    IMAGE_SIZE: Tuple[int, int] = (128, 128)
    COLOR_CHANNELS: int = 3
    
    # Minimum prediction confidence percentage to accept character
    CONFIDENCE_THRESHOLD: float = Field(
        default_factory=lambda: float(os.getenv("CONFIDENCE_THRESHOLD", 60.0))
    )
    
    # Hand Detector Parameters (cvzone / MediaPipe)
    MAX_HANDS: int = 1
    DETECTION_CONFIDENCE: float = 0.7
    
    # Skeleton Drawing Parameters
    SKELETON_BG_COLOR: Tuple[int, int, int] = (255, 255, 255)  # White RGB
    SKELETON_LINE_COLOR: Tuple[int, int, int] = (0, 255, 0)     # Green RGB (0, 255, 0)
    SKELETON_NODE_COLOR: Tuple[int, int, int] = (0, 0, 255)     # Red RGB (0, 0, 255)
    LINE_THICKNESS: int = 4
    NODE_RADIUS: int = 5

    # CORS Configuration for Local & Vercel Production Deployment
    CORS_ORIGINS: List[str] = Field(
        default_factory=lambda: [
            "http://localhost:5173",  # Vite Dev Server
            "http://localhost:3000",  # React CRA Server
            "http://127.0.0.1:5173",
            "http://127.0.0.1:3000",
            os.getenv("FRONTEND_URL", "https://sign-language-blond.vercel.app").rstrip("/")
        ]
    )


# Instantiate Singleton Settings Object
settings = Settings()


if __name__ == "__main__":
    print(f"App Name: {settings.APP_NAME}")
    print(f"Model Path: {settings.MODEL_PATH}")
    print(f"Confidence Threshold: {settings.CONFIDENCE_THRESHOLD}")
    print("Configuration module verified successfully!")
