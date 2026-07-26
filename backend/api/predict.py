"""
Sign Language Recognition - Prediction & Sentence API Router

This module defines FastAPI endpoints for real-time sign language prediction
and sentence management:
1. POST /predict: Decodes webcam frame, detects landmarks, generates skeleton, predicts letter & confidence %.
2. POST /sentence/add: Appends a specific predicted letter to sentence buffer.
3. POST /sentence/space: Appends a space to active sentence.
4. POST /sentence/delete: Removes last character from active sentence.
5. POST /sentence/clear: Resets active sentence string.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Ensure project root and backend directory are in sys.path
project_root = Path(__file__).resolve().parent.parent.parent
backend_dir = Path(__file__).resolve().parent.parent
for p in [str(project_root), str(backend_dir)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from fastapi import APIRouter, UploadFile, File, HTTPException, status
from pydantic import BaseModel, Field

try:
    from backend.utils.image_utils import decode_image_bytes, decode_base64_image, encode_image_to_base64
    from backend.services.hand_detector import hand_detector_service
    from backend.services.skeleton_generator import skeleton_generator_service
    from backend.services.preprocess import preprocess_service
    from backend.services.predictor import predictor_service
    from backend.services.sentence_builder import sentence_builder_service
    from backend.utils.constants import ERR_INVALID_IMAGE, ERR_NO_HAND_DETECTED
    from backend.config import settings
except ModuleNotFoundError:
    from utils.image_utils import decode_image_bytes, decode_base64_image, encode_image_to_base64
    from services.hand_detector import hand_detector_service
    from services.skeleton_generator import skeleton_generator_service
    from services.preprocess import preprocess_service
    from services.predictor import predictor_service
    from services.sentence_builder import sentence_builder_service
    from utils.constants import ERR_INVALID_IMAGE, ERR_NO_HAND_DETECTED
    from config import settings


router = APIRouter(tags=["Prediction"])


class Base64PredictRequest(BaseModel):
    """Schema for Base64 encoded webcam image request."""
    image: str = Field(..., description="Base64 encoded webcam image string (data:image/jpeg;base64,...)")


class AddLetterRequest(BaseModel):
    """Schema for adding a specific letter to sentence buffer."""
    letter: str = Field(..., description="Letter string to append to sentence buffer")


class PredictionResponse(BaseModel):
    """Schema for sign language prediction response."""
    letter: Optional[str] = Field(None, example="A", description="Predicted sign language letter")
    confidence: float = Field(..., example=99.2, description="Prediction confidence percentage")
    sentence: str = Field(..., example="HELLO A", description="Active buffered sentence string")
    skeleton_image: Optional[str] = Field(None, description="Base64 encoded synthesized skeleton image preview")
    hand_detected: bool = Field(..., description="Boolean flag indicating whether a hand was detected")
    hand_stable: bool = Field(False, description="Boolean flag indicating hand remained stable")
    stability_progress: float = Field(0.0, example=80.0, description="Stability timer progress percentage (0-100%)")


@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict Sign Language Letter from Binary Image Upload"
)
async def predict_image_file(file: UploadFile = File(...)) -> PredictionResponse:
    """
    Accept multipart binary webcam frame upload, extract hand landmarks, generate skeleton image,
    execute CNN prediction, and return letter + confidence %.
    """
    try:
        contents = await file.read()
        image = decode_image_bytes(contents)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"{ERR_INVALID_IMAGE} Details: {str(e)}")

    return process_frame_and_predict(image)


@router.post(
    "/predict/base64",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict Sign Language Letter from Base64 Image String"
)
async def predict_base64_image(payload: Base64PredictRequest) -> PredictionResponse:
    """
    Accept Base64 webcam JSON payload, extract hand landmarks, generate skeleton image,
    execute CNN prediction, and return letter + confidence %.
    """
    try:
        image = decode_base64_image(payload.image)
    except Exception as e:
        print(f"[Backend Error] Failed to decode Base64 image: {str(e)}")
        raise HTTPException(status_code=400, detail=f"{ERR_INVALID_IMAGE} Details: {str(e)}")

    return process_frame_and_predict(image)


def process_frame_and_predict(image) -> PredictionResponse:
    """
    Core pipeline handler:
    Webcam Image -> Hand Landmark Detection -> Skeleton Synthesis -> Tensor Preprocessing -> CNN Prediction
    Returns letter and confidence percentage instantly on every frame, buffering when stable.
    """
    if image is None or image.size == 0:
        print("[Backend Warning] Received empty image matrix")

    # 1. Hand Landmark Detection with Logging
    landmarks, bbox, is_stable, progress_pct = hand_detector_service.detect_hand_landmarks_with_stability(
        image, required_stability_sec=1.5
    )

    current_sentence = sentence_builder_service.get_sentence()

    if landmarks is None or len(landmarks) < 21:
        return PredictionResponse(
            letter=None,
            confidence=0.0,
            sentence=current_sentence,
            skeleton_image=None,
            hand_detected=False,
            hand_stable=False,
            stability_progress=0.0
        )

    # 2. Skeleton Image Generation for UI preview
    skeleton_img = skeleton_generator_service.generate_skeleton_image(landmarks, bbox)
    skeleton_b64 = encode_image_to_base64(skeleton_img)

    # 3. Preprocess Tensor for CNN
    tensor = preprocess_service.preprocess_image(skeleton_img)

    # 4. Neural Network Forward Pass Prediction (Instant per-frame execution)
    try:
        predicted_letter, confidence, _ = predictor_service.predict(tensor, landmarks=landmarks)
        print(f"[Backend Prediction] Letter='{predicted_letter}', Confidence={confidence}%, Hand Stable={is_stable}")
    except Exception as e:
        print(f"[Backend Error] Inference Engine Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Inference Engine Error: {str(e)}")

    # 5. Buffer Prediction into Sentence Builder when hand is stable and confidence is high
    if is_stable and confidence >= settings.CONFIDENCE_THRESHOLD:
        sentence_builder_service.add_prediction(predicted_letter, confidence)
        hand_detector_service.reset_stability_after_prediction()

    updated_sentence = sentence_builder_service.get_sentence()

    return PredictionResponse(
        letter=predicted_letter,
        confidence=confidence,
        sentence=updated_sentence,
        skeleton_image=skeleton_b64,
        hand_detected=True,
        hand_stable=is_stable,
        stability_progress=progress_pct
    )


@router.post("/sentence/add", summary="Append Specific Letter to Sentence Buffer")
async def append_letter(payload: AddLetterRequest) -> Dict[str, str]:
    """Append a specific predicted letter to the active sentence buffer."""
    sentence_info = sentence_builder_service.add_prediction(payload.letter, 100.0)
    return {"sentence": sentence_info["sentence"]}


@router.post("/sentence/space", summary="Append Space to Sentence Buffer")
async def append_space() -> Dict[str, str]:
    """Append a space character to the active sentence."""
    updated_sentence = sentence_builder_service.add_space()
    return {"sentence": updated_sentence}


@router.post("/sentence/delete", summary="Delete Last Character from Sentence Buffer")
async def delete_character() -> Dict[str, str]:
    """Delete the last character from the active sentence."""
    updated_sentence = sentence_builder_service.delete_last_character()
    return {"sentence": updated_sentence}


@router.post("/sentence/clear", summary="Clear Active Sentence Buffer")
async def clear_sentence() -> Dict[str, str]:
    """Clear the active sentence buffer."""
    updated_sentence = sentence_builder_service.clear_sentence()
    return {"sentence": updated_sentence}


@router.get("/sentence", summary="Get Active Sentence Buffer")
async def get_sentence() -> Dict[str, str]:
    """Retrieve the current active sentence string."""
    return {"sentence": sentence_builder_service.get_sentence()}
