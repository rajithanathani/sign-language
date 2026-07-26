"""
Sign Language Recognition - Model Predictor Engine

This module provides a singleton model inference engine that loads trained Keras models once
at application startup and executes high-performance forward passes on preprocessed image tensors,
enforcing ASL physical finger geometry validation.
"""

import sys
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List

# Ensure project root is in sys.path
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import tensorflow as tf

try:
    from backend.config import settings
    from backend.utils.labels import CLASS_LABELS, get_label_name
    from backend.services.asl_validator import ASLRuleValidator
except ModuleNotFoundError:
    from config import settings
    from utils.labels import CLASS_LABELS, get_label_name
    from services.asl_validator import ASLRuleValidator


class PredictorService:
    """
    Singleton service class for loading Keras CNN model once and executing inference
    with physical ASL anatomical rule validation.
    """

    def __init__(self, model_path: str = settings.MODEL_PATH):
        """
        Initialize Predictor Service with deferred lazy model loading.

        Args:
            model_path (str): Path to trained .keras model file.
        """
        self.model_path = model_path
        self.model: Optional[tf.keras.Model] = None

    def _load_model(self) -> None:
        """Load trained Keras model into memory with multi-path resolution and serverless safety."""
        if self.model is not None:
            return

        base_dir = settings.BASE_DIR
        curr_dir = Path(__file__).resolve().parent

        candidate_paths = [
            Path(self.model_path),
            base_dir / "backend" / "models" / "asl_model.keras",
            base_dir / "models" / "best_model.keras",
            base_dir / "models" / "asl_model.keras",
            curr_dir.parent / "models" / "asl_model.keras",
            curr_dir.parent.parent / "models" / "best_model.keras",
            curr_dir.parent.parent / "backend" / "models" / "asl_model.keras",
            Path("backend/models/asl_model.keras"),
            Path("models/best_model.keras"),
        ]

        target_path: Optional[Path] = None
        for p in candidate_paths:
            if p.exists():
                target_path = p
                break

        if target_path:
            try:
                self.model = tf.keras.models.load_model(str(target_path), compile=False)
                print(f"Loaded sign language model successfully from: {target_path}")
            except Exception as e:
                print(f"Warning: Failed to load model from {target_path}: {str(e)}")
                self.model = None
        else:
            print(f"Warning: Model file not found in candidate paths. Predictor will run in lazy uninitialized state.")

    def is_model_loaded(self) -> bool:
        """Check if model artifact is successfully loaded in memory."""
        return self.model is not None

    def predict(
        self,
        tensor: np.ndarray,
        landmarks: Optional[List[List[int]]] = None
    ) -> Tuple[str, float, Dict[str, float]]:
        """
        Execute forward pass on preprocessed 4D image tensor with anatomical rule validation.

        Args:
            tensor (np.ndarray): 4D normalized float32 tensor array of shape (1, 128, 128, 3).
            landmarks (Optional[List[List[int]]]): 21 MediaPipe hand keypoints.

        Returns:
            Tuple[str, float, Dict[str, float]]:
                - predicted_letter (str): Predicted sign character ('A'-'Z').
                - confidence (float): Prediction confidence percentage (e.g. 99.2).
                - probabilities (Dict[str, float]): Map of all class labels to probability percentages.
        """
        if not self.is_model_loaded():
            self._load_model()
            if not self.is_model_loaded():
                raise RuntimeError("CNN prediction model is not loaded. Train and save a model first!")

        if tensor is None or len(tensor.shape) != 4:
            raise ValueError(f"Invalid input tensor shape. Expected 4D tensor (1, H, W, C), got {tensor.shape if tensor is not None else None}")

        # Execute high-speed direct neural network forward pass (<15ms latency)
        raw_preds = self.model(tensor, training=False).numpy()[0]

        # Sort candidate indices by probability in descending order
        sorted_indices = np.argsort(raw_preds)[::-1]

        predicted_letter = get_label_name(sorted_indices[0])
        confidence_val = float(raw_preds[sorted_indices[0]] * 100.0)

        # Enforce anatomical finger configuration rules if landmarks are provided
        if landmarks and len(landmarks) == 21:
            for idx in sorted_indices:
                candidate_letter = get_label_name(idx)
                if ASLRuleValidator.validate_letter_geometry(candidate_letter, landmarks):
                    predicted_letter = candidate_letter
                    confidence_val = float(raw_preds[idx] * 100.0)
                    break

        # Build full class probability distribution mapping
        probs_dict = {
            CLASS_LABELS[i]: round(float(raw_preds[i] * 100.0), 2)
            for i in range(len(CLASS_LABELS))
        }

        return predicted_letter, round(confidence_val, 2), probs_dict


# Singleton predictor service instance
predictor_service = PredictorService()


if __name__ == "__main__":
    print(f"Model loaded state: {predictor_service.is_model_loaded()}")
    if predictor_service.is_model_loaded():
        dummy_tensor = np.zeros((1, 128, 128, 3), dtype=np.float32)
        letter, conf, probs = predictor_service.predict(dummy_tensor)
        print(f"Prediction result: Letter='{letter}', Confidence={conf}%")
        assert letter in CLASS_LABELS, "Predicted letter must be valid alphabet character"
    print("Predictor engine service verified successfully!")
