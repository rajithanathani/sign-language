"""
Sign Language Recognition - Preprocessing Service

This module prepares synthesized skeleton images for CNN model inference.
It performs color space conversion (BGR to RGB), spatial resizing (128x128),
floating-point pixel normalization ([0.0, 1.0]), and 4D batch dimension expansion (1, 128, 128, 3).
"""

import sys
from pathlib import Path
from typing import Tuple

# Ensure project root is in sys.path
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import cv2
import numpy as np

try:
    from backend.config import settings
except ModuleNotFoundError:
    from config import settings


class PreprocessService:
    """
    Service responsible for converting raw OpenCV image arrays into normalized 4D Keras input tensors.
    """

    def __init__(self, target_size: Tuple[int, int] = settings.IMAGE_SIZE):
        """
        Initialize preprocessing configuration parameters.

        Args:
            target_size (Tuple[int, int]): Target image dimensions (width, height). Defaults to (128, 128).
        """
        self.target_size = target_size

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess a BGR skeleton image array for model input.

        Pipeline Steps:
        1. BGR to RGB color space conversion.
        2. Bilinear interpolation resize to target shape (128, 128).
        3. Normalization of uint8 pixel values [0..255] to float32 [0.0..1.0].
        4. Expansion of batch dimension from (128, 128, 3) to (1, 128, 128, 3).

        Args:
            image (np.ndarray): Input OpenCV skeleton image array in BGR format.

        Returns:
            np.ndarray: 4D normalized float32 tensor array of shape (1, 128, 128, 3).

        Raises:
            ValueError: If input image is empty or invalid.
        """
        if image is None or image.size == 0:
            raise ValueError("Invalid input image array provided for preprocessing.")

        # 1. Convert BGR to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image

        # 2. Resize to target dimension (128, 128)
        resized_image = cv2.resize(rgb_image, self.target_size, interpolation=cv2.INTER_LINEAR)

        # 3. Cast to float32 and normalize pixel range to [0.0, 1.0]
        normalized_tensor = resized_image.astype(np.float32) / 255.0

        # 4. Expand batch dimension -> (1, 128, 128, 3)
        batched_tensor = np.expand_dims(normalized_tensor, axis=0)

        return batched_tensor


# Singleton service instance
preprocess_service = PreprocessService()


if __name__ == "__main__":
    dummy_skeleton = np.full((400, 400, 3), 255, dtype=np.uint8)
    tensor = preprocess_service.preprocess_image(dummy_skeleton)
    
    assert tensor.shape == (1, 128, 128, 3), f"Tensor shape mismatch: expected (1, 128, 128, 3), got {tensor.shape}"
    assert tensor.dtype == np.float32, "Tensor dtype must be float32"
    assert tensor.min() >= 0.0 and tensor.max() <= 1.0, "Pixel normalization out of range [0, 1]"
    print("Preprocessing service verified successfully!")
