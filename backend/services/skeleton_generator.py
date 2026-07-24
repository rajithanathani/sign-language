"""
Sign Language Recognition - Skeleton Generator Service

This module transforms 21 MediaPipe 2D hand landmark keypoints into a synthetic
skeleton image: a crisp white background canvas with green joint connection lines
and red landmark node circles. This matches the exact visual modality of the training dataset.
"""

import sys
from pathlib import Path
from typing import List, Tuple, Optional

# Ensure project root is in sys.path
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import cv2
import numpy as np

try:
    from backend.config import settings
    from backend.utils.constants import (
        HAND_CONNECTIONS,
        WHITE_BGR,
        GREEN_BGR,
        RED_BGR
    )
except ModuleNotFoundError:
    from config import settings
    from utils.constants import (
        HAND_CONNECTIONS,
        WHITE_BGR,
        GREEN_BGR,
        RED_BGR
    )


class SkeletonGeneratorService:
    """
    Service responsible for synthesizing green skeleton drawings on white background canvases.
    """

    def __init__(
        self,
        canvas_size: Tuple[int, int] = (400, 400),
        bg_color: Tuple[int, int, int] = WHITE_BGR,
        line_color: Tuple[int, int, int] = GREEN_BGR,
        node_color: Tuple[int, int, int] = RED_BGR,
        line_thickness: int = settings.LINE_THICKNESS,
        node_radius: int = settings.NODE_RADIUS
    ):
        """
        Initialize skeleton drawing configuration settings.

        Args:
            canvas_size (Tuple[int, int]): Dimensions (width, height) of output image canvas.
            bg_color (Tuple[int, int, int]): BGR color for background canvas (White: 255, 255, 255).
            line_color (Tuple[int, int, int]): BGR color for connection lines (Green: 0, 255, 0).
            node_color (Tuple[int, int, int]): BGR color for landmark circles (Red: 0, 0, 255).
            line_thickness (int): Thickness of connection lines in pixels.
            node_radius (int): Radius of landmark keypoint circles in pixels.
        """
        self.canvas_size = canvas_size
        self.bg_color = bg_color
        self.line_color = line_color
        self.node_color = node_color
        self.line_thickness = line_thickness
        self.node_radius = node_radius

    def generate_skeleton_image(
        self,
        landmarks: List[List[int]],
        bbox: Optional[Tuple[int, int, int, int]] = None,
        padding: int = 20
    ) -> np.ndarray:
        """
        Synthesize a skeleton drawing from 21 hand keypoint landmarks.

        Args:
            landmarks (List[List[int]]): List of 21 landmark [x, y, z] coordinate triplets.
            bbox (Optional[Tuple[int, int, int, int]]): Bounding box (x, y, w, h) around detected hand.
            padding (int): Padding margin around bounding box when centering hand.

        Returns:
            np.ndarray: Synthetic skeleton image array of shape (400, 400, 3) in BGR color space.
        """
        # Create solid white canvas
        canvas_h, canvas_w = self.canvas_size
        skeleton_img = np.full((canvas_h, canvas_w, 3), self.bg_color, dtype=np.uint8)

        if not landmarks or len(landmarks) < 21:
            return skeleton_img

        # Extract 2D (x, y) landmark points
        pts = np.array([[lm[0], lm[1]] for lm in landmarks], dtype=np.float32)

        # Compute bounding box if not provided
        if bbox is None or bbox == (0, 0, 0, 0):
            x_min, y_min = np.min(pts, axis=0)
            x_max, y_max = np.max(pts, axis=0)
            w, h = max(1, x_max - x_min), max(1, y_max - y_min)
        else:
            x_min, y_min, w, h = bbox
            x_max, y_max = x_min + w, y_min + h

        # Force square bounding box to preserve exact 1:1 aspect ratio
        max_dim = max(w, h) + 2 * padding
        center_x = x_min + w / 2.0
        center_y = y_min + h / 2.0

        crop_x1 = center_x - max_dim / 2.0
        crop_y1 = center_y - max_dim / 2.0

        scale = (canvas_w - 60) / max_dim
        offset_x = (canvas_w - max_dim * scale) / 2.0
        offset_y = (canvas_h - max_dim * scale) / 2.0

        # Transform landmarks to centered canvas coordinate space
        canvas_pts = []
        for x, y in pts:
            nx = int((x - crop_x1) * scale + offset_x)
            ny = int((y - crop_y1) * scale + offset_y)
            canvas_pts.append((nx, ny))

        # 1. Draw connection lines (Green)
        for start_idx, end_idx in HAND_CONNECTIONS:
            pt1 = canvas_pts[start_idx]
            pt2 = canvas_pts[end_idx]
            cv2.line(skeleton_img, pt1, pt2, self.line_color, self.line_thickness, cv2.LINE_AA)

        # 2. Draw joint node circles (Red)
        for pt in canvas_pts:
            cv2.circle(skeleton_img, pt, self.node_radius, self.node_color, -1, cv2.LINE_AA)

        return skeleton_img


# Singleton service instance
skeleton_generator_service = SkeletonGeneratorService()


if __name__ == "__main__":
    # Test skeleton synthesis with dummy 21 landmarks
    dummy_lms = [[100 + i * 5, 100 + i * 10, 0] for i in range(21)]
    skeleton = skeleton_generator_service.generate_skeleton_image(dummy_lms)
    
    assert skeleton.shape == (400, 400, 3), "Skeleton output shape mismatch"
    assert np.array_equal(skeleton[0, 0], [255, 255, 255]), "Canvas background should be white"
    print("Skeleton generator service verified successfully!")
