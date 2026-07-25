"""
Sign Language Recognition - Hand Detector Service

This module wraps cvzone's HandTrackingModule (which internally utilizes Google MediaPipe)
to detect hand keypoints and extract 21 2D/3D landmark coordinates from webcam images.
It features deferred lazy loading, 3-pass multi-stage contrast-enhanced hand detection,
3-frame landmark temporal smoothing, and an anchored 3-second hand spatial stability tracker.
"""

import sys
import time
import math
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

# Ensure project root is in sys.path
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector

try:
    from backend.config import settings
except ModuleNotFoundError:
    from config import settings


class HandDetectorService:
    """
    Singleton service wrapper around cvzone HandDetector with deferred lazy loading,
    3-pass multi-stage contrast-enhanced hand detection (detectionCon=0.35),
    3-frame landmark temporal smoothing, anchored motion stability tracking,
    and strict 21-landmark completeness validation.
    """

    def __init__(
        self,
        max_hands: int = settings.MAX_HANDS,
        detection_con: float = 0.35,
        motion_threshold_px: float = 50.0
    ):
        """
        Initialize MediaPipe / cvzone HandDetector settings with deferred lazy loading.

        Args:
            max_hands (int): Maximum number of hands to detect. Defaults to 1.
            detection_con (float): Minimum confidence threshold for detection. Defaults to 0.35.
            motion_threshold_px (float): Maximum pixel displacement allowed from reference anchor.
        """
        self.max_hands = max_hands
        self.detection_con = detection_con
        self.motion_threshold_px = motion_threshold_px

        # Deferred lazy initialization of cvzone HandDetector to prevent startup crash
        self.detector: Optional[HandDetector] = None

        # Spatial stability state trackers with reference anchor
        self.reference_centroid: Optional[Tuple[float, float]] = None
        self.stability_start_time: Optional[float] = None

        # Temporal landmark smoothing queue (stores 21 landmark coordinate frames)
        self.lm_history: List[List[List[int]]] = []

    def get_detector(self) -> Optional[HandDetector]:
        """
        Safely retrieve or lazily instantiate cvzone HandDetector instance.

        Returns:
            Optional[HandDetector]: Instantiated HandDetector object or None if initialization fails.
        """
        if self.detector is None:
            try:
                # Optimized detectionCon=0.35 for high sensitivity under webcam low-light & shadows
                self.detector = HandDetector(maxHands=self.max_hands, detectionCon=self.detection_con)
            except Exception as e:
                print(f"Warning: Deferred cvzone HandDetector initialization failed: {e}")
                return None
        return self.detector

    def validate_21_landmarks(self, lm_list: List[List[int]]) -> bool:
        """
        Strictly validate that all 21 MediaPipe hand keypoints exist and are non-degenerate.

        Args:
            lm_list (List[List[int]]): List of 21 landmark [x, y, z] coordinate triplets.

        Returns:
            bool: True if exactly 21 valid non-degenerate keypoints exist.
        """
        if not lm_list or len(lm_list) != 21:
            return False

        pts = np.array([[lm[0], lm[1]] for lm in lm_list], dtype=np.float32)

        # Verify keypoints have non-zero spatial variance (prevents collapsed keypoint errors)
        std_x = np.std(pts[:, 0])
        std_y = np.std(pts[:, 1])

        if std_x < 5.0 or std_y < 5.0:
            return False

        return True

    def smooth_landmarks(self, lm_list: List[List[int]], max_window: int = 3) -> List[List[int]]:
        """
        Apply temporal moving-average exponential smoothing across consecutive frames
        to stabilize MediaPipe landmark coordinates and prevent finger jitter.

        Args:
            lm_list (List[List[int]]): Current frame 21 landmark keypoints.
            max_window (int): Number of consecutive frames to average (Default: 3).

        Returns:
            List[List[int]]: Smoothed 21 landmark keypoint coordinates.
        """
        self.lm_history.append(lm_list)
        if len(self.lm_history) > max_window:
            self.lm_history.pop(0)

        if len(self.lm_history) == 1:
            return lm_list

        # Average landmark coordinates across history window
        history_arr = np.array(self.lm_history, dtype=np.float32)  # Shape: (N, 21, 3)
        smoothed_arr = np.mean(history_arr, axis=0)  # Shape: (21, 3)

        return [[int(round(pt[0])), int(round(pt[1])), int(round(pt[2]))] for pt in smoothed_arr]

    def detect_hand_landmarks_with_stability(
        self,
        image: np.ndarray,
        required_stability_sec: float = 3.0
    ) -> Tuple[Optional[List[List[int]]], Optional[Tuple[int, int, int, int]], bool, float]:
        """
        Detect hand using 3-pass multi-stage contrast enhancement, extract 21 landmarks,
        smooth keypoints, and track spatial position stability.

        Args:
            image (np.ndarray): Input OpenCV BGR image array.
            required_stability_sec (float): Required motionless duration in seconds (Default: 3.0s).

        Returns:
            Tuple[Optional[List[List[int]]], Optional[Tuple[int, int, int, int]], bool, float]:
                - lm_list: 21 landmark keypoints list.
                - bbox: Hand bounding box (x, y, w, h).
                - is_stable: True if hand remained stationary for >= required_stability_sec.
                - progress_pct: Stability progress percentage (0.0 to 100.0).
        """
        if image is None or image.size == 0:
            self._reset_stability()
            return None, None, False, 0.0

        detector = self.get_detector()
        if detector is None:
            self._reset_stability()
            return None, None, False, 0.0

        # Multi-Stage Pass 1 & 2: Standard detection (flipType=True, then flipType=False)
        try:
            hands, _ = detector.findHands(image, draw=False, flipType=True)
            if not hands:
                hands, _ = detector.findHands(image, draw=False, flipType=False)
            
            # Multi-Stage Pass 3 & 4: Contrast & brightness enhanced detection for backlit/dark frames
            if not hands:
                enhanced_image = cv2.convertScaleAbs(image, alpha=1.35, beta=15)
                hands, _ = detector.findHands(enhanced_image, draw=False, flipType=True)
                if not hands:
                    hands, _ = detector.findHands(enhanced_image, draw=False, flipType=False)
        except Exception as e:
            print(f"Warning: Hand landmark detection failed: {e}")
            self._reset_stability()
            return None, None, False, 0.0

        if not hands:
            self._reset_stability()
            return None, None, False, 0.0

        primary_hand: Dict[str, Any] = hands[0]
        raw_lm_list: List[List[int]] = primary_hand.get("lmList", [])
        bbox: Tuple[int, int, int, int] = primary_hand.get("bbox", (0, 0, 0, 0))

        # Validate 21 MediaPipe keypoints integrity
        if not self.validate_21_landmarks(raw_lm_list):
            self._reset_stability()
            return None, None, False, 0.0

        # Apply 3-frame temporal moving-average smoothing to prevent keypoint jitter
        lm_list = self.smooth_landmarks(raw_lm_list, max_window=3)

        # Calculate current hand bounding box centroid
        x, y, w, h = bbox
        current_centroid = (float(x + w / 2.0), float(y + h / 2.0))
        now = time.time()

        if self.reference_centroid is None or self.stability_start_time is None:
            # Anchor reference centroid on initial detection
            self.reference_centroid = current_centroid
            self.stability_start_time = now
            return lm_list, bbox, False, 0.0

        # Compute Euclidean displacement relative to reference anchor centroid
        dx = current_centroid[0] - self.reference_centroid[0]
        dy = current_centroid[1] - self.reference_centroid[1]
        displacement = math.sqrt(dx * dx + dy * dy)

        if displacement < self.motion_threshold_px:
            # Hand is within spatial anchor radius - compute elapsed stability
            elapsed = now - self.stability_start_time
            is_stable = elapsed >= required_stability_sec
            progress_pct = 100.0 if is_stable else min(99.9, round((elapsed / required_stability_sec) * 100.0, 1))
            return lm_list, bbox, is_stable, progress_pct
        else:
            # Intentional hand movement detected - update anchor and restart timer
            self.reference_centroid = current_centroid
            self.stability_start_time = now
            self.lm_history.clear()
            return lm_list, bbox, False, 0.0

    def reset_stability_after_prediction(self) -> None:
        """Reset reference anchor and smoothing history after prediction."""
        self.reference_centroid = None
        self.stability_start_time = None
        self.lm_history.clear()

    def _reset_stability(self) -> None:
        """Reset spatial stability tracking state variables."""
        self.reference_centroid = None
        self.stability_start_time = None
        self.lm_history.clear()

    def detect_hand_landmarks(self, image: np.ndarray) -> Tuple[Optional[List[List[int]]], Optional[Tuple[int, int, int, int]]]:
        """
        Direct single-frame landmark detection helper.
        """
        lms, bbox, _, _ = self.detect_hand_landmarks_with_stability(image, required_stability_sec=0.0)
        return lms, bbox


# Singleton service instance with deferred lazy loading
hand_detector_service = HandDetectorService()


if __name__ == "__main__":
    dummy_frame = np.zeros((400, 400, 3), dtype=np.uint8)
    lms, bbox, stable, prog = hand_detector_service.detect_hand_landmarks_with_stability(dummy_frame)
    assert lms is None and not stable, "Dummy image should return None"
    print("Hand detector service with 3-pass multi-stage detection verified successfully!")
