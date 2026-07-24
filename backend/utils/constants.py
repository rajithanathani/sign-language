"""
Sign Language Recognition - Utility Constants Module

This module defines MediaPipe 21 hand landmark connections, drawing constants,
API error messages, and network status codes.
"""

from typing import List, Tuple

# Number of MediaPipe 3D Hand Keypoint Landmarks
NUM_LANDMARKS: int = 21

# Landmark Connections array defining keypoint segment pairs for drawing skeleton lines
# Connections conform to standard MediaPipe hand landmark topology
HAND_CONNECTIONS: List[Tuple[int, int]] = [
    # Palm / Wrist Base Connections
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index Finger
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle Finger
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring Finger
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) # Pinky Finger
]

# Color Palettes (BGR / RGB Format for OpenCV canvas drawing)
WHITE_BGR: Tuple[int, int, int] = (255, 255, 255)
GREEN_BGR: Tuple[int, int, int] = (0, 255, 0)
RED_BGR: Tuple[int, int, int] = (0, 0, 255)
BLACK_BGR: Tuple[int, int, int] = (0, 0, 0)

# API Response Status Strings
STATUS_SUCCESS: str = "success"
STATUS_ERROR: str = "error"
STATUS_NO_HAND_DETECTED: str = "no_hand_detected"

# Error Messages
ERR_INVALID_IMAGE: str = "Failed to decode uploaded image file. Please provide a valid JPEG or PNG image."
ERR_NO_HAND_DETECTED: str = "No hand detected in the frame. Please position your hand clearly in front of the camera."
ERR_MODEL_NOT_LOADED: str = "Prediction model is not initialized or model file is missing."


if __name__ == "__main__":
    print(f"Total Landmarks: {NUM_LANDMARKS}")
    print(f"Total Connections: {len(HAND_CONNECTIONS)}")
    print("Constants module verified successfully!")
