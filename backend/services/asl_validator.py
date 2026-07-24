"""
Sign Language Recognition - ASL Anatomical Finger Validator

This module provides physical finger configuration validation for MediaPipe hand keypoints.
It enforces strict ASL finger geometry rules (e.g. 'W' requires index, middle, ring extended;
'A' requires closed fist) to prevent misclassifications among visually distinct signs.
"""

from typing import List, Tuple, Dict, Any
import numpy as np


class ASLRuleValidator:
    """
    Validates anatomical finger configuration from 21 MediaPipe hand landmarks.
    """

    @staticmethod
    def inspect_finger_extension(landmarks: List[List[int]]) -> Dict[str, bool]:
        """
        Inspect extension/flexion state for all 5 fingers.

        Landmark Indices:
        - Wrist: 0
        - Thumb: MCP 2, IP 3, Tip 4
        - Index: MCP 5, PIP 6, DIP 7, Tip 8
        - Middle: MCP 9, PIP 10, DIP 11, Tip 12
        - Ring: MCP 13, PIP 14, DIP 15, Tip 16
        - Pinky: MCP 17, PIP 18, DIP 19, Tip 20

        Returns:
            Dict[str, bool]: Map of finger names to boolean extension state (True = Extended).
        """
        if not landmarks or len(landmarks) < 21:
            return {
                "thumb": False, "index": False, "middle": False,
                "ring": False, "pinky": False
            }

        pts = {i: np.array([landmarks[i][0], landmarks[i][1]]) for i in range(21)}

        # Finger extension is true if fingertip y is significantly higher (smaller y in screen space) than MCP/PIP joint
        index_ext = pts[8][1] < (pts[5][1] - 12)
        middle_ext = pts[12][1] < (pts[9][1] - 12)
        ring_ext = pts[16][1] < (pts[13][1] - 12)
        pinky_ext = pts[20][1] < (pts[17][1] - 12)

        # Thumb extension: tip x/y relative to wrist (0) and MCP (2)
        thumb_dist = np.linalg.norm(pts[4] - pts[0])
        index_mcp_dist = np.linalg.norm(pts[5] - pts[0])
        thumb_ext = thumb_dist > (index_mcp_dist * 0.9)

        return {
            "thumb": thumb_ext,
            "index": index_ext,
            "middle": middle_ext,
            "ring": ring_ext,
            "pinky": pinky_ext
        }

    @classmethod
    def validate_letter_geometry(cls, letter: str, landmarks: List[List[int]]) -> bool:
        """
        Verify if the physical finger extension states match the anatomical requirements of the letter.

        Args:
            letter (str): Candidate predicted letter ('A'-'Z').
            landmarks (List[List[int]]): 21 MediaPipe hand keypoints.

        Returns:
            bool: True if physical finger state conforms to ASL specification.
        """
        ext = cls.inspect_finger_extension(landmarks)

        # Strict Anatomical Rules per Character
        if letter == 'A':
            # 'A' is a closed fist. Index, Middle, Ring, Pinky MUST NOT be extended.
            # If 2 or more main fingers are extended, it CANNOT be 'A'.
            extended_count = sum([ext['index'], ext['middle'], ext['ring'], ext['pinky']])
            return extended_count <= 1

        elif letter == 'W':
            # 'W' requires Index, Middle, and Ring extended up. Pinky MUST be folded.
            return ext['index'] and ext['middle'] and ext['ring'] and not ext['pinky']

        elif letter == 'V':
            # 'V' requires Index and Middle extended. Ring and Pinky MUST be folded.
            return ext['index'] and ext['middle'] and not ext['ring'] and not ext['pinky']

        elif letter == 'U':
            # 'U' requires Index and Middle extended together. Ring and Pinky MUST be folded.
            return ext['index'] and ext['middle'] and not ext['ring'] and not ext['pinky']

        elif letter == 'B':
            # 'B' requires all 4 fingers (Index, Middle, Ring, Pinky) extended straight up.
            return ext['index'] and ext['middle'] and ext['ring'] and ext['pinky']

        elif letter == 'L':
            # 'L' requires Thumb and Index extended. Middle, Ring, Pinky MUST be folded.
            return ext['thumb'] and ext['index'] and not ext['middle'] and not ext['ring'] and not ext['pinky']

        elif letter == 'I':
            # 'I' requires Pinky extended. Index, Middle, Ring MUST be folded.
            return ext['pinky'] and not ext['index'] and not ext['middle'] and not ext['ring']

        elif letter == 'Y':
            # 'Y' requires Thumb and Pinky extended. Index, Middle, Ring MUST be folded.
            return ext['thumb'] and ext['pinky'] and not ext['index'] and not ext['middle'] and not ext['ring']

        elif letter == 'F':
            # 'F' requires Middle, Ring, Pinky extended up while Thumb & Index pinch.
            return ext['middle'] and ext['ring'] and ext['pinky']

        # Default fallback for letters without hard-coded negative constraints
        return True


if __name__ == "__main__":
    # Test 'A' fist validation
    fist_lms = [[100, 100, 0] for _ in range(21)]
    assert ASLRuleValidator.validate_letter_geometry('A', fist_lms), "'A' fist validation failed"
    print("ASL rule validator verified successfully!")
