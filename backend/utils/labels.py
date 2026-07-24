"""
Sign Language Recognition - Backend Label Utility Module

This module defines label mappings and conversion functions for the FastAPI backend services.
"""

from typing import List, Dict

# Complete alphabet class label array corresponding to CNN output probabilities
CLASS_LABELS: List[str] = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z'
]

NUM_CLASSES: int = len(CLASS_LABELS)


def get_label_name(index: int) -> str:
    """
    Map class integer index (0-25) to string character ('A'-'Z').

    Args:
        index (int): Model argmax output index.

    Returns:
        str: Target sign language character.
    """
    if not (0 <= index < NUM_CLASSES):
        raise ValueError(f"Invalid index {index}. Index must be between 0 and {NUM_CLASSES - 1}.")
    return CLASS_LABELS[index]


def get_label_index(label: str) -> int:
    """
    Map character label ('A'-'Z') to class integer index (0-25).

    Args:
        label (str): Character string.

    Returns:
        int: Zero-based class index.
    """
    upper_label = label.upper()
    if upper_label not in CLASS_LABELS:
        raise ValueError(f"Invalid label '{label}'. Label must be one of {CLASS_LABELS}.")
    return CLASS_LABELS.index(upper_label)


if __name__ == "__main__":
    assert get_label_name(0) == 'A'
    assert get_label_index('Z') == 25
    print("Backend labels module verified successfully!")
