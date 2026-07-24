"""
Sign Language Recognition - Label Definitions Module

This module defines the mapping between integer indices and string class labels
for the 26 English sign language alphabet characters ('A' through 'Z').
"""

from typing import List, Dict


# List of 26 alphabet labels corresponding to dataset folder names
CLASS_LABELS: List[str] = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z'
]

# Total number of output classes
NUM_CLASSES: int = len(CLASS_LABELS)


def get_label_mapping() -> Dict[int, str]:
    """
    Generate a dictionary mapping class integer index to class string label.

    Returns:
        Dict[int, str]: Dictionary mapping 0 -> 'A', 1 -> 'B', ..., 25 -> 'Z'.
    """
    return {index: label for index, label in enumerate(CLASS_LABELS)}


def get_reverse_label_mapping() -> Dict[str, int]:
    """
    Generate a dictionary mapping class string label to class integer index.

    Returns:
        Dict[str, int]: Dictionary mapping 'A' -> 0, 'B' -> 1, ..., 'Z' -> 25.
    """
    return {label: index for index, label in enumerate(CLASS_LABELS)}


def get_label_name(index: int) -> str:
    """
    Retrieve string label corresponding to integer class index.

    Args:
        index (int): Predicted class index (0-25).

    Returns:
        str: Class string label ('A'-'Z').

    Raises:
        ValueError: If index is out of valid range (0-25).
    """
    if not (0 <= index < NUM_CLASSES):
        raise ValueError(f"Invalid class index {index}. Expected index in range [0, {NUM_CLASSES - 1}].")
    return CLASS_LABELS[index]


def get_label_index(label: str) -> int:
    """
    Retrieve integer class index corresponding to string label.

    Args:
        label (str): Class string label ('A'-'Z').

    Returns:
        int: Class integer index (0-25).

    Raises:
        ValueError: If string label is not a valid alphabet character in CLASS_LABELS.
    """
    upper_label = label.upper()
    if upper_label not in CLASS_LABELS:
        raise ValueError(f"Invalid label '{label}'. Must be one of {CLASS_LABELS}.")
    return CLASS_LABELS.index(upper_label)


if __name__ == "__main__":
    print(f"Total Classes: {NUM_CLASSES}")
    print(f"Mapping sample: {get_label_mapping()}")
    assert get_label_name(0) == 'A'
    assert get_label_index('Z') == 25
    print("Label verification tests passed successfully!")
