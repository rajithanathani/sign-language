"""
Sign Language Recognition - Sentence Builder Service

This module handles real-time letter buffering, prediction debouncing, space insertion,
character deletion, and sentence string assembly for the user interface.
"""

import sys
from pathlib import Path
from typing import Dict, Any

# Ensure project root is in sys.path
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from backend.config import settings
except ModuleNotFoundError:
    from config import settings


class SentenceBuilderService:
    """
    Service for buffering predicted characters, stabilizing real-time predictions,
    and constructing coherent sentences.
    """

    def __init__(
        self,
        confidence_threshold: float = settings.CONFIDENCE_THRESHOLD
    ):
        """
        Initialize sentence builder state and thresholds.

        Args:
            confidence_threshold (float): Minimum confidence percentage to process letter. Defaults to 60.0.
        """
        self.confidence_threshold = confidence_threshold
        self.current_sentence: str = ""
        self.last_predicted_letter: str = ""

    def add_prediction(self, letter: str, confidence: float = 100.0) -> Dict[str, Any]:
        """
        Append a validated letter prediction character to the current sentence buffer.

        Args:
            letter (str): Predicted character label ('A'-'Z').
            confidence (float): Confidence percentage (0.0 - 100.0).

        Returns:
            Dict[str, Any]: Dictionary containing updated sentence string and state flags.
        """
        letter_added = False

        if letter and str(letter).strip() != "":
            clean_letter = str(letter).strip().upper()
            self.current_sentence += clean_letter
            self.last_predicted_letter = clean_letter
            letter_added = True

        return {
            "sentence": self.current_sentence,
            "last_letter": letter,
            "confidence": confidence,
            "letter_added": letter_added
        }

    def add_space(self) -> str:
        """
        Append a space character to the current sentence.

        Returns:
            str: Updated sentence string.
        """
        if self.current_sentence and not self.current_sentence.endswith(" "):
            self.current_sentence += " "
            self.last_predicted_letter = ""
        return self.current_sentence

    def delete_last_character(self) -> str:
        """
        Remove the last character from the current sentence buffer.

        Returns:
            str: Updated sentence string.
        """
        if self.current_sentence:
            self.current_sentence = self.current_sentence[:-1]
            self.last_predicted_letter = ""
        return self.current_sentence

    def clear_sentence(self) -> str:
        """
        Reset and clear the current sentence buffer.

        Returns:
            str: Empty sentence string ("").
        """
        self.current_sentence = ""
        self.last_predicted_letter = ""
        return self.current_sentence

    def get_sentence(self) -> str:
        """
        Retrieve current active sentence string.

        Returns:
            str: Active sentence string.
        """
        return self.current_sentence


# Singleton sentence builder instance
sentence_builder_service = SentenceBuilderService()


if __name__ == "__main__":
    builder = SentenceBuilderService(confidence_threshold=50.0)
    r1 = builder.add_prediction("A", 90.0)
    assert r1["sentence"] == "A", f"Expected sentence 'A', got {r1['sentence']}"

    r2 = builder.add_prediction("B", 92.0)
    assert r2["sentence"] == "AB", f"Expected sentence 'AB', got {r2['sentence']}"

    builder.add_space()
    assert builder.get_sentence() == "AB "

    builder.delete_last_character()
    assert builder.get_sentence() == "AB"

    builder.clear_sentence()
    assert builder.get_sentence() == ""
    print("Sentence builder service verified successfully!")
