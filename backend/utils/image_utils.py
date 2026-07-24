"""
Sign Language Recognition - Image Processing Utilities

This module provides helper functions for decoding image payloads (binary byte streams
and Base64 data strings sent by frontend webcams) into OpenCV BGR image arrays,
and re-encoding image arrays back to Base64 data strings for UI visualization.
"""

import base64
import io
from typing import Tuple

import cv2
import numpy as np
from PIL import Image


def decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    """
    Decode raw binary image bytes into an OpenCV BGR numpy image array.

    Args:
        image_bytes (bytes): Raw binary byte payload from HTTP file upload.

    Returns:
        np.ndarray: OpenCV image array in BGR color space of shape (H, W, 3).

    Raises:
        ValueError: If binary byte payload cannot be decoded into a valid image array.
    """
    if not image_bytes:
        raise ValueError("Empty image byte buffer provided.")

    # Convert binary buffer into numpy 1D array
    nparr = np.frombuffer(image_bytes, np.uint8)

    # Decode array into 3-channel OpenCV image
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        # Fallback decoding via PIL Image
        try:
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception as e:
            raise ValueError(f"Failed to decode image payload: {str(e)}")

    return image


def decode_base64_image(base64_str: str) -> np.ndarray:
    """
    Decode Base64 string payload into an OpenCV BGR numpy image array.

    Args:
        base64_str (str): Base64 encoded string (with or without 'data:image/...;base64,' prefix).

    Returns:
        np.ndarray: OpenCV BGR image array.

    Raises:
        ValueError: If base64 payload is invalid or empty.
    """
    if not base64_str:
        raise ValueError("Empty Base64 image string provided.")

    # Strip data URL prefix if present (e.g. 'data:image/jpeg;base64,...')
    if "," in base64_str:
        base64_str = base64_str.split(",", 1)[1]

    try:
        decoded_bytes = base64.b64decode(base64_str)
        return decode_image_bytes(decoded_bytes)
    except Exception as e:
        raise ValueError(f"Invalid Base64 string payload: {str(e)}")


def encode_image_to_base64(image: np.ndarray, extension: str = ".jpg") -> str:
    """
    Encode an OpenCV image array into a Base64 data string.

    Args:
        image (np.ndarray): OpenCV BGR numpy image array.
        extension (str): Target image format extension. Defaults to ".jpg".

    Returns:
        str: Base64 data URL string ('data:image/jpeg;base64,...').
    """
    success, buffer = cv2.imencode(extension, image)
    if not success:
        raise ValueError("Failed to encode image array to base64.")

    b64_data = base64.b64encode(buffer).decode("utf-8")
    mime_type = "jpeg" if extension in [".jpg", ".jpeg"] else "png"
    return f"data:image/{mime_type};base64,{b64_data}"


if __name__ == "__main__":
    # Test encoding and decoding round-trip
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    b64 = encode_image_to_base64(dummy_img)
    decoded = decode_base64_image(b64)
    assert decoded.shape == (100, 100, 3), "Decoded shape mismatch"
    print("Image utilities module verified successfully!")
