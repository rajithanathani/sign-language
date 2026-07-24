"""
Sign Language Recognition - Custom CNN Architecture Module

This module defines a deep Convolutional Neural Network (CNN) tailored for
classifying 26 sign language skeleton images. It utilizes 4 Convolutional blocks
with Batch Normalization, Max Pooling, and Dropout regularization to prevent overfitting.
"""

from typing import Tuple

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    BatchNormalization,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)


def build_sign_language_model(
    input_shape: Tuple[int, int, int] = (128, 128, 3),
    num_classes: int = 26
) -> tf.keras.Model:
    """
    Construct and compile the custom CNN architecture for Sign Language classification.

    Architecture Breakdown:
    - Input Layer: (128, 128, 3) image tensor.
    - Block 1: Conv2D (32, 3x3, relu) -> BatchNormalization -> MaxPooling2D (2x2) -> Dropout (0.1)
    - Block 2: Conv2D (64, 3x3, relu) -> BatchNormalization -> MaxPooling2D (2x2) -> Dropout (0.2)
    - Block 3: Conv2D (128, 3x3, relu) -> BatchNormalization -> MaxPooling2D (2x2) -> Dropout (0.3)
    - Block 4: Conv2D (256, 3x3, relu) -> BatchNormalization -> MaxPooling2D (2x2) -> Dropout (0.4)
    - Classification Head: Flatten -> Dense (256, relu) -> BatchNormalization -> Dropout (0.5) -> Dense (26, Softmax)

    Args:
        input_shape (Tuple[int, int, int]): Shape of input image tensor. Defaults to (128, 128, 3).
        num_classes (int): Number of target classification outputs. Defaults to 26.

    Returns:
        tf.keras.Model: Uncompiled Keras Sequential model ready for compilation/training.
    """
    model = Sequential([
        # Input Specifier
        Input(shape=input_shape, name="input_image_layer"),

        # Block 1
        Conv2D(32, (3, 3), padding='same', activation='relu', name="conv1"),
        BatchNormalization(name="batch_norm1"),
        MaxPooling2D(pool_size=(2, 2), name="pool1"),
        Dropout(0.1, name="dropout1"),

        # Block 2
        Conv2D(64, (3, 3), padding='same', activation='relu', name="conv2"),
        BatchNormalization(name="batch_norm2"),
        MaxPooling2D(pool_size=(2, 2), name="pool2"),
        Dropout(0.2, name="dropout2"),

        # Block 3
        Conv2D(128, (3, 3), padding='same', activation='relu', name="conv3"),
        BatchNormalization(name="batch_norm3"),
        MaxPooling2D(pool_size=(2, 2), name="pool3"),
        Dropout(0.3, name="dropout3"),

        # Block 4
        Conv2D(256, (3, 3), padding='same', activation='relu', name="conv4"),
        BatchNormalization(name="batch_norm4"),
        MaxPooling2D(pool_size=(2, 2), name="pool4"),
        Dropout(0.4, name="dropout4"),

        # Fully Connected Dense Head
        Flatten(name="flatten"),
        Dense(256, activation='relu', name="dense_features"),
        BatchNormalization(name="batch_norm_dense"),
        Dropout(0.5, name="dropout_dense"),
        Dense(num_classes, activation='softmax', name="output_probabilities")
    ], name="SignLanguageCNN")

    return model


if __name__ == "__main__":
    model = build_sign_language_model()
    model.summary()
    
    # Test forward pass with dummy tensor
    dummy_input = tf.random.uniform((2, 128, 128, 3))
    output_probs = model(dummy_input)
    assert output_probs.shape == (2, 26), f"Output shape mismatch: expected (2, 26), got {output_probs.shape}"
    print(f"Forward pass successful. Output shape: {output_probs.shape}")
    print("Model architecture verification passed successfully!")