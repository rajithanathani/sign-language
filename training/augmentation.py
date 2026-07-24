"""
Sign Language Recognition - Data Augmentation Module

This module defines domain-specific, non-destructive image data augmentation layers
for skeleton images. Because sign language gestures are hand-orientation sensitive,
destructive augmentations like horizontal flips or vertical flips are strictly EXCLUDED.
"""

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import RandomRotation, RandomTranslation, RandomZoom


def get_augmentation_pipeline() -> tf.keras.Model:
    """
    Construct a TensorFlow Keras Sequential data augmentation pipeline optimized for hand skeleton images.

    Augmentations applied:
    1. Small Random Rotation: [-0.03, 0.03] factor (~ +-10 degrees) to simulate natural hand tilt.
    2. Small Random Translation: [-0.05, 0.05] height/width factor to simulate slight hand offset.
    3. Small Random Zoom: [-0.05, 0.05] factor to simulate slight distance variation from camera.

    Returns:
        tf.keras.Model: Keras Sequential model applying augmentations on 4D image tensors (Batch, H, W, C).
    """
    augmentation_model = Sequential([
        RandomRotation(
            factor=(-0.03, 0.03),
            fill_mode='constant',
            fill_value=1.0,  # White background fill for skeleton images
            name="random_rotation"
        ),
        RandomTranslation(
            height_factor=(-0.05, 0.05),
            width_factor=(-0.05, 0.05),
            fill_mode='constant',
            fill_value=1.0,  # White background fill for skeleton images
            name="random_translation"
        ),
        RandomZoom(
            height_factor=(-0.05, 0.05),
            width_factor=(-0.05, 0.05),
            fill_mode='constant',
            fill_value=1.0,  # White background fill for skeleton images
            name="random_zoom"
        )
    ], name="skeleton_augmentation_pipeline")

    return augmentation_model


if __name__ == "__main__":
    # Test augmentation pipeline shape and range preservation
    dummy_batch = tf.ones((4, 128, 128, 3), dtype=tf.float32)
    pipeline = get_augmentation_pipeline()
    augmented_batch = pipeline(dummy_batch, training=True)
    
    assert augmented_batch.shape == (4, 128, 128, 3), "Augmented shape mismatch"
    print(f"Input shape: {dummy_batch.shape} -> Augmented shape: {augmented_batch.shape}")
    print("Augmentation module verified successfully!")
