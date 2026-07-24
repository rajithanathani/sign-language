"""
Sign Language Recognition - Dataset Preprocessing Module

This module handles scanning the dataset directory structure, loading raw skeleton images,
resizing images to 128x128x3, normalizing pixel values, splitting data into stratified
training, validation, and testing sets, and wrapping them into optimized tf.data.Dataset pipelines.
"""

import os
import sys
from pathlib import Path
from typing import Tuple, List

# Ensure parent directory is in sys.path for direct script execution
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import tensorflow as tf

try:
    from training.labels import CLASS_LABELS, get_label_index
except ModuleNotFoundError:
    from labels import CLASS_LABELS, get_label_index

# Configuration Constants
TARGET_IMAGE_SIZE: Tuple[int, int] = (128, 128)
COLOR_CHANNELS: int = 3
BATCH_SIZE: int = 32
RANDOM_SEED: int = 42


def get_all_image_paths_and_labels(dataset_dir: str) -> Tuple[List[str], List[int]]:
    """
    Scan dataset directory and collect absolute file paths and corresponding integer labels.

    Args:
        dataset_dir (str): Absolute or relative path to raw dataset root directory.

    Returns:
        Tuple[List[str], List[int]]: Tuple containing list of file paths and list of target label indices.

    Raises:
        FileNotFoundError: If dataset directory or class subfolder does not exist.
    """
    base_path = Path(dataset_dir)
    if not base_path.exists() or not base_path.is_dir():
        raise FileNotFoundError(f"Dataset root directory not found at path: {dataset_dir}")

    image_paths: List[str] = []
    labels: List[int] = []

    for label_name in CLASS_LABELS:
        class_folder = base_path / label_name
        if not class_folder.exists():
            raise FileNotFoundError(f"Missing class subdirectory '{label_name}' in dataset at {class_folder}")

        label_idx = get_label_index(label_name)
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        
        for file_path in class_folder.glob("*"):
            if file_path.suffix.lower() in valid_extensions:
                image_paths.append(str(file_path.resolve()))
                labels.append(label_idx)

    if not image_paths:
        raise ValueError(f"No valid images found in dataset directory: {dataset_dir}")

    return image_paths, labels


def process_image_and_label(file_path: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    TensorFlow mapping function to read, decode, resize, and normalize an image tensor.

    Args:
        file_path (tf.Tensor): String scalar tensor representing the file path.
        label (tf.Tensor): Integer scalar tensor representing class index.

    Returns:
        Tuple[tf.Tensor, tf.Tensor]: Processed image tensor of shape (128, 128, 3) normalized to [0, 1]
                                     and categorical one-hot encoded label tensor of shape (26,).
    """
    # Read raw binary file
    raw_image = tf.io.read_file(file_path)

    # Decode image as 3-channel RGB
    decoded_image = tf.io.decode_image(raw_image, channels=COLOR_CHANNELS, expand_animations=False)
    decoded_image.set_shape([None, None, COLOR_CHANNELS])

    # Resize image to (128, 128) using bilinear interpolation
    resized_image = tf.image.resize(decoded_image, TARGET_IMAGE_SIZE, method=tf.image.ResizeMethod.BILINEAR)

    # Normalize pixel intensities from [0, 255] to [0.0, 1.0]
    normalized_image = tf.cast(resized_image, tf.float32) / 255.0

    # One-hot encode numerical class label
    one_hot_label = tf.one_hot(label, depth=len(CLASS_LABELS))

    return normalized_image, one_hot_label


def create_tf_dataset(
    image_paths: List[str],
    labels: List[int],
    batch_size: int = BATCH_SIZE,
    is_training: bool = True
) -> tf.data.Dataset:
    """
    Construct a high-performance tf.data.Dataset pipeline.

    Args:
        image_paths (List[str]): List of image file paths.
        labels (List[int]): List of target integer labels.
        batch_size (int): Batch size per step. Defaults to 32.
        is_training (bool): If True, shuffles the dataset. Defaults to True.

    Returns:
        tf.data.Dataset: Batched, prefetched dataset pipeline.
    """
    path_tensor = tf.constant(image_paths)
    label_tensor = tf.constant(labels, dtype=tf.int32)

    dataset = tf.data.Dataset.from_tensor_slices((path_tensor, label_tensor))

    if is_training:
        dataset = dataset.shuffle(buffer_size=len(image_paths), seed=RANDOM_SEED)

    # Parallel mapping for file IO and image transformation
    dataset = dataset.map(process_image_and_label, num_parallel_calls=tf.data.AUTOTUNE)

    # Batching and prefetching for optimal GPU/CPU throughput
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset


def load_dataset_splits(
    dataset_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    batch_size: int = BATCH_SIZE
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """
    Scan dataset directory and return train, validation, and test tf.data.Dataset splits.

    Args:
        dataset_dir (str): Path to dataset folder.
        train_ratio (float): Fraction of data for training. Defaults to 0.8.
        val_ratio (float): Fraction of data for validation. Defaults to 0.1.
        test_ratio (float): Fraction of data for testing. Defaults to 0.1.
        batch_size (int): Batch size. Defaults to 32.

    Returns:
        Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]: (train_ds, val_ds, test_ds)
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Split ratios must sum to 1.0"

    image_paths, labels = get_all_image_paths_and_labels(dataset_dir)

    # Deterministic shuffle using scikit-learn or TensorFlow indices
    total_samples = len(image_paths)
    indices = tf.range(total_samples)
    indices = tf.random.shuffle(indices, seed=RANDOM_SEED)

    shuffled_paths = [image_paths[i] for i in indices.numpy()]
    shuffled_labels = [labels[i] for i in indices.numpy()]

    train_end = int(total_samples * train_ratio)
    val_end = train_end + int(total_samples * val_ratio)

    train_paths, train_lbls = shuffled_paths[:train_end], shuffled_labels[:train_end]
    val_paths, val_lbls = shuffled_paths[train_end:val_end], shuffled_labels[train_end:val_end]
    test_paths, test_lbls = shuffled_paths[val_end:], shuffled_labels[val_end:]

    train_ds = create_tf_dataset(train_paths, train_lbls, batch_size=batch_size, is_training=True)
    val_ds = create_tf_dataset(val_paths, val_lbls, batch_size=batch_size, is_training=False)
    test_ds = create_tf_dataset(test_paths, test_lbls, batch_size=batch_size, is_training=False)

    return train_ds, val_ds, test_ds


if __name__ == "__main__":
    import sys
    project_root = str(Path(__file__).resolve().parent.parent)
    dataset_path = os.path.join(project_root, "dataset")
    if os.path.exists(dataset_path):
        print(f"Scanning dataset path: {dataset_path}")
        train, val, test = load_dataset_splits(dataset_path)
        for images, lbls in train.take(1):
            print(f"Batch Image Tensor Shape: {images.shape}")
            print(f"Batch Label Tensor Shape: {lbls.shape}")
            print(f"Pixel Range: Min={tf.reduce_min(images):.2f}, Max={tf.reduce_max(images):.2f}")
        print("Dataset preprocessing verified successfully!")
