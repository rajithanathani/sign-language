"""
Sign Language Recognition - Model Training Pipeline

This script orchestrates the full model training lifecycle:
1. Resolves dataset paths and loads train/val/test splits.
2. Wraps the input training pipeline with real-time data augmentation.
3. Instantiates the custom CNN architecture.
4. Compiles the model with Adam optimizer and Categorical Crossentropy loss.
5. Attaches production Keras callbacks (ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger).
6. Trains the network and exports model artifacts (.keras) to `models/`.
7. Generates and saves Accuracy & Loss training history graphs to `training/plots/`.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Tuple

# Ensure project root is in sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import tensorflow as tf
import matplotlib.pyplot as plt

try:
    from training.preprocess_dataset import load_dataset_splits, BATCH_SIZE
    from training.model import build_sign_language_model
    from training.augmentation import get_augmentation_pipeline
    from training.labels import NUM_CLASSES
except ModuleNotFoundError:
    from preprocess_dataset import load_dataset_splits, BATCH_SIZE
    from model import build_sign_language_model
    from augmentation import get_augmentation_pipeline
    from labels import NUM_CLASSES


def parse_args():
    """Parse command-line arguments for configurable training hyper-parameters."""
    parser = argparse.ArgumentParser(description="Train Sign Language Recognition CNN Model")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size per training step")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial Adam learning rate")
    parser.add_argument("--dataset-dir", type=str, default=os.path.join(project_root, "dataset"), help="Path to dataset directory")
    parser.add_argument("--output-dir", type=str, default=os.path.join(project_root, "models"), help="Path to save trained model artifacts")
    return parser.parse_args()


def plot_and_save_training_history(history: tf.keras.callbacks.History, save_path: str) -> None:
    """
    Plot and save training & validation Accuracy and Loss curves.

    Args:
        history (tf.keras.callbacks.History): Training history object returned by model.fit().
        save_path (str): File path to save the output PNG plot.
    """
    acc = history.history.get('accuracy', [])
    val_acc = history.history.get('val_accuracy', [])
    loss = history.history.get('loss', [])
    val_loss = history.history.get('val_loss', [])

    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(14, 6))

    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, 'b-o', label='Training Accuracy')
    plt.plot(epochs_range, val_acc, 'r-s', label='Validation Accuracy')
    plt.title('Training & Validation Accuracy', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right')

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, 'b-o', label='Training Loss')
    plt.plot(epochs_range, val_loss, 'r-s', label='Validation Loss')
    plt.title('Training & Validation Loss', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper right')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Training curves saved to: {save_path}")


def train_pipeline(
    dataset_dir: str,
    output_dir: str,
    epochs: int = 30,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = 1e-3
) -> Tuple[tf.keras.Model, tf.keras.callbacks.History]:
    """
    Execute full dataset loading, augmentation binding, model training, and artifact saving pipeline.

    Returns:
        Tuple[tf.keras.Model, tf.keras.callbacks.History]: (trained_model, history_object)
    """
    # 1. Ensure target output directories exist
    os.makedirs(output_dir, exist_ok=True)
    plots_dir = os.path.join(project_root, "training", "plots")
    os.makedirs(plots_dir, exist_ok=True)

    best_model_path = os.path.join(output_dir, "best_model.keras")
    log_csv_path = os.path.join(plots_dir, "training_log.csv")
    plot_png_path = os.path.join(plots_dir, "training_curves.png")

    print("==================================================")
    print("STARTING SIGN LANGUAGE CNN TRAINING PIPELINE")
    print("==================================================")
    print(f"Dataset Directory: {dataset_dir}")
    print(f"Model Output Path: {best_model_path}")
    print(f"Epochs: {epochs} | Batch Size: {batch_size} | Initial LR: {learning_rate}")

    # 2. Load dataset splits
    train_ds, val_ds, test_ds = load_dataset_splits(dataset_dir, batch_size=batch_size)

    # 3. Apply real-time Data Augmentation to training dataset
    augmentation_model = get_augmentation_pipeline()
    augmented_train_ds = train_ds.map(
        lambda x, y: (augmentation_model(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE)

    # 4. Build custom CNN architecture
    model = build_sign_language_model(input_shape=(128, 128, 3), num_classes=NUM_CLASSES)

    # 5. Compile model
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # 6. Configure production callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=best_model_path,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger(
            filename=log_csv_path,
            append=False
        )
    ]

    # 7. Start Training
    history = model.fit(
        augmented_train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )

    # 8. Save final plot curves
    plot_and_save_training_history(history, plot_png_path)

    print("==================================================")
    print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
    print(f"Best Model Artifact: {best_model_path}")
    print("==================================================")

    return model, history


if __name__ == "__main__":
    args = parse_args()
    train_pipeline(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )
