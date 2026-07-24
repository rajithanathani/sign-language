"""
Sign Language Recognition - Model Evaluation Module

This script evaluates a trained model artifact (.keras) against the hold-out test set:
1. Calculates overall Test Loss and Test Accuracy.
2. Computes per-class Precision, Recall, and F1-Score via Scikit-Learn's classification report.
3. Generates and renders a 26x26 Confusion Matrix heatmap saved as `training/plots/confusion_matrix.png`.
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

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

try:
    from training.preprocess_dataset import load_dataset_splits, BATCH_SIZE
    from training.labels import CLASS_LABELS
except ModuleNotFoundError:
    from preprocess_dataset import load_dataset_splits, BATCH_SIZE
    from labels import CLASS_LABELS


def parse_args():
    """Parse command line arguments for evaluation script."""
    parser = argparse.ArgumentParser(description="Evaluate Sign Language Recognition CNN Model")
    parser.add_argument("--model-path", type=str, default=os.path.join(project_root, "models", "best_model.keras"), help="Path to trained .keras model")
    parser.add_argument("--dataset-dir", type=str, default=os.path.join(project_root, "dataset"), help="Path to dataset directory")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size for evaluation")
    return parser.parse_args()


def plot_and_save_confusion_matrix(
    cm: np.ndarray,
    labels: list,
    save_path: str
) -> None:
    """
    Render and save a clear confusion matrix heatmap for 26 alphabet classes.

    Args:
        cm (np.ndarray): 26x26 Confusion matrix array.
        labels (list): List of class string names ('A'-'Z').
        save_path (str): File path to save output heatmap image.
    """
    fig, ax = plt.subplots(figsize=(14, 12))
    cax = ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.85)
    fig.colorbar(cax)

    # Set labels
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)

    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.title('Sign Language Recognition - Confusion Matrix', fontsize=14, fontweight='bold', pad=20)

    # Annotate values inside matrix cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            if val > 0:
                color = "white" if val > thresh else "black"
                ax.text(j, i, str(val), ha="center", va="center", color=color, fontsize=7)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Confusion matrix plot saved to: {save_path}")


def evaluate_model_pipeline(
    model_path: str,
    dataset_dir: str,
    batch_size: int = BATCH_SIZE
) -> Tuple[float, float]:
    """
    Execute full evaluation suite against hold-out test dataset.

    Returns:
        Tuple[float, float]: (test_loss, test_accuracy)
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model artifact not found at: {model_path}. Train model first!")

    print("==================================================")
    print("STARTING MODEL EVALUATION SUITE")
    print("==================================================")
    print(f"Loading Model: {model_path}")
    model = tf.keras.models.load_model(model_path)

    # Load test dataset
    _, _, test_ds = load_dataset_splits(dataset_dir, batch_size=batch_size)

    # Evaluate loss and accuracy
    test_loss, test_acc = model.evaluate(test_ds, verbose=1)
    print(f"\n---> TEST ACCURACY: {test_acc * 100:.2f}%")
    print(f"---> TEST LOSS:     {test_loss:.4f}\n")

    # Collect predictions and true targets
    y_true_list = []
    y_pred_list = []

    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true_batch = np.argmax(labels.numpy(), axis=1)
        y_pred_batch = np.argmax(preds, axis=1)

        y_true_list.extend(y_true_batch)
        y_pred_list.extend(y_pred_batch)

    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)

    # Generate classification report
    print("==================================================")
    print("CLASSIFICATION REPORT")
    print("==================================================")
    report = classification_report(y_true, y_pred, target_names=CLASS_LABELS, digits=4)
    print(report)

    # Save classification report to text file
    reports_dir = os.path.join(project_root, "training", "plots")
    os.makedirs(reports_dir, exist_ok=True)
    report_file_path = os.path.join(reports_dir, "classification_report.txt")
    with open(report_file_path, "w") as f:
        f.write(report)
    print(f"Classification report saved to: {report_file_path}")

    # Generate & plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plot_path = os.path.join(reports_dir, "confusion_matrix.png")
    plot_and_save_confusion_matrix(cm, CLASS_LABELS, plot_path)

    return test_loss, test_acc


if __name__ == "__main__":
    args = parse_args()
    evaluate_model_pipeline(
        model_path=args.model_path,
        dataset_dir=args.dataset_dir,
        batch_size=args.batch_size
    )
