"""Reusable, tested numerical helpers for the CNN exercise.

The VGGFace architecture/preprocessing follow rcmalli/keras-vggface (MIT):
https://github.com/rcmalli/keras-vggface . Original pretrained weights are used.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
import pickle
from time import perf_counter
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Markdown, display
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, log_loss, roc_auc_score
from tensorflow import keras

LOGGER = logging.getLogger(name="cnn_exercise")
CACHE_DIRECTORY = Path(__file__).resolve().parents[2] / ".cache" / "cnn"


def get_blacked_variations(*, image: np.ndarray, square_size: int = 4) -> np.ndarray:
    """Occlude a clipped square around each pixel of a CIFAR-10 image.

    Args:
        image: Float RGB image with shape (32, 32, 3), in [0, 1].
        square_size: Square edge length, from 1 through 32.

    Returns:
        Independent float32 variations in row-major center order.
    """
    if image.shape != (32, 32, 3) or not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
        LOGGER.error("Invalid occlusion image shape or values")
        raise ValueError("Expected a finite (32,32,3) RGB image in [0,1]")
    if isinstance(square_size, bool) or not isinstance(square_size, int) or not 1 <= square_size <= 32:
        LOGGER.error("Invalid square size: %s", square_size)
        raise ValueError("square_size must be an integer from 1 through 32")
    centers = np.arange(stop=1024)
    row_start = centers // 32 - square_size // 2
    column_start = centers % 32 - square_size // 2
    rows = np.arange(stop=32)[None, :, None]
    columns = np.arange(stop=32)[None, None, :]
    inside = ((rows >= row_start[:, None, None]) & (rows < row_start[:, None, None] + square_size)
              & (columns >= column_start[:, None, None]) & (columns < column_start[:, None, None] + square_size))
    return np.asarray(a=image, dtype=np.float32)[None, ...] * (~inside)[..., None]


def mark_focus(*, image: np.ndarray, heatmap: np.ndarray, number_of_pixels: int = 80) -> tuple[np.ndarray, np.ndarray]:
    """Mark exactly the lowest-probability locations, resolving ties by index.

    Args:
        image: Source RGB image in [0,1].
        heatmap: True-class probability after each occlusion.
        number_of_pixels: Number of locations to mark.

    Returns:
        Independent marked image and boolean location mask.
    """
    if image.shape != (32, 32, 3) or heatmap.shape != (32, 32) or not np.isfinite(heatmap).all():
        LOGGER.error("Invalid focus image or heatmap")
        raise ValueError("Expected image (32,32,3) and finite heatmap (32,32)")
    if isinstance(number_of_pixels, bool) or not isinstance(number_of_pixels, int) or not 1 <= number_of_pixels <= 1024:
        LOGGER.error("Invalid highlighted pixel count")
        raise ValueError("number_of_pixels must be an integer from 1 through 1024")
    chosen = np.argsort(a=heatmap.reshape(-1), kind="stable")[:number_of_pixels]
    mask = np.zeros(shape=1024, dtype=bool)
    mask[chosen] = True
    mask = mask.reshape(32, 32)
    marked = image.copy()
    marked[..., 0] = np.clip(a=marked[..., 0] + 0.5 * mask, a_min=0.0, a_max=1.0)
    return marked, mask


def preprocess_input(*, x: np.ndarray, version: int = 1) -> np.ndarray:
    """Apply the original VGGFace version-1 preprocessing to RGB intensities.

    Args:
        x: RGB intensities in [0,255], with channels last.
        version: Original VGG16/VGGFace preprocessing version (1).

    Returns:
        Independent BGR float32 array with VGGFace means subtracted.
    """
    if version != 1 or x.shape[-1] != 3 or not np.isfinite(x).all() or x.min() < 0 or x.max() > 255:
        LOGGER.error("Invalid VGGFace version or RGB inputs")
        raise ValueError("Expected version=1 and finite RGB intensities in [0,255]")
    return np.asarray(a=x[..., ::-1], dtype=np.float32).copy() - np.array(object=[93.5940, 104.7624, 129.1863], dtype=np.float32)


def load_face_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the original course pickles after archive/provenance validation."""
    arrays = []
    for name in ("images.pkl", "images_labels.pkl", "face_example.pkl"):
        with (CACHE_DIRECTORY / name).open(mode="rb") as input_file:
            arrays.append(np.asarray(a=pickle.load(file=input_file, encoding="latin1")))
    images, labels, example = arrays
    if images.shape[1:] != (224, 224, 3) or images.shape[0] != labels.size or np.unique(ar=labels).size != 3:
        LOGGER.error("Unexpected original face dataset")
        raise ValueError("Expected aligned images and three celebrity classes")
    return images, labels.reshape(-1), example


def build_vggface(*, weights_path: Path = CACHE_DIRECTORY / "vggface_vgg16.h5") -> keras.Model:
    """Reconstruct the original VGGFace network and strictly load face weights.

    Args:
        weights_path: Published full 2,622-class VGGFace HDF5 weights.

    Returns:
        The original pretrained architecture using Keras 3-compatible names.
    """
    # Separate activations retain both fc6 and fc6_relu for the requested SVM study.
    model_input = keras.Input(shape=(224, 224, 3), name="face_rgb")
    activation = model_input
    for block_number, (filters, convolution_count) in enumerate(((64, 2), (128, 2), (256, 3), (512, 3), (512, 3)), start=1):
        for convolution_number in range(1, convolution_count + 1):
            activation = keras.layers.Conv2D(filters=filters, kernel_size=(3, 3), padding="same", activation="relu", name=f"conv{block_number}_{convolution_number}")(inputs=activation)
        activation = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name=f"pool{block_number}")(inputs=activation)
    activation = keras.layers.Flatten(name="flatten")(inputs=activation)
    for name in ("fc6", "fc7"):
        activation = keras.layers.Dense(units=4096, name=name)(inputs=activation)
        activation = keras.layers.Activation(activation="relu", name=f"{name}_relu")(inputs=activation)
    activation = keras.layers.Dense(units=2622, name="fc8")(inputs=activation)
    model_output = keras.layers.Activation(activation="softmax", name="fc8_softmax")(inputs=activation)
    model = keras.Model(inputs=model_input, outputs=model_output, name="vggface_vgg16")
    model.load_weights(filepath=weights_path, skip_mismatch=False)
    if model.count_params() != 145_002_878:
        raise AssertionError("Unexpected VGGFace parameter count")
    LOGGER.info("Loaded all %d VGGFace parameters from %s", model.count_params(), weights_path)
    return model


def decode_predictions(*, preds: np.ndarray, top: int = 5) -> list[list[tuple[str, float]]]:
    """Decode the original VGGFace output classes.

    Args:
        preds: Probabilities with 2,622 columns.
        top: Number of ranked identities to return.

    Returns:
        Ranked identity/probability tuples for each input.
    """
    labels = np.load(file=CACHE_DIRECTORY / "vggface_labels.npy", allow_pickle=True, encoding="latin1")
    if preds.ndim != 2 or preds.shape[1] != 2622 or len(labels) != 2622 or not 1 <= top <= 2622:
        raise ValueError("Expected original VGGFace probabilities and class names")
    return [[(str(labels[class_index]), float(row[class_index])) for class_index in np.argsort(a=-row, kind="stable")[:top]] for row in preds]


def evaluate_probabilities(*, model_name: str, partition: str, targets: np.ndarray, probabilities: np.ndarray, class_names: list[str], predictions: np.ndarray | None = None) -> dict[str, Any]:
    """Evaluate class-aligned probabilities using consistent multiclass metrics.

    Args:
        model_name: Human-readable model label.
        partition: Train, validation, or test.
        targets: Integer class labels.
        probabilities: Columns in the class_names order.
        class_names: Class labels in integer-label order.
        predictions: Optional native classifier decisions, such as SVC.predict.

    Returns:
        Scalar metrics plus the confusion matrix and classification report.
    """
    targets = np.asarray(a=targets).reshape(-1)
    if probabilities.shape != (targets.size, len(class_names)) or not np.isfinite(probabilities).all():
        raise ValueError("Invalid probability dimensions or values")
    np.testing.assert_allclose(actual=probabilities.sum(axis=1), desired=1.0, atol=1e-5)
    if predictions is None:
        predictions = np.argmax(a=probabilities, axis=1)
    class_labels = np.arange(stop=len(class_names))
    matrix = confusion_matrix(y_true=targets, y_pred=predictions, labels=class_labels)
    balanced_accuracy = float(balanced_accuracy_score(y_true=targets, y_pred=predictions))
    np.testing.assert_allclose(actual=balanced_accuracy, desired=np.mean(a=np.diag(v=matrix) / matrix.sum(axis=1)), atol=1e-12)
    return {"model": model_name, "partition": partition, "samples": targets.size,
            "loss": float(log_loss(y_true=targets, y_pred=probabilities, labels=class_labels)),
            "accuracy": float(np.mean(a=predictions == targets)), "balanced_accuracy": balanced_accuracy,
            "macro_auc": float(roc_auc_score(y_true=targets, y_score=probabilities, labels=class_labels, multi_class="ovr", average="macro")),
            "confusion_matrix": matrix.tolist(),
            "classification_report": classification_report(y_true=targets, y_pred=predictions, labels=class_labels, target_names=class_names, output_dict=True, zero_division=0)}


def report_results(*, results: list[dict[str, Any]], class_names: list[str], output_directory: Path, file_name: str) -> None:
    """Display section-ending metrics and save the test confusion matrix.

    Args:
        results: Train/validation/test metric mappings for one model.
        class_names: Class names for the confusion matrix axes.
        output_directory: Current run directory.
        file_name: Confusion matrix PNG name.
    """
    display(pd.DataFrame(data=[{key: result[key] for key in ("model", "partition", "samples", "loss", "accuracy", "balanced_accuracy", "macro_auc")} for result in results]))
    report_lines = [f"{result['partition']} loss: {result['loss']:.6f} | accuracy: {result['accuracy']:.6f} | balanced accuracy: {result['balanced_accuracy']:.6f} | AUC: {result['macro_auc']:.6f}" for result in results]
    display(Markdown(data="```text\n" + "\n".join(report_lines) + "\n```"))
    test_result = next(result for result in results if result["partition"] == "test")
    display(pd.DataFrame(data=test_result["classification_report"]).T)
    matrix = np.asarray(a=test_result["confusion_matrix"])
    figure, axis = plt.subplots(figsize=(8, 6), layout="constrained")
    image = axis.imshow(X=matrix, cmap="Blues")
    axis.set(title=f"{test_result['model']} — test confusion matrix", xlabel="Predicted class", ylabel="True class", xticks=np.arange(stop=len(class_names)), yticks=np.arange(stop=len(class_names)), xticklabels=class_names, yticklabels=class_names)
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right")
    for row_number, column_number in np.ndindex(matrix.shape):
        axis.text(x=column_number, y=row_number, s=str(matrix[row_number, column_number]), ha="center", va="center", color="white" if matrix[row_number, column_number] > matrix.max() / 2 else "black", fontsize=8)
    figure.colorbar(mappable=image, ax=axis, label="Images")
    figure.savefig(fname=output_directory / file_name, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig=figure)


def plot_history(*, history: keras.callbacks.History, title: str, output_directory: Path, file_name: str) -> None:
    """Plot training and validation loss/accuracy with readable labels.

    Args:
        history: Model.fit history.
        title: Model label.
        output_directory: Current run directory.
        file_name: Learning-curve PNG name.
    """
    figure, axes = plt.subplots(nrows=1, ncols=2, figsize=(11, 4), layout="constrained")
    for axis, metric in zip(axes, ("loss", "accuracy"), strict=True):
        for prefix, label in (("", "Training (during fit)"), ("val_", "Validation")):
            values = history.history[prefix + metric]
            axis.plot(np.arange(start=1, stop=len(values) + 1), values, label=label)
        axis.set(title=f"{title}: {metric}", xlabel="Epoch", ylabel=metric)
        axis.grid(visible=True, alpha=0.25)
        axis.legend()
    figure.savefig(fname=output_directory / file_name, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig=figure)


class TrainingLogger(keras.callbacks.Callback):
    """Log progress and optionally enforce the dense model's wall-time budget."""

    def __init__(self, *, label: str, maximum_seconds: float | None = None) -> None:
        super().__init__()
        self.label = label
        self.maximum_seconds = maximum_seconds
        self.started = 0.0

    def on_train_begin(self, logs: dict[str, Any] | None = None) -> None:
        self.started = perf_counter()

    def on_train_batch_end(self, batch: int, logs: dict[str, Any] | None = None) -> None:
        if self.maximum_seconds is not None and perf_counter() - self.started >= self.maximum_seconds:
            self.model.stop_training = True

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        LOGGER.info("%s epoch %d | %s | elapsed %.1fs", self.label, epoch + 1, logs, perf_counter() - self.started)


def weight_digest(*, layers: list[keras.layers.Layer]) -> str:
    """Hash frozen weights without retaining a second copy of the backbone.

    Args:
        layers: Layers whose weights must remain unchanged.

    Returns:
        SHA-256 digest of all supplied weight arrays.
    """
    digest = hashlib.sha256()
    for layer in layers:
        for variable in layer.weights:
            digest.update(variable.numpy().tobytes())
    return digest.hexdigest()
