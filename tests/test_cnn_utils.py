"""CNN numerical helper checks using the original course image data."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

MODULE_PATH = Path(__file__).resolve().parents[1] / "E - nn_intro" / "part2" / "cnn_utils.py"
SPEC = importlib.util.spec_from_file_location(name="cnn_utils", location=MODULE_PATH)
cnn_utils = importlib.util.module_from_spec(spec=SPEC)
SPEC.loader.exec_module(module=cnn_utils)


@pytest.fixture(scope="module")
def face_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read real supplied data once for this module."""
    return cnn_utils.load_face_data()


@pytest.fixture()
def image(face_data: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    """Use a 32-pixel grid sampled from a real course photograph."""
    return face_data[0][0, ::7, ::7].astype(dtype=np.float32) / 255.0


@pytest.mark.parametrize(argnames="square_size", argvalues=[1, 3, 4, 32])
def test_occlusion_pixels(*, image: np.ndarray, square_size: int) -> None:
    """Independently compare several clipped squares to direct image slices."""
    original = image.copy()
    variations = cnn_utils.get_blacked_variations(image=image, square_size=square_size)
    assert variations.shape == (1024, 32, 32, 3)
    assert variations.dtype == np.float32
    for center_row, center_column in ((0, 0), (31, 31), (0, 15), (15, 16)):
        expected = image.copy()
        row_start = center_row - square_size // 2
        column_start = center_column - square_size // 2
        expected[max(0, row_start):min(32, row_start + square_size), max(0, column_start):min(32, column_start + square_size)] = 0.0
        np.testing.assert_array_equal(actual=variations[center_row * 32 + center_column], desired=expected)
    np.testing.assert_array_equal(actual=image, desired=original)


@pytest.mark.parametrize(argnames="square_size", argvalues=[0, 33, -1, 2.5, True])
def test_invalid_square(*, image: np.ndarray, square_size: object) -> None:
    with pytest.raises(expected_exception=ValueError):
        cnn_utils.get_blacked_variations(image=image, square_size=square_size)


def test_focus_ties_and_color(*, image: np.ndarray) -> None:
    """Flat heatmaps still select exactly 80 locations; only red may change."""
    original = image.copy()
    marked, mask = cnn_utils.mark_focus(image=image, heatmap=np.full(shape=(32, 32), fill_value=0.5))
    assert mask.sum() == 80
    assert mask.reshape(-1)[:80].all()
    assert not mask.reshape(-1)[80:].any()
    np.testing.assert_array_equal(actual=marked[..., 1:], desired=image[..., 1:])
    np.testing.assert_allclose(actual=marked[..., 0][mask], desired=np.minimum(image[..., 0][mask] + 0.5, 1.0))
    np.testing.assert_array_equal(actual=image, desired=original)


def test_preprocessing(*, face_data: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    image = face_data[0][0]
    original = image.copy()
    processed = cnn_utils.preprocess_input(x=image, version=1)
    assert processed.dtype == np.float32
    for output_channel, source_channel, mean in ((0, 2, 93.5940), (1, 1, 104.7624), (2, 0, 129.1863)):
        np.testing.assert_allclose(actual=processed[..., output_channel], desired=image[..., source_channel].astype(dtype=np.float32) - mean, atol=1e-5)
    np.testing.assert_array_equal(actual=image, desired=original)


def test_metrics_with_real_labels(*, face_data: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    """Check known prior probabilities on the actual imbalanced class labels."""
    class_names, targets = np.unique(ar=face_data[1], return_inverse=True)
    counts = np.unique(ar=targets, return_counts=True)[1]
    probabilities = np.broadcast_to(array=counts / targets.size, shape=(targets.size, 3)).copy()
    result = cnn_utils.evaluate_probabilities(model_name="class-prior baseline", partition="test", targets=targets, probabilities=probabilities, class_names=class_names.tolist())
    np.testing.assert_allclose(actual=result["accuracy"], desired=counts.max() / targets.size)
    np.testing.assert_allclose(actual=result["balanced_accuracy"], desired=1 / 3)
    np.testing.assert_allclose(actual=result["macro_auc"], desired=0.5)
    assert np.asarray(a=result["confusion_matrix"]).sum() == targets.size


def test_vggface_pretrained(*, face_data: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    """Load all original weights and check frozen features and face prediction."""
    model = cnn_utils.build_vggface()
    assert model.count_params() == 145_002_878
    assert model.get_layer(name="fc6").output.shape[-1] == 4096
    assert model.get_layer(name="fc6_relu").output.shape[-1] == 4096
    processed = cnn_utils.preprocess_input(x=face_data[2][None, ...])
    probabilities = model.predict(x=processed, verbose=0)
    assert probabilities.shape == (1, 2622)
    np.testing.assert_allclose(actual=probabilities.sum(axis=1), desired=1.0, atol=1e-5)
    assert len(cnn_utils.decode_predictions(preds=probabilities, top=5)[0]) == 5
