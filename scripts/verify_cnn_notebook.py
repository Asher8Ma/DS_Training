"""Check preserved CNN notebook cells, syntax, and (optionally) executed outputs."""

import argparse
import json
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger(name=__name__)


def main() -> None:
    """Verify the notebook without rewriting its JSON or running its models."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executed", action="store_true")
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    notebook_path = ROOT / "E - nn_intro" / "part2" / "CNNs and some more.ipynb"
    notebook = json.loads(s=notebook_path.read_text(encoding="utf-8"))
    baseline = json.loads(s=(ROOT / ".cache" / "cnn" / "original_notebook_cells.json").read_text(encoding="utf-8"))
    current = [{"cell_type": cell["cell_type"], "source": "".join(cell["source"])} for cell in notebook["cells"]]
    next_original = 0
    for cell in current:
        if next_original == len(baseline):
            break
        original = baseline[next_original]
        if cell == original:
            next_original += 1
        elif original["source"].startswith("from keras_vggface import") and cell["source"].startswith("from cnn_utils import build_vggface as VGGFace"):
            assert "from tensorflow.keras.preprocessing.image import ImageDataGenerator" in cell["source"]
            assert cell["cell_type"] == original["cell_type"]
            next_original += 1
    assert next_original == len(baseline), f"Original cell {next_original} was changed, removed, or reordered"
    number_of_code_cells = 0
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        python_source = "\n".join(line for line in source.splitlines() if not line.startswith("%"))
        compile(source=python_source, filename=f"notebook cell {cell.get('id')}", mode="exec")
        if source.strip():
            number_of_code_cells += 1
            if arguments.executed:
                assert cell["execution_count"] is not None, f"Unexecuted cell {cell.get('id')}"
                assert not any(output["output_type"] == "error" for output in cell.get("outputs", [])), f"Error in {cell.get('id')}"
    LOGGER.info("Preserved all %d original cells (one approved import update); compiled %d nonempty code cells", len(baseline), number_of_code_cells)
    if arguments.executed:
        LOGGER.info("All nonempty code cells have execution counts and no error outputs")


if __name__ == "__main__":
    main()
