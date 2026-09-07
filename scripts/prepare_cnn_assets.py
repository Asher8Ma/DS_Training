"""Download and validate the original CNN exercise assets without editing notebooks."""

from __future__ import annotations

import hashlib
from html.parser import HTMLParser
import json
import logging
from pathlib import Path
import pickle
import ssl
import zipfile

import numpy as np
import requests

LOGGER = logging.getLogger(name=__name__)
ROOT = Path(__file__).resolve().parents[1]
PART = ROOT / "E - nn_intro" / "part2"
CACHE = ROOT / ".cache" / "cnn"
SOURCES = {
    "faces.zip": "https://drive.google.com/uc?export=download&id=1go4U-nI4H5kBz6hstI4_PYtzS35jHEDM",
    "vggface_vgg16.h5": "https://github.com/rcmalli/keras-vggface/releases/download/v2.0/rcmalli_vggface_tf_vgg16.h5",
    "vggface_labels.npy": "https://github.com/rcmalli/keras-vggface/releases/download/v2.0/rcmalli_vggface_labels_v1.npy",
}


class DownloadForm(HTMLParser):
    """Read Google's public large-file confirmation form."""

    def __init__(self) -> None:
        super().__init__()
        self.action = ""
        self.fields: dict[str, str] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Collect the download form fields required by HTMLParser's interface."""
        attributes = dict(attrs)
        if tag == "form" and attributes.get("id") == "download-form":
            self.action = attributes.get("action") or ""
        if tag == "input" and attributes.get("name"):
            self.fields[attributes["name"]] = attributes.get("value") or ""


def download(*, file_name: str, url: str) -> dict[str, object]:
    """Fetch a binary asset and return its provenance.

    Args:
        file_name: Destination cache filename.
        url: Published source URL.

    Returns:
        Source URL, byte count, and SHA-256 digest.
    """
    target = CACHE / file_name
    if not target.exists():
        if file_name != "faces.zip":
            raise FileNotFoundError("Run scripts/download_cnn_windows.ps1 first to fetch weights using Windows certificate trust")
        LOGGER.info("Downloading %s", url)
        session = requests.Session()
        response = session.get(url=url, stream=True, timeout=120)
        response.raise_for_status()
        if "text/html" in response.headers.get("Content-Type", ""):
            parser = DownloadForm()
            parser.feed(data=response.text)
            if not parser.action.startswith("https://drive.usercontent.google.com/download"):
                raise ValueError(f"Expected Google's download confirmation, received HTML: {url}")
            response.close()
            response = session.get(url=parser.action, params=parser.fields, stream=True, timeout=120)
            response.raise_for_status()
        if "text/html" in response.headers.get("Content-Type", ""):
            raise ValueError(f"Expected binary data, received HTML: {url}")
        temporary = target.with_suffix(target.suffix + ".partial")
        with temporary.open(mode="wb") as output_file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                output_file.write(chunk)
        response.close()
        temporary.replace(target=target)
    with target.open(mode="rb") as input_file:
        digest = hashlib.file_digest(input_file, "sha256").hexdigest()
    LOGGER.info("Validated download %s (%d bytes)", target, target.stat().st_size)
    return {"url": url, "bytes": target.stat().st_size, "sha256": digest}


def main() -> None:
    """Preserve the notebook source and prepare its original dataset/weights."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    CACHE.mkdir(parents=True, exist_ok=True)
    baseline_path = CACHE / "original_notebook_cells.json"
    if not baseline_path.exists():
        notebook = json.loads(s=(PART / "CNNs and some more.ipynb").read_text(encoding="utf-8"))
        baseline_path.write_text(data=json.dumps(obj=[{"cell_type": cell["cell_type"], "source": "".join(cell["source"])} for cell in notebook["cells"]], indent=2), encoding="utf-8")
    manifest = {name: download(file_name=name, url=url) for name, url in SOURCES.items()}
    with zipfile.ZipFile(file=CACHE / "faces.zip") as archive:
        for name in ("images.pkl", "images_labels.pkl", "face_example.pkl"):
            matches = [member for member in archive.namelist() if Path(member).name == name]
            if len(matches) != 1:
                raise ValueError(f"Expected exactly one {name}, found {matches}")
            (CACHE / name).write_bytes(data=archive.read(name=matches[0]))
    # These are the explicitly supplied, original course pickle files.
    with (CACHE / "images.pkl").open(mode="rb") as input_file:
        images = np.asarray(pickle.load(file=input_file, encoding="latin1"))
    with (CACHE / "images_labels.pkl").open(mode="rb") as input_file:
        labels = np.asarray(pickle.load(file=input_file, encoding="latin1"))
    LOGGER.info("Face images: shape=%s dtype=%s range=(%s,%s)", images.shape, images.dtype, images.min(), images.max())
    LOGGER.info("Labels: %s", np.unique(ar=labels, return_counts=True))
    if images.shape[1:] != (224, 224, 3) or len(images) != len(labels) or len(np.unique(ar=labels)) != 3:
        raise ValueError("Unexpected supplied face dataset")
    (CACHE / "asset_manifest.json").write_text(data=json.dumps(obj=manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        LOGGER.exception("CNN asset preparation failed")
        raise
