"""Deterministic CPU runtime and artifact locations for the CNN exercise."""

import logging
import os
from datetime import datetime
from pathlib import Path
import warnings

warnings.filterwarnings(action="ignore", category=FutureWarning)
RANDOM_SEED = 42
NOTEBOOK_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_DIRECTORY = NOTEBOOK_DIRECTORY.parents[1]
CACHE_DIRECTORY = REPOSITORY_DIRECTORY / ".cache" / "cnn"
OUTPUT_DIRECTORY = NOTEBOOK_DIRECTORY / "outputs" / datetime.now().strftime("cnn_%Y%m%d_%H%M%S")
OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=False)
os.environ["KERAS_HOME"] = str(REPOSITORY_DIRECTORY / ".cache" / "keras")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"

LOGGER = logging.getLogger(name="cnn_exercise")
LOGGER.setLevel(level=logging.DEBUG)
LOGGER.propagate = False
for previous_handler in LOGGER.handlers[:]:
    previous_handler.close()
    LOGGER.removeHandler(hdlr=previous_handler)
for current_handler in (logging.StreamHandler(), logging.FileHandler(filename=OUTPUT_DIRECTORY / "execution.log", encoding="utf-8")):
    current_handler.setFormatter(fmt=logging.Formatter(fmt="%(asctime)s | %(levelname)s | %(message)s"))
    LOGGER.addHandler(hdlr=current_handler)
LOGGER.info("Timestamped output directory: %s", OUTPUT_DIRECTORY)
