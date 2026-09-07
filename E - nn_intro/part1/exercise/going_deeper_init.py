"""Shared deterministic runtime configuration for the going-deeper exercise."""

import logging
import os
from datetime import datetime
from pathlib import Path
import sys
import warnings


warnings.filterwarnings(action="ignore", category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    stream=sys.stdout,
)

LOGGER = logging.getLogger(name="going_deeper")
RANDOM_SEED = 42

EXERCISE_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_DIRECTORY = EXERCISE_DIRECTORY.parents[2]
CACHE_DIRECTORY = REPOSITORY_DIRECTORY / ".cache" / "keras"
OUTPUT_DIRECTORY = (
    EXERCISE_DIRECTORY
    / "outputs"
    / datetime.now().strftime("going_deeper_%Y%m%d_%H%M%S")
)

CACHE_DIRECTORY.mkdir(parents=True, exist_ok=True)
OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=False)

os.environ["KERAS_HOME"] = str(CACHE_DIRECTORY)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"

LOGGER.info("Output directory: %s", OUTPUT_DIRECTORY.resolve())
