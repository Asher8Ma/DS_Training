"""Fetch the original CIFAR archive and verify Keras's published SHA-256."""

import argparse
import hashlib
import json
import logging
from pathlib import Path
import ssl
import urllib.request

LOGGER = logging.getLogger(name=__name__)
ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SHA256 = "6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce"
URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"


def main() -> None:
    """Download the official archive and validate its published hash."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=URL, help="Explicit archive URL; the official SHA-256 remains mandatory")
    arguments = parser.parse_args()
    context = ssl.create_default_context()
    # The trusted enterprise proxy certificate lacks Authority Key Identifier.
    # Match pre-3.13 validation policy: retain CA-chain and hostname verification.
    context.verify_flags &= ~ssl.VERIFY_X509_STRICT
    LOGGER.info("Using Windows trusted CAs and hostname verification; Python 3.13 structural certificate strictness disabled for this download only")
    target = ROOT / ".cache" / "keras" / "datasets" / "cifar-10-batches-py-target_archive"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        with target.open(mode="rb") as cached_file:
            cached_digest = hashlib.file_digest(cached_file, "sha256").hexdigest()
        if cached_digest != EXPECTED_SHA256:
            raise ValueError(f"Existing CIFAR cache failed integrity validation: {target}")
        LOGGER.info("Existing CIFAR archive passed official SHA-256: %s", target)
        return
    temporary = target.with_suffix(suffix=f".{hashlib.sha256(arguments.url.encode()).hexdigest()[:10]}.download")
    digest = hashlib.sha256()
    byte_count = 0
    with urllib.request.urlopen(url=arguments.url, context=context, timeout=120) as response, temporary.open(mode="wb") as output_file:
        LOGGER.info("HTTP %s; headers: %s", response.status, dict(response.headers))
        if response.status != 200:
            raise ValueError("Expected a complete HTTP 200 archive response")
        while chunk := response.read(1024 * 1024):
            output_file.write(chunk)
            digest.update(chunk)
            byte_count += len(chunk)
            if byte_count % (20 * 1024 * 1024) == 0:
                LOGGER.info("Downloaded %.0f MiB", byte_count / 1024**2)
    if digest.hexdigest() != EXPECTED_SHA256:
        raise ValueError(f"CIFAR archive checksum mismatch: {digest.hexdigest()}")
    temporary.replace(target=target)
    (target.parent / "cifar_download.json").write_text(
        data=json.dumps(obj={"url": arguments.url, "bytes": byte_count, "sha256": digest.hexdigest()}, indent=2),
        encoding="utf-8",
    )
    LOGGER.info("Verified %d bytes against official Keras SHA-256: %s", byte_count, target)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        LOGGER.exception("CIFAR download failed")
        raise
