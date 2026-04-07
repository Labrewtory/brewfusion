#!/usr/bin/env python3
"""Download and extract BrewFusion model weights from GitHub Releases.

This script fetches the pre-trained DiT checkpoint, GNN embeddings, and
registry mappings necessary for running inference locally without retraining.
"""

import logging
import os
import tarfile
import urllib.request
from pathlib import Path

from brewfusion.config import PROJECT_ROOT

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

# GitHub Release URL configuration
RELEASE_VERSION = "v1.0.0"
ASSET_NAME = "brewfusion_weights_v1.tar.gz"
DOWNLOAD_URL = f"https://github.com/Labrewtory/brewfusion/releases/download/{RELEASE_VERSION}/{ASSET_NAME}"


def download_file(url: str, dest: Path) -> None:
    """Download a file with a basic progress indicator."""
    logger.info("Downloading from %s...", url)

    def reporthook(blocknum, blocksize, totalsize):
        readsofar = blocknum * blocksize
        if totalsize > 0:
            percent = readsofar * 1e2 / totalsize
            s = "\r%5.1f%% %*d / %d" % (
                percent,
                len(str(totalsize)),
                readsofar,
                totalsize,
            )
            print(s, end="")
            if readsofar >= totalsize:
                print()
        else:
            print(f"\rRead {readsofar} bytes", end="")

    urllib.request.urlretrieve(url, str(dest), reporthook)
    logger.info("Download complete.")


def extract_tarball(archive_path: Path, extract_dir: Path) -> None:
    """Extract standard tarball into the target directory."""
    logger.info("Extracting %s into %s", archive_path.name, extract_dir)
    with tarfile.open(archive_path, "r:gz") as tar:
        # Check security (avoid absolute paths)
        for member in tar.getmembers():
            if member.name.startswith("/") or ".." in member.name:
                raise ValueError("Insecure tarball detected.")

        def is_within_directory(directory, target):
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
            prefix = os.path.commonprefix([abs_directory, abs_target])
            return prefix == abs_directory

        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
            tar.extractall(path, members, numeric_owner=numeric_owner)

        safe_extract(tar, extract_dir)
    logger.info("Extraction complete.")


def main():
    target_dir = PROJECT_ROOT / "data"
    target_dir.mkdir(parents=True, exist_ok=True)

    archive_path = target_dir / ASSET_NAME

    # Check if files already exist via standard paths
    if (target_dir / "models" / "dit_best.pt").exists() and (
        target_dir / "graph" / "gnn_embeddings.pt"
    ).exists():
        logger.info(
            "Weights appear to be already installed. Use --force to redownload. (Not implemented)"
        )
        return

    try:
        download_file(DOWNLOAD_URL, archive_path)
        extract_tarball(archive_path, target_dir)
    except Exception as e:
        logger.error("Failed to download weights: %s", e)
        logger.info(
            "Please download manually from https://github.com/Labrewtory/brewfusion/releases"
        )
    finally:
        # Cleanup tarball
        if archive_path.exists():
            archive_path.unlink()


if __name__ == "__main__":
    main()
