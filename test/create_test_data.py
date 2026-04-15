#!/usr/bin/env python3
"""
Creates synthetic TIFF test fixtures for mesmer_segment pytests.

Generates small (64x64) two-channel images (nuclear + membrane) in three
formats that exercise every code path in tiff_to_xarray:

  test_mibi.tiff   — MIBI-style TIFF with per-page JSON metadata
  test_ome.tiff    — OME-TIFF with proper OME-XML metadata
  test_plain.tiff  — Plain TIFF with no metadata (fallback path)

No licensed data is used; all images are synthetic numpy arrays.

Usage:
    python test/create_test_data.py [--output-dir test/]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import tifffile


# ---------------------------------------------------------------------------
# Synthetic image parameters
# ---------------------------------------------------------------------------

IMAGE_SIZE = 64
CHANNEL_NAMES = ["nuclear", "membrane"]
FOV_SIZE_MICRONS = 100.0
FRAME_SIZE = IMAGE_SIZE
MPP = FOV_SIZE_MICRONS / FRAME_SIZE  # µm per pixel


# ---------------------------------------------------------------------------
# Synthetic image generation
# ---------------------------------------------------------------------------

def _gaussian(y: np.ndarray, x: np.ndarray, cy: int, cx: int, sigma: float) -> np.ndarray:
    return np.exp(-((y - cy) ** 2 + (x - cx) ** 2) / (2 * sigma ** 2))


def make_synthetic_image(seed: int = 42, size: int = IMAGE_SIZE) -> np.ndarray:
    """
    Return a (2, size, size) uint16 array simulating two-channel fluorescence:
      - channel 0 (nuclear):  Gaussian blobs
      - channel 1 (membrane): hollow ring halos around each nucleus
    """
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size]

    nuclear = np.zeros((size, size), dtype=np.float32)
    membrane = np.zeros((size, size), dtype=np.float32)

    n_cells = 8
    margin = size // 8
    centres = rng.integers(margin, size - margin, size=(n_cells, 2))

    for cy, cx in centres:
        intensity = float(rng.uniform(0.5, 1.0))
        nuclear += _gaussian(y, x, cy, cx, sigma=4) * intensity
        # membrane is a hollow ring: difference of two Gaussians
        ring = (_gaussian(y, x, cy, cx, sigma=7) - _gaussian(y, x, cy, cx, sigma=5)).clip(0)
        membrane += ring * float(rng.uniform(0.5, 1.0))

    scale = np.iinfo(np.uint16).max
    nuclear_u16 = (nuclear / nuclear.max() * scale).astype(np.uint16)
    membrane_u16 = (membrane / membrane.max() * scale).astype(np.uint16)

    return np.stack([nuclear_u16, membrane_u16])  # (C, Y, X)


# ---------------------------------------------------------------------------
# Format-specific savers
# ---------------------------------------------------------------------------

def save_mibi_tiff(data: np.ndarray, path: Path) -> None:
    """Save (C, Y, X) array as a MIBI-style TIFF with per-page JSON metadata."""
    with tifffile.TiffWriter(path) as tw:
        for i, name in enumerate(CHANNEL_NAMES):
            desc = json.dumps({
                "channel.target": name,
                "raw_description": {
                    "fovSizeMicrons": FOV_SIZE_MICRONS,
                    "frameSize": FRAME_SIZE,
                },
            })
            tw.write(data[i], description=desc, photometric="minisblack")


def save_ome_tiff(data: np.ndarray, path: Path) -> None:
    """Save (C, Y, X) array as a proper OME-TIFF with channel and resolution metadata."""
    tifffile.imwrite(
        path,
        data,
        ome=True,
        metadata={
            "Channel": {"Name": CHANNEL_NAMES},
            "PhysicalSizeX": MPP,
            "PhysicalSizeY": MPP,
            "PhysicalSizeXUnit": "µm",
            "PhysicalSizeYUnit": "µm",
        },
        compression="deflate",
    )


def save_plain_tiff(data: np.ndarray, path: Path) -> None:
    """Save (C, Y, X) array as a plain TIFF with no metadata (exercises the fallback path)."""
    tifffile.imwrite(path, data)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def create_all(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    data = make_synthetic_image()

    save_mibi_tiff(data, output_dir / "test_mibi.tiff")
    save_ome_tiff(data, output_dir / "test_ome.tiff")
    save_plain_tiff(data, output_dir / "test_plain.tiff")

    print(f"Wrote test_mibi.tiff, test_ome.tiff, test_plain.tiff → {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent,
        help="Directory to write test TIFFs into (default: same dir as this script)",
    )
    args = parser.parse_args()
    create_all(args.output_dir)
