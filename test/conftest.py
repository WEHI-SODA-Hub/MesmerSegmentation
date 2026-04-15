"""
Shared pytest fixtures for mesmer_segment tests.

TIFF files are written to a temporary directory for each test that needs them,
using the same savers as create_test_data.py so the fixtures stay in sync.
"""
import sys
from pathlib import Path

import pytest
import numpy as np

# Ensure tests import the local package from src/ (not an installed version),
# and allow importing sibling modules from test/.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from create_test_data import (  # noqa: E402
    CHANNEL_NAMES,
    FOV_SIZE_MICRONS,
    FRAME_SIZE,
    MPP,
    make_synthetic_image,
    save_mibi_tiff,
    save_ome_tiff,
    save_plain_tiff,
)


@pytest.fixture(scope="session")
def synthetic_data() -> np.ndarray:
    """(2, 64, 64) uint16 array — generated once per test session."""
    return make_synthetic_image()


@pytest.fixture
def mibi_tiff(tmp_path: Path, synthetic_data: np.ndarray) -> Path:
    path = tmp_path / "test_mibi.tiff"
    save_mibi_tiff(synthetic_data, path)
    return path


@pytest.fixture
def ome_tiff(tmp_path: Path, synthetic_data: np.ndarray) -> Path:
    path = tmp_path / "test_ome.tiff"
    save_ome_tiff(synthetic_data, path)
    return path


@pytest.fixture
def plain_tiff(tmp_path: Path, synthetic_data: np.ndarray) -> Path:
    path = tmp_path / "test_plain.tiff"
    save_plain_tiff(synthetic_data, path)
    return path
