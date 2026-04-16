"""
Tests for mesmer_segment.segmentation — channel operations and mask correction.
"""
import numpy as np
import pytest
from xarray import DataArray

from mesmer_segment.segmentation import (
    CombineMethod,
    calculate_maxima_threshold,
    combine_channels,
    extract_channels,
    fix_mask_orientation,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def two_channel_array() -> DataArray:
    """Simple (2, 64, 64) DataArray matching dims returned by tiff_to_xarray."""
    data = np.zeros((2, 64, 64), dtype=np.uint16)
    data[0] = 100  # nuclear
    data[1] = 200  # membrane
    return DataArray(data, dims=["C", "Y", "X"], coords={"C": ["nuclear", "membrane"]})


@pytest.fixture
def two_channel_array_nonsquare() -> DataArray:
    """Non-square (2, 20, 30) array to catch Y/X axis swaps."""
    data = np.zeros((2, 20, 30), dtype=np.uint16)
    data[0] = 100  # nuclear
    data[1] = 200  # membrane
    return DataArray(data, dims=["C", "Y", "X"], coords={"C": ["nuclear", "membrane"]})


# ---------------------------------------------------------------------------
# combine_channels
# ---------------------------------------------------------------------------

class TestCombineChannels:

    def test_single_channel_returns_input_unchanged(self, two_channel_array):
        result = combine_channels(two_channel_array, ["nuclear"], "combined", CombineMethod.MAX)
        assert result is two_channel_array

    def test_max_output_has_combined_channel(self, two_channel_array):
        result = combine_channels(
            two_channel_array, ["nuclear", "membrane"], "combined", CombineMethod.MAX
        )
        assert "combined" in result.coords["C"].values

    def test_max_is_elementwise_maximum(self, two_channel_array):
        result = combine_channels(
            two_channel_array, ["nuclear", "membrane"], "combined", CombineMethod.MAX
        )
        combined = result.sel(C="combined").values
        # nuclear=100, membrane=200 → max should be 200
        assert np.all(combined == 200)

    def test_prod_output_has_combined_channel(self, two_channel_array):
        result = combine_channels(
            two_channel_array, ["nuclear", "membrane"], "combined", CombineMethod.PROD
        )
        assert "combined" in result.coords["C"].values

    def test_output_has_one_extra_channel(self, two_channel_array):
        result = combine_channels(
            two_channel_array, ["nuclear", "membrane"], "combined", CombineMethod.MAX
        )
        assert result.sizes["C"] == 3

    def test_original_channels_preserved(self, two_channel_array):
        result = combine_channels(
            two_channel_array, ["nuclear", "membrane"], "combined", CombineMethod.MAX
        )
        assert np.all(result.sel(C="nuclear").values == 100)
        assert np.all(result.sel(C="membrane").values == 200)

    def test_preserves_input_dim_order(self, two_channel_array):
        result = combine_channels(
            two_channel_array, ["nuclear", "membrane"], "combined", CombineMethod.MAX
        )
        assert result.dims == two_channel_array.dims


# ---------------------------------------------------------------------------
# extract_channels
# ---------------------------------------------------------------------------

class TestExtractChannels:

    def test_output_shape(self, two_channel_array):
        arr = extract_channels(two_channel_array, "nuclear", "membrane")
        # (batch=1, Y=64, X=64, channels=2)
        assert arr.shape == (1, 64, 64, 2)

    def test_padding_reduces_spatial_dims(self, two_channel_array):
        arr = extract_channels(two_channel_array, "nuclear", "membrane", padding=4)
        assert arr.shape == (1, 56, 56, 2)

    def test_channel_order_respected(self, two_channel_array):
        arr = extract_channels(two_channel_array, "nuclear", "membrane")
        assert np.all(arr[0, :, :, 0] == 100), "nuclear should be first"
        assert np.all(arr[0, :, :, 1] == 200), "membrane should be second"

    def test_channel_order_reversed(self, two_channel_array):
        arr = extract_channels(two_channel_array, "membrane", "nuclear")
        assert np.all(arr[0, :, :, 0] == 200), "membrane should be first"
        assert np.all(arr[0, :, :, 1] == 100), "nuclear should be second"

    def test_nonsquare_shape_keeps_yx_order(self, two_channel_array_nonsquare):
        arr = extract_channels(two_channel_array_nonsquare, "nuclear", "membrane")
        assert arr.shape == (1, 20, 30, 2)

    def test_missing_channel_raises_clear_error(self, two_channel_array):
        with pytest.raises(
            ValueError,
            match=(
                r"Requested channel\(s\) not found: missing\. "
                r"Available channels: nuclear, membrane"
            ),
        ):
            extract_channels(two_channel_array, "missing", "membrane")


class TestCombineChannelsErrors:

    def test_missing_channel_raises_clear_error(self, two_channel_array):
        with pytest.raises(
            ValueError,
            match=(
                r"Requested channel\(s\) not found: missing\. "
                r"Available channels: nuclear, membrane"
            ),
        ):
            combine_channels(
                two_channel_array,
                ["nuclear", "missing"],
                "combined",
                CombineMethod.MAX,
            )


# ---------------------------------------------------------------------------
# calculate_maxima_threshold
# ---------------------------------------------------------------------------

class TestCalculateMaximaThreshold:

    def test_level_zero_returns_01(self):
        assert calculate_maxima_threshold(0) == pytest.approx(0.1)

    def test_threshold_decreases_with_level(self):
        # higher level = more segmentation = lower threshold
        assert calculate_maxima_threshold(3) < calculate_maxima_threshold(1)
        assert calculate_maxima_threshold(8) < calculate_maxima_threshold(5)

    def test_all_levels_in_valid_range(self):
        for level in range(11):
            t = calculate_maxima_threshold(level)
            assert 0 < t <= 0.1, f"level {level} gave out-of-range threshold {t}"


# ---------------------------------------------------------------------------
# fix_mask_orientation
# ---------------------------------------------------------------------------

class TestFixMaskOrientation:

    def test_correct_shape_returned_unchanged(self):
        mask = np.zeros((10, 20), dtype=np.uint32)
        result = fix_mask_orientation(mask, (10, 20))
        assert result is mask  # same object — no copy made

    def test_transposed_shape_is_corrected(self):
        mask = np.zeros((20, 10), dtype=np.uint32)
        result = fix_mask_orientation(mask, (10, 20))
        assert result.shape == (10, 20)

    def test_values_preserved_after_transpose(self):
        mask = np.arange(6, dtype=np.uint32).reshape(2, 3)
        result = fix_mask_orientation(mask, (3, 2))
        assert result.shape == (3, 2)
        np.testing.assert_array_equal(result, mask.T)

    def test_force_transpose_always_applies(self):
        mask = np.zeros((10, 20), dtype=np.uint32)
        result = fix_mask_orientation(mask, (10, 20), force_transpose=True)
        assert result.shape == (20, 10)

    def test_square_mask_not_auto_corrected(self):
        # Square images are ambiguous — auto-correction should not apply
        mask = np.zeros((10, 10), dtype=np.uint32)
        result = fix_mask_orientation(mask, (10, 10))
        assert result is mask

    def test_force_transpose_on_square(self):
        mask = np.zeros((10, 10), dtype=np.uint32)
        result = fix_mask_orientation(mask, (10, 10), force_transpose=True)
        assert result.shape == (10, 10)  # transpose of square is still square
