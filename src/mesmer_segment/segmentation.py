"""
Segmentation logic: channel manipulation and Mesmer prediction.
"""
import sys
from enum import Enum

import numpy as np
from numpy.typing import NDArray
from xarray import DataArray
from skimage.util import crop


# Properties extracted from regionprops_table per channel
channel_properties = [
    'intensity_mean',
    'intensity_min',
    'intensity_max',
    'intensity_std',
]


class CombineMethod(str, Enum):
    PROD = "prod"
    MAX = "max"


class Compartment(str, Enum):
    WHOLE_CELL = "whole-cell"
    NUCLEAR = "nuclear"


def combine_channels(
    array: DataArray,
    channels: list[str],
    combined_name: str,
    combine_method: CombineMethod,
) -> DataArray:
    """
    Combine multiple channels into a single channel using the specified method
    (prod or max) and append the result to the array.
    """
    if len(channels) == 1:
        return array

    selected_data = array.sel(C=channels).values

    if combine_method == CombineMethod.MAX:
        combined_data = np.max(selected_data, axis=0, keepdims=True)
    elif combine_method == CombineMethod.PROD:
        selected_data = selected_data.astype(np.uint64)
        combined_data = np.prod(selected_data, axis=0, keepdims=True)
    else:
        raise ValueError(f"Unknown combine_method: {combine_method}")

    max_val: int = np.iinfo(np.uint16).max
    if np.max(combined_data) > max_val:
        scale_factor: float = (np.iinfo(np.uint16).max - 1) / max_val
        combined_data = np.clip(
            combined_data * scale_factor, 0, np.iinfo(np.uint16).max
        ).astype(np.uint16)
    else:
        combined_data = combined_data.astype(np.uint16)

    new_data = np.concatenate([array.values, combined_data], axis=0)
    new_coords = list(array.coords["C"].values) + [combined_name]

    del selected_data, combined_data

    return DataArray(
        data=new_data, dims=array.dims,
        coords={"C": new_coords}, attrs=array.attrs,
    )


def extract_channels(
    array: DataArray,
    nuclear_channel: str,
    membrane_channel: str,
    padding: int = 0,
) -> np.ndarray:
    """
    Extract the nuclear and membrane channels from the input array and return
    as a 4D numpy array (batch, Y, X, C) ready for Mesmer. Optionally crops
    by `padding` pixels on each side.
    """
    seg_array = (
        array.sel(C=[nuclear_channel, membrane_channel])
        .expand_dims("batch")
        .transpose("batch", "Y", "X", "C")
        .to_numpy()
    )

    if padding > 0:
        seg_array = crop(
            seg_array, ((0, 0), (padding, padding), (padding, padding), (0, 0))
        )

    return seg_array


def calculate_maxima_threshold(segmentation_level: int) -> float:
    """
    Calculate maxima_threshold from a 0-10 segmentation level.
    Kept for backwards compatibility with the old MIBIextension tool.
    """
    if segmentation_level < 5:
        subtractive_factor = 0.0002 * segmentation_level
    else:
        subtractive_factor = (
            0.2 * (0.9 - 0.001) * segmentation_level + 2 * 0.001 - 0.9
        )
    return 0.1 - 0.1 * subtractive_factor


def fix_mask_orientation(
    mask: NDArray,
    expected_shape: tuple[int, int],
    force_transpose: bool = False,
) -> NDArray:
    """
    Detect and correct transposed X/Y dimensions in a Mesmer output mask.

    Mesmer can produce masks with swapped X,Y axes. This compares the mask
    shape against `expected_shape` (spatial dims of the input image after any
    padding) and transposes if they are the reverse of each other.

    `force_transpose` always transposes regardless of the shape check, which
    is useful when the spatial orientation is visibly wrong but the shapes
    happen to be equal (e.g. a square image).
    """
    if force_transpose:
        print(
            f"Force-transposing mask from {mask.shape} to {mask.T.shape}",
            file=sys.stderr,
        )
        return mask.T
    if mask.shape != expected_shape and mask.shape == expected_shape[::-1]:
        print(
            f"Transposing mask from {mask.shape} to {expected_shape}",
            file=sys.stderr,
        )
        return mask.T
    return mask


def get_segmentation_predictions(
    seg_array: np.ndarray,
    mpp: float,
    compartment: Compartment,
    kwargs_nuclear: dict[str, float],
    kwargs_whole_cell: dict[str, float],
) -> NDArray:
    """
    Run Mesmer segmentation on a 4D array (batch, X, Y, channel).
    Returns a 2D uint32 label array.
    """
    from deepcell.applications import Mesmer
    model = Mesmer()
    return model.predict(
        seg_array,
        image_mpp=mpp,
        compartment=compartment,
        postprocess_kwargs_nuclear=kwargs_nuclear,
        postprocess_kwargs_whole_cell=kwargs_whole_cell,
    ).squeeze().astype("uint32")  # type: ignore
