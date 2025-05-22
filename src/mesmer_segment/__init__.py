import json
import sys
from enum import Enum
from pathlib import Path
from typing import Annotated, List

import numpy as np
import typer
from numpy.typing import NDArray
from tifffile import TiffFile, imwrite
from xarray import DataArray, concat
from skimage.segmentation import clear_border
from skimage.util import crop

app = typer.Typer(rich_markup_mode="markdown")
MISSING = object()

# Define the properties to extract from regionprops_table per channel
channel_properties = [
    'intensity_mean',
    'intensity_min',
    'intensity_max',
    'intensity_std'
]


class CombineMethod(str, Enum):
    PROD = "prod"
    MAX = "max"


class Compartment(str, Enum):
    WHOLE_CELL = "whole-cell"
    NUCLEAR = "nuclear"


def mibi_tiff_to_xarray(tiff: TiffFile) -> DataArray:
    """
    Takes a MIBI TIFF and converts it to an xarray with relevant axis,
    coordinate and metadata attached. Note: won"t work with a regular
    TIFF as this depends on MIBI specific metadata
    """
    channel_names: list[str] = []
    attrs: dict[str, int] = {}
    #: List of channels, each of which are 2D
    channels = []

    for page in tiff.pages:
        description = json.loads(page.description)
        channel_names.append(description["channel.target"])
        attrs["fov_size"] = description["raw_description"]["fovSizeMicrons"]
        attrs["frame_size"] = description["raw_description"]["frameSize"]
        channels.append(page.asarray())

    return DataArray(data=channels, dims=["C", "X", "Y"],
                     coords={"C": channel_names}, attrs=attrs)


def combine_channels(array: DataArray, channels: List[str], combined_name: str,
                     combine_method: CombineMethod) -> DataArray:
    """
    Combines multiple channels into a single channel using the specified method
    (prod or max). Adds the combined channel to the array.
    """

    if len(channels) == 1:
        return array

    combined = array.sel(C=channels)

    if combine_method == CombineMethod.MAX:
        combined = combined.max(dim="C")
    elif combine_method == CombineMethod.PROD:
        combined = combined.prod(dim="C")

    combined = combined.expand_dims("C").assign_coords(C=[combined_name])

    return concat([array, combined], dim="C")


def extract_channels(array: DataArray, nuclear_channel: str,
                     membrane_channel: str, padding: int = 0) -> np.ndarray:
    """
    Extract the nuclear and membrane channels from the input array and return
    as a 4D numpy array as preparation for segmentation input. Optionally crop
    the image to a specified padding.
    """

    seg_array = array.sel(
        C=[nuclear_channel, membrane_channel]
    ).expand_dims(
        "batch"
    ).transpose(
        "batch", "X", "Y", "C"
    ).to_numpy()

    # Crop the image if padding is specified
    if padding > 0:
        seg_array = crop(seg_array, ((0, 0), (padding, padding),
                                     (padding, padding), (0, 0)))

    return seg_array


def calculate_maxima_threshold(segmentation_level: int) -> float:
    """
    Calculate maxima threshold based on code used by MIBIextension tool
    This uses a linear function to scale maxima_threshold based on segmentation
    level input. Keeping this for compatibility, but the logic of this
    calculation is not clear and should be re-evaluated.
    """
    if segmentation_level < 5:
        subtractive_factor = 0.0002 * segmentation_level
    else:
        subtractive_factor = \
            0.2 * (0.9 - 0.001) * segmentation_level + 2 * 0.001 - 0.9

    return 0.1 - 0.1 * subtractive_factor


def get_segmentation_predictions(seg_array: np.ndarray,
                                 mpp: float,
                                 compartment: Compartment,
                                 kwargs_nuclear: dict[str, float],
                                 kwargs_whole_cell: dict[str, float]
                                 ) -> NDArray:
    """
    Segments the input array using Mesmer.
    Mesmer assumes the input is a 4D array with dimensions (batch, x, y,
    channel). There must be exactly 2 channels, and they have to correspond
    to nuclear and channel markers respectively
    """
    from deepcell.applications import Mesmer
    app = Mesmer()
    return app.predict(
        seg_array,
        image_mpp=mpp,
        compartment=compartment,
        postprocess_kwargs_nuclear=kwargs_nuclear,
        postprocess_kwargs_whole_cell=kwargs_whole_cell
    ).squeeze().astype("uint32")


@app.command(
    help="Segments a MIBI TIFF using Mesmer, and prints the result to"
         "stdout. Note that you will need to obtain and export a "
         "DeepCell API key as explained "
         "here: https://deepcell.readthedocs.io/en/master/API-key.html"
)
def main(
    mibi_tiff: Annotated[Path, typer.Argument(
        help="Path to the MIBI TIFF input file."
    )],
    nuclear_channel: Annotated[str, typer.Option(
        help="Name of the nuclear channel."
    )],
    membrane_channel: Annotated[List[str], typer.Option(
        help="Name(s) of the membrane channels (can be repeated)"
             "Ensure that channels with spaces are quoted.")
    ],
    compartment: Annotated[Compartment, typer.Option(
        help="Compartment to segment (whole-cell or nuclear).")
    ] = Compartment.WHOLE_CELL,
    combine_method: Annotated[CombineMethod, typer.Option(
        help="Method to use for combining channels (prod or max).")
    ] = CombineMethod.PROD,
    segmentation_level: Annotated[int, typer.Option(
        help="Segmentation level between 0-10 where 0 is "
             "less segmentation and 10 is more. Set to -1 "
             "to use maxima_threshold instead. (This option "
             "is for backwards compatibility with an old tool.)",
        min=-1, max=10)
    ] = -1,
    maxima_threshold: Annotated[float, typer.Option(
        help="Controls segmentation level directly in mesmer, "
             "not sure scaling via segmentation_level (lower "
             "values = more cells, higher values = fewer cells). "
             "Provide a value >0 to use this parameter.", min=0)
    ] = 0.1,
    interior_threshold: Annotated[float, typer.Option(
        help="Controls how conservative model is in distinguishing "
             "cell from background (lower values = larger cells, "
             "higher values = smaller cells).")
    ] = 0.3,
    maxima_smooth: Annotated[float, typer.Option(
        help="Controls what is considered a unique cell (lower values "
             "= more separate cells, higher values = fewer cells).", min=0)
    ] = 0,
    min_nuclei_area: Annotated[int, typer.Option(
        help="Minimum area of nuclei to keep.", min=0)
    ] = 15,
    remove_cells_touching_border: Annotated[bool, typer.Option(
        help="Whether to remove cells touching the border of the image.")
    ] = True,
    pixel_expansion: Annotated[int, typer.Option(
        help="Specify a manual pixel expansion after segmentation.", min=0)
    ] = 0,
    padding: Annotated[int, typer.Option(
        help="Number of pixels to crop the image by before segmentation.",
        min=0)
    ] = 0,
):

    tiff = TiffFile(mibi_tiff)
    full_array = mibi_tiff_to_xarray(tiff)

    # Combine channels and prepare image array
    combined_membrane_channel = "combined_membrane" \
        if len(membrane_channel) > 1 else membrane_channel[0]
    full_array = combine_channels(full_array, membrane_channel,
                                  combined_membrane_channel,
                                  CombineMethod(combine_method))
    seg_array = extract_channels(full_array, nuclear_channel,
                                 combined_membrane_channel, padding)

    # Collate args and run segmentation
    mpp = full_array.attrs["fov_size"] / full_array.attrs["frame_size"]

    # if no direct maxima_threshold is provided, use the segmentation_level
    if segmentation_level > -1:
        maxima_threshold = calculate_maxima_threshold(segmentation_level)

    print(f"Segmenting with MPP: {mpp} and compartment: {compartment} "
          f"using maxima_threshold: {maxima_threshold}",
          file=sys.stderr)

    kwargs_nuclear = {
        "maxima_threshold": maxima_threshold,
        "maxima_smooth": maxima_smooth,
        "interior_threshold": interior_threshold,
        "small_objects_threshold": min_nuclei_area
    }

    kwargs_whole_cell = {
        "pixel_expansion": pixel_expansion,
        "maxima_threshold": maxima_threshold,
        "maxima_smooth": maxima_smooth,
        "interior_threshold": interior_threshold
    }

    segmentation_predictions = get_segmentation_predictions(
        seg_array, mpp, compartment, kwargs_nuclear, kwargs_whole_cell
    )

    # Post processing functions
    if remove_cells_touching_border:
        segmentation_predictions = clear_border(segmentation_predictions)

    imwrite(sys.stdout.buffer, segmentation_predictions)
