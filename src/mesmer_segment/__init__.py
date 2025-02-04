import json
import sys
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Annotated, List

import geopandas as gpd
import numpy as np
import rasterio.features
import typer
from numpy.typing import NDArray
from tifffile import TiffFile
from xarray import DataArray, concat
from deepcell.applications import Mesmer
from skimage.segmentation import clear_border

app = typer.Typer(rich_markup_mode="markdown")
MISSING = object()


class CombineMethod(str, Enum):
    PROD = "prod"
    MAX = "max"


def mibi_tiff_to_xarray(tiff: TiffFile) -> DataArray:
    """
    Takes a MIBI TIFF and converts it to an xarray with relevant axis, coordinate and metadata attached.
    Note: won"t work with a regular TIFF as this depends on MIBI specific metadata
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

    return DataArray(data=channels, dims=["C", "X", "Y"], coords={"C": channel_names}, attrs=attrs)


def combine_channels(array: DataArray, channels: List[str], combined_name: str, combine_method: CombineMethod) -> DataArray:
    """
    Combines multiple channels into a single channel by taking the maximum value at each pixel.
    Adds the combined channel to the array.
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


def calculate_maxima_threshold(segmentation_level: int) -> float:
    """
    Calculate maxima threshold based on code used by MIBIextension tool
    This uses a linear function to scale maxima_threshold based on segmentation level input.
    Keeping this for compatibility, but the logic of this calculation is not clear and should
    be re-evaluated.
    """
    if segmentation_level < 5:
        subtractive_factor = 0.0002 * segmentation_level
    else:
        subtractive_factor = 0.2 * (0.9 - 0.001) * segmentation_level + 2 * 0.001 - 0.9

    return 0.1 - 0.1 * subtractive_factor


def get_segmentation_predictions(array: DataArray, nuclear_channel: str, membrane_channel: List[str], kwargs_nuclear: dict[str, float], kwargs_whole_cell: dict[str, float]) -> NDArray:
    """
    Segments the input array using Mesmer.
    Mesmer assumes the input is a 4D array with dimensions (batch, x, y, channel).
    There must be exactly 2 channels, and they have to correspond to nuclear and channel markers respectively
    """
    combined_membrane_channel = "combined_membrane" if len(membrane_channel) > 1 else membrane_channel[0]
    np_array = array.sel(
        C=[nuclear_channel, combined_membrane_channel]
    ).expand_dims(
        "batch"
    ).transpose(
        "batch", "X", "Y", "C"
    ).to_numpy()

    app = Mesmer()
    mpp = array.attrs["fov_size"] / array.attrs["frame_size"]

    # Set compartment to nuclear if using pixel expansion
    assert 'pixel_expansion' in kwargs_nuclear, "pixel_expansion must be specified in kwargs_nuclear"
    compartment = "nuclear" if kwargs_nuclear['pixel_expansion'] > 0 else "whole-cell"

    # The result is a 4D array, but the first and last dimensions are both 1
    return app.predict(
        np_array,
        image_mpp=mpp,
        compartment=compartment,
        postprocess_kwargs_nuclear=kwargs_nuclear,
        postprocess_kwargs_whole_cell=kwargs_whole_cell
    ).squeeze().astype("int32")


def labels_to_features(lab: np.ndarray, object_type="annotation", connectivity: int=4,
                      mask=None, classification=None):
    """
    Create a GeoJSON FeatureCollection from a labeled image.
    """
    features = []

    # Ensure types are valid
    if lab.dtype == bool:
        mask = lab
        lab = lab.astype(np.uint8)
    else:
        mask = lab > 0

    # Trace geometries
    for s in rasterio.features.shapes(lab, mask=mask, connectivity=connectivity):
        # Create properties
        props = dict(object_type=object_type)

        # Just to show how a classification can be added
        if classification is not None:
            props["classification"] = classification

        # Wrap in a dict to effectively create a GeoJSON Feature
        po = dict(type="Feature", geometry=s[0], properties=props)

        features.append(po)

    return features

@app.command(help="Segments a MIBI TIFF using Mesmer, and prints the result to stdout. Note that you will need to obtain and export a DeepCell API key as explained [here](https://deepcell.readthedocs.io/en/master/API-key.html).")
def main(
    mibi_tiff: Annotated[Path, typer.Argument(help="Path to the MIBI TIFF input file")],
    nuclear_channel: Annotated[str, typer.Option(help="Name of the nuclear channel")],
    membrane_channel: Annotated[List[str], typer.Option(help="Name(s) of the membrane channels (can be repeated)")],
    combine_method: Annotated[CombineMethod, typer.Option(help="Method to use for combining channels (prod or max)")] = CombineMethod.PROD,
    segmentation_level: Annotated[int, typer.Option(help="Segmentation level between 0-10 where 0 is less segmentation and 10 is more", min=0, max=10)] = 5,
    interior_threshold: Annotated[float, typer.Option(help="Controls how conservative model is in distinguishing cell from background (lower values = larger cells, higher values = smaller cells)")] = 0.3,
    remove_cells_touching_border: Annotated[bool, typer.Option(help="Whether to remove cells touching the border of the image")] = True,
    pixel_expansion: Annotated[int, typer.Option(help="Specify a manual pixel expansion after segmentation. NOTE: This will perform segmentation in nuclear mode only.")] = 0,
):

    tiff = TiffFile(mibi_tiff)
    array = mibi_tiff_to_xarray(tiff)
    array = combine_channels(array, membrane_channel, "combined_membrane", CombineMethod(combine_method))

    # Collate args and run segmentation
    maxima_threshold = calculate_maxima_threshold(segmentation_level)
    kwargs_nuclear = {'pixel_expansion': pixel_expansion,
                      'maxima_threshold': maxima_threshold,
                      'interior_threshold': interior_threshold}
    kwargs_whole_cell = {'maxima_threshold': maxima_threshold,
                         'interior_threshold': interior_threshold}
    segmentation_predictions = get_segmentation_predictions(array, nuclear_channel, membrane_channel,
                                                            kwargs_nuclear, kwargs_whole_cell)

    # Post processing functions
    if remove_cells_touching_border:
        segmentation_predictions = clear_border(segmentation_predictions)

    # Convert to GeoJSON features for output
    features = labels_to_features(segmentation_predictions, object_type="annotation")

    # Extract the geometries and properties of each feature
    geoms = []
    for feature in features:
        geoms.append(feature["geometry"])

    # Create a geopandas dataframe from the geometries and properties
    gdf = gpd.GeoDataFrame.from_features(features)

    # Write the geopandas dataframe to a GeoJSON file
    with BytesIO() as buffer:
        gdf.to_file(buffer, driver="GeoJSON")
        buffer.seek(0)
        sys.stdout.buffer.write(buffer.read())
