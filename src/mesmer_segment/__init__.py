import json
import sys
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

app = typer.Typer(rich_markup_mode="markdown")
MISSING = object()

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


def combine_channels(array: DataArray, channels: List[str], combined_name: str) -> DataArray:
    """
    Combines multiple channels into a single channel by taking the maximum value at each pixel.
    Adds the combined channel to the array.
    """

    if len(channels) == 1:
        return array

    combined = array.sel(
        C=channels
    ).max(
        dim="C"
    ).expand_dims(
        "C"
    ).assign_coords(
        C=[combined_name]
    )

    return concat([array, combined], dim="C")

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
):
    from deepcell.applications import Mesmer

    tiff = TiffFile(mibi_tiff)
    array = mibi_tiff_to_xarray(tiff)
    array = combine_channels(array, membrane_channel, "combined_membrane")

    # Mesmer assumes the input is:
    # A 4D array with dimensions (batch, x, y, channel)
    # There must be exactly 2 channels, and they have to correspond to nuclear and channel markers respectively
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
    # The result is a 4D array, but the first and last dimensions are both 1
    segmentation_predictions: NDArray = app.predict(np_array, image_mpp=mpp, compartment="whole-cell").squeeze().astype("int32")

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
