import typer
from pathlib import Path
from tifffile import TiffFile, imwrite
from xarray import DataArray, concat
import json
from typing import Annotated, List
from numpy.typing import NDArray
import sys

app = typer.Typer(rich_markup_mode="markdown")
MISSING = object()

def mibi_tiff_to_xarray(tiff: TiffFile) -> DataArray:
    """
    Takes a MIBI TIFF and converts it to an xarray with relevant axis, coordinate and metadata attached.
    Note: won't work with a regular TIFF as this depends on MIBI specific metadata
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

@app.command(help="Segments a MIBI TIFF using Mesmer, and prints the result to stdout. Note that you will need to obtain and export a DeepCell API key as explained [here](https://deepcell.readthedocs.io/en/master/API-key.html).")
def main(
    mibi_tiff: Annotated[Path, typer.Argument(help="Path to the MIBI TIFF input file")],
    nuclear_channel: Annotated[str, typer.Option(help="Name of the nuclear channel")],
    membrane_channel: Annotated[List[str], typer.Option(help="Name(s) of the membrane channels (can be repeated)")],
):
    from deepcell.applications import Mesmer
    from deepcell.utils.plot_utils import create_rgb_image, make_outline_overlay

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
    segmentation_predictions: NDArray = app.predict(np_array, image_mpp=mpp, compartment="whole-cell")
    rgb_images = create_rgb_image(np_array, channel_colors=['green', 'blue'])
    overlay = make_outline_overlay(rgb_data=rgb_images, predictions=segmentation_predictions)

    imwrite(sys.stdout.buffer, overlay)
