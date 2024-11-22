import typer
from pathlib import Path
from tifffile import TiffFile, imwrite
from xarray import DataArray
import json
from typing import Annotated
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

    return DataArray(data=channels, dims=["C", "Y", "X"], coords={"C": channel_names}, attrs=attrs)

@app.command(help="Segments a MIBI TIFF using Mesmer, and prints the result to stdout. Note that you will need to obtain and export a DeepCell API key as explained [here](https://deepcell.readthedocs.io/en/master/API-key.html).")
def main(
    mibi_tiff: Annotated[Path, typer.Argument(help="Path to the MIBI TIFF input file")],
    nuclear_channel: Annotated[str, typer.Option(help="Name of the nuclear channel")],
    membrane_channel: Annotated[str, typer.Option(help="Name of the membrane channel")]
):
    from deepcell.applications import Mesmer
    tiff = TiffFile(mibi_tiff)
    array = mibi_tiff_to_xarray(tiff)

    # Mesmer assumes the input is:
    # A 4D array with dimensions (batch, x, y, channel)
    # There must be exactly 2 channels, and they have to correspond to nuclear and channel markers respectively
    np_array = array.sel(
        C=[nuclear_channel, membrane_channel]
    ).expand_dims(
        "batch"
    ).transpose(
        "batch", "X", "Y", "C"
    ).to_numpy()

    app = Mesmer()
    mpp = array.attrs["fov_size"] / array.attrs["frame_size"]
    # The result is a 4D array, but the first and last dimensions are both 1
    segmentation_predictions: NDArray = app.predict(np_array, image_mpp=mpp, compartment="whole-cell").squeeze()
    imwrite(sys.stdout.buffer, segmentation_predictions)
