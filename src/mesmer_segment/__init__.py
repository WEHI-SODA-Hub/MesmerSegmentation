from deepcell.applications import Mesmer
import typer
from pathlib import Path
from tifffile import TiffFile
from xarray import DataArray
import json

mesmer = Mesmer()
app = typer.Typer()
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


@app.command()
def main(
    mibi_tiff: Path
):
    tiff = TiffFile(mibi_tiff)
    array = mibi_tiff_to_xarray(tiff)

    app = Mesmer()
    mpp = array.attrs["fov_size"] / array.attrs["frame_size"]
    segmentation_predictions = app.predict(array, image_mpp=mpp, compartment="whole-cell")
