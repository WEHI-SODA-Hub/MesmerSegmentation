"""
CLI entry point for mesmer-segment.
"""
import sys
from pathlib import Path
from typing import Annotated

import typer
from tifffile import imwrite
from skimage.segmentation import clear_border

from mesmer_segment.io import tiff_to_xarray
from mesmer_segment.segmentation import (
    CombineMethod,
    Compartment,
    calculate_maxima_threshold,
    combine_channels,
    extract_channels,
    fix_mask_orientation,
    get_segmentation_predictions,
)

app = typer.Typer(rich_markup_mode="markdown")


@app.command(
    help="Segments a MIBI or OME-XML TIFF using Mesmer, and prints the result "
         "to stdout. Note that you will need to obtain and export a DeepCell "
         "API key as explained here: "
         "https://deepcell.readthedocs.io/en/master/API-key.html"
)
def main(
    tiff: Annotated[Path, typer.Argument(
        help="Path to the TIFF input file."
    )],
    nuclear_channel: Annotated[str, typer.Option(
        help="Name of the nuclear channel."
    )],
    membrane_channel: Annotated[list[str], typer.Option(
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
             "(lower values = more cells, higher values = fewer cells). "
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
    force_transpose: Annotated[bool, typer.Option(
        help="Always transpose the output mask, regardless of shape check. "
             "Useful when the mask spatial orientation is visibly flipped.")
    ] = False,
):
    full_array = tiff_to_xarray(tiff)

    combined_membrane_channel = (
        "combined_membrane" if len(membrane_channel) > 1 else membrane_channel[0]
    )
    full_array = combine_channels(
        full_array, membrane_channel, combined_membrane_channel,
        CombineMethod(combine_method),
    )
    seg_array = extract_channels(
        full_array, nuclear_channel, combined_membrane_channel, padding
    )

    if "fov_size" not in full_array.attrs:
        mpp = full_array.attrs.get("microns_per_pixel", 0.5)
    else:
        mpp = full_array.attrs["fov_size"] / full_array.attrs["frame_size"]

    if segmentation_level > -1:
        maxima_threshold = calculate_maxima_threshold(segmentation_level)

    print(
        f"Segmenting with MPP: {mpp} and compartment: {compartment} "
        f"using maxima_threshold: {maxima_threshold}",
        file=sys.stderr,
    )

    kwargs_nuclear = {
        "maxima_threshold": maxima_threshold,
        "maxima_smooth": maxima_smooth,
        "interior_threshold": interior_threshold,
        "small_objects_threshold": min_nuclei_area,
    }
    kwargs_whole_cell = {
        "pixel_expansion": pixel_expansion,
        "maxima_threshold": maxima_threshold,
        "maxima_smooth": maxima_smooth,
        "interior_threshold": interior_threshold,
    }

    segmentation_predictions = get_segmentation_predictions(
        seg_array, mpp, compartment, kwargs_nuclear, kwargs_whole_cell
    )

    if remove_cells_touching_border:
        segmentation_predictions = clear_border(segmentation_predictions)

    # seg_array has shape (batch, X, Y, C); spatial dims are [1] and [2]
    expected_shape = (seg_array.shape[1], seg_array.shape[2])
    segmentation_predictions = fix_mask_orientation(
        segmentation_predictions, expected_shape, force_transpose
    )

    imwrite(sys.stdout.buffer, segmentation_predictions)


# ---------------------------------------------------------------------------
# Standalone postprocess entry point (for fixing already-written mask files)
# ---------------------------------------------------------------------------

postprocess_app = typer.Typer(rich_markup_mode="markdown")


@postprocess_app.command(
    help="Fix transposed X/Y dimensions in an existing Mesmer output mask. "
         "The original input TIFF is used to determine the expected spatial shape."
)
def postprocess(
    input_tiff: Annotated[Path, typer.Argument(
        help="Original input TIFF used during segmentation."
    )],
    mask_path: Annotated[Path, typer.Argument(
        help="Mesmer output mask TIFF to post-process (edited in-place)."
    )],
    force_transpose: Annotated[bool, typer.Option(
        help="Always transpose the mask, regardless of shape check.")
    ] = False,
):
    import tifffile as _tifffile
    from mesmer_segment.segmentation import fix_mask_orientation as _fix

    input_img = _tifffile.imread(input_tiff)
    expected_shape: tuple[int, int] = (
        input_img.shape[1], input_img.shape[2]
    ) if input_img.ndim == 3 else (input_img.shape[0], input_img.shape[1])

    mask = _tifffile.imread(mask_path)
    fixed = _fix(mask, expected_shape, force_transpose)
    if fixed is not mask:
        _tifffile.imwrite(mask_path, fixed, compression="deflate")
    else:
        print(f"Mask dimensions correct: {mask.shape}", file=sys.stderr)
