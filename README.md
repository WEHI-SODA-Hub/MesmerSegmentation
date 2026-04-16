# Mesmer Segmentation

This repository provides a command-line interface for Mesmer segmentation of MIBI and OME-XML TIFFs.

## Installation

Install this repo. You can do so using `pip install git+https://github.com/WEHI-SODA-Hub/MibiSegmentation.git` or by cloning the repo and running `pip install .`.

## Testing

Run the test suite with:

```bash
pytest
```

The current `pytest` suite covers local utilities and CLI-adjacent behavior, but it does **not** exercise the DeepCell-backed segmentation call itself. In particular, it does not validate end-to-end Mesmer segmentation against the external DeepCell API.

To run the DeepCell-backed integration test, first provide a token either by exporting it in your shell or by filling in the blank line near the top of `test/run_integration.sh`:

```bash
export DEEPCELL_ACCESS_TOKEN="YOURTOKEN"
test/run_integration.sh
```

The integration script creates the synthetic test TIFFs if they do not already exist, then runs `mesmer-segment` for MIBI, OME, and plain TIFF inputs across both `whole-cell` and `nuclear` compartments. Output masks are written into `test/`.

## CLI Usage

Don't forget to obtain an API key (see [here](https://deepcell.readthedocs.io/en/master/API-key.html))
and export it like so:

```bash
export DEEPCELL_ACCESS_TOKEN="YOURTOKEN"
```

### `mesmer-segment`

The main entry point is:

```bash
mesmer-segment [OPTIONS] TIFF > result.tiff
```

This command reads a MIBI or OME-XML TIFF, runs Mesmer segmentation, and writes the mask TIFF to `stdout`.

Required arguments and options:

- `TIFF`: path to the input TIFF.
- `--nuclear-channel TEXT`: nuclear channel name.
- `--membrane-channel TEXT`: membrane channel name; repeat this option to provide multiple membrane channels.

Optional segmentation controls:

| Option                                                              | Default                           | Notes                                                                                                    |
| ------------------------------------------------------------------- | --------------------------------- | -------------------------------------------------------------------------------------------------------- |
| `--compartment` (`whole-cell` or `nuclear`)                         | `whole-cell`                      | Segment whole cells or nuclei only.                                                                      |
| `--combine-method` (`prod` or `max`)                                | `prod`                            | How multiple membrane channels are combined.                                                             |
| `--segmentation-level INTEGER`                                      | `-1`                              | Legacy 0-10 control. If set to `0-10`, it is converted into `maxima_threshold`.                          |
| `--maxima-threshold FLOAT`                                          | `0.1`                             | Lower values produce more cells; higher values produce fewer. Used if `--segmentation-level` is unset.   |
| `--interior-threshold FLOAT`                                        | `0.3`                             | Lower values give larger cells; higher values give smaller cells.                                        |
| `--maxima-smooth FLOAT`                                             | `0`                               | Lower values split more cells; higher values merge more.                                                 |
| `--min-nuclei-area INTEGER`                                         | `15`                              | Minimum nucleus area kept in nuclear segmentation.                                                       |
| `--remove-cells-touching-border` / `--no-remove-cells-touching-border` | `--remove-cells-touching-border` | Remove segmented objects that touch the image border.                                                  |
| `--pixel-expansion INTEGER`                                         | `0`                               | Manual pixel expansion applied after segmentation.                                                       |
| `--padding INTEGER`                                                 | `0`                               | Crop this many pixels before segmentation.                                                               |
| `--force-transpose`                                                 | `False`                           | Always transpose the output mask. Useful when output orientation is visibly flipped.                     |

A typical usage example is:

```bash
mesmer-segment image.tiff \
    --nuclear-channel dsDNA \
    --membrane-channel panCK \
    --membrane-channel "MHC I" \
    --compartment whole-cell > result.tiff
```

Note: make sure to put quotes around channel names with spaces, e.g., `--membrane-channel "MHC I (HLA A B C)"`.
Also note that QuPath may display these channels with underscores, e.g., `MHC_I _(HLA_A_B_C)`.

### `mesmer-postprocess`

For masks that were already written to disk, a separate post-processing entry point is also installed:

```bash
mesmer-postprocess [OPTIONS] INPUT_TIFF MASK_PATH
```

- `INPUT_TIFF`: the original segmentation input TIFF.
- `MASK_PATH`: Mesmer output mask TIFF to update in place.
- `--force-transpose`: always transpose the mask instead of relying on the shape check.

Example:

```bash
mesmer-postprocess image.tiff result.tiff
```
