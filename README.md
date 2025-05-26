# Mesmer Segmentation

This repository provides a command-line interface for Mesmer segmentation of MIBI TIFFs.

## Installation

Install this repo. You can do so using `pip install git+https://github.com/WEHI-SODA-Hub/MibiSegmentation.git` or by cloning the repo and running `pip install .`.

## CLI Usage

Don't forget to obtain an API key (see [here](https://deepcell.readthedocs.io/en/master/API-key.html))
and export it like so:

```bash
export DEEPCELL_ACCESS_TOKEN="YOURTOKEN"
```

```
Usage: mesmer-segment [OPTIONS] MIBI_TIFF

 Segments a MIBI TIFF using Mesmer, and prints the result tostdout. Note that you will need to obtain and export a DeepCell API key as explained here:
 https://deepcell.readthedocs.io/en/master/API-key.html

╭─ Arguments ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *    mibi_tiff      PATH  Path to the MIBI TIFF input file.                                                                                               │
│                           [default: None]                                                                                                                 │
│                           [required]                                                                                                                      │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *  --nuclear-channel                                                      TEXT                       Name of the nuclear channel.                         │
│                                                                                                      [default: None]                                      │
│                                                                                                      [required]                                           │
│ *  --membrane-channel                                                     TEXT                       Name(s) of the membrane channels (can be             │
│                                                                                                      repeated)Ensure that channels with spaces are        │
│                                                                                                      quoted.                                              │
│                                                                                                      [default: None]                                      │
│                                                                                                      [required]                                           │
│    --compartment                                                          [whole-cell|nuclear]       Compartment to segment (whole-cell or nuclear).      │
│                                                                                                      [default: whole-cell]                                │
│    --combine-method                                                       [prod|max]                 Method to use for combining channels (prod or max).  │
│                                                                                                      [default: prod]                                      │
│    --segmentation-level                                                   INTEGER RANGE [-1<=x<=10]  Segmentation level between 0-10 where 0 is less      │
│                                                                                                      segmentation and 10 is more. Set to -1 to use        │
│                                                                                                      maxima_threshold instead. (This option is for        │
│                                                                                                      backwards compatibility with an old tool.)           │
│                                                                                                      [default: -1]                                        │
│    --maxima-threshold                                                     FLOAT RANGE [x>=0]         Controls segmentation level directly in mesmer, not  │
│                                                                                                      sure scaling via segmentation_level (lower values =  │
│                                                                                                      more cells, higher values = fewer cells). Provide a  │
│                                                                                                      value >0 to use this parameter.                      │
│                                                                                                      [default: 0.1]                                       │
│    --interior-threshold                                                   FLOAT                      Controls how conservative model is in distinguishing │
│                                                                                                      cell from background (lower values = larger cells,   │
│                                                                                                      higher values = smaller cells).                      │
│                                                                                                      [default: 0.3]                                       │
│    --maxima-smooth                                                        FLOAT RANGE [x>=0]         Controls what is considered a unique cell (lower     │
│                                                                                                      values = more separate cells, higher values = fewer  │
│                                                                                                      cells).                                              │
│                                                                                                      [default: 0]                                         │
│    --min-nuclei-area                                                      INTEGER RANGE [x>=0]       Minimum area of nuclei to keep.                      │
│                                                                                                      [default: 15]                                        │
│    --remove-cells-touching-border    --no-remove-cells-touching-border                               Whether to remove cells touching the border of the   │
│                                                                                                      image.                                               │
│                                                                                                      [default: remove-cells-touching-border]              │
│    --pixel-expansion                                                      INTEGER RANGE [x>=0]       Specify a manual pixel expansion after segmentation. │
│                                                                                                      [default: 0]                                         │
│    --padding                                                              INTEGER RANGE [x>=0]       Number of pixels to crop the image by before         │
│                                                                                                      segmentation.                                        │
│                                                                                                      [default: 0]                                         │
│    --install-completion                                                                              Install completion for the current shell.            │
│    --show-completion                                                                                 Show completion for the current shell, to copy it or │
│                                                                                                      customize the installation.                          │
│    --help                                                                                            Show this message and exit.                          │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

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
