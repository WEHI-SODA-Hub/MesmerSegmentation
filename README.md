# Mesmer Segmentation

This repository provides a command-line interface for Mesmer segmentation of MIBI TIFFs.

## Installation

Install this repo. You can do so using `pip install git+https://github.com/WEHI-SODA-Hub/MibiSegmentation.git` or by cloning the repo and running `pip install .`.

## CLI Usage

Don't forget to obtain and `export` an API key!

```
 Usage: mesmer-segment [OPTIONS] MIBI_TIFF

 Segments a MIBI TIFF using Mesmer, and prints the result to stdout. Note that you will need to obtain and export a DeepCell API key as explained here.

╭─ Arguments ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *    mibi_tiff      PATH  Path to the MIBI TIFF input file                                                                                               │
│                           [default: None]                                                                                                                │
│                           [required]                                                                                                                     │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *  --nuclear-channel                                                      TEXT                      Name of the nuclear channel                          │
│                                                                                                     [default: None]                                      │
│                                                                                                     [required]                                           │
│ *  --membrane-channel                                                     TEXT                      Name(s) of the membrane channels (can be repeated)   │
│                                                                                                     [default: None]                                      │
│                                                                                                     [required]                                           │
│    --compartment                                                          [whole-cell|nuclear]      Compartment to segment (whole-cell or nuclear)       │
│                                                                                                     [default: whole-cell]                                │
│    --combine-method                                                       [prod|max]                Method to use for combining channels (prod or max)   │
│                                                                                                     [default: prod]                                      │
│    --segmentation-level                                                   INTEGER RANGE [0<=x<=10]  Segmentation level between 0-10 where 0 is less      │
│                                                                                                     segmentation and 10 is more                          │
│                                                                                                     [default: 5]                                         │
│    --maxima-threshold                                                     FLOAT                     Controls segmentation level directly in mesmer, not  │
│                                                                                                     sure scaling via segmentation_level (lower values =  │
│                                                                                                     more cells, higher values = fewer cells). Provide a  │
│                                                                                                     value >0 to use this parameter                       │
│                                                                                                     [default: -1]                                        │
│    --interior-threshold                                                   FLOAT                     Controls how conservative model is in distinguishing │
│                                                                                                     cell from background (lower values = larger cells,   │
│                                                                                                     higher values = smaller cells)                       │
│                                                                                                     [default: 0.3]                                       │
│    --maxima-smooth                                                        FLOAT RANGE [x>=0]        Controls what is considered a unique cell (lower     │
│                                                                                                     values = more separate cells, higher values = fewer  │
│                                                                                                     cells)                                               │
│                                                                                                     [default: 0]                                         │
│    --min-nuclei-area                                                      INTEGER RANGE [x>=0]      Minimum area of nuclei to keep                       │
│                                                                                                     [default: 15]                                        │
│    --remove-cells-touching-border    --no-remove-cells-touching-border                              Whether to remove cells touching the border of the   │
│                                                                                                     image                                                │
│                                                                                                     [default: remove-cells-touching-border]              │
│    --include-measurements            --no-include-measurements                                      Whether to include shape and marker measurements in  │
│                                                                                                     the output GeoJSON                                   │
│                                                                                                     [default: include-measurements]                      │
│    --pixel-expansion                                                      INTEGER                   Specify a manual pixel expansion after segmentation. │
│                                                                                                     [default: 0]                                         │
│    --padding                                                              INTEGER RANGE [x>=0]      Number of pixels to crop the image by before         │
│                                                                                                     segmentation                                         │
│                                                                                                     [default: 96]                                        │
│    --output-type                                                          [geojson|tiff]            Output format (geojson or tiff)                      │
│                                                                                                     [default: geojson]                                   │
│    --install-completion                                                                             Install completion for the current shell.            │
│    --show-completion                                                                                Show completion for the current shell, to copy it or │
│                                                                                                     customize the installation.                          │
│    --help                                                                                           Show this message and exit.                          │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

A typical usage example is:

```bash
mesmer-segment image.tiff \
    --nuclear-channel dsDNA \
    --membrane-channel panCK \
    --membrane-channel "MHC I" \
    --compartment whole-cell \
    --padding 0 \
    --output-type tiff > result.tiff
```
