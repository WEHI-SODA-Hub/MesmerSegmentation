# Mesmer Segmentation

This repository provides a command-line interface for Mesmer segmentation of MIBI TIFFs.

## Installation

Install this repo. You can do so using `pip install git+https://github.com/WEHI-SODA-Hub/MibiSegmentation.git` or by cloning the repo and running `pip install .`.

## CLI Usage

```
 Usage: mesmer-segment [OPTIONS] MIBI_TIFF                                                                                                                                           
                                                                                                                                                                                     
 Segments a MIBI TIFF using Mesmer, and prints the result to stdout.                                                                                                                 
                                                                                                                                                                                     
╭─ Arguments ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *    mibi_tiff      PATH  Path to the MIBI TIFF input file [default: None] [required]                                                                                             │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *  --nuclear-channel           TEXT  Name of the nuclear channel [default: None] [required]                                                                                       │
│ *  --membrane-channel          TEXT  Name of the membrane channel [default: None] [required]                                                                                      │
│    --install-completion              Install completion for the current shell.                                                                                                    │
│    --show-completion                 Show completion for the current shell, to copy it or customize the installation.                                                             │
│    --help                            Show this message and exit.                                                                                                                  │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

A typical usage example is:

```bash
mesmer-segment 11MH0285_2_0.tiff --nuclear-channel dsDNA --membrane-channel panCK > result.tiff
```
