"""
TIFF I/O: loading OME-TIFF, MIBI TIFF, and plain TIFF files into xarray
DataArrays. Non-OME TIFFs are handled transparently in-memory by extracting
whatever channel/resolution metadata is available.
"""
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from tifffile import TiffFile
from xarray import DataArray


# ---------------------------------------------------------------------------
# OME-XML helpers (used for well-formed OME-TIFFs)
# ---------------------------------------------------------------------------

def _get_pixels_tag(xml_str: str) -> ET.Element:
    """Parse OME-XML and return the Pixels element."""
    root = ET.fromstring(xml_str)
    ns = {'ome': root.tag.split('}')[0].strip('{')}
    image_tag = root.find('ome:Image', ns)
    if image_tag is None:
        raise ValueError("No Image tag found in the XML.")
    pixels_tag = image_tag.find('ome:Pixels', ns)
    if pixels_tag is None:
        raise ValueError("No Pixels tag found in the Image tag.")
    return pixels_tag


def extract_channel_names(xml_str: str) -> list[str]:
    """
    Extract all channel 'Name' attributes from OME-XML in order.
    Returns a list of channel names by index.
    """
    pixels_tag = _get_pixels_tag(xml_str)
    root = ET.fromstring(xml_str)
    ns = {'ome': root.tag.split('}')[0].strip('{')}
    channel_tags = pixels_tag.findall('ome:Channel', ns)
    return [ch.get('Name', 'Unknown') for ch in channel_tags]


def extract_microns_per_pixel(xml_str: str) -> float:
    """
    Parse OME-XML and return microns per pixel from PhysicalSizeX/Y.
    Raises ValueError if the values are missing or unequal.
    """
    pixels_tag = _get_pixels_tag(xml_str)
    physical_size_x = pixels_tag.get('PhysicalSizeX')
    physical_size_y = pixels_tag.get('PhysicalSizeY')

    if physical_size_x is None or physical_size_y is None:
        raise ValueError("PhysicalSizeX or PhysicalSizeY not found in the XML.")

    if physical_size_x != physical_size_y:
        raise ValueError("PhysicalSizeX and PhysicalSizeY are not equal.")

    return float(physical_size_x)


# ---------------------------------------------------------------------------
# Non-OME / generic TIFF helpers
# ---------------------------------------------------------------------------

def _local_tag_name(tag: str) -> str:
    """Return XML tag name without namespace prefix."""
    return tag.split("}", 1)[-1]


def has_valid_ome_xml(tif: TiffFile) -> bool:
    """
    Return True if the TIFF already carries metadata that tiff_to_xarray can
    parse natively (proper OME-TIFF with Channel elements, or MIBI TIFF with
    per-page JSON).
    """
    # OME-TIFF with Channel elements in the file-level OME metadata
    try:
        ome_xml = tif.ome_metadata
        if ome_xml:
            root = ET.fromstring(ome_xml)
            for elem in root.iter():
                if _local_tag_name(elem.tag) == "Channel":
                    return True
    except Exception:
        pass

    if tif.pages:
        desc = tif.pages[0].description  # type: ignore
        if desc:
            # OME-XML embedded in the first page description
            try:
                root = ET.fromstring(desc)
                for elem in root.iter():
                    if _local_tag_name(elem.tag) == "Channel":
                        return True
            except ET.ParseError:
                pass

            # MIBI TIFFs: per-page JSON with channel.target
            try:
                meta = json.loads(desc)
                if "channel.target" in meta:
                    return True
            except (json.JSONDecodeError, TypeError):
                pass

    return False


def _get_channel_names_generic(tif: TiffFile, n_channels: int) -> list[str]:
    """
    Extract channel names from a non-OME TIFF. Tries (in order):
      1. ImageJ metadata Labels
      2. Fallback to numbered names
    """
    if hasattr(tif, "imagej_metadata") and tif.imagej_metadata:
        labels = tif.imagej_metadata.get("Labels")
        if labels:
            return list(labels)[:n_channels]

    return [f"Channel_{i}" for i in range(n_channels)]


def _get_resolution_generic(tif: TiffFile) -> tuple[float, str] | tuple[None, None]:
    """
    Try to extract pixel size in microns from ImageJ or TIFF resolution tags.
    Returns (physical_size_µm, unit) or (None, None).
    """
    if hasattr(tif, "imagej_metadata") and tif.imagej_metadata:
        ij = tif.imagej_metadata
        if ij.get("unit") in ("um", "µm", "micron", "\u00B5m"):
            page = tif.pages[0]
            x_res = page.tags.get("XResolution")  # type: ignore[union-attr]
            if x_res is not None:
                val = x_res.value
                # XResolution is stored as a rational (numerator, denominator)
                # or occasionally as a bare float/int.
                if isinstance(val, tuple):
                    ppu: float = val[0] / val[1] if val[1] else 0.0
                else:
                    ppu = float(val)
                if ppu > 0:
                    return 1.0 / ppu, "µm"
    return None, None


# ---------------------------------------------------------------------------
# Main loading function
# ---------------------------------------------------------------------------

def tiff_to_xarray(tiff_path: Path) -> DataArray:
    """
    Load a TIFF file into an xarray DataArray with channel names and optional
    resolution metadata attached.

    Supports three TIFF variants — all handled in-memory, no temp files:
      - MIBI TIFF (per-page JSON metadata)
      - OME-TIFF (OME-XML in first-page description or file metadata)
      - Plain / ImageJ TIFF (channel names and resolution recovered from
        whatever metadata is available)
    """
    channel_names: list[str] = []
    attrs: dict[str, float] = {}
    channels: list[NDArray] = []

    with TiffFile(tiff_path) as tif:
        first_page = tif.pages[0]

        # --- MIBI TIFF ---
        is_mibi = False
        try:
            desc = json.loads(first_page.description)  # type: ignore
            is_mibi = "channel.target" in desc
        except (json.JSONDecodeError, TypeError):
            pass

        if is_mibi:
            for page in tif.pages:
                meta = json.loads(page.description)  # type: ignore
                channel_names.append(meta["channel.target"])
                if not attrs:
                    raw = meta["raw_description"]
                    attrs["fov_size"] = raw["fovSizeMicrons"]
                    attrs["frame_size"] = raw["frameSize"]
                channels.append(page.asarray())

        # --- OME-TIFF ---
        elif has_valid_ome_xml(tif):
            xml_str = first_page.description  # type: ignore
            channel_names = extract_channel_names(xml_str)
            attrs["microns_per_pixel"] = extract_microns_per_pixel(xml_str)
            for page in tif.pages:
                channels.append(page.asarray())

        # --- Plain / ImageJ TIFF (no valid OME-XML) ---
        else:
            data = tif.asarray()

            # Normalise to (C, Y, X)
            if data.ndim == 2:
                data = data[np.newaxis, ...]           # (1, Y, X)
            elif data.ndim == 3:
                # Interleaved (Y, X, C) — channels last and C < spatial dims
                if data.shape[2] < data.shape[0] and data.shape[2] < data.shape[1]:
                    data = np.transpose(data, (2, 0, 1))
            # data is now (C, Y, X)

            n_channels = data.shape[0]
            channel_names = _get_channel_names_generic(tif, n_channels)
            physical_size, _ = _get_resolution_generic(tif)
            if physical_size is not None:
                attrs["microns_per_pixel"] = physical_size

            # Build channels list from the already-loaded array (no re-read)
            for i in range(n_channels):
                channels.append(data[i])

    return DataArray(
        data=channels,
        dims=["C", "Y", "X"],
        coords={"C": channel_names},
        attrs=attrs,
    )
