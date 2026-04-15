"""
Tests for mesmer_segment.io — TIFF loading and OME-XML parsing.
"""
import pytest
from tifffile import TiffFile
from xarray import DataArray

from conftest import CHANNEL_NAMES, FOV_SIZE_MICRONS, FRAME_SIZE, MPP
from mesmer_segment.io import (
    extract_channel_names,
    extract_microns_per_pixel,
    has_valid_ome_xml,
    tiff_to_xarray,
)

# Minimal OME-XML used to unit-test the XML parsing functions directly
_OME_XML = """<?xml version="1.0" encoding="UTF-8"?>
<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">
  <Image ID="Image:0">
    <Pixels ID="Pixels:0" PhysicalSizeX="0.5" PhysicalSizeY="0.5">
      <Channel ID="Channel:0:0" Name="nuclear"/>
      <Channel ID="Channel:0:1" Name="membrane"/>
    </Pixels>
  </Image>
</OME>"""


# ---------------------------------------------------------------------------
# tiff_to_xarray
# ---------------------------------------------------------------------------

class TestTiffToXarray:

    def test_returns_dataarray(self, mibi_tiff):
        assert isinstance(tiff_to_xarray(mibi_tiff), DataArray)

    # --- MIBI path ---

    def test_mibi_shape(self, mibi_tiff, synthetic_data):
        da = tiff_to_xarray(mibi_tiff)
        assert da.shape == synthetic_data.shape

    def test_mibi_dims(self, mibi_tiff):
        da = tiff_to_xarray(mibi_tiff)
        assert da.dims == ("C", "Y", "X")

    def test_mibi_channel_names(self, mibi_tiff):
        da = tiff_to_xarray(mibi_tiff)
        assert list(da.coords["C"].values) == CHANNEL_NAMES

    def test_mibi_fov_attrs(self, mibi_tiff):
        da = tiff_to_xarray(mibi_tiff)
        assert da.attrs["fov_size"] == FOV_SIZE_MICRONS
        assert da.attrs["frame_size"] == FRAME_SIZE

    # --- OME-TIFF path ---

    def test_ome_shape(self, ome_tiff, synthetic_data):
        da = tiff_to_xarray(ome_tiff)
        assert da.shape == synthetic_data.shape

    def test_ome_channel_names(self, ome_tiff):
        da = tiff_to_xarray(ome_tiff)
        assert list(da.coords["C"].values) == CHANNEL_NAMES

    def test_ome_mpp(self, ome_tiff):
        da = tiff_to_xarray(ome_tiff)
        assert da.attrs["microns_per_pixel"] == pytest.approx(MPP)

    # --- Plain TIFF fallback path ---

    def test_plain_shape(self, plain_tiff, synthetic_data):
        da = tiff_to_xarray(plain_tiff)
        assert da.shape == synthetic_data.shape

    def test_plain_fallback_channel_names(self, plain_tiff):
        da = tiff_to_xarray(plain_tiff)
        assert list(da.coords["C"].values) == ["Channel_0", "Channel_1"]

    def test_plain_no_mpp(self, plain_tiff):
        da = tiff_to_xarray(plain_tiff)
        assert "microns_per_pixel" not in da.attrs


# ---------------------------------------------------------------------------
# has_valid_ome_xml
# ---------------------------------------------------------------------------

class TestHasValidOmeXml:

    def test_true_for_ome_tiff(self, ome_tiff):
        with TiffFile(ome_tiff) as tif:
            assert has_valid_ome_xml(tif) is True

    def test_true_for_mibi_tiff(self, mibi_tiff):
        with TiffFile(mibi_tiff) as tif:
            assert has_valid_ome_xml(tif) is True

    def test_false_for_plain_tiff(self, plain_tiff):
        with TiffFile(plain_tiff) as tif:
            assert has_valid_ome_xml(tif) is False


# ---------------------------------------------------------------------------
# extract_channel_names / extract_microns_per_pixel (OME-XML parsing)
# ---------------------------------------------------------------------------

class TestExtractChannelNames:

    def test_extracts_names_in_order(self):
        names = extract_channel_names(_OME_XML)
        assert names == ["nuclear", "membrane"]

    def test_missing_name_returns_unknown(self):
        xml = _OME_XML.replace('Name="nuclear"', "")
        names = extract_channel_names(xml)
        assert names[0] == "Unknown"

    def test_no_image_tag_raises(self):
        bad_xml = '<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06"/>'
        with pytest.raises(ValueError, match="No Image tag"):
            extract_channel_names(bad_xml)

    def test_no_pixels_tag_raises(self):
        bad_xml = """<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">
          <Image ID="Image:0"/>
        </OME>"""
        with pytest.raises(ValueError, match="No Pixels tag"):
            extract_channel_names(bad_xml)


class TestExtractMicronsPerPixel:

    def test_returns_correct_value(self):
        mpp = extract_microns_per_pixel(_OME_XML)
        assert mpp == pytest.approx(0.5)

    def test_unequal_xy_raises(self):
        xml = _OME_XML.replace('PhysicalSizeY="0.5"', 'PhysicalSizeY="1.0"')
        with pytest.raises(ValueError, match="not equal"):
            extract_microns_per_pixel(xml)

    def test_missing_physical_size_raises(self):
        xml = _OME_XML.replace(' PhysicalSizeX="0.5" PhysicalSizeY="0.5"', "")
        with pytest.raises(ValueError, match="not found"):
            extract_microns_per_pixel(xml)
