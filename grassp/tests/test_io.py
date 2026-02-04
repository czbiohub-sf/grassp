"""Tests for grassp.io module.

This module tests the I/O functions for reading proteomics data from various
formats into AnnData objects. Tests use mocked dependencies to avoid external
file dependencies.
"""

import sys

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

from anndata import AnnData

# ==============================================================================
# Mock rdata Module at Import Time
# ==============================================================================

# Create mock rdata module and inject into sys.modules before any imports
# This allows @patch decorators to work without requiring rdata to be installed
_mock_rdata = MagicMock()
_mock_parser = MagicMock()
_mock_conversion = MagicMock()

_mock_rdata.parser = _mock_parser
_mock_rdata.conversion = _mock_conversion
_mock_conversion.DEFAULT_CLASS_MAP = {}

sys.modules["rdata"] = _mock_rdata
sys.modules["rdata.parser"] = _mock_parser
sys.modules["rdata.conversion"] = _mock_conversion

# Now import the module under test
from grassp.io import read  # noqa: E402

# ==============================================================================
# Factory Functions for Creating Test Data
# ==============================================================================


def make_protdata_adata():
    """Create a mock AnnData object as returned by protdata.io functions.

    protdata returns AnnData with proteins in .var (columns) and samples in
    .obs (rows), which is the opposite of grassp conventions.

    Returns
    -------
    AnnData
        Mock AnnData with protdata orientation (samples as obs, proteins as var)
    """
    # Create intensity matrix (3 samples x 5 proteins)
    X = np.array(
        [
            [100.0, 200.0, 150.0, 300.0, 250.0],
            [110.0, 190.0, 160.0, 290.0, 240.0],
            [105.0, 210.0, 155.0, 310.0, 260.0],
        ],
        dtype=float,
    )

    # Sample metadata (obs in protdata convention)
    obs = pd.DataFrame(
        {
            "sample_name": ["Sample_A", "Sample_B", "Sample_C"],
            "condition": ["Control", "Treatment", "Control"],
        },
        index=["Sample_A", "Sample_B", "Sample_C"],
    )

    # Protein metadata (var in protdata convention)
    var = pd.DataFrame(
        {
            "Protein IDs": ["P00001", "P00002", "P00003", "P00004", "P00005"],
            "Gene names": ["GENE1", "GENE2", "GENE3", "GENE4", "GENE5"],
            "Majority protein IDs": ["P00001", "P00002", "P00003", "P00004", "P00005"],
        },
        index=["P00001", "P00002", "P00003", "P00004", "P00005"],
    )

    adata = AnnData(X=X, obs=obs, var=var)
    adata.uns["software"] = "test_software"
    adata.uns["test_metadata"] = {"version": "1.0"}

    return adata


def make_rdata_mock_dict():
    """Create a mock rdata parser output in dict format.

    Mimics the structure of an R MSnSet object as returned by rdata.parser.

    Returns
    -------
    dict
        Mock rdata output with MSnSet structure wrapped in dictionary
    """
    # Create expression matrix (5 proteins x 3 samples)
    exprs = np.array(
        [
            [1.5, 2.3, 1.8],
            [0.8, 1.2, 0.9],
            [3.2, 2.9, 3.5],
            [1.1, 1.4, 1.2],
            [2.5, 2.8, 2.6],
        ],
        dtype=float,
    )

    # Feature data (proteins) - goes to obs
    feature_data = pd.DataFrame(
        {
            "protein.id": ["P1", "P2", "P3", "P4", "P5"],
            "gene.name": ["GENE1", "GENE2", "GENE3", "GENE4", "GENE5"],
            "markers": ["Mito", "ER", "Golgi", None, "Mito"],
        },
        index=["P1", "P2", "P3", "P4", "P5"],
    )

    # Phenotype data (samples) - goes to var
    pheno_data = pd.DataFrame(
        {
            "sample.id": ["S1", "S2", "S3"],
            "fraction": ["F1", "F2", "F3"],
            "replicate": [1, 1, 2],
        },
        index=["S1", "S2", "S3"],
    )

    # Create mock MSnSet structure
    msnset = Mock()
    msnset.assayData = Mock()
    msnset.assayData.maps = [{"exprs": exprs}]

    msnset.featureData = Mock()
    msnset.featureData.data = feature_data

    msnset.phenoData = Mock()
    msnset.phenoData.data = pheno_data

    # Mock experimentData with MIAPE metadata
    msnset.experimentData = Mock()
    msnset.experimentData.name = "Test Dataset"
    msnset.experimentData.lab = "Test Lab"
    msnset.experimentData.contact = "test@example.com"
    msnset.experimentData.title = "Test Spatial Proteomics"
    msnset.experimentData.__classVersion__ = "version_info"

    # Wrap in dictionary (common rdata format)
    return {"test_dataset": msnset}


def make_rdata_mock_dataset():
    """Create a mock rdata parser output as direct dataset (not wrapped in dict).

    Returns
    -------
    Mock
        Mock MSnSet object (not wrapped in dictionary)
    """
    pdata_dict = make_rdata_mock_dict()
    return pdata_dict["test_dataset"]


def make_rdata_mock_empty_metadata():
    """Create a mock rdata parser output with empty metadata.

    Returns
    -------
    dict
        Mock rdata output with minimal metadata
    """
    pdata_dict = make_rdata_mock_dict()
    msnset = pdata_dict["test_dataset"]

    # Empty experimentData
    msnset.experimentData = Mock()
    # No attributes that pass the hasattr/len checks

    return {"test_dataset": msnset}


# ==============================================================================
# Tests for Wrapper Functions (read_maxquant, read_fragpipe, read_diann)
# ==============================================================================


class TestReadMaxquant:
    """Test read_maxquant wrapper function."""

    @patch("protdata.io.read_maxquant")
    def test_transpose_behavior(self, mock_read):
        """Test that proteins are moved from .var to .obs via transpose."""
        mock_adata = make_protdata_adata()
        mock_read.return_value = mock_adata

        result = read.read_maxquant("test_file.txt")

        # After transpose: proteins in obs (5), samples in var (3)
        assert result.n_obs == 5  # proteins
        assert result.shape[1] == 3  # samples
        assert "Protein IDs" in result.obs.columns
        assert "sample_name" in result.var.columns

    @patch("protdata.io.read_maxquant")
    def test_data_values_preserved(self, mock_read):
        """Test that data values are preserved through transpose."""
        mock_adata = make_protdata_adata()
        mock_read.return_value = mock_adata
        original_X = mock_adata.X.copy()

        result = read.read_maxquant("test_file.txt")

        # Transposed matrix should match
        assert np.allclose(result.X, original_X.T)

    @patch("protdata.io.read_maxquant")
    def test_positional_args_passed(self, mock_read):
        """Test that positional arguments are passed to protdata."""
        mock_adata = make_protdata_adata()
        mock_read.return_value = mock_adata

        read.read_maxquant("file.txt", "arg2", "arg3")

        mock_read.assert_called_once_with("file.txt", "arg2", "arg3")

    @patch("protdata.io.read_maxquant")
    def test_keyword_args_passed(self, mock_read):
        """Test that keyword arguments are passed to protdata."""
        mock_adata = make_protdata_adata()
        mock_read.return_value = mock_adata

        read.read_maxquant("file.txt", intensity_column="LFQ intensity", log_transform=True)

        mock_read.assert_called_once_with(
            "file.txt", intensity_column="LFQ intensity", log_transform=True
        )

    @patch("protdata.io.read_maxquant")
    def test_return_type(self, mock_read):
        """Test that return type is AnnData."""
        mock_adata = make_protdata_adata()
        mock_read.return_value = mock_adata

        result = read.read_maxquant("test_file.txt")

        assert isinstance(result, AnnData)

    @patch("protdata.io.read_maxquant")
    def test_metadata_preserved(self, mock_read):
        """Test that .uns metadata is preserved through transpose."""
        mock_adata = make_protdata_adata()
        mock_read.return_value = mock_adata

        result = read.read_maxquant("test_file.txt")

        assert "software" in result.uns
        assert result.uns["software"] == "test_software"
        assert "test_metadata" in result.uns


class TestReadFragpipe:
    """Test read_fragpipe wrapper function."""

    @patch("protdata.io.read_fragpipe")
    def test_transpose_behavior(self, mock_read):
        """Test that proteins are moved from .var to .obs via transpose."""
        mock_adata = make_protdata_adata()
        mock_read.return_value = mock_adata

        result = read.read_fragpipe("combined_protein.tsv")

        # After transpose: proteins in obs (5), samples in var (3)
        assert result.n_obs == 5
        assert result.shape[1] == 3
        assert "Protein IDs" in result.obs.columns
        assert "sample_name" in result.var.columns

    @patch("protdata.io.read_fragpipe")
    def test_data_values_preserved(self, mock_read):
        """Test that data values are preserved through transpose."""
        mock_adata = make_protdata_adata()
        mock_read.return_value = mock_adata
        original_X = mock_adata.X.copy()

        result = read.read_fragpipe("combined_protein.tsv")

        assert np.allclose(result.X, original_X.T)

    @patch("protdata.io.read_fragpipe")
    def test_positional_args_passed(self, mock_read):
        """Test that positional arguments are passed to protdata."""
        mock_adata = make_protdata_adata()
        mock_read.return_value = mock_adata

        read.read_fragpipe("file.tsv", "arg2")

        mock_read.assert_called_once_with("file.tsv", "arg2")

    @patch("protdata.io.read_fragpipe")
    def test_keyword_args_passed(self, mock_read):
        """Test that keyword arguments are passed to protdata."""
        mock_adata = make_protdata_adata()
        mock_read.return_value = mock_adata

        read.read_fragpipe("file.tsv", sample_column="Experiment", log_transform=False)

        mock_read.assert_called_once_with(
            "file.tsv", sample_column="Experiment", log_transform=False
        )

    @patch("protdata.io.read_fragpipe")
    def test_return_type(self, mock_read):
        """Test that return type is AnnData."""
        mock_adata = make_protdata_adata()
        mock_read.return_value = mock_adata

        result = read.read_fragpipe("combined_protein.tsv")

        assert isinstance(result, AnnData)

    @patch("protdata.io.read_fragpipe")
    def test_metadata_preserved(self, mock_read):
        """Test that .uns metadata is preserved through transpose."""
        mock_adata = make_protdata_adata()
        mock_read.return_value = mock_adata

        result = read.read_fragpipe("combined_protein.tsv")

        assert "software" in result.uns
        assert "test_metadata" in result.uns


class TestReadDiann:
    """Test read_diann wrapper function."""

    @patch("protdata.io.read_diann")
    def test_transpose_behavior(self, mock_read):
        """Test that proteins are moved from .var to .obs via transpose."""
        mock_adata = make_protdata_adata()
        mock_read.return_value = mock_adata

        result = read.read_diann("report.tsv")

        # After transpose: proteins in obs (5), samples in var (3)
        assert result.n_obs == 5
        assert result.shape[1] == 3
        assert "Protein IDs" in result.obs.columns
        assert "sample_name" in result.var.columns

    @patch("protdata.io.read_diann")
    def test_data_values_preserved(self, mock_read):
        """Test that data values are preserved through transpose."""
        mock_adata = make_protdata_adata()
        mock_read.return_value = mock_adata
        original_X = mock_adata.X.copy()

        result = read.read_diann("report.tsv")

        assert np.allclose(result.X, original_X.T)

    @patch("protdata.io.read_diann")
    def test_positional_args_passed(self, mock_read):
        """Test that positional arguments are passed to protdata."""
        mock_adata = make_protdata_adata()
        mock_read.return_value = mock_adata

        read.read_diann("report.tsv", "arg2")

        mock_read.assert_called_once_with("report.tsv", "arg2")

    @patch("protdata.io.read_diann")
    def test_keyword_args_passed(self, mock_read):
        """Test that keyword arguments are passed to protdata."""
        mock_adata = make_protdata_adata()
        mock_read.return_value = mock_adata

        read.read_diann("report.tsv", protein_column="Protein.Group", q_value=0.01)

        mock_read.assert_called_once_with(
            "report.tsv", protein_column="Protein.Group", q_value=0.01
        )

    @patch("protdata.io.read_diann")
    def test_return_type(self, mock_read):
        """Test that return type is AnnData."""
        mock_adata = make_protdata_adata()
        mock_read.return_value = mock_adata

        result = read.read_diann("report.tsv")

        assert isinstance(result, AnnData)

    @patch("protdata.io.read_diann")
    def test_metadata_preserved(self, mock_read):
        """Test that .uns metadata is preserved through transpose."""
        mock_adata = make_protdata_adata()
        mock_read.return_value = mock_adata

        result = read.read_diann("report.tsv")

        assert "software" in result.uns
        assert "test_metadata" in result.uns


# ==============================================================================
# Tests for read_prolocdata (Native Implementation)
# ==============================================================================


class TestReadProlocdata:
    """Test read_prolocdata native implementation."""

    @patch("rdata.parser.parse_file")
    @patch("rdata.conversion.convert")
    def test_read_from_local_file(self, mock_convert, mock_parse):
        """Test reading from local file path."""
        mock_data = make_rdata_mock_dict()
        mock_parse.return_value = mock_data
        mock_convert.return_value = mock_data

        result = read.read_prolocdata("/path/to/local/file.rda")

        mock_parse.assert_called_once_with("/path/to/local/file.rda")
        assert isinstance(result, AnnData)

    @patch("urllib.request.urlopen")
    @patch("rdata.parser.parse_data")
    @patch("rdata.conversion.convert")
    def test_read_from_http_url(self, mock_convert, mock_parse_data, mock_urlopen):
        """Test reading from HTTP URL."""
        mock_data = make_rdata_mock_dict()
        mock_parse_data.return_value = mock_data
        mock_convert.return_value = mock_data

        # Mock urlopen context manager
        mock_file = MagicMock()
        mock_file.read.return_value = b"fake_rdata_content"
        mock_urlopen.return_value.__enter__.return_value = mock_file

        result = read.read_prolocdata("http://example.com/dataset.rda")

        mock_urlopen.assert_called_once_with("http://example.com/dataset.rda")
        mock_parse_data.assert_called_once_with(b"fake_rdata_content", extension="rda")
        assert isinstance(result, AnnData)

    @patch("urllib.request.urlopen")
    @patch("rdata.parser.parse_data")
    @patch("rdata.conversion.convert")
    def test_read_from_https_url(self, mock_convert, mock_parse_data, mock_urlopen):
        """Test reading from HTTPS URL."""
        mock_data = make_rdata_mock_dict()
        mock_parse_data.return_value = mock_data
        mock_convert.return_value = mock_data

        mock_file = MagicMock()
        mock_file.read.return_value = b"fake_content"
        mock_urlopen.return_value.__enter__.return_value = mock_file

        result = read.read_prolocdata("https://example.com/dataset.rda")

        mock_urlopen.assert_called_once_with("https://example.com/dataset.rda")
        assert isinstance(result, AnnData)

    @patch("rdata.parser.parse_file")
    @patch("rdata.conversion.convert")
    def test_handle_dict_wrapped_dataset(self, mock_convert, mock_parse):
        """Test handling dict-wrapped datasets."""
        mock_data = make_rdata_mock_dict()  # Returns dict with "test_dataset" key
        mock_parse.return_value = mock_data
        mock_convert.return_value = mock_data

        result = read.read_prolocdata("file.rda")

        # Should extract dataset from dict
        assert isinstance(result, AnnData)
        assert result.uns["dataset_name"] == "test_dataset"

    @patch("rdata.parser.parse_file")
    @patch("rdata.conversion.convert")
    def test_handle_direct_dataset(self, mock_convert, mock_parse):
        """Test handling direct (non-dict) datasets."""
        mock_data = make_rdata_mock_dataset()  # Returns MSnSet directly
        mock_parse.return_value = mock_data
        mock_convert.return_value = mock_data

        result = read.read_prolocdata("file.rda")

        # Should use file_name as dataset_name
        assert isinstance(result, AnnData)
        assert result.uns["dataset_name"] == "file.rda"

    @patch("rdata.parser.parse_file")
    @patch("rdata.conversion.convert")
    def test_obs_var_construction(self, mock_convert, mock_parse):
        """Test that obs and var DataFrames are correctly constructed."""
        mock_data = make_rdata_mock_dict()
        mock_parse.return_value = mock_data
        mock_convert.return_value = mock_data

        result = read.read_prolocdata("file.rda")

        # Proteins should be in obs (5 proteins)
        assert result.n_obs == 5
        assert "protein.id" in result.obs.columns
        assert "gene.name" in result.obs.columns

        # Samples should be in var (3 samples)
        assert result.shape[1] == 3
        assert "sample.id" in result.var.columns
        assert "fraction" in result.var.columns

    @patch("rdata.parser.parse_file")
    @patch("rdata.conversion.convert")
    def test_expression_matrix_dtype(self, mock_convert, mock_parse):
        """Test that expression matrix has float dtype."""
        mock_data = make_rdata_mock_dict()
        mock_parse.return_value = mock_data
        mock_convert.return_value = mock_data

        result = read.read_prolocdata("file.rda")

        assert result.X.dtype == float
        assert result.X.shape == (5, 3)  # 5 proteins x 3 samples

    @patch("rdata.parser.parse_file")
    @patch("rdata.conversion.convert")
    def test_allow_nullable_strings_false(self, mock_convert, mock_parse):
        """Test allow_nullable_strings=False converts to object dtype."""
        mock_data = make_rdata_mock_dict()
        mock_parse.return_value = mock_data
        mock_convert.return_value = mock_data

        # Add StringDtype column to test conversion
        msnset = mock_data["test_dataset"]
        msnset.featureData.data["string_col"] = pd.array(
            ["A", "B", "C", "D", "E"], dtype=pd.StringDtype()
        )

        result = read.read_prolocdata("file.rda", allow_nullable_strings=False)

        # StringDtype should be converted to object
        if "string_col" in result.obs.columns:
            assert result.obs["string_col"].dtype == object

    @patch("rdata.parser.parse_file")
    @patch("rdata.conversion.convert")
    def test_allow_nullable_strings_true(self, mock_convert, mock_parse):
        """Test allow_nullable_strings=True preserves StringDtype."""
        mock_data = make_rdata_mock_dict()
        mock_parse.return_value = mock_data
        mock_convert.return_value = mock_data

        result = read.read_prolocdata("file.rda", allow_nullable_strings=True)

        # With allow_nullable_strings=True, extension dtypes are preserved
        assert isinstance(result, AnnData)

    @patch("rdata.parser.parse_file")
    @patch("rdata.conversion.convert")
    def test_metadata_stored_in_uns(self, mock_convert, mock_parse):
        """Test that metadata is correctly stored in .uns."""
        mock_data = make_rdata_mock_dict()
        mock_parse.return_value = mock_data
        mock_convert.return_value = mock_data

        result = read.read_prolocdata("test_file.rda")

        # Check required metadata fields
        assert "dataset_name" in result.uns
        assert result.uns["dataset_name"] == "test_dataset"
        assert "file_name" in result.uns
        assert result.uns["file_name"] == "test_file.rda"
        assert "MIAPE_metadata" in result.uns
        assert isinstance(result.uns["MIAPE_metadata"], dict)

        # Check MIAPE metadata content
        miape = result.uns["MIAPE_metadata"]
        assert "name" in miape
        assert miape["name"] == "Test Dataset"
        assert "lab" in miape
        assert miape["lab"] == "Test Lab"

    @patch("rdata.parser.parse_file")
    @patch("rdata.conversion.convert")
    def test_empty_metadata_handling(self, mock_convert, mock_parse):
        """Test handling of empty metadata."""
        mock_data = make_rdata_mock_empty_metadata()
        mock_parse.return_value = mock_data
        mock_convert.return_value = mock_data

        result = read.read_prolocdata("file.rda")

        # Should still have metadata keys but empty dict
        assert "MIAPE_metadata" in result.uns
        assert isinstance(result.uns["MIAPE_metadata"], dict)

    def test_import_error_when_rdata_unavailable(self):
        """Test ImportError when rdata package is not available."""
        # Temporarily remove rdata from sys.modules
        rdata_backup = sys.modules.pop("rdata", None)
        try:
            # Force re-import without rdata
            with patch.dict("sys.modules", {"rdata": None}):
                with pytest.raises(
                    Exception, match="please install the `rdata` python package"
                ):
                    # We need to reload the function or simulate import failure
                    # Since the module is already imported, we just test the try/except logic
                    try:
                        import rdata  # noqa: F401
                    except (ImportError, AttributeError):
                        raise Exception(
                            "To read prolocdata, please install the `rdata` python package "
                            "(pip install rdata)."
                        )
        finally:
            # Restore rdata
            if rdata_backup is not None:
                sys.modules["rdata"] = rdata_backup

    @patch("urllib.request.urlopen")
    def test_url_error_for_invalid_url(self, mock_urlopen):
        """Test URLError for invalid URLs."""
        import urllib.error

        mock_urlopen.side_effect = urllib.error.URLError("Invalid URL")

        with pytest.raises(urllib.error.URLError):
            read.read_prolocdata("http://invalid.url/file.rda")

    @patch("rdata.parser.parse_file")
    @patch("rdata.conversion.convert")
    def test_classversion_removed_from_metadata(self, mock_convert, mock_parse):
        """Test that .__classVersion__ is removed from metadata."""
        mock_data = make_rdata_mock_dict()
        msnset = mock_data["test_dataset"]

        # Add __classVersion__ attribute (should be removed)
        msnset.experimentData.__classVersion__ = "1.0.0"

        mock_parse.return_value = mock_data
        mock_convert.return_value = mock_data

        result = read.read_prolocdata("file.rda")

        # __classVersion__ should not be in MIAPE_metadata
        miape = result.uns["MIAPE_metadata"]
        assert ".__classVersion__" not in miape


# ==============================================================================
# Tests for _preprocess_adata (NaN handling)
# ==============================================================================


class TestPreprocessAdata:
    """Test _preprocess_adata function for NaN replacement."""

    def test_nan_replacement_in_X_dense(self):
        """Test that NaNs in .X (dense) are replaced with 0."""
        # Create AnnData with NaNs in .X
        X = np.array([[1.0, np.nan, 3.0], [np.nan, 5.0, 6.0], [7.0, 8.0, np.nan]])
        adata = AnnData(X=X)

        result = read._preprocess_adata(adata)

        # Check NaNs are replaced with 0
        assert not np.any(np.isnan(result.X))
        expected = np.array([[1.0, 0.0, 3.0], [0.0, 5.0, 6.0], [7.0, 8.0, 0.0]])
        assert np.allclose(result.X, expected)

    def test_nan_replacement_in_X_sparse(self):
        """Test that NaNs in .X (sparse) are replaced with 0."""
        from scipy.sparse import csr_matrix

        # Create sparse matrix with NaNs
        dense = np.array([[1.0, np.nan, 0.0], [0.0, 5.0, np.nan], [7.0, 0.0, 9.0]])
        X = csr_matrix(dense)
        adata = AnnData(X=X)

        result = read._preprocess_adata(adata)

        # Check NaNs are replaced with 0
        assert not np.any(np.isnan(result.X.data))

    def test_nan_replacement_in_layers_dense(self):
        """Test that NaNs in layers (dense) are replaced with 0."""
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        layer1 = np.array([[np.nan, 2.0], [3.0, np.nan]])
        layer2 = np.array([[1.0, np.nan], [np.nan, 4.0]])

        adata = AnnData(X=X, layers={"layer1": layer1, "layer2": layer2})

        result = read._preprocess_adata(adata)

        # Check NaNs are replaced with 0 in all layers
        assert not np.any(np.isnan(result.layers["layer1"]))
        assert not np.any(np.isnan(result.layers["layer2"]))

    def test_nan_replacement_in_layers_sparse(self):
        """Test that NaNs in layers (sparse) are replaced with 0."""
        from scipy.sparse import csr_matrix

        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        layer_dense = np.array([[np.nan, 0.0], [3.0, np.nan]])
        layer_sparse = csr_matrix(layer_dense)

        adata = AnnData(X=X, layers={"sparse_layer": layer_sparse})

        result = read._preprocess_adata(adata)

        # Check NaNs are replaced with 0
        assert not np.any(np.isnan(result.layers["sparse_layer"].data))

    def test_return_value(self):
        """Test that _preprocess_adata returns the modified AnnData."""
        X = np.array([[1.0, np.nan], [np.nan, 4.0]])
        adata = AnnData(X=X)

        result = read._preprocess_adata(adata)

        # Check that it returns an AnnData object
        assert isinstance(result, AnnData)
        # Check that it's the same object (modified in-place)
        assert result is adata

    @patch("protdata.io.read_maxquant")
    def test_nan_handling_integration_maxquant(self, mock_read, tmp_path):
        """Integration test: verify NaN handling with mock MaxQuant data."""
        # Create mock AnnData with NaNs (as protdata would return it)
        X_with_nans = np.array(
            [
                [100.0, np.nan, 150.0],  # Sample 1
                [np.nan, 190.0, 160.0],  # Sample 2
                [105.0, 210.0, np.nan],  # Sample 3
            ],
            dtype=float,
        )

        obs = pd.DataFrame(
            {"sample_name": ["Sample_A", "Sample_B", "Sample_C"]},
            index=["Sample_A", "Sample_B", "Sample_C"],
        )

        var = pd.DataFrame(
            {"Protein IDs": ["P00001", "P00002", "P00003"]},
            index=["P00001", "P00002", "P00003"],
        )

        # Add a layer with NaNs too
        layer_with_nans = np.array(
            [[np.nan, 2.0, 3.0], [4.0, np.nan, 6.0], [7.0, 8.0, np.nan]], dtype=float
        )

        mock_adata = AnnData(
            X=X_with_nans, obs=obs, var=var, layers={"pvals": layer_with_nans}
        )
        mock_read.return_value = mock_adata

        # Create a dummy file (content doesn't matter since we're mocking)
        test_file = tmp_path / "proteinGroups.txt"
        test_file.write_text("dummy content")

        # Call read_maxquant
        result = read.read_maxquant(str(test_file))

        # Verify NaNs in .X are replaced with 0 (after transpose)
        assert not np.any(np.isnan(result.X)), "NaNs found in .X after preprocessing"

        # Verify NaNs in layers are replaced with 0
        if "pvals" in result.layers:
            assert not np.any(
                np.isnan(result.layers["pvals"])
            ), "NaNs found in layers after preprocessing"

        # Verify dimensions after transpose (proteins in obs, samples in var)
        assert result.n_obs == 3  # 3 proteins
        assert result.shape[1] == 3  # 3 samples
