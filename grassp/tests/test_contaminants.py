"""Tests for contaminant removal functionality."""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from anndata import AnnData

from grassp.preprocessing import contaminants


def make_test_data_with_crap(n_proteins=100, n_crap=10):
    """Helper: Create AnnData with real cRAP proteins.

    Parameters
    ----------
    n_proteins
        Total number of proteins in dataset
    n_crap
        Number of cRAP proteins to include

    Returns
    -------
    tuple of (AnnData, int)
        (test dataset, number of cRAP proteins included)
    """
    # Load actual cRAP database
    module_path = Path(contaminants.__file__).parent.parent
    crap_file = module_path / "datasets" / "external" / "cRAP.tsv"
    crap_df = pd.read_csv(crap_file, sep="\t")

    # Mix cRAP + synthetic proteins
    # Use distinct IDs that won't collide with real UniProt IDs
    crap_ids = crap_df['id'].tolist()[:n_crap]
    crap_entries = crap_df['entry_name'].tolist()[:n_crap]
    normal_ids = [f"TEST{str(i).zfill(5)}" for i in range(n_proteins - n_crap)]

    # Create AnnData with obs_names = all_ids
    X = np.random.randn(n_proteins, 10)
    obs = pd.DataFrame(index=crap_ids + normal_ids)
    obs['entry_name'] = crap_entries + [f"GENE{i}_HUMAN" for i in range(len(normal_ids))]

    return AnnData(X=X, obs=obs), n_crap


class TestRemoveCRAPProteins:
    """Test remove_cRAP_proteins function."""

    def test_remove_crap_basic(self):
        """Test basic removal with uniprot IDs."""
        adata, n_crap = make_test_data_with_crap(n_proteins=100, n_crap=10)
        original_shape = adata.shape

        contaminants.remove_cRAP_proteins(adata)

        assert adata.shape[0] == original_shape[0] - n_crap
        assert adata.shape[1] == original_shape[1]

    def test_remove_crap_entry_names(self):
        """Test removal with entry names."""
        adata, n_crap = make_test_data_with_crap(n_proteins=100, n_crap=10)
        original_shape = adata.shape

        contaminants.remove_cRAP_proteins(
            adata, id_column='entry_name', id_type='uniprot_entry_name'
        )

        assert adata.shape[0] == original_shape[0] - n_crap
        assert adata.shape[1] == original_shape[1]

    def test_remove_crap_custom_column(self):
        """Test using id_column parameter."""
        adata, n_crap = make_test_data_with_crap(n_proteins=100, n_crap=10)

        # Add a custom column with IDs
        adata.obs['protein_id'] = adata.obs_names

        contaminants.remove_cRAP_proteins(adata, id_column='protein_id')

        assert adata.shape[0] == 90

    def test_remove_crap_inplace_false(self):
        """Test inplace=False returns copy."""
        adata, n_crap = make_test_data_with_crap(n_proteins=100, n_crap=10)
        original_shape = adata.shape

        adata_filtered = contaminants.remove_cRAP_proteins(adata, inplace=False)

        # Original unchanged
        assert adata.shape == original_shape
        # Filtered copy has proteins removed
        assert adata_filtered.shape[0] == original_shape[0] - n_crap

    def test_remove_crap_no_matches(self):
        """Test warning when no cRAP proteins found."""
        # Create dataset with no cRAP proteins (use TEST prefix to avoid collisions)
        X = np.random.randn(50, 10)
        obs = pd.DataFrame(index=[f"TEST{str(i).zfill(5)}" for i in range(50)])
        adata = AnnData(X=X, obs=obs)

        with pytest.warns(UserWarning, match="No cRAP proteins found"):
            contaminants.remove_cRAP_proteins(adata)

        # No proteins removed
        assert adata.shape[0] == 50

    def test_remove_crap_isoforms(self):
        """Test handling IDs with isoform suffixes."""
        # Create dataset with isoform-suffixed IDs
        crap_file = (
            Path(contaminants.__file__).parent.parent / "datasets" / "external" / "cRAP.tsv"
        )
        crap_df = pd.read_csv(crap_file, sep="\t")

        # Use first 5 cRAP IDs with isoform suffixes
        crap_ids_with_isoforms = [f"{id}-1" for id in crap_df['id'].tolist()[:5]]
        normal_ids = [f"TEST{str(i).zfill(5)}" for i in range(45)]

        X = np.random.randn(50, 10)
        obs = pd.DataFrame(index=crap_ids_with_isoforms + normal_ids)
        adata = AnnData(X=X, obs=obs)

        contaminants.remove_cRAP_proteins(adata)

        # Should remove the 5 cRAP proteins (isoform suffixes stripped)
        assert adata.shape[0] == 45

    def test_remove_crap_invalid_column(self):
        """Test ValueError for invalid id_column."""
        adata, _ = make_test_data_with_crap(n_proteins=100, n_crap=10)

        with pytest.raises(ValueError, match="Column 'nonexistent' not found"):
            contaminants.remove_cRAP_proteins(adata, id_column='nonexistent')

    def test_remove_crap_invalid_id_type(self):
        """Test ValueError for invalid id_type."""
        adata, _ = make_test_data_with_crap(n_proteins=100, n_crap=10)

        with pytest.raises(ValueError, match="Invalid id_type"):
            contaminants.remove_cRAP_proteins(adata, id_type='invalid')

    def test_remove_crap_verbose_false(self):
        """Test verbose=False suppresses detailed output."""
        adata, _ = make_test_data_with_crap(n_proteins=100, n_crap=10)

        # Should not raise any errors with verbose=False
        contaminants.remove_cRAP_proteins(adata, verbose=False)
        assert adata.shape[0] == 90

    def test_remove_crap_verbose_true(self):
        """Test verbose=True prints removed IDs."""
        adata, _ = make_test_data_with_crap(n_proteins=100, n_crap=10)

        # Capture stdout to verify verbose output (just ensure it doesn't error)
        contaminants.remove_cRAP_proteins(adata, verbose=True)
        assert adata.shape[0] == 90


class TestRemoveContaminants:
    """Test remove_contaminants function (moved from simple.py)."""

    def test_remove_contaminants_basic(self):
        """Test basic contaminant removal."""
        # Create test data with contaminant markers
        X = np.random.randn(100, 10)
        obs = pd.DataFrame({'Potential contaminant': ['+'] * 10 + [''] * 90})
        adata = AnnData(X=X, obs=obs)

        contaminants.remove_contaminants(
            adata, filter_columns=['Potential contaminant'], filter_value='+'
        )

        assert adata.shape[0] == 90

    def test_remove_contaminants_inplace_false(self):
        """Test inplace=False returns copy."""
        X = np.random.randn(100, 10)
        obs = pd.DataFrame({'Potential contaminant': ['+'] * 10 + [''] * 90})
        adata = AnnData(X=X, obs=obs)

        adata_filtered = contaminants.remove_contaminants(
            adata, filter_columns=['Potential contaminant'], filter_value='+', inplace=False
        )

        # Original unchanged
        assert adata.shape[0] == 100
        # Filtered copy has contaminants removed
        assert adata_filtered.shape[0] == 90


class TestUpdateCRAPScript:
    """Test update_cRAP.py helper functions."""

    def test_parse_fasta_header_swissprot(self):
        """Test parsing SwissProt format."""
        from grassp.datasets.marker_curation.update_cRAP import parse_fasta_header

        entry_name = parse_fasta_header(">sp|ALBU_BOVIN|")
        assert entry_name == "ALBU_BOVIN"

    def test_parse_fasta_header_trembl(self):
        """Test parsing TrEMBL format."""
        from grassp.datasets.marker_curation.update_cRAP import parse_fasta_header

        entry_name = parse_fasta_header(">tr|K1C10_HUMAN|")
        assert entry_name == "K1C10_HUMAN"

    def test_parse_fasta_header_malformed(self):
        """Test handling malformed headers."""
        from grassp.datasets.marker_curation.update_cRAP import parse_fasta_header

        entry_name = parse_fasta_header(">malformed")
        assert entry_name is None

        entry_name = parse_fasta_header(">")
        assert entry_name is None
