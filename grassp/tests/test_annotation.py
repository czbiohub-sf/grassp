import numpy as np
import pandas as pd
import pytest

from anndata import AnnData

from grassp.preprocessing import annotation


def make_test_anndata_with_uniprot_ids():
    """Create a simple AnnData object with UniProt IDs for testing."""
    X = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12],
            [13, 14, 15],
        ],
        dtype=float,
    )
    # Use some real UniProt IDs that exist in hsap_markers.tsv
    obs = pd.DataFrame(
        index=["P17050", "P07858", "O43676", "P05386", "P12814"],  # Real human proteins
    )
    var = pd.DataFrame(index=["sample1", "sample2", "sample3"])
    adata = AnnData(X=X, obs=obs, var=var)
    return adata


def test_add_markers_all_authors():
    """Test adding all marker columns."""
    adata = make_test_anndata_with_uniprot_ids()

    # Add all markers
    annotation.add_markers(adata, species="hsap", authors=None)

    # Check that marker columns were added
    expected_columns = [
        "lilley",
        "christopher",
        "geladaki",
        "itzhak",
        "villaneuva",
        "hein2024_component",
        "hein2024_gt_component",
    ]
    for col in expected_columns:
        assert col in adata.obs.columns, f"Expected column {col} not found"

    # Check that at least some proteins were annotated
    n_annotated = adata.obs["lilley"].notna().sum()
    assert n_annotated > 0, "No proteins were annotated"


def test_add_markers_specific_authors():
    """Test adding specific author columns."""
    adata = make_test_anndata_with_uniprot_ids()

    # Add only specific authors
    annotation.add_markers(adata, species="hsap", authors=["lilley", "christopher"])

    # Check that requested columns were added
    assert "lilley" in adata.obs.columns
    assert "christopher" in adata.obs.columns

    # Check that other columns were NOT added
    assert "geladaki" not in adata.obs.columns
    assert "itzhak" not in adata.obs.columns


def test_add_markers_single_author_as_string():
    """Test adding a single author as string (not list)."""
    adata = make_test_anndata_with_uniprot_ids()

    # Add single author as string
    annotation.add_markers(adata, species="hsap", authors="lilley")

    # Check that column was added
    assert "lilley" in adata.obs.columns

    # Check that at least some proteins were annotated
    n_annotated = adata.obs["lilley"].notna().sum()
    assert n_annotated > 0, "No proteins were annotated"


def test_add_markers_missing_author_warning():
    """Test that warning is raised when some authors are not found."""
    adata = make_test_anndata_with_uniprot_ids()

    # Request one valid and one invalid author
    with pytest.warns(UserWarning, match="Some authors not found"):
        annotation.add_markers(adata, species="hsap", authors=["lilley", "nonexistent_author"])

    # Check that valid author was still added
    assert "lilley" in adata.obs.columns


def test_add_markers_all_authors_missing_error():
    """Test that error is raised when all requested authors are not found."""
    adata = make_test_anndata_with_uniprot_ids()

    # Request only invalid authors
    with pytest.raises(ValueError, match="None of the specified authors"):
        annotation.add_markers(adata, species="hsap", authors=["nonexistent1", "nonexistent2"])


def test_add_markers_invalid_species():
    """Test that error is raised for invalid species."""
    adata = make_test_anndata_with_uniprot_ids()

    with pytest.raises(FileNotFoundError, match="Species not found"):
        annotation.add_markers(adata, species="invalid_species")


def test_add_markers_with_protein_id_column():
    """Test using a specific column for protein IDs instead of obs_names."""
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
    obs = pd.DataFrame(
        {
            "uniprot_id": ["P17050", "P07858", "O43676"],
            "gene_name": ["GENE1", "GENE2", "GENE3"],
        },
        index=["protein1", "protein2", "protein3"],
    )
    var = pd.DataFrame(index=["sample1", "sample2", "sample3"])
    adata = AnnData(X=X, obs=obs, var=var)

    # Use the uniprot_id column
    annotation.add_markers(adata, species="hsap", uniprot_id_column="uniprot_id")

    # Check that marker columns were added
    assert "lilley" in adata.obs.columns

    # Check that at least some proteins were annotated
    n_annotated = adata.obs["lilley"].notna().sum()
    assert n_annotated > 0, "No proteins were annotated"


def test_add_markers_invalid_protein_id_column():
    """Test that error is raised for invalid protein_id_column."""
    adata = make_test_anndata_with_uniprot_ids()

    with pytest.raises(ValueError, match="Column .* not found"):
        annotation.add_markers(adata, species="hsap", uniprot_id_column="nonexistent_column")


def test_add_markers_annotations_content():
    """Test that actual annotations are present and correct."""
    adata = make_test_anndata_with_uniprot_ids()

    annotation.add_markers(adata, species="hsap", authors=["lilley"])

    # Check that we have some non-null annotations
    annotations = adata.obs["lilley"].dropna()
    assert len(annotations) > 0, "No annotations found"

    # Annotations should be strings (subcellular locations)
    assert all(isinstance(val, str) for val in annotations), "Annotations should be strings"


def test_add_markers_other_species():
    """Test with a different species (mouse)."""
    # Create test data with mouse UniProt IDs
    X = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
    # Use some mouse UniProt IDs from mmus_markers.tsv
    obs = pd.DataFrame(index=["A2AJ15", "A2ATU0"])
    var = pd.DataFrame(index=["sample1", "sample2", "sample3"])
    adata = AnnData(X=X, obs=obs, var=var)

    # Add mouse markers
    annotation.add_markers(adata, species="mmus")

    # Check that marker columns exist (mouse has fewer authors)
    assert "lilley" in adata.obs.columns

    # Check that proteins were annotated
    n_annotated = adata.obs["lilley"].notna().sum()
    assert n_annotated > 0, "No mouse proteins were annotated"


def test_add_markers_no_matching_proteins():
    """Test behavior when no proteins match the marker file."""
    # Create test data with fake UniProt IDs
    X = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
    obs = pd.DataFrame(index=["FAKE001", "FAKE002"])
    var = pd.DataFrame(index=["sample1", "sample2", "sample3"])
    adata = AnnData(X=X, obs=obs, var=var)

    # Add markers - should not error but should report 0 matches
    annotation.add_markers(adata, species="hsap", authors=["lilley"])

    # Check that column was added but all values are NaN
    assert "lilley" in adata.obs.columns
    assert adata.obs["lilley"].isna().all(), "Expected all NaN for non-matching proteins"


def test_add_markers_preserves_existing_obs():
    """Test that existing .obs columns are preserved."""
    adata = make_test_anndata_with_uniprot_ids()

    # Add some existing columns
    adata.obs["existing_column"] = ["A", "B", "C", "D", "E"]

    # Add markers
    annotation.add_markers(adata, species="hsap", authors=["lilley"])

    # Check that existing column is still there
    assert "existing_column" in adata.obs.columns
    assert list(adata.obs["existing_column"]) == ["A", "B", "C", "D", "E"]


def test_add_markers_creates_categorical():
    """Test that marker columns are categorical."""
    adata = make_test_anndata_with_uniprot_ids()
    annotation.add_markers(adata, species="hsap", authors=["lilley"])

    assert pd.api.types.is_categorical_dtype(adata.obs["lilley"])


def test_add_markers_adds_colors_to_uns():
    """Test that colors are added to .uns."""
    adata = make_test_anndata_with_uniprot_ids()
    annotation.add_markers(adata, species="hsap", authors=["lilley"])

    assert "lilley_colors" in adata.uns
    assert isinstance(adata.uns["lilley_colors"], list)


def test_add_markers_colors_match_categories():
    """Test that color list length matches number of categories."""
    adata = make_test_anndata_with_uniprot_ids()
    annotation.add_markers(adata, species="hsap", authors=["lilley"])

    n_categories = len(adata.obs["lilley"].cat.categories)
    n_colors = len(adata.uns["lilley_colors"])
    assert n_categories == n_colors


def test_add_markers_no_colors_option():
    """Test add_colors=False option."""
    adata = make_test_anndata_with_uniprot_ids()
    annotation.add_markers(adata, species="hsap", authors=["lilley"], add_colors=False)

    assert "lilley_colors" not in adata.uns
    # Should still be categorical
    assert pd.api.types.is_categorical_dtype(adata.obs["lilley"])


def test_add_markers_colors_are_valid_hex():
    """Test that all colors are valid hex codes."""
    import re

    adata = make_test_anndata_with_uniprot_ids()
    annotation.add_markers(adata, species="hsap", authors=["lilley"])

    hex_pattern = re.compile(r'^#[0-9A-Fa-f]{6}$')
    for color in adata.uns["lilley_colors"]:
        assert hex_pattern.match(color), f"Invalid hex color: {color}"


def test_add_markers_multiple_authors_with_colors():
    """Test that colors are added for multiple authors."""
    adata = make_test_anndata_with_uniprot_ids()
    annotation.add_markers(adata, species="hsap", authors=["lilley", "christopher"])

    # Check both are categorical
    assert pd.api.types.is_categorical_dtype(adata.obs["lilley"])
    assert pd.api.types.is_categorical_dtype(adata.obs["christopher"])

    # Check both have colors
    assert "lilley_colors" in adata.uns
    assert "christopher_colors" in adata.uns


def test_add_markers_colors_use_predefined():
    """Test that predefined colors from MARKER_COLORS are used."""
    adata = make_test_anndata_with_uniprot_ids()
    annotation.add_markers(adata, species="hsap", authors=["lilley"])

    # Check that at least one color matches the predefined dictionary
    # We know "Cytosol" exists in markers and has color "#1B9E9E"
    categories = adata.obs["lilley"].cat.categories
    colors = adata.uns["lilley_colors"]

    if "Cytosol" in categories:
        cytosol_idx = list(categories).index("Cytosol")
        assert colors[cytosol_idx] == "#1B9E9E", "Predefined color for Cytosol not used"
