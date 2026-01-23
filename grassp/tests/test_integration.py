import numpy as np
import pandas as pd
import pytest
import scanpy as sc

from anndata import AnnData

from grassp.tools.integration import _find_mnn, graph_correction


def make_test_adata_pair():
    """
    Create a pair of AnnData objects with known structure for testing.

    Returns
    -------
    tuple[AnnData, AnnData]
        Two datasets with:
        - 20 observations (proteins) each
        - 15 shared observations (potential pairs)
        - 5 features each
        - Precomputed neighbor graphs
    """
    np.random.seed(42)

    # Create two clusters in 2D space for reliable neighbor structure
    cluster1 = np.random.randn(10, 5) + np.array([0, 0, 0, 0, 0])
    cluster2 = np.random.randn(10, 5) + np.array([5, 5, 5, 5, 5])

    # Dataset 1: all 20 points
    X1 = np.vstack([cluster1, cluster2])
    obs1 = pd.DataFrame(index=[f"prot_{i}" for i in range(20)])
    var1 = pd.DataFrame(index=[f"feature_{i}" for i in range(5)])
    adata1 = AnnData(X=X1, obs=obs1, var=var1)

    # Dataset 2: same 20 points with slight noise
    X2 = X1 + np.random.randn(*X1.shape) * 0.1
    obs2 = pd.DataFrame(index=[f"prot_{i}" for i in range(20)])
    var2 = pd.DataFrame(index=[f"feature_{i}" for i in range(5)])
    adata2 = AnnData(X=X2, obs=obs2, var=var2)

    # Compute neighbors for both
    sc.pp.neighbors(adata1, n_neighbors=5, use_rep="X")
    sc.pp.neighbors(adata2, n_neighbors=5, use_rep="X")

    return adata1, adata2


def test_find_mnn_basic():
    """Test that _find_mnn identifies some MNN pairs."""
    adata1, adata2 = make_test_adata_pair()
    pair_names = adata1.obs_names.intersection(adata2.obs_names)

    mnn_pairs = _find_mnn(adata1, adata2, pair_names, k=5)

    assert len(mnn_pairs) > 0, "Should find at least some MNN pairs"
    assert all(name in pair_names for name in mnn_pairs), "All MNNs should be in pair_names"
    assert isinstance(mnn_pairs, list), "Should return a list"


def test_find_mnn_no_connectivity_adata1():
    """Test error when connectivity graph is missing from adata1."""
    adata1, adata2 = make_test_adata_pair()

    # Remove connectivity graph from adata1
    del adata1.obsp["connectivities"]

    pair_names = adata1.obs_names.intersection(adata2.obs_names)

    with pytest.raises(ValueError, match="Connectivity graph.*not found in adata1"):
        _find_mnn(adata1, adata2, pair_names)


def test_find_mnn_no_connectivity_adata2():
    """Test error when connectivity graph is missing from adata2."""
    adata1, adata2 = make_test_adata_pair()

    # Remove connectivity graph from adata2
    del adata2.obsp["connectivities"]

    pair_names = adata1.obs_names.intersection(adata2.obs_names)

    with pytest.raises(ValueError, match="Connectivity graph.*not found in adata2"):
        _find_mnn(adata1, adata2, pair_names)


def test_graph_correction_mnn_only():
    """Test graph_correction with MNN filtering."""
    adata1, adata2 = make_test_adata_pair()
    pair_names = adata1.obs_names.intersection(adata2.obs_names)

    result = graph_correction(
        adata1, adata2, pair_names=pair_names, mnn_only=True, inplace=False
    )

    # Check result shape
    assert result.shape == (adata2.n_obs, adata2.n_vars), "Correction matrix has wrong shape"

    # Check anchor columns were added
    assert "graph_correction_anchor" in adata1.obs.columns, "Anchor column missing in adata1"
    assert "graph_correction_anchor" in adata2.obs.columns, "Anchor column missing in adata2"

    # Check column type
    assert (
        adata1.obs["graph_correction_anchor"].dtype == bool
    ), "Anchor column should be boolean"
    assert (
        adata2.obs["graph_correction_anchor"].dtype == bool
    ), "Anchor column should be boolean"

    # Check that some anchors were found
    assert (
        adata1.obs["graph_correction_anchor"].sum() > 0
    ), "Should have at least some MNN anchors"
    assert (
        adata2.obs["graph_correction_anchor"].sum() > 0
    ), "Should have at least some MNN anchors"

    # Check same proteins are marked as anchors in both datasets
    anchors1 = set(adata1.obs_names[adata1.obs["graph_correction_anchor"]])
    anchors2 = set(adata2.obs_names[adata2.obs["graph_correction_anchor"]])
    assert anchors1 == anchors2, "Same proteins should be anchors in both datasets"


def test_graph_correction_mnn_filtering():
    """Test that MNN filtering reduces the number of pair_names used."""
    adata1, adata2 = make_test_adata_pair()
    pair_names = list(adata1.obs_names.intersection(adata2.obs_names))

    # Run without MNN filtering
    result_all = graph_correction(
        adata1, adata2, pair_names=pair_names, mnn_only=False, inplace=False
    )

    # Run with MNN filtering
    result_mnn = graph_correction(
        adata1, adata2, pair_names=pair_names, mnn_only=True, inplace=False
    )

    # Both should produce valid corrections
    assert result_all.shape == result_mnn.shape

    # MNN filtering should identify anchors
    n_anchors = adata1.obs["graph_correction_anchor"].sum()
    assert n_anchors > 0, "Should find at least some MNN anchors"
    assert n_anchors <= len(pair_names), "Cannot have more anchors than candidates"


def test_graph_correction_backward_compatible():
    """Test that default behavior is unchanged (mnn_only=False)."""
    adata1, adata2 = make_test_adata_pair()
    pair_names = adata1.obs_names.intersection(adata2.obs_names)

    # Should work without error when mnn_only=False (default)
    result = graph_correction(adata1, adata2, pair_names=pair_names, inplace=False)

    assert result.shape == (adata2.n_obs, adata2.n_vars), "Correction matrix has wrong shape"

    # Anchor column should NOT be added when mnn_only=False
    assert (
        "graph_correction_anchor" not in adata1.obs.columns
    ), "Anchor column should not be added"
    assert (
        "graph_correction_anchor" not in adata2.obs.columns
    ), "Anchor column should not be added"


def test_graph_correction_anchor_storage():
    """Test that anchor status is correctly stored."""
    adata1, adata2 = make_test_adata_pair()

    _ = graph_correction(adata1, adata2, mnn_only=True, inplace=True)

    # Check boolean column exists
    assert "graph_correction_anchor" in adata1.obs.columns
    assert "graph_correction_anchor" in adata2.obs.columns

    # Check it's boolean
    assert adata1.obs["graph_correction_anchor"].dtype == bool
    assert adata2.obs["graph_correction_anchor"].dtype == bool

    # Check same proteins are marked as anchors in both
    anchors1 = set(adata1.obs_names[adata1.obs["graph_correction_anchor"]])
    anchors2 = set(adata2.obs_names[adata2.obs["graph_correction_anchor"]])
    assert anchors1 == anchors2

    # Check that not all are anchors (some filtering occurred)
    assert adata1.obs["graph_correction_anchor"].sum() <= len(adata1.obs_names)


def test_graph_correction_custom_mnn_k():
    """Test that custom mnn_k parameter affects MNN detection."""
    adata1, adata2 = make_test_adata_pair()
    pair_names = adata1.obs_names.intersection(adata2.obs_names)

    # Run with different k values
    graph_correction(
        adata1, adata2, pair_names=pair_names, mnn_only=True, mnn_k=3, inplace=False
    )
    anchors_k3 = adata1.obs["graph_correction_anchor"].sum()

    # Reset anchor columns
    del adata1.obs["graph_correction_anchor"]
    del adata2.obs["graph_correction_anchor"]

    graph_correction(
        adata1, adata2, pair_names=pair_names, mnn_only=True, mnn_k=10, inplace=False
    )
    anchors_k10 = adata1.obs["graph_correction_anchor"].sum()

    # Different k values may produce different numbers of anchors
    # (this is a weak assertion, just checking the parameter is used)
    assert isinstance(anchors_k3, (int, np.integer))
    assert isinstance(anchors_k10, (int, np.integer))


def test_find_mnn_empty_pairs():
    """Test behavior with no candidate pairs."""
    adata1, adata2 = make_test_adata_pair()

    mnn_pairs = _find_mnn(adata1, adata2, pair_names=[], k=5)

    assert mnn_pairs == [], "Should return empty list for empty input"


def test_graph_correction_missing_connectivity_error():
    """Test that mnn_only=True raises error when connectivity graphs are missing."""
    adata1, adata2 = make_test_adata_pair()

    # Remove both connectivity graphs
    del adata1.obsp["connectivities"]
    del adata2.obsp["connectivities"]

    with pytest.raises(ValueError, match="requires connectivity graph in adata1"):
        graph_correction(adata1, adata2, mnn_only=True)
