"""Integration tests for grassp tools functions.

This module tests complete analytical workflows and individual tools
functions using miniaturized synthetic datasets with realistic structure.
Tests are organized by functional category.
"""

import warnings

import networkx as nx
import numpy as np
import pandas as pd
import pytest
import scanpy as sc

from anndata import AnnData

from grassp.preprocessing import simple
from grassp.tools import (
    clustering,
    integration,
    scoring,
)

# ==============================================================================
# Helper Functions for Creating Test Data with Structure
# ==============================================================================


def make_enriched_data_with_structure(
    n_proteins=200,
    n_samples=10,
    n_compartments=5,
    marker_fraction=0.3,
    add_neighbors=True,
    add_umap=True,
):
    """Generate synthetic enrichment data with compartment structure.

    Parameters
    ----------
    n_proteins : int
        Number of proteins
    n_samples : int
        Number of samples/conditions
    n_compartments : int
        Number of subcellular compartments
    marker_fraction : float
        Fraction of proteins to mark with known compartments
    add_neighbors : bool
        Whether to compute KNN graph
    add_umap : bool
        Whether to compute UMAP embedding

    Returns
    -------
    AnnData
        Enrichment-style dataset with compartment structure
    """
    np.random.seed(42)

    # Create compartment-specific protein groups
    compartment_names = [f"Compartment{i}" for i in range(1, n_compartments + 1)]
    proteins_per_comp = n_proteins // n_compartments

    # Generate enrichment data with compartment-specific patterns
    X = np.zeros((n_proteins, n_samples))

    for i, comp in enumerate(compartment_names):
        start_idx = i * proteins_per_comp
        end_idx = start_idx + proteins_per_comp if i < n_compartments - 1 else n_proteins

        # Create enrichment pattern for this compartment
        # Some samples are enriched for this compartment
        enriched_samples = [i, (i + 1) % n_samples]
        for j in range(start_idx, end_idx):
            # Base enrichment pattern
            pattern = np.random.normal(0, 0.5, n_samples)
            # Enrich in specific samples
            for s in enriched_samples:
                pattern[s] += np.random.normal(3, 0.5)
            X[j, :] = pattern

    # Add some noise
    X += np.random.normal(0, 0.2, X.shape)

    # Create obs DataFrame (proteins)
    protein_ids = [f"P{str(i).zfill(5)}" for i in range(n_proteins)]
    obs = pd.DataFrame(index=protein_ids)
    obs["Gene names"] = [f"GENE{i}" for i in range(n_proteins)]

    # Add markers for a fraction of proteins
    markers = np.array([None] * n_proteins, dtype=object)
    n_markers = int(n_proteins * marker_fraction)
    for i in range(n_markers):
        comp_idx = i % n_compartments
        markers[i] = compartment_names[comp_idx]
    obs["markers"] = pd.Categorical(markers)

    # Create var DataFrame (samples)
    var = pd.DataFrame(index=[f"Sample{i}" for i in range(n_samples)])

    adata = AnnData(X=X, obs=obs, var=var)

    # Add neighbors if requested
    if add_neighbors:
        simple.neighbors(adata, n_neighbors=min(15, n_proteins - 1))

    # Add UMAP if requested (requires neighbors)
    if add_umap and add_neighbors:
        sc.tl.umap(adata)

    return adata


def make_multi_dataset_for_integration(n_datasets=3, n_proteins=150, n_samples=8):
    """Generate multiple datasets for integration testing.

    Parameters
    ----------
    n_datasets : int
        Number of datasets to create
    n_proteins : int
        Number of proteins per dataset
    n_samples : int
        Number of samples per dataset

    Returns
    -------
    list[AnnData]
        List of synthetic datasets with overlapping proteins
    """
    np.random.seed(42)

    # Create a common set of proteins (70% overlap)
    n_common = int(n_proteins * 0.7)
    n_unique = n_proteins - n_common
    common_proteins = [f"P{str(i).zfill(5)}" for i in range(n_common)]

    datasets = []
    for d in range(n_datasets):
        # Each dataset has common + some unique proteins
        unique_proteins = [
            f"P{str(i + n_common + d * n_unique).zfill(5)}" for i in range(n_unique)
        ]
        all_proteins = common_proteins + unique_proteins

        # Generate data with dataset-specific shift
        X = np.random.normal(d * 2, 1.0, (len(all_proteins), n_samples))

        obs = pd.DataFrame(index=all_proteins)
        obs["Gene names"] = [f"GENE{i}" for i in range(len(all_proteins))]

        var = pd.DataFrame(index=[f"S{i}" for i in range(n_samples)])
        var["dataset"] = f"Dataset{d + 1}"

        adata = AnnData(X=X, obs=obs, var=var)
        datasets.append(adata)

    return datasets


# ==============================================================================
# Clustering Function Tests
# ==============================================================================


class TestClusteringFunctions:
    """Test clustering and annotation functions."""

    def test_markov_clustering_basic(self):
        """Test Markov clustering."""
        adata = make_enriched_data_with_structure(n_proteins=100, add_neighbors=True)

        clustering.markov_clustering(adata, resolution=1.5)

        assert "mc_cluster" in adata.obs.columns
        assert adata.obs["mc_cluster"].dtype.name == "category"
        # Should have multiple clusters
        assert adata.obs["mc_cluster"].nunique() > 1

    def test_markov_clustering_no_neighbors_error(self):
        """Test error when neighbors not computed."""
        adata = make_enriched_data_with_structure(n_proteins=100, add_neighbors=False)

        with pytest.raises(ValueError, match="Connectivities matrix not found"):
            clustering.markov_clustering(adata)

    def test_leiden_mito_sweep_basic(self):
        """Test Leiden mitochondria sweep."""
        adata = make_enriched_data_with_structure(n_proteins=100, add_neighbors=True)

        # Rename one compartment to mitochondria
        markers = adata.obs["markers"].copy()
        markers = markers.cat.rename_categories({"Compartment1": "mitochondria"})
        adata.obs["protein_ground_truth"] = markers

        # Run sweep (suppress print output)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            clustering.leiden_mito_sweep(
                adata,
                starting_resolution=0.5,
                resolution_increments=0.5,
                min_mito_fraction=0.8,
                increment_threshold=0.1,
                protein_ground_truth_column="protein_ground_truth",
            )

        assert "leiden" in adata.obs.columns
        assert "leiden" in adata.uns
        assert "mito_majority_fraction" in adata.uns["leiden"]

    def test_knn_annotation_basic(self):
        """Test KNN annotation propagation."""
        adata = make_enriched_data_with_structure(
            n_proteins=100, marker_fraction=0.3, add_neighbors=True
        )

        clustering.knn_annotation(
            adata,
            gt_col="markers",
            key_added="knn_annotation",
            min_probability=0.3,
        )

        assert "knn_annotation" in adata.obs.columns
        assert "knn_annotation_probabilities" in adata.obsm
        assert "knn_annotation_probability" in adata.obs.columns
        # Should have annotated proteins
        assert adata.obs["knn_annotation"].notna().sum() > 0
        # Probabilities should be within valid range
        assert (adata.obs["knn_annotation_probability"] <= 1.0).all()

    def test_knn_annotation_fix_markers(self):
        """Test KNN annotation with fixed markers."""
        adata = make_enriched_data_with_structure(
            n_proteins=100, marker_fraction=0.3, add_neighbors=True
        )

        clustering.knn_annotation(
            adata,
            gt_col="markers",
            key_added="knn_fixed",
            fix_markers=True,
            min_probability=0.3,
        )

        # Markers should have probability 1.0
        marker_mask = adata.obs["markers"].notna()
        assert np.allclose(adata.obs.loc[marker_mask, "knn_fixed_probability"], 1.0)

    def test_to_knn_graph(self):
        """Test conversion to networkx graph."""
        adata = make_enriched_data_with_structure(n_proteins=50, add_neighbors=True)

        G = clustering.to_knn_graph(adata)

        assert isinstance(G, nx.Graph)
        assert G.number_of_nodes() == adata.n_obs
        # Should have edges
        assert G.number_of_edges() > 0

    def test_get_n_nearest_neighbors(self):
        """Test getting nearest neighbors from graph."""
        adata = make_enriched_data_with_structure(n_proteins=50, add_neighbors=True)
        G = clustering.to_knn_graph(adata)

        node = adata.obs_names[0]
        neighbors = clustering.get_n_nearest_neighbors(G, node, order=1, n=10)

        assert isinstance(neighbors, set)
        assert node in neighbors
        # Should have at least the node itself
        assert len(neighbors) >= 1

    def test_calculate_interfacialness_score(self):
        """Test interfacialness score calculation."""
        adata = make_enriched_data_with_structure(
            n_proteins=100, marker_fraction=0.5, add_neighbors=True
        )

        result = clustering.calculate_interfacialness_score(
            adata,
            compartment_annotation_column="markers",
        )

        assert "jaccard_score" in result.obs.columns
        assert "jaccard_d1" in result.obs.columns
        assert "jaccard_d2" in result.obs.columns
        assert "jaccard_k1" in result.obs.columns
        assert "jaccard_k2" in result.obs.columns
        # Scores should be non-negative
        assert (result.obs["jaccard_score"] >= 0).all()


# ==============================================================================
# Scoring Function Tests
# ==============================================================================


class TestScoringFunctions:
    """Test scoring and quality metric functions."""

    def test_class_balance(self):
        """Test class balancing."""
        adata = make_enriched_data_with_structure(
            n_proteins=200, marker_fraction=0.4, add_neighbors=True
        )

        balanced = scoring.class_balance(adata, label_key="markers", min_class_size=10)

        # Should be a view
        assert balanced.is_view
        # All classes should have the same size
        class_sizes = balanced.obs["markers"].value_counts()
        assert class_sizes.nunique() == 1

    def test_class_balance_error_small_class(self):
        """Test error when class too small."""
        adata = make_enriched_data_with_structure(n_proteins=50, marker_fraction=0.1)

        with pytest.raises(ValueError, match="Smallest class"):
            scoring.class_balance(adata, label_key="markers", min_class_size=50)

    def test_silhouette_score(self):
        """Test silhouette score calculation."""
        adata = make_enriched_data_with_structure(
            n_proteins=100, marker_fraction=0.5, add_neighbors=True, add_umap=True
        )

        scoring.silhouette_score(adata, gt_col="markers", use_rep="X_umap")

        assert "silhouette" in adata.obs.columns
        assert "silhouette" in adata.uns
        assert "mean_silhouette_score" in adata.uns["silhouette"]
        assert "cluster_mean_silhouette" in adata.uns["silhouette"]
        # Scores should be between -1 and 1
        assert (adata.obs["silhouette"].dropna() >= -1).all()
        assert (adata.obs["silhouette"].dropna() <= 1).all()

    def test_calinski_habarasz_score(self):
        """Test Calinski-Harabasz score."""
        adata = make_enriched_data_with_structure(
            n_proteins=100, marker_fraction=0.5, add_neighbors=True, add_umap=True
        )

        scoring.calinski_habarasz_score(adata, gt_col="markers", use_rep="X_umap")

        assert "ch_score" in adata.uns
        assert isinstance(adata.uns["ch_score"], (float, np.floating))
        # Score should be positive
        assert adata.uns["ch_score"] > 0

    def test_calinski_habarasz_score_with_class_balance(self):
        """Test CH score with class balancing."""
        adata = make_enriched_data_with_structure(
            n_proteins=150, marker_fraction=0.5, add_neighbors=True, add_umap=True
        )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Smallest class")
            scoring.calinski_habarasz_score(
                adata,
                gt_col="markers",
                use_rep="X_umap",
                class_balance=True,
            )

        assert "ch_score" in adata.uns
        assert adata.uns["ch_score"] > 0

    def test_qsep_score(self):
        """Test QSep score calculation."""
        adata = make_enriched_data_with_structure(
            n_proteins=80, marker_fraction=0.5, add_neighbors=True
        )

        scoring.qsep_score(adata, gt_col="markers", use_rep="X")

        assert "full_distances" in adata.obs.columns
        assert "cluster_distances" in adata.uns
        assert "distances" in adata.uns["cluster_distances"]
        assert "clusters" in adata.uns["cluster_distances"]

    def test_knn_f1_score_basic(self):
        """Test KNN F1 score."""
        adata = make_enriched_data_with_structure(
            n_proteins=100, marker_fraction=0.4, add_neighbors=True
        )

        f1 = scoring.knn_f1_score(adata, gt_col="markers", pred_col=None, average="macro")

        assert isinstance(f1, (float, np.floating))
        # F1 score should be between 0 and 1
        assert 0 <= f1 <= 1

    def test_knn_f1_score_with_prediction(self):
        """Test F1 score with existing predictions."""
        adata = make_enriched_data_with_structure(
            n_proteins=100, marker_fraction=0.4, add_neighbors=True
        )

        # First create predictions (use min_probability=0 to get all predictions)
        clustering.knn_annotation(
            adata, gt_col="markers", key_added="predictions", min_probability=0
        )

        f1 = scoring.knn_f1_score(
            adata,
            gt_col="markers",
            pred_col="predictions",
            average="macro",
        )

        assert isinstance(f1, (float, np.floating))
        assert 0 <= f1 <= 1


# ==============================================================================
# Integration Function Tests
# ==============================================================================


class TestIntegrationFunctions:
    """Test dataset integration functions."""

    def test_align_adatas_intersect_both(self):
        """Test aligning datasets with intersection of obs and var."""
        datasets = make_multi_dataset_for_integration(n_datasets=3, n_proteins=100)

        aligned = integration.align_adatas(datasets, intersect_obs=True, intersect_var=True)

        # Should return same number of datasets
        assert len(aligned) == len(datasets)
        # All should have same shape
        shapes = [ad.shape for ad in aligned]
        assert all(s == shapes[0] for s in shapes)
        # All should have same proteins
        obs_names = [ad.obs_names for ad in aligned]
        assert all(obs_names[0].equals(names) for names in obs_names[1:])

    def test_align_adatas_no_intersect_obs(self):
        """Test aligning without obs intersection."""
        datasets = make_multi_dataset_for_integration(n_datasets=2, n_proteins=100)

        aligned = integration.align_adatas(datasets, intersect_obs=False, intersect_var=True)

        # Each dataset should keep its own obs
        assert len(aligned[0].obs) == len(datasets[0].obs)
        assert len(aligned[1].obs) == len(datasets[1].obs)

    def test_aligned_umap_basic(self):
        """Test aligned UMAP embedding."""
        datasets = make_multi_dataset_for_integration(n_datasets=2, n_proteins=50, n_samples=6)

        # Align first
        aligned = integration.align_adatas(datasets)

        # Run aligned UMAP
        result = integration.aligned_umap(
            aligned,
            align_data=False,
            n_neighbors=5,
            n_epochs=50,
            random_state=42,
        )

        assert len(result) == len(aligned)
        # Each should have aligned UMAP coordinates
        for ad in result:
            assert "X_aligned_umap" in ad.obsm
            assert ad.obsm["X_aligned_umap"].shape[1] == 2
            assert "aligned_umap" in ad.uns

    def test_aligned_umap_with_auto_align(self):
        """Test aligned UMAP with automatic alignment."""
        datasets = make_multi_dataset_for_integration(n_datasets=2, n_proteins=50, n_samples=6)

        result = integration.aligned_umap(
            datasets,
            align_data=True,
            n_neighbors=5,
            n_epochs=50,
            random_state=42,
        )

        assert len(result) == len(datasets)
        # All should have same proteins
        obs_names = [ad.obs_names for ad in result]
        assert all(obs_names[0].equals(names) for names in obs_names[1:])


# ==============================================================================
# Complete Workflow Tests
# ==============================================================================


class TestCompleteWorkflows:
    """Test complete analytical workflows."""

    def test_clustering_annotation_workflow(self):
        """Test complete clustering and annotation workflow."""
        # Create data with structure
        adata = make_enriched_data_with_structure(
            n_proteins=150,
            n_samples=12,
            n_compartments=4,
            marker_fraction=0.3,
            add_neighbors=True,
            add_umap=True,
        )

        # Step 1: Markov clustering
        clustering.markov_clustering(adata, resolution=1.5, key_added="mc_cluster")
        assert "mc_cluster" in adata.obs.columns

        # Step 2: KNN annotation
        clustering.knn_annotation(
            adata,
            gt_col="markers",
            key_added="knn_annotation",
            min_probability=0.5,
        )
        assert "knn_annotation" in adata.obs.columns

        # Step 3: Calculate scores
        scoring.silhouette_score(adata, gt_col="markers", use_rep="X_umap")
        assert "silhouette" in adata.uns

        scoring.calinski_habarasz_score(adata, gt_col="markers", use_rep="X_umap")
        assert "ch_score" in adata.uns

        # Step 4: F1 score
        f1 = scoring.knn_f1_score(adata, gt_col="markers", pred_col="knn_annotation")
        assert 0 <= f1 <= 1

    def test_integration_workflow(self):
        """Test complete dataset integration workflow."""
        # Create multiple datasets
        datasets = make_multi_dataset_for_integration(
            n_datasets=3,
            n_proteins=100,
            n_samples=8,
        )

        # Step 1: Align datasets
        aligned = integration.align_adatas(datasets)
        assert len(aligned) == 3
        # All should have same shape
        assert all(ad.shape == aligned[0].shape for ad in aligned)

        # Step 2: Aligned UMAP
        with_umap = integration.aligned_umap(
            aligned,
            align_data=False,
            n_neighbors=10,
            n_epochs=50,
            random_state=42,
        )

        # All should have aligned UMAP
        for ad in with_umap:
            assert "X_aligned_umap" in ad.obsm
            assert ad.obsm["X_aligned_umap"].shape[1] == 2

    def test_quality_assessment_workflow(self):
        """Test complete quality assessment workflow."""
        adata = make_enriched_data_with_structure(
            n_proteins=120,
            marker_fraction=0.4,
            add_neighbors=True,
            add_umap=True,
        )

        # Calculate multiple quality metrics
        scoring.silhouette_score(adata, gt_col="markers", use_rep="X_umap")
        scoring.calinski_habarasz_score(adata, gt_col="markers", use_rep="X_umap")
        scoring.qsep_score(adata, gt_col="markers", use_rep="X")

        # All metrics should be present
        assert "silhouette" in adata.uns
        assert "ch_score" in adata.uns
        assert "cluster_distances" in adata.uns

        # Silhouette should have detailed info
        assert "mean_silhouette_score" in adata.uns["silhouette"]
        assert "cluster_mean_silhouette" in adata.uns["silhouette"]


# ==============================================================================
# Error Handling Tests
# ==============================================================================


class TestErrorHandling:
    """Test error handling across tools functions."""

    def test_knn_annotation_missing_column(self):
        """Test error when annotation column missing."""
        adata = make_enriched_data_with_structure(n_proteins=50, add_neighbors=True)

        with pytest.raises(KeyError):
            clustering.knn_annotation(adata, gt_col="nonexistent_column")

    def test_silhouette_score_missing_embedding(self):
        """Test error when embedding not found."""
        adata = make_enriched_data_with_structure(
            n_proteins=50, marker_fraction=0.5, add_neighbors=True, add_umap=False
        )

        with pytest.raises(KeyError):
            scoring.silhouette_score(adata, gt_col="markers", use_rep="X_umap")

    def test_aligned_umap_mismatched_shapes(self):
        """Test error when datasets have different shapes."""
        datasets = make_multi_dataset_for_integration(n_datasets=2, n_proteins=100)

        # Make shapes incompatible by removing some proteins from one dataset
        datasets[1] = datasets[1][:50, :]

        with pytest.raises(AssertionError):
            integration.aligned_umap(
                datasets,
                align_data=False,
                n_neighbors=10,
            )

    def test_class_balance_missing_label_key(self):
        """Test error when label key not found."""
        adata = make_enriched_data_with_structure(n_proteins=50)

        with pytest.raises(ValueError, match="Label key.*not found"):
            scoring.class_balance(adata, label_key="nonexistent")

    def test_calculate_interfacialness_missing_column(self):
        """Test error when compartment column missing."""
        adata = make_enriched_data_with_structure(n_proteins=50, add_neighbors=True)

        with pytest.raises(ValueError, match="Compartment annotation column.*not found"):
            clustering.calculate_interfacialness_score(
                adata,
                compartment_annotation_column="nonexistent",
            )
