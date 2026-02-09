"""Integration tests for grassp plotting functions.

This module tests plotting functions using miniaturized synthetic datasets
with realistic structure. Tests are smoke tests that verify plots execute
without errors, not that the visual output is correct.
"""

import warnings

import matplotlib

matplotlib.use('Agg')  # Non-interactive backend for testing

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
import scanpy as sc  # noqa: E402

from anndata import AnnData  # noqa: E402

from grassp.plotting import clustering, heatmaps, integration, qc, ternary  # noqa: E402
from grassp.preprocessing import enrichment, simple  # noqa: E402
from grassp.tools import localization as tl_localization  # noqa: E402
from grassp.tools import scoring  # noqa: E402

# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture(autouse=True)
def cleanup_figures():
    """Automatically close all matplotlib figures after each test.

    This prevents memory leaks from unclosed figures and ensures test isolation.
    Using autouse=True means this applies to all tests in the module without
    needing to explicitly request the fixture.
    """
    yield
    plt.close('all')


# ==============================================================================
# Helper Functions - Reuse from other test files
# ==============================================================================


def make_enriched_data_with_structure(
    n_proteins=100,
    n_samples=10,
    n_compartments=5,
    marker_fraction=0.3,
    add_neighbors=True,
    add_umap=True,
):
    """Generate synthetic enrichment data with compartment structure."""
    np.random.seed(42)

    compartment_names = [f"Compartment{i}" for i in range(1, n_compartments + 1)]
    proteins_per_comp = n_proteins // n_compartments

    # Generate enrichment data with compartment-specific patterns
    X = np.zeros((n_proteins, n_samples))

    for i, comp in enumerate(compartment_names):
        start_idx = i * proteins_per_comp
        end_idx = start_idx + proteins_per_comp if i < n_compartments - 1 else n_proteins

        enriched_samples = [i, (i + 1) % n_samples]
        for j in range(start_idx, end_idx):
            pattern = np.random.normal(0, 0.5, n_samples)
            for s in enriched_samples:
                pattern[s] += np.random.normal(3, 0.5)
            X[j, :] = pattern

    X += np.random.normal(0, 0.2, X.shape)

    protein_ids = [f"P{str(i).zfill(5)}" for i in range(n_proteins)]
    obs = pd.DataFrame(index=protein_ids)
    obs["Gene names"] = [f"GENE{i}" for i in range(n_proteins)]

    # Add markers
    markers = np.array([None] * n_proteins, dtype=object)
    n_markers = int(n_proteins * marker_fraction)
    for i in range(n_markers):
        comp_idx = i % n_compartments
        markers[i] = compartment_names[comp_idx]
    obs["markers"] = pd.Categorical(markers)

    var = pd.DataFrame(index=[f"Sample{i}" for i in range(n_samples)])

    adata = AnnData(X=X, obs=obs, var=var)

    if add_neighbors:
        simple.neighbors(adata, n_neighbors=min(15, n_proteins - 1))

    if add_umap and add_neighbors:
        sc.tl.umap(adata)

    return adata


def make_mini_orgip_data(n_proteins=50, n_baits=2, n_replicates=2):
    """Generate synthetic OrgIP-style dataset with pvals layer."""
    np.random.seed(42)

    bait_names = [f"BAIT{i}" for i in range(1, n_baits + 1)]
    bait_names_with_untagged = bait_names + ["UNTAGGED"]

    samples = []
    subcellular_enrichment = []
    covariate_bait = []
    covariate_batch = []

    for bait in bait_names_with_untagged:
        for rep in range(1, n_replicates + 1):
            samples.append(f"{bait}_R{rep}")
            subcellular_enrichment.append(bait)
            covariate_bait.append(bait)
            covariate_batch.append(f"Batch{(rep - 1) % 2 + 1}")

    n_samples = len(samples)
    X = np.zeros((n_proteins, n_samples))

    # Add realistic intensity patterns
    for i, bait in enumerate(bait_names_with_untagged):
        bait_mask = np.array(subcellular_enrichment) == bait
        bait_indices = np.where(bait_mask)[0]

        if bait == "UNTAGGED":
            base_intensity = np.random.lognormal(mean=20, sigma=1.5, size=n_proteins)
            for idx in bait_indices:
                noise = np.random.normal(0, 0.2, n_proteins)
                X[:, idx] = base_intensity * (1 + noise)
        else:
            n_enriched = n_proteins // (n_baits * 2)
            start_idx = i * n_enriched
            end_idx = start_idx + n_enriched

            base_intensity = np.random.lognormal(mean=20, sigma=1.5, size=n_proteins)
            for idx in bait_indices:
                noise = np.random.normal(0, 0.2, n_proteins)
                intensities = base_intensity * (1 + noise)
                intensities[start_idx:end_idx] *= np.random.uniform(5, 10, n_enriched)
                X[:, idx] = intensities

    # Add missing values
    mask = np.random.rand(*X.shape) < 0.3
    X[mask] = 0

    protein_ids = [f"P{str(i).zfill(5)}" for i in range(n_proteins)]
    obs = pd.DataFrame(index=protein_ids)
    obs["Gene names"] = [f"GENE{i}" for i in range(n_proteins)]

    var = pd.DataFrame(
        {
            "subcellular_enrichment": subcellular_enrichment,
            "covariate_Bait": covariate_bait,
            "covariate_Batch": covariate_batch,
        },
        index=samples,
    )

    adata = AnnData(X=X, obs=obs, var=var)
    return adata


def make_multi_dataset_for_integration(n_datasets=2, n_proteins=50, n_samples=6):
    """Generate multiple datasets for integration testing."""
    np.random.seed(42)

    n_common = int(n_proteins * 0.7)
    n_unique = n_proteins - n_common
    common_proteins = [f"P{str(i).zfill(5)}" for i in range(n_common)]

    datasets = []
    for d in range(n_datasets):
        unique_proteins = [
            f"P{str(i + n_common + d * n_unique).zfill(5)}" for i in range(n_unique)
        ]
        all_proteins = common_proteins + unique_proteins

        X = np.random.normal(d * 2, 1.0, (len(all_proteins), n_samples))

        obs = pd.DataFrame(index=all_proteins)
        obs["Gene names"] = [f"GENE{i}" for i in range(len(all_proteins))]
        obs["leiden"] = pd.Categorical([f"Cluster{i % 3}" for i in range(len(all_proteins))])

        var = pd.DataFrame(index=[f"S{i}" for i in range(n_samples)])
        var["dataset"] = f"Dataset{d + 1}"

        adata = AnnData(X=X, obs=obs, var=var)
        datasets.append(adata)

    return datasets


# ==============================================================================
# Clustering Plot Tests
# ==============================================================================


class TestClusteringPlots:
    """Test clustering visualization functions."""

    def test_knn_violin_smoke(self):
        """Verify knn_violin executes without error."""
        adata = make_enriched_data_with_structure(
            n_proteins=80, marker_fraction=0.4, add_neighbors=True
        )

        # Run KNN annotation to get predictions
        tl_localization.knn_annotation(
            adata, gt_col="markers", key_added="knn_pred", min_probability=0
        )

        # Plot
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            ax = clustering.knn_violin(adata, gt_col="markers", pred_col="knn_pred")

        assert ax is not None

    def test_tagm_map_contours_smoke(self):
        """Verify tagm_map_contours executes without error."""
        pytest.importorskip("mpltern")

        adata = make_enriched_data_with_structure(
            n_proteins=60, marker_fraction=0.5, add_neighbors=True, add_umap=True
        )

        # Add mock TAGM parameters
        n_components = 3
        n_features = adata.shape[1]
        adata.uns["tagm.map.params"] = {
            "posteriors": {
                "mu": np.random.randn(n_components, n_features),
                "sigma": np.array([np.eye(n_features) for _ in range(n_components)]),
            },
            "gt_col": "markers",
        }
        adata.uns["tagm.map.allocation_colors"] = ["red", "blue", "green"]

        # Plot
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            ax = clustering.tagm_map_contours(adata, embedding="umap", size=50)

        assert ax is not None

    def test_tagm_map_pca_ellipses_smoke(self):
        """Verify tagm_map_pca_ellipses executes without error."""
        adata = make_enriched_data_with_structure(
            n_proteins=60, marker_fraction=0.5, add_neighbors=True, add_umap=False
        )

        # Run PCA
        sc.pp.pca(adata, n_comps=5)

        # Add mock TAGM parameters
        n_components = 3
        n_features = adata.shape[1]
        adata.uns["tagm.map.params"] = {
            "posteriors": {
                "mu": np.random.randn(n_components, n_features),
                "sigma": np.array([np.eye(n_features) * 0.5 for _ in range(n_components)]),
            }
        }
        adata.uns["tagm.map.allocation_colors"] = ["red", "blue", "green"]

        # Plot
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            ax = clustering.tagm_map_pca_ellipses(adata, stds=[1, 2])

        assert ax is not None


# ==============================================================================
# Heatmap Plot Tests
# ==============================================================================


class TestHeatmapPlots:
    """Test heatmap visualization functions."""

    def test_protein_clustermap_smoke(self):
        """Verify protein_clustermap executes without error."""
        adata = make_enriched_data_with_structure(
            n_proteins=50, marker_fraction=0.5, add_neighbors=False
        )

        # Filter to only proteins with markers (protein_clustermap expects all to have annotations)
        adata_filtered = adata[adata.obs["markers"].notna()].copy()

        # Plot (show=False to get return value)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            g = heatmaps.protein_clustermap(
                adata_filtered, annotation_key="markers", show=False
            )

        assert g is not None

    def test_sample_heatmap_smoke(self):
        """Verify sample_heatmap executes without error."""
        adata = make_enriched_data_with_structure(n_proteins=50, n_samples=8)

        # Plot
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            g = heatmaps.sample_heatmap(adata, show=False)

        assert g is not None

    def test_qsep_heatmap_smoke(self):
        """Verify qsep_heatmap executes without error."""
        adata = make_enriched_data_with_structure(
            n_proteins=80, marker_fraction=0.5, add_neighbors=True
        )

        # Run qsep_score to generate cluster_distances
        scoring.qsep_score(adata, gt_col="markers", use_rep="X")

        # Plot
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            ax = heatmaps.qsep_heatmap(adata, show=False)

        assert ax is not None

    def test_qsep_boxplot_smoke(self):
        """Verify qsep_boxplot executes without error."""
        adata = make_enriched_data_with_structure(
            n_proteins=80, marker_fraction=0.5, add_neighbors=True
        )

        # Run qsep_score
        scoring.qsep_score(adata, gt_col="markers", use_rep="X")

        # Plot
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            ax = heatmaps.qsep_boxplot(adata, show=False)

        assert ax is not None


# ==============================================================================
# Integration Plot Tests
# ==============================================================================


class TestIntegrationPlots:
    """Test integration visualization functions."""

    def test_aligned_umap_smoke(self):
        """Verify aligned_umap executes without error."""
        datasets = make_multi_dataset_for_integration(n_datasets=2, n_proteins=50, n_samples=6)

        # Align datasets
        from grassp.tools.integration import align_adatas
        from grassp.tools.integration import aligned_umap as tl_aligned_umap

        aligned = align_adatas(datasets)

        # Run aligned UMAP
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            result = tl_aligned_umap(
                aligned, align_data=False, n_neighbors=5, n_epochs=50, random_state=42
            )

        # Plot
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            ax = integration.aligned_umap(
                result[0],
                result[1],
                aligned_umap_key="X_aligned_umap",
                show=False,
            )

        assert ax is not None

    def test_mr_plot_smoke(self):
        """Verify mr_plot executes without error."""
        adata = make_enriched_data_with_structure(n_proteins=60)

        # Add mock MR scores
        adata.obs["mr_scores_M"] = np.random.uniform(0, 10, adata.n_obs)
        adata.obs["mr_scores_R"] = np.random.uniform(0, 1, adata.n_obs)

        # Plot
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            ax = integration.mr_plot(adata, mr_key="mr_scores", show=False)

        assert ax is not None

    def test_remodeling_sankey_smoke(self):
        """Verify remodeling_sankey executes without error."""
        datasets = make_multi_dataset_for_integration(n_datasets=2, n_proteins=50, n_samples=6)

        # Align datasets
        from grassp.tools.integration import align_adatas

        aligned = align_adatas(datasets)

        # Plot
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            ax = integration.remodeling_sankey(
                aligned[0], aligned[1], cluster_key="leiden", show=False
            )

        assert ax is not None

    def test_remodeling_score_smoke(self):
        """Verify remodeling_score plot executes without error."""
        # Generate random remodeling scores
        scores = np.random.randn(100)

        # Plot
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            result = integration.remodeling_score(scores, show=False)

        assert result is not None
        assert len(result) == 2  # Returns [ax_hist, ax_box]


# ==============================================================================
# QC Plot Tests
# ==============================================================================


class TestQCPlots:
    """Test QC visualization functions."""

    def test_bait_volcano_plots_smoke(self):
        """Verify bait_volcano_plots executes without error."""
        adata = make_mini_orgip_data(n_proteins=50, n_baits=2, n_replicates=2)

        # Process to get enrichment and pvals
        sc.pp.log1p(adata)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            enriched = enrichment.calculate_enrichment_vs_untagged(
                adata,
                covariates=["covariate_Batch"],
                subcellular_enrichment_column="subcellular_enrichment",
                untagged_name="UNTAGGED",
                drop_untagged=True,
            )

        # Plot
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            axs = qc.bait_volcano_plots(enriched, show=False)

        assert axs is not None

    def test_highly_variable_proteins_smoke(self):
        """Verify highly_variable_proteins executes without error."""
        adata = make_enriched_data_with_structure(n_proteins=100, n_samples=10)

        # Calculate highly variable proteins
        simple.highly_variable_proteins(adata, n_top_proteins=50)

        # Plot
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            ax = qc.highly_variable_proteins(adata, show=False)

        assert ax is not None

    def test_marker_profiles_split_smoke(self):
        """Verify marker_profiles_split executes without error."""
        # Create small fractionation-style dataset
        # Samples in obs, proteins in var
        np.random.seed(42)
        n_samples = 12  # 2 replicates × 6 fractions
        n_proteins = 40

        # Create sample/fraction metadata
        obs_data = {
            'Fraction': [f'F{i % 6 + 1}' for i in range(n_samples)],
            'Replicate': [f'R{i // 6 + 1}' for i in range(n_samples)],
            'Compartment': pd.Categorical(
                ['Cytosol'] * 4 + ['Nucleus'] * 4 + ['Mitochondria'] * 4
            ),
        }
        obs = pd.DataFrame(obs_data, index=[f'Sample{i}' for i in range(n_samples)])

        # Create protein metadata
        var_data = {
            'Replicate': [f'R{i // (n_proteins // 2) + 1}' for i in range(n_proteins)],
        }
        var = pd.DataFrame(var_data, index=[f'P{str(i).zfill(5)}' for i in range(n_proteins)])

        # Create intensity matrix with compartment-specific patterns
        X = np.random.lognormal(mean=5, sigma=1, size=(n_samples, n_proteins))

        adata = AnnData(X=X, obs=obs, var=var)

        # Test basic plot
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            axs = qc.marker_profiles_split(
                adata, marker_column='Compartment', n_columns=2, show=False
            )

        assert axs is not None

        # Test with xticklabels
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            axs = qc.marker_profiles_split(
                adata,
                marker_column='Compartment',
                xticklabels=True,
                show=False,
            )

        assert axs is not None

        # Test with replicate_column
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            axs = qc.marker_profiles_split(
                adata,
                marker_column='Compartment',
                replicate_column='Replicate',
                show=False,
            )

        assert axs is not None

        # Test with custom ylabel and plot_mean
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            axs = qc.marker_profiles_split(
                adata,
                marker_column='Compartment',
                ylabel='Log2 Intensity',
                plot_mean=True,
                show=False,
            )

        assert axs is not None


# ==============================================================================
# Ternary Plot Tests
# ==============================================================================


class TestTernaryPlots:
    """Test ternary diagram visualization functions."""

    def test_ternary_smoke(self):
        """Verify ternary plot executes without error."""
        pytest.importorskip("mpltern")

        # Create data with exactly 3 columns
        n_proteins = 50
        np.random.seed(42)

        # Generate proportions that sum to 1
        proportions = np.random.dirichlet(np.ones(3), size=n_proteins)

        protein_ids = [f"P{str(i).zfill(5)}" for i in range(n_proteins)]
        obs = pd.DataFrame(index=protein_ids)
        obs["category"] = pd.Categorical([f"Cat{i % 3}" for i in range(n_proteins)])

        var = pd.DataFrame(index=["Comp1", "Comp2", "Comp3"])

        adata = AnnData(X=proportions, obs=obs, var=var)

        # Plot
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            ax = ternary(adata, color="category", show=False)

        assert ax is not None


# ==============================================================================
# Error Handling Tests
# ==============================================================================


class TestPlottingErrorHandling:
    """Test error handling in plotting functions."""

    def test_qsep_heatmap_missing_data_error(self):
        """Test error when cluster_distances not computed."""
        adata = make_enriched_data_with_structure(n_proteins=50)

        with pytest.raises(ValueError, match="Cluster distances not found"):
            heatmaps.qsep_heatmap(adata)

    def test_mr_plot_missing_scores_error(self):
        """Test error when MR scores not found."""
        adata = make_enriched_data_with_structure(n_proteins=50)

        with pytest.raises(ValueError, match="MR scores not found"):
            integration.mr_plot(adata)

    def test_ternary_wrong_dimensions_error(self):
        """Test error when data doesn't have exactly 3 columns."""
        pytest.importorskip("mpltern")

        adata = make_enriched_data_with_structure(n_proteins=50, n_samples=5)

        with pytest.raises(ValueError, match="Ternary plots requires.*3 samples"):
            ternary(adata)
