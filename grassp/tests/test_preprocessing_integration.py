"""Integration tests for grassp preprocessing functions.

This module tests complete preprocessing workflows and individual preprocessing
functions using miniaturized synthetic datasets. Tests are organized by workflow
type and function category.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
import scanpy as sc

from anndata import AnnData

from grassp.preprocessing import (
    annotation,
    enrichment,
    imputation,
    simple,
)

# ==============================================================================
# Helper Functions for Creating Test Data
# ==============================================================================


def make_mini_orgip_data(
    n_proteins=100,
    n_baits=3,
    n_replicates=3,
    add_contaminants=True,
    add_proteome=True,
    add_markers=False,
):
    """Generate synthetic OrgIP-style dataset.

    Parameters
    ----------
    n_proteins : int
        Number of proteins to include (default: 100)
    n_baits : int
        Number of bait conditions (default: 3)
    n_replicates : int
        Number of biological replicates per bait (default: 3)
    add_contaminants : bool
        Whether to add contaminant proteins (default: True)
    add_proteome : bool
        Whether to add PROTEOME control samples (default: True)
    add_markers : bool
        Whether to add marker protein annotations (default: False)

    Returns
    -------
    AnnData
        OrgIP-style dataset with structure:
        - .obs: proteins (rows)
        - .var: samples (columns)
        - Includes covariate columns (covariate_Bait, covariate_Batch)
        - Includes UNTAGGED control samples
        - Optionally includes contaminants and PROTEOME samples
    """
    # Define bait names
    bait_names = [f"BAIT{i}" for i in range(1, n_baits + 1)]
    bait_names_with_untagged = bait_names + ["UNTAGGED"]
    if add_proteome:
        bait_names_with_untagged += ["PROTEOME"]

    # Create sample structure
    samples = []
    subcellular_enrichment = []
    covariate_bait = []
    covariate_batch = []
    biological_replicate = []

    for bait in bait_names_with_untagged:
        for rep in range(1, n_replicates + 1):
            samples.append(f"{bait}_R{rep}")
            subcellular_enrichment.append(bait)
            covariate_bait.append(bait)
            covariate_batch.append(f"Batch{(rep - 1) % 2 + 1}")
            biological_replicate.append(str(rep))

    n_samples = len(samples)

    # Create protein structure
    n_contaminants = 10 if add_contaminants else 0
    n_real_proteins = n_proteins - n_contaminants

    # Generate realistic intensity matrix
    np.random.seed(42)
    X = np.zeros((n_proteins, n_samples))

    # Add realistic intensity patterns
    for i, bait in enumerate(bait_names_with_untagged):
        bait_mask = np.array(subcellular_enrichment) == bait
        bait_indices = np.where(bait_mask)[0]

        if bait == "UNTAGGED" or bait == "PROTEOME":
            # Background/proteome: lower intensities, more uniform
            base_intensity = np.random.lognormal(mean=20, sigma=1.5, size=n_real_proteins)
            for idx in bait_indices:
                noise = np.random.normal(0, 0.2, n_real_proteins)
                X[:n_real_proteins, idx] = base_intensity * (1 + noise)
        else:
            # IP samples: some proteins enriched
            n_enriched = n_real_proteins // (n_baits * 2)
            start_idx = i * n_enriched
            end_idx = start_idx + n_enriched

            base_intensity = np.random.lognormal(mean=20, sigma=1.5, size=n_real_proteins)
            for idx in bait_indices:
                noise = np.random.normal(0, 0.2, n_real_proteins)
                intensities = base_intensity * (1 + noise)
                # Enrich specific proteins
                intensities[start_idx:end_idx] *= np.random.uniform(5, 10, n_enriched)
                X[:n_real_proteins, idx] = intensities

    # Add contaminants (high intensity in all samples)
    if add_contaminants:
        contam_intensity = np.random.lognormal(mean=24, sigma=1.0, size=n_contaminants)
        for idx in range(n_samples):
            X[n_real_proteins:, idx] = contam_intensity * np.random.uniform(
                0.8, 1.2, n_contaminants
            )

    # Add realistic missingness (30% missing)
    add_realistic_missingness(X, missing_rate=0.3)

    # Create obs DataFrame (proteins)
    protein_ids = [f"P{str(i).zfill(5)}" for i in range(n_real_proteins)]
    if add_contaminants:
        protein_ids += [f"CONTAM{i}" for i in range(n_contaminants)]

    obs = pd.DataFrame(index=protein_ids)
    obs["Gene names"] = [f"GENE{i}" for i in range(n_proteins)]

    # Add contaminant metadata
    if add_contaminants:
        obs["Reverse"] = ""
        obs["Potential contaminant"] = ""
        obs.loc[protein_ids[-n_contaminants:], "Potential contaminant"] = "+"

    # Add marker annotations if requested
    if add_markers:
        compartments = ["Mitochondrion", "ER", "Golgi", "Cytosol"]
        markers = np.array([None] * n_proteins, dtype=object)
        n_markers_per_comp = min(5, n_real_proteins // len(compartments))
        for i, comp in enumerate(compartments):
            start = i * n_markers_per_comp
            end = start + n_markers_per_comp
            markers[start:end] = comp
        obs["markers"] = pd.Categorical(markers)

    # Create var DataFrame (samples)
    var = pd.DataFrame(
        {
            "subcellular_enrichment": subcellular_enrichment,
            "covariate_Bait": covariate_bait,
            "covariate_Batch": covariate_batch,
            "biological_replicate": biological_replicate,
        },
        index=samples,
    )

    adata = AnnData(X=X, obs=obs, var=var)

    # Store metadata
    adata.uns["RawInfo"] = {"filter_columns": ["Reverse", "Potential contaminant"]}

    return adata


def make_mini_dc_data(
    n_proteins=100,
    n_fractions=7,
    n_replicates=3,
    add_contaminants=True,
):
    """Generate synthetic differential centrifugation (DC) dataset.

    Parameters
    ----------
    n_proteins : int
        Number of proteins to include (default: 100)
    n_fractions : int
        Number of fractions (default: 7)
    n_replicates : int
        Number of biological replicates (default: 3)
    add_contaminants : bool
        Whether to add contaminant proteins (default: True)

    Returns
    -------
    AnnData
        DC-style dataset with structure:
        - .obs: proteins (rows)
        - .var: samples (columns)
        - Fractions: 1K, 3K, 5K, 12K, 24K, 80K, Cyt
        - Includes subcellular_enrichment and biological_replicate columns
    """
    # Define fraction names
    fraction_names = ["1K", "3K", "5K", "12K", "24K", "80K", "Cyt"][:n_fractions]

    # Create sample structure
    samples = []
    subcellular_enrichment = []
    biological_replicate = []

    for rep in range(1, n_replicates + 1):
        for frac in fraction_names:
            samples.append(f"{frac}_R{rep}")
            subcellular_enrichment.append(frac)
            biological_replicate.append(str(rep))

    n_samples = len(samples)

    # Create protein structure
    n_contaminants = 10 if add_contaminants else 0
    n_real_proteins = n_proteins - n_contaminants

    # Generate realistic intensity matrix
    np.random.seed(42)
    X = np.zeros((n_proteins, n_samples))

    # Define which proteins are enriched in which fractions
    # Nuclear proteins: enriched in low-speed fractions (1K, 3K)
    # Mitochondrial: enriched in 5K, 12K
    # Cytosolic: enriched in Cyt
    fraction_patterns = {
        "nuclear": [1.0, 0.8, 0.3, 0.1, 0.05, 0.02, 0.01],
        "mitochondrial": [0.2, 0.5, 1.0, 0.8, 0.3, 0.1, 0.05],
        "cytosolic": [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0],
    }

    # Assign proteins to patterns
    n_per_pattern = n_real_proteins // len(fraction_patterns)
    for i, (_pattern_name, pattern) in enumerate(fraction_patterns.items()):
        start_idx = i * n_per_pattern
        end_idx = (
            start_idx + n_per_pattern if i < len(fraction_patterns) - 1 else n_real_proteins
        )

        base_intensity = np.random.lognormal(mean=20, sigma=1.5, size=(end_idx - start_idx))

        for j, frac in enumerate(fraction_names):
            frac_mask = np.array(subcellular_enrichment) == frac
            frac_indices = np.where(frac_mask)[0]

            for idx in frac_indices:
                noise = np.random.normal(0, 0.15, end_idx - start_idx)
                intensities = base_intensity * pattern[j] * (1 + noise)
                X[start_idx:end_idx, idx] = intensities

    # Add contaminants (present in all fractions)
    if add_contaminants:
        contam_intensity = np.random.lognormal(mean=24, sigma=1.0, size=n_contaminants)
        for idx in range(n_samples):
            X[n_real_proteins:, idx] = contam_intensity * np.random.uniform(
                0.8, 1.2, n_contaminants
            )

    # Add realistic missingness (30% missing)
    add_realistic_missingness(X, missing_rate=0.3)

    # Create obs DataFrame (proteins)
    protein_ids = [f"P{str(i).zfill(5)}" for i in range(n_real_proteins)]
    if add_contaminants:
        protein_ids += [f"CONTAM{i}" for i in range(n_contaminants)]

    obs = pd.DataFrame(index=protein_ids)
    obs["Gene names"] = [f"GENE{i}" for i in range(n_proteins)]

    # Add contaminant metadata
    if add_contaminants:
        obs["Reverse"] = ""
        obs["Potential contaminant"] = ""
        obs.loc[protein_ids[-n_contaminants:], "Potential contaminant"] = "+"

    # Create var DataFrame (samples)
    var = pd.DataFrame(
        {
            "subcellular_enrichment": subcellular_enrichment,
            "biological_replicate": biological_replicate,
        },
        index=samples,
    )

    adata = AnnData(X=X, obs=obs, var=var)

    # Store metadata
    adata.uns["RawInfo"] = {"filter_columns": ["Reverse", "Potential contaminant"]}

    return adata


def add_realistic_missingness(X, missing_rate=0.3, pattern="MCAR"):
    """Add missing values (zeros) to dataset.

    Parameters
    ----------
    X : np.ndarray
        Data matrix to add missingness to (modified in-place)
    missing_rate : float
        Fraction of values to set to zero (default: 0.3)
    pattern : str
        Missingness pattern (default: "MCAR" - Missing Completely At Random)
    """
    if pattern == "MCAR":
        # Missing Completely At Random
        mask = np.random.rand(*X.shape) < missing_rate
        X[mask] = 0
    else:
        raise NotImplementedError(f"Missingness pattern {pattern} not implemented")


def add_marker_proteins(
    adata,
    marker_column="markers",
    n_markers_per_compartment=10,
    compartments=None,
):
    """Add known marker protein annotations for testing.

    Parameters
    ----------
    adata : AnnData
        AnnData object to annotate
    marker_column : str
        Column name to add markers to (default: "markers")
    n_markers_per_compartment : int
        Number of markers per compartment (default: 10)
    compartments : list or None
        List of compartment names. If None, uses default list.
    """
    if compartments is None:
        compartments = ["Mitochondrion", "ER", "Golgi", "Cytosol"]

    n_proteins = adata.n_obs
    markers = np.array([None] * n_proteins, dtype=object)

    n_available = min(n_markers_per_compartment * len(compartments), n_proteins)
    n_per_comp = n_available // len(compartments)

    for i, comp in enumerate(compartments):
        start = i * n_per_comp
        end = start + n_per_comp
        if end <= n_proteins:
            markers[start:end] = comp

    adata.obs[marker_column] = pd.Categorical(markers)


# ==============================================================================
# Integration Workflow Tests
# ==============================================================================


class TestOrgIPWorkflow:
    """Test complete OrgIP preprocessing workflow."""

    def test_orgip_complete_workflow(self):
        """Test complete OrgIP workflow from raw data to enrichment."""
        # Load mini OrgIP data
        adata = make_mini_orgip_data(
            n_proteins=100,
            n_baits=3,
            n_replicates=3,
            add_contaminants=True,
            add_proteome=True,
        )

        # Add markers
        add_marker_proteins(adata)
        assert "markers" in adata.obs.columns

        # Remove contaminants
        n_before = adata.n_obs
        simple.remove_contaminants(
            adata,
            filter_columns=["Potential contaminant"],
            filter_value="+",
        )
        assert adata.n_obs < n_before  # Some proteins removed

        # Filter by proteome presence
        proteome_mask = adata.var["subcellular_enrichment"] == "PROTEOME"
        if proteome_mask.any():
            proteome_present = (adata[:, proteome_mask].X > 0).any(axis=1)
            adata._inplace_subset_obs(adata.obs_names[proteome_present])

        # Filter proteins per replicate
        n_before = adata.n_obs
        simple.filter_proteins_per_replicate(
            adata,
            grouping_columns=["subcellular_enrichment"],
            min_replicates=2,
            min_samples=2,
        )
        assert adata.n_obs <= n_before

        # Log transform and store in layer
        sc.pp.log1p(adata)
        adata.layers["log1p"] = adata.X.copy()

        # Impute gaussian
        imputation.impute_gaussian(adata, width=0.3, distance=1.8)
        assert "n_imputed" in adata.obs.columns
        assert np.sum(adata.X == 0) == 0  # No more zeros

        # Calculate enrichment vs untagged
        enriched = enrichment.calculate_enrichment_vs_untagged(
            adata,
            covariates=["covariate_Batch"],
            subcellular_enrichment_column="subcellular_enrichment",
            untagged_name="UNTAGGED",
            drop_untagged=True,
        )

        # Verify results
        assert isinstance(enriched, AnnData)
        assert "pvals" in enriched.layers
        assert enriched.n_obs > 0
        assert enriched.shape[1] > 0  # n_var doesn't exist, use shape[1]
        # Should have negative and positive values
        assert np.any(enriched.X < 0)
        assert np.any(enriched.X > 0)
        # P-values should be between 0 and 1
        assert np.all((enriched.layers["pvals"] >= 0) | np.isnan(enriched.layers["pvals"]))
        assert np.all((enriched.layers["pvals"] <= 1) | np.isnan(enriched.layers["pvals"]))


class TestDCWorkflow:
    """Test complete differential centrifugation workflow."""

    def test_dc_complete_workflow(self):
        """Test complete DC workflow from raw data to enrichment."""
        # Load mini DC data
        adata = make_mini_dc_data(
            n_proteins=100,
            n_fractions=7,
            n_replicates=3,
            add_contaminants=True,
        )

        # Add markers
        add_marker_proteins(adata)
        assert "markers" in adata.obs.columns

        # Calculate QC metrics (use percent_top that doesn't exceed number of samples)
        simple.calculate_qc_metrics(adata, percent_top=(10,))
        assert "n_samples_by_intensity" in adata.obs.columns
        assert "total_intensity" in adata.obs.columns

        # Remove contaminants
        n_before = adata.n_obs
        simple.remove_contaminants(
            adata,
            filter_columns=["Potential contaminant"],
            filter_value="+",
        )
        assert adata.n_obs < n_before

        # Filter proteins per replicate
        simple.filter_proteins_per_replicate(
            adata,
            grouping_columns=["subcellular_enrichment"],
            min_replicates=2,
            min_samples=3,
        )

        # Log transform
        sc.pp.log1p(adata)

        # Impute gaussian
        imputation.impute_gaussian(adata, width=0.3, distance=1.8)
        assert np.sum(adata.X == 0) == 0

        # Calculate enrichment vs all (lfc method)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="invalid value encountered in divide",
                category=RuntimeWarning,
            )
            enriched = enrichment.calculate_enrichment_vs_all(
                adata,
                covariates=[],
                subcellular_enrichment_column="subcellular_enrichment",
                enrichment_method="lfc",
                correlation_threshold=1.0,
            )

        # Verify results
        assert isinstance(enriched, AnnData)
        assert "pvals" in enriched.layers
        assert "enriched_vs" in enriched.var.columns

        # Aggregate samples by fraction
        aggregated = simple.aggregate_samples(
            enriched,
            grouping_columns=["subcellular_enrichment"],
        )
        assert "n_merged_samples" in aggregated.var.columns
        assert aggregated.shape[1] <= enriched.shape[1]  # May be equal if already aggregated


# ==============================================================================
# Enrichment Function Tests
# ==============================================================================


class TestEnrichmentFunctions:
    """Test enrichment calculation functions."""

    def test_calculate_enrichment_vs_untagged_basic(self):
        """Test basic enrichment calculation vs untagged."""
        adata = make_mini_orgip_data(n_proteins=50, n_baits=2, n_replicates=2)
        sc.pp.log1p(adata)

        result = enrichment.calculate_enrichment_vs_untagged(
            adata,
            covariates=["covariate_Batch"],
            subcellular_enrichment_column="subcellular_enrichment",
            untagged_name="UNTAGGED",
            drop_untagged=True,
        )

        assert isinstance(result, AnnData)
        assert "pvals" in result.layers
        assert result.layers["pvals"].shape == result.X.shape
        # Untagged should be dropped
        assert "UNTAGGED" not in result.var["subcellular_enrichment"].values

    def test_calculate_enrichment_vs_untagged_no_untagged_error(self):
        """Test error when no untagged samples present."""
        adata = make_mini_orgip_data(n_proteins=50, n_baits=2, n_replicates=2)
        # Remove untagged
        adata = adata[:, adata.var["subcellular_enrichment"] != "UNTAGGED"]

        with pytest.raises(ValueError, match="No UNTAGGED samples found"):
            enrichment.calculate_enrichment_vs_untagged(
                adata,
                covariates=["covariate_Batch"],
                subcellular_enrichment_column="subcellular_enrichment",
                untagged_name="UNTAGGED",
            )

    def test_calculate_enrichment_vs_untagged_drop_parameter(self):
        """Test drop_untagged parameter."""
        adata = make_mini_orgip_data(n_proteins=50, n_baits=2, n_replicates=2)
        sc.pp.log1p(adata)

        # With drop_untagged=False
        result = enrichment.calculate_enrichment_vs_untagged(
            adata,
            covariates=["covariate_Batch"],
            subcellular_enrichment_column="subcellular_enrichment",
            untagged_name="UNTAGGED",
            drop_untagged=False,
        )

        assert "UNTAGGED" in result.var["subcellular_enrichment"].values

    def test_calculate_enrichment_vs_all_lfc_method(self):
        """Test enrichment vs all with lfc method."""
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="invalid value encountered in divide",
                category=RuntimeWarning,
            )
            adata = make_mini_dc_data(n_proteins=50, n_fractions=5, n_replicates=2)
            sc.pp.log1p(adata)

            result = enrichment.calculate_enrichment_vs_all(
                adata,
                covariates=[],
                subcellular_enrichment_column="subcellular_enrichment",
                enrichment_method="lfc",
                correlation_threshold=1.0,
            )

            assert isinstance(result, AnnData)
            assert "pvals" in result.layers
            assert "enriched_vs" in result.var.columns
            # Should have positive and negative values
            assert np.any(result.X < 0)
            assert np.any(result.X > 0)

    def test_calculate_enrichment_vs_all_proportion_method(self):
        """Test enrichment vs all with proportion method."""
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="invalid value encountered in divide",
                category=RuntimeWarning,
            )
            adata = make_mini_dc_data(n_proteins=50, n_fractions=5, n_replicates=2)
            sc.pp.log1p(adata)

            result = enrichment.calculate_enrichment_vs_all(
                adata,
                covariates=[],
                subcellular_enrichment_column="subcellular_enrichment",
                enrichment_method="proportion",
                correlation_threshold=1.0,
            )

            assert isinstance(result, AnnData)
            assert "pvals" in result.layers
            # Proportions should be between 0 and 1
            assert np.all((result.X >= 0) | np.isnan(result.X))
            assert np.all((result.X <= 1) | np.isnan(result.X))

    def test_calculate_enrichment_vs_all_correlation_threshold(self):
        """Test correlation_threshold parameter."""
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="invalid value encountered in divide",
                category=RuntimeWarning,
            )
            warnings.filterwarnings(
                "ignore", message="No sufficiently uncorrelated controls", category=UserWarning
            )
            adata = make_mini_dc_data(n_proteins=50, n_fractions=5, n_replicates=2)
            sc.pp.log1p(adata)

            # Lower threshold should be more permissive
            result = enrichment.calculate_enrichment_vs_all(
                adata,
                covariates=[],
                subcellular_enrichment_column="subcellular_enrichment",
                enrichment_method="lfc",
                correlation_threshold=0.5,
            )

            assert isinstance(result, AnnData)


# ==============================================================================
# Filtering Function Tests
# ==============================================================================


class TestFilteringFunctions:
    """Test filtering functions."""

    def test_filter_proteins_min_samples(self):
        """Test filter_proteins with min_samples."""
        adata = make_mini_dc_data(n_proteins=50, n_fractions=5, n_replicates=3)
        n_before = adata.n_obs

        simple.filter_proteins(adata, min_samples=5)

        assert adata.n_obs <= n_before
        # All remaining proteins should be present in at least 5 samples
        assert np.all((adata.X > 0).sum(axis=1) >= 5)

    def test_filter_proteins_min_counts(self):
        """Test filter_proteins with min_counts."""
        adata = make_mini_dc_data(n_proteins=50, n_fractions=5, n_replicates=3)
        n_before = adata.n_obs

        # Add some counts to make this meaningful
        sc.pp.log1p(adata)
        simple.filter_proteins(adata, min_counts=10.0)

        assert adata.n_obs <= n_before

    def test_filter_samples_basic(self):
        """Test filter_samples with min_proteins."""
        adata = make_mini_dc_data(n_proteins=50, n_fractions=5, n_replicates=3)
        n_before = adata.shape[1]  # n_var doesn't exist, use shape[1]

        simple.filter_samples(adata, min_proteins=10)

        assert adata.shape[1] <= n_before  # n_var doesn't exist, use shape[1]

    def test_filter_proteins_per_replicate(self):
        """Test filter_proteins_per_replicate."""
        adata = make_mini_orgip_data(n_proteins=50, n_baits=2, n_replicates=3)
        n_before = adata.n_obs

        simple.filter_proteins_per_replicate(
            adata,
            grouping_columns=["subcellular_enrichment"],
            min_replicates=2,
            min_samples=2,
        )

        assert adata.n_obs <= n_before

    def test_filter_min_consecutive_fractions(self):
        """Test filter_min_consecutive_fractions."""
        adata = make_mini_dc_data(n_proteins=50, n_fractions=7, n_replicates=2)

        # Create data with known consecutive patterns
        adata.X[:, :] = 0
        # Protein 0: present in first 3 fractions
        adata.X[0, [0, 1, 2, 7, 8, 9]] = 100

        n_before = adata.n_obs
        simple.filter_min_consecutive_fractions(adata, min_consecutive=3)

        assert adata.n_obs <= n_before
        assert "consecutive_fractions" in adata.obs.columns

    def test_filter_min_consecutive_fractions_with_replicates(self):
        """Test filter_min_consecutive_fractions with replicates."""
        adata = make_mini_dc_data(n_proteins=50, n_fractions=7, n_replicates=3)

        simple.filter_min_consecutive_fractions(
            adata,
            min_consecutive=2,
            replicate_column="biological_replicate",
            min_replicates=2,
        )

        assert "n_replicates_with_minimum_2_fractions" in adata.obs.columns

    def test_remove_contaminants(self):
        """Test remove_contaminants."""
        adata = make_mini_orgip_data(
            n_proteins=50, n_baits=2, n_replicates=2, add_contaminants=True
        )
        n_before = adata.n_obs

        simple.remove_contaminants(
            adata,
            filter_columns=["Potential contaminant"],
            filter_value="+",
        )

        assert adata.n_obs < n_before
        # No contaminants should remain
        assert not (adata.obs["Potential contaminant"] == "+").any()


# ==============================================================================
# Aggregation Function Tests
# ==============================================================================


class TestAggregationFunctions:
    """Test aggregation functions."""

    def test_aggregate_proteins_by_gene(self):
        """Test aggregate_proteins by gene names."""
        adata = make_mini_orgip_data(n_proteins=50, n_baits=2, n_replicates=2)

        # Create duplicate gene names (need exactly 50 values)
        adata.obs["Gene names"] = ["GENE1"] * 10 + [f"GENE{i}" for i in range(2, 42)]

        aggregated = simple.aggregate_proteins(
            adata,
            grouping_columns="Gene names",
        )

        assert "n_merged_proteins" in aggregated.obs.columns
        assert aggregated.n_obs < adata.n_obs
        # GENE1 should have n_merged_proteins > 1
        assert aggregated.obs.loc["GENE1", "n_merged_proteins"] == 10

    def test_aggregate_proteins_different_agg_func(self):
        """Test aggregate_proteins with different aggregation function."""
        adata = make_mini_orgip_data(n_proteins=50, n_baits=2, n_replicates=2)

        # Create duplicate gene names (need exactly 50 values)
        adata.obs["Gene names"] = ["GENE1"] * 10 + [f"GENE{i}" for i in range(2, 42)]

        # Test with mean
        aggregated = simple.aggregate_proteins(
            adata,
            grouping_columns="Gene names",
            agg_func=np.mean,
        )

        assert "n_merged_proteins" in aggregated.obs.columns
        assert aggregated.n_obs < adata.n_obs

    def test_aggregate_samples_basic(self):
        """Test aggregate_samples."""
        adata = make_mini_dc_data(n_proteins=50, n_fractions=5, n_replicates=3)

        aggregated = simple.aggregate_samples(
            adata,
            grouping_columns="subcellular_enrichment",
        )

        assert "n_merged_samples" in aggregated.var.columns
        assert aggregated.shape[1] < adata.shape[1]  # n_var doesn't exist, use shape[1]
        # Each fraction should have n_replicates samples merged
        assert aggregated.var["n_merged_samples"].max() == 3

    def test_aggregate_samples_keep_raw(self):
        """Test aggregate_samples with keep_raw parameter."""
        adata = make_mini_dc_data(n_proteins=50, n_fractions=5, n_replicates=3)

        aggregated = simple.aggregate_samples(
            adata,
            grouping_columns="subcellular_enrichment",
            keep_raw=True,
        )

        assert aggregated.raw is not None
        assert aggregated.raw.shape == adata.shape


# ==============================================================================
# Imputation Function Tests
# ==============================================================================


class TestImputationFunctions:
    """Test imputation functions."""

    def test_impute_gaussian_basic(self):
        """Test basic Gaussian imputation."""
        adata = make_mini_orgip_data(n_proteins=50, n_baits=2, n_replicates=2)
        sc.pp.log1p(adata)

        n_zeros_before = np.sum(adata.X == 0)
        assert n_zeros_before > 0  # Should have missing values

        imputation.impute_gaussian(adata, width=0.3, distance=1.8)

        assert "n_imputed" in adata.obs.columns
        assert np.sum(adata.X == 0) == 0  # No more zeros
        assert adata.obs["n_imputed"].sum() == n_zeros_before

    def test_impute_gaussian_per_sample(self):
        """Test Gaussian imputation with per_sample parameter."""
        adata = make_mini_orgip_data(n_proteins=50, n_baits=2, n_replicates=2)
        sc.pp.log1p(adata)

        # Test per_sample=False
        imputation.impute_gaussian(adata, width=0.3, distance=1.8, per_sample=False)

        assert np.sum(adata.X == 0) == 0

    def test_impute_gaussian_parameters(self):
        """Test Gaussian imputation with different parameters."""
        adata = make_mini_orgip_data(n_proteins=50, n_baits=2, n_replicates=2)
        sc.pp.log1p(adata)

        # Test with different width and distance
        imputation.impute_gaussian(adata, width=0.5, distance=2.0)

        assert np.sum(adata.X == 0) == 0

    def test_impute_knn_basic(self):
        """Test KNN imputation."""
        adata = make_mini_orgip_data(n_proteins=50, n_baits=2, n_replicates=2)
        sc.pp.log1p(adata)

        n_zeros_before = np.sum(adata.X == 0)
        assert n_zeros_before > 0

        imputation.impute_knn(adata, k=5, weights="uniform")

        assert np.sum(adata.X == 0) == 0


# ==============================================================================
# QC and Annotation Tests
# ==============================================================================


class TestQCAnnotation:
    """Test QC metrics and annotation functions."""

    def test_calculate_qc_metrics(self):
        """Test QC metrics calculation."""
        adata = make_mini_dc_data(n_proteins=50, n_fractions=5, n_replicates=2)

        # Use percent_top that doesn't exceed number of samples
        simple.calculate_qc_metrics(adata, percent_top=(5,))

        # Check that QC columns were added
        assert "n_samples_by_intensity" in adata.obs.columns
        assert "total_intensity" in adata.obs.columns
        assert "n_proteins_by_intensity" in adata.var.columns
        assert "total_intensity" in adata.var.columns

    def test_neighbors_basic(self):
        """Test neighbors calculation."""
        adata = make_mini_dc_data(n_proteins=50, n_fractions=5, n_replicates=2)
        sc.pp.log1p(adata)
        imputation.impute_gaussian(adata)

        simple.neighbors(adata, n_neighbors=15)

        assert "connectivities" in adata.obsp
        assert "distances" in adata.obsp

    def test_neighbors_layer_parameter(self):
        """Test neighbors with layer parameter."""
        adata = make_mini_dc_data(n_proteins=50, n_fractions=5, n_replicates=2)
        sc.pp.log1p(adata)
        adata.layers["log1p"] = adata.X.copy()
        imputation.impute_gaussian(adata)

        simple.neighbors(adata, layer="log1p", n_neighbors=15)

        assert "connectivities" in adata.obsp

    def test_add_markers_error_on_invalid_species(self):
        """Test add_markers with invalid species."""
        adata = make_mini_orgip_data(n_proteins=50, n_baits=2, n_replicates=2)

        with pytest.raises(FileNotFoundError, match="Species not found"):
            annotation.add_markers(adata, species="invalid_species")


# ==============================================================================
# Utility Function Tests
# ==============================================================================


class TestUtilityFunctions:
    """Test utility preprocessing functions."""

    def test_normalize_total(self):
        """Test total normalization."""
        adata = make_mini_dc_data(n_proteins=50, n_fractions=5, n_replicates=2)

        # Check column sums before
        col_sums_before = adata.X.sum(axis=0)
        assert not np.allclose(col_sums_before, 1e6)

        simple.normalize_total(adata, target_sum=1e6)

        # Check column sums after
        col_sums_after = adata.X.sum(axis=0)
        assert np.allclose(col_sums_after, 1e6, rtol=1e-5)

    def test_highly_variable_proteins(self):
        """Test highly variable proteins identification."""
        adata = make_mini_dc_data(n_proteins=100, n_fractions=5, n_replicates=2)
        sc.pp.log1p(adata)
        imputation.impute_gaussian(adata)

        # Calculate enrichment to get variation
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            enriched = enrichment.calculate_enrichment_vs_all(
                adata,
                covariates=[],
                subcellular_enrichment_column="subcellular_enrichment",
                enrichment_method="lfc",
            )

        simple.highly_variable_proteins(enriched, n_top_proteins=50)

        assert "highly_variable" in enriched.obs.columns
        assert enriched.obs["highly_variable"].sum() == 50

    def test_calculate_replicate_cv(self):
        """Test replicate CV calculation."""
        adata = make_mini_dc_data(n_proteins=50, n_fractions=5, n_replicates=3)
        sc.pp.log1p(adata)

        cv_df = simple.calculate_replicate_cv(
            adata,
            grouping_columns="subcellular_enrichment",
            is_log=True,
        )

        assert isinstance(cv_df, pd.DataFrame)
        assert cv_df.shape[0] == adata.n_obs  # One row per protein


# ==============================================================================
# Error Handling Tests
# ==============================================================================


class TestErrorHandling:
    """Test error handling across preprocessing functions."""

    def test_enrichment_missing_covariate(self):
        """Test error when covariate column is missing."""
        adata = make_mini_orgip_data(n_proteins=50, n_baits=2, n_replicates=2)
        sc.pp.log1p(adata)

        with pytest.raises(ValueError, match="Covariate.*not found"):
            enrichment.calculate_enrichment_vs_untagged(
                adata,
                covariates=["nonexistent_column"],
                subcellular_enrichment_column="subcellular_enrichment",
                untagged_name="UNTAGGED",
            )

    def test_remove_contaminants_missing_column(self):
        """Test error when filter column doesn't exist."""
        adata = make_mini_orgip_data(n_proteins=50, n_baits=2, n_replicates=2)

        # Remove the column
        adata.obs = adata.obs.drop(columns=["Potential contaminant"])

        with pytest.raises(KeyError):
            simple.remove_contaminants(
                adata,
                filter_columns=["Potential contaminant"],
                filter_value="+",
            )

    def test_aggregate_proteins_missing_column(self):
        """Test error when grouping column doesn't exist."""
        adata = make_mini_orgip_data(n_proteins=50, n_baits=2, n_replicates=2)

        with pytest.raises(KeyError):
            simple.aggregate_proteins(
                adata,
                grouping_columns="nonexistent_column",
            )

    def test_filter_proteins_per_replicate_missing_column(self):
        """Test error when grouping column doesn't exist."""
        adata = make_mini_orgip_data(n_proteins=50, n_baits=2, n_replicates=2)

        with pytest.raises(KeyError):
            simple.filter_proteins_per_replicate(
                adata,
                grouping_columns="nonexistent_column",
                min_replicates=2,
            )
