from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy.plotting._tools.scatterplots
import scanpy.plotting._utils as _utils

from anndata import AnnData
from matplotlib import rcParams


# This is a slightly modified function from scanpy to use "proteins" instead of "genes"
# For the original function, see: https://github.com/scverse/scanpy/blob/a91bb02b31a637caeee77c71dcd9fbce8437cb7d/src/scanpy/plotting/_preprocessing.py
def highly_variable_proteins(
    adata_or_result: AnnData | pd.DataFrame | np.recarray,
    *,
    highly_variable_proteins: bool = True,
    show: bool | None = None,
    save: bool | str | None = None,
    log: bool = False,
) -> plt.Axes | None:
    """Mean–variance relationship for protein expression.

    A lightweight wrapper around :func:`scanpy.pl.highly_variable_genes` which
    substitutes *genes* with *proteins*.  It visualises the mean intensity
    versus dispersion (or variance for the *Seurat v3* flavour) and
    highlights the subset flagged as *highly variable* during
    :func:`~grassp.pp.highly_variable_proteins`.

    Parameters
    ----------
    adata_or_result
        Either an :class:`~anndata.AnnData` object that already contains the
        *highly-variable-proteins* results (``adata.uns['hvg']`` & friends) or
        the corresponding result DataFrame/recarray returned by
        :func:`~grassp.pp.highly_variable_proteins`.
    highly_variable_proteins
        If ``True`` (default) only the highly variable subset is highlighted in
        black; otherwise the entire set is displayed uniformly.
    show
        If ``True`` (default) the plot is shown and the function returns ``None``.
    save
        If ``True`` or a ``str``, save the figure. A string is appended to the default filename.
        Infer the filetype if ending on ``{'.pdf', '.png', '.svg'}``.
    log
        Plot axes on log-scale (disabled by default).

    Returns
    -------
    Returns the current axes if ``show`` is ``False``.

    Notes
    -----
    Adapted from Scanpy’s implementation to use *proteins* terminology.
    """

    if isinstance(adata_or_result, AnnData):
        result = adata_or_result.obs
        seurat_v3_flavor = adata_or_result.uns["hvg"]["flavor"] == "seurat_v3"
    else:
        result = adata_or_result
        if isinstance(result, pd.DataFrame):
            seurat_v3_flavor = "variances_norm" in result.columns
        else:
            seurat_v3_flavor = False
    if highly_variable_proteins:
        protein_subset = result.highly_variable
    else:
        protein_subset = result.protein_subset
    means = result.means

    if seurat_v3_flavor:
        var_or_disp = result.variances
        var_or_disp_norm = result.variances_norm
    else:
        var_or_disp = result.dispersions
        var_or_disp_norm = result.dispersions_norm
    size = rcParams["figure.figsize"]
    plt.figure(figsize=(2 * size[0], size[1]))
    plt.subplots_adjust(wspace=0.3)
    for idx, d in enumerate([var_or_disp_norm, var_or_disp]):
        plt.subplot(1, 2, idx + 1)
        for label, color, mask in zip(
            ["highly variable proteins", "other proteins"],
            ["black", "grey"],
            [protein_subset, ~protein_subset],
        ):
            if False:
                means_, var_or_disps_ = np.log10(means[mask]), np.log10(d[mask])
            else:
                means_, var_or_disps_ = means[mask], d[mask]
            plt.scatter(means_, var_or_disps_, label=label, c=color, s=1)
        if log:  # there's a bug in autoscale
            plt.xscale("log")
            plt.yscale("log")
            y_min = np.min(var_or_disp)
            y_min = 0.95 * y_min if y_min > 0 else 1e-1
            plt.xlim(0.95 * np.min(means), 1.05 * np.max(means))
            plt.ylim(y_min, 1.05 * np.max(var_or_disp))
        if idx == 0:
            plt.legend()
        plt.xlabel(("$log_{10}$ " if False else "") + "mean intensities of proteins")
        data_type = "dispersions" if not seurat_v3_flavor else "variances"
        plt.ylabel(
            ("$log_{10}$ " if False else "")
            + f"{data_type} of proteins"
            + (" (normalized)" if idx == 0 else " (not normalized)")
        )

    _utils.savefig_or_show("filter_proteins_dispersion", show=show, save=save)
    if show:
        return None
    return plt.gca()


def bait_volcano_plots(
    data: AnnData,
    baits: List[str] | None = None,
    sig_cutoff: float = 0.05,
    lfc_cutoff: float = 2,
    n_cols: int = 3,
    base_figsize: float = 5,
    annotate_top_n: int = 10,
    color_by: str | None = None,
    highlight: List[str] | None = None,
    title: str | None = None,
    show: bool = False,
) -> plt.Axes | None:
    """Volcano plots for enrichment of pull-down *baits*.

    Parameters
    ----------
    data
        AnnData object where variables correspond to bait experiments and
        observations to proteins.  A layer named ``"pvals"`` must contain the
        per-protein p-values.
    baits
        Names of the bait columns to visualise.  By default all variables are
        plotted.
    sig_cutoff, lfc_cutoff
        Thresholds for statistical significance and magnitude (*log2 fold
        change*).
    n_cols, base_figsize
        Layout parameters for the multi-panel figure.
    annotate_top_n
        Annotate the *n* proteins with the smallest p-values.
    color_by
        Optional observation annotation used for point color.
    highlight
        Protein IDs to emphasise across all panels.
    title
        Suptitle for the figure.
    show
        If ``True`` (default) the figure is displayed; otherwise the array of
        Axes is returned.

    Returns
    -------
    Returns the array of Axes (when ``show`` is ``False``) or ``None``.
    """
    if baits is None:
        baits = list(data.var_names)
    else:
        assert set(baits).issubset(data.var_names)

    n_baits = len(baits)
    n_cols = min(n_cols, n_baits)
    n_rows = np.ceil(n_baits / n_cols).astype(int)
    fig, axs = plt.subplots(
        n_rows, n_cols, figsize=(base_figsize * n_cols, base_figsize * n_rows)
    )
    if title is not None:
        fig.suptitle(title)
    for i, bait in enumerate(baits):
        ax = axs.flatten()[i]
        data_sub = data[:, bait]
        enrichments = data_sub.X[:, 0]
        if "pvals" not in data_sub.layers.keys():
            raise ValueError("anndata object must contain a 'pvals' layer")
        pvals = -np.log10(data_sub.layers["pvals"][:, 0])
        ax.scatter(enrichments, pvals, c="black", s=1, marker=".")
        ax.axhline(-np.log10(sig_cutoff), color="black", linestyle="--")
        ax.axvline(lfc_cutoff, color="black", linestyle="--")
        ax.axvline(-lfc_cutoff, color="black", linestyle="--")
        mask = (np.abs(enrichments) > lfc_cutoff) & (pvals > -np.log10(sig_cutoff))
        ax.scatter(enrichments[mask], pvals[mask], c="red", s=1, marker=".")
        lim = np.abs(enrichments).max()
        ax.set_xlim(-lim, lim)
        ax.set_title(bait)
        if highlight is not None:
            hl_mask = data.obs_names.isin(highlight)
            ax.scatter(
                enrichments[hl_mask],
                pvals[hl_mask],
                c="blue",
                s=20,
                marker=".",
                label="highlight",
            )
        if annotate_top_n > 0:
            top_n = np.argsort(pvals)[::-1][:annotate_top_n]
            for j in top_n:
                ax.text(
                    enrichments[j],
                    pvals[j],
                    data.obs_names[j],
                    fontsize=8,
                    ha="center",
                    va="center",
                )
        if i % n_cols == 0:
            ax.set_ylabel("-log10(p-value)")
        if i >= n_baits - n_cols:
            ax.set_xlabel("log2(fold change)")
    if not show:
        return axs


def _prepare_marker_profile_data(
    adata: AnnData,
    marker_column: str,
    plot_nan: bool,
    replicate_column: str | None,
):
    """Internal function to prepare data for marker profile plotting.

    Parameters
    ----------
    adata
        AnnData object with proteins in `.var` and samples/fractions in `.obs`.
    marker_column
        Column name in ``adata.obs`` containing marker annotations.
    plot_nan
        If ``True``, NaN entries in the marker column are included.
    replicate_column
        Column name in ``adata.var`` indicating replicate groups.

    Returns
    -------
    marker_series
        Series containing marker annotations from adata.obs
    categories
        List of marker categories to plot
    palette
        Dictionary mapping categories to colors
    replicate_boundaries
        List of x-positions for replicate boundary lines
    """
    # Validation
    if marker_column not in adata.obs.columns:
        raise ValueError(f"Column '{marker_column}' not found in adata.obs")

    if replicate_column is not None and replicate_column not in adata.var.columns:
        raise ValueError(f"Column '{replicate_column}' not found in adata.var")

    # Get marker categories
    if not isinstance(adata.obs[marker_column].dtype, pd.CategoricalDtype):
        adata.obs[marker_column] = adata.obs[marker_column].astype('category')

    marker_series = adata.obs[marker_column]
    if plot_nan:
        categories = marker_series.cat.categories
        categories = categories[pd.notna(categories)] if not plot_nan else categories
    else:
        categories = marker_series.dropna().unique()

    categories = sorted([cat for cat in categories if pd.notna(cat)])
    if plot_nan and marker_series.isna().any():
        categories.append(np.nan)

    # Get colors for each marker category
    palette = scanpy.plotting._tools.scatterplots._get_palette(adata, marker_column)

    # Find replicate boundaries if replicate_column is provided
    replicate_boundaries = []
    if replicate_column is not None:
        replicate_series = adata.var[replicate_column]
        # Find indices where replicate changes (on the last protein of each replicate)
        for i in range(1, len(replicate_series)):
            if replicate_series.iloc[i - 1] != replicate_series.iloc[i]:
                replicate_boundaries.append(i - 1)

    return marker_series, categories, palette, replicate_boundaries


def marker_profiles_split(
    adata: AnnData,
    marker_column: str,
    plot_nan: bool = False,
    n_columns: int = 3,
    xticklabels: bool = False,
    ylabel: str = 'Abundance',
    replicate_column: str | None = None,
    plot_mean: bool = True,
    show: bool = True,
    save: bool | str | None = None,
) -> plt.Axes | None:
    """Plot sample/fraction profiles grouped by marker annotation.

    Creates a grid of subplots where each subplot shows profiles for all
    samples/fractions assigned to a specific marker category. Each line represents
    one sample/fraction's profile across all proteins.

    This function assumes that ``adata.var`` is sorted by Replicate (if present)
    and Fraction/Pulldown, so that related measurements are adjacent on the x-axis.

    Parameters
    ----------
    adata
        AnnData object with proteins in `.var` and samples/fractions in `.obs`.
    marker_column
        Column name in ``adata.obs`` containing marker annotations.
    plot_nan
        If ``True``, NaN entries in the marker column get their own facet;
        otherwise they are skipped.
    n_columns
        Number of columns in the plot grid.
    xticklabels
        If ``True``, label x-ticks with ``adata.var_names``.
    ylabel
        Label for the y-axis. Default is ``'Abundance'``.
    replicate_column
        Column name in ``adata.var`` indicating replicate groups. If provided,
        vertical dashed lines are drawn at replicate boundaries (after the last
        instance of each replicate).
    plot_mean
        If ``True`` (default), plot the mean profile across all samples/fractions
        in each category as a black line. A legend is added to the last subplot.
    show
        If ``True`` (default) the plot is shown and the function returns ``None``.
    save
        If ``True`` or a ``str``, save the figure. A string is appended to the default filename.
        Infer the filetype if ending on ``{'.pdf', '.png', '.svg'}``.

    Returns
    -------
    Returns the array of Axes if ``show`` is ``False``, otherwise ``None``.

    See Also
    --------
    marker_profiles : Plot mean profiles with error bands in a single plot.
    """
    # Prepare data
    marker_series, categories, palette, replicate_boundaries = _prepare_marker_profile_data(
        adata, marker_column, plot_nan, replicate_column
    )

    # Create figure
    n_categories = len(categories)
    n_cols = min(n_columns, n_categories)
    n_rows = int(np.ceil(n_categories / n_cols))

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), squeeze=False)
    axs = axs.flatten()

    # Plot each category
    for idx, category in enumerate(categories):
        ax = axs[idx]

        # Get samples in this category
        if pd.isna(category):
            mask = marker_series.isna()
            category_name = "NaN"
            color = "gray"
        else:
            mask = marker_series == category
            category_name = str(category)
            color = palette.get(category, "gray")

        # Get data for this category (each row is a sample/fraction profile)
        category_data = adata[mask, :].X

        # Plot each sample/fraction profile
        for i in range(category_data.shape[0]):
            ax.plot(category_data[i, :], color=color, alpha=0.7, linewidth=1)

        # Plot mean profile if requested
        if plot_mean and category_data.shape[0] > 0:
            mean_profile = np.mean(category_data, axis=0)
            ax.plot(mean_profile, color='black', linewidth=2, label='Mean')

        # Add vertical dashed lines at replicate boundaries
        if replicate_boundaries:
            for boundary in replicate_boundaries:
                ax.axvline(boundary, color="gray", linestyle="--", linewidth=1, alpha=0.7)

        n_proteins = adata.n_vars
        ax.set_title(category_name, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.set_xlim(-0.5, n_proteins - 0.5)

        # Set x-tick labels if requested
        if xticklabels:
            ax.set_xticks(range(n_proteins))
            ax.set_xticklabels(adata.var_names, rotation=90, ha="center")

        # Add legend to the last subplot with data
        if plot_mean and idx == len(categories) - 1:
            ax.legend()

    # Hide unused subplots
    for idx in range(n_categories, len(axs)):
        axs[idx].axis("off")

    plt.tight_layout()

    _utils.savefig_or_show("marker_profiles_split", show=show, save=save)
    if show:
        return None
    return axs


def marker_profiles(
    adata: AnnData,
    marker_column: str,
    plot_nan: bool = False,
    error_type: str = 'std',
    xticklabels: bool = False,
    ylabel: str = 'Abundance',
    replicate_column: str | None = None,
    show: bool = True,
    save: bool | str | None = None,
) -> plt.Axes | None:
    """Plot mean profiles with error bands for each marker annotation.

    Creates a single plot showing the mean profile for each marker category
    with shaded error regions (standard deviation or standard error).

    This function assumes that ``adata.var`` is sorted by Replicate (if present)
    and Fraction/Pulldown, so that related measurements are adjacent on the x-axis.

    Parameters
    ----------
    adata
        AnnData object with proteins in `.var` and samples/fractions in `.obs`.
    marker_column
        Column name in ``adata.obs`` containing marker annotations.
    plot_nan
        If ``True``, NaN entries in the marker column are included;
        otherwise they are skipped.
    error_type
        Type of error to display: ``'std'`` for standard deviation or
        ``'sem'`` for standard error of the mean. Default is ``'std'``.
    xticklabels
        If ``True``, label x-ticks with ``adata.var_names``.
    ylabel
        Label for the y-axis. Default is ``'Abundance'``.
    replicate_column
        Column name in ``adata.var`` indicating replicate groups. If provided,
        vertical dashed lines are drawn at replicate boundaries (after the last
        instance of each replicate).
    show
        If ``True`` (default) the plot is shown and the function returns ``None``.
    save
        If ``True`` or a ``str``, save the figure. A string is appended to the default filename.
        Infer the filetype if ending on ``{'.pdf', '.png', '.svg'}``.

    Returns
    -------
    Returns the Axes if ``show`` is ``False``, otherwise ``None``.

    See Also
    --------
    marker_profiles_split : Plot individual profiles in separate subplots.
    """
    if error_type not in ['std', 'sem']:
        raise ValueError("error_type must be 'std' or 'sem'")

    # Prepare data
    marker_series, categories, palette, replicate_boundaries = _prepare_marker_profile_data(
        adata, marker_column, plot_nan, replicate_column
    )

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    n_proteins = adata.n_vars

    # Plot each category
    for category in categories:
        # Get samples in this category
        if pd.isna(category):
            mask = marker_series.isna()
            category_name = "NaN"
            color = "gray"
        else:
            mask = marker_series == category
            category_name = str(category)
            color = palette.get(category, "gray")

        # Get data for this category (each row is a sample/fraction profile)
        category_data = adata[mask, :].X

        if category_data.shape[0] == 0:
            continue

        # Calculate mean and error
        mean_profile = np.mean(category_data, axis=0)
        if error_type == 'std':
            error = np.std(category_data, axis=0)
        else:  # 'sem'
            error = np.std(category_data, axis=0) / np.sqrt(category_data.shape[0])

        # Plot mean line with shaded error region
        x = np.arange(n_proteins)
        ax.plot(x, mean_profile, color=color, linewidth=2, label=category_name)
        ax.fill_between(
            x,
            mean_profile - error,
            mean_profile + error,
            color=color,
            alpha=0.2,
        )

    # Add vertical dashed lines at replicate boundaries
    if replicate_boundaries:
        for boundary in replicate_boundaries:
            ax.axvline(boundary, color="gray", linestyle="--", linewidth=1, alpha=0.7)

    ax.set_ylabel(ylabel)
    ax.set_xlim(-0.5, n_proteins - 0.5)
    ax.legend()

    # Set x-tick labels if requested
    if xticklabels:
        ax.set_xticks(range(n_proteins))
        ax.set_xticklabels(adata.var_names, rotation=90, ha="center")

    plt.tight_layout()

    _utils.savefig_or_show("marker_profiles", show=show, save=save)
    if show:
        return None
    return ax
