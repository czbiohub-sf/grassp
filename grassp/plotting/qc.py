from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
) -> None | plt.Axes:
    """Plot dispersions versus means for highly variable proteins.

    Parameters
    ----------
    adata_or_result
        Annotated data matrix or DataFrame/recarray containing results from
        highly_variable_proteins computation.
    highly_variable_proteins
        Whether to plot highly variable proteins or all proteins. Default: True.
    show
        Show the plot. If None, use scanpy's plotting settings.
    save
        If True or a str, save the figure. A string is appended to the default
        filename. Infer the filetype if ending on {'.pdf', '.png', '.svg'}.
    log
        Plot on log scale. Default: False.

    Returns
    -------
    matplotlib.pyplot.Axes or None
        If `show=False`, returns matplotlib axes object. Otherwise returns None and shows or saves the plot.

    Notes
    -----
    This is a modified version of scanpy's highly_variable_genes plot adapted for
    proteomics data. It shows the relationship between mean protein intensity and
    dispersion/variance, highlighting highly variable proteins.
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
) -> None | plt.Axes:
    """Create volcano plots for bait enrichment analysis.

    Parameters
    ----------
    data
        Annotated data matrix with proteins as observations (rows) and baits as variables (columns).
        Must contain a 'pvals' layer with p-values.
    baits
        List of bait names to plot. If None, plot all baits in data.var_names.
    sig_cutoff
        P-value significance cutoff for highlighting enriched proteins.
    lfc_cutoff
        Log fold change cutoff for highlighting enriched proteins.
    n_cols
        Number of columns in the plot grid.
    base_figsize
        Base figure size for each subplot.
    annotate_top_n
        Number of top proteins to annotate with text labels.
    color_by
        Column in data.obs to color points by.
    highlight
        List of protein names to highlight in blue.
    title
        Title for the overall figure.
    show
        Whether to display the plot.

    Returns
    -------
    None or matplotlib.axes.Axes
        If show=True, returns None. Otherwise returns the matplotlib axes.
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
