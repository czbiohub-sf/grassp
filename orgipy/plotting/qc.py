from __future__ import annotations

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
) -> None:
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
