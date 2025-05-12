from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Optional, List
    from anndata import AnnData

import matplotlib.pyplot as plt
import numpy as np

from scanpy.plotting._tools.scatterplots import (
    _add_categorical_legend,
    _color_vector,
    _get_color_source_vector,
    _get_palette,
)


def ternary(
    adata: AnnData,
    color: Optional[str] = None,
    ax=None,
    labels: Optional[List[str]] = None,
    show: bool = True,
    colorbar_loc: Optional[str] = None,
    legend_loc: Optional[str] = None,
    legend_fontweight: Optional[str] = None,
    legend_fontsize: Optional[int] = None,
    legend_fontoutline: Optional[str] = None,
    na_in_legend: Optional[bool] = None,
    **kwargs,
):
    try:
        import mpltern  # noqa: F401
    except ImportError:
        raise ImportError(
            "mpltern is not installed. Please install it with `pip install mpltern`"
        )
    if adata.X.shape[1] != 3:
        raise ValueError("Ternary plots requires adata object with 3 samples (columns)")
    if ax is None:
        ax = plt.subplot(projection="ternary")
    if labels is None:
        labels = adata.var_names

    csv = _get_color_source_vector(adata, color)

    cv, color_type = _color_vector(adata, values_key=color, values=csv, palette=None)

    # Make sure that nan values are plottted below the other points
    nan_mask = np.isnan(csv) if isinstance(csv, np.ndarray) else csv.isna()
    if nan_mask.any():
        nan_points = adata[nan_mask].X
        ax.scatter(
            nan_points[:, 0],
            nan_points[:, 1],
            nan_points[:, 2],
            c=cv[nan_mask],
            **kwargs,
            zorder=0,
        )
    cax = ax.scatter(
        adata.X[~nan_mask, 0],
        adata.X[~nan_mask, 1],
        adata.X[~nan_mask, 2],
        zorder=1,
        c=cv[~nan_mask],
        **kwargs,
    )
    ax.taxis.set_label_position("tick1")
    ax.raxis.set_label_position("tick1")
    ax.laxis.set_label_position("tick1")
    ax.set_tlabel(labels[0])
    ax.set_llabel(labels[1])
    ax.set_rlabel(labels[2])

    if color_type == "cat":
        _add_categorical_legend(
            ax,
            csv,
            palette=_get_palette(adata, color),
            scatter_array=None,
            legend_loc=legend_loc,
            legend_fontweight=legend_fontweight,
            legend_fontsize=legend_fontsize,
            legend_fontoutline=legend_fontoutline,
            na_color="grey",
            na_in_legend=na_in_legend,
            multi_panel=False,
        )
    elif colorbar_loc is not None:
        plt.colorbar(cax, ax=ax, pad=0.01, fraction=0.08, aspect=30, location=colorbar_loc)
    if show:
        plt.show()
    return ax
