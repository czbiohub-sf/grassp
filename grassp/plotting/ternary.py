from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Optional, List
    from anndata import AnnData

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

from scanpy.plotting._tools.scatterplots import (
    _color_vector,
    _get_color_source_vector,
    _get_palette,
)


def _add_ternary_categorical_legend(
    ax,
    color_source_vector,
    palette,
    legend_loc=None,
    legend_fontweight=None,
    legend_fontsize=None,
    legend_fontoutline=None,
    na_color="grey",
    na_in_legend=None,
):
    """Add a categorical legend to a ternary scatter plot.

    This helper mimics :func:`scanpy.plotting._utils._add_categorical_legend`
    but is adapted for ``mpltern`` axes.  It inspects ``color_source_vector``
    for unique categories, maps them to colors provided in ``palette`` and
    attaches a legend to ``ax``.

    Parameters
    ----------
    ax
        mpltern axes instance returned by :func:`matplotlib.pyplot.subplot`
        with ``projection="ternary"``.
    color_source_vector
        Iterable of categorical labels (pandas Series or numpy array).
    palette
        Mapping from category name to colour.
    legend_loc
        Position of the legend (*e.g.* ``"upper right"``).
    legend_fontsize
        Font size used for category labels.
    legend_fontweight
        Font weight used for labels (e.g. ``"bold"``).
    legend_fontoutline
        Outline colour for legend text (if supported).
    na_color
        Colour used for missing category values.
    na_in_legend
        If ``True`` include a legend entry for missing values.
    """
    import pandas as pd

    # Get unique categories - handle mixed types properly
    if isinstance(color_source_vector, pd.Series):
        cats = color_source_vector.cat.categories
    else:
        # Convert to pandas Series to handle mixed types properly
        series = pd.Series(color_source_vector)
        cats = series.dropna().unique()

    # Create legend handles
    handles = []
    for cat in cats:
        if pd.isna(cat):
            if na_in_legend:
                handles.append(
                    mlines.Line2D(
                        [],
                        [],
                        color=na_color,
                        marker="o",
                        linestyle="None",
                        markersize=6,
                        label=str(cat),
                    )
                )
        else:
            handles.append(
                mlines.Line2D(
                    [],
                    [],
                    color=palette[cat],
                    marker="o",
                    linestyle="None",
                    markersize=6,
                    label=str(cat),
                )
            )

    # Add legend with positioning to avoid overlap
    if legend_loc is not None and legend_loc != "none":
        # Define bbox_to_anchor positions for common legend locations
        bbox_positions = {
            "upper right": (1.15, 1.0),
            "upper left": (-0.15, 1.0),
            "lower right": (1.15, 0.0),
            "right": (1.15, 0.5),
            "left": (-0.15, 0.5),
            "lower left": (-0.15, 0.0),
            "center right": (1.15, 0.5),
            "center left": (-0.15, 0.5),
            "upper center": (0.5, 1.15),
            "lower center": (0.5, -0.15),
        }

        # Use bbox_to_anchor for better positioning
        if legend_loc in bbox_positions:
            bbox_to_anchor = bbox_positions[legend_loc]
            loc = (
                "center left"
                if "right" in legend_loc
                else "center right" if "left" in legend_loc else "lower center"
            )
        else:
            # Fallback for other locations
            bbox_to_anchor = None
            loc = legend_loc

        ax.legend(
            handles=handles,
            loc=loc,
            bbox_to_anchor=bbox_to_anchor,
            fontsize=legend_fontsize,
            frameon=False,
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
    """Scatter plot of 3-part compositions in a ternary diagram.

    The function expects that ``adata.X`` has exactly three columns which
    represent the proportions of each component and therefore should sum to
    1 (or all share the same unit).

    Parameters
    ----------
    adata
        AnnData with **three** variables (columns).  Observations are plotted
        as points in barycentric (ternary) coordinates.
    color
        Key passed to Scanpy’s color utilities (e.g. column in ``adata.obs``
        or layer key).  If ``None`` the default color cycle is used.
    ax
        Existing mpltern axes.  If ``None`` a new ternary subplot is created.
    labels
        Axis labels for the three corners.  Defaults to ``adata.var_names``.
    show
        Whether to immediately display the plot.
    colorbar_loc
        Location argument forwarded to :func:`matplotlib.pyplot.colorbar` when
        ``color`` is continuous.
    legend_loc
        Location string for the categorical legend.
    legend_fontweight
        Weight of legend text.
    legend_fontsize
        Size of legend text.
    legend_fontoutline
        Outline colour for legend text.
    na_in_legend
        Whether to show a legend entry for ``NaN`` values.
    **kwargs
        Additional keyword arguments passed to
        :func:`matplotlib.axes.Axes.scatter`.

    Returns
    -------
    If ``show`` is ``False``, returns the Ternary axes containing the scatter plot.
    """

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
        _add_ternary_categorical_legend(
            ax,
            csv,
            palette=_get_palette(adata, color),
            legend_loc=legend_loc,
            # legend_fontweight=legend_fontweight,
            legend_fontsize=legend_fontsize,
            legend_fontoutline=legend_fontoutline,
            na_color="grey",
            na_in_legend=na_in_legend,
        )
    elif colorbar_loc is not None:
        plt.colorbar(cax, ax=ax, pad=0.01, fraction=0.08, aspect=30, location=colorbar_loc)
    if show:
        plt.show()
    else:
        return ax
