"""Embedding scatter plots with per-point opacity (e.g. confidence)."""

from __future__ import annotations
from typing import TYPE_CHECKING, List, Literal, Sequence

if TYPE_CHECKING:
    from anndata import AnnData
    from matplotlib.axes import Axes
    from matplotlib.colors import Normalize

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

from adjustText import adjust_text
from matplotlib import patheffects
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, to_rgba
from matplotlib.lines import Line2D
from matplotlib.patheffects import withStroke
from scipy.spatial import ConvexHull, KDTree

# Module-level constants for numerical stability and algorithmic parameters
_NUMERICAL_EPSILON = 1e-12  # Small value to prevent division by zero
_MIN_DISTANCE_SQ = 1e-4  # Minimum distance squared for repulsion forces
_FALLBACK_VECTOR_EPSILON = 1e-6  # Fallback when center equals point

# Why this is written from scratch rather than wrapping ``sc.pl.embedding``:
# - ``sc.pl.embedding`` exposes only a *scalar* ``alpha`` that applies to every
#   point. Per-point alpha would require post-hoc patching of the scatter
#   ``PathCollection`` (e.g. setting the 4th channel of ``get_facecolors()``),
#   which is brittle and tied to scanpy internals.
# - We need a second colorbar — one for the *opacity* mapping — that scanpy's
#   layout doesn't accommodate. Building the figure ourselves makes that clean.
# - Categorical palettes are still picked up from ``adata.uns[f"{color}_colors"]``
#   so visual consistency with ``sc.pl.umap`` etc. is preserved.


def _set_vector_friendly_fonts() -> None:
    """Make text in vector exports (PDF/EPS/SVG) editable in Illustrator.

    Matplotlib defaults to Type-3 fonts in PDF/EPS and outlines text to paths
    in SVG, neither of which Illustrator can select or re-edit. These rcParams
    switch PDF/EPS to embedded TrueType (``fonttype = 42``) and keep SVG text as
    real ``<text>`` elements. They are consulted by matplotlib at *save* time,
    so they must live on the global rcParams (not a temporary ``rc_context``) to
    still be in effect when the caller later calls ``savefig``.
    """
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["svg.fonttype"] = "none"


def _get_basis(adata: AnnData, basis: str) -> tuple[str, np.ndarray]:
    """Resolve ``basis`` against ``adata.obsm``, accepting both ``"umap"`` and
    ``"X_umap"`` (matches scanpy)."""
    if basis in adata.obsm:
        return basis, np.asarray(adata.obsm[basis])
    if f"X_{basis}" in adata.obsm:
        return f"X_{basis}", np.asarray(adata.obsm[f"X_{basis}"])
    raise KeyError(
        f"'{basis}' (and 'X_{basis}') not in adata.obsm. "
        f"Available: {list(adata.obsm.keys())}"
    )


def _basis2name(basis: str) -> str:
    """Axis-label root for a basis (matches scanpy)."""
    key = basis[2:] if basis.startswith("X_") else basis
    return {
        "umap": "UMAP",
        "tsne": "tSNE",
        "pca": "PC",
        "diffmap": "DC",
    }.get(key, key.upper())


def _resolve_vbound(v: float | str | None, data: np.ndarray) -> float | None:
    """Resolve a vmin/vmax-style argument that may be a float, a percentile
    string ``"pN"``, or a callable on the data."""
    if v is None:
        return None
    if isinstance(v, str):
        if not v.startswith("p"):
            raise ValueError(
                f"string vmin/vmax {v!r} must start with 'p' (e.g. 'p99' for 99th percentile)"
            )
        return float(np.nanpercentile(data, float(v[1:])))
    if callable(v):
        return float(v(data))
    return float(v)


def _build_norm(
    vmin: float | str | None,
    vmax: float | str | None,
    vcenter: float | str | None,
    norm: Normalize | None,
    data: np.ndarray,
) -> Normalize:
    """Build a matplotlib Normalize from vmin/vmax/vcenter/norm. Explicit
    ``norm`` overrides everything else."""
    if norm is not None:
        return norm
    v_min = _resolve_vbound(vmin, data)
    v_max = _resolve_vbound(vmax, data)
    v_center = _resolve_vbound(vcenter, data)
    if v_min is None:
        v_min = float(np.nanmin(data)) if np.isfinite(np.nanmin(data)) else 0.0
    if v_max is None:
        v_max = float(np.nanmax(data)) if np.isfinite(np.nanmax(data)) else 1.0
    if v_center is not None:
        return TwoSlopeNorm(vcenter=v_center, vmin=v_min, vmax=v_max)
    return plt.Normalize(vmin=v_min, vmax=v_max)


def embedding_prob(
    adata: AnnData,
    basis: str,
    color: str,
    color_prob: str,
    *,
    cmap: str = "viridis",
    palette: list | None = None,
    na_color: str = "lightgray",
    na_in_legend: bool = True,
    size: float | None = None,
    marker: str = ".",
    vmin: float | str | None = None,
    vmax: float | str | None = None,
    vcenter: float | str | None = None,
    norm: Normalize | None = None,
    prob_vmin: float | str | None = None,
    prob_vmax: float | str | None = None,
    alpha_min: float = 0.05,
    alpha_max: float = 1.0,
    opacity_color: str = "black",
    sort_order: bool = True,
    legend_loc: str = "right margin",
    legend_fontsize: float | str | None = None,
    legend_fontweight: str | int = "bold",
    legend_fontoutline: float | None = None,
    colorbar_loc: Literal["right", "left", "top", "bottom"] | None = "right",
    frameon: bool = True,
    title: str | None = None,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (7, 5),
    show: bool = True,
) -> Axes:
    """Scatter plot of an embedding where colour encodes one variable and
    per-point opacity encodes another.

    Each observation is drawn at its position in ``adata.obsm[basis]``, coloured
    according to ``adata.obs[color]`` and made transparent in proportion to
    ``adata.obs[color_prob]``. A standard colour legend (categorical) or
    colourbar (numeric) is drawn for ``color``, and a separate opacity
    colourbar (a fade from ``alpha_min`` to ``alpha_max`` at a fixed reference
    colour) is drawn for ``color_prob``.

    Parameters
    ----------
    adata
        :class:`anndata.AnnData` with the embedding stored in
        ``.obsm[basis]`` (or ``.obsm[f"X_{basis}"]``) and both ``color`` and
        ``color_prob`` present in ``.obs``.
    basis
        Key in ``adata.obsm``. Both ``"umap"`` and ``"X_umap"`` are accepted.
    color
        Single ``.obs`` column for point colour. May be categorical, boolean,
        string, or numeric.
    color_prob
        Numeric ``.obs`` column whose values map to per-point opacity.
        Higher values → more opaque.
    cmap
        Matplotlib colormap name used when ``color`` is numeric.
    palette
        Optional list of colours for categorical ``color``. When ``None``,
        ``adata.uns[f"{color}_colors"]`` is used if present, otherwise
        matplotlib's default ``tab20``.
    na_color
        Colour for observations whose ``color`` value is NaN. Default
        ``"lightgray"`` (matches scanpy).
    na_in_legend
        If ``True`` (default) and the data contain NaN, append an ``"NA"``
        entry to the categorical legend.
    size
        Marker size. Defaults to a value scaled by ``n_obs``.
    marker
        Matplotlib marker style.
    vmin, vmax, vcenter
        Limits and optional centre for the colour scale (numeric ``color`` only).
        May be a float, a percentile string (e.g. ``"p5"``, ``"p99.9"``), or a
        callable ``f(values) -> float``. ``vcenter`` triggers a
        :class:`~matplotlib.colors.TwoSlopeNorm`, useful for diverging cmaps.
    norm
        Explicit :class:`~matplotlib.colors.Normalize` instance, overriding
        ``vmin``/``vmax``/``vcenter``.
    prob_vmin, prob_vmax
        Limits for the opacity scale. Same flexible types as ``vmin``/``vmax``.
        ``None`` uses ``color_prob`` min/max.
    alpha_min, alpha_max
        Output opacity range for ``color_prob``. ``alpha_min`` is applied to
        the smallest displayed value, ``alpha_max`` to the largest.
    opacity_color
        Reference colour used to render the opacity colourbar (a fade from the
        figure's facecolor to fully solid). Pure visual choice; doesn't affect
        any plotted point colours.
    sort_order
        If ``True`` (default), points are sorted by ``color_prob`` ascending so
        high-confidence points render on top of low-confidence ones — important
        when per-point alpha is in play. If ``False``, points are plotted in
        ``adata.obs`` order.
    legend_loc
        Location of the *categorical* colour legend. ``"right margin"`` (default)
        places it outside the axes on the right; ``"on data"`` overlays the
        category name at each cluster's centroid; ``"none"`` suppresses the
        categorical legend; any other matplotlib location string
        (``"best"``, ``"upper left"``, ...) is passed through. Numeric colour
        always uses ``colorbar_loc`` instead.
    legend_fontsize
        Font size of legend text and ``"on data"`` labels. ``None`` uses
        matplotlib's ``rcParams["legend.fontsize"]``.
    legend_fontweight
        Font weight of ``"on data"`` labels. Default ``"bold"`` (matches scanpy).
    legend_fontoutline
        Width (in points) of a white outline drawn around ``"on data"``
        labels for legibility on top of points. Only used when
        ``legend_loc="on data"``. ``None`` disables the outline.
    colorbar_loc
        Where to place the *numeric* colour colourbar
        (``"right" | "left" | "top" | "bottom"``). ``None`` suppresses the
        colourbar; the opacity colourbar is unaffected. Ignored for
        categorical ``color``.
    frameon
        If ``True`` (default), keep axis labels (matching scanpy). If
        ``False``, hide all spines and axis labels. Ticks are always hidden.
    title
        Plot title. Defaults to ``f"{color} (opacity ~ {color_prob})"``.
    ax
        Existing matplotlib axes to draw into. If ``None``, a new figure is
        created.
    figsize
        Figure size when ``ax`` is None.
    show
        If ``True``, call ``plt.show()`` at the end.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the scatter.

    Examples
    --------
    >>> import grassp as gr
    >>> adata = gr.ds.itzhak_2016()
    >>> # colour by compartment, fade points by how many maps they were profiled in
    >>> gr.pl.embedding_prob(
    ...     adata,
    ...     basis="X_umap",
    ...     color="compartment",
    ...     color_prob="Profiled in how many maps?",
    ... )
    """
    # ---------------------------------------------------------------- validate
    resolved_basis, coords = _get_basis(adata, basis)
    if color not in adata.obs.columns:
        raise KeyError(f"'{color}' not in adata.obs")
    if color_prob not in adata.obs.columns:
        raise KeyError(f"'{color_prob}' not in adata.obs")
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"adata.obsm['{resolved_basis}'] must be 2D with >=2 columns")
    x_full, y_full = coords[:, 0], coords[:, 1]
    n = adata.n_obs

    # ------------------------------------------------------------ probabilities
    probs = pd.to_numeric(adata.obs[color_prob], errors="coerce").to_numpy(dtype=float)
    pv_min = _resolve_vbound(prob_vmin, probs)
    if pv_min is None:
        pv_min = float(np.nanmin(probs)) if np.any(~np.isnan(probs)) else 0.0
    pv_max = _resolve_vbound(prob_vmax, probs)
    if pv_max is None:
        pv_max = float(np.nanmax(probs)) if np.any(~np.isnan(probs)) else 1.0
    if pv_max == pv_min:
        normed = np.ones_like(probs)
    else:
        normed = np.clip((probs - pv_min) / (pv_max - pv_min), 0.0, 1.0)
    alphas = alpha_min + (alpha_max - alpha_min) * normed
    alphas[np.isnan(probs)] = 0.0

    # ------------------------------------------------------------------ colors
    series = adata.obs[color]
    if series.dtype == bool:
        series = series.astype(str).astype("category")
    is_categorical = isinstance(series.dtype, pd.CategoricalDtype) or series.dtype == object

    na_rgba = np.array(to_rgba(na_color))

    if is_categorical:
        cats = series.astype("category")
        categories = cats.cat.categories
        if palette is None:
            uns_key = f"{color}_colors"
            if uns_key in adata.uns and len(adata.uns[uns_key]) >= len(categories):
                palette = list(adata.uns[uns_key][: len(categories)])
            else:
                cmap20 = plt.get_cmap("tab20")
                palette = [cmap20(i % 20) for i in range(len(categories))]
        palette_rgba = np.array([to_rgba(c) for c in palette])
        codes = cats.cat.codes.to_numpy()
        rgba = np.tile(na_rgba, (n, 1))
        valid = codes >= 0
        rgba[valid] = palette_rgba[codes[valid]]
        norm_obj = None
        cmap_obj = None
        has_na = (~valid).any()
    else:
        vals = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
        cmap_obj = plt.get_cmap(cmap).with_extremes(bad=na_color)
        norm_obj = _build_norm(vmin, vmax, vcenter, norm, vals)
        rgba = cmap_obj(norm_obj(vals))
        # cmap.with_extremes covers NaN, but only when norm returns masked. Belt-and-braces:
        nan_mask = np.isnan(vals)
        rgba[nan_mask] = na_rgba
        codes = None
        categories = None
        palette_rgba = None
        has_na = nan_mask.any()

    rgba[:, 3] = alphas

    # ------------------------------------------------------------------- order
    if sort_order:
        # Plot high-confidence points on top so opacity is honoured in dense
        # regions. NaN probabilities go to bottom (treated as smallest).
        sort_key = np.where(np.isnan(probs), -np.inf, probs)
        order = np.argsort(sort_key, kind="stable")
    else:
        order = np.arange(n)

    x_plot = x_full[order]
    y_plot = y_full[order]
    rgba_plot = rgba[order]

    # ------------------------------------------------------------------ figure
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    else:
        fig = ax.figure

    if size is None:
        size = max(120000.0 / n, 4.0)

    ax.scatter(
        x_plot,
        y_plot,
        c=rgba_plot,
        s=size,
        marker=marker,
        edgecolors="none",
        linewidths=0,
        plotnonfinite=True,
        rasterized=True,
    )
    ax.autoscale_view()
    ax.set_aspect("equal", adjustable="datalim")

    name = _basis2name(resolved_basis)
    ax.set_xticks([])
    ax.set_yticks([])
    if frameon:
        ax.set_xlabel(f"{name}1")
        ax.set_ylabel(f"{name}2")
        ax.spines[["top", "right"]].set_visible(False)
    else:
        for spine in ax.spines.values():
            spine.set_visible(False)
    ax.set_title(title if title is not None else f"{color} (opacity ~ {color_prob})")

    # ----------------------------------------------------- color legend / cbar
    if is_categorical:
        if legend_loc == "none":
            pass
        elif legend_loc == "on data":
            for i, cat in enumerate(categories):
                mask = codes == i
                if not mask.any():
                    continue
                cx = float(np.median(x_full[mask]))
                cy = float(np.median(y_full[mask]))
                txt = ax.text(
                    cx,
                    cy,
                    str(cat),
                    fontsize=legend_fontsize,
                    fontweight=legend_fontweight,
                    ha="center",
                    va="center",
                )
                if legend_fontoutline:
                    txt.set_path_effects(
                        [
                            patheffects.withStroke(
                                linewidth=legend_fontoutline, foreground="white"
                            )
                        ]
                    )
        else:
            handles = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="",
                    markerfacecolor=palette_rgba[i],
                    markeredgecolor="none",
                    markersize=8,
                    label=str(categories[i]),
                )
                for i in range(len(categories))
            ]
            if has_na and na_in_legend:
                handles.append(
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        linestyle="",
                        markerfacecolor=tuple(na_rgba),
                        markeredgecolor="none",
                        markersize=8,
                        label="NA",
                    )
                )
            n_handles = len(handles)
            ncol = 1 if n_handles <= 14 else 2 if n_handles <= 30 else 3
            if legend_loc == "right margin":
                ax.legend(
                    handles=handles,
                    title=color,
                    loc="center left",
                    bbox_to_anchor=(1.02, 0.5),
                    frameon=False,
                    fontsize=legend_fontsize,
                    ncol=ncol,
                )
            else:
                ax.legend(
                    handles=handles,
                    title=color,
                    loc=legend_loc,
                    frameon=False,
                    fontsize=legend_fontsize,
                    ncol=ncol,
                )
    else:
        if colorbar_loc is not None:
            sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm_obj)
            sm.set_array([])
            fig.colorbar(
                sm,
                ax=ax,
                label=color,
                fraction=0.04,
                pad=0.04,
                location=colorbar_loc,
            )

    # --------------------------------------------------------- opacity colorbar
    base = np.array(to_rgba(opacity_color)[:3])
    bg_rgba = np.array(to_rgba(fig.get_facecolor()))
    bg = bg_rgba[:3] if bg_rgba[3] > 0 else np.array([1.0, 1.0, 1.0])
    n_steps = 256
    bar_alphas = np.linspace(alpha_min, alpha_max, n_steps)
    blended = bar_alphas[:, None] * base + (1.0 - bar_alphas[:, None]) * bg
    op_cmap = LinearSegmentedColormap.from_list(
        "opacity_bar", np.column_stack([blended, np.ones(n_steps)])
    )
    op_sm = plt.cm.ScalarMappable(cmap=op_cmap, norm=plt.Normalize(vmin=pv_min, vmax=pv_max))
    op_sm.set_array([])
    fig.colorbar(op_sm, ax=ax, label=f"{color_prob} (opacity)", fraction=0.04, pad=0.12)

    if show:
        plt.show()
    return ax


def umap_prob(adata: AnnData, color: str, color_prob: str, **kwargs) -> Axes:
    """Convenience wrapper for :func:`embedding_prob` on ``adata.obsm["X_umap"]``.

    ``color``, ``color_prob`` and any extra keyword arguments are forwarded to
    :func:`embedding_prob`; see there for the full parameter list.
    """
    return embedding_prob(adata, "X_umap", color, color_prob, **kwargs)


def pca_prob(adata: AnnData, color: str, color_prob: str, **kwargs) -> Axes:
    """Convenience wrapper for :func:`embedding_prob` on ``adata.obsm["X_pca"]``.

    ``color``, ``color_prob`` and any extra keyword arguments are forwarded to
    :func:`embedding_prob`; see there for the full parameter list.
    """
    return embedding_prob(adata, "X_pca", color, color_prob, **kwargs)


def tsne_prob(adata: AnnData, color: str, color_prob: str, **kwargs) -> Axes:
    """Convenience wrapper for :func:`embedding_prob` on ``adata.obsm["X_tsne"]``.

    ``color``, ``color_prob`` and any extra keyword arguments are forwarded to
    :func:`embedding_prob`; see there for the full parameter list.
    """
    return embedding_prob(adata, "X_tsne", color, color_prob, **kwargs)


def umap(
    adata: AnnData,
    highlight: Sequence[str] | None = None,
    annotate_by: str | None = None,
    highlight_edgecolor: str = "black",
    highlight_linewidth: float = 1.5,
    annotate_fontsize: float | int = 8,
    annotate_color: str = "black",
    annotate_offset: tuple[float, float] = (5, 5),
    annotate_fontoutline: int | None = 2,
    annotate_adjust: bool = True,
    adjust_text_kwargs: dict = dict(arrowprops=dict(color="grey", lw=0.5)),
    ax: Axes | None = None,
    show: bool | None = None,
    save: bool | str | None = None,
    **kwargs,
) -> Axes | None:
    """
    Wrapper around sc.pl.umap that highlights specific observations with outlined markers.

    This function creates a UMAP plot using scanpy's plotting function and adds the
    ability to highlight specific observations (e.g., proteins) with a black outline
    and optionally annotate them with text labels from a specified column.

    Parameters
    ----------
    adata
        Annotated data matrix with UMAP coordinates in `.obsm['X_umap']`.
    highlight
        List of observation names (from `adata.obs_names`) to highlight with outlines.
        If None, no observations are highlighted.
    annotate_by
        Column name in `adata.obs` to use for annotating highlighted observations.
        If None, no annotations are added. Only highlighted observations are annotated.
    highlight_edgecolor
        Color of the outline around highlighted points.
    highlight_linewidth
        Width of the outline around highlighted points.
    annotate_fontsize
        Font size for annotation text.
    annotate_color
        Color of annotation text.
    annotate_offset
        Tuple of (x, y) offset in points for annotation text positioning relative
        to the highlighted markers. Only used if `annotate_adjust` is False.
    annotate_fontoutline
        Width of the outline around annotation text. Only used if `annotate_adjust` is True.
    annotate_adjust
        If True, uses the adjustText library to automatically adjust label positions
        to avoid overlaps. If False, uses fixed offset positioning.
    adjust_text_kwargs
        Additional keyword arguments passed to `adjust_text()`.
    ax
        A matplotlib axes object. If None, uses current axes.
    show
        Whether to show the figure. If None, uses scanpy settings.
    save
        If True or a string, save the figure. A string is used as filename.
    **kwargs
        Additional keyword arguments passed to `sc.pl.umap()`.

    Returns
    -------
    If `show==False`, returns matplotlib axes object. Otherwise returns None.

    Examples
    --------
    >>> import grassp as gr
    >>> adata = gr.datasets.itzhak_2016()
    >>> adata.obs.set_index("Lead ID", inplace=True)
    >>> # Highlight specific proteins with annotation
    >>> gr.pl.umap(
    ...    adata,
    ...    color="compartment",
    ...    highlight=["P00533", "P29353"],
    ...    annotate_by="Lead gene name")
    """
    # Store original show parameter
    show = sc.settings.autoshow if show is None else show

    # Get or create axes
    if ax is None:
        ax = plt.gca()

    # Call sc.pl.umap with show=False to get the axes back
    sc.pl.umap(adata, ax=ax, show=False, save=False, **kwargs)

    # If highlight is specified, add outlines and annotations
    if highlight is not None and len(highlight) > 0:
        # Get UMAP coordinates
        if "X_umap" not in adata.obsm:
            raise ValueError('UMAP coordinates not found in adata.obsm["X_umap"]')

        umap_coords = adata.obsm["X_umap"]

        # Find indices of highlighted observations
        highlight_indices = []
        highlight_names = []
        for obs_name in highlight:
            if obs_name in adata.obs_names:
                idx = adata.obs_names.get_loc(obs_name)
                highlight_indices.append(idx)
                highlight_names.append(obs_name)
            else:
                print(f"Warning: '{obs_name}' not found in adata.obs_names, skipping")

        if len(highlight_indices) > 0:
            # Get coordinates for highlighted points
            highlight_coords = umap_coords[highlight_indices]

            # Plot highlighted points with outline
            # We need to get the facecolors from the original scatter plot
            # to maintain color consistency
            original_collection = ax.collections[0]
            original_colors = original_collection.get_facecolors()

            # Get colors for highlighted points
            if len(original_colors) == len(adata.obs_names):
                # Colors are per observation
                highlight_colors = original_colors[highlight_indices]
            elif len(original_colors) == 1:
                # Single color for all points
                highlight_colors = original_colors
            else:
                # Fallback to original colors
                highlight_colors = original_colors[0:1]

            # Plot highlighted points with edge
            ax.scatter(
                highlight_coords[:, 0],
                highlight_coords[:, 1],
                c=highlight_colors,
                edgecolors=highlight_edgecolor,
                linewidths=highlight_linewidth,
                s=(
                    original_collection.get_sizes()[0]
                    if len(original_collection.get_sizes()) > 0
                    else 20
                ),
                zorder=10,
            )

            # Add annotations if requested
            if annotate_by is not None:
                if annotate_by not in adata.obs.columns:
                    raise ValueError(f"Column '{annotate_by}' not found in adata.obs")

                path_effect = [withStroke(linewidth=annotate_fontoutline, foreground="w")]
                if annotate_adjust:
                    # Use adjustText for automatic label placement
                    texts = []
                    for i, obs_name in enumerate(highlight_names):
                        label = str(adata.obs.loc[obs_name, annotate_by])
                        text = ax.text(
                            highlight_coords[i, 0],
                            highlight_coords[i, 1],
                            label,
                            fontsize=annotate_fontsize,
                            path_effects=path_effect,
                            color=annotate_color,
                            zorder=11,
                        )
                        texts.append(text)
                    # Adjust text positions to avoid overlaps
                    if "min_arrow_len" not in adjust_text_kwargs:
                        adjust_text_kwargs["min_arrow_len"] = (
                            ax.get_xlim()[1] - ax.get_xlim()[0]
                        ) * 0.01

                    adjust_text(texts, ax=ax, **adjust_text_kwargs)
                else:
                    # Use fixed offset positioning
                    for i, obs_name in enumerate(highlight_names):
                        label = str(adata.obs.loc[obs_name, annotate_by])
                        ax.annotate(
                            label,
                            xy=(highlight_coords[i, 0], highlight_coords[i, 1]),
                            xytext=annotate_offset,
                            textcoords="offset points",
                            fontsize=annotate_fontsize,
                            path_effects=path_effect,
                            color=annotate_color,
                            zorder=11,
                        )

    # Handle show and save
    sc.pl._utils.savefig_or_show("umap", show=show, save=save)
    if show:
        return None
    return ax


def pretty_embedding(
    adata: AnnData,
    color: str | List[str] | None = None,
    basis: str = "X_umap",
    figsize: tuple[float, float] = (8, 8),
    dpi: int = 200,
    legend_fontsize: int = 5,
    label_fontsize: int = 12,
    label_stroke_width: float = 0.0,
    label_stroke_color: str = "darkgray",
    edge_frac: float = 0.97,
    label_plot_lines: bool = True,
    vector_text: bool = True,
    **kwargs,
) -> Axes:
    """
    Create a publication-quality embedding plot with automatic label placement.

    This function wraps scanpy's embedding plot and adds automatically positioned
    labels along the convex hull boundary using a force-directed layout algorithm.
    Labels are placed to avoid overlap with each other and with data points.

    Parameters
    ----------
    adata
        Annotated data matrix with embedding coordinates in `.obsm[basis]`.
    color
        Key in `adata.obs` or `adata.var_names` for coloring the points.
        If None, all points are colored the same.
    basis
        Key in `.obsm` for the embedding coordinates (e.g., 'X_umap', 'X_pca').
    figsize
        Figure size as (width, height) in inches.
    dpi
        Figure resolution in dots per inch.
    legend_fontsize
        Font size for the legend (if shown).
    label_fontsize
        Font size for cluster/group labels placed on the plot.
    label_stroke_width
        Width of the white outline/stroke (halo) drawn around label text for
        readability. Default ``0.0`` (no halo) so that labels export as fully
        editable text in vector formats — a non-zero halo is drawn with a
        matplotlib path-effect, which outlines the label glyphs to vector paths
        and makes them no longer selectable/editable as text in Illustrator. Set
        a value like ``1.0`` if on-screen readability matters more than
        vector-text editability.
    label_stroke_color
        Color of the outline/stroke (halo) around label text. Only used when
        ``label_stroke_width`` is non-zero.
    edge_frac
        Controls how close to the convex hull edge labels are placed (0-1).
        Higher values place labels closer to the edge.
    label_plot_lines
        Whether to draw connector lines (arrows) from labels to cluster centroids.
    vector_text
        If ``True`` (default), set matplotlib's rcParams so that text in vector
        exports (PDF/EPS/SVG) stays as editable TrueType text rather than Type-3
        fonts or outlined paths — i.e. selectable and editable when the figure is
        opened in Illustrator. This updates the *global* rcParams (needed because
        they are read at ``savefig`` time, after this function returns).
    **kwargs
        Additional keyword arguments passed to `sc.pl.embedding()`.

    Returns
    -------
    Matplotlib axes object containing the plot.

    Examples
    --------
    >>> import grassp as gr
    >>> adata = gr.ds.itzhak_2016()
    >>> gr.pl.pretty_embedding(adata, color="compartment")
    """
    if vector_text:
        _set_vector_friendly_fonts()
    with plt.rc_context(
        {
            "figure.figsize": figsize,
            "figure.dpi": dpi,
            "figure.frameon": False,
        }
    ):
        ax = sc.pl.embedding(
            adata,
            basis=basis,
            color=color,
            legend_loc=None,
            legend_fontsize=legend_fontsize,
            legend_fontoutline=True,
            add_outline=False,
            frameon=False,
            show=False,
            **kwargs,
        )

        # A path-effect (the readability halo) forces matplotlib to render the
        # label glyphs as outlined vector paths, which defeats ``vector_text``
        # for exactly these labels. Only add the halo when a stroke width is
        # requested; ``label_stroke_width=0``/``None`` keeps labels as editable
        # text in PDF/SVG.
        text_kwargs = dict(fontsize=label_fontsize)
        if label_stroke_width:
            text_kwargs["path_effects"] = [
                pe.withStroke(linewidth=label_stroke_width, foreground=label_stroke_color)
            ]
        gen_mpl_labels(
            adata,
            color,
            ax=ax,
            text_kwargs=text_kwargs,
            color_by_group=True,
            basis_key=basis,
            edge_frac=edge_frac,
            plot_lines=label_plot_lines,
        )
        return ax


def _compute_convex_hull(pts):
    """
    Compute convex hull of 2D points using scipy.

    Parameters
    ----------
    pts : array-like
        2D points as an (N, 2) array.

    Returns
    -------
    hull_points : ndarray
        Convex hull vertices as an (H, 2) array in counter-clockwise order.
    """
    pts = np.asarray(pts, float)
    pts = np.unique(pts, axis=0)
    if len(pts) <= 2:
        return pts
    hull_obj = ConvexHull(pts)
    # Get hull vertices in counter-clockwise order (scipy returns them in CCW for 2D)
    return pts[hull_obj.vertices]


def _ray_segment_intersection(C, v, A, B):
    """
    Compute intersection between a ray and a line segment.

    Solves for ray C + t*v (where t >= 0) intersecting segment A->B.

    Parameters
    ----------
    C : array-like
        Ray origin point (2D).
    v : array-like
        Ray direction vector (2D).
    A : array-like
        Segment start point (2D).
    B : array-like
        Segment end point (2D).

    Returns
    -------
    t : float or None
        Ray parameter at intersection (distance along ray), or None if no intersection.
    u : float or None
        Segment parameter at intersection (0 to 1), or None if no intersection.
    """
    # Solve C + t v = A + u (B-A), t>=0, u in [0,1]
    r = v
    s = B - A
    den = r[0] * s[1] - r[1] * s[0]  # 2D cross(r,s)
    if np.isclose(den, 0.0):
        return (None, None)
    AC = A - C
    t = (AC[0] * s[1] - AC[1] * s[0]) / den
    u = (AC[0] * r[1] - AC[1] * r[0]) / den
    if t >= 0 and 0 <= u <= 1:
        return (t, u)
    return (None, None)


def _project_to_hull(center, p, hull, edge_frac=0.98):
    """
    Project a point onto the convex hull boundary along a ray from the center.

    Projects point p onto the convex hull boundary by casting a ray from center
    through p and finding the first intersection with the hull. The projection
    is scaled by edge_frac to place it slightly inside or outside the hull.

    Parameters
    ----------
    center : array-like
        Center point from which to cast the ray (typically median of all points).
    p : array-like
        Point to project onto the hull (2D).
    hull : ndarray
        Convex hull vertices as an (H, 2) array in counter-clockwise order.
    edge_frac : float, default=0.98
        Fraction of distance to hull edge (0-1). Values <1 place the projection
        inside the hull, >1 places it outside.

    Returns
    -------
    proj : ndarray
        Projected point on (or near) the hull boundary (2D).
    """
    v = np.asarray(p, float) - np.asarray(center, float)
    if np.allclose(v, 0):
        v = np.array([_FALLBACK_VECTOR_EPSILON, 0.0])
    # find first intersection
    ts = []
    for i in range(len(hull)):
        A = hull[i]
        B = hull[(i + 1) % len(hull)]
        t, _ = _ray_segment_intersection(center, v, A, B)
        if t is not None:
            ts.append(t)
    if not ts:
        # fallback: just return p
        return p
    tmin = min(ts)
    proj = np.asarray(center) + edge_frac * tmin * v
    return proj


def _nearest_point_query_builder(X):
    """
    Build a nearest-point query function using KDTree.

    Parameters
    ----------
    X : array-like
        2D points as an (N, 2) array.

    Returns
    -------
    query_fn : callable
        Function that takes a point and returns the nearest point in X.
    X : ndarray
        The input points array.
    """
    X = np.asarray(X, float)
    tree = KDTree(X)

    def q(p):
        _, idx = tree.query(p, k=1)
        return X[int(idx)]

    return q, X


def gen_mpl_labels(
    adata,
    groupby,
    exclude=(),
    ax=None,
    text_kwargs=None,
    color_by_group=False,
    basis_key="X_umap",
    edge_frac=0.98,
    # objective hyperparams
    n_iter=180,
    lr=0.06,
    k_attr=1.0,
    k_label=2.1,
    k_point=0.8,
    min_d2=_MIN_DISTANCE_SQ,
    max_step=0.2,
    # alignment & nudge
    horiz_bias=1.15,
    nudge_frac=0.012,
    arrowprops=None,
    plot_lines=True,
):
    """
    Place group labels along the convex hull outline of an embedding plot.

    This function uses a force-directed layout algorithm to automatically position
    labels for groups/clusters along the convex hull boundary. Labels are initialized
    near their group medians and iteratively repositioned using three forces:

    1. Attraction to cluster median (k_attr)
    2. Repulsion between labels (k_label)
    3. Repulsion from nearest data points (k_point)

    After optimization, labels are projected onto the hull boundary and text alignment
    is automatically determined based on the outward direction from the nearest point.
    Arrows connect labels back to their cluster medians.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with embedding coordinates in `.obsm[basis_key]`.
    groupby : str
        Column name in `adata.obs` containing group/cluster labels.
    exclude : tuple of str, default=()
        Group names to exclude from labeling.
    ax : plt.Axes, optional
        Matplotlib axes object. If None, uses current axes.
    text_kwargs : dict, optional
        Additional keyword arguments passed to `ax.text()` for label styling.
    color_by_group : bool, default=False
        If True, labels are colored according to `{groupby}_colors` in `adata.uns`.
    basis_key : str, default='X_umap'
        Key in `.obsm` for the embedding coordinates.
    edge_frac : float, default=0.98
        Fraction of distance to convex hull edge for label placement (0-1).
    n_iter : int, default=180
        Number of optimization iterations for force-directed layout.
    lr : float, default=0.06
        Learning rate for gradient descent optimization.
    k_attr : float, default=1.0
        Weight for attraction force toward cluster median.
    k_label : float, default=2.1
        Weight for label-label repulsion force.
    k_point : float, default=0.8
        Weight for label-point repulsion force.
    min_d2 : float, default=1e-4
        Minimum squared distance for repulsion (prevents singularities).
    max_step : float, default=0.2
        Maximum step size per iteration (prevents overshooting).
    horiz_bias : float, default=1.15
        Threshold for preferring horizontal/vertical text alignment.
        Higher values favor cardinal directions over diagonals.
    nudge_frac : float, default=0.012
        Fraction of axis span to nudge labels away from anchor points.
    arrowprops : dict, optional
        Arrow style properties passed to `ax.annotate()` for connector arrows.
    plot_lines : bool, default=True
        Whether to draw connector lines (arrows) from labels to cluster medians.

    Returns
    -------
    texts : list of matplotlib.text.Text
        List of text objects for the placed labels.
    """
    if text_kwargs is None:
        text_kwargs = {}
    if ax is None:
        ax = plt.gca()
    if arrowprops is None:
        arrowprops = dict(arrowstyle="-", lw=0.8, alpha=0.7, color="black")

    # --- stable groups ---
    X = np.asarray(adata.obsm[basis_key], float)
    center = np.median(X, axis=0)
    g_to_idx = adata.obs.groupby(groupby).groups
    cats = list(adata.obs[groupby].cat.categories)
    groups = [g for g in cats if (g in g_to_idx) and (g not in exclude)]
    if not groups:
        return []

    medians = np.vstack(
        [np.median(adata[g_to_idx[g]].obsm[basis_key], axis=0) for g in groups]
    )

    # palette-aligned colors
    text_colors = {g: None for g in cats}
    if color_by_group and groupby + "_colors" in adata.uns:
        for i, g in enumerate(cats):
            text_colors[g] = adata.uns[groupby + "_colors"][i]

    # --- hull + NN ---
    hull = _compute_convex_hull(X)
    nn_query, _ = _nearest_point_query_builder(X)

    # --- init on hull ---
    P = np.vstack([_project_to_hull(center, m, hull, edge_frac=edge_frac) for m in medians])

    # --- optimize (same forces as before) ---
    for _ in range(n_iter):
        F = np.zeros_like(P)
        F += k_attr * (medians - P)  # attract to medians

        # label-label repulsion
        for i in range(len(P)):
            d = P[i] - P
            d2 = np.sum(d * d, axis=1) + min_d2
            d2[i] = np.inf
            F[i] += k_label * (d / d2[:, None]).sum(axis=0)

        # nearest-point repulsion (query all points at once for efficiency)
        nearest_points = np.array([nn_query(p) for p in P])
        dq = P - nearest_points
        d2q = np.sum(dq * dq, axis=1, keepdims=True) + min_d2
        F += k_point * (dq / d2q)

        step = lr * F
        step_norm = np.linalg.norm(step, axis=1, keepdims=True) + _NUMERICAL_EPSILON
        step = step * np.minimum(1.0, (max_step / step_norm))
        P = P + step

        # keep on the outline
        P = np.vstack([_project_to_hull(center, p, hull, edge_frac=edge_frac) for p in P])

    # --- alignment based on direction AWAY from nearest point ---
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    xspan = x1 - x0
    yspan = y1 - y0

    def _ha_va_and_nudge(p):
        q = nn_query(p)  # nearest point (inward)
        v = p - q  # outward direction (away from nearest point)
        # Decide alignment from signs; bias toward horizontal/vertical when one dominates.
        ha = "center"
        va = "center"
        if abs(v[0]) > horiz_bias * abs(v[1]):
            ha = "left" if v[0] > 0 else "right"
            va = "center"
        elif abs(v[1]) > horiz_bias * abs(v[0]):
            va = "bottom" if v[1] > 0 else "top"
            ha = "center"
        else:
            # diagonal: pick both
            ha = "left" if v[0] > 0 else "right"
            va = "bottom" if v[1] > 0 else "top"

        # small outward nudge so text sits "outside" its anchor
        norm = np.linalg.norm(v) + _NUMERICAL_EPSILON
        u = v / norm
        nudge = nudge_frac * np.array([u[0] * xspan, u[1] * yspan])
        return ha, va, p + nudge

    # --- draw texts & arrows, preserving group order ---
    texts = []
    for g, p, m in zip(groups, P, medians):
        ha, va, pn = _ha_va_and_nudge(p)
        t = ax.text(
            pn[0],
            pn[1],
            s=g,
            color=text_colors.get(g),
            horizontalalignment=ha,
            verticalalignment=va,
            zorder=10,
            **text_kwargs,
        )
        texts.append(t)
        # arrow back to the median
        if plot_lines:
            ax.annotate(
                "",
                xy=(m[0], m[1]),
                xytext=(pn[0], pn[1]),
                arrowprops=arrowprops,
                zorder=9,
            )

    ax.margins(x=0.06, y=0.06)
    return texts
