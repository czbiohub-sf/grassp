"""Embedding scatter plots: per-point opacity, hull-placed group labels, and
highlighting of individual observations."""

from __future__ import annotations
import warnings

from typing import TYPE_CHECKING, Literal, Sequence

if TYPE_CHECKING:
    from anndata import AnnData
    from matplotlib.axes import Axes
    from matplotlib.colors import Normalize
    from matplotlib.text import Text

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

from adjustText import adjust_text
from matplotlib import patheffects
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, to_rgba
from matplotlib.lines import Line2D
from scipy.spatial import ConvexHull, KDTree  # pylint: disable=no-name-in-module

# Module-level constants for numerical stability and algorithmic parameters
_NUMERICAL_EPSILON = 1e-12  # Small value to prevent division by zero
_MIN_DISTANCE_SQ = 1e-4  # Minimum distance squared for repulsion forces
_FALLBACK_VECTOR_EPSILON = 1e-6  # Fallback when center equals point
# Absolute tolerance for the hull-projection geometry: below this a ray and a hull
# edge count as parallel, and a ray counts as zero-length. Matches the default atol
# of the ``np.isclose``/``np.allclose`` calls these tests replaced, which cost ~190x
# more per call and ran once per hull edge per label per iteration.
_GEOMETRY_ATOL = 1e-8

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
    adjust_text_kwargs: dict | None = None,
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
        Width of the white outline drawn around annotation text, or None for no
        outline.
    annotate_adjust
        If True, uses the adjustText library to automatically adjust label positions
        to avoid overlaps. If False, uses fixed offset positioning.
    adjust_text_kwargs
        Additional keyword arguments passed to `adjust_text()`. Defaults to grey
        connector arrows. `min_arrow_len` is derived from the axes width unless
        given explicitly.
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
    show = sc.settings.autoshow if show is None else show
    # Copied rather than mutated in place: this dict picks up a `min_arrow_len`
    # scaled to the current axes below, and writing that back into a caller's dict
    # (or, when it was a default argument, into a dict shared by every call) would
    # pin every later plot to the first plot's scale.
    adjust_text_kwargs = (
        dict(arrowprops=dict(color="grey", lw=0.5))
        if adjust_text_kwargs is None
        else dict(adjust_text_kwargs)
    )

    if ax is None:
        ax = plt.gca()

    # show=False so the axes come back for the highlight overlay below
    sc.pl.umap(adata, ax=ax, show=False, save=False, **kwargs)

    if highlight is not None and len(highlight) > 0:
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
                warnings.warn(
                    f"{obs_name!r} not found in adata.obs_names, skipping",
                    stacklevel=2,
                )

        if len(highlight_indices) > 0:
            # Get coordinates for highlighted points
            highlight_coords = umap_coords[highlight_indices]

            # Reuse the scatter's own facecolors so the highlighted points keep
            # the colour they already had, and only gain an outline.
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
                # Unexpected length (e.g. scanpy changed its scatter layout);
                # fall back to the first colour rather than mis-pairing them.
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

                # linewidth=None is not a valid stroke, so `None` has to mean
                # "no outline" rather than being handed to withStroke. Mirrors how
                # scanpy treats its own `legend_fontoutline`.
                path_effect = (
                    [patheffects.withStroke(linewidth=annotate_fontoutline, foreground="w")]
                    if annotate_fontoutline is not None
                    else None
                )
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
    color: str | None = None,
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
        Categorical key in `adata.obs` to colour and label groups by. A single key
        only -- the hull label pass places one label per category, so it has no
        meaning for a list of keys or for a continuous variable. If None, points
        are drawn in one colour and no labels are placed.
    basis
        Key in `.obsm` for the embedding coordinates (e.g., 'X_umap', 'X_pca').
    figsize
        Figure size as (width, height) in inches.
    dpi
        Figure resolution in dots per inch.
    legend_fontsize
        Font size for the legend. No legend is drawn by default (the hull labels
        replace it); this applies only if one is re-enabled by passing
        ``legend_loc`` through ``**kwargs``.
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
        Additional keyword arguments passed to `sc.pl.embedding()`. The defaults
        this function sets -- ``legend_loc``, ``frameon`` and ``add_outline`` --
        can be overridden here.

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
        # setdefault rather than positional so callers can override any of them;
        # the hull labels stand in for a legend, hence legend_loc=None.
        kwargs.setdefault("legend_loc", None)
        kwargs.setdefault("add_outline", False)
        kwargs.setdefault("frameon", False)
        ax = sc.pl.embedding(
            adata,
            basis=basis,
            color=color,
            legend_fontsize=legend_fontsize,
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
                patheffects.withStroke(
                    linewidth=label_stroke_width, foreground=label_stroke_color
                )
            ]
        # Nothing to label without a grouping column; the plain scatter is the
        # whole figure in that case.
        if color is not None:
            _gen_mpl_labels(
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


def _compute_convex_hull(pts: np.ndarray) -> np.ndarray:
    """Convex hull of 2D points, as an (H, 2) array of vertices in
    counter-clockwise order. Degenerate inputs (<=2 distinct points) are returned
    as-is, which ``_project_to_hull`` treats as "no edge to hit"."""
    pts = np.unique(np.asarray(pts, float), axis=0)
    if len(pts) <= 2:
        return pts
    return pts[ConvexHull(pts).vertices]


def _project_to_hull(
    center: np.ndarray, points: np.ndarray, hull: np.ndarray, edge_frac: float = 0.98
) -> np.ndarray:
    """Project each of ``points`` onto the convex hull boundary along its own ray
    from ``center``.

    Takes an (N, 2) array and returns one. For each point, the first hull crossing
    is scaled by ``edge_frac``, so values below 1 land just inside the hull and
    above 1 just outside; points whose ray misses every edge are returned
    unchanged.

    Every point is tested against every hull edge in one pass. This runs twice per
    optimizer iteration, and the nested Python loops it replaced (per point, then
    per edge) dominated the runtime of ``_gen_mpl_labels``.
    """
    center = np.asarray(center, float)
    pts = np.atleast_2d(np.asarray(points, float))
    rays = pts - center
    # A point sitting exactly on the center has no direction to project along;
    # send it out along +x instead. Cheap scalar test rather than np.allclose,
    # which was itself a third of this function's cost.
    degenerate = np.abs(rays).max(axis=1) <= _GEOMETRY_ATOL
    if degenerate.any():
        rays = rays.copy()
        rays[degenerate] = (_FALLBACK_VECTOR_EPSILON, 0.0)

    edges = np.asarray(hull, float)
    if len(edges) < 2:
        return pts.copy()
    # Edge i runs edges[i] -> edges[i] + spans[i]. Solve center + t*ray == edge +
    # u*span for every (point, edge) pair and keep crossings with t >= 0 (ahead of
    # the ray) and u in [0, 1] (inside the segment). `den` is the 2D cross product
    # ray x span, which vanishes when the two are parallel. Numerators that depend
    # only on the edge are computed once and broadcast.
    spans = np.roll(edges, -1, axis=0) - edges
    offsets = edges - center
    t_num = offsets[:, 0] * spans[:, 1] - offsets[:, 1] * spans[:, 0]
    den = np.outer(rays[:, 0], spans[:, 1]) - np.outer(rays[:, 1], spans[:, 0])
    with np.errstate(divide="ignore", invalid="ignore"):
        t = t_num / den
        u = (np.outer(rays[:, 1], offsets[:, 0]) - np.outer(rays[:, 0], offsets[:, 1])) / den
    hit = (np.abs(den) > _GEOMETRY_ATOL) & (t >= 0) & (u >= 0) & (u <= 1)

    # Nearest crossing per point; points that hit nothing keep their input
    # position, and are left out of the arithmetic rather than being computed
    # from an infinite t and overwritten afterwards.
    t_first = np.where(hit, t, np.inf).min(axis=1)
    found = np.isfinite(t_first)
    projected = pts.copy()
    projected[found] = center + edge_frac * t_first[found, None] * rays[found]
    return projected


def _gen_mpl_labels(
    adata: AnnData,
    groupby: str,
    exclude: Sequence[str] = (),
    ax: Axes | None = None,
    text_kwargs: dict | None = None,
    color_by_group: bool = False,
    basis_key: str = "X_umap",
    edge_frac: float = 0.98,
    # objective hyperparams
    n_iter: int = 180,
    lr: float = 0.06,
    k_attr: float = 1.0,
    k_label: float = 2.1,
    k_point: float = 0.8,
    min_d2: float = _MIN_DISTANCE_SQ,
    max_step: float = 0.2,
    # alignment & nudge
    horiz_bias: float = 1.15,
    nudge_frac: float = 0.012,
    arrowprops: dict | None = None,
    plot_lines: bool = True,
) -> list[Text]:
    """
    Place group labels along the convex hull outline of an embedding plot.

    Labels are initialized near their group medians and iteratively repositioned
    by a force-directed layout with three forces:

    1. Attraction to the cluster median (``k_attr``)
    2. Repulsion between labels (``k_label``)
    3. Repulsion from the nearest data point (``k_point``)

    After each step labels are re-projected onto the hull, so they travel along
    the outline rather than drifting into the point cloud. Text alignment is then
    chosen from the outward direction at each final position, and arrows connect
    the labels back to their cluster medians.

    Parameters
    ----------
    adata
        Annotated data matrix with embedding coordinates in `.obsm[basis_key]`.
    groupby
        Categorical column in `adata.obs` holding the group labels. Categories
        with no members are skipped.
    exclude
        Group names to leave unlabelled.
    ax
        Matplotlib axes object. If None, uses current axes.
    text_kwargs
        Additional keyword arguments passed to `ax.text()` for label styling.
    color_by_group
        If True, colour each label using `adata.uns[f"{groupby}_colors"]`.
    basis_key
        Key in `.obsm` for the embedding coordinates.
    edge_frac
        How close to the hull edge labels sit (0-1); higher is closer.
    n_iter
        Number of optimizer iterations.
    lr
        Step size multiplier for the force updates.
    k_attr
        Weight of the attraction toward the cluster median.
    k_label
        Weight of the label-label repulsion.
    k_point
        Weight of the label-point repulsion.
    min_d2
        Floor on squared distances in the repulsion terms, which keeps
        coincident points from producing infinite forces.
    max_step
        Maximum distance a label may move per iteration.
    horiz_bias
        How strongly to prefer cardinal over diagonal text alignment. One axis
        must dominate the other by this factor to win outright.
    nudge_frac
        Outward offset applied to each label, as a fraction of the axis span.
    arrowprops
        Arrow style properties passed to `ax.annotate()` for connector arrows.
    plot_lines
        Whether to draw connector arrows from labels to cluster medians.

    Returns
    -------
    The placed :class:`~matplotlib.text.Text` objects, in group order.
    """
    if text_kwargs is None:
        text_kwargs = {}
    if ax is None:
        ax = plt.gca()
    if arrowprops is None:
        arrowprops = dict(arrowstyle="-", lw=0.8, alpha=0.7, color="black")

    coords = np.asarray(adata.obsm[basis_key], float)
    center = np.median(coords, axis=0)

    # Membership straight off the categorical codes. A category that is declared
    # but unused -- routine after subsetting -- has no rows, so its median would
    # be nan and would later surface as an opaque "must be finite" error from the
    # KDTree query. Drop those categories instead of labelling them.
    col = adata.obs[groupby]
    cats = list(col.cat.categories)
    codes = np.asarray(col.cat.codes)
    members = [codes == i for i in range(len(cats))]
    keep = [i for i, g in enumerate(cats) if g not in exclude and members[i].any()]
    if not keep:
        return []
    groups = [cats[i] for i in keep]
    medians = np.vstack([np.median(coords[members[i]], axis=0) for i in keep])

    # Labels take the colour scanpy assigned each category, matched by position.
    text_colors: dict = {}
    if color_by_group:
        palette = adata.uns.get(f"{groupby}_colors", [])
        text_colors = {cats[i]: palette[i] for i in keep if i < len(palette)}

    hull = _compute_convex_hull(coords)
    tree = KDTree(coords)

    positions = _project_to_hull(center, medians, hull, edge_frac=edge_frac)

    for _ in range(n_iter):
        forces = k_attr * (medians - positions)  # attract to medians

        # label-label repulsion
        for i in range(len(positions)):
            d = positions[i] - positions
            d2 = np.sum(d * d, axis=1) + min_d2
            d2[i] = np.inf
            forces[i] += k_label * (d / d2[:, None]).sum(axis=0)

        # repulsion from the nearest data point, all labels in one batched query
        dq = positions - coords[tree.query(positions, k=1)[1]]
        d2q = np.sum(dq * dq, axis=1, keepdims=True) + min_d2
        forces += k_point * (dq / d2q)

        step = lr * forces
        step_norm = np.linalg.norm(step, axis=1, keepdims=True) + _NUMERICAL_EPSILON
        step = step * np.minimum(1.0, (max_step / step_norm))
        positions = positions + step

        # keep on the outline
        positions = _project_to_hull(center, positions, hull, edge_frac=edge_frac)

    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    xspan = x1 - x0
    yspan = y1 - y0
    # Direction away from the nearest data point, i.e. pointing out of the cloud.
    outward = positions - coords[tree.query(positions, k=1)[1]]

    def _ha_va_and_nudge(p: np.ndarray, v: np.ndarray) -> tuple[str, str, np.ndarray]:
        """Text alignment and a small outward offset for a label at ``p``, given
        the outward direction ``v``."""
        # Align from the signs of v, preferring horizontal/vertical when one axis
        # clearly dominates and falling back to a diagonal when neither does.
        if abs(v[0]) > horiz_bias * abs(v[1]):
            ha, va = ("left" if v[0] > 0 else "right"), "center"
        elif abs(v[1]) > horiz_bias * abs(v[0]):
            ha, va = "center", ("bottom" if v[1] > 0 else "top")
        else:
            ha = "left" if v[0] > 0 else "right"
            va = "bottom" if v[1] > 0 else "top"

        # small outward nudge so text sits "outside" its anchor
        unit = v / (np.linalg.norm(v) + _NUMERICAL_EPSILON)
        return ha, va, p + nudge_frac * np.array([unit[0] * xspan, unit[1] * yspan])

    texts = []
    for g, p, m, v in zip(groups, positions, medians, outward):
        ha, va, pn = _ha_va_and_nudge(p, v)
        texts.append(
            ax.text(
                pn[0],
                pn[1],
                s=g,
                color=text_colors.get(g),
                horizontalalignment=ha,
                verticalalignment=va,
                zorder=10,
                **text_kwargs,
            )
        )
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
