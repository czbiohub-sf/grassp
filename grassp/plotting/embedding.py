"""Embedding scatter plots with per-point opacity (e.g. confidence)."""

from __future__ import annotations
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from anndata import AnnData
    from matplotlib.axes import Axes
    from matplotlib.colors import Normalize

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib import patheffects
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, to_rgba
from matplotlib.lines import Line2D

# Why this is written from scratch rather than wrapping ``sc.pl.embedding``:
# - ``sc.pl.embedding`` exposes only a *scalar* ``alpha`` that applies to every
#   point. Per-point alpha would require post-hoc patching of the scatter
#   ``PathCollection`` (e.g. setting the 4th channel of ``get_facecolors()``),
#   which is brittle and tied to scanpy internals.
# - We need a second colorbar — one for the *opacity* mapping — that scanpy's
#   layout doesn't accommodate. Building the figure ourselves makes that clean.
# - Categorical palettes are still picked up from ``adata.uns[f"{color}_colors"]``
#   so visual consistency with ``sc.pl.umap`` etc. is preserved.


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
    """Convenience wrapper for :func:`embedding_prob` on ``adata.obsm["X_umap"]``."""
    return embedding_prob(adata, "X_umap", color, color_prob, **kwargs)


def pca_prob(adata: AnnData, color: str, color_prob: str, **kwargs) -> Axes:
    """Convenience wrapper for :func:`embedding_prob` on ``adata.obsm["X_pca"]``."""
    return embedding_prob(adata, "X_pca", color, color_prob, **kwargs)


def tsne_prob(adata: AnnData, color: str, color_prob: str, **kwargs) -> Axes:
    """Convenience wrapper for :func:`embedding_prob` on ``adata.obsm["X_tsne"]``."""
    return embedding_prob(adata, "X_tsne", color, color_prob, **kwargs)
