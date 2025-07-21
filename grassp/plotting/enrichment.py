from __future__ import annotations
from typing import TYPE_CHECKING

import pandas as pd
import seaborn as sns

if TYPE_CHECKING:
    from typing import List

    from anndata import AnnData


def plot_protein_enrichment_profiles(
    data: AnnData,
    protein_names: List[str],
    order: List[str] | None = None,
    compartment_col: str = "Compartment",
    fraction_or_ip_col: str = "subcellular_enrichment",
    **kwargs,
):
    """
    Visualizes multiple proteins across subcellular fractions or IPs, with optional legend filtering
    and styling control.

    This function plots log2 fold change profiles for one or more proteins across experimental
    fractions or IPs. It is designed to be flexible and visually clear when working with many proteins.

    Parameters
    ----------
    data : AnnData
        Annotated data matrix with expression values in `.X`, fraction info in `.var`,
        and compartment or condition metadata in `.obs`.

    protein_names : list of str
        List of protein IDs to include in the plot.

    order : list of str, optional
        List of labels defining the desired order for `fraction_or_ip_col` on the x-axis.
        Useful for preserving biological gradients (e.g., centrifugation fractions).

    compartment_col : str, default="Compartment"
        Name of the column in `data.obs` indicating the protein's subcellular compartment.

    fraction_or_ip_col : str, default="subcellular_enrichment"
        Column in `data.var` that provides annotation labels for each fraction (x-axis).


    Returns
    -------
    matplotlib.axes._subplots.AxesSubplot
        The seaborn lineplot object.

    Notes
    -----
    - Designed to reduce clutter when plotting many proteins by hiding redundant legend entries.
    - Legend filtering removes labels starting with 'P' or 'Q', assuming they are protein IDs.
    - Solid lines are used for consistency (`dashes=False`).
    """

    if compartment_col not in data.obs.columns:
        raise ValueError(f"{compartment_col} not found in data.obs")
    if fraction_or_ip_col not in data.var.columns:
        raise ValueError(f"{fraction_or_ip_col} not found in data.var")

    df = data.to_df()
    df[compartment_col] = data.obs[compartment_col].values
    df = df.reset_index()
    df = df[df["Protein IDs"].isin(protein_names)]

    melted_df = df.melt(
        id_vars=["Protein IDs", compartment_col],
        var_name="Fraction",
        value_name="log2_fold_change or proportion",
    )

    enrichment_map = data.var[fraction_or_ip_col].to_dict()
    melted_df[fraction_or_ip_col] = melted_df["Fraction"].map(enrichment_map)

    if order is not None:
        melted_df[fraction_or_ip_col] = pd.Categorical(
            melted_df[fraction_or_ip_col], categories=order, ordered=True
        )
        melted_df = melted_df.sort_values(by=["Protein IDs", fraction_or_ip_col])

    ax = sns.lineplot(
        data=melted_df,
        x=fraction_or_ip_col,
        y="log2_fold_change or proportion",
        hue=compartment_col,
        style="Protein IDs",
        dashes=False,
        **kwargs,
    )

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

    handles, labels = ax.get_legend_handles_labels()
    compartments = set(data.obs[compartment_col].unique())
    filtered = [
        (h, l)
        for h, l in zip(handles, labels)
        if (l in compartments or not any(c.isupper() for c in l))
    ]

    if filtered:
        handles, labels = zip(*filtered)
        ax.legend(handles=handles, labels=labels)
    return ax
