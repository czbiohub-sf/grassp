"""Contaminant protein removal functionality."""

from __future__ import annotations
import warnings

from pathlib import Path
from typing import TYPE_CHECKING, List, Literal

import pandas as pd

if TYPE_CHECKING:
    from anndata import AnnData


def remove_contaminants(
    data: AnnData,
    filter_columns: List[str] | None = None,
    filter_value: str | None = None,
    inplace: bool = True,
) -> AnnData | None:
    """Remove contaminant proteins from the data matrix.

    Parameters
    ----------
    data
        The annotated data matrix with proteins as observations (rows).
    filter_columns
        Column names in data.obs to use for filtering contaminants. If None, uses
        columns specified in data.uns['RawInfo']['filter_columns'].
    filter_value
        If provided, first convert filter columns to boolean by comparing to this value.
        If None, assumes filter columns are already boolean.
    inplace
        Whether to modify data in place or return a copy.

    Returns
    -------

    * If `inplace=False`, returns filtered data.
    * If `inplace=True`, returns None.

    Examples
    --------
    >>> import grassp as gr
    >>> adata = gr.datasets.hein_2024(enrichment="raw")
    >>> adata.shape
    (8538, 183)
    >>> gr.pp.remove_contaminants(
    ...     adata,
    ...     filter_columns=['Potential contaminant'],
    ...     filter_value='+'
    ... )
    >>> adata.shape  # Contaminants removed
    (8538, 183)
    """

    if filter_columns is None:
        filter_columns = data.uns["RawInfo"]["filter_columns"]
    elif isinstance(filter_columns, str):
        filter_columns = [filter_columns]

    if filter_value is not None:
        data.obs[filter_columns] = data.obs[filter_columns].eq(filter_value)
    is_contaminant = data.obs[filter_columns].any(axis=1)
    if not inplace:
        return data.copy()[~is_contaminant, :]
    data._inplace_subset_obs(data.obs.index[~is_contaminant])


def remove_cRAP_proteins(
    data: AnnData,
    id_column: str | None = None,
    id_type: Literal["uniprot", "uniprot_entry_name"] = "uniprot",
    inplace: bool = True,
    verbose: bool = True,
) -> AnnData | None:
    """Remove cRAP (common Repository of Adventitious Proteins) contaminants.

    This function removes common laboratory contaminants from proteomics datasets
    using the cRAP database maintained at https://ftp.thegpm.org/fasta/crap/.
    Protein IDs are matched against the cRAP database, with support for both
    UniProt accession IDs (e.g., P00330) and entry names (e.g., ADH1_YEAST).

    Parameters
    ----------
    data
        The annotated data matrix with proteins as observations (rows).
    id_column
        Column name in data.obs containing protein IDs to match against cRAP database.
        If None, uses data.obs_names (row index).
    id_type
        Type of protein identifier to match:
        - 'uniprot': UniProt accession IDs (e.g., P00330)
        - 'uniprot_entry_name': UniProt entry names (e.g., ADH1_YEAST)
    inplace
        Whether to modify data in place or return a copy.
    verbose
        If True, print the list of removed protein IDs. Default is True.

    Returns
    -------
    * If `inplace=False`, returns filtered data with cRAP proteins removed.
    * If `inplace=True`, modifies data in place and returns None.

    Notes
    -----
    - Protein IDs with isoform suffixes (e.g., P00330-1) are automatically cleaned
      to base accession (P00330) before matching.
    - If no cRAP proteins are found in the dataset, a warning is issued but the
      function completes successfully.
    - The cRAP database is included with grassp. To update it, run:
      `python -m grassp.datasets.marker_curation.update_cRAP`

    See Also
    --------
    remove_contaminants : Remove contaminants based on custom filter columns.

    Examples
    --------
    Remove cRAP proteins using UniProt IDs from row index:

    >>> import grassp as gr
    >>> adata = gr.datasets.hein_2024(enrichment="raw")
    >>> adata.shape
    (8538, 183)
    >>> gr.pp.remove_cRAP_proteins(adata)
    >>> adata.shape  # Some cRAP proteins removed
    (8520, 183)
    """
    # Load cRAP database
    module_path = Path(__file__).parent.parent
    crap_file = module_path / "datasets" / "external" / "cRAP.tsv"

    if not crap_file.exists():
        raise FileNotFoundError(
            f"cRAP database not found at {crap_file}. "
            "Please run: python -m grassp.datasets.marker_curation.update_cRAP"
        )

    crap_df = pd.read_csv(crap_file, sep="\t")

    # Extract protein IDs from AnnData
    if id_column is None:
        protein_ids = data.obs_names.tolist()
    else:
        if id_column not in data.obs.columns:
            available = ', '.join(data.obs.columns.tolist()[:10])
            raise ValueError(
                f"Column '{id_column}' not found in data.obs. "
                f"Available columns include: {available}..."
            )
        protein_ids = data.obs[id_column].tolist()

    # Clean IDs by removing isoform suffixes (e.g., P00330-1 -> P00330)
    cleaned_ids = []
    for pid in protein_ids:
        if pd.isna(pid):
            cleaned_ids.append(None)
        else:
            pid_str = str(pid)
            # Remove isoform suffix (everything after '-')
            if '-' in pid_str:
                pid_str = pid_str.split('-')[0]
            cleaned_ids.append(pid_str)

    # Match based on id_type
    if id_type == "uniprot":
        crap_ids = set(crap_df['id'].tolist())
    elif id_type == "uniprot_entry_name":
        crap_ids = set(crap_df['entry_name'].tolist())
    else:
        raise ValueError(
            f"Invalid id_type '{id_type}'. Must be 'uniprot' or 'uniprot_entry_name'."
        )

    # Create boolean mask of cRAP proteins
    is_crap = [cid in crap_ids if cid is not None else False for cid in cleaned_ids]
    n_crap = sum(is_crap)

    # Handle edge cases
    if n_crap == 0:
        warnings.warn(
            "No cRAP proteins found in dataset. Check that id_column and id_type "
            "are correct for your data.",
            UserWarning,
        )
        if not inplace:
            return data.copy()
        return None

    if n_crap == len(is_crap):
        warnings.warn(
            "All proteins matched cRAP database! This is likely an error. "
            "No proteins will be removed.",
            UserWarning,
        )
        if not inplace:
            return data.copy()
        return None

    # Filter proteins
    if verbose:
        # Collect removed protein IDs (use obs_names which are the actual indices)
        removed_ids = [data.obs_names[i] for i, is_c in enumerate(is_crap) if is_c]
        print(f"Removing {n_crap} cRAP proteins from dataset:")
        for rid in removed_ids:
            print(f"  - {rid}")
    else:
        print(f"Removing {n_crap} cRAP proteins from dataset")

    keep_mask = [not x for x in is_crap]
    if not inplace:
        return data[keep_mask, :].copy()

    data._inplace_subset_obs(data.obs.index[keep_mask])
    return None
