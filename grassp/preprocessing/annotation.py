from __future__ import annotations
import time
import warnings

from pathlib import Path

import numpy as np
import pandas as pd
import requests

from anndata import AnnData
from tqdm import tqdm

from ..datasets.uniprot_cc import find_roots, uniprot_subcellular_vocabulary

BASE_URL = "https://rest.uniprot.org"

# Load vocabulary once at module level for efficiency
_VOCAB_CACHE = None


# Comprehensive color dictionary for all known marker annotations
# Colors are organized by biological system/compartment for consistency
MARKER_COLORS = {
    # Core compartments - user provided
    # Example visually distinct colors for additional annotation needs (rename to new compartments)
    "Royal Blue": "#344B9E",
    "Cytosol": "#1B9E9E",
    "Cytosolic": "#1B9E9E",
    "Cytoplasm": "#1B9E9E",
    "Cytoplasmic": "#1B9E9E",
    "Nucleus": "#C9A27C",
    "Nuclear": "#C9A27C",
    "Nucleolus": "#6A3D9A",
    "Nucleolar": "#6A3D9A",
    "ER": "#7A4A2E",
    "Endoplasmic reticulum": "#7A4A2E",
    "Golgi": "#2AA1D8",
    "Golgi apparatus": "#2AA1D8",
    "TGN": "#2F2F2F",
    "Peroxisome": "#3F1861",
    "Lysosome": "#E41A1C",
    "Early endosome": "#00FF44",
    "Endosome": "#6DBE45",
    "Recycling endosome": "#BEFF5E",
    "Centrosome": "#1F78B4",
    "P-body": "#333333",
    # Stress granules
    "Stress granule": "#559529",
    "Stress granules": "#559529",
    "RNA granule": "#559529",
    "RNA granules": "#559529",
    "SG: "  # 559529",
    # 14-3-3
    "14-3-3": "#4D4D4D",
    "14-3-3 scaffold": "#4D4D4D",
    "14-3-3_scaffold": "#4D4D4D",
    # ERGIC
    "ERGIC": "#E0B400",
    "ER-Golgi intermediate compartment": "#E0B400",
    "Endoplasmic reticulum-Golgi intermediate compartment": "#E0B400",
    # Mitochondrial subcompartments (red-orange-brown gradient)
    "Mitochondrion": "#F28E1C",
    "Mitochondria": "#F28E1C",
    "Mito": "#F28E1C",
    "Mitochondrion - IM": "#F28E1C",
    "Mitochondrion - OM": "#E67E1A",
    "Mitochondrion - Matrix": "#DA6E18",
    "Mitochondria - Matrix": "#DA6E18",
    "Mitochondria Matrix": "#DA6E18",
    "Mitochondrion - Matrix 1": "#DA6E18",
    "Mitochondrion - Matrix 2": "#C24E14",
    "Mitochondrion - Soluble": "#DA6E18",
    "Mitochondrion - Membranes": "#B63E12",
    "Mitochondrion Membranes": "#B63E12",
    "Mitochondrion Membrane": "#B63E12",
    "Mitochondria - Membranes": "#B63E12",
    "Mitochondria Membranes": "#B63E12",
    "Mitochondria Membrane": "#B63E12",
    # ER system (yellow-gold-brown shades)
    "ER Lumen": "#E0B400",
    "ER Membrane": "#D4A800",
    "Endoplasmic reticulum membrane": "#D4A800",
    "ER 1": "#C89C00",
    "ER 2": "#BC9000",
    "ER High Curvature": "#B08400",
    # Cilium
    "Cilium": "#00847F",
    # Lipid droplets
    "Lipid droplet": "#FFB300",
    "Lipid droplets": "#FFB300",
    # Nuclear subcompartments (purple-tan shades)
    "Nucleus - Chromatin": "#C9A27C",
    "Nucleus - Non-Chromatin": "#B5947C",
    "Nuclear Pore Complex": "#A1867C",
    "Nuclear pore complex": "#8D787C",
    "Nuclear pore": "#8D787C",
    "Nuclear membrane": "#A3784D",
    "Nucleus membrane": "#A3784D",
    "Nucleoplasm-1": "#6A3D9A",
    "Nucleoplasm-2": "#5E3588",
    # Ribosomal compartments (blue shades)
    "Ribosome": "#E7298A",
    "Translation": "#E7298A",
    "Ribosome/Complexes": "#DB1D7E",
    "37S Ribosome": "#CF1172",
    "40S Ribosome": "#C30566",
    "54S Ribosome": "#B7005A",
    "60S Ribosome": "#AB004E",
    "Ribonucleoproteins 1": "#9F0042",
    # Proteasome variants (pink-magenta shades)
    "Proteasome": "#9F36B7",
    "19S Proteasome": "#9F36B7",
    "20S Proteasome": "#8C289E",
    "Proteasome Regulatory Particle": "#8C289E",
    # Cytoskeleton (green-teal shades)
    "Cytoskeleton": "#00CFFF",
    "Actin Cytoskeleton": "#F4A6C1",
    "Actin binding proteins": "#F4A6C1",
    "Actin-binding protein": "#F4A6C1",
    "Tubulin Cytoskeleton": "#0F8E8E",
    # Endocytic/secretory system (mixed)
    "Secretory/Endocytic 1": "#8DD3C7",
    "Secretory/Endocytic 2": "#7DC3B7",
    "Secretory/Endocytic 3": "#6DB3A7",
    "Endomembrane Vesicles": "#FFFFB3",
    "trans-Golgi network": "#2F2F2F",
    "trans-Golgi": "#2F2F2F",
    # Plasma membrane variants (blue-gray shades)
    "Membrane": "#7A8DA6",
    "PM": "#7A8DA6",
    "Plasma membrane": "#7A8DA6",
    "Cell membrane": "#7A8DA6",
    "PM - Integral": "#7A8DA6",
    "PM - Peripheral 1": "#6A7D96",
    "PM - Peripheral 2": "#5A6D86",
    "Cell cortex": "#202b39",
    # Protein complexes (gray shades)
    "Protein Complex": "#BEBADA",
    "Large Protein Complex": "#AEAACA",
    # Toxoplasma-specific (distinct colors)
    "Apicoplast": "#FB8072",
    "Apical 1": "#FDB462",
    "Apical 2": "#FCA452",
    "Dense Granules": "#B3DE69",
    "Micronemes": "#FCCDE5",
    "Rhoptries 1": "#D9D9D9",
    "Rhoptries 2": "#C9C9C9",
    "IMC": "#BC80BD",
    # Trypanosoma-specific (distinct colors)
    "Flagellum 1": "#CCEBC5",
    "Intraflagellar Transport": "#BCDB95",
    "Microtubule Structures 1": "#80B1D3",
    "Microtubule Structures 2": "#70A1C3",
    "Acidocalcisomes": "#FFED6F",
    "Glycosomes": "#FDB462",
    "STR": "#8DA0CB",
    # Plant-specific (green shades)
    "Plastid": "#66C2A5",
    "Vacuole": "#56B295",
    "THY": "#46A285",
    # Additional markers
    "ECM": "#E5C494",
    "Envelope": "#D5B484",
    # unannotated
    "None": "#BEBEBE",
    "null": "#BEBEBE",
    "unannotated": "#BEBEBE",
    "unspecified": "#BEBEBE",
    "unassigned": "#BEBEBE",
    "unlabeled": "#BEBEBE",
    "unmarked": "#BEBEBE",
    "unidentified": "#BEBEBE",
    "unmeasurable": "#BEBEBE",
    "unknown": "#BEBEBE",
    "unclassified": "#BEBEBE",
    "ambiguous": "#BEBEBE",
    "not annotated": "#BEBEBE",
    "not-annotated": "#BEBEBE",
    "notassigned": "#BEBEBE",
    "not assigned": "#BEBEBE",
    "undetermined": "#BEBEBE",
    "not determined": "#BEBEBE",
    "undefined": "#BEBEBE",
    "novel": "#BEBEBE",
    "miscellaneous": "#BEBEBE",
    "other": "#BEBEBE",
    "NA": "#BEBEBE",
    "0": "#BEBEBE",
    "N/A": "#BEBEBE",
    "nan": "#BEBEBE",
}
# Case-insensitive lookup: lowercase key -> canonical hex color
_MARKER_COLORS_LOWER: dict[str, str] = {k.lower(): v for k, v in MARKER_COLORS.items()}


def _get_vocab():
    """Get cached vocabulary or load it."""
    global _VOCAB_CACHE
    if _VOCAB_CACHE is None:
        _VOCAB_CACHE = uniprot_subcellular_vocabulary()
    return _VOCAB_CACHE


def _submit_id_mapping(protein_ids: list[str]) -> str | None:
    """
    Submit a batch mapping job to UniProt.

    Parameters
    ----------
    protein_ids
        List of UniProt protein IDs.

    Returns
    -------
    str or None
        Job ID if successful, None if failed.
    """
    if not protein_ids:
        return None

    # Clean IDs (remove isoform suffixes)
    clean_ids = [str(pid).split("-")[0].strip() for pid in protein_ids]
    clean_ids = [pid for pid in clean_ids if pid]  # Remove empty strings

    if not clean_ids:
        return None

    url = f"{BASE_URL}/idmapping/run"
    try:
        response = requests.post(
            url,
            data={
                "from": "UniProtKB_AC-ID",
                "to": "UniProtKB",
                "ids": ",".join(clean_ids),
            },
        )
        response.raise_for_status()
        return response.json()["jobId"]
    except (requests.RequestException, ValueError, KeyError):
        return None


def _wait_for_job(job_id: str, pbar: tqdm | None = None) -> tuple[bool, list]:
    """
    Poll UniProt until job is finished.

    Parameters
    ----------
    job_id
        Job ID from submit_id_mapping.
    pbar
        Optional tqdm progress bar to update.

    Returns
    -------
    tuple
        (success: bool, failed_ids: list)
    """
    status_url = f"{BASE_URL}/idmapping/status/{job_id}"

    while True:
        try:
            response = requests.get(status_url)
            response.raise_for_status()
            status = response.json()

            if status.get("jobStatus") == "RUNNING":
                if pbar:
                    pbar.set_postfix_str("waiting for UniProt...")
                time.sleep(1)
                continue

            # Job finished
            failed_ids = status.get("failedIds", [])
            return True, failed_ids

        except (requests.RequestException, ValueError):
            return False, []


def _fetch_entries(job_id: str) -> dict[str, dict]:
    """
    Fetch full entry records with subcellular location data.

    Parameters
    ----------
    job_id
        Job ID from submit_id_mapping.

    Returns
    -------
    dict
        Dictionary mapping protein ID to UniProt JSON data.
    """
    # First, get the redirectURL from details endpoint
    details_url = f"{BASE_URL}/idmapping/details/{job_id}"
    try:
        details_response = requests.get(details_url)
        details_response.raise_for_status()
        details = details_response.json()
        redirect_url = details.get("redirectURL")
        if not redirect_url:
            return {}
    except (requests.RequestException, ValueError, KeyError):
        return {}

    # Use stream endpoint to get all results (not paginated)
    if "/stream/" not in redirect_url:
        redirect_url = redirect_url.replace("/results/", "/results/stream/")

    # Fetch from the stream endpoint with desired fields
    try:
        response = requests.get(
            redirect_url,
            params={
                "format": "json",
                "fields": "accession,cc_subcellular_location",
            },
        )
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError):
        return {}

    # Map results by accession
    # ID mapping response has nested structure: results[i]['to']['primaryAccession']
    results = {}
    if isinstance(data, dict) and "results" in data:
        for entry in data["results"]:
            to_entry = entry.get("to", {})
            # 'to' can be a string if mapping failed, skip those
            if not isinstance(to_entry, dict):
                continue
            accession = to_entry.get("primaryAccession")
            if accession:
                results[accession] = to_entry
    elif isinstance(data, list):
        for entry in data:
            to_entry = entry.get("to", {})
            # 'to' can be a string if mapping failed, skip those
            if not isinstance(to_entry, dict):
                continue
            accession = to_entry.get("primaryAccession")
            if accession:
                results[accession] = to_entry

    return results


def _parse_subcellular_locations(uniprot_data: dict) -> dict[str, list[str]]:
    """
    Extract subcellular locations from UniProt JSON response.

    Uses the UniProt controlled vocabulary to map location IDs to standardized
    terms and determine hierarchical relationships. The fine location is the
    specific term, and the coarse location is the root of the hierarchy
    (determined by following HP part-of relationships).

    Parameters
    ----------
    uniprot_data
        JSON response from UniProt API.

    Returns
    -------
    dict
        Dictionary with keys 'coarse' and 'fine', each containing a list of
        location terms.
    """
    locations = {"coarse": [], "fine": []}

    if not uniprot_data or "comments" not in uniprot_data:
        return locations

    # Load vocabulary
    vocab = _get_vocab()

    for comment in uniprot_data.get("comments", []):
        if comment.get("commentType") != "SUBCELLULAR LOCATION":
            continue
        # Skip isoform-specific entries
        if "molecule" in comment:
            continue

        for subloc in comment.get("subcellularLocations", []):
            location_obj = subloc.get("location", {})

            # Try to get the location ID (accession)
            location_id = location_obj.get("id")

            if location_id and location_id in vocab:
                # Use vocabulary to get standardized name
                fine_name = vocab[location_id]["ID"]

                # Find root location(s) using HP (part-of) relationships
                roots = find_roots(location_id, vocab, relationship="HP")

                # Use the first root's name as coarse location
                if roots:
                    coarse_name = vocab[roots[0]]["ID"]
                else:
                    # If no roots found, use the location itself
                    coarse_name = fine_name

                locations["fine"].append(fine_name)
                locations["coarse"].append(coarse_name)
            elif location_obj.get("value"):
                # Fallback to old method if ID not available or not in vocab
                location_value = location_obj["value"]
                parts = [p.strip() for p in location_value.split(",")]
                if parts:
                    locations["coarse"].append(parts[0])
                    locations["fine"].append(parts[-1])

    return locations


def annotate_uniprot_cc(
    data: AnnData,
    uniprot_id_column: str | None = None,
    include_multiloc: bool = True,
) -> None:
    """
    Annotate proteins with UniProt subcellular location annotations.

    Queries the UniProt REST API to retrieve subcellular localization (CC)
    data for each protein in the dataset. Uses the UniProt controlled
    vocabulary to map location IDs to standardized terms and determine
    hierarchical relationships. The fine location is the specific subcellular
    location term, while the coarse location is the root of the hierarchy
    (determined by following HP part-of relationships in the vocabulary).

    This function adds annotation columns to ``.obs`` and modifies the
    AnnData object in-place.

    Parameters
    ----------
    data
        AnnData object.
    uniprot_id_column
        Column in ``.obs`` containing UniProt IDs. If None, uses ``obs_names``.
    include_multiloc
        If True, include columns with all locations (comma-separated) for
        multi-localizing proteins. If False, only include primary (first)
        location columns.

    Returns
    -------
    None
        Modifies ``data.obs`` in-place by adding columns:

        * ``uniprot_cc_primary_coarse``: First location at top hierarchy level
        * ``uniprot_cc_primary_fine``: First location at most specific level
        * ``uniprot_cc_all_coarse``: All coarse locations, comma-separated
          (only if ``include_multiloc=True``)
        * ``uniprot_cc_all_fine``: All fine locations, comma-separated
          (only if ``include_multiloc=True``)

        Proteins with missing or failed queries will have NaN values.

    Examples
    --------
    >>> import grassp as gr  # doctest: +SKIP
    >>> adata = gr.datasets.hein_2024()  # doctest: +SKIP
    >>> gr.pp.annotate_uniprot_cc(adata)  # doctest: +SKIP
    >>> adata.obs[['uniprot_cc_primary_coarse', 'uniprot_cc_primary_fine']].head()  # doctest: +SKIP

    Filter for nuclear proteins:

    >>> nuclear = adata[adata.obs['uniprot_cc_primary_coarse'] == 'Nucleus']  # doctest: +SKIP

    """

    # Extract protein IDs
    if uniprot_id_column is None:
        protein_ids = list(data.obs_names)
    else:
        if uniprot_id_column not in data.obs.columns:
            raise ValueError(f"Column '{uniprot_id_column}' not found in data.obs.columns")
        protein_ids = list(data.obs[uniprot_id_column])

    n_proteins = len(protein_ids)

    # Submit ID mapping job
    with tqdm(total=3, desc="Querying UniProt", unit="step") as pbar:
        pbar.set_postfix_str("submitting job...")
        job_id = _submit_id_mapping(protein_ids)

        if job_id is None:
            raise RuntimeError("Failed to submit UniProt ID mapping job")

        pbar.update(1)

        # Wait for job to complete
        pbar.set_postfix_str("waiting for job...")
        success, failed_ids = _wait_for_job(job_id, pbar)

        if not success:
            raise RuntimeError("UniProt ID mapping job failed")

        pbar.update(1)

        # Fetch results
        pbar.set_postfix_str("fetching results...")
        uniprot_results = _fetch_entries(job_id)
        pbar.update(1)

    # Process results for each protein
    all_results = {}
    for protein_id in protein_ids:
        clean_id = str(protein_id).split("-")[0].strip()

        if clean_id in uniprot_results:
            uniprot_data = uniprot_results[clean_id]
            locations = _parse_subcellular_locations(uniprot_data)
            all_results[protein_id] = locations
        else:
            # Protein not found or failed
            all_results[protein_id] = {"coarse": [], "fine": []}

    # Build result columns
    primary_coarse = []
    primary_fine = []
    all_coarse = []
    all_fine = []

    for protein_id in protein_ids:
        locations = all_results.get(protein_id, {"coarse": [], "fine": []})

        # Primary locations
        if locations["coarse"]:
            primary_coarse.append(locations["coarse"][0])
            primary_fine.append(locations["fine"][0])
        else:
            primary_coarse.append(None)
            primary_fine.append(None)

        # All locations
        if locations["coarse"]:
            unique_coarse = list(dict.fromkeys(locations["coarse"]))
            unique_fine = list(dict.fromkeys(locations["fine"]))
            all_coarse.append(", ".join(unique_coarse))
            all_fine.append(", ".join(unique_fine))
        else:
            all_coarse.append(None)
            all_fine.append(None)

    # Add columns to data.obs
    data.obs["uniprot_cc_primary_coarse"] = pd.Series(
        primary_coarse, index=data.obs_names, dtype="category"
    )
    data.obs["uniprot_cc_primary_fine"] = pd.Series(
        primary_fine, index=data.obs_names, dtype="category"
    )

    if include_multiloc:
        data.obs["uniprot_cc_all_coarse"] = pd.Series(
            all_coarse, index=data.obs_names, dtype="string"
        )
        data.obs["uniprot_cc_all_fine"] = pd.Series(
            all_fine, index=data.obs_names, dtype="string"
        )

    n_failed = sum(1 for locs in all_results.values() if not locs["coarse"])
    warnings.warn(
        f"Annotation complete: {n_proteins} proteins queried, "
        f"{n_failed} not found ({n_failed / n_proteins * 100:.1f}%)"
    )


def add_external_validation_markers(
    data: AnnData,
    species: str,
    columns: list[str] | str | None = None,
    uniprot_id_column: str | None = None,
    ignore_isoform_extensions: bool = False,
    use_gene_names: bool = False,
    gene_names_column: str | None = None,
) -> None:
    """Annotate proteins with external validation markers (MitoCarta, MitoCop, topology, etc.).

    Matches protein IDs in ``.obs`` against curated external marker datasets
    that include protein topology annotations, signal peptides, transmembrane
    domains, and mitochondrial annotations. These markers are useful for validating
    subcellular localization predictions.

    Available columns include:

    * ``Topological domain``: UniProt topology annotations
    * ``Transmembrane``: Transmembrane domain annotations (e.g., "Helical")
    * ``Intramembrane``: Intramembrane domain annotations
    * ``has_signal``: Boolean indicating presence of signal peptide
    * ``has_transmem``: Boolean indicating presence of transmembrane domain
    * ``has_intramem``: Boolean indicating presence of intramembrane domain
    * ``has_topo_dom``: Boolean indicating presence of topological domain
    * ``mitocarta``: MitoCarta annotation (human/mouse only)
    * ``mitocarta_evidence``: MitoCarta evidence code (human/mouse only)
    * ``mitocarta_subloc``: MitoCarta subcellular location (human/mouse only)
    * ``mitocop``: MitoCop annotation (human only)

    This function modifies the AnnData object in-place by adding annotation
    columns to ``.obs``.

    Parameters
    ----------
    data
        AnnData object.
    species
        Species code to determine which marker file to read. Examples: 'hsap'
        (human), 'mmus' (mouse), 'scer' (yeast), 'atha' (Arabidopsis), 'dmel'
        (fly).
    columns
        Specific column(s) to include from the marker file. If None, includes
        all available columns. Can be a single column name (string) or a list
        of column names.
    uniprot_id_column
        Column in ``.obs`` containing UniProt IDs. If None, uses ``.obs_names``.
    ignore_isoform_extensions
        If True, strip UniProt isoform extensions (e.g., ``-2`` from ``Q53H12-2``)
        before merging. The original column values are retained after the merge.
        Default: False.
    use_gene_names
        If True, merge markers using gene names instead of UniProt IDs.
        When True, the marker DataFrame will be deduplicated by gene name
        before merging (keeping row with fewest NaN values). Default: False.
    gene_names_column
        Column in ``.obs`` containing gene names/symbols. Only used when
        ``use_gene_names=True``. If None, uses ``.obs_names``. Default: None.

    Returns
    -------
    None
        Modifies ``data.obs`` in-place by adding marker annotation columns
        (converted to categorical dtype for string columns).

    Examples
    --------
    >>> import grassp as gr
    >>> import pandas as pd
    >>> adata = gr.datasets.hein_2024(enrichment='raw')
    >>> # Add all external validation markers
    >>> gr.pp.add_external_validation_markers(adata, species='hsap')  # doctest: +ELLIPSIS
    Added Topological domain annotations for ...
    >>> # Add specific columns only
    >>> gr.pp.add_external_validation_markers(
    ...     adata, species='hsap',
    ...     columns=['mitocarta', 'has_transmem']
    ... )  # doctest: +ELLIPSIS
    Added mitocarta annotations for ...

    """
    # Construct file path
    module_path = Path(__file__).parent.parent
    marker_file = module_path / "datasets" / "external" / f"external_markers_{species}.tsv"

    # Get available species
    marker_dir = module_path / "datasets" / "external"
    available_species = [
        f.stem.replace("external_markers_", "")
        for f in marker_dir.glob("external_markers_*.tsv")
        if f.is_file()
    ]

    if not marker_file.exists():
        raise FileNotFoundError(
            f"Species not found: {species}. "
            f"Please use one of the available species: {available_species}"
        )

    # Read marker file
    markers_df = pd.read_csv(marker_file, sep="\t", dtype=str, index_col="id")

    # Validate use_gene_names
    if use_gene_names:
        if 'gene_name' not in markers_df.columns:
            raise ValueError(
                f"Cannot use gene names: 'gene_name' column not found in marker file "
                f"for species '{species}'. Available columns: {list(markers_df.columns)}"
            )

        if gene_names_column is not None and gene_names_column not in data.obs.columns:
            raise ValueError(f"Column '{gene_names_column}' not found in data.obs.columns")
    else:
        markers_df = markers_df.drop(columns=['gene_name'])

    # Get all available columns (everything except id, which is the index)
    all_columns = [col for col in markers_df.columns]

    # Determine which columns to include
    if columns is None:
        # Include all columns
        columns_to_include = all_columns
    else:
        # Convert single string to list
        if isinstance(columns, str):
            columns = [columns]

        # Check which columns are available
        available_columns = set(all_columns)
        requested_columns = set(columns)

        found_columns = requested_columns & available_columns
        missing_columns = requested_columns - available_columns

        if not found_columns:
            raise ValueError(
                f"None of the specified columns {list(requested_columns)} "
                f"were found in the marker file. "
                f"Available columns: {list(available_columns)}"
            )

        if missing_columns:
            warnings.warn(
                f"Some columns not found in marker file: {list(missing_columns)}. "
                f"Available columns: {list(available_columns)}. "
                f"Proceeding with: {list(found_columns)}"
            )

        columns_to_include = list(found_columns)

    # Deduplicate by gene name if requested
    if use_gene_names:
        original_count = len(markers_df)

        # Reset index to make 'id' a regular column
        markers_df = markers_df.reset_index()

        # For each gene_name, select the row with fewest NaN values
        def select_best_row(group):
            # Count NaN values per row
            nan_counts = group.isna().sum(axis=1)
            return group.loc[nan_counts.idxmin()]

        # Group by gene_name and select best row for each group
        markers_df = markers_df.groupby('gene_name', dropna=False, as_index=False).apply(
            select_best_row, include_groups=False
        )

        # Set gene_name as index
        markers_df = markers_df.set_index('gene_name')

        duplicate_count = original_count - len(markers_df)
        if duplicate_count > 0:
            print(f"Deduplicated {duplicate_count} proteins by gene name")

        # Remove gene_name from columns_to_include since it's now the index
        if 'gene_name' in columns_to_include:
            columns_to_include.remove('gene_name')

    # Raise if any of the new columns already exist
    for col in columns_to_include:
        if col in data.obs.columns:
            raise ValueError(
                f"Column '{col}' already exists in data.obs.columns. "
                f"Please remove/rename the column before adding markers."
            )

    # Handle isoform stripping (only for UniProt ID mode)
    temp_col_created = False
    if ignore_isoform_extensions and not use_gene_names:
        temp_col = '_temp_id_for_merge'
        if uniprot_id_column is None:
            data.obs[temp_col] = data.obs.index.str.replace(r'-\d+$', '', regex=True)
        else:
            data.obs[temp_col] = data.obs[uniprot_id_column].str.replace(
                r'-\d+$', '', regex=True
            )
        uniprot_id_column = temp_col
        temp_col_created = True

    # Perform merge based on mode
    if use_gene_names:
        # Gene name-based merge
        if gene_names_column is None:
            data.obs = data.obs.join(markers_df.loc[:, columns_to_include])
        else:
            data.obs = data.obs.merge(
                markers_df.loc[:, columns_to_include],
                left_on=gene_names_column,
                right_index=True,
                how="left",
                sort=False,
            )
    else:
        # UniProt ID-based merge (existing)
        if uniprot_id_column is None:
            data.obs = data.obs.join(markers_df.loc[:, columns_to_include])
        else:
            if uniprot_id_column not in data.obs.columns:
                raise ValueError(f"Column '{uniprot_id_column}' not found in data.obs.columns")
            data.obs = data.obs.merge(
                markers_df.loc[:, columns_to_include],
                left_on=uniprot_id_column,
                right_index=True,
                how="left",
                sort=False,
            )

    # Clean up temporary column if created
    if temp_col_created:
        data.obs = data.obs.drop(columns=[temp_col])

    # Convert string columns to categorical
    for col in columns_to_include:
        # Convert non-boolean string columns to categorical
        # Boolean columns (has_signal, has_transmem, etc.) should stay as strings
        if not col.startswith("has_"):
            data.obs[col] = pd.Categorical(data.obs[col])

    # Report statistics
    for col in columns_to_include:
        n_proteins = len(data.obs)
        n_annotated = sum(~data.obs[col].isna())
        print(
            f"Added {col} annotations for {n_annotated}/{n_proteins} proteins "
            f"({n_annotated / n_proteins * 100:.1f}%)"
        )


def add_markers(
    data: AnnData,
    species: str,
    authors: list[str] | str | None = None,
    uniprot_id_column: str | None = None,
    add_colors: bool = True,
) -> None:
    """Annotate proteins with marker annotations from literature.

    Matches protein IDs in ``.obs`` against a collection of marker annotations
    from different authors. Note that marker IDs are species-specific and may
    not be UniProt accessions (see table below).

    Marker annotations are sourced from:

    .. list-table::
        :header-rows: 1

        * - authors
          - source
        * - hein2024_gt_component
          - Marker list used in Hein et al. 2024, Cell, https://doi.org/10.1016/j.cell.2024.11.028
        * - hein2024_component
          - Full annotations from Hein et al. 2024, Cell, https://doi.org/10.1016/j.cell.2024.11.028
        * - lilley, christopher, geladaki, itzhak, villaneuva, christoforou
          - Obtained from pRoloc. See: https://bioconductor.org/packages/pRoloc/
            and https://lgatto.github.io/pRoloc/reference/pRolocmarkers.html

    **Protein ID types by species:**

    .. list-table::
        :header-rows: 1

        * - Species Code
          - Common Name
          - ID Type
          - Example ID
        * - atha
          - *Arabidopsis thaliana*
          - TAIR/Araport
          - AT1G01620
        * - dmel
          - *Drosophila melanogaster*
          - UniProt
          - A1Z6P3
        * - ggal
          - *Gallus gallus* (Chicken)
          - IPI
          - IPI00570752.1
        * - hsap
          - *Homo sapiens* (Human)
          - UniProt
          - A0AVT1
        * - mmus
          - *Mus musculus* (Mouse)
          - UniProt
          - A2AJ15
        * - scer
          - *Saccharomyces cerevisiae* (Yeast)
          - UniProt
          - D6VTK4
        * - toxo
          - *Toxoplasma gondii*
          - ToxoDB Gene IDs
          - TGME49_200250
        * - tryp
          - *Trypanosoma brucei*
          - TriTrypDB Gene IDs
          - Tb11.v5.0162

    This function modifies the AnnData object in-place by adding marker
    annotation columns to ``.obs``.

    Parameters
    ----------
    data
        AnnData object.
    species
        Species code to determine which marker file to read. Examples: 'hsap'
        (human), 'mmus' (mouse), 'scer' (yeast), 'atha' (Arabidopsis), 'dmel'
        (fly), 'toxo' (Toxoplasma), 'tryp' (Trypanosoma), 'ggal' (chicken).
    authors
        Specific author column(s) to include from the marker file. If None,
        includes all available author columns. Can be a single author name
        (string) or a list of author names.
    uniprot_id_column
        Column in ``.obs`` containing protein IDs (see the specific ID needed in the description above). If None, uses ``.obs_names``.
    add_colors
        If True, automatically add color mappings to ``.uns`` for each marker
        column, following scanpy plotting conventions. Colors are stored as
        ``'{author}_colors'`` lists matching categorical order.

    Returns
    -------
    None
        Modifies ``data.obs`` in-place by adding marker annotation columns
        (converted to categorical dtype). If ``add_colors=True``, also adds
        color mappings to ``data.uns`` as ``'{author}_colors'`` lists.

    Examples
    --------
    >>> import grassp as gr
    >>> import pandas as pd
    >>> adata = gr.datasets.hein_2024(enrichment='raw')
    >>> # Add specific author annotations
    >>> gr.pp.add_markers(adata, species='hsap', authors=['christopher'])  # doctest: +ELLIPSIS
    Added christopher annotations for ...
    >>> # Check categorical dtype and colors
    >>> isinstance(adata.obs['christopher'].dtype, pd.CategoricalDtype)
    True
    >>> 'christopher_colors' in adata.uns
    True
    >>> # Disable automatic color mapping
    >>> gr.pp.add_markers(adata, species='hsap', authors=['lilley'], add_colors=False)  # doctest: +ELLIPSIS
    Added lilley annotations for ...
    """
    # Construct file path
    module_path = Path(__file__).parent.parent
    marker_file = module_path / "datasets" / "external" / f"{species}_markers.tsv"
    # Available species are determined by the marker files present in datasets/external
    marker_dir = module_path / "datasets" / "external"
    available_species = [
        f.stem.replace("_markers", "") for f in marker_dir.glob("*_markers.tsv") if f.is_file()
    ]

    if not marker_file.exists():
        raise FileNotFoundError(
            f"Species not found: {species}. "
            f"Please use one of the available species: {available_species}"
        )

    # Read marker file
    markers_df = pd.read_csv(marker_file, sep="\t", dtype=str, index_col="id")

    # Get all author columns (everything except id, which is the index)
    all_author_columns = [col for col in markers_df.columns]

    # Determine which columns to include
    if authors is None:
        # Include all author columns
        columns_to_include = all_author_columns
    else:
        # Convert single string to list
        if isinstance(authors, str):
            authors = [authors]

        # Check which authors are available
        available_authors = set(all_author_columns)
        requested_authors = set(authors)

        found_authors = requested_authors & available_authors
        missing_authors = requested_authors - available_authors

        if not found_authors:
            raise ValueError(
                f"None of the specified authors {list(requested_authors)} "
                f"were found in the marker file. "
                f"Available authors: {list(available_authors)}"
            )

        if missing_authors:
            warnings.warn(
                f"Some authors not found in marker file: {list(missing_authors)}. "
                f"Available authors: {list(available_authors)}. "
                f"Proceeding with: {list(found_authors)}"
            )

        columns_to_include = list(found_authors)

    # Raise if any of the new columns already exist
    for col in columns_to_include:
        if col in data.obs.columns:
            raise ValueError(
                f"Column '{col}' already exists in data.obs.columns. Please remove/rename the column before adding markers."
            )

    # Extract protein IDs from AnnData
    if uniprot_id_column is None:
        data.obs = data.obs.join(markers_df.loc[:, columns_to_include])
    else:
        if uniprot_id_column not in data.obs.columns:
            raise ValueError(f"Column '{uniprot_id_column}' not found in data.obs.columns")
        data.obs = data.obs.merge(
            markers_df.loc[:, columns_to_include],
            left_on=uniprot_id_column,
            right_index=True,
            how="left",
            sort=False,
        )

    # Convert to categorical and optionally add color mappings
    for col in columns_to_include:
        # Convert to categorical for scanpy compatibility
        data.obs[col] = pd.Categorical(data.obs[col])

        # Add color mappings to .uns for scanpy plotting
        if add_colors:
            # Get unique categories (excluding NaN)
            categories = data.obs[col].cat.categories

            # Create color list matching categorical order (case-insensitive)
            colors = []
            for category in categories:
                color = _MARKER_COLORS_LOWER.get(str(category).lower())
                if color is not None:
                    colors.append(color)
                elif category in [None, np.nan]:
                    continue
                else:
                    # Fallback: use gray for unmapped categories
                    colors.append('#808080')

            # Store in .uns following scanpy convention
            data.uns[f'{col}_colors'] = colors

    # Report statistics
    for author in columns_to_include:
        n_proteins = len(data.obs)
        n_annotated = sum(~data.obs[author].isna())
        print(
            f"Added {author} annotations for {n_annotated}/{n_proteins} proteins "
            f"({n_annotated / n_proteins * 100:.1f}%)"
        )


def set_sensible_compartment_colors(
    data: AnnData,
    cutoff: float = 0.3,
    verbose: bool = False,
    plot_mapping: bool = False,
) -> None:
    """Assign colors to compartment annotation columns in ``.obs`` using the built-in color dictionary.

    Scans all columns in ``.obs`` and identifies those that look like compartment
    annotation columns by checking what fraction of their unique values appear in the
    built-in :data:`MARKER_COLORS` dictionary. Matching is case-insensitive and treats
    underscores as spaces (e.g. ``"endoplasmic_reticulum"`` matches ``"Endoplasmic Reticulum"``).

    Columns where at least ``cutoff`` fraction of unique values are recognised compartments
    receive a ``'{col}_colors'`` entry in ``.uns``, following scanpy plotting conventions.
    Known compartments get their canonical color from :data:`MARKER_COLORS`; unknown
    categories receive distinct colors drawn from matplotlib's ``tab20`` / ``tab20b`` /
    ``tab20c`` palettes. All assigned colors are deduplicated so that every category in a
    column maps to a unique color.

    Parameters
    ----------
    data
        AnnData object whose ``.obs`` columns are scanned for compartment annotations.
    cutoff
        Minimum fraction of unique values in a column that must be recognised compartments
        for that column to receive a ``'{col}_colors'`` entry in ``.uns``. Default is ``0.3``.
    verbose
        If ``True``, print diagnostic information for each processed column, including the
        matched fraction, assigned colors, and any fallback assignments.
    plot_mapping
        If ``True``, display a grid plot showing the color assigned to each category for
        every processed column, with fallback colors visually distinguished.
    """
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    # Build a fallback palette from matplotlib's tab20 family (60 distinct colors)
    _fallback_palette = [
        mcolors.to_hex(c)
        for palette in (plt.cm.tab20.colors, plt.cm.tab20b.colors, plt.cm.tab20c.colors)
        for c in palette
    ]

    plotted_color_maps: dict[str, dict[str, str]] = {}
    plotted_fallback_masks: dict[str, set[str]] = {}

    for col in data.obs.columns:
        series = data.obs[col]

        # Collect unique non-null string-like values
        unique_vals = [v for v in series.dropna().unique() if v is not None]
        if not unique_vals:
            continue

        def _normalize(v: str) -> str:
            return str(v).lower().replace('_', ' ')

        n_matching = sum(1 for v in unique_vals if _normalize(v) in _MARKER_COLORS_LOWER)
        if n_matching / len(unique_vals) < cutoff:
            continue

        # Respect categorical order if present, otherwise use sorted unique values
        if hasattr(series, 'cat'):
            categories = list(series.cat.categories)
        else:
            categories = sorted(str(v) for v in unique_vals)

        # Assign fallback colors to categories not in MARKER_COLORS (case-insensitive)
        unknown = [c for c in categories if _normalize(c) not in _MARKER_COLORS_LOWER]
        fallback = {
            cat: _fallback_palette[i % len(_fallback_palette)] for i, cat in enumerate(unknown)
        }

        colors = [
            _MARKER_COLORS_LOWER.get(_normalize(cat), fallback.get(cat, '#808080'))
            for cat in categories
        ]

        # Replace scanpy's reserved NaN color (#D3D3D3) wherever it appears
        colors = ['#BEBEBE' if c.upper() == '#D3D3D3' else c for c in colors]

        # Deduplicate: each category must have a unique color
        used: set[str] = set()
        deduped: list[str] = []
        for color in colors:
            if color not in used:
                used.add(color)
                deduped.append(color)
            else:
                replacement = next((c for c in _fallback_palette if c not in used), color)
                used.add(replacement)
                deduped.append(replacement)
        if verbose:
            print(f"Added {col}_colors for {len(deduped)} categories")
            print(f"Categories: {deduped}")
            print(f"Colors: {colors}")
            print(f"Used: {used}")
            print(f"Fallback: {fallback}")
            print(f"Categories: {categories}")
            print(f"Unique values: {unique_vals}")
            print(f"Matching: {n_matching}")
            print(f"Fraction: {n_matching / len(unique_vals)}")
        if plot_mapping:
            plotted_color_maps[col] = dict(zip(categories, deduped))
            plotted_fallback_masks[col] = {
                str(category).lower()
                for category in categories
                if _normalize(category) not in _MARKER_COLORS_LOWER
            }

        data.uns[f'{col}_colors'] = deduped

    if plot_mapping and plotted_color_maps:
        row_labels = list(plotted_color_maps.keys())
        all_compartments = {
            compartment
            for colormap in plotted_color_maps.values()
            for compartment in colormap.keys()
        }
        import re

        def _natsort_key(s: str) -> list:
            parts = re.split(r'(\d+)', str(s).lower())
            return [int(p) if p.isdigit() else p for p in parts]

        col_labels = sorted(all_compartments, key=_natsort_key)

        nrows = len(row_labels)
        ncols = len(col_labels)
        fig, ax = plt.subplots(figsize=(1.0 + 0.65 * ncols, 1.0 + 0.55 * nrows))

        for i, row in enumerate(row_labels):
            row_colormap = plotted_color_maps[row]
            row_fallbacks = plotted_fallback_masks[row]
            for j, compartment in enumerate(col_labels):
                row_y = nrows - i - 1
                has_compartment = compartment in row_colormap
                color = row_colormap.get(compartment, '#F2F2F2')
                ax.add_patch(plt.Rectangle((j, row_y), 1, 1, color=color, ec='k', lw=0.7))

                if has_compartment and str(compartment).lower() in row_fallbacks:
                    ax.plot([j, j + 1], [row_y, row_y + 1], color='k', lw=1.0)

        ax.set_yticks([nrows - i - 0.5 for i in range(nrows)])
        ax.set_yticklabels(row_labels, fontsize=7)
        ax.set_xticks([j + 0.5 for j in range(ncols)])
        ax.set_xticklabels(col_labels, fontsize=6, rotation=90, ha='center', va='top')
        ax.set_xlim(0, ncols)
        ax.set_ylim(0, nrows)
        ax.set_aspect('equal')
        ax.set_title('Color mapping across annotation columns', fontsize=9, pad=8)
        ax.set_xlabel('Compartments', fontsize=8, labelpad=8)
        ax.set_ylabel('Annotation columns', fontsize=8, labelpad=6)
        ax.tick_params(axis='both', which='both', length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        plt.tight_layout()
        plt.show()
