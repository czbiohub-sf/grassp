from __future__ import annotations
import time
import warnings

import pandas as pd
import requests

from anndata import AnnData
from tqdm import tqdm

from ..datasets.uniprot_cc import find_roots, uniprot_subcellular_vocabulary

BASE_URL = "https://rest.uniprot.org"

# Load vocabulary once at module level for efficiency
_VOCAB_CACHE = None


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
    protein_id_column: str | None = None,
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
        AnnData object with proteins in ``.obs`` (``proteins_as_obs=True``).
    protein_id_column
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
    >>> import grassp as gr
    >>> adata = gr.datasets.hein_2024()
    >>> gr.pp.annotate_uniprot_cc(adata)
    >>> adata.obs[['uniprot_cc_primary_coarse', 'uniprot_cc_primary_fine']].head()

    Filter for nuclear proteins:

    >>> nuclear = adata[adata.obs['uniprot_cc_primary_coarse'] == 'Nucleus']

    """

    # Extract protein IDs
    if protein_id_column is None:
        protein_ids = list(data.obs_names)
    else:
        if protein_id_column not in data.obs.columns:
            raise ValueError(f"Column '{protein_id_column}' not found in data.obs.columns")
        protein_ids = list(data.obs[protein_id_column])

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
