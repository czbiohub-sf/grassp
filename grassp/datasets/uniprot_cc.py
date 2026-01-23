from __future__ import annotations
import gzip
import pickle
import warnings

from pathlib import Path
from typing import Any

import requests

BASE_URL = 'https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/docs/subcell.txt'


def _download_subcell_vocabulary(url: str) -> str:
    """
    Download subcellular location vocabulary from UniProt.

    Parameters
    ----------
    url
        URL to subcell.txt file.

    Returns
    -------
    str
        File content as string.
    """
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        raise RuntimeError(f'Failed to download subcellular vocabulary: {e}')


def _parse_subcell_vocabulary(content: str) -> dict[str, dict[str, Any]]:
    """
    Parse UniProt subcellular location vocabulary file.

    Parameters
    ----------
    content
        Raw content from subcell.txt file.

    Returns
    -------
    dict
        Dictionary mapping accession IDs to location metadata.
    """
    vocab = {}

    # Split into entries (delimited by //)
    entries = content.split('\n//\n')

    for entry_text in entries:
        if not entry_text.strip():
            continue

        entry = {'HI': [], 'HP': []}
        accession = None
        de_lines = []

        for line in entry_text.split('\n'):
            if not line.strip():
                continue

            # Extract field code (first 2 characters)
            if len(line) < 5:
                continue

            field_code = line[:2]
            value = line[5:].strip() if len(line) > 5 else ''

            # Remove trailing period
            if value.endswith('.'):
                value = value[:-1]

            if field_code == 'AC':
                accession = value
            elif field_code == 'ID':
                entry['ID'] = value
            elif field_code == 'DE':
                de_lines.append(value)
            elif field_code == 'HI':
                entry['HI'].append(value)
            elif field_code == 'HP':
                entry['HP'].append(value)
            elif field_code == 'SY':
                # Parse synonyms (semicolon-delimited)
                entry['SY'] = [s.strip() for s in value.split(';') if s.strip()]
            elif field_code == 'GO':
                # Parse GO terms (format: GO:XXXXXXX; term name)
                if 'GO' not in entry:
                    entry['GO'] = []
                go_id = value.split(';')[0].strip() if ';' in value else value.strip()
                entry['GO'].append(go_id)
            elif field_code == 'SL':
                entry['SL'] = value

        # Join multi-line definitions
        if de_lines:
            entry['DE'] = ' '.join(de_lines)

        # Add to vocabulary if we have an accession
        if accession and 'ID' in entry:
            vocab[accession] = entry

    return vocab


def _get_cache_path(cache_dir: Path | None = None) -> Path:
    """
    Get path to cached vocabulary file.

    Parameters
    ----------
    cache_dir
        Optional cache directory. If None, uses package data directory
        or scanpy settings.

    Returns
    -------    Path
        Path to cached vocabulary file.
    """
    filename = 'uniprot_subcell_vocab.pkl.gz'

    # Priority 1: Check package external/ directory
    package_path = Path(__file__).parent / 'external' / filename
    if package_path.exists():
        return package_path

    # Priority 2: Use provided cache_dir
    if cache_dir is not None:
        cache_path = Path(cache_dir) / filename
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        return cache_path

    # Priority 3: Use scanpy settings datadir
    try:
        import scanpy

        cache_path = Path(scanpy.settings.datasetdir) / filename
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        return cache_path
    except (ImportError, AttributeError):
        # Fallback to package external/ directory
        package_path.parent.mkdir(parents=True, exist_ok=True)
        return package_path


def uniprot_subcellular_vocabulary(
    force_download: bool = False, cache_dir: Path | None = None
) -> dict[str, dict[str, Any]]:
    """
    Load UniProt controlled vocabulary for subcellular locations.

    This function downloads and parses the UniProt subcellular location
    controlled vocabulary, which provides standardized terms for cellular
    compartments along with their hierarchical relationships.

    Parameters
    ----------
    force_download
        If True, re-download and rebuild the vocabulary even if cached.
    cache_dir
        Directory to cache the vocabulary file. If None, uses package
        data directory or scanpy settings.

    Returns
    -------
    dict
        Dictionary mapping accession IDs (e.g., 'SL-0476') to location
        metadata. Each entry contains:

        - ID: Location name
        - DE: Definition/description
        - HI: List of is-a relationships (inheritance)
        - HP: List of part-of relationships (composition)
        - SY: List of synonyms (optional)
        - GO: List of Gene Ontology term IDs (optional)
        - SL: Hierarchy path string (optional)

    Examples
    --------
    Load the vocabulary:

    >>> from grassp.datasets.uniprot_cc import uniprot_subcellular_vocabulary
    >>> vocab = uniprot_subcellular_vocabulary()
    >>> len(vocab)
    564

    Look up a specific location:

    >>> nucleus = vocab['SL-0191']
    >>> nucleus['ID']
    'Nucleus'
    >>> nucleus['HP']  # What contains the nucleus
    ['Intracellular']

    Explore hierarchical relationships:

    >>> # Find locations that are part of the nucleus
    >>> for acc, entry in vocab.items():
    ...     if 'Nucleus' in entry.get('HP', []):
    ...         print(f"{entry['ID']} is part of Nucleus")

    Notes
    -----
    The vocabulary is cached locally after the first download to avoid
    repeated network requests. Use ``force_download=True`` to refresh
    the cached data.
    """
    cache_path = _get_cache_path(cache_dir)

    # Load from cache if exists and not forcing download
    if cache_path.exists() and not force_download:
        try:
            with gzip.open(cache_path, 'rb') as f:
                vocab = pickle.load(f)
            return vocab
        except (OSError, pickle.PickleError) as e:
            warnings.warn(f'Failed to load cached vocabulary: {e}. Re-downloading...')

    # Download and parse
    content = _download_subcell_vocabulary(BASE_URL)
    vocab = _parse_subcell_vocabulary(content)

    # Save to cache
    try:
        with gzip.open(cache_path, 'wb') as f:
            pickle.dump(vocab, f, protocol=pickle.HIGHEST_PROTOCOL)
    except (OSError, pickle.PickleError) as e:
        warnings.warn(f'Failed to cache vocabulary: {e}')

    return vocab


def find_roots(
    accession: str, vocab: dict[str, dict[str, Any]] | None = None, relationship: str = 'HP'
) -> list[str]:
    """
    Find root location(s) by traversing hierarchical relationships.

    Traverses the hierarchy upward from the given location to find root
    locations (those with no parent relationships). By default, follows
    HP (part-of) relationships to find what the location is contained within.

    Parameters
    ----------
    accession
        Accession ID of the starting location (e.g., 'SL-0191').
    vocab
        Vocabulary dictionary. If None, loads using uniprot_subcellular_vocabulary().
    relationship
        Type of relationship to follow: 'HP' (part-of) or 'HI' (is-a).
        Default is 'HP'.

    Returns
    -------
    list
        List of accession IDs representing root locations. Returns the input
        accession itself if it has no parent relationships.

    Examples
    --------
    Find what contains a location:

    >>> from grassp.datasets.uniprot_cc import find_roots
    >>> roots = find_roots('SL-0191')  # Nucleus
    >>> roots
    ['SL-0191']  # Nucleus is already a root (not contained in anything)

    Find root of a nested location:

    >>> roots = find_roots('SL-0350')  # Nuclear membrane
    >>> # Returns the top-level container(s)

    Notes
    -----
    - If a location has multiple parent relationships, all paths are explored
    - Cycles are detected and handled gracefully
    - Root locations are those with no parent relationships
    """
    if vocab is None:
        vocab = uniprot_subcellular_vocabulary()

    if accession not in vocab:
        raise ValueError(f'Accession {accession} not found in vocabulary')

    # Build name-to-accession mapping for efficient lookup
    name_to_acc = {entry['ID']: acc for acc, entry in vocab.items()}

    # Track visited to avoid cycles
    visited = set()
    roots = set()

    def _traverse(current_acc: str) -> None:
        """Recursively traverse up the hierarchy."""
        if current_acc in visited:
            return

        visited.add(current_acc)
        entry = vocab[current_acc]

        # Get parent relationships
        parents = entry.get(relationship, [])

        if not parents:
            # This is a root (no parents)
            roots.add(current_acc)
            return

        # Traverse each parent
        for parent_name in parents:
            # Look up parent by name
            parent_acc = name_to_acc.get(parent_name)
            if parent_acc:
                _traverse(parent_acc)
            else:
                # Parent not found in vocabulary, treat current as root
                roots.add(current_acc)

    _traverse(accession)

    return sorted(roots)  # Sort for consistent ordering
