#!/usr/bin/env python3
"""
Fetch external validation markers from UniProt, MitoCarta, and MitoCop for multiple species.

This script downloads:
1. UniProt features (signal peptides, transmembrane domains, etc.) via REST API
2. MitoCarta mitochondrial protein annotations (human and mouse only)
3. MitoCop mitochondrial protein annotations (human only)

Output: TSV files with merged external markers for each species.
"""

from __future__ import annotations
import argparse
import logging
import re
import time

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import requests

if TYPE_CHECKING:
    from typing import Optional

# Species configuration
SPECIES_CONFIG = {
    'hsap': {
        'taxid': 9606,
        'name': 'Human',
        'mitocarta_url': 'https://personal.broadinstitute.org/scalvo/MitoCarta3.0/Human.MitoCarta3.0.xls',
    },
    'mmus': {
        'taxid': 10090,
        'name': 'Mouse',
        'mitocarta_url': 'https://personal.broadinstitute.org/scalvo/MitoCarta3.0/Mouse.MitoCarta3.0.xls',
    },
    'atha': {'taxid': 3702, 'name': 'Arabidopsis thaliana'},
    'scer': {'taxid': 559292, 'name': 'Saccharomyces cerevisiae'},  # S288C strain
    'dmel': {'taxid': 7227, 'name': 'Drosophila melanogaster'},
}

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)


def parse_feature_notes(feature_string: str) -> str:
    """
    Parse UniProt feature annotation and extract unique note values.

    Removes "Name=xx" suffixes from notes to get just the feature type.

    Parameters
    ----------
    feature_string
        Raw UniProt feature annotation (e.g., 'TRANSMEM 1..20; /note="Helical"; ...')

    Returns
    -------
    str
        Semicolon-separated unique note values, or empty string if none found

    Examples
    --------
    >>> parse_feature_notes('TOPO_DOM 2..7; /note="Mitochondrial intermembrane"; TOPO_DOM 38..74; /note="Mitochondrial matrix"')
    'Mitochondrial intermembrane; Mitochondrial matrix'
    >>> parse_feature_notes('TRANSMEM 1..20; /note="Helical; Name=1"; TRANSMEM 40..60; /note="Helical; Name=2"')
    'Helical'
    """
    if pd.isna(feature_string) or feature_string == '':
        return ''

    # Find all /note="..." patterns
    note_pattern = r'/note="([^"]+)"'
    notes = re.findall(note_pattern, feature_string)

    # Clean each note: remove "; Name=xx" or similar suffixes
    cleaned_notes = []
    for note in notes:
        # Remove everything from "; Name=" onwards
        cleaned = re.sub(r'; Name=.*$', '', note)
        cleaned_notes.append(cleaned)

    # Get unique notes while preserving order
    unique_notes = []
    seen = set()
    for note in cleaned_notes:
        if note not in seen:
            unique_notes.append(note)
            seen.add(note)

    return '; '.join(unique_notes) if unique_notes else ''


def find_column(df: pd.DataFrame, possible_names: list[str]) -> Optional[str]:
    """
    Find first matching column name from list of possibilities.

    Parameters
    ----------
    df
        DataFrame to search
    possible_names
        List of possible column names

    Returns
    -------
    str or None
        First matching column name, or None if not found
    """
    for name in possible_names:
        if name in df.columns:
            return name
    return None


def fetch_uniprot_features(species_code: str, reviewed_only: bool = True) -> pd.DataFrame:
    """
    Fetch UniProt features for a species via REST API.

    Parameters
    ----------
    species_code
        Species code (e.g., 'hsap', 'mmus')
    reviewed_only
        If True, only fetch reviewed (SwissProt) entries

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: id, Transmembrane, Intramembrane,
        Topological domain, has_signal, has_transmem, has_intramem, has_topo_dom
    """
    config = SPECIES_CONFIG[species_code]
    taxid = config['taxid']
    species_name = config['name']

    logger.info(f"Fetching UniProt features for {species_name} (taxid: {taxid})...")

    # Build query
    reviewed_query = "reviewed:true" if reviewed_only else "reviewed:false"
    query = f"(model_organism:{taxid}) AND ({reviewed_query})"

    # API endpoint
    url = "https://rest.uniprot.org/uniprotkb/stream"
    params = {
        'query': query,
        'format': 'tsv',
        'fields': 'accession,gene_primary,ft_topo_dom,ft_transmem,ft_intramem,ft_signal',
    }

    # Retry logic with exponential backoff
    for attempt in range(3):
        try:
            response = requests.get(url, params=params, timeout=120)
            response.raise_for_status()
            break
        except requests.RequestException as e:
            if attempt == 2:
                logger.error(f"Failed to fetch UniProt data after 3 attempts: {e}")
                raise
            wait_time = 2**attempt
            logger.warning(f"Request failed, retrying in {wait_time}s... ({e})")
            time.sleep(wait_time)

    # Parse TSV response
    from io import StringIO

    df = pd.read_csv(StringIO(response.text), sep='\t')

    # Rename 'Entry' to 'id' and 'Gene Names (primary)' to 'gene_name'
    df = df.rename(columns={'Entry': 'id', 'Gene Names (primary)': 'gene_name'})

    # Create boolean columns (before parsing, to check for any content)
    # Use True for matches and NaN for non-matches (no False values)

    # Handle Signal peptide column if it exists (API may not return it)
    if 'Signal peptide' in df.columns:
        df['has_signal'] = (
            df['Signal peptide'].notna() & (df['Signal peptide'] != '')
        ).replace(False, np.nan)
        # Parse and then drop the column
        df['Signal peptide'] = df['Signal peptide'].apply(parse_feature_notes)
        df = df.drop(columns=['Signal peptide'])
    else:
        # If no signal column, set has_signal to NaN for all
        df['has_signal'] = np.nan
        logger.warning("Signal peptide column not returned by UniProt API")

    df['has_transmem'] = (df['Transmembrane'].notna() & (df['Transmembrane'] != '')).replace(
        False, np.nan
    )
    df['has_intramem'] = (df['Intramembrane'].notna() & (df['Intramembrane'] != '')).replace(
        False, np.nan
    )
    df['has_topo_dom'] = (
        df['Topological domain'].notna() & (df['Topological domain'] != '')
    ).replace(False, np.nan)

    # Parse feature annotations to extract note values
    df['Topological domain'] = df['Topological domain'].apply(parse_feature_notes)
    df['Transmembrane'] = df['Transmembrane'].apply(parse_feature_notes)
    df['Intramembrane'] = df['Intramembrane'].apply(parse_feature_notes)

    logger.info(f"Fetched {len(df)} proteins from UniProt")
    logger.info(f"  - Signal peptides: {df['has_signal'].sum()}")
    logger.info(f"  - Transmembrane: {df['has_transmem'].sum()}")
    logger.info(f"  - Intramembrane: {df['has_intramem'].sum()}")
    logger.info(f"  - Topological domain: {df['has_topo_dom'].sum()}")

    return df


def fetch_mitocarta_data(
    species_code: str, score_threshold: float = 1.0, cache_dir: Path = Path('./cache')
) -> Optional[pd.DataFrame]:
    """
    Fetch MitoCarta mitochondrial protein annotations.

    Only available for human (hsap) and mouse (mmus).

    Parameters
    ----------
    species_code
        Species code (e.g., 'hsap', 'mmus')
    score_threshold
        Minimum MitoCarta score to include
    cache_dir
        Directory to cache downloaded Excel files

    Returns
    -------
    pd.DataFrame or None
        DataFrame with columns: id, mitocarta, mitocarta_evidence, mitocarta_subloc
        Returns None if species doesn't have MitoCarta or download fails
    """
    config = SPECIES_CONFIG[species_code]

    # Check if MitoCarta is available for this species
    if 'mitocarta_url' not in config:
        return None

    species_name = config['name']
    url = config['mitocarta_url']

    logger.info(f"Fetching MitoCarta data for {species_name}...")

    # Create cache directory
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / f'MitoCarta_{species_code}.xls'

    # Download file if not cached
    if not cache_file.exists():
        logger.info(f"Downloading MitoCarta file to {cache_file}...")
        try:
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            cache_file.write_bytes(response.content)
        except requests.RequestException as e:
            logger.error(f"Failed to download MitoCarta data: {e}")
            return None
    else:
        logger.info(f"Using cached MitoCarta file: {cache_file}")

    # Read Excel file (second sheet, 0-indexed)
    # Use xlrd directly to bypass pandas version check
    try:
        import xlrd

        workbook = xlrd.open_workbook(cache_file)
        sheet = workbook.sheet_by_index(1)  # Second sheet (0-indexed)

        # Get column names from first row
        headers = [sheet.cell_value(0, col) for col in range(sheet.ncols)]

        # Read data rows
        data = []
        for row_idx in range(1, sheet.nrows):
            row_data = [sheet.cell_value(row_idx, col) for col in range(sheet.ncols)]
            data.append(row_data)

        # Create DataFrame
        df = pd.DataFrame(data, columns=headers)
    except Exception as e:
        logger.error(f"Failed to read MitoCarta Excel file: {e}")
        return None

    # Find columns with flexible matching
    uniprot_col = find_column(df, ['Uniprot', 'UniProt', 'UniProtID', 'UniProt ID'])
    score_col = find_column(df, ['MitoCarta2.0_Score', 'MitoCarta3.0_Score', 'Score'])
    evidence_col = find_column(
        df, ['MitoCarta3.0_Evidence', 'MitoCarta2.0_Evidence', 'Evidence']
    )
    subloc_col = find_column(
        df,
        [
            'MitoCarta3.0_SubMitoLocalization',
            'MitoCarta2.0_SubMitoLocalization',
            'SubMitoLocalization',
        ],
    )
    gene_col = find_column(df, ['Symbol', 'Gene', 'Gene name', 'gene symbol', 'gene_name'])

    # Report found columns
    logger.info("MitoCarta columns found:")
    logger.info(f"  - UniProt ID: {uniprot_col}")
    logger.info(f"  - Score: {score_col}")
    logger.info(f"  - Evidence: {evidence_col}")
    logger.info(f"  - SubLocalization: {subloc_col}")
    logger.info(f"  - Gene: {gene_col}")

    # Check required columns
    if uniprot_col is None:
        logger.error("Could not find UniProt ID column in MitoCarta file")
        return None
    if score_col is None:
        logger.error("Could not find Score column in MitoCarta file")
        return None

    # Convert score column to numeric, handling any non-numeric values
    df[score_col] = pd.to_numeric(df[score_col], errors='coerce')

    # Filter by score threshold
    df_filtered = df[df[score_col] > score_threshold].copy()
    logger.info(
        f"Filtered {len(df_filtered)} proteins with score > {score_threshold} "
        f"(from {len(df)} total)"
    )

    # Create output DataFrame
    result = pd.DataFrame()
    result['id'] = df_filtered[uniprot_col]
    if gene_col:
        result['gene_name'] = df_filtered[gene_col].values
    result['mitocarta'] = True

    # Add optional columns if available
    if evidence_col:
        result['mitocarta_evidence'] = df_filtered[evidence_col].values
    if subloc_col:
        result['mitocarta_subloc'] = df_filtered[subloc_col].values

    return result


def fetch_mitocop_data(
    species_code: str, cache_dir: Path = Path('./cache')
) -> Optional[pd.DataFrame]:
    """
    Fetch MitoCop mitochondrial protein annotations.

    Only available for human (hsap).

    Reference: Konig et al. 2021, Cell Metabolism
    https://doi.org/10.1016/j.cmet.2021.11.010

    Parameters
    ----------
    species_code
        Species code (e.g., 'hsap')
    cache_dir
        Directory to cache downloaded Excel files

    Returns
    -------
    pd.DataFrame or None
        DataFrame with columns: id, mitocop
        Returns None if species is not human or download fails
    """
    # MitoCop is only available for human
    if species_code != 'hsap':
        return None

    species_name = SPECIES_CONFIG[species_code]['name']
    url = 'https://ars.els-cdn.com/content/image/1-s2.0-S1550413121005295-mmc2.xlsx'

    logger.info(f"Fetching MitoCop data for {species_name}...")

    # Create cache directory
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / 'MitoCop_hsap.xlsx'

    # Download file if not cached
    if not cache_file.exists():
        logger.info(f"Downloading MitoCop file to {cache_file}...")
        try:
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            cache_file.write_bytes(response.content)
        except requests.RequestException as e:
            logger.error(f"Failed to download MitoCop data: {e}")
            return None
    else:
        logger.info(f"Using cached MitoCop file: {cache_file}")

    # Read Excel file (third sheet, 0-indexed = sheet 2)
    # Headers are in the second row, so skip the first row
    try:
        df = pd.read_excel(cache_file, sheet_name=2, header=1)
    except Exception as e:
        logger.error(f"Failed to read MitoCop Excel file: {e}")
        return None

    # Find the protein ID column
    id_col = find_column(df, ['Simplified protein IDs', 'Protein IDs', 'UniProt'])

    if id_col is None:
        logger.error("Could not find UniProt ID column in MitoCop file")
        logger.info(f"Available columns: {list(df.columns)}")
        return None

    gene_col = find_column(df, ['Gene name', 'Gene', 'gene symbol', 'Symbol'])

    logger.info("MitoCop columns found:")
    logger.info(f"  - UniProt ID: {id_col}")
    logger.info(f"  - Gene: {gene_col}")

    # Filter out rows with missing IDs
    df_filtered = df[df[id_col].notna()].copy()

    # Split rows where multiple IDs are present (separated by semicolon)
    rows_to_expand = []
    for idx, row in df_filtered.iterrows():
        ids = str(row[id_col]).split(';')
        # Clean and strip whitespace from each ID
        ids = [id_val.strip() for id_val in ids if id_val.strip()]

        gene_name = None
        if gene_col and pd.notna(row[gene_col]):
            gene_name = str(row[gene_col]).strip()

        for protein_id in ids:
            row_data = {'id': protein_id}
            if gene_name:
                row_data['gene_name'] = gene_name
            rows_to_expand.append(row_data)

    # Create output DataFrame
    result = pd.DataFrame(rows_to_expand)
    result['mitocop'] = True

    # Remove duplicates (if any protein appears multiple times)
    result = result.drop_duplicates(subset='id')

    logger.info(
        f"Extracted {len(result)} unique proteins from MitoCop "
        f"(from {len(df_filtered)} original rows)"
    )

    return result


def create_external_markers(
    species_code: str,
    reviewed_only: bool = True,
    mitocarta_threshold: float = 0.0,
    cache_dir: Path = Path('./cache'),
) -> pd.DataFrame:
    """
    Create merged external markers for a species.

    Parameters
    ----------
    species_code
        Species code (e.g., 'hsap', 'mmus')
    reviewed_only
        If True, only fetch reviewed UniProt entries
    mitocarta_threshold
        Minimum MitoCarta score to include
    cache_dir
        Directory to cache downloaded files

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with UniProt features, MitoCarta, and MitoCop annotations
    """
    # Fetch UniProt features
    uniprot_df = fetch_uniprot_features(species_code, reviewed_only)

    # Fetch MitoCarta data (if available)
    mitocarta_df = fetch_mitocarta_data(species_code, mitocarta_threshold, cache_dir)

    # Merge if MitoCarta data available
    if mitocarta_df is not None:
        merged_df = uniprot_df.merge(
            mitocarta_df, on='id', how='left', suffixes=('', '_mitocarta')
        )

        # Resolve gene_name: prefer UniProt, fill NaN from MitoCarta
        if 'gene_name_mitocarta' in merged_df.columns:
            merged_df['gene_name'] = merged_df['gene_name'].fillna(
                merged_df['gene_name_mitocarta']
            )
            merged_df = merged_df.drop(columns=['gene_name_mitocarta'])

        logger.info(
            f"Merged MitoCarta data: {merged_df['mitocarta'].sum()} mitochondrial proteins"
        )
    else:
        merged_df = uniprot_df

    # Fetch MitoCop data (if available)
    mitocop_df = fetch_mitocop_data(species_code, cache_dir)

    # Merge if MitoCop data available
    if mitocop_df is not None:
        merged_df = merged_df.merge(mitocop_df, on='id', how='left', suffixes=('', '_mitocop'))

        # Resolve gene_name: prefer existing, fill NaN from MitoCop
        if 'gene_name_mitocop' in merged_df.columns:
            merged_df['gene_name'] = merged_df['gene_name'].fillna(
                merged_df['gene_name_mitocop']
            )
            merged_df = merged_df.drop(columns=['gene_name_mitocop'])

        logger.info(
            f"Merged MitoCop data: {merged_df['mitocop'].sum()} mitochondrial proteins"
        )

    # Filter out proteins with no markers (all boolean columns are NaN)
    bool_cols = ['has_signal', 'has_transmem', 'has_intramem', 'has_topo_dom']
    if 'mitocarta' in merged_df.columns:
        bool_cols.append('mitocarta')
    if 'mitocop' in merged_df.columns:
        bool_cols.append('mitocop')

    # Keep rows where at least one boolean column is True (NaN is treated as False)
    has_any_marker = merged_df[bool_cols].any(axis=1)
    filtered_df = merged_df[has_any_marker].copy()

    logger.info(
        f"Filtered {len(merged_df) - len(filtered_df)} proteins with no markers "
        f"({len(filtered_df)} proteins retained)"
    )

    return filtered_df


def validate_dataframe(df: pd.DataFrame, species_code: str) -> bool:
    """
    Validate output DataFrame.

    Parameters
    ----------
    df
        DataFrame to validate
    species_code
        Species code

    Returns
    -------
    bool
        True if validation passes
    """
    # Check for duplicates
    duplicates = df['id'].duplicated().sum()
    if duplicates > 0:
        logger.error(f"Found {duplicates} duplicate IDs in output")
        return False

    # Check row count
    if len(df) < 100:
        logger.warning(f"Low row count: {len(df)} proteins (expected > 100)")

    # Check boolean columns (now contain True/NaN, so dtype is float or object)
    bool_cols = ['has_signal', 'has_transmem', 'has_intramem', 'has_topo_dom']
    for col in bool_cols:
        if col in df.columns:
            # Check that column only contains True or NaN
            unique_vals = df[col].dropna().unique()
            if len(unique_vals) > 0 and not all(val for val in unique_vals):
                logger.error(f"Column {col} contains unexpected values: {unique_vals}")
                return False

    # Check gene_name column exists
    if 'gene_name' not in df.columns:
        logger.error("gene_name column is missing")
        return False

    # Check coverage
    n_with_gene_name = df['gene_name'].notna().sum()
    coverage = n_with_gene_name / len(df) * 100
    logger.info(f"Gene name coverage: {n_with_gene_name}/{len(df)} ({coverage:.1f}%)")

    if coverage < 50:
        logger.warning(f"Low gene name coverage: {coverage:.1f}%")

    logger.info(f"Validation passed for {species_code}")
    return True


def main():
    """Main script entry point."""
    parser = argparse.ArgumentParser(
        description='Fetch external validation markers from UniProt and MitoCarta'
    )
    parser.add_argument(
        '--species',
        nargs='+',
        default=['all'],
        choices=['all', 'hsap', 'mmus', 'atha', 'scer', 'dmel'],
        help='Species to process (default: all)',
    )
    parser.add_argument(
        '--reviewed-only',
        action='store_true',
        default=True,
        help='Only fetch reviewed (SwissProt) UniProt entries (default: True)',
    )
    parser.add_argument(
        '--mitocarta-threshold',
        type=float,
        default=-100.0,
        help='Minimum MitoCarta score threshold (default: -100.0 to include all)',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('../external/'),
        help='Output directory (default: ../external/)',
    )
    parser.add_argument(
        '--cache-dir',
        type=Path,
        default=Path('./cache'),
        help='Cache directory for downloaded files (default: ./cache)',
    )

    args = parser.parse_args()

    # Determine species to process
    if 'all' in args.species:
        species_list = list(SPECIES_CONFIG.keys())
    else:
        species_list = args.species

    logger.info(f"Processing {len(species_list)} species: {', '.join(species_list)}")
    logger.info(f"Reviewed only: {args.reviewed_only}")
    logger.info(f"MitoCarta threshold: {args.mitocarta_threshold}")
    logger.info(f"Output directory: {args.output_dir}")

    # Create output directory
    args.output_dir.mkdir(exist_ok=True, parents=True)

    # Process each species
    results = {}
    for species_code in species_list:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Processing {SPECIES_CONFIG[species_code]['name']} ({species_code})")
        logger.info(f"{'=' * 60}")

        try:
            # Create external markers
            df = create_external_markers(
                species_code,
                args.reviewed_only,
                args.mitocarta_threshold,
                args.cache_dir,
            )

            # Validate
            if not validate_dataframe(df, species_code):
                logger.error(f"Validation failed for {species_code}, skipping...")
                continue

            # Save
            output_file = args.output_dir / f'external_markers_{species_code}.tsv'
            df.to_csv(output_file, sep='\t', index=False)
            logger.info(f"Saved {len(df)} proteins to {output_file}")

            # Store results for summary
            results[species_code] = {
                'total': len(df),
                'has_signal': df['has_signal'].sum(),
                'has_transmem': df['has_transmem'].sum(),
                'has_intramem': df['has_intramem'].sum(),
                'has_topo_dom': df['has_topo_dom'].sum(),
                'mitocarta': df['mitocarta'].sum() if 'mitocarta' in df.columns else 0,
                'mitocop': df['mitocop'].sum() if 'mitocop' in df.columns else 0,
            }

        except Exception as e:
            logger.error(f"Failed to process {species_code}: {e}", exc_info=True)
            continue

    # Print summary
    logger.info(f"\n{'=' * 60}")
    logger.info("SUMMARY")
    logger.info(f"{'=' * 60}")
    logger.info(f"Successfully processed {len(results)}/{len(species_list)} species")

    for species_code, stats in results.items():
        logger.info(f"\n{SPECIES_CONFIG[species_code]['name']} ({species_code}):")
        logger.info(f"  Total proteins: {stats['total']}")
        logger.info(f"  Signal peptides: {stats['has_signal']}")
        logger.info(f"  Transmembrane: {stats['has_transmem']}")
        logger.info(f"  Intramembrane: {stats['has_intramem']}")
        logger.info(f"  Topological domain: {stats['has_topo_dom']}")
        if stats['mitocarta'] > 0:
            logger.info(f"  MitoCarta proteins: {stats['mitocarta']}")
        if stats['mitocop'] > 0:
            logger.info(f"  MitoCop proteins: {stats['mitocop']}")


if __name__ == '__main__':
    main()
