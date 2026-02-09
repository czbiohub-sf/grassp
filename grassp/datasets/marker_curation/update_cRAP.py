#!/usr/bin/env python3
"""Download and parse cRAP contaminant database.

The cRAP database (common Repository of Adventitious Proteins) is maintained
at https://ftp.thegpm.org/fasta/crap/ and contains common laboratory contaminants
found in proteomics experiments.

This script downloads the FASTA file, parses UniProt headers, and creates a
tab-separated file with columns: id, entry_name, source.

Usage:
    python -m grassp.datasets.marker_curation.update_cRAP
    python -m grassp.datasets.marker_curation.update_cRAP --cache-dir /tmp/cache
"""

from __future__ import annotations
import argparse
import logging
import time

from ftplib import FTP
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

# cRAP database URL
CRAP_URL = "ftp://ftp.thegpm.org/fasta/cRAP/crap.fasta"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)


def parse_fasta_header(header: str) -> Optional[str]:
    """Parse cRAP FASTA header to extract entry name.

    The cRAP FASTA has simplified headers with format:
    - SwissProt: >sp|ENTRY_NAME|
    - TrEMBL: >tr|ENTRY_NAME|

    Note: The cRAP database uses a simplified format without accession IDs.
    We extract the entry name and will use UniProt mapping if needed.

    Parameters
    ----------
    header
        FASTA header line (including or excluding leading '>')

    Returns
    -------
    str or None
        Entry name if parseable, None otherwise

    Examples
    --------
    >>> parse_fasta_header(">sp|ALBU_BOVIN|")
    'ALBU_BOVIN'
    >>> parse_fasta_header(">tr|K1C10_HUMAN|")
    'K1C10_HUMAN'
    """
    # Remove leading '>' if present
    header = header.lstrip('>')

    try:
        # Split by '|' to get database, entry_name
        # Format: sp|ENTRY_NAME| or tr|ENTRY_NAME|
        parts = header.split('|')
        if len(parts) < 2:
            return None

        # Extract entry name (second part)
        entry_name = parts[1].strip()

        return entry_name if entry_name else None

    except (IndexError, AttributeError):
        return None


def download_crap_fasta(url: str, cache_dir: Path) -> Path:
    """Download cRAP FASTA file with retry logic and caching.

    Parameters
    ----------
    url
        URL to download cRAP FASTA from (ftp:// or https://)
    cache_dir
        Directory to cache downloaded file

    Returns
    -------
    Path
        Path to downloaded FASTA file

    Raises
    ------
    RuntimeError
        If download fails after 3 attempts
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "crap.fasta"

    # Check cache first
    if cache_file.exists():
        logger.info(f"Using cached cRAP FASTA from {cache_file}")
        return cache_file

    logger.info(f"Downloading cRAP database from {url}...")

    # Parse URL
    if url.startswith('ftp://'):
        # Use FTP protocol
        # Parse: ftp://ftp.thegpm.org/fasta/cRAP/crap.fasta
        url_parts = url.replace('ftp://', '').split('/', 1)
        host = url_parts[0]
        remote_path = url_parts[1] if len(url_parts) > 1 else ''

        # Retry logic with exponential backoff
        for attempt in range(3):
            try:
                ftp = FTP(host, timeout=60)
                ftp.login()  # Anonymous login
                logger.debug(f"Connected to {host}")

                # Download file
                with open(cache_file, 'wb') as f:
                    ftp.retrbinary(f'RETR {remote_path}', f.write)

                ftp.quit()
                logger.info(f"Downloaded cRAP FASTA to {cache_file}")
                return cache_file

            except Exception as e:
                wait_time = 2**attempt  # Exponential backoff: 1s, 2s, 4s
                logger.warning(
                    f"Download attempt {attempt + 1}/3 failed: {e}. "
                    f"Retrying in {wait_time}s..."
                )
                if attempt < 2:  # Don't sleep on last attempt
                    time.sleep(wait_time)

    else:
        raise ValueError(f"Unsupported URL scheme: {url}. Only ftp:// is supported.")

    raise RuntimeError(f"Failed to download cRAP database from {url} after 3 attempts")


def map_entry_names_to_ids(entry_names: list[str]) -> dict[str, str]:
    """Map UniProt entry names to accession IDs using UniProt search API.

    Parameters
    ----------
    entry_names
        List of UniProt entry names (e.g., ['ALBU_BOVIN', 'AMYS_HUMAN'])

    Returns
    -------
    dict
        Mapping from entry_name to accession ID
    """
    logger.info(f"Mapping {len(entry_names)} entry names to accession IDs via UniProt API...")

    mapping = {}

    # Query UniProt for each entry name individually
    # Use the search API which is more reliable for entry names
    base_url = "https://rest.uniprot.org/uniprotkb/search"

    for i, entry_name in enumerate(entry_names):
        try:
            # Search for exact entry name match
            params = {
                'query': f'id:{entry_name}',
                'format': 'tsv',
                'fields': 'accession,id',
                'size': 1,
            }

            response = requests.get(base_url, params=params, timeout=10)
            response.raise_for_status()

            # Parse TSV response
            lines = response.text.strip().split('\n')
            if len(lines) > 1:  # Has header + data
                data_line = lines[1]
                accession = data_line.split('\t')[0]
                mapping[entry_name] = accession

            # Rate limiting
            if (i + 1) % 10 == 0:
                logger.info(f"Mapped {i + 1}/{len(entry_names)} entry names")
                time.sleep(0.5)  # Small delay to avoid rate limiting

        except Exception as e:
            logger.debug(f"Failed to map {entry_name}: {e}")

    logger.info(f"Successfully mapped {len(mapping)}/{len(entry_names)} entry names")
    return mapping


def parse_crap_fasta(fasta_path: Path) -> pd.DataFrame:
    """Parse cRAP FASTA file into DataFrame.

    Parameters
    ----------
    fasta_path
        Path to cRAP FASTA file

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: id, entry_name, source
    """
    logger.info(f"Parsing cRAP FASTA file from {fasta_path}...")

    entry_names = []
    skipped = 0

    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if not line.startswith('>'):
                continue

            entry_name = parse_fasta_header(line)

            if entry_name:
                entry_names.append(entry_name)
            else:
                skipped += 1
                logger.debug(f"Skipped malformed header: {line[:80]}")

    logger.info(f"Parsed {len(entry_names)} entry names ({skipped} skipped)")

    # Map entry names to accession IDs
    id_mapping = map_entry_names_to_ids(entry_names)

    # Build DataFrame
    entries = []
    for entry_name in entry_names:
        accession = id_mapping.get(entry_name)
        if accession:
            entries.append(
                {
                    'id': accession,
                    'entry_name': entry_name,
                    'source': 'cRAP',
                }
            )
        else:
            logger.warning(f"Could not map entry name to accession: {entry_name}")

    df = pd.DataFrame(entries)

    logger.info(f"Created DataFrame with {len(df)} cRAP entries")

    return df


def validate_dataframe(df: pd.DataFrame) -> bool:
    """Validate parsed cRAP DataFrame.

    Parameters
    ----------
    df
        DataFrame to validate

    Returns
    -------
    bool
        True if validation passes, False otherwise
    """
    logger.info("Validating parsed cRAP data...")

    # Check required columns
    required_cols = ['id', 'entry_name', 'source']
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        return False

    # Check for duplicate IDs
    duplicates = df['id'].duplicated()
    if duplicates.any():
        dup_ids = df.loc[duplicates, 'id'].tolist()
        logger.error(f"Found {len(dup_ids)} duplicate IDs: {dup_ids[:5]}...")
        return False

    # Check minimum entry count (expect ~100-150)
    if len(df) < 50:
        logger.error(f"Too few entries: {len(df)} (expected ~100-150)")
        return False

    # Check for missing values
    for col in required_cols:
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            logger.error(f"Column '{col}' has {n_missing} missing values")
            return False

    logger.info(f"Validation passed: {len(df)} unique cRAP entries")
    return True


def main():
    """Main function to download, parse, and save cRAP database."""
    parser = argparse.ArgumentParser(
        description="Download and parse cRAP contaminant database"
    )
    parser.add_argument(
        '--url',
        default=CRAP_URL,
        help=f"URL to download cRAP FASTA from (default: {CRAP_URL})",
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help="Output directory for cRAP.tsv (default: grassp/datasets/external/)",
    )
    parser.add_argument(
        '--cache-dir',
        type=Path,
        default=Path.home() / '.cache' / 'grassp',
        help="Cache directory for downloaded files",
    )

    args = parser.parse_args()

    # Determine output directory
    if args.output_dir is None:
        # Default to grassp/datasets/external/
        script_dir = Path(__file__).parent
        output_dir = script_dir.parent / 'external'
    else:
        output_dir = args.output_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'cRAP.tsv'

    try:
        # Download FASTA file
        fasta_path = download_crap_fasta(args.url, args.cache_dir)

        # Parse FASTA to DataFrame
        df = parse_crap_fasta(fasta_path)

        # Validate DataFrame
        if not validate_dataframe(df):
            logger.error("Validation failed")
            return 1

        # Save as TSV
        df.to_csv(output_file, sep='\t', index=False)
        logger.info(f"Saved cRAP database to {output_file}")

        return 0

    except Exception as e:
        logger.error(f"Error: {e}")
        return 1


if __name__ == '__main__':
    exit(main())
