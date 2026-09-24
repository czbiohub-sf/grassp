"""Shared machinery for the vocabulary-building scripts.

``fetch_compartments.py`` and ``fetch_go_cc.py`` were ported from two repositories that
had each grown their own copy of the near-duplicate collapse, the GMT writer and the
download helper -- and the copies had drifted (one applied an explicit drop-list the
other only approximated with regexes). Anything both recipes share lives here so the
next port has nothing left to fork.

Every script writes a provenance sidecar next to its GMT. The vendored vocabularies are
built from rolling downloads -- COMPARTMENTS publishes no versioned archive at all -- so
without a recorded build date and source a shipped GMT is an artifact nobody can date or
reproduce. :mod:`grassp.datasets.gene_sets` reads the sidecar back, and a test asserts
the accessor docstrings still agree with it.
"""

from __future__ import annotations
import json

from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import requests
import scipy.sparse as sp

#: Where the vendored GMTs and their sidecars go.
EXTERNAL = Path(__file__).resolve().parent.parent / "external"

REQUEST_TIMEOUT = 600


def download(url: str, dest: Path, force: bool = False) -> Path:
    """Fetch ``url`` to ``dest``, reusing the file if it is already there."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        print(f"[cache] {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
        return dest
    print(f"[get  ] {url}")
    response = requests.get(url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    dest.write_bytes(response.content)
    print(f"        -> {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


def deduplicate_identical(gene_sets: dict[str, list[str]]) -> dict[str, list[str]]:
    """Collapse terms whose gene membership is byte-identical, keeping the first seen.

    Runs before the Jaccard pass, matching what ``_load_gmt`` does on read, so the
    collapse sees the same vocabulary the published builds did.
    """
    seen: set[frozenset] = set()
    out: dict[str, list[str]] = {}
    for term, genes in gene_sets.items():
        signature = frozenset(genes)
        if signature in seen:
            continue
        seen.add(signature)
        out[term] = list(genes)
    return out


def collapse_near_duplicates(
    gene_sets: dict[str, list[str]], threshold: float, most_specific: bool
) -> dict[str, list[str]]:
    """Union-find collapse of terms whose gene sets have Jaccard >= ``threshold``.

    ``most_specific=True`` keeps the smallest gene set per cluster (GO CC, where the
    finer term is the informative one); ``False`` keeps the shortest name (COMPARTMENTS,
    where the canonical organelle name is).

    Note that this is transitive: A and B can be merged through a bridging term even
    when their own Jaccard is below the threshold. That is why the generic-term removal
    has to run *before* this and not after -- dropping a broad term afterwards does not
    undo the merges it caused.
    """
    terms = list(gene_sets)
    sets = [set(gene_sets[t]) for t in terms]
    sizes = np.array([len(s) for s in sets], dtype=float)
    genes = sorted(set().union(*sets)) if sets else []
    index = {g: i for i, g in enumerate(genes)}
    rows, cols = [], []
    for i, members in enumerate(sets):
        for g in members:
            rows.append(i)
            cols.append(index[g])
    matrix = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(terms), len(genes)))
    intersection = (matrix @ matrix.T).toarray()
    union = sizes[:, None] + sizes[None, :] - intersection
    jaccard = np.divide(
        intersection, union, out=np.zeros_like(intersection, dtype=float), where=union > 0
    )
    np.fill_diagonal(jaccard, 0)

    parent = list(range(len(terms)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in zip(*np.where(np.triu(jaccard >= threshold, 1))):
        parent[find(i)] = find(j)
    clusters: dict[int, list[int]] = defaultdict(list)
    for i in range(len(terms)):
        clusters[find(i)].append(i)

    # The term name is the final tie-break. Without it the two published builds
    # disagreed on four GO CC terms -- 'amylin receptor complex 1' vs '2', 'inner' vs
    # 'outer dense plaque of desmosome' -- exact ties on (size, name length) where
    # `min` fell through to dict iteration order, so the vendored file depended on the
    # order the GAF happened to be read in.
    key = (
        (lambda i: (sizes[i], len(terms[i]), terms[i]))
        if most_specific
        else (lambda i: (len(terms[i]), sizes[i], terms[i]))
    )
    keep = {terms[min(members, key=key)] for members in clusters.values()}
    return {t: gene_sets[t] for t in terms if t in keep}


def write_gmt(
    path: Path, gene_sets: dict[str, list[str]], descriptions: dict[str, str] | None = None
) -> None:
    """Write ``{term: genes}`` as a GMT, term order preserved.

    The description column carries the ontology identifier rather than a placeholder, so
    a term can be looked up in its source ontology without parsing the label.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for term, genes in gene_sets.items():
            description = (descriptions or {}).get(term, ".")
            handle.write("\t".join([term, description, *sorted(genes)]) + "\n")
    print(f"[write] {path.name}: {len(gene_sets)} terms")


def write_provenance(gmt_path: Path, **fields: Any) -> Path:
    """Record how a vendored GMT was built, as ``<stem>.json`` beside it.

    ``built`` and ``n_terms`` are filled in automatically; everything else is the
    recipe's own parameters, which is what makes a rebuild comparable to the shipped
    file rather than merely similar to it.
    """
    n_terms = sum(1 for line in gmt_path.open() if line.strip())
    payload = {
        "gmt": gmt_path.name,
        "built": date.today().isoformat(),
        "n_terms": n_terms,
        **fields,
    }
    out = gmt_path.with_suffix(".json")
    out.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
    print(f"[write] {out.name}: built {payload['built']}, {n_terms} terms")
    return out
