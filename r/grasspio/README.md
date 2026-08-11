# grasspio

Move spatial proteomics data between MSnbase `MSnSet` objects — as used by
[pRoloc](https://bioconductor.org/packages/pRoloc/) and
[bandle](https://bioconductor.org/packages/bandle/) — and the h5ad files written by the
Python package [grassp](https://github.com/czbiohub-sf/grassp).

```r
install.packages("remotes")
# anndataR (scverse's native R implementation of the AnnData spec) is a hard dependency and
# comes in automatically. rhdf5 does not: anndataR calls it but declares it only in Suggests.
BiocManager::install("rhdf5")
remotes::install_github("czbiohub-sf/grassp", subdir = "r/grasspio")
```

Needs R >= 4.5, which is anndataR's own requirement.

```r
library(grasspio)
library(pRoloc)

x <- grassp_as_msnset("experiment.h5ad")       # -> MSnSet, ready for pRoloc
x <- svmClassification(x, fcol = "markers", scores = "all")
grassp_write_msnset(x, "results.h5ad")         # -> h5ad, ready for grassp
```

Then in Python: `adata = gr.io.read_msnset("results.h5ad")`.

pRoloc and MSnbase ship no exporter of their own, so `grassp_write_msnset()` is useful even
if you never touch grassp — it is a way to hand a classified `MSnSet` to anything that reads
h5ad.

Two vignettes:

- `vignette("grasspio")` — the exchange format itself: the mapping table, what crosses, and the
  limitations. Start here if you are moving your own object between R and Python.
- `vignette("portal")` — download a dataset from the
  [grassp portal](https://grassp.apps.czbiohub.org/datasets) and analyse it in pRoloc, including
  reproducing the PCA coordinates the portal ships. No Python involved.

See `docs/source/api/io.md` in the parent repository for the Python side.

## A known-good development environment

`anndataR` needs R >= 4.5, and `rhdf5` is required at runtime even though `anndataR` does not
declare it as a hard dependency. On Apple silicon, `bioconductor-msnbase` currently has no
`osx-arm64` build (its `affyio` dependency is missing), so build the environment for `osx-64`
and let Rosetta run it:

```sh
CONDA_SUBDIR=osx-64 conda create -n grassp-r -c conda-forge -c bioconda \
    bioconductor-msnbase bioconductor-anndatar bioconductor-rhdf5 r-testthat

# pRoloc is not in bioconda; build it from source. It needs a C++ toolchain.
CONDA_SUBDIR=osx-64 conda install -n grassp-r -c conda-forge \
    clang_osx-64 clangxx_osx-64 gfortran_osx-64 make
conda activate grassp-r
R CMD INSTALL /path/to/pRoloc

R CMD INSTALL r/grasspio
# test_local() loads only grasspio's own namespace, so it catches a test that passes merely
# because pRoloc happened to be attached.
Rscript -e 'setwd("r/grasspio"); library(testthat); test_local()'
```

The combination this package was last verified against:

    R 4.5.3 | MSnbase 2.36.0 | pRoloc 1.51.1 | anndataR 1.0.2 | rhdf5 2.54.1

On Linux, or on Intel macOS, the `CONDA_SUBDIR` prefix is unnecessary.
