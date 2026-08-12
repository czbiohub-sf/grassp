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

See `vignette("grasspio")` for the mapping table, the two conversions it handles for you, and
the limitations; and `docs/source/api/io.md` in the parent repository for the Python side.

The ~100 curated datasets on the [grassp portal](https://grassp.apps.czbiohub.org/datasets) are
h5ad files, so `grassp_as_msnset()` reads them directly — no Python needed to use them.

## A development environment

The environment is declared in [`r-environment.yml`](../../r-environment.yml) at the repository
root. From there:

```sh
make setup-r    # creates the conda env, installs grasspio, registers the `ir` Jupyter kernel
make test-r     # runs the testthat suite
```

`make test-r` uses `testthat::test_local()`, which loads only grasspio's own namespace — that
catches a test which passes merely because `pRoloc` happened to be attached.

Three things the file encodes that are easy to get wrong by hand:

- `anndataR` needs R >= 4.5, which sets the floor for everything else.
- `rhdf5` is required at runtime even though `anndataR` declares it only in *Suggests*, so
  nothing else pulls it in.
- On Apple silicon the environment must be built for `osx-64`: `bioconductor-msnbase` has no
  `osx-arm64` build, because its `affyio` dependency is missing there. Rosetta runs the x86_64
  packages without trouble, and `make setup-r` sets `CONDA_SUBDIR` for you. On Linux and Intel
  macOS that is a no-op.

Both `pRoloc` and `pRolocdata` are in bioconda, so nothing here needs a C++ toolchain or a
source build. The combination this package was last verified against:

    R 4.5.3 | MSnbase 2.36.0 | pRoloc 1.51.1 | anndataR 1.0.2 | rhdf5 2.54.1
