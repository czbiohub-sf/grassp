# grasspio

Move spatial proteomics data between MSnbase `MSnSet` objects — as used by
[pRoloc](https://bioconductor.org/packages/pRoloc/) and
[bandle](https://bioconductor.org/packages/bandle/) — and h5ad, the format of the Python
package [grassp](https://github.com/czbiohub-sf/grassp).

There is no exchange format and no version block to keep in step: the files are ordinary h5ad.
grassp writes one with `adata.write_h5ad()` and reads one with `anndata.read_h5ad()`, and
everything pRoloc-specific happens on this side.

```r
install.packages("remotes")
# anndataR (scverse's native R implementation of the AnnData spec) is a hard dependency and
# comes in automatically. rhdf5 does not: anndataR calls it but declares it only in Suggests.
BiocManager::install("rhdf5")
remotes::install_github("czbiohub-sf/grassp", subdir = "r/grasspio")
```

Needs R >= 4.5, which is anndataR's own requirement.

> **anndataR comes from a fork, for now.** Matrix-valued `fData` columns are written as `obsm`
> data frames so that they carry their own class names, and the fix that makes anndataR store the
> parent's `obs_names` as such a frame's index is newer than the last release — without it, Python
> refuses to open the file with *"value.index does not match parent's obs names"*. The fork is
> pinned in the `Remotes:` field of [`DESCRIPTION`](DESCRIPTION), which `remotes` and `pak` both
> honour, so the install above picks it up and CI needs no special case. **When the fix ships,
> deleting those two lines is the only change needed.**

```r
library(grasspio)
library(pRoloc)

x <- grassp_as_msnset("experiment.h5ad")       # -> MSnSet, ready for pRoloc
x <- svmClassification(x, fcol = "markers", scores = "all")
grassp_write_msnset(x, "results.h5ad")         # -> h5ad, ready for grassp
```

Then in Python: `adata = anndata.read_h5ad("results.h5ad")`.

pRoloc and MSnbase ship no exporter of their own, so `grassp_write_msnset()` is useful even
if you never touch grassp — it is a way to hand a classified `MSnSet` to anything that reads
h5ad.

See `vignette("grasspio")` for the mapping table, the one value translation it handles for you,
and the limitations; and `docs/source/api/io.md` in the parent repository for the Python side.

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

    R 4.5.3 | MSnbase 2.36.0 | pRoloc 1.51.1 | anndataR 1.3.1 (from GitHub) | rhdf5 2.54.1
