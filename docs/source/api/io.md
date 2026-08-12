# IO: `io`

```{eval-rst}
.. module:: grassp.io
```

```{eval-rst}
.. currentmodule:: grassp
```

Read proteomics data from various search engines and file formats into AnnData objects. These functions leverage the [protdata](https://protdata.sf.czbiohub.org/) package to parse search engine outputs and standardize proteomics data for analysis.

## Search Engine Outputs

Read proteomics data from common search engine outputs (MaxQuant, DIA-NN, FragPipe).

```{eval-rst}
.. autosummary::
   :nosignatures:
   :toctree: ../generated/

   io.read_maxquant
   io.read_diann
   io.read_fragpipe
```

## Other Formats

Read data from other subcellular proteomics analysis tools.

```{eval-rst}
.. autosummary::
   :nosignatures:
   :toctree: ../generated/

   io.read_prolocdata
```

## pRoloc interoperability

Round-trip grassp objects through the R/Bioconductor
[pRoloc](https://bioconductor.org/packages/pRoloc/) framework, so that grassp's
preprocessing and plotting can be combined with pRoloc's classifiers (SVM, k-NN, TAGM,
phenoDisco) and, ultimately, with [bandle](https://bioconductor.org/packages/bandle/) for
differential localisation.

The exchange format is h5ad in **both** directions. The R half is a companion package that
lives in this repository and is installed the R way — no Python needed on the R side:

```r
install.packages(c("remotes", "BiocManager"))
BiocManager::install(c("pRoloc", "rhdf5"))
remotes::install_github("czbiohub-sf/grassp", subdir = "r/grasspio")
```

There are two tutorials: {doc}`the Python side <../tutorials/notebooks/proloc_tutorial>` (the
round trip below) and {doc}`the R side <../tutorials/proloc_r_tutorial>` (reading a grassp portal
dataset in pRoloc, with no Python at all).

A full round trip. In Python:

```python
import grassp as gr

adata = gr.ds.load_dataset("hek_dc_2025")
gr.pp.add_markers(adata, species="hsap")
gr.io.write_msnset(adata, "experiment.h5ad", write_script=True)
```

then in R (`write_script=True` generates exactly this as `experiment_run_proloc.R`):

```r
library(grasspio)
library(pRoloc)

x <- grassp_as_msnset("experiment.h5ad")
x <- svmClassification(x, fcol = "markers", scores = "all")
x <- tagmMapPredict(x, params = tagmMapTrain(x), probJoint = TRUE)
grassp_write_msnset(x, "results.h5ad")
```

and back in Python:

```python
gr.io.list_msnset_results("results.h5ad")      # look before you read
annotated = gr.io.read_msnset("results.h5ad")

annotated.obs["tagm.map.allocation"]           # pRoloc's own column names, verbatim
annotated.obsm["tagm.map.joint"]               # its per-compartment probabilities
annotated.uns["neighbors"]                     # and everything you had before you left
```

`annotated` is as close to `adata` as the two data models allow — every `.obs`, `.var`, `.obsm`,
`.varm`, `.layers` and `.uns` entry survives the trip, plus whatever pRoloc added. The only
exception is `.obsp`/`.varp`; see [Known limitations](#known-limitations).

If you would rather keep working on the object already in your session,
{func}`~grassp.io.read_proloc_results` grafts the new `.obs` columns and `.obsm` matrices onto it
instead of rebuilding it:

```python
gr.io.read_proloc_results("results.h5ad", adata)   # in place; leaves .X and .layers alone
```

```{eval-rst}
.. autosummary::
   :nosignatures:
   :toctree: ../generated/

   io.write_msnset
   io.read_msnset
   io.read_proloc_results
   io.list_msnset_results
```

### The mapping contract

| grassp `AnnData` | pRoloc `MSnSet` |
| --- | --- |
| `.X` or `.layers[layer]` | `exprs()` — **no transpose**; grassp already keeps proteins in `.obs`, matching MSnSet's features-by-fractions orientation |
| `.obs_names` / `.var_names` | `featureNames()` / `sampleNames()` |
| `.obs` scalar columns | `fData()` scalar columns |
| `.obsm[k]` + `.uns["obsm_colnames"][k]` | `fData()[[k]]` **matrix-valued** column |
| `.var` | `pData()` |
| `.varm[k]` + `.uns["varm_colnames"][k]` | `pData()[[k]]` **matrix-valued** column |
| `.layers[k]` | extra `assayData` elements |
| `.uns["processing"]` | `processingData()@processing` |
| everything else in `.uns` | `experimentData()@other$grassp_uns` |
| `NaN` in any text column | the literal string `"unknown"` |

The matrix-column rows are what make this work. A matrix stored *inside a single* `fData`
column is how pRoloc represents per-protein × per-compartment values — `svm.all.scores`,
`tagm.map.joint`, `tagm.mcmc.joint`, `bandle.joint`, `Markers`, `GOAnnotations` — and it
corresponds exactly to `.obsm`. `pData` is the same `AnnotatedDataFrame` class, so `.varm` maps
onto matrix-valued `pData` columns by the same route, and `.layers` onto the extra matrices
`assayData` can already hold (it is a Biobase environment, not a single matrix). Because
`.obsm`/`.varm` arrays carry no column names, those travel in `.uns["obsm_colnames"]` and
`.uns["varm_colnames"]`.

All three are safe under subsetting, which is what makes them usable rather than merely
storable — verified against pRoloc:

| Operation in R | what it subsets |
| --- | --- |
| `markerMSnSet(x)`, `x[i, ]` | matrix `fData` columns, and every `assayData` element |
| `x[, j]` | matrix `pData` columns, and every `assayData` element |
| `svmClassification`, `normalise` | preserve all of them untouched |

`exprs()` remains the matrix pRoloc's functions operate on; the extra `assayData` elements are
reachable with `assayDataElementNames(x)` and `assayDataElement(x, "pvals")`.

The last row of the table is what makes the trip *lossless* rather than merely useful. An
`MSnSet` has no slot for arbitrary metadata, so `grassp_as_msnset` parks any `.uns` entry the
contract does not map — neighbour parameters, PCA variance ratios, colour maps, schema versions —
on `experimentData(x)@other$grassp_uns`, and `grassp_write_msnset` hands it straight back.
pRoloc never looks there. Unlike a graph, `.uns` is neither row- nor column-aligned, so there is
nothing for subsetting to get wrong; verified that it survives `[`, `markerMSnSet`, `normalise`
and `svmClassification`, and that `validObject` still passes.

The contract's own `uns` keys — `msnset_spec`, `msnset_exprs_layer`, `msnset_dropped`,
`obsm_colnames`, `varm_colnames` — are the exception: both sides regenerate them from the object
on every write rather than copying them, so an artifact always describes itself even after the
object has been subset in R.

Import is therefore **structural, and does almost nothing**. Every scalar `fData` column
becomes an `.obs` column and every matrix-valued one becomes an `.obsm` entry, under its own
name and with its own dtype. So `svm`, `knn`, `nb`, `nnet`, `rf`, `plsda`, `perTurbo`, `ksvm`,
`phenoDisco`, TAGM-MAP, TAGM-MCMC and BANDLE all work — as does any classifier added to pRoloc
after this was written. There is no mapping table and no list of method names.

Columns keep **pRoloc's own names**, so what you see in Python is what `fvarLabels(x)` showed
in R. Where a grassp plotting helper wants a probability, pass pRoloc's score column:

```python
gr.pl.umap_prob(adata, color="svm.pred", color_prob="svm.scores")
```

Two things genuinely differ between the frameworks, and both are handled on the R side because
they are pRoloc's conventions:

- **`"unknown"` ↔ `NaN`.** pRoloc encodes unlabelled features as the literal string
  `"unknown"` and needs it — `markerMSnSet` and `unknownMSnSet` fail outright on `NA`. grassp
  uses `NaN`, and its annotators select markers with `.notna()`, so an untranslated `"unknown"`
  would be trained on as a real compartment. `write_msnset(nan_to_unknown=True)` converts on the
  way out and `grassp_write_msnset(unknown_to_na = TRUE)` on the way back, in **every** text
  column — string, object and Categorical alike, with numeric and boolean columns left alone.

  Note that no column is nominated as *the* marker column. pRoloc's `fcol` is a per-call
  argument, so one `MSnSet` can carry `markers`, `markers.orig`, `pd.markers` and more at the
  same time and point different functions at different ones — verified against pRoloc, where
  each is independently usable as `fcol`. That is exactly how AnnData behaves too, so the bridge
  imposes no choice: you pick in R, at the call.
- **Missing values in text columns.** `anndataR` maps types faithfully in general — verified by
  round-tripping one column of each kind: `float64`/`numeric` (`NaN` included), `int64`/`integer`,
  `bool`/`logical`, pandas Categorical ↔ R `factor` with its levels *and* its ordered-ness
  intact, and nullable `Int64`/`string`. The single exception is that it writes an R *character*
  `NA` as the literal two-character string `"NA"`, which would arrive as a compartment called
  "NA". `grasspio` therefore writes any `NA`-carrying column as a factor, whose `NA` survives —
  which is also why label columns come back as Categoricals with proper `NaN`.

The one exception is {func}`~grassp.io.read_prolocdata`, which parses `.rda` files directly
with the pure-Python `rdata` package. No R is involved there, so it does the `"unknown"`
conversion itself.

(known-limitations)=
### Known limitations

- **`.obsp`/`.varp` are the one thing that cannot cross.** `eSet` has no pairwise slot, and
  pRoloc has no graph structure to map onto. Its one neighbour-ish representation, `nndist()`,
  writes `2k` flat `fData` columns of **positional** indices — not an `n × n` graph, and silently
  wrong after any subsetting — so there is nothing to map to. This is also the slot with the best
  reason to stay in Python: a graph is derived from `.X`, so `gr.pp.neighbors` rebuilds it in one
  call, and pRoloc recomputes its own neighbours for everything it does anyway. It is listed in
  `.uns["msnset_dropped"]` rather than warned about, because a warning would fire on every real
  dataset.
- Everything else crosses by default, including embeddings — a matrix `fData` column is safe
  whatever it holds. Class names come from `uns["obsm_colnames"]` / `uns["varm_colnames"]`, else
  `uns[f"{key}_categories"]`, else the categories of the companion label column `obs[stem]`,
  else `V1..Vn`. That third source matters: portal datasets carry
  `harmonized_annotation_propagated_probabilities` with no `_categories` entry, and its names
  live on `obs["harmonized_annotation_propagated"]`. Use `layers` / `obsm_keys` / `varm_keys` /
  `obs_columns` / `var_columns` to send less; whatever you exclude is listed in
  `.uns["msnset_dropped"]` too.
- {func}`~grassp.io.read_proloc_results` is narrower than
  {func}`~grassp.io.read_msnset` **by design**: it copies only `.obs` and `.obsm`, because its
  promise is not to disturb the object it is merging onto. So `.X`, `.layers`, `.var`, `.varm` and
  `.obsp` do not come back that way — if the R side altered a matrix, read the artifact with
  `read_msnset`. What it *does* write goes under the artifact's own names, so a same-named `.obs`
  column or `.obsm` entry is replaced; since an export carries every `.obsm` entry by default,
  that includes your embeddings. Pass `key_prefix` or `suffix` to keep the two apart.
- Multi-localisation has no dedicated support, because it needs none. pRoloc represents it as a
  binary `Markers` matrix in `fData`, and `.obsm` is already exported as matrix `fData` columns —
  so one-hot the labels yourself and pass the result through `obsm_keys`:

  ```python
  labels = adata.obs["soft_multiloc_label"].str.split(" / ").explode()
  onehot = pd.get_dummies(labels).groupby(level=0).max().reindex(
      adata.obs_names, fill_value=0
  )
  adata.obsm["Markers"] = onehot.to_numpy(dtype=float)
  adata.uns.setdefault("obsm_colnames", {})["Markers"] = list(onehot.columns)

  gr.io.write_msnset(adata, "experiment.h5ad", obsm_keys=["Markers"])
  ```

  Two caveats, both pRoloc's: an all-zero row is how pRoloc encodes *unknown* under the matrix
  encoding, and `pRoloc::mrkMatToVec` collapses any protein with more than one label back to
  `"unknown"`, so do not round-trip through the vector encoding. Note also that pRoloc's
  classifiers cannot train on a `Markers` matrix (`svmClassification` raises *"NA encountered in
  data"*) — it is for inspecting and subsetting marker sets, via `getMarkerClasses`,
  `markerMSnSet` and `unknownMSnSet`.
- Optimisation and MCMC side objects (`GenRegRes`, `MAPParams`, `bandleParams`) live outside
  `fData`, so only the summaries pRoloc writes into `fData` cross over.
- h5ad is the only exchange format, and `anndataR` — scverse's native R implementation of the
  AnnData on-disk spec — the only reader/writer on the R side. `grasspio` depends on it outright,
  which is where its R >= 4.5 requirement comes from; you do need to install `rhdf5` yourself,
  because anndataR calls it but declares it only in *Suggests*. A `zellkonverter` fallback was
  tried and removed: `SingleCellExperiment` has no `varm` analogue and `readH5AD` does not
  surface `layers`, so both were silently dropped in either direction, and mapping `.obs` onto
  SCE *columns* transposes the whole object relative to an `MSnSet`.
- Nothing is renamed, so a column called `svm` in R is a column called `svm` here. If you want
  grassp's own naming conventions, rename it yourself — the bridge deliberately does not guess.
