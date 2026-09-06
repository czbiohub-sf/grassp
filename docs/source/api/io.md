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

**There is nothing to import from `grassp.io` for this.** The exchange format is h5ad in both
directions, so Python writes with `adata.write_h5ad()` and reads with `anndata.read_h5ad()` —
no wrapper, no version block, no format to keep in step. Everything pRoloc-specific lives in the
companion R package, which is where pRoloc's conventions belong. It sits in this repository and
installs the R way, with no Python on the R side:

```r
install.packages(c("remotes", "BiocManager"))
BiocManager::install(c("pRoloc", "rhdf5"))
remotes::install_github("czbiohub-sf/grassp", subdir = "r/grasspio")
```

There are two tutorials: {doc}`for grassp users <../tutorials/notebooks/proloc_tutorial>` (the
round trip below) and {doc}`for pRoloc users <../tutorials/proloc_r_tutorial>` (reading a grassp
portal dataset in pRoloc, with no Python at all).

A full round trip. In Python:

```python
import anndata
import grassp as gr

adata = gr.ds.load_dataset("hek_dc_2025")
gr.pp.add_markers(adata, species="hsap")
adata.write_h5ad("experiment.h5ad")
```

then in R:

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
annotated = anndata.read_h5ad("results.h5ad")

annotated.obs["tagm.map.allocation"]           # pRoloc's own column names, verbatim
annotated.obsm["tagm.map.joint"]               # a DataFrame, whose columns are the classes
annotated.uns["neighbors"]                     # and everything you had before you left
```

Those are pRoloc's field names, kept verbatim so the round trip is lossless. grassp's *own*
TAGM ({func}`~grassp.tl.tagm_map_train` / {func}`~grassp.tl.tagm_map_predict`) writes under
`key_added` like every other annotator — `obs["tagm_map"]`, `obsm["tagm_map_probabilities"]`,
`uns["tagm_map_model"]` — and reads the dotted `uns["tagm.map.params"]` as well, so a
`MAPParams` grafted on from pRoloc can be predicted from directly. The two spellings coexist
by provenance: dotted came from pRoloc, underscored grassp produced.

`annotated` is as close to `adata` as the two data models allow — every `.obs`, `.var`, `.obsm`,
`.varm`, `.layers` and `.uns` entry survives the trip, plus whatever pRoloc added. The only
exception is `.obsp`/`.varp`; see [Known limitations](#known-limitations).

### Merging results onto the object you already have

pRoloc shrinks objects routinely — `filterNA()`, `markerMSnSet()`, plain `x[i, ]` — so
`results.h5ad` often covers fewer proteins than the session you sent it from. Copy over what you
want, reindexed:

```python
res = anndata.read_h5ad("results.h5ad")

adata.obs["svm.pred"] = res.obs["svm.pred"].reindex(adata.obs_names)
adata.obsm["svm.all.scores"] = res.obsm["svm.all.scores"].reindex(adata.obs_names)
```

The `reindex` is the load-bearing part: without it, a proteins-mismatch either raises (for
`.obsm`, which anndata index-checks) or, worse, aligns silently on position. Proteins the artifact
never saw come out `NaN`, which is what every grassp annotator treats as unlabelled.

### The mapping

| grassp `AnnData` | pRoloc `MSnSet` |
| --- | --- |
| `.X` | `exprs()` — **no transpose**; grassp already keeps proteins in `.obs`, matching MSnSet's features-by-fractions orientation |
| `.obs_names` / `.var_names` | `featureNames()` / `sampleNames()` |
| `.obs` scalar columns | `fData()` scalar columns |
| `.obsm[k]` as a **DataFrame** | `fData()[[k]]` **matrix-valued** column |
| `.var` | `pData()` |
| `.varm[k]` as a **DataFrame** | `pData()[[k]]` **matrix-valued** column |
| `.layers[k]` | extra `assayData` elements |
| `.uns["processing"]` | `processingData()@processing` |
| everything else in `.uns` | `experimentData()@other$grassp_uns` |
| `NaN` in any text column | the literal string `"unknown"` |

The matrix-column rows are what make this work. A matrix stored *inside a single* `fData`
column is how pRoloc represents per-protein × per-compartment values — `svm.all.scores`,
`tagm.map.joint`, `tagm.mcmc.joint`, `bandle.joint`, `Markers`, `GOAnnotations` — and it
corresponds exactly to `.obsm`. `pData` is the same `AnnotatedDataFrame` class, so `.varm` maps
onto matrix-valued `pData` columns by the same route, and `.layers` onto the extra matrices
`assayData` can already hold (it is a Biobase environment, not a single matrix).

Those come back as **DataFrames**, which is what lets the objects describe themselves instead of
needing a side table of column names: `annotated.obsm["svm.all.scores"].columns` is pRoloc's own
list of classes. An `.obsm` entry that goes *out* as a plain array is read just as happily — the
class names are looked for in `.uns[f"{key}_categories"]`, which is the fallback the R writer
uses for names HDF5 cannot take, and then in the categories of the companion label column.
Failing both it stays nameless, which is the right answer for an embedding: `X_pca` and `X_umap`
have no class names to recover, and they come back as the arrays they left as rather than as
frames of invented ones.

All three are safe under subsetting, which is what makes them usable rather than merely
storable — verified against pRoloc:

| Operation in R | what it subsets |
| --- | --- |
| `markerMSnSet(x)`, `x[i, ]` | matrix `fData` columns, and every `assayData` element |
| `x[, j]` | matrix `pData` columns, and every `assayData` element |
| `svmClassification`, `normalise` | preserve all of them untouched |

`exprs()` remains the matrix pRoloc's functions operate on; the extra `assayData` elements are
reachable with `assayDataElementNames(x)` and `assayDataElement(x, "pvals")`.

The `experimentData@other` row is what makes the trip *lossless* rather than merely useful. An
`MSnSet` has no slot for arbitrary metadata, so `grassp_as_msnset` parks any `.uns` entry with no
`MSnSet` slot — neighbour parameters, PCA variance ratios, colour maps, schema versions — on
`experimentData(x)@other$grassp_uns`, and `grassp_write_msnset` hands it straight back.
pRoloc never looks there. Unlike a graph, `.uns` is neither row- nor column-aligned, so there is
nothing for subsetting to get wrong; verified that it survives `[`, `markerMSnSet`, `normalise`
and `svmClassification`, and that `validObject` still passes.

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
  `"unknown"` and needs it — `markerMSnSet` and `unknownMSnSet` fail outright on `NA`, and
  `plot2D` colours it as a class of its own. grassp uses `NaN`, and its annotators select markers
  with `.notna()`, so an untranslated `"unknown"` would be trained on as a real compartment.
  `grassp_as_msnset(nan_to_unknown = TRUE)` fills it in as the `MSnSet` is built and
  `grassp_write_msnset(unknown_to_na = TRUE)` strips it on the way back, in **every** text
  column. The sentinel therefore never touches an h5ad file: it exists inside R, where it is the
  local convention, and nowhere else.

  Note that no column is nominated as *the* marker column. pRoloc's `fcol` is a per-call
  argument, so one `MSnSet` can carry `markers`, `markers.orig`, `pd.markers` and more at the
  same time and point different functions at different ones — verified against pRoloc, where
  each is independently usable as `fcol`. That is exactly how AnnData behaves too, so the bridge
  imposes no choice: you pick in R, at the call.
- **Missing values in text columns.** Going *out*, nothing is needed: `write_h5ad` runs
  anndata's own `strings_to_categoricals`, and a column containing `NaN` always has fewer
  distinct values than rows, so it is always stored as an h5ad categorical and arrives in R as a
  factor with a proper `NA`. Coming *back*, `anndataR` has no nullable-string encoding — it
  writes an R *character* `NA` as the literal two characters `"NA"`, which would arrive as a
  compartment called "NA" — so `grasspio` writes any `NA`-carrying character column as a factor
  instead, whose `NA` survives as a categorical with a `-1` code. Factors are never flattened the
  other way, which is what keeps a Categorical's **level order** and its `ordered` flag intact in
  both directions. Everything else `anndataR` maps faithfully already, verified column by column:
  `float64`/`numeric` (`NaN` included), `int64`/`integer`, `bool`/`logical`, and nullable
  `Int64`/`string`.

The one exception is {func}`~grassp.io.read_prolocdata`, which parses `.rda` files directly
with the pure-Python `rdata` package. No R is involved there, so it does the `"unknown"`
conversion itself.

(known-limitations)=
### Known limitations

- **`.obsp`/`.varp` are the only slots that cannot cross.** `eSet` has no pairwise slot, and
  pRoloc's one neighbour-ish representation, `nndist()`, writes flat *positional* indices that
  are silently wrong after any subsetting — so there is nothing to map onto. A graph is derived
  from `.X` anyway, so `gr.pp.neighbors` rebuilds it in one call. Nothing warns about it,
  because a warning would fire on every real dataset.
- **A class name containing `/` cannot be a DataFrame column,** because HDF5 reads it as a path
  separator. Both sides handle it, differently, and both round-trip: going *out* of R the matrix
  is written as a plain array with its class names in `.uns[f"{key}_categories"]`, and a message
  says so; going *out* of Python, {func}`grassp.util.set_matrix` rewrites each `/` to `; ` and
  warns. Either way nothing is silently lost. These names are common enough to plan for —
  `pp.add_markers` ships `Ribosome/Complexes` (human, `marker_christopher`) and
  `Secretory/Endocytic 1`–`3` (trypanosome, `marker_moloney`), and pRolocdata's `tan2009r1`
  labels a class `ER/Golgi`.
- **pRoloc does not write every result back to the object.** Optimisation and MCMC side objects
  — `GenRegRes` from `svmOptimisation` and friends, `MAPParams` from `tagmMapTrain`,
  `bandleParams` — are *returned* rather than added to the `MSnSet`, so they never reach the
  h5ad and do not come back. Only the summaries pRoloc writes into `fData` cross over. If you
  need one on the Python side, extract what you want into `fData`/`experimentData@other` in R
  before writing.
- **Nothing is renamed,** so a column called `svm` in R is a column called `svm` here. If you
  want grassp's own naming (`svm_annotation_probabilities` and friends), rename it yourself —
  the bridge deliberately does not guess.
- **Multi-localisation needs no dedicated support.** pRoloc represents it as a binary `Markers`
  matrix in `fData`, which is just a matrix-valued column — so one-hot the labels into `.obsm`
  as a DataFrame and it arrives as one:

  ```python
  from grassp.util import MULTILOC_SEP

  labels = adata.obs["soft_annotation_resolved_multiloc_label"].str.split(MULTILOC_SEP).explode()
  onehot = pd.get_dummies(labels).groupby(level=0).max().reindex(
      adata.obs_names, fill_value=0
  )
  adata.obsm["Markers"] = onehot.astype(float)   # a DataFrame: its columns are the class names
  adata.write_h5ad("experiment.h5ad")
  ```

  Two caveats, both pRoloc's: an all-zero row is how it encodes *unknown* under the matrix
  encoding, and `pRoloc::mrkMatToVec` collapses any protein with more than one label back to
  `"unknown"`, so do not round-trip through the vector encoding. pRoloc's classifiers also
  cannot train on a `Markers` matrix (`svmClassification` raises *"NA encountered in data"*) —
  it is for inspecting and subsetting marker sets, via `getMarkerClasses`, `markerMSnSet` and
  `unknownMSnSet`.
- **h5ad is the only format,** and [anndataR](https://github.com/scverse/anndataR) — scverse's
  native R implementation of the AnnData on-disk spec — the only reader/writer on the R side.
  That is where `grasspio`'s R >= 4.5 requirement comes from. You do need to install `rhdf5`
  yourself, because anndataR calls it but declares it only in *Suggests*; `grasspio` pins the
  anndataR revision it needs in its `Remotes:` field and pulls it in for you.
