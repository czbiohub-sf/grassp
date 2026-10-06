#!/usr/bin/env Rscript
## Generate the `mini_*` pRolocdata fixtures: real pRolocdata objects cut down to a few dozen
## proteins, so `gr.io.read_prolocdata` has something genuine to parse in the test suite.
##
## Run from the repository root, in an R environment that has MSnbase and pRolocdata:
##
##     Rscript grassp/tests/fixtures/make_prolocdata_fixtures.R
##
## Every other `read_prolocdata` test mocks the `rdata` package and so never exercises its
## conversion of R's serialisation format -- which is exactly where things break (hyperLOPIT2015
## was unreadable until 2026-10-05 and nothing noticed). These slices are the reader's contract.
##
## The objects are *subset*, never rebuilt: `x[idx, ]` keeps fData, pData, experimentData, the
## processing log and the S4 class versions exactly as pRolocdata serialised them, so whatever
## quirk the original carries, the slice carries too (the one thing subsetting changes, the
## processing log, is restored). The R object keeps its original name --
## `read_prolocdata` reports the first object in the file as `uns["dataset_name"]` -- and each
## file keeps the original's extension, since both `.rda` and `.RData` occur in pRolocdata.
##
## Rows: up to ROWS_PER_CLASS proteins per marker class, sampled with a fixed seed, plus
## N_UNKNOWN unlabelled ones -- drawn from the proteins with a missing quantitation where the
## dataset has any, so the NaN path is exercised too. Original row order is preserved.
##
## Why these four (chosen by the parser quirk each carries, not by fame):
##
##   * hyperLOPIT2015 -- `fData$TAGM` is a *nested data.frame* (the published TAGM-MAP and
##     TAGM-MCMC results), `.RData` extension, 15 character marker classes whose names contain
##     "/" and " - ", and a processing log.
##   * dunkley2006 -- factor columns (`markers.orig`, `pd.markers`, `pd.2013`, `assigned`,
##     `new`) next to a character `markers`, so the "unknown" sentinel appears in both a
##     Categorical and an object column; `.RData`; the dataset the pRoloc tutorial downloads.
##   * itzhak2016stcSILAC -- NaN in `exprs` (1499 incomplete proteins), fData column names with
##     spaces and a double space ("Organellar  markers with sub-compartments"), and an empty
##     processing log (so `uns["processing"]` must be absent, not an empty list); `.rda`.
##   * tan2009r1 -- four fractions only, factor columns with non-alphabetical level order
##     (`PLSDA`, `markers.orig`, `markers.tl`) that must survive into pandas Categoricals.

suppressMessages({
  library(MSnbase)
  library(pRolocdata)
})

ROWS_PER_CLASS <- 2L
N_UNKNOWN <- 6L
UNKNOWN <- "unknown"

datasets <- c(
  hyperLOPIT2015 = "RData",
  dunkley2006 = "RData",
  itzhak2016stcSILAC = "rda",
  tan2009r1 = "RData"
)

cat("pRolocdata", as.character(packageVersion("pRolocdata")),
    "MSnbase", as.character(packageVersion("MSnbase")), "\n")

pick_rows <- function(x) {
  set.seed(2015L)
  markers <- fData(x)$markers
  labelled <- which(markers != UNKNOWN)
  by_class <- split(labelled, markers[labelled])
  keep <- unlist(lapply(by_class, function(rows) {
    if (length(rows) <= ROWS_PER_CLASS) rows else sort(sample(rows, ROWS_PER_CLASS))
  }), use.names = FALSE)
  unknown <- which(markers == UNKNOWN)
  incomplete <- which(!complete.cases(exprs(x)))
  pool <- intersect(unknown, incomplete)
  if (length(pool) < N_UNKNOWN) pool <- unknown
  keep <- c(keep, sample(pool, N_UNKNOWN))
  sort(unique(keep))
}

for (name in names(datasets)) {
  data(list = name, envir = environment())
  full <- get(name)
  idx <- pick_rows(full)
  mini <- full[idx, ]
  ## `[` logs "Subset [n,m][k,m] <date>" into processingData; put the original log back so the
  ## slice carries exactly the provenance pRolocdata shipped (itzhak2016stcSILAC's is empty,
  ## and the reader must then leave `uns["processing"]` out rather than store an empty list).
  mini@processingData@processing <- full@processingData@processing
  stopifnot(validObject(mini), nrow(mini) == length(idx))
  assign(name, mini)
  out <- sprintf("grassp/tests/fixtures/mini_%s.%s", name, datasets[[name]])
  save(list = name, file = out, compress = "xz")
  cat(sprintf("wrote %-45s %3d x %2d  (%5.1f kB)  NA in exprs: %d\n", out, nrow(mini), ncol(mini),
              file.size(out) / 1000, sum(is.na(exprs(mini)))))
}
