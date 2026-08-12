## The MSnSet <-> h5ad conversion itself.
##
## The contract, in one line each:
##
##   exprs()                    <-> X                (features x fractions; NO transpose)
##   featureNames()             <-> obs_names
##   sampleNames()              <-> var_names
##   fData() scalar columns     <-> obs columns
##   fData() matrix columns     <-> obsm + uns$obsm_colnames
##   pData() scalar columns     <-> var columns
##   pData() matrix columns     <-> varm + uns$varm_colnames
##   extra assayData elements   <-> layers
##   processingData()@processing<-> uns$processing
##   experimentData@other$grassp_uns <-> everything else in uns
##   "unknown"                  <-> NA
##
## The matrix-column rows are the ones that carry the weight: pRoloc stores per-protein by
## per-compartment score matrices *inside a single fData column* (svm.all.scores,
## tagm.map.joint, bandle.joint, Markers, GOAnnotations), and that is exactly what obsm is
## for. AnnData's obsm has no column names, so the class names are carried separately in
## uns$obsm_colnames. pData is the same AnnotatedDataFrame class, so varm works identically on
## the sample axis.
##
## The `experimentData@other` row is what makes the trip lossless rather than merely usable: an
## MSnSet has no slot for arbitrary metadata, so unmapped uns entries are parked there and
## handed straight back on the way out. Only .obsp/.varp cannot cross at all -- eSet has no
## pairwise slot.

.as_character_vector <- function(x) {
  if (is.null(x)) {
    return(character(0))
  }
  as.character(unlist(x, use.names = FALSE))
}

#' Read a grassp h5ad file as an MSnSet
#'
#' Converts an h5ad written by grassp's `gr.io.write_msnset()` into an MSnbase
#' [MSnbase::MSnSet-class] object ready for pRoloc. The artifact is self-describing, so no
#' arguments are normally needed.
#'
#' Matrix-valued feature metadata columns are rebuilt from `obsm` using the class names in
#' `uns$obsm_colnames`, and matrix-valued `pData` columns from `varm` -- `pData` is the same
#' `AnnotatedDataFrame` class, so both work the same way. AnnData layers become additional
#' `assayData` elements, reachable with `assayDataElementNames(x)` and
#' `assayDataElement(x, "pvals")`; `exprs()` remains the matrix pRoloc's functions operate on.
#'
#' Whatever is left in `uns` -- neighbour parameters, PCA variance ratios, colour maps, schema
#' versions -- is parked on `experimentData(x)@other$grassp_uns`, so that
#' [grassp_write_msnset()] can hand it back and a grassp -> pRoloc -> grassp trip keeps it.
#' pRoloc never looks there, and it survives subsetting, `markerMSnSet()` and the classifiers.
#'
#' No column is treated as *the* marker column. Which one a function uses is its `fcol`
#' argument, so inspect `fvarLabels(x)` and pick per call.
#'
#' @param path Path to a `.h5ad` file.
#' @param nan_to_unknown Convert any remaining `NA` in text columns to pRoloc's `"unknown"`
#'   sentinel. grassp normally does this on export, so this is a safety net for artifacts
#'   written with `nan_to_unknown = FALSE` or produced by another tool. Note that it applies to
#'   every text column, not to a nominated marker column: `fcol` is a per-call argument in
#'   pRoloc, so one `MSnSet` can carry `markers`, `markers.orig`, `pd.markers` and more at once.
#'
#' @return An [MSnbase::MSnSet-class] object with proteins as features and fractions as
#'   samples.
#'
#' @seealso [grassp_write_msnset()] for the reverse direction.
#'
#' @examples
#' \dontrun{
#' x <- grassp_as_msnset("experiment.h5ad")
#' getMarkerClasses(x, fcol = "markers")
#' x <- pRoloc::svmClassification(x, fcol = "markers", scores = "all")
#' grassp_write_msnset(x, "results.h5ad")
#' }
#' @export
grassp_as_msnset <- function(path, nan_to_unknown = TRUE) {
  .require_rhdf5()
  if (!file.exists(path)) {
    stop("No such file: ", path, call. = FALSE)
  }
  adata <- anndataR::read_h5ad(path)
  ## Bind each slot once -- these are R6 active bindings, not plain fields.
  uns <- adata$uns
  obs_names <- as.character(adata$obs_names)
  var_names <- as.character(adata$var_names)
  layers <- adata$layers
  .check_spec(uns$msnset_spec)

  exprs_matrix <- as.matrix(adata$X)
  storage.mode(exprs_matrix) <- "double"
  rownames(exprs_matrix) <- obs_names
  colnames(exprs_matrix) <- var_names

  ## Column names cross verbatim in both directions. pRoloc tolerates non-syntactic fData and
  ## pData names, so there is nothing to translate and no name map to keep in step.
  fdata <- as.data.frame(adata$obs, stringsAsFactors = FALSE)
  rownames(fdata) <- obs_names

  pdata <- as.data.frame(adata$var, stringsAsFactors = FALSE)
  rownames(pdata) <- var_names

  ## grassp converts NaN to "unknown" on export, so normally there is nothing to do here.
  ## This is a safety net for an artifact written with nan_to_unknown = FALSE, or one produced
  ## by some other tool: pRoloc needs the sentinel, since markerMSnSet() and unknownMSnSet()
  ## fail outright on NA. It applies to every text column rather than to a nominated one,
  ## because an MSnSet legitimately carries several marker sets at once.
  if (nan_to_unknown) {
    for (column in colnames(fdata)) {
      values <- fdata[[column]]
      if ((is.character(values) || is.factor(values)) && anyNA(values)) {
        fdata[[column]] <- .to_unknown(values)
      }
    }
  }

  ## Matrix-valued fData columns rebuilt from obsm, and pData ones from varm. pData is the same
  ## AnnotatedDataFrame class, so the mechanism is identical -- only the axis differs.
  fdata <- .attach_matrix_columns(fdata, adata$obsm, uns[[.OBSM_COLNAMES]], obs_names)
  pdata <- .attach_matrix_columns(pdata, adata$varm, uns[[.VARM_COLNAMES]], var_names)

  msnset <- methods::new(
    "MSnSet",
    exprs = exprs_matrix,
    featureData = methods::new("AnnotatedDataFrame", data = fdata),
    phenoData = methods::new("AnnotatedDataFrame", data = pdata)
  )

  ## AnnData layers become additional assayData elements. assayData is a Biobase environment
  ## holding any number of equal-dimension matrices, so these behave like exprs: pRoloc's
  ## functions leave them alone, and `[`, markerMSnSet() and filterNA() subset them along with
  ## it -- which is why this is safe in a way a positional side table would not be.
  for (nm in names(layers)) {
    m <- as.matrix(layers[[nm]])
    storage.mode(m) <- "double"
    dimnames(m) <- dimnames(exprs_matrix)
    Biobase::assayDataElement(msnset, nm) <- m
  }
  processing <- .as_character_vector(uns$processing)
  msnset@processingData@processing <- c(
    processing,
    paste0("Imported from grassp h5ad [", basename(path), "]: ", date())
  )

  ## Everything else in uns rides along on experimentData@other, so that
  ## grassp_write_msnset() can hand it back unchanged and the round trip does not lose the
  ## neighbour parameters, the PCA variance ratios or a schema version just because an MSnSet
  ## has no slot for them. `other` is a free-form list on MIAxE, and this is safe here for the
  ## reason it would *not* be for a graph: uns is neither row- nor column-aligned, so there is
  ## nothing for subsetting to get wrong. Verified: survives `[`, markerMSnSet() and
  ## svmClassification(), and passes validObject().
  msnset@experimentData@other$grassp_uns <- uns[setdiff(names(uns), .RESERVED_UNS)]

  problems <- methods::validObject(msnset, test = TRUE)
  if (!isTRUE(problems)) {
    stop(
      "The reconstructed MSnSet is invalid: ", paste(problems, collapse = "; "),
      call. = FALSE
    )
  }
  msnset
}

## Rebuild matrix-valued columns on an AnnotatedDataFrame's data frame. Used for both axes:
## obsm -> fData columns, varm -> pData columns.
.attach_matrix_columns <- function(df, matrices, colnames_map, row_names) {
  for (key in names(matrices)) {
    m <- as.matrix(matrices[[key]])
    storage.mode(m) <- "double"
    rownames(m) <- row_names
    declared <- colnames_map[[key]]
    if (!is.null(declared) && length(declared) == ncol(m)) {
      colnames(m) <- .as_character_vector(declared)
    }
    df[[key]] <- m
  }
  df
}

## Split a data frame into its scalar columns and its matrix columns. Must run BEFORE any
## as.data.frame() call: coercing a DFrame that holds a matrix column silently flattens it into
## dot-pasted scalars, which is how pRoloc's classifiers hand back svm.all.scores.
.split_matrix_columns <- function(df) {
  is_matrix_column <- vapply(seq_len(ncol(df)), function(i) is.matrix(df[[i]]), logical(1))
  matrices <- list()
  colnames_map <- list()
  for (nm in colnames(df)[is_matrix_column]) {
    m <- as.matrix(df[[nm]])
    storage.mode(m) <- "double"
    colnames_map[[nm]] <- if (is.null(colnames(m))) {
      paste0("V", seq_len(ncol(m)))
    } else {
      colnames(m)
    }
    ## Drop the dimnames now that the colnames are safely in colnames_map. obsm/varm arrays
    ## carry no column names in the AnnData model, which is the whole reason the contract puts
    ## them in uns -- and anndataR >= 1.2 warns once per matrix if you hand it dimnames it is
    ## about to discard. Alignment is positional, and the reader restores rownames from
    ## obs_names/var_names.
    dimnames(m) <- NULL
    matrices[[nm]] <- m
  }
  list(
    scalar = as.data.frame(df[, !is_matrix_column, drop = FALSE], stringsAsFactors = FALSE),
    matrices = matrices,
    colnames = colnames_map
  )
}

.to_unknown <- function(values) {
  out <- as.character(values)
  out[is.na(out) | !nzchar(trimws(out))] <- "unknown"
  out
}

.to_na <- function(values) {
  out <- as.character(values)
  out[out == "unknown"] <- NA_character_
  out
}

#' Write an MSnSet to a grassp h5ad file
#'
#' The reverse of [grassp_as_msnset()]. Splits `fData()` into its scalar columns, which become
#' `obs`, and its matrix-valued columns, which become `obsm` entries with their class names
#' recorded in `uns$obsm_colnames`; `pData()` the same way into `var` and `varm`. Extra
#' `assayData` elements become AnnData layers.
#'
#' If `x` came from [grassp_as_msnset()], anything that was in the original `uns` and has no
#' `MSnSet` slot was parked on `experimentData(x)@other$grassp_uns` and is written back here, so
#' the round trip keeps it. The contract's own `uns` keys are regenerated from `x` rather than
#' carried, so they describe what is actually being written even after subsetting in R.
#'
#' Useful even if you never touch grassp: pRoloc and MSnbase have no exporter, so this is a
#' way to hand a classified `MSnSet` to any tool that reads h5ad.
#'
#' @param x An [MSnbase::MSnSet-class] object.
#' @param path Destination `.h5ad` path.
#' @param columns Optional character vector restricting which `fData` columns are written.
#'   `NULL` (default) writes all of them.
#' @param drop Optional character vector of `fData` columns to omit.
#' @param unknown_to_na Convert pRoloc's `"unknown"` sentinel to `NA`. Defaults to `TRUE`,
#'   because grassp encodes unlabelled features as `NA` and its annotators select markers with
#'   `.notna()` -- an untranslated `"unknown"` there becomes a spurious compartment class.
#'   This package owns the conversion in both directions, since `"unknown"` is pRoloc's own
#'   convention. Set `FALSE` to keep the artifact byte-faithful to the `MSnSet`.
#' @param overwrite Replace `path` if it already exists. Defaults to `FALSE`, matching
#'   grassp's `write_msnset()`.
#'
#' @return The path written, invisibly.
#'
#' @examples
#' \dontrun{
#' x <- grassp_as_msnset("experiment.h5ad")
#' x <- pRoloc::tagmMapPredict(x, params = pRoloc::tagmMapTrain(x), probJoint = TRUE)
#' grassp_write_msnset(x, "results.h5ad")
#' }
#' @export
grassp_write_msnset <- function(x,
                                path,
                                columns = NULL,
                                drop = NULL,
                                unknown_to_na = TRUE,
                                overwrite = FALSE) {
  if (!methods::is(x, "MSnSet")) {
    stop("`x` must be an MSnSet, not ", class(x)[[1]], ".", call. = FALSE)
  }
  .require_rhdf5()
  if (file.exists(path) && !overwrite) {
    stop(path, " already exists. Pass overwrite = TRUE to replace it.", call. = FALSE)
  }

  ## Split matrix columns out of both axes -- fData -> obsm, pData -> varm. This must happen
  ## BEFORE any as.data.frame() call: pRoloc's classifiers can leave fData as an S4 DFrame, and
  ## coercing one that holds a matrix column silently flattens it into dot-pasted scalars,
  ## losing the matrix structure this contract depends on.
  fsplit <- .split_matrix_columns(Biobase::fData(x))
  psplit <- .split_matrix_columns(Biobase::pData(x))
  obsm <- fsplit$matrices
  obsm_colnames <- fsplit$colnames
  varm <- psplit$matrices
  varm_colnames <- psplit$colnames

  scalar <- fsplit$scalar
  keep <- colnames(scalar)
  if (!is.null(columns)) {
    keep <- intersect(keep, columns)
  }
  if (!is.null(drop)) {
    keep <- setdiff(keep, drop)
  }
  scalar <- scalar[, keep, drop = FALSE]

  ## Convert pRoloc's "unknown" sentinel to NA, which is how grassp encodes unlabelled
  ## features, and then make any column that carries NA a factor.
  ##
  ## That second step is not cosmetic. anndataR writes a character NA as the literal
  ## two-character string "NA", which arrives in Python as a compartment called "NA" -- silent
  ## corruption. A factor's NA round-trips correctly, as an h5ad categorical with a -1 code.
  ## This applies to any character column with missing values, not just the marker column.
  for (column in colnames(scalar)) {
    values <- scalar[[column]]
    if (is.factor(values)) {
      values <- as.character(values)
    }
    if (is.character(values)) {
      if (unknown_to_na) {
        values <- .to_na(values)
      }
      if (anyNA(values)) {
        values <- factor(values, levels = sort(unique(values[!is.na(values)])))
      }
    }
    scalar[[column]] <- values
  }

  pdata <- psplit$scalar

  exprs_matrix <- Biobase::exprs(x)
  storage.mode(exprs_matrix) <- "double"

  ## Every assayData element other than exprs becomes an AnnData layer.
  layers <- list()
  for (nm in setdiff(Biobase::assayDataElementNames(x), "exprs")) {
    m <- as.matrix(Biobase::assayDataElement(x, nm))
    storage.mode(m) <- "double"
    dimnames(m) <- NULL # see .split_matrix_columns; anndataR >= 1.2 warns about these
    layers[[nm]] <- m
  }

  ## Anything grassp_as_msnset() parked on experimentData@other rides back out, so a
  ## grassp -> pRoloc -> grassp trip keeps the whole of .uns. The contract's own keys are
  ## regenerated from this object rather than carried, so they always describe what is actually
  ## being written -- which matters after any subsetting in R.
  carried <- Biobase::experimentData(x)@other$grassp_uns
  uns <- if (is.null(carried)) list() else carried[setdiff(names(carried), .RESERVED_UNS)]
  uns$msnset_spec <- GRASSP_MSNSET_SPEC
  uns$msnset_exprs_layer <- ""
  uns$msnset_dropped <- character(0)
  uns$processing <- as.character(MSnbase::processingData(x)@processing)
  uns[[.OBSM_COLNAMES]] <- obsm_colnames
  uns[[.VARM_COLNAMES]] <- varm_colnames

  adata <- anndataR::AnnData(
    X = exprs_matrix,
    obs = scalar,
    var = pdata,
    obsm = obsm,
    varm = varm,
    layers = layers,
    uns = uns
  )
  ## anndataR defaults to mode "w-", which refuses to touch an existing file; that would make
  ## any re-run of a user's script fail on its second invocation.
  anndataR::write_h5ad(adata, path, mode = if (overwrite) "w" else "w-")
  message(
    "Wrote ", path, " (", nrow(scalar), " features x ", ncol(exprs_matrix),
    " fractions; ", length(obsm), " matrix column(s); ",
    length(layers), " extra assay element(s)).\n",
    "Back in Python:\n",
    "  adata = gr.io.read_msnset(\"", basename(path), "\")"
  )
  invisible(path)
}
