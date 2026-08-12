## The MSnSet <-> h5ad conversion itself.
##
## The mapping, in one line each:
##
##   exprs()                         <-> X                 (features x fractions; NO transpose)
##   featureNames()                  <-> obs_names
##   sampleNames()                   <-> var_names
##   fData() scalar columns          <-> obs columns
##   fData() matrix columns          <-> obsm data frames
##   pData() scalar columns          <-> var columns
##   pData() matrix columns          <-> varm data frames
##   extra assayData elements        <-> layers
##   processingData()@processing     <-> uns$processing
##   experimentData@other$grassp_uns <-> everything else in uns
##   "unknown"                       <-> NA
##
## There is no exchange format and no version block: the artifact is an ordinary h5ad. grassp
## writes one with `adata.write_h5ad()` and reads one with `anndata.read_h5ad()`, and everything
## pRoloc-specific happens here, on pRoloc's side of the boundary. So the objects have to
## describe themselves, which is what the two data-frame rows are about: pRoloc stores
## per-protein by per-compartment score matrices *inside a single fData column*
## (svm.all.scores, tagm.map.joint, bandle.joint, Markers, GOAnnotations), and writing those as
## obsm/varm **data frames** rather than bare arrays keeps the class names attached to the data
## instead of in a side table that can fall out of step with it.
##
## The `experimentData@other` row is what makes the trip lossless rather than merely usable: an
## MSnSet has no slot for arbitrary metadata, so unmapped uns entries are parked there and
## handed straight back on the way out. Only .obsp/.varp cannot cross at all -- eSet has no
## pairwise slot.

## The literal string pRoloc uses for unlabelled features.
.UNKNOWN <- "unknown"

## Suffixes grassp's own annotators append to a label column's name when they store the matching
## probability matrix. Stripping one recovers the label column, whose levels name the columns.
.MATRIX_SUFFIXES <- c("_probabilities", ".probabilities", "_one_hot_labels")


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

#' Read a grassp h5ad file as an MSnSet
#'
#' Converts an h5ad written by grassp -- with plain `adata.write_h5ad()`; there is nothing
#' special about the file -- into an MSnbase [MSnbase::MSnSet-class] object ready for pRoloc.
#'
#' Matrix-valued feature metadata columns are rebuilt from `obsm`, and matrix-valued `pData`
#' columns from `varm`; `pData` is the same `AnnotatedDataFrame` class, so both work the same
#' way. An entry that arrives as a data frame -- which is what [grassp_write_msnset()] emits --
#' names its own columns. For a bare array, the class names are looked for in
#' `uns[["<key>_categories"]]` (the convention grassp's annotators write) and then in the levels
#' of the companion label column; failing both, the column stays nameless, which is the right
#' answer for an embedding and is what sends it back out as a plain array.
#'
#' AnnData layers become additional `assayData` elements, reachable with
#' `assayDataElementNames(x)` and `assayDataElement(x, "pvals")`; `exprs()` remains the matrix
#' pRoloc's functions operate on.
#'
#' Whatever is left in `uns` -- neighbour parameters, PCA variance ratios, colour maps -- is
#' parked on `experimentData(x)@other$grassp_uns`, so that [grassp_write_msnset()] can hand it
#' back and a grassp -> pRoloc -> grassp trip keeps it. pRoloc never looks there, and it
#' survives subsetting, `markerMSnSet()` and the classifiers.
#'
#' No column is treated as *the* marker column. Which one a function uses is its `fcol`
#' argument, so inspect `fvarLabels(x)` and pick per call.
#'
#' @param path Path to a `.h5ad` file.
#' @param nan_to_unknown Fill `NA` in text `fData` columns with pRoloc's `"unknown"` sentinel.
#'   pRoloc needs it: `markerMSnSet()` and `unknownMSnSet()` fail outright on `NA`, a
#'   classifier's training set is chosen with `fData(object)[, fcol] != "unknown"`, and
#'   `plot2D()` colours it as its own class. grassp uses `NA`, so this is the one value
#'   translation the bridge makes -- and it belongs here, because the sentinel is pRoloc's
#'   convention. It applies to every text column rather than to a nominated marker column: `fcol`
#'   is a per-call argument, so one `MSnSet` can carry `markers`, `markers.orig`, `pd.markers`
#'   and more at once. [grassp_write_msnset()] reverses it.
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
  .check_feature_names(obs_names)

  exprs_matrix <- .as_double_matrix(adata$X, "X")
  if (is.null(exprs_matrix)) {
    stop(
      "This h5ad has no X, so there is no quantitation to become exprs(). ",
      "If the matrix you want is a layer, make it .X in Python first.",
      call. = FALSE
    )
  }
  dimnames(exprs_matrix) <- list(obs_names, var_names)
  .report_exprs(exprs_matrix)

  ## Column names cross verbatim in both directions. pRoloc tolerates non-syntactic fData and
  ## pData names, so there is nothing to translate and no name map to keep in step.
  fdata <- as.data.frame(adata$obs, stringsAsFactors = FALSE)
  rownames(fdata) <- obs_names

  pdata <- as.data.frame(adata$var, stringsAsFactors = FALSE)
  rownames(pdata) <- var_names

  ## Matrix-valued fData columns rebuilt from obsm, and pData ones from varm. pData is the same
  ## AnnotatedDataFrame class, so the mechanism is identical -- only the axis differs.
  ##
  ## Before the sentinel is filled in, not after: a nameless matrix takes its class names from the
  ## levels of its companion label column, matched on width, and filling NA with "unknown" adds a
  ## level. A curated portal dataset is exactly that case --
  ## harmonized_annotation_propagated_probabilities is 16 columns wide and its label column has 16
  ## categories and some NA -- so doing this the other way round loses the names to an off-by-one.
  fdata <- .attach_matrix_columns(fdata, adata$obsm, uns, obs_names, "obsm")
  pdata <- .attach_matrix_columns(pdata, adata$varm, uns, var_names, "varm")

  if (nan_to_unknown) {
    ## Only text columns are touched, so the matrix columns just attached are left alone.
    fdata <- .fill_unknown(fdata)
  }

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
    m <- .as_double_matrix(layers[[nm]], nm)
    if (is.null(m)) {
      message("Skipping layer '", nm, "': not numeric, so it cannot become an assayData element.")
      next
    }
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
  ## neighbour parameters, the PCA variance ratios or a colour map just because an MSnSet has no
  ## slot for them. `other` is a free-form list on MIAxE, and this is safe here for the reason it
  ## would *not* be for a graph: uns is neither row- nor column-aligned, so there is nothing for
  ## subsetting to get wrong. Verified: survives `[`, markerMSnSet() and svmClassification(),
  ## and passes validObject().
  msnset@experimentData@other$grassp_uns <- uns[setdiff(names(uns), "processing")]

  problems <- methods::validObject(msnset, test = TRUE)
  if (!isTRUE(problems)) {
    stop(
      "The reconstructed MSnSet is invalid: ", paste(problems, collapse = "; "),
      call. = FALSE
    )
  }
  msnset
}

## Reject index values R cannot use as rownames. Python only warns about duplicate obs_names, so
## an object can reach here with them; an MSnSet cannot hold them at all.
.check_feature_names <- function(names) {
  blank <- !nzchar(trimws(names)) | tolower(names) %in% c("na", "nan", "none")
  if (any(blank)) {
    stop(
      sum(blank), " feature names are blank or NA-like, which R cannot use as rownames: ",
      paste(utils::head(names[blank], 5), collapse = ", "),
      ". Filter or rename those proteins in Python first.",
      call. = FALSE
    )
  }
  if (anyDuplicated(names)) {
    duplicated <- unique(names[duplicated(names)])
    stop(
      "Feature names must be unique to become rownames, but ", length(duplicated),
      " are duplicated: ", paste(utils::head(duplicated, 5), collapse = ", "),
      ". Deduplicate in Python with `adata.obs_names_make_unique()`, or aggregate first.",
      call. = FALSE
    )
  }
  invisible(TRUE)
}

## A numeric matrix from whatever anndataR hands back -- a dense matrix, the dgCMatrix a Python
## sparse write produces, or a data frame -- or NULL when the value cannot become one.
.as_double_matrix <- function(value, what) {
  if (is.null(value)) {
    return(NULL)
  }
  if (is.data.frame(value)) {
    if (!all(vapply(value, function(column) is.numeric(column) || is.logical(column), logical(1)))) {
      return(NULL)
    }
  } else if (!(is.numeric(value) || is.logical(value) || methods::is(value, "Matrix"))) {
    return(NULL)
  }
  m <- as.matrix(value)
  storage.mode(m) <- "double"
  m
}

## Advisory only: two properties of the quantitation that bite later, on pRoloc's side.
.report_exprs <- function(m) {
  n_na <- sum(is.na(m))
  if (n_na > 0) {
    message(
      "exprs() has ", n_na, " missing values. Several pRoloc methods need complete profiles; ",
      "see filterNA(), or impute in grassp with gr.pp.impute_knn()."
    )
  }
  sums <- rowSums(m, na.rm = TRUE)
  off <- abs(sums - 1) > 1e-3
  if (any(off)) {
    message(
      sum(off), " of ", length(sums), " profiles do not sum to 1 (observed range ",
      signif(min(sums), 3), "-", signif(max(sums), 3), "). pRoloc's distance-based methods and ",
      "its plots assume sum-normalised profiles; see normalise() or gr.pp.normalize_total()."
    )
  }
}

## Fill pRoloc's "unknown" sentinel into every text column that uses NA. A factor gains the level
## rather than being flattened to character, so its level order -- and its ordered-ness -- survive.
.fill_unknown <- function(df) {
  for (column in colnames(df)) {
    values <- df[[column]]
    if (!anyNA(values)) {
      next
    }
    if (is.factor(values)) {
      if (!(.UNKNOWN %in% levels(values))) {
        levels(values) <- c(levels(values), .UNKNOWN)
      }
      values[is.na(values)] <- .UNKNOWN
      df[[column]] <- values
    } else if (is.character(values)) {
      values[is.na(values)] <- .UNKNOWN
      df[[column]] <- values
    }
  }
  df
}

## Rebuild matrix-valued columns on an AnnotatedDataFrame's data frame. Used for both axes:
## obsm -> fData columns, varm -> pData columns.
.attach_matrix_columns <- function(df, matrices, uns, row_names, slot) {
  for (key in names(matrices)) {
    m <- .as_double_matrix(matrices[[key]], key)
    if (is.null(m)) {
      message(
        "Skipping ", slot, "$", key, ": not numeric, so it cannot become a matrix ",
        if (identical(slot, "obsm")) "fData" else "pData", " column."
      )
      next
    }
    dimnames(m) <- list(row_names, .matrix_colnames(matrices[[key]], key, ncol(m), uns, df))
    df[[key]] <- m
  }
  df
}

## Where the class names of a matrix column come from, in order of authority:
##
##   1. the entry's own column names -- a data frame, which is what grassp_write_msnset() emits;
##   2. uns[["<key>_categories"]], the convention grassp's own annotators write, including after
##      stripping the suffix they append to the label column's name;
##   3. the levels of that companion label column, which is where the names actually live for a
##      curated portal dataset: harmonized_annotation_propagated_probabilities has no
##      _categories entry, but obs["harmonized_annotation_propagated"] is a factor of exactly
##      the right width.
##
## NULL when none of the three has them, rather than an invented V1..Vn. Most nameless obsm
## entries are embeddings -- X_pca, X_umap -- whose columns have no names to recover, and staying
## nameless is what lets the writer send them back out as the plain arrays they arrived as.
.matrix_colnames <- function(value, key, width, uns, df) {
  own <- colnames(value)
  if (!is.null(own) && length(own) == width) {
    return(as.character(own))
  }
  for (stem in .stems(key)) {
    declared <- .as_character_vector(uns[[paste0(stem, "_categories")]])
    if (length(declared) == width) {
      return(declared)
    }
  }
  for (stem in .stems(key)) {
    companion <- df[[stem]]
    if (is.factor(companion) && length(levels(companion)) == width) {
      return(levels(companion))
    }
  }
  NULL
}

.stems <- function(key) {
  hits <- .MATRIX_SUFFIXES[endsWith(key, .MATRIX_SUFFIXES)]
  c(key, substr(rep(key, length(hits)), 1L, nchar(key) - nchar(hits)))
}


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

#' Write an MSnSet to an h5ad file grassp can read
#'
#' The reverse of [grassp_as_msnset()]. Splits `fData()` into its scalar columns, which become
#' `obs`, and its matrix-valued columns, which become `obsm` data frames carrying their own
#' class names; `pData()` the same way into `var` and `varm`. A matrix column with no column
#' names -- an embedding, typically -- goes out as a plain array instead, since there is nothing
#' to carry and Python expects an embedding to be one. Extra `assayData` elements become AnnData
#' layers. The result is an ordinary h5ad, opened in Python with `anndata.read_h5ad()` -- there
#' is no exchange format to keep in step.
#'
#' If `x` came from [grassp_as_msnset()], anything that was in the original `uns` and has no
#' `MSnSet` slot was parked on `experimentData(x)@other$grassp_uns` and is written back here, so
#' the round trip keeps it.
#'
#' Useful even if you never touch grassp: pRoloc and MSnbase have no exporter, so this is a
#' way to hand a classified `MSnSet` to any tool that reads h5ad.
#'
#' @param x An [MSnbase::MSnSet-class] object.
#' @param path Destination `.h5ad` path.
#' @param columns Optional character vector restricting which `fData` columns are written.
#'   `NULL` (default) writes all of them.
#' @param drop Optional character vector of `fData` columns to omit.
#' @param unknown_to_na Convert pRoloc's `"unknown"` sentinel back to `NA`. Defaults to `TRUE`,
#'   because grassp encodes unlabelled features as `NA` and its annotators select markers with
#'   `.notna()` -- an untranslated `"unknown"` there becomes a spurious compartment class. This
#'   package owns the conversion in both directions, since `"unknown"` is pRoloc's own
#'   convention. Set `FALSE` to keep the artifact faithful to the `MSnSet`.
#' @param overwrite Replace `path` if it already exists. Defaults to `FALSE`.
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
  ## losing the matrix structure this mapping depends on.
  fsplit <- .split_matrix_columns(Biobase::fData(x))
  psplit <- .split_matrix_columns(Biobase::pData(x))

  scalar <- fsplit$scalar
  keep <- colnames(scalar)
  if (!is.null(columns)) {
    keep <- intersect(keep, columns)
  }
  if (!is.null(drop)) {
    keep <- setdiff(keep, drop)
  }
  scalar <- scalar[, keep, drop = FALSE]

  obs <- .normalise_labels(scalar, unknown_to_na)
  var <- .normalise_labels(psplit$scalar, unknown_to_na)

  exprs_matrix <- Biobase::exprs(x)
  storage.mode(exprs_matrix) <- "double"

  ## Every assayData element other than exprs becomes an AnnData layer.
  layers <- list()
  for (nm in setdiff(Biobase::assayDataElementNames(x), "exprs")) {
    m <- as.matrix(Biobase::assayDataElement(x, nm))
    storage.mode(m) <- "double"
    ## Alignment is positional and the reader restores dimnames from obs_names/var_names, while
    ## anndataR >= 1.2 warns once per matrix about dimnames it is about to discard.
    dimnames(m) <- NULL
    layers[[nm]] <- m
  }

  ## Anything grassp_as_msnset() parked on experimentData@other rides back out, so a
  ## grassp -> pRoloc -> grassp trip keeps the whole of .uns. `processing` is regenerated from
  ## this object rather than carried, because it is a real MSnSet slot and has grown since.
  carried <- Biobase::experimentData(x)@other$grassp_uns
  uns <- if (is.null(carried)) list() else carried[setdiff(names(carried), "processing")]
  uns$processing <- as.character(MSnbase::processingData(x)@processing)

  ## A class name containing "/" cannot be a data-frame column: HDF5 reads it as a path, and
  ## rhdf5 will not create the intermediate group, so the entry is silently unwritable. pRoloc
  ## does produce such names (hyperLOPIT's "Endoplasmic reticulum/Golgi apparatus"), so those
  ## matrices fall back to a bare array plus uns[["<key>_categories"]] -- grassp's own
  ## convention, which grassp_as_msnset() and grassp's annotators both already read. Nothing is
  ## renamed either way.
  obsm <- .demote_unwritable(fsplit$matrices, uns, "obsm")
  varm <- .demote_unwritable(psplit$matrices, obsm$uns, "varm")

  adata <- anndataR::AnnData(
    X = exprs_matrix,
    obs = obs,
    var = var,
    obsm = obsm$matrices,
    varm = varm$matrices,
    layers = layers,
    uns = varm$uns
  )
  ## anndataR defaults to mode "w-", which refuses to touch an existing file; that would make
  ## any re-run of a user's script fail on its second invocation.
  anndataR::write_h5ad(adata, path, mode = if (overwrite) "w" else "w-")
  message(
    "Wrote ", path, " (", nrow(obs), " features x ", ncol(exprs_matrix),
    " fractions; ", length(obsm$matrices), " matrix column(s); ",
    length(layers), " extra assay element(s)).\n",
    "Back in Python:\n",
    "  adata = anndata.read_h5ad(\"", basename(path), "\")"
  )
  invisible(path)
}

## Split a data frame into its scalar columns and its matrix columns, the latter as data frames
## bound for obsm/varm. Must run BEFORE any as.data.frame() call; see grassp_write_msnset().
.split_matrix_columns <- function(df) {
  is_matrix_column <- vapply(seq_len(ncol(df)), function(i) is.matrix(df[[i]]), logical(1))
  matrices <- list()
  for (nm in colnames(df)[is_matrix_column]) {
    m <- as.matrix(df[[nm]])
    storage.mode(m) <- "double"
    if (is.null(colnames(m))) {
      ## Nothing to carry, so send it as it came. This is the embedding case -- X_pca, X_umap --
      ## where a data frame would hand Python a frame of V1..Vn instead of the array everything
      ## downstream, scanpy included, expects an embedding to be.
      dimnames(m) <- NULL
      matrices[[nm]] <- m
    } else {
      ## A data frame rather than a bare matrix, because its column names are pRoloc's class
      ## names and this is what carries them to Python attached to the data. check.names = FALSE:
      ## those names are not syntactic -- svmClassification decorates them as
      ## "<class>.svm.scores", and a class can be "ER lumen". Row names are left to anndataR,
      ## which writes the parent object's featureNames/sampleNames as the stored index.
      matrices[[nm]] <- as.data.frame(m, check.names = FALSE)
    }
  }
  list(
    scalar = as.data.frame(df[, !is_matrix_column, drop = FALSE], stringsAsFactors = FALSE),
    matrices = matrices
  )
}

## Convert pRoloc's "unknown" sentinel to NA, which is how grassp encodes unlabelled features,
## and keep every text column in a dtype whose NA survives the trip.
##
## That second part is not cosmetic. anndataR writes a character NA as the literal two-character
## string "NA", which arrives in Python as a compartment called "NA" -- silent corruption -- and
## it has no nullable-string encoding to write instead. A factor's NA round-trips correctly, as
## an h5ad categorical with a -1 code, so any character column carrying NA becomes one.
##
## Factors are never flattened to character: that would cost the level order (and the ordered
## flag) on the way through, and Python would get an object-dtype column instead of a Categorical.
.normalise_labels <- function(df, unknown_to_na) {
  for (column in colnames(df)) {
    values <- df[[column]]
    if (is.factor(values)) {
      if (unknown_to_na && .UNKNOWN %in% levels(values)) {
        ## Setting one level to NA drops just that level, keeping the order of the rest -- unlike
        ## droplevels(), which would also quietly drop a compartment that has no members.
        lv <- levels(values)
        lv[lv == .UNKNOWN] <- NA
        levels(values) <- lv
      }
    } else if (is.character(values)) {
      if (unknown_to_na) {
        values[values == .UNKNOWN] <- NA_character_
      }
      if (anyNA(values)) {
        values <- factor(values)
      }
    }
    df[[column]] <- values
  }
  df
}

## Any matrix whose class names HDF5 cannot use as dataset names is written as a bare array, with
## its names moved to uns[["<key>_categories"]]. Returns both, since uns grows.
.demote_unwritable <- function(matrices, uns, slot) {
  for (key in names(matrices)) {
    names_ <- colnames(matrices[[key]])
    if (!any(grepl("/", names_, fixed = TRUE))) {
      next
    }
    message(
      slot, "$", key, " has class names containing \"/\", which cannot be HDF5 dataset names; ",
      "writing it as a plain array with its names in uns[[\"", key, "_categories\"]]."
    )
    m <- as.matrix(matrices[[key]])
    dimnames(m) <- NULL
    matrices[[key]] <- m
    uns[[paste0(key, "_categories")]] <- names_
  }
  list(matrices = matrices, uns = uns)
}
