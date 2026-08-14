## Tests for the R half of the grassp <-> pRoloc bridge.
##
## anndataR is a hard dependency, so there is no backend to skip on. The h5ad round trip does
## still need rhdf5, which anndataR calls but declares only in Suggests, so the tests that touch
## a file skip without it.
##
## The load-bearing ones are the dtype tests. Everything here crosses as an ordinary h5ad, so a
## mistake does not raise -- it arrives in Python as an object-dtype column, a re-sorted level
## set, or a compartment called "NA".

make_msnset <- function(n_features = 8, n_fractions = 5, n_classes = 3) {
  set.seed(42)
  exprs_matrix <- matrix(
    stats::runif(n_features * n_fractions),
    nrow = n_features,
    dimnames = list(
      sprintf("P%03d", seq_len(n_features)),
      sprintf("Fraction.%d", seq_len(n_fractions))
    )
  )
  exprs_matrix <- exprs_matrix / rowSums(exprs_matrix)

  classes <- paste("Compartment", seq_len(n_classes))
  markers <- rep("unknown", n_features)
  markers[seq_len(n_classes)] <- classes

  fdata <- data.frame(
    markers = markers,
    svm.scores = seq(0.5, 0.99, length.out = n_features),
    row.names = rownames(exprs_matrix),
    stringsAsFactors = FALSE
  )
  # A matrix-valued fData column, exactly as svmClassification(scores = "all") writes it.
  score_matrix <- matrix(
    stats::runif(n_features * n_classes),
    nrow = n_features,
    dimnames = list(rownames(exprs_matrix), paste0(classes, ".svm.scores"))
  )
  fdata$svm.all.scores <- score_matrix

  methods::new(
    "MSnSet",
    exprs = exprs_matrix,
    featureData = methods::new("AnnotatedDataFrame", data = fdata),
    phenoData = methods::new("AnnotatedDataFrame", data = data.frame(
      fraction = colnames(exprs_matrix),
      row.names = colnames(exprs_matrix),
      stringsAsFactors = FALSE
    ))
  )
}

skip_without_rhdf5 <- function() {
  if (!requireNamespace("rhdf5", quietly = TRUE)) {
    testthat::skip("rhdf5 is not installed, so anndataR cannot touch an h5ad file")
  }
}

## What a written artifact actually holds, read back with anndataR rather than reconstructed as an
## MSnSet -- which is the only way to see what Python will be handed.
read_artifact <- function(x, ...) {
  skip_without_rhdf5()
  path <- tempfile(fileext = ".h5ad")
  suppressMessages(grassp_write_msnset(x, path, ...))
  anndataR::read_h5ad(path)
}

## An h5ad standing in for one Python wrote, built with anndataR directly so that nothing in this
## package is involved in producing it.
write_python_style <- function(obs = NULL, var = NULL, obsm = list(), varm = list(),
                               uns = list(), layers = list(), x = NULL,
                               n_features = 4L, n_fractions = 3L,
                               feature_names = sprintf("P%02d", seq_len(n_features))) {
  skip_without_rhdf5()
  set.seed(7)
  fraction_names <- sprintf("F%d", seq_len(n_fractions))
  if (is.null(x)) {
    x <- matrix(stats::runif(n_features * n_fractions), nrow = n_features)
    x <- x / rowSums(x)
  }
  if (is.null(obs)) {
    obs <- data.frame(row.names = feature_names)
  } else {
    rownames(obs) <- feature_names
  }
  if (is.null(var)) {
    var <- data.frame(row.names = fraction_names)
  } else {
    rownames(var) <- fraction_names
  }
  path <- tempfile(fileext = ".h5ad")
  anndataR::write_h5ad(
    anndataR::AnnData(
      X = x, obs = obs, var = var, obsm = obsm, varm = varm, layers = layers, uns = uns
    ),
    path,
    mode = "w"
  )
  path
}


# ---------------------------------------------------------------------------
# The sentinel
# ---------------------------------------------------------------------------

test_that("'unknown' becomes NA on the way out, in every text column", {
  x <- make_msnset()
  fd <- Biobase::fData(x)
  fd$markers.orig <- fd$markers
  Biobase::fData(x) <- fd

  obs <- read_artifact(x)$obs
  expect_true(anyNA(obs$markers))
  expect_true(anyNA(obs$markers.orig))
  expect_false(any(as.character(obs$markers) == "unknown", na.rm = TRUE))
})

test_that("unknown_to_na = FALSE keeps the artifact faithful to the MSnSet", {
  obs <- read_artifact(make_msnset(), unknown_to_na = FALSE)$obs
  expect_true("unknown" %in% as.character(obs$markers))
  expect_false(anyNA(obs$markers))
})

test_that("NA becomes the sentinel on the way in, and a factor keeps its levels", {
  path <- write_python_style(obs = data.frame(
    markers = factor(c("Nucleus", NA, "Cytosol", NA), levels = c("Nucleus", "Cytosol")),
    stringsAsFactors = FALSE
  ))
  markers <- Biobase::fData(suppressMessages(grassp_as_msnset(path)))$markers
  expect_false(anyNA(markers))
  expect_true(is.factor(markers))
  # the level gained, not a flattening to character: the original order still leads
  expect_equal(levels(markers), c("Nucleus", "Cytosol", "unknown"))
  expect_equal(as.character(markers), c("Nucleus", "unknown", "Cytosol", "unknown"))
})

test_that("nan_to_unknown = FALSE leaves NA alone", {
  path <- write_python_style(obs = data.frame(
    markers = c("Nucleus", NA, "Cytosol", NA), stringsAsFactors = FALSE
  ))
  markers <- Biobase::fData(suppressMessages(grassp_as_msnset(path, nan_to_unknown = FALSE)))$markers
  expect_true(anyNA(markers))
})

test_that("the sentinel round trips, so pRoloc's marker helpers work either way", {
  path <- write_python_style(obs = data.frame(
    markers = factor(c("Nucleus", NA, "Cytosol", "Nucleus")), stringsAsFactors = FALSE
  ))
  x <- suppressMessages(grassp_as_msnset(path))
  expect_equal(nrow(pRoloc::markerMSnSet(x, fcol = "markers")), 3L)
  expect_equal(nrow(pRoloc::unknownMSnSet(x, fcol = "markers")), 1L)
  expect_setequal(pRoloc::getMarkerClasses(x, fcol = "markers"), c("Nucleus", "Cytosol"))

  obs <- read_artifact(x)$obs
  expect_true(is.na(obs$markers[[2]]))
})


# ---------------------------------------------------------------------------
# Dtypes: what Python is actually handed
# ---------------------------------------------------------------------------

test_that("a factor keeps its level order and its ordered flag", {
  x <- make_msnset()
  fd <- Biobase::fData(x)
  # deliberately not alphabetical, and with no NA at all -- the case that used to be flattened
  fd$compartment <- factor(
    rep(c("Nucleus", "Cytosol", "Golgi", "ER"), length.out = nrow(fd)),
    levels = c("Nucleus", "Cytosol", "Golgi", "ER"),
    ordered = TRUE
  )
  Biobase::fData(x) <- fd

  written <- read_artifact(x)$obs$compartment
  expect_true(is.factor(written))
  expect_equal(levels(written), c("Nucleus", "Cytosol", "Golgi", "ER"))
  expect_true(is.ordered(written))
})

test_that("dropping the sentinel level leaves the other levels in place and in order", {
  x <- make_msnset()
  fd <- Biobase::fData(x)
  fd$markers <- factor(fd$markers, levels = c("Compartment 3", "unknown", "Compartment 1",
                                              "Compartment 2", "Compartment 4"))
  Biobase::fData(x) <- fd

  written <- read_artifact(x)$obs$markers
  # "Compartment 4" has no members but was declared, so it must survive: droplevels() would
  # silently take it with the sentinel
  expect_equal(levels(written),
               c("Compartment 3", "Compartment 1", "Compartment 2", "Compartment 4"))
})

test_that("a character column carrying NA is written as a factor, not the string 'NA'", {
  # anndataR has no nullable-string encoding: it writes a character NA as the literal two
  # characters "NA", which arrives in Python as a compartment called "NA".
  x <- make_msnset()
  fd <- Biobase::fData(x)
  fd$`Gene names` <- c(NA, sprintf("GENE%d", seq_len(nrow(fd) - 1L)))
  Biobase::fData(x) <- fd

  written <- read_artifact(x)$obs$`Gene names`
  expect_true(is.factor(written))
  expect_true(is.na(written[[1]]))
  expect_false("NA" %in% levels(written))
})

test_that("pData gets the same treatment as fData", {
  # the axis that used to be skipped: a pData NA arrived in Python as the string "NA"
  x <- make_msnset()
  pd <- Biobase::pData(x)
  pd$condition <- c("treated", NA, "control", NA, "treated")
  pd$stage <- factor(c("late", "early", "late", "early", "late"),
                     levels = c("late", "early"), ordered = TRUE)
  Biobase::pData(x) <- pd

  written <- read_artifact(x)$var
  expect_true(is.na(written$condition[[2]]))
  expect_false("NA" %in% levels(written$condition))
  expect_equal(levels(written$stage), c("late", "early"))
  expect_true(is.ordered(written$stage))
})


# ---------------------------------------------------------------------------
# Matrix-valued columns <-> obsm/varm
# ---------------------------------------------------------------------------

test_that("a matrix fData column is written as an obsm data frame that names itself", {
  written <- read_artifact(make_msnset())
  scores <- written$obsm$svm.all.scores
  expect_s3_class(scores, "data.frame")
  expect_equal(
    colnames(scores),
    paste0(paste("Compartment", 1:3), ".svm.scores")
  )
  # no side table is needed, and none is written
  expect_false("obsm_colnames" %in% names(written$uns))
})

test_that("matrix fData columns survive the round trip", {
  skip_without_rhdf5()
  original <- make_msnset()
  path <- tempfile(fileext = ".h5ad")
  expect_message(grassp_write_msnset(original, path), "matrix column")

  restored <- suppressMessages(grassp_as_msnset(path))

  expect_true(methods::validObject(restored))
  expect_equal(dim(Biobase::exprs(restored)), dim(Biobase::exprs(original)))
  expect_equal(
    unname(Biobase::exprs(restored)),
    unname(Biobase::exprs(original)),
    tolerance = 1e-8
  )
  expect_equal(Biobase::featureNames(restored), Biobase::featureNames(original))
  expect_equal(Biobase::sampleNames(restored), Biobase::sampleNames(original))

  # a matrix column, not flattened into dot-pasted scalars
  expect_true(is.matrix(Biobase::fData(restored)$svm.all.scores))
  expect_equal(
    colnames(Biobase::fData(restored)$svm.all.scores),
    colnames(Biobase::fData(original)$svm.all.scores)
  )
  expect_equal(
    unname(Biobase::fData(restored)$svm.all.scores),
    unname(Biobase::fData(original)$svm.all.scores),
    tolerance = 1e-8
  )
})

test_that("class names containing '/' fall back to a plain array plus uns categories", {
  skip_without_rhdf5()
  # HDF5 reads "/" as a path separator and rhdf5 will not create the intermediate group, so such
  # a name cannot be a data-frame column. pRoloc produces them: hyperLOPIT's marker classes
  # include "Endoplasmic reticulum/Golgi apparatus".
  x <- make_msnset()
  fd <- Biobase::fData(x)
  classes <- c("Endoplasmic reticulum/Golgi apparatus", "Cytosol")
  fd$tagm.map.joint <- matrix(
    seq_len(nrow(fd) * 2) / 100, nrow = nrow(fd),
    dimnames = list(rownames(fd), classes)
  )
  Biobase::fData(x) <- fd

  path <- tempfile(fileext = ".h5ad")
  expect_message(grassp_write_msnset(x, path), "cannot be HDF5 dataset names")

  written <- anndataR::read_h5ad(path)
  expect_false(is.data.frame(written$obsm$tagm.map.joint))
  expect_equal(.as_character_vector(written$uns$tagm.map.joint_categories), classes)

  # and the names come back on the way in
  restored <- suppressMessages(grassp_as_msnset(path))
  expect_equal(colnames(Biobase::fData(restored)$tagm.map.joint), classes)
})

test_that("varm round trips as matrix pData columns and subsets with samples", {
  skip_without_rhdf5()
  original <- make_msnset()
  pcs <- matrix(seq_len(ncol(original) * 3) / 10, nrow = ncol(original),
                dimnames = list(Biobase::sampleNames(original), paste0("PC", 1:3)))
  pd <- Biobase::pData(original)
  pd$PCs <- pcs
  Biobase::pData(original) <- pd
  expect_true(is.matrix(Biobase::pData(original)$PCs))

  path <- tempfile(fileext = ".h5ad")
  suppressMessages(grassp_write_msnset(original, path))
  restored <- suppressMessages(grassp_as_msnset(path))

  expect_true(is.matrix(Biobase::pData(restored)$PCs))
  expect_equal(colnames(Biobase::pData(restored)$PCs), c("PC1", "PC2", "PC3"))
  expect_equal(unname(Biobase::pData(restored)$PCs), unname(pcs), tolerance = 1e-8)
  # column subsetting takes the samples with it -- the property that makes this safe
  sub <- restored[, 1:2]
  expect_equal(nrow(Biobase::pData(sub)$PCs), 2L)
})

test_that("a bare obsm array takes its names from uns[['<key>_categories']]", {
  # what grassp's own annotators write: an array plus the categories under a sibling uns key
  categories <- c("Cytosol", "Nucleus", "Golgi")
  path <- write_python_style(
    obsm = list(ann_probabilities = matrix(stats::runif(12), nrow = 4)),
    uns = list(ann_categories = categories)
  )
  restored <- suppressMessages(grassp_as_msnset(path))
  expect_equal(colnames(Biobase::fData(restored)$ann_probabilities), categories)
})

test_that("failing that, a bare obsm array takes them from its companion label column", {
  # how portal datasets are curated: no _categories entry, but the label column is a factor of
  # exactly the right width
  path <- write_python_style(
    obs = data.frame(ann = factor(c("Cytosol", "Nucleus", "Golgi", "Cytosol"),
                                  levels = c("Cytosol", "Nucleus", "Golgi"))),
    obsm = list(ann_probabilities = matrix(stats::runif(12), nrow = 4))
  )
  restored <- suppressMessages(grassp_as_msnset(path))
  expect_equal(
    colnames(Biobase::fData(restored)$ann_probabilities),
    c("Cytosol", "Nucleus", "Golgi")
  )
})

test_that("the companion column is read before the sentinel is filled in", {
  # The width match is against the label column's *original* levels. Filling NA with "unknown"
  # adds one, so doing that first would lose the names by one column -- and a label column with
  # unlabelled proteins is the normal case, not the exception.
  path <- write_python_style(
    obs = data.frame(ann = factor(c("Cytosol", NA, "Golgi", "Cytosol"),
                                  levels = c("Cytosol", "Nucleus", "Golgi"))),
    obsm = list(ann_probabilities = matrix(stats::runif(12), nrow = 4))
  )
  restored <- suppressMessages(grassp_as_msnset(path))
  expect_equal(
    colnames(Biobase::fData(restored)$ann_probabilities),
    c("Cytosol", "Nucleus", "Golgi")
  )
  # and the sentinel still made it into the label column
  expect_true("unknown" %in% as.character(Biobase::fData(restored)$ann))
})

test_that("an embedding stays nameless, and goes back out as the array it arrived as", {
  # The alternative -- inventing V1..Vn -- would send X_umap back to Python as a data frame of
  # meaningless columns, where everything downstream, scanpy included, expects an array.
  embedding <- matrix(stats::runif(8), nrow = 4)
  path <- write_python_style(obsm = list(X_umap = embedding))
  restored <- suppressMessages(grassp_as_msnset(path))
  expect_true(is.matrix(Biobase::fData(restored)$X_umap))
  expect_null(colnames(Biobase::fData(restored)$X_umap))

  written <- read_artifact(restored)$obsm$X_umap
  expect_false(is.data.frame(written))
  expect_equal(as.matrix(written), embedding, ignore_attr = TRUE, tolerance = 1e-8)
})

test_that("a named matrix column still goes out as a data frame", {
  written <- read_artifact(make_msnset())$obsm$svm.all.scores
  expect_s3_class(written, "data.frame")
})

test_that("a non-numeric obsm entry is skipped rather than aborting the import", {
  path <- write_python_style(
    obsm = list(notes = data.frame(a = c("x", "y", "z", "w"), b = c("1", "2", "3", "4")))
  )
  expect_message(grassp_as_msnset(path), "Skipping obsm\\$notes")
  restored <- suppressMessages(grassp_as_msnset(path))
  expect_false("notes" %in% Biobase::fvarLabels(restored))
})


# ---------------------------------------------------------------------------
# Reading what Python writes
# ---------------------------------------------------------------------------

test_that("a sparse X is densified into exprs", {
  skip_without_rhdf5()
  n <- 4L
  sparse <- Matrix::rsparsematrix(n, 3L, density = 0.6)
  sparse <- methods::as(abs(sparse), "CsparseMatrix")
  path <- write_python_style(x = sparse)
  restored <- suppressMessages(grassp_as_msnset(path))
  expect_true(is.matrix(Biobase::exprs(restored)))
  expect_equal(Biobase::exprs(restored), as.matrix(sparse),
               ignore_attr = TRUE, tolerance = 1e-8)
})

test_that("duplicated or blank feature names are rejected, since R cannot use them", {
  # Tested at the function rather than through a file: Python only *warns* about duplicate
  # obs_names, so an object can reach here with them, but R forbids duplicate row.names outright,
  # so no such h5ad can be built with anndataR to read back.
  expect_error(.check_feature_names(c("P01", "P01", "P02")), "must be unique")
  expect_error(.check_feature_names(c("P01", "", "P02")), "blank or NA-like")
  expect_error(.check_feature_names(c("P01", "nan", "P02")), "blank or NA-like")
  expect_error(.check_feature_names(c("P01", "  ", "P02")), "blank or NA-like")
  expect_true(.check_feature_names(c("P01", "P02", "P03")))
})

test_that("missing values in exprs get an advisory, since pRoloc methods stop on them", {
  skip_without_rhdf5()
  x <- matrix(c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), nrow = 4)
  x[1, 1] <- NA_real_
  expect_message(grassp_as_msnset(write_python_style(x = x)), "missing values")
})

test_that("a profile scale other than sum-to-1 is not remarked on", {
  # Sum normalisation is a convention of the field, not something pRoloc enforces, and
  # pRolocdata is full of legitimate objects on other scales -- dunkley2006's rows sum to 4.
  skip_without_rhdf5()
  x <- matrix(c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), nrow = 4)
  expect_silent(grassp_as_msnset(write_python_style(x = x)))
})

test_that("layers round trip as assayData elements and subset with exprs", {
  skip_without_rhdf5()
  original <- make_msnset()
  pv <- Biobase::exprs(original)
  pv[] <- seq_along(pv) / length(pv)
  Biobase::assayDataElement(original, "pvals") <- pv
  expect_setequal(Biobase::assayDataElementNames(original), c("exprs", "pvals"))

  path <- tempfile(fileext = ".h5ad")
  suppressMessages(grassp_write_msnset(original, path))
  restored <- suppressMessages(grassp_as_msnset(path))

  expect_setequal(Biobase::assayDataElementNames(restored), c("exprs", "pvals"))
  expect_equal(
    unname(Biobase::assayDataElement(restored, "pvals")),
    unname(pv),
    tolerance = 1e-8
  )
  # exprs stays the matrix pRoloc operates on
  expect_equal(
    unname(Biobase::exprs(restored)), unname(Biobase::exprs(original)),
    tolerance = 1e-8
  )
  # and the extra element subsets along with it -- the property that makes this safe
  sub <- restored[1:4, 1:3]
  expect_equal(dim(Biobase::assayDataElement(sub, "pvals")), c(4L, 3L))
  mk <- pRoloc::markerMSnSet(restored, fcol = "markers")
  expect_equal(nrow(Biobase::assayDataElement(mk, "pvals")), nrow(mk))
})

test_that("an object with no extra elements still round trips", {
  skip_without_rhdf5()
  path <- tempfile(fileext = ".h5ad")
  suppressMessages(grassp_write_msnset(make_msnset(), path))
  restored <- suppressMessages(grassp_as_msnset(path))
  expect_equal(Biobase::assayDataElementNames(restored), "exprs")
})


# ---------------------------------------------------------------------------
# Names, selection, and the file itself
# ---------------------------------------------------------------------------

test_that("column names cross verbatim in both directions", {
  skip_without_rhdf5()
  original <- make_msnset()
  # a non-syntactic name, which pRoloc tolerates -- svmClassification runs on such an object
  fd <- Biobase::fData(original)
  fd$`Gene names` <- sprintf("G%d", seq_len(nrow(fd)))
  Biobase::fData(original) <- fd

  path <- tempfile(fileext = ".h5ad")
  suppressMessages(grassp_write_msnset(original, path))
  restored <- suppressMessages(grassp_as_msnset(path))

  expect_true("Gene names" %in% colnames(Biobase::fData(restored)))
  expect_equal(Biobase::sampleNames(restored), Biobase::sampleNames(original))
})

test_that("write_msnset rejects a non-MSnSet", {
  expect_error(grassp_write_msnset(data.frame(a = 1), tempfile()), "must be an MSnSet")
})

test_that("the processing log is appended to rather than replaced", {
  skip_without_rhdf5()
  path <- tempfile(fileext = ".h5ad")
  suppressMessages(grassp_write_msnset(make_msnset(), path))
  restored <- suppressMessages(grassp_as_msnset(path))
  log <- MSnbase::processingData(restored)@processing
  expect_true(any(grepl("Imported from grassp h5ad", log)))
})

test_that("columns and drop restrict what is written", {
  skip_without_rhdf5()
  path <- tempfile(fileext = ".h5ad")
  suppressMessages(grassp_write_msnset(make_msnset(), path, drop = "svm.scores"))
  restored <- suppressMessages(grassp_as_msnset(path))
  expect_false("svm.scores" %in% colnames(Biobase::fData(restored)))
  expect_true("markers" %in% colnames(Biobase::fData(restored)))
})

test_that("selecting columns keeps only those, plus matrix columns", {
  skip_without_rhdf5()
  path <- tempfile(fileext = ".h5ad")
  suppressMessages(grassp_write_msnset(make_msnset(), path, columns = c("markers")))
  restored <- suppressMessages(grassp_as_msnset(path))
  expect_true("markers" %in% colnames(Biobase::fData(restored)))
  expect_false("svm.scores" %in% colnames(Biobase::fData(restored)))
  # matrix columns are selected separately from the scalar `columns` filter
  expect_true(is.matrix(Biobase::fData(restored)$svm.all.scores))
})

test_that("reading a missing file fails clearly", {
  skip_without_rhdf5()
  expect_error(grassp_as_msnset(tempfile(fileext = ".h5ad")), "No such file")
})

test_that("an existing file is not clobbered without overwrite", {
  skip_without_rhdf5()
  path <- tempfile(fileext = ".h5ad")
  x <- make_msnset()
  suppressMessages(grassp_write_msnset(x, path))
  # anndataR's default mode is "w-", so without handling this a re-run of any user script
  # would fail on its second invocation with a confusing error.
  expect_error(grassp_write_msnset(x, path), "overwrite = TRUE")
  expect_silent(suppressMessages(grassp_write_msnset(x, path, overwrite = TRUE)))
})


# ---------------------------------------------------------------------------
# Unmapped uns rides on experimentData@other
# ---------------------------------------------------------------------------
#
# This is what makes grassp -> pRoloc -> grassp lossless rather than merely usable: an MSnSet has
# no slot for arbitrary metadata, so entries with no MSnSet slot are parked on
# experimentData@other and handed straight back out.

test_that("uns entries with no MSnSet slot are parked on experimentData@other", {
  path <- write_python_style(uns = list(
    neighbors = list(params = list(n_neighbors = 15L)),
    schema_version = "0.3.0"
  ))
  x <- suppressMessages(grassp_as_msnset(path))
  carried <- Biobase::experimentData(x)@other$grassp_uns
  expect_equal(as.character(carried$schema_version), "0.3.0")
  expect_equal(as.integer(carried$neighbors$params$n_neighbors), 15L)
  expect_true(isTRUE(methods::validObject(x, test = TRUE)))
})

test_that("carried uns survives real pRoloc work, then rides back out", {
  path <- write_python_style(
    obs = data.frame(markers = factor(rep(c("Nucleus", "Cytosol"), each = 2L))),
    uns = list(schema_version = "0.3.0", umap = list(params = list(a = 1.5)))
  )
  x <- suppressMessages(grassp_as_msnset(path))
  # `other` is not row- or column-aligned, which for uns is exactly right: unlike a graph, there
  # is nothing for subsetting to get wrong.
  x <- x[1:3, 1:2]
  x <- MSnbase::normalise(x, method = "sum")
  expect_false(is.null(Biobase::experimentData(x)@other$grassp_uns))
  expect_false(
    is.null(
      Biobase::experimentData(
        pRoloc::markerMSnSet(x, fcol = "markers")
      )@other$grassp_uns
    )
  )

  uns <- read_artifact(x)$uns
  expect_equal(as.character(uns$schema_version), "0.3.0")
  expect_equal(as.numeric(uns$umap$params$a), 1.5)
})

test_that("the processing log is regenerated from the object rather than carried", {
  path <- write_python_style(uns = list(processing = "Loaded in grassp"))
  x <- suppressMessages(grassp_as_msnset(path))
  processing <- .as_character_vector(read_artifact(x)$uns$processing)
  expect_true(any(grepl("Loaded in grassp", processing)))
  expect_true(any(grepl("Imported from grassp h5ad", processing)))
})

test_that("an MSnSet that never saw grassp still writes", {
  written <- read_artifact(make_msnset())
  expect_equal(nrow(written$obs), 8L)
  expect_true("svm.all.scores" %in% names(written$obsm))
})
