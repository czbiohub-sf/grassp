## Tests for the R half of the grassp <-> pRoloc bridge.
##
## anndataR is a hard dependency, so there is no backend to skip on. The h5ad round trip does
## still need rhdf5, which anndataR calls but declares only in Suggests, so the tests that touch
## a file skip without it. The pieces that do not -- the spec-version guard, the matrix/scalar
## split, the "unknown" encoding -- always run, because those break silently rather than loudly.

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

test_that("the declared spec version is well formed", {
  expect_match(grassp_msnset_spec(), "^grassp-msnset/[0-9]+$")
})

test_that("a newer artifact is rejected rather than misread", {
  expect_error(
    grasspio:::.check_spec("grassp-msnset/99"),
    "Update grasspio"
  )
})

test_that("the current spec is accepted", {
  expect_silent(grasspio:::.check_spec(grassp_msnset_spec()))
})

test_that("a malformed spec raises", {
  expect_error(grasspio:::.check_spec("nonsense"), "Unrecognised")
})

test_that("a missing spec is tolerated, so plain h5ad stays readable", {
  expect_true(is.na(grasspio:::.check_spec(NULL)))
})

test_that("NA and blank markers become pRoloc's 'unknown'", {
  expect_equal(
    grasspio:::.to_unknown(c("Golgi", NA, "", "  ", "ER")),
    c("Golgi", "unknown", "unknown", "unknown", "ER")
  )
})

test_that("'unknown' becomes NA on the way out", {
  expect_equal(
    grasspio:::.to_na(c("Golgi", "unknown", "ER")),
    c("Golgi", NA, "ER")
  )
})

test_that("column names cross verbatim in both directions", {
  skip_without_rhdf5()
  original <- make_msnset()
  # a non-syntactic name, which pRoloc tolerates -- svmClassification runs on such an object
  fd <- Biobase::fData(original)
  fd$`Gene names` <- sprintf("G%d", seq_len(nrow(fd)))
  Biobase::fData(original) <- fd

  path <- tempfile(fileext = ".h5ad")
  grassp_write_msnset(original, path)
  restored <- grassp_as_msnset(path)

  expect_true("Gene names" %in% colnames(Biobase::fData(restored)))
  expect_equal(Biobase::sampleNames(restored), Biobase::sampleNames(original))
})

test_that("write_msnset rejects a non-MSnSet", {
  expect_error(grassp_write_msnset(data.frame(a = 1), tempfile()), "must be an MSnSet")
})

test_that("matrix fData columns survive the round trip as obsm", {
  skip_without_rhdf5()
  original <- make_msnset()
  path <- tempfile(fileext = ".h5ad")
  expect_message(grassp_write_msnset(original, path), "matrix column")

  restored <- grassp_as_msnset(path)

  expect_true(methods::validObject(restored))
  expect_equal(dim(Biobase::exprs(restored)), dim(Biobase::exprs(original)))
  expect_equal(
    unname(Biobase::exprs(restored)),
    unname(Biobase::exprs(original)),
    tolerance = 1e-8
  )
  expect_equal(Biobase::featureNames(restored), Biobase::featureNames(original))
  expect_equal(Biobase::sampleNames(restored), Biobase::sampleNames(original))

  # The score matrix must come back as a matrix column, not flattened into scalars.
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

test_that("markers keep the 'unknown' encoding pRoloc requires", {
  skip_without_rhdf5()
  path <- tempfile(fileext = ".h5ad")
  grassp_write_msnset(make_msnset(), path)
  restored <- grassp_as_msnset(path)
  markers <- Biobase::fData(restored)$markers
  expect_false(anyNA(markers))
  expect_true("unknown" %in% markers)
})

test_that("the processing log is appended to rather than replaced", {
  skip_without_rhdf5()
  path <- tempfile(fileext = ".h5ad")
  grassp_write_msnset(make_msnset(), path)
  restored <- grassp_as_msnset(path)
  log <- MSnbase::processingData(restored)@processing
  expect_true(any(grepl("Imported from grassp h5ad", log)))
})

test_that("columns and drop restrict what is written", {
  skip_without_rhdf5()
  path <- tempfile(fileext = ".h5ad")
  grassp_write_msnset(make_msnset(), path, drop = "svm.scores")
  restored <- grassp_as_msnset(path)
  expect_false("svm.scores" %in% colnames(Biobase::fData(restored)))
  expect_true("markers" %in% colnames(Biobase::fData(restored)))
})

test_that("reading a missing file fails clearly", {
  skip_without_rhdf5()
  expect_error(grassp_as_msnset(tempfile(fileext = ".h5ad")), "No such file")
})

test_that("an existing file is not clobbered without overwrite", {
  skip_without_rhdf5()
  path <- tempfile(fileext = ".h5ad")
  x <- make_msnset()
  grassp_write_msnset(x, path)
  # anndataR's default mode is "w-", so without handling this a re-run of any user script
  # would fail on its second invocation with a confusing error.
  expect_error(grassp_write_msnset(x, path), "overwrite = TRUE")
  expect_silent(suppressMessages(grassp_write_msnset(x, path, overwrite = TRUE)))
})

test_that("layers round trip as assayData elements and subset with exprs", {
  skip_without_rhdf5()
  original <- make_msnset()
  pv <- Biobase::exprs(original)
  pv[] <- seq_along(pv) / length(pv)
  Biobase::assayDataElement(original, "pvals") <- pv
  expect_setequal(Biobase::assayDataElementNames(original), c("exprs", "pvals"))

  path <- tempfile(fileext = ".h5ad")
  grassp_write_msnset(original, path)
  restored <- grassp_as_msnset(path)

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
  grassp_write_msnset(original, path)
  restored <- grassp_as_msnset(path)

  expect_true(is.matrix(Biobase::pData(restored)$PCs))
  expect_equal(colnames(Biobase::pData(restored)$PCs), c("PC1", "PC2", "PC3"))
  expect_equal(unname(Biobase::pData(restored)$PCs), unname(pcs), tolerance = 1e-8)
  # column subsetting takes the samples with it -- the property that makes this safe
  sub <- restored[, 1:2]
  expect_equal(nrow(Biobase::pData(sub)$PCs), 2L)
})

test_that("an object with no extra elements still round trips", {
  skip_without_rhdf5()
  path <- tempfile(fileext = ".h5ad")
  grassp_write_msnset(make_msnset(), path)
  restored <- grassp_as_msnset(path)
  expect_equal(Biobase::assayDataElementNames(restored), "exprs")
})

## --- unmapped uns rides on experimentData@other ---------------------------------------------
##
## This is what makes grassp -> pRoloc -> grassp lossless rather than merely usable: an MSnSet
## has no slot for arbitrary metadata, so entries the contract does not map are parked on
## experimentData@other and handed straight back out.

## The uns a written artifact actually carries.
read_uns <- function(path) anndataR::read_h5ad(path)$uns

## An artifact whose uns has extra entries, standing in for what grassp emits.
make_h5ad_with_uns <- function(extra) {
  skip_without_rhdf5()
  path <- tempfile(fileext = ".h5ad")
  grassp_write_msnset(make_msnset(), path)
  source <- anndataR::read_h5ad(path)
  out <- tempfile(fileext = ".h5ad")
  anndataR::write_h5ad(
    anndataR::AnnData(
      X = source$X,
      obs = source$obs,
      var = source$var,
      obsm = source$obsm,
      uns = utils::modifyList(source$uns, extra)
    ),
    out,
    mode = "w"
  )
  out
}

test_that("uns entries with no MSnSet slot are parked on experimentData@other", {
  path <- make_h5ad_with_uns(list(
    neighbors = list(params = list(n_neighbors = 15L)),
    schema_version = "0.3.0"
  ))
  x <- grassp_as_msnset(path)
  carried <- Biobase::experimentData(x)@other$grassp_uns
  expect_equal(carried$schema_version, "0.3.0")
  expect_equal(carried$neighbors$params$n_neighbors, 15L)
  # the contract's own keys are not duplicated in there
  expect_false(any(c("msnset_spec", "obsm_colnames") %in% names(carried)))
  expect_true(isTRUE(methods::validObject(x, test = TRUE)))
})

test_that("carried uns survives real pRoloc work, then rides back out", {
  path <- make_h5ad_with_uns(
    list(schema_version = "0.3.0", umap = list(params = list(a = 1.5)))
  )
  x <- grassp_as_msnset(path)
  # `other` is not row- or column-aligned, which for uns is exactly right: unlike a graph,
  # there is nothing for subsetting to get wrong.
  x <- x[1:6, 1:4]
  x <- MSnbase::normalise(x, method = "sum")
  expect_false(is.null(Biobase::experimentData(x)@other$grassp_uns))
  expect_false(
    is.null(
      Biobase::experimentData(
        pRoloc::markerMSnSet(x, fcol = "markers")
      )@other$grassp_uns
    )
  )

  out <- tempfile(fileext = ".h5ad")
  grassp_write_msnset(x, out)
  uns <- read_uns(out)
  expect_equal(as.character(uns$schema_version), "0.3.0")
  expect_equal(as.numeric(uns$umap$params$a), 1.5)
})

test_that("the contract's uns keys are regenerated from the object, not carried", {
  path <- make_h5ad_with_uns(list(
    msnset_dropped = "obsp:connectivities",
    obsm_colnames = list(gone = c("a", "b"))
  ))
  x <- grassp_as_msnset(path)
  out <- tempfile(fileext = ".h5ad")
  grassp_write_msnset(x, out)
  uns <- read_uns(out)
  expect_equal(uns$msnset_spec, grassp_msnset_spec())
  expect_length(uns$msnset_dropped, 0L)
  expect_false("gone" %in% names(uns$obsm_colnames))
  expect_true("svm.all.scores" %in% names(uns$obsm_colnames))
})

test_that("an MSnSet that never saw grassp still writes", {
  skip_without_rhdf5()
  path <- tempfile(fileext = ".h5ad")
  grassp_write_msnset(make_msnset(), path)
  expect_equal(read_uns(path)$msnset_spec, grassp_msnset_spec())
})

test_that("selecting columns keeps only those, plus matrix columns", {
  skip_without_rhdf5()
  path <- tempfile(fileext = ".h5ad")
  grassp_write_msnset(make_msnset(), path, columns = c("markers"))
  restored <- grassp_as_msnset(path)
  expect_true("markers" %in% colnames(Biobase::fData(restored)))
  expect_false("svm.scores" %in% colnames(Biobase::fData(restored)))
  # matrix columns are selected separately from the scalar `columns` filter
  expect_true(is.matrix(Biobase::fData(restored)$svm.all.scores))
})
