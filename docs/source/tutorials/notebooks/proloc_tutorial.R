#!/usr/bin/env Rscript
# Produces the cached pRoloc results the grassp pRoloc tutorial reads.
#   Rscript proloc_tutorial.R <experiment.h5ad> <results.h5ad>
#
# The output is ~4 MB, so it is NOT committed. It is published alongside the portal datasets at
#   https://public.czbiohub.org/proteinxlocation/internal/proloc_tutorial_results.h5ad
# and the notebook downloads it at build time. If you change anything here, regenerate the file
# and re-upload it, or the tutorial's prose will describe output that no longer exists.
suppressMessages({library(grasspio); library(pRoloc); library(Biobase)})
set.seed(1)

args <- commandArgs(trailingOnly = TRUE)
infile  <- if (length(args) >= 1) args[[1]] else "experiment.h5ad"
outfile <- if (length(args) >= 2) args[[2]] else "proloc_tutorial_results.h5ad"

x <- grassp_as_msnset(infile)
cat("MSnSet:", paste(dim(x), collapse = " x "), "features x fractions\n")
cat("classes:", length(getMarkerClasses(x, fcol = "markers")),
    "| markers:", sum(fData(x)$markers != "unknown"), "\n")

## ---- Thin marker classes -------------------------------------------------
## Two of the twelve classes have fewer than ten markers -- too few to learn from, and with
## twelve classes libsvm's one-vs-one vote (66 pairwise comparisons) gets unstable. minMarkers()
## demotes those to "unknown" in a new `markers10` column: 391 markers over 10 classes, the
## smallest with 13. Cross-validated on this dataset that is worth about +0.05 macro-F1, so both
## classifiers below train on it rather than on `markers`.
x <- minMarkers(x, n = 10, fcol = "markers")

## ---- Support vector machine ----------------------------------------------
## `times`/`xval` are small here so this finishes quickly; raise them for real work. The
## hyperparameters below came from svmOptimisation() on this dataset.
x <- svmClassification(x, fcol = "markers10", sigma = 0.1, cost = 16,
                       scores = "all", verbose = FALSE)
## `scores = "all"` stores the per-class matrix but NOT the scalar winning score, which the
## plots on the Python side use -- so derive it from the matrix.
##
## Deliberately no orgQuants()/getPredictions() here: their per-class thresholding is pRoloc
## teaching material rather than anything the bridge needs, and it lives in the R tutorial
## instead. `svm` is populated for every protein; threshold it yourself if you want to.
fData(x)$svm.scores <- apply(fData(x)$svm.all.scores, 1, max)

## ---- k nearest neighbours ------------------------------------------------
x <- knnClassification(x, fcol = "markers10", k = 5, scores = "prediction")

cat("result columns:", paste(fvarLabels(x), collapse = ", "), "\n")

## Send everything back. Whatever grassp put in the object that an MSnSet has no slot for was
## parked on experimentData(x)@other and rides back out with it, so anndata.read_h5ad() on the
## other side gets a complete object rather than just the new columns.
grassp_write_msnset(x, outfile, overwrite = TRUE)
