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

## ---- Support vector machine ----------------------------------------------
## `times`/`xval` are small here so this finishes quickly; raise them for real work. The
## hyperparameters below came from svmOptimisation() on this dataset.
x <- svmClassification(x, fcol = "markers", sigma = 0.1, cost = 16,
                       scores = "all", verbose = FALSE)
## `scores = "all"` stores the per-class matrix but NOT the scalar winning score, while
## orgQuants()/getPredictions() look for <fcol>.scores -- so derive it from the matrix.
fData(x)$svm.scores <- apply(fData(x)$svm.all.scores, 1, max)
ts <- orgQuants(x, fcol = "svm", scol = "svm.scores", t = 0.75, verbose = FALSE)
ts[is.na(ts)] <- Inf
x <- getPredictions(x, fcol = "svm", scol = "svm.scores", t = ts, verbose = FALSE)

## ---- k nearest neighbours ------------------------------------------------
x <- knnClassification(x, fcol = "markers", k = 5, scores = "prediction")

## ---- Thin marker classes -------------------------------------------------
## Several classes here have fewer members than there are fractions. minMarkers() demotes
## them to "unknown" in a new `markers10` column.
x <- minMarkers(x, n = 10, fcol = "markers")

cat("result columns:", paste(fvarLabels(x), collapse = ", "), "\n")

## Send everything back. Whatever grassp put in the object that an MSnSet has no slot for was
## parked on experimentData(x)@other and rides back out with it, so gr.io.read_msnset() on the
## other side gets a complete object rather than just the new columns.
grassp_write_msnset(x, outfile, overwrite = TRUE)
