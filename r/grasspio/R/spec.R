## The exchange contract: its version, the uns keys it owns, and the guard that stops an
## artifact from a newer grassp being misread. The R-side counterpart of grassp's
## `grassp/io/_msnset.py`; `msnset.R` is the counterpart of `grassp/io/proloc.py`.

GRASSP_MSNSET_SPEC <- "grassp-msnset/1"

## AnnData's obsm/varm arrays carry no column names, so the class names belonging to a
## matrix-valued fData/pData column travel in uns under these keys.
.OBSM_COLNAMES <- "obsm_colnames"
.VARM_COLNAMES <- "varm_colnames"

## uns keys the contract itself owns. They are regenerated from the object on every write and
## never carried across, so an artifact always describes itself rather than whatever object it
## was last read from. Mirrors grassp's `_msnset.RESERVED_UNS_KEYS`, plus `processing`, which
## comes from processingData().
.RESERVED_UNS <- c(
  "msnset_spec", "msnset_exprs_layer", "msnset_dropped",
  .OBSM_COLNAMES, .VARM_COLNAMES, "processing"
)

#' The exchange contract version this package implements
#'
#' Artifacts record their contract version in `uns$msnset_spec`. Both sides assert the major
#' version, so an artifact written by a newer grassp fails loudly instead of being
#' misinterpreted.
#'
#' @return A single string, e.g. `"grassp-msnset/1"`.
#' @export
#' @examples
#' grassp_msnset_spec()
grassp_msnset_spec <- function() {
  GRASSP_MSNSET_SPEC
}

.spec_major <- function(spec) {
  if (is.null(spec) || !length(spec) || !nzchar(spec)) {
    return(NA_integer_)
  }
  matched <- regmatches(spec, regexec("^grassp-msnset/([0-9]+)$", spec))[[1]]
  if (length(matched) != 2L) {
    stop(
      "Unrecognised msnset_spec '", spec, "'; expected 'grassp-msnset/<major>'.",
      call. = FALSE
    )
  }
  as.integer(matched[[2]])
}

.check_spec <- function(spec) {
  found <- .spec_major(spec)
  if (is.na(found)) {
    # A plain h5ad with no contract block. Readable, just heuristically.
    return(invisible(NA_integer_))
  }
  ours <- .spec_major(GRASSP_MSNSET_SPEC)
  if (found > ours) {
    stop(
      "This artifact declares '", spec, "' but grasspio implements '",
      GRASSP_MSNSET_SPEC, "'. Update grasspio with ",
      "remotes::install_github(\"czbiohub-sf/grassp\", subdir = \"r/grasspio\").",
      call. = FALSE
    )
  }
  invisible(found)
}

## anndataR reaches for rhdf5 without declaring it (it is in anndataR's Suggests, so depending
## on anndataR does not pull it in), and calls `rhdf5::` unguarded -- so a user missing it gets
## a bare "there is no package called 'rhdf5'" that names neither h5ad nor the install line.
.require_rhdf5 <- function() {
  if (!requireNamespace("rhdf5", quietly = TRUE)) {
    stop(
      "Reading or writing h5ad needs the 'rhdf5' package, which anndataR uses but does not ",
      "declare as a hard dependency. Install it with:\n",
      "  BiocManager::install(\"rhdf5\")",
      call. = FALSE
    )
  }
  invisible(TRUE)
}
