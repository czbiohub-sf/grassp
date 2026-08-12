## Small helpers shared by the reader and the writer.

## Whatever an h5ad reader hands back for a name list, as a plain character vector.
.as_character_vector <- function(x) {
  if (is.null(x)) {
    return(character(0))
  }
  as.character(unlist(x, use.names = FALSE))
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
