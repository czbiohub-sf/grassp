#' @keywords internal
"_PACKAGE"

## Every external call in this package is written `pkg::fn()`, so these directives are not what
## makes the code resolve. They keep NAMESPACE explicit about the package's external surface,
## and mean an unqualified call added later still finds its function.
#' @importFrom anndataR AnnData read_h5ad write_h5ad
#' @importFrom Biobase exprs fData pData
#' @importFrom methods is new validObject
#' @importFrom MSnbase processingData
NULL
