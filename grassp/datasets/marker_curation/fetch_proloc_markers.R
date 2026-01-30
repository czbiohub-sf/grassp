#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
out_dir <- if (length(args) >= 1) args[[1]] else file.path(".")

# GitHub repository details
github_base_url <- "https://raw.githubusercontent.com/lgatto/pRoloc/devel/inst/extdata"

# List of marker2 RDS files (from GitHub repository)
marker_files <- c(
  "marker2_atha.rds",
  "marker2_dmel.rds",
  "marker2_ggal.rds",
  "marker2_hsap.rds",
  "marker2_hsap_christopher.rds",
  "marker2_hsap_geladaki.rds",
  "marker2_hsap_itzhak.rds",
  "marker2_hsap_villaneuva.rds",
  "marker2_mmus.rds",
  "marker2_mmus_christoforou.rds",
  "marker2_scer_uniprot.rds",
  "marker2_toxo_barylyuk.rds",
  "marker2_tryp_moloney.rds"
)

# Create output directory
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# Create temporary directory for downloads
temp_dir <- tempdir()
message("Downloading files to temporary directory: ", temp_dir)

# Download files from GitHub
downloaded_files <- character()
for (filename in marker_files) {
  url <- file.path(github_base_url, filename)
  local_path <- file.path(temp_dir, filename)

  message("Downloading: ", filename)
  tryCatch({
    download.file(url, local_path, mode = "wb", quiet = TRUE)
    downloaded_files <- c(downloaded_files, local_path)
  }, error = function(e) {
    warning("Failed to download ", filename, ": ", e$message)
  })
}

if (length(downloaded_files) == 0) {
  stop("No files were successfully downloaded")
}

message("\nSuccessfully downloaded ", length(downloaded_files), " files")

# Function to normalize compartment names
normalize_compartment_name <- function(name) {
  if (is.na(name) || !nzchar(name)) {
    return(name)
  }

  # Apply naming standardizations based on naming_resolutions.txt
  # 1. Nucleus/Chromatin -> Nucleus - Chromatin
  name <- gsub("^Nucleus/Chromatin$", "Nucleus - Chromatin", name)

  # 2. Chromatin alone -> Nucleus - Chromatin
  name <- gsub("^Chromatin$", "Nucleus - Chromatin", name)

  # 3. ERGIC/Cis Golgi -> ERGIC
  name <- gsub("^ERGIC/Cis Golgi$", "ERGIC", name)

  # 4. ER/Golgi -> ERGIC
  name <- gsub("^ER/Golgi$", "ERGIC", name)

  # 5. Actin Binding Proteins -> Actin Cytoskeleton
  name <- gsub("^Actin Binding Proteins$", "Actin Cytoskeleton", name)

  return(name)
}

# Function to extract species and author from filename
extract_info <- function(path) {
  base <- sub("\\.rds$", "", basename(path))
  base <- sub("^marker_2_", "", base)
  base <- sub("^marker2_", "", base)
  parts <- strsplit(base, "_")[[1]]
  species <- parts[[1]]
  author <- if (length(parts) > 1) paste(parts[-1], collapse = "_") else "lilley"

  # Special case: marker2_scer_uniprot.rds should use "lilley" as author
  if (basename(path) == "marker2_scer_uniprot.rds") {
    author <- "lilley"
  }

  list(path = path, species = species, author = author)
}

# Function to collapse multiple values
collapse_values <- function(x) {
  values <- unique(na.omit(as.character(x)))
  if (length(values) == 0) {
    return(NA_character_)
  }
  paste(values, collapse = ";")
}

# Function to read and process marker files
read_markers <- function(info) {
  obj <- readRDS(info$path)
  df <- as.data.frame(obj, stringsAsFactors = FALSE)

  # The protein IDs are in the row names
  protein_ids <- rownames(df)

  # The compartments are in the 'markers' column
  compartments <- df$markers

  # Normalize compartment names
  compartments <- sapply(compartments, normalize_compartment_name, USE.NAMES = FALSE)

  # Create a clean data frame
  data <- data.frame(
    id = protein_ids,
    compartment = compartments,
    stringsAsFactors = FALSE
  )

  # Remove rows with missing or empty protein IDs
  data <- data[!is.na(data$id) & nzchar(data$id), ]

  # Aggregate by protein ID in case there are duplicates
  data <- aggregate(data$compartment, by = list(id = data$id), FUN = collapse_values)
  names(data) <- c("id", info$author)

  data
}

# Process downloaded files
message("\nProcessing files...")
file_info <- lapply(downloaded_files, extract_info)
species_groups <- split(file_info, vapply(file_info, `[[`, character(1), "species"))

for (species in names(species_groups)) {
  infos <- species_groups[[species]]
  authors <- vapply(infos, `[[`, character(1), "author")

  # Make unique author names if there are duplicates
  authors <- make.unique(authors)

  for (i in seq_along(infos)) {
    infos[[i]]$author <- authors[[i]]
  }

  data_list <- lapply(infos, read_markers)

  # Merge all data frames for this species
  merged <- Reduce(function(x, y) merge(x, y, by = "id", all = TRUE), data_list)

  # Order columns: id first, then authors in order
  ordered_cols <- c("id", authors)
  ordered_cols <- ordered_cols[ordered_cols %in% names(merged)]
  merged <- merged[, ordered_cols, drop = FALSE]

  # Sort by id
  merged <- merged[order(merged$id), ]

  out_file <- file.path(out_dir, paste0("marker2_", species, "_merged.csv"))
  write.csv(merged, out_file, row.names = FALSE, na = "")

  message("Wrote ", nrow(merged), " rows to: ", out_file)
}

message("\nAll merged CSV files written to: ", out_dir)
message("\nNote: Downloaded files are in temporary directory and will be cleaned up automatically")
