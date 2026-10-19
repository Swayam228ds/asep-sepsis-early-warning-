# Turns the full Yale dataset (Results/5v_cleandf.RData, 560,486 x 972) into the triage-only CSV
# used by m2_triage.py. Python's pyreadr needs more than 14 GB of RAM for this file; R's load() is fine.
#
# Usage: Rscript training/prepare_yale.R path/to/5v_cleandf.RData data/raw/yale_triage_subset.csv.gz
args <- commandArgs(TRUE)
e <- new.env(); nm <- load(args[1], envir = e); df <- get(nm[1], envir = e)
cols <- names(df)
keep <- c("disposition", "dep_name", "esi", "age", "gender", "ethnicity", "race", "lang", "religion",
          "maritalstatus", "employstatus", "insurance_status", "arrivalmode", "arrivalmonth", "arrivalday",
          "arrivalhour_bin", "previousdispo", "n_edvisits", "n_admissions", "n_surgeries",
          grep("^triage_vital", cols, value = TRUE), grep("^cc_", cols, value = TRUE))
keep <- intersect(keep, cols)
write.csv(df[keep], gzfile(args[2]), row.names = FALSE)
cat("wrote", nrow(df), "rows x", length(keep), "columns to", args[2], "\n")
