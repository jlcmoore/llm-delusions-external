###
# Conversation-level GLMMs linking annotation dynamics to conversation length.
#
# This script reads the preprocessed per-message annotation table
# `annotations/all_annotations__preprocessed.parquet`, aggregates to
# conversation level, and fits models where total conversation length
# (in messages) is the outcome and within-conversation annotation
# dynamics are predictors.
#
# For each annotation score column, it computes per-conversation
# features based on the presence of that specific annotation:
#   - `n_pos`: number of positive messages (score >= cutoff).
#   - `first_pos_index`: index of the first positive message.
#   - `last_pos_index`: index of the last positive message.
# The script then fits, per annotation:
#   1) A Poisson GLMM with a participant-level random intercept.
#   2) A Poisson GLM with participant fixed effects.
# Both models treat conversation length as a count outcome.
#
# Results are written as a CSV that can be consumed from Python for
# plotting, with one row per (annotation, model_type, predictor).
#
# Usage (from repo root, using a local R installation):
#   Rscript analysis/conversation_length_glmm.R
#
# Optional positional arguments:
#   1. Path to the Parquet file. Defaults to
#      "annotations/all_annotations__preprocessed.parquet".
#   2. Annotation score column to analyse:
#        - "ALL" (default) to analyse all score__* columns.
#        - A single column name (for example, "score__user-endorses-delusion").
#   3. Numeric cutoff for defining positives. Defaults to 5.
#   4. Output CSV path. Defaults to
#      "analysis/data/conversation_length_glmm_results.csv".
###

suppressPackageStartupMessages({
  library(arrow)
  library(dplyr)
  library(lme4)
  library(tidyr)
})

args <- commandArgs(trailingOnly = TRUE)

parquet_path <- if (length(args) >= 1 && nzchar(args[1])) {
  args[1]
} else {
  "annotations/all_annotations__preprocessed.parquet"
}

score_arg <- if (length(args) >= 2 && nzchar(args[2])) {
  args[2]
} else {
  "ALL"
}

cutoff <- if (length(args) >= 3 && nzchar(args[3])) {
  as.numeric(args[3])
} else {
  5
}

output_path <- if (length(args) >= 4 && nzchar(args[4])) {
  args[4]
} else {
  "analysis/data/conversation_length_glmm_results.csv"
}

cat("Reading preprocessed annotations from:", parquet_path, "\n")
cat("Score selection:", score_arg, "  cutoff >=", cutoff, "\n")
cat("Output CSV:", output_path, "\n")

if (!file.exists(parquet_path)) {
  stop("Parquet file not found: ", parquet_path)
}

tbl <- read_parquet(parquet_path)
tbl <- as_tibble(tbl)

required_cols <- c("participant", "chat_index", "message_index")
missing_cols <- setdiff(required_cols, names(tbl))
if (length(missing_cols) > 0L) {
  stop(
    "Missing required columns in table: ",
    paste(missing_cols, collapse = ", ")
  )
}

score_cols_all <- grep("^score__", names(tbl), value = TRUE)
if (length(score_cols_all) == 0L) {
  stop("No score__* columns found in the preprocessed table.")
}

if (identical(toupper(score_arg), "ALL")) {
  score_cols <- score_cols_all
} else {
  if (!score_arg %in% score_cols_all) {
    stop("Requested score column not found among score__* columns: ", score_arg)
  }
  score_cols <- score_arg
}

# Construct a conversation identifier. Here we treat each distinct
# (participant, chat_index) pair as a conversation.
tbl <- tbl |>
  arrange(participant, chat_index, message_index) |>
  mutate(
    conversation_id = interaction(participant, chat_index, drop = TRUE)
  )

cat("Total rows in message-level table:", nrow(tbl), "\n")

conv_lengths <- tbl |>
  group_by(conversation_id, participant) |>
  summarise(
    n_messages = n(),
    .groups = "drop"
  ) |>
  mutate(
    participant = factor(participant),
    n_messages = as.integer(n_messages)
  )

if (nrow(conv_lengths) == 0L) {
  stop("No conversations discovered in the preprocessed table.")
}

results <- list()

extract_coef_row <- function(
  fit,
  term_name,
  annotation_id,
  score_column,
  model_type
) {
  summ <- summary(fit)
  coefs <- coef(summ)
  if (is.null(coefs) || !is.matrix(coefs)) {
    return(NULL)
  }
  if (!(term_name %in% rownames(coefs))) {
    return(NULL)
  }
  row <- coefs[term_name, , drop = TRUE]
  estimate <- as.numeric(row[["Estimate"]])
  se <- as.numeric(row[["Std. Error"]])
  z_col <- intersect(c("z value", "t value"), colnames(coefs))
  if (length(z_col) == 0L) {
    z_value <- NA_real_
  } else {
    z_value <- as.numeric(row[[z_col[1L]]])
  }
  p_cols <- grep("^Pr", colnames(coefs), value = TRUE)
  if (length(p_cols) == 0L) {
    p_value <- NA_real_
  } else {
    p_value <- as.numeric(row[[p_cols[1L]]])
  }
  data.frame(
    annotation_id = annotation_id,
    score_column = score_column,
    model_type = model_type,
    predictor = term_name,
    beta = estimate,
    se = se,
    z_value = z_value,
    p_value = p_value,
    stringsAsFactors = FALSE
  )
}

for (score_col in score_cols) {
  cat("\n=== Processing annotation score column:", score_col, "===\n")
  annotation_id <- sub("^score__", "", score_col)

  tbl_ann <- tbl |>
    mutate(is_positive = .data[[score_col]] >= cutoff)

  conv_ann <- tbl_ann |>
    group_by(conversation_id, participant) |>
    summarise(
      n_pos = sum(is_positive, na.rm = TRUE),
      first_pos_index = if_else(
        any(is_positive, na.rm = TRUE),
        min(message_index[is_positive], na.rm = TRUE),
        NA_integer_
      ),
      last_pos_index = if_else(
        any(is_positive, na.rm = TRUE),
        max(message_index[is_positive], na.rm = TRUE),
        NA_integer_
      ),
      .groups = "drop"
    )

  conv_model <- conv_lengths |>
    left_join(
      conv_ann,
      by = c("conversation_id", "participant")
    ) |>
    mutate(
      n_pos = replace_na(n_pos, 0),
      first_pos_index = as.numeric(first_pos_index),
      last_pos_index = as.numeric(last_pos_index)
    ) |>
    filter(!is.na(n_messages) & n_messages > 0L)

  if (nrow(conv_model) == 0L) {
    cat("No conversations with usable data for score column:", score_col, "\n")
    next
  }

  cat("Conversations in model for", score_col, ":", nrow(conv_model), "\n")

  # GLMM with participant random intercept.
  glmm_formula <- (
    n_messages ~ n_pos +
      first_pos_index +
      last_pos_index +
      (1 | participant)
  )
  cat("Fitting GLMM (random participant intercept)...\n")
  glmm_fit <- try(
    glmer(
      formula = glmm_formula,
      data = conv_model,
      family = poisson(link = "log"),
      control = glmerControl(optimizer = "bobyqa")
    ),
    silent = TRUE
  )

  if (!inherits(glmm_fit, "try-error")) {
    for (term in c("n_pos", "first_pos_index", "last_pos_index")) {
      row <- extract_coef_row(
        glmm_fit,
        term_name = term,
        annotation_id = annotation_id,
        score_column = score_col,
        model_type = "glmm_random_intercept"
      )
      if (!is.null(row)) {
        results[[length(results) + 1L]] <- row
      }
    }
  } else {
    cat("GLMM fit failed for", score_col, ":\n")
    print(glmm_fit)
  }

  # GLM with participant fixed effects.
  glm_formula <- (
    n_messages ~ n_pos +
      first_pos_index +
      last_pos_index +
      participant
  )
  cat("Fitting GLM (participant fixed effects)...\n")
  glm_fit <- try(
    glm(
      formula = glm_formula,
      data = conv_model,
      family = poisson(link = "log")
    ),
    silent = TRUE
  )

  if (!inherits(glm_fit, "try-error")) {
    for (term in c("n_pos", "first_pos_index", "last_pos_index")) {
      row <- extract_coef_row(
        glm_fit,
        term_name = term,
        annotation_id = annotation_id,
        score_column = score_col,
        model_type = "glm_participant_fixed"
      )
      if (!is.null(row)) {
        results[[length(results) + 1L]] <- row
      }
    }
  } else {
    cat("GLM fit with participant fixed effects failed for", score_col, ":\n")
    print(glm_fit)
  }
}

if (length(results) == 0L) {
  cat("\nNo model results were produced; CSV will not be written.\n")
} else {
  results_df <- bind_rows(results)
  dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
  write.csv(results_df, file = output_path, row.names = FALSE)
  cat("\nWrote model coefficient summary to:", output_path, "\n")
}

cat("\nDone.\n")
