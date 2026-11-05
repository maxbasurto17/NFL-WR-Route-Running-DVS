# =============================================================================
# Train_Incomplete_EP_Model.R
#
# Trains and validates the expected-points (EP) state model that All_WR_Cov_Eff.R
# uses to value an INCOMPLETE pass:
#
#     dEPA_incomplete = EP(state after the incompletion) - EP(current state)
#
# Model:  EP = f(down, yards to go, yardline_100), a GAM fit on the plays in
#         full_data.csv (one row per play, target = the play's pre-snap
#         `expected_points`).
#
# Outputs (written to CONFIG$data_dir):
#   incomplete_ep_model.rds                 <- loaded by All_WR_Cov_Eff.R
#   incomplete_ep_validation_cv.csv         <- out-of-fold accuracy by down vs a baseline
#   incomplete_ep_validation_delta.csv      <- sanity checks on the dEPA_incomplete it implies
#
# How to run:   Rscript Train_Incomplete_EP_Model.R     (then run All_WR_Cov_Eff.R)
#
# Everything lives in CONFIG (constant) or in functions; the only top-level
# statement that executes anything is the final call to main().
#
# NOTE: the feature construction (yardline clamp, yards-to-go capped at the
# yardline and at yd_cap) is duplicated in All_WR_Cov_Eff.R's make_ep_fn();
# yd_cap travels inside the saved .rds so the two cannot drift apart silently.
# =============================================================================


# =============================================================================
# 0. CONFIG (global constant)
# =============================================================================

CONFIG <- list(
  data_dir        = "~/Documents/Football_R/CMSAC--Field Value",
  tracking_file   = "full_data.csv",
  model_file      = "incomplete_ep_model.rds",
  validation_stem = "incomplete_ep_validation",   # -> <stem>_cv.csv, <stem>_delta.csv

  # Candidate column names in full_data.csv (first match wins)
  state_cols  = list(down = "down", yards_to_go = "yards_to_go"),
  goal_line_x = 110,     # offense's goal line on the 0-120 field; yardline_100 = goal_line_x - los_x_std

  # Model
  yd_cap     = 25,       # yards to go is capped at this (and at the yardline)
  k_yardline = 8,        # basis size of s(yardline_100) within each down
  k_yd       = 5,        # basis size of s(yards to go) within each down

  # Validation: K-fold CV grouped by game (no play from a held-out game is ever trained on)
  n_folds = 5,
  seed    = 2026,

  min_plays_per_down = 40   # stop if any down has fewer plays than this
)


# =============================================================================
# 1. HELPERS
# =============================================================================

load_packages <- function() {
  pkgs <- c("dplyr", "readr", "tibble", "mgcv")
  missing <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing) > 0) stop("Please install missing packages: ", paste(missing, collapse = ", "))
  suppressPackageStartupMessages(invisible(lapply(pkgs, library, character.only = TRUE)))
}

resolve_state_cols <- function(raw_names, cfg) {
  pick <- function(cands) {
    hit <- cands[cands %in% raw_names]
    if (length(hit) > 0) hit[1] else NA_character_
  }
  c(down = pick(cfg$state_cols$down), yards_to_go = pick(cfg$state_cols$yards_to_go))
}

# One row per play with the columns the model needs.
load_play_state <- function(cfg) {
  path <- file.path(path.expand(cfg$data_dir), cfg$tracking_file)
  if (!file.exists(path)) stop("Tracking file not found: ", path)

  message("Reading play state from ", path, " ...")
  raw <- readr::read_csv(
    path,
    col_select = dplyr::any_of(c("game_id", "play_id", "los_x_std", "expected_points",
                                 unlist(cfg$state_cols))),
    show_col_types = FALSE, progress = FALSE
  )

  src <- resolve_state_cols(names(raw), cfg)
  need <- setdiff(c("game_id", "play_id", "los_x_std", "expected_points"), names(raw))
  if (anyNA(src) || length(need) > 0) {
    stop("full_data.csv is missing needed columns: ",
         paste(c(need, names(src)[is.na(src)]), collapse = ", "),
         ". Edit CONFIG$state_cols if they are named differently.")
  }

  raw %>%
    dplyr::rename(down = dplyr::all_of(src[["down"]]), yards_to_go = dplyr::all_of(src[["yards_to_go"]])) %>%
    dplyr::distinct(game_id, play_id, .keep_all = TRUE) %>%
    dplyr::mutate(yardline_100 = pmin(pmax(cfg$goal_line_x - los_x_std, 1), 99)) %>%
    dplyr::filter(down %in% 1:4, !is.na(yards_to_go), !is.na(yardline_100), !is.na(expected_points)) %>%
    dplyr::mutate(down_f = factor(down, levels = 1:4),
                  yd = pmin(pmax(yards_to_go, 1), yardline_100, cfg$yd_cap))
}

fit_ep_model <- function(d, cfg) {
  f <- stats::as.formula(sprintf(
    "expected_points ~ down_f + s(yardline_100, by = down_f, k = %d) + s(yd, by = down_f, k = %d)",
    cfg$k_yardline, cfg$k_yd))
  mgcv::gam(f, data = d, method = "REML")
}

# Same feature construction as make_ep_fn() in All_WR_Cov_Eff.R.
ep_predict <- function(model, yd_cap, down, yards_to_go, yardline_100) {
  yl <- pmin(pmax(yardline_100, 1), 99)
  nd <- data.frame(down_f = factor(down, levels = 1:4),
                   yardline_100 = yl,
                   yd = pmin(pmax(yards_to_go, 1), yl, yd_cap))
  as.numeric(predict(model, newdata = nd, type = "response"))
}


# =============================================================================
# 2. VALIDATION
# =============================================================================

# Out-of-fold predictions, folds grouped by game.
cross_validate <- function(d, cfg) {
  set.seed(cfg$seed)
  games <- unique(d$game_id)
  fold_of_game <- sample(rep(seq_len(cfg$n_folds), length.out = length(games)))
  d$fold <- fold_of_game[match(d$game_id, games)]
  d$oof_model    <- NA_real_
  d$oof_baseline <- NA_real_

  for (k in seq_len(cfg$n_folds)) {
    train <- d[d$fold != k, , drop = FALSE]
    test  <- d$fold == k
    m <- fit_ep_model(train, cfg)
    d$oof_model[test] <- ep_predict(m, cfg$yd_cap, d$down[test], d$yards_to_go[test], d$yardline_100[test])
    base <- tapply(train$expected_points, train$down_f, mean)       # baseline: mean EP of that down
    d$oof_baseline[test] <- unname(base[as.character(d$down[test])])
    message(sprintf("  fold %d/%d done", k, cfg$n_folds))
  }
  d
}

accuracy_row <- function(y, p) {
  tibble::tibble(n = length(y),
                 rmse = sqrt(mean((y - p)^2)),
                 mae  = mean(abs(y - p)),
                 r2   = 1 - sum((y - p)^2) / sum((y - mean(y))^2))
}

cv_table <- function(d) {
  groups <- c(list(all = rep(TRUE, nrow(d))),
              stats::setNames(lapply(1:4, function(k) d$down == k), paste0("down_", 1:4)))
  dplyr::bind_rows(lapply(names(groups), function(g) {
    i <- groups[[g]]
    dplyr::bind_rows(
      dplyr::mutate(accuracy_row(d$expected_points[i], d$oof_model[i]),    subset = g, predictor = "gam_oof"),
      dplyr::mutate(accuracy_row(d$expected_points[i], d$oof_baseline[i]), subset = g, predictor = "down_mean_baseline")
    )
  })) %>% dplyr::select(subset, predictor, n, rmse, mae, r2)
}

# dEPA if incomplete, from the model alone. Mirrors compute_incomplete_epa() in
# All_WR_Cov_Eff.R (1st-3rd: down + 1; 4th: turnover on downs at the spot).
implied_incomplete_delta <- function(model, d, cfg) {
  ep <- function(down, yd, yl) ep_predict(model, cfg$yd_cap, down, yd, yl)
  now  <- ep(d$down, d$yards_to_go, d$yardline_100)
  nxt  <- rep(NA_real_, nrow(d))
  d123 <- d$down <= 3
  nxt[d123] <- ep(d$down[d123] + 1, d$yards_to_go[d123], d$yardline_100[d123])
  d4 <- d$down == 4
  opp <- 100 - d$yardline_100[d4]
  nxt[d4] <- -ep(rep(1, sum(d4)), pmin(10, opp), opp)
  nxt - now
}

delta_checks <- function(model, d, cfg) {
  d$delta <- implied_incomplete_delta(model, d, cfg)
  # 3rd -> 4th down is an extrapolation: the data only hold 4th downs where a team went for it
  q95_4th   <- stats::quantile(d$yards_to_go[d$down == 4], 0.95, names = FALSE)
  extrap_3rd <- mean(d$yards_to_go[d$down == 3] > q95_4th)

  out <- d %>%
    dplyr::group_by(down) %>%
    dplyr::summarize(n_plays = dplyr::n(),
                     mean_delta   = mean(delta),
                     p05_delta    = unname(stats::quantile(delta, 0.05)),
                     p95_delta    = unname(stats::quantile(delta, 0.95)),
                     share_negative = mean(delta < 0),     # an incompletion should almost always cost EP
                     .groups = "drop") %>%
    dplyr::mutate(extrapolated_share = dplyr::case_when(
      down == 3 ~ extrap_3rd,   # share of 3rd-down plays whose distance exceeds the 95th pct of 4th-down distances
      TRUE ~ NA_real_))
  attr(out, "q95_4th_yards_to_go") <- q95_4th
  out
}


# =============================================================================
# 3. MAIN
# =============================================================================

main <- function(cfg = CONFIG) {
  load_packages()
  out_dir <- path.expand(cfg$data_dir)

  d <- load_play_state(cfg)
  n_by_down <- table(d$down_f)
  message(sprintf("Plays: %d | by down 1/2/3/4: %s", nrow(d), paste(as.integer(n_by_down), collapse = "/")))
  if (any(n_by_down < cfg$min_plays_per_down)) {
    stop("Too few plays on some down (min ", cfg$min_plays_per_down, "): ",
         paste(names(n_by_down), as.integer(n_by_down), sep = "=", collapse = ", "))
  }

  message("Cross-validating (", cfg$n_folds, " folds grouped by game)...")
  d_cv <- cross_validate(d, cfg)
  cv <- cv_table(d_cv)
  print(as.data.frame(cv), digits = 3, row.names = FALSE)

  gam_all  <- cv[cv$subset == "all" & cv$predictor == "gam_oof", ]
  base_all <- cv[cv$subset == "all" & cv$predictor == "down_mean_baseline", ]
  if (gam_all$rmse >= base_all$rmse) warning("GAM is not beating the by-down mean out of fold.", call. = FALSE)

  message("Fitting final model on all plays...")
  model <- fit_ep_model(d, cfg)

  delta <- delta_checks(model, d, cfg)
  message("Implied incomplete-pass dEPA by down:")
  print(as.data.frame(delta), digits = 3, row.names = FALSE)

  saved <- list(
    model = model, yd_cap = cfg$yd_cap, goal_line_x = cfg$goal_line_x,
    formula = formula(model), n_train = nrow(d),
    n_by_down = as.integer(n_by_down),
    cv_rmse = gam_all$rmse, cv_r2 = gam_all$r2,
    cv = cv, delta_check = delta,
    trained_at = format(Sys.time(), "%Y-%m-%d %H:%M")
  )
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
  model_path <- file.path(out_dir, cfg$model_file)
  saveRDS(saved, model_path)
  readr::write_csv(cv,    file.path(out_dir, paste0(cfg$validation_stem, "_cv.csv")))
  readr::write_csv(delta, file.path(out_dir, paste0(cfg$validation_stem, "_delta.csv")))

  message(sprintf("Saved model: %s\n  out-of-fold RMSE %.3f (baseline %.3f) | R2 %.3f",
                  model_path, gam_all$rmse, base_all$rmse, gam_all$r2))
  invisible(model_path)
}

main()
