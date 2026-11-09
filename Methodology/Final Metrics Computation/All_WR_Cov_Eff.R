# =============================================================================
# All_WR_Cov_Eff.R
#
# Play-by-play coverage efficiency for EVERY WR in full_df.
#
#   * One WR at a time (bounded RAM).
#   * Each WR's play-level rows are APPENDED to a single final CSV.
#   * Memory is released after every WR (and workers are periodically restarted).
#   * Resumable: finished WRs are logged and skipped on re-run.
#
# How to run:
#   Rscript All_WR_Cov_Eff.R            # or source() it and call main()
#
# Everything lives in CONFIG (constant) or in functions; the only top-level
# statement that executes anything is the final call to main().
#
# MEMORY DESIGN (the big change vs. the Rmd):
#   The Rmd saved the full 120 x 54 grid for every frame of every play and
#   then re-read it all. Only cells inside the target-zone envelope ever feed
#   the efficiency numbers, so here each play is processed end-to-end inside
#   the worker: pitch control -> GAM prediction -> efficiency, on in-zone cells
#   only. Only the tiny frame-level summary leaves the worker. The math is the
#   same; every grid cell is computed independently, so dropping out-of-zone
#   cells does not change any in-zone value.
# =============================================================================


# =============================================================================
# 0. CONFIG (global constant)
# =============================================================================

CONFIG <- list(
  # ---- paths ---------------------------------------------------------------
  data_dir      = "~/Documents/Football_R/CMSAC--Field Value",
  tracking_file = "full_data.csv",
  gam_file      = "gam_field_value_model.rds",
  envelope_file = "optimal_envelope_parameters.rds",  # looked for in data_dir, then working dir
  # Incomplete-pass EP model, produced by Train_Incomplete_EP_Model.R (run that first)
  ep_model_file = "incomplete_ep_model.rds",

  output_dir    = "~/Documents/Football_R/CMSAC--Field Value/PBP_Efficiency",
  output_pbp    = "all_wr_play_coverage_efficiency.csv",   # FINAL play-by-play file
  output_frames = "all_wr_frame_coverage_efficiency.csv",  # only written if save_frame_level
  progress_file = "all_wr_progress.csv",                   # one row per finished WR

  # ---- behavior ------------------------------------------------------------
  save_frame_level      = TRUE,  # also append frame-by-frame rows to output_frames
  resume                = TRUE,   # skip WRs already in progress_file; FALSE = start fresh (deletes outputs)
  wr_limit              = NULL,   # e.g. 3 to test on the first 3 WRs; NULL = all
  n_workers             = 8,      # 1 = no parallelism
  restart_workers_every = 20,     # restart worker processes every N WRs to release worker RAM

  # ---- field grid ----------------------------------------------------------
  # goal_line_x: x of the offense's goal line on the 0-120 standardized field
  # (offense always moves toward +x). yardline_100 = goal_line_x - los_x_std.
  pitch = list(length = 120.0, width = 53.3, n_cols = 120, n_rows = 54, goal_line_x = 110),

  # ---- value of a throw (RTE) and plain field control (OFC) -----------------
  # Per grid cell, with p = ppcf_att^completion_prob_power (completion weight):
  #   value = p * sign(dEPA_c) * sqrt(|dEPA_c|)  +  (1 - p) * dEPA_inc
  #   dEPA_c   = GAM post-pass EP - pre-snap EP         (completion branch, unchanged)
  #   dEPA_inc = EP(state after an incompletion) - EP(current state)   (linear, no sqrt)
  # Incomplete state: 1st-3rd down -> same spot / same distance, down + 1.
  #                   4th down     -> see fourth_down below.
  # NOTE: changing anything in `value` changes the output; re-run with resume = FALSE.
  value = list(
    include_incomplete    = TRUE,   # FALSE = legacy completion-only RTE
    completion_prob_power = 1,      # legacy RTE used ppcf_att^2 as the completion weight
    fourth_down           = "turnover_on_downs",  # or "zero" = no incomplete term on 4th down
    ofc_power             = 1,      # OFC cell value = ppcf_att^ofc_power (1 = plain control)
    ep_fn                 = NULL    # optional function(down, yards_to_go, yardline_100) -> EP
                                    # (e.g. wrapping nflfastR). NULL = load ep_model_file.
  ),

  # Candidate column names in full_data.csv for the play state (first match wins)
  state_cols = list(down = "down", yards_to_go = "yards_to_go"),

  # Columns the output must contain (resume guard against an old-schema output file)
  output_ofc_cols = c("peak_frame_ofc", "avg_play_ofc", "final_frame_ofc"),

  # ---- tracking columns to read (everything else is dropped to save RAM) ---
  tracking_cols = c(
    "game_id", "play_id", "frame_id", "nfl_id", "player_name", "player_position",
    "player_side", "phase", "x_std", "y_std", "s", "a", "dir_std",
    "los_x_std", "ball_land_y_std", "expected_points", "dropback_distance"
  ),
  optional_cols = "a"   # fine if missing (acceleration falls back to 8)
)


# =============================================================================
# 1. SETUP HELPERS
# =============================================================================

load_packages <- function() {
  pkgs <- c("dplyr", "readr", "tibble", "future", "furrr", "mgcv")
  missing <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing) > 0) {
    stop("Please install missing packages: ", paste(missing, collapse = ", "))
  }
  suppressPackageStartupMessages(invisible(lapply(pkgs, library, character.only = TRUE)))
}

resolve_paths <- function(cfg) {
  in_dir  <- path.expand(cfg$data_dir)
  out_dir <- path.expand(cfg$output_dir)
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
  list(
    tracking = file.path(in_dir, cfg$tracking_file),
    gam      = file.path(in_dir, cfg$gam_file),
    ep_model = file.path(in_dir, cfg$ep_model_file),
    envelope =c(file.path(in_dir, cfg$envelope_file), cfg$envelope_file),
    pbp      = file.path(out_dir, cfg$output_pbp),
    frames   = file.path(out_dir, cfg$output_frames),
    progress = file.path(out_dir, cfg$progress_file)
  )
}

default_envelope_model <- function() {
  list(
    parameters = tibble::tibble(
      pass_depth_bin   = factor(c("Short", "Medium", "Deep"), levels = c("Short", "Medium", "Deep")),
      backward_cushion = c(1.0, 1.5, 2.0),
      corridor_width   = c(2.5, 3.5, 4.5),
      frames_ahead     = c(10, 15, 20)
    ),
    breaks = c(-Inf, 4.5, 12.5, Inf),
    labels = c("Short", "Medium", "Deep")
  )
}

load_envelope_model <- function(candidate_paths) {
  hit <- candidate_paths[file.exists(candidate_paths)]
  if (length(hit) > 0) {
    message("Envelope model: ", hit[1])
    return(readRDS(hit[1]))
  }
  warning("optimal_envelope_parameters.rds NOT found in: ",
          paste(candidate_paths, collapse = " | "),
          "\n  Falling back to the default envelope parameters. ",
          "Results will differ from your tuned envelope!", call. = FALSE)
  default_envelope_model()
}

append_csv <- function(df, path) {
  if (is.null(df) || nrow(df) == 0) return(invisible(FALSE))
  exists_already <- file.exists(path)
  readr::write_csv(df, path, append = exists_already, col_names = !exists_already)
  invisible(TRUE)
}


# =============================================================================
# 2. DATA PREPARATION (runs once)
# =============================================================================

compute_player_profiles <- function(df) {
  has_a <- "a" %in% names(df)
  df %>%
    dplyr::filter(!is.na(nfl_id), !is.na(s)) %>%
    dplyr::group_by(nfl_id) %>%
    dplyr::summarize(
      vmax = pmax(pmin(unname(quantile(s, 0.99, na.rm = TRUE)), 12.5), 7.0),
      acceleration = if (has_a) {
        pmax(pmin(unname(quantile(a, 0.99, na.rm = TRUE)), 10.0), 5.0)
      } else 8.0,
      .groups = "drop"
    )
}

# Reads full_df once, keeps only needed columns / rows, precomputes velocity.
load_inputs <- function(cfg, paths) {
  if (!file.exists(paths$tracking)) stop("Tracking file not found: ", paths$tracking)
  if (!file.exists(paths$gam))      stop("GAM model file not found: ", paths$gam)

  # Fail fast (before the big read) if the incomplete-pass EP model is needed but missing
  ep_fn <- if (isTRUE(cfg$value$include_incomplete)) resolve_ep_fn(cfg, paths) else NULL

  message("Reading tracking data...")
  raw <- readr::read_csv(
    paths$tracking,
    col_select = dplyr::any_of(c(cfg$tracking_cols, unlist(cfg$state_cols))),
    show_col_types = FALSE, progress = FALSE
  )

  required <- setdiff(cfg$tracking_cols, cfg$optional_cols)
  absent <- setdiff(required, names(raw))
  if (length(absent) > 0) stop("full_data.csv is missing columns: ", paste(absent, collapse = ", "))

  # Down / yards-to-go -> standard names `down`, `yards_to_go` (needed for the incomplete term)
  state_src <- resolve_state_cols(names(raw), cfg)
  if (isTRUE(cfg$value$include_incomplete)) {
    if (anyNA(state_src)) {
      stop("include_incomplete = TRUE needs down and yards-to-go columns in full_data.csv. ",
           "Looked for: down = {", paste(cfg$state_cols$down, collapse = ", "), "}, ",
           "yards to go = {", paste(cfg$state_cols$yards_to_go, collapse = ", "), "}. ",
           "Edit CONFIG$state_cols or set value$include_incomplete = FALSE.")
    }
    raw <- dplyr::rename(raw, down = dplyr::all_of(state_src[["down"]]),
                              yards_to_go = dplyr::all_of(state_src[["yards_to_go"]]))
  }

  # WR roster (as specified)
  all_wrs <- raw %>%
    dplyr::filter(player_position == "WR") %>%
    dplyr::select(nfl_id, player_name) %>%
    dplyr::distinct() %>%
    dplyr::distinct(nfl_id, .keep_all = TRUE) %>%   # guard: one row per nfl_id
    dplyr::arrange(nfl_id)

  # Athletic profiles use every phase / every player, same as the Rmd
  profiles <- compute_player_profiles(raw)

  tracking <- raw %>%
    dplyr::filter(phase == "pre_pass", !is.na(nfl_id)) %>%
    dplyr::left_join(profiles, by = "nfl_id") %>%
    dplyr::mutate(
      vmax         = dplyr::coalesce(vmax, 11.0),
      acceleration = dplyr::coalesce(acceleration, 8.0),
      player_side  = tolower(player_side),
      is_qb        = tolower(player_position) == "qb",
      vx           = s * sin(dir_std * pi / 180),
      vy           = s * cos(dir_std * pi / 180)
    ) %>%
    dplyr::filter(!is.na(x_std), !is.na(y_std), !is.na(vx), !is.na(vy)) %>%
    dplyr::select(game_id, play_id, frame_id, nfl_id, player_side, is_qb,
                  x_std, y_std, s, vx, vy, vmax, acceleration,
                  los_x_std, ball_land_y_std, expected_points, dropback_distance,
                  dplyr::any_of(c("down", "yards_to_go")))

  # Per-play value of an incompletion (one number per play; same for every grid cell,
  # because an incompletion always returns the ball to the line of scrimmage).
  if (isTRUE(cfg$value$include_incomplete)) {
    play_state <- build_play_state(tracking, cfg)
    play_state <- compute_incomplete_epa(play_state, ep_fn, cfg$value$fourth_down)
    tracking <- tracking %>%
      dplyr::left_join(play_state[, c("game_id", "play_id", "epa_incomplete")],
                       by = c("game_id", "play_id")) %>%
      dplyr::select(-dplyr::any_of(c("down", "yards_to_go")))
    rm(play_state, ep_fn)
  } else {
    tracking$epa_incomplete <- 0
    tracking <- dplyr::select(tracking, -dplyr::any_of(c("down", "yards_to_go")))
  }

  # Which plays each WR appears in (pre_pass)
  play_index <- tracking %>%
    dplyr::filter(nfl_id %in% all_wrs$nfl_id) %>%
    dplyr::distinct(nfl_id, game_id, play_id)

  rm(raw, profiles)
  invisible(gc(verbose = FALSE))

  list(tracking = tracking, play_index = play_index, all_wrs = all_wrs)
}


# =============================================================================
# 2b. VALUE OF AN INCOMPLETE PASS (play-level, computed once)
# =============================================================================

resolve_state_cols <- function(raw_names, cfg) {
  pick <- function(cands) {
    hit <- cands[cands %in% raw_names]
    if (length(hit) > 0) hit[1] else NA_character_
  }
  c(down = pick(cfg$state_cols$down), yards_to_go = pick(cfg$state_cols$yards_to_go))
}

# One row per play: down, distance, spot, pre-snap EP.
build_play_state <- function(tracking, cfg) {
  tracking %>%
    dplyr::select(game_id, play_id, down, yards_to_go, los_x_std, expected_points) %>%
    dplyr::distinct(game_id, play_id, .keep_all = TRUE) %>%
    dplyr::mutate(yardline_100 = pmin(pmax(cfg$pitch$goal_line_x - los_x_std, 1), 99))
}

# EP(down, yards to go, yardline_100): trained and validated in
# Train_Incomplete_EP_Model.R, loaded here. Only DIFFERENCES of this model enter
# the output (next state - current state), so its overall bias cancels.
# The feature construction below must match the trainer (it is stored in the
# saved object as `yd_cap`).
load_ep_model <- function(path) {
  if (!file.exists(path)) {
    stop("Incomplete-pass EP model not found: ", path,
         "\n  Run Train_Incomplete_EP_Model.R first (or supply CONFIG$value$ep_fn).")
  }
  obj <- readRDS(path)
  if (!all(c("model", "yd_cap") %in% names(obj))) {
    stop("Unexpected format in ", path, " (expected a list with `model` and `yd_cap`).")
  }
  message(sprintf("EP model: %s (trained %s on %d plays; CV RMSE %.3f)", path,
                  obj$trained_at, obj$n_train, obj$cv_rmse))
  obj
}

make_ep_fn <- function(ep_obj) {
  function(down, yards_to_go, yardline_100) {
    yl <- pmin(pmax(yardline_100, 1), 99)
    nd <- data.frame(down_f = factor(down, levels = 1:4),
                     yardline_100 = yl,
                     yd = pmin(pmax(yards_to_go, 1), yl, ep_obj$yd_cap))
    as.numeric(predict(ep_obj$model, newdata = nd, type = "response"))
  }
}

resolve_ep_fn <- function(cfg, paths) {
  if (is.function(cfg$value$ep_fn)) {
    message("EP model: user-supplied ep_fn")
    return(cfg$value$ep_fn)
  }
  make_ep_fn(load_ep_model(paths$ep_model))
}

# dEPA if the pass is NOT completed = EP(next state) - EP(current state).
#   1st-3rd down: same spot, same distance, down + 1.
#   4th down "turnover_on_downs": defense takes over at the spot, 1st & 10
#     (goal-to-go if inside the 10): EP_next = -EP_defense(1st, min(10, yds to goal), 100 - yardline_100).
#   4th down "zero": no incomplete term.
compute_incomplete_epa <- function(play_state, ep_fn, fourth_down = "turnover_on_downs") {
  n  <- nrow(play_state)
  ok <- !is.na(play_state$down) & play_state$down %in% 1:4 &
        !is.na(play_state$yards_to_go) & !is.na(play_state$yardline_100)

  ep_now  <- ep_fn(play_state$down, play_state$yards_to_go, play_state$yardline_100)
  ep_next <- rep(NA_real_, n)

  d123 <- ok & play_state$down <= 3
  if (any(d123)) {
    ep_next[d123] <- ep_fn(play_state$down[d123] + 1, play_state$yards_to_go[d123],
                           play_state$yardline_100[d123])
  }

  d4 <- ok & play_state$down == 4
  if (any(d4)) {
    if (identical(fourth_down, "turnover_on_downs")) {
      opp_yl <- 100 - play_state$yardline_100[d4]
      ep_next[d4] <- -ep_fn(rep(1, sum(d4)), pmin(10, opp_yl), opp_yl)
    } else {
      ep_next[d4] <- ep_now[d4]   # -> incomplete term = 0
    }
  }

  out <- ep_next - ep_now
  n_missing <- sum(is.na(out))
  if (n_missing > 0) {
    warning(n_missing, " plays have no usable down / distance / spot; their incomplete term is set to 0.",
            call. = FALSE)
    out[is.na(out)] <- 0
  }
  message(sprintf("Incomplete-pass dEPA: mean %.3f | by down: %s", mean(out),
                  paste(sprintf("%d=%.2f", 1:4,
                                vapply(1:4, function(k) mean(out[play_state$down %in% k]), numeric(1))),
                        collapse = ", ")))
  play_state$epa_incomplete <- out
  play_state
}

# Value of throwing to one cell (vectorized). Completion branch is the legacy RTE term.
compute_cell_value <- function(ppcf_att, epa_complete, epa_incomplete, value_cfg) {
  w <- ppcf_att^value_cfg$completion_prob_power
  complete_term <- epa_complete * w
  if (!isTRUE(value_cfg$include_incomplete)) return(complete_term)
  return(complete_term + (pmax(1 - w, 0)^2) * 0.75*epa_incomplete)
}


# =============================================================================
# 3. GRID + TARGET ENVELOPE
# =============================================================================

build_nfl_grid <- function(pitch_length = 120.0, pitch_width = 53.3, n_cols = 120, n_rows = 54) {
  xs <- seq(0, pitch_length, length.out = n_cols)
  ys <- seq(0, pitch_width,  length.out = n_rows)
  list(
    x = matrix(rep(xs, each = n_rows),  nrow = n_rows, ncol = n_cols),
    y = matrix(rep(ys, times = n_cols), nrow = n_rows, ncol = n_cols)
  )
}

# Works elementwise on vectors or matrices of grid coordinates.
compute_target_envelope <- function(grid_x, grid_y, wr_x, wr_y, wr_vx, wr_vy, wr_speed,
                                    pass_depth, envelope_model) {
  depth_bin <- as.character(cut(pass_depth, breaks = envelope_model$breaks,
                                labels = envelope_model$labels))
  p <- envelope_model$parameters
  row_idx <- if (is.na(depth_bin)) integer(0) else which(as.character(p$pass_depth_bin) == depth_bin)
  row_idx <- if (length(row_idx) == 0) 1L else row_idx[1]

  cushion      <- p$backward_cushion[row_idx]
  corridor_w   <- p$corridor_width[row_idx]
  frames_ahead <- p$frames_ahead[row_idx]

  if (wr_speed > 1e-4) {
    ux <- wr_vx / wr_speed
    uy <- wr_vy / wr_speed
  } else {
    ux <- 0; uy <- 0
  }

  proj_len <- wr_speed * (frames_ahead / 10.0)   # 1 frame = 0.1s; yards

  p0_x <- wr_x - ux * cushion
  p0_y <- wr_y - uy * cushion
  p1_x <- wr_x + ux * proj_len
  p1_y <- wr_y + uy * proj_len

  seg_dx <- p1_x - p0_x
  seg_dy <- p1_y - p0_y
  denom  <- seg_dx^2 + seg_dy^2

  if (denom > 0) {
    t <- ((grid_x - p0_x) * seg_dx + (grid_y - p0_y) * seg_dy) / denom
    t <- pmin(pmax(t, 0), 1)
  } else {
    t <- 0
  }

  proj_x <- p0_x + t * seg_dx
  proj_y <- p0_y + t * seg_dy
  sqrt((grid_x - proj_x)^2 + (grid_y - proj_y)^2) <= corridor_w
}


# =============================================================================
# 4. KINEMATICS & PITCH CONTROL  (vector version of the Rmd functions)
# =============================================================================

player_time_to_intercept <- function(player, target_x, target_y,
                                     default_vmax = 11.0, default_accel = 8.0) {
  vmax         <- if (!is.null(player$vmax)) player$vmax else default_vmax
  acceleration <- if (!is.null(player$acceleration)) player$acceleration else default_accel

  dx <- target_x - player$x
  dy <- target_y - player$y

  distances <- sqrt(dx^2 + dy^2)
  distances[distances == 0 | is.na(distances)] <- 1e-8

  unit_x <- dx / distances
  unit_y <- dy / distances

  v0_proj <- player$vx * unit_x + player$vy * unit_y
  v0_proj <- pmin(v0_proj, vmax)

  t_acc <- (vmax - v0_proj) / acceleration
  d_acc <- (v0_proj * t_acc) + (0.5 * acceleration * (t_acc^2))

  tti <- numeric(length(distances))

  acc_phase_mask <- d_acc >= distances
  acc_phase_mask[is.na(acc_phase_mask)] <- FALSE
  cruise_phase_mask <- !acc_phase_mask
  cruise_phase_mask[is.na(cruise_phase_mask)] <- FALSE

  sqrt_term <- pmax(v0_proj^2 + 2 * acceleration * distances, 0)

  if (any(acc_phase_mask)) {
    tti[acc_phase_mask] <- (-v0_proj[acc_phase_mask] + sqrt(sqrt_term[acc_phase_mask])) / acceleration
  }
  if (any(cruise_phase_mask)) {
    d_cruise <- distances[cruise_phase_mask] - d_acc[cruise_phase_mask]
    tti[cruise_phase_mask] <- t_acc[cruise_phase_mask] + (d_cruise / vmax)
  }
  tti
}

estimate_nfl_pass_bounds <- function(distances) {
  list(min = pmax(distances / 23.0, 0.0), max = pmax(distances / 14.0, 0.0))
}

football_travel_time <- function(qb_x, qb_y, target_x, target_y, attacking_players) {
  distances <- sqrt((target_x - qb_x)^2 + (target_y - qb_y)^2)
  bounds <- estimate_nfl_pass_bounds(distances)

  nearest_attacker_tti <- rep(Inf, length(target_x))
  for (player in attacking_players) {
    if (abs(player$x - qb_x) < 0.1 && abs(player$y - qb_y) < 0.1) next
    nearest_attacker_tti <- pmin(nearest_attacker_tti,
                                 player_time_to_intercept(player, target_x, target_y))
  }
  pmin(pmax(nearest_attacker_tti, bounds$min), bounds$max)
}

arrival_probability <- function(t, tti, tti_sigma = 0.54) {
  exponent <- -pi / sqrt(3.0) / tti_sigma * (t - tti)
  exponent <- pmin(pmax(exponent, -50), 50)
  1.0 / (1.0 + exp(exponent))
}

control_rate <- function(t, tti, lambda_j = 3.99, tti_sigma = 0.54) {
  arrival_probability(t, tti, tti_sigma) * lambda_j
}

# grid_x / grid_y are plain vectors of the cells to evaluate (in-zone cells only).
nfl_dynamic_pitch_control <- function(attacking_players, defending_players, qb_pos,
                                      grid_x, grid_y,
                                      dt = 0.05, max_time = 8.0,
                                      lambda_att = 3.89, kappa = 1.18, tti_sigma = 0.67) {
  lambda_def <- lambda_att * kappa
  TTI_att <- lapply(attacking_players, function(p) player_time_to_intercept(p, grid_x, grid_y))
  TTI_def <- lapply(defending_players, function(p) player_time_to_intercept(p, grid_x, grid_y))

  T_ball <- football_travel_time(qb_pos$x, qb_pos$y, grid_x, grid_y, attacking_players)

  n <- length(grid_x)
  PPCF_att <- numeric(n)
  PPCF_def <- numeric(n)
  PPCF_tot <- numeric(n)

  for (t in seq(0.0, max_time, by = dt)) {
    calc_mask <- (t >= T_ball) & (PPCF_tot < 0.999)
    if (!any(calc_mask)) next

    rate_att <- Reduce("+", lapply(TTI_att, function(m) control_rate(t, m[calc_mask], lambda_att, tti_sigma)))
    rate_def <- Reduce("+", lapply(TTI_def, function(m) control_rate(t, m[calc_mask], lambda_def, tti_sigma)))

    uncontrolled <- 1.0 - PPCF_tot[calc_mask]
    d_att <- uncontrolled * rate_att * dt
    d_def <- uncontrolled * rate_def * dt

    PPCF_att[calc_mask] <- PPCF_att[calc_mask] + d_att
    PPCF_def[calc_mask] <- PPCF_def[calc_mask] + d_def
    PPCF_tot[calc_mask] <- PPCF_tot[calc_mask] + d_att + d_def
  }

  PPCF_att[grid_x < qb_pos$x] <- 0.0
  list(PPCF_att = PPCF_att, PPCF_def = PPCF_def)
}


# =============================================================================
# 5. PER-PLAY PROCESSING (runs inside workers)
# =============================================================================

# GAM is loaded ONCE per worker process and cached, instead of being
# serialized to the workers for every chunk.
get_gam_model <- function(path) {
  key <- ".allwr_gam_cache"
  cached <- if (exists(key, envir = .GlobalEnv, inherits = FALSE)) get(key, envir = .GlobalEnv) else NULL
  if (is.null(cached) || !identical(cached$path, path)) {
    cached <- list(path = path, model = readRDS(path))
    assign(key, cached, envir = .GlobalEnv)
  }
  cached$model
}

to_player_list <- function(df) {
  lapply(seq_len(nrow(df)), function(i) {
    list(id = df$nfl_id[i], x = df$x_std[i], y = df$y_std[i],
         vx = df$vx[i], vy = df$vy[i],
         vmax = df$vmax[i], acceleration = df$acceleration[i])
  })
}

# One frame -> in-zone cells with offense control probability (or NULL to skip).
compute_frame_cells <- function(fd, wr_id, gx, gy, envelope_model) {
  off <- fd[fd$player_side == "offense", , drop = FALSE]
  def <- fd[fd$player_side == "defense", , drop = FALSE]

  qb_row <- dplyr::slice(off[off$is_qb, , drop = FALSE], 1)
  if (nrow(qb_row) == 0) qb_row <- dplyr::slice(off, 1)   # same fallback as the Rmd
  if (nrow(qb_row) > 0) {
    qb_pos <- list(x = qb_row$x_std, y = qb_row$y_std)
    off <- off[off$nfl_id != qb_row$nfl_id, , drop = FALSE]
  } else {
    qb_pos <- list(x = 60, y = 26.6)
  }

  target <- off[off$nfl_id == wr_id, , drop = FALSE]
  if (nrow(target) == 0 || nrow(def) == 0) return(NULL)

  los_x <- target$los_x_std[1]
  in_zone <- compute_target_envelope(
    grid_x = gx, grid_y = gy,
    wr_x = target$x_std[1], wr_y = target$y_std[1],
    wr_vx = target$vx[1], wr_vy = target$vy[1],
    wr_speed = target$s[1],
    pass_depth = target$x_std[1] - los_x,
    envelope_model = envelope_model
  )
  if (!any(in_zone)) return(NULL)

  zx <- gx[in_zone]
  zy <- gy[in_zone]

  pc <- nfl_dynamic_pitch_control(
    attacking_players = to_player_list(target),
    defending_players = to_player_list(def),
    qb_pos = qb_pos, grid_x = zx, grid_y = zy
  )

  tibble::tibble(
    game_id = target$game_id[1], play_id = target$play_id[1], frame_id = target$frame_id[1],
    x = zx, y = zy,
    ppcf_att = pc$PPCF_att,
    in_target_zone = TRUE,
    ball_land_y_std = target$ball_land_y_std[1],
    expected_points = target$expected_points[1],
    epa_incomplete = target$epa_incomplete[1],
    los_x_std = los_x,
    dropback_distance = target$dropback_distance[1],
    pass_depth = zx - los_x
  )
}

# Whole play: pitch control for every frame -> one GAM prediction -> frame summary.
process_play <- function(play_data, wr_id, envelope_model, gam_path, pitch, value_cfg) {
  grid <- build_nfl_grid(pitch$length, pitch$width, pitch$n_cols, pitch$n_rows)
  gx <- as.vector(grid$x)
  gy <- as.vector(grid$y)

  frame_list <- split(play_data, play_data$frame_id)
  cells <- vector("list", length(frame_list))
  n_errors <- 0L
  first_error <- NA_character_

  for (i in seq_along(frame_list)) {
    res <- tryCatch(compute_frame_cells(frame_list[[i]], wr_id, gx, gy, envelope_model),
                    error = function(e) e)
    if (inherits(res, "error")) {
      n_errors <- n_errors + 1L
      if (is.na(first_error)) first_error <- conditionMessage(res)
    } else if (!is.null(res)) {
      cells[[i]] <- res
    }
  }

  cells <- dplyr::bind_rows(cells)
  if (nrow(cells) == 0) {
    return(list(frames = NULL, n_errors = n_errors, first_error = first_error))
  }

  gam_model <- get_gam_model(gam_path)
  gam_vars  <- intersect(all.vars(formula(gam_model)), names(cells))
  cells$pred_post_epa <- as.numeric(
    predict(gam_model, newdata = cells[, gam_vars, drop = FALSE], type = "response", se.fit = FALSE)
  )

  frames <- cells %>%
    dplyr::mutate(
      epa_complete = pred_post_epa - expected_points,
      cell_value   = compute_cell_value(ppcf_att, epa_complete, epa_incomplete, value_cfg),
      # zero-out cells behind the QB dropback position (same mask for RTE and OFC)
      behind_qb  = (pass_depth + dropback_distance) <= 0,
      efficiency = ifelse(behind_qb, 0, cell_value),
      ofc_cell   = ifelse(behind_qb, 0, ppcf_att^value_cfg$ofc_power)
    ) %>%
    dplyr::group_by(game_id, play_id, frame_id) %>%
    dplyr::summarize(
      total_wr_control = sum(ppcf_att, na.rm = TRUE),
      mean_wr_control  = mean(ppcf_att, na.rm = TRUE),
      frame_efficiency = sum(efficiency, na.rm = TRUE),
      frame_ofc        = sum(ofc_cell, na.rm = TRUE),
      pass_depth       = dplyr::first(pass_depth),
      .groups = "drop"
    )

  list(frames = frames, n_errors = n_errors, first_error = first_error)
}

safe_process_play <- function(play_data, wr_id, envelope_model, gam_path, pitch, value_cfg) {
  tryCatch(
    process_play(play_data, wr_id, envelope_model, gam_path, pitch, value_cfg),
    error = function(e) list(frames = NULL, n_errors = 1L, first_error = conditionMessage(e))
  )
}


# =============================================================================
# 6. PER-WR PROCESSING
# =============================================================================

summarize_play_level <- function(frame_level) {
  frame_level %>%
    dplyr::arrange(game_id, play_id, frame_id) %>%
    dplyr::group_by(game_id, play_id) %>%
    dplyr::summarize(
      avg_play_efficiency    = mean(frame_efficiency, na.rm = TRUE),
      peak_frame_efficiency  = max(frame_efficiency, na.rm = TRUE),
      final_frame_efficiency = dplyr::last(frame_efficiency),
      avg_play_ofc           = mean(frame_ofc, na.rm = TRUE),
      peak_frame_ofc         = max(frame_ofc, na.rm = TRUE),
      final_frame_ofc        = dplyr::last(frame_ofc),
      total_frames_analyzed  = dplyr::n_distinct(frame_id),
      .groups = "drop"
    )
}

process_wr <- function(wr_id, wr_name, tracking, play_index, envelope_model, paths, cfg) {
  keys <- play_index[play_index$nfl_id == wr_id, c("game_id", "play_id")]

  empty <- list(play_level = NULL, frame_level = NULL, n_candidate_plays = nrow(keys),
                n_errors = 0L, first_error = NA_character_)
  if (nrow(keys) == 0) return(empty)

  plays <- tracking %>%
    dplyr::semi_join(keys, by = c("game_id", "play_id")) %>%
    dplyr::filter(player_side == "defense" | nfl_id == wr_id | is_qb) %>%
    dplyr::group_by(game_id, play_id) %>%
    dplyr::group_split()

  results <- furrr::future_map(
    plays, safe_process_play,
    wr_id = wr_id, envelope_model = envelope_model,
    gam_path = paths$gam, pitch = cfg$pitch,
    value_cfg = cfg$value[c("include_incomplete", "completion_prob_power", "ofc_power")],
    .options = furrr::furrr_options(packages = c("dplyr", "mgcv"))
  )
  rm(plays)

  frame_level <- dplyr::bind_rows(lapply(results, `[[`, "frames"))
  n_errors    <- sum(vapply(results, function(r) as.integer(r$n_errors), integer(1)))
  errs        <- vapply(results, function(r) as.character(r$first_error), character(1))
  first_error <- if (any(!is.na(errs))) errs[!is.na(errs)][1] else NA_character_
  rm(results)

  if (nrow(frame_level) == 0) {
    empty$n_errors <- n_errors
    empty$first_error <- first_error
    return(empty)
  }

  frame_level <- frame_level %>%
    dplyr::mutate(nfl_id = wr_id, player_name = wr_name, .before = 1)

  play_level <- summarize_play_level(frame_level) %>%
    dplyr::mutate(nfl_id = wr_id, player_name = wr_name, .before = 1)

  list(play_level = play_level, frame_level = frame_level,
       n_candidate_plays = nrow(keys), n_errors = n_errors, first_error = first_error)
}


# =============================================================================
# 7. WORKERS, PROGRESS LOG, MAIN
# =============================================================================

restart_workers <- function(n_workers) {
  future::plan(future::sequential)   # shuts down old workers -> frees their RAM
  invisible(gc(verbose = FALSE))
  if (n_workers > 1) future::plan(future::multisession, workers = n_workers)
}

# Appending new columns to an old-schema CSV would silently corrupt it.
check_output_schema <- function(paths, cfg) {
  if (!isTRUE(cfg$resume) || !file.exists(paths$pbp)) return(invisible(TRUE))
  hdr <- names(readr::read_csv(paths$pbp, n_max = 0, show_col_types = FALSE, progress = FALSE))
  miss <- setdiff(cfg$output_ofc_cols, hdr)
  if (length(miss) > 0) {
    stop("Existing output (", paths$pbp, ") was made by an older version of this script ",
         "(missing: ", paste(miss, collapse = ", "), "). Set CONFIG$resume = FALSE to start fresh ",
         "(this deletes the old output and progress log), or move those files first.")
  }
  invisible(TRUE)
}

read_completed_ids <- function(progress_path) {
  if (!file.exists(progress_path)) return(numeric(0))
  readr::read_csv(progress_path, col_select = dplyr::all_of("nfl_id"),
                  show_col_types = FALSE, progress = FALSE)$nfl_id
}

main <- function(cfg = CONFIG) {
  load_packages()
  paths <- resolve_paths(cfg)
  on.exit(future::plan(future::sequential), add = TRUE)

  # Fresh start or resume
  if (!isTRUE(cfg$resume)) {
    unlink(c(paths$pbp, paths$frames, paths$progress))
  }
  check_output_schema(paths, cfg)

  envelope_model <- load_envelope_model(paths$envelope)
  inputs <- load_inputs(cfg, paths)
  tracking   <- inputs$tracking
  play_index <- inputs$play_index
  all_wrs    <- inputs$all_wrs
  rm(inputs)

  done <- read_completed_ids(paths$progress)
  todo <- all_wrs[!(all_wrs$nfl_id %in% done), , drop = FALSE]
  if (!is.null(cfg$wr_limit)) todo <- utils::head(todo, cfg$wr_limit)

  message(sprintf("WRs total: %d | already done: %d | to run now: %d",
                  nrow(all_wrs), sum(all_wrs$nfl_id %in% done), nrow(todo)))
  if (nrow(todo) == 0) {
    message("Nothing to do.")
    return(invisible(paths$pbp))
  }

  t_start <- Sys.time()

  for (i in seq_len(nrow(todo))) {
    wr_id   <- todo$nfl_id[i]
    wr_name <- todo$player_name[i]

    if ((i - 1) %% cfg$restart_workers_every == 0) restart_workers(cfg$n_workers)

    t0  <- Sys.time()
    res <- process_wr(wr_id, wr_name, tracking, play_index, envelope_model, paths, cfg)

    # Append this WR to the single final file
    append_csv(res$play_level, paths$pbp)
    if (isTRUE(cfg$save_frame_level)) append_csv(res$frame_level, paths$frames)

    secs <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
    n_out <- if (is.null(res$play_level)) 0L else nrow(res$play_level)

    # Log AFTER the append so a finished WR is never lost or double-counted on resume
    append_csv(
      tibble::tibble(
        nfl_id = wr_id, player_name = wr_name,
        n_candidate_plays = res$n_candidate_plays, n_plays_output = n_out,
        n_errors = res$n_errors, first_error = res$first_error,
        seconds = round(secs, 1), finished_at = format(Sys.time())
      ),
      paths$progress
    )

    elapsed <- as.numeric(difftime(Sys.time(), t_start, units = "mins"))
    eta     <- elapsed / i * (nrow(todo) - i)
    message(sprintf("[%d/%d] %s (%s): %d/%d plays, %d errors, %.1fs | elapsed %.1f min, ETA %.1f min",
                    i, nrow(todo), wr_name, wr_id, n_out, res$n_candidate_plays,
                    res$n_errors, secs, elapsed, eta))
    if (res$n_errors > 0) message("    first error: ", res$first_error)

    # Release everything WR-specific
    rm(res, secs, n_out)
    invisible(gc(verbose = FALSE))
  }

  message("Done. Final play-by-play file: ", paths$pbp)
  invisible(paths$pbp)
}

main()
