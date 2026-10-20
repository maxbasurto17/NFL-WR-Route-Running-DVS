library(dplyr)
library(quantreg)
library(purrr)
library(tidyr)
library(ggplot2)
library(splines)

# =============================================================================
# 1. PREPARE DATA & TRAIN / TEST SPLIT
# =============================================================================
processed_data <- completed_pass_bounds %>%
  mutate(
    receiver_depth = pmax(abs(x_std - los_x_std), 0.1),
    err_x = ball_land_x_std - x_std,
    err_y = ball_land_y_std - y_std
  )

set.seed(42)
train_idx  <- sample(seq_len(nrow(processed_data)), size = floor(0.80 * nrow(processed_data)))
train_data <- processed_data[train_idx, ]
test_data  <- processed_data[-train_idx, ]

# --- RIGOROUS BIAS CALCULATION (FIT ON TRAIN DATA ONLY) ---
v_test_train <- t.test(train_data$err_x)
train_p_val  <- v_test_train$p.value
depth_bias   <- if (train_p_val < 0.05) unname(v_test_train$estimate) else 0.0

# Apply bias correction to train/test targets upfront
train_data <- train_data %>% 
  mutate(
    center_x = x_std + depth_bias,
    adj_err_x = ball_land_x_std - center_x,
    abs_err_x = abs(adj_err_x),
    abs_err_y = abs(err_y)
  )

test_data <- test_data %>% 
  mutate(
    center_x = x_std + depth_bias,
    adj_err_x = ball_land_x_std - center_x,
    abs_err_x = abs(adj_err_x),
    abs_err_y = abs(err_y)
  )

# =============================================================================
# 2. DEFINE MODEL FORMULAS
# =============================================================================
base_formulas <- list(
  "linear"      = abs_err ~ receiver_depth,
  "sqrt"        = abs_err ~ sqrt(receiver_depth),
  "log"         = abs_err ~ log(receiver_depth + 1),
  "quadratic"   = abs_err ~ receiver_depth + I(receiver_depth^2),
  "exponential" = abs_err ~ I(1 - exp(-0.08 * receiver_depth))
)

spline_grid <- expand_grid(
  type = c("ns", "bs"),
  df = 2:4,
  degree = 2:3
) %>%
  filter(type == "ns" | (type == "bs" & df > degree)) %>%
  mutate(
    name = ifelse(type == "ns", sprintf("ns_df%d_deg%d", df, degree), sprintf("bs_df%d_deg%d", df, degree)),
    formula_str = ifelse(
      type == "ns",
      sprintf("abs_err ~ ns(receiver_depth, df = %d)", df),
      sprintf("abs_err ~ bs(receiver_depth, df = %d, degree = %d)", df, degree)
    )
  )

spline_formulas <- set_names(map(spline_grid$formula_str, as.formula), spline_grid$name)
model_formulas  <- c(base_formulas, spline_formulas)

tau_x_cand <- seq(0.85, 0.995, by = 0.005)
tau_y_cand <- seq(0.85, 0.995, by = 0.005)

# =============================================================================
# 3. RUN GRID SEARCH (CONSISTENT BIAS-ADJUSTED COVERAGE)
# =============================================================================
grid_results <- map_dfr(names(model_formulas), function(shape_name) {
  
  f_x <- update(model_formulas[[shape_name]], abs_err_x ~ .)
  f_y <- update(model_formulas[[shape_name]], abs_err_y ~ .)
  
  models_x <- map(tau_x_cand, ~ rq(f_x, tau = .x, data = train_data, method = "fn"))
  models_y <- map(tau_y_cand, ~ rq(f_y, tau = .x, data = train_data, method = "fn"))
  names(models_x) <- as.character(tau_x_cand)
  names(models_y) <- as.character(tau_y_cand)
  
  preds_x_train <- map(models_x, ~ pmax(predict(.x, newdata = train_data), 0.1))
  preds_y_train <- map(models_y, ~ pmax(predict(.x, newdata = train_data), 0.1))
  preds_x_test  <- map(models_x, ~ pmax(predict(.x, newdata = test_data), 0.1))
  preds_y_test  <- map(models_y, ~ pmax(predict(.x, newdata = test_data), 0.1))
  
  tau_grid <- expand_grid(tau_x = tau_x_cand, tau_y = tau_y_cand)
  
  map2_dfr(tau_grid$tau_x, tau_grid$tau_y, function(tx, ty) {
    rx_tr <- preds_x_train[[as.character(tx)]]
    ry_tr <- preds_y_train[[as.character(ty)]]
    cov_tr <- mean(((train_data$adj_err_x / rx_tr)^2 + (train_data$err_y / ry_tr)^2) <= 1.0)
    area_tr <- mean(pi * rx_tr * ry_tr)
    
    rx_te <- preds_x_test[[as.character(tx)]]
    ry_te <- preds_y_test[[as.character(ty)]]
    cov_te <- mean(((test_data$adj_err_x / rx_te)^2 + (test_data$err_y / ry_te)^2) <= 1.0)
    area_te <- mean(pi * rx_te * ry_te)
    
    tibble(
      shape = shape_name,
      tau_x = tx,
      tau_y = ty,
      train_coverage = cov_tr,
      train_avg_area = area_tr,
      test_coverage  = cov_te,
      test_avg_area  = area_te
    )
  })
})

# =============================================================================
# 4. PARETO FRONTIER RANKING
# =============================================================================
best_by_shape <- grid_results %>%
  group_by(shape) %>%
  arrange(train_avg_area) %>%
  mutate(max_cov = cummax(train_coverage)) %>%
  filter(train_coverage == max_cov) %>%
  distinct(train_coverage, .keep_all = TRUE) %>%
  mutate(
    d_cov = train_coverage - lag(train_coverage),
    d_area = train_avg_area - lag(train_avg_area),
    train_efficiency = d_cov / d_area
  ) %>%
  ungroup() %>%
  filter(train_coverage >= 0.90) %>%
  group_by(shape) %>%
  arrange(desc(train_efficiency)) %>%
  slice(1) %>%
  ungroup() %>%
  arrange(desc(train_efficiency))

cat("--- COMPARISON OF BOUNDARY SHAPES (Gated >= 90% Coverage) ---\n")
print(best_by_shape %>% select(shape, tau_x, tau_y, train_coverage, test_coverage, train_avg_area, train_efficiency))

# =============================================================================
# 5. SEQUENTIAL SPATIAL CALIBRATION CHECK ON OUT-OF-SAMPLE DATA
# =============================================================================
winning_model_found <- FALSE

for (i in seq_len(nrow(best_by_shape))) {
  
  candidate <- best_by_shape[i, ]
  shape_name <- candidate$shape
  tx <- candidate$tau_x
  ty <- candidate$tau_y
  
  cat(sprintf("\n=========================================================\n"))
  cat(sprintf("EVALUATING RANK %d MODEL: %s (tau_x: %.3f, tau_y: %.3f)\n", i, toupper(shape_name), tx, ty))
  cat(sprintf("=========================================================\n"))
  
  f_x <- update(model_formulas[[shape_name]], abs_err_x ~ .)
  f_y <- update(model_formulas[[shape_name]], abs_err_y ~ .)
  
  mod_x <- rq(f_x, tau = tx, data = train_data, method = "fn")
  mod_y <- rq(f_y, tau = ty, data = train_data, method = "fn")
  
  calibration_table <- test_data %>%
    mutate(
      r_x = pmax(predict(mod_x, newdata = .), 0.1),
      r_y = pmax(predict(mod_y, newdata = .), 0.1),
      in_ellipse = (((adj_err_x) / r_x)^2 + ((err_y) / r_y)^2) <= 1.0,
      depth_bucket = cut(
        receiver_depth, 
        breaks = c(0, 5, 12, 20, 60), 
        labels = c("Short (0-5y)", "Medium (5-12y)", "Deep (12-20y)", "Ultra-Deep (20y+)")
      )
    ) %>%
    group_by(depth_bucket) %>%
    summarize(
      n_passes = n(),
      empirical_coverage = mean(in_ellipse),
      avg_rx = mean(r_x),
      avg_ry = mean(r_y),
      .groups = "drop"
    )
  
  print(calibration_table)
  
  if (all(calibration_table$empirical_coverage >= 0.85)) {
    cat(sprintf("\n>>> SUCCESS: '%s' model passed all tests with >= 85%% coverage across all pass depths! <<<\n", shape_name))
    winning_model_found <- TRUE
    winning_shape <- shape_name
    winning_tau_x <- tx
    winning_tau_y <- ty
    break
  } else {
    cat(sprintf("\n---> REJECTED: '%s' model failed the >= 85%% bucket coverage threshold. Checking next model...\n", shape_name))
  }
}

# =============================================================================
# 6. PRODUCTION FIT & SAFE ARTIFACT EXPORT
# =============================================================================
if (winning_model_found) {
  cat(sprintf("\n--- FITTING FINAL PRODUCTION MODEL (%s) ON FULL DATASET ---\n", toupper(winning_shape)))
  
  # Compute final bias on full dataset
  full_v_test <- t.test(processed_data$err_x)
  final_lead_bias <- if (full_v_test$p.value < 0.05) unname(full_v_test$estimate) else 0.0
  
  prod_data <- processed_data %>%
    mutate(
      center_x = x_std + final_lead_bias,
      abs_err_x = abs(ball_land_x_std - center_x),
      abs_err_y = abs(err_y)
    )
  
  f_prod_x <- update(model_formulas[[winning_shape]], abs_err_x ~ .)
  f_prod_y <- update(model_formulas[[winning_shape]], abs_err_y ~ .)
  
  qr_x_production <- rq(f_prod_x, tau = winning_tau_x, data = prod_data, method = "fn")
  qr_y_production <- rq(f_prod_y, tau = winning_tau_y, data = prod_data, method = "fn")
  
  # Bundle and export safely inside the conditional block
  model_bundle <- list(
    qr_x = qr_x_production,
    qr_y = qr_y_production,
    winning_shape = winning_shape,
    winning_tau_x = winning_tau_x,
    winning_tau_y = winning_tau_y,
    lead_bias = final_lead_bias
  )
  
  save_path <- "~/Documents/Sports Analytics/2026 CMSAC RRC/pitch_control_spline_model.rds"
  saveRDS(model_bundle, file = save_path)
  cat(sprintf("Model bundle successfully exported to: %s\n", save_path))
  
} else {
  cat("\nNo models passed the >= 85% bucket coverage requirement across all pass depths.\n")
}
