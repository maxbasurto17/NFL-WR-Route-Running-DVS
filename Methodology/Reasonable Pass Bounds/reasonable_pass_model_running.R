library(dplyr)
library(quantreg)
library(splines)

# =============================================================================
# 1. LOAD MODEL ARTIFACTS
# =============================================================================
model_path <- "~/Documents/Sports Analytics/2026 CMSAC RRC/pitch_control_spline_model.rds"
pitch_model <- readRDS(model_path)

# Print loaded model configuration summary
cat(sprintf("Loaded Production Model: %s\n", toupper(pitch_model$winning_shape)))
cat(sprintf("Parameters: tau_x = %.3f | tau_y = %.3f | Lead Bias Offset = +%.2f yds\n", 
            pitch_model$winning_tau_x, pitch_model$winning_tau_y, pitch_model$lead_bias))

# =============================================================================
# 2. DEFINE PRODUCTION PREDICTION FUNCTION
# =============================================================================
predict_throw_boundary <- function(data_df, model_bundle = pitch_model) {
  data_df %>%
    mutate(
      # Ensure depth feature matches model training definition
      receiver_depth = pmax(abs(x_std - los_x_std), 0.1),
      
      # Predict dynamic radii (r_x and r_y)
      r_x = pmax(predict(model_bundle$qr_x, newdata = .), 0.1),
      r_y = pmax(predict(model_bundle$qr_y, newdata = .), 0.1),
      
      # Apply statistically calibrated bias offset to center location
      center_x = x_std + model_bundle$lead_bias,
      center_y = y_std,
      
      # Compute total catchment ellipse area (sq. yards)
      area_sq_yds = pi * r_x * r_y
    )
}

# =============================================================================
# 3. EXAMPLE USAGE ON NEW TRACKING DATA
# =============================================================================
# Replace 'new_tracking_data' with your incoming pass dataset:
# pass_boundaries <- predict_throw_boundary(new_tracking_data)
