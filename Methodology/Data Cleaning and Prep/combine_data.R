library(tidyverse)

# ---------------------------------------------------------
# 1. Your Existing Data Import
# ---------------------------------------------------------
supp_df   <- read_csv("~/Documents/Sports Analytics/2026 Big Data Bowl /All Data Files/supplementary_data.csv")
input_df  <- read_csv("~/Documents/Sports Analytics/2026 Big Data Bowl /All Data Files/inputs.csv")
output_df <- read_csv("~/Documents/Sports Analytics/2026 Big Data Bowl /All Data Files/outputs.csv")

# ---------------------------------------------------------
# 2. Add Phase Identifiers
# ---------------------------------------------------------
input_df  <- input_df %>% mutate(phase = "pre_pass")
output_df <- output_df %>% mutate(phase = "post_pass")

# ---------------------------------------------------------
# 3. Enrich Output Data with Metadata from Input Data
# ---------------------------------------------------------
meta_cols <- c(
  "game_id", "play_id", "nfl_id", "player_name", "player_position", 
  "player_side", "player_role", "player_to_predict", "num_frames_output", 
  "ball_land_x", "ball_land_y", "play_direction", "absolute_yardline_number"
)

# Extract unique static player metadata present in the input frames
static_meta <- input_df %>% 
  select(all_of(meta_cols)) %>% 
  distinct()

# Join that static metadata onto output frames
output_df <- output_df %>% 
  left_join(static_meta, by = c("game_id", "play_id", "nfl_id"))

# ---------------------------------------------------------
# 4. Concatenate Tracking Data & Merge Supplementary Context
# ---------------------------------------------------------
# Stack pre-pass and post-pass tracking vertically
tracking_df <- bind_rows(input_df, output_df)

# Merge play-level metadata (down, distance, pass_result, EPA, etc.)
full_df <- tracking_df %>% 
  left_join(supp_df, by = c("game_id", "play_id"))

# Standardize Data
full_df <- full_df %>% 
  mutate(
    # 1. Standardize Player Coordinates
    x_std = if_else(play_direction == "left", 120 - x, x),
    y_std = if_else(play_direction == "left", 53.3 - y, y),
    
    # 2. Standardize Player Orientation & Motion Direction Angles (Rotate 180°)
    o_std = if_else(play_direction == "left", (o + 180) %% 360, o),
    dir_std = if_else(play_direction == "left", (dir + 180) %% 360, dir),
    
    # 3. Standardize Line of Scrimmage (LOS) & First Down Line
    los_x_std = if_else(play_direction == "left", 120 - absolute_yardline_number, absolute_yardline_number),
    first_down_x_std = los_x_std + yards_to_go,
    
    # 4. Standardize Ball Landing Location
    ball_land_x_std = if_else(play_direction == "left", 120 - ball_land_x, ball_land_x),
    ball_land_y_std = if_else(play_direction == "left", 53.3 - ball_land_y, ball_land_y)
  )

# Save full_df so i dont have to continuously run this code
write_csv(full_df, "~/Documents/Sports Analytics/Personal Projects/NFL Field Value Model/full_data.csv")
