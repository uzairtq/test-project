from pathlib import Path

# Shared config (replaces local .env for sharing)

# Camera sources for the three-car experiment
# Car1: baseline calibration run
CAR1_CAM1 = "videos/CAM1-Car1.mp4"
CAR1_CAM2 = "videos/CAM2-Car1.mp4"

# Car2: same camera geometry, different path
CAR2_CAM1 = "videos/CAM1-Car2.mp4"
CAR2_CAM2 = "videos/CAM2-Car2.mp4"

# Car3: camera2 angle changed
CAR3_CAM1 = "videos/CAM1-Car3.mp4"
CAR3_CAM2 = "videos/CAM2-Car3-AngleChanged.mp4"

# Default run (you can switch these to any of the above)
CAM1_SRC = CAR1_CAM1
CAM2_SRC = CAR1_CAM2

WINDOW_SECONDS = 10
FPS_SAMPLE = 5
STREAM_FPS = 3.0
DISPLAY_SCALE = 0.5

# Motion / detection knobs
MIN_MOTION_PIXELS = 10
MIN_FOREGROUND_RATIO = 0.15
FG_BLUR_KERNEL = 5
FG_OPEN_ITERS = 2
FG_CLOSE_ITERS = 1

# Calibration / drift monitoring
CALIBRATION_FILE = "calibration.json"
CALIBRATION_TAG = "car0"
CALIBRATION_MIN_POINTS = 600
MONITOR_MIN_POINTS = 30
HOMOGRAPHY_ERROR_THRESHOLD = 220.0
HOMOGRAPHY_GRID_ROWS = 3
HOMOGRAPHY_GRID_COLS = 3
LOWER_HALF_Y_THRESHOLD = 0.5
LOWER_HALF_WEIGHT = 4.0
MOVING_STREAK_FRAMES = 2
PAIR_TIME_TOLERANCE_MS = 150

CALIBRATION_PATH = Path(CALIBRATION_FILE)
