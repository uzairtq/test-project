# Two-Camera Traffic Prototype

Quick prototype that ingests two camera feeds, estimates per-camera motion activity, and monitors camera health to distinguish real-world changes from sensor issues.

## Setup

1. Create and activate a virtual environment (Windows shell shown):

```
python -m venv .venv
.\.venv\Scripts\activate
```

2. Install dependencies:

```
pip install -r requirements.txt
```

3. Update `config.py` if you want to change camera/video sources or tuning knobs.

By default, `config.py` points to a Car1/Car2/Car3 stereo experiment in the `videos/` folder and exposes the same knobs that used to live in `.env`:

- `CAM1_SRC`, `CAM2_SRC`: webcam indices, MP4 paths, or RTSP URLs (ints become device IDs, everything else is treated as a path/URL).
- `WINDOW_SECONDS`, `FPS_SAMPLE`, `STREAM_FPS`, `DISPLAY_SCALE`: sampling rate and UI scaling.
- `MIN_MOTION_PIXELS`, `FG_BLUR_KERNEL`, `FG_OPEN_ITERS`, `FG_CLOSE_ITERS`, `MIN_FOREGROUND_RATIO`: motion and foreground cleanup.
- `CALIBRATION_FILE`, `CALIBRATION_TAG`, `CALIBRATION_MIN_POINTS`, `MONITOR_MIN_POINTS`: calibration storage and point budgets.
- `HOMOGRAPHY_ERROR_THRESHOLD`, `HOMOGRAPHY_GRID_ROWS`, `HOMOGRAPHY_GRID_COLS`, `LOWER_HALF_Y_THRESHOLD`, `LOWER_HALF_WEIGHT`: stereo drift detector shape and road-plane weighting.
- `MOVING_STREAK_FRAMES`, `PAIR_TIME_TOLERANCE_MS`: how long a car must be seen before we trust it, and how much temporal skew we allow between cameras.

Bounding-box size filtering is automatic: clips whose filenames contain `car1` use a stricter `MIN_BOX_SIDE=120` to ignore tiny wheel detections, while `car2`/`car3` clips (and everything else) default to `MIN_BOX_SIDE=50` so smaller cars still register.

## Running

```
python main.py
```

When both sources are finite videos, the longer clip is trimmed to the shorter one so their loops stay synchronized. The loop grabs frames at ~5 FPS, updates traffic counts, and prints sliding-window summaries plus warnings when the two views disagree.

Press `ESC` to exit. Two windows show the annotated camera views.

Each moving blob gets a green/orange "Car" bounding box plus a center-of-gravity marker (midpoint of the box) so you can visually inspect how both cameras track the same objects. Stationary objects caused by camera drift are ignored when their center moves less than `MIN_MOTION_PIXELS`, the auto `MIN_BOX_SIDE` heuristic (120px for Car1 clips, 50px otherwise) suppresses tiny detections, noisy speckles in the foreground mask can be dialed down via the `FG_*` parameters, and outlines with little true foreground fill are dropped via `MIN_FOREGROUND_RATIO`.

### Calibration workflow

1. **Baseline capture** – If no baseline exists yet, the system automatically enters calibration mode and aggregates a dense grid of correspondence points per detection (corners + interior samples) across the clips. Points that land in the lower half of the frame can be up-weighted (`LOWER_HALF_*`) so the math pays closer attention to the road plane. Once it accumulates `CALIBRATION_MIN_POINTS`, it fits a homography (via RANSAC) that maps Cam1 pixels into Cam2 space and stores that matrix plus its reprojection error in `calibration.json`. You can force a recalibration by keeping `CALIBRATION_TAG` in both file names.
2. **Monitoring runs** – Later vehicles (even different routes) steadily feed new correspondence points into a rolling buffer. The code fits a fresh homography per run and compares it against the saved baseline. If their divergence exceeds `HOMOGRAPHY_ERROR_THRESHOLD` pixels, the camera windows tint red and the console prints a “camera drift” warning—making it obvious when the hardware alignment changed instead of the traffic itself.

### Stereo Car1/Car2/Car3 experiment

The repo ships with a small stereo experiment under `videos/` to exercise redundancy and drift detection:

- **Car1 (baseline)** – Both cameras watch the same car pass with a fixed physical setup. Set `CAM1_SRC`/`CAM2_SRC` in `config.py` to the Car1 files and run once. The system enters calibration, aggregates a weighted grid of points while the car moves through the overlap, and saves a baseline homography.
- **Car2 (same geometry)** – Swap `CAM1_SRC`/`CAM2_SRC` to the Car2 clips. The path is different and the car may be smaller, but the camera geometry is unchanged. You should see “Moving car detected” once the streak is satisfied, and the homography divergence stays below `HOMOGRAPHY_ERROR_THRESHOLD`, so no drift alert is raised.
- **Car3 (camera moved)** – Keep `CAM1_SRC` on the Car3 clip and point `CAM2_SRC` at the `Car3`-angle-changed file. The traffic still looks reasonable, but one camera has been bumped. As soon as a moving car is confirmed, the fresh homography no longer matches the baseline, divergence exceeds the threshold, and the red “CAMERA DRIFT ALERT” band latches for the rest of the loop.

This experiment shows how a simple stereo pair plus a homography baseline can call out physical camera movement separately from changes in the traffic itself.

## Demo Talking Points

- **Stereo redundancy**: Two views form a lightweight stereo pair; if one camera drifts, the shared geometry breaks even when the traffic pattern looks normal.
- **Camera health**: Brightness, blur, and occlusion heuristics show when a single lens gets darker, blurrier, or blocked compared to its twin. They don’t fix shared conditions (e.g., night or fog that affect both cameras equally) but they do flag one-off sensor issues like mud on one lens.
- **Seasonality-ready**: Sliding window stats can feed into longer historical baselines per intersection.

For a production follow-up, swap the motion counter with vehicle detection, push metrics to Prometheus/Grafana, and backfill seasonal baselines from historical data.
