import json
import math
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from config import (
    CAM1_SRC,
    CAM2_SRC,
    WINDOW_SECONDS,
    FPS_SAMPLE,
    STREAM_FPS,
    DISPLAY_SCALE,
    MIN_MOTION_PIXELS,
    MIN_FOREGROUND_RATIO,
    FG_BLUR_KERNEL,
    FG_OPEN_ITERS,
    FG_CLOSE_ITERS,
    CALIBRATION_FILE,
    CALIBRATION_TAG,
    CALIBRATION_MIN_POINTS,
    MONITOR_MIN_POINTS,
    HOMOGRAPHY_ERROR_THRESHOLD,
    HOMOGRAPHY_GRID_ROWS,
    HOMOGRAPHY_GRID_COLS,
    LOWER_HALF_Y_THRESHOLD,
    LOWER_HALF_WEIGHT,
    MOVING_STREAK_FRAMES,
    PAIR_TIME_TOLERANCE_MS,
    CALIBRATION_PATH,
)

from camera_stream import CameraStream
from health_metrics import (
    compute_blur,
    compute_brightness,
    compute_occlusion_score,
)
from traffic_counter import TrafficCounter


def _resolve_source(value: str):
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def _source_has_tag(value: str, tag: str) -> bool:
    if not isinstance(value, str):
        return False
    return tag in value.lower()


def _infer_car_label(value: str):
    if not isinstance(value, str):
        return None
    lower = Path(value).name.lower()
    for label in ("car1", "car2", "car3"):
        if label in lower:
            return label
    return None


def _default_min_box_side(src1_raw: str, src2_raw: str) -> int:
    label_order = [_infer_car_label(src1_raw), _infer_car_label(src2_raw)]
    for label in label_order:
        if label == "car1":
            print(f"Label: {label}, Min box side: 120")
            return 120
        if label in {"car2", "car3"}:
            print(f"Label: {label}, Min box side: 50")
            return 50
    print("No specific label found, using default min box side: 60")
    return 60


CAM1_SRC_RAW = str(CAM1_SRC)
CAM2_SRC_RAW = str(CAM2_SRC)
CAM1_SRC = _resolve_source(CAM1_SRC_RAW)
CAM2_SRC = _resolve_source(CAM2_SRC_RAW)

MIN_MOTION = int(MIN_MOTION_PIXELS)
MIN_SIDE = _default_min_box_side(CAM1_SRC_RAW, CAM2_SRC_RAW)
MIN_FILL_RATIO = float(MIN_FOREGROUND_RATIO)
REPROJECTION_THRESHOLD = float(HOMOGRAPHY_ERROR_THRESHOLD)
GRID_ROWS = int(HOMOGRAPHY_GRID_ROWS)
GRID_COLS = int(HOMOGRAPHY_GRID_COLS)
LOWER_HALF_THRESHOLD = float(LOWER_HALF_Y_THRESHOLD)
LOWER_HALF_WEIGHT = float(LOWER_HALF_WEIGHT)
PAIR_TOLERANCE_S = max(0.0, PAIR_TIME_TOLERANCE_MS / 1000.0)
CALIBRATION_TAG = CALIBRATION_TAG.lower()
IS_CALIBRATION_RUN = _source_has_tag(CAM1_SRC_RAW, CALIBRATION_TAG) and _source_has_tag(
    CAM2_SRC_RAW, CALIBRATION_TAG
)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _resize_display(frame):
    if DISPLAY_SCALE == 1.0 or frame is None:
        return frame
    return cv2.resize(frame, (0, 0), fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)


def _annotate_frame(frame, detections, color):
    annotated = frame.copy()
    if not detections:
        return annotated
    for idx, det in enumerate(detections, start=1):
        x, y, w, h = det["bbox"]
        cx, cy = det["center"]
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
        cv2.circle(annotated, (cx, cy), 4, color, -1)
        label = f"Car {idx} ({cx},{cy})"
        cv2.putText(
            annotated,
            label,
            (x, max(15, y - 5)),
            FONT,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return annotated


def _draw_status_text(frame, text, color, row=0):
    if frame is None:
        return
    y = 20 + row * 22
    cv2.putText(frame, text, (10, y), FONT, 0.6, color, 2, cv2.LINE_AA)


def _generate_grid_points(det, rows, cols):
    if det is None:
        return None
    x, y, w, h = det["bbox"]
    xs = np.linspace(x, x + w, max(2, cols))
    ys = np.linspace(y, y + h, max(2, rows))
    return np.array([(float(cx), float(cy)) for cy in ys for cx in xs], dtype=np.float32)


def _collect_correspondences(det1, det2, frame_shape=None, weighted=True):
    """Return grid correspondences. Use weighted=True for calibration, False for monitoring."""
    if det1 is None or det2 is None:
        return None, None
    pts1 = _generate_grid_points(det1, GRID_ROWS, GRID_COLS)
    pts2 = _generate_grid_points(det2, GRID_ROWS, GRID_COLS)
    if pts1 is None or pts2 is None:
        return None, None
    if not weighted:
        return pts1, pts2
    height = frame_shape[0] if frame_shape is not None and len(frame_shape) >= 1 else None
    if height is None or height <= 0:
        return pts1, pts2
    weighted1 = []
    weighted2 = []
    for p1, p2 in zip(pts1, pts2):
        weighted1.append(tuple(p1))
        weighted2.append(tuple(p2))
        avg_ratio = max(p1[1], p2[1]) / height
        if avg_ratio >= LOWER_HALF_THRESHOLD:
            extra = max(0, int(math.ceil(max(1.0, LOWER_HALF_WEIGHT)) - 1))
            for _ in range(extra):
                weighted1.append(tuple(p1))
                weighted2.append(tuple(p2))
    return np.array(weighted1, dtype=np.float32), np.array(weighted2, dtype=np.float32)


def _estimate_homography_matrix(pts1, pts2):
    if pts1 is None or pts2 is None:
        return None, None
    pts1 = np.asarray(pts1, dtype=np.float32)
    pts2 = np.asarray(pts2, dtype=np.float32)
    if len(pts1) < 4 or len(pts2) < 4:
        return None, None
    matrix, inliers = cv2.findHomography(
        pts1,
        pts2,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        maxIters=2000,
    )
    return matrix, inliers


def _reprojection_error(matrix, pts1, pts2):
    if matrix is None or pts1 is None or pts2 is None:
        return float("inf")
    pts1 = np.asarray(pts1, dtype=np.float32)
    pts2 = np.asarray(pts2, dtype=np.float32)
    if pts1.size == 0 or pts2.size == 0:
        return float("inf")
    pts1_reshaped = pts1.reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(pts1_reshaped, matrix).reshape(-1, 2)
    residuals = np.linalg.norm(projected - pts2, axis=1)
    return float(residuals.mean())


def _homography_divergence(H_baseline, H_current, frame_shape):
    """Measure how much two homographies differ by comparing where they map test points.
    
    Uses a dense grid of test points in the lower half of the frame (where cars travel)
    and computes the median difference to be robust to outliers from unstable fits.
    """
    if H_baseline is None or H_current is None:
        return float("inf")
    # Normalize both matrices so H[2,2] = 1 (standard homography normalization)
    if abs(H_baseline[2, 2]) > 1e-9:
        H_baseline = H_baseline / H_baseline[2, 2]
    if abs(H_current[2, 2]) > 1e-9:
        H_current = H_current / H_current[2, 2]
    # Create a dense grid of test points in the lower half of the frame
    h, w = frame_shape[:2] if frame_shape is not None else (720, 1280)
    xs = np.linspace(w * 0.1, w * 0.9, 5)
    ys = np.linspace(h * 0.5, h * 0.9, 4)
    test_pts = np.array([[x, y] for y in ys for x in xs], dtype=np.float32).reshape(-1, 1, 2)
    proj_baseline = cv2.perspectiveTransform(test_pts, H_baseline).reshape(-1, 2)
    proj_current = cv2.perspectiveTransform(test_pts, H_current).reshape(-1, 2)
    # Use median for robustness
    divergence = np.median(np.linalg.norm(proj_baseline - proj_current, axis=1))
    return float(divergence)


def _apply_alert_tint(frame):
    if frame is None:
        return frame
    overlay = np.full_like(frame, (0, 0, 255))
    return cv2.addWeighted(frame, 0.8, overlay, 0.2, 0)


def _select_primary_detection(detections):
    if not detections:
        return None
    return max(detections, key=lambda det: det.get("area", 0))


def _sync_detection(current_det, prev_state, now_ts, tolerance_s):
    if current_det is not None:
        return current_det, (current_det, now_ts)
    if prev_state is None:
        return None, None
    prev_det, prev_ts = prev_state
    if now_ts - prev_ts <= tolerance_s:
        return prev_det, prev_state
    return None, None


def _load_calibration_baseline():
    if not CALIBRATION_PATH.exists():
        return None
    try:
        with CALIBRATION_PATH.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        matrix = data.get("matrix")
        if matrix:
            data["matrix_np"] = np.array(matrix, dtype=np.float32)
            return data
        print(
            f"⚠ Calibration file {CALIBRATION_PATH} uses an old format. "
            "Re-running calibration now."
        )
    except (OSError, json.JSONDecodeError) as exc:
        print(f"⚠ Failed to load calibration file {CALIBRATION_PATH}: {exc}")
    return None


def _save_homography_calibration(pts1, pts2):
    matrix, inliers = _estimate_homography_matrix(pts1, pts2)
    if matrix is None:
        return None
    if inliers is not None:
        pts1 = np.asarray(pts1)[inliers.ravel() == 1]
        pts2 = np.asarray(pts2)[inliers.ravel() == 1]
    error = _reprojection_error(matrix, pts1, pts2)
    payload = {
        "tag": CALIBRATION_TAG,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "matrix": matrix.tolist(),
        "reproj_error": error,
        "points": int(len(pts1)),
    }
    payload["matrix_np"] = matrix
    CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CALIBRATION_PATH.open("w", encoding="utf-8") as fh:
        json.dump({k: v for k, v in payload.items() if k != "matrix_np"}, fh, indent=2)
    print(
        f"✅ Calibration saved to {CALIBRATION_PATH} with {payload['points']} correspondences, "
        f"baseline reproj error = {error:.2f}px (threshold = {REPROJECTION_THRESHOLD}px)"
    )
    return payload


def summarize_window(stats1, stats2) -> None:
    arr1 = np.array(stats1)
    arr2 = np.array(stats2)

    mean_count1, mean_br1, mean_bl1, mean_occ1 = arr1.mean(axis=0)
    mean_count2, mean_br2, mean_bl2, mean_occ2 = arr2.mean(axis=0)

    print("---- WINDOW SUMMARY ----")
    print(
        f"Counts: cam1={mean_count1:.1f}, cam2={mean_count2:.1f}, diff={abs(mean_count1 - mean_count2):.1f}"
    )
    print(
        f"Brightness: cam1={mean_br1:.1f}, cam2={mean_br2:.1f}, diff={abs(mean_br1 - mean_br2):.1f}"
    )
    print(
        f"Blur: cam1={mean_bl1:.1f}, cam2={mean_bl2:.1f}, diff={abs(mean_bl1 - mean_bl2):.1f}"
    )
    print(
        f"Occlusion: cam1={mean_occ1:.2f}, cam2={mean_occ2:.2f}, diff={abs(mean_occ1 - mean_occ2):.2f}"
    )

    if abs(mean_count1 - mean_count2) > 5:
        print("⚠ Large mismatch in traffic counts between cameras")

    if abs(mean_br1 - mean_br2) > 30:
        print("⚠ Brightness mismatch – potential occlusion or exposure issue")

    if abs(mean_bl1 - mean_bl2) > 200:
        print("⚠ Blur mismatch – one camera may be out of focus or dirty")

    if abs(mean_occ1 - mean_occ2) > 0.3 or max(mean_occ1, mean_occ2) > 0.6:
        print("⚠ Strong occlusion or blockage detected")

    print("------------------------\n")


def main() -> None:
    cam1 = CameraStream(CAM1_SRC, name="cam1", target_fps=STREAM_FPS)
    cam2 = CameraStream(CAM2_SRC, name="cam2", target_fps=STREAM_FPS)

    if cam1.frame_count and cam2.frame_count:
        sync_frames = min(cam1.frame_count, cam2.frame_count)
        cam1.set_loop_length(sync_frames)
        cam2.set_loop_length(sync_frames)

    counter1 = TrafficCounter(
        min_motion=MIN_MOTION,
        min_side=MIN_SIDE,
        blur_kernel=FG_BLUR_KERNEL,
        open_iters=FG_OPEN_ITERS,
        close_iters=FG_CLOSE_ITERS,
        min_fill_ratio=MIN_FILL_RATIO,
    )
    counter2 = TrafficCounter(
        min_motion=MIN_MOTION,
        min_side=MIN_SIDE,
        blur_kernel=FG_BLUR_KERNEL,
        open_iters=FG_OPEN_ITERS,
        close_iters=FG_CLOSE_ITERS,
        min_fill_ratio=MIN_FILL_RATIO,
    )

    calibration_active = IS_CALIBRATION_RUN or not CALIBRATION_PATH.exists()
    calibration_pts1 = []
    calibration_pts2 = []
    baseline_record = None if calibration_active else _load_calibration_baseline()
    missing_baseline_logged = False
    alert_latched = False

    stats1 = deque(maxlen=WINDOW_SECONDS * FPS_SAMPLE)
    stats2 = deque(maxlen=WINDOW_SECONDS * FPS_SAMPLE)
    prev_primary1 = None
    prev_primary2 = None
    monitor_pts1 = deque(maxlen=MONITOR_MIN_POINTS * 2)
    monitor_pts2 = deque(maxlen=MONITOR_MIN_POINTS * 2)
    moving_car_active = False
    detection_streak = 0
    drift_latched = False

    last_time = time.time()

    try:
        while True:
            now = time.time()
            if now - last_time < 1.0 / FPS_SAMPLE:
                time.sleep(0.005)
                continue
            last_time = now

            # Check if either video looped - clear monitoring buffer
            if cam1.check_looped() or cam2.check_looped():
                monitor_pts1.clear()
                monitor_pts2.clear()
                moving_car_active = False
                detection_streak = 0
                drift_latched = False
                alert_latched = False

            frame1 = cam1.read()
            frame2 = cam2.read()
            if frame1 is None or frame2 is None:
                print("Waiting for frames...")
                time.sleep(0.1)
                continue

            detections1, fg1 = counter1.count(frame1)
            detections2, fg2 = counter2.count(frame2)
            count1 = len(detections1)
            count2 = len(detections2)

            br1 = compute_brightness(frame1)
            br2 = compute_brightness(frame2)
            bl1 = compute_blur(frame1)
            bl2 = compute_blur(frame2)
            occ1 = compute_occlusion_score(frame1)
            occ2 = compute_occlusion_score(frame2)

            stats1.append((count1, br1, bl1, occ1))
            stats2.append((count2, br2, bl2, occ2))

            primary1 = _select_primary_detection(detections1)
            primary2 = _select_primary_detection(detections2)
            synced1, prev_primary1 = _sync_detection(
                primary1, prev_primary1, now, PAIR_TOLERANCE_S
            )
            synced2, prev_primary2 = _sync_detection(
                primary2, prev_primary2, now, PAIR_TOLERANCE_S
            )
            if frame1 is not None:
                frame_shape = frame1.shape
            elif frame2 is not None:
                frame_shape = frame2.shape
            else:
                frame_shape = None
            # Always use weighted=True so calibration and monitoring produce comparable homographies
            corr1, corr2 = _collect_correspondences(synced1, synced2, frame_shape, weighted=True)

            status_messages = []
            moving_msg_added = False
            alert_active = drift_latched

            if calibration_active:
                drift_latched = False
                collected = len(calibration_pts1)
                target = CALIBRATION_MIN_POINTS
                status_messages.append((f"Calibrating ({collected}/{target})", (0, 255, 255)))
                if corr1 is not None:
                    calibration_pts1.extend(corr1.tolist())
                    calibration_pts2.extend(corr2.tolist())
                    if len(calibration_pts1) >= CALIBRATION_MIN_POINTS:
                        baseline_record = _save_homography_calibration(
                            calibration_pts1, calibration_pts2
                        )
                        calibration_active = False
                        status_messages = [("Calibration saved", (0, 255, 0))]
                else:
                    status_messages.append(("Waiting for overlapping car", (0, 255, 255)))
            else:
                if baseline_record is None:
                    if not missing_baseline_logged:
                        print(
                            f"⚠ No calibration baseline found at {CALIBRATION_PATH}. "
                            "Run calibration clips to capture geometry."
                        )
                        missing_baseline_logged = True
                else:
                    if corr1 is not None:
                        detection_streak += 1
                        if not moving_car_active and detection_streak >= MOVING_STREAK_FRAMES:
                            moving_car_active = True
                        if moving_car_active:
                            if not moving_msg_added:
                                status_messages.append(("Moving car detected", (0, 200, 0)))
                                moving_msg_added = True
                            for p1, p2 in zip(corr1, corr2):
                                monitor_pts1.append(tuple(p1))
                                monitor_pts2.append(tuple(p2))
                            if len(monitor_pts1) >= MONITOR_MIN_POINTS:
                                pts1_arr = np.array(monitor_pts1, dtype=np.float32)
                                pts2_arr = np.array(monitor_pts2, dtype=np.float32)
                                fresh_matrix, _ = _estimate_homography_matrix(pts1_arr, pts2_arr)
                                baseline_matrix = baseline_record.get("matrix_np")
                                divergence = _homography_divergence(
                                    baseline_matrix, fresh_matrix, frame_shape
                                )
                                status_messages.append((f"H diverge {divergence:.1f}px", (255, 255, 0)))
                                if divergence > REPROJECTION_THRESHOLD:
                                    drift_latched = True
                                    alert_active = True
                                    status_messages.append(("CAMERA DRIFT ALERT", (0, 0, 255)))
                                    if not alert_latched:
                                        print(
                                            "⚠ Camera geometry drift detected: homography divergence="
                                            f"{divergence:.1f}px (>{REPROJECTION_THRESHOLD}). "
                                            "Camera angle may have changed."
                                        )
                                    alert_latched = True
                                else:
                                    drift_latched = False
                                    alert_latched = False
                            else:
                                status_messages.append(
                                    (f"Gathering ({len(monitor_pts1)}/{MONITOR_MIN_POINTS})", (255, 255, 0))
                                )
                                alert_active = drift_latched
                        else:
                            status_messages.append(
                                (
                                    f"Confirming car ({detection_streak}/{MOVING_STREAK_FRAMES})",
                                    (0, 200, 200),
                                )
                            )
                    else:
                        if not moving_car_active:
                            detection_streak = 0
                            status_messages.append(("Waiting for car", (200, 200, 200)))
                        else:
                            if not moving_msg_added:
                                status_messages.append(("Moving car detected", (0, 200, 0)))
                                moving_msg_added = True

            if drift_latched:
                if not any(text == "CAMERA DRIFT ALERT" for text, _ in status_messages):
                    status_messages.append(("CAMERA DRIFT ALERT", (0, 0, 255)))
                alert_active = True

            annotated1 = _annotate_frame(frame1, detections1, (0, 255, 0))
            annotated2 = _annotate_frame(frame2, detections2, (0, 165, 255))

            for idx, (text, color) in enumerate(status_messages):
                _draw_status_text(annotated1, text, color, row=idx)
                _draw_status_text(annotated2, text, color, row=idx)

            if alert_active:
                annotated1 = _apply_alert_tint(annotated1)
                annotated2 = _apply_alert_tint(annotated2)

            cv2.imshow("cam1_raw", _resize_display(annotated1))
            cv2.imshow("cam2_raw", _resize_display(annotated2))

            if cv2.waitKey(1) & 0xFF == 27:
                break

            if len(stats1) == stats1.maxlen and len(stats2) == stats2.maxlen:
                summarize_window(stats1, stats2)

    finally:
        cam1.release()
        cam2.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
