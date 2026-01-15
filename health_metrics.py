import cv2
import numpy as np


def compute_brightness(frame) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(gray.mean())


def compute_contrast(frame) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(gray.std())


def compute_blur(frame) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def compute_occlusion_score(frame) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    norm = gray / 255.0

    dark_mask = norm < 0.15
    dark_ratio = dark_mask.mean()

    blurred = cv2.GaussianBlur(norm, (21, 21), 0)
    diff = np.abs(norm - blurred)
    low_var_mask = diff < 0.03
    low_var_ratio = low_var_mask.mean()

    score = 0.5 * dark_ratio + 0.5 * low_var_ratio
    return float(min(1.0, score))
