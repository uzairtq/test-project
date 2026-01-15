import cv2


class TrafficCounter:
    """Motion-based foreground tracker that returns bounding boxes."""

    def __init__(
        self,
        min_area: int = 500,
        min_motion: int = 8,
        min_side: int = 20,
        blur_kernel: int = 5,
        open_iters: int = 2,
        close_iters: int = 1,
        min_fill_ratio: float = 0.2,
    ) -> None:
        self.bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=32, detectShadows=True
        )
        self.min_area = min_area
        self.min_motion = min_motion
        self.min_side = min_side
        self.min_fill_ratio = max(0.0, min(1.0, min_fill_ratio))
        self.blur_kernel = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
        self.open_iters = max(1, open_iters)
        self.close_iters = max(0, close_iters)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.prev_centers = []

    def count(self, frame):
        fg = self.bg_sub.apply(frame)
        if self.blur_kernel > 1:
            fg = cv2.GaussianBlur(fg, (self.blur_kernel, self.blur_kernel), 0)
        _, fg_bin = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
        fg_bin = cv2.morphologyEx(
            fg_bin, cv2.MORPH_OPEN, self.kernel, iterations=self.open_iters
        )
        if self.close_iters:
            fg_bin = cv2.morphologyEx(
                fg_bin, cv2.MORPH_CLOSE, self.kernel, iterations=self.close_iters
            )
        fg_bin = cv2.morphologyEx(fg_bin, cv2.MORPH_DILATE, self.kernel, iterations=2)

        contours, _ = cv2.findContours(fg_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            if w < self.min_side or h < self.min_side:
                continue
            roi = fg_bin[y : y + h, x : x + w]
            if roi.size == 0:
                continue
            fill_ratio = float(cv2.mean(roi)[0]) / 255.0
            if fill_ratio < self.min_fill_ratio:
                continue
            cx = x + w // 2
            cy = y + h // 2
            detections.append(
                {
                    "bbox": (x, y, w, h),
                    "center": (cx, cy),
                    "area": w * h,
                }
            )

        filtered = []
        for det in detections:
            cx, cy = det["center"]
            if self.prev_centers:
                min_dist = min(
                    ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5 for px, py in self.prev_centers
                )
                if min_dist < self.min_motion:
                    continue
            filtered.append(det)

        self.prev_centers = [det["center"] for det in detections]

        return filtered, fg_bin
