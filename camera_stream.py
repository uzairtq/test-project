import threading
import time
from typing import Optional

import cv2


class CameraStream:
    """Threaded frame grabber so counting logic never blocks on I/O."""

    def __init__(self, src, name: str = "cam", target_fps: Optional[float] = None) -> None:
        self.src = src
        self.name = name
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open source {src} for {name}")

        self.lock = threading.Lock()
        self.frame: Optional["cv2.Mat"] = None
        self.stopped = False
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
        self.max_frames = self.total_frames
        self.current_frame = 0
        self.target_period = 1.0 / target_fps if target_fps else None
        self.should_loop = isinstance(src, str)
        self._looped = False  # Flag set when video loops back

        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()

    def _rewind(self) -> None:
        if self.should_loop:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.current_frame = 0
            self._looped = True

    def _update_loop(self) -> None:
        while not self.stopped:
            loop_start = time.time()
            ret, frame = self.cap.read()
            if not ret:
                self._rewind()
                time.sleep(0.05)
                continue

            self.current_frame += 1
            if self.max_frames and self.current_frame >= self.max_frames:
                self._rewind()
                continue

            with self.lock:
                self.frame = frame

            if self.target_period:
                elapsed = time.time() - loop_start
                remaining = self.target_period - elapsed
                if remaining > 0:
                    time.sleep(remaining)

    def read(self):
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()

    @property
    def frame_count(self) -> Optional[int]:
        return self.total_frames

    def set_loop_length(self, max_frames: int) -> None:
        if not max_frames:
            return
        if self.total_frames:
            self.max_frames = min(max_frames, self.total_frames)
        else:
            self.max_frames = max_frames

    def check_looped(self) -> bool:
        """Return True if video looped since last check, then reset flag."""
        if self._looped:
            self._looped = False
            return True
        return False

    def release(self) -> None:
        self.stopped = True
        self.thread.join(timeout=1.0)
        self.cap.release()
