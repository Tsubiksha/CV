"""
camera_utils.py - Threaded Camera Capture with Lock-Free Frame Buffer
---------------------------------------------------------------------------
Thread-safe architecture:
  - CameraBuffer runs a single capture thread at 30fps into a deque(maxlen=1)
  - get_latest_frame() is non-blocking — always returns the most recent frame
  - Frozen-frame detection via MD5 to catch stuck USB cameras
  - open_camera() preserved for backward compatibility with app.py
"""
import os
import cv2
import hashlib
import threading
import time
from collections import deque

USB_CAMERA_INDEX = 0
_FROZEN_TIMEOUT = 3.0   # seconds before declaring camera frozen


class CameraBuffer:
    """Background-threaded camera capture. Thread-safe via RLock."""

    def __init__(self, index=None):
        self._lock = threading.RLock()
        self._buf: deque = deque(maxlen=1)     # always holds freshest frame
        self._cap = None
        self._index = index if index is not None else find_external_camera()
        self._running = False
        self._thread: threading.Thread | None = None

        # Frozen-frame detection
        self._last_hash: str = ""
        self._last_hash_time: float = time.time()

        self._open_camera()

    def _open_camera(self):
        with self._lock:
            if self._cap and self._cap.isOpened():
                self._cap.release()
            cap = cv2.VideoCapture(self._index, cv2.CAP_DSHOW)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            if not cap.isOpened():
                print(f"[CameraBuffer] ⚠ Could not open camera {self._index}")
                return
            ret, frame = cap.read()
            if not ret or frame is None:
                cap.release()
                print(f"[CameraBuffer] ⚠ Camera {self._index} opened but unreadable")
                return
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"[CameraBuffer] ✓ Camera {self._index} ready — {w}x{h}")
            self._cap = cap
            self._buf.append(frame)

    def start(self):
        """Start the background capture thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True, name="CameraCapture")
        self._thread.start()
        print("[CameraBuffer] Capture thread started")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    def _capture_loop(self):
        fail_count = 0
        while self._running:
            with self._lock:
                if self._cap is None or not self._cap.isOpened():
                    self._open_camera()
                    time.sleep(1)
                    continue
                ret, frame = self._cap.read()

            if not ret or frame is None:
                fail_count += 1
                if fail_count > 10:
                    print("[CameraBuffer] Too many read failures — reopening camera")
                    with self._lock:
                        self._open_camera()
                    fail_count = 0
                time.sleep(0.05)
                continue

            fail_count = 0

            # Frozen-frame detection
            h = hashlib.md5(frame.tobytes()).hexdigest()
            now = time.time()
            if h == self._last_hash:
                if now - self._last_hash_time > _FROZEN_TIMEOUT:
                    print("[CameraBuffer] ⚠ Frozen frame detected — reopening camera")
                    with self._lock:
                        self._open_camera()
                    self._last_hash_time = now
                    continue
            else:
                self._last_hash = h
                self._last_hash_time = now

            self._buf.append(frame)
            time.sleep(1 / 30)   # target 30fps

    def get_latest_frame(self):
        """Non-blocking: return the most recent captured frame or None."""
        if self._buf:
            return self._buf[-1].copy()
        return None

    def is_ready(self):
        return bool(self._buf)

    @property
    def index(self):
        return self._index


# ── Module-level singleton ─────────────────────────────────────────────────────

_buffer: CameraBuffer | None = None
_buf_lock = threading.Lock()


def get_camera_buffer() -> CameraBuffer:
    """Return (and lazily start) the global CameraBuffer singleton."""
    global _buffer
    with _buf_lock:
        if _buffer is None:
            _buffer = CameraBuffer()
            _buffer.start()
    return _buffer

# ── Backward-compat helpers used by app.py ────────────────────────────────────
def find_external_camera() -> int:
    print(f"[Camera] Using USB camera at index {USB_CAMERA_INDEX}")
    return USB_CAMERA_INDEX


def open_camera(index=None):
    """
    Legacy function: open a camera at 'index' and return (cap, index).
    Used by the old snapshot endpoint; new streaming code uses CameraBuffer.
    """
    if index is None:
        index = find_external_camera()
    print(f"[Camera] Opening index {index}...")
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    if not cap.isOpened():
        raise RuntimeError(f"Camera index {index} failed to open.")
    ret, frame = cap.read()
    if not ret or frame is None:
        cap.release()
        raise RuntimeError(f"Camera {index} opened but cannot read frames.")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[Camera] ✓ Camera {index} ready — {w}x{h}")
    return cap, index