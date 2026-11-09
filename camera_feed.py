"""Capture only the latest frame; do not accumulate a queue or record video."""
import threading
import time

REOPEN_AFTER = 2.0  # Seconds without a frame before the camera is reopened.
RETRY_DELAY = 0.05


class LatestFrame:
    def __init__(self, open_camera):
        self.open_camera = open_camera
        self.capture = open_camera()
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.frame = None
        self.sequence = 0
        self.timestamp = 0.0
        self.problem = None
        self.thread = threading.Thread(target=self._read, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def _grab(self):
        try:
            ok, frame = self.capture.read()
        except Exception:
            return None
        return frame if ok else None

    def _reopen(self):
        try:
            self.capture.release()
            self.capture = self.open_camera()
        except Exception:
            pass  # Still gone; the read loop tries again.

    def _read(self):
        last_frame = time.monotonic()
        try:
            while not self.stop.is_set():
                frame = self._grab()
                now = time.monotonic()
                if frame is not None:
                    last_frame = now
                    if self.problem is not None:
                        self.problem = None
                        print('Camera is back.', flush=True)
                    with self.lock:
                        self.frame = frame
                        self.sequence += 1
                        self.timestamp = now
                    continue
                if now - last_frame >= REOPEN_AFTER:
                    if self.problem is None:
                        self.problem = 'Camera stopped supplying frames; reconnecting.'
                        print(f'{self.problem} Check Camera permission if this persists.', flush=True)
                    self._reopen()
                    last_frame = time.monotonic()
                self.stop.wait(RETRY_DELAY)
        finally:
            self.capture.release()

    def latest(self):
        with self.lock:
            return self.sequence, self.timestamp, self.frame

    def close(self):
        self.stop.set()
        self.thread.join(timeout=2)
