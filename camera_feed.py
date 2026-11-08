"""Capture only the latest frame; do not accumulate a queue or record video."""
import threading
import time


class LatestFrame:
    def __init__(self, capture):
        self.capture = capture
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.frame = None
        self.sequence = 0
        self.timestamp = 0.0
        self.error = None
        self.thread = threading.Thread(target=self._read, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def _read(self):
        try:
            while not self.stop.is_set():
                ok, frame = self.capture.read()
                if not ok:
                    self.error = "Camera stopped supplying frames. Check Camera permission and restart."
                    return
                with self.lock:
                    self.frame = frame
                    self.sequence += 1
                    self.timestamp = time.monotonic()
        except Exception as exc:
            self.error = str(exc)
        finally:
            self.capture.release()

    def latest(self):
        with self.lock:
            return self.sequence, self.timestamp, self.frame

    def close(self):
        self.stop.set()
        self.thread.join(timeout=2)

