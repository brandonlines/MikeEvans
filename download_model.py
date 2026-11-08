"""Download Google's versioned tracking model; never upload webcam data."""
from pathlib import Path
import urllib.request
import hashlib

URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
ROOT = Path(__file__).resolve().parent
SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"

def main():
    target = ROOT / "face_landmarker.task"
    if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == SHA256:
        print("Tracking model already present.")
        return
    temporary = target.with_suffix(".download")
    print("Downloading Google Face Landmarker (tracking only, not face generation).")
    with urllib.request.urlopen(URL, timeout=120) as response, temporary.open("wb") as stream:
        while block := response.read(1024*1024):
            stream.write(block)
    if hashlib.sha256(temporary.read_bytes()).hexdigest() != SHA256:
        raise RuntimeError("Model download checksum mismatch; run setup again.")
    temporary.replace(target)
    print("Model SHA256:", hashlib.sha256(target.read_bytes()).hexdigest())

if __name__ == "__main__":
    main()
