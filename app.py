"""Mike Evans V11.1 Fresh: standalone avatar with repaired neck attachment."""
from pathlib import Path
import argparse
import time
import sys
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from puppet import PhotoPuppet
from head_motion import HeadMotion
from body_motion import BodyMotion
from layers import prepare_layers, overlay
from mouth_blend import MouthBlender
from camera_feed import LatestFrame

ROOT = Path(__file__).resolve().parent
WINDOW = "Mike Evans Layered - Live"

audio_rms = 0.0
def audio_callback(indata, frames, time_info, status):
    global audio_rms
    if status:
        return
    rms = np.sqrt(np.mean(indata**2))
    audio_rms = 0.8 * audio_rms + 0.2 * float(rms)

def make_tracker(video=False):
    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(ROOT / 'face_landmarker.task'),
                                        delegate=python.BaseOptions.Delegate.CPU),
        running_mode=vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE,
        num_faces=1, min_face_detection_confidence=0.5, min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5, output_face_blendshapes=False,
        output_facial_transformation_matrixes=False)
    return vision.FaceLandmarker.create_from_options(options)

def detect(tracker, frame, timestamp=None):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    result = tracker.detect(image) if timestamp is None else tracker.detect_for_video(image, timestamp)
    if not result.face_landmarks:
        return None
    h, w = frame.shape[:2]
    points = np.array([[p.x * w, p.y * h] for p in result.face_landmarks[0]], np.float32)
    return points if np.isfinite(points).all() else None

def load_portrait(width):
    image = cv2.imread(str(ROOT / 'mike-evans.png'))
    if image is None:
        raise RuntimeError('Could not read mike-evans.png')
    image = cv2.resize(image, (width, round(image.shape[0] * width / image.shape[1])))
    with make_tracker() as tracker:
        points = detect(tracker, image)
    if points is None:
        raise RuntimeError('Could not detect portrait landmarks.')
    return image, points

def draw_hud(image, text, progress=None):
    h, w = image.shape[:2]
    banner_h = 44
    overlay_img = image.copy()
    cv2.rectangle(overlay_img, (0, h - banner_h), (w, h), (18, 18, 22), -1)
    if progress is not None:
        pw = int(w * np.clip(progress, 0.0, 1.0))
        cv2.rectangle(overlay_img, (0, h - 3), (pw, h), (0, 180, 255), -1)
    cv2.addWeighted(overlay_img, 0.85, image, 0.15, 0, image)
    cv2.putText(image, text, (20, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (235, 235, 240), 1, cv2.LINE_AA)

def main():
    global audio_rms
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--width', type=int, default=960)
    parser.add_argument('--strength', type=float, default=0.20)
    parser.add_argument('--brow-strength', type=float, default=1.2)
    parser.add_argument('--body-motion', type=float, default=1.0)
    parser.add_argument('--head-motion', type=float, default=1.0)
    parser.add_argument('--no-live-mouth', action='store_true')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--no-glasses', action='store_true', help='Hide the separate frame overlay for comparison')
    args = parser.parse_args()

    cv2.setNumThreads(2)
    portrait, points = load_portrait(args.width)
    portrait, alpha, background, glasses, points = prepare_layers(ROOT, portrait, points, make_tracker, detect)
    
    if args.no_glasses:
        glasses = np.zeros_like(glasses)
    puppet = PhotoPuppet(portrait, points)
    head = HeadMotion(portrait, points, alpha, background)
    body = BodyMotion(ROOT, background)
    mouth = MouthBlender()

    if args.check:
        preview = head.render(overlay(puppet.render(points), glasses), points, points)
        preview = body.render(preview, head.last_alpha, head.state, args.body_motion)
        if preview.shape != portrait.shape or not np.isfinite(preview).all():
            raise RuntimeError('Portrait renderer check failed')
        print(f'CHECK PASSED: model loads, {len(points)} portrait landmarks, renderer works.', flush=True)
        return

    audio_stream = None
    try:
        import sounddevice as sd
        audio_stream = sd.InputStream(channels=1, samplerate=16000, blocksize=512, callback=audio_callback)
        audio_stream.start()
        print('Microphone energy stream active.', flush=True)
    except Exception as exc:
        print(f'Audio monitoring unavailable ({exc}); continuing video-only.', flush=True)

    backend = cv2.CAP_AVFOUNDATION if sys.platform == 'darwin' else cv2.CAP_ANY
    camera = cv2.VideoCapture(args.camera, backend)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError('Cannot open webcam. Check camera permissions.')
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_FPS, 30)

    feed = LatestFrame(camera).start()
    neutral = None
    calibration = []
    output = overlay(portrait, glasses)
    paused = False
    live_mouth = not args.no_live_mouth
    next_calibration = time.monotonic() + 3.0

    last_sequence = 0
    last_stamp = 0
    render_count = 0
    total_track = 0.0
    total_render = 0.0
    report = time.monotonic()
    last_good = report
    capture_baseline = 0

    expression_session = None
    prompts = [
        "Relax brows, eyes open, mouth closed",
        "Raise eyebrows and hold",
        "Furrow eyebrows and hold",
        "Close eyes gently and hold"
    ]

    print('Mike Evans V11.1 Fresh — neck repair included.', flush=True)
    print('Click output: R neutral | C expression calibration | M live mouth | P freeze | Q quit.', flush=True)

    try:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, portrait.shape[1], portrait.shape[0])
        with make_tracker(video=True) as tracker:
            while True:
                display_frame = output.copy()

                now = time.monotonic()
                if paused:
                    draw_hud(display_frame, "PAUSED (press P to resume)")
                elif expression_session is not None:
                    step = expression_session['step']
                    elapsed = now - expression_session['start']
                    prog = elapsed / 3.0
                    draw_hud(display_frame, f"Calibration ({step + 1}/4): {prompts[step]}", prog)
                elif now < next_calibration:
                    prog = 1.0 - (next_calibration - now) / 3.0
                    draw_hud(display_frame, "CALIBRATING: Look straight ahead, mouth closed...", prog)
                elif neutral is None:
                    draw_hud(display_frame, "Sampling neutral pose...")

                cv2.imshow(WINDOW, display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key == ord('c'):
                    expression_session = {'step': 0, 'start': time.monotonic(), 'samples': [], 'poses': []}
                    paused = False
                    print(f'Expression calibration: {prompts[0]} (3 seconds)', flush=True)
                if key == ord('r'):
                    expression_session = None
                    neutral = None
                    calibration = []
                    puppet.smoothed = None
                    head.reset()
                    mouth.reset()
                    body.reset()
                    next_calibration = time.monotonic() + 2.0
                    print('Recalibrating neutral position...', flush=True)
                if key == ord('m'):
                    live_mouth = not live_mouth
                    print('Live mouth ON' if live_mouth else 'Live mouth OFF', flush=True)
                if key == ord('p'):
                    paused = not paused

                if feed.error:
                    raise RuntimeError(feed.error)

                sequence, stamp, frame = feed.latest()
                if paused or now < next_calibration or frame is None or sequence == last_sequence:
                    time.sleep(0.003)
                    continue

                last_sequence = sequence
                timestamp = max(last_stamp + 1, int(stamp * 1000))
                last_stamp = timestamp

                t_track_start = time.perf_counter()
                current = detect(tracker, frame, timestamp)
                t_tracked = time.perf_counter()

                if current is None:
                    if audio_rms > 0.02 and neutral is not None:
                        target = neutral.copy()
                        target[152, 1] += audio_rms * puppet.scale * 30.0
                        output = puppet.render(target)
                        output = overlay(output, glasses)
                    elif now - last_good > 1.0:
                        output = overlay(portrait, glasses)
                    continue

                last_good = now

                if expression_session is not None:
                    session = expression_session
                    elapsed = now - session['start']
                    if elapsed >= 1.5:
                        session['samples'].append(current.copy())
                    if elapsed >= 3.0 and len(session['samples']) >= 8:
                        session['poses'].append(np.median(session['samples'], axis=0).astype(np.float32))
                        session['step'] += 1
                        session['start'] = now
                        session['samples'] = []
                        if session['step'] == 4:
                            neutral = session['poses'][0]
                            puppet.expression.calibrate(*session['poses'])
                            puppet.smoothed = None
                            head.reset()
                            mouth.reset()
                            body.reset()
                            expression_session = None
                            print('Expression calibration complete. Speak naturally.', flush=True)
                        else:
                            print(f'Expression calibration: {prompts[session["step"]]} (3 seconds)', flush=True)
                    continue

                if neutral is None:
                    calibration.append(current)
                    if len(calibration) >= 12:
                        neutral = np.median(calibration, axis=0).astype(np.float32)
                        print('Calibrated. Speak naturally.', flush=True)
                        report = time.monotonic()
                        render_count = 0
                        total_track = 0.0
                        total_render = 0.0
                        capture_baseline = sequence
                    continue

                t_render_start = time.perf_counter()
                target = puppet.target(current, neutral, args.strength, args.brow_strength)
                output = puppet.render(target)
                if live_mouth:
                    output = mouth.render(output, frame, current, target)
                output = overlay(output, glasses)
                output = head.render(output, current, neutral, args.head_motion)
                output = body.render(output, head.last_alpha, head.state, args.body_motion)
                t_finished = time.perf_counter()

                render_count += 1
                total_track += (t_tracked - t_track_start)
                total_render += (t_finished - t_render_start)

                if now - report >= 5.0:
                    dt_span = now - report
                    render_fps = render_count / dt_span
                    cam_fps = (sequence - capture_baseline) / dt_span if capture_baseline else render_fps
                    avg_track_ms = 1000.0 * total_track / max(1, render_count)
                    avg_render_ms = 1000.0 * total_render / max(1, render_count)
                    print(f'Render {render_fps:.1f} FPS | Camera {cam_fps:.1f} FPS | Track {avg_track_ms:.1f}ms | Warp {avg_render_ms:.1f}ms', flush=True)
                    report = now
                    render_count = 0
                    total_track = 0.0
                    total_render = 0.0
                    capture_baseline = sequence
    finally:
        if audio_stream is not None:
            audio_stream.stop()
            audio_stream.close()
        feed.close()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f'Could not continue: {exc}', file=sys.stderr)
        raise SystemExit(1)
