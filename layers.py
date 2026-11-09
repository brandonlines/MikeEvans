"""Align the transparent head asset, protect jawline contour, erase wings, and smooth neck tone."""
import cv2
import numpy as np

from puppet import rig_scale

EAR_FLOOR = 0.8  # Eye-spans below the eye line: under the ear lobes, above the shoulders.


def overlay(image, rgba):
    alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
    return np.clip(image * (1.0 - alpha) + rgba[:, :, :3] * alpha, 0, 255).astype(np.uint8)


def jaw_boundary(points, x, scale=1.0):
    """Measured lower jaw, held below the ear lobes beside it: never cut through ears/hair."""
    from scipy.interpolate import PchipInterpolator
    ids = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148,
           152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454]
    points = np.asarray(points, np.float32)
    jaw = points[ids]
    jaw = jaw[np.argsort(jaw[:, 0])]
    xs, indices = np.unique(jaw[:, 0], return_index=True)
    ys = jaw[indices, 1]
    curve = PchipInterpolator(xs, ys)
    jaw_line = curve(np.clip(x, xs[0], xs[-1])) + 5.0 * scale
    ear_floor = points[[33, 263], 1].mean() + EAR_FLOOR * np.linalg.norm(points[263] - points[33])
    return np.maximum(jaw_line, ear_floor).astype(np.float32)


def prepare_layers(root, original, points, make_tracker, detect):
    h, w = original.shape[:2]
    plate = cv2.imread(str(root / 'assets/background.png'))
    head = cv2.imread(str(root / 'assets/head.png'), cv2.IMREAD_UNCHANGED)
    if plate is None or head is None:
        raise RuntimeError('Missing assets; check assets/ folder.')
    if head.shape[2] != 4:
        raise RuntimeError('Head needs real alpha transparency.')

    plate = cv2.resize(plate, (w, h))
    scaled = cv2.resize(head, (w, h))
    view = overlay(np.full_like(original, 127), scaled)
    with make_tracker() as tracker:
        src = detect(tracker, view)
    if src is None:
        raise RuntimeError('Could not align the head asset.')

    ids = [33, 263, 1, 61, 291, 152]
    matrix, _ = cv2.estimateAffinePartial2D(src[ids], points[ids], method=cv2.LMEDS)
    if matrix is None:
        raise RuntimeError('Head asset alignment failed.')

    aligned = cv2.warpAffine(scaled, matrix, (w, h), borderMode=cv2.BORDER_CONSTANT)
    fitted = cv2.transform(src[None], matrix.astype(np.float32))[0]
    cx = float(fitted[152, 0])
    scale = rig_scale(fitted)
    yy, xx = np.mgrid[:h, :w].astype(np.float32)

    # Preserve the asset's natural silhouette above its measured lower jaw.
    dy = yy - jaw_boundary(fitted, xx, scale)

    # 2. Below the jawline (dy > 0): restrict to central neck column, erasing lateral collar flaps
    allowed_hw = np.maximum(22.0 * scale, 48.0 * scale - 0.8 * dy)
    dist_x = np.abs(xx - cx)
    feather_x = 8.0 * scale
    t_x = np.clip((dist_x - (allowed_hw - feather_x)) / feather_x, 0.0, 1.0)
    fade_x = 1.0 - t_x * t_x * (3.0 - 2.0 * t_x)

    # Vertical taper down the neck into the shirt collar (terminates within 26px below jaw)
    t_y = np.clip(dy / (26.0 * scale), 0.0, 1.0)
    fade_y = (1.0 - t_y**2)**2

    blend = t_y * t_y * (3.0 - 2.0 * t_y)
    fade = (1.0 - blend * (1.0 - fade_x * fade_y)).astype(np.float32)
    original_alpha = aligned[:, :, 3].astype(np.float32) / 255.0
    aligned[:, :, 3] = np.round(original_alpha * fade * 255.0).astype(np.uint8)

    # 3. Seamless skin-tone gradient: gently transition head skin into plate neck
    neck_zone = (dy > 0) & (dy < 24.0 * scale) & (dist_x < 36.0 * scale)
    if neck_zone.sum() > 20:
        head_color = np.median(aligned[:, :, :3][neck_zone], axis=0).astype(np.float32)
        plate_color = np.median(plate[neck_zone], axis=0).astype(np.float32)
        diff = np.clip(plate_color - head_color, -35, 35)
        ramp = np.clip(dy / (24.0 * scale), 0.0, 1.0)[:, :, None]
        aligned[:, :, :3] = np.clip(aligned[:, :, :3].astype(np.float32) + diff * ramp * neck_zone[:, :, None], 0, 255).astype(np.uint8)

    alpha = aligned[:, :, 3].astype(np.float32) / 255.0
    portrait = overlay(plate, aligned)

    return portrait, alpha, plate, fitted
