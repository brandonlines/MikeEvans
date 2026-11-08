"""Align transparent assets, protect jawline contour, erase wings, and smooth neck tone."""
import cv2
import numpy as np

def overlay(image, rgba):
    alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
    return np.clip(image * (1.0 - alpha) + rgba[:, :, :3] * alpha, 0, 255).astype(np.uint8)

def jaw_boundary(points, x, scale=1.0):
    """Measured lower jaw, constant outside ears: never cut through temples/hair."""
    from scipy.interpolate import PchipInterpolator
    ids = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148,
           152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454]
    jaw = np.asarray(points, np.float32)[ids]
    jaw = jaw[np.argsort(jaw[:, 0])]
    xs, indices = np.unique(jaw[:, 0], return_index=True)
    ys = jaw[indices, 1]
    curve = PchipInterpolator(xs, ys)
    return (curve(np.clip(x, xs[0], xs[-1])) + 5.0 * scale).astype(np.float32)


def prepare_layers(root, original, points, make_tracker, detect):
    h, w = original.shape[:2]
    plate = cv2.imread(str(root / 'assets/background.png'))
    head = cv2.imread(str(root / 'assets/head.png'), cv2.IMREAD_UNCHANGED)
    specs = cv2.imread(str(root / 'assets/glasses.png'), cv2.IMREAD_UNCHANGED)
    if plate is None or head is None or specs is None:
        raise RuntimeError('Missing assets; check assets/ folder.')
    if head.shape[2] != 4 or specs.shape[2] != 4:
        raise RuntimeError('Head and glasses need real alpha transparency.')
    
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
    chin = float(fitted[152, 1])
    cx = float(fitted[152, 0])
    scale = w / 960.0
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

    # 4. Landmark-anchored glasses
    ys, xs = np.where(specs[:, :, 3] > 16)
    if len(xs) == 0:
        raise RuntimeError('Empty glasses asset.')
    crop = specs[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    
    p33, p263, p168 = fitted[33], fitted[263], fitted[168]
    eye_dist = float(np.linalg.norm(p263 - p33))
    eye_center = (p33 + p263) * 0.5

    target_w = eye_dist * 1.87
    crop_h, crop_w = crop.shape[:2]
    target_h = target_w * (crop_h / crop_w)
    center_y = p168[1] + eye_dist * 0.05

    gx = max(0, int(round(eye_center[0] - target_w * 0.5)))
    gx2 = min(w, int(round(eye_center[0] + target_w * 0.5)))
    gy = max(0, int(round(center_y - target_h * 0.5)))
    gy2 = min(h, int(round(center_y + target_h * 0.5)))

    glasses = np.zeros((h, w, 4), np.uint8)
    if gx2 > gx and gy2 > gy:
        glasses[gy:gy2, gx:gx2] = cv2.resize(crop, (gx2 - gx, gy2 - gy))

    return portrait, alpha, plate, glasses, fitted
