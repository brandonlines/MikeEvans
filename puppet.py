"""Lightweight 2D photo puppet with locked glasses geometry and natural speech aperture."""
import cv2
import numpy as np
from scipy.spatial import Delaunay

OVAL = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109]
EYES = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246,263,249,390,373,374,380,381,382,362,398,384,385,386,387,388,466]

BROWS_UPPER_R = [70, 63, 105, 66]
BROWS_UPPER_L = [296, 334, 293, 300]
BROWS_FRAME_LOCK = [107, 55, 65, 52, 53, 46, 336, 285, 295, 282, 283, 276]
BROWS = sorted(set(BROWS_UPPER_R + BROWS_UPPER_L + BROWS_FRAME_LOCK))

FRAME_ANCHORS = [168, 6, 197, 195, 5, 4, 1, 2, 8, 9, 151, 111, 116, 226, 340, 345, 446, 127, 356]

LIPS_OUTER_UPPER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
LIPS_INNER_UPPER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
LIPS_OUTER_LOWER = [146, 91, 181, 84, 17, 314, 405, 321, 375]
LIPS_INNER_LOWER = [95, 88, 178, 87, 14, 317, 402, 318, 324]
LIPS = sorted(set(LIPS_OUTER_UPPER + LIPS_INNER_UPPER + LIPS_OUTER_LOWER + LIPS_INNER_LOWER + [78, 308]))

IDS = np.array(sorted(set(OVAL + EYES + BROWS + LIPS + FRAME_ANCHORS + [50, 280, 205, 425, 101, 330])), dtype=int)


def canonical(points):
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[0] < 468 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("Invalid face landmarks")
    left, right = points[33], points[263]
    vector = right - left
    distance = float(np.linalg.norm(vector))
    if distance < 1e-5:
        raise ValueError("Face landmarks collapsed; recalibrate")
    axis = vector / distance
    basis = np.array([axis, [-axis[1], axis[0]]], dtype=np.float32)
    center = (left + right) / 2
    return ((points - center) @ basis.T) / distance, center, basis, distance


class PhotoPuppet:
    def __init__(self, image_bgr, landmarks):
        self.source = image_bgr.copy()
        self.points = np.asarray(landmarks, dtype=np.float32)
        self.canon, self.center, self.basis, self.scale = canonical(self.points)
        h, w = image_bgr.shape[:2]
        face = self.points[OVAL]
        low = face.min(axis=0)
        high = face.max(axis=0)
        padding = (high - low) * 0.25
        x, y = np.maximum(0, np.floor(low - padding)).astype(int)
        x2, y2 = np.minimum([w, h], np.ceil(high + padding)).astype(int)
        self.x, self.y, self.w, self.h = int(x), int(y), int(x2 - x), int(y2 - y)
        self.origin = np.array([x, y], np.float32)
        self.patch = image_bgr[y:y2, x:x2].copy()

        anchors = []
        for t in np.linspace(0, 1, 6):
            anchors.extend([[t * (self.w - 1), 0],
                            [t * (self.w - 1), self.h - 1],
                            [0, t * (self.h - 1)],
                            [self.w - 1, t * (self.h - 1)]])
        self.anchors = np.unique(np.array(anchors, np.float32), axis=0)
        vertices = np.vstack([self.points[IDS] - self.origin, self.anchors])
        tri = Delaunay(vertices)

        gh, gw = 96, 96
        yy, xx = np.mgrid[0:self.h - 1:complex(gh), 0:self.w - 1:complex(gw)]
        grid = np.column_stack([xx.ravel(), yy.ravel()])
        simplex = tri.find_simplex(grid, tol=1e-6)
        valid = simplex >= 0
        safe = np.maximum(simplex, 0)
        trans = tri.transform[safe]
        bary = np.einsum('nij,nj->ni', trans[:, :2], grid - trans[:, 2])
        self.weights = np.c_[bary, 1 - bary.sum(axis=1)].astype(np.float32)
        self.weights[~valid] = 0
        self.indices = tri.simplices[safe]
        self.grid_shape = (gh, gw)

        yy, xx = np.mgrid[:self.h, :self.w]
        self.xx = xx.astype(np.float32)
        self.yy = yy.astype(np.float32)
        self.smoothed = None

        from expression_map import ExpressionMap
        self.expression = ExpressionMap()

    def target(self, current, neutral, strength=0.35, brow_strength=1.1):
        now, _, basis, _ = canonical(current)
        base, _, base_basis, _ = canonical(neutral)
        raw = np.clip(now - base, -0.10, 0.10) * self.scale * strength

        delta = np.zeros_like(raw)

        # 1. Natural curved speech aperture
        opening = max(0.0, (now[14, 1] - now[13, 1]) - (base[14, 1] - base[13, 1]))
        jaw = min(opening, 0.16) * self.scale * 0.44

        # Controlled horizontal mouth expansion
        delta[LIPS, 0] = raw[LIPS, 0] * 0.45

        # Upper lip: small upward travel (0.15) to maintain mustache stability
        delta[LIPS_OUTER_UPPER, 1] = raw[LIPS_OUTER_UPPER, 1] * 0.15
        delta[LIPS_INNER_UPPER, 1] = raw[LIPS_INNER_UPPER, 1] * 0.20

        # Lower lip: deep center descent (landmark 14 & 17) creating an elliptical curve
        delta[LIPS_OUTER_LOWER, 1] = raw[LIPS_OUTER_LOWER, 1] * 0.85 + jaw * 0.75
        delta[LIPS_INNER_LOWER, 1] = raw[LIPS_INNER_LOWER, 1] * 0.95 + jaw * 0.90
        delta[[14, 17], 1] += jaw * 0.15
        delta[[78, 308], 1] = jaw * 0.20

        # Jaw drop to chin and lower cheeks
        delta[[152, 148, 377, 176, 400], 1] += jaw
        delta[[205, 425, 50, 280], 1] += jaw * 0.35

        # 2. Eyebrow Motion: strictly decoupled from glasses frame
        brows, closure = self.expression.values(current, neutral)

        delta[BROWS_UPPER_R, 1] = -brows[0] * self.scale * 0.040 * brow_strength
        delta[BROWS_UPPER_L, 1] = -brows[1] * self.scale * 0.040 * brow_strength

        furrow = float(max(0.0, -brows[0], -brows[1]))
        if furrow > 0.05:
            delta[BROWS_UPPER_R, 0] = furrow * self.scale * 0.012
            delta[BROWS_UPPER_L, 0] = -furrow * self.scale * 0.012

        delta[BROWS_FRAME_LOCK] = 0.0
        delta[FRAME_ANCHORS] = 0.0

        # 3. Eyelids: Smooth blink closure inside frames
        delta[EYES] = 0.0
        for side, upper, lower in [(0, [159, 158, 160], [145, 153, 144]),
                                   (1, [386, 385, 387], [374, 380, 373])]:
            gap = max(0.0, self.canon[lower, 1].mean() - self.canon[upper, 1].mean())
            cl = closure[side]
            delta[upper, 1] = cl * gap * self.scale * 0.65
            delta[lower, 1] = -cl * gap * self.scale * 0.25

        target = self.points + delta @ self.basis
        if self.smoothed is None:
            self.smoothed = target
        else:
            self.smoothed = 0.75 * target + 0.25 * self.smoothed
        return self.smoothed.copy()

    def render(self, target):
        displacement = np.vstack([target[IDS] - self.points[IDS], np.zeros_like(self.anchors)])
        field = np.einsum('ni,nij->nj', self.weights, displacement[self.indices]).reshape(*self.grid_shape, 2)
        field = cv2.resize(field, (self.w, self.h), interpolation=cv2.INTER_LINEAR)

        warped = cv2.remap(self.patch, self.xx - field[:, :, 0], self.yy - field[:, :, 1],
                           cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        output = self.source.copy()
        output[self.y:self.y + self.h, self.x:self.x + self.w] = warped
        return output
