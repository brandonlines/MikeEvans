"""Bounded 2.5D depth surface with 6-state lean tracking, neck-pivot scaling, and collar blending."""
import cv2
import numpy as np
import time
from layers import jaw_boundary
from puppet import canonical, rig_scale

# Rigid spans that set the apparent face size: across the eyes, and forehead to nose base.
# A head turn shortens one and a nod the other; only moving toward the camera grows both.
SIZE_SPANS = ([33, 10], [263, 2])
FIELD_STEP = 2  # The warp field is smooth, so it is solved on a coarser grid and upsampled.


def face_size(points):
    return np.linalg.norm(points[SIZE_SPANS[0]] - points[SIZE_SPANS[1]], axis=1)


class HeadMotion:
    def __init__(self, image, points, alpha, background):
        self.image = image
        h, w = image.shape[:2]
        self.scale = rig_scale(points)
        self.alpha = alpha.astype(np.float32)
        self.background = background.copy()

        # Work region: the head layer plus room for it to move.
        ys, xs = np.where(self.alpha > 0.01)
        pad = int(40 * self.scale)
        self.x0 = max(0, int(xs.min()) - pad)
        self.y0 = max(0, int(ys.min()) - pad)
        self.x1 = min(w, int(xs.max()) + pad + 1)
        self.y1 = min(h, int(ys.max()) + pad + 1)
        rw, rh = self.x1 - self.x0, self.y1 - self.y0
        self.size = (rw, rh)
        self.background_roi = np.ascontiguousarray(self.background[self.y0:self.y1, self.x0:self.x1])

        # Coarse grid whose samples sit where cv2.resize expects them when scaling back up.
        gw, gh = -(-rw // FIELD_STEP), -(-rh // FIELD_STEP)
        gx = self.x0 + (np.arange(gw, dtype=np.float32) + 0.5) * (rw / gw) - 0.5
        gy = self.y0 + (np.arange(gh, dtype=np.float32) + 0.5) * (rh / gh) - 0.5
        self.xx, self.yy = np.meshgrid(gx, gy)

        # Head ellipsoid, placed from the face (proportions in eye-spans).
        eyes = (points[33] + points[263]) / 2.0
        span = canonical(points)[3]
        self.cx = float(eyes[0])
        self.cy = float(eyes[1]) + 0.058 * span
        self.rx = 1.016 * span
        self.ry = 1.572 * span
        self.nose_y = float(points[1, 1]) - 0.113 * span
        self.chin = float(points[152, 1])

        # Synchronize rotation mobility with anatomical jawline
        dy = self.yy - jaw_boundary(points, self.xx, self.scale)
        t_mob = np.clip(np.maximum(0.0, dy) / (24.0 * self.scale), 0.0, 1.0)
        self.mobility = np.where(dy <= 0, 1.0, (1.0 - t_mob**2)**2).astype(np.float32)

        # Depth surface normals for directional relighting
        yy, xx = np.mgrid[self.y0:self.y1, self.x0:self.x1].astype(np.float32)
        Z = self.depth(xx, yy)
        dzdx = cv2.Sobel(Z, cv2.CV_32F, 1, 0, ksize=3)
        dzdy = cv2.Sobel(Z, cv2.CV_32F, 0, 1, ksize=3)
        norm = np.sqrt(dzdx**2 + dzdy**2 + 1.0)
        self.normals = [-dzdx / norm, -dzdy / norm, 1.0 / norm]
        self.light_dir = np.array([-0.2, -0.7, 0.6], dtype=np.float32)
        self.light_dir /= np.linalg.norm(self.light_dir)

        self.last_alpha = np.zeros((h, w), dtype=np.float32)
        self.reset()

    def reset(self):
        self.state = np.zeros(6, np.float32)   # [pitch, yaw, roll, tx, ty, lean]
        self.last = time.monotonic()

    @staticmethod
    def pose(points):
        ids = [1, 152, 33, 263, 61, 291]
        model = np.array([[0, 0, 35], [0, 90, -15], [-45, -35, 0], [45, -35, 0], [-30, 45, 0], [30, 45, 0]], np.float64)
        p = np.asarray(points, np.float64)
        focal = max(640.0, float(np.ptp(p[:, 0]) * 3))
        camera = np.array([[focal, 0, 320], [0, focal, 240], [0, 0, 1]], np.float64)
        ok, r, t = cv2.solvePnP(model, p[ids], camera, None, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return np.eye(3)
        return cv2.Rodrigues(r)[0]

    def depth(self, x, y):
        q = np.maximum(0, 1 - ((x - self.cx) / self.rx)**2 - ((y - self.cy) / self.ry)**2)
        nose = np.exp(-((x - self.cx) / (19 * self.scale))**2 - ((y - self.nose_y) / (40 * self.scale))**2)
        return 65 * self.scale * np.sqrt(q) + 18 * self.scale * nose

    def render(self, image, current, neutral, amount=1.0):
        if amount == 0:
            self.state[:] = 0
            self.last_alpha = np.zeros_like(self.background[:, :, 0], dtype=np.float32)
            self.last_alpha[self.y0:self.y1, self.x0:self.x1] = self.alpha[self.y0:self.y1, self.x0:self.x1]
            return image

        relative = self.pose(current) @ self.pose(neutral).T
        angles = cv2.Rodrigues(relative)[0].ravel()
        _, c, basis, d = canonical(current)
        _, b, base_basis, bd = canonical(neutral)

        roll = np.arctan2(basis[0, 1], basis[0, 0]) - np.arctan2(base_basis[0, 1], base_basis[0, 0])
        angles[2] = np.arctan2(np.sin(roll), np.cos(roll)) * 1.25

        lean = float(np.max(face_size(current) / face_size(neutral))) - 1.0
        lean = np.sign(lean) * max(0.0, abs(lean) - 0.015)

        desired = np.r_[np.clip(angles, [-0.075, -0.12, -0.12], [0.075, 0.12, 0.12]),
                        np.clip((c - b) / bd * 25, -4, 4),
                        np.clip(lean * 0.8, -0.06, 0.06)] * amount

        now = time.monotonic()
        dt = np.clip(now - self.last, 0.005, 0.1)
        self.last = now
        weight = 1.0 - np.exp(-dt / 0.055)
        self.state += weight * (desired - self.state)

        R = cv2.Rodrigues(self.state[:3].astype(np.float64))[0].astype(np.float32)
        tx, ty = self.state[3:5] * self.scale
        zoom = 1.0 + float(self.state[5])
        px, py = self.cx, self.chin + 25.0 * self.scale

        xe = px + (self.xx - px) / zoom
        ye = py + (self.yy - py) / zoom

        inv = np.linalg.inv(R[:2, :2])
        u = xe - self.cx - tx
        v = ye - self.cy - ty
        sx = xe.copy()
        sy = ye.copy()

        for _ in range(4):
            z = self.depth(sx, sy) - 35 * self.scale
            a = u - R[0, 2] * z
            b = v - R[1, 2] * z
            sx = inv[0, 0] * a + inv[0, 1] * b + self.cx
            sy = inv[1, 0] * a + inv[1, 1] * b + self.cy

        sx = cv2.resize(xe + (sx - xe) * self.mobility, self.size, interpolation=cv2.INTER_LINEAR)
        sy = cv2.resize(ye + (sy - ye) * self.mobility, self.size, interpolation=cv2.INTER_LINEAR)

        color = cv2.remap(image, sx, sy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        alpha = cv2.remap(self.alpha, sx, sy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        light = R.T @ self.light_dir
        nx, ny, nz = self.normals
        diffuse = nx * light[0] + ny * light[1] + nz * light[2]
        intensity = np.clip(1.0 + (diffuse - 0.5) * 0.28, 0.86, 1.14)
        color = cv2.multiply(color, cv2.merge([intensity] * 3), dtype=cv2.CV_8U)

        self.last_alpha = np.zeros_like(self.background[:, :, 0], dtype=np.float32)
        self.last_alpha[self.y0:self.y1, self.x0:self.x1] = alpha

        out = self.background.copy()
        out[self.y0:self.y1, self.x0:self.x1] = cv2.blendLinear(color, self.background_roi, alpha, 1.0 - alpha)
        return out
