"""Calibrated face-size lean proxy; coordinated head/torso composite."""
import cv2
import numpy as np
import time

from puppet import canonical, rig_scale


class BodyMotion:
    def __init__(self, root, reference, points):
        self.h, self.w = reference.shape[:2]
        self.s = rig_scale(points)
        clean = cv2.imread(str(root / 'assets/empty_room.png'))
        torso = cv2.imread(str(root / 'assets/torso_mask.png'), cv2.IMREAD_GRAYSCALE)
        if clean is None or torso is None:
            raise RuntimeError('Missing assets/empty_room.png or assets/torso_mask.png.')
        self.clean = cv2.resize(clean, (self.w, self.h))
        self.torso = cv2.resize(torso, (self.w, self.h)).astype(np.float32) / 255.0

        # The torso pivots about the bottom of the frame. Motion is applied one eye-span above
        # the chin; a lower control edge would exaggerate breathing at the top of the head.
        xs = np.where(self.torso > 0.5)[1]
        left, right = float(xs.min()), float(xs.max())
        y_control = float(points[152, 1]) - canonical(points)[3]
        self.src_quad = np.array([[left, y_control], [right, y_control],
                                  [right, self.h], [left, self.h]], dtype=np.float32)
        self.reset()

    def reset(self):
        self.start_time = time.monotonic()

    def render(self, image, head_alpha, head_state, strength=1.0):
        if strength == 0:
            return image

        lean = float(head_state[5])     # same value the head uses
        yaw = float(head_state[1])
        roll = float(head_state[2])
        tx = float(head_state[3])

        now = time.monotonic()
        breath = np.sin((now - self.start_time) * 2.0 * np.pi * 0.2) * (1.6 * self.s)

        # Torso gets a fraction of the head motion, with the same sign
        widen = lean * 0.6 * 150.0 * self.s * strength        # lean in = shoulders widen
        tilt = roll * 0.6 * 120.0 * self.s * strength         # head roll = shoulder tilt
        lateral = (tx * 0.5 + yaw * 90.0) * self.s * strength # head turn drags torso

        dst_quad = self.src_quad.copy()
        dst_quad[0] += [-widen + lateral, breath - tilt]
        dst_quad[1] += [ widen + lateral, breath + tilt]
        dst_quad[2] += [ lateral * 0.3, 0]      # bottom of frame stays put
        dst_quad[3] += [ lateral * 0.3, 0]

        P = cv2.getPerspectiveTransform(self.src_quad, dst_quad)

        # Hair clipping guard (unchanged)
        x, y, w, h = cv2.boundingRect((head_alpha > 0.2).view(np.uint8))
        if w:
            top_pts = np.array([[[x, y], [x + w - 1, y]]], dtype=np.float32)
            top_proj = cv2.perspectiveTransform(top_pts, P)[0, :, 1].min()
            if top_proj < 4.0 * self.s:
                P[1, 2] += (4.0 * self.s - top_proj)

        # Replicate the border so the jacket stays solid along the bottom edge of the frame.
        alpha = np.maximum(self.torso, head_alpha)
        color = cv2.warpPerspective(image, P, (self.w, self.h), borderMode=cv2.BORDER_REPLICATE)
        mask = cv2.warpPerspective(alpha, P, (self.w, self.h), borderMode=cv2.BORDER_REPLICATE)
        return cv2.blendLinear(color, self.clean, mask, 1.0 - mask)
