"""Calibrated face-size lean proxy; coordinated head/torso composite."""
import cv2
import numpy as np
import time

class BodyMotion:
    def __init__(self, root, reference):
        self.h, self.w = reference.shape[:2]
        self.s = self.w / 960.0
        clean = cv2.imread(str(root / 'assets/empty_room.png'))
        if clean is None:
            raise RuntimeError('Missing assets/empty_room.png; extract the complete package.')
        self.clean = cv2.resize(clean, (self.w, self.h))
        self.reference = reference

        self.torso = np.zeros((self.h, self.w), np.uint8)
        pts = np.array([[405, 282], [531, 282], [565, 332], [641, 356], [719, 394], [772, 450],
                        [806, 532], [159, 532], [175, 456], [202, 400], [276, 359], [356, 329]], np.float32) * self.s
        cv2.fillPoly(self.torso, [pts.astype(np.int32)], 255)
        self.torso = cv2.GaussianBlur(self.torso.astype(np.float32) / 255.0, (0, 0), 1.2 * self.s)

        # Foreground desk/nameplate/microphone remains in front of the moving person.
        self.front = np.zeros((self.h, self.w), np.uint8)
        self.front[round(520 * self.s):] = 255
        for poly in [ [[350, 470], [612, 470], [616, 535], [348, 535]],
                      [[766, 365], [792, 370], [811, 397], [841, 450], [858, 522], [830, 522], [825, 455], [790, 405], [766, 392]] ]:
            cv2.fillPoly(self.front, [np.round(np.array(poly) * self.s).astype(np.int32)], 255)
        self.front = cv2.GaussianBlur(self.front.astype(np.float32) / 255.0, (0, 0), 0.6 * self.s)[:, :, None]

        hidden = ((self.front[:, :, 0] > 0.01) & (self.torso > 0.01)).astype(np.uint8) * 255
        self.underlay = cv2.inpaint(reference, hidden, 5, cv2.INPAINT_TELEA)
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

        y_desk = 530.0 * self.s
        y_collar = 280.0 * self.s
        src_quad = np.array([
            [160.0 * self.s, y_collar],
            [800.0 * self.s, y_collar],
            [810.0 * self.s, y_desk],
            [150.0 * self.s, y_desk]
        ], dtype=np.float32)

        dst_quad = src_quad.copy()
        dst_quad[0] += [-widen + lateral, breath - tilt]
        dst_quad[1] += [ widen + lateral, breath + tilt]
        dst_quad[2] += [ lateral * 0.3, 0]      # desk end stays put
        dst_quad[3] += [ lateral * 0.3, 0]

        P = cv2.getPerspectiveTransform(src_quad, dst_quad)

        # Hair clipping guard (unchanged)
        ys, xs = np.where(head_alpha > 0.2)
        if len(xs):
            top_pts = np.array([[[float(xs.min()), float(ys.min())], [float(xs.max()), float(ys.min())]]], dtype=np.float32)
            top_proj = cv2.perspectiveTransform(top_pts, P)[0, :, 1].min()
            if top_proj < 4.0 * self.s:
                P[1, 2] += (4.0 * self.s - top_proj)

        alpha = np.maximum(self.torso, head_alpha)
        source = image.astype(np.float32) * (1.0 - self.front) + self.underlay.astype(np.float32) * self.front
        premult = source * alpha[:, :, None]

        color = cv2.warpPerspective(premult, P, (self.w, self.h))
        mask = cv2.warpPerspective(alpha, P, (self.w, self.h))[:, :, None]

        out = color + self.clean.astype(np.float32) * (1.0 - mask)
        out = out * (1.0 - self.front) + self.reference.astype(np.float32) * self.front
        return np.clip(out, 0, 255).astype(np.uint8)
