"""Rebuild the rig's scene layers from the two pixel-aligned photos in assets/source/.

    cutout.webp   the person on a transparent background (supplies the matte)
    scene.webp    the same framing in the room (supplies every visible pixel)

Writes mike-evans.png and assets/{head,background,empty_room,torso_mask}.png.
Run with: .venv/bin/python build_assets.py
"""
from pathlib import Path

import cv2
import numpy as np

from app import detect, make_tracker
from layers import jaw_boundary
from puppet import canonical, rig_scale

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / 'assets'


def crop_16_9(image):
    """Keep the top of the photo, where the head is; only the jacket hem is lost."""
    width = image.shape[1]
    return image[:width * 9 // 16]


def disc(radius):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


def fill_background(scene, hole):
    """Continue each row of the room across the hole.

    The rig only ever reveals a narrow band just inside the person's outline, so the room
    needs to be believable there; deep inside the hole is never on screen.
    """
    known = (~hole).astype(np.float32)
    weight = np.maximum(cv2.GaussianBlur(known, (0, 0), 4), 1e-3)[:, :, None]
    soft = cv2.GaussianBlur(scene.astype(np.float32) * known[:, :, None], (0, 0), 4) / weight
    filled = soft.copy()
    for y in range(scene.shape[0]):
        gap = np.flatnonzero(hole[y])
        have = np.flatnonzero(~hole[y])
        if len(gap) and len(have):
            for channel in range(3):
                filled[y, gap, channel] = np.interp(gap, have, soft[y, have, channel])
    filled = cv2.GaussianBlur(filled, (1, 31), sigmaX=0, sigmaY=6)
    seam = cv2.GaussianBlur(hole.astype(np.float32), (0, 0), 2)[:, :, None]
    seam = np.maximum(seam, hole[:, :, None].astype(np.float32))
    return np.clip(scene * (1 - seam) + filled * seam, 0, 255).astype(np.uint8)


def main():
    cutout = crop_16_9(cv2.imread(str(ASSETS / 'source/cutout.webp'), cv2.IMREAD_UNCHANGED))
    scene = crop_16_9(cv2.imread(str(ASSETS / 'source/scene.webp'), cv2.IMREAD_COLOR))
    if cutout.shape[2] != 4 or cutout.shape[:2] != scene.shape[:2]:
        raise RuntimeError('Need a transparent cutout that is pixel-aligned with the scene photo.')
    matte = cutout[:, :, 3].astype(np.float32) / 255.0

    with make_tracker() as tracker:
        points = detect(tracker, scene)
    if points is None:
        raise RuntimeError('Could not find the face in the scene photo.')

    h, w = scene.shape[:2]
    scale = rig_scale(points)
    eye_span = canonical(points)[3]
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    boundary = jaw_boundary(points, xx, scale)

    # Keep some beard behind the jaw on the plate so head motion never opens a gap at the neck.
    # Beside the jaw the cut stays at the ear floor, so no piece of ear is left behind.
    head_limit = np.maximum(boundary - 14.0 * scale, boundary.min())
    person = (matte > 0.02).astype(np.uint8)

    # The head comes off the plate, along with the flyaway hairs the matte leaves behind.
    # Those only stick out around the hair, so leave the room beside the ears untouched.
    near_person = cv2.dilate(person, disc(10)) > 0
    near_hair = (cv2.dilate(person, disc(round(0.2 * eye_span))) > 0) & (yy < points[[33, 263], 1].mean())
    head_zone = (near_person | near_hair) & (yy < head_limit)
    hole = head_zone | near_person
    room = fill_background(scene, hole)

    erase = cv2.GaussianBlur(head_zone.astype(np.float32), (0, 0), 3)[:, :, None]
    plate = np.clip(scene * (1 - erase) + room * erase, 0, 255).astype(np.uint8)

    ramp = np.clip((yy - head_limit) / (6.0 * scale), 0.0, 1.0)
    torso = matte * (ramp * ramp * (3.0 - 2.0 * ramp))

    head = np.dstack([scene, cutout[:, :, 3]])

    cv2.imwrite(str(ROOT / 'mike-evans.png'), scene)
    cv2.imwrite(str(ASSETS / 'head.png'), head)
    cv2.imwrite(str(ASSETS / 'background.png'), plate)
    cv2.imwrite(str(ASSETS / 'empty_room.png'), room)
    cv2.imwrite(str(ASSETS / 'torso_mask.png'), np.round(torso * 255).astype(np.uint8))
    print(f'Built {w}x{h} layers. Eye span {eye_span:.1f}px, chin y={points[152, 1]:.1f}.')


if __name__ == '__main__':
    main()
