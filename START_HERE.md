# Mike Evans V11 Fresh

A webcam-driven photo puppet of Mike Evans, the fictional character from
*The Three-Body Problem*. The portrait and room are built from the two photos in
`assets/source/`.

## Install
Extract into Downloads, then:

    cd ~/Downloads/MikeEvansV11Fresh
    bash 01_Setup.command

Wait for SETUP PASSED, then:

    bash 02_Start_Layered.command

No earlier folders needed. Setup installs Python/dependencies and downloads the
verified tracking model. Keep the folder in place after installation.

## Using it on Zoom
The app opens two windows:

- **Mike Evans Layered - Live** is the clean feed. It never shows text or prompts.
  In OBS, add a macOS Screen Capture source set to this window (crop the title bar
  if it shows), then Start Virtual Camera. In Zoom, choose OBS Virtual Camera.
- **Mike Evans Layered - Operator** is for you: a small preview with status and
  calibration prompts. Do not capture this one.

Click either window to use the keys: C calibrates expressions (follow the prompts,
best done before going on air), R recenters, M toggles live mouth, P freezes the
picture, Q quits. Your microphone goes directly to Zoom. The app also listens to
mic level locally, only to keep the jaw moving while your face is out of view.

Do not minimize the Live window during a call. The app keeps running, but OBS cannot
capture a minimized window.

If your face leaves the frame, the avatar holds its last pose and eases back to
neutral after a second. If the camera drops out, the picture holds while the app
reconnects, and the Operator window tells you.

## Changing the photos
Replace `assets/source/cutout.webp` (the person on transparency) and
`assets/source/scene.webp` (same framing, in the room). They must be pixel-aligned.
Then run:

    .venv/bin/python build_assets.py

## Scope and limits
This is a 2.5D photo puppet, not a 3D model. Head turns are deliberately small, and
wide turns stop at the limit. The glasses are part of the photo and move with the
head. The room behind him is reconstructed, so a little smearing can show beside the
head at the far end of its movement.

Live appearance with a real face, the real mouth insert, and the OBS-to-Zoom path
still need a rehearsal on the machine you will present from.
