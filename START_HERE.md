# Mike Evans V11 Fresh — separate skin and glasses

Fresh standalone package based on the uploaded V9 and repaired V10 renderer.
The head artwork has been edited to remove baked-in glasses. Eyebrows/eyelids and
wrinkle shading act on the glasses-free portrait, then separate frames are added.
The previous eye-freeze/protection patches are NOT present in this installation.

## Install
Extract into Downloads, then:

    cd ~/Downloads/MikeEvansV11Fresh
    bash 01_Setup.command

Wait for SETUP PASSED, then:

    bash 02_Start_Layered.command

No earlier folders needed. Setup installs Python/dependencies and downloads the
verified tracking model. Keep the folder in place after installation.

Look for Mike Evans V11 Fresh at startup. Press C to calibrate expressions, following
the prompts. R recenters, M toggles live mouth, P pauses, Q quits. OBS still captures
Mike Evans Layered - Live. Your microphone goes directly to Zoom. As in your V9,
the app also tries local microphone energy input for audio-reactive movement.

## Compare without glasses
Quit and run:

    bash 02_Start_Layered.command --no-glasses

This hides ONLY the separate glasses overlay. It is useful for checking that the
underlying portrait has no second set of frames. Restart normally to restore them.

## Scope and limits
Head/torso/neck/mouth code and assets otherwise follow V10. The glasses remain
independent of brow expression; head pose still transforms the assembled head.
This is not a full 3D glasses model. Reconstructed eye-area details may differ from
the original, and wide turns/blinks or neck texture differences can show artifacts.
The app cannot access newer local changes beyond the V9 ZIP you uploaded.

Validation: source portrait visually checked for removed glasses; actual model and
asset alignment exercised; brow extremes tested with stationary frame geometry before pose
and frame-core colour differences bounded by overlay transparency; separate no-glasses mode and startup tested. Live M3 appearance/FPS
still need testing. Existing calibration banners appear in OBS, so calibrate before
presenting.
