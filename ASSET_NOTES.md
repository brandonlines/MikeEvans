# Asset provenance

Mike Evans is a fictional character from *The Three-Body Problem*.

`assets/source/` holds the two supplied photos:

- `cutout.webp`: the person on a transparent background. Only its transparency is
  used. Its colors carry a stray black mark on the glasses bridge.
- `scene.webp`: the same framing in the room. Every visible pixel comes from this.

`build_assets.py` crops both to 16:9 and generates the runtime layers:

- `mike-evans.png`: the scene, used to locate the face.
- `assets/head.png`: scene colors with the cutout's transparency. `layers.py`
  restricts its visible area to the head and a tapered neck at startup.
- `assets/background.png`: the scene with the head removed and the neck kept.
- `assets/empty_room.png`: the room with the person removed. Only a narrow band
  around the person's outline is ever seen, so deep inside that area it is just a
  smooth fill.
- `assets/torso_mask.png`: which pixels of the background plate move with the body.

The glasses are part of the photos, so there is no separate glasses layer. Eyebrow
and blink motion is kept away from the rims in `puppet.py`.
