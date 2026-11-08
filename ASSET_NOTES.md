# V11 asset provenance

The supplied V9/V10 head asset contained baked-in glasses. V11 replaces it with an
AI-edited transparent portrait with those glasses removed and the eyes, brows and
nose bridge reconstructed. The edit requested preservation of identity, pose,
hair, beard, neck, jacket, plaque, composition and transparency. Generated details
can differ slightly from the input. Runtime landmark alignment corrects scale and
position before compositing.

The separate glasses.png frame overlay and remaining scene assets are inherited
from the uploaded V9/V10 package. Expressions deform the glasses-free skin first;
the frame overlay is then composited, before shared head pose is applied.

The generated source retains torso pixels, but layers.py restricts its visible
alpha to the head and tapered neck. This is a 2.5D photo puppet, not a complete
3D head or frame model.
