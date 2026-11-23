"""The renderer's detector stage (electron chain, vidicon, calibration).

Re-exports the pipeline entry point and the tested primitives.  The implementation is
split across these modules:

- :mod:`~spindoctor.sim.render.detector.params`: the resolved detector parameters,
  collapsed from the scene, the navigator block, and the detector defaults.
- :mod:`~spindoctor.sim.render.detector.chain`: the unit chain (the CCD electron
  chain, the vidicon DN chain, and the calibrated-I/F inversion), and the
  orchestration that applies every effect at its physical point.
- :mod:`~spindoctor.sim.render.detector.shot_noise`: the per-pixel Poisson sampler.
- :mod:`~spindoctor.sim.render.detector.dark`: the dark current and the readout dark.
- :mod:`~spindoctor.sim.render.detector.bias`: the bias structure.
- :mod:`~spindoctor.sim.render.detector.banding`: the banding.
- :mod:`~spindoctor.sim.render.detector.pixel_defects`: the hot pixels and the
  anti-blooming pairs.
- :mod:`~spindoctor.sim.render.detector.fixed_pattern`: the fixed-pattern response and
  offsets.
- :mod:`~spindoctor.sim.render.detector.transients`: the cosmic rays and the radiation
  hits.
- :mod:`~spindoctor.sim.render.detector.saturation`: the full well, the column bloom,
  and the serial tail.
- :mod:`~spindoctor.sim.render.detector.shutter`: the shutter shading and the
  frame-transfer smear.
- :mod:`~spindoctor.sim.render.detector.vidicon`: the vidicon's residual image and beam
  bend.
- :mod:`~spindoctor.sim.render.detector.quantization`: the quantization sub-modes.
- :mod:`~spindoctor.sim.render.detector.sqrt_lut`: the companding tables a camera
  downlinks its DN through, and the decode curve the ``sqrt_lut`` quantization
  sub-mode applies.
- :mod:`~spindoctor.sim.render.detector.detector_maps`: the maps fixed to the emulated
  detector, its hot pixels and its column pattern.
"""

from spindoctor.sim.render.detector.chain import apply_detector
from spindoctor.sim.render.detector.params import DetectorParams, resolve_detector_params
from spindoctor.sim.render.detector.quantization import quantize_dn
from spindoctor.sim.render.detector.saturation import apply_saturation

__all__ = [
    'DetectorParams',
    'apply_detector',
    'apply_saturation',
    'quantize_dn',
    'resolve_detector_params',
]
