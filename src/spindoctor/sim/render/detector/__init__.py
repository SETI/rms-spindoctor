"""The renderer's detector stage (electron chain, vidicon, calibration).

Re-exports the pipeline entry point and the tested primitives.  The implementation is
split across these modules:

- :mod:`~spindoctor.sim.render.detector.params`: the resolved detector parameters,
  collapsed from the scene, the navigator block, and the detector defaults.
- :mod:`~spindoctor.sim.render.detector.chain`: the unit chain (the CCD electron
  chain, the vidicon DN chain, and the calibrated-I/F inversion), quantization, and
  the orchestration that applies every sub-effect at its physical point.
- :mod:`~spindoctor.sim.render.detector.shot_noise`: the per-pixel Poisson sampler.
- :mod:`~spindoctor.sim.render.detector.noise_stages`: the stochastic and structured
  noise sub-effects (dark current, hot pixels, banding, bias structure, and cosmic
  rays).
- :mod:`~spindoctor.sim.render.detector.electronics_stages`: the mechanics of the
  detector-stage artifact modes, each drawn onto a detector-grid plane.
- :mod:`~spindoctor.sim.render.detector.sqrt_lut`: the companding tables a camera
  downlinks its DN through, and the decode curve the ``sqrt_lut`` quantization
  sub-mode applies.
"""

from spindoctor.sim.render.detector.chain import (
    apply_detector,
    apply_saturation,
    quantize_dn,
)
from spindoctor.sim.render.detector.params import DetectorParams, resolve_detector_params

__all__ = [
    'DetectorParams',
    'apply_detector',
    'apply_saturation',
    'quantize_dn',
    'resolve_detector_params',
]
