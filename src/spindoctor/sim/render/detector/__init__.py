"""The renderer's detector stage (electron chain, vidicon, calibration).

Re-exports the pipeline entry point and the tested primitives; the
implementation is split across :mod:`~spindoctor.sim.render.detector.params`
(resolved parameters), :mod:`~spindoctor.sim.render.detector.chain` (the unit
chain, quantization, and orchestration),
:mod:`~spindoctor.sim.render.detector.shot_noise` (the per-pixel Poisson
sampler), and :mod:`~spindoctor.sim.render.detector.noise_stages` (the
stochastic and structured sub-effects).
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
