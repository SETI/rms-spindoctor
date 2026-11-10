"""The detector stage of the forward model (electron chain, vidicon, calibration).

Re-exports the pipeline entry point and the tested primitives; the
implementation is split across :mod:`~spindoctor.sim.forward.detector.params`
(resolved parameters), :mod:`~spindoctor.sim.forward.detector.chain` (the unit
chain, quantization, and orchestration),
:mod:`~spindoctor.sim.forward.detector.shot_noise` (the per-pixel Poisson
sampler), and :mod:`~spindoctor.sim.forward.detector.noise_stages` (the
stochastic and structured sub-effects).
"""

from spindoctor.sim.forward.detector.chain import (
    apply_detector,
    apply_saturation,
    quantize_dn,
)
from spindoctor.sim.forward.detector.params import DetectorParams, resolve_detector_params

__all__ = [
    'DetectorParams',
    'apply_detector',
    'apply_saturation',
    'quantize_dn',
    'resolve_detector_params',
]
