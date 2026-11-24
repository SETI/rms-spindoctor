"""The tuning keys every distance-transform technique holds.

``BodyLimbNav``, ``BodyTerminatorNav``, and ``RingEdgeNav`` share the coarse-NCC,
Levenberg-Marquardt, and Tukey machinery of :mod:`spindoctor.nav_technique.dt_fitting`,
and with it the keys that tune that machinery and judge its result.  :class:`DTTuning`
declares those keys once.  Each technique's own tuning type extends it with the keys
only that technique reads, and the schema core parses the technique's ``tuning`` block
into that type when the configuration loads.
"""

from dataclasses import dataclass

from spindoctor.nav_technique.dt_fit_gates import DTFitGateConfig
from spindoctor.schema import SCHEMA_CONFIG, NonNeg, NonNegInt, Positive, Unit

__all__ = ['DTTuning']


@dataclass(frozen=True)
class DTTuning(DTFitGateConfig):
    """The tuning keys shared by the distance-transform techniques.

    The keys of the shared fit-quality gates come from :class:`DTFitGateConfig`, so a
    technique passes its tuning to
    :func:`~spindoctor.nav_technique.dt_fit_gates.evaluate_dt_fit_gates` as the gate
    configuration.

    Parameters:
        spurious_dt_rms_factor: A final DT residual RMS above this many times the
            smallest per-vertex sigma marks the fit spurious. Zero leaves the floor
            as the whole threshold.
        spurious_dt_floor_px: The least RMS threshold of that test, in pixels.
        spurious_min_inliers: Fewer Tukey inliers than this marks the fit spurious.
        spurious_min_inlier_fraction: A Tukey inlier fraction below this marks the fit
            spurious.
        spurious_max_lm_displacement_px: An LM that moves farther than this from the
            integer coarse seed, in pixels, marks the fit spurious.
        lm_trust_region_px: The farthest the LM may move from the coarse seed, in
            pixels.  A trial step that would land outside it is refused.
        lm_tikhonov_alpha: The strength of the Tikhonov anchor toward the coarse seed;
            0 disables it.
        at_edge_tolerance_px: Pixels of slack around the search-window bounds for the
            at-edge test.
        rotation_at_edge_fraction: The fraction of the largest allowed rotation at
            which a fitted rotation trips the at-edge flag.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    spurious_dt_rms_factor: NonNeg
    spurious_dt_floor_px: NonNeg
    spurious_min_inliers: NonNegInt
    spurious_min_inlier_fraction: Unit
    spurious_max_lm_displacement_px: NonNeg
    lm_trust_region_px: Positive
    lm_tikhonov_alpha: NonNeg
    at_edge_tolerance_px: NonNeg
    rotation_at_edge_fraction: Unit
