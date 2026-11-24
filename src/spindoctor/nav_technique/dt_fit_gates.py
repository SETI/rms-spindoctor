"""Shared fit-quality gates for the distance-transform techniques.

The DT techniques (``BodyLimbNav``, ``BodyTerminatorNav``, ``RingEdgeNav``)
share the coarse-NCC + Levenberg-Marquardt + Tukey machinery of
:mod:`spindoctor.nav_technique.dt_fitting`; this module is the shared verdict
layer over that machinery's health signals.  Three signals feed it:

- **LM convergence.**  ``LMRefineResult.converged=False`` means the LM bailed
  at its iteration cap without ever meeting the step-norm tolerance -- an
  unverified fit.  The consequence is a confidence DEMOTION, not a reject: an
  LM at the cap usually sits near its (trust-region-bounded) seed, so a hard
  reject would strafe healthy frames that merely oscillate at the tolerance
  boundary, but the fit has not earned a high-tier vote either.
- **Polarity-rejection fraction.**  The fraction of model vertices whose
  local image gradient direction disagrees with the model's outward normal at
  the seed.  Healthy multi-body frames run around 11 % (small secondary
  bodies on an ansa contribute wrong-polarity vertices while the dominant
  limb fits cleanly), so the standalone threshold is set far above that; the
  discriminating form is the COMBINED gate -- an unconverged LM whose
  polarity rejection is also elevated is the signature of a coarse-seed
  mis-lock that the LM could neither escape nor verify (the Cassini Tethys
  body-mostly-offscreen investigation measured 12.7 % rejection at the
  30-iteration cap on a seed 6.8 px off).
- **Coarse-NCC peak quality.**  The winning coarse shift's per-vertex match
  fraction (:class:`~spindoctor.nav_technique.dt_fitting.CoarseSearchResult`).
  A seed whose best shift still leaves most of the model off every detected
  edge never had a lock to refine; the LM then polishes noise.  This gate
  ships as plumbing plus a conservative default; a library-calibrated
  threshold lock is separate follow-on calibration work.

All thresholds are per-technique tuning keys (``techniques.<key>.tuning`` in
``config_510_techniques.yaml``).  :class:`DTFitGateConfig` declares them, and each DT
technique's tuning type extends it (:class:`~spindoctor.nav_technique.dt_tuning.DTTuning`),
so a technique passes its own tuning as the gate configuration.
"""

from __future__ import annotations

from dataclasses import dataclass

from spindoctor.nav_technique.dt_fitting import LMRefineResult
from spindoctor.schema import SCHEMA_CONFIG, Unit

__all__ = [
    'DTFitGateConfig',
    'DTFitGateVerdict',
    'evaluate_dt_fit_gates',
]


@dataclass(frozen=True)
class DTFitGateConfig:
    """The tuning keys of the shared DT fit-quality gates.

    Parameters:
        lm_unconverged_confidence_cap: Post-sigmoid confidence cap applied
            when the LM exits at its iteration cap with ``converged=False``.
            Sits below the high tier's confidence boundary so an unverified
            fit cannot carry a high-tier result on its own, while genuine
            corroboration by other techniques can still rescue the frame.
        spurious_max_polarity_rejection_fraction: Standalone hard gate: a
            polarity-rejection fraction at or above this marks the fit
            spurious regardless of convergence.  Deliberately far above the
            healthy multi-body range.
        spurious_unconverged_polarity_rejection_fraction: Combined gate: a
            fit that is BOTH unconverged AND has a polarity-rejection
            fraction at or above this is spurious.  The mis-lock signature;
            healthy frames with comparable rejection converge.
        spurious_min_coarse_peak_fraction: A coarse-NCC winning score
            (in-bounds vertex match fraction) strictly below this marks the
            fit spurious: the acquisition never had a lock.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    lm_unconverged_confidence_cap: Unit
    spurious_max_polarity_rejection_fraction: Unit
    spurious_unconverged_polarity_rejection_fraction: Unit
    spurious_min_coarse_peak_fraction: Unit


@dataclass(frozen=True)
class DTFitGateVerdict:
    """Outcome of :func:`evaluate_dt_fit_gates` for one DT fit.

    Parameters:
        spurious_reasons: Names of the hard gates that fired (empty when
            none did).  The technique ORs ``spurious`` into its own
            spurious decision and logs the reasons.
        confidence_cap: Post-sigmoid cap the technique must apply to its
            calibrated confidence (``min(confidence, cap)``), or ``None``
            when no demotion applies.
        polarity_rejection_fraction: The measured rejection fraction
            (``0.0`` for a non-polarity fit), recorded for diagnostics.
        coarse_peak_fraction: The measured coarse-NCC winning score,
            recorded for diagnostics.
        lm_converged: The LM convergence flag, recorded for diagnostics.
    """

    spurious_reasons: tuple[str, ...]
    confidence_cap: float | None
    polarity_rejection_fraction: float
    coarse_peak_fraction: float
    lm_converged: bool

    @property
    def spurious(self) -> bool:
        """True when any hard gate fired."""
        return bool(self.spurious_reasons)


def evaluate_dt_fit_gates(
    lm_result: LMRefineResult,
    gate_config: DTFitGateConfig,
    *,
    coarse_peak_fraction: float,
    total_vertex_count: int,
    use_polarity: bool,
) -> DTFitGateVerdict:
    """Evaluate the shared fit-quality gates for one DT technique fit.

    Parameters:
        lm_result: The technique's :class:`LMRefineResult`.
        gate_config: The technique's gate thresholds.
        coarse_peak_fraction: The winning coarse-NCC score
            (``CoarseSearchResult.score``).
        total_vertex_count: Number of aggregated model vertices the fit ran
            over (the polarity-fraction denominator).
        use_polarity: Whether the technique ran the polarity filter.  When
            False the polarity gates are inert and the recorded fraction is
            ``0.0`` (``RingEdgeNav`` runs polarity-free until the ring
            polarity-predictable flag is wired).

    Returns:
        :class:`DTFitGateVerdict` with the hard-gate reasons, the
        confidence cap (when the LM did not converge), and the measured
        quantities for the technique's diagnostics.
    """
    reasons: list[str] = []
    if use_polarity and total_vertex_count > 0:
        rejection_fraction = float(lm_result.polarity_rejected_count) / float(total_vertex_count)
    else:
        rejection_fraction = 0.0
    if use_polarity and rejection_fraction >= gate_config.spurious_max_polarity_rejection_fraction:
        reasons.append('polarity_rejection_fraction')
    if (
        use_polarity
        and not lm_result.converged
        and rejection_fraction >= gate_config.spurious_unconverged_polarity_rejection_fraction
    ):
        reasons.append('lm_unconverged_with_polarity_rejection')
    # Coarse-acquisition quality: the default threshold is deliberately
    # conservative plumbing; the full library-calibrated lock is #179.
    if coarse_peak_fraction < gate_config.spurious_min_coarse_peak_fraction:
        reasons.append('coarse_peak_fraction')
    confidence_cap = None if lm_result.converged else gate_config.lm_unconverged_confidence_cap
    return DTFitGateVerdict(
        spurious_reasons=tuple(reasons),
        confidence_cap=confidence_cap,
        polarity_rejection_fraction=rejection_fraction,
        coarse_peak_fraction=float(coarse_peak_fraction),
        lm_converged=bool(lm_result.converged),
    )
