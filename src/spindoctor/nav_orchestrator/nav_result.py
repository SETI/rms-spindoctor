"""NavResult — full in-memory output of a single navigation.

Carries the headline (offset ± uncertainty + simple rank) plus full
diagnostic information about every technique that ran, every feature that
was extracted, and provenance.  Not intended to be JSON-serialized
directly; the curator builds a curated JSON-friendly subset.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from spindoctor.annotation import Annotations
from spindoctor.nav_orchestrator.feature_summary import NavFeatureSummary
from spindoctor.nav_orchestrator.image_classifier_result import NavImageClassifierResult
from spindoctor.nav_orchestrator.provenance import Provenance
from spindoctor.nav_technique.technique_result import NavTechniqueResult
from spindoctor.support.cmatrix import PointingSolution
from spindoctor.support.status_reason import NavStatusReason
from spindoctor.support.types import NDArrayFloatType

__all__ = ['NavInternalErrorRecord', 'NavResult']


Status = Literal['success', 'failed', 'conflicted']
"""Top-level status set on every NavResult."""

ConfidenceRank = Literal['high', 'medium', 'low', 'conflicted', 'failed']
"""Five-bucket confidence rank presented to downstream consumers."""


@dataclass(frozen=True)
class NavInternalErrorRecord:
    """Which plugin raised, and what class of exception it raised.

    Set on a result whose ``status_reason`` is
    ``NavStatusReason.INTERNAL_ERROR``, and on no other, so that a document
    saying the navigation hit an internal error always says which component
    hit it.  The two travel together as one object rather than as two
    optional fields for the same reason: two encodings of one fact that
    nothing holds in step eventually disagree.

    Parameters:
        component: What raised, as ``<registered name>.<method>`` -- for a
            model the instance name it was built under, as in
            ``rings:SATURN.create_model``, and for a technique its
            class-level name, as in ``RingEdgeNav.navigate``.
        exception_type: The class name of the exception raised, for example
            ``AttributeError``.  The message and traceback are deliberately
            not carried here: they are unbounded free text, and a field of
            the metadata document is not the place for it.  The per-image
            error log has both.

    Raises:
        TypeError: If either field is not a ``str``.
        ValueError: If either field is empty or only whitespace.  A record
            naming nothing is worse than no record: it says a component
            failed and declines to say which, and every consumer of the
            document then has a field it must special-case.
    """

    component: str
    exception_type: str

    def __post_init__(self) -> None:
        """Refuse a record that names no component or no exception.

        Validated at runtime rather than left to the type checker because
        this is a published package: mypy runs over this repository's call
        sites and not over a third-party caller's.  The same pair of checks,
        in the same order, guards ``ConfidenceTerm``.
        """
        for name in ('component', 'exception_type'):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise TypeError(
                    f'NavInternalErrorRecord.{name} must be str; got {type(value).__name__}'
                )
            if not value.strip():
                raise ValueError(f'NavInternalErrorRecord.{name} must be a non-empty string')


@dataclass(frozen=True, eq=False)
class NavResult:
    """Full in-memory navigation output for one image.

    Constructors ``NavResult.success``, ``NavResult.failed``, and
    ``NavResult.conflicted`` are the canonical entry points; direct
    instantiation is also supported.

    Every number the result reports -- the offset, its per-axis sigma, and
    the fitted rotation with its sigma -- is finite.  Each is this class's
    own arithmetic over per-technique results whose own offset and rotation
    are already required to be finite, so a non-finite one is a defect in
    the combine rather than a value to report, and construction refuses it.
    ``sigma_along_unobservable_px`` is the exception: infinity there is how
    an unobservable translation axis is reported.

    Parameters:
        status: One of ``'success'``, ``'failed'``, ``'conflicted'``.
        offset_px: ``(dv, du)`` offset; ``None`` on failure.
        sigma_px: Per-axis 1sigma marginal uncertainty; ``None`` on failure.
        sigma_along_unobservable_px: Set when covariance is rank-1
            (e.g. flat-ring-only scenes); ``None`` for full-rank results.
        confidence_rank: Five-bucket rank derived from confidence + status.
        confidence: Underlying calibrated confidence score in ``[0, 1]``.
        status_reason: NavStatusReason value explaining the outcome.
        covariance_px2: Full 2x2 covariance (or 3x3 with rotation);
            ``None`` on failure.
        per_technique: List of every technique's result (whether kept or
            dropped by the ensemble).
        excluded_from_consensus: Technique names of viable results the
            ensemble left out of the reported combine -- outliers rejected
            against a multi-technique consensus, or (on a conflicted
            result) the runner-up alternative.  Empty when every viable
            result contributed.
        consensus_techniques: Technique names of the results the ensemble
            actually combined into the reported offset (the winning
            consensus subset), in input order.  Empty on failure.  The
            orchestrator uses this to stamp pass-2 results with the
            techniques that seeded their prior.
        feature_inventory: Per-feature summary entries — what was
            extracted, what survived the gate, and why.
        image_classifier: The image-quality classifier's verdict.
        model_metadata: Per-NavModel diagnostic dicts keyed by model name.
        annotations: Composite annotation collection assembled from every
            registered NavModel's ``to_annotations`` plus orchestrator
            additions.  Empty by default; intended for the summary-PNG
            renderer.
        provenance: Reproducibility envelope.
        rotation_rad: Optional fitted camera rotation (radians); ``None``
            when ``fit_camera_rotation`` is False.
        sigma_rotation_rad: Optional 1-sigma rotation uncertainty.
        pointing: Optional corrected-attitude solution: the uncorrected and
            corrected C-matrices, the SPICE frame identities, and the
            exposure times.  Stamped by the orchestrator once the
            observation is in hand; ``None`` for a host whose SPICE frames
            are unknown, or when the attitude could not be computed.
        internal_error: Which NavModel or NavTechnique raised, and what it
            raised.  Set when and only when ``status_reason`` is
            ``NavStatusReason.INTERNAL_ERROR``.
    """

    status: Status
    offset_px: tuple[float, float] | None
    sigma_px: tuple[float, float] | None
    sigma_along_unobservable_px: float | None
    confidence_rank: ConfidenceRank
    confidence: float
    status_reason: NavStatusReason
    covariance_px2: NDArrayFloatType | None
    per_technique: list[NavTechniqueResult]
    feature_inventory: list[NavFeatureSummary]
    image_classifier: NavImageClassifierResult
    provenance: Provenance
    excluded_from_consensus: list[str] = field(default_factory=list)
    consensus_techniques: list[str] = field(default_factory=list)
    model_metadata: dict[str, dict[str, Any]] = field(default_factory=dict)
    annotations: Annotations = field(default_factory=Annotations)
    rotation_rad: float | None = None
    sigma_rotation_rad: float | None = None
    pointing: PointingSolution | None = None
    internal_error: NavInternalErrorRecord | None = None

    def __post_init__(self) -> None:
        """Validate status against offset and reason, and the solution's finiteness."""
        if self.status == 'failed' and self.offset_px is not None:
            raise ValueError('status=failed must have offset_px=None')
        if self.status == 'success' and self.offset_px is None:
            raise ValueError('status=success must have a non-None offset_px')
        # The offset is the one number that reaches the metadata document
        # unrounded, so nothing downstream maps it onto the finite sentinel a
        # curated float gets; and the design keeps the per-axis sigmas finite
        # on purpose, reporting an unobservable axis through
        # ``sigma_along_unobservable_px`` instead of an inflated sigma.  So a
        # non-finite value in any of these is this code's arithmetic gone
        # wrong, and is refused here as it already is per technique.
        if self.offset_px is not None and not all(math.isfinite(v) for v in self.offset_px):
            raise ValueError(f'offset_px must be finite; got {self.offset_px!r}')
        if self.sigma_px is not None and not all(math.isfinite(v) for v in self.sigma_px):
            raise ValueError(f'sigma_px must be finite; got {self.sigma_px!r}')
        if self.rotation_rad is not None and not math.isfinite(self.rotation_rad):
            raise ValueError(f'rotation_rad must be finite when set; got {self.rotation_rad!r}')
        if self.sigma_rotation_rad is not None and not math.isfinite(self.sigma_rotation_rad):
            raise ValueError(
                f'sigma_rotation_rad must be finite when set; got {self.sigma_rotation_rad!r}'
            )
        # The status reason and the record are two statements of one fact,
        # so they are held in step here rather than left to each construction
        # site to remember: a document reporting an internal error always
        # says which component hit it, and no other document carries a
        # component that never raised.
        if (self.status_reason is NavStatusReason.INTERNAL_ERROR) != (
            self.internal_error is not None
        ):
            raise ValueError(
                'status_reason=internal_error and internal_error must be set together; got '
                f'status_reason={self.status_reason!r}, internal_error={self.internal_error!r}'
            )
        # An internal error is always fatal to the image, so the status says
        # so too.  Direct instantiation is supported, and without this a
        # hand-built result could report success while carrying the block
        # that says navigation did not complete -- the curator would emit
        # both, and a consumer would have to decide which half to believe.
        if self.status_reason is NavStatusReason.INTERNAL_ERROR and self.status != 'failed':
            raise ValueError(
                f'status_reason=internal_error requires status=failed; got status={self.status!r}'
            )
        if self.confidence_rank == 'failed' and self.status != 'failed':
            raise ValueError('confidence_rank=failed requires status=failed')
        if (self.confidence_rank == 'conflicted') != (self.status == 'conflicted'):
            raise ValueError(
                "confidence_rank 'conflicted' and status 'conflicted' must agree; got "
                f'confidence_rank={self.confidence_rank!r}, status={self.status!r}'
            )
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f'confidence must lie in [0, 1]; got {self.confidence!r}')
        if self.covariance_px2 is not None:
            cov = np.asarray(self.covariance_px2, np.float64)
            if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
                raise ValueError(f'covariance_px2 must be square 2-D; got shape {cov.shape}')
            cov.setflags(write=False)
            object.__setattr__(self, 'covariance_px2', cov)

    @classmethod
    def failed(
        cls,
        *,
        status_reason: NavStatusReason,
        image_classifier: NavImageClassifierResult,
        provenance: Provenance,
        per_technique: list[NavTechniqueResult] | None = None,
        feature_inventory: list[NavFeatureSummary] | None = None,
        model_metadata: dict[str, dict[str, Any]] | None = None,
        annotations: Annotations | None = None,
        internal_error: NavInternalErrorRecord | None = None,
    ) -> 'NavResult':
        """Construct a NavResult for a failed navigation.

        Parameters:
            status_reason: Discrete reason from ``NavStatusReason``.
            image_classifier: Image-quality classifier verdict.
            provenance: Reproducibility envelope.
            per_technique: Optional list of technique results (e.g. when
                every technique returned spurious).
            feature_inventory: Optional list of feature summaries.
            model_metadata: Optional model metadata dict.
            annotations: Optional pre-built annotation collection
                (typically empty on failure).
            internal_error: Which component raised and what it raised.
                Required when ``status_reason`` is
                ``NavStatusReason.INTERNAL_ERROR``, and refused otherwise.

        Returns:
            NavResult with ``status='failed'``, ``confidence=0.0``,
            ``confidence_rank='failed'``, and no offset.
        """
        return cls(
            status='failed',
            offset_px=None,
            sigma_px=None,
            sigma_along_unobservable_px=None,
            confidence_rank='failed',
            confidence=0.0,
            status_reason=status_reason,
            covariance_px2=None,
            per_technique=per_technique or [],
            feature_inventory=feature_inventory or [],
            image_classifier=image_classifier,
            provenance=provenance,
            model_metadata=model_metadata or {},
            annotations=annotations if annotations is not None else Annotations(),
            internal_error=internal_error,
        )

    @classmethod
    def success(
        cls,
        *,
        offset_px: tuple[float, float],
        covariance_px2: NDArrayFloatType,
        confidence: float,
        confidence_rank: ConfidenceRank,
        status_reason: NavStatusReason,
        per_technique: list[NavTechniqueResult],
        feature_inventory: list[NavFeatureSummary],
        image_classifier: NavImageClassifierResult,
        provenance: Provenance,
        sigma_along_unobservable_px: float | None = None,
        excluded_from_consensus: list[str] | None = None,
        consensus_techniques: list[str] | None = None,
        model_metadata: dict[str, dict[str, Any]] | None = None,
        annotations: Annotations | None = None,
        rotation_rad: float | None = None,
        sigma_rotation_rad: float | None = None,
    ) -> 'NavResult':
        """Construct a NavResult for a successful navigation.

        Parameters: see dataclass field docs above.  ``sigma_px`` is
        derived from the diagonal of ``covariance_px2``.
        """
        cov = np.asarray(covariance_px2, np.float64)
        sigma_dv = float(np.sqrt(max(cov[0, 0], 0.0)))
        sigma_du = float(np.sqrt(max(cov[1, 1], 0.0)))
        return cls(
            status='success',
            offset_px=offset_px,
            sigma_px=(sigma_dv, sigma_du),
            sigma_along_unobservable_px=sigma_along_unobservable_px,
            confidence_rank=confidence_rank,
            confidence=confidence,
            status_reason=status_reason,
            covariance_px2=cov,
            per_technique=per_technique,
            feature_inventory=feature_inventory,
            image_classifier=image_classifier,
            provenance=provenance,
            excluded_from_consensus=excluded_from_consensus or [],
            consensus_techniques=consensus_techniques or [],
            model_metadata=model_metadata or {},
            annotations=annotations if annotations is not None else Annotations(),
            rotation_rad=rotation_rad,
            sigma_rotation_rad=sigma_rotation_rad,
        )

    @classmethod
    def conflicted(
        cls,
        *,
        offset_px: tuple[float, float],
        covariance_px2: NDArrayFloatType,
        confidence: float,
        per_technique: list[NavTechniqueResult],
        feature_inventory: list[NavFeatureSummary],
        image_classifier: NavImageClassifierResult,
        provenance: Provenance,
        excluded_from_consensus: list[str] | None = None,
        consensus_techniques: list[str] | None = None,
        status_reason: NavStatusReason = NavStatusReason.CONFLICTED_TECHNIQUES,
        model_metadata: dict[str, dict[str, Any]] | None = None,
        annotations: Annotations | None = None,
    ) -> 'NavResult':
        """Construct a NavResult for a conflicted (best-group reported) navigation.

        ``confidence_rank`` is hard-set to ``'conflicted'``; downstream
        consumers refuse to use these results without explicit opt-in.
        ``status_reason`` defaults to the summed-confidence-gap conflict; a
        cross-technique veto that reports its best group conflicted (for
        example a suspected body shape lock) passes its own reason.
        """
        cov = np.asarray(covariance_px2, np.float64)
        sigma_dv = float(np.sqrt(max(cov[0, 0], 0.0)))
        sigma_du = float(np.sqrt(max(cov[1, 1], 0.0)))
        return cls(
            status='conflicted',
            offset_px=offset_px,
            sigma_px=(sigma_dv, sigma_du),
            sigma_along_unobservable_px=None,
            confidence_rank='conflicted',
            confidence=confidence,
            status_reason=status_reason,
            covariance_px2=cov,
            per_technique=per_technique,
            feature_inventory=feature_inventory,
            image_classifier=image_classifier,
            provenance=provenance,
            excluded_from_consensus=excluded_from_consensus or [],
            consensus_techniques=consensus_techniques or [],
            model_metadata=model_metadata or {},
            annotations=annotations if annotations is not None else Annotations(),
        )
