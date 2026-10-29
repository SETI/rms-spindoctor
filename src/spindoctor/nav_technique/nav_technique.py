"""NavTechnique base class for the autonomous-navigation pipeline.

Concrete subclasses self-register via ``__init_subclass__`` and the
orchestrator iterates the registry.  Every concrete technique is safe to
instantiate per-image without depending on prior runs;
``__init_subclass__`` only records a class reference — no instances are
constructed at import time.
"""

from __future__ import annotations

import fnmatch
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast

import numpy as np

from spindoctor.feature.feature import NavFeature
from spindoctor.feature.feature_type import NavFeatureType
from spindoctor.nav_technique.confidence import ConfidenceBreakdown, ConfidenceSpec
from spindoctor.nav_technique.diagnostics import NavTechniqueDiagnostics
from spindoctor.nav_technique.feasibility import NavFeasibilityReport
from spindoctor.nav_technique.technique_result import NavTechniqueResult
from spindoctor.support.constants import PIXEL_CENTER_TO_CORNER_PX
from spindoctor.support.nav_base import NavBase
from spindoctor.support.types import NDArrayFloatType

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    from spindoctor.nav_orchestrator.nav_context import NavContext

__all__ = [
    'ROTATION_AT_EDGE_FRACTION',
    'ROTATION_UNOBSERVABLE_VARIANCE',
    'NCCCovarianceTuning',
    'NavTechnique',
    'add_model_error_floor',
    'add_size_scaled_model_error',
    'embed_rotation_unobservable',
    'filter_technique_names',
    'load_model_error_floor',
    'load_ncc_covariance_tuning',
    'log_confidence_breakdown',
    'reported_position_vu',
    'rotation_pivot_distance_px',
    'rotation_unobservable_sigma_rad',
    'search_window_for_obs',
    'technique_tier',
    'validate_registered_confidence_specs',
]


ROTATION_UNOBSERVABLE_VARIANCE: float = 1.0e15
"""Finite sentinel used as the rotation-axis variance when a technique
provides no rotation evidence (centroid-based fits, single-star matches,
template NCC without an outer rotation search).

The pseudoinverse used by the ensemble combine maps this huge variance to
a near-zero information contribution in the rotation direction, so the
technique contributes no rotation constraint to the precision-weighted
merge.  A literal ``+inf`` would break covariance validation
(``np.linalg.eigvalsh`` returns NaN for non-finite matrices), so this
finite sentinel is preferred.
"""


ROTATION_AT_EDGE_FRACTION: float = 0.95
"""Default fraction of ``max_rotation_deg`` at which the rotation
parameter trips the technique's ``at_edge`` flag.

When the converged ``|theta|`` exceeds
``rotation_at_edge_fraction * max_rotation_deg`` the LM (or Procrustes /
NCC pyramid) is reporting a rotation right against the configured cap,
which usually signals that the cap is too tight or the geometry is
unobservable in the rotation direction.

Each 3-DoF technique reads its threshold from
``config_510_techniques.yaml`` under
``techniques.<name>.tuning.rotation_at_edge_fraction``; this constant
is the documented default that ships in the YAML.  Re-exported so
tests and downstream tools can reference the canonical default
without re-reading the YAML.
"""


def embed_rotation_unobservable(cov_2x2: NDArrayFloatType) -> NDArrayFloatType:
    """Promote a 2x2 translation covariance to a 3x3 with rotation unobservable.

    The third diagonal carries :data:`ROTATION_UNOBSERVABLE_VARIANCE`; the
    cross-covariance entries are zero.  The ensemble's pseudo-inverse
    combine treats the rotation eigenvalue as null on input from this
    technique, so the rotation direction is ignored when fusing with
    rotation-aware techniques.

    Parameters:
        cov_2x2: 2x2 translation covariance.

    Returns:
        ``(3, 3)`` covariance matrix with the original 2x2 in the
        top-left block and the rotation-unobservable sentinel on the
        bottom-right diagonal.

    Raises:
        ValueError: if ``cov_2x2`` is not a 2x2 ndarray.
    """
    arr = np.asarray(cov_2x2, dtype=np.float64)
    if arr.ndim != 2 or arr.shape != (2, 2):
        raise ValueError(f'cov_2x2 must be a (2, 2) array; got shape {arr.shape}')
    out = np.zeros((3, 3), dtype=np.float64)
    out[:2, :2] = arr
    out[2, 2] = ROTATION_UNOBSERVABLE_VARIANCE
    return cast(NDArrayFloatType, out)


def rotation_unobservable_sigma_rad() -> float:
    """Return the rotation sigma corresponding to the unobservable sentinel.

    Convenience for techniques that report ``sigma_rotation_rad`` on
    their :class:`NavTechniqueResult` when rotation is unconstrained.

    Returns:
        ``sqrt(ROTATION_UNOBSERVABLE_VARIANCE)`` (radians, as a float).
    """
    return float(math.sqrt(ROTATION_UNOBSERVABLE_VARIANCE))


def search_window_for_obs(context: NavContext) -> tuple[int, int]:
    """Return the ``(margin_v, margin_u)`` extfov margin from the obs.

    Every translation-fit technique reads the per-instrument extfov
    margin from ``context.obs.extfov_margin_vu`` to bound its at-edge
    detection.  Centralizing the read in one helper keeps the convention
    uniform: a test obs stand-in that omits the attribute surfaces an
    ``AttributeError`` rather than silently falling through to a
    fabricated default.
    """
    margin = context.obs.extfov_margin_vu  # type: ignore[attr-defined]
    return (int(margin[0]), int(margin[1]))


def reported_position_vu(
    position_vu: tuple[float, float], margin_vu: tuple[int, int]
) -> tuple[float, float]:
    """Return a technique's working position in the form a person reads.

    A technique works in the extended frame, pixel-centric, because that is
    what it measures the padded image array in.  A position stated to a person
    is in the nominal (unpadded) frame, pixel-corner, which is the form that
    compares directly against a scene file, an image viewer, or another
    program's output.  A position inside the extended-FOV margin comes back
    negative, which is where it is.

    Parameters:
        position_vu: ``(v, u)`` in the extended frame, pixel-centric.
        margin_vu: ``(margin_v, margin_u)`` extended-FOV margin in whole
            pixels of padding.

    Returns:
        ``(v, u)`` in the nominal frame, pixel-corner.
    """
    return (
        position_vu[0] + PIXEL_CENTER_TO_CORNER_PX - margin_vu[0],
        position_vu[1] + PIXEL_CENTER_TO_CORNER_PX - margin_vu[1],
    )


def rotation_pivot_distance_px(
    pivot_vu: tuple[float, float], image_shape_vu: tuple[int, int]
) -> float:
    """Return the rotation-pivot's distance from the image center.

    Used by 3-DoF DT-fit techniques to convert the LM rotation step into a
    pixel-equivalent value for the convergence test.

    Parameters:
        pivot_vu: Pivot coordinates ``(v, u)`` in image pixels.
        image_shape_vu: Image shape ``(height, width)`` in V/U pixels.

    Returns:
        Euclidean distance from ``pivot_vu`` to the image center, floored
        at ``1.0`` so a pivot landing exactly on the image center still
        yields a non-zero convergence threshold.
    """
    center_v = float(image_shape_vu[0]) / 2.0
    center_u = float(image_shape_vu[1]) / 2.0
    dist = math.hypot(float(pivot_vu[0]) - center_v, float(pivot_vu[1]) - center_u)
    return max(dist, 1.0)


def log_confidence_breakdown(
    logger: Any, breakdown: ConfidenceBreakdown, *, low_threshold: float = 0.1
) -> None:
    """Emit a human-readable per-term explanation of a confidence value.

    Always logs the breakdown at DEBUG.  When the confidence falls below
    ``low_threshold`` the breakdown is *also* emitted at INFO so an
    operator running at the default INFO level sees why a fit reported
    near-zero confidence (typically: a single un-divided term has driven
    the sigmoid argument to a large negative value).

    Parameters:
        logger: A pdslogger compatible with ``info(...)`` / ``debug(...)``.
        breakdown: The :class:`ConfidenceBreakdown` returned by
            ``evaluate_sigmoid_combination(..., return_breakdown=True)``.
        low_threshold: Confidence at or below this value triggers the
            promotion from DEBUG to INFO.
    """
    summary_fmt = 'Confidence breakdown: alpha0=%.3f, sigmoid_arg=%.3f -> confidence=%.4f%s'
    summary_args = (
        breakdown.alpha0,
        breakdown.sigmoid_arg,
        breakdown.confidence,
        ' (hard_cap applied)' if breakdown.hard_cap_applied else '',
    )
    term_fmt = '  term %r: raw=%.4g, normalized=%.4g, alpha=%+.3f -> contribution=%+.4g'
    if breakdown.hard_zero is not None:
        logger.debug(summary_fmt, *summary_args)
        for term in breakdown.terms:
            logger.debug(
                term_fmt,
                term.feature,
                term.raw,
                term.normalized,
                term.alpha,
                term.contribution,
            )
        logger.info('Confidence forced to 0 by hard_zero_if[%r]=True', breakdown.hard_zero)
        return
    logger.debug(summary_fmt, *summary_args)
    for term in breakdown.terms:
        logger.debug(
            term_fmt,
            term.feature,
            term.raw,
            term.normalized,
            term.alpha,
            term.contribution,
        )
    if breakdown.confidence <= low_threshold:
        logger.info(summary_fmt, *summary_args)
        for term in breakdown.terms:
            logger.info(
                term_fmt,
                term.feature,
                term.raw,
                term.normalized,
                term.alpha,
                term.contribution,
            )


class NavTechnique(NavBase, ABC):
    """Abstract base for autonomous-navigation techniques.

    Concrete subclasses self-register via ``__init_subclass__``.  The
    orchestrator iterates ``NavTechnique._registry`` to discover the
    techniques it should run.  Subclasses must override the class
    attributes ``name``, ``accepts_feature_types``, and (when relevant)
    ``requires_prior``.
    """

    _registry: ClassVar[list[type[NavTechnique]]] = []
    _abstract: ClassVar[bool] = True

    #: Human-readable technique name; used as the registry key.
    name: ClassVar[str] = ''
    #: Frozen set of feature types this technique consumes.  The
    #: orchestrator skips invocation when no input feature has a
    #: matching type.
    accepts_feature_types: ClassVar[frozenset[NavFeatureType]] = frozenset()
    #: If ``True``, this technique requires a prior offset on
    #: ``NavContext`` and is run only on pass 2.
    requires_prior: ClassVar[bool] = False
    #: Technique tier consumed by the ensemble.  ``'primary'``
    #: techniques always vote; ``'fallback'`` techniques are dropped
    #: when any non-spurious primary result exists for the same source
    #: body (matched by body name extracted from ``feature_id``).  The
    #: classic example is ``BodyTerminatorNav`` (fallback) being
    #: superseded by a non-spurious ``BodyLimbNav`` or
    #: ``BodyDiscCorrelateNav`` result on the same body — limb / disc
    #: are geometric features whose failure modes are visibility-
    #: driven, while the terminator is a photometric feature that can
    #: mis-converge on textured surfaces with no per-technique signal
    #: to detect the failure.
    tier: ClassVar[Literal['primary', 'fallback']] = 'primary'
    #: If ``True``, this fallback technique still runs even when a
    #: non-spurious primary already covers its source body, so its
    #: pose-free estimate reaches the ensemble as an independent
    #: cross-check.  The ensemble still supersedes it in the fuse
    #: (``_drop_superseded_fallbacks``); only the body-witness veto reads
    #: it, to catch a geometric technique that locked onto a wrong shape.
    #: Ignored for primary-tier techniques, which always run.
    runs_as_witness: ClassVar[bool] = False
    #: Confidence-formula spec consumed by ``evaluate_sigmoid_combination``.
    #: Loaded from ``config_510_techniques.yaml`` and assigned at
    #: ``Config.read_config`` time.  ``None`` for techniques that opt out
    #: of the autonomous registry (e.g. ``NavTechniqueManual``).
    confidence_spec: ClassVar[ConfidenceSpec | None] = None
    #: Per-technique runtime tuning loaded from
    #: ``config_510_techniques.yaml.techniques.<name>.tuning``.  Each
    #: technique pulls the values it needs by name from this dict and
    #: falls back to the module-level default constant when a key is
    #: missing.  Empty for techniques that opt out of the autonomous
    #: registry or that have no tunable parameters.
    tuning: ClassVar[dict[str, float | int]] = {}
    #: Names of every attribute the technique's confidence spec may read
    #: (diagnostics fields plus side-channel flags such as ``at_edge``).
    #: ``validate_registered_confidence_specs`` ensures every term in
    #: ``confidence_spec`` references a member of this set.
    confidence_attributes: ClassVar[frozenset[str]] = frozenset()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Auto-register concrete subclasses."""
        super().__init_subclass__(**kwargs)
        if not cls.__dict__.get('_abstract', False):
            NavTechnique._registry.append(cls)

    @abstractmethod
    def is_feasible(self, features: list[NavFeature]) -> NavFeasibilityReport:
        """Return whether this technique can run on the supplied features.

        ``is_feasible`` reads feature metadata only — never pixels.  It is
        called *before* ``navigate`` and short-circuits work that has no
        chance of producing a useful result.

        Parameters:
            features: Full feature set after the reliability gate.  The
                technique filters to its accepted types internally.

        Returns:
            A ``NavFeasibilityReport`` whose ``feasible`` field tells the
            orchestrator whether to invoke ``navigate``.
        """

    @abstractmethod
    def navigate(self, features: list[NavFeature], context: NavContext) -> NavTechniqueResult:
        """Compute and return a single per-technique offset estimate.

        Parameters:
            features: Features filtered to this technique's accepted types
                (the orchestrator pre-filters; the technique need not
                re-filter unless an internal sub-check requires it).
            context: Per-image NavContext with image, masks, and shared
                derivatives.

        Returns:
            A ``NavTechniqueResult`` with offset, covariance, confidence,
            and per-technique diagnostics.
        """

    def _spurious_result(
        self,
        *,
        feature_ids: tuple[str, ...],
        diagnostics: NavTechniqueDiagnostics,
        fit_rotation: bool = False,
        source_bodies: frozenset[str] = frozenset(),
    ) -> NavTechniqueResult:
        """Return the canonical zero-confidence, spurious result.

        Encodes the shared "no usable information" shape every technique
        emits when it must hard-reject: zero offset, a finite-but-huge
        (effectively zero-information) covariance, ``confidence=0.0``,
        ``spurious=True``, ``at_edge=False``, and the rotation-unobservable
        sentinel embedded when ``fit_rotation`` is set.

        Parameters:
            feature_ids: Feature ids consumed by the rejected attempt.
            diagnostics: The technique's typed diagnostics dataclass.
            fit_rotation: When True the covariance is ``(3, 3)`` with the
                rotation-unobservable sentinel and the rotation fields are
                populated; otherwise the covariance is ``(2, 2)`` and both
                rotation fields are ``None``.
            source_bodies: Optional SPICE body names the attempt covered
                (body techniques pass these; ring / star techniques omit).

        Returns:
            A spurious :class:`NavTechniqueResult`.
        """
        cov_2x2 = 1e6 * np.eye(2, dtype=np.float64)
        cov = embed_rotation_unobservable(cov_2x2) if fit_rotation else cov_2x2
        return NavTechniqueResult(
            technique_name=self.name,
            feature_ids=feature_ids,
            offset_px=(0.0, 0.0),
            covariance_px2=cov,
            confidence=0.0,
            spurious=True,
            at_edge=False,
            diagnostics=diagnostics,
            rotation_rad=0.0 if fit_rotation else None,
            sigma_rotation_rad=(rotation_unobservable_sigma_rad() if fit_rotation else None),
            source_bodies=source_bodies,
        )


def load_model_error_floor(tuning: dict[str, Any], technique_name: str) -> float:
    """Read and validate a technique's ``model_error_floor_px`` tunable.

    The floor is added in quadrature to the technique's reported translation
    covariance (#210): robust-fit and NCC covariances measure statistical
    precision only and under-report the model error the fit cannot see
    (silhouette mismatch, photometric error, template pivot), leaving those
    results over-weighted in the ensemble.

    Parameters:
        tuning: The technique's YAML ``tuning`` mapping.
        technique_name: Technique name for the error message.

    Returns:
        The floor in pixels; ``0.0`` (disabled) when the key is absent.

    Raises:
        ValueError: If the configured value is negative or non-finite (a
            negative or NaN floor would silently disable the quadrature sum;
            an infinite floor would poison the covariance and tier gates).
    """
    raw = float(tuning.get('model_error_floor_px', 0.0))
    if not math.isfinite(raw) or raw < 0.0:
        raise ValueError(
            f'{technique_name}: model_error_floor_px must be finite and >= 0; got {raw!r}'
        )
    return raw


def add_model_error_floor(covariance: NDArrayFloatType, floor_px: float) -> NDArrayFloatType:
    """Return ``covariance`` with ``floor_px**2`` added to the translation diagonal.

    Shared by every technique that applies the #210 model-error floor, so the
    quadrature-sum convention and its rationale live in one place (see
    :func:`load_model_error_floor`).  The input is not mutated.

    Parameters:
        covariance: ``(2, 2)`` or ``(3, 3)`` covariance; only the leading two
            diagonal entries (the translation axes) are floored.
        floor_px: Validated floor in pixels; ``0.0`` returns the input
            unchanged.

    Returns:
        The floored covariance (a copy when ``floor_px > 0``).
    """
    if floor_px <= 0.0:
        return covariance
    out = covariance.copy()
    out[0, 0] += floor_px**2
    out[1, 1] += floor_px**2
    return out


@dataclass(frozen=True)
class NCCCovarianceTuning:
    """Model-error terms folded into an NCC technique's translation covariance.

    The bare peak-curvature (matched-filter) covariance measures only the
    statistical precision at the correlation peak.  On a well-matched
    template that precision is far smaller than the true registration error,
    which is dominated by silhouette / photometric model error and by
    localization ambiguity.  Three quadrature terms restore calibration:

    - ``localization_uncertainty_scale`` multiplies the inter-pyramid-level
      peak migration (applied inside the correlator, carried here for
      completeness of the covariance model).
    - ``model_error_size_frac`` scales a body / template size in pixels into
      a size-proportional silhouette-error sigma (a coherent shape mismatch
      displaces the peak by a roughly fixed *fraction* of the extent).
    - ``model_error_floor_px`` is a size-independent absolute floor
      (pointing, camera distortion, sub-pixel interpolation).

    Parameters:
        localization_uncertainty_scale: Multiplier on the correlator
            ``consistency`` (px); ``0.0`` disables it.
        model_error_size_frac: Fraction of the template extent (px) taken as
            the silhouette-error sigma; ``0.0`` disables it.
        model_error_floor_px: Absolute floor (px); ``0.0`` disables it.
    """

    localization_uncertainty_scale: float
    model_error_size_frac: float
    model_error_floor_px: float


def load_ncc_covariance_tuning(tuning: dict[str, Any], technique_name: str) -> NCCCovarianceTuning:
    """Read and validate an NCC technique's covariance model-error tunables.

    Parameters:
        tuning: The technique's YAML ``tuning`` mapping.
        technique_name: Technique name for error messages.

    Returns:
        The validated :class:`NCCCovarianceTuning`.

    Raises:
        ValueError: If any term is negative or non-finite.
    """
    floor_px = load_model_error_floor(tuning, technique_name)
    scale = float(tuning.get('localization_uncertainty_scale', 0.0))
    size_frac = float(tuning.get('model_error_size_frac', 0.0))
    for name, value in (
        ('localization_uncertainty_scale', scale),
        ('model_error_size_frac', size_frac),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f'{technique_name}: {name} must be finite and >= 0; got {value!r}')
    return NCCCovarianceTuning(
        localization_uncertainty_scale=scale,
        model_error_size_frac=size_frac,
        model_error_floor_px=floor_px,
    )


def add_size_scaled_model_error(
    covariance: NDArrayFloatType,
    *,
    size_px: float,
    size_frac: float,
    floor_px: float,
) -> NDArrayFloatType:
    """Add a size-proportional silhouette error and an absolute floor in quadrature.

    Adds ``(size_frac * size_px)**2 + floor_px**2`` to the leading two
    (translation) diagonal entries.  The size term makes the reported sigma
    track body / template extent -- a large body's silhouette mismatch
    displaces the correlation peak by more pixels than a small body's -- so a
    single constant floor no longer has to cover the whole size range.  The
    input is not mutated; the localization-spread term is applied separately
    inside the correlator.

    Parameters:
        covariance: ``(2, 2)`` or ``(3, 3)`` covariance; only the translation
            diagonal is modified.
        size_px: Template extent in pixels (body diameter, ring radial span).
        size_frac: Silhouette-error fraction of ``size_px``.
        floor_px: Absolute floor in pixels.

    Returns:
        The inflated covariance (a copy when any term is positive).
    """
    extra = (size_frac * size_px) ** 2 + floor_px**2
    if extra <= 0.0:
        return covariance
    out = covariance.copy()
    out[0, 0] += extra
    out[1, 1] += extra
    return out


def technique_tier(technique_name: str) -> str:
    """Look up a registered technique's ``tier`` from the registry.

    Returns ``'primary'`` for an unregistered name so unknown techniques (test
    stubs, future plug-ins) are treated as primary by construction; an explicit
    ``tier='fallback'`` declaration is required to opt into the fallback-drop
    behavior.  Shared by the orchestrator's fallback pre-filter and the
    ensemble's fallback-supersession pass so the tier rule lives in one place.
    """
    for cls in NavTechnique._registry:
        if cls.name == technique_name:
            return cls.tier
    return 'primary'


def validate_registered_confidence_specs() -> None:
    """Validate every registered ``NavTechnique``'s confidence spec.

    Each technique whose spec was loaded from
    ``config_510_techniques.yaml`` (assigned by
    :meth:`spindoctor.config.config.Config._validate_registered_techniques`)
    must declare the full set of valid attribute names in
    ``confidence_attributes``.  Every term's ``feature`` and every
    ``hard_zero_if`` key must appear in that set; otherwise the
    technique would raise at navigate time.  Validation runs at
    config-load time so the failure surfaces during process startup
    rather than mid-image.  Techniques whose spec is ``None`` (test-
    only registry entries opted out of YAML lookup) are skipped.

    Raises:
        ValueError: if any term references an unknown attribute.  The
            message names both the technique class and the bad
            attribute.
    """
    for cls in NavTechnique._registry:
        spec = cls.confidence_spec
        if spec is None:
            continue
        valid = cls.confidence_attributes
        for term in spec.terms:
            if term.feature not in valid:
                raise ValueError(
                    f'NavTechnique {cls.name!r}: confidence term references unknown '
                    f'attribute {term.feature!r}; declared attributes are '
                    f'{sorted(valid)!r}'
                )
        for key in spec.hard_zero_if:
            if key not in valid:
                raise ValueError(
                    f'NavTechnique {cls.name!r}: hard_zero_if references unknown '
                    f'attribute {key!r}; declared attributes are {sorted(valid)!r}'
                )


def filter_technique_names(names: list[str], patterns: str | list[str]) -> list[str]:
    """Return ``names`` filtered by gitignore-style glob patterns.

    A leading ``!`` marks an exclusion pattern; otherwise the pattern is
    inclusion.  An exclusion-only pattern list implies an inclusion default
    of ``'*'``.

    Parameters:
        names: List of technique names (or any candidate strings).
        patterns: Single pattern or list of patterns.  Patterns matching
            shell-glob syntax (``*``, ``?``, ``[abc]``).  Leading ``!``
            means exclusion.

    Returns:
        Names that match at least one inclusion pattern and no exclusion
        patterns.

    Raises:
        ValueError: if an empty pattern list is supplied.
    """
    if isinstance(patterns, str):
        patterns = [patterns]
    if not patterns:
        raise ValueError('patterns must contain at least one entry')
    includes: list[str] = []
    excludes: list[str] = []
    for pat in patterns:
        if pat.startswith('!'):
            excludes.append(pat[1:])
        else:
            includes.append(pat)
    if not includes:
        # Pure-exclusion pattern list: implicit '*' include.
        includes = ['*']
    out: list[str] = []
    for name in names:
        if not any(fnmatch.fnmatch(name, p) for p in includes):
            continue
        if any(fnmatch.fnmatch(name, p) for p in excludes):
            continue
        out.append(name)
    return out
