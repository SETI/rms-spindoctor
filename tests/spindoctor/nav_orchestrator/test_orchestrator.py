"""Integration tests for ``spindoctor.nav_orchestrator.orchestrator.NavOrchestrator``."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from spindoctor.annotation import Annotations
from spindoctor.config import LogLevels, set_log_levels
from spindoctor.feature.feature import (
    NavFeature,
    NavReliabilityBreakdown,
    body_names_from_features,
)
from spindoctor.feature.feature_type import NavFeatureType
from spindoctor.feature.flags import BodyBlobFlags, BodyDiscFlags, StarFlags, TitanHazeFlags
from spindoctor.feature.geometry import (
    BodyBlobGeometry,
    BodyDiscGeometry,
    StarGeometry,
    TitanHazeGeometry,
)
from spindoctor.nav_model import NavModel
from spindoctor.nav_orchestrator.image_classifier import ImageQualityThresholds
from spindoctor.nav_orchestrator.nav_context import NavContext
from spindoctor.nav_orchestrator.orchestrator import NavOrchestrator
from spindoctor.nav_technique.diagnostics import StarFieldDiagnostics
from spindoctor.nav_technique.feasibility import NavFeasibilityReport
from spindoctor.nav_technique.nav_technique import NavTechnique
from spindoctor.nav_technique.technique_result import NavTechniqueResult
from spindoctor.support.cmatrix import PointingSolution
from spindoctor.support.exceptions import NavContractError, NavPointingError
from spindoctor.support.filters import NavFilterKind, NavFilterSpec
from spindoctor.support.status_reason import NavStatusReason


class _FakeObs:
    """Minimal object satisfying the orchestrator's obs.* attribute access."""

    def __init__(
        self,
        *,
        image: np.ndarray | None = None,
        sensor_mask: np.ndarray | None = None,
        midtime: float = 0.0,
        extfov_margin: tuple[int, int] = (0, 0),
        inst_config: dict[str, Any] | None = None,
    ) -> None:
        self.inst_config = inst_config
        if image is None:
            rng = np.random.default_rng(seed=42)
            image = rng.standard_normal(size=(64, 64)) + 100.0
        self.data = image
        # Build an extfov-padded image around ``data`` so the fake matches
        # the real obs API: ``extdata`` is the canonical input the
        # orchestrator reads.
        margin_v, margin_u = extfov_margin
        self.extfov_margin_vu = (margin_v, margin_u)
        ext_shape = (image.shape[0] + 2 * margin_v, image.shape[1] + 2 * margin_u)
        ext = np.zeros(ext_shape, dtype=image.dtype)
        ext[margin_v : margin_v + image.shape[0], margin_u : margin_u + image.shape[1]] = image
        self.extdata = ext
        if sensor_mask is None:
            sensor_mask = np.zeros(ext_shape, bool)
            sensor_mask[
                margin_v : margin_v + image.shape[0],
                margin_u : margin_u + image.shape[1],
            ] = True
        self._sensor_mask = sensor_mask
        self.midtime = midtime

    def extfov_data_sensor_mask(self) -> np.ndarray:
        return self._sensor_mask


class _FakeStarModel(NavModel):
    """Fake NavModel that emits one STAR feature per ``feature_count``."""

    def __init__(self, obs: Any, *, feature_count: int = 1) -> None:
        super().__init__('stars', obs)
        self._feature_count = feature_count

    def create_model(self) -> None:
        self._metadata['feature_count'] = self._feature_count

    def to_features(self, context: NavContext) -> list[NavFeature]:
        features: list[NavFeature] = []
        for i in range(self._feature_count):
            features.append(
                NavFeature(
                    feature_id=f'star:test:{i}',
                    feature_type=NavFeatureType.STAR,
                    source_model='stars',
                    geometry=StarGeometry(
                        predicted_vu=(float(10 + i), float(20 + i)),
                        catalog_vu=(float(10 + i), float(20 + i)),
                        bbox_extfov_vu=(0, 0, 16, 16),
                    ),
                    subject_range_km=1.0e10,
                    position_cov_px=np.eye(2, dtype=np.float64) * 0.25,
                    intensity_sigma_rel=0.05,
                    preferred_filter=NavFilterSpec(kind=NavFilterKind.NONE),
                    reliability=0.8,
                    reliability_reasons=NavReliabilityBreakdown(predicted_snr=10.0),
                    usable_types=frozenset({NavFeatureType.STAR}),
                    flags=StarFlags(),
                )
            )
        return features

    def to_annotations(self, context: NavContext) -> Annotations:
        return Annotations()


class _FakeTitanModel(NavModel):
    """Fake Titan model emitting one TITAN_LIMB feature at a chosen reliability.

    Frame quality on a hazy body is carried by the feature's reliability, not
    by a decline, so the fake exposes the reliability as a knob: ``0.0``
    reproduces a hard-zero frame (unframeable, occluded, or too small) and a
    value above the type gate reproduces a navigable one.
    """

    _abstract = True

    def __init__(self, obs: Any, *, reliability: float = 0.0) -> None:
        super().__init__('titan:TITAN', obs)
        self._reliability = reliability

    def create_model(self) -> None:
        self._metadata['body'] = 'TITAN'

    def to_features(self, context: NavContext) -> list[NavFeature]:
        return [
            NavFeature(
                feature_id='titan_limb:TITAN',
                feature_type=NavFeatureType.TITAN_LIMB,
                source_model='titan:TITAN',
                geometry=TitanHazeGeometry(
                    predicted_center_vu=(32.0, 32.0),
                    sun_angle_rad=0.0,
                    axis_degenerate=False,
                    phase_deg=30.0,
                    r_solid_px=12.0,
                    r_env_px=14.0,
                    km_per_px=25.0,
                    contaminant_mask=None,
                    filters=('CL1', 'CL2'),
                    bbox_extfov_vu=(18, 18, 47, 47),
                ),
                subject_range_km=1.2e6,
                position_cov_px=None,
                intensity_sigma_rel=0.0,
                preferred_filter=NavFilterSpec(kind=NavFilterKind.NONE),
                reliability=self._reliability,
                reliability_reasons=NavReliabilityBreakdown(
                    titan_envelope_diameter_px=28.0,
                    titan_occluded_fraction=0.0,
                ),
                usable_types=frozenset({NavFeatureType.TITAN_LIMB}),
                flags=TitanHazeFlags(body_name='TITAN'),
            )
        ]

    def to_annotations(self, context: NavContext) -> Annotations:
        return Annotations()


class _FakeCoveringBodyModel(NavModel):
    """A body model that declined because its body covers the extended frame.

    It emits nothing, which is what a declined body model does, and records
    the decline in its metadata, which is where the orchestrator reads it.
    """

    _abstract = True

    def __init__(self, obs: Any) -> None:
        super().__init__('body:SATURN', obs)

    def create_model(self) -> None:
        self._metadata['fills_extfov'] = True

    def to_features(self, context: NavContext) -> list[NavFeature]:
        return []

    def to_annotations(self, context: NavContext) -> Annotations:
        return Annotations()


class _FakeStarTechnique(NavTechnique):
    """Stand-in technique that always reports a fixed offset.

    ``_abstract = True`` keeps this (and every fake below) out of the
    process-wide ``NavTechnique._registry`` at import; the autouse
    ``_register_fakes`` fixture registers them for the duration of each
    test in this module.  A permanently registered fake would otherwise
    run inside every later full-ensemble navigation in the same worker
    (e.g. the sim invariant tests) and fuse its hardcoded offset into
    real results.
    """

    name = '_FakeStarTechnique'
    _abstract = True
    accepts_feature_types = frozenset({NavFeatureType.STAR})

    def is_feasible(self, features: list[NavFeature]) -> NavFeasibilityReport:
        if len(features) >= 1:
            return NavFeasibilityReport(
                feasible=True, reason='ok', consumed_feature_count=len(features)
            )
        return NavFeasibilityReport(feasible=False, reason='no_stars')

    def navigate(self, features: list[NavFeature], context: NavContext) -> NavTechniqueResult:
        return NavTechniqueResult(
            technique_name=self.name,
            feature_ids=tuple(f.feature_id for f in features),
            offset_px=(1.5, 2.5),
            covariance_px2=np.eye(2, dtype=np.float64) * 0.25,
            confidence=0.85,
            spurious=False,
            at_edge=False,
            diagnostics=StarFieldDiagnostics(n_inliers=len(features)),
        )


@pytest.fixture
def fake_obs() -> _FakeObs:
    """Provide a clean fake observation."""
    return _FakeObs()


@pytest.fixture(autouse=True)
def _fakes_report_as_simulated(fakes_report_as_simulated: None) -> None:
    """Apply the shared simulated-instrument report to every test in this module."""


@pytest.fixture(autouse=True)
def _register_fakes() -> Iterator[None]:
    """Register this module's fake techniques for the duration of one test.

    The fakes carry ``_abstract = True`` so they never enter the global
    ``NavTechnique._registry`` at import; this fixture makes them
    name-resolvable (``only_techniques=['_FakeStarTechnique']``) inside
    this module's tests only, and guarantees no other test in the same
    process ever sees them.

    ``_RaisingTechnique`` is deliberately not among them and is registered by
    its own fixture instead.  A technique that raises now fails the image it
    raised on, so a module-wide registration would fail every test in this
    file whose scene offers a STAR feature -- and would fail them with a
    status reason describing the fake rather than whatever each test is
    about.
    """
    fakes: list[type[NavTechnique]] = [
        _FakeStarTechnique,
        _PassTwoTechnique,
        _InfeasibleTechnique,
        _FakeBodyPrimary,
        _FakeBodyFallback,
    ]
    NavTechnique._registry.extend(fakes)
    try:
        yield
    finally:
        for fake in fakes:
            NavTechnique._registry.remove(fake)


def test_orchestrator_runs_pipeline_end_to_end(fake_obs: _FakeObs) -> None:
    """A clean image + 1 star model + 1 technique -> status='success'."""
    obs = fake_obs
    model = _FakeStarModel(obs, feature_count=3)
    orch = NavOrchestrator(
        [model],
        only_techniques=['_FakeStarTechnique'],
    )
    result = orch.navigate(obs)  # type: ignore[arg-type]
    assert result.status == 'success'
    assert result.offset_px == (1.5, 2.5)
    assert [t.technique_name for t in result.per_technique] == ['_FakeStarTechnique']
    assert result.confidence_rank == 'high'


def test_orchestrator_handles_nonzero_extfov_margin() -> None:
    """The pipeline accepts obs whose ``extdata`` is larger than ``data``.

    The image-quality classifier consumes ``obs.extdata`` and
    ``obs.extfov_data_sensor_mask()`` together, which must share the
    extended-FOV shape regardless of the per-instrument extfov margin.
    """
    obs = _FakeObs(
        image=np.full((64, 64), 100.0, dtype=np.float64),
        extfov_margin=(8, 12),
    )
    assert obs.extdata.shape == (80, 88)
    assert obs.extfov_data_sensor_mask().shape == (80, 88)
    model = _FakeStarModel(obs, feature_count=3)
    orch = NavOrchestrator([model], only_techniques=['_FakeStarTechnique'])
    result = orch.navigate(obs)  # type: ignore[arg-type]
    assert result.status == 'success'


def test_orchestrator_blank_image_short_circuits(fake_obs: _FakeObs) -> None:
    """A blank image yields status='failed' before any technique runs."""
    obs = _FakeObs(image=np.zeros((64, 64), np.float64))
    model = _FakeStarModel(obs, feature_count=3)
    orch = NavOrchestrator([model])
    result = orch.navigate(obs)  # type: ignore[arg-type]
    assert result.status == 'failed'
    assert result.status_reason == NavStatusReason.NO_SIGNAL_IN_IMAGE
    assert result.per_technique == []


def test_orchestrator_no_features_emitted_yields_no_features_extracted(
    fake_obs: _FakeObs,
) -> None:
    """A NavModel emitting zero features yields no_features_extracted."""
    obs = fake_obs
    model = _FakeStarModel(obs, feature_count=0)
    orch = NavOrchestrator([model])
    result = orch.navigate(obs)  # type: ignore[arg-type]
    assert result.status == 'failed'
    assert result.status_reason == NavStatusReason.NO_FEATURES_EXTRACTED


def test_orchestrator_titan_hard_zero_yields_all_features_gated(
    fake_obs: _FakeObs,
) -> None:
    """A Titan-only frame at hard-zero reliability ends all_features_gated."""
    obs = fake_obs
    model = _FakeTitanModel(obs, reliability=0.0)
    orch = NavOrchestrator([model])
    result = orch.navigate(obs)  # type: ignore[arg-type]
    assert result.status == 'failed'
    assert result.status_reason == NavStatusReason.ALL_FEATURES_GATED


def test_orchestrator_titan_hard_zero_records_gated_feature(
    fake_obs: _FakeObs,
) -> None:
    """The gated Titan frame carries an attributing TITAN_LIMB inventory record."""
    obs = fake_obs
    model = _FakeTitanModel(obs, reliability=0.0)
    orch = NavOrchestrator([model])
    result = orch.navigate(obs)  # type: ignore[arg-type]
    gated = [
        entry
        for entry in result.feature_inventory
        if entry.feature_type is NavFeatureType.TITAN_LIMB and entry.gated
    ]
    assert len(gated) == 1


def test_orchestrator_titan_hard_zero_gate_record_carries_breakdown(
    fake_obs: _FakeObs,
) -> None:
    """The gate record names the envelope diameter that produced the score."""
    obs = fake_obs
    model = _FakeTitanModel(obs, reliability=0.0)
    orch = NavOrchestrator([model])
    result = orch.navigate(obs)  # type: ignore[arg-type]
    entry = next(e for e in result.feature_inventory if e.feature_type is NavFeatureType.TITAN_LIMB)
    assert entry.reliability_reasons.titan_envelope_diameter_px == 28.0


def test_orchestrator_titan_plus_stars_navigates_normally(
    fake_obs: _FakeObs,
) -> None:
    """A gated Titan alongside navigable stars still navigates on the stars."""
    obs = fake_obs
    models = [_FakeStarModel(obs, feature_count=3), _FakeTitanModel(obs, reliability=0.0)]
    orch = NavOrchestrator(models, only_techniques=['_FakeStarTechnique'])
    result = orch.navigate(obs)  # type: ignore[arg-type]
    assert result.status == 'success'
    assert result.status_reason == NavStatusReason.OK


def test_orchestrator_titan_usable_but_unfittable_yields_all_techniques_spurious() -> None:
    """A usable Titan feature the real fit rejects ends all_techniques_spurious.

    The image is pure noise, so every symmetry and arc gate fails; the
    technique is expected to report a named gate rather than an offset.
    """
    obs = _FakeObs(extfov_margin=(8, 8))
    model = _FakeTitanModel(obs, reliability=0.9)
    orch = NavOrchestrator([model], only_techniques=['TitanHazeNav'])
    result = orch.navigate(obs)  # type: ignore[arg-type]
    assert result.status == 'failed'
    assert result.status_reason == NavStatusReason.ALL_TECHNIQUES_SPURIOUS


def test_a_covering_body_decides_the_reason_a_failed_ensemble_reports() -> None:
    """The ensemble reports a symptom; the covering body is the cause.

    A frame where a body covers the extended field of view has no sky in it,
    so features from anything else are not evidence that it was navigable.
    Whatever the techniques then failed to agree on, the reason a reader wants
    is the geometry -- and this is the path where the reason comes from the
    ensemble rather than from a gate, which is the path that used to keep the
    ensemble's own answer.
    """
    obs = _FakeObs(extfov_margin=(8, 8))
    models = [_FakeTitanModel(obs, reliability=0.9), _FakeCoveringBodyModel(obs)]
    orch = NavOrchestrator(models, only_techniques=['TitanHazeNav'])
    result = orch.navigate(obs)  # type: ignore[arg-type]
    assert result.status == 'failed'
    assert result.status_reason == NavStatusReason.BODY_FILLS_FOV


def test_a_covering_body_does_not_cost_the_ensemble_its_diagnostics() -> None:
    """Substituting the reason keeps everything the ensemble worked out.

    The per-technique records are why a covering frame can still be read for
    what the techniques did, and replacing the whole result would lose them.
    """
    obs = _FakeObs(extfov_margin=(8, 8))
    plain = NavOrchestrator(
        [_FakeTitanModel(obs, reliability=0.9)], only_techniques=['TitanHazeNav']
    )
    covered = NavOrchestrator(
        [_FakeTitanModel(obs, reliability=0.9), _FakeCoveringBodyModel(obs)],
        only_techniques=['TitanHazeNav'],
    )
    without = plain.navigate(obs)  # type: ignore[arg-type]
    with_body = covered.navigate(obs)  # type: ignore[arg-type]
    assert without.status_reason == NavStatusReason.ALL_TECHNIQUES_SPURIOUS
    assert with_body.status_reason == NavStatusReason.BODY_FILLS_FOV
    assert len(with_body.per_technique) == len(without.per_technique)
    assert len(with_body.per_technique) > 0


def test_orchestrator_titan_spurious_result_names_a_gate() -> None:
    """The spurious Titan result attributes its failure to a named fit gate."""
    obs = _FakeObs(extfov_margin=(8, 8))
    model = _FakeTitanModel(obs, reliability=0.9)
    orch = NavOrchestrator([model], only_techniques=['TitanHazeNav'])
    result = orch.navigate(obs)  # type: ignore[arg-type]
    diagnostics = result.per_technique[0].diagnostics
    assert diagnostics.gate_failed is not None  # type: ignore[union-attr]


def test_normalize_model_patterns_expands_bare_prefix() -> None:
    """A bare prefix without colon expands to ``prefix:*`` so ``rings`` matches ``rings:SATURN``."""
    from spindoctor.nav_orchestrator.orchestrator import _normalize_model_patterns

    names = ['stars', 'rings:SATURN', 'body:DIONE', 'body:SATURN']
    out = _normalize_model_patterns('rings', names)
    assert 'rings:*' in out


def test_normalize_model_patterns_uppercases_value_after_colon() -> None:
    """``body:saturn`` is normalized to ``body:SATURN``."""
    from spindoctor.nav_orchestrator.orchestrator import _normalize_model_patterns

    names = ['body:SATURN', 'body:DIONE']
    assert _normalize_model_patterns('body:saturn', names) == ['body:SATURN']


def test_normalize_model_patterns_preserves_bang_exclusion() -> None:
    """A leading ``!`` is preserved through the normalization."""
    from spindoctor.nav_orchestrator.orchestrator import _normalize_model_patterns

    names = ['stars', 'rings:SATURN', 'body:DIONE']
    out = _normalize_model_patterns('!rings', names)
    assert '!rings:*' in out


def test_normalize_model_patterns_preserves_star_token_for_unnamespaced_models() -> None:
    """``stars`` (with no namespace) still matches ``stars`` after normalization."""
    from spindoctor.nav_orchestrator.orchestrator import _normalize_model_patterns

    names = ['stars', 'rings:SATURN']
    # With both styles in the registry the bare token should match both
    # the literal name and any namespaced variant.
    out = _normalize_model_patterns('stars', names)
    assert 'stars' in out  # matches 'stars'
    assert 'stars:*' in out  # matches potential 'stars:FOO'


def test_orchestrator_only_models_filter_drops_models(fake_obs: _FakeObs) -> None:
    """only_models='!stars' drops the stars model entirely."""
    obs = fake_obs
    model = _FakeStarModel(obs, feature_count=3)
    orch = NavOrchestrator([model], only_models='!stars')
    result = orch.navigate(obs)  # type: ignore[arg-type]
    assert result.status == 'failed'
    assert result.status_reason == NavStatusReason.NO_FEATURES_EXTRACTED


def test_orchestrator_only_techniques_filter_drops_techniques(
    fake_obs: _FakeObs,
) -> None:
    """An exclusion glob that rejects every STAR technique yields ``no_feasible_techniques``.

    The exclude list covers ``_FakeStarTechnique`` plus every shipped
    STAR-accepting NavTechnique so the only feasible matches against
    the fake STAR features are filtered out, leaving an empty pass-1
    result list and the corresponding status reason.
    """
    obs = fake_obs
    model = _FakeStarModel(obs, feature_count=3)
    orch = NavOrchestrator(
        [model],
        only_techniques=[
            '!_FakeStarTechnique',
            '!Star*',
        ],
    )
    result = orch.navigate(obs)  # type: ignore[arg-type]
    assert result.status == 'failed'
    assert result.status_reason == NavStatusReason.NO_FEASIBLE_TECHNIQUES


def test_orchestrator_only_models_mixed_include_exclude(fake_obs: _FakeObs) -> None:
    """Mixed include/exclude patterns on ``only_models`` apply both gates.

    ``only_models`` accepts the same glob-with-negation grammar as
    ``only_techniques``: include patterns admit every match; the
    leading-bang exclusion drops any name matching the exclude
    pattern, applied after inclusion.
    """
    obs = fake_obs
    model = _FakeStarModel(obs, feature_count=3)
    # Include everything ('*') but exclude names starting with 'st' (the
    # stars model).  No models survive, so feature extraction yields the
    # NO_FEATURES_EXTRACTED status the single-pattern test already covers.
    orch = NavOrchestrator([model], only_models=['*', '!st*'])
    result = orch.navigate(obs)  # type: ignore[arg-type]
    assert result.status == 'failed'
    assert result.status_reason == NavStatusReason.NO_FEATURES_EXTRACTED


def test_orchestrator_only_models_mixed_keeps_matching_inclusion(
    fake_obs: _FakeObs,
) -> None:
    """Mixed pattern ``['stars', '!ring*']`` keeps ``stars`` (no exclusion match)."""
    obs = fake_obs
    model = _FakeStarModel(obs, feature_count=3)
    # Include only 'stars'; exclude any 'ring*' (no match — kept).
    orch = NavOrchestrator(
        [model],
        only_models=['stars', '!ring*'],
        only_techniques=['_FakeStarTechnique'],
    )
    result = orch.navigate(obs)  # type: ignore[arg-type]
    assert result.status == 'success'


def test_orchestrator_only_techniques_mixed_include_exclude(
    fake_obs: _FakeObs,
) -> None:
    """Mixed include/exclude patterns on ``only_techniques`` apply both gates."""
    obs = fake_obs
    model = _FakeStarModel(obs, feature_count=3)
    # Include everything but exclude techniques whose name contains 'Pass'.
    # ``_FakeStarTechnique`` survives; ``_PassTwoTechnique`` (registered
    # later in this module) does not.
    orch = NavOrchestrator(
        [model],
        only_techniques=['*', '!*Pass*'],
    )
    result = orch.navigate(obs)  # type: ignore[arg-type]
    assert result.status == 'success'
    technique_names = {t.technique_name for t in result.per_technique}
    assert '_FakeStarTechnique' in technique_names
    assert '_PassTwoTechnique' not in technique_names


def test_orchestrator_marks_technique_results_in_inventory(
    fake_obs: _FakeObs,
) -> None:
    """Successful runs carry a feature_inventory listing the kept features."""
    obs = fake_obs
    model = _FakeStarModel(obs, feature_count=3)
    orch = NavOrchestrator([model])
    result = orch.navigate(obs)  # type: ignore[arg-type]
    assert len(result.feature_inventory) == 3
    assert all(not entry.gated for entry in result.feature_inventory)
    assert all(entry.gate_reason is None for entry in result.feature_inventory)


def test_orchestrator_passes_classifier_thresholds_through(fake_obs: _FakeObs) -> None:
    """Custom ImageQualityThresholds change the classification verdict."""
    image = np.full((64, 64), 4095.0, np.float64)
    obs = _FakeObs(image=image)
    model = _FakeStarModel(obs, feature_count=3)
    orch = NavOrchestrator(
        [model],
        image_quality_thresholds=ImageQualityThresholds(saturation_threshold_dn=4095.0),
    )
    result = orch.navigate(obs)  # type: ignore[arg-type]
    assert result.status_reason == NavStatusReason.IMAGE_OVEREXPOSED


class _RaisingModel(NavModel):
    """NavModel whose ``to_features`` always raises a RuntimeError."""

    def __init__(self, obs: Any) -> None:
        super().__init__('raising', obs)

    def create_model(self) -> None:
        self._metadata['raised'] = True

    def to_features(self, context: NavContext) -> list[NavFeature]:
        raise RuntimeError('synthetic to_features failure')

    def to_annotations(self, context: NavContext) -> Annotations:
        raise RuntimeError('synthetic to_annotations failure')


class _RaisingTechnique(NavTechnique):
    """NavTechnique whose ``navigate`` always raises a RuntimeError."""

    name = '_RaisingTechnique'
    _abstract = True  # scoped to this module via _register_fakes
    accepts_feature_types = frozenset({NavFeatureType.STAR})

    def is_feasible(self, features: list[NavFeature]) -> NavFeasibilityReport:
        return NavFeasibilityReport(
            feasible=True, reason='ok', consumed_feature_count=len(features)
        )

    def navigate(self, features: list[NavFeature], context: NavContext) -> NavTechniqueResult:
        raise RuntimeError('synthetic navigate failure')


class _PassTwoTechnique(NavTechnique):
    """Pass-2 (requires_prior=True) technique that captures the prior offset."""

    name = '_PassTwoTechnique'
    _abstract = True  # scoped to this module via _register_fakes
    accepts_feature_types = frozenset({NavFeatureType.STAR})
    requires_prior = True
    captured_prior: tuple[float, float] | None = None

    def is_feasible(self, features: list[NavFeature]) -> NavFeasibilityReport:
        return NavFeasibilityReport(feasible=True, reason='ok')

    def navigate(self, features: list[NavFeature], context: NavContext) -> NavTechniqueResult:
        type(self).captured_prior = context.prior_offset_px
        return NavTechniqueResult(
            technique_name=self.name,
            feature_ids=tuple(f.feature_id for f in features),
            offset_px=(1.5, 2.5),
            covariance_px2=np.eye(2, dtype=np.float64) * 0.04,
            confidence=0.9,
            spurious=False,
            at_edge=False,
            diagnostics=StarFieldDiagnostics(n_inliers=len(features)),
        )


class _InfeasibleTechnique(NavTechnique):
    """NavTechnique whose ``is_feasible`` always refuses with a fixed reason."""

    name = '_InfeasibleTechnique'
    _abstract = True  # scoped to this module via _register_fakes
    accepts_feature_types = frozenset({NavFeatureType.STAR})

    def is_feasible(self, features: list[NavFeature]) -> NavFeasibilityReport:
        return NavFeasibilityReport(feasible=False, reason='needs at least 999 stars')

    def navigate(self, features: list[NavFeature], context: NavContext) -> NavTechniqueResult:
        raise AssertionError('an infeasible technique must never be navigated')


@pytest.fixture
def _register_raising_technique() -> Iterator[None]:
    """Make ``_RaisingTechnique`` name-resolvable for one test.

    Scoped to the tests that want it because a technique that raises fails
    the image, so a module-wide registration would fail every star-bearing
    scene in this file.
    """
    NavTechnique._registry.append(_RaisingTechnique)
    try:
        yield
    finally:
        NavTechnique._registry.remove(_RaisingTechnique)


def test_a_raising_to_features_fails_the_image(
    fake_obs: _FakeObs, capsys: pytest.CaptureFixture[str]
) -> None:
    """A NavModel.to_features that raises fails the image as internal_error."""
    obs = fake_obs
    model = _RaisingModel(obs)
    orch = NavOrchestrator([model])
    result = orch.navigate(obs)  # type: ignore[arg-type]
    captured = capsys.readouterr()
    assert result.status == 'failed'
    assert result.status_reason == NavStatusReason.INTERNAL_ERROR
    assert result.internal_error is not None
    assert result.internal_error.component == 'raising.to_features'
    assert result.internal_error.exception_type == 'RuntimeError'
    assert 'synthetic to_features failure' in captured.out


def test_a_raising_create_model_fails_the_image(fake_obs: _FakeObs) -> None:
    """A NavModel.create_model that raises fails the image as internal_error."""
    obs = fake_obs

    class _BadBuildModel(_FakeStarModel):
        """Star model whose ``create_model`` raises."""

        def create_model(self) -> None:
            """Raise instead of building."""
            raise ValueError('synthetic create_model failure')

    orch = NavOrchestrator([_BadBuildModel(obs, feature_count=2)])
    result = orch.navigate(obs)  # type: ignore[arg-type]
    assert result.status == 'failed'
    assert result.status_reason == NavStatusReason.INTERNAL_ERROR
    assert result.internal_error is not None
    assert result.internal_error.component == 'stars.create_model'
    assert result.internal_error.exception_type == 'ValueError'


@pytest.mark.usefixtures('_register_raising_technique')
def test_a_raising_technique_fails_the_image(
    fake_obs: _FakeObs, capsys: pytest.CaptureFixture[str]
) -> None:
    """A NavTechnique.navigate that raises fails the image as internal_error."""
    obs = fake_obs
    model = _FakeStarModel(obs, feature_count=3)
    orch = NavOrchestrator([model], only_techniques='_RaisingTechnique')
    result = orch.navigate(obs)  # type: ignore[arg-type]
    captured = capsys.readouterr()
    assert result.status == 'failed'
    assert result.status_reason == NavStatusReason.INTERNAL_ERROR
    assert result.internal_error is not None
    assert result.internal_error.component == '_RaisingTechnique.navigate'
    assert result.internal_error.exception_type == 'RuntimeError'
    assert 'synthetic navigate failure' in captured.out


@pytest.mark.usefixtures('_register_raising_technique')
def test_a_raising_technique_is_not_answered_from_the_survivors(
    fake_obs: _FakeObs,
) -> None:
    """One technique raising fails the image although another one succeeded.

    This is the property the whole change exists for: before it, the working
    technique's offset was reported as a success and nothing downstream could
    tell that answer apart from one computed with every technique intact.
    """
    obs = fake_obs
    model = _FakeStarModel(obs, feature_count=3)
    orch = NavOrchestrator([model], only_techniques=['_FakeStarTechnique', '_RaisingTechnique'])
    result = orch.navigate(obs)  # type: ignore[arg-type]
    assert result.status == 'failed'
    assert result.status_reason == NavStatusReason.INTERNAL_ERROR
    assert result.offset_px is None


def test_a_contract_violation_is_not_reclassified_as_an_internal_error(
    fake_obs: _FakeObs,
) -> None:
    """A NavContractError still reports contract_violation, carrying no record."""
    obs = fake_obs

    class _ContractBreakingModel(_FakeStarModel):
        """Star model whose ``to_features`` violates an internal contract."""

        def to_features(self, context: NavContext) -> list[NavFeature]:
            """Raise the contract error."""
            raise NavContractError('synthetic contract violation')

    orch = NavOrchestrator([_ContractBreakingModel(obs, feature_count=2)])
    result = orch.navigate(obs)  # type: ignore[arg-type]
    assert result.status_reason == NavStatusReason.CONTRACT_VIOLATION
    assert result.internal_error is None


def test_orchestrator_ensemble_contract_violation_yields_failed_result(
    fake_obs: _FakeObs, capsys: pytest.CaptureFixture[str]
) -> None:
    """An over-bound rotation reaching the ensemble fails with contract_violation."""

    class _OverRotatedTechnique(NavTechnique):
        """Technique whose 3-DoF rotation violates the ensemble small-angle bound."""

        name = '_OverRotatedTechnique'
        accepts_feature_types = frozenset({NavFeatureType.STAR})

        def is_feasible(self, features: list[NavFeature]) -> NavFeasibilityReport:
            return NavFeasibilityReport(
                feasible=True, reason='ok', consumed_feature_count=len(features)
            )

        def navigate(self, features: list[NavFeature], context: NavContext) -> NavTechniqueResult:
            return NavTechniqueResult(
                technique_name=self.name,
                feature_ids=tuple(f.feature_id for f in features),
                offset_px=(1.5, 2.5),
                covariance_px2=np.diag([0.04, 0.04, 1.0e-4]).astype(np.float64),
                confidence=0.85,
                spurious=False,
                at_edge=False,
                diagnostics=StarFieldDiagnostics(n_inliers=len(features)),
                rotation_rad=float(np.radians(10.0)),
                sigma_rotation_rad=0.01,
            )

    try:
        obs = fake_obs
        model = _FakeStarModel(obs, feature_count=3)
        orch = NavOrchestrator([model], only_techniques=['_OverRotatedTechnique'])
        result = orch.navigate(obs)  # type: ignore[arg-type]
    finally:
        NavTechnique._registry.remove(_OverRotatedTechnique)
    captured = capsys.readouterr()
    assert result.status == 'failed'
    assert result.status_reason == NavStatusReason.CONTRACT_VIOLATION
    assert 'CONTRACT VIOLATION' in captured.out
    assert 'violates small-angle bound' in captured.out


def test_orchestrator_technique_contract_error_not_swallowed_as_no_result(
    fake_obs: _FakeObs, capsys: pytest.CaptureFixture[str]
) -> None:
    """A NavContractError from a technique is not sandboxed as an ordinary failure."""

    class _ContractRaisingTechnique(NavTechnique):
        """Technique whose ``navigate`` raises NavContractError directly."""

        name = '_ContractRaisingTechnique'
        accepts_feature_types = frozenset({NavFeatureType.STAR})

        def is_feasible(self, features: list[NavFeature]) -> NavFeasibilityReport:
            return NavFeasibilityReport(
                feasible=True, reason='ok', consumed_feature_count=len(features)
            )

        def navigate(self, features: list[NavFeature], context: NavContext) -> NavTechniqueResult:
            raise NavContractError('synthetic contract violation in navigate')

    try:
        obs = fake_obs
        model = _FakeStarModel(obs, feature_count=3)
        orch = NavOrchestrator([model], only_techniques=['_ContractRaisingTechnique'])
        result = orch.navigate(obs)  # type: ignore[arg-type]
    finally:
        NavTechnique._registry.remove(_ContractRaisingTechnique)
    captured = capsys.readouterr()
    # An ordinary technique failure would yield no_feasible_techniques (the
    # sandbox drops the result); the contract violation must surface instead.
    assert result.status_reason == NavStatusReason.CONTRACT_VIOLATION
    assert 'CONTRACT VIOLATION in NavTechnique _ContractRaisingTechnique' in captured.out
    assert 'synthetic contract violation in navigate' in captured.out


def test_orchestrator_pass2_receives_pass1_prior(fake_obs: _FakeObs) -> None:
    """Pass-2 techniques observe the pass-1 ensemble's offset as ``prior_offset_px``."""
    obs = fake_obs
    model = _FakeStarModel(obs, feature_count=3)
    _PassTwoTechnique.captured_prior = None
    orch = NavOrchestrator(
        [model],
        only_techniques=['_FakeStarTechnique', '_PassTwoTechnique'],
    )
    result = orch.navigate(obs)  # type: ignore[arg-type]
    assert result.status == 'success'
    # The pass-2 technique captured the pass-1 ensemble's offset (which equals
    # the single _FakeStarTechnique offset of (1.5, 2.5)).
    assert _PassTwoTechnique.captured_prior == (1.5, 2.5)


def test_orchestrator_records_model_metadata(fake_obs: _FakeObs) -> None:
    """The NavResult.model_metadata dict is populated from each model.metadata."""
    obs = fake_obs
    model = _FakeStarModel(obs, feature_count=2)
    orch = NavOrchestrator([model])
    result = orch.navigate(obs)  # type: ignore[arg-type]
    assert 'stars' in result.model_metadata
    assert result.model_metadata['stars']['feature_count'] == 2


def test_orchestrator_records_annotations(fake_obs: _FakeObs) -> None:
    """Annotations from every NavModel are merged into NavResult.annotations.

    The navigation is asserted to have reached the merge, because a failed
    result carries an empty ``Annotations`` too: checking only the type
    passes on a run that never called ``to_annotations`` at all.
    """
    obs = fake_obs
    model = _FakeStarModel(obs, feature_count=2)
    orch = NavOrchestrator([model])
    result = orch.navigate(obs)  # type: ignore[arg-type]
    assert result.status_reason != NavStatusReason.INTERNAL_ERROR
    assert isinstance(result.annotations, Annotations)


def test_orchestrator_low_reliability_features_all_gated(fake_obs: _FakeObs) -> None:
    """Every feature below the gate yields ALL_FEATURES_GATED."""
    obs = fake_obs

    class _LowReliabilityStarModel(_FakeStarModel):
        def to_features(self, context: NavContext) -> list[NavFeature]:
            return [replace(f, reliability=0.01) for f in super().to_features(context)]

    model = _LowReliabilityStarModel(obs, feature_count=3)
    orch = NavOrchestrator([model])
    result = orch.navigate(obs)  # type: ignore[arg-type]
    assert result.status_reason == NavStatusReason.ALL_FEATURES_GATED


def test_a_raising_to_annotations_fails_the_image(
    fake_obs: _FakeObs, capsys: pytest.CaptureFixture[str]
) -> None:
    """A failing to_annotations fails the image, though a sibling model worked.

    Annotations only feed the summary PNG, so failing the whole image for one
    is the severest reading of the rule.  It is the rule all the same: a model
    that cannot describe what it found is in a state nobody planned for, and
    an offset reported from a run that hit one would be indistinguishable from
    an offset reported from a run that did not.
    """
    obs = fake_obs

    class _GoodModel(_FakeStarModel):
        """Star model that annotates normally."""

    class _BadAnnotationModel(_FakeStarModel):
        """Star model whose ``to_annotations`` raises."""

        def to_annotations(self, context: NavContext) -> Annotations:
            """Raise instead of annotating."""
            raise RuntimeError('synthetic to_annotations failure')

    bad = _BadAnnotationModel(obs, feature_count=2)
    good = _GoodModel(obs, feature_count=1)
    orch = NavOrchestrator([bad, good], only_techniques=['_FakeStarTechnique'])
    result = orch.navigate(obs)  # type: ignore[arg-type]
    captured = capsys.readouterr()
    assert result.status == 'failed'
    assert result.status_reason == NavStatusReason.INTERNAL_ERROR
    assert result.internal_error is not None
    assert result.internal_error.component == 'stars.to_annotations'
    assert 'synthetic to_annotations failure' in captured.out


def test_prepare_returns_context_and_features(fake_obs: _FakeObs) -> None:
    """``prepare`` builds the context and returns the gated feature list."""
    obs = fake_obs
    model = _FakeStarModel(obs, feature_count=3)
    orch = NavOrchestrator([model])
    prep = orch.prepare(obs)  # type: ignore[arg-type]
    assert prep.context.obs is obs
    assert len(prep.features) == 3
    assert {f.feature_type for f in prep.features} == {NavFeatureType.STAR}


def test_prepare_does_not_short_circuit_on_blank_image() -> None:
    """``prepare`` keeps going on hard-failure images so manual nav can inspect them.

    ``navigate`` returns a failed NavResult on a blank image, but
    ``prepare`` is the entry point for the manual-nav dialog — the
    operator may legitimately want to look at a blank or saturated frame.
    """
    obs = _FakeObs(image=np.zeros((64, 64), np.float64))
    model = _FakeStarModel(obs, feature_count=0)
    orch = NavOrchestrator([model])
    prep = orch.prepare(obs)  # type: ignore[arg-type]
    assert prep.context.image_classifier.image_class == 'blank'
    assert prep.features == []


def test_prepare_drops_models_via_only_models_filter() -> None:
    """The ``only_models`` glob filter is honored by ``prepare`` too."""
    obs = _FakeObs()
    model = _FakeStarModel(obs, feature_count=3)
    orch = NavOrchestrator([model], only_models='!stars')
    prep = orch.prepare(obs)  # type: ignore[arg-type]
    assert prep.features == []


def test_prepare_apply_gate_false_returns_gated_features() -> None:
    """``apply_gate=False`` returns every emitted feature, including gated ones."""

    class _LowReliabilityModel(_FakeStarModel):
        def to_features(self, context: NavContext) -> list[NavFeature]:
            return [replace(f, reliability=0.01) for f in super().to_features(context)]

    obs = _FakeObs()
    model = _LowReliabilityModel(obs, feature_count=2)
    orch = NavOrchestrator([model])
    gated_prep = orch.prepare(obs, apply_gate=True)  # type: ignore[arg-type]
    full_prep = orch.prepare(obs, apply_gate=False)  # type: ignore[arg-type]
    assert gated_prep.features == []
    assert len(full_prep.features) == 2


# --- Pass-1 fallback-skip regression tests ---


class _FakeBodyModel(NavModel):
    """Fake NavModel that emits one BODY_DISC + one BODY_BLOB for one body."""

    def __init__(self, obs: Any, *, body_name: str = 'TestMoon') -> None:
        super().__init__('body:test', obs)
        self._body_name = body_name

    def create_model(self) -> None:
        self._metadata['feature_count'] = 2

    def to_features(self, context: NavContext) -> list[NavFeature]:
        disc = NavFeature(
            feature_id=f'body_disc:{self._body_name}',
            feature_type=NavFeatureType.BODY_DISC,
            source_model='body:test',
            geometry=BodyDiscGeometry(
                bbox_extfov_vu=(0, 0, 16, 16),
                predicted_center_vu=(8.0, 8.0),
                overflow_fraction=0.0,
            ),
            usable_types=frozenset({NavFeatureType.BODY_DISC}),
            flags=BodyDiscFlags(body_name=self._body_name),
            subject_range_km=1.0e8,
            position_cov_px=np.eye(2, dtype=np.float64) * 0.25,
            intensity_sigma_rel=0.05,
            preferred_filter=NavFilterSpec(kind=NavFilterKind.NONE),
            reliability=0.8,
            reliability_reasons=NavReliabilityBreakdown(visible_lit_fraction=1.0),
        )
        blob = NavFeature(
            feature_id=f'body_blob:{self._body_name}',
            feature_type=NavFeatureType.BODY_BLOB,
            source_model='body:test',
            geometry=BodyBlobGeometry(
                bbox_extfov_vu=(0, 0, 16, 16),
                predicted_center_vu=(8.0, 8.0),
                predicted_diameter_px=8.0,
            ),
            usable_types=frozenset({NavFeatureType.BODY_BLOB}),
            flags=BodyBlobFlags(body_name=self._body_name, predicted_diameter_px=8.0),
            subject_range_km=1.0e8,
            position_cov_px=np.eye(2, dtype=np.float64) * 0.25,
            intensity_sigma_rel=0.05,
            preferred_filter=NavFilterSpec(kind=NavFilterKind.NONE),
            reliability=0.8,
            reliability_reasons=NavReliabilityBreakdown(visible_lit_fraction=1.0),
        )
        return [disc, blob]

    def to_annotations(self, context: NavContext) -> Annotations:
        return Annotations()


class _FakeBodyPrimary(NavTechnique):
    """Primary technique consuming BODY_DISC; tracks how many times it ran."""

    name = '_FakeBodyPrimary'
    _abstract = True  # scoped to this module via _register_fakes
    accepts_feature_types = frozenset({NavFeatureType.BODY_DISC})
    tier = 'primary'
    run_count: int = 0
    spurious_override: bool = False

    def is_feasible(self, features: list[NavFeature]) -> NavFeasibilityReport:
        eligible = [f for f in features if f.feature_type is NavFeatureType.BODY_DISC]
        if not eligible:
            return NavFeasibilityReport(feasible=False, reason='no_body_disc')
        return NavFeasibilityReport(
            feasible=True, reason='ok', consumed_feature_count=len(eligible)
        )

    def navigate(self, features: list[NavFeature], context: NavContext) -> NavTechniqueResult:
        type(self).run_count += 1
        return NavTechniqueResult(
            technique_name=self.name,
            feature_ids=tuple(f.feature_id for f in features),
            offset_px=(1.0, 2.0),
            covariance_px2=np.eye(2, dtype=np.float64) * 0.25,
            confidence=0.85,
            spurious=type(self).spurious_override,
            at_edge=False,
            diagnostics=StarFieldDiagnostics(n_inliers=len(features)),
            source_bodies=body_names_from_features(features),
        )


class _FakeBodyFallback(NavTechnique):
    """Fallback technique consuming BODY_BLOB; tracks how many times it ran."""

    name = '_FakeBodyFallback'
    _abstract = True  # scoped to this module via _register_fakes
    accepts_feature_types = frozenset({NavFeatureType.BODY_BLOB})
    tier = 'fallback'
    run_count: int = 0

    def is_feasible(self, features: list[NavFeature]) -> NavFeasibilityReport:
        eligible = [f for f in features if f.feature_type is NavFeatureType.BODY_BLOB]
        if not eligible:
            return NavFeasibilityReport(feasible=False, reason='no_body_blob')
        return NavFeasibilityReport(
            feasible=True, reason='ok', consumed_feature_count=len(eligible)
        )

    def navigate(self, features: list[NavFeature], context: NavContext) -> NavTechniqueResult:
        type(self).run_count += 1
        return NavTechniqueResult(
            technique_name=self.name,
            feature_ids=tuple(f.feature_id for f in features),
            offset_px=(0.5, 0.5),
            covariance_px2=np.eye(2, dtype=np.float64) * 0.25,
            confidence=0.6,
            spurious=False,
            at_edge=False,
            diagnostics=StarFieldDiagnostics(n_inliers=len(features)),
            source_bodies=body_names_from_features(features),
        )


def test_orchestrator_skips_fallback_when_primary_covers_body() -> None:
    """A non-spurious primary suppresses the fallback technique on the same body.

    Operators expect the fallback to run only when the primary fails;
    the prior behaviour ran every feasible technique and dropped the
    fallback in the ensemble post-hoc, wasting its compute.  This
    regression test verifies the primary-then-fallback scheduling
    actually skips the fallback's ``navigate`` call.
    """
    _FakeBodyPrimary.run_count = 0
    _FakeBodyPrimary.spurious_override = False
    _FakeBodyFallback.run_count = 0
    obs = _FakeObs()
    model = _FakeBodyModel(obs, body_name='TestMoon')
    orch = NavOrchestrator([model], only_techniques=['_FakeBodyPrimary', '_FakeBodyFallback'])
    result = orch.navigate(obs)  # type: ignore[arg-type]
    assert result.status == 'success'
    assert _FakeBodyPrimary.run_count == 1
    # Fallback is NOT called because the primary covered TestMoon.
    assert _FakeBodyFallback.run_count == 0
    technique_names = {t.technique_name for t in result.per_technique}
    assert '_FakeBodyPrimary' in technique_names
    assert '_FakeBodyFallback' not in technique_names


def test_orchestrator_runs_fallback_when_primary_is_spurious() -> None:
    """A spurious primary does not cover the body — the fallback runs."""
    _FakeBodyPrimary.run_count = 0
    _FakeBodyPrimary.spurious_override = True
    _FakeBodyFallback.run_count = 0
    obs = _FakeObs()
    model = _FakeBodyModel(obs, body_name='TestMoon')
    orch = NavOrchestrator([model], only_techniques=['_FakeBodyPrimary', '_FakeBodyFallback'])
    result = orch.navigate(obs)  # type: ignore[arg-type]
    # The orchestrator returns whatever status the ensemble derives —
    # the contract under test is that the fallback DID run.
    del result
    assert _FakeBodyPrimary.run_count == 1
    assert _FakeBodyFallback.run_count == 1
    _FakeBodyPrimary.spurious_override = False  # reset for any later test


# --- Calibrated-IF NaN missing-data sentinel regression tests (CODE-ORCH-003) ---


def _ciss_calib_inst_config() -> dict[str, Any]:
    """Return a CISS-CALIB-style inst_config: calibrated_if + NaN marker.

    Mirrors the shape ``instrument_settings_from_obs`` expects for a
    calibrated-IF camera: ``data_units='calibrated_if'``,
    ``noise.marker_value: NaN``, and the I/F-keyed
    ``image_quality_thresholds`` block (no DN-keyed fields, no
    ``saturation_threshold_if``).
    """
    return {
        'data_units': 'calibrated_if',
        'noise': {'marker_value': 'NaN'},
        'image_quality_thresholds': {
            'max_overexposed_frac_clean': 0.80,
            'max_missing_frac_clean': 0.30,
            'partial_dropout_min_frac': 0.05,
            'blank_max_if': 1.0e-4,
            'noisy_threshold_if': 0.01,
        },
    }


def test_orchestrator_calibrated_if_nan_pixels_do_not_raise() -> None:
    """A calibrated-IF image with NaN missing-data markers navigates without raising.

    For ``calibrated_if`` instruments the missing-data sentinel is NaN.
    The orchestrator must sanitise those NaN before the finite-only
    derivative path runs; otherwise ``_smooth_and_compute_gradients``
    raises a ValueError that would propagate out of ``navigate`` and
    violate the never-raise contract.
    """
    image = np.full((64, 64), 0.5, np.float64)
    image[:6, :] = np.nan  # ~9.4% NaN (missing) — below mostly_missing threshold
    obs = _FakeObs(image=image, inst_config=_ciss_calib_inst_config())
    model = _FakeStarModel(obs, feature_count=3)
    orch = NavOrchestrator([model], only_techniques=['_FakeStarTechnique'])
    result = orch.navigate(obs)  # type: ignore[arg-type]
    # The pipeline ran end-to-end and produced a successful offset.
    assert result.status == 'success'
    assert result.offset_px == (1.5, 2.5)


def test_orchestrator_calibrated_if_missing_frac_reflects_nan() -> None:
    """The classifier verdict's missing_frac reflects the NaN fraction.

    The orchestrator threads the true missing fraction (computed from the
    NaN mask before sanitisation) into the classifier rather than relying
    on ``sensor == marker`` (which can never match NaN).
    """
    image = np.full((64, 64), 0.5, np.float64)
    image[:6, :] = np.nan  # ~9.4% NaN
    obs = _FakeObs(image=image, inst_config=_ciss_calib_inst_config())
    model = _FakeStarModel(obs, feature_count=3)
    orch = NavOrchestrator([model], only_techniques=['_FakeStarTechnique'])
    prep = orch.prepare(obs)  # type: ignore[arg-type]
    assert prep.image_classifier.missing_frac == pytest.approx(6.0 / 64.0)


def test_orchestrator_calibrated_if_mostly_nan_short_circuits() -> None:
    """A calibrated-IF image dominated by NaN markers short-circuits as missing.

    With the NaN fraction above ``max_missing_frac_clean`` the classifier
    returns ``mostly_missing_data`` and the orchestrator fails before any
    technique runs — proving the NaN-aware missing detection actually
    drives the hard-failure short-circuit for calibrated images.
    """
    image = np.full((64, 64), 0.5, np.float64)
    image[:48, :] = np.nan  # 75% NaN, above the 0.30 clean threshold
    obs = _FakeObs(image=image, inst_config=_ciss_calib_inst_config())
    model = _FakeStarModel(obs, feature_count=3)
    orch = NavOrchestrator([model], only_techniques=['_FakeStarTechnique'])
    result = orch.navigate(obs)  # type: ignore[arg-type]
    assert result.status == 'failed'
    assert result.status_reason == NavStatusReason.MISSING_DATA_DOMINANT
    assert result.per_technique == []


def test_orchestrator_stamps_pass2_prior_source_techniques(fake_obs: _FakeObs) -> None:
    """Pass-2 results carry the technique names whose consensus seeded the prior."""
    obs = fake_obs
    model = _FakeStarModel(obs, feature_count=3)
    orch = NavOrchestrator(
        [model],
        only_techniques=['_FakeStarTechnique', '_PassTwoTechnique'],
    )
    result = orch.navigate(obs)  # type: ignore[arg-type]
    assert result.status == 'success'
    by_name = {r.technique_name: r for r in result.per_technique}
    assert by_name['_PassTwoTechnique'].prior_source_techniques == frozenset({'_FakeStarTechnique'})
    # Pass-1 results are never stamped.
    assert by_name['_FakeStarTechnique'].prior_source_techniques == frozenset()


# --- Status-reason INFO emission at every failed-nav site (#180) ---
#
# pdslogger writes through its own stream handler, so the emitted lines
# are captured with ``capsys`` (never ``caplog``).  Each test provokes
# one NavResult.failed site and asserts the operator-readable
# STATUS_REASON_INFO_TEMPLATE lines actually reached the log.


def test_blank_image_failure_emits_status_reason_info(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The hard-failure short-circuit (blank frame) emits the templated INFO lines."""
    obs = _FakeObs(image=np.zeros((64, 64), np.float64))
    model = _FakeStarModel(obs, feature_count=3)
    orch = NavOrchestrator([model])
    result = orch.navigate(obs)  # type: ignore[arg-type]
    captured = capsys.readouterr()
    assert result.status_reason == NavStatusReason.NO_SIGNAL_IN_IMAGE
    assert 'Final: status=failed' in captured.out
    assert 'Image classifier: blank / dark frame' in captured.out


def test_overexposed_image_failure_emits_status_reason_info(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The hard-failure short-circuit (overexposed frame) emits the templated INFO lines."""
    obs = _FakeObs(image=np.full((64, 64), 4095.0, np.float64))
    model = _FakeStarModel(obs, feature_count=3)
    orch = NavOrchestrator(
        [model],
        image_quality_thresholds=ImageQualityThresholds(saturation_threshold_dn=4095.0),
    )
    result = orch.navigate(obs)  # type: ignore[arg-type]
    captured = capsys.readouterr()
    assert result.status_reason == NavStatusReason.IMAGE_OVEREXPOSED
    assert 'Final: status=failed' in captured.out
    assert 'Image classifier: most pixels at full-well DN' in captured.out


def test_mostly_missing_failure_emits_status_reason_info(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The hard-failure short-circuit (missing-data frame) emits the templated INFO lines."""
    image = np.full((64, 64), 0.5, np.float64)
    image[:48, :] = np.nan  # 75% NaN, above the 0.30 clean threshold
    obs = _FakeObs(image=image, inst_config=_ciss_calib_inst_config())
    model = _FakeStarModel(obs, feature_count=3)
    orch = NavOrchestrator([model], only_techniques=['_FakeStarTechnique'])
    result = orch.navigate(obs)  # type: ignore[arg-type]
    captured = capsys.readouterr()
    assert result.status_reason == NavStatusReason.MISSING_DATA_DOMINANT
    assert 'Final: status=failed' in captured.out
    assert 'Image classifier: missing-data marker dominates' in captured.out


def test_no_features_extracted_emits_status_reason_info(
    fake_obs: _FakeObs, capsys: pytest.CaptureFixture[str]
) -> None:
    """The no-features gate failure emits the templated INFO lines."""
    obs = fake_obs
    model = _FakeStarModel(obs, feature_count=0)
    orch = NavOrchestrator([model])
    result = orch.navigate(obs)  # type: ignore[arg-type]
    captured = capsys.readouterr()
    assert result.status_reason == NavStatusReason.NO_FEATURES_EXTRACTED
    assert 'Final: status=failed' in captured.out
    assert 'No extractor produced a feature' in captured.out


def test_all_features_gated_emits_status_reason_info(
    fake_obs: _FakeObs, capsys: pytest.CaptureFixture[str]
) -> None:
    """The all-features-gated failure emits the templated INFO lines."""
    obs = fake_obs

    class _LowReliabilityStarModel(_FakeStarModel):
        def to_features(self, context: NavContext) -> list[NavFeature]:
            return [replace(f, reliability=0.01) for f in super().to_features(context)]

    model = _LowReliabilityStarModel(obs, feature_count=3)
    orch = NavOrchestrator([model])
    result = orch.navigate(obs)  # type: ignore[arg-type]
    captured = capsys.readouterr()
    assert result.status_reason == NavStatusReason.ALL_FEATURES_GATED
    assert 'Final: status=failed' in captured.out
    assert 'Every feature dropped by the reliability gate' in captured.out


def test_no_feasible_techniques_emits_status_reason_info(
    fake_obs: _FakeObs, capsys: pytest.CaptureFixture[str]
) -> None:
    """The no-feasible-techniques failure emits the templated INFO lines."""
    obs = fake_obs
    model = _FakeStarModel(obs, feature_count=3)
    orch = NavOrchestrator(
        [model],
        only_techniques=[
            '!_FakeStarTechnique',
            '!_InfeasibleTechnique',
            '!Star*',
        ],
    )
    result = orch.navigate(obs)  # type: ignore[arg-type]
    captured = capsys.readouterr()
    assert result.status_reason == NavStatusReason.NO_FEASIBLE_TECHNIQUES
    assert 'Final: status=failed' in captured.out
    assert "No technique's is_feasible returned True" in captured.out


def test_all_techniques_spurious_emits_status_reason_info(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An ensemble failure (every result spurious) emits the templated INFO lines.

    The failed NavResult is constructed inside ``ensemble``; the
    orchestrator caller is responsible for emitting the templated
    reason lines when it returns that result.
    """
    _FakeBodyPrimary.run_count = 0
    _FakeBodyPrimary.spurious_override = True
    try:
        obs = _FakeObs()
        model = _FakeBodyModel(obs, body_name='TestMoon')
        orch = NavOrchestrator([model], only_techniques=['_FakeBodyPrimary'])
        result = orch.navigate(obs)  # type: ignore[arg-type]
    finally:
        _FakeBodyPrimary.spurious_override = False
    captured = capsys.readouterr()
    assert result.status_reason == NavStatusReason.ALL_TECHNIQUES_SPURIOUS
    assert 'Final: status=failed' in captured.out
    assert 'Every technique returned spurious=True' in captured.out


def test_infeasible_technique_rejection_logged_at_debug(
    fake_obs: _FakeObs, capsys: pytest.CaptureFixture[str]
) -> None:
    """An infeasible technique's rejection reason lands in the DEBUG log."""
    # The orchestration section is opened at the orchestrator's configured
    # level, so a test asserting DEBUG output has to ask for DEBUG.
    set_log_levels(LogLevels(image='DEBUG'))
    obs = fake_obs
    model = _FakeStarModel(obs, feature_count=3)
    orch = NavOrchestrator([model], only_techniques=['_InfeasibleTechnique'])
    result = orch.navigate(obs)  # type: ignore[arg-type]
    captured = capsys.readouterr()
    assert result.status_reason == NavStatusReason.NO_FEASIBLE_TECHNIQUES
    assert 'Technique _InfeasibleTechnique infeasible: needs at least 999 stars' in captured.out


def test_feasible_technique_report_logged_at_debug(
    fake_obs: _FakeObs, capsys: pytest.CaptureFixture[str]
) -> None:
    """A feasible technique's consumed-feature report lands in the DEBUG log."""
    # The orchestration section is opened at the orchestrator's configured
    # level, so a test asserting DEBUG output has to ask for DEBUG.
    set_log_levels(LogLevels(image='DEBUG'))
    obs = fake_obs
    model = _FakeStarModel(obs, feature_count=3)
    orch = NavOrchestrator([model], only_techniques=['_FakeStarTechnique'])
    result = orch.navigate(obs)  # type: ignore[arg-type]
    captured = capsys.readouterr()
    assert result.status == 'success'
    assert 'Technique _FakeStarTechnique feasible: would consume 3 feature(s)' in captured.out


def test_orchestrator_records_no_pointing_without_a_spice_camera_frame(
    fake_obs: _FakeObs,
) -> None:
    """An observation with no known SPICE camera frame records no pointing."""
    obs = fake_obs
    model = _FakeStarModel(obs, feature_count=3)
    orch = NavOrchestrator([model], only_techniques=['_FakeStarTechnique'])
    result = orch.navigate(obs)  # type: ignore[arg-type]
    assert result.pointing is None


def test_orchestrator_stamps_pointing_on_a_successful_result(
    fake_obs: _FakeObs, monkeypatch: pytest.MonkeyPatch, sentinel_pointing: PointingSolution
) -> None:
    """A computed pointing solution is carried on the returned NavResult.

    The SPICE lookups behind the real computation need furnished kernels, so
    the wiring is exercised with a stand-in solution.
    """
    pointing = sentinel_pointing
    monkeypatch.setattr(
        'spindoctor.nav_orchestrator.orchestrator.compute_pointing',
        lambda obs, *, offset_px, rotation_fitted: pointing,
    )
    obs = fake_obs
    model = _FakeStarModel(obs, feature_count=3)
    orch = NavOrchestrator([model], only_techniques=['_FakeStarTechnique'])
    result = orch.navigate(obs)  # type: ignore[arg-type]
    assert result.status == 'success'
    assert result.pointing is pointing


def test_orchestrator_stamps_pointing_on_a_short_circuited_failure(
    monkeypatch: pytest.MonkeyPatch, sentinel_pointing: PointingSolution
) -> None:
    """A hard-failure image still records its uncorrected attitude and times."""
    pointing = sentinel_pointing
    monkeypatch.setattr(
        'spindoctor.nav_orchestrator.orchestrator.compute_pointing',
        lambda obs, *, offset_px, rotation_fitted: pointing,
    )
    obs = _FakeObs(image=np.zeros((64, 64), np.float64))
    model = _FakeStarModel(obs, feature_count=3)
    orch = NavOrchestrator([model])
    result = orch.navigate(obs)  # type: ignore[arg-type]
    assert result.status == 'failed'
    assert result.pointing is pointing


def test_orchestrator_passes_the_fitted_rotation_flag_through(
    fake_obs: _FakeObs, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fitted-rotation flag reflects whether the result carries a rotation."""
    seen: dict[str, Any] = {}

    def _record(obs: Any, *, offset_px: Any, rotation_fitted: bool) -> None:
        seen['offset_px'] = offset_px
        seen['rotation_fitted'] = rotation_fitted
        return None

    monkeypatch.setattr('spindoctor.nav_orchestrator.orchestrator.compute_pointing', _record)
    obs = fake_obs
    model = _FakeStarModel(obs, feature_count=3)
    orch = NavOrchestrator([model], only_techniques=['_FakeStarTechnique'])
    result = orch.navigate(obs)  # type: ignore[arg-type]
    assert seen['offset_px'] == result.offset_px
    assert seen['rotation_fitted'] is False


def test_orchestrator_reports_a_fitted_rotation_to_the_pointing(
    fake_obs: _FakeObs, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A result carrying a fitted rotation is reported as one.

    The rotation turns about a per-technique pivot that no result records, so
    an attitude cannot express it and none is claimed.  Reporting such a result
    as unrotated would publish a C-matrix computed from the translation alone,
    with the rotation silently dropped -- and the metadata is a product in its
    own right, whatever the kernel writer later does with it.
    """
    seen: dict[str, Any] = {}

    def _record(obs: Any, *, offset_px: Any, rotation_fitted: bool) -> None:
        seen['rotation_fitted'] = rotation_fitted
        return None

    monkeypatch.setattr('spindoctor.nav_orchestrator.orchestrator.compute_pointing', _record)
    obs = fake_obs
    model = _FakeStarModel(obs, feature_count=3)
    orch = NavOrchestrator([model], only_techniques=['_FakeStarTechnique'])
    result = orch.navigate(obs)  # type: ignore[arg-type]
    seen.clear()
    orch.with_pointing(replace(result, rotation_rad=0.0035), obs)  # type: ignore[arg-type]
    assert seen['rotation_fitted'] is True


def test_orchestrator_reports_but_survives_a_failed_pointing_computation(
    fake_obs: _FakeObs, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A pointing computation that raises is logged and leaves the result usable."""

    def _boom(obs: Any, *, offset_px: Any, rotation_fitted: bool) -> None:
        raise NavPointingError('cmatrix is not a proper rotation')

    monkeypatch.setattr('spindoctor.nav_orchestrator.orchestrator.compute_pointing', _boom)
    obs = fake_obs
    model = _FakeStarModel(obs, feature_count=3)
    orch = NavOrchestrator([model], only_techniques=['_FakeStarTechnique'])
    result = orch.navigate(obs)  # type: ignore[arg-type]
    assert result.status == 'success'
    assert result.pointing is None
    assert 'Could not compute the corrected pointing' in capsys.readouterr().out


def test_orchestrator_reports_a_failed_pointing_to_the_run_log(
    fake_obs: _FakeObs, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A failed pointing computation reaches the run log, not only the image log.

    A batch operator watching the run log must not have to open every
    per-image log to learn that pointing stopped being recorded.
    """

    def _boom(obs: Any, *, offset_px: Any, rotation_fitted: bool) -> None:
        raise NavPointingError('the furnished kernels cannot supply the rotation')

    monkeypatch.setattr('spindoctor.nav_orchestrator.orchestrator.compute_pointing', _boom)
    obs = fake_obs
    model = _FakeStarModel(obs, feature_count=3)
    orch = NavOrchestrator([model], only_techniques=['_FakeStarTechnique'])
    orch.navigate(obs)  # type: ignore[arg-type]
    assert 'Corrected pointing not recorded' in capsys.readouterr().out


def test_orchestrator_does_not_swallow_a_defect_in_the_pointing_computation(
    fake_obs: _FakeObs, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A programming error in the computation is not caught.

    Catching everything would let one typo silently drop pointing from an
    entire batch while every image still reported success.
    """

    def _typo(obs: Any, *, offset_px: Any, rotation_fitted: bool) -> None:
        raise AttributeError("'NoneType' object has no attribute 'vals'")

    monkeypatch.setattr('spindoctor.nav_orchestrator.orchestrator.compute_pointing', _typo)
    obs = fake_obs
    model = _FakeStarModel(obs, feature_count=3)
    orch = NavOrchestrator([model], only_techniques=['_FakeStarTechnique'])
    with pytest.raises(AttributeError, match='has no attribute'):
        orch.navigate(obs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    'defect',
    [ValueError, RuntimeError, LookupError, OSError],
    ids=['value_error', 'runtime_error', 'lookup_error', 'os_error'],
)
def test_orchestrator_does_not_swallow_an_ordinary_exception_from_the_pointing_computation(
    fake_obs: _FakeObs, monkeypatch: pytest.MonkeyPatch, defect: type[Exception]
) -> None:
    """Only the computation's own typed failure is absorbed.

    A defect inside the computation raises an ordinary exception type, and
    the types SPICE failures arrive as are the same ones ordinary defects
    raise.  Absorbing that family would make the two indistinguishable and
    drop pointing from a whole batch while every image reported success, so
    the computation reports what it expects as a ``NavPointingError`` and
    everything else propagates.
    """

    def _defective(obs: Any, *, offset_px: Any, rotation_fitted: bool) -> None:
        raise defect('index 3 is out of bounds for axis 0 with size 3')

    monkeypatch.setattr('spindoctor.nav_orchestrator.orchestrator.compute_pointing', _defective)
    obs = fake_obs
    model = _FakeStarModel(obs, feature_count=3)
    orch = NavOrchestrator([model], only_techniques=['_FakeStarTechnique'])
    with pytest.raises(defect, match='out of bounds'):
        orch.navigate(obs)  # type: ignore[arg-type]


def test_orchestrator_warns_when_a_registered_instrument_has_no_frame_mapping(
    fake_obs: _FakeObs, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An instrument that navigates but maps to no SPICE frame is reported.

    A fifth instrument added without its frame mapping would otherwise
    produce no corrected attitude for any of its images, with nothing above
    debug to say so.
    """
    monkeypatch.setattr(
        'spindoctor.nav_orchestrator.orchestrator.obs_class_to_inst_name', lambda cls: 'newinst'
    )
    monkeypatch.setattr(
        'spindoctor.nav_orchestrator.orchestrator.compute_pointing',
        lambda obs, *, offset_px, rotation_fitted: None,
    )
    obs = fake_obs
    model = _FakeStarModel(obs, feature_count=3)
    orch = NavOrchestrator([model], only_techniques=['_FakeStarTechnique'])
    result = orch.navigate(obs)  # type: ignore[arg-type]
    assert result.pointing is None
    assert 'No SPICE camera frame is mapped for instrument newinst' in capsys.readouterr().out


def test_orchestrator_does_not_warn_for_a_simulated_image(
    fake_obs: _FakeObs, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A simulated image records no attitude without raising a warning."""
    monkeypatch.setattr(
        'spindoctor.nav_orchestrator.orchestrator.obs_class_to_inst_name', lambda cls: 'sim'
    )
    monkeypatch.setattr(
        'spindoctor.nav_orchestrator.orchestrator.compute_pointing',
        lambda obs, *, offset_px, rotation_fitted: None,
    )
    obs = fake_obs
    model = _FakeStarModel(obs, feature_count=3)
    orch = NavOrchestrator([model], only_techniques=['_FakeStarTechnique'])
    orch.navigate(obs)  # type: ignore[arg-type]
    out = capsys.readouterr().out
    # The positive control proves the log actually reached stdout, so the
    # negative assertion cannot pass vacuously on an empty capture.
    assert 'ORCHESTRATION' in out
    assert 'No SPICE camera frame is mapped' not in out
