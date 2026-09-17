"""Smoke tests for ``spindoctor.nav_technique.nav_technique_manual``.

The dialog itself is GUI-only and is exercised by hand; these tests
mock ``ManualNavDialog.run_modal`` so the technique logic
(``run_manual_nav`` orchestration + accept / cancel translation) can be
asserted without firing up Qt.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
from tests.spindoctor.nav_technique.conftest import NavFeatureFactory

from spindoctor.annotation import Annotations
from spindoctor.feature.composition import compose_dialog_overlay
from spindoctor.feature.feature import NavFeature, NavReliabilityBreakdown
from spindoctor.feature.feature_type import NavFeatureType
from spindoctor.feature.flags import StarFlags
from spindoctor.feature.geometry import StarGeometry
from spindoctor.nav_model import NavModel
from spindoctor.nav_technique import NavTechniqueManual
from spindoctor.nav_technique.nav_technique_manual import run_manual_nav
from spindoctor.support.cmatrix import PointingSolution
from spindoctor.support.filters import NavFilterKind, NavFilterSpec


class _FakeObsForRunManual:
    """Minimal stand-in for ObsSnapshotInst sufficient for run_manual_nav."""

    def __init__(self) -> None:
        rng = np.random.default_rng(seed=7)
        self.data = rng.standard_normal((48, 48)) + 100.0
        self.extdata = self.data
        self._sensor_mask = np.ones(self.data.shape, bool)
        self.midtime = 0.0

    def extfov_data_sensor_mask(self) -> np.ndarray:
        return self._sensor_mask


class _StubStarModel(NavModel):
    """NavModel that emits a single STAR feature with a tiny template."""

    def __init__(self, obs: Any, *, with_template: bool) -> None:
        super().__init__('stars', obs)
        self._with_template = with_template
        self._obs = obs

    def create_model(self) -> None:
        return None

    def to_features(self, context: Any) -> list[NavFeature]:
        template_img: np.ndarray | None
        template_mask: np.ndarray | None
        if self._with_template:
            # Template arrays are sized to the bbox (6x6 here for the
            # (8, 8, 14, 14) bbox below) so ``compose_template_features``
            # paints a non-empty patch into the ext-FOV composite.
            template_img = np.ones((6, 6), dtype=np.float64)
            template_mask = np.ones((6, 6), dtype=bool)
        else:
            template_img = None
            template_mask = None
        return [
            NavFeature(
                feature_id='star:test:0',
                feature_type=NavFeatureType.STAR,
                source_model='stars',
                geometry=StarGeometry(
                    predicted_vu=(11.0, 11.0),
                    catalog_vu=(11.0, 11.0),
                    bbox_extfov_vu=(8, 8, 14, 14),
                ),
                subject_range_km=1e10,
                position_cov_px=np.eye(2, dtype=np.float64) * 0.25,
                intensity_sigma_rel=0.05,
                preferred_filter=NavFilterSpec(kind=NavFilterKind.NONE),
                reliability=0.9,
                reliability_reasons=NavReliabilityBreakdown(predicted_snr=10.0),
                usable_types=frozenset({NavFeatureType.STAR}),
                flags=StarFlags(),
                template_img=template_img,
                template_mask=template_mask,
            )
        ]

    def to_annotations(self, context: Any) -> Annotations:
        return Annotations()


@pytest.fixture(autouse=True)
def _patch_build_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``build_models_for_obs`` deterministic for these tests.

    ``run_manual_nav`` calls the autoregistered builder; replacing it
    with a single-stub-model factory keeps the test self-contained.
    """
    from spindoctor.nav_technique import nav_technique_manual

    def _stub_builder(obs: Any) -> list[NavModel]:
        return [_StubStarModel(obs, with_template=getattr(obs, 'with_template', True))]

    monkeypatch.setattr(
        nav_technique_manual,
        'build_models_for_obs',
        _stub_builder,
        raising=False,
    )
    # The import inside run_manual_nav happens at call time; patch the
    # source binding too so the local import in the function picks the stub.
    import spindoctor.nav_model as nav_model_pkg

    monkeypatch.setattr(nav_model_pkg, 'build_models_for_obs', _stub_builder, raising=False)


@pytest.fixture(autouse=True)
def _fakes_report_as_simulated(fakes_report_as_simulated: None) -> None:
    """Apply the shared simulated-instrument report to every test in this module."""


def _make_fake_dialog(
    next_modal_return: tuple[bool, tuple[float, float] | None, float | None],
) -> tuple[type, list[Any]]:
    """Build a fresh per-test ``ManualNavDialog`` stub.

    Each call returns a brand-new class plus the per-test instances list so
    no shared state leaks between pytest-xdist workers.  The previous design
    used a single shared `_FakeDialog` whose ClassVar mutable state raced
    when tests ran in parallel.
    """
    captured_return = next_modal_return
    instances: list[Any] = []

    class _FakeDialog:
        def __init__(self, **_kwargs: Any) -> None:
            instances.append(self)

        def run_modal(self) -> tuple[bool, tuple[float, float] | None, float | None]:
            return captured_return

    return _FakeDialog, instances


def test_run_manual_nav_returns_result_on_accept() -> None:
    """When the dialog returns ``accepted=True`` the NavResult is ``ok``."""
    obs = _FakeObsForRunManual()
    obs.with_template = True  # type: ignore[attr-defined]
    fake_dialog, instances = _make_fake_dialog((True, (3.0, -2.0), 0.92))
    with patch('spindoctor.ui.manual_nav_dialog.ManualNavDialog', fake_dialog):
        result = run_manual_nav(obs)  # type: ignore[arg-type]
    assert result is not None
    assert result.status == 'success'
    assert result.offset_px == (3.0, -2.0)
    assert result.confidence_rank == 'high'
    assert len(result.per_technique) == 1
    assert result.per_technique[0].technique_name == NavTechniqueManual.name
    assert len(instances) == 1


def test_run_manual_nav_returns_none_on_cancel() -> None:
    """When the dialog is canceled ``run_manual_nav`` returns ``None``."""
    obs = _FakeObsForRunManual()
    obs.with_template = True  # type: ignore[attr-defined]
    fake_dialog, _instances = _make_fake_dialog((False, None, None))
    with patch('spindoctor.ui.manual_nav_dialog.ManualNavDialog', fake_dialog):
        result = run_manual_nav(obs)  # type: ignore[arg-type]
    assert result is None


def test_run_manual_nav_runs_on_template_less_star_feature() -> None:
    """A STAR feature without a template still renders as a marker rectangle.

    The dialog opens because ``StarGeometry`` is recognized by both
    ``compose_dialog_overlay`` and ``NavTechniqueManual.is_feasible`` —
    the absence of ``template_img`` / ``template_mask`` no longer
    short-circuits feasibility for star-only scenes.
    """
    obs = _FakeObsForRunManual()
    obs.with_template = False  # type: ignore[attr-defined]
    fake_dialog, instances = _make_fake_dialog((True, (1.5, -0.5), 0.7))
    with patch('spindoctor.ui.manual_nav_dialog.ManualNavDialog', fake_dialog):
        result = run_manual_nav(obs)  # type: ignore[arg-type]
    assert result is not None
    assert result.status == 'success'
    assert result.offset_px == (1.5, -0.5)
    assert len(instances) == 1


class _StubEmptyModel(NavModel):
    """NavModel that emits zero features.  Drives the no-feasible path."""

    def __init__(self, obs: Any) -> None:
        super().__init__('stars', obs)

    def create_model(self) -> None:
        return None

    def to_features(self, context: Any) -> list[NavFeature]:
        return []

    def to_annotations(self, context: Any) -> Annotations:
        return Annotations()


def test_run_manual_nav_returns_none_when_no_features(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A scene that emits zero features → dialog is not opened, ``None`` is returned."""
    obs = _FakeObsForRunManual()

    from spindoctor.nav_technique import nav_technique_manual

    def _empty_builder(obs: Any) -> list[NavModel]:
        return [_StubEmptyModel(obs)]

    monkeypatch.setattr(
        nav_technique_manual,
        'build_models_for_obs',
        _empty_builder,
        raising=False,
    )
    import spindoctor.nav_model as nav_model_pkg

    monkeypatch.setattr(nav_model_pkg, 'build_models_for_obs', _empty_builder, raising=False)
    result = run_manual_nav(obs)  # type: ignore[arg-type]
    assert result is None
    captured = capsys.readouterr()
    assert 'Manual navigation skipped' in captured.out


class _StubLimbModel(NavModel):
    """NavModel that emits a single LIMB_ARC feature (polyline only, no template)."""

    def __init__(self, obs: Any) -> None:
        super().__init__('body', obs)
        self._obs = obs

    def create_model(self) -> None:
        return None

    def to_features(self, context: Any) -> list[NavFeature]:
        from spindoctor.feature.flags import LimbArcFlags
        from spindoctor.feature.geometry import LimbPolyline

        verts = np.array([[10.0, 10.0], [12.0, 12.0], [14.0, 14.0]], dtype=np.float64)
        return [
            NavFeature(
                feature_id='limb_arc:test',
                feature_type=NavFeatureType.LIMB_ARC,
                source_model='body',
                geometry=LimbPolyline(
                    vertices_vu=verts,
                    normals_vu=np.zeros_like(verts),
                    sigma_normal_per_vertex_px=np.full(verts.shape[0], 0.5),
                    sigma_tangent_per_vertex_px=np.full(verts.shape[0], 0.5),
                    bbox_extfov_vu=(8, 8, 16, 16),
                ),
                subject_range_km=1e6,
                position_cov_px=None,
                intensity_sigma_rel=0.0,
                preferred_filter=NavFilterSpec(kind=NavFilterKind.NONE),
                reliability=0.9,
                reliability_reasons=NavReliabilityBreakdown(visible_arc_fraction=1.0),
                usable_types=frozenset({NavFeatureType.LIMB_ARC}),
                flags=LimbArcFlags(body_name='X', visible_arc_fraction=1.0),
            )
        ]

    def to_annotations(self, context: Any) -> Annotations:
        return Annotations()


def test_run_manual_nav_runs_for_polyline_only_scene(monkeypatch: pytest.MonkeyPatch) -> None:
    """A scene with only LIMB_ARC features (no full-disc templates) is still
    feasible — the dialog gets a polyline-rasterized overlay."""
    obs = _FakeObsForRunManual()
    import spindoctor.nav_model as nav_model_pkg
    from spindoctor.nav_technique import nav_technique_manual

    def _limb_only_builder(obs_arg: Any) -> list[NavModel]:
        return [_StubLimbModel(obs_arg)]

    monkeypatch.setattr(
        nav_technique_manual, 'build_models_for_obs', _limb_only_builder, raising=False
    )
    monkeypatch.setattr(nav_model_pkg, 'build_models_for_obs', _limb_only_builder, raising=False)

    fake_dialog, instances = _make_fake_dialog((True, (1.0, 2.0), 0.5))
    with patch('spindoctor.ui.manual_nav_dialog.ManualNavDialog', fake_dialog):
        result = run_manual_nav(obs)  # type: ignore[arg-type]
    assert result is not None
    assert result.status == 'success'
    assert result.offset_px == (1.0, 2.0)
    assert len(instances) == 1


# ---------------------------------------------------------------------------
# Titan-only scenes
# ---------------------------------------------------------------------------


def _titan_feature(make_titan_feature: NavFeatureFactory) -> NavFeature:
    """Build the single haze feature a Titan-only frame carries."""
    return make_titan_feature(predicted_center_vu=(24.0, 24.0), r_solid_px=10.0, r_env_px=12.0)


def test_titan_only_feature_set_is_manual_nav_feasible(
    make_titan_feature: NavFeatureFactory,
) -> None:
    """A hazy body alone is enough to open the dialog.

    Manual navigation is the curation fallback when the autonomous
    technique fails, so a Titan-only frame -- which carries neither a
    template nor a polyline -- must still be manually navigable.
    """
    feature = _titan_feature(make_titan_feature)
    assert NavTechniqueManual().is_feasible([feature]).feasible is True


def test_titan_only_feature_set_counts_one_renderable_feature(
    make_titan_feature: NavFeatureFactory,
) -> None:
    """The haze feature is counted as the feature the dialog will render."""
    feature = _titan_feature(make_titan_feature)
    assert NavTechniqueManual().is_feasible([feature]).consumed_feature_count == 1


def test_titan_only_feature_set_composes_a_non_empty_overlay(
    make_titan_feature: NavFeatureFactory,
) -> None:
    """The dialog's draggable overlay actually paints the haze envelope.

    Feasibility counts geometry kinds; this is the composed bitmap the
    operator drags, which ``run_manual_nav`` separately requires to be
    non-empty before opening the dialog.
    """
    feature = _titan_feature(make_titan_feature)
    _image, mask = compose_dialog_overlay([feature], (48, 48))
    assert bool(mask.any()) is True


def test_run_manual_nav_stamps_the_corrected_pointing(
    monkeypatch: pytest.MonkeyPatch, sentinel_pointing: PointingSolution
) -> None:
    """An operator-picked offset carries the same recorded attitude as an autonomous one.

    Operator-ratified offsets are the highest-quality pointing in the corpus,
    so they must not be the one subset a generated C-kernel omits.  The SPICE
    lookups behind the real computation need furnished kernels, so the wiring
    is exercised with a stand-in solution.
    """
    pointing = sentinel_pointing
    monkeypatch.setattr(
        'spindoctor.nav_orchestrator.orchestrator.compute_pointing',
        lambda obs, *, offset_px, rotation_fitted: pointing,
    )
    obs = _FakeObsForRunManual()
    obs.with_template = True  # type: ignore[attr-defined]
    fake_dialog, _instances = _make_fake_dialog((True, (3.0, -2.0), 0.92))
    with patch('spindoctor.ui.manual_nav_dialog.ManualNavDialog', fake_dialog):
        result = run_manual_nav(obs)  # type: ignore[arg-type]
    assert result is not None
    assert result.pointing is pointing


def test_run_manual_nav_passes_the_operator_offset_to_the_pointing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The attitude is computed from the offset the operator actually picked."""
    seen: dict[str, Any] = {}

    def _record(obs: Any, *, offset_px: Any, rotation_fitted: bool) -> None:
        seen['offset_px'] = offset_px
        seen['rotation_fitted'] = rotation_fitted
        return None

    monkeypatch.setattr('spindoctor.nav_orchestrator.orchestrator.compute_pointing', _record)
    obs = _FakeObsForRunManual()
    obs.with_template = True  # type: ignore[attr-defined]
    fake_dialog, _instances = _make_fake_dialog((True, (3.0, -2.0), 0.92))
    with patch('spindoctor.ui.manual_nav_dialog.ManualNavDialog', fake_dialog):
        run_manual_nav(obs)  # type: ignore[arg-type]
    assert seen['offset_px'] == (3.0, -2.0)
    assert seen['rotation_fitted'] is False
