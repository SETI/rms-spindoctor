"""Tests for ``spindoctor.nav_model.nav_model_titan.NavModelTitan``.

The model is built whenever Titan is inside the extended FOV and always
emits exactly one ``TITAN_LIMB`` feature; frame quality is carried by that
feature's reliability rather than by a decline.  The tests split along the
same seam the code does: the reliability and feature-payload rules are
exercised on directly-constructed :class:`TitanGeometryInputs` with no
observation at all, while the contaminant-mask assembly runs against an
analytic backplane stand-in patched over the module's ``oops`` names.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from tests.shims import BodyBackplaneData, FakeBackplane, FakeObs

import spindoctor.nav_model.titan_geometry as geometry_module
from spindoctor.config import Config
from spindoctor.feature.feature_type import NavFeatureType
from spindoctor.feature.geometry import TitanHazeGeometry
from spindoctor.nav_model.nav_model_body import NavModelBody
from spindoctor.nav_model.nav_model_titan import (
    NavModelTitan,
    build_titan_feature,
    haze_overlay,
    titan_haze_reliability,
)
from spindoctor.nav_model.titan_geometry import TitanGeometryInputs

_EXTFOV_SHAPE = (200, 200)
_WINDOW_PX = 10.0
_TITAN_RADIUS_KM = 2575.0


# ---------------------------------------------------------------------------
# Geometry-inputs factory (no observation required)
# ---------------------------------------------------------------------------


def _inputs(
    *,
    r_env_px: float = 40.0,
    occluded_fraction: float = 0.0,
    center_vu: tuple[float, float] = (100.0, 100.0),
    phase_deg: float = 30.0,
    theta_rad: float = 0.5,
    axis_degenerate: bool = False,
    filters: tuple[str, ...] = ('CL1', 'CL2'),
    extfov_shape_vu: tuple[int, int] = _EXTFOV_SHAPE,
    extfov_margin_vu: tuple[float, float] = (_WINDOW_PX, _WINDOW_PX),
) -> TitanGeometryInputs:
    """Build a ``TitanGeometryInputs`` with everything but the varied field fixed."""
    return TitanGeometryInputs(
        predicted_center_vu=center_vu,
        r_solid_px=0.9 * r_env_px,
        r_env_px=r_env_px,
        km_per_px=_TITAN_RADIUS_KM / max(0.9 * r_env_px, 1.0e-9),
        phase_deg=phase_deg,
        theta_rad=theta_rad,
        axis_degenerate=axis_degenerate,
        occluded_fraction=occluded_fraction,
        contaminant_mask=None,
        extfov_shape_vu=extfov_shape_vu,
        window_px=max(extfov_margin_vu),
        extfov_margin_vu=extfov_margin_vu,
        bbox_extfov_vu=(60, 60, 141, 141),
        subject_range_km=1.2e6,
        filters=filters,
    )


@pytest.fixture
def config() -> Config:
    """Project configuration carrying the shipped ``titan`` defaults."""
    return Config()


# ---------------------------------------------------------------------------
# Reliability
# ---------------------------------------------------------------------------


def test_reliability_follows_the_diameter_sigmoid(config: Config) -> None:
    """A well-resolved unoccluded envelope scores its sigmoid value.

    With the shipped midpoint of 52 px and scale of 14 px, a 104 px
    envelope sits 52 px above the midpoint.
    """
    reliability, _ = titan_haze_reliability(_inputs(r_env_px=52.0), config=config)
    assert reliability == pytest.approx(1.0 / (1.0 + math.exp(-(104.0 - 52.0) / 14.0)))


def test_reliability_scales_down_with_occlusion(config: Config) -> None:
    """Partial occlusion multiplies the size term by the visible fraction."""
    clean, _ = titan_haze_reliability(_inputs(r_env_px=52.0), config=config)
    occluded, _ = titan_haze_reliability(
        _inputs(r_env_px=52.0, occluded_fraction=0.05), config=config
    )
    assert occluded == pytest.approx(clean * 0.95)


@pytest.mark.parametrize(
    ('occluded_fraction', 'crossing_diameter_px'),
    [(0.0, 40.138), (0.10, 42.296)],
    ids=['unoccluded', 'max_permitted_occlusion'],
)
def test_reliability_crosses_the_type_gate_near_the_diameter_floor(
    config: Config, occluded_fraction: float, crossing_diameter_px: float
) -> None:
    """The sigmoid crosses the 0.30 TITAN_LIMB gate just above the hard floor.

    The two gates are deliberately aligned: an envelope barely above the
    40 px emission floor is also barely above the type threshold.  The
    crossing moves up with occlusion, and at the maximum occlusion the
    hard-zero condition permits it is still only 2.2 px above the floor, so
    the band of emit-then-gate frames stays narrow.
    """
    reliability, _ = titan_haze_reliability(
        _inputs(r_env_px=0.5 * crossing_diameter_px, occluded_fraction=occluded_fraction),
        config=config,
    )
    assert reliability == pytest.approx(0.30, abs=0.005)


def test_reliability_reports_the_envelope_diameter(config: Config) -> None:
    """The breakdown names the diameter that drove the score."""
    _, breakdown = titan_haze_reliability(_inputs(r_env_px=52.0), config=config)
    assert breakdown.titan_envelope_diameter_px == pytest.approx(104.0)


def test_reliability_reports_the_occluded_fraction(config: Config) -> None:
    """The breakdown names the occlusion that drove the score."""
    _, breakdown = titan_haze_reliability(
        _inputs(r_env_px=52.0, occluded_fraction=0.05), config=config
    )
    assert breakdown.titan_occluded_fraction == pytest.approx(0.05)


def test_hard_zero_when_the_envelope_cannot_be_framed(config: Config) -> None:
    """An envelope whose search window leaves the frame scores exactly zero.

    Full visibility is a property of the TRUE position, so the disc is
    dilated by the search window before the containment test: a body 45 px
    from the edge with a 40 px envelope and a 10 px window fails.
    """
    reliability, _ = titan_haze_reliability(
        _inputs(r_env_px=52.0, center_vu=(52.0, 100.0)), config=config
    )
    assert reliability == 0.0


def test_framed_envelope_is_not_hard_zeroed(config: Config) -> None:
    """The same envelope one pixel further inside the frame scores normally."""
    reliability, _ = titan_haze_reliability(
        _inputs(r_env_px=52.0, center_vu=(63.0, 100.0)), config=config
    )
    assert reliability > 0.0


def test_visibility_dilation_uses_each_axis_own_margin(config: Config) -> None:
    """An envelope clearing the detector on both axes scores normally.

    The margins here are deliberately unequal (5 rows against 40 columns,
    the shape a Cassini NAC has).  The body centre sits at row 60 of the
    EXTENDED frame, so it is 55 px inside the detector's top edge, and its
    52 px envelope clears that edge by 3 px.  Dilating the row axis by the
    40 px column margin instead would demand 92 px and fail the frame for
    no physical reason.
    """
    reliability, _ = titan_haze_reliability(
        _inputs(r_env_px=52.0, center_vu=(60.0, 100.0), extfov_margin_vu=(5.0, 40.0)),
        config=config,
    )
    assert reliability > 0.0


def test_visibility_still_hard_zeroes_when_an_axis_is_clipped(config: Config) -> None:
    """The same body four pixels nearer the top edge fails the row test.

    At row 56 of the extended frame the centre is 51 px inside the
    detector's top edge, one pixel less than the envelope radius, so the
    envelope genuinely crosses it.
    """
    reliability, _ = titan_haze_reliability(
        _inputs(r_env_px=52.0, center_vu=(56.0, 100.0), extfov_margin_vu=(5.0, 40.0)),
        config=config,
    )
    assert reliability == 0.0


def test_hard_zero_when_occlusion_exceeds_the_maximum(config: Config) -> None:
    """Occlusion past ``max_occluded_fraction`` scores exactly zero."""
    reliability, _ = titan_haze_reliability(
        _inputs(r_env_px=52.0, occluded_fraction=0.11), config=config
    )
    assert reliability == 0.0


def test_hard_zero_when_the_envelope_is_too_small(config: Config) -> None:
    """An envelope below ``min_envelope_diameter_px`` scores exactly zero."""
    reliability, _ = titan_haze_reliability(_inputs(r_env_px=19.0), config=config)
    assert reliability == 0.0


def test_hard_zero_breakdown_still_reports_its_cause(config: Config) -> None:
    """A hard-zeroed feature still names the quantity that zeroed it."""
    _, breakdown = titan_haze_reliability(_inputs(r_env_px=19.0), config=config)
    assert breakdown.titan_envelope_diameter_px == pytest.approx(38.0)


def test_zero_radius_geometry_is_hard_zeroed(config: Config) -> None:
    """The degenerate-geometry default (zero radii) scores exactly zero."""
    degenerate = _inputs(r_env_px=0.0)
    reliability, _ = titan_haze_reliability(degenerate, config=config)
    assert reliability == 0.0


# ---------------------------------------------------------------------------
# Feature payload
# ---------------------------------------------------------------------------


def test_feature_uses_the_documented_identity(config: Config) -> None:
    """The feature id follows the ``<type_lc>:<scope>`` convention."""
    feature = build_titan_feature(_inputs(), source_model='titan:TITAN', config=config)
    assert feature.feature_id == 'titan_limb:TITAN'


def test_feature_declares_its_usable_type(config: Config) -> None:
    """The feature is consumable only as TITAN_LIMB."""
    feature = build_titan_feature(_inputs(), source_model='titan:TITAN', config=config)
    assert feature.usable_types == frozenset({NavFeatureType.TITAN_LIMB})


def test_feature_records_its_source_model(config: Config) -> None:
    """The emitting model's name travels with the feature."""
    feature = build_titan_feature(_inputs(), source_model='titan:TITAN', config=config)
    assert feature.source_model == 'titan:TITAN'


def test_feature_attributes_itself_to_the_body(config: Config) -> None:
    """Body attribution reads off the flags dataclass, not the feature id."""
    feature = build_titan_feature(_inputs(), source_model='titan:TITAN', config=config)
    assert feature.body_name == 'TITAN'


def test_feature_geometry_carries_the_symmetry_axis(config: Config) -> None:
    """The geometry payload hands the technique the axis the model derived."""
    feature = build_titan_feature(
        _inputs(theta_rad=0.75), source_model='titan:TITAN', config=config
    )
    assert isinstance(feature.geometry, TitanHazeGeometry)
    assert feature.geometry.sun_angle_rad == pytest.approx(0.75)


def test_surface_window_filter_flag_fires_for_a_methane_window(config: Config) -> None:
    """A filter that sees through to the surface sets the flag."""
    feature = build_titan_feature(
        _inputs(filters=('CB3', 'CL2')), source_model='titan:TITAN', config=config
    )
    assert feature.flags.surface_window_filter is True  # type: ignore[union-attr]


def test_surface_window_filter_flag_is_clear_for_a_clear_filter(config: Config) -> None:
    """A filter that does not reach the surface leaves the flag clear."""
    feature = build_titan_feature(_inputs(), source_model='titan:TITAN', config=config)
    assert feature.flags.surface_window_filter is False  # type: ignore[union-attr]


def test_high_phase_flag_fires_above_the_threshold(config: Config) -> None:
    """A crescent-phase frame is flagged for its thin sunward arc."""
    feature = build_titan_feature(
        _inputs(phase_deg=160.0), source_model='titan:TITAN', config=config
    )
    assert feature.flags.high_phase is True  # type: ignore[union-attr]


def test_high_phase_flag_is_clear_at_moderate_phase(config: Config) -> None:
    """A moderate-phase frame is not flagged."""
    feature = build_titan_feature(_inputs(), source_model='titan:TITAN', config=config)
    assert feature.flags.high_phase is False  # type: ignore[union-attr]


def test_high_phase_threshold_is_configurable(config: Config, tmp_path: Path) -> None:
    """Lowering titan.navigation.high_phase_deg moves the flag's threshold."""
    override = tmp_path / 'high_phase_override.yaml'
    override.write_text('titan:\n  navigation:\n    high_phase_deg: 90.0\n')
    config.update_config(override)
    feature = build_titan_feature(
        _inputs(phase_deg=120.0), source_model='titan:TITAN', config=config
    )
    assert feature.flags.high_phase is True  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------


def _titan_only_config(tmp_path: Path) -> Config:
    """Configuration whose Saturn satellite list contains Titan alone."""
    override = tmp_path / 'satellites_override.yaml'
    override.write_text('satellites:\n  SATURN:\n    - TITAN\n')
    config = Config()
    config.update_config(override)
    return config


def _inventory_entry(
    center_vu: tuple[float, float], half_size_px: float, range_km: float
) -> dict[str, Any]:
    """Build an inventory record for a body of the given centre and size."""
    return {
        'u_min_unclipped': center_vu[1] - half_size_px,
        'u_max_unclipped': center_vu[1] + half_size_px,
        'v_min_unclipped': center_vu[0] - half_size_px,
        'v_max_unclipped': center_vu[0] + half_size_px,
        'u_pixel_size': 2.0 * half_size_px,
        'v_pixel_size': 2.0 * half_size_px,
        'range': range_km,
        'center_uv': np.array([center_vu[1], center_vu[0]]),
    }


def test_titan_model_built_for_titan_in_fov(tmp_path: Path) -> None:
    """Titan in the FOV builds one active NavModelTitan."""
    config = _titan_only_config(tmp_path)
    obs = FakeObs(
        data=np.zeros((120, 120)),
        closest_planet='SATURN',
        inventory_records={'TITAN': _inventory_entry((60.0, 60.0), 30.0, 1.2e6)},
    )
    instances = NavModelTitan.instances_for_obs(cast(Any, obs), config=config)
    assert [m.name for m in instances] == ['titan:TITAN']


def test_body_model_excludes_titan(tmp_path: Path) -> None:
    """The shape-based body model builds nothing for Titan."""
    config = _titan_only_config(tmp_path)
    obs = FakeObs(
        data=np.zeros((120, 120)),
        closest_planet='SATURN',
        inventory_records={'TITAN': _inventory_entry((60.0, 60.0), 30.0, 1.2e6)},
    )
    instances = NavModelBody.instances_for_obs(cast(Any, obs), config=config)
    assert all('TITAN' not in m.name.upper() for m in instances)


def test_simulated_obs_builds_no_titan_model(tmp_path: Path) -> None:
    """Simulated observations drive model selection from operator parameters."""

    class _SimObs:
        is_simulated = True

    config = _titan_only_config(tmp_path)
    assert NavModelTitan.instances_for_obs(cast(Any, _SimObs()), config=config) == []


# ---------------------------------------------------------------------------
# Analytic geometry scene
# ---------------------------------------------------------------------------


class _Pair:
    """Stand-in for the ``polymath.Pair`` a meshgrid exposes as ``uv``."""

    def __init__(self, vals: np.ndarray) -> None:
        self.vals = vals


class _BoxMeshgrid:
    """Meshgrid stand-in materializing the ``(u, v)`` of every box sample.

    Mirrors ``oops.Meshgrid.for_fov`` closely enough for the model's
    coordinate arithmetic: one sample per pixel centre over the requested
    nominal-frame box, indexed ``(v, u)``.
    """

    def __init__(
        self, origin: tuple[float, float], limit: tuple[float, float], undersample: int = 1
    ) -> None:
        step = float(max(1, undersample))
        us = np.arange(origin[0], limit[0] + 0.5, step, dtype=np.float64)
        vs = np.arange(origin[1], limit[1] + 0.5, step, dtype=np.float64)
        uu, vv = np.meshgrid(us, vs, indexing='xy')
        self.us = us
        self.vs = vs
        self.uu = uu
        self.vv = vv
        self.uv = _Pair(np.stack([uu, vv], axis=-1))

    @classmethod
    def for_fov(
        cls,
        fov: Any,
        *,
        origin: tuple[float, float],
        limit: tuple[float, float],
        oversample: tuple[int, int] = (1, 1),
        undersample: int = 1,
        swap: bool = False,
    ) -> _BoxMeshgrid:
        """Mirror the ``oops.Meshgrid.for_fov`` factory signature."""
        del fov, oversample, swap
        return cls(origin, limit, undersample)


class _MaskedArray:
    """Minimal stand-in for the masked array a backplane Scalar exposes."""

    def __init__(self, values: np.ndarray, mask: np.ndarray) -> None:
        self._values = values
        self._mask = mask

    def filled(self, fill: Any) -> np.ndarray:
        """Return the values with masked entries replaced by ``fill``."""
        return np.where(self._mask, fill, self._values)


class _Scalar:
    """Minimal stand-in for the ``polymath.Scalar`` a backplane returns."""

    def __init__(self, values: np.ndarray, mask: np.ndarray) -> None:
        self.vals = values
        self.mvals = _MaskedArray(values, mask)
        self._mask = mask

    def expand_mask(self) -> _Scalar:
        """Return self; the shim always carries a full-shape mask."""
        return self

    @property
    def mask(self) -> np.ndarray:
        """Boolean array marking invalid samples."""
        return self._mask


class _SceneBackplane:
    """Analytic backplane over a box meshgrid: one lit body, one occluder, rings.

    ``incidence_angle`` puts its minimum exactly at the planted sub-solar
    pixel, so the model's minimum-incidence axis rule has a known answer.
    """

    sub_solar_offset_vu: tuple[float, float] = (0.0, 0.0)
    titan_center_vu: tuple[float, float] = (60.5, 60.5)
    titan_radius_px: float = 26.0
    occluder_center_vu: tuple[float, float] | None = None
    occluder_radius_px: float = 10.0
    ring_radius_at_u: float | None = None
    ring_distance_km: float = 1.0e9

    def __init__(self, obs: Any, *, meshgrid: _BoxMeshgrid) -> None:
        del obs
        self._mg = meshgrid

    def incidence_angle(self, body_name: str) -> _Scalar:
        """Return incidence rising with distance from the sub-solar pixel.

        Every position here is a field-of-view coordinate, the same frame
        the meshgrid samples and the same frame the model builds its
        predicted center in.  No half-pixel adjustment is applied: baking
        one in would hide a frame mismatch between the two ends of the
        angle the model computes.
        """
        del body_name
        centre_v, centre_u = self.titan_center_vu
        off_body = np.hypot(self._mg.vv - centre_v, self._mg.uu - centre_u) > self.titan_radius_px
        sun_v = centre_v + self.sub_solar_offset_vu[0]
        sun_u = centre_u + self.sub_solar_offset_vu[1]
        values = np.hypot(self._mg.vv - sun_v, self._mg.uu - sun_u)
        return _Scalar(values, off_body)

    def where_in_front(self, sibling_name: str, body_name: str) -> _Scalar:
        """Return the planted occluder silhouette, or nothing when unplanted."""
        del sibling_name, body_name
        if self.occluder_center_vu is None:
            hidden = np.zeros(self._mg.vv.shape, dtype=bool)
        else:
            hidden = (
                np.hypot(
                    self._mg.vv - self.occluder_center_vu[0],
                    self._mg.uu - self.occluder_center_vu[1],
                )
                <= self.occluder_radius_px
            )
        return _Scalar(hidden, np.zeros(hidden.shape, dtype=bool))

    def ring_radius(self, ring_target: str) -> _Scalar:
        """Return a ring-plane radius ramp, all-masked when rings are unplanted."""
        del ring_target
        shape = self._mg.vv.shape
        if self.ring_radius_at_u is None:
            return _Scalar(np.zeros(shape), np.ones(shape, dtype=bool))
        values = np.full(shape, self.ring_radius_at_u, dtype=np.float64)
        return _Scalar(values, np.zeros(shape, dtype=bool))

    def distance(self, ring_target: str, direction: str = 'dep') -> _Scalar:
        """Return the constant ring-intercept distance."""
        del ring_target, direction
        shape = self._mg.vv.shape
        return _Scalar(np.full(shape, self.ring_distance_km), np.zeros(shape, dtype=bool))


def _scene_obs(*, titan_entry: dict[str, Any], extra: dict[str, dict[str, Any]]) -> FakeObs:
    """Build the FakeObs the contaminant-mask tests share."""
    records = {'TITAN': titan_entry}
    records.update(extra)
    body = BodyBackplaneData(
        body_mask=np.ones((120, 120), dtype=bool),
        incidence_rad=np.zeros((120, 120)),
        default_resolution_km_px=_TITAN_RADIUS_KM / 26.0,
        center_phase_rad=math.radians(30.0),
    )
    return FakeObs(
        data=np.zeros((120, 120)),
        extfov_margin_vu=(10, 10),
        closest_planet='SATURN',
        ext_bp=cast(Any, FakeBackplane(per_body={'TITAN': body})),
        inventory_records=records,
    )


@pytest.fixture
def scene(monkeypatch: pytest.MonkeyPatch) -> type[_SceneBackplane]:
    """Patch the module's oops names onto the analytic scene and return it."""

    class _Scene(_SceneBackplane):
        pass

    monkeypatch.setattr(geometry_module, 'Meshgrid', _BoxMeshgrid)
    monkeypatch.setattr(geometry_module, 'Backplane', _Scene)
    monkeypatch.setattr(geometry_module, '_body_radius_km', lambda name: _TITAN_RADIUS_KM)
    monkeypatch.setattr(geometry_module, 'stars_in_extfov', lambda *a, **k: [])
    return _Scene


def _geometry(obs: FakeObs, config: Config) -> TitanGeometryInputs:
    """Run the model's observation-side geometry extraction."""
    model = NavModelTitan.instances_for_obs(cast(Any, obs), config=config)[0]
    model.create_model()
    return cast(NavModelTitan, model).geometry_inputs


# ---------------------------------------------------------------------------
# Nothing here absorbs an exception
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'raised',
    [
        MemoryError,
        KeyError,
        ZeroDivisionError,
        NameError,
        AttributeError,
        TypeError,
        ValueError,
        LookupError,
        OSError,
        NotImplementedError,
    ],
)
@pytest.mark.parametrize('patched', ['_restricted_backplane', '_body_scale', 'stars_in_extfov'])
def test_no_stage_absorbs_an_exception(
    scene: type[_SceneBackplane],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    patched: str,
    raised: type[Exception],
) -> None:
    """Whatever a stage raises, and wherever it raises it, the navigation fails.

    No type is evidence of anything: oops declines with ValueError and
    LookupError, and so does a defect inside oops, inside numpy, or here. A
    stage that carried on would still produce an offset, and the document
    written for it would say a navigation ran and concluded something -- which
    no error filter selects and no later pass corrects.

    Parameters:
        patched: Name of the module-level function made to raise.
        raised: The exception it raises.
    """
    scene.sub_solar_offset_vu = (10.0, 0.0)
    scene.occluder_center_vu = None
    scene.ring_radius_at_u = None

    def _raises(*args: Any, **kwargs: Any) -> Any:
        raise raised('the stage could not be evaluated')

    monkeypatch.setattr(geometry_module, patched, _raises)
    obs = _scene_obs(titan_entry=_inventory_entry((60.5, 60.5), 26.0, 1.2e6), extra={})
    with pytest.raises(raised, match='the stage could not be evaluated'):
        _geometry(obs, _titan_only_config(tmp_path))


def test_symmetry_axis_points_at_the_sub_solar_pixel(
    scene: type[_SceneBackplane], tmp_path: Path
) -> None:
    """The minimum-incidence pixel sets the axis angle."""
    scene.sub_solar_offset_vu = (10.0, 0.0)
    scene.occluder_center_vu = None
    scene.ring_radius_at_u = None
    obs = _scene_obs(titan_entry=_inventory_entry((60.5, 60.5), 26.0, 1.2e6), extra={})
    geometry = _geometry(obs, _titan_only_config(tmp_path))
    assert geometry.theta_rad == pytest.approx(math.pi / 2.0, abs=1.0e-6)


def test_near_zero_phase_axis_is_degenerate(scene: type[_SceneBackplane], tmp_path: Path) -> None:
    """A sub-solar point at the disc centre marks the axis degenerate."""
    scene.sub_solar_offset_vu = (0.0, 0.0)
    scene.occluder_center_vu = None
    scene.ring_radius_at_u = None
    obs = _scene_obs(titan_entry=_inventory_entry((60.5, 60.5), 26.0, 1.2e6), extra={})
    geometry = _geometry(obs, _titan_only_config(tmp_path))
    assert geometry.axis_degenerate is True


def test_sibling_moon_enters_the_contaminant_mask(
    scene: type[_SceneBackplane], tmp_path: Path
) -> None:
    """A moon beside Titan is masked so its sliver cannot pollute the fit."""
    scene.sub_solar_offset_vu = (10.0, 0.0)
    scene.occluder_center_vu = None
    scene.ring_radius_at_u = None
    obs = _scene_obs(
        titan_entry=_inventory_entry((60.5, 60.5), 26.0, 1.2e6),
        extra={'RHEA': _inventory_entry((60.0, 100.0), 4.0, 2.0e6)},
    )
    config = _titan_only_config(tmp_path)
    config.update_config(_satellites_override(tmp_path, ('TITAN', 'RHEA')))
    geometry = _geometry(obs, config)
    assert geometry.contaminant_mask is not None
    assert bool(geometry.contaminant_mask[70, 110]) is True


def test_farther_sibling_does_not_count_as_occlusion(
    scene: type[_SceneBackplane], tmp_path: Path
) -> None:
    """A moon behind Titan hides nothing, so it stays out of the fraction."""
    scene.sub_solar_offset_vu = (10.0, 0.0)
    scene.occluder_center_vu = None
    scene.ring_radius_at_u = None
    obs = _scene_obs(
        titan_entry=_inventory_entry((60.5, 60.5), 26.0, 1.2e6),
        extra={'RHEA': _inventory_entry((60.0, 100.0), 4.0, 2.0e6)},
    )
    config = _titan_only_config(tmp_path)
    config.update_config(_satellites_override(tmp_path, ('TITAN', 'RHEA')))
    geometry = _geometry(obs, config)
    assert geometry.occluded_fraction == 0.0


def test_nearer_moon_counts_toward_the_occluded_fraction(
    scene: type[_SceneBackplane], tmp_path: Path
) -> None:
    """A nearer moon covering the disc raises the occluded fraction."""
    scene.sub_solar_offset_vu = (10.0, 0.0)
    scene.occluder_center_vu = (60.5, 60.5)
    scene.occluder_radius_px = 8.0
    scene.ring_radius_at_u = None
    obs = _scene_obs(
        titan_entry=_inventory_entry((60.5, 60.5), 26.0, 1.2e6),
        extra={'RHEA': _inventory_entry((60.0, 60.0), 8.0, 5.0e5)},
    )
    config = _titan_only_config(tmp_path)
    config.update_config(_satellites_override(tmp_path, ('TITAN', 'RHEA')))
    geometry = _geometry(obs, config)
    assert geometry.occluded_fraction > 0.0


def test_nearer_moon_also_enters_the_contaminant_mask(
    scene: type[_SceneBackplane], tmp_path: Path
) -> None:
    """The occluding moon is masked as well as counted."""
    scene.sub_solar_offset_vu = (10.0, 0.0)
    scene.occluder_center_vu = (60.5, 60.5)
    scene.occluder_radius_px = 8.0
    scene.ring_radius_at_u = None
    obs = _scene_obs(
        titan_entry=_inventory_entry((60.5, 60.5), 26.0, 1.2e6),
        extra={'RHEA': _inventory_entry((60.0, 60.0), 8.0, 5.0e5)},
    )
    config = _titan_only_config(tmp_path)
    config.update_config(_satellites_override(tmp_path, ('TITAN', 'RHEA')))
    geometry = _geometry(obs, config)
    assert geometry.contaminant_mask is not None
    assert bool(geometry.contaminant_mask[70, 70]) is True


def test_rings_in_front_count_toward_the_occluded_fraction(
    scene: type[_SceneBackplane], tmp_path: Path
) -> None:
    """Main-ring material nearer than the body occludes it."""
    scene.sub_solar_offset_vu = (10.0, 0.0)
    scene.occluder_center_vu = None
    scene.ring_radius_at_u = 100000.0
    scene.ring_distance_km = 5.0e5
    obs = _scene_obs(titan_entry=_inventory_entry((60.5, 60.5), 26.0, 1.2e6), extra={})
    geometry = _geometry(obs, _titan_only_config(tmp_path))
    assert geometry.occluded_fraction > 0.5


def test_rings_behind_the_body_do_not_occlude(scene: type[_SceneBackplane], tmp_path: Path) -> None:
    """Ring material farther than the body hides nothing."""
    scene.sub_solar_offset_vu = (10.0, 0.0)
    scene.occluder_center_vu = None
    scene.ring_radius_at_u = 100000.0
    scene.ring_distance_km = 5.0e6
    obs = _scene_obs(titan_entry=_inventory_entry((60.5, 60.5), 26.0, 1.2e6), extra={})
    geometry = _geometry(obs, _titan_only_config(tmp_path))
    assert geometry.occluded_fraction == 0.0


def test_ring_material_outside_the_annulus_does_not_occlude(
    scene: type[_SceneBackplane], tmp_path: Path
) -> None:
    """Ring-plane intercepts outside the configured radii are not opaque."""
    scene.sub_solar_offset_vu = (10.0, 0.0)
    scene.occluder_center_vu = None
    scene.ring_radius_at_u = 500000.0
    scene.ring_distance_km = 5.0e5
    obs = _scene_obs(titan_entry=_inventory_entry((60.5, 60.5), 26.0, 1.2e6), extra={})
    geometry = _geometry(obs, _titan_only_config(tmp_path))
    assert geometry.occluded_fraction == 0.0


def _satellites_override(tmp_path: Path, names: tuple[str, ...]) -> Path:
    """Write a satellite-list override naming exactly ``names`` for Saturn."""
    path = tmp_path / f'satellites_{"_".join(names)}.yaml'
    body = '\n'.join(f'    - {name}' for name in names)
    path.write_text(f'satellites:\n  SATURN:\n{body}\n')
    return path


# ---------------------------------------------------------------------------
# Bright-star masking
# ---------------------------------------------------------------------------


class _FakeStar:
    """Catalog-star stand-in exposing the predicted pixel position and magnitude."""

    def __init__(self, v: float, u: float, vmag: float) -> None:
        self.v = v
        self.u = u
        self.vmag = vmag


def _install_star_catalog(monkeypatch: pytest.MonkeyPatch, stars: list[_FakeStar]) -> None:
    """Serve ``stars`` from the magnitude window each catalog query asks for."""

    def _fake_query(
        obs: Any,
        config: Any,
        *,
        catalog_name: str,
        mag_min: float,
        mag_max: float,
        radec_movement: Any = None,
    ) -> list[_FakeStar]:
        del obs, config, catalog_name, radec_movement
        return [s for s in stars if mag_min <= s.vmag < mag_max]

    monkeypatch.setattr(geometry_module, 'stars_in_extfov', _fake_query)


def test_bright_star_contributes_a_masked_disc(
    scene: type[_SceneBackplane], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A star brighter than the mask limit paints a disc at its prediction."""
    scene.sub_solar_offset_vu = (10.0, 0.0)
    scene.occluder_center_vu = None
    scene.ring_radius_at_u = None
    _install_star_catalog(monkeypatch, [_FakeStar(v=60.0, u=95.0, vmag=5.0)])
    obs = _scene_obs(titan_entry=_inventory_entry((60.5, 60.5), 26.0, 1.2e6), extra={})
    geometry = _geometry(obs, _titan_only_config(tmp_path))
    assert geometry.contaminant_mask is not None
    assert bool(geometry.contaminant_mask[70, 105]) is True


def test_faint_star_is_left_unmasked(
    scene: type[_SceneBackplane], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A star fainter than ``star_mask_vmag_limit`` is deliberately not masked."""
    scene.sub_solar_offset_vu = (10.0, 0.0)
    scene.occluder_center_vu = None
    scene.ring_radius_at_u = None
    _install_star_catalog(monkeypatch, [_FakeStar(v=60.0, u=95.0, vmag=9.0)])
    obs = _scene_obs(titan_entry=_inventory_entry((60.5, 60.5), 26.0, 1.2e6), extra={})
    geometry = _geometry(obs, _titan_only_config(tmp_path))
    assert geometry.contaminant_mask is None


def _record_star_queries(
    scene: type[_SceneBackplane], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> list[tuple[str, float, float]]:
    """Run the geometry and return the ``(catalog, mag_min, mag_max)`` queries it made."""
    queried: list[tuple[str, float, float]] = []

    def _record(
        obs: Any,
        config: Any,
        *,
        catalog_name: str,
        mag_min: float,
        mag_max: float,
        radec_movement: Any = None,
    ) -> list[_FakeStar]:
        del obs, config, radec_movement
        queried.append((catalog_name, mag_min, mag_max))
        return []

    scene.sub_solar_offset_vu = (10.0, 0.0)
    scene.occluder_center_vu = None
    scene.ring_radius_at_u = None
    monkeypatch.setattr(geometry_module, 'stars_in_extfov', _record)
    obs = _scene_obs(titan_entry=_inventory_entry((60.5, 60.5), 26.0, 1.2e6), extra={})
    _geometry(obs, _titan_only_config(tmp_path))
    return queried


def test_star_mask_queries_exactly_the_photometry_catalogs(
    scene: type[_SceneBackplane], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The mask is built from two windows of the two photometry catalogs.

    Pinned exactly: the windows must tile ``[-2.0, star_mask_vmag_limit)``
    without a gap, and each must come from the catalog whose bright-end
    photometry is trusted over that range.
    """
    queried = _record_star_queries(scene, monkeypatch, tmp_path)
    assert queried == [('ybsc', -2.0, 6.5), ('tycho2', 6.5, 8.0)]


def test_star_mask_never_queries_ucac4(
    scene: type[_SceneBackplane], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """UCAC4's saturated bright end is never used to build the mask."""
    queried = _record_star_queries(scene, monkeypatch, tmp_path)
    assert 'ucac4' not in [name for name, _lo, _hi in queried]


# ---------------------------------------------------------------------------
# Never-raise behavior and emission
# ---------------------------------------------------------------------------


class _BrokenObs(FakeObs):
    """Observation whose inventory lookup fails the way a bad frame does."""

    def inventory(self, body_list: list[str], *, return_type: str = 'full') -> Any:
        """Raise the way an unresolvable body does inside oops."""
        del body_list, return_type
        raise ValueError('body not resolvable in this scene')


def test_an_unresolvable_body_fails_the_image(tmp_path: Path) -> None:
    """A frame whose geometry cannot be evaluated fails rather than emitting.

    The orchestrator fails the image with ``status_reason=internal_error``,
    naming the component that raised and the type it raised.  A zero-reliability
    feature in its place would be removed by the gate without a record, and
    would read downstream as a navigation that ran and concluded.
    """
    obs = _BrokenObs(data=np.zeros((120, 120)), extfov_margin_vu=(10, 10))
    model = NavModelTitan('titan:TITAN', cast(Any, obs), config=_titan_only_config(tmp_path))
    with pytest.raises(ValueError) as exc_info:
        model.create_model()
    assert 'body not resolvable' in str(exc_info.value)


def _non_finite_geometry_feature(
    scene: type[_SceneBackplane],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    bbox_fill: float | None = None,
    radius_km: float | None = None,
) -> Any:
    """Emit the haze feature for a frame carrying a non-finite quantity.

    ``bbox_fill`` poisons every inventory bounding-box coordinate;
    ``radius_km`` poisons the registered body radius.  Either is a real
    outcome of an unresolvable SPICE geometry.
    """
    scene.sub_solar_offset_vu = (10.0, 0.0)
    scene.occluder_center_vu = None
    scene.ring_radius_at_u = None
    if radius_km is not None:
        monkeypatch.setattr(geometry_module, '_body_radius_km', lambda name: radius_km)
    entry = _inventory_entry((60.5, 60.5), 26.0, 1.2e6)
    if bbox_fill is not None:
        for key in (
            'u_min_unclipped',
            'u_max_unclipped',
            'v_min_unclipped',
            'v_max_unclipped',
        ):
            entry[key] = bbox_fill
        entry['center_uv'] = np.array([bbox_fill, bbox_fill])
    obs = _scene_obs(titan_entry=entry, extra={})
    model = NavModelTitan(
        'titan:TITAN',
        cast(Any, obs),
        inventory=entry,
        siblings=[],
        config=_titan_only_config(tmp_path),
    )
    model.create_model()
    features = model.to_features(cast(Any, None))
    assert len(features) == 1
    return features[0]


@pytest.mark.parametrize(
    ('bbox_fill', 'radius_km'),
    [
        (float('nan'), None),
        (float('inf'), None),
        (None, float('nan')),
    ],
    ids=['nan_bbox', 'inf_bbox', 'nan_radius'],
)
def test_non_finite_geometry_emits_rather_than_raising(
    scene: type[_SceneBackplane],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    bbox_fill: float | None,
    radius_km: float | None,
) -> None:
    """A non-finite inventory or body radius still produces a feature.

    A NaN that reaches ``NavFeature`` is rejected at construction, which
    would surface as a raising ``to_features`` -- read by the orchestrator
    as zero features, and so as an unattributable failure.
    """
    feature = _non_finite_geometry_feature(
        scene, monkeypatch, tmp_path, bbox_fill=bbox_fill, radius_km=radius_km
    )
    assert feature.feature_type is NavFeatureType.TITAN_LIMB


@pytest.mark.parametrize(
    ('bbox_fill', 'radius_km'),
    [
        (float('nan'), None),
        (float('inf'), None),
        (None, float('nan')),
    ],
    ids=['nan_bbox', 'inf_bbox', 'nan_radius'],
)
def test_non_finite_geometry_scores_zero_reliability(
    scene: type[_SceneBackplane],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    bbox_fill: float | None,
    radius_km: float | None,
) -> None:
    """The feature emitted for a non-finite frame is hard-zeroed."""
    feature = _non_finite_geometry_feature(
        scene, monkeypatch, tmp_path, bbox_fill=bbox_fill, radius_km=radius_km
    )
    assert feature.reliability == 0.0


@pytest.mark.parametrize(
    ('bbox_fill', 'radius_km'),
    [
        (float('nan'), None),
        (float('inf'), None),
        (None, float('nan')),
    ],
    ids=['nan_bbox', 'inf_bbox', 'nan_radius'],
)
def test_non_finite_geometry_marks_the_axis_degenerate(
    scene: type[_SceneBackplane],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    bbox_fill: float | None,
    radius_km: float | None,
) -> None:
    """The feature emitted for a non-finite frame declares no usable axis."""
    feature = _non_finite_geometry_feature(
        scene, monkeypatch, tmp_path, bbox_fill=bbox_fill, radius_km=radius_km
    )
    assert feature.geometry.axis_degenerate is True


def test_create_model_records_the_geometry(scene: type[_SceneBackplane], tmp_path: Path) -> None:
    """``create_model`` publishes the envelope size into the model metadata."""
    scene.sub_solar_offset_vu = (10.0, 0.0)
    scene.occluder_center_vu = None
    scene.ring_radius_at_u = None
    obs = _scene_obs(titan_entry=_inventory_entry((60.5, 60.5), 26.0, 1.2e6), extra={})
    model = NavModelTitan.instances_for_obs(cast(Any, obs), config=_titan_only_config(tmp_path))[0]
    model.create_model()
    assert model.metadata['envelope_diameter_px'] > 0.0


def test_create_model_logs_the_titan_section(
    scene: type[_SceneBackplane], tmp_path: Path, capsys: Any
) -> None:
    """The model keeps its named log section for per-image inspection."""
    scene.sub_solar_offset_vu = (10.0, 0.0)
    scene.occluder_center_vu = None
    scene.ring_radius_at_u = None
    obs = _scene_obs(titan_entry=_inventory_entry((60.5, 60.5), 26.0, 1.2e6), extra={})
    model = NavModelTitan.instances_for_obs(cast(Any, obs), config=_titan_only_config(tmp_path))[0]
    model.create_model()
    assert 'TITAN MODEL' in capsys.readouterr().out


def test_to_annotations_is_empty_for_unevaluated_geometry() -> None:
    """A frame whose geometry never evaluated draws no overlay at all.

    Its defaults are a zero-radius envelope at the frame origin; painting a
    center mark there would claim a position the frame does not support. Read
    off the geometry directly, since an observation that cannot answer now
    fails the image rather than leaving a model standing with its defaults.

    """
    unevaluated = _inputs(r_env_px=0.0, theta_rad=0.0, axis_degenerate=True)
    assert len(haze_overlay(unevaluated, sector_half_angle_deg=45.0).nonzero()[0]) == 0


# ---------------------------------------------------------------------------
# Overlay rasterization
# ---------------------------------------------------------------------------


_SECTOR_HALF_ANGLE_DEG = 60.0
"""Arc-fit sector half-width the overlay tests draw with."""


_GATED_DOT_SPACING = 4
"""Dot spacing the gated-overlay tests draw with (the shipped config value)."""


def _overlay(
    *,
    dot_spacing: int = 1,
    center_vu: tuple[float, float] = (100.0, 100.0),
    r_env_px: float = 60.0,
    theta_rad: float = 0.0,
) -> np.ndarray:
    """Rasterize the overlay of a haze geometry with the given parameters."""
    return haze_overlay(
        _inputs(r_env_px=r_env_px, center_vu=center_vu, theta_rad=theta_rad),
        sector_half_angle_deg=_SECTOR_HALF_ANGLE_DEG,
        dot_spacing=dot_spacing,
    )


def _painted_near(
    overlay: np.ndarray,
    center_vu: tuple[float, float],
    radius_px: float,
    phi_deg: float,
) -> bool:
    """Whether any pixel within one pixel of ``(radius, phi)`` is painted.

    Curves are rasterized from samples half a pixel apart, so the pixel a
    given polar position lands on can differ by one from the ideal point's;
    the tests assert on a one-pixel neighborhood rather than pinning the
    rounding.
    """
    phi_rad = math.radians(phi_deg)
    v = round(center_vu[0] + radius_px * math.sin(phi_rad))
    u = round(center_vu[1] + radius_px * math.cos(phi_rad))
    return bool(overlay[v - 1 : v + 2, u - 1 : u + 2].any())


@pytest.mark.parametrize('phi_deg', [0.0, 45.0, 90.0, 180.0, 270.0], ids=str)
def test_overlay_draws_the_envelope_circle(phi_deg: float) -> None:
    """The envelope circle is painted all the way around the disc."""
    overlay = _overlay()
    assert _painted_near(overlay, (100.0, 100.0), 60.0, phi_deg) is True


def test_overlay_draws_the_sunward_half_of_the_symmetry_axis() -> None:
    """The mirror plane is drawn through the disc toward the sub-solar side."""
    overlay = _overlay(theta_rad=math.radians(30.0))
    assert _painted_near(overlay, (100.0, 100.0), 30.0, 30.0) is True


def test_overlay_draws_the_anti_sunward_half_of_the_symmetry_axis() -> None:
    """The chord spans both sides, so the mirror plane reads as a line."""
    overlay = _overlay(theta_rad=math.radians(30.0))
    assert _painted_near(overlay, (100.0, 100.0), 30.0, 210.0) is True


def test_overlay_marks_the_disc_center() -> None:
    """A cross marks the center the summary PNG then shifts onto the fit."""
    overlay = _overlay()
    assert bool(overlay[100, 100]) is True


def test_overlay_draws_the_sector_arc_at_the_solid_radius() -> None:
    """The arc-fit sector's inner boundary sits at the solid-body radius.

    ``_inputs`` builds the solid radius at nine tenths of the envelope, so
    54 px is the solid radius of the 60 px envelope drawn here.
    """
    overlay = _overlay(theta_rad=math.radians(30.0))
    assert _painted_near(overlay, (100.0, 100.0), 54.0, 75.0) is True


def test_overlay_draws_the_sector_edge_rays() -> None:
    """The sector edges join the solid radius to the envelope."""
    overlay = _overlay(theta_rad=math.radians(30.0))
    edge_deg = 30.0 + _SECTOR_HALF_ANGLE_DEG
    assert _painted_near(overlay, (100.0, 100.0), 57.0, edge_deg) is True


def test_overlay_leaves_the_band_outside_the_sector_clear() -> None:
    """Between the two radii, only the sector and the axis are drawn."""
    overlay = _overlay(theta_rad=math.radians(30.0))
    assert _painted_near(overlay, (100.0, 100.0), 57.0, 150.0) is False


def test_gated_overlay_paints_fewer_pixels() -> None:
    """A gated feature's curves are dotted rather than solid."""
    solid = _overlay()
    dotted = _overlay(dot_spacing=_GATED_DOT_SPACING)
    assert int(dotted.sum()) < int(solid.sum())


def test_gated_overlay_paints_a_subset_of_the_solid_one() -> None:
    """Dotting drops samples; it never moves the curves."""
    solid = _overlay()
    dotted = _overlay(dot_spacing=_GATED_DOT_SPACING)
    assert bool(np.all(solid[dotted]))


def test_overlay_clips_a_body_hanging_off_the_frame_edge() -> None:
    """A partly-framed disc paints the part that is inside the frame."""
    overlay = _overlay(center_vu=(5.0, 5.0))
    assert int(overlay.sum()) > 0


def test_overlay_is_empty_for_a_body_nowhere_near_the_frame() -> None:
    """A disc entirely outside the frame paints nothing."""
    overlay = _overlay(center_vu=(-500.0, -500.0))
    assert int(overlay.sum()) == 0


def test_overlay_is_empty_for_an_unevaluated_geometry() -> None:
    """A zero-radius envelope draws nothing, not a cross at the frame origin."""
    overlay = _overlay(r_env_px=0.0, center_vu=(0.0, 0.0))
    assert int(overlay.sum()) == 0


def test_overlay_is_empty_for_a_non_finite_axis() -> None:
    """An undefined symmetry axis draws nothing rather than raising.

    Nothing in the geometry path produces one today, but the overlay is
    built inside a sandboxed model hook: a curve sampled from a NaN angle
    would raise where the contract promises an annotation collection.
    """
    overlay = _overlay(theta_rad=float('nan'))
    assert int(overlay.sum()) == 0


def test_overlay_is_empty_for_a_non_finite_center() -> None:
    """An undefined disc center draws nothing rather than raising."""
    overlay = _overlay(center_vu=(float('nan'), float('nan')))
    assert int(overlay.sum()) == 0


def test_overlay_is_empty_for_a_non_finite_solid_radius() -> None:
    """An undefined solid radius draws nothing rather than raising."""
    inputs = dataclasses.replace(_inputs(r_env_px=60.0), r_solid_px=float('nan'))
    overlay = haze_overlay(inputs, sector_half_angle_deg=_SECTOR_HALF_ANGLE_DEG)
    assert int(overlay.sum()) == 0


# ---------------------------------------------------------------------------
# Annotations
# ---------------------------------------------------------------------------


def _annotated_model(
    scene: type[_SceneBackplane],
    tmp_path: Path,
    *,
    center_vu: tuple[float, float] = (60.5, 60.5),
    occluder: bool = False,
    config_override_yaml: str | None = None,
) -> tuple[NavModelTitan, FakeObs]:
    """Build and evaluate a Titan model over the analytic scene.

    Parameters:
        scene: The patched analytic backplane class.
        tmp_path: Per-test directory for the satellite-list override.
        center_vu: Titan's field-of-view center.
        occluder: When True, a nearer moon covers enough of the disc to
            drive the occlusion hard-zero, leaving every drawn quantity
            (center, radii, axis) untouched.
        config_override_yaml: Optional YAML text layered onto the config
            after every other override.
    """
    scene.sub_solar_offset_vu = (10.0, 0.0)
    scene.ring_radius_at_u = None
    scene.titan_center_vu = center_vu
    scene.occluder_center_vu = center_vu if occluder else None
    scene.occluder_radius_px = 20.0
    extra = (
        {'RHEA': _inventory_entry((center_vu[0], center_vu[1]), 20.0, 5.0e5)} if occluder else {}
    )
    obs = _scene_obs(titan_entry=_inventory_entry(center_vu, 26.0, 1.2e6), extra=extra)
    config = _titan_only_config(tmp_path)
    if occluder:
        config.update_config(_satellites_override(tmp_path, ('TITAN', 'RHEA')))
    if config_override_yaml is not None:
        extra_override = tmp_path / 'extra_override.yaml'
        extra_override.write_text(config_override_yaml)
        config.update_config(extra_override)
    model = NavModelTitan.instances_for_obs(cast(Any, obs), config=config)[0]
    model.create_model()
    return cast(NavModelTitan, model), obs


def test_to_annotations_emits_one_annotation(scene: type[_SceneBackplane], tmp_path: Path) -> None:
    """A navigable Titan contributes exactly one annotation."""
    model, _obs = _annotated_model(scene, tmp_path)
    assert len(model.to_annotations(cast(Any, None)).annotations) == 1


def test_annotation_overlay_matches_the_extended_frame(
    scene: type[_SceneBackplane], tmp_path: Path
) -> None:
    """The overlay is extfov-shaped, which the Annotation itself enforces."""
    model, obs = _annotated_model(scene, tmp_path)
    annotation = model.to_annotations(cast(Any, None)).annotations[0]
    assert annotation.overlay.shape == obs.extdata_shape_vu


def test_annotation_draws_the_envelope_circle(scene: type[_SceneBackplane], tmp_path: Path) -> None:
    """The circle is painted at the envelope radius around the disc center."""
    model, _obs = _annotated_model(scene, tmp_path)
    geometry = model.geometry_inputs
    annotation = model.to_annotations(cast(Any, None)).annotations[0]
    painted = _painted_near(
        annotation.overlay, geometry.predicted_center_vu, geometry.r_env_px, 0.0
    )
    assert painted is True


def test_annotation_uses_the_body_overlay_color(
    scene: type[_SceneBackplane], tmp_path: Path
) -> None:
    """Titan's overlay is drawn in the same color as every other body's limb."""
    model, _obs = _annotated_model(scene, tmp_path)
    annotation = model.to_annotations(cast(Any, None)).annotations[0]
    assert tuple(annotation.overlay_color) == tuple(Config().bodies.label_limb_color)


def test_annotation_labels_the_body(scene: type[_SceneBackplane], tmp_path: Path) -> None:
    """The overlay carries the body label every other body overlay carries."""
    model, _obs = _annotated_model(scene, tmp_path)
    annotation = model.to_annotations(cast(Any, None)).annotations[0]
    assert annotation.text_info_list[0].text == 'TITAN'


def test_annotation_label_names_a_low_reliability_feature(
    scene: type[_SceneBackplane], tmp_path: Path
) -> None:
    """A hard-zeroed frame says so; a dotted circle alone would not explain it.

    The label states what the model measured -- reliability below the
    per-type threshold -- rather than a gate verdict, because manual
    navigation renders the same overlay with the gate deliberately skipped.
    """
    model, _obs = _annotated_model(scene, tmp_path, occluder=True)
    annotation = model.to_annotations(cast(Any, None)).annotations[0]
    assert annotation.text_info_list[0].text == 'TITAN (low reliability)'


def test_low_reliability_annotation_overlay_is_dotted(
    scene: type[_SceneBackplane], tmp_path: Path
) -> None:
    """The same geometry draws fewer pixels once its feature scores below the gate."""
    kept_model, _kept_obs = _annotated_model(scene, tmp_path)
    kept = kept_model.to_annotations(cast(Any, None)).annotations[0]
    gated_model, _gated_obs = _annotated_model(scene, tmp_path, occluder=True)
    gated = gated_model.to_annotations(cast(Any, None)).annotations[0]
    assert int(gated.overlay.sum()) < int(kept.overlay.sum())


def test_gated_dot_spacing_is_configurable(scene: type[_SceneBackplane], tmp_path: Path) -> None:
    """titan.annotation.gated_dot_spacing of one renders a gated overlay solid."""
    kept_model, _kept_obs = _annotated_model(scene, tmp_path)
    kept = kept_model.to_annotations(cast(Any, None)).annotations[0]
    solid_gated_model, _obs = _annotated_model(
        scene,
        tmp_path,
        occluder=True,
        config_override_yaml='titan:\n  annotation:\n    gated_dot_spacing: 1\n',
    )
    solid_gated = solid_gated_model.to_annotations(cast(Any, None)).annotations[0]
    assert int(solid_gated.overlay.sum()) == int(kept.overlay.sum())


def test_annotation_avoids_placing_text_over_the_disc(
    scene: type[_SceneBackplane], tmp_path: Path
) -> None:
    """The label-avoid mask covers the envelope so text lands off the body."""
    model, _obs = _annotated_model(scene, tmp_path)
    geometry = model.geometry_inputs
    annotation = model.to_annotations(cast(Any, None)).annotations[0]
    assert annotation.avoid_mask is not None
    v = round(geometry.predicted_center_vu[0])
    u = round(geometry.predicted_center_vu[1] + 0.5 * geometry.r_env_px)
    assert bool(annotation.avoid_mask[v, u]) is True


def test_annotation_renders_into_the_summary_overlay(
    scene: type[_SceneBackplane], tmp_path: Path
) -> None:
    """The annotation composites into the RGB the summary PNG is written from."""
    model, obs = _annotated_model(scene, tmp_path)
    geometry = model.geometry_inputs
    rgb = model.to_annotations(cast(Any, None)).combine(offset=(0.0, 0.0), include_text=False)
    assert rgb is not None
    margin_v, margin_u = obs.extfov_margin_vu
    center_fov = (
        geometry.predicted_center_vu[0] - margin_v,
        geometry.predicted_center_vu[1] - margin_u,
    )
    assert _painted_near(rgb[..., 0] > 0, center_fov, geometry.r_env_px, 0.0) is True


def test_summary_overlay_follows_the_navigated_offset(
    scene: type[_SceneBackplane], tmp_path: Path
) -> None:
    """The committed offset shifts the overlay onto the fitted center.

    This is what makes a predicted-geometry overlay a picture of the fit:
    the summary PNG combines annotations at the navigated offset, so the
    center cross lands where the technique put the body.
    """
    model, obs = _annotated_model(scene, tmp_path)
    geometry = model.geometry_inputs
    rgb = model.to_annotations(cast(Any, None)).combine(offset=(4.0, -3.0), include_text=False)
    assert rgb is not None
    margin_v, margin_u = obs.extfov_margin_vu
    center_fov = (
        geometry.predicted_center_vu[0] - margin_v + 4.0,
        geometry.predicted_center_vu[1] - margin_u - 3.0,
    )
    assert _painted_near(rgb[..., 0] > 0, center_fov, 0.0, 0.0) is True
