"""Spec-first tests for the ``NavModelBody`` rendering pipeline.

Covers ``_render`` / ``_build_backplane_model`` plus the parts of
``instances_for_obs`` / ``to_features`` / ``to_annotations`` that depend on the
rendered state.  The contract asserted here comes from
``docs/dev_guide/dev_guide_navigation_models_body.rst`` and the module / method
docstrings, not from the current implementation:

- The predicted brightness image is extfov-shaped, float64, zero off-body, and
  is the Lambert cosine plus a small floor on lit silhouette pixels (or the
  0.01 visibility floor when the body is entirely dark).
- The limb mask is the set of *lit* silhouette pixels with at least one
  off-body neighbour; the terminator mask is the set of lit pixels with at
  least one dark neighbour.
- ``visible_lit_fraction`` and ``overflow_fraction`` follow the documented
  formulas over the discrete masks.
- An empty silhouette collapses every downstream product (masks, samplers,
  km/px) to its empty / zero form without special-case failures.
- Feature emission gates (LIMB_ARC / BODY_DISC / BODY_BLOB / TERMINATOR_ARC)
  fire exactly per the documented thresholds.

The oops ``Meshgrid`` / ``Backplane`` names inside the module under test are
monkeypatched with an analytic orthographic-sphere backplane so the tests are
hermetic (no SPICE, no holdings).
"""

from __future__ import annotations

import math
import types
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import polymath
import pytest
from tests.shims import (
    BodyBackplaneData,
    FakeBackplane,
    FakeMeshgrid,
    FakeObs,
    bare_nav_context,
    probe_grid_vu,
)

import spindoctor.nav_model.nav_model_body as nav_model_body_module
from spindoctor.annotation import Annotations
from spindoctor.config.config import Config
from spindoctor.feature.feature import NavFeature
from spindoctor.feature.geometry import LimbPolyline
from spindoctor.nav_model.nav_model_body import NavModelBody
from spindoctor.nav_orchestrator.nav_context import NavContext
from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType

_BODY = 'TESTBODY'


@dataclass(frozen=True)
class _SphereSpec:
    """Analytic orthographic-sphere scene driving the fake backplane.

    Parameters:
        center_vu: Sphere centre in FOV pixel coordinates ``(v, u)``.
        radius_px: Sphere radius in pixels.
        sun_vuz: Sun direction in the ``(v, u, z)`` image frame where ``+z``
            points toward the observer.  Normalised on use.
        km_per_px: Constant km/px scale on the resolved body.
    """

    center_vu: tuple[float, float]
    radius_px: float
    sun_vuz: tuple[float, float, float] = (0.0, 0.0, 1.0)
    km_per_px: float = 5.0


def _sun_for_angle(alpha_deg: float) -> tuple[float, float, float]:
    """Return a sun direction tilted ``alpha_deg`` from the observer axis.

    Parameters:
        alpha_deg: Tilt angle in degrees; the tilt is along the +u axis so the
            terminator appears on the -u side of the disc.
    """
    alpha = math.radians(alpha_deg)
    return (0.0, math.sin(alpha), math.cos(alpha))


def _sphere_backplane_class(spec: _SphereSpec) -> type:
    """Build a Backplane stand-in rendering ``spec`` over any meshgrid.

    The class mirrors the three per-pixel products ``_build_backplane_model``
    queries: ``incidence_angle`` (masked off the silhouette), ``lambert_law``
    (cosine of incidence clipped at zero, masked off the silhouette), and
    ``resolution`` (constant km/px, masked off the silhouette).

    Parameters:
        spec: Sphere geometry to render.
    """
    sun = np.asarray(spec.sun_vuz, dtype=np.float64)
    sun = sun / np.linalg.norm(sun)

    class _SphereBackplane:
        """Backplane stand-in computing sphere geometry over a fake meshgrid.

        Parameters:
            obs: Observation (unused; accepted for signature parity).
            meshgrid: ``FakeMeshgrid`` carrying origin / limit / oversample.
        """

        def __init__(self, obs: Any, meshgrid: FakeMeshgrid | None = None) -> None:
            assert meshgrid is not None
            self._mg = meshgrid

        def _grid(self) -> tuple[NDArrayFloatType, NDArrayFloatType]:
            """Return ``(vv, uu)`` sample-centre coordinate arrays."""
            scattered = probe_grid_vu(self._mg)
            if scattered is not None:
                return scattered
            assert self._mg.origin is not None
            assert self._mg.limit is not None
            assert self._mg.oversample is not None
            origin_u, origin_v = self._mg.origin
            limit_u, limit_v = self._mg.limit
            over_u, over_v = self._mg.oversample
            n_u = round((limit_u - origin_u) * over_u) + 1
            n_v = round((limit_v - origin_v) * over_v) + 1
            u = origin_u + np.arange(n_u, dtype=np.float64) / over_u
            v = origin_v + np.arange(n_v, dtype=np.float64) / over_v
            vv, uu = np.meshgrid(v, u, indexing='ij')
            return vv, uu

        def _geometry(self) -> tuple[NDArrayBoolType, NDArrayFloatType, NDArrayFloatType]:
            """Return ``(on_body, incidence_rad, cos_incidence)`` arrays."""
            vv, uu = self._grid()
            center_v, center_u = spec.center_vu
            dv = vv - center_v
            du = uu - center_u
            dist = np.hypot(dv, du)
            on_body = dist <= spec.radius_px
            r = spec.radius_px
            z = np.sqrt(np.clip(r * r - dist * dist, 0.0, None))
            cos_inc = (dv * sun[0] + du * sun[1] + z * sun[2]) / r
            incidence = np.arccos(np.clip(cos_inc, -1.0, 1.0))
            return on_body, incidence, cos_inc

        def incidence_angle(self, body_name: str) -> polymath.Scalar:
            """Return per-sample incidence angle masked off the silhouette."""
            del body_name
            on_body, incidence, _ = self._geometry()
            return polymath.Scalar(np.ma.array(incidence, mask=~on_body))

        def lambert_law(self, body_name: str) -> polymath.Scalar:
            """Return per-sample Lambert reflectance masked off the silhouette."""
            del body_name
            on_body, _, cos_inc = self._geometry()
            lam = np.clip(cos_inc, 0.0, None)
            return polymath.Scalar(np.ma.array(lam, mask=~on_body))

        def resolution(self, body_name: str) -> polymath.Scalar:
            """Return the constant km/px scale masked off the silhouette."""
            del body_name
            on_body, _, _ = self._geometry()
            res = np.full(on_body.shape, spec.km_per_px, dtype=np.float64)
            return polymath.Scalar(np.ma.array(res, mask=~on_body))

    return _SphereBackplane


def _make_obs(
    spec: _SphereSpec,
    *,
    data_shape: tuple[int, int] = (100, 100),
    margin: int = 10,
    phase_deg: float = 30.0,
    sub_solar_lonlat_deg: tuple[float, float] = (10.0, 20.0),
    sub_observer_lonlat_deg: tuple[float, float] = (30.0, 40.0),
) -> tuple[FakeObs, dict[str, Any]]:
    """Build a ``FakeObs`` plus the sphere's inventory record.

    Parameters:
        spec: Sphere geometry (drives the inventory bounding box).
        data_shape: Sensor-area ``(rows, cols)`` shape.
        margin: Extfov margin applied to both axes.
        phase_deg: Centre phase angle reported by the geometry backplane.
        sub_solar_lonlat_deg: ``(lon, lat)`` reported for the sub-solar point.
        sub_observer_lonlat_deg: ``(lon, lat)`` reported for the sub-observer
            point.
    """
    center_v, center_u = spec.center_vu
    r = spec.radius_px
    inventory = {
        'u_min_unclipped': int(np.floor(center_u - r)),
        'u_max_unclipped': int(np.ceil(center_u + r)),
        'v_min_unclipped': int(np.floor(center_v - r)),
        'v_max_unclipped': int(np.ceil(center_v + r)),
        'u_pixel_size': 2.0 * r,
        'v_pixel_size': 2.0 * r,
        'range': 1.0e6,
    }
    geometry_body = BodyBackplaneData(
        body_mask=np.ones((1, 1), dtype=bool),
        incidence_rad=np.zeros((1, 1), dtype=np.float64),
        sub_solar_lon_rad=math.radians(sub_solar_lonlat_deg[0]),
        sub_solar_lat_rad=math.radians(sub_solar_lonlat_deg[1]),
        sub_observer_lon_rad=math.radians(sub_observer_lonlat_deg[0]),
        sub_observer_lat_rad=math.radians(sub_observer_lonlat_deg[1]),
        center_phase_rad=math.radians(phase_deg),
    )
    obs = FakeObs(
        data=np.zeros(data_shape, dtype=np.float64),
        extfov_margin_vu=(margin, margin),
        closest_planet='SATURN',
        ext_bp=FakeBackplane(per_body={_BODY: geometry_body}),
        inventory_records={_BODY: inventory},
    )
    return obs, inventory


def _make_model(
    monkeypatch: pytest.MonkeyPatch,
    spec: _SphereSpec,
    *,
    config: Config | None = None,
    **obs_kwargs: Any,
) -> tuple[NavModelBody, FakeObs]:
    """Build a ``NavModelBody`` wired to the analytic sphere backplane.

    Parameters:
        monkeypatch: Pytest monkeypatch fixture (patches the module's oops
            ``Meshgrid`` / ``Backplane`` names).
        spec: Sphere geometry to render.
        config: Optional ``Config`` override for the model.
        **obs_kwargs: Forwarded to :func:`_make_obs`.
    """
    obs, inventory = _make_obs(spec, **obs_kwargs)
    monkeypatch.setattr(nav_model_body_module, 'Meshgrid', FakeMeshgrid)
    monkeypatch.setattr(nav_model_body_module, 'Backplane', _sphere_backplane_class(spec))
    model = NavModelBody(f'body:{_BODY}', cast(Any, obs), _BODY, inventory=inventory, config=config)
    return model, obs


def _noise_context(obs: FakeObs, *, seed: int = 12345) -> NavContext:
    """Return a NavContext over a deterministic unit-sigma noise frame.

    Parameters:
        obs: Observation whose extfov shape sizes the frame.
        seed: RNG seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    image = rng.standard_normal(obs.extdata_shape_vu)
    return bare_nav_context(cast(Any, obs), image)


def _bodies_config(**overrides: Any) -> Config:
    """Return a ``Config`` with ``bodies`` keys overridden.

    Parameters:
        **overrides: Key / value pairs written into the ``bodies`` block.
    """
    config = Config()
    config.read_config()
    bodies = dict(config._config_dict['bodies'])
    bodies.update(overrides)
    config._config_dict['bodies'] = bodies
    config._update_attrdicts()
    return config


def _feature_types(features: list[NavFeature]) -> set[str]:
    """Return the set of feature-type names emitted.

    Parameters:
        features: Features returned by ``to_features``.
    """
    return {f.feature_type.name for f in features}


def _analytic_disc(obs: FakeObs, spec: _SphereSpec) -> NDArrayBoolType:
    """Return the analytic pixel-centre silhouette in extfov coordinates.

    Pixel index ``i`` covers continuous coordinate ``[i, i + 1)``, so its
    centre sits at ``i + 0.5`` in the FOV frame the sphere is defined in.

    Parameters:
        obs: Observation defining the extfov grid.
        spec: Sphere geometry.
    """
    shape = obs.extdata_shape_vu
    vv, uu = np.indices(shape, dtype=np.float64)
    center_v = spec.center_vu[0] + obs.extfov_margin_v
    center_u = spec.center_vu[1] + obs.extfov_margin_u
    disc: NDArrayBoolType = np.hypot(vv + 0.5 - center_v, uu + 0.5 - center_u) <= spec.radius_px
    return disc


# ---------------------------------------------------------------------------
# _render / _build_backplane_model: brightness image contract
# ---------------------------------------------------------------------------


def test_model_img_is_extfov_shaped_float64(monkeypatch: pytest.MonkeyPatch) -> None:
    """The rendered brightness image is float64 over the extended FOV grid."""
    model, obs = _make_model(monkeypatch, _SphereSpec((50.0, 50.0), 20.0))
    model.create_model()
    assert model._model_img is not None
    assert model._model_img.shape == obs.extdata_shape_vu
    assert model._model_img.dtype == np.float64


def test_model_img_zero_outside_bbox(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pixels outside the body bounding box carry exactly zero."""
    model, _obs = _make_model(monkeypatch, _SphereSpec((50.0, 50.0), 20.0))
    model.create_model()
    assert model._model_img is not None
    v_min, u_min, v_max, u_max = model._bbox_extfov_vu
    outside = np.ones_like(model._model_img, dtype=bool)
    outside[v_min:v_max, u_min:u_max] = False
    assert float(np.abs(model._model_img[outside]).max()) == 0.0


def test_model_img_zero_on_off_body_pixels_inside_bbox(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Off-body pixels inside the bounding box carry exactly zero."""
    model, _obs = _make_model(monkeypatch, _SphereSpec((50.0, 50.0), 20.0))
    model.create_model()
    assert model._model_img is not None
    assert model._body_mask is not None
    v_min, u_min, v_max, u_max = model._bbox_extfov_vu
    inside_bbox = np.zeros_like(model._body_mask)
    inside_bbox[v_min:v_max, u_min:u_max] = True
    off_body = inside_bbox & ~model._body_mask
    assert bool(off_body.any())
    assert float(np.abs(model._model_img[off_body]).max()) == 0.0


def test_model_img_lambert_plus_floor_at_disc_center(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A lit pixel's value is the Lambert cosine plus the 0.05 floor."""
    # Centre the sphere on a pixel centre (pixel 50 spans [50, 51)) so the
    # centre pixel's surface normal is +z and cos(incidence) = cos(60).
    spec = _SphereSpec((50.5, 50.5), 20.0, sun_vuz=_sun_for_angle(60.0))
    model, obs = _make_model(monkeypatch, spec)
    model.create_model()
    assert model._model_img is not None
    center_v = 50 + obs.extfov_margin_v
    center_u = 50 + obs.extfov_margin_u
    expected = math.cos(math.radians(60.0)) + 0.05
    assert model._model_img[center_v, center_u] == pytest.approx(expected, abs=0.02)


def test_dark_body_renders_visibility_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    """An entirely-dark silhouette renders the 0.01 visibility floor on-body."""
    model, _obs = _make_model(
        monkeypatch, _SphereSpec((50.0, 50.0), 20.0, sun_vuz=(0.0, 0.0, -1.0))
    )
    model.create_model()
    assert model._model_img is not None
    assert model._body_mask is not None
    on_body_values = np.unique(model._model_img[model._body_mask])
    assert on_body_values.tolist() == [0.01]


def test_binary_silhouette_when_lambert_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """With ``use_lambert`` false the brightness image is the binary silhouette."""
    config = _bodies_config(use_lambert=False)
    model, _obs = _make_model(monkeypatch, _SphereSpec((50.0, 50.0), 20.0), config=config)
    model.create_model()
    assert model._model_img is not None
    assert model._body_mask is not None
    on_body_values = np.unique(model._model_img[model._body_mask])
    assert on_body_values.tolist() == [1.0]


def test_albedo_scales_brightness(monkeypatch: pytest.MonkeyPatch) -> None:
    """With ``use_albedo`` the brightness image is scaled by the body's albedo."""
    albedo = 0.25
    plain = _bodies_config(use_albedo=False)
    scaled = _bodies_config(use_albedo=True, geometric_albedo={_BODY: albedo})
    spec = _SphereSpec((50.0, 50.0), 20.0)
    model_plain, _obs = _make_model(monkeypatch, spec, config=plain)
    model_plain.create_model()
    model_scaled, _obs2 = _make_model(monkeypatch, spec, config=scaled)
    model_scaled.create_model()
    assert model_plain._model_img is not None
    assert model_scaled._model_img is not None
    ratio = model_scaled._model_img.max() / model_plain._model_img.max()
    assert ratio == pytest.approx(albedo, rel=1e-9)


# ---------------------------------------------------------------------------
# _render: mask contracts
# ---------------------------------------------------------------------------


def test_body_mask_matches_analytic_silhouette(monkeypatch: pytest.MonkeyPatch) -> None:
    """The rendered silhouette agrees with the analytic disc (IoU > 0.9)."""
    spec = _SphereSpec((50.0, 50.0), 20.0)
    model, obs = _make_model(monkeypatch, spec)
    model.create_model()
    assert model._body_mask is not None
    analytic = _analytic_disc(obs, spec)
    intersection = int((model._body_mask & analytic).sum())
    union = int((model._body_mask | analytic).sum())
    assert intersection / union > 0.9


def test_limb_mask_is_boundary_subset_of_body(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every limb pixel is a silhouette pixel with an off-body 4-neighbour."""
    model, _obs = _make_model(monkeypatch, _SphereSpec((50.0, 50.0), 20.0))
    model.create_model()
    assert model._limb_mask is not None
    assert model._body_mask is not None
    limb = model._limb_mask
    body = model._body_mask
    assert bool(limb.any())
    assert bool((limb & ~body).sum() == 0)
    off = ~body
    has_space_neighbour = (
        np.roll(off, 1, axis=0)
        | np.roll(off, -1, axis=0)
        | np.roll(off, 1, axis=1)
        | np.roll(off, -1, axis=1)
    )
    assert bool((limb & ~has_space_neighbour).sum() == 0)


def test_limb_mask_empty_for_fully_dark_body(monkeypatch: pytest.MonkeyPatch) -> None:
    """The limb mask keeps only lit-side vertices, so a dark body has none."""
    model, _obs = _make_model(
        monkeypatch, _SphereSpec((50.0, 50.0), 20.0, sun_vuz=(0.0, 0.0, -1.0))
    )
    model.create_model()
    assert model._limb_mask is not None
    assert int(model._limb_mask.sum()) == 0


def test_terminator_location_matches_analytic_prediction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The terminator ridge sits at ``u = center_u - r*cos(alpha)`` mid-disc.

    For an orthographic sphere lit from ``alpha`` degrees off the observer
    axis (tilt along +u), the terminator's innermost column is at
    ``center_u - radius * cos(alpha)``.
    """
    alpha_deg = 60.0
    spec = _SphereSpec((50.0, 50.0), 20.0, sun_vuz=_sun_for_angle(alpha_deg))
    model, obs = _make_model(monkeypatch, spec, phase_deg=alpha_deg)
    model.create_model()
    assert model._terminator_mask is not None
    vs, us = np.where(model._terminator_mask)
    assert vs.size > 0
    expected_u = (
        spec.center_vu[1] - spec.radius_px * math.cos(math.radians(alpha_deg)) + obs.extfov_margin_u
    )
    assert float(us.min()) == pytest.approx(expected_u, abs=2.0)


def test_terminator_empty_when_fully_lit(monkeypatch: pytest.MonkeyPatch) -> None:
    """A sub-solar-illuminated disc has no lit-to-dark boundary."""
    model, _obs = _make_model(monkeypatch, _SphereSpec((50.0, 50.0), 20.0))
    model.create_model()
    assert model._terminator_mask is not None
    assert int(model._terminator_mask.sum()) == 0


def test_nonsquare_fov_places_body_correctly(monkeypatch: pytest.MonkeyPatch) -> None:
    """On a non-square FOV the silhouette lands at the predicted extfov centre."""
    spec = _SphereSpec((30.0, 90.0), 15.0)
    model, obs = _make_model(monkeypatch, spec, data_shape=(60, 120))
    model.create_model()
    assert model._body_mask is not None
    vs, us = np.where(model._body_mask)
    assert vs.size > 0
    assert float(vs.mean()) == pytest.approx(30.0 + obs.extfov_margin_v, abs=1.0)
    assert float(us.mean()) == pytest.approx(90.0 + obs.extfov_margin_u, abs=1.0)


# ---------------------------------------------------------------------------
# _render: metadata and derived scalars
# ---------------------------------------------------------------------------


def test_metadata_geometry_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    """The documented geometry metadata entries reflect the backplane values."""
    model, _obs = _make_model(
        monkeypatch,
        _SphereSpec((50.0, 50.0), 20.0),
        phase_deg=45.0,
        sub_solar_lonlat_deg=(11.0, 22.0),
        sub_observer_lonlat_deg=(33.0, 44.0),
    )
    model.create_model()
    meta = model.metadata
    assert meta['body_name'] == _BODY
    assert meta['sub_solar_lon_deg'] == pytest.approx(11.0)
    assert meta['sub_solar_lat_deg'] == pytest.approx(22.0)
    assert meta['sub_observer_lon_deg'] == pytest.approx(33.0)
    assert meta['sub_observer_lat_deg'] == pytest.approx(44.0)
    assert meta['phase_angle_deg'] == pytest.approx(45.0)


def test_metadata_bookkeeping_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Timing, bbox-area, and size entries are populated by ``create_model``."""
    model, _obs = _make_model(monkeypatch, _SphereSpec((50.0, 50.0), 20.0))
    model.create_model()
    meta = model.metadata
    assert meta['end_time'] is not None
    assert meta['elapsed_time_sec'] is not None
    assert meta['bbox_area_px'] == pytest.approx(1600.0)
    assert meta['size_ok'] is True
    assert meta['predicted_diameter_px'] == pytest.approx(40.0)


def test_km_per_pixel_at_limb_positive_and_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    """The mean limb km/px is positive and at most the configured constant.

    Anti-aliased limb pixels average the on-body scale with off-body zeros, so
    the mean sits below the constant but must remain strictly positive.
    """
    spec = _SphereSpec((50.0, 50.0), 20.0, km_per_px=5.0)
    model, _obs = _make_model(monkeypatch, spec)
    model.create_model()
    assert model._km_per_pixel_at_limb > 0.0
    assert model._km_per_pixel_at_limb <= 5.0


def test_visible_lit_fraction_matches_illuminated_area(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """For a fully-framed sphere the lit fraction is ``(1 + cos(alpha)) / 2``."""
    alpha_deg = 60.0
    spec = _SphereSpec((50.0, 50.0), 20.0, sun_vuz=_sun_for_angle(alpha_deg))
    model, _obs = _make_model(monkeypatch, spec, phase_deg=alpha_deg)
    model.create_model()
    expected = (1.0 + math.cos(math.radians(alpha_deg))) / 2.0
    assert model._visible_lit_fraction == pytest.approx(expected, abs=0.05)


def test_overflow_zero_for_centered_body(monkeypatch: pytest.MonkeyPatch) -> None:
    """A body fully inside the sensor has zero overflow."""
    model, _obs = _make_model(monkeypatch, _SphereSpec((50.0, 50.0), 20.0))
    model.create_model()
    assert model._overflow_fraction == pytest.approx(0.0)


def test_overflow_fraction_matches_off_sensor_area(monkeypatch: pytest.MonkeyPatch) -> None:
    """Overflow equals the fraction of silhouette pixels outside the sensor.

    Checked twice per the documented formula ``1 - |body & sensor| / |body|``:
    exactly against the model's own discrete masks, and approximately against
    an independent analytic pixel-centre count (the rendered mask is
    anti-aliased, so the analytic count carries a small boundary tolerance).
    """
    spec = _SphereSpec((50.0, 95.0), 10.0)
    model, obs = _make_model(monkeypatch, spec)
    model.create_model()
    assert model._body_mask is not None
    sensor = obs.extfov_data_sensor_mask()
    body = model._body_mask
    from_masks = 1.0 - int((body & sensor).sum()) / int(body.sum())
    assert model._overflow_fraction == pytest.approx(from_masks)
    analytic = _analytic_disc(obs, spec)
    independent = 1.0 - int((analytic & sensor).sum()) / int(analytic.sum())
    assert model._overflow_fraction == pytest.approx(independent, abs=0.02)


def test_body_entirely_in_margin_scores_zero_visible_lit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A body rendered wholly inside the extfov margin has vlf 0 / overflow 1."""
    model, _obs = _make_model(monkeypatch, _SphereSpec((50.0, -5.0), 4.0))
    model.create_model()
    assert model._body_mask is not None
    assert int(model._body_mask.sum()) > 0
    assert model._visible_lit_fraction == pytest.approx(0.0)
    assert model._overflow_fraction == pytest.approx(1.0)


def test_body_overflowing_every_edge(monkeypatch: pytest.MonkeyPatch) -> None:
    """A frame-filling body clips against every extfov border and overflows."""
    model, _obs = _make_model(monkeypatch, _SphereSpec((50.0, 50.0), 80.0))
    model.create_model()
    assert model._body_mask is not None
    body = model._body_mask
    assert bool(body[0, :].any())
    assert bool(body[-1, :].any())
    assert bool(body[:, 0].any())
    assert bool(body[:, -1].any())
    assert model._overflow_fraction > 0.0


def test_guaranteed_visible_flag_true_for_inner_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bbox fully inside the sensor-minus-margin area sets the flag True."""
    model, _obs = _make_model(monkeypatch, _SphereSpec((50.0, 50.0), 20.0))
    model.create_model()
    assert model.metadata['guaranteed_visible_in_fov'] is True


def test_guaranteed_visible_flag_false_near_edge(monkeypatch: pytest.MonkeyPatch) -> None:
    """A bbox crossing the sensor-minus-margin boundary sets the flag False."""
    model, _obs = _make_model(monkeypatch, _SphereSpec((50.0, 95.0), 10.0))
    model.create_model()
    assert model.metadata['guaranteed_visible_in_fov'] is False


# ---------------------------------------------------------------------------
# _render: degenerate silhouettes
# ---------------------------------------------------------------------------


def test_empty_silhouette_produces_empty_products(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty silhouette collapses masks, image, and km/px without failing.

    The sphere is placed between the oversampled sample centres so no sample
    lands on the body; per the dev guide, an empty mask collapses the sampler
    to zero-length arrays and the downstream gates skip every feature.
    """
    model, obs = _make_model(monkeypatch, _SphereSpec((50.5, 50.5), 0.1))
    model.create_model()
    assert model._body_mask is not None
    assert int(model._body_mask.sum()) == 0
    assert model._model_img is not None
    assert float(np.abs(model._model_img).max()) == 0.0
    assert model._km_per_pixel_at_limb == pytest.approx(0.0)
    assert model._visible_lit_fraction == pytest.approx(0.0)
    assert model._overflow_fraction == pytest.approx(1.0)
    features = model.to_features(bare_nav_context(cast(Any, obs)))
    assert features == []


def test_body_beyond_extfov_renders_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """A body whose bbox lies wholly beyond the extfov renders nothing.

    The clipped bbox degenerates to the extfov edge (exercising the one-pixel
    bump) and the silhouette query returns all-masked, so every mask is empty.
    """
    model, _obs = _make_model(monkeypatch, _SphereSpec((50.0, 200.0), 20.0))
    model.create_model()
    assert model._body_mask is not None
    assert int(model._body_mask.sum()) == 0
    assert model._model_img is not None
    assert float(np.abs(model._model_img).max()) == 0.0
    assert model._overflow_fraction == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# to_features: emission gates over rendered state
# ---------------------------------------------------------------------------


def test_limb_arc_emitted_for_resolved_lit_body(monkeypatch: pytest.MonkeyPatch) -> None:
    """A well-resolved lit body with low limb uncertainty emits LIMB_ARC."""
    model, obs = _make_model(monkeypatch, _SphereSpec((50.0, 50.0), 20.0))
    model.create_model()
    features = model.to_features(bare_nav_context(cast(Any, obs)))
    assert 'LIMB_ARC' in _feature_types(features)


def test_body_disc_emitted_alongside_limb_arc(monkeypatch: pytest.MonkeyPatch) -> None:
    """BODY_DISC accompanies LIMB_ARC when visibility / overflow gates pass."""
    model, obs = _make_model(monkeypatch, _SphereSpec((50.0, 50.0), 20.0))
    model.create_model()
    types_emitted = _feature_types(model.to_features(bare_nav_context(cast(Any, obs))))
    assert 'LIMB_ARC' in types_emitted
    assert 'BODY_DISC' in types_emitted


def test_terminator_arc_emitted_when_gates_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    """TERMINATOR_ARC fires with enough vertices and sin(phase) above 0.05."""
    spec = _SphereSpec((50.0, 50.0), 20.0, sun_vuz=_sun_for_angle(60.0))
    model, obs = _make_model(monkeypatch, spec, phase_deg=60.0)
    model.create_model()
    types_emitted = _feature_types(model.to_features(bare_nav_context(cast(Any, obs))))
    assert 'TERMINATOR_ARC' in types_emitted


def test_no_terminator_below_phase_factor(monkeypatch: pytest.MonkeyPatch) -> None:
    """TERMINATOR_ARC is suppressed when sin(phase) is below 0.05."""
    spec = _SphereSpec((50.0, 50.0), 20.0, sun_vuz=_sun_for_angle(60.0))
    model, obs = _make_model(monkeypatch, spec, phase_deg=2.0)
    model.create_model()
    types_emitted = _feature_types(model.to_features(bare_nav_context(cast(Any, obs))))
    assert 'TERMINATOR_ARC' not in types_emitted


def test_blob_emitted_when_limb_uncertainty_too_high(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """BODY_BLOB replaces LIMB_ARC when the limb-uncertainty cap is exceeded.

    A tiny km/px scale maps the ellipsoid residual to far more than the 3 px
    cap, so the extractor must fall back to the centroid path.
    """
    spec = _SphereSpec((50.0, 50.0), 10.0, km_per_px=0.1)
    model, obs = _make_model(monkeypatch, spec)
    model.create_model()
    types_emitted = _feature_types(model.to_features(_noise_context(obs)))
    assert 'LIMB_ARC' not in types_emitted
    assert 'BODY_BLOB' in types_emitted


def test_no_disc_when_overflow_exceeds_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    """BODY_DISC is suppressed when the overflow fraction exceeds 0.3."""
    spec = _SphereSpec((50.0, 101.0), 8.0)
    model, obs = _make_model(monkeypatch, spec)
    model.create_model()
    assert model._overflow_fraction > 0.3
    types_emitted = _feature_types(model.to_features(_noise_context(obs)))
    assert 'BODY_DISC' not in types_emitted


def test_no_features_for_subpixel_body(monkeypatch: pytest.MonkeyPatch) -> None:
    """A body below the blob-diameter floor emits nothing at all."""
    model, obs = _make_model(monkeypatch, _SphereSpec((50.0, 50.0), 0.6))
    model.create_model()
    features = model.to_features(_noise_context(obs))
    assert features == []


def test_fully_dark_body_emits_no_photometric_blob(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Per the dev guide, an entirely-shadowed body emits no BODY_BLOB."""
    model, obs = _make_model(monkeypatch, _SphereSpec((50.0, 50.0), 20.0, sun_vuz=(0.0, 0.0, -1.0)))
    model.create_model()
    types_emitted = _feature_types(model.to_features(_noise_context(obs)))
    assert 'BODY_BLOB' not in types_emitted


# ---------------------------------------------------------------------------
# to_features: consistency between rendered silhouette and emitted polylines
# ---------------------------------------------------------------------------


def _limb_feature(model: NavModelBody, context: NavContext) -> NavFeature:
    """Return the emitted LIMB_ARC feature.

    Parameters:
        model: Model whose ``create_model`` already ran.
        context: NavContext for ``to_features``.
    """
    features = model.to_features(context)
    return next(f for f in features if f.feature_type.name == 'LIMB_ARC')


def test_limb_arc_vertices_lie_on_silhouette_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every LIMB_ARC vertex sits on the sub-pixel silhouette boundary.

    The probe refinement moves each ridge-pixel vertex onto the true
    silhouette, so for the analytic sphere every vertex's distance from the
    body centre must equal the radius to within the probe resolution
    (0.125 px ladder step bracketed at its midpoint, widened by the
    8-quantized normal directions).
    """
    spec = _SphereSpec((50.0, 50.0), 20.0)
    model, obs = _make_model(monkeypatch, spec)
    model.create_model()
    limb = _limb_feature(model, bare_nav_context(cast(Any, obs)))
    geometry = limb.geometry
    assert isinstance(geometry, LimbPolyline)
    margin_v, margin_u = obs.extfov_margin_v, obs.extfov_margin_u
    pos_v = geometry.vertices_vu[:, 0] - margin_v + 0.5 - spec.center_vu[0]
    pos_u = geometry.vertices_vu[:, 1] - margin_u + 0.5 - spec.center_vu[1]
    radial_error = np.hypot(pos_v, pos_u) - spec.radius_px
    assert float(np.abs(radial_error).max()) < 0.15


def test_limb_arc_normals_point_outward(monkeypatch: pytest.MonkeyPatch) -> None:
    """LIMB_ARC normals point away from the body centre at >=95% of vertices."""
    spec = _SphereSpec((50.0, 50.0), 20.0)
    model, obs = _make_model(monkeypatch, spec)
    model.create_model()
    limb = _limb_feature(model, bare_nav_context(cast(Any, obs)))
    geometry = limb.geometry
    assert isinstance(geometry, LimbPolyline)
    center = np.array(
        [
            spec.center_vu[0] + obs.extfov_margin_v,
            spec.center_vu[1] + obs.extfov_margin_u,
        ]
    )
    radial = geometry.vertices_vu - center
    dots = np.sum(geometry.normals_vu * radial, axis=1)
    outward_fraction = float(np.count_nonzero(dots > 0)) / dots.shape[0]
    assert outward_fraction >= 0.95


def test_limb_arc_sigmas_positive_and_finite(monkeypatch: pytest.MonkeyPatch) -> None:
    """The per-vertex sigma arrays are strictly positive and finite."""
    model, obs = _make_model(monkeypatch, _SphereSpec((50.0, 50.0), 20.0))
    model.create_model()
    limb = _limb_feature(model, bare_nav_context(cast(Any, obs)))
    geometry = limb.geometry
    assert isinstance(geometry, LimbPolyline)
    assert bool(np.all(geometry.sigma_normal_per_vertex_px > 0.0))
    assert bool(np.all(np.isfinite(geometry.sigma_normal_per_vertex_px)))


def test_disc_template_matches_model_crop(monkeypatch: pytest.MonkeyPatch) -> None:
    """The BODY_DISC template is the brightness image cropped to the bbox."""
    model, obs = _make_model(monkeypatch, _SphereSpec((50.0, 50.0), 20.0))
    model.create_model()
    features = model.to_features(bare_nav_context(cast(Any, obs)))
    disc = next(f for f in features if f.feature_type.name == 'BODY_DISC')
    assert model._model_img is not None
    assert model._body_mask is not None
    v_min, u_min, v_max, u_max = model._bbox_extfov_vu
    assert disc.template_img is not None
    assert disc.template_mask is not None
    assert np.array_equal(disc.template_img, model._model_img[v_min:v_max, u_min:u_max])
    assert np.array_equal(disc.template_mask, model._body_mask[v_min:v_max, u_min:u_max])


# ---------------------------------------------------------------------------
# instances_for_obs
# ---------------------------------------------------------------------------


def test_instances_for_obs_skips_simulated_obs() -> None:
    """Simulated observations use the sim sibling, so no instances here."""
    obs = types.SimpleNamespace(is_simulated=True)
    assert NavModelBody.instances_for_obs(cast(Any, obs)) == []


def test_instances_for_obs_requires_closest_planet() -> None:
    """No closest planet means no candidate body list."""
    obs = types.SimpleNamespace(closest_planet=None)
    assert NavModelBody.instances_for_obs(cast(Any, obs)) == []


def test_instances_for_obs_requires_callable_inventory() -> None:
    """A non-callable inventory attribute yields no instances."""
    obs = types.SimpleNamespace(closest_planet='SATURN', inventory=None)
    assert NavModelBody.instances_for_obs(cast(Any, obs)) == []


def test_instances_for_obs_tolerates_inventory_failure() -> None:
    """An inventory query that raises ValueError yields no instances."""

    def _raise(body_list: list[str], *, return_type: str = 'full') -> dict[str, Any]:
        """Raise the error the SPICE-less inventory path produces."""
        raise ValueError('no SPICE kernels')

    obs = types.SimpleNamespace(closest_planet='SATURN', inventory=_raise)
    assert NavModelBody.instances_for_obs(cast(Any, obs)) == []


def test_instances_for_obs_returns_one_model_per_body_in_extfov() -> None:
    """One instance per inventory entry whose bbox overlaps the extfov."""
    in_fov = {
        'u_min_unclipped': 30,
        'u_max_unclipped': 70,
        'v_min_unclipped': 30,
        'v_max_unclipped': 70,
        'u_pixel_size': 40.0,
        'v_pixel_size': 40.0,
        'range': 1.0e6,
    }
    out_of_fov = dict(in_fov)
    out_of_fov.update(u_min_unclipped=500, u_max_unclipped=600)
    obs = FakeObs(
        data=np.zeros((100, 100), dtype=np.float64),
        extfov_margin_vu=(10, 10),
        closest_planet='SATURN',
        inventory_records={'SATURN': in_fov, 'MIMAS': out_of_fov},
    )
    instances = NavModelBody.instances_for_obs(cast(Any, obs))
    assert [inst.name for inst in instances] == ['body:SATURN']


# ---------------------------------------------------------------------------
# to_annotations
# ---------------------------------------------------------------------------


def test_to_annotations_empty_before_create_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without a rendered model the annotation collection is empty."""
    model, obs = _make_model(monkeypatch, _SphereSpec((50.0, 50.0), 20.0))
    annotations = model.to_annotations(bare_nav_context(cast(Any, obs)))
    assert isinstance(annotations, Annotations)
    assert annotations.annotations == []


def test_to_annotations_emits_body_overlay(monkeypatch: pytest.MonkeyPatch) -> None:
    """After rendering, the model contributes one body annotation."""
    model, obs = _make_model(monkeypatch, _SphereSpec((50.0, 50.0), 20.0))
    model.create_model()
    annotations = model.to_annotations(bare_nav_context(cast(Any, obs)))
    assert len(annotations.annotations) == 1
