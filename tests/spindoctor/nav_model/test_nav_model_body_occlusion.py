"""Body-body occlusion tests for the SPICE-backed ``NavModelBody``.

A nearer sibling body hides part of the target's predicted disc.  The model
must drop the hidden limb / terminator vertices from the emitted polylines
(so the distance-transform fit never chases an arc the image does not show),
report an honest ``visible_arc_fraction`` that falls with occlusion depth, and
trim the hidden region out of the ``BODY_DISC`` correlation template (so the
correlator does not score against disc brightness that is not present).

The scene is analytic and hermetic: two orthographic spheres over a fake
meshgrid, with a ``where_in_front`` stand-in that returns the nearer body's
silhouette.  No SPICE, no holdings.
"""

from __future__ import annotations

import math
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
from spindoctor.feature.feature import NavFeature
from spindoctor.feature.flags import LimbArcFlags, TerminatorArcFlags
from spindoctor.feature.geometry import LimbPolyline
from spindoctor.nav_model.nav_model_body import NavModelBody, occluder_mask_for_body
from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType

_TARGET = 'TARGET'
_OCCLUDER = 'OCCLUDER'


@dataclass(frozen=True)
class _Sphere:
    """Analytic orthographic-sphere geometry driving the fake backplane.

    Parameters:
        center_vu: Sphere centre in FOV pixel coordinates ``(v, u)``.
        radius_px: Sphere radius in pixels.
        sun_vuz: Sun direction in the ``(v, u, z)`` image frame; ``+z`` points
            toward the observer.  Normalised on use.
        km_per_px: Constant km/px scale on the resolved body.
    """

    center_vu: tuple[float, float]
    radius_px: float
    sun_vuz: tuple[float, float, float] = (0.0, 0.0, 1.0)
    km_per_px: float = 5.0


def _grid(mg: FakeMeshgrid) -> tuple[NDArrayFloatType, NDArrayFloatType]:
    """Return ``(vv, uu)`` sample-centre coordinate arrays for a meshgrid."""
    scattered = probe_grid_vu(mg)
    if scattered is not None:
        return scattered
    assert mg.origin is not None
    assert mg.limit is not None
    assert mg.oversample is not None
    origin_u, origin_v = mg.origin
    limit_u, limit_v = mg.limit
    over_u, over_v = mg.oversample
    n_u = round((limit_u - origin_u) * over_u) + 1
    n_v = round((limit_v - origin_v) * over_v) + 1
    u = origin_u + np.arange(n_u, dtype=np.float64) / over_u
    v = origin_v + np.arange(n_v, dtype=np.float64) / over_v
    vv, uu = np.meshgrid(v, u, indexing='ij')
    return vv, uu


def _sphere_geometry(
    spec: _Sphere, vv: NDArrayFloatType, uu: NDArrayFloatType
) -> tuple[NDArrayBoolType, NDArrayFloatType, NDArrayFloatType]:
    """Return ``(on_body, incidence_rad, cos_incidence)`` over a grid."""
    sun = np.asarray(spec.sun_vuz, dtype=np.float64)
    sun = sun / np.linalg.norm(sun)
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


def _two_body_backplane_class(target: _Sphere, occluder: _Sphere) -> type:
    """Build a Backplane stand-in rendering ``target`` with an occluder in front.

    ``incidence_angle`` / ``lambert_law`` / ``resolution`` render ``target``
    (as the single-body sphere backplane does).  ``where_in_front(event,
    surface)`` returns the ``occluder`` silhouette whenever ``event`` names the
    occluder body -- the pixels where the nearer body sits in front of the
    target.

    Parameters:
        target: Sphere the backplane renders as the navigated body.
        occluder: Nearer sphere whose silhouette hides part of ``target``.
    """

    class _TwoBodyBackplane:
        """Backplane stand-in over a fake meshgrid for a two-body scene.

        Parameters:
            obs: Observation (unused; accepted for signature parity).
            meshgrid: ``FakeMeshgrid`` carrying origin / limit / oversample.
        """

        def __init__(self, obs: Any, meshgrid: FakeMeshgrid | None = None) -> None:
            assert meshgrid is not None
            self._mg = meshgrid

        def incidence_angle(self, body_name: str) -> polymath.Scalar:
            """Return per-sample incidence angle masked off the target."""
            del body_name
            vv, uu = _grid(self._mg)
            on_body, incidence, _ = _sphere_geometry(target, vv, uu)
            return polymath.Scalar(np.ma.array(incidence, mask=~on_body))

        def lambert_law(self, body_name: str) -> polymath.Scalar:
            """Return per-sample Lambert reflectance masked off the target."""
            del body_name
            vv, uu = _grid(self._mg)
            on_body, _, cos_inc = _sphere_geometry(target, vv, uu)
            lam = np.clip(cos_inc, 0.0, None)
            return polymath.Scalar(np.ma.array(lam, mask=~on_body))

        def resolution(self, body_name: str) -> polymath.Scalar:
            """Return the constant km/px scale masked off the target."""
            del body_name
            vv, uu = _grid(self._mg)
            on_body, _, _ = _sphere_geometry(target, vv, uu)
            res = np.full(on_body.shape, target.km_per_px, dtype=np.float64)
            return polymath.Scalar(np.ma.array(res, mask=~on_body))

        def where_in_front(self, event_key: str, surface_key: str) -> polymath.Scalar:
            """Return the occluder silhouette where it sits in front of the target."""
            del surface_key
            vv, uu = _grid(self._mg)
            if event_key.upper() == _OCCLUDER:
                on_occ, _, _ = _sphere_geometry(occluder, vv, uu)
                return polymath.Scalar(on_occ.astype(np.float64))
            return polymath.Scalar(np.zeros(vv.shape, dtype=np.float64))

    return _TwoBodyBackplane


def _make_obs(
    target: _Sphere,
    *,
    data_shape: tuple[int, int] = (100, 100),
    margin: int = 10,
    phase_deg: float = 30.0,
    target_range_km: float = 1.0e6,
) -> tuple[FakeObs, dict[str, Any]]:
    """Build a ``FakeObs`` plus the target sphere's inventory record.

    Parameters:
        target: Target sphere geometry (drives the inventory bounding box).
        data_shape: Sensor-area ``(rows, cols)`` shape.
        margin: Extfov margin applied to both axes.
        phase_deg: Centre phase angle reported by the geometry backplane.
        target_range_km: Subject range recorded in the inventory.
    """
    center_v, center_u = target.center_vu
    r = target.radius_px
    inventory = {
        'u_min_unclipped': int(np.floor(center_u - r)),
        'u_max_unclipped': int(np.ceil(center_u + r)),
        'v_min_unclipped': int(np.floor(center_v - r)),
        'v_max_unclipped': int(np.ceil(center_v + r)),
        'u_pixel_size': 2.0 * r,
        'v_pixel_size': 2.0 * r,
        'range': target_range_km,
    }
    geometry_body = BodyBackplaneData(
        body_mask=np.ones((1, 1), dtype=bool),
        incidence_rad=np.zeros((1, 1), dtype=np.float64),
        center_phase_rad=math.radians(phase_deg),
    )
    obs = FakeObs(
        data=np.zeros(data_shape, dtype=np.float64),
        extfov_margin_vu=(margin, margin),
        closest_planet='SATURN',
        ext_bp=FakeBackplane(per_body={_TARGET: geometry_body}),
        inventory_records={_TARGET: inventory},
    )
    return obs, inventory


def _make_model(
    monkeypatch: pytest.MonkeyPatch,
    target: _Sphere,
    occluder: _Sphere | None,
    *,
    occluder_range_km: float = 5.0e5,
    target_range_km: float = 1.0e6,
    phase_deg: float = 30.0,
) -> tuple[NavModelBody, FakeObs]:
    """Build a ``NavModelBody`` wired to the analytic two-body backplane.

    Parameters:
        monkeypatch: Pytest monkeypatch fixture patching the module's oops
            ``Meshgrid`` / ``Backplane`` names.
        target: Sphere the model navigates.
        occluder: Nearer sphere wired in as a sibling, or ``None`` for the
            single-body baseline.
        occluder_range_km: Sibling range recorded on the wiring; strictly
            nearer than ``target_range_km`` unless overridden.
        target_range_km: Subject range of the navigated body.
        phase_deg: Centre phase angle reported by the geometry backplane.
    """
    obs, inventory = _make_obs(target, phase_deg=phase_deg, target_range_km=target_range_km)
    render_occluder = occluder if occluder is not None else target
    monkeypatch.setattr(nav_model_body_module, 'Meshgrid', FakeMeshgrid)
    monkeypatch.setattr(
        nav_model_body_module, 'Backplane', _two_body_backplane_class(target, render_occluder)
    )
    siblings = [] if occluder is None else [(_OCCLUDER, occluder_range_km)]
    model = NavModelBody(
        f'body:{_TARGET}',
        cast(Any, obs),
        _TARGET,
        inventory=inventory,
        siblings=siblings,
    )
    return model, obs


def _limb_feature(model: NavModelBody, obs: FakeObs) -> NavFeature:
    """Create the model and return its emitted LIMB_ARC feature."""
    model.create_model()
    features = model.to_features(bare_nav_context(cast(Any, obs)))
    return next(f for f in features if f.feature_type.name == 'LIMB_ARC')


def _disc_feature(model: NavModelBody, obs: FakeObs) -> NavFeature:
    """Create the model and return its emitted BODY_DISC feature."""
    model.create_model()
    features = model.to_features(bare_nav_context(cast(Any, obs)))
    return next(f for f in features if f.feature_type.name == 'BODY_DISC')


def _arc_fraction(feature: NavFeature) -> float:
    """Return the ``visible_arc_fraction`` of a limb / terminator feature."""
    flags = feature.flags
    assert isinstance(flags, LimbArcFlags | TerminatorArcFlags)
    return float(flags.visible_arc_fraction)


def _limb_vertices(feature: NavFeature) -> NDArrayFloatType:
    """Return the limb polyline vertices of a LIMB_ARC feature."""
    geometry = feature.geometry
    assert isinstance(geometry, LimbPolyline)
    return np.asarray(geometry.vertices_vu, dtype=np.float64)


# A target sphere fully lit and framed, plus a nearer sibling overlapping its
# +u half.  The overlap hides the right-side limb.
_TARGET_SPHERE = _Sphere((50.0, 50.0), 20.0)
_OCCLUDER_SPHERE = _Sphere((50.0, 62.0), 20.0)
# A lighter sibling clipping only the +u edge: the disc stays above the
# visible-lit gate so BODY_DISC is still emitted, but its template is trimmed.
_DISC_OCCLUDER = _Sphere((50.0, 75.0), 18.0)


def test_occluded_limb_fraction_drops(monkeypatch: pytest.MonkeyPatch) -> None:
    """A nearer sibling hides part of the limb; the fraction reports the loss."""
    alone_model, alone_obs = _make_model(monkeypatch, _TARGET_SPHERE, None)
    alone = _limb_feature(alone_model, alone_obs)
    occ_model, occ_obs = _make_model(monkeypatch, _TARGET_SPHERE, _OCCLUDER_SPHERE)
    occluded = _limb_feature(occ_model, occ_obs)
    assert _arc_fraction(occluded) < _arc_fraction(alone) - 0.1


def test_occluded_limb_reliability_drops(monkeypatch: pytest.MonkeyPatch) -> None:
    """The reduced arc fraction lowers the limb reliability."""
    alone_model, alone_obs = _make_model(monkeypatch, _TARGET_SPHERE, None)
    alone = _limb_feature(alone_model, alone_obs)
    occ_model, occ_obs = _make_model(monkeypatch, _TARGET_SPHERE, _OCCLUDER_SPHERE)
    occluded = _limb_feature(occ_model, occ_obs)
    assert occluded.reliability < alone.reliability


def test_occluded_limb_vertices_leave_the_polyline(monkeypatch: pytest.MonkeyPatch) -> None:
    """No surviving limb vertex sits inside the nearer sibling's silhouette."""
    model, obs = _make_model(monkeypatch, _TARGET_SPHERE, _OCCLUDER_SPHERE)
    occluded = _limb_feature(model, obs)
    vertices = _limb_vertices(occluded)
    center_v = _OCCLUDER_SPHERE.center_vu[0] + obs.extfov_margin_v
    center_u = _OCCLUDER_SPHERE.center_vu[1] + obs.extfov_margin_u
    dist = np.hypot(vertices[:, 0] - center_v, vertices[:, 1] - center_u)
    assert int(np.count_nonzero(dist < _OCCLUDER_SPHERE.radius_px - 1.0)) == 0


def test_farther_sibling_does_not_occlude(monkeypatch: pytest.MonkeyPatch) -> None:
    """A sibling with a larger range hides nothing of this body's limb."""
    alone_model, alone_obs = _make_model(monkeypatch, _TARGET_SPHERE, None)
    alone = _limb_feature(alone_model, alone_obs)
    behind_model, behind_obs = _make_model(
        monkeypatch, _TARGET_SPHERE, _OCCLUDER_SPHERE, occluder_range_km=2.0e6
    )
    behind = _limb_feature(behind_model, behind_obs)
    assert _arc_fraction(behind) == pytest.approx(_arc_fraction(alone))


def test_disc_template_trimmed_of_occluded_pixels(monkeypatch: pytest.MonkeyPatch) -> None:
    """The BODY_DISC template mask excludes pixels the nearer body hides."""
    model, obs = _make_model(monkeypatch, _TARGET_SPHERE, _DISC_OCCLUDER)
    disc = _disc_feature(model, obs)
    v_min, u_min, _v_max, _u_max = disc.geometry.bbox_extfov_vu
    template_mask = np.asarray(disc.template_mask, dtype=bool)
    vs, us = np.where(template_mask)
    center_v = _DISC_OCCLUDER.center_vu[0] + obs.extfov_margin_v - v_min
    center_u = _DISC_OCCLUDER.center_vu[1] + obs.extfov_margin_u - u_min
    dist = np.hypot(vs - center_v, us - center_u)
    assert int(np.count_nonzero(dist < _DISC_OCCLUDER.radius_px - 1.0)) == 0


def test_disc_template_img_zero_under_occluder(monkeypatch: pytest.MonkeyPatch) -> None:
    """The BODY_DISC template brightness is zero where the occluder hides it."""
    model, obs = _make_model(monkeypatch, _TARGET_SPHERE, _DISC_OCCLUDER)
    disc = _disc_feature(model, obs)
    v_min, u_min, _v_max, _u_max = disc.geometry.bbox_extfov_vu
    template_img = np.asarray(disc.template_img, dtype=np.float64)
    rows, cols = template_img.shape
    vv, uu = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')
    center_v = _DISC_OCCLUDER.center_vu[0] + obs.extfov_margin_v - v_min
    center_u = _DISC_OCCLUDER.center_vu[1] + obs.extfov_margin_u - u_min
    deep = np.hypot(vv - center_v, uu - center_u) < _DISC_OCCLUDER.radius_px - 3.0
    assert float(np.max(template_img[deep])) == 0.0


def test_occluded_disc_reliability_drops(monkeypatch: pytest.MonkeyPatch) -> None:
    """Occlusion lowers the emitted BODY_DISC feature's reliability."""
    alone_model, alone_obs = _make_model(monkeypatch, _TARGET_SPHERE, None)
    alone = _disc_feature(alone_model, alone_obs)
    occ_model, occ_obs = _make_model(monkeypatch, _TARGET_SPHERE, _DISC_OCCLUDER)
    occluded = _disc_feature(occ_model, occ_obs)
    assert occluded.reliability < alone.reliability


def test_occluded_terminator_fraction_drops(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sibling occlusion also reduces the terminator's visible-arc fraction."""
    tilt = math.radians(60.0)
    target = _Sphere((50.0, 50.0), 20.0, sun_vuz=(0.0, math.sin(tilt), math.cos(tilt)))
    # A sibling clipping the upper -u part of the terminator arc: it removes
    # some, not all, terminator vertices so the arc is still emitted.
    occluder = _Sphere((38.0, 44.0), 14.0)
    alone_model, alone_obs = _make_model(monkeypatch, target, None, phase_deg=60.0)
    alone_model.create_model()
    alone_features = alone_model.to_features(bare_nav_context(cast(Any, alone_obs)))
    alone_term = next(f for f in alone_features if f.feature_type.name == 'TERMINATOR_ARC')
    occ_model, occ_obs = _make_model(monkeypatch, target, occluder, phase_deg=60.0)
    occ_model.create_model()
    occ_features = occ_model.to_features(bare_nav_context(cast(Any, occ_obs)))
    occ_term = next(f for f in occ_features if f.feature_type.name == 'TERMINATOR_ARC')
    assert _arc_fraction(occ_term) < _arc_fraction(alone_term) - 0.05


def test_instances_for_obs_wires_nearer_siblings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each per-body model instance receives the other in-FOV bodies as siblings."""
    obs, target_inv = _make_obs(_TARGET_SPHERE)
    occ_inv = dict(target_inv)
    occ_inv['range'] = 5.0e5
    obs.inventory_records[_OCCLUDER] = occ_inv
    # Drive selection past the satellite-catalog lookup by naming both bodies
    # as the in-FOV set directly; the wiring under test is the sibling list.
    entries = [
        (_TARGET, obs.inventory_records[_TARGET]),
        (_OCCLUDER, occ_inv),
    ]
    monkeypatch.setattr(nav_model_body_module, 'bodies_in_extfov', lambda *_a, **_k: entries)
    models = cast(list[NavModelBody], NavModelBody.instances_for_obs(cast(Any, obs)))
    target_model = next(m for m in models if m._body_name == _TARGET)
    assert target_model._siblings == [(_OCCLUDER, 5.0e5)]


# ---------------------------------------------------------------------------
# The shared module-level occlusion helper
# ---------------------------------------------------------------------------


def _helper_backplane(target: _Sphere, occluder: _Sphere) -> Any:
    """Instantiate the two-body backplane stand-in over a one-sample-per-pixel box."""
    backplane_cls = _two_body_backplane_class(target, occluder)
    meshgrid = FakeMeshgrid(origin=(0.5, 0.5), limit=(79.5, 79.5), oversample=(1, 1), swap=True)
    return backplane_cls(None, meshgrid=meshgrid)


def test_occluder_helper_masks_the_nearer_sibling() -> None:
    """The shared helper returns the silhouette of a strictly nearer sibling."""
    bp = _helper_backplane(_TARGET_SPHERE, _DISC_OCCLUDER)
    mask = occluder_mask_for_body(
        bp, _TARGET, [(_OCCLUDER, 5.0e5)], 1.0e6, oversample_v=1, oversample_u=1
    )
    assert mask is not None
    assert bool(mask.any()) is True


def test_occluder_helper_ignores_a_farther_sibling() -> None:
    """A sibling behind the subject body occludes nothing."""
    bp = _helper_backplane(_TARGET_SPHERE, _DISC_OCCLUDER)
    mask = occluder_mask_for_body(
        bp, _TARGET, [(_OCCLUDER, 2.0e6)], 1.0e6, oversample_v=1, oversample_u=1
    )
    assert mask is None


def test_occluder_helper_returns_none_without_siblings() -> None:
    """A single-body scene costs nothing and reports no occlusion."""
    bp = _helper_backplane(_TARGET_SPHERE, _DISC_OCCLUDER)
    mask = occluder_mask_for_body(bp, _TARGET, [], 1.0e6, oversample_v=1, oversample_u=1)
    assert mask is None


def test_occluder_helper_degrades_on_a_backplane_failure() -> None:
    """A backplane that cannot answer leaves the caller's mask untrimmed."""

    class _RaisingBackplane:
        """Backplane stand-in whose depth test fails the way a bad scene does."""

        def where_in_front(self, sibling_name: str, body_name: str) -> Any:
            """Raise the way an unresolvable occlusion query does inside oops."""
            del sibling_name, body_name
            raise ValueError('cannot resolve occlusion for this scene')

    mask = occluder_mask_for_body(
        cast(Any, _RaisingBackplane()),
        _TARGET,
        [(_OCCLUDER, 5.0e5)],
        1.0e6,
        oversample_v=1,
        oversample_u=1,
    )
    assert mask is None
