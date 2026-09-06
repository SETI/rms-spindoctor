"""Spec-first tests for the ``NavModelRings`` rendering pipeline.

Covers ``_render`` (including the ``min_res_at_radius`` closure and the sparse
visibility pre-check) plus the parts of ``instances_for_obs`` / ``to_features``
/ ``to_annotations`` that depend on the rendered state.  The contract asserted
here comes from ``docs/dev_guide/dev_guide_navigation_models_ring.rst`` and the
module / method docstrings, not from the current implementation:

- One RING_EDGE per surviving catalog edge on the "edges resolve" path; a
  single composite RING_ANNULUS per planet on the "edges compress" path.
- Per-vertex radial sigma is the feature's catalog RMS (max across edges)
  divided by the radial km/px; the along-edge sigma is the 0.5 px constant.
- Straight-line polylines are flagged ``is_straight_line`` and receive the
  rank-1 reliability penalty.
- ``min_res_at_radius`` maps a radius to the minimum positive finite radial
  resolution along that radius's border pixels, or ``None`` when the radius
  is not resolvable in the image (no border pixels, zero, or non-finite).
- Skip paths: no ring-plane intersection, visible range beyond the catalog,
  every feature filtered out, empty catalog.

Backplane data is supplied by the table-driven ``tests.shims`` fakes; the
sparse pre-check's oops ``Meshgrid`` / ``Backplane`` names are monkeypatched
so the tests are hermetic (no SPICE, no holdings).
"""

from __future__ import annotations

import types
from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import polymath
import pytest
from tests.shims import FakeBackplane, FakeObs, RingBackplaneData, bare_nav_context

import spindoctor.nav_model.nav_model_rings as nav_model_rings_module
from spindoctor.annotation import Annotations
from spindoctor.config.config import Config
from spindoctor.feature.feature import NavFeature
from spindoctor.feature.geometry import RingAnnulusGeometry, RingEdgePolyline
from spindoctor.nav_model.nav_model_rings import NavModelRings
from spindoctor.nav_model.rings import RingFeature, RingFeatureFilter
from spindoctor.support.types import NDArrayBoolType

_SHAPE = (100, 100)
_TARGET = 'saturn:ring'
_EPOCH = '2007-01-01 00:00:00'


class _RingBackplane(FakeBackplane):
    """FakeBackplane plus the ``radial_mode`` call the edge renderer chains.

    The synthetic catalogs used here are circular (``ae = 0``) with no
    perturbations, for which the multi-mode radius equals the base ring
    radius, so the identity mapping is the spec-correct behaviour.
    """

    def radial_mode(
        self, key: tuple[Any, ...], mode: int, epoch: float, *args: Any, **kwargs: Any
    ) -> polymath.Scalar:
        """Return the ring-radius Scalar unchanged (circular base orbit)."""
        del mode, epoch, args, kwargs
        return self.ring_radius(key[1])


class _FakeSparseMeshgrid:
    """Meshgrid stand-in accepting the sparse pre-check's ``undersample``.

    Parameters:
        origin: ``(u, v)`` grid origin.
        limit: ``(u, v)`` grid limit.
        undersample: Per-axis undersampling steps.
        swap: Whether arrays are (v, u)-indexed.
    """

    def __init__(
        self,
        origin: tuple[float, float],
        limit: tuple[float, float],
        *,
        undersample: tuple[int, int] = (1, 1),
        swap: bool = False,
    ) -> None:
        self.origin = origin
        self.limit = limit
        self.undersample = undersample
        self.swap = swap

    @classmethod
    def for_fov(
        cls,
        fov: Any,
        *,
        origin: tuple[float, float],
        limit: tuple[float, float],
        undersample: tuple[int, int] = (1, 1),
        swap: bool = False,
    ) -> _FakeSparseMeshgrid:
        """Mirror the ``oops.Meshgrid.for_fov`` sparse-grid signature."""
        del fov
        return cls(origin, limit, undersample=undersample, swap=swap)


class _DelegatingSparseBackplane:
    """Backplane stand-in for the sparse pre-check.

    Delegates ``ring_radius`` to the observation's dense ``ext_bp`` so the
    sparse visibility verdict always matches the dense data the test wired.

    Parameters:
        obs: Observation carrying the fake ``ext_bp``.
        meshgrid: The strip's meshgrid.  Its origin and limit are what say
            which rows of the dense arrays this stand-in answers for; None
            answers for the whole frame.
    """

    def __init__(self, obs: Any, meshgrid: Any = None) -> None:
        self._obs = obs
        self._meshgrid = meshgrid

    def _rows(self) -> slice:
        """Rows of the dense arrays this stand-in was built to cover.

        The render evaluates its whole-frame backplanes a strip of rows at a
        time, so a stand-in for one strip has to answer for that strip and not
        for the frame. The strip is read back from the meshgrid's origin and
        limit, which is where the caller put it.
        """
        mg = self._meshgrid
        if mg is None or not hasattr(mg, 'origin') or not hasattr(mg, 'limit'):
            return slice(None)
        first = round(mg.origin[1] - 0.5) - self._obs.extfov_v_min
        last = round(mg.limit[1] - 0.5) - self._obs.extfov_v_min
        return slice(first, last + 1)

    def ring_radius(self, ring_target: str) -> Any:
        """Return the dense fake backplane's ring radius Scalar."""
        return self._obs.ext_bp.ring_radius(ring_target)

    def distance(self, ring_target: str, direction: str = 'dep') -> Any:
        """Return this strip's rows of the dense ring-plane distance."""
        return self._obs.ext_bp.distance(ring_target, direction=direction)[self._rows()]

    def where_inside_shadow(self, ring_target: str, planet: str) -> Any:
        """Return this strip's rows of the dense planet-shadow mask."""
        return self._obs.ext_bp.where_inside_shadow(ring_target, planet)[self._rows()]

    def where_in_front(self, near_target: str, far_target: str) -> Any:
        """Return this strip's rows of the dense occlusion mask."""
        return self._obs.ext_bp.where_in_front(near_target, far_target)[self._rows()]


def _ramp_ring(
    *,
    r0: float = 100_000.0,
    slope: float = 100.0,
    res: float = 10.0,
    mask: NDArrayBoolType | None = None,
    res_array: np.ndarray | None = None,
) -> RingBackplaneData:
    """Build ring backplane data with radius increasing linearly along u.

    ``radius(v, u) = r0 + slope * u`` produces straight vertical ring edges.
    The border for radius ``a`` is the pixel column within half a resolution
    of ``a``.

    Parameters:
        r0: Ring radius at the leftmost column (km).
        slope: Radius increase per pixel column (km/px).
        res: Constant radial resolution (km/px). The default sits below
            every planet's system-level annulus km/px threshold so scenes
            built from it exercise the per-edge emission path.
        mask: Optional ring-plane mask; defaults to all-True.
        res_array: Optional explicit per-pixel resolution array overriding
            the constant ``res``.
    """
    vv, uu = np.indices(_SHAPE, dtype=np.float64)
    del vv
    radius = r0 + slope * uu
    ring_mask = np.ones(_SHAPE, dtype=bool) if mask is None else mask
    resolutions = np.full(_SHAPE, res, dtype=np.float64) if res_array is None else res_array

    def _border(key: tuple[Any, ...], a: float) -> NDArrayBoolType:
        """Mark pixels whose ring radius is within half a resolution of ``a``."""
        del key
        border: NDArrayBoolType = (np.abs(radius - a) <= res / 2.0) & ring_mask
        return border

    return RingBackplaneData(
        ring_radius_km=radius,
        ring_mask=ring_mask,
        radial_resolution_km_px=resolutions,
        border_atop=_border,
    )


def _arc_ring(*, scale: float = 800.0) -> RingBackplaneData:
    """Build ring backplane data whose iso-radius contours are curved arcs.

    ``radius(v, u) = scale * hypot(v + 100, u - 50)`` centres the ring system
    100 rows above the frame so edges arc visibly across the FOV.

    Parameters:
        scale: km per pixel of radial distance.
    """
    vv, uu = np.indices(_SHAPE, dtype=np.float64)
    radius = scale * np.hypot(vv + 100.0, uu - 50.0)
    ring_mask = np.ones(_SHAPE, dtype=bool)

    def _border(key: tuple[Any, ...], a: float) -> NDArrayBoolType:
        """Mark pixels whose ring radius is within half a resolution of ``a``."""
        del key
        border: NDArrayBoolType = np.abs(radius - a) <= scale / 2.0
        return border

    return RingBackplaneData(
        ring_radius_km=radius,
        ring_mask=ring_mask,
        radial_resolution_km_px=np.full(_SHAPE, scale, dtype=np.float64),
        border_atop=_border,
    )


def _ringlet(
    inner_a: float,
    outer_a: float,
    *,
    rms_inner: float = 10.0,
    rms_outer: float = 20.0,
    **extra: Any,
) -> dict[str, Any]:
    """Return a two-edge RINGLET catalog entry.

    Parameters:
        inner_a: Inner-edge semi-major axis (km).
        outer_a: Outer-edge semi-major axis (km).
        rms_inner: Inner-edge RMS (km).
        rms_outer: Outer-edge RMS (km).
        **extra: Additional per-feature keys (e.g. ``end_date``).
    """
    entry: dict[str, Any] = {
        'feature_type': 'RINGLET',
        'name': 'TESTR',
        'inner_data': [
            {
                'mode': 1,
                'a': inner_a,
                'rms': rms_inner,
                'ae': 0.0,
                'long_peri': 0.0,
                'rate_peri': 0.0,
            }
        ],
        'outer_data': [
            {
                'mode': 1,
                'a': outer_a,
                'rms': rms_outer,
                'ae': 0.0,
                'long_peri': 0.0,
                'rate_peri': 0.0,
            }
        ],
    }
    entry.update(extra)
    return entry


def _gap_inner(a: float, *, rms: float = 8.0) -> dict[str, Any]:
    """Return a single-inner-edge GAP catalog entry.

    Parameters:
        a: Edge semi-major axis (km).
        rms: Edge RMS (km).
    """
    return {
        'feature_type': 'GAP',
        'name': 'TESTG',
        'inner_data': [
            {'mode': 1, 'a': a, 'rms': rms, 'ae': 0.0, 'long_peri': 0.0, 'rate_peri': 0.0}
        ],
    }


def _ring_config(features: dict[str, Any], *, planet_entry: dict[str, Any] | None = None) -> Config:
    """Return a ``Config`` whose SATURN ring catalog is exactly ``features``.

    The bundled defaults are loaded first (so every other section keeps its
    normal values), then ``rings.ring_features`` is replaced wholesale with a
    single synthetic SATURN entry.

    Parameters:
        features: Per-feature catalog entries keyed by feature name.
        planet_entry: Optional full replacement for the SATURN planet block
            (used by the config-validation tests); when given, ``features``
            is ignored.
    """
    config = Config()
    config.read_config()
    entry: dict[str, Any]
    if planet_entry is not None:
        entry = planet_entry
    else:
        entry = {
            'epoch': _EPOCH,
            'fade_width_pix': 10.0,
            'min_allowed_fade_width_pix': 2.0,
            'min_feature_pixels': 2.0,
            'features': features,
        }
    config._config_dict['rings'] = dict(config._config_dict['rings'])
    config._config_dict['rings']['ring_features'] = {'SATURN': entry}
    config._update_attrdicts()
    return config


def _make_obs(ring_data: RingBackplaneData) -> FakeObs:
    """Return a zero-margin FakeObs over the fake ring backplane.

    Parameters:
        ring_data: Ring backplane tables for the ``saturn:ring`` target.
    """
    return FakeObs(
        data=np.zeros(_SHAPE, dtype=np.float64),
        extfov_margin_vu=(0, 0),
        closest_planet='SATURN',
        ext_bp=_RingBackplane(per_ring={_TARGET: ring_data}),
        midtime=2.5e8,
    )


def _make_model(
    monkeypatch: pytest.MonkeyPatch,
    ring_data: RingBackplaneData,
    config: Config,
) -> tuple[NavModelRings, FakeObs]:
    """Build a ``NavModelRings`` wired to the fake backplanes.

    Parameters:
        monkeypatch: Pytest monkeypatch fixture (patches the module's oops
            ``Meshgrid`` / ``Backplane`` used by the sparse pre-check).
        ring_data: Ring backplane tables.
        config: Config carrying the synthetic catalog.
    """
    obs = _make_obs(ring_data)
    monkeypatch.setattr(nav_model_rings_module, 'Meshgrid', _FakeSparseMeshgrid)
    monkeypatch.setattr(nav_model_rings_module, 'Backplane', _DelegatingSparseBackplane)
    model = NavModelRings('rings:SATURN', cast(Any, obs), config=config)
    return model, obs


def _features(model: NavModelRings, obs: FakeObs) -> list[NavFeature]:
    """Run ``to_features`` with a bare context.

    Parameters:
        model: Model whose ``create_model`` already ran.
        obs: Observation the model was built against.
    """
    return model.to_features(bare_nav_context(cast(Any, obs)))


# ---------------------------------------------------------------------------
# _render: metadata and the edges-resolve path
# ---------------------------------------------------------------------------


def test_create_model_populates_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    """The documented metadata entries are populated for a surviving catalog."""
    config = _ring_config({'TESTR': _ringlet(103_000.0, 107_000.0)})
    model, _obs = _make_model(monkeypatch, _ramp_ring(), config)
    model.create_model()
    meta = model.metadata
    assert meta['planet'] == 'SATURN'
    assert meta['epoch'] == _EPOCH
    assert meta['feature_count'] == 1
    assert meta['features'] == [{'name': 'TESTR', 'type': 'RINGLET'}]
    assert meta['elapsed_time_sec'] is not None


def test_km_per_pixel_radial_is_mean_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    """The per-planet radial scale is the mean of the resolution backplane."""
    config = _ring_config({'TESTR': _ringlet(103_000.0, 107_000.0)})
    model, _obs = _make_model(monkeypatch, _ramp_ring(res=100.0), config)
    model.create_model()
    assert model._km_per_pixel_radial == pytest.approx(100.0)


def test_ring_edges_emitted_per_surviving_edge(monkeypatch: pytest.MonkeyPatch) -> None:
    """The edges-resolve path emits one RING_EDGE per catalog edge."""
    config = _ring_config({'TESTR': _ringlet(103_000.0, 107_000.0)})
    model, obs = _make_model(monkeypatch, _ramp_ring(), config)
    model.create_model()
    features = _features(model, obs)
    ids = sorted(f.feature_id for f in features)
    assert ids == ['ring_edge:SATURN:TESTR:IER', 'ring_edge:SATURN:TESTR:OER']
    assert {f.feature_type.name for f in features} == {'RING_EDGE'}


def test_edge_vertices_lie_on_catalog_radius(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each edge polyline sits on the pixel column of its catalog radius.

    With ``radius = 100000 + 100 * u``, the 103000 km inner edge maps to the
    u = 30 column and the 107000 km outer edge to u = 70.
    """
    config = _ring_config({'TESTR': _ringlet(103_000.0, 107_000.0)})
    model, obs = _make_model(monkeypatch, _ramp_ring(), config)
    model.create_model()
    by_id = {f.feature_id: f for f in _features(model, obs)}
    inner_geometry = by_id['ring_edge:SATURN:TESTR:IER'].geometry
    outer_geometry = by_id['ring_edge:SATURN:TESTR:OER'].geometry
    assert isinstance(inner_geometry, RingEdgePolyline)
    assert isinstance(outer_geometry, RingEdgePolyline)
    assert set(inner_geometry.vertices_vu[:, 1].tolist()) == {30.0}
    assert set(outer_geometry.vertices_vu[:, 1].tolist()) == {70.0}


def test_straight_edge_flag_and_rank1_penalty(monkeypatch: pytest.MonkeyPatch) -> None:
    """A straight polyline is flagged and gets the 0.7 rank-1 multiplier."""
    config = _ring_config({'TESTR': _ringlet(103_000.0, 107_000.0)})
    model, obs = _make_model(monkeypatch, _ramp_ring(), config)
    model.create_model()
    features = _features(model, obs)
    for feature in features:
        geometry = feature.geometry
        assert isinstance(geometry, RingEdgePolyline)
        assert geometry.is_straight_line is True
        # catalog_default (0.7) * visible_arc (1.0) * straight penalty (0.7)
        assert feature.reliability == pytest.approx(0.7 * 0.7)


def test_edge_sigma_radial_from_catalog_rms(monkeypatch: pytest.MonkeyPatch) -> None:
    """Radial sigma is the feature's max edge RMS over the radial km/px.

    The dev guide specifies the conservative maximum of the two edge RMS
    values (20 km here) divided by the per-image radial scale (10 km/px,
    below the Saturn annulus threshold so the per-edge path emits),
    broadcast across every vertex of both edge polylines.
    """
    config = _ring_config({'TESTR': _ringlet(103_000.0, 107_000.0, rms_inner=10.0, rms_outer=20.0)})
    model, obs = _make_model(monkeypatch, _ramp_ring(res=10.0), config)
    model.create_model()
    for feature in _features(model, obs):
        geometry = feature.geometry
        assert isinstance(geometry, RingEdgePolyline)
        assert bool(np.all(geometry.sigma_radial_per_vertex_px == pytest.approx(2.0)))


def test_edge_sigma_along_is_constant(monkeypatch: pytest.MonkeyPatch) -> None:
    """The along-edge sigma is the 0.5 px sampling-resolution constant."""
    config = _ring_config({'TESTR': _ringlet(103_000.0, 107_000.0)})
    model, obs = _make_model(monkeypatch, _ramp_ring(), config)
    model.create_model()
    for feature in _features(model, obs):
        geometry = feature.geometry
        assert isinstance(geometry, RingEdgePolyline)
        assert bool(np.all(geometry.sigma_along_edge_per_vertex_px == pytest.approx(0.5)))


def test_edge_normals_unit_length(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every per-vertex normal on an emitted edge polyline is unit length."""
    config = _ring_config({'TESTR': _ringlet(103_000.0, 107_000.0)})
    model, obs = _make_model(monkeypatch, _ramp_ring(), config)
    model.create_model()
    for feature in _features(model, obs):
        geometry = feature.geometry
        assert isinstance(geometry, RingEdgePolyline)
        norms = np.hypot(geometry.normals_vu[:, 0], geometry.normals_vu[:, 1])
        assert bool(np.allclose(norms, 1.0))


def test_curved_edge_not_flagged_straight(monkeypatch: pytest.MonkeyPatch) -> None:
    """An edge arcing across the FOV escapes the straight-line rank-1 flag."""
    # scale=20 keeps the scene below the Saturn annulus km/px threshold
    # while preserving the same 130 px contour curvature the default
    # geometry drew (2600 km at 20 km/px = 104000 km at 800 km/px).
    config = _ring_config({'TESTG': _gap_inner(2_600.0)})
    model, obs = _make_model(monkeypatch, _arc_ring(scale=20.0), config)
    model.create_model()
    features = _features(model, obs)
    assert len(features) == 1
    geometry = features[0].geometry
    assert isinstance(geometry, RingEdgePolyline)
    assert geometry.is_straight_line is False
    assert features[0].reliability == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# to_features: the edges-compress (annulus) path
# ---------------------------------------------------------------------------


def test_annulus_forced_at_low_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    """Above the per-planet km/px threshold a single RING_ANNULUS is emitted.

    A 1500 km/px scene sits far above Saturn's
    ``feature_emission.ring_annulus`` threshold, so every edge must
    collapse into one composite annulus.
    """
    config = _ring_config({'TESTR': _ringlet(80_000.0, 110_000.0)})
    ring = _ramp_ring(r0=50_000.0, slope=1500.0, res=1500.0)
    model, obs = _make_model(monkeypatch, ring, config)
    model.create_model()
    features = _features(model, obs)
    assert len(features) == 1
    assert features[0].feature_type.name == 'RING_ANNULUS'


def test_annulus_template_cropped_to_bbox(monkeypatch: pytest.MonkeyPatch) -> None:
    """The annulus template is a postage stamp sized to its bounding box."""
    config = _ring_config({'TESTR': _ringlet(80_000.0, 110_000.0)})
    ring = _ramp_ring(r0=50_000.0, slope=1500.0, res=1500.0)
    model, obs = _make_model(monkeypatch, ring, config)
    model.create_model()
    annulus = _features(model, obs)[0]
    geometry = annulus.geometry
    assert isinstance(geometry, RingAnnulusGeometry)
    v_min, u_min, v_max, u_max = geometry.bbox_extfov_vu
    assert annulus.template_img is not None
    assert annulus.template_mask is not None
    assert annulus.template_img.shape == (v_max - v_min, u_max - u_min)
    assert annulus.template_mask.shape == (v_max - v_min, u_max - u_min)


def test_annulus_carries_constituent_count_and_center(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The composite annulus reports its constituent edges and frame centre."""
    config = _ring_config({'TESTR': _ringlet(80_000.0, 110_000.0)})
    ring = _ramp_ring(r0=50_000.0, slope=1500.0, res=1500.0)
    model, obs = _make_model(monkeypatch, ring, config)
    model.create_model()
    annulus = _features(model, obs)[0]
    assert annulus.flags is not None
    assert getattr(annulus.flags, 'constituent_edge_count', None) == 2
    geometry = annulus.geometry
    assert isinstance(geometry, RingAnnulusGeometry)
    assert geometry.predicted_center_vu == (50.0, 50.0)


# ---------------------------------------------------------------------------
# _render: skip paths
# ---------------------------------------------------------------------------


def test_no_ring_plane_intersection_skips_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """An all-masked ring radius yields an empty model and no features."""
    config = _ring_config({'TESTR': _ringlet(103_000.0, 107_000.0)})
    ring = _ramp_ring(mask=np.zeros(_SHAPE, dtype=bool))
    model, obs = _make_model(monkeypatch, ring, config)
    model.create_model()
    assert 'feature_count' not in model.metadata
    assert model._render_results == []
    assert _features(model, obs) == []


def test_visible_range_beyond_catalog_skips(monkeypatch: pytest.MonkeyPatch) -> None:
    """A FOV whose radii all exceed the catalog max extent renders nothing."""
    config = _ring_config({'TESTR': _ringlet(50_000.0, 60_000.0)})
    model, obs = _make_model(monkeypatch, _ramp_ring(), config)
    model.create_model()
    assert model._render_results == []
    assert _features(model, obs) == []


def test_all_features_date_filtered_out(monkeypatch: pytest.MonkeyPatch) -> None:
    """A catalog whose features expired before the image renders nothing."""
    expired = _ringlet(103_000.0, 107_000.0, end_date='2000-01-01 00:00:00')
    config = _ring_config({'TESTR': expired})
    model, obs = _make_model(monkeypatch, _ramp_ring(), config)
    model.create_model()
    assert model._render_results == []
    assert _features(model, obs) == []


def test_empty_features_dict_yields_empty_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty per-planet feature dict is tolerated and renders nothing."""
    config = _ring_config({})
    model, obs = _make_model(monkeypatch, _ramp_ring(), config)
    model.create_model()
    assert model._render_results == []
    assert _features(model, obs) == []


def test_edge_on_single_row_ring_plane(monkeypatch: pytest.MonkeyPatch) -> None:
    """A one-row ring-plane projection still yields per-edge features.

    Simulates an edge-on viewing geometry where the ring plane collapses to a
    single pixel row; the surviving polylines are confined to that row.
    """
    row_mask = np.zeros(_SHAPE, dtype=bool)
    row_mask[50, :] = True
    config = _ring_config({'TESTR': _ringlet(103_000.0, 107_000.0)})
    model, obs = _make_model(monkeypatch, _ramp_ring(mask=row_mask), config)
    model.create_model()
    features = _features(model, obs)
    assert len(features) == 2
    for feature in features:
        geometry = feature.geometry
        assert isinstance(geometry, RingEdgePolyline)
        assert set(geometry.vertices_vu[:, 0].tolist()) == {50.0}


# ---------------------------------------------------------------------------
# _render: shadow handling
# ---------------------------------------------------------------------------


def test_shadow_removal_zeroes_shadowed_pixels(monkeypatch: pytest.MonkeyPatch) -> None:
    """With ``remove_planet_shadow`` the rendered model is zero in shadow."""
    ring = _ramp_ring()
    shadow = np.zeros(_SHAPE, dtype=bool)
    shadow[:, 60:] = True
    ring.shadow_mask = shadow
    config = _ring_config({'TESTR': _ringlet(103_000.0, 107_000.0)})
    model, _obs = _make_model(monkeypatch, ring, config)
    model.create_model()
    assert len(model._render_results) == 1
    _feat, model_img, model_mask, _unc, _edges = model._render_results[0]
    assert float(model_img[:, 60:].max()) == 0.0
    assert not bool(model_mask[:, 60:].any())
    assert bool(model_mask[:, :60].any())


# ---------------------------------------------------------------------------
# min_res_at_radius: documented closure contract
# ---------------------------------------------------------------------------


def _capture_min_res(
    monkeypatch: pytest.MonkeyPatch,
    ring_data: RingBackplaneData,
    config: Config,
) -> Any:
    """Run ``_render`` far enough to capture its ``min_res_at_radius`` closure.

    The module's ``RingFeatureFilter`` is replaced with a capturing subclass
    whose ``filter`` returns no survivors, so ``_render`` stops immediately
    after the closure is built and none of the later validation runs.

    Parameters:
        monkeypatch: Pytest monkeypatch fixture.
        ring_data: Ring backplane tables.
        config: Config carrying the synthetic catalog.
    """
    captured: dict[str, Any] = {}

    class _CapturingFilter(RingFeatureFilter):
        """RingFeatureFilter that records the closure and survives nothing."""

        def __init__(self, **kwargs: Any) -> None:
            captured['fn'] = kwargs['min_res_at_radius']
            super().__init__(**kwargs)

        def filter(self, features: Sequence[RingFeature]) -> list[RingFeature]:
            """Return no survivors so ``_render`` stops after the capture."""
            del features
            return []

    monkeypatch.setattr(nav_model_rings_module, 'RingFeatureFilter', _CapturingFilter)
    model, _obs = _make_model(monkeypatch, ring_data, config)
    model.create_model()
    return captured['fn']


def test_min_res_at_radius_returns_min_resolution_when_covered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A radius with border pixels maps to the minimum radial resolution."""
    config = _ring_config({'TESTR': _ringlet(103_000.0, 107_000.0)})
    fn = _capture_min_res(monkeypatch, _ramp_ring(res=100.0), config)
    assert fn(103_000.0) == pytest.approx(100.0)


def test_min_res_at_radius_at_exact_coverage_edges(monkeypatch: pytest.MonkeyPatch) -> None:
    """Radii exactly at the visible minimum and maximum are still resolvable.

    The ramp covers [100000, 109900] km; queries at both extremes hit the
    first / last pixel columns and must return the resolution, not ``None``.
    """
    config = _ring_config({'TESTR': _ringlet(103_000.0, 107_000.0)})
    fn = _capture_min_res(monkeypatch, _ramp_ring(res=100.0), config)
    assert fn(100_000.0) == pytest.approx(100.0)
    assert fn(109_900.0) == pytest.approx(100.0)


def test_min_res_at_radius_none_outside_coverage(monkeypatch: pytest.MonkeyPatch) -> None:
    """Radii with no border pixels in the image map to ``None``."""
    config = _ring_config({'TESTR': _ringlet(103_000.0, 107_000.0)})
    fn = _capture_min_res(monkeypatch, _ramp_ring(res=100.0), config)
    assert fn(150_000.0) is None
    assert fn(90_000.0) is None


def test_min_res_at_radius_none_for_zero_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    """A radius whose border pixels all have zero resolution maps to ``None``."""
    res_array = np.full(_SHAPE, 100.0, dtype=np.float64)
    res_array[:, 30] = 0.0  # the u = 30 column carries radius 103000 km
    config = _ring_config({'TESTR': _ringlet(103_000.0, 107_000.0)})
    fn = _capture_min_res(monkeypatch, _ramp_ring(res=100.0, res_array=res_array), config)
    assert fn(103_000.0) is None


def test_min_res_at_radius_none_for_nonfinite_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A radius whose border resolutions are all non-finite maps to ``None``."""
    res_array = np.full(_SHAPE, 100.0, dtype=np.float64)
    res_array[:, 30] = np.inf
    config = _ring_config({'TESTR': _ringlet(103_000.0, 107_000.0)})
    fn = _capture_min_res(monkeypatch, _ramp_ring(res=100.0, res_array=res_array), config)
    assert fn(103_000.0) is None


# ---------------------------------------------------------------------------
# _render: config validation
# ---------------------------------------------------------------------------


def _entry_without(*keys: str, **overrides: Any) -> dict[str, Any]:
    """Return a SATURN planet block with keys removed / overridden.

    Parameters:
        *keys: Keys deleted from the baseline block.
        **overrides: Key / value pairs written over the baseline block.
    """
    entry: dict[str, Any] = {
        'epoch': _EPOCH,
        'fade_width_pix': 10.0,
        'min_allowed_fade_width_pix': 2.0,
        'min_feature_pixels': 2.0,
        'features': {'TESTR': _ringlet(103_000.0, 107_000.0)},
    }
    for key in keys:
        del entry[key]
    entry.update(overrides)
    return entry


def test_missing_epoch_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A planet block without an epoch fails fast."""
    config = _ring_config({}, planet_entry=_entry_without('epoch'))
    model, _obs = _make_model(monkeypatch, _ramp_ring(), config)
    with pytest.raises(ValueError, match='No epoch configured for planet SATURN'):
        model.create_model()


def test_non_string_epoch_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-string epoch fails fast with a type-naming message."""
    config = _ring_config({}, planet_entry=_entry_without(epoch=12345))
    model, _obs = _make_model(monkeypatch, _ramp_ring(), config)
    with pytest.raises(ValueError, match=r'epoch for planet .SATURN. must be a string'):
        model.create_model()


def test_invalid_epoch_string_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unparseable epoch string fails fast naming the bad value."""
    config = _ring_config({}, planet_entry=_entry_without(epoch='not a date'))
    model, _obs = _make_model(monkeypatch, _ramp_ring(), config)
    with pytest.raises(ValueError, match='is not a valid UTC'):
        model.create_model()


def test_missing_features_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A planet block without a features mapping fails fast."""
    config = _ring_config({}, planet_entry=_entry_without('features'))
    model, _obs = _make_model(monkeypatch, _ramp_ring(), config)
    with pytest.raises(ValueError, match='Missing required ring configuration key "features"'):
        model.create_model()


def test_non_dict_features_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A features value that is not a mapping fails fast."""
    config = _ring_config({}, planet_entry=_entry_without(features=['not', 'a', 'dict']))
    model, _obs = _make_model(monkeypatch, _ramp_ring(), config)
    with pytest.raises(ValueError, match=r'"features" for planet .SATURN. must be a dict'):
        model.create_model()


def test_non_dict_feature_entry_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A per-feature entry that is not a mapping fails fast naming the key."""
    config = _ring_config({'BAD': 'not a dict'})
    model, _obs = _make_model(monkeypatch, _ramp_ring(), config)
    with pytest.raises(ValueError, match=r"features.'BAD' must be a dict"):
        model.create_model()


def test_non_dict_planet_entry_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A planet block that is not a mapping fails fast."""
    config = Config()
    config.read_config()
    config._config_dict['rings'] = dict(config._config_dict['rings'])
    config._config_dict['rings']['ring_features'] = {'SATURN': 'not a dict'}
    config._update_attrdicts()
    model, _obs = _make_model(monkeypatch, _ramp_ring(), config)
    with pytest.raises(ValueError, match=r'entry for planet .SATURN. must be'):
        model.create_model()


# ---------------------------------------------------------------------------
# instances_for_obs
# ---------------------------------------------------------------------------


def test_instances_for_obs_returns_model_for_cataloged_planet() -> None:
    """A planet with a bundled ring catalog gets exactly one model."""
    obs = _make_obs(_ramp_ring())
    instances = NavModelRings.instances_for_obs(cast(Any, obs))
    assert [inst.name for inst in instances] == ['rings:SATURN']


def test_instances_for_obs_skips_simulated_obs() -> None:
    """Simulated observations use the sim sibling, so no instances here."""
    obs = types.SimpleNamespace(is_simulated=True)
    assert NavModelRings.instances_for_obs(cast(Any, obs)) == []


def test_instances_for_obs_requires_closest_planet() -> None:
    """No closest planet means no ring model."""
    obs = types.SimpleNamespace(closest_planet=None)
    assert NavModelRings.instances_for_obs(cast(Any, obs)) == []


def test_instances_for_obs_requires_extfov_surface() -> None:
    """An obs without the extfov / backplane surface gets no ring model."""
    obs = types.SimpleNamespace(closest_planet='SATURN')
    assert NavModelRings.instances_for_obs(cast(Any, obs)) == []


def test_instances_for_obs_requires_cataloged_planet() -> None:
    """A planet absent from the ring catalog gets no ring model."""
    obs = FakeObs(
        data=np.zeros(_SHAPE, dtype=np.float64),
        extfov_margin_vu=(0, 0),
        closest_planet='PLUTO',
        ext_bp=_RingBackplane(per_ring={}),
    )
    assert NavModelRings.instances_for_obs(cast(Any, obs)) == []


# ---------------------------------------------------------------------------
# to_annotations
# ---------------------------------------------------------------------------


def test_to_annotations_empty_before_create_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without a rendered model the annotation collection is empty."""
    config = _ring_config({'TESTR': _ringlet(103_000.0, 107_000.0)})
    model, obs = _make_model(monkeypatch, _ramp_ring(), config)
    annotations = model.to_annotations(bare_nav_context(cast(Any, obs)))
    assert isinstance(annotations, Annotations)
    assert annotations.annotations == []


def test_to_annotations_emits_edge_overlays(monkeypatch: pytest.MonkeyPatch) -> None:
    """After rendering, each surviving edge contributes an annotation."""
    config = _ring_config({'TESTR': _ringlet(103_000.0, 107_000.0)})
    model, obs = _make_model(monkeypatch, _ramp_ring(), config)
    model.create_model()
    annotations = model.to_annotations(bare_nav_context(cast(Any, obs)))
    assert len(annotations.annotations) == 2
