"""``RingEdgeNav`` reports how weakly a gently curved arc constrains its along-edge axis."""

from __future__ import annotations

import math

import numpy as np
import pytest
from tests.spindoctor.nav_technique.conftest import (
    CirclePolylineFactory,
    DiscImageFactory,
    NavContextFactory,
    NavFeatureFactory,
)

from spindoctor.nav_technique.diagnostics import RingEdgeDiagnostics
from spindoctor.nav_technique.nav_technique_ring_edge import RingEdgeNav
from spindoctor.nav_technique.ring_edge_geometry import is_rank_1
from spindoctor.nav_technique.ring_edge_stats import _matched_edge_strength_fraction
from spindoctor.nav_technique.technique_result import NavTechniqueResult

_SHAPE = (240, 240)
# A disc of radius 3000 px whose center lies far off the frame toward -v, -u at
# 30 degrees from +v: its edge crosses the frame as a staircase arc about 300 px long.
_ARC_RADIUS = 3000.0
_ARC_DIRECTION = np.array([math.cos(math.radians(30.0)), math.sin(math.radians(30.0))])
_ARC_CENTER = (120.0 - _ARC_RADIUS * _ARC_DIRECTION[0], 120.0 - _ARC_RADIUS * _ARC_DIRECTION[1])


def _gentle_arc() -> tuple[np.ndarray, np.ndarray]:
    """Model vertices and outward normals of the arc, one per pixel, inside the frame."""
    angles = math.atan2(_ARC_DIRECTION[1], _ARC_DIRECTION[0]) + np.arange(-200, 200) / _ARC_RADIUS
    normals = np.stack([np.cos(angles), np.sin(angles)], axis=-1)
    vertices = np.array(_ARC_CENTER) + _ARC_RADIUS * normals
    inside = np.all((vertices > 10.0) & (vertices < _SHAPE[0] - 10.0), axis=1)
    return vertices[inside], normals[inside]


def _staircase_normals(count: int) -> np.ndarray:
    """Pixel-neighbor-like normals for the arc: every third along 45 degrees, the rest along +v.

    Their mean points near the arc's true 30-degree normal, as an edge mask's
    pixel-neighbor normals do, but none of them is that normal.
    """
    angles = np.where(np.arange(count) % 3 == 0, math.radians(45.0), 0.0)
    return np.stack([np.cos(angles), np.sin(angles)], axis=-1)


def _navigate_arc(
    normals: np.ndarray,
    pixel_normals: np.ndarray | None,
    make_ring_feature: NavFeatureFactory,
    make_nav_context: NavContextFactory,
    image: np.ndarray,
    *,
    shape_sigma_px: float | None = None,
    min_edge_strength_fraction: float | None = None,
) -> NavTechniqueResult:
    """Run RingEdgeNav on the gentle arc with the given model and pixel normals."""
    vertices, _outward = _gentle_arc()
    feature = make_ring_feature(
        'gentle',
        vertices=vertices,
        outward_normals=normals,
        is_straight_line=False,
        pixel_normals=pixel_normals,
    )
    technique = RingEdgeNav()
    if shape_sigma_px is not None:
        technique._shape_sigma_px = shape_sigma_px
    if min_edge_strength_fraction is not None:
        technique._spurious_min_edge_strength_fraction = min_edge_strength_fraction
    return technique.navigate([feature], make_nav_context(image))


def _along_edge_sigma(result: NavTechniqueResult) -> float:
    """The reported sigma along the arc's middle tangent."""
    tangent = np.array([-_ARC_DIRECTION[1], _ARC_DIRECTION[0]])
    covariance = np.asarray(result.covariance_px2)[:2, :2]
    return float(np.sqrt(tangent @ covariance @ tangent))


def test_gentle_arc_reports_a_large_along_edge_sigma(
    disc_image: DiscImageFactory,
    make_ring_feature: NavFeatureFactory,
    make_nav_context: NavContextFactory,
) -> None:
    """A low-curvature arc's covariance is several pixels along the edge, not tenths."""
    image = disc_image(_SHAPE, _ARC_CENTER, _ARC_RADIUS)
    vertices, outward = _gentle_arc()
    feature = make_ring_feature(
        'gentle', vertices=vertices, outward_normals=outward, is_straight_line=False
    )
    result = RingEdgeNav().navigate([feature], make_nav_context(image))
    tangent = np.array([-_ARC_DIRECTION[1], _ARC_DIRECTION[0]])
    covariance = np.asarray(result.covariance_px2)[:2, :2]
    assert float(np.sqrt(tangent @ covariance @ tangent)) > 3.0


def test_gentle_arc_keeps_its_radial_sigma_small(
    disc_image: DiscImageFactory,
    make_ring_feature: NavFeatureFactory,
    make_nav_context: NavContextFactory,
) -> None:
    """The same arc still constrains its radial axis to well under a pixel."""
    image = disc_image(_SHAPE, _ARC_CENTER, _ARC_RADIUS)
    vertices, outward = _gentle_arc()
    feature = make_ring_feature(
        'gentle', vertices=vertices, outward_normals=outward, is_straight_line=False
    )
    result = RingEdgeNav().navigate([feature], make_nav_context(image))
    covariance = np.asarray(result.covariance_px2)[:2, :2]
    assert float(np.sqrt(_ARC_DIRECTION @ covariance @ _ARC_DIRECTION)) < 0.5


def test_full_ring_reports_a_small_sigma_on_both_axes(
    disc_image: DiscImageFactory,
    circle_polyline: CirclePolylineFactory,
    make_ring_feature: NavFeatureFactory,
    make_nav_context: NavContextFactory,
) -> None:
    """A strongly curved edge constrains both axes, so neither sigma is inflated."""
    image = disc_image((200, 200), (100.0, 100.0), 32.0)
    vertices, outward = circle_polyline((100.0, 100.0), 32.0, 200)
    feature = make_ring_feature(
        'ring', vertices=vertices, outward_normals=outward, is_straight_line=False
    )
    result = RingEdgeNav().navigate([feature], make_nav_context(image))
    assert float(np.sqrt(np.linalg.eigvalsh(result.covariance_px2[:2, :2])[-1])) < 0.5


def test_gentle_arc_records_the_shape_term(
    disc_image: DiscImageFactory,
    make_ring_feature: NavFeatureFactory,
    make_nav_context: NavContextFactory,
) -> None:
    """The diagnostics carry the shape term's major-axis sigma, about sqrt(12) R / L times it."""
    image = disc_image(_SHAPE, _ARC_CENTER, _ARC_RADIUS)
    vertices, outward = _gentle_arc()
    feature = make_ring_feature(
        'gentle', vertices=vertices, outward_normals=outward, is_straight_line=False
    )
    technique = RingEdgeNav()
    result = technique.navigate([feature], make_nav_context(image))
    assert isinstance(result.diagnostics, RingEdgeDiagnostics)
    length = float(np.linalg.norm(vertices[-1] - vertices[0]))
    expected = technique._shape_sigma_px * math.sqrt(12.0) * _ARC_RADIUS / length
    assert result.diagnostics.sigma_shape_major_px == pytest.approx(expected, rel=0.05)


def test_edge_strength_scores_read_the_pixel_normals(
    disc_image: DiscImageFactory,
    make_ring_feature: NavFeatureFactory,
    make_nav_context: NavContextFactory,
) -> None:
    """Pixel normals running along the edge make the seed check reject the overlap seed.

    With the radial normals the fit stays at the true (zero) offset; the
    edge-strength scores read the pixel normals instead, and normals along the
    edge see a stronger "edge" elsewhere, so the seed moves several pixels off.
    """
    image = disc_image(_SHAPE, _ARC_CENTER, _ARC_RADIUS)
    vertices, outward = _gentle_arc()
    along_edge = np.stack([-outward[:, 1], outward[:, 0]], axis=-1)
    feature = make_ring_feature(
        'gentle',
        vertices=vertices,
        outward_normals=outward,
        is_straight_line=False,
        pixel_normals=along_edge,
    )
    result = RingEdgeNav().navigate([feature], make_nav_context(image))
    assert math.hypot(*result.offset_px) > 5.0


def test_shape_term_is_added_to_the_reported_covariance(
    disc_image: DiscImageFactory,
    make_ring_feature: NavFeatureFactory,
    make_nav_context: NavContextFactory,
) -> None:
    """The covariance grows by exactly the recorded shape term over a run without it."""
    image = disc_image(_SHAPE, _ARC_CENTER, _ARC_RADIUS)
    _vertices, outward = _gentle_arc()
    with_term = _navigate_arc(outward, None, make_ring_feature, make_nav_context, image)
    without = _navigate_arc(
        outward, None, make_ring_feature, make_nav_context, image, shape_sigma_px=0.0
    )
    added = (
        np.asarray(with_term.covariance_px2)[:2, :2] - np.asarray(without.covariance_px2)[:2, :2]
    )
    assert isinstance(with_term.diagnostics, RingEdgeDiagnostics)
    assert math.sqrt(float(np.linalg.eigvalsh(added)[-1])) == pytest.approx(
        with_term.diagnostics.sigma_shape_major_px, rel=1e-6
    )


def test_covariance_is_formed_on_the_radial_normals(
    disc_image: DiscImageFactory,
    make_ring_feature: NavFeatureFactory,
    make_nav_context: NavContextFactory,
) -> None:
    """Staircase pixel normals do not shrink the statistical along-edge sigma.

    With the shape term off, the along-edge sigma of a fit that carries staircase
    pixel normals matches that of a fit whose pixel normals are the radial ones,
    several pixels; formed on the staircase normals it would be a fraction of one.
    """
    image = disc_image(_SHAPE, _ARC_CENTER, _ARC_RADIUS)
    _vertices, outward = _gentle_arc()
    staircase = _staircase_normals(outward.shape[0])
    with_staircase = _navigate_arc(
        outward, staircase, make_ring_feature, make_nav_context, image, shape_sigma_px=0.0
    )
    radial_only = _navigate_arc(
        outward, None, make_ring_feature, make_nav_context, image, shape_sigma_px=0.0
    )
    assert _along_edge_sigma(with_staircase) == pytest.approx(
        _along_edge_sigma(radial_only), rel=0.01
    )


def test_fit_offset_depends_only_on_the_pixel_normals(
    disc_image: DiscImageFactory,
    make_ring_feature: NavFeatureFactory,
    make_nav_context: NavContextFactory,
) -> None:
    """The seed and the gradient-ridge pass read the pixel normals alone.

    A feature whose model normals are radial and whose pixel normals are a
    staircase converges exactly where a feature carrying the staircase as both
    does, and not where the radial normals alone take the fit.
    """
    image = disc_image(_SHAPE, _ARC_CENTER, _ARC_RADIUS)
    _vertices, outward = _gentle_arc()
    staircase = _staircase_normals(outward.shape[0])
    mixed = _navigate_arc(outward, staircase, make_ring_feature, make_nav_context, image)
    staircase_only = _navigate_arc(staircase, None, make_ring_feature, make_nav_context, image)
    assert mixed.offset_px == pytest.approx(staircase_only.offset_px, abs=1e-12)


def test_faint_edge_veto_reads_the_pixel_normals(
    disc_image: DiscImageFactory,
    make_ring_feature: NavFeatureFactory,
    make_nav_context: NavContextFactory,
) -> None:
    """Pixel normals 70 degrees off the radial ones score the edge faint enough to veto.

    The floor is set between the strengths the two sets of normals read at the
    true offset, so the fit is spurious only because the veto reads the pixel
    normals.
    """
    image = disc_image(_SHAPE, _ARC_CENTER, _ARC_RADIUS)
    vertices, outward = _gentle_arc()
    turned = np.arctan2(outward[:, 1], outward[:, 0]) + math.radians(70.0)
    pixel = np.stack([np.cos(turned), np.sin(turned)], axis=-1)
    context = make_nav_context(image)
    assert context.image_edge_dt_ext is not None
    assert context.image_gradient_vu_ext is not None
    valid = np.ones(image.shape, dtype=bool)
    radial_strength = _matched_edge_strength_fraction(
        context.image_edge_dt_ext,
        context.image_gradient_vu_ext,
        vertices,
        outward,
        (0.0, 0.0),
        valid_mask=valid,
    )
    pixel_strength = _matched_edge_strength_fraction(
        context.image_edge_dt_ext,
        context.image_gradient_vu_ext,
        vertices,
        pixel,
        (0.0, 0.0),
        valid_mask=valid,
    )
    result = _navigate_arc(
        outward,
        pixel,
        make_ring_feature,
        make_nav_context,
        image,
        min_edge_strength_fraction=0.5 * (radial_strength + pixel_strength),
    )
    assert result.spurious is True


def test_rank_1_result_carries_no_shape_term(
    make_ring_feature: NavFeatureFactory,
    make_nav_context: NavContextFactory,
) -> None:
    """Two straight edges at right angles take the rank-1 path, which adds no shape term.

    Their normals differ and their centers lie apart, so the shape term would not
    be zero; the rank-1 result reports the along-edge axis unobservable instead.
    """
    image = np.zeros((200, 200), dtype=np.float64)
    image[100:, 120:] = 100.0
    us = np.arange(130.0, 190.0)
    horizontal = np.stack([np.full_like(us, 99.5), us], axis=-1)
    vs = np.arange(110.0, 170.0)
    vertical = np.stack([vs, np.full_like(vs, 119.5)], axis=-1)
    features = [
        make_ring_feature(
            'horizontal',
            vertices=horizontal,
            outward_normals=np.tile([1.0, 0.0], (us.size, 1)),
            is_straight_line=True,
        ),
        make_ring_feature(
            'vertical',
            vertices=vertical,
            outward_normals=np.tile([0.0, 1.0], (vs.size, 1)),
            is_straight_line=True,
        ),
    ]
    result = RingEdgeNav().navigate(features, make_nav_context(image))
    assert isinstance(result.diagnostics, RingEdgeDiagnostics)
    assert result.diagnostics.sigma_shape_major_px == 0.0
    assert is_rank_1(np.asarray(result.covariance_px2)) is True


@pytest.mark.parametrize('sigma', [-0.1, math.inf])
def test_ring_edge_nav_rejects_a_bad_shape_sigma(
    sigma: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A negative or non-finite shape sigma fails at construction."""
    bad_tuning = dict(RingEdgeNav.tuning)
    bad_tuning['shape_sigma_px'] = sigma
    monkeypatch.setattr(RingEdgeNav, 'tuning', bad_tuning)
    with pytest.raises(ValueError, match='shape_sigma_px must be a finite number >= 0'):
        RingEdgeNav()
