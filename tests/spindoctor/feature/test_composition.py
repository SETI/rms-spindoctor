"""Tests for ``spindoctor.feature.composition.compose_template_features``."""

import dataclasses

import numpy as np
import pytest

from spindoctor.feature.composition import compose_template_features
from spindoctor.feature.feature import NavFeature, NavReliabilityBreakdown
from spindoctor.feature.feature_type import NavFeatureType
from spindoctor.feature.flags import BodyDiscFlags
from spindoctor.feature.geometry import BodyDiscGeometry
from spindoctor.support.filters import NavFilterKind, NavFilterSpec


def _make_body(
    *,
    feature_id: str,
    bbox: tuple[int, int, int, int],
    template_value: float,
    subject_range_km: float,
) -> NavFeature:
    """Build a BODY_DISC feature with a uniform-value template."""
    h = bbox[2] - bbox[0]
    w = bbox[3] - bbox[1]
    template_img = np.full((h, w), template_value, dtype=np.float64)
    template_mask = np.ones((h, w), dtype=bool)
    return NavFeature(
        feature_id=feature_id,
        feature_type=NavFeatureType.BODY_DISC,
        source_model='body',
        geometry=BodyDiscGeometry(
            bbox_extfov_vu=bbox,
            predicted_center_vu=((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2),
            overflow_fraction=0.0,
        ),
        subject_range_km=subject_range_km,
        position_cov_px=None,
        intensity_sigma_rel=0.0,
        preferred_filter=NavFilterSpec(kind=NavFilterKind.NONE),
        reliability=1.0,
        reliability_reasons=NavReliabilityBreakdown(visible_lit_fraction=1.0),
        usable_types=frozenset({NavFeatureType.BODY_DISC}),
        flags=BodyDiscFlags(body_name='X', overflow_fov_fraction=0.0),
        template_img=template_img,
        template_mask=template_mask,
    )


def test_compose_empty_features_yields_zero_image() -> None:
    """No template features → zero image and false mask."""
    image, mask = compose_template_features([], (10, 10))
    assert image.shape == (10, 10)
    assert image.sum() == 0.0
    assert not mask.any()


def test_compose_single_feature_paints_template() -> None:
    """A single feature paints its template at the bbox location."""
    feat = _make_body(
        feature_id='body_disc:A',
        bbox=(2, 3, 4, 6),
        template_value=5.0,
        subject_range_km=100.0,
    )
    image, mask = compose_template_features([feat], (10, 10))
    assert image[2, 3] == 5.0
    assert image[3, 5] == 5.0
    assert image[5, 5] == 0.0
    assert mask[2:4, 3:6].all()
    assert not mask[0, 0]


def test_compose_nearer_feature_overwrites_farther() -> None:
    """Z-buffer paint: nearer subject_range_km overwrites farther on overlap."""
    far = _make_body(
        feature_id='body_disc:far',
        bbox=(0, 0, 4, 4),
        template_value=1.0,
        subject_range_km=1000.0,
    )
    near = _make_body(
        feature_id='body_disc:near',
        bbox=(2, 2, 6, 6),
        template_value=9.0,
        subject_range_km=10.0,
    )
    image, _ = compose_template_features([far, near], (10, 10))
    # Far-only region (0:2, 0:2) keeps far's value 1.0.
    assert image[0, 0] == 1.0
    # Overlap region (2:4, 2:4) takes near's value 9.0.
    assert image[2, 2] == 9.0
    assert image[3, 3] == 9.0
    # Near-only region (4:6, 4:6) is near's value.
    assert image[5, 5] == 9.0


def test_compose_input_order_does_not_matter() -> None:
    """The composite is order-independent (sort by range internally)."""
    far = _make_body(
        feature_id='body_disc:far',
        bbox=(0, 0, 4, 4),
        template_value=1.0,
        subject_range_km=1000.0,
    )
    near = _make_body(
        feature_id='body_disc:near',
        bbox=(2, 2, 6, 6),
        template_value=9.0,
        subject_range_km=10.0,
    )
    img1, _ = compose_template_features([far, near], (10, 10))
    img2, _ = compose_template_features([near, far], (10, 10))
    assert np.array_equal(img1, img2)


def test_compose_skips_features_without_templates() -> None:
    """Features without template_img+template_mask are silently skipped."""
    # Polyline-style feature: no template.
    from spindoctor.feature.flags import LimbArcFlags
    from spindoctor.feature.geometry import LimbPolyline

    polyline = NavFeature(
        feature_id='limb_arc:X',
        feature_type=NavFeatureType.LIMB_ARC,
        source_model='body',
        geometry=LimbPolyline(
            vertices_vu=np.array([[1.0, 1.0], [2.0, 2.0]]),
            normals_vu=np.array([[1.0, 0.0], [0.0, 1.0]]),
            sigma_normal_per_vertex_px=np.array([0.5, 0.5]),
            sigma_tangent_per_vertex_px=np.array([0.5, 0.5]),
            bbox_extfov_vu=(0, 0, 5, 5),
        ),
        subject_range_km=100.0,
        position_cov_px=None,
        intensity_sigma_rel=0.0,
        preferred_filter=NavFilterSpec(kind=NavFilterKind.NONE),
        reliability=0.9,
        reliability_reasons=NavReliabilityBreakdown(visible_arc_fraction=1.0),
        usable_types=frozenset({NavFeatureType.LIMB_ARC}),
        flags=LimbArcFlags(body_name='X', visible_arc_fraction=1.0),
    )
    body = _make_body(
        feature_id='body_disc:X',
        bbox=(0, 0, 4, 4),
        template_value=3.0,
        subject_range_km=100.0,
    )
    image, mask = compose_template_features([polyline, body], (10, 10))
    assert image[0, 0] == 3.0
    assert mask[0, 0]


def test_compose_rejects_template_larger_than_declared_bbox() -> None:
    """A template bigger than its bbox extents raises instead of painting displaced content."""
    feat = _make_body(
        feature_id='body_disc:oversized',
        bbox=(2, 3, 4, 6),
        template_value=5.0,
        subject_range_km=100.0,
    )
    oversized = dataclasses.replace(
        feat,
        template_img=np.full((10, 10), 5.0, dtype=np.float64),
        template_mask=np.ones((10, 10), dtype=bool),
    )
    with pytest.raises(ValueError, match='bbox-local postage stamps'):
        compose_template_features([oversized], (10, 10))


def test_compose_clamps_bbox_to_extfov() -> None:
    """Templates with bbox extending past the ext-FOV are clamped."""
    feat = _make_body(
        feature_id='body_disc:X',
        bbox=(8, 8, 12, 12),  # extends past 10x10 array
        template_value=4.0,
        subject_range_km=100.0,
    )
    image, mask = compose_template_features([feat], (10, 10))
    # The 2x2 in-bounds region is painted.
    assert image[8, 8] == 4.0
    assert image[9, 9] == 4.0
    assert mask[8, 8]
    assert mask[9, 9]
    # No paint leaks outside the (8:10, 8:10) clipped region.
    assert image[:8, :8].sum() == 0.0
    assert not mask[:8, :8].any()


def _make_limb(
    *, feature_id: str, vertices: list[tuple[float, float]], bbox: tuple[int, int, int, int]
) -> NavFeature:
    """Build a LIMB_ARC feature with the given vertices."""
    from spindoctor.feature.flags import LimbArcFlags
    from spindoctor.feature.geometry import LimbPolyline

    verts = np.asarray(vertices, dtype=np.float64).reshape(-1, 2)
    return NavFeature(
        feature_id=feature_id,
        feature_type=NavFeatureType.LIMB_ARC,
        source_model='body',
        geometry=LimbPolyline(
            vertices_vu=verts,
            normals_vu=np.zeros_like(verts),
            sigma_normal_per_vertex_px=np.full(verts.shape[0], 0.5),
            sigma_tangent_per_vertex_px=np.full(verts.shape[0], 0.5),
            bbox_extfov_vu=bbox,
        ),
        subject_range_km=100.0,
        position_cov_px=None,
        intensity_sigma_rel=0.0,
        preferred_filter=NavFilterSpec(kind=NavFilterKind.NONE),
        reliability=0.9,
        reliability_reasons=NavReliabilityBreakdown(visible_arc_fraction=1.0),
        usable_types=frozenset({NavFeatureType.LIMB_ARC}),
        flags=LimbArcFlags(body_name='X', visible_arc_fraction=1.0),
    )


# ---------------------------------------------------------------------------
# compose_dialog_overlay
# ---------------------------------------------------------------------------


def test_compose_dialog_overlay_paints_limb_polyline_pixels() -> None:
    """Limb-arc vertices are rasterized into the composite mask."""
    from spindoctor.feature.composition import compose_dialog_overlay

    limb = _make_limb(
        feature_id='limb_arc:X',
        vertices=[(2.0, 3.0), (4.0, 5.0), (6.0, 7.0)],
        bbox=(2, 3, 7, 8),
    )
    image, mask = compose_dialog_overlay([limb], (10, 10))
    for v, u in [(2, 3), (4, 5), (6, 7)]:
        assert mask[v, u]
        assert image[v, u] == 1.0
    # Pixels not in the polyline are untouched.
    assert image[0, 0] == 0.0
    assert not mask[0, 0]


def test_compose_dialog_overlay_drops_out_of_bounds_vertices() -> None:
    """Vertices outside the ext-FOV are silently clipped."""
    from spindoctor.feature.composition import compose_dialog_overlay

    limb = _make_limb(
        feature_id='limb_arc:edge',
        vertices=[(-1.0, 5.0), (5.0, 5.0), (5.0, 99.0)],
        bbox=(0, 0, 10, 10),
    )
    _image, mask = compose_dialog_overlay([limb], (10, 10))
    assert mask[5, 5]
    assert mask.sum() == 1


def test_compose_dialog_overlay_combines_template_and_polyline() -> None:
    """A body-disc template + a limb-arc polyline both land in the composite."""
    from spindoctor.feature.composition import compose_dialog_overlay

    body = _make_body(
        feature_id='body_disc:X',
        bbox=(0, 0, 3, 3),
        template_value=4.0,
        subject_range_km=100.0,
    )
    limb = _make_limb(
        feature_id='limb_arc:X',
        vertices=[(7.0, 7.0)],
        bbox=(7, 7, 8, 8),
    )
    image, mask = compose_dialog_overlay([body, limb], (10, 10))
    assert image[0, 0] == 4.0  # body template kept
    assert mask[0, 0]
    assert image[7, 7] == 1.0  # polyline pixel painted
    assert mask[7, 7]


def _make_blob(
    *,
    feature_id: str,
    center: tuple[float, float],
    diameter_px: float,
    bbox: tuple[int, int, int, int],
) -> NavFeature:
    """Build a BODY_BLOB feature."""
    from spindoctor.feature.flags import BodyBlobFlags
    from spindoctor.feature.geometry import BodyBlobGeometry

    return NavFeature(
        feature_id=feature_id,
        feature_type=NavFeatureType.BODY_BLOB,
        source_model='body',
        geometry=BodyBlobGeometry(
            predicted_center_vu=center,
            bbox_extfov_vu=bbox,
            predicted_diameter_px=diameter_px,
        ),
        subject_range_km=100.0,
        position_cov_px=None,
        intensity_sigma_rel=0.0,
        preferred_filter=NavFilterSpec(kind=NavFilterKind.NONE),
        reliability=0.5,
        reliability_reasons=NavReliabilityBreakdown(blob_snr=0.5),
        usable_types=frozenset({NavFeatureType.BODY_BLOB}),
        flags=BodyBlobFlags(body_name='X', predicted_diameter_px=diameter_px),
    )


def test_compose_dialog_overlay_renders_body_blob_circle() -> None:
    """A BODY_BLOB feature renders a 1-pixel circle outline at its centroid.

    Pin the exact pixel set Bresenham produces for radius 5 centered at
    (20, 20) so a future change to the circle algorithm or the dialog
    overlay's painting logic fails this test loud.
    """
    from spindoctor.feature.composition import compose_dialog_overlay

    blob = _make_blob(
        feature_id='body_blob:X',
        center=(20.0, 20.0),
        diameter_px=10.0,
        bbox=(15, 15, 25, 25),
    )
    image, mask = compose_dialog_overlay([blob], (40, 40))
    expected_pixels = {
        (15, 18),
        (15, 19),
        (15, 20),
        (15, 21),
        (15, 22),
        (16, 17),
        (16, 23),
        (17, 16),
        (17, 24),
        (18, 15),
        (18, 25),
        (19, 15),
        (19, 25),
        (20, 15),
        (20, 25),
        (21, 15),
        (21, 25),
        (22, 15),
        (22, 25),
        (23, 16),
        (23, 24),
        (24, 17),
        (24, 23),
        (25, 18),
        (25, 19),
        (25, 20),
        (25, 21),
        (25, 22),
    }
    actual_pixels = set(zip(*np.nonzero(image), strict=True))
    assert actual_pixels == expected_pixels
    assert set(zip(*np.nonzero(mask), strict=True)) == expected_pixels
    assert np.count_nonzero(image) == len(expected_pixels)
    assert np.count_nonzero(mask) == len(expected_pixels)


def test_compose_dialog_overlay_blob_clips_to_extfov_bounds() -> None:
    """A blob whose circle extends past ext-FOV is silently clipped.

    With center (2, 2) and radius 10 against a 10x10 ext-FOV, the
    Bresenham outline only intersects a single in-bounds pixel.  Pin
    the exact pixel set so a regression in either the clipping logic
    or the circle algorithm fails this test loud.
    """
    from spindoctor.feature.composition import compose_dialog_overlay

    blob = _make_blob(
        feature_id='body_blob:edge',
        center=(2.0, 2.0),
        diameter_px=20.0,  # radius 10, way past (0,0)
        bbox=(-10, -10, 12, 12),
    )
    image, mask = compose_dialog_overlay([blob], (10, 10))
    assert image.shape == (10, 10)
    expected_pixels = {(9, 9)}
    assert set(zip(*np.nonzero(image), strict=True)) == expected_pixels
    assert set(zip(*np.nonzero(mask), strict=True)) == expected_pixels
    assert np.count_nonzero(image) == 1
    assert np.count_nonzero(mask) == 1


def _make_star(
    *,
    feature_id: str,
    predicted_vu: tuple[float, float],
    bbox_pad: int = 6,
) -> NavFeature:
    """Build a STAR feature with a PSF-sized bbox around the prediction."""
    from spindoctor.feature.flags import StarFlags
    from spindoctor.feature.geometry import StarGeometry

    pv, pu = predicted_vu
    bbox = (
        int(np.floor(pv - bbox_pad)),
        int(np.floor(pu - bbox_pad)),
        int(np.ceil(pv + bbox_pad)) + 1,
        int(np.ceil(pu + bbox_pad)) + 1,
    )
    return NavFeature(
        feature_id=feature_id,
        feature_type=NavFeatureType.STAR,
        source_model='stars',
        geometry=StarGeometry(
            predicted_vu=predicted_vu,
            catalog_vu=predicted_vu,
            bbox_extfov_vu=bbox,
        ),
        subject_range_km=float('inf'),
        position_cov_px=None,
        intensity_sigma_rel=0.0,
        preferred_filter=NavFilterSpec(kind=NavFilterKind.NONE),
        reliability=0.95,
        reliability_reasons=NavReliabilityBreakdown(predicted_snr=1.0),
        usable_types=frozenset({NavFeatureType.STAR}),
        flags=StarFlags(predicted_snr=10.0),
    )


def test_compose_dialog_overlay_renders_star_marker_rectangle() -> None:
    """A STAR feature renders an exact rectangle-outline pixel set.

    Pin the full perimeter — top + bottom + left + right at half-width
    5 around (20, 25) — so a regression in the marker geometry, the
    bbox-to-half-extent conversion, or the choice of outline-vs-filled
    rasteriser fails this test loud.
    """
    from spindoctor.feature.composition import compose_dialog_overlay

    star = _make_star(
        feature_id='star:UCAC4:1',
        predicted_vu=(20.0, 25.0),
        bbox_pad=6,  # bbox spans (14, 19, 27, 32) -> half-width 5
    )
    image, mask = compose_dialog_overlay([star], (40, 40))
    expected_pixels = set()
    half = 5
    v_min, v_max = 20 - half, 20 + half
    u_min, u_max = 25 - half, 25 + half
    for u in range(u_min, u_max + 1):
        expected_pixels.add((v_min, u))  # top
        expected_pixels.add((v_max, u))  # bottom
    for v in range(v_min, v_max + 1):
        expected_pixels.add((v, u_min))  # left
        expected_pixels.add((v, u_max))  # right
    actual_pixels = set(zip(*np.nonzero(image), strict=True))
    assert actual_pixels == expected_pixels
    assert set(zip(*np.nonzero(mask), strict=True)) == expected_pixels
    assert np.count_nonzero(image) == len(expected_pixels)
    assert np.count_nonzero(mask) == len(expected_pixels)


def test_compose_dialog_overlay_star_marker_off_image_no_op() -> None:
    """A STAR whose predicted (v, u) is off-image paints no pixels."""
    from spindoctor.feature.composition import compose_dialog_overlay

    star = _make_star(
        feature_id='star:UCAC4:offscreen',
        predicted_vu=(-50.0, -50.0),
    )
    image, mask = compose_dialog_overlay([star], (20, 20))
    assert np.count_nonzero(image) == 0
    assert np.count_nonzero(mask) == 0


def test_compose_dialog_overlay_star_marker_clamps_at_edge() -> None:
    """A STAR center near the FOV edge produces an exact 3x3-perimeter pixel set.

    With predicted (1, 1) on a 20x20 FOV, the marker can extend at most
    1 pixel before hitting the edge — the floor (3) does not apply here
    because the explicit edge clamp is tighter.  The full 8-pixel
    perimeter (3x3 outline minus the center) is asserted as an exact
    set so a regression in the half-width clamp logic fails loud.
    """
    from spindoctor.feature.composition import compose_dialog_overlay

    star = _make_star(
        feature_id='star:UCAC4:edge',
        predicted_vu=(1.0, 1.0),
        bbox_pad=6,
    )
    image, mask = compose_dialog_overlay([star], (20, 20))
    # Half-width clamped to 1 -> 3x3 outline around (1, 1).
    expected_pixels = {
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 2),
        (2, 0),
        (2, 1),
        (2, 2),
    }
    actual_pixels = set(zip(*np.nonzero(image), strict=True))
    assert actual_pixels == expected_pixels
    assert set(zip(*np.nonzero(mask), strict=True)) == expected_pixels
    assert np.count_nonzero(image) == len(expected_pixels)
    assert np.count_nonzero(mask) == len(expected_pixels)
    assert image[1, 1] == 0.0  # center not painted


def _make_titan(
    *,
    feature_id: str,
    center: tuple[float, float],
    r_env_px: float,
    r_solid_px: float = 1.0,
) -> NavFeature:
    """Build a TITAN_LIMB feature carrying only its haze geometry."""
    from spindoctor.feature.flags import TitanHazeFlags
    from spindoctor.feature.geometry import TitanHazeGeometry

    pad = int(np.ceil(r_env_px))
    bbox = (
        int(center[0]) - pad,
        int(center[1]) - pad,
        int(center[0]) + pad + 1,
        int(center[1]) + pad + 1,
    )
    return NavFeature(
        feature_id=feature_id,
        feature_type=NavFeatureType.TITAN_LIMB,
        source_model='titan:TITAN',
        geometry=TitanHazeGeometry(
            predicted_center_vu=center,
            sun_angle_rad=0.0,
            axis_degenerate=False,
            phase_deg=30.0,
            r_solid_px=r_solid_px,
            r_env_px=r_env_px,
            km_per_px=20.0,
            contaminant_mask=None,
            filters=('CL1', 'CL2'),
            bbox_extfov_vu=bbox,
        ),
        subject_range_km=1.2e6,
        position_cov_px=None,
        intensity_sigma_rel=0.0,
        preferred_filter=NavFilterSpec(kind=NavFilterKind.NONE),
        reliability=0.8,
        reliability_reasons=NavReliabilityBreakdown(titan_envelope_diameter_px=2.0 * r_env_px),
        usable_types=frozenset({NavFeatureType.TITAN_LIMB}),
        flags=TitanHazeFlags(body_name='TITAN'),
    )


def test_compose_dialog_overlay_renders_titan_envelope_circle() -> None:
    """A TITAN_LIMB feature paints a circle outline the operator can drag.

    Without this branch a Titan-only frame composes an empty overlay and
    manual navigation is impossible on it.
    """
    from spindoctor.feature.composition import compose_dialog_overlay

    titan = _make_titan(feature_id='titan_limb:TITAN', center=(20.0, 20.0), r_env_px=5.0)
    _image, mask = compose_dialog_overlay([titan], (40, 40))
    assert np.count_nonzero(mask) > 0


def test_compose_dialog_overlay_titan_circle_matches_the_blob_circle() -> None:
    """The haze circle is rasterized exactly like a blob circle of equal radius.

    Both branches paint the same predicted-silhouette outline, so pinning
    them to each other keeps a change to one from silently diverging the
    dialog's two circle renderings.
    """
    from spindoctor.feature.composition import compose_dialog_overlay

    titan = _make_titan(feature_id='titan_limb:TITAN', center=(20.0, 20.0), r_env_px=5.0)
    blob = _make_blob(
        feature_id='body_blob:X',
        center=(20.0, 20.0),
        diameter_px=10.0,
        bbox=(15, 15, 25, 25),
    )
    titan_image, _titan_mask = compose_dialog_overlay([titan], (40, 40))
    blob_image, _blob_mask = compose_dialog_overlay([blob], (40, 40))
    assert np.array_equal(titan_image, blob_image)


def test_compose_dialog_overlay_titan_circle_uses_the_envelope_radius() -> None:
    """The circle is drawn at the haze envelope, not at the solid body."""
    from spindoctor.feature.composition import compose_dialog_overlay

    titan = _make_titan(
        feature_id='titan_limb:TITAN', center=(20.0, 20.0), r_env_px=8.0, r_solid_px=3.0
    )
    _image, mask = compose_dialog_overlay([titan], (40, 40))
    assert bool(mask[20, 28]) is True


def test_compose_dialog_overlay_titan_circle_leaves_the_solid_radius_clear() -> None:
    """Nothing is painted at the solid radius; the envelope is what is drawn."""
    from spindoctor.feature.composition import compose_dialog_overlay

    titan = _make_titan(
        feature_id='titan_limb:TITAN', center=(20.0, 20.0), r_env_px=8.0, r_solid_px=3.0
    )
    _image, mask = compose_dialog_overlay([titan], (40, 40))
    assert bool(mask[20, 23]) is False


def test_compose_dialog_overlay_titan_clips_to_extfov_bounds() -> None:
    """A haze circle reaching past the ext-FOV paints only its visible arc."""
    from spindoctor.feature.composition import compose_dialog_overlay

    titan = _make_titan(feature_id='titan_limb:TITAN', center=(2.0, 2.0), r_env_px=10.0)
    image, mask = compose_dialog_overlay([titan], (10, 10))
    assert image.shape == (10, 10)
    assert np.count_nonzero(mask) < 10
