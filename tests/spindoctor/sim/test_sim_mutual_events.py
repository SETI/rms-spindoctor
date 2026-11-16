"""Mutual-event occlusion truth: rendered bookkeeping vs analytic geometry.

The renderer composites bodies near-over-far; ``body_occlusion`` records, per
body, the visible fraction of its unoccluded silhouette and the limb arc its
occluders hide.  For two circular discs both quantities have closed forms, so
a constructed two-circle scene pins the measured truth against the analytic
values.
"""

import math
from typing import Any

import pytest

from spindoctor.sim.render import render_combined_model

_R_FAR = 60.0
_R_NEAR = 50.0
_FAR_U = 95.0


def _two_circle_scene(separation_px: float) -> dict[str, Any]:
    """Two spheres at controlled center separation, far body first."""
    return {
        'instrument': 'coiss_nac',
        'size_v': 256,
        'size_u': 256,
        'random_seed': 42,
        'bodies': [
            {
                'name': 'FAR',
                'center_v': 128.0,
                'center_u': _FAR_U,
                'semi_a_px': _R_FAR,
                'semi_b_px': _R_FAR,
                'semi_c_px': _R_FAR,
                'illumination_angle': 20.0,
                'phase_angle': 30.0,
                'range_km': 700000.0,
            },
            {
                'name': 'NEAR',
                'center_v': 128.0,
                'center_u': _FAR_U + separation_px,
                'semi_a_px': _R_NEAR,
                'semi_b_px': _R_NEAR,
                'semi_c_px': _R_NEAR,
                'illumination_angle': 20.0,
                'phase_angle': 30.0,
                'range_km': 500000.0,
            },
        ],
    }


def _analytic_occlusion(d: float, r1: float, r2: float) -> tuple[float, float]:
    """Visible fraction and hidden limb arc (deg) of circle 1 behind circle 2."""
    if d >= r1 + r2:
        return 1.0, 0.0
    half_angle_1 = math.acos((d * d + r1 * r1 - r2 * r2) / (2.0 * d * r1))
    half_angle_2 = math.acos((d * d + r2 * r2 - r1 * r1) / (2.0 * d * r2))
    lens = r1 * r1 * (half_angle_1 - math.sin(2.0 * half_angle_1) / 2.0) + r2 * r2 * (
        half_angle_2 - math.sin(2.0 * half_angle_2) / 2.0
    )
    return 1.0 - lens / (math.pi * r1 * r1), math.degrees(2.0 * half_angle_1)


@pytest.mark.parametrize('separation_px', [100.0, 70.0, 55.0])
def test_far_body_occlusion_matches_two_circle_geometry(separation_px: float) -> None:
    """visible_fraction and occluded_limb_arc_deg match the closed forms."""
    _, meta = render_combined_model(_two_circle_scene(separation_px))
    truth = meta['body_occlusion']['FAR']
    visible_expected, arc_expected = _analytic_occlusion(separation_px, _R_FAR, _R_NEAR)
    assert truth['visible_fraction'] == pytest.approx(visible_expected, abs=0.02)
    assert truth['occluded_limb_arc_deg'] == pytest.approx(arc_expected, abs=8.0)


def test_near_body_is_recorded_fully_visible() -> None:
    """The occluding (nearer) body carries no occlusion of its own."""
    _, meta = render_combined_model(_two_circle_scene(70.0))
    truth = meta['body_occlusion']['NEAR']
    assert truth['visible_fraction'] == 1.0
    assert truth['occluded_limb_arc_deg'] == 0.0


def test_disjoint_bodies_record_no_occlusion() -> None:
    """Non-overlapping bodies record full visibility on both sides."""
    _, meta = render_combined_model(_two_circle_scene(140.0))
    for name in ('FAR', 'NEAR'):
        assert meta['body_occlusion'][name]['visible_fraction'] == 1.0
        assert meta['body_occlusion'][name]['occluded_limb_arc_deg'] == 0.0


def test_single_body_scene_records_trivial_occlusion() -> None:
    """A one-body scene still carries the truth block, trivially unoccluded."""
    scene = _two_circle_scene(70.0)
    scene['bodies'] = scene['bodies'][:1]
    _, meta = render_combined_model(scene)
    assert meta['body_occlusion'] == {
        'FAR': {'visible_fraction': 1.0, 'occluded_limb_arc_deg': 0.0}
    }
