"""Tests of the realism match's limb gradient-peak statistic, and its regression check.

The regression check holds the committed realism record (or the summary that
``SPINDOCTOR_TEST_REALISM_RECORD`` points to): in every limb bin that has a verdict, the
matched frames' median gradient-peak offset must lie within the band set from the real
frames.  The other tests build their frames and scenes in memory.  Nothing here reads
holdings, so the file runs in the plain unit tier.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from scipy.special import erfc

from spindoctor.feature.geometry import LimbPolyline
from tests.integration.realism_regression import REGENERATE_HINT, load_record, made_at
from tests.integration.sim_realism_limb_peak import (
    compare_limb_peaks,
    limb_peak_offset,
    matched_silhouette_offsets,
    record_has_limb_peaks,
    record_limb_peak_problems,
)

_RECORD = load_record()

_MARGIN_VU = (5, 7)
_OBS = SimpleNamespace(extfov_margin_vu=_MARGIN_VU)


def _sphere_scene(center_vu: tuple[float, float], radius: float) -> dict[str, Any]:
    """A matched-style scene holding one sphere at a pixel-corner center, lit from +u."""
    return {
        'bodies': [
            {
                'name': 'REALISM_BODY',
                'center_v': center_vu[0],
                'center_u': center_vu[1],
                'semi_a_px': radius,
                'semi_b_px': radius,
                'semi_c_px': radius,
                'illumination_angle': 90.0,
                'phase_angle': 30.0,
            }
        ]
    }


def _disc_image(center_vu: tuple[float, float], radius: float, *, size: int = 120) -> np.ndarray:
    """A uniformly bright disc whose edge is blurred by a Gaussian of 0.8 px.

    The center is pixel centric, the convention of the image array.
    """
    vv, uu = np.mgrid[0:size, 0:size].astype(np.float64)
    r = np.hypot(vv - center_vu[0], uu - center_vu[1])
    return np.asarray(0.5 * erfc((r - radius) / (np.sqrt(2.0) * 0.8)))


def _limb(
    center_vu: tuple[float, float], radius: float, n: int = 90, *, facing_deg: float = 0.0
) -> LimbPolyline:
    """A half-limb of vertices 0.4 px inside a circle, in extended-FOV coordinates.

    Its middle normal points ``facing_deg`` from +u toward +v: 0 faces +u, the lit half
    under ``_sphere_scene``.
    """
    angles = np.radians(np.linspace(-85.0, 85.0, n) + facing_deg)
    normals = np.column_stack([np.sin(angles), np.cos(angles)])
    vertices = np.asarray(center_vu) + normals * (radius - 0.4)
    return LimbPolyline(
        vertices_vu=vertices,
        normals_vu=normals,
        sigma_normal_per_vertex_px=np.full(n, 0.5),
        sigma_tangent_per_vertex_px=np.full(n, 0.5),
        bbox_extfov_vu=(0, 0, 1, 1),
    )


def test_the_matched_silhouette_is_the_scenes_circle() -> None:
    """A vertex 0.4 px inside the scene's circle lies 0.4 px from its silhouette."""
    offsets = matched_silhouette_offsets(_sphere_scene((60.5, 60.5), 30.0), _OBS)
    center = np.array([60.0 + _MARGIN_VU[0], 60.0 + _MARGIN_VU[1]])
    limb = _limb((float(center[0]), float(center[1])), 30.0)
    distances = offsets('body_sim:REALISM_BODY', limb.vertices_vu, limb.normals_vu)
    assert np.max(np.abs(distances - 0.4)) < 1e-9


@pytest.mark.parametrize(
    ('illumination_deg', 'facing_deg'),
    [(90.0, 180.0), (0.0, 90.0)],
    ids=['lit-from-right', 'lit-from-top'],
)
def test_the_matched_silhouette_leaves_out_the_unlit_limb(
    illumination_deg: float, facing_deg: float
) -> None:
    """Vertices on the half of the limb facing away from the light get no silhouette.

    ``illumination_angle`` 0 lights the top of the image (-v) and 90 the right (+u), so
    the unlit half faces -u in the first case and +v in the second.
    """
    scene = _sphere_scene((60.5, 60.5), 30.0)
    scene['bodies'][0]['illumination_angle'] = illumination_deg
    offsets = matched_silhouette_offsets(scene, _OBS)
    center = (60.0 + _MARGIN_VU[0], 60.0 + _MARGIN_VU[1])
    limb = _limb(center, 30.0, facing_deg=facing_deg)
    distances = offsets('body_sim:REALISM_BODY', limb.vertices_vu, limb.normals_vu)
    assert np.all(np.isnan(distances))


def test_a_matched_scene_with_two_bodies_is_refused() -> None:
    """The matched silhouette reads a scene that holds one body."""
    scene = _sphere_scene((60.5, 60.5), 30.0)
    scene['bodies'].append(dict(scene['bodies'][0]))
    with pytest.raises(ValueError, match='holds one body, not 2'):
        matched_silhouette_offsets(scene, _OBS)


def test_a_matched_body_that_is_not_a_sphere_is_refused() -> None:
    """The matched silhouette is a circle, so a flattened body is refused."""
    scene = _sphere_scene((60.5, 60.5), 30.0)
    scene['bodies'][0]['semi_c_px'] = 28.0
    with pytest.raises(ValueError, match="'REALISM_BODY' is not a sphere"):
        matched_silhouette_offsets(scene, _OBS)


@pytest.mark.parametrize(
    ('edge_radius', 'expected'),
    [(30.0, 0.0), (29.5, -0.5)],
    ids=['edge-on-silhouette', 'edge-inside'],
)
def test_a_limbs_peak_offset_is_its_edge_less_its_silhouette(
    edge_radius: float, expected: float
) -> None:
    """A disc stated at radius 30 whose edge lies at ``edge_radius`` peaks that far off."""
    offsets = matched_silhouette_offsets(_sphere_scene((60.5, 60.5), 30.0), _OBS)
    image = _disc_image((60.0, 60.0), edge_radius)
    limb = _limb((60.0 + _MARGIN_VU[0], 60.0 + _MARGIN_VU[1]), 30.0)
    peak = limb_peak_offset(
        image,
        limb,
        'body_sim:REALISM_BODY',
        shift_vu=-np.asarray(_MARGIN_VU, dtype=np.float64),
        silhouette_offsets=offsets,
    )
    assert peak == pytest.approx(expected, abs=0.02)


def test_a_misplaced_limb_moves_the_translation_and_keeps_the_offset() -> None:
    """A frame offset 1.3 px wrong toward the lit limb leaves the peak offset alone."""
    offsets = matched_silhouette_offsets(_sphere_scene((60.5, 60.5), 30.0), _OBS)
    image = _disc_image((60.0, 61.3), 29.5)
    limb = _limb((60.0 + _MARGIN_VU[0], 60.0 + _MARGIN_VU[1]), 30.0)
    peak = limb_peak_offset(
        image,
        limb,
        'body_sim:REALISM_BODY',
        shift_vu=-np.asarray(_MARGIN_VU, dtype=np.float64),
        silhouette_offsets=offsets,
    )
    assert peak == pytest.approx(-0.5, abs=0.03)


def test_a_bin_with_three_real_limbs_sets_its_band_from_its_own_scatter() -> None:
    """Three real limbs at -0.5, -0.3, and -0.1 px give a band of 0.34 px."""
    bins = compare_limb_peaks({'limb_peak_p0_r1': [-0.5, -0.3, -0.1]}, {'limb_peak_p0_r1': [-0.05]})
    peak = bins['p0_r1']
    assert peak.band_px == pytest.approx(2.0 * 1.4826 * 0.2 / np.sqrt(3.0))
    assert peak.within_band is True


def test_a_matched_median_beyond_the_band_is_outside_it() -> None:
    """A matched median 0.36 px from the real one lies outside a 0.34 px band."""
    bins = compare_limb_peaks({'limb_peak_p0_r1': [-0.5, -0.3, -0.1]}, {'limb_peak_p0_r1': [0.06]})
    assert bins['p0_r1'].within_band is False


def test_a_bin_with_two_real_limbs_is_not_judged() -> None:
    """Two real limbs give no band and no verdict, however many the instrument has."""
    real = {'limb_peak_p0_r1': [-0.5, -0.3, -0.1], 'limb_peak_p1_r2': [0.4, 1.08]}
    bins = compare_limb_peaks(real, {'limb_peak_p1_r2': [0.0]})
    assert bins['p1_r2'].band_px is None
    assert bins['p1_r2'].within_band is None


def test_a_tight_bin_is_held_to_the_band_floor() -> None:
    """Real limbs that agree exactly still get a band of 0.1 px."""
    bins = compare_limb_peaks({'limb_peak_p0_r0': [0.2, 0.2, 0.2]}, {'limb_peak_p0_r0': [0.0]})
    assert bins['p0_r0'].band_px == pytest.approx(0.1)


def test_a_bin_without_matched_limbs_is_not_judged() -> None:
    """A bin populated only by real limbs records its band and no verdict."""
    bins = compare_limb_peaks({'limb_peak_p0_r0': [0.2, 0.0, -0.2]}, {})
    assert bins['p0_r0'].within_band is None


def _record_bin(within: bool | None) -> dict[str, Any]:
    """One limb bin's entry as the realism summary writes it."""
    return {
        'n_real': 4,
        'n_sim': 4,
        'real_median_px': -0.1,
        'sim_median_px': 0.3,
        'real_scatter_px': 0.2,
        'band_px': 0.2,
        'within_band': within,
    }


def test_a_record_bin_outside_its_band_is_a_problem() -> None:
    """Only the bin whose verdict is False is reported."""
    record = {
        'instruments': {
            'coiss_calib_nac': {
                'limb_peaks': {
                    'p0_r0': _record_bin(False),
                    'p0_r1': _record_bin(True),
                    'p1_r0': _record_bin(None),
                }
            }
        }
    }
    problems = record_limb_peak_problems(record)
    assert problems == [
        'coiss_calib_nac p0_r0: matched median +0.300 px (4 limbs) lies more than '
        '0.200 px from the real median -0.100 px (4 limbs)'
    ]


@pytest.mark.skipif(
    not record_has_limb_peaks(_RECORD),
    reason='the realism record predates the limb gradient-peak comparison',
)
def test_the_recorded_limb_peaks_lie_within_their_bands() -> None:
    """In every bin with a verdict, the matched median lies within the real band."""
    problems = record_limb_peak_problems(_RECORD)
    assert not problems, (
        f'The limb gradient peaks of the record made at {made_at(_RECORD)} leave their '
        'bands:\n' + '\n'.join(problems) + f'\n{REGENERATE_HINT}'
    )
