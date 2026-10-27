"""Ring azimuthal clutter and embedded moonlets: truth-side structure.

The azimuthal block (brightness modulation, planet-shadow wedge, seeded
spokes) scales the emitted intensity and leaves tau -- and so the
transmission screen -- untouched; moonlets are opaque discs at the ring's
depth that emit, extinguish the background, and composite even inside an
empty gap; propellers disturb the local tau.  Spokes draw from the
'scene_radiance/ring_system/spokes' stream, so a scene re-renders
bit-identically and reseeds with the scene seed.
"""

import math
from typing import Any

import numpy as np
import pytest

from spindoctor.sim.forward.ring_system import RingSystemMaps, render_ring_system
from spindoctor.sim.render import render_combined_model
from spindoctor.sim.scene import validate_sim_params

_SIZE = 96
_CENTER = 48.5


def _sheet(tau: float = 1.0) -> dict[str, Any]:
    """A broad face-on sheet covering radii 5..40."""
    return {'kind': 'ringlet', 'tau': tau, 'width_px': 35.0, 'orbit': {'a_px': 5.0}}


def _render(system_extra: dict[str, Any], *, spokes_seed: int = 0) -> RingSystemMaps:
    system: dict[str, Any] = {
        'geometry': {
            'center_v': _CENTER,
            'center_u': _CENTER,
            'opening_deg_obs': 90.0,
            'opening_deg_sun': 90.0,
            'node_deg': 0.0,
        },
        'features': [_sheet()],
    }
    system.update(system_extra)
    return render_ring_system(
        (_SIZE, _SIZE),
        system,
        center_v=_CENTER,
        center_u=_CENTER,
        node_deg=0.0,
        spokes_seed=spokes_seed,
    )


# Face-on with node = 0: lam = atan2(y, x) with x = du and y = -dv, so the
# +u axis is lam = 0 and the -v axis is lam = 90 degrees.
_PIXEL_LAM_0 = (48, 68)  # 20 px along +u
_PIXEL_LAM_90 = (28, 48)  # 20 px along -v


def test_modulation_scales_intensity_by_the_cosine() -> None:
    """1 + amplitude * cos(m * (lam - phase)) at two probe longitudes."""
    base = _render({})
    modulated = _render(
        {'azimuthal': {'modulation': {'amplitude': 0.25, 'm': 2, 'lam_peak_deg': 0.0}}}
    )
    # m = 2, phase 0: factor 1.25 at lam = 0, 0.75 at lam = 90.
    assert modulated.intensity[_PIXEL_LAM_0] == pytest.approx(
        1.25 * base.intensity[_PIXEL_LAM_0], rel=1e-12
    )
    assert modulated.intensity[_PIXEL_LAM_90] == pytest.approx(
        0.75 * base.intensity[_PIXEL_LAM_90], rel=1e-12
    )


def test_shadow_wedge_darkens_intensity_only() -> None:
    """The planet-shadow wedge darkens emission; the screen is untouched."""
    base = _render({})
    shadowed = _render(
        {'azimuthal': {'shadow': {'start_deg': 45.0, 'extent_deg': 90.0, 'darkness': 1.0}}}
    )
    # lam = 90 is inside [45, 135); lam = 0 is outside.
    assert shadowed.intensity[_PIXEL_LAM_90] == 0.0
    assert shadowed.intensity[_PIXEL_LAM_0] == pytest.approx(
        base.intensity[_PIXEL_LAM_0], rel=1e-12
    )
    np.testing.assert_array_equal(shadowed.transmission, base.transmission)


def _spoked(seed: int) -> RingSystemMaps:
    return _render(
        {
            'azimuthal': {
                'spokes': {
                    'count': 5,
                    'r_inner_px': 10.0,
                    'r_outer_px': 35.0,
                    'contrast': -0.6,
                    'width_deg': 15.0,
                }
            }
        },
        spokes_seed=seed,
    )


def test_spokes_darken_the_sheet_inside_their_band() -> None:
    """Dark spokes lower intensity somewhere in the radial band."""
    base = _render({})
    spoked = _spoked(7)
    assert bool((spoked.intensity < base.intensity - 1e-9).any())
    np.testing.assert_array_equal(spoked.transmission, base.transmission)


def test_spokes_are_deterministic_per_seed() -> None:
    """The same seed renders the same spoke field bit for bit."""
    np.testing.assert_array_equal(_spoked(7).intensity, _spoked(7).intensity)


def test_spokes_reseed_with_the_stream_seed() -> None:
    """A different stream seed draws a different spoke field."""
    assert bool((_spoked(7).intensity != _spoked(8).intensity).any())


def test_spoke_field_respects_its_radial_band() -> None:
    """Outside r_inner_px..r_outer_px the sheet is untouched."""
    base = _render({})
    spoked = _render(
        {
            'azimuthal': {
                'spokes': {
                    'count': 8,
                    'r_inner_px': 20.0,
                    'r_outer_px': 30.0,
                    'contrast': -0.9,
                    'width_deg': 30.0,
                }
            }
        },
        spokes_seed=3,
    )
    # 8 px along +u sits at ring radius 8, well inside r_inner_px.
    assert spoked.intensity[48, 56] == pytest.approx(base.intensity[48, 56], rel=1e-12)


def test_moonlet_disc_emits_and_occults() -> None:
    """The disc replaces ring emission with its own and zeroes the screen."""
    moonlet = {'a_px': 20.0, 'lam_deg': 0.0, 'radius_px': 2.0, 'amplitude': 0.9}
    maps = _render({'moonlets': [moonlet]})
    assert maps.intensity[_PIXEL_LAM_0] == pytest.approx(0.9, rel=1e-12)
    assert maps.transmission[_PIXEL_LAM_0] == 0.0
    assert bool(maps.mask[_PIXEL_LAM_0])


def test_moonlet_in_an_empty_gap_still_composites() -> None:
    """A moonlet with no tau under it joins the mask (it must composite)."""
    moonlet = {'a_px': 20.0, 'lam_deg': 0.0, 'radius_px': 2.0, 'amplitude': 0.9}
    maps = _render({'features': [], 'moonlets': [moonlet]})
    assert bool(maps.mask[_PIXEL_LAM_0])
    assert maps.intensity[_PIXEL_LAM_0] == pytest.approx(0.9, rel=1e-12)
    # Away from the disc nothing renders.
    assert not bool(maps.mask[_PIXEL_LAM_90])


def test_moonlet_extinguishes_a_background_star() -> None:
    """Through the full render path, a star behind the disc vanishes.

    The scene renders with and without the star; a dark (amplitude 0)
    moonlet disc in front leaves the two images identical -- the star's
    flux never reaches the detector.
    """

    def _scene(stars: list[dict[str, Any]]) -> dict[str, Any]:
        return validate_sim_params(
            {
                'instrument': 'coiss_nac',
                'size_v': _SIZE,
                'size_u': _SIZE,
                'random_seed': 3,
                'stars': stars,
                'ring_system': {
                    'geometry': {
                        'center_v': _CENTER,
                        'center_u': _CENTER,
                        'opening_deg_obs': 90.0,
                        'opening_deg_sun': 90.0,
                        'node_deg': 0.0,
                    },
                    'features': [],
                    'moonlets': [
                        {'a_px': 20.0, 'lam_deg': 0.0, 'radius_px': 3.0, 'amplitude': 0.0}
                    ],
                },
            }
        )

    with_star, _m0 = render_combined_model(
        _scene([{'name': 'S', 'v': 48.5, 'u': 68.5, 'vmag': 6.0}])
    )
    without_star, _m1 = render_combined_model(_scene([]))
    np.testing.assert_array_equal(with_star, without_star)


def test_propeller_carves_local_tau_dips() -> None:
    """Negative contrast opens partial gaps straddling the moonlet."""
    moonlet = {
        'a_px': 20.0,
        'lam_deg': 90.0,
        'radius_px': 1.0,
        'amplitude': 0.2,
        'propeller': {'length_deg': 30.0, 'width_px': 3.0, 'contrast': -0.9},
    }
    base = _render({})
    with_prop = _render({'moonlets': [moonlet]})
    # Lobe centers: (a + 3 px, lam + 15 deg) and (a - 3 px, lam - 15 deg).
    lobe1_v = 48 - round(23.0 * math.sin(math.radians(105.0)))
    lobe1_u = 48 + round(23.0 * math.cos(math.radians(105.0)))
    assert with_prop.transmission[lobe1_v, lobe1_u] > base.transmission[lobe1_v, lobe1_u]
    # Far from the propeller the sheet's tau is untouched.
    assert with_prop.transmission[_PIXEL_LAM_0] == pytest.approx(
        base.transmission[_PIXEL_LAM_0], rel=1e-9
    )


def test_full_scene_with_spokes_renders_deterministically() -> None:
    """Two renders of a spoked scene are bit-identical (seeded stream)."""
    scene = validate_sim_params(
        {
            'instrument': 'coiss_nac',
            'size_v': _SIZE,
            'size_u': _SIZE,
            'random_seed': 11,
            'ring_system': {
                'geometry': {
                    'center_v': _CENTER,
                    'center_u': _CENTER,
                    'opening_deg_obs': 90.0,
                    'opening_deg_sun': 90.0,
                    'node_deg': 0.0,
                },
                'features': [_sheet()],
                'azimuthal': {
                    'spokes': {
                        'count': 4,
                        'r_inner_px': 10.0,
                        'r_outer_px': 35.0,
                        'contrast': -0.5,
                        'width_deg': 12.0,
                    }
                },
            },
        }
    )
    img_a, _m0 = render_combined_model(scene)
    img_b, _m1 = render_combined_model(scene)
    np.testing.assert_array_equal(img_a, img_b)
