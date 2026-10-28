"""Each unit-named ring key is read under its name, on both sides of the boundary.

A reader that looks a key up under any other name reads it as absent: the
reader falls back to its default and the scene renders as though the key were
missing, with nothing to say so.  These tests perturb each ring-plane length
and angle that sets a feature's radial shape, its catalog orbit, its m-modes
and edge wave, the brightness modulation's peak, the spoke band, and a
moonlet's orbit radius, in a scene that carries all of them, and check that the
change reaches every reader that should see it.  The rendered image must
change for every key.  The navigator's prediction must change for each
idealized key it reads, and stay identical for the rest: the truth-side clutter
never crosses the information boundary, and the prediction is geometric, so a
wave train's radial wavelength and damping and an inward ramp's width (its
sharp end is the orbit itself) leave it alone.
"""

import copy
from typing import Any

import numpy as np
import pytest

from spindoctor.nav_model.sim_ring import predict_ring_feature
from spindoctor.sim.render import render_combined_model
from spindoctor.sim.scene import build_nav_params, validate_sim_params
from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType

_SIZE = 200


def _orbit(a_px: float, **extra: Any) -> dict[str, Any]:
    """An eccentric, precessing catalog orbit at ``a_px``."""
    orbit: dict[str, Any] = {
        'a_px': a_px,
        'ae_px': 2.0,
        'long_peri_deg': 33.0,
        'rate_peri_deg_per_day': 1.7,
    }
    orbit.update(extra)
    return orbit


def _edge_wave(
    amp_px: float, wavelength_px: float, edge_wave_damping_deg: float, lam0_deg: float
) -> dict[str, float]:
    """An edge-wave mapping."""
    return {
        'amp_px': amp_px,
        'wavelength_px': wavelength_px,
        'edge_wave_damping_deg': edge_wave_damping_deg,
        'lam0_deg': lam0_deg,
    }


def _scene() -> dict[str, Any]:
    """A scene whose ring system carries every key the cases perturb.

    The scene time sits three days past the ring epoch, so the pericenter
    precession rate moves the orbit, and the render oversamples, so the
    renderer's scaling of each length is exercised as well.
    """
    return {
        'instrument': 'coiss_nac',
        'size_v': _SIZE,
        'size_u': _SIZE,
        'random_seed': 5,
        'oversample': 3,
        'time': 86400.0 * 3,
        'ring_epoch': 0.0,
        'ring_system': {
            'geometry': {
                'center_v': 101.3,
                'center_u': 98.7,
                'opening_deg_obs': 35.0,
                'opening_deg_sun': 25.0,
                'node_deg': 20.0,
            },
            'phase_deg': 30.0,
            'features': [
                {'name': 'SHEET', 'kind': 'edge', 'tau': 0.8, 'orbit': _orbit(150.0)},
                {
                    'name': 'RL',
                    'kind': 'ringlet',
                    'tau': 1.0,
                    'width_px': 7.5,
                    'navigable': True,
                    'orbit': _orbit(
                        60.0,
                        modes=[{'m': 3, 'amp_px': 2.5, 'peri_deg': 70.0}],
                        edge_wave=_edge_wave(1.5, 9.0, 40.0, 110.0),
                    ),
                },
                {
                    'name': 'RAMPOUT',
                    'kind': 'ramp',
                    'tau': 0.4,
                    'width_px': 9.0,
                    'navigable': True,
                    'orbit': _orbit(115.0),
                },
                {
                    'name': 'RAMPIN',
                    'kind': 'ramp',
                    'side': 'in',
                    'tau': 0.4,
                    'width_px': 6.0,
                    'navigable': True,
                    'orbit': _orbit(130.0),
                },
                {
                    'name': 'WV',
                    'kind': 'wave',
                    'tau': 0.3,
                    'wavelength_px': 5.0,
                    'density_wave_damping_px': 11.0,
                    'navigable': True,
                    'orbit': _orbit(75.0),
                },
            ],
            'azimuthal': {
                'modulation': {'amplitude': 0.3, 'm': 2, 'brightness_peak_longitude_deg': 55.0},
                'spokes': {
                    'count': 4,
                    'r_inner_px': 80.0,
                    'r_outer_px': 140.0,
                    'contrast': -0.5,
                    'width_deg': 6.0,
                },
            },
            'moonlets': [{'a_px': 102.0, 'lam_deg': 140.0, 'radius_px': 1.5, 'amplitude': 0.6}],
        },
    }


# Feature indices in _scene()'s feature list.
_RL, _RAMPOUT, _RAMPIN, _WV = 1, 2, 3, 4

# (path under ring_system, whether the navigator's prediction reads the key).
_CASES: list[tuple[tuple[str | int, ...], bool]] = [
    (('features', _RL, 'width_px'), True),
    (('features', _RAMPOUT, 'width_px'), True),
    (('features', _RAMPIN, 'width_px'), False),
    (('features', _WV, 'wavelength_px'), False),
    (('features', _WV, 'density_wave_damping_px'), False),
    *[
        (('features', _RL, 'orbit', key), True)
        for key in ('a_px', 'ae_px', 'long_peri_deg', 'rate_peri_deg_per_day')
    ],
    *[(('features', _RL, 'orbit', 'modes', 0, key), True) for key in ('amp_px', 'peri_deg')],
    *[
        (('features', _RL, 'orbit', 'edge_wave', key), True)
        for key in ('amp_px', 'wavelength_px', 'edge_wave_damping_deg', 'lam0_deg')
    ],
    (('azimuthal', 'modulation', 'brightness_peak_longitude_deg'), False),
    (('azimuthal', 'spokes', 'r_inner_px'), False),
    (('azimuthal', 'spokes', 'r_outer_px'), False),
    (('moonlets', 0, 'a_px'), False),
]
_IDS = ['.'.join(str(step) for step in path) for path, _ in _CASES]


def _render(scene: dict[str, Any]) -> NDArrayFloatType:
    """The rendered image of ``scene``."""
    img, _meta = render_combined_model(copy.deepcopy(scene))
    return np.asarray(img, dtype=np.float64)


def _predictions(scene: dict[str, Any]) -> list[NDArrayBoolType]:
    """Every navigable feature's predicted coverage and edge masks, in order."""
    nav = build_nav_params(scene)
    geometry = nav['ring_system']['geometry']
    masks: list[NDArrayBoolType] = []
    for feature in nav['ring_system']['features']:
        prediction = predict_ring_feature(
            (_SIZE, _SIZE),
            feature,
            center_v=geometry['center_v'],
            center_u=geometry['center_u'],
            opening_deg_obs=geometry['opening_deg_obs'],
            node_deg=geometry['node_deg'],
            time=scene['time'],
            epoch=scene['ring_epoch'],
        )
        masks.append(prediction.mask)
        masks.extend(edge.mask for edge in prediction.edges)
    return masks


def _perturbed(path: tuple[str | int, ...]) -> dict[str, Any]:
    """The scene with the value at ``path`` changed by 30% plus 0.7."""
    scene = _scene()
    node: Any = scene['ring_system']
    for step in path[:-1]:
        node = node[step]
    node[path[-1]] = node[path[-1]] * 1.3 + 0.7
    return validate_sim_params(scene, source='perturbed')


@pytest.fixture(scope='module')
def baseline() -> tuple[NDArrayFloatType, list[NDArrayBoolType]]:
    """The unperturbed scene's render and predictions."""
    scene = validate_sim_params(_scene(), source='baseline')
    return _render(scene), _predictions(scene)


@pytest.mark.parametrize(('path', 'navigator_reads'), _CASES, ids=_IDS)
def test_perturbing_a_key_changes_the_render(
    path: tuple[str | int, ...],
    navigator_reads: bool,
    baseline: tuple[NDArrayFloatType, list[NDArrayBoolType]],
) -> None:
    """The renderer reads the key under its name, so changing it changes the image."""
    image = _render(_perturbed(path))
    assert not np.array_equal(image, baseline[0])


@pytest.mark.parametrize(('path', 'navigator_reads'), _CASES, ids=_IDS)
def test_perturbing_a_key_changes_the_prediction_exactly_when_the_navigator_reads_it(
    path: tuple[str | int, ...],
    navigator_reads: bool,
    baseline: tuple[NDArrayFloatType, list[NDArrayBoolType]],
) -> None:
    """The navigator's prediction moves with each key it reads and with no other."""
    masks = _predictions(_perturbed(path))
    base = baseline[1]
    changed = len(masks) != len(base) or any(
        not np.array_equal(mask, base_mask) for mask, base_mask in zip(masks, base, strict=False)
    )
    assert changed == navigator_reads
