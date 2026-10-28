"""The dark sky of a calibrated frame stays off the missing-data marker.

Every calibrated instrument marks a lost pixel with 0, so a valid pixel must never
read exactly 0.  The calibration inverse adds the instrument's calibrated sky
residual and then moves any pixel that still lands on a numeric marker to the next
float above it.  The sky then looks the way the real products show it: a Voyager
GEOMED sky sits above 0, and a Cassini CALIB sky scatters around 0 with negatives.
"""

from typing import Any

import numpy as np
import pytest

from spindoctor.sim.forward.artifacts_catalog import resolve_detector_defaults
from spindoctor.sim.render import render_combined_model

_CALIBRATED = ('coiss_calib_nac', 'coiss_calib_wac', 'vgiss')
_RENDERS = {'floor': {}, 'instrument_defaults': {'artifacts': {'instrument_defaults': True}}}


def _disc(instrument: str, **extra: Any) -> dict[str, Any]:
    """A 128 x 128 frame whose lit disc leaves about two thirds of it dark sky."""
    scene: dict[str, Any] = {
        'size_v': 128,
        'size_u': 128,
        'instrument': instrument,
        'random_seed': 42,
        'offset_v': 3.0,
        'offset_u': -2.0,
        'bodies': [
            {
                'name': 'RHEA',
                'center_v': 64.0,
                'center_u': 64.0,
                'axis1': 86.0,
                'axis2': 86.0,
                'axis3': 86.0,
                'illumination_angle': 18.0,
                'phase_angle': 30.0,
            }
        ],
    }
    scene.update(extra)
    return scene


def _sky(instrument: str, render: str) -> np.ndarray:
    """The dark-sky pixels of the disc frame: those at the floor render's minimum."""
    floor, _ = render_combined_model(_disc(instrument))
    img, _ = render_combined_model(_disc(instrument, **_RENDERS[render]))
    return np.asarray(img[floor == floor.min()])


@pytest.mark.parametrize('render', sorted(_RENDERS))
@pytest.mark.parametrize('instrument', _CALIBRATED)
def test_no_pixel_of_a_calibrated_frame_reads_zero(instrument: str, render: str) -> None:
    """A calibrated frame without planted loss carries no pixel at the 0 marker."""
    img, _ = render_combined_model(_disc(instrument, **_RENDERS[render]))
    assert int((img == 0.0).sum()) == 0


@pytest.mark.parametrize('render', sorted(_RENDERS))
def test_vgiss_sky_sits_above_zero(render: str) -> None:
    """A vgiss sky's 1st percentile is positive, as inside a real GEOMED border."""
    assert float(np.percentile(_sky('vgiss', render), 1)) > 0.0


@pytest.mark.parametrize('instrument', ['coiss_calib_nac', 'coiss_calib_wac'])
def test_cassini_calibrated_floor_sky_sits_near_zero(instrument: str) -> None:
    """A noise-free Cassini CALIB sky sits within the real sky's 1.9e-4 I/F sigma of 0."""
    sky = _sky(instrument, 'floor')
    assert float(np.abs(sky).min()) > 0.0
    assert float(np.abs(sky).max()) < 1.9e-4


@pytest.mark.parametrize('instrument', ['coiss_calib_nac', 'coiss_calib_wac'])
def test_cassini_calibrated_sky_scatters_around_zero(instrument: str) -> None:
    """Under instrument_defaults a Cassini CALIB sky has negative and positive pixels."""
    negative = float((_sky(instrument, 'instrument_defaults') < 0.0).mean())
    assert 0.2 < negative < 0.8


def _cassini_coincidences(instrument: str) -> list[tuple[int, float]]:
    """Gain states and exposures where the dark pedestal puts a DN on exactly 0 I/F.

    Under instrument_defaults the inverse subtracts the dark pedestal
    ``dark_current * exposure / gain`` DN and adds the sky residual, which is
    ``residual * frac * full_well / gain`` DN at the same exposure.  Their
    difference is a whole number of DN, so an integer DN lands on exactly 0 I/F,
    at ``k * gain / (dark_current - residual * frac * full_well)`` seconds.
    """
    catalog = resolve_detector_defaults(instrument)
    per_sec = float(catalog['dark_current_e_per_sec']) - 4.0e-5 * 0.5 * float(
        catalog['full_well_e']
    )
    return [
        (state, k * gain / per_sec)
        for state, gain in catalog['gain_e_per_dn_by_state'].items()
        for k in (1, 2)
    ]


@pytest.mark.parametrize(
    ('instrument', 'gain_state', 'exposure_sec'),
    [
        (instrument, state, exposure)
        for instrument in ('coiss_calib_nac', 'coiss_calib_wac')
        for state, exposure in _cassini_coincidences(instrument)
    ],
)
def test_no_pixel_reads_zero_where_the_lattice_crosses_it(
    instrument: str, gain_state: int, exposure_sec: float
) -> None:
    """At an exposure whose DN lattice crosses 0 I/F, no sky pixel reads exactly 0."""
    params = {
        'size_v': 64,
        'size_u': 64,
        'instrument': instrument,
        'random_seed': 3,
        'exposure_sec': exposure_sec,
        'detector': {'gain_state': gain_state},
        'artifacts': {'instrument_defaults': True},
    }
    img, _ = render_combined_model(params)
    assert int((img == 0.0).sum()) == 0


@pytest.mark.parametrize('sigma_dn', [0.25, 1.0, 3.0])
def test_vgiss_sky_never_reads_zero_at_any_noise_level(sigma_dn: float) -> None:
    """The vidicon's residual is half a DN off the DN grid, so noise never lands on 0."""
    params = {
        'size_v': 64,
        'size_u': 64,
        'instrument': 'vgiss',
        'random_seed': 3,
        'noise': {'vidicon': {'read_noise_pixel_dn': sigma_dn}},
    }
    img, _ = render_combined_model(params)
    assert int((img == 0.0).sum()) == 0


def test_raw_frames_carry_no_sky_residual() -> None:
    """A raw-DN frame's dark sky stays at the bias pedestal, in whole DN."""
    sky = _sky('coiss_nac', 'floor')
    assert set(np.unique(sky).tolist()) == {20.0}


@pytest.mark.parametrize('instrument', ['coiss_calib_nac', 'coiss_calib_wac'])
def test_cassini_calibrated_floor_sky_is_the_measured_residual(instrument: str) -> None:
    """A noise-free Cassini CALIB sky reads 4.0e-5 I/F, the real dark-sky median."""
    assert set(np.unique(_sky(instrument, 'floor')).tolist()) == {4.0e-5}


def test_vgiss_floor_sky_is_one_and_a_half_dn() -> None:
    """A noise-free vgiss sky reads 1.5 DN at 127.5 DN per unit I/F."""
    sky = np.unique(_sky('vgiss', 'floor'))
    assert sky.tolist() == [pytest.approx(1.5 / 127.5, rel=1e-12)]
