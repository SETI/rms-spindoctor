"""Everything a CCD frame collects follows its integration time.

A camera whose entry states an extra integration time collects for its commanded
exposure plus that time.  Each case turns on one charge the frame collects and renders
the scene twice: once on a fictional camera that states an extra time, and once on
the generic camera, which states none, commanded for the sum of the two times.  The
two frames must be the same pixel for pixel, which holds only when the term follows
the integration time.  The cameras share the generic detector's hot-pixel map, so
only the timing differs between them.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from tests.spindoctor.sim.fictional_camera import catalog_dir_with, keyed

from spindoctor.sim.instruments import catalog as instrument_catalog
from spindoctor.sim.render.driver import render_scene
from spindoctor.sim.render.radiance.compose import sky_star_faint_cutoff_mag

# Binary fractions, so the commanded exposure plus the extra time is exact.
_EXPOSURE_SEC = 0.0625
_EXTRA_SEC = 0.0625
_CAMERA = 'slow_cam'


@pytest.fixture
def slow_camera(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A catalog holding a generic CCD that collects 0.0625 s beyond its exposure."""
    entry = keyed(
        _CAMERA,
        f'base: generic\ndetector_id: generic\ndetector: {{extra_integration_sec: {_EXTRA_SEC}}}\n',
    )
    directory = catalog_dir_with(tmp_path, {'slow_cam.yaml': entry})
    monkeypatch.setattr(instrument_catalog, 'INSTRUMENT_CONFIG_DIR', directory)


def _scene(instrument: str, exposure_sec: float, **extra: Any) -> dict[str, Any]:
    """A small empty frame on ``instrument``, with the keys that turn one term on."""
    scene: dict[str, Any] = {
        'schema_version': 2,
        'scene_name': 'integration_time',
        'size_v': 96,
        'size_u': 96,
        'random_seed': 11,
        'instrument': instrument,
        'exposure_sec': exposure_sec,
        **extra,
    }
    return scene


# Each case: the scene keys that turn one collected term on, and the case's id.
_TERMS: list[tuple[dict[str, Any], str]] = [
    ({'sky_counts': {'density_factor': 1.0e5}}, 'sky-stars'),
    ({'noise': {'dark_current_e_per_sec': 4000.0}}, 'dark-current'),
    ({'noise': {'cosmic_ray_events_per_pixel_per_sec': 0.02}}, 'cosmic-rays'),
    (
        {'artifacts': {'radiation_transients': {'expected_events_per_sec': 400.0}}},
        'radiation-transients',
    ),
    (
        {'artifacts': {'hot_pixels': {'pixel_fraction': 0.02, 'rate_e_per_sec': 3000.0}}},
        'hot-pixels',
    ),
    (
        {
            'artifacts': {
                'bright_dark_pairs': {'expected_events_per_sec': 400.0, 'amplitude_e': 2000.0}
            }
        },
        'bright-dark-pairs',
    ),
]


@pytest.mark.parametrize('extra', [case for case, _ in _TERMS], ids=[i for _, i in _TERMS])
@pytest.mark.usefixtures('slow_camera')
def test_a_collected_term_follows_the_integration_time(extra: dict[str, Any]) -> None:
    """A frame of exposure t plus extra time d renders as a frame of t + d."""
    slow, _ = render_scene(_scene(_CAMERA, _EXPOSURE_SEC, **extra))
    summed, _ = render_scene(_scene('generic', _EXPOSURE_SEC + _EXTRA_SEC, **extra))
    assert np.array_equal(slow, summed)


@pytest.mark.parametrize('extra', [case for case, _ in _TERMS], ids=[i for _, i in _TERMS])
@pytest.mark.usefixtures('slow_camera')
def test_a_collected_term_shows_the_extra_time(extra: dict[str, Any]) -> None:
    """Each term differs between the extra time and none, so the check above has teeth."""
    slow, _ = render_scene(_scene(_CAMERA, _EXPOSURE_SEC, **extra))
    commanded, _ = render_scene(_scene('generic', _EXPOSURE_SEC, **extra))
    assert not np.array_equal(slow, commanded)


@pytest.mark.usefixtures('slow_camera')
def test_the_sky_stars_stop_where_the_integration_time_puts_them() -> None:
    """The faint cutoff of the background sky follows the integration time."""
    slow = sky_star_faint_cutoff_mag(_scene(_CAMERA, _EXPOSURE_SEC, sky_counts={}))
    summed = sky_star_faint_cutoff_mag(_scene('generic', _EXPOSURE_SEC + _EXTRA_SEC, sky_counts={}))
    assert slow == summed


@pytest.mark.usefixtures('slow_camera')
def test_the_vidicon_chain_collects_for_the_exposure_alone() -> None:
    """A camera with an extra time, rendered through the vidicon chain, takes none."""
    extra = {
        'detector': {'detector_model': 'vidicon'},
        'noise': {'cosmic_ray_events_per_pixel_per_sec': 0.02},
        'stars': [{'name': 'S1', 'v': 40.0, 'u': 50.0, 'vmag': 6.0}],
    }
    slow, _ = render_scene(_scene(_CAMERA, _EXPOSURE_SEC, **extra))
    commanded, _ = render_scene(_scene('generic', _EXPOSURE_SEC, **extra))
    assert np.array_equal(slow, commanded)


@pytest.fixture
def slow_calibrated_camera(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A catalog holding a calibrated CCD that collects 0.0625 s beyond its exposure."""
    entry = keyed(
        _CAMERA,
        f'base: coiss_calib_nac\ndetector: {{extra_integration_sec: {_EXTRA_SEC}}}\n',
    )
    directory = catalog_dir_with(tmp_path, {'slow_cam.yaml': entry})
    monkeypatch.setattr(instrument_catalog, 'INSTRUMENT_CONFIG_DIR', directory)


@pytest.mark.usefixtures('slow_calibrated_camera')
def test_a_calibrated_frame_keeps_no_dark_pedestal() -> None:
    """The inverse removes all the dark charge, so dark current leaves no pedestal."""
    # Over the 0.125 s integration time this collects 300 e-, exactly 10 DN at 30 e-/DN, so
    # the digitized dark is a whole number of DN and the comparison below is exact.
    dark = {'noise': {'dark_current_e_per_sec': 2400.0}}
    with_dark, _ = render_scene(_scene(_CAMERA, _EXPOSURE_SEC, **dark))
    without_dark, _ = render_scene(_scene(_CAMERA, _EXPOSURE_SEC))
    assert np.max(np.abs(with_dark - without_dark)) < 1.0e-9
