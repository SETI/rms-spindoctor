"""Whole-scene PSF shaping of the simulator's point-source stars.

Stars deposit as point masses; the scene optics PSF is their only convolution,
so the rendered star spread scales with the PSF sigma.  The navigator-matched
floor form (``optics.psf: {match_navigator: true}``) resolves to the emulated
instrument's configured ``star_psf_sigma``, so a coiss frame renders tighter
stars than a gossi frame.  A faint magnitude keeps the star unsaturated so the
moment measurement sees the whole profile.
"""

from typing import Any

import numpy as np

from spindoctor.config import DEFAULT_CONFIG
from spindoctor.sim.instrument_navigator import resolve_sim_inst_config
from spindoctor.sim.render import render_combined_model


def _rms_spread(image: np.ndarray) -> float:
    """Intensity-weighted RMS radius of a single star patch."""
    vv, uu = np.mgrid[0 : image.shape[0], 0 : image.shape[1]]
    total = float(image.sum())
    cv = float((image * vv).sum()) / total
    cu = float((image * uu).sum()) / total
    var = float((image * ((vv - cv) ** 2 + (uu - cu) ** 2)).sum()) / total
    return float(np.sqrt(var))


def _render_one_star(sigma: float) -> np.ndarray:
    """Render a single centered faint star through a scene PSF of the given sigma."""
    scene: dict[str, Any] = {
        'size_v': 41,
        'size_u': 41,
        'random_seed': 1,
        'instrument': 'coiss_nac',
        'exposure_sec': 1.0,
        'noise': {'poisson': False, 'read_noise_dn': 0.0, 'bias_dn': 0.0},
        'optics': {'psf': {'sigma_v': sigma, 'sigma_u': sigma, 'w': 0.0, 'r0': 2.0, 'n': 3.0}},
        'stars': [{'name': 's', 'v': 20.0, 'u': 20.0, 'vmag': 8.0}],
    }
    img, _ = render_combined_model(scene)
    return img


def test_spread_increases_with_sigma() -> None:
    """A larger scene PSF sigma yields a broader rendered star."""
    narrow = _rms_spread(_render_one_star(0.6))
    wide = _rms_spread(_render_one_star(3.0))
    assert wide > narrow


def test_spread_monotonic_across_sigmas() -> None:
    """Rendered star spread rises monotonically with the scene PSF sigma."""
    spreads = [_rms_spread(_render_one_star(s)) for s in (0.6, 1.0, 2.0, 3.0)]
    assert all(spreads[i + 1] > spreads[i] for i in range(len(spreads) - 1))


def test_coiss_and_gossi_psf_sigmas_differ() -> None:
    """The two instruments declare different star PSF sigmas (test premise)."""
    coiss = float(resolve_sim_inst_config(DEFAULT_CONFIG, 'coiss_nac')['star_psf_sigma'])
    gossi = float(resolve_sim_inst_config(DEFAULT_CONFIG, 'gossi')['star_psf_sigma'])
    assert coiss < gossi


def _match_navigator_star_scene(instrument: str, *, size: int = 41) -> dict[str, Any]:
    """A single faint star through the instrument's navigator-matched PSF."""
    return {
        'size_v': size,
        'size_u': size,
        'random_seed': 1,
        'instrument': instrument,
        'exposure_sec': 1.0,
        'noise': {'poisson': False, 'read_noise_dn': 0.0, 'bias_dn': 0.0},
        'optics': {'psf': {'match_navigator': True}},
        # Bright enough to survive gossi's coarse gain, faint enough to stay
        # below the coiss_nac full well: both frames keep an unclipped profile.
        'stars': [{'name': 's', 'v': size / 2, 'u': size / 2, 'vmag': 5.5}],
    }


def test_render_uses_instrument_psf_sigma() -> None:
    """A coiss frame renders tighter stars than a gossi frame end-to-end."""
    coiss_img, _ = render_combined_model(_match_navigator_star_scene('coiss_nac'))
    gossi_img, _ = render_combined_model(_match_navigator_star_scene('gossi'))
    assert _rms_spread(coiss_img) < _rms_spread(gossi_img)
