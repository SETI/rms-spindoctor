"""The self-consistency floor: the image PSF equals the navigator's model.

A floor scene sets the image-side PSF equal to the navigator's own model -- a
pure Gaussian at the instrument's configured ``star_psf_sigma``, with no Moffat
wing and no field variation -- via ``optics.psf: {match_navigator: true}``.
The authored form is preserved by validation, saving, and loading (so it
survives an editor round-trip); the renderer resolves it into concrete kernel
parameters only when it builds the kernel.  These tests check the resolution
on coiss_wac, whose configured 0.77 differs from its empirical 0.64.
"""

from pathlib import Path
from typing import Any

import numpy as np

from spindoctor.config import DEFAULT_CONFIG
from spindoctor.sim.forward.optics import effective_psf
from spindoctor.sim.forward.psf import psf_kernel
from spindoctor.sim.instrument_navigator import resolve_sim_inst_config
from spindoctor.sim.scene import load_sim_scene, save_sim_scene, validate_sim_params


def _floor_scene(instrument: str) -> dict[str, Any]:
    """A minimal scene requesting navigator-matched PSF on an instrument."""
    return {
        'schema_version': 2,
        'scene_name': 'floor',
        'instrument': instrument,
        'size_v': 64,
        'size_u': 64,
        'random_seed': 1,
        'optics': {'psf': {'match_navigator': True}},
    }


def test_validate_preserves_the_authored_match_navigator_form() -> None:
    """Validation leaves the authored match-navigator block untouched."""
    scene = validate_sim_params(_floor_scene('coiss_wac'))
    assert scene['optics']['psf'] == {'match_navigator': True}


def test_save_load_round_trips_the_authored_form(tmp_path: Path) -> None:
    """A saved floor scene persists match_navigator and loads it back as authored."""
    scene = _floor_scene('coiss_wac')
    path = tmp_path / 'floor.yaml'
    save_sim_scene(scene, path)
    assert scene['optics']['psf'] == {'match_navigator': True}
    loaded = load_sim_scene(path)
    assert loaded['optics']['psf'] == {'match_navigator': True}


def test_effective_psf_resolves_to_the_configured_wac_sigma() -> None:
    """coiss_wac resolves to a pure 0.77 Gaussian, not the empirical 0.64."""
    psf = effective_psf(_floor_scene('coiss_wac'))
    assert psf is not None
    assert psf['sigma_v'] == 0.77
    assert psf['sigma_u'] == 0.77
    assert psf['w'] == 0.0


def test_resolved_kernel_is_a_pure_gaussian_at_the_configured_sigma() -> None:
    """The floor kernel is the navigator's Gaussian (w = 0, no wing)."""
    psf = effective_psf(_floor_scene('coiss_wac'))
    assert psf is not None
    kernel = psf_kernel(
        psf['sigma_v'],
        psf['sigma_u'],
        psf['w'],
        psf['r0'],
        psf['n'],
        truncation_px=16,
        oversample=1,
    )
    radius = kernel.shape[0] // 2
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    dv, du = np.meshgrid(offsets, offsets, indexing='ij')
    expected = np.exp(-0.5 * (dv**2 + du**2) / 0.77**2)
    expected /= expected.sum()
    assert np.allclose(kernel, expected, atol=1e-12)


def test_match_navigator_tracks_the_instrument_config() -> None:
    """The resolved sigma equals the emulated instrument's star_psf_sigma."""
    psf = effective_psf(_floor_scene('coiss_wac'))
    assert psf is not None
    inst = resolve_sim_inst_config(DEFAULT_CONFIG, 'coiss_wac', None)
    assert psf['sigma_v'] == float(inst['star_psf_sigma'])


def test_floor_psf_must_be_explicit() -> None:
    """A scene with no optics block plants no PSF at all (stage activation)."""
    scene = _floor_scene('coiss_wac')
    del scene['optics']
    validated = validate_sim_params(scene)
    assert 'optics' not in validated
    assert effective_psf(validated) is None
