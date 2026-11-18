"""ObsSim construction: renderer output metadata and scene-path handling.

``ObsSim.from_file`` renders the scene and stores the renderer's truth-side
output on the snapshot for the image-side consumers (recovery scoring, the
backplane writer).  These tests guard the per-body mask map: a renderer
change that stops populating ``body_mask_map`` degrades those consumers
silently, because nothing on the navigation path reads it.

They also check that a scene supplied in memory is validated before it renders,
that a scene file is validated once although it is both loaded and rendered, and
how the scene path is resolved.  The path of a scene handed over in memory is only
a label, so resolving it must not reach storage; a scene read from a file or a URL must be
fetched before it can be parsed.

Last, they check that the faint-star limit follows the frame's readout binning, reading
each mode's values from the navigator's own block, and that an unbinned frame's limit
reads the unbinned values.
"""

import math
import re
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from spindoctor.obs.obs_inst_sim import ObsSim
from spindoctor.sim.scene.io import save_sim_scene
from spindoctor.sim.scene.rules_common import SimSceneValidationError
from spindoctor.sim.scene.truth_echo import SimSceneTruthEchoWarning

_SIZE = 96


def _body_scene() -> dict[str, Any]:
    """A minimal scene with one named ellipsoid body."""
    return {
        'size_v': _SIZE,
        'size_u': _SIZE,
        'instrument': 'coiss_nac',
        'random_seed': 7,
        'bodies': [
            {
                'name': 'PEBBLE',
                'center_v': _SIZE / 2.0,
                'center_u': _SIZE / 2.0,
                'semi_a_px': 20.0,
                'semi_b_px': 15.0,
            }
        ],
    }


def test_body_mask_map_is_populated() -> None:
    """A scene with a body yields a non-empty per-body mask map."""
    obs = ObsSim.from_file('/tmp/obs_sim_body.yaml', sim_params=_body_scene())
    assert obs.sim_body_mask_map
    assert 'PEBBLE' in obs.sim_body_mask_map


def test_body_mask_shape_matches_the_render() -> None:
    """Each body mask is a boolean plane the shape of the rendered image."""
    obs = ObsSim.from_file('/tmp/obs_sim_body.yaml', sim_params=_body_scene())
    mask = obs.sim_body_mask_map['PEBBLE']
    assert mask.dtype == np.bool_
    assert mask.shape == np.asarray(obs.data).shape
    assert mask.any()


def test_invalid_in_memory_scene_is_refused() -> None:
    """A scene supplied in memory is validated, and the error starts with its label."""
    scene = {**_body_scene(), 'exposure_seconds': 2.0}
    with pytest.raises(
        SimSceneValidationError,
        match='^' + re.escape("obs_sim_bad.yaml: unknown scene keys: ['exposure_seconds']"),
    ):
        ObsSim.from_file('obs_sim_bad.yaml', sim_params=scene)


def test_in_memory_scene_creates_nothing_beside_the_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A scene supplied in memory leaves the working directory untouched.

    The scene exists only as a mapping, so its name is a label rather than a
    file: resolving it against storage would materialize a directory in
    whatever directory the process happens to be running in.
    """
    monkeypatch.chdir(tmp_path)
    obs = ObsSim.from_file('sim://in_memory', sim_params=_body_scene())
    assert list(tmp_path.iterdir()) == []
    assert obs.abspath.name == 'in_memory'


def test_scene_file_supplies_the_sim_params(tmp_path: Path) -> None:
    """With no sim_params supplied the scene is read from the named file."""
    scene_path = tmp_path / 'pebble.yaml'
    save_sim_scene(_body_scene(), scene_path)
    obs = ObsSim.from_file(scene_path)
    assert obs.sim_params['scene_name'] == 'pebble'
    assert np.asarray(obs.data).shape == (_SIZE, _SIZE)


def test_a_scene_file_is_validated_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A scene file that ObsSim loads and then renders prints its truth echo once.

    The scene is given by a relative path, which ObsSim resolves to an absolute one
    for loading, so the render must label the scene as the load did.
    """
    monkeypatch.chdir(tmp_path)
    scene = {
        **_body_scene(),
        'noise': {'read_noise_dn': 4.0},
        'instrument_config': {'noise': {'read_noise_dn': 4.0}},
    }
    scene_path = tmp_path / 'echo.yaml'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', SimSceneTruthEchoWarning)
        save_sim_scene(scene, scene_path)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        ObsSim.from_file('echo.yaml')
    echoes = [w for w in caught if issubclass(w.category, SimSceneTruthEchoWarning)]
    assert len(echoes) == 1


def test_scene_url_is_fetched_before_it_is_parsed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A scene named by URL is brought to local storage before it is read.

    A ``file://`` URL is not a path any reader can open as it stands; only the
    file cache turns it into one.  Merely rendering the URL absolute leaves a
    path that does not exist, so the scene must be fetched.
    """
    monkeypatch.chdir(tmp_path)
    scene_path = tmp_path / 'pebble_url.yaml'
    save_sim_scene(_body_scene(), scene_path)
    # as_uri() rather than an f-string: it escapes the characters a URL has to
    # escape and spells a drive letter the way a URL spells one, neither of
    # which an interpolated POSIX path does.
    obs = ObsSim.from_file(scene_path.as_uri())
    assert obs.sim_params['scene_name'] == 'pebble_url'


def _star_limit_obs(instrument: str, exposure_sec: float, *, binning: int | None = None) -> ObsSim:
    """A star-free frame, binned by ``binning`` when given."""
    scene: dict[str, Any] = {
        'instrument': instrument,
        'size_v': 64,
        'size_u': 64,
        'random_seed': 1,
        'exposure_sec': exposure_sec,
    }
    if binning is not None:
        scene['detector'] = {'binning': binning}
    return ObsSim.from_file('/tmp/obs_sim_star_limit.yaml', sim_params=scene)


def _star_limit(instrument: str, exposure_sec: float, *, binning: int | None = None) -> float:
    """The faint-star limit of a star-free frame, binned by ``binning`` when given."""
    return _star_limit_obs(instrument, exposure_sec, binning=binning).star_max_usable_vmag()


@pytest.mark.parametrize(
    ('instrument', 'exposure_sec', 'expected'),
    [
        ('nhlorri', 5.0, 11.414754682632388),
        ('coiss_nac', 1.0, 10.890967146275809),
        ('coiss_wac', 1.0, 8.734411020746549),
        ('gossi', 1.0, 5.692403499719424),
    ],
)
def test_unbinned_star_limit_is_unchanged(
    instrument: str, exposure_sec: float, expected: float
) -> None:
    """An unbinned frame's limit is the value the navigator's published block gives.

    The expected limits are fixed numbers on purpose: each is the limit the formula
    gives from the camera's unbinned zero point, read noise, and PSF core at that
    exposure, so a change to the formula or to the values it reads shows here, where an
    expected value derived through the same formula would change with it.
    """
    assert _star_limit(instrument, exposure_sec) == pytest.approx(expected, rel=0.0, abs=1e-12)


@pytest.mark.parametrize(
    ('instrument', 'binning', 'expected_shift_mag'),
    [
        # From the navigator's LORRI block: unbinned, 3.33e6 DN/s over 4.0 DN of read
        # noise; binned 4x4, 3.61e6 DN/s over 1.19 DN, with the 3.0 px core shrunk to
        # 0.75 px, so on a sixteenth of the pixels.
        ('nhlorri', 4, 2.5 * math.log10((3.61e6 / 3.33e6) * (4.0 / 1.19) * 4**2)),
        # From the navigator's NAC block: unbinned, 3.33e5 DN/s over 4.0 DN; summed
        # 2x2, the same zero point over 0.400 DN, which the 1 DN floor raises to 1.0,
        # and the 0.54 px core shrinks below one pixel, which then holds it all.
        ('coiss_nac', 2, 2.5 * math.log10((4.0 / 1.0) * 2.0 * math.pi * 0.54**2)),
    ],
    ids=['nhlorri-4x4', 'coiss_nac-2x2'],
)
def test_binned_star_limit_reads_the_modes_values(
    instrument: str, binning: int, expected_shift_mag: float
) -> None:
    """A binned frame's limit reads its mode's zero point and read noise.

    A star's total charge is the same in both frames, so the limit moves by 2.5 log10 of
    how much the peak pixel's DN, over the read noise per pixel, grows.
    """
    unbinned = _star_limit(instrument, 1.0)
    binned = _star_limit(instrument, 1.0, binning=binning)
    assert binned - unbinned == pytest.approx(expected_shift_mag, rel=0.0, abs=1e-9)


def test_a_mode_without_values_reads_the_unbinned_ones() -> None:
    """With no entry for its mode, a binned frame's limit moves by its smaller core alone.

    The LORRI 3.0 px core shrinks to 0.75 px on a 4x4-binned frame, so the peak pixel
    holds 16 times the share of the star at the same zero point and read noise.
    """
    obs = _star_limit_obs('nhlorri', 1.0, binning=4)
    assert obs._inst_config is not None
    noise = {k: v for k, v in obs._inst_config['noise'].items() if k != 'by_binning'}
    obs._inst_config = {**obs._inst_config, 'noise': noise}
    shift = obs.star_max_usable_vmag() - _star_limit('nhlorri', 1.0)
    assert shift == pytest.approx(2.5 * math.log10(4**2), rel=0.0, abs=1e-9)
