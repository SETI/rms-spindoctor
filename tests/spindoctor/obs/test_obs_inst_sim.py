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
"""

import re
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from spindoctor.obs.obs_inst_sim import ObsSim
from spindoctor.sim.scene import (
    SimSceneTruthEchoWarning,
    SimSceneValidationError,
    save_sim_scene,
)

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
                'axis1': 40.0,
                'axis2': 30.0,
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
