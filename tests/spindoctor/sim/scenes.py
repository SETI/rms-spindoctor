"""Shared base scenes and scene blocks for the simulator tests.

A scene is the flat mapping the simulator renders.  Most tests need the same frame
header (instrument, image size, and random seed) plus one or two blocks, so the
header and the common blocks are built here, once.  Every function returns a fresh
mapping, and deep-copies the values a caller passes in, so no test can change a
scene another test holds.  They are plain functions rather than fixtures so that a
module can build parametrize lists and constants from them at import time;
``conftest.py`` beside this module wraps the base scenes as fixtures for tests that
prefer them.
"""

import copy
from typing import Any

from spindoctor.sim.scene.validate import CURRENT_SCHEMA_VERSION

__all__ = [
    'body_scene',
    'disc_body',
    'frame_scene',
    'quiet_noise',
    'ring_system_block',
    'star_entry',
]


def quiet_noise() -> dict[str, Any]:
    """A ``noise`` block that turns off shot noise, read noise, and the bias pedestal.

    Returns:
        A fresh ``noise`` mapping.
    """
    return {'poisson': False, 'read_noise_dn': 0.0, 'bias_dn': 0.0}


def frame_scene(
    *,
    instrument: str = 'coiss_nac',
    size: int | tuple[int, int] = 64,
    seed: int = 1,
    scene_name: str | None = None,
    **blocks: Any,
) -> dict[str, Any]:
    """An empty frame: the scene header and nothing in view unless a block adds it.

    Parameters:
        instrument: The ``instrument`` key.
        size: The image size in pixels, one number for a square frame or a
            ``(size_v, size_u)`` pair.
        seed: The ``random_seed`` key.
        scene_name: When given, the scene also carries ``scene_name`` and the
            current ``schema_version``, as a scene file does.
        **blocks: Further top-level scene keys (``bodies``, ``noise``,
            ``exposure_sec``, and so on), deep-copied into the scene.

    Returns:
        A fresh scene mapping.
    """
    size_v, size_u = (size, size) if isinstance(size, int) else size
    scene: dict[str, Any] = {}
    if scene_name is not None:
        scene['schema_version'] = CURRENT_SCHEMA_VERSION
        scene['scene_name'] = scene_name
    scene.update(
        {'instrument': instrument, 'size_v': size_v, 'size_u': size_u, 'random_seed': seed}
    )
    scene.update(copy.deepcopy(blocks))
    return scene


def disc_body(
    center_v: float, center_u: float, radius: float, *, name: str = 'B', **keys: Any
) -> dict[str, Any]:
    """A spherical body entry: all three semi-axes equal ``radius``.

    Parameters:
        center_v: The body center's ``center_v``.
        center_u: The body center's ``center_u``.
        radius: The ``semi_a_px``, ``semi_b_px``, and ``semi_c_px`` of the body.
        name: The body's ``name``.
        **keys: Further body keys (``illumination_angle``, ``range_km``, and so on),
            deep-copied into the entry.

    Returns:
        A fresh body mapping, ready for a scene's ``bodies`` list.
    """
    return {
        'name': name,
        'center_v': center_v,
        'center_u': center_u,
        'semi_a_px': radius,
        'semi_b_px': radius,
        'semi_c_px': radius,
        **copy.deepcopy(keys),
    }


def body_scene(
    radius: float = 10.0, *, body: dict[str, Any] | None = None, **frame: Any
) -> dict[str, Any]:
    """A frame holding one fully lit spherical body, named RHEA, at the frame center.

    The body is lit from the observer's direction: its ``illumination_angle`` and
    ``phase_angle`` are both 0.

    Parameters:
        radius: The body's semi-axes, in pixels.
        body: Body keys that add to or replace the default body's keys.
        **frame: Keyword arguments for :func:`frame_scene`.

    Returns:
        A fresh scene mapping.
    """
    scene = frame_scene(**frame)
    entry = disc_body(
        scene['size_v'] / 2.0,
        scene['size_u'] / 2.0,
        radius,
        name='RHEA',
        illumination_angle=0.0,
        phase_angle=0.0,
    )
    if body is not None:
        entry.update(copy.deepcopy(body))
    scene['bodies'] = [entry]
    return scene


def star_entry(v: float, u: float, vmag: float, *, name: str = 'S', **keys: Any) -> dict[str, Any]:
    """A star entry.

    Parameters:
        v: The star's ``v`` position.
        u: The star's ``u`` position.
        vmag: The star's visual magnitude.
        name: The star's ``name``.
        **keys: Further star keys (``psf_sigma``, ``catalog_error_v``, and so on),
            deep-copied into the entry.

    Returns:
        A fresh star mapping, ready for a scene's ``stars`` list.
    """
    return {'name': name, 'v': v, 'u': u, 'vmag': vmag, **copy.deepcopy(keys)}


def ring_system_block(
    center_v: float,
    center_u: float,
    features: list[dict[str, Any]],
    *,
    opening_deg_obs: float = 90.0,
    opening_deg_sun: float = 90.0,
    node_deg: float = 0.0,
    **keys: Any,
) -> dict[str, Any]:
    """A ``ring_system`` block, seen face-on and lit face-on unless told otherwise.

    Parameters:
        center_v: The ring center's ``center_v``.
        center_u: The ring center's ``center_u``.
        features: The ring features, deep-copied into the block.
        opening_deg_obs: The ring opening angle toward the observer, in degrees.
        opening_deg_sun: The ring opening angle toward the Sun, in degrees.
        node_deg: The sky position angle of the ascending node, in degrees.
        **keys: Further ``ring_system`` keys (``phase_deg``, ``azimuthal``, and so
            on), deep-copied into the block.

    Returns:
        A fresh ``ring_system`` mapping.
    """
    return {
        'geometry': {
            'center_v': center_v,
            'center_u': center_u,
            'opening_deg_obs': opening_deg_obs,
            'opening_deg_sun': opening_deg_sun,
            'node_deg': node_deg,
        },
        'features': copy.deepcopy(features),
        **copy.deepcopy(keys),
    }
