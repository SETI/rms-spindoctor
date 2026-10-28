"""Matched sim-scene construction for the Section 7 realism runner.

Every real cohort frame gets one matched simulated frame: same instrument
signal chain (``artifacts.instrument_defaults``), same exposure, and scene
content of the same class -- a star field for a star frame, an ellipsoid
whose apparent diameter and phase angle mirror the real body for a limb
frame, a multi-feature ring system for a ring frame, a bare sky for the
negative / scattered-light frames.  Matching content class and exposure is
what makes the FOM 5 strata and the FOM 3 bins compare the forward model
rather than what the spacecraft happened to point at.

Scenes are deterministic: the random seed derives from the real frame's
image_id, so re-running the runner re-renders identical frames.

The matched scenes plant **zero offset**: realism statistics measure image
texture, not recovery, so predicted feature positions equal actual ones and
no shifting is needed on the sim side.
"""

from __future__ import annotations

import zlib
from typing import Any

from spindoctor.sim.scene import validate_sim_params

__all__ = [
    'limb_scene',
    'matched_scene',
    'ring_scene',
    'sky_scene',
    'star_field_scene',
]

# Fallback matched-frame size when the caller supplies no real frame shape:
# large enough for 32-px sky patches and the PSD's 64-px tiles.
_DEFAULT_SCENE_SIZE = 512

# Star classes place this many stars; magnitudes ladder from bright to the
# detection edge so the FOM 2 sample spans the usable brightness range.
_STAR_COUNT = 25
_STAR_VMAG_BRIGHT = 7.0
_STAR_VMAG_STEP = 0.2

# Scene classes mapped to matched content builders (sidecar primary tags).
_STAR_CLASSES = frozenset(
    {
        'star_dominated',
        'stars_plus_body',
        'faint_stars',
        'one_bright_star_no_body',
        'two_bright_stars_no_body',
    }
)
_LIMB_CLASSES = frozenset(
    {'body_full_fov', 'body_partial_overflow', 'high_phase_terminator', 'multi_body'}
)
_RING_CLASSES = frozenset({'ring_only_flat', 'ring_only_curved', 'ring_plus_body'})


def _seed_for(image_id: str) -> int:
    """Deterministic scene seed from the matched real frame's image_id."""
    return int(zlib.crc32(image_id.encode('utf-8')) & 0x7FFFFFFF)


def _base_scene(
    scene_name: str, instrument: str, exposure_sec: float, seed: int, size_vu: tuple[int, int]
) -> dict[str, Any]:
    """The shared scene skeleton: instrument chain on, zero offset."""
    return {
        'schema_version': 2,
        'scene_name': scene_name,
        'instrument': instrument,
        'size_v': int(size_vu[0]),
        'size_u': int(size_vu[1]),
        'random_seed': seed,
        'exposure_sec': float(exposure_sec),
        'offset_v': 0.0,
        'offset_u': 0.0,
        'artifacts': {'instrument_defaults': True},
        'bodies': [],
    }


def _star_list(
    seed: int, count: int, size_vu: tuple[int, int], *, vmag_bright: float, vmag_step: float
) -> list[dict[str, Any]]:
    """A deterministic scattered star field with a magnitude ladder."""
    # A small linear-congruential walk keeps this independent of numpy so
    # scene construction (not rendering) never draws from a global RNG.
    state = seed or 1
    stars: list[dict[str, Any]] = []
    margin = 24.0
    span = min(size_vu) - 2 * margin
    for i in range(count):
        state = (1103515245 * state + 12345) % (1 << 31)
        v = margin + span * (state / float(1 << 31))
        state = (1103515245 * state + 12345) % (1 << 31)
        u = margin + span * (state / float(1 << 31))
        stars.append(
            {
                'name': f'R{i}',
                'v': round(v, 2),
                'u': round(u, 2),
                'vmag': round(vmag_bright + vmag_step * i, 2),
            }
        )
    return stars


def sky_scene(
    image_id: str, instrument: str, exposure_sec: float, size_vu: tuple[int, int]
) -> dict[str, Any]:
    """A matched frame for negative / scattered-light / offscreen classes.

    A sparse dim star field keeps the frame from being pathologically
    empty while leaving nearly every patch pure detector output.
    """
    seed = _seed_for(image_id)
    scene = _base_scene(f'realism_sky_{image_id}', instrument, exposure_sec, seed, size_vu)
    scene['stars'] = _star_list(seed, 5, size_vu, vmag_bright=10.0, vmag_step=0.5)
    return scene


def star_field_scene(
    image_id: str,
    instrument: str,
    exposure_sec: float,
    size_vu: tuple[int, int],
    *,
    count: int = _STAR_COUNT,
) -> dict[str, Any]:
    """A matched star-field frame for the star scene classes."""
    seed = _seed_for(image_id)
    scene = _base_scene(f'realism_stars_{image_id}', instrument, exposure_sec, seed, size_vu)
    scene['stars'] = _star_list(
        seed, count, size_vu, vmag_bright=_STAR_VMAG_BRIGHT, vmag_step=_STAR_VMAG_STEP
    )
    return scene


def limb_scene(
    image_id: str,
    instrument: str,
    exposure_sec: float,
    size_vu: tuple[int, int],
    *,
    diameter_px: float,
    phase_angle_deg: float,
) -> dict[str, Any]:
    """A matched limb frame: one ellipsoid at the real body's scale and phase.

    The frame renders at the real frame's size with the real body's
    apparent diameter, so a frame-filling real body is matched by a
    frame-filling simulated one -- clamping the body into a smaller frame
    would leave sky corners the real frame does not have, and the FOM 1/5
    floor statistics would compare different scene contents.

    Parameters:
        image_id: The matched real frame's image_id (drives the seed).
        instrument: Sim instrument name.
        exposure_sec: The real frame's exposure.
        size_vu: The real frame's (v, u) shape.
        diameter_px: The real body's predicted apparent diameter.
        phase_angle_deg: The real frame's phase angle.
    """
    seed = _seed_for(image_id)
    scene = _base_scene(f'realism_limb_{image_id}', instrument, exposure_sec, seed, size_vu)
    # The scene 'axis' values are apparent extents in pixels (a body with
    # axis1 = D renders a D-px-diameter silhouette).
    axis = max(40.0, float(diameter_px))
    center = min(size_vu) / 2.0
    scene['bodies'] = [
        {
            'name': 'REALISM_BODY',
            'center_v': center,
            'center_u': center,
            'axis1': axis,
            'axis2': axis,
            'axis3': axis,
            'illumination_angle': 0.0,
            'phase_angle': float(min(max(phase_angle_deg, 0.0), 179.0)),
            # Content matching, not tuning: the cohort's limb frames are
            # airless icy satellites, whose disks follow a Lommel-Seeliger
            # law (flat toward the limb) far better than Lambert.  Under
            # Lambert the FOM 3 rise width would measure the shading ramp,
            # not the limb.
            'photometric_law': 'lommel_seeliger',
        }
    ]
    scene['stars'] = _star_list(seed, 5, size_vu, vmag_bright=9.0, vmag_step=0.5)
    return scene


def ring_scene(
    image_id: str,
    instrument: str,
    exposure_sec: float,
    size_vu: tuple[int, int],
    *,
    curved: bool,
    with_body: bool,
    diameter_px: float = 150.0,
    phase_angle_deg: float = 45.0,
) -> dict[str, Any]:
    """A matched ring frame: three sharp-edged features around an annulus.

    Parameters:
        image_id: The matched real frame's image_id (drives the seed).
        instrument: Sim instrument name.
        exposure_sec: The real frame's exposure.
        curved: True for an open (curved-edge) geometry, False for a
            shallow opening whose edges cross the frame nearly straight.
        with_body: Add a central body (the ring_plus_body class).
        diameter_px: The real body's predicted apparent diameter for the
            ``with_body`` case.  The cohort's ring_plus_body frames carry
            small moons at every phase, and a fixed-size body would place
            the sim widths in a different FOM 3 stratum than the matched
            real body's.
        phase_angle_deg: The real frame's phase angle for the ``with_body``
            case (same stratification argument).
    """
    seed = _seed_for(image_id)
    scene = _base_scene(f'realism_ring_{image_id}', instrument, exposure_sec, seed, size_vu)
    opening = 65.0 if curved else 12.0
    # Ring radii and widths scale with the frame so edge coverage matches
    # the real frame's, keeping per-frame profile counts comparable.
    scale = min(size_vu) / _DEFAULT_SCENE_SIZE
    center_v = size_vu[0] / 2.0
    center_u = size_vu[1] / 2.0
    scene['ring_system'] = {
        'geometry': {
            'center_v': center_v,
            'center_u': center_u,
            'opening_deg_obs': opening,
            'opening_deg_sun': opening,
            'node_deg': 0.0,
        },
        'features': [
            {
                'name': 'REALISM_RING_A',
                'kind': 'ringlet',
                'tau': 2.0,
                'width_px': round(30.0 * scale, 1),
                'navigable': True,
                'orbit': {
                    'a_px': round(150.0 * scale, 1),
                    'ae_px': 0.0,
                    'long_peri_deg': 0.0,
                    'rate_peri_deg_per_day': 0.0,
                },
            },
            {
                'name': 'REALISM_RING_B',
                'kind': 'ringlet',
                'tau': 0.8,
                'width_px': round(18.0 * scale, 1),
                'navigable': True,
                'orbit': {
                    'a_px': round(205.0 * scale, 1),
                    'ae_px': 0.0,
                    'long_peri_deg': 0.0,
                    'rate_peri_deg_per_day': 0.0,
                },
            },
            {
                'name': 'REALISM_GAP',
                'kind': 'gap',
                'tau': 0.05,
                'width_px': round(12.0 * scale, 1),
                'navigable': True,
                'orbit': {
                    'a_px': round(160.0 * scale, 1),
                    'ae_px': 0.0,
                    'long_peri_deg': 0.0,
                    'rate_peri_deg_per_day': 0.0,
                },
            },
        ],
    }
    if with_body:
        axis = max(40.0, float(diameter_px))
        scene['bodies'] = [
            {
                'name': 'REALISM_PLANET',
                'center_v': center_v,
                'center_u': center_u,
                'axis1': axis,
                'axis2': axis,
                'axis3': axis,
                'illumination_angle': 0.0,
                'phase_angle': float(min(max(phase_angle_deg, 0.0), 179.0)),
                # Same content-matching choice as the limb scenes: airless
                # bodies shade Lommel-Seeliger, and a Lambert ramp would
                # masquerade as limb softness in the FOM 3 widths.
                'photometric_law': 'lommel_seeliger',
            }
        ]
    scene['stars'] = _star_list(seed, 5, size_vu, vmag_bright=9.5, vmag_step=0.5)
    return scene


def matched_scene(
    image_id: str,
    scene_class: str,
    instrument: str,
    exposure_sec: float,
    *,
    size_vu: tuple[int, int] = (_DEFAULT_SCENE_SIZE, _DEFAULT_SCENE_SIZE),
    diameter_px: float = 150.0,
    phase_angle_deg: float = 45.0,
) -> dict[str, Any]:
    """The matched sim scene for one real cohort frame.

    Parameters:
        image_id: The real frame's image_id.
        scene_class: The sidecar's primary scene tag.
        instrument: Sim instrument name matching the cohort's signal chain
            and units (e.g. ``coiss_calib_nac`` for the CALIB cohort).
        exposure_sec: The real frame's exposure (seconds).
        size_vu: The real frame's (v, u) shape; the matched frame renders
            at the same size so floor and coverage statistics compare the
            same scene geometry.
        diameter_px: Real body's apparent diameter for the limb classes
            and ring_plus_body (from the navigator's model metadata; the
            default covers frames where no body model could be built).
        phase_angle_deg: Real frame's phase angle for the limb classes
            and ring_plus_body.

    Returns:
        A validated scene mapping.
    """
    if scene_class in _STAR_CLASSES:
        count = {
            'one_bright_star_no_body': 1,
            'two_bright_stars_no_body': 2,
        }.get(scene_class, _STAR_COUNT)
        scene = star_field_scene(image_id, instrument, exposure_sec, size_vu, count=count)
    elif scene_class in _LIMB_CLASSES:
        scene = limb_scene(
            image_id,
            instrument,
            exposure_sec,
            size_vu,
            diameter_px=diameter_px,
            phase_angle_deg=phase_angle_deg,
        )
    elif scene_class in _RING_CLASSES:
        scene = ring_scene(
            image_id,
            instrument,
            exposure_sec,
            size_vu,
            curved=scene_class != 'ring_only_flat',
            with_body=scene_class == 'ring_plus_body',
            diameter_px=diameter_px,
            phase_angle_deg=phase_angle_deg,
        )
    else:
        scene = sky_scene(image_id, instrument, exposure_sec, size_vu)
    validate_sim_params(scene)
    return scene
