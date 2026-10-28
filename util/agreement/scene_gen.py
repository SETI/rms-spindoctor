"""Composition-controlled sim scenes for the agreement-estimator validation.

Generates seeded sim_params dicts (the same flat mapping the calibration
campaign feeds ObsSim) for cohorts whose *estimator composition* is the
controlled variable, together with the geometry angles the covariance solve
needs:

- limb_disc              : one resolved body, fully in frame; BodyLimbNav and
                           BodyDiscCorrelateNav both run (no ring).
- limb_disc_psf          : identical geometry to limb_disc but rendered through
                           an active whole-scene PSF, so the limb edge and the
                           disc silhouette carry a sub-pixel-resolved blurred
                           edge.  This is the scene family the base limb-disc
                           pair needs to probe a shared PSF edge bias (the
                           PSF-layer injection in collect.py); that bias cannot
                           appear in the PSF-free families.
- limb_disc_ring_fixed   : the same body plus a straight face-on ringlet whose
                           radial direction is FROZEN across the cohort (the
                           degenerate rank-1 regime the identifiability map
                           must demonstrate).
- limb_disc_ring_diverse : identical except the ring radial direction is drawn
                           uniformly per scene (the orientation-diverse regime
                           the full-matrix solve needs).
- limb_ring_aniso_fixed  : a large partially-clipped body (anisotropic limb
                           covariance, arc orientation frozen) plus the ring
                           at a frozen relative angle.
- limb_ring_aniso_diverse: the clipped-body arc orientation AND the ring
                           radial direction drawn per scene (the rotating-
                           anisotropy trap carried in full matrix form).
- multi_body             : two resolved bodies (RHEA and DIONE), each large
                           enough for limb + disc; the collector navigates
                           each body separately via the model filter.

Scenes are deliberately clean -- smooth ellipsoids, moderate noise, no
planted model error -- because this campaign validates the estimator's
mathematics against known-by-construction errors, not the techniques'
robustness (that is the calibration campaign's job).  All randomness comes
from a caller-seeded ``random.Random`` keyed by (campaign_seed, family,
index), so any scene regenerates without replaying the campaign.

Angle convention: an angle ``alpha`` (degrees here, radians in the
estimator) denotes the image-plane unit direction ``(cos alpha, sin alpha)``
in (v, u) coordinates.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from typing import Any

FAMILIES = (
    'limb_disc',
    'limb_disc_psf',
    'limb_disc_ring_fixed',
    'limb_disc_ring_diverse',
    'limb_ring_aniso_fixed',
    'limb_ring_aniso_diverse',
    'multi_body',
)

FRAME_SIZE = 256
_FRAME_CENTER = FRAME_SIZE / 2.0

# Frozen directions for the *_fixed cohorts (degrees).  Arbitrary but fixed:
# the point is that they do not vary across the cohort.
_FIXED_RING_RADIAL_DEG = 200.0
_FIXED_ARC_THETA_DEG = 40.0
# Relative angle between the clipped-limb outward-arc direction and the ring
# radial direction in the aniso_fixed cohort (degrees).
_FIXED_ANISO_RELATIVE_DEG = -20.0

# Ring semimajor-axis range (px): large enough that the edge crossing the
# frame is nearly straight (sagitta under ~6 px across the full frame), the
# rank-1 regime the estimator declares for the ring instance.
_RING_A_RANGE = (1200.0, 2500.0)
# Distance range of the ring-edge line from the frame center (px).  The
# lower bound clears the fully-framed body (radius <= 65 px, center within
# 10 px of frame center) with margin; the upper bound keeps the ringlet
# well inside the frame.
_RING_H_RANGE = (92.0, 105.0)

# Bodies drawn for the single-body cohorts (regular, near-spherical).
_BODY_NAMES = ('RHEA', 'DIONE', 'TETHYS')


def _base(rng: random.Random) -> dict[str, Any]:
    """Common sim_params skeleton: planted offset, moderate noise.

    Parameters:
        rng: Scene-local random generator.

    Returns:
        The base scene parameter dict (no bodies or rings yet).
    """
    return {
        'instrument': 'coiss_nac',
        'size_v': FRAME_SIZE,
        'size_u': FRAME_SIZE,
        'random_seed': rng.randrange(2**31),
        'exposure_sec': 1.0,
        'bodies': [],
        'noise': {
            'poisson': True,
            'read_noise_dn': math.exp(rng.uniform(math.log(2.0), math.log(8.0))),
        },
        'offset_v': rng.uniform(-3.0, 3.0),
        'offset_u': rng.uniform(-3.0, 3.0),
        'offset_rotation_deg': 0.0,
    }


def _framed_body(rng: random.Random) -> tuple[dict[str, Any], float, float]:
    """One smooth ellipsoid body fully inside the frame, limb-eligible.

    The diameter range (104-130 px) clears the simulated body model's
    100 px limb-emission floor with rasterization margin while leaving the
    whole silhouette inside the frame, so BodyLimbNav (closed,
    near-isotropic limb) and BodyDiscCorrelateNav both run.

    Parameters:
        rng: Scene-local random generator.

    Returns:
        ``(body_entry, center_v, center_u)``.
    """
    diameter = rng.uniform(104.0, 130.0)
    center_v = _FRAME_CENTER + rng.uniform(-10.0, 10.0)
    center_u = _FRAME_CENTER + rng.uniform(-10.0, 10.0)
    body = {
        'name': rng.choice(_BODY_NAMES),
        'center_v': center_v,
        'center_u': center_u,
        'axis1': diameter,
        'axis2': diameter * rng.uniform(0.97, 1.0),
        'axis3': diameter * rng.uniform(0.96, 1.0),
        'illumination_angle': rng.uniform(0.0, 360.0),
        'phase_angle': rng.uniform(10.0, 45.0),
    }
    return body, center_v, center_u


def _ring_system(radial_deg: float, h: float, a: float, rng: random.Random) -> dict[str, Any]:
    """A face-on straight-edge ringlet crossing the frame.

    The edge line passes at distance ``h`` from the frame center along the
    radial direction ``radial_deg``; the ring center sits ``a`` pixels
    behind it, so over the 256 px frame the edges are nearly straight
    (rank-1).  Face-on (opening 90 deg) keeps the drawn radial direction an
    exact image-plane direction.

    Parameters:
        radial_deg: Radial (increasing-radius) direction at the crossing.
        h: Signed distance of the ringlet mid-line from the frame center.
        a: Ring semimajor axis (px).
        rng: Scene-local random generator (optical depth, width).

    Returns:
        The ``ring_system`` scene block.
    """
    phi = math.radians(radial_deg)
    mid_v = _FRAME_CENTER + h * math.cos(phi)
    mid_u = _FRAME_CENTER + h * math.sin(phi)
    return {
        'geometry': {
            'center_v': mid_v - a * math.cos(phi),
            'center_u': mid_u - a * math.sin(phi),
            'opening_deg_obs': 90.0,
            'opening_deg_sun': 90.0,
            'node_deg': 0.0,
        },
        'features': [
            {
                'name': 'SATURN-1',
                'kind': 'ringlet',
                'tau': math.exp(rng.uniform(math.log(0.8), math.log(2.5))),
                'width_px': rng.uniform(8.0, 16.0),
                'navigable': True,
                'orbit': {
                    'a_px': a,
                    'ae_px': 0.0,
                    'long_peri_deg': 0.0,
                    'rate_peri_deg_per_day': 0.0,
                },
            }
        ],
    }


def _gen_limb_disc(rng: random.Random) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fully-framed body: limb + disc, no ring."""
    params = _base(rng)
    body, _, _ = _framed_body(rng)
    params['bodies'] = [body]
    geometry = {'composition': 'limb_disc'}
    return params, geometry


def _gen_limb_disc_psf(rng: random.Random) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fully-framed partially-lit body rendered through an active whole-scene PSF.

    Identical geometry to :func:`_gen_limb_disc` (limb + disc, no ring), but the
    scene carries ``optics.psf: {match_navigator: true}`` so the control render
    equals the navigator's own Gaussian (the self-consistency floor: no
    render-vs-navigate PSF mismatch).  The PSF-layer injection (see
    ``collect.py``) replaces that block with a broadened or anisotropic kernel
    the navigator does not model; a shared PSF edge bias between the limb
    (distance-transform) and disc (normalized cross-correlation) estimators can
    appear only against a rendered blurred edge, which the PSF-free families
    lack.  The body is partially lit (phase 10-45 deg), so the illumination
    direction breaks the silhouette's symmetry and a symmetric broadening can
    still bias the edge; the illumination direction is recorded so the analysis
    can test whether any induced coupling is locked to it.

    Parameters:
        rng: Scene-local random generator.
    """
    params = _base(rng)
    body, _, _ = _framed_body(rng)
    params['bodies'] = [body]
    params['optics'] = {'psf': {'match_navigator': True}}
    geometry = {
        'composition': 'limb_disc_psf',
        'illumination_deg': float(body['illumination_angle']),
        'phase_deg': float(body['phase_angle']),
    }
    return params, geometry


def _gen_limb_disc_ring(
    rng: random.Random, *, diverse: bool
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fully-framed body plus a straight ringlet clear of the body.

    Parameters:
        rng: Scene-local random generator.
        diverse: Draw the ring radial direction uniformly per scene when
            True; freeze it at the cohort constant when False.
    """
    params = _base(rng)
    body, _, _ = _framed_body(rng)
    params['bodies'] = [body]
    radial_deg = rng.uniform(0.0, 360.0) if diverse else _FIXED_RING_RADIAL_DEG
    h = rng.uniform(*_RING_H_RANGE)
    a = rng.uniform(*_RING_A_RANGE)
    params['ring_system'] = _ring_system(radial_deg, h, a, rng)
    geometry = {
        'composition': 'limb_disc_ring',
        'ring_radial_deg': radial_deg,
    }
    return params, geometry


def _line_clears_body(
    radial_deg: float, h: float, body_center_vu: tuple[float, float], clearance: float
) -> bool:
    """True when the ring mid-line stays ``clearance`` px from the body center.

    The (nearly straight) ringlet mid-line is the line perpendicular to the
    radial direction at signed distance ``h`` from the frame center; the
    signed distance of any point from it is the point's radial coordinate
    minus ``h``.

    Parameters:
        radial_deg: Ring radial direction (degrees).
        h: Signed distance of the mid-line from the frame center.
        body_center_vu: Body center in image coordinates.
        clearance: Required distance (px).

    Returns:
        Whether the line clears the body.
    """
    phi = math.radians(radial_deg)
    rel_v = body_center_vu[0] - _FRAME_CENTER
    rel_u = body_center_vu[1] - _FRAME_CENTER
    body_radial = rel_v * math.cos(phi) + rel_u * math.sin(phi)
    return abs(body_radial - h) >= clearance


def _gen_limb_ring_aniso(
    rng: random.Random, *, diverse: bool
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Clipped large body (anisotropic limb arc) plus the straight ringlet.

    The body center sits ~0.55-0.65 radii from the frame center along the
    arc direction ``theta``, so roughly half the limb is visible: the limb
    covariance is genuinely anisotropic, tight along the outward-arc
    (radial) direction ``theta + 180`` and loose along the arc tangent.
    The ring's radial direction is drawn around the clear side, offset from
    the arc direction by a relative angle that is frozen for the fixed
    cohort and drawn uniformly (within the clearance envelope) for the
    diverse cohort.  Low phase keeps the lit-region boundary close to the
    geometric limb so the DT fit stays on the arc.

    Parameters:
        rng: Scene-local random generator.
        diverse: Draw arc/ring directions per scene when True.
    """
    params = _base(rng)
    radius = rng.uniform(100.0, 120.0)
    theta_deg = rng.uniform(0.0, 360.0) if diverse else _FIXED_ARC_THETA_DEG
    theta = math.radians(theta_deg)
    dist = rng.uniform(0.55, 0.65) * radius
    center_v = _FRAME_CENTER + dist * math.cos(theta)
    center_u = _FRAME_CENTER + dist * math.sin(theta)
    diameter = 2.0 * radius
    body = {
        'name': rng.choice(_BODY_NAMES),
        'center_v': center_v,
        'center_u': center_u,
        'axis1': diameter,
        'axis2': diameter * rng.uniform(0.985, 1.0),
        'axis3': diameter * rng.uniform(0.98, 1.0),
        'illumination_angle': rng.uniform(0.0, 360.0),
        'phase_angle': rng.uniform(8.0, 14.0),
    }
    params['bodies'] = [body]
    # Ring on the clear side: radial direction opposite the arc direction
    # plus a relative offset, with a numerical clearance check against the
    # body (rejection-sample the offset and line distance).
    h = 0.0
    radial_deg = 0.0
    clearance = radius + 14.0
    for attempt in range(64):
        if diverse:
            delta = rng.uniform(-60.0, 60.0)
        else:
            delta = _FIXED_ANISO_RELATIVE_DEG
        radial_deg = (theta_deg + 180.0 + delta) % 360.0
        h = rng.uniform(60.0, 100.0)
        if _line_clears_body(radial_deg, h, (center_v, center_u), clearance):
            break
        if not diverse and attempt >= 8:
            # The frozen relative angle must not be resampled; widen h only.
            h = rng.uniform(80.0, 105.0)
    if not _line_clears_body(radial_deg, h, (center_v, center_u), clearance):
        # Deterministic fallback (no further RNG draws, so the normal
        # path's stream -- and therefore every recorded campaign scene --
        # is unchanged): the line exactly opposite the arc at a large h is
        # always clear, because its distance from the body center is
        # h + dist * cos(delta) with dist <= 0.65 * radius, which exceeds
        # radius + 14 for h = 100 at every drawn geometry.  The fixed
        # family keeps its frozen relative angle.
        delta = 0.0 if diverse else _FIXED_ANISO_RELATIVE_DEG
        radial_deg = (theta_deg + 180.0 + delta) % 360.0
        h = 100.0
        if not _line_clears_body(radial_deg, h, (center_v, center_u), clearance):
            raise RuntimeError(
                f'aniso ring placement failed clearance even at fallback: '
                f'radius={radius:.1f} dist={dist:.1f} delta={delta:.1f}'
            )
    a = rng.uniform(*_RING_A_RANGE)
    params['ring_system'] = _ring_system(radial_deg, h, a, rng)
    geometry = {
        'composition': 'limb_ring_aniso',
        'ring_radial_deg': radial_deg,
        'limb_arc_outward_deg': (theta_deg + 180.0) % 360.0,
    }
    return params, geometry


def _gen_multi_body(rng: random.Random) -> tuple[dict[str, Any], dict[str, Any]]:
    """Two resolved bodies, each limb + disc eligible, well separated."""
    params = _base(rng)
    phase = rng.uniform(10.0, 40.0)
    illumination = rng.uniform(0.0, 360.0)
    bodies = []
    for name, base_v, base_u, dia_lo, dia_hi in (
        ('RHEA', 68.0, 68.0, 100.0, 115.0),
        ('DIONE', 188.0, 188.0, 100.0, 112.0),
    ):
        diameter = rng.uniform(dia_lo, dia_hi)
        bodies.append(
            {
                'name': name,
                'center_v': base_v + rng.uniform(-6.0, 6.0),
                'center_u': base_u + rng.uniform(-6.0, 6.0),
                'axis1': diameter,
                'axis2': diameter * rng.uniform(0.97, 1.0),
                'axis3': diameter * rng.uniform(0.96, 1.0),
                'illumination_angle': illumination,
                'phase_angle': phase,
            }
        )
    params['bodies'] = bodies
    geometry = {'composition': 'multi_body', 'body_names': ['RHEA', 'DIONE']}
    return params, geometry


_GENERATORS: dict[str, Callable[[random.Random], tuple[dict[str, Any], dict[str, Any]]]] = {
    'limb_disc': _gen_limb_disc,
    'limb_disc_psf': _gen_limb_disc_psf,
    'limb_disc_ring_fixed': lambda rng: _gen_limb_disc_ring(rng, diverse=False),
    'limb_disc_ring_diverse': lambda rng: _gen_limb_disc_ring(rng, diverse=True),
    'limb_ring_aniso_fixed': lambda rng: _gen_limb_ring_aniso(rng, diverse=False),
    'limb_ring_aniso_diverse': lambda rng: _gen_limb_ring_aniso(rng, diverse=True),
    'multi_body': _gen_multi_body,
}


def generate_scenes(
    family: str, count: int, *, campaign_seed: int
) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    """Return ``count`` seeded scenes as ``(scene_id, sim_params, geometry)``.

    Each scene draws from its own ``random.Random`` seeded by
    ``(campaign_seed, family, index)`` so any single scene regenerates
    without replaying the campaign.  ``geometry`` carries the angle
    metadata the estimator consumes (ring radial direction, limb arc
    orientation) -- derived from the same idealized scene parameters the
    navigator is told, never from truth keys.

    Parameters:
        family: One of :data:`FAMILIES`.
        count: Number of scenes.
        campaign_seed: Campaign-level seed recorded in the output manifest.

    Returns:
        List of ``(scene_id, sim_params, geometry)`` triples.

    Raises:
        ValueError: for an unknown family name.
    """
    from spindoctor.sim.scene import validate_sim_params

    if family not in _GENERATORS:
        raise ValueError(f'unknown scene family {family!r}; valid: {sorted(_GENERATORS)}')
    generator = _GENERATORS[family]
    scenes = []
    for index in range(count):
        rng = random.Random(f'{campaign_seed}/{family}/{index}')
        scene_id = f'{family}_{index:05d}'
        params, geometry = generator(rng)
        scenes.append((scene_id, validate_sim_params(params, source=scene_id), geometry))
    return scenes
