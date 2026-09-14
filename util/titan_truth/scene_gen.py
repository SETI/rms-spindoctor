"""Randomized planted-truth scene generation for the haze-symmetry navigator.

Generates seeded, randomized sim_params dicts (the same flat mapping the
sim-sweep harness feeds ObsSim) for a hazy Titan across the axes the method's
accuracy claim has to survive.  Every scene plants a known ``(offset_v,
offset_u)`` and the navigator recovers it from the image alone, so each scene
is one measurement of the recovery error.

The families are organized so that one of them -- ``clean`` -- holds the
estimator's own assumptions true and every other family breaks exactly one
class of them:

- clean       : the assumptions hold.  Geometry (apparent size, phase, sun
                direction), pointing offset, and noise vary; the haze is the
                plain mirror-symmetric column and the frame carries nothing
                else.  This is the family the accuracy bound is stated on.
- clouds      : Gaussian clouds on the disc, the classic reason a real haze
                is not mirror-symmetric.
- asymmetry   : the rendered symmetry axis tilted off the geometric sun
                direction, a NON-affine hemispheric falloff difference, a
                hemispheric brightness scaling (the affine control the
                Pearson score should absorb), an interior brightness ramp,
                and a limb sharpness gradient across the sunward sector.
- stars       : a star field over the frame.  Stars fainter than the mask
                limit are deliberately NOT masked, so this family is what
                decides whether that policy is safe.
- artifacts   : cosmic rays, hot pixels, and the instrument's own defect
                population -- unmaskable point contamination with no
                predicted position to ride the offset hypothesis with.
                Deliberately a STRESS condition; see ``_artifact_blocks``.
- artifacts_nominal
              : the same contamination class at the shipped, realism-matched
                incidence for the emulated instrument, nothing overridden.
                This is the family an operational prediction reads from;
                ``artifacts`` bounds the regime rather than describing it.
- combined    : every axis at once, at the same draw strengths.

All randomness comes from a caller-seeded ``random.Random`` so a campaign is
reproducible from its seed.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterator
from typing import Any

FAMILIES: tuple[str, ...] = (
    'clean',
    'clouds',
    'asymmetry',
    'stars',
    'artifacts',
    'combined',
    # Appended rather than inserted: the per-scene seed derives from a
    # family's index, so a new family in the middle would silently redraw
    # every family after it and invalidate comparisons against earlier runs.
    'artifacts_nominal',
)

# Frame geometry.  The body sits at frame center in every scene: the search
# window is +-50 px (the coiss_nac extfov margin) and the largest envelope
# drawn below reaches 99 px, so a centered body keeps the whole envelope inside
# the extended frame at every planted offset and the visibility hard-zero
# never fires for a reason unrelated to the estimator.  Framing is not one of
# the axes this campaign measures.
FRAME_PX: int = 360
CENTER_PX: float = FRAME_PX / 2.0

# Titan's published mean radius; every scene renders the real body and varies
# only how far away it is, so the apparent size axis is a real range axis
# rather than an invented one.
TITAN_RADIUS_KM: float = 2575.0
RANGE_KM: float = 1.2e6

# Apparent solid radius range, in pixels.  The configured 700 km atmosphere
# height adds ~27% to it, so the smallest body here has a 71 px envelope
# diameter -- clear of the 40 px hard-zero floor, and clear enough of the
# 52 px reliability midpoint that the type gate is not what this campaign
# ends up measuring.
MIN_SOLID_RADIUS_PX: float = 28.0
MAX_SOLID_RADIUS_PX: float = 78.0

# Planted pointing error, in pixels: uniform in direction, uniform in
# magnitude out to 80% of the search window, so the campaign exercises the
# recenter pass and the wide end of the scan without testing the window bound
# itself (which is the at-edge flag's job, not the accuracy bound's).
MAX_PLANTED_OFFSET_PX: float = 40.0

# Offset from the in-plane sun direction to the body roll that puts the
# hemispheric split ACROSS the sun axis, degrees.
#
# The structure keys split hemispheres on ``v_rot``, the body-frame first-axis
# coordinate, whose zero line runs along ``(sin rotation_z, cos rotation_z)``
# in image coordinates.  The sun direction is ``(-cos a, sin a)`` for
# illumination angle ``a``, so setting ``rotation_z = a - 90`` puts the split
# line along the sun axis -- which is what makes ``ns_asymmetry_amplitude``
# the AFFINE control the design intends: the mirror about that axis maps one
# hemisphere onto the other, so a pure brightness scaling of one of them is
# exactly the relation the Pearson score is invariant to.  Leaving the roll
# at zero instead would let the split fall along the sun axis at most
# illumination angles, and the control would be measuring an unrelated
# gradient.
#
# Carrying the roll rather than pinning the sun to two directions is what
# keeps the campaign's angular coverage complete: the sun is drawn uniformly
# over the circle, and the roll follows it.  On the structureless spheres
# every family renders, an in-plane roll changes no rendered pixel.
BODY_ROLL_FROM_SUN_DEG: float = -90.0

# Phase range, degrees.  Below ~10 deg the disc is very nearly rotationally
# symmetric and the axis-degeneracy branch takes over; above ~140 deg the
# sunward limb has shrunk to a crescent and the arc fit loses its support.
# Both ends are real regimes, deferred to their own follow-ups; this is the
# working range the accuracy bound is claimed over.
MIN_PHASE_DEG: float = 10.0
MAX_PHASE_DEG: float = 140.0

# Read-noise range in DN, drawn log-uniformly: a clean frame through to a
# noisy one, spanning roughly the SNR range the technique characterization
# sweeps.
MIN_READ_NOISE_DN: float = 1.5
MAX_READ_NOISE_DN: float = 16.0


def _haze_block(rng: random.Random, r_solid_px: float) -> dict[str, Any]:
    """Draw the plain (mirror-symmetric) haze column for one scene.

    The scale height is drawn as a fraction of the apparent radius so the
    rendered optical limb sits a few pixels above the solid surface at every
    apparent size -- the altitude mismatch the free-radius arc fit absorbs --
    rather than vanishing on a large body and swamping a small one.

    Parameters:
        rng: The scene's random source.
        r_solid_px: The apparent solid radius in pixels.

    Returns:
        The ``atmosphere`` mapping.
    """
    return {
        'scale_height_px': round(r_solid_px * rng.uniform(0.08, 0.18), 4),
        'tau_ref': round(rng.uniform(2.0, 5.0), 4),
        'ref_altitude_px': round(r_solid_px * rng.uniform(0.02, 0.10), 4),
        'g': round(rng.uniform(0.2, 0.7), 4),
    }


def _cloud_blobs(rng: random.Random, r_solid_px: float) -> list[dict[str, Any]]:
    """Draw one to four Gaussian clouds inside the disc.

    Positions are drawn uniformly over the disc area (the square-root radial
    draw) and kept clear of the limb so a cloud is a disc feature rather than
    a limb perturbation, which is the sharpness-gradient axis's job.

    Parameters:
        rng: The scene's random source.
        r_solid_px: The apparent solid radius in pixels.

    Returns:
        The ``cloud_blobs`` list.
    """
    blobs: list[dict[str, Any]] = []
    for _ in range(rng.randint(1, 4)):
        radius = 0.75 * r_solid_px * math.sqrt(rng.random())
        angle = rng.uniform(0.0, 2.0 * math.pi)
        blobs.append(
            {
                'center_vu': [
                    round(radius * math.sin(angle), 4),
                    round(radius * math.cos(angle), 4),
                ],
                'sigma_px': round(r_solid_px * rng.uniform(0.05, 0.20), 4),
                'amplitude': round(rng.choice((-1.0, 1.0)) * rng.uniform(0.10, 0.35), 4),
            }
        )
    return blobs


def _asymmetry_keys(rng: random.Random) -> dict[str, Any]:
    """Draw the symmetry-breaking structure keys of the haze.

    The tilt range brackets the few degrees Titan's atmospheric symmetry axis
    is known to sit off the spin axis; the falloff ratio and the sharpness
    gradient are the two shape (non-affine) axes, and the hemispheric
    amplitude is the affine control the Pearson score is supposed to absorb.

    Parameters:
        rng: The scene's random source.

    Returns:
        The structure keys to merge into the ``atmosphere`` mapping.
    """
    return {
        'axis_tilt_deg': round(rng.uniform(-12.0, 12.0), 4),
        'ns_falloff_ratio': round(rng.uniform(0.6, 1.8), 4),
        'ns_asymmetry_amplitude': round(rng.uniform(-0.35, 0.35), 4),
        'sector_sharpness_gradient': round(rng.uniform(-0.3, 1.0), 4),
        'interior_ramp_amplitude': round(rng.uniform(-0.25, 0.25), 4),
    }


def _star_field(rng: random.Random, r_solid_px: float) -> list[dict[str, Any]]:
    """Draw a star field over the frame, clear of the disc interior.

    Stars are placed outside the solid disc (a star behind the body is not a
    contaminant, it is occulted) but inside the frame, so they land in the
    symmetry annulus and the arc rays where they can actually do harm.  The
    magnitude range straddles the mask limit deliberately: the bright ones
    are masked from predicted positions, the faint ones are not, and the
    campaign is what shows the unmasked ones are harmless.

    Parameters:
        rng: The scene's random source.
        r_solid_px: The apparent solid radius in pixels.

    Returns:
        The ``stars`` list.
    """
    stars: list[dict[str, Any]] = []
    for index in range(rng.randint(3, 12)):
        while True:
            v = rng.uniform(4.0, FRAME_PX - 4.0)
            u = rng.uniform(4.0, FRAME_PX - 4.0)
            if math.hypot(v - CENTER_PX, u - CENTER_PX) > r_solid_px:
                break
        stars.append(
            {
                'name': f'S{index}',
                'v': round(v, 4),
                'u': round(u, 4),
                'vmag': round(rng.uniform(4.0, 9.5), 3),
            }
        )
    return stars


def _artifact_blocks(rng: random.Random) -> tuple[dict[str, Any], dict[str, Any]]:
    """Draw the STRESS cosmic-ray / hot-pixel contamination of one scene.

    Provenance of the draw ranges, stated because they are deliberately NOT
    the instrument's realism-matched values and a reader would otherwise take
    them for an operational prediction:

    - Hot-pixel incidence is drawn over 2e-4 to 2e-3.  The emulated
      instrument's own catalog carries 2e-3 (its 5.2 interim figure) while
      the realism match measures a transient single-pixel spike fraction of
      2.75e-4 per frame on the CALIB NAC cohort, so the range spans from
      about the measured transient population up to the interim catalog
      value -- an order of magnitude of stress, not a distribution.
    - The cosmic-ray rate is drawn strictly positive although the realism
      recalibration RETAINED zero for it: the instrument's tuned hot-pixel
      fraction already carries the measured spike population, so a nonzero
      rate here double-counts it on purpose.

    Both choices make this family an upper bound on point-source
    contamination.  The ``artifacts_nominal`` family is the matching
    realism-calibrated condition, and it is the one to quote.

    Parameters:
        rng: The scene's random source.

    Returns:
        ``(artifacts_block, noise_extras)``: the instrument defect population
        plus an explicit hot-pixel incidence, and the cosmic-ray rate the
        noise block carries.
    """
    artifacts = {
        'instrument_defaults': True,
        'hot_pixels': {
            'incidence': round(rng.uniform(2.0e-4, 2.0e-3), 6),
            'amplitude_e': round(rng.uniform(2.0e4, 6.0e4), 1),
        },
    }
    noise_extras = {
        'cosmic_ray_rate_per_sec': round(rng.uniform(5.0e-4, 5.0e-3), 6),
        'bloom_length': rng.randint(0, 3),
    }
    return artifacts, noise_extras


def _nominal_artifact_block() -> dict[str, Any]:
    """The instrument's own realism-matched defect population, unoverridden.

    ``instrument_defaults`` alone switches on the shipped, calibrated
    artifact catalog for the emulated instrument -- the hot-pixel fraction
    and amplitude the realism match tuned, and the cosmic-ray rate it
    deliberately left at zero.  Nothing is drawn, so this condition is the
    same on every scene of the family and the family's spread comes purely
    from the geometry and noise the base scene already varies.

    Returns:
        The ``artifacts`` mapping.
    """
    return {'instrument_defaults': True}


def _base_scene(rng: random.Random, seed: int) -> tuple[dict[str, Any], float]:
    """Build the geometry, pointing, and noise common to every family.

    Parameters:
        rng: The scene's random source.
        seed: The scene's render seed.

    Returns:
        ``(sim_params, r_solid_px)``.
    """
    r_solid_px = rng.uniform(MIN_SOLID_RADIUS_PX, MAX_SOLID_RADIUS_PX)
    offset_mag = rng.uniform(0.0, MAX_PLANTED_OFFSET_PX)
    offset_dir = rng.uniform(0.0, 2.0 * math.pi)
    illumination_deg = round(rng.uniform(0.0, 360.0), 4)
    log_lo, log_hi = math.log(MIN_READ_NOISE_DN), math.log(MAX_READ_NOISE_DN)
    sim_params: dict[str, Any] = {
        'instrument': 'coiss_nac',
        'size_v': FRAME_PX,
        'size_u': FRAME_PX,
        'random_seed': seed,
        'exposure_sec': 1.0,
        'offset_v': round(offset_mag * math.sin(offset_dir), 5),
        'offset_u': round(offset_mag * math.cos(offset_dir), 5),
        'offset_rotation_deg': 0.0,
        'bodies': [
            {
                'name': 'TITAN',
                'shape_model': 'ellipsoid',
                'center_v': CENTER_PX,
                'center_u': CENTER_PX,
                'axis1': round(2.0 * r_solid_px, 4),
                'axis2': round(2.0 * r_solid_px, 4),
                'axis3': round(2.0 * r_solid_px, 4),
                'illumination_angle': illumination_deg,
                'rotation_z': round(illumination_deg + BODY_ROLL_FROM_SUN_DEG, 4),
                'phase_angle': round(rng.uniform(MIN_PHASE_DEG, MAX_PHASE_DEG), 4),
                'km_per_pixel': round(TITAN_RADIUS_KM / r_solid_px, 6),
                'range_km': RANGE_KM,
                'atmosphere': _haze_block(rng, r_solid_px),
            }
        ],
        'noise': {
            'poisson': True,
            'read_noise_dn': round(math.exp(rng.uniform(log_lo, log_hi)), 4),
        },
    }
    return sim_params, r_solid_px


def generate_scenes(
    family: str, count: int, *, campaign_seed: int
) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield ``(scene_id, sim_params)`` for one family of a campaign.

    Parameters:
        family: One of :data:`FAMILIES`.
        count: How many scenes to generate.
        campaign_seed: Campaign seed; the per-scene seed is derived from it,
            the family name, and the scene index, so a family can be
            regenerated on its own and reproduce byte-identical scenes.

    Yields:
        ``(scene_id, sim_params)`` pairs.

    Raises:
        ValueError: If ``family`` is not a known family.
    """
    if family not in FAMILIES:
        raise ValueError(f'unknown family {family!r}; expected one of {list(FAMILIES)}')
    for index in range(count):
        seed = (campaign_seed + 1_000_003 * (FAMILIES.index(family) + 1) + index) % (2**31)
        rng = random.Random(seed)
        sim_params, r_solid_px = _base_scene(rng, seed)
        atmosphere = sim_params['bodies'][0]['atmosphere']
        if family in ('clouds', 'combined'):
            atmosphere['cloud_blobs'] = _cloud_blobs(rng, r_solid_px)
        if family in ('asymmetry', 'combined'):
            atmosphere.update(_asymmetry_keys(rng))
        if family in ('stars', 'combined'):
            sim_params['stars'] = _star_field(rng, r_solid_px)
        if family in ('artifacts', 'combined'):
            artifacts, noise_extras = _artifact_blocks(rng)
            sim_params['artifacts'] = artifacts
            sim_params['noise'].update(noise_extras)
        if family == 'artifacts_nominal':
            sim_params['artifacts'] = _nominal_artifact_block()
        yield f'{family}_{index:04d}', sim_params
