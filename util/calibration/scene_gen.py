"""Randomized sim-scene generation for the confidence calibration.

Generates seeded, randomized sim_params dicts (the same flat mapping the
sim-sweep harness feeds ObsSim) spanning each NavTechnique's operating
regime, from clean well-resolved frames through the failure cliff:

- disc        : resolved regular/irregular body, low-moderate phase
                (BodyDiscCorrelateNav; mesh-vs-ellipsoid shape mismatch)
- limb        : large body partially off-frame, low phase (BodyLimbNav;
                pose-disagreement and shape-mismatch slices)
- terminator  : high-phase crescent (BodyTerminatorNav)
- blob        : small / distant body (BodyBlobNav; irregular slice)
- ring        : 1-2 ringlets, curved through near-flat arcs, face-on
                through moderately inclined (RingEdgeNav and, via the
                always-emitted template, RingAnnulusNav); slices draw the
                ring truth vocabulary: eccentric orbits with occasional
                m-modes, satellite edge waves, planted per-feature orbit
                errors, and non-navigable distractor features
- star_field  : 3-15 stars, bright through sub-detection (the star
                techniques incl. the pass-2 StarRefineNav); a controlled
                fraction plant catalog-position scatter and non-navigable
                confounders (the astrometric-error and star-clutter regimes)
- star_unique : 1-2 star scenes with varying brightness margin
                (StarUniqueMatchNav one- and two-star paths); a controlled
                fraction plant a per-star catalog error or an unresolved
                companion (astrometric and photocenter-bias regimes)
- titan       : a hazy Titan across apparent size, phase, and sun direction
                (TitanHazeNav); controlled fractions break the mirror
                symmetry the method assumes -- clouds, a tilted haze axis,
                a non-affine hemispheric falloff, an interior ramp, a limb
                sharpness gradient -- and add point-source contamination

Every scene carries a planted (offset_v, offset_u) ground truth. The noise
level, feature sizes/counts, and the model-error axes (mesh lumpiness vs an
ellipsoidal nav prediction, predicted-pose error) are drawn from ranges wide
enough that each technique's diagnostics span healthy to broken -- the spread
a logistic calibration fit needs on both label classes.

All randomness comes from a caller-seeded ``random.Random`` so a campaign is
reproducible from its seed.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from typing import Any

# The body families draw the full set of body truth axes the renderer
# carries (photometric law / opposition surge, the limb-relief field,
# albedo texture, mesh shading / pose scatter; the disc family adds the
# giant-planet disc_texture / transits slice), so every sim-anchored
# coefficient is fit on the renderer it ships with.  Each mismatch axis
# is drawn on a minority of scenes so the clean label class stays
# dominant; see _surface_truth_axes.
FAMILIES = (
    'disc',
    'limb',
    'terminator',
    'blob',
    'ring',
    'star_field',
    'star_unique',
    'titan',
)

# Body identities the campaign draws from, with mean radius and published
# ellipsoid RMS residual pulled live from config_220_body_shape.yaml (so
# the campaign always matches the shipped table).  Every rendered body is
# a polyhedral mesh with mesh_lumpiness 3x the body's fractional RMS
# residual; the mesh's non-ellipsoidal RMS is lumpiness / 8, so that plants
# 3/8 of the published residual (8x would match it), while the navigator
# predicts the smooth (zero-relief) limit -- so the rendered shape error,
# and hence the recovered-offset error, physically tracks the
# max_phase_irregularity_factor the confidence formula consumes.  The
# scene also carries km_per_pixel = mean_radius_km / radius_px, without
# which the sim blob feature reports a zero irregularity factor.
_REGULAR_BODIES = ('MIMAS', 'ENCELADUS', 'TETHYS', 'DIONE', 'RHEA')
_IRREGULAR_BODIES = ('HYPERION', 'PHOEBE', 'JANUS', 'EPIMETHEUS')


def _body_catalog() -> dict[str, tuple[float, float]]:
    """Return ``{name: (mean_radius_km, rms_residual_km)}`` from config_220."""
    from spindoctor.config import DEFAULT_CONFIG

    catalog: dict[str, tuple[float, float]] = {}
    for name in _REGULAR_BODIES + _IRREGULAR_BODIES:
        entry = DEFAULT_CONFIG.body_shape[name]
        radii = entry['radii_km']
        catalog[name] = (
            sum(float(r) for r in radii) / len(radii),
            float(entry['ellipsoid_rms_residual_km']),
        )
    return catalog


_CATALOG: dict[str, tuple[float, float]] | None = None


def _catalog() -> dict[str, tuple[float, float]]:
    """Lazily built body catalog (config load deferred past import)."""
    global _CATALOG
    if _CATALOG is None:
        _CATALOG = _body_catalog()
    return _CATALOG


def _read_noise(rng: random.Random, *, lo: float = 1.0, hi: float = 48.0) -> float:
    """Log-uniform read-noise draw (DN) between ``lo`` and ``hi``."""
    return math.exp(rng.uniform(math.log(lo), math.log(hi)))


def _planted_offset(rng: random.Random) -> tuple[float, float]:
    """Planted (dv, du): mostly a few px, a tail out to +/-10 px."""
    span = 4.0 if rng.random() < 0.7 else 10.0
    return rng.uniform(-span, span), rng.uniform(-span, span)


def _base(rng: random.Random, *, size: int) -> dict[str, Any]:
    """Common sim_params skeleton every family builds on."""
    dv, du = _planted_offset(rng)
    return {
        'instrument': 'coiss_nac',
        'size_v': size,
        'size_u': size,
        'random_seed': rng.randrange(2**31),
        'exposure_sec': 1.0,
        'bodies': [],
        'noise': {'poisson': True, 'read_noise_dn': _read_noise(rng)},
        'offset_v': dv,
        'offset_u': du,
        'offset_rotation_deg': 0.0,
    }


# Controlled fractions of the body families that draw each surface /
# photometric truth axis.  Every axis is a render-side mismatch the
# navigator's smooth-Lambert prediction cannot model, so each stays a
# minority draw: the clean class must remain dominant for the logistic
# fit, and stacking every axis on every scene would fit the techniques
# against a renderer regime no catalog scene uses.
_BODY_NONLAMBERT_FRAC = 0.40  # photometric_law lommel_seeliger / minnaert
_BODY_MINNAERT_FRAC = 0.15  # of which: minnaert with a drawn k
_BODY_SURGE_FRAC = 0.25  # opposition_surge (matters at low phase)
_BODY_LIMB_RELIEF_FRAC = 0.40  # limb_relief_rms / limb_relief_corr_deg
_BODY_ALBEDO_TEXTURE_FRAC = 0.40  # multiplicative albedo noise field
_BODY_SMOOTH_SHADING_FRAC = 0.30  # mesh gouraud shading (render side only)
_BODY_POSE_SCATTER_FRAC = 0.15  # per-frame unmodelable pose error


def _surface_truth_axes(rng: random.Random, body: dict[str, Any]) -> None:
    """Draw the render-side surface / photometric truth axes onto a body.

    Every key drawn here is a truth key the boundary filter strips: the
    navigator predicts the smooth flat-shaded Lambert body at the catalog
    pose, so each draw plants a controlled model error whose recovered
    offset consequences the calibration fit observes.  Mutually
    independent draws from the family RNG keep the campaign reproducible.

    Parameters:
        rng: Scene-local random generator.
        body: The body entry to extend in place.
    """
    roll = rng.random()
    if roll < _BODY_MINNAERT_FRAC:
        body['photometric_law'] = 'minnaert'
        body['minnaert_k'] = rng.uniform(0.5, 1.1)
    elif roll < _BODY_NONLAMBERT_FRAC:
        body['photometric_law'] = 'lommel_seeliger'
    if rng.random() < _BODY_SURGE_FRAC:
        body['opposition_surge'] = {
            'amplitude': rng.uniform(0.2, 1.0),
            'width_deg': rng.uniform(2.0, 8.0),
        }
    if rng.random() < _BODY_LIMB_RELIEF_FRAC:
        # h/R from smooth-icy (~0.002) through battered-small-moon (~0.03).
        body['limb_relief_rms'] = math.exp(rng.uniform(math.log(0.002), math.log(0.03)))
        body['limb_relief_corr_deg'] = rng.uniform(5.0, 25.0)
    if rng.random() < _BODY_ALBEDO_TEXTURE_FRAC:
        body['albedo_texture'] = {
            'rms': rng.uniform(0.05, 0.30),
            'corr_px': rng.uniform(5.0, 30.0),
        }
    if rng.random() < _BODY_SMOOTH_SHADING_FRAC:
        body['shading'] = 'gouraud'
    if rng.random() < _BODY_POSE_SCATTER_FRAC:
        body['pose_scatter'] = {'sigma_deg': rng.uniform(0.5, 5.0)}


def _catalog_body(
    rng: random.Random,
    *,
    center_v: float,
    center_u: float,
    radius: float,
    phase: float,
    illumination: float,
    irregular_fraction: float,
    pose_error_deg: float = 0.0,
) -> dict[str, Any]:
    """A named body whose rendered relief tracks its published shape residual.

    Draws the identity from the config_220-backed catalog (regular vs
    irregular per ``irregular_fraction``), renders a polyhedral mesh with
    ``mesh_lumpiness`` 3x the body's fractional RMS residual (a
    non-ellipsoidal RMS of ``lumpiness / 8``, so 3/8 of the residual; 8x
    would match it), and has the navigator predict the smooth (zero-relief)
    limit -- so the shape error the navigator cannot model scales with the
    same residual-over-radius ratio the ``max_phase_irregularity_factor``
    confidence term consumes.  ``km_per_pixel`` gives the sim blob feature
    the physical scale that factor needs (it reports 0.0 without it).
    Every body also rolls the surface / photometric truth axes
    (:func:`_surface_truth_axes`).

    Parameters:
        rng: Scene-local random generator.
        center_v: Body center row in image coordinates (px).
        center_u: Body center column in image coordinates (px).
        radius: Apparent body radius (px).
        phase: Phase angle (deg).
        illumination: Image-plane illumination direction (deg).
        irregular_fraction: Probability of drawing an irregular-catalog body.
        pose_error_deg: Pose error applied to the predicted (nav) mesh (deg).

    Returns:
        A sim ``bodies`` entry dict ready for the scene parameter file.
    """
    if rng.random() < irregular_fraction:
        name = rng.choice(_IRREGULAR_BODIES)
    else:
        name = rng.choice(_REGULAR_BODIES)
    mean_radius_km, rms_residual_km = _catalog()[name]
    residual_fraction = rms_residual_km / mean_radius_km
    lumpiness = 3.0 * residual_fraction
    pose = [rng.uniform(0.0, 60.0), rng.uniform(0.0, 60.0), 0.0]
    nav_override: dict[str, Any] = {'mesh_lumpiness': 0.0}
    if pose_error_deg != 0.0:
        nav_override['pose_euler_deg'] = [pose[0], pose[1] + pose_error_deg, pose[2]]
    # Irregular bodies get genuinely tri-axial base ellipsoids; the
    # near-spherical regulars keep modest axis spreads.
    if name in _IRREGULAR_BODIES:
        axis2 = radius * rng.uniform(0.75, 0.95)
        axis3 = radius * rng.uniform(0.6, 0.85)
    else:
        axis2 = radius * rng.uniform(0.97, 1.0)
        axis3 = radius * rng.uniform(0.95, 1.0)
    body: dict[str, Any] = {
        'name': name,
        'shape_model': 'polyhedral_mesh',
        'mesh_lumpiness': lumpiness,
        'mesh_seed': rng.randrange(2**31),
        'pose_euler_deg': pose,
        'center_v': center_v,
        'center_u': center_u,
        'axis1': radius,
        'axis2': axis2,
        'axis3': axis3,
        'illumination_angle': illumination,
        'phase_angle': phase,
        'km_per_pixel': mean_radius_km / radius,
        'nav_override': nav_override,
    }
    _surface_truth_axes(rng, body)
    return body


# Fraction of disc scenes rendering a banded giant-planet-class disc
# (zones/belts plus storm ovals), and, within those, the fraction adding
# a transiting moon and its cast shadow -- the sharp circular false
# crater the disc correlation can lock onto.
_DISC_BANDED_FRAC = 0.15
_DISC_TRANSIT_FRAC = 0.40


def gen_disc(rng: random.Random) -> dict[str, Any]:
    """Resolved body at low-moderate phase (BodyDiscCorrelateNav regime).

    A minority slice renders the giant-planet disc regime: latitude bands
    with storm ovals (``disc_texture``), optionally crossed by a
    transiting moon and its shadow (``transits``) -- all truth-side disc
    texture the navigator's smooth template cannot model.

    Parameters:
        rng: Scene-local random generator.

    Returns:
        A complete sim scene parameter dict.
    """
    size = 200
    params = _base(rng, size=size)
    radius = rng.uniform(18.0, 95.0)
    phase = rng.uniform(5.0, 80.0)
    illumination = rng.uniform(0.0, 40.0)
    center_v = size / 2 + rng.uniform(-0.15, 0.15) * size
    center_u = size / 2 + rng.uniform(-0.15, 0.15) * size
    body = _catalog_body(
        rng,
        center_v=center_v,
        center_u=center_u,
        radius=radius,
        phase=phase,
        illumination=illumination,
        irregular_fraction=0.35,
    )
    if rng.random() < _DISC_BANDED_FRAC:
        storms = []
        for _ in range(rng.randint(0, 2)):
            storms.append(
                {
                    'lat_deg': rng.uniform(-50.0, 50.0),
                    'lon_deg': rng.uniform(60.0, 120.0),
                    'radius_deg': rng.uniform(4.0, 12.0),
                    'albedo_factor': rng.uniform(0.7, 1.3),
                }
            )
        body['disc_texture'] = {
            'band_amplitude': rng.uniform(0.1, 0.3),
            'band_wavenumber': rng.uniform(4.0, 10.0),
            'band_phase_deg': rng.uniform(0.0, 360.0),
            'storms': storms,
        }
        if rng.random() < _DISC_TRANSIT_FRAC:
            # Transiting moon inside the disc; its radius stays well below
            # the planet's so the disc silhouette is unchanged.  The cast
            # shadow disc sits a few pixels from the moon along a random
            # displacement (projected sun-moon-disc geometry), the sharp
            # dark circle the disc correlation can mistake for a crater.
            moon_radius = radius / 2.0 * rng.uniform(0.10, 0.25)
            dv = rng.uniform(-0.5, 0.5) * radius / 2.0
            du = rng.uniform(-0.5, 0.5) * radius / 2.0
            shadow_angle = rng.uniform(0.0, 2.0 * math.pi)
            shadow_sep = moon_radius * rng.uniform(0.5, 2.5)
            body['transits'] = [
                {
                    'moon': {
                        'dv_px': dv,
                        'du_px': du,
                        'radius_px': moon_radius,
                        'albedo_factor': rng.uniform(0.6, 1.4),
                    },
                    'shadow': {
                        'dv_px': dv + shadow_sep * math.sin(shadow_angle),
                        'du_px': du + shadow_sep * math.cos(shadow_angle),
                        'radius_px': moon_radius * rng.uniform(0.9, 1.1),
                        'darkness': rng.uniform(0.5, 0.95),
                    },
                }
            ]
    params['bodies'] = [body]
    return params


def gen_limb(rng: random.Random) -> dict[str, Any]:
    """Large body, often partially off-frame, low phase (BodyLimbNav regime).

    Parameters:
        rng: Scene-local random generator.

    Returns:
        A complete sim scene parameter dict.
    """
    size = 220
    params = _base(rng, size=size)
    radius = rng.uniform(90.0, 190.0)
    phase = rng.uniform(8.0, 55.0)
    illumination = rng.uniform(0.0, 35.0)
    # Slide the center along a random direction so the visible limb arc
    # fraction spans "fully in frame" to "small arc in one corner".
    theta = rng.uniform(0.0, 2.0 * math.pi)
    displacement = rng.uniform(0.0, radius * 0.9)
    center_v = size / 2 + displacement * math.sin(theta)
    center_u = size / 2 + displacement * math.cos(theta)
    # A pose-error slice keeps the "confidently wrong limb" failure mode
    # in the cohort (the trained-away B7-3 scenario).
    pose_error = rng.uniform(2.0, 35.0) if rng.random() < 0.2 else 0.0
    body = _catalog_body(
        rng,
        center_v=center_v,
        center_u=center_u,
        radius=radius,
        phase=phase,
        illumination=illumination,
        irregular_fraction=0.4,
        pose_error_deg=pose_error,
    )
    params['bodies'] = [body]
    return params


def gen_terminator(rng: random.Random) -> dict[str, Any]:
    """High-phase crescent (BodyTerminatorNav regime).

    Parameters:
        rng: Scene-local random generator.

    Returns:
        A complete sim scene parameter dict.
    """
    size = 200
    params = _base(rng, size=size)
    radius = rng.uniform(35.0, 90.0)
    phase = rng.uniform(95.0, 150.0)
    illumination = rng.uniform(5.0, 35.0)
    center_v = size / 2 + rng.uniform(-0.1, 0.1) * size
    center_u = size / 2 + rng.uniform(-0.1, 0.1) * size
    body = _catalog_body(
        rng,
        center_v=center_v,
        center_u=center_u,
        radius=radius,
        phase=phase,
        illumination=illumination,
        irregular_fraction=0.4,
    )
    params['bodies'] = [body]
    return params


def gen_blob(rng: random.Random) -> dict[str, Any]:
    """Small-to-mid body (BodyBlobNav regime).

    The radius range spans from below the 5 px BODY_BLOB emission floor
    (those scenes exercise the fused failure path) through the sizes where
    the disc correlation takes over, so the fit sees the technique's whole
    design regime including the small-blob band the detection-SNR gate
    admits.

    Parameters:
        rng: Scene-local random generator.

    Returns:
        A complete sim scene parameter dict.
    """
    size = 128
    params = _base(rng, size=size)
    radius = rng.uniform(2.5, 30.0)
    phase = rng.uniform(10.0, 130.0)
    illumination = rng.uniform(0.0, 35.0)
    center_v = size / 2 + rng.uniform(-0.2, 0.2) * size
    center_u = size / 2 + rng.uniform(-0.2, 0.2) * size
    body = _catalog_body(
        rng,
        center_v=center_v,
        center_u=center_u,
        radius=radius,
        phase=phase,
        illumination=illumination,
        irregular_fraction=0.5,
    )
    params['bodies'] = [body]
    return params


# Controlled fractions of the ring family drawing the renderer's ring
# truth / orbit vocabulary.  The orbit-shape draws (eccentricity, m-modes,
# edge waves) are CATALOG knowledge -- both sides place the same perturbed
# edges, so they vary geometric complexity without planting a model error
# -- while the orbit-error slice is a render-side truth draw the navigator
# must absorb (the ring analog of the body ephemeris-error axis) and the
# distractor slice adds structure the navigator is never told about.  The
# error/distractor slices stay minority draws so the clean label class
# remains dominant, mirroring the body families' truth-axis fractions.
_RING_FACE_ON_FRAC = 0.4  # scenes keeping the exact face-on (B = 90) identity
_RING_ECCENTRIC_FRAC = 0.35  # features drawing an eccentric orbit (ae > 0)
_RING_MMODE_FRAC = 0.4  # of eccentric features: add one m >= 2 mode
_RING_EDGE_WAVE_FRAC = 0.15  # features carrying a satellite edge wave
_RING_ORBIT_ERROR_FRAC = 0.15  # navigable features planting an orbit error
_RING_DISTRACTOR_FRAC = 0.2  # scenes adding a non-navigable distractor


def _ring_feature(
    rng: random.Random, *, name: str, a: float, width: float, navigable: bool = True
) -> dict[str, Any]:
    """One ringlet feature: log-uniform tau plus the drawn orbit vocabulary.

    tau spans faint (~0.2, a low-contrast band over the noise) through
    optically thick (~4, the saturated closed-form brightness), so the
    ring family's contrast regime spans healthy to marginal.  The orbit
    draws (documented at the fraction constants above):

    - eccentric slice: ae log-uniform 0.5-6 px (radial amplitude well below
      any drawn a) with a drawn pericenter; a sub-slice adds one m = 2-7
      mode at 0.5-4 px amplitude (B-ring-outer-edge-class shapes).
    - edge-wave slice: amplitude 0.5-2.5 px, arc wavelength 6-20 px,
      damping 0.3-1.5 rad (inside the schema's 2.0 rad wrap-seam cap).
    - orbit-error slice (navigable features only): a planted render-side
      radial displacement delta_a_px of 0.5-3 px, either sign -- the
      published-ephemeris error scale for well-tracked ring features --
      with declared_orbit_sigma error bars (the uncertainty the
      navigator is entitled to know) drawn independently of the realized
      error, from the same prior; eccentric features may also draw a
      pericenter error of 5-25 deg.
    """
    tau = math.exp(rng.uniform(math.log(0.2), math.log(4.0)))
    orbit: dict[str, Any] = {
        'a_px': a,
        'ae_px': 0.0,
        'long_peri_deg': 0.0,
        'rate_peri_deg_per_day': 0.0,
    }
    eccentric = rng.random() < _RING_ECCENTRIC_FRAC
    if eccentric:
        orbit['ae_px'] = math.exp(rng.uniform(math.log(0.5), math.log(6.0)))
        orbit['long_peri_deg'] = rng.uniform(0.0, 360.0)
        if rng.random() < _RING_MMODE_FRAC:
            orbit['modes'] = [
                {
                    'm': rng.randint(2, 7),
                    'amp_px': rng.uniform(0.5, 4.0),
                    'peri_deg': rng.uniform(0.0, 360.0),
                }
            ]
    if rng.random() < _RING_EDGE_WAVE_FRAC:
        orbit['edge_wave'] = {
            'amp_px': rng.uniform(0.5, 2.5),
            'wavelength_px': rng.uniform(6.0, 20.0),
            'damping_rad': rng.uniform(0.3, 1.5),
            'lam0_deg': rng.uniform(0.0, 360.0),
        }
    feature: dict[str, Any] = {
        'name': name,
        'kind': 'ringlet',
        'tau': tau,
        'width_px': width,
        'navigable': navigable,
        'orbit': orbit,
    }
    if navigable and rng.random() < _RING_ORBIT_ERROR_FRAC:
        # The declared error bar is drawn INDEPENDENTLY of the realized
        # planted error, from the same prior the error magnitude is drawn
        # from.  A sigma derived from the realization (e.g. 0.8-1.5x the
        # drawn |delta_a|) would let any future fit that consumes the
        # declared sigma learn that it predicts the per-frame realized
        # error -- information a real published uncertainty never carries
        # -- and the calibration would overstate such a fix's value.
        delta_a = rng.uniform(0.5, 3.0) * rng.choice((-1.0, 1.0))
        orbit_error: dict[str, float] = {'delta_a_px': delta_a}
        declared_sigma: dict[str, float] = {'sigma_a_px': rng.uniform(0.5, 3.0)}
        if eccentric and rng.random() < 0.5:
            orbit_error['delta_long_peri_deg'] = rng.uniform(5.0, 25.0) * rng.choice((-1.0, 1.0))
            declared_sigma['sigma_long_peri_deg'] = rng.uniform(5.0, 25.0)
        feature['orbit_error'] = orbit_error
        feature['declared_orbit_sigma'] = declared_sigma
    return feature


def gen_ring(rng: random.Random) -> dict[str, Any]:
    """Ringlet scene: curved through near-flat arcs, 1-2 ringlets, narrow to wide.

    Curvature is controlled by pushing the ring center off-frame: an arc from
    a ring of radius R passing through a frame R away from its center is
    nearly straight (the rank-1 regime), while an in-frame center gives
    fully-curved closed edges.  A face-on slice (B = 90, the
    sky-plane-circle identity) keeps the drawn radii as exact image radii;
    the rest of the family draws a moderately-inclined observer opening
    angle (B_obs 12-80 deg, uniformly: gently tilted through strongly
    foreshortened ellipses) with an independently perturbed solar opening
    angle B_sun (same sign, so the lit-face closed form applies) and a
    random node, spanning the projection regimes the flat identity cannot
    reach.  Brightness follows each feature's tau and both opening angles
    through the single-scattering photometry.  A distractor slice appends
    a non-navigable ringlet the navigator is never told about, radially
    adjacent to the navigable features.
    """
    import numpy as np

    from spindoctor.sim.ring_geometry import ring_plane_from_sky

    size = 220
    params = _base(rng, size=size)
    if rng.random() < _RING_FACE_ON_FRAC:
        b_obs = 90.0
        b_sun = 90.0
        node = 0.0
    else:
        b_obs = rng.uniform(12.0, 80.0)
        b_sun = min(90.0, max(5.0, b_obs + rng.uniform(-15.0, 15.0)))
        node = rng.uniform(0.0, 360.0)
    features: list[dict[str, Any]] = []
    n_ringlets = 1 if rng.random() < 0.7 else 2
    flat = rng.random() < 0.3
    if flat:
        # Distant center: edges cross the frame as gentle arcs.  The base
        # semimajor axis anchors on the ring-plane radius of the frame
        # center under the drawn projection (the shared inverse
        # projection), which keeps the navigable feature crossing the
        # frame for every (B, node, center) combination; at B = 90 it
        # reduces exactly to the sky-plane distance, the previous
        # face-on-only behavior.
        big_r = rng.uniform(300.0, 900.0)
        theta = rng.uniform(0.0, 2.0 * math.pi)
        center_v = size / 2 + big_r * math.sin(theta)
        center_u = size / 2 + big_r * math.cos(theta)
        r_center, _lam, _x, _y = ring_plane_from_sky(
            np.asarray(size / 2 - center_v),
            np.asarray(size / 2 - center_u),
            opening_deg_obs=b_obs,
            node_deg=node,
        )
        base_a = float(r_center) + rng.uniform(-30.0, 30.0)
    else:
        # In-frame center: fully-curved closed edges.  The drawn radii
        # exceed the largest center displacement, so the projected
        # ellipse's edges stay in frame at every drawn opening angle
        # (the minor axis compresses toward the center as B falls).
        center_v = size / 2 + rng.uniform(-40.0, 40.0)
        center_u = size / 2 + rng.uniform(-40.0, 40.0)
        base_a = rng.uniform(40.0, 80.0)
    for index in range(n_ringlets):
        width = rng.uniform(2.0, 35.0)
        features.append(_ring_feature(rng, name=f'SATURN-{index + 1}', a=base_a, width=width))
        base_a += width + rng.uniform(10.0, 30.0)
    if rng.random() < _RING_DISTRACTOR_FRAC:
        # Non-navigable clutter: a ringlet just outside the navigable
        # ones.  It renders (and can alias a coarse edge search) but is
        # dropped from nav_params by the boundary filter.
        width = rng.uniform(2.0, 20.0)
        features.append(
            _ring_feature(
                rng,
                name='DISTRACTOR',
                a=base_a + rng.uniform(5.0, 25.0),
                width=width,
                navigable=False,
            )
        )
    params['ring_system'] = {
        'geometry': {
            'center_v': center_v,
            'center_u': center_u,
            'opening_deg_obs': b_obs,
            'opening_deg_sun': b_sun,
            'node_deg': node,
        },
        'features': features,
    }
    return params


# Controlled fractions of the star families that exercise the star
# information-asymmetry regimes.  They are deliberately a minority so the
# clean-lock label class stays dominant, and they are drawn from the family's
# own RNG so a campaign stays reproducible.
_STAR_FIELD_CATALOG_SCATTER_FRAC = 0.3  # scenes planting a small catalog scatter
_STAR_FIELD_CONFOUNDER_FRAC = 0.25  # scenes marking some field stars non-navigable
_STAR_UNIQUE_CATALOG_ERROR_FRAC = 0.3  # scenes planting a per-star catalog error
_STAR_UNIQUE_COMPANION_FRAC = 0.2  # scenes planting an unresolved companion


def gen_star_field(rng: random.Random) -> dict[str, Any]:
    """3-15 stars, bright through sub-detection (star-technique regimes)."""
    size = 128
    params = _base(rng, size=size)
    # Stars deposit as flux-normalized point masses; the navigator-matched PSF
    # gives them the floor-form profile (no 1-pixel spike).
    params['optics'] = {'psf': {'match_navigator': True}}
    # The flux-normalized coiss_nac limiting magnitude is ~10.9 at read noise 4
    # (ObsSim.star_max_usable_vmag), ~5 mag deeper than the prior peak-normalized
    # model; the magnitude ranges shift with it so the brightness regime still
    # spans "well above the limit" to "straddling it".  Cap the star-scene noise
    # a little lower than the body/ring families.
    params['noise']['read_noise_dn'] = _read_noise(rng, hi=16.0)
    n_stars = rng.randint(3, 15)
    # Per-scene brightness regime so whole frames span easy to marginal
    # (a per-star-only draw would almost always leave >= 3 bright stars).
    mag_lo = rng.uniform(7.0, 11.5)
    mag_hi = mag_lo + rng.uniform(0.5, 2.5)
    margin = 12.0
    stars = []
    for index in range(n_stars):
        stars.append(
            {
                'name': f'C{index + 1}',
                'v': rng.uniform(margin, size - margin),
                'u': rng.uniform(margin, size - margin),
                'vmag': rng.uniform(mag_lo, mag_hi),
            }
        )
    params['stars'] = stars
    # Astrometric-error regime: a small scene-level catalog scatter displaces
    # every rendered star off its predicted position, sweeping the tolerance the
    # star fit absorbs before the match degrades.
    if rng.random() < _STAR_FIELD_CATALOG_SCATTER_FRAC:
        params['star_catalog_scatter_px'] = rng.uniform(0.2, 1.0)
    # Star-clutter regime: mark a fraction of the field non-navigable so it
    # renders as a confounder the navigator has no knowledge of, while keeping at
    # least three navigable stars (the pattern matcher's minimum).
    max_confounders = max(0, n_stars - 3)
    if max_confounders > 0 and rng.random() < _STAR_FIELD_CONFOUNDER_FRAC:
        n_confounders = rng.randint(1, max_confounders)
        for star in rng.sample(stars, n_confounders):
            star['navigable'] = False
    if rng.random() < 0.25:
        params['optics']['stray_light'] = {
            'amplitude': rng.uniform(0.05, 0.5),
            'direction_deg': rng.uniform(0.0, 360.0),
            'model': 'linear',
        }
    return params


def gen_star_unique(rng: random.Random) -> dict[str, Any]:
    """1-2 star scenes with varying brightness margin (StarUniqueMatchNav)."""
    size = 128
    params = _base(rng, size=size)
    params['optics'] = {'psf': {'match_navigator': True}}
    params['noise']['read_noise_dn'] = _read_noise(rng, hi=16.0)
    margin = 16.0
    primary_mag = rng.uniform(7.0, 11.5)
    primary = {
        'name': 'U1',
        'v': rng.uniform(margin, size - margin),
        'u': rng.uniform(margin, size - margin),
        'vmag': primary_mag,
    }
    # Physical or planted catalog error on the primary: either an unresolved
    # companion (a magnitude-weighted photocenter shift) or a small planted
    # per-star position error.  These are mutually exclusive so the drawn regime
    # is unambiguous, and the companion is kept faint and close so the biased
    # centroid stays a sub-pixel model error rather than a resolved second star.
    roll = rng.random()
    if roll < _STAR_UNIQUE_COMPANION_FRAC:
        primary['companion'] = {
            'sep_px': rng.uniform(1.5, 3.0),
            'delta_mag': rng.uniform(1.5, 3.0),
            'angle_deg': rng.uniform(0.0, 360.0),
        }
    elif roll < _STAR_UNIQUE_COMPANION_FRAC + _STAR_UNIQUE_CATALOG_ERROR_FRAC:
        primary['catalog_error_v'] = rng.uniform(-1.0, 1.0)
        primary['catalog_error_u'] = rng.uniform(-1.0, 1.0)
    stars = [primary]
    if rng.random() < 0.55:
        # Two-star scene; the pairwise brightness gap varies the
        # assignment ambiguity the two-star path must resolve.
        stars.append(
            {
                'name': 'U2',
                'v': rng.uniform(margin, size - margin),
                'u': rng.uniform(margin, size - margin),
                'vmag': primary_mag + rng.uniform(0.1, 3.0),
            }
        )
    params['stars'] = stars
    return params


# Titan family: the apparent-size range, phase range, and haze column the
# planted-truth campaign draws (util/titan_truth/scene_gen.py), narrowed
# only where this campaign's purpose differs.  The planted offset comes from
# the shared ``_base`` draw, so Titan rows sit in the same pointing-error
# regime as every other family and the fitted confidence is comparable
# across techniques.
_TITAN_RADIUS_KM = 2575.0
_TITAN_FRAME_PX = 360
_TITAN_MIN_SOLID_RADIUS_PX = 28.0
_TITAN_MAX_SOLID_RADIUS_PX = 78.0
_TITAN_MIN_PHASE_DEG = 10.0
_TITAN_MAX_PHASE_DEG = 140.0

# Controlled fractions of the Titan family breaking one class of the
# method's assumptions each.  Kept a minority for the same reason the body
# families keep theirs: the logistic fit needs the clean label class to stay
# dominant, and a campaign where every scene is broken measures the failure
# regime rather than the operating one.
_TITAN_CLOUD_FRAC = 0.30
_TITAN_ASYMMETRY_FRAC = 0.30
_TITAN_STARS_FRAC = 0.25
_TITAN_ARTIFACT_FRAC = 0.20


def gen_titan(rng: random.Random) -> dict[str, Any]:
    """A hazy Titan with a drawn size, phase, sun direction, and structure."""
    params = _base(rng, size=_TITAN_FRAME_PX)
    params['noise']['read_noise_dn'] = _read_noise(rng, hi=16.0)
    center = _TITAN_FRAME_PX / 2.0
    r_solid_px = rng.uniform(_TITAN_MIN_SOLID_RADIUS_PX, _TITAN_MAX_SOLID_RADIUS_PX)
    illumination_deg = rng.uniform(0.0, 360.0)
    atmosphere: dict[str, Any] = {
        'scale_height_px': r_solid_px * rng.uniform(0.08, 0.18),
        'tau_ref': rng.uniform(2.0, 5.0),
        'ref_altitude_px': r_solid_px * rng.uniform(0.02, 0.10),
        'g': rng.uniform(0.2, 0.7),
    }
    if rng.random() < _TITAN_CLOUD_FRAC:
        blobs = []
        for _ in range(rng.randint(1, 4)):
            radius = 0.75 * r_solid_px * math.sqrt(rng.random())
            angle = rng.uniform(0.0, 2.0 * math.pi)
            blobs.append(
                {
                    'center_vu': [radius * math.sin(angle), radius * math.cos(angle)],
                    'sigma_px': r_solid_px * rng.uniform(0.05, 0.20),
                    'amplitude': rng.choice((-1.0, 1.0)) * rng.uniform(0.10, 0.35),
                }
            )
        atmosphere['cloud_blobs'] = blobs
    if rng.random() < _TITAN_ASYMMETRY_FRAC:
        atmosphere.update(
            {
                'axis_tilt_deg': rng.uniform(-12.0, 12.0),
                'ns_falloff_ratio': rng.uniform(0.6, 1.8),
                'ns_asymmetry_amplitude': rng.uniform(-0.35, 0.35),
                'sector_sharpness_gradient': rng.uniform(-0.3, 1.0),
                'interior_ramp_amplitude': rng.uniform(-0.25, 0.25),
            }
        )
    params['bodies'] = [
        {
            'name': 'TITAN',
            'shape_model': 'ellipsoid',
            'center_v': center,
            'center_u': center,
            'axis1': 2.0 * r_solid_px,
            'axis2': 2.0 * r_solid_px,
            'axis3': 2.0 * r_solid_px,
            'illumination_angle': illumination_deg,
            # The structure keys split hemispheres on the body-frame first
            # axis; this roll puts that split line along the sun axis, which
            # is what makes ns_asymmetry_amplitude the affine control the
            # Pearson score is meant to absorb -- affine only poleward of the
            # renderer's 10 degree equatorial blend band, since a continuous
            # blend must match the two halves on the split line itself.
            'rotation_z': illumination_deg - 90.0,
            'phase_angle': rng.uniform(_TITAN_MIN_PHASE_DEG, _TITAN_MAX_PHASE_DEG),
            'km_per_pixel': _TITAN_RADIUS_KM / r_solid_px,
            'range_km': 1.2e6,
            'atmosphere': atmosphere,
        }
    ]
    if rng.random() < _TITAN_STARS_FRAC:
        stars = []
        for index in range(rng.randint(3, 12)):
            while True:
                v = rng.uniform(4.0, _TITAN_FRAME_PX - 4.0)
                u = rng.uniform(4.0, _TITAN_FRAME_PX - 4.0)
                if math.hypot(v - center, u - center) > r_solid_px:
                    break
            stars.append({'name': f'S{index}', 'v': v, 'u': u, 'vmag': rng.uniform(4.0, 9.5)})
        params['stars'] = stars
    if rng.random() < _TITAN_ARTIFACT_FRAC:
        # The instrument's own realism-matched defect population, nothing
        # overridden: the planted-truth campaign showed the stress condition
        # halves the commit rate, which would move this fit off the
        # operating regime it is meant to calibrate.
        params['artifacts'] = {'instrument_defaults': True}
    return params


_GENERATORS: dict[str, Callable[[random.Random], dict[str, Any]]] = {
    'disc': gen_disc,
    'limb': gen_limb,
    'terminator': gen_terminator,
    'blob': gen_blob,
    'ring': gen_ring,
    'star_field': gen_star_field,
    'star_unique': gen_star_unique,
    'titan': gen_titan,
}


def generate_scenes(
    family: str, count: int, *, campaign_seed: int
) -> list[tuple[str, dict[str, Any]]]:
    """Return ``count`` seeded scenes for ``family`` as (scene_id, sim_params).

    Each scene draws from its own ``random.Random`` seeded by
    ``(campaign_seed, family, index)`` so any single scene can be regenerated
    without replaying the whole campaign.

    Parameters:
        family: One of :data:`FAMILIES`.
        count: Number of scenes to generate.
        campaign_seed: Campaign-level seed recorded in the output manifest.

    Returns:
        List of ``(scene_id, sim_params)`` pairs.
    """
    # Local import keeps this module importable without the package installed
    # (mirrors the deferred config import above).
    from spindoctor.sim.scene import validate_sim_params

    if family not in _GENERATORS:
        raise ValueError(f'unknown scene family {family!r}; valid: {sorted(_GENERATORS)}')
    generator = _GENERATORS[family]
    scenes = []
    for index in range(count):
        rng = random.Random(f'{campaign_seed}/{family}/{index}')
        scene_id = f'{family}_{index:05d}'
        # Campaign scenes are dict-authored (never files), so they validate
        # here; a generator drifting from the schema fails at generation.
        scenes.append((scene_id, validate_sim_params(generator(rng), source=scene_id)))
    return scenes
