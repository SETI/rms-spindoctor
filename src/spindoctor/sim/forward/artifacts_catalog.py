"""Per-instrument artifact defaults for the forward model.

These tables hold the per-instrument optical parameters an ``instrument_defaults``
scene turns on: the whole-scene PSF kernel and the residual geometric-distortion
amplitude.  Every value here is interim -- sized from published FWHMs and
documented residual-error bounds, pending the per-instrument measurement passes
-- and is provenance-tagged as such in the comments beside it.

Keys are sim instrument names (see ``spindoctor.sim.instruments.SIM_INSTRUMENTS``).
"""

import copy
from collections.abc import Mapping
from typing import Any

from spindoctor.sim.forward.artifact_modes import ARTIFACT_MODES, MODE_KEYS

__all__ = [
    'ARTIFACT_MODES',
    'DETECTOR_DEFAULTS',
    'DISTORTION_RESIDUAL_PARAMS',
    'MODE_KEYS',
    'PSF_KERNELS',
    'SKY_PIXEL_SCALE_ARCSEC',
    'STAR_FLUX_DN_PER_S_VMAG0',
    'STAR_FLUX_E_PER_S_VMAG0',
    'resolve_detector_defaults',
    'resolve_mode_with_catalog',
    'resolve_sky_pixel_scale_arcsec',
    'resolve_star_flux_zero_point',
]

# The artifact-mode registry (:mod:`spindoctor.sim.forward.artifact_modes`) is the
# single source of truth for the ``artifacts`` block's per-mode keys, their
# rendering stage, per-instrument availability, and parameter schemas.  It is
# re-exported here so the detector catalog and the mode registry share one
# import point.  The *missing-data* loss modes default to 0 even under
# ``instrument_defaults`` -- naming an instrument selects a signal chain, not a
# set of transmission defects -- so they are planted only by scenes that
# configure them explicitly.  Cosmic rays are different: radiation on the
# detector is environment physics, so each instrument entry carries a
# ``cosmic_ray_rate_per_sec`` that is either cohort-measured or an explicit
# retained zero with its reason (see the per-entry comments).

# Whole-scene PSF kernel parameters per instrument, all radii in detector
# pixels.  Wing parameters (w, r0, n) are expressed as wing energy fractions
# so the delivered kernels conserve flux.
#
# Provenance:
# - coiss_nac: TUNED 2026-07-17 by the realism match against the Cassini
#   CALIB image-library cohort (brightest unsaturated stars over the 11
#   contributing NAC star frames): cohort encircled-energy radii EE50 = 0.91 px,
#   EE80 = 1.79 px; these parameters reproduce EE50/EE80 = 0.90/1.72 px
#   through the same estimator.  The EE80/EE50 ratio (1.97 vs 1.52 for a
#   pure Gaussian) requires substantial mid-range wing energy, so this is
#   an *effective* in-window kernel: the fitted w = 0.12 carries the
#   measured 1-8 px halo energy that the interim FWHM-derived core
#   (sigma 0.55, w 0.025) put nowhere.  A first fit at w = 0.20 matched
#   the encircled energy but measurably over-lifted the wide-field halo
#   (sky-patch means and mid-size-body limb widths); w = 0.12 with the
#   softer core matches the stars without that excess.  The far halo
#   beyond the truncation window remains stray-light scope.  Replaces the
#   interim FWHM/2.355 value.
# - coiss_wac: TUNED 2026-07-17 from the cohort's two WAC star frames
#   (9 usable star cutouts): cohort EE50/EE80 = 1.33/2.16 px through the
#   final estimator; sigma follows the measured EE50 with the NAC wing
#   shape.  Two-frame evidence -- treat as cohort-limited, revisit when
#   more WAC star frames land.  Replaces the interim FWHM/2.355 value.
# - vgiss: RETAINED interim estimate: the cohort's one star frame (8
#   usable star cutouts, EE50 1.22 px) cannot constrain the kernel
#   through GEOMED resampling (the flat-top guard leaves two simulated
#   cutouts to compare against); the Voyager references publish no FWHM.
# - gossi: RETAINED interim published sigma: the Galileo cohort holds no
#   star frames (negative cases only), so there is no independent PSF
#   evidence; gossi sim accuracy is bounded by unverified PSF fidelity
#   until the star-calibration frames land.
# - nhlorri: RETAINED interim elliptical values (LORRI PSF references,
#   1x1 mode).  The cohort's two star frames are 4x4-binned (measured
#   binned-pixel EE50 0.59 px) and cannot constrain the 1x1 kernel; a
#   per-readout-mode kernel is future work.
PSF_KERNELS: dict[str, dict[str, float]] = {
    'coiss_nac': {'sigma_v': 0.70, 'sigma_u': 0.70, 'w': 0.12, 'r0': 3.0, 'n': 4.0},
    'coiss_wac': {'sigma_v': 1.05, 'sigma_u': 1.05, 'w': 0.12, 'r0': 3.0, 'n': 4.0},
    'vgiss': {'sigma_v': 0.85, 'sigma_u': 0.85, 'w': 1.2e-2, 'r0': 2.0, 'n': 3.0},
    'gossi': {'sigma_v': 0.80, 'sigma_u': 0.80, 'w': 1.2e-2, 'r0': 2.0, 'n': 3.0},
    'nhlorri': {'sigma_v': 1.13, 'sigma_u': 0.87, 'w': 1.2e-2, 'r0': 2.0, 'n': 3.0},
}

# Residual geometric-distortion RMS displacement over the frame, in detector
# The residual geometric distortion remaining after the navigator applies each
# instrument's known distortion model -- the only distortion actually present
# in the frames the pipeline consumes.  Each block is a radial polynomial in the
# normalized field radius (``k1`` on rho^2, ``k2`` on rho^4, matching the sim
# distortion warp) plus a non-radial wander RMS.  Values are measured from the
# per-instrument star-field FOV distortion analysis: the Cassini corrected field
# is sub-pixel and close to radially symmetric, Galileo carries a pincushion
# term reaching ~0.5 px at the corner, New Horizons LORRI is near its noise
# floor, and the Voyager GEOMED products carry several tenths of a pixel of both
# radial and non-radial distortion.  The Voyager block is the Voyager 2 WAC
# measurement; the two Voyager missions and their narrow- and wide-angle cameras
# have distinct optics, so a single key under-represents that spread until
# per-camera keys and locked frames for each are available.
DISTORTION_RESIDUAL_PARAMS: dict[str, dict[str, float]] = {
    'coiss_nac': {'k1': 3.17e-04, 'k2': -3.51e-04, 'nonradial_rms_px': 0.0},
    'coiss_wac': {'k1': 9.0e-06, 'k2': 5.48e-05, 'nonradial_rms_px': 0.0},
    'gossi': {'k1': -3.47e-04, 'k2': 1.72e-03, 'nonradial_rms_px': 0.0},
    'nhlorri': {'k1': 8.13e-04, 'k2': -1.10e-03, 'nonradial_rms_px': 0.0},
    'vgiss': {'k1': -6.88e-03, 'k2': 1.46e-02, 'nonradial_rms_px': 0.2},
}


# Photometric zero point: the flux a magnitude-0 star deposits per second of
# exposure, so a star's total signal is
# ``zero_point * 10**(-0.4 * vmag) * exposure_sec`` -- flux normalization, not a
# peak-normalized profile (the optics PSF then dictates the shape and the peak).
# The CCD table is in electrons per second (deposited into the detector's
# electron plane before Poisson); the vidicon table is in DN per second (the
# vidicon has no electron domain, so its point sources are DN).  All values are
# interim and provenance-tagged:
# - The Cassini / LORRI / Galileo electron zero points are sized from each
#   camera's Section 5 limiting magnitude at SNR ~5 (a magnitude-0 star of a
#   given zero point lands its limiting magnitude near the matched-filter
#   detection boundary at that camera's read noise and PSF core).
# - The Voyager DN zero point is sized so the 5.3 vidicon limiting magnitudes
#   land at the matched-filter boundary in the DN chain.
# - The generic zero point is DERIVED, not measured: it reproduces the prior
#   peak-normalized brightness of a magnitude-4 star at the generic detector
#   (signal_full_scale_frac 0.5, full_well 4095 e-, star_psf_sigma 1.0), whose
#   integrated signal was ``(1 / 2.512**4) * 2*pi * 1.0**2 * 0.5 * 4095`` ~ 324
#   electrons; ``1.3e4 * 10**(-0.4 * 4)`` ~ 326 matches it, so a generic scene's
#   stars keep their sane DN range and the unit-test workhorse is undisturbed.
STAR_FLUX_E_PER_S_VMAG0: dict[str, float] = {
    'coiss_nac': 1.0e7,
    'coiss_wac': 2.6e6,
    'gossi': 4.0e6,
    'nhlorri': 7.0e7,
    'generic': 1.3e4,
}

# Vidicon photometric zero point in DN per second (Voyager: no electron domain).
STAR_FLUX_DN_PER_S_VMAG0: dict[str, float] = {
    'vgiss': 3.0e3,
}

# Angular pixel scale (arcsec / detector pixel) used to convert a scene's
# pixel-area field of view into square degrees for the background-sky star
# count.  A camera constant, kept here beside the other interim sim per-instrument
# radiometry tables rather than in the shared real-pipeline config blocks.  All
# values interim, provenance-tagged from the published plate scales (Section 12):
# COISS NAC ~6 urad/px, WAC ~60 urad/px, Galileo SSI ~10.2 urad/px, LORRI ~5
# urad/px (1x1), Voyager ISS ~9.3 urad/px; the generic block is a nominal 1
# arcsec/px.
SKY_PIXEL_SCALE_ARCSEC: dict[str, float] = {
    'coiss_nac': 1.237,
    'coiss_wac': 12.37,
    'gossi': 2.095,
    'nhlorri': 1.023,
    'vgiss': 1.910,
    'generic': 1.0,
}


def _coiss_alias(table: dict[str, Any]) -> None:
    """Point the calibrated Cassini instrument names at their raw entries."""
    for calib, raw in (('coiss_calib_nac', 'coiss_nac'), ('coiss_calib_wac', 'coiss_wac')):
        if raw in table and calib not in table:
            table[calib] = table[raw]


_coiss_alias(PSF_KERNELS)
_coiss_alias(DISTORTION_RESIDUAL_PARAMS)
_coiss_alias(STAR_FLUX_E_PER_S_VMAG0)
_coiss_alias(SKY_PIXEL_SCALE_ARCSEC)


def _catalog_key(instrument: str | None) -> str:
    """The catalog key for a sim instrument name (generic aliases fold to one)."""
    if instrument is None or instrument in ('generic', 'sim'):
        return 'generic'
    return instrument


def resolve_star_flux_zero_point(instrument: str | None) -> tuple[float, str]:
    """Return the star photometric zero point and its unit domain.

    Parameters:
        instrument: The sim instrument name, a generic alias, or ``None``.

    Returns:
        A ``(zero_point, domain)`` pair.  ``domain`` is ``'dn'`` for the vidicon
        (its point sources are DN) and ``'electrons'`` for every CCD camera and
        the generic detector.  ``zero_point`` is the flux a magnitude-0 star
        deposits per second of exposure in that domain.
    """
    key = _catalog_key(instrument)
    if key in STAR_FLUX_DN_PER_S_VMAG0:
        return STAR_FLUX_DN_PER_S_VMAG0[key], 'dn'
    return STAR_FLUX_E_PER_S_VMAG0.get(key, STAR_FLUX_E_PER_S_VMAG0['generic']), 'electrons'


def resolve_sky_pixel_scale_arcsec(instrument: str | None) -> float:
    """Return the angular pixel scale (arcsec / pixel) for the sky-count FOV area.

    Parameters:
        instrument: The sim instrument name, a generic alias, or ``None``.

    Returns:
        The interim plate scale in arcsec per detector pixel.
    """
    return SKY_PIXEL_SCALE_ARCSEC.get(_catalog_key(instrument), SKY_PIXEL_SCALE_ARCSEC['generic'])


# Per-instrument detector electron-chain and noise-model parameters.  Every
# value is interim and provenance-tagged; the electron chain, gain tables, read
# noise, and vidicon numbers come from the instrument calibration references
# (plan Section 5) and are the first quantities the realism-match pass revisits.
#
# Chain (CCD): electrons = signal * signal_full_scale_frac * full_well_e *
# (exposure_sec / exposure_ref_sec); Poisson; electron-domain full-well bloom
# against full_well_e; read noise (read_noise_e); DN = electrons / gain_e_per_dn
# + bias_dn; quantization; clip at saturation_dn.  gain_e_per_dn is selected
# from gain_e_per_dn_by_state[gain_state]; a scene selecting a state absent from
# that table is a validation error (5.2), not a silent guess.  The image-side
# well in DN is DERIVED (full_well_e / gain_e_per_dn); the navigator-side
# full_well_dn config key is a separate published ADC-referenced value.
#
# The 'instrument_defaults' physical-chain knobs (dark_current_e_per_sec,
# hot_pixel_fraction, hot_pixel_amplitude_e, banding_amplitude_e,
# banding_period_px, bias_pedestal_sigma_dn, bias_row_gradient_dn,
# bias_col_gradient_dn, bloom_length, quantization) are interim placeholders
# sized from the 5.2-5.6 descriptions; per-scene overrides ride in the
# truth-side noise block.  instrument_defaults also turns on Poisson shot
# noise (a property of the electron chain itself, not a per-camera number)
# and the entry's cosmic_ray_rate_per_sec; the missing-data loss modes are
# artifact incidences and stay at zero.
#
# cosmic_ray_rate_per_sec is in events per pixel per second: the chain
# multiplies it by exposure_sec, pixel_area_cm2 (default 1.0), and the pixel
# count, so at the default unit pixel area the fluence reads per pixel.
# The realism match's FOM 6 transient split (single-pixel spikes that do
# NOT recur at fixed detector positions across the cohort) measures the
# candidate rate per instrument.  Every current entry retains zero, each
# with its measured rate, the specific reason adoption fails through this
# chain (exposure-scaling / full-well-amplitude mismatch for Cassini,
# double-counting against the tuned per-scene hot pixels for Galileo,
# star-field contamination for LORRI, no vidicon stage for Voyager), and
# the unblocking condition, recorded beside the value.
#
# The vidicon (Voyager) path skips the electron conversion; its noise is applied
# directly in DN (5.3): line-correlated read noise (per-line offset +
# within-line white), a faint coherent periodic component, and 8-bit
# quantization.  A vidicon entry carries a 'vidicon' sub-map instead of the CCD
# electron keys.
DETECTOR_DEFAULTS: dict[str, dict[str, Any]] = {
    'coiss_nac': {
        'detector_model': 'ccd',
        # 5.2 interim: full well ~110k e- (saturates ~3600 DN at gain 2, below
        # the 4095 ADC clip).
        'full_well_e': 110.0e3,
        # 5.2 interim: a typical NAC science exposure, so a converted scene at
        # this exposure keeps today's brightness scale.
        'exposure_ref_sec': 1.0,
        # 5.2 interim gain states ~233/95/30/13 e-/DN; tour-standard state 2.
        'gain_e_per_dn_by_state': {0: 233.0, 1: 95.0, 2: 30.0, 3: 13.0},
        'default_gain_state': 2,
        'read_noise_e': 12.0,  # 5.2 interim
        'bias_dn': 20.0,
        'dark_current_e_per_sec': 5.0,  # 5.2 interim (RBI-dominated dark)
        'hot_pixel_fraction': 2.0e-3,  # 5.2 interim (~0.15-0.28% of pixels)
        'hot_pixel_amplitude_e': 4.0e4,  # 5.2 interim (near full well)
        # 5.2 interim: total-charge fraction bled into the warm column above a
        # hot pixel (the streak integral, not a per-pixel amplitude).
        'hot_pixel_column_factor': 0.3,
        # RETAINED zero 2026-07-18 after an adoption attempt.  The realism
        # match measures a real transient single-pixel spike fraction of
        # 2.75e-4 per frame on the 58-frame CALIB NAC cohort (stationary
        # hot pixels excluded; star-bearing and body/ring frames agree at
        # 2.60e-4 vs 2.79e-4, so scene point sources are not the driver).
        # But the measured incidence is exposure-INdependent -- mean
        # fraction ~2e-4 in every exposure band from <= 0.2 s to > 2 s,
        # and the 22 s frame measures 1.1e-4 where an exposure-scaling
        # fluence fitted at the median frame predicts ~3e-3 -- while this
        # chain's cosmic-ray stage scales event counts with exposure_sec
        # and deposits near full well.  Adopting a fitted 1.5e-4
        # events/px/s (tried 2026-07-17) corrupted the matched frames'
        # FOM 1 statistics (sim sky-patch sigma median 2.2e-4 -> 1.0e-3
        # I/F against 2.1e-4 real) and overshot the FOM 6 transient
        # fraction ~2x through the multi-pixel event morphology.
        # Unblocked by a per-readout (exposure-independent),
        # modest-amplitude transient term in the detector chain.
        'cosmic_ray_rate_per_sec': 0.0,
        'banding_amplitude_e': 30.0,  # 5.2 interim (~30 e- NAC 2 Hz)
        'banding_period_px': 64.0,  # 5.2 interim (line-readout-rate period)
        'bias_pedestal_sigma_dn': 2.0,  # 5.2 interim (per-image pedestal jitter)
        'bias_row_gradient_dn': 1.0,  # 5.2 interim (readout-direction gradient)
        'bias_col_gradient_dn': 0.5,  # 5.2 interim
        'bloom_length': 4,  # 5.2 interim (no antiblooming; column bleed above the well)
        'quantization': 'exact',
        # Per-mode shape defaults (interim, 5.2).  incidence is never cataloged:
        # a mode activates only when a scene sets it.  Shapes here are the values
        # a scene inherits when it names a mode without spelling every parameter.
        'artifact_modes': {
            # ~30 e- NAC 2 Hz horizontal banding, line-readout-rate period.
            'banding_coherent': {
                'amplitude_e': 30.0,
                'period_px': 64.0,
                'orientation': 'horizontal',
            },
            # Per-image pedestal jitter plus readout-direction gradients.
            'bias_structure': {
                'pedestal_sigma_dn': 2.0,
                'row_gradient_dn': 1.0,
                'col_gradient_dn': 0.5,
            },
            # RBI-dominated dark grows toward the last readout line.
            'dark_ramp': {'kind': 'dark_gradient', 'amplitude_e': 150.0},
            'bloom': {'bloom_length': 4},
            # Anti-blooming vertical 2-px pairs in unsummed long exposures.
            'bright_dark_pairs': {'amplitude_e': 4.0e3},
            # No published ISS rate; the interplanetary regime near full-well amp.
            'radiation_transients': {'amplitude_e': 4.0e4},
            # Dust donuts (<1% each) accumulating over the mission.
            'fixed_pattern': {'dust_donut_count': 5},
        },
    },
    'coiss_wac': {
        'detector_model': 'ccd',
        'full_well_e': 95.0e3,  # 5.2 interim
        'exposure_ref_sec': 1.0,
        # 5.2: only state 2 is cataloged for the WAC; selecting another WAC
        # state is a validation error until its full table is sourced.
        'gain_e_per_dn_by_state': {2: 28.0},
        'default_gain_state': 2,
        'read_noise_e': 12.0,  # 5.2 interim
        'bias_dn': 20.0,
        'dark_current_e_per_sec': 5.0,
        'hot_pixel_fraction': 2.0e-3,
        'hot_pixel_amplitude_e': 3.5e4,
        'hot_pixel_column_factor': 0.3,
        # RETAINED zero 2026-07-18: same exposure-independence and
        # full-well-amplitude mismatch as the NAC entry (the two cameras
        # share this chain model), and the WAC's own 4-frame cohort is
        # star-contaminated besides -- its measured 4.89e-4 per-frame
        # transient fraction comes entirely from the three star-bearing
        # frames while the one body frame measures zero.  Unblocked
        # together with the NAC's per-readout transient term.
        'cosmic_ray_rate_per_sec': 0.0,
        'banding_amplitude_e': 6.0,  # 5.2 interim (~6 e- WAC 4 Hz)
        'banding_period_px': 64.0,
        'bias_pedestal_sigma_dn': 2.0,
        'bias_row_gradient_dn': 1.0,
        'bias_col_gradient_dn': 0.5,
        'bloom_length': 4,  # 5.2 interim (no antiblooming; column bleed above the well)
        'quantization': 'exact',
        # Per-mode shape defaults (interim, 5.2); ~6 e- WAC 4 Hz banding.
        'artifact_modes': {
            'banding_coherent': {
                'amplitude_e': 6.0,
                'period_px': 64.0,
                'orientation': 'horizontal',
            },
            'bias_structure': {
                'pedestal_sigma_dn': 2.0,
                'row_gradient_dn': 1.0,
                'col_gradient_dn': 0.5,
            },
            'dark_ramp': {'kind': 'dark_gradient', 'amplitude_e': 150.0},
            'bloom': {'bloom_length': 4},
            'bright_dark_pairs': {'amplitude_e': 3.5e3},
            'radiation_transients': {'amplitude_e': 3.5e4},
            'fixed_pattern': {'dust_donut_count': 5},
        },
    },
    'gossi': {
        'detector_model': 'ccd',
        'full_well_e': 108.0e3,  # 5.4 interim
        'exposure_ref_sec': 0.2,  # 5.4 interim (typical science exposure)
        # 5.4 interim gain states ~1822/377/187/39 e-/DN; common science state 2.
        'gain_e_per_dn_by_state': {0: 1822.0, 1: 377.0, 2: 187.0, 3: 39.0},
        'default_gain_state': 2,
        'read_noise_e': 31.0,  # 5.4 (full-res; 44 e- in summation mode)
        'bias_dn': 20.0,
        'dark_current_e_per_sec': 10.0,  # 5.4 interim (RTG-driven dark spikes)
        # TUNED 2026-07-17 by the realism match: the 8-frame Galileo REDR
        # cohort's median measured spike fraction is 1.2e-4 with a
        # stationary component of only 1e-6 -- the REDR blemish correction
        # removes most hot pixels -- so the interim 3e-3 overstated the
        # products the navigator actually sees by ~25x.
        'hot_pixel_fraction': 1.0e-4,
        'hot_pixel_amplitude_e': 4.0e4,
        # 5.4 interim (early-blooming columns; total-charge fraction).
        'hot_pixel_column_factor': 0.5,
        # RETAINED zero 2026-07-18 after an adoption attempt.  The realism
        # match measures a real transient spike fraction of 1.17e-4 per
        # frame (8-frame REDR cohort, all negative cases, so no scene
        # point sources contaminate the split) -- but that population is
        # already carried by the tuned hot_pixel_fraction above, which was
        # sized to the cohort's TOTAL single-pixel incidence (1.2e-4;
        # the truly stationary component is only 1e-6) and which the FOM 6
        # split itself counts as transient because sim hot pixels reseed
        # per scene.  Adding a separate cosmic-ray term double-counts the
        # same measured spikes: two rates were tried 2026-07-18 (1.6e-3
        # from the raw split, then 5.4e-4 refit excluding star frames);
        # even the refit raised the simulated fraction to 2.3e-4 against
        # the 1.2e-4 real.
        # Unblocked by a fixed-position per-detector hot-pixel map, which
        # would free the transient budget for a real radiation term.
        'cosmic_ray_rate_per_sec': 0.0,
        'banding_amplitude_e': 65.0,  # 5.4 interim (~0.35 DN at gain 2 -> ~65 e-)
        'banding_period_px': 42.0,  # 5.4 (2400 Hz supply-noise comb every 42 px)
        'bias_pedestal_sigma_dn': 1.0,
        'bias_row_gradient_dn': 1.0,  # 5.4 (summation-mode L-R shading ramp)
        'bias_col_gradient_dn': 0.5,
        'bloom_length': 6,  # 5.4 interim (early-blooming columns)
        'quantization': 'exact',
        # Per-mode shape defaults (interim, 5.4).
        'artifact_modes': {
            # 42-px vertical supply-noise comb (~0.35 DN at gain 2 -> ~65 e-),
            # plus a <8 Hz horizontal component in high gain (orientation both).
            'banding_coherent': {'amplitude_e': 65.0, 'period_px': 42.0, 'orientation': 'both'},
            'bias_structure': {
                'pedestal_sigma_dn': 1.0,
                'row_gradient_dn': 1.0,
                'col_gradient_dn': 0.5,
            },
            # Shutter line-dependent exposure offset (~1.5 -> ~1.05 ms, line 1->800).
            'dark_ramp': {'kind': 'exposure_shading', 'top_factor': 1.5, 'bottom_factor': 1.05},
            'bloom': {'bloom_length': 6},
            # Ganymede-distance regime: ~1e4 spikes/frame scale in the readout,
            # amplitudes few DN steeply falling (near full well at the top).
            'radiation_transients': {'amplitude_e': 4.0e4},
            # 33-px photolithography stitch comb plus corner vignetting; the
            # 8-bit ADC contours worst at DN multiples of 8.
            'fixed_pattern': {
                'stitch_period_px': 33,
                'stitch_amplitude_dn': 1.0,
                'vignetting_frac': 0.03,
                'dust_donut_count': 4,
            },
            'contouring_8bit': {'step': 8},
            # HMA/HCA vertical decimation (5.4): only every Nth line carries
            # valid data, so the Galileo periodic-line default is 'keep'.
            'alternating_lines': {'mode': 'keep'},
        },
    },
    'nhlorri': {
        'detector_model': 'ccd',
        # 5.5: ADC-limited full well 4095 DN x 21 e-/DN.
        'full_well_e': 86.0e3,
        'exposure_ref_sec': 0.1,  # 5.5 interim (typical encounter exposure)
        # 5.5: single gain state ~21 e-/DN (1x1); 4x4 binning reads ~19.4.
        'gain_e_per_dn_by_state': {0: 21.0},
        'default_gain_state': 0,
        'read_noise_e': 23.0,  # 5.5 (~1.1 DN)
        'bias_dn': 545.0,  # 5.5 (~545 DN estimated per-image from dark columns)
        'dark_current_e_per_sec': 0.04,  # 5.5 (negligible dark current)
        # 5.5: LORRI has NO hot pixels (PDS maps are zeroes); disabled.
        'hot_pixel_fraction': 0.0,
        'hot_pixel_amplitude_e': 0.0,
        'hot_pixel_column_factor': 0.0,
        # RETAINED zero 2026-07-17: the realism match measured a 3.36e-4
        # per-frame transient spike fraction (two 5 s frames), but both
        # cohort frames are 4x4-binned star fields and a binned faint star
        # is indistinguishable from a transient to the single-pixel spike
        # detector -- the cohort has no star-free frame to bound that
        # contamination, so the measured rate is not adoptable.  Unblocked
        # by any non-star LORRI frame entering the library; scenes can
        # still plant the radiation_transients mode explicitly.
        'cosmic_ray_rate_per_sec': 0.0,
        'banding_amplitude_e': 17.0,  # 5.5 interim (~0.8 DN vertical banding)
        'banding_period_px': 128.0,  # 5.5 interim
        'bias_pedestal_sigma_dn': 1.0,
        'bias_row_gradient_dn': 0.5,
        'bias_col_gradient_dn': 0.5,
        'bloom_length': 2,  # 5.5 interim (short column bleed)
        'quantization': 'exact',
        # Per-mode shape defaults (interim, 5.5).  frame_transfer_smear is the
        # one artifact mode LORRI turns on under instrument_defaults (the defining
        # LORRI artifact, per 15.7): the detector resolver injects it at these
        # nominal scrub/transfer times when instrument_defaults is on and the
        # scene has not overridden it.  Every other mode stays off until a scene
        # sets its incidence.
        'artifact_modes': {
            # ~12 ms pre-exposure scrub, ~11 ms post-exposure transfer.
            'frame_transfer_smear': {'t_scrub_sec': 0.012, 't_transfer_sec': 0.011},
            # Saturated compact sources undershoot up to ~12 DN along readout.
            'serial_tail': {'amplitude_dn': 12.0, 'length_px': 8, 'direction': 'right'},
            # <=1 DN horizontal striping plus ~0.8 DN vertical banding near low
            # columns (~17 e-); orientation both approximates the two families.
            'banding_coherent': {'amplitude_e': 17.0, 'period_px': 128.0, 'orientation': 'both'},
            # ~4% corner vignetting, 0.9% PRNU, <=0.5 DN even/odd jail bars,
            # ~1% dust donuts.
            'fixed_pattern': {
                'vignetting_frac': 0.04,
                'prnu_rms': 0.009,
                'jail_bar_dn': 0.5,
                'dust_donut_count': 3,
            },
            # ~16 hits per readout-dominated short exposure; mostly single px.
            'radiation_transients': {'amplitude_e': 8.6e4},
        },
    },
    'vgiss': {
        'detector_model': 'vidicon',
        # The vidicon skips the electron conversion; signal maps straight to the
        # 8-bit DN full scale, then the DN-domain vidicon noise model applies.
        'exposure_ref_sec': 1.0,
        'bias_dn': 20.0,
        'quantization': '8bit',
        # Cosmic rays RETAINED at zero 2026-07-17 (no key: the vidicon DN
        # path carries no cosmic-ray stage, so a rate here would be inert).
        # The realism match measured a 3.5e-5 transient spike fraction over
        # the 3-frame GEOMED cohort -- GEOMED's reseau-anchored resampling
        # interpolates most single-pixel transients away.  Unblocked by a
        # DN-domain transient stage plus more Voyager frames.
        'vidicon': {
            # TUNED 2026-07-17 by the realism match: the 3-frame GEOMED
            # cohort's paired-difference sky sigma is ~0.22 DN equivalent
            # (median 1.75e-3 I/F), placing the products in the low-gain
            # regime of the published 0.3-0.75 DN range after GEOMED
            # resampling smoothing.  The interim 1.8 DN values (the
            # high-gain end) overstated the cohort's measured 0.22 DN sky
            # sigma about 9x.  0.25 DN per
            # component (quadrature sum ~0.35 DN, the low end of the
            # published range): the sim's 8-bit quantization floors any
            # sub-LSB value, so the choice is anchored to the published
            # low-gain range rather than to the (unreachable) 0.22 DN
            # cohort estimate -- see the realism report's known-gaps list.
            # 3 frames is limited evidence; revisit as VGISS frames land.
            'read_noise_line_dn': 0.25,
            'read_noise_pixel_dn': 0.25,
            # 5.3: faint coherent periodic component (2.4 kHz vertical, ~0.5 DN
            # peak-to-peak).
            'coherent_amplitude_dn': 0.25,
            'coherent_period_px': 8.0,
        },
        # Per-mode shape defaults (interim, 5.3), all GEOMED-level: the beam-bend
        # limb bias that survives reseau-anchored correction, the shortened
        # erase-cycle residual image, the readout dark-current line ramp, the
        # reseau-removal scars on the ~46-px lattice, and the GEOMED resample
        # texture (blank border + missing-line interpolation banding).
        'artifact_modes': {
            'beam_bend': {'amplitude_px': 1.0},
            'residual_image': {'amplitude': 0.05, 'prior': 'self_offset', 'offset_px': [5, 5]},
            # 48 x n s readout ramp, nonlinear in wait time.
            'dark_ramp': {'kind': 'dark_gradient', 'amplitude_e': 6.0, 'nonlinear': 1.5},
            'contouring_8bit': {'step': 8},
            'reseau_scars': {'spacing_px': 46, 'patch_radius_px': 4},
            'resample_texture': {
                'warp_amp_px': 0.3,
                'blank_border_px': 2,
                'missing_line_interp': False,
            },
        },
    },
    # The instrument-agnostic 'generic' / 'sim' block: an ideal 12-bit detector
    # whose electron well equals its DN depth at unit gain, so a generic scene's
    # electron chain reproduces the direct signal-to-DN mapping (electrons == DN)
    # rather than imposing a specific camera's radiometry.
    'generic': {
        'detector_model': 'ccd',
        'full_well_e': 4095.0,
        'exposure_ref_sec': 1.0,
        'gain_e_per_dn_by_state': {0: 1.0},
        'default_gain_state': 0,
        'read_noise_e': 1.0,
        'bias_dn': 20.0,
        'dark_current_e_per_sec': 0.0,
        'hot_pixel_fraction': 0.0,
        'hot_pixel_amplitude_e': 0.0,
        'hot_pixel_column_factor': 0.0,
        # The generic block is an ideal detector with no environment, so
        # there is no cosmic-ray flux to measure; explicitly zero.
        'cosmic_ray_rate_per_sec': 0.0,
        'banding_amplitude_e': 0.0,
        'banding_period_px': 64.0,
        'bias_pedestal_sigma_dn': 0.0,
        'bias_row_gradient_dn': 0.0,
        'bias_col_gradient_dn': 0.0,
        'bloom_length': 0,
        'quantization': 'exact',
    },
}

# Calibrated Cassini products (CALIB, what the navigator's cassini_iss_calib
# path consumes) run the same cameras through CISSCAL, whose bias/dark
# subtraction, flat-fielding, and 2-Hz noise removal strip most of the raw
# chain's structure -- so the calibrated instruments get their own detector
# entries instead of the raw alias.  TUNED 2026-07-17 by the realism match
# against the 62-frame CALIB image-library cohort; every key not overridden
# is inherited from the matching raw entry:
# - hot_pixel_fraction: measured stationary spike fraction 1.6e-5
#   (single-pixel spikes recurring at fixed positions across the 58-frame
#   NAC cohort; the interim raw-chain 2e-3 overstated the calibrated
#   products by ~100x and dominated their p99 signal percentile).
# - banding_amplitude_e 0: CISSCAL's 2-Hz noise removal strips the coherent
#   banding, and the cohort sky floor sits below the level at which the raw
#   chain's 30 e- banding would register.
# - bias_pedestal_sigma_dn / bias_row_gradient_dn / bias_col_gradient_dn 0:
#   CISSCAL subtracts the full 2-D bias structure, not just the constant
#   pedestal the calibration inverse already removes.
_COISS_CALIB_OVERRIDES: dict[str, Any] = {
    'hot_pixel_fraction': 1.6e-5,
    'banding_amplitude_e': 0.0,
    'bias_pedestal_sigma_dn': 0.0,
    'bias_row_gradient_dn': 0.0,
    'bias_col_gradient_dn': 0.0,
}
DETECTOR_DEFAULTS['coiss_calib_nac'] = {
    **copy.deepcopy(DETECTOR_DEFAULTS['coiss_nac']),
    **_COISS_CALIB_OVERRIDES,
}
DETECTOR_DEFAULTS['coiss_calib_wac'] = {
    **copy.deepcopy(DETECTOR_DEFAULTS['coiss_wac']),
    **_COISS_CALIB_OVERRIDES,
}

_coiss_alias(DETECTOR_DEFAULTS)


def resolve_detector_defaults(instrument: str | None) -> dict[str, Any]:
    """Return the detector-parameter defaults for a sim instrument.

    Parameters:
        instrument: The sim instrument name (see
            ``spindoctor.sim.instruments.SIM_INSTRUMENTS``), one of the generic
            aliases, or ``None`` for the instrument-agnostic block.

    Returns:
        A fresh copy of the instrument's ``DETECTOR_DEFAULTS`` entry, falling
        back to the generic block for the generic aliases or an unknown name.
    """
    key = 'generic' if instrument is None or instrument in ('generic', 'sim') else instrument
    entry = DETECTOR_DEFAULTS.get(key, DETECTOR_DEFAULTS['generic'])
    return copy.deepcopy(entry)


def resolve_mode_with_catalog(
    mode_name: str, scene_cfg: Mapping[str, Any], instrument: str | None
) -> dict[str, Any]:
    """Resolve an artifact mode's parameters with the per-instrument catalog.

    The resolution precedence for a mode's shape parameters is scene value, then
    the instrument's catalog default block (``artifact_modes`` in
    ``DETECTOR_DEFAULTS``), then the registry default.  ``incidence`` is never
    read from the catalog: a mode activates only when a scene sets its incidence
    (or, for the one physical-signal-chain member LORRI turns on, when the
    detector resolver injects it under ``instrument_defaults``).

    Parameters:
        mode_name: A registered artifact-mode name.
        scene_cfg: The scene's map for the mode (already validated).
        instrument: The sim instrument name, for the catalog lookup.

    Returns:
        A fresh dict carrying every parameter at its resolved value.
    """
    catalog_modes = resolve_detector_defaults(instrument).get('artifact_modes') or {}
    catalog = catalog_modes.get(mode_name) or {}
    resolved: dict[str, Any] = {}
    for param in ARTIFACT_MODES[mode_name].params:
        name = param.name
        if name in scene_cfg and scene_cfg[name] is not None:
            resolved[name] = scene_cfg[name]
        elif name != 'incidence' and name in catalog:
            resolved[name] = catalog[name]
        else:
            resolved[name] = param.default
    return resolved
