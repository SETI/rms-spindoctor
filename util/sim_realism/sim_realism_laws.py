"""The photometric-law study of the realism match.

A simulated body's brightness across its disc comes from its photometric law.  This
module checks the law against the real frames of the cohort, and compares it with every
other law the simulator offers, so the choice of the default law rests on the record.

**The real discs.**  :mod:`util.sim_realism.sim_realism_laws_discs` measures every moon
in each calibrated real frame and keeps, with each disc, the pixels its fits counted.

**The twins.**  For each real disc, :func:`run_law_study` renders a twin under every law
in :data:`~spindoctor.sim.render.bodies.photometry.PHOTOMETRIC_LAWS`.  The twin is a
sphere at the moon's apparent size and position in the real frame, lit from the real
direction at the real phase, through the real frame's camera settings and exposure.  Its
normal albedo is measured from the real disc against the twin's own law
(:func:`~util.sim_realism.albedo.disc_normal_albedo`).  Minnaert's law needs an
exponent, and the twin takes it from a straight line fitted to the real discs' own
exponents against phase (:func:`minnaert_twin_k`), so that law is the best one
Minnaert's form allows.

**The pairing.**  The real disc's fits count at most 60,000 of its pixels, and every
twin's fit counts the same pixels, less any the twins' sphere geometry leaves out and
any that any one of the twins reads at its camera's ceiling (:func:`twin_ceiling_if`).
The real disc is fitted again over the pixels they all counted, and every twin is fitted
over the pixels that fit keeps, without dropping any of its own, so a marking the real
fit drops is dropped from every twin's fit.  So a real disc that lost pixels to its
terrain cap, a nearer body, the rings, a shadow, or clipping has twins fitted over the
same part of the disc, and the laws are compared over the same pixels.  A disc that
does not give a normal albedo under every law is left out.  A twin whose disc is
clipped as a real disc would be (at least 1% of its interior at the ceiling) does not
give a limb gradient peak, because its limb rises to a plateau.

**The statistics.**  For each disc and law: the twin's exponent less the real one over
the paired pixels, the residual between the two equator profiles' shapes over those
pixels, and, where the realism match measured the real frame's limb gradient peak, the
twin's limb gradient peak.  Each is reported by phase bin and by body.

**The disc-profile check.**  For the default law, :func:`compare_disc_profiles` judges
each phase bin with at least 3 discs.  Its statistic is the median of the paired
differences, each twin's exponent less its own real disc's, and its band is 2 standard
errors of those differences' robust scatter, never less than 0.02.  The band is
therefore set by the same frames it judges: it says how far the median may sit from
zero given how much the pairs scatter, and it widens when the twins follow the real
discs less evenly.

**The law comparison.**  :func:`law_comparison` asks, in each phase bin with at least 3
discs that have a twin exponent under every law, whether the default law's exponents
agree with the real ones at least as well as every other law's.  It needs at least two
laws.  Two laws separate in a bin when their median absolute differences differ by more
than the bin's band.  The band is the disc-profile band of all of the default law's
pairs in that bin, over every instrument, including the discs that lack an exponent
under another law.  Where two laws do not separate, no order between them is claimed.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.ndimage
from util.sim_realism.albedo import disc_normal_albedo
from util.sim_realism.disc_profile import (
    LunarLambertFit,
    MinnaertFit,
    equator_profile,
    fit_minnaert_paired,
    profile_residual,
    select_disc_pixels,
)
from util.sim_realism.sim_realism_body import EDGE_CLEARANCE_PX, MIN_CLIPPED_FRACTION
from util.sim_realism.sim_realism_limb_peak import (
    LIMB_PEAK_KIND_PREFIX,
    LimbPeakBin,
    LimbSelection,
    compare_limb_peaks,
    matched_orientation,
    matched_silhouette_offsets,
)
from util.sim_realism.sim_realism_scenes import MATCHED_BODY_LAW, matched_scene
from util.sim_realism.sim_realism_support import (
    PHASE_BIN_EDGES_DEG,
    extract_feature_samples,
    prepare_frame_features,
)

from spindoctor.feature.geometry import LimbPolyline
from spindoctor.obs.obs_inst_sim import ObsSim
from spindoctor.sim.render.bodies.photometry import PHOTOMETRIC_LAWS, incidence_cosines
from spindoctor.sim.render.detector.params import calibration_map, resolve_detector_params
from spindoctor.sim.render.detector.quantization import quantize_dn
from spindoctor.sim.scene.validate import validate_sim_params
from spindoctor.support.memory import release_transient_memory
from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType

__all__ = [
    'DEFAULT_LAW',
    'STUDY_LAWS',
    'DiscProfileBin',
    'DiscRow',
    'ExcludedDisc',
    'LawBin',
    'LawComparisonBin',
    'LawStudy',
    'LawStudyInputs',
    'RealDisc',
    'StudyFrame',
    'TwinDisc',
    'compare_disc_profiles',
    'crop_image',
    'law_comparison',
    'minnaert_line',
    'minnaert_twin_k',
    'phase_bin',
    'run_law_study',
    'twin_ceiling_if',
]

STUDY_LAWS: tuple[str, ...] = tuple(sorted(PHOTOMETRIC_LAWS))
"""Every law the study renders twins under."""

DEFAULT_LAW = MATCHED_BODY_LAW
"""The law the disc-profile check judges: the matched bodies' law, the scene default."""

# The disc-profile band rule, in units of Minnaert's exponent: 2 standard errors of the
# robust scatter of the bin's paired differences, a floor, and a verdict only for a bin
# with at least 3 discs.
DISC_BAND_MULTIPLE = 2.0
DISC_BAND_FLOOR = 0.02
MIN_BIN_DISCS = 3

# Bounds on the exponent the Minnaert twins take from the fitted line.
_MINNAERT_K_RANGE = (0.2, 1.5)

# The contact-sheet crops are block-averaged to at most this many pixels on a side.
_CROP_PX = 160

_MAD_TO_SIGMA = 1.4826

# A twin pixel at its camera's ceiling reads a little below it after the noise the chain
# adds once it clips; this many standard deviations of each noise are allowed for.
_CEILING_NOISE_SIGMAS = 4.0


@dataclass(frozen=True)
class StudyFrame:
    """The facts about one real frame a twin renders through.

    Parameters:
        image_id: The real frame's image ID.
        instrument: The sim instrument it is matched against.
        offset_vu: The frame's verified offset.
        exposure_sec: Its exposure.
        frame_shape_vu: Its (v, u) shape.
        filter_name: Its filter combination, or None.
        gain_state: Its gain state, or None.
        quantization: Its downlink quantization, or None.
    """

    image_id: str
    instrument: str
    offset_vu: tuple[float, float]
    exposure_sec: float
    frame_shape_vu: tuple[int, int]
    filter_name: str | None = None
    gain_state: int | None = None
    quantization: str | None = None


@dataclass
class RealDisc:
    """One real moon's disc, or one terrain of it, measured for the study.

    Parameters:
        frame: The real frame.
        body: The moon, as SPICE knows it.
        part: The terrain, such as ``'dark terrain'``, of a two-tone moon measured one
            terrain at a time, else None.
        phase_deg: The phase angle at the disc's counted pixels, in degrees.
        diameter_px: The moon's apparent diameter.
        center_vu: The moon's center in the image, as a pixel-corner position.
        illumination_deg: The in-image direction the light comes from, as a scene's
            ``illumination_angle`` gives it.
        intensity: The counted pixels' I/F, at most 60,000 of them.
        mu0: Their cosines of incidence.
        mu: Their cosines of emission.
        minnaert: Minnaert's law fitted to them.
        lunar_lambert: The lunar-Lambert blend fitted to them, or None.
        profile: Their equator profile.
        crop: The disc's image over ``crop_bounds``, block-averaged for the contact
            sheet.
        crop_bounds: The crop's ``(v0, v1, u0, u1)`` bounds: the moon's silhouette with
            a margin.  Its twins are cropped to the same bounds.
        selection: Which of the frame's pixels the fits counted: lit pixels clear of the
            limb and the terminator, inside the terrain cap for a part, neither clipped
            nor hidden, and thinned to at most 60,000.  ``intensity``, ``mu0``, and
            ``mu`` hold these pixels in row-major order.
        limb_peak: The realism match's limb gradient-peak sample for the frame, as
            ``(kind, offset)``, when this moon is the one the match measured.
        limb_directions_deg: The directions of the real limb's vertices whose peaks
            the fit kept, which its twins' limbs are measured over (see
            :class:`~util.sim_realism.sim_realism_limb_peak.LimbOrientation`).
    """

    frame: StudyFrame
    body: str
    part: str | None
    phase_deg: float
    diameter_px: float
    center_vu: tuple[float, float]
    illumination_deg: float
    intensity: NDArrayFloatType
    mu0: NDArrayFloatType
    mu: NDArrayFloatType
    minnaert: MinnaertFit
    lunar_lambert: LunarLambertFit | None
    profile: NDArrayFloatType
    crop: NDArrayFloatType
    crop_bounds: tuple[int, int, int, int]
    selection: NDArrayBoolType
    limb_peak: tuple[str, float] | None = None
    limb_directions_deg: tuple[float, ...] = ()

    @property
    def label(self) -> str:
        """The moon, with its terrain when it has one, in title case."""
        name = self.body.title()
        return name if self.part is None else f'{name} ({self.part})'


@dataclass(frozen=True)
class ExcludedDisc:
    """A moon in view that the study does not measure, and why.

    Parameters:
        image_id: The real frame's image ID.
        body: The moon.
        reason: Why it is left out.
    """

    image_id: str
    body: str
    reason: str


@dataclass
class LawStudyInputs:
    """The real discs the realism match collects for the study, frame by frame.

    Parameters:
        discs: The discs measured.
        excluded: The moons in view that were not.
    """

    discs: list[RealDisc] = field(default_factory=list)
    excluded: list[ExcludedDisc] = field(default_factory=list)

    def add_limb_peak(
        self,
        image_id: str,
        body: str | None,
        samples: Mapping[str, Any],
        *,
        directions_deg: tuple[float, ...] = (),
    ) -> None:
        """Attach a frame's limb gradient-peak sample to the disc it was measured on.

        The realism match calls this with the real frame's own samples, before it pairs
        them with the matched frame's, so every limb peak measured on a real frame enters
        the study whether or not its matched frame measured one.

        Parameters:
            image_id: The real frame's image ID.
            body: The body the realism match measured the limb of, or None.
            samples: The real frame's samples; only ``limb_peak_*`` kinds are read.
            directions_deg: The directions of the vertices the real fit kept.
        """
        kinds = [k for k in samples if k.startswith(LIMB_PEAK_KIND_PREFIX)]
        if body is None or len(kinds) != 1 or len(samples[kinds[0]]) != 1:
            return
        for disc in self.discs:
            if disc.frame.image_id == image_id and disc.body == body and disc.part is None:
                disc.limb_peak = (kinds[0], float(samples[kinds[0]][0]))
                disc.limb_directions_deg = directions_deg


@dataclass(frozen=True)
class TwinDisc:
    """One twin of a real disc under one law.

    Parameters:
        law: The twin's law.
        normal_albedo: The albedo measured from the real disc against that law.
        minnaert_k: The exponent of a Minnaert twin, else None.
        k: Minnaert's exponent fitted to the twin's counted pixels, or None.
        real_k: Minnaert's exponent fitted to the real disc over the same pixels, or
            None; every twin of the disc shares it.  It is the real disc's own exponent
            when the twins counted every pixel the real fits did.
        profile_residual: The residual between the real and the twin's equator
            profiles over those pixels, or None.
        limb_peak: The twin's limb gradient-peak offset, when the real frame's was
            measured and the twin's could be.
        limb_peak_refusal: Why the twin's limb gave no offset when the real frame's
            did, else None.
        crop: The twin's image over the real disc's crop bounds, block-averaged the same
            way.
        clipped_px: How many of the real fits' pixels the twin reads at its camera's
            ceiling; they are left out of the fit of every twin of the disc.
    """

    law: str
    normal_albedo: float
    minnaert_k: float | None
    k: float | None
    real_k: float | None
    profile_residual: float | None
    limb_peak: float | None
    crop: NDArrayFloatType
    clipped_px: int
    limb_peak_refusal: str | None = None


@dataclass
class DiscRow:
    """One real disc and its twins.

    Parameters:
        disc: The real disc.
        twins: Its twin under each law, keyed by law.
    """

    disc: RealDisc
    twins: dict[str, TwinDisc]

    def delta_k(self, law: str) -> float | None:
        """The twin's exponent less the real one over the same pixels, under one law."""
        twin = self.twins.get(law)
        if twin is None or twin.k is None or twin.real_k is None:
            return None
        return twin.k - twin.real_k


@dataclass(frozen=True)
class DiscProfileBin:
    """One phase bin of the disc-profile check.

    Parameters:
        n: Discs in the bin.
        real_median_k: The real exponents' median, each over its pair's pixels.
        sim_median_k: The default-law twins' median.
        median_delta_k: The median of each twin's exponent less its real disc's.
        delta_scatter_k: The robust scatter of those paired differences, or None below
            3 discs.
        band_k: The band's half-width, or None below 3 discs.
        within_band: Whether the median difference lies within the band, or None
            without a band.
    """

    n: int
    real_median_k: float
    sim_median_k: float
    median_delta_k: float
    delta_scatter_k: float | None
    band_k: float | None
    within_band: bool | None


@dataclass(frozen=True)
class LawBin:
    """One law's statistics over one group of discs.

    Parameters:
        n: Discs in the group with a twin exponent.
        median_delta_k: The median of the twins' exponent less the real one.
        median_abs_delta_k: The median of its absolute value.
        median_profile_residual: The median equator-profile residual, or None.
    """

    n: int
    median_delta_k: float | None
    median_abs_delta_k: float | None
    median_profile_residual: float | None


@dataclass(frozen=True)
class LawComparisonBin:
    """Whether the default law agrees best in one phase bin.

    Parameters:
        n: Discs in the bin.
        band_k: The bin's band: the disc-profile band of the default law's paired
            differences in the bin (see :class:`DiscProfileBin`).
        agreement: Each law's median absolute exponent difference.
        separates: For each other law, whether its agreement differs from the
            default law's by more than the band.
        default_best: Whether the default law agrees at least as well as every law
            it separates from.
    """

    n: int
    band_k: float
    agreement: dict[str, float]
    separates: dict[str, bool]
    default_best: bool


@dataclass
class LawStudy:
    """The study's results.

    Parameters:
        rows: Each measured disc with its twins.
        excluded: The moons in view that were not measured.
        minnaert_k0: The fitted line's exponent at zero phase.
        minnaert_k1: Its slope per degree of phase.
        law_bins: Per law, per phase bin, the statistics.
        body_bins: Per law, per body label, the statistics.
        limb_peaks: Per law, per limb bin, the gradient-peak comparison.
        disc_check: Per instrument, per phase bin, the default law's check.
        comparison: Per phase bin with at least 3 discs, the law comparison.
    """

    rows: list[DiscRow]
    excluded: list[ExcludedDisc]
    minnaert_k0: float
    minnaert_k1: float
    law_bins: dict[str, dict[str, LawBin]]
    body_bins: dict[str, dict[str, LawBin]]
    limb_peaks: dict[str, dict[str, LimbPeakBin]]
    disc_check: dict[str, dict[str, DiscProfileBin]]
    comparison: dict[str, LawComparisonBin]


def phase_bin(phase_deg: float) -> str:
    """The phase-bin label (``'p0'``, ``'p1'``, ``'p2'``) of the limb strata."""
    index = int(np.searchsorted(np.asarray(PHASE_BIN_EDGES_DEG), phase_deg, side='right'))
    return f'p{index}'


def _sphere_cosines(
    shape_vu: tuple[int, int],
    center_vu: tuple[float, float],
    radius_px: float,
    *,
    illumination_deg: float,
    phase_deg: float,
) -> tuple[NDArrayBoolType, NDArrayFloatType, NDArrayFloatType]:
    """Which pixels a sphere covers and the two cosines at each pixel's center."""
    rows, cols = np.indices(shape_vu, dtype=np.float64)
    normal_v = (rows + 0.5 - center_vu[0]) / radius_px
    normal_u = (cols + 0.5 - center_vu[1]) / radius_px
    squared = 1.0 - normal_v**2 - normal_u**2
    on_body = squared > 0.0
    normal_z = np.sqrt(np.clip(squared, 0.0, None))
    mu0 = incidence_cosines(
        normal_v,
        normal_u,
        normal_z,
        illumination_angle=math.radians(illumination_deg),
        phase_angle=math.radians(phase_deg),
    )
    return on_body, np.where(on_body, mu0, -1.0), np.where(on_body, normal_z, -1.0)


def crop_image(image: NDArrayFloatType, box: tuple[int, int, int, int]) -> NDArrayFloatType:
    """A crop of an image, block-averaged to at most 160 px on a side.

    Parameters:
        image: The image.
        box: The crop's ``(v0, v1, u0, u1)`` bounds.

    Returns:
        The block-averaged crop.  Each block is the mean of its finite pixels, and a
        block that has none is NaN.
    """
    v0, v1, u0, u1 = box
    patch = np.asarray(image[v0:v1, u0:u1], dtype=np.float64)
    factor = max(1, math.ceil(max(patch.shape) / _CROP_PX))
    rows = patch.shape[0] // factor * factor
    cols = patch.shape[1] // factor * factor
    if rows == 0 or cols == 0:
        return patch.astype(np.float32)
    blocks = patch[:rows, :cols].reshape(rows // factor, factor, cols // factor, factor)
    finite = np.isfinite(blocks)
    counts = np.sum(finite, axis=(1, 3))
    sums = np.sum(np.where(finite, blocks, 0.0), axis=(1, 3))
    means = np.full(counts.shape, np.nan)
    np.divide(sums, counts, out=means, where=counts > 0)
    return means.astype(np.float32)


def minnaert_line(discs: Iterable[RealDisc]) -> tuple[float, float]:
    """A straight line through the real discs' Minnaert exponents against phase.

    Parameters:
        discs: The real discs.

    Returns:
        The exponent at zero phase and the slope per degree, from an unweighted least
        squares fit; ``(0.5, 0.0)`` when fewer than 2 discs are given; and the mean
        exponent with a slope of 0 when every disc lies at the same phase, which cannot
        set a slope.
    """
    pairs = [(d.phase_deg, d.minnaert.k) for d in discs]
    phases = np.array([p for p, _ in pairs], dtype=np.float64)
    ks = np.array([k for _, k in pairs], dtype=np.float64)
    if phases.size < 2:
        return 0.5, 0.0
    if np.all(phases == phases[0]):
        return float(np.mean(ks)), 0.0
    slope, intercept = np.polyfit(phases, ks, 1)
    return float(intercept), float(slope)


def minnaert_twin_k(k0: float, k1: float, phase_deg: float) -> float:
    """The exponent a Minnaert twin takes at one phase.

    Parameters:
        k0: The line's exponent at zero phase (:func:`minnaert_line`).
        k1: Its slope per degree of phase.
        phase_deg: The disc's phase angle in degrees.

    Returns:
        The line's value at the phase, held to the range 0.2 to 1.5.
    """
    return min(max(k0 + k1 * phase_deg, _MINNAERT_K_RANGE[0]), _MINNAERT_K_RANGE[1])


def twin_ceiling_if(scene: Mapping[str, Any]) -> float:
    """The I/F at and above which a twin's pixel reads its camera's ceiling.

    A pixel clips where its charge fills the full well or its count reaches the ADC
    limit, whichever comes first.  The scene's detector settings, and the instrument
    entry behind them, set both.  The read noise and the quantization then act on the
    clipped count, and calibration takes off the readout dark line and the residual
    structure, so a clipped pixel's I/F lies a little below the ceiling's own.  The level
    allows 4 standard deviations of the read noise and of the residual structure, and
    the readout dark of the frame's last line, the most that any line carries.

    Parameters:
        scene: The twin's scene.

    Returns:
        The level in I/F, or infinity for a frame left in DN.
    """
    dp = resolve_detector_params(scene)
    calibration = calibration_map(dp)
    if calibration is None:
        return math.inf
    b = dp.binning
    readout_dark = dp.detector_modes.get('readout_dark') or {}
    if dp.detector_model == 'vidicon':
        well_dn = dp.full_well_dn
        read_dn = math.hypot(
            dp.vidicon.get('read_noise_line_dn', 0.0), dp.vidicon.get('read_noise_pixel_dn', 0.0)
        )
        dark_dn = 0.0
    else:
        well_dn = dp.full_well_e * b**2 / dp.gain_e_per_dn
        read_dn = dp.read_noise_e / dp.gain_e_per_dn
        dark_dn = float(readout_dark.get('amplitude') or 0.0) * b**2 / dp.gain_e_per_dn
    # The column residual adds over a binned pixel's b rows and b columns, the pixel
    # residual over its b**2 native pixels.
    residual_dn = (
        math.hypot(dp.calibration_column_residual_e * b**1.5, dp.calibration_pixel_residual_e * b)
        / dp.gain_e_per_dn
    )
    ceiling_dn = min(dp.saturation_dn, well_dn + dp.bias_dn)
    lowest_read = quantize_dn(
        np.array([ceiling_dn - _CEILING_NOISE_SIGMAS * read_dn]),
        mode=dp.quantization,
        saturation_dn=dp.saturation_dn,
        contour_step=dp.quantization_contour_step,
        sqrt_lut=dp.sqrt_lut,
    )
    lowest_dn = min(float(lowest_read[0]), dp.saturation_dn)
    lowest_dn -= _CEILING_NOISE_SIGMAS * residual_dn + dark_dn
    return float(calibration.to_if(np.array([lowest_dn]))[0])


def _disc_clipped(on_body: NDArrayBoolType, at_ceiling: NDArrayBoolType) -> bool:
    """Whether a twin's disc is clipped by the test a real disc meets."""
    interior = scipy.ndimage.binary_erosion(on_body, iterations=EDGE_CLEARANCE_PX)
    return bool(interior.any()) and float(np.mean(at_ceiling[interior])) >= MIN_CLIPPED_FRACTION


@dataclass(frozen=True)
class _TwinRender:
    """One law's rendered twin of a real disc, before its fit.

    Parameters:
        law: The twin's law.
        normal_albedo: The albedo measured from the real disc against that law.
        data: The twin's image.
        at_ceiling: Which of its pixels read at its camera's ceiling.
        limb_peak: Its limb gradient-peak offset, or None.
        limb_peak_refusal: Why its limb gave no offset when one was asked for, else None.
    """

    law: str
    normal_albedo: float
    data: NDArrayFloatType
    at_ceiling: NDArrayBoolType
    limb_peak: float | None
    limb_peak_refusal: str | None


def _twins(
    disc: RealDisc, laws: Iterable[str], *, minnaert_k: float, limb_peak: bool
) -> dict[str, TwinDisc] | None:
    """Render one real disc's twin under each law and fit every one over the same pixels.

    The twins and the real disc are all fitted over the real fits' own pixels, less any
    the twins' sphere leaves out and any that any of the twins reads at its camera's
    ceiling.  Every twin is fitted over the pixels the real disc's fit of them keeps
    (:func:`fit_minnaert_paired`), so a marking that fit drops is left out of every
    twin's fit as well.  So every law is compared with
    the real disc over the same part of it, and a law whose twin clips the pixels it
    matches worst does not shed them from its fit.

    Parameters:
        disc: The real disc.
        laws: The laws to render twins under.
        minnaert_k: The exponent of a Minnaert twin.
        limb_peak: Whether to measure each twin's limb gradient peak.

    Returns:
        Each law's twin, keyed by law.  None, before any twin is rendered, when the real
        disc does not give a normal albedo under one of the laws, so a twin could not
        match its brightness.
    """
    phase_rad = math.radians(disc.phase_deg)
    albedos = {
        law: disc_normal_albedo(
            disc.intensity, disc.mu0, disc.mu, law=law, phase_angle=phase_rad, minnaert_k=minnaert_k
        )
        for law in laws
    }
    radius = disc.diameter_px / 2.0
    on_body, mu0, mu = _sphere_cosines(
        disc.frame.frame_shape_vu,
        disc.center_vu,
        radius,
        illumination_deg=disc.illumination_deg,
        phase_deg=disc.phase_deg,
    )
    renders = []
    for law, albedo in albedos.items():
        if albedo is None:
            return None
        renders.append(
            _render_twin(
                disc,
                law,
                normal_albedo=albedo,
                minnaert_k=minnaert_k,
                limb_peak=limb_peak,
                on_body=on_body,
            )
        )
        release_transient_memory()
    usable = np.array(disc.selection, dtype=bool)
    for render in renders:
        usable &= np.isfinite(render.data) & (render.data > 0.0) & ~render.at_ceiling
    counted = select_disc_pixels(on_body, mu0, mu, usable=usable)
    shared = counted[disc.selection]
    real = (disc.intensity[shared], disc.mu0[shared], disc.mu[shared])
    real_profile = disc.profile if bool(shared.all()) else equator_profile(*real, phase_rad)
    real_fit, fits = fit_minnaert_paired(
        real, [(r.data[counted], mu0[counted], mu[counted]) for r in renders]
    )
    real_k = None if real_fit is None else real_fit.k
    twins: dict[str, TwinDisc] = {}
    for render, fit in zip(renders, fits, strict=True):
        values = render.data[counted]
        twin_profile = equator_profile(values, mu0[counted], mu[counted], phase_rad)
        twins[render.law] = TwinDisc(
            law=render.law,
            normal_albedo=render.normal_albedo,
            minnaert_k=minnaert_k if render.law == 'minnaert' else None,
            k=None if fit is None else fit.k,
            real_k=real_k,
            profile_residual=profile_residual(real_profile, twin_profile),
            limb_peak=render.limb_peak,
            crop=crop_image(render.data, disc.crop_bounds),
            clipped_px=int(np.count_nonzero(render.at_ceiling & disc.selection)),
            limb_peak_refusal=render.limb_peak_refusal,
        )
    return twins


def _render_twin(
    disc: RealDisc,
    law: str,
    *,
    normal_albedo: float,
    minnaert_k: float,
    limb_peak: bool,
    on_body: NDArrayBoolType,
) -> _TwinRender:
    """Render one real disc's twin under one law and measure its limb.

    Parameters:
        disc: The real disc.
        law: The twin's law.
        normal_albedo: The albedo measured from the real disc against that law.
        minnaert_k: The exponent of a Minnaert twin.
        limb_peak: Whether to measure the twin's limb gradient peak.
        on_body: The twin's sphere's pixels.

    Returns:
        The rendered twin.
    """
    frame = disc.frame
    scene = matched_scene(
        f'{frame.image_id}_{disc.body}',
        'body_full_fov',
        frame.instrument,
        frame.exposure_sec,
        size_vu=frame.frame_shape_vu,
        diameter_px=disc.diameter_px,
        phase_angle_deg=disc.phase_deg,
        normal_albedo=normal_albedo,
        filter_name=frame.filter_name,
        gain_state=frame.gain_state,
        quantization=frame.quantization,
    )
    body = scene['bodies'][0]
    radius = disc.diameter_px / 2.0
    body.update(
        center_v=disc.center_vu[0],
        center_u=disc.center_vu[1],
        semi_a_px=radius,
        semi_b_px=radius,
        semi_c_px=radius,
        illumination_angle=disc.illumination_deg,
        photometric_law=law,
    )
    if law == 'minnaert':
        body['minnaert_k'] = minnaert_k
    scene['stars'] = []
    validate_sim_params(scene)
    obs = ObsSim.from_file(f'/tmp/sim_realism_law_{frame.image_id}.yaml', sim_params=scene)
    data = np.asarray(obs.data, dtype=np.float64)
    at_ceiling = data >= twin_ceiling_if(scene)
    peak: float | None = None
    refusal: str | None = None
    if limb_peak and disc.limb_peak:
        if _disc_clipped(on_body, at_ceiling):
            refusal = "its disc is clipped at the camera's ceiling"
        else:
            peak, refusal = _twin_limb_peak(
                obs, scene, disc.limb_peak[0], directions_deg=disc.limb_directions_deg
            )
    return _TwinRender(
        law=law,
        normal_albedo=normal_albedo,
        data=data,
        at_ceiling=at_ceiling,
        limb_peak=peak,
        limb_peak_refusal=refusal,
    )


def _twin_limb_peak(
    obs: Any, scene: dict[str, Any], kind: str, *, directions_deg: tuple[float, ...]
) -> tuple[float | None, str | None]:
    """The twin's limb gradient-peak offset in the real frame's limb bin, and a refusal.

    The twin's limb is measured over the directions the real fit kept.  It gives no
    offset, with the reason, when its limb is refused, and when its offset falls in
    another limb bin than the real one.
    """
    image, features, model_metadata = prepare_frame_features(obs, only_models='body_sim:*')
    features = [f for f in features if isinstance(f.geometry, LimbPolyline)]
    model = f'body_sim:{scene["bodies"][0]["name"]}'
    frame = extract_feature_samples(
        image,
        features,
        model_metadata,
        offset_vu=(0.0, 0.0),
        extfov_margin_vu=obs.extfov_margin_vu,
        saturation_level=math.nan,
        matched_model=model,
        silhouette_offsets=matched_silhouette_offsets(scene, obs),
        limb_peak_model=model,
        limb_selection=LimbSelection(matched_orientation(scene), peak_deg=directions_deg),
    )
    if frame.limb_peak is None:
        return None, 'the navigator placed no limb'
    if frame.limb_peak.offset_px is None:
        return None, frame.limb_peak.refusal
    values = frame.samples.get(kind, [])
    if len(values) != 1:
        return None, 'its limb falls in another bin'
    return float(values[0]), None


def _robust_scatter(values: NDArrayFloatType) -> float:
    """The scaled median absolute deviation of ``values``."""
    return float(_MAD_TO_SIGMA * np.median(np.abs(values - np.median(values))))


def _band(delta_k: NDArrayFloatType) -> float:
    """The disc-profile band for a bin's paired differences (at least 3 of them)."""
    scatter = _robust_scatter(delta_k)
    return max(DISC_BAND_MULTIPLE * scatter / math.sqrt(delta_k.size), DISC_BAND_FLOOR)


def compare_disc_profiles(
    rows: Iterable[DiscRow], law: str = DEFAULT_LAW
) -> dict[str, DiscProfileBin]:
    """Judge each phase bin's twins under one law against the real exponents.

    Only a bin with at least 3 discs is judged.  Its statistic is the median of the
    paired differences, each twin's exponent less its real disc's over the same pixels
    (:meth:`DiscRow.delta_k`); the band's half-width
    is 2 standard errors of those differences' robust scatter, never less than 0.02; and
    the median must lie within it.  The band comes from the same pairs it judges.

    Parameters:
        rows: The discs, each with its twins.
        law: The law whose twins are judged.

    Returns:
        One entry per phase bin that holds a disc with a twin exponent under the law.
    """
    by_bin: dict[str, list[tuple[float, float]]] = {}
    for row in rows:
        twin = row.twins.get(law)
        if twin is None or twin.k is None or twin.real_k is None:
            continue
        by_bin.setdefault(phase_bin(row.disc.phase_deg), []).append((twin.real_k, twin.k))
    out: dict[str, DiscProfileBin] = {}
    for label, pairs in sorted(by_bin.items()):
        real_k = np.array([p[0] for p in pairs])
        sim_k = np.array([p[1] for p in pairs])
        deltas = sim_k - real_k
        delta = float(np.median(deltas))
        scatter = band = None
        within = None
        if deltas.size >= MIN_BIN_DISCS:
            scatter = _robust_scatter(deltas)
            band = _band(deltas)
            within = bool(abs(delta) <= band)
        out[label] = DiscProfileBin(
            n=int(real_k.size),
            real_median_k=float(np.median(real_k)),
            sim_median_k=float(np.median(sim_k)),
            median_delta_k=delta,
            delta_scatter_k=scatter,
            band_k=band,
            within_band=within,
        )
    return out


def _law_bin(rows: list[DiscRow], law: str) -> LawBin:
    """One law's statistics over a group of discs."""
    deltas = np.array([d for d in (r.delta_k(law) for r in rows) if d is not None])
    residuals = np.array(
        [
            t.profile_residual
            for t in (r.twins.get(law) for r in rows)
            if t is not None and t.profile_residual is not None
        ]
    )
    return LawBin(
        n=int(deltas.size),
        median_delta_k=float(np.median(deltas)) if deltas.size else None,
        median_abs_delta_k=float(np.median(np.abs(deltas))) if deltas.size else None,
        median_profile_residual=float(np.median(residuals)) if residuals.size else None,
    )


def law_comparison(
    rows: Iterable[DiscRow], *, laws: Iterable[str], default: str = DEFAULT_LAW
) -> dict[str, LawComparisonBin]:
    """Ask in each phase bin whether the default law agrees best with the real discs.

    Parameters:
        rows: The discs, each with its twins.
        laws: The laws to compare, the default among them.
        default: The default law.

    Returns:
        One entry per phase bin holding at least 3 discs whose twins have an exponent
        under every one of ``laws``.  The bin's band comes from every disc in it whose
        default-law twin has an exponent, so a disc missing another law's exponent still
        sets the band, though it does not enter the agreement.

    Raises:
        ValueError: If ``laws`` holds fewer than 2 laws or leaves out the default.
    """
    compared = sorted(set(laws))
    if len(compared) < 2 or default not in compared:
        raise ValueError(
            f'a law comparison needs at least 2 laws, the default {default!r} among them; '
            f'got {compared}'
        )
    by_bin: dict[str, list[DiscRow]] = {}
    default_deltas: dict[str, list[float]] = {}
    for row in rows:
        label = phase_bin(row.disc.phase_deg)
        delta = row.delta_k(default)
        if delta is not None:
            default_deltas.setdefault(label, []).append(delta)
        if all(row.delta_k(law) is not None for law in compared):
            by_bin.setdefault(label, []).append(row)
    out: dict[str, LawComparisonBin] = {}
    for label, group in sorted(by_bin.items()):
        if len(group) < MIN_BIN_DISCS:
            continue
        band = _band(np.array(default_deltas[label], dtype=np.float64))
        agreement = {
            law: float(np.median([abs(r.delta_k(law) or 0.0) for r in group])) for law in compared
        }
        separates = {
            law: bool(abs(agreement[law] - agreement[default]) > band)
            for law in compared
            if law != default
        }
        best = all(
            agreement[default] <= agreement[law] for law, apart in separates.items() if apart
        )
        out[label] = LawComparisonBin(
            n=len(group), band_k=band, agreement=agreement, separates=separates, default_best=best
        )
    return out


def run_law_study(inputs: LawStudyInputs, *, laws: tuple[str, ...] = STUDY_LAWS) -> LawStudy:
    """Render every real disc's twins and gather the study's statistics.

    Parameters:
        inputs: The real discs and the moons left out.
        laws: The laws to render twins under.

    Returns:
        The study.  Its law comparison is empty unless ``laws`` holds the default law
        and at least one other.
    """
    k0, k1 = minnaert_line(inputs.discs)
    rows: list[DiscRow] = []
    excluded = list(inputs.excluded)
    for disc in inputs.discs:
        k_line = minnaert_twin_k(k0, k1, disc.phase_deg)
        twins = _twins(disc, laws, minnaert_k=k_line, limb_peak=disc.limb_peak is not None)
        if twins is None:
            name = disc.body if disc.part is None else f'{disc.body} ({disc.part})'
            reason = 'its disc does not give a normal albedo for its twins'
            excluded.append(ExcludedDisc(disc.frame.image_id, name, reason))
            continue
        rows.append(DiscRow(disc=disc, twins=twins))
    law_bins: dict[str, dict[str, LawBin]] = {}
    body_bins: dict[str, dict[str, LawBin]] = {}
    limb_peaks: dict[str, dict[str, LimbPeakBin]] = {}
    by_phase: dict[str, list[DiscRow]] = {}
    by_body: dict[str, list[DiscRow]] = {}
    for row in rows:
        by_phase.setdefault(phase_bin(row.disc.phase_deg), []).append(row)
        by_body.setdefault(row.disc.label, []).append(row)
    for law in laws:
        law_bins[law] = {label: _law_bin(group, law) for label, group in sorted(by_phase.items())}
        body_bins[law] = {label: _law_bin(group, law) for label, group in sorted(by_body.items())}
        real_peaks: dict[str, list[float]] = {}
        twin_peaks: dict[str, list[float]] = {}
        for row in rows:
            twin = row.twins[law]
            if row.disc.limb_peak is None or twin.limb_peak is None:
                continue
            kind, value = row.disc.limb_peak
            real_peaks.setdefault(kind, []).append(value)
            twin_peaks.setdefault(kind, []).append(twin.limb_peak)
        limb_peaks[law] = compare_limb_peaks(real_peaks, twin_peaks)
    by_instrument: dict[str, list[DiscRow]] = {}
    for row in rows:
        by_instrument.setdefault(row.disc.frame.instrument, []).append(row)
    return LawStudy(
        rows=rows,
        excluded=excluded,
        minnaert_k0=k0,
        minnaert_k1=k1,
        law_bins=law_bins,
        body_bins=body_bins,
        limb_peaks=limb_peaks,
        disc_check={
            instrument: compare_disc_profiles(group)
            for instrument, group in sorted(by_instrument.items())
        },
        comparison=(
            law_comparison(rows, laws=laws) if DEFAULT_LAW in laws and len(set(laws)) >= 2 else {}
        ),
    )
