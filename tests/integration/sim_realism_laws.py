"""The photometric-law study of the realism match.

A simulated body's brightness across its disc comes from its photometric law.  This
module checks the law against the real frames of the cohort, and compares it with every
other law the simulator offers, so the choice of the default law rests on the record.

**The real discs.**  :func:`collect_real_discs` looks at every moon in each calibrated
real frame, whatever the frame's scene class.  A moon is measured when its shape is
regular, it has no atmosphere, its disc is not clipped at the camera's full scale, at
least a quarter of its lit disc lies in the frame, and enough lit pixels lie clear of
the limb and the terminator.  Every moon in view that fails one of these is listed with
its reason.  The pixels a nearer moon or the planet covers are left out of a moon's
disc, since the frame's geometry traces each body alone.  The disc's pixels are selected and fitted with
:mod:`spindoctor.sim.realism.disc_profile`: Minnaert's exponent ``k``, the lunar-Lambert
blend ``L``, and the brightness profile along the photometric equator.  A moon whose two
hemispheres differ in albedo, as the mission's frame reader says, is measured one
hemisphere at a time.

**The twins.**  For each real disc, :func:`run_law_study` renders a twin under every law
in :data:`~spindoctor.sim.render.bodies.photometry.PHOTOMETRIC_LAWS`.  The twin is a
sphere at the moon's apparent size and position in the real frame, lit from the real
direction at the real phase, through the real frame's camera settings and exposure.  Its
normal albedo is measured from the real disc against the twin's own law
(:func:`~spindoctor.sim.realism.albedo.disc_normal_albedo`).  Minnaert's law needs an
exponent, and the twin takes it from a straight line fitted to the real discs' own
exponents against phase, so that law is the best one Minnaert's form allows.  The twin's
pixels are selected and fitted exactly as the real disc's were, from the sphere's own
geometry.

**The statistics.**  For each disc and law: the twin's exponent less the real one, the
residual between the two equator profiles' shapes, and, where the realism match measured
the real frame's limb gradient peak, the twin's limb gradient peak.  Each is reported by
phase bin and by body.

**The disc-profile check.**  For the default law, :func:`compare_disc_profiles` judges
each phase bin as the limb-peak check judges its bins: a bin with at least 3 discs gets
a band of 2 standard errors of the real exponents' robust scatter, never less than 0.02,
and the twins' median difference must lie within it.

**The law comparison.**  :func:`law_comparison` asks, in each phase bin with at least 3
discs, whether the default law's exponents agree with the real ones at least as well as
every other law's.  Two laws separate in a bin when their median absolute differences
differ by more than the bin's band; where they do not, no order between them is
claimed.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.ndimage

from spindoctor.config import DEFAULT_CONFIG
from spindoctor.feature.geometry import LimbPolyline
from spindoctor.nav_model.body_shape import load_body_shape
from spindoctor.nav_model.nav_model_body import bodies_in_extfov
from spindoctor.obs.obs_inst_sim import ObsSim
from spindoctor.sim.realism.albedo import disc_normal_albedo
from spindoctor.sim.realism.disc_profile import (
    DISC_CLEARANCE_PX,
    LunarLambertFit,
    MinnaertFit,
    equator_profile,
    fit_lunar_lambert,
    fit_minnaert,
    profile_residual,
    select_disc_pixels,
)
from spindoctor.sim.render.bodies.photometry import PHOTOMETRIC_LAWS, incidence_cosines
from spindoctor.sim.scene.validate import validate_sim_params
from spindoctor.support.memory import release_transient_memory
from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType
from tests.integration.sim_realism_body import EDGE_CLEARANCE_PX, MIN_CLIPPED_FRACTION
from tests.integration.sim_realism_limb_peak import (
    LIMB_PEAK_KIND_PREFIX,
    LimbPeakBin,
    compare_limb_peaks,
    matched_silhouette_offsets,
)
from tests.integration.sim_realism_scenes import MATCHED_BODY_LAW, matched_scene
from tests.integration.sim_realism_support import (
    PHASE_BIN_EDGES_DEG,
    extract_feature_samples,
    prepare_frame_features,
)

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
    'collect_real_discs',
    'compare_disc_profiles',
    'law_comparison',
    'minnaert_line',
    'phase_bin',
    'run_law_study',
]

STUDY_LAWS: tuple[str, ...] = tuple(sorted(PHOTOMETRIC_LAWS))
"""Every law the study renders twins under."""

DEFAULT_LAW = MATCHED_BODY_LAW
"""The law the disc-profile check judges: the matched bodies' law, the scene default."""

# The bodies whose atmosphere hides the surface the law describes.
_HAZE_BODIES = frozenset({'TITAN'})

# A moon in view smaller than this is listed by its size and not measured.
_MIN_DIAMETER_PX = 10.0

# A disc of which less than this share of the lit disc lies in the frame is left out:
# a strip along the limb alone cannot tell the laws apart.
MIN_LIT_FRACTION = 0.25

# The disc-profile band rule, in units of Minnaert's exponent.  It mirrors the limb
# gradient-peak rule: 2 standard errors of the bin's robust real scatter, a floor, and a
# verdict only for a bin with at least 3 discs.
DISC_BAND_MULTIPLE = 2.0
DISC_BAND_FLOOR = 0.02
MIN_BIN_DISCS = 3

# Bounds on the exponent the Minnaert twins take from the fitted line.
_MINNAERT_K_RANGE = (0.2, 1.5)

# At most this many pixels of a disc are kept for the fits, every n-th pixel.
_MAX_KEPT_PIXELS = 60_000

# The contact-sheet crops are block-averaged to at most this many pixels on a side.
_CROP_PX = 160
_CROP_MARGIN_PX = 8

_MAD_TO_SIGMA = 1.4826


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
    """One real moon's disc, or one hemisphere of it, measured for the study.

    Parameters:
        frame: The real frame.
        body: The moon, as SPICE knows it.
        part: ``'dark hemisphere'`` or ``'bright hemisphere'`` for a two-tone moon
            measured one hemisphere at a time, else None.
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
        crop: The disc's image, block-averaged for the contact sheet.
        limb_peak: The realism match's limb gradient-peak sample for the frame, as
            ``(kind, offset)``, when this moon is the one the match measured.
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
    limb_peak: tuple[str, float] | None = None

    @property
    def label(self) -> str:
        """The moon, with its hemisphere when it has one, in title case."""
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

    def add_limb_peak(self, image_id: str, body: str | None, samples: Mapping[str, Any]) -> None:
        """Attach a frame's limb gradient-peak sample to the disc it was measured on.

        Parameters:
            image_id: The real frame's image ID.
            body: The body the realism match measured the limb of, or None.
            samples: The real frame's samples after pairing; only ``limb_peak_*``
                kinds are read.
        """
        kinds = [k for k in samples if k.startswith(LIMB_PEAK_KIND_PREFIX)]
        if body is None or len(kinds) != 1 or len(samples[kinds[0]]) != 1:
            return
        for disc in self.discs:
            if disc.frame.image_id == image_id and disc.body == body and disc.part is None:
                disc.limb_peak = (kinds[0], float(samples[kinds[0]][0]))


@dataclass(frozen=True)
class TwinDisc:
    """One twin of a real disc under one law.

    Parameters:
        law: The twin's law.
        normal_albedo: The albedo measured from the real disc against that law.
        minnaert_k: The exponent of a Minnaert twin, else None.
        k: Minnaert's exponent fitted to the twin's counted pixels, or None.
        profile_residual: The residual between the real and the twin's equator
            profiles, or None.
        limb_peak: The twin's limb gradient-peak offset, when the real frame's was
            measured and the twin's could be.
        crop: The twin's image over the real disc's crop, block-averaged the same way.
    """

    law: str
    normal_albedo: float | None
    minnaert_k: float | None
    k: float | None
    profile_residual: float | None
    limb_peak: float | None
    crop: NDArrayFloatType


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
        """The twin's exponent less the real one, under one law."""
        twin = self.twins.get(law)
        if twin is None or twin.k is None:
            return None
        return twin.k - self.disc.minnaert.k


@dataclass(frozen=True)
class DiscProfileBin:
    """One phase bin of the disc-profile check.

    Parameters:
        n: Discs in the bin.
        real_median_k: The real exponents' median.
        sim_median_k: The default-law twins' median.
        median_delta_k: The median of each twin's exponent less its real disc's.
        real_scatter_k: The robust scatter of the real exponents, or None below 3
            discs.
        band_k: The band's half-width, or None below 3 discs.
        within_band: Whether the median difference lies within the band, or None
            without a band.
    """

    n: int
    real_median_k: float
    sim_median_k: float
    median_delta_k: float
    real_scatter_k: float | None
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
        band_k: The bin's band (see :class:`DiscProfileBin`).
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


def _shift(
    values: NDArrayFloatType | NDArrayBoolType, offset_vu: tuple[float, float], fill: Any
) -> Any:
    """Move a backplane array onto the image by the frame's offset, in whole pixels."""
    dv, du = round(offset_vu[0]), round(offset_vu[1])
    out = np.full(values.shape, fill, dtype=values.dtype)
    rows, cols = values.shape
    src = values[max(0, -dv) : rows - max(0, dv), max(0, -du) : cols - max(0, du)]
    out[max(0, dv) : max(0, dv) + src.shape[0], max(0, du) : max(0, du) + src.shape[1]] = src
    return out


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


def _light_direction_deg(
    on_body: NDArrayBoolType,
    mu0: NDArrayFloatType,
    center_vu: tuple[float, float],
    radius_px: float,
) -> float:
    """The in-image light direction that best explains a disc's incidence cosines.

    Each lit pixel's cosine is the dot product of its sphere normal with the direction
    to the Sun, so a least-squares fit over the lit pixels gives that direction.

    Returns:
        The direction as a scene's ``illumination_angle``: 0 when the light comes from
        the top of the image, 90 from the right.
    """
    rows, cols = np.nonzero(on_body & (mu0 > 0.05))
    normal_v = (rows + 0.5 - center_vu[0]) / radius_px
    normal_u = (cols + 0.5 - center_vu[1]) / radius_px
    normal_z = np.sqrt(np.clip(1.0 - normal_v**2 - normal_u**2, 0.0, None))
    design = np.column_stack([normal_v, normal_u, normal_z])
    sun, *_ = np.linalg.lstsq(design, mu0[rows, cols], rcond=None)
    return float(math.degrees(math.atan2(sun[1], -sun[0])) % 360.0)


def _crop(image: NDArrayFloatType, box: tuple[int, int, int, int]) -> NDArrayFloatType:
    """A crop of an image, block-averaged to at most 160 px on a side."""
    v0, v1, u0, u1 = box
    patch = np.asarray(image[v0:v1, u0:u1], dtype=np.float64)
    factor = max(1, math.ceil(max(patch.shape) / _CROP_PX))
    rows = patch.shape[0] // factor * factor
    cols = patch.shape[1] // factor * factor
    if rows == 0 or cols == 0:
        return patch.astype(np.float32)
    blocks = patch[:rows, :cols].reshape(rows // factor, factor, cols // factor, factor)
    return np.nanmean(blocks, axis=(1, 3)).astype(np.float32)


def _crop_box(on_body: NDArrayBoolType) -> tuple[int, int, int, int]:
    """The bounding box of a body's pixels, with a margin, inside the image."""
    rows, cols = np.nonzero(on_body)
    shape = on_body.shape
    return (
        max(int(rows.min()) - _CROP_MARGIN_PX, 0),
        min(int(rows.max()) + _CROP_MARGIN_PX + 1, shape[0]),
        max(int(cols.min()) - _CROP_MARGIN_PX, 0),
        min(int(cols.max()) + _CROP_MARGIN_PX + 1, shape[1]),
    )


def _decimate(mask: NDArrayBoolType) -> NDArrayBoolType:
    """Keep every n-th selected pixel so that at most 60,000 remain."""
    count = int(mask.sum())
    if count <= _MAX_KEPT_PIXELS:
        return mask
    step = math.ceil(count / _MAX_KEPT_PIXELS)
    flat = np.zeros(mask.size, dtype=bool)
    flat[np.flatnonzero(mask)[::step]] = True
    return flat.reshape(mask.shape)


def _hemispheres(
    obs: Any, body: str, offset_vu: tuple[float, float], dark_west_lon_deg: float | None
) -> dict[str | None, NDArrayBoolType | None]:
    """The parts a moon is measured in: the whole disc, or its two hemispheres."""
    if dark_west_lon_deg is None:
        return {None: None}
    longitude = np.ma.filled(
        obs.bp.longitude(body, reference='iau', direction='west').mvals, np.nan
    )
    west_deg = _shift(np.degrees(np.asarray(longitude, dtype=np.float64)), offset_vu, np.nan)
    apart = np.abs((west_deg - dark_west_lon_deg + 180.0) % 360.0 - 180.0)
    return {'dark hemisphere': apart < 90.0, 'bright hemisphere': apart >= 90.0}


def collect_real_discs(
    inputs: LawStudyInputs,
    obs: Any,
    frame: StudyFrame,
    *,
    clipped: NDArrayBoolType | None,
    dark_hemispheres: Mapping[str, float],
) -> None:
    """Measure every moon in one calibrated real frame for the study.

    Parameters:
        inputs: Where the measured discs and the moons left out are added.
        obs: The real observation; its data are I/F.
        frame: The frame's facts.
        clipped: Which of the image's pixels read the camera's full scale, or None.
        dark_hemispheres: For each two-tone moon, the west longitude in degrees at the
            middle of its dark hemisphere.
    """
    planet = getattr(obs, 'closest_planet', None)
    data = np.asarray(obs.data, dtype=np.float64)
    full_scale = (
        np.zeros(data.shape, dtype=bool) if clipped is None else np.asarray(clipped, dtype=bool)
    )
    usable = np.isfinite(data) & (data > 0.0) & ~full_scale
    in_view = [
        (body, entry)
        for body, entry in bodies_in_extfov(obs, config=DEFAULT_CONFIG)
        if obs.inventory_body_in_fov(entry)
    ]
    for body, entry in in_view:
        if body == planet:
            continue
        diameter = float(max(entry['u_pixel_size'], entry['v_pixel_size']))
        if diameter < 1.0:
            reason: str | None = 'under 1 px across'
        elif diameter < _MIN_DIAMETER_PX:
            reason = f'about {diameter:.0f} px across'
        else:
            reason = _exclusion_by_kind(body)
        if reason is not None:
            inputs.excluded.append(ExcludedDisc(frame.image_id, body, reason))
            continue
        _measure_moon(
            inputs,
            obs,
            frame,
            body,
            diameter_px=diameter,
            center_uv=(float(entry['center_uv'][0]), float(entry['center_uv'][1])),
            data=data,
            usable=usable & ~_nearer_cover(obs, frame.offset_vu, entry, in_view),
            full_scale=full_scale,
            dark_west_lon_deg=dark_hemispheres.get(body),
        )


def _boxes_overlap(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    """Whether two inventory entries' pixel boxes overlap."""
    return bool(
        a['u_min'] <= b['u_max']
        and b['u_min'] <= a['u_max']
        and a['v_min'] <= b['v_max']
        and b['v_min'] <= a['v_max']
    )


def _nearer_cover(
    obs: Any,
    offset_vu: tuple[float, float],
    entry: Mapping[str, Any],
    in_view: Sequence[tuple[str, Mapping[str, Any]]],
) -> NDArrayBoolType:
    """The pixels that a body nearer than ``entry``'s covers, widened by the blur.

    The frame's geometry traces each body alone, so where a nearer moon or the planet
    passes in front of a moon, the moon's backplanes still mark those pixels as its
    own.  They are left out of its disc, with a margin of the disc clearance for the
    blur of the nearer body's limb.
    """
    cover = np.zeros(np.shape(obs.data), dtype=bool)
    for other, other_entry in in_view:
        if other_entry is entry or other_entry['range'] >= entry['range']:
            continue
        if not _boxes_overlap(entry, other_entry):
            continue
        hit = np.asarray(obs.bp.where_intercepted(other).vals, dtype=bool)
        cover |= _shift(hit, offset_vu, False)
    if cover.any():
        cover = scipy.ndimage.binary_dilation(cover, iterations=math.ceil(DISC_CLEARANCE_PX))
    return np.asarray(cover, dtype=bool)


def _exclusion_by_kind(body: str) -> str | None:
    """Why a moon is left out whatever the frame, or None."""
    if body in _HAZE_BODIES:
        return 'its atmosphere hides the surface'
    shape = load_body_shape(body).shape_class_hint
    if shape != 'regular':
        return f'{shape.replace("_", " ")} shape, which the simulator does not model'
    return None


def _measure_moon(
    inputs: LawStudyInputs,
    obs: Any,
    frame: StudyFrame,
    body: str,
    *,
    diameter_px: float,
    center_uv: tuple[float, float],
    data: NDArrayFloatType,
    usable: NDArrayBoolType,
    full_scale: NDArrayBoolType,
    dark_west_lon_deg: float | None,
) -> None:
    """Measure one moon, or each of its hemispheres, and add it to the inputs."""
    offset = frame.offset_vu
    backplane = obs.bp
    on_body = _shift(np.asarray(backplane.where_intercepted(body).vals, dtype=bool), offset, False)
    if not on_body.any():
        inputs.excluded.append(ExcludedDisc(frame.image_id, body, 'its disc is off the image'))
        return
    interior = scipy.ndimage.binary_erosion(on_body, iterations=EDGE_CLEARANCE_PX)
    if interior.any() and float(np.mean(full_scale[interior])) >= MIN_CLIPPED_FRACTION:
        inputs.excluded.append(
            ExcludedDisc(frame.image_id, body, "its disc is clipped at the camera's full scale")
        )
        return
    incidence = np.ma.filled(backplane.incidence_angle(body).mvals, np.pi)
    emission = np.ma.filled(backplane.emission_angle(body).mvals, np.pi)
    phase = np.ma.filled(backplane.phase_angle(body).mvals, np.nan)
    mu0 = np.cos(_shift(np.asarray(incidence, dtype=np.float64), offset, np.pi))
    mu = np.cos(_shift(np.asarray(emission, dtype=np.float64), offset, np.pi))
    phase_deg_map = np.degrees(_shift(np.asarray(phase, dtype=np.float64), offset, np.nan))
    radius = diameter_px / 2.0
    center_vu = (center_uv[1] + offset[0], center_uv[0] + offset[1])
    lit = on_body & (mu0 > 0.0)
    for part, part_mask in _hemispheres(obs, body, offset, dark_west_lon_deg).items():
        label = body if part is None else f'{body} ({part})'
        selected = select_disc_pixels(on_body, mu0, mu, usable=usable)
        if part_mask is not None:
            selected &= part_mask
        lit_part = lit if part_mask is None else lit & part_mask
        if not lit_part.any():
            inputs.excluded.append(
                ExcludedDisc(frame.image_id, label, 'none of its lit disc lies in the frame')
            )
            continue
        phase_deg = float(np.nanmedian(phase_deg_map[selected if selected.any() else lit_part]))
        lit_share = _lit_share(lit, part_mask, radius, phase_deg)
        if lit_share < MIN_LIT_FRACTION:
            inputs.excluded.append(
                ExcludedDisc(
                    frame.image_id,
                    label,
                    f'only {100.0 * lit_share:.0f}% of its lit disc lies in the frame',
                )
            )
            continue
        kept = _decimate(selected)
        minnaert = fit_minnaert(data[kept], mu0[kept], mu[kept])
        if minnaert is None:
            inputs.excluded.append(
                ExcludedDisc(
                    frame.image_id,
                    label,
                    f'only {int(selected.sum())} lit pixels lie {DISC_CLEARANCE_PX:g} px '
                    'clear of the limb and the terminator',
                )
            )
            continue
        inputs.discs.append(
            RealDisc(
                frame=frame,
                body=body,
                part=part,
                phase_deg=phase_deg,
                diameter_px=diameter_px,
                center_vu=center_vu,
                illumination_deg=_light_direction_deg(on_body, mu0, center_vu, radius),
                intensity=data[kept],
                mu0=mu0[kept],
                mu=mu[kept],
                minnaert=minnaert,
                lunar_lambert=fit_lunar_lambert(data[kept], mu0[kept], mu[kept]),
                profile=equator_profile(data[kept], mu0[kept], mu[kept], math.radians(phase_deg)),
                crop=_crop(data, _crop_box(on_body)),
            )
        )


def _lit_share(
    lit: NDArrayBoolType, part_mask: NDArrayBoolType | None, radius_px: float, phase_deg: float
) -> float:
    """The share of a moon's lit disc, or of one hemisphere's, that lies in the frame."""
    if not math.isfinite(phase_deg):
        return 0.0
    whole = math.pi * radius_px**2 * (1.0 + math.cos(math.radians(phase_deg))) / 2.0
    in_frame = lit if part_mask is None else lit & part_mask
    share = float(in_frame.sum()) / max(whole, 1.0)
    # A hemisphere holds about half of the lit disc at most.
    return min(1.0, share if part_mask is None else 2.0 * share)


def minnaert_line(discs: Iterable[RealDisc]) -> tuple[float, float]:
    """A straight line through the real discs' Minnaert exponents against phase.

    Parameters:
        discs: The real discs.

    Returns:
        The exponent at zero phase and the slope per degree, from an unweighted least
        squares fit; ``(0.5, 0.0)`` when fewer than 2 discs are given.
    """
    phases = np.array([d.phase_deg for d in discs], dtype=np.float64)
    ks = np.array([d.minnaert.k for d in discs], dtype=np.float64)
    if phases.size < 2:
        return 0.5, 0.0
    slope, intercept = np.polyfit(phases, ks, 1)
    return float(intercept), float(slope)


def _twin(disc: RealDisc, law: str, *, minnaert_k: float, limb_peak: bool) -> TwinDisc:
    """Render and measure one real disc's twin under one law."""
    frame = disc.frame
    phase_rad = math.radians(disc.phase_deg)
    albedo = disc_normal_albedo(
        disc.intensity, disc.mu0, disc.mu, law=law, phase_angle=phase_rad, minnaert_k=minnaert_k
    )
    scene = matched_scene(
        f'{frame.image_id}_{disc.body}',
        'body_full_fov',
        frame.instrument,
        frame.exposure_sec,
        size_vu=frame.frame_shape_vu,
        diameter_px=disc.diameter_px,
        phase_angle_deg=disc.phase_deg,
        normal_albedo=albedo,
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
    on_body, mu0, mu = _sphere_cosines(
        data.shape,
        disc.center_vu,
        radius,
        illumination_deg=disc.illumination_deg,
        phase_deg=disc.phase_deg,
    )
    selected = select_disc_pixels(on_body, mu0, mu, usable=np.isfinite(data) & (data > 0.0))
    kept = _decimate(selected)
    fit = fit_minnaert(data[kept], mu0[kept], mu[kept])
    twin_profile = equator_profile(data[kept], mu0[kept], mu[kept], phase_rad)
    peak = _twin_limb_peak(obs, scene, disc.limb_peak[0]) if limb_peak and disc.limb_peak else None
    return TwinDisc(
        law=law,
        normal_albedo=albedo,
        minnaert_k=minnaert_k if law == 'minnaert' else None,
        k=None if fit is None else fit.k,
        profile_residual=profile_residual(disc.profile, twin_profile),
        limb_peak=peak,
        crop=_crop(data, _crop_box(on_body)),
    )


def _twin_limb_peak(obs: Any, scene: dict[str, Any], kind: str) -> float | None:
    """The twin's limb gradient-peak offset in the real frame's limb bin, if measured."""
    image, features, model_metadata = prepare_frame_features(obs, only_models='body_sim:*')
    features = [f for f in features if isinstance(f.geometry, LimbPolyline)]
    samples = extract_feature_samples(
        image,
        features,
        model_metadata,
        offset_vu=(0.0, 0.0),
        extfov_margin_vu=obs.extfov_margin_vu,
        saturation_level=math.nan,
        silhouette_offsets=matched_silhouette_offsets(scene, obs),
        limb_peak_model=f'body_sim:{scene["bodies"][0]["name"]}',
    ).samples
    values = samples.get(kind, [])
    return float(values[0]) if len(values) == 1 else None


def _robust_scatter(values: NDArrayFloatType) -> float:
    """The scaled median absolute deviation of ``values``."""
    return float(_MAD_TO_SIGMA * np.median(np.abs(values - np.median(values))))


def _band(real_k: NDArrayFloatType) -> float:
    """The disc-profile band for a bin's real exponents (at least 3 of them)."""
    scatter = _robust_scatter(real_k)
    return max(DISC_BAND_MULTIPLE * scatter / math.sqrt(real_k.size), DISC_BAND_FLOOR)


def compare_disc_profiles(
    rows: Iterable[DiscRow], law: str = DEFAULT_LAW
) -> dict[str, DiscProfileBin]:
    """Judge each phase bin's twins under one law against the real exponents.

    The band is set from the real exponents before the twins are looked at, as the limb
    gradient-peak check sets its bands: only a bin with at least 3 discs gets one, its
    half-width is 2 standard errors of the bin's robust real scatter and never less than
    0.02, and the median of the twins' exponent less the real one must lie within it.

    Parameters:
        rows: The discs, each with its twins.
        law: The law whose twins are judged.

    Returns:
        One entry per phase bin that holds a disc with a twin exponent under the law.
    """
    by_bin: dict[str, list[tuple[float, float]]] = {}
    for row in rows:
        twin = row.twins.get(law)
        if twin is None or twin.k is None:
            continue
        by_bin.setdefault(phase_bin(row.disc.phase_deg), []).append((row.disc.minnaert.k, twin.k))
    out: dict[str, DiscProfileBin] = {}
    for label, pairs in sorted(by_bin.items()):
        real_k = np.array([p[0] for p in pairs])
        sim_k = np.array([p[1] for p in pairs])
        delta = float(np.median(sim_k - real_k))
        scatter = band = None
        within = None
        if real_k.size >= MIN_BIN_DISCS:
            scatter = _robust_scatter(real_k)
            band = _band(real_k)
            within = bool(abs(delta) <= band)
        out[label] = DiscProfileBin(
            n=int(real_k.size),
            real_median_k=float(np.median(real_k)),
            sim_median_k=float(np.median(sim_k)),
            median_delta_k=delta,
            real_scatter_k=scatter,
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
    rows: Iterable[DiscRow], *, default: str = DEFAULT_LAW
) -> dict[str, LawComparisonBin]:
    """Ask in each phase bin whether the default law agrees best with the real discs.

    Parameters:
        rows: The discs, each with its twins under every law.
        default: The default law.

    Returns:
        One entry per phase bin holding at least 3 discs whose twins all have an
        exponent.
    """
    by_bin: dict[str, list[DiscRow]] = {}
    for row in rows:
        if all(row.delta_k(law) is not None for law in row.twins):
            by_bin.setdefault(phase_bin(row.disc.phase_deg), []).append(row)
    out: dict[str, LawComparisonBin] = {}
    for label, group in sorted(by_bin.items()):
        if len(group) < MIN_BIN_DISCS:
            continue
        band = _band(np.array([r.disc.minnaert.k for r in group]))
        laws = sorted(group[0].twins)
        agreement = {
            law: float(np.median([abs(r.delta_k(law) or 0.0) for r in group])) for law in laws
        }
        separates = {
            law: bool(abs(agreement[law] - agreement[default]) > band)
            for law in laws
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
        The study.
    """
    k0, k1 = minnaert_line(inputs.discs)
    rows: list[DiscRow] = []
    for disc in inputs.discs:
        k_line = min(max(k0 + k1 * disc.phase_deg, _MINNAERT_K_RANGE[0]), _MINNAERT_K_RANGE[1])
        twins = {}
        for law in laws:
            twins[law] = _twin(disc, law, minnaert_k=k_line, limb_peak=disc.limb_peak is not None)
            release_transient_memory()
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
        excluded=list(inputs.excluded),
        minnaert_k0=k0,
        minnaert_k1=k1,
        law_bins=law_bins,
        body_bins=body_bins,
        limb_peaks=limb_peaks,
        disc_check={
            instrument: compare_disc_profiles(group)
            for instrument, group in sorted(by_instrument.items())
        },
        comparison=law_comparison(rows) if DEFAULT_LAW in laws else {},
    )
