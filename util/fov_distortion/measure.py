"""Per-frame star-field measurement of twist and residual distortion.

For one image this module:

1. Loads the observation through the instrument's ``from_file`` and runs
   star-only navigation to recover the translation prior (the residual
   spacecraft pointing offset).
2. Predicts every catalog star the star model would use and, shifting each
   prediction by the navigation offset, fits a sub-pixel PSF centroid to the
   image at that location.
3. Rejects stars with no detectable peak, a centroid that ran to the search
   limit, or a post-fit residual far larger than the rest, then hands the
   surviving predicted / detected pairs to :func:`decompose_frame`.

This is the only module in the package that depends on spindoctor and the
navigation holdings.  The rotation fit is done here, on the collected pairs,
independent of the navigation ``fit_camera_rotation`` setting, so an
instrument that ships with rotation fitting off is still measured.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import cast

import numpy as np
from filecache import FCPath
from numpy.typing import NDArray
from psfmodel import GaussianPSF
from util.fov_distortion.decompose import FrameDecomposition, decompose_frame

from spindoctor.config import DEFAULT_CONFIG, Config
from spindoctor.nav_model.nav_model import build_models_for_obs
from spindoctor.nav_model.stars.nav_model_stars import NavModelStars
from spindoctor.nav_orchestrator.orchestrator import NavOrchestrator
from spindoctor.obs import inst_name_to_obs_class
from spindoctor.obs.obs_snapshot_inst import ObsSnapshotInst

__all__ = [
    'FrameMeasurement',
    'MeasureParams',
    'StarMeasurement',
    'measure_frame',
]

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class MeasureParams:
    """Detection and rejection parameters for one measurement.

    Parameters:
        max_stars: Maximum catalog stars the star model reduces to; sets how
            densely the field is sampled.
        psf_sigma_px: Gaussian PSF sigma for the centroid fit.  Deliberately a
            little wider than the instrument PSF so a slightly displaced,
            smeared, or saturated star is still fitted.
        box_half_px: Half-width of the PSF fit box in pixels (the full box is
            ``2 * box_half_px + 1``, forced odd).
        search_limit_px: Maximum distance the centroid may move from the
            shifted prediction before the detection is rejected as a wrong peak.
        edge_margin_px: Stars whose detection lands within this many pixels of
            the image edge are dropped (edge-truncated PSFs bias the centroid).
        min_peak_over_background: Minimum ratio of the detected peak over the
            local background for the detection to be accepted.
        residual_clip_px: After a first decomposition, stars whose post-twist
            residual exceeds this many pixels are dropped and the decomposition
            is repeated once.
        min_stars: Fewest surviving stars for a usable frame decomposition.
        radial_powers: Polynomial powers (in normalized radius) for the radial
            distortion fit; the default matches the simulator distortion warp.
        fast_distortion: Passed to ``from_file``.  When True the faster
            approximate oops distortion model is used (matching production
            navigation); when False the full model is applied, so the measured
            residual is purely uncorrected distortion rather than the
            fast-model approximation.
    """

    max_stars: int = 100
    psf_sigma_px: float = 1.5
    box_half_px: int = 7
    search_limit_px: float = 2.5
    edge_margin_px: float = 10.0
    min_peak_over_background: float = 1.3
    residual_clip_px: float = 1.0
    min_stars: int = 6
    radial_powers: tuple[int, ...] = (3, 5)
    fast_distortion: bool = True


@dataclass(frozen=True)
class StarMeasurement:
    """One star's predicted and measured position.

    Parameters:
        predicted_vu: Catalog-predicted ``(v, u)`` before the navigation
            offset, in sensor pixels.
        detected_vu: PSF-centroided ``(v, u)`` in sensor pixels.
        vmag: Catalog visual magnitude.
        peak_dn: Detected peak value over the local background, in DN.
    """

    predicted_vu: tuple[float, float]
    detected_vu: tuple[float, float]
    vmag: float
    peak_dn: float


@dataclass(frozen=True)
class FrameMeasurement:
    """Result of measuring one frame.

    Parameters:
        image_name: Short image identifier (URL stem).
        url: Source URL (unexpanded).
        inst_id: Instrument id (``coiss`` / ``vgiss`` / ``nhlorri`` / ``gossi``).
        image_shape: ``(height, width)`` of the sensor image.
        offset_vu: Navigation translation offset ``(dv, du)``, or ``None`` if
            navigation failed.
        center_vu: Optical center used for the decomposition.
        rho_ref_px: Normalizing radius (half image diagonal).
        stars: Surviving per-star measurements.
        decomposition: The twist + radial decomposition, or ``None`` when the
            frame did not yield enough stars.
        status: ``'ok'`` or a short failure tag.
        reason: Human-readable status detail.
    """

    image_name: str
    url: str
    inst_id: str
    image_shape: tuple[int, int]
    offset_vu: tuple[float, float] | None
    center_vu: tuple[float, float]
    rho_ref_px: float
    stars: list[StarMeasurement] = field(default_factory=list)
    decomposition: FrameDecomposition | None = None
    status: str = 'ok'
    reason: str = ''


def _odd_box(box_half_px: int) -> tuple[int, int]:
    """Return an odd ``(box, box)`` size from a half-width."""
    box = 2 * int(box_half_px) + 1
    return (box, box)


def _centroid_star(
    image: FloatArray,
    psf: GaussianPSF,
    predicted_vu: tuple[float, float],
    params: MeasureParams,
) -> tuple[tuple[float, float], float] | None:
    """PSF-fit one star; return ``(detected_vu, peak_over_background)`` or None."""
    v_pred, u_pred = predicted_vu
    h, w = image.shape
    box = _odd_box(params.box_half_px)
    half_v, half_u = box[0] // 2, box[1] // 2
    if v_pred < half_v or v_pred > h - half_v - 1 or u_pred < half_u or u_pred > w - half_u - 1:
        return None
    # ``find_position`` takes and returns positions in the ``eval_rect``
    # convention: an offset is measured from the upper left corner of a pixel and
    # its own default (0.5, 0.5) sits on that pixel's centre, so the centre of
    # row i comes back as i + 0.5.  That is pixel corner, the same system
    # ``star.v`` / ``star.u`` are in, so the prediction goes in and the detection
    # comes out without a conversion and the two subtract directly.  Do not take
    # a half pixel off either one.
    ret = psf.find_position(
        image,
        box,
        (v_pred, u_pred),
        search_limit=(params.search_limit_px, params.search_limit_px),
    )
    if ret is None:
        return None
    det_v, det_u, metadata = ret
    # Pixel corner spans [0, h] for h rows, so ``det_v < margin`` and
    # ``det_v > h - margin`` cut bands of the same width at both edges.
    if (
        det_v < params.edge_margin_px
        or det_v > h - params.edge_margin_px
        or det_u < params.edge_margin_px
        or det_u > w - params.edge_margin_px
    ):
        return None
    subimg = np.asarray(metadata['subimg'], dtype=np.float64)
    background = float(np.median(subimg))
    peak = float(subimg.max())
    if background <= 0.0:
        over_background = peak
    else:
        over_background = peak / background
    if over_background < params.min_peak_over_background:
        return None
    return (float(det_v), float(det_u)), peak - background


def measure_frame(
    url: str,
    inst_id: str,
    *,
    params: MeasureParams | None = None,
    config: Config | None = None,
) -> FrameMeasurement:
    """Measure the twist and residual distortion of one image.

    Parameters:
        url: Image URL; may embed ``${PDS3_HOLDINGS_DIR}`` and similar tokens.
        inst_id: Instrument id used to select the observation class.
        params: Detection / rejection parameters; defaults if omitted.
        config: Configuration; ``DEFAULT_CONFIG`` if omitted.  ``stars.max_stars``
            is set from ``params.max_stars`` only while the star models are built
            and is restored afterward, so a shared config is left unchanged.

    Returns:
        A :class:`FrameMeasurement`.  Frames that fail to load, fail
        navigation, or yield too few stars carry a non-``ok`` status and a
        ``None`` decomposition rather than raising.
    """
    params = params or MeasureParams()
    config = config or DEFAULT_CONFIG
    config.read_config()
    image_name = url.split('/')[-1].split('.')[0]

    try:
        obs_class = inst_name_to_obs_class(inst_id)
        obs = cast(
            ObsSnapshotInst,
            obs_class.from_file(FCPath(url).expandvars(), fast_distortion=params.fast_distortion),
        )
    except Exception as exc:
        return FrameMeasurement(
            image_name=image_name,
            url=url,
            inst_id=inst_id,
            image_shape=(0, 0),
            offset_vu=None,
            center_vu=(0.0, 0.0),
            rho_ref_px=0.0,
            status='load_failed',
            reason=str(exc),
        )

    image = np.nan_to_num(np.asarray(obs.data, dtype=np.float64))
    h, w = image.shape
    # Radial origin for the decomposition.  This is the pixel centric frame
    # centre; the star positions it is used with are pixel corner, whose frame
    # centre is (h / 2, w / 2), so the origin sits half a pixel up and left of
    # the field centre those positions measure from.  Against a half-diagonal
    # normalizing radius that half pixel moves the fitted k1 by four parts in ten
    # thousand, k2 by three in a thousand and the twist by a microdegree, all of
    # them an order of magnitude inside the error the fit itself makes on a known
    # distortion, so it is recorded here rather than chased.  Either way it is
    # the frame centre, not the camera's boresight, which this tool does not ask
    # about.
    center = ((h - 1) / 2.0, (w - 1) / 2.0)
    rho_ref = 0.5 * math.hypot(h, w)

    original_max_stars = config.stars.max_stars
    config.stars.max_stars = params.max_stars
    try:
        models = build_models_for_obs(obs, config=config)
    finally:
        config.stars.max_stars = original_max_stars
    star_models = [m for m in models if isinstance(m, NavModelStars)]
    result = NavOrchestrator(models, only_models='stars').navigate(obs)
    if result.offset_px is None:
        return FrameMeasurement(
            image_name=image_name,
            url=url,
            inst_id=inst_id,
            image_shape=(h, w),
            offset_vu=None,
            center_vu=center,
            rho_ref_px=rho_ref,
            status='nav_failed',
            reason=f'navigation status {result.status}: {result.status_reason}',
        )
    offset = (float(result.offset_px[0]), float(result.offset_px[1]))

    # Reuse the star model the orchestrator already populated during navigate;
    # its predicted positions are the ones the offset above was fitted against.
    if not star_models or not star_models[0].stars:
        return FrameMeasurement(
            image_name=image_name,
            url=url,
            inst_id=inst_id,
            image_shape=(h, w),
            offset_vu=offset,
            center_vu=center,
            rho_ref_px=rho_ref,
            status='no_star_model',
            reason='navigation produced no populated star model',
        )
    model = star_models[0]
    psf = GaussianPSF(sigma=params.psf_sigma_px)

    measurements: list[StarMeasurement] = []
    for star in model.stars:
        predicted = (float(star.v), float(star.u))
        shifted = (predicted[0] + offset[0], predicted[1] + offset[1])
        detected = _centroid_star(image, psf, shifted, params)
        if detected is None:
            continue
        (det_v, det_u), peak = detected
        measurements.append(
            StarMeasurement(
                predicted_vu=predicted,
                detected_vu=(det_v, det_u),
                vmag=float(getattr(star, 'vmag', float('nan'))),
                peak_dn=peak,
            )
        )

    frame = FrameMeasurement(
        image_name=image_name,
        url=url,
        inst_id=inst_id,
        image_shape=(h, w),
        offset_vu=offset,
        center_vu=center,
        rho_ref_px=rho_ref,
        stars=measurements,
    )
    if len(measurements) < params.min_stars:
        return _with_status(
            frame,
            'too_few_stars',
            f'{len(measurements)} stars < min {params.min_stars}',
        )

    decomposition = _decompose_with_clip(measurements, center, rho_ref, params)
    surviving = decomposition[1]
    if len(surviving) < params.min_stars:
        return _with_status(
            frame,
            'too_few_stars',
            f'{len(surviving)} stars after clip < min {params.min_stars}',
        )
    return FrameMeasurement(
        image_name=image_name,
        url=url,
        inst_id=inst_id,
        image_shape=(h, w),
        offset_vu=offset,
        center_vu=center,
        rho_ref_px=rho_ref,
        stars=surviving,
        decomposition=decomposition[0],
    )


def _decompose_with_clip(
    measurements: list[StarMeasurement],
    center: tuple[float, float],
    rho_ref: float,
    params: MeasureParams,
) -> tuple[FrameDecomposition, list[StarMeasurement]]:
    """Decompose, drop stars with a large post-twist residual, decompose once more."""
    pred = np.array([m.predicted_vu for m in measurements], dtype=np.float64)
    det = np.array([m.detected_vu for m in measurements], dtype=np.float64)
    first = decompose_frame(pred, det, center, rho_ref, powers=params.radial_powers)
    residual_mag = np.hypot(first.twist.residuals_vu[:, 0], first.twist.residuals_vu[:, 1])
    keep = residual_mag <= params.residual_clip_px
    if keep.all() or int(keep.sum()) < params.min_stars:
        return first, measurements
    surviving = [m for m, k in zip(measurements, keep, strict=True) if k]
    pred2 = pred[keep]
    det2 = det[keep]
    second = decompose_frame(pred2, det2, center, rho_ref, powers=params.radial_powers)
    return second, surviving


def _with_status(frame: FrameMeasurement, status: str, reason: str) -> FrameMeasurement:
    """Return a copy of ``frame`` with a failure status set."""
    return FrameMeasurement(
        image_name=frame.image_name,
        url=frame.url,
        inst_id=frame.inst_id,
        image_shape=frame.image_shape,
        offset_vu=frame.offset_vu,
        center_vu=frame.center_vu,
        rho_ref_px=frame.rho_ref_px,
        stars=frame.stars,
        decomposition=None,
        status=status,
        reason=reason,
    )
