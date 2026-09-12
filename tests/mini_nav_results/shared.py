"""The constants and writer wrappers every host's documents are built from.

One navigation run recorded the same reproducibility envelope, the same
registered techniques and the same configuration digest for every image it
navigated, so those are stated once here.  So are the wrappers the per-host
modules build their documents through: the writer's own metadata and timing
builders, the inventory entries whose shape repeats, and the attitude and clock
block an orchestrator stamps onto a result, which a :class:`Host` describes the
host of.

Nothing here is a document or a host.  The documents are in the per-host modules
beside it, each host's camera frames, exposure and clock in a module named for
the host, and the package they belong to is the public surface.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from spindoctor.feature.feature import NavReliabilityBreakdown
from spindoctor.feature.feature_type import NavFeatureType
from spindoctor.nav_orchestrator.feature_summary import NavFeatureSummary
from spindoctor.nav_orchestrator.image_classifier_result import NavImageClassifierResult
from spindoctor.nav_orchestrator.nav_result import NavResult
from spindoctor.nav_orchestrator.provenance import Provenance
from spindoctor.navigate_image_files import build_metadata_from_result, build_timing_section
from spindoctor.support.cmatrix import AttitudeBaseline, PointingSolution
from spindoctor.support.types import NDArrayFloatType

# ---------------------------------------------------------------------------
# What every document of one run shares
# ---------------------------------------------------------------------------

_VERSION = '0.0.0'
"""Package version the run recorded."""


_GIT_SHA = '719cde5'
"""Short git SHA the run recorded."""


_CONFIG_HASH = '3ca76ec39b1fb875a86bed2793adc4430785242e07d705f2d65581963040a6b6'
"""Digest of the fully resolved configuration the run used."""


_PIPELINE_RUN = '2026-08-08T16:46:29Z'
"""When the run began, in the spelling the orchestrator stamps.

Seconds precision and a ``Z`` designator, which is what
``datetime.isoformat(timespec='seconds')`` produces for a UTC moment.
"""


_TECHNIQUE_NAMES = (
    'BodyBlobNav',
    'BodyDiscCorrelateNav',
    'BodyLimbNav',
    'BodyTerminatorNav',
    'RingAnnulusNav',
    'RingEdgeNav',
    'StarFieldFromCatalogNav',
    'StarRefineNav',
    'StarUniqueMatchNav',
    'TitanHazeNav',
)
"""Every technique the run had registered.

Written out rather than read from the registry: what a stored document holds is
what one run recorded, and a tree that changed shape whenever a technique was
added would move the frozen report for a reason the report is not about.
"""


_STATIC_DATA_HASHES = {
    'config_220_body_shape.yaml': (
        'ac10e82c9c141c0e449dcfc92d8c4f341400ffa51976f53c94c98eaabac7a52a'
    ),
    'config_310_saturn_rings.yaml': (
        '5f5f0b8f9a3b4d8c6e2a1c7d9b0e4f3a2d6c8b1e7f0a9d3c5b2e8f1a4d7c0b63'
    ),
    'config_400_inst_coiss.yaml': (
        '8c20d352ed0b5b690f7fc573f505f062551966c3305a01ae0e6fba63a8400f17'
    ),
}
"""Digests of the shipped static data the run hashed."""


_STAR_CATALOGS = {
    'tycho2': '/resources/SPICE/Stars',
    'ucac4': '/star-catalogs/UCAC4',
    'ybsc': '/star-catalogs/YBSC',
}
"""Where the run resolved each configured star catalog."""


def rotation(z_deg: float, y_deg: float, x_deg: float) -> NDArrayFloatType:
    """Return the proper rotation ``Rz . Ry . Rx`` for three angles.

    Parameters:
        z_deg: Rotation about the third axis, in degrees.
        y_deg: Rotation about the second axis, in degrees.
        x_deg: Rotation about the first axis, in degrees.

    Returns:
        The 3x3 rotation, orthonormal to float64 precision.
    """
    z, y, x = np.radians(np.array([z_deg, y_deg, x_deg], dtype=np.float64))
    rz = np.array(
        [[np.cos(z), -np.sin(z), 0.0], [np.sin(z), np.cos(z), 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    ry = np.array(
        [[np.cos(y), 0.0, np.sin(y)], [0.0, 1.0, 0.0], [-np.sin(y), 0.0, np.cos(y)]],
        dtype=np.float64,
    )
    rx = np.array(
        [[1.0, 0.0, 0.0], [0.0, np.cos(x), -np.sin(x)], [0.0, np.sin(x), np.cos(x)]],
        dtype=np.float64,
    )
    product: NDArrayFloatType = rz @ ry @ rx
    return product


def provenance(
    *, image_et: float, kernels: tuple[str, ...], extractors: tuple[str, ...]
) -> Provenance:
    """Return the reproducibility envelope one image's run recorded.

    Parameters:
        image_et: The observation midtime, which is the image's epoch.
        kernels: Kernel basenames the run had loaded.
        extractors: Names of the models built for the observation.

    Returns:
        The envelope.
    """
    return Provenance(
        spindoctor_version=_VERSION,
        image_et=image_et,
        pipeline_run_iso8601=_PIPELINE_RUN,
        spindoctor_git_sha=_GIT_SHA,
        spice_kernels=kernels,
        static_data_hashes=_STATIC_DATA_HASHES,
        technique_names=_TECHNIQUE_NAMES,
        extractor_names=extractors,
        config_hash=_CONFIG_HASH,
        config_overrides=(),
        star_catalogs=_STAR_CATALOGS,
    )


def classifier(
    *, noise_sigma: float, max_dn: float, gradient_score: float | None
) -> NavImageClassifierResult:
    """Return a clean-image classifier verdict.

    Parameters:
        noise_sigma: The MAD-based noise sigma the classifier measured.
        max_dn: The largest DN in the image.
        gradient_score: The background-gradient score, or None for an image
            whose downsample is perfectly flat.

    Returns:
        The verdict.
    """
    return NavImageClassifierResult(
        image_class='clean',
        saturation_frac=0.0,
        missing_frac=0.0,
        noise_sigma=noise_sigma,
        max_dn=max_dn,
        background_gradient_score=gradient_score,
        flags=[],
    )


def _pointing(
    *,
    camera_frame: str,
    camera_frame_id: int,
    ck_frame_id: int,
    oops_from_spice: NDArrayFloatType,
    midtime_et: float,
    exposure_s: float,
    sclk: tuple[str, str, str],
    original: NDArrayFloatType,
    corrected: NDArrayFloatType | None,
) -> PointingSolution:
    """Return the attitude solution the orchestrator stamps onto a result.

    Parameters:
        camera_frame: SPICE name of the camera frame.
        camera_frame_id: SPICE id of that frame.
        ck_frame_id: SPICE id of the object a corrected C-kernel targets.
        oops_from_spice: Constant rotation between the two frame conventions.
        midtime_et: Exposure midtime, which is also the image's epoch.
        exposure_s: Exposure duration in seconds.
        sclk: Spacecraft clock strings at start, midtime and stop.
        original: The uncorrected attitude at midtime.
        corrected: The corrected attitude, or None for a result that produced
            no offset and therefore no correction.

    Returns:
        The solution.
    """
    start_et, _midtime_et, stop_et = exposure_span(midtime_et, exposure_s)
    baseline = AttitudeBaseline(
        cmatrix_original=original,
        oops_from_spice=oops_from_spice,
        camera_frame=camera_frame,
        camera_frame_id=camera_frame_id,
        ck_frame_id=ck_frame_id,
        start_et=start_et,
        stop_et=stop_et,
        midtime_et=midtime_et,
        exposure_s=exposure_s,
        sclk_start=sclk[0],
        sclk_midtime=sclk[1],
        sclk_stop=sclk[2],
    )
    return PointingSolution(baseline=baseline, cmatrix=corrected)


def star(
    unique_number: int, *, reliability: float, snr_score: float, bbox: tuple[int, int, int, int]
) -> NavFeatureSummary:
    """Return one ungated catalog-star inventory entry.

    Parameters:
        unique_number: The star's UCAC4 identifier.
        reliability: The self-assessed score.
        snr_score: The detection component of that score.
        bbox: The star's extfov bounding box.

    Returns:
        The entry.
    """
    return NavFeatureSummary(
        feature_id=f'star:UCAC4:{unique_number}',
        feature_type=NavFeatureType.STAR,
        source_model='stars',
        reliability=reliability,
        gated=False,
        gate_reason=None,
        bbox_extfov_vu=bbox,
        reliability_reasons=NavReliabilityBreakdown(
            predicted_snr=snr_score,
            in_body_silhouette=False,
            in_saturation_or_cosmic=False,
            smear_length_ok=True,
        ),
    )


def faint_star(unique_number: int, *, bbox: tuple[int, int, int, int]) -> NavFeatureSummary:
    """Return one star gated out below the STAR reliability threshold.

    Parameters:
        unique_number: The star's UCAC4 identifier.
        bbox: The star's extfov bounding box.

    Returns:
        The entry, carrying the reason the gate writes.
    """
    return NavFeatureSummary(
        feature_id=f'star:UCAC4:{unique_number}',
        feature_type=NavFeatureType.STAR,
        source_model='stars',
        reliability=0.12,
        gated=True,
        gate_reason='reliability_0.120_below_threshold_0.200',
        bbox_extfov_vu=bbox,
        reliability_reasons=NavReliabilityBreakdown(
            predicted_snr=0.12,
            in_body_silhouette=False,
            in_saturation_or_cosmic=False,
            smear_length_ok=True,
        ),
    )


def ring_edge(
    ring_key: str,
    edge_label: str,
    *,
    planet: str,
    reliability: float,
    gated: bool,
    gate_reason: str | None,
    bbox: tuple[int, int, int, int],
) -> NavFeatureSummary:
    """Return one ring-edge inventory entry.

    Parameters:
        ring_key: The catalog key of the ring feature the edge belongs to.
        edge_label: ``IEG`` or ``OEG`` for the two edges of a gap.
        planet: The planet whose ring system the edge belongs to, as the ring
            model names it.
        reliability: The self-assessed score.
        gated: Whether the gate dropped it.
        gate_reason: Why, when it did.
        bbox: The edge's extfov bounding box.

    Returns:
        The entry.
    """
    return NavFeatureSummary(
        feature_id=f'ring_edge:{planet}:{ring_key}:{edge_label}',
        feature_type=NavFeatureType.RING_EDGE,
        source_model=f'rings:{planet}',
        reliability=reliability,
        gated=gated,
        gate_reason=gate_reason,
        bbox_extfov_vu=bbox,
        reliability_reasons=NavReliabilityBreakdown(
            visible_arc_fraction=1.0 if not gated else 0.21,
            shadow_occluded_fraction=0.0,
        ),
    )


@dataclass(frozen=True)
class Host:
    """What stamping an attitude solution onto a result needs to know about its host.

    Each host with a SPICE camera frame describes itself with one of these, in a
    module named for it, so that :func:`with_pointing` states every host's solution
    the same way.

    Attributes:
        camera_frames: The SPICE name and id of each camera's frame, keyed by the
            camera.
        ck_frame_id: The SPICE id of the object a corrected C-kernel targets.
        oops_from_spice: The constant rotation between the oops and SPICE camera
            frames.
        exposure_s: The exposure the host's images were taken with, in seconds.
        tick_s: How long one tick of the host's spacecraft clock lasts.
        spell: How a reading of that clock, as a count of ticks, is written.
    """

    camera_frames: Mapping[str, tuple[str, int]]
    ck_frame_id: int
    oops_from_spice: NDArrayFloatType
    exposure_s: float
    tick_s: float
    spell: Callable[[int], str]


# ---------------------------------------------------------------------------
# The spacecraft clock each host's readings are counted and spelled on
# ---------------------------------------------------------------------------
#
# A stored clock triple is whatever the epoch-to-clock conversion returned for
# the three epochs beside it, so the interval it spans is the interval those
# epochs span, to within the tick the clock counts in.  A triple that spans
# less than its own exposure is one no conversion could have produced, and
# every reader that subtracts two of its readings is then measuring a shutter
# that was never open that long.  So the readings are counted here rather than
# written out: each is the count at shutter open plus the ticks the epochs put
# between them, and each is spelled in its host's own fields.


def elapsed_ticks(seconds: float, tick_s: float) -> int:
    """Return how many ticks of a clock a span of time covers.

    The count is taken to the nearest tick rather than truncated.  The epochs
    and the tick are both decimal quantities, so a span covering a whole number
    of ticks in decimal reaches binary floating point a fraction of a tick to
    one side of it, and truncating charges the ones that land below as a whole
    tick that did not elapse.  Every reading is then within half a tick of its
    epoch, which is the most a counter of that tick can say about it.

    Parameters:
        seconds: How long the span is.
        tick_s: How long one tick of the clock is.

    Returns:
        The number of ticks.
    """
    return round(seconds / tick_s)


def sclk_triple(
    open_ticks: int,
    *,
    start_et: float,
    midtime_et: float,
    stop_et: float,
    tick_s: float,
    spell: Callable[[int], str],
) -> tuple[str, str, str]:
    """Return the clock readings at the three epochs of one exposure.

    Parameters:
        open_ticks: The reading at shutter open, as a count of ticks.
        start_et: When the shutter opened.
        midtime_et: The exposure midtime.
        stop_et: When the shutter closed.
        tick_s: How long one tick of the clock is.
        spell: How that clock's readings are written.

    Returns:
        The readings at start, midtime and stop.
    """
    return (
        spell(open_ticks),
        spell(open_ticks + elapsed_ticks(midtime_et - start_et, tick_s)),
        spell(open_ticks + elapsed_ticks(stop_et - start_et, tick_s)),
    )


def exposure_span(midtime_et: float, exposure_s: float) -> tuple[float, float, float]:
    """Return the start, midtime and stop epochs of one exposure.

    Parameters:
        midtime_et: The exposure midtime.
        exposure_s: How long the exposure was.

    Returns:
        The three epochs, in that order.
    """
    return midtime_et - exposure_s / 2.0, midtime_et, midtime_et + exposure_s / 2.0


def navigated(
    result: NavResult,
    *,
    image_name: str,
    image_path: Path | None = None,
    instrument: str,
    camera: str,
    shutter_mode: str | None,
    image_shape: tuple[int, int],
    start: datetime,
    elapsed_s: float,
    peak_memory_bytes: int,
) -> dict[str, Any]:
    """Return the document the writer builds for one navigated image.

    Parameters:
        result: The navigation result to curate.
        image_name: Basename of the source image.
        image_path: The file the run read, which a run records in full: the
            holdings root it was given, the volume set and volume under it, and
            the observation directory the image sits in.  Defaults to the
            basename directly under a holdings root, which is enough for a
            document whose reader asks nothing of the path.
        instrument: Registered instrument name of the observation class.
        camera: The camera that took the image.
        shutter_mode: The shutter mode the label recorded, or None for a host
            whose labels carry none.
        image_shape: The loaded image's ``(v, u)`` pixel dimensions.
        start: When this image's run began.
        elapsed_s: How long it took.
        peak_memory_bytes: The peak resident size to record for it.

    Returns:
        The document, as the writer assembles it.
    """
    return build_metadata_from_result(
        result,
        image_path if image_path is not None else Path(f'/holdings/{image_name}'),
        image_name,
        instrument=instrument,
        camera=camera,
        shutter_mode=shutter_mode,
        image_shape=image_shape,
        timing=pinned_timing(start, elapsed_s, peak_memory_bytes),
    )


def pinned_timing(start: datetime, elapsed_s: float, peak_memory_bytes: int) -> dict[str, Any]:
    """Build the timing block with every machine-taken value pinned.

    The writer reads the peak out of the running process, which is the right
    thing for a run and the wrong thing for a fixture: a document a test holds
    against a stored one has to be the same document every time it is built.
    The moments are pinned by being passed in; the peak is pinned by being
    written over what the process happened to reach.

    Parameters:
        start: The moment the run began.
        elapsed_s: How long it took.
        peak_memory_bytes: The peak resident size to record.

    Returns:
        The timing block, with nothing in it read from this machine.
    """
    # peak_measured=False so nothing about this machine reaches the block; the
    # fixture's own figure is written over it below.
    timing = build_timing_section(start, start + timedelta(seconds=elapsed_s), peak_measured=False)
    timing['peak_memory_bytes'] = peak_memory_bytes
    return timing


def with_pointing(
    result: NavResult,
    *,
    host: Host,
    camera: str,
    midtime_et: float,
    sclk_open: int,
    original: NDArrayFloatType,
    corrected: NDArrayFloatType | None,
) -> NavResult:
    """Stamp an attitude solution onto a result, as the orchestrator does.

    Parameters:
        result: The result to stamp.
        host: The host whose camera frames, exposure and clock the solution is
            stated in.
        camera: The camera that took the image, which names its frame.
        midtime_et: Exposure midtime, which is also the image's epoch.
        sclk_open: The host clock's reading at shutter open, as a tick count.
        original: The uncorrected attitude at midtime.
        corrected: The corrected attitude, or None for a result with no offset.

    Returns:
        The same result, carrying the solution.
    """
    camera_frame, camera_frame_id = host.camera_frames[camera]
    start_et, _midtime_et, stop_et = exposure_span(midtime_et, host.exposure_s)
    solution = _pointing(
        camera_frame=camera_frame,
        camera_frame_id=camera_frame_id,
        ck_frame_id=host.ck_frame_id,
        oops_from_spice=host.oops_from_spice,
        midtime_et=midtime_et,
        exposure_s=host.exposure_s,
        sclk=sclk_triple(
            sclk_open,
            start_et=start_et,
            midtime_et=midtime_et,
            stop_et=stop_et,
            tick_s=host.tick_s,
            spell=host.spell,
        ),
        original=original,
        corrected=corrected,
    )
    return dataclasses.replace(result, pointing=solution)
