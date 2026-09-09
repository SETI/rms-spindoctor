"""The constants and writer wrappers every host's documents are built from.

One navigation run recorded the same reproducibility envelope, the same
registered techniques and the same configuration digest for every image it
navigated, so those are stated once here.  So are the wrappers the per-host
modules build their documents through: the writer's own metadata and timing
builders, the inventory entries whose shape repeats, and the attitude and clock
block an orchestrator stamps onto a result.

Nothing here is a document.  The documents are in the per-host modules beside
it, and the package they belong to is the public surface.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
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

COISS_SUBTREE = 'COISS_2001/data/1294561143_1295221348'
"""The Cassini volume and observation directory the Cassini images sit under."""

VGISS_SUBTREE = 'VGISS_5101/data/C13854XX'
"""The Voyager volume and image directory the Voyager images sit under."""


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

COISS_KERNELS = (
    '05138_05159ra.bc',
    'cas00172.tsc',
    'cpck15Dec2017.tpc',
    'naif0012.tls',
    'sat428.bsp',
)
"""Kernels loaded for the Cassini images."""

VGISS_KERNELS = (
    'naif0012.tls',
    'vg100019.tsc',
    'vg1_saturn.bsp',
    'vg1_super.bc',
)
"""Kernels loaded for the Voyager images."""

SIM_KERNELS: tuple[str, ...] = ()
"""Kernels loaded for the simulated scene, of which there are none.

An empty list is a statement about the run rather than an absent value, and a
simulated scene is the run that makes it.
"""

_CASSINI_OOPS_FROM_SPICE: NDArrayFloatType = np.diag([-1.0, -1.0, 1.0])
"""The constant rotation between the oops and SPICE Cassini ISS camera frames."""

_VOYAGER_OOPS_FROM_SPICE: NDArrayFloatType = np.eye(3)
"""The constant rotation between the oops and SPICE Voyager ISS camera frames."""


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
    start_et, _midtime_et, stop_et = _exposure_span(midtime_et, exposure_s)
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
    reliability: float,
    gated: bool,
    gate_reason: str | None,
    bbox: tuple[int, int, int, int],
) -> NavFeatureSummary:
    """Return one Saturn ring-edge inventory entry.

    Parameters:
        ring_key: The catalog key of the ring feature the edge belongs to.
        edge_label: ``IEG`` or ``OEG`` for the two edges of a gap.
        reliability: The self-assessed score.
        gated: Whether the gate dropped it.
        gate_reason: Why, when it did.
        bbox: The edge's extfov bounding box.

    Returns:
        The entry.
    """
    return NavFeatureSummary(
        feature_id=f'ring_edge:SATURN:{ring_key}:{edge_label}',
        feature_type=NavFeatureType.RING_EDGE,
        source_model='rings:SATURN',
        reliability=reliability,
        gated=gated,
        gate_reason=gate_reason,
        bbox_extfov_vu=bbox,
        reliability_reasons=NavReliabilityBreakdown(
            visible_arc_fraction=1.0 if not gated else 0.21,
            shadow_occluded_fraction=0.0,
        ),
    )


_CASSINI_CAMERA_FRAME_IDS = {'NAC': -82360, 'WAC': -82361}
"""SPICE frame id of each Cassini ISS camera frame."""

_VOYAGER_CAMERA_FRAME_IDS = {'NAC': -31101, 'WAC': -31102}
"""SPICE frame id of each Voyager 1 ISS camera frame."""

_CASSINI_EXPOSURE_S = 0.46
"""Exposure the Cassini images were taken with."""

_VOYAGER_EXPOSURE_S = 1.44
"""Exposure the Voyager images were taken with."""


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


_CASSINI_SCLK_TICKS_PER_SECOND = 256
"""Ticks in one second of the Cassini clock, the modulus of its second field.

The clock is two fields, whole seconds and a fractional field counting ticks of
one 256th of a second, so a reading is a count of those ticks and the fields
are its quotient and its remainder by this.
"""

_CASSINI_SCLK_TICK_S = 1.0 / _CASSINI_SCLK_TICKS_PER_SECOND
"""Seconds in one tick of the Cassini clock."""

_VOYAGER_SCLK_LINES_PER_MINOR = 800
"""Lines in one minor frame, the modulus of the Voyager clock's line field."""

_VOYAGER_SCLK_MINORS_PER_FRAME = 60
"""Minor frames in one FDS frame, the modulus of the clock's minor-frame field."""

_VOYAGER_SCLK_FIRST_LINE = 1
"""The line field's offset: it counts from one rather than from zero."""

_VOYAGER_SCLK_TICK_S = 0.06
"""Seconds in one line, which is the tick the Voyager clock counts in.

A minor frame is 800 lines and an FDS frame is 60 minor frames, which puts an
FDS frame at 2880 seconds, the rate the clock kernel records for it.
"""


def _elapsed_ticks(seconds: float, tick_s: float) -> int:
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


def _cassini_sclk_reading(ticks: int) -> str:
    """Spell a Cassini clock tick count the way the conversion spells it.

    The two fields are written behind the clock partition and separated by a
    period, each zero padded to the digits its own modulus needs: ten for the
    seconds and three for the 256 ticks of the fraction.  A tick count past the
    fraction's modulus therefore carries into the seconds field rather than
    widening the fraction.

    Parameters:
        ticks: The reading, as a count of ticks of one 256th of a second.

    Returns:
        The clock string.
    """
    seconds, fraction = divmod(ticks, _CASSINI_SCLK_TICKS_PER_SECOND)
    return f'1/{seconds:010d}.{fraction:03d}'


def _voyager_sclk_reading(ticks: int) -> str:
    """Spell a Voyager clock tick count the way the conversion spells it.

    The three fields -- the FDS frame count, the minor frame within it and the
    line within that -- are written behind the clock partition and separated by
    colons, zero padded to five, two and three digits.  The line field counts
    from one, and a count that fills a field carries into the field above it
    rather than widening it.

    Parameters:
        ticks: The reading, as a count of line ticks.

    Returns:
        The clock string.
    """
    lines_per_frame = _VOYAGER_SCLK_MINORS_PER_FRAME * _VOYAGER_SCLK_LINES_PER_MINOR
    frame, within_frame = divmod(ticks, lines_per_frame)
    minor, line = divmod(within_frame, _VOYAGER_SCLK_LINES_PER_MINOR)
    return f'1/{frame:05d}:{minor:02d}:{line + _VOYAGER_SCLK_FIRST_LINE:03d}'


def cassini_sclk_open(image_number: int, tick: int) -> int:
    """Return a Cassini image's clock reading at shutter open, as a tick count.

    A Cassini image is named for the whole-second field of the reading its
    shutter opened at: a label carrying ``IMAGE_NUMBER = "1454725799"`` carries
    ``SPACECRAFT_CLOCK_START_COUNT = "1454725799.102"`` beside it.  So the
    image number and the tick the shutter opened on are the two fields of that
    reading.

    Parameters:
        image_number: The image number, which is the whole-second field.
        tick: The fractional field, in ticks of one 256th of a second.

    Returns:
        The reading, as a count of ticks.
    """
    return image_number * _CASSINI_SCLK_TICKS_PER_SECOND + tick


def voyager_sclk_open(frame: int, minor: int) -> int:
    """Return a Voyager image's clock reading at shutter open, as a tick count.

    A Voyager image is named for the frame and minor-frame fields of the
    reading its shutter closed at: a label carrying ``IMAGE_NUMBER = "13854.55"``
    carries ``SPACECRAFT_CLOCK_STOP_COUNT = "13854:55:001"`` beside it.  So the
    reading at shutter open is one exposure of ticks before the first line of
    the minor frame the image is named for.

    Parameters:
        frame: The FDS frame count the image is named for.
        minor: The minor frame within it the image is named for.

    Returns:
        The reading, as a count of ticks.
    """
    close = (frame * _VOYAGER_SCLK_MINORS_PER_FRAME + minor) * _VOYAGER_SCLK_LINES_PER_MINOR
    return close - _elapsed_ticks(_VOYAGER_EXPOSURE_S, _VOYAGER_SCLK_TICK_S)


def _sclk_triple(
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
        spell(open_ticks + _elapsed_ticks(midtime_et - start_et, tick_s)),
        spell(open_ticks + _elapsed_ticks(stop_et - start_et, tick_s)),
    )


def _exposure_span(midtime_et: float, exposure_s: float) -> tuple[float, float, float]:
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
        Path(f'/holdings/{image_name}'),
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
    camera: str,
    midtime_et: float,
    sclk_open: int,
    original: NDArrayFloatType,
    corrected: NDArrayFloatType | None,
    instrument: str = 'coiss',
) -> NavResult:
    """Stamp an attitude solution onto a result, as the orchestrator does.

    Parameters:
        result: The result to stamp.
        camera: The camera that took the image, which names its frame.
        midtime_et: Exposure midtime, which is also the image's epoch.
        sclk_open: The host clock's reading at shutter open, as a tick count.
        original: The uncorrected attitude at midtime.
        corrected: The corrected attitude, or None for a result with no offset.
        instrument: Which host's frames, clock and exposure to use.

    Returns:
        The same result, carrying the solution.
    """
    spell: Callable[[int], str]
    if instrument == 'coiss':
        camera_frame = f'CASSINI_ISS_{camera}'
        camera_frame_id = _CASSINI_CAMERA_FRAME_IDS[camera]
        ck_frame_id = -82000
        oops_from_spice = _CASSINI_OOPS_FROM_SPICE
        exposure_s = _CASSINI_EXPOSURE_S
        tick_s = _CASSINI_SCLK_TICK_S
        spell = _cassini_sclk_reading
    else:
        camera_frame = f'VG1_ISS{camera[0]}A'
        camera_frame_id = _VOYAGER_CAMERA_FRAME_IDS[camera]
        ck_frame_id = -31100
        oops_from_spice = _VOYAGER_OOPS_FROM_SPICE
        exposure_s = _VOYAGER_EXPOSURE_S
        tick_s = _VOYAGER_SCLK_TICK_S
        spell = _voyager_sclk_reading
    start_et, _midtime_et, stop_et = _exposure_span(midtime_et, exposure_s)
    solution = _pointing(
        camera_frame=camera_frame,
        camera_frame_id=camera_frame_id,
        ck_frame_id=ck_frame_id,
        oops_from_spice=oops_from_spice,
        midtime_et=midtime_et,
        exposure_s=exposure_s,
        sclk=_sclk_triple(
            sclk_open,
            start_et=start_et,
            midtime_et=midtime_et,
            stop_et=stop_et,
            tick_s=tick_s,
            spell=spell,
        ),
        original=original,
        corrected=corrected,
    )
    return dataclasses.replace(result, pointing=solution)
