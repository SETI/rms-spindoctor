"""The three Cassini images the bundle cohort is built from.

Selected for what bundle generation reads and the statistics report has no use
for: two successes whose numbers shard into different bundle directories, one
of them with ring backplanes and one without, and one image whose navigation
did not succeed, which is what the bundle stage skips.  The documents in
``cassini`` are selected for the report instead, and the two sets stay apart --
a fixture chosen against two unrelated criteria stops being legible for either.

Each image is built from its epoch and nothing else.  The clock readings come
from :func:`~tests.mini_nav_results.shared.cassini_sclk_triple` and the image
number from :func:`~tests.mini_nav_results.shared.cassini_image_number`, so the
name, the readings, the index row and the document cannot disagree about when
the shutter was open.

The index row is the row an enumeration hands on with an image.  An enumeration
reads the columns it declares, which today are the file specification and the
instrument, so a row a run carries holds those two; the row here holds every
column the label stage reads, which is the row that stage is written against.
Its values are a real COISS index row's, except for the identity, time and
clock columns, which are this image's own.  The column names are the index
file's own names, so a label variable read from a column that index has no
such name for renders empty here exactly as it does on a real image.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import julian
import numpy as np

from spindoctor.feature.feature import NavReliabilityBreakdown
from spindoctor.feature.feature_type import NavFeatureType
from spindoctor.nav_orchestrator.ensemble import derive_confidence_rank
from spindoctor.nav_orchestrator.feature_summary import NavFeatureSummary
from spindoctor.nav_orchestrator.nav_result import NavResult
from spindoctor.nav_technique.diagnostics import BodyLimbDiagnostics, RingEdgeDiagnostics
from spindoctor.nav_technique.technique_result import NavTechniqueResult
from spindoctor.support.status_reason import NavStatusReason

from .backplanes import COHORT_SHAPE_VU, CohortBody
from .shared import (
    CASSINI_EXPOSURE_MS,
    COISS_KERNELS,
    cassini_exposure_span,
    cassini_image_number,
    cassini_sclk_triple,
    classifier,
    navigated,
    provenance,
    ring_edge,
    rotation,
    with_pointing_from_epoch,
)

LIMB_MIDTIME_ET = 129400000.0
"""The epoch of the image navigated on a satellite's limb."""

RINGS_MIDTIME_ET = 130700000.31
"""The epoch of the image navigated on a ring edge, fifteen days later.

Far enough from the first that the two shard into different bundle directories
at both levels, which is the layout the collections and the inventories are
assembled over.
"""

GATED_MIDTIME_ET = 129400823.55
"""The epoch of the image whose features all fell below the gate.

Minutes after the first, so the two sit in one observation directory: an image
the bundle has nothing to say about is the ordinary neighbour of one it does.
"""

_LIMB_SUBTREE = 'COISS_2001/data/1454725799_1455008789'
"""The volume and observation directory the first and third images sit under."""

_RINGS_SUBTREE = 'COISS_2001/data/1456049355_1456316702'
"""The volume and observation directory the second image sits under.

Each observation directory of a Cassini volume is named for the range of image
numbers it holds, and each of these holds the number derived from its image's
own epoch.
"""


def _image_name(midtime_et: float, camera_letter: str) -> str:
    """Return the name of the calibrated image taken at an epoch.

    Parameters:
        midtime_et: The exposure midtime, which is the image's epoch.
        camera_letter: ``N`` or ``W``, the camera that took it.

    Returns:
        The image name, without its extension.
    """
    return f'{camera_letter}{cassini_image_number(midtime_et)}_1_CALIB'


LIMB_IMAGE_NAME = _image_name(LIMB_MIDTIME_ET, 'N')
"""Name of the narrow-angle image navigated on a limb."""

RINGS_IMAGE_NAME = _image_name(RINGS_MIDTIME_ET, 'W')
"""Name of the wide-angle image navigated on a ring edge."""

GATED_IMAGE_NAME = _image_name(GATED_MIDTIME_ET, 'N')
"""Name of the image whose navigation did not succeed."""

LIMB_STUB = f'{_LIMB_SUBTREE}/{LIMB_IMAGE_NAME}'
"""Where the limb image's results sit under a results root."""

RINGS_STUB = f'{_RINGS_SUBTREE}/{RINGS_IMAGE_NAME}'
"""Where the ring image's results sit under a results root."""

GATED_STUB = f'{_LIMB_SUBTREE}/{GATED_IMAGE_NAME}'
"""Where the gated image's results sit under a results root."""


HOLDINGS_SUBTREE = 'calibrated/COISS_2xxx'
"""Where a Cassini volume's calibrated images sit under a holdings root.

An enumeration finds an image under the directory its products were calibrated
into and the volume set its volume belongs to, so a path that names neither
tells a reader who parses one nothing at all.
"""

_RECORDED_HOLDINGS_ROOT = '/holdings'
"""The holdings root the run that wrote these documents was given.

A document records the file that run read, on the machine it read it on, and
nothing rewrites that when the results are copied somewhere else or read by a
bundle run given holdings of its own.  So the cohort's own holdings root, which
is wherever it was written, is deliberately not this: what is shared between
the two is the layout below the root, which is the half anything derives a
volume or a collection from.
"""


def _image_path(stub: str) -> Path:
    """Return the file a run read for the image whose results sit at a stub.

    Parameters:
        stub: Where the image's results sit under a results root.

    Returns:
        The full path, holdings root and all.
    """
    return Path(f'{_RECORDED_HOLDINGS_ROOT}/{HOLDINGS_SUBTREE}/{stub}.IMG')


def _doy(epoch_et: float) -> str:
    """Return an epoch in the day-of-year spelling a PDS3 index records.

    Parameters:
        epoch_et: The epoch.

    Returns:
        The UTC time, to milliseconds.
    """
    return str(julian.iso_from_tai(julian.tai_from_tdb(epoch_et), digits=3, ymd=False))


def _index_row(stub: str, midtime_et: float, *, camera: str, shutter_mode: str) -> dict[str, Any]:
    """Return the index row an enumeration hands on with one cohort image.

    The volume and the directory come from the stub the results are written
    under, so the file the row names is the file the document is written for
    rather than a second answer to the same question.  The times and the
    exposure come from the one epoch and the one exposure the clock triple is
    counted over, for the same reason: a row is free to record a shutter that
    was open for one interval beside readings taken over another, and every
    reader of it holds only one of the two.

    Parameters:
        stub: Where the image's results sit under a results root.
        midtime_et: The exposure midtime, which is the image's epoch.
        camera: ``NAC`` or ``WAC``, the camera that took the image.
        shutter_mode: The shutter mode the label records.

    Returns:
        The row, keyed by the index file's own column names.
    """
    start_et, _midtime_et, stop_et = cassini_exposure_span(midtime_et)
    sclk_start, _sclk_midtime, sclk_stop = cassini_sclk_triple(midtime_et)
    image_number = cassini_image_number(midtime_et)
    letter = 'N' if camera == 'NAC' else 'W'
    volume, under_the_volume = stub.split('/', 1)
    directory = under_the_volume.rsplit('/', 1)[0]
    return {
        'FILE_NAME': f'{letter}{image_number}_1.IMG',
        'FILE_SPECIFICATION_NAME': f'{directory}/{letter}{image_number}_1.IMG',
        'VOLUME_ID': volume,
        'ANTIBLOOMING_STATE_FLAG': 'OFF',
        'BIAS_STRIP_MEAN': 14.869863,
        'CALIBRATION_LAMP_STATE_FLAG': 'N/A',
        'COMMAND_FILE_NAME': 'trigger_9_4.ioi',
        'COMMAND_SEQUENCE_NUMBER': 8,
        'DARK_STRIP_MEAN': 0.0,
        'DATA_CONVERSION_TYPE': '12BIT',
        'DATA_SET_ID': 'CO-S-ISSNA/ISSWA-2-EDR-V1.0',
        'DELAYED_READOUT_FLAG': 'NO',
        'DESCRIPTION': 'N/A',
        'DETECTOR_TEMPERATURE': -89.243546,
        'EARTH_RECEIVED_START_TIME': _doy(midtime_et + 86400.0),
        'EARTH_RECEIVED_STOP_TIME': _doy(midtime_et + 86461.0),
        'ELECTRONICS_BIAS': 112,
        'EXPECTED_MAXIMUM': np.array([50.0, 75.0]),
        'EXPECTED_PACKETS': 1143,
        'EXPOSURE_DURATION': CASSINI_EXPOSURE_MS,
        'FILTER_NAME': ('CL1', 'CL2'),
        'FILTER_TEMPERATURE': -0.468354,
        'FLIGHT_SOFTWARE_VERSION_ID': '1.3',
        'GAIN_MODE_ID': '29 ELECTRONS PER DN',
        'IMAGE_MID_TIME': _doy(midtime_et),
        'IMAGE_NUMBER': str(image_number),
        'IMAGE_OBSERVATION_TYPE': 'SCIENCE',
        'IMAGE_TIME': _doy(stop_et),
        'INSTRUMENT_DATA_RATE': 365.567993,
        'INSTRUMENT_HOST_NAME': 'CASSINI ORBITER',
        'INSTRUMENT_ID': 'ISSNA' if camera == 'NAC' else 'ISSWA',
        'INSTRUMENT_MODE_ID': 'FULL',
        'INST_CMPRS_RATE': np.array([6.0, 2.116882]),
        'INST_CMPRS_RATIO': 7.558285,
        'INST_CMPRS_TYPE': 'LOSSLESS',
        'LIGHT_FLOOD_STATE_FLAG': 'ON',
        'METHOD_DESC': 'RINGS',
        'MISSING_LINES': 0,
        'MISSING_PACKET_FLAG': 'NO',
        'MISSION_NAME': 'CASSINI-HUYGENS',
        'MISSION_PHASE_NAME': 'APPROACH_SCIENCE',
        'OBSERVATION_ID': 'ISS_C42SA_SATURN001_PRIME',
        'OPTICS_TEMPERATURE': np.array([0.712693, 1.820474]),
        'ORDER_NUMBER': 0,
        'PARALLEL_CLOCK_VOLTAGE_INDEX': 9,
        'PREPARE_CYCLE_INDEX': 3,
        'PRODUCT_CREATION_TIME': _doy(midtime_et + 172800.0),
        'PRODUCT_VERSION_TYPE': 'FINAL',
        'READOUT_CYCLE_INDEX': 5,
        'RECEIVED_PACKETS': 309,
        'SENSOR_HEAD_ELEC_TEMPERATURE': 1.633024,
        'SEQUENCE_ID': 'C42',
        'SEQUENCE_NUMBER': 1,
        'SEQUENCE_TITLE': '--',
        'SHUTTER_MODE_ID': shutter_mode,
        'SHUTTER_STATE_ID': 'ENABLED',
        'SOFTWARE_VERSION_ID': 'ISS 9.00 05-22-2003',
        'SPACECRAFT_CLOCK_CNT_PARTITION': 1,
        'SPACECRAFT_CLOCK_START_COUNT': sclk_start.split('/', 1)[1],
        'SPACECRAFT_CLOCK_STOP_COUNT': sclk_stop.split('/', 1)[1],
        'START_TIME': _doy(start_et),
        'STOP_TIME': _doy(stop_et),
        'TARGET_DESC': 'SATURN',
        'TARGET_LIST': 'N/A',
        'TARGET_NAME': 'SATURN',
        'TELEMETRY_FORMAT_ID': 'S_N_ER_5',
        'VALID_MAXIMUM': np.array([4095, 4095]),
    }


def cassini_body_limb() -> dict[str, Any]:
    """A narrow-angle success navigated on the limb of a satellite.

    One body in the frame and no rings, which is the image the bundle describes
    with body backplanes alone.

    Returns:
        The document.
    """
    inventory = [
        NavFeatureSummary(
            feature_id='body_disc:ENCELADUS',
            feature_type=NavFeatureType.BODY_DISC,
            source_model='body:ENCELADUS',
            reliability=0.71,
            gated=False,
            gate_reason=None,
            bbox_extfov_vu=(2, 4, 10, 14),
            reliability_reasons=NavReliabilityBreakdown(
                visible_lit_fraction=0.71, overflow_fraction=0.0
            ),
        ),
        NavFeatureSummary(
            feature_id='limb_arc:ENCELADUS',
            feature_type=NavFeatureType.LIMB_ARC,
            source_model='body:ENCELADUS',
            reliability=0.86,
            gated=False,
            gate_reason=None,
            bbox_extfov_vu=(2, 4, 10, 14),
            reliability_reasons=NavReliabilityBreakdown(
                visible_arc_fraction=0.97, incidence_factor=0.91
            ),
        ),
    ]
    per_technique = [
        NavTechniqueResult(
            technique_name='BodyLimbNav',
            feature_ids=('limb_arc:ENCELADUS',),
            offset_px=(1.2, -0.75),
            covariance_px2=np.diag([0.0576, 0.0441]),
            confidence=0.88,
            spurious=False,
            at_edge=False,
            diagnostics=BodyLimbDiagnostics(
                visible_limb_arc_fraction=0.97,
                visible_arc_px=31.0,
                dt_fit_rms_px=0.194,
                lm_iterations=9,
                tukey_inlier_count=30,
                lm_converged=True,
                polarity_rejection_fraction=0.0,
                coarse_peak_fraction=0.688,
            ),
        )
    ]
    result = NavResult.success(
        offset_px=(1.2, -0.75),
        covariance_px2=np.diag([0.0576, 0.0441]),
        confidence=0.88,
        confidence_rank=derive_confidence_rank(confidence=0.88, sigma_px=(0.24, 0.21)),
        status_reason=NavStatusReason.OK,
        per_technique=per_technique,
        feature_inventory=inventory,
        image_classifier=classifier(noise_sigma=0.68, max_dn=0.55, gradient_score=1.284),
        provenance=provenance(
            image_et=LIMB_MIDTIME_ET,
            kernels=COISS_KERNELS,
            extractors=('body:ENCELADUS', 'stars'),
        ),
        consensus_techniques=['BodyLimbNav'],
    )
    result = with_pointing_from_epoch(
        result,
        camera='NAC',
        midtime_et=LIMB_MIDTIME_ET,
        corrected=rotation(38.114, -9.402, 85.771),
        original=rotation(38.110, -9.399, 85.769),
    )
    return navigated(
        result,
        image_name=f'{LIMB_IMAGE_NAME}.IMG',
        image_path=_image_path(LIMB_STUB),
        instrument='coiss',
        camera='NAC',
        shutter_mode='NACONLY',
        image_shape=COHORT_SHAPE_VU,
        start=datetime(2026, 9, 8, 11, 2, 14, 118304, tzinfo=UTC),
        elapsed_s=9.5,
    )


def cassini_ring_edges() -> dict[str, Any]:
    """A wide-angle success navigated on a ring edge, with Saturn in the frame.

    Both a body and the ring system, which is the image the bundle describes
    with ring backplanes beside body ones.

    Returns:
        The document.
    """
    inventory = [
        NavFeatureSummary(
            feature_id='body_disc:SATURN',
            feature_type=NavFeatureType.BODY_DISC,
            source_model='body:SATURN',
            reliability=0.64,
            gated=False,
            gate_reason=None,
            bbox_extfov_vu=(3, 2, 14, 8),
            reliability_reasons=NavReliabilityBreakdown(
                visible_lit_fraction=0.64, overflow_fraction=0.12
            ),
        ),
        ring_edge(
            'encke_gap',
            'IEG',
            reliability=0.79,
            gated=False,
            gate_reason=None,
            bbox=(1, 0, 14, 15),
        ),
    ]
    per_technique = [
        NavTechniqueResult(
            technique_name='RingEdgeNav',
            feature_ids=('ring_edge:SATURN:encke_gap:IEG',),
            offset_px=(-0.42, 0.18),
            covariance_px2=np.diag([0.1024, 0.0729]),
            confidence=0.77,
            spurious=False,
            at_edge=False,
            diagnostics=RingEdgeDiagnostics(
                total_edge_length_px=15.0,
                per_edge_dt_rms_summed=0.243,
                per_edge_dt_rms_mean=0.243,
                per_edge_dt_median_max=0.201,
                edge_count=1,
                is_rank_1=False,
                lm_converged=True,
                coarse_peak_fraction=0.774,
                sigma_orbit_radial_px=0.044,
            ),
        )
    ]
    result = NavResult.success(
        offset_px=(-0.42, 0.18),
        covariance_px2=np.diag([0.1024, 0.0729]),
        confidence=0.77,
        confidence_rank=derive_confidence_rank(confidence=0.77, sigma_px=(0.32, 0.27)),
        status_reason=NavStatusReason.OK,
        per_technique=per_technique,
        feature_inventory=inventory,
        image_classifier=classifier(noise_sigma=0.81, max_dn=1.34, gradient_score=2.017),
        provenance=provenance(
            image_et=RINGS_MIDTIME_ET,
            kernels=COISS_KERNELS,
            extractors=('body:SATURN', 'rings:SATURN'),
        ),
        consensus_techniques=['RingEdgeNav'],
    )
    result = with_pointing_from_epoch(
        result,
        camera='WAC',
        midtime_et=RINGS_MIDTIME_ET,
        corrected=rotation(52.663, -14.208, 91.442),
        original=rotation(52.651, -14.199, 91.437),
    )
    return navigated(
        result,
        image_name=f'{RINGS_IMAGE_NAME}.IMG',
        image_path=_image_path(RINGS_STUB),
        instrument='coiss',
        camera='WAC',
        shutter_mode='WACONLY',
        image_shape=COHORT_SHAPE_VU,
        start=datetime(2026, 9, 8, 11, 2, 23, 840117, tzinfo=UTC),
        elapsed_s=7.25,
    )


def cassini_all_features_gated() -> dict[str, Any]:
    """A navigation that did not succeed: every feature fell below the gate.

    The bundle has nothing to describe for it -- no offset, and no backplanes
    were ever generated -- so the bundle stage skips it rather than failing.

    Returns:
        The document.
    """
    inventory = [
        NavFeatureSummary(
            feature_id='body_disc:ENCELADUS',
            feature_type=NavFeatureType.BODY_DISC,
            source_model='body:ENCELADUS',
            reliability=0.19,
            gated=True,
            gate_reason='reliability_0.190_below_threshold_0.300',
            bbox_extfov_vu=(5, 5, 10, 10),
            reliability_reasons=NavReliabilityBreakdown(
                visible_lit_fraction=0.19, overflow_fraction=0.0
            ),
        )
    ]
    result = NavResult.failed(
        status_reason=NavStatusReason.ALL_FEATURES_GATED,
        image_classifier=classifier(noise_sigma=0.68, max_dn=0.11, gradient_score=1.902),
        provenance=provenance(
            image_et=GATED_MIDTIME_ET,
            kernels=COISS_KERNELS,
            extractors=('body:ENCELADUS', 'stars'),
        ),
        feature_inventory=inventory,
    )
    result = with_pointing_from_epoch(
        result,
        camera='NAC',
        midtime_et=GATED_MIDTIME_ET,
        corrected=None,
        original=rotation(38.221, -9.377, 85.804),
    )
    return navigated(
        result,
        image_name=f'{GATED_IMAGE_NAME}.IMG',
        image_path=_image_path(GATED_STUB),
        instrument='coiss',
        camera='NAC',
        shutter_mode='NACONLY',
        image_shape=COHORT_SHAPE_VU,
        start=datetime(2026, 9, 8, 11, 2, 31, 502776, tzinfo=UTC),
        elapsed_s=5.75,
    )


@dataclass(frozen=True)
class CohortImage:
    """One image of the cohort, and everything a run leaves on disk for it.

    Attributes:
        stub: Where its results sit under a results root.
        image_name: The calibrated image's name, without its extension.
        camera: The camera that took it.
        midtime_et: The exposure midtime, which is its epoch and the one input
            every other value here is derived from.
        document: The navigation document a run wrote for it.
        index_file_row: The index row an enumeration hands on with it.
        bodies: The bodies its backplanes cover, empty when it has none.
        rings: Whether its backplanes cover the ring system.
    """

    stub: str
    image_name: str
    camera: str
    midtime_et: float
    document: dict[str, Any]
    index_file_row: dict[str, Any]
    bodies: tuple[CohortBody, ...]
    rings: bool

    @property
    def navigated(self) -> bool:
        """Whether the navigation succeeded, which is what the bundle stage reads."""
        return bool(self.document.get('status') == 'success')


def cohort_images() -> tuple[CohortImage, ...]:
    """Return every cohort image, in the order a run would have processed them.

    Returns:
        The images, each with its document, its index row and its backplanes.
    """
    return (
        CohortImage(
            stub=LIMB_STUB,
            image_name=LIMB_IMAGE_NAME,
            camera='NAC',
            midtime_et=LIMB_MIDTIME_ET,
            document=cassini_body_limb(),
            index_file_row=_index_row(
                LIMB_STUB, LIMB_MIDTIME_ET, camera='NAC', shutter_mode='NACONLY'
            ),
            bodies=(
                CohortBody(
                    name='ENCELADUS',
                    center_vu=(6.0, 9.0),
                    radii_vu=(4.0, 5.0),
                    range_km=284913.0,
                ),
            ),
            rings=False,
        ),
        CohortImage(
            stub=RINGS_STUB,
            image_name=RINGS_IMAGE_NAME,
            camera='WAC',
            midtime_et=RINGS_MIDTIME_ET,
            document=cassini_ring_edges(),
            index_file_row=_index_row(
                RINGS_STUB, RINGS_MIDTIME_ET, camera='WAC', shutter_mode='WACONLY'
            ),
            bodies=(
                CohortBody(
                    name='SATURN',
                    center_vu=(8.5, 5.0),
                    radii_vu=(5.5, 3.5),
                    range_km=1904772.0,
                ),
            ),
            rings=True,
        ),
        CohortImage(
            stub=GATED_STUB,
            image_name=GATED_IMAGE_NAME,
            camera='NAC',
            midtime_et=GATED_MIDTIME_ET,
            document=cassini_all_features_gated(),
            index_file_row=_index_row(
                GATED_STUB, GATED_MIDTIME_ET, camera='NAC', shutter_mode='NACONLY'
            ),
            bodies=(),
            rings=False,
        ),
    )
