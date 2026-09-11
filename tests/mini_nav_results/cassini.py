"""The five Cassini documents of the fixture tree.

Four navigated images and one whose load failed before an observation existed.
Two of the four are the BOTSIM pair, which is one shutter over two cameras and
so one epoch and one clock reading on both; the other two carry the single
camera mode their labels record.  Between them they hold the two Cassini
outcomes the report reads, the four feature sources, the gated features, the
suspect offset, the ensemble exclusion and the spurious technique.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
from filecache import FCPath

from spindoctor.dataset.dataset import ImageFile, ImageFiles
from spindoctor.feature.feature import NavReliabilityBreakdown
from spindoctor.feature.feature_type import NavFeatureType
from spindoctor.nav_orchestrator.ensemble import derive_confidence_rank
from spindoctor.nav_orchestrator.feature_summary import NavFeatureSummary
from spindoctor.nav_orchestrator.nav_result import NavResult
from spindoctor.nav_technique.diagnostics import (
    BodyBlobDiagnostics,
    BodyDiscDiagnostics,
    BodyLimbDiagnostics,
    RingEdgeDiagnostics,
    StarFieldDiagnostics,
    StarUniqueMatchDiagnostics,
)
from spindoctor.nav_technique.technique_result import NavTechniqueResult
from spindoctor.navigate_image_files import navigate_image_files
from spindoctor.obs import ObsCassiniISS
from spindoctor.support.status_reason import NavStatusReason

from .cassini_host import CASSINI_ISS, COISS_KERNELS, cassini_sclk_open
from .shared import (
    classifier,
    faint_star,
    navigated,
    pinned_timing,
    provenance,
    ring_edge,
    rotation,
    star,
    with_pointing,
)

COISS_SUBTREE = 'COISS_2001/data/1294561143_1295221348'
"""The Cassini volume and observation directory the Cassini images sit under."""


def cassini_star_and_limb() -> dict[str, Any]:
    """The NAC half of the BOTSIM pair: stars and a body limb, both contributing.

    Two techniques agree closely, which is what the cross-technique agreement
    and confidence-calibration sections measure, and the fused offset is ten
    times its WAC partner's, which is what a consistent BOTSIM pair looks like.

    Returns:
        The document.
    """
    stars = [
        star(100000, reliability=0.91, snr_score=0.91, bbox=(142, 611, 153, 622)),
        star(100001, reliability=0.84, snr_score=0.84, bbox=(268, 344, 279, 355)),
        star(100002, reliability=0.77, snr_score=0.77, bbox=(401, 802, 412, 813)),
        star(100003, reliability=0.69, snr_score=0.69, bbox=(556, 178, 567, 189)),
        star(100004, reliability=0.55, snr_score=0.55, bbox=(704, 923, 715, 934)),
        star(100005, reliability=0.41, snr_score=0.41, bbox=(881, 466, 892, 477)),
    ]
    inventory = [
        *stars,
        faint_star(100099, bbox=(933, 705, 944, 716)),
        NavFeatureSummary(
            feature_id='body_disc:IAPETUS',
            feature_type=NavFeatureType.BODY_DISC,
            source_model='body:IAPETUS',
            reliability=0.62,
            gated=False,
            gate_reason=None,
            bbox_extfov_vu=(388, 402, 636, 650),
            reliability_reasons=NavReliabilityBreakdown(
                visible_lit_fraction=0.62, overflow_fraction=0.0
            ),
        ),
        NavFeatureSummary(
            feature_id='limb_arc:IAPETUS',
            feature_type=NavFeatureType.LIMB_ARC,
            source_model='body:IAPETUS',
            reliability=0.814,
            gated=False,
            gate_reason=None,
            bbox_extfov_vu=(388, 402, 636, 650),
            reliability_reasons=NavReliabilityBreakdown(
                visible_arc_fraction=1.0, incidence_factor=0.93
            ),
        ),
    ]
    per_technique = [
        NavTechniqueResult(
            technique_name='StarFieldFromCatalogNav',
            feature_ids=tuple(summary.feature_id for summary in stars),
            offset_px=(3.3, -1.45),
            covariance_px2=np.diag([0.1444, 0.1156]),
            confidence=0.93,
            spurious=False,
            at_edge=False,
            diagnostics=StarFieldDiagnostics(
                n_inliers=6,
                median_residual_px=0.184,
                n_detected_sources=11,
                n_catalog_predicted=7,
                n_triplets_evaluated=35,
                rotation_below_separability_floor=False,
                wide_offset_lock=False,
                wide_offset_false_lock_expectation=0.002,
            ),
        ),
        NavTechniqueResult(
            technique_name='BodyLimbNav',
            feature_ids=('limb_arc:IAPETUS',),
            offset_px=(3.1, -1.62),
            covariance_px2=np.diag([0.2916, 0.2401]),
            confidence=0.84,
            spurious=False,
            at_edge=False,
            diagnostics=BodyLimbDiagnostics(
                visible_limb_arc_fraction=0.986,
                visible_arc_px=277.0,
                dt_fit_rms_px=0.278,
                lm_iterations=12,
                tukey_inlier_count=269,
                lm_converged=True,
                polarity_rejection_fraction=0.0,
                coarse_peak_fraction=0.625,
            ),
        ),
    ]
    covariance = np.diag([0.0961, 0.0784])
    result = NavResult.success(
        offset_px=(3.25, -1.5),
        covariance_px2=covariance,
        confidence=0.91,
        confidence_rank=derive_confidence_rank(confidence=0.91, sigma_px=(0.31, 0.28)),
        status_reason=NavStatusReason.OK,
        per_technique=per_technique,
        feature_inventory=inventory,
        image_classifier=classifier(noise_sigma=0.75, max_dn=0.42, gradient_score=1.732),
        provenance=provenance(
            image_et=170000000.0,
            kernels=COISS_KERNELS,
            extractors=('body:IAPETUS', 'stars'),
        ),
        consensus_techniques=['StarFieldFromCatalogNav', 'BodyLimbNav'],
    )
    result = with_pointing(
        result,
        host=CASSINI_ISS,
        camera='NAC',
        midtime_et=170000000.0,
        sclk_open=cassini_sclk_open(1294561202, 77),
        corrected=rotation(41.208, -12.664, 87.311),
        original=rotation(41.204, -12.661, 87.309),
    )
    return navigated(
        result,
        image_name='N1294561202_1_CALIB.IMG',
        instrument='coiss',
        camera='NAC',
        shutter_mode='BOTSIM',
        image_shape=(1024, 1024),
        start=datetime(2026, 8, 8, 16, 46, 25, 933806, tzinfo=UTC),
        elapsed_s=12.5,
        peak_memory_bytes=5368709120,
    )


def cassini_all_features_gated() -> dict[str, Any]:
    """A Cassini failure whose two body features both fell below the gate.

    The reason and the inventory agree: every feature gated is what
    ``all_features_gated`` means, and it leaves the scene classified by content
    as a single body all the same.

    Returns:
        The document.
    """
    inventory = [
        NavFeatureSummary(
            feature_id='body_disc:IAPETUS',
            feature_type=NavFeatureType.BODY_DISC,
            source_model='body:IAPETUS',
            reliability=0.21,
            gated=True,
            gate_reason='reliability_0.210_below_threshold_0.300',
            bbox_extfov_vu=(470, 486, 552, 568),
            reliability_reasons=NavReliabilityBreakdown(
                visible_lit_fraction=0.21, overflow_fraction=0.0
            ),
        ),
        NavFeatureSummary(
            feature_id='limb_arc:IAPETUS',
            feature_type=NavFeatureType.LIMB_ARC,
            source_model='body:IAPETUS',
            reliability=0.24,
            gated=True,
            gate_reason='reliability_0.240_below_threshold_0.300',
            bbox_extfov_vu=(470, 486, 552, 568),
            reliability_reasons=NavReliabilityBreakdown(
                visible_arc_fraction=0.24, incidence_factor=0.31
            ),
        ),
    ]
    result = NavResult.failed(
        status_reason=NavStatusReason.ALL_FEATURES_GATED,
        image_classifier=classifier(noise_sigma=0.75, max_dn=0.18, gradient_score=2.104),
        provenance=provenance(
            image_et=170000800.0,
            kernels=COISS_KERNELS,
            extractors=('body:IAPETUS', 'stars'),
        ),
        feature_inventory=inventory,
    )
    result = with_pointing(
        result,
        host=CASSINI_ISS,
        camera='NAC',
        midtime_et=170000800.0,
        sclk_open=cassini_sclk_open(1294562000, 55),
        corrected=None,
        original=rotation(43.917, -11.902, 87.664),
    )
    return navigated(
        result,
        image_name='N1294562000_1_CALIB.IMG',
        instrument='coiss',
        camera='NAC',
        shutter_mode='NACONLY',
        image_shape=(1024, 1024),
        start=datetime(2026, 8, 8, 16, 46, 38, 532110, tzinfo=UTC),
        elapsed_s=8.25,
        peak_memory_bytes=1610612736,
    )


_LOAD_ERROR_IMAGE_NAME = 'N1294563000_1_CALIB.IMG'
"""Basename of the image whose load fails."""

LOAD_ERROR_STUB = f'{COISS_SUBTREE}/N1294563000_1_CALIB'
"""Where the failed image's document sits under the results root."""

_LOAD_ERROR_MESSAGE = (
    'SPICE(SPKINSUFFDATA) -- Insufficient ephemeris data has been loaded to compute '
    'the state of body -82 relative to body 699 at the ephemeris epoch '
    '2005 MAY 22 02:26:36.000.'
)
"""What SPICE says when the kernels cannot place the spacecraft.

The driver classifies a load failure by the hints in this text, so a document
whose ``status_error`` reads ``missing_spice_data`` has to carry a message that
says so.
"""

_LOAD_ERROR_TRACEBACK = (
    'Traceback (most recent call last):\n'
    '  File "spindoctor/navigate_image_files.py", line 267, in navigate_image_files\n'
    '    snapshot = obs_class.from_file(image_url, **extra_params)\n'
    '  File "spindoctor/obs/obs_cassini_iss.py", line 141, in from_file\n'
    '    return cls(iss.from_file(local_path, fast_distortion=True))\n'
    f'RuntimeError: {_LOAD_ERROR_MESSAGE}'
)
"""The traceback the stored document carries, in place of the built one.

A real traceback names the absolute source paths of the machine that raised it
and the line numbers of the moment it did, neither of which a stored document
can hold: the tree would differ on every checkout and after every edit above
the raising line.  This one is shaped exactly as the writer's, so what the tree
holds is still what a reader of a failed image's document sees.
"""


@contextlib.contextmanager
def _load_always_failing(message: str) -> Iterator[None]:
    """Make the Cassini host's loader raise, and put it back afterwards.

    The driver decides the instrument by exact identity against the registry,
    so the failing loader has to be the registered class's own rather than a
    stand-in subclass.

    Parameters:
        message: What the loader raises.

    Yields:
        Nothing; the loader is replaced for the body of the block.
    """

    def raising(cls: type, /, path: Any, **kwargs: Any) -> Any:
        """Fail the load the way an image with no kernel coverage fails.

        Parameters:
            cls: The observation class; unread.
            path: The image URL the driver resolved; unread.
            kwargs: Further loader options; unread.

        Raises:
            RuntimeError: always, carrying the SPICE coverage text.
        """
        raise RuntimeError(message)

    original = ObsCassiniISS.__dict__.get('from_file')
    setattr(ObsCassiniISS, 'from_file', classmethod(raising))  # noqa: B010
    try:
        yield
    finally:
        if original is None:
            delattr(ObsCassiniISS, 'from_file')
        else:
            setattr(ObsCassiniISS, 'from_file', original)  # noqa: B010


def cassini_load_error() -> dict[str, Any]:
    """The document the driver returns for an image with no kernel coverage.

    Built by the driver rather than by the metadata writer, because that is
    what writes this shape: no navigation result at all, no image shape, no
    epoch, and a ``status_error`` in place of a ``status_reason``.  The camera
    survives, since the dataset index named it without opening the image.

    Returns:
        The document, with its wall-clock timing and its traceback replaced by
        fixed ones.
    """
    image_path = Path('/holdings') / _LOAD_ERROR_IMAGE_NAME
    with TemporaryDirectory() as scratch, _load_always_failing(_LOAD_ERROR_MESSAGE):
        entry = ImageFile(
            image_file_url=FCPath(image_path.as_posix()),
            label_file_url=FCPath(image_path.with_suffix('.LBL').as_posix()),
            results_path_stub=LOAD_ERROR_STUB,
            camera='NAC',
            # The local path the driver would have downloaded to.  Supplied
            # rather than fetched: the load is the step under test and it
            # fails before any byte of the file is read.
            _image_file_path=image_path,
        )
        _success, document = navigate_image_files(
            ObsCassiniISS,
            ImageFiles(image_files=[entry]),
            FCPath(Path(scratch) / 'results'),
            write_output_files=False,
        )
    start = datetime(2026, 8, 8, 16, 46, 47, 104881, tzinfo=UTC)
    # The driver stamps the moments it ran at and reads the peak out of the
    # process it ran in, none of which a stored document can hold.  Rebuilt
    # through the writer's own section builder from fixed values.  The peak is
    # small because this image never loaded.
    document['timing'] = pinned_timing(start, 1.5, 268435456)
    document['status_traceback'] = _LOAD_ERROR_TRACEBACK
    return document


def cassini_suspect_offset() -> dict[str, Any]:
    """A Cassini success at the search limit, with an outlier and a spurious result.

    Three techniques reported.  The disc correlation is the consensus; the blob
    centroid is viable and sits far enough away to be rejected as an outlier,
    which is what ``excluded_from_consensus`` records; the single-star match
    self-flagged spurious and was dropped before consensus selection, which is
    why it is not in that list.  The fused offset reaches the configured search
    margin for this image size, so the frame is reported as suspect.

    Returns:
        The document.
    """
    stars = [
        star(200000, reliability=0.58, snr_score=0.58, bbox=(96, 233, 107, 244)),
        star(200001, reliability=0.52, snr_score=0.52, bbox=(214, 655, 225, 666)),
        star(200002, reliability=0.47, snr_score=0.47, bbox=(377, 91, 388, 102)),
        star(200003, reliability=0.39, snr_score=0.39, bbox=(512, 744, 523, 755)),
        star(200004, reliability=0.31, snr_score=0.31, bbox=(690, 318, 701, 329)),
        star(200005, reliability=0.24, snr_score=0.24, bbox=(842, 587, 853, 598)),
    ]
    inventory = [
        NavFeatureSummary(
            feature_id='body_disc:IAPETUS',
            feature_type=NavFeatureType.BODY_DISC,
            source_model='body:IAPETUS',
            reliability=0.55,
            gated=False,
            gate_reason=None,
            bbox_extfov_vu=(322, 448, 430, 556),
            reliability_reasons=NavReliabilityBreakdown(
                visible_lit_fraction=0.55, overflow_fraction=0.0
            ),
        ),
        NavFeatureSummary(
            feature_id='body_blob:IAPETUS',
            feature_type=NavFeatureType.BODY_BLOB,
            source_model='body:IAPETUS',
            reliability=0.38,
            gated=False,
            gate_reason=None,
            bbox_extfov_vu=(322, 448, 430, 556),
            reliability_reasons=NavReliabilityBreakdown(blob_snr=0.61, blob_extent_px=108.0),
        ),
        NavFeatureSummary(
            feature_id='limb_arc:IAPETUS',
            feature_type=NavFeatureType.LIMB_ARC,
            source_model='body:IAPETUS',
            reliability=0.68,
            gated=False,
            gate_reason=None,
            bbox_extfov_vu=(322, 448, 430, 556),
            reliability_reasons=NavReliabilityBreakdown(
                visible_arc_fraction=0.74, incidence_factor=0.88
            ),
        ),
        *stars,
        faint_star(200099, bbox=(908, 122, 919, 133)),
    ]
    per_technique = [
        NavTechniqueResult(
            technique_name='BodyDiscCorrelateNav',
            feature_ids=('body_disc:IAPETUS',),
            offset_px=(59.5, -12.0),
            covariance_px2=np.diag([6.8142, 5.2015]),
            confidence=0.42,
            spurious=False,
            at_edge=False,
            diagnostics=BodyDiscDiagnostics(
                ncc_peak=0.412,
                peak_to_runner_up_ratio=1.18,
                consistency_px=2.85,
                consistency_ratio=0.94,
                used_gradient=True,
                body_count=1,
            ),
        ),
        NavTechniqueResult(
            technique_name='BodyBlobNav',
            feature_ids=('body_blob:IAPETUS',),
            offset_px=(34.5, -3.2),
            covariance_px2=np.diag([182.25, 156.25]),
            confidence=0.24,
            spurious=False,
            at_edge=False,
            diagnostics=BodyBlobDiagnostics(
                body_snr_inside_predicted_bbox=41.9,
                body_extent_px=108.037,
                blob_count=1,
                residual_px=26.503,
                max_phase_angle_deg=78.412,
                max_phase_irregularity_factor=0.114,
            ),
        ),
        NavTechniqueResult(
            technique_name='StarUniqueMatchNav',
            feature_ids=('star:UCAC4:200001',),
            offset_px=(-8.0, 44.0),
            covariance_px2=np.diag([0.64, 0.64]),
            confidence=0.12,
            spurious=True,
            at_edge=True,
            diagnostics=StarUniqueMatchDiagnostics(
                mode='one_star',
                predicted_snr=6.4,
                brightness_margin_mag=0.35,
                residual_px=3.9,
                detection_peak_ratio=1.05,
            ),
        ),
    ]
    result = NavResult.success(
        offset_px=(59.5, -12.0),
        covariance_px2=np.diag([6.8142, 5.2015]),
        confidence=0.42,
        confidence_rank=derive_confidence_rank(confidence=0.42, sigma_px=(2.6104, 2.2807)),
        status_reason=NavStatusReason.OK,
        per_technique=per_technique,
        feature_inventory=inventory,
        image_classifier=classifier(noise_sigma=0.75, max_dn=1.06, gradient_score=3.298),
        provenance=provenance(
            image_et=170002800.0,
            kernels=COISS_KERNELS,
            extractors=('body:IAPETUS', 'stars'),
        ),
        excluded_from_consensus=['BodyBlobNav'],
        consensus_techniques=['BodyDiscCorrelateNav'],
    )
    result = with_pointing(
        result,
        host=CASSINI_ISS,
        camera='NAC',
        midtime_et=170002800.0,
        sclk_open=cassini_sclk_open(1294564000, 11),
        corrected=rotation(46.882, -10.117, 88.204),
        original=rotation(46.861, -10.104, 88.196),
    )
    return navigated(
        result,
        image_name='N1294564000_1_CALIB.IMG',
        instrument='coiss',
        camera='NAC',
        shutter_mode='NACONLY',
        image_shape=(1024, 1024),
        start=datetime(2026, 8, 8, 16, 46, 48, 921574, tzinfo=UTC),
        elapsed_s=31.75,
        peak_memory_bytes=8589934592,
    )


def cassini_ring_edges() -> dict[str, Any]:
    """The WAC half of the BOTSIM pair: one Encke gap edge navigated the frame.

    It shares its partner's spacecraft clock and epoch, since a BOTSIM pair is
    one shutter, and its offset is a tenth of the NAC image's, since one WAC
    pixel is ten NAC pixels.

    Returns:
        The document.
    """
    inventory = [
        ring_edge(
            'encke_gap',
            'IEG',
            planet='SATURN',
            reliability=0.72,
            gated=False,
            gate_reason=None,
            bbox=(118, 40, 402, 472),
        ),
        ring_edge(
            'encke_gap',
            'OEG',
            planet='SATURN',
            reliability=0.18,
            gated=True,
            gate_reason='reliability_0.180_below_threshold_0.300',
            bbox=(120, 41, 405, 474),
        ),
    ]
    per_technique = [
        NavTechniqueResult(
            technique_name='RingEdgeNav',
            feature_ids=('ring_edge:SATURN:encke_gap:IEG',),
            offset_px=(0.36, -0.11),
            covariance_px2=np.diag([0.0961, 0.0784]),
            confidence=0.74,
            spurious=False,
            at_edge=False,
            diagnostics=RingEdgeDiagnostics(
                total_edge_length_px=431.0,
                per_edge_dt_rms_summed=0.216,
                per_edge_dt_rms_mean=0.216,
                per_edge_dt_median_max=0.184,
                edge_count=1,
                is_rank_1=False,
                lm_converged=True,
                coarse_peak_fraction=0.812,
                sigma_orbit_radial_px=0.031,
            ),
        )
    ]
    result = NavResult.success(
        offset_px=(0.35, -0.12),
        covariance_px2=np.diag([0.0961, 0.0784]),
        confidence=0.74,
        confidence_rank=derive_confidence_rank(confidence=0.74, sigma_px=(0.31, 0.28)),
        status_reason=NavStatusReason.OK,
        per_technique=per_technique,
        feature_inventory=inventory,
        image_classifier=classifier(noise_sigma=0.75, max_dn=0.88, gradient_score=1.417),
        provenance=provenance(
            image_et=170000000.0,
            kernels=COISS_KERNELS,
            extractors=('rings:SATURN', 'stars'),
        ),
        consensus_techniques=['RingEdgeNav'],
    )
    result = with_pointing(
        result,
        host=CASSINI_ISS,
        camera='WAC',
        midtime_et=170000000.0,
        sclk_open=cassini_sclk_open(1294561202, 77),
        corrected=rotation(41.211, -12.669, 87.315),
        original=rotation(41.204, -12.661, 87.309),
    )
    return navigated(
        result,
        image_name='W1294561202_1_CALIB.IMG',
        instrument='coiss',
        camera='WAC',
        shutter_mode='BOTSIM',
        image_shape=(512, 512),
        start=datetime(2026, 8, 8, 16, 47, 21, 8443, tzinfo=UTC),
        elapsed_s=12.5,
        peak_memory_bytes=2147483648,
    )


def results_tree_documents() -> dict[str, dict[str, Any]]:
    """Return the Cassini documents of the fixture tree, keyed by results path stub.

    Returns:
        Stub to document, in the order the tree is written.
    """
    return {
        f'{COISS_SUBTREE}/N1294561202_1_CALIB': cassini_star_and_limb(),
        f'{COISS_SUBTREE}/N1294562000_1_CALIB': cassini_all_features_gated(),
        LOAD_ERROR_STUB: cassini_load_error(),
        f'{COISS_SUBTREE}/N1294564000_1_CALIB': cassini_suspect_offset(),
        f'{COISS_SUBTREE}/W1294561202_1_CALIB': cassini_ring_edges(),
    }
