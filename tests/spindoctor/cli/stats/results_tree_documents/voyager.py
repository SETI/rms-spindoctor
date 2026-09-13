"""The two Voyager documents of the fixture tree.

One success on a ring-gap edge, at an image size no search limit is configured
for, and one failure in which no extractor produced a feature at all.  Voyager
labels record no shutter mode, so neither document carries one.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import numpy as np

from spindoctor.nav_orchestrator.ensemble import derive_confidence_rank
from spindoctor.nav_orchestrator.nav_result import NavResult
from spindoctor.nav_technique.diagnostics import RingEdgeDiagnostics
from spindoctor.nav_technique.technique_result import NavTechniqueResult
from spindoctor.obs.obs_inst_voyager_iss import _published_sclk
from spindoctor.support.status_reason import NavStatusReason

from .shared import (
    VGISS_KERNELS,
    classifier,
    holdings_path,
    navigated,
    provenance,
    published_times,
    recorded_exposure,
    ring_edge,
    rotation,
    voyager_sclk_open,
    with_pointing,
)


def voyager_ring_edges() -> dict[str, Any]:
    """A Voyager success on a Huygens gap edge, at an image size with no search limit.

    Its size has no configured extfov margin, which is what makes the suspect
    offset section report that a limit could not be resolved for it.

    Returns:
        The document.
    """
    inventory = [
        ring_edge(
            'huygens_gap',
            'IEG',
            reliability=0.64,
            gated=False,
            gate_reason=None,
            bbox=(212, 96, 588, 704),
        ),
        ring_edge(
            'huygens_gap',
            'OEG',
            reliability=0.19,
            gated=True,
            gate_reason='reliability_0.190_below_threshold_0.300',
            bbox=(215, 98, 592, 708),
        ),
    ]
    per_technique = [
        NavTechniqueResult(
            technique_name='RingEdgeNav',
            feature_ids=('ring_edge:SATURN:huygens_gap:IEG',),
            offset_px=(-2.7, 4.55),
            covariance_px2=np.diag([0.0961, 0.0784]),
            confidence=0.66,
            spurious=False,
            at_edge=False,
            diagnostics=RingEdgeDiagnostics(
                total_edge_length_px=612.0,
                per_edge_dt_rms_summed=0.341,
                per_edge_dt_rms_mean=0.341,
                per_edge_dt_median_max=0.288,
                edge_count=1,
                is_rank_1=False,
                lm_converged=True,
                coarse_peak_fraction=0.703,
                sigma_orbit_radial_px=0.118,
            ),
        )
    ]
    result = NavResult.success(
        offset_px=(-2.75, 4.5),
        covariance_px2=np.diag([0.0961, 0.0784]),
        confidence=0.66,
        confidence_rank=derive_confidence_rank(confidence=0.66, sigma_px=(0.31, 0.28)),
        status_reason=NavStatusReason.OK,
        per_technique=per_technique,
        feature_inventory=inventory,
        image_classifier=classifier(noise_sigma=0.75, max_dn=212.0, gradient_score=0.914),
        provenance=provenance(
            image_et=-660000000.0,
            kernels=VGISS_KERNELS,
            extractors=('rings:SATURN', 'stars'),
        ),
        consensus_techniques=['RingEdgeNav'],
    )
    result = with_pointing(
        result,
        camera='NAC',
        instrument='vgiss',
        midtime_et=-660000000.0,
        sclk_open=voyager_sclk_open(13854, 55),
        corrected=rotation(112.447, 3.918, -64.220),
        original=rotation(112.438, 3.911, -64.213),
    )
    return navigated(
        result,
        image_name='C1385455_GEOMED.IMG',
        instrument='vgiss',
        camera='NAC',
        shutter_mode=None,
        image_shape=(800, 800),
        public_metadata=_public_metadata(
            result,
            image_name='C1385455_GEOMED.IMG',
            camera='NAC',
            image_shape=(800, 800),
            filter_name='CLEAR',
            # The counts of the real C1385455_GEOMED label, in VGISS_5101.
            label_counts=('13854:54:794', '13854:55:001'),
        ),
        start=datetime(2026, 8, 8, 16, 47, 33, 799265, tzinfo=UTC),
        elapsed_s=12.5,
        peak_memory_bytes=3221225472,
    )


def voyager_no_features() -> dict[str, Any]:
    """A Voyager failure in which no extractor produced a feature at all.

    An empty inventory is what ``no_features_extracted`` means, and it is the
    scene the failure taxonomy classifies as holding no features.

    The minor-frame field this image is named for fills its own modulus of 60,
    so the reading it names carries into the frame count and its clock strings
    read ``13855:00`` rather than ``13854:60``.  That is the same instant
    written the only way the clock writes it; a real Voyager image number would
    have been the carried one.

    Returns:
        The document.
    """
    result = NavResult.failed(
        status_reason=NavStatusReason.NO_FEATURES_EXTRACTED,
        image_classifier=classifier(noise_sigma=0.75, max_dn=96.0, gradient_score=0.622),
        provenance=provenance(
            image_et=-659999000.0,
            kernels=VGISS_KERNELS,
            extractors=('rings:SATURN', 'stars'),
        ),
    )
    result = with_pointing(
        result,
        camera='WAC',
        instrument='vgiss',
        midtime_et=-659999000.0,
        sclk_open=voyager_sclk_open(13854, 60),
        corrected=None,
        original=rotation(113.005, 4.212, -63.884),
    )
    return navigated(
        result,
        image_name='C1385460_GEOMED.IMG',
        instrument='vgiss',
        camera='WAC',
        shutter_mode=None,
        image_shape=(1024, 1024),
        public_metadata=_public_metadata(
            result,
            image_name='C1385460_GEOMED.IMG',
            camera='WAC',
            image_shape=(1024, 1024),
            filter_name='GREEN',
            # No archive label has this image number, so the counts are made up in the
            # real form: the stop count is the readout frame's, at line 001, and the start
            # count lies one 1.44 s exposure before that frame.
            label_counts=('13854:59:777', '13855:00:001'),
        ),
        start=datetime(2026, 8, 8, 16, 47, 46, 612907, tzinfo=UTC),
        elapsed_s=8.25,
        peak_memory_bytes=2684354560,
    )


def _public_metadata(
    result: NavResult,
    *,
    image_name: str,
    camera: str,
    image_shape: tuple[int, int],
    filter_name: str,
    label_counts: tuple[str, str],
) -> dict[str, Any]:
    """Return what the Voyager ISS host publishes about one Voyager 1 image of this run.

    The host publishes the PDS3 label's start and stop clock counts through its own
    conversion, which is used here too, and no midtime count, since a Voyager label's stop
    count is that of the frame the image was read out in.  The counts are given in the
    label's own form rather than taken from the attitude block's clock strings: those are
    SPICE's readings at the exposure epochs, and would bracket the exposure where a real
    label's counts do not.

    Parameters:
        result: The image's result, carrying its attitude solution.
        image_name: Basename of the source image.
        camera: ``NAC`` or ``WAC``.
        image_shape: The loaded image's ``(v, u)`` pixel dimensions.
        filter_name: The filter the label records.
        label_counts: The label's ``SPACECRAFT_CLOCK_START_COUNT`` and
            ``SPACECRAFT_CLOCK_STOP_COUNT``.

    Returns:
        The published facts, in the host's own key order.
    """
    exposure = recorded_exposure(result)
    return {
        'image_path': holdings_path(image_name).as_posix(),
        'image_name': image_name,
        'instrument_host_lid': 'urn:nasa:pds:context:instrument_host:spacecraft.vg1',
        'instrument_lid': f'urn:nasa:pds:context:instrument:vg1.iss{camera[0].lower()}',
        **published_times(exposure),
        **_published_sclk(*label_counts),
        'image_shape_xy': (image_shape[1], image_shape[0]),
        'camera': camera,
        'exposure_time': exposure.exposure_s,
        'filters': [filter_name],
    }
