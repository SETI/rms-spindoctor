"""Shared fixtures and results-tree factories for the ``tests/spindoctor`` subtree.

The document factories build metadata documents in the shape
``navigate_image_files`` writes, so a test that cares about one field does not
have to restate the surrounding document, and the writers put them where a walk
of a results tree finds them.  They live here rather than beside any one
consumer because a navigation document is the vocabulary the whole subtree
shares: the record seam reads one, the index ingests one, the reprojection
stage takes its pointing out of one, and a dataset filters on what one records.

The index helpers build a real index over a real tree, because the ingest
guarantees that matter -- what is keyed by what, what is read a second time --
are properties of the walk and the writer together.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pdslogger
import pytest

from spindoctor.cli.results_index import IngestCounts, ingest_metadata_files
from spindoctor.nav_records import TreeTuning
from spindoctor.results_index import open_index
from spindoctor.support.cmatrix import AttitudeBaseline, PointingSolution


@pytest.fixture
def sentinel_pointing() -> PointingSolution:
    """Build a PointingSolution a wiring test can identify by reference.

    Returns:
        A solution whose attitudes are identity matrices and whose baseline
        names the Cassini narrow angle camera.
    """
    baseline = AttitudeBaseline(
        cmatrix_original=np.eye(3),
        oops_from_spice=np.eye(3),
        camera_frame='CASSINI_ISS_NAC',
        camera_frame_id=-82360,
        ck_frame_id=-82000,
        start_et=1.0,
        stop_et=2.0,
        midtime_et=1.5,
        exposure_s=1.0,
        sclk_start='1/1.000',
        sclk_midtime='1/1.500',
        sclk_stop='1/2.000',
    )
    return PointingSolution(baseline=baseline, cmatrix=np.eye(3))


@pytest.fixture
def fakes_report_as_simulated(monkeypatch: pytest.MonkeyPatch) -> None:
    """Report a module's fake observations as simulated.

    ``obs_class_to_inst_name`` cannot identify a test fake and returns
    ``'unknown'``, which the orchestrator treats as a build defect and warns
    about.  The fakes in the modules requesting this fixture stand in for an
    observation carrying no SPICE camera frame, which is exactly what a
    simulated image is, so they report that instead of shaping the production
    set around the test suite.

    Deliberately not autouse: each module whose fakes reach the orchestrator
    opts in with its own one-line autouse wrapper, so the patch never touches
    tests that exercise the real instrument registry.
    """
    monkeypatch.setattr(
        'spindoctor.nav_orchestrator.orchestrator.obs_class_to_inst_name', lambda cls: 'sim'
    )


def technique(
    name: str,
    offset: tuple[float, float],
    *,
    confidence: float = 0.7,
    spurious: bool = False,
    at_edge: bool = False,
) -> dict[str, Any]:
    """Build one ``per_technique`` entry.

    Parameters:
        name: Technique class name.
        offset: The technique's ``(dv, du)`` estimate.
        confidence: The technique's calibrated confidence.
        spurious: Whether the technique flagged its own result as spurious.
        at_edge: Whether the fit landed at the edge of its search space.

    Returns:
        The entry.
    """
    return {
        'technique_name': name,
        'feature_ids': [f'{name.lower()}:IAPETUS'],
        'offset_px': list(offset),
        'covariance_px2': [[0.01, 0.0], [0.0, 0.01]],
        'confidence': confidence,
        'spurious': spurious,
        'at_edge': at_edge,
        'diagnostics': {'a': 1},
    }


def metadata_document(
    *,
    image_name: str = 'N1454725799_1_CALIB.IMG',
    instrument: str | None = 'coiss',
    camera: str | None = 'NAC',
    status: str = 'success',
    status_reason: str | None = 'ok',
    status_error: str | None = None,
    status_traceback: str | None = None,
    offset: list[float] | None = None,
    confidence: float = 0.8,
    confidence_rank: str = 'high',
    per_technique: list[dict[str, Any]] | None = None,
    excluded: list[str] | None = None,
    image_et: float | None = 0.0,
    image_shape: list[int] | None = None,
    elapsed_s: float | None = 3.25,
    times: dict[str, Any] | None = None,
    pointing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a metadata document in the ``navigate_image_files`` shape.

    Parameters:
        image_name: Recorded ``observation.image_name``.
        instrument: Recorded ``observation.instrument``; None omits the field,
            which models a file that is not a navigation document.
        camera: Recorded ``observation.camera``; None omits the field, as
            happens for an image that never loaded.
        status: Top-level status.
        status_reason: The navigator's explanation; None omits the field.
        status_error: The fatal error; None omits the field.
        status_traceback: The traceback of that error; None omits the field.
        offset: The authoritative top-level offset; None omits it, and a
            successful document defaults to one.
        confidence: Top-level confidence.
        confidence_rank: Recorded confidence tier.
        per_technique: Technique entries, from :func:`technique`.
        excluded: Technique names the ensemble excluded.
        image_et: Recorded provenance epoch.
        image_shape: Recorded ``observation.image_shape``; None omits it.
        elapsed_s: Recorded run time; None omits the whole timing section.
        times: Recorded ``navigation_result.times``; None omits the block, as
            it is for an image whose host has no SPICE camera frame.
        pointing: Recorded ``navigation_result.pointing``; None omits the
            block likewise.

    Returns:
        The document.
    """
    if offset is None and status == 'success':
        offset = [1.5, -2.5]
    observation: dict[str, Any] = {
        'image_path': f'/holdings/{image_name}',
        'image_name': image_name,
    }
    if instrument is not None:
        observation['instrument'] = instrument
    if camera is not None:
        observation['camera'] = camera
    if image_shape is not None:
        observation['image_shape'] = image_shape
    entries = [] if per_technique is None else per_technique
    navigation_result: dict[str, Any] = {
        'status': status,
        'offset_px': offset,
        'sigma_px': [0.1, 0.2] if offset is not None else None,
        'confidence': confidence,
        'confidence_rank': confidence_rank,
        'covariance_px2': [[0.01, 0.0], [0.0, 0.04]] if offset is not None else None,
        'techniques_used': sorted({t['technique_name'] for t in entries}),
        'excluded_from_consensus': [] if excluded is None else excluded,
        'per_technique': entries,
        'feature_inventory': [
            {
                'feature_id': 'body_disc:IAPETUS',
                'feature_type': 'BODY_DISC',
                'source_model': 'body:IAPETUS',
                'gated': False,
            },
            {
                'feature_id': 'star:UCAC4:10230452',
                'feature_type': 'STAR',
                'source_model': 'stars',
                'gated': True,
            },
        ],
        'image_classifier': {'class': 'clean', 'noise_sigma': 1.0, 'max_dn': 255.0},
        'provenance': {
            'spindoctor_git_sha': 'abc1234',
            'config_hash': 'deadbeef',
            'image_et': image_et,
            'pipeline_run_iso8601': '2026-07-11T00:00:00Z',
        },
    }
    if status_reason is not None:
        navigation_result['status_reason'] = status_reason
    if times is not None:
        navigation_result['times'] = times
    if pointing is not None:
        navigation_result['pointing'] = pointing
    document: dict[str, Any] = {
        'status': status,
        'observation': observation,
        'navigation_result': navigation_result,
        'confidence': confidence,
    }
    if offset is not None:
        document['offset'] = list(offset)
    if status_error is not None:
        document['status_error'] = status_error
    if status_traceback is not None:
        document['status_traceback'] = status_traceback
    if elapsed_s is not None:
        document['timing'] = {
            'start_iso8601': '2026-07-11T00:00:00Z',
            'end_iso8601': '2026-07-11T00:00:03.250000Z',
            'elapsed_s': elapsed_s,
        }
    return document


def write_metadata(root: Path, stub: str, document: dict[str, Any]) -> Path:
    """Write one metadata document into a results tree.

    Parameters:
        root: The results root.
        stub: The document's results path stub under that root.
        document: The document to write.

    Returns:
        The path written.
    """
    path = root / f'{stub}_metadata.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding='utf-8')
    return path


def write_summary_png(root: Path, stub: str) -> Path:
    """Write a stand-in summary PNG beside a document.

    A results tree holds one of these beside every navigated image, so it is
    the file a walk of a real root meets most often after the documents
    themselves.  Only its name matters here: nothing opens one.

    Parameters:
        root: The results root.
        stub: The image's results path stub.

    Returns:
        The path written.
    """
    path = root / f'{stub}_summary.png'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'\x89PNG\r\n\x1a\n')
    return path


REFUSED_DOCUMENT = '{"edges": []}'
"""A document that reads as JSON and is not a navigation result of any schema."""


def write_refusal(root: Path, stub: str) -> Path:
    """Write a document under a root that no pass can turn into an image row.

    Parameters:
        root: The results root to write under.
        stub: The document's results path stub under that root.

    Returns:
        The path written.
    """
    path = root / f'{stub}_metadata.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(REFUSED_DOCUMENT, encoding='utf-8')
    return path


def index_url(path: Path) -> str:
    """Return the SQLite URL naming an index file.

    Parameters:
        path: The database file's path.

    Returns:
        The URL.
    """
    return f'sqlite:///{path.as_posix()}'


def ingest_tree(
    url: str,
    roots: list[Path],
    *,
    logger: pdslogger.PdsLogger,
    force: bool = False,
    tuning: TreeTuning | None = None,
) -> IngestCounts:
    """Create an index and ingest one or more results trees into it.

    Parameters:
        url: The index URL to create or add to.
        roots: The results roots to walk.
        logger: Logger the ingest reports through.
        force: Whether to re-read every document.
        tuning: How much of the pass runs at once, or None for the library's
            defaults.  A test wanting a small chunk or batch passes one, which
            is what a program does.

    Returns:
        What the pass did.
    """
    engine = open_index(url, create=True)
    try:
        return ingest_metadata_files(
            engine,
            [root.as_posix() for root in roots],
            force=force,
            logger=logger,
            tuning=TreeTuning() if tuning is None else tuning,
        )
    finally:
        engine.dispose()
