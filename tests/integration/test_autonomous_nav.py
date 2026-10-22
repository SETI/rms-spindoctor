"""End-to-end per-image regression test against the operator-curated library.

For each sidecar discovered under ``tests/integration/image_library/images/``
this test runs ``navigate_image_files`` against the real PDS3 holdings
(via ``PDS3_HOLDINGS_DIR``) and asserts:

(a) ``NavResult.status`` exact-match against ``expected.status``.
(b) For ``ok`` results, ``offset_px`` is within
    ``offset_uncertainty_px + 0.5 px`` slack of the operator-supplied
    ground-truth on each axis.  Checked before the tier, so a frame whose
    offset is wrong is reported as wrong rather than as a tier change.
(c) ``NavResult.confidence_rank`` exact-match against ``expected.confidence_tier``.
(d) ``expected.primary_technique`` is the highest-confidence per-technique
    result with ties broken by ``(-confidence, technique_name)`` ascending
    (Part 0 §14; deterministic, registration-order-independent).
(e) Every name in ``techniques_must_run`` appears in ``per_technique``;
    no name in ``techniques_must_skip`` does.

The whole module is gated by ``pytestmark = pytest.mark.integration`` and
skipped automatically when ``PDS3_HOLDINGS_DIR`` is not set.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import pytest
from filecache import FCPath

pytestmark = pytest.mark.integration

if 'PDS3_HOLDINGS_DIR' not in os.environ:
    pytest.skip(
        'PDS3_HOLDINGS_DIR is not set; skipping autonomous-nav integration tests',
        allow_module_level=True,
    )

from spindoctor.dataset.dataset import ImageFile, ImageFiles  # noqa: E402  (guarded import)
from spindoctor.navigate_image_files import navigate_image_files  # noqa: E402
from spindoctor.obs import (  # noqa: E402
    ObsCassiniISS,
    ObsGalileoSSI,
    ObsNewHorizonsLORRI,
    ObsSnapshotInst,
    ObsVoyagerISS,
)
from tests.integration.sidecar import (  # noqa: E402
    LibraryRoot,
    Sidecar,
    load_sidecar,
)

# Mission-string -> Obs class.  Keys match the sidecar schema's ``mission`` enum
# (upper-cased dataset names from :mod:`spindoctor.dataset`).
_MISSION_TO_OBS_CLASS: dict[str, type[ObsSnapshotInst]] = {
    'COISS': ObsCassiniISS,
    'VGISS': ObsVoyagerISS,
    'GOSSI': ObsGalileoSSI,
    'NHLORRI': ObsNewHorizonsLORRI,
}


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def pytest_generate_tests(metafunc: Any) -> None:
    """Parametrize ``test_one_library_image`` — one case per discovered sidecar."""
    if 'sidecar' not in metafunc.fixturenames:
        return
    paths = LibraryRoot().discover_sidecar_paths()
    sidecars = [load_sidecar(p) for p in paths]
    metafunc.parametrize('sidecar', sidecars, ids=[s.image_id for s in sidecars])


# ---------------------------------------------------------------------------
# Per-image regression
# ---------------------------------------------------------------------------


def test_one_library_image(sidecar: Sidecar, tmp_path: Path) -> None:
    """Run the orchestrator end-to-end and check the four assertion blocks."""
    obs_class = _MISSION_TO_OBS_CLASS[sidecar.mission]
    image_url = _resolve_pds3_url(sidecar.image_url)
    image_files = ImageFiles(
        image_files=[
            ImageFile(
                image_file_url=image_url,
                label_file_url=image_url,
                results_path_stub=sidecar.image_id,
            )
        ]
    )
    success, metadata = navigate_image_files(
        obs_class,
        image_files,
        FCPath(str(tmp_path)),
        write_output_files=False,
    )
    nav_meta = metadata.get('navigation_result') or {}

    # (a) status
    actual_status = nav_meta.get('status') or metadata.get('status')
    assert actual_status == sidecar.expected.status, (
        f'{sidecar.image_id}: expected status={sidecar.expected.status}, got {actual_status}'
    )
    expected_success = sidecar.expected.status == 'success'
    assert success == expected_success, (
        f'{sidecar.image_id}: success={success!r} disagrees with status={actual_status!r} '
        f'(expected success={expected_success!r} for expected.status='
        f'{sidecar.expected.status!r})'
    )

    # (b) offset within slack (only for ``ok`` outcomes), before the tier so a
    # wrong offset is not reported as a tier change.  A rank-1
    # constrained ground truth compares only the component along the
    # constraint normal — the along-edge component is physically
    # unobservable, so per-axis comparison would score noise.
    if sidecar.expected.status == 'success':
        offset = metadata.get('offset')
        assert offset is not None, f'{sidecar.image_id}: status=ok but metadata carries no offset'
        constraint = sidecar.ground_truth.constraint
        if constraint is not None:
            angle = math.radians(constraint.normal_angle_deg)
            actual_along_normal = float(offset[0]) * math.cos(angle) + float(offset[1]) * math.sin(
                angle
            )
            slack = constraint.uncertainty_px + 0.5
            normal_err = abs(actual_along_normal - constraint.offset_along_normal_px)
            assert normal_err <= slack, (
                f'{sidecar.image_id}: edge-normal offset error {normal_err:.3f} px '
                f'exceeds tolerance {slack:.3f} px (constraint normal at '
                f'{constraint.normal_angle_deg:.2f} deg)'
            )
        else:
            slack = sidecar.ground_truth.offset_uncertainty_px + 0.5
            dv_err = abs(float(offset[0]) - sidecar.ground_truth.offset_dv_px)
            du_err = abs(float(offset[1]) - sidecar.ground_truth.offset_du_px)
            assert dv_err <= slack, (
                f'{sidecar.image_id}: dv error {dv_err:.3f} px exceeds tolerance {slack:.3f} px'
            )
            assert du_err <= slack, (
                f'{sidecar.image_id}: du error {du_err:.3f} px exceeds tolerance {slack:.3f} px'
            )

    # (c) confidence_rank (no slack, exact match), plus the optional
    # status_reason pin (e.g. ``rank_1_only`` for a flat-ring frame).
    actual_rank = nav_meta.get('confidence_rank')
    assert actual_rank == sidecar.expected.confidence_tier, (
        f'{sidecar.image_id}: expected confidence_tier='
        f'{sidecar.expected.confidence_tier}, got {actual_rank}'
    )
    if sidecar.expected.status_reason is not None:
        actual_reason = nav_meta.get('status_reason')
        assert actual_reason == sidecar.expected.status_reason, (
            f'{sidecar.image_id}: expected status_reason='
            f'{sidecar.expected.status_reason}, got {actual_reason}'
        )

    per_technique = nav_meta.get('per_technique', [])
    technique_names = [entry.get('technique_name') for entry in per_technique]

    # (d) primary technique = highest confidence, tie-break by name ascending.
    if sidecar.expected.status == 'success' and per_technique:
        primary = _primary_technique(per_technique)
        assert primary == sidecar.expected.primary_technique, (
            f'{sidecar.image_id}: expected primary_technique='
            f'{sidecar.expected.primary_technique}, got {primary} '
            f'(per_technique={technique_names})'
        )

    # (e) must-run / must-skip set membership
    for name in sidecar.expected.techniques_must_run:
        assert name in technique_names, (
            f'{sidecar.image_id}: technique {name!r} did not produce a '
            f'result (per_technique={technique_names})'
        )
    for name in sidecar.expected.techniques_must_skip:
        assert name not in technique_names, (
            f'{sidecar.image_id}: technique {name!r} unexpectedly produced '
            f'a result (per_technique={technique_names})'
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_pds3_url(url: str) -> FCPath:
    """Resolve a sidecar's ``pds3://...`` opaque URL through ``PDS3_HOLDINGS_DIR``.

    The ``pds3://`` scheme is implicit relative to the holdings root; any
    other scheme is passed through to ``FCPath`` so direct ``https://`` or
    ``gs://`` URLs work too (useful for one-off debugging).
    """
    if url.startswith('pds3://'):
        rel = url[len('pds3://') :]
        holdings_root = os.environ['PDS3_HOLDINGS_DIR'].rstrip('/')
        return FCPath(f'{holdings_root}/{rel}')
    return FCPath(url)


def _primary_technique(per_technique: list[dict[str, Any]]) -> str:
    """Pick the highest-confidence technique with a deterministic tie-break.

    Tie-break is ``(-confidence, technique_name)`` ascending so the result
    is independent of registration order (Part 0 §14).
    """
    ordered = sorted(
        per_technique,
        key=lambda entry: (-float(entry.get('confidence', 0.0)), str(entry.get('technique_name'))),
    )
    return str(ordered[0].get('technique_name'))
