"""Regression-baseline tests for the per-image library.

Two layers, both under ``tests/integration/``:

- **Citations** — every baseline JSON file must cite an existing sidecar
  (no orphaned baselines).  Runs in the fast suite without holdings
  access.
- **Per-image regression** — for each baseline, run the orchestrator
  against the real holdings and assert exact-equality on the rounded
  ``(offset_dv_px, offset_du_px, confidence)`` triple.  Gated by the
  ``integration`` marker and skipped when ``PDS3_HOLDINGS_DIR`` is unset.

Baselines are populated incrementally; an empty ``baselines/`` directory
is the valid Phase-4 starting state.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
from filecache import FCPath

from tests.integration.baseline import (
    Baseline,
    discover_baseline_paths,
    load_baseline,
)
from tests.integration.sidecar import (
    LibraryRoot,
    Sidecar,
    load_sidecar,
)


@pytest.fixture(scope='module')
def library() -> LibraryRoot:
    return LibraryRoot()


# ---------------------------------------------------------------------------
# Citations: every baseline JSON points at an existing sidecar
# ---------------------------------------------------------------------------


def test_every_baseline_cites_a_sidecar(library: LibraryRoot) -> None:
    """A baseline at ``baselines/<image_id>.json`` requires a sidecar
    ``image_library/images/*/<image_id>.yaml``.

    Catches the common drift where a sidecar is renamed or deleted but
    its baseline lingers behind.
    """
    sidecar_ids = {load_sidecar(p).image_id for p in library.discover_sidecar_paths()}
    for baseline_path in discover_baseline_paths(library.baselines):
        baseline = load_baseline(baseline_path)
        # Filename convention: stem matches image_id.
        assert baseline_path.stem == baseline.image_id, (
            f'{baseline_path}: filename stem does not match image_id={baseline.image_id!r}'
        )
        assert baseline.image_id in sidecar_ids, (
            f'{baseline_path}: image_id {baseline.image_id!r} has no '
            f'matching sidecar under image_library/images/'
        )


# ---------------------------------------------------------------------------
# Round-trip / serialization unit tests (do not require holdings)
# ---------------------------------------------------------------------------


def test_baseline_from_run_rounds_offsets_to_4_decimals() -> None:
    """``from_run`` rounds offset_px to 4 decimals (Part 0 §17)."""
    b = Baseline.from_run(
        image_id='X',
        offset_px=(1.23456789, -0.987654321),
        confidence=0.123456,
    )
    assert b.offset_dv_px == 1.2346
    assert b.offset_du_px == -0.9877


def test_baseline_from_run_rounds_confidence_to_3_decimals() -> None:
    """``from_run`` rounds confidence to 3 decimals (Part 0 §17)."""
    b = Baseline.from_run(image_id='X', offset_px=(0.0, 0.0), confidence=0.123456)
    assert b.confidence == 0.123


def test_baseline_json_roundtrip(tmp_path: Path) -> None:
    """A baseline serialized and reloaded from disk compares equal."""
    b = Baseline(
        image_id='ROUNDTRIP_001',
        offset_dv_px=12.3456,
        offset_du_px=-7.8901,
        confidence=0.875,
    )
    p = tmp_path / 'ROUNDTRIP_001.json'
    p.write_text(b.to_json())
    assert load_baseline(p) == b


def test_baseline_json_is_deterministic() -> None:
    """JSON serialization is byte-stable (sorted keys + trailing newline)."""
    b1 = Baseline(image_id='A', offset_dv_px=1.0, offset_du_px=2.0, confidence=0.5)
    b2 = Baseline(image_id='A', offset_dv_px=1.0, offset_du_px=2.0, confidence=0.5)
    assert b1.to_json() == b2.to_json()
    assert b1.to_json().endswith('\n')


# ---------------------------------------------------------------------------
# Per-image regression (real holdings)
# ---------------------------------------------------------------------------

_HAS_HOLDINGS = bool(os.environ.get('PDS3_HOLDINGS_DIR'))


def _baseline_pairs() -> list[tuple[Baseline, Sidecar]]:
    """Pair every baseline with its sidecar; skip baselines whose sidecar
    has been removed (the citations test would have already failed)."""
    library = LibraryRoot()
    sidecars = {load_sidecar(p).image_id: load_sidecar(p) for p in library.discover_sidecar_paths()}
    pairs: list[tuple[Baseline, Sidecar]] = []
    for baseline_path in discover_baseline_paths(library.baselines):
        baseline = load_baseline(baseline_path)
        if baseline.image_id in sidecars:
            pairs.append((baseline, sidecars[baseline.image_id]))
    return pairs


def pytest_generate_tests(metafunc: Any) -> None:
    """Parametrize the regression test — one case per (baseline, sidecar) pair."""
    if 'pair' not in metafunc.fixturenames:
        return
    pairs = _baseline_pairs()
    metafunc.parametrize('pair', pairs, ids=[b.image_id for b, _ in pairs])


@pytest.mark.integration
@pytest.mark.skipif(not _HAS_HOLDINGS, reason='PDS3_HOLDINGS_DIR unset')
def test_regression_baseline_exact_match(pair: tuple[Baseline, Sidecar], tmp_path: Path) -> None:
    """Recompute the rounded baseline and assert exact equality.

    ``pipeline_run_iso8601`` is the only provenance field allowed to vary
    between runs (Part 0 §11), and the baseline schema does not include
    it — so this is a true byte-equal comparison on every recorded
    output.
    """
    # Local imports keep the citations / unit tests importable without
    # the heavy obs / orchestrator stack.
    from spindoctor.dataset.dataset import ImageFile, ImageFiles
    from spindoctor.navigate_image_files import navigate_image_files
    from tests.integration.test_autonomous_nav import (
        _MISSION_TO_OBS_CLASS,
        _resolve_pds3_url,
    )

    expected, sidecar = pair
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
    _success, metadata = navigate_image_files(
        obs_class,
        image_files,
        FCPath(str(tmp_path)),
        write_output_files=False,
    )
    offset = metadata.get('offset')
    confidence = metadata.get('confidence', 0.0)
    assert offset is not None, (
        f'{sidecar.image_id}: orchestrator produced no offset; cannot compare against baseline'
    )
    actual = Baseline.from_run(
        image_id=sidecar.image_id,
        offset_px=(float(offset[0]), float(offset[1])),
        confidence=float(confidence),
    )
    assert actual == expected, (
        f'{sidecar.image_id}: regression baseline mismatch\n'
        f'  expected: {expected}\n'
        f'  actual:   {actual}\n'
        f'  if this change is intended, update '
        f'tests/integration/baselines/{sidecar.image_id}.json '
        f'in the same PR (operator review required)'
    )
