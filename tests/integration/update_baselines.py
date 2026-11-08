"""Refresh per-image regression baselines from live navigation results.

Developer tool — not a user-facing CLI.  Runs the autonomous orchestrator
against one or more library sidecars and writes the resulting rounded
``(offset_dv_px, offset_du_px, confidence)`` triples to
``tests/integration/baselines/<image_id>.json``.

Usage (run from the project checkout):

    python -m tests.integration.update_baselines --all
    python -m tests.integration.update_baselines --image-id ID [...]
    python -m tests.integration.update_baselines --all --dry-run

Requirements:

* ``PDS3_HOLDINGS_DIR`` must be set (the orchestrator needs holdings
  access to navigate each image).
* Must be invoked from a project checkout — the module imports
  :mod:`tests.integration.sidecar` and :mod:`tests.integration.baseline`
  for schema and discovery.

Exit codes:

* ``0`` — every requested image succeeded (created / updated / unchanged).
* ``1`` — one or more images failed (orchestrator returned no offset, or
  the named ``--image-id`` did not match any sidecar).
* ``2`` — argument-parsing error / preconditions not met (handled by
  :mod:`argparse` and the precondition checks below).
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from tests.integration.checkout_guard import require_checkout_package

if TYPE_CHECKING:  # pragma: no cover - typing-only imports
    from tests.integration.baseline import Baseline
    from tests.integration.sidecar import LibraryRoot, Sidecar


@dataclass(frozen=True)
class _ImageOutcome:
    """Per-image result row reported on stdout and used for the exit code."""

    image_id: str
    kind: str  # 'CREATE' | 'UPDATE' | 'UNCHANGED' | 'FAILED'
    detail: str

    def line(self) -> str:
        """Format the outcome as ``<KIND>  <image_id>  <detail>`` for stdout."""
        return f'{self.kind:9s} {self.image_id}  {self.detail}'


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse CLI arguments.

    Parameters:
        argv: Argument list (excluding the program name).

    Returns:
        ``argparse.Namespace`` with the validated fields.

    Exits:
        2 (via argparse) when ``--all`` and ``--image-id`` are both
        omitted, or when both are supplied (mutually exclusive).
    """
    parser = argparse.ArgumentParser(
        prog='python -m tests.integration.update_baselines',
        description=(
            'Run the autonomous orchestrator against library sidecars and '
            'write rounded baseline JSON files under '
            'tests/integration/baselines/.'
        ),
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        '--all',
        action='store_true',
        help='Process every discovered sidecar (every image_id under '
        'tests/integration/image_library/images/).',
    )
    selection.add_argument(
        '--image-id',
        action='append',
        default=[],
        metavar='ID',
        help='Process only the named sidecar; may be repeated to process a hand-picked batch.',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Compute the new baseline for each image but do not write '
        'any files; print would-create / would-update / unchanged.',
    )
    parser.add_argument(
        '--baselines-dir',
        type=Path,
        default=None,
        help='Override the destination directory for baseline JSON '
        'files (default: tests/integration/baselines under the project '
        'checkout).',
    )
    parser.add_argument(
        '--library-root',
        type=Path,
        default=None,
        help='Override the library root used for sidecar discovery '
        '(default: tests/integration/image_library under the project '
        'checkout).',
    )
    return parser.parse_args(argv)


def select_sidecars(
    library: LibraryRoot,
    *,
    use_all: bool,
    image_ids: list[str],
) -> tuple[list[Sidecar], list[str]]:
    """Discover sidecars and filter by ``--image-id`` when given.

    Parameters:
        library: Library root (sidecars live under ``library.images``).
        use_all: ``True`` when ``--all`` was supplied.
        image_ids: ``--image-id`` values from the CLI; ignored when
            ``use_all`` is True.

    Returns:
        A tuple ``(sidecars, missing_ids)``.  ``missing_ids`` lists any
        ``--image-id`` value that did not match a discovered sidecar
        (empty when ``use_all`` is True).
    """
    from tests.integration.sidecar import load_sidecar

    every = [load_sidecar(p) for p in library.discover_sidecar_paths()]
    if use_all:
        return every, []
    by_id = {s.image_id: s for s in every}
    selected: list[Sidecar] = []
    missing: list[str] = []
    for image_id in image_ids:
        sidecar = by_id.get(image_id)
        if sidecar is None:
            missing.append(image_id)
        else:
            selected.append(sidecar)
    return selected, missing


def update_one(sidecar: Sidecar, *, baselines_dir: Path, dry_run: bool) -> _ImageOutcome:
    """Run the orchestrator on one sidecar and write its baseline.

    Parameters:
        sidecar: Validated sidecar.
        baselines_dir: Destination directory for ``<image_id>.json``.
        dry_run: When True, compute the new baseline but do not write.

    Returns:
        An ``_ImageOutcome`` describing what changed.
    """
    # Local imports keep the module importable without holdings / oops.
    from filecache import FCPath

    from spindoctor.dataset.dataset import ImageFile, ImageFiles
    from spindoctor.navigate_image_files import navigate_image_files
    from tests.integration.baseline import (
        Baseline,
        baseline_path,
        load_baseline,
    )
    from tests.integration.test_autonomous_nav import (
        _MISSION_TO_OBS_CLASS,
        _resolve_pds3_url,
    )

    obs_class = _MISSION_TO_OBS_CLASS[sidecar.mission]
    image_url = _resolve_pds3_url(sidecar.image_url)
    with tempfile.TemporaryDirectory() as scratch:
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
            FCPath(scratch),
            write_output_files=False,
        )
    offset = metadata.get('offset')
    if offset is None:
        return _ImageOutcome(
            image_id=sidecar.image_id,
            kind='FAILED',
            detail='orchestrator produced no offset',
        )
    confidence = float(metadata.get('confidence', 0.0))
    fresh = Baseline.from_run(
        image_id=sidecar.image_id,
        offset_px=(float(offset[0]), float(offset[1])),
        confidence=confidence,
    )
    target = baseline_path(baselines_dir, sidecar.image_id)
    if target.is_file():
        existing = load_baseline(target)
        if existing == fresh:
            return _ImageOutcome(sidecar.image_id, 'UNCHANGED', _format_baseline(fresh))
        detail = _format_baseline_diff(existing, fresh)
        kind = 'UPDATE'
    else:
        kind = 'CREATE'
        detail = _format_baseline(fresh)
    if not dry_run:
        baselines_dir.mkdir(parents=True, exist_ok=True)
        target.write_text(fresh.to_json())
    return _ImageOutcome(sidecar.image_id, kind, detail)


def _format_baseline(b: Baseline) -> str:
    """One-line summary of a baseline, used for CREATE / UNCHANGED rows."""
    return f'dv={b.offset_dv_px:+.4f} du={b.offset_du_px:+.4f} confidence={b.confidence:.3f}'


def _format_baseline_diff(old: Baseline, new: Baseline) -> str:
    """Field-level diff between two baselines, used for UPDATE rows."""
    parts: list[str] = []
    if old.offset_dv_px != new.offset_dv_px:
        parts.append(f'dv {old.offset_dv_px:+.4f} -> {new.offset_dv_px:+.4f}')
    if old.offset_du_px != new.offset_du_px:
        parts.append(f'du {old.offset_du_px:+.4f} -> {new.offset_du_px:+.4f}')
    if old.confidence != new.confidence:
        parts.append(f'conf {old.confidence:.3f} -> {new.confidence:.3f}')
    return ', '.join(parts) if parts else '(no field-level diff?)'


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``python -m tests.integration.update_baselines``.

    Parameters:
        argv: Optional argument list (defaults to ``sys.argv[1:]``);
            primarily used by tests.

    Returns:
        Exit code: 0 on full success, 1 when at least one image failed
        or a requested ``--image-id`` matched no sidecar.
    """
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if not os.environ.get('PDS3_HOLDINGS_DIR'):
        print(
            'update_baselines: PDS3_HOLDINGS_DIR is not set; cannot '
            'navigate images without the holdings cache',
            file=sys.stderr,
        )
        return 2
    # Local import so the module can be imported without the test stack
    # (e.g. by the unit tests that exercise parse_args).
    from tests.integration.sidecar import LibraryRoot

    library = LibraryRoot() if args.library_root is None else LibraryRoot(root=args.library_root)
    baselines_dir = (
        Path(args.baselines_dir) if args.baselines_dir is not None else library.baselines
    )
    sidecars, missing = select_sidecars(
        library, use_all=bool(args.all), image_ids=list(args.image_id)
    )
    if missing:
        for image_id in missing:
            print(f'FAILED    {image_id}  no matching sidecar in library', file=sys.stderr)
    if not sidecars:
        print(
            'update_baselines: no sidecars selected (library empty?)',
            file=sys.stderr,
        )
        return 1 if missing else 0
    print(
        f'Updating {len(sidecars)} baseline(s) under {baselines_dir}'
        + (' [DRY RUN]' if args.dry_run else '')
    )
    failed = bool(missing)
    summary: dict[str, int] = {'CREATE': 0, 'UPDATE': 0, 'UNCHANGED': 0, 'FAILED': 0}
    for sidecar in sidecars:
        outcome = _safe_update(sidecar, baselines_dir=baselines_dir, dry_run=args.dry_run)
        print(outcome.line())
        summary[outcome.kind] += 1
        if outcome.kind == 'FAILED':
            failed = True
    print(
        'Summary: '
        + ', '.join(f'{count} {kind.lower()}' for kind, count in summary.items() if count)
    )
    return 1 if failed else 0


def _safe_update(sidecar: Sidecar, *, baselines_dir: Path, dry_run: bool) -> _ImageOutcome:
    """Wrap :func:`update_one` so a single image's exception does not
    abort the whole batch.

    The orchestrator captures most failures internally, but a corrupt
    image-file fetch or an `oops` parse error can still raise.  When
    that happens, log the exception text against the offending image
    and continue with the next sidecar so the operator gets a complete
    summary.
    """
    try:
        return update_one(sidecar, baselines_dir=baselines_dir, dry_run=dry_run)
    except Exception as exc:
        return _ImageOutcome(
            image_id=sidecar.image_id,
            kind='FAILED',
            detail=f'{type(exc).__name__}: {exc}',
        )


if __name__ == '__main__':
    require_checkout_package(__file__)
    sys.exit(main())
