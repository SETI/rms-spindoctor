"""Shift-equivariance sweep: does each technique return the shift it is handed?

For every operator-curated success sidecar in
``tests/integration/image_library/images/`` this navigates the image once
as-is (the reference run) and once per planted pointing shift, applying the
shift to the observation's FOV with the same ``apply_offset_to_obs`` helper
the mosaic / backplane consumers use to apply a measured navigation offset
(``oops.fov.OffsetFOV``).  That is a faithful in-process analogue of the
corrected-C-kernel round trip: furnishing a corrected kernel moves the
predicted geometry by the measured offset exactly as ``OffsetFOV`` does,
without the kernel-writing machinery.

A perfectly shift-equivariant technique satisfies
``offset(shift) = offset(0) - shift`` (offset convention: predicted
position ``(v, u)`` means actual position is ``(v + dv, u + du)``; applying
a measured offset via ``OffsetFOV`` therefore re-navigates to zero).  The
per-row residual reported here is ``offset(shift) - offset(0) + shift``,
which is exactly the round-trip residual the corrected-pointing validation
measures.

One JSON line is written per (image, shift) navigation so a partial run is
still usable; the companion report step aggregates residuals per technique.

Needs the local-holdings environment (``source /seti/newnav/setup.sh``).

Run:

    venv/bin/python util/calibration/shift_equivariance.py \
        --workers 8 --out _work/nav447/shift_equivariance.jsonl

    venv/bin/python util/calibration/shift_equivariance.py \
        --report _work/nav447/shift_equivariance.jsonl \
        --out _work/nav447/shift_equivariance.md
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# Pin native (BLAS/OpenMP) thread pools to one thread per process before the
# first numpy import (see collect.py for the fork-inheritance rationale).
for _thread_var in (
    'OMP_NUM_THREADS',
    'OPENBLAS_NUM_THREADS',
    'MKL_NUM_THREADS',
    'NUMEXPR_NUM_THREADS',
):
    os.environ.setdefault(_thread_var, '1')

HERE = Path(__file__).parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'src'))

# Default planted shifts (dv, du) in pixels.  Chosen to cover sub-pixel
# phases (the DT edge mask and any correlation grid quantize at integer or
# rational fractions of a pixel), the ~1.9 px magnitude of the round-trip
# finding, and multi-pixel magnitudes out to the typical search window.
DEFAULT_SHIFTS: list[tuple[float, float]] = [
    (0.30, -0.20),
    (0.50, 0.50),
    (1.86, 0.00),
    (-2.60, 1.30),
    (5.50, -3.25),
    (-10.20, 7.40),
]


def _navigate_one(task: tuple[str, tuple[float, float] | None]) -> dict[str, Any]:
    """Navigate one library image under one planted shift (or none) in a pool worker.

    Parameters:
        task: ``(sidecar_path, shift_vu)`` pair; ``shift_vu`` is the planted
            ``(dv, du)`` pointing shift in pixels, or ``None`` for the
            reference (as-is) navigation.

    Returns:
        One JSONL-ready row with the image identity, the planted shift, the
        elapsed time, the ensemble offset, and the per-technique offsets with
        their spurious / at-edge flags and confidences.  A navigation that
        raises returns a row carrying an ``error`` string instead of results.
    """
    import time

    from filecache import FCPath
    from tests.integration.sidecar import load_sidecar

    from spindoctor.cli.reproj.offsets import apply_offset_to_obs
    from spindoctor.nav_model import build_models_for_obs
    from spindoctor.nav_orchestrator.orchestrator import NavOrchestrator
    from spindoctor.obs import (
        ObsCassiniISS,
        ObsGalileoSSI,
        ObsNewHorizonsLORRI,
        ObsVoyagerISS,
    )

    mission_to_obs = {
        'COISS': ObsCassiniISS,
        'VGISS': ObsVoyagerISS,
        'GOSSI': ObsGalileoSSI,
        'NHLORRI': ObsNewHorizonsLORRI,
    }
    sidecar_path_str, shift = task
    sidecar = load_sidecar(Path(sidecar_path_str))
    row: dict[str, Any] = {
        'image_id': sidecar.image_id,
        'scene_class': sidecar.scene_tags[0] if sidecar.scene_tags else '?',
        'mission': sidecar.mission,
        'camera': sidecar.camera,
        'shift_vu': list(shift) if shift is not None else None,
    }
    url = sidecar.image_url
    if url.startswith('pds3://'):
        holdings_root = os.environ['PDS3_HOLDINGS_DIR'].rstrip('/')
        url = f'{holdings_root}/{url[len("pds3://") :]}'
    start = time.time()
    try:
        obs = mission_to_obs[sidecar.mission].from_file(FCPath(url))
        if shift is not None:
            apply_offset_to_obs(obs, shift[0], shift[1])
        orchestrator = NavOrchestrator(build_models_for_obs(obs))
        result = orchestrator.navigate(obs)
    except Exception as exc:
        row['error'] = f'{type(exc).__name__}: {exc}'
        return row
    row['elapsed_s'] = round(time.time() - start, 1)
    row['status'] = result.status
    row['ensemble_offset_vu'] = (
        [float(result.offset_px[0]), float(result.offset_px[1])]
        if result.offset_px is not None
        else None
    )
    row['per_technique'] = [
        {
            'technique': r.technique_name,
            'offset_vu': [float(r.offset_px[0]), float(r.offset_px[1])],
            'spurious': bool(r.spurious),
            'at_edge': bool(r.at_edge),
            'confidence': round(float(r.confidence), 4),
        }
        for r in result.per_technique
    ]
    return row


def _init_worker() -> None:
    """Silence the navigation loggers in each pool worker.

    Same rationale as ``collect.py``: the per-image log stream from many
    concurrent navigations is noise here, and the null handler keeps
    pdslogger from writing through an inherited handler after ``fork``.
    """
    import pdslogger

    from spindoctor.config import IMAGE_LOGGER, MAIN_LOGGER

    for logger in (IMAGE_LOGGER, MAIN_LOGGER):
        logger.remove_all_handlers()
        logger.add_handler(pdslogger.NULL_HANDLER)


def _read_jsonl_rows(path: Path) -> tuple[list[dict[str, Any]], bool]:
    """Read a sweep JSONL file, tolerating an interrupted final record.

    A collection run can be killed mid-write, leaving a partial final line
    that would otherwise break both resume and report mode (and a resumed
    append would then corrupt the file further).  The final line is therefore
    allowed to be unparseable and is dropped; an unparseable line anywhere
    else means the file is corrupt and raises.

    Parameters:
        path: The JSONL file to read.

    Returns:
        ``(rows, dropped_partial)``: the parsed rows, and whether a partial
        final line was dropped.

    Raises:
        ValueError: A non-final line failed to parse, with its line number.
    """
    lines = path.read_text().splitlines()
    while lines and not lines[-1].strip():
        lines.pop()
    rows: list[dict[str, Any]] = []
    for lineno, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            if lineno == len(lines):
                return rows, True
            raise ValueError(f'{path}:{lineno}: corrupt JSONL record: {exc}') from exc
    return rows, False


def _collect(args: argparse.Namespace) -> int:
    """Run the sweep and write one JSON line per (image, shift) navigation.

    Parameters:
        args: Parsed CLI namespace; uses ``images`` / ``shifts`` to select
            the task set, ``workers`` for the pool width, ``resume`` to skip
            rows already present in ``out``, and ``out`` for the JSONL path.

    Returns:
        Process exit status: 0 on success, 1 when the holdings environment
        is not configured.
    """
    if not os.environ.get('PDS3_HOLDINGS_DIR'):
        print('PDS3_HOLDINGS_DIR must be set', file=sys.stderr)
        return 1

    from tests.integration.sidecar import LibraryRoot, load_sidecar

    paths: list[str] = []
    for path in LibraryRoot().discover_sidecar_paths():
        sidecar = load_sidecar(Path(path))
        if sidecar.expected.status != 'success':
            continue
        if args.images and sidecar.image_id not in args.images:
            continue
        paths.append(str(path))
    shifts: list[tuple[float, float] | None] = [None]
    shifts += [(dv, du) for dv, du in (args.shifts or DEFAULT_SHIFTS)]
    tasks = [(p, s) for p in paths for s in shifts]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.resume and args.out.exists():
        rows, dropped_partial = _read_jsonl_rows(args.out)
        if dropped_partial:
            # Rewrite the file without the interrupted final record so the
            # coming append starts on a clean line.
            args.out.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            print('resume: dropped an interrupted partial final record')
        have = set()
        for row in rows:
            shift = tuple(row['shift_vu']) if row['shift_vu'] is not None else None
            have.add((row['image_id'], shift))

        def _key(task: tuple[str, tuple[float, float] | None]) -> tuple[str, Any]:
            from tests.integration.sidecar import load_sidecar as _load

            return (_load(Path(task[0])).image_id, task[1])

        tasks = [t for t in tasks if _key(t) not in have]
        print(f'resume: {len(have)} rows already collected')
    print(f'{len(paths)} images x {len(shifts)} runs = {len(tasks)} navigations to do')
    done = 0
    with (
        # maxtasksperchild=1: one navigation per worker process, so every
        # observation, model, and derivative image is released to the OS
        # before the next frame starts (nothing accumulates across frames).
        multiprocessing.Pool(
            processes=args.workers, initializer=_init_worker, maxtasksperchild=1
        ) as pool,
        args.out.open('a' if args.resume else 'w') as fp,
    ):
        for row in pool.imap_unordered(_navigate_one, tasks):
            fp.write(json.dumps(row) + '\n')
            fp.flush()
            done += 1
            print(f'{done}/{len(tasks)} {row["image_id"]} shift={row["shift_vu"]}', flush=True)
    print(f'Wrote {args.out}')
    return 0


def _residual_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pair shifted runs with their reference run and compute residuals.

    The residual is ``offset(shift) - offset(0) + shift`` per axis: zero for
    a perfectly shift-equivariant technique.  Only techniques present in
    both runs produce a residual row; spurious flags from both runs are
    carried so the report can separate trusted from rejected fits.

    Parameters:
        rows: Sweep rows as written by the collector, mixing reference
            (``shift_vu is None``) and shifted navigations for any number of
            images; rows carrying an ``error`` key are skipped.

    Returns:
        One residual row per (image, shift, technique) pair present in both
        the shifted and the reference run, plus an ``ENSEMBLE`` row per
        (image, shift) where both ensemble offsets exist.
    """
    reference: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row['shift_vu'] is None and 'error' not in row:
            reference[row['image_id']] = row
    out: list[dict[str, Any]] = []
    for row in rows:
        if row['shift_vu'] is None or 'error' in row:
            continue
        ref = reference.get(row['image_id'])
        if ref is None:
            continue
        shift_v, shift_u = row['shift_vu']
        ref_by_name = {t['technique']: t for t in ref['per_technique']}
        for tech in row['per_technique']:
            ref_tech = ref_by_name.get(tech['technique'])
            if ref_tech is None:
                continue
            res_v = tech['offset_vu'][0] - ref_tech['offset_vu'][0] + shift_v
            res_u = tech['offset_vu'][1] - ref_tech['offset_vu'][1] + shift_u
            out.append(
                {
                    'image_id': row['image_id'],
                    'scene_class': row['scene_class'],
                    'technique': tech['technique'],
                    'shift_vu': [shift_v, shift_u],
                    'residual_vu': [res_v, res_u],
                    'spurious_ref': ref_tech['spurious'],
                    'spurious_shifted': tech['spurious'],
                }
            )
        if row['ensemble_offset_vu'] is not None and ref['ensemble_offset_vu'] is not None:
            out.append(
                {
                    'image_id': row['image_id'],
                    'scene_class': row['scene_class'],
                    'technique': 'ENSEMBLE',
                    'shift_vu': [shift_v, shift_u],
                    'residual_vu': [
                        row['ensemble_offset_vu'][0] - ref['ensemble_offset_vu'][0] + shift_v,
                        row['ensemble_offset_vu'][1] - ref['ensemble_offset_vu'][1] + shift_u,
                    ],
                    'spurious_ref': False,
                    'spurious_shifted': False,
                }
            )
    return out


def _quantile(sorted_values: list[float], q: float) -> float:
    """Nearest-rank quantile of an already-sorted list.

    Parameters:
        sorted_values: Values in ascending order; may be empty.
        q: Quantile in ``[0, 1]``.

    Returns:
        The nearest-rank element, or ``NaN`` for an empty list.
    """
    if not sorted_values:
        return float('nan')
    idx = min(len(sorted_values) - 1, max(0, round(q * (len(sorted_values) - 1))))
    return sorted_values[idx]


def _report(args: argparse.Namespace) -> int:
    """Aggregate a sweep JSONL into the per-technique residual report.

    Parameters:
        args: Parsed CLI namespace; ``report`` names the input JSONL and
            ``out`` the markdown report to write.

    Returns:
        Process exit status: always 0.
    """
    rows, dropped_partial = _read_jsonl_rows(args.report)
    if dropped_partial:
        print('report: dropped an interrupted partial final record')
    residuals = _residual_rows(rows)
    lines = [
        '# Shift-equivariance sweep (planted OffsetFOV shifts vs reference run)',
        '',
        f'{len(rows)} navigations, {len(residuals)} paired technique residuals.',
        'Residual = offset(shift) - offset(0) + shift; 0 for an equivariant technique.',
        '',
    ]
    by_tech: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for res in residuals:
        by_tech[res['technique']].append(res)
    lines += [
        '| technique | n (both non-spurious) | med |r| v/u | p90 |r| v/u | max |r| v/u '
        '| mean r v/u | spurious flips |',
        '|---|---|---|---|---|---|---|',
    ]
    for tech in sorted(by_tech):
        entries = by_tech[tech]
        clean = [e for e in entries if not e['spurious_ref'] and not e['spurious_shifted']]
        flips = sum(1 for e in entries if e['spurious_ref'] != e['spurious_shifted'])
        if not clean:
            lines.append(f'| {tech} | 0 | - | - | - | - | {flips} |')
            continue
        abs_v = sorted(abs(e['residual_vu'][0]) for e in clean)
        abs_u = sorted(abs(e['residual_vu'][1]) for e in clean)
        mean_v = sum(e['residual_vu'][0] for e in clean) / len(clean)
        mean_u = sum(e['residual_vu'][1] for e in clean) / len(clean)
        lines.append(
            f'| {tech} | {len(clean)} '
            f'| {_quantile(abs_v, 0.5):.4f} / {_quantile(abs_u, 0.5):.4f} '
            f'| {_quantile(abs_v, 0.9):.4f} / {_quantile(abs_u, 0.9):.4f} '
            f'| {abs_v[-1]:.4f} / {abs_u[-1]:.4f} '
            f'| {mean_v:+.4f} / {mean_u:+.4f} | {flips} |'
        )
    lines += ['', '## Residual vs shift magnitude (non-spurious pairs)', '']
    lines += [
        '| technique | shift (dv, du) | n | med |r| v/u | max |r| v/u |',
        '|---|---|---|---|---|',
    ]
    for tech in sorted(by_tech):
        by_shift: dict[tuple[float, float], list[dict[str, Any]]] = defaultdict(list)
        for entry in by_tech[tech]:
            if not entry['spurious_ref'] and not entry['spurious_shifted']:
                by_shift[tuple(entry['shift_vu'])].append(entry)
        for shift in sorted(by_shift):
            entries = by_shift[shift]
            abs_v = sorted(abs(e['residual_vu'][0]) for e in entries)
            abs_u = sorted(abs(e['residual_vu'][1]) for e in entries)
            lines.append(
                f'| {tech} | ({shift[0]:+.2f}, {shift[1]:+.2f}) | {len(entries)} '
                f'| {_quantile(abs_v, 0.5):.4f} / {_quantile(abs_u, 0.5):.4f} '
                f'| {abs_v[-1]:.4f} / {abs_u[-1]:.4f} |'
            )
    lines += ['', '## Worst 25 residuals (non-spurious pairs)', '']
    lines += ['| image | technique | shift | residual (v, u) |', '|---|---|---|---|']
    clean_all = [r for r in residuals if not r['spurious_ref'] and not r['spurious_shifted']]
    clean_all.sort(
        key=lambda r: max(abs(r['residual_vu'][0]), abs(r['residual_vu'][1])), reverse=True
    )
    for entry in clean_all[:25]:
        lines.append(
            f'| {entry["image_id"]} | {entry["technique"]} '
            f'| ({entry["shift_vu"][0]:+.2f}, {entry["shift_vu"][1]:+.2f}) '
            f'| ({entry["residual_vu"][0]:+.4f}, {entry["residual_vu"][1]:+.4f}) |'
        )
    errors = [r for r in rows if 'error' in r]
    lines += ['', f'{len(errors)} navigation errors.']
    for row in errors:
        lines.append(f'- {row["image_id"]} shift={row["shift_vu"]}: {row["error"][:100]}')
    lines.append('')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text('\n'.join(lines))
    print(f'Wrote {args.out}')
    return 0


def _parse_shift(text: str) -> tuple[float, float]:
    """Parse a ``dv,du`` command-line shift.

    Parameters:
        text: The raw argument, two comma-separated pixel components (a
            leading space lets a negative ``dv`` pass through argparse).

    Returns:
        The ``(dv, du)`` shift in pixels.

    Raises:
        argparse.ArgumentTypeError: The argument is not two comma-separated
            finite numbers.
    """
    parts = text.split(',')
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f'expected dv,du; got {text!r}')
    try:
        dv, du = float(parts[0]), float(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f'expected dv,du as numbers; got {text!r}') from exc
    if not (math.isfinite(dv) and math.isfinite(du)):
        raise argparse.ArgumentTypeError(f'shift components must be finite; got {text!r}')
    return dv, du


def main(argv: list[str] | None = None) -> int:
    """Dispatch to the sweep collector or the report aggregator.

    Parameters:
        argv: CLI arguments; ``None`` reads ``sys.argv`` as usual.

    Returns:
        Process exit status from the selected mode.
    """
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument(
        '--report',
        type=Path,
        default=None,
        help='Aggregate this sweep JSONL into a markdown report instead of collecting.',
    )
    parser.add_argument(
        '--images',
        nargs='*',
        default=None,
        help='Optional image_id filter for a drill-down run.',
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Skip (image, shift) rows already present in the output JSONL and append.',
    )
    parser.add_argument(
        '--shifts',
        nargs='*',
        type=_parse_shift,
        default=None,
        help='Planted shifts as dv,du pairs (default: the documented sweep).',
    )
    args = parser.parse_args(argv)
    if args.report is not None:
        return _report(args)
    return _collect(args)


if __name__ == '__main__':
    raise SystemExit(main())
