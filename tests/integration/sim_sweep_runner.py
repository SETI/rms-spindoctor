"""Run the sim parameter sweeps and emit per-step JSON (Phase T3).

Invoked as ``python -m tests.integration.sim_sweep_runner`` from a clean
checkout.  For every spec under ``tests/integration/sim_sweeps/`` it navigates
each sweep step and writes ``tests/integration/sim_sweeps/results/<name>.json``
containing one row per value: ``value``, ``status``, ``offset_error_px``,
``confidence``, and ``primary_technique``.  The JSON is a human-readable
diagnostic artifact (response curves), not a re-blessed regression baseline --
``test_sim_sweeps`` asserts the invariants directly.

Pass ``--dump-images DIR`` to additionally render every sweep step to a viewable
PNG under ``DIR/<sweep_name>/`` -- the frames the sweep navigates -- so the
collection of images behind a response curve can be inspected by eye.  Pass
``--only NAME`` to restrict to one sweep and ``--no-navigate`` to dump images
without running navigation or writing JSON/figures.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from spindoctor.sim.png_export import render_scene_png
from tests.integration.checkout_guard import require_checkout_package
from tests.integration.sim_sweep import (
    build_sweep_params,
    iter_sweep_paths,
    load_sweep,
    run_sweep,
)

_SWEEPS_ROOT = Path(__file__).parent / 'sim_sweeps'
_RESULTS_ROOT = _SWEEPS_ROOT / 'results'


def _dump_sweep_images(
    spec_name: str, steps: list[tuple[float, dict[str, Any]]], out_root: Path
) -> int:
    """Render every step of one sweep to a PNG under ``out_root/<spec_name>/``.

    Parameters:
        spec_name: The sweep name (used as the per-sweep subdirectory).
        steps: The ``(value, sim_params)`` pairs from ``build_sweep_params``.
        out_root: The image-dump root directory.

    Returns:
        The number of PNGs written.
    """
    sweep_dir = out_root / spec_name
    for index, (value, sim_params) in enumerate(steps):
        # Render what the sweep navigates (offset applied), lifting dim features
        # with a mild gamma so a crescent or a faint field stays visible.
        render_scene_png(
            sim_params,
            sweep_dir / f'{index:02d}_value_{value:g}.png',
            ignore_offset=False,
            gamma=1.4,
            upscale=2,
        )
    return len(steps)


def main(argv: list[str] | None = None) -> int:
    """Render + navigate every sweep, write its per-step JSON, and plot figures."""
    parser = argparse.ArgumentParser(description='Run the sim parameter sweeps.')
    parser.add_argument(
        '--dump-images',
        metavar='DIR',
        type=Path,
        default=None,
        help='also render every sweep step to a viewable PNG under DIR/<sweep_name>/',
    )
    parser.add_argument(
        '--only',
        metavar='NAME',
        default=None,
        help='restrict to the sweep whose name (filename stem) is NAME',
    )
    parser.add_argument(
        '--no-navigate',
        action='store_true',
        help='only dump images (skip navigation, JSON, and figures)',
    )
    args = parser.parse_args(argv)

    sweep_paths = iter_sweep_paths(_SWEEPS_ROOT)
    if args.only is not None:
        sweep_paths = [p for p in sweep_paths if p.stem == args.only]
    if not sweep_paths:
        print('No sweeps found; nothing to do.', file=sys.stderr)
        return 1

    if args.dump_images is not None:
        total = 0
        for sweep_path in sweep_paths:
            spec = load_sweep(sweep_path)
            total += _dump_sweep_images(spec.sweep_name, build_sweep_params(spec), args.dump_images)
        print(f'Wrote {total} sweep image(s) to {args.dump_images}')

    if args.no_navigate:
        return 0

    _RESULTS_ROOT.mkdir(exist_ok=True)
    for sweep_path in sweep_paths:
        spec = load_sweep(sweep_path)
        rows = run_sweep(spec)
        document = {
            'sweep_name': spec.sweep_name,
            'base_scene': str(spec.base_scene.relative_to(_SWEEPS_ROOT.parent / 'sim_scenes')),
            'parameters': list(spec.parameters),
            'rows': [asdict(row) for row in rows],
        }
        out_path = _RESULTS_ROOT / f'{spec.sweep_name}.json'
        out_path.write_text(json.dumps(document, indent=2, sort_keys=True) + '\n')
        techniques = [row.primary_technique for row in rows]
        print(f'{spec.sweep_name}: {len(spec.values)} steps -> {techniques}')
    print(f'Wrote {len(sweep_paths)} sweep result(s) to {_RESULTS_ROOT}')
    try:
        from tests.integration.sim_sweep_plots import generate_plots
    except ImportError:
        print('matplotlib not available; skipping figures.', file=sys.stderr)
        return 0
    figures = generate_plots()
    print(f'Wrote {len(figures)} figure(s) to {figures[0].parent if figures else "(none)"}')
    return 0


if __name__ == '__main__':
    require_checkout_package(__file__)
    raise SystemExit(main())
