"""Collect planted-truth rows: navigate randomized Titan scenes, record error.

For every generated scene (see ``scene_gen``) this pins ``TitanHazeNav``,
navigates in-process (the sim needs no external holdings), and writes one JSON
line per frame carrying what the accuracy claim is measured from:

- the planted ``(offset_v, offset_u)`` ground truth and the recovered offset;
- the error resolved onto the technique's OWN symmetry axis -- cross-track and
  along-track -- because the two axes have different bounds and an unresolved
  Euclidean error would hide which one moved;
- the reported one-sigma on each of those axes, so the z-scores that calibrate
  ``cross_sigma_scale`` / ``along_sigma_scale`` can be formed;
- confidence, the at-edge and spurious flags, and the named gate that rejected
  the frame when one did.

Run (from an activated project venv; ``source /seti/newnav/setup.sh``)::

    venv/bin/python util/titan_truth/collect.py \\
        --per-family 100 --workers 8 --out _work/titan_truth/rows.jsonl

The campaign seed defaults to 20260725; pass ``--seed`` to draw a different
campaign.  Rows are written incrementally so a partial run is still usable.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import multiprocessing
import os
import sys
import time
from pathlib import Path
from typing import Any

# Pin native (BLAS/OpenMP) thread pools to one thread per process before the
# first numpy import, which happens transitively through ``scene_gen`` below.
# The worker pool uses the default fork start method, so each worker inherits
# the parent's already-initialized threading runtime: setting these variables
# in the pool initializer would run after that runtime is configured and have
# no effect.  ``setdefault`` leaves an explicit operator override in place.
for _thread_var in (
    'OMP_NUM_THREADS',
    'OPENBLAS_NUM_THREADS',
    'MKL_NUM_THREADS',
    'NUMEXPR_NUM_THREADS',
):
    os.environ.setdefault(_thread_var, '1')

HERE = Path(__file__).parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / 'src'))

from scene_gen import FAMILIES, generate_scenes  # noqa: E402

DEFAULT_SEED = 20260725

# Scalar haze-structure keys recorded per row for per-axis attribution.
# ``cloud_blobs`` is deliberately absent: it is a list, not a scalar, so it
# cannot be correlated against an error the way the others can.
_STRUCTURE_KEYS: frozenset[str] = frozenset(
    {
        'axis_tilt_deg',
        'ns_falloff_ratio',
        'ns_asymmetry_amplitude',
        'sector_sharpness_gradient',
        'interior_ramp_amplitude',
    }
)
DEFAULT_OUT = REPO / '_work/titan_truth/rows.jsonl'
TECHNIQUE = 'TitanHazeNav'


def _json_safe(value: Any) -> Any:
    """Coerce a diagnostics value to something ``json.dumps`` accepts.

    Parameters:
        value: A diagnostics field value.

    Returns:
        The value itself when JSON-encodable, a tagged mapping for a
        non-finite float, and its ``repr`` otherwise.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return {'__nonfinite__': repr(value)}
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    return repr(value)


def _axis_unit_vectors(theta_rad: float) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return ``(c_hat, a_hat)`` in ``(v, u)`` for a symmetry-axis angle.

    Repeated here rather than imported so the analysis path states the sign
    convention it measures against explicitly: ``a_hat`` points toward the
    sub-solar side and ``c_hat`` is the positive cross-track direction.

    Parameters:
        theta_rad: The symmetry-axis angle in radians.

    Returns:
        ``(c_hat, a_hat)`` as ``(v, u)`` pairs.
    """
    sin_t, cos_t = math.sin(theta_rad), math.cos(theta_rad)
    return (cos_t, -sin_t), (sin_t, cos_t)


def _axis_sigma(covariance: Any, axis_vu: tuple[float, float]) -> float | None:
    """Return the reported one-sigma along a unit direction, or None.

    ``sigma^2 = n^T Sigma n`` for a unit vector ``n``: the covariance the
    ensemble consumes, model-error floor included, projected onto the axis the
    error was resolved onto.

    Parameters:
        covariance: The technique's covariance array.
        axis_vu: A ``(v, u)`` unit direction.

    Returns:
        The one-sigma in pixels, or None when the covariance is unusable.
    """
    import numpy as np

    try:
        cov = np.asarray(covariance, dtype=float)[:2, :2]
    except (TypeError, ValueError):
        return None
    n = np.asarray(axis_vu, dtype=float)
    variance = float(n @ cov @ n)
    if not math.isfinite(variance) or variance <= 0.0:
        return None
    return math.sqrt(variance)


def _navigate_one(task: tuple[str, str, dict[str, Any]]) -> dict[str, Any]:
    """Worker: navigate one scene and return its planted-truth row."""
    scene_id, family, sim_params = task
    from spindoctor.nav_model import build_models_for_obs
    from spindoctor.nav_orchestrator import NavOrchestrator
    from spindoctor.nav_technique.diagnostics import TitanHazeDiagnostics
    from spindoctor.obs.obs_inst_sim import ObsSim

    planted_v = float(sim_params.get('offset_v', 0.0))
    planted_u = float(sim_params.get('offset_u', 0.0))
    body = sim_params['bodies'][0]
    row: dict[str, Any] = {
        'scene_id': scene_id,
        'family': family,
        'planted': {'dv': planted_v, 'du': planted_u},
        'geometry': {
            'r_solid_px': float(body['semi_a_px']),
            'phase_deg': float(body['phase_angle']),
            'illumination_deg': float(body['illumination_angle']),
            'read_noise_dn': float(sim_params['noise']['read_noise_dn']),
        },
        # The haze structure keys the scene planted, so an error can be
        # attributed to the axis that caused it rather than to the family it
        # happened to be drawn in.  A family varies several keys at once, and
        # "the asymmetry family is worse" is not a finding until it says WHICH
        # key made it worse.
        'structure': {
            key: value
            for key, value in (body.get('atmosphere') or {}).items()
            if key in _STRUCTURE_KEYS
        },
    }
    try:
        obs = ObsSim.from_file(f'/tmp/{scene_id}.json', sim_params=sim_params)
        result = NavOrchestrator(
            build_models_for_obs(obs), only_models='*', only_techniques=TECHNIQUE
        ).navigate(obs)
    except Exception as exc:
        row['error'] = f'{type(exc).__name__}: {exc}'
        return row

    row['status'] = str(result.status)
    row['status_reason'] = str(result.status_reason)
    pinned = next((t for t in result.per_technique if t.technique_name == TECHNIQUE), None)
    if pinned is None:
        # The feature never reached the technique: the reliability gate
        # removed it, or the model built nothing.  A legible outcome, not an
        # error -- recorded so the gated fraction is measurable.
        row['outcome'] = 'no_technique_result'
        return row
    diagnostics = pinned.diagnostics
    if not isinstance(diagnostics, TitanHazeDiagnostics):
        # Unreachable while the run pins TitanHazeNav; a mismatched payload
        # would silently corrupt every axis-resolved quantity below, so it is
        # recorded as an error rather than coerced.
        row['error'] = f'unexpected diagnostics type {type(diagnostics).__name__}'
        return row
    row['confidence'] = float(pinned.confidence)
    row['spurious'] = bool(pinned.spurious)
    row['at_edge'] = bool(pinned.at_edge)
    # The full diagnostics payload, so a candidate confidence spec can be
    # scored against this campaign offline: the no-confident-wrong criterion
    # is a statement about (confidence, error) pairs, and re-deriving the
    # confidence of a proposed anchor set needs the terms it reads.
    row['diagnostics'] = {
        field.name: _json_safe(getattr(diagnostics, field.name))
        for field in dataclasses.fields(diagnostics)
    }
    row['gate_failed'] = diagnostics.gate_failed
    row['phase_deg'] = float(diagnostics.phase_deg)
    row['envelope_diameter_px'] = float(diagnostics.envelope_diameter_px)
    row['axis_degenerate'] = bool(diagnostics.axis_degenerate)
    row['recentered'] = bool(diagnostics.recentered)
    if pinned.spurious or pinned.offset_px is None:
        row['outcome'] = 'spurious'
        return row
    row['outcome'] = 'committed'
    row['recovered'] = {'dv': float(pinned.offset_px[0]), 'du': float(pinned.offset_px[1])}
    error_v = float(pinned.offset_px[0]) - planted_v
    error_u = float(pinned.offset_px[1]) - planted_u
    c_hat, a_hat = _axis_unit_vectors(math.radians(float(diagnostics.sun_angle_deg)))
    row['error_px'] = {
        'cross': error_v * c_hat[0] + error_u * c_hat[1],
        'along': error_v * a_hat[0] + error_u * a_hat[1],
        'total': math.hypot(error_v, error_u),
    }
    row['sigma_px'] = {
        'cross': _axis_sigma(pinned.covariance_px2, c_hat),
        'along': _axis_sigma(pinned.covariance_px2, a_hat),
    }
    return row


def _init_worker() -> None:
    """Silence the per-image and main loggers in a pool worker.

    Native thread-pool pinning is handled at module import (before the first
    numpy import, so every forked worker inherits it); doing it here would be
    too late under the fork start method.
    """
    import pdslogger

    from spindoctor.config import IMAGE_LOGGER, MAIN_LOGGER

    for logger in (IMAGE_LOGGER, MAIN_LOGGER):
        logger.remove_all_handlers()
        logger.add_handler(pdslogger.NULL_HANDLER)


def _model_error_floor_px() -> float:
    """Return the configured covariance model-error floor for the technique."""
    from spindoctor.nav_technique import TitanHazeNav
    from spindoctor.nav_technique.nav_technique import load_model_error_floor

    TitanHazeNav()  # populates the class-level tuning from config
    return load_model_error_floor(TitanHazeNav.tuning, TECHNIQUE)


def _sigma_scales() -> dict[str, float]:
    """Return the configured cross- and along-track sigma scales."""
    from spindoctor.config import DEFAULT_CONFIG

    navigation = DEFAULT_CONFIG.titan.navigation
    return {
        'cross': navigation.symmetry.cross_sigma_scale,
        'along': navigation.arc.along_sigma_scale,
    }


def _sigma_floors() -> dict[str, float]:
    """Return the configured per-axis sigma floors.

    Recorded in the manifest because the scale calibration has to know which
    rows are censored: a fit whose sigma was clamped to its floor carries no
    information about the multiplier applied before the clamp.
    """
    from spindoctor.config import DEFAULT_CONFIG

    navigation = DEFAULT_CONFIG.titan.navigation
    return {
        'cross': navigation.symmetry.sigma_floor_cross_px,
        'along': navigation.arc.sigma_floor_along_px,
    }


def main(argv: list[str] | None = None) -> int:
    """Generate scenes, navigate them in a worker pool, write JSONL rows."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--per-family', type=int, default=100)
    parser.add_argument('--families', default=','.join(FAMILIES))
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    families = [f.strip() for f in args.families.split(',') if f.strip()]
    tasks: list[tuple[str, str, dict[str, Any]]] = []
    for family in families:
        for scene_id, sim_params in generate_scenes(
            family, args.per_family, campaign_seed=args.seed
        ):
            tasks.append((scene_id, family, sim_params))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    done = 0
    failed = 0
    with args.out.open('w') as out:
        out.write(
            json.dumps(
                {
                    'manifest': True,
                    'campaign_seed': args.seed,
                    'per_family': args.per_family,
                    'families': families,
                    'n_scenes': len(tasks),
                    'technique': TECHNIQUE,
                    'model_error_floor_px': _model_error_floor_px(),
                    'sigma_scales': _sigma_scales(),
                    'sigma_floors': _sigma_floors(),
                }
            )
            + '\n'
        )
        with multiprocessing.Pool(
            processes=args.workers, initializer=_init_worker, maxtasksperchild=200
        ) as pool:
            for row in pool.imap_unordered(_navigate_one, tasks, chunksize=4):
                out.write(json.dumps(row, sort_keys=True) + '\n')
                done += 1
                if 'error' in row:
                    failed += 1
                if done % 100 == 0:
                    elapsed = time.monotonic() - start
                    rate = done / elapsed
                    remaining = (len(tasks) - done) / rate if rate > 0 else 0
                    print(
                        f'{done}/{len(tasks)} rows ({failed} errors), '
                        f'{rate:.1f}/s, ~{remaining / 60:.1f} min left',
                        flush=True,
                    )
    elapsed = time.monotonic() - start
    print(f'Wrote {done} rows ({failed} errors) to {args.out} in {elapsed / 60:.1f} min')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
