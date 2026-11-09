"""Collect agreement rows: navigate composition scenes, record per-technique offsets.

For every generated scene (see ``scene_gen``) this runs the autonomous
pipeline in-process and writes one JSON line per scene: the planted
(offset_v, offset_u) truth, the scene's geometry angles, and every
per-technique result from each configured run.  Single-body compositions
navigate once with every model; the multi_body composition navigates once
per body (``only_models='body_sim:<NAME>'``) so each body contributes its
own limb/disc estimates; ring-bearing compositions add a blob-only run
(``only_techniques=['BodyBlobNav']``) to supply an extra estimator that
never touches the shared gradient/DT products.

Shared-bias injection (the bias-independence stage) is harness-level only.
Three families of injection exist, all recorded per row and none touching
``src/``:

- Shared preprocessing layer (nav-side monkeypatch of the orchestrator
  module): ``--injection dt_shift`` monkeypatches
  ``compute_all_image_derivatives`` so every DT technique sees the shared
  gradient / edge-distance-transform products translated by a per-scene
  random bias, and ``--injection noise_scale`` monkeypatches
  ``estimate_image_noise_sigma`` to scale the single shared noise estimate.
  Only the distance-transform techniques read those products, so a
  correlation/centroid technique (disc, blob) is decoupled by construction.
- Reliability gate (nav-side monkeypatch of ``FeatureReliabilityGate.apply``):
  ``--injection reliability_gate`` tightens every per-type threshold by a
  fixed depression (``--gate-depression``), so more features are dropped and
  whole techniques stop reporting on the lower-reliability scenes.  Unlike
  the other channels the gate never shifts a surviving technique's offset --
  it only admits or drops -- so it is the pure *selection* channel: pairing a
  gate pass with the control pass by scene id recovers, for every scene, both
  the technique's true (unperturbed) error and whether it survived.
- PSF layer (render-side, via the scene's ``optics.psf`` block):
  ``--injection psf_broaden`` renders the whole-scene PSF with its core sigma
  scaled up by a per-scene factor, and ``--injection psf_aniso`` renders it
  broadened along one axis only (a zero-mean elliptical kernel, so no global
  translation).  The navigator keeps its own modeled PSF sigma, so the
  rendered blurred edge no longer matches the template edge -- the only
  channel through which a shared PSF edge bias can reach both the limb
  (distance-transform) and disc (normalized cross-correlation) estimators.
  These require a PSF-bearing scene family (``limb_disc_psf``), whose control
  render matches the navigator (``optics.psf: {match_navigator: true}``).

Run (from an activated project venv; ``source /seti/newnav/setup.sh``):

    venv/bin/python util/agreement/collect.py \
        --per-family 400 --workers 8 --out _work/agreement/rows.jsonl

The campaign seed defaults to 20260719; pass ``--seed`` to draw a different
campaign.  Rows are written incrementally so a partial run is still usable.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

# Pin native (BLAS/OpenMP) thread pools to one thread per process before the
# first numpy import, which happens transitively through ``scene_gen`` below.
# The worker pool uses the default fork start method, so each worker inherits
# the parent's already-initialized threading runtime: setting these variables
# in the pool initializer runs after that runtime is configured and has no
# effect.  Setting them here, ahead of numpy's first import, makes the parent
# (and therefore every forked worker it inherits from) single-threaded, which
# is what keeps the pool from oversubscribing the machine.  ``setdefault``
# leaves an explicit operator override in place.
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

DEFAULT_SEED = 20260719
DEFAULT_OUT = REPO / '_work/agreement/rows.jsonl'

INJECTION_KINDS = (
    'none',
    'dt_shift',
    'noise_scale',
    'reliability_gate',
    'psf_broaden',
    'psf_aniso',
)

# PSF-layer injection: per-scene multiplicative broadening of the rendered PSF
# core sigma above the navigator's modeled sigma (log-uniform draw).  The
# anisotropic variant applies the factor to one axis only.  The ranges start
# above 1 (a genuine mismatch) and stay inside the sim's resolvable,
# realism-supported PSF regime for the Cassini NAC.
_PSF_BROADEN_RANGE = (1.4, 3.0)
_PSF_ANISO_RANGE = (1.6, 3.5)


def _runs_for_family(family: str) -> list[dict[str, Any]]:
    """Return the orchestrator run configurations for a scene family.

    Parameters:
        family: Scene family name.

    Returns:
        List of run dicts with ``key``, ``only_models``, ``only_techniques``.
    """
    if family == 'multi_body':
        return [
            {'key': 'body:RHEA', 'only_models': 'body_sim:RHEA', 'only_techniques': '*'},
            {'key': 'body:DIONE', 'only_models': 'body_sim:DIONE', 'only_techniques': '*'},
        ]
    runs: list[dict[str, Any]] = [{'key': 'full', 'only_models': '*', 'only_techniques': '*'}]
    if 'ring' in family:
        # BodyBlobNav is a fallback the orchestrator skips whenever a primary
        # body technique produced a result, so the shared-layer-independent
        # blob estimate needs its own restricted run.
        runs.append({'key': 'blob', 'only_models': '*', 'only_techniques': ['BodyBlobNav']})
    return runs


def _draw_injection(
    kind: str, scene_id: str, seed: int, sigma_px: float, gate_depression: float
) -> dict[str, Any]:
    """Draw the per-scene injection values (recorded in the row).

    Parameters:
        kind: One of :data:`INJECTION_KINDS`.
        scene_id: Scene identifier (part of the draw key).
        seed: Campaign seed (part of the draw key).
        sigma_px: Per-axis standard deviation of the dt_shift bias draw.
        gate_depression: Fixed amount every reliability threshold is raised
            by in the ``reliability_gate`` channel (constant across the
            pass, so the per-scene dropout is set by the scene's own feature
            reliabilities rather than a per-scene draw).

    Returns:
        Dict describing the injection (``{'kind': 'none'}`` when disabled).
    """
    if kind == 'none':
        return {'kind': 'none'}
    if kind == 'reliability_gate':
        return {'kind': 'reliability_gate', 'depression': gate_depression}
    rng = random.Random(f'{seed}/{scene_id}/inject')
    if kind == 'dt_shift':
        return {
            'kind': 'dt_shift',
            'bias_v': rng.gauss(0.0, sigma_px),
            'bias_u': rng.gauss(0.0, sigma_px),
        }
    if kind == 'noise_scale':
        return {
            'kind': 'noise_scale',
            'factor': math.exp(rng.uniform(math.log(2.0), math.log(8.0))),
        }
    if kind == 'psf_broaden':
        lo, hi = _PSF_BROADEN_RANGE
        return {
            'kind': 'psf_broaden',
            'factor': math.exp(rng.uniform(math.log(lo), math.log(hi))),
        }
    lo, hi = _PSF_ANISO_RANGE
    return {
        'kind': 'psf_aniso',
        'factor': math.exp(rng.uniform(math.log(lo), math.log(hi))),
        'axis': rng.choice(('v', 'u')),
    }


def _apply_injection(injection: dict[str, Any]) -> list[tuple[Any, str, Any]]:
    """Monkeypatch the shared preprocessing layer per the injection spec.

    The patch targets the orchestrator module's imported references, so
    every technique downstream of the shared per-image products sees the
    injected version while techniques that read the image directly are
    untouched.

    Parameters:
        injection: Draw from :func:`_draw_injection`.

    Returns:
        List of ``(module, attribute, original)`` entries for restoration.
    """
    import numpy as np

    import spindoctor.nav_orchestrator.orchestrator as orch_mod

    patched: list[tuple[Any, str, Any]] = []
    if injection['kind'] == 'dt_shift':
        from scipy.ndimage import shift as nd_shift

        bias = (float(injection['bias_v']), float(injection['bias_u']))
        real_compute = orch_mod.compute_all_image_derivatives  # type: ignore[attr-defined]

        def injected_compute(
            image_ext: Any, image_noise_sigma: float, *, config: Any = None
        ) -> Any:
            gradient, edge_dt, gradient_vu = real_compute(
                image_ext, image_noise_sigma, config=config
            )
            kw = {'order': 1, 'mode': 'nearest'}
            gradient = nd_shift(gradient, bias, **kw)
            edge_dt = nd_shift(edge_dt, bias, **kw)
            gradient_vu = np.stack(
                [
                    nd_shift(gradient_vu[..., 0], bias, **kw),
                    nd_shift(gradient_vu[..., 1], bias, **kw),
                ],
                axis=-1,
            )
            return gradient, edge_dt, gradient_vu

        patched.append((orch_mod, 'compute_all_image_derivatives', real_compute))
        orch_mod.compute_all_image_derivatives = injected_compute  # type: ignore[attr-defined]
    elif injection['kind'] == 'noise_scale':
        factor = float(injection['factor'])
        real_noise = orch_mod.estimate_image_noise_sigma  # type: ignore[attr-defined]

        def injected_noise(image: Any, sensor_mask: Any) -> float:
            return factor * float(real_noise(image, sensor_mask))

        patched.append((orch_mod, 'estimate_image_noise_sigma', real_noise))
        orch_mod.estimate_image_noise_sigma = injected_noise  # type: ignore[attr-defined,assignment]
    elif injection['kind'] == 'reliability_gate':
        from spindoctor.feature.reliability import FeatureReliabilityGate, GatedFeatureRecord

        depression = float(injection['depression'])
        real_apply = FeatureReliabilityGate.apply

        def injected_apply(self: Any, features: list[Any]) -> tuple[list[Any], list[Any]]:
            # Reproduce the real gate logic with every threshold raised by
            # ``depression``; at depression 0 this is byte-for-byte the
            # production admission rule.  Dropping features is all the gate
            # does, so a surviving technique's offset is unchanged.
            kept: list[Any] = []
            gated: list[Any] = []
            for feature in features:
                threshold = self.thresholds.get(feature.feature_type, 0.0) + depression
                if feature.reliability < threshold:
                    gated.append(
                        GatedFeatureRecord(
                            feature=feature,
                            reason=f'reliability_gate_injection_depression_{depression:.3f}',
                        )
                    )
                else:
                    kept.append(feature)
            return kept, gated

        patched.append((FeatureReliabilityGate, 'apply', real_apply))
        FeatureReliabilityGate.apply = injected_apply  # type: ignore[method-assign]
    return patched


def _restore(patched: list[tuple[Any, str, Any]]) -> None:
    """Undo :func:`_apply_injection`."""
    for module, attribute, original in patched:
        setattr(module, attribute, original)


def _inject_sim_params(
    sim_params: dict[str, Any], injection: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the render-side sim_params for a PSF-layer injection.

    The PSF injections act on the *rendered* scene rather than on the
    navigator: they overwrite the scene's ``optics.psf`` block with an
    explicit kernel broadened above the navigator's modeled sigma, while the
    navigator's belief (its ``star_psf_sigma`` config) is untouched, so the
    rendered blurred edge no longer matches the template edge.  ``psf_broaden``
    scales both core sigmas by the drawn factor (isotropic); ``psf_aniso``
    scales one axis only (a zero-mean elliptical kernel that softens the edge
    directionally without translating the image).  Non-PSF injections leave the
    scene unchanged (they act on the nav side; see :func:`_apply_injection`).

    Parameters:
        sim_params: The generated (validated) scene parameters.
        injection: Draw from :func:`_draw_injection`.

    Returns:
        ``(render_params, applied)`` where ``applied`` records the resolved
        kernel sigmas (empty for non-PSF injections).
    """
    kind = injection['kind']
    if not kind.startswith('psf_'):
        return sim_params, {}

    import copy as _copy

    from spindoctor.config import DEFAULT_CONFIG
    from spindoctor.sim.instrument_navigator import navigator_matched_psf
    from spindoctor.sim.scene import validate_sim_params

    params = _copy.deepcopy(dict(sim_params))
    base = navigator_matched_psf(
        DEFAULT_CONFIG, params.get('instrument'), params.get('instrument_config')
    )
    base_sigma = float(base['sigma_v'])
    factor = float(injection['factor'])
    if kind == 'psf_broaden':
        sigma_v = sigma_u = base_sigma * factor
    elif injection.get('axis') == 'v':
        sigma_v, sigma_u = base_sigma * factor, base_sigma
    else:
        sigma_v, sigma_u = base_sigma, base_sigma * factor
    optics = dict(params.get('optics') or {})
    optics['psf'] = {'sigma_v': sigma_v, 'sigma_u': sigma_u, 'w': 0.0, 'r0': 2.0, 'n': 3.0}
    params['optics'] = optics
    render_params = validate_sim_params(params, source='psf_inject')
    applied = {
        'base_sigma_px': base_sigma,
        'sigma_v_px': sigma_v,
        'sigma_u_px': sigma_u,
    }
    return render_params, applied


def _navigate_one(
    task: tuple[str, str, dict[str, Any], dict[str, Any], dict[str, Any]],
) -> dict[str, Any]:
    """Worker: navigate one scene through its configured runs."""
    scene_id, family, sim_params, geometry, injection = task
    import numpy as np

    from spindoctor.nav_model import build_models_for_obs
    from spindoctor.nav_orchestrator import NavOrchestrator
    from spindoctor.obs.obs_inst_sim import ObsSim

    row: dict[str, Any] = {
        'scene_id': scene_id,
        'family': family,
        'planted': {
            'dv': float(sim_params.get('offset_v', 0.0)),
            'du': float(sim_params.get('offset_u', 0.0)),
        },
        'geometry': geometry,
        'injection': injection,
        'runs': {},
    }
    render_params, applied = _inject_sim_params(sim_params, injection)
    if applied:
        row['injection'] = {**injection, **applied}
    patched = _apply_injection(injection)
    try:
        # The path is a label only (sim_params overrides the file read), but
        # FCPath resolves it, so it must sit under a real directory.
        obs = ObsSim.from_file(f'/tmp/{scene_id}.json', sim_params=render_params)
        for run in _runs_for_family(family):
            orchestrator = NavOrchestrator(
                build_models_for_obs(obs),
                only_models=run['only_models'],
                only_techniques=run['only_techniques'],
            )
            result = orchestrator.navigate(obs)
            techniques = []
            for tr in result.per_technique:
                cov = np.asarray(tr.covariance_px2, dtype=float)[:2, :2]
                techniques.append(
                    {
                        'name': tr.technique_name,
                        'offset_v': float(tr.offset_px[0]),
                        'offset_u': float(tr.offset_px[1]),
                        'covariance_px2': [float(x) for x in cov.ravel()],
                        'confidence': float(tr.confidence),
                        'spurious': bool(tr.spurious),
                        'at_edge': bool(tr.at_edge),
                    }
                )
            row['runs'][run['key']] = techniques
    except Exception as exc:
        row['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        _restore(patched)
    return row


def _init_worker() -> None:
    """Silence the per-image and main loggers in a pool worker.

    Runs once per worker: at fork when the pool starts, and again after each
    ``maxtasksperchild`` respawn.  Native thread-pool pinning is handled at
    module import (the BLAS/OpenMP/NumExpr pools are capped to one thread per
    process, set before the first numpy import and inherited by every forked
    worker), not here: under the default fork start method a worker inherits
    the parent's already-initialized threading runtime, so setting the thread
    variables in this initializer would be too late to take effect.
    """
    import pdslogger

    from spindoctor.config import IMAGE_LOGGER, MAIN_LOGGER

    null_handler = pdslogger.NULL_HANDLER
    for logger in (IMAGE_LOGGER, MAIN_LOGGER):
        logger.remove_all_handlers()
        logger.add_handler(null_handler)


def main(argv: list[str] | None = None) -> int:
    """Generate scenes, navigate them in a worker pool, write JSONL rows."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--per-family', type=int, default=400)
    parser.add_argument('--families', default=','.join(FAMILIES))
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--injection', choices=INJECTION_KINDS, default='none')
    parser.add_argument(
        '--injection-sigma-px',
        type=float,
        default=0.7,
        help='per-axis sigma of the dt_shift bias draw',
    )
    parser.add_argument(
        '--gate-depression',
        type=float,
        default=0.15,
        help='amount every reliability threshold is raised in the reliability_gate channel',
    )
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    if not math.isfinite(args.gate_depression) or args.gate_depression < 0.0:
        parser.error('--gate-depression must be a non-negative finite number')

    families = [f.strip() for f in args.families.split(',') if f.strip()]
    tasks = []
    for family in families:
        for scene_id, sim_params, geometry in generate_scenes(
            family, args.per_family, campaign_seed=args.seed
        ):
            injection = _draw_injection(
                args.injection,
                scene_id,
                args.seed,
                args.injection_sigma_px,
                args.gate_depression,
            )
            tasks.append((scene_id, family, sim_params, geometry, injection))

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
                    'injection': args.injection,
                    'injection_sigma_px': args.injection_sigma_px,
                    'gate_depression': args.gate_depression,
                    'n_scenes': len(tasks),
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
