"""Fit and verify the haze technique's confidence anchors on planted truth.

The generic calibration fitter (``util/calibration/fit.py``) refits every
technique's alphas against a "recovered within ``--err-ok-px``" label.  It
cannot set this technique's anchors on its own, for two reasons this script
exists to handle:

- **The label population is the wrong one.**  What the anchors have to do is
  separate the rows that are wrong by more than twice a stated per-axis bound
  while still calling themselves confident.  Those rows are a handful out of
  several hundred, so an unweighted fit optimizes the bulk and leaves them
  exactly where they were.  This script up-weights them.
- **One coefficient does not transfer.**  Left unbounded, the fit drives the
  arc-residual alpha to -15.12, which is a near-hard gate at half a pixel
  of residual.  That is correct in a simulator, whose rendered haze limb IS
  the circle being fitted, and ruinous on real frames, whose median residual
  is about 1.1 px.  The bound in :data:`TERMS` is what keeps the fitted
  spec usable outside the simulator, and ``--unbounded`` prints the
  unconstrained solve so the size of that effect stays on the record.

Run::

    source /seti/newnav/setup.sh
    python util/titan_truth/fit_confidence.py _work/titan_truth/rows_final_config.jsonl

With no ``--propose`` the script VERIFIES the shipped anchors read from
``config_510_techniques.yaml``; with it, the script fits and prints a
proposal in the same form.  ``--real-rows`` additionally scores a real-frame
cohort run under the same anchors, which is the transfer check.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / 'src'))

TECHNIQUE = 'TitanHazeNav'

# The spec's terms, their normalization, and the sign / magnitude bounds the
# fit runs under.  Order matches config_510_techniques.yaml.
TERMS: tuple[tuple[str, float, float, float, tuple[float, float]], ...] = (
    ('symmetry_peak_score', 0.0, 1.0, 1.0, (0.0, 6.0)),
    ('symmetry_valid_fraction', 0.0, 1.0, 1.0, (0.0, 6.0)),
    ('arc_inlier_fraction', 0.0, 1.0, 1.0, (0.0, 6.0)),
    ('envelope_diameter_px', 0.0, 160.0, 1.0, (0.0, 8.0)),
    # Bounded; see the module docstring.
    ('arc_residual_rms_px', 0.0, 3.0, 1.0, (-2.5, 0.0)),
)

# Twice the per-axis bounds the accuracy claim is stated at (1 px cross,
# 3 px along): a committed row outside either of these is what the
# no-confident-wrong criterion is about.
CROSS_LIMIT_PX = 2.0
ALONG_LIMIT_PX = 6.0

# Fit hyper-parameters.  The weight is what makes the fit answer the
# separation question rather than the bulk-accuracy one; the L2 is set as
# loosely as the separation allows, so the coefficients stay identifiable.
L2 = 0.0002
WEIGHT_ON_WRONG_ROWS = 80.0
ERR_OK_PX = 1.0


def _features(diagnostics: dict[str, Any]) -> list[float]:
    """Normalize one row's diagnostics exactly as the runtime spec does."""
    out = []
    for name, offset, divisor, cap, _ in TERMS:
        raw = diagnostics[name]
        if isinstance(raw, dict):  # a tagged non-finite value
            raw = float('nan')
        out.append(min(max((float(raw) - offset) / divisor, 0.0), cap))
    return out


def _load_sim(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load committed planted-truth rows as (features, cross, along, total, ids)."""
    rows = [json.loads(line) for line in path.open()][1:]
    committed = [r for r in rows if r.get('outcome') == 'committed']
    features = np.array([_features(r['diagnostics']) for r in committed])
    cross = np.array([r['error_px']['cross'] for r in committed])
    along = np.array([r['error_px']['along'] for r in committed])
    total = np.array([r['error_px']['total'] for r in committed])
    return features, cross, along, total, [r['scene_id'] for r in committed]


def _shipped_anchors() -> np.ndarray:
    """The alphas currently in ``config_510_techniques.yaml``, ordered by TERMS."""
    from spindoctor.config import DEFAULT_CONFIG
    from spindoctor.nav_technique.confidence_config import load_confidence_spec

    spec = load_confidence_spec(DEFAULT_CONFIG.category('techniques'), TECHNIQUE)
    by_feature = {term.feature: term.alpha for term in spec.terms}
    return np.array([spec.alpha0] + [by_feature[name] for name, *_ in TERMS])


def fit_anchors(
    features: np.ndarray, wrong: np.ndarray, ok: np.ndarray, *, bounded: bool
) -> np.ndarray:
    """Weighted L2-regularized logistic fit; returns ``[alpha0, alpha...]``.

    Parameters:
        features: One normalized row per committed scene.
        wrong: Boolean mask of rows outside twice a stated per-axis bound.
        ok: Float label, 1.0 when the row recovered within ``ERR_OK_PX``.
        bounded: Apply the :data:`TERMS` coefficient bounds.  False reproduces
            the unconstrained solve the docstring warns about.

    Returns:
        The fitted coefficient vector.
    """
    n, k = features.shape
    design = np.hstack([np.ones((n, 1)), features])
    weights = np.where(wrong, WEIGHT_ON_WRONG_ROWS, 1.0)
    weights = weights / weights.mean()
    signed = 2.0 * ok - 1.0

    def loss(beta: np.ndarray) -> float:
        margin = signed * (design @ beta)
        return float((weights * np.logaddexp(0.0, -margin)).mean() + L2 * float(beta @ beta))

    bounds: list[tuple[float | None, float | None]] = [(None, None)]
    for _, _, _, _, (low, high) in TERMS:
        bounds.append((low, high) if bounded else (None, None))
    result = minimize(
        loss, np.zeros(k + 1), method='L-BFGS-B', bounds=bounds, options={'maxiter': 4000}
    )
    return np.asarray(result.x)


def verify(
    beta: np.ndarray,
    features: np.ndarray,
    cross: np.ndarray,
    along: np.ndarray,
    total: np.ndarray,
    ids: list[str],
) -> dict[str, Any]:
    """Score one anchor set against the acceptance checks it has to meet."""
    confidence = 1.0 / (1.0 + np.exp(-(beta[0] + features @ beta[1:])))
    wrong = (np.abs(cross) > CROSS_LIMIT_PX) | (np.abs(along) > ALONG_LIMIT_PX)
    confident = confidence >= 0.5
    order = np.argsort(confidence)
    quintiles = [float(np.abs(total[part]).mean()) for part in np.array_split(order, 5)]
    return {
        'n_committed': len(confidence),
        'n_wrong': int(wrong.sum()),
        'confident_wrong': int((confident & wrong).sum()),
        'wrong_row_confidence': {
            ids[i]: round(float(confidence[i]), 3) for i in np.flatnonzero(wrong)
        },
        'n_confident': int(confident.sum()),
        'cross_p99_confident': round(float(np.percentile(np.abs(cross[confident]), 99)), 3),
        'along_p99_confident': round(float(np.percentile(np.abs(along[confident]), 99)), 3),
        'along_max_confident': round(float(np.abs(along[confident]).max()), 3),
        'mean_abs_error_by_confidence_quintile': [round(v, 3) for v in quintiles],
        'good_row_confidence_p5_p50': [
            round(float(np.percentile(confidence[~wrong], 5)), 3),
            round(float(np.percentile(confidence[~wrong], 50)), 3),
        ],
    }


def _score_real(beta: np.ndarray, path: Path) -> dict[str, Any]:
    """Score a real-frame cohort run's committed rows under one anchor set."""
    rows = []
    for line in path.open():
        record = json.loads(line)
        entry = (record.get('techniques') or {}).get(TECHNIQUE)
        if entry and not entry['spurious']:
            rows.append(_features(entry['diagnostics']))
    if not rows:
        return {}
    values = np.array(rows)
    confidence = 1.0 / (1.0 + np.exp(-(beta[0] + values @ beta[1:])))
    return {
        'n': len(confidence),
        'min': round(float(confidence.min()), 3),
        'p50': round(float(np.median(confidence)), 3),
        'max': round(float(confidence.max()), 3),
        'below_0.5': int((confidence < 0.5).sum()),
    }


def _format(beta: np.ndarray) -> str:
    """Render a coefficient vector in the config's own order."""
    parts = [f'alpha0 {beta[0]:+.4f}']
    parts += [f'{name} {alpha:+.4f}' for (name, *_), alpha in zip(TERMS, beta[1:], strict=True)]
    return '  '.join(parts)


def main(argv: list[str] | None = None) -> int:
    """Verify the shipped anchors, or fit and print a proposal."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('rows', type=Path)
    parser.add_argument('--propose', action='store_true', help='fit rather than verify')
    parser.add_argument('--unbounded', action='store_true', help='also print the unbounded solve')
    parser.add_argument('--real-rows', type=Path, default=None)
    args = parser.parse_args(argv)

    features, cross, along, total, ids = _load_sim(args.rows)
    wrong = (np.abs(cross) > CROSS_LIMIT_PX) | (np.abs(along) > ALONG_LIMIT_PX)
    ok = (total <= ERR_OK_PX).astype(float)
    print(f'{len(ids)} committed rows; {int(wrong.sum())} outside twice a stated axis bound')

    if args.unbounded:
        print(f'unbounded solve: {_format(fit_anchors(features, wrong, ok, bounded=False))}')

    beta = fit_anchors(features, wrong, ok, bounded=True) if args.propose else _shipped_anchors()
    print(('proposal: ' if args.propose else 'shipped:  ') + _format(beta))
    print(json.dumps(verify(beta, features, cross, along, total, ids), indent=2))
    if args.real_rows:
        print('real-frame transfer: ' + json.dumps(_score_real(beta, args.real_rows)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
