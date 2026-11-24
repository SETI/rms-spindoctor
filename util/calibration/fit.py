"""Fit the per-technique confidence calibration from sim planted truth.

Consumes the JSONL rows written by ``collect.py`` and, for every technique
with enough usable rows, refits the sigmoid-of-linear-combination alpha
coefficients (``config_510_techniques.yaml``) so the reported confidence
tracks the empirical probability that the technique's own recovered offset
lies within ``--err-ok-px`` (default 1.0 px) of the planted truth.

Methodology (sim-anchored regime):

- Rows where a hard gate fired (``spurious`` / ``at_edge``) are excluded:
  the gates force confidence to zero regardless of the alphas, so they are
  not part of the sigmoid fit.
- Each YAML term's ``offset`` / ``divisor`` / ``cap_at`` normalization is
  kept exactly as configured; only ``alpha0`` and the per-term ``alpha``
  are refit.  The fit is therefore a drop-in coefficient update
  (Platt scaling).
- L2-regularized logistic regression (lambda ``--l2``), so a technique
  whose usable rows are single-class converges to a bounded, honest
  plateau instead of a runaway intercept.
- Post-sigmoid structural caps (BodyBlobNav's 0.4 ensemble cap, the
  star-unique per-mode caps) are design decisions about cross-technique
  trust, not calibration outputs; they are left in place and reported.

Outputs a machine-readable proposal (``--out-json``) and a human report
(``--out-report``) with reliability tables, class balance, AUC/Brier
before vs after, and error percentiles by confidence decile.

Run:

    venv/bin/python util/calibration/fit.py _work/calibration/rows_v1.jsonl \
        --out-json _work/calibration/fit_v1.json \
        --out-report _work/calibration/fit_v1.md
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / 'src'))

from spindoctor.config import DEFAULT_CONFIG  # noqa: E402
from spindoctor.nav_technique.confidence import ConfidenceSpec  # noqa: E402
from spindoctor.nav_technique.nav_technique import NavTechnique  # noqa: E402

TECHNIQUES = (
    'BodyDiscCorrelateNav',
    'BodyBlobNav',
    'BodyLimbNav',
    'BodyTerminatorNav',
    'RingEdgeNav',
    'RingAnnulusNav',
    'StarFieldFromCatalogNav',
    'StarUniqueMatchNav',
    'StarRefineNav',
    'TitanHazeNav',
)

# Diagnostics attributes not yet wired as YAML terms but recorded per row.
# Adding a term is a pure-YAML change (the runtime resolves any diagnostics
# attribute by name), so the fit considers these alongside the configured
# terms; a candidate only survives into the proposal if its fitted weight
# is materially nonzero.  Normalizations chosen to put the typical healthy
# range near [0, 1], same convention as the configured terms.
CANDIDATE_TERMS: dict[str, list[dict[str, Any]]] = {}

# Normalization overrides for configured terms whose current offset /
# divisor / cap_at pins the normalized value at its cap (or floor) for
# nearly every usable row (see the per-term health table in the report).
# A pinned term is a constant: it carries no information and its weight is
# arbitrary.  Overrides are keyed by (technique, feature) and merged over
# the YAML values before fitting; they are calibration outputs exactly
# like the alphas (the YAML comment on each term already reserves the
# transform for the calibration sweep to tune).  Values chosen so the
# campaign's raw p5-p95 span maps onto roughly [0, 1] (see the raw
# percentiles in the term-health table of the v2 report).
TRANSFORM_OVERRIDES: dict[tuple[str, str], dict[str, float | None]] = {
    # raw p5/p50/p95 = 6.3 / 9.6 / 19.9; the old /6 cap-1 pinned every row.
    ('BodyDiscCorrelateNav', 'ncc_peak'): {'offset': 6.0, 'divisor': 14.0, 'cap_at': 1.0},
    # raw 17.7 / 55.7 / 648.8 (heavy tail); old /4 cap-1 pinned every row.
    ('BodyBlobNav', 'body_snr_inside_predicted_bbox'): {'divisor': 600.0, 'cap_at': 1.0},
    # raw 22 / 50 / 139; old (x-8)/8 cap-1 pinned every row.
    ('BodyBlobNav', 'body_extent_px'): {'offset': 8.0, 'divisor': 130.0, 'cap_at': 1.0},
    # raw 151 / 280 / 440; old /100 cap-1 pinned every row.
    ('BodyLimbNav', 'visible_arc_px'): {'divisor': 440.0, 'cap_at': 1.0},
    # raw 440 / 761 / 1552; old /200 cap-1 pinned every row.
    ('RingEdgeNav', 'total_edge_length_px'): {'divisor': 1500.0, 'cap_at': 1.0},
    # raw 10.8 / 33.2 / 52.0; old /6 cap-1 pinned every row.
    ('RingAnnulusNav', 'ncc_peak'): {'offset': 6.0, 'divisor': 45.0, 'cap_at': 1.0},
    # raw 15 / 85 / 252 on the 20260718 campaign (the flux-normalized
    # star deposit brightened the healthy range); the /100 transform
    # pinned 44% of rows at the cap.
    ('StarUniqueMatchNav', 'predicted_snr'): {'divisor': 250.0, 'cap_at': 1.0},
    # raw 0.002 / 0.047 / 0.367 on the physical-body campaign; the design
    # /0.15 saturated 29% of rows.
    ('BodyBlobNav', 'max_phase_irregularity_factor'): {'divisor': 0.35, 'cap_at': 1.0},
}

# Sign constraints for the fitted alphas, by feature name.  Error-like
# diagnostics must not earn positive weight (a confounder in the sim
# cohort must not turn "higher residual" into "more confidence"); quality
# diagnostics must not earn negative weight.  Features not listed are
# unconstrained -- notably BodyBlobNav's body_extent_px, where the
# sim-anchored data reverses the design prior (absolute centroid error
# grows with apparent size in the regime the reliability gate admits).
SIGN_BY_FEATURE: dict[str, str] = {
    'ncc_peak': '+',
    'peak_to_runner_up_ratio': '+',
    'consistency_ratio': '-',
    'body_snr_inside_predicted_bbox': '+',
    'visible_limb_arc_fraction': '+',
    'visible_terminator_arc_fraction': '+',
    'visible_arc_px': '+',
    'dt_fit_rms_px': '-',
    'per_edge_dt_rms_mean': '-',
    'total_edge_length_px': '+',
    'annulus_count': '+',
    'blob_count': '+',
    'body_count': '+',
    'max_phase_irregularity_factor': '-',
    'n_inliers': '+',
    'median_residual_px': '-',
    'n_detected_sources': '+',
    'n_catalog_predicted': '+',
    'predicted_snr': '+',
    'brightness_margin_mag': '+',
    'residual_px': '-',
    'n_stars_used': '+',
    'median_pos_err_px': '-',
    'residual_scatter_px': '-',
    'mean_phase_angle_factor': '+',
    'mean_albedo_penalty': '-',
    'symmetry_peak_score': '+',
    'symmetry_valid_fraction': '+',
    'arc_inlier_fraction': '+',
    'envelope_diameter_px': '+',
    'arc_residual_rms_px': '-',
}

# Alphas to hold at a fixed (design-prior) value when the campaign left the
# feature CONSTANT across every usable row -- the fit cannot identify the
# coefficient, but zeroing it would discard a structurally sound prior the
# real data may exercise (multi-body scenes, irregular-shape configs).  The
# fit drops the constant column from the design and afterwards corrects
# alpha0 by -alpha_frozen * constant_value so the net logit on the
# campaign's rows is unchanged.
FROZEN_ALPHAS: dict[tuple[str, str], float] = {
    # Single-body campaign scenes; keep the design's multi-body reward.
    # (The irregularity factor and the limb arc fraction, frozen in
    # earlier campaigns, now vary -- config_220 residuals are populated
    # and the sim limb reports its honest clipped/occluded fraction --
    # so both are ordinary fitted terms.)
    ('BodyDiscCorrelateNav', 'body_count'): 0.4,
    ('BodyBlobNav', 'blob_count'): 0.4,
    # The detector always fills its max_sources=30 budget (noise peaks
    # included), so the count carries no information as wired.
    ('StarFieldFromCatalogNav', 'n_detected_sources'): 0.0,
}


def _decode(value: Any) -> float:
    """Decode a diagnostics scalar from the JSONL encoding."""
    if isinstance(value, dict) and '__nonfinite__' in value:
        return float(value['__nonfinite__'])
    return float(value)


def _term_list(spec: ConfidenceSpec, technique: str) -> list[dict[str, Any]]:
    """Configured YAML terms plus any not-yet-wired candidate terms."""
    terms = [
        {
            'feature': term.feature,
            'offset': term.offset,
            'divisor': term.divisor,
            'cap_at': term.cap_at,
            'added': False,
            **TRANSFORM_OVERRIDES.get((technique, term.feature), {}),
        }
        for term in spec.terms
    ]
    configured = {t['feature'] for t in terms}
    for candidate in CANDIDATE_TERMS.get(technique, []):
        if candidate['feature'] in configured:
            continue
        terms.append({**candidate, 'added': True})
    return terms


def _normalized_features(terms: list[dict[str, Any]], diagnostics: dict[str, Any]) -> list[float]:
    """Apply each term's offset/divisor/cap_at exactly as the runtime does."""
    feats = []
    for term in terms:
        raw = _decode(diagnostics[term['feature']])
        scaled = (raw - term['offset']) / term['divisor']
        if term['cap_at'] is not None:
            scaled = min(max(scaled, 0.0), term['cap_at'])
        feats.append(scaled)
    return feats


def _fit_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    l2: float,
    signs: list[str | None] | None = None,
) -> np.ndarray:
    """L2-regularized logistic fit; returns [alpha0, alpha_1..alpha_k].

    ``signs`` optionally constrains each coefficient: ``'+'`` -> alpha >= 0,
    ``'-'`` -> alpha <= 0, ``None`` -> unconstrained.  The intercept is
    always unconstrained.
    """
    n, k = features.shape
    design = np.hstack([np.ones((n, 1)), features])

    def loss_and_grad(beta: np.ndarray) -> tuple[float, np.ndarray]:
        z = design @ beta
        # log(1 + exp(-y*z)) with y in {-1, +1}, numerically stable
        y = 2.0 * labels - 1.0
        margin = y * z
        ll = np.logaddexp(0.0, -margin).sum()
        prob = 1.0 / (1.0 + np.exp(-z))
        grad = design.T @ (prob - labels) / n + 2.0 * l2 * beta
        return float(ll / n + l2 * float(beta @ beta)), grad

    bounds: list[tuple[float | None, float | None]] = [(None, None)]
    for sign in signs or [None] * k:
        if sign == '+':
            bounds.append((0.0, None))
        elif sign == '-':
            bounds.append((None, 0.0))
        else:
            bounds.append((None, None))
    beta0 = np.zeros(k + 1)
    result = minimize(
        loss_and_grad,
        beta0,
        method='L-BFGS-B',
        jac=True,
        bounds=bounds,
        options={'maxiter': 500},
    )
    return np.asarray(result.x)


def _predict(beta: np.ndarray, features: np.ndarray) -> np.ndarray:
    z = beta[0] + features @ beta[1:]
    return 1.0 / (1.0 + np.exp(-z))


def _auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    """Rank AUC with tie-averaged ranks; None when only one class is present."""
    from scipy.stats import rankdata

    pos = scores[labels == 1.0]
    neg = scores[labels == 0.0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    ranks = rankdata(np.concatenate([neg, pos]))
    rank_pos = float(ranks[len(neg) :].sum())
    auc = (rank_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))
    return round(float(auc), 4)


def _reliability_table(
    confidence: np.ndarray, labels: np.ndarray, errors: np.ndarray, *, n_bins: int = 10
) -> list[dict[str, float | int | None]]:
    """Binned reported-confidence vs empirical success + error percentiles."""
    table = []
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for lo, hi in itertools.pairwise(edges):
        mask = (confidence >= lo) & (confidence < hi if hi < 1.0 else confidence <= hi)
        n = int(mask.sum())
        entry: dict[str, float | int | None] = {
            'bin_lo': round(float(lo), 2),
            'bin_hi': round(float(hi), 2),
            'n': n,
        }
        if n:
            entry['mean_confidence'] = round(float(confidence[mask].mean()), 3)
            entry['empirical_success'] = round(float(labels[mask].mean()), 3)
            entry['err_p50_px'] = round(float(np.percentile(errors[mask], 50)), 3)
            entry['err_p90_px'] = round(float(np.percentile(errors[mask], 90)), 3)
        else:
            entry['mean_confidence'] = None
            entry['empirical_success'] = None
            entry['err_p50_px'] = None
            entry['err_p90_px'] = None
        table.append(entry)
    return table


def _packaged_spec(name: str) -> ConfidenceSpec:
    """The confidence formula the packaged configuration gives the technique ``name``."""
    DEFAULT_CONFIG.read_config()
    technique = next(cls for cls in NavTechnique._registry if cls.name == name)
    if technique.confidence_spec is None:
        raise ValueError(f'{name} has no confidence formula')
    return technique.confidence_spec


def fit_technique(
    name: str,
    rows: list[dict[str, Any]],
    *,
    err_ok_px: float,
    l2: float,
) -> dict[str, Any] | None:
    """Fit one technique's alphas; returns the proposal dict or None."""
    spec = _packaged_spec(name)
    usable = [
        t
        for row in rows
        for t in row.get('techniques', [])
        if t['name'] == name
        and not t['spurious']
        and not t['at_edge']
        and t['offset_error_px'] is not None
    ]
    if len(usable) < 50:
        return None
    terms = _term_list(spec, name)
    features = np.array([_normalized_features(terms, t['diagnostics']) for t in usable])
    errors = np.array([t['offset_error_px'] for t in usable])
    labels = (errors <= err_ok_px).astype(float)
    current_conf = np.array([t['confidence'] for t in usable])

    # Constant columns cannot be fit (collinear with the intercept): drop
    # them from the design and afterwards report the frozen design-prior
    # alpha (FROZEN_ALPHAS; default 0.0), correcting alpha0 by
    # -alpha_frozen * constant_value so the net logit on the campaign's
    # rows is unchanged.
    constant = features.std(axis=0) < 1e-9
    fit_columns = [i for i in range(len(terms)) if not constant[i]]
    signs = [SIGN_BY_FEATURE.get(terms[i]['feature']) for i in fit_columns]
    beta_fit = _fit_logistic(features[:, fit_columns], labels, l2=l2, signs=signs)
    beta = np.zeros(len(terms) + 1)
    beta[0] = beta_fit[0]
    for out_index, col in enumerate(fit_columns):
        beta[col + 1] = beta_fit[out_index + 1]
    frozen_notes = []
    for i in range(len(terms)):
        if not constant[i]:
            continue
        frozen = FROZEN_ALPHAS.get((name, terms[i]['feature']), 0.0)
        beta[i + 1] = frozen
        constant_value = float(features[0, i])
        beta[0] -= frozen * constant_value
        frozen_notes.append(
            {
                'feature': terms[i]['feature'],
                'alpha_frozen': frozen,
                'constant_value': round(constant_value, 4),
            }
        )
    fitted_conf = _predict(beta, features)

    # Per-term health: a term pinned at its cap (or floor) for nearly every
    # usable row is a constant -- it carries no information at the current
    # normalization and its weight is arbitrary (collinear with alpha0).
    term_health = []
    raw_matrix = np.array(
        [[_decode(t['diagnostics'][term['feature']]) for term in terms] for t in usable]
    )
    for index, term in enumerate(terms):
        column = features[:, index]
        raw_column = raw_matrix[:, index]
        raw_finite = raw_column[np.isfinite(raw_column)]
        at_cap = (
            float((column >= term['cap_at'] - 1e-12).mean()) if term['cap_at'] is not None else 0.0
        )
        term_health.append(
            {
                'feature': term['feature'],
                'frac_at_cap': round(at_cap, 3),
                'frac_at_floor': round(float((column <= 1e-12).mean()), 3),
                'std': round(float(column.std()), 4),
                'raw_p5': round(float(np.percentile(raw_finite, 5)), 3)
                if len(raw_finite)
                else None,
                'raw_p50': round(float(np.percentile(raw_finite, 50)), 3)
                if len(raw_finite)
                else None,
                'raw_p95': round(float(np.percentile(raw_finite, 95)), 3)
                if len(raw_finite)
                else None,
            }
        )

    def brier(conf: np.ndarray) -> float:
        return float(np.mean((conf - labels) ** 2))

    proposal = {
        'technique': name,
        'n_usable': len(usable),
        'n_success': int(labels.sum()),
        'base_rate': round(float(labels.mean()), 3),
        'err_ok_px': err_ok_px,
        'alpha0': round(float(beta[0]), 3),
        'terms': [
            {
                'feature': term['feature'],
                'alpha': round(float(a), 3),
                'offset': term['offset'],
                'divisor': term['divisor'],
                'cap_at': term['cap_at'],
                'added': term['added'],
            }
            for term, a in zip(terms, beta[1:], strict=True)
        ],
        'term_health': term_health,
        'frozen_terms': frozen_notes,
        'metrics': {
            'auc_current': _auc(current_conf, labels),
            'auc_fitted': _auc(fitted_conf, labels),
            'brier_current': round(brier(current_conf), 4),
            'brier_fitted': round(brier(fitted_conf), 4),
        },
        'reliability_fitted': _reliability_table(fitted_conf, labels, errors),
        'reliability_current': _reliability_table(current_conf, labels, errors),
        'hard_cap': spec.hard_cap,
    }
    return proposal


def _format_report(proposals: list[dict[str, Any]], *, err_ok_px: float) -> str:
    """Render the human-readable calibration report."""
    lines = [
        '# Sim-anchored confidence calibration fit',
        '',
        f'Label: technique offset error <= {err_ok_px} px vs planted truth.',
        'Rows where a hard gate fired (spurious / at_edge) are excluded; the',
        'gates zero confidence regardless of alphas.',
        '',
    ]
    for p in proposals:
        m = p['metrics']
        lines += [
            f'## {p["technique"]}',
            '',
            f'- usable rows: {p["n_usable"]} (success {p["n_success"]}, '
            f'base rate {p["base_rate"]})',
            f'- AUC current -> fitted: {m["auc_current"]} -> {m["auc_fitted"]}',
            f'- Brier current -> fitted: {m["brier_current"]} -> {m["brier_fitted"]}',
            f'- alpha0: {p["alpha0"]}'
            + (f' (hard_cap {p["hard_cap"]} retained)' if p['hard_cap'] else ''),
            '',
            '| feature | alpha | offset | divisor | cap_at |',
            '|---|---|---|---|---|',
        ]
        for t in p['terms']:
            lines.append(
                f'| {t["feature"]} | {t["alpha"]} | {t["offset"]} | '
                f'{t["divisor"]} | {t["cap_at"]} |'
            )
        lines += [
            '',
            '| fitted conf bin | n | empirical success | err p50 | err p90 |',
            '|---|---|---|---|---|',
        ]
        for b in p['reliability_fitted']:
            if b['n'] == 0:
                continue
            lines.append(
                f'| {b["bin_lo"]}-{b["bin_hi"]} | {b["n"]} | '
                f'{b["empirical_success"]} | {b["err_p50_px"]} | {b["err_p90_px"]} |'
            )
        lines.append('')
    return '\n'.join(lines) + '\n'


def main(argv: list[str] | None = None) -> int:
    """Load rows, fit every technique, write the proposal + report."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('rows', type=Path)
    parser.add_argument('--err-ok-px', type=float, default=1.0)
    parser.add_argument('--l2', type=float, default=0.01)
    parser.add_argument('--out-json', type=Path, required=True)
    parser.add_argument('--out-report', type=Path, required=True)
    args = parser.parse_args(argv)

    rows = []
    with args.rows.open() as fh:
        for line in fh:
            record = json.loads(line)
            if not record.get('manifest'):
                rows.append(record)
    print(f'{len(rows)} scene rows loaded')

    proposals = []
    for name in TECHNIQUES:
        proposal = fit_technique(name, rows, err_ok_px=args.err_ok_px, l2=args.l2)
        if proposal is None:
            print(f'{name}: insufficient usable rows; skipped')
            continue
        m = proposal['metrics']
        print(
            f'{name}: n={proposal["n_usable"]} base={proposal["base_rate"]} '
            f'AUC {m["auc_current"]} -> {m["auc_fitted"]} '
            f'Brier {m["brier_current"]} -> {m["brier_fitted"]}'
        )
        proposals.append(proposal)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(proposals, indent=2) + '\n')
    args.out_report.write_text(_format_report(proposals, err_ok_px=args.err_ok_px))
    print(f'Wrote {args.out_json} and {args.out_report}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
