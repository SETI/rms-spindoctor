"""Render the sweep response curves as figures for the performance report.

Reads the per-sweep JSON written by :mod:`sim_sweep_runner` and writes PNG charts
under ``docs/simulator_report/_figures/``: the sub-pixel and wide-range offset
accuracy of every technique, and the camera-roll recovery.  Run as part of
``python -m tests.integration.sim_sweep_runner`` (or standalone) -- not part of
pytest.  Requires ``matplotlib``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

_RESULTS_ROOT = Path(__file__).parent / 'sim_sweeps' / 'results'
_FIGURES_ROOT = Path(__file__).parent.parent.parent / 'docs' / 'simulator_report' / '_figures'

# Technique -> (fine sweep, wide sweep, marker, label) for the offset figures.
_OFFSET_TECHNIQUES = [
    ('disc', 'disc_offset_fine', 'disc_offset_wide', 'o', 'BodyDiscCorrelateNav'),
    ('blob', 'blob_offset_fine', 'blob_offset_wide', 's', 'BodyBlobNav'),
    ('limb', 'limb_offset_fine', 'limb_offset_wide', '^', 'BodyLimbNav'),
    ('ring', 'ring_offset_fine', 'ring_offset_wide', 'D', 'RingEdgeNav'),
    ('star', 'star_offset_fine', 'star_offset_wide', 'v', 'StarFieldFromCatalogNav'),
    ('titan', 'titan_offset_fine', 'titan_offset_wide', 'P', 'TitanHazeNav'),
]


def _load(name: str) -> list[dict[str, Any]]:
    """Return the rows of one sweep result, or an empty list if absent."""
    path = _RESULTS_ROOT / f'{name}.json'
    if not path.is_file():
        return []
    return list(json.loads(path.read_text())['rows'])


def _xy(rows: list[dict[str, Any]], key: str) -> tuple[list[float], list[float]]:
    """Return (value, key) pairs for rows where ``key`` is populated."""
    xs = [r['value'] for r in rows if r.get(key) is not None]
    ys = [r[key] for r in rows if r.get(key) is not None]
    return xs, ys


def _offset_figure(which: str, title: str, out_name: str) -> None:
    """Plot offset error vs planted offset for every technique's ``which`` sweep."""
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for _key, fine, wide, marker, label in _OFFSET_TECHNIQUES:
        name = fine if which == 'fine' else wide
        xs, ys = _xy(_load(name), 'offset_error_px')
        if xs:
            ax.plot(xs, ys, marker=marker, ms=4, lw=1.2, label=label)
    ax.set_yscale('log')
    ax.set_xlabel('planted offset (px)')
    ax.set_ylabel('recovered-offset error (px, log scale)')
    ax.set_title(title)
    ax.grid(True, which='both', ls=':', alpha=0.5)
    ax.legend(fontsize=8, loc='best')
    fig.tight_layout()
    fig.savefig(_FIGURES_ROOT / out_name, dpi=110)
    plt.close(fig)


def _star_regime_figure() -> None:
    """Overlay the dim- and bright-star field sweeps to show the SNR crossover.

    The two sweeps share geometry and planted offset; only the stars' brightness
    differs.  The dim field rides the PSF-refined error floor; the bright field
    keeps the moment centroid (above the configured SNR ceiling) and reaches a
    far smaller error -- the visible payoff of the per-star moment/PSF choice.
    """
    dim = _load('star_offset_fine')
    bright = _load('star_offset_fine_bright')
    if not dim and not bright:
        return
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for rows, marker, label in [
        (dim, 'v', 'dim field (PSF-refined, vmag 3-4)'),
        (bright, 'o', 'bright field (moment, vmag 0-0.8)'),
    ]:
        xs, ys = _xy(rows, 'offset_error_px')
        if xs:
            ax.plot(xs, ys, marker=marker, ms=4, lw=1.2, label=label)
    ax.set_yscale('log')
    ax.set_xlabel('planted offset (px)')
    ax.set_ylabel('recovered-offset error (px, log scale)')
    ax.set_title('Star-field sub-pixel accuracy: dim vs bright (PSF-refine crossover)')
    ax.grid(True, which='both', ls=':', alpha=0.5)
    ax.legend(fontsize=8, loc='best')
    fig.tight_layout()
    fig.savefig(_FIGURES_ROOT / 'star_regime_accuracy.png', dpi=110)
    plt.close(fig)


def _rotation_figure() -> None:
    """Plot recovered-roll error vs planted roll for the star-field sweep."""
    rows = _load('star_rotation')
    xs, ys = _xy(rows, 'rotation_error_deg')
    if not xs:
        return
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(xs, ys, marker='o', ms=5, lw=1.2, color='tab:purple')
    ax.set_xlabel('planted camera roll (deg)')
    ax.set_ylabel('recovered-roll error (deg)')
    ax.set_title('Star-field camera-roll recovery')
    ax.grid(True, ls=':', alpha=0.5)
    fig.tight_layout()
    fig.savefig(_FIGURES_ROOT / 'rotation_accuracy.png', dpi=110)
    plt.close(fig)


def _mesh_irregularity_figure() -> None:
    """Plot the shape-mismatch centroid bias and confidence vs mesh relief.

    The navigator predicts the smooth (zero-relief) limit of the rendered mesh,
    so each step widens a pure shape mismatch.  The recovered-offset error -- the
    centroid bias the ellipsoidal model cannot remove -- grows with the rendered
    relief while the fused confidence falls, the regime the
    ``phase_irregularity_factor`` term is meant to capture.
    """
    rows = _load('irregularity_shape_mismatch')
    xs, errs = _xy(rows, 'offset_error_px')
    if not xs:
        return
    conf_x = [r['value'] for r in rows if r.get('confidence') is not None]
    conf_y = [r['confidence'] for r in rows if r.get('confidence') is not None]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(xs, errs, marker='o', ms=5, lw=1.3, color='tab:red', label='offset error')
    ax.set_xlabel('rendered mesh lumpiness (relief fraction)')
    ax.set_ylabel('recovered-offset error (px)', color='tab:red')
    ax.tick_params(axis='y', labelcolor='tab:red')
    ax.grid(True, ls=':', alpha=0.5)
    if conf_x:
        ax2 = ax.twinx()
        ax2.plot(conf_x, conf_y, marker='s', ms=5, lw=1.3, color='tab:blue', label='confidence')
        ax2.set_ylabel('fused confidence', color='tab:blue')
        ax2.tick_params(axis='y', labelcolor='tab:blue')
    ax.set_title('Shape mismatch: centroid bias and confidence vs mesh relief')
    fig.tight_layout()
    fig.savefig(_FIGURES_ROOT / 'mesh_irregularity.png', dpi=110)
    plt.close(fig)


def _mesh_pose_figure() -> None:
    """Plot mesh-limb degradation as the predicted pose leaves the true pose.

    The rendered mesh pose is fixed; the navigator's predicted pose is walked
    away from it.  The pinned limb's recovered-offset error grows with the
    disagreement and then the technique self-flags spurious (no point plotted),
    the navigator declining to trust a confidently-wrong limb.
    """
    rows = _load('pose_disagreement')
    if not rows:
        return
    true_pose = rows[0]['value']
    solved = [
        (r['value'] - true_pose, r['offset_error_px'])
        for r in rows
        if r.get('offset_error_px') is not None
    ]
    failed = [r['value'] - true_pose for r in rows if r.get('offset_error_px') is None]
    if not solved:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(
        *zip(*solved, strict=True), marker='^', ms=6, lw=1.3, color='tab:green', label='limb solved'
    )
    for x in failed:
        ax.axvline(x, color='tab:gray', ls='--', alpha=0.6)
    if failed:
        ax.axvline(failed[0], color='tab:gray', ls='--', alpha=0.6, label='limb spurious (no fix)')
    ax.set_xlabel('predicted-pose disagreement (deg from true pose)')
    ax.set_ylabel('limb recovered-offset error (px)')
    ax.set_title('Mesh-limb degradation under pose disagreement')
    ax.grid(True, ls=':', alpha=0.5)
    ax.legend(fontsize=8, loc='best')
    fig.tight_layout()
    fig.savefig(_FIGURES_ROOT / 'mesh_pose_disagreement.png', dpi=110)
    plt.close(fig)


# The Section 8 model-mismatch axes: (sweep name, x-axis label, panel title).
# Each walks one render-vs-navigate mismatch from a self-consistency floor (the
# first sweep value) into the mismatched regime; the recovery-error-vs-mismatch
# curve is the product.  The floor point is drawn distinctly and labeled.
_MISMATCH_AXES = [
    ('psf_limb_mismatch', 'rendered PSF sigma (px)', 'PSF core mismatch (limb)'),
    ('psf_limb_wings', 'rendered PSF wing energy fraction', 'PSF wing mismatch (limb)'),
    ('photometric_minnaert_mismatch', 'Minnaert k (1.0 = Lambert)', 'Photometric mismatch (disc)'),
    ('spk_parallax_error', 'planted parallax shift (px)', 'Spacecraft ephemeris'),
    ('differential_smear', 'star trail length (px)', 'Differential smear (stars)'),
    ('ring_orbit_error', 'planted ring radial error (px)', 'Ring ephemeris (edge)'),
    ('atmosphere_haze', 'haze reference optical depth', 'Atmosphere (limb)'),
]


def _mismatch_axes_figure() -> None:
    """Grid the Section 8 recovery-error-vs-mismatch curves, floor point labeled.

    One panel per mismatch axis: recovered-offset error versus the swept
    mismatch parameter.  The zero-mismatch point (equality with the navigator's
    configuration) is drawn as the self-consistency floor -- a bound on
    reproducibility, not accuracy -- and a failed step (the navigability cliff)
    is annotated on the axis rather than plotted.  The grid is sized to the
    axis count; any unused trailing panel is hidden.
    """
    n_cols = 3
    n_rows = -(-len(_MISMATCH_AXES) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(13.5, 4.0 * n_rows))
    flat_axes = list(axes.ravel())
    for ax in flat_axes[len(_MISMATCH_AXES) :]:
        ax.set_axis_off()
    for ax, (name, xlabel, title) in zip(flat_axes, _MISMATCH_AXES, strict=False):
        rows = _load(name)
        solved = [
            (r['value'], r['offset_error_px']) for r in rows if r.get('offset_error_px') is not None
        ]
        failed = [r['value'] for r in rows if r.get('offset_error_px') is None]
        if solved:
            xs, ys = zip(*solved, strict=True)
            ax.plot(xs, ys, marker='o', ms=4, lw=1.2, color='tab:blue', label='recovery error')
            ax.plot(
                [xs[0]],
                [ys[0]],
                marker='*',
                ms=13,
                color='tab:green',
                lw=0,
                label='self-consistency floor (not accuracy)',
            )
        for x in failed:
            ax.axvline(x, color='tab:gray', ls='--', alpha=0.6)
        if failed:
            ax.axvline(failed[0], color='tab:gray', ls='--', alpha=0.6, label='fit failed (cliff)')
        ax.set_yscale('log')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('recovered-offset error (px)')
        ax.set_title(title, fontsize=10)
        ax.grid(True, which='both', ls=':', alpha=0.5)
        ax.legend(fontsize=7, loc='best')
    fig.suptitle(
        'Recovery error vs model mismatch (Section 8 axes); zero-mismatch = self-consistency floor',
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(_FIGURES_ROOT / 'mismatch_axes.png', dpi=110)
    plt.close(fig)


def generate_plots() -> list[Path]:
    """Render all report figures from the sweep results; return their paths."""
    _FIGURES_ROOT.mkdir(parents=True, exist_ok=True)
    _offset_figure(
        'fine',
        'Sub-pixel offset accuracy by technique (dense fractional sweep)',
        'offset_accuracy_fine.png',
    )
    _offset_figure(
        'wide',
        'Wide-range offset accuracy by technique (to the navigable ceiling)',
        'offset_accuracy_wide.png',
    )
    _star_regime_figure()
    _rotation_figure()
    _mesh_irregularity_figure()
    _mesh_pose_figure()
    _mismatch_axes_figure()
    return sorted(_FIGURES_ROOT.glob('*.png'))


if __name__ == '__main__':
    for path in generate_plots():
        print(f'wrote {path}')
