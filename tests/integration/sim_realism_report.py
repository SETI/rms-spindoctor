"""Figures and JSON summary for the realism-match runner.

Writes one figure per figure of merit per instrument (real/sim histogram
overlays plus the curve overlays) into ``docs/simulator_report/_figures/``
as ``realism_<instrument>_<fom>.png``, and a deterministic JSON summary
into ``tests/integration/realism_results/realism_summary.json`` that backs
the report's tables (per-kind W1 divergences, per-curve density-W1
divergences, support labels, the FOM 3 one-sided-stratum disclosure, the
FOM 7 diagnostic rows, and the artifact-incidence comparison against the
catalog defaults).

Two fixed series colors carry sides everywhere: real cohort in blue,
simulated frames in orange -- a colorblind-safe pair, never cycled.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from spindoctor.sim.forward.artifacts_catalog import resolve_detector_defaults
from spindoctor.support.types import NDArrayFloatType
from tests.integration.sim_realism import InstrumentComparison, RealismResults

__all__ = ['write_figures', 'write_summary']

_FIGURES_ROOT = Path(__file__).parent.parent.parent / 'docs' / 'simulator_report' / '_figures'
_RESULTS_ROOT = Path(__file__).parent / 'realism_results'

# Fixed series colors (real vs sim), used in every figure.
_REAL_COLOR = '#31688e'
_SIM_COLOR = '#d9752e'


def _hist_overlay(
    ax: Axes,
    real: list[float],
    sim: list[float],
    *,
    title: str,
    xlabel: str,
    log_x: bool = False,
) -> None:
    """One histogram overlay panel; robust shared binning from both samples."""
    real_arr = np.asarray(real, dtype=np.float64)
    sim_arr = np.asarray(sim, dtype=np.float64)
    real_arr = real_arr[np.isfinite(real_arr)]
    sim_arr = sim_arr[np.isfinite(sim_arr)]
    pooled = np.concatenate([real_arr, sim_arr]) if sim_arr.size else real_arr
    if pooled.size == 0:
        ax.set_title(f'{title} (no data)', fontsize=9)
        ax.set_axis_off()
        return
    lo, hi = (float(q) for q in np.quantile(pooled, [0.005, 0.995]))
    positive = pooled[pooled > 0.0]
    if log_x and positive.size:
        log_lo = max(float(positive.min()), 1e-12)
        log_hi = max(hi, log_lo * 10.0)
        bins = [float(b) for b in np.geomspace(log_lo, log_hi, 25)]
        ax.set_xscale('log')
    else:
        if hi <= lo:
            lo, hi = lo - 0.5, hi + 0.5
        bins = [float(b) for b in np.linspace(lo, hi, 25)]
    # Winsorize into the bin range for display (mirrors the W1 clip): a
    # sample falling entirely outside the bins would otherwise produce an
    # empty density histogram (0/0).
    real_disp = np.clip(real_arr, bins[0], bins[-1])
    sim_disp = np.clip(sim_arr, bins[0], bins[-1])
    if real_disp.size:
        ax.hist(
            real_disp,
            bins=bins,
            density=True,
            histtype='stepfilled',
            alpha=0.35,
            color=_REAL_COLOR,
            label=f'real (n={real_disp.size})',
        )
        ax.hist(real_disp, bins=bins, density=True, histtype='step', color=_REAL_COLOR)
    if sim_disp.size:
        ax.hist(
            sim_disp,
            bins=bins,
            density=True,
            histtype='stepfilled',
            alpha=0.35,
            color=_SIM_COLOR,
            label=f'sim (n={sim_disp.size})',
        )
        ax.hist(sim_disp, bins=bins, density=True, histtype='step', color=_SIM_COLOR)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel('density', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, loc='best')
    ax.grid(alpha=0.2)


def _curve_overlay(
    ax: Axes,
    real_curves: list[tuple[NDArrayFloatType, NDArrayFloatType]],
    sim_curves: list[tuple[NDArrayFloatType, NDArrayFloatType]],
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    log_y: bool = False,
) -> None:
    """Mean-curve overlay panel with faint per-frame traces."""
    if not real_curves and not sim_curves:
        ax.set_title(f'{title} (no data)', fontsize=9)
        ax.set_axis_off()
        return
    for curves, color, label in (
        (real_curves, _REAL_COLOR, 'real'),
        (sim_curves, _SIM_COLOR, 'sim'),
    ):
        if not curves:
            continue
        for x, y in curves:
            ax.plot(x, y, color=color, alpha=0.12, lw=0.7)
        x0 = curves[0][0]
        stack = np.stack([y for _x, y in curves])
        with warnings.catch_warnings():
            # A bin empty in every frame is legitimately all-NaN.
            warnings.simplefilter('ignore', category=RuntimeWarning)
            mean_curve = np.nanmean(stack, axis=0)
        ax.plot(
            x0,
            mean_curve,
            color=color,
            lw=2.0,
            label=f'{label} mean (n={len(curves)})',
        )
    if log_y:
        ax.set_yscale('log')
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, loc='best')
    ax.grid(alpha=0.2)


def _kinds_with_prefix(comparison: InstrumentComparison, prefix: str) -> list[str]:
    """Sample kinds starting with ``prefix`` present on either side."""
    kinds = set(comparison.real.samples) | set(comparison.sim.samples)
    return sorted(k for k in kinds if k.startswith(prefix))


def _figure_noise(comparison: InstrumentComparison, out: Path) -> bool:
    """FOM 1 figure: sky sigma / mean / signal sigma / sky PSD."""
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.5))
    _hist_overlay(
        axes[0, 0],
        comparison.real.samples.get('sky_sigma', []),
        comparison.sim.samples.get('sky_sigma', []),
        title='sky-patch noise sigma (paired difference)',
        xlabel='sigma (native units)',
        log_x=True,
    )
    _hist_overlay(
        axes[0, 1],
        comparison.real.samples.get('sky_mean_minus_floor', []),
        comparison.sim.samples.get('sky_mean_minus_floor', []),
        title='sky-patch mean above frame floor',
        xlabel='mean minus frame p1 (native units)',
    )
    _hist_overlay(
        axes[1, 0],
        comparison.real.samples.get('signal_sigma', []),
        comparison.sim.samples.get('signal_sigma', []),
        title='noise vs signal: uniform-patch sigma above sky',
        xlabel='sigma (native units)',
        log_x=True,
    )
    _curve_overlay(
        axes[1, 1],
        comparison.real.curves.get('sky_psd', []),
        comparison.sim.curves.get('sky_psd', []),
        title='sky spatial power spectrum (unit-total)',
        xlabel='spatial frequency (cycles/px)',
        ylabel='fraction of power',
        log_y=True,
    )
    fig.suptitle(f'{comparison.instrument}: FOM 1 sky noise', fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return True


def _figure_psf(comparison: InstrumentComparison, out: Path) -> bool:
    """FOM 2 figure: star radial profile and encircled-energy radii."""
    has_data = bool(
        comparison.real.samples.get('star_ee50') or comparison.sim.samples.get('star_ee50')
    )
    if not has_data:
        return False
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.6))
    _curve_overlay(
        axes[0],
        comparison.real.curves.get('star_profile', []),
        comparison.sim.curves.get('star_profile', []),
        title='star radial profile (peak-normalized)',
        xlabel='radius (px)',
        ylabel='relative intensity',
    )
    _hist_overlay(
        axes[1],
        comparison.real.samples.get('star_ee50', []),
        comparison.sim.samples.get('star_ee50', []),
        title='half-energy radius EE50',
        xlabel='radius (px)',
    )
    _hist_overlay(
        axes[2],
        comparison.real.samples.get('star_ee80', []),
        comparison.sim.samples.get('star_ee80', []),
        title='80%-energy radius EE80',
        xlabel='radius (px)',
    )
    fig.suptitle(f'{comparison.instrument}: FOM 2 star PSF / encircled energy', fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return True


def _figure_limb(comparison: InstrumentComparison, out: Path) -> bool:
    """FOM 3 figure: limb profile plus rise-width distributions per bin."""
    kinds = _kinds_with_prefix(comparison, 'limb_width')
    if not kinds:
        return False
    n_panels = 1 + len(kinds)
    n_cols = min(3, n_panels)
    n_rows = (n_panels + n_cols - 1) // n_cols
    fig, axes_grid = plt.subplots(n_rows, n_cols, figsize=(3.7 * n_cols, 3.4 * n_rows))
    axes = np.atleast_1d(axes_grid).ravel()
    _curve_overlay(
        axes[0],
        comparison.real.curves.get('limb_profile', []),
        comparison.sim.curves.get('limb_profile', []),
        title='normalized limb profile',
        xlabel='distance along outward normal (px)',
        ylabel='normalized intensity',
    )
    for ax, kind in zip(axes[1:], kinds, strict=False):
        _hist_overlay(
            ax,
            comparison.real.samples.get(kind, []),
            comparison.sim.samples.get(kind, []),
            title=f'10-90% rise width: {kind}',
            xlabel='width (px)',
        )
    for ax in axes[n_panels:]:
        ax.set_axis_off()
    fig.suptitle(
        f'{comparison.instrument}: FOM 3 limb gradient profiles '
        '(p = phase bin, r = resolution bin)',
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return True


def _figure_ring(comparison: InstrumentComparison, out: Path) -> bool:
    """FOM 4 figure: ring-edge radial profile and rise widths."""
    if not (
        comparison.real.samples.get('ring_edge_width')
        or comparison.sim.samples.get('ring_edge_width')
    ):
        return False
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6))
    _curve_overlay(
        axes[0],
        comparison.real.curves.get('ring_radial_profile', []),
        comparison.sim.curves.get('ring_radial_profile', []),
        title='ring-edge radial brightness profile',
        xlabel='distance along radial normal (px)',
        ylabel='normalized intensity',
    )
    _hist_overlay(
        axes[1],
        comparison.real.samples.get('ring_edge_width', []),
        comparison.sim.samples.get('ring_edge_width', []),
        title='ring-edge 10-90% rise width',
        xlabel='width (px)',
    )
    fig.suptitle(f'{comparison.instrument}: FOM 4 ring edges', fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return True


def _figure_dynrange(comparison: InstrumentComparison, out: Path) -> bool:
    """FOM 5 figure: per-stratum near-floor fraction and signal stretch."""
    kinds = (
        _kinds_with_prefix(comparison, 'frac_near_floor')
        + _kinds_with_prefix(comparison, 'frac_saturated')
        + _kinds_with_prefix(comparison, 'signal_p95_minus_p50')
    )
    if not kinds:
        return False
    n_cols = min(3, len(kinds))
    n_rows = (len(kinds) + n_cols - 1) // n_cols
    fig, axes_grid = plt.subplots(n_rows, n_cols, figsize=(3.7 * n_cols, 3.2 * n_rows))
    axes = np.atleast_1d(axes_grid).ravel()
    for ax, kind in zip(axes, kinds, strict=False):
        _hist_overlay(
            ax,
            comparison.real.samples.get(kind, []),
            comparison.sim.samples.get(kind, []),
            title=kind,
            xlabel='fraction' if kind.startswith('frac') else 'native units',
        )
    for ax in axes[len(kinds) :]:
        ax.set_axis_off()
    fig.suptitle(f'{comparison.instrument}: FOM 5 dynamic range (exposure-stratified)', fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return True


def _figure_artifacts(comparison: InstrumentComparison, out: Path) -> bool:
    """FOM 6 figure: measured incidence rates with catalog-default markers."""
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6))
    _hist_overlay(
        axes[0],
        comparison.real.samples.get('artifact_missing_line_frac', []),
        comparison.sim.samples.get('artifact_missing_line_frac', []),
        title='missing/interpolated line fraction',
        xlabel='fraction of rows',
    )
    _hist_overlay(
        axes[1],
        comparison.real.samples.get('artifact_spike_frac', []),
        comparison.sim.samples.get('artifact_spike_frac', []),
        title='single-pixel spike fraction',
        xlabel='fraction of pixels',
        log_x=True,
    )
    defaults = resolve_detector_defaults(comparison.instrument)
    hot_default = float(defaults.get('hot_pixel_fraction', 0.0))
    if hot_default > 0.0:
        axes[1].axvline(
            hot_default,
            color='#444444',
            ls='--',
            lw=1.0,
            label=f'catalog hot_pixel_fraction {hot_default:g}',
        )
        axes[1].legend(fontsize=7, loc='best')
    fig.suptitle(f'{comparison.instrument}: FOM 6 artifact incidence', fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return True


def write_figures(results: RealismResults, *, figures_root: Path | None = None) -> list[Path]:
    """Write every supported figure; returns the paths written.

    Parameters:
        results: The realism-match results.
        figures_root: Output directory override (tests); defaults to the
            simulator report's ``_figures`` directory.
    """
    root = figures_root if figures_root is not None else _FIGURES_ROOT
    root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    makers = {
        'noise': _figure_noise,
        'psf': _figure_psf,
        'limb': _figure_limb,
        'ring': _figure_ring,
        'dynrange': _figure_dynrange,
        'artifacts': _figure_artifacts,
    }
    for instrument, comparison in sorted(results.comparisons.items()):
        for fom_name, maker in makers.items():
            out = root / f'realism_{instrument}_{fom_name}.png'
            if maker(comparison, out):
                written.append(out)
    return written


def _round(value: float, ndigits: int = 6) -> float | None:
    """JSON-safe rounded float (None for non-finite)."""
    if not np.isfinite(value):
        return None
    return round(float(value), ndigits)


def _summary_dict(results: RealismResults) -> dict[str, Any]:
    """The deterministic JSON-serializable summary."""
    out: dict[str, Any] = {'runtime_sec': round(results.runtime_sec, 1), 'instruments': {}}
    for instrument, comparison in sorted(results.comparisons.items()):
        defaults = resolve_detector_defaults(instrument)
        kinds: dict[str, Any] = {}
        for kind, div in sorted(comparison.divergences.items()):
            real_values = np.asarray(comparison.real.samples.get(kind, []), dtype=np.float64)
            sim_values = np.asarray(comparison.sim.samples.get(kind, []), dtype=np.float64)
            kinds[kind] = {
                'w1': _round(div.w1),
                'w1_normalized': _round(div.w1_normalized),
                'real_iqr': _round(div.real_iqr),
                'n_real': div.n_real,
                'n_sim': div.n_sim,
                'real_median': _round(float(np.median(real_values))) if real_values.size else None,
                'sim_median': _round(float(np.median(sim_values))) if sim_values.size else None,
            }
        curve_kinds: dict[str, Any] = {}
        for kind, div in sorted(comparison.curve_divergences.items()):
            curve_kinds[kind] = {
                'w1': _round(div.w1),
                'w1_normalized': _round(div.w1_normalized),
                'real_iqr': _round(div.real_iqr),
                'n_real_curves': len(comparison.real.curves.get(kind, [])),
                'n_sim_curves': len(comparison.sim.curves.get(kind, [])),
            }
        out['instruments'][instrument] = {
            'n_frames': len(comparison.records),
            'frames': [r.image_id for r in comparison.records],
            'fom_support': dict(sorted(comparison.fom_support.items())),
            'fom_frames': dict(sorted(comparison.fom_frames.items())),
            'divergences': kinds,
            'curve_divergences': curve_kinds,
            'limb_bins_real_only': list(comparison.limb_bins_real_only),
            'limb_bins_sim_only': list(comparison.limb_bins_sim_only),
            'spike_split_real': [
                _round(comparison.spike_split_real[0]),
                _round(comparison.spike_split_real[1]),
            ],
            'spike_split_sim': [
                _round(comparison.spike_split_sim[0]),
                _round(comparison.spike_split_sim[1]),
            ],
            'catalog_defaults': {
                'hot_pixel_fraction': float(defaults.get('hot_pixel_fraction', 0.0)),
                'cosmic_ray_events_per_pixel_per_sec': float(
                    defaults.get('cosmic_ray_events_per_pixel_per_sec', 0.0)
                ),
                'read_noise_e': float(defaults.get('read_noise_e', 0.0))
                if 'read_noise_e' in defaults
                else None,
            },
            'fom7_rows': comparison.fom7_rows,
        }
    return out


def write_summary(results: RealismResults, *, results_root: Path | None = None) -> Path:
    """Write the JSON summary; returns its path.

    Parameters:
        results: The realism-match results.
        results_root: Output directory override (tests); defaults to
            ``tests/integration/realism_results``.
    """
    root = results_root if results_root is not None else _RESULTS_ROOT
    root.mkdir(parents=True, exist_ok=True)
    path = root / 'realism_summary.json'
    payload = _summary_dict(results)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n')
    return path
