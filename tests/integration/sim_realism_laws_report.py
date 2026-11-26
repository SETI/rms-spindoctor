"""The photometric-law study's figures, its block of the realism record, and its checks.

The study (:mod:`tests.integration.sim_realism_laws`) writes three figures beside the
realism match's own:

- ``realism_law_fits.png``: the real discs' fitted Minnaert exponent and lunar-Lambert
  blend against phase, one marker per body, with McEwen's blend and the straight line
  the Minnaert twins take their exponent from;
- ``realism_disc_profiles.png``: each disc's twin exponent less its real one against
  phase under every law, and the default law's check by phase bin;
- ``realism_law_sheet.png``: real discs beside their twins under every law, each row at
  the real disc's own stretch.

Its record block, ``law_study``, holds every disc's fits and twins, the moons left out
with their reasons, the statistics by phase bin, by body, and by limb bin, and the law
comparison.  Each instrument's entry also holds the default law's ``disc_profiles``.
:func:`record_disc_profile_problems` and :func:`record_law_comparison_problems` read a
record back for the regression tests.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from spindoctor.sim.render.bodies.photometry import lunar_lambert_blend
from tests.integration.sim_realism_laws import (
    DEFAULT_LAW,
    DiscProfileBin,
    DiscRow,
    LawBin,
    LawComparisonBin,
    LawStudy,
)
from tests.integration.sim_realism_limb_peak import LimbPeakBin
from tests.integration.sim_realism_support import PHASE_BIN_EDGES_DEG

__all__ = [
    'disc_profile_entries',
    'law_study_summary',
    'record_disc_profile_problems',
    'record_has_law_study',
    'record_law_comparison_problems',
    'write_law_figures',
]

# One color per law (a colorblind-safe set), the default law's first.
_LAW_COLORS = {
    'lunar_lambert': '#d9752e',
    'lambert': '#31688e',
    'lommel_seeliger': '#35b779',
    'minnaert': '#8e44ad',
}
_REAL_COLOR = '#31688e'
_SIM_COLOR = '#d9752e'
_MARKERS = ('o', 's', '^', 'v', 'D', 'P', 'X', '<', '>', '*', 'h')
_SHEET_ROWS = 8
_SHEET_MIN_DIAMETER_PX = 100.0


def _round(value: float | None, digits: int = 6) -> float | None:
    """A JSON-safe float rounded to ``digits`` significant digits, or None."""
    if value is None or not math.isfinite(value):
        return None
    return float(f'{float(value):.{digits}g}')


def _law_bin_entry(entry: LawBin) -> dict[str, Any]:
    """One law's statistics over a group of discs, as the record holds them."""
    return {
        'n': entry.n,
        'median_delta_k': _round(entry.median_delta_k),
        'median_abs_delta_k': _round(entry.median_abs_delta_k),
        'median_profile_residual': _round(entry.median_profile_residual),
    }


def _limb_entry(peak: LimbPeakBin) -> dict[str, Any]:
    """One limb bin's comparison under one law, as the record holds it."""
    return {
        'n_real': peak.n_real,
        'n_sim': peak.n_sim,
        'real_median_px': _round(peak.real_median_px),
        'sim_median_px': _round(peak.sim_median_px),
        'band_px': _round(peak.band_px),
        'within_band': peak.within_band,
    }


def _comparison_entry(entry: LawComparisonBin) -> dict[str, Any]:
    """One phase bin's law comparison, as the record holds it."""
    return {
        'n': entry.n,
        'band_k': _round(entry.band_k),
        'agreement': {law: _round(v) for law, v in sorted(entry.agreement.items())},
        'separates': dict(sorted(entry.separates.items())),
        'default_best': entry.default_best,
    }


def _disc_entry(row: DiscRow) -> dict[str, Any]:
    """One disc's fits and twins, as the record holds them."""
    disc = row.disc
    blend = disc.lunar_lambert
    return {
        'image_id': disc.frame.image_id,
        'instrument': disc.frame.instrument,
        'body': disc.body,
        'part': disc.part,
        'phase_deg': _round(disc.phase_deg),
        'diameter_px': _round(disc.diameter_px),
        'n_pixels': disc.minnaert.n_pixels,
        'k': _round(disc.minnaert.k),
        'k_sigma': _round(disc.minnaert.k_sigma),
        'blend': None if blend is None else _round(blend.blend),
        'blend_sigma': None if blend is None else _round(blend.blend_sigma),
        'limb_peak_kind': None if disc.limb_peak is None else disc.limb_peak[0],
        'limb_peak_px': None if disc.limb_peak is None else _round(disc.limb_peak[1]),
        'twins': {
            law: {
                'normal_albedo': _round(twin.normal_albedo),
                'minnaert_k': _round(twin.minnaert_k),
                'k': _round(twin.k),
                'delta_k': _round(row.delta_k(law)),
                'profile_residual': _round(twin.profile_residual),
                'limb_peak_px': _round(twin.limb_peak),
            }
            for law, twin in sorted(row.twins.items())
        },
    }


def disc_profile_entries(bins: dict[str, DiscProfileBin]) -> dict[str, Any]:
    """An instrument's disc-profile check, as its record entry holds it.

    Parameters:
        bins: The check's phase bins.

    Returns:
        One entry per bin, keyed by its label.
    """
    return {
        label: {
            'n': entry.n,
            'real_median_k': _round(entry.real_median_k),
            'sim_median_k': _round(entry.sim_median_k),
            'median_delta_k': _round(entry.median_delta_k),
            'real_scatter_k': _round(entry.real_scatter_k),
            'band_k': _round(entry.band_k),
            'within_band': entry.within_band,
        }
        for label, entry in sorted(bins.items())
    }


def law_study_summary(study: LawStudy) -> dict[str, Any]:
    """The study's block of the realism record.

    Parameters:
        study: The study.

    Returns:
        A JSON-serializable mapping with sorted contents.
    """
    rows = sorted(study.rows, key=lambda r: (r.disc.frame.image_id, r.disc.label))
    return {
        'default_law': DEFAULT_LAW,
        'laws': sorted(study.law_bins),
        'minnaert_k_line': {
            'k_at_zero_phase': _round(study.minnaert_k0),
            'slope_per_deg': _round(study.minnaert_k1),
        },
        'discs': [_disc_entry(row) for row in rows],
        'excluded': [
            {'image_id': e.image_id, 'body': e.body, 'reason': e.reason}
            for e in sorted(study.excluded, key=lambda e: (e.image_id, e.body))
        ],
        'phase_bins': {
            law: {label: _law_bin_entry(v) for label, v in sorted(bins.items())}
            for law, bins in sorted(study.law_bins.items())
        },
        'bodies': {
            law: {label: _law_bin_entry(v) for label, v in sorted(bins.items())}
            for law, bins in sorted(study.body_bins.items())
        },
        'limb_peaks': {
            law: {label: _limb_entry(v) for label, v in sorted(bins.items())}
            for law, bins in sorted(study.limb_peaks.items())
        },
        'comparison': {
            label: _comparison_entry(v) for label, v in sorted(study.comparison.items())
        },
    }


def _body_markers(rows: list[DiscRow]) -> dict[str, str]:
    """One marker per body label, in alphabetical order."""
    labels = sorted({row.disc.label for row in rows})
    return {label: _MARKERS[i % len(_MARKERS)] for i, label in enumerate(labels)}


def _phase_bin_lines(ax: Any) -> None:
    """Mark the phase-bin edges on a phase axis."""
    for edge in PHASE_BIN_EDGES_DEG:
        ax.axvline(edge, color='0.7', lw=0.8, ls=':')


def _figure_fits(study: LawStudy, out: Path) -> None:
    """The real discs' fitted exponent and blend against phase, one marker per body."""
    markers = _body_markers(study.rows)
    fig, (ax_k, ax_l) = plt.subplots(1, 2, figsize=(11.0, 4.4))
    phases = np.linspace(0.0, 180.0, 181)
    for row in study.rows:
        disc = row.disc
        marker = markers[disc.label]
        ax_k.errorbar(
            disc.phase_deg,
            disc.minnaert.k,
            yerr=disc.minnaert.k_sigma,
            fmt=marker,
            color=_REAL_COLOR,
            ms=6,
            mfc='none',
            label=disc.label,
        )
        if disc.lunar_lambert is not None:
            ax_l.errorbar(
                disc.phase_deg,
                disc.lunar_lambert.blend,
                yerr=disc.lunar_lambert.blend_sigma,
                fmt=marker,
                color=_REAL_COLOR,
                ms=6,
                mfc='none',
            )
    ax_k.plot(
        phases,
        study.minnaert_k0 + study.minnaert_k1 * phases,
        color=_LAW_COLORS['minnaert'],
        lw=1.2,
        label='line the Minnaert twins use',
    )
    ax_l.plot(
        phases,
        [lunar_lambert_blend(math.radians(a)) for a in phases],
        color=_LAW_COLORS['lunar_lambert'],
        lw=1.5,
        label="McEwen's L(phase)",
    )
    for ax, ylabel in ((ax_k, 'Minnaert exponent k'), (ax_l, 'lunar-Lambert blend L')):
        _phase_bin_lines(ax)
        ax.set_xlabel('phase angle (deg)')
        ax.set_ylabel(ylabel)
        ax.set_xlim(0.0, 180.0)
    ax_l.set_ylim(-0.6, 1.6)
    handles, labels = ax_k.get_legend_handles_labels()
    unique = dict(zip(labels, handles, strict=True))
    ax_k.legend(unique.values(), unique.keys(), fontsize=7, loc='lower right')
    ax_l.legend(fontsize=8, loc='upper right')
    fig.suptitle('Real moon discs: fitted Minnaert exponent and lunar-Lambert blend', fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def _figure_profiles(study: LawStudy, out: Path) -> None:
    """Each disc's twin exponent less its real one under every law, by phase."""
    fig, (ax_d, ax_c) = plt.subplots(1, 2, figsize=(11.0, 4.4))
    for law in sorted(study.law_bins):
        points = [(r.disc.phase_deg, r.delta_k(law)) for r in study.rows]
        kept = [(p, d) for p, d in points if d is not None]
        if kept:
            ax_d.plot(
                [p for p, _ in kept],
                [d for _, d in kept],
                'o',
                ms=4,
                color=_LAW_COLORS.get(law, 'k'),
                label=law.replace('_', '-'),
                alpha=0.8,
            )
    ax_d.axhline(0.0, color='0.4', lw=0.8)
    _phase_bin_lines(ax_d)
    ax_d.set_xlabel('phase angle (deg)')
    ax_d.set_ylabel('twin k less real k')
    ax_d.set_title('every law', fontsize=10)
    ax_d.legend(fontsize=8)
    real = [(r.disc.phase_deg, r.disc.minnaert.k) for r in study.rows]
    twin = [
        (r.disc.phase_deg, r.twins[DEFAULT_LAW].k)
        for r in study.rows
        if DEFAULT_LAW in r.twins and r.twins[DEFAULT_LAW].k is not None
    ]
    ax_c.plot([p for p, _ in real], [k for _, k in real], 'o', color=_REAL_COLOR, label='real')
    ax_c.plot(
        [p for p, _ in twin],
        [k for _, k in twin],
        's',
        color=_SIM_COLOR,
        mfc='none',
        label=f'twin ({DEFAULT_LAW.replace("_", "-")})',
    )
    _phase_bin_lines(ax_c)
    ax_c.set_xlabel('phase angle (deg)')
    ax_c.set_ylabel('Minnaert exponent k')
    ax_c.set_title('the default law', fontsize=10)
    ax_c.legend(fontsize=8)
    fig.suptitle('Disc profiles: twins against their real discs', fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def _sheet_rows(study: LawStudy) -> list[DiscRow]:
    """Up to 8 whole discs at least 100 px across, spread over phase."""
    candidates = sorted(
        (
            r
            for r in study.rows
            if r.disc.part is None and r.disc.diameter_px >= _SHEET_MIN_DIAMETER_PX
        ),
        key=lambda r: r.disc.phase_deg,
    )
    if len(candidates) <= _SHEET_ROWS:
        return candidates
    picks = np.unique(np.round(np.linspace(0, len(candidates) - 1, _SHEET_ROWS)).astype(int))
    return [candidates[i] for i in picks]


def _figure_sheet(study: LawStudy, out: Path) -> None:
    """Real discs beside their twins under every law, each row at its real stretch."""
    rows = _sheet_rows(study)
    laws = sorted(study.law_bins)
    fig, axes = plt.subplots(
        len(rows), 1 + len(laws), figsize=(2.0 * (1 + len(laws)), 2.0 * len(rows)), squeeze=False
    )
    for i, row in enumerate(rows):
        real = row.disc.crop
        top = float(np.nanpercentile(real, 99.5)) if np.isfinite(real).any() else 1.0
        panels = [('real', real)] + [(law.replace('_', '-'), row.twins[law].crop) for law in laws]
        for j, (title, image) in enumerate(panels):
            ax = axes[i, j]
            ax.imshow(image, cmap='gray', vmin=0.0, vmax=max(top, 1e-12), interpolation='nearest')
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(title, fontsize=9)
        axes[i, 0].set_ylabel(
            f'{row.disc.label}\n{row.disc.phase_deg:.0f} deg, {row.disc.diameter_px:.0f} px',
            fontsize=8,
        )
    fig.suptitle('Real moon discs and their twins under each law', fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=90)
    plt.close(fig)


def write_law_figures(study: LawStudy, root: Path) -> list[Path]:
    """Write the study's three figures.

    Parameters:
        study: The study.
        root: The figures directory.

    Returns:
        The paths written.
    """
    written: list[Path] = []
    for name, maker in (
        ('realism_law_fits.png', _figure_fits),
        ('realism_disc_profiles.png', _figure_profiles),
        ('realism_law_sheet.png', _figure_sheet),
    ):
        path = root / name
        maker(study, path)
        written.append(path)
    return written


def record_has_law_study(record: dict[str, Any]) -> bool:
    """Whether a realism record holds the photometric-law study.

    Parameters:
        record: A realism summary.

    Returns:
        True when the record carries a ``law_study`` block.
    """
    return 'law_study' in record


def record_disc_profile_problems(record: dict[str, Any]) -> list[str]:
    """The phase bins of a record whose default-law disc profiles leave their band.

    Parameters:
        record: A realism summary.

    Returns:
        One line per bin whose verdict is False; bins without a verdict are left out.
    """
    problems: list[str] = []
    for instrument, entry in sorted(record.get('instruments', {}).items()):
        for label, check in sorted(entry.get('disc_profiles', {}).items()):
            if check['within_band'] is not False:
                continue
            problems.append(
                f"{instrument} {label}: the twins' Minnaert exponents differ from the real "
                f'ones by a median {check["median_delta_k"]:+.3f}, more than the band of '
                f'{check["band_k"]:.3f} ({check["n"]} discs)'
            )
    return problems


def record_law_comparison_problems(record: dict[str, Any]) -> list[str]:
    """The phase bins of a record where another law agrees better than the default.

    Parameters:
        record: A realism summary.

    Returns:
        One line per bin and per law that separates from the default law and agrees
        better with the real discs.
    """
    study = record.get('law_study', {})
    default = study.get('default_law', DEFAULT_LAW)
    problems: list[str] = []
    for label, entry in sorted(study.get('comparison', {}).items()):
        agreement = entry['agreement']
        for law, apart in sorted(entry['separates'].items()):
            if apart and agreement[law] < agreement[default]:
                problems.append(
                    f'{label}: {law} agrees with the real discs to a median |dk| of '
                    f"{agreement[law]:.3f}, better than {default}'s {agreement[default]:.3f} "
                    f'by more than the band of {entry["band_k"]:.3f} ({entry["n"]} discs)'
                )
    return problems
