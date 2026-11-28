"""Render the limb-bias diagnosis plots from the generated CSV tables.

Reads ``sim_sweeps.csv`` next to this file and writes two compact PNGs:
``bias_vs_illumination.png`` (the bias vector as the lit limb rotates) and
``bias_vs_subpixel.png`` (the signed per-axis error versus sub-pixel offset
phase).  Run after ``util/sim_reports/limb_bias_runner.py`` has produced the CSVs::

    PYTHONPATH=src python util/calibration/limb_bias/make_plots.py
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent


def _rows(sweep: str) -> list[tuple[float, float, float]]:
    out: list[tuple[float, float, float]] = []
    with (_HERE / 'sim_sweeps.csv').open() as fh:
        for row in csv.DictReader(fh):
            if row['sweep'] != sweep or not row['err_v_px']:
                continue
            out.append((float(row['value']), float(row['err_v_px']), float(row['err_u_px'])))
    return out


def plot_illumination() -> None:
    """Bias vector versus illumination direction on a polar-style quiver."""
    rows = _rows('illumination')
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    for illum, ev, eu in rows:
        ax.arrow(0.0, 0.0, eu, ev, head_width=0.008, length_includes_head=True, color='C0')
        ax.annotate(f'{int(illum)}', (eu, ev), fontsize=7)
    ax.set_xlabel('u error (px)')
    ax.set_ylabel('v error (px)')
    ax.set_title('Limb-fit bias vector vs illumination direction')
    ax.axhline(0.0, color='0.8', lw=0.5)
    ax.axvline(0.0, color='0.8', lw=0.5)
    ax.set_aspect('equal')
    fig.tight_layout()
    fig.savefig(_HERE / 'bias_vs_illumination.png', dpi=90)
    plt.close(fig)


def plot_subpixel() -> None:
    """Signed per-axis error versus sub-pixel offset phase (interpolation ripple)."""
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    for sweep, marker in (('offset_v', 'o'), ('offset_u', 's')):
        rows = _rows(sweep)
        xs = [r[0] for r in rows]
        mags = [math.hypot(r[1], r[2]) for r in rows]
        ax.plot(xs, mags, marker=marker, label=f'{sweep} sweep |error|')
    ax.set_xlabel('sub-pixel offset (px)')
    ax.set_ylabel('|limb-fit error| (px)')
    ax.set_title('Limb-fit error vs sub-pixel offset phase')
    ax.axhline(0.0, color='0.8', lw=0.5)
    ax.legend(fontsize=8, loc='best')
    fig.tight_layout()
    fig.savefig(_HERE / 'bias_vs_subpixel.png', dpi=90)
    plt.close(fig)


def main() -> None:
    """Render both plots."""
    plot_illumination()
    plot_subpixel()
    print(f'Plots written under {_HERE}')


if __name__ == '__main__':
    main()
