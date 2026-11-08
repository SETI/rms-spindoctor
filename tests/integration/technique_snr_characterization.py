"""Per-technique accuracy-vs-SNR characterization across background conditions.

Runner-only (NOT part of pytest): for every navigation technique (disc, blob,
limb, ring edge, star field, haze symmetry) this sweeps the per-image read noise
from a clean frame down toward the navigability cliff -- i.e. across a wide
signal-to-noise range -- at a fixed representative sub-pixel offset, under two
backgrounds (nominal and a stray-light gradient).  It writes one comparison
figure per background to ``docs/simulator_report/_figures/`` so each
technique's recovered-offset error can be read against SNR side by side.

The SNR axis is a uniform per-image proxy ``(peak - background) / robust_noise`` so
the extended-feature techniques (disc, limb, ring) and the point-source technique
(star) land on a comparable scale; the underlying knob swept is
``noise.read_noise_dn``.

Run with::

    PYTHONPATH=. python -m tests.integration.technique_snr_characterization

The figures feed the simulator performance report.  A real image's true offset is
unknown, so the whole characterization is simulated.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from spindoctor.nav_model import build_models_for_obs
from spindoctor.nav_orchestrator import NavOrchestrator
from spindoctor.obs.obs_inst_sim import ObsSim
from spindoctor.sim.scene import load_sim_scene
from tests.integration.checkout_guard import require_checkout_package

_SCENES = Path(__file__).parent / 'sim_scenes'
_FIGURES_ROOT = Path(__file__).parent.parent.parent / 'docs' / 'simulator_report' / '_figures'

# For the SNR-vs-accuracy figures: a fixed, modest sub-pixel offset inside every
# technique's capture range, so the only variable across that sweep is the noise
# (and the background).
_OFFSET = (0.317, -0.211)
# Read-noise grid (DN) for the SNR-vs-accuracy figures: clean frame down toward the
# navigability cliff.
_READ_NOISE = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
_SEEDS = list(range(4))

# For the accuracy-vs-injected-offset figures: vary a pure-vertical injected offset
# (``offset_u`` held at 0 so the x-axis is the injected magnitude) over a grid that
# includes the half / quarter / non-power-of-two anchors, at each of a small set of
# SNR levels.  The high-SNR panel mirrors the report's dense fractional sweep.
_OFFSET_GRID = [0.0, 0.1, 0.137, 0.25, 0.382, 0.5, 0.611, 0.75, 0.823, 1.0, 1.25, 1.5, 1.75]
_OFFSET_SWEEP_SEEDS = list(range(3))
# (label, read_noise_dn) -- a small set of SNR choices for the offset panels.
_SNR_LEVELS = [
    ('high SNR (read noise 1 DN)', 1.0, 'technique_offset_high_snr.png'),
    ('medium SNR (read noise 8 DN)', 8.0, 'technique_offset_medium_snr.png'),
    ('low SNR (read noise 32 DN)', 32.0, 'technique_offset_low_snr.png'),
]

# (label, base scene, technique, marker)
_TECHNIQUES = [
    ('disc', 'phase_sweep_regular_body/regular_sphere_base.yaml', 'BodyDiscCorrelateNav', 'o'),
    ('blob', 'phase_sweep_regular_body/small_sphere_base.yaml', 'BodyBlobNav', 's'),
    ('limb', 'algorithmic_invariants/planted_offset_limb.yaml', 'BodyLimbNav', '^'),
    ('ring', 'algorithmic_invariants/planted_offset_ring.yaml', 'RingEdgeNav', 'D'),
    (
        'star',
        'algorithmic_invariants/planted_offset_star_field.yaml',
        'StarFieldFromCatalogNav',
        'v',
    ),
    ('titan haze', 'atmosphere/titan_haze.yaml', 'TitanHazeNav', 'P'),
]

# (label, stray_light dict | None, output filename)
_BACKGROUNDS = [
    ('nominal', None, 'technique_snr_nominal.png'),
    (
        'stray-light gradient',
        {'model': 'linear', 'amplitude': 0.03, 'direction_deg': 35.0},
        'technique_snr_gradient.png',
    ),
]


def _sim_params(
    base: str,
    read_noise: float,
    stray: dict[str, Any] | None,
    *,
    offset: tuple[float, float] = _OFFSET,
    seed: int | None = None,
) -> dict[str, Any]:
    """Build sim params for a base scene at ``offset``, given noise + background."""
    params = load_sim_scene(_SCENES / base)
    params['offset_v'] = offset[0]
    params['offset_u'] = offset[1]
    params['noise'] = {**params.get('noise', {}), 'poisson': True, 'read_noise_dn': read_noise}
    if stray is not None:
        params['optics'] = {'stray_light': dict(stray)}
    if seed is not None:
        params['random_seed'] = seed
    return params


def _offset_error(
    params: dict[str, Any], technique: str, offset: tuple[float, float] = _OFFSET
) -> float | None:
    """Pin ``technique``, navigate, and return the recovered-offset error."""
    obs = ObsSim.from_file('/tmp/tech_snr.json', sim_params=params)
    result = NavOrchestrator(
        build_models_for_obs(obs), only_models='*', only_techniques=technique
    ).navigate(obs)
    pinned = next((t for t in result.per_technique if t.technique_name == technique), None)
    if pinned is None or pinned.spurious or pinned.offset_px is None:
        return None
    return math.hypot(pinned.offset_px[0] - offset[0], pinned.offset_px[1] - offset[1])


def _image_snr(params: dict[str, Any]) -> float:
    """Uniform per-image SNR proxy: (peak - background) / robust noise sigma.

    The noise sigma is estimated from adjacent-pixel differences (divided by
    ``sqrt(2)``) so a smooth stray-light gradient does not inflate it; otherwise
    the ramp would dominate a whole-image MAD and crush the SNR axis.
    """
    obs = ObsSim.from_file('/tmp/tech_snr.json', sim_params=params)
    image = np.asarray(obs.data, np.float64)
    background = float(np.median(image))
    diffs = image[1:, :] - image[:-1, :]
    mad = float(np.median(np.abs(diffs - np.median(diffs))))
    noise_sigma = (1.4826 * mad / math.sqrt(2.0)) or 1.0
    return (float(image.max()) - background) / noise_sigma


def _collect() -> dict[str, dict[str, Any]]:
    """Run the full grid; return per-background per-technique SNR/error curves."""
    out: dict[str, dict[str, Any]] = {}
    for bg_label, stray, out_name in _BACKGROUNDS:
        curves: dict[str, dict[str, list[float]]] = {}
        for tech_label, base, technique, _marker in _TECHNIQUES:
            snr_axis: list[float] = []
            err_axis: list[float] = []
            for read_noise in _READ_NOISE:
                snr_axis.append(_image_snr(_sim_params(base, read_noise, stray, seed=0)))
                errs = [
                    e
                    for seed in _SEEDS
                    if (
                        e := _offset_error(
                            _sim_params(base, read_noise, stray, seed=seed), technique
                        )
                    )
                    is not None
                ]
                err_axis.append(float(np.median(errs)) if errs else math.nan)
            curves[tech_label] = {'snr': snr_axis, 'err': err_axis}
        out[bg_label] = {'curves': curves, 'out_name': out_name}
        print(f'{bg_label}: done')
    return out


def _collect_offset_sweep() -> dict[str, dict[str, Any]]:
    """Run accuracy-vs-injected-offset for every technique at each SNR level."""
    out: dict[str, dict[str, Any]] = {}
    for level_label, read_noise, out_name in _SNR_LEVELS:
        curves: dict[str, list[float]] = {}
        for tech_label, base, technique, _marker in _TECHNIQUES:
            errs: list[float] = []
            for offset_v in _OFFSET_GRID:
                offset = (offset_v, 0.0)
                vals = [
                    e
                    for seed in _OFFSET_SWEEP_SEEDS
                    if (
                        e := _offset_error(
                            _sim_params(base, read_noise, None, offset=offset, seed=seed),
                            technique,
                            offset,
                        )
                    )
                    is not None
                ]
                errs.append(float(np.median(vals)) if vals else math.nan)
            curves[tech_label] = errs
        out[level_label] = {'curves': curves, 'out_name': out_name}
        print(f'{level_label}: done')
    return out


def generate() -> list[Path]:
    """Render one error-vs-SNR comparison figure per background condition."""
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    _FIGURES_ROOT.mkdir(parents=True, exist_ok=True)
    marker_for = {label: marker for label, _, _, marker in _TECHNIQUES}
    written: list[Path] = []

    # Family 1: accuracy vs SNR, at a fixed injected offset, one panel per background.
    offset_note = f'injected offset = ({_OFFSET[0]:+.2f}, {_OFFSET[1]:+.2f}) px (dv, du)'
    for bg_label, payload in _collect().items():
        fig, ax = plt.subplots(figsize=(8.0, 5.0))
        for tech_label, _, technique, _marker in _TECHNIQUES:
            curve = payload['curves'][tech_label]
            ax.plot(
                curve['snr'],
                curve['err'],
                marker=marker_for[tech_label],
                ms=4,
                lw=1.3,
                label=technique,
            )
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('per-image SNR proxy ((peak - background) / robust noise)')
        ax.set_ylabel('recovered-offset error (px, log scale)')
        ax.set_title(
            f'Per-technique accuracy vs SNR -- {bg_label} background\n{offset_note}', fontsize=11
        )
        ax.grid(True, which='both', ls=':', alpha=0.5)
        ax.legend(fontsize=8, loc='best')
        fig.tight_layout()
        path = _FIGURES_ROOT / payload['out_name']
        fig.savefig(path, dpi=110)
        plt.close(fig)
        written.append(path)

    # Family 2: accuracy vs injected offset, one panel per SNR level.  Share a y-range
    # so the panels are directly comparable as SNR drops.
    for level_label, payload in _collect_offset_sweep().items():
        fig, ax = plt.subplots(figsize=(8.0, 5.0))
        for tech_label, _, technique, _marker in _TECHNIQUES:
            ax.plot(
                _OFFSET_GRID,
                payload['curves'][tech_label],
                marker=marker_for[tech_label],
                ms=4,
                lw=1.3,
                label=technique,
            )
        ax.set_yscale('log')
        ax.set_ylim(1.0e-3, 2.0)
        ax.set_xlabel('injected offset along v (px), u held at 0')
        ax.set_ylabel('recovered-offset error (px, log scale)')
        ax.set_title(f'Per-technique accuracy vs injected offset -- {level_label}')
        ax.grid(True, which='both', ls=':', alpha=0.5)
        ax.legend(fontsize=8, loc='best')
        fig.tight_layout()
        path = _FIGURES_ROOT / payload['out_name']
        fig.savefig(path, dpi=110)
        plt.close(fig)
        written.append(path)
    return written


if __name__ == '__main__':
    require_checkout_package(__file__)
    for fig_path in generate():
        print(f'wrote {fig_path}')
