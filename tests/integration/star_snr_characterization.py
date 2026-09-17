"""Star-field centroiding characterization across SNR and background conditions.

Runner-only (NOT part of pytest): sweeps a uniform-brightness star field across a
wide integrated-SNR range and, at each step, compares the three centroiding modes
of :class:`~spindoctor.nav_technique.nav_technique_star_field.StarFieldFromCatalogNav` --
moment-only, the PSF-fit-everywhere extreme, and the shipped SNR-adaptive choice --
under several background conditions (clean, elevated read noise, a stray-light
gradient).  It writes one comparison figure per background to
``docs/simulator_report/_figures/`` so the moment/PSF crossover and the adaptive
choice's lower-envelope behavior can be read off directly.

Run with::

    PYTHONPATH=. python -m tests.integration.star_snr_characterization

The figures feed the simulator performance report's star-field centroiding
section.  A real image's true offset is unknown, so the whole characterization is
simulated.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

import spindoctor.nav_technique.nav_technique_star_field as star_field_mod
from spindoctor.nav_model import build_models_for_obs
from spindoctor.nav_orchestrator import NavOrchestrator
from spindoctor.nav_technique.nav_technique_star_field import StarFieldFromCatalogNav
from spindoctor.obs.obs_inst_sim import ObsSim
from spindoctor.sim.scene import load_sim_scene

_TECHNIQUE = 'StarFieldFromCatalogNav'
_SCENES_ROOT = Path(__file__).parent / 'sim_scenes'
_BASE_SCENE = _SCENES_ROOT / 'algorithmic_invariants' / 'planted_offset_star_field_bright.yaml'
_FIGURES_ROOT = Path(__file__).parent.parent.parent / 'docs' / 'simulator_report' / '_figures'

# Uniform visual magnitude grid -> a wide brightness (and so SNR) range.  Bright
# (vmag 0) sits well above the crossover; faint (vmag 5.5) sits near the detection
# floor.  Peak DN scales as ~2.512**(-vmag).
_VMAGS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5]
_SEEDS = list(range(6))
_PLANTED = (0.382, -0.213)  # one representative off-grid sub-pixel offset

# (label, mode-override) for the three centroiding modes compared at every step.
_SHIPPED_SNR_MAX = 30.0  # mirrors config_510 psf_refine_snr_max
_MODES = [
    ('moment only', {'enabled': False, 'snr_max': 0.0}),
    ('PSF everywhere', {'enabled': True, 'snr_max': 1.0e18}),
    ('SNR-adaptive (shipped)', {'enabled': True, 'snr_max': _SHIPPED_SNR_MAX}),
]

# (label, noise dict, stray_light dict | None) background conditions.
_BACKGROUNDS = [
    ('clean', {'poisson': True, 'read_noise_dn': 4.0}, None, 'star_snr_clean.png'),
    (
        'elevated read noise',
        {'poisson': True, 'read_noise_dn': 20.0},
        None,
        'star_snr_highnoise.png',
    ),
    (
        'stray-light gradient',
        {'poisson': True, 'read_noise_dn': 4.0},
        {'model': 'linear', 'amplitude': 0.15, 'direction_deg': 35.0},
        'star_snr_gradient.png',
    ),
]


def _install_mode_patch() -> dict[str, Any]:
    """Monkeypatch the technique ctor so a mode override sets the refine knobs."""
    override: dict[str, Any] = {'enabled': True, 'snr_max': 40.0}
    original_init = StarFieldFromCatalogNav.__init__

    def patched(self: StarFieldFromCatalogNav, *, config: Any = None) -> None:
        original_init(self, config=config)
        self._psf_refine_enabled = bool(override['enabled'])
        self._psf_refine_snr_max = float(override['snr_max'])

    star_field_mod.StarFieldFromCatalogNav.__init__ = patched  # type: ignore[method-assign]
    return override


def _sim_params(vmag: float, noise: dict[str, Any], stray: dict[str, Any] | None) -> dict[str, Any]:
    """Build sim params for the base scene with a uniform vmag and a background."""
    params = load_sim_scene(_BASE_SCENE)
    params['offset_v'] = _PLANTED[0]
    params['offset_u'] = _PLANTED[1]
    params['noise'] = noise
    for star in params['stars']:
        star['vmag'] = vmag
    if stray is not None:
        params['optics'] = {'stray_light': dict(stray)}
    return params


def _field_error(params: dict[str, Any]) -> float | None:
    """Pin the star field, navigate, and return the recovered-offset error."""
    obs = ObsSim.from_file('/tmp/star_snr.json', sim_params=params)
    result = NavOrchestrator(
        build_models_for_obs(obs), only_models='*', only_techniques=_TECHNIQUE
    ).navigate(obs)
    pinned = next((t for t in result.per_technique if t.technique_name == _TECHNIQUE), None)
    if pinned is None or pinned.spurious or pinned.offset_px is None:
        return None
    return math.hypot(pinned.offset_px[0] - _PLANTED[0], pinned.offset_px[1] - _PLANTED[1])


def _median_star_snr(params: dict[str, Any]) -> float:
    """Median per-star box SNR over the planted star positions (mode-independent)."""
    obs = ObsSim.from_file('/tmp/star_snr.json', sim_params=params)
    image = np.asarray(obs.data, np.float64)
    noise_sigma = float(np.std(image[:16, :16])) or 1.0
    half = 5
    h, w = image.shape
    snrs: list[float] = []
    for star in params['stars']:
        v_pix = round(star['v'] + _PLANTED[0])
        u_pix = round(star['u'] + _PLANTED[1])
        if half <= v_pix < h - half and half <= u_pix < w - half:
            snrs.append(StarFieldFromCatalogNav._box_snr(image, v_pix, u_pix, half, noise_sigma))
    return float(np.median(snrs)) if snrs else 0.0


def _collect() -> dict[str, dict[str, Any]]:
    """Run the full grid; return per-background SNR axis and per-mode error curves."""
    override = _install_mode_patch()
    out: dict[str, dict[str, Any]] = {}
    for bg_label, noise, stray, out_name in _BACKGROUNDS:
        snr_axis: list[float] = []
        mode_curves: dict[str, list[float]] = {label: [] for label, _ in _MODES}
        for vmag in _VMAGS:
            base = _sim_params(vmag, noise, stray)
            snr_axis.append(_median_star_snr(base))
            for label, mode in _MODES:
                override['enabled'] = mode['enabled']
                override['snr_max'] = mode['snr_max']
                errs = [
                    e
                    for seed in _SEEDS
                    if (e := _field_error({**_sim_params(vmag, noise, stray), 'random_seed': seed}))
                    is not None
                ]
                mode_curves[label].append(float(np.median(errs)) if errs else math.nan)
        out[bg_label] = {'snr': snr_axis, 'curves': mode_curves, 'out_name': out_name}
        print(f'{bg_label}: SNR {snr_axis[-1]:.1f}..{snr_axis[0]:.1f}')
    return out


def generate() -> list[Path]:
    """Render one error-vs-SNR comparison figure per background condition."""
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    _FIGURES_ROOT.mkdir(parents=True, exist_ok=True)
    data = _collect()
    written: list[Path] = []
    for bg_label, payload in data.items():
        snr = payload['snr']
        fig, ax = plt.subplots(figsize=(8.0, 5.0))
        for label, _ in _MODES:
            ys = payload['curves'][label]
            ax.plot(snr, ys, marker='o', ms=4, lw=1.3, label=label)
        ax.axvline(
            _SHIPPED_SNR_MAX,
            color='gray',
            ls='--',
            lw=1.0,
            label=f'psf_refine_snr_max ({_SHIPPED_SNR_MAX:.0f})',
        )
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('median per-star integrated SNR')
        ax.set_ylabel('recovered-offset error (px, log scale)')
        ax.set_title(f'Star-field centroiding vs SNR -- {bg_label} background')
        ax.grid(True, which='both', ls=':', alpha=0.5)
        ax.legend(fontsize=8, loc='best')
        fig.tight_layout()
        path = _FIGURES_ROOT / payload['out_name']
        fig.savefig(path, dpi=110)
        plt.close(fig)
        written.append(path)
    return written


if __name__ == '__main__':
    for fig_path in generate():
        print(f'wrote {fig_path}')
