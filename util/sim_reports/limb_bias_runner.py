"""Command-line driver that measures and tabulates the limb-navigation bias.

Runs three blocks and writes CSV tables plus a console summary:

1. Simulator renderer validation.  Confirms the sim body renderer plants a
   body at its requested sub-pixel center with no positional bias, and
   quantifies how far the brightness gradient ridge sits inside the geometric
   limb (the photometric roll-off signature).

2. Sim planted-truth sweeps.  Navigates ``BodyLimbNav`` against noise-free
   sim scenes while sweeping, one axis at a time, the sub-pixel offset, the
   phase angle, the body diameter, and the illumination direction, recording
   the signed per-axis limb-fit error against planted truth.

3. Real-frame limb-versus-star gap.  Navigates the operator-curated
   ``stars_plus_body`` frames and records the signed gap between the
   ``BodyLimbNav`` offset and the independent star-technique offset on the
   same frame.  Requires ``PDS3_HOLDINGS_DIR`` (skipped otherwise).

Run with::

    PYTHONPATH=src python -m util.sim_reports.limb_bias_runner

CSV outputs land under ``util/calibration/limb_bias/`` by default.
"""

from __future__ import annotations

import copy
import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tests.integration.checkout_guard import require_checkout_package
from tests.integration.limb_bias import (
    LimbBiasSample,
    build_body_scene,
    measure_sim_limb_bias,
    renderer_centroid_offset,
    ridge_inset_phase_zero,
    technique_offset,
)

from spindoctor.nav_model import build_models_for_obs
from spindoctor.nav_orchestrator import NavOrchestrator

_REPO_ROOT = Path(__file__).resolve().parents[2]
_OUT_DIR = _REPO_ROOT / 'util' / 'calibration' / 'limb_bias'

# Fixed probe geometry for the single-axis sweeps.  A moderate phase and a
# well-resolved diameter put the frame in the regime where the limb fit is the
# primary technique; the non-zero sub-pixel offset exercises the interpolation
# phase without sitting on an integer grid point.
_PROBE_DIAMETER_PX = 160.0
_PROBE_PHASE_DEG = 30.0
_PROBE_ILLUM_DEG = 25.0
_PROBE_OFFSET = (0.3, 0.3)


# Star technique names used as the independent real-frame ground truth,
# most precise first.  The first one that produces a non-spurious result on a
# frame is taken as that frame's star reference offset.
_STAR_TECHNIQUES: tuple[str, ...] = (
    'StarRefineNav',
    'StarUniqueMatchNav',
    'StarFieldFromCatalogNav',
)


@dataclass(frozen=True)
class RealLimbStarSample:
    """Limb-versus-star offset comparison on a single real frame.

    Attributes:
        image_id: Short image identifier.
        limb_offset_vu: ``BodyLimbNav`` offset, or ``None``.
        star_offset_vu: Independent star-technique offset, or ``None``.
        star_technique: Name of the star technique that supplied the
            reference, or ``None``.
        gap_vu: Signed ``limb - star`` per axis, or ``None`` when either
            offset is missing.
    """

    image_id: str
    limb_offset_vu: tuple[float, float] | None
    star_offset_vu: tuple[float, float] | None
    star_technique: str | None
    gap_vu: tuple[float, float] | None

    @property
    def gap_mag_px(self) -> float | None:
        """Euclidean magnitude of the limb-minus-star gap, or ``None``."""
        if self.gap_vu is None:
            return None
        return math.hypot(self.gap_vu[0], self.gap_vu[1])


def measure_real_limb_vs_star(obs: Any, image_id: str) -> RealLimbStarSample:
    """Compare the limb and star offsets on one navigated real observation.

    Runs the full technique ensemble once and reads the ``BodyLimbNav``
    offset and the most precise available star-technique offset from the same
    navigation, so both are measured against the identical image geometry.

    Parameters:
        obs: A real :class:`ObsSnapshotInst` already built from an image.
        image_id: Short identifier for reporting.

    Returns:
        A :class:`RealLimbStarSample` carrying both offsets and their signed
        gap.
    """
    result = NavOrchestrator(
        build_models_for_obs(obs), only_models='*', only_techniques='*'
    ).navigate(obs)
    limb = technique_offset(result, 'BodyLimbNav')
    star: tuple[float, float] | None = None
    star_name: str | None = None
    for name in _STAR_TECHNIQUES:
        star = technique_offset(result, name)
        if star is not None:
            star_name = name
            break
    gap: tuple[float, float] | None = None
    if limb is not None and star is not None:
        gap = (limb[0] - star[0], limb[1] - star[1])
    return RealLimbStarSample(
        image_id=image_id,
        limb_offset_vu=limb,
        star_offset_vu=star,
        star_technique=star_name,
        gap_vu=gap,
    )


def sweep_scenes(
    axis: str,
    values: list[float],
    *,
    base: dict[str, Any],
) -> list[tuple[float, dict[str, Any]]]:
    """Return ``(value, sim_params)`` pairs varying one axis of a base scene.

    Parameters:
        axis: One of ``'phase'``, ``'diameter'``, ``'illumination'``,
            ``'offset_v'``, or ``'offset_u'``.
        values: The values to sweep.
        base: A base scene from :func:`build_body_scene`.

    Returns:
        One ``(value, sim_params)`` pair per value.

    Raises:
        ValueError: If ``axis`` is not a recognized sweep axis.
    """
    out: list[tuple[float, dict[str, Any]]] = []
    for value in values:
        scene = copy.deepcopy(base)
        body = scene['bodies'][0]
        if axis == 'phase':
            body['phase_angle'] = value
        elif axis == 'diameter':
            body['semi_a_px'] = body['semi_b_px'] = body['semi_c_px'] = value / 2.0
        elif axis == 'illumination':
            body['illumination_angle'] = value
        elif axis == 'offset_v':
            scene['offset_v'] = value
        elif axis == 'offset_u':
            scene['offset_u'] = value
        else:
            raise ValueError(f'unknown sweep axis: {axis!r}')
        out.append((value, scene))
    return out


def _base_scene() -> dict[str, Any]:
    return build_body_scene(
        diameter_px=_PROBE_DIAMETER_PX,
        phase_deg=_PROBE_PHASE_DEG,
        illumination_deg=_PROBE_ILLUM_DEG,
        offset_vu=_PROBE_OFFSET,
    )


def _fmt(sample: LimbBiasSample) -> str:
    if sample.error_vu is None:
        return 'no-result' + (' (spurious)' if sample.spurious else '')
    return (
        f'err=({sample.error_vu[0]:+.4f}, {sample.error_vu[1]:+.4f}) mag={sample.error_mag_px:.4f}'
    )


def run_renderer_validation(writer: Any) -> None:
    """Print and record the renderer geometry validation block."""
    print('\n=== Block 1: simulator renderer validation ===')
    print('Intensity-weighted centroid vs geometric center (phase 0 sphere):')
    writer.writerow(['sub_check', 'center_v', 'center_u', 'err_v_px', 'err_u_px', 'err_mag_px'])
    worst = 0.0
    for cv in (100.0, 100.25, 100.5, 100.75):
        for cu in (100.0, 100.33, 100.67):
            chk = renderer_centroid_offset(center_vu=(cv, cu), diameter_px=140.0)
            worst = max(worst, chk.centroid_error_mag_px)
            print(
                f'  center=({cv:.2f}, {cu:.2f}): '
                f'err=({chk.centroid_error_vu[0]:+.5f}, {chk.centroid_error_vu[1]:+.5f}) '
                f'mag={chk.centroid_error_mag_px:.5f} px'
            )
            writer.writerow(
                [
                    'centroid',
                    f'{cv:.2f}',
                    f'{cu:.2f}',
                    f'{chk.centroid_error_vu[0]:+.6f}',
                    f'{chk.centroid_error_vu[1]:+.6f}',
                    f'{chk.centroid_error_mag_px:.6f}',
                ]
            )
    verdict = 'CLEAN' if worst < 0.02 else 'BIASED'
    print(f'  renderer centroid worst-case error = {worst:.5f} px -> {verdict} (<< 0.1 px)')
    print('Brightness gradient-ridge inset inside the geometric limb (phase 0):')
    for diam in (120.0, 160.0, 220.0):
        inset = ridge_inset_phase_zero(diameter_px=diam)
        print(f'  diameter={diam:.0f} px: ridge sits {inset:+.3f} px inside geometric limb')
        writer.writerow(['ridge_inset', f'{diam:.0f}', '', '', '', f'{inset:+.6f}'])


def run_sim_sweeps(writer: Any) -> None:
    """Print and record every single-axis sim planted-truth sweep."""
    print('\n=== Block 2: sim planted-truth limb-fit bias sweeps (noise-free) ===')
    writer.writerow(['sweep', 'value', 'err_v_px', 'err_u_px', 'err_mag_px', 'spurious'])
    base = _base_scene()
    sweeps: list[tuple[str, str, list[float]]] = [
        ('offset_v', 'sub-pixel offset_v (offset_u=0.3)', [i / 10.0 for i in range(10)]),
        ('offset_u', 'sub-pixel offset_u (offset_v=0.3)', [i / 10.0 for i in range(10)]),
        ('phase', 'phase angle (deg)', [0.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0]),
        ('diameter', 'body diameter (px)', [110.0, 140.0, 170.0, 200.0, 230.0]),
        ('illumination', 'illumination direction (deg)', [float(a) for a in range(0, 360, 45)]),
    ]
    for axis, label, values in sweeps:
        print(f'\n  -- sweep: {label} --')
        for value, scene in sweep_scenes(axis, values, base=base):
            sample = measure_sim_limb_bias(scene)
            print(f'    {axis}={value:8.3f}: {_fmt(sample)}')
            ev = '' if sample.error_vu is None else f'{sample.error_vu[0]:+.6f}'
            eu = '' if sample.error_vu is None else f'{sample.error_vu[1]:+.6f}'
            em = '' if sample.error_mag_px is None else f'{sample.error_mag_px:.6f}'
            writer.writerow([axis, f'{value:.3f}', ev, eu, em, str(sample.spurious)])


def run_real_frames(writer: Any) -> None:
    """Print and record the real-frame limb-versus-star gap block."""
    print('\n=== Block 3: real-frame limb-vs-star gap (stars_plus_body) ===')
    if 'PDS3_HOLDINGS_DIR' not in os.environ:
        print('  PDS3_HOLDINGS_DIR not set; skipping real-frame block.')
        return
    from tests.integration.sidecar import LibraryRoot, load_sidecar
    from tests.integration.test_autonomous_nav import _MISSION_TO_OBS_CLASS, _resolve_pds3_url

    writer.writerow(
        [
            'image_id',
            'limb_v',
            'limb_u',
            'star_tech',
            'star_v',
            'star_u',
            'gap_v',
            'gap_u',
            'gap_mag',
        ]
    )
    paths = LibraryRoot().discover_sidecar_paths()
    for path in paths:
        if path.parent.name != 'stars_plus_body':
            continue
        sidecar = load_sidecar(path)
        obs_class = _MISSION_TO_OBS_CLASS[sidecar.mission]
        obs = obs_class.from_file(_resolve_pds3_url(sidecar.image_url))
        sample = measure_real_limb_vs_star(obs, sidecar.image_id)
        limb = sample.limb_offset_vu
        star = sample.star_offset_vu
        gap = sample.gap_vu
        print(
            f'  {sidecar.image_id}: limb={_pair(limb)} '
            f'star[{sample.star_technique}]={_pair(star)} '
            f'gap={_pair(gap)} mag={_optmag(sample.gap_mag_px)}'
        )
        writer.writerow(
            [
                sidecar.image_id,
                _c(limb, 0),
                _c(limb, 1),
                sample.star_technique or '',
                _c(star, 0),
                _c(star, 1),
                _c(gap, 0),
                _c(gap, 1),
                _optmag(sample.gap_mag_px),
            ]
        )


def _pair(vu: tuple[float, float] | None) -> str:
    return 'None' if vu is None else f'({vu[0]:+.4f}, {vu[1]:+.4f})'


def _c(vu: tuple[float, float] | None, i: int) -> str:
    return '' if vu is None else f'{vu[i]:+.6f}'


def _optmag(mag: float | None) -> str:
    return '' if mag is None else f'{mag:.4f}'


def main() -> None:
    """Run all three measurement blocks and write CSV tables."""
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (_OUT_DIR / 'renderer_validation.csv').open('w', newline='') as fh:
        run_renderer_validation(csv.writer(fh))
    with (_OUT_DIR / 'sim_sweeps.csv').open('w', newline='') as fh:
        run_sim_sweeps(csv.writer(fh))
    with (_OUT_DIR / 'real_limb_vs_star.csv').open('w', newline='') as fh:
        run_real_frames(csv.writer(fh))
    print(f'\nCSV tables written under {_OUT_DIR}')


if __name__ == '__main__':
    require_checkout_package(__file__)
    main()
