"""Image-side telemetry stage: what transmission loses or mangles.

The telemetry stage runs at the detector grid (after the box downsample), so
its loss geometry is not coupled to the oversampling factor: a missing line is
one detector line, and a missing block aligns to the detector-row compression
grid.  It applies its sub-effects in the physical order of transmission:

1. Lossy DCT compression (``compression_dct``): the codec blockiness a lossy
   downlink plants on the transmitted signal before any packet loss, honoring
   the commanded ``truth_window`` carve-out.
2. Structured data loss: the registry loss modes in
   :data:`~spindoctor.sim.forward.artifact_modes.STRUCTURED_LOSS_ORDER`
   (commanded frame shapes, then line losses, then block losses, then garble,
   then per-pixel losses, then the row-0 header), with the commanded
   ``truth_window`` carve-out resolved first and passed to ``missing_blocks``.
   Each mode is honored only where its registry availability lists the
   instrument, and its shape parameters resolve through the per-instrument
   catalog (scene value over catalog default over registry default), like the
   flanking artifact loops.
3. Voyager GEOMED archive-processing scars, applied to the already-loss-bearing
   frame: ``reseau_scars`` (reseau-removal smudges on the lattice) then
   ``resample_texture`` (the GEOMED resample warp, blank border, and
   missing-line interpolation banding).
4. Missing-data markers: the generic per-pixel ``noise.missing_data_rate``
   dropout knob (a worst-case stress knob, not an instrument artifact),
   applied on the raw-DN path only.

The stage also records ``frame.truth['rendered_marker_frac']``: the share of
the frame's pixels that sat at the marker before telemetry ran and still do
after it.  Those pixels are at the marker because the scene renders them there
(a ``vgiss`` frame's dark sky at exactly 0 I/F); the rest of the frame's marker
pixels are planted loss.  The renderer's sky-at-marker warning reads it.

Every structured loss mode is opt-in through the ``artifacts`` block, disabled
when its activation key (``expected_events`` or ``probability``), or its fixed
``count`` where given, is 0, and individually seeded via
``derive_effect_seed(random_seed, 'telemetry/<mode>')`` so toggling one never
perturbs another.  Each applied mode records its realized geometry into
``frame.truth['artifacts'][mode]`` for later planted-vs-measured comparison.
Adversarial placement (``artifacts.adversarial``) biases each stochastic mode
onto the navigation features; it is uniform otherwise.
"""

from collections.abc import Mapping
from typing import Any

import numpy as np

from spindoctor.config import DEFAULT_CONFIG
from spindoctor.sim.forward.artifact_modes import (
    STRUCTURED_LOSS_ORDER,
    TELEMETRY_POST_LOSS_ORDER,
    TELEMETRY_PRE_LOSS_ORDER,
    mode_available,
    resolve_mode_config,
)
from spindoctor.sim.forward.artifacts_catalog import resolve_mode_with_catalog
from spindoctor.sim.forward.feature_loci import FeatureLoci, extract_feature_loci
from spindoctor.sim.forward.stages import SimFrame
from spindoctor.sim.forward.telemetry_artifacts import (
    apply_compression_dct,
    apply_resample_texture,
    apply_reseau_scars,
)
from spindoctor.sim.forward.telemetry_loss import LOSS_APPLIERS
from spindoctor.sim.instruments import navigator_marker_value, resolve_sim_inst_config
from spindoctor.sim.seeds import derive_effect_seed
from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType

__all__ = ['apply_telemetry']

_EMPTY_LOCI = FeatureLoci(
    rows=np.empty(0, dtype=np.int64),
    pixel_v=np.empty(0, dtype=np.int64),
    pixel_u=np.empty(0, dtype=np.int64),
)


def apply_telemetry(
    frame: SimFrame,
    *,
    params: Mapping[str, Any],
    rng: np.random.Generator,
) -> None:
    """Telemetry stage: apply structured data loss and missing-data markers.

    Runs after the detector stage, so loss overwrites readout values the way a
    downlink dropout erases transmitted pixels.

    Parameters:
        frame: The frame whose signal plane is modified in place; its
            ``truth['artifacts']`` map receives each applied mode's realized
            geometry, and ``truth['rendered_marker_frac']`` the share of pixels
            at the marker both before and after the stage.  Must be at the
            detector grid (``oversample == 1``).
        params: The full scene mapping; reads the ``artifacts`` block (per-mode
            maps plus ``adversarial``) and the ``noise`` block's
            ``missing_data_rate``.
        rng: The stage generator, used for the generic missing-data markers; the
            structured loss modes derive their own per-mode streams from the
            scene seed.
    """
    if frame.oversample != 1:
        raise AssertionError(
            'telemetry runs at the detector grid; downsample must precede it '
            f'(oversample={frame.oversample})'
        )
    artifacts = params.get('artifacts') or {}
    inst_config = resolve_sim_inst_config(
        DEFAULT_CONFIG, params.get('instrument'), params.get('instrument_config')
    )
    marker_dn = navigator_marker_value(inst_config)
    dn_ceiling = _saturation_value(inst_config)
    random_seed = int(params.get('random_seed', 42))
    adversarial = bool(artifacts.get('adversarial', False))
    loci = extract_feature_loci(frame.truth, frame.signal.shape) if adversarial else _EMPTY_LOCI

    instrument = params.get('instrument')
    rendered_at_marker = _at_marker(frame.signal, marker_dn)
    records: dict[str, Any] = {}
    # The commanded truth window is resolved first so both compression and
    # missing_blocks can leave it clean.
    protect = _resolve_truth_window(artifacts, frame.signal.shape, random_seed, records)

    # Sub-stage 1: lossy DCT compression, on the transmitted signal before any
    # packet loss (a lossy codec compresses, then packets drop).
    for mode_name in TELEMETRY_PRE_LOSS_ORDER:
        raw_cfg = artifacts.get(mode_name)
        if raw_cfg is None or not mode_available(mode_name, instrument):
            continue
        cfg = resolve_mode_with_catalog(mode_name, raw_cfg, instrument)
        mode_rng = np.random.default_rng(derive_effect_seed(random_seed, f'telemetry/{mode_name}'))
        records[mode_name] = apply_compression_dct(frame.signal, cfg, rng=mode_rng, protect=protect)

    # Sub-stage 2: structured data loss.
    for mode_name in STRUCTURED_LOSS_ORDER:
        raw_cfg = artifacts.get(mode_name)
        if raw_cfg is None or not mode_available(mode_name, instrument):
            continue
        cfg = resolve_mode_with_catalog(mode_name, raw_cfg, instrument)
        mode_rng = np.random.default_rng(derive_effect_seed(random_seed, f'telemetry/{mode_name}'))
        result = LOSS_APPLIERS[mode_name](
            frame.signal,
            cfg,
            marker_dn=marker_dn,
            dn_ceiling=dn_ceiling,
            rng=mode_rng,
            loci=loci,
            adversarial=adversarial,
            protect=protect,
        )
        records[mode_name] = result

    # Sub-stage 3: Voyager GEOMED archive-processing scars, applied to the
    # already-loss-bearing frame (they emulate archive processing, not raw
    # readout): reseau-removal smudges, then the GEOMED resample texture.
    for mode_name in TELEMETRY_POST_LOSS_ORDER:
        raw_cfg = artifacts.get(mode_name)
        if raw_cfg is None or not mode_available(mode_name, instrument):
            continue
        cfg = resolve_mode_with_catalog(mode_name, raw_cfg, instrument)
        mode_rng = np.random.default_rng(derive_effect_seed(random_seed, f'telemetry/{mode_name}'))
        if mode_name == 'reseau_scars':
            records[mode_name] = apply_reseau_scars(frame.signal, cfg, rng=mode_rng)
        else:
            records[mode_name] = apply_resample_texture(frame.signal, cfg, rng=mode_rng)

    # Sub-stage 4: the generic missing-data markers (raw-DN path only).
    _apply_missing_data_markers(frame, params, inst_config, marker_dn, rng)

    if records:
        frame.truth.setdefault('artifacts', {}).update(records)
    still_at_marker = rendered_at_marker & _at_marker(frame.signal, marker_dn)
    frame.truth['rendered_marker_frac'] = (
        float(still_at_marker.mean()) if still_at_marker.size else 0.0
    )


def _at_marker(signal: NDArrayFloatType, marker: float) -> NDArrayBoolType:
    """Return the mask of pixels carrying the marker (NaN pixels for a NaN marker)."""
    return np.isnan(signal) if np.isnan(marker) else signal == marker


def _saturation_value(inst_config: Mapping[str, Any]) -> float:
    """The emulated instrument's ADC ceiling in DN (255 for the 8-bit cameras).

    The loss appliers bound the wrong values they write (garble fills, pixel
    spikes) by this ceiling, so an 8-bit camera never carries an impossible DN.
    """
    inst_noise = inst_config.get('noise') or {}
    sim_noise = DEFAULT_CONFIG.category('sim')['noise']
    return float(inst_noise.get('saturation_dn', sim_noise['saturation_dn']))


def _resolve_truth_window(
    artifacts: Mapping[str, Any],
    shape: tuple[int, int],
    random_seed: int,
    records: dict[str, Any],
) -> tuple[int, int, int, int] | None:
    """Resolve the commanded truth-window carve-out (a protected clean region).

    The truth window is a losslessly-clean square that missing_blocks (and,
    later, compression) must leave untouched.  It is a commanded region:
    activation is by its ``probability``, drawn from its own stream.  Returns the
    protected ``(v0, v1, u0, u1)`` rectangle, or None when it is not active, and
    records its geometry.
    """
    raw_cfg = artifacts.get('truth_window')
    if raw_cfg is None:
        return None
    cfg = resolve_mode_config('truth_window', raw_cfg)
    rng = np.random.default_rng(derive_effect_seed(random_seed, 'telemetry/truth_window'))
    probability = float(cfg['probability'])
    if probability <= 0.0 or not bool(rng.random() < probability):
        records['truth_window'] = {'active': False}
        return None
    size_v, size_u = shape
    size = int(cfg['size'])
    position = cfg.get('position')
    if position is not None:
        v0, u0 = int(position[0]), int(position[1])
    else:
        v0 = (size_v - size) // 2
        u0 = (size_u - size) // 2
    v0 = max(0, min(v0, size_v))
    u0 = max(0, min(u0, size_u))
    v1 = min(size_v, v0 + size)
    u1 = min(size_u, u0 + size)
    records['truth_window'] = {'active': True, 'rect': [v0, v1, u0, u1]}
    return v0, v1, u0, u1


def _apply_missing_data_markers(
    frame: SimFrame,
    params: Mapping[str, Any],
    inst_config: Mapping[str, Any],
    marker_dn: float,
    rng: np.random.Generator,
) -> None:
    """Overwrite lost pixels with the missing-data marker (raw-DN path only).

    The generic per-pixel ``noise.missing_data_rate`` dropout stays as the
    worst-case stress knob, applied only on the raw-DN path.  The structured loss
    modes write the marker on every path, calibrated frames included.
    """
    if inst_config.get('data_units', 'raw_dn') != 'raw_dn':
        return
    sim_noise = DEFAULT_CONFIG.category('sim')['noise']
    scene_noise = params.get('noise') or {}
    # A null rate reads as absent and takes the configuration default.
    scene_rate = scene_noise.get('missing_data_rate')
    missing_data_rate = float(
        sim_noise.get('missing_data_rate', 0.0) if scene_rate is None else scene_rate
    )
    if missing_data_rate <= 0.0:
        return
    missing_mask = rng.random(size=frame.signal.shape) < missing_data_rate
    frame.signal[missing_mask] = marker_dn
