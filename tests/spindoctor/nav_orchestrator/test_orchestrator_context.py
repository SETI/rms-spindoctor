"""Tests for the per-image NavContext the orchestrator builds from an observation."""

from __future__ import annotations

from pathlib import Path

from spindoctor.nav_orchestrator.orchestrator import NavOrchestrator
from spindoctor.nav_orchestrator.provenance import Provenance
from spindoctor.obs.obs_inst_sim import ObsSim


def _sim_obs(tmp_path: Path) -> ObsSim:
    """Return a small noiseless emulated coiss_nac observation with no bodies."""
    params: dict[str, object] = {
        'size_v': 64,
        'size_u': 64,
        'random_seed': 7,
        'instrument': 'coiss_nac',
        'exposure_sec': 1.0,
        'offset_v': 0.0,
        'offset_u': 0.0,
        'bodies': [],
        'noise': {'poisson': False, 'read_noise_dn': 0.0, 'bias_dn': 0.0},
    }
    return ObsSim.from_file(str(tmp_path / 'scene.json'), sim_params=params)


def test_context_carries_the_camera_model_error_floors(
    tmp_path: Path, provenance: Provenance
) -> None:
    """The instrument config's per-technique floors reach the NavContext techniques read."""
    obs = _sim_obs(tmp_path)
    assert obs._inst_config is not None
    obs._inst_config = {**obs._inst_config, 'model_error_floor_px': {'RingAnnulusNav': 0.9}}
    context, _classifier = NavOrchestrator([])._make_context(obs, provenance)
    assert dict(context.model_error_floor_px) == {'RingAnnulusNav': 0.9}
