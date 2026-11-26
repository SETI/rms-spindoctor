"""Tests that model selection honors a per-run config override (issue #146)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np

from spindoctor.config import Config
from spindoctor.nav_model.nav_model_body import NavModelBody
from spindoctor.nav_model.nav_model_rings import NavModelRings


class _FakeJupiterObs:
    """Observation stand-in exposing what ring-model selection inspects."""

    closest_planet = 'JUPITER'
    extdata_shape_vu = (10, 10)
    ext_bp = object()


def _config_with_jupiter_rings(tmp_path: Path) -> Config:
    override = tmp_path / 'override.yaml'
    override.write_text(
        "rings:\n  ring_features:\n    JUPITER:\n      epoch: '2000-01-01 12:00:00'\n"
        '      fade_width_pix: 100.0\n      min_allowed_fade_width_pix: 3.0\n'
        '      min_feature_pixels: 2.0\n      features: {}\n',
        encoding='utf-8',
    )
    config = Config()
    config.update_config(override)
    return config


def test_rings_selection_defaults_to_no_jupiter_model() -> None:
    # The bundled defaults carry no Jupiter ring catalog, so no model applies.
    instances = NavModelRings.instances_for_obs(cast(Any, _FakeJupiterObs()))
    assert instances == []


def test_rings_selection_honors_config_override(tmp_path: Path) -> None:
    # A per-run override that adds a Jupiter ring catalog must change which
    # models are instantiated, not just how the instances behave.
    config = _config_with_jupiter_rings(tmp_path)
    instances = NavModelRings.instances_for_obs(cast(Any, _FakeJupiterObs()), config=config)
    assert len(instances) == 1
    assert instances[0].name == 'rings:JUPITER'
    assert instances[0]._config is config


class _FakeSaturnObs:
    """Observation stand-in with an inventory for body-model selection."""

    closest_planet = 'saturn'

    def inventory(self, body_list: list[str], return_type: str = 'full') -> dict[str, Any]:
        """Report every requested body as present in the FOV."""
        del return_type
        entry = {'center_uv': np.array([5.0, 5.0]), 'u_pixel_size': 4.0, 'v_pixel_size': 4.0}
        return dict.fromkeys(body_list, entry)

    def inventory_body_in_extfov(self, entry: dict[str, Any]) -> bool:
        """Every reported body is inside the extended FOV."""
        del entry
        return True


def _config_with_titan_only_satellites(tmp_path: Path) -> Config:
    override = tmp_path / 'satellites_override.yaml'
    # List values replace wholesale on merge, so this shrinks Saturn's
    # satellite catalog to a single body.
    override.write_text('satellites:\n  SATURN:\n    - TITAN\n')
    config = Config()
    config.update_config(override)
    return config


def test_body_selection_uses_config_satellite_catalog(tmp_path: Path) -> None:
    # The satellite catalog in the supplied config decides which bodies are
    # even considered.  Titan is handled as a special opaque-atmosphere case,
    # so it never builds a shape-based NavModelBody even when it is the only
    # satellite: only Saturn remains.
    config = _config_with_titan_only_satellites(tmp_path)
    instances = NavModelBody.instances_for_obs(cast(Any, _FakeSaturnObs()), config=config)
    assert sorted(m.name for m in instances) == ['body:saturn']
    default_instances = NavModelBody.instances_for_obs(cast(Any, _FakeSaturnObs()))
    assert len(default_instances) > len(instances)


def test_body_selection_threads_config_to_instances(tmp_path: Path) -> None:
    # The selection stage and the constructed instances must share the same
    # per-run config object.
    config = _config_with_jupiter_rings(tmp_path)
    instances = NavModelBody.instances_for_obs(cast(Any, _FakeSaturnObs()), config=config)
    assert len(instances) > 0
    assert all(m._config is config for m in instances)
