"""Model selection for simulated observations.

A simulated obs must build the sim-params-driven NavModels (not the SPICE-backed
ones), and the real models must opt out of a simulated obs.  Titan is the one
body routed away from the generic body model on both sides: its haze hides the
surface an ellipsoid prediction assumes, so it belongs to the haze model alone
and the two must never both claim it.
"""

from typing import Any, cast

import pytest

from spindoctor.nav_model import build_models_for_obs
from spindoctor.nav_model.nav_model_body import NavModelBody
from spindoctor.nav_model.nav_model_body_simulated import NavModelBodySimulated
from spindoctor.nav_model.nav_model_titan import NavModelTitan
from spindoctor.nav_model.nav_model_titan_simulated import (
    REQUIRED_SIM_PARAMS,
    NavModelTitanSimulated,
)
from spindoctor.obs import ObsSnapshotInst
from spindoctor.obs.obs_inst_sim import ObsSim


class _FakeObs:
    """Minimal stand-in for the is_simulated guard tests.

    Also carries a ``nav_params`` mapping, so a selection rule can be
    exercised on a body the renderer could not draw (a body with no stated
    size renders as a divide-by-zero ellipse, which would fail the fixture
    before the rule under test ran).
    """

    def __init__(self, *, is_simulated: bool, nav_params: dict[str, Any] | None = None) -> None:
        self.is_simulated = is_simulated
        self.nav_params = nav_params if nav_params is not None else {}


def _fake_obs(*, is_simulated: bool, nav_params: dict[str, Any] | None = None) -> ObsSnapshotInst:
    """Return a ``_FakeObs`` typed as the observation the selection rules take."""
    return cast(ObsSnapshotInst, _FakeObs(is_simulated=is_simulated, nav_params=nav_params))


def _sim_obs(bodies: list[dict[str, Any]]) -> ObsSim:
    return ObsSim.from_file(
        '/tmp/sel.json',
        sim_params={'size_v': 96, 'size_u': 96, 'instrument': 'coiss_nac', 'bodies': bodies},
    )


def _body(name: str, center: float) -> dict[str, Any]:
    return {
        'name': name,
        'center_v': center,
        'center_u': center,
        'axis1': 50.0,
        'axis2': 50.0,
        'axis3': 50.0,
        'illumination_angle': 20.0,
        'phase_angle': 30.0,
        # The soft anti-aliased rims of the two-body fixture overlap by a
        # fraction of a pixel, and overlapping bodies must carry explicit
        # compositing depths.
        'range_km': 500000.0 + center,
    }


def _titan_body(center: float = 48.0, **overrides: Any) -> dict[str, Any]:
    """A simulated Titan carrying every parameter the haze model needs."""
    return {**_body('TITAN', center), 'km_per_pixel': 80.0, **overrides}


def test_sim_obs_builds_simulated_body_model() -> None:
    """A simulated obs yields a NavModelBodySimulated for its body."""
    models = build_models_for_obs(_sim_obs([_body('RHEA', 48.0)]))
    assert any(isinstance(m, NavModelBodySimulated) for m in models)


def test_sim_obs_builds_one_model_per_body() -> None:
    """Each body in the sim scene gets its own simulated model."""
    models = build_models_for_obs(_sim_obs([_body('A', 30.0), _body('B', 66.0)]))
    body_models = [m for m in models if isinstance(m, NavModelBodySimulated)]
    assert len(body_models) == 2


def test_sim_obs_skips_spice_body_model() -> None:
    """The SPICE-backed body model is not built for a simulated obs."""
    models = build_models_for_obs(_sim_obs([_body('RHEA', 48.0)]))
    assert not any(isinstance(m, NavModelBody) for m in models)


def test_simulated_model_empty_for_real_obs() -> None:
    """NavModelBodySimulated builds nothing for a non-simulated obs."""
    assert NavModelBodySimulated.instances_for_obs(_fake_obs(is_simulated=False)) == []


def test_spice_model_empty_for_simulated_obs() -> None:
    """NavModelBody builds nothing for a simulated obs."""
    assert NavModelBody.instances_for_obs(_fake_obs(is_simulated=True)) == []


def test_sim_titan_builds_the_haze_model() -> None:
    """A configured simulated Titan yields a NavModelTitanSimulated."""
    models = build_models_for_obs(_sim_obs([_titan_body()]))
    assert any(isinstance(m, NavModelTitanSimulated) for m in models)


def test_sim_titan_does_not_build_a_body_model() -> None:
    """The generic simulated body model declines Titan.

    Without the exclusion both models would claim the same body, and the
    shape-based techniques would fit an ellipsoid limb the haze hides.
    """
    models = build_models_for_obs(_sim_obs([_titan_body()]))
    assert not any(isinstance(m, NavModelBodySimulated) for m in models)


def test_sim_non_titan_body_is_unaffected_by_the_exclusion() -> None:
    """Every other body still gets its simulated body model."""
    models = build_models_for_obs(_sim_obs([_body('RHEA', 48.0)]))
    assert any(isinstance(m, NavModelBodySimulated) for m in models)


def test_sim_non_titan_body_builds_no_haze_model() -> None:
    """A body that is not Titan is never routed to the haze model."""
    models = build_models_for_obs(_sim_obs([_body('RHEA', 48.0)]))
    assert not any(isinstance(m, NavModelTitanSimulated) for m in models)


def test_sim_mixed_scene_routes_each_body_once() -> None:
    """In a Titan-plus-moon scene each body reaches exactly one model family."""
    models = build_models_for_obs(_sim_obs([_titan_body(30.0), _body('RHEA', 70.0)]))
    body_models = [m for m in models if isinstance(m, NavModelBodySimulated)]
    haze_models = [m for m in models if isinstance(m, NavModelTitanSimulated)]
    assert [m.name for m in body_models] == ['body_sim:RHEA']
    assert [m.name for m in haze_models] == ['titan_sim:TITAN']


def test_sim_titan_without_pixel_scale_builds_no_model() -> None:
    """An unconfigured simulated Titan yields no model instead of crashing.

    The pixel scale is what turns the configured atmosphere height into an
    envelope radius; without it the haze model has no envelope to bound the
    fit with, so it declines and the frame resolves through the standard
    generic reasons for a scene with nothing to navigate.
    """
    body = _titan_body()
    del body['km_per_pixel']
    models = build_models_for_obs(_sim_obs([body]))
    assert models == []


def test_sim_titan_without_axes_builds_no_model() -> None:
    """A simulated Titan with no stated size yields no model."""
    body = _titan_body()
    del body['axis2']
    obs = _fake_obs(is_simulated=True, nav_params={'bodies': [body]})
    assert NavModelTitanSimulated.instances_for_obs(obs) == []


def test_sim_titan_with_every_required_param_builds_one_model() -> None:
    """The guard admits a body carrying all of REQUIRED_SIM_PARAMS.

    The companion to the two omission tests: without this, deleting a
    required key from the tuple would silently stop being tested.
    """
    obs = _fake_obs(is_simulated=True, nav_params={'bodies': [_titan_body()]})
    instances = NavModelTitanSimulated.instances_for_obs(obs)
    assert len(instances) == 1


@pytest.mark.parametrize('key', sorted(REQUIRED_SIM_PARAMS))
def test_required_sim_param_is_load_bearing(key: str) -> None:
    """Omitting any single required parameter suppresses the model."""
    body = _titan_body()
    del body[key]
    obs = _fake_obs(is_simulated=True, nav_params={'bodies': [body]})
    assert NavModelTitanSimulated.instances_for_obs(obs) == []


def test_titan_sim_model_empty_for_real_obs() -> None:
    """NavModelTitanSimulated builds nothing for a non-simulated obs."""
    assert NavModelTitanSimulated.instances_for_obs(_fake_obs(is_simulated=False)) == []


def test_spice_titan_model_empty_for_simulated_obs() -> None:
    """NavModelTitan builds nothing for a simulated obs."""
    assert NavModelTitan.instances_for_obs(_fake_obs(is_simulated=True)) == []
