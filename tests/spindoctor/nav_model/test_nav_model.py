"""The NavModel registry and the builder's choice of model classes.

``build_models_for_obs`` asks a registered class for instances only when the class's
``inputs`` matches the observation's ``model_source``.  These tests record which
classes the builder asks, so they need no SPICE geometry and no rendered scene.
"""

from types import SimpleNamespace
from typing import cast

import pytest

from spindoctor.config import Config
from spindoctor.nav_model import NavModel, build_models_for_obs
from spindoctor.nav_model.stars.nav_model_stars import NavModelStars
from spindoctor.obs import ObsSnapshotInst


def _classes_asked(monkeypatch: pytest.MonkeyPatch, model_source: str) -> list[str]:
    """Return the shipped model classes the builder asks for an observation.

    Parameters:
        monkeypatch: Replaces each registered class's ``instances_for_obs`` with a
            recorder for the duration of the test.
        model_source: The observation's ``model_source``.

    Returns:
        The names of the shipped classes asked, in registry order.
    """
    asked: list[str] = []

    def _record(
        cls: type[NavModel], obs: ObsSnapshotInst, *, config: Config | None = None
    ) -> list[NavModel]:
        del obs, config
        asked.append(cls.__name__)
        return []

    registry = NavModel._registry
    shipped = {cls.__name__ for cls in registry if cls.__module__.startswith('spindoctor.')}
    for cls in registry:
        monkeypatch.setattr(cls, 'instances_for_obs', classmethod(_record))
    build_models_for_obs(cast(ObsSnapshotInst, SimpleNamespace(model_source=model_source)))
    return [name for name in asked if name in shipped]


def test_scene_view_observation_asks_only_the_simulated_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A simulated frame's models come from the classes that read the scene view."""
    assert _classes_asked(monkeypatch, 'scene_view') == [
        'NavModelStarsSimulated',
        'NavModelBodySimulated',
        'NavModelRingsSimulated',
        'NavModelTitanSimulated',
    ]


def test_spice_observation_asks_only_the_production_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real frame's models come from the classes that work from SPICE geometry."""
    assert _classes_asked(monkeypatch, 'spice') == [
        'NavModelStars',
        'NavModelBody',
        'NavModelRings',
        'NavModelTitan',
    ]


def test_concrete_model_must_set_its_own_inputs() -> None:
    """A registered class that only inherits ``inputs`` is refused at definition.

    The production star model is concrete and builds from SPICE, so a subclass that
    forgot to set ``inputs`` would otherwise be built for every real frame.
    """
    registry_before = list(NavModel._registry)
    with pytest.raises(TypeError, match='_Forgetful must set inputs'):

        class _Forgetful(NavModelStars):
            pass

    assert NavModel._registry == registry_before
