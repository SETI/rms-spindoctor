"""Validation of a body's ``atmosphere`` block: albedo, phase terms, and named models."""

from typing import Any

import pytest
from tests.spindoctor.sim.scenes import disc_body, frame_scene

from spindoctor.sim.scene import rules_body
from spindoctor.sim.scene.errors import SimSceneValidationError
from spindoctor.sim.scene.validate import validate_sim_params

_SIZE = 96
_CENTER = 48.0
_RADIUS = 15.0


def _compose_body() -> dict[str, Any]:
    """A centered atmospheric body with an explicit haze column."""
    body = disc_body(_CENTER, _CENTER, _RADIUS, name='TITAN', illumination_angle=0.0)
    body['atmosphere'] = {'scale_height_px': 5.0, 'tau_ref': 2.0, 'g': 0.6}
    return body


def _albedo_scene(albedo: float) -> dict[str, Any]:
    """A minimal scene whose one atmospheric body states ``albedo``."""
    body = _compose_body()
    body['atmosphere']['albedo'] = albedo
    return frame_scene(size=_SIZE, seed=3, exposure_sec=1.0, bodies=[body])


@pytest.mark.parametrize('albedo', [0.0, 0.4, 1.0])
def test_a_haze_albedo_in_the_unit_interval_validates(albedo: float) -> None:
    """Any haze albedo in [0, 1] validates and is kept as stated."""
    scene = validate_sim_params(_albedo_scene(albedo))
    assert scene['bodies'][0]['atmosphere']['albedo'] == albedo


@pytest.mark.parametrize('albedo', [-0.1, 1.5])
def test_a_haze_albedo_outside_the_unit_interval_is_refused(albedo: float) -> None:
    """A haze albedo below 0 or above 1 fails validation and names the key."""
    with pytest.raises(SimSceneValidationError, match=r'atmosphere\.albedo'):
        validate_sim_params(_albedo_scene(albedo))


def _second_term_scene(**keys: float) -> dict[str, Any]:
    """A minimal scene whose one atmospheric body states the given phase keys."""
    body = _compose_body()
    body['atmosphere'].update(keys)
    return frame_scene(size=_SIZE, seed=3, exposure_sec=1.0, bodies=[body])


def test_a_second_phase_term_validates() -> None:
    """A second Henyey-Greenstein term with its share validates and is kept."""
    scene = validate_sim_params(_second_term_scene(g2=-0.2, g2_fraction=0.1))
    assert scene['bodies'][0]['atmosphere']['g2_fraction'] == 0.1


def test_a_second_phase_term_without_its_share_is_refused() -> None:
    """``g2`` without ``g2_fraction`` fails validation and names both keys."""
    with pytest.raises(SimSceneValidationError, match=r'g2 and .*g2_fraction'):
        validate_sim_params(_second_term_scene(g2=-0.2))


def test_a_second_phase_term_share_without_its_term_is_refused() -> None:
    """``g2_fraction`` without ``g2`` fails validation and names both keys."""
    with pytest.raises(SimSceneValidationError, match=r'g2 and .*g2_fraction'):
        validate_sim_params(_second_term_scene(g2_fraction=0.1))


def test_a_second_phase_term_asymmetry_outside_the_open_interval_is_refused() -> None:
    """A second-term asymmetry of 1 or more fails validation and names the key."""
    with pytest.raises(SimSceneValidationError, match=r'atmosphere\.g2 must lie in \(-1, 1\)'):
        validate_sim_params(_second_term_scene(g2=1.0, g2_fraction=0.1))


def test_a_second_phase_term_share_above_one_is_refused() -> None:
    """A second-term share above 1 fails validation and names the key."""
    with pytest.raises(SimSceneValidationError, match=r'atmosphere\.g2_fraction'):
        validate_sim_params(_second_term_scene(g2=-0.2, g2_fraction=1.5))


def _titan_model_scene(*, instrument: str = 'coiss_nac', **atmosphere: Any) -> dict[str, Any]:
    """A minimal scene whose one body takes Titan's haze by name."""
    body = disc_body(_CENTER, _CENTER, _RADIUS, name='TITAN')
    body['atmosphere'] = {'haze_model': 'titan', **atmosphere}
    return frame_scene(instrument=instrument, size=_SIZE, seed=3, exposure_sec=1.0, bodies=[body])


def _with_filter(scene: dict[str, Any], filter_name: str) -> dict[str, Any]:
    """The scene with ``detector.filter`` set."""
    scene['detector'] = {'filter': filter_name}
    return scene


def test_a_titan_model_haze_validates_in_the_clear_filter() -> None:
    """A body may take Titan's haze by name in the camera's clear filter."""
    scene = validate_sim_params(_titan_model_scene())
    assert scene['bodies'][0]['atmosphere'] == {'haze_model': 'titan'}


def test_a_titan_model_haze_that_also_sets_its_column_is_refused() -> None:
    """A block that names the model and also sets tau_ref is refused, naming the key."""
    with pytest.raises(SimSceneValidationError, match=r"model sets the haze.*'tau_ref'"):
        validate_sim_params(_titan_model_scene(tau_ref=2.0))


def test_a_titan_model_haze_with_a_detached_shell_is_refused() -> None:
    """A detached shell cannot join a named haze model, whose depth would set the shell's."""
    with pytest.raises(SimSceneValidationError, match=r'detached_px cannot join a haze_model'):
        validate_sim_params(_titan_model_scene(detached_px=10.0))


def test_a_titan_model_haze_checks_its_structure_keys() -> None:
    """A structure key under a named haze model is validated as under an explicit one."""
    with pytest.raises(SimSceneValidationError, match=r'atmosphere\.ns_falloff_ratio'):
        validate_sim_params(_titan_model_scene(ns_falloff_ratio=-1.0))


def test_an_unknown_haze_model_is_refused() -> None:
    """Only ``titan`` names a haze model."""
    scene = _titan_model_scene()
    scene['bodies'][0]['atmosphere']['haze_model'] = 'venus'
    with pytest.raises(SimSceneValidationError, match=r"atmosphere\.haze_model must be 'titan'"):
        validate_sim_params(scene)


def test_a_titan_haze_in_a_methane_band_filter_is_refused() -> None:
    """A methane-band filter is refused for Titan's haze, naming the filter."""
    with pytest.raises(SimSceneValidationError, match=r"methane-band.*'CL1\+MT3'"):
        validate_sim_params(_with_filter(_titan_model_scene(), 'CL1+MT3'))


def test_a_titan_haze_in_a_methane_window_filter_is_refused() -> None:
    """A methane-window filter is refused for Titan's haze, naming the filter."""
    with pytest.raises(SimSceneValidationError, match=r"methane-window.*'CL1\+CB3'"):
        validate_sim_params(_with_filter(_titan_model_scene(), 'CL1+CB3'))


def test_a_titan_haze_in_a_filter_overlapping_a_methane_band_is_refused() -> None:
    """A filter that puts a quarter of its light in a methane band core is refused."""
    with pytest.raises(SimSceneValidationError, match=r"24% of the light of 'CL1\+IR3' falls"):
        validate_sim_params(_with_filter(_titan_model_scene(), 'CL1+IR3'))


def test_a_titan_haze_in_an_ultraviolet_filter_is_refused() -> None:
    """A filter whose light lies outside the modeled span is refused, giving the share."""
    with pytest.raises(SimSceneValidationError, match=r"100% of the light of 'CL1\+UV3'"):
        validate_sim_params(_with_filter(_titan_model_scene(), 'CL1+UV3'))


def test_a_titan_haze_with_just_under_a_quarter_of_its_light_outside_validates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A passband with 0.24 of its light outside the modeled span validates."""
    monkeypatch.setattr(rules_body, 'uncovered_fraction', lambda _passband: 0.24)
    scene = validate_sim_params(_titan_model_scene())
    assert scene['bodies'][0]['atmosphere'] == {'haze_model': 'titan'}


def test_a_titan_haze_with_just_over_a_quarter_of_its_light_outside_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A passband with 0.26 of its light outside the modeled span is refused."""
    monkeypatch.setattr(rules_body, 'uncovered_fraction', lambda _passband: 0.26)
    with pytest.raises(SimSceneValidationError, match=r'26% of the light'):
        validate_sim_params(_titan_model_scene())


def test_a_titan_haze_on_a_camera_without_passbands_is_refused() -> None:
    """Titan's haze depends on wavelength, so a camera without passbands is refused."""
    with pytest.raises(SimSceneValidationError, match="'vgiss_1_nac' has no passband table"):
        validate_sim_params(_titan_model_scene(instrument='vgiss_1_nac'))
