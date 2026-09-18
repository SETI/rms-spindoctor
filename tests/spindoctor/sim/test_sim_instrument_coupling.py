"""Per-instrument coupling for the simulator (B2).

A sim scene may name an instrument so the rendered frame and its obs go through
the same per-instrument units, noise, and saturation settings the navigator
applies to a real frame.  These tests cover the config resolver, the obs-level
``InstrumentSettings``, and the DN-vs-I/F rendering split.
"""

from pathlib import Path
from typing import Any

import pytest

from spindoctor.config import DEFAULT_CONFIG
from spindoctor.nav_orchestrator.instrument_config import instrument_settings_from_obs
from spindoctor.obs.obs_inst_sim import ObsSim
from spindoctor.sim.instruments import resolve_sim_inst_config
from spindoctor.sim.render import render_combined_model


def _obs(instrument: str | None, *, size: int = 64) -> ObsSim:
    """Build an ObsSim for the given instrument from a minimal scene."""
    params: dict[str, Any] = {'size_v': size, 'size_u': size, 'random_seed': 1}
    if instrument is not None:
        params['instrument'] = instrument
    return ObsSim.from_file('/tmp/sim_test.json', sim_params=params)


def test_resolver_maps_coiss_nac_to_raw_dn() -> None:
    """The coiss_nac name resolves to the Cassini NAC raw block."""
    cfg = resolve_sim_inst_config(DEFAULT_CONFIG, 'coiss_nac')
    assert cfg['data_units'] == 'raw_dn'


def test_resolver_maps_coiss_nac_saturation() -> None:
    """The resolved coiss_nac block carries the Cassini full-well DN."""
    cfg = resolve_sim_inst_config(DEFAULT_CONFIG, 'coiss_nac')
    assert float(cfg['noise']['saturation_dn']) == 4095.0


def test_resolver_maps_vgiss_to_calibrated_if() -> None:
    """The vgiss name resolves to a calibrated-IF block."""
    cfg = resolve_sim_inst_config(DEFAULT_CONFIG, 'vgiss')
    assert cfg['data_units'] == 'calibrated_if'


def test_resolver_generic_returns_sim_block() -> None:
    """A None instrument resolves to the standalone sim block."""
    cfg = resolve_sim_inst_config(DEFAULT_CONFIG, None)
    assert cfg['data_units'] == 'raw_dn'
    assert 'signal_full_scale_frac' in cfg['noise']


def test_resolver_rejects_unknown_instrument() -> None:
    """An unrecognized instrument name raises with a clear message."""
    with pytest.raises(ValueError, match='unknown sim instrument'):
        resolve_sim_inst_config(DEFAULT_CONFIG, 'hubble_wfc3')


def test_settings_coiss_nac_saturation() -> None:
    """A coiss_nac obs reports the Cassini saturation DN to the orchestrator."""
    settings = instrument_settings_from_obs(_obs('coiss_nac'))
    assert settings.saturation_dn == 4095.0


def test_settings_gossi_does_not_fit_rotation() -> None:
    """A gossi obs fits no camera rotation, as every instrument now does."""
    settings = instrument_settings_from_obs(_obs('gossi'))
    assert settings.fit_camera_rotation is False


def test_settings_vgiss_is_calibrated_if() -> None:
    """A vgiss obs reports calibrated-IF units with no saturation gate."""
    settings = instrument_settings_from_obs(_obs('vgiss'))
    assert settings.data_units == 'calibrated_if'
    assert settings.saturation_dn is None


def test_settings_generic_defaults_to_raw_dn() -> None:
    """An instrument-less obs falls back to the generic raw_dn sim block."""
    settings = instrument_settings_from_obs(_obs(None))
    assert settings.data_units == 'raw_dn'


def _lit_body_scene(instrument: str, *, size: int = 64) -> dict[str, Any]:
    """A scene with a single lit body for the given instrument."""
    return {
        'size_v': size,
        'size_u': size,
        'random_seed': 1,
        'instrument': instrument,
        'bodies': [
            {
                'name': 'B',
                'center_v': size / 2,
                'center_u': size / 2,
                'axis1': size * 0.6,
                'axis2': size * 0.5,
                'axis3': size * 0.5,
                'illumination_angle': 20.0,
                'phase_angle': 30.0,
            }
        ],
    }


def test_raw_dn_render_is_in_dn() -> None:
    """A raw_dn instrument render reaches DN well above the [0, 1] signal range."""
    img, _ = render_combined_model(_lit_body_scene('coiss_nac'))
    assert float(img.max()) > 100.0


def test_gossi_scales_to_8bit_well() -> None:
    """The gossi render is bounded by its 8-bit ADC ceiling, unlike 12-bit coiss."""
    gossi_img, _ = render_combined_model(_lit_body_scene('gossi'))
    coiss_img, _ = render_combined_model(_lit_body_scene('coiss_nac'))
    # Galileo's 8-bit ADC clips at 255 DN (the CCD full well sits above it in DN),
    # while the Cassini 12-bit chain reaches well past 255.
    assert float(gossi_img.max()) <= 255.0
    assert float(coiss_img.max()) > 255.0


def test_calibrated_if_render_stays_in_if_range() -> None:
    """A calibrated_if instrument render leaves the signal in I/F [0, 1]."""
    img, _ = render_combined_model(_lit_body_scene('vgiss'))
    assert float(img.max()) <= 1.0


def test_overrides_pin_an_individual_key() -> None:
    """An override pins one key while the rest still come from the instrument."""
    base = resolve_sim_inst_config(DEFAULT_CONFIG, 'coiss_nac')
    cfg = resolve_sim_inst_config(DEFAULT_CONFIG, 'coiss_nac', {'star_psf_sigma': 0.9})
    assert cfg['star_psf_sigma'] == 0.9
    assert cfg['data_units'] == base['data_units']


def test_overrides_deep_merge_nested_block() -> None:
    """A nested override changes one sub-key and preserves its siblings."""
    base = resolve_sim_inst_config(DEFAULT_CONFIG, 'coiss_nac')
    cfg = resolve_sim_inst_config(DEFAULT_CONFIG, 'coiss_nac', {'noise': {'read_noise_dn': 7.0}})
    assert cfg['noise']['read_noise_dn'] == 7.0
    assert cfg['noise']['saturation_dn'] == base['noise']['saturation_dn']


def test_overrides_do_not_mutate_the_live_block() -> None:
    """Resolving with overrides leaves the underlying config block untouched."""
    before = float(resolve_sim_inst_config(DEFAULT_CONFIG, 'coiss_nac')['star_psf_sigma'])
    resolve_sim_inst_config(DEFAULT_CONFIG, 'coiss_nac', {'star_psf_sigma': 9.0})
    after = float(resolve_sim_inst_config(DEFAULT_CONFIG, 'coiss_nac')['star_psf_sigma'])
    assert before == after


def test_generic_plus_overrides_self_specifies() -> None:
    """The generic block plus overrides expresses a fully self-specified config."""
    cfg = resolve_sim_inst_config(
        DEFAULT_CONFIG,
        'generic',
        {'star_psf_sigma': 1.5, 'data_units': 'raw_dn', 'noise': {'read_noise_dn': 3.0}},
    )
    assert cfg['star_psf_sigma'] == 1.5
    assert cfg['data_units'] == 'raw_dn'
    assert cfg['noise']['read_noise_dn'] == 3.0


def test_no_overrides_returns_the_live_block() -> None:
    """Without overrides the resolver returns the instrument block unchanged."""
    cfg = resolve_sim_inst_config(DEFAULT_CONFIG, 'coiss_nac', None)
    assert cfg is resolve_sim_inst_config(DEFAULT_CONFIG, 'coiss_nac')


def test_overrides_reach_the_obs_inst_config() -> None:
    """A scene's instrument_config override is visible on the obs inst config."""
    params: dict[str, Any] = {
        'size_v': 64,
        'size_u': 64,
        'random_seed': 1,
        'instrument': 'coiss_nac',
        'instrument_config': {'noise': {'read_noise_dn': 12.0}},
    }
    obs = ObsSim.from_file('/tmp/sim_test.json', sim_params=params)
    assert obs._inst_config is not None
    assert float(obs._inst_config['noise']['read_noise_dn']) == 12.0


def test_instrument_config_round_trips_through_the_scene_schema(tmp_path: Path) -> None:
    """instrument_config survives a scene save/load round-trip (GUI save path)."""
    from spindoctor.sim.scene import load_sim_scene, save_sim_scene

    sim_params: dict[str, Any] = {
        'size_v': 64,
        'size_u': 64,
        'random_seed': 1,
        'instrument': 'generic',
        'instrument_config': {'star_psf_sigma': 1.5, 'noise': {'read_noise_dn': 3.0}},
    }
    path = tmp_path / 'example.yaml'
    save_sim_scene(sim_params, path)
    scene = load_sim_scene(path)
    assert scene['instrument_config'] == {'star_psf_sigma': 1.5, 'noise': {'read_noise_dn': 3.0}}


def test_extfov_margin_unmatched_size_names_the_available_keys() -> None:
    """An uncovered image size fails with the size and the fallback keys."""
    from spindoctor.sim.instruments import resolve_extfov_margin

    with pytest.raises(ValueError, match=r'image size 777.*fallback sizes.*\[64, 512\]'):
        resolve_extfov_margin(
            {'extfov_margin_vu': {512: [10, 10]}},
            {'extfov_margin_vu': {64: [4, 4], 512: [10, 10]}},
            777,
        )


def test_extfov_margin_fallback_covers_uncatalogued_size() -> None:
    """A size the instrument table lacks resolves through the generic table."""
    from spindoctor.sim.instruments import resolve_extfov_margin

    margin = resolve_extfov_margin(
        {'extfov_margin_vu': {512: [10, 10]}},
        {'extfov_margin_vu': {64: [4, 4], 512: [10, 10]}},
        64,
    )
    assert margin == [4, 4]
