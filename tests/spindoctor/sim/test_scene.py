"""Flat-schema validation, boundary classification, and save/load round-trip."""

from pathlib import Path
from typing import Any

import pytest

from spindoctor.sim.scene import (
    _ALLOWED_KEYS,
    _OBJECT_BLOCKS,
    TOP_LEVEL_IDEALIZED_KEYS,
    TOP_LEVEL_TEST_ONLY_KEYS,
    TOP_LEVEL_TRUTH_KEYS,
    TRUTH_KEYS,
    SimSceneValidationError,
    load_sim_scene,
    save_sim_scene,
    validate_sim_params,
)


def _sim_params() -> dict[str, Any]:
    return {
        'size_v': 128,
        'size_u': 128,
        'random_seed': 7,
        'instrument': 'coiss_nac',
        'offset_v': 3.0,
        'offset_u': -2.0,
        'bodies': [{'name': 'RHEA', 'center_v': 64.0, 'center_u': 64.0, 'axis1': 80.0}],
        'noise': {'poisson': True, 'read_noise_dn': 4.0},
        'sky_counts': {'density_factor': 12.0},
    }


def test_save_injects_schema_version_and_scene_name(tmp_path: Path) -> None:
    """A saved scene carries the schema version and the filename stem as its name."""
    path = tmp_path / 'roundtrip.yaml'
    save_sim_scene(_sim_params(), path)
    scene = load_sim_scene(path)
    assert scene['schema_version'] == 2
    assert scene['scene_name'] == 'roundtrip'


def test_loaded_scene_is_the_flat_sim_params(tmp_path: Path) -> None:
    """A loaded scene is the flat sim_params dict the renderer consumes."""
    path = tmp_path / 'rt2.yaml'
    save_sim_scene(_sim_params(), path)
    scene = load_sim_scene(path)
    assert scene['instrument'] == 'coiss_nac'
    assert scene['size_v'] == 128
    assert scene['size_u'] == 128
    assert scene['offset_v'] == 3.0
    assert scene['offset_u'] == -2.0
    assert scene['bodies'][0]['name'] == 'RHEA'
    assert scene['sky_counts']['density_factor'] == 12.0


def test_save_then_load_preserves_values(tmp_path: Path) -> None:
    """Saving then loading reproduces every flat field verbatim."""
    path = tmp_path / 'named_scene.yaml'
    params = _sim_params()
    save_sim_scene(params, path)
    scene = load_sim_scene(path)
    for key, value in params.items():
        assert scene[key] == value


def test_load_rejects_bad_instrument(tmp_path: Path) -> None:
    """An unknown instrument fails validation."""
    path = tmp_path / 'bad.yaml'
    path.write_text(
        'schema_version: 2\nscene_name: bad\ninstrument: hubble\n'
        'size_v: 64\nsize_u: 64\nrandom_seed: 1\n'
    )
    with pytest.raises(SimSceneValidationError, match='instrument'):
        load_sim_scene(path)


def test_load_rejects_unknown_key(tmp_path: Path) -> None:
    """An unmodeled top-level key fails validation."""
    path = tmp_path / 'typo.yaml'
    path.write_text(
        'schema_version: 2\nscene_name: typo\ninstrument: coiss_nac\n'
        'size_v: 64\nsize_u: 64\nrandom_seed: 1\nwobble: 5\n'
    )
    with pytest.raises(SimSceneValidationError, match='unknown scene keys'):
        load_sim_scene(path)


def test_load_rejects_nonpositive_size(tmp_path: Path) -> None:
    """A non-positive image size fails validation."""
    path = tmp_path / 'small.yaml'
    path.write_text(
        'schema_version: 2\nscene_name: small\ninstrument: coiss_nac\n'
        'size_v: 0\nsize_u: 64\nrandom_seed: 1\n'
    )
    with pytest.raises(SimSceneValidationError, match='size_v'):
        load_sim_scene(path)


def test_load_rejects_inf_positive_number(tmp_path: Path) -> None:
    """YAML .inf on a positive-number key (exposure_sec) fails validation."""
    path = tmp_path / 'infexp.yaml'
    path.write_text(
        'schema_version: 2\nscene_name: infexp\ninstrument: coiss_nac\n'
        'size_v: 64\nsize_u: 64\nrandom_seed: 1\nexposure_sec: .inf\n'
    )
    with pytest.raises(SimSceneValidationError, match='exposure_sec must be finite'):
        load_sim_scene(path)


def test_load_rejects_inf_nonnegative_number(tmp_path: Path) -> None:
    """YAML .inf on a non-negative-number key (noise.read_noise_dn) fails."""
    path = tmp_path / 'infnoise.yaml'
    path.write_text(
        'schema_version: 2\nscene_name: infnoise\ninstrument: coiss_nac\n'
        'size_v: 64\nsize_u: 64\nrandom_seed: 1\n'
        'noise:\n  read_noise_dn: .inf\n'
    )
    with pytest.raises(SimSceneValidationError, match='read_noise_dn must be finite'):
        load_sim_scene(path)


def test_validate_sim_params_rejects_nan_positive_number() -> None:
    """NaN on a positive-number key fails validation."""
    params = _sim_params()
    params['exposure_sec'] = float('nan')
    with pytest.raises(SimSceneValidationError, match='exposure_sec must be finite'):
        validate_sim_params(params)


def test_load_rejects_schema_version_1(tmp_path: Path) -> None:
    """The loader accepts only the current schema version."""
    path = tmp_path / 'v1.yaml'
    path.write_text(
        'schema_version: 1\nscene_name: v1\ninstrument: coiss_nac\n'
        'size_v: 64\nsize_u: 64\nrandom_seed: 1\n'
    )
    with pytest.raises(SimSceneValidationError, match='schema_version must be 2'):
        load_sim_scene(path)


def test_validate_sim_params_accepts_dict_author_scene() -> None:
    """A programmatic scene without schema_version/scene_name validates."""
    params = _sim_params()
    assert validate_sim_params(params) is params


def test_validate_sim_params_rejects_unknown_body_key() -> None:
    """An unmodeled per-body key fails validation."""
    params = _sim_params()
    params['bodies'][0]['albedo_wobble'] = 0.5
    with pytest.raises(SimSceneValidationError, match=r'bodies\[0\].*albedo_wobble'):
        validate_sim_params(params)


def test_validate_sim_params_accepts_body_relief_and_photometry_keys() -> None:
    """The topographic truth keys validate with well-typed values."""
    params = _sim_params()
    params['bodies'][0].update(
        {
            'limb_relief_rms': 0.02,
            'limb_relief_corr_deg': 12.0,
            'photometric_law': 'minnaert',
            'minnaert_k': 0.6,
            'opposition_surge': {'amplitude': 0.5, 'width_deg': 5.0},
        }
    )
    assert validate_sim_params(params) is params


def test_validate_sim_params_rejects_negative_limb_relief_rms() -> None:
    """A negative relief RMS fails validation."""
    params = _sim_params()
    params['bodies'][0]['limb_relief_rms'] = -0.01
    with pytest.raises(SimSceneValidationError, match=r'limb_relief_rms'):
        validate_sim_params(params)


def test_validate_sim_params_rejects_nonpositive_relief_corr() -> None:
    """A non-positive relief correlation length fails validation."""
    params = _sim_params()
    params['bodies'][0]['limb_relief_corr_deg'] = 0.0
    with pytest.raises(SimSceneValidationError, match=r'limb_relief_corr_deg'):
        validate_sim_params(params)


def test_validate_sim_params_rejects_unknown_photometric_law() -> None:
    """A law outside the renderer's vocabulary fails with the choices listed."""
    params = _sim_params()
    params['bodies'][0]['photometric_law'] = 'hapke'
    with pytest.raises(SimSceneValidationError, match=r'photometric_law.*lommel_seeliger'):
        validate_sim_params(params)


def test_validate_sim_params_rejects_unknown_surge_key() -> None:
    """An unmodeled opposition_surge key fails validation."""
    params = _sim_params()
    params['bodies'][0]['opposition_surge'] = {'amplitude': 0.5, 'sharpness': 2.0}
    with pytest.raises(SimSceneValidationError, match=r'opposition_surge.*sharpness'):
        validate_sim_params(params)


def test_validate_sim_params_rejects_nonpositive_surge_width() -> None:
    """A non-positive surge width fails validation."""
    params = _sim_params()
    params['bodies'][0]['opposition_surge'] = {'amplitude': 0.5, 'width_deg': 0.0}
    with pytest.raises(SimSceneValidationError, match=r'width_deg'):
        validate_sim_params(params)


def test_validate_sim_params_rejects_unknown_star_key() -> None:
    """An unmodeled per-star key fails validation."""
    params = _sim_params()
    params['stars'] = [{'name': 'S', 'v': 10.0, 'u': 10.0, 'vmag': 5.0, 'color': 'red'}]
    with pytest.raises(SimSceneValidationError, match=r'stars\[0\].*color'):
        validate_sim_params(params)


def test_validate_sim_params_accepts_star_information_asymmetry_keys() -> None:
    """The navigable flag, planted catalog error, companion, and delta_mag validate."""
    params = _sim_params()
    params['star_catalog_scatter_px'] = 2.0
    params['stars'] = [
        {
            'name': 'S',
            'v': 40.0,
            'u': 40.0,
            'vmag': 4.0,
            'navigable': True,
            'catalog_error_v': 1.0,
            'catalog_error_u': -0.5,
            'companion': {'sep_px': 2.0, 'delta_mag': 1.5, 'angle_deg': 30.0},
            'delta_mag': 0.4,
        },
        {'name': 'CONF', 'v': 60.0, 'u': 60.0, 'vmag': 4.1, 'navigable': False},
    ]
    assert validate_sim_params(params) is params


def test_validate_sim_params_rejects_unknown_companion_key() -> None:
    """An unmodeled companion key fails validation."""
    params = _sim_params()
    params['stars'] = [
        {'name': 'S', 'v': 40.0, 'u': 40.0, 'vmag': 4.0, 'companion': {'sep_px': 2.0, 'pa': 3.0}}
    ]
    with pytest.raises(SimSceneValidationError, match=r'companion.*pa'):
        validate_sim_params(params)


def test_validate_sim_params_rejects_non_bool_navigable() -> None:
    """A non-boolean navigable flag fails validation."""
    params = _sim_params()
    params['stars'] = [{'name': 'S', 'v': 40.0, 'u': 40.0, 'vmag': 4.0, 'navigable': 'yes'}]
    with pytest.raises(SimSceneValidationError, match=r'navigable'):
        validate_sim_params(params)


def test_validate_sim_params_rejects_negative_catalog_scatter() -> None:
    """A negative scene-level catalog scatter sigma fails validation."""
    params = _sim_params()
    params['star_catalog_scatter_px'] = -1.0
    with pytest.raises(SimSceneValidationError, match=r'star_catalog_scatter_px'):
        validate_sim_params(params)


def test_validate_sim_params_rejects_legacy_rings_key() -> None:
    """The retired painted-annulus 'rings' list is an unknown key now."""
    params = _sim_params()
    params['rings'] = [{'name': 'R'}]
    with pytest.raises(SimSceneValidationError, match=r'unknown scene keys.*rings'):
        validate_sim_params(params)


def test_validate_sim_params_rejects_legacy_shade_solid_rings_key() -> None:
    """The retired 'shade_solid_rings' knob is an unknown key now."""
    params = _sim_params()
    params['shade_solid_rings'] = True
    with pytest.raises(SimSceneValidationError, match=r'unknown scene keys.*shade_solid_rings'):
        validate_sim_params(params)


def test_validate_sim_params_rejects_unknown_noise_key() -> None:
    """An unmodeled noise key fails validation instead of silently doing nothing."""
    params = _sim_params()
    params['noise'] = {'poisson': True, 'shot_noise': True}
    with pytest.raises(SimSceneValidationError, match=r'noise.*shot_noise'):
        validate_sim_params(params)


def test_validate_sim_params_accepts_the_full_noise_inventory() -> None:
    """Every accepted noise key (including the vidicon sub-map) validates."""
    params = _sim_params()
    params['noise'] = {
        'poisson': True,
        'read_noise_dn': 4.0,
        'bias_dn': 20.0,
        'cosmic_ray_rate_per_sec': 0.001,
        'missing_data_rate': 0.01,
        'signal_full_scale_frac': 0.5,
        'pixel_area_cm2': 1.0,
        'dark_current_e_per_sec': 5.0,
        'hot_pixel_fraction': 0.002,
        'hot_pixel_amplitude_e': 4.0e4,
        'hot_pixel_column_factor': 0.3,
        'banding_amplitude_e': 30.0,
        'banding_period_px': 64.0,
        'bias_pedestal_sigma_dn': 2.0,
        'bias_row_gradient_dn': 1.0,
        'bias_col_gradient_dn': 0.5,
        'bloom_length': 4,
        'vidicon': {
            'read_noise_line_dn': 1.8,
            'read_noise_pixel_dn': 1.8,
            'coherent_amplitude_dn': 0.25,
            'coherent_period_px': 8.0,
        },
    }
    assert validate_sim_params(params) is params


def test_validate_sim_params_rejects_mistyped_noise_value() -> None:
    """A non-boolean poisson value fails with the offending key named."""
    params = _sim_params()
    params['noise'] = {'poisson': 'yes'}
    with pytest.raises(SimSceneValidationError, match=r'noise\.poisson'):
        validate_sim_params(params)


def test_validate_sim_params_rejects_unknown_vidicon_key() -> None:
    """An unmodeled vidicon sub-key fails validation."""
    params = _sim_params()
    params['noise'] = {'vidicon': {'sweep_rate_dn': 1.0}}
    with pytest.raises(SimSceneValidationError, match=r'vidicon.*sweep_rate_dn'):
        validate_sim_params(params)


def test_validate_sim_params_rejects_uncatalogued_wac_gain_state() -> None:
    """A WAC gain state outside the catalog fails at validation time."""
    params = _sim_params()
    params['instrument'] = 'coiss_wac'
    params['detector'] = {'gain_state': 3}
    with pytest.raises(
        SimSceneValidationError, match=r'gain_state 3 is not cataloged.*coiss_wac.*\[2\]'
    ):
        validate_sim_params(params)


def test_validate_sim_params_accepts_cataloged_gain_state() -> None:
    """The cataloged WAC state 2 validates."""
    params = _sim_params()
    params['instrument'] = 'coiss_wac'
    params['detector'] = {'gain_state': 2}
    assert validate_sim_params(params) is params


def test_validate_sim_params_rejects_v1_body_range_key() -> None:
    """The v1 per-body 'range' key is gone; 'range_km' replaced it."""
    params = _sim_params()
    params['bodies'][0]['range'] = 500000.0
    with pytest.raises(SimSceneValidationError, match=r'bodies\[0\].*range'):
        validate_sim_params(params)


def test_validate_sim_params_accepts_body_range_km() -> None:
    """The per-body 'range_km' key validates."""
    params = _sim_params()
    params['bodies'][0]['range_km'] = 500000.0
    assert validate_sim_params(params) is params


def test_validate_sim_params_rejects_truth_key_in_nav_override() -> None:
    """nav_override may only carry idealized body keys (believed geometry)."""
    params = _sim_params()
    params['bodies'][0]['nav_override'] = {'crater_fill': 0.0}
    with pytest.raises(SimSceneValidationError, match='nav_override may only override'):
        validate_sim_params(params)


def test_validate_sim_params_accepts_idealized_nav_override() -> None:
    """nav_override with idealized keys (the shape-mismatch fixture) validates."""
    params = _sim_params()
    params['bodies'][0]['nav_override'] = {'mesh_lumpiness': 0.0}
    assert validate_sim_params(params) is params


def test_validate_sim_params_rejects_empty_psf_block() -> None:
    """A present optics.psf block without a core width fails at validation."""
    params = _sim_params()
    params['optics'] = {'psf': {}}
    with pytest.raises(SimSceneValidationError, match=r'optics\.psf needs sigma_v'):
        validate_sim_params(params)


def test_validate_sim_params_rejects_psf_without_sigma_v() -> None:
    """Wing-only PSF parameters without sigma_v fail at validation."""
    params = _sim_params()
    params['optics'] = {'psf': {'w': 0.1, 'r0': 2.0, 'n': 3.0}}
    with pytest.raises(SimSceneValidationError, match=r'optics\.psf needs sigma_v'):
        validate_sim_params(params)


def test_validate_sim_params_rejects_false_match_navigator() -> None:
    """match_navigator: false authors a kernel-less block and fails."""
    params = _sim_params()
    params['optics'] = {'psf': {'match_navigator': False}}
    with pytest.raises(SimSceneValidationError, match='match_navigator must be true'):
        validate_sim_params(params)


def test_validate_sim_params_accepts_match_navigator_only_psf() -> None:
    """The navigator-matched form validates without explicit parameters."""
    params = _sim_params()
    params['optics'] = {'psf': {'match_navigator': True}}
    assert validate_sim_params(params) is params


def test_validate_sim_params_accepts_sigma_v_only_psf() -> None:
    """A psf block with just the core width validates."""
    params = _sim_params()
    params['optics'] = {'psf': {'sigma_v': 1.2}}
    assert validate_sim_params(params) is params


def test_validate_sim_params_rejects_zero_crater_max_radius() -> None:
    """A zero crater_max_radius fails at validation, not deep in the sampler."""
    params = _sim_params()
    params['bodies'][0]['crater_fill'] = 0.2
    params['bodies'][0]['crater_max_radius'] = 0
    with pytest.raises(SimSceneValidationError, match='crater_max_radius must be a positive'):
        validate_sim_params(params)


def test_validate_sim_params_rejects_zero_crater_min_radius() -> None:
    """A zero crater_min_radius fails at validation."""
    params = _sim_params()
    params['bodies'][0]['crater_min_radius'] = 0.0
    with pytest.raises(SimSceneValidationError, match='crater_min_radius must be a positive'):
        validate_sim_params(params)


def test_validate_sim_params_rejects_inverted_crater_radius_band() -> None:
    """crater_min_radius >= crater_max_radius fails."""
    params = _sim_params()
    params['bodies'][0]['crater_min_radius'] = 0.3
    params['bodies'][0]['crater_max_radius'] = 0.1
    with pytest.raises(SimSceneValidationError, match='must be less than crater_max_radius'):
        validate_sim_params(params)


def test_validate_sim_params_rejects_crater_max_below_default_min() -> None:
    """A lone crater_max_radius below the defaulted minimum fails."""
    params = _sim_params()
    params['bodies'][0]['crater_max_radius'] = 0.04
    with pytest.raises(SimSceneValidationError, match='must be less than crater_max_radius'):
        validate_sim_params(params)


def test_validate_sim_params_accepts_valid_crater_radius_band() -> None:
    """A positive, ordered crater radius band validates."""
    params = _sim_params()
    params['bodies'][0]['crater_min_radius'] = 0.02
    params['bodies'][0]['crater_max_radius'] = 0.2
    assert validate_sim_params(params) is params


def test_validate_sim_params_rejects_crater_power_law_exponent_at_one() -> None:
    """crater_power_law_exponent <= 1 fails with the normalizability message."""
    params = _sim_params()
    params['bodies'][0]['crater_power_law_exponent'] = 1.0
    with pytest.raises(SimSceneValidationError, match='must exceed 1'):
        validate_sim_params(params)


def test_validate_sim_params_accepts_crater_power_law_exponent_above_one() -> None:
    """A proper power-law exponent validates."""
    params = _sim_params()
    params['bodies'][0]['crater_power_law_exponent'] = 2.5
    assert validate_sim_params(params) is params


def _mesh_body() -> dict[str, Any]:
    """A minimal valid polyhedral-mesh body entry."""
    return {
        'name': 'LUMPY',
        'shape_model': 'polyhedral_mesh',
        'axis1': 30.0,
        'axis2': 26.0,
        'axis3': 24.0,
        'mesh_lumpiness': 0.4,
        'mesh_seed': 3,
    }


@pytest.mark.parametrize(
    ('key', 'value'),
    [
        ('atmosphere', {'scale_height_px': 4.0, 'tau_ref': 1.0}),
        ('photometric_law', 'minnaert'),
        ('minnaert_k', 0.6),
        ('opposition_surge', {'amplitude': 0.4, 'width_deg': 5.0}),
        ('albedo_texture', {'rms': 0.05, 'corr_px': 10.0}),
        ('disc_texture', {'band_amplitude': 0.1}),
        ('transits', [{'moon': {'radius_px': 3.0}}]),
        ('crater_fill', 0.5),
        ('crater_min_radius', 0.06),
        ('crater_max_radius', 0.2),
        ('crater_power_law_exponent', 2.5),
        ('crater_relief_scale', 0.5),
    ],
)
def test_validate_sim_params_rejects_ellipsoid_only_key_on_mesh(key: str, value: Any) -> None:
    """An ellipsoid-only appearance key on a polyhedral_mesh body fails."""
    params = _sim_params()
    body = _mesh_body()
    body[key] = value
    params['bodies'] = [body]
    with pytest.raises(SimSceneValidationError, match='not supported on'):
        validate_sim_params(params)


def test_validate_sim_params_accepts_mesh_appearance_keys() -> None:
    """The mesh-supported appearance keys validate on a mesh body."""
    params = _sim_params()
    body = _mesh_body()
    body.update(
        {
            'shading': 'gouraud',
            'limb_relief_rms': 0.02,
            'limb_relief_corr_deg': 12.0,
            'mesh_detail_octaves': 2,
            'pose_scatter': {'sigma_deg': 1.0},
            'anti_aliasing': 0.5,
        }
    )
    params['bodies'] = [body]
    assert validate_sim_params(params) is params


def test_validate_sim_params_accepts_ellipsoid_appearance_keys() -> None:
    """The same appearance keys stay valid on an ellipsoid body."""
    params = _sim_params()
    params['bodies'][0].update(
        {
            'photometric_law': 'minnaert',
            'minnaert_k': 0.6,
            'crater_fill': 0.5,
            'atmosphere': {'scale_height_px': 4.0, 'tau_ref': 1.0},
        }
    )
    assert validate_sim_params(params) is params


def test_validate_sim_params_rejects_duplicate_body_names() -> None:
    """Two bodies with the same name fail (name-keyed truth would collide)."""
    params = _sim_params()
    params['bodies'] = [{'name': 'RHEA', 'axis1': 40.0}, {'name': 'RHEA', 'axis1': 20.0}]
    with pytest.raises(SimSceneValidationError, match="share the effective name 'RHEA'"):
        validate_sim_params(params)


def test_validate_sim_params_rejects_case_colliding_body_names() -> None:
    """Body names collide case-insensitively (truth keys are upper-cased)."""
    params = _sim_params()
    params['bodies'] = [{'name': 'Rhea', 'axis1': 40.0}, {'name': 'RHEA', 'axis1': 20.0}]
    with pytest.raises(SimSceneValidationError, match='share the effective name'):
        validate_sim_params(params)


def test_validate_sim_params_rejects_body_name_colliding_with_default() -> None:
    """An explicit name colliding with an unnamed body's positional default fails."""
    params = _sim_params()
    params['bodies'] = [{'axis1': 40.0}, {'name': 'SIM-BODY-1', 'axis1': 20.0}]
    with pytest.raises(SimSceneValidationError, match="share the effective name 'SIM-BODY-1'"):
        validate_sim_params(params)


def test_validate_sim_params_accepts_distinct_body_names() -> None:
    """Distinctly named (and defaulted) bodies validate."""
    params = _sim_params()
    params['bodies'] = [{'name': 'RHEA', 'axis1': 40.0}, {'axis1': 20.0}]
    assert validate_sim_params(params) is params


def _ring_system_params() -> dict[str, Any]:
    """A scene with a minimal valid ring_system block."""
    params = _sim_params()
    del params['bodies']
    params['ring_system'] = {
        'geometry': {
            'center_v': 64.0,
            'center_u': 64.0,
            'opening_deg_obs': 30.0,
            'opening_deg_sun': 20.0,
            'node_deg': 0.0,
        },
        'features': [
            {
                'name': 'F1',
                'kind': 'ringlet',
                'tau': 1.0,
                'width': 10.0,
                'orbit': {'a': 40.0, 'ae': 0.0, 'long_peri': 0.0, 'rate_peri': 0.0},
                'albedo': 0.5,
                'phase_g': -0.3,
            }
        ],
    }
    return params


def test_validate_sim_params_accepts_ring_system() -> None:
    """The minimal ring_system block validates."""
    params = _ring_system_params()
    assert validate_sim_params(params) is params


def test_ring_system_round_trips_through_yaml(tmp_path: Path) -> None:
    """A ring_system scene saves and reloads verbatim."""
    path = tmp_path / 'rs.yaml'
    params = _ring_system_params()
    save_sim_scene(params, path)
    scene = load_sim_scene(path)
    assert scene['ring_system'] == params['ring_system']


def test_validate_sim_params_rejects_unknown_ring_system_key() -> None:
    """An unmodeled ring_system key fails validation."""
    params = _ring_system_params()
    params['ring_system']['spokes'] = {}
    with pytest.raises(SimSceneValidationError, match=r'ring_system.*spokes'):
        validate_sim_params(params)


def test_validate_sim_params_rejects_duplicate_ring_feature_names() -> None:
    """Two ring features with the same name fail (name-keyed consumers collide)."""
    params = _ring_system_params()
    second = dict(params['ring_system']['features'][0])
    params['ring_system']['features'].append(second)
    with pytest.raises(SimSceneValidationError, match="share the effective name 'F1'"):
        validate_sim_params(params)


def test_validate_sim_params_rejects_ring_feature_name_colliding_with_default() -> None:
    """An explicit feature name colliding with a positional default fails."""
    params = _ring_system_params()
    first = dict(params['ring_system']['features'][0])
    del first['name']
    second = dict(params['ring_system']['features'][0])
    second['name'] = 'RING-FEATURE-1'
    params['ring_system']['features'] = [first, second]
    with pytest.raises(SimSceneValidationError, match="share the effective name 'RING-FEATURE-1'"):
        validate_sim_params(params)


def test_validate_sim_params_accepts_distinct_ring_feature_names() -> None:
    """Distinctly named ring features validate."""
    params = _ring_system_params()
    second = dict(params['ring_system']['features'][0])
    second['name'] = 'F2'
    params['ring_system']['features'].append(second)
    assert validate_sim_params(params) is params


def test_validate_sim_params_rejects_ring_orbit_ae_at_a() -> None:
    """orbit.ae >= orbit.a fails at validation, not deep in the edge math."""
    params = _ring_system_params()
    params['ring_system']['features'][0]['orbit']['ae'] = 40.0
    with pytest.raises(SimSceneValidationError, match=r'ae must be less than orbit\.a'):
        validate_sim_params(params)


def test_validate_sim_params_accepts_subcritical_ring_orbit_ae() -> None:
    """orbit.ae below orbit.a validates."""
    params = _ring_system_params()
    params['ring_system']['features'][0]['orbit']['ae'] = 5.0
    assert validate_sim_params(params) is params


def test_validate_sim_params_rejects_orbit_error_negating_a() -> None:
    """A delta_a_px that drives the effective semimajor axis <= 0 fails."""
    params = _ring_system_params()
    params['ring_system']['features'][0]['orbit_error'] = {'delta_a_px': -40.0}
    with pytest.raises(SimSceneValidationError, match='positive semimajor axis'):
        validate_sim_params(params)


def test_validate_sim_params_rejects_orbit_error_driving_ae_past_a() -> None:
    """A delta_ae_px that pushes the effective eccentricity to 1 fails."""
    params = _ring_system_params()
    params['ring_system']['features'][0]['orbit']['ae'] = 5.0
    params['ring_system']['features'][0]['orbit_error'] = {'delta_ae_px': 35.0}
    with pytest.raises(SimSceneValidationError, match='effective eccentric'):
        validate_sim_params(params)


def test_validate_sim_params_accepts_bounded_orbit_error() -> None:
    """Orbit-error deltas that keep the effective orbit physical validate."""
    params = _ring_system_params()
    params['ring_system']['features'][0]['orbit']['ae'] = 5.0
    params['ring_system']['features'][0]['orbit_error'] = {
        'delta_a_px': -2.0,
        'delta_ae_px': 1.0,
        'delta_long_peri_deg': 15.0,
    }
    assert validate_sim_params(params) is params


def test_validate_sim_params_requires_ring_system_geometry() -> None:
    """A ring_system without its shared geometry block fails."""
    params = _ring_system_params()
    del params['ring_system']['geometry']
    with pytest.raises(SimSceneValidationError, match='geometry is required'):
        validate_sim_params(params)


def test_validate_sim_params_rejects_unknown_ring_system_geometry_key() -> None:
    """An unmodeled geometry key fails validation."""
    params = _ring_system_params()
    params['ring_system']['geometry']['tilt_deg'] = 5.0
    with pytest.raises(SimSceneValidationError, match=r'geometry.*tilt_deg'):
        validate_sim_params(params)


def test_validate_sim_params_requires_both_opening_angles() -> None:
    """Both opening angles are required (no silent face-on default)."""
    params = _ring_system_params()
    del params['ring_system']['geometry']['opening_deg_sun']
    with pytest.raises(SimSceneValidationError, match='opening_deg_sun is required'):
        validate_sim_params(params)


def test_validate_sim_params_rejects_out_of_range_opening() -> None:
    """Opening angles live in (-90, 90]."""
    params = _ring_system_params()
    params['ring_system']['geometry']['opening_deg_obs'] = -90.0
    with pytest.raises(SimSceneValidationError, match=r'opening_deg_obs must lie in \(-90, 90\]'):
        validate_sim_params(params)


def test_validate_sim_params_rejects_unknown_ring_feature_key() -> None:
    """An unmodeled feature key fails validation."""
    params = _ring_system_params()
    params['ring_system']['features'][0]['contrast'] = 5.0
    with pytest.raises(SimSceneValidationError, match=r'features\[0\].*contrast'):
        validate_sim_params(params)


def test_validate_sim_params_rejects_unknown_ring_feature_kind() -> None:
    """A kind outside the vocabulary fails with the choices listed."""
    params = _ring_system_params()
    params['ring_system']['features'][0]['kind'] = 'sheet'
    with pytest.raises(SimSceneValidationError, match=r'kind must be one of.*sheet'):
        validate_sim_params(params)


@pytest.mark.parametrize('key', ['tau', 'width'])
def test_validate_sim_params_requires_ring_feature_scalars(key: str) -> None:
    """tau (every kind) and width (the banded kinds) are required."""
    params = _ring_system_params()
    del params['ring_system']['features'][0][key]
    with pytest.raises(SimSceneValidationError, match=f'{key} is required'):
        validate_sim_params(params)


def _edge_feature() -> dict[str, Any]:
    """A minimal one-sided edge feature."""
    return {'name': 'E1', 'kind': 'edge', 'tau': 1.0, 'orbit': {'a': 40.0}}


def _wave_feature() -> dict[str, Any]:
    """A minimal density-wave-train feature."""
    return {
        'name': 'W1',
        'kind': 'wave',
        'tau': 0.4,
        'wavelength': 6.0,
        'damping': 12.0,
        'orbit': {'a': 40.0},
    }


def test_validate_sim_params_accepts_edge_ramp_wave_kinds() -> None:
    """The one-sided, ramp, and wave kinds validate with their shape keys."""
    params = _ring_system_params()
    params['ring_system']['features'] = [
        dict(_edge_feature(), side='out'),
        {
            'name': 'R1',
            'kind': 'ramp',
            'tau': 0.8,
            'width': 12.0,
            'side': 'in',
            'orbit': {'a': 60.0},
        },
        _wave_feature(),
    ]
    assert validate_sim_params(params) is params


def test_validate_sim_params_rejects_width_on_an_edge() -> None:
    """A one-sided edge has no radial width; a stray width fails loudly."""
    params = _ring_system_params()
    params['ring_system']['features'] = [dict(_edge_feature(), width=5.0)]
    with pytest.raises(SimSceneValidationError, match=r"width is not allowed for kind 'edge'"):
        validate_sim_params(params)


def test_validate_sim_params_rejects_side_on_a_ringlet() -> None:
    """side belongs to the one-sided kinds only."""
    params = _ring_system_params()
    params['ring_system']['features'][0]['side'] = 'in'
    with pytest.raises(SimSceneValidationError, match=r"side is not allowed for kind 'ringlet'"):
        validate_sim_params(params)


def test_validate_sim_params_rejects_bad_side_vocabulary() -> None:
    """side must be 'in' or 'out'."""
    params = _ring_system_params()
    params['ring_system']['features'] = [dict(_edge_feature(), side='left')]
    with pytest.raises(SimSceneValidationError, match=r"side must be 'in' or 'out'"):
        validate_sim_params(params)


@pytest.mark.parametrize('key', ['wavelength', 'damping'])
def test_validate_sim_params_requires_wave_train_keys(key: str) -> None:
    """A wave feature needs both its radial train parameters."""
    params = _ring_system_params()
    feature = _wave_feature()
    del feature[key]
    params['ring_system']['features'] = [feature]
    with pytest.raises(SimSceneValidationError, match=f"{key} is required for kind 'wave'"):
        validate_sim_params(params)


@pytest.mark.parametrize('key', ['wavelength', 'damping'])
def test_validate_sim_params_rejects_wave_keys_on_other_kinds(key: str) -> None:
    """The wave-train keys are rejected on kinds that ignore them."""
    params = _ring_system_params()
    params['ring_system']['features'][0][key] = 5.0
    with pytest.raises(SimSceneValidationError, match=f'{key} is not allowed for kind'):
        validate_sim_params(params)


def test_validate_sim_params_accepts_orbit_modes_and_edge_wave() -> None:
    """An orbit with m-modes and an edge wave validates."""
    params = _ring_system_params()
    params['ring_system']['features'][0]['orbit'] = {
        'a': 40.0,
        'ae': 1.0,
        'long_peri': 10.0,
        'rate_peri': 0.5,
        'modes': [{'m': 2, 'amp': 1.5, 'peri': 30.0}],
        'edge_wave': {'amp': 1.0, 'wavelength': 8.0, 'damp': 0.5, 'lam0': 90.0},
    }
    assert validate_sim_params(params) is params


def test_validate_sim_params_rejects_mode_1_in_modes_list() -> None:
    """The modes list carries m >= 2 only (mode 1 is the base ellipse)."""
    params = _ring_system_params()
    params['ring_system']['features'][0]['orbit']['modes'] = [{'m': 1, 'amp': 1.0, 'peri': 0.0}]
    with pytest.raises(SimSceneValidationError, match=r'modes\[0\]\.m must be an integer >= 2'):
        validate_sim_params(params)


def test_validate_sim_params_rejects_unknown_mode_key() -> None:
    """An unmodeled m-mode key fails validation."""
    params = _ring_system_params()
    params['ring_system']['features'][0]['orbit']['modes'] = [
        {'m': 2, 'amp': 1.0, 'peri': 0.0, 'rate': 1.0}
    ]
    with pytest.raises(SimSceneValidationError, match=r'modes\[0\].*rate'):
        validate_sim_params(params)


@pytest.mark.parametrize('key', ['wavelength', 'damp'])
def test_validate_sim_params_requires_edge_wave_scales(key: str) -> None:
    """An edge wave needs its wavelength and damping constants."""
    params = _ring_system_params()
    wave = {'amp': 1.0, 'wavelength': 8.0, 'damp': 0.5, 'lam0': 90.0}
    del wave[key]
    params['ring_system']['features'][0]['orbit']['edge_wave'] = wave
    with pytest.raises(SimSceneValidationError, match=f'edge_wave.{key} is required'):
        validate_sim_params(params)


def test_validate_sim_params_accepts_edge_wave_damp_at_the_cap() -> None:
    """An edge-wave damp of exactly 2.0 radians (the cap) validates."""
    params = _ring_system_params()
    params['ring_system']['features'][0]['orbit']['edge_wave'] = {
        'amp': 1.0,
        'wavelength': 8.0,
        'damp': 2.0,
        'lam0': 90.0,
    }
    assert validate_sim_params(params) is params


def test_validate_sim_params_rejects_edge_wave_damp_above_the_cap() -> None:
    """damp > 2.0 radians fails: the modular wrap seam would exceed exp(-pi)."""
    params = _ring_system_params()
    params['ring_system']['features'][0]['orbit']['edge_wave'] = {
        'amp': 1.0,
        'wavelength': 8.0,
        'damp': 2.5,
        'lam0': 90.0,
    }
    with pytest.raises(SimSceneValidationError, match=r'damp must be <= 2\.0 radians'):
        validate_sim_params(params)


def test_validate_sim_params_rejects_unknown_edge_wave_key() -> None:
    """An unmodeled edge-wave key fails validation."""
    params = _ring_system_params()
    params['ring_system']['features'][0]['orbit']['edge_wave'] = {
        'amp': 1.0,
        'wavelength': 8.0,
        'damp': 0.5,
        'lam0': 90.0,
        'phase': 1.0,
    }
    with pytest.raises(SimSceneValidationError, match=r'edge_wave.*phase'):
        validate_sim_params(params)


def test_validate_sim_params_requires_ring_feature_orbit() -> None:
    """The catalog orbit map (with a) is required on every feature."""
    params = _ring_system_params()
    del params['ring_system']['features'][0]['orbit']
    with pytest.raises(SimSceneValidationError, match='orbit is required'):
        validate_sim_params(params)


def test_validate_sim_params_rejects_out_of_range_phase_g() -> None:
    """The Henyey-Greenstein asymmetry parameter lives strictly inside (-1, 1)."""
    params = _ring_system_params()
    params['ring_system']['features'][0]['phase_g'] = 1.0
    with pytest.raises(SimSceneValidationError, match=r'phase_g must lie in \(-1, 1\)'):
        validate_sim_params(params)


def test_spk_error_requires_ring_system_range_km() -> None:
    """A scene planting spacecraft-ephemeris error must range its ring system."""
    params = _ring_system_params()
    params['spk_error'] = {'dv_px': 1.0, 'du_px': 0.0, 'reference_range_km': 1.0e5}
    with pytest.raises(SimSceneValidationError, match='ring_system needs range_km'):
        validate_sim_params(params)


def test_every_top_level_key_is_classified() -> None:
    """Every allowed top-level key has exactly one of the three boundary classes."""
    classified = TOP_LEVEL_IDEALIZED_KEYS | TOP_LEVEL_TRUTH_KEYS | TOP_LEVEL_TEST_ONLY_KEYS
    assert classified == _ALLOWED_KEYS
    assert not TOP_LEVEL_IDEALIZED_KEYS & TOP_LEVEL_TRUTH_KEYS
    assert not TOP_LEVEL_IDEALIZED_KEYS & TOP_LEVEL_TEST_ONLY_KEYS
    assert not TOP_LEVEL_TRUTH_KEYS & TOP_LEVEL_TEST_ONLY_KEYS


def test_test_only_keys_are_not_truth_keys() -> None:
    """The test-only class is disjoint from the truth set the boundary iterates."""
    assert not (TOP_LEVEL_TEST_ONLY_KEYS & TRUTH_KEYS)


@pytest.mark.parametrize('block', sorted(_OBJECT_BLOCKS))
def test_every_object_key_is_classified(block: str) -> None:
    """Every allowed per-object key has exactly one boundary classification."""
    allowed, idealized, truth = _OBJECT_BLOCKS[block]
    assert idealized | truth == allowed
    assert not idealized & truth


def test_truth_keys_cover_top_level_and_blocks() -> None:
    """TRUTH_KEYS carries the top-level names plus dotted per-block paths."""
    assert TOP_LEVEL_TRUTH_KEYS <= TRUTH_KEYS
    for block, (_allowed, _idealized, truth) in _OBJECT_BLOCKS.items():
        for key in truth:
            assert f'{block}.{key}' in TRUTH_KEYS
