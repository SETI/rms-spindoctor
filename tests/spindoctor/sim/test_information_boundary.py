"""The information boundary: no truth key is reachable through nav_params.

This is the independence guarantee of the simulator-realism program: the
navigator-side models consume only ``obs.nav_params``, and this test builds a
scene exercising EVERY entry of the schema's ``TRUTH_KEYS`` inventory and
asserts none is reachable through the filtered view.  The test iterates the
frozenset itself, so a truth key added to the schema without a filter/test
extension fails here, not silently.
"""

from typing import Any

import pytest

from spindoctor.obs.obs_inst_sim import ObsSim
from spindoctor.sim.render import render_combined_model
from spindoctor.sim.scene import (
    ATMOSPHERE_KEYS,
    TOP_LEVEL_TEST_ONLY_KEYS,
    TOP_LEVEL_TRUTH_KEYS,
    TRUTH_KEYS,
    build_nav_params,
    validate_sim_params,
)

# One concrete exercising value per truth key.  The completeness check below
# fails if TRUTH_KEYS gains an entry with no sample here, so a later phase
# adding a truth key must extend this scene in the same change.
_TRUTH_SAMPLES: dict[str, Any] = {
    'random_seed': 99,
    'offset_v': 2.5,
    'offset_u': -1.5,
    'offset_rotation_deg': 0.4,
    'sky_counts': {'a': -3.1, 'b': 0.34, 'density_factor': 5.0, 'diffuse_e_per_px': 1.0},
    'noise': {'poisson': True, 'read_noise_dn': 4.0, 'bias_dn': 12.0},
    'oversample': 4,
    'optics': {
        'psf': {'sigma_v': 0.6, 'sigma_u': 0.6, 'w': 0.02, 'r0': 2.0, 'n': 3.0},
        'smear': [{'dv_px': 1.5, 'du_px': 0.0, 'object_class': 'all'}],
        'distortion': {'k1': 0.01, 'k2': 0.0, 'center_v': 48.0, 'center_u': 48.0},
        'ghosts': [{'dv_px': 5.0, 'du_px': -3.0, 'amplitude': 0.05, 'defocus_sigma': 2.0}],
        'stray_light': {'amplitude': 0.2, 'direction_deg': 30.0, 'model': 'linear'},
    },
    'detector': {
        'gain_state': 2,
        'detector_model': 'ccd',
        'exposure_ref_sec': 1.0,
        'quantization': 'exact',
    },
    'artifacts': {
        'instrument_defaults': True,
        'adversarial': True,
        'missing_lines': {'expected_events': 2.0, 'contiguous_run': True},
        'hot_pixels': {'pixel_fraction': 0.001, 'amplitude_e': 40000.0},
    },
    'spk_error': {'dv_px': 0.8, 'du_px': -0.4, 'reference_range_km': 100000.0},
    'star_catalog_scatter_px': 3.0,
    'bodies.crater_fill': 0.4,
    'bodies.crater_min_radius': 0.06,
    'bodies.crater_max_radius': 0.2,
    'bodies.crater_power_law_exponent': 2.5,
    'bodies.crater_relief_scale': 0.5,
    'bodies.limb_relief_rms': 0.02,
    'bodies.limb_relief_corr_deg': 12.0,
    'bodies.photometric_law': 'lommel_seeliger',
    'bodies.minnaert_k': 0.6,
    'bodies.opposition_surge': {'amplitude': 0.5, 'width_deg': 5.0},
    'bodies.albedo_texture': {
        'rms': 0.15,
        'corr_px': 12.0,
        'spots': [{'lat_deg': 45.0, 'lon_deg': 20.0, 'radius_deg': 8.0, 'albedo_factor': 0.6}],
    },
    'bodies.disc_texture': {
        'band_amplitude': 0.2,
        'band_wavenumber': 8.0,
        'band_phase_deg': 15.0,
        'storms': [{'lat_deg': -20.0, 'lon_deg': 90.0, 'radius_deg': 6.0, 'albedo_factor': 1.4}],
    },
    'bodies.transits': [
        {
            'moon': {'dv_px': -3.0, 'du_px': 2.0, 'radius_px': 3.0, 'albedo_factor': 1.5},
            'shadow': {'dv_px': 1.0, 'du_px': 4.0, 'radius_px': 3.0, 'darkness': 0.8},
        }
    ],
    # The atmosphere block is ONE truth key, so the parametrized reachability
    # test below probes it as a unit and a key omitted here would be neither
    # rendered nor probed.  Every key the validator accepts is therefore
    # listed, and ``test_atmosphere_sample_covers_every_key`` fails when the
    # block's inventory grows without this sample growing with it.
    'bodies.atmosphere': {
        'scale_height_px': 2.0,
        'tau_ref': 1.5,
        'ref_altitude_px': 1.0,
        'g': 0.6,
        'detached_px': 5.0,
        'interior_ramp_amplitude': 0.2,
        'ns_asymmetry_amplitude': 0.3,
        'ns_falloff_ratio': 1.8,
        'axis_tilt_deg': 12.0,
        'sector_sharpness_gradient': 0.5,
        'cloud_blobs': [{'center_vu': [-4.0, 3.0], 'sigma_px': 2.5, 'amplitude': 0.35}],
    },
    'bodies.shading': 'gouraud',
    'bodies.pose_scatter': {'sigma_deg': 2.0},
    'bodies.seed': 11,
    'bodies.anti_aliasing': 0.7,
    'bodies.nav_override': {'mesh_lumpiness': 0.0},
    'stars.psf_sigma': 2.5,
    'stars.catalog_error_v': 1.2,
    'stars.catalog_error_u': -0.8,
    'stars.companion': {'sep_px': 2.0, 'delta_mag': 1.5, 'angle_deg': 45.0},
    'stars.delta_mag': 0.7,
    'ring_system.features.albedo': 0.7,
    'ring_system.features.phase_g': 0.4,
    'ring_system.features.orbit_error': {
        'delta_a_px': 1.5,
        'delta_ae_px': 0.4,
        'delta_long_peri_deg': 10.0,
    },
    'ring_system.azimuthal': {
        'modulation': {'amplitude': 0.2, 'm': 2, 'brightness_peak_longitude_deg': 30.0},
        'shadow': {'start_deg': 100.0, 'extent_deg': 40.0, 'darkness': 0.9},
        'spokes': {
            'count': 3,
            'r_inner_px': 3.0,
            'r_outer_px': 6.0,
            'contrast': -0.4,
            'width_deg': 12.0,
        },
    },
    'ring_system.moonlets': [
        {
            'a_px': 6.0,
            'lam_deg': 45.0,
            'radius_px': 1.0,
            'amplitude': 0.4,
            'propeller': {'length_deg': 20.0, 'width_px': 1.0, 'contrast': -0.6},
        }
    ],
}

# The true (rendered) values nav_override hides from the navigator.
_TRUE_MESH_LUMPINESS = 0.4


def _truth_exercising_scene() -> dict[str, Any]:
    """A validated scene carrying every truth key with a non-default value."""
    scene: dict[str, Any] = {
        'schema_version': 2,
        'scene_name': 'boundary_probe',
        'instrument': 'coiss_nac',
        'size_v': 96,
        'size_u': 96,
        'exposure_sec': 1.5,
        'time': 100.0,
        'ring_epoch': 50.0,
        'closest_planet': 'SATURN',
        'bodies': [
            {
                'name': 'LUMPY',
                'shape_model': 'polyhedral_mesh',
                'center_v': 40.0,
                'center_u': 40.0,
                'axis1': 30.0,
                'axis2': 26.0,
                'axis3': 26.0,
                'illumination_angle': 20.0,
                'phase_angle': 30.0,
                'range_km': 500000.0,
                'mesh_lumpiness': _TRUE_MESH_LUMPINESS,
                'mesh_seed': 3,
                'shading': _TRUTH_SAMPLES['bodies.shading'],
                'pose_scatter': dict(_TRUTH_SAMPLES['bodies.pose_scatter']),
                'seed': _TRUTH_SAMPLES['bodies.seed'],
                'anti_aliasing': _TRUTH_SAMPLES['bodies.anti_aliasing'],
                'nav_override': dict(_TRUTH_SAMPLES['bodies.nav_override']),
            },
            {
                # An ellipsoid body carrying the ellipsoid-path truth keys
                # (craters, relief field, photometric law, opposition surge,
                # texture, transits, atmosphere), which the validator rejects
                # on the mesh body above.
                'name': 'ROUND',
                'center_v': 70.0,
                'center_u': 24.0,
                'axis1': 20.0,
                'axis2': 18.0,
                'axis3': 18.0,
                'illumination_angle': 40.0,
                'phase_angle': 60.0,
                'range_km': 800000.0,
                'crater_fill': _TRUTH_SAMPLES['bodies.crater_fill'],
                'crater_min_radius': _TRUTH_SAMPLES['bodies.crater_min_radius'],
                'crater_max_radius': _TRUTH_SAMPLES['bodies.crater_max_radius'],
                'crater_power_law_exponent': (_TRUTH_SAMPLES['bodies.crater_power_law_exponent']),
                'crater_relief_scale': _TRUTH_SAMPLES['bodies.crater_relief_scale'],
                'limb_relief_rms': _TRUTH_SAMPLES['bodies.limb_relief_rms'],
                'limb_relief_corr_deg': _TRUTH_SAMPLES['bodies.limb_relief_corr_deg'],
                'photometric_law': _TRUTH_SAMPLES['bodies.photometric_law'],
                'minnaert_k': _TRUTH_SAMPLES['bodies.minnaert_k'],
                'opposition_surge': dict(_TRUTH_SAMPLES['bodies.opposition_surge']),
                'albedo_texture': dict(_TRUTH_SAMPLES['bodies.albedo_texture']),
                'disc_texture': dict(_TRUTH_SAMPLES['bodies.disc_texture']),
                'transits': list(_TRUTH_SAMPLES['bodies.transits']),
                'atmosphere': dict(_TRUTH_SAMPLES['bodies.atmosphere']),
            },
        ],
        'stars': [
            {
                'name': 'S1',
                'v': 70.0,
                'u': 70.0,
                'vmag': 4.0,
                'psf_sigma': _TRUTH_SAMPLES['stars.psf_sigma'],
                'catalog_error_v': _TRUTH_SAMPLES['stars.catalog_error_v'],
                'catalog_error_u': _TRUTH_SAMPLES['stars.catalog_error_u'],
                'companion': dict(_TRUTH_SAMPLES['stars.companion']),
                'delta_mag': _TRUTH_SAMPLES['stars.delta_mag'],
            }
        ],
        # A small inclined ring system in the frame corner, clear of the
        # bodies (overlap without explicit depth is a render error),
        # carrying the per-feature photometric truth keys.
        'ring_system': {
            'geometry': {
                'center_v': 12.0,
                'center_u': 12.0,
                'opening_deg_obs': 30.0,
                'opening_deg_sun': 20.0,
                'node_deg': 25.0,
            },
            'range_km': 300000.0,
            'km_per_pixel': 500.0,
            'phase_deg': 40.0,
            'features': [
                {
                    'name': 'F1',
                    'kind': 'ringlet',
                    'tau': 1.2,
                    'width_px': 3.0,
                    'navigable': True,
                    'orbit': {
                        'a_px': 4.0,
                        'ae_px': 0.0,
                        'long_peri_deg': 0.0,
                        'rate_peri_deg_per_day': 0.0,
                    },
                    'declared_orbit_sigma': {'sigma_a_px': 0.5},
                    'albedo': _TRUTH_SAMPLES['ring_system.features.albedo'],
                    'phase_g': _TRUTH_SAMPLES['ring_system.features.phase_g'],
                    'orbit_error': dict(_TRUTH_SAMPLES['ring_system.features.orbit_error']),
                }
            ],
            'azimuthal': dict(_TRUTH_SAMPLES['ring_system.azimuthal']),
            'moonlets': list(_TRUTH_SAMPLES['ring_system.moonlets']),
        },
    }
    for key in TOP_LEVEL_TRUTH_KEYS:
        scene[key] = _TRUTH_SAMPLES[key]
    return validate_sim_params(scene, source='boundary_probe')


def _addressed_objects(view: dict[str, Any], truth_key: str) -> tuple[list[dict[str, Any]], str]:
    """The object mappings a dotted truth key addresses in a scene-shaped view.

    ``<block>.<key>`` addresses every entry of an object-block list;
    ``ring_system.<key>`` addresses the ring_system mapping itself; and
    ``ring_system.features.<key>`` addresses every feature entry.

    Parameters:
        view: A full scene or a filtered ``nav_params`` mapping.
        truth_key: A dotted TRUTH_KEYS entry.

    Returns:
        ``(objects, leaf_key)``: the mappings to probe and the key to probe
        them for.
    """
    if truth_key.startswith('ring_system.features.'):
        leaf = truth_key.rsplit('.', 1)[1]
        ring_system = view.get('ring_system') or {}
        return list(ring_system.get('features') or []), leaf
    if truth_key.startswith('ring_system.'):
        return [view.get('ring_system') or {}], truth_key.split('.', 1)[1]
    block, leaf = truth_key.split('.', 1)
    return list(view.get(block) or []), leaf


def test_every_truth_key_has_an_exercising_sample() -> None:
    """The probe scene covers TRUTH_KEYS exactly (no stale or missing samples)."""
    assert set(_TRUTH_SAMPLES) == set(TRUTH_KEYS)


@pytest.mark.parametrize('truth_key', sorted(TRUTH_KEYS))
def test_truth_key_unreachable_through_nav_params(truth_key: str) -> None:
    """No TRUTH_KEYS entry is reachable through the filtered view."""
    nav = build_nav_params(_truth_exercising_scene())
    if '.' in truth_key:
        objects, leaf = _addressed_objects(nav, truth_key)
        for obj in objects:
            assert leaf not in obj
    else:
        assert truth_key not in nav


def test_atmosphere_sample_covers_every_key() -> None:
    """The atmosphere sample exercises every key the block may carry.

    ``bodies.atmosphere`` is a single entry of ``TRUTH_KEYS``, so the
    parametrized reachability test treats the whole block as one atomic
    unit: a key added to the block but omitted from the sample would fail
    nothing and would sit un-probed for leakage.  This is the check that
    turns the block's own inventory into the completeness requirement.
    """
    assert set(_TRUTH_SAMPLES['bodies.atmosphere']) == set(ATMOSPHERE_KEYS)


@pytest.mark.parametrize('key', sorted(ATMOSPHERE_KEYS))
def test_atmosphere_key_unreachable_through_nav_params(key: str) -> None:
    """No individual atmosphere key survives the filter on any body.

    The atomic-block test alone would pass if the filter dropped the block
    but some other channel republished a key at body level; this asserts the
    per-key absence directly.
    """
    nav = build_nav_params(_truth_exercising_scene())
    for body in nav['bodies']:
        assert key not in body


def test_probe_scene_actually_carries_every_truth_key() -> None:
    """Guards the parametrized test: the probe exercises what it claims."""
    scene = _truth_exercising_scene()
    for truth_key in TRUTH_KEYS:
        if '.' in truth_key:
            objects, leaf = _addressed_objects(scene, truth_key)
            assert any(leaf in obj for obj in objects)
        else:
            assert truth_key in scene


def test_nav_override_is_overlaid_not_exposed() -> None:
    """The navigator sees the believed value; the true value never crosses."""
    nav = build_nav_params(_truth_exercising_scene())
    body = nav['bodies'][0]
    assert body['mesh_lumpiness'] == 0.0
    assert 'nav_override' not in body


def test_idealized_keys_survive_the_filter() -> None:
    """Catalog geometry and epochs the navigator may know pass through."""
    nav = build_nav_params(_truth_exercising_scene())
    assert nav['ring_epoch'] == 50.0
    assert nav['time'] == 100.0
    assert nav['bodies'][0]['range_km'] == 500000.0
    star = nav['stars'][0]
    assert star['vmag'] == 4.0
    assert nav['ring_system']['features'][0]['orbit']['a_px'] == 4.0


def test_ring_system_geometry_crosses_with_the_navigable_feature() -> None:
    """The shared geometry and the navigable feature's catalog view cross.

    Both sides project through the geometry block by design, so it crosses
    with the block-level scale keys.  The probe feature is navigable, so it
    crosses too -- carrying its idealized keys (kind, shape, catalog orbit,
    declared orbit uncertainty) and none of its truth (the parametrized
    boundary test covers albedo / phase_g / orbit_error explicitly).
    """
    nav = build_nav_params(_truth_exercising_scene())
    ring_system = nav['ring_system']
    assert ring_system['geometry']['opening_deg_obs'] == 30.0
    assert ring_system['geometry']['node_deg'] == 25.0
    assert ring_system['range_km'] == 300000.0
    assert ring_system['km_per_pixel'] == 500.0
    assert ring_system['phase_deg'] == 40.0
    features = ring_system['features']
    assert len(features) == 1
    assert features[0]['name'] == 'F1'
    assert features[0]['kind'] == 'ringlet'
    assert features[0]['orbit'] == {
        'a_px': 4.0,
        'ae_px': 0.0,
        'long_peri_deg': 0.0,
        'rate_peri_deg_per_day': 0.0,
    }
    assert features[0]['declared_orbit_sigma'] == {'sigma_a_px': 0.5}
    assert features[0]['navigable'] is True


def test_non_navigable_ring_feature_is_dropped_but_still_renders() -> None:
    """A non-navigable feature is invisible to the navigator yet on the frame.

    The default ``navigable: false`` drops the feature from ``nav_params``
    entirely, while the renderer draws it regardless -- the false-lock /
    distractor regime.  The rendered frame must change when the confounder
    is added; the navigator's view must not.
    """
    scene = _truth_exercising_scene()
    confounder = {
        'name': 'CONFOUNDER',
        'kind': 'ringlet',
        'tau': 2.0,
        'width_px': 2.0,
        'orbit': {'a_px': 8.0, 'ae_px': 0.0, 'long_peri_deg': 0.0, 'rate_peri_deg_per_day': 0.0},
    }
    with_confounder = _truth_exercising_scene()
    with_confounder['ring_system']['features'] = [
        *with_confounder['ring_system']['features'],
        confounder,
    ]
    validate_sim_params(with_confounder, source='boundary_probe')

    nav_without = build_nav_params(scene)
    nav_with = build_nav_params(with_confounder)
    assert nav_with['ring_system'] == nav_without['ring_system']

    img_without, _meta0 = render_combined_model(scene)
    img_with, _meta1 = render_combined_model(with_confounder)
    assert bool((img_with != img_without).any())


def test_obs_sim_exposes_only_the_filtered_view() -> None:
    """ObsSim publishes nav_params and no renderer star records."""
    obs = ObsSim.from_file('/tmp/boundary_probe.yaml', sim_params=_truth_exercising_scene())
    for truth_key in TRUTH_KEYS:
        if '.' in truth_key:
            objects, leaf = _addressed_objects(obs.nav_params, truth_key)
            for obj in objects:
                assert leaf not in obj
        else:
            assert truth_key not in obs.nav_params
    # The star-list replumb: the renderer's output star records no longer
    # cross to the navigator side under any name.
    assert not hasattr(obs, 'sim_star_list')


def test_star_limit_is_independent_of_scene_noise() -> None:
    """The star detection limit derives from published config, not scene noise.

    Two observations differing only in the truth-side ``noise`` block must
    report the same ``star_max_usable_vmag``: the navigator's limiting
    magnitude comes from the emulated instrument's published detector model,
    so the scene's planted noise cannot leak through this channel.  A scene
    that plants noise different from the published values gets an
    honestly-wrong detection limit by design.
    """
    scene_quiet = _truth_exercising_scene()
    scene_noisy = _truth_exercising_scene()
    scene_noisy['noise'] = {
        'poisson': False,
        'read_noise_dn': 250.0,
        'bias_dn': 0.0,
        'signal_full_scale_frac': 0.01,
    }
    obs_quiet = ObsSim.from_file('/tmp/boundary_probe.yaml', sim_params=scene_quiet)
    obs_noisy = ObsSim.from_file('/tmp/boundary_probe.yaml', sim_params=scene_noisy)
    assert obs_noisy.star_max_usable_vmag() == obs_quiet.star_max_usable_vmag()


def _star_only_scene(exposure_sec: float) -> dict[str, Any]:
    """A minimal star-only scene at the given exposure."""
    return {
        'instrument': 'coiss_nac',
        'size_v': 64,
        'size_u': 64,
        'random_seed': 5,
        'exposure_sec': exposure_sec,
        'stars': [{'name': 'S', 'v': 32.0, 'u': 32.0, 'vmag': 4.0}],
    }


def test_star_limit_scales_with_scene_exposure() -> None:
    """The star detection limit tracks the scene's idealized exposure.

    ``exposure_sec`` is commanded, navigator-visible information, and the
    renderer scales every star's deposited flux by it, so the matched-filter
    limiting magnitude must move with it through the flux formula: a factor of
    ten in exposure is exactly 2.5 magnitudes of limit, in the same direction.
    """
    reference = ObsSim.from_file(
        '/tmp/exposure_probe.yaml', sim_params=_star_only_scene(1.0)
    ).star_max_usable_vmag()
    short = ObsSim.from_file(
        '/tmp/exposure_probe.yaml', sim_params=_star_only_scene(0.1)
    ).star_max_usable_vmag()
    long_exp = ObsSim.from_file(
        '/tmp/exposure_probe.yaml', sim_params=_star_only_scene(10.0)
    ).star_max_usable_vmag()
    assert short == pytest.approx(reference - 2.5, abs=1e-9)
    assert long_exp == pytest.approx(reference + 2.5, abs=1e-9)


def test_test_only_keys_are_stripped_from_nav_params() -> None:
    """The scene-level ``expected`` block is a third class the navigator never sees.

    ``expected`` is neither idealized nor truth: the assertion machinery reads
    it, but the boundary filter's default-deny keeps it out of ``nav_params``
    exactly as it keeps the truth keys out.
    """
    scene = _truth_exercising_scene()
    scene['expected'] = {'status': 'failed', 'confidence_tier': 'failed'}
    nav = build_nav_params(validate_sim_params(scene, source='boundary_probe'))
    for key in TOP_LEVEL_TEST_ONLY_KEYS:
        assert key not in nav


def test_non_navigable_star_is_dropped_but_still_a_catalog_star_for_others() -> None:
    """A non-navigable star renders but is absent from the navigator's catalog.

    The ``navigable`` flag drives the boundary filter: a ``navigable: false``
    star is dropped from ``nav_params`` entirely (a confounder the navigator has
    no knowledge of), while navigable stars survive with their catalog values.
    """
    scene = _truth_exercising_scene()
    scene['stars'] = [
        {'name': 'KNOWN', 'v': 40.0, 'u': 40.0, 'vmag': 4.0, 'navigable': True},
        {'name': 'CONFOUNDER', 'v': 60.0, 'u': 60.0, 'vmag': 4.2, 'navigable': False},
    ]
    nav = build_nav_params(validate_sim_params(scene, source='boundary_probe'))
    names = {star['name'] for star in nav['stars']}
    assert names == {'KNOWN'}


def test_nav_params_values_are_isolated_copies() -> None:
    """Mutating the filtered view cannot reach back into the scene."""
    scene = _truth_exercising_scene()
    nav = build_nav_params(scene)
    nav['bodies'][0]['axis1'] = 999.0
    nav['ring_system']['features'][0]['orbit']['a_px'] = 999.0
    assert scene['bodies'][0]['axis1'] == 30.0
    assert scene['ring_system']['features'][0]['orbit']['a_px'] == 4.0
