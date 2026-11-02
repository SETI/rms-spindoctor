"""A scene whose published instrument config repeats its planted truth warns.

``instrument_config`` crosses the information boundary unfiltered, so a planted
value copied into it reaches the navigator as published knowledge.  The validator
warns for each such coincidence, naming both sites, unless the scene lists the
truth path in ``published_equals_truth``.
"""

import warnings
from typing import Any

import pytest

from spindoctor.sim.scene import (
    SimSceneTruthEchoWarning,
    SimSceneValidationError,
    build_nav_params,
    validate_sim_params,
)
from spindoctor.sim.scene_truth_echo import truth_echoes


def _scene(**extra: Any) -> dict[str, Any]:
    """A minimal scene with a planted noise block and a Gaussian PSF."""
    return {
        'size_v': 64,
        'size_u': 64,
        'random_seed': 7,
        'instrument': 'coiss_nac',
        'noise': {'poisson': True, 'read_noise_dn': 4.0, 'bias_dn': 20.0},
        'optics': {'psf': {'sigma_v': 1.3, 'sigma_u': 1.1}},
        **extra,
    }


def test_copying_the_planted_read_noise_warns_naming_both_sites() -> None:
    """The warning names the truth key and the published key that repeats it."""
    scene = _scene(instrument_config={'noise': {'read_noise_dn': 4.0}})
    with pytest.warns(
        SimSceneTruthEchoWarning,
        match=r'instrument_config\.noise\.read_noise_dn repeats the planted '
        r'noise\.read_noise_dn',
    ):
        validate_sim_params(scene, source='echo.yaml')


def test_a_vidicon_echo_warns_without_claiming_the_navigator_reads_it() -> None:
    """The navigator doesn't read vidicon keys, so their warning says what the copy does."""
    scene = _scene(instrument_config={'noise': {'vidicon': {'read_noise_line_dn': 2.0}}})
    scene['noise']['vidicon'] = {'read_noise_line_dn': 2.0}
    with pytest.warns(SimSceneTruthEchoWarning) as caught:
        validate_sim_params(scene, source='echo.yaml')
    assert [str(warning.message) for warning in caught] == [
        'echo.yaml: instrument_config.noise.vidicon.read_noise_line_dn repeats the planted '
        "noise.vidicon.read_noise_line_dn, so the published config carries the scene's "
        "realization instead of the camera's; list 'noise.vidicon.read_noise_line_dn' in "
        'published_equals_truth if they are equal on purpose'
    ]


def test_each_repeated_value_is_listed() -> None:
    """Noise keys, vidicon keys, and the PSF sigma each count as an echo."""
    scene = _scene(
        instrument_config={
            'star_psf_sigma': 1.3,
            'noise': {
                'read_noise_dn': 4.0,
                'bias_dn': 20.0,
                'vidicon': {'read_noise_line_dn': 2.0},
            },
        }
    )
    scene['noise']['vidicon'] = {'read_noise_line_dn': 2.0}
    assert truth_echoes(scene) == [
        ('noise.read_noise_dn', 'instrument_config.noise.read_noise_dn'),
        ('noise.bias_dn', 'instrument_config.noise.bias_dn'),
        ('noise.vidicon.read_noise_line_dn', 'instrument_config.noise.vidicon.read_noise_line_dn'),
        ('optics.psf.sigma_v', 'instrument_config.star_psf_sigma'),
    ]


@pytest.mark.parametrize(
    'instrument_config',
    [
        {'noise': {'read_noise_dn': 5.0}},
        {'noise': {'saturation_dn': 4095}},
        {'star_psf_sigma': 1.2},
    ],
    ids=['different-value', 'published-only-key', 'different-sigma'],
)
def test_published_values_that_differ_from_the_truth_validate_quietly(
    instrument_config: dict[str, Any],
) -> None:
    """A published value that doesn't repeat a planted one doesn't warn."""
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        validate_sim_params(_scene(instrument_config=instrument_config))


def test_a_declared_coincidence_validates_quietly() -> None:
    """Listing the truth path in published_equals_truth silences its warning."""
    scene = _scene(
        instrument_config={'noise': {'read_noise_dn': 4.0}},
        published_equals_truth=['noise.read_noise_dn'],
    )
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        validate_sim_params(scene)


def test_a_declaration_covers_only_the_paths_it_lists() -> None:
    """An undeclared coincidence still warns beside a declared one."""
    scene = _scene(
        instrument_config={'noise': {'read_noise_dn': 4.0, 'bias_dn': 20.0}},
        published_equals_truth=['noise.read_noise_dn'],
    )
    with pytest.warns(SimSceneTruthEchoWarning, match=r'the planted noise\.bias_dn') as caught:
        validate_sim_params(scene)
    assert len(caught) == 1


def test_a_stale_declaration_is_refused() -> None:
    """Declaring a path whose values differ fails, so an opt-out can't outlive its reason."""
    scene = _scene(
        instrument_config={'noise': {'read_noise_dn': 5.0}},
        published_equals_truth=['noise.read_noise_dn'],
    )
    with pytest.raises(
        SimSceneValidationError,
        match=r"lists \['noise\.read_noise_dn'\], but instrument_config doesn't repeat",
    ):
        validate_sim_params(scene)


def test_a_malformed_declaration_is_refused() -> None:
    """published_equals_truth must be a list of strings."""
    with pytest.raises(SimSceneValidationError, match='must be a list of truth paths'):
        validate_sim_params(_scene(published_equals_truth='noise.read_noise_dn'))


def test_the_declaration_stays_behind_the_boundary() -> None:
    """The navigator's view doesn't carry published_equals_truth."""
    scene = _scene(
        instrument_config={'noise': {'read_noise_dn': 4.0}},
        published_equals_truth=['noise.read_noise_dn'],
    )
    assert 'published_equals_truth' not in build_nav_params(scene)
