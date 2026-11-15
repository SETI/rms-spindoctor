"""The navigator's star values for the simulated cameras.

Each raw-DN navigator block states its own DN zero point and, for every binned mode its
camera offers, the mode's read noise and zero point.  Each value agrees with the
simulator's camera within the digits the navigator's config file writes it with.  A
scene's ``instrument_config`` overrides of the binned-mode table are checked as the
configuration loader checks the table, and a binned scene can't set an unbinned value
that its frame reads from the table instead.
"""

from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from ruamel.yaml import YAML
from ruamel.yaml.nodes import MappingNode, Node

import spindoctor
from spindoctor.config import DEFAULT_CONFIG
from spindoctor.sim.forward.detector.params import resolve_detector_params
from spindoctor.sim.instrument_catalog import (
    GENERIC_INSTRUMENT,
    load_instrument_catalog,
    sim_instrument_names,
)
from spindoctor.sim.instrument_navigator import resolve_sim_inst_config
from spindoctor.sim.instrument_resolvers import (
    cataloged_binnings,
    resolve_data_units,
    resolve_star_flux_zero_point,
)
from spindoctor.sim.scene import validate_scene
from spindoctor.sim.scene_schema import SimSceneValidationError

_CONFIG_FILES_DIR = Path(spindoctor.__file__).parent / 'config_files'


def _raw_dn_navigator_instruments() -> list[str]:
    """The entries whose navigator block reads raw DN, so it uses a DN star zero point.

    The navigator's star limit (``ObsSim.star_max_usable_vmag``) reads
    ``star_flux_dn_per_s_vmag0`` exactly when the block's ``data_units`` is
    ``raw_dn``; a calibrated block never consults it.  It reads each entry's own units
    instead of resolving its navigator block, so a block without units fails its own
    test instead of breaking collection of this file.
    """
    return [
        name
        for name in (*sim_instrument_names(), GENERIC_INSTRUMENT)
        if resolve_data_units(name) == 'raw_dn'
    ]


def _child(node: Node, key: str) -> Node | None:
    """The value node under ``key`` in a mapping node, or None when there is none."""
    if not isinstance(node, MappingNode):
        return None
    return next((value for name, value in node.value if name.value == key), None)


def _written_step(instrument: str, *keys: str | int) -> float:
    """The place value of the last digit a navigator value is written with.

    The parsed number can't tell ``0.400`` from ``0.4``, so this reads the text of the
    value in the packaged config file that sets it last, under the instrument's
    navigator section and detector: ``3.61e6`` gives 1e4, and ``0.400`` gives 1e-3.

    Parameters:
        instrument: The simulated instrument whose navigator block holds the value.
        *keys: The value's key path inside the block.

    Returns:
        The place value of the value's last written digit.
    """
    navigator = load_instrument_catalog().entry(instrument)['navigator']
    path = [navigator['section'], navigator.get('detector'), *keys]
    text = None
    for config_file in sorted(_CONFIG_FILES_DIR.glob('*.yaml')):
        node: Node | None = YAML().compose(config_file.read_text(encoding='utf-8'))
        for key in path:
            if key is not None and node is not None:
                node = _child(node, str(key))
        if node is not None:
            text = node.value
    assert isinstance(text, str), f'no config file sets {path}'
    exponent = Decimal(text).as_tuple().exponent
    assert isinstance(exponent, int)
    return float(Decimal(1).scaleb(exponent))


@pytest.mark.parametrize('instrument', _raw_dn_navigator_instruments())
def test_the_navigator_states_a_zero_point_for_a_raw_dn_block(instrument: str) -> None:
    """A raw-DN navigator block states its own DN zero point.

    Without one the navigator falls back to the generic block's value silently.
    """
    navigator = resolve_sim_inst_config(DEFAULT_CONFIG, instrument)['noise']
    assert 'star_flux_dn_per_s_vmag0' in navigator


@pytest.mark.parametrize('instrument', _raw_dn_navigator_instruments())
def test_the_navigator_zero_point_matches_the_simulators(instrument: str) -> None:
    """The navigator's DN zero point is the simulator's, within its own rounding.

    The navigator states its value separately, so the information boundary holds;
    this test detects the two drifting apart by more than half the place value of the
    last digit the navigator's config file writes, so a difference the navigator's
    rounding explains passes.  A CCD's zero point is in electrons, converted through
    the gain that a scene with no detector settings renders with; a vidicon's is
    already DN.
    """
    zero_point, domain = resolve_star_flux_zero_point(instrument)
    if domain == 'electrons':
        zero_point /= resolve_detector_params({'instrument': instrument}).gain_e_per_dn
    # A block without the key fails here and in the test above, which says why.
    stated = resolve_sim_inst_config(DEFAULT_CONFIG, instrument)['noise'][
        'star_flux_dn_per_s_vmag0'
    ]
    step = _written_step(instrument, 'noise', 'star_flux_dn_per_s_vmag0')
    assert zero_point == pytest.approx(stated, rel=0.0, abs=step / 2.0)


@pytest.mark.parametrize('instrument', _raw_dn_navigator_instruments())
def test_the_navigator_states_values_for_each_binned_mode(instrument: str) -> None:
    """A raw-DN navigator block has a ``by_binning`` entry for each binned mode offered.

    Without one the star limit of a frame in that mode reads the unbinned values.
    """
    navigator = resolve_sim_inst_config(DEFAULT_CONFIG, instrument)['noise']
    offered = {binning for binning in cataloged_binnings(instrument) if binning != 1}
    assert set(navigator.get('by_binning', {})) == offered


_BINNED_RAW_DN_MODES = [
    (instrument, binning)
    for instrument in _raw_dn_navigator_instruments()
    for binning in cataloged_binnings(instrument)
    if binning != 1
]


def _binned_mode_detector(instrument: str, binning: int) -> Any:
    """The detector parameters a scene in this mode renders with, catalog values on."""
    scene = {
        'instrument': instrument,
        'size_v': 32,
        'size_u': 32,
        'detector': {'binning': binning},
        'artifacts': {'instrument_defaults': True},
    }
    return resolve_detector_params(scene)


@pytest.mark.parametrize(('instrument', 'binning'), _BINNED_RAW_DN_MODES)
def test_the_navigator_binned_zero_point_matches_the_simulators(
    instrument: str, binning: int
) -> None:
    """A binned mode's DN zero point is the simulator's, within the navigator's rounding.

    The simulator's electron zero point is converted through the mode's gain.
    """
    zero_point, domain = resolve_star_flux_zero_point(instrument)
    assert domain == 'electrons'
    zero_point /= _binned_mode_detector(instrument, binning).gain_e_per_dn
    mode = resolve_sim_inst_config(DEFAULT_CONFIG, instrument)['noise']['by_binning'][binning]
    step = _written_step(instrument, 'noise', 'by_binning', binning, 'star_flux_dn_per_s_vmag0')
    assert zero_point == pytest.approx(mode['star_flux_dn_per_s_vmag0'], rel=0.0, abs=step / 2.0)


@pytest.mark.parametrize(('instrument', 'binning'), _BINNED_RAW_DN_MODES)
def test_the_navigator_binned_read_noise_matches_the_simulators(
    instrument: str, binning: int
) -> None:
    """A binned mode's read noise in DN is the simulator's, within the navigator's rounding.

    The simulator's read noise for the mode, in electrons, is converted through the
    mode's gain.
    """
    detector = _binned_mode_detector(instrument, binning)
    read_noise_dn = detector.read_noise_e / detector.gain_e_per_dn
    mode = resolve_sim_inst_config(DEFAULT_CONFIG, instrument)['noise']['by_binning'][binning]
    step = _written_step(instrument, 'noise', 'by_binning', binning, 'read_noise_dn')
    assert read_noise_dn == pytest.approx(mode['read_noise_dn'], rel=0.0, abs=step / 2.0)


def _binned_scene(instrument_config: dict[str, Any], *, instrument: str = 'coiss_nac') -> Any:
    """A scene summed 2x2 with the given ``instrument_config`` overrides."""
    return {
        'instrument': instrument,
        'size_v': 32,
        'size_u': 32,
        'random_seed': 1,
        'detector': {'binning': 2},
        'instrument_config': instrument_config,
    }


@pytest.mark.parametrize(
    ('table', 'message'),
    [
        ([2], r'instrument_config\.noise\.by_binning must be a mapping keyed by binning'),
        (
            {2: {'star_flux_dn_per_s_vmag0': -3.33e5}},
            r'instrument_config\.noise\.by_binning\.2\.star_flux_dn_per_s_vmag0 must be '
            r'finite and positive',
        ),
        (
            {'x2': {'read_noise_dn': 1.0, 'star_flux_dn_per_s_vmag0': 1.0}},
            r'instrument_config\.noise\.by_binning: a key must be a binning factor of at '
            r"least 2; got 'x2'",
        ),
    ],
    ids=['not_a_mapping', 'negative_zero_point', 'not_a_binning'],
)
def test_a_scene_with_a_malformed_by_binning_override_is_refused(table: Any, message: str) -> None:
    """A scene's binned-mode table is checked as the configuration loader checks it."""
    scene = _binned_scene({'noise': {'by_binning': table}})
    with pytest.raises(SimSceneValidationError, match=message):
        validate_scene(scene, source='by_binning_probe')


def test_a_binning_key_written_as_digits_names_that_mode() -> None:
    """A scene that writes a binned mode's key as ``'2'`` overrides mode 2.

    A scene read through JSON carries every key as a string, so both spellings must
    reach the same entry.
    """
    overrides = {'noise': {'by_binning': {'2': {'read_noise_dn': 3.0}}}}
    noise = resolve_sim_inst_config(DEFAULT_CONFIG, 'coiss_nac', overrides)['noise']
    assert noise['by_binning'][2]['read_noise_dn'] == 3.0


@pytest.mark.parametrize('key', ['read_noise_dn', 'star_flux_dn_per_s_vmag0'])
def test_a_binned_scene_that_sets_only_an_unbinned_value_is_refused(key: str) -> None:
    """A summed frame reads its mode's value, so setting only the unbinned one fails."""
    scene = _binned_scene({'noise': {key: 3.0}})
    with pytest.raises(
        SimSceneValidationError,
        match=(
            rf'instrument_config\.noise\.{key} is set, but a frame binned by 2 reads '
            rf'instrument_config\.noise\.by_binning\.2\.{key} instead\. Set '
            rf'instrument_config\.noise\.by_binning\.2\.{key} too\.'
        ),
    ):
        validate_scene(scene, source='by_binning_probe')


@pytest.mark.parametrize(
    ('instrument', 'noise'),
    [
        ('coiss_nac', {'read_noise_dn': 3.0, 'by_binning': {2: {'read_noise_dn': 0.75}}}),
        ('coiss_calib_nac', {'read_noise_dn': 3.0}),
    ],
    ids=['mode_value_set', 'no_mode_entry'],
)
def test_a_binned_scene_may_set_an_unbinned_value_its_frame_reads(
    instrument: str, noise: dict[str, Any]
) -> None:
    """A binned scene may set an unbinned value when its mode's value is set as well.

    A camera whose block has no entry for the mode reads the unbinned value, so it
    needs no mode value.
    """
    validate_scene(_binned_scene({'noise': noise}, instrument=instrument), source='probe')
