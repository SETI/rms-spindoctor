"""The navigator's star values for the simulated cameras.

Each raw-DN navigator block states its own DN zero point and, for every binned mode its
camera offers, the mode's read noise and zero point.  Each value agrees with the
simulator's camera within the digits the navigator's config file writes it with.

A scene's ``instrument_config.noise`` overrides must be a mapping, and their binned-mode
table holds no null.  The scene validator checks every value the faint-star limit reads
from the block a scene's overrides resolve to, unbinned and binned, at every binning and
on every block: a read noise is a finite number of at least 0, and a zero point and the
PSF width are finite and positive.  The table spells each binning once, holds only
entries some frame of the instrument reads, and each of its entries states exactly a read
noise and a zero point.  A binned scene can't set an unbinned value that its frame reads
from the table instead, and the limit of a binned frame reads the value the scene pins
for its mode.
"""

import json
import math
import re
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from ruamel.yaml import YAML
from ruamel.yaml.nodes import MappingNode, Node

import spindoctor
from spindoctor.config import DEFAULT_CONFIG
from spindoctor.obs.obs_inst_sim import ObsSim
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
from spindoctor.sim.scene import validate_scene, validate_sim_params
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


def _scene(
    instrument_config: Any, *, instrument: str = 'coiss_nac', binning: int = 2
) -> dict[str, Any]:
    """A scene with the given ``instrument_config`` overrides, summed 2x2 by default."""
    scene: dict[str, Any] = {
        'instrument': instrument,
        'size_v': 32,
        'size_u': 32,
        'random_seed': 1,
        'instrument_config': instrument_config,
    }
    if binning != 1:
        scene['detector'] = {'binning': binning}
    return scene


@pytest.mark.parametrize(
    ('table', 'message'),
    [
        ([2], r'instrument_config\.noise\.by_binning must be a mapping keyed by binning'),
        (
            {2: {'star_flux_dn_per_s_vmag0': -3.33e5}},
            r'instrument_config\.noise\.by_binning\.2\.star_flux_dn_per_s_vmag0 must be '
            r'a finite number greater than 0; got -333000\.0$',
        ),
        (
            {'x2': {'read_noise_dn': 1.0, 'star_flux_dn_per_s_vmag0': 1.0}},
            r'instrument_config\.noise\.by_binning: a key must be an integer binning factor '
            r"of at least 2; got 'x2'",
        ),
        (
            {2: {'gain_e_per_dn': 28.0}},
            r'instrument_config\.noise\.by_binning\.2 must state exactly read_noise_dn, '
            r'star_flux_dn_per_s_vmag0; got ',
        ),
        (
            {2: 5.0},
            r'instrument_config\.noise\.by_binning\.2 must state exactly read_noise_dn, '
            r'star_flux_dn_per_s_vmag0; got 5\.0$',
        ),
    ],
    ids=[
        'not_a_mapping',
        'negative_zero_point',
        'not_a_binning',
        'unknown_value',
        'entry_not_a_mapping',
    ],
)
def test_a_scene_with_a_malformed_by_binning_override_is_refused(table: Any, message: str) -> None:
    """A scene's binned-mode table must map binning factors to the two usable values."""
    scene = _scene({'noise': {'by_binning': table}})
    with pytest.raises(SimSceneValidationError, match=message):
        validate_scene(scene, source='by_binning_probe')


def test_an_unbinned_scene_with_a_malformed_by_binning_override_is_refused() -> None:
    """An unbinned frame never reads the table, but the table is checked all the same.

    The entry's negative read noise is refused under the rule for an unbinned one.
    """
    scene = _scene({'noise': {'by_binning': {2: {'read_noise_dn': -1.0}}}}, binning=1)
    with pytest.raises(
        SimSceneValidationError,
        match=r'instrument_config\.noise\.by_binning\.2\.read_noise_dn must be a finite '
        r'number of at least 0; got -1\.0$',
    ):
        validate_sim_params(scene, source='by_binning_probe')


@pytest.mark.parametrize(
    ('value', 'shown'),
    [(math.inf, 'inf'), ('abc', "'abc'"), (True, 'True')],
    ids=['infinite', 'text', 'boolean'],
)
def test_a_binned_read_noise_that_is_not_a_finite_number_is_refused(value: Any, shown: str) -> None:
    """A binned mode's read noise must be a finite number, and a boolean isn't one."""
    scene = _scene({'noise': {'by_binning': {2: {'read_noise_dn': value}}}})
    with pytest.raises(
        SimSceneValidationError,
        match=r'instrument_config\.noise\.by_binning\.2\.read_noise_dn must be a finite '
        rf'number of at least 0; got {re.escape(shown)}$',
    ):
        validate_sim_params(scene, source='by_binning_probe')


def test_an_unbinned_negative_read_noise_is_refused() -> None:
    """The unbinned read noise follows the same rule as a binned mode's."""
    scene = _scene({'noise': {'read_noise_dn': -1.0}}, binning=1)
    with pytest.raises(
        SimSceneValidationError,
        match=r'instrument_config\.noise\.read_noise_dn must be a finite number of at '
        r'least 0; got -1\.0$',
    ):
        validate_sim_params(scene, source='by_binning_probe')


def test_an_unbinned_zero_point_of_zero_is_refused() -> None:
    """A zero point of 0 leaves no star above the noise, so it must be positive."""
    scene = _scene({'noise': {'star_flux_dn_per_s_vmag0': 0.0}}, binning=1)
    with pytest.raises(
        SimSceneValidationError,
        match=r'instrument_config\.noise\.star_flux_dn_per_s_vmag0 must be a finite number '
        r'greater than 0; got 0\.0$',
    ):
        validate_sim_params(scene, source='by_binning_probe')


def test_a_star_psf_sigma_of_zero_is_refused() -> None:
    """The faint-star limit divides by the PSF width, so it must be positive."""
    scene = _scene({'star_psf_sigma': 0.0}, binning=1)
    with pytest.raises(
        SimSceneValidationError,
        match=r'instrument_config\.star_psf_sigma must be a finite number greater than 0; '
        r'got 0\.0$',
    ):
        validate_sim_params(scene, source='by_binning_probe')


def test_a_generic_scene_with_a_malformed_by_binning_override_is_refused() -> None:
    """The generic block, which has no table of its own, is checked too."""
    scene = _scene({'noise': {'by_binning': [2]}}, instrument=GENERIC_INSTRUMENT, binning=1)
    with pytest.raises(
        SimSceneValidationError,
        match=r'instrument_config\.noise\.by_binning must be a mapping keyed by binning',
    ):
        validate_sim_params(scene, source='by_binning_probe')


@pytest.mark.parametrize('noise', [[2], 'read_noise_dn'], ids=['list', 'string'])
def test_a_noise_override_that_is_not_a_mapping_is_refused(noise: Any) -> None:
    """``instrument_config.noise`` must be a mapping, or the frame's star limit can't read it."""
    with pytest.raises(
        SimSceneValidationError,
        match=r'instrument_config\.noise must be a mapping when present',
    ):
        validate_sim_params(_scene({'noise': noise}), source='by_binning_probe')


@pytest.mark.parametrize(
    ('instrument_config', 'path'),
    [
        ({'noise': None}, r'instrument_config\.noise'),
        ({'noise': {'by_binning': None}}, r'instrument_config\.noise\.by_binning'),
        (
            {'noise': {'by_binning': {2: {'read_noise_dn': None}}}},
            r'instrument_config\.noise\.by_binning\.2\.read_noise_dn',
        ),
    ],
    ids=['noise', 'table', 'mode_value'],
)
def test_a_null_noise_block_or_binned_value_is_refused(instrument_config: Any, path: str) -> None:
    """A null noise block or binned-mode value is refused with the way to take the default."""
    with pytest.raises(
        SimSceneValidationError,
        match=rf'{path} is null; leave the key out to take its default',
    ):
        validate_sim_params(_scene(instrument_config), source='by_binning_probe')


def test_a_binning_key_written_as_digits_names_that_mode() -> None:
    """A scene that writes a binned mode's key as ``'2'`` overrides mode 2.

    A scene read through JSON carries every key as a string, so both spellings must
    reach the same entry.
    """
    overrides = {'noise': {'by_binning': {'2': {'read_noise_dn': 3.0}}}}
    noise = resolve_sim_inst_config(DEFAULT_CONFIG, 'coiss_nac', overrides)['noise']
    assert noise['by_binning'][2]['read_noise_dn'] == 3.0


def test_a_binning_written_both_as_an_integer_and_as_digits_is_refused() -> None:
    """Keys ``2`` and ``'2'`` name one mode, so a table holding both sets it twice."""
    table = {2: {'read_noise_dn': 3.0}, '2': {'read_noise_dn': 1.0}}
    with pytest.raises(
        SimSceneValidationError,
        match=r"instrument_config\.noise\.by_binning holds both 2 and '2', which name the "
        r'same binning; keep one',
    ):
        validate_sim_params(_scene({'noise': {'by_binning': table}}), source='by_binning_probe')


@pytest.mark.parametrize(
    ('through_json', 'shown'),
    [(False, '2.0'), (True, "'2.0'")],
    ids=['as_written', 'through_json'],
)
def test_a_binning_key_written_as_a_float_is_refused(through_json: bool, shown: str) -> None:
    """A key ``2.0`` is refused whether the scene is read as written or through JSON.

    As written, ``2.0`` equals and hashes as the packaged table's key ``2``, so it must
    be refused before the scene's table is merged over the packaged one.
    """
    table = {2.0: {'read_noise_dn': 1.0, 'star_flux_dn_per_s_vmag0': 1.0e6}}
    scene = _scene({'noise': {'by_binning': table}})
    if through_json:
        scene = json.loads(json.dumps(scene))
    with pytest.raises(
        SimSceneValidationError,
        match=r'instrument_config\.noise\.by_binning: a key must be an integer binning factor '
        rf'of at least 2; got {re.escape(shown)}$',
    ):
        validate_sim_params(scene, source='by_binning_probe')


@pytest.mark.parametrize(
    ('key', 'through_json', 'shown'),
    [
        (-1, False, '-1'),
        (-1, True, "'-1'"),
        (0, False, '0'),
        (0, True, "'0'"),
        (1, False, '1'),
        (1, True, "'1'"),
    ],
    ids=[
        'minus_one_as_written',
        'minus_one_through_json',
        'zero_as_written',
        'zero_through_json',
        'one_as_written',
        'one_through_json',
    ],
)
def test_a_binning_key_below_2_is_refused(key: int, through_json: bool, shown: str) -> None:
    """A key below 2 gets the same refusal whether read as written or through JSON."""
    table = {key: {'read_noise_dn': 1.0, 'star_flux_dn_per_s_vmag0': 1.0e6}}
    scene = _scene({'noise': {'by_binning': table}})
    if through_json:
        scene = json.loads(json.dumps(scene))
    with pytest.raises(
        SimSceneValidationError,
        match=r'instrument_config\.noise\.by_binning: a key must be an integer binning factor '
        rf'of at least 2; got {re.escape(shown)}$',
    ):
        validate_sim_params(scene, source='by_binning_probe')


def test_a_float_key_beside_its_digits_is_refused_for_its_form() -> None:
    """A table holding ``'2'`` and ``2.0`` is refused for the float key.

    Every key's form is checked before any two keys are compared, so the float isn't
    reported as a second spelling of binning 2.
    """
    entry = {'read_noise_dn': 1.0, 'star_flux_dn_per_s_vmag0': 1.0e6}
    table = {'2': dict(entry), 2.0: dict(entry)}
    with pytest.raises(
        SimSceneValidationError,
        match=r'instrument_config\.noise\.by_binning: a key must be an integer binning factor '
        r'of at least 2; got 2\.0$',
    ):
        validate_sim_params(_scene({'noise': {'by_binning': table}}), source='by_binning_probe')


def test_a_binning_key_with_a_leading_zero_is_refused() -> None:
    """A key ``'02'`` would be a second spelling of binning 2."""
    table = {'02': {'read_noise_dn': 3.0}}
    with pytest.raises(
        SimSceneValidationError,
        match=r"instrument_config\.noise\.by_binning: the key '02' has a leading zero; "
        r'write it as 2',
    ):
        validate_sim_params(_scene({'noise': {'by_binning': table}}), source='by_binning_probe')


def test_an_entry_for_a_binning_the_instrument_does_not_offer_is_refused() -> None:
    """No frame of the instrument reads an entry for a binning its readout modes lack."""
    table = {2: {'read_noise_dn': 1.0, 'star_flux_dn_per_s_vmag0': 1.0e6}}
    scene = _scene({'noise': {'by_binning': table}}, instrument='nhlorri', binning=1)
    with pytest.raises(
        SimSceneValidationError,
        match=r"instrument_config\.noise\.by_binning\.2 is never read: instrument 'nhlorri' "
        r'offers binnings 1, 4\. Leave the entry out\.',
    ):
        validate_sim_params(scene, source='by_binning_probe')


def test_a_by_binning_override_on_a_calibrated_unit_camera_is_refused() -> None:
    """A calibrated-unit camera's star limit is a constant that reads no noise value.

    The camera offers 2x2 summing, but no frame's limit reads the table, at any binning.
    """
    table = {2: {'read_noise_dn': 1.0, 'star_flux_dn_per_s_vmag0': 1.0e6}}
    scene = _scene({'noise': {'by_binning': table}}, instrument='coiss_calib_nac')
    with pytest.raises(
        SimSceneValidationError,
        match=r'instrument_config\.noise\.by_binning is never read: instrument '
        r"'coiss_calib_nac' renders calibrated_if frames, whose star limit reads no noise "
        r'value\. Leave the table out\.',
    ):
        validate_sim_params(scene, source='by_binning_probe')


@pytest.mark.parametrize('key', ['read_noise_dn', 'star_flux_dn_per_s_vmag0'])
def test_a_binned_scene_that_sets_only_an_unbinned_value_is_refused(key: str) -> None:
    """A summed frame reads its mode's value, so setting only the unbinned one fails."""
    scene = _scene({'noise': {key: 3.0}})
    with pytest.raises(
        SimSceneValidationError,
        match=(
            rf'instrument_config\.noise\.{key} is set, but a frame binned by 2 reads '
            rf'instrument_config\.noise\.by_binning\.2\.{key} instead\. Set '
            rf'instrument_config\.noise\.by_binning\.2\.{key} too\.'
        ),
    ):
        validate_scene(scene, source='by_binning_probe')


def _pinned_mode_scene(mode_read_noise_dn: float) -> dict[str, Any]:
    """A 2x2 NAC scene that pins the unbinned read noise and the 2x2 mode's."""
    noise = {
        'read_noise_dn': 3.0,
        'by_binning': {2: {'read_noise_dn': mode_read_noise_dn}},
    }
    return _scene({'noise': noise})


def test_a_binned_scene_that_sets_the_mode_value_too_is_accepted() -> None:
    """A binned scene may set an unbinned value when it sets its mode's value as well."""
    params = _pinned_mode_scene(2.0)
    assert validate_sim_params(params, source='by_binning_probe') is params


def test_a_binned_read_noise_of_zero_is_accepted() -> None:
    """A mode's read noise of 0 is an ideal detector, which the star limit can read."""
    params = _pinned_mode_scene(0.0)
    assert validate_sim_params(params, source='by_binning_probe') is params


def test_the_binned_star_limit_floors_a_zero_read_noise_at_1_dn() -> None:
    """The limit of a 2x2 frame whose mode has no read noise is the limit at 1 DN."""
    ideal = ObsSim.from_file('/tmp/pinned_mode.yaml', sim_params=_pinned_mode_scene(0.0))
    floor = ObsSim.from_file('/tmp/pinned_mode.yaml', sim_params=_pinned_mode_scene(1.0))
    assert ideal.star_max_usable_vmag() == pytest.approx(
        floor.star_max_usable_vmag(), rel=0.0, abs=1e-12
    )


def test_the_binned_star_limit_reads_the_pinned_mode_value() -> None:
    """The limit of a frame summed 2x2 reads the read noise the scene pins for 2x2.

    Both read noises sit above the limit's 1 DN floor, so doubling the mode's value
    moves the limit by 2.5 log10(2) brighter; the unbinned value, the same in both
    scenes, would move it by nothing.
    """
    low = ObsSim.from_file('/tmp/pinned_mode.yaml', sim_params=_pinned_mode_scene(2.0))
    high = ObsSim.from_file('/tmp/pinned_mode.yaml', sim_params=_pinned_mode_scene(4.0))
    shift = high.star_max_usable_vmag() - low.star_max_usable_vmag()
    assert shift == pytest.approx(-2.5 * math.log10(2.0), rel=0.0, abs=1e-12)


def test_a_binned_calibrated_unit_scene_may_set_an_unbinned_noise_value() -> None:
    """A binned scene on a calibrated-unit camera may set an unbinned noise value alone.

    The camera's block has no binned-mode table, so the requirement to set the mode's
    value as well doesn't apply; its star limit is a constant that reads neither value.
    Every raw-DN block has an entry for each binned mode its camera offers, and a
    scene's overrides can add to the table but not remove an entry, so no raw-DN scene
    reaches this case.
    """
    params = _scene({'noise': {'read_noise_dn': 3.0}}, instrument='coiss_calib_nac')
    assert validate_sim_params(params, source='by_binning_probe') is params
