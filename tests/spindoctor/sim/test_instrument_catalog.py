"""The simulated-instrument catalog: checked YAML entries, keyed by instrument.

Each violation the loader refuses is one case, with the file, the entry, and the key
path in its message.  A camera added only as an entry in a copy of the packaged
directory validates, resolves, and renders.  The packaged catalog's entries point at
navigator blocks that exist and set the same units as those blocks, and each
navigator star zero point agrees with the simulator's.  An entry fails when its
navigator block is resolved if their units differ, or if the block doesn't set
data_units.  The detector chain renders in the entry's units.
"""

from collections.abc import Mapping
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from ruamel.yaml import YAML
from tests.spindoctor.sim.fictional_camera import (
    FICTIONAL_CAMERA,
    FICTIONAL_CAMERA_NAME,
    catalog_dir_with,
    keyed,
)

from spindoctor.config import DEFAULT_CONFIG
from spindoctor.sim import instrument_catalog, instrument_navigator
from spindoctor.sim.forward.detector.params import DetectorParamError, resolve_detector_params
from spindoctor.sim.forward.detector.sqrt_lut import SqrtLut, TabulatedLut
from spindoctor.sim.instrument_catalog import (
    GENERIC_INSTRUMENT,
    INSTRUMENT_CONFIG_DIR,
    InstrumentConfigError,
    load_instrument_catalog,
    sim_instrument_names,
)
from spindoctor.sim.instrument_checks import _parametric_decoded_dn
from spindoctor.sim.instrument_navigator import resolve_sim_inst_config
from spindoctor.sim.instrument_resolvers import (
    cataloged_binnings,
    resolve_data_units,
    resolve_detector_defaults,
    resolve_star_flux_zero_point,
)
from spindoctor.sim.render import render_combined_model
from spindoctor.sim.scene import validate_sim_params
from spindoctor.sim.scene_schema import SimSceneValidationError

# A complete camera entry, written out so each case changes one thing in it.
_CAMERA = """\
data_units: raw_dn
navigator:
  section: sim
detector:
  detector_model: ccd
  full_well_e: 5000.0
  exposure_ref_sec: 1.0
  gain_e_per_dn_by_state: {0: 2.0}
  default_gain_state: 0
  read_noise_e: 3.0
  bias_dn: 10.0
  dark_current_e_per_sec: 0.0
  hot_pixel_fraction: 0.0
  hot_pixel_amplitude_e: 0.0
  hot_pixel_column_factor: 0.0
  cosmic_ray_events_per_pixel_per_sec: 0.0
  banding_amplitude_e: 0.0
  banding_period_px: 64.0
  bias_pedestal_sigma_dn: 0.0
  bias_row_gradient_dn: 0.0
  bias_col_gradient_dn: 0.0
  bloom_length: 0
  quantization: exact
readout_modes:
  1: {}
star_flux:
  e_per_s_vmag0: 1.0e+5
sky_pixel_scale_arcsec: 2.5
psf:
  truncation_px: 16
artifact_modes:
  offered: [missing_lines]
"""

_OFFERED = 'offered: [missing_lines]'
_VIDICON_BLOCK = (
    '  vidicon: {read_noise_line_dn: 1.0, read_noise_pixel_dn: 1.0, '
    'coherent_amplitude_dn: 1.0, coherent_period_px: 8.0}\n'
)


def _cam(entry: str) -> str:
    """A config file holding ``entry`` as the instrument ``cam``."""
    return keyed('cam', entry)


def _with(old: str, new: str) -> str:
    """The test camera with one passage replaced."""
    assert old in _CAMERA
    return _CAMERA.replace(old, new)


def _generic_without(line: str) -> str:
    """The packaged generic entry with one line removed."""
    text = (INSTRUMENT_CONFIG_DIR / 'generic.yaml').read_text()
    assert text.count(line) == 1
    return text.replace(line, '')


# Each case: the files written over the packaged directory, the expected message,
# and the case's id.
_REFUSED: list[tuple[dict[str, str], str, str]] = [
    (
        {'cam.yaml': _cam(_CAMERA + 'sensor: {}\n')},
        r"cam\.yaml: cam: unknown keys: \['sensor'\]",
        'unknown-key',
    ),
    (
        {'cam.yaml': _cam(_with('sky_pixel_scale_arcsec: 2.5\n', ''))},
        r"cam\.yaml: cam: missing required keys: \['sky_pixel_scale_arcsec'\]",
        'missing-required-key',
    ),
    (
        {'cam.yaml': _cam(_with('data_units: raw_dn', 'data_units: reflectance'))},
        r"cam\.yaml: cam: data_units: 'reflectance' is not one of \['calibrated_if', 'raw_dn'\]",
        'unknown-data-units',
    ),
    (
        {'generic.yaml': _generic_without('  data_units: raw_dn\n')},
        r"generic\.yaml: generic: missing required keys: \['data_units'\]",
        'generic-without-data-units',
    ),
    (
        {'cam.yaml': _cam(_with('bloom_length: 0', 'bloom_length: 0.5'))},
        r'cam\.yaml: cam: detector\.bloom_length: expected an integer, got float 0\.5',
        'wrong-type',
    ),
    (
        {'cam.yaml': _cam(_with('full_well_e: 5000.0', 'full_well_e: .5e4'))},
        r"detector\.full_well_e: expected a number, got str '\.5e4' \(a number written",
        'number-read-as-a-string',
    ),
    (
        {'cam.yaml': _cam(_with('full_well_e: 5000.0', 'full_well_e: .nan'))},
        r'cam\.yaml: cam: detector\.full_well_e: expected a finite number, got float nan',
        'non-finite-number',
    ),
    (
        {'cam.yaml': _cam(_with('{0: 2.0}', '{0: 0.0}'))},
        r'cam\.yaml: cam: detector\.gain_e_per_dn_by_state\.0: expected a positive number, '
        r'got 0\.0',
        'non-positive-gain',
    ),
    (
        {'cam.yaml': _cam(_with('hot_pixel_fraction: 0.0', 'hot_pixel_fraction: 5.0'))},
        r'cam\.yaml: cam: detector\.hot_pixel_fraction: expected a number in \[0, 1\], got 5\.0',
        'fraction-above-one',
    ),
    (
        {'cam.yaml': _cam(_with('  1: {}', '  1: {}\n  0: {}'))},
        r'cam\.yaml: cam: readout_modes\.0: expected a positive integer, got 0',
        'binning-below-one',
    ),
    (
        {'cam.yaml': _cam('navigator: {section: sim\n')},
        r'cam\.yaml: not valid YAML',
        'yaml-syntax-error',
    ),
    (
        {'cam.yaml': _cam(_with('read_noise_e: 3.0', 'read_noise_e: 3.0\n  read_noise_e: 9.0'))},
        r'(?s)cam\.yaml: not valid YAML: .*duplicate key',
        'duplicate-key',
    ),
    (
        {'cam.yml': _CAMERA},
        r"\['cam\.yml'\] are not instrument config files",
        'file-not-named-yaml',
    ),
    (
        {'cam.yaml': _cam(_with('read_noise_e: 3.0', 'read_noise_e: null'))},
        r'cam\.yaml: cam: detector\.read_noise_e: null removes an inherited key, and this entry '
        r'sets no base',
        'null-without-a-base',
    ),
    (
        {'cam.yaml': _cam('base: generic\ndistortion_residual: null\n')},
        r'cam\.yaml: cam: distortion_residual: null removes an inherited key, and the base has no '
        r'such key',
        'null-for-a-key-the-base-lacks',
    ),
    (
        {'cam.yaml': _cam('base: coiss_nac\nsky_pixel_scale_arcsec: null\n')},
        r"cam\.yaml: cam: missing required keys: \['sky_pixel_scale_arcsec'\]",
        'null-removing-a-required-key',
    ),
    (
        {
            'cam.yaml': _cam(
                _CAMERA + '  shape_defaults:\n    missing_lines: {expected_events: 5.0}\n'
            )
        },
        r"shape_defaults\.missing_lines\.expected_events: 'expected_events' turns the mode on",
        'activation-key-in-shape-defaults',
    ),
    (
        {
            'cam.yaml': _cam(
                _with(_OFFERED, 'offered: [missing_lines, hot_pixels]')
                + '  shape_defaults:\n    hot_pixels: {amplitude_e: 5.0}\n'
            )
        },
        r"shape_defaults\.hot_pixels: a shape default for 'hot_pixels' does nothing",
        'shape-default-the-renderer-ignores',
    ),
    (
        {
            'cam.yaml': _cam(
                _with(_OFFERED, 'offered: [edited_frame]')
                + '  shape_defaults:\n    edited_frame: {band_width_px: 0}\n'
            )
        },
        r'shape_defaults\.edited_frame\.band_width_px: expected a positive_int value, got int 0',
        'shape-default-out-of-range',
    ),
    (
        {
            'cam.yaml': _cam(
                _with(_OFFERED, 'offered: [frame_transfer_smear]')
                + '  instrument_defaults:\n    frame_transfer_smear: {t_scrub_sec: 0.01}\n'
            )
        },
        r'instrument_defaults\.frame_transfer_smear: an instrument_defaults entry turns the '
        r"mode on, so it sets a positive 'probability'",
        'instrument-default-without-activation',
    ),
    (
        {'cam.yaml': _cam(_CAMERA + '  shape_defaults:\n    bloom: {bloom_length: 2}\n')},
        r"cam\.yaml: cam: artifact_modes\.shape_defaults\.bloom: 'bloom' is not in "
        r'artifact_modes\.offered',
        'mode-not-offered',
    ),
    (
        {
            'cam.yaml': _cam(
                _CAMERA + 'filters:\n  throughput_e_per_s_per_if: {CLR: 1.0}\n  clear: CL1\n'
            )
        },
        r"cam\.yaml: cam: filters\.clear: 'CL1' is not in throughput_e_per_s_per_if",
        'filter-not-cataloged',
    ),
    (
        {
            'cam.yaml': _cam(
                _CAMERA + 'filters:\n  throughput_e_per_s_per_if: {Red: 1.0}\n  clear: Red\n'
            )
        },
        r"filters\.throughput_e_per_s_per_if\.Red: 'Red' is not in canonical form; write 'RED'",
        'filter-not-canonical',
    ),
    (
        {
            'cam.yaml': _cam(
                _CAMERA
                + 'filters:\n  throughput_e_per_s_per_if: {CLR: 1.0}\n  clear: CLR\n'
                + '  polarizer_clear: {p0: CLR}\n'
            )
        },
        r"filters\.polarizer_clear\.p0: 'p0' is not in canonical form; write 'P0'",
        'polarizer-not-canonical',
    ),
    (
        {
            'cam.yaml': _cam(
                _CAMERA
                + 'filters:\n  throughput_e_per_s_per_if: {CL1+GRN: 1.0, P0+GRN: 1.0}\n'
                + '  clear: CL1+GRN\n  polarizer_clear: {P0: CL1}\n'
            )
        },
        r"filters\.throughput_e_per_s_per_if\.P0\+GRN: the polarizer 'P0' folds onto 'CL1', "
        r"so a scene that sets 'P0\+GRN' reads 'CL1\+GRN'",
        'filter-keyed-by-a-polarizer',
    ),
    (
        {
            'cam.yaml': _cam(
                'base: vgiss\nfilters:\n  throughput_e_per_s_per_if: {CLEAR: 1.0}\n'
                + '  clear: CLEAR\n'
            )
        },
        r'cam\.yaml: cam: filters: a filter sets the CCD signal scale; the vidicon DN chain has no '
        r'reference exposure',
        'filters-on-a-vidicon',
    ),
    (
        {'cam.yaml': _cam(_with('default_gain_state: 0', 'default_gain_state: 3'))},
        r'cam\.yaml: cam: detector\.default_gain_state: state 3 is not in '
        r'gain_e_per_dn_by_state \[0\]',
        'gain-state-not-cataloged',
    ),
    (
        {'cam.yaml': _cam(_with('{0: 2.0}', '{0: 2.0, 2: 1.0}'))},
        r'detector\.gain_e_per_dn_by_state: the gain states \[0, 2\] must be contiguous '
        r'integers from 0',
        'gain-states-not-contiguous',
    ),
    (
        {
            'cam.yaml': _cam(
                _with('  quantization: exact\n', '  quantization: exact\n' + _VIDICON_BLOCK)
            )
        },
        r'cam\.yaml: cam: detector\.vidicon: a ccd detector has no vidicon chain to configure',
        'vidicon-chain-on-a-ccd',
    ),
    (
        {'cam.yaml': _cam(_with('  1: {}', '  1: {}\n  2: {detector_model: vidicon}'))},
        r'readout_modes\.2\.detector_model: a readout mode bins the same detector',
        'readout-mode-changing-the-model',
    ),
    (
        {
            'cam.yaml': _cam(
                _with('  1: {}', '  1: {}\n  2: {gain_e_per_dn_by_state: {0: 2.0, 1: 1.0}}')
            )
        },
        r'readout_modes\.2\.gain_e_per_dn_by_state: the states \[0, 1\] differ',
        'readout-mode-changing-the-gain-states',
    ),
    (
        {'cam.yaml': _cam('base: vgiss\nreadout_modes: {2: {}}\n')},
        r'cam\.yaml: cam: readout_modes\.2: the vidicon DN chain has no binned readout',
        'binned-vidicon',
    ),
    (
        {'cam.yaml': _cam(_with('  1: {}', '  2: {}'))},
        r'cam\.yaml: cam: readout_modes: binning 1 \(the unbinned readout\) is not defined',
        'unbinned-readout-missing',
    ),
    (
        {
            'cam.yaml': _cam(
                _CAMERA
                + 'downlink_tables:\n  default: other\n  tables:\n'
                + '    main: {decoded_dn: [0.0, 1.0]}\n'
            )
        },
        r"downlink_tables\.default: 'other' is not one of the tables \['main'\]",
        'downlink-default-not-listed',
    ),
    (
        {
            'cam.yaml': _cam(
                _CAMERA
                + 'downlink_tables:\n  default: main\n  tables:\n'
                + '    main: {decoded_dn: [0.0, 1.0], codes: 2}\n'
            )
        },
        r'downlink_tables\.tables\.main: state either decoded_dn alone',
        'downlink-table-mixing-forms',
    ),
    (
        {
            'cam.yaml': _cam(
                _CAMERA
                + 'downlink_tables:\n  default: main\n  tables:\n'
                + '    main: {codes: 1, linear_dn: 1.0, quadratic_dn: 0.0, whole_dn: true}\n'
            )
        },
        r'downlink_tables\.tables\.main\.codes: a downlink table sends at least 2 codes, got 1',
        'downlink-table-with-one-code',
    ),
    (
        {
            'cam.yaml': _cam(
                _CAMERA
                + 'downlink_tables:\n  default: main\n  tables:\n'
                + '    main: {decoded_dn: [5.0, 5.0, 5.0]}\n'
            )
        },
        r'downlink_tables\.tables\.main\.decoded_dn: a downlink table decodes to at least 2 '
        r'distinct values, got 1',
        'downlink-table-with-one-distinct-value',
    ),
    (
        {
            'cam.yaml': _cam(
                _CAMERA
                + 'downlink_tables:\n  default: main\n  tables:\n'
                + '    main: {decoded_dn: [0.0, .inf]}\n'
            )
        },
        r'downlink_tables\.tables\.main\.decoded_dn\[1\]: expected a finite number, got float inf',
        'downlink-table-with-a-non-finite-value',
    ),
    (
        {
            'cam.yaml': _cam(
                _CAMERA
                + 'downlink_tables:\n  default: main\n  tables:\n'
                + '    main: {decoded_dn: [-1.0, 1.0]}\n'
            )
        },
        r'downlink_tables\.tables\.main\.decoded_dn\[0\]: expected a non-negative number, '
        r'got -1\.0',
        'downlink-table-with-a-negative-value',
    ),
    (
        {
            'cam.yaml': _cam(
                _CAMERA
                + 'downlink_tables:\n  default: main\n  tables:\n'
                + '    main: {codes: 4, linear_dn: 0.1, quadratic_dn: 0.0, whole_dn: true}\n'
            )
        },
        r'downlink_tables\.tables\.main: a downlink table decodes to at least 2 distinct '
        r'values, got 1',
        'parametric-table-rounding-to-one-value',
    ),
    (
        {
            'cam.yaml': _cam(
                _CAMERA
                + 'downlink_tables:\n  default: main\n  tables:\n'
                + '    main: {codes: 4, linear_dn: -1.0, quadratic_dn: 0.0, whole_dn: false}\n'
            )
        },
        r'downlink_tables\.tables\.main: a downlink table decodes to no negative DN, got -3',
        'parametric-table-decoding-negative',
    ),
    (
        {'generic.yaml': _generic_without('    calibrated_sky_if: 4.0e-5\n')},
        r'generic\.yaml: generic: detector\.calibrated_sky_if: the generic entry states the '
        r'calibrated sky residual',
        'generic-without-a-calibrated-sky',
    ),
    (
        {'cam.yaml': _cam(_with(_OFFERED, 'offered: [beam_bend]'))},
        r"cam\.yaml: cam: artifact_modes\.offered: 'beam_bend' renders only in the vidicon "
        r'detector chain; a ccd detector never applies it',
        'offered-mode-the-chain-never-applies',
    ),
    (
        {
            'cam.yaml': _cam(
                'base: generic\nartifact_modes:\n  instrument_defaults:\n'
                + '    residual_image: {probability: 1.0}\n'
            )
        },
        r"cam\.yaml: cam: artifact_modes\.instrument_defaults\.residual_image: 'residual_image' "
        r'renders only in the vidicon detector chain',
        'default-mode-the-chain-never-applies',
    ),
    (
        {'cam.yaml': _cam('base: coiss_nac\naliases: [x, null]\n')},
        r'cam\.yaml: cam: aliases\[1\]: a list item cannot be null',
        'null-list-item',
    ),
    (
        {
            'cam.yaml': _cam(
                'base: generic\nfilters:\n'
                + '  throughput_e_per_s_per_if: {CLEAR: 1.0}\n'
                + '  clear: CLEAR\n'
                + '  polarizer_clear: {P0: null}\n'
            )
        },
        r'cam\.yaml: cam: filters\.polarizer_clear\.P0: null removes an inherited key, and the '
        r'base has no such key',
        'null-under-a-key-the-base-lacks',
    ),
    (
        {'cam.yaml': _cam('base: nothing\n')},
        r"cam\.yaml: cam: base: 'nothing' is not an instrument in the catalog",
        'unknown-base',
    ),
    (
        {
            'cam_a.yaml': keyed('cam_a', 'base: cam_b\n'),
            'cam_b.yaml': keyed('cam_b', 'base: cam_a\n'),
        },
        r'cam_b\.yaml: cam_b: base: inheritance loops \(cam_a -> cam_b -> cam_a\)',
        'cyclic-base',
    ),
    (
        {'cam.yaml': _cam(_CAMERA + 'realism_cohort:\n  - {mission: COISS, camera: NAC}\n')},
        r"coiss\.yaml: coiss_calib_nac: realism_cohort\[0\]: \('COISS', 'NAC'\) is already "
        r'claimed by cam\.yaml: cam \(realism_cohort\[0\]\)',
        'realism-cohort-claimed-twice',
    ),
    (
        {'cam.yaml': _cam(_CAMERA), 'more.yaml': _cam(_CAMERA)},
        r'more\.yaml: cam: the instrument is already defined in cam\.yaml',
        'instrument-defined-in-two-files',
    ),
    (
        {'cam.yaml': 'cam: 5\n'},
        r'cam\.yaml: cam: expected a mapping, got int 5',
        'entry-not-a-mapping',
    ),
    (
        {'cam.yaml': _cam('base: generic\naliases: [gossi]\n')},
        r"cam\.yaml: cam: aliases: 'gossi' is already an instrument, defined in gossi\.yaml",
        'alias-in-use-as-a-name',
    ),
    (
        {'cam.yaml': _cam('base: generic\naliases: [sim]\n')},
        r"generic\.yaml: generic: aliases: 'sim' is already an alias of 'cam' in cam\.yaml",
        'alias-in-use-as-an-alias',
    ),
    (
        {'cam.yaml': '- x\n'},
        r'cam\.yaml: the file must map instrument names to entries',
        'file-not-a-mapping',
    ),
    (
        {'generic.yaml': keyed('cam', 'base: gossi\n')},
        r"no 'generic' entry \(the instrument-agnostic entry\)",
        'no-generic-entry',
    ),
]


def test_the_packaged_catalog_holds_every_top_level_key() -> None:
    """Every top-level key of a packaged file, other than the generic entry, is a camera."""
    yaml = YAML(typ='safe')
    keys = [key for path in INSTRUMENT_CONFIG_DIR.glob('*.yaml') for key in yaml.load(path)]
    assert [*sim_instrument_names(), 'generic'] == sorted(keys, key=lambda s: (s == 'generic', s))


@pytest.mark.parametrize(
    ('files', 'message'),
    [(files, message) for files, message, _ in _REFUSED],
    ids=[case_id for _, _, case_id in _REFUSED],
)
def test_the_loader_refuses_a_bad_config(
    tmp_path: Path, files: dict[str, str], message: str
) -> None:
    """Each violation fails at load, with the file and the key path in the message."""
    with pytest.raises(InstrumentConfigError, match=message):
        load_instrument_catalog(config_dir=catalog_dir_with(tmp_path, files))


def test_a_null_removes_an_inherited_key(tmp_path: Path) -> None:
    """An entry that sets an inherited readout mode to null drops it."""
    directory = catalog_dir_with(
        tmp_path, {'cam.yaml': _cam('base: coiss_nac\nreadout_modes: {4: null}\n')}
    )
    entry = load_instrument_catalog(config_dir=directory).entry('cam')
    assert sorted(entry['readout_modes']) == [1, 2]


def test_a_base_does_not_pass_on_its_realism_cohort(tmp_path: Path) -> None:
    """If an entry's base claims a realism cohort, the entry itself claims none."""
    directory = catalog_dir_with(tmp_path, {'cam.yaml': _cam('base: coiss_calib_nac\n')})
    entry = load_instrument_catalog(config_dir=directory).entry('cam')
    assert 'realism_cohort' not in entry


def _shipped_parametric_tables() -> list[dict[str, Any]]:
    """Each distinct parametric downlink table in the packaged catalog."""
    catalog = load_instrument_catalog()
    tables: list[dict[str, Any]] = []
    for name in sorted(catalog.entries):
        for table in (
            (catalog.entries[name].get('downlink_tables') or {}).get('tables', {}).values()
        ):
            if 'codes' in table and table not in tables:
                tables.append(dict(table))
    return tables


@pytest.mark.parametrize('table', _shipped_parametric_tables())
def test_the_load_check_decodes_a_parametric_table_as_the_renderer_does(
    table: dict[str, Any],
) -> None:
    """The load check's decoded levels are the levels the companding stage decodes to."""
    checked = np.unique(_parametric_decoded_dn(table))
    np.testing.assert_array_equal(checked, SqrtLut(**table).levels())


@pytest.mark.parametrize('instrument', [*sim_instrument_names(), GENERIC_INSTRUMENT])
def test_each_entry_names_a_navigator_block_that_exists(instrument: str) -> None:
    """Each entry's navigator section and detector point at a block of the packaged config."""
    block = resolve_sim_inst_config(DEFAULT_CONFIG, instrument)
    assert isinstance(block.get('noise'), Mapping)


def test_an_inherited_detector_under_another_section_is_caught(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An entry that changes its base's navigator section but inherits its detector fails."""
    text = _cam('base: coiss_nac\nnavigator: {section: galileo_ssi}\n')
    monkeypatch.setattr(
        instrument_catalog, 'INSTRUMENT_CONFIG_DIR', catalog_dir_with(tmp_path, {'cam.yaml': text})
    )
    with pytest.raises(ValueError, match=r"references detector 'nac' in section 'galileo_ssi'"):
        resolve_sim_inst_config(DEFAULT_CONFIG, 'cam')


@pytest.mark.parametrize('instrument', [*sim_instrument_names(), GENERIC_INSTRUMENT])
def test_each_entry_states_the_units_its_navigator_block_reads(instrument: str) -> None:
    """Each entry's units are the units its navigator block sets."""
    navigator = load_instrument_catalog().entry(instrument)['navigator']
    block = DEFAULT_CONFIG.category(navigator['section'])
    if 'detector' in navigator:
        block = block[navigator['detector']]
    assert resolve_data_units(instrument) == block['data_units']


@pytest.fixture
def calibrated_cam_on_raw_block(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Point the resolvers at a camera that renders I/F against the raw sim block."""
    text = _cam(_with('data_units: raw_dn', 'data_units: calibrated_if'))
    monkeypatch.setattr(
        instrument_catalog, 'INSTRUMENT_CONFIG_DIR', catalog_dir_with(tmp_path, {'cam.yaml': text})
    )
    return 'cam'


def test_units_that_differ_from_the_navigator_block_fail(calibrated_cam_on_raw_block: str) -> None:
    """An entry fails when its navigator block is resolved if their units differ."""
    message = (
        r'cam\.yaml: cam: data_units: the entry renders calibrated_if frames, but the '
        r'navigator reads frames of block sim as raw_dn\. The two must match\.'
    )
    with pytest.raises(InstrumentConfigError, match=message):
        resolve_sim_inst_config(DEFAULT_CONFIG, calibrated_cam_on_raw_block)


def test_a_navigator_block_without_units_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A navigator block that doesn't set data_units fails when it is resolved."""
    monkeypatch.setattr(
        instrument_catalog,
        'INSTRUMENT_CONFIG_DIR',
        catalog_dir_with(tmp_path, {'cam.yaml': _cam(_CAMERA)}),
    )
    block = dict(DEFAULT_CONFIG.category('sim'))
    del block['data_units']
    monkeypatch.setattr(DEFAULT_CONFIG, 'category', lambda section: block)
    message = (
        r"cam\.yaml: cam: data_units: the navigator block sim doesn't set data_units, which "
        r'the navigator requires'
    )
    with pytest.raises(InstrumentConfigError, match=message):
        resolve_sim_inst_config(DEFAULT_CONFIG, 'cam')


def test_the_frame_units_follow_the_entry(
    calibrated_cam_on_raw_block: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The detector chain renders in the entry's units, not the navigator block's."""
    # The check makes the two sources agree, so switch it off to see which one the frame follows.
    monkeypatch.setattr(instrument_navigator, '_check_units_agree', lambda *_, **__: None)
    params = resolve_detector_params({'instrument': calibrated_cam_on_raw_block})
    assert params.data_units == 'calibrated_if'


def _raw_dn_navigator_instruments() -> list[str]:
    """The entries whose navigator block reads raw DN, so it uses a DN star zero point.

    The navigator's star limit (``ObsSim.star_max_usable_vmag``) reads
    ``star_flux_dn_per_s_vmag0`` exactly when the block's ``data_units`` is
    ``raw_dn``; a calibrated block never consults it.  The list reads each entry's
    own units, which the test above ties to its block's, so collecting this file
    doesn't resolve any navigator block.
    """
    return [
        name
        for name in (*sim_instrument_names(), GENERIC_INSTRUMENT)
        if resolve_data_units(name) == 'raw_dn'
    ]


def _rounding_step(value: float) -> float:
    """The place value of the last significant digit ``value`` is written with."""
    exponent = Decimal(repr(value)).normalize().as_tuple().exponent
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
    this test detects the two drifting apart by more than half the rounding step of
    the navigator's stated value (the place value of its last significant digit), so
    a difference the navigator's rounding explains passes.  A CCD's zero point is in
    electrons, converted through the default gain state; a vidicon's is already DN.
    """
    zero_point, domain = resolve_star_flux_zero_point(instrument)
    if domain == 'electrons':
        detector = resolve_detector_defaults(instrument)
        zero_point /= detector['gain_e_per_dn_by_state'][detector['default_gain_state']]
    # A block without the key fails here and in the test above, which says why.
    stated = resolve_sim_inst_config(DEFAULT_CONFIG, instrument)['noise'][
        'star_flux_dn_per_s_vmag0'
    ]
    assert zero_point == pytest.approx(stated, rel=0.0, abs=_rounding_step(stated) / 2.0)


@pytest.fixture
def fictional_camera(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Point the resolvers at a catalog holding the fictional camera; return its name."""
    directory = catalog_dir_with(tmp_path, {f'{FICTIONAL_CAMERA_NAME}.yaml': FICTIONAL_CAMERA})
    monkeypatch.setattr(instrument_catalog, 'INSTRUMENT_CONFIG_DIR', directory)
    return FICTIONAL_CAMERA_NAME


def _fictional_scene(instrument: str) -> dict[str, Any]:
    """A binned, filtered, companded scene on the fictional camera."""
    return {
        'instrument': instrument,
        'size_v': 32,
        'size_u': 32,
        'random_seed': 1,
        'detector': {
            'binning': 2,
            'filter': 'red',
            'gain_state': 2,
            'quantization': 'sqrt_lut',
            'downlink_table': 'stepped',
        },
        'artifacts': {'instrument_defaults': True},
    }


def test_a_camera_added_as_an_entry_validates_and_renders(fictional_camera: str) -> None:
    """A scene using the camera's filter, binning, gain, and table validates and renders."""
    scene = _fictional_scene(fictional_camera)
    validate_sim_params(scene)
    image, _ = render_combined_model(scene)
    assert image.shape == (32, 32)


def test_a_camera_added_as_an_entry_decodes_through_its_named_table(fictional_camera: str) -> None:
    """The scene's downlink table is the tabulated one the entry lists."""
    curve = resolve_detector_params(_fictional_scene(fictional_camera)).sqrt_lut
    assert curve == TabulatedLut(
        decoded_dn=(0.0, 10.0, 30.0, 70.0, 150.0, 310.0, 630.0, 1270.0, 4095.0)
    )


def test_a_camera_added_as_an_entry_lists_its_readout_modes(fictional_camera: str) -> None:
    """The camera's binned readout mode is cataloged beside the unbinned one."""
    assert cataloged_binnings(fictional_camera) == (1, 2)


def test_a_camera_added_as_an_entry_states_why_a_mode_is_unavailable(fictional_camera: str) -> None:
    """If a scene asks for a mode the camera does not offer, it fails with the entry's reason."""
    scene = {**_fictional_scene(fictional_camera), 'artifacts': {'hot_pixels': {}}}
    with pytest.raises(SimSceneValidationError, match='the fictional camera has none'):
        validate_sim_params(scene)


@pytest.mark.parametrize(
    ('instrument', 'detector', 'mode', 'model'),
    [
        ('generic', None, 'beam_bend', 'ccd'),
        ('generic', {'detector_model': 'vidicon'}, 'fixed_pattern', 'vidicon'),
    ],
    ids=['instrument-model', 'scene-model'],
)
def test_a_scene_naming_a_mode_its_detector_chain_ignores_fails(
    instrument: str, detector: dict[str, Any] | None, mode: str, model: str
) -> None:
    """A scene may set a mode only when its detector model's chain applies it."""
    scene: dict[str, Any] = {
        'instrument': instrument,
        'size_v': 32,
        'size_u': 32,
        'random_seed': 1,
        'artifacts': {mode: {'probability': 1.0}},
    }
    if detector is not None:
        scene['detector'] = detector
    with pytest.raises(
        SimSceneValidationError,
        match=rf"artifact mode '{mode}' renders only in the .* detector chain; this scene "
        rf'renders with detector_model {model}',
    ):
        validate_sim_params(scene)


def test_instrument_defaults_turning_on_a_mode_the_chain_ignores_fails() -> None:
    """instrument_defaults may not turn on a mode the scene's own detector model ignores."""
    scene = {
        'instrument': 'nhlorri',
        'size_v': 32,
        'size_u': 32,
        'random_seed': 1,
        'detector': {'detector_model': 'vidicon'},
        'artifacts': {'instrument_defaults': True},
    }
    with pytest.raises(
        SimSceneValidationError,
        match=r"instrument_defaults turns on artifact mode 'frame_transfer_smear', which "
        r'renders only in the ccd detector chain; this scene renders with detector_model '
        r'vidicon',
    ):
        validate_sim_params(scene)


def test_vgiss_states_why_it_has_no_fixed_pattern() -> None:
    """A vgiss scene that sets fixed_pattern fails with the reason the vgiss entry states."""
    scene = {
        'instrument': 'vgiss',
        'size_v': 32,
        'size_u': 32,
        'random_seed': 1,
        'artifacts': {'fixed_pattern': {'probability': 1.0}},
    }
    with pytest.raises(
        SimSceneValidationError, match=r'vidicon DN chain has no fixed-pattern stage'
    ):
        validate_sim_params(scene)


def test_a_scene_naming_an_unlisted_downlink_table_fails() -> None:
    """A scene's downlink table must be one its camera lists."""
    scene = {
        'instrument': 'coiss_nac',
        'size_v': 32,
        'size_u': 32,
        'random_seed': 1,
        'detector': {'downlink_table': 'x'},
    }
    with pytest.raises(SimSceneValidationError, match=r"available tables: \['table_12_to_8'\]"):
        validate_sim_params(scene)


def test_a_downlink_table_no_quantization_reads_fails() -> None:
    """A scene may set a downlink table only when its frame is companded."""
    scene = {
        'instrument': 'coiss_nac',
        'size_v': 32,
        'size_u': 32,
        'random_seed': 1,
        'detector': {'downlink_table': 'table_12_to_8'},
    }
    with pytest.raises(
        SimSceneValidationError,
        match=r"detector\.downlink_table 'table_12_to_8' has no effect: only the sqrt_lut "
        r"quantization decodes through a downlink table, and this scene quantizes with 'exact'; "
        r'set detector\.quantization: sqrt_lut, add the quantization_sqrt_lut artifact mode, or '
        r'drop the key',
    ):
        validate_sim_params(scene)


def test_a_downlink_table_companded_by_the_native_readout_mode_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no binning set, validation reads the native readout mode, as a render does."""
    text = FICTIONAL_CAMERA.replace('  1: {}\n', '  1: {quantization: sqrt_lut}\n')
    assert text != FICTIONAL_CAMERA
    directory = catalog_dir_with(tmp_path, {f'{FICTIONAL_CAMERA_NAME}.yaml': text})
    monkeypatch.setattr(instrument_catalog, 'INSTRUMENT_CONFIG_DIR', directory)
    scene = {
        'instrument': FICTIONAL_CAMERA_NAME,
        'size_v': 32,
        'size_u': 32,
        'random_seed': 1,
        'detector': {'downlink_table': 'stepped'},
    }
    assert validate_sim_params(scene)['detector'] == {'downlink_table': 'stepped'}


@pytest.mark.parametrize(
    'table',
    [
        '{decoded_dn: [0.0, 100.0, 5000.0]}',
        '{codes: 256, linear_dn: 2.0, quadratic_dn: 0.1, whole_dn: true}',
    ],
    ids=['tabulated', 'parametric'],
)
def test_a_table_decoding_above_the_adc_ceiling_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, table: str
) -> None:
    """A companded frame refuses a table whose top value exceeds saturation_dn."""
    text = _cam(_CAMERA + f'downlink_tables:\n  default: high\n  tables:\n    high: {table}\n')
    monkeypatch.setattr(
        instrument_catalog, 'INSTRUMENT_CONFIG_DIR', catalog_dir_with(tmp_path, {'cam.yaml': text})
    )
    scene = {'instrument': 'cam', 'detector': {'quantization': 'sqrt_lut'}}
    with pytest.raises(
        DetectorParamError, match=r'above the ADC ceiling \(saturation_dn\) of 4095 DN'
    ):
        resolve_detector_params(scene)
