"""The simulated-instrument catalog: checked YAML entries, keyed by instrument.

Each violation the loader refuses is one case, with the file, the entry, and the key
path in its message.  A camera added only as an entry in a copy of the packaged
directory validates, resolves, and renders.  The packaged catalog's entries point at
navigator blocks that exist and set the same units as those blocks.  Resolving an
entry's navigator block fails if their units differ or if the block doesn't set
data_units.  The detector chain renders in the entry's units.  The navigator's star
values are checked against the catalog in ``test_navigator_star_values.py``.
"""

import pickle
from collections.abc import Mapping
from pathlib import Path
from typing import Any, get_type_hints

import numpy as np
import pytest
from ruamel.yaml import YAML
from tests.spindoctor.sim.fictional_camera import (
    FICTIONAL_CAMERA,
    FICTIONAL_CAMERA_NAME,
    catalog_dir_with,
    keyed,
)
from tests.spindoctor.sim.scenes import frame_scene

from spindoctor.config import DEFAULT_CONFIG
from spindoctor.schema import SchemaError
from spindoctor.sim.instruments import catalog as instrument_catalog
from spindoctor.sim.instruments import navigator_link
from spindoctor.sim.instruments.catalog import (
    GENERIC_INSTRUMENT,
    INSTRUMENT_CONFIG_DIR,
    load_instrument_catalog,
    sim_instrument_names,
)
from spindoctor.sim.instruments.navigator_link import resolve_sim_inst_config
from spindoctor.sim.instruments.resolvers import (
    cataloged_binnings,
    resolve_calibrated_sky_if,
    resolve_data_units,
    resolve_detector_defaults_native,
    resolve_downlink_table,
    resolve_readout_mode,
)
from spindoctor.sim.instruments.schema import DownlinkTable, parametric_decoded_dn
from spindoctor.sim.instruments.schema_detector import Detector, ReadoutMode
from spindoctor.sim.render.detector.params import DetectorParamError, resolve_detector_params
from spindoctor.sim.render.detector.sqrt_lut import LUTSqrt, LUTTabulated
from spindoctor.sim.render.driver import render_scene
from spindoctor.sim.scene.errors import SimSceneValidationError
from spindoctor.sim.scene.validate import validate_sim_params

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
  saturation_dn: 4095.0
  full_well_dn: 4095.0
  signal_full_scale_frac: 0.5
  missing_data_rate: 0.0
  dark_current_e_per_sec: 0.0
  cosmic_ray_events_per_pixel_per_sec: 0.0
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
_VIDICON_BLOCK = '  vidicon: {read_noise_line_dn: 1.0, read_noise_pixel_dn: 1.0}\n'


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


# Each case: the files written over the packaged directory, the expected message, and
# the case's id.  A message that starts with ^ starts a line, so its file is the only one.
_REFUSED: list[tuple[dict[str, str], str, str]] = [
    (
        {'cam.yaml': _cam(_CAMERA + 'sensor: {}\n')},
        r"^cam\.yaml: cam: unknown keys: \['sensor'\]",
        'unknown-key',
    ),
    (
        {'cam.yaml': _cam(_with('sky_pixel_scale_arcsec: 2.5\n', ''))},
        r"^cam\.yaml: cam: missing required keys: \['sky_pixel_scale_arcsec'\]",
        'missing-required-key',
    ),
    (
        {'cam.yaml': _cam(_with('data_units: raw_dn', 'data_units: reflectance'))},
        r"^cam\.yaml: cam: data_units: 'reflectance' is not one of \['calibrated_if', 'raw_dn'\]",
        'unknown-data-units',
    ),
    (
        {'generic.yaml': _generic_without('  data_units: raw_dn\n')},
        r"^generic\.yaml: generic: missing required keys: \['data_units'\]",
        'generic-without-data-units',
    ),
    (
        {'cam.yaml': _cam(_with('default_gain_state: 0', 'default_gain_state: 0.5'))},
        r'^cam\.yaml: cam: detector\.default_gain_state: expected an integer, got float 0\.5',
        'wrong-type',
    ),
    (
        {'cam.yaml': _cam(_with('full_well_e: 5000.0', 'full_well_e: .5e4'))},
        r"detector\.full_well_e: expected a number, got str '\.5e4' \(a number written",
        'number-read-as-a-string',
    ),
    (
        {'cam.yaml': _cam(_with('full_well_e: 5000.0', 'full_well_e: .nan'))},
        r'^cam\.yaml: cam: detector\.full_well_e: expected a finite number, got float nan',
        'non-finite-number',
    ),
    (
        {'cam.yaml': _cam(_with('{0: 2.0}', '{0: 0.0}'))},
        r'^cam\.yaml: cam: detector\.gain_e_per_dn_by_state\.0: expected a positive number, '
        r'got 0\.0',
        'non-positive-gain',
    ),
    (
        {
            'cam.yaml': _cam(
                _with('  quantization: exact', '  quantization: exact\n  dark_sky_ceiling_if: 0.0')
            )
        },
        r'^cam\.yaml: cam: detector\.dark_sky_ceiling_if: expected a positive number, got 0\.0',
        'non-positive-dark-sky-ceiling',
    ),
    (
        {
            'cam.yaml': _cam(
                _with(
                    '  quantization: exact', '  quantization: exact\n  dark_sky_ceiling_if: 2.5e-3'
                )
            )
        },
        r'^cam\.yaml: cam: detector\.dark_sky_ceiling_if: the dark-sky ceiling applies only to '
        r"calibrated_if products, and this entry's data_units is raw_dn",
        'dark-sky-ceiling-on-raw-dn',
    ),
    (
        {
            'cam.yaml': _cam(
                _with(_OFFERED, 'offered: [readout_dark]')
                + '  shape_defaults:\n    readout_dark: {rbi_column_factor: 5.0}\n'
            )
        },
        r'^cam\.yaml: cam: artifact_modes\.shape_defaults\.readout_dark\.rbi_column_factor: '
        r'expected a number in \[0, 1\], got 5\.0',
        'fraction-above-one',
    ),
    (
        {'cam.yaml': _cam(_with('  1: {}', '  1: {}\n  0: {}'))},
        r'^cam\.yaml: cam: readout_modes\.0: expected a positive integer as the key, got 0',
        'binning-below-one',
    ),
    (
        {'cam.yaml': _cam('navigator: {section: sim\n')},
        r'^cam\.yaml: not valid YAML',
        'yaml-syntax-error',
    ),
    (
        {'cam.yaml': _cam(_with('read_noise_e: 3.0', 'read_noise_e: 3.0\n  read_noise_e: 9.0'))},
        r'^cam\.yaml: not valid YAML: (?s:.*)duplicate key',
        'duplicate-key',
    ),
    (
        {'cam.yml': _CAMERA},
        r"\['cam\.yml'\] are not instrument config files",
        'file-not-named-yaml',
    ),
    (
        {'cam.yaml': _cam(_with('read_noise_e: 3.0', 'read_noise_e: null'))},
        r'^cam\.yaml: cam: detector\.read_noise_e: null removes the value an earlier layer sets, '
        r'and no earlier layer sets this key',
        'null-without-a-base',
    ),
    (
        {'cam.yaml': _cam('base: generic\ndistortion_residual: null\n')},
        r'^cam\.yaml: cam: distortion_residual: null removes the value an earlier layer sets, and '
        r'no earlier layer sets this key',
        'null-for-a-key-the-base-lacks',
    ),
    (
        {'cam.yaml': _cam('base: coiss_nac\nsky_pixel_scale_arcsec: null\n')},
        r"^cam\.yaml: cam: missing required keys: \['sky_pixel_scale_arcsec'\]",
        'null-removing-a-required-key',
    ),
    (
        {'cam.yaml': _cam('base: coiss_nac\npsf: {truncation_px: null}\n')},
        r"^cam\.yaml: cam: psf: missing required keys: \['truncation_px'\]",
        'null-removing-a-nested-required-key',
    ),
    (
        {'cam.yaml': _cam('base: coiss_nac\ndetector: {read_noise_e: null}\n')},
        r"^cam\.yaml: cam: detector: a ccd detector requires the keys \['read_noise_e'\]",
        'null-removing-a-ccd-key',
    ),
    (
        {'cam.yaml': _cam('base: vgiss_1_nac\ndetector: {vidicon: null}\n')},
        r"^cam\.yaml: cam: detector: a vidicon detector requires the keys \['vidicon'\]",
        'null-removing-the-vidicon-chain',
    ),
    (
        {'cam.yaml': _cam('base: coiss_nac\nreadout_modes: {1: null}\n')},
        r'^cam\.yaml: cam: readout_modes: binning 1 \(the unbinned readout\) is not defined',
        'null-removing-the-unbinned-readout',
    ),
    (
        {'cam.yaml': _cam('base: coiss_nac\nnative_frame_size_px: 1022\n')},
        r'^cam\.yaml: cam: readout_modes\.4: binning 4 does not divide native_frame_size_px 1022',
        'binning-not-dividing-the-native-size',
    ),
    (
        {
            'cam.yaml': _cam(
                _CAMERA + '  shape_defaults:\n    missing_lines: {expected_events: 5.0}\n'
            )
        },
        r'^cam\.yaml: cam: artifact_modes\.shape_defaults\.missing_lines\.expected_events: '
        r"'expected_events' turns the mode on",
        'activation-key-in-shape-defaults',
    ),
    (
        {
            'cam.yaml': _cam(
                _with(_OFFERED, 'offered: [dead_pixels]')
                + '  shape_defaults:\n    dead_pixels: {count: 3}\n'
            )
        },
        r"^cam\.yaml: cam: artifact_modes\.shape_defaults\.dead_pixels\.count: 'count' turns "
        r'the mode on',
        'count-in-shape-defaults',
    ),
    (
        {
            'cam.yaml': _cam(
                _with(_OFFERED, 'offered: [edited_frame]')
                + '  shape_defaults:\n    edited_frame: {band_width_px: 0}\n'
            )
        },
        r'shape_defaults\.edited_frame\.band_width_px: expected a positive integer, got 0',
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
        {
            'cam.yaml': _cam(
                _CAMERA + '  instrument_defaults:\n    missing_lines: {expected_events: 1.0}\n'
            )
        },
        r'^cam\.yaml: cam: artifact_modes\.instrument_defaults\.missing_lines: '
        r"'missing_lines' is not a detector-stage mode",
        'instrument-default-not-a-detector-stage',
    ),
    (
        {'cam.yaml': _cam(_CAMERA + '  shape_defaults:\n    bloom: {bloom_length: 2}\n')},
        r"^cam\.yaml: cam: artifact_modes\.shape_defaults\.bloom: 'bloom' is not in "
        r'artifact_modes\.offered',
        'mode-not-offered',
    ),
    (
        {'cam.yaml': _cam(_with(_OFFERED, 'offered: [missing_lines, bogus]'))},
        r"^cam\.yaml: cam: artifact_modes\.offered\[1\]: 'bogus' is not a registered artifact "
        r'mode',
        'offered-mode-not-registered',
    ),
    (
        {'cam.yaml': _cam(_CAMERA + '  unavailable_reasons: {bogus: not offered}\n')},
        r"^cam\.yaml: cam: artifact_modes\.unavailable_reasons\.bogus: 'bogus' is not a "
        r'registered artifact mode',
        'unavailable-mode-not-registered',
    ),
    (
        {'cam.yaml': _cam(_CAMERA + '  unavailable_reasons: {missing_lines: not offered}\n')},
        r"^cam\.yaml: cam: artifact_modes\.unavailable_reasons\.missing_lines: 'missing_lines' "
        r'is offered, so it has no unavailability reason',
        'unavailable-mode-offered',
    ),
    (
        {
            'cam.yaml': _cam(
                _CAMERA + 'filters:\n  throughput_e_per_s_per_if: {CLR: 1.0}\n  clear: CL1\n'
            )
        },
        r"^cam\.yaml: cam: filters\.clear: 'CL1' is not in throughput_e_per_s_per_if",
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
                _CAMERA + 'filters:\n  throughput_e_per_s_per_if: {CLR+: 1.0}\n  clear: CLR+\n'
            )
        },
        r"filters\.throughput_e_per_s_per_if\.CLR\+: 'CLR\+' has an empty filter element",
        'filter-with-an-empty-element',
    ),
    (
        {
            'cam.yaml': _cam(
                _CAMERA
                + 'filters:\n  throughput_e_per_s_per_if: {CLR: 1.0}\n  clear: CLR\n'
                + '  polarizer_clear: {P0+P1: CLR}\n'
            )
        },
        r'filters\.polarizer_clear\.P0\+P1: a polarizer is one filter element, without a \+',
        'polarizer-of-two-elements',
    ),
    (
        {
            'cam.yaml': _cam(
                _CAMERA
                + 'filters:\n  throughput_e_per_s_per_if: {CLR: 1.0}\n  clear: CLR\n'
                + '  polarizer_clear: {P0: CL1}\n'
            )
        },
        r"^cam\.yaml: cam: filters\.polarizer_clear\.P0: 'CL1' is not an element of any "
        r'cataloged filter combination',
        'polarizer-folding-onto-no-element',
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
        r"^cam\.yaml: cam: filters\.throughput_e_per_s_per_if\.P0\+GRN: the polarizer 'P0' "
        r"folds onto 'CL1', so a scene that sets 'P0\+GRN' reads 'CL1\+GRN'",
        'filter-keyed-by-a-polarizer',
    ),
    (
        {
            'cam.yaml': _cam(
                'base: vgiss_1_nac\nfilters:\n  throughput_e_per_s_per_if: {CLEAR: 1.0}\n'
                + '  clear: CLEAR\n'
            )
        },
        r'^cam\.yaml: cam: filters: a filter sets the CCD signal scale; the vidicon DN chain has '
        r'no reference exposure',
        'filters-on-a-vidicon',
    ),
    (
        {
            'cam.yaml': _cam(
                _with(
                    '  e_per_s_vmag0: 1.0e+5', '  e_per_s_vmag0: 1.0e+5\n  dn_per_s_vmag0: 1.0e+5'
                )
            )
        },
        r'^cam\.yaml: cam: star_flux: state exactly one of e_per_s_vmag0 and dn_per_s_vmag0',
        'star-zero-point-in-both-units',
    ),
    (
        {
            'cam.yaml': _cam(
                'base: vgiss_1_nac\nstar_flux: {dn_per_s_vmag0: null, e_per_s_vmag0: 1.0e+5}\n'
            )
        },
        r'^cam\.yaml: cam: star_flux: a vidicon detector states dn_per_s_vmag0',
        'vidicon-star-zero-point-in-electrons',
    ),
    (
        {'cam.yaml': _cam(_with('  e_per_s_vmag0: 1.0e+5', '  dn_per_s_vmag0: 1.0e+5'))},
        r'^cam\.yaml: cam: star_flux: a ccd detector states e_per_s_vmag0',
        'ccd-star-zero-point-in-dn',
    ),
    (
        {'cam.yaml': _cam(_with('default_gain_state: 0', 'default_gain_state: 3'))},
        r'^cam\.yaml: cam: detector\.default_gain_state: state 3 is not in '
        r'gain_e_per_dn_by_state \[0\]',
        'gain-state-not-cataloged',
    ),
    (
        {'cam.yaml': _cam(_with('{0: 2.0}', '{-1: 4.0, 0: 2.0}'))},
        r'^cam\.yaml: cam: detector\.gain_e_per_dn_by_state\.-1: expected a non-negative integer '
        r'as the key, got -1',
        'gain-state-below-zero',
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
        r'^cam\.yaml: cam: detector\.vidicon: a ccd detector has no vidicon chain to configure',
        'vidicon-chain-on-a-ccd',
    ),
    (
        {'cam.yaml': _cam(_with('  1: {}', '  1: {}\n  2: {detector_model: vidicon}'))},
        r'^cam\.yaml: cam: readout_modes\.2\.detector_model: a readout mode bins the same '
        r'detector, so it cannot change detector\.detector_model',
        'readout-mode-changing-the-model',
    ),
    (
        {'cam.yaml': _cam(_with('  1: {}', '  1: {}\n  2: {dark_sky_ceiling_if: 2.5e-3}'))},
        r'^cam\.yaml: cam: readout_modes\.2\.dark_sky_ceiling_if: the realism comparison reads '
        r'the dark-sky ceiling from the detector block',
        'readout-mode-setting-the-dark-sky-ceiling',
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
        {'cam.yaml': _cam(_with('  1: {}', '  1: {}\n  2: {default_gain_state: 5}'))},
        r'^cam\.yaml: cam: readout_modes\.2\.default_gain_state: state 5 is not in '
        r'gain_e_per_dn_by_state \[0\]',
        'readout-mode-gain-state-not-cataloged',
    ),
    (
        {'cam.yaml': _cam('base: vgiss_1_nac\nreadout_modes: {2: {}}\n')},
        r'^cam\.yaml: cam: readout_modes\.2: the vidicon DN chain has no binned readout',
        'binned-vidicon',
    ),
    (
        {'cam.yaml': _cam(_with('  1: {}', '  2: {}'))},
        r'^cam\.yaml: cam: readout_modes: binning 1 \(the unbinned readout\) is not defined',
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
        r"^cam\.yaml: cam: downlink_tables\.default: 'other' is not one of the tables \['main'\]",
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
        r'^generic\.yaml: generic: detector\.calibrated_sky_if: the generic entry states the '
        r'calibrated sky residual',
        'generic-without-a-calibrated-sky',
    ),
    (
        {'cam.yaml': _cam(_with(_OFFERED, 'offered: [beam_bend]'))},
        r"^cam\.yaml: cam: artifact_modes\.offered: 'beam_bend' renders only in the vidicon "
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
        r'^cam\.yaml: cam: artifact_modes\.instrument_defaults\.residual_image: '
        r"'residual_image' renders only in the vidicon detector chain",
        'default-mode-the-chain-never-applies',
    ),
    (
        {'cam.yaml': _cam('base: coiss_nac\naliases: [x, null]\n')},
        r'^cam\.yaml: cam: aliases\[1\]: a list item cannot be null',
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
        r'^cam\.yaml: cam: filters\.polarizer_clear\.P0: null removes the value an earlier layer '
        r'sets, and no earlier layer sets this key',
        'null-under-a-key-the-base-lacks',
    ),
    (
        {'cam.yaml': _cam('base: nothing\n')},
        r"^cam\.yaml: cam: base: 'nothing' is not an entry",
        'unknown-base',
    ),
    (
        {
            'cam_a.yaml': keyed('cam_a', 'base: cam_b\n'),
            'cam_b.yaml': keyed('cam_b', 'base: cam_a\n'),
        },
        r'^cam_a\.yaml: cam_a: base: inheritance loops \(cam_a -> cam_b -> cam_a\)',
        'cyclic-base',
    ),
    (
        {'cam.yaml': _cam(_CAMERA + 'realism_cohort:\n  - {mission: COISS, camera: NAC}\n')},
        r"^coiss\.yaml: coiss_calib_nac: realism_cohort\[0\]: \('COISS', None, 'NAC'\) is already "
        r'claimed by cam\.yaml: cam \(realism_cohort\[0\]\)',
        'realism-cohort-claimed-twice',
    ),
    (
        {'cam.yaml': _cam(_CAMERA), 'more.yaml': _cam(_CAMERA)},
        r'^more\.yaml: cam: the instrument is already defined in cam\.yaml',
        'instrument-defined-in-two-files',
    ),
    (
        {'cam.yaml': 'cam: 5\n'},
        r'^cam\.yaml: cam: expected a mapping, got int 5',
        'entry-not-a-mapping',
    ),
    (
        {'cam.yaml': _cam('base: generic\naliases: [gossi]\n')},
        r"^cam\.yaml: cam: aliases: 'gossi' is already an instrument, defined in gossi\.yaml",
        'alias-in-use-as-a-name',
    ),
    (
        {'cam.yaml': _cam('base: generic\naliases: [sim]\n')},
        r"^generic\.yaml: generic: aliases: 'sim' is already an alias of 'cam' in cam\.yaml",
        'alias-in-use-as-an-alias',
    ),
    (
        {'cam.yaml': '- x\n'},
        r"^cam\.yaml: expected a mapping at the top level, got list \['x'\]",
        'file-not-a-mapping',
    ),
    (
        {'cam.yaml': '# No entry.\n'},
        r'^cam\.yaml: the file holds no instrument entry',
        'empty-file',
    ),
    (
        {'cam.yaml': '123:\n  base: generic\n'},
        r'^cam\.yaml: the entry name 123 must be a string',
        'instrument-name-not-a-string',
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
    with pytest.raises(SchemaError, match=f'(?m){message}'):
        load_instrument_catalog(config_dir=catalog_dir_with(tmp_path, files))


def test_a_null_removes_an_inherited_key(tmp_path: Path) -> None:
    """An entry that sets an inherited readout mode to null drops it."""
    directory = catalog_dir_with(
        tmp_path, {'cam.yaml': _cam('base: coiss_nac\nreadout_modes: {4: null}\n')}
    )
    entry = load_instrument_catalog(config_dir=directory).entry('cam')
    assert sorted(entry.readout_modes) == [1, 2]


def test_a_base_does_not_pass_on_its_realism_cohort(tmp_path: Path) -> None:
    """If an entry's base claims a realism cohort, the entry itself claims none."""
    directory = catalog_dir_with(tmp_path, {'cam.yaml': _cam('base: coiss_calib_nac\n')})
    entry = load_instrument_catalog(config_dir=directory).entry('cam')
    assert entry.realism_cohort == []


def test_a_problem_two_entries_inherit_is_reported_once(tmp_path: Path) -> None:
    """A bad value in a base is one line, though both entries that inherit it fail."""
    files = {
        'generic.yaml': _generic_without('    bias_dn: 20.0\n').replace(
            '    quantization: exact\n', '    quantization: exact\n    bias_dn: -1.0\n'
        ),
        'cam.yaml': _cam('base: generic\n'),
    }
    line = 'generic.yaml: generic: detector.bias_dn: expected a non-negative number, got -1.0'
    with pytest.raises(SchemaError) as caught:
        load_instrument_catalog(config_dir=catalog_dir_with(tmp_path, files))
    assert str(caught.value).splitlines() == [line]


def test_a_rule_two_entries_break_by_inheriting_is_reported_once(tmp_path: Path) -> None:
    """A rule a base breaks between two of its own keys is one line, for the base alone."""
    text = (INSTRUMENT_CONFIG_DIR / 'coiss.yaml').read_text()
    bad = text.replace('    2: {}\n', '    2: {gain_e_per_dn_by_state: {0: 1.0}}\n', 1)
    assert bad != text
    line = (
        'coiss.yaml: coiss_nac: readout_modes.2.gain_e_per_dn_by_state: the states [0] differ '
        'from detector.gain_e_per_dn_by_state [0, 1, 2, 3]'
    )
    with pytest.raises(SchemaError) as caught:
        load_instrument_catalog(config_dir=catalog_dir_with(tmp_path, {'coiss.yaml': bad}))
    assert str(caught.value).splitlines() == [line]


# An entry that turns an inherited CCD into a vidicon, with its binned modes dropped.
_AS_VIDICON = (
    'base: coiss_nac\nreadout_modes: {2: null, 4: null}\ndetector:\n  detector_model: vidicon\n'
    + _VIDICON_BLOCK
)
_AS_VIDICON_STARS = _AS_VIDICON + 'star_flux: {e_per_s_vmag0: null, dn_per_s_vmag0: 1.0e+5}\n'

# A base whose rules hold, for the entries that break one by changing one of its keys.
_BASE = keyed(
    'base_cam',
    _with('{0: 2.0}', '{0: 2.0, 1: 1.0}').replace(
        '  1: {}', '  1: {}\n  2: {default_gain_state: 1}'
    )
    + 'filters:\n  throughput_e_per_s_per_if: {CLR: 1.0, CL1+GRN: 1.0}\n'
    + '  clear: CLR\n  polarizer_clear: {P0: CL1}\n',
).replace(
    'offered: [missing_lines]', 'offered: [missing_lines]\n    unavailable_reasons: {bloom: x}'
)


@pytest.mark.parametrize(
    ('entry', 'path'),
    [
        ('base: coiss_calib_nac\ndata_units: raw_dn\n', 'detector.dark_sky_ceiling_if'),
        (_AS_VIDICON.replace('2: null, ', ''), 'readout_modes.2'),
        (_AS_VIDICON, 'star_flux'),
        (_AS_VIDICON_STARS, 'filters'),
        (_AS_VIDICON_STARS + 'filters: null\n', 'artifact_modes.offered'),
        (
            'base: vgiss_1_nac\n'
            + _CAMERA[_CAMERA.index('detector:') : _CAMERA.index('readout_modes:')],
            'detector.vidicon',
        ),
        (
            'base: nhlorri\ndetector: {gain_e_per_dn_by_state: {1: 20.0}}\n',
            'readout_modes.4.gain_e_per_dn_by_state',
        ),
        (
            'base: base_cam\ndetector: {gain_e_per_dn_by_state: {1: null}}\n',
            'readout_modes.2.default_gain_state',
        ),
        (
            'base: gossi\ndetector: {gain_e_per_dn_by_state: {2: null, 3: null}}\n',
            'detector.default_gain_state',
        ),
        (
            'base: coiss_nac\ndownlink_tables:\n'
            + '  tables: {table_12_to_8: null, main: {decoded_dn: [0.0, 1.0]}}\n',
            'downlink_tables.default',
        ),
        (
            'base: coiss_nac\nfilters: {throughput_e_per_s_per_if: {CL1+CL2: null}}\n',
            'filters.clear',
        ),
        (
            'base: base_cam\nfilters: {polarizer_clear: {GRN: CLR}}\n',
            'filters.throughput_e_per_s_per_if.CL1+GRN',
        ),
        (
            'base: base_cam\nfilters: {throughput_e_per_s_per_if: {CL1+GRN: null}}\n',
            'filters.polarizer_clear.P0',
        ),
        (
            'base: nhlorri\nartifact_modes: {offered: [missing_lines], shape_defaults: null}\n',
            'artifact_modes.instrument_defaults.banding',
        ),
        (
            'base: base_cam\nartifact_modes: {offered: [missing_lines, bloom]}\n',
            'artifact_modes.unavailable_reasons.bloom',
        ),
    ],
    ids=[
        'dark-sky-ceiling',
        'binned-vidicon',
        'star-zero-point-unit',
        'filters-on-a-vidicon',
        'mode-chain',
        'vidicon-chain-on-a-ccd',
        'readout-gain-states',
        'readout-default-gain-state',
        'default-gain-state',
        'default-downlink-table',
        'clear-filter',
        'filter-keyed-by-a-polarizer',
        'polarizer-folding-onto-no-element',
        'default-mode-not-offered',
        'unavailable-mode-offered',
    ],
)
def test_the_entry_that_breaks_a_rule_with_its_base_is_the_one_reported(
    tmp_path: Path, entry: str, path: str
) -> None:
    """When an entry's key and an inherited key don't fit, the line gives the entry alone."""
    files = {'base.yaml': _BASE, 'cam.yaml': _cam(entry)}
    with pytest.raises(SchemaError) as caught:
        load_instrument_catalog(config_dir=catalog_dir_with(tmp_path, files))
    assert [(issue.source, issue.path) for issue in caught.value.issues] == [
        ('cam.yaml: cam', path)
    ]


def test_an_offered_mode_that_is_not_registered_is_one_line(tmp_path: Path) -> None:
    """A bad item of the offered list is reported once, not again for the 'all' choice."""
    text = _cam(_with(_OFFERED, 'offered: [missing_lines, bogus]'))
    with pytest.raises(SchemaError) as caught:
        load_instrument_catalog(config_dir=catalog_dir_with(tmp_path, {'cam.yaml': text}))
    assert len(caught.value.issues) == 1


def test_an_entry_survives_pickling() -> None:
    """Every packaged entry pickles, its artifact-mode tables included."""
    entries = dict(load_instrument_catalog().entries)
    assert pickle.loads(pickle.dumps(entries)) == entries


def test_a_readout_mode_types_each_detector_key_as_the_detector_does() -> None:
    """A readout mode's fields are the detector's, with the same types, all optional."""
    detector = get_type_hints(Detector, include_extras=True)
    expected = {name: hint | None for name, hint in detector.items()}
    assert get_type_hints(ReadoutMode, include_extras=True) == expected


def test_changing_a_resolved_detector_leaves_the_catalog_unchanged() -> None:
    """A detector a resolver returns is a copy, its gain table included."""
    table = resolve_detector_defaults_native('coiss_nac').gain_e_per_dn_by_state
    assert table is not None
    table.clear()
    assert resolve_detector_defaults_native('coiss_nac').gain_e_per_dn_by_state != {}


def test_a_calibrated_sky_residual_comes_from_the_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An entry that states its own calibrated sky residual keeps it."""
    text = _cam('base: generic\ndetector: {calibrated_sky_if: 7.0e-5}\n')
    monkeypatch.setattr(
        instrument_catalog, 'INSTRUMENT_CONFIG_DIR', catalog_dir_with(tmp_path, {'cam.yaml': text})
    )
    assert resolve_calibrated_sky_if('cam') == 7.0e-5


def _shipped_parametric_tables() -> list[DownlinkTable]:
    """Each distinct parametric downlink table in the packaged catalog."""
    catalog = load_instrument_catalog()
    tables: list[DownlinkTable] = []
    for name in sorted(catalog.entries):
        listed = catalog.entries[name].downlink_tables
        for table in [] if listed is None else listed.tables.values():
            if table.codes is not None and table not in tables:
                tables.append(table)
    return tables


# A table that keeps fractional DN, which no packaged table does.
_FRACTIONAL_TABLE = DownlinkTable(codes=4, linear_dn=0.1, quadratic_dn=0.0, whole_dn=False)


@pytest.mark.parametrize('table', [*_shipped_parametric_tables(), _FRACTIONAL_TABLE])
def test_the_load_check_decodes_a_parametric_table_as_the_renderer_does(
    table: DownlinkTable,
) -> None:
    """The load check's decoded levels are the levels the companding stage decodes to."""
    checked = np.unique(parametric_decoded_dn(table))
    curve = LUTSqrt(
        codes=table.codes or 0,
        linear_dn=table.linear_dn or 0.0,
        quadratic_dn=table.quadratic_dn or 0.0,
        whole_dn=bool(table.whole_dn),
    )
    np.testing.assert_array_equal(checked, curve.levels())


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
    navigator = load_instrument_catalog().entry(instrument).navigator
    block = DEFAULT_CONFIG.category(navigator.section)
    if navigator.detector is not None:
        block = block[navigator.detector]
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
    """Resolving an entry's navigator block fails if their units differ."""
    message = (
        r'cam\.yaml: cam: data_units: the entry renders calibrated_if frames, but its '
        r'navigator block sim reads frames as raw_dn; the units must match$'
    )
    with pytest.raises(SchemaError, match=message):
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
    with pytest.raises(SchemaError, match=message):
        resolve_sim_inst_config(DEFAULT_CONFIG, 'cam')


def test_the_frame_units_follow_the_entry(
    calibrated_cam_on_raw_block: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The detector chain renders in the entry's units, not the navigator block's."""
    # The check refuses an entry whose units differ from its block's, so switch it off to
    # see which of the two the frame follows.
    monkeypatch.setattr(navigator_link, '_check_units_agree', lambda *_, **__: None)
    params = resolve_detector_params({'instrument': calibrated_cam_on_raw_block, 'random_seed': 42})
    assert params.data_units == 'calibrated_if'


@pytest.fixture
def fictional_camera(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Point the resolvers at a catalog holding the fictional camera; return its name."""
    directory = catalog_dir_with(tmp_path, {f'{FICTIONAL_CAMERA_NAME}.yaml': FICTIONAL_CAMERA})
    monkeypatch.setattr(instrument_catalog, 'INSTRUMENT_CONFIG_DIR', directory)
    return FICTIONAL_CAMERA_NAME


def _fictional_scene(instrument: str) -> dict[str, Any]:
    """A binned, filtered, companded scene on the fictional camera."""
    return frame_scene(
        instrument=instrument,
        size=32,
        detector={
            'binning': 2,
            'filter': 'red',
            'gain_state': 2,
            'quantization': 'sqrt_lut',
            'downlink_table': 'stepped',
        },
        artifacts={'instrument_defaults': True},
    )


def test_a_camera_added_as_an_entry_validates_and_renders(fictional_camera: str) -> None:
    """A scene using the camera's filter, binning, gain, and table validates and renders."""
    scene = _fictional_scene(fictional_camera)
    validate_sim_params(scene)
    image, _ = render_scene(scene)
    assert image.shape == (32, 32)


def test_a_camera_added_as_an_entry_uses_its_gain_state(fictional_camera: str) -> None:
    """The scene's gain state 2 takes the entry's gain for that state, 2 e-/DN."""
    assert resolve_detector_params(_fictional_scene(fictional_camera)).gain_e_per_dn == 2.0


def test_a_camera_added_as_an_entry_uses_its_detector_values(fictional_camera: str) -> None:
    """The bias, ADC ceiling, full well, signal scale, and loss rate are the entry's own.

    The entry's navigator block is the generic one, which sets a 4095 DN ceiling.
    """
    params = resolve_detector_params(_fictional_scene(fictional_camera))
    assert (
        params.bias_dn,
        params.saturation_dn,
        params.full_well_dn,
        params.signal_full_scale_frac,
        params.missing_data_rate,
    ) == (10.0, 5000.0, 4500.0, 0.25, 0.125)


def test_a_camera_added_as_an_entry_scales_its_signal_by_the_filter(
    fictional_camera: str,
) -> None:
    """The scene's red filter sets the reference exposure from the entry's throughput.

    A signal of 1.0 fills the scaled well (20000 e- times the full-scale fraction) at
    the red filter's 5000 e-/s.
    """
    params = resolve_detector_params(_fictional_scene(fictional_camera))
    expected = params.signal_full_scale_frac * 20000.0 / 5000.0
    assert params.exposure_ref_sec == pytest.approx(expected)


def test_a_camera_added_as_an_entry_decodes_through_its_named_table(fictional_camera: str) -> None:
    """The scene's downlink table is the tabulated one the entry lists."""
    curve = resolve_detector_params(_fictional_scene(fictional_camera)).sqrt_lut
    assert curve == LUTTabulated(
        decoded_dn=(0.0, 10.0, 30.0, 70.0, 150.0, 310.0, 630.0, 1270.0, 4095.0)
    )


def test_changing_a_resolved_downlink_table_leaves_the_catalog_unchanged(
    fictional_camera: str,
) -> None:
    """A downlink table a resolver returns is a copy, its levels included."""
    table = resolve_downlink_table(fictional_camera, 'stepped')
    assert table is not None
    assert table.decoded_dn is not None
    table.decoded_dn.clear()
    again = resolve_downlink_table(fictional_camera, 'stepped')
    assert again is not None
    assert again.decoded_dn != []


def test_changing_a_resolved_readout_mode_leaves_the_catalog_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A readout mode a resolver returns is a copy, its gain table included."""
    text = _cam(_with('  1: {}', '  1: {gain_e_per_dn_by_state: {0: 2.0}}'))
    monkeypatch.setattr(
        instrument_catalog, 'INSTRUMENT_CONFIG_DIR', catalog_dir_with(tmp_path, {'cam.yaml': text})
    )
    table = resolve_readout_mode('cam', 1).gain_e_per_dn_by_state
    assert table is not None
    table.clear()
    assert resolve_readout_mode('cam', 1).gain_e_per_dn_by_state == {0: 2.0}


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
    scene = frame_scene(instrument=instrument, size=32, artifacts={mode: {'probability': 1.0}})
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
    scene = frame_scene(
        instrument='nhlorri',
        size=32,
        detector={'detector_model': 'vidicon'},
        artifacts={'instrument_defaults': True},
    )
    with pytest.raises(
        SimSceneValidationError,
        match=r"instrument_defaults turns on artifact mode 'frame_transfer_smear', which "
        r'renders only in the ccd detector chain; this scene renders with detector_model '
        r'vidicon',
    ):
        validate_sim_params(scene)


def test_vgiss_states_why_it_has_no_fixed_pattern() -> None:
    """A vgiss scene that sets fixed_pattern fails with the reason the vgiss entry states."""
    scene = frame_scene(
        instrument='vgiss_1_nac', size=32, artifacts={'fixed_pattern': {'probability': 1.0}}
    )
    with pytest.raises(
        SimSceneValidationError, match=r'vidicon DN chain has no fixed-pattern stage'
    ):
        validate_sim_params(scene)


_VOYAGER_CAMERAS = ('vgiss_1_nac', 'vgiss_1_wac', 'vgiss_2_nac', 'vgiss_2_wac')


def test_each_voyager_camera_is_an_instrument() -> None:
    """The four Voyager cameras are in the catalog, each with its own entry."""
    names = set(load_instrument_catalog().instrument_names())
    assert set(_VOYAGER_CAMERAS) <= names


@pytest.mark.parametrize('name', _VOYAGER_CAMERAS)
def test_each_voyager_camera_emulates_the_voyager_navigator_block(name: str) -> None:
    """Every Voyager camera points at the one voyager_iss section, with no detector."""
    navigator = load_instrument_catalog().entry(name).navigator
    assert navigator.section == 'voyager_iss'
    assert navigator.detector is None


# The optical keys of a Voyager camera, as paths into its own entry: the PSF, the residual
# distortion, the plate scale, the star zero point, and the vidicon values.
_VOYAGER_OPTICAL_KEYS = (
    ('psf', 'kernel'),
    ('psf', 'truncation_px'),
    ('distortion_residual',),
    ('sky_pixel_scale_arcsec',),
    ('star_flux', 'dn_per_s_vmag0'),
    ('detector', 'vidicon', 'read_noise_line_dn'),
    ('detector', 'vidicon', 'read_noise_pixel_dn'),
    ('detector', 'exposure_ref_sec'),
    ('detector', 'bias_dn'),
    ('detector', 'saturation_dn'),
    ('detector', 'full_well_dn'),
    ('detector', 'signal_full_scale_frac'),
    ('detector', 'quantization'),
    ('detector', 'calibrated_sky_if'),
    ('artifact_modes', 'shape_defaults', 'banding'),
    ('artifact_modes', 'shape_defaults', 'beam_bend'),
    ('artifact_modes', 'shape_defaults', 'residual_image'),
    ('artifact_modes', 'shape_defaults', 'readout_dark'),
    ('artifact_modes', 'shape_defaults', 'reseau_scars'),
    ('artifact_modes', 'shape_defaults', 'resample_texture'),
)


@pytest.mark.parametrize('name', _VOYAGER_CAMERAS)
def test_each_voyager_camera_states_every_optical_key_itself(name: str) -> None:
    """A Voyager entry inherits nothing: it has no base and states each optical key."""
    raw = YAML(typ='safe').load((INSTRUMENT_CONFIG_DIR / 'vgiss.yaml').read_text())
    entry = raw[name]
    assert 'base' not in entry
    missing = []
    for path in _VOYAGER_OPTICAL_KEYS:
        node: Any = entry
        for key in path:
            node = node.get(key) if isinstance(node, dict) else None
        if node is None:
            missing.append('.'.join(path))
    assert missing == []


def test_the_bare_vgiss_name_is_not_an_instrument() -> None:
    """A scene that writes the family name instead of a camera fails."""
    with pytest.raises(ValueError, match=r'vgiss'):
        load_instrument_catalog().entry('vgiss')


def test_a_scene_naming_an_unlisted_downlink_table_fails() -> None:
    """A scene's downlink table must be one its camera lists."""
    scene = frame_scene(size=32, detector={'downlink_table': 'x'})
    with pytest.raises(SimSceneValidationError, match=r"available tables: \['table_12_to_8'\]"):
        validate_sim_params(scene)


def test_a_downlink_table_no_quantization_reads_fails() -> None:
    """A scene may set a downlink table only when its frame is companded."""
    scene = frame_scene(size=32, detector={'downlink_table': 'table_12_to_8'})
    with pytest.raises(
        SimSceneValidationError,
        match=r"detector\.downlink_table 'table_12_to_8' has no effect: only the sqrt_lut "
        r"quantization reads it, and this scene quantizes with 'exact'; "
        r'set detector\.quantization: sqrt_lut, or drop the key',
    ):
        validate_sim_params(scene)


def test_a_contour_step_the_quantization_ignores_fails() -> None:
    """A scene may set the contour step only when its quantization is contour_8bit."""
    scene = frame_scene(size=32, detector={'quantization': '8bit', 'contour_step': 4})
    with pytest.raises(
        SimSceneValidationError,
        match=r'detector\.contour_step 4 has no effect: only the contour_8bit quantization',
    ):
        validate_sim_params(scene)


def test_a_contour_step_below_one_fails() -> None:
    """A contour step of 0 would divide by it, so validation refuses it."""
    scene = frame_scene(size=32, detector={'quantization': 'contour_8bit', 'contour_step': 0})
    with pytest.raises(
        SimSceneValidationError,
        match=r'detector\.contour_step: expected an integer from 1 to 255, got 0',
    ):
        validate_sim_params(scene)


def test_a_contour_step_above_the_8bit_top_code_fails() -> None:
    """A step above 255 would posterize the 8-bit word to at most two levels."""
    scene = frame_scene(size=32, detector={'quantization': 'contour_8bit', 'contour_step': 256})
    with pytest.raises(
        SimSceneValidationError,
        match=r'detector\.contour_step: expected an integer from 1 to 255, got 256',
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
    scene = frame_scene(
        instrument=FICTIONAL_CAMERA_NAME, size=32, detector={'downlink_table': 'stepped'}
    )
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
    scene = {'instrument': 'cam', 'random_seed': 42, 'detector': {'quantization': 'sqrt_lut'}}
    with pytest.raises(
        DetectorParamError, match=r'above the ADC ceiling \(saturation_dn\) of 4095 DN'
    ):
        resolve_detector_params(scene)


def test_a_scene_ceiling_below_the_downlink_table_top_fails_validation() -> None:
    """A scene whose ceiling the camera's table decodes above is refused before a render."""
    scene = frame_scene(size=32, detector={'saturation_dn': 1000.0, 'quantization': 'sqrt_lut'})
    with pytest.raises(
        SimSceneValidationError,
        match=r'detector\.saturation_dn 1000: .*above the ADC ceiling',
    ):
        validate_sim_params(scene)


def test_a_missing_data_rate_above_one_fails() -> None:
    """The rate is a fraction of the pixels, so more than 1 is refused."""
    scene = frame_scene(size=32, detector={'missing_data_rate': 1.5})
    with pytest.raises(
        SimSceneValidationError,
        match=r'detector\.missing_data_rate: expected a number in \[0, 1\], got 1\.5',
    ):
        validate_sim_params(scene)


def test_a_zero_signal_full_scale_frac_fails() -> None:
    """A signal of 1.0 has to fill some of the well, so zero is refused."""
    scene = frame_scene(size=32, detector={'signal_full_scale_frac': 0.0})
    with pytest.raises(
        SimSceneValidationError,
        match=r'detector\.signal_full_scale_frac: expected a positive number',
    ):
        validate_sim_params(scene)
