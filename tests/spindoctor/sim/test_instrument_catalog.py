"""The simulated-instrument catalog: one checked YAML file per camera.

Each schema violation the loader refuses is one case, with the file and key path in
its message; a camera added only as a file in a copy of the packaged directory
validates, resolves, and renders.
"""

import shutil
from pathlib import Path

import pytest

from spindoctor.sim import instrument_catalog
from spindoctor.sim.forward.artifacts_catalog import (
    resolve_detector_defaults,
    resolve_sky_pixel_scale_arcsec,
)
from spindoctor.sim.instrument_catalog import (
    INSTRUMENT_CONFIG_DIR,
    InstrumentConfigError,
    load_instrument_catalog,
    sim_instrument_names,
)
from spindoctor.sim.render import render_combined_model
from spindoctor.sim.scene import validate_sim_params

# A complete camera entry, written out so each case changes one thing in it.
_CAMERA = """\
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


def _catalog_dir(tmp_path: Path, files: dict[str, str]) -> Path:
    """A copy of the packaged config directory with ``files`` written over it."""
    directory = tmp_path / 'instrument_configs'
    shutil.copytree(INSTRUMENT_CONFIG_DIR, directory)
    for name, text in files.items():
        (directory / name).write_text(text)
    return directory


def test_the_packaged_catalog_names_every_config_file() -> None:
    """Every packaged file other than the generic entry is a camera."""
    stems = sorted(path.stem for path in INSTRUMENT_CONFIG_DIR.glob('*.yaml'))
    assert [*sim_instrument_names(), 'generic'] == sorted(stems, key=lambda s: s == 'generic')


@pytest.mark.parametrize(
    ('files', 'message'),
    [
        (
            {'cam.yaml': _CAMERA + 'sensor: {}\n'},
            r"cam\.yaml: unknown keys: \['sensor'\]",
        ),
        (
            {'cam.yaml': _CAMERA.replace('sky_pixel_scale_arcsec: 2.5\n', '')},
            r"cam\.yaml: missing required keys: \['sky_pixel_scale_arcsec'\]",
        ),
        (
            {'cam.yaml': _CAMERA.replace('bloom_length: 0', 'bloom_length: 0.5')},
            r'cam\.yaml: detector\.bloom_length: expected an integer, got float 0\.5',
        ),
        (
            {'cam.yaml': _CAMERA.replace('full_well_e: 5000.0', 'full_well_e: .5e4')},
            r"detector\.full_well_e: expected a number, got str '\.5e4' \(a number written",
        ),
        (
            {'cam.yaml': _CAMERA + '  shape_defaults:\n    bloom: {bloom_length: 2}\n'},
            r"cam\.yaml: artifact_modes\.shape_defaults\.bloom: 'bloom' is not in "
            r'artifact_modes\.offered',
        ),
        (
            {
                'cam.yaml': _CAMERA
                + 'filters:\n  throughput_e_per_s_per_if: {CLR: 1.0}\n  clear: CL1\n'
            },
            r"cam\.yaml: filters\.clear: 'CL1' is not in throughput_e_per_s_per_if",
        ),
        (
            {'cam.yaml': _CAMERA.replace('default_gain_state: 0', 'default_gain_state: 3')},
            r'cam\.yaml: detector\.default_gain_state: state 3 is not in '
            r'gain_e_per_dn_by_state \[0\]',
        ),
        (
            {'cam.yaml': _CAMERA.replace('  1: {}', '  2: {}')},
            r'cam\.yaml: readout_modes: binning 1 \(the unbinned readout\) is not defined',
        ),
        (
            {'cam.yaml': 'base: nothing\n'},
            r"cam\.yaml: base: 'nothing' is not an instrument in the catalog",
        ),
        (
            {'cam_a.yaml': 'base: cam_b\n', 'cam_b.yaml': 'base: cam_a\n'},
            r'cam_b\.yaml: base: inheritance loops \(cam_a -> cam_b -> cam_a\)',
        ),
    ],
    ids=[
        'unknown-key',
        'missing-required-key',
        'wrong-type',
        'number-read-as-a-string',
        'mode-not-offered',
        'filter-not-cataloged',
        'gain-state-not-cataloged',
        'unbinned-readout-missing',
        'unknown-base',
        'cyclic-base',
    ],
)
def test_the_loader_refuses_a_schema_violation(
    tmp_path: Path, files: dict[str, str], message: str
) -> None:
    """Each violation fails at load, naming the file and the key path."""
    with pytest.raises(InstrumentConfigError, match=message):
        load_instrument_catalog(config_dir=_catalog_dir(tmp_path, files))


def test_a_camera_is_one_config_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A camera added only as a file validates, resolves its own and inherited keys, and renders."""
    text = (
        'base: generic\n'
        'navigator: {section: sim}\n'
        'detector: {read_noise_e: 3.0}\n'
        'sky_pixel_scale_arcsec: 2.5\n'
        'artifact_modes: {offered: [missing_lines]}\n'
    )
    monkeypatch.setattr(
        instrument_catalog, 'INSTRUMENT_CONFIG_DIR', _catalog_dir(tmp_path, {'cam.yaml': text})
    )
    scene = {
        'instrument': 'cam',
        'size_v': 32,
        'size_u': 32,
        'random_seed': 1,
        'artifacts': {'instrument_defaults': True},
    }
    validate_sim_params(scene)
    image, _ = render_combined_model(scene)
    assert image.shape == (32, 32)
    assert resolve_sky_pixel_scale_arcsec('cam') == 2.5
    assert resolve_detector_defaults('cam')['read_noise_e'] == 3.0
    assert (
        resolve_detector_defaults('cam')['full_well_e']
        == (resolve_detector_defaults('generic')['full_well_e'])
    )
