"""A fictional simulated camera, added as one config file in a copy of the catalog.

The tests that show a camera needs nothing but its config file use it: it has a
filter table with a clear filter, a binned readout mode beside a native frame
size, two downlink tables (one parametric, one tabulated), three gain states, and
an artifact mode it does not offer, with the reason.
"""

import shutil
from pathlib import Path

from spindoctor.sim.instrument_catalog import INSTRUMENT_CONFIG_DIR

__all__ = ['FICTIONAL_CAMERA', 'FICTIONAL_CAMERA_NAME', 'catalog_dir_with']

# The instrument name the fictional camera's file stem gives it.
FICTIONAL_CAMERA_NAME = 'fictional_cam'

FICTIONAL_CAMERA = """\
navigator:
  section: sim
native_frame_size_px: 64
detector:
  detector_model: ccd
  full_well_e: 20000.0
  exposure_ref_sec: 1.0
  gain_e_per_dn_by_state: {0: 8.0, 1: 4.0, 2: 2.0}
  default_gain_state: 1
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
  2: {read_noise_e: 4.0}
star_flux:
  e_per_s_vmag0: 1.0e+5
sky_pixel_scale_arcsec: 2.5
psf:
  truncation_px: 16
downlink_tables:
  default: coarse
  tables:
    coarse: {codes: 64, linear_dn: 4.0, quadratic_dn: 0.5, whole_dn: true}
    stepped: {decoded_dn: [0.0, 10.0, 30.0, 70.0, 150.0, 310.0, 630.0, 1270.0, 4095.0]}
filters:
  throughput_e_per_s_per_if: {CLEAR: 2.0e+4, RED: 5.0e+3}
  clear: CLEAR
artifact_modes:
  offered: [missing_lines]
  unavailable_reasons:
    hot_pixels: not offered, since the fictional camera has none
"""


def catalog_dir_with(tmp_path: Path, files: dict[str, str]) -> Path:
    """A copy of the packaged config directory with ``files`` written into it.

    Parameters:
        tmp_path: A scratch directory to hold the copy.
        files: Each file name mapped to the text to write.

    Returns:
        The copy's path.
    """
    directory = tmp_path / 'instrument_configs'
    shutil.copytree(INSTRUMENT_CONFIG_DIR, directory)
    for name, text in files.items():
        (directory / name).write_text(text)
    return directory
