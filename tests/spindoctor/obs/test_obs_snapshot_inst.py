"""Unit tests for ``ObsSnapshotInst.star_sharpness_psf``."""

from types import SimpleNamespace
from typing import Any, cast

import pytest

from spindoctor.obs.obs_snapshot_inst import ObsSnapshotInst


def _sharpness_psf(inst_config: dict[str, Any] | None, rows: int = 1024) -> Any:
    """Call ``star_sharpness_psf`` on a stand-in carrying a config and an image size."""
    stand_in = SimpleNamespace(_inst_config=inst_config, data_shape_vu=(rows, rows))
    return ObsSnapshotInst.star_sharpness_psf(cast(ObsSnapshotInst, stand_in))


def test_a_single_sigma_serves_every_image_size() -> None:
    """A scalar ``star_sharpness_psf_sigma`` is the PSF's sigma."""
    psf = _sharpness_psf({'star_sharpness_psf_sigma': 0.5}, rows=512)
    assert psf.sigma_y == pytest.approx(0.5)


def test_a_sigma_per_row_count_is_chosen_by_the_image_size() -> None:
    """A mapping is read at the image's row count, as summing narrows a star in pixels."""
    psf = _sharpness_psf({'star_sharpness_psf_sigma': {512: 0.25, 1024: 0.5}}, rows=512)
    assert psf.sigma_x == pytest.approx(0.25)


def test_no_configured_sigma_gives_no_psf() -> None:
    """An instrument that gives no ``star_sharpness_psf_sigma`` has no sharpness PSF."""
    assert _sharpness_psf({'star_psf_sigma': 3.0}) is None


def test_no_instrument_config_gives_no_psf() -> None:
    """An observation without an instrument configuration has no sharpness PSF."""
    assert _sharpness_psf(None) is None


def test_a_row_count_the_mapping_omits_gives_no_psf() -> None:
    """An image size with no measured narrowest star has no sharpness PSF."""
    assert _sharpness_psf({'star_sharpness_psf_sigma': {1024: 0.5}}, rows=256) is None
