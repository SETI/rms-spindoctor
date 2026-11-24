"""Unit tests for the simulated cameras' filter passbands."""

import numpy as np
import pytest

from spindoctor.sim.instruments.passbands import (
    PASSBAND_DIR,
    solar_weighted_passband,
)
from spindoctor.sim.instruments.resolvers import FilterCatalogError, cataloged_filters


def test_a_scene_without_a_filter_takes_the_clear_combination() -> None:
    """A scene that names no filter reads the camera's clear combination."""
    assert solar_weighted_passband('coiss_nac', None).filter_name == 'CL1+CL2'


def test_the_clear_effective_wavelength_matches_the_camera_teams() -> None:
    """The narrow angle clear passband's solar-weighted mean wavelength is 651.1 nm.

    The camera team's calibration tables give 651.06 nm for the same combination.
    """
    passband = solar_weighted_passband('coiss_nac', 'CL1+CL2')
    effective = float(np.sum(passband.weight * passband.wavelength_nm))
    assert effective == pytest.approx(651.06, abs=0.1)


@pytest.mark.parametrize('camera', ['coiss_nac', 'coiss_wac'])
def test_every_cataloged_combination_has_a_passband(camera: str) -> None:
    """Each filter combination the camera catalogs has a column in its passband table.

    Parameters:
        camera: The camera entry.
    """
    header = (PASSBAND_DIR / f'{camera}.csv').read_text().split('\n', 1)[0].split(',')
    assert set(cataloged_filters(camera)) <= set(header)


def test_a_camera_without_passbands_is_refused() -> None:
    """A camera whose entry names no passband table raises, naming the camera."""
    with pytest.raises(FilterCatalogError, match="'vgiss_1_nac' has no passband table"):
        solar_weighted_passband('vgiss_1_nac', None)


def test_changing_a_passbands_wavelengths_leaves_the_next_lookup_alone() -> None:
    """A caller that changes the returned wavelengths in place does not change the table."""
    first = solar_weighted_passband('coiss_nac', 'CL1+CL2')
    original = first.wavelength_nm.copy()
    first.wavelength_nm[:] = 0.0
    second = solar_weighted_passband('coiss_nac', 'CL1+CL2')
    np.testing.assert_array_equal(second.wavelength_nm, original)
