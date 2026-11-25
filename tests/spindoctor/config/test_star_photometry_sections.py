"""Tests for the typed star photometry sections of the Galileo, LORRI, and Voyager cameras.

The schema core checks each key's type and range, and is tested in its own suite.  What
is tested here is that each packaged section parses, and the LORRI section's cross-field
rule.
"""

import pytest

from spindoctor.config import Config
from spindoctor.obs.star_photometry_galileo_ssi import GalileoSSIStarPhotometrySection
from spindoctor.obs.star_photometry_newhorizons_lorri import (
    NewHorizonsLORRIStarPhotometry,
    NewHorizonsLORRIStarPhotometrySection,
)
from spindoctor.obs.star_photometry_voyager_iss import VoyagerISSStarPhotometrySection
from spindoctor.schema import RuleError


def test_packaged_galileo_ssi_star_photometry_section_parses() -> None:
    """The packaged ``galileo_ssi_star_photometry`` section parses."""
    section = Config().galileo_ssi_star_photometry
    assert isinstance(section, GalileoSSIStarPhotometrySection)


def test_packaged_newhorizons_lorri_star_photometry_section_parses() -> None:
    """The packaged ``newhorizons_lorri_star_photometry`` section parses."""
    section = Config().newhorizons_lorri_star_photometry
    assert isinstance(section, NewHorizonsLORRIStarPhotometrySection)


def test_packaged_voyager_iss_star_photometry_section_parses() -> None:
    """The packaged ``voyager_iss_star_photometry`` section parses."""
    section = Config().voyager_iss_star_photometry
    assert isinstance(section, VoyagerISSStarPhotometrySection)


def test_lorri_tables_must_give_the_same_binning_modes() -> None:
    """A binning mode one table gives and another leaves out is refused."""
    with pytest.raises(RuleError, match=r"gives the binning modes \['1X1'\], but"):
        NewHorizonsLORRIStarPhotometry(
            flux_electrons_per_s_vmag0=7.57e8,
            electrons_per_dn={'1X1': 21.0, '4X4': 19.4},
            min_detectable_dn={'1X1': 110.0},
            search_depth_mag=1.0,
            star_psf_sigma={'1X1': 0.9, '4X4': 0.45},
        )
