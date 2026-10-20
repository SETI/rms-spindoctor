"""Tests for the photometric star limiting magnitude of ``ObsCassiniISS``."""

import math
from typing import Any

import pytest
from tests.spindoctor.inst.conftest import VicarLabelStandIn, bare_observation

from spindoctor.config import DEFAULT_CONFIG
from spindoctor.obs.obs_inst_cassini_iss import ObsCassiniISS

_SECTION = 'cassini_iss_star_photometry'


def _obs(
    *,
    detector: str = 'NAC',
    filters: tuple[str, str] = ('CL1', 'CL2'),
    texp: float = 1.0,
    gain: str = '29 ELECTRONS PER DN',
) -> ObsCassiniISS:
    """Build a bare observation carrying what the star limit reads."""
    return bare_observation(
        ObsCassiniISS,
        VicarLabelStandIn(GAIN_MODE_ID=gain),
        detector=detector,
        filter1=filters[0],
        filter2=filters[1],
        texp=texp,
        _config=DEFAULT_CONFIG,
    )


def _photometry(detector: str) -> Any:
    """Return the configured star photometry of one camera."""
    return DEFAULT_CONFIG.category(_SECTION)[detector.lower()]


@pytest.mark.parametrize(
    ('detector', 'expected'),
    [('NAC', 14.753), ('WAC', 11.664)],
)
def test_clear_one_second_limit_is_the_calibrated_value(detector: str, expected: float) -> None:
    """At 1 s through CL1+CL2 at 29 e/DN each camera reaches its calibrated limit.

    The values are the half-plateau detection magnitudes measured on F ring frames (NAC
    13.75, WAC 10.66) plus the 1 mag search depth.
    """
    assert _obs(detector=detector).star_max_usable_vmag() == pytest.approx(expected, abs=0.001)


def test_limit_is_the_dimmest_visible_star_plus_the_search_depth() -> None:
    """The limit is the magnitude depositing the minimum DN, plus the search depth."""
    photometry = _photometry('NAC')
    dn_vmag0 = (
        photometry['flux_electrons_per_s_vmag0']
        / photometry['electrons_per_dn']['29 ELECTRONS PER DN']
    )
    expected = (
        2.5 * math.log10(dn_vmag0 / photometry['min_detectable_dn'])
        + photometry['search_depth_mag']
    )
    assert _obs().star_max_usable_vmag() == pytest.approx(expected, abs=1e-9)


def test_ten_times_the_exposure_deepens_the_limit_by_two_and_a_half_magnitudes() -> None:
    """The limit moves with the exposure by 2.5 log10 of its ratio."""
    deeper = _obs(texp=10.0).star_max_usable_vmag() - _obs(texp=1.0).star_max_usable_vmag()
    assert deeper == pytest.approx(2.5, abs=1e-9)


def test_a_lower_gain_state_deepens_the_limit_by_its_electrons_per_dn_ratio() -> None:
    """At 12 e/DN a star puts 30.27 / 12.841 times the DN of 29 e/DN in the image."""
    deeper = _obs(gain='12 ELECTRONS PER DN').star_max_usable_vmag() - _obs().star_max_usable_vmag()
    assert deeper == pytest.approx(2.5 * math.log10(30.27 / 12.841), abs=1e-6)


@pytest.mark.parametrize(
    ('detector', 'filters', 'offset'),
    [('NAC', ('CL1', 'GRN'), 1.60), ('WAC', ('CL1', 'IR1'), 0.62)],
)
def test_a_filter_pair_brightens_the_limit_by_its_magnitude_offset(
    detector: str, filters: tuple[str, str], offset: float
) -> None:
    """A narrower filter pair passes less light; its CISSCAL offset brightens the limit."""
    shift = (
        _obs(detector=detector, filters=filters).star_max_usable_vmag()
        - _obs(detector=detector).star_max_usable_vmag()
    )
    assert shift == pytest.approx(-offset, abs=1e-9)


def test_a_filter_pair_is_looked_up_in_its_canonical_spelling() -> None:
    """The label's wheel order does not matter: GRN+CL1 reads the CL1+GRN entry."""
    assert _obs(filters=('GRN', 'CL1')).star_max_usable_vmag() == pytest.approx(
        _obs(filters=('CL1', 'GRN')).star_max_usable_vmag(), abs=1e-12
    )


def test_a_non_positive_exposure_admits_no_star() -> None:
    """A zero-length exposure records no starlight."""
    assert _obs(texp=0.0).star_max_usable_vmag() == -math.inf


def test_a_filter_pair_without_an_offset_is_refused() -> None:
    """A pair the photometry table does not carry names itself in the error."""
    with pytest.raises(ValueError, match=r"filter pair 'CL1\+XYZ'"):
        _obs(filters=('XYZ', 'CL1')).star_max_usable_vmag()


def test_a_gain_state_without_electrons_per_dn_is_refused() -> None:
    """A gain state the photometry table does not carry names itself in the error."""
    with pytest.raises(ValueError, match="gain state '7 ELECTRONS PER DN'"):
        _obs(gain='7 ELECTRONS PER DN').star_max_usable_vmag()


def test_a_label_without_a_gain_state_is_refused() -> None:
    """The limit needs the gain state; a label that does not state one names the gap."""
    obs = _obs()
    obs.dict = VicarLabelStandIn()
    with pytest.raises(ValueError, match='label states no GAIN_MODE_ID'):
        obs.star_max_usable_vmag()
