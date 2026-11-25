import math

import pytest
from tests.config import REQUIRES_EXTERNAL_DATA, URL_GALILEO_SSI_IO_01
from tests.spindoctor.inst.conftest import (
    VicarLabelStandIn,
    bare_observation,
    published_clock_counts,
)
from tests.spindoctor.public_metadata_galileo_ssi import GALILEO_SSI_PUBLIC_METADATA

import spindoctor.obs.obs_inst_galileo_ssi as obstgossi
from spindoctor.config import DEFAULT_CONFIG
from spindoctor.obs.obs_inst_galileo_ssi import ObsGalileoSSI

# The marker is applied per test rather than module-wide: the tests of the star gate and
# of clock counts fetch nothing, so they run even where the external trees are absent.


def _star_obs(*, filter_name: str = 'CLEAR', texp: float = 1.0, gain: int = 4) -> ObsGalileoSSI:
    """Build a bare observation carrying what the star limit reads.

    The VICAR ``GAIN`` item 4 is the 10K gain state.
    """
    return bare_observation(
        ObsGalileoSSI,
        VicarLabelStandIn(GAIN=gain),
        filter=filter_name,
        texp=texp,
        _config=DEFAULT_CONFIG,
    )


@REQUIRES_EXTERNAL_DATA
def test_galileo_ssi_basic() -> None:
    obs = obstgossi.ObsGalileoSSI.from_file(URL_GALILEO_SSI_IO_01)
    assert obs.midtime == -110923771.04177806


def test_clear_one_second_limit_is_the_calibrated_value() -> None:
    """At 1 s through CLEAR at 10K the limit is the dimmest visible star plus 1 mag.

    2.34e8 e-/s over 38.7 e-/DN puts 6.05e6 DN/s in the image for V = 0, which is
    140 DN, the dimmest visible star, at V 11.59.
    """
    assert _star_obs().star_max_usable_vmag() == pytest.approx(12.588, abs=0.001)


def test_ten_times_the_exposure_deepens_the_limit_by_two_and_a_half_magnitudes() -> None:
    """The limit moves with the exposure by 2.5 log10 of its ratio."""
    deeper = _star_obs(texp=10.0).star_max_usable_vmag() - _star_obs().star_max_usable_vmag()
    assert deeper == pytest.approx(2.5, abs=1e-9)


def test_the_gain_item_selects_the_gain_state() -> None:
    """GAIN 2 is the 100K state, whose 378.1 e-/DN dims a star against 10K's 38.7."""
    shift = _star_obs(gain=2).star_max_usable_vmag() - _star_obs().star_max_usable_vmag()
    assert shift == pytest.approx(-2.5 * math.log10(378.1 / 38.7), abs=1e-9)


def test_a_filter_brightens_the_limit_by_its_magnitude_offset() -> None:
    """IR-9680 passes less sunlight than CLEAR; its offset brightens the limit."""
    shift = (
        _star_obs(filter_name='IR-9680').star_max_usable_vmag() - _star_obs().star_max_usable_vmag()
    )
    assert shift == pytest.approx(-5.77, abs=1e-9)


def test_a_non_positive_exposure_admits_no_star() -> None:
    """A zero-length exposure records no starlight."""
    assert _star_obs(texp=0.0).star_max_usable_vmag() == -math.inf


def test_a_filter_without_an_offset_is_refused() -> None:
    """A filter the photometry table does not carry names itself in the error."""
    with pytest.raises(ValueError, match="no filter_mag_offset for the filter 'IR-1234'"):
        _star_obs(filter_name='IR-1234').star_max_usable_vmag()


def test_a_label_without_a_known_gain_state_is_refused() -> None:
    """A GAIN item that names no gain state names itself in the error."""
    with pytest.raises(ValueError, match='SSI label states no known gain state: GAIN is 7'):
        _star_obs(gain=7).star_max_usable_vmag()


def test_the_published_count_is_a_fractional_rim_count_with_no_stop() -> None:
    """The label's frame count is published in RIM counts, its finer fields a fraction.

    C0360361168R's VICAR label gives its frame count as RIM 3603611, MOD91 68, MOD10 2
    and MOD8 4.  A Galileo label records no count at the end of the image, so the
    midtime and end counts are null.
    """
    label = VicarLabelStandIn(RIM=3603611, MOD91=68, MOD10=2, MOD8=4)
    obs = bare_observation(ObsGalileoSSI, label, filter='CLEAR')
    start, midtime, end = published_clock_counts(obs)
    assert start == pytest.approx(3603611 + 68 / 91 + 2 / 910 + 4 / 7280, abs=1e-9)
    assert midtime is None
    assert end is None


def test_the_metadata_fixture_holds_the_keys_the_host_publishes() -> None:
    """The metadata chapter's Galileo fixture holds the host's keys, in the host's order.

    The chapter's staleness guard takes the Galileo facts from that fixture, so a fact the
    host gains has to reach the fixture, and through it the chapter.
    """
    label = VicarLabelStandIn(RIM=3603611, MOD91=68, MOD10=2, MOD8=4)
    public = bare_observation(ObsGalileoSSI, label, filter='CLEAR').get_public_metadata()
    assert list(public) == list(GALILEO_SSI_PUBLIC_METADATA)


def test_a_label_missing_a_clock_item_publishes_no_count() -> None:
    """A label lacking any of its four clock items publishes no count."""
    label = VicarLabelStandIn(RIM=3603611, MOD91=68, MOD10=2)
    obs = bare_observation(ObsGalileoSSI, label, filter='CLEAR')
    assert published_clock_counts(obs)[0] is None
