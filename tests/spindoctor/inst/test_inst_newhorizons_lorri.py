import math

import pytest
from tests.config import REQUIRES_EXTERNAL_DATA, URL_NEWHORIZONS_LORRI_CHARON_01
from tests.spindoctor.inst.conftest import bare_observation, published_clock_counts

import spindoctor.obs.obs_inst_newhorizons_lorri as obstnhlorri
from spindoctor.config import DEFAULT_CONFIG
from spindoctor.obs.obs_inst_newhorizons_lorri import ObsNewHorizonsLORRI, _published_sclk

# The marker is applied per test rather than module-wide: the tests of the star gate and
# of clock counts fetch nothing, so they run even where the external trees are absent.


def _star_obs(*, rows: int = 1024, texp: float = 1.0) -> ObsNewHorizonsLORRI:
    """Build a bare observation carrying what the star limit reads."""
    return bare_observation(
        ObsNewHorizonsLORRI,
        {},
        texp=texp,
        _data_shape_vu=(rows, rows),
        _config=DEFAULT_CONFIG,
    )


@REQUIRES_EXTERNAL_DATA
def test_newhorizons_lorri_basic() -> None:
    obs = obstnhlorri.ObsNewHorizonsLORRI.from_file(URL_NEWHORIZONS_LORRI_CHARON_01)
    assert obs.midtime == 490113790.9641424


@pytest.mark.parametrize(('rows', 'expected'), [(1024, 14.789), (256, 15.533)])
def test_one_second_limit_is_each_binning_modes_calibrated_value(
    rows: int, expected: float
) -> None:
    """At 1 s each binning mode reaches its own limit.

    7.57e8 e-/s over 21.0 e-/DN puts 3.60e7 DN/s in an unbinned frame for V = 0, so the
    dimmest visible star, 110 DN, is V 13.79; a binned frame reads 19.4 e-/DN and 60 DN.
    """
    assert _star_obs(rows=rows).star_max_usable_vmag() == pytest.approx(expected, abs=0.001)


def test_ten_times_the_exposure_deepens_the_limit_by_two_and_a_half_magnitudes() -> None:
    """The limit moves with the exposure by 2.5 log10 of its ratio."""
    deeper = _star_obs(texp=10.0).star_max_usable_vmag() - _star_obs().star_max_usable_vmag()
    assert deeper == pytest.approx(2.5, abs=1e-9)


def test_a_non_positive_exposure_admits_no_star() -> None:
    """A zero-length exposure records no starlight."""
    assert _star_obs(texp=0.0).star_max_usable_vmag() == -math.inf


def test_a_row_count_of_no_binning_mode_is_refused() -> None:
    """A frame of neither 1024 nor 256 rows gives its row count in the error."""
    with pytest.raises(
        ValueError, match='LORRI frame has 512 rows; a binning mode needs 1024 or 256'
    ):
        _star_obs(rows=512).star_max_usable_vmag()


@pytest.mark.parametrize(('rows', 'expected'), [(1024, 0.9), (256, 0.45)])
def test_the_star_width_is_each_binning_modes_own(rows: int, expected: float) -> None:
    """The star PSF has the sigma of the frame's binning mode, in its own pixels."""
    assert _star_obs(rows=rows).star_psf_sigma() == pytest.approx(expected, abs=1e-12)


def test_the_published_counts_are_fractional_seconds_and_their_exact_mean() -> None:
    """The label's counts are published as clock seconds, and their mean as the midtime.

    lor_0003104398's one-second exposure crosses a second: its label counts are
    0003104396:49000 and 0003104397:49000, in ticks of 1/50000 second.
    """
    obs = bare_observation(
        ObsNewHorizonsLORRI, {}, _label_clock_counts=('0003104396:49000', '0003104397:49000')
    )
    start, midtime, end = published_clock_counts(obs)
    assert start == pytest.approx(3104396.98, abs=1e-9)
    assert midtime == pytest.approx(3104397.48, abs=1e-9)
    assert end == pytest.approx(3104397.98, abs=1e-9)


def test_the_midtime_count_is_the_float_nearest_the_exact_mean() -> None:
    """The midtime count is the float nearest the exact mean of the two counts.

    lor_0019683105's label counts, 0019683104:48900 and 0019683104:49000, are
    19683104.978 and 19683104.98 seconds.  The mean of the two floats nearest them is
    one unit in the last place above 19683104.979.
    """
    counts = _published_sclk('0019683104:48900', '0019683104:49000')
    assert counts['midtime_sclk'] == 19683104.979


@REQUIRES_EXTERNAL_DATA
def test_newhorizons_lorri_metadata_clock_counts_are_the_pds3_labels() -> None:
    """The published clock counts are those of the PDS3 label beside the FITS image.

    The Charon test image's PDS3 label gives 0299147640:41500 and 0299147640:49000; the
    FITS header carries the start count only.
    """
    obs = obstnhlorri.ObsNewHorizonsLORRI.from_file(URL_NEWHORIZONS_LORRI_CHARON_01)
    meta = obs.get_public_metadata()
    counts = [meta[key] for key in ('start_time_sclk', 'midtime_sclk', 'end_time_sclk')]
    assert counts == pytest.approx([299147640.83, 299147640.905, 299147640.98], abs=1e-9)
