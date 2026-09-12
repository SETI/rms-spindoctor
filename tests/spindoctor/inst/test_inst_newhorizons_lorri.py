import math

import pytest
from tests.config import REQUIRES_EXTERNAL_DATA, URL_NEWHORIZONS_LORRI_CHARON_01
from tests.spindoctor.inst.conftest import bare_observation, published_clock_counts

import spindoctor.obs.obs_inst_newhorizons_lorri as obstnhlorri
from spindoctor.obs.obs_inst_newhorizons_lorri import ObsNewHorizonsLORRI, _published_sclk

# The marker is applied per test rather than module-wide: the tests of the star gate and
# of clock counts fetch nothing, so they run even where the external trees are absent.

# Documented anchor limiting magnitude (limiting mag at texp = 1 s).
_LORRI_ANCHOR = 11.7


def _make_obs(texp: float) -> ObsNewHorizonsLORRI:
    """Build a bare ObsNewHorizonsLORRI carrying only texp.

    The star-magnitude gate is a pure function of ``self.texp``, so a fully
    constructed observation (and an external image fetch) is unnecessary for
    testing it.
    """
    obs = object.__new__(ObsNewHorizonsLORRI)
    obs.texp = texp
    return obs


@REQUIRES_EXTERNAL_DATA
def test_newhorizons_lorri_basic() -> None:
    obs = obstnhlorri.ObsNewHorizonsLORRI.from_file(URL_NEWHORIZONS_LORRI_CHARON_01)
    assert obs.midtime == 490113790.9641424


def test_star_max_usable_vmag_anchor_at_unit_exposure() -> None:
    """The limiting magnitude equals the anchor at texp = 1 s."""
    obs = _make_obs(1.0)
    assert obs.star_max_usable_vmag() == pytest.approx(_LORRI_ANCHOR, abs=1e-6)


def test_star_max_usable_vmag_gains_one_mag_per_pogson_ratio() -> None:
    """A 2.512x longer exposure deepens the limit by ~1 mag."""
    base = _make_obs(1.0).star_max_usable_vmag()
    deeper = _make_obs(2.512).star_max_usable_vmag()
    assert deeper - base == pytest.approx(1.0, abs=1e-3)


def test_star_max_usable_vmag_in_sane_range() -> None:
    """The limiting magnitude stays within a sane 3..15 range."""
    vmag = _make_obs(1.0).star_max_usable_vmag()
    assert 3.0 <= vmag <= 15.0


def test_star_max_usable_vmag_is_finite() -> None:
    """The limiting magnitude is finite."""
    vmag = _make_obs(1.0).star_max_usable_vmag()
    assert math.isfinite(vmag)


def test_star_max_usable_vmag_non_positive_exposure_returns_anchor() -> None:
    """A non-positive exposure falls back to the anchor magnitude."""
    obs = _make_obs(0.0)
    assert obs.star_max_usable_vmag() == pytest.approx(_LORRI_ANCHOR, abs=1e-6)


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
