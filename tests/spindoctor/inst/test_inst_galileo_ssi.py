import math

import pytest
from tests.config import REQUIRES_EXTERNAL_DATA, URL_GALILEO_SSI_IO_01
from tests.spindoctor.inst.conftest import (
    VicarLabelStandIn,
    bare_observation,
    published_clock_counts,
)

import spindoctor.obs.obs_inst_galileo_ssi as obstgossi
from spindoctor.obs.obs_inst_galileo_ssi import ObsGalileoSSI

# The marker is applied per test rather than module-wide: the tests of the star gate and
# of clock counts fetch nothing, so they run even where the external trees are absent.

# Documented anchor limiting magnitude (limiting mag at texp = 1 s).
_GALILEO_ANCHOR = 10.3


def _make_obs(texp: float) -> ObsGalileoSSI:
    """Build a bare ObsGalileoSSI carrying only texp.

    The star-magnitude gate is a pure function of ``self.texp``, so a fully
    constructed observation (and an external image fetch) is unnecessary for
    testing it.
    """
    obs = object.__new__(ObsGalileoSSI)
    obs.texp = texp
    return obs


@REQUIRES_EXTERNAL_DATA
def test_galileo_ssi_basic() -> None:
    obs = obstgossi.ObsGalileoSSI.from_file(URL_GALILEO_SSI_IO_01)
    assert obs.midtime == -110923771.01052806


def test_star_max_usable_vmag_anchor_at_unit_exposure() -> None:
    """The limiting magnitude equals the anchor at texp = 1 s."""
    obs = _make_obs(1.0)
    assert obs.star_max_usable_vmag() == pytest.approx(_GALILEO_ANCHOR, abs=1e-6)


def test_star_max_usable_vmag_gains_one_mag_per_pogson_ratio() -> None:
    """A 2.512x longer exposure deepens the limit by ~1 mag."""
    base = _make_obs(1.0).star_max_usable_vmag()
    deeper = _make_obs(2.512).star_max_usable_vmag()
    assert deeper - base == pytest.approx(1.0, abs=1e-3)


def test_star_max_usable_vmag_in_sane_range() -> None:
    """The limiting magnitude stays within a sane 3..15 range."""
    vmag = _make_obs(2.512).star_max_usable_vmag()
    assert 3.0 <= vmag <= 15.0


def test_star_max_usable_vmag_is_finite() -> None:
    """The limiting magnitude is finite."""
    vmag = _make_obs(2.512).star_max_usable_vmag()
    assert math.isfinite(vmag)


def test_star_max_usable_vmag_non_positive_exposure_returns_anchor() -> None:
    """A non-positive exposure falls back to the anchor magnitude."""
    obs = _make_obs(0.0)
    assert obs.star_max_usable_vmag() == pytest.approx(_GALILEO_ANCHOR, abs=1e-6)


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


def test_a_label_missing_a_clock_item_publishes_no_count() -> None:
    """A label lacking any of its four clock items publishes no count."""
    label = VicarLabelStandIn(RIM=3603611, MOD91=68, MOD10=2)
    obs = bare_observation(ObsGalileoSSI, label, filter='CLEAR')
    assert published_clock_counts(obs)[0] is None
