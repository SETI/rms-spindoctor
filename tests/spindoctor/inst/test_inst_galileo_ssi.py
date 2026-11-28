import math
from typing import Any

import pytest
from tests.config import REQUIRES_EXTERNAL_DATA, URL_GALILEO_SSI_IO_01
from tests.packaged_sections import packaged_camera
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


def _star_obs(
    *,
    filter_name: str = 'CLEAR',
    texp: float = 1.0,
    gain: int = 4,
    tlmfmt: str = 'HCM',
    **block_changes: Any,
) -> ObsGalileoSSI:
    """Build a bare observation carrying what the star limit reads.

    The VICAR ``GAIN`` item 4 is the 10K gain state, and the ``TLMFMT`` item HCM is a
    full-frame format.  The observation carries the packaged camera block, with any
    fields given in ``block_changes`` changed.
    """
    return bare_observation(
        ObsGalileoSSI,
        VicarLabelStandIn(GAIN=gain, TLMFMT=tlmfmt),
        filter=filter_name,
        texp=texp,
        _config=DEFAULT_CONFIG,
        _camera_block=packaged_camera('galileo_ssi', **block_changes),
    )


@REQUIRES_EXTERNAL_DATA
def test_galileo_ssi_basic() -> None:
    obs = obstgossi.ObsGalileoSSI.from_file(URL_GALILEO_SSI_IO_01)
    assert obs.midtime == -110923771.04177806


@pytest.mark.parametrize(('tlmfmt', 'expected'), [('HCM', 12.513), ('AI8', 13.604)])
def test_clear_one_second_limit_is_each_readout_modes_calibrated_value(
    tlmfmt: str, expected: float
) -> None:
    """At 1 s through CLEAR at 10K each readout mode reaches its own limit.

    2.385e8 e-/s over 38.7 e-/DN puts 6.16e6 DN/s in the image for a star of camera
    magnitude 0, so the dimmest visible star of a full frame, 153 DN, is at camera
    magnitude 11.51, and of a summed frame, 56 DN, at 12.60.
    """
    assert _star_obs(tlmfmt=tlmfmt).star_max_usable_vmag() == pytest.approx(expected, abs=0.001)


def test_the_color_term_is_the_camera_blocks() -> None:
    """The camera's color term is its block's measured -0.41."""
    assert _star_obs().star_color_term() == pytest.approx(-0.41)


def test_ten_times_the_exposure_deepens_the_limit_by_two_and_a_half_magnitudes() -> None:
    """The limit moves with the exposure by 2.5 log10 of its ratio."""
    deeper = _star_obs(texp=10.0).star_max_usable_vmag() - _star_obs().star_max_usable_vmag()
    assert deeper == pytest.approx(2.5, abs=1e-9)


def test_the_gain_item_selects_the_gain_state() -> None:
    """GAIN 2 is the 100K state, whose 352.3 e-/DN dims a star against 10K's 38.7."""
    shift = _star_obs(gain=2).star_max_usable_vmag() - _star_obs().star_max_usable_vmag()
    assert shift == pytest.approx(-2.5 * math.log10(352.3 / 38.7), abs=1e-9)


def test_a_filter_brightens_the_limit_by_its_magnitude_offset() -> None:
    """IR-9680 passes less sunlight than CLEAR; its offset brightens the limit."""
    shift = (
        _star_obs(filter_name='IR-9680').star_max_usable_vmag() - _star_obs().star_max_usable_vmag()
    )
    assert shift == pytest.approx(-5.77, abs=1e-9)


def test_a_non_positive_exposure_admits_no_star() -> None:
    """A zero-length exposure gets a limit of -inf."""
    assert _star_obs(texp=0.0).star_max_usable_vmag() == -math.inf


def test_a_filter_without_an_offset_is_refused() -> None:
    """A filter that the photometry table lacks appears in the error."""
    with pytest.raises(
        ValueError,
        match=r"star_photometry\.filter_mag_offset has no entry for the filter 'IR-1234'",
    ):
        _star_obs(filter_name='IR-1234').star_max_usable_vmag()


def test_a_camera_block_without_star_photometry_is_refused() -> None:
    """The star limit reads the block's star photometry, so a block without it is refused."""
    with pytest.raises(ValueError, match=r'galileo_ssi\.star_photometry is not set'):
        _star_obs(star_photometry=None).star_max_usable_vmag()


def test_a_label_without_a_known_gain_state_is_refused() -> None:
    """A GAIN item outside 1 to 4 appears in the error."""
    with pytest.raises(ValueError, match='SSI label does not give a known gain state: GAIN is 7'):
        _star_obs(gain=7).star_max_usable_vmag()


def test_a_label_without_a_known_telemetry_format_is_refused() -> None:
    """A TLMFMT item outside the camera's telemetry formats appears in the error."""
    with pytest.raises(
        ValueError, match="SSI label does not give a known telemetry format: TLMFMT is 'XYZ'"
    ):
        _star_obs(tlmfmt='XYZ').star_max_usable_vmag()


@pytest.mark.parametrize(('tlmfmt', 'expected'), [('HCM', 0.65), ('AI8', 0.40)])
def test_the_star_width_is_each_readout_modes_own(tlmfmt: str, expected: float) -> None:
    """The star PSF has the sigma of the frame's readout mode, in its own pixels."""
    assert _star_obs(tlmfmt=tlmfmt).star_psf_sigma() == pytest.approx(expected, abs=1e-12)


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
