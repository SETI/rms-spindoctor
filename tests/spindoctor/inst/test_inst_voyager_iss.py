import math

import pytest
from tests.config import REQUIRES_EXTERNAL_DATA, URL_VOYAGER_ISS_IO_01
from tests.spindoctor.inst.conftest import (
    VicarLabelStandIn,
    bare_observation,
    published_clock_counts,
)

import spindoctor.obs.obs_inst_voyager_iss as obstvgiss
from spindoctor.config import DEFAULT_CONFIG
from spindoctor.obs.obs_inst_voyager_iss import (
    ObsVoyagerISS,
    _voyager_if_factor,
    _voyager_spacecraft_digit,
)

# The marker is applied per test rather than module-wide: the tests of label strings,
# clock counts and the star gate fetch nothing, so they run even where the external
# trees are absent.

# A real LAB03 record (fixed format), captured from a holdings GEOMED image, with the
# gain state written in.
_LAB03 = 'NA CAMERA  EXP    5760.0 MSEC FILT 2(BLUE  )  {gain} GAIN  SCAN RATE  1:1  C'


def _star_obs(
    *,
    spacecraft: str = '2',
    detector: str = 'NAC',
    filter_name: str = 'CLEAR',
    texp: float = 1.0,
    gain: str = 'LO',
) -> ObsVoyagerISS:
    """Build a bare observation carrying what the star limit reads."""
    lab02 = _LAB02_V1 if spacecraft == '1' else _LAB02_V2
    return bare_observation(
        ObsVoyagerISS,
        VicarLabelStandIn(LAB02=lab02, LAB03=_LAB03.format(gain=gain)),
        detector=detector,
        filter=filter_name,
        texp=texp,
        _config=DEFAULT_CONFIG,
    )


# Real Voyager VICAR LAB02 records (fixed-format strings; index 4 is the
# spacecraft digit).  Captured from a holdings GEOMED image.
_LAB02_V1 = 'VGR-1   FDS 20621.33   PICNO 1326J2-002   SCET 79.189 16:08:47         C'
_LAB02_V2 = 'VGR-2   FDS 20621.33   PICNO 1326J2-002   SCET 79.189 16:08:47         C'
_LABEL3 = 'FOR (I/F)*10000., MULTIPLY DN VALUE BY               1.00000'


@REQUIRES_EXTERNAL_DATA
def test_voyager_iss_basic() -> None:
    obs = obstvgiss.ObsVoyagerISS.from_file(URL_VOYAGER_ISS_IO_01)
    assert obs.midtime == -646429822.8760977


@REQUIRES_EXTERNAL_DATA
def test_voyager_iss_metadata_spacecraft_lid() -> None:
    """Public metadata derives the instrument-host LID from LAB02."""
    obs = obstvgiss.ObsVoyagerISS.from_file(URL_VOYAGER_ISS_IO_01)
    meta = obs.get_public_metadata()
    # The IO test image is a Voyager 2 (VGR-2) frame.
    assert meta['instrument_host_lid'].endswith('spacecraft.vg2')


def test_spacecraft_digit_voyager1() -> None:
    """A VGR-1 LAB02 record yields spacecraft digit '1'."""
    assert _voyager_spacecraft_digit(_LAB02_V1) == '1'


def test_spacecraft_digit_voyager2() -> None:
    """A VGR-2 LAB02 record yields spacecraft digit '2'."""
    assert _voyager_spacecraft_digit(_LAB02_V2) == '2'


def test_spacecraft_digit_too_short_raises() -> None:
    """A LAB02 string shorter than 5 chars raises a clear ValueError."""
    with pytest.raises(ValueError, match='Unexpected Voyager LAB02 format'):
        _voyager_spacecraft_digit('VGR')


def test_spacecraft_digit_non_string_raises() -> None:
    """A non-string LAB02 value raises a clear ValueError."""
    with pytest.raises(ValueError, match='Unexpected Voyager LAB02 format'):
        _voyager_spacecraft_digit(None)


def test_spacecraft_digit_unknown_id_raises() -> None:
    """A LAB02 with an out-of-range spacecraft id raises ValueError."""
    bad = 'VGR-3   FDS 20621.33'
    with pytest.raises(ValueError, match='expected 1 or 2'):
        _voyager_spacecraft_digit(bad)


def test_if_factor_parses_value() -> None:
    """A well-formed LABEL3 yields its trailing numeric factor."""
    assert _voyager_if_factor(_LABEL3) == 1.0


def test_if_factor_missing_phrase_raises() -> None:
    """A LABEL3 lacking the fixed phrase raises a clear ValueError."""
    with pytest.raises(ValueError, match='Unexpected Voyager LABEL3 format'):
        _voyager_if_factor('SOME OTHER LABEL 1.0')


def test_if_factor_non_numeric_remainder_raises() -> None:
    """A LABEL3 whose remainder is not numeric raises a clear ValueError."""
    bad = 'FOR (I/F)*10000., MULTIPLY DN VALUE BY    NOT_A_NUMBER'
    with pytest.raises(ValueError, match='Unexpected Voyager LABEL3 format'):
        _voyager_if_factor(bad)


def test_if_factor_non_string_raises() -> None:
    """A non-string LABEL3 value raises a clear ValueError."""
    with pytest.raises(ValueError, match='Unexpected Voyager LABEL3 format'):
        _voyager_if_factor(None)


@pytest.mark.parametrize(
    ('spacecraft', 'detector', 'expected'),
    [('1', 'NAC', 7.769), ('1', 'WAC', 5.734), ('2', 'NAC', 8.240), ('2', 'WAC', 6.297)],
)
def test_clear_one_second_limit_is_each_cameras_calibrated_value(
    spacecraft: str, detector: str, expected: float
) -> None:
    """At 1 s through CLEAR at low gain each camera reaches its own limit.

    Each is 2.5 log10 of the camera's V = 0 rate over its dimmest visible DN, plus 1 mag:
    8.66e4 / 110 for the Voyager 2 narrow-angle camera, for example.
    """
    obs = _star_obs(spacecraft=spacecraft, detector=detector)
    assert obs.star_max_usable_vmag() == pytest.approx(expected, abs=0.001)


def test_ten_times_the_exposure_deepens_the_limit_by_two_and_a_half_magnitudes() -> None:
    """The limit moves with the exposure by 2.5 log10 of its ratio."""
    deeper = _star_obs(texp=10.0).star_max_usable_vmag() - _star_obs().star_max_usable_vmag()
    assert deeper == pytest.approx(2.5, abs=1e-9)


def test_high_gain_deepens_the_limit_by_its_dn_factor() -> None:
    """At high gain a Voyager 2 narrow-angle star puts 4.24 times the DN in the image."""
    shift = _star_obs(gain='HI').star_max_usable_vmag() - _star_obs().star_max_usable_vmag()
    assert shift == pytest.approx(2.5 * math.log10(4.24), abs=1e-9)


def test_a_filter_brightens_the_limit_by_its_magnitude_offset() -> None:
    """UV passes less light than CLEAR; its offset brightens the limit."""
    shift = _star_obs(filter_name='UV').star_max_usable_vmag() - _star_obs().star_max_usable_vmag()
    assert shift == pytest.approx(-3.98, abs=1e-9)


def test_a_non_positive_exposure_admits_no_star() -> None:
    """A zero-length exposure records no starlight."""
    assert _star_obs(texp=0.0).star_max_usable_vmag() == -math.inf


def test_a_filter_the_camera_lacks_is_refused() -> None:
    """The narrow-angle camera has no SODIUM filter, and the error names it."""
    with pytest.raises(ValueError, match="no filter_mag_offset for the NAC filter 'SODIUM'"):
        _star_obs(filter_name='SODIUM').star_max_usable_vmag()


def test_a_label_without_a_gain_state_is_refused() -> None:
    """The limit needs the gain state; a LAB03 record that states none names the gap."""
    with pytest.raises(ValueError, match='NAC label states no gain state in LAB03'):
        _star_obs(gain='XX').star_max_usable_vmag()


@pytest.mark.parametrize(
    ('spacecraft', 'detector', 'expected'),
    [('1', 'NAC', 3.0), ('1', 'WAC', 3.0), ('2', 'NAC', 1.2), ('2', 'WAC', 1.1)],
)
def test_the_star_width_is_each_cameras_own(
    spacecraft: str, detector: str, expected: float
) -> None:
    """Each camera's star PSF has the sigma of its own entry."""
    obs = _star_obs(spacecraft=spacecraft, detector=detector)
    assert obs.star_psf_sigma() == pytest.approx(expected, abs=1e-12)


@REQUIRES_EXTERNAL_DATA
def test_voyager_iss_reports_spacecraft_digit() -> None:
    """The spacecraft digit is read from the image label."""
    obs = obstvgiss.ObsVoyagerISS.from_file(URL_VOYAGER_ISS_IO_01)
    assert obs.spacecraft_digit == '2'


def test_the_published_counts_are_fractional_leading_units_with_no_midtime() -> None:
    """The label's counts are published in the clock's leading units, with no midtime.

    C1480500_GEOMED's label counts, 14804:59:784 and 14805:00:001, cross a leading
    unit.  A frame is 1/60 of a leading unit, and a line, counted from 1, is 1/800 of a
    frame.  The stop count is the readout frame's, so the two do not bracket the
    exposure.
    """
    obs = bare_observation(
        ObsVoyagerISS,
        VicarLabelStandIn(LAB02=_LAB02_V1),
        detector='WAC',
        filter='CLEAR',
        _label_clock_counts=('14804:59:784', '14805:00:001'),
    )
    start, midtime, end = published_clock_counts(obs)
    assert start == pytest.approx(14804 + 59 / 60 + (784 - 1) / (60 * 800), abs=1e-9)
    assert midtime is None
    assert end == pytest.approx(14805.0, abs=1e-9)


@REQUIRES_EXTERNAL_DATA
def test_voyager_iss_metadata_clock_counts_are_the_pds3_labels() -> None:
    """The published clock counts are those of the PDS3 label beside the image.

    The IO test image's PDS3 label gives 20621:32:798 and 20621:33:001; the VICAR label
    the observation keeps carries neither.
    """
    obs = obstvgiss.ObsVoyagerISS.from_file(URL_VOYAGER_ISS_IO_01)
    meta = obs.get_public_metadata()
    counts = [meta[key] for key in ('start_time_sclk', 'midtime_sclk', 'end_time_sclk')]
    assert counts == [
        pytest.approx(20621 + 32 / 60 + (798 - 1) / (60 * 800), abs=1e-9),
        None,
        pytest.approx(20621 + 33 / 60, abs=1e-9),
    ]
