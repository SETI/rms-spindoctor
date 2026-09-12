import math

import pytest
from tests.config import REQUIRES_EXTERNAL_DATA, URL_VOYAGER_ISS_IO_01
from tests.spindoctor.inst.conftest import (
    VicarLabelStandIn,
    bare_observation,
    published_clock_counts,
)

import spindoctor.obs.obs_inst_voyager_iss as obstvgiss
from spindoctor.obs.obs_inst_voyager_iss import (
    ObsVoyagerISS,
    _voyager_if_factor,
    _voyager_spacecraft_digit,
)

# The marker is applied per test rather than module-wide: the tests of label strings,
# clock counts and the star gate fetch nothing, so they run even where the external
# trees are absent.

# Documented anchor limiting magnitudes (limiting mag at texp = 1 s).
_VOYAGER_NAC_ANCHOR = 8.3
_VOYAGER_WAC_ANCHOR = 5.9


def _make_obs(detector: str, texp: float) -> ObsVoyagerISS:
    """Build a bare ObsVoyagerISS carrying only detector and texp.

    The star-magnitude gate is a pure function of ``self.detector`` and
    ``self.texp``, so a fully constructed observation (and an external image
    fetch) is unnecessary for testing it.
    """
    obs = object.__new__(ObsVoyagerISS)
    obs.detector = detector
    obs.texp = texp
    return obs


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


def test_star_max_usable_vmag_nac_anchor_at_unit_exposure() -> None:
    """The NAC limiting magnitude equals its anchor at texp = 1 s."""
    obs = _make_obs('NAC', 1.0)
    assert obs.star_max_usable_vmag() == pytest.approx(_VOYAGER_NAC_ANCHOR, abs=1e-6)


def test_star_max_usable_vmag_wac_anchor_at_unit_exposure() -> None:
    """The WAC limiting magnitude equals its anchor at texp = 1 s."""
    obs = _make_obs('WAC', 1.0)
    assert obs.star_max_usable_vmag() == pytest.approx(_VOYAGER_WAC_ANCHOR, abs=1e-6)


def test_star_max_usable_vmag_gains_one_mag_per_pogson_ratio() -> None:
    """A 2.512x longer exposure deepens the limit by ~1 mag."""
    base = _make_obs('NAC', 1.0).star_max_usable_vmag()
    deeper = _make_obs('NAC', 2.512).star_max_usable_vmag()
    assert deeper - base == pytest.approx(1.0, abs=1e-3)


def test_star_max_usable_vmag_in_sane_range() -> None:
    """The NAC limiting magnitude stays within a sane 3..15 range."""
    vmag = _make_obs('NAC', 15.0).star_max_usable_vmag()
    assert 3.0 <= vmag <= 15.0


def test_star_max_usable_vmag_is_finite() -> None:
    """The NAC limiting magnitude is finite."""
    vmag = _make_obs('NAC', 15.0).star_max_usable_vmag()
    assert math.isfinite(vmag)


def test_star_max_usable_vmag_non_positive_exposure_returns_anchor() -> None:
    """A non-positive exposure falls back to the anchor magnitude."""
    obs = _make_obs('NAC', 0.0)
    assert obs.star_max_usable_vmag() == pytest.approx(_VOYAGER_NAC_ANCHOR, abs=1e-6)


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
    assert published_clock_counts(obs) == [
        pytest.approx(14804 + 59 / 60 + (784 - 1) / (60 * 800), abs=1e-9),
        None,
        pytest.approx(14805.0, abs=1e-9),
    ]


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
