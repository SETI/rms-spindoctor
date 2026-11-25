import math

import numpy as np
import pytest
from tests.config import REQUIRES_EXTERNAL_DATA, URL_VOYAGER_ISS_IO_01
from tests.spindoctor.inst.conftest import bare_observation, published_clock_counts

import spindoctor.obs.obs_inst_voyager_iss as obstvgiss
from spindoctor.obs.obs_inst_voyager_iss import ObsVoyagerISS, _if_data

# The marker is applied per test rather than module-wide: the tests of the I/F
# conversion, clock counts and the star gate fetch nothing, so they run even where the
# external trees are absent.

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


@REQUIRES_EXTERNAL_DATA
def test_voyager_iss_basic() -> None:
    obs = obstvgiss.ObsVoyagerISS.from_file(URL_VOYAGER_ISS_IO_01)
    assert obs.midtime == -646429822.8760977


@REQUIRES_EXTERNAL_DATA
def test_voyager_iss_metadata_spacecraft_lid() -> None:
    """Public metadata derives the instrument-host LID from the host's spacecraft."""
    obs = obstvgiss.ObsVoyagerISS.from_file(URL_VOYAGER_ISS_IO_01)
    meta = obs.get_public_metadata()
    # The IO test image is a Voyager 2 frame.
    assert meta['instrument_host_lid'].endswith('spacecraft.vg2')


def test_the_data_are_scaled_to_if_by_the_hosts_factor() -> None:
    """The data are multiplied by the host's ``if_scale``, and by nothing else."""
    data = np.array([[100, 200]], dtype=np.uint8)
    if_data = _if_data(data, {'if_scale': 0.0003345, 'spacecraft': 1, 'planet': 'SATURN'})
    assert if_data == pytest.approx(np.array([[0.03345, 0.0669]]), rel=1e-12)


def test_an_image_with_no_if_factor_is_refused() -> None:
    """A product for which no label the host read gives an I/F factor is refused."""
    data = np.array([[100]], dtype=np.uint8)
    with pytest.raises(ValueError, match='no label of the image gives an I/F factor'):
        _if_data(data, {'spacecraft': 2, 'planet': 'JUPITER'})


def test_the_spacecraft_digit_is_the_hosts_spacecraft() -> None:
    """The spacecraft digit is the host's ``spacecraft`` subfield, as text."""
    obs = object.__new__(ObsVoyagerISS)
    obs.spacecraft = 1
    assert obs.spacecraft_digit == '1'


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
    """The spacecraft digit comes from the host."""
    obs = obstvgiss.ObsVoyagerISS.from_file(URL_VOYAGER_ISS_IO_01)
    assert obs.spacecraft_digit == '2'


def test_the_published_counts_are_fractional_leading_units_with_no_midtime() -> None:
    """The host's counts are published in the clock's leading units, with no midtime.

    C1480500_GEOMED's label counts, 14804:59:784 and 14805:00:001, cross a leading
    unit.  A frame is 1/60 of a leading unit, and a line, counted from 1, is 1/800 of a
    frame.  The stop count is the readout frame's, so the two do not bracket the
    exposure.
    """
    obs = bare_observation(
        ObsVoyagerISS,
        {},
        spacecraft=1,
        detector='WAC',
        subfields={
            'filter': 'CLEAR',
            'spacecraft_clock_start_count': '14804:59:784',
            'spacecraft_clock_stop_count': '14805:00:001',
        },
    )
    start, midtime, end = published_clock_counts(obs)
    assert start == pytest.approx(14804 + 59 / 60 + (784 - 1) / (60 * 800), abs=1e-9)
    assert midtime is None
    assert end == pytest.approx(14805.0, abs=1e-9)


def test_an_image_read_without_its_label_publishes_no_start_count() -> None:
    """The VICAR header gives only the stop count, so without the label there is no start."""
    obs = bare_observation(
        ObsVoyagerISS,
        {},
        spacecraft=1,
        detector='WAC',
        subfields={'filter': 'CLEAR', 'spacecraft_clock_stop_count': '14805:00:001'},
    )
    assert published_clock_counts(obs)[0] is None


def test_an_image_with_no_filter_publishes_no_filters() -> None:
    """When no label gives the filter, no filter is published."""
    obs = bare_observation(ObsVoyagerISS, {}, spacecraft=2, detector='NAC', subfields={})
    assert obs.get_public_metadata()['filters'] == []


@REQUIRES_EXTERNAL_DATA
def test_voyager_iss_metadata_clock_counts_are_the_pds3_labels() -> None:
    """The published clock counts are those of the PDS3 label beside the image.

    The IO test image's PDS3 label gives 20621:32:798 and 20621:33:001.  The VICAR
    header gives the stop count alone, so the start count comes from the host's read of
    the label.
    """
    obs = obstvgiss.ObsVoyagerISS.from_file(URL_VOYAGER_ISS_IO_01)
    meta = obs.get_public_metadata()
    counts = [meta[key] for key in ('start_time_sclk', 'midtime_sclk', 'end_time_sclk')]
    assert counts == [
        pytest.approx(20621 + 32 / 60 + (798 - 1) / (60 * 800), abs=1e-9),
        None,
        pytest.approx(20621 + 33 / 60, abs=1e-9),
    ]
