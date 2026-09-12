"""Tests for ``spindoctor.obs.obs_inst_cassini_iss.ObsCassiniISS``."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from tests.config import REQUIRES_EXTERNAL_DATA, URL_CASSINI_ISS_RHEA_01

import spindoctor.obs.obs_inst_cassini_iss as obstcoiss
from spindoctor.obs.obs_inst_cassini_iss import ObsCassiniISS

# The marker is applied per test rather than module-wide: the shutter-mode
# label tests build a bare observation and fetch nothing, so they run even
# where the external trees are absent.


def _obs_with_label(label: dict[str, Any]) -> ObsCassiniISS:
    """Build a bare ObsCassiniISS carrying only the given label dict.

    ``shutter_mode`` is a pure function of ``self.dict``, so a fully
    constructed observation (and an external image fetch) is unnecessary for
    testing it.
    """
    obs = object.__new__(ObsCassiniISS)
    obs.dict = label
    return obs


@REQUIRES_EXTERNAL_DATA
def test_cassini_iss_basic() -> None:
    obs = obstcoiss.ObsCassiniISS.from_file(URL_CASSINI_ISS_RHEA_01)
    assert obs.midtime == 196177280.54761


@REQUIRES_EXTERNAL_DATA
def test_cassini_iss_calib_filename_selects_calib_inst_config() -> None:
    """A ``_CALIB.IMG`` filename selects the calibrated_if config block.

    Regression: CALIB I/F products were previously loaded with the raw_dn
    config block, causing the image-quality classifier to flag every
    CALIB image as ``blank`` (max I/F < 1.0 against the 5.0 DN floor).
    """
    obs = obstcoiss.ObsCassiniISS.from_file(URL_CASSINI_ISS_RHEA_01)
    assert obs.inst_config is not None
    assert obs.inst_config['data_units'] == 'calibrated_if'
    # Calibrated_if blocks expose the I/F-keyed thresholds, not DN-keyed
    # ones.  Saturation is intentionally NOT keyed in I/F (Phase 10 §F):
    # calibration is exposure-/filter-/gain-dependent, so a single I/F
    # threshold cannot identify physically saturated pixels.  The
    # orchestrator leaves the per-pixel saturation mask empty for
    # calibrated_if input.
    iqt = obs.inst_config['image_quality_thresholds']
    assert 'saturation_threshold_if' not in iqt
    assert 'blank_max_if' in iqt
    assert 'noisy_threshold_if' in iqt


@REQUIRES_EXTERNAL_DATA
def test_cassini_iss_reports_shutter_mode() -> None:
    """The shutter mode is read from the image label.

    The Rhea test frame was taken with both cameras exposed at once, so it
    reports the simultaneous mode rather than a single-camera one.
    """
    obs = obstcoiss.ObsCassiniISS.from_file(URL_CASSINI_ISS_RHEA_01)
    assert obs.shutter_mode == 'BOTSIM'


def test_shutter_mode_absent_from_the_label_reads_as_none() -> None:
    """A label carrying no SHUTTER_MODE_ID reports no shutter mode."""
    assert _obs_with_label({}).shutter_mode is None


def test_shutter_mode_null_label_value_reads_as_none() -> None:
    """A SHUTTER_MODE_ID present but null reports no shutter mode, not 'None'."""
    assert _obs_with_label({'SHUTTER_MODE_ID': None}).shutter_mode is None


def test_shutter_mode_non_text_label_value_is_refused() -> None:
    """A non-string SHUTTER_MODE_ID raises rather than serializing the object.

    ``str()`` would render any object without complaint, and the result would
    pass downstream as a legible shutter mode.
    """
    with pytest.raises(ValueError, match='SHUTTER_MODE_ID is not text'):
        _ = _obs_with_label({'SHUTTER_MODE_ID': 42}).shutter_mode


def test_the_midtime_clock_count_is_halfway_in_whole_ticks_rounded_down() -> None:
    """The clock counts are published as label text, the midpoint in whole ticks.

    N1459552248_1_CALIB's label counts are 1459552247.012 and 1459552248.137.  Their
    seconds fields sum to an odd number, so the mean of the two as decimal numbers,
    1459552247.5745, reads as 574 ticks in a field that holds 256.  Halfway in ticks is
    202.5 past the start's second, which rounds down to 202.
    """
    obs = _obs_with_label(
        {
            'SPACECRAFT_CLOCK_START_COUNT': '1459552247.012',
            'SPACECRAFT_CLOCK_STOP_COUNT': '1459552248.137',
        }
    )
    obs.detector = 'NAC'
    obs.image_url = '/holdings/N1459552248_1_CALIB.IMG'
    obs.abspath = Path('/cache/N1459552248_1_CALIB.IMG')
    obs.cadence = SimpleNamespace(time=(0.0, 1.5), midtime=0.75)
    obs.texp = 1.5
    obs._data_shape_uv = (1024, 1024)
    obs.filter1, obs.filter2 = 'CL1', 'CL2'
    obs.sampling = 'FULL'
    obs.gain_mode = 2
    public = obs.get_public_metadata()
    assert (public['start_time_scet'], public['midtime_scet'], public['end_time_scet']) == (
        '1459552247.012',
        '1459552247.202',
        '1459552248.137',
    )
