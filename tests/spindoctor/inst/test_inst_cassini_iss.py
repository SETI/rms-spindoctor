"""Tests for ``spindoctor.obs.obs_inst_cassini_iss.ObsCassiniISS``."""

from typing import Any

import pytest
from tests.config import REQUIRES_EXTERNAL_DATA, URL_CASSINI_ISS_RHEA_01
from tests.spindoctor.inst.conftest import (
    VicarLabelStandIn,
    bare_observation,
    published_clock_counts,
)

import spindoctor.obs.obs_inst_cassini_iss as obstcoiss
from spindoctor.obs.obs_inst_cassini_iss import ObsCassiniISS, _sclk_count

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


def _cassini_observation(label: VicarLabelStandIn) -> ObsCassiniISS:
    """Build a bare narrow-angle observation whose public metadata can be read.

    Parameters:
        label: The image's VICAR label items.

    Returns:
        The observation.
    """
    return bare_observation(
        ObsCassiniISS,
        label,
        detector='NAC',
        filter1='CL1',
        filter2='CL2',
        sampling='FULL',
        gain_mode=2,
    )


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


def test_the_published_counts_are_fractional_seconds_and_their_exact_mean() -> None:
    """The label's counts are published as clock seconds, and their mean as the midtime.

    N1459552248_1_CALIB's exposure crosses a second: its label counts, 1459552247.012
    and 1459552248.137, are 1459552247 + 12/256 and 1459552248 + 137/256 seconds, and
    the midtime count is exactly halfway between them.
    """
    label = VicarLabelStandIn(
        SPACECRAFT_CLOCK_START_COUNT='1459552247.012',
        SPACECRAFT_CLOCK_STOP_COUNT='1459552248.137',
    )
    assert published_clock_counts(_cassini_observation(label)) == [
        1459552247.046875,
        1459552247.791015625,
        1459552248.53515625,
    ]


def test_a_tick_field_that_lost_its_trailing_zeros_is_padded_back() -> None:
    """A count whose tick field lost its trailing zeros is read with them restored.

    The COISS index writes N1347929382_3's start count, 1347929382.110, as
    1347929382.11.
    """
    assert _sclk_count('1347929382.11') == 1347929382 + 110 / 256


def test_a_label_without_clock_counts_publishes_null_counts() -> None:
    """A label carrying no clock counts publishes all three as null.

    Some labels, W1294561143_1_CALIB's among them, carry neither
    SPACECRAFT_CLOCK_START_COUNT nor SPACECRAFT_CLOCK_STOP_COUNT.
    """
    counts = published_clock_counts(_cassini_observation(VicarLabelStandIn()))
    assert counts == [None, None, None]
