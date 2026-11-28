"""Tests for the star techniques' tuning types.

The schema core checks each key's type and range, and the techniques section's suite
checks that every packaged block parses.  What is tested here is the rules a star
tuning type adds to its fields.
"""

import re

import pytest
from tests.packaged_sections import TEST_OVERRIDE_LABEL, NavConfigWith

from spindoctor.schema import SchemaError


def test_an_even_psf_refine_box_is_refused(nav_config_with: NavConfigWith) -> None:
    """A PSF-fit box with an even side, which lacks a center pixel, is refused at load."""
    overrides = {'techniques': {'star_field_from_catalog': {'tuning': {'psf_refine_box_px': 10}}}}
    expected = (
        f'{TEST_OVERRIDE_LABEL}: techniques.star_field_from_catalog.tuning.psf_refine_box_px: '
        'expected an odd number of pixels'
    )
    with pytest.raises(SchemaError, match=re.escape(expected)):
        nav_config_with(overrides)


def test_an_inverted_isolation_annulus_is_refused(nav_config_with: NavConfigWith) -> None:
    """An inner isolation radius above the outer one is refused, and the message has both keys."""
    overrides = {'techniques': {'star_unique_match': {'tuning': {'source_isolation_inner_px': 7}}}}
    expected = (
        'techniques.star_unique_match.tuning.source_isolation_inner_px: '
        'source_isolation_inner_px must not exceed source_isolation_outer_px, got 7 and 6'
    )
    with pytest.raises(SchemaError, match=re.escape(expected)):
        nav_config_with(overrides)


def test_a_max_peak_sharpness_below_one_is_refused(nav_config_with: NavConfigWith) -> None:
    """A peak ratio is at least 1, so a smaller one is refused."""
    overrides = {'techniques': {'star_refine': {'tuning': {'max_peak_sharpness': 0.5}}}}
    expected = 'techniques.star_refine.tuning.max_peak_sharpness: expected a number of at least 1'
    with pytest.raises(SchemaError, match=re.escape(expected)):
        nav_config_with(overrides)


def test_a_count_below_three_asks_for_an_integer(nav_config_with: NavConfigWith) -> None:
    """A count below 3 is refused in words that ask for an integer."""
    overrides = {'techniques': {'star_field_from_catalog': {'tuning': {'max_sources': 2}}}}
    expected = (
        'techniques.star_field_from_catalog.tuning.max_sources: '
        'expected an integer of at least 3, got 2'
    )
    with pytest.raises(SchemaError, match=re.escape(expected)):
        nav_config_with(overrides)
