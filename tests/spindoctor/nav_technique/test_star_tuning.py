"""Tests for the star techniques' tuning types.

The schema core checks each key's type and range, and the techniques section's suite
checks that every packaged block parses.  What is tested here is the rules a star
tuning type adds to its fields.
"""

import re
from pathlib import Path

import pytest

from spindoctor.config import Config
from spindoctor.schema import SchemaError


def test_an_even_psf_refine_box_is_refused(tmp_path: Path) -> None:
    """A PSF-fit box with an even side, which has no center pixel, is refused at load."""
    user_path = tmp_path / 'user_config.yaml'
    user_path.write_text(
        'techniques:\n  star_field_from_catalog:\n    tuning:\n      psf_refine_box_px: 10\n',
        encoding='utf-8',
    )
    expected = (
        f'{user_path}: techniques.star_field_from_catalog.tuning.psf_refine_box_px: '
        'expected an odd number of pixels'
    )
    with pytest.raises(SchemaError, match=re.escape(expected)):
        Config().update_config(user_path)


def test_an_inverted_isolation_annulus_is_refused(tmp_path: Path) -> None:
    """An inner isolation radius above the outer one is refused, naming both keys."""
    user_path = tmp_path / 'user_config.yaml'
    user_path.write_text(
        'techniques:\n  star_unique_match:\n    tuning:\n      source_isolation_inner_px: 7\n',
        encoding='utf-8',
    )
    expected = (
        'techniques.star_unique_match.tuning.source_isolation_inner_px: '
        'source_isolation_inner_px must not exceed source_isolation_outer_px, got 7 and 6'
    )
    with pytest.raises(SchemaError, match=re.escape(expected)):
        Config().update_config(user_path)


def test_a_max_peak_sharpness_below_one_is_refused(tmp_path: Path) -> None:
    """A peak ratio is at least 1, so a smaller one is refused."""
    user_path = tmp_path / 'user_config.yaml'
    user_path.write_text(
        'techniques:\n  star_refine:\n    tuning:\n      max_peak_sharpness: 0.5\n',
        encoding='utf-8',
    )
    expected = 'techniques.star_refine.tuning.max_peak_sharpness: expected a number of at least 1'
    with pytest.raises(SchemaError, match=re.escape(expected)):
        Config().update_config(user_path)


def test_a_count_below_three_asks_for_an_integer(tmp_path: Path) -> None:
    """A count below 3 is refused in words that ask for an integer."""
    user_path = tmp_path / 'user_config.yaml'
    user_path.write_text(
        'techniques:\n  star_field_from_catalog:\n    tuning:\n      max_sources: 2\n',
        encoding='utf-8',
    )
    expected = (
        'techniques.star_field_from_catalog.tuning.max_sources: '
        'expected an integer of at least 3, got 2'
    )
    with pytest.raises(SchemaError, match=re.escape(expected)):
        Config().update_config(user_path)
