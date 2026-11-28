"""Tests for the typed ``rings`` section and its ring catalogs.

The schema core checks each key's type and range, and is tested in its own suite.  What
is tested here is that the packaged section parses, each cross-field rule of a catalog,
its features, and their mode entries, the check that a time can be read, and the rule
that a catalog's planet is a planet of the ``planets`` section.
"""

import re
from typing import Any

import pytest
from tests.packaged_sections import TEST_OVERRIDE_LABEL, NavConfigWith

from spindoctor.config import Config
from spindoctor.config.sections_rings import RingCatalog, RingFeatureEntry, RingsSection
from spindoctor.schema import SchemaError, parse

_SOURCE = 'test.yaml'


def _base(a: float, **changes: Any) -> dict[str, Any]:
    """Return a base-orbit mode entry, with the given keys changed.

    Parameters:
        a: The semi-major axis, in km.
        **changes: Keys to change or add.
    """
    return {'mode': 1, 'a': a, 'rms': 0.5, 'ae': 0.0, 'long_peri': 0.0, 'rate_peri': 0.0} | changes


def _perturbation(**changes: Any) -> dict[str, Any]:
    """Return a perturbation mode entry, with the given keys changed.

    Parameters:
        **changes: Keys to change or add.
    """
    return {'mode': 2, 'amplitude': 1.0, 'phase': 0.0, 'pattern_speed': 0.0} | changes


def _ringlet(inner_a: float, outer_a: float, **changes: Any) -> dict[str, Any]:
    """Return a ringlet with both edges, with the given keys changed.

    Parameters:
        inner_a: The inner edge's semi-major axis, in km.
        outer_a: The outer edge's semi-major axis, in km.
        **changes: Keys to change or add.
    """
    return {
        'feature_type': 'RINGLET',
        'inner_data': [_base(inner_a)],
        'outer_data': [_base(outer_a)],
    } | changes


def _catalog(features: dict[str, Any], **changes: Any) -> dict[str, Any]:
    """Return a ring catalog of ``features``, with the given keys changed.

    Parameters:
        features: The catalog's features, by key.
        **changes: Keys to change or add.
    """
    return {
        'epoch': '2008-01-01 12:00:00',
        'fade_width_pix': 100.0,
        'min_allowed_fade_width_pix': 3.0,
        'min_feature_pixels': 2.0,
        'features': features,
    } | changes


def _refused(cls: type, data: dict[str, Any], expected: str) -> None:
    """Assert that ``data`` is refused as ``cls`` with the problem ``expected``.

    Parameters:
        cls: The schema type.
        data: The mapping to parse.
        expected: The problem's key path and message, after the source.
    """
    with pytest.raises(SchemaError, match=re.escape(f'{_SOURCE}: {expected}')):
        parse(cls, data, source=_SOURCE)


def test_packaged_rings_section_parses() -> None:
    """The packaged ``rings`` section parses, with a catalog for Saturn."""
    rings = Config().rings
    assert isinstance(rings, RingsSection)
    assert isinstance(rings.ring_features['SATURN'], RingCatalog)


def test_a_base_orbit_entry_is_mode_one() -> None:
    """An entry that sets ``a`` is the base orbit, so its mode must be 1."""
    feature = {'feature_type': 'GAP', 'inner_data': [_base(1000.0, mode=2)]}
    _refused(
        RingFeatureEntry,
        feature,
        'inner_data[0].mode: an entry that sets a is the base orbit, which is mode 1, not mode 2',
    )


def test_a_base_orbit_may_leave_out_rms() -> None:
    """A base orbit without a fitted residual parses, with no ``rms``."""
    entry = _base(1000.0)
    del entry['rms']
    data = {'feature_type': 'GAP', 'inner_data': [entry]}
    feature = parse(RingFeatureEntry, data, source=_SOURCE)
    assert feature.inner_data is not None
    assert feature.inner_data[0].rms is None


@pytest.mark.parametrize(
    ('modes', 'expected'),
    [
        (
            [_base(1000.0), {'mode': 2, 'phase': 0.0, 'pattern_speed': 0.0}],
            'inner_data[1]: a perturbation entry must set amplitude, phase, and pattern_speed, but '
            'this one leaves out amplitude',
        ),
        (
            [_base(1000.0, phase=3.0)],
            'inner_data[0]: a base-orbit entry may set only a, rms, ae, long_peri, and rate_peri, '
            'but this one also sets phase',
        ),
        (
            [{'mode': 1, 'rms': 0.5, 'amplitude': 1.0, 'phase': 0.0, 'pattern_speed': 0.0}],
            'inner_data[0]: this entry mixes the two kinds: it leaves out a but sets a key of a '
            'base orbit; a base orbit sets a, rms, ae, long_peri, and rate_peri, and a '
            'perturbation sets amplitude, phase, and pattern_speed',
        ),
    ],
    ids=['missing', 'stray', 'mixed'],
)
def test_a_mode_entry_sets_the_keys_of_its_kind(modes: list[dict[str, Any]], expected: str) -> None:
    """A mode entry sets every key of its kind and no key of the other kind.

    Parameters:
        modes: The edge's mode entries.
        expected: The problem's key path and message.
    """
    _refused(RingFeatureEntry, {'feature_type': 'GAP', 'inner_data': modes}, expected)


@pytest.mark.parametrize(
    ('modes', 'count'),
    [([_perturbation()], 0), ([_base(1000.0), _base(1001.0)], 2)],
    ids=['none', 'two'],
)
def test_an_edge_has_exactly_one_base_orbit(modes: list[dict[str, Any]], count: int) -> None:
    """An edge's mode list holds exactly one base-orbit entry.

    Parameters:
        modes: The edge's mode entries.
        count: How many of them are base orbits.
    """
    _refused(
        RingFeatureEntry,
        {'feature_type': 'RINGLET', 'outer_data': modes},
        f'outer_data: an edge must have exactly one base-orbit entry (one that sets a), but '
        f'this one has {count}',
    )


def test_a_feature_has_at_least_one_edge() -> None:
    """A feature that sets neither edge is refused."""
    _refused(
        RingFeatureEntry,
        {'feature_type': 'RINGLET', 'name': 'Nothing'},
        'a feature needs an inner edge, an outer edge, or both; set inner_data, outer_data, '
        'or both',
    )


def test_a_feature_starts_before_it_ends() -> None:
    """A feature's start date must come before its end date."""
    feature = _ringlet(1000.0, 1001.0, start_date='2010-01-01', end_date='2009-01-01')
    _refused(
        RingFeatureEntry,
        feature,
        'end_date: start_date (2010-01-01) must be before end_date (2009-01-01)',
    )


def test_a_feature_cannot_start_when_it_ends() -> None:
    """A feature whose start date equals its end date is refused."""
    feature = _ringlet(1000.0, 1001.0, start_date='2009-01-01', end_date='2009-01-01')
    _refused(
        RingFeatureEntry,
        feature,
        'end_date: start_date (2009-01-01) must be before end_date (2009-01-01)',
    )


def test_a_time_that_cannot_be_read_is_refused() -> None:
    """A catalog's epoch must be a UTC time in ISO 8601 form."""
    _refused(
        RingCatalog,
        _catalog({}, epoch='not a date'),
        "epoch: 'not a date' is not a UTC time: unrecognized ISO date format",
    )


def test_the_least_fade_is_no_wider_than_the_fade() -> None:
    """``min_allowed_fade_width_pix`` must not exceed ``fade_width_pix``."""
    _refused(
        RingCatalog,
        _catalog({}, fade_width_pix=5.0, min_allowed_fade_width_pix=6.0),
        'min_allowed_fade_width_pix: min_allowed_fade_width_pix (6.0) must not exceed '
        'fade_width_pix (5.0)',
    )


def test_the_least_fade_may_equal_the_fade() -> None:
    """``min_allowed_fade_width_pix`` may be as wide as ``fade_width_pix``."""
    catalog = parse(
        RingCatalog,
        _catalog({}, fade_width_pix=5.0, min_allowed_fade_width_pix=5.0),
        source=_SOURCE,
    )
    assert catalog.min_allowed_fade_width_pix == catalog.fade_width_pix


def test_two_features_at_the_same_radii_and_time_are_refused() -> None:
    """Two features whose radii and dates overlap are refused; the message gives both."""
    features = {
        'alpha': _ringlet(1000.0, 1010.0, start_date='2008-01-01', end_date='2011-01-01'),
        'beta': _ringlet(1005.0, 1015.0, start_date='2009-01-01'),
    }
    _refused(
        RingCatalog,
        _catalog(features),
        "features.beta: 'alpha' and 'beta' cover overlapping radii over overlapping dates; "
        'change the dates so that they do not overlap',
    )


def test_features_whose_radii_touch_are_refused() -> None:
    """Features whose radial extents share one radius overlap in radius."""
    features = {
        'alpha': _ringlet(1000.0, 1010.0, start_date='2008-01-01'),
        'beta': _ringlet(1010.0, 1020.0, start_date='2009-01-01'),
    }
    _refused(
        RingCatalog,
        _catalog(features),
        "features.beta: 'alpha' and 'beta' cover overlapping radii over overlapping dates",
    )


@pytest.mark.parametrize(
    'features',
    [
        {'alpha': _ringlet(1000.0, 1010.0), 'beta': _ringlet(1000.0, 1010.0)},
        {
            'alpha': _ringlet(1000.0, 1010.0, end_date='2010-01-01'),
            'beta': _ringlet(1000.0, 1010.0, start_date='2010-01-01'),
        },
        {
            'alpha': _ringlet(1000.0, 1010.0, start_date='2008-01-01'),
            'beta': _ringlet(2000.0, 2010.0, start_date='2008-01-01'),
        },
        {
            'alpha': _ringlet(1000.0, 1010.0),
            'beta': _ringlet(1000.0, 1010.0, start_date='2008-01-01'),
        },
    ],
    ids=['undated', 'consecutive_dates', 'apart', 'undated_beside_dated'],
)
def test_features_that_do_not_overlap_are_accepted(features: dict[str, Any]) -> None:
    """Undated features, consecutive dates, separate radii, and an undated feature do not clash.

    Parameters:
        features: The catalog's features.
    """
    catalog = parse(RingCatalog, _catalog(features), source=_SOURCE)
    assert list(catalog.features) == list(features)


def test_a_ring_catalog_planet_not_in_the_planets_section_is_refused(
    nav_config_with: NavConfigWith,
) -> None:
    """A catalog's planet must be a planet of ``planets``, written in upper case."""
    catalog = {
        'epoch': '2008-01-01 12:00:00',
        'fade_width_pix': 100.0,
        'min_allowed_fade_width_pix': 3.0,
        'min_feature_pixels': 2.0,
        'features': {},
    }
    expected = (
        f"{TEST_OVERRIDE_LABEL}: rings.ring_features.saturn: 'saturn' is not a planet of the "
        'planets section; a planet is written in upper case, as SPICE spells it'
    )
    with pytest.raises(SchemaError, match=re.escape(expected)):
        nav_config_with({'rings': {'ring_features': {'saturn': catalog}}})
