"""Unit tests for ``RingFeature`` and ``validate_no_date_overlaps``.

Tests cover ``from_config`` construction and validation, query methods
(``is_visible_at``, ``is_in_radius_range``, ``all_base_radii``,
``uses_fade_for_edge``, ``uncertainty``, ``edge_labels``), ``render`` dispatch
via mocked backplanes, and cross-feature date overlap validation.
"""

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from spindoctor.nav_model.rings.ring_feature import RingFeature, validate_no_date_overlaps
from spindoctor.nav_model.rings.ring_render_context import RingsRenderContext
from spindoctor.nav_model.rings.ring_types import RingFeatureType
from spindoctor.support.time import utc_to_et

# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------


def _make_edge_data(
    a: float = 100_000.0,
    rms: float = 1.0,
    ae: float = 10.0,
    long_peri: float = 0.0,
    rate_peri: float = 0.0,
) -> list[dict[str, Any]]:
    """Return a YAML edge mode list for a single-mode edge."""
    return [
        {'mode': 1, 'a': a, 'rms': rms, 'ae': ae, 'long_peri': long_peri, 'rate_peri': rate_peri}
    ]


def _make_ringlet_data(
    inner_a: float = 100_000.0,
    outer_a: float = 101_000.0,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    """Return a full ringlet feature dict."""
    d: dict[str, Any] = {
        'feature_type': 'RINGLET',
        'name': 'Test Ringlet',
        'inner_data': _make_edge_data(a=inner_a),
        'outer_data': _make_edge_data(a=outer_a),
    }
    if start_date:
        d['start_date'] = start_date
    if end_date:
        d['end_date'] = end_date
    return d


def _make_gap_data(
    inner_a: float = 100_000.0,
    outer_a: float = 101_000.0,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    """Return a full gap feature dict."""
    d: dict[str, Any] = {
        'feature_type': 'GAP',
        'name': 'Test Gap',
        'inner_data': _make_edge_data(a=inner_a),
        'outer_data': _make_edge_data(a=outer_a),
    }
    if start_date:
        d['start_date'] = start_date
    if end_date:
        d['end_date'] = end_date
    return d


# ---------------------------------------------------------------------------
# RingFeature.from_config -- valid cases
# ---------------------------------------------------------------------------


def test_from_config_ringlet_both_edges() -> None:
    """from_config constructs a valid RINGLET with both edges."""
    feature = RingFeature.from_config('test_ringlet', _make_ringlet_data())
    assert feature.key == 'test_ringlet'
    assert feature.name == 'Test Ringlet'
    assert feature.feature_type is RingFeatureType.RINGLET
    assert feature.inner_edge is not None
    assert feature.outer_edge is not None
    assert feature.inner_edge.base_radius == pytest.approx(100_000.0)
    assert feature.outer_edge.base_radius == pytest.approx(101_000.0)


def test_from_config_gap_both_edges() -> None:
    """from_config constructs a valid GAP with both edges."""
    feature = RingFeature.from_config('test_gap', _make_gap_data())
    assert feature.feature_type is RingFeatureType.GAP
    assert feature.inner_edge is not None
    assert feature.outer_edge is not None


def test_from_config_single_inner_edge() -> None:
    """from_config accepts a feature with only inner_data."""
    data: dict[str, Any] = {
        'feature_type': 'RINGLET',
        'name': 'Single Edge',
        'inner_data': _make_edge_data(a=100_000.0),
    }
    feature = RingFeature.from_config('single', data)
    assert feature.inner_edge is not None
    assert feature.outer_edge is None


def test_from_config_single_outer_edge() -> None:
    """from_config accepts a feature with only outer_data."""
    data: dict[str, Any] = {
        'feature_type': 'GAP',
        'name': 'Outer Only',
        'outer_data': _make_edge_data(a=105_000.0),
    }
    feature = RingFeature.from_config('outer_only', data)
    assert feature.inner_edge is None
    assert feature.outer_edge is not None


def test_from_config_date_range() -> None:
    """from_config stores date range strings."""
    feature = RingFeature.from_config(
        'dated',
        _make_ringlet_data(
            start_date='2008-01-01 12:00:00',
            end_date='2010-01-01 12:00:00',
        ),
    )
    assert feature.start_date == '2008-01-01 12:00:00'
    assert feature.end_date == '2010-01-01 12:00:00'


def test_from_config_no_name() -> None:
    """from_config accepts missing name (name is None)."""
    data: dict[str, Any] = {
        'feature_type': 'RINGLET',
        'inner_data': _make_edge_data(),
    }
    feature = RingFeature.from_config('no_name', data)
    assert feature.name is None


def test_from_config_with_perturbation_mode() -> None:
    """from_config correctly parses mode data with a perturbation mode."""
    data: dict[str, Any] = {
        'feature_type': 'RINGLET',
        'name': 'Perturbed',
        'inner_data': [
            {'mode': 1, 'a': 100_000.0, 'rms': 1.0, 'ae': 5.0, 'long_peri': 0.0, 'rate_peri': 0.0},
            {'mode': 2, 'amplitude': 3.0, 'phase': 45.0, 'pattern_speed': 1.0},
        ],
    }
    feature = RingFeature.from_config('perturbed', data)
    assert feature.inner_edge is not None
    assert len(feature.inner_edge.perturbations) == 1
    assert feature.inner_edge.perturbations[0].mode_num == 2


def test_from_config_perturbation_missing_field_raises() -> None:
    """Perturbation entries must list amplitude, phase, and pattern_speed explicitly."""
    data: dict[str, Any] = {
        'feature_type': 'RINGLET',
        'inner_data': [
            {'mode': 1, 'a': 100_000.0, 'rms': 1.0, 'ae': 0.0, 'long_peri': 0.0, 'rate_peri': 0.0},
            {'mode': 2, 'phase': 0.0, 'pattern_speed': 0.0},
        ],
    }
    with pytest.raises(ValueError) as exc_info:
        RingFeature.from_config('pert_miss', data)
    msg = str(exc_info.value)
    assert "Feature 'pert_miss' inner_data[1]:" in msg
    assert "missing required key 'amplitude'" in msg


# ---------------------------------------------------------------------------
# RingFeature.from_config -- validation errors
# ---------------------------------------------------------------------------


def test_from_config_invalid_feature_type_raises() -> None:
    """from_config raises ValueError for unrecognized feature_type."""
    data: dict[str, Any] = {
        'feature_type': 'BAND',
        'inner_data': _make_edge_data(),
    }
    with pytest.raises(ValueError, match='feature_type'):
        RingFeature.from_config('bad_type', data)


def test_from_config_no_edges_raises() -> None:
    """from_config raises ValueError when neither inner_data nor outer_data present."""
    data: dict[str, Any] = {'feature_type': 'RINGLET', 'name': 'No Edges'}
    with pytest.raises(ValueError, match='edge'):
        RingFeature.from_config('no_edges', data)


def test_from_config_negative_a_raises() -> None:
    """from_config raises ValueError when mode-1 a is non-positive."""
    data: dict[str, Any] = {
        'feature_type': 'RINGLET',
        'inner_data': [
            {'mode': 1, 'a': -100.0, 'rms': 1.0, 'ae': 0.0, 'long_peri': 0.0, 'rate_peri': 0.0}
        ],
    }
    with pytest.raises(ValueError, match=r'RingBaseOrbitMode\.a must be > 0'):
        RingFeature.from_config('neg_a', data)


def test_from_config_base_orbit_wrong_mode_raises() -> None:
    """from_config rejects entries with 'a' when mode is not 1."""
    data: dict[str, Any] = {
        'feature_type': 'RINGLET',
        'inner_data': [
            {'mode': 2, 'a': 100_000.0, 'rms': 1.0, 'ae': 0.0, 'long_peri': 0.0, 'rate_peri': 0.0}
        ],
    }
    with pytest.raises(ValueError, match=r'mode 1|base orbit'):
        RingFeature.from_config('wrong_mode_a', data)


def test_from_config_missing_mode_1_raises() -> None:
    """from_config raises ValueError when mode list has no mode-1 entry."""
    data: dict[str, Any] = {
        'feature_type': 'RINGLET',
        'inner_data': [{'mode': 2, 'amplitude': 5.0, 'phase': 0.0, 'pattern_speed': 0.0}],
    }
    with pytest.raises(ValueError, match='mode 1'):
        RingFeature.from_config('no_mode1', data)


def test_from_config_negative_rms_raises() -> None:
    """from_config raises ValueError when rms is negative."""
    data: dict[str, Any] = {
        'feature_type': 'RINGLET',
        'inner_data': [
            {'mode': 1, 'a': 100_000.0, 'rms': -1.0, 'ae': 0.0, 'long_peri': 0.0, 'rate_peri': 0.0}
        ],
    }
    with pytest.raises(ValueError, match=r'RingBaseOrbitMode\.rms must be >= 0'):
        RingFeature.from_config('neg_rms', data)


def test_from_config_duplicate_base_orbit_raises() -> None:
    """from_config rejects two mode dicts with ``a`` on the same edge."""
    data: dict[str, Any] = {
        'feature_type': 'RINGLET',
        'inner_data': [
            {'mode': 1, 'a': 100_000.0, 'rms': 1.0, 'ae': 0.0, 'long_peri': 0.0, 'rate_peri': 0.0},
            {'mode': 1, 'a': 100_500.0, 'rms': 1.0, 'ae': 0.0, 'long_peri': 0.0, 'rate_peri': 0.0},
        ],
    }
    with pytest.raises(ValueError, match=r'duplicate base-orbit'):
        RingFeature.from_config('dup_base', data)


def test_from_config_empty_mode_list_raises() -> None:
    """from_config raises ValueError when inner_data/outer_data is empty list."""
    data: dict[str, Any] = {
        'feature_type': 'RINGLET',
        'inner_data': [],
    }
    with pytest.raises(ValueError, match=r'non-empty'):
        RingFeature.from_config('empty_modes', data)


# ---------------------------------------------------------------------------
# RingFeature.is_visible_at
# ---------------------------------------------------------------------------

_OBS_TIME = utc_to_et('2009-06-01 00:00:00')
_BEFORE = utc_to_et('2007-01-01 00:00:00')
_AFTER = utc_to_et('2012-01-01 00:00:00')


def test_is_visible_at_no_dates() -> None:
    """Feature with no date range is always visible."""
    feature = RingFeature.from_config('nd', _make_ringlet_data())
    assert feature.is_visible_at(_OBS_TIME)


def test_is_visible_at_within_range() -> None:
    """Feature is visible when obs time is within [start_date, end_date)."""
    feature = RingFeature.from_config(
        'in',
        _make_ringlet_data(start_date='2008-01-01 12:00:00', end_date='2011-01-01 12:00:00'),
    )
    assert feature.is_visible_at(_OBS_TIME)


def test_is_visible_at_before_start() -> None:
    """Feature is not visible when obs time is before start_date."""
    feature = RingFeature.from_config(
        'before',
        _make_ringlet_data(start_date='2008-01-01 12:00:00', end_date='2011-01-01 12:00:00'),
    )
    assert not feature.is_visible_at(_BEFORE)


def test_is_visible_at_after_end() -> None:
    """Feature is not visible when obs time is at or after end_date."""
    feature = RingFeature.from_config(
        'after',
        _make_ringlet_data(start_date='2008-01-01 12:00:00', end_date='2011-01-01 12:00:00'),
    )
    assert not feature.is_visible_at(_AFTER)


def test_is_visible_at_only_start_date() -> None:
    """Feature with only start_date is visible at and after start_date."""
    feature = RingFeature.from_config(
        'start_only',
        _make_ringlet_data(start_date='2008-01-01 12:00:00'),
    )
    assert feature.is_visible_at(_OBS_TIME)
    assert not feature.is_visible_at(_BEFORE)


def test_is_visible_at_only_end_date() -> None:
    """Feature with only end_date is visible before end_date."""
    feature = RingFeature.from_config(
        'end_only',
        _make_ringlet_data(end_date='2011-01-01 12:00:00'),
    )
    assert feature.is_visible_at(_OBS_TIME)
    assert not feature.is_visible_at(_AFTER)


# ---------------------------------------------------------------------------
# RingFeature.is_in_radius_range
# ---------------------------------------------------------------------------


def test_is_in_radius_range_both_edges_in() -> None:
    """Returns True when both edges are in range."""
    feature = RingFeature.from_config(
        'both_in',
        _make_ringlet_data(inner_a=100_000.0, outer_a=101_000.0),
    )
    assert feature.is_in_radius_range(90_000.0, 150_000.0)


def test_is_in_radius_range_inner_only_in() -> None:
    """Returns True when only inner edge is in range (partial visibility)."""
    feature = RingFeature.from_config(
        'inner_in',
        _make_ringlet_data(inner_a=100_000.0, outer_a=200_000.0),
    )
    assert feature.is_in_radius_range(90_000.0, 150_000.0)


def test_is_in_radius_range_outer_only_in() -> None:
    """Returns True when only outer edge is in range (partial visibility)."""
    feature = RingFeature.from_config(
        'outer_in',
        _make_ringlet_data(inner_a=50_000.0, outer_a=101_000.0),
    )
    assert feature.is_in_radius_range(90_000.0, 150_000.0)


def test_is_in_radius_range_neither_in() -> None:
    """Returns False when neither edge is in range."""
    feature = RingFeature.from_config(
        'neither',
        _make_ringlet_data(inner_a=200_000.0, outer_a=210_000.0),
    )
    assert not feature.is_in_radius_range(90_000.0, 150_000.0)


def test_is_in_radius_range_single_edge_in() -> None:
    """Single-edge feature: returns True when the edge is in range."""
    data: dict[str, Any] = {
        'feature_type': 'GAP',
        'name': 'Single',
        'inner_data': _make_edge_data(a=100_000.0),
    }
    feature = RingFeature.from_config('single_in', data)
    assert feature.is_in_radius_range(90_000.0, 150_000.0)


def test_is_in_radius_range_single_edge_out() -> None:
    """Single-edge feature: returns False when the edge is not in range."""
    data: dict[str, Any] = {
        'feature_type': 'GAP',
        'name': 'Single Out',
        'inner_data': _make_edge_data(a=50_000.0),
    }
    feature = RingFeature.from_config('single_out', data)
    assert not feature.is_in_radius_range(90_000.0, 150_000.0)


# ---------------------------------------------------------------------------
# RingFeature.all_base_radii
# ---------------------------------------------------------------------------


def test_all_base_radii_ringlet() -> None:
    """all_base_radii returns correct (radius, label) pairs for a RINGLET."""
    feature = RingFeature.from_config('r', _make_ringlet_data(inner_a=100_000.0, outer_a=101_000.0))
    radii = feature.all_base_radii()
    assert len(radii) == 2
    radii_dict = {label: r for r, label in radii}
    assert radii_dict['IER'] == pytest.approx(100_000.0)
    assert radii_dict['OER'] == pytest.approx(101_000.0)


def test_all_base_radii_gap() -> None:
    """all_base_radii returns IEG/OEG labels for a GAP."""
    feature = RingFeature.from_config('g', _make_gap_data(inner_a=100_000.0, outer_a=101_000.0))
    radii = feature.all_base_radii()
    radii_dict = {label: r for r, label in radii}
    assert 'IEG' in radii_dict
    assert 'OEG' in radii_dict


def test_all_base_radii_single_inner_edge() -> None:
    """Single inner edge returns only one pair."""
    data: dict[str, Any] = {
        'feature_type': 'RINGLET',
        'inner_data': _make_edge_data(a=100_000.0),
    }
    feature = RingFeature.from_config('single_inner', data)
    radii = feature.all_base_radii()
    assert len(radii) == 1
    assert radii[0][1] == 'IER'


# ---------------------------------------------------------------------------
# RingFeature.uses_fade_for_edge
# ---------------------------------------------------------------------------


def test_uses_fade_gap_inner_edge() -> None:
    """GAP inner edge always uses fade."""
    feature = RingFeature.from_config('g', _make_gap_data())
    assert feature.uses_fade_for_edge('inner')


def test_uses_fade_gap_outer_edge() -> None:
    """GAP outer edge always uses fade."""
    feature = RingFeature.from_config('g', _make_gap_data())
    assert feature.uses_fade_for_edge('outer')


def test_uses_fade_ringlet_both_edges_no_fade() -> None:
    """RINGLET with both edges does NOT use fade (uses solid fill)."""
    feature = RingFeature.from_config('r', _make_ringlet_data())
    assert not feature.uses_fade_for_edge('inner')
    assert not feature.uses_fade_for_edge('outer')


def test_uses_fade_ringlet_single_inner_uses_fade() -> None:
    """Single-edge RINGLET (inner only) uses fade for that edge."""
    data: dict[str, Any] = {
        'feature_type': 'RINGLET',
        'inner_data': _make_edge_data(),
    }
    feature = RingFeature.from_config('r_single', data)
    assert feature.uses_fade_for_edge('inner')


# ---------------------------------------------------------------------------
# RingFeature.uncertainty
# ---------------------------------------------------------------------------


def test_uncertainty_both_edges_max_rms() -> None:
    """uncertainty is max(inner.rms, outer.rms)."""
    data: dict[str, Any] = {
        'feature_type': 'RINGLET',
        'inner_data': [
            {'mode': 1, 'a': 100_000.0, 'rms': 1.5, 'ae': 0.0, 'long_peri': 0.0, 'rate_peri': 0.0}
        ],
        'outer_data': [
            {'mode': 1, 'a': 101_000.0, 'rms': 3.7, 'ae': 0.0, 'long_peri': 0.0, 'rate_peri': 0.0}
        ],
    }
    feature = RingFeature.from_config('unc', data)
    assert feature.uncertainty == pytest.approx(3.7)


def test_uncertainty_single_inner_edge() -> None:
    """uncertainty is the inner edge rms when outer edge absent."""
    data: dict[str, Any] = {
        'feature_type': 'RINGLET',
        'inner_data': [
            {'mode': 1, 'a': 100_000.0, 'rms': 2.2, 'ae': 0.0, 'long_peri': 0.0, 'rate_peri': 0.0}
        ],
    }
    feature = RingFeature.from_config('unc_s', data)
    assert feature.uncertainty == pytest.approx(2.2)


def test_uncertainty_single_outer_edge() -> None:
    """uncertainty is the outer edge rms when inner edge absent."""
    data: dict[str, Any] = {
        'feature_type': 'GAP',
        'outer_data': [
            {'mode': 1, 'a': 100_000.0, 'rms': 0.8, 'ae': 0.0, 'long_peri': 0.0, 'rate_peri': 0.0}
        ],
    }
    feature = RingFeature.from_config('unc_outer', data)
    assert feature.uncertainty == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# RingFeature.edge_labels
# ---------------------------------------------------------------------------


def test_edge_labels_ringlet() -> None:
    """edge_labels for RINGLET returns IER/OER."""
    feature = RingFeature.from_config('r', _make_ringlet_data())
    labels = feature.edge_labels
    assert labels['inner'] == 'IER'
    assert labels['outer'] == 'OER'


def test_edge_labels_gap() -> None:
    """edge_labels for GAP returns IEG/OEG."""
    feature = RingFeature.from_config('g', _make_gap_data())
    labels = feature.edge_labels
    assert labels['inner'] == 'IEG'
    assert labels['outer'] == 'OEG'


# ---------------------------------------------------------------------------
# validate_no_date_overlaps
# ---------------------------------------------------------------------------


def test_validate_no_overlaps_no_dates() -> None:
    """Features without date ranges never overlap."""
    features = [
        RingFeature.from_config('a', _make_ringlet_data(inner_a=100_000.0, outer_a=101_000.0)),
        RingFeature.from_config('b', _make_gap_data(inner_a=100_500.0, outer_a=101_500.0)),
    ]
    # Should not raise
    validate_no_date_overlaps(features)


def test_validate_no_overlaps_non_overlapping_dates() -> None:
    """Features with non-overlapping date ranges in same radial region pass."""
    features = [
        RingFeature.from_config(
            'a',
            _make_ringlet_data(
                inner_a=100_000.0,
                outer_a=101_000.0,
                start_date='2008-01-01 12:00:00',
                end_date='2010-01-01 12:00:00',
            ),
        ),
        RingFeature.from_config(
            'b',
            _make_ringlet_data(
                inner_a=100_000.0,
                outer_a=101_000.0,
                start_date='2010-01-01 12:00:00',
                end_date='2012-01-01 12:00:00',
            ),
        ),
    ]
    validate_no_date_overlaps(features)


def test_validate_no_overlaps_different_radial_regions_with_dates() -> None:
    """Features at different radii with overlapping dates do not conflict."""
    features = [
        RingFeature.from_config(
            'a',
            _make_ringlet_data(
                inner_a=100_000.0,
                outer_a=101_000.0,
                start_date='2008-01-01 12:00:00',
                end_date='2011-01-01 12:00:00',
            ),
        ),
        RingFeature.from_config(
            'b',
            _make_ringlet_data(
                inner_a=200_000.0,
                outer_a=201_000.0,
                start_date='2008-01-01 12:00:00',
                end_date='2011-01-01 12:00:00',
            ),
        ),
    ]
    validate_no_date_overlaps(features)


def test_validate_raises_on_overlap() -> None:
    """Two features with overlapping radii AND overlapping dates raise ValueError."""
    features = [
        RingFeature.from_config(
            'a',
            _make_ringlet_data(
                inner_a=100_000.0,
                outer_a=101_000.0,
                start_date='2008-01-01 12:00:00',
                end_date='2011-01-01 12:00:00',
            ),
        ),
        RingFeature.from_config(
            'b',
            _make_ringlet_data(
                inner_a=100_500.0,
                outer_a=101_500.0,
                start_date='2009-01-01 12:00:00',
                end_date='2012-01-01 12:00:00',
            ),
        ),
    ]
    with pytest.raises(ValueError, match='overlap'):
        validate_no_date_overlaps(features)


def test_validate_raises_message_contains_feature_keys() -> None:
    """ValueError message identifies the conflicting feature keys."""
    features = [
        RingFeature.from_config(
            'alpha_ring',
            _make_ringlet_data(
                inner_a=100_000.0,
                outer_a=101_000.0,
                start_date='2008-01-01 12:00:00',
                end_date='2011-01-01 12:00:00',
            ),
        ),
        RingFeature.from_config(
            'beta_ring',
            _make_ringlet_data(
                inner_a=100_000.0,
                outer_a=101_000.0,
                start_date='2009-01-01 12:00:00',
                end_date='2012-01-01 12:00:00',
            ),
        ),
    ]
    with pytest.raises(ValueError) as exc_info:
        validate_no_date_overlaps(features)
    assert 'alpha_ring' in str(exc_info.value)
    assert 'beta_ring' in str(exc_info.value)


# ---------------------------------------------------------------------------
# RingFeature.render -- dispatch via mocked backplanes
# ---------------------------------------------------------------------------


def _make_mock_context(
    shape: tuple[int, int] = (20, 20),
    fade_width_pix: float = 10.0,
) -> tuple[RingsRenderContext, MagicMock]:
    """Build a RingsRenderContext with a mock obs."""
    obs = MagicMock()
    resolutions = np.full(shape, 5.0, dtype=np.float64)

    # Mock the ring_radius backplane
    bp_radii = MagicMock()
    radii_vals = np.full(shape, 100_500.0, dtype=np.float64)
    radii_vals_masked = MagicMock()
    radii_vals_masked.filled = MagicMock(return_value=radii_vals)
    bp_radii.mvals = radii_vals_masked

    # Mock ring_radius call
    obs.ext_bp.ring_radius.return_value = bp_radii

    # Mock radial_mode to return something with a .key and .mvals
    def _mock_radial_mode(*args: object, **kwargs: object) -> MagicMock:
        m = MagicMock()
        m.key = 'mock_backplane'
        m.mvals = radii_vals_masked
        return m

    obs.ext_bp.radial_mode.side_effect = _mock_radial_mode

    # Mock border_atop for annotation creation
    border_bp = MagicMock()
    border_arr = MagicMock()
    edge_mask_vals = np.zeros(shape, dtype=bool)
    edge_mask_arr = MagicMock()
    edge_mask_arr.filled = MagicMock(return_value=edge_mask_vals)
    border_arr.astype = MagicMock(return_value=edge_mask_arr)
    border_bp.mvals = border_arr
    obs.ext_bp.border_atop.return_value = border_bp

    ctx = RingsRenderContext(
        obs=obs,
        ring_target='saturn:ring',
        epoch=252460865.0,
        resolutions=resolutions,
        fade_width_pix=fade_width_pix,
        all_edge_radii=(),
        logger=MagicMock(),
    )
    return ctx, obs


def test_render_full_ringlet_returns_single_result() -> None:
    """Full ringlet (both edges, RINGLET type) returns one RingRenderResult."""
    ctx, _obs = _make_mock_context()
    feature = RingFeature.from_config('r', _make_ringlet_data())
    results = feature.render(ctx)
    assert len(results) == 1
    assert results[0].uncertainty == pytest.approx(feature.uncertainty)


def test_render_gap_both_edges_returns_two_results() -> None:
    """GAP with both edges returns two RingRenderResults (one per edge)."""
    ctx, _obs = _make_mock_context()
    feature = RingFeature.from_config('g', _make_gap_data())
    results = feature.render(ctx)
    assert len(results) == 2


def test_render_single_inner_edge_returns_one_result() -> None:
    """Feature with only inner edge returns one result."""
    data: dict[str, Any] = {
        'feature_type': 'RINGLET',
        'inner_data': _make_edge_data(a=100_000.0),
    }
    ctx, _obs = _make_mock_context()
    feature = RingFeature.from_config('single', data)
    results = feature.render(ctx)
    assert len(results) == 1


def test_render_single_outer_edge_returns_one_result() -> None:
    """Feature with only outer edge returns one result."""
    data: dict[str, Any] = {
        'feature_type': 'GAP',
        'outer_data': _make_edge_data(a=101_000.0),
    }
    ctx, _obs = _make_mock_context()
    feature = RingFeature.from_config('outer', data)
    results = feature.render(ctx)
    assert len(results) == 1


def test_render_result_model_img_shape() -> None:
    """Rendered model_img has the same shape as resolutions."""
    ctx, _obs = _make_mock_context(shape=(15, 25))
    feature = RingFeature.from_config('r', _make_ringlet_data())
    results = feature.render(ctx)
    assert results[0].model_img.shape == (15, 25)


def test_render_result_uncertainty_matches_feature() -> None:
    """uncertainty in result matches feature.uncertainty."""
    data: dict[str, Any] = {
        'feature_type': 'RINGLET',
        'inner_data': [
            {'mode': 1, 'a': 100_000.0, 'rms': 4.0, 'ae': 0.0, 'long_peri': 0.0, 'rate_peri': 0.0}
        ],
        'outer_data': [
            {'mode': 1, 'a': 101_000.0, 'rms': 2.5, 'ae': 0.0, 'long_peri': 0.0, 'rate_peri': 0.0}
        ],
    }
    ctx, _obs = _make_mock_context()
    feature = RingFeature.from_config('unc', data)
    results = feature.render(ctx)
    assert results[0].uncertainty == pytest.approx(4.0)  # max(4.0, 2.5)


def test_render_calls_radial_mode_for_edge() -> None:
    """render() calls ext_bp.radial_mode for each edge."""
    ctx, obs = _make_mock_context()
    feature = RingFeature.from_config('r', _make_ringlet_data())
    feature.render(ctx)
    assert obs.ext_bp.radial_mode.call_count >= 1
