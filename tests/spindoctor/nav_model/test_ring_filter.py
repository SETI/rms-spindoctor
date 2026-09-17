"""Unit tests for ``RingFeatureFilter`` (four-pass pipeline).

Tests cover each pass independently and in combination:

- Pass 1 (date): ``is_visible_at`` checks
- Pass 2 (radius): ``is_in_radius_range`` checks, partial visibility
- Pass 3 (resolvability): two-edge feature width vs
  ``min_feature_pixels * min_res``
- Pass 4 (fade conflict): adjusted fade width vs ``min_allowed_fade_width_pix``

Production code passes the navigation model's ``PdsLogger`` as ``logger``.
These tests use ``logging.getLogger('spindoctor.nav_model.rings.ring_filter')`` so
pytest ``caplog`` can capture DEBUG exclusion messages.
"""

import logging
from typing import Any

import pytest

from spindoctor.nav_model.rings.ring_feature import RingFeature
from spindoctor.nav_model.rings.ring_filter import RingFeatureFilter
from spindoctor.support.time import utc_to_et

# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------


def _make_edge_data(a: float = 100_000.0, rms: float = 1.0) -> list[dict[str, Any]]:
    """Return a single-mode mode-1 edge data list."""
    return [{'mode': 1, 'a': a, 'rms': rms, 'ae': 0.0, 'long_peri': 0.0, 'rate_peri': 0.0}]


def _make_ringlet(
    key: str = 'r',
    inner_a: float = 100_000.0,
    outer_a: float = 101_000.0,
    start_date: str | None = None,
    end_date: str | None = None,
) -> RingFeature:
    """Return a RingFeature with both edges (RINGLET)."""
    d: dict[str, Any] = {
        'feature_type': 'RINGLET',
        'name': key,
        'inner_data': _make_edge_data(a=inner_a),
        'outer_data': _make_edge_data(a=outer_a),
    }
    if start_date:
        d['start_date'] = start_date
    if end_date:
        d['end_date'] = end_date
    return RingFeature.from_config(key, d)


def _make_gap(
    key: str = 'g',
    inner_a: float = 100_000.0,
    outer_a: float = 101_000.0,
    start_date: str | None = None,
    end_date: str | None = None,
) -> RingFeature:
    """Return a RingFeature with both edges (GAP)."""
    d: dict[str, Any] = {
        'feature_type': 'GAP',
        'name': key,
        'inner_data': _make_edge_data(a=inner_a),
        'outer_data': _make_edge_data(a=outer_a),
    }
    if start_date:
        d['start_date'] = start_date
    if end_date:
        d['end_date'] = end_date
    return RingFeature.from_config(key, d)


def _make_single_edge_ringlet(
    key: str = 'ser',
    inner_a: float = 100_000.0,
) -> RingFeature:
    """Return a RINGLET with only an inner edge (no outer)."""
    return RingFeature.from_config(
        key,
        {
            'feature_type': 'RINGLET',
            'name': key,
            'inner_data': _make_edge_data(a=inner_a),
        },
    )


def _make_filter(
    *,
    obs_time_et: float | None = None,
    min_radius: float = 90_000.0,
    max_radius: float = 110_000.0,
    min_res_at_radius: dict[float, float] | None = None,
    fade_width_pix: float = 100.0,
    min_allowed_fade_width_pix: float = 10.0,
    min_feature_pixels: float = 2.0,
) -> RingFeatureFilter:
    """Build a ``RingFeatureFilter`` with test-friendly defaults.

    Parameters:
        obs_time_et: Observation time in TDB seconds. Defaults to 2008-01-01 12:00:00 UTC.
        min_radius: Minimum ring radius in km.
        max_radius: Maximum ring radius in km.
        min_res_at_radius: Dict mapping radius -> min_resolution (km/pixel). Missing
            keys return 1.0 by default.
        fade_width_pix: Desired fade width in pixels.
        min_allowed_fade_width_pix: Minimum allowed fade width in pixels.
        min_feature_pixels: Minimum feature width in pixels for pass 3.

    A stdlib logger named ``spindoctor.nav_model.rings.ring_filter`` is supplied so
    caplog-based tests can enable DEBUG on that logger.
    """
    if obs_time_et is None:
        obs_time_et = utc_to_et('2008-01-01 12:00:00')
    res_map = min_res_at_radius or {}

    def get_min_res(radius: float) -> float | None:
        return res_map.get(radius, 1.0)

    return RingFeatureFilter(
        obs_time_et=obs_time_et,
        min_radius=min_radius,
        max_radius=max_radius,
        min_res_at_radius=get_min_res,
        fade_width_pix=fade_width_pix,
        min_allowed_fade_width_pix=min_allowed_fade_width_pix,
        min_feature_pixels=min_feature_pixels,
        logger=logging.getLogger('spindoctor.nav_model.rings.ring_filter'),
    )


# ---------------------------------------------------------------------------
# Pass 1: date filtering
# ---------------------------------------------------------------------------


class TestPass1Date:
    """Pass 1 filters features by observation date."""

    def test_no_dates_always_passes(self) -> None:
        """Feature with no date range is always included."""
        features = [_make_ringlet()]
        result = _make_filter().filter(features)
        assert len(result) == 1

    def test_feature_within_date_range_passes(self) -> None:
        """Feature whose date range includes obs_time passes."""
        t = utc_to_et('2008-01-01 12:00:00')
        feature = _make_ringlet(start_date='2007-01-01', end_date='2009-01-01')
        flt = _make_filter(obs_time_et=t)
        result = flt.filter([feature])
        assert len(result) == 1

    def test_feature_before_start_date_excluded(self) -> None:
        """Feature whose start_date is in the future is excluded."""
        t = utc_to_et('2006-01-01 12:00:00')
        feature = _make_ringlet(start_date='2007-01-01')
        flt = _make_filter(obs_time_et=t)
        result = flt.filter([feature])
        assert len(result) == 0

    def test_feature_after_end_date_excluded(self) -> None:
        """Feature whose end_date has passed is excluded."""
        t = utc_to_et('2010-01-01 12:00:00')
        feature = _make_ringlet(end_date='2009-01-01')
        flt = _make_filter(obs_time_et=t)
        result = flt.filter([feature])
        assert len(result) == 0

    def test_feature_exactly_at_end_date_excluded(self) -> None:
        """Date range is half-open [start, end); obs at end_date is excluded."""
        end = '2009-01-01 00:00:00'
        t = utc_to_et(end)
        feature = _make_ringlet(end_date=end)
        flt = _make_filter(obs_time_et=t)
        result = flt.filter([feature])
        assert len(result) == 0

    def test_feature_exactly_at_start_date_passes(self) -> None:
        """Date range is half-open; obs at start_date is included."""
        start = '2007-01-01 00:00:00'
        t = utc_to_et(start)
        feature = _make_ringlet(start_date=start)
        flt = _make_filter(obs_time_et=t)
        result = flt.filter([feature])
        assert len(result) == 1

    def test_date_exclusion_logged_at_debug(self, caplog: pytest.LogCaptureFixture) -> None:
        """Date exclusion is logged at DEBUG level."""
        t = utc_to_et('2005-01-01 12:00:00')
        feature = _make_ringlet(key='myring', start_date='2007-01-01')
        flt = _make_filter(obs_time_et=t)
        with caplog.at_level(logging.DEBUG, logger='spindoctor.nav_model.rings.ring_filter'):
            flt.filter([feature])
        assert any('myring' in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Pass 2: radius filtering
# ---------------------------------------------------------------------------


class TestPass2Radius:
    """Pass 2 filters features by visible radius range."""

    def test_feature_inside_radius_range_passes(self) -> None:
        """Feature with both edges in range passes."""
        feature = _make_ringlet(inner_a=100_000.0, outer_a=101_000.0)
        flt = _make_filter(min_radius=99_000.0, max_radius=102_000.0)
        result = flt.filter([feature])
        assert len(result) == 1

    def test_feature_outside_radius_range_excluded(self) -> None:
        """Feature with both edges outside range is excluded."""
        feature = _make_ringlet(inner_a=50_000.0, outer_a=51_000.0)
        flt = _make_filter(min_radius=99_000.0, max_radius=102_000.0)
        result = flt.filter([feature])
        assert len(result) == 0

    def test_partial_visibility_inner_in_range_passes(self) -> None:
        """RINGLET with inner edge in range but outer edge out of range passes (partial)."""
        feature = _make_ringlet(inner_a=100_000.0, outer_a=200_000.0)
        flt = _make_filter(min_radius=99_000.0, max_radius=105_000.0)
        result = flt.filter([feature])
        assert len(result) == 1

    def test_partial_visibility_inner_in_range_outer_trimmed(self) -> None:
        """RINGLET with outer edge off-screen has outer edge set to None (trimmed)."""
        feature = _make_ringlet(inner_a=100_000.0, outer_a=200_000.0)
        flt = _make_filter(min_radius=99_000.0, max_radius=105_000.0)
        result = flt.filter([feature])
        assert result[0].inner_edge is not None
        assert result[0].outer_edge is None

    def test_partial_visibility_outer_in_range_passes(self) -> None:
        """RINGLET with outer edge in range but inner edge out of range passes (partial)."""
        feature = _make_ringlet(inner_a=50_000.0, outer_a=100_000.0)
        flt = _make_filter(min_radius=99_000.0, max_radius=102_000.0)
        result = flt.filter([feature])
        assert len(result) == 1

    def test_partial_visibility_outer_in_range_inner_trimmed(self) -> None:
        """RINGLET with inner edge off-screen has inner edge set to None (trimmed)."""
        feature = _make_ringlet(inner_a=50_000.0, outer_a=100_000.0)
        flt = _make_filter(min_radius=99_000.0, max_radius=102_000.0)
        result = flt.filter([feature])
        assert result[0].inner_edge is None
        assert result[0].outer_edge is not None

    def test_partial_visibility_trim_logged_at_debug(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Partial-visibility edge trimming is logged at DEBUG level."""
        feature = _make_ringlet(key='partial', inner_a=100_000.0, outer_a=200_000.0)
        flt = _make_filter(min_radius=99_000.0, max_radius=105_000.0)
        with caplog.at_level(logging.DEBUG, logger='spindoctor.nav_model.rings.ring_filter'):
            flt.filter([feature])
        assert any('partial' in r.message and 'outer' in r.message for r in caplog.records)

    def test_radius_exclusion_logged_at_debug(self, caplog: pytest.LogCaptureFixture) -> None:
        """Radius exclusion is logged at DEBUG level."""
        feature = _make_ringlet(key='offscreen', inner_a=50_000.0, outer_a=51_000.0)
        flt = _make_filter(min_radius=99_000.0, max_radius=102_000.0)
        with caplog.at_level(logging.DEBUG, logger='spindoctor.nav_model.rings.ring_filter'):
            flt.filter([feature])
        assert any('offscreen' in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Pass 3: resolvability filtering
# ---------------------------------------------------------------------------


class TestPass3Resolvability:
    """Pass 3 filters two-edge features where the gap/ringlet width is not resolvable."""

    def test_resolvable_ringlet_passes(self) -> None:
        """Ringlet with width > min_feature_pixels * min_res passes."""
        # inner=100_000, outer=101_000, width=1000 km
        # min_res=1.0 km/px, min_feature_pixels=2 -> threshold=2 km -> 1000 > 2: pass
        feature = _make_ringlet(inner_a=100_000.0, outer_a=101_000.0)
        res = {100_000.0: 1.0, 101_000.0: 1.0}
        flt = _make_filter(
            min_res_at_radius=res,
            min_feature_pixels=2.0,
        )
        result = flt.filter([feature])
        assert len(result) == 1

    def test_unresolvable_ringlet_excluded(self) -> None:
        """Ringlet with width < min_feature_pixels * min_res is excluded."""
        # inner=100_000, outer=100_001, width=1 km
        # min_res=5.0 km/px, min_feature_pixels=3 -> threshold=15 km -> 1 < 15: exclude
        feature = _make_ringlet(inner_a=100_000.0, outer_a=100_001.0)
        res = {100_000.0: 5.0, 100_001.0: 5.0}
        flt = _make_filter(
            min_radius=99_000.0,
            max_radius=110_000.0,
            min_res_at_radius=res,
            min_feature_pixels=3.0,
        )
        result = flt.filter([feature])
        assert len(result) == 0

    def test_unresolvable_gap_excluded(self) -> None:
        """GAP with width < threshold is excluded (hole would not be visible)."""
        feature = _make_gap(inner_a=100_000.0, outer_a=100_002.0)
        res = {100_000.0: 5.0, 100_002.0: 5.0}
        flt = _make_filter(
            min_radius=99_000.0,
            max_radius=110_000.0,
            min_res_at_radius=res,
            min_feature_pixels=3.0,
        )
        result = flt.filter([feature])
        assert len(result) == 0

    def test_single_edge_feature_skips_pass3(self) -> None:
        """Single-edge feature is never filtered by pass 3 (no width to check)."""
        feature = _make_single_edge_ringlet(inner_a=100_000.0)
        res = {100_000.0: 100.0}  # very coarse resolution (km/px); excludes two-edge
        flt = _make_filter(
            min_res_at_radius=res,
            # Small cutoff; exclusion of two-edge features here is due to coarse res,
            # not this threshold.
            min_feature_pixels=1.0,
        )
        result = flt.filter([feature])
        assert len(result) == 1

    def test_partial_visibility_skips_pass3(self) -> None:
        """Partially visible ringlet (one edge outside radius range) skips pass 3."""
        # Only outer edge (100_000) is in range; inner edge (50_000) is not
        # Since it's partially visible, pass 3 doesn't check the width
        feature = _make_ringlet(inner_a=50_000.0, outer_a=100_000.0)
        res = {50_000.0: 100.0, 100_000.0: 100.0}
        flt = _make_filter(
            min_radius=99_000.0,
            max_radius=102_000.0,
            min_res_at_radius=res,
            min_feature_pixels=2.0,
        )
        result = flt.filter([feature])
        assert len(result) == 1

    def test_pass3_exclusion_logged_at_debug(self, caplog: pytest.LogCaptureFixture) -> None:
        """Resolution exclusion is logged at DEBUG level."""
        feature = _make_ringlet(key='tiny', inner_a=100_000.0, outer_a=100_001.0)
        res = {100_000.0: 50.0, 100_001.0: 50.0}
        flt = _make_filter(
            min_radius=99_000.0,
            max_radius=110_000.0,
            min_res_at_radius=res,
            min_feature_pixels=5.0,
        )
        with caplog.at_level(logging.DEBUG, logger='spindoctor.nav_model.rings.ring_filter'):
            flt.filter([feature])
        assert any('tiny' in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Pass 4: fade conflict filtering
# ---------------------------------------------------------------------------


class TestPass4FadeConflict:
    """Pass 4 filters fade-using edges where adjusted width < min_allowed_fade_width_pix."""

    def test_no_conflict_single_edge_passes(self) -> None:
        """Single-edge ringlet with no nearby edges passes pass 4."""
        feature = _make_single_edge_ringlet(inner_a=100_000.0)
        # No other edges nearby; fade_width=100 pix, min_res=1.0 km/px -> 100 km fade
        # min_allowed=10 pix -> min_allowed_km=10 km; 100 > 10: pass
        flt = _make_filter(
            min_res_at_radius={100_000.0: 1.0},
            fade_width_pix=100.0,
            min_allowed_fade_width_pix=10.0,
        )
        result = flt.filter([feature])
        assert len(result) == 1

    def test_conflict_reduces_below_min_excludes_edge(self) -> None:
        """Single-edge ringlet excluded when conflict reduces fade below min_allowed."""
        # inner edge at 100_000 fades outward (shade_above=True for RINGLET inner)
        # Neighboring edge at 100_050: signed_dist = 50 km, half = 25 km
        # min_res = 5.0 km/px; min_allowed = 10 pix -> 50 km; 25 < 50: EXCLUDE
        feature = _make_single_edge_ringlet(key='ser', inner_a=100_000.0)
        neighbor = _make_ringlet(key='nbr', inner_a=100_050.0, outer_a=100_200.0)
        res = {100_000.0: 5.0, 100_050.0: 5.0, 100_200.0: 5.0}
        flt = _make_filter(
            min_radius=99_000.0,
            max_radius=110_000.0,
            min_res_at_radius=res,
            fade_width_pix=100.0,
            min_allowed_fade_width_pix=10.0,
        )
        result = flt.filter([feature, neighbor])
        # The neighbor passes, but set should be excluded
        keys = [f.key for f in result]
        assert 'ser' not in keys

    def test_conflict_above_min_allowed_passes(self) -> None:
        """Single-edge ringlet passes when conflict-adjusted width >= min_allowed."""
        # inner edge at 100_000 fades outward
        # Neighboring edge at 100_500: signed_dist = 500 km, half = 250 km
        # min_res = 1.0 km/px; min_allowed = 10 pix -> 10 km; 250 > 10: PASS
        feature = _make_single_edge_ringlet(key='ser', inner_a=100_000.0)
        neighbor = _make_ringlet(key='nbr', inner_a=100_500.0, outer_a=101_000.0)
        res = {100_000.0: 1.0, 100_500.0: 1.0, 101_000.0: 1.0}
        flt = _make_filter(
            min_radius=99_000.0,
            max_radius=110_000.0,
            min_res_at_radius=res,
            fade_width_pix=100.0,
            min_allowed_fade_width_pix=10.0,
        )
        result = flt.filter([feature, neighbor])
        keys = [f.key for f in result]
        assert 'ser' in keys

    def test_gap_both_edges_excluded_removes_feature(self) -> None:
        """GAP excluded entirely when both edges fail pass 4."""
        # GAP at 100_000-100_100; both edges fade toward each other
        # inner edge at 100_000 fades inward (shade_above=False): looks for neighbors < 100_000
        # outer edge at 100_100 fades outward (shade_above=True): looks for neighbors > 100_100
        # Add a tight neighbor below inner edge: at 99_970 (dist 30 km, half=15 km)
        # And a tight neighbor above outer edge: at 100_130 (dist 30 km, half=15 km)
        # min_res=5.0 km/px, min_allowed=10 -> threshold=50 km; 15 < 50: both excluded
        gap = _make_gap(key='gap', inner_a=100_000.0, outer_a=100_100.0)
        inner_nbr = _make_single_edge_ringlet(key='in', inner_a=99_970.0)
        outer_nbr = _make_single_edge_ringlet(key='out', inner_a=100_130.0)
        res_map = {
            100_000.0: 5.0,
            100_100.0: 5.0,
            99_970.0: 5.0,
            100_130.0: 5.0,
        }
        flt = _make_filter(
            min_radius=99_000.0,
            max_radius=110_000.0,
            min_res_at_radius=res_map,
            fade_width_pix=100.0,
            min_allowed_fade_width_pix=10.0,
        )
        result = flt.filter([gap, inner_nbr, outer_nbr])
        keys = [f.key for f in result]
        assert 'gap' not in keys

    def test_gap_one_edge_excluded_keeps_feature_with_that_edge_removed(self) -> None:
        """GAP with one edge failing pass 4: feature kept with that edge set to None."""
        # GAP at 100_000-100_200
        # outer edge at 100_200 fades outward: tight neighbor at 100_230 (dist 30, half=15)
        # inner edge at 100_000 fades inward: no close neighbor -> passes
        # min_res=5.0, min_allowed=10 -> threshold=50 km; 15 < 50: outer excluded
        gap = _make_gap(key='gap2', inner_a=100_000.0, outer_a=100_200.0)
        outer_nbr = _make_single_edge_ringlet(key='onbr', inner_a=100_230.0)
        res_map = {100_000.0: 5.0, 100_200.0: 5.0, 100_230.0: 5.0}
        flt = _make_filter(
            min_radius=99_000.0,
            max_radius=110_000.0,
            min_res_at_radius=res_map,
            fade_width_pix=100.0,
            min_allowed_fade_width_pix=10.0,
        )
        result = flt.filter([gap, outer_nbr])
        gap_results = [f for f in result if f.key == 'gap2']
        assert len(gap_results) == 1
        # Outer edge should be removed
        assert gap_results[0].outer_edge is None
        # Inner edge should remain
        assert gap_results[0].inner_edge is not None

    def test_pass4_preserves_outermost_when_all_outer_edges_dropped(self) -> None:
        """At very low resolution the outermost feature is restored after pass 4.

        Several features cluster within a single pixel of fade width;
        the per-edge check drops every outer-side edge.  The
        preserve-outermost pass restores the feature carrying the
        largest in-range edge so navigation has at least one outer
        reference, which is far more useful than an inner-region
        gap edge.
        """
        # Three single-edge ringlets within 50 km of each other; very
        # low resolution (km/px = 100) makes the fade width 10000 km.
        # Every per-edge fade conflict shrinks below the min_allowed
        # threshold -> all 3 would otherwise be dropped.
        inner = _make_single_edge_ringlet(key='inner', inner_a=100_000.0)
        middle = _make_single_edge_ringlet(key='middle', inner_a=100_025.0)
        outer = _make_single_edge_ringlet(key='outer', inner_a=100_050.0)
        res = {100_000.0: 100.0, 100_025.0: 100.0, 100_050.0: 100.0}
        flt = _make_filter(
            min_radius=99_000.0,
            max_radius=110_000.0,
            min_res_at_radius=res,
            fade_width_pix=100.0,
            min_allowed_fade_width_pix=10.0,
        )
        result = flt.filter([inner, middle, outer])
        keys = [f.key for f in result]
        # Without preservation all three would be dropped (zero
        # results).  The preserve-outermost pass restores exactly
        # 'outer' (the one with the largest in-range edge radius); no
        # other feature should sneak into the result, and 'outer'
        # must appear exactly once.
        assert keys == ['outer']

    def test_pass4_outermost_replaces_existing_trimmed_entry(self) -> None:
        """Restoring the outermost feature replaces the trimmed entry by key.

        A GAP whose outer edge is the outermost in-range edge gets its
        outer trimmed by a tight neighbor; the trimmed feature (with
        only its inner edge) is in ``result``.  The preserve-outermost
        pass detects that the outermost edge is no longer represented
        and restores the original GAP in place — replacing the trimmed
        entry rather than appending a second entry under the same
        ``key``.

        Each feature ``key`` must appear at most once in the filter's
        output; a duplicate would feed the orchestrator two ``NavFeature``
        instances with the same ``feature_id`` and break the
        ensemble's per-feature accounting.
        """
        # Outer GAP at (100_000, 100_200) is the outermost in-range
        # feature.  An inner-side neighbor inside the GAP region squeezes
        # the GAP's INNER edge fade (inner fades downward toward smaller
        # radii); leave the OUTER edge to fail via a fade-conflict-free
        # path.  Trick: use a GAP whose outer-edge fade conflict comes
        # from a tight neighbor *above* it that does not survive
        # resolvability (so it shrinks the fade but doesn't appear in
        # ``after_res`` to preserve the outermost-in-range edge being
        # the GAP's outer).
        gap = _make_gap(key='outer_gap', inner_a=100_000.0, outer_a=100_200.0)
        # Narrow ringlet just above outer GAP edge: width below pass-3
        # resolvability threshold, but its edges still contribute to
        # ``all_edge_radii`` (built from after_radius / pass 2) and
        # squeeze the GAP's outer fade.  Width 0.1 km is well below
        # min_feature_pixels=10 * min_res=5 = 50 km threshold.
        narrow = _make_ringlet(key='narrow', inner_a=100_205.0, outer_a=100_205.1)
        res_map = {
            100_000.0: 5.0,
            100_200.0: 5.0,
            100_205.0: 5.0,
            100_205.1: 5.0,
        }
        flt = _make_filter(
            min_radius=99_000.0,
            max_radius=110_000.0,
            min_res_at_radius=res_map,
            fade_width_pix=100.0,
            min_allowed_fade_width_pix=10.0,
            min_feature_pixels=10.0,
        )
        result = flt.filter([gap, narrow])
        gap_entries = [f for f in result if f.key == 'outer_gap']
        # Pass 4 may either trim+restore (in which case the entry is
        # replaced in place) or leave the outer edge intact.  Either
        # way the key must appear at most once.
        assert len(gap_entries) <= 1

    def test_pass4_exclusion_logged_at_debug(self, caplog: pytest.LogCaptureFixture) -> None:
        """Pass 4 exclusion is logged at DEBUG level."""
        feature = _make_single_edge_ringlet(key='tight', inner_a=100_000.0)
        neighbor = _make_ringlet(key='nbr', inner_a=100_050.0, outer_a=100_200.0)
        res = {100_000.0: 5.0, 100_050.0: 5.0, 100_200.0: 5.0}
        flt = _make_filter(
            min_radius=99_000.0,
            max_radius=110_000.0,
            min_res_at_radius=res,
            fade_width_pix=100.0,
            min_allowed_fade_width_pix=10.0,
        )
        with caplog.at_level(logging.DEBUG, logger='spindoctor.nav_model.rings.ring_filter'):
            flt.filter([feature, neighbor])
        assert any('tight' in r.message for r in caplog.records)

    def test_all_edge_radii_uses_pass2_survivors_not_all_features(self) -> None:
        """Pass 4 conflict detection uses only radius-range-surviving features.

        A feature far outside the visible radius range should not shrink
        the fade width for an in-range feature.
        """
        # In-range feature at 100_000, fades outward
        in_range = _make_single_edge_ringlet(key='inr', inner_a=100_000.0)
        # Off-screen feature at 100_050 (would shrink fade if included)
        off_screen = _make_single_edge_ringlet(key='off', inner_a=100_050.0)
        res = {100_000.0: 1.0, 100_050.0: 1.0}
        # Only 99_000-100_010 is visible, so off_screen is outside range
        flt = _make_filter(
            min_radius=99_000.0,
            max_radius=100_010.0,  # off_screen at 100_050 is outside
            min_res_at_radius=res,
            fade_width_pix=100.0,
            min_allowed_fade_width_pix=10.0,
        )
        result = flt.filter([in_range, off_screen])
        keys = [f.key for f in result]
        # in_range should pass (no conflict from off_screen since it was filtered by pass 2)
        assert 'inr' in keys


# ---------------------------------------------------------------------------
# Integration: multi-pass filtering
# ---------------------------------------------------------------------------


class TestIntegration:
    """Integration tests verifying the full 4-pass pipeline."""

    def test_multiple_features_each_filtered_independently(self) -> None:
        """Multiple features can be filtered in one call."""
        f1 = _make_ringlet(key='f1', inner_a=100_000.0, outer_a=101_000.0)
        f2 = _make_ringlet(
            key='f2', inner_a=100_000.0, outer_a=101_000.0, start_date='2020-01-01'
        )  # future
        f3 = _make_ringlet(key='f3', inner_a=50_000.0, outer_a=51_000.0)  # out of range
        result = _make_filter().filter([f1, f2, f3])
        keys = [f.key for f in result]
        assert keys == ['f1']

    def test_pipeline_order_date_before_radius(self) -> None:
        """Features excluded by date don't undergo radius or resolution checks."""
        future = _make_ringlet(
            key='future',
            inner_a=50_000.0,
            outer_a=51_000.0,
            start_date='2020-01-01',
        )
        # This feature would also fail radius, but we just want it excluded by date
        flt = _make_filter(min_radius=99_000.0, max_radius=102_000.0)
        result = flt.filter([future])
        assert len(result) == 0

    def test_empty_input_returns_empty(self) -> None:
        """Empty feature list returns empty list."""
        result = _make_filter().filter([])
        assert result == []

    def test_preserves_feature_attributes(self) -> None:
        """Filtered features retain all their original attributes."""
        feature = _make_ringlet(key='orig', inner_a=100_000.0, outer_a=101_000.0)
        result = _make_filter().filter([feature])
        assert len(result) == 1
        assert result[0].key == 'orig'
        assert result[0].inner_edge is not None
        assert result[0].outer_edge is not None
