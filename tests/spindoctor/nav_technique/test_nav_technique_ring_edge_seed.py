"""``RingEdgeNav``'s coarse seed against ring texture and radial aliases."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
from tests.spindoctor.nav_technique.conftest import (
    FlatPolylineFactory,
    NavContextFactory,
    NavFeatureFactory,
)

from spindoctor.feature.feature import NavFeature
from spindoctor.nav_technique import nav_technique_ring_edge, ring_edge_seed
from spindoctor.nav_technique.diagnostics import RingEdgeDiagnostics
from spindoctor.nav_technique.dt_fitting import CoarseSearchResult
from spindoctor.nav_technique.nav_technique_ring_edge import RingEdgeNav
from spindoctor.nav_technique.ring_edge_seed import SeedBasinCheck, check_seed_basin


def _textured_ring_image(shape_vu: tuple[int, int], edge_row: int) -> np.ndarray:
    """Return ring material above ``edge_row`` striped by weak concentric texture.

    The material is 100 bright with an 8 unit square-wave ripple of period 6 px
    in ``v``, so its interior carries a detected edge every 3 rows, each running
    parallel to the material's outer edge and an order of magnitude weaker
    than it.
    """
    rows = np.arange(shape_vu[0], dtype=np.float64)[:, None] * np.ones((1, shape_vu[1]))
    image = np.zeros(shape_vu, dtype=np.float64)
    material = rows < edge_row
    image[material] = 100.0 + 8.0 * np.sign(np.sin(2.0 * np.pi * rows[material] / 6.0))
    return image


def test_ring_edge_nav_seeds_on_the_strong_edge_not_the_interior_texture(
    flat_polyline: FlatPolylineFactory,
    make_ring_feature: NavFeatureFactory,
    make_nav_context: NavContextFactory,
) -> None:
    """A model laid inside the ring's texture still recovers the material's edge.

    The predicted edge sits 7 px inside the material, where every few rows a
    texture edge runs parallel to it.  Counting detected edge pixels ties
    those texture edges with the true one and keeps the alias nearest the
    prediction; weighting by edge strength finds the material's edge.
    """
    image = _textured_ring_image((200, 160), 120)
    planted_dv = 7.0
    # The step lies between rows 119 and 120.
    vertices, outward = flat_polyline(119.5 - planted_dv, 20.0, 140.0, 121)
    feature = make_ring_feature(
        'edge', vertices=vertices, outward_normals=outward, is_straight_line=True
    )
    result = RingEdgeNav().navigate([feature], make_nav_context(image))
    assert result.offset_px[0] == pytest.approx(planted_dv, abs=0.3)
    assert result.spurious is False


def test_ring_edge_nav_keeps_the_overlap_seed_on_a_comparably_strong_edge(
    flat_polyline: FlatPolylineFactory,
    make_ring_feature: NavFeatureFactory,
    make_nav_context: NavContextFactory,
) -> None:
    """A slightly stronger edge farther away does not displace the overlap seed.

    Two clean steps run parallel to the model: one 3 px from the prediction and
    one, 10 percent stronger, 9 px away.  Both carry the whole model, so the
    overlap search takes the nearer one, and its strength is well above the
    keep ratio of the stronger, so the seed stays there.
    """
    shape = (200, 160)
    rows = np.arange(shape[0], dtype=np.float64)[:, None] * np.ones((1, shape[1]))
    image = np.where(rows < 103.0, 100.0, 0.0) + np.where(rows < 109.0, 110.0, 0.0)
    vertices, outward = flat_polyline(99.5, 20.0, 140.0, 121)
    feature = make_ring_feature(
        'edge', vertices=vertices, outward_normals=outward, is_straight_line=True
    )
    result = RingEdgeNav().navigate([feature], make_nav_context(image))
    assert result.offset_px[0] == pytest.approx(3.0, abs=0.3)


@pytest.mark.parametrize('ratio', [-0.1, 1.5, math.nan])
def test_ring_edge_nav_rejects_a_coarse_strength_keep_ratio_outside_unit_range(
    ratio: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A keep ratio outside [0, 1], or NaN, fails at construction."""
    bad_tuning = dict(RingEdgeNav.tuning)
    bad_tuning['coarse_strength_keep_ratio'] = ratio
    monkeypatch.setattr(RingEdgeNav, 'tuning', bad_tuning)
    with pytest.raises(ValueError, match='coarse_strength_keep_ratio must lie in'):
        RingEdgeNav()


@pytest.mark.parametrize('margin', [-1, 2.5], ids=['negative', 'fractional'])
def test_ring_edge_nav_rejects_a_coarse_sensor_margin_that_is_not_a_count(
    margin: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A negative or fractional margin raises instead of being truncated."""
    bad_tuning = dict(RingEdgeNav.tuning)
    bad_tuning['coarse_sensor_margin_px'] = margin
    monkeypatch.setattr(RingEdgeNav, 'tuning', bad_tuning)
    with pytest.raises(ValueError, match='coarse_sensor_margin_px must be a non-negative'):
        RingEdgeNav()


def _seed_with(
    monkeypatch: pytest.MonkeyPatch, *, overlap_strength: float, best_strength: float
) -> SeedBasinCheck:
    """Run ``_coarse_seed`` with its searches and strength scores replaced by fixed answers.

    The overlap search answers ``(1, 1)`` and the strongest-edge search ``(5, 5)``;
    the edge strength under each is the matching argument.
    """
    monkeypatch.setattr(
        nav_technique_ring_edge,
        'coarse_ncc_search_scored',
        lambda *_args, **_kwargs: CoarseSearchResult(offset_vu=(1, 1), score=1.0),
    )
    monkeypatch.setattr(
        nav_technique_ring_edge,
        'coarse_edge_strength_search_scored',
        lambda *_args, **_kwargs: CoarseSearchResult(offset_vu=(5, 5), score=1.0),
    )
    strengths = {(1, 1): overlap_strength, (5, 5): best_strength}

    def _score(*args: object, **_kwargs: object) -> float:
        """Return the fixed strength of the seed at ``args[4]``."""
        offset = args[4]
        assert isinstance(offset, tuple)
        return strengths[offset]

    monkeypatch.setattr(nav_technique_ring_edge, 'edge_strength_score', _score)
    empty = np.zeros((4, 4), dtype=bool)
    return RingEdgeNav()._coarse_seed(
        empty,
        np.zeros((4, 4, 2)),
        np.zeros((1, 2)),
        np.zeros((1, 2)),
        (1, 1),
        valid_mask=empty,
    )


@pytest.mark.parametrize(
    ('case', 'offset'), [('at_ratio', (1, 1)), ('just_below', (5, 5)), ('no_strength', (1, 1))]
)
def test_coarse_seed_keeps_the_overlap_seed_at_or_above_the_keep_ratio(
    case: str,
    offset: tuple[int, int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The keep ratio is inclusive, and a frame with no strength anywhere keeps the overlap seed."""
    keep = RingEdgeNav()._coarse_strength_keep_ratio
    overlap_strength, best_strength = {
        'at_ratio': (keep, 1.0),
        'just_below': (float(np.nextafter(keep, 0.0)), 1.0),
        'no_strength': (0.0, 0.0),
    }[case]
    seed = _seed_with(monkeypatch, overlap_strength=overlap_strength, best_strength=best_strength)
    assert seed.seed.offset_vu == offset


def test_ring_edge_nav_keeps_the_seed_off_the_sensor_boundary(
    flat_polyline: FlatPolylineFactory,
    make_ring_feature: NavFeatureFactory,
    make_nav_context: NavContextFactory,
) -> None:
    """The step onto the zero-padded margin does not pull the seed or fail the fit.

    The sensor ends at row 130, where the bright frame drops to the empty
    margin: a step ten times the ring step at row 120.  The model sits between
    them; without the sensor-border exclusion the strength check would move
    the seed onto the frame border, and the border step would set the strength
    the matched ring edge is measured against.
    """
    shape = (200, 160)
    rows = np.arange(shape[0], dtype=np.float64)[:, None] * np.ones((1, shape[1]))
    image = np.where(rows < 130.0, 100.0, 0.0) + np.where(rows < 120.0, 10.0, 0.0)
    context = make_nav_context(image)
    sensor = rows < 130.0
    vertices, outward = flat_polyline(124.5, 20.0, 140.0, 121)
    feature = make_ring_feature(
        'edge', vertices=vertices, outward_normals=outward, is_straight_line=True
    )
    result = RingEdgeNav().navigate(
        [feature], dataclasses.replace(context, image_ext=image, sensor_mask_ext=sensor)
    )
    assert result.offset_px[0] == pytest.approx(-5.0, abs=0.5)
    assert result.spurious is False


def _three_edge_alias_scene(
    flat_polyline: FlatPolylineFactory, make_ring_feature: NavFeatureFactory
) -> tuple[np.ndarray, list[NavFeature]]:
    """Return three modeled ring edges and an image that offers a radial alias.

    The model edges lie at rows 50, 70 and 90; the image's edges lie 2 px
    below, 1 px above and 1 px below them, each a step of 6.  A step of 10 at
    row 96 is the strongest edge in the frame.  No shift puts more than one
    model edge exactly on an edge pixel, so the exact scores prefer the shift
    that lays one model edge on the strong step.
    """
    shape = (200, 160)
    rows = np.arange(shape[0], dtype=np.float64)[:, None] * np.ones((1, shape[1]))
    image = 6.0 * (rows < 52.0) + 6.0 * (rows < 69.0) + 6.0 * (rows < 91.0) + 10.0 * (rows < 97.0)
    features = []
    for index, row in enumerate((50.0, 70.0, 90.0)):
        vertices, outward = flat_polyline(row, 20.0, 140.0, 121)
        features.append(
            make_ring_feature(
                f'edge{index}', vertices=vertices, outward_normals=outward, is_straight_line=True
            )
        )
    return image, features


def test_ring_edge_nav_seeds_where_every_edge_lies_near_its_image_edge(
    flat_polyline: FlatPolylineFactory,
    make_ring_feature: NavFeatureFactory,
    make_nav_context: NavContextFactory,
) -> None:
    """The basin check moves the seed off the strong step to the three-edge alignment."""
    image, features = _three_edge_alias_scene(flat_polyline, make_ring_feature)
    result = RingEdgeNav().navigate(features, make_nav_context(image))
    assert abs(result.offset_px[0]) < 2.5


def test_ring_edge_nav_keeps_a_fit_the_basin_check_seeded(
    flat_polyline: FlatPolylineFactory,
    make_ring_feature: NavFeatureFactory,
    make_nav_context: NavContextFactory,
) -> None:
    """The coarse-acquisition gate reads the basin's best exact placement and passes it.

    No vertex of the three-edge alignment lies exactly on an edge pixel, so the
    tolerant maximum's own plain match fraction is zero.
    """
    image, features = _three_edge_alias_scene(flat_polyline, make_ring_feature)
    result = RingEdgeNav().navigate(features, make_nav_context(image))
    assert result.spurious is False


def test_ring_edge_nav_records_the_basin_ratio(
    flat_polyline: FlatPolylineFactory,
    make_ring_feature: NavFeatureFactory,
    make_nav_context: NavContextFactory,
) -> None:
    """The diagnostic carries the replaced seed's ratio, below the keep ratio."""
    image, features = _three_edge_alias_scene(flat_polyline, make_ring_feature)
    result = RingEdgeNav().navigate(features, make_nav_context(image))
    assert isinstance(result.diagnostics, RingEdgeDiagnostics)
    assert result.diagnostics.coarse_basin_ratio < 0.6


def test_ring_edge_nav_applies_the_configured_basin_keep_ratio(
    flat_polyline: FlatPolylineFactory,
    make_ring_feature: NavFeatureFactory,
    make_nav_context: NavContextFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A keep ratio of zero keeps every seed, so the alias stands."""
    tuning = dict(RingEdgeNav.tuning)
    tuning['coarse_basin_keep_ratio'] = 0.0
    monkeypatch.setattr(RingEdgeNav, 'tuning', tuning)
    image, features = _three_edge_alias_scene(flat_polyline, make_ring_feature)
    result = RingEdgeNav().navigate(features, make_nav_context(image))
    assert result.offset_px[0] > 5.0


def test_ring_edge_nav_without_the_basin_check_keeps_the_alias(
    flat_polyline: FlatPolylineFactory,
    make_ring_feature: NavFeatureFactory,
    make_nav_context: NavContextFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no tolerance the check repeats the exact score and the alias stands."""
    tuning = dict(RingEdgeNav.tuning)
    tuning['coarse_basin_tolerance_px'] = 0
    monkeypatch.setattr(RingEdgeNav, 'tuning', tuning)
    image, features = _three_edge_alias_scene(flat_polyline, make_ring_feature)
    result = RingEdgeNav().navigate(features, make_nav_context(image))
    assert result.offset_px[0] > 5.0


@pytest.mark.parametrize('tolerance', [-1, True, 1.5])
def test_ring_edge_nav_rejects_a_basin_tolerance_that_is_not_a_non_negative_int(
    tolerance: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A negative, boolean, or fractional basin tolerance fails at construction."""
    bad_tuning = dict(RingEdgeNav.tuning)
    bad_tuning['coarse_basin_tolerance_px'] = tolerance
    monkeypatch.setattr(RingEdgeNav, 'tuning', bad_tuning)
    with pytest.raises(ValueError, match='coarse_basin_tolerance_px must be a non-negative'):
        RingEdgeNav()


@pytest.mark.parametrize('count', [0, True, 2.5])
def test_ring_edge_nav_rejects_a_basin_shared_count_that_is_not_a_positive_int(
    count: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A zero, boolean, or fractional minimum shared count fails at construction."""
    bad_tuning = dict(RingEdgeNav.tuning)
    bad_tuning['coarse_basin_min_shared_vertices'] = count
    monkeypatch.setattr(RingEdgeNav, 'tuning', bad_tuning)
    with pytest.raises(ValueError, match='coarse_basin_min_shared_vertices must be a positive'):
        RingEdgeNav()


@pytest.mark.parametrize('ratio', [-0.1, 1.5, math.nan])
def test_ring_edge_nav_rejects_a_basin_keep_ratio_outside_unit_range(
    ratio: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A basin keep ratio outside [0, 1], or NaN, fails at construction."""
    bad_tuning = dict(RingEdgeNav.tuning)
    bad_tuning['coarse_basin_keep_ratio'] = ratio
    monkeypatch.setattr(RingEdgeNav, 'tuning', bad_tuning)
    with pytest.raises(ValueError, match='coarse_basin_keep_ratio must lie in'):
        RingEdgeNav()


class _FixedTolerant:
    """Stand-in for ``TolerantEdgeStrength`` answering fixed searches and scores."""

    def __init__(
        self, best: tuple[int, int], strengths: dict[tuple[int, int], float], shared: int
    ) -> None:
        """Record the search's answer, the score of each shift and the shared count.

        Parameters:
            best: The shift the search returns.
            strengths: The tolerant score of each shift.
            shared: The number of vertices both shifts keep on the sensor.
        """
        self._best = best
        self._strengths = strengths
        self._shared = shared

    def search(self, _window: tuple[int, int]) -> CoarseSearchResult:
        """Return the fixed best shift."""
        return CoarseSearchResult(offset_vu=self._best, score=0.5)

    def compare(self, first: tuple[int, int], second: tuple[int, int]) -> tuple[float, float, int]:
        """Return the fixed credits of the two shifts and the shared count."""
        return self._strengths[first], self._strengths[second], self._shared

    def best_match_fraction(self, _offset: tuple[int, int]) -> float:
        """Return a fixed best exact match fraction."""
        return 0.375


def _basin_check_with(
    monkeypatch: pytest.MonkeyPatch,
    *,
    best: tuple[int, int] = (5, 5),
    seed_strength: float = 1.0,
    basin_strength: float = 1.0,
    shared: int = 500,
) -> tuple[CoarseSearchResult, SeedBasinCheck]:
    """Run ``check_seed_basin`` on the seed ``(1, 1)`` with fixed tolerant answers.

    The tolerant search answers ``best``; the tolerant strengths of the seed and
    of ``(5, 5)`` are the matching arguments.
    """
    strengths = {(1, 1): seed_strength, (5, 5): basin_strength}
    monkeypatch.setattr(
        ring_edge_seed,
        'TolerantEdgeStrength',
        lambda *_args, **_kwargs: _FixedTolerant(best, strengths, shared),
    )
    seed = CoarseSearchResult(offset_vu=(1, 1), score=1.0)
    empty = np.zeros((4, 4), dtype=bool)
    check = check_seed_basin(
        seed,
        edge_mask=empty,
        gradient_vu=np.zeros((4, 4, 2)),
        vertices=np.zeros((1, 2)),
        normals=np.zeros((1, 2)),
        search_window_vu=(1, 1),
        valid_mask=empty,
        tolerance_px=2,
        keep_ratio=0.6,
        min_shared_vertices=100,
    )
    return seed, check


@pytest.mark.parametrize(
    ('case', 'offset'), [('at_ratio', (1, 1)), ('just_below', (5, 5)), ('no_strength', (1, 1))]
)
def test_basin_check_keeps_the_seed_at_or_above_the_keep_ratio(
    case: str, offset: tuple[int, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The basin keep ratio is inclusive, and no strength anywhere keeps the seed."""
    seed_strength, basin_strength = {
        'at_ratio': (0.6, 1.0),
        'just_below': (float(np.nextafter(0.6, 0.0)), 1.0),
        'no_strength': (0.0, 0.0),
    }[case]
    _seed, check = _basin_check_with(
        monkeypatch, seed_strength=seed_strength, basin_strength=basin_strength
    )
    assert check.seed.offset_vu == offset


def test_basin_check_reports_the_seed_ratio_to_the_tolerant_maximum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reported ratio is the seed's tolerant strength over the maximum's."""
    _seed, check = _basin_check_with(monkeypatch, seed_strength=0.25, basin_strength=1.0)
    assert check.ratio == pytest.approx(0.25)


def test_basin_check_keeps_a_seed_that_is_the_tolerant_maximum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the tolerant search lands on the seed itself the seed stands at ratio 1."""
    seed, check = _basin_check_with(monkeypatch, best=(1, 1), seed_strength=0.0)
    assert check.seed is seed


def test_basin_check_keeps_the_seed_when_too_few_vertices_are_shared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With fewer shared vertices than the minimum there is nothing to compare."""
    _seed, check = _basin_check_with(monkeypatch, seed_strength=0.1, basin_strength=1.0, shared=99)
    assert check.seed.offset_vu == (1, 1)


def test_basin_check_reports_a_ratio_above_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """A seed earning more than the maximum on the shared vertices reports the excess."""
    _seed, check = _basin_check_with(monkeypatch, seed_strength=2.0, basin_strength=1.0)
    assert check.ratio == pytest.approx(2.0)


def test_basin_check_gives_a_replacement_the_basin_match_fraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A replacing maximum carries the best exact match fraction within the tolerance."""
    _seed, check = _basin_check_with(monkeypatch, seed_strength=0.1, basin_strength=1.0)
    assert check.seed.score == pytest.approx(0.375)
