"""``RingEdgeNav``'s coarse seed against a ring's own interior texture."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from tests.spindoctor.nav_technique.conftest import (
    FlatPolylineFactory,
    NavContextFactory,
    NavFeatureFactory,
)

from spindoctor.nav_technique import nav_technique_ring_edge
from spindoctor.nav_technique.dt_fitting import CoarseSearchResult
from spindoctor.nav_technique.nav_technique_ring_edge import RingEdgeNav


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


@pytest.mark.parametrize('ratio', [-0.1, 1.5])
def test_ring_edge_nav_rejects_a_coarse_strength_keep_ratio_outside_unit_range(
    ratio: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A keep ratio outside [0, 1] fails at construction."""
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
) -> CoarseSearchResult:
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
    assert seed.offset_vu == offset


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
