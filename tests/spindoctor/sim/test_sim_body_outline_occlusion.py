"""A body occludes by its solid outline, never by how bright it renders.

A star or a farther body behind any part of a body's outline is hidden: behind
the unlit side of a crescent, and behind a pixel a surface stage renders black,
such as a transit's full umbra.  The body's mask, which drives the paint, star
extinction, and the truth maps, covers those pixels too.
"""

from typing import Any

import numpy as np
import pytest

from spindoctor.sim.render import render_combined_model

_SIZE = 128
_CENTER = 64.0


def _body(**extra: Any) -> dict[str, Any]:
    """A 60 px sphere at the frame center, nearer than any other body."""
    return {
        'name': 'NEAR',
        'center_v': _CENTER,
        'center_u': _CENTER,
        'axis1': 60.0,
        'axis2': 60.0,
        'axis3': 60.0,
        'phase_angle': 0.0,
        'illumination_angle': 0.0,
        'range_km': 1.0e6,
        **extra,
    }


def _umbra_body() -> dict[str, Any]:
    """The sphere with a fully dark transit shadow over its center."""
    return _body(
        transits=[{'shadow': {'dv_px': 0.0, 'du_px': 0.0, 'radius_px': 8.0, 'darkness': 1.0}}]
    )


def _scene(*bodies: dict[str, Any], **extra: Any) -> dict[str, Any]:
    """A noiseless coiss_nac frame holding ``bodies``."""
    return {
        'instrument': 'coiss_nac',
        'size_v': _SIZE,
        'size_u': _SIZE,
        'random_seed': 1,
        'exposure_sec': 1.0,
        'bodies': list(bodies),
        **extra,
    }


def _star_light(scene: dict[str, Any], v: float, u: float) -> float:
    """The most light a bright star at ``(v, u)`` adds to ``scene``'s render."""
    without, _meta = render_combined_model(scene)
    with_star, _meta = render_combined_model(
        {**scene, 'stars': [{'name': 'S', 'v': v, 'u': u, 'vmag': 2.0, 'psf_sigma': 1.0}]}
    )
    return float(np.max(with_star - without))


def test_a_full_umbra_renders_black_inside_the_body() -> None:
    """The fixture's umbra really is black, so the tests below probe a zero pixel."""
    image, _meta = render_combined_model(_scene(_umbra_body()))
    sky, _meta = render_combined_model(_scene())
    assert image[64, 64] == sky[64, 64]


def test_the_body_mask_covers_a_full_umbra() -> None:
    """A black umbra pixel still belongs to the body's mask."""
    _image, meta = render_combined_model(_scene(_umbra_body()))
    assert bool(meta['body_masks'][0][64, 64])


def test_a_star_behind_a_full_umbra_is_hidden() -> None:
    """A star stated behind the umbra adds no light to the frame."""
    assert _star_light(_scene(_umbra_body()), 64.5, 64.5) == 0.0


def test_a_farther_body_behind_a_full_umbra_is_hidden() -> None:
    """A larger, farther body behind the umbra doesn't show through it."""
    far = {**_body(name='FAR', axis1=100.0, axis2=100.0, axis3=100.0), 'range_km': 5.0e6}
    alone, _meta = render_combined_model(_scene(_umbra_body()))
    stacked, _meta = render_combined_model(_scene(_umbra_body(), far))
    assert stacked[64, 64] == alone[64, 64]


@pytest.mark.parametrize(
    'extra',
    [{}, {'photometric_law': 'minnaert'}, {'limb_relief_rms': 0.03}],
    ids=['lambert', 'minnaert', 'relief'],
)
def test_a_star_behind_a_crescents_unlit_side_is_hidden(extra: dict[str, Any]) -> None:
    """A star behind the dark side of a 150 degree crescent adds no light."""
    crescent = _body(phase_angle=150.0, **extra)
    assert _star_light(_scene(crescent), 76.5, 64.5) == 0.0
