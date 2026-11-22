"""The simulated body model's silhouette overflow.

The overflow is the share of the whole predicted silhouette that falls outside the
frame.  It decides the body's disc and blob features, as the SPICE-backed model's
overflow does.
"""

import math
from typing import Any

import numpy as np
import pytest
from tests.shims import bare_nav_context
from tests.spindoctor.nav_model.simulated.test_body import (
    _SIZE,
    _body_feature_types,
    _body_params,
    _mesh_params,
    _obs,
)

from spindoctor.nav_model.simulated.body import NavModelBodySimulated

# ---------------------------------------------------------------------------
# Silhouette overflow: the SPICE-backed model's BODY_DISC and BODY_BLOB gates.
# ---------------------------------------------------------------------------


def _sphere(center_u: float) -> dict[str, Any]:
    """A 60 px lit sphere centered on the frame's middle row at ``center_u``."""
    return _body_params(
        name='MIMAS',
        semi_a_px=30.0,
        semi_b_px=30.0,
        semi_c_px=30.0,
        phase_angle=20.0,
        center_u=center_u,
    )


def _overflow_model(body_params: dict[str, Any]) -> NavModelBodySimulated:
    """The built model for ``body_params`` on the standard obs."""
    model = NavModelBodySimulated('body', _obs(), body_params['name'], body_params)
    model.create_model()
    return model


@pytest.mark.parametrize(
    'body_params',
    [_sphere(_SIZE / 2.0 + 0.3), _mesh_params()],
    ids=['sphere', 'mesh'],
)
def test_a_framed_body_has_zero_overflow(body_params: dict[str, Any]) -> None:
    """A body wholly inside the frame overflows by exactly 0."""
    assert _overflow_model(body_params)._overflow_fraction == 0.0


def test_a_body_centered_on_the_frame_edge_overflows_by_half() -> None:
    """A sphere centered on the left edge has about half its silhouette off frame."""
    assert _overflow_model(_sphere(0.0))._overflow_fraction == pytest.approx(0.5, abs=0.02)


def test_a_body_half_off_the_frame_withholds_its_disc_and_blob() -> None:
    """Past both overflow limits the model emits neither a disc nor a blob."""
    types = _body_feature_types(_obs(), _sphere(0.0))
    assert 'BODY_DISC' not in types
    assert 'BODY_BLOB' not in types


def test_a_slightly_clipped_body_keeps_its_disc_and_withholds_its_blob() -> None:
    """About 11 percent off frame is under the disc limit and over the blob limit."""
    model = _overflow_model(_sphere(20.0))
    assert 0.05 < model._overflow_fraction < 0.3
    types = [f.feature_type.name for f in model.to_features(bare_nav_context(_obs()))]
    assert 'BODY_DISC' in types
    assert 'BODY_BLOB' not in types


def test_the_disc_reports_its_overflow() -> None:
    """The emitted disc carries the model's overflow fraction in its geometry."""
    model = _overflow_model(_sphere(20.0))
    disc = next(
        f for f in model.to_features(bare_nav_context(_obs())) if f.feature_type.name == 'BODY_DISC'
    )
    assert disc.geometry.overflow_fraction == model._overflow_fraction  # type: ignore[union-attr]


def test_a_frame_filling_body_counts_its_overflow_on_a_bounded_canvas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A body wider than the overflow canvas cap counts on a downsampled canvas.

    The cap is lowered so a 300 px sphere on the 96 px frame needs downsampling;
    its overflow then matches the full-resolution count to within a percent, and
    no render exceeds the cap.
    """
    from spindoctor.nav_model.simulated import body as module

    body = _sphere(_SIZE / 2.0) | {'semi_a_px': 150.0, 'semi_b_px': 150.0, 'semi_c_px': 150.0}
    full = _overflow_model(body)._overflow_fraction
    sizes: list[int] = []
    original = NavModelBodySimulated._render_body_image

    def recording(self: NavModelBodySimulated, *args: Any, **kwargs: Any) -> Any:
        size = kwargs.get('size')
        if size is not None:
            sizes.append(max(size))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(module, '_OVERFLOW_CANVAS_MAX_PX', 128)
    monkeypatch.setattr(NavModelBodySimulated, '_render_body_image', recording)
    bounded = _overflow_model(body)._overflow_fraction
    assert bounded == pytest.approx(full, abs=0.01)
    assert max(sizes) <= 128


# A body flattened across (b = 8) and long in depth (c = 40), turned 70 degrees
# toward the observer, so semi-axis c fills most of its outline along the pole.
_TURNED = {
    'name': 'MIMAS',
    'semi_a_px': 30.0,
    'semi_b_px': 8.0,
    'semi_c_px': 40.0,
    'sub_observer_lat_deg': 70.0,
}


def test_a_turned_body_counts_its_whole_silhouette_in_its_overflow() -> None:
    """A turned body near the frame edge reports the overflow of its whole silhouette.

    The reference draws the same prediction on the frame padded by 100 px on every
    side, at the same sub-pixel phase, and counts the share of it outside the frame.
    """
    body = _body_params(**_TURNED, center_v=30.3)
    model = _overflow_model(body)
    pad = 100
    whole = model._render_body_image(
        math.radians(body['phase_angle']),
        size=(_SIZE + 2 * pad, _SIZE + 2 * pad),
        center=(body['center_v'] + pad, body['center_u'] + pad),
    )
    inside = whole[pad : pad + _SIZE, pad : pad + _SIZE]
    expected = 1.0 - np.count_nonzero(inside > 0.0) / np.count_nonzero(whole > 0.0)
    assert model._overflow_fraction == pytest.approx(expected, abs=1e-12)
