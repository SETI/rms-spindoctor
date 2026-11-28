"""Tests of the realism match's shared feature extraction: which limbs give FOM 3 widths.

The frames are built in memory, so nothing here reads holdings and the file runs in the
plain unit tier.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
from scipy.special import erfc

from spindoctor.feature.geometry import LimbPolyline
from tests.integration.sim_realism_limb_peak import (
    LimbOrientation,
    LimbSelection,
    tracks_touching,
)
from tests.integration.sim_realism_support import extract_feature_samples

_SIZE = 120
_CENTER = 60.0
_RADIUS = 30.0
_METADATA = {
    'body:MOON': {'phase_angle_deg': 30.0, 'predicted_diameter_px': 60.0},
    'body:OTHER': {'phase_angle_deg': 30.0, 'predicted_diameter_px': 60.0},
}
_BIN = 'limb_width_p0_r0'


def _disc_image() -> np.ndarray:
    """A uniformly bright disc whose edge is blurred by a Gaussian of 0.8 px."""
    vv, uu = np.mgrid[0:_SIZE, 0:_SIZE].astype(np.float64)
    r = np.hypot(vv - _CENTER, uu - _CENTER)
    return np.asarray(0.5 * erfc((r - _RADIUS) / (np.sqrt(2.0) * 0.8)))


def _limb_feature(model: str) -> Any:
    """A feature from ``model`` whose limb is the disc's edge facing +u, 170 deg long.

    Its 90 vertices lie on the edge, at angles from -85 to 85 degrees from +u toward
    +v, so 45 of them lie above the center row and 45 below it.
    """
    angles = np.radians(np.linspace(-85.0, 85.0, 90))
    normals = np.column_stack([np.sin(angles), np.cos(angles)])
    geometry = LimbPolyline(
        vertices_vu=_CENTER + normals * _RADIUS,
        normals_vu=normals,
        sigma_normal_per_vertex_px=np.full(90, 0.5),
        sigma_tangent_per_vertex_px=np.full(90, 0.5),
        bbox_extfov_vu=(0, 0, 1, 1),
    )
    return SimpleNamespace(source_model=model, geometry=geometry)


_ORIENTATION = LimbOrientation((_CENTER, _CENTER), 90.0)


def _samples(features: list[Any], selection: LimbSelection) -> Any:
    """The feature samples the features give when the matched frame renders MOON."""
    return extract_feature_samples(
        _disc_image(),
        features,
        _METADATA,
        offset_vu=(0.0, 0.0),
        extfov_margin_vu=(0, 0),
        saturation_level=np.nan,
        matched_model='body:MOON',
        limb_selection=selection,
    )


def _widths(features: list[Any], selection: LimbSelection | None = None) -> list[float]:
    """The FOM 3 widths the features give, by default with nothing left out."""
    chosen = LimbSelection(_ORIENTATION) if selection is None else selection
    widths: list[float] = _samples(features, chosen).samples.get(_BIN, [])
    return widths


def _upper_sky_taken() -> LimbSelection:
    """A real frame's selection whose other body fills the sky above the center row."""
    vv, uu = np.mgrid[0:_SIZE, 0:_SIZE].astype(np.float64)
    other = (np.hypot(vv - _CENTER, uu - _CENTER) > _RADIUS + 3.0) & (vv < _CENTER)
    return LimbSelection(
        _ORIENTATION,
        clear_track=lambda points, normals, half: ~tracks_touching(other, points, normals, half),
    )


def test_a_width_whose_track_crosses_another_body_is_left_out() -> None:
    """Another body filling the sky above the center row takes the upper 45 widths.

    Each upper vertex's 8 px track reaches beyond the limb into that body; no lower
    vertex's track rises above the center row.
    """
    assert len(_widths([_limb_feature('body:MOON')], _upper_sky_taken())) == 45


def test_a_real_frame_records_the_directions_of_its_widths() -> None:
    """The 45 widths below the center row record directions from 0 to 90 degrees.

    The light comes from +u, so a vertex's direction is its angle from +u toward +v.
    """
    samples = _samples([_limb_feature('body:MOON')], _upper_sky_taken())
    directions = np.asarray(samples.limb_width_directions_deg)
    assert directions.size == 45
    assert bool(np.all((directions > 0.0) & (directions < 90.0)))


def test_a_width_whose_track_leaves_the_frame_records_no_direction() -> None:
    """Each width of a limb whose middle runs off the frame records one direction.

    Moved 25 px toward +u, 28 of the limb's 90 tracks leave the frame and give no width.
    """
    samples = extract_feature_samples(
        _disc_image(),
        [_limb_feature('body:MOON')],
        _METADATA,
        offset_vu=(0.0, 25.0),
        extfov_margin_vu=(0, 0),
        saturation_level=np.nan,
        matched_model='body:MOON',
        limb_selection=LimbSelection(_ORIENTATION),
    )
    widths = samples.samples[_BIN]
    assert len(samples.limb_width_directions_deg) == len(widths)


def test_a_matched_frame_takes_its_widths_near_the_real_widths_directions() -> None:
    """Real widths from 20 to 85 degrees keep the twin's within two spacings of them.

    The vertices lie 1.9 degrees apart, so no kept width lies below 16 degrees.
    """
    angles = np.linspace(-85.0, 85.0, 90)
    selection = LimbSelection(_ORIENTATION, width_deg=tuple(angles[angles >= 20.0]))
    samples = _samples([_limb_feature('body:MOON')], selection)
    assert min(samples.limb_width_directions_deg) > 16.0


def test_a_width_from_a_body_the_matched_frame_lacks_is_left_out() -> None:
    """A second body's limb adds no widths when the matched frame renders only MOON."""
    alone = _widths([_limb_feature('body:MOON')])
    both = _widths([_limb_feature('body:MOON'), _limb_feature('body:OTHER')])
    assert alone
    assert len(both) == len(alone)
