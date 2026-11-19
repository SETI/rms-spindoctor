"""Crater and no-crater shading share one illumination convention (SIM-3).

A cratered body tilts the same base-surface normals a smooth body is shaded
with (including the rotation_z back-rotation to image coordinates), so a
cratered body's lit hemisphere and terminator match the same body rendered
without craters at any pose.  Shading against the raw depth-field gradient
instead would put the craters in a frame that diverges from the smooth
body's for rotated poses.
"""

import numpy as np
import pytest

from spindoctor.sim.render.bodies.ellipsoid import EllipsoidBodySpec, create_ellipsoid_body

_SIZE = (101, 101)
_CENTER = (50.5, 50.5)
_AXES = (35.0, 15.0, 15.0)
_ILLUMINATION_ANGLE = np.pi / 4
_PHASE_ANGLE = np.pi / 2
_CENTROID_TOLERANCE = 0.5


def _render(*, rotation_z: float, rotation_tilt: float, crater_fill: float) -> np.ndarray:
    """Render the reference body at the given pose and crater fill."""
    return create_ellipsoid_body(
        _SIZE,
        _CENTER,
        EllipsoidBodySpec(
            semi_a_px=_AXES[0],
            semi_b_px=_AXES[1],
            semi_c_px=_AXES[2],
            rotation_z=rotation_z,
            rotation_tilt=rotation_tilt,
            illumination_angle=_ILLUMINATION_ANGLE,
            phase_angle=_PHASE_ANGLE,
            crater_fill=crater_fill,
            anti_aliasing=1.0,
            # A fixed realization that puts craters on the lit side in both poses
            # while its crater-placement asymmetry sits well below the tolerance;
            # the guarded shading-frame defect displaced the centroid by ~9 px, so
            # the margin is wide.
            crater_seed=52,
        ),
    )


def _bright_centroid(img: np.ndarray) -> tuple[float, float]:
    """Intensity-weighted centroid of the bright (lit) side of the body."""
    bright = img * (img > 0.5 * img.max())
    vv, uu = np.mgrid[0 : img.shape[0], 0 : img.shape[1]]
    total = float(bright.sum())
    return float((bright * vv).sum() / total), float((bright * uu).sum() / total)


@pytest.mark.parametrize('axis', [0, 1], ids=['v', 'u'])
@pytest.mark.parametrize(
    ('rotation_z', 'rotation_tilt'),
    [(np.pi / 2, 1.0), (0.0, 0.0)],
    ids=['rotated', 'unrotated'],
)
def test_cratered_body_matches_smooth_bright_centroid(
    rotation_z: float, rotation_tilt: float, axis: int
) -> None:
    """The cratered lit hemisphere matches the smooth one at the same pose.

    Before the shared-convention fix the rotated pose displaced the u centroid
    by roughly 9 pixels, so the tolerance is a strong regression guard; the
    unrotated pose guards the shading paths' agreement at identity pose.
    """
    smooth = _render(rotation_z=rotation_z, rotation_tilt=rotation_tilt, crater_fill=0.0)
    cratered = _render(rotation_z=rotation_z, rotation_tilt=rotation_tilt, crater_fill=0.3)
    delta = abs(_bright_centroid(cratered)[axis] - _bright_centroid(smooth)[axis])
    assert delta < _CENTROID_TOLERANCE


def test_rotated_cratered_body_actually_has_craters() -> None:
    """The rotated cratered render differs from the smooth one (test premise)."""
    smooth = _render(rotation_z=np.pi / 2, rotation_tilt=1.0, crater_fill=0.0)
    cratered = _render(rotation_z=np.pi / 2, rotation_tilt=1.0, crater_fill=0.3)
    assert not np.array_equal(cratered, smooth)
