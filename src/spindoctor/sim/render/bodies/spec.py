"""The typed, frozen form of one scene ``bodies`` entry.

Every key of a scene body is optional, and the validator refuses a key written as
null, so a scene leaves a key out to take its default.  :func:`body_spec_from_params`
parses a body mapping once into a :class:`BodySpec`.  Every default is applied there,
so the renderer reads typed fields.

Two sub-blocks stay as their validated mappings: ``atmosphere``, whose pixel
lengths scale with the render grid's oversampling, and ``albedo_texture``, whose
noise field draws its seed from the body's identity.  The renderer builds their
specs once it knows the oversampling factor and the body's identity seed, through
:func:`~spindoctor.sim.render.bodies.atmosphere.atmosphere_spec_from_params` and
:func:`~spindoctor.sim.render.bodies.texture.albedo_spec_from_params`, which apply
the defaults of those blocks.  The renderer does not read ``km_per_pixel``, which
describes the body to the navigator, or ``nav_override``, which holds the values
the navigator is given in place of the truth, so the spec leaves both out.

:func:`body_identity_seed` chooses the seed that every random stream of a body,
such as its craters, its relief, or its albedo noise, derives from, and
:func:`scale_by_normal_albedo` scales a body's shaded surface by its normal albedo.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from spindoctor.sim.render.bodies.texture import (
    DiscTextureSpec,
    TransitSpec,
    disc_texture_spec_from_params,
    transit_specs_from_params,
)
from spindoctor.sim.scene.fields import default_body_name, scene_value
from spindoctor.sim.seeds import derive_effect_seed
from spindoctor.sim.shared.mesh import MeshBodySpec, mesh_spec_from_params
from spindoctor.support.types import NDArrayFloatType

__all__ = [
    'CRATER_MAX_RADIUS_DEFAULT',
    'CRATER_MIN_RADIUS_DEFAULT',
    'NORMAL_ALBEDO_DEFAULT',
    'BodySpec',
    'body_identity_seed',
    'body_spec_from_params',
    'scale_by_normal_albedo',
]

CRATER_MIN_RADIUS_DEFAULT: float = 0.05
"""The smallest crater radius a body draws when it leaves the key out, as a
fraction of semi-axis a."""

CRATER_MAX_RADIUS_DEFAULT: float = 0.25
"""The largest crater radius a body draws when it leaves the key out, as a
fraction of semi-axis a."""

NORMAL_ALBEDO_DEFAULT: float = 1.0
"""The normal albedo a body takes when it leaves the key out: a white surface."""

_CRATER_POWER_LAW_EXPONENT_DEFAULT = 3.0
_CRATER_RELIEF_SCALE_DEFAULT = 0.6
_ANTI_ALIASING_DEFAULT = 1.0
_LIMB_RELIEF_CORR_DEG_DEFAULT = 15.0
_PHOTOMETRIC_LAW_DEFAULT = 'lambert'
_MINNAERT_K_DEFAULT = 0.5
_SURGE_WIDTH_DEG_DEFAULT = 6.0


@dataclass(frozen=True)
class BodySpec:
    """One scene body with every default applied.

    Most fields carry the scene key of the same name, in the scene's units:
    pixels for centers and semi-axes, and degrees for angles.  The entries below
    note the fields that come from other keys.

    When the scene radiance stage renders a body, it fills the spec with
    render-grid values.  It shifts the center by the ephemeris parallax and
    applies the planted roll, which rotates the center about the frame center and
    turns the body about its own center by the same angle
    (:func:`~spindoctor.sim.render.roll.roll_body_params`), and records
    the roll in ``roll_deg``.  It then scales the center and semi-axes by the
    oversampling factor.  :func:`~spindoctor.sim.render.bodies.render.render_single_body`
    adds the planted offset, turned by the roll, when it draws the body.

    Parameters:
        name: The body's name as written, or ``SIM-BODY-<index + 1>`` when the
            scene leaves it out.
        shape_model: ``'ellipsoid'`` or ``'polyhedral_mesh'``.
        center_v: The body center's v coordinate, or None for the frame
            center.
        center_u: The body center's u coordinate, or None for the frame
            center.
        semi_a_px: Semi-axis a, half the width along axis 1.
        semi_b_px: Semi-axis b, half the width along axis 2.
        semi_c_px: Semi-axis c, half the depth along axis 3, which defaults to
            the smaller of ``semi_a_px`` and ``semi_b_px``.
        rotation_z: The in-plane rotation, in degrees.
        sub_observer_lat_deg: The latitude of the point on the body directly below
            the observer, in degrees: positive when the north pole, the end of
            semi-axis a on +v before ``rotation_z``, tips toward the observer.
        illumination_angle: The image-plane azimuth of the light, in degrees.
        phase_angle: The phase angle, in degrees.
        range_km: The observer distance, or None when the scene leaves it out.
        crater_fill: The fraction of the surface covered by craters.
        crater_min_radius: The smallest crater radius, a fraction of ``semi_a_px``.
        crater_max_radius: The largest crater radius, a fraction of ``semi_a_px``.
        crater_power_law_exponent: The exponent of the crater-radius
            distribution.
        crater_relief_scale: The crater depth scale.
        anti_aliasing: The limb supersampling control.
        limb_relief_rms: The limb-relief amplitude, a fraction of the radius.
        limb_relief_corr_deg: The limb-relief correlation length, in degrees.
        photometric_law: The surface scattering law.
        minnaert_k: The Minnaert exponent.
        normal_albedo: The surface's normal albedo, its I/F when lit and viewed
            head-on at zero phase.  It multiplies the law's value, which is 1
            there.
        surge_amplitude: The ``opposition_surge`` block's ``amplitude``.  Zero
            turns the surge off.
        surge_width_deg: The ``opposition_surge`` block's ``width_deg``, in
            degrees of phase.
        shading: A mesh body's shading mode, ``'flat'`` or ``'gouraud'``.
        pose_scatter_sigma_deg: The ``pose_scatter`` block's ``sigma_deg``, the
            per-axis sigma of a mesh body's pose scatter in degrees.  Zero turns
            the scatter off.
        mesh: A mesh body's shape and pose, built from the ``mesh_*`` keys and
            ``pose_euler_deg``.
        albedo_texture: The validated ``albedo_texture`` mapping, or None.
        disc_texture: The banded disc texture, or None.
        transits: The transiting moons and shadows, or None when the scene leaves
            out the ``transits`` list.  An empty list is an empty tuple.
        atmosphere: The validated ``atmosphere`` mapping, or None.
        roll_deg: The planted roll the body was turned by, in degrees, which the
            scene radiance stage sets; 0 for the body as the scene states it.  The
            other fields already carry the roll, except the crater layout, which
            is drawn on the image grid: the ellipsoid renderer lays the craters
            out for the unrolled body and turns them by this angle.
    """

    name: str
    shape_model: str
    center_v: float | None
    center_u: float | None
    semi_a_px: float
    semi_b_px: float
    semi_c_px: float
    rotation_z: float
    sub_observer_lat_deg: float
    illumination_angle: float
    phase_angle: float
    range_km: float | None
    crater_fill: float
    crater_min_radius: float
    crater_max_radius: float
    crater_power_law_exponent: float
    crater_relief_scale: float
    anti_aliasing: float
    limb_relief_rms: float
    limb_relief_corr_deg: float
    photometric_law: str
    minnaert_k: float
    normal_albedo: float
    surge_amplitude: float
    surge_width_deg: float
    shading: str
    pose_scatter_sigma_deg: float
    mesh: MeshBodySpec
    albedo_texture: Mapping[str, Any] | None
    disc_texture: DiscTextureSpec | None
    transits: tuple[TransitSpec, ...] | None
    atmosphere: Mapping[str, Any] | None
    roll_deg: float = 0.0

    @property
    def has_texture(self) -> bool:
        """Whether the body carries an albedo texture, a disc texture, or transits."""
        return (
            self.albedo_texture is not None
            or self.disc_texture is not None
            or self.transits is not None
        )


def _optional_float(block: Mapping[str, Any], key: str) -> float | None:
    """The key's value as a float, or None when it is absent."""
    value = block.get(key)
    return None if value is None else float(value)


def body_spec_from_params(body_params: Mapping[str, Any], *, index: int) -> BodySpec:
    """Parse one validated scene body into a :class:`BodySpec`.

    A key that the entry leaves out takes its default.  The mapping is only read.

    Parameters:
        body_params: One entry of the scene's ``bodies`` list.
        index: The entry's position in that list, which sets the default name.

    Returns:
        The parsed body.
    """
    semi_a_px = float(scene_value(body_params, 'semi_a_px', 0.0))
    semi_b_px = float(scene_value(body_params, 'semi_b_px', 0.0))
    surge = scene_value(body_params, 'opposition_surge', {})
    scatter = scene_value(body_params, 'pose_scatter', {})
    transits = body_params.get('transits')
    return BodySpec(
        name=str(scene_value(body_params, 'name', default_body_name(index))),
        shape_model=str(scene_value(body_params, 'shape_model', 'ellipsoid')),
        center_v=_optional_float(body_params, 'center_v'),
        center_u=_optional_float(body_params, 'center_u'),
        semi_a_px=semi_a_px,
        semi_b_px=semi_b_px,
        semi_c_px=float(scene_value(body_params, 'semi_c_px', min(semi_a_px, semi_b_px))),
        rotation_z=float(scene_value(body_params, 'rotation_z', 0.0)),
        sub_observer_lat_deg=float(scene_value(body_params, 'sub_observer_lat_deg', 0.0)),
        illumination_angle=float(scene_value(body_params, 'illumination_angle', 0.0)),
        phase_angle=float(scene_value(body_params, 'phase_angle', 0.0)),
        range_km=_optional_float(body_params, 'range_km'),
        crater_fill=float(scene_value(body_params, 'crater_fill', 0.0)),
        crater_min_radius=float(
            scene_value(body_params, 'crater_min_radius', CRATER_MIN_RADIUS_DEFAULT)
        ),
        crater_max_radius=float(
            scene_value(body_params, 'crater_max_radius', CRATER_MAX_RADIUS_DEFAULT)
        ),
        crater_power_law_exponent=float(
            scene_value(
                body_params, 'crater_power_law_exponent', _CRATER_POWER_LAW_EXPONENT_DEFAULT
            )
        ),
        crater_relief_scale=float(
            scene_value(body_params, 'crater_relief_scale', _CRATER_RELIEF_SCALE_DEFAULT)
        ),
        anti_aliasing=float(scene_value(body_params, 'anti_aliasing', _ANTI_ALIASING_DEFAULT)),
        limb_relief_rms=float(scene_value(body_params, 'limb_relief_rms', 0.0)),
        limb_relief_corr_deg=float(
            scene_value(body_params, 'limb_relief_corr_deg', _LIMB_RELIEF_CORR_DEG_DEFAULT)
        ),
        photometric_law=str(scene_value(body_params, 'photometric_law', _PHOTOMETRIC_LAW_DEFAULT)),
        minnaert_k=float(scene_value(body_params, 'minnaert_k', _MINNAERT_K_DEFAULT)),
        normal_albedo=float(scene_value(body_params, 'normal_albedo', NORMAL_ALBEDO_DEFAULT)),
        surge_amplitude=float(scene_value(surge, 'amplitude', 0.0)),
        surge_width_deg=float(scene_value(surge, 'width_deg', _SURGE_WIDTH_DEG_DEFAULT)),
        shading=str(scene_value(body_params, 'shading', 'flat')),
        pose_scatter_sigma_deg=float(scene_value(scatter, 'sigma_deg', 0.0)),
        mesh=mesh_spec_from_params(body_params),
        albedo_texture=body_params.get('albedo_texture'),
        disc_texture=disc_texture_spec_from_params(body_params),
        transits=None if transits is None else transit_specs_from_params(body_params),
        atmosphere=body_params.get('atmosphere'),
    )


def body_identity_seed(body_streams_seed: int, body: BodySpec, body_index: int) -> int:
    """The seed every random stream of one body derives from.

    The scene's seed for the bodies' streams is mixed with the body's index and its
    upper-cased name, so two bodies of the same geometry draw independent surfaces:
    ``derive_effect_seed(body_streams_seed, 'body:<index>:<NAME>')``.

    Parameters:
        body_streams_seed: The scene's seed for the bodies' streams.
        body: The body (its name is read).
        body_index: The body's position in the scene's body list.

    Returns:
        The body's identity seed, below 2**32.
    """
    return derive_effect_seed(body_streams_seed, f'body:{body_index}:{body.name.upper()}')


def scale_by_normal_albedo(intensity: NDArrayFloatType, body: BodySpec) -> NDArrayFloatType:
    """A body's shaded surface scaled by its normal albedo.

    Parameters:
        intensity: The surface as the photometric law shades it, 1 where the surface
            is lit and viewed head-on at zero phase.  It is only read.
        body: The body (its ``normal_albedo`` is read).

    Returns:
        ``intensity`` itself when the normal albedo is 1, so a body that leaves the key
        out renders bit for bit as an unscaled one; otherwise a new array.
    """
    if body.normal_albedo == NORMAL_ALBEDO_DEFAULT:
        return intensity
    return np.asarray(intensity * body.normal_albedo, dtype=np.float64)
