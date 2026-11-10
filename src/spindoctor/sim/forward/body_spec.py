"""The typed, frozen form of one scene ``bodies`` entry.

Every key of a scene body is optional, and the validator reads a key written as
null as if it were left out.  :func:`body_spec_from_params` parses a body mapping
once into a :class:`BodySpec`.  Every default is applied there, and a null takes
the same default an absent key takes, so the renderer reads typed fields and never
meets a null.

Two sub-blocks stay as their validated mappings: ``atmosphere``, whose pixel
lengths scale with the render grid's oversampling, and ``albedo_texture``, whose
noise field draws its seed from the body's identity.  The renderer builds their
specs when it knows both, through
:func:`~spindoctor.sim.forward.atmosphere.atmosphere_spec_from_params` and
:func:`~spindoctor.sim.forward.body_texture.albedo_spec_from_params`, which read
a null key as absent too.  The renderer does not read ``km_per_pixel``, which
describes the body to the navigator, or ``nav_override``, which is what the
navigator believes, so the spec leaves both out.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from spindoctor.sim.forward.body_texture import (
    DiscTextureSpec,
    TransitSpec,
    disc_texture_spec_from_params,
    transit_specs_from_params,
)
from spindoctor.sim.mesh_geometry import MeshBodySpec, mesh_spec_from_params
from spindoctor.sim.scene_schema import default_body_name, scene_value

__all__ = [
    'CRATER_MAX_RADIUS_DEFAULT',
    'CRATER_MIN_RADIUS_DEFAULT',
    'BodySpec',
    'body_spec_from_params',
]

CRATER_MIN_RADIUS_DEFAULT: float = 0.05
"""The smallest crater radius a body draws when it leaves the key out, as a
fraction of axis1."""

CRATER_MAX_RADIUS_DEFAULT: float = 0.25
"""The largest crater radius a body draws when it leaves the key out, as a
fraction of axis1."""

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

    The fields carry the scene keys of the same names, in the scene's units:
    pixels for centers and axes, and degrees for angles.  A body the scene
    radiance stage renders carries its render-grid values, which are the scene's
    centers and axes scaled by the oversampling factor, with the planted
    pointing applied to the center and roll.

    Parameters:
        name: The body's name as written, or ``SIM-BODY-<index + 1>`` when the
            scene leaves it out.
        shape_model: ``'ellipsoid'`` or ``'polyhedral_mesh'``.
        center_v: The body center's v coordinate, or None for the frame
            center.
        center_u: The body center's u coordinate, or None for the frame
            center.
        axis1: The full width along axis 1.
        axis2: The full width along axis 2.
        axis3: The full width along axis 3, which defaults to the smaller of
            ``axis1`` and ``axis2``.
        rotation_z: The in-plane rotation, in degrees.
        rotation_tilt: The tilt of the polar axis toward the observer, in
            degrees.
        illumination_angle: The image-plane azimuth of the light, in degrees.
        phase_angle: The phase angle, in degrees.
        range_km: The observer distance, or None when the scene leaves it out.
        crater_fill: The fraction of the surface covered by craters.
        crater_min_radius: The smallest crater radius, a fraction of axis1.
        crater_max_radius: The largest crater radius, a fraction of axis1.
        crater_power_law_exponent: The exponent of the crater-radius
            distribution.
        crater_relief_scale: The crater depth scale.
        anti_aliasing: The limb supersampling control.
        limb_relief_rms: The limb-relief amplitude, a fraction of the radius.
        limb_relief_corr_deg: The limb-relief correlation length, in degrees.
        photometric_law: The surface scattering law.
        minnaert_k: The Minnaert exponent.
        surge_amplitude: The opposition-surge amplitude; 0 is no surge.
        surge_width_deg: The opposition-surge width, in degrees of phase.
        seed: The body's own seed, or None.
        shading: A mesh body's shading mode, ``'flat'`` or ``'gouraud'``.
        pose_scatter_sigma_deg: The per-axis sigma of a mesh body's pose
            scatter, in degrees; 0 is no scatter.
        mesh: A mesh body's shape and pose.
        albedo_texture: The validated ``albedo_texture`` mapping, or None.
        disc_texture: The banded disc texture, or None.
        transits: The transiting moons and shadows, or None when the scene has
            no ``transits`` list.  An empty list is an empty tuple.
        atmosphere: The validated ``atmosphere`` mapping, or None.
    """

    name: str
    shape_model: str
    center_v: float | None
    center_u: float | None
    axis1: float
    axis2: float
    axis3: float
    rotation_z: float
    rotation_tilt: float
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
    surge_amplitude: float
    surge_width_deg: float
    seed: int | None
    shading: str
    pose_scatter_sigma_deg: float
    mesh: MeshBodySpec
    albedo_texture: Mapping[str, Any] | None
    disc_texture: DiscTextureSpec | None
    transits: tuple[TransitSpec, ...] | None
    atmosphere: Mapping[str, Any] | None

    @property
    def has_texture(self) -> bool:
        """Whether the body carries an albedo texture, a disc texture, or transits."""
        return (
            self.albedo_texture is not None
            or self.disc_texture is not None
            or self.transits is not None
        )


def _optional_float(block: Mapping[str, Any], key: str) -> float | None:
    """The key's value as a float, or None when it is absent or null."""
    value = block.get(key)
    return None if value is None else float(value)


def body_spec_from_params(body_params: Mapping[str, Any], *, index: int) -> BodySpec:
    """Parse one validated scene body into a :class:`BodySpec`.

    A key that is absent or null takes its default.  The mapping is only read.

    Parameters:
        body_params: One entry of the scene's ``bodies`` list.
        index: The entry's position in that list, which sets the default name.

    Returns:
        The parsed body.
    """
    axis1 = float(scene_value(body_params, 'axis1', 0.0))
    axis2 = float(scene_value(body_params, 'axis2', 0.0))
    seed = body_params.get('seed')
    surge = scene_value(body_params, 'opposition_surge', {})
    scatter = scene_value(body_params, 'pose_scatter', {})
    transits = body_params.get('transits')
    return BodySpec(
        name=str(scene_value(body_params, 'name', default_body_name(index))),
        shape_model=str(scene_value(body_params, 'shape_model', 'ellipsoid')),
        center_v=_optional_float(body_params, 'center_v'),
        center_u=_optional_float(body_params, 'center_u'),
        axis1=axis1,
        axis2=axis2,
        axis3=float(scene_value(body_params, 'axis3', min(axis1, axis2))),
        rotation_z=float(scene_value(body_params, 'rotation_z', 0.0)),
        rotation_tilt=float(scene_value(body_params, 'rotation_tilt', 0.0)),
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
        surge_amplitude=float(scene_value(surge, 'amplitude', 0.0)),
        surge_width_deg=float(scene_value(surge, 'width_deg', _SURGE_WIDTH_DEG_DEFAULT)),
        seed=None if seed is None else int(seed),
        shading=str(scene_value(body_params, 'shading', 'flat')),
        pose_scatter_sigma_deg=float(scene_value(scatter, 'sigma_deg', 0.0)),
        mesh=mesh_spec_from_params(body_params),
        albedo_texture=body_params.get('albedo_texture'),
        disc_texture=disc_texture_spec_from_params(body_params),
        transits=None if transits is None else transit_specs_from_params(body_params),
        atmosphere=body_params.get('atmosphere'),
    )
