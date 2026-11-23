"""Truth-side multiplicative surface textures for the ellipsoid body renderer.

Three texture families live here, all applied to the body's SHADING (the
photometric-law output) and never to the silhouette, so they change what disc
correlation sees without moving the limb the navigator fits:

- **Albedo texture** (``albedo_texture``): a band-limited multiplicative
  noise field on the surface plus discrete circular albedo spots -- the
  realistic disc contrast of an icy moon.  The noise field reuses the
  relief-field spectral synthesis with its own seeded ``'albedo'`` stream
  (:func:`spindoctor.sim.render.bodies.relief.synthesize_albedo_field`).
- **Disc texture** (``disc_texture``): a low-frequency latitude-banded
  pattern plus discrete storm ovals -- the zones/belts and GRS-class storms
  a giant-planet disc presents to disc correlation.
- **Transits** (``transits``): a moon disc in front of the planet disc
  (bright or dark against the bands) and/or its cast shadow.  Both are
  texture on the rendered disc, not body-on-body illumination coupling: the
  shadow is the sharp, high-contrast circular *false crater* that disc
  correlation and blob techniques can lock onto.

**Coordinate convention.** Surface points come from the shared projection's unit-sphere
parameterization ``(x, y, z)``
(:func:`~spindoctor.sim.shared.ellipsoid.surface_unit_coords`): the body-frame point
under each pixel divided by the semi-axes, along the pole, semi-axis b, and semi-axis c,
so ``x^2 + y^2 + z^2 = 1``. A pixel outside the smooth limb, which relief can cover,
reads the limb point at its azimuth on the outline. On the disc of a body seen equator-on
``x = v_rot / semi_a`` and ``y = u_rot / semi_b``; a sub-observer latitude tips the
pole toward the observer, so the bands curve across the disc. Every texture -- the
albedo noise field, albedo spots, bands, and storms -- lives in one body-polar frame:
the pole is the direction of the body's semi-axis a (``lat = arcsin(x)``), and
longitude is measured from the +b direction toward +c, the axis toward the observer
when the body is seen equator-on (``lon = atan2(z, y)``; 90 deg = the sub-observer
meridian). Texture is terrain attached to the body, so it turns in the image with the
body's ``rotation_z`` pose and sub-observer latitude, bands follow the planet's belts,
and the lat-lon synthesis grid's polar pinch sits at the foreshortened limb points on
the pole axis rather than anywhere on the disc face.

The commanded albedo correlation length is ``corr_px`` in *detector pixels
on the disc*: it is converted to degrees of surface arc using the body's
mean projected radius, so it is exact where the surface is face-on (the
field's equator crosses the disc center) and foreshortens toward the limb
the way real surface texture does.

Transit geometry (``dv_px`` / ``du_px`` offsets from the body center and
``radius_px``) is in detector pixels on the image plane; the renderer scales it
to the grid it draws transits on.  A transiting disc renders only where it
overlaps the parent silhouette -- a moon off the limb is a separate scene body,
not a transit entry.

All keys here are truth keys: the navigator's predicted body is always the
untextured smooth-law template, so every texture is planted model error by
construction.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from spindoctor.sim.render.bodies.relief import synthesize_albedo_field
from spindoctor.sim.scene.fields import scene_value
from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType

__all__ = [
    'AlbedoTextureSpec',
    'DiscTextureSpec',
    'TransitSpec',
    'albedo_spec_from_params',
    'apply_transits',
    'disc_texture_spec_from_params',
    'surface_texture_factor',
    'transit_specs_from_params',
]

# Default albedo-noise correlation length in detector pixels on the disc.
_DEFAULT_ALBEDO_CORR_PX = 20.0


@dataclass(frozen=True)
class AlbedoTextureSpec:
    """One body's ``albedo_texture`` block, resolved for rendering.

    Parameters:
        rms: Global standard deviation of the multiplicative noise field
            (0 disables the field; spots still apply).
        corr_px: Noise correlation length in detector pixels on the disc.
        spots: Circular albedo spots as ``(lat_deg, lon_deg, radius_deg,
            albedo_factor)`` tuples in the body-polar frame.
        seed: The 'albedo' stream seed of the noise-field realization.
    """

    rms: float = 0.0
    corr_px: float = _DEFAULT_ALBEDO_CORR_PX
    spots: tuple[tuple[float, float, float, float], ...] = ()
    seed: int = 0


@dataclass(frozen=True)
class DiscTextureSpec:
    """One body's ``disc_texture`` block (zones/belts plus storm ovals).

    The band factor is ``1 + band_amplitude * cos(band_wavenumber * lat_p
    + band_phase)`` with ``lat_p`` the body-polar latitude in radians, so
    ``band_wavenumber`` counts full cosine cycles per radian of latitude
    (a Saturn-like scene uses ~6-10) and ``band_phase_deg`` slides the
    pattern in latitude.

    Parameters:
        band_amplitude: Multiplicative contrast of the banded pattern
            (0 disables the bands; storms still apply).
        band_wavenumber: Cosine cycles per radian of body-polar latitude.
        band_phase_deg: Phase offset of the band pattern, in degrees.
        storms: Storm ovals as ``(lat_deg, lon_deg, radius_deg,
            albedo_factor)`` tuples in the body-polar frame.
    """

    band_amplitude: float = 0.0
    band_wavenumber: float = 8.0
    band_phase_deg: float = 0.0
    storms: tuple[tuple[float, float, float, float], ...] = ()


@dataclass(frozen=True)
class TransitSpec:
    """One ``transits`` entry: a transiting moon disc and/or its cast shadow.

    Parameters:
        moon: ``(dv_px, du_px, radius_px, albedo_factor)`` -- the moon disc's
            offset from the parent body center, its radius, and its brightness
            as a factor of the local smooth-law shading (above 1 for a moon
            brighter than its parent, below 1 for a darker one).  None for a
            shadow-only entry.
        shadow: ``(dv_px, du_px, radius_px, darkness)`` -- the cast shadow
            disc; the textured shading inside it is multiplied by
            ``1 - darkness``.  None for a moon-only entry.
    """

    moon: tuple[float, float, float, float] | None = None
    shadow: tuple[float, float, float, float] | None = None


def _spot_tuple(spot: Mapping[str, Any]) -> tuple[float, float, float, float]:
    """Resolve one spot/storm mapping into its hashable tuple form."""
    return (
        float(scene_value(spot, 'lat_deg', 0.0)),
        float(scene_value(spot, 'lon_deg', 90.0)),
        float(scene_value(spot, 'radius_deg', 10.0)),
        float(scene_value(spot, 'albedo_factor', 0.7)),
    )


def albedo_spec_from_params(
    block: Mapping[str, Any] | None, *, seed: int
) -> AlbedoTextureSpec | None:
    """Build an :class:`AlbedoTextureSpec` from a body's ``albedo_texture`` block.

    A key that the block leaves out takes its default.

    Parameters:
        block: One body's ``albedo_texture`` mapping, or None.
        seed: The body's 'albedo' stream seed.

    Returns:
        The resolved spec, or None when ``block`` is not a mapping.
    """
    if not isinstance(block, Mapping):
        return None
    return AlbedoTextureSpec(
        rms=float(scene_value(block, 'rms', 0.0)),
        corr_px=float(scene_value(block, 'corr_px', _DEFAULT_ALBEDO_CORR_PX)),
        spots=tuple(_spot_tuple(spot) for spot in block.get('spots') or []),
        seed=int(seed),
    )


def disc_texture_spec_from_params(body_params: Mapping[str, Any]) -> DiscTextureSpec | None:
    """Build a :class:`DiscTextureSpec` from a body mapping, or None.

    A key that the block leaves out takes its default.

    Parameters:
        body_params: One scene body entry.

    Returns:
        The resolved spec, or None when the body has no ``disc_texture``.
    """
    block = body_params.get('disc_texture')
    if not isinstance(block, Mapping):
        return None
    return DiscTextureSpec(
        band_amplitude=float(scene_value(block, 'band_amplitude', 0.0)),
        band_wavenumber=float(scene_value(block, 'band_wavenumber', 8.0)),
        band_phase_deg=float(scene_value(block, 'band_phase_deg', 0.0)),
        storms=tuple(_spot_tuple(storm) for storm in block.get('storms') or []),
    )


def transit_specs_from_params(body_params: Mapping[str, Any]) -> tuple[TransitSpec, ...]:
    """Build the body's :class:`TransitSpec` tuple (empty without ``transits``).

    A key that an entry leaves out takes its default.

    Parameters:
        body_params: One scene body entry.

    Returns:
        One spec per ``transits`` entry, in scene order.
    """
    entries = body_params.get('transits')
    if not isinstance(entries, list):
        return ()
    specs: list[TransitSpec] = []
    for entry in entries:
        moon = entry.get('moon')
        shadow = entry.get('shadow')
        specs.append(
            TransitSpec(
                moon=(
                    (
                        float(scene_value(moon, 'dv_px', 0.0)),
                        float(scene_value(moon, 'du_px', 0.0)),
                        float(scene_value(moon, 'radius_px', 5.0)),
                        float(scene_value(moon, 'albedo_factor', 1.0)),
                    )
                    if isinstance(moon, Mapping)
                    else None
                ),
                shadow=(
                    (
                        float(scene_value(shadow, 'dv_px', 0.0)),
                        float(scene_value(shadow, 'du_px', 0.0)),
                        float(scene_value(shadow, 'radius_px', 5.0)),
                        float(scene_value(shadow, 'darkness', 0.8)),
                    )
                    if isinstance(shadow, Mapping)
                    else None
                ),
            )
        )
    return tuple(specs)


def _spot_factor(
    factor: NDArrayFloatType,
    spots: tuple[tuple[float, float, float, float], ...],
    *,
    axis_a: NDArrayFloatType,
    axis_b: NDArrayFloatType,
    axis_c: NDArrayFloatType,
) -> None:
    """Multiply circular spots into ``factor`` in place.

    A spot at ``(lat_deg, lon_deg)`` has its center unit vector
    ``(sin(lat), cos(lat) * cos(lon), cos(lat) * sin(lon))`` in the
    ``(axis_a, axis_b, axis_c)`` frame handed in; points within
    ``radius_deg`` of angular separation take the spot's albedo factor.

    Parameters:
        factor: The multiplicative field to update in place.
        spots: ``(lat_deg, lon_deg, radius_deg, albedo_factor)`` tuples.
        axis_a: Polar-axis component of each surface point's unit vector.
        axis_b: First equatorial component of each unit vector.
        axis_c: Second equatorial component of each unit vector.
    """
    for lat_deg, lon_deg, radius_deg, albedo_factor in spots:
        lat = math.radians(lat_deg)
        lon = math.radians(lon_deg)
        cos_sep = (
            math.sin(lat) * axis_a
            + math.cos(lat) * math.cos(lon) * axis_b
            + math.cos(lat) * math.sin(lon) * axis_c
        )
        inside = cos_sep >= math.cos(math.radians(radius_deg))
        factor[inside] *= albedo_factor


def surface_texture_factor(
    albedo: AlbedoTextureSpec | None,
    disc_texture: DiscTextureSpec | None,
    *,
    x: NDArrayFloatType,
    y: NDArrayFloatType,
    z: NDArrayFloatType,
    mean_radius_px: float,
) -> NDArrayFloatType | None:
    """The combined multiplicative texture factor over the shading grid.

    Parameters:
        albedo: The body's albedo-texture spec, or None.
        disc_texture: The body's disc-texture spec, or None.
        x: Unit-sphere component along the pole per shading pixel.
        y: Unit-sphere component along semi-axis b per shading pixel.
        z: Unit-sphere component along semi-axis c, the axis toward the observer
            for a body seen equator-on (outside the disc, the limb point the line
            of sight passes closest to, so pixels just past the limb carry the
            limb's values).
        mean_radius_px: Mean projected body radius in detector
            pixels, the ``corr_px`` -> surface-arc conversion scale.

    Returns:
        The per-pixel multiplicative factor (clipped non-negative), or None
        when neither texture is configured.
    """
    if albedo is None and disc_texture is None:
        return None
    factor = np.ones_like(x)
    # One body-polar frame for every texture: pole along the a axis (x),
    # longitudes from +b (y) toward the observer (z).
    lat_p = np.arcsin(np.clip(x, -1.0, 1.0))

    if albedo is not None:
        if albedo.rms > 0.0 and albedo.corr_px > 0.0 and mean_radius_px > 0.0:
            # corr_px of image distance where the surface is face-on =
            # corr_px of surface arc = corr_px / R radians of arc.
            corr_deg = math.degrees(albedo.corr_px / mean_radius_px)
            field = synthesize_albedo_field(albedo.rms, corr_deg, albedo.seed)
            lon_p = np.mod(np.arctan2(z, y), 2.0 * np.pi)
            factor *= 1.0 + field.sample(lat_p, lon_p)
        _spot_factor(factor, albedo.spots, axis_a=x, axis_b=y, axis_c=z)

    if disc_texture is not None:
        if disc_texture.band_amplitude > 0.0:
            phase = math.radians(disc_texture.band_phase_deg)
            factor *= 1.0 + disc_texture.band_amplitude * np.cos(
                disc_texture.band_wavenumber * lat_p + phase
            )
        _spot_factor(factor, disc_texture.storms, axis_a=x, axis_b=y, axis_c=z)

    return np.asarray(np.maximum(factor, 0.0), dtype=np.float64)


def apply_transits(
    intensity: NDArrayFloatType,
    transits: tuple[TransitSpec, ...],
    *,
    base_intensity: NDArrayFloatType,
    v_ctr: NDArrayFloatType,
    u_ctr: NDArrayFloatType,
    disc_mask: NDArrayBoolType,
) -> None:
    """Render the transit shadows and moon discs into ``intensity`` in place.

    Render order per the layering contract: the banded texture is already in
    ``intensity``; every shadow multiplies that texture first, then every
    moon disc overwrites on top (a moon in front occludes the texture and
    any shadow under it).  The moon disc's brightness derives from the
    *smooth-law* shading (``base_intensity``): a foreground moon does not
    inherit the parent's texture.

    Parameters:
        intensity: The textured shading-grid intensity, updated in place.
        transits: The body's transit specs.
        base_intensity: The pre-texture smooth-law shading.
        v_ctr: Body-centered v coordinate of each shading pixel.
        u_ctr: Body-centered u coordinate of each shading pixel.
        disc_mask: True inside the parent disc; transits render only there.
    """
    for transit in transits:
        if transit.shadow is not None:
            dv, du, radius, darkness = transit.shadow
            inside = disc_mask & ((v_ctr - dv) ** 2 + (u_ctr - du) ** 2 <= radius * radius)
            intensity[inside] *= 1.0 - min(max(darkness, 0.0), 1.0)
    for transit in transits:
        if transit.moon is not None:
            dv, du, radius, albedo_factor = transit.moon
            inside = disc_mask & ((v_ctr - dv) ** 2 + (u_ctr - du) ** 2 <= radius * radius)
            intensity[inside] = base_intensity[inside] * albedo_factor
