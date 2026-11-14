"""Shared construction of catalog star records from scene star parameters.

Both simulator sides build :class:`~spindoctor.support.types.MutableStar`
records from a scene's per-star mappings: the image-side renderer
(``spindoctor.sim.forward.star``) to know what to draw and to report in its
output metadata, and the navigator-side star model
(``spindoctor.nav_model.stars.nav_model_stars_simulated``) to build its
catalog from the filtered idealized view (``obs.nav_params``).  Sharing one
builder keeps the two sides' defaults (position, magnitude, spectral class,
PSF window) identical, so a scene that omits a field cannot silently give
the renderer and the navigator different catalogs.

Only idealized star keys are read here.  The record's ``dn`` field is the
catalog-derived relative flux ``2.512 ** -(vmag - 4)`` -- a pure function of
the catalog magnitude, matching the real catalog reduction
(``spindoctor.nav_model.stars.catalog``) -- not a rendered pixel value.

A scene states every position as a pixel corner -- integer ``N`` is the
boundary between pixel ``N - 1`` and pixel ``N``, so the center of pixel
``N`` is ``N + 0.5`` -- which is the pixel corner coordinate a star record declares.  The
scene value therefore IS the record value and this builder copies it
through.  The half pixel that separates it from a pixel-centric position comes off
only where something indexes an array: the renderer's deposit and the star
model's extended-FOV position.
"""

from typing import Any, cast

from starcat import Star

from spindoctor.sim.scene_schema import scene_value
from spindoctor.support.types import MutableStar

__all__ = [
    'DEFAULT_BMV',
    'DEFAULT_PSF_SIZE',
    'DEFAULT_SPECTRAL_CLASS',
    'star_record_from_params',
]

# The PSF fitting-window size a star entry gets when the scene omits one,
# in detector pixels.  Renderer-side scaling to an oversampled grid must
# materialize this default so the downsample's rescale returns it intact.
DEFAULT_PSF_SIZE: tuple[int, int] = (11, 11)

# The spectral class of a star entry that omits one.  G2 is the Sun's class.
DEFAULT_SPECTRAL_CLASS = 'G2'

# The Sun's Johnson B-V color index.  A star gets this color when the star
# catalog cannot map its spectral class to a color.
DEFAULT_BMV = 0.63


def star_record_from_params(
    star_params: dict[str, Any],
    *,
    index: int,
    default_v: float,
    default_u: float,
) -> MutableStar:
    """Build one catalog star record from a scene star mapping.

    The scene's ``v`` / ``u`` (and the defaults standing in for them) are
    already the pixel corner coordinate :class:`~spindoctor.support.types.MutableStar`
    declares, on whatever grid the caller's scene values describe, so they
    are copied through unchanged.  ``move_v`` / ``move_u`` are displacements
    and so need no conversion either.

    Parameters:
        star_params: One scene ``stars`` entry (idealized keys only are read).
        index: Zero-based position in the scene's star list; drives the
            record's unique number and default name.
        default_v: V position in pixel corner coordinates used when the entry has no ``v``
            (frame center).
        default_u: U position in pixel corner coordinates used when the entry has no ``u``
            (frame center).

    Returns:
        A fully populated star record at the unshifted catalog position, in
        pixel corner coordinates.
    """
    star = cast(MutableStar, Star())
    star.unique_number = index + 1
    star.catalog_name = str(scene_value(star_params, 'catalog_name', 'SIM'))
    star.pretty_name = str(scene_value(star_params, 'name', f'SIM-{index + 1}'))
    star.name = star.pretty_name
    star.v = float(scene_value(star_params, 'v', default_v))
    star.u = float(scene_value(star_params, 'u', default_u))
    star.move_v = float(scene_value(star_params, 'move_v', 0.0))
    star.move_u = float(scene_value(star_params, 'move_u', 0.0))
    star.vmag = float(scene_value(star_params, 'vmag', 8.0))
    star.spectral_class = str(scene_value(star_params, 'spectral_class', DEFAULT_SPECTRAL_CLASS))
    star.temperature = Star.temperature_from_sclass(star.spectral_class)
    star.temperature_faked = star.temperature is None
    if star.temperature is None:
        star.temperature = 5780.0
    star.johnson_mag_v = star.vmag
    # The lookup returns None for a class it cannot map, and only then does the
    # star take the default color.  A0 maps to exactly 0.0, so a truthiness test
    # would give every A0 star the solar color.
    bmv = Star.bmv_from_sclass(star.spectral_class or DEFAULT_SPECTRAL_CLASS)
    if bmv is None:
        bmv = DEFAULT_BMV
    star.johnson_mag_b = star.johnson_mag_v + bmv
    star.johnson_mag_faked = False
    # Simulated photometry is authoritative by construction, so it is never
    # saturated nor corrected (see nav_model/stars/saturation.py).
    star.photometry_corrected = False
    star.photometry_saturated = False
    star.ra_pm = 0.0
    star.dec_pm = 0.0
    star.conflicts = ''
    star.psf_size = tuple(scene_value(star_params, 'psf_size', DEFAULT_PSF_SIZE))
    # Catalog-relative flux in the navigator's DN convention (see
    # nav_model/stars/catalog.py); derived from vmag, never from the render.
    star.dn = 2.512 ** -(star.vmag - 4.0)
    return star
