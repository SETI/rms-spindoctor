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

A scene states a star's position as the pixel index it is drawn on, which is
what the renderer deposits into and what the editor's click targets use.  A
star record states it in oops uv, because that is what the real catalog
reduction produces.  This builder is the boundary between the two, so it is
where the half-pixel datum is added; nothing downstream may add it again.
"""

from typing import Any, cast

from starcat import Star

from spindoctor.support.types import STAR_UV_DATUM_PX, MutableStar

__all__ = ['DEFAULT_PSF_SIZE', 'star_record_from_params']

# The PSF fitting-window size a star entry gets when the scene omits one,
# in detector pixels.  Renderer-side scaling to an oversampled grid must
# materialize this default so the downsample's rescale returns it intact.
DEFAULT_PSF_SIZE: tuple[int, int] = (11, 11)


def star_record_from_params(
    star_params: dict[str, Any],
    *,
    index: int,
    default_v: float,
    default_u: float,
) -> MutableStar:
    """Build one catalog star record from a scene star mapping.

    The scene's ``v`` / ``u`` (and the defaults standing in for them) are
    pixel indices on the grid the caller's scene values describe; the
    record's ``v`` / ``u`` are the oops uv
    :class:`~spindoctor.support.types.MutableStar` declares, so they come out
    ``STAR_UV_DATUM_PX`` higher.  ``move_v`` / ``move_u`` are displacements
    and carry no datum.

    Parameters:
        star_params: One scene ``stars`` entry (idealized keys only are read).
        index: Zero-based position in the scene's star list; drives the
            record's unique number and default name.
        default_v: V pixel index used when the entry has no ``v`` (frame
            centre).
        default_u: U pixel index used when the entry has no ``u`` (frame
            centre).

    Returns:
        A fully populated star record at the unshifted catalog position, in
        oops uv.
    """
    star = cast(MutableStar, Star())
    star.unique_number = index + 1
    star.catalog_name = str(star_params.get('catalog_name', 'SIM'))
    star.pretty_name = str(star_params.get('name', f'SIM-{index + 1}'))
    star.name = star.pretty_name
    star.v = float(star_params.get('v', default_v)) + STAR_UV_DATUM_PX
    star.u = float(star_params.get('u', default_u)) + STAR_UV_DATUM_PX
    star.move_v = float(star_params.get('move_v', 0.0))
    star.move_u = float(star_params.get('move_u', 0.0))
    star.vmag = float(star_params.get('vmag', 8.0))
    star.spectral_class = str(star_params.get('spectral_class', 'G2'))
    star.temperature = Star.temperature_from_sclass(star.spectral_class)
    star.temperature_faked = star.temperature is None
    if star.temperature is None:
        star.temperature = 5780.0
    star.johnson_mag_v = star.vmag
    bmv = Star.bmv_from_sclass(star.spectral_class or 'G2') or 0.63
    star.johnson_mag_b = star.johnson_mag_v + bmv
    star.johnson_mag_faked = False
    # Simulated photometry is authoritative by construction, so it is never
    # saturated nor corrected (see nav_model/stars/saturation.py).
    star.photometry_corrected = False
    star.photometry_saturated = False
    star.ra_pm = 0.0
    star.dec_pm = 0.0
    star.conflicts = ''
    star.psf_size = tuple(star_params.get('psf_size', DEFAULT_PSF_SIZE))
    # Catalog-relative flux in the navigator's DN convention (see
    # nav_model/stars/catalog.py); derived from vmag, never from the render.
    star.dn = 2.512 ** -(star.vmag - 4.0)
    return star
