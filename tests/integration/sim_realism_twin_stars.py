"""Catalog star lists for the realism match's star twins.

The matched simulated frame of a real star frame (its twin) holds the catalog stars in
the real frame's field of view.  Each star sits where it appears in the real frame: its
predicted position at the real frame's pointing, moved by the frame's operator-verified
offset.  Each keeps its catalog magnitude.  The renderer dims every star by the twin's
filter, as the real filter does, so the filter decides how many of the stars can be
seen, in the twin as in the real frame.

The list comes from the navigator's own catalog reduction on the real observation, so
the twin is reproducible from the real frame: the catalogs and the SPICE kernels are
deterministic.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol, cast

from spindoctor.config import DEFAULT_CONFIG
from spindoctor.nav_model.stars.catalog import reduce_catalogs
from spindoctor.nav_model.stars.conflicts import mark_body_and_ring_conflicts
from spindoctor.obs import ObsSnapshotInst
from spindoctor.sim.forward.scene_radiance import _nominal_read_noise
from spindoctor.sim.forward.star import faint_sky_cutoff_mag
from spindoctor.sim.instrument_navigator import resolve_sim_inst_config
from spindoctor.sim.instrument_resolvers import (
    binning_for_frame_size,
    resolve_star_flux_zero_point,
)

__all__ = [
    'CatalogStar',
    'catalog_twin_stars',
    'twin_faint_cutoff_vmag',
    'twin_star_entries',
]

# The conflict prefixes the navigator's star model gives a star whose predicted position
# lies on a body or on an opaque ring annulus.  The real frame can't show such a star,
# and a star twin has no body or ring to hide it, so the twin leaves it out.
_HIDDEN_CONFLICT_PREFIXES = ('BODY', 'RING')


class CatalogStar(Protocol):
    """The fields of a navigator star record that a twin's star list reads.

    ``v`` and ``u`` are the predicted position in pixel corner coordinates in the
    nominal frame.  ``vmag`` is the catalog magnitude, and ``conflicts`` starts with
    ``'BODY'`` or ``'RING'`` when the star lies behind a body or an opaque ring.
    """

    unique_number: int | None
    catalog_name: str
    v: float
    u: float
    vmag: float | None
    conflicts: str


def twin_faint_cutoff_vmag(instrument: str, exposure_sec: float, size_vu: tuple[int, int]) -> float:
    """The faintest catalog magnitude a star twin holds.

    In the simulated camera, a star at the cutoff deposits a total flux equal to the
    camera's nominal read noise over its PSF core, through the clear filter.  That is a
    signal-to-noise ratio of 1 even with every electron in the core and a dark sky,
    while a star technique needs several times that to find a star.  So the cutoff
    includes every star the simulated camera could show through the clear filter.  A
    filter only dims stars, so no filter shows a star the clear filter can't, and the
    cutoff doesn't depend on the frame's filter.  The renderer cuts
    off its background sky stars by the same rule, through the scene's filter.

    Parameters:
        instrument: The sim instrument the frame is matched against.
        exposure_sec: The real frame's exposure in seconds.
        size_vu: The real frame's ``(v, u)`` size, which sets its readout binning.

    Returns:
        The cutoff magnitude.
    """
    zero_point, domain = resolve_star_flux_zero_point(instrument)
    binning = binning_for_frame_size(instrument, size_vu)
    psf_sigma = float(resolve_sim_inst_config(DEFAULT_CONFIG, instrument)['star_psf_sigma'])
    return faint_sky_cutoff_mag(
        zero_point=zero_point,
        exposure_sec=exposure_sec,
        read_noise=_nominal_read_noise(instrument, domain, binning=binning),
        psf_sigma=psf_sigma / binning,
    )


def twin_star_entries(
    stars: Iterable[CatalogStar],
    *,
    offset_vu: tuple[float, float],
    size_vu: tuple[int, int],
    faint_cutoff_vmag: float,
) -> list[dict[str, Any]]:
    """Turn the navigator's catalog stars into a star twin's scene star entries.

    Each star moves from its predicted position by the real frame's verified offset:
    a star predicted at ``(v, u)`` appears at ``(v + dv, u + du)``.  A star is left out
    when it is fainter than the cutoff, when it lies behind a body or an opaque ring,
    or when its moved position is outside the frame.  Positions and magnitudes are
    rounded to 0.01, and the entries are sorted by magnitude and then by position, so
    the same stars in any order give the same list.

    Parameters:
        stars: The catalog stars at the real frame's pointing.
        offset_vu: The real frame's operator-verified ``(dv, du)`` offset.
        size_vu: The frame's ``(v, u)`` size.  A star is in the frame when its moved
            pixel corner position lies in ``[0, size_v)`` by ``[0, size_u)``.
        faint_cutoff_vmag: The faintest magnitude kept.

    Returns:
        One scene star entry per kept star, with the keys ``name``, ``v``, ``u``, and
        ``vmag``.
    """
    dv, du = float(offset_vu[0]), float(offset_vu[1])
    size_v, size_u = float(size_vu[0]), float(size_vu[1])
    entries: list[dict[str, Any]] = []
    for star in stars:
        # The catalog reduction drops every record that has no magnitude.
        vmag = float(cast(float, star.vmag))
        if vmag > faint_cutoff_vmag:
            continue
        if star.conflicts.startswith(_HIDDEN_CONFLICT_PREFIXES):
            continue
        v = float(star.v) + dv
        u = float(star.u) + du
        if not (0.0 <= v < size_v and 0.0 <= u < size_u):
            continue
        entries.append(
            {
                'name': f'{star.catalog_name.upper()}-{star.unique_number}',
                'v': round(v, 2),
                'u': round(u, 2),
                'vmag': round(vmag, 2),
            }
        )
    entries.sort(key=lambda entry: (entry['vmag'], entry['v'], entry['u'], entry['name']))
    return entries


class _DeepCatalogObs:
    """A real observation whose faintest usable star magnitude is set by the caller.

    The navigator's catalog reduction stops at the observation's own faint limit.  That
    limit is the faintest star worth navigating on, and the clear filter can show stars
    fainter than it, so the twin's list asks for its own cutoff.  Every other attribute
    is the real observation's.
    """

    def __init__(self, obs: ObsSnapshotInst, faint_vmag: float) -> None:
        self._obs = obs
        self._faint_vmag = faint_vmag

    def star_max_usable_vmag(self) -> float:
        """The caller's faint cutoff."""
        return self._faint_vmag

    def __getattr__(self, name: str) -> Any:
        """Every other attribute of the real observation."""
        return getattr(self._obs, name)


def catalog_twin_stars(
    obs: ObsSnapshotInst,
    *,
    instrument: str,
    exposure_sec: float,
    offset_vu: tuple[float, float],
) -> list[dict[str, Any]]:
    """The scene star entries for one real star frame's twin.

    The navigator's catalog reduction runs on the real observation down to
    :func:`twin_faint_cutoff_vmag`, and its body and ring check marks the stars hidden
    behind a body or a ring.  :func:`twin_star_entries` then moves each star by the
    verified offset and keeps the ones the twin holds.  The reduction keeps at most the
    configured number of brightest stars (``stars.max_stars``).

    Parameters:
        obs: The real observation.
        instrument: The sim instrument the frame is matched against.
        exposure_sec: The real frame's exposure in seconds.
        offset_vu: The real frame's operator-verified ``(dv, du)`` offset.

    Returns:
        The twin's scene star entries; empty when no catalog star qualifies.
    """
    size_vu = (int(obs.data_shape_vu[0]), int(obs.data_shape_vu[1]))
    cutoff = twin_faint_cutoff_vmag(instrument, exposure_sec, size_vu)
    deep_obs = cast(ObsSnapshotInst, _DeepCatalogObs(obs, cutoff))
    stars = reduce_catalogs(deep_obs, DEFAULT_CONFIG)
    mark_body_and_ring_conflicts(obs, DEFAULT_CONFIG, stars)
    return twin_star_entries(stars, offset_vu=offset_vu, size_vu=size_vu, faint_cutoff_vmag=cutoff)
