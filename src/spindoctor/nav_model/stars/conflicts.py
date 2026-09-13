"""Body and ring conflict marking for star records.

A star whose predicted pixel falls inside the silhouette of a body, or
inside a known opaque ring annulus, cannot be detected — the body or
ring overrides its signal.  ``mark_body_and_ring_conflicts`` walks every
star in the reduced list and, for each one, builds a tiny ``oops``
backplane around the predicted position and queries the body intercept
plus the ring radius.  When either query fires, the star's
``conflicts`` field is set to a human-readable string starting with
``'BODY: '`` or ``'RING: '``.

Body-vs-ring precedence: a body intercept always wins, so a star whose
predicted pixel lies on a moon in front of Saturn's rings is tagged
with the moon, not the rings.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from oops import Meshgrid
from oops.backplane import Backplane

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    from spindoctor.config import Config
    from spindoctor.obs import ObsSnapshot
    from spindoctor.support.types import MutableStar

__all__ = [
    'mark_body_and_ring_conflicts',
    'parse_ring_occlusion_annuli',
]


def parse_ring_occlusion_annuli(
    raw: dict[str, list[list[float]]] | None,
) -> dict[str, list[tuple[float, float]]]:
    """Validate and normalise a ring-occlusion annulus mapping.

    The YAML config exposes per-planet annulus pairs as nested lists
    (``[[inner_km, outer_km], ...]``); this helper validates each pair,
    rejects degenerate (inner >= outer) annuli, and normalises the
    planet keys to upper case so lookup is case-insensitive.

    Parameters:
        raw: Mapping returned by ``config.stars.ring_occlusion_radii_km``.
            ``None`` is treated as the empty mapping.

    Returns:
        ``{PLANET_UPPER: [(inner_km, outer_km), ...]}`` with float
        entries.

    Raises:
        ValueError: If an annulus is malformed or has ``inner >= outer``.
    """
    if not raw:
        return {}
    out: dict[str, list[tuple[float, float]]] = {}
    for planet, pairs in raw.items():
        validated: list[tuple[float, float]] = []
        for pair in pairs:
            inner, outer = _parse_annulus_pair(pair, planet)
            if inner >= outer:
                raise ValueError(
                    f'ring_occlusion_radii_km: invalid annulus for {planet}: '
                    f'inner {inner} km >= outer {outer} km'
                )
            validated.append((inner, outer))
        out[planet.upper()] = validated
    return out


def _parse_annulus_pair(pair: object, planet_key: str) -> tuple[float, float]:
    """Parse one annulus into ``(inner_km, outer_km)`` floats.

    Parameters:
        pair: Sequence of two finite numbers.
        planet_key: Planet name used in error messages.

    Returns:
        ``(inner_km, outer_km)`` as finite floats.

    Raises:
        ValueError: If ``pair`` is not a length-2 sequence of finite
            real numbers.
    """
    if isinstance(pair, (str, bytes)) or not isinstance(pair, Sequence):
        raise ValueError(
            f'ring_occlusion_radii_km: annulus for {planet_key!r} must be a '
            f'length-2 sequence of numbers, got {type(pair).__name__}: {pair!r}'
        )
    if len(pair) != 2:
        raise ValueError(
            f'ring_occlusion_radii_km: annulus for {planet_key!r} must have '
            f'exactly 2 elements, got {len(pair)}: {pair!r}'
        )
    out: list[float] = []
    for label, raw in (('inner', pair[0]), ('outer', pair[1])):
        if isinstance(raw, bool):
            raise ValueError(
                f'ring_occlusion_radii_km: {label} radius for {planet_key!r} '
                f'must be numeric, got bool: {raw!r}'
            )
        if not isinstance(raw, (int, float, np.integer, np.floating)):
            raise ValueError(
                f'ring_occlusion_radii_km: {label} radius for {planet_key!r} '
                f'must be numeric, got {type(raw).__name__}: {raw!r}'
            )
        val = float(raw)
        if not math.isfinite(val):
            raise ValueError(
                f'ring_occlusion_radii_km: {label} radius for {planet_key!r} '
                f'must be finite, got {raw!r}'
            )
        out.append(val)
    return out[0], out[1]


def _conflict_body_list(obs: ObsSnapshot, config: Config) -> list[str]:
    """Return the bodies the star pipeline checks for occlusion conflicts."""
    closest = obs.closest_planet
    body_list: list[str] = [closest] if closest is not None else []
    body_list += list(config.satellites(closest or ''))
    return body_list


def _ring_opaque_fraction(
    radii_km: np.ndarray,
    valid_mask: np.ndarray,
    annuli: list[tuple[float, float]],
) -> float:
    """Fraction of the valid window pixels that lie inside an opaque annulus.

    Parameters:
        radii_km: Ring-radius backplane values over the conflict window, in km.
        valid_mask: True where the backplane value is valid (ring plane
            intercepted); same shape as ``radii_km``.
        annuli: List of opaque ``(inner_km, outer_km)`` annuli.

    Returns:
        Opaque-pixel count divided by valid-pixel count, or 0.0 when no
        window pixel is valid.
    """
    n_valid = int(np.count_nonzero(valid_mask))
    if n_valid == 0:
        return 0.0
    opaque = np.zeros(radii_km.shape, dtype=bool)
    for inner_km, outer_km in annuli:
        opaque |= (radii_km >= inner_km) & (radii_km <= outer_km)
    return float(np.count_nonzero(opaque & valid_mask)) / n_valid


def _check_one_star(
    *,
    obs: ObsSnapshot,
    star: MutableStar,
    body_list: list[str],
    ring_annuli: dict[str, list[tuple[float, float]]],
    rings_can_conflict: bool,
    body_conflict_margin: float,
    ring_min_opaque_fraction: float,
) -> bool:
    """Return True if ``star`` conflicts with a body or ring; sets the flag.

    Ring membership is tested per pixel over the conflict window: the star is
    ring-occluded when at least ``ring_min_opaque_fraction`` of the valid
    window pixels fall inside an opaque annulus.  A single collapsed statistic
    (the window's median radius) would mis-classify stars whose window
    straddles a ringlet edge or a gap boundary -- exactly where occlusion
    matters most.

    Parameters:
        obs: Observation snapshot.
        star: Star record to inspect (mutated when a conflict is found).
        body_list: List of body names checked for intercepts.
        ring_annuli: Per-planet list of opaque ring annuli in km.
        rings_can_conflict: Toggle the ring check.
        body_conflict_margin: Pixel slop around the star when building
            the conflict-check meshgrid.
        ring_min_opaque_fraction: Minimum opaque fraction of valid window
            pixels for a ring conflict.

    Returns:
        True if a conflict was found (and ``star.conflicts`` set);
        False otherwise.
    """
    # ``Meshgrid.for_fov`` states its origin and limit in FOV uv, which is
    # what a star record carries, so the window is built from the record's
    # position with no datum shift.
    meshgrid = Meshgrid.for_fov(
        obs.fov,
        origin=(star.u - body_conflict_margin, star.v - body_conflict_margin),
        limit=(star.u + body_conflict_margin, star.v + body_conflict_margin),
    )
    backplane = Backplane(obs, meshgrid)

    for body_name in body_list:
        intercepted = backplane.where_intercepted(body_name)
        if intercepted.any():
            star.conflicts = f'BODY: {body_name}'
            return True

    if rings_can_conflict and obs.closest_planet is not None:
        annuli = ring_annuli.get(obs.closest_planet.upper(), [])
        if annuli:
            ring_target = f'{obs.closest_planet.lower()}:ring'
            bp_radii = backplane.ring_radius(ring_target)
            if not bp_radii.is_all_masked():
                radii_km = np.asarray(bp_radii.vals, dtype=np.float64)
                valid_mask = ~np.broadcast_to(np.asarray(bp_radii.mask, dtype=bool), radii_km.shape)
                fraction = _ring_opaque_fraction(radii_km, valid_mask, annuli)
                if fraction >= ring_min_opaque_fraction:
                    star.conflicts = f'RING: {obs.closest_planet}'
                    return True
    return False


def mark_body_and_ring_conflicts(
    obs: ObsSnapshot,
    config: Config,
    stars: list[MutableStar],
) -> None:
    """Tag each star whose predicted pixel is occluded by a body or ring.

    The check has two parts:

    1. **Body intercepts.**  A small meshgrid around the predicted star
       pixel is fed through ``Backplane.where_intercepted(body)`` for
       every body in the planet+satellites list pulled from
       ``config.satellites``.  Any intercept marks the star with
       ``conflicts = 'BODY: <body>'`` and short-circuits the ring check.

    2. **Ring annulus occlusion.**  When ``stars.ring_occlusion_enabled``
       is True and the closest-planet has annuli configured, the same
       meshgrid is queried for ``ring_radius`` and each valid pixel's
       radius is tested against the annuli.  When at least
       ``stars.ring_occlusion_min_opaque_fraction`` of the valid window
       pixels are opaque, the star is marked with
       ``conflicts = 'RING: <planet>'``.

    Stars already marked with a non-empty ``conflicts`` (e.g. ``'STAR'``
    from visual overlap) are left alone.

    Parameters:
        obs: Observation snapshot.
        config: Project ``Config``.
        stars: Star list to mutate in place.
    """
    stars_config = config.stars
    body_list = _conflict_body_list(obs, config)
    ring_annuli = parse_ring_occlusion_annuli(_cast_dict(stars_config.ring_occlusion_radii_km))
    rings_can_conflict = bool(stars_config.ring_occlusion_enabled)
    margin = float(stars_config.body_conflict_margin)
    min_opaque_fraction = float(stars_config.ring_occlusion_min_opaque_fraction)
    for star in stars:
        if star.conflicts:
            continue
        _check_one_star(
            obs=obs,
            star=star,
            body_list=body_list,
            ring_annuli=ring_annuli,
            rings_can_conflict=rings_can_conflict,
            body_conflict_margin=margin,
            ring_min_opaque_fraction=min_opaque_fraction,
        )


def _cast_dict(
    raw: Any,
) -> dict[str, list[list[float]]] | None:
    """Narrow a raw config value to the ring-occlusion mapping shape.

    The YAML loader returns ``AttrDict`` and ``list`` instances; we
    accept any mapping with string keys and list-of-list values, and
    treat ``None`` as the empty mapping.  Mismatches raise so the error
    points at the YAML site, not at the ring-radius math.

    Parameters:
        raw: Value pulled from ``config.stars.ring_occlusion_radii_km``.

    Returns:
        ``None`` (no annuli) or the validated mapping.
    """
    if raw is None:
        return None
    if not hasattr(raw, 'items'):
        raise ValueError(
            f'ring_occlusion_radii_km must be a mapping or None; got {type(raw).__name__}'
        )
    out: dict[str, list[list[float]]] = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            raise ValueError(
                f'ring_occlusion_radii_km keys must be strings; got {type(key).__name__}'
            )
        if not isinstance(value, list):
            raise ValueError(
                f'ring_occlusion_radii_km[{key!r}] must be a list; got {type(value).__name__}'
            )
        out[key] = value
    return out
