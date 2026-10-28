"""An instrument's measured height of a planet's visible limb, and its validation.

A per-camera instrument block may carry ``visible_limb_altitude``: a mapping from
upper-case SPICE body name to the measured height of that body's visible limb above
its reference ellipsoid.  :class:`VisibleLimbModel` is one validated entry;
:func:`visible_limb_model_for` reads the entry for one body from an instrument block;
:func:`validate_visible_limb_config` checks every entry in a merged configuration so a
malformed one fails when the configuration is loaded rather than when an image is
navigated.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    'VISIBLE_LIMB_KEY',
    'VisibleLimbModel',
    'validate_visible_limb_config',
    'visible_limb_model_for',
]

VISIBLE_LIMB_KEY: str = 'visible_limb_altitude'
"""Key of the per-body visible-limb entries in a per-camera instrument block."""


def _pair(body_name: str, field: str, value: Any) -> tuple[float, float]:
    """Return a two-number ``value`` as floats.

    Parameters:
        body_name: The body, for the error message.
        field: The field, for the error message.
        value: The value to read.

    Returns:
        The two numbers.

    Raises:
        ValueError: If the value is not two finite numbers.
    """
    try:
        first, second = (float(x) for x in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{body_name}: {field} must be two numbers; got {value!r}') from exc
    if not (math.isfinite(first) and math.isfinite(second)):
        raise ValueError(f'{body_name}: {field} values must be finite; got {value!r}')
    return first, second


@dataclass(frozen=True)
class VisibleLimbModel:
    """An instrument's measured height of one body's visible limb above its ellipsoid.

    Parameters:
        altitude_km: ``(phase_deg, km)`` pairs in strictly increasing phase for
            broadband and continuum filters; the altitude is linearly interpolated
            between them and held at the end values.
        sigma_km: One-sigma uncertainty of the altitude, common to every vertex of an
            image's limb.
        band_offsets_km: ``(FILTER, km)`` pairs added to the altitude when the image
            was taken through that (upper-case) filter; the largest applies when both
            filters are listed.
        km_per_px_range: ``(min, max)`` limb scale the model was measured over.
        phase_deg_range: ``(min, max)`` phase angle the model was measured over.
    """

    altitude_km: tuple[tuple[float, float], ...]
    sigma_km: float
    band_offsets_km: tuple[tuple[str, float], ...]
    km_per_px_range: tuple[float, float]
    phase_deg_range: tuple[float, float]

    @classmethod
    def from_mapping(cls, body_name: str, mapping: Any) -> VisibleLimbModel:
        """Build and validate a model from one body's configuration entry.

        Parameters:
            body_name: The body, for error messages.
            mapping: The entry.  Required keys: ``altitude_km`` (a list of
                ``[phase_deg, km]``), ``sigma_km``, ``km_per_px_range`` and
                ``phase_deg_range`` (each ``[min, max]``).  Optional:
                ``band_offsets_km`` (filter name to km).

        Returns:
            The validated model.

        Raises:
            ValueError: If the entry is not a mapping, a required key is missing, the
                table is empty, a row is not a pair of finite numbers, the phases do
                not strictly increase, the sigma is negative or not finite, an offset
                is not finite, or a range is not two finite numbers with
                ``min < max`` (and ``0 < min`` for the scale).
        """
        if not isinstance(mapping, Mapping):
            raise ValueError(f'{body_name}: the entry must be a mapping; got {mapping!r}')
        for key in ('altitude_km', 'sigma_km', 'km_per_px_range', 'phase_deg_range'):
            if key not in mapping:
                raise ValueError(f'{body_name}: {key} is required')
        try:
            rows = list(mapping['altitude_km'])
        except TypeError as exc:
            raise ValueError(f'{body_name}: altitude_km must be a list of pairs') from exc
        if len(rows) == 0:
            raise ValueError(f'{body_name}: altitude_km must list at least one pair')
        table = [_pair(body_name, 'altitude_km', row) for row in rows]
        if any(b[0] <= a[0] for a, b in itertools.pairwise(table)):
            raise ValueError(f'{body_name}: altitude_km phases must strictly increase')
        try:
            sigma = float(mapping['sigma_km'])
        except (TypeError, ValueError) as exc:
            raise ValueError(f'{body_name}: sigma_km must be a number') from exc
        if not (math.isfinite(sigma) and sigma >= 0.0):
            raise ValueError(f'{body_name}: sigma_km must be finite and >= 0; got {sigma!r}')
        raw_offsets = mapping.get('band_offsets_km') or {}
        if not isinstance(raw_offsets, Mapping):
            raise ValueError(f'{body_name}: band_offsets_km must map filter names to km')
        offsets = []
        for band, km in raw_offsets.items():
            try:
                value = float(km)
            except (TypeError, ValueError) as exc:
                raise ValueError(f'{body_name}: band_offsets_km must be numbers') from exc
            if not math.isfinite(value):
                raise ValueError(f'{body_name}: band_offsets_km must be finite')
            offsets.append((str(band).upper(), value))
        scale = _pair(body_name, 'km_per_px_range', mapping['km_per_px_range'])
        if not 0.0 < scale[0] < scale[1]:
            raise ValueError(f'{body_name}: km_per_px_range must satisfy 0 < min < max')
        phase = _pair(body_name, 'phase_deg_range', mapping['phase_deg_range'])
        if not phase[0] < phase[1]:
            raise ValueError(f'{body_name}: phase_deg_range must satisfy min < max')
        return cls(
            altitude_km=tuple(table),
            sigma_km=sigma,
            band_offsets_km=tuple(sorted(offsets)),
            km_per_px_range=scale,
            phase_deg_range=phase,
        )

    def applies_at(self, km_per_px: float, phase_deg: float) -> bool:
        """Return whether a limb lies in the scale and phase ranges the model covers.

        Parameters:
            km_per_px: The limb's scale.
            phase_deg: The body's center phase angle, in degrees.

        Returns:
            True when both lie within their ranges, ends included.
        """
        return (
            self.km_per_px_range[0] <= km_per_px <= self.km_per_px_range[1]
            and self.phase_deg_range[0] <= phase_deg <= self.phase_deg_range[1]
        )

    def altitude_for_image_km(self, phase_deg: float, filters: list[str]) -> float:
        """Return the visible limb's altitude for one image.

        Parameters:
            phase_deg: The body's center phase angle, in degrees.
            filters: The image's filter names (case-insensitive).

        Returns:
            The table altitude at ``phase_deg`` plus the largest band offset of the
            image's filters, in kilometers.
        """
        base = float(
            np.interp(
                float(phase_deg),
                [p for p, _ in self.altitude_km],
                [km for _, km in self.altitude_km],
            )
        )
        names = {str(f).upper() for f in filters}
        offsets = [km for band, km in self.band_offsets_km if band in names]
        return base + (max(offsets) if offsets else 0.0)


def visible_limb_model_for(
    inst_config: Mapping[str, Any] | None, body_name: str
) -> VisibleLimbModel | None:
    """Return the instrument's visible-limb model for a body, if it has one.

    Parameters:
        inst_config: A per-camera instrument block, or None.
        body_name: SPICE name of the body (case-insensitive).

    Returns:
        The validated model from ``inst_config['visible_limb_altitude'][BODY]``, or
        None when the instrument gives none for the body.

    Raises:
        ValueError: If the entry is malformed (see :meth:`VisibleLimbModel.from_mapping`).
    """
    if inst_config is None or VISIBLE_LIMB_KEY not in inst_config:
        return None
    entries = inst_config[VISIBLE_LIMB_KEY] or {}
    upper = body_name.upper()
    if upper not in entries:
        return None
    return VisibleLimbModel.from_mapping(upper, entries[upper])


def validate_visible_limb_config(config_dict: Mapping[str, Any]) -> None:
    """Check every ``visible_limb_altitude`` block in a merged configuration.

    A block is looked for in each per-camera mapping of each top-level section (an
    instrument section maps camera names to blocks).

    Parameters:
        config_dict: The merged configuration.

    Raises:
        ValueError: If a block is not a mapping or one of its entries is malformed;
            the message names the section and camera.
    """
    for section, cameras in config_dict.items():
        if not isinstance(cameras, Mapping):
            continue
        for camera, block in cameras.items():
            if not isinstance(block, Mapping) or VISIBLE_LIMB_KEY not in block:
                continue
            entries = block[VISIBLE_LIMB_KEY] or {}
            if not isinstance(entries, Mapping):
                raise ValueError(f'{section}.{camera}.{VISIBLE_LIMB_KEY} must be a mapping')
            for body, entry in entries.items():
                try:
                    VisibleLimbModel.from_mapping(str(body).upper(), entry)
                except ValueError as exc:
                    raise ValueError(f'{section}.{camera}.{VISIBLE_LIMB_KEY}: {exc}') from exc
