"""The rules for a scene's bodies that reach past one body entry.

A body entry's keys, their ranges, and the rules among its own keys are checked by its
type (:class:`~spindoctor.sim.scene.truth_body.Body` and the blocks it holds).  Two
rules need more than one entry: :func:`check_body_names_unique` compares the bodies'
name strings with one another, and :func:`check_model_atmosphere_filters` reads the
scene's camera and filter for a body that takes Titan's haze by name.  The Titan haze
is modeled over a span of wavelengths and leaves out methane absorption, so this
module also holds what the model covers: :data:`TITAN_HAZE_SPAN_NM`,
:data:`METHANE_BANDS_NM`, and the share of a passband's light each may leave uncovered.

:func:`spindoctor.sim.scene.validate.parse_scene` applies both rules to the typed scene.
"""

from __future__ import annotations

import numpy as np

from spindoctor.sim.instruments.passbands import (
    Passband,
    has_passbands,
    solar_weighted_passband,
)
from spindoctor.sim.scene.errors import SimSceneValidationError
from spindoctor.sim.scene.truth import Scene
from spindoctor.sim.scene.truth_body import Body

__all__ = [
    'MAX_METHANE_BAND_FRACTION',
    'MAX_UNCOVERED_FRACTION',
    'METHANE_BANDS_NM',
    'TITAN_HAZE_SPAN_NM',
    'check_body_names_unique',
    'check_model_atmosphere_filters',
    'methane_band_fraction',
    'uncovered_fraction',
]

TITAN_HAZE_SPAN_NM: tuple[float, float] = (430.0, 1078.0)
"""The wavelengths, in nm, over which Titan's haze is tabulated: the DISR span.

Outside it the haze table holds its end values.
"""

METHANE_BANDS_NM: tuple[tuple[float, float], ...] = ((619.0, 4.0), (727.0, 4.0), (889.0, 10.0))
"""The cores of Titan's strong methane bands, as ``(center, half width)`` in nm."""

MAX_METHANE_BAND_FRACTION = 0.1
"""The largest share of a passband's light in the methane band cores a Titan haze takes.

Broadband and continuum passbands put 0.07 or less of their light there, and those
that overlap a band 0.11 or more.
"""

MAX_UNCOVERED_FRACTION = 0.25
"""The largest share of a passband's light outside the haze table's span a scene takes."""

# The filter elements that pass a methane band or a methane window, by name.
_METHANE_FILTER_PREFIXES = ('MT', 'CB')


def check_body_names_unique(bodies: tuple[Body, ...], *, source: str) -> None:
    """Refuse two bodies that share a name, on either side of the boundary.

    The renderer keys its truth records (the body masks, the inventory, the render
    order) by each body's upper-cased name, and an unnamed body takes the name its
    position gives it (:meth:`~spindoctor.sim.scene.idealized.IdealizedBody.name_at`).
    Two entries with one name would overwrite each other's records, so the collision
    is refused.  The navigator keys its predicted bodies by the name its view carries,
    where a ``nav_override`` ``name`` replaces the body's own, so the same rule holds
    for those names.

    Parameters:
        bodies: The scene's bodies.
        source: The label that begins the error message.

    Raises:
        SimSceneValidationError: If two bodies share a name, compared without case.
    """
    for navigator_side in (False, True):
        seen: dict[str, int] = {}
        for index, body in enumerate(bodies):
            name = body.name_at(index)
            override = body.nav_override
            if navigator_side and override is not None and override.name is not None:
                name = override.name
            effective = name.upper()
            if effective in seen:
                side = 'navigator-side ' if navigator_side else ''
                raise SimSceneValidationError(
                    f'{source}: bodies[{seen[effective]}] and bodies[{index}] share the '
                    f'{side}effective name {effective!r} (names are case-insensitive and an '
                    f'unnamed body defaults to its positional SIM-BODY-<n> name); '
                    f'name-keyed truth records would collide, so give each body a '
                    f'unique name'
                )
            seen[effective] = index


def uncovered_fraction(passband: Passband) -> float:
    """The share of a passband's solar-weighted light outside :data:`TITAN_HAZE_SPAN_NM`.

    Parameters:
        passband: The passband.

    Returns:
        The weight below the span's first wavelength or above its last.
    """
    wavelength = passband.wavelength_nm
    low, high = TITAN_HAZE_SPAN_NM
    outside = (wavelength < low) | (wavelength > high)
    return float(np.sum(passband.weight[outside]))


def methane_band_fraction(passband: Passband) -> float:
    """The share of a passband's solar-weighted light in the methane band cores.

    Parameters:
        passband: The passband.

    Returns:
        The weight within :data:`METHANE_BANDS_NM`.
    """
    wavelength = passband.wavelength_nm
    inside = np.zeros(wavelength.shape, dtype=np.bool_)
    for center, half_width in METHANE_BANDS_NM:
        inside |= np.abs(wavelength - center) <= half_width
    return float(np.sum(passband.weight[inside]))


def check_model_atmosphere_filters(scene: Scene, *, source: str) -> None:
    """Refuse a Titan haze in a filter the haze is not modeled in.

    Titan's haze depends on wavelength, so it needs the camera's passbands, and its
    model leaves out methane absorption and covers :data:`TITAN_HAZE_SPAN_NM`.

    Parameters:
        scene: The typed scene.
        source: The label that begins the error message.

    Raises:
        SimSceneValidationError: If a body takes Titan's haze and the camera has no
            passband table, or the scene's filter is a methane-band or methane-window
            filter, or too much of its light falls outside the modeled span.
    """
    hazy = [
        body
        for body in scene.bodies or ()
        if body.atmosphere is not None and body.atmosphere.haze_model is not None
    ]
    if not hazy:
        return
    instrument = scene.instrument
    if not has_passbands(instrument):
        raise SimSceneValidationError(
            f"{source}: Titan's haze depends on wavelength, and sim instrument "
            f'{instrument!r} has no passband table'
        )
    detector_filter = None if scene.detector is None else scene.detector.filter
    passband = solar_weighted_passband(instrument, detector_filter)
    # Methane-band and methane-window filters are out of scope (#1189).
    in_methane = any(
        element.startswith(_METHANE_FILTER_PREFIXES) for element in passband.filter_name.split('+')
    )
    if in_methane:
        raise SimSceneValidationError(
            f"{source}: Titan's haze is not modeled in methane-band or methane-window "
            f'filters, whose light methane absorbs or lets through to the surface; '
            f'the scene uses {passband.filter_name!r}'
        )
    in_bands = methane_band_fraction(passband)
    if in_bands > MAX_METHANE_BAND_FRACTION:
        raise SimSceneValidationError(
            f"{source}: Titan's haze is not modeled where methane absorbs, and "
            f'{in_bands:.0%} of the light of {passband.filter_name!r} falls in the cores of '
            f'the methane bands'
        )
    outside = uncovered_fraction(passband)
    if outside > MAX_UNCOVERED_FRACTION:
        low, high = TITAN_HAZE_SPAN_NM
        raise SimSceneValidationError(
            f"{source}: Titan's haze is modeled from {low:g} to {high:g} nm, and "
            f'{outside:.0%} of the light of {passband.filter_name!r} lies outside that span'
        )
