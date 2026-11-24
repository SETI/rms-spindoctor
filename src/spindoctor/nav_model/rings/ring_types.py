"""Immutable data types for ring feature orbital parameters.

This module defines the core value objects that represent ring edge orbital
data: ``RingFeatureType``, ``RingBaseOrbitMode``, ``RingPerturbationMode``,
and ``RingEdgeData``. These are frozen dataclasses (immutable) because orbital
parameters from YAML config are physical constants that should never be modified
after loading. The ring catalogs are checked when the configuration loads
(:mod:`spindoctor.config.sections_rings`), so these types do not check their values
again.

Separating these types into their own module keeps them free of rendering
dependencies (oops, numpy), enabling lightweight import for testing.

Design notes:

- ``RingBaseOrbitMode`` and ``RingPerturbationMode`` are distinct types to
  resolve the ambiguity in the YAML config where ``mode: 1`` can appear with
  either base-orbit fields (``a``, ``ae``, ...) or perturbation fields
  (``amplitude``, ``phase``, ...). Having separate types makes the dispatch
  explicit in ``RingFeature.from_entry()``.

- Inclination modes (``mode_num > 90``) are stored in ``RingEdgeData.perturbations``
  because they are valid YAML data. However, ``RingEdgeData.radial_perturbations()``
  and ``parsed_modes_for_backplane()`` exclude them, because the ``oops`` backplane
  ``radial_mode()`` function only handles radial (in-plane) perturbations. Making
  the limitation visible here rather than silently skipping them in rendering code
  surfaces the issue for future implementors.
"""

import enum
import math
from dataclasses import dataclass
from typing import Any

# Seconds per day: degrees/day (or similar) to radians/second for backplane tuples.
SECONDS_PER_DAY = 86400.0


class RingFeatureType(enum.Enum):
    """Classification of a ring feature as a gap or ringlet.

    Determines the rendering polarity:
    - RINGLET: image is brightened between the two edges (fill between).
    - GAP: image is darkened between the two edges (clear between).
    Single-edge features of either type use fading instead of solid fill.
    """

    GAP = 'GAP'
    RINGLET = 'RINGLET'


@dataclass(frozen=True)
class RingBaseOrbitMode:
    """Mode-1 orbital parameters defining the base orbit of a ring edge.

    This represents the fundamental circular (or nearly circular) orbit of a
    ring edge before any higher-order perturbation modes are applied. It is
    always present in the data; higher modes are optional perturbations.

    Parameters:
        a: Semi-major axis in km.
        ae: Eccentricity amplitude in km. Zero means circular.
        long_peri: Longitude of pericenter in degrees.
        rate_peri: Precession rate of pericenter in degrees/day.
        rms: RMS residual of the orbit fit in km.
            Used as the uncertainty measure for navigation.
    """

    a: float
    ae: float
    long_peri: float
    rate_peri: float
    rms: float


@dataclass(frozen=True)
class RingPerturbationMode:
    """A single radial or inclination perturbation mode for a ring edge.

    Higher-order perturbations are superimposed on the base orbit defined by
    ``RingBaseOrbitMode``. They represent resonance-driven distortions.

    Modes with ``mode_num > 90`` are inclination (out-of-plane) perturbations.
    These are stored in the data model because they appear in real YAML config
    files (e.g. Cassini Division features). However, inclination modes are not
    supported for radial backplane rendering because ``oops.ext_bp.radial_mode``
    only handles in-plane distortions. Use ``is_inclination_mode`` or
    ``RingEdgeData.radial_perturbations()`` to filter them out.

    Parameters:
        mode_num: Perturbation mode number passed to ``oops.ext_bp.radial_mode``.
            Values > 90 indicate inclination modes. Mission ring tables also use
            non-positive indices (e.g. ``0`` or negative modes in Saturn YAML).
        amplitude: Perturbation amplitude in km.
        phase: Perturbation phase in degrees.
        pattern_speed: Pattern speed in degrees/day.
    """

    mode_num: int
    amplitude: float
    phase: float
    pattern_speed: float

    @property
    def is_inclination_mode(self) -> bool:
        """Return True if this is an inclination (out-of-plane) mode.

        Inclination modes have ``mode_num > 90``. They represent vertical
        perturbations that require out-of-plane backplane support, which is
        not yet implemented. Callers should exclude these from radial rendering.
        """
        return self.mode_num > 90


@dataclass(frozen=True)
class RingEdgeData:
    """All orbital mode data for one edge of a ring feature.

    Combines the base orbit (``RingBaseOrbitMode``) with zero or more
    higher-order perturbations (``RingPerturbationMode``). This is the
    complete description needed to compute the radius of one ring edge at
    any point in the image backplane.

    Parameters:
        base_orbit: The mode-1 base orbit parameters.
        perturbations: Tuple of higher-order perturbation modes. May be empty.
            Inclination modes (mode_num > 90) are accepted here but excluded
            from backplane computation -- see ``radial_perturbations()``.
    """

    base_orbit: RingBaseOrbitMode
    perturbations: tuple['RingPerturbationMode', ...]

    @property
    def base_radius(self) -> float:
        """Semi-major axis of the base orbit in km.

        This is the nominal radius used for spatial filtering and conflict
        detection. It is the mean radius of the edge, not the instantaneous
        (perturbed) radius at any given longitude.
        """
        return self.base_orbit.a

    @property
    def rms(self) -> float:
        """RMS residual of the orbit fit in km.

        Propagated to ``RingFeature.uncertainty`` (max of inner and outer edge
        RMS values).
        """
        return self.base_orbit.rms

    def radial_perturbations(self) -> tuple['RingPerturbationMode', ...]:
        """Return only radial (non-inclination) perturbation modes.

        Inclination modes (mode_num > 90) require out-of-plane backplane
        support that is not yet implemented in oops. This method filters them
        out so callers do not need to check individually.

        Returns:
            Tuple of perturbation modes with mode_num <= 90.
        """
        return tuple(p for p in self.perturbations if not p.is_inclination_mode)

    def parsed_modes_for_backplane(self) -> list[tuple[Any, ...]]:
        """Convert edge data to tuples for oops radial_mode computation.

        Returns a list of tuples in the format expected by
        ``oops.ext_bp.radial_mode()``:

        - Mode 1 (base orbit): ``(1, a, ae, long_peri_rad, rate_peri_rad_per_sec)``
        - Other modes: ``(mode_num, amplitude, phase_rad, speed_rad_per_sec)``

        Inclination modes (mode_num > 90) are excluded because
        ``oops.ext_bp.radial_mode`` only supports in-plane perturbations.
        The base orbit always comes first, followed by perturbations in their
        original order.

        Returns:
            List of mode tuples. Always contains at least the base orbit tuple.
        """
        result: list[tuple[Any, ...]] = []

        # Base orbit: convert degrees and degrees/day to radians and radians/second
        long_peri_rad = math.radians(self.base_orbit.long_peri)
        rate_peri_rad_per_sec = math.radians(self.base_orbit.rate_peri) / SECONDS_PER_DAY
        result.append(
            (1, self.base_orbit.a, self.base_orbit.ae, long_peri_rad, rate_peri_rad_per_sec)
        )

        # Radial perturbation modes
        for p in self.radial_perturbations():
            phase_rad = math.radians(p.phase)
            speed_rad_per_sec = math.radians(p.pattern_speed) / SECONDS_PER_DAY
            result.append((p.mode_num, p.amplitude, phase_rad, speed_rad_per_sec))

        return result
