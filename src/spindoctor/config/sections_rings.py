"""The typed ``rings`` section of the navigator configuration, with the ring catalogs.

The ``rings`` section (:class:`RingsSection`) holds the ring model's settings and, under
``ring_features``, one ring catalog per planet (:class:`RingCatalog`).  A catalog lists
the planet's ring features (:class:`RingFeatureEntry`), and each edge of a feature is
a list of mode entries (:class:`RingModeEntry`): the edge's base orbit and its
perturbations.

The catalog values are measured orbits taken from published fits, and the ring model
turns each entry into a :class:`~spindoctor.nav_model.rings.RingFeature` when it
renders.  Every check on a catalog is made here, when the configuration loads.

A planet key of ``ring_features`` must be a planet of the ``planets`` section.  That
rule reads two sections, so :class:`~spindoctor.config.sections.TypedSections` checks
it.

A field that has no default takes its value from the packaged YAML.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import AfterValidator

from spindoctor.config.sections_models import Rgb
from spindoctor.schema import (
    SCHEMA_CONFIG,
    Bool,
    Float,
    Int,
    NonNeg,
    NonNegInt,
    Positive,
    PositiveInt,
    RuleError,
    Str,
    Unit,
)
from spindoctor.support.time import utc_to_et

__all__ = [
    'RingCatalog',
    'RingFeatureEntry',
    'RingModeEntry',
    'RingsSection',
    'UtcTime',
]

# The keys of a base-orbit entry, and of a perturbation entry, in the order the
# catalogs write them.
_BASE_ORBIT_KEYS = ('a', 'rms', 'ae', 'long_peri', 'rate_peri')
_PERTURBATION_KEYS = ('amplitude', 'phase', 'pattern_speed')

# The one key an entry may leave out: a catalog without a fitted residual for a base
# orbit leaves out rms, and the edge's whole-edge orbit term then takes
# rings.default_orbit_radial_sigma_km.
_OPTIONAL_KEYS = frozenset({'rms'})


def _listed(keys: Sequence[str]) -> str:
    """Return keys as an English list, such as ``'a'``, ``'a and b'``, or ``'a, b, and c'``.

    Parameters:
        keys: The keys, at least one.

    Returns:
        The list.
    """
    if len(keys) < 3:
        return ' and '.join(keys)
    return f'{", ".join(keys[:-1])}, and {keys[-1]}'


def _check_utc(value: str) -> str:
    """Return a UTC time string unchanged if it can be read; refuse it if not.

    Parameters:
        value: The time, such as ``'2008-01-01 12:00:00'``.

    Returns:
        ``value``.

    Raises:
        ValueError: If the string is not a UTC time in ISO 8601 form.
    """
    try:
        utc_to_et(value)
    except ValueError as exc:
        raise ValueError(f'{value!r} is not a UTC time: {exc}') from exc
    return value


#: A UTC time in ISO 8601 form, such as ``'2008-01-01 12:00:00'`` or ``'2005-05-01'``.
UtcTime = Annotated[Str, AfterValidator(_check_utc)]


@dataclass(frozen=True)
class RingModeEntry:
    """One entry of a ring edge's mode list: the edge's base orbit, or one perturbation.

    An entry that sets ``a`` is the edge's base orbit.  It is mode 1, sets ``a``, ``ae``,
    ``long_peri``, and ``rate_peri``, and may set ``rms``; an edge whose base orbit leaves
    out ``rms`` takes ``rings.default_orbit_radial_sigma_km`` for its whole-edge orbit
    term.  Any other entry is a perturbation and sets ``amplitude``, ``phase``, and
    ``pattern_speed``.
    The mode number alone cannot tell the two apart, because a catalog may perturb an
    edge with a mode-1 term as well.

    Parameters:
        mode: The mode number.  A mode above 90 is an inclination mode, which the ring
            model keeps but does not render.
        a: The semi-major axis of the base orbit, in km.
        rms: The residual of the base orbit's fit, in km, or None when the catalog gives
            none.
        ae: The base orbit's radial eccentricity amplitude, in km.
        long_peri: The base orbit's longitude of pericenter at the catalog's epoch, in
            degrees.
        rate_peri: The base orbit's pericenter precession rate, in degrees per day.
        amplitude: The perturbation's radial amplitude, in km.
        phase: The perturbation's phase at the catalog's epoch, in degrees.
        pattern_speed: The perturbation's pattern speed, in degrees per day.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    mode: Int
    a: Positive | None = None
    rms: NonNeg | None = None
    ae: NonNeg | None = None
    long_peri: Float | None = None
    rate_peri: Float | None = None
    amplitude: NonNeg | None = None
    phase: Float | None = None
    pattern_speed: Float | None = None

    @property
    def is_base_orbit(self) -> bool:
        """Whether the entry is the edge's base orbit, which is when it sets ``a``."""
        return self.a is not None

    def __post_init__(self) -> None:
        """Refuse an entry that is neither a whole base orbit nor a whole perturbation."""
        own: tuple[str, ...] = _PERTURBATION_KEYS
        other: tuple[str, ...] = _BASE_ORBIT_KEYS
        kind = 'perturbation'
        if self.is_base_orbit:
            own, other, kind = other, own, 'base-orbit'
            if self.mode != 1:
                raise RuleError(
                    f'an entry that sets a is the base orbit, which is mode 1, not mode '
                    f'{self.mode}',
                    at=('mode',),
                    reads=(('mode',), ('a',)),
                )
        elif any(getattr(self, key) is not None for key in _BASE_ORBIT_KEYS):
            raise RuleError(
                f'this entry mixes the two kinds: it leaves out a but sets a key of a base '
                f'orbit; a base orbit sets {_listed(_BASE_ORBIT_KEYS)}, and a perturbation sets '
                f'{_listed(_PERTURBATION_KEYS)}',
                at=(),
            )
        required = [key for key in own if key not in _OPTIONAL_KEYS]
        missing = [key for key in required if getattr(self, key) is None]
        if missing:
            raise RuleError(
                f'a {kind} entry must set {_listed(required)}, but this one leaves out '
                f'{_listed(missing)}',
                at=(),
            )
        stray = [key for key in other if getattr(self, key) is not None]
        if stray:
            raise RuleError(
                f'a {kind} entry may set only {_listed(own)}, but this one also sets '
                f'{_listed(stray)}',
                at=(),
            )


def _check_edge(modes: tuple[RingModeEntry, ...], *, at: str) -> None:
    """Refuse an edge whose mode list has no base orbit, or more than one.

    Parameters:
        modes: The edge's mode entries.
        at: The edge's key, ``'inner_data'`` or ``'outer_data'``.

    Raises:
        RuleError: If the edge has no base-orbit entry or more than one.
    """
    count = sum(1 for mode in modes if mode.is_base_orbit)
    if count != 1:
        raise RuleError(
            f'an edge must have exactly one base-orbit entry (one that sets a), but this one '
            f'has {count}',
            at=(at,),
        )


@dataclass(frozen=True)
class RingFeatureEntry:
    """One ring feature of a catalog: a gap or a ringlet, with one or both edges.

    A feature with dates is in the catalog from ``start_date`` up to, but not including,
    ``end_date``; a date left out leaves that end open.

    Parameters:
        feature_type: ``'GAP'`` (dark between its edges) or ``'RINGLET'`` (bright
            between them).
        name: The name shown in labels; a feature left unnamed is labeled ``UNNAMED``.
        inner_data: The inner edge's mode entries, or None if the edge is not in the
            catalog.
        outer_data: The outer edge's mode entries, or None if the edge is not in the
            catalog.
        start_date: The UTC time from which the feature is in the catalog, or None for
            no start.
        end_date: The UTC time at which the feature leaves the catalog, or None for no
            end.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    feature_type: Literal['GAP', 'RINGLET']
    name: Str | None = None
    inner_data: tuple[RingModeEntry, ...] | None = None
    outer_data: tuple[RingModeEntry, ...] | None = None
    start_date: UtcTime | None = None
    end_date: UtcTime | None = None

    def __post_init__(self) -> None:
        """Refuse a feature that has no edge, a bad edge, or dates out of order."""
        if self.inner_data is None and self.outer_data is None:
            raise RuleError(
                'a feature needs an inner edge, an outer edge, or both; set inner_data, '
                'outer_data, or both',
                at=(),
                reads=(('inner_data',), ('outer_data',)),
            )
        if self.inner_data is not None:
            _check_edge(self.inner_data, at='inner_data')
        if self.outer_data is not None:
            _check_edge(self.outer_data, at='outer_data')
        start_et, end_et = self.dates_et
        if start_et is not None and end_et is not None and start_et >= end_et:
            raise RuleError(
                f'start_date ({self.start_date}) must be before end_date ({self.end_date})',
                at=('end_date',),
                reads=(('start_date',), ('end_date',)),
            )

    @property
    def base_radii(self) -> tuple[float, ...]:
        """The semi-major axis, in km, of the base orbit of each edge in the catalog."""
        radii: list[float] = []
        for modes in (self.inner_data, self.outer_data):
            if modes is not None:
                radii.extend(mode.a for mode in modes if mode.a is not None)
        return tuple(radii)

    @property
    def dates_et(self) -> tuple[float | None, float | None]:
        """``start_date`` and ``end_date`` as TDB seconds, each None if it is left out."""
        start_et = None if self.start_date is None else utc_to_et(self.start_date)
        end_et = None if self.end_date is None else utc_to_et(self.end_date)
        return start_et, end_et


def _overlap(first: RingFeatureEntry, second: RingFeatureEntry) -> bool:
    """Whether two features cover overlapping radii over overlapping dates.

    Two features overlap only when both have at least one date.  A feature that has no
    dates is in the catalog at every time, and may stand beside a dated feature at the
    same radii.  A date left out is an open end, and the date ranges are half open.

    Parameters:
        first: One feature.
        second: The other feature.

    Returns:
        True if the features' radial extents meet and their date ranges overlap.
    """
    first_radii, second_radii = first.base_radii, second.base_radii
    if max(first_radii) < min(second_radii) or max(second_radii) < min(first_radii):
        return False
    first_span, second_span = _date_span(first), _date_span(second)
    if first_span is None or second_span is None:
        return False
    return first_span[0] < second_span[1] and second_span[0] < first_span[1]


def _date_span(entry: RingFeatureEntry) -> tuple[float, float] | None:
    """A feature's dates as TDB seconds, with an open end infinite; None if it has none.

    Parameters:
        entry: The feature.

    Returns:
        ``(start, end)``, or None if the feature has neither date.
    """
    start_et, end_et = entry.dates_et
    if start_et is None and end_et is None:
        return None
    return (
        float('-inf') if start_et is None else start_et,
        float('inf') if end_et is None else end_et,
    )


@dataclass(frozen=True)
class RingCatalog:
    """One planet's ring catalog, and how the ring model filters and renders it.

    Parameters:
        epoch: The UTC time at which each orbit's longitudes and phases are given.
        fade_width_pix: The width, in pixels, of the fade drawn inward from an edge that
            has no partner edge.
        min_allowed_fade_width_pix: The fade width, in pixels, below which an edge
            whose fade is cut short by a neighboring edge is dropped.
        min_feature_pixels: The smallest width, in pixels, that a feature with both edges
            must have to be rendered.
        features: The features, keyed by a name unique within the catalog.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    epoch: UtcTime
    fade_width_pix: Positive
    min_allowed_fade_width_pix: Positive
    min_feature_pixels: Positive
    features: dict[Str, RingFeatureEntry]

    def __post_init__(self) -> None:
        """Refuse a minimum fade width above the fade width, and two features that overlap.

        Two features overlap when they cover overlapping radii over overlapping dates,
        which is two entries for one ring edge at one time.
        """
        if self.min_allowed_fade_width_pix > self.fade_width_pix:
            raise RuleError(
                f'min_allowed_fade_width_pix ({self.min_allowed_fade_width_pix}) must not '
                f'exceed fade_width_pix ({self.fade_width_pix})',
                at=('min_allowed_fade_width_pix',),
                reads=(('min_allowed_fade_width_pix',), ('fade_width_pix',)),
            )
        entries = list(self.features.items())
        for index, (first_key, first) in enumerate(entries):
            for second_key, second in entries[index + 1 :]:
                if _overlap(first, second):
                    raise RuleError(
                        f'{first_key!r} and {second_key!r} cover overlapping radii over '
                        'overlapping dates; change the dates so that they do not overlap',
                        at=('features', second_key),
                        reads=(('features', first_key), ('features', second_key)),
                    )


@dataclass(frozen=True)
class RingsSection:
    """The ``rings`` section: the ring model's settings and each planet's ring catalog.

    Parameters:
        default_orbit_radial_sigma_km: The radial orbit uncertainty, in km at one sigma,
            of an edge whose catalog entry has no fitted rms.
        orbit_radial_sigma_correlated_fraction: The share of an edge's orbit uncertainty
            that moves the whole edge together, and so is added to the ring-edge fit's
            covariance.
        remove_planet_shadow: Whether the part of the rings in the planet's shadow is
            left out of the ring model.
        label_font: The font file of a ring edge's label.
        label_font_size: The size of a ring edge's label, in points.
        label_font_color: The color of a ring edge's label.
        label_mask_enlarge: The distance, in pixels, around a ring that a label of
            another feature avoids.
        label_horiz_gap: The gap, in pixels, between an edge and a label to its left or
            right.
        label_vert_gap: The gap, in pixels, between an edge and a label above or below
            it.
        label_limb_color: The color of the ring edge outline in the overlay.
        ring_features: Each planet's ring catalog, keyed by the planet's SPICE name,
            which must be in the ``planets`` section.  A planet left out has no ring
            model.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    default_orbit_radial_sigma_km: NonNeg
    orbit_radial_sigma_correlated_fraction: Unit
    remove_planet_shadow: Bool
    label_font: Str
    label_font_size: PositiveInt
    label_font_color: Rgb
    label_mask_enlarge: PositiveInt
    label_horiz_gap: NonNegInt
    label_vert_gap: NonNegInt
    label_limb_color: Rgb
    ring_features: dict[Str, RingCatalog]
