"""The schema of an instrument entry, as typed frozen dataclasses.

Every simulated camera is one top-level entry of a YAML file in
``spindoctor/sim/instruments/configs/`` (see :mod:`spindoctor.sim.instruments.catalog`).
This module states what such an entry may hold.  An entry, after its base is merged
in, is parsed into an :class:`InstrumentEntry` by the schema core
(:func:`spindoctor.schema.parse`).  Each field's annotation gives the key's type and
range, and a field without a default is a key every entry must carry.  The detector
block and the readout modes are in :mod:`spindoctor.sim.instruments.schema_detector`.

The rules that tie one part of an entry to another are ``__post_init__`` methods of
the type that holds both parts, and a rule about one value is a check on that value's
annotation.  So a typo, a value written as a string, a negative gain, a missing
table, a setting the renderer could not honor, or a setting that would do nothing
fails when the catalog loads, and the message gives the file, the entry, and the key
path.  A rule on the entry as a whole begins its message with the key path it is
about, so its line reads like any other.

The two kinds of artifact-mode table an entry may state, ``shape_defaults`` and
``instrument_defaults``, have one optional field per registered mode, typed as that
mode's parameters (:attr:`spindoctor.sim.artifact_modes.ArtifactMode.params_type`).
Both are built from the mode registry when this module is imported, so each
parameter is declared once, in the registry.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, Any, Literal, TypeAlias

import numpy as np
from pydantic import AfterValidator

from spindoctor.schema import (
    SCHEMA_CONFIG,
    Bool,
    Float,
    NonNeg,
    NonNegInt,
    Positive,
    PositiveInt,
    Str,
    Unit,
)
from spindoctor.sim.artifact_modes import ARTIFACT_MODES, DETECTOR_MODE_ORDER, mode_scene_map
from spindoctor.sim.instruments.schema_detector import Detector, ReadoutModes

__all__ = [
    'CATALOG_BLIND_MODES',
    'GENERIC_INSTRUMENT',
    'NON_INHERITED_KEYS',
    'OFFER_EVERY_MODE',
    'ArtifactModes',
    'CohortCamera',
    'DataUnits',
    'DistortionResidual',
    'DownlinkTable',
    'DownlinkTables',
    'Filters',
    'InstrumentDefaultModes',
    'InstrumentEntry',
    'NavigatorLink',
    'Psf',
    'PsfKernel',
    'ShapeDefaults',
    'StarFlux',
    'canonical_filter_elements',
    'mode_table_maps',
    'parametric_decoded_dn',
]

# The name of the instrument-agnostic entry, which a scene renders through when it
# sets no instrument.  It is an entry like any other, in ``generic.yaml``.
GENERIC_INSTRUMENT = 'generic'

# The units a rendered frame is written in: detector counts, or calibrated I/F.
DataUnits = Literal['raw_dn', 'calibrated_if']

# The value of ``artifact_modes.offered`` that offers every registered mode.
OFFER_EVERY_MODE = 'all'

# Keys an entry does not pass on to an entry that sets it as its ``base``: the
# alternative names, and the real cameras the realism match pairs with the entry,
# each of which belongs to one entry only.
NON_INHERITED_KEYS: frozenset[str] = frozenset({'aliases', 'realism_cohort'})

# Modes whose shape the renderer takes from somewhere other than the per-instrument
# shape defaults, so a shape default for one would do nothing.  Each maps to where
# its shape comes from instead.
CATALOG_BLIND_MODES: dict[str, str] = {
    'hot_pixels': (
        'its shape comes from detector.hot_pixel_amplitude_e and detector.hot_pixel_column_factor'
    ),
}

# The fewest codes a downlink table may send.
_MIN_TABLE_CODES = 2


def canonical_filter_elements(filter_name: str) -> list[str]:
    """Split a filter combination into its elements as the catalog spells them.

    Each ``+``-separated element is stripped of surrounding whitespace and
    upper-cased, which is how a scene's ``detector.filter`` is read before lookup.

    Parameters:
        filter_name: A filter combination, such as ``'cl1 + grn'``.

    Returns:
        The elements, such as ``['CL1', 'GRN']``.
    """
    return [element.strip().upper() for element in filter_name.split('+')]


def _registered_mode(name: str) -> str:
    """Return ``name`` if it is a registered artifact mode.

    Raises:
        ValueError: If it is not.
    """
    if name not in ARTIFACT_MODES:
        raise ValueError(f'{name!r} is not a registered artifact mode')
    return name


def _canonical_combination(name: str) -> str:
    """Return a filter combination if a scene's filter could fold onto its spelling.

    A scene's filter is stripped and upper-cased element by element before lookup,
    so a key spelled any other way could never be selected.

    Raises:
        ValueError: If an element is empty or the spelling is not canonical.
    """
    elements = canonical_filter_elements(name)
    if '' in elements:
        raise ValueError(f'{name!r} has an empty filter element')
    canonical = '+'.join(elements)
    if canonical != name:
        raise ValueError(
            f'{name!r} is not in canonical form; write {canonical!r} (a scene filter is read '
            'stripped and upper-cased)'
        )
    return name


def _canonical_element(name: str) -> str:
    """Return a polarizer name if it is one filter element, spelled canonically.

    Raises:
        ValueError: If it holds a ``+`` or is not spelled canonically.
    """
    if len(canonical_filter_elements(name)) != 1:
        raise ValueError('a polarizer is one filter element, without a +')
    return _canonical_combination(name)


def _at_least_two_codes(codes: int) -> int:
    """Return a downlink table's code count if it is at least two.

    Raises:
        ValueError: If it is fewer.
    """
    if codes < _MIN_TABLE_CODES:
        raise ValueError(f'a downlink table sends at least {_MIN_TABLE_CODES} codes, got {codes}')
    return codes


def _check_decoded_levels(decoded: np.ndarray[Any, np.dtype[np.float64]]) -> None:
    """Refuse a downlink table with fewer than two decoded levels or a negative one.

    Codes that decode to one value are one quantization level, so a table needs as
    many distinct values as a working table needs codes; and the encoder snaps each
    DN, never negative, to the nearest level, so a negative level is DN no frame
    could hold.

    Raises:
        ValueError: If the table has fewer than two distinct levels or a negative one.
    """
    distinct = len(np.unique(decoded))
    if distinct < _MIN_TABLE_CODES:
        raise ValueError(
            f'a downlink table decodes to at least {_MIN_TABLE_CODES} distinct values, '
            f'got {distinct}'
        )
    lowest = float(decoded.min())
    if lowest < 0.0:
        raise ValueError(f'a downlink table decodes to no negative DN, got {lowest:g}')


def _decoded_levels(decoded: list[float]) -> list[float]:
    """Return a tabulated downlink table's levels if they pass the level check."""
    _check_decoded_levels(np.asarray(decoded, dtype=np.float64))
    return decoded


ModeName = Annotated[Str, AfterValidator(_registered_mode)]
FilterCombination = Annotated[Str, AfterValidator(_canonical_combination)]
Polarizer = Annotated[Str, AfterValidator(_canonical_element)]
TableCodes = Annotated[PositiveInt, AfterValidator(_at_least_two_codes)]
DecodedLevels = Annotated[list[NonNeg], AfterValidator(_decoded_levels)]


@dataclass(frozen=True)
class NavigatorLink:
    """The navigator block an instrument emulates.

    Parameters:
        section: The navigator configuration section, such as ``cassini_iss``.
        detector: The detector block inside the section, such as ``nac``, or None
            when the section itself is the block.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    section: Str
    detector: Str | None = None


@dataclass(frozen=True)
class StarFlux:
    """The star photometric zero point, in the detector model's unit.

    Exactly one of the two fields is set.

    Parameters:
        e_per_s_vmag0: Electrons per second from a magnitude-0 star (a CCD).
        dn_per_s_vmag0: DN per second from a magnitude-0 star (a vidicon).
    """

    __pydantic_config__ = SCHEMA_CONFIG

    e_per_s_vmag0: Positive | None = None
    dn_per_s_vmag0: Positive | None = None

    def __post_init__(self) -> None:
        """Refuse a zero point stated in both units or in neither."""
        if (self.e_per_s_vmag0 is None) == (self.dn_per_s_vmag0 is None):
            raise ValueError('state exactly one of e_per_s_vmag0 and dn_per_s_vmag0')


@dataclass(frozen=True)
class PsfKernel:
    """The native whole-scene PSF kernel: a Gaussian core with a Moffat wing.

    Parameters:
        sigma_v: The core's sigma along v, in native pixels.
        sigma_u: The core's sigma along u, in native pixels.
        w: The wing's share of the energy.
        r0: The wing's Moffat radius, in native pixels.
        n: The wing's Moffat index.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    sigma_v: Positive
    sigma_u: Positive
    w: Unit
    r0: Positive
    n: Float


@dataclass(frozen=True)
class Psf:
    """The PSF block.

    Parameters:
        truncation_px: The radius, in detector pixels, past which a kernel is cut
            and renormalized.
        kernel: The instrument's own kernel, or None for a camera that blurs only
            through a scene's ``optics.psf``.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    truncation_px: NonNegInt
    kernel: PsfKernel | None = None


@dataclass(frozen=True)
class DistortionResidual:
    """The geometric distortion left after the navigator's own distortion model.

    Parameters:
        k1: The radial coefficient on rho^2, rho the field radius normalized to the
            frame.
        k2: The radial coefficient on rho^4.
        nonradial_rms_px: The RMS of the non-radial wander, in native pixels.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    k1: Float
    k2: Float
    nonradial_rms_px: NonNeg


@dataclass(frozen=True)
class DownlinkTable:
    """One downlink table, in its parametric or its tabulated form.

    A parametric table sets ``codes``, ``linear_dn``, ``quadratic_dn``, and
    ``whole_dn``: code ``c`` decodes to ``linear_dn * c + quadratic_dn * c**2``,
    rounded to whole DN when ``whole_dn`` is set.  A tabulated table sets
    ``decoded_dn`` alone, the DN of each code in code order.  Either way the table
    decodes to at least two distinct levels, none negative.

    Parameters:
        codes: The number of codes a parametric table sends, at least 2.
        linear_dn: The linear coefficient of a parametric table.
        quadratic_dn: The quadratic coefficient of a parametric table.
        whole_dn: Whether a parametric table rounds to whole DN.
        decoded_dn: The levels of a tabulated table.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    codes: TableCodes | None = None
    linear_dn: Float | None = None
    quadratic_dn: Float | None = None
    whole_dn: Bool | None = None
    decoded_dn: DecodedLevels | None = None

    def __post_init__(self) -> None:
        """Refuse a table that mixes the two forms, or a parametric table's levels."""
        parametric = {
            'codes': self.codes,
            'linear_dn': self.linear_dn,
            'quadratic_dn': self.quadratic_dn,
            'whole_dn': self.whole_dn,
        }
        stated = sorted(
            name
            for name, value in (*parametric.items(), ('decoded_dn', self.decoded_dn))
            if value is not None
        )
        if stated == ['decoded_dn']:
            return
        if stated != sorted(parametric):
            raise ValueError(
                f'state either decoded_dn alone or all of {sorted(parametric)}; got {stated}'
            )
        _check_decoded_levels(parametric_decoded_dn(self))


def parametric_decoded_dn(table: DownlinkTable) -> np.ndarray[Any, np.dtype[np.float64]]:
    """The DN each code of a parametric table decodes to, in code order.

    This is the curve :class:`~spindoctor.sim.render.detector.sqrt_lut.LUTSqrt`
    decodes through: ``linear_dn * c + quadratic_dn * c**2`` for code ``c``, rounded
    to whole DN when ``whole_dn`` is set.

    Parameters:
        table: A parametric table: all four of its parametric fields are set.

    Returns:
        The decoded DN of each code.
    """
    assert table.codes is not None
    assert table.linear_dn is not None
    assert table.quadratic_dn is not None
    codes = np.arange(table.codes, dtype=np.float64)
    decoded = table.linear_dn * codes + table.quadratic_dn * codes**2
    return np.round(decoded) if table.whole_dn else decoded


@dataclass(frozen=True)
class DownlinkTables:
    """The downlink tables a camera lists.

    Parameters:
        default: The table a frame decodes through when its scene selects none.
        tables: Each table, keyed by name.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    default: Str
    tables: dict[Str, DownlinkTable]

    def __post_init__(self) -> None:
        """Refuse a default that is not one of the tables."""
        if self.default not in self.tables:
            raise ValueError(
                f'the default {self.default!r} is not one of the tables {sorted(self.tables)}'
            )


@dataclass(frozen=True)
class Filters:
    """The filter combinations a CCD camera catalogs.

    Parameters:
        throughput_e_per_s_per_if: Each combination's electrons per second per native
            pixel at I/F 1.0, keyed by the combination in canonical spelling, such as
            ``CL1+GRN``.
        clear: The clear combination the star zero point refers to.
        polarizer_clear: The clear element each polarizer folds onto, keyed by the
            polarizer.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    throughput_e_per_s_per_if: dict[FilterCombination, Positive]
    clear: Str
    polarizer_clear: dict[Polarizer, Str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Refuse a key no scene could select, a missing clear filter, or polarizer.

        A scene's polarizer folds onto its wheel's clear element before lookup, so a
        throughput key that holds a polarizer could never be selected.
        """
        polarizers = self.polarizer_clear
        for combination in self.throughput_e_per_s_per_if:
            parts = combination.split('+')
            for element in parts:
                if element in polarizers:
                    folded = '+'.join(polarizers.get(part, part) for part in parts)
                    raise ValueError(
                        f'the throughput key {combination!r} holds the polarizer {element!r}, '
                        f'which folds onto {polarizers[element]!r}, so a scene that sets '
                        f'{combination!r} reads {folded!r}; key the throughput by that'
                    )
        if self.clear not in self.throughput_e_per_s_per_if:
            raise ValueError(f'the clear filter {self.clear!r} is not in throughput_e_per_s_per_if')
        elements = {
            element
            for combination in self.throughput_e_per_s_per_if
            for element in combination.split('+')
        }
        for polarizer, clear in polarizers.items():
            if clear not in elements:
                raise ValueError(
                    f'the polarizer {polarizer} folds onto {clear!r}, which is not an element '
                    'of any cataloged filter combination'
                )


@dataclass(frozen=True)
class CohortCamera:
    """A real camera whose products an entry emulates in the realism match.

    Parameters:
        mission: The real cohort's mission, such as ``COISS``.
        camera: The camera within the mission, such as ``NAC``.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    mission: Str
    camera: Str


def _shape_default_rule(mode_name: str) -> AfterValidator:
    """The check of one mode's shape defaults.

    A shape default never turns a mode on, so it may carry neither the mode's
    activation key nor its fixed count, and a mode whose shape the renderer takes
    from elsewhere has no shape default at all.
    """
    mode = ARTIFACT_MODES[mode_name]

    def check(values: Any) -> Any:
        """Return the shape defaults if they could take effect."""
        if mode_name in CATALOG_BLIND_MODES:
            raise ValueError(
                f'a shape default for {mode_name!r} does nothing: {CATALOG_BLIND_MODES[mode_name]}'
            )
        for key in (mode.activation, mode.count_alternative):
            if key is not None and key in mode_scene_map(values):
                raise ValueError(
                    f'{key!r} turns the mode on, and a shape default never does; a scene '
                    'sets it, or the mode goes in artifact_modes.instrument_defaults'
                )
        return values

    return AfterValidator(check)


def _instrument_default_rule(mode_name: str) -> AfterValidator:
    """The check of one mode's ``instrument_defaults`` map.

    The detector chain consults ``instrument_defaults`` for its own stages only, and
    an entry there turns the mode on, so it sets a positive activation key.
    """
    activation = ARTIFACT_MODES[mode_name].activation

    def check(values: Any) -> Any:
        """Return the map if it turns a detector-stage mode on."""
        if mode_name not in DETECTOR_MODE_ORDER:
            raise ValueError(
                f'{mode_name!r} is not a detector-stage mode the detector chain consults'
            )
        if not mode_scene_map(values).get(activation):
            raise ValueError(
                f'an instrument_defaults entry turns the mode on, so it sets a positive '
                f'{activation!r}'
            )
        return values

    return AfterValidator(check)


def _mode_table(class_name: str, doc: str, rule: Any) -> type[Any]:
    """Build a table with one optional field per registered mode.

    Parameters:
        class_name: The class name.
        doc: The class docstring.
        rule: Called with a mode name, it returns the check of that mode's field.

    Returns:
        The frozen schema dataclass.
    """
    fields = [
        (
            name,
            Annotated[mode.params_type | None, rule(name)],
            dataclasses.field(default=None),
        )
        for name, mode in ARTIFACT_MODES.items()
    ]
    return dataclasses.make_dataclass(
        class_name,
        fields,
        frozen=True,
        namespace={'__pydantic_config__': SCHEMA_CONFIG, '__doc__': doc, '__module__': __name__},
    )


# Each table is a dataclass built from the registry, which a type checker cannot see
# through, so it reads the two names as Any.
if TYPE_CHECKING:
    ShapeDefaults: TypeAlias = Any
    InstrumentDefaultModes: TypeAlias = Any
else:
    ShapeDefaults = _mode_table(
        'ShapeDefaults',
        'The shape parameters a scene inherits for each mode it sets without them.',
        _shape_default_rule,
    )
    InstrumentDefaultModes = _mode_table(
        'InstrumentDefaultModes',
        "Each detector-stage mode a scene's instrument_defaults turns on, with its map.",
        _instrument_default_rule,
    )


def mode_table_maps(table: Any) -> dict[str, dict[str, Any]]:
    """The map of each mode a :data:`ShapeDefaults` or similar table sets.

    Parameters:
        table: A :data:`ShapeDefaults` or :data:`InstrumentDefaultModes` instance.

    Returns:
        Each mode the table sets, keyed by name, with its map as a scene writes it.
    """
    return {
        name: mode_scene_map(getattr(table, name))
        for name in ARTIFACT_MODES
        if getattr(table, name) is not None
    }


@dataclass(frozen=True)
class ArtifactModes:
    """The artifact modes an instrument offers, with their defaults.

    Parameters:
        offered: ``'all'``, or the registered modes the instrument offers.
        unavailable_reasons: Why the instrument does not offer a mode, keyed by the
            mode, for a scene that asks for it.
        instrument_defaults: The detector-stage modes a scene's
            ``instrument_defaults`` turns on, each with its map.
        shape_defaults: The shape parameters a scene inherits for each mode it sets
            without them.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    offered: Literal['all'] | list[ModeName]
    unavailable_reasons: dict[ModeName, Str] = field(default_factory=dict)
    instrument_defaults: InstrumentDefaultModes = field(default_factory=InstrumentDefaultModes)
    shape_defaults: ShapeDefaults = field(default_factory=ShapeDefaults)

    def __post_init__(self) -> None:
        """Refuse defaults for a mode that is not offered, or a reason for one that is."""
        for section, table in (
            ('shape_defaults', self.shape_defaults),
            ('instrument_defaults', self.instrument_defaults),
        ):
            for name in mode_table_maps(table):
                if not self.offers(name):
                    raise ValueError(f'{section} sets {name!r}, which is not in offered')
        for name in self.unavailable_reasons:
            if self.offers(name):
                raise ValueError(f'{name!r} is offered, so it has no unavailability reason')

    def offers(self, mode_name: str) -> bool:
        """Whether the instrument offers ``mode_name``.

        Parameters:
            mode_name: A registered mode name.

        Returns:
            True if ``offered`` is ``'all'`` or lists the mode.
        """
        return self.offered == OFFER_EVERY_MODE or mode_name in self.offered


@dataclass(frozen=True)
class InstrumentEntry:
    """One simulated camera's entry, after its base is merged in.

    Parameters:
        data_units: The units its frames are written in.
        navigator: The navigator block it emulates.
        detector: The detector chain's values.
        readout_modes: The detector values each binning changes, keyed by the
            binning factor; binning 1 is always present.
        star_flux: The star zero point.
        sky_pixel_scale_arcsec: The plate scale, in arcsec per native pixel.
        psf: The PSF block.
        artifact_modes: The artifact modes it offers, with their defaults.
        aliases: Other names a scene may give it.
        detector_id: The physical detector its frames come from, when it shares one
            with another entry; None for its own.
        native_frame_size_px: The native full frame's size, in pixels on a side.
        distortion_residual: The residual distortion, or None for none.
        downlink_tables: Its downlink tables, or None for none.
        filters: Its filter table, or None for none.
        realism_cohort: The real cameras whose products it emulates.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    data_units: DataUnits
    navigator: NavigatorLink
    detector: Detector
    readout_modes: ReadoutModes
    star_flux: StarFlux
    sky_pixel_scale_arcsec: Positive
    psf: Psf
    artifact_modes: ArtifactModes
    aliases: list[Str] = field(default_factory=list)
    detector_id: Str | None = None
    native_frame_size_px: PositiveInt | None = None
    distortion_residual: DistortionResidual | None = None
    downlink_tables: DownlinkTables | None = None
    filters: Filters | None = None
    realism_cohort: list[CohortCamera] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Refuse a combination of parts the renderer could not honor."""
        model = self.detector.detector_model
        if self.data_units != 'calibrated_if' and self.detector.dark_sky_ceiling_if is not None:
            raise ValueError(
                'detector.dark_sky_ceiling_if: the dark-sky ceiling applies only to '
                f"calibrated_if products, and this entry's data_units is {self.data_units}"
            )
        self._check_readout_modes()
        if model == 'vidicon' and self.star_flux.dn_per_s_vmag0 is None:
            raise ValueError('star_flux: a vidicon detector states dn_per_s_vmag0')
        if model != 'vidicon' and self.star_flux.e_per_s_vmag0 is None:
            raise ValueError(f'star_flux: a {model} detector states e_per_s_vmag0')
        if self.filters is not None and model == 'vidicon':
            raise ValueError(
                'filters: a filter sets the CCD signal scale; the vidicon DN chain has no '
                'reference exposure for it to set'
            )
        self._check_mode_chains()

    def _check_readout_modes(self) -> None:
        """Check the readout modes against the unbinned detector and the native size.

        A scene's gain state is checked against the unbinned table, so a readout mode
        may restate each state's gain but not add or drop a state, and its default
        state, if it sets one, must be in the table.
        """
        base_table = self.detector.gain_e_per_dn_by_state or {}
        native = self.native_frame_size_px
        for binning, mode in self.readout_modes.items():
            path = f'readout_modes.{binning}'
            if self.detector.detector_model == 'vidicon' and binning != 1:
                raise ValueError(f'{path}: the vidicon DN chain has no binned readout')
            table = mode.gain_e_per_dn_by_state
            if table is None:
                table = base_table
            if sorted(table) != sorted(base_table):
                raise ValueError(
                    f'{path}.gain_e_per_dn_by_state: the states {sorted(table)} differ from '
                    f'detector.gain_e_per_dn_by_state {sorted(base_table)}'
                )
            state = mode.default_gain_state
            if state is not None and state not in table:
                raise ValueError(
                    f'{path}.default_gain_state: state {state} is not in '
                    f'gain_e_per_dn_by_state {sorted(table)}'
                )
            if native is not None and native % binning != 0:
                raise ValueError(
                    f'{path}: binning {binning} does not divide native_frame_size_px {native}'
                )

    def _check_mode_chains(self) -> None:
        """Refuse a mode the entry lists that its detector model's chain never applies.

        If the entry offers a mode, gives it shape defaults, or turns it on, a scene
        that enables it would otherwise render nothing.
        """
        model = self.detector.detector_model
        modes = self.artifact_modes
        listed: list[tuple[str, str]] = []
        if modes.offered != OFFER_EVERY_MODE:
            listed.extend(('offered', name) for name in modes.offered)
        for section, table in (
            ('shape_defaults', modes.shape_defaults),
            ('instrument_defaults', modes.instrument_defaults),
        ):
            listed.extend((f'{section}.{name}', name) for name in mode_table_maps(table))
        for key, name in listed:
            chains = ARTIFACT_MODES[name].detector_models
            if model not in chains:
                raise ValueError(
                    f'artifact_modes.{key}: {name!r} renders only in the '
                    f'{" and ".join(sorted(chains))} detector chain; a {model} detector never '
                    'applies it'
                )

    def shape_default_map(self, mode_name: str) -> Mapping[str, Any]:
        """The shape defaults the entry states for one mode, as a scene writes them.

        Parameters:
            mode_name: A registered mode name.

        Returns:
            A fresh dict; empty when the entry states none for the mode.
        """
        values = getattr(self.artifact_modes.shape_defaults, mode_name)
        return {} if values is None else mode_scene_map(values)
