"""The artifact-mode registry: the single source of truth for scene defects.

The ``artifacts`` scene block is keyed, besides ``instrument_defaults`` and
``adversarial``, by the artifact-mode names in this registry.  Each mode is
described once here -- its rendering stage (``telemetry`` or ``detector``), its
parameter schema, and whether it is implemented yet -- and every consumer (the
scene validator, the detector stage, the telemetry stage) reads that
description rather than carrying its own copy.  Registering a mode is therefore
the whole job of adding one: a detector-stage mode that is unimplemented today
drops in by flipping its ``implemented`` flag and adding its rendering code, with
no change to the validator or the block schema.  The module also lists the detector
models and the quantization modes that the scene validator, the instrument
schema, and the sim editor accept.

**Availability.**  Which instruments offer a mode is not part of the registry:
each instrument entry lists the modes it offers (see
:mod:`spindoctor.sim.instruments.catalog`), and may state why it does not offer
one.  If a scene sets a mode that its instrument does not offer, validation fails
with that reason, or with a message that lists the instruments that offer it.  The
instrument-agnostic generic entry offers every mode, which keeps unit scenes free
to exercise any shape.

**Activation.**  Every mode carries exactly one activation key, and the key's name
says what quantity it holds, so a scene line reads the same way on every mode:

* ``expected_events`` (count modes: lost lines, lost blocks, downlink gaps, spiked
  pixels, dead pixels and columns): the expected number of events per frame, drawn
  Poisson.  It is a non-negative number.  Each of these modes caps the draw at
  the number of places its events can land (the frame's lines, columns, or
  pixels, or ``missing_blocks``'s compression blocks).
* ``expected_events_per_sec`` (rate modes: ``bright_dark_pairs`` and
  ``radiation_transients``): the expected number of events per second, drawn
  Poisson and uncapped.  ``bright_dark_pairs`` expects this times
  ``exposure_sec``.  ``radiation_transients`` counts hits during the readout
  dwell as well as the exposure, in an environment scaled by its
  ``environment_factor``: it expects ``expected_events_per_sec *
  environment_factor * (exposure_sec + readout_dwell_sec)``.
* ``probability`` (commanded and periodic modes, and every deterministic-shape
  detector mode): the per-frame probability, in [0, 1], that the mode activates at
  all.
* ``pixel_fraction`` (``hot_pixels`` alone): the fraction of detector pixels that
  are hot, a spatial density in [0, 1].

The key defaults to 0.  Under ``instrument_defaults``, a mode that the camera's entry
turns on takes its activation key from the entry unless the scene sets it, so
``hot_pixels: {}`` plants the camera's hot pixels, and an activation key of 0 turns the
mode off.  A mode renders as a no-op when its activation key is 0, or, on
``dead_pixels`` and ``dead_columns``, when its ``count`` is 0 where given (the
stage-activation rule).  Those two modes accept ``count``, a fixed number of events in
place of the Poisson draw, and ``count`` and ``expected_events`` are mutually
exclusive: a map may set one of them.  Each mode's
:attr:`ArtifactMode.activation_semantics` says what its events are.

**Parameter types.**  Each parameter's type is an annotation from the schema core
(:mod:`spindoctor.schema`), such as ``NonNeg`` or ``Literal['drop', 'keep']``.  Each
mode builds one frozen dataclass from its parameters when this module is imported,
:attr:`ArtifactMode.params_type`, and :func:`parse_mode_params` checks a mode's map
against it.  Scene validation and the instrument catalog both check a mode's map
this way.  Every field of the type is optional and defaults to None, which stands
for a key the map leaves unset, so the type records which keys a map sets and leaves
the registry defaults to
:func:`~spindoctor.sim.instruments.resolvers.resolve_mode_with_catalog`.
"""

from __future__ import annotations

import copy
import dataclasses
import itertools
import keyword
from dataclasses import dataclass
from typing import Annotated, Any, Literal, get_args, get_origin

from annotated_types import Len
from pydantic import AfterValidator, Field

from spindoctor.schema import (
    SCHEMA_CONFIG,
    Bool,
    Int,
    NonNeg,
    NonNegInt,
    PositiveInt,
    Unit,
    parse,
)

__all__ = [
    'ACTIVATION_KEYS',
    'ARTIFACT_MODES',
    'DETECTOR_MODELS',
    'DETECTOR_MODE_ORDER',
    'EXPECTED_EVENTS',
    'EXPECTED_EVENTS_PER_SEC',
    'MODE_KEYS',
    'PERCENTILE_POINTS',
    'PIXEL_FRACTION',
    'PROBABILITY',
    'QUANTIZATION_MODES',
    'STRUCTURED_LOSS_ORDER',
    'TELEMETRY_POST_LOSS_ORDER',
    'TELEMETRY_PRE_LOSS_ORDER',
    'ArtifactMode',
    'DetectorModel',
    'IntChoice',
    'ModeParam',
    'QuantizationMode',
    'int_choice',
    'int_list',
    'mode_scene_map',
    'parse_mode_params',
    'unit_list',
]


@dataclass(frozen=True)
class ModeParam:
    """One parameter of an artifact mode.

    Parameters:
        name: The parameter key inside the mode's scene map.
        annotation: The value's type, written with the schema core's aliases:
            ``Bool``, ``NonNeg``, ``Unit``, ``NonNegInt``, ``PositiveInt``, a
            ``Literal`` of strings, :func:`int_choice`, :func:`int_list`, or
            :func:`unit_list`.
        default: The value used when the key is absent.  ``None`` marks an
            optional parameter with no default shape (the renderer supplies its
            own fallback, e.g. a centered window).
    """

    name: str
    annotation: Any
    default: Any = None

    @property
    def choices(self) -> tuple[Any, ...] | None:
        """The values a choice parameter takes, in order; None for any other parameter."""
        if get_origin(self.annotation) is Literal:
            return get_args(self.annotation)
        for marker in getattr(self.annotation, '__metadata__', ()):
            if isinstance(marker, AfterValidator) and isinstance(marker.func, IntChoice):
                return marker.func.choices
        return None

    @property
    def length(self) -> int | None:
        """The number of items of a list parameter; None for any other parameter."""
        for marker in getattr(self.annotation, '__metadata__', ()):
            if isinstance(marker, Len):
                return int(marker.min_length)
        return None

    @property
    def item_annotation(self) -> Any:
        """The type of each item of a list parameter; None for any other parameter."""
        if self.length is None:
            return None
        list_type = get_args(self.annotation)[0]
        return get_args(list_type)[0]


@dataclass(frozen=True)
class IntChoice:
    """The check of an integer parameter that takes one of a few values.

    A ``Literal`` holds only strings in a schema, so an integer choice is an ``Int``
    checked by this callable.  The sim editor reads :attr:`choices` for its menu.

    Parameters:
        choices: The permitted values, in the order the sim editor lists them.
    """

    choices: tuple[int, ...]

    def __call__(self, value: int) -> int:
        """Return ``value`` if it is one of the choices.

        Raises:
            ValueError: If it is not.
        """
        if value not in self.choices:
            raise ValueError(f'{value!r} is not one of {list(self.choices)}')
        return value


def int_choice(*choices: int) -> Any:
    """The type of an integer parameter that takes one of ``choices``.

    Parameters:
        *choices: The permitted values.

    Returns:
        ``Annotated[Int, AfterValidator(IntChoice(choices))]``.
    """
    return Annotated[Int, AfterValidator(IntChoice(choices))]


def int_list(length: int) -> Any:
    """The type of a parameter that is a list of exactly ``length`` integers.

    Parameters:
        length: The number of items.

    Returns:
        ``Annotated[list[Int], Len(length, length)]``.
    """
    return Annotated[list[Int], Len(length, length)]


def _nondecreasing(values: list[float]) -> list[float]:
    """Return ``values`` if no item is smaller than the one before it.

    Raises:
        ValueError: If one is.
    """
    if any(later < earlier for earlier, later in itertools.pairwise(values)):
        raise ValueError(f'{values!r} decreases; each value must be at least the one before it')
    return values


def unit_list(length: int) -> Any:
    """The type of a parameter that is a nondecreasing list of ``length`` numbers in [0, 1].

    A percentile table takes this type: its values rise with the percentile.

    Parameters:
        length: The number of items.

    Returns:
        ``Annotated[list[Unit], Len(length, length), AfterValidator(...)]``, which
        refuses a list whose values decrease.
    """
    return Annotated[list[Unit], Len(length, length), AfterValidator(_nondecreasing)]


# The cumulative probabilities of a percentile table: a mode parameter that gives a
# distribution by its values at the 0th, 10th, 25th, 50th, 75th, 90th, and 100th
# percentiles.  A renderer draws from it by interpolating linearly between the points.
PERCENTILE_POINTS: tuple[float, ...] = (0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0)


# The detector models, each a chain the detector stage implements: the CCD
# electron chain and the vidicon DN chain.  The order is the order the sim editor
# lists them in.
DetectorModel = Literal['ccd', 'vidicon']
DETECTOR_MODELS: tuple[str, ...] = get_args(DetectorModel)

# Quantization sub-modes: 'exact' rounds to integer DN (uniform bins); the ADC modes
# reproduce a coarse or uneven ADC's histogram structure, and 'sqrt_lut' companding
# decodes through a camera's downlink table.  The order is the order the sim editor
# lists them in.
QuantizationMode = Literal['exact', '8bit', 'uneven_12bit', 'sqrt_lut', 'ls8b', 'contour_8bit']
QUANTIZATION_MODES: tuple[str, ...] = get_args(QuantizationMode)

# The models a mode renders under: every model, or one chain's alone.
_EVERY_MODEL: frozenset[str] = frozenset(DETECTOR_MODELS)
_CCD_ONLY: frozenset[str] = frozenset({'ccd'})
_VIDICON_ONLY: frozenset[str] = frozenset({'vidicon'})

# The four activation keys.  Each mode carries exactly one of them as its first
# parameter; which one is part of the mode's registry entry.
EXPECTED_EVENTS = 'expected_events'
EXPECTED_EVENTS_PER_SEC = 'expected_events_per_sec'
PROBABILITY = 'probability'
PIXEL_FRACTION = 'pixel_fraction'

# Every activation key.  ``expected_events`` and ``expected_events_per_sec`` hold
# a Poisson expected event count, ``probability`` a per-frame activation
# probability in [0, 1], and ``pixel_fraction`` the fraction of detector pixels
# affected, in [0, 1].  A mode's own activation_semantics says what its events
# are and how its count scales with the exposure.
ACTIVATION_KEYS: frozenset[str] = frozenset(
    {EXPECTED_EVENTS, EXPECTED_EVENTS_PER_SEC, PROBABILITY, PIXEL_FRACTION}
)

# The schema of each activation key, disabled at its default of 0.
_ACTIVATION_PARAMS: dict[str, ModeParam] = {
    EXPECTED_EVENTS: ModeParam(EXPECTED_EVENTS, NonNeg, 0.0),
    EXPECTED_EVENTS_PER_SEC: ModeParam(EXPECTED_EVENTS_PER_SEC, NonNeg, 0.0),
    PROBABILITY: ModeParam(PROBABILITY, Unit, 0.0),
    PIXEL_FRACTION: ModeParam(PIXEL_FRACTION, Unit, 0.0),
}

# The fixed-count alternative to ``expected_events`` on the dead-pixel and
# dead-column modes (a static detector defect has a definite count).
_COUNT = ModeParam('count', NonNegInt, None)


@dataclass(frozen=True)
class ArtifactMode:
    """A registered artifact mode.

    Parameters:
        name: The registry key (also the ``artifacts`` block key).
        stage: The rendering stage that applies it, ``telemetry`` or
            ``detector``.
        implemented: Whether the renderer implements it yet.  An unimplemented
            mode is a reserved registry entry: it fixes the name and stage so
            the eventual implementation drops in without a
            schema change, and it fails validation with a clear message until
            its rendering code lands.
        params: The mode's parameters (the activation key first, then
            mode-specific).
        activation: The mode's activation key, one of :data:`ACTIVATION_KEYS`.
        activation_semantics: Human-readable meaning of the activation key for
            the mode, for the sim editor's tooltip and the developer guide.
        detector_models: The detector models whose chain applies the mode.  A
            telemetry-stage mode acts on the frame whatever its detector, and a
            detector-stage mode only where its chain has a stage for it, so a
            camera or scene whose model is not listed never renders it.

    The mode builds ``params_type``, the frozen schema dataclass of its map, from
    ``params`` when it is created.  The class has one field per parameter, each
    optional with a default of None.  A field whose key is a Python keyword, such as
    ``from``, takes the key followed by an underscore as its name and reads the key
    itself from a map.
    """

    name: str
    stage: str
    implemented: bool
    params: tuple[ModeParam, ...]
    activation: str
    activation_semantics: str
    detector_models: frozenset[str] = _EVERY_MODEL
    params_type: type[Any] = dataclasses.field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Build :attr:`params_type` from the parameters."""
        object.__setattr__(self, 'params_type', _build_params_type(self.name, self.params))

    @property
    def param_map(self) -> dict[str, ModeParam]:
        """The mode's parameters keyed by name."""
        return {param.name: param for param in self.params}

    @property
    def count_alternative(self) -> str | None:
        """The fixed-count key that replaces ``expected_events``, or None.

        Only ``dead_pixels`` and ``dead_columns`` carry one (``count``); the
        validator refuses a scene map that sets both keys.
        """
        return _COUNT.name if _COUNT.name in self.param_map else None


# The field metadata key that holds a parameter's key in a mode's map.
_MAP_KEY = 'key'


def _build_params_type(mode_name: str, params: tuple[ModeParam, ...]) -> type[Any]:
    """Build the schema dataclass of one mode's map.

    Parameters:
        mode_name: The mode's registry name, from which the class name is made.
        params: The mode's parameters.

    Returns:
        A frozen dataclass with one optional field per parameter, defaulting to None.
    """
    fields: list[tuple[str, Any, Any]] = []
    for param in params:
        annotation: Any = param.annotation | None
        attribute = param.name
        if keyword.iskeyword(param.name):
            attribute = f'{param.name}_'
            annotation = Annotated[annotation, Field(alias=param.name)]
        fields.append(
            (
                attribute,
                annotation,
                dataclasses.field(default=None, metadata={_MAP_KEY: param.name}),
            )
        )
    class_name = ''.join(part.title() for part in mode_name.split('_')) + 'Params'
    return dataclasses.make_dataclass(
        class_name,
        fields,
        frozen=True,
        namespace={
            '__pydantic_config__': SCHEMA_CONFIG,
            '__doc__': f'The parameters of the {mode_name!r} artifact mode.',
            '__module__': __name__,
        },
    )


def parse_mode_params(mode_name: str, raw: object, *, source: str) -> object:
    """Check a mode's map against its parameter types.

    Parameters:
        mode_name: A registered mode name.
        raw: The map, as a scene or a catalog entry writes it.
        source: What an error message gives as the map's source.

    Returns:
        The instance of the mode's :attr:`ArtifactMode.params_type`, whose values
        :func:`mode_scene_map` reads.  The class is built from the registry, so a type
        checker sees only an object.

    Raises:
        ~spindoctor.schema.errors.SchemaError: Listing each unknown key and each value
            of the wrong type or out of range.  A key path in a message is relative to
            the map.
    """
    params_type: type[object] = ARTIFACT_MODES[mode_name].params_type
    return parse(params_type, raw, source=source)


def mode_scene_map(values: object) -> dict[str, Any]:
    """The keys a mode's parsed map sets, as a scene writes them.

    Parameters:
        values: An instance of a mode's :attr:`ArtifactMode.params_type`.

    Returns:
        A fresh dict of each parameter the map sets, keyed as in the scene, such as
        ``from``.  A list in it is fresh too.  A parameter left unset is absent.
    """
    assert dataclasses.is_dataclass(values)
    scene_map: dict[str, Any] = {}
    for item in dataclasses.fields(values):
        value = getattr(values, item.name)
        if value is not None:
            scene_map[item.metadata[_MAP_KEY]] = copy.copy(value)
    return scene_map


def _telemetry_mode(
    name: str,
    *extra_params: ModeParam,
    activation: str,
    activation_semantics: str,
) -> ArtifactMode:
    """Build an implemented telemetry-stage loss mode led by its activation key."""
    return ArtifactMode(
        name=name,
        stage='telemetry',
        implemented=True,
        params=(_ACTIVATION_PARAMS[activation], *extra_params),
        activation=activation,
        activation_semantics=activation_semantics,
    )


def _detector_mode(
    name: str,
    *extra_params: ModeParam,
    activation: str,
    activation_semantics: str,
    detector_models: frozenset[str] = _EVERY_MODEL,
) -> ArtifactMode:
    """Build an implemented detector-stage electronics mode led by its activation key."""
    return ArtifactMode(
        name=name,
        stage='detector',
        implemented=True,
        params=(_ACTIVATION_PARAMS[activation], *extra_params),
        activation=activation,
        activation_semantics=activation_semantics,
        detector_models=detector_models,
    )


# ---------------------------------------------------------------------------
# The registry.
# ---------------------------------------------------------------------------

_TELEMETRY_LOSS_MODES: tuple[ArtifactMode, ...] = (
    _telemetry_mode(
        'missing_lines',
        ModeParam('contiguous_run', Bool, False),
        activation=EXPECTED_EVENTS,
        activation_semantics='a Poisson expected count of whole lines lost per frame',
    ),
    _telemetry_mode(
        'partial_lines',
        ModeParam('max_surviving_segments', PositiveInt, 1),
        activation=EXPECTED_EVENTS,
        activation_semantics='a Poisson expected count of truncated lines per frame',
    ),
    _telemetry_mode(
        'alternating_lines',
        ModeParam('period', int_choice(2, 4), 2),
        ModeParam('phase', NonNegInt, 0),
        # 'drop' blanks every Nth line (severe-Huffman entropy dropout); 'keep'
        # blanks all BUT every Nth line (a vertical-decimation readout format).
        ModeParam('mode', Literal['drop', 'keep'], 'drop'),
        activation=PROBABILITY,
        activation_semantics='per-frame probability the periodic line pattern is active',
    ),
    _telemetry_mode(
        'compression_truncated_lines',
        # A lossless compressor that must fit each pair of lines into half its raw
        # size drops the end of the pair's second line when the pair will not
        # compress that far.  The generic fallbacks cut at any sample and lose up
        # to half the line, uniformly; a camera's config states its compressor's
        # step and its measured lost shares as shape defaults.
        ModeParam('step_samples', PositiveInt, 1),
        ModeParam(
            'lost_share_percentiles', unit_list(7), (0.0, 0.05, 0.125, 0.25, 0.375, 0.45, 0.5)
        ),
        ModeParam('start_fraction', Unit, 0.0),
        activation=PROBABILITY,
        activation_semantics=(
            'per-frame probability that the second line of every line pair loses its end'
        ),
    ),
    _telemetry_mode(
        'edited_frame',
        # A generic fallback band width, so a bare probability keeps a centered
        # band; a camera's config states its own as a shape default.  An
        # explicit half_frame wins over the band.
        ModeParam('band_width_px', PositiveInt, 440),
        ModeParam('half_frame', Bool, False),
        ModeParam('half', Literal['top', 'bottom'], 'top'),
        activation=PROBABILITY,
        activation_semantics='per-frame probability the commanded edit is applied',
    ),
    _telemetry_mode(
        'truncated_frame',
        # A quarter-frame cut by default, so a bare probability truncates; an
        # explicit lines count wins over the fraction.
        ModeParam('fraction', Unit, 0.25),
        ModeParam('lines', NonNegInt, None),
        ModeParam('from', Literal['bottom', 'top'], 'bottom'),
        activation=PROBABILITY,
        activation_semantics='per-frame probability the frame is truncated',
    ),
    _telemetry_mode(
        'missing_blocks',
        # A generic fallback block height; a camera's config states its
        # compression block height as a shape default.
        ModeParam('block_lines', PositiveInt, 8),
        ModeParam('start_mid_line', Bool, False),
        activation=EXPECTED_EVENTS,
        activation_semantics='a Poisson expected count of compression blocks lost per frame',
    ),
    _telemetry_mode(
        'downlink_gaps',
        # Packets lost on the way to the ground take a group of whole lines that
        # starts on any line.  The generic fallbacks lose 8-line groups and keep
        # the line above each group whole; a camera's config states its measured
        # group sizes and head-line losses as shape defaults.
        ModeParam('median_lines', NonNeg, 8.0),
        ModeParam('log_sigma', NonNeg, 0.0),
        ModeParam('head_probability', Unit, 0.0),
        ModeParam('step_samples', PositiveInt, 1),
        activation=EXPECTED_EVENTS,
        activation_semantics=(
            'a Poisson expected count of downlink gaps per frame, each losing a group of '
            'whole lines'
        ),
    ),
    _telemetry_mode(
        'line_garble',
        activation=EXPECTED_EVENTS,
        activation_semantics='a Poisson expected count of garbled lines per frame',
    ),
    _telemetry_mode(
        'pixel_spikes',
        ModeParam('amplitude', Literal['bitflip', 'uniform'], 'bitflip'),
        activation=EXPECTED_EVENTS,
        activation_semantics='a Poisson expected count of spiked pixels per frame',
    ),
    _telemetry_mode(
        'dead_pixels',
        _COUNT,
        ModeParam('low_dn', NonNeg, 0.0),
        activation=EXPECTED_EVENTS,
        activation_semantics=(
            "a Poisson expected count of dead pixels per frame; 'count' plants a fixed number "
            'instead'
        ),
    ),
    _telemetry_mode(
        'dead_columns',
        _COUNT,
        ModeParam('low_dn', NonNeg, 0.0),
        activation=EXPECTED_EVENTS,
        activation_semantics=(
            "a Poisson expected count of dead columns per frame; 'count' plants a fixed number "
            'instead'
        ),
    ),
    _telemetry_mode(
        'embedded_header',
        # A generic fallback header length; a camera's config states its own as
        # a shape default.
        ModeParam('header_px', PositiveInt, 34),
        activation=PROBABILITY,
        activation_semantics='per-frame probability the row-0 housekeeping header is written',
    ),
    _telemetry_mode(
        'truth_window',
        # A generic fallback window size; a camera's config states its own as a
        # shape default.
        ModeParam('size', PositiveInt, 96),
        ModeParam('position', int_list(2)),
        activation=PROBABILITY,
        activation_semantics='per-frame probability the losslessly-clean carve-out is commanded',
    ),
    _telemetry_mode(
        'cutout_window',
        ModeParam('rect', int_list(4)),
        activation=PROBABILITY,
        activation_semantics='per-frame probability the commanded cut-out window is applied',
    ),
)

# Detector / electronics modes.  Each renders inside the detector stage at the
# physically right point in the unit chain (the readout dark with the dark
# current, banding in electrons pre-DN, fixed_pattern PRNU multiplicative
# pre-Poisson, serial_tail as a post-gain DN undershoot, and so on); the exact
# placement is documented per stage in the detector chain.  A mode with a
# per-instrument shape reads its shape parameters from the catalog when the
# scene leaves them unset.
_DETECTOR_MODES: tuple[ArtifactMode, ...] = (
    _detector_mode(
        'banding',
        ModeParam('amplitude', NonNeg, None),
        ModeParam('period_px', NonNeg, None),
        ModeParam('orientation', Literal['horizontal', 'vertical', 'both'], 'horizontal'),
        ModeParam('horizontal_pattern', Literal['periodic', 'random'], 'periodic'),
        ModeParam('freq_step_factor', NonNeg, 1.0),
        ModeParam('dark_step_dn', NonNeg, 0.0),
        ModeParam('ringing_amplitude', NonNeg, 0.0),
        ModeParam('ringing_period_px', NonNeg, 0.0),
        ModeParam('ringing_decay_px', NonNeg, 0.0),
        activation=PROBABILITY,
        activation_semantics='per-frame probability the banding is active',
    ),
    _detector_mode(
        'bias_structure',
        ModeParam('pedestal_sigma_dn', NonNeg, None),
        ModeParam('row_gradient_dn', NonNeg, None),
        ModeParam('col_gradient_dn', NonNeg, None),
        activation=PROBABILITY,
        activation_semantics='per-frame probability the bias pedestal / gradients are active',
    ),
    _detector_mode(
        'readout_dark',
        ModeParam('amplitude', NonNeg, None),
        ModeParam('nonlinear', NonNeg, 1.0),
        ModeParam('rbi_column_factor', Unit, 0.0),
        activation=PROBABILITY,
        activation_semantics='per-frame probability the readout dark is present',
    ),
    _detector_mode(
        'shutter_shading',
        ModeParam('top_factor', NonNeg, None),
        ModeParam('bottom_factor', NonNeg, None),
        activation=PROBABILITY,
        activation_semantics='per-frame probability the shutter shading is applied',
    ),
    _detector_mode(
        'bloom',
        ModeParam('bloom_length', PositiveInt, None),
        activation=PROBABILITY,
        activation_semantics='per-frame probability the electron-domain column bloom is active',
        detector_models=_CCD_ONLY,
    ),
    _detector_mode(
        'hot_pixels',
        ModeParam('amplitude_law', Literal['exponential', 'power_law'], 'exponential'),
        ModeParam('amplitude_index', NonNeg, None),
        ModeParam('amplitude_max', NonNeg, None),
        ModeParam('rate_e_per_sec', NonNeg, None),
        ModeParam('readout_sec', NonNeg, None),
        ModeParam('readout_e', NonNeg, None),
        ModeParam('streak_sec', NonNeg, None),
        ModeParam('streak_e', NonNeg, None),
        activation=PIXEL_FRACTION,
        activation_semantics=(
            'the fraction of detector pixels that are hot, a spatial density in [0, 1]'
        ),
        # The vidicon DN chain has no hot-pixel stage.
        detector_models=_CCD_ONLY,
    ),
    _detector_mode(
        'radiation_transients',
        ModeParam('environment_factor', NonNeg, 1.0),
        ModeParam('readout_dwell_sec', NonNeg, 0.0),
        ModeParam('amplitude_e', NonNeg, None),
        activation=EXPECTED_EVENTS_PER_SEC,
        activation_semantics=(
            'a Poisson expected count of radiation spikes per second; the frame expects this '
            'times environment_factor times (exposure_sec + readout_dwell_sec)'
        ),
        detector_models=_CCD_ONLY,
    ),
    _detector_mode(
        'bright_dark_pairs',
        ModeParam('amplitude_e', NonNeg, None),
        activation=EXPECTED_EVENTS_PER_SEC,
        activation_semantics=(
            'a Poisson expected count of bright/dark pairs per second of exposure; the frame '
            'expects this times exposure_sec'
        ),
        detector_models=_CCD_ONLY,
    ),
    _detector_mode(
        'frame_transfer_smear',
        ModeParam('t_scrub_sec', NonNeg, None),
        ModeParam('t_transfer_sec', NonNeg, None),
        activation=PROBABILITY,
        activation_semantics='per-frame probability the frame-transfer smear is applied',
        detector_models=_CCD_ONLY,
    ),
    _detector_mode(
        'serial_tail',
        ModeParam('amplitude_dn', NonNeg, 12.0),
        ModeParam('length_px', PositiveInt, 8),
        ModeParam('direction', Literal['right', 'left'], 'right'),
        ModeParam('saturation_frac', Unit, 0.99),
        activation=PROBABILITY,
        activation_semantics='per-frame probability the saturation serial tail is applied',
        detector_models=_CCD_ONLY,
    ),
    _detector_mode(
        'beam_bend',
        ModeParam('amplitude_px', NonNeg, 1.0),
        activation=PROBABILITY,
        activation_semantics='per-frame probability the brightness-dependent limb bend is applied',
        detector_models=_VIDICON_ONLY,
    ),
    _detector_mode(
        'residual_image',
        ModeParam('amplitude', Unit, 0.05),
        ModeParam('prior', Literal['self_offset', 'flat'], 'self_offset'),
        ModeParam('offset_px', int_list(2)),
        activation=PROBABILITY,
        activation_semantics='per-frame probability the prior-frame ghost is added',
        detector_models=_VIDICON_ONLY,
    ),
    _detector_mode(
        'fixed_pattern',
        ModeParam('prnu_rms', Unit, None),
        ModeParam('vignetting_frac', Unit, None),
        ModeParam('stitch_period_px', PositiveInt, None),
        ModeParam('stitch_amplitude_dn', NonNeg, None),
        ModeParam('jail_bar_dn', NonNeg, None),
        ModeParam('dust_donut_count', NonNegInt, None),
        activation=PROBABILITY,
        activation_semantics='per-frame probability the static fixed-pattern composite is applied',
        detector_models=_CCD_ONLY,
    ),
)

# Telemetry-stage artifact modes beyond the structured loss modes: the lossy
# compression blockiness (before structured loss) and the geometric-resample
# archive-processing scars (after structured loss).  ``resample_texture`` is a
# telemetry mode, not a detector mode: it emulates the archive resample the
# pipeline consumes, so it runs after the loss modes rather than in the detector.
_TELEMETRY_ARTIFACT_MODES: tuple[ArtifactMode, ...] = (
    _telemetry_mode(
        'compression_dct',
        ModeParam('scale_factor', NonNeg, 8.0),
        ModeParam('block', PositiveInt, 8),
        activation=PROBABILITY,
        activation_semantics='per-frame probability the lossy DCT compression is applied',
    ),
    _telemetry_mode(
        'reseau_scars',
        # Generic fallbacks for the lattice spacing, the scar radius, and the
        # number of marks the lattice is thinned to; a camera's config states
        # its own as shape defaults.
        ModeParam('spacing_px', PositiveInt, 46),
        ModeParam('patch_radius_px', PositiveInt, 4),
        ModeParam('mark_count', PositiveInt, 202),
        activation=PROBABILITY,
        activation_semantics='per-frame probability the reseau-removal scars are applied',
    ),
    _telemetry_mode(
        'resample_texture',
        ModeParam('warp_amp_px', NonNeg, 0.3),
        ModeParam('blank_border_px', NonNegInt, 0),
        ModeParam('missing_line_interp', Bool, False),
        activation=PROBABILITY,
        activation_semantics='per-frame probability the archive resample texture is applied',
    ),
)

ARTIFACT_MODES: dict[str, ArtifactMode] = {
    mode.name: mode
    for mode in (*_TELEMETRY_LOSS_MODES, *_DETECTOR_MODES, *_TELEMETRY_ARTIFACT_MODES)
}

# Pickle finds a class by its name in its module, so each registered mode's built class
# is put there, and an instrument entry that holds its instances can be pickled.
globals().update({mode.params_type.__name__: mode.params_type for mode in ARTIFACT_MODES.values()})

# The complete key set the ``artifacts`` block may carry beyond the two switch
# keys.  The validator keys on this exact set.
MODE_KEYS: frozenset[str] = frozenset(ARTIFACT_MODES)

# The fixed order the telemetry stage applies its implemented loss modes in:
# frame-level commanded shapes, then line losses, then block losses and downlink
# gaps, then garble, then per-pixel losses, then the row-0 header last.  ``truth_window``
# is not here: it is a protective carve-out resolved before the loop and passed
# to ``missing_blocks``, not a signal-mutating loss.
STRUCTURED_LOSS_ORDER: tuple[str, ...] = (
    'cutout_window',
    'edited_frame',
    'truncated_frame',
    'missing_lines',
    'partial_lines',
    'alternating_lines',
    'compression_truncated_lines',
    'missing_blocks',
    'downlink_gaps',
    'line_garble',
    'dead_columns',
    'pixel_spikes',
    'dead_pixels',
    'embedded_header',
)

# Telemetry-stage artifact modes that flank the structured loss loop.  Lossy DCT
# compression runs first, on the transmitted signal before any loss (a lossy
# codec compresses, then packets drop); the geometric-resample archive-processing
# scars run last, on the loss-bearing frame (they emulate the archive processing
# the pipeline consumes, after the raw losses are already present).
TELEMETRY_PRE_LOSS_ORDER: tuple[str, ...] = ('compression_dct',)
TELEMETRY_POST_LOSS_ORDER: tuple[str, ...] = ('reseau_scars', 'resample_texture')

# The detector-stage electronics modes, in the order the detector chain consults
# them.  Placement inside the unit chain is per-mode (documented in the chain);
# this order only fixes which mode's truth record is written first.
DETECTOR_MODE_ORDER: tuple[str, ...] = (
    'shutter_shading',
    'fixed_pattern',
    'residual_image',
    'readout_dark',
    'frame_transfer_smear',
    'hot_pixels',
    'bright_dark_pairs',
    'radiation_transients',
    'bloom',
    'banding',
    'serial_tail',
    'bias_structure',
    'beam_bend',
)
