"""The artifact-mode registry: the single source of truth for scene defects.

The ``artifacts`` scene block is keyed, besides ``instrument_defaults`` and
``adversarial``, by the artifact-mode names in this registry.  Each mode is
described once here -- its rendering stage (``telemetry`` or ``detector``), its
parameter schema, and whether it is implemented yet -- and every consumer (the
scene validator, the telemetry stage, the detector hot-pixel routing) reads that
description rather than carrying its own copy.  Registering a mode is therefore
the whole job of adding one: a detector-stage mode that is unimplemented today
drops in by flipping its ``implemented`` flag and adding its rendering code, with
no change to the validator or the block schema.  The module also lists the detector
models and the quantization modes that the scene validator, the instrument
schema, and the sim editor accept.

**Availability.**  Which instruments offer a mode is not part of the registry:
each simulated instrument's config entry lists the modes it offers (see
:mod:`spindoctor.sim.instrument_catalog`), and may state why it does not offer
one.  If a scene sets a mode that its instrument does not offer, validation fails
with that reason, or with a message that lists the instruments that offer it.  The
instrument-agnostic generic entry offers every mode, which keeps unit scenes free
to exercise any shape.

**Activation.**  Every mode carries exactly one activation key, and the key's name
says what quantity it holds, so a scene line reads the same way on every mode:

* ``expected_events`` (count modes: lost lines, lost blocks, spiked pixels,
  dead pixels and columns): the expected number of events per frame, drawn
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

The key defaults to 0, also under ``instrument_defaults``.  A mode renders as a
no-op when its activation key is 0, or, on ``dead_pixels`` and ``dead_columns``,
when its ``count`` is 0 where given (the stage-activation rule).  Those two modes
accept ``count``, a fixed number of events in place of the Poisson draw, and
``count`` and ``expected_events`` are mutually exclusive: a map may set one of
them (a key set to null counts as unset).  Each mode's
:attr:`ArtifactMode.activation_semantics` says what its events are.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

__all__ = [
    'ACTIVATION_KEYS',
    'ARTIFACT_MODES',
    'COUNT_ACTIVATION_KEYS',
    'DETECTOR_MODELS',
    'DETECTOR_MODE_ORDER',
    'EXPECTED_EVENTS',
    'EXPECTED_EVENTS_PER_SEC',
    'MODE_KEYS',
    'PIXEL_FRACTION',
    'PROBABILITY',
    'QUANTIZATION_MODES',
    'STRUCTURED_LOSS_ORDER',
    'TELEMETRY_POST_LOSS_ORDER',
    'TELEMETRY_PRE_LOSS_ORDER',
    'ArtifactMode',
    'ModeParam',
    'resolve_mode_config',
]


@dataclass(frozen=True)
class ModeParam:
    """One parameter of an artifact mode.

    Parameters:
        name: The parameter key inside the mode's scene map.
        kind: The value's type tag, one of ``bool``, ``nonneg_number``,
            ``unit_interval``, ``int``, ``nonneg_int``, ``positive_int``,
            ``enum`` (with ``choices``), or ``int_list`` (with ``length``).
        default: The value used when the key is absent.  ``None`` marks an
            optional parameter with no default shape (the renderer supplies its
            own fallback, e.g. a centered window).
        choices: The permitted values for an ``enum`` parameter.
        length: The required length of an ``int_list`` parameter.
    """

    name: str
    kind: str
    default: Any = None
    choices: tuple[Any, ...] | None = None
    length: int | None = None


# The detector models, each a chain the detector stage implements: the CCD
# electron chain and the vidicon DN chain.  The order is the order the sim editor
# lists them in.
DETECTOR_MODELS: tuple[str, ...] = ('ccd', 'vidicon')

# Quantization sub-modes: 'exact' rounds to integer DN (uniform bins); the ADC modes
# reproduce a coarse or uneven ADC's histogram structure, and 'sqrt_lut' companding
# decodes through a camera's downlink table.  The order is the order the sim editor
# lists them in.
QUANTIZATION_MODES: tuple[str, ...] = (
    'exact',
    '8bit',
    'uneven_12bit',
    'sqrt_lut',
    'ls8b',
    'contour_8bit',
)

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

# The activation keys that hold a Poisson expectation: the mode's mechanic draws
# the event count, so the mode is active whenever the key is positive.
COUNT_ACTIVATION_KEYS: frozenset[str] = frozenset({EXPECTED_EVENTS, EXPECTED_EVENTS_PER_SEC})

# The schema of each activation key, disabled at its default of 0.
_ACTIVATION_PARAMS: dict[str, ModeParam] = {
    EXPECTED_EVENTS: ModeParam(EXPECTED_EVENTS, 'nonneg_number', 0.0),
    EXPECTED_EVENTS_PER_SEC: ModeParam(EXPECTED_EVENTS_PER_SEC, 'nonneg_number', 0.0),
    PROBABILITY: ModeParam(PROBABILITY, 'unit_interval', 0.0),
    PIXEL_FRACTION: ModeParam(PIXEL_FRACTION, 'unit_interval', 0.0),
}

# The fixed-count alternative to ``expected_events`` on the dead-pixel and
# dead-column modes (a static detector defect has a definite count).
_COUNT = ModeParam('count', 'nonneg_int', None)


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
    """

    name: str
    stage: str
    implemented: bool
    params: tuple[ModeParam, ...]
    activation: str
    activation_semantics: str
    detector_models: frozenset[str] = _EVERY_MODEL

    @property
    def param_map(self) -> dict[str, ModeParam]:
        """The mode's parameters keyed by name."""
        return {param.name: param for param in self.params}

    @property
    def count_alternative(self) -> str | None:
        """The fixed-count key that replaces ``expected_events``, or None.

        Only ``dead_pixels`` and ``dead_columns`` carry one (``count``); the
        validator refuses a scene map that sets both keys to a value (a key
        set to null counts as unset).
        """
        return _COUNT.name if _COUNT.name in self.param_map else None


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

_IMPLEMENTED_MODES: tuple[ArtifactMode, ...] = (
    _telemetry_mode(
        'missing_lines',
        ModeParam('contiguous_run', 'bool', False),
        activation=EXPECTED_EVENTS,
        activation_semantics='a Poisson expected count of whole lines lost per frame',
    ),
    _telemetry_mode(
        'partial_lines',
        ModeParam('max_surviving_segments', 'positive_int', 1),
        activation=EXPECTED_EVENTS,
        activation_semantics='a Poisson expected count of truncated lines per frame',
    ),
    _telemetry_mode(
        'alternating_lines',
        ModeParam('period', 'enum', 2, choices=(2, 4)),
        ModeParam('phase', 'nonneg_int', 0),
        # 'drop' blanks every Nth line (severe-Huffman entropy dropout); 'keep'
        # blanks all BUT every Nth line (a vertical-decimation readout format).
        ModeParam('mode', 'enum', 'drop', choices=('drop', 'keep')),
        activation=PROBABILITY,
        activation_semantics='per-frame probability the periodic line pattern is active',
    ),
    _telemetry_mode(
        'edited_frame',
        # A generic fallback band width, so a bare probability keeps a centered
        # band; a camera's config states its own as a shape default.  An
        # explicit half_frame wins over the band.
        ModeParam('band_width_px', 'positive_int', 440),
        ModeParam('half_frame', 'bool', False),
        ModeParam('half', 'enum', 'top', choices=('top', 'bottom')),
        activation=PROBABILITY,
        activation_semantics='per-frame probability the commanded edit is applied',
    ),
    _telemetry_mode(
        'truncated_frame',
        # A quarter-frame cut by default, so a bare probability truncates; an
        # explicit lines count wins over the fraction.
        ModeParam('fraction', 'unit_interval', 0.25),
        ModeParam('lines', 'nonneg_int', None),
        ModeParam('from', 'enum', 'bottom', choices=('bottom', 'top')),
        activation=PROBABILITY,
        activation_semantics='per-frame probability the frame is truncated',
    ),
    _telemetry_mode(
        'missing_blocks',
        # A generic fallback block height; a camera's config states its
        # compression block height as a shape default.
        ModeParam('block_lines', 'positive_int', 8),
        ModeParam('start_mid_line', 'bool', False),
        activation=EXPECTED_EVENTS,
        activation_semantics='a Poisson expected count of compression blocks lost per frame',
    ),
    _telemetry_mode(
        'line_garble',
        activation=EXPECTED_EVENTS,
        activation_semantics='a Poisson expected count of garbled lines per frame',
    ),
    _telemetry_mode(
        'pixel_spikes',
        ModeParam('amplitude', 'enum', 'bitflip', choices=('bitflip', 'uniform')),
        activation=EXPECTED_EVENTS,
        activation_semantics='a Poisson expected count of spiked pixels per frame',
    ),
    _telemetry_mode(
        'dead_pixels',
        _COUNT,
        ModeParam('low_dn', 'nonneg_number', 0.0),
        activation=EXPECTED_EVENTS,
        activation_semantics=(
            "a Poisson expected count of dead pixels per frame; 'count' plants a fixed number "
            'instead'
        ),
    ),
    _telemetry_mode(
        'dead_columns',
        _COUNT,
        ModeParam('low_dn', 'nonneg_number', 0.0),
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
        ModeParam('header_px', 'positive_int', 34),
        activation=PROBABILITY,
        activation_semantics='per-frame probability the row-0 housekeeping header is written',
    ),
    _telemetry_mode(
        'truth_window',
        # A generic fallback window size; a camera's config states its own as a
        # shape default.
        ModeParam('size', 'positive_int', 96),
        ModeParam('position', 'int_list', None, length=2),
        activation=PROBABILITY,
        activation_semantics='per-frame probability the losslessly-clean carve-out is commanded',
    ),
    _telemetry_mode(
        'cutout_window',
        ModeParam('rect', 'int_list', None, length=4),
        activation=PROBABILITY,
        activation_semantics='per-frame probability the commanded cut-out window is applied',
    ),
    # hot_pixels is implemented, but on the DETECTOR stage: the registry routes
    # its parameters to the detector hot-pixel population (params.py) rather
    # than to a telemetry applier.  Its activation key is pixel_fraction, the
    # fraction of pixels that are hot (a spatial density).
    ArtifactMode(
        name='hot_pixels',
        stage='detector',
        implemented=True,
        params=(
            _ACTIVATION_PARAMS[PIXEL_FRACTION],
            ModeParam('amplitude_e', 'nonneg_number', None),
            ModeParam('column_factor', 'unit_interval', None),
        ),
        activation=PIXEL_FRACTION,
        activation_semantics=(
            'the fraction of detector pixels that are hot, a spatial density in [0, 1]'
        ),
        # The vidicon DN chain has no hot-pixel stage.
        detector_models=_CCD_ONLY,
    ),
)

# Detector / electronics modes.  Each renders inside the detector stage at the
# physically right point in the unit chain (dark and dark_ramp with the dark
# pedestal, banding in electrons pre-DN, fixed_pattern PRNU multiplicative
# pre-Poisson, serial_tail as a post-gain DN undershoot, and so on); the exact
# placement is documented per stage in the detector chain.  A mode with a
# per-instrument shape reads its shape parameters from the catalog when the
# scene leaves them unset, and an explicit mode wins over the generic noise-block
# knob for the same mechanic (the precedence hot_pixels already uses).
_DETECTOR_MODES: tuple[ArtifactMode, ...] = (
    _detector_mode(
        'banding_coherent',
        ModeParam('amplitude_e', 'nonneg_number', None),
        ModeParam('period_px', 'nonneg_number', None),
        ModeParam('orientation', 'enum', 'horizontal', choices=('horizontal', 'vertical', 'both')),
        ModeParam('freq_step_factor', 'nonneg_number', 1.0),
        ModeParam('dark_step_dn', 'nonneg_number', 0.0),
        activation=PROBABILITY,
        activation_semantics='per-frame probability the coherent banding is active',
    ),
    _detector_mode(
        'bias_structure',
        ModeParam('pedestal_sigma_dn', 'nonneg_number', None),
        ModeParam('row_gradient_dn', 'nonneg_number', None),
        ModeParam('col_gradient_dn', 'nonneg_number', None),
        activation=PROBABILITY,
        activation_semantics='per-frame probability the bias pedestal / gradients are active',
    ),
    _detector_mode(
        'dark_ramp',
        ModeParam('kind', 'enum', 'dark_gradient', choices=('dark_gradient', 'exposure_shading')),
        ModeParam('amplitude_e', 'nonneg_number', None),
        ModeParam('nonlinear', 'nonneg_number', 1.0),
        ModeParam('rbi_column_factor', 'unit_interval', 0.0),
        ModeParam('top_factor', 'nonneg_number', None),
        ModeParam('bottom_factor', 'nonneg_number', None),
        activation=PROBABILITY,
        activation_semantics='per-frame probability the readout / shutter line ramp is active',
    ),
    _detector_mode(
        'bloom',
        ModeParam('bloom_length', 'positive_int', None),
        activation=PROBABILITY,
        activation_semantics='per-frame probability the electron-domain column bloom is active',
        detector_models=_CCD_ONLY,
    ),
    _detector_mode(
        'radiation_transients',
        ModeParam('environment_factor', 'nonneg_number', 1.0),
        ModeParam('readout_dwell_sec', 'nonneg_number', 0.0),
        ModeParam('amplitude_e', 'nonneg_number', None),
        activation=EXPECTED_EVENTS_PER_SEC,
        activation_semantics=(
            'a Poisson expected count of radiation spikes per second; the frame expects this '
            'times environment_factor times (exposure_sec + readout_dwell_sec)'
        ),
        detector_models=_CCD_ONLY,
    ),
    _detector_mode(
        'bright_dark_pairs',
        ModeParam('amplitude_e', 'nonneg_number', None),
        activation=EXPECTED_EVENTS_PER_SEC,
        activation_semantics=(
            'a Poisson expected count of bright/dark pairs per second of exposure; the frame '
            'expects this times exposure_sec'
        ),
        detector_models=_CCD_ONLY,
    ),
    _detector_mode(
        'frame_transfer_smear',
        ModeParam('t_scrub_sec', 'nonneg_number', None),
        ModeParam('t_transfer_sec', 'nonneg_number', None),
        activation=PROBABILITY,
        activation_semantics='per-frame probability the frame-transfer smear is applied',
        detector_models=_CCD_ONLY,
    ),
    _detector_mode(
        'serial_tail',
        ModeParam('amplitude_dn', 'nonneg_number', 12.0),
        ModeParam('length_px', 'positive_int', 8),
        ModeParam('direction', 'enum', 'right', choices=('right', 'left')),
        ModeParam('saturation_frac', 'unit_interval', 0.99),
        activation=PROBABILITY,
        activation_semantics='per-frame probability the saturation serial tail is applied',
        detector_models=_CCD_ONLY,
    ),
    _detector_mode(
        'beam_bend',
        ModeParam('amplitude_px', 'nonneg_number', 1.0),
        activation=PROBABILITY,
        activation_semantics='per-frame probability the brightness-dependent limb bend is applied',
        detector_models=_VIDICON_ONLY,
    ),
    _detector_mode(
        'residual_image',
        ModeParam('amplitude', 'unit_interval', 0.05),
        ModeParam('prior', 'enum', 'self_offset', choices=('self_offset', 'flat')),
        ModeParam('offset_px', 'int_list', None, length=2),
        activation=PROBABILITY,
        activation_semantics='per-frame probability the prior-frame ghost is added',
        detector_models=_VIDICON_ONLY,
    ),
    _detector_mode(
        'quantization_sqrt_lut',
        activation=PROBABILITY,
        activation_semantics=(
            "per-frame probability the frame is companded through the camera's sqrt_lut table"
        ),
    ),
    _detector_mode(
        'quantization_ls8b',
        activation=PROBABILITY,
        activation_semantics='per-frame probability the LS8B wraparound quantization is active',
    ),
    _detector_mode(
        'quantization_contour_8bit',
        ModeParam('step', 'positive_int', 8),
        activation=PROBABILITY,
        activation_semantics='per-frame probability the 8-bit contouring quantization is active',
    ),
    _detector_mode(
        'fixed_pattern',
        ModeParam('prnu_rms', 'unit_interval', None),
        ModeParam('vignetting_frac', 'unit_interval', None),
        ModeParam('stitch_period_px', 'positive_int', None),
        ModeParam('stitch_amplitude_dn', 'nonneg_number', None),
        ModeParam('jail_bar_dn', 'nonneg_number', None),
        ModeParam('dust_donut_count', 'nonneg_int', None),
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
        ModeParam('scale_factor', 'nonneg_number', 8.0),
        ModeParam('block', 'positive_int', 8),
        activation=PROBABILITY,
        activation_semantics='per-frame probability the lossy DCT compression is applied',
    ),
    _telemetry_mode(
        'reseau_scars',
        # Generic fallbacks for the lattice spacing, the scar radius, and the
        # number of marks the lattice is thinned to; a camera's config states
        # its own as shape defaults.
        ModeParam('spacing_px', 'positive_int', 46),
        ModeParam('patch_radius_px', 'positive_int', 4),
        ModeParam('mark_count', 'positive_int', 202),
        activation=PROBABILITY,
        activation_semantics='per-frame probability the reseau-removal scars are applied',
    ),
    _telemetry_mode(
        'resample_texture',
        ModeParam('warp_amp_px', 'nonneg_number', 0.3),
        ModeParam('blank_border_px', 'nonneg_int', 0),
        ModeParam('missing_line_interp', 'bool', False),
        activation=PROBABILITY,
        activation_semantics='per-frame probability the archive resample texture is applied',
    ),
)

ARTIFACT_MODES: dict[str, ArtifactMode] = {
    mode.name: mode for mode in (*_IMPLEMENTED_MODES, *_DETECTOR_MODES, *_TELEMETRY_ARTIFACT_MODES)
}

# The complete key set the ``artifacts`` block may carry beyond the two switch
# keys.  The validator keys on this exact set.
MODE_KEYS: frozenset[str] = frozenset(ARTIFACT_MODES)

# The fixed order the telemetry stage applies its implemented loss modes in:
# frame-level commanded shapes, then line losses, then block losses, then
# garble, then per-pixel losses, then the row-0 header last.  ``truth_window``
# is not here: it is a protective carve-out resolved before the loop and passed
# to ``missing_blocks``, not a signal-mutating loss.  ``hot_pixels`` is not here
# either: it is a detector-stage mode.
STRUCTURED_LOSS_ORDER: tuple[str, ...] = (
    'cutout_window',
    'edited_frame',
    'truncated_frame',
    'missing_lines',
    'partial_lines',
    'alternating_lines',
    'missing_blocks',
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
    'fixed_pattern',
    'residual_image',
    'dark_ramp',
    'frame_transfer_smear',
    'bright_dark_pairs',
    'radiation_transients',
    'bloom',
    'banding_coherent',
    'serial_tail',
    'bias_structure',
    'beam_bend',
    'quantization_sqrt_lut',
    'quantization_ls8b',
    'quantization_contour_8bit',
)


def resolve_mode_config(mode_name: str, raw_config: Mapping[str, Any]) -> dict[str, Any]:
    """Fill a mode's scene map with its parameter defaults for rendering.

    Parameters:
        mode_name: A registered mode name.
        raw_config: The scene's map for the mode (already validated).

    Returns:
        A fresh dict carrying every parameter, scene value overriding default.
        A key set to ``None`` reads as unset and takes the default, as in
        :func:`~spindoctor.sim.instrument_resolvers.resolve_mode_with_catalog`.
    """
    mode = ARTIFACT_MODES[mode_name]
    resolved: dict[str, Any] = {}
    for param in mode.params:
        value = raw_config.get(param.name)
        resolved[param.name] = param.default if value is None else value
    return resolved
