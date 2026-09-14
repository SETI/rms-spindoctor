"""The artifact-mode registry: the single source of truth for scene defects.

The ``artifacts`` scene block is keyed, besides ``instrument_defaults`` and
``adversarial``, by the artifact-mode names in this registry.  Each mode is
described once here -- its rendering stage (``telemetry`` or ``detector``), its
parameter schema, the instruments it is available on, and whether it is
implemented yet -- and every consumer (the scene validator, the telemetry
stage, the detector hot-pixel routing) reads that description rather than
carrying its own copy.  Registering a mode is therefore the whole job of adding
one: a detector-stage mode that is unimplemented today drops in by flipping its
``implemented`` flag and adding its rendering code, with no change to the
validator or the block schema.

**Availability.**  A mode lists the sim instruments it is available on; a scene
that names the mode on any other instrument fails validation with a clear
message (the LORRI hot-pixel case carries a bespoke one, since LORRI has no hot
pixels by construction).  The instrument-agnostic ``generic`` / ``sim`` block
accepts every mode, which keeps unit scenes free to exercise any shape.

**Incidence.**  Every mode takes an ``incidence`` parameter, disabled at 0 (the
stage-activation rule: an incidence of 0, which is also the default even under
``instrument_defaults``, renders the mode as a no-op).  Its meaning is per-mode
and documented in each mode's ``incidence_semantics``: for count modes it is the
expected number of events (lost lines, blocks, spiked pixels) per frame, drawn
Poisson; for commanded / periodic modes it is the per-frame probability that the
mode activates at all.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    'ARTIFACT_MODES',
    'DETECTOR_MODE_ORDER',
    'MODE_KEYS',
    'STRUCTURED_LOSS_ORDER',
    'TELEMETRY_POST_LOSS_ORDER',
    'TELEMETRY_PRE_LOSS_ORDER',
    'ArtifactMode',
    'ModeParam',
    'mode_available',
    'mode_unavailable_message',
    'normalize_instrument',
    'resolve_mode_config',
]


# Instrument groupings used by the availability tables below.  ``generic`` is
# never listed: it is accepted unconditionally by mode_available.
_ALL_CCD: frozenset[str] = frozenset({'coiss_nac', 'coiss_wac', 'gossi', 'nhlorri'})
_ALL_INSTRUMENTS: frozenset[str] = _ALL_CCD | frozenset({'vgiss'})


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


# The incidence parameter every mode carries: expected events per frame (count
# modes) or per-frame activation probability (commanded / periodic modes),
# disabled at 0.
_INCIDENCE = ModeParam('incidence', 'nonneg_number', 0.0)


@dataclass(frozen=True)
class ArtifactMode:
    """A registered artifact mode.

    Parameters:
        name: The registry key (also the ``artifacts`` block key).
        stage: The rendering stage that applies it, ``telemetry`` or
            ``detector``.
        implemented: Whether the renderer implements it yet.  An unimplemented
            mode is a reserved registry entry: it fixes the name, stage, and
            availability so the eventual implementation drops in without a
            schema change, and it fails validation with a clear message until
            its rendering code lands.
        params: The mode's parameters (``incidence`` first, then mode-specific).
        availability: The sim instruments the mode is available on (the generic
            block is always accepted, so it is never listed here).
        incidence_semantics: Human-readable meaning of ``incidence`` for the
            mode, for the validator's messages and the developer guide.
        unavailable_reason: Per-instrument bespoke unavailability messages
            (e.g. LORRI hot pixels); other instruments get a generic message.
    """

    name: str
    stage: str
    implemented: bool
    params: tuple[ModeParam, ...]
    availability: frozenset[str]
    incidence_semantics: str
    unavailable_reason: Mapping[str, str] = field(default_factory=dict)

    @property
    def param_map(self) -> dict[str, ModeParam]:
        """The mode's parameters keyed by name."""
        return {param.name: param for param in self.params}


def _telemetry_mode(
    name: str,
    *extra_params: ModeParam,
    availability: frozenset[str],
    incidence_semantics: str,
    unavailable_reason: Mapping[str, str] | None = None,
) -> ArtifactMode:
    """Build an implemented telemetry-stage loss mode with an incidence param."""
    return ArtifactMode(
        name=name,
        stage='telemetry',
        implemented=True,
        params=(_INCIDENCE, *extra_params),
        availability=availability,
        incidence_semantics=incidence_semantics,
        unavailable_reason=unavailable_reason or {},
    )


def _detector_mode(
    name: str,
    *extra_params: ModeParam,
    availability: frozenset[str],
    incidence_semantics: str,
    unavailable_reason: Mapping[str, str] | None = None,
) -> ArtifactMode:
    """Build an implemented detector-stage electronics mode with an incidence param."""
    return ArtifactMode(
        name=name,
        stage='detector',
        implemented=True,
        params=(_INCIDENCE, *extra_params),
        availability=availability,
        incidence_semantics=incidence_semantics,
        unavailable_reason=unavailable_reason or {},
    )


# ---------------------------------------------------------------------------
# The registry.
# ---------------------------------------------------------------------------

_IMPLEMENTED_MODES: tuple[ArtifactMode, ...] = (
    _telemetry_mode(
        'missing_lines',
        ModeParam('contiguous_run', 'bool', False),
        availability=frozenset({'coiss_nac', 'coiss_wac', 'vgiss', 'gossi', 'nhlorri'}),
        incidence_semantics='expected number of whole lines lost per frame (Poisson)',
    ),
    _telemetry_mode(
        'partial_lines',
        ModeParam('max_surviving_segments', 'positive_int', 1),
        availability=frozenset({'coiss_nac', 'coiss_wac', 'vgiss'}),
        incidence_semantics='expected number of truncated lines per frame (Poisson)',
    ),
    _telemetry_mode(
        'alternating_lines',
        ModeParam('period', 'enum', 2, choices=(2, 4)),
        ModeParam('phase', 'nonneg_int', 0),
        # 'drop' blanks every Nth line (severe-Huffman entropy dropout); 'keep'
        # blanks all BUT every Nth line (Galileo HMA/HCA vertical decimation).
        ModeParam('mode', 'enum', 'drop', choices=('drop', 'keep')),
        availability=frozenset({'coiss_nac', 'coiss_wac', 'gossi'}),
        incidence_semantics='per-frame probability the periodic line pattern is active',
    ),
    _telemetry_mode(
        'edited_frame',
        # The Voyager IM edited-mode band width, so a bare incidence keeps a
        # physical centered band; an explicit half_frame wins over the band.
        ModeParam('band_width_px', 'positive_int', 440),
        ModeParam('half_frame', 'bool', False),
        ModeParam('half', 'enum', 'top', choices=('top', 'bottom')),
        availability=frozenset({'vgiss', 'gossi'}),
        incidence_semantics='per-frame probability the commanded edit is applied',
    ),
    _telemetry_mode(
        'truncated_frame',
        # A quarter-frame cut by default, so a bare incidence truncates; an
        # explicit lines count wins over the fraction.
        ModeParam('fraction', 'unit_interval', 0.25),
        ModeParam('lines', 'nonneg_int', None),
        ModeParam('from', 'enum', 'bottom', choices=('bottom', 'top')),
        availability=frozenset({'coiss_nac', 'coiss_wac', 'gossi'}),
        incidence_semantics='per-frame probability the frame is truncated',
    ),
    _telemetry_mode(
        'missing_blocks',
        ModeParam('block_lines', 'positive_int', 8),
        ModeParam('start_mid_line', 'bool', False),
        availability=frozenset({'gossi', 'coiss_nac', 'coiss_wac'}),
        incidence_semantics='expected number of compression blocks lost per frame (Poisson)',
    ),
    _telemetry_mode(
        'line_garble',
        availability=frozenset({'vgiss', 'gossi'}),
        incidence_semantics='expected number of garbled lines per frame (Poisson)',
    ),
    _telemetry_mode(
        'pixel_spikes',
        ModeParam('amplitude', 'enum', 'bitflip', choices=('bitflip', 'uniform')),
        availability=frozenset({'vgiss'}),
        incidence_semantics='expected number of spiked pixels per frame (Poisson)',
    ),
    _telemetry_mode(
        'dead_pixels',
        ModeParam('count', 'nonneg_int', None),
        ModeParam('low_dn', 'nonneg_number', 0.0),
        availability=frozenset({'coiss_nac', 'coiss_wac', 'gossi', 'nhlorri'}),
        incidence_semantics='expected number of dead pixels per frame (Poisson) when count absent',
    ),
    _telemetry_mode(
        'dead_columns',
        ModeParam('count', 'nonneg_int', None),
        ModeParam('low_dn', 'nonneg_number', 0.0),
        availability=frozenset({'coiss_nac', 'coiss_wac', 'gossi', 'nhlorri'}),
        incidence_semantics='expected number of dead columns per frame (Poisson) when count absent',
    ),
    _telemetry_mode(
        'embedded_header',
        ModeParam('header_px', 'positive_int', 34),
        availability=frozenset({'nhlorri'}),
        incidence_semantics='per-frame probability the row-0 housekeeping header is written',
    ),
    _telemetry_mode(
        'truth_window',
        ModeParam('size', 'positive_int', 96),
        ModeParam('position', 'int_list', None, length=2),
        availability=frozenset({'gossi'}),
        incidence_semantics='per-frame probability the losslessly-clean carve-out is commanded',
    ),
    _telemetry_mode(
        'cutout_window',
        ModeParam('rect', 'int_list', None, length=4),
        availability=frozenset({'gossi', 'nhlorri'}),
        incidence_semantics='per-frame probability the commanded cut-out window is applied',
    ),
    # hot_pixels is implemented, but on the DETECTOR stage: the registry routes
    # its parameters to the detector hot-pixel population (params.py) rather
    # than to a telemetry applier.  incidence is the fraction of pixels that are
    # hot (a spatial density, not a per-frame event count).
    ArtifactMode(
        name='hot_pixels',
        stage='detector',
        implemented=True,
        params=(
            _INCIDENCE,
            ModeParam('amplitude_e', 'nonneg_number', None),
            ModeParam('column_factor', 'unit_interval', None),
        ),
        availability=frozenset({'coiss_nac', 'coiss_wac', 'gossi'}),
        incidence_semantics='fraction of pixels that are hot (spatial density)',
        unavailable_reason={
            'nhlorri': 'explicitly disabled for LORRI, which has none',
        },
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
        availability=_ALL_INSTRUMENTS,
        incidence_semantics='per-frame probability the coherent banding is active',
    ),
    _detector_mode(
        'bias_structure',
        ModeParam('pedestal_sigma_dn', 'nonneg_number', None),
        ModeParam('row_gradient_dn', 'nonneg_number', None),
        ModeParam('col_gradient_dn', 'nonneg_number', None),
        availability=_ALL_INSTRUMENTS,
        incidence_semantics='per-frame probability the bias pedestal / gradients are active',
    ),
    _detector_mode(
        'dark_ramp',
        ModeParam('kind', 'enum', 'dark_gradient', choices=('dark_gradient', 'exposure_shading')),
        ModeParam('amplitude_e', 'nonneg_number', None),
        ModeParam('nonlinear', 'nonneg_number', 1.0),
        ModeParam('rbi_column_factor', 'unit_interval', 0.0),
        ModeParam('top_factor', 'nonneg_number', None),
        ModeParam('bottom_factor', 'nonneg_number', None),
        availability=_ALL_INSTRUMENTS,
        incidence_semantics='per-frame probability the readout / shutter line ramp is active',
    ),
    _detector_mode(
        'bloom',
        ModeParam('bloom_length', 'positive_int', None),
        availability=frozenset({'coiss_nac', 'coiss_wac', 'gossi'}),
        incidence_semantics='per-frame probability the electron-domain column bloom is active',
        unavailable_reason={
            'nhlorri': 'explicitly disabled for LORRI, an antiblooming CCD with no column bloom',
        },
    ),
    _detector_mode(
        'radiation_transients',
        ModeParam('environment_factor', 'nonneg_number', 1.0),
        ModeParam('readout_dwell_sec', 'nonneg_number', 0.0),
        ModeParam('amplitude_e', 'nonneg_number', None),
        availability=_ALL_CCD,
        incidence_semantics='expected radiation spikes per frame at environment_factor 1 (Poisson)',
    ),
    _detector_mode(
        'bright_dark_pairs',
        ModeParam('amplitude_e', 'nonneg_number', None),
        availability=frozenset({'coiss_nac', 'coiss_wac'}),
        incidence_semantics=(
            'expected bright/dark pairs per frame at unit exposure (Poisson; scales with exposure)'
        ),
    ),
    _detector_mode(
        'frame_transfer_smear',
        ModeParam('t_scrub_sec', 'nonneg_number', None),
        ModeParam('t_transfer_sec', 'nonneg_number', None),
        availability=frozenset({'nhlorri'}),
        incidence_semantics='per-frame probability the frame-transfer smear is applied',
    ),
    _detector_mode(
        'serial_tail',
        ModeParam('amplitude_dn', 'nonneg_number', 12.0),
        ModeParam('length_px', 'positive_int', 8),
        ModeParam('direction', 'enum', 'right', choices=('right', 'left')),
        ModeParam('saturation_frac', 'unit_interval', 0.99),
        availability=frozenset({'nhlorri'}),
        incidence_semantics='per-frame probability the saturation serial tail is applied',
    ),
    _detector_mode(
        'beam_bend',
        ModeParam('amplitude_px', 'nonneg_number', 1.0),
        availability=frozenset({'vgiss'}),
        incidence_semantics='per-frame probability the brightness-dependent limb bend is applied',
    ),
    _detector_mode(
        'residual_image',
        ModeParam('amplitude', 'unit_interval', 0.05),
        ModeParam('prior', 'enum', 'self_offset', choices=('self_offset', 'flat')),
        ModeParam('offset_px', 'int_list', None, length=2),
        availability=frozenset({'vgiss'}),
        incidence_semantics='per-frame probability the prior-frame ghost is added',
    ),
    _detector_mode(
        'quantization_lut',
        availability=frozenset({'coiss_nac', 'coiss_wac'}),
        incidence_semantics='per-frame probability the sqrt-companding LUT quantization is active',
    ),
    _detector_mode(
        'quantization_ls8b',
        availability=frozenset({'coiss_nac', 'coiss_wac'}),
        incidence_semantics='per-frame probability the LS8B wraparound quantization is active',
    ),
    _detector_mode(
        'contouring_8bit',
        ModeParam('step', 'positive_int', 8),
        availability=frozenset({'gossi', 'vgiss'}),
        incidence_semantics='per-frame probability the 8-bit contouring quantization is active',
    ),
    _detector_mode(
        'fixed_pattern',
        ModeParam('prnu_rms', 'unit_interval', None),
        ModeParam('vignetting_frac', 'unit_interval', None),
        ModeParam('stitch_period_px', 'positive_int', None),
        ModeParam('stitch_amplitude_dn', 'nonneg_number', None),
        ModeParam('jail_bar_dn', 'nonneg_number', None),
        ModeParam('dust_donut_count', 'nonneg_int', None),
        availability=_ALL_INSTRUMENTS,
        incidence_semantics='per-frame probability the static fixed-pattern composite is applied',
    ),
)

# Telemetry-stage artifact modes beyond the structured loss modes: the lossy
# compression blockiness (before structured loss) and the Voyager GEOMED
# archive-processing scars (after structured loss).  ``resample_texture`` is a
# telemetry mode, not a detector mode: it emulates the archive resample the
# pipeline consumes, so it runs after the loss modes rather than in the detector.
_TELEMETRY_ARTIFACT_MODES: tuple[ArtifactMode, ...] = (
    _telemetry_mode(
        'compression_dct',
        ModeParam('scale_factor', 'nonneg_number', 8.0),
        ModeParam('block', 'positive_int', 8),
        availability=_ALL_CCD,
        incidence_semantics='per-frame probability the lossy DCT compression is applied',
    ),
    _telemetry_mode(
        'reseau_scars',
        ModeParam('spacing_px', 'positive_int', 46),
        ModeParam('patch_radius_px', 'positive_int', 4),
        availability=frozenset({'vgiss'}),
        incidence_semantics='per-frame probability the reseau-removal scars are applied',
    ),
    _telemetry_mode(
        'resample_texture',
        ModeParam('warp_amp_px', 'nonneg_number', 0.3),
        ModeParam('blank_border_px', 'nonneg_int', 0),
        ModeParam('missing_line_interp', 'bool', False),
        availability=frozenset({'vgiss'}),
        incidence_semantics='per-frame probability the GEOMED resample texture is applied',
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
# codec compresses, then packets drop); the Voyager GEOMED archive-processing
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
    'quantization_lut',
    'quantization_ls8b',
    'contouring_8bit',
)


def normalize_instrument(instrument: str | None) -> str:
    """Collapse an instrument name to its availability key.

    The calibrated Cassini aliases share the raw detector's availability, and
    the generic aliases (and ``None``) map to ``generic``, which accepts every
    mode.

    Parameters:
        instrument: A sim instrument name, a generic alias, or ``None``.

    Returns:
        The normalized availability key.
    """
    if instrument is None or instrument in ('generic', 'sim'):
        return 'generic'
    aliases = {'coiss_calib_nac': 'coiss_nac', 'coiss_calib_wac': 'coiss_wac'}
    return aliases.get(instrument, instrument)


def mode_available(mode_name: str, instrument: str | None) -> bool:
    """Whether ``mode_name`` is available on ``instrument``.

    Parameters:
        mode_name: A registered mode name.
        instrument: A sim instrument name, a generic alias, or ``None``.

    Returns:
        True if the generic block is selected (which accepts every mode) or the
        instrument is in the mode's availability set.
    """
    norm = normalize_instrument(instrument)
    if norm == 'generic':
        return True
    return norm in ARTIFACT_MODES[mode_name].availability


def mode_unavailable_message(mode_name: str, instrument: str | None) -> str:
    """A validation message explaining why a mode is unavailable on an instrument."""
    norm = normalize_instrument(instrument)
    mode = ARTIFACT_MODES[mode_name]
    bespoke = mode.unavailable_reason.get(norm)
    if bespoke is not None:
        return f'artifact mode {mode_name!r} is {bespoke}'
    available = sorted(mode.availability)
    return (
        f'artifact mode {mode_name!r} is not available for instrument {instrument!r}; '
        f'available on: {available}'
    )


def resolve_mode_config(mode_name: str, raw_config: Mapping[str, Any]) -> dict[str, Any]:
    """Fill a mode's scene map with its parameter defaults for rendering.

    Parameters:
        mode_name: A registered mode name.
        raw_config: The scene's map for the mode (already validated).

    Returns:
        A fresh dict carrying every parameter, scene value overriding default.
    """
    mode = ARTIFACT_MODES[mode_name]
    resolved: dict[str, Any] = {}
    for param in mode.params:
        resolved[param.name] = raw_config.get(param.name, param.default)
    return resolved
