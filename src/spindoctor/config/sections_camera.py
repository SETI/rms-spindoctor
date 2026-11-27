"""The typed camera blocks of the navigator configuration.

A camera block holds what the navigator knows about one camera: the units its images are
read in, the margin its offset search may reach, its star PSF, its noise and saturation,
the limits that sort an image into a quality class, the filter applied to an image before
navigation, and the magnitude offset between the star catalog and the camera.  Every
camera's block has the same keys and parses into the same type, :class:`CameraBlock`.
Where a key means something only for some cameras, a rule of the type says which, in
terms of another key of the block (``data_units``), never in terms of the instrument.

A camera block is found by the key every block sets, ``data_units``: a top-level section
that sets it is one camera's block, and a section whose entries set it holds several
cameras, keyed by camera name.  :func:`camera_sections` finds them, and
:class:`~spindoctor.config.sections.TypedSections` gives each one a field.

Each field's type is exported as a named alias, such as :data:`ReadNoiseDn`, so a reader
that states a subset of the same keys can type them with the same ranges.

Every field without a default is a key whose value lives in the packaged YAML only.  A
field that defaults to None is a key a camera leaves out when it does not apply.
"""

import dataclasses
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, Any, Literal

from annotated_types import Ge, Gt, Le, MinLen

from spindoctor.schema import (
    SCHEMA_CONFIG,
    Bool,
    DigitInt,
    Float,
    FloatOrNan,
    NonNeg,
    NonNegInt,
    Positive,
    PositiveInt,
    RuleError,
    Str,
    Unit,
    parse,
    tagged,
)

__all__ = [
    'CAMERA_MARKER_KEY',
    'BandpassFilter',
    'BinnedNoise',
    'BinningFactor',
    'BinningTable',
    'BlankMaxDn',
    'BlankMaxIf',
    'BlurFilter',
    'CameraBlock',
    'CameraNoise',
    'DataUnits',
    'ExtfovMargin',
    'ExtfovMarginTable',
    'ExtfovMarginVu',
    'FilterMagOffsetTable',
    'FitCameraRotation',
    'ImageSize',
    'MagOffset',
    'MagOffsetEntry',
    'MarkerValue',
    'MaxMissingFracClean',
    'MaxOverexposedFracClean',
    'MaxRotationDeg',
    'MinDetectableDnTable',
    'NoisyThresholdDn',
    'NoisyThresholdIf',
    'OneStarEllipseFraction',
    'PartialDropoutMinFrac',
    'QualityThresholds',
    'ReadNoiseDn',
    'SaturationDn',
    'SaturationThresholdDn',
    'SearchDepthMag',
    'SourceImageFilter',
    'StarFluxDnPerSVmag0',
    'StarFluxPerSVmag0',
    'StarFluxUnit',
    'StarGainTable',
    'StarMaxPeakPerPredictedSnr',
    'StarPhotometry',
    'StarPhotometryPsfSigmaTable',
    'StarPsfSigma',
    'StarPsfSize',
    'StarPsfSizes',
    'as_camera_block',
    'camera_sections',
]

CAMERA_MARKER_KEY = 'data_units'
"""The key every camera block sets, by which the configuration's camera blocks are found."""

# The simulator's camera block (config_440_sim.yaml) is typed by the simulator under its
# own issue, with this module's type; until then the loader leaves it untyped, and
# as_camera_block reads it where the navigator needs it (#1029).
_UNTYPED_CAMERA_SECTIONS = frozenset({'sim'})

#: The units a camera's images are read in: raw detector counts, or calibrated I/F.
DataUnits = Literal['raw_dn', 'calibrated_if']
#: An image's height in pixels, the key of a margin table.
ImageSize = Annotated[DigitInt, Gt(0)]
#: The margin, in pixels, ``[v, u]``, that the offset search may reach on each axis.
ExtfovMarginVu = tuple[NonNegInt, NonNegInt]
#: The margin of each image height, keyed by the height in pixels.
ExtfovMarginTable = dict[ImageSize, ExtfovMarginVu]
#: The offset search's margin: one margin for every image, or a margin per image height.
ExtfovMargin = ExtfovMarginVu | ExtfovMarginTable
#: The sigma, in pixels, of the Gaussian star PSF.
StarPsfSigma = Positive
#: The size, in pixels, ``[v, u]``, of the window a star's PSF is drawn in.
StarPsfSize = tuple[PositiveInt, PositiveInt]
#: The PSF window of each magnitude bin, keyed by the bin's upper magnitude.
StarPsfSizes = Annotated[dict[DigitInt, StarPsfSize], MinLen(1)]
#: The fraction of the offset window, per axis, inside which a lone star is believed.
OneStarEllipseFraction = Annotated[Positive, Le(1)]
#: The largest peak, in noise sigmas per unit of predicted SNR, that a lone star can have.
StarMaxPeakPerPredictedSnr = Positive
#: Whether a technique's fit solves for a camera rotation as well as a translation.
FitCameraRotation = Bool
#: The largest rotation, in degrees, that a fit may find.
MaxRotationDeg = Positive
#: The ADC ceiling, in DN, at and above which a pixel is saturated.
SaturationDn = Positive
#: The value a missing pixel carries: a number, or NaN for a product that marks a missing
#: pixel with NaN.  NaN is written ``.nan``, or ``NaN``, which YAML reads as a string.
MarkerValue = FloatOrNan | Literal['NaN']
#: The read noise of one pixel, in DN; 0 is an ideal detector.
ReadNoiseDn = NonNeg
#: The DN per second that a star of visual magnitude 0 deposits.
StarFluxDnPerSVmag0 = Positive
#: A readout mode's binning factor; the unbinned values are the noise block's own.
BinningFactor = Annotated[DigitInt, Ge(2)]
#: The peak below which an image is blank, in DN.
BlankMaxDn = NonNeg
#: The DN at and above which a pixel counts as saturated.
SaturationThresholdDn = Positive
#: The noise sigma above which an image is flagged noisy, in DN.
NoisyThresholdDn = Positive
#: The peak below which an image is blank, in I/F.
BlankMaxIf = NonNeg
#: The noise sigma above which an image is flagged noisy, in I/F.
NoisyThresholdIf = Positive
#: The largest share of missing pixels a clean image may have.
MaxMissingFracClean = Unit
#: The largest share of saturated pixels a clean image may have.
MaxOverexposedFracClean = Unit
#: The share of missing pixels above which a clean image is flagged partial dropout.
PartialDropoutMinFrac = Unit
#: What a camera's star rate counts: electrons, or DN for a camera whose calibration
#: works in DN and does not convert to electrons.
StarFluxUnit = Literal['electrons', 'dn']
#: The rate, per second, at which a star of V magnitude 0 deposits signal through the
#: reference filter, in the unit that ``flux_unit`` gives.
StarFluxPerSVmag0 = Positive
#: Each gain state's factor, keyed by the gain state as the instrument reads it: the
#: electrons per DN when the rate counts electrons, or, when it counts DN, the factor by
#: which a star's DN is multiplied against the reference gain state.
StarGainTable = Annotated[dict[Str, Positive], MinLen(1)]
#: How many magnitudes each filter dims a star, against the reference filter, keyed by
#: the filter as the instrument reads it.
FilterMagOffsetTable = Annotated[dict[Str, Float], MinLen(1)]
#: The DN of the dimmest star a frame shows, keyed by the readout mode as the instrument
#: spells it.
MinDetectableDnTable = Annotated[dict[Str, Positive], MinLen(1)]
#: The Gaussian sigma of a star, in the frame's own pixels, keyed by the readout mode as
#: the instrument spells it.
StarPhotometryPsfSigmaTable = Annotated[dict[Str, StarPsfSigma], MinLen(1)]
#: The magnitudes past the dimmest visible star that the catalog search reaches.
SearchDepthMag = NonNeg


@dataclass(frozen=True)
class BinnedNoise:
    """The noise values of one binned readout mode.

    Parameters:
        read_noise_dn: The read noise of one binned pixel, in DN.
        star_flux_dn_per_s_vmag0: The DN per second that a star of visual magnitude 0
            deposits in this mode.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    read_noise_dn: ReadNoiseDn
    star_flux_dn_per_s_vmag0: StarFluxDnPerSVmag0


#: The binned readout modes, keyed by binning factor.
BinningTable = dict[BinningFactor, BinnedNoise]


@dataclass(frozen=True)
class CameraNoise:
    """A camera's ``noise`` block: its missing-pixel value, saturation, and star photometry.

    A camera whose ``data_units`` is ``raw_dn`` sets ``saturation_dn``,
    ``read_noise_dn``, and ``star_flux_dn_per_s_vmag0``, and a ``calibrated_if`` camera
    leaves them and ``by_binning`` out (see :class:`CameraBlock`).

    Parameters:
        marker_value: The value a missing pixel carries, or ``'NaN'`` for NaN (see
            :meth:`marker`).
        saturation_dn: The ADC ceiling, in DN.
        read_noise_dn: The read noise of one unbinned pixel, in DN.
        star_flux_dn_per_s_vmag0: The DN per second that a star of visual magnitude 0
            deposits in an unbinned pixel.
        by_binning: The noise values of each binned readout mode, keyed by binning
            factor.  A binned mode without an entry reads the unbinned values.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    marker_value: MarkerValue
    saturation_dn: SaturationDn | None = None
    read_noise_dn: ReadNoiseDn | None = None
    star_flux_dn_per_s_vmag0: StarFluxDnPerSVmag0 | None = None
    by_binning: BinningTable | None = None

    def marker(self) -> float:
        """Return the value a missing pixel carries, as a number.

        Returns:
            ``marker_value``, with ``'NaN'`` read as NaN.
        """
        return math.nan if self.marker_value == 'NaN' else float(self.marker_value)


@dataclass(frozen=True)
class QualityThresholds:
    """A camera's ``image_quality_thresholds`` block: the limits of the image classes.

    The blank and noise limits are given in the camera's units: a ``raw_dn`` camera sets
    the ``_dn`` keys, and a ``calibrated_if`` camera sets the ``_if`` keys and has no
    saturation limit (see :class:`CameraBlock`).

    Parameters:
        max_missing_frac_clean: The largest share of missing pixels a clean image may
            have.
        max_overexposed_frac_clean: The largest share of saturated pixels a clean image
            may have.
        partial_dropout_min_frac: The share of missing pixels above which a clean image
            is flagged ``partial_dropout``.
        blank_max_dn: The peak, in DN, below which an image is blank.
        saturation_threshold_dn: The DN at and above which a pixel counts as saturated.
        noisy_threshold_dn: The noise sigma, in DN, above which an image is flagged
            noisy.
        blank_max_if: The peak, in I/F, below which an image is blank.
        noisy_threshold_if: The noise sigma, in I/F, above which an image is flagged
            noisy.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    max_missing_frac_clean: MaxMissingFracClean
    max_overexposed_frac_clean: MaxOverexposedFracClean
    partial_dropout_min_frac: PartialDropoutMinFrac
    blank_max_dn: BlankMaxDn | None = None
    saturation_threshold_dn: SaturationThresholdDn | None = None
    noisy_threshold_dn: NoisyThresholdDn | None = None
    blank_max_if: BlankMaxIf | None = None
    noisy_threshold_if: NoisyThresholdIf | None = None


@dataclass(frozen=True)
class BandpassFilter:
    """A difference-of-Gaussians filter applied to an image before navigation.

    Parameters:
        kind: ``BANDPASS_DOG``.
        enabled: Whether the filter is applied.
        lo_sigma_px: The sigma, in pixels, of the heavy blur that is subtracted.
        hi_sigma_px: The sigma, in pixels, of the light blur it is subtracted from.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    kind: Literal['BANDPASS_DOG']
    enabled: Bool
    lo_sigma_px: Positive
    hi_sigma_px: Positive

    def __post_init__(self) -> None:
        """Refuse a heavy blur that is not wider than the light one."""
        if self.lo_sigma_px <= self.hi_sigma_px:
            raise RuleError(
                f'lo_sigma_px ({self.lo_sigma_px}) must be greater than hi_sigma_px '
                f'({self.hi_sigma_px})',
                at=('lo_sigma_px',),
                reads=(('lo_sigma_px',), ('hi_sigma_px',)),
            )


@dataclass(frozen=True)
class BlurFilter:
    """A Gaussian filter of one sigma applied to an image before navigation.

    Parameters:
        kind: ``ISOTROPIC_GAUSSIAN`` for a blur, ``GRADIENT_OF_GAUSSIAN`` for the
            gradient of a blur, or ``MORPH_DILATE`` for a dilation.
        enabled: Whether the filter is applied.
        sigma_px: The sigma, in pixels, on both axes.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    kind: Literal['ISOTROPIC_GAUSSIAN', 'GRADIENT_OF_GAUSSIAN', 'MORPH_DILATE']
    enabled: Bool
    sigma_px: Positive


#: A camera's ``source_image_filter`` block, chosen by its ``kind``.
SourceImageFilter = Annotated[BandpassFilter | BlurFilter, tagged('kind')]


@dataclass(frozen=True)
class MagOffsetEntry:
    """The magnitude offset of one filter combination.

    Parameters:
        default: The offset added to a catalog magnitude to give the camera's magnitude.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    default: Float


@dataclass(frozen=True)
class MagOffset:
    """A camera's ``mag_offset`` block: the catalog-to-camera magnitude offsets.

    Parameters:
        fallback_combo: The filter combination whose offset the star model applies.
        mag_offset_table: The offset of each filter combination, keyed by combination.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    fallback_combo: Str
    mag_offset_table: dict[Str, MagOffsetEntry]

    def __post_init__(self) -> None:
        """Refuse a fallback combination that the table has no entry for."""
        if self.fallback_combo not in self.mag_offset_table:
            raise RuleError(
                f'{self.fallback_combo!r} has no entry in mag_offset_table',
                at=('fallback_combo',),
                reads=(('fallback_combo',), ('mag_offset_table',)),
            )

    def offset(self) -> float:
        """Return the magnitude offset of the fallback combination.

        Returns:
            The table entry's ``default``.
        """
        return self.mag_offset_table[self.fallback_combo].default


@dataclass(frozen=True)
class StarPhotometry:
    """A camera's ``star_photometry`` block: its measured star limit and star width.

    A star of V magnitude ``m`` and the default spectral class deposits
    ``flux_per_s_vmag0 * 10**(-0.4 * (m + filter_mag_offset[filter])) * texp`` in the
    unit that ``flux_unit`` gives, divided by the gain state's electrons per DN when the
    unit is ``electrons``, or multiplied by the gain state's factor when it is ``dn``.
    The dimmest visible star deposits ``min_detectable_dn[mode]`` DN, and the catalog
    search reaches ``search_depth_mag`` past it.  The keys of the gain, filter, and mode tables
    are spelled the way the camera's reader spells the image's gain state, filter, and
    readout mode; a camera with one readout mode gives each mode table one entry.  The
    reader refuses an image when the tables leave out its gain state, filter, or mode.

    Parameters:
        flux_unit: What the rate counts: ``electrons``, or ``dn``.
        flux_per_s_vmag0: The rate per second of a V = 0 star through the reference
            filter, at the reference gain state when the unit is ``dn``.
        gain: Each gain state's electrons per DN, or its DN factor (see above).
        min_detectable_dn: The DN of the dimmest star a frame of each readout mode shows.
        psf_sigma: The Gaussian sigma of a star, in the pixels of a frame of each readout
            mode.
        search_depth_mag: The magnitudes past the dimmest visible star that the catalog
            search reaches.
        filter_mag_offset: How many magnitudes each filter dims a star, against the
            reference filter.  A camera whose reader does not read a filter leaves it out.

    Raises:
        RuleError: If ``psf_sigma`` and ``min_detectable_dn`` give different modes.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    flux_unit: StarFluxUnit
    flux_per_s_vmag0: StarFluxPerSVmag0
    gain: StarGainTable
    min_detectable_dn: MinDetectableDnTable
    psf_sigma: StarPhotometryPsfSigmaTable
    search_depth_mag: SearchDepthMag
    filter_mag_offset: FilterMagOffsetTable | None = None

    def __post_init__(self) -> None:
        """Refuse mode tables that do not give the same modes.

        A frame reads both tables for its readout mode.  If one table gives a mode that
        the other leaves out, the mistake would show only when a frame of that mode
        arrives.
        """
        if set(self.psf_sigma) != set(self.min_detectable_dn):
            raise RuleError(
                f'gives the modes {sorted(self.psf_sigma)}, but min_detectable_dn gives '
                f'{sorted(self.min_detectable_dn)}; each mode table gives the same modes',
                at=('psf_sigma',),
                reads=(('psf_sigma',), ('min_detectable_dn',)),
            )


# The quality limits in each of the units.
_DN_LIMITS = ('blank_max_dn', 'saturation_threshold_dn', 'noisy_threshold_dn')
_IF_LIMITS = ('blank_max_if', 'noisy_threshold_if')
# The noise values a raw_dn camera sets.
_RAW_NOISE = ('saturation_dn', 'read_noise_dn', 'star_flux_dn_per_s_vmag0')
# For each of the units: each block's key, the keys a camera in those units sets in it,
# and the keys it leaves out.
_KEYS_BY_UNITS: dict[str, tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...]] = {
    'raw_dn': (
        ('noise', _RAW_NOISE, ()),
        ('image_quality_thresholds', _DN_LIMITS, _IF_LIMITS),
    ),
    'calibrated_if': (
        ('noise', (), (*_RAW_NOISE, 'by_binning')),
        ('image_quality_thresholds', _IF_LIMITS, _DN_LIMITS),
    ),
}


@dataclass(frozen=True)
class CameraBlock:
    """What the navigator knows about one camera.

    The keys a block sets follow its ``data_units``.  A ``raw_dn`` camera sets the
    saturation, read noise, and zero point of its ``noise`` block, and the DN limits of
    its quality thresholds.  A ``calibrated_if`` camera sets the I/F limits, and leaves
    out every DN value: the same full-well count maps to a different I/F for every
    exposure, filter, and gain, so it has no saturation limit, and its star limit does not
    read noise values.

    Parameters:
        data_units: The units the camera's images are read in.
        extfov_margin_vu: The margin, in pixels, that the offset search may reach on each
            axis: one ``[v, u]`` pair, or a pair per image height.
        star_psf_sigma: The sigma, in pixels, of the Gaussian star PSF.
        star_psf_sizes: The window, in pixels, a star's PSF is drawn in, keyed by the
            upper magnitude of each bin.  A star takes the first bin whose magnitude it is
            brighter than, and a star fainter than every bin takes the last.
        one_star_ellipse_fraction: The fraction of the offset window, per axis, inside
            which a lone star is believed, in (0, 1].
        fit_camera_rotation: Whether a technique's fit solves for a camera rotation.
        max_rotation_deg: The largest rotation, in degrees, a fit may find.
        noise: The missing-pixel value, saturation, and star photometry.
        image_quality_thresholds: The limits of the image classes.
        source_image_filter: The filter applied to an image before navigation.
        mag_offset: The catalog-to-camera magnitude offsets.
        star_max_peak_per_predicted_snr: The largest peak, in noise sigmas per unit of
            predicted SNR, that a lone star can have.  A camera that leaves it out puts
            no bound on a lone star's peak.
        star_photometry: The camera's measured star limit and star width.  A camera
            without measured photometry leaves it out, and a reader that needs it refuses
            the image.  Its star width is a separate key from ``star_psf_sigma``.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    data_units: DataUnits
    extfov_margin_vu: ExtfovMargin
    star_psf_sigma: StarPsfSigma
    star_psf_sizes: StarPsfSizes
    one_star_ellipse_fraction: OneStarEllipseFraction
    fit_camera_rotation: FitCameraRotation
    max_rotation_deg: MaxRotationDeg
    noise: CameraNoise
    image_quality_thresholds: QualityThresholds
    source_image_filter: SourceImageFilter
    mag_offset: MagOffset
    star_max_peak_per_predicted_snr: StarMaxPeakPerPredictedSnr | None = None
    star_photometry: StarPhotometry | None = None

    def __post_init__(self) -> None:
        """Refuse a key the camera's units do not read, and a missing key they do read."""
        units = self.data_units
        for block_key, required, refused in _KEYS_BY_UNITS[units]:
            block = getattr(self, block_key)
            for key in required:
                if getattr(block, key) is None:
                    raise RuleError(
                        f'a camera whose data_units is {units} must set this key',
                        at=(block_key, key),
                        reads=(('data_units',), (block_key, key)),
                    )
            for key in refused:
                if getattr(block, key) is not None:
                    raise RuleError(
                        f'a camera whose data_units is {units} does not read this key; '
                        'leave it out',
                        at=(block_key, key),
                        reads=(('data_units',), (block_key, key)),
                    )

    def extfov_margin_for(self, size_v: int) -> tuple[int, int]:
        """Return the offset search's margin for an image of a given height.

        Parameters:
            size_v: The image's height in pixels.

        Returns:
            The margin, ``(v, u)``, in pixels.

        Raises:
            ValueError: If the margin is given per image height and the table has no
                entry for ``size_v``.
        """
        margin = self.extfov_margin_vu
        if isinstance(margin, tuple):
            return margin
        if size_v not in margin:
            raise ValueError(
                f'extfov_margin_vu has no entry for an image {size_v} pixels high; it has '
                f'entries for {sorted(margin)}'
            )
        return margin[size_v]


def as_camera_block(block: CameraBlock | Mapping[str, Any], *, source: str) -> CameraBlock:
    """Return a camera block as a :class:`CameraBlock`, parsing it if it is a mapping.

    A simulated observation holds its camera block as a mapping, with its scene's
    overrides merged over it, and the simulator's own camera section is read as a
    mapping; this reads either through the same type as every other camera block.  A
    scene's overrides may carry top-level keys of their own, which the block does not
    declare, and those are not read.

    Parameters:
        block: The camera block, typed or as a mapping.
        source: The label a message gives for a problem in a mapping.

    Returns:
        ``block`` itself when it is typed, otherwise the parsed block.

    Raises:
        ~spindoctor.schema.SchemaError: If a mapping breaks the camera block's schema;
            each line gives ``source`` and the key path.
    """
    if isinstance(block, CameraBlock):
        return block
    # The simulator's scene overrides are merged in whole, keys of their own included, and
    # its readers have only ever read the keys they know; its own overlay type decides
    # which keys a scene may set (#1029).
    declared = {item.name for item in dataclasses.fields(CameraBlock)}
    raw = {key: value for key, value in block.items() if key in declared}
    return parse(CameraBlock, raw, source=source)


def camera_sections(
    config: Mapping[str, Any],
    *,
    typed: frozenset[str],
    known: frozenset[str] = frozenset(),
) -> tuple[tuple[str, bool], ...]:
    """Find the top-level sections of a configuration that hold camera blocks.

    A section that sets :data:`CAMERA_MARKER_KEY` is one camera's block.  A section, any
    of whose entries is a mapping that sets it, holds several cameras keyed by camera
    name.  A section the configuration already types for another purpose is never a
    camera section.  A section in ``known`` stays a camera section whatever an override
    has put in it, so a malformed override is checked against the camera type instead of
    escaping it; it holds several cameras unless it sets the marker itself.

    Parameters:
        config: The merged configuration.
        typed: The names of the sections typed for another purpose.
        known: The camera sections of the configuration an override is merged over.

    Returns:
        ``(name, holds_several)`` for each camera section, in the configuration's order.
    """
    found: list[tuple[str, bool]] = []
    for name, section in config.items():
        if name in typed or name in _UNTYPED_CAMERA_SECTIONS or not isinstance(section, Mapping):
            continue
        if CAMERA_MARKER_KEY in section:
            found.append((name, False))
        elif name in known or any(
            isinstance(entry, Mapping) and CAMERA_MARKER_KEY in entry for entry in section.values()
        ):
            found.append((name, True))
    return tuple(found)
