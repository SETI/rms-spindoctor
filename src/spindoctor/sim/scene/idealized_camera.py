"""The scene's ``instrument_config`` block: what the scene changes in the navigator block.

A scene's camera is read by the navigator through the camera block its instrument entry
points at (:class:`~spindoctor.config.sections_camera.CameraBlock`).  The scene's
``instrument_config`` block changes keys of that block.  The block is idealized: it is
published knowledge, which crosses into the navigator's view.  Its types are an
all-optional form of the camera block's types, and each value takes the camera block's
own field type, imported from :mod:`spindoctor.config.sections_camera`, so a scene and
the packaged configuration accept the same range for every key.  ``data_units`` is the
one key a scene can't set: a frame's units come from the instrument entry, and the
navigator block must agree with them.

:func:`overlay_camera_block` merges a scene's block over the navigator block.  A mapping
merges key by key, at every depth, so a scene sets only the keys it changes; any other
value replaces the block's.  Two cases have a rule of their own:

- A scene that sets ``source_image_filter.kind`` to a kind other than the block's
  replaces the whole filter block, so it sets every key the new kind reads.  The keys of
  the old kind are not carried over, since the new kind does not read them.
- Each entry of ``mag_offset.mag_offset_table`` is a mapping that sets ``default``, as
  in the camera block, so a bare number is refused when the scene is validated.

The merged block is parsed with the camera block's type, so every rule of that type,
such as which keys a camera's units read, holds for the merged block too.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from spindoctor.config.sections_camera import (
    BinningFactor,
    BlankMaxDn,
    BlankMaxIf,
    CameraBlock,
    ExtfovMargin,
    FitCameraRotation,
    MarkerValue,
    MaxMissingFracClean,
    MaxOverexposedFracClean,
    MaxRotationDeg,
    NoisyThresholdDn,
    NoisyThresholdIf,
    OneStarEllipseFraction,
    PartialDropoutMinFrac,
    ReadNoiseDn,
    SaturationDn,
    SaturationThresholdDn,
    StarFluxDnPerSVmag0,
    StarMaxPeakPerPredictedSnr,
    StarPsfSigma,
    StarPsfSizes,
)
from spindoctor.schema import SCHEMA_CONFIG, Bool, Float, Positive, Str, parse
from spindoctor.sim.scene.declare import optional, required

__all__ = [
    'FilterKind',
    'InstrumentConfig',
    'InstrumentConfigBinnedNoise',
    'InstrumentConfigFilter',
    'InstrumentConfigMagOffset',
    'InstrumentConfigMagOffsetEntry',
    'InstrumentConfigNoise',
    'InstrumentConfigQuality',
    'overlay_camera_block',
]

#: Every kind of source-image filter a camera block takes: the kind of
#: :class:`~spindoctor.config.sections_camera.BandpassFilter` and the kinds of
#: :class:`~spindoctor.config.sections_camera.BlurFilter`.
FilterKind = Literal['BANDPASS_DOG', 'ISOTROPIC_GAUSSIAN', 'GRADIENT_OF_GAUSSIAN', 'MORPH_DILATE']

# What a scene that leaves a key out gets.
_BLOCK_VALUE = "the navigator block's value"


@dataclass(frozen=True, kw_only=True)
class InstrumentConfigBinnedNoise:
    """The values a scene changes in one binned readout mode of the navigator block.

    Parameters:
        read_noise_dn: The read noise of one binned pixel, in DN.
        star_flux_dn_per_s_vmag0: The DN per second that a star of visual magnitude 0
            deposits in this mode.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    read_noise_dn: ReadNoiseDn | None = optional(
        'idealized',
        'The read noise of one binned pixel, in DN. 0 is an ideal detector.',
        when_absent=_BLOCK_VALUE,
    )
    star_flux_dn_per_s_vmag0: StarFluxDnPerSVmag0 | None = optional(
        'idealized',
        'The DN per second that a star of visual magnitude 0 deposits in this readout mode.',
        when_absent=_BLOCK_VALUE,
    )


@dataclass(frozen=True, kw_only=True)
class InstrumentConfigNoise:
    """The values a scene changes in the navigator block's ``noise`` block.

    Parameters:
        marker_value: The value a missing pixel carries.
        saturation_dn: The ADC ceiling, in DN.
        read_noise_dn: The read noise of one unbinned pixel, in DN.
        star_flux_dn_per_s_vmag0: The DN per second that a star of visual magnitude 0
            deposits in an unbinned pixel.
        by_binning: The values of each binned readout mode, keyed by binning factor.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    marker_value: MarkerValue | None = optional(
        'idealized',
        'The value a missing pixel carries: a number, or NaN, written ``.nan`` or ``NaN``.',
        when_absent=_BLOCK_VALUE,
    )
    saturation_dn: SaturationDn | None = optional(
        'idealized',
        'The ADC ceiling, in DN, at and above which a pixel is saturated. Only a camera '
        'whose frames are in raw DN reads it.',
        when_absent=_BLOCK_VALUE,
    )
    read_noise_dn: ReadNoiseDn | None = optional(
        'idealized',
        'The read noise of one unbinned pixel, in DN, which the faint-star limit reads. '
        'Only a camera whose frames are in raw DN reads it.',
        when_absent=_BLOCK_VALUE,
    )
    star_flux_dn_per_s_vmag0: StarFluxDnPerSVmag0 | None = optional(
        'idealized',
        'The DN per second that a star of visual magnitude 0 deposits in an unbinned '
        'pixel, which the faint-star limit reads. Only a camera whose frames are in raw '
        'DN reads it.',
        when_absent=_BLOCK_VALUE,
    )
    by_binning: dict[BinningFactor, InstrumentConfigBinnedNoise] | None = optional(
        'idealized',
        'The values of each binned readout mode, keyed by binning factor. A frame binned '
        'by that factor reads them in place of the unbinned values. Only a camera whose '
        'frames are in raw DN reads it, and only for a binning the camera offers.',
        when_absent=_BLOCK_VALUE,
    )


@dataclass(frozen=True, kw_only=True)
class InstrumentConfigQuality:
    """The limits a scene changes in the navigator block's ``image_quality_thresholds``.

    Parameters:
        max_missing_frac_clean: The largest share of missing pixels a clean image may
            have.
        max_overexposed_frac_clean: The largest share of saturated pixels a clean image
            may have.
        partial_dropout_min_frac: The share of missing pixels above which a clean image
            is flagged partial dropout.
        blank_max_dn: The peak, in DN, below which an image is blank.
        saturation_threshold_dn: The DN at and above which a pixel counts as saturated.
        noisy_threshold_dn: The noise sigma, in DN, above which an image is flagged
            noisy.
        blank_max_if: The peak, in I/F, below which an image is blank.
        noisy_threshold_if: The noise sigma, in I/F, above which an image is flagged
            noisy.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    max_missing_frac_clean: MaxMissingFracClean | None = optional(
        'idealized',
        'The largest share of missing pixels a clean image may have.',
        when_absent=_BLOCK_VALUE,
    )
    max_overexposed_frac_clean: MaxOverexposedFracClean | None = optional(
        'idealized',
        'The largest share of saturated pixels a clean image may have.',
        when_absent=_BLOCK_VALUE,
    )
    partial_dropout_min_frac: PartialDropoutMinFrac | None = optional(
        'idealized',
        'The share of missing pixels above which a clean image is flagged partial dropout.',
        when_absent=_BLOCK_VALUE,
    )
    blank_max_dn: BlankMaxDn | None = optional(
        'idealized',
        'The peak, in DN, below which an image is blank. Only a camera whose frames are '
        'in raw DN reads it.',
        when_absent=_BLOCK_VALUE,
    )
    saturation_threshold_dn: SaturationThresholdDn | None = optional(
        'idealized',
        'The DN at and above which a pixel counts as saturated. Only a camera whose '
        'frames are in raw DN reads it.',
        when_absent=_BLOCK_VALUE,
    )
    noisy_threshold_dn: NoisyThresholdDn | None = optional(
        'idealized',
        'The noise sigma, in DN, above which an image is flagged noisy. Only a camera '
        'whose frames are in raw DN reads it.',
        when_absent=_BLOCK_VALUE,
    )
    blank_max_if: BlankMaxIf | None = optional(
        'idealized',
        'The peak, in I/F, below which an image is blank. Only a camera whose frames are '
        'in calibrated I/F reads it.',
        when_absent=_BLOCK_VALUE,
    )
    noisy_threshold_if: NoisyThresholdIf | None = optional(
        'idealized',
        'The noise sigma, in I/F, above which an image is flagged noisy. Only a camera '
        'whose frames are in calibrated I/F reads it.',
        when_absent=_BLOCK_VALUE,
    )


@dataclass(frozen=True, kw_only=True)
class InstrumentConfigFilter:
    """The keys a scene changes in the navigator block's ``source_image_filter``.

    A scene that sets ``kind`` to the block's kind, or leaves it out, changes keys of the
    block's filter.  A scene that sets another kind replaces the whole filter, so it sets
    ``enabled`` and every sigma the new kind reads.

    Parameters:
        kind: The filter's kind.
        enabled: Whether the filter is applied.
        lo_sigma_px: The sigma, in pixels, of the heavy blur a ``BANDPASS_DOG`` filter
            subtracts.
        hi_sigma_px: The sigma, in pixels, of the light blur a ``BANDPASS_DOG`` filter
            subtracts it from.
        sigma_px: The sigma, in pixels, of every other kind.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    kind: FilterKind | None = optional(
        'idealized',
        "The filter's kind. A kind other than the navigator block's replaces the whole "
        'filter, so the scene then sets ``enabled`` and every sigma the new kind reads.',
        when_absent=_BLOCK_VALUE,
    )
    enabled: Bool | None = optional(
        'idealized',
        'Whether the filter is applied to the frame before navigation.',
        when_absent=_BLOCK_VALUE,
    )
    lo_sigma_px: Positive | None = optional(
        'idealized',
        'The sigma, in pixels, of the heavy blur a ``BANDPASS_DOG`` filter subtracts. It '
        'must be greater than ``hi_sigma_px``.',
        when_absent=_BLOCK_VALUE,
    )
    hi_sigma_px: Positive | None = optional(
        'idealized',
        'The sigma, in pixels, of the light blur a ``BANDPASS_DOG`` filter subtracts the '
        'heavy blur from.',
        when_absent=_BLOCK_VALUE,
    )
    sigma_px: Positive | None = optional(
        'idealized',
        'The sigma, in pixels, on both axes, of an ``ISOTROPIC_GAUSSIAN``, '
        '``GRADIENT_OF_GAUSSIAN``, or ``MORPH_DILATE`` filter.',
        when_absent=_BLOCK_VALUE,
    )


@dataclass(frozen=True, kw_only=True)
class InstrumentConfigMagOffsetEntry:
    """The magnitude offset of one filter combination, as the camera block writes it.

    Parameters:
        default: The offset added to a catalog magnitude to give the camera's magnitude.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    default: Float = required(
        'idealized',
        "The offset added to a catalog magnitude to give the camera's magnitude.",
    )


@dataclass(frozen=True, kw_only=True)
class InstrumentConfigMagOffset:
    """The values a scene changes in the navigator block's ``mag_offset`` block.

    Parameters:
        fallback_combo: The filter combination whose offset the star model applies.
        mag_offset_table: The offset of each filter combination, keyed by combination.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    fallback_combo: Str | None = optional(
        'idealized',
        'The filter combination whose offset the star model applies. The merged table '
        'must have an entry for it.',
        when_absent=_BLOCK_VALUE,
    )
    mag_offset_table: dict[Str, InstrumentConfigMagOffsetEntry] | None = optional(
        'idealized',
        'The offset of each filter combination, keyed by combination. Each entry is a '
        "mapping that sets ``default``, and replaces the navigator block's entry for its "
        'combination.',
        when_absent=_BLOCK_VALUE,
    )


@dataclass(frozen=True, kw_only=True)
class InstrumentConfig:
    """The keys a scene changes in its camera's navigator block.

    Parameters:
        extfov_margin_vu: The margin, in pixels, that the offset search may reach.
        star_psf_sigma: The sigma, in pixels, of the Gaussian star PSF.
        star_psf_sizes: The window a star's PSF is drawn in, keyed by magnitude bin.
        one_star_ellipse_fraction: The fraction of the offset window, per axis, inside
            which a lone star is believed.
        fit_camera_rotation: Whether a technique's fit solves for a camera rotation.
        max_rotation_deg: The largest rotation, in degrees, a fit may find.
        noise: The missing-pixel value, saturation, and star photometry.
        image_quality_thresholds: The limits of the image classes.
        source_image_filter: The filter applied to the frame before navigation.
        mag_offset: The catalog-to-camera magnitude offsets.
        star_max_peak_per_predicted_snr: The largest peak, in noise sigmas per unit of
            predicted SNR, that a lone star can have.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    extfov_margin_vu: ExtfovMargin | None = optional(
        'idealized',
        'The margin, in pixels, ``[v, u]``, that the offset search may reach on each axis: '
        'one pair, or a pair per frame height, keyed by the height.',
        when_absent=_BLOCK_VALUE,
    )
    star_psf_sigma: StarPsfSigma | None = optional(
        'idealized',
        "The sigma, in pixels, of the navigator's Gaussian star PSF.",
        when_absent=_BLOCK_VALUE,
    )
    star_psf_sizes: StarPsfSizes | None = optional(
        'idealized',
        "The window, in pixels, ``[v, u]``, a star's PSF is drawn in, keyed by the upper "
        'magnitude of each bin.',
        when_absent=_BLOCK_VALUE,
    )
    one_star_ellipse_fraction: OneStarEllipseFraction | None = optional(
        'idealized',
        'The fraction of the offset window, per axis, inside which a lone star is believed.',
        when_absent=_BLOCK_VALUE,
    )
    fit_camera_rotation: FitCameraRotation | None = optional(
        'idealized',
        "Whether a technique's fit solves for a camera rotation. The scene's top-level "
        '``fit_camera_rotation`` wins over it.',
        when_absent=_BLOCK_VALUE,
    )
    max_rotation_deg: MaxRotationDeg | None = optional(
        'idealized',
        'The largest rotation, in degrees, a fit may find.',
        when_absent=_BLOCK_VALUE,
    )
    noise: InstrumentConfigNoise | None = optional(
        'idealized',
        "The camera's missing-pixel value, saturation, and star photometry.",
        when_absent="the navigator block's values",
    )
    image_quality_thresholds: InstrumentConfigQuality | None = optional(
        'idealized',
        'The limits that sort a frame into an image class.',
        when_absent="the navigator block's values",
    )
    source_image_filter: InstrumentConfigFilter | None = optional(
        'idealized',
        'The filter applied to the frame before navigation.',
        when_absent="the navigator block's filter",
    )
    mag_offset: InstrumentConfigMagOffset | None = optional(
        'idealized',
        "The offsets between catalog magnitudes and the camera's magnitudes.",
        when_absent="the navigator block's values",
    )
    star_max_peak_per_predicted_snr: StarMaxPeakPerPredictedSnr | None = optional(
        'idealized',
        'The largest peak, in noise sigmas per unit of predicted SNR, that a lone star can have.',
        when_absent=_BLOCK_VALUE,
    )


def _set_values(value: Any) -> Any:
    """Return a mapping without the keys whose value is None, at every depth.

    Parameters:
        value: A mapping, or any other value, which is returned unchanged.

    Returns:
        The mapping without its unset keys.
    """
    if isinstance(value, Mapping):
        return {key: _set_values(item) for key, item in value.items() if item is not None}
    return value


def _merge(base: Mapping[Any, Any], overlay: Mapping[Any, Any]) -> dict[Any, Any]:
    """Return ``base`` with ``overlay`` merged over it, key by key at every depth.

    Parameters:
        base: The mapping merged over.
        overlay: The mapping whose values win.

    Returns:
        A new mapping: a mapping in both merges, and any other value of ``overlay``
        replaces the value of ``base``.
    """
    merged = dict(base)
    for key, value in overlay.items():
        existing = merged.get(key)
        if isinstance(value, Mapping) and isinstance(existing, Mapping):
            merged[key] = _merge(existing, value)
        else:
            merged[key] = value
    return merged


def overlay_camera_block(
    base: CameraBlock, overlay: InstrumentConfig, *, source: str
) -> CameraBlock:
    """Merge a scene's ``instrument_config`` over a navigator block.

    Each mapping merges key by key, and any other value the scene sets replaces the
    block's.  A filter of another kind replaces the block's whole filter.

    Parameters:
        base: The navigator block.
        overlay: The scene's ``instrument_config``.
        source: The label a message gives for a problem with the merged block.

    Returns:
        The merged block.

    Raises:
        ~spindoctor.schema.SchemaError: If the merged block breaks the camera block's
            schema; each line gives ``source`` and the key path in the block.
    """
    raw = _set_values(dataclasses.asdict(base))
    changes = _set_values(dataclasses.asdict(overlay))
    kind = changes.get('source_image_filter', {}).get('kind')
    if kind is not None and kind != base.source_image_filter.kind:
        # The new kind's block replaces the old one whole, since the new kind reads none
        # of the old kind's sigmas (#1237).
        del raw['source_image_filter']
    return parse(CameraBlock, _merge(raw, changes), source=source)
