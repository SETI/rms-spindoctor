"""Per-instrument detector, optics, and radiometry values for the forward model.

Every value here comes from the simulated instrument's config file (see
:mod:`spindoctor.sim.instrument_catalog`): the detector chain an
``instrument_defaults`` scene turns on, the readout modes, the star zero point,
the plate scale the background-sky star count uses, the whole-scene PSF kernel,
the residual geometric distortion, the downlink tables, and the artifact modes the
camera offers with their shape defaults.  These resolvers are how the forward
model reads that catalog; each takes a sim instrument name, an alias, or ``None``
for the generic entry, and raises :class:`ValueError` if the catalog doesn't hold
the name.
"""

import copy
from collections.abc import Mapping
from typing import Any

from spindoctor.sim.forward.artifact_modes import ARTIFACT_MODES, MODE_KEYS
from spindoctor.sim.instrument_catalog import (
    GENERIC_INSTRUMENT,
    instrument_entry,
    load_instrument_catalog,
)
from spindoctor.sim.instrument_schema import OFFER_EVERY_MODE

__all__ = [
    'ARTIFACT_MODES',
    'MODE_KEYS',
    'binned_psf_kernel',
    'binning_for_frame_size',
    'cataloged_binnings',
    'downlink_table_names',
    'instrument_default_modes',
    'mode_available',
    'mode_unavailable_message',
    'resolve_calibrated_sky_if',
    'resolve_detector_defaults',
    'resolve_detector_id',
    'resolve_detector_model',
    'resolve_distortion_residual',
    'resolve_downlink_table',
    'resolve_mode_with_catalog',
    'resolve_native_frame_size_px',
    'resolve_psf_kernel',
    'resolve_readout_mode',
    'resolve_sky_pixel_scale_arcsec',
    'resolve_star_flux_zero_point',
    'scene_binning',
]

# The *missing-data* loss modes default to 0 even under ``instrument_defaults`` --
# choosing an instrument selects a signal chain and plants no transmission defects --
# so they are planted only by scenes that configure them explicitly.  Cosmic rays
# are different: radiation on the detector is environment physics, so each CCD
# instrument's detector entry carries a ``cosmic_ray_events_per_pixel_per_sec``
# that is either cohort-measured or an explicit retained zero with its reason.


def resolve_star_flux_zero_point(instrument: str | None) -> tuple[float, str]:
    """Return the star photometric zero point and its unit domain.

    A star's total signal is ``zero_point * 10**(-0.4 * vmag) * exposure_sec`` --
    flux normalization, not a peak-normalized profile (the optics PSF then dictates
    the shape and the peak).

    Parameters:
        instrument: The sim instrument name, an alias, or ``None``.

    Returns:
        A ``(zero_point, domain)`` pair.  ``domain`` is ``'dn'`` for an instrument
        whose config states ``star_flux.dn_per_s_vmag0`` (a vidicon, whose point
        sources are DN) and ``'electrons'`` for one stating
        ``star_flux.e_per_s_vmag0``.  ``zero_point`` is the flux a magnitude-0 star
        deposits per second of exposure in that domain.

    Raises:
        ValueError: If the instrument is not in the catalog.
    """
    star_flux = instrument_entry(instrument)['star_flux']
    if 'dn_per_s_vmag0' in star_flux:
        return star_flux['dn_per_s_vmag0'], 'dn'
    return star_flux['e_per_s_vmag0'], 'electrons'


def resolve_sky_pixel_scale_arcsec(instrument: str | None) -> float:
    """Return the angular pixel scale (arcsec / pixel) for the sky-count FOV area.

    Parameters:
        instrument: The sim instrument name, an alias, or ``None``.

    Returns:
        The plate scale in arcsec per native detector pixel.

    Raises:
        ValueError: If the instrument is not in the catalog.
    """
    scale: float = instrument_entry(instrument)['sky_pixel_scale_arcsec']
    return scale


def resolve_psf_kernel(instrument: str | None) -> dict[str, float] | None:
    """Return the instrument's native whole-scene PSF kernel, if it has one.

    Parameters:
        instrument: The sim instrument name, an alias, or ``None``.

    Returns:
        A fresh ``sigma_v`` / ``sigma_u`` / ``w`` / ``r0`` / ``n`` mapping, radii in
        native detector pixels and ``w`` the wing energy fraction; None for an
        instrument whose config states no ``psf.kernel``.

    Raises:
        ValueError: If the instrument is not in the catalog.
    """
    kernel = instrument_entry(instrument)['psf'].get('kernel')
    return None if kernel is None else dict(kernel)


def resolve_distortion_residual(instrument: str | None) -> dict[str, float] | None:
    """Return the instrument's residual geometric distortion, if it has one.

    The residual is what remains after the navigator applies the instrument's own
    distortion model: a radial polynomial in the normalized field radius (``k1`` on
    rho^2, ``k2`` on rho^4) plus a non-radial wander RMS in native pixels.

    Parameters:
        instrument: The sim instrument name, an alias, or ``None``.

    Returns:
        A fresh ``k1`` / ``k2`` / ``nonradial_rms_px`` mapping; None for an
        instrument whose config states no ``distortion_residual``.

    Raises:
        ValueError: If the instrument is not in the catalog.
    """
    residual = instrument_entry(instrument).get('distortion_residual')
    return None if residual is None else dict(residual)


def resolve_detector_id(instrument: str | None) -> str:
    """Return the physical detector that an instrument's frames come from.

    Two instruments that are products of one camera (a raw and a calibrated
    product) share a detector, so they share its fixed hot-pixel map.

    Parameters:
        instrument: The sim instrument name, an alias, or ``None``.

    Returns:
        The config's ``detector_id``, else the instrument's own catalog name
        (``generic`` for ``None`` and the generic aliases).

    Raises:
        ValueError: If the instrument is not in the catalog.
    """
    catalog = load_instrument_catalog()
    name = catalog.canonical_name(instrument)
    detector_id: str = catalog.entry(name).get('detector_id', name)
    return detector_id


def resolve_calibrated_sky_if(instrument: str | None) -> float:
    """Return the calibrated sky residual, in I/F, of a calibrated frame.

    Parameters:
        instrument: The sim instrument name, an alias, or ``None``.

    Returns:
        The instrument's ``detector.calibrated_sky_if``, else the generic entry's.

    Raises:
        ValueError: If the instrument is not in the catalog.
    """
    own = instrument_entry(instrument)['detector'].get('calibrated_sky_if')
    if own is not None:
        return float(own)
    return float(instrument_entry(GENERIC_INSTRUMENT)['detector']['calibrated_sky_if'])


def downlink_table_names(instrument: str | None) -> tuple[str, ...]:
    """The downlink tables an instrument lists, sorted.

    Parameters:
        instrument: The sim instrument name, an alias, or ``None``.

    Returns:
        The table names its config's ``downlink_tables`` lists; empty for an
        instrument that lists none.

    Raises:
        ValueError: If the instrument is not in the catalog.
    """
    tables = instrument_entry(instrument).get('downlink_tables')
    return () if tables is None else tuple(sorted(tables['tables']))


def resolve_downlink_table(instrument: str | None, name: str | None) -> dict[str, Any] | None:
    """The downlink table the ``sqrt_lut`` quantization decodes a frame through.

    Parameters:
        instrument: The sim instrument name, an alias, or ``None``.
        name: The table a scene's ``detector.downlink_table`` selects, or ``None`` for
            the instrument's default table.

    Returns:
        A fresh copy of the table's entry (``codes``, ``linear_dn``,
        ``quadratic_dn``, and ``whole_dn``, or ``decoded_dn``); None for an
        instrument that lists no tables, which decodes through the plain
        square-root curve.

    Raises:
        ValueError: If the instrument is not in the catalog, or ``name`` is not a
            table it lists.
    """
    tables = instrument_entry(instrument).get('downlink_tables')
    if tables is None:
        if name is None:
            return None
        raise ValueError(f'sim instrument {instrument!r} lists no downlink tables; got {name!r}')
    chosen = tables['default'] if name is None else name
    if chosen not in tables['tables']:
        raise ValueError(
            f'downlink table {chosen!r} is not listed for sim instrument {instrument!r}; '
            f'available tables: {sorted(tables["tables"])}'
        )
    return copy.deepcopy(dict(tables['tables'][chosen]))


def _offers(entry: Mapping[str, Any], mode_name: str) -> bool:
    """Whether a catalog entry offers ``mode_name``."""
    offered = entry['artifact_modes']['offered']
    return bool(offered == OFFER_EVERY_MODE or mode_name in offered)


def mode_available(mode_name: str, instrument: str | None) -> bool:
    """Whether ``mode_name`` is available on ``instrument``.

    Parameters:
        mode_name: A registered mode name.
        instrument: A sim instrument name, an alias, or ``None``.

    Returns:
        True if the instrument's ``artifact_modes.offered`` is ``'all'`` (the
        generic entry) or lists the mode.

    Raises:
        ValueError: If the instrument is not in the catalog.
    """
    return _offers(instrument_entry(instrument), mode_name)


def mode_unavailable_message(mode_name: str, instrument: str | None) -> str:
    """A validation message explaining why a mode is unavailable on an instrument.

    Parameters:
        mode_name: A registered mode name.
        instrument: A sim instrument name, an alias, or ``None``.

    Returns:
        The instrument's own reason from ``artifact_modes.unavailable_reasons`` when
        it states one, else a message listing the instruments that offer the mode.

    Raises:
        ValueError: If the instrument is not in the catalog.
    """
    reasons = instrument_entry(instrument)['artifact_modes'].get('unavailable_reasons') or {}
    if mode_name in reasons:
        return f'artifact mode {mode_name!r} is {reasons[mode_name]}'
    catalog = load_instrument_catalog()
    available = [
        name for name in catalog.instrument_names() if _offers(catalog.entry(name), mode_name)
    ]
    return (
        f'artifact mode {mode_name!r} is not available for instrument {instrument!r}; '
        f'available on: {available}'
    )


def instrument_default_modes(instrument: str | None) -> dict[str, dict[str, Any]]:
    """The detector-stage modes ``instrument_defaults`` turns on for an instrument.

    Parameters:
        instrument: The sim instrument name, an alias, or ``None``.

    Returns:
        A fresh copy of the config's ``artifact_modes.instrument_defaults``: each
        mode's scene map, which the detector resolver uses when the scene has not
        set the mode itself.  Empty for an instrument that turns no mode on.

    Raises:
        ValueError: If the instrument is not in the catalog.
    """
    modes = instrument_entry(instrument)['artifact_modes'].get('instrument_defaults') or {}
    return copy.deepcopy(dict(modes))


def cataloged_binnings(instrument: str | None) -> tuple[int, ...]:
    """The binning factors a camera's readout modes offer, ascending.

    Parameters:
        instrument: The sim instrument name, an alias, or ``None``.

    Returns:
        The factors the config's ``readout_modes`` lists; ``(1,)`` for a camera
        without binned modes.

    Raises:
        ValueError: If the instrument is not in the catalog.
    """
    return tuple(sorted(instrument_entry(instrument)['readout_modes']))


def resolve_readout_mode(instrument: str | None, binning: int) -> dict[str, Any]:
    """The detector-key overrides of one cataloged readout mode.

    A b-by-b block of native pixels reads out as one pixel; the mode overrides the
    detector keys its published characterization changes, and the keys it leaves
    out keep the 1x1 values.

    Parameters:
        instrument: The sim instrument name, an alias, or ``None``.
        binning: A binning factor from :func:`cataloged_binnings`.

    Returns:
        A fresh copy of the mode's overrides; empty for a mode that keeps every
        1x1 value, and for an uncataloged binning (the resolver refuses those).

    Raises:
        ValueError: If the instrument is not in the catalog.
    """
    modes = instrument_entry(instrument)['readout_modes']
    return copy.deepcopy(modes.get(int(binning), {}))


def resolve_native_frame_size_px(instrument: str | None) -> int | None:
    """The native full-frame size, in pixels on a side, of a camera.

    Parameters:
        instrument: The sim instrument name, an alias, or ``None``.

    Returns:
        The config's ``native_frame_size_px``, or None for a camera that states
        none.

    Raises:
        ValueError: If the instrument is not in the catalog.
    """
    native: int | None = instrument_entry(instrument).get('native_frame_size_px')
    return native


def scene_binning(params: Mapping[str, Any]) -> int:
    """The binning factor a scene's ``detector`` block selects, 1 by default.

    Parameters:
        params: The full scene mapping.

    Returns:
        The stated ``detector.binning``, or 1 when the scene leaves it out.
    """
    detector = params.get('detector') or {}
    binning = detector.get('binning') if isinstance(detector, Mapping) else None
    return 1 if binning is None else int(binning)


def binning_for_frame_size(instrument: str | None, size_vu: tuple[int, int]) -> int:
    """The binning factor of a full frame of ``size_vu`` on ``instrument``.

    Parameters:
        instrument: The sim instrument name.
        size_vu: The frame's ``(v, u)`` shape.

    Returns:
        ``native size / frame size`` when the frame is square and that ratio
        is a cataloged factor; 1 for a camera without a native size.

    Raises:
        ValueError: If the camera has a native size and the frame isn't a
            full frame in any cataloged mode.
    """
    native = resolve_native_frame_size_px(instrument)
    if native is None:
        return 1
    size_v, size_u = (int(size_vu[0]), int(size_vu[1]))
    for binning in cataloged_binnings(instrument):
        if size_v == size_u == native // binning:
            return binning
    raise ValueError(
        f'a {size_v}x{size_u} frame is not a full {instrument!r} frame in any cataloged '
        f'readout mode (native {native} px, binnings {list(cataloged_binnings(instrument))})'
    )


def binned_psf_kernel(kernel: Mapping[str, float], binning: int) -> dict[str, float]:
    """A native PSF kernel restated in pixels binned by ``binning``.

    The radii (``sigma_v``, ``sigma_u``, ``r0``) shrink by the factor; the wing
    energy fraction ``w`` and the wing index ``n`` are dimensionless.  The
    binned pixel's own b-by-b box is the render grid's pixel integration, so
    it needs no term here.

    Parameters:
        kernel: The native kernel, radii in native pixels.
        binning: The binning factor.

    Returns:
        A new kernel mapping in binned pixels.
    """
    scaled = dict(kernel)
    for radius in ('sigma_v', 'sigma_u', 'r0'):
        if radius in scaled:
            scaled[radius] = float(scaled[radius]) / binning
    return scaled


def resolve_detector_defaults(instrument: str | None) -> dict[str, Any]:
    """Return the detector-parameter defaults for a sim instrument.

    Parameters:
        instrument: The sim instrument name, an alias, or ``None`` for the generic
            entry.

    Returns:
        A fresh copy of the config's ``detector`` block, plus, under
        ``artifact_modes``, the config's per-mode shape defaults
        (``artifact_modes.shape_defaults``) when it states any.

    Raises:
        ValueError: If the instrument is not in the catalog.
    """
    entry = instrument_entry(instrument)
    defaults: dict[str, Any] = copy.deepcopy(entry['detector'])
    shape_defaults = entry['artifact_modes'].get('shape_defaults')
    if shape_defaults is not None:
        defaults['artifact_modes'] = copy.deepcopy(shape_defaults)
    return defaults


def resolve_detector_model(instrument: str | None, detector: Mapping[str, Any] | None) -> str:
    """The detector model a scene renders with, read as the detector resolver reads it.

    The scene's own ``detector.detector_model`` wins; otherwise the camera's
    default in the scene's readout mode (the mode's values over the native ones).
    A null key reads as absent, as it does everywhere in the detector resolver.

    Parameters:
        instrument: The sim instrument name, an alias, or ``None``.
        detector: The scene ``detector`` block, or None when the scene has none.

    Returns:
        The detector model, one of
        :data:`~spindoctor.sim.forward.artifact_modes.DETECTOR_MODELS`.

    Raises:
        ValueError: If the instrument is not in the catalog.
    """
    block = detector or {}
    model = block.get('detector_model')
    if model is not None:
        return str(model)
    binning = block.get('binning')
    catalog = {
        **resolve_detector_defaults(instrument),
        **resolve_readout_mode(instrument, 1 if binning is None else binning),
    }
    return str(catalog['detector_model'])


def resolve_mode_with_catalog(
    mode_name: str, scene_cfg: Mapping[str, Any], instrument: str | None
) -> dict[str, Any]:
    """Resolve an artifact mode's parameters with the per-instrument catalog.

    The resolution precedence for a mode's shape parameters is scene value, then
    the instrument's shape default (``artifact_modes.shape_defaults`` in its
    config), then the registry default.  A mode's activation key
    (``expected_events``, ``expected_events_per_sec``, ``probability``, or
    ``pixel_fraction``) and the fixed-count alternative ``count`` are never read
    from the shape defaults: a mode activates only when a scene sets one (or, if
    the instrument lists the mode in ``artifact_modes.instrument_defaults``, when
    the detector resolver supplies it under ``instrument_defaults``).

    Parameters:
        mode_name: A registered artifact-mode name.
        scene_cfg: The scene's map for the mode (already validated).
        instrument: The sim instrument name, for the catalog lookup.

    Returns:
        A fresh dict carrying every parameter at its resolved value.
    """
    catalog_modes = resolve_detector_defaults(instrument).get('artifact_modes') or {}
    catalog = catalog_modes.get(mode_name) or {}
    mode = ARTIFACT_MODES[mode_name]
    scene_only = {mode.activation, mode.count_alternative}
    resolved: dict[str, Any] = {}
    for param in mode.params:
        name = param.name
        if name in scene_cfg and scene_cfg[name] is not None:
            resolved[name] = scene_cfg[name]
        elif name not in scene_only and name in catalog:
            resolved[name] = catalog[name]
        else:
            resolved[name] = param.default
    return resolved
