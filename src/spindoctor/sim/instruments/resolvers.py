"""Every typed read of an instrument entry.

Every value here comes from a simulated camera's instrument entry (see
:mod:`spindoctor.sim.instruments.catalog`): the units a frame is written in, the
detector chain an ``instrument_defaults`` scene turns on, the readout modes, the star
zero point, the plate scale the background-sky star count uses, the whole-scene PSF
kernel and its truncation radius, the residual geometric distortion, the downlink
tables, the filter throughputs, and the artifact modes the camera offers with their
shape defaults.  The rest of the simulator reads the catalog through these resolvers,
which return the entry's typed parts (:mod:`spindoctor.sim.instruments.schema`).  The
catalog is shared by every caller, so a resolver whose result holds a list or a dict
returns a fresh copy of it.
Each takes a sim instrument name, an alias, or ``None`` for the generic entry, and
raises :class:`ValueError` if the catalog doesn't hold the name.

**Filters.**  A scene's optional ``detector.filter`` key sets the filter
combination the frame was shot through, written as the product label writes it,
with the elements joined by ``+`` (for example ``'CL1+CL2'`` for a two-wheel
camera).  Each instrument's entry catalogs its combinations under ``filters``,
along with the clear combination its star zero point refers to and the clear
element each polarizer folds onto.  The catalog value is the electrons per second
one native pixel collects from a surface at I/F 1.0, so the detector chain renders
a signal of 1.0 at that rate and the calibrated products divide by the same scale.
If a scene sets no filter, it keeps the nominal scale its detector entry derives
(``signal_full_scale_frac * full_well_e / exposure_ref_sec``).  The throughput is
the only filter-dependent quantity the simulator models: haze, body photometry,
star color, and the PSF render the same for every filter.
"""

import copy
from collections.abc import Mapping
from typing import Any

from spindoctor.sim.artifact_modes import ARTIFACT_MODES
from spindoctor.sim.instruments.catalog import (
    GENERIC_INSTRUMENT,
    instrument_entry,
    load_instrument_catalog,
)
from spindoctor.sim.instruments.schema import (
    DistortionResidual,
    DownlinkTable,
    Filters,
    PsfKernel,
    canonical_filter_elements,
    mode_table_maps,
)
from spindoctor.sim.instruments.schema_detector import Detector, ReadoutMode

__all__ = [
    'FilterCatalogError',
    'binning_for_frame_size',
    'canonical_filter',
    'cataloged_binnings',
    'cataloged_filters',
    'downlink_table_names',
    'filter_star_flux_scale',
    'instrument_default_modes',
    'mode_available',
    'mode_unavailable_message',
    'resolve_calibrated_sky_if',
    'resolve_dark_sky_ceiling_if',
    'resolve_data_units',
    'resolve_detector_defaults',
    'resolve_detector_defaults_native',
    'resolve_detector_id',
    'resolve_detector_model',
    'resolve_distortion_residual',
    'resolve_downlink_table',
    'resolve_filter_throughput',
    'resolve_mode_with_catalog',
    'resolve_native_frame_size_px',
    'resolve_psf_kernel',
    'resolve_psf_truncation_px',
    'resolve_readout_mode',
    'resolve_sky_pixel_scale_arcsec',
    'resolve_star_flux_zero_point',
]

# The *missing-data* loss modes default to 0 even under ``instrument_defaults`` --
# choosing an instrument selects a signal chain and plants no transmission defects --
# so they are planted only by scenes that configure them explicitly.  Cosmic rays
# are different: radiation on the detector is environment physics, so each CCD
# instrument's detector entry carries a ``cosmic_ray_events_per_pixel_per_sec``
# that is either cohort-measured or an explicit retained zero with its reason.


def resolve_data_units(instrument: str | None) -> str:
    """Return the units an instrument's frames are written in.

    Parameters:
        instrument: The sim instrument name, an alias, or ``None``.

    Returns:
        The entry's ``data_units``: ``'raw_dn'`` for detector counts, or
        ``'calibrated_if'`` for calibrated I/F.

    Raises:
        ValueError: If the instrument is not in the catalog.
    """
    return instrument_entry(instrument).data_units


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
    star_flux = instrument_entry(instrument).star_flux
    if star_flux.dn_per_s_vmag0 is not None:
        return star_flux.dn_per_s_vmag0, 'dn'
    # The schema states exactly one of the two zero points.
    assert star_flux.e_per_s_vmag0 is not None
    return star_flux.e_per_s_vmag0, 'electrons'


def resolve_sky_pixel_scale_arcsec(instrument: str | None) -> float:
    """Return the angular pixel scale (arcsec / pixel) for the sky-count FOV area.

    Parameters:
        instrument: The sim instrument name, an alias, or ``None``.

    Returns:
        The plate scale in arcsec per native detector pixel.

    Raises:
        ValueError: If the instrument is not in the catalog.
    """
    return instrument_entry(instrument).sky_pixel_scale_arcsec


def resolve_psf_kernel(instrument: str | None) -> PsfKernel | None:
    """Return the instrument's native whole-scene PSF kernel, if it has one.

    Parameters:
        instrument: The sim instrument name, an alias, or ``None``.

    Returns:
        The kernel, radii in native detector pixels and ``w`` the wing energy
        fraction; None for an instrument whose config states no ``psf.kernel``.

    Raises:
        ValueError: If the instrument is not in the catalog.
    """
    return instrument_entry(instrument).psf.kernel


def resolve_psf_truncation_px(instrument: str | None) -> int:
    """Return the PSF truncation radius (detector px) for an instrument.

    The truncation radius is the window past which the Gaussian core and Moffat
    wing are cut and the kernel renormalized; the far halo beyond it is
    stray-light scope, not kernel scope.

    Parameters:
        instrument: The sim instrument name, an alias, or None for the generic
            entry.

    Returns:
        The instrument entry's ``psf.truncation_px``.

    Raises:
        ValueError: If the instrument is not in the catalog.
    """
    return instrument_entry(instrument).psf.truncation_px


def resolve_distortion_residual(instrument: str | None) -> DistortionResidual | None:
    """Return the instrument's residual geometric distortion, if it has one.

    The residual is what remains after the navigator applies the instrument's own
    distortion model: a radial polynomial in the normalized field radius (``k1`` on
    rho^2, ``k2`` on rho^4) plus a non-radial wander RMS in native pixels.

    Parameters:
        instrument: The sim instrument name, an alias, or ``None``.

    Returns:
        The residual; None for an instrument whose config states no
        ``distortion_residual``.

    Raises:
        ValueError: If the instrument is not in the catalog.
    """
    return instrument_entry(instrument).distortion_residual


def resolve_detector_id(instrument: str | None) -> str:
    """Return the physical detector that an instrument's frames come from.

    Two instruments that are products of one camera (a raw and a calibrated
    product) share a detector, so they share its fixed hot-pixel map.

    Parameters:
        instrument: The sim instrument name, an alias, or ``None``.

    Returns:
        The entry's ``detector_id``, else the instrument's own catalog name
        (``generic`` for ``None`` and the generic aliases).

    Raises:
        ValueError: If the instrument is not in the catalog.
    """
    catalog = load_instrument_catalog()
    name = catalog.canonical_name(instrument)
    detector_id = catalog.entry(name).detector_id
    return name if detector_id is None else detector_id


def resolve_calibrated_sky_if(instrument: str | None) -> float:
    """Return the calibrated sky residual, in I/F, of a calibrated frame.

    Parameters:
        instrument: The sim instrument name, an alias, or ``None``.

    Returns:
        The instrument's ``detector.calibrated_sky_if``, else the generic entry's.

    Raises:
        ValueError: If the instrument is not in the catalog.
    """
    own = instrument_entry(instrument).detector.calibrated_sky_if
    if own is not None:
        return own
    generic = instrument_entry(GENERIC_INSTRUMENT).detector.calibrated_sky_if
    # The catalog refuses a generic entry without a calibrated sky residual.
    assert generic is not None
    return generic


def resolve_dark_sky_ceiling_if(instrument: str | None) -> float | None:
    """Return an instrument's dark-sky ceiling for the realism comparison, in I/F.

    The realism comparison counts a sky patch as dark sky when its mean is at most
    the calibrated sky residual plus this ceiling.  Only an entry for calibrated
    products may set one.

    Parameters:
        instrument: The sim instrument name, an alias, or ``None``.

    Returns:
        The instrument's ``detector.dark_sky_ceiling_if``, or None when the entry
        doesn't set one.

    Raises:
        ValueError: If the instrument is not in the catalog.
    """
    return instrument_entry(instrument).detector.dark_sky_ceiling_if


def downlink_table_names(instrument: str | None) -> tuple[str, ...]:
    """The downlink tables an instrument lists, sorted.

    Parameters:
        instrument: The sim instrument name, an alias, or ``None``.

    Returns:
        The table names its entry's ``downlink_tables`` lists; empty for an
        instrument that lists none.

    Raises:
        ValueError: If the instrument is not in the catalog.
    """
    tables = instrument_entry(instrument).downlink_tables
    return () if tables is None else tuple(sorted(tables.tables))


def resolve_downlink_table(instrument: str | None, name: str | None) -> DownlinkTable | None:
    """The downlink table the ``sqrt_lut`` quantization decodes a frame through.

    Parameters:
        instrument: The sim instrument name, an alias, or ``None``.
        name: The table a scene's ``detector.downlink_table`` selects, or ``None`` for
            the instrument's default table.

    Returns:
        A fresh copy of the table; None for an instrument that lists no tables, which
        decodes through the plain square-root curve.

    Raises:
        ValueError: If the instrument is not in the catalog, or ``name`` is not a
            table it lists.
    """
    tables = instrument_entry(instrument).downlink_tables
    if tables is None:
        if name is None:
            return None
        raise ValueError(f'sim instrument {instrument!r} lists no downlink tables; got {name!r}')
    chosen = tables.default if name is None else name
    if chosen not in tables.tables:
        raise ValueError(
            f'downlink table {chosen!r} is not listed for sim instrument {instrument!r}; '
            f'available tables: {sorted(tables.tables)}'
        )
    return copy.deepcopy(tables.tables[chosen])


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
    return instrument_entry(instrument).artifact_modes.offers(mode_name)


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
    reasons = instrument_entry(instrument).artifact_modes.unavailable_reasons
    if mode_name in reasons:
        return f'artifact mode {mode_name!r} is {reasons[mode_name]}'
    catalog = load_instrument_catalog()
    available = [
        name
        for name in catalog.instrument_names()
        if catalog.entry(name).artifact_modes.offers(mode_name)
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
        Each mode the entry's ``artifact_modes.instrument_defaults`` sets, with its
        map as a scene writes it, which the detector resolver uses when the scene has
        not set the mode itself.  Empty for an instrument that turns no mode on.  The
        dicts, and the lists in them, are fresh.

    Raises:
        ValueError: If the instrument is not in the catalog.
    """
    return mode_table_maps(instrument_entry(instrument).artifact_modes.instrument_defaults)


def cataloged_binnings(instrument: str | None) -> tuple[int, ...]:
    """The binning factors a camera's readout modes offer, ascending.

    Parameters:
        instrument: The sim instrument name, an alias, or ``None``.

    Returns:
        The factors the entry's ``readout_modes`` lists; ``(1,)`` for a camera
        without binned modes.

    Raises:
        ValueError: If the instrument is not in the catalog.
    """
    return tuple(sorted(instrument_entry(instrument).readout_modes))


def resolve_readout_mode(instrument: str | None, binning: int) -> ReadoutMode:
    """The detector-key overrides of one cataloged readout mode.

    A b-by-b block of native pixels reads out as one pixel; the mode overrides the
    detector keys its published characterization changes, and the keys it leaves
    out keep the 1x1 values.

    Parameters:
        instrument: The sim instrument name, an alias, or ``None``.
        binning: A binning factor from :func:`cataloged_binnings`.

    Returns:
        A fresh copy of the mode; a mode that keeps every 1x1 value for an uncataloged
        binning (the resolver refuses those).

    Raises:
        ValueError: If the instrument is not in the catalog.
    """
    modes = instrument_entry(instrument).readout_modes
    return copy.deepcopy(modes.get(int(binning), ReadoutMode()))


def resolve_native_frame_size_px(instrument: str | None) -> int | None:
    """The native full-frame size, in pixels on a side, of a camera.

    Parameters:
        instrument: The sim instrument name, an alias, or ``None``.

    Returns:
        The entry's ``native_frame_size_px``, or None for a camera that states
        none.

    Raises:
        ValueError: If the instrument is not in the catalog.
    """
    return instrument_entry(instrument).native_frame_size_px


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


def resolve_detector_defaults_native(instrument: str | None) -> Detector:
    """Return a sim instrument's native detector values, before a readout mode is merged.

    :func:`resolve_detector_defaults` merges a readout mode over these values.

    Parameters:
        instrument: The sim instrument name, an alias, or ``None`` for the generic
            entry.

    Returns:
        A fresh copy of the instrument entry's ``detector`` block.

    Raises:
        ValueError: If the instrument is not in the catalog.
    """
    return copy.deepcopy(instrument_entry(instrument).detector)


def resolve_detector_defaults(instrument: str | None, binning: int) -> Detector:
    """Return a sim instrument's detector defaults in one readout mode.

    Where the readout mode sets a key, its value replaces the native one; the keys
    it leaves out keep the native values.

    Parameters:
        instrument: The sim instrument name, an alias, or ``None`` for the generic
            entry.
        binning: A binning factor from :func:`cataloged_binnings`.

    Returns:
        :func:`resolve_detector_defaults_native` with :func:`resolve_readout_mode`
        merged over it.

    Raises:
        ValueError: If the instrument is not in the catalog.
    """
    native = resolve_detector_defaults_native(instrument)
    return native.with_readout_mode(resolve_readout_mode(instrument, binning))


def resolve_detector_model(instrument: str | None, detector: Mapping[str, Any] | None) -> str:
    """The detector model a scene renders with, read as the detector resolver reads it.

    The scene's own ``detector.detector_model`` wins; otherwise the camera's
    default in the scene's readout mode (the mode's values over the native ones).

    Parameters:
        instrument: The sim instrument name, an alias, or ``None``.
        detector: The scene ``detector`` block, or None when the scene has none.

    Returns:
        The detector model, one of
        :data:`~spindoctor.sim.artifact_modes.DETECTOR_MODELS`.

    Raises:
        ValueError: If the instrument is not in the catalog.
    """
    block = detector or {}
    model = block.get('detector_model')
    if model is not None:
        return str(model)
    binning = block.get('binning')
    detector_defaults = resolve_detector_defaults(instrument, 1 if binning is None else binning)
    return detector_defaults.detector_model


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
    the detector resolver supplies it under ``instrument_defaults``).  The catalog
    refuses a shape default that sets either key.

    Parameters:
        mode_name: A registered artifact-mode name.
        scene_cfg: The scene's map for the mode (already validated).
        instrument: The sim instrument name, for the catalog lookup.

    Returns:
        A fresh dict carrying every parameter at its resolved value.
    """
    catalog = instrument_entry(instrument).shape_default_map(mode_name)
    resolved: dict[str, Any] = {}
    for param in ARTIFACT_MODES[mode_name].params:
        name = param.name
        if name in scene_cfg:
            resolved[name] = scene_cfg[name]
        elif name in catalog:
            resolved[name] = catalog[name]
        else:
            resolved[name] = param.default
    return resolved


class FilterCatalogError(ValueError):
    """A scene sets a filter its camera does not catalog."""


def _require_filters(instrument: str | None) -> Filters:
    """The ``filters`` section of an instrument's config, refusing one without it."""
    filters = instrument_entry(instrument).filters
    if filters is None:
        catalog = load_instrument_catalog()
        cataloged = [
            name for name in catalog.instrument_names() if catalog.entry(name).filters is not None
        ]
        raise FilterCatalogError(
            f'sim instrument {instrument!r} has no filter table; instruments with one: {cataloged}'
        )
    return filters


def canonical_filter(instrument: str | None, filter_name: str) -> str:
    """Fold a filter combination onto its cataloged spelling.

    Each ``+``-separated element is stripped of surrounding whitespace and
    upper-cased, and a polarizer listed in the instrument's ``filters.polarizer_clear``
    becomes the clear element of its wheel, so ``'p120+grn'`` reads as
    ``'CL1+GRN'`` on a camera whose ``P120`` folds onto ``CL1``.  The catalog
    refuses a filter table or polarizer spelled any other way, since no scene
    spelling could reach it.

    Parameters:
        instrument: The sim instrument name.
        filter_name: The filter combination as the scene writes it.

    Returns:
        The combination as the catalog keys it.

    Raises:
        ValueError: If the instrument is not in the catalog.
    """
    filters = instrument_entry(instrument).filters
    polarizers = {} if filters is None else filters.polarizer_clear
    elements = canonical_filter_elements(filter_name)
    return '+'.join(polarizers.get(element, element) for element in elements)


def cataloged_filters(instrument: str | None) -> tuple[str, ...]:
    """The filter combinations an instrument catalogs, sorted.

    Parameters:
        instrument: The sim instrument name.

    Returns:
        The combinations, as the catalog keys them; empty for an instrument
        without a filter table.

    Raises:
        ValueError: If the instrument is not in the catalog.
    """
    filters = instrument_entry(instrument).filters
    return () if filters is None else tuple(sorted(filters.throughput_e_per_s_per_if))


def resolve_filter_throughput(instrument: str | None, filter_name: str) -> float:
    """Return a filter combination's throughput in e-/s per native pixel at I/F 1.0.

    Parameters:
        instrument: The sim instrument name.
        filter_name: The filter combination, in any spelling
            :func:`canonical_filter` folds onto a catalog key.

    Returns:
        The throughput.

    Raises:
        FilterCatalogError: If the instrument has no filter table, or the
            combination is not in it.
        ValueError: If the instrument is not in the catalog.
    """
    table = _require_filters(instrument).throughput_e_per_s_per_if
    canonical = canonical_filter(instrument, filter_name)
    if canonical not in table:
        raise FilterCatalogError(
            f'filter {filter_name!r} is not cataloged for sim instrument {instrument!r}; '
            f'available filters: {sorted(table)}'
        )
    return table[canonical]


def filter_star_flux_scale(instrument: str | None, filter_name: str | None) -> float:
    """Return the factor a filter applies to the camera's star zero point.

    The zero point refers to the camera's clear combination (``filters.clear``),
    so a scene's filter scales it by that filter's throughput over the clear
    throughput.

    Parameters:
        instrument: The sim instrument name.
        filter_name: The scene's ``detector.filter``, or ``None``.

    Returns:
        1.0 when the scene sets no filter, else the throughput ratio.

    Raises:
        FilterCatalogError: As :func:`resolve_filter_throughput`.
    """
    if filter_name is None:
        return 1.0
    throughput = resolve_filter_throughput(instrument, filter_name)
    clear = _require_filters(instrument).clear
    return throughput / resolve_filter_throughput(instrument, clear)
