"""The rules that check a scene's camera and ``detector`` block against the catalog.

The typed scene checks every key on its own and the rules inside one block
(:mod:`spindoctor.sim.scene.truth_detector`).  The rules here read the instrument
catalog as well: the scene's camera must be cataloged, and so must the gain state,
readout mode, filter, and downlink table its ``detector`` block selects.  A key that
the frame's quantization never reads is refused, and so is a downlink table that
decodes above the frame's ADC ceiling.

:func:`spindoctor.sim.scene.validate.parse_scene` runs :func:`check_scene_camera`
after the typed scene is built.
"""

from __future__ import annotations

from spindoctor.sim.instruments.catalog import load_instrument_catalog
from spindoctor.sim.instruments.resolvers import (
    FilterCatalogError,
    cataloged_binnings,
    downlink_table_names,
    resolve_detector_defaults,
    resolve_downlink_table,
    resolve_filter_throughput,
)
from spindoctor.sim.instruments.schema_detector import Detector
from spindoctor.sim.scene.errors import SimSceneValidationError
from spindoctor.sim.scene.fields import DEFAULT_BINNING
from spindoctor.sim.scene.truth import Scene
from spindoctor.sim.scene.truth_detector import DetectorOverride

__all__ = ['check_scene_camera']


def check_scene_camera(scene: Scene, *, source: str) -> None:
    """Check a scene's camera, and its ``detector`` block, against the catalog.

    Parameters:
        scene: The typed scene.
        source: The label that begins each error message.

    Raises:
        SimSceneValidationError: If the catalog has no such camera; if the
            ``detector`` block selects a gain state, readout mode, filter, or
            downlink table the camera does not catalog, or sets a key that the
            frame's quantization never reads; or if the frame's downlink table
            decodes above its ADC ceiling.
    """
    instrument = scene.instrument
    allowed = load_instrument_catalog().allowed_names()
    if instrument not in allowed:
        raise SimSceneValidationError(
            f'{source}: instrument {instrument!r} is not one of {sorted(allowed)}'
        )
    detector = scene.detector or DetectorOverride()
    _check_cataloged(detector, instrument=instrument, source=source)
    defaults = resolve_detector_defaults(instrument, detector.binning or DEFAULT_BINNING)
    _check_gain_state(detector, defaults, instrument=instrument, source=source)
    _check_quantization_keys_used(detector, defaults, source=source)
    _check_downlink_table_ceiling(detector, defaults, instrument=instrument, source=source)


def _check_cataloged(detector: DetectorOverride, *, instrument: str, source: str) -> None:
    """Refuse a readout mode, filter, or downlink table the camera lacks."""
    if detector.binning is not None:
        available = cataloged_binnings(instrument)
        if detector.binning not in available:
            raise SimSceneValidationError(
                f'{source}: detector.binning {detector.binning} is not a cataloged readout '
                f'mode of instrument {instrument!r}; available binnings: {list(available)}'
            )
    if detector.filter is not None:
        try:
            resolve_filter_throughput(instrument, detector.filter)
        except FilterCatalogError as exc:
            raise SimSceneValidationError(f'{source}: detector.filter: {exc}') from exc
    if detector.downlink_table is not None:
        tables = downlink_table_names(instrument)
        if detector.downlink_table not in tables:
            raise SimSceneValidationError(
                f'{source}: detector.downlink_table {detector.downlink_table!r} is not listed '
                f'for instrument {instrument!r}; available tables: {list(tables)}'
            )


def _check_gain_state(
    detector: DetectorOverride, defaults: Detector, *, instrument: str, source: str
) -> None:
    """Refuse a gain state that the selected readout mode does not catalog.

    A readout mode may state its own gain table, and the renderer reads the selected
    mode's, so the state is checked against the table ``defaults`` holds.
    """
    if detector.gain_state is None:
        return
    table = defaults.gain_e_per_dn_by_state or {0: 1.0}
    if detector.gain_state not in table:
        raise SimSceneValidationError(
            f'{source}: detector.gain_state {detector.gain_state} is not cataloged for '
            f'instrument {instrument!r}; available states: {sorted(table)}'
        )


def _check_quantization_keys_used(
    detector: DetectorOverride, defaults: Detector, *, source: str
) -> None:
    """Refuse a detector key that the frame's quantization never reads.

    Only the ``sqrt_lut`` quantization decodes through ``downlink_table``, and only
    ``contour_8bit`` posterizes to ``contour_step``.  The frame's quantization is the
    block's own, else the camera's for the selected readout mode, which ``defaults``
    holds.
    """
    quantization = detector.quantization or defaults.quantization
    keyed = (
        ('downlink_table', detector.downlink_table, 'sqrt_lut'),
        ('contour_step', detector.contour_step, 'contour_8bit'),
    )
    for key, value, reader in keyed:
        if value is not None and quantization != reader:
            raise SimSceneValidationError(
                f'{source}: detector.{key} {value!r} has no effect: only the {reader} '
                f'quantization reads it, and this scene quantizes with {quantization!r}; set '
                f'detector.quantization: {reader}, or drop the key'
            )


def _check_downlink_table_ceiling(
    detector: DetectorOverride, defaults: Detector, *, instrument: str, source: str
) -> None:
    """Refuse a companded frame whose downlink table decodes above the ADC ceiling.

    The rule is the table's own, which the detector stage applies too:
    :meth:`~spindoctor.sim.instruments.schema.DownlinkTable.check_ceiling`.  The
    ceiling is the block's ``saturation_dn``, else the camera's.  Only the ``sqrt_lut``
    quantization decodes through a table.  ``defaults`` holds the camera's values in the
    selected readout mode.
    """
    if (detector.quantization or defaults.quantization) != 'sqrt_lut':
        return
    table = resolve_downlink_table(instrument, detector.downlink_table)
    if table is None:
        return
    saturation_dn = detector.saturation_dn or defaults.saturation_dn
    try:
        table.check_ceiling(saturation_dn)
    except ValueError as exc:
        raise SimSceneValidationError(
            f'{source}: detector.saturation_dn {saturation_dn:g}: {exc}; raise the ceiling '
            f'or choose another downlink table'
        ) from exc
