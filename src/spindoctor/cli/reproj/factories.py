"""Factory functions that build BodyMosaic / RingMosaic from parsed CLI args."""

import argparse
import math

import numpy as np

from spindoctor.reproj.bodies import BodyMosaic, BodyMosaicMergeStrategy
from spindoctor.reproj.photometric_model import (
    LambertModel,
    LommelSeeligerModel,
    MinnaertModel,
    PhotometricModel,
)
from spindoctor.reproj.ring_orbit_model import BRING_OUTER_EDGE, FRING_CORE, RingOrbitModel
from spindoctor.reproj.rings import RingMosaic, RingMosaicMergeStrategy


def _deg_to_rad(deg: float) -> float:
    """Convert degrees to radians.

    Parameters:
        deg: Angle in degrees.

    Returns:
        The same angle in radians.
    """
    return math.radians(deg)


def _parse_photometric_model(name: str) -> PhotometricModel | None:
    """Return a photometric model instance for CLI name ``name``.

    Parameters:
        name: One of ``none``, ``lambert``, ``lommel-seeliger``, ``minnaert``.

    Returns:
        ``None`` for ``none``, else a fresh model instance.

    Raises:
        ValueError: If ``name`` is not recognized.
    """
    if name == 'none':
        return None
    if name == 'lambert':
        return LambertModel()
    if name == 'lommel-seeliger':
        return LommelSeeligerModel()
    if name == 'minnaert':
        return MinnaertModel()
    raise ValueError(f'Unknown photometric model: {name!r}')


def build_body_mosaic(args: argparse.Namespace) -> BodyMosaic:
    """Construct a BodyMosaic from parsed CLI arguments.

    Parameters:
        args: Namespace produced by a parser that includes ``add_body_args``.

    Returns:
        A freshly-initialized ``BodyMosaic``.

    Raises:
        ValueError: If ``args.photometric_model`` is not one of the supported names
            (see :func:`_parse_photometric_model`).
    """
    phot_model = _parse_photometric_model(str(args.photometric_model))

    lat_range = None
    if args.lat_range is not None:
        lat_range = (
            _deg_to_rad(float(args.lat_range[0])),
            _deg_to_rad(float(args.lat_range[1])),
        )

    lon_range = None
    if args.lon_range is not None:
        lon_range = (
            _deg_to_rad(float(args.lon_range[0])),
            _deg_to_rad(float(args.lon_range[1])),
        )

    return BodyMosaic(
        body_name=args.body_name.upper(),
        lat_resolution=_deg_to_rad(float(args.lat_resolution)),
        lon_resolution=_deg_to_rad(float(args.lon_resolution)),
        lat_range=lat_range,
        lon_range=lon_range,
        dynamic=args.dynamic,
        max_incidence=(
            _deg_to_rad(float(args.max_incidence)) if args.max_incidence is not None else None
        ),
        max_emission=(
            _deg_to_rad(float(args.max_emission)) if args.max_emission is not None else None
        ),
        max_resolution=float(args.max_resolution) if args.max_resolution is not None else None,
        edge_margin=int(args.edge_margin),
        zoom=int(args.zoom),
        latlon_type=args.latlon_type,
        lon_direction=args.lon_direction,
        photometric_model=phot_model,
        merge_strategy=BodyMosaicMergeStrategy.BEST_RESOLUTION,
        image_dtype=np.dtype(args.image_dtype),
        metadata_dtype=np.dtype(args.metadata_dtype),
    )


def build_ring_mosaic(args: argparse.Namespace) -> RingMosaic:
    """Construct a RingMosaic from parsed CLI arguments.

    Parameters:
        args: Namespace produced by a parser that includes ``add_ring_args``.

    Returns:
        A freshly-initialized ``RingMosaic``.

    Raises:
        ValueError: If ``args.orbit_model`` is not a supported name (validated before
            constructing the mosaic). If ``args.merge_strategy`` is not one of the
            supported strategies. If ``args.photometric_model`` is not recognized
            (see :func:`_parse_photometric_model`).
    """
    orbit_model_name: str = args.orbit_model
    orbit_model: RingOrbitModel | None
    if orbit_model_name == 'none':
        orbit_model = None
    elif orbit_model_name == 'f_ring_core_albers_2007':
        orbit_model = FRING_CORE
    elif orbit_model_name == 'bring_outer_edge':
        orbit_model = BRING_OUTER_EDGE
    else:
        raise ValueError(f'Unknown orbit model: {orbit_model_name!r}')

    if orbit_model is None:
        if args.radius_inner is None or args.radius_outer is None:
            raise ValueError(
                '--radius-inner and --radius-outer are required when --orbit-model is none'
            )
        if (
            getattr(args, 'radius_inner_offset', None) is not None
            or getattr(args, 'radius_outer_offset', None) is not None
        ):
            raise ValueError(
                '--radius-inner-offset and --radius-outer-offset must not be used '
                'when --orbit-model is none'
            )
        radius_inner = float(args.radius_inner)
        radius_outer = float(args.radius_outer)
    else:
        if (
            getattr(args, 'radius_inner_offset', None) is None
            or getattr(args, 'radius_outer_offset', None) is None
        ):
            raise ValueError(
                '--radius-inner-offset and --radius-outer-offset are required '
                'when --orbit-model is specified'
            )
        if args.radius_inner is not None or args.radius_outer is not None:
            raise ValueError(
                '--radius-inner and --radius-outer must not be used when --orbit-model is '
                'specified; use --radius-inner-offset and --radius-outer-offset instead'
            )
        # RingMosaic stores offsets directly when an orbit model is set.
        radius_inner = float(args.radius_inner_offset)
        radius_outer = float(args.radius_outer_offset)

    merge_strategy_name: str = args.merge_strategy
    if merge_strategy_name == 'best_resolution':
        merge_strategy = RingMosaicMergeStrategy.BEST_RESOLUTION
    elif merge_strategy_name == 'most_coverage_then_resolution':
        merge_strategy = RingMosaicMergeStrategy.MOST_COVERAGE_THEN_RESOLUTION
    else:
        raise ValueError(f'Unknown merge strategy: {merge_strategy_name!r}')

    phot_model = _parse_photometric_model(str(args.photometric_model))

    return RingMosaic(
        body_name=args.planet.upper(),
        radius_inner=radius_inner,
        radius_outer=radius_outer,
        longitude_resolution=_deg_to_rad(float(args.longitude_resolution)),
        radius_resolution=float(args.radius_resolution),
        merge_strategy=merge_strategy,
        orbit_model=orbit_model,
        image_dtype=np.dtype(args.image_dtype),
        metadata_dtype=np.dtype(args.metadata_dtype),
        photometric_model=phot_model,
    )


def parse_zoom_arg(zoom_str: str) -> int | tuple[int, int]:
    """Parse the ``--zoom`` argument, which may be ``"N"`` or ``"R,L"``.

    Parameters:
        zoom_str: String from the CLI ``--zoom`` argument.

    Returns:
        An integer, or a ``(radial, longitudinal)`` tuple of integers.

    Raises:
        ValueError: If the string cannot be parsed as a valid zoom specification.
    """
    zoom_str = zoom_str.strip()
    if ',' in zoom_str:
        parts = zoom_str.split(',')
        if len(parts) != 2:
            raise ValueError(f'--zoom: expected "N" or "R,L", got {zoom_str!r}')
        try:
            r = int(parts[0].strip())
            lon_z = int(parts[1].strip())
        except (TypeError, ValueError) as exc:
            raise ValueError(f'--zoom: expected "N" or "R,L", got {zoom_str!r}') from exc
        return (r, lon_z)
    try:
        return int(zoom_str)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'--zoom: expected "N" or "R,L", got {zoom_str!r}') from exc
