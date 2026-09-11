"""Argument-parsing helpers for sd_mosaic and sd_mosaic_display.

Functions:
    add_common_env_args   -- environment / config-file / logging args.
    add_common_output_args -- output directory, prefix, format, overwrite, pass control,
        per-image ``--image-name`` label for reprojection/mosaic metadata.
    add_body_args          -- BodyMosaic-specific args.
    add_ring_args          -- RingMosaic-specific args.
    add_display_args       -- sd_mosaic_display args (stretch, overlays).
"""

import argparse

from spindoctor.cli.logging_args import add_logging_arguments

# Defaults for :func:`add_body_args` (single source of truth for CLI literals).
DEFAULT_LAT_RESOLUTION: float = 0.1
DEFAULT_LON_RESOLUTION: float = 0.1
DEFAULT_EDGE_MARGIN: int = 3
DEFAULT_BODY_ZOOM: int = 1
DEFAULT_RESOLUTION_THRESHOLD: float = 1.0
DEFAULT_COPY_SLOP: int = 0


def add_common_env_args(parser: argparse.ArgumentParser) -> None:
    """Add environment / config-file / logging arguments.

    Parameters:
        parser: Target ``argparse.ArgumentParser`` (or subparser) to extend.

    Side effects:
        Adds three groups (``Environment``, ``Logging``, ``Miscellaneous``) with
        flags ``--config-file`` (appendable), ``--nav-results-root``,
        ``--results-index-db``, the shared logging options, and ``--profile``.
        Where the PDS3 images themselves are read from is named by
        ``--pds3-holdings-root``, which the selected dataset declares.
        Defaults do not
        read the environment implicitly beyond what downstream nav code does when
        these flags are omitted. No files are read at parse time. Does not raise.
    """
    env = parser.add_argument_group('Environment')
    env.add_argument(
        '--config-file',
        action='append',
        default=None,
        help='Config file(s) to override defaults; may be specified multiple times.',
    )
    env.add_argument(
        '--nav-results-root',
        type=str,
        default=None,
        help=(
            'Root directory of sd_offset results. When provided, pre-computed offsets '
            'from _metadata.json files are applied before reprojection. If omitted '
            '(or if an image has no success metadata), uncorrected pointing is used.'
        ),
    )
    env.add_argument(
        '--results-index-db',
        type=str,
        default=None,
        metavar='URL',
        help=(
            'Connection URL of the results index written by sd_results_index (a sqlite: '
            'URL naming a local path, or a postgresql+psycopg: URL naming a server); '
            'overrides the environment.results_index_db configuration variable and '
            "NAV_RESULTS_INDEX_DB. Each image's navigation record is then read as one row "
            'instead of one file, and --nav-results-root names the ingested root the rows are '
            'read under. Pass --results-index-db none to read the files even where an index '
            'is configured.'
        ),
    )
    add_logging_arguments(parser)
    misc = parser.add_argument_group('Miscellaneous')
    misc.add_argument(
        '--profile',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Enable cProfile performance profiling.',
    )


def add_common_output_args(parser: argparse.ArgumentParser) -> None:
    """Add output-directory / format / pass-control arguments.

    Parameters:
        parser: Target ``argparse.ArgumentParser`` (or subparser) to extend.

    Side effects:
        Adds an ``Output`` group with ``--output-dir`` (required), ``--prefix``,
        ``--format`` (``fits`` or ``npz``), ``--overwrite``, ``--skip-reproject``,
        ``--skip-mosaic``, ``--dry-run``, ``--no-write-output-files``, and
        ``--image-name``. Boolean flags use argparse store_true / false patterns as
        defined below. Parsing does not touch the filesystem. Does not raise.
    """
    out = parser.add_argument_group('Output')
    out.add_argument(
        '--output-dir',
        required=True,
        type=str,
        help='Directory for reprojection files and the final mosaic.',
    )
    out.add_argument(
        '--prefix',
        type=str,
        default='',
        help='Optional filename prefix (e.g. "saturn_2005"). Default: empty string.',
    )
    out.add_argument(
        '--format',
        choices=['fits', 'npz'],
        default='fits',
        help='Output file format. Default: fits.',
    )
    out.add_argument(
        '--overwrite',
        action='store_true',
        default=False,
        help='Re-compute and overwrite existing per-image reprojection files.',
    )
    out.add_argument(
        '--skip-reproject',
        action='store_true',
        default=False,
        help=(
            'Skip the reprojection pass; go straight to the mosaic pass using '
            'existing reprojection files. Implies the reprojection files already exist.'
        ),
    )
    out.add_argument(
        '--skip-mosaic',
        action='store_true',
        default=False,
        help='Skip the mosaic-building pass; only produce per-image reprojection files.',
    )
    out.add_argument(
        '--dry-run',
        action='store_true',
        default=False,
        help='Print what would be done without writing any files.',
    )
    out.add_argument(
        '--no-write-output-files',
        action='store_true',
        default=False,
        help='Do not write any output files (useful for testing).',
    )
    out.add_argument(
        '--image-name',
        type=str,
        default=None,
        metavar='LABEL',
        help=(
            'Label stored in each reprojection file and in the mosaic contributing-image list. '
            'When omitted, the dataset image file stem is used for each image.'
        ),
    )


def add_body_args(parser: argparse.ArgumentParser) -> None:
    """Add BodyMosaic-specific arguments.

    Parameters:
        parser: Target ``argparse.ArgumentParser`` (or subparser) to extend.

    Side effects:
        Adds a ``Body reprojection`` group covering body name, lat/lon grid,
        incidence/emission/resolution limits, margins, integer ``--zoom``,
        photometric model, dtypes, merge-related thresholds, and ``--copy-slop``.
        Defaults use module constants (e.g. :data:`DEFAULT_LAT_RESOLUTION`). Does
        not raise during registration.
    """
    grp = parser.add_argument_group('Body reprojection')
    grp.add_argument(
        '--body-name',
        required=True,
        type=str,
        help='Body to reproject (e.g. MIMAS, TITAN). Case-insensitive; stored as upper-case.',
    )
    grp.add_argument(
        '--lat-resolution',
        type=float,
        default=DEFAULT_LAT_RESOLUTION,
        metavar='DEG',
        help=f'Latitude resolution in degrees/pixel. Default: {DEFAULT_LAT_RESOLUTION}.',
    )
    grp.add_argument(
        '--lon-resolution',
        type=float,
        default=DEFAULT_LON_RESOLUTION,
        metavar='DEG',
        help=f'Longitude resolution in degrees/pixel. Default: {DEFAULT_LON_RESOLUTION}.',
    )
    grp.add_argument(
        '--lat-range',
        type=float,
        nargs=2,
        default=None,
        metavar=('MIN_DEG', 'MAX_DEG'),
        help='Latitude extent (deg). Default: full valid range.',
    )
    grp.add_argument(
        '--lon-range',
        type=float,
        nargs=2,
        default=None,
        metavar=('MIN_DEG', 'MAX_DEG'),
        help='Longitude extent (deg). Default: full valid range.',
    )
    grp.add_argument(
        '--max-incidence',
        type=float,
        default=None,
        metavar='DEG',
        help='Maximum incidence angle (deg) for valid pixels. Default: no limit.',
    )
    grp.add_argument(
        '--max-emission',
        type=float,
        default=None,
        metavar='DEG',
        help='Maximum emission angle (deg) for valid pixels. Default: no limit.',
    )
    grp.add_argument(
        '--max-resolution',
        type=float,
        default=None,
        metavar='KM',
        help='Maximum resolution (km/pixel) for valid pixels. Default: no limit.',
    )
    grp.add_argument(
        '--edge-margin',
        type=int,
        default=DEFAULT_EDGE_MARGIN,
        help=f'Number of edge pixels to discard. Default: {DEFAULT_EDGE_MARGIN}.',
    )
    grp.add_argument(
        '--zoom',
        type=int,
        default=DEFAULT_BODY_ZOOM,
        help=(
            'Zoom factor for sub-pixel interpolation during reprojection (integer). '
            f'Default: {DEFAULT_BODY_ZOOM}.'
        ),
    )
    grp.add_argument(
        '--latlon-type',
        choices=['centric', 'graphic', 'squashed'],
        default='centric',
        help='Coordinate system for latitudes and longitudes. Default: centric.',
    )
    grp.add_argument(
        '--lon-direction',
        choices=['east', 'west'],
        default='east',
        help='Longitude direction convention. Default: east.',
    )
    grp.add_argument(
        '--photometric-model',
        choices=['none', 'lambert', 'lommel-seeliger', 'minnaert'],
        default='none',
        help='Photometric correction to apply. Default: none (raw brightness).',
    )
    grp.add_argument(
        '--no-dynamic',
        dest='dynamic',
        action='store_false',
        default=True,
        help='Disable dynamic mosaic growth; data outside lat/lon-range is clipped.',
    )
    grp.add_argument(
        '--image-dtype',
        type=str,
        default='float64',
        help='NumPy dtype for reprojected brightness arrays. Default: float64.',
    )
    grp.add_argument(
        '--metadata-dtype',
        type=str,
        default='float32',
        help='NumPy dtype for geometry metadata arrays. Default: float32.',
    )
    grp.add_argument(
        '--resolution-threshold',
        type=float,
        default=DEFAULT_RESOLUTION_THRESHOLD,
        help=(
            'Effective-resolution improvement factor required to overwrite a pixel. '
            f'Default: {DEFAULT_RESOLUTION_THRESHOLD}.'
        ),
    )
    grp.add_argument(
        '--copy-slop',
        type=int,
        default=DEFAULT_COPY_SLOP,
        help=(
            'Extra pixels around each copied pixel to reduce isolated-pixel artefacts. '
            f'Default: {DEFAULT_COPY_SLOP}.'
        ),
    )


def add_ring_args(parser: argparse.ArgumentParser) -> None:
    """Add RingMosaic-specific arguments.

    Parameters:
        parser: Target ``argparse.ArgumentParser`` (or subparser) to extend.

    Side effects:
        Adds a ``Ring reprojection`` group with planet/radius bounds, resolutions,
        merge strategy, orbit model, margins, string ``--zoom`` (see note below),
        shadow handling, optional lon/radius ranges, dtypes, and photometric model.
        Does not raise during registration.

    Note:
        ``--zoom`` is registered as ``type=str`` (default ``'1'``) so values may be
        either a single integer string or ``\"R,L\"`` for separate radial and
        longitudinal zoom. :func:`add_body_args` registers body ``--zoom`` as
        ``type=int`` instead, because body reprojection only supports a single scalar
        zoom factor.
    """
    grp = parser.add_argument_group('Ring reprojection')
    grp.add_argument(
        '--planet',
        required=True,
        type=str,
        help='Planet name (e.g. SATURN). Case-insensitive; stored as upper-case.',
    )
    grp.add_argument(
        '--radius-inner',
        required=False,
        default=None,
        type=float,
        metavar='KM',
        help='Inner radius of the mosaic (km). Required when --orbit-model is none.',
    )
    grp.add_argument(
        '--radius-outer',
        required=False,
        default=None,
        type=float,
        metavar='KM',
        help='Outer radius of the mosaic (km). Required when --orbit-model is none.',
    )
    grp.add_argument(
        '--radius-inner-offset',
        required=False,
        default=None,
        type=float,
        metavar='KM',
        help=(
            'Inner-radius offset (km) from the orbit model radius at each '
            '(longitude, time). For an eccentric orbit that radius varies '
            'between a*(1-e) and a*(1+e) with longitude; using offsets makes '
            'the eccentric ring appear as a straight line in the reprojection. '
            'Typically negative (e.g. -1000). Required when --orbit-model is '
            'not none; must not be used otherwise.'
        ),
    )
    grp.add_argument(
        '--radius-outer-offset',
        required=False,
        default=None,
        type=float,
        metavar='KM',
        help=(
            'Outer-radius offset (km) from the orbit model radius at each '
            '(longitude, time). See --radius-inner-offset for details. '
            'Required when --orbit-model is not none; must not be used '
            'otherwise.'
        ),
    )
    grp.add_argument(
        '--longitude-resolution',
        type=float,
        default=0.02,
        metavar='DEG',
        help='Longitude resolution in degrees/pixel. Default: 0.02.',
    )
    grp.add_argument(
        '--radius-resolution',
        type=float,
        default=5.0,
        metavar='KM',
        help='Radius resolution in km/pixel. Default: 5.0.',
    )
    grp.add_argument(
        '--merge-strategy',
        choices=['best_resolution', 'most_coverage_then_resolution'],
        default='most_coverage_then_resolution',
        help='Conflict-resolution strategy. Default: most_coverage_then_resolution.',
    )
    grp.add_argument(
        '--orbit-model',
        choices=['none', 'f_ring_core_albers_2007', 'bring_outer_edge'],
        default='none',
        help=(
            'Ring orbit model for co-rotating longitude conversion. '
            'When "none" (the default), longitudes are inertial (J2000-aligned) '
            'ring longitudes measured eastward from the ascending node. '
            'When an orbit model is specified, longitudes are converted to the '
            'co-rotating frame of that model before binning, and '
            '--radius-inner-offset / --radius-outer-offset (rather than '
            '--radius-inner / --radius-outer) supply the radial bounds as '
            'signed offsets from the orbital radius at each (longitude, time). '
            'Choices: none, f_ring_core_albers_2007 (Albers et al. 2012 Table 3 '
            'Fit #2; epoch 2007-01-01T00:00:00Z), bring_outer_edge. Default: none.'
        ),
    )
    grp.add_argument(
        '--margin',
        type=int,
        default=3,
        help='Number of edge pixels to exclude during reprojection. Default: 3.',
    )
    # String type: accepts "N" or "R,L" (parsed by spindoctor.cli.reproj.factories.parse_zoom_arg).
    # Contrast add_body_args, where --zoom is type=int for a single scalar factor.
    grp.add_argument(
        '--zoom',
        type=str,
        default='1',
        metavar='N or R,L',
        help=(
            'Zoom factor for sub-pixel interpolation. An integer applies uniformly; '
            '"R,L" sets separate radial and longitudinal zoom factors. Default: 1.'
        ),
    )
    grp.add_argument(
        '--no-omit-shadow',
        dest='omit_shadow',
        action='store_false',
        default=True,
        help='Include pixels inside the planet shadow (default: shadow pixels are masked).',
    )
    grp.add_argument(
        '--longitude-range',
        type=float,
        nargs=2,
        default=None,
        metavar=('START_DEG', 'END_DEG'),
        help=(
            'Longitude range to reproject (deg). With --orbit-model none these are '
            'inertial; with an orbit model they are co-rotating. Default: full 0..360.'
        ),
    )
    grp.add_argument(
        '--radius-range',
        type=float,
        nargs=2,
        default=None,
        metavar=('INNER_KM', 'OUTER_KM'),
        help=(
            'Radius range to reproject (km). With --orbit-model none these are '
            'absolute ring radii; with an orbit model they are signed offsets from '
            'the orbit model radius at each (longitude, time) (same convention as '
            '--radius-inner-offset / --radius-outer-offset). Default: mosaic '
            'radius bounds.'
        ),
    )
    grp.add_argument(
        '--image-dtype',
        type=str,
        default='float64',
        help='NumPy dtype for reprojected brightness arrays. Default: float64.',
    )
    grp.add_argument(
        '--metadata-dtype',
        type=str,
        default='float32',
        help='NumPy dtype for geometry metadata arrays. Default: float32.',
    )
    grp.add_argument(
        '--photometric-model',
        choices=['none', 'lambert', 'lommel-seeliger', 'minnaert'],
        default='none',
        help=(
            'Photometric correction applied during ring reproject(). '
            'Default: none (raw brightness).'
        ),
    )


def add_display_args(parser: argparse.ArgumentParser) -> None:
    """Add sd_mosaic_display arguments (stretch, overlays, verbosity).

    Parameters:
        parser: Target ``argparse.ArgumentParser`` (or subparser) to extend.

    Side effects:
        Adds a ``Display`` group with stretch black/white/gamma defaults and
        optional body/ring overlay toggles plus ``--verbose``. Defaults match the
        viewer's auto-stretch behaviour when black/white are omitted. Does not raise.
    """
    disp = parser.add_argument_group('Display')
    disp.add_argument(
        '--stretch-black',
        type=float,
        default=None,
        help='Initial black-point for image stretch. Default: auto (data minimum).',
    )
    disp.add_argument(
        '--stretch-white',
        type=float,
        default=None,
        help='Initial white-point for image stretch. Default: auto (data maximum).',
    )
    disp.add_argument(
        '--stretch-gamma',
        type=float,
        default=0.5,
        help='Initial gamma for image stretch. Default: 0.5.',
    )
    disp.add_argument(
        '--show-radii',
        action='store_true',
        default=False,
        help='(Rings) Overlay radius tick marks on the display.',
    )
    disp.add_argument(
        '--show-parallels',
        action='store_true',
        default=False,
        help='(Bodies) Overlay latitude parallel lines on the display.',
    )
    disp.add_argument(
        '--show-meridians',
        action='store_true',
        default=False,
        help='(Bodies) Overlay longitude meridian lines on the display.',
    )
    disp.add_argument(
        '--verbose',
        action='store_true',
        default=False,
        help='Print additional diagnostic output.',
    )
