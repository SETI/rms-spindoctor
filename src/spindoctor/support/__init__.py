"""Shared utilities for navigation code.

This package groups small, reusable helpers. Import from submodules directly, for
example ``from spindoctor.support import image`` or
``from spindoctor.support.image import shift_array``.

Modules:

    ``types``
        NumPy typing aliases (e.g. ``NDArrayFloatType``), ``PathLike``, and protocols
        such as ``MutableStar``.
    ``image``
        Two-dimensional array helpers: shifting, padding, cropping, normalization,
        and FFT-related image operations.
    ``cmatrix``
        ``compute_pointing`` and its supporting types: the corrected and uncorrected
        C-matrices a navigated offset implies, in the SPICE camera-frame convention.
    ``command_line``
        ``masked_command_line`` -- the command line as a run log is allowed to
        record it, with the value of every connection-URL option hidden.
    ``correlate``
        Fourier-domain and template-matching utilities (e.g. normalized
        cross-correlation) built on ``image`` and ``misc``.
    ``misc``
        Miscellaneous helpers, including sky-coordinate formatting and oops-backed
        utilities.
    ``time``
        Wall-clock helpers (ISO strings, timezone-aware datetimes), and the one
        conversion of an ET into UTC, in the plain and the PDS4 spelling.
    ``file``
        YAML/JSON serialization helpers and ``clean_obj`` for stripping NumPy scalars
        from nested structures.
    ``constants``
        Common mathematical constants (e.g. ``PI``, ``HALFPI``).
    ``exceptions``
        ``NavContractError`` -- typed exception for internal contract violations.
        ``NavPointingError`` names the failures the corrected-attitude computation
        expects, so a caller can absorb exactly those.
    ``nav_base``
        ``NavBase``, a small base class wiring ``Config`` and ``PdsLogger`` for nav
        objects.
    ``attrdict``
        ``AttrDict``, a ``dict`` subclass that supports attribute-style key access.
    ``flux``
        Legacy flux and filter-convolution experiments; most of the implementation is
        commented out but kept for reference.
    ``filters``
        ``NavFilterSpec`` / ``NavFilterKind`` and the dispatcher ``apply_filter`` used
        across feature extraction and matching techniques.
    ``filter_combo``
        ``canonicalize`` for normalizing multi-filter combos into a stable key.
    ``status_reason``
        ``NavStatusReason`` enum carried on every ``NavResult``.
    ``noise_estimate``
        ``estimate_image_noise_sigma`` -- robust per-image noise estimator.
    ``image_quality``
        ``saturation_mask`` and ``cosmic_ray_mask`` -- global image-quality
        masks consumed by extractors.
    ``distance_transform``
        ``apply_translation`` and ``sample_dt_bilinear`` -- chamfer-matching
        helpers built on top of an externally-computed distance transform.
"""

__all__ = [
    'attrdict',
    'cmatrix',
    'command_line',
    'constants',
    'correlate',
    'distance_transform',
    'exceptions',
    'file',
    'filter_combo',
    'filters',
    'flux',
    'image',
    'image_quality',
    'misc',
    'nav_base',
    'noise_estimate',
    'status_reason',
    'time',
    'types',
]
