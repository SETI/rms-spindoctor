"""Field helpers every reader of a sim scene shares.

The renderer, the validator, the scene editor, and the navigator-side simulated
models all read scene keys through these helpers, so they agree on what a key
means: :func:`scene_value` reads an optional key and treats an explicit null as
unset, :func:`scene_binning` reads the readout binning, and
:func:`default_body_name` and :func:`default_body_range_km` give the name and range
a body takes when its entry leaves them out.  The tables here say which shape keys
each ring feature kind takes (:data:`RING_KIND_SHAPE_KEYS`) and which side an edge
or ramp fills by default (:data:`RING_SIDE_DEFAULTS`), which body keys only one
shape model consumes (:data:`MESH_UNSUPPORTED_KEYS`,
:data:`ELLIPSOID_UNSUPPORTED_KEYS`) with the sentences that refuse them, and what
each ring-plane length and angle key holds (:data:`RING_KEY_DESCRIPTIONS`).

The module imports nothing from the simulator, so navigator-side code may import
it; :mod:`spindoctor.sim` explains the rule.
"""

from collections.abc import Iterable, Mapping
from typing import Any

# Body keys the polyhedral-mesh render path does not consume: the ellipsoid
# renderer's crater carving, photometric laws, and
# surface texture, the ellipsoid-frame atmosphere halo (whose geometry is
# built from rotation_z / sub_observer_lat_deg around the ellipse, which a mesh's
# arbitrary Euler pose and lumpy limb do not follow), and the ellipsoid's
# sub-observer latitude (a mesh body is turned by its Euler pose instead).  A mesh body giving one
# of these a value would silently render without it, so the validator refuses
# a non-null value of any of them (a null reads as an absent key); a mesh
# body's appearance vocabulary is shading, limb_relief_*,
# mesh_detail_octaves, and pose_scatter.  The sim editor reads this table too:
# a mesh body disables every control that writes one of these keys.
MESH_UNSUPPORTED_KEYS: tuple[str, ...] = (
    'atmosphere',
    'photometric_law',
    'minnaert_k',
    'opposition_surge',
    'albedo_texture',
    'disc_texture',
    'transits',
    'crater_fill',
    'crater_min_radius',
    'crater_max_radius',
    'crater_power_law_exponent',
    'crater_relief_scale',
    'sub_observer_lat_deg',
)


def mesh_unsupported_message(keys: Iterable[str]) -> str:
    """The sentence that refuses ellipsoid-only appearance keys on a mesh body.

    The scene validator raises it, prefixed with the scene source and the body's
    label, when a ``polyhedral_mesh`` body gives keys from
    :data:`MESH_UNSUPPORTED_KEYS` a non-null value; the sim editor shows it as the
    tooltip of each control it disables on a mesh body, and of every control
    inside one, so the two say the same thing.

    Parameters:
        keys: The refused keys to name; the sentence lists them sorted.

    Returns:
        The sentence, such as ``"['minnaert_k', 'photometric_law'] are not
        supported on polyhedral_mesh bodies (...)"``.
    """
    return (
        f'{sorted(keys)} are not supported on polyhedral_mesh bodies (the mesh '
        f'renderer would silently ignore them); a mesh body carries the shading, '
        f'limb_relief_*, mesh_detail_octaves, and pose_scatter appearance keys'
    )


# Body keys only the polyhedral-mesh render path consumes: the rasterizer's
# smooth-shading mode, the higher-degree relief bands added to the mesh, and the
# per-frame pose error.  The ellipsoid renderer never reads them, so the
# validator refuses a non-null value of any of them on an ellipsoid body (a body
# without a shape_model is one).  The navigator's own mesh, when a nav_override
# makes it one, takes its geometry from that override.  The sim editor reads
# this table too: switching a body to an ellipsoid moves these keys out of the
# scene.
ELLIPSOID_UNSUPPORTED_KEYS: tuple[str, ...] = (
    'shading',
    'mesh_detail_octaves',
    'pose_scatter',
)


def ellipsoid_unsupported_message(keys: Iterable[str]) -> str:
    """The sentence that refuses mesh-only keys on an ellipsoid body.

    The scene validator raises it, prefixed with the scene source and the body's
    label, when an ellipsoid body gives keys from
    :data:`ELLIPSOID_UNSUPPORTED_KEYS` a non-null value.

    Parameters:
        keys: The refused keys to name; the sentence lists them sorted.

    Returns:
        The sentence, such as ``"['shading'] are not supported on ellipsoid
        bodies (...)"``.
    """
    return (
        f'{sorted(keys)} are not supported on ellipsoid bodies (the ellipsoid '
        f'renderer would silently ignore them); set shape_model: polyhedral_mesh '
        f'to use them'
    )


# Which radial shape keys each feature kind takes: 'required', 'optional', or
# absent from the kind's map, which means the validator rejects the key on
# that kind (a stray shape key would silently author a different feature).
# The validator, the sim editor's kind switch, and the per-kind tables in the
# developer guide and the scene README all follow this one table.
RING_SHAPE_KEYS: tuple[str, ...] = ('width_px', 'side', 'wavelength_px', 'density_wave_damping_px')
RING_KIND_SHAPE_KEYS: dict[str, dict[str, str]] = {
    'ringlet': {'width_px': 'required'},
    'gap': {'width_px': 'required'},
    'edge': {'side': 'optional'},
    'ramp': {'width_px': 'required', 'side': 'optional'},
    'wave': {'wavelength_px': 'required', 'density_wave_damping_px': 'required'},
}
# The 'side' an edge or ramp takes when the key is absent.  An edge's default
# 'in' fills every radius inward of its orbit (the sheet bounded by its outer
# edge); a ramp's default 'out' rises outward from zero at its orbit.
RING_SIDE_DEFAULTS: dict[str, str] = {'edge': 'in', 'ramp': 'out'}


def scene_value(block: Mapping[str, Any], key: str, default: Any) -> Any:
    """Read an optional scene key, treating an explicit null as unset.

    The validator accepts null for every optional key and reads it as absent, so
    a reader must too: ``float(block.get(key, default))`` would raise on a null
    the validator passed.  The ring readers (renderer, navigator-side orbit
    parser, validator bounds, and sim editor) read their optional keys here.

    Parameters:
        block: A scene mapping.
        key: The key to read.
        default: The value an absent key takes.

    Returns:
        ``block[key]``, or ``default`` when the key is absent or null.
    """
    value = block.get(key)
    return default if value is None else value


def default_body_name(index: int) -> str:
    """The name a scene body takes when its entry leaves ``name`` absent or null.

    The renderer keys a body's truth records by this name, the validator refuses two
    bodies that share one, and the scene editor finds a body's truth records by it.

    Parameters:
        index: The entry's zero-based position in the scene's ``bodies`` list.

    Returns:
        ``SIM-BODY-<index + 1>``, such as ``SIM-BODY-1`` for the first entry.
    """
    return f'SIM-BODY-{index + 1}'


def default_body_range_km(index: int) -> float:
    """The range a scene body takes when its entry leaves ``range_km`` absent or null.

    The renderer orders bodies by this range and records it in the render truth.  The
    scene editor shows it, and counts it when it picks a range for an added body and
    when it looks for two bodies at one range.

    Parameters:
        index: The entry's zero-based position in the scene's ``bodies`` list.

    Returns:
        ``index + 1``, such as 1.0 for the first entry.
    """
    return float(index + 1)


def scene_binning(params: Mapping[str, Any]) -> int:
    """The binning factor a scene's ``detector`` block selects, 1 by default.

    Parameters:
        params: The full scene mapping.

    Returns:
        The stated ``detector.binning``, or 1 when the scene leaves the block or the
        key out or sets either to null.

    Raises:
        ValueError: If ``detector.binning`` is set to anything but an integer (a bool
            counts as neither).  Validation refuses such a scene first; the check
            keeps a scene that skips validation from truncating a fraction to a mode.
    """
    detector = params.get('detector') or {}
    binning = detector.get('binning') if isinstance(detector, Mapping) else None
    if binning is None:
        return 1
    if isinstance(binning, bool) or not isinstance(binning, int):
        raise ValueError(f'detector.binning must be an integer, got {binning!r}')
    return binning


# What each ring-plane length and angle key the sim editor's tooltips describe
# holds, per block.  Each of these keys names its unit.  A ring-plane pixel is
# a distance in the ring plane in detector pixels: it projects to that many
# image pixels along the line of nodes, and is foreshortened by sin(B) across
# it.  Block paths drop list indices ('ring_system.features' covers every
# feature entry).
RING_KEY_DESCRIPTIONS: dict[str, dict[str, str]] = {
    'ring_system.features': {
        'width_px': 'the radial width, in ring-plane pixels',
        'wavelength_px': "the 'wave' train's radial wavelength, in ring-plane pixels",
        'density_wave_damping_px': (
            "the 'wave' train's radial e-folding length, in ring-plane pixels"
        ),
    },
    'ring_system.features.orbit': {
        'a_px': 'the semimajor axis, in ring-plane pixels',
        'ae_px': 'the eccentricity times the semimajor axis, in ring-plane pixels',
        'long_peri_deg': 'the pericenter longitude, in degrees from the ascending node',
        'rate_peri_deg_per_day': 'the pericenter precession rate, in degrees per day',
    },
    'ring_system.features.orbit.modes': {
        'amp_px': 'the radial amplitude, in ring-plane pixels',
        'peri_deg': (
            "the longitude of one of the mode's m pericenters, in degrees from the "
            'ascending node; it stays fixed in time'
        ),
    },
    'ring_system.features.orbit.edge_wave': {
        'amp_px': 'the radial amplitude at the launch longitude, in ring-plane pixels',
        'wavelength_px': (
            'the azimuthal wavelength as an arc length along the orbit, in ring-plane pixels'
        ),
        'edge_wave_damping_deg': (
            'the azimuthal e-folding length, in degrees of downstream longitude'
        ),
        'lam0_deg': 'the launch longitude, in degrees from the ascending node',
    },
    'ring_system.azimuthal.modulation': {
        'brightness_peak_longitude_deg': (
            'the longitude of a brightness peak, in degrees from the ascending node'
        ),
    },
    'ring_system.azimuthal.spokes': {
        'r_inner_px': 'the inner radius of the spoke band, in ring-plane pixels',
        'r_outer_px': 'the outer radius of the spoke band, in ring-plane pixels',
    },
    'ring_system.moonlets': {
        'a_px': "the moonlet's orbit radius, in ring-plane pixels",
    },
}


def ring_key_description(block: str, key: str) -> str:
    """What a ring key holds, read from :data:`RING_KEY_DESCRIPTIONS`.

    Parameters:
        block: A block path that is a key of :data:`RING_KEY_DESCRIPTIONS`, such
            as ``'ring_system.features.orbit'``.
        key: A key the table describes in that block, such as ``'a_px'``.

    Returns:
        The key's description, such as ``'the semimajor axis, in ring-plane
        pixels'``.

    Raises:
        KeyError: If the table doesn't describe ``key`` in ``block``.
    """
    descriptions = RING_KEY_DESCRIPTIONS[block]
    if key not in descriptions:
        raise KeyError(f'{block}.{key} is not in RING_KEY_DESCRIPTIONS')
    return descriptions[key]
