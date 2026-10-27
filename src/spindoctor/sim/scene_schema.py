"""Key inventory and information-boundary classification for sim scenes.

Every top-level and per-object key a scene may carry is inventoried here
(unknown keys fail validation so typos do not silently render the default
scene), and every inventory key is classified as either idealized
(information the production pipeline could know from catalogs, SPICE, labels,
or config: exposed to the navigator through ``obs.nav_params``), truth
(nature's values, planted errors, variance knobs, and contaminants: readable
only by the image-side renderer), or test-only (the scene's expected
navigation outcome: read by the integration-test assertion machinery, and by
neither the renderer nor the navigator).  :data:`TRUTH_KEYS` is the
machine-readable truth set the ObsSim boundary filter strips and the
structural boundary test iterates; the test-only keys are likewise stripped
from the filtered view (default-deny), so they never reach the navigator
either.  The import-time completeness assertion below keeps the classification
complete and disjoint over all three classes, so a key added to the schema
without a classification fails everything loudly, not just one test.

:mod:`spindoctor.sim.scene` is the public entry point: it consumes this
inventory for validation, builds the filtered navigator view from the
classification, and re-exports the boundary names.
"""

from collections.abc import Iterable, Mapping
from typing import Any


class SimSceneValidationError(ValueError):
    """Raised when a sim scene YAML is missing or malformed."""


# ---------------------------------------------------------------------------
# Key inventory and information-boundary classification.
#
# _ALLOWED_KEYS / _*_KEYS are the complete inventory (unknown keys fail
# validation so typos do not silently render the default scene).  The
# *_IDEALIZED_KEYS / *_TRUTH_KEYS sets classify every inventory key for the
# information boundary; the import-time assertion below keeps the
# classification complete and disjoint.
# ---------------------------------------------------------------------------

# Every top-level key a scene may carry.  These are the flat runtime sim_params
# names the renderer / ObsSim consume directly, plus the schema_version /
# scene_name metadata the renderer ignores.
_ALLOWED_KEYS: frozenset[str] = frozenset(
    {
        'schema_version',
        'scene_name',
        'instrument',
        'size_v',
        'size_u',
        'random_seed',
        'exposure_sec',
        'offset_v',
        'offset_u',
        'offset_rotation_deg',
        'midtime_utc',
        'closest_planet',
        'time',
        'ring_epoch',
        'oversample',
        'optics',
        'detector',
        'artifacts',
        'spk_error',
        'sky_counts',
        'star_catalog_scatter_px',
        'expected',
        'bodies',
        'ring_system',
        'stars',
        'noise',
        'instrument_config',
        'fit_camera_rotation',
    }
)

# Top-level idealized keys: frame identity, emulated-instrument configuration,
# and epoch/timing values the production pipeline reads from labels and
# published models.  'ring_epoch' is deliberately idealized: the precessing
# ring model's epoch is catalog knowledge the navigator-side ring model reads.
TOP_LEVEL_IDEALIZED_KEYS: frozenset[str] = frozenset(
    {
        'schema_version',
        'scene_name',
        'instrument',
        'size_v',
        'size_u',
        'exposure_sec',
        'midtime_utc',
        'closest_planet',
        'time',
        'ring_epoch',
        'bodies',
        'ring_system',
        'stars',
        'instrument_config',
        'fit_camera_rotation',
    }
)

# Top-level truth keys: the planted pointing error the navigator must recover,
# the RNG realization, and the contaminant / noise fields.
TOP_LEVEL_TRUTH_KEYS: frozenset[str] = frozenset(
    {
        'random_seed',
        'offset_v',
        'offset_u',
        'offset_rotation_deg',
        'oversample',
        'optics',
        'detector',
        'artifacts',
        'spk_error',
        'sky_counts',
        'star_catalog_scatter_px',
        'noise',
    }
)

# Top-level test-only keys: the scene's expected navigation outcome.  It is
# read only by the sim integration suite's assertion machinery -- consumed by
# neither the image-side renderer nor the navigator, so it is neither idealized
# information the pipeline could know nor a planted truth the renderer draws.
# It is a third, disjoint class; the boundary filter's default-deny keeps it
# out of ``nav_params`` alongside the truth keys.
TOP_LEVEL_TEST_ONLY_KEYS: frozenset[str] = frozenset({'expected'})

# Per-body idealized keys: the ellipsoid/mesh geometry, pose, lighting, and
# physical scale the production pipeline knows from SPICE and shape catalogs.
# The mesh keys are idealized because the published shape model of an
# irregular body is catalog knowledge; a scene plants shape error through
# 'nav_override', not by hiding the mesh.
_BODY_IDEALIZED_KEYS: frozenset[str] = frozenset(
    {
        'name',
        'shape_model',
        'center_v',
        'center_u',
        'axis1',
        'axis2',
        'axis3',
        'rotation_z',
        'rotation_tilt',
        'illumination_angle',
        'phase_angle',
        'range_km',
        'km_per_pixel',
        'mesh_lumpiness',
        'mesh_n_lat',
        'mesh_n_lon',
        'mesh_seed',
        'mesh_detail_octaves',
        'pose_euler_deg',
    }
)

# Per-body truth keys: surface texture (craters, the multiplicative albedo
# texture, giant-planet bands/storms, transiting moons and their shadows)
# and the limb-relief field are nature's terrain, 'seed' is its
# realization, the photometric law and opposition surge are nature's
# scattering behavior (the navigator's template always shades Lambert),
# 'atmosphere' is the exponential haze layer above the surface (the
# navigator's template keeps a hard limb at the reference radius, so the
# soft rendered limb is a designed mismatch, and the block's structure keys
# -- axis tilt, hemispheric and azimuthal falloff scaling, hemispheric
# brightness, the axial interior ramp, and clouds -- are the ways nature
# breaks the mirror symmetry a haze navigator assumes; the whole block is
# one atomic truth entry, so every key inside it is truth), 'shading' is the rendered
# mesh's smooth-shading mode (the shared rasterizer gains the capability;
# the navigator's predicted mesh keeps flat shading because this key never
# crosses), 'pose_scatter' is a per-frame unmodelable rotation-state error
# (the navigator predicts the catalog pose), and 'anti_aliasing' is an
# image-side rendering-fidelity knob (the navigator's template always
# renders at full anti-aliasing).  'nav_override' is special: its VALUES
# are what the navigator believes (idealized), so build_nav_params overlays
# them onto the body and drops the key; the underlying overridden true
# values never cross.
_BODY_TRUTH_KEYS: frozenset[str] = frozenset(
    {
        'crater_fill',
        'crater_min_radius',
        'crater_max_radius',
        'crater_power_law_exponent',
        'crater_relief_scale',
        'limb_relief_rms',
        'limb_relief_corr_deg',
        'photometric_law',
        'minnaert_k',
        'opposition_surge',
        'albedo_texture',
        'disc_texture',
        'transits',
        'atmosphere',
        'shading',
        'pose_scatter',
        'seed',
        'anti_aliasing',
        'nav_override',
    }
)

_BODY_KEYS: frozenset[str] = _BODY_IDEALIZED_KEYS | _BODY_TRUTH_KEYS

# Body keys the polyhedral-mesh render path does not consume: the ellipsoid
# renderer's crater carving, the topographic path's photometric laws and
# surface texture, and the ellipsoid-frame atmosphere halo (whose geometry is
# built from rotation_z / rotation_tilt around the ellipse, which a mesh's
# arbitrary Euler pose and lumpy limb do not follow).  A mesh body giving one
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


# Per-star idealized keys: catalog identity, position, magnitude, spectral
# class, the predicted smear vector (the pipeline computes it from attitude
# telemetry), and the PSF fitting-window size (instrument configuration).
# 'navigable' is idealized: it is the flag that drives the boundary filter --
# a non-navigable star is dropped from nav_params entirely, so a surviving
# star's flag is always true and carries no hidden truth.
_STAR_IDEALIZED_KEYS: frozenset[str] = frozenset(
    {
        'name',
        'catalog_name',
        'v',
        'u',
        'vmag',
        'spectral_class',
        'move_v',
        'move_u',
        'psf_size',
        'navigable',
    }
)

# Per-star truth keys: a per-star PSF width override is an anomaly of the
# rendered image (the navigator only knows the instrument's published PSF);
# 'catalog_error_v/u' displace the RENDERED star from the catalog position the
# navigator predicts from; 'companion' plants an unresolved binary whose
# photocenter sits off the catalog position (a physical catalog error); and
# 'delta_mag' renders a variable star at a brightness other than its cataloged
# vmag.  None of these are knowable from a catalog, so all are truth.
_STAR_TRUTH_KEYS: frozenset[str] = frozenset(
    {'psf_sigma', 'catalog_error_v', 'catalog_error_u', 'companion', 'delta_mag'}
)

_STAR_KEYS: frozenset[str] = _STAR_IDEALIZED_KEYS | _STAR_TRUTH_KEYS

# The object blocks of the schema: block name -> (allowed, idealized, truth).
# Rings are not an object block: the optical-depth ring system is a single
# mapping (shared projection geometry plus a feature list) with its own
# two-level classification below.
_OBJECT_BLOCKS: dict[str, tuple[frozenset[str], frozenset[str], frozenset[str]]] = {
    'bodies': (_BODY_KEYS, _BODY_IDEALIZED_KEYS, _BODY_TRUTH_KEYS),
    'stars': (_STAR_KEYS, _STAR_IDEALIZED_KEYS, _STAR_TRUTH_KEYS),
}

# ---------------------------------------------------------------------------
# The ring_system block: the optical-depth ring system.  Unlike the object
# blocks above it is a mapping (shared projection geometry plus a feature
# list), so it carries its own two-level classification here and is
# special-cased by the validator and the boundary filter.
# ---------------------------------------------------------------------------

# Block-level idealized keys: the shared projection geometry (opening angles
# and node are SPICE knowledge), the physical range and pixel scale (SPICE),
# and the phase angle the photometry evaluates at.  'features' is the feature
# list itself; its entries carry the two-level classification below.
_RING_SYSTEM_IDEALIZED_KEYS: frozenset[str] = frozenset(
    {'geometry', 'features', 'range_km', 'km_per_pixel', 'phase_deg'}
)

# Block-level truth keys: 'azimuthal' (brightness modulation, seeded spokes,
# planet-shadow wedge) is non-navigable clutter crossing the navigable
# edges, and 'moonlets' are embedded confounders sitting on the features --
# neither is catalog knowledge, so neither reaches the navigator.
_RING_SYSTEM_TRUTH_KEYS: frozenset[str] = frozenset({'azimuthal', 'moonlets'})

_RING_SYSTEM_KEYS: frozenset[str] = _RING_SYSTEM_IDEALIZED_KEYS | _RING_SYSTEM_TRUTH_KEYS

# The shared geometry sub-block (all idealized: both sides project through
# it, per the plan's shared-helper rule).
_RING_SYSTEM_GEOMETRY_KEYS: frozenset[str] = frozenset(
    {'center_v', 'center_u', 'opening_deg_obs', 'opening_deg_sun', 'node_deg'}
)

# Per-feature idealized keys: the feature kind, its catalog orbit (mode-1
# ellipse plus m-modes and edge wave), its radial shape (width / side /
# wave train), its declared optical depth, and the orbit uncertainty the
# navigator is entitled to know ('declared_orbit_sigma' -- the error BARS,
# never the drawn error values) are catalog knowledge.  'navigable'
# (default false) drives the boundary filter: a non-navigable feature is
# dropped from nav_params entirely, so the rendered system is full of
# structure the navigator was never told about, and a surviving feature's
# flag is always true and carries no hidden truth.
_RING_FEATURE_IDEALIZED_KEYS: frozenset[str] = frozenset(
    {
        'name',
        'kind',
        'width_px',
        'tau',
        'orbit',
        'side',
        'wavelength_px',
        'damping_px',
        'navigable',
        'declared_orbit_sigma',
    }
)

# Per-feature truth keys: the single-scattering albedo and the
# Henyey-Greenstein asymmetry parameter are nature's scattering behavior
# (the navigator's ring template is geometric, never photometric), and
# 'orbit_error' is the planted ephemeris error applied on the render side
# only -- the navigator predicts from the catalog orbit and must absorb the
# misplacement honestly (the ring analog of the body ephemeris-error axis).
_RING_FEATURE_TRUTH_KEYS: frozenset[str] = frozenset({'albedo', 'phase_g', 'orbit_error'})

_RING_FEATURE_KEYS: frozenset[str] = _RING_FEATURE_IDEALIZED_KEYS | _RING_FEATURE_TRUTH_KEYS

# Which radial shape keys each feature kind takes: 'required', 'optional', or
# absent from the kind's map, which means the validator rejects the key on
# that kind (a stray shape key would silently author a different feature).
# The validator, the sim editor's kind switch, and the per-kind tables in the
# developer guide and the scene README all follow this one table.
RING_SHAPE_KEYS: tuple[str, ...] = ('width_px', 'side', 'wavelength_px', 'damping_px')
RING_KIND_SHAPE_KEYS: dict[str, dict[str, str]] = {
    'ringlet': {'width_px': 'required'},
    'gap': {'width_px': 'required'},
    'edge': {'side': 'optional'},
    'ramp': {'width_px': 'required', 'side': 'optional'},
    'wave': {'wavelength_px': 'required', 'damping_px': 'required'},
}
# The 'side' an edge or ramp takes when the key is absent.  An edge's default
# 'in' fills every radius inward of its orbit (the sheet bounded by its outer
# edge); a ramp's default 'out' rises outward from zero at its orbit.
RING_SIDE_DEFAULTS: dict[str, str] = {'edge': 'in', 'ramp': 'out'}

# Retired ring keys, per block, mapped to (the key that replaced it, what the
# key holds).  Each ring-plane length and each angle in the ring_system block
# names its unit.  The validator and the ring renderer refuse a retired key
# (refuse_renamed_ring_keys) with a message naming its replacement, nothing
# translates it, and the sim editor's tooltips read the descriptions from
# here.  A ring-plane pixel is a distance in the ring plane in detector
# pixels: it projects to that many image pixels along the line of nodes, and
# is foreshortened by sin(B) across it.  Block paths drop list indices
# ('ring_system.features' covers every feature entry).
RENAMED_RING_KEYS: dict[str, dict[str, tuple[str, str]]] = {
    'ring_system.features': {
        'width': ('width_px', 'the radial width, in ring-plane pixels'),
        'wavelength': (
            'wavelength_px',
            "the 'wave' train's radial wavelength, in ring-plane pixels",
        ),
        'damping': (
            'damping_px',
            "the 'wave' train's radial e-folding length, in ring-plane pixels",
        ),
    },
    'ring_system.features.orbit': {
        'a': ('a_px', 'the semimajor axis, in ring-plane pixels'),
        'ae': ('ae_px', 'the eccentricity times the semimajor axis, in ring-plane pixels'),
        'long_peri': (
            'long_peri_deg',
            'the pericenter longitude, in degrees from the ascending node',
        ),
        'rate_peri': (
            'rate_peri_deg_per_day',
            'the pericenter precession rate, in degrees per day',
        ),
    },
    'ring_system.features.orbit.modes': {
        'amp': ('amp_px', 'the radial amplitude, in ring-plane pixels'),
        'peri': (
            'peri_deg',
            "the longitude of one of the mode's m pericenters, in degrees from the "
            'ascending node; it stays fixed in time',
        ),
    },
    'ring_system.features.orbit.edge_wave': {
        'amp': ('amp_px', 'the radial amplitude at the launch longitude, in ring-plane pixels'),
        'wavelength': (
            'wavelength_px',
            'the azimuthal wavelength as an arc length along the orbit, in ring-plane pixels',
        ),
        'damp': (
            'damping_rad',
            'the azimuthal e-folding length, in radians of downstream longitude',
        ),
        'lam0': ('lam0_deg', 'the launch longitude, in degrees from the ascending node'),
    },
    'ring_system.azimuthal.modulation': {
        'phase_deg': (
            'lam_peak_deg',
            'the longitude of a brightness peak, in degrees from the ascending node',
        ),
    },
    'ring_system.azimuthal.spokes': {
        'r_inner': ('r_inner_px', 'the inner radius of the spoke band, in ring-plane pixels'),
        'r_outer': ('r_outer_px', 'the outer radius of the spoke band, in ring-plane pixels'),
    },
    'ring_system.moonlets': {
        'a': ('a_px', "the moonlet's orbit radius, in ring-plane pixels"),
    },
}


def ring_key_description(block: str, key: str) -> str:
    """What a renamed ring key holds, read from :data:`RENAMED_RING_KEYS`.

    Parameters:
        block: A block path that is a key of :data:`RENAMED_RING_KEYS`, such as
            ``'ring_system.features.orbit'``.
        key: The current name of a key that replaced a retired one in that
            block, such as ``'a_px'``.

    Returns:
        The description recorded beside the rename, such as ``'the semimajor
        axis, in ring-plane pixels'``.

    Raises:
        KeyError: If ``key`` did not replace a retired key in ``block``.
    """
    for new_key, description in RENAMED_RING_KEYS[block].values():
        if new_key == key:
            return description
    raise KeyError(f'{block}.{key} did not replace a retired ring key')


def renamed_ring_key_message(block: str, old_key: str) -> str:
    """The validation message for a retired ring key.

    Parameters:
        block: A block path that is a key of :data:`RENAMED_RING_KEYS`.
        old_key: A retired key listed for that block.

    Returns:
        ``'<old>' was renamed; use '<new>' (<what it holds>)``.
    """
    new_key, description = RENAMED_RING_KEYS[block][old_key]
    return f'{old_key!r} was renamed; use {new_key!r} ({description})'


def _mapping_entries(value: Any) -> list[tuple[int, Mapping[str, Any]]]:
    """The ``(index, entry)`` pairs of a list's mapping entries.

    Parameters:
        value: A scene value that should be a list of mappings.

    Returns:
        Each mapping entry with its list index; empty when ``value`` is not a
        list.  Entries that are not mappings are left out.
    """
    if not isinstance(value, list):
        return []
    return [(index, entry) for index, entry in enumerate(value) if isinstance(entry, Mapping)]


def _ring_blocks(ring_system: Mapping[str, Any]) -> list[tuple[str, str, Mapping[str, Any]]]:
    """Every mapping in a ring system that :data:`RENAMED_RING_KEYS` covers.

    Entries that are not mappings, and lists that are not lists, are skipped:
    the validator reports those shapes itself.

    Parameters:
        ring_system: The scene's ``ring_system`` mapping.

    Returns:
        ``(block, label, mapping)`` triples in scene order, where ``block`` is
        the mapping's path in :data:`RENAMED_RING_KEYS` and ``label`` its path
        with list indices, such as ``'ring_system.features[0].orbit'``.
    """
    blocks: list[tuple[str, str, Mapping[str, Any]]] = []
    for index, feature in _mapping_entries(ring_system.get('features')):
        label = f'ring_system.features[{index}]'
        blocks.append(('ring_system.features', label, feature))
        orbit = feature.get('orbit')
        if not isinstance(orbit, Mapping):
            continue
        blocks.append(('ring_system.features.orbit', f'{label}.orbit', orbit))
        for mode_index, mode in _mapping_entries(orbit.get('modes')):
            blocks.append(
                ('ring_system.features.orbit.modes', f'{label}.orbit.modes[{mode_index}]', mode)
            )
        wave = orbit.get('edge_wave')
        if isinstance(wave, Mapping):
            blocks.append(
                ('ring_system.features.orbit.edge_wave', f'{label}.orbit.edge_wave', wave)
            )
    azimuthal = ring_system.get('azimuthal')
    if isinstance(azimuthal, Mapping):
        for key in ('modulation', 'spokes'):
            sub = azimuthal.get(key)
            if isinstance(sub, Mapping):
                path = f'ring_system.azimuthal.{key}'
                blocks.append((path, path, sub))
    for index, moonlet in _mapping_entries(ring_system.get('moonlets')):
        blocks.append(('ring_system.moonlets', f'ring_system.moonlets[{index}]', moonlet))
    return blocks


def refuse_renamed_ring_keys(ring_system: Mapping[str, Any], *, source: str | None = None) -> None:
    """Refuse a ring system that carries a retired key anywhere.

    The scene validator and the ring renderer
    (:func:`spindoctor.sim.forward.ring_system.render_ring_system`) both call
    this, so a retired key fails the same way whether the scene came from a
    file or an in-memory dict that skipped validation.

    Parameters:
        ring_system: The scene's ``ring_system`` mapping.
        source: The scene's label, prefixed to the message when given.

    Raises:
        SimSceneValidationError: On the first mapping, in scene order, that
            carries a key listed for its block in :data:`RENAMED_RING_KEYS`;
            the message is ``[<source>: ]<label>: <renamed_ring_key_message>``
            for every retired key in that mapping, in sorted order, joined by
            semicolons.
    """
    prefix = '' if source is None else f'{source}: '
    for block, label, mapping in _ring_blocks(ring_system):
        retired = sorted(set(mapping) & set(RENAMED_RING_KEYS[block]))
        if retired:
            messages = '; '.join(renamed_ring_key_message(block, key) for key in retired)
            raise SimSceneValidationError(f'{prefix}{label}: {messages}')


# The machine-readable truth-key set the ObsSim boundary filter strips and
# the structural boundary test iterates.  Per-object-block entries use dotted
# '<block>.<key>' paths ('ring_system.features.<key>' for the two-level
# feature entries); top-level entries are bare key names.
TRUTH_KEYS: frozenset[str] = (
    frozenset(TOP_LEVEL_TRUTH_KEYS)
    | frozenset(
        f'{block}.{key}'
        for block, (_allowed, _idealized, truth) in _OBJECT_BLOCKS.items()
        for key in truth
    )
    | frozenset(f'ring_system.{key}' for key in _RING_SYSTEM_TRUTH_KEYS)
    | frozenset(f'ring_system.features.{key}' for key in _RING_FEATURE_TRUTH_KEYS)
)


def _assert_boundary_classification_complete() -> None:
    """Every schema key must be classified idealized, truth, or test-only.

    The three top-level classes must partition the inventory: every key falls
    in exactly one, and no key falls in two.  Runs at import so a schema change
    that adds a key without classifying it fails everything loudly, not just one
    test.
    """
    classes = {
        'idealized': TOP_LEVEL_IDEALIZED_KEYS,
        'truth': TOP_LEVEL_TRUTH_KEYS,
        'test-only': TOP_LEVEL_TEST_ONLY_KEYS,
    }
    for (name_a, set_a), (name_b, set_b) in (
        (('idealized', classes['idealized']), ('truth', classes['truth'])),
        (('idealized', classes['idealized']), ('test-only', classes['test-only'])),
        (('truth', classes['truth']), ('test-only', classes['test-only'])),
    ):
        overlap = set_a & set_b
        assert not overlap, (
            f'top-level keys classified both {name_a} and {name_b}: {sorted(overlap)}'
        )
    classified = TOP_LEVEL_IDEALIZED_KEYS | TOP_LEVEL_TRUTH_KEYS | TOP_LEVEL_TEST_ONLY_KEYS
    unclassified = _ALLOWED_KEYS - classified
    assert not unclassified, f'top-level keys with no boundary class: {sorted(unclassified)}'
    unknown = classified - _ALLOWED_KEYS
    assert not unknown, f'classified top-level keys not in the inventory: {sorted(unknown)}'
    blocks: dict[str, tuple[frozenset[str], frozenset[str], frozenset[str]]] = {
        **_OBJECT_BLOCKS,
        'ring_system': (_RING_SYSTEM_KEYS, _RING_SYSTEM_IDEALIZED_KEYS, _RING_SYSTEM_TRUTH_KEYS),
        'ring_system.features': (
            _RING_FEATURE_KEYS,
            _RING_FEATURE_IDEALIZED_KEYS,
            _RING_FEATURE_TRUTH_KEYS,
        ),
    }
    for block, (allowed, idealized, truth) in blocks.items():
        overlap = idealized & truth
        assert not overlap, f'{block} keys classified both idealized and truth: {sorted(overlap)}'
        unclassified = allowed - (idealized | truth)
        assert not unclassified, f'{block} keys with no boundary class: {sorted(unclassified)}'


_assert_boundary_classification_complete()
