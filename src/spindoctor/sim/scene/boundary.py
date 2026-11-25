"""Key inventory and information-boundary classification for sim scenes.

Every top-level and per-object key a scene may carry is inventoried here, and every
key is classified as idealized (information the production pipeline could know from
catalogs, SPICE, labels, or config: exposed to the navigator through
``obs.nav_params``), truth (nature's values, planted errors, variance knobs, and
contaminants: readable only by the image-side renderer), or test-only (read by the
integration tests and the validator, and by neither the renderer nor the navigator).

The top-level sets are derived from the typed scene: a key of
:class:`~spindoctor.sim.scene.idealized.IdealizedScene` is idealized, and a key that
:class:`~spindoctor.sim.scene.truth.Scene` adds is truth or test-only, as its field
declares.  A block the typed scene does not hold yet takes its class from
:data:`~spindoctor.sim.scene.truth.UNTYPED_BLOCKS`.  The per-object sets of the
``bodies``, ``stars``, and ``ring_system`` blocks are written out below until those
blocks are typed, and an import-time assertion keeps them complete and disjoint.

:data:`TRUTH_KEYS` is the machine-readable truth set the ObsSim boundary filter strips
and the structural boundary test iterates; the test-only keys are likewise stripped
from the navigator view (default-deny), so they never reach the navigator either.
:func:`build_nav_params` builds the filtered navigator view from the classification.
"""

import copy
import dataclasses
from typing import Any

from spindoctor.sim.scene.declare import Boundary, declaration_of
from spindoctor.sim.scene.fields import scene_binning
from spindoctor.sim.scene.idealized import IdealizedScene
from spindoctor.sim.scene.truth import UNTYPED_BLOCKS, Scene

__all__ = [
    'PLANTED_POINTING_KEYS',
    'TOP_LEVEL_IDEALIZED_KEYS',
    'TOP_LEVEL_KEYS',
    'TOP_LEVEL_TEST_ONLY_KEYS',
    'TOP_LEVEL_TRUTH_KEYS',
    'TRUTH_KEYS',
    'build_nav_params',
]


def _top_level_keys(boundary: Boundary) -> frozenset[str]:
    """The top-level keys of one boundary class.

    A field of :class:`~spindoctor.sim.scene.idealized.IdealizedScene` is idealized.
    A field :class:`~spindoctor.sim.scene.truth.Scene` adds belongs to the class its
    declaration gives, which is truth or test.  An untyped block belongs to the class
    :data:`~spindoctor.sim.scene.truth.UNTYPED_BLOCKS` gives it.

    Parameters:
        boundary: The boundary class.

    Returns:
        The keys of that class.
    """
    idealized = {item.name for item in dataclasses.fields(IdealizedScene)}
    if boundary == 'idealized':
        typed = idealized
    else:
        typed = {
            item.name
            for item in dataclasses.fields(Scene)
            if item.name not in idealized and declaration_of(item).boundary == boundary
        }
    untyped = {name for name, block_class in UNTYPED_BLOCKS.items() if block_class == boundary}
    return frozenset(typed | untyped)


# Top-level idealized keys: frame identity, emulated-instrument configuration, and
# epoch and timing values the production pipeline reads from labels and published
# models.
TOP_LEVEL_IDEALIZED_KEYS: frozenset[str] = _top_level_keys('idealized')

# Top-level truth keys: the planted pointing error the navigator must recover, the RNG
# realizations, and the contaminant and noise blocks.  One value inside the 'detector'
# block is published: build_nav_params echoes the readout binning, which a real frame's
# label states, as a top-level 'binning' key of the navigator view.
TOP_LEVEL_TRUTH_KEYS: frozenset[str] = _top_level_keys('truth')

# Top-level test-only keys: the scene's expected navigation outcome, read only by the
# sim integration suite's assertion machinery, and the truth paths the author declares
# equal to the published instrument_config on purpose, read only by the validator's
# truth-echo check.  The boundary filter's default-deny keeps them out of 'nav_params'
# alongside the truth keys.
TOP_LEVEL_TEST_ONLY_KEYS: frozenset[str] = _top_level_keys('test')

# Every top-level key a scene may carry.
TOP_LEVEL_KEYS: frozenset[str] = (
    TOP_LEVEL_IDEALIZED_KEYS | TOP_LEVEL_TRUTH_KEYS | TOP_LEVEL_TEST_ONLY_KEYS
)

# The planted pointing keys: the navigated frame applies them, and an offset-free
# render zeroes them and leaves them out of its cache key.
PLANTED_POINTING_KEYS: tuple[str, ...] = ('offset_v', 'offset_u', 'offset_rotation_deg')

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
        'semi_a_px',
        'semi_b_px',
        'semi_c_px',
        'rotation_z',
        'sub_observer_lat_deg',
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
# and the limb-relief field are nature's terrain, the photometric law and
# opposition surge are nature's
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
        'anti_aliasing',
        'nav_override',
    }
)

_BODY_KEYS: frozenset[str] = _BODY_IDEALIZED_KEYS | _BODY_TRUTH_KEYS


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
        'density_wave_damping_px',
        'navigable',
        'declared_orbit_sigma',
    }
)

# Per-feature truth keys: the single-scattering albedo and the
# Henyey-Greenstein asymmetry parameter are nature's scattering behavior
# (the navigator's ring template is geometric, never photometric), and
# 'orbit_error' is the planted ephemeris error applied on the render side
# only -- the navigator predicts from the catalog orbit and sees the
# misplacement only in the rendered image (the ring analog of the body
# ephemeris-error axis).
_RING_FEATURE_TRUTH_KEYS: frozenset[str] = frozenset({'albedo', 'phase_g', 'orbit_error'})

_RING_FEATURE_KEYS: frozenset[str] = _RING_FEATURE_IDEALIZED_KEYS | _RING_FEATURE_TRUTH_KEYS


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
    """Every key of an untyped object block must be classified idealized or truth.

    Runs at import so a schema change that adds a key without classifying it fails
    everything loudly, not just one test.  The typed blocks are classified by their
    types.
    """
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


def build_nav_params(sim_params: dict[str, Any]) -> dict[str, Any]:
    """Build the navigator view of a scene.

    This is the information boundary (the independence guarantee of the
    simulator-realism program): the returned mapping contains only keys
    classified idealized, with every :data:`TRUTH_KEYS` entry stripped.  For
    bodies, a ``nav_override`` mapping is overlaid first and the key dropped,
    so the navigator sees the geometry it *believes* without learning the
    true values underneath.  An object flagged ``navigable: false`` is dropped
    entirely, so a surviving object's flag is always true and carries no hidden
    truth.  All values are deep copies, so navigator-side code cannot mutate the
    renderer's scene.

    One value from the truth-side ``detector`` block crosses: the readout binning, as
    a top-level ``binning`` key (1 when the scene sets none).  A real frame's label
    states its readout mode, so a real pipeline always knows it.  Nothing else in the
    block crosses.

    The scene has been validated, so it holds no null at any depth.  A key that
    ``nav_override`` leaves out keeps the body's own value.

    Parameters:
        sim_params: The full scene mapping (the renderer's input).

    Returns:
        The filtered ``nav_params`` mapping exposed as ``obs.nav_params``.
    """
    nav: dict[str, Any] = {}
    for key, value in sim_params.items():
        if key in _OBJECT_BLOCKS or key == 'ring_system':
            continue  # handled below
        if key in TOP_LEVEL_IDEALIZED_KEYS:
            nav[key] = copy.deepcopy(value)
        # Anything else (truth keys, unknown keys) stays behind the boundary:
        # the filter is default-deny.
    # The readout binning is a commanded mode that a real frame's label states, so
    # the navigator may know it; the rest of the detector block stays behind (#993).
    nav['binning'] = scene_binning(sim_params)
    ring_system = sim_params.get('ring_system')
    if isinstance(ring_system, dict):
        nav['ring_system'] = _filter_ring_system(ring_system)
    for block, (_allowed, idealized, _truth) in _OBJECT_BLOCKS.items():
        if block not in sim_params:
            continue
        filtered_objects: list[dict[str, Any]] = []
        for obj in sim_params.get(block) or []:
            if not isinstance(obj, dict):
                continue
            if obj.get('navigable') is False:
                continue
            merged = dict(obj)
            if block == 'bodies':
                override = merged.pop('nav_override', None)
                if isinstance(override, dict):
                    merged.update(override)
            filtered_objects.append(
                {k: copy.deepcopy(v) for k, v in merged.items() if k in idealized}
            )
        nav[block] = filtered_objects
    return nav


def _filter_ring_system(ring_system: dict[str, Any]) -> dict[str, Any]:
    """The navigator view of the ``ring_system`` block.

    Block-level idealized keys (the shared projection geometry, range, pixel
    scale, phase angle) pass through as deep copies.  Features cross only
    when flagged ``navigable: true`` (the default is false), so the rendered
    system is full of structure the navigator was never told about -- the
    distractor regime the ring system exists to produce.  A crossing feature
    exposes only its idealized keys (kind, shape, catalog orbit, tau, and
    the declared orbit uncertainty), never the photometric truth (albedo,
    phase_g) or the planted ``orbit_error`` -- the navigator knows the error
    bars, never the drawn error values.

    Parameters:
        ring_system: The full ``ring_system`` mapping (renderer input).

    Returns:
        The filtered mapping exposed under ``nav_params['ring_system']``.
    """
    filtered: dict[str, Any] = {
        key: copy.deepcopy(value)
        for key, value in ring_system.items()
        if key in _RING_SYSTEM_IDEALIZED_KEYS and key != 'features'
    }
    features: list[dict[str, Any]] = []
    for feature in ring_system.get('features') or []:
        if not isinstance(feature, dict) or feature.get('navigable') is not True:
            continue
        features.append(
            {k: copy.deepcopy(v) for k, v in feature.items() if k in _RING_FEATURE_IDEALIZED_KEYS}
        )
    filtered['features'] = features
    return filtered
