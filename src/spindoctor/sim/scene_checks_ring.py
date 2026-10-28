"""Ring-system field validators for the sim-scene schema.

The ``ring_system`` block is the schema's one mapping-valued object block: the
shared projection geometry, the radial optical-depth feature list (each entry
carrying a kind, kind-specific shape keys, a catalog orbit with m-modes and an
edge wave, planted orbit error, and photometric truth), and the truth-side
azimuthal / moonlet clutter.  Its checkers live here, beside their key
inventories; the scalar primitives they share with the other block checkers
stay in :mod:`spindoctor.sim.scene_checks`, which also documents the overall
validation contract (unknown keys fail, every violation raises
:class:`spindoctor.sim.scene_schema.SimSceneValidationError`).

:func:`spindoctor.sim.scene.validate_sim_params` drives ``_check_ring_system``
once per scene.
"""

from __future__ import annotations

import math
from typing import Any

from spindoctor.sim.scene_checks import (
    _check_optional_bool,
    _check_optional_mapping_list,
    _check_optional_nonnegative_number,
    _check_optional_number,
    _check_optional_positive_number,
    _check_optional_str,
)
from spindoctor.sim.scene_schema import (
    _RING_FEATURE_KEYS,
    _RING_SYSTEM_GEOMETRY_KEYS,
    _RING_SYSTEM_KEYS,
    RING_KIND_SHAPE_KEYS,
    RING_SHAPE_KEYS,
    SimSceneValidationError,
    scene_value,
)

# The block, geometry, and feature key inventories are single-sourced from
# the boundary classification in scene_schema (the idealized | truth unions
# imported above), the same way the bodies / stars inventories are: a key
# added to the schema is accepted by the validator with no hand-mirrored copy
# to drift, and the editor coverage test (which builds its expected key set
# from these same names) fails loudly until the editor can author it.  The
# sub-block inventories below have no schema counterpart (the boundary
# classifies whole sub-mappings such as 'orbit' as single keys), so they are
# owned here.
_RING_FEATURE_KINDS: frozenset[str] = frozenset(RING_KIND_SHAPE_KEYS)
_RING_FEATURE_ORBIT_KEYS: frozenset[str] = frozenset(
    {'a_px', 'ae_px', 'long_peri_deg', 'rate_peri_deg_per_day', 'modes', 'edge_wave'}
)
_RING_ORBIT_MODE_KEYS: frozenset[str] = frozenset({'m', 'amp_px', 'peri_deg'})
_RING_EDGE_WAVE_KEYS: frozenset[str] = frozenset(
    {'amp_px', 'wavelength_px', 'edge_wave_damping_deg', 'lam0_deg'}
)
# The edge-wave longitude difference is taken modulo 2*pi, so just upstream
# of the launch longitude the wave carries a wrap-seam residual of
# amp_px * exp(-2*pi/damping) with the damping in radians.  Capping the
# damping at 2 radians, 360/pi degrees (about 114.59), bounds that residual at
# exp(-pi), about 4.3% of amp_px, keeping the seam visually negligible.
_RING_EDGE_WAVE_DAMPING_MAX_DEG: float = 360.0 / math.pi
# The planted per-feature ephemeris error (truth: render side only) and the
# uncertainty the navigator is entitled to know (idealized error bars).
_RING_ORBIT_ERROR_KEYS: frozenset[str] = frozenset(
    {'delta_a_px', 'delta_ae_px', 'delta_long_peri_deg'}
)
_RING_ORBIT_SIGMA_KEYS: frozenset[str] = frozenset(
    {'sigma_a_px', 'sigma_ae_px', 'sigma_long_peri_deg'}
)
_RING_FEATURE_SIDES: frozenset[str] = frozenset({'in', 'out'})
# Truth-side azimuthal structure (intensity only, never tau) and embedded
# moonlets (opaque discs at ring depth, with optional propeller lobes).
_RING_AZIMUTHAL_KEYS: frozenset[str] = frozenset({'modulation', 'spokes', 'shadow'})
_RING_MODULATION_KEYS: frozenset[str] = frozenset(
    {'amplitude', 'm', 'brightness_peak_longitude_deg'}
)
_RING_SHADOW_KEYS: frozenset[str] = frozenset({'start_deg', 'extent_deg', 'darkness'})
_RING_SPOKES_KEYS: frozenset[str] = frozenset(
    {'count', 'r_inner_px', 'r_outer_px', 'contrast', 'width_deg'}
)
_RING_MOONLET_KEYS: frozenset[str] = frozenset(
    {'a_px', 'lam_deg', 'radius_px', 'amplitude', 'propeller'}
)
_RING_PROPELLER_KEYS: frozenset[str] = frozenset({'length_deg', 'width_px', 'contrast'})


def _check_ring_system(value: Any, *, source: str) -> None:
    """Validate the scene-level ``ring_system`` block.

    The block carries the shared projection geometry (opening angles, node,
    center), the system's physical range and pixel scale, the phase angle,
    and the list of radial optical-depth features.  The geometry block and
    both opening angles are required: an unstated opening angle has no
    sensible default (0 renders nothing and 90 silently degenerates to
    sky-plane circles).

    Parameters:
        value: The ``ring_system`` mapping, or None when the block is absent.
        source: Label used in error messages.

    Raises:
        SimSceneValidationError: On any unknown, missing, or invalid field.
    """
    if value is None:
        return
    if not isinstance(value, dict):
        raise SimSceneValidationError(f'{source}: ring_system must be a mapping when present')
    unknown = set(value) - _RING_SYSTEM_KEYS
    if unknown:
        raise SimSceneValidationError(f'{source}: ring_system: unknown keys: {sorted(unknown)}')
    geometry = value.get('geometry')
    if not isinstance(geometry, dict):
        raise SimSceneValidationError(f'{source}: ring_system.geometry is required (a mapping)')
    unknown = set(geometry) - _RING_SYSTEM_GEOMETRY_KEYS
    if unknown:
        raise SimSceneValidationError(
            f'{source}: ring_system.geometry: unknown keys: {sorted(unknown)}'
        )
    for key in ('center_v', 'center_u', 'node_deg'):
        _check_optional_number(geometry.get(key), f'ring_system.geometry.{key}', source=source)
    for key in ('opening_deg_obs', 'opening_deg_sun'):
        opening = geometry.get(key)
        label = f'ring_system.geometry.{key}'
        if opening is None:
            raise SimSceneValidationError(f'{source}: {label} is required')
        _check_optional_number(opening, label, source=source)
        if not -90.0 < float(opening) <= 90.0:
            raise SimSceneValidationError(
                f'{source}: {label} must lie in (-90, 90]; got {opening!r}'
            )
    _check_optional_positive_number(value.get('range_km'), 'ring_system.range_km', source=source)
    _check_optional_positive_number(
        value.get('km_per_pixel'), 'ring_system.km_per_pixel', source=source
    )
    phase = value.get('phase_deg')
    _check_optional_number(phase, 'ring_system.phase_deg', source=source)
    if phase is not None and not 0.0 <= float(phase) <= 180.0:
        raise SimSceneValidationError(
            f'{source}: ring_system.phase_deg must lie in [0, 180]; got {phase!r}'
        )
    features = value.get('features')
    _check_optional_mapping_list(features, 'ring_system.features', source=source)
    for index, feature in enumerate(features or []):
        _check_ring_feature(feature, index=index, source=source)
    _check_ring_feature_names_unique(features or [], source=source)
    _check_ring_azimuthal(value.get('azimuthal'), source=source)
    moonlets = value.get('moonlets')
    _check_optional_mapping_list(moonlets, 'ring_system.moonlets', source=source)
    for index, moonlet in enumerate(moonlets or []):
        _check_ring_moonlet(moonlet, index=index, source=source)


def _check_ring_azimuthal(value: Any, *, source: str) -> None:
    """Validate the truth-side ``ring_system.azimuthal`` structure block.

    ``modulation`` is a low-frequency brightness modulation
    (self-gravity-wake asymmetry) with an azimuthal wavenumber ``m`` of at
    least 1 (default 1) and a brightness peak at ``brightness_peak_longitude_deg``, ``shadow``
    a planet-shadow darkening wedge, and ``spokes`` a seeded field of
    azimuthally sharp, radially broad albedo wedges between ``r_inner_px``
    and ``r_outer_px``.  All three modulate the emitted intensity only --
    they are albedo/illumination structure, not optical depth.

    Parameters:
        value: The ``azimuthal`` mapping, or None when absent.
        source: Label used in error messages.

    Raises:
        SimSceneValidationError: On any unknown, missing, or invalid field.
    """
    if value is None:
        return
    if not isinstance(value, dict):
        raise SimSceneValidationError(
            f'{source}: ring_system.azimuthal must be a mapping when present'
        )
    unknown = set(value) - _RING_AZIMUTHAL_KEYS
    if unknown:
        raise SimSceneValidationError(
            f'{source}: ring_system.azimuthal: unknown keys: {sorted(unknown)}'
        )
    modulation = value.get('modulation')
    if modulation is not None:
        label = 'ring_system.azimuthal.modulation'
        if not isinstance(modulation, dict):
            raise SimSceneValidationError(f'{source}: {label} must be a mapping when present')
        unknown = set(modulation) - _RING_MODULATION_KEYS
        if unknown:
            raise SimSceneValidationError(f'{source}: {label}: unknown keys: {sorted(unknown)}')
        if modulation.get('amplitude') is None:
            raise SimSceneValidationError(f'{source}: {label}.amplitude is required')
        _check_optional_nonnegative_number(
            modulation.get('amplitude'), f'{label}.amplitude', source=source
        )
        m = modulation.get('m')
        if m is not None and (isinstance(m, bool) or not isinstance(m, int) or m < 1):
            raise SimSceneValidationError(
                f'{source}: {label}.m must be an integer >= 1 when present; got {m!r}'
            )
        _check_optional_number(
            modulation.get('brightness_peak_longitude_deg'),
            f'{label}.brightness_peak_longitude_deg',
            source=source,
        )
    shadow = value.get('shadow')
    if shadow is not None:
        label = 'ring_system.azimuthal.shadow'
        if not isinstance(shadow, dict):
            raise SimSceneValidationError(f'{source}: {label} must be a mapping when present')
        unknown = set(shadow) - _RING_SHADOW_KEYS
        if unknown:
            raise SimSceneValidationError(f'{source}: {label}: unknown keys: {sorted(unknown)}')
        if shadow.get('extent_deg') is None:
            raise SimSceneValidationError(f'{source}: {label}.extent_deg is required')
        _check_optional_positive_number(
            shadow.get('extent_deg'), f'{label}.extent_deg', source=source
        )
        _check_optional_number(shadow.get('start_deg'), f'{label}.start_deg', source=source)
        darkness = shadow.get('darkness')
        if darkness is None:
            raise SimSceneValidationError(f'{source}: {label}.darkness is required')
        _check_optional_number(darkness, f'{label}.darkness', source=source)
        if not 0.0 <= float(darkness) <= 1.0:
            raise SimSceneValidationError(
                f'{source}: {label}.darkness must lie in [0, 1]; got {darkness!r}'
            )
    spokes = value.get('spokes')
    if spokes is not None:
        label = 'ring_system.azimuthal.spokes'
        if not isinstance(spokes, dict):
            raise SimSceneValidationError(f'{source}: {label} must be a mapping when present')
        unknown = set(spokes) - _RING_SPOKES_KEYS
        if unknown:
            raise SimSceneValidationError(f'{source}: {label}: unknown keys: {sorted(unknown)}')
        count = spokes.get('count')
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise SimSceneValidationError(
                f'{source}: {label}.count must be an integer >= 1; got {count!r}'
            )
        for key in ('r_inner_px', 'r_outer_px', 'width_deg'):
            if spokes.get(key) is None:
                raise SimSceneValidationError(f'{source}: {label}.{key} is required')
            _check_optional_positive_number(spokes.get(key), f'{label}.{key}', source=source)
        if float(spokes['r_outer_px']) <= float(spokes['r_inner_px']):
            raise SimSceneValidationError(
                f'{source}: {label}.r_outer_px must exceed r_inner_px; got '
                f'{spokes["r_inner_px"]!r}..{spokes["r_outer_px"]!r}'
            )
        if spokes.get('contrast') is None:
            raise SimSceneValidationError(f'{source}: {label}.contrast is required')
        _check_optional_number(spokes.get('contrast'), f'{label}.contrast', source=source)


def _check_ring_feature_names_unique(features: list[dict[str, Any]], *, source: str) -> None:
    """Fail validation when two ring features share an effective name.

    A feature's name is its identity everywhere identity matters: the
    per-feature truth entries carry it, and the navigator-side ring model
    builds its upper-cased ``feature_id`` from it (an unnamed feature takes
    a positional ``RING-FEATURE-<n>`` default).  Two features resolving to
    the same name would collide in every such name-keyed consumer, so the
    collision fails here, mirroring the body-name rule.

    Parameters:
        features: The ``ring_system.features`` list (entries already
            type-checked).
        source: Label used in error messages.

    Raises:
        SimSceneValidationError: If two features share an effective name
            (case-insensitive, positional defaults included).
    """
    seen: dict[str, int] = {}
    for index, feature in enumerate(features):
        effective = str(feature.get('name') or f'RING-FEATURE-{index + 1}').upper()
        if effective in seen:
            raise SimSceneValidationError(
                f'{source}: ring_system.features[{seen[effective]}] and '
                f'ring_system.features[{index}] share the effective name {effective!r} '
                f'(names are case-insensitive and an unnamed feature defaults to its '
                f'positional RING-FEATURE-<n> name); name-keyed consumers would '
                f'collide, so give each feature a unique name'
            )
        seen[effective] = index


def _check_ring_moonlet(obj: dict[str, Any], *, index: int, source: str) -> None:
    """Validate one ``ring_system.moonlets`` entry.

    A moonlet is an opaque disc embedded in the ring plane at polar
    placement ``(a_px, lam_deg)``, emitting ``amplitude`` in normalized signal
    units, optionally with a stylized propeller disturbance (two tau lobes
    straddling it radially and azimuthally).

    Parameters:
        obj: The moonlet mapping.
        index: The moonlet's index in the list (for error messages).
        source: Label used in error messages.

    Raises:
        SimSceneValidationError: On any unknown, missing, or invalid field.
    """
    label = f'ring_system.moonlets[{index}]'
    unknown = set(obj) - _RING_MOONLET_KEYS
    if unknown:
        raise SimSceneValidationError(f'{source}: {label}: unknown keys: {sorted(unknown)}')
    if obj.get('a_px') is None:
        raise SimSceneValidationError(f'{source}: {label}.a_px is required')
    _check_optional_positive_number(obj.get('a_px'), f'{label}.a_px', source=source)
    if obj.get('lam_deg') is None:
        raise SimSceneValidationError(f'{source}: {label}.lam_deg is required')
    _check_optional_number(obj.get('lam_deg'), f'{label}.lam_deg', source=source)
    if obj.get('amplitude') is None:
        raise SimSceneValidationError(f'{source}: {label}.amplitude is required')
    _check_optional_nonnegative_number(obj.get('amplitude'), f'{label}.amplitude', source=source)
    _check_optional_positive_number(obj.get('radius_px'), f'{label}.radius_px', source=source)
    propeller = obj.get('propeller')
    if propeller is not None:
        prop_label = f'{label}.propeller'
        if not isinstance(propeller, dict):
            raise SimSceneValidationError(f'{source}: {prop_label} must be a mapping when present')
        unknown = set(propeller) - _RING_PROPELLER_KEYS
        if unknown:
            raise SimSceneValidationError(
                f'{source}: {prop_label}: unknown keys: {sorted(unknown)}'
            )
        for key in ('length_deg', 'width_px'):
            if propeller.get(key) is None:
                raise SimSceneValidationError(f'{source}: {prop_label}.{key} is required')
            _check_optional_positive_number(
                propeller.get(key), f'{prop_label}.{key}', source=source
            )
        if propeller.get('contrast') is None:
            raise SimSceneValidationError(f'{source}: {prop_label}.contrast is required')
        _check_optional_number(propeller.get('contrast'), f'{prop_label}.contrast', source=source)


def _check_ring_feature(obj: dict[str, Any], *, index: int, source: str) -> None:
    """Validate one ``ring_system.features`` entry.

    The shape keys are kind-specific, per
    :data:`~spindoctor.sim.scene_schema.RING_KIND_SHAPE_KEYS`: ``width_px`` is
    required by the banded kinds (ringlet / gap / ramp) and rejected elsewhere,
    ``side`` is optional on the one-sided kinds (edge / ramp) and rejected
    elsewhere, and ``wavelength_px`` / ``density_wave_damping_px`` are the ``wave`` kind's
    required radial train parameters, rejected elsewhere.  A stray shape key
    on a kind that ignores it fails loudly rather than silently authoring a
    different feature.

    Parameters:
        obj: The feature mapping.
        index: The feature's index in the list (for error messages).
        source: Label used in error messages.

    Raises:
        SimSceneValidationError: On any unknown, missing, or invalid field.
    """
    label = f'ring_system.features[{index}]'
    unknown = set(obj) - _RING_FEATURE_KEYS
    if unknown:
        raise SimSceneValidationError(f'{source}: {label}: unknown keys: {sorted(unknown)}')
    _check_optional_str(obj.get('name'), f'{label}.name', source=source)
    kind = obj.get('kind')
    if kind not in _RING_FEATURE_KINDS:
        raise SimSceneValidationError(
            f'{source}: {label}.kind must be one of {sorted(_RING_FEATURE_KINDS)}; got {kind!r}'
        )
    tau = obj.get('tau')
    if tau is None:
        raise SimSceneValidationError(f'{source}: {label}.tau is required')
    _check_optional_nonnegative_number(tau, f'{label}.tau', source=source)
    rules = RING_KIND_SHAPE_KEYS[kind]
    for key in RING_SHAPE_KEYS:
        value = obj.get(key)
        rule = rules.get(key)
        if rule is None:
            if value is not None:
                raise SimSceneValidationError(
                    f'{source}: {label}.{key} is not allowed for kind {kind!r}'
                )
            continue
        if value is None:
            if rule == 'required':
                raise SimSceneValidationError(
                    f'{source}: {label}.{key} is required for kind {kind!r}'
                )
            continue
        if key == 'side':
            if value not in _RING_FEATURE_SIDES:
                raise SimSceneValidationError(
                    f"{source}: {label}.side must be 'in' or 'out'; got {value!r}"
                )
        else:
            _check_optional_positive_number(value, f'{label}.{key}', source=source)
    _check_ring_feature_orbit(obj.get('orbit'), label=label, source=source)
    _check_optional_bool(obj.get('navigable'), f'{label}.navigable', source=source)
    orbit_error = obj.get('orbit_error')
    if orbit_error is not None:
        error_label = f'{label}.orbit_error'
        if not isinstance(orbit_error, dict):
            raise SimSceneValidationError(f'{source}: {error_label} must be a mapping when present')
        unknown = set(orbit_error) - _RING_ORBIT_ERROR_KEYS
        if unknown:
            raise SimSceneValidationError(
                f'{source}: {error_label}: unknown keys: {sorted(unknown)}'
            )
        for key in _RING_ORBIT_ERROR_KEYS:
            _check_optional_number(orbit_error.get(key), f'{error_label}.{key}', source=source)
        _check_ring_orbit_error_bounds(
            obj.get('orbit') or {}, orbit_error, label=label, source=source
        )
    sigma = obj.get('declared_orbit_sigma')
    if sigma is not None:
        sigma_label = f'{label}.declared_orbit_sigma'
        if not isinstance(sigma, dict):
            raise SimSceneValidationError(f'{source}: {sigma_label} must be a mapping when present')
        unknown = set(sigma) - _RING_ORBIT_SIGMA_KEYS
        if unknown:
            raise SimSceneValidationError(
                f'{source}: {sigma_label}: unknown keys: {sorted(unknown)}'
            )
        for key in _RING_ORBIT_SIGMA_KEYS:
            _check_optional_nonnegative_number(
                sigma.get(key), f'{sigma_label}.{key}', source=source
            )
    _check_optional_nonnegative_number(obj.get('albedo'), f'{label}.albedo', source=source)
    phase_g = obj.get('phase_g')
    _check_optional_number(phase_g, f'{label}.phase_g', source=source)
    if phase_g is not None and not -1.0 < float(phase_g) < 1.0:
        raise SimSceneValidationError(
            f'{source}: {label}.phase_g must lie in (-1, 1); got {phase_g!r}'
        )


def _check_ring_orbit_error_bounds(
    orbit: dict[str, Any],
    orbit_error: dict[str, Any],
    *,
    label: str,
    source: str,
) -> None:
    """Bound a feature's planted orbit error to a renderable effective orbit.

    The renderer draws the catalog orbit displaced by the ``orbit_error``
    deltas, so the *effective* orbit must stay physical: a delta that drives
    the semimajor axis to zero or below, or the (zero-clamped) effective
    ``ae_px`` to the effective ``a_px`` or beyond, validated and then raised (or
    rendered garbage) deep inside the edge-radius math.  Both bounds are
    enforced here, on the same values the renderer will combine.

    Parameters:
        orbit: The feature's (already validated) ``orbit`` mapping.
        orbit_error: The feature's (already type-checked) ``orbit_error``
            mapping.
        label: The feature label for error messages.
        source: Label used in error messages.

    Raises:
        SimSceneValidationError: If the effective semimajor axis is not
            positive, or the effective eccentric amplitude reaches it.
    """
    a = float(scene_value(orbit, 'a_px', 0.0))
    ae = float(scene_value(orbit, 'ae_px', 0.0))
    effective_a = a + float(scene_value(orbit_error, 'delta_a_px', 0.0))
    if effective_a <= 0.0:
        raise SimSceneValidationError(
            f'{source}: {label}.orbit_error.delta_a_px pushes the effective '
            f'semimajor axis to {effective_a!r} (catalog a_px = {a!r}); the '
            f'rendered orbit needs a positive semimajor axis'
        )
    # The renderer clamps a negative effective ae at a circular edge, so only
    # the upper bound can go out of domain.
    effective_ae = max(0.0, ae + float(scene_value(orbit_error, 'delta_ae_px', 0.0)))
    if effective_ae >= effective_a:
        raise SimSceneValidationError(
            f'{source}: {label}.orbit_error pushes the effective eccentric '
            f'amplitude to {effective_ae!r} px with effective semimajor axis '
            f'{effective_a!r} px; the rendered edge eccentricity ae_px/a_px must '
            f'stay below 1'
        )


def _check_ring_feature_orbit(orbit: Any, *, label: str, source: str) -> None:
    """Validate one feature's ``orbit`` mapping (mode 1 + modes + edge wave).

    Every key names its unit.  Every orbital angle (``long_peri_deg``, a
    mode's ``peri_deg``, the edge wave's ``lam0_deg``) is a ring-plane
    longitude in degrees measured from the ascending node, and
    ``rate_peri_deg_per_day`` mode 1's precession rate (a mode's ``peri_deg``
    stays fixed).  The edge wave's
    ``edge_wave_damping_deg`` is in degrees of downstream longitude.  ``a_px``,
    ``ae_px``, and the ``amp_px`` amplitudes are radial distances, and the edge
    wave's ``wavelength_px`` an arc length, all in ring-plane pixels.  An
    m-mode's ``m`` is an integer of at least 2 (mode 1 is the base ellipse).

    Parameters:
        orbit: The ``orbit`` value (must be a mapping).
        label: The feature label for error messages.
        source: Label used in error messages.

    Raises:
        SimSceneValidationError: On any unknown, missing, or invalid field.
    """
    if not isinstance(orbit, dict):
        raise SimSceneValidationError(f'{source}: {label}.orbit is required (a mapping)')
    orbit_label = f'{label}.orbit'
    unknown = set(orbit) - _RING_FEATURE_ORBIT_KEYS
    if unknown:
        raise SimSceneValidationError(f'{source}: {orbit_label}: unknown keys: {sorted(unknown)}')
    if orbit.get('a_px') is None:
        raise SimSceneValidationError(f'{source}: {orbit_label}.a_px is required')
    _check_optional_positive_number(orbit.get('a_px'), f'{orbit_label}.a_px', source=source)
    _check_optional_nonnegative_number(orbit.get('ae_px'), f'{orbit_label}.ae_px', source=source)
    if orbit.get('ae_px') is not None and float(orbit['ae_px']) >= float(orbit['a_px']):
        raise SimSceneValidationError(
            f'{source}: {orbit_label}.ae_px must be less than orbit.a_px (the edge '
            f'eccentricity ae_px/a_px must stay below 1 for a closed elliptical '
            f'edge); got ae_px = {orbit["ae_px"]!r} with a_px = {orbit["a_px"]!r}'
        )
    for key in ('long_peri_deg', 'rate_peri_deg_per_day'):
        _check_optional_number(orbit.get(key), f'{orbit_label}.{key}', source=source)
    modes = orbit.get('modes')
    _check_optional_mapping_list(modes, f'{label}.orbit.modes', source=source)
    for mode_index, mode in enumerate(modes or []):
        mode_label = f'{label}.orbit.modes[{mode_index}]'
        unknown = set(mode) - _RING_ORBIT_MODE_KEYS
        if unknown:
            raise SimSceneValidationError(
                f'{source}: {mode_label}: unknown keys: {sorted(unknown)}'
            )
        m = mode.get('m')
        if isinstance(m, bool) or not isinstance(m, int) or m < 2:
            raise SimSceneValidationError(
                f'{source}: {mode_label}.m must be an integer >= 2; got {m!r}'
            )
        _check_optional_nonnegative_number(
            mode.get('amp_px'), f'{mode_label}.amp_px', source=source
        )
        _check_optional_number(mode.get('peri_deg'), f'{mode_label}.peri_deg', source=source)
    wave = orbit.get('edge_wave')
    if wave is not None:
        wave_label = f'{label}.orbit.edge_wave'
        if not isinstance(wave, dict):
            raise SimSceneValidationError(f'{source}: {wave_label} must be a mapping when present')
        unknown = set(wave) - _RING_EDGE_WAVE_KEYS
        if unknown:
            raise SimSceneValidationError(
                f'{source}: {wave_label}: unknown keys: {sorted(unknown)}'
            )
        _check_optional_nonnegative_number(
            wave.get('amp_px'), f'{wave_label}.amp_px', source=source
        )
        for key in ('wavelength_px', 'edge_wave_damping_deg'):
            value = wave.get(key)
            if value is None:
                raise SimSceneValidationError(f'{source}: {wave_label}.{key} is required')
            _check_optional_positive_number(value, f'{wave_label}.{key}', source=source)
        if float(wave['edge_wave_damping_deg']) > _RING_EDGE_WAVE_DAMPING_MAX_DEG:
            raise SimSceneValidationError(
                f'{source}: {wave_label}.edge_wave_damping_deg must be <= 360/pi, about '
                f'{_RING_EDGE_WAVE_DAMPING_MAX_DEG:.2f} degrees (the modular longitude wrap '
                f'leaves an upstream residual of amp_px * exp(-360/edge_wave_damping_deg); '
                f'the cap bounds it at exp(-pi), about 4.3% of amp_px); '
                f'got {wave["edge_wave_damping_deg"]!r}'
            )
        _check_optional_number(wave.get('lam0_deg'), f'{wave_label}.lam0_deg', source=source)
