"""Body-entry field validators for the sim-scene schema.

One ``bodies`` entry carries the largest key surface in the schema: the
idealized geometry, the mesh and crater shape knobs, and the body-appearance
truth blocks (pose scatter, limb relief and photometry, albedo texture, disc
texture, transits).  The checkers for that entry live here, beside their key
inventories; the scalar primitives they share with the other block checkers
stay in :mod:`spindoctor.sim.scene.rules_common`, which also documents the overall
validation contract (unknown keys fail, every violation raises
:class:`spindoctor.sim.scene.errors.SimSceneValidationError`).

:func:`spindoctor.sim.scene.validate.validate_sim_params` drives ``_check_body_object``
once per ``bodies`` entry.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from spindoctor.sim.instruments.passbands import has_passbands, solar_weighted_passband
from spindoctor.sim.render.bodies.haze import (
    CLOUD_BLOB_KEYS,
    HAZE_STRUCTURE_KEYS,
    SECTOR_REFERENCE_HALF_ANGLE_DEG,
)
from spindoctor.sim.render.bodies.photometry import PHOTOMETRIC_LAWS
from spindoctor.sim.render.bodies.relief import MAX_LIMB_RELIEF_RMS, MIN_LIMB_RELIEF_CORR_DEG
from spindoctor.sim.render.bodies.spec import CRATER_MAX_RADIUS_DEFAULT, CRATER_MIN_RADIUS_DEFAULT
from spindoctor.sim.render.bodies.titan_haze import (
    MAX_METHANE_BAND_FRACTION,
    MAX_UNCOVERED_FRACTION,
    TITAN_HAZE_MODEL,
    methane_band_fraction,
    uncovered_fraction,
)
from spindoctor.sim.scene.boundary import _BODY_IDEALIZED_KEYS
from spindoctor.sim.scene.errors import SimSceneValidationError
from spindoctor.sim.scene.fields import (
    ELLIPSOID_UNSUPPORTED_KEYS,
    MESH_UNSUPPORTED_KEYS,
    default_body_name,
    ellipsoid_unsupported_message,
    mesh_unsupported_message,
    scene_value,
)
from spindoctor.sim.scene.rules_common import (
    _check_optional_mapping_list,
    _check_optional_nonnegative_int,
    _check_optional_nonnegative_number,
    _check_optional_number,
    _check_optional_positive_number,
    _check_optional_str,
)


def _check_body_names_unique(bodies: list[dict[str, Any]], *, source: str) -> None:
    """Fail validation when two ``bodies`` entries share an effective name.

    The renderer keys its truth records (``body_masks`` mapping, inventory,
    render order) by each body's upper-cased name, defaulting an unnamed
    entry to its positional ``SIM-BODY-<index + 1>`` name.  Two entries that
    resolve to the same key would silently overwrite each other's truth
    records, so the collision fails here instead.  The navigator keys its predicted
    bodies by the name its view carries, where a ``nav_override`` ``name`` replaces
    the body's own, so the same rule holds for those names.

    Parameters:
        bodies: The scene's ``bodies`` list (entries already type-checked).
        source: Label used in error messages.

    Raises:
        SimSceneValidationError: If two entries share an effective name
            (case-insensitive, positional defaults included).
    """
    for navigator_side in (False, True):
        seen: dict[str, int] = {}
        for index, obj in enumerate(bodies):
            name_source = obj
            override = obj.get('nav_override')
            if navigator_side and isinstance(override, dict) and 'name' in override:
                name_source = override
            effective = str(scene_value(name_source, 'name', default_body_name(index))).upper()
            if effective in seen:
                side = 'navigator-side ' if navigator_side else ''
                raise SimSceneValidationError(
                    f'{source}: bodies[{seen[effective]}] and bodies[{index}] share the '
                    f'{side}effective name {effective!r} (names are case-insensitive and an '
                    f'unnamed body defaults to its positional SIM-BODY-<n> name); '
                    f'name-keyed truth records would collide, so give each body a '
                    f'unique name'
                )
            seen[effective] = index


# The idealized body keys that hold a number, and those that hold an integer.  A
# nav_override may set any idealized key, so its values get these same checks.
_IDEALIZED_NUMBER_KEYS: tuple[str, ...] = (
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
)
_IDEALIZED_INT_KEYS: tuple[str, ...] = ('mesh_n_lat', 'mesh_n_lon', 'mesh_seed')
_IDEALIZED_OTHER_KEYS: tuple[str, ...] = (
    'name',
    'shape_model',
    'mesh_detail_octaves',
    'pose_euler_deg',
)
# Every idealized key `_check_idealized_body_values` checks; a test holds this set equal
# to the keys the navigator's view treats as idealized.
_IDEALIZED_CHECKED_KEYS: frozenset[str] = frozenset(
    (*_IDEALIZED_NUMBER_KEYS, *_IDEALIZED_INT_KEYS, *_IDEALIZED_OTHER_KEYS)
)


def _check_int_key(obj: dict[str, Any], key: str, *, label: str, source: str) -> None:
    """Fail validation if ``obj[key]`` is present and isn't an integer (bools rejected)."""
    if key not in obj:
        return
    value = obj[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise SimSceneValidationError(f'{source}: {label}.{key} must be an integer')


def _check_idealized_body_values(obj: dict[str, Any], *, label: str, source: str) -> None:
    """Validate the values of the idealized keys a body entry or its override sets.

    These are the keys a ``nav_override`` may set, so a body's own values and its
    override's values pass the same checks.  Each check reads one key alone; the rules
    that compare keys with each other apply to the body entry only.

    Parameters:
        obj: The body entry, or its ``nav_override`` mapping.
        label: The mapping's key path for error messages, such as ``bodies[0]`` or
            ``bodies[0].nav_override``.
        source: Label used in error messages.

    Raises:
        SimSceneValidationError: If ``name`` isn't a string, ``shape_model`` isn't
            ``ellipsoid`` or ``polyhedral_mesh``, a number key isn't a finite number,
            ``sub_observer_lat_deg`` lies outside [-90, 90], a mesh integer key isn't an
            integer, ``mesh_detail_octaves`` isn't a non-negative integer, or
            ``pose_euler_deg`` isn't a list of 3 numbers.
    """
    _check_optional_str(obj.get('name'), f'{label}.name', source=source)
    shape_model = obj.get('shape_model')
    if shape_model is not None and shape_model not in ('ellipsoid', 'polyhedral_mesh'):
        raise SimSceneValidationError(
            f"{source}: {label}.shape_model must be 'ellipsoid' or 'polyhedral_mesh' "
            f'when present; got {shape_model!r}'
        )
    for key in _IDEALIZED_NUMBER_KEYS:
        _check_optional_number(obj.get(key), f'{label}.{key}', source=source)
    sub_observer_lat = obj.get('sub_observer_lat_deg')
    if sub_observer_lat is not None and not -90.0 <= float(sub_observer_lat) <= 90.0:
        raise SimSceneValidationError(
            f'{source}: {label}.sub_observer_lat_deg must lie in [-90, 90] degrees; '
            f'got {sub_observer_lat!r}'
        )
    for key in _IDEALIZED_INT_KEYS:
        _check_int_key(obj, key, label=label, source=source)
    _check_optional_nonnegative_int(
        obj.get('mesh_detail_octaves'), f'{label}.mesh_detail_octaves', source=source
    )
    pose = obj.get('pose_euler_deg')
    if pose is not None:
        if not isinstance(pose, (list, tuple)) or len(pose) != 3:
            raise SimSceneValidationError(
                f'{source}: {label}.pose_euler_deg must be a list of 3 angles when present'
            )
        _check_number_list(pose, f'{label}.pose_euler_deg', source=source)


def _check_body_object(obj: dict[str, Any], *, index: int, source: str) -> None:
    """Validate one ``bodies`` entry's field types."""
    label = f'bodies[{index}]'
    _check_idealized_body_values(obj, label=label, source=source)
    shape_model = obj.get('shape_model')
    if shape_model == 'polyhedral_mesh':
        unsupported = [key for key in MESH_UNSUPPORTED_KEYS if obj.get(key) is not None]
        if unsupported:
            raise SimSceneValidationError(
                f'{source}: {label}: {mesh_unsupported_message(unsupported)}'
            )
    else:
        unsupported = [key for key in ELLIPSOID_UNSUPPORTED_KEYS if obj.get(key) is not None]
        if unsupported:
            raise SimSceneValidationError(
                f'{source}: {label}: {ellipsoid_unsupported_message(unsupported)}'
            )
    for key in ('crater_fill', 'crater_relief_scale', 'anti_aliasing'):
        _check_optional_number(obj.get(key), f'{label}.{key}', source=source)
    _check_crater_shape_keys(obj, label=label, source=source)
    shading = obj.get('shading')
    if shading is not None and shading not in ('flat', 'gouraud'):
        raise SimSceneValidationError(
            f"{source}: {label}.shading must be 'flat' or 'gouraud' when present; got {shading!r}"
        )
    _check_pose_scatter(obj.get('pose_scatter'), label=label, source=source)
    _check_body_relief_and_photometry(obj, label=label, source=source)
    _check_albedo_texture(obj.get('albedo_texture'), label=label, source=source)
    _check_disc_texture(obj.get('disc_texture'), label=label, source=source)
    _check_transits(obj.get('transits'), label=label, source=source)
    _check_atmosphere(obj.get('atmosphere'), label=label, source=source)
    override = obj.get('nav_override')
    if override is not None:
        if not isinstance(override, dict):
            raise SimSceneValidationError(
                f'{source}: {label}.nav_override must be a mapping when present'
            )
        # The override expresses what the navigator BELIEVES about idealized
        # geometry, so only idealized body keys may appear in it.
        bad = set(override) - _BODY_IDEALIZED_KEYS
        if bad:
            raise SimSceneValidationError(
                f'{source}: {label}.nav_override may only override idealized body '
                f'keys; got {sorted(bad)}'
            )
        # The navigator reads an override value in place of the body's, so it
        # passes the check the body's own value does.
        _check_idealized_body_values(override, label=f'{label}.nav_override', source=source)


def _check_number_list(values: Any, key: str, *, source: str) -> None:
    """Validate that every entry of a fixed-length list is a number.

    Parameters:
        values: The list or tuple, already checked for its length.
        key: The list's label for error messages.
        source: Label used in error messages.

    Raises:
        SimSceneValidationError: If an entry is not a number.
    """
    for value in values:
        _check_optional_number(value, f'{key}[]', source=source)


def _check_crater_shape_keys(obj: dict[str, Any], *, label: str, source: str) -> None:
    """Validate one body's crater radius band and power-law exponent.

    The crater radii are sampled from ``p(R) ~ R**-alpha`` over
    ``[crater_min_radius, crater_max_radius]`` (fractions of semi-axis a), which
    needs a strictly positive band with ``min < max`` and ``alpha > 1``;
    out-of-domain values validated before and then crashed (or raised) deep
    inside the sampler, so the domain is enforced here instead.

    Parameters:
        obj: The body entry mapping.
        label: The ``bodies[<index>]`` label for error messages.
        source: Label used in error messages.

    Raises:
        SimSceneValidationError: On a non-positive radius, an inverted or
            empty radius band, or an exponent of 1 or less.
    """
    for key in ('crater_min_radius', 'crater_max_radius'):
        _check_optional_positive_number(obj.get(key), f'{label}.{key}', source=source)
    min_radius = obj.get('crater_min_radius')
    max_radius = obj.get('crater_max_radius')
    effective_min = CRATER_MIN_RADIUS_DEFAULT if min_radius is None else float(min_radius)
    effective_max = CRATER_MAX_RADIUS_DEFAULT if max_radius is None else float(max_radius)
    if (min_radius is not None or max_radius is not None) and effective_min >= effective_max:
        raise SimSceneValidationError(
            f'{source}: {label}: crater_min_radius must be less than '
            f'crater_max_radius; got {effective_min!r} >= {effective_max!r} '
            f'(an omitted bound takes the renderer default '
            f'{CRATER_MIN_RADIUS_DEFAULT} / {CRATER_MAX_RADIUS_DEFAULT})'
        )
    exponent = obj.get('crater_power_law_exponent')
    _check_optional_number(exponent, f'{label}.crater_power_law_exponent', source=source)
    if exponent is not None and float(exponent) <= 1.0:
        raise SimSceneValidationError(
            f'{source}: {label}.crater_power_law_exponent must exceed 1 (the '
            f'crater radius distribution p(R) ~ R**-alpha is only normalizable '
            f'for alpha > 1); got {exponent!r}'
        )


# The opposition-surge map's key inventory (a simple normalized exponential
# surge: amplitude plus angular e-folding width in degrees of phase).  The
# photometric-law vocabulary is the renderer's own PHOTOMETRIC_LAWS.
_OPPOSITION_SURGE_KEYS: frozenset[str] = frozenset({'amplitude', 'width_deg'})

# The per-frame pose scatter: a seeded Gaussian perturbation (sigma per
# Euler axis, degrees) added to the RENDERED mesh pose only; the navigator
# predicts the catalog pose.
_POSE_SCATTER_KEYS: frozenset[str] = frozenset({'sigma_deg'})


def _check_pose_scatter(value: Any, *, label: str, source: str) -> None:
    """Validate one body's ``pose_scatter`` map (per-frame pose perturbation)."""
    if value is None:
        return
    if not isinstance(value, dict):
        raise SimSceneValidationError(
            f'{source}: {label}.pose_scatter must be a mapping when present'
        )
    unknown = set(value) - _POSE_SCATTER_KEYS
    if unknown:
        raise SimSceneValidationError(
            f'{source}: {label}.pose_scatter: unknown keys: {sorted(unknown)}'
        )
    _check_optional_nonnegative_number(
        value.get('sigma_deg'), f'{label}.pose_scatter.sigma_deg', source=source
    )


def _check_body_relief_and_photometry(obj: dict[str, Any], *, label: str, source: str) -> None:
    """Validate one body's limb-relief and photometric truth keys."""
    _check_optional_nonnegative_number(
        obj.get('limb_relief_rms'), f'{label}.limb_relief_rms', source=source
    )
    # The bounds and their reasons live with the constants in
    # spindoctor.sim.render.bodies.relief.  The RMS cap is a limit of the scene
    # format: at the cap the relief surface stays about half the radius out from
    # the body center.  The correlation floor bounds the field's synthesis cost.
    # Each key's range is checked on its own, as every other key's is, so the floor
    # applies even to a body whose relief is off.
    relief_rms = obj.get('limb_relief_rms')
    if relief_rms is not None and float(relief_rms) > MAX_LIMB_RELIEF_RMS:
        raise SimSceneValidationError(
            f'{source}: {label}.limb_relief_rms is a fraction of the body radius and '
            f'must be at most {MAX_LIMB_RELIEF_RMS}; got {relief_rms}'
        )
    _check_optional_positive_number(
        obj.get('limb_relief_corr_deg'), f'{label}.limb_relief_corr_deg', source=source
    )
    relief_corr = obj.get('limb_relief_corr_deg')
    if relief_corr is not None and float(relief_corr) < MIN_LIMB_RELIEF_CORR_DEG:
        raise SimSceneValidationError(
            f'{source}: {label}.limb_relief_corr_deg must be at least '
            f'{MIN_LIMB_RELIEF_CORR_DEG} degrees; got {relief_corr}'
        )
    law = obj.get('photometric_law')
    if law is not None and law not in PHOTOMETRIC_LAWS:
        raise SimSceneValidationError(
            f'{source}: {label}.photometric_law must be one of '
            f'{sorted(PHOTOMETRIC_LAWS)}; got {law!r}'
        )
    _check_optional_positive_number(obj.get('minnaert_k'), f'{label}.minnaert_k', source=source)
    surge = obj.get('opposition_surge')
    if surge is None:
        return
    if not isinstance(surge, dict):
        raise SimSceneValidationError(
            f'{source}: {label}.opposition_surge must be a mapping when present'
        )
    unknown = set(surge) - _OPPOSITION_SURGE_KEYS
    if unknown:
        raise SimSceneValidationError(
            f'{source}: {label}.opposition_surge: unknown keys: {sorted(unknown)}'
        )
    _check_optional_nonnegative_number(
        surge.get('amplitude'), f'{label}.opposition_surge.amplitude', source=source
    )
    _check_optional_positive_number(
        surge.get('width_deg'), f'{label}.opposition_surge.width_deg', source=source
    )


# The exponential haze layer: scale height (required, positive) and
# reference-altitude tangent optical depth (required, non-negative; zero is a
# clear atmosphere that leaves the limb hard) define the column; the
# reference altitude, the Henyey-Greenstein asymmetry (forward-scattering
# when positive) and an optional second term's asymmetry and share, an
# optional detached-shell altitude, and the haze's
# single-scattering albedo (in [0, 1]) are optional.  The
# structure keys folded in from the renderer break the column's mirror
# symmetry (axis tilt, hemispheric and azimuthal falloff scaling, hemispheric
# brightness, an axial interior ramp, and clouds); all are optional.  Instead of the
# column's keys, ``haze_model: titan`` takes Titan's published haze, which depends on the
# scene's filter.
ATMOSPHERE_KEYS: frozenset[str] = (
    frozenset(
        {
            'haze_model',
            'scale_height_px',
            'tau_ref',
            'ref_altitude_px',
            'g',
            'g2',
            'g2_fraction',
            'detached_px',
            'albedo',
        }
    )
    | HAZE_STRUCTURE_KEYS
)
"""Every key one body's ``atmosphere`` block may carry.

Public because the boundary tests iterate it: the whole block is ONE entry of
the schema's truth-key inventory, so a key added here would otherwise be
invisible to the boundary suite's completeness check.
"""

# The column keys a named haze model sets, which its block may not also set.
_MODEL_SET_KEYS = frozenset(
    {'scale_height_px', 'tau_ref', 'ref_altitude_px', 'g', 'g2', 'g2_fraction', 'albedo'}
)

# The filter elements that pass a methane band or a methane window, by name.
_METHANE_FILTER_PREFIXES = ('MT', 'CB')

MIN_SECTOR_SHARPNESS_GRADIENT: float = -SECTOR_REFERENCE_HALF_ANGLE_DEG / 180.0
"""Exclusive lower bound on ``sector_sharpness_gradient``.

The gradient scales the haze falloff length by ``1 + g * azimuth /
reference``, and the azimuth runs to ``pi`` at the anti-sunward point, so a
``g`` at or below ``-reference / pi`` asks for a zero or negative length
there.  Bounding it here means the renderer never has to decide what a
negative atmosphere means.
"""


def _check_atmosphere(value: Any, *, label: str, source: str) -> None:
    """Validate one body's ``atmosphere`` map (exponential haze layer)."""
    if value is None:
        return
    if not isinstance(value, dict):
        raise SimSceneValidationError(
            f'{source}: {label}.atmosphere must be a mapping when present'
        )
    unknown = set(value) - ATMOSPHERE_KEYS
    if unknown:
        raise SimSceneValidationError(
            f'{source}: {label}.atmosphere: unknown keys: {sorted(unknown)}'
        )
    if 'haze_model' in value:
        _check_atmosphere_model(value, label=label, source=source)
        _check_haze_structure(value, label=label, source=source)
        return
    for required in ('scale_height_px', 'tau_ref'):
        if value.get(required) is None:
            raise SimSceneValidationError(f'{source}: {label}.atmosphere.{required} is required')
    _check_optional_positive_number(
        value.get('scale_height_px'), f'{label}.atmosphere.scale_height_px', source=source
    )
    _check_optional_nonnegative_number(
        value.get('tau_ref'), f'{label}.atmosphere.tau_ref', source=source
    )
    _check_optional_nonnegative_number(
        value.get('ref_altitude_px'), f'{label}.atmosphere.ref_altitude_px', source=source
    )
    _check_optional_positive_number(
        value.get('detached_px'), f'{label}.atmosphere.detached_px', source=source
    )
    albedo = value.get('albedo')
    _check_optional_nonnegative_number(albedo, f'{label}.atmosphere.albedo', source=source)
    if albedo is not None and float(albedo) > 1.0:
        raise SimSceneValidationError(
            f'{source}: {label}.atmosphere.albedo must lie in [0, 1]; got {albedo!r}'
        )
    g = value.get('g')
    _check_optional_number(g, f'{label}.atmosphere.g', source=source)
    if g is not None and not -1.0 < float(g) < 1.0:
        raise SimSceneValidationError(
            f'{source}: {label}.atmosphere.g must lie in (-1, 1); got {g!r}'
        )
    _check_second_phase_term(value, label=label, source=source)
    _check_haze_structure(value, label=label, source=source)


def _check_second_phase_term(value: dict[str, Any], *, label: str, source: str) -> None:
    """Validate the second term of a two-term Henyey-Greenstein phase function.

    ``g2`` and ``g2_fraction`` come together or not at all: one without the other
    states half a phase function.
    """
    g2 = value.get('g2')
    fraction = value.get('g2_fraction')
    if (g2 is None) != (fraction is None):
        raise SimSceneValidationError(
            f'{source}: {label}.atmosphere.g2 and {label}.atmosphere.g2_fraction '
            'must be given together'
        )
    if g2 is None:
        return
    _check_optional_number(g2, f'{label}.atmosphere.g2', source=source)
    if not -1.0 < float(g2) < 1.0:
        raise SimSceneValidationError(
            f'{source}: {label}.atmosphere.g2 must lie in (-1, 1); got {g2!r}'
        )
    _check_optional_nonnegative_number(fraction, f'{label}.atmosphere.g2_fraction', source=source)
    if fraction is not None and float(fraction) > 1.0:
        raise SimSceneValidationError(
            f'{source}: {label}.atmosphere.g2_fraction must lie in [0, 1]; got {fraction!r}'
        )


def _check_atmosphere_model(value: dict[str, Any], *, label: str, source: str) -> None:
    """Validate an ``atmosphere`` block that takes a published haze by name.

    The model sets the column, so a block that names one may not also set the
    column's own keys.  Nor may it add a detached shell: the shell's depth is a
    multiple of the column's reference depth, which for Titan's haze would make the
    shell about 15 tangent optical depths deep, far deeper than Titan's detached
    layer.
    """
    haze_model = value.get('haze_model')
    if haze_model != TITAN_HAZE_MODEL:
        raise SimSceneValidationError(
            f'{source}: {label}.atmosphere.haze_model must be {TITAN_HAZE_MODEL!r}; '
            f'got {haze_model!r}'
        )
    stated = sorted(set(value) & _MODEL_SET_KEYS)
    if stated:
        raise SimSceneValidationError(
            f'{source}: {label}.atmosphere.haze_model sets the haze, so the block may not also '
            f'set {stated}'
        )
    if 'detached_px' in value:
        raise SimSceneValidationError(
            f"{source}: {label}.atmosphere.detached_px cannot join a haze_model: the shell's "
            "depth follows the column's reference depth, which would make it as opaque as "
            "Titan's main haze"
        )


def check_model_atmosphere_filters(
    sim_params: Mapping[str, Any], *, instrument: str, source: str
) -> None:
    """Refuse a Titan haze in a filter the haze is not modeled in.

    Titan's haze depends on wavelength, so it needs the camera's passbands, and its
    model leaves out methane absorption and covers 430 to 1078 nm.

    Parameters:
        sim_params: The scene.
        instrument: The scene's instrument.
        source: Label used in error messages.

    Raises:
        SimSceneValidationError: If a body takes Titan's haze and the camera has no
            passband table, or the scene's filter is a methane-band or methane-window
            filter, or too much of its light falls outside the modeled span.
    """
    hazy = [
        body
        for body in sim_params.get('bodies') or []
        if isinstance(body.get('atmosphere'), dict) and 'haze_model' in body['atmosphere']
    ]
    if not hazy:
        return
    if not has_passbands(instrument):
        raise SimSceneValidationError(
            f"{source}: Titan's haze depends on wavelength, and sim instrument "
            f'{instrument!r} has no passband table'
        )
    detector_filter = (sim_params.get('detector') or {}).get('filter')
    passband = solar_weighted_passband(
        instrument, None if detector_filter is None else str(detector_filter)
    )
    # Methane-band and methane-window filters are out of scope (#1189).
    in_methane = any(
        element.startswith(_METHANE_FILTER_PREFIXES) for element in passband.filter_name.split('+')
    )
    if in_methane:
        raise SimSceneValidationError(
            f"{source}: Titan's haze is not modeled in methane-band or methane-window "
            f'filters, whose light methane absorbs or lets through to the surface; '
            f'the scene uses {passband.filter_name!r}'
        )
    in_bands = methane_band_fraction(passband)
    if in_bands > MAX_METHANE_BAND_FRACTION:
        raise SimSceneValidationError(
            f"{source}: Titan's haze is not modeled where methane absorbs, and "
            f'{in_bands:.0%} of the light of {passband.filter_name!r} falls in the cores of '
            f'the methane bands'
        )
    outside = uncovered_fraction(passband)
    if outside > MAX_UNCOVERED_FRACTION:
        raise SimSceneValidationError(
            f"{source}: Titan's haze is modeled from 430 to 1078 nm, and "
            f'{outside:.0%} of the light of {passband.filter_name!r} lies outside that span'
        )


def _check_haze_structure(value: dict[str, Any], *, label: str, source: str) -> None:
    """Validate the symmetry-breaking structure keys of an ``atmosphere`` map.

    The amplitudes and the axis tilt are plain numbers (a negative amplitude
    darkens, a negative tilt rotates the other way), and the falloff ratio
    must be positive because it multiplies a length.  The sharpness gradient
    multiplies that same length by ``1 + g * azimuth / reference``, and the
    azimuth reaches ``pi`` at the anti-sunward point, so keeping the length
    positive everywhere on the limb bounds ``g`` below at
    ``-reference / pi`` -- one third for the 60 degree reference sector.

    Parameters:
        value: One body's ``atmosphere`` mapping.
        label: Body label used in error messages.
        source: Scene source used in error messages.
    """
    for key in ('interior_ramp_amplitude', 'ns_asymmetry_amplitude', 'axis_tilt_deg'):
        _check_optional_number(value.get(key), f'{label}.atmosphere.{key}', source=source)
    _check_optional_positive_number(
        value.get('ns_falloff_ratio'), f'{label}.atmosphere.ns_falloff_ratio', source=source
    )
    gradient = value.get('sector_sharpness_gradient')
    _check_optional_number(gradient, f'{label}.atmosphere.sector_sharpness_gradient', source=source)
    if gradient is not None and float(gradient) <= MIN_SECTOR_SHARPNESS_GRADIENT:
        raise SimSceneValidationError(
            f'{source}: {label}.atmosphere.sector_sharpness_gradient must exceed '
            f'{MIN_SECTOR_SHARPNESS_GRADIENT:.4f} (below it the falloff length it '
            f'scales turns negative on the anti-sunward limb); got {gradient!r}'
        )
    blobs = value.get('cloud_blobs')
    if blobs is None:
        return
    _check_optional_mapping_list(blobs, f'{label}.atmosphere.cloud_blobs', source=source)
    for index, blob in enumerate(blobs):
        blob_label = f'{label}.atmosphere.cloud_blobs[{index}]'
        unknown = set(blob) - CLOUD_BLOB_KEYS
        if unknown:
            raise SimSceneValidationError(
                f'{source}: {blob_label}: unknown keys: {sorted(unknown)}'
            )
        for required in ('sigma_px', 'amplitude'):
            if blob.get(required) is None:
                raise SimSceneValidationError(f'{source}: {blob_label}.{required} is required')
        _check_optional_positive_number(
            blob.get('sigma_px'), f'{blob_label}.sigma_px', source=source
        )
        _check_optional_number(blob.get('amplitude'), f'{blob_label}.amplitude', source=source)
        center = blob.get('center_vu')
        if center is None:
            continue
        if not isinstance(center, (list, tuple)) or len(center) != 2:
            raise SimSceneValidationError(
                f'{source}: {blob_label}.center_vu must be a list of 2 offsets when present'
            )
        _check_number_list(center, f'{blob_label}.center_vu', source=source)


# The multiplicative albedo texture: a band-limited noise field (rms +
# correlation length in detector pixels on the disc) plus discrete circular
# spots in the body-polar surface frame (pole along semi-axis a; longitude 90 deg
# is the sub-observer meridian).
_ALBEDO_TEXTURE_KEYS: frozenset[str] = frozenset({'rms', 'corr_px', 'spots'})
_SURFACE_SPOT_KEYS: frozenset[str] = frozenset(
    {'lat_deg', 'lon_deg', 'radius_deg', 'albedo_factor'}
)


def _check_surface_spot_list(value: Any, *, key: str, source: str) -> None:
    """Validate one spots/storms list (circular multiplicative albedo marks)."""
    if value is None:
        return
    _check_optional_mapping_list(value, key, source=source)
    for index, spot in enumerate(value):
        label = f'{key}[{index}]'
        unknown = set(spot) - _SURFACE_SPOT_KEYS
        if unknown:
            raise SimSceneValidationError(f'{source}: {label}: unknown keys: {sorted(unknown)}')
        _check_optional_number(spot.get('lat_deg'), f'{label}.lat_deg', source=source)
        _check_optional_number(spot.get('lon_deg'), f'{label}.lon_deg', source=source)
        _check_optional_positive_number(
            spot.get('radius_deg'), f'{label}.radius_deg', source=source
        )
        _check_optional_nonnegative_number(
            spot.get('albedo_factor'), f'{label}.albedo_factor', source=source
        )


def _check_albedo_texture(value: Any, *, label: str, source: str) -> None:
    """Validate one body's ``albedo_texture`` map (noise field + spots)."""
    if value is None:
        return
    if not isinstance(value, dict):
        raise SimSceneValidationError(
            f'{source}: {label}.albedo_texture must be a mapping when present'
        )
    unknown = set(value) - _ALBEDO_TEXTURE_KEYS
    if unknown:
        raise SimSceneValidationError(
            f'{source}: {label}.albedo_texture: unknown keys: {sorted(unknown)}'
        )
    _check_optional_nonnegative_number(
        value.get('rms'), f'{label}.albedo_texture.rms', source=source
    )
    _check_optional_positive_number(
        value.get('corr_px'), f'{label}.albedo_texture.corr_px', source=source
    )
    _check_surface_spot_list(value.get('spots'), key=f'{label}.albedo_texture.spots', source=source)


# The giant-planet disc texture: a low-frequency latitude-banded
# multiplicative pattern (cosine in body-polar latitude) plus discrete
# storm ovals in the body-polar frame (the pole is the direction of the
# body's semi-axis a, so bands rotate with the rotation_z pose).
_DISC_TEXTURE_KEYS: frozenset[str] = frozenset(
    {'band_amplitude', 'band_wavenumber', 'band_phase_deg', 'storms'}
)
# One transits entry: a transiting moon disc and/or its cast shadow disc,
# both texture on the rendered parent disc (offsets from the body center
# and radii in detector pixels).
_TRANSIT_ENTRY_KEYS: frozenset[str] = frozenset({'moon', 'shadow'})
_TRANSIT_MOON_KEYS: frozenset[str] = frozenset({'dv_px', 'du_px', 'radius_px', 'albedo_factor'})
_TRANSIT_SHADOW_KEYS: frozenset[str] = frozenset({'dv_px', 'du_px', 'radius_px', 'darkness'})


def _check_disc_texture(value: Any, *, label: str, source: str) -> None:
    """Validate one body's ``disc_texture`` map (bands + storm ovals)."""
    if value is None:
        return
    if not isinstance(value, dict):
        raise SimSceneValidationError(
            f'{source}: {label}.disc_texture must be a mapping when present'
        )
    unknown = set(value) - _DISC_TEXTURE_KEYS
    if unknown:
        raise SimSceneValidationError(
            f'{source}: {label}.disc_texture: unknown keys: {sorted(unknown)}'
        )
    _check_optional_nonnegative_number(
        value.get('band_amplitude'), f'{label}.disc_texture.band_amplitude', source=source
    )
    _check_optional_nonnegative_number(
        value.get('band_wavenumber'), f'{label}.disc_texture.band_wavenumber', source=source
    )
    _check_optional_number(
        value.get('band_phase_deg'), f'{label}.disc_texture.band_phase_deg', source=source
    )
    _check_surface_spot_list(value.get('storms'), key=f'{label}.disc_texture.storms', source=source)


def _check_transit_disc(value: Any, allowed: frozenset[str], *, label: str, source: str) -> None:
    """Validate one transit sub-map (the moon disc or the shadow disc)."""
    if value is None:
        return
    if not isinstance(value, dict):
        raise SimSceneValidationError(f'{source}: {label} must be a mapping when present')
    unknown = set(value) - allowed
    if unknown:
        raise SimSceneValidationError(f'{source}: {label}: unknown keys: {sorted(unknown)}')
    _check_optional_number(value.get('dv_px'), f'{label}.dv_px', source=source)
    _check_optional_number(value.get('du_px'), f'{label}.du_px', source=source)
    _check_optional_positive_number(value.get('radius_px'), f'{label}.radius_px', source=source)
    _check_optional_nonnegative_number(
        value.get('albedo_factor'), f'{label}.albedo_factor', source=source
    )
    darkness = value.get('darkness')
    _check_optional_nonnegative_number(darkness, f'{label}.darkness', source=source)
    if darkness is not None and float(darkness) > 1.0:
        raise SimSceneValidationError(
            f'{source}: {label}.darkness must lie in [0, 1]; got {darkness!r}'
        )


def _check_transits(value: Any, *, label: str, source: str) -> None:
    """Validate one body's ``transits`` list (moon and/or shadow discs)."""
    if value is None:
        return
    _check_optional_mapping_list(value, f'{label}.transits', source=source)
    for index, entry in enumerate(value):
        entry_label = f'{label}.transits[{index}]'
        unknown = set(entry) - _TRANSIT_ENTRY_KEYS
        if unknown:
            raise SimSceneValidationError(
                f'{source}: {entry_label}: unknown keys: {sorted(unknown)}'
            )
        if entry.get('moon') is None and entry.get('shadow') is None:
            raise SimSceneValidationError(
                f'{source}: {entry_label} needs a moon and/or a shadow map'
            )
        _check_transit_disc(
            entry.get('moon'), _TRANSIT_MOON_KEYS, label=f'{entry_label}.moon', source=source
        )
        _check_transit_disc(
            entry.get('shadow'),
            _TRANSIT_SHADOW_KEYS,
            label=f'{entry_label}.shadow',
            source=source,
        )
