"""A key a scene leaves out renders exactly as its declared default.

Each typed scene key that declares one default value is tested the same way.  A base
scene sets every such key to a value other than its default, so the render reads each
of them.  The test then renders the base with the key left out and with the key set to
its declared default, and the two frames, and the star records the render makes, must
match to the last bit.  It also checks that the base value renders differently from the
default, so a key the render ignores cannot pass.  Some defaults are constants that the
renderer imports from the scene types; the rest are values the renderer writes where it
reads the key, and this test is what holds those to the declarations.  The keys of a body's
``nav_override`` are left out: the renderer never reads them, and they are the body's
own idealized keys, which the test covers on the body.
"""

import copy
import dataclasses
from typing import Any, get_args, get_origin, get_type_hints

import numpy as np
import pytest
from tests.spindoctor.sim.scenes import disc_body, frame_scene, ring_system_block, star_entry

from spindoctor.sim.render.driver import render_scene
from spindoctor.sim.scene.declare import declaration_of
from spindoctor.sim.scene.truth import Scene
from spindoctor.support.types import NDArrayFloatType

# A key path into a scene: mapping keys, and 0 for the first entry of a list.
Path = tuple[str | int, ...]

_RINGLET = {
    'name': 'P',
    'kind': 'ringlet',
    'tau': 2.0,
    'width_px': 4.0,
    'orbit': {'a_px': 20.0, 'ae_px': 3.0, 'long_peri_deg': 20.0, 'rate_peri_deg_per_day': 15.0},
}

# Every key with a declared default is set here to another value, and every key that
# decides whether such a key matters (a ghost's amplitude, a star in view) is on.
_BASE = frame_scene(
    size=48,
    exposure_sec=0.5,
    time=864000.0,
    ring_epoch=345600.0,
    offset_v=1.3,
    offset_u=-0.7,
    offset_rotation_deg=3.0,
    star_catalog_scatter_px=0.4,
    oversample=1,
    bodies=[disc_body(30.0, 18.0, 6.0, range_km=1.0e6)],
    stars=[
        star_entry(
            10.0,
            38.0,
            7.0,
            catalog_name='TYCHO',
            spectral_class='K5',
            move_v=1.5,
            move_u=-0.8,
            navigable=False,
            catalog_error_v=0.4,
            catalog_error_u=-0.3,
            companion={'sep_px': 2.0, 'delta_mag': 1.0, 'angle_deg': 30.0},
            delta_mag=0.5,
        ),
        star_entry(40.0, 40.0, 8.0),
    ],
    ring_system=ring_system_block(24.0, 24.0, [_RINGLET], range_km=2.0e6),
    spk_error={'dv_px': 0.8, 'du_px': -0.6, 'reference_range_km': 1.0e6},
    optics={
        'psf': {'sigma_v': 1.0, 'w': 0.2, 'r0': 2.5, 'n': 2.5},
        'smear': [{'dv_px': 1.5, 'du_px': 0.7, 'object_class': 'stars'}],
        'distortion': {'k1': 0.02, 'k2': 0.01, 'nonradial_rms_px': 0.3},
        'ghosts': [{'dv_px': 4.0, 'du_px': 3.0, 'amplitude': 0.1, 'defocus_sigma': 1.5}],
        'stray_light': {'amplitude': 0.05, 'direction_deg': 30.0, 'model': 'radial'},
    },
    detector={'binning': 2},
)

# The spot or storm every surface-mark list of the body base holds.
_MARK = {'lat_deg': 10.0, 'lon_deg': 80.0, 'radius_deg': 15.0, 'albedo_factor': 0.5}

# The base of the body keys: one large ellipsoid that sets every ellipsoid key to a value
# other than its default, alone in its frame so every key it sets shows.
_BODY_BASE = frame_scene(
    size=64,
    oversample=1,
    bodies=[
        {
            'name': 'B',
            'center_v': 32.0,
            'center_u': 32.0,
            'semi_a_px': 20.0,
            'semi_b_px': 16.0,
            'semi_c_px': 14.0,
            'rotation_z': 20.0,
            'sub_observer_lat_deg': 15.0,
            'illumination_angle': 30.0,
            'phase_angle': 40.0,
            'crater_fill': 0.5,
            'crater_min_radius': 0.08,
            'crater_max_radius': 0.2,
            'crater_power_law_exponent': 2.5,
            'crater_relief_scale': 0.8,
            'anti_aliasing': 0.5,
            'limb_relief_rms': 0.02,
            'limb_relief_corr_deg': 10.0,
            'photometric_law': 'minnaert',
            'minnaert_k': 0.8,
            'opposition_surge': {'amplitude': 0.5, 'width_deg': 10.0},
            'albedo_texture': {'rms': 0.1, 'corr_px': 8.0, 'spots': [dict(_MARK)]},
            'disc_texture': {
                'band_amplitude': 0.2,
                'band_wavenumber': 5.0,
                'band_phase_deg': 30.0,
                'storms': [dict(_MARK, lon_deg=100.0)],
            },
            'transits': [
                {
                    'moon': {'dv_px': 3.0, 'du_px': -2.0, 'radius_px': 4.0, 'albedo_factor': 0.6},
                    'shadow': {'dv_px': -4.0, 'du_px': 5.0, 'radius_px': 3.0, 'darkness': 0.5},
                }
            ],
            'atmosphere': {
                'scale_height_px': 2.0,
                'tau_ref': 1.0,
                'ref_altitude_px': 1.0,
                'g': 0.3,
                'albedo': 0.4,
                'interior_ramp_amplitude': 0.05,
                'ns_asymmetry_amplitude': 0.1,
                'ns_falloff_ratio': 1.3,
                'axis_tilt_deg': 10.0,
                'sector_sharpness_gradient': 0.2,
            },
        }
    ],
)

# A mesh body, which the mesh keys need: the body base's geometry and lighting, drawn
# from a mesh.
_MESH_BODY = {
    'name': 'B',
    'center_v': 32.0,
    'center_u': 32.0,
    'semi_a_px': 20.0,
    'semi_b_px': 16.0,
    'semi_c_px': 14.0,
    'illumination_angle': 30.0,
    'phase_angle': 40.0,
    'shape_model': 'polyhedral_mesh',
}
# The mesh body with every mesh key set to a value other than its default.
_FULL_MESH_BODY = _MESH_BODY | {
    'mesh_lumpiness': 0.2,
    'mesh_n_lat': 24,
    'mesh_n_lon': 48,
    'mesh_seed': 3,
    'mesh_detail_octaves': 1,
    'pose_euler_deg': [10.0, 20.0, 30.0],
    'shading': 'gouraud',
    'pose_scatter': {'sigma_deg': 2.0},
}

# The keys a key needs set before the render reads it, where the base cannot set them
# for every key at once.  The contour step is read only by the contour quantization, and
# the ramp direction only by the linear stray-light model.
_CONTEXT: dict[Path, dict[Path, Any]] = {
    ('detector', 'contour_step'): {
        ('detector', 'quantization'): 'contour_8bit',
        ('detector', 'contour_step'): 5,
    },
    ('optics', 'stray_light', 'direction_deg'): {('optics', 'stray_light', 'model'): 'linear'},
    # A mesh body writing the default shape keeps every key an ellipsoid accepts.
    ('bodies', 0, 'shape_model'): {('bodies', 0): _MESH_BODY},
    **{
        ('bodies', 0, key): {('bodies', 0): _FULL_MESH_BODY}
        for key in (
            'mesh_lumpiness',
            'mesh_n_lat',
            'mesh_n_lon',
            'mesh_seed',
            'mesh_detail_octaves',
            'shading',
        )
    },
    ('bodies', 0, 'pose_scatter', 'sigma_deg'): {('bodies', 0): _FULL_MESH_BODY},
}


def _record(annotation: Any) -> tuple[type | None, bool]:
    """The typed block a field holds, and whether it holds a list of them."""
    members = [member for member in get_args(annotation) if member is not type(None)]
    inner = members[0] if len(members) == 1 else annotation
    is_list = get_origin(inner) is tuple
    if is_list:
        inner = get_args(inner)[0]
    if isinstance(inner, type) and dataclasses.is_dataclass(inner):
        return inner, is_list
    return None, False


def _defaulted(cls: type, prefix: Path = ()) -> list[tuple[Path, Any]]:
    """Every key of a typed block, and of the blocks in it, that declares a default."""
    found: list[tuple[Path, Any]] = []
    hints = get_type_hints(cls, include_extras=True)
    for item in dataclasses.fields(cls):
        name = item.name
        if name == 'nav_override':
            continue
        path = (*prefix, name)
        declared = declaration_of(item)
        if declared.default is not None:
            found.append((path, declared.default))
        record, is_list = _record(hints[name])
        if record is not None:
            found += _defaulted(record, (*path, 0) if is_list else path)
    return found


_DEFAULTED = _defaulted(Scene)


def _with(scene: dict[str, Any], path: Path, value: Any) -> dict[str, Any]:
    """A copy of a scene with the key at ``path`` set to ``value``."""
    copied = copy.deepcopy(scene)
    holder: Any = copied
    for key in path[:-1]:
        holder = holder[key]
    holder[path[-1]] = value
    return copied


def _without(scene: dict[str, Any], path: Path) -> dict[str, Any]:
    """A copy of a scene with the key at ``path`` left out."""
    copied = copy.deepcopy(scene)
    holder: Any = copied
    for key in path[:-1]:
        holder = holder[key]
    del holder[path[-1]]
    return copied


def _render(scene: dict[str, Any]) -> tuple[NDArrayFloatType, str]:
    """The frame a scene renders, and its star records written out."""
    image, truth = render_scene(scene, source='defaults')
    records = repr([vars(star) for star in truth['stars']]) + repr(truth['star_info'])
    return image, records


def _same(first: tuple[NDArrayFloatType, str], second: tuple[NDArrayFloatType, str]) -> bool:
    """Whether two renders give the same frame and the same star records."""
    return bool(np.array_equal(first[0], second[0])) and first[1] == second[1]


def _base_for(path: Path) -> dict[str, Any]:
    """The base scene with the keys the key at ``path`` needs to be read."""
    scene = _BODY_BASE if path[0] == 'bodies' else _BASE
    for needed, value in _CONTEXT.get(path, {}).items():
        scene = _with(scene, needed, value)
    return scene


def test_every_key_with_a_default_is_tested() -> None:
    """The walk finds each typed block's defaulted keys, nested list entries included."""
    paths = {path for path, _ in _DEFAULTED}
    assert ('optics', 'ghosts', 0, 'defocus_sigma') in paths


def test_the_walk_reaches_the_blocks_of_a_body() -> None:
    """The walk finds a key inside a body's nested blocks."""
    paths = {path for path, _ in _DEFAULTED}
    assert ('bodies', 0, 'transits', 0, 'shadow', 'darkness') in paths


def test_the_walk_leaves_out_the_navigation_override() -> None:
    """No key the walk finds lies inside a body's nav_override."""
    inside = [path for path, _ in _DEFAULTED if 'nav_override' in path]
    assert inside == []


@pytest.mark.parametrize(
    ('path', 'default'),
    _DEFAULTED,
    ids=['.'.join(str(key) for key in path) for path, _ in _DEFAULTED],
)
def test_a_left_out_key_renders_as_its_declared_default(path: Path, default: Any) -> None:
    """Leaving a key out renders the same frame as writing its declared default."""
    base = _base_for(path)
    declared = _render(_with(base, path, default))
    assert not _same(_render(base), declared), 'the base scene does not read the key'
    assert _same(_render(_without(base, path)), declared)


def _left_out_matches(base: dict[str, Any], path: Path, value: Any, *, other: Any) -> None:
    """Assert a key left out renders as ``value`` and differently from ``other``."""
    left_out = _render(_without(base, path))
    assert _same(left_out, _render(_with(base, path, value)))
    assert not _same(left_out, _render(_with(base, path, other)))


def test_a_left_out_semi_axis_c_renders_as_the_smaller_of_a_and_b() -> None:
    """A body without semi-axis c draws it as the smaller of semi-axes a and b."""
    path: Path = ('bodies', 0, 'semi_c_px')
    _left_out_matches(_BODY_BASE, path, 16.0, other=10.0)


def test_a_left_out_psf_window_renders_as_eleven_by_eleven() -> None:
    """A star without a PSF window gets the 11 by 11 window."""
    path: Path = ('stars', 0, 'psf_size')
    _left_out_matches(_with(_BASE, path, [13, 13]), path, [11, 11], other=[15, 15])


def test_a_left_out_mesh_pose_renders_as_no_turn() -> None:
    """A mesh body without Euler angles is posed at zero on every axis."""
    base = _with(_BODY_BASE, ('bodies', 0), _FULL_MESH_BODY)
    _left_out_matches(
        base, ('bodies', 0, 'pose_euler_deg'), [0.0, 0.0, 0.0], other=[0.0, 0.0, 40.0]
    )


def test_a_left_out_cloud_center_renders_at_the_body_center() -> None:
    """A cloud without a center is drawn at the body center."""
    cloud: dict[str, Any] = {'sigma_px': 3.0, 'amplitude': 0.3}
    base = _with(
        _BODY_BASE, ('bodies', 0, 'atmosphere', 'cloud_blobs'), [{**cloud, 'center_vu': [1.0, 1.0]}]
    )
    path: Path = ('bodies', 0, 'atmosphere', 'cloud_blobs', 0, 'center_vu')
    _left_out_matches(base, path, [0.0, 0.0], other=[5.0, 4.0])
