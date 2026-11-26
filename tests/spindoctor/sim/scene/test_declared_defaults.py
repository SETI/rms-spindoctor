"""A key a scene leaves out renders exactly as its declared default.

Each typed scene key that declares one default value is tested the same way.  A base
scene sets every such key to a value other than its default, so the render reads each
of them.  The test then renders the base with the key left out and with the key set to
its declared default, and the two frames must match to the last bit.  It also checks
that the base value renders differently from the default, so a key the render ignores
cannot pass.  Some defaults are constants that the renderer imports from the scene
types; the rest are values the renderer writes where it reads the key, and this test
is what holds those to the declarations.
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
    stars=[star_entry(10.0, 38.0, 7.0), star_entry(40.0, 40.0, 8.0)],
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

# The keys a key needs set before the render reads it, where the base cannot set them
# for every key at once.  The contour step is read only by the contour quantization, and
# the ramp direction only by the linear stray-light model.
_CONTEXT: dict[Path, dict[Path, Any]] = {
    ('detector', 'contour_step'): {
        ('detector', 'quantization'): 'contour_8bit',
        ('detector', 'contour_step'): 5,
    },
    ('optics', 'stray_light', 'direction_deg'): {('optics', 'stray_light', 'model'): 'linear'},
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


def _render(scene: dict[str, Any]) -> NDArrayFloatType:
    """The frame a scene renders."""
    image, _ = render_scene(scene, source='defaults')
    return image


def _base_for(path: Path) -> dict[str, Any]:
    """The base scene with the keys the key at ``path`` needs to be read."""
    scene = _BASE
    for needed, value in _CONTEXT.get(path, {}).items():
        scene = _with(scene, needed, value)
    return scene


def test_every_key_with_a_default_is_tested() -> None:
    """The walk finds each typed block's defaulted keys, nested list entries included."""
    paths = {path for path, _ in _DEFAULTED}
    assert ('optics', 'ghosts', 0, 'defocus_sigma') in paths


@pytest.mark.parametrize(
    ('path', 'default'),
    _DEFAULTED,
    ids=['.'.join(str(key) for key in path) for path, _ in _DEFAULTED],
)
def test_a_left_out_key_renders_as_its_declared_default(path: Path, default: Any) -> None:
    """Leaving a key out renders the same frame as writing its declared default."""
    base = _base_for(path)
    declared = _render(_with(base, path, default))
    assert not np.array_equal(_render(base), declared), 'the base scene does not read the key'
    assert np.array_equal(_render(_without(base, path)), declared)
