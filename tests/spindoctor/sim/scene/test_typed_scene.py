"""The typed scene: its declarations, its boundary classes, its ranges, and its reference.

Each test checks one seam the typed scene creates: that every field declares itself,
that a field's boundary class agrees with the type that declares it, that a detector
value has the catalog's range, that ``parse_scene`` returns the typed scene and leaves
the mapping alone, and that the generated reference lists every typed key.
"""

import dataclasses
import types
from typing import Any, Union, get_args, get_origin, get_type_hints

import pytest
from tests.spindoctor.sim.scenes import frame_scene

from spindoctor.sim.instruments.schema_detector import Detector, VidiconChain
from spindoctor.sim.scene.declare import DECLARATION, declaration_of, optional
from spindoctor.sim.scene.idealized import IdealizedScene
from spindoctor.sim.scene.reference import scene_reference
from spindoctor.sim.scene.truth import UNTYPED_BLOCKS, Scene
from spindoctor.sim.scene.truth_detector import DetectorOverride, Noise, NoiseVidicon
from spindoctor.sim.scene.validate import parse_scene


def _scene_only_fields() -> list[dataclasses.Field[Any]]:
    """The fields :class:`Scene` adds to :class:`IdealizedScene`."""
    idealized = {item.name for item in dataclasses.fields(IdealizedScene)}
    return [item for item in dataclasses.fields(Scene) if item.name not in idealized]


def _typed_fields(cls: type, prefix: str = '') -> list[tuple[str, dataclasses.Field[Any]]]:
    """Every field of a typed block and of the blocks inside it, with its dotted path."""
    found: list[tuple[str, dataclasses.Field[Any]]] = []
    hints = get_type_hints(cls, include_extras=True)
    for item in dataclasses.fields(cls):
        path = f'{prefix}{item.name}'
        found.append((path, item))
        inner = _without_none(hints[item.name])
        if get_origin(inner) is tuple:
            inner = get_args(inner)[0]
        if isinstance(inner, type) and dataclasses.is_dataclass(inner):
            found += _typed_fields(inner, f'{path}.')
    return found


def test_every_typed_field_carries_a_declaration() -> None:
    """Every field of every typed block carries its declaration."""
    undeclared = [path for path, item in _typed_fields(Scene) if DECLARATION not in item.metadata]
    assert undeclared == []


def test_every_idealized_field_declares_itself_idealized() -> None:
    """A field of the idealized scene type declares the idealized class."""
    classes = {
        item.name: declaration_of(item).boundary for item in dataclasses.fields(IdealizedScene)
    }
    assert set(classes.values()) == {'idealized'}


def test_no_field_the_scene_adds_declares_itself_idealized() -> None:
    """A field the truth scene type adds declares the truth or the test class."""
    idealized = [
        item.name for item in _scene_only_fields() if declaration_of(item).boundary == 'idealized'
    ]
    assert idealized == []


def test_no_untyped_block_is_a_typed_field() -> None:
    """A block listed as untyped is not also a field of the typed scene."""
    typed = {item.name for item in dataclasses.fields(Scene)}
    assert typed & set(UNTYPED_BLOCKS) == set()


def _catalog_pairs() -> list[tuple[type, type, str]]:
    """Each scene field that the catalog's detector also declares, with both owners."""
    catalog = get_type_hints(Detector, include_extras=True)
    vidicon = get_type_hints(VidiconChain, include_extras=True)
    pairs: list[tuple[type, type, str]] = []
    for owner in (DetectorOverride, Noise):
        pairs += [(owner, Detector, name) for name in get_type_hints(owner) if name in catalog]
    # The vidicon chain is a block of its own on both sides, compared key by key.
    pairs = [pair for pair in pairs if pair[2] != 'vidicon']
    pairs += [(NoiseVidicon, VidiconChain, name) for name in vidicon]
    return pairs


def _without_none(annotation: Any) -> Any:
    """An optional annotation's other member, or the annotation itself."""
    if get_origin(annotation) not in (Union, types.UnionType):
        return annotation
    members = [member for member in get_args(annotation) if member is not type(None)]
    return members[0] if len(members) == 1 else annotation


def test_every_shared_detector_field_takes_the_catalog_type() -> None:
    """A scene field the catalog also declares is typed by the catalog's declaration."""
    differing = []
    for owner, catalog, name in _catalog_pairs():
        scene_type = _without_none(get_type_hints(owner, include_extras=True)[name])
        catalog_type = _without_none(get_type_hints(catalog, include_extras=True)[name])
        if scene_type != catalog_type:
            differing.append(f'{owner.__name__}.{name}')
    assert differing == []


def test_the_shared_fields_include_the_noise_block() -> None:
    """The pairs the type check walks include the noise block's catalog keys."""
    names = {name for owner, _, name in _catalog_pairs() if owner is Noise}
    assert 'cosmic_ray_spread_px' in names


def test_parse_scene_returns_the_typed_scene() -> None:
    """The typed scene holds the values the mapping sets."""
    scene = parse_scene(frame_scene(noise={'read_noise_dn': 3.0}), source='probe')
    assert scene.noise == Noise(read_noise_dn=3.0)


def test_parse_scene_leaves_a_key_left_out_as_none() -> None:
    """A key the mapping leaves out is None in the typed scene, whatever its default."""
    scene = parse_scene(frame_scene(), source='probe')
    assert scene.offset_v is None


def test_parse_scene_leaves_the_mapping_unchanged() -> None:
    """Parsing reads the mapping and writes nothing into it."""
    raw = frame_scene(optics={'psf': {'match_navigator': True}})
    before = repr(raw)
    parse_scene(raw, source='probe')
    assert repr(raw) == before


def test_the_reference_lists_every_typed_key() -> None:
    """The generated reference has a row for every key of every typed block."""
    page = scene_reference()
    missing = [path for path, item in _typed_fields(Scene) if f'   * - ``{item.name}``' not in page]
    assert missing == []


def test_the_reference_lists_every_untyped_block() -> None:
    """The generated reference names each block the typed scene does not hold yet."""
    page = scene_reference()
    missing = [name for name in UNTYPED_BLOCKS if f'   * - ``{name}``' not in page]
    assert missing == []


def test_an_optional_field_needs_a_default_or_its_absence_explained() -> None:
    """Declaring an optional key with neither a default nor a meaning for absence fails."""
    with pytest.raises(TypeError, match='gives default or when_absent'):
        optional('truth', 'A key.')


def test_a_field_without_a_declaration_is_refused() -> None:
    """Reading the declaration of a plain dataclass field fails."""

    @dataclasses.dataclass
    class Plain:
        """A dataclass whose field carries no declaration."""

        key: int = 0

    with pytest.raises(TypeError, match='declared with required'):
        declaration_of(dataclasses.fields(Plain)[0])
