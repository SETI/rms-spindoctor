"""The typed scene: its declarations, its boundary classes, its ranges, and its reference.

Each test checks one seam the typed scene creates: that every field declares itself,
that a field's boundary class agrees with where the field sits in the types, that a
detector value has the catalog's range, that the rules the types carry refuse what
they should, that ``parse_scene`` returns the typed scene and leaves the mapping alone,
and that the generated reference lists every typed key under its block.
"""

import dataclasses
import types
from typing import Any, Union, get_args, get_origin, get_type_hints

import pytest
from tests.spindoctor.sim.scenes import frame_scene

from spindoctor.sim.instruments.resolvers import resolve_detector_defaults
from spindoctor.sim.instruments.schema_detector import Detector, VidiconChain
from spindoctor.sim.scene import boundary
from spindoctor.sim.scene.declare import DECLARATION, declaration_of, optional
from spindoctor.sim.scene.errors import SimSceneValidationError
from spindoctor.sim.scene.reference import scene_reference
from spindoctor.sim.scene.truth import UNTYPED_BLOCKS, Scene
from spindoctor.sim.scene.truth_detector import DetectorOverride, Noise, NoiseVidicon
from spindoctor.sim.scene.validate import (
    clear_validation_cache,
    parse_scene,
    validate_scene,
    validate_sim_params,
)

# A scene key the catalog declares under another name: the scene's key, then the
# catalog's.
_RENAMED_IN_CATALOG = {'gain_state': 'default_gain_state'}


def _without_none(annotation: Any) -> Any:
    """An optional annotation's other member, or the annotation itself."""
    if get_origin(annotation) not in (Union, types.UnionType):
        return annotation
    members = [member for member in get_args(annotation) if member is not type(None)]
    return members[0] if len(members) == 1 else annotation


def _record(annotation: Any) -> tuple[type | None, bool]:
    """The typed block a field holds, and whether it holds a list or mapping of them."""
    inner = _without_none(annotation)
    is_list = get_origin(inner) in (tuple, dict)
    if get_origin(inner) is tuple:
        inner = get_args(inner)[0]
    elif get_origin(inner) is dict:
        inner = get_args(inner)[1]
    if isinstance(inner, type) and dataclasses.is_dataclass(inner):
        return inner, is_list
    return None, False


def _typed_fields(cls: type, prefix: str = '') -> list[tuple[str, dataclasses.Field[Any]]]:
    """Every field of a typed block and of the blocks inside it, with its dotted path."""
    found: list[tuple[str, dataclasses.Field[Any]]] = []
    hints = get_type_hints(cls, include_extras=True)
    for item in dataclasses.fields(cls):
        path = f'{prefix}{item.name}'
        found.append((path, item))
        record, _ = _record(hints[item.name])
        if record is not None:
            found += _typed_fields(record, f'{path}.')
    return found


def _idealized_field_names(cls: type) -> set[str]:
    """The fields a type takes from its idealized types, itself included."""
    return {
        item.name
        for base in cls.__mro__
        if base.__name__.startswith('Idealized') and dataclasses.is_dataclass(base)
        for item in dataclasses.fields(base)
    }


def _structural_classes(
    cls: type, holder: str, prefix: str = ''
) -> list[tuple[str, str, frozenset[str]]]:
    """Each field's path, its declared class, and the classes its place allows.

    A field of an idealized type is idealized.  A field a truth type adds to its
    idealized type is truth or test.  A field of a block with no idealized type takes
    the class of the field that holds the block.
    """
    idealized = _idealized_field_names(cls)
    has_idealized_type = bool(idealized)
    found: list[tuple[str, str, frozenset[str]]] = []
    hints = get_type_hints(cls, include_extras=True)
    for item in dataclasses.fields(cls):
        path = f'{prefix}{item.name}'
        declared = declaration_of(item).boundary
        if item.name in idealized:
            allowed = frozenset({'idealized'})
        elif has_idealized_type:
            allowed = frozenset({'truth', 'test'})
        else:
            allowed = frozenset({holder})
        found.append((path, declared, allowed))
        record, _ = _record(hints[item.name])
        if record is not None:
            found += _structural_classes(record, declared, f'{path}.')
    return found


def test_every_typed_field_carries_a_declaration() -> None:
    """Every field of every typed block carries its declaration."""
    undeclared = [path for path, item in _typed_fields(Scene) if DECLARATION not in item.metadata]
    assert undeclared == []


def test_every_field_of_an_idealized_type_declares_itself_idealized() -> None:
    """A field an idealized type declares, at any depth, declares the idealized class."""
    wrong = [
        path
        for path, declared, allowed in _structural_classes(Scene, 'truth')
        if allowed == {'idealized'} and declared != 'idealized'
    ]
    assert wrong == []


def test_every_other_field_declares_the_class_its_place_gives() -> None:
    """A field a truth type adds, or a field of a truth block, declares truth or test."""
    wrong = [
        path
        for path, declared, allowed in _structural_classes(Scene, 'truth')
        if allowed != {'idealized'} and declared not in allowed
    ]
    assert wrong == []


def test_no_untyped_block_is_a_typed_field() -> None:
    """A block listed as untyped is not also a field of the typed scene."""
    typed = {item.name for item in dataclasses.fields(Scene)}
    assert typed & set(UNTYPED_BLOCKS) == set()


def _catalog_pairs() -> list[tuple[type, str, type, str]]:
    """Each scene field the catalog's detector also declares: owner and key on each side."""
    catalog = get_type_hints(Detector, include_extras=True)
    vidicon = get_type_hints(VidiconChain, include_extras=True)
    pairs: list[tuple[type, str, type, str]] = []
    for owner in (DetectorOverride, Noise):
        for name in get_type_hints(owner):
            catalog_name = _RENAMED_IN_CATALOG.get(name, name)
            # The vidicon chain is a block of its own on both sides, compared key by key.
            if catalog_name in catalog and name != 'vidicon':
                pairs.append((owner, name, Detector, catalog_name))
    pairs += [(NoiseVidicon, name, VidiconChain, name) for name in vidicon]
    return pairs


def test_every_shared_detector_field_takes_the_catalog_type() -> None:
    """A scene field the catalog also declares is typed by the catalog's declaration."""
    differing = []
    for owner, name, catalog, catalog_name in _catalog_pairs():
        scene_type = _without_none(get_type_hints(owner, include_extras=True)[name])
        catalog_type = _without_none(get_type_hints(catalog, include_extras=True)[catalog_name])
        if scene_type != catalog_type:
            differing.append(f'{owner.__name__}.{name}')
    assert differing == []


def test_the_shared_fields_include_the_noise_block() -> None:
    """The pairs the type check walks include the noise block's catalog keys."""
    names = {name for owner, name, _, _ in _catalog_pairs() if owner is Noise}
    assert 'cosmic_ray_spread_px' in names


def test_the_shared_fields_include_the_renamed_gain_state() -> None:
    """The pairs the type check walks include the key the catalog calls by another name."""
    names = {name for owner, name, _, _ in _catalog_pairs() if owner is DetectorOverride}
    assert 'gain_state' in names


def test_a_psf_matched_to_the_navigator_refuses_a_kernel_key() -> None:
    """The navigator-matched PSF form is refused beside a key of the explicit kernel."""
    scene = frame_scene(optics={'psf': {'match_navigator': True, 'w': 0.1}})
    with pytest.raises(
        SimSceneValidationError, match=r"match_navigator is exclusive; drop \['w'\]"
    ):
        parse_scene(scene, source='probe')


def test_a_smear_entry_refuses_an_unknown_object_class() -> None:
    """A smear entry's object class is one of the classes the renderer separates."""
    scene = frame_scene(optics={'smear': [{'dv_px': 1.0, 'object_class': 'moons'}]})
    with pytest.raises(
        SimSceneValidationError,
        match=r"optics\.smear\[0\]\.object_class: 'moons' is not one of "
        r"\['all', 'bodies', 'rings', 'stars'\]",
    ):
        parse_scene(scene, source='probe')


def test_a_ghost_refuses_a_negative_defocus_sigma() -> None:
    """A ghost's blur sigma is never negative."""
    scene = frame_scene(optics={'ghosts': [{'amplitude': 0.1, 'defocus_sigma': -1.0}]})
    with pytest.raises(
        SimSceneValidationError,
        match=r'optics\.ghosts\[0\]\.defocus_sigma: expected a non-negative number, got -1\.0',
    ):
        parse_scene(scene, source='probe')


def test_a_psf_wing_fraction_above_one_is_refused() -> None:
    """The Moffat wing's energy fraction lies in [0, 1]."""
    scene = frame_scene(optics={'psf': {'sigma_v': 1.0, 'w': 1.5}})
    with pytest.raises(
        SimSceneValidationError, match=r'optics\.psf\.w: expected a number in \[0, 1\], got 1\.5'
    ):
        parse_scene(scene, source='probe')


def test_a_set_of_published_truth_paths_is_refused() -> None:
    """``published_equals_truth`` takes a list, as a scene file writes it."""
    scene = frame_scene(published_equals_truth={'noise.read_noise_dn'})
    with pytest.raises(
        SimSceneValidationError, match=r'published_equals_truth: expected a list, got set'
    ):
        parse_scene(scene, source='probe')


def test_a_typed_block_instance_is_refused() -> None:
    """A block is a mapping, as a scene file writes it, never a typed instance."""
    scene = frame_scene()
    scene['detector'] = DetectorOverride(bias_dn=7.0)
    with pytest.raises(
        SimSceneValidationError,
        match=r'probe: detector: expected a mapping, a list, .*got Detector',
    ):
        validate_sim_params(scene, source='probe')


def test_a_generator_of_published_truth_paths_is_refused_unconsumed() -> None:
    """A generator is refused before the schema reads it, so it is not exhausted."""
    paths = (path for path in ['noise.read_noise_dn'])
    scene = frame_scene()
    scene['published_equals_truth'] = paths
    with pytest.raises(
        SimSceneValidationError, match=r'probe: published_equals_truth: expected a mapping'
    ):
        validate_sim_params(scene, source='probe')
    assert list(paths) == ['noise.read_noise_dn']


def test_an_empty_tuple_is_refused_after_the_empty_list_is_cached() -> None:
    """A warm cache for ``[]`` does not make ``()`` valid."""
    clear_validation_cache()
    validate_scene(frame_scene(published_equals_truth=[]), source='probe')
    with pytest.raises(
        SimSceneValidationError, match=r'probe: published_equals_truth: expected a list, got tuple'
    ):
        validate_scene(frame_scene(published_equals_truth=()), source='probe')


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


def _page_sections(page: str) -> dict[str, str]:
    """The reference page's text under each heading, keyed by the heading."""
    lines = page.splitlines()
    sections: dict[str, list[str]] = {}
    current: list[str] | None = None
    for line, below in zip(lines, [*lines[1:], ''], strict=True):
        if line and below and set(below) == {'='} and len(below) == len(line):
            current = sections.setdefault(line, [])
        elif current is not None:
            current.append(line)
    return {heading: '\n'.join(text) for heading, text in sections.items()}


def _block_headings(cls: type, path: str = '', is_list: bool = False) -> list[tuple[str, type]]:
    """Each typed block's heading on the reference page, with the block's type."""
    if not path:
        heading = 'The top level'
    elif is_list:
        heading = f'Each entry of ``{path}``'
    else:
        heading = f'The ``{path}`` block'
    found = [(heading, cls)]
    hints = get_type_hints(cls, include_extras=True)
    for item in dataclasses.fields(cls):
        record, holds_list = _record(hints[item.name])
        if record is not None:
            child = f'{path}.{item.name}' if path else item.name
            found += _block_headings(record, child, holds_list)
    return found


def test_the_reference_lists_every_typed_key_under_its_block() -> None:
    """The generated reference has a row for every key of a block, under that block."""
    sections = _page_sections(scene_reference())
    missing = [
        f'{heading}: {item.name}'
        for heading, cls in _block_headings(Scene)
        for item in dataclasses.fields(cls)
        if f'   * - ``{item.name}``' not in sections.get(heading, '')
    ]
    assert missing == []


def test_the_reference_lists_every_untyped_block() -> None:
    """The generated reference lists each block the typed scene does not hold yet."""
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
        """A dataclass whose field has no declaration."""

        key: int = 0

    with pytest.raises(TypeError, match='declared with required'):
        declaration_of(dataclasses.fields(Plain)[0])


def test_a_key_both_typed_and_untyped_fails_the_boundary_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A typed field listed again as an untyped block fails the import-time check."""
    monkeypatch.setitem(UNTYPED_BLOCKS, 'size_v', 'idealized')
    with pytest.raises(
        AssertionError, match=r"both typed and listed as untyped blocks: \['size_v'\]"
    ):
        boundary._assert_boundary_classification_complete()


def test_a_key_in_two_top_level_classes_fails_the_boundary_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A top-level key in two boundary classes fails the import-time check."""
    monkeypatch.setattr(
        boundary, 'TOP_LEVEL_TRUTH_KEYS', boundary.TOP_LEVEL_TRUTH_KEYS | {'expected'}
    )
    with pytest.raises(AssertionError, match=r"both truth and test: \['expected'\]"):
        boundary._assert_boundary_classification_complete()


def _binned_mode_with_own_gain_states(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give the Cassini NAC's 2x2 mode the gain states 4 and 5 alone."""

    def patched(instrument: str | None, binning: int) -> Detector:
        defaults = resolve_detector_defaults(instrument, binning)
        if binning == 2:
            return dataclasses.replace(
                defaults, gain_e_per_dn_by_state={4: 9.0, 5: 4.0}, default_gain_state=4
            )
        return defaults

    monkeypatch.setattr('spindoctor.sim.scene.rules_detector.resolve_detector_defaults', patched)


def test_a_gain_state_only_the_selected_mode_catalogs_is_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A gain state the binned mode offers passes, though the native table lacks it."""
    _binned_mode_with_own_gain_states(monkeypatch)
    scene = frame_scene(detector={'binning': 2, 'gain_state': 4})
    clear_validation_cache()
    assert validate_sim_params(scene) is scene


def test_a_gain_state_the_selected_mode_lacks_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A native gain state the binned mode does not offer is refused."""
    _binned_mode_with_own_gain_states(monkeypatch)
    scene = frame_scene(detector={'binning': 2, 'gain_state': 0})
    clear_validation_cache()
    with pytest.raises(SimSceneValidationError, match=r'available states: \[4, 5\]'):
        validate_sim_params(scene)
