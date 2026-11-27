"""The scene's ``instrument_config`` block: its types and its merge over the navigator block.

Each overlay value is typed by the camera block's own field type, the overlay covers
every camera key a scene may set, and the merge follows the two rules of its own: a
filter of another kind replaces the whole filter, and a magnitude-offset entry is a
mapping, so a bare number is refused when the scene is validated.
"""

import dataclasses
import functools
import operator
import types
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

import pytest
from tests.spindoctor.sim.scenes import frame_scene

from spindoctor.config import DEFAULT_CONFIG
from spindoctor.config.sections_camera import (
    BandpassFilter,
    BinnedNoise,
    BlurFilter,
    CameraBlock,
    CameraNoise,
    MagOffset,
    MagOffsetEntry,
    QualityThresholds,
)
from spindoctor.sim.instruments.navigator_link import resolve_sim_inst_config
from spindoctor.sim.scene.errors import SimSceneValidationError
from spindoctor.sim.scene.idealized_camera import (
    FilterKind,
    InstrumentConfig,
    InstrumentConfigBinnedNoise,
    InstrumentConfigFilter,
    InstrumentConfigMagOffset,
    InstrumentConfigMagOffsetEntry,
    InstrumentConfigNoise,
    InstrumentConfigQuality,
)
from spindoctor.sim.scene.validate import (
    clear_validation_cache,
    validate_scene,
    validate_sim_params,
)

# Each overlay type and the camera block types whose fields it states.
_PAIRS: tuple[tuple[type, tuple[type, ...]], ...] = (
    (InstrumentConfig, (CameraBlock,)),
    (InstrumentConfigNoise, (CameraNoise,)),
    (InstrumentConfigBinnedNoise, (BinnedNoise,)),
    (InstrumentConfigQuality, (QualityThresholds,)),
    (InstrumentConfigFilter, (BandpassFilter, BlurFilter)),
    (InstrumentConfigMagOffset, (MagOffset,)),
    (InstrumentConfigMagOffsetEntry, (MagOffsetEntry,)),
)
_OVERLAY_TYPES = {overlay for overlay, _ in _PAIRS}


def _without_none(annotation: Any) -> Any:
    """An optional annotation's other member, or the annotation itself."""
    if get_origin(annotation) not in (Union, types.UnionType):
        return annotation
    members = [member for member in get_args(annotation) if member is not type(None)]
    return functools.reduce(operator.or_, members)


def _holds_overlay(annotation: Any) -> bool:
    """Whether a field holds an overlay type, alone or as a mapping's values."""
    if get_origin(annotation) is dict:
        annotation = get_args(annotation)[1]
    return annotation in _OVERLAY_TYPES


def _key_type(annotation: Any) -> Any:
    """The key type of a mapping field, or None when the field is not a mapping."""
    annotation = _without_none(annotation)
    return get_args(annotation)[0] if get_origin(annotation) is dict else None


def _camera_hint(cameras: tuple[type, ...], name: str) -> Any:
    """The type the camera block types give a field."""
    for camera in cameras:
        hints = get_type_hints(camera, include_extras=True)
        if name in hints:
            return _without_none(hints[name])
    raise KeyError(name)


def test_every_overlay_value_takes_the_camera_block_type() -> None:
    """An overlay value is typed by the camera block's declaration of the same key."""
    differing = []
    for overlay, cameras in _PAIRS:
        for name, hint in get_type_hints(overlay, include_extras=True).items():
            scene_type = _without_none(hint)
            if _holds_overlay(scene_type) or (overlay, name) == (InstrumentConfigFilter, 'kind'):
                continue
            if scene_type != _camera_hint(cameras, name):
                differing.append(f'{overlay.__name__}.{name}')
    assert differing == []


def test_every_overlay_mapping_takes_the_camera_block_key_type() -> None:
    """An overlay mapping's keys are typed by the camera block's mapping, as its values are."""
    differing = []
    checked = []
    for overlay, cameras in _PAIRS:
        for name, hint in get_type_hints(overlay, include_extras=True).items():
            scene_key = _key_type(hint)
            if scene_key is None:
                continue
            checked.append(f'{overlay.__name__}.{name}')
            if scene_key != _key_type(_camera_hint(cameras, name)):
                differing.append(f'{overlay.__name__}.{name}')
    assert differing == []
    assert checked == [
        'InstrumentConfigNoise.by_binning',
        'InstrumentConfigMagOffset.mag_offset_table',
    ]


def test_the_overlay_states_every_camera_key_but_the_units() -> None:
    """A scene may change every key of the camera block except ``data_units``."""
    missing = [
        f'{overlay.__name__}: {name}'
        for overlay, cameras in _PAIRS
        for camera in cameras
        for name in get_type_hints(camera)
        if name not in get_type_hints(overlay) and name != 'data_units'
    ]
    assert missing == []


def test_the_filter_kinds_are_the_camera_block_kinds() -> None:
    """The overlay's filter kinds are the kinds of the camera block's two filter types."""
    kinds: set[str] = set()
    for camera in (BandpassFilter, BlurFilter):
        hint = get_type_hints(camera)['kind']
        assert get_origin(hint) is Literal
        kinds.update(get_args(hint))
    assert set(get_args(FilterKind)) == kinds


def test_a_filter_of_another_kind_replaces_the_whole_filter() -> None:
    """A blur over the generic block's bandpass filter keeps none of the bandpass sigmas."""
    overlay = InstrumentConfig(
        source_image_filter=InstrumentConfigFilter(
            kind='ISOTROPIC_GAUSSIAN', enabled=True, sigma_px=1.5
        )
    )
    merged = resolve_sim_inst_config(DEFAULT_CONFIG, 'generic', overlay)
    assert merged.source_image_filter == BlurFilter(
        kind='ISOTROPIC_GAUSSIAN', enabled=True, sigma_px=1.5
    )


def test_a_filter_change_without_a_kind_keeps_the_block_filter() -> None:
    """A filter change that leaves ``kind`` out changes keys of the block's own filter."""
    base = resolve_sim_inst_config(DEFAULT_CONFIG, 'generic')
    overlay = InstrumentConfig(source_image_filter=InstrumentConfigFilter(enabled=True))
    merged = resolve_sim_inst_config(DEFAULT_CONFIG, 'generic', overlay)
    assert merged.source_image_filter == dataclasses.replace(base.source_image_filter, enabled=True)


def test_a_filter_of_the_blocks_own_kind_changes_keys_of_the_block_filter() -> None:
    """Setting ``kind`` to the block's kind keeps the filter's other keys."""
    base = resolve_sim_inst_config(DEFAULT_CONFIG, 'generic')
    own_kind = base.source_image_filter.kind
    overlay = InstrumentConfig(
        source_image_filter=InstrumentConfigFilter(kind=own_kind, enabled=True)
    )
    merged = resolve_sim_inst_config(DEFAULT_CONFIG, 'generic', overlay)
    assert merged.source_image_filter == dataclasses.replace(base.source_image_filter, enabled=True)


def test_a_filter_of_another_kind_without_its_keys_is_refused_at_validation() -> None:
    """A new kind replaces the filter, so a scene that leaves ``enabled`` out is refused."""
    overlay = {'source_image_filter': {'kind': 'ISOTROPIC_GAUSSIAN', 'sigma_px': 1.0}}
    scene = frame_scene(instrument='generic', instrument_config=overlay)
    with pytest.raises(
        SimSceneValidationError,
        match=r'probe: instrument_config merged over the navigator block sim: '
        r"source_image_filter: missing required keys: \['enabled'\]$",
    ):
        validate_sim_params(scene, source='probe')


def test_a_merged_block_that_breaks_a_camera_rule_is_refused_at_validation() -> None:
    """A heavy blur narrower than the block's light blur fails the merged block's rule."""
    overlay = {'source_image_filter': {'lo_sigma_px': 0.5}}
    scene = frame_scene(instrument='generic', instrument_config=overlay)
    with pytest.raises(
        SimSceneValidationError,
        match=r'instrument_config merged over the navigator block sim: '
        r'source_image_filter\.lo_sigma_px: lo_sigma_px \(0\.5\) must be greater than '
        r'hi_sigma_px \(0\.7\)$',
    ):
        validate_sim_params(scene, source='probe')


def test_a_cached_check_refuses_an_integer_key_in_the_offset_table() -> None:
    """The cached check gives the answer of the uncached one for a table key ``1``."""
    overlay = {'mag_offset': {'fallback_combo': '1', 'mag_offset_table': {1: {'default': 0.5}}}}
    scene = frame_scene(instrument='generic', instrument_config=overlay)
    clear_validation_cache()
    message = r'probe: instrument_config\.mag_offset\.mag_offset_table\.1: expected a string '
    message += r'as the key, got int 1$'
    with pytest.raises(SimSceneValidationError, match=message):
        validate_sim_params(scene, source='probe')
    with pytest.raises(SimSceneValidationError, match=message):
        validate_scene(scene, source='probe')


def test_an_integer_binning_key_is_accepted_cached_and_uncached() -> None:
    """A binned mode's key written as the number 2 passes both checks."""
    table = {2: {'read_noise_dn': 1.0, 'star_flux_dn_per_s_vmag0': 1.0e6}}
    scene = frame_scene(instrument='coiss_nac', instrument_config={'noise': {'by_binning': table}})
    clear_validation_cache()
    validate_sim_params(scene, source='probe')
    validate_scene(scene, source='probe')


def test_every_line_of_a_merged_block_error_gives_the_scene_source() -> None:
    """Two problems in the merged block are two lines, each beginning with the source."""
    overlay = {
        'source_image_filter': {'lo_sigma_px': 0.5},
        'mag_offset': {'fallback_combo': 'XX'},
    }
    scene = frame_scene(instrument='generic', instrument_config=overlay)
    with pytest.raises(SimSceneValidationError) as caught:
        validate_sim_params(scene, source='probe')
    lines = str(caught.value).splitlines()
    assert len(lines) == 2
    prefix = 'probe: instrument_config merged over the navigator block sim: '
    assert all(line.startswith(prefix) for line in lines)


def test_a_bare_magnitude_offset_is_refused_at_validation() -> None:
    """An offset written as a bare number is refused: an entry sets ``default``."""
    overlay = {'mag_offset': {'mag_offset_table': {'CL': 0.5}}}
    scene = frame_scene(instrument='generic', instrument_config=overlay)
    with pytest.raises(
        SimSceneValidationError,
        match=r'probe: instrument_config\.mag_offset\.mag_offset_table\.CL: expected a '
        r'mapping, got float 0\.5$',
    ):
        validate_sim_params(scene, source='probe')


def test_a_magnitude_offset_entry_replaces_the_block_entry() -> None:
    """A scene's entry for a combination replaces the block's entry for it."""
    entry = InstrumentConfigMagOffsetEntry(default=0.5)
    overlay = InstrumentConfig(mag_offset=InstrumentConfigMagOffset(mag_offset_table={'CL': entry}))
    merged = resolve_sim_inst_config(DEFAULT_CONFIG, 'generic', overlay)
    assert merged.mag_offset.offset() == 0.5
