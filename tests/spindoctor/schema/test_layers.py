"""Tests for :mod:`spindoctor.schema.layers`: reading, merging, nulls, and bases."""

from pathlib import Path

import pytest

from spindoctor.schema.errors import SchemaError
from spindoctor.schema.layers import (
    Layer,
    NullPolicy,
    merge,
    read_layer,
    resolve_bases,
    split_entries,
)

# A base layer that the merge tests lay other layers over.
_BASE = Layer({'a': 1, 'block': {'x': 1, 'y': 2}, 'items': [1, 2]}, 'base.yaml')


def test_read_layer_drops_note_keys(tmp_path: Path) -> None:
    """A key that begins with an underscore is dropped at every depth, lists included."""
    path = tmp_path / 'f.yaml'
    path.write_text('_note: x\nblock:\n  _sources: y\n  v: 1\nitems:\n  - {_c: 1, w: 2}\n')
    assert read_layer(path).data == {'block': {'v': 1}, 'items': [{'w': 2}]}


def test_read_layer_labels_the_layer_with_the_file_name(tmp_path: Path) -> None:
    """A layer read without a label is labeled with the file's name."""
    path = tmp_path / 'f.yaml'
    path.write_text('a: 1\n')
    assert read_layer(path).label == 'f.yaml'


@pytest.mark.parametrize(
    ('text', 'match'),
    [
        ('a: 1\na: 2\n', 'f.yaml: not valid YAML: '),
        ('- 1\n- 2\n', 'f.yaml: expected a mapping at the top level, got list'),
    ],
    ids=['duplicate-key', 'not-a-mapping'],
)
def test_read_layer_refuses_a_bad_file(tmp_path: Path, text: str, match: str) -> None:
    """A file that is not valid YAML, or holds no mapping, fails with the file's name."""
    path = tmp_path / 'f.yaml'
    path.write_text(text)
    with pytest.raises(SchemaError, match=match):
        read_layer(path)


def test_merge_merges_mappings_key_by_key() -> None:
    """A later layer that sets one key of a mapping keeps the mapping's other keys."""
    merged = merge([_BASE, Layer({'block': {'y': 3}}, 'user.yaml')], nulls=NullPolicy.REMOVE)
    assert merged.data['block'] == {'x': 1, 'y': 3}


def test_merge_replaces_a_list_whole() -> None:
    """A later layer's list replaces the earlier list."""
    merged = merge([_BASE, Layer({'items': [3]}, 'user.yaml')], nulls=NullPolicy.REMOVE)
    assert merged.data['items'] == [3]


def test_merge_records_the_layer_that_set_each_value() -> None:
    """Each value's origin is the last layer that set it."""
    user = Layer({'block': {'y': 3}, 'items': [3]}, 'user.yaml')
    merged = merge([_BASE, user], nulls=NullPolicy.REMOVE)
    assert merged.origins == {
        ('a',): 'base.yaml',
        ('block', 'x'): 'base.yaml',
        ('block', 'y'): 'user.yaml',
        ('items',): 'user.yaml',
    }


def test_merge_leaves_the_layers_unchanged() -> None:
    """Merging copies the values, so changing the result changes no layer."""
    merged = merge([_BASE], nulls=NullPolicy.REMOVE)
    merged.data['block']['x'] = 9
    assert _BASE.data['block']['x'] == 1


def test_merge_copies_mappings_inside_a_list() -> None:
    """A list is copied with its contents, so changing an item changes no layer."""
    layer = Layer({'items': [{'w': 1}]}, 'base.yaml')
    merged = merge([layer], nulls=NullPolicy.REMOVE)
    merged.data['items'][0]['w'] = 9
    assert layer.data['items'][0]['w'] == 1


def test_value_over_a_mapping_drops_the_mapping_origins() -> None:
    """A scalar that replaces a whole mapping is the only origin left at that path."""
    merged = merge([_BASE, Layer({'block': 5}, 'user.yaml')], nulls=NullPolicy.REMOVE)
    assert merged.origins == {('a',): 'base.yaml', ('items',): 'base.yaml', ('block',): 'user.yaml'}


def test_filling_an_empty_mapping_drops_its_origin() -> None:
    """An empty mapping that a later layer fills is no longer a value of its own."""
    layers = [Layer({'m': {}}, 'base.yaml'), Layer({'m': {'k': 1}}, 'user.yaml')]
    merged = merge(layers, nulls=NullPolicy.REMOVE)
    assert merged.origins == {('m', 'k'): 'user.yaml'}


def test_read_layer_reads_an_empty_file_as_an_empty_mapping(tmp_path: Path) -> None:
    """An empty file is an empty layer."""
    path = tmp_path / 'f.yaml'
    path.write_text('')
    assert read_layer(path).data == {}


def test_remove_records_the_layer_that_removed_the_value() -> None:
    """Under REMOVE, a null deletes a mapping's origins and records its own layer there."""
    merged = merge([_BASE, Layer({'block': None}, 'user.yaml')], nulls=NullPolicy.REMOVE)
    assert merged.origins == {
        ('a',): 'base.yaml',
        ('items',): 'base.yaml',
        ('block',): 'user.yaml',
    }


def test_setting_a_removed_key_again_drops_the_removal() -> None:
    """A later layer that sets a removed key again is the only origin left there."""
    layers = [
        Layer({'m': 1}, 'base.yaml'),
        Layer({'m': None}, 'user.yaml'),
        Layer({'m': {'k': 1}}, 'last.yaml'),
    ]
    merged = merge(layers, nulls=NullPolicy.REMOVE)
    assert merged.origins == {('m', 'k'): 'last.yaml'}


@pytest.mark.parametrize(
    ('overlay', 'message'),
    [
        (
            {'b': None},
            'user.yaml: b: null removes the value an earlier layer sets, and no earlier layer '
            'sets this key',
        ),
        (
            {'new': {'q': None}},
            'user.yaml: new.q: null removes the value an earlier layer sets, and no earlier '
            'layer sets this key',
        ),
        (
            {'items': [1, None]},
            'user.yaml: items[1]: a list item cannot be null: a list replaces the earlier list '
            'whole, so an item has nothing to remove',
        ),
    ],
    ids=['nothing-to-remove', 'inside-a-new-mapping', 'list-item'],
)
def test_remove_refuses_a_null_with_nothing_to_remove(
    overlay: dict[str, object], message: str
) -> None:
    """Under REMOVE, a null that has no earlier value beneath it is an error."""
    with pytest.raises(SchemaError) as caught:
        merge([_BASE, Layer(overlay, 'user.yaml')], nulls=NullPolicy.REMOVE)
    assert str(caught.value) == message


@pytest.mark.parametrize(
    ('overlay', 'message'),
    [
        ({'a': None}, 'scene.yaml: a: null is refused; leave the key out to take its default'),
        (
            {'items': [{'w': None}]},
            'scene.yaml: items[0].w: null is refused; leave the key out to take its default',
        ),
        (
            {'items': [None]},
            'scene.yaml: items[0]: a list item cannot be null; give it a value or remove it',
        ),
    ],
    ids=['over-a-value', 'inside-a-list', 'list-item'],
)
def test_refuse_refuses_every_null(overlay: dict[str, object], message: str) -> None:
    """Under REFUSE, a null is an error even where an earlier layer set the key."""
    with pytest.raises(SchemaError) as caught:
        merge([_BASE, Layer(overlay, 'scene.yaml')], nulls=NullPolicy.REFUSE)
    assert str(caught.value) == message


def test_merge_reports_every_null_at_once() -> None:
    """Nulls in two layers give one issue each."""
    layers = [Layer({'a': None}, 'one.yaml'), Layer({'b': None}, 'two.yaml')]
    with pytest.raises(SchemaError) as caught:
        merge(layers, nulls=NullPolicy.REFUSE)
    assert [issue.source for issue in caught.value.issues] == ['one.yaml', 'two.yaml']


def test_split_entries_labels_each_entry() -> None:
    """Each entry's layer carries the file's label and the entry's name."""
    entries = split_entries(Layer({'nac': {'a': 1}}, 'coiss.yaml'))
    assert entries['nac'].label == 'coiss.yaml: nac'


@pytest.mark.parametrize(
    ('data', 'message'),
    [
        ({3: {'a': 1}}, 'coiss.yaml: the entry name 3 must be a string'),
        ({'nac': [1]}, 'coiss.yaml: nac: expected a mapping, got list [1]'),
    ],
    ids=['name-not-a-string', 'entry-not-a-mapping'],
)
def test_split_entries_refuses_a_bad_entry(data: dict[object, object], message: str) -> None:
    """An entry name that is not a string, or an entry that is not a mapping, fails."""
    with pytest.raises(SchemaError) as caught:
        split_entries(Layer(data, 'coiss.yaml'))
    assert str(caught.value) == message


# Three entries in a chain: wac inherits nac, which inherits generic.
_CHAIN = {
    'generic': Layer({'gain': 1, 'bias': 2, 'aliases': ['g']}, 'generic.yaml: generic'),
    'nac': Layer({'base': 'generic', 'gain': 3}, 'coiss.yaml: nac'),
    'wac': Layer({'base': 'nac', 'bias': None, 'aliases': ['w']}, 'coiss.yaml: wac'),
}


def test_resolve_bases_merges_the_chain() -> None:
    """An entry holds its chain's values merged in order, its own last, without its base."""
    resolved = resolve_bases(_CHAIN, nulls=NullPolicy.REMOVE, not_inherited=frozenset({'aliases'}))
    assert resolved['wac'].data == {'gain': 3, 'aliases': ['w']}


def test_resolve_bases_records_the_entry_that_set_each_value() -> None:
    """An inherited value's origin is the entry that set it."""
    resolved = resolve_bases(_CHAIN, nulls=NullPolicy.REMOVE, not_inherited=frozenset({'aliases'}))
    assert resolved['nac'].origins == {
        ('bias',): 'generic.yaml: generic',
        ('gain',): 'coiss.yaml: nac',
    }


def test_resolve_bases_does_not_pass_on_a_kept_key() -> None:
    """A key in ``not_inherited`` reaches no entry that inherits it."""
    resolved = resolve_bases(_CHAIN, nulls=NullPolicy.REMOVE, not_inherited=frozenset({'aliases'}))
    assert 'aliases' not in resolved['nac'].data


@pytest.mark.parametrize(
    ('entries', 'message'),
    [
        ({'a': Layer({'base': 'b'}, 'f.yaml: a')}, "f.yaml: a: base: 'b' is not an entry"),
        (
            {'a': Layer({'base': 3}, 'f.yaml: a')},
            'f.yaml: a: base: expected the name of an entry, got int 3',
        ),
        (
            {'a': Layer({'base': 'a'}, 'f.yaml: a')},
            'f.yaml: a: base: inheritance loops (a -> a)',
        ),
    ],
    ids=['unknown-base', 'base-not-a-string', 'loop'],
)
def test_resolve_bases_refuses_a_bad_base(entries: dict[str, Layer], message: str) -> None:
    """A base that is not a string or not an entry, or that loops, fails."""
    with pytest.raises(SchemaError) as caught:
        resolve_bases(entries, nulls=NullPolicy.REMOVE)
    assert str(caught.value) == message


def test_resolve_bases_reports_a_problem_once() -> None:
    """A problem in a base is reported once, though every entry that inherits it meets it."""
    entries = {
        'root': Layer({'gone': None}, 'f.yaml: root'),
        'one': Layer({'base': 'root'}, 'f.yaml: one'),
        'two': Layer({'base': 'root'}, 'f.yaml: two'),
    }
    with pytest.raises(SchemaError) as caught:
        resolve_bases(entries, nulls=NullPolicy.REMOVE)
    assert len(caught.value.issues) == 1


def test_resolve_bases_keeps_an_entry_null_problem_behind_a_bad_base() -> None:
    """Behind a bad base, a refused list-item null is reported; a removing null is not."""
    entries = {'a': Layer({'base': 'gone', 'drop': None, 'items': [None]}, 'f.yaml: a')}
    with pytest.raises(SchemaError) as caught:
        resolve_bases(entries, nulls=NullPolicy.REMOVE)
    assert [issue.path for issue in caught.value.issues] == ['base', 'items[0]']
