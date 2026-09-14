"""The global index labels the Cassini ISS Saturn bundle ships, and the tables beside them.

What each index label says of its table: its ``Header`` is the table's header line and its
``Table_Character`` begins where that line ends; it counts the table's rows as its
records and its columns as its fields; each ``Field_Character`` is numbered by its
position and lands on the column it names in every record; each fixed field states the
data type of its values, and each statistic field the data type, unit and description
its configuration gives it; each label names the table beside it; and each row's
exposure start and stop are the ones its data label
states.  The miscellaneous collection and the bundle's entry for it state the
miscellaneous types.  Those are asked of the cohort's bundle.  Two more questions need a
configuration the test chooses, and are asked of the shipped templates over plumbing
inputs: that a plane added to the configuration adds a column to the table and a
``Field_Character`` to the label, and that a missing statistic's cell, in either table,
holds the constant its field declares, in the spelling the label declares it.  The
plumbing is tested over stand-in templates in ``test_global_index.py``.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import pytest
from filecache import FCPath
from tests.mini_nav_results.cohort import WrittenCohorts
from tests.mini_nav_results.cohort_cassini import LIMB_STUB, RINGS_STUB, CohortCassiniISSSaturn

from spindoctor.cli.backplanes.statistics import statistics_units
from spindoctor.cli.pds4.global_index import INDEX_VALUE_FORMATS, generate_global_index_files
from spindoctor.config import DEFAULT_CONFIG, MAIN_LOGGER

from .conftest import (
    index_entry,
    make_bundle_env,
    read_csv_rows,
    read_index_rows,
    ring_metadata,
    touch_label,
    write_cohort_bundle,
    write_supplemental,
)

PDS4_NAMESPACES = {'pds': 'http://pds.nasa.gov/pds4/pds/v1'}
"""The PDS4 common dictionary's namespace, under the prefix the paths below use."""

AREA = 'pds:File_Area_Ancillary'
"""Where an index label describes its table."""

RECORD = f'{AREA}/pds:Table_Character/pds:Record_Character'
"""Where an index label describes each record of its table."""

INDEX_NAMES = ('global_bodies_index', 'global_rings_index')
"""The two index products, each a table and a label of that stem."""

NAVIGATED_STUBS = (LIMB_STUB, RINGS_STUB)
"""The cohort's two navigated images, by results path stub."""


@pytest.fixture
def cassini_cohort(mini_nav_cohorts: WrittenCohorts) -> CohortCassiniISSSaturn:
    """Return the Cassini ISS Saturn cohort, as the session wrote it.

    Parameters:
        mini_nav_cohorts: What the session's cohorts are written by.

    Returns:
        The written cohort.
    """
    return mini_nav_cohorts(CohortCassiniISSSaturn)


@dataclass(frozen=True)
class _FieldCharacter:
    """What one ``Field_Character`` of an index label states.

    Attributes:
        name: Its name.
        number: Its ``field_number``.
        location: The byte its field begins at in a record, counted from 1.
        length: Its field's length in bytes.
        data_type: Its data type.
        unit: Its unit, or an empty string.
        description: Its description, its whitespace collapsed to single spaces.
        missing_constant: The missing constant it declares, or an empty string.
    """

    name: str
    number: int
    location: int
    length: int
    data_type: str
    unit: str
    description: str
    missing_constant: str

    def value(self, record: str) -> str:
        """Return the text of this field in one record, its padding kept.

        Parameters:
            record: The record.

        Returns:
            The bytes the field's location and length give.
        """
        return record[self.location - 1 : self.location - 1 + self.length]


def _text(root: ElementTree.Element, path: str) -> str:
    """Return the stripped text of the element a path names in a label.

    Parameters:
        root: The label's root element.
        path: The element's path from the root, in the ``pds`` prefix.

    Returns:
        Its text, stripped, or an empty string when there is no such element.
    """
    return root.findtext(path, '', PDS4_NAMESPACES).strip()


def _field_characters(root: ElementTree.Element) -> list[_FieldCharacter]:
    """Return every ``Field_Character`` an index label states, in order.

    Parameters:
        root: The label's root element.

    Returns:
        One entry per ``Field_Character``.
    """
    return [
        _FieldCharacter(
            name=_text(field, 'pds:name'),
            number=int(_text(field, 'pds:field_number')),
            location=int(_text(field, 'pds:field_location')),
            length=int(_text(field, 'pds:field_length')),
            data_type=_text(field, 'pds:data_type'),
            unit=_text(field, 'pds:unit'),
            description=' '.join(_text(field, 'pds:description').split()),
            missing_constant=_text(field, 'pds:Special_Constants/pds:missing_constant'),
        )
        for field in root.iterfind(f'{RECORD}/pds:Field_Character', PDS4_NAMESPACES)
    ]


@dataclass(frozen=True)
class _IndexProduct:
    """One index table and its label, as a bundle holds them.

    Attributes:
        lines: The table's lines, each with its line feed: the header line, then the
            records.
        label: The label's root element.
    """

    lines: list[str]
    label: ElementTree.Element

    @property
    def header_names(self) -> list[str]:
        """The names the header line gives, in order."""
        return self.lines[0].rstrip('\n').split(',')

    @property
    def records(self) -> list[str]:
        """The table's records, each with its line feed."""
        return self.lines[1:]


def _index_product(miscellaneous: Path, name: str) -> _IndexProduct:
    """Read one index table and its label out of a bundle's miscellaneous directory.

    Parameters:
        miscellaneous: The bundle's ``miscellaneous`` directory.
        name: The product's name, the stem of its table and its label.

    Returns:
        The product.
    """
    return _IndexProduct(
        lines=(miscellaneous / f'{name}.tab').read_bytes().decode('ascii').splitlines(True),
        label=ElementTree.parse(miscellaneous / f'{name}.lblx').getroot(),
    )


def _index_products(miscellaneous: Path) -> dict[str, _IndexProduct]:
    """Read both index tables and their labels out of a bundle's miscellaneous directory.

    Parameters:
        miscellaneous: The bundle's ``miscellaneous`` directory.

    Returns:
        Each product, by its name.
    """
    return {name: _index_product(miscellaneous, name) for name in INDEX_NAMES}


def _cohort_index_products(
    cohort: CohortCassiniISSSaturn, tmp_path: Path
) -> dict[str, _IndexProduct]:
    """Build the cohort's bundle and read both of its index products.

    Parameters:
        cohort: The Cassini ISS Saturn cohort.
        tmp_path: Base temporary directory for the bundle.

    Returns:
        Each product, by its name.
    """
    env = write_cohort_bundle(cohort, tmp_path, NAVIGATED_STUBS)
    return _index_products(env.bundle_dir / 'miscellaneous')


def test_each_index_label_s_header_is_the_header_line_and_its_table_follows_it(
    cassini_cohort: CohortCassiniISSSaturn, tmp_path: Path
) -> None:
    """The ``Header`` is the table's first line, its line feed included; the table follows.

    A reader takes the table's records from the offset the ``Table_Character`` states, so
    that offset has to be where the header line ends.
    """
    products = _cohort_index_products(cassini_cohort, tmp_path)
    header_lengths = {name: len(product.lines[0]) for name, product in products.items()}
    object_lengths = {
        name: int(_text(product.label, f'{AREA}/pds:Header/pds:object_length'))
        for name, product in products.items()
    }
    offsets = {
        name: int(_text(product.label, f'{AREA}/pds:Table_Character/pds:offset'))
        for name, product in products.items()
    }
    assert object_lengths == header_lengths
    assert offsets == header_lengths


def test_each_index_label_counts_its_table_s_rows_as_its_records(
    cassini_cohort: CohortCassiniISSSaturn, tmp_path: Path
) -> None:
    """``records`` is the number of rows, the header line not counted.

    The cohort's two navigated images give the bodies table a row each, one body in each,
    and the rings table one row, for the one image with ring backplanes.
    """
    products = _cohort_index_products(cassini_cohort, tmp_path)
    rows = {name: len(product.records) for name, product in products.items()}
    records = {
        name: int(_text(product.label, f'{AREA}/pds:Table_Character/pds:records'))
        for name, product in products.items()
    }
    assert rows == {'global_bodies_index': 2, 'global_rings_index': 1}
    assert records == rows


def test_each_index_label_counts_its_table_s_columns_as_its_fields(
    cassini_cohort: CohortCassiniISSSaturn, tmp_path: Path
) -> None:
    """``fields`` is the number of columns the header line names."""
    products = _cohort_index_products(cassini_cohort, tmp_path)
    columns = {name: len(product.header_names) for name, product in products.items()}
    fields = {
        name: int(_text(product.label, f'{RECORD}/pds:fields'))
        for name, product in products.items()
    }
    assert fields == columns


def test_every_field_character_lands_on_the_column_it_names(
    cassini_cohort: CohortCassiniISSSaturn, tmp_path: Path
) -> None:
    """Each ``Field_Character`` names a column, and its location and length are that column's.

    The fields a label states, in order, name the columns the header line names, each
    numbered by its position from 1.  Laid end to end with a comma between them, the
    fields' bytes rebuild every record exactly, so each field begins where its column's
    value begins and ends where it ends, and every record is the ``record_length`` the
    label states.
    """
    products = _cohort_index_products(cassini_cohort, tmp_path)
    described = {name: _field_characters(product.label) for name, product in products.items()}
    names = {name: [field.name for field in fields] for name, fields in described.items()}
    numbers = {name: [field.number for field in fields] for name, fields in described.items()}
    rebuilt = {
        name: [
            ','.join(field.value(record) for field in described[name]) + '\n'
            for record in product.records
        ]
        for name, product in products.items()
    }
    lengths = {
        name: {len(record) for record in product.records} for name, product in products.items()
    }
    record_lengths = {
        name: {int(_text(product.label, f'{RECORD}/pds:record_length'))}
        for name, product in products.items()
    }
    assert names == {name: product.header_names for name, product in products.items()}
    assert numbers == {name: list(range(1, len(fields) + 1)) for name, fields in described.items()}
    assert rebuilt == {name: product.records for name, product in products.items()}
    assert lengths == record_lengths


FIXED_COLUMNS = frozenset(
    {
        'pds:logical_identifier',
        'body_name',
        'file_spec',
        'pds:start_date_time',
        'pds:stop_date_time',
    }
)
"""The columns an index table gives whatever planes the configuration declares."""


def _configured_statistic_fields(
    entries: Sequence[Mapping[str, Any]],
) -> list[tuple[str, str, str, str]]:
    """Return what each statistic field of a table has to state, from its configuration.

    Parameters:
        entries: The table's configured planes, ``backplanes.bodies`` or
            ``backplanes.rings``, in order.

    Returns:
        For each plane its minimum column and then its maximum: the column's name, the
        data type its entry gives, the unit its statistic is in, and the description its
        own entry gives, its whitespace collapsed to single spaces.
    """
    return [
        (
            entry['index'][end]['name'],
            entry['index']['data_type'],
            statistics_units(entry['units']),
            ' '.join(entry['index'][end]['description'].split()),
        )
        for entry in entries
        for end in ('minimum', 'maximum')
    ]


def test_every_field_states_its_data_type_and_each_statistic_field_its_unit_and_description(
    cassini_cohort: CohortCassiniISSSaturn, tmp_path: Path
) -> None:
    """Each field states its data type, and each statistic field its unit and description.

    A fixed column's data type is the one its values are, in both tables:
    ``pds:logical_identifier`` an ``ASCII_LID``, ``body_name`` and ``file_spec`` each an
    ``ASCII_String``, and the two times each an ``ASCII_Date_Time_YMD_UTC``.  A statistic
    field's data type and description are the ones the column's own entry in the
    configuration gives, and its unit is the one its plane's statistic is in: the plane's
    unit restated through ``statistics_units``, so an angle in radians is a column in
    degrees.  Every field after the fixed columns is held, in order, to the configured
    planes' minimum and maximum columns, in both tables.
    """
    products = _cohort_index_products(cassini_cohort, tmp_path)
    backplanes = cassini_cohort.dataset().config.backplanes
    configured = {
        'global_bodies_index': _configured_statistic_fields(backplanes.bodies),
        'global_rings_index': _configured_statistic_fields(backplanes.rings),
    }
    stated = {
        name: [
            (field.name, field.data_type, field.unit, field.description)
            for field in _field_characters(product.label)
            if field.name not in FIXED_COLUMNS
        ]
        for name, product in products.items()
    }
    fixed = {
        name: {
            field.name: field.data_type
            for field in _field_characters(product.label)
            if field.name in FIXED_COLUMNS
        }
        for name, product in products.items()
    }
    both = list(INDEX_NAMES)
    assert stated == configured
    assert {
        name: types['pds:logical_identifier'] for name, types in fixed.items()
    } == dict.fromkeys(both, 'ASCII_LID')
    assert fixed['global_bodies_index']['body_name'] == 'ASCII_String'
    assert {name: types['file_spec'] for name, types in fixed.items()} == dict.fromkeys(
        both, 'ASCII_String'
    )
    assert {name: types['pds:start_date_time'] for name, types in fixed.items()} == dict.fromkeys(
        both, 'ASCII_Date_Time_YMD_UTC'
    )
    assert {name: types['pds:stop_date_time'] for name, types in fixed.items()} == dict.fromkeys(
        both, 'ASCII_Date_Time_YMD_UTC'
    )


def test_each_index_label_names_the_table_beside_it(
    cassini_cohort: CohortCassiniISSSaturn, tmp_path: Path
) -> None:
    """Each index label's ``file_name`` is the table of its own stem, beside it.

    A PDS4 label names its file with no directory part, so the file it names has to be
    the table in the label's own directory, which is the one read here.
    """
    products = _cohort_index_products(cassini_cohort, tmp_path)
    named = {
        name: _text(product.label, f'{AREA}/pds:File/pds:file_name')
        for name, product in products.items()
    }
    assert named == {name: f'{name}.tab' for name in INDEX_NAMES}


TIME_COORDINATES = 'pds:Observation_Area/pds:Time_Coordinates'
"""Where a data label states its exposure's start and stop."""

TIME_FIELDS = ('pds:start_date_time', 'pds:stop_date_time')
"""The two fields of each index table giving a row's exposure start and stop."""


def test_each_row_s_times_are_the_start_and_stop_its_data_label_states(
    cassini_cohort: CohortCassiniISSSaturn, tmp_path: Path
) -> None:
    """A row's exposure start and stop are the ones the data label of its product states.

    Each table is read by the fields its label states, and each row's two times are held
    to the ``start_date_time`` and ``stop_date_time`` of the data label whose LID the row
    gives: two rows of the bodies table and one of the rings table.
    """
    env = write_cohort_bundle(cassini_cohort, tmp_path, NAVIGATED_STUBS)
    stated: dict[str, tuple[str, ...]] = {}
    for data_label in (env.bundle_dir / 'data').rglob('*_backplanes.lblx'):
        root = ElementTree.parse(data_label).getroot()
        lid = _text(root, 'pds:Identification_Area/pds:logical_identifier')
        stated[lid] = tuple(_text(root, f'{TIME_COORDINATES}/{name}') for name in TIME_FIELDS)
    rows: list[tuple[str, tuple[str, ...]]] = []
    for product in _index_products(env.bundle_dir / 'miscellaneous').values():
        fields = {field.name: field for field in _field_characters(product.label)}
        for record in product.records:
            lid = fields['pds:logical_identifier'].value(record).strip()
            times = tuple(fields[name].value(record).strip() for name in TIME_FIELDS)
            rows.append((lid, times))
    assert len(rows) == 3
    assert rows == [(lid, stated[lid]) for lid, _ in rows]


def test_the_rings_index_states_the_plain_ring_longitude_range(
    cassini_cohort: CohortCassiniISSSaturn, tmp_path: Path
) -> None:
    """The ring image's row states the plain least and greatest ring longitude.

    Its data label states the range wrapped at zero, 216.000 to 204.706, and the index
    columns are named for the plain range, 0 to 360.
    """
    env = write_cohort_bundle(cassini_cohort, tmp_path, NAVIGATED_STUBS)
    header, row = read_index_rows(env.bundle_dir / 'miscellaneous' / 'global_rings_index.tab')
    metadata_path = cassini_cohort.backplane_results_root / f'{RINGS_STUB}_backplane_metadata.json'
    rings = json.loads(metadata_path.read_text(encoding='utf-8'))['rings']
    statistic = rings['backplanes']['ring_longitude']
    plain = [INDEX_VALUE_FORMATS['deg'].render(statistic[end]) for end in ('min', 'max')]
    stated = [row[header.index(f'{end}_ring_longitude')] for end in ('minimum', 'maximum')]
    assert stated == plain


def _shipped_index_templates(cohort: CohortCassiniISSSaturn) -> dict[str, str]:
    """Return the two index label templates the cohort's dataset ships, by name.

    Parameters:
        cohort: The Cassini ISS Saturn cohort, whose dataset names the template directory.

    Returns:
        Each template's text.
    """
    shipped = Path(cohort.dataset().pds4_bundle_template_dir())
    return {
        f'{name}.lblx': (shipped / f'{name}.lblx').read_text(encoding='utf-8')
        for name in INDEX_NAMES
    }


LATITUDE_ONLY = {'MOON': {'backplanes': {'latitude': {'min': -10.0, 'max': 20.0, 'units': 'deg'}}}}
"""One body's statistics: a latitude, in degrees as a radian plane's statistic is, alone."""


RADIUS_ONLY = ring_metadata({'radius': {'min': 80000.0, 'max': 90000.0, 'units': 'km'}})
"""The rings' statistics: a radius, in kilometers, alone."""


def _index_products_over(
    tmp_path: Path,
    templates: dict[str, str],
    *,
    bodies: list[dict[str, Any]],
    rings: list[dict[str, Any]],
) -> dict[str, _IndexProduct]:
    """Write one image's index tables and their labels, under the given planes.

    The image's one body has a latitude statistic and no other, and its rings a radius
    statistic and no other, so any other plane the configuration declares has no
    statistic in either table.

    Parameters:
        tmp_path: The directory the bundle environment is built in.
        templates: The index label templates to render, by name.
        bodies: The ``backplanes.bodies`` entries the configuration declares.
        rings: The ``backplanes.rings`` entries the configuration declares.

    Returns:
        Both tables and their labels, by name.
    """
    env = make_bundle_env(tmp_path, template_contents=templates, bodies=bodies, rings=rings)
    touch_label(env.bundle_dir / 'data', 'shard0/1234567890w')
    write_supplemental(
        env.bundle_dir / 'data', 'shard0/1234567890w', bodies=LATITUDE_ONLY, rings=RADIUS_ONLY
    )
    generate_global_index_files(
        FCPath(env.bundle_results_root), env.dataset.as_dataset(), MAIN_LOGGER
    )
    return _index_products(env.bundle_dir / 'miscellaneous')


def test_a_plane_added_to_the_configuration_adds_a_column_and_a_field_character(
    cassini_cohort: CohortCassiniISSSaturn, tmp_path: Path
) -> None:
    """A body plane added to the configuration adds its two columns and two fields.

    The table's columns and the label's ``Field_Character`` blocks are built from the same
    configured list, so the added plane's minimum and maximum follow the columns before
    them in both, under the names its entry gives them, and nothing else changes.
    """
    templates = _shipped_index_templates(cassini_cohort)
    latitude = index_entry('latitude', 'rad')
    resolution = index_entry('resolution', 'km/pixel')
    radius = [index_entry('radius', 'km')]
    before = _index_products_over(tmp_path / 'before', templates, bodies=[latitude], rings=radius)[
        'global_bodies_index'
    ]
    after = _index_products_over(
        tmp_path / 'after', templates, bodies=[latitude, resolution], rings=radius
    )['global_bodies_index']
    added = ['minimum_resolution', 'maximum_resolution']
    fields_before = [field.name for field in _field_characters(before.label)]
    fields_after = [field.name for field in _field_characters(after.label)]
    assert after.header_names == before.header_names + added
    assert fields_after == fields_before + added


@pytest.mark.parametrize('index', INDEX_NAMES, ids=['bodies', 'rings'])
def test_a_missing_statistic_s_cell_holds_the_constant_its_field_declares(
    cassini_cohort: CohortCassiniISSSaturn, tmp_path: Path, index: str
) -> None:
    """A cell whose plane has no statistic holds its field's missing constant, as declared.

    Neither the body nor the rings have a resolution statistic, so both resolution cells
    of each table hold the configured masked value.  Each of those fields declares that
    value as its missing constant in the spelling its cells have.  The NASA PDS
    ``validate`` tool accepts such a cell as a real, as it would any number, so it is the
    declaration that tells a reader the value means the plane has no statistic there.

    Parameters:
        cassini_cohort: The session's Cassini ISS Saturn cohort, whose dataset ships the
            templates.
        tmp_path: Base temporary directory for this test's bundle.
        index: The table whose resolution cells are read.
    """
    products = _index_products_over(
        tmp_path,
        _shipped_index_templates(cassini_cohort),
        bodies=[index_entry('latitude', 'rad'), index_entry('resolution', 'km/pixel')],
        rings=[index_entry('radius', 'km'), index_entry('resolution', 'km/pixel')],
    )
    product = products[index]
    fields = {field.name: field for field in _field_characters(product.label)}
    missing = [fields['minimum_resolution'], fields['maximum_resolution']]
    cells = [field.value(product.records[0]).strip() for field in missing]
    declared = [field.missing_constant for field in missing]
    masked_value = float(DEFAULT_CONFIG.backplanes.masked_value)
    assert cells == declared
    assert [float(constant) for constant in declared] == [masked_value, masked_value]


def test_each_primary_member_of_the_miscellaneous_inventory_is_a_label_beside_it(
    cassini_cohort: CohortCassiniISSSaturn, tmp_path: Path
) -> None:
    """The inventory's primary members are the index labels in the collection's directory.

    Each ``P`` line names, as its LID and version, an index product's label in the
    ``miscellaneous`` directory beside the inventory, and each such label is named.
    """
    env = write_cohort_bundle(cassini_cohort, tmp_path, NAVIGATED_STUBS)
    miscellaneous = env.bundle_dir / 'miscellaneous'
    rows = read_csv_rows(miscellaneous / 'collection_miscellaneous.csv')
    primaries = sorted(lidvid for status, lidvid in rows if status == 'P')
    roots = [
        ElementTree.parse(label).getroot()
        for label in miscellaneous.glob('*.lblx')
        if not label.name.startswith('collection_')
    ]
    identification = 'pds:Identification_Area'
    labels = sorted(
        f'{_text(root, f"{identification}/pds:logical_identifier")}'
        f'::{_text(root, f"{identification}/pds:version_id")}'
        for root in roots
    )
    assert len(primaries) == len(INDEX_NAMES)
    assert primaries == labels


def test_the_miscellaneous_labels_state_the_miscellaneous_types(
    cassini_cohort: CohortCassiniISSSaturn, tmp_path: Path
) -> None:
    """The collection and the bundle's entry for it state the miscellaneous types.

    Each field's schema permits other collections' values as well, so a label carrying
    another collection's type passes every schema check: the collection's type, and the
    reference type of the bundle's member entry naming the collection.
    """
    env = write_cohort_bundle(cassini_cohort, tmp_path, NAVIGATED_STUBS)
    collection_label = env.bundle_dir / 'miscellaneous' / 'collection_miscellaneous.lblx'
    collection = ElementTree.parse(collection_label).getroot()
    lid = _text(collection, 'pds:Identification_Area/pds:logical_identifier')
    bundle = ElementTree.parse(env.bundle_dir / 'bundle.lblx').getroot()
    reference_types = [
        _text(entry, 'pds:reference_type')
        for entry in bundle.iterfind('pds:Bundle_Member_Entry', PDS4_NAMESPACES)
        if _text(entry, 'pds:lid_reference') == lid
    ]
    assert _text(collection, 'pds:Collection/pds:collection_type') == 'Miscellaneous'
    assert reference_types == ['bundle_has_miscellaneous_collection']
