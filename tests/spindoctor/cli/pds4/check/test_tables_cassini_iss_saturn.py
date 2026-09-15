"""The Cassini ISS Saturn bundle's tables read through their labels, broken on purpose.

Neither schema reads a table, so these breaks pass both: a field's length, the header's
length, the table's offset, a record one byte short, a respelled missing constant, a unit
and a data type, each of the global index tables; a field's number, two fields that
overlap, and bytes before the first object, of the bodies table; and an inventory's
record count, its field count, its record endings and its last delimiter.  Each test
copies the cohort's bundle, changes one table or label, and holds the table reader to
finding the change.
"""

from pathlib import Path

import pytest
from tests.mini_nav_results.cohort import WrittenCohorts
from tests.mini_nav_results.cohort_cassini import LIMB_STUB, RINGS_STUB, CohortCassiniISSSaturn

from spindoctor.cli.pds4.check.findings import CheckName, Finding
from spindoctor.cli.pds4.check.schemas import label_schema
from spindoctor.cli.pds4.check.statistic_columns import (
    statistic_column_findings,
    statistic_columns,
)
from spindoctor.cli.pds4.check.tables import table_findings
from spindoctor.config import DEFAULT_CONFIG

from ..cohort_bundle import write_cohort_bundle
from .controls import (
    copy_bundle,
    parse,
    substitute_once,
    table_field,
    table_field_names,
    table_records,
)

NAVIGATED_STUBS = (LIMB_STUB, RINGS_STUB)
"""The cohort's two navigated images, by results path stub."""

BODIES = 'miscellaneous/global_bodies_index.lblx'
"""The bodies index table's label."""

RINGS = 'miscellaneous/global_rings_index.lblx'
"""The rings index table's label."""

DATA_INVENTORY = 'data/collection_data.lblx'
"""The data collection's label, which describes its inventory."""

HEADER = '/Product_Ancillary/File_Area_Ancillary/Header'
"""Where an index table's label describes its header line."""

TABLE = '/Product_Ancillary/File_Area_Ancillary/Table_Character'
"""Where an index table's label describes its records."""

INVENTORY = '/Product_Collection/File_Area_Inventory/Inventory'
"""Where a collection's label describes its inventory."""


@pytest.fixture(scope='module')
def plain_bundle(
    mini_nav_cohorts: WrittenCohorts, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    """Return the cohort's bundle, written once for the module, without a user guide.

    Parameters:
        mini_nav_cohorts: What the session's cohorts are written by.
        tmp_path_factory: Factory the bundle's directory is made under.

    Returns:
        The bundle's directory, which no test changes.
    """
    cohort = mini_nav_cohorts(CohortCassiniISSSaturn)
    return write_cohort_bundle(cohort, tmp_path_factory.mktemp('plain'), NAVIGATED_STUBS).bundle_dir


def _read(bundle_dir: Path, file: str) -> list[Finding]:
    """Read every table a label describes, and hold its statistic columns to the configuration.

    Parameters:
        bundle_dir: The bundle's directory.
        file: The label's path relative to it.

    Returns:
        The table reader's findings and the statistic column check's.
    """
    document = parse(bundle_dir / file)
    schema = label_schema(file, document).schema
    return table_findings(file, bundle_dir / file, document, schema) + statistic_column_findings(
        file, document, statistic_columns(DEFAULT_CONFIG)
    )


def test_a_field_longer_than_its_values_is_found(plain_bundle: Path, tmp_path: Path) -> None:
    """A ``field_length`` one byte long takes in the comma, which no real holds."""
    bundle = copy_bundle(plain_bundle, tmp_path)
    location, start, stop = table_field(bundle, BODIES, 'minimum_body_longitude')
    _, records = table_records(bundle, BODIES)
    substitute_once(
        bundle / BODIES,
        r'<field_length unit="byte">\d+</field_length>',
        f'<field_length unit="byte">{stop - start + 1}</field_length>',
        within='minimum_body_longitude',
    )
    value = records[0][start : stop + 1].decode('ascii').strip(' ')
    expected = Finding(
        BODIES,
        CheckName.TABLE,
        location,
        f'record 1 holds {value!r}, not ASCII_Real, and 1 more record(s) like it',
    )
    assert expected in _read(bundle, BODIES)


def test_a_header_shorter_than_its_line_is_found(plain_bundle: Path, tmp_path: Path) -> None:
    """A header's ``object_length`` one short leaves the header line's line feed out."""
    bundle = copy_bundle(plain_bundle, tmp_path)
    header_length, _ = table_records(bundle, BODIES)
    substitute_once(
        bundle / BODIES,
        r'<object_length unit="byte">\d+</object_length>',
        f'<object_length unit="byte">{header_length - 1}</object_length>',
    )
    expected = Finding(
        BODIES,
        CheckName.TABLE,
        HEADER,
        f'object_length is {header_length - 1}, but {header_length} bytes lie between its '
        'offset 0 and the next object',
    )
    assert expected in _read(bundle, BODIES)


def test_a_table_offset_one_short_is_found(plain_bundle: Path, tmp_path: Path) -> None:
    """A ``Table_Character`` ``offset`` one short begins the records on the header's line feed."""
    bundle = copy_bundle(plain_bundle, tmp_path)
    header_length, records = table_records(bundle, RINGS)
    record_length = len(records[0])
    substitute_once(
        bundle / RINGS,
        r'(<offset unit="byte">)\d+(</offset>\s*<records>)',
        rf'\g<1>{header_length - 1}\g<2>',
    )
    expected = Finding(
        RINGS,
        CheckName.TABLE,
        TABLE,
        f'{len(records)} record(s) of {record_length} bytes are {len(records) * record_length} '
        f'bytes, but {len(records) * record_length + 1} lie between its offset '
        f'{header_length - 1} and the end of the file',
    )
    assert expected in _read(bundle, RINGS)


def test_a_record_one_byte_short_is_found(plain_bundle: Path, tmp_path: Path) -> None:
    """A record that lost one byte leaves the table a byte short of its records."""
    bundle = copy_bundle(plain_bundle, tmp_path)
    header_length, records = table_records(bundle, RINGS)
    record_length = len(records[0])
    substitute_once(bundle / RINGS.replace('.lblx', '.tab'), r',0\.000,360\.000,', ',0.00,360.000,')
    expected = Finding(
        RINGS,
        CheckName.TABLE,
        TABLE,
        f'1 record(s) of {record_length} bytes are {record_length} bytes, but '
        f'{record_length - 1} lie between its offset {header_length} and the end of the file',
    )
    assert expected in _read(bundle, RINGS)


def test_a_missing_constant_respelled_is_found(plain_bundle: Path, tmp_path: Path) -> None:
    """A ``missing_constant`` of ``-999`` is not the masked value as a degrees column writes it."""
    bundle = copy_bundle(plain_bundle, tmp_path)
    substitute_once(
        bundle / BODIES,
        r'<missing_constant>-999\.000</missing_constant>',
        '<missing_constant>-999</missing_constant>',
        within='geom:minimum_phase_angle',
    )
    location, _, _ = table_field(bundle, BODIES, 'geom:minimum_phase_angle')
    expected = Finding(
        BODIES,
        CheckName.TABLE,
        location,
        "geom:minimum_phase_angle declares the missing constant '-999', but the masked value "
        "in its format is '-999.000'",
    )
    assert expected in _read(bundle, BODIES)


def test_an_angle_column_in_radians_is_found(plain_bundle: Path, tmp_path: Path) -> None:
    """A ``unit`` of ``rad`` on an angle column is not the degrees its statistic is in."""
    bundle = copy_bundle(plain_bundle, tmp_path)
    substitute_once(
        bundle / RINGS,
        r'<unit>deg</unit>',
        '<unit>rad</unit>',
        within='rings:minimum_emission_angle',
    )
    location, _, _ = table_field(bundle, RINGS, 'rings:minimum_emission_angle')
    expected = Finding(
        RINGS,
        CheckName.TABLE,
        location,
        "rings:minimum_emission_angle states rad, but its plane's statistic is in deg",
    )
    assert expected in _read(bundle, RINGS)


def test_a_column_of_reals_typed_integer_is_found(plain_bundle: Path, tmp_path: Path) -> None:
    """A ``data_type`` of ``ASCII_Integer`` over values written with a decimal is refused."""
    bundle = copy_bundle(plain_bundle, tmp_path)
    substitute_once(
        bundle / RINGS,
        r'<data_type>ASCII_Real</data_type>',
        '<data_type>ASCII_Integer</data_type>',
        within='rings:minimum_ring_radius',
    )
    location, start, stop = table_field(bundle, RINGS, 'rings:minimum_ring_radius')
    _, records = table_records(bundle, RINGS)
    value = records[0][start:stop].decode('ascii').strip(' ')
    expected = Finding(
        RINGS, CheckName.TABLE, location, f'record 1 holds {value!r}, not ASCII_Integer'
    )
    assert expected in _read(bundle, RINGS)


def test_a_field_numbered_out_of_its_place_is_found(plain_bundle: Path, tmp_path: Path) -> None:
    """A ``field_number`` other than the field's position in its record is found."""
    bundle = copy_bundle(plain_bundle, tmp_path)
    substitute_once(
        bundle / BODIES,
        r'<field_number>\d+</field_number>',
        '<field_number>99</field_number>',
        within='minimum_body_longitude',
    )
    location, _, _ = table_field(bundle, BODIES, 'minimum_body_longitude')
    position = table_field_names(bundle, BODIES).index('minimum_body_longitude') + 1
    expected = Finding(
        BODIES,
        CheckName.TABLE,
        location,
        f'field_number is 99, but the field is number {position} of its record',
    )
    assert expected in _read(bundle, BODIES)


def test_a_field_overlapping_the_one_before_it_is_found(plain_bundle: Path, tmp_path: Path) -> None:
    """A field moved two bytes back takes in the last byte of the field before it."""
    bundle = copy_bundle(plain_bundle, tmp_path)
    location, start, _ = table_field(bundle, BODIES, 'body_name')
    substitute_once(
        bundle / BODIES,
        r'<field_location unit="byte">\d+</field_location>',
        f'<field_location unit="byte">{start - 1}</field_location>',
        within='body_name',
    )
    expected = Finding(
        BODIES, CheckName.TABLE, location, 'overlaps the field before it, pds:logical_identifier'
    )
    assert expected in _read(bundle, BODIES)


def test_bytes_before_the_first_object_of_a_file_are_found(
    plain_bundle: Path, tmp_path: Path
) -> None:
    """A header at offset 1 leaves the file's first byte described by no object."""
    bundle = copy_bundle(plain_bundle, tmp_path)
    substitute_once(
        bundle / BODIES, r'(<Header>\s*<offset unit="byte">)0(</offset>)', r'\g<1>1\g<2>'
    )
    expected = Finding(
        BODIES, CheckName.TABLE, HEADER, 'the file begins with 1 byte(s) no object describes'
    )
    assert expected in _read(bundle, BODIES)


def test_an_inventory_whose_records_count_one_too_many_is_found(
    plain_bundle: Path, tmp_path: Path
) -> None:
    """An inventory's ``records`` one more than its lines is found."""
    bundle = copy_bundle(plain_bundle, tmp_path)
    lines = (bundle / 'data' / 'collection_data.csv').read_bytes().count(b'\n')
    substitute_once(
        bundle / DATA_INVENTORY, r'<records>\d+</records>', f'<records>{lines + 1}</records>'
    )
    expected = Finding(
        DATA_INVENTORY,
        CheckName.TABLE,
        INVENTORY,
        f'records is {lines + 1}, but {lines} record(s) lie between its offset 0 and the end '
        'of the file',
    )
    assert expected in _read(bundle, DATA_INVENTORY)


def test_an_inventory_whose_fields_count_one_too_many_is_found(
    plain_bundle: Path, tmp_path: Path
) -> None:
    """An inventory's ``fields`` one more than the fields its record describes is found."""
    bundle = copy_bundle(plain_bundle, tmp_path)
    substitute_once(bundle / DATA_INVENTORY, r'<fields>2</fields>', '<fields>3</fields>')
    expected = Finding(
        DATA_INVENTORY,
        CheckName.TABLE,
        f'{INVENTORY}/Record_Delimited',
        'fields is 3, but the record describes 2',
    )
    assert expected in _read(bundle, DATA_INVENTORY)


def test_an_inventory_whose_records_end_in_a_carriage_return_is_found(
    plain_bundle: Path, tmp_path: Path
) -> None:
    """Records ending in a carriage return and a line feed are not ones ending in a line feed."""
    bundle = copy_bundle(plain_bundle, tmp_path)
    inventory = bundle / 'data' / 'collection_data.csv'
    data = inventory.read_bytes()
    inventory.write_bytes(data.replace(b'\n', b'\r\n'))
    more = data.count(b'\n') - 1
    expected = Finding(
        DATA_INVENTORY,
        CheckName.TABLE,
        INVENTORY,
        'record 1 holds a carriage return or a line feed that is not its record delimiter, '
        f'and {more} more record(s) like it',
    )
    assert expected in _read(bundle, DATA_INVENTORY)


def test_an_inventory_without_its_last_delimiter_is_found(
    plain_bundle: Path, tmp_path: Path
) -> None:
    """An inventory whose last record has lost its line feed does not end in its delimiter."""
    bundle = copy_bundle(plain_bundle, tmp_path)
    inventory = bundle / 'data' / 'collection_data.csv'
    inventory.write_bytes(inventory.read_bytes().removesuffix(b'\n'))
    expected = Finding(
        DATA_INVENTORY,
        CheckName.TABLE,
        INVENTORY,
        'the bytes between its offset 0 and the end of the file do not end in its record delimiter',
    )
    assert expected in _read(bundle, DATA_INVENTORY)


def test_a_missing_cell_spelled_otherwise_than_its_constant_is_found(
    plain_bundle: Path, tmp_path: Path
) -> None:
    """A cell holding the missing constant's number must hold it as the constant is spelled."""
    bundle = copy_bundle(plain_bundle, tmp_path)
    location, start, stop = table_field(bundle, BODIES, 'geom:minimum_phase_angle')
    header_length, _ = table_records(bundle, BODIES)
    table = (bundle / BODIES).with_suffix('.tab')
    data = bytearray(table.read_bytes())
    data[header_length + start : header_length + stop] = '-999.0'.rjust(stop - start).encode()
    table.write_bytes(bytes(data))
    expected = Finding(
        BODIES,
        CheckName.TABLE,
        location,
        "record 1 holds '-999.0', the missing constant '-999.000' spelled otherwise",
    )
    assert expected in _read(bundle, BODIES)
