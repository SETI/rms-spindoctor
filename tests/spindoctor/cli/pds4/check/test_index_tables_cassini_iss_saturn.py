"""The Cassini ISS Saturn bundle's bodies index table held to its layout and its tree.

Each test copies the cohort's bundle and changes the bodies table or its label.  The
layout tests hold the index layout check to finding the change -- a header line naming a
field otherwise, a byte other than a comma after a field, a field apart from the one
before it, a last field short of the record delimiter -- and the record tests hold the
integrity check to finding it: a record of a product the tree no longer holds, a record
whose start time or data label differs from its product's, and a product missing a
record its supplemental file calls for.
"""

from pathlib import Path

import pytest
from tests.mini_nav_results.cohort import WrittenCohorts
from tests.mini_nav_results.cohort_cassini import LIMB_STUB, RINGS_STUB, CohortCassiniISSSaturn

from spindoctor.cli.pds4.check.findings import CheckName, Finding
from spindoctor.cli.pds4.check.index_tables import index_layout_findings
from spindoctor.cli.pds4.check.integrity import integrity_findings

from ..cohort_bundle import write_cohort_bundle
from .controls import (
    copy_bundle,
    parse,
    parsed_labels,
    substitute_once,
    table_field,
    table_field_names,
    table_records,
)

NAVIGATED_STUBS = (LIMB_STUB, RINGS_STUB)
"""The cohort's two navigated images, by results path stub."""

BODIES = 'miscellaneous/global_bodies_index.lblx'
"""The bodies index table's label."""

HEADER = '/Product_Ancillary/File_Area_Ancillary/Header'
"""Where an index table's label describes its header line."""

TABLE = '/Product_Ancillary/File_Area_Ancillary/Table_Character'
"""Where an index table's label describes its records."""


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


def _layout(bundle_dir: Path) -> list[Finding]:
    """Hold the bodies table to the layout the summary pass writes.

    Parameters:
        bundle_dir: The bundle's directory.

    Returns:
        The index layout check's findings.
    """
    return index_layout_findings(BODIES, bundle_dir / BODIES, parse(bundle_dir / BODIES))


def _check(bundle_dir: Path) -> list[Finding]:
    """Check a bundle's tree as a whole.

    Parameters:
        bundle_dir: The bundle's directory.

    Returns:
        The integrity check's findings.
    """
    return integrity_findings(bundle_dir, parsed_labels(bundle_dir))


def _cell(bundle_dir: Path, number: int, name: str) -> str:
    """Return one value of the bodies table.

    Parameters:
        bundle_dir: The bundle's directory.
        number: The record's number, counted from 1.
        name: The field's name.

    Returns:
        The value, its padding stripped.
    """
    _, start, stop = table_field(bundle_dir, BODIES, name)
    _, records = table_records(bundle_dir, BODIES)
    return records[number - 1][start:stop].decode('ascii').strip(' ')


def _overwrite(bundle_dir: Path, number: int, start: int, new: bytes) -> None:
    """Overwrite bytes of one record of the bodies table.

    Parameters:
        bundle_dir: The bundle's directory.
        number: The record's number, counted from 1.
        start: Where in the record the bytes begin, counted from 0.
        new: The bytes.
    """
    header_length, records = table_records(bundle_dir, BODIES)
    offset = header_length + sum(len(record) for record in records[: number - 1]) + start
    table = (bundle_dir / BODIES).with_suffix('.tab')
    data = bytearray(table.read_bytes())
    data[offset : offset + len(new)] = new
    table.write_bytes(bytes(data))


def test_a_header_line_naming_a_field_otherwise_is_found(
    plain_bundle: Path, tmp_path: Path
) -> None:
    """A header line whose name for a field is not the label's is found."""
    bundle = copy_bundle(plain_bundle, tmp_path)
    table = (bundle / BODIES).with_suffix('.tab')
    table.write_bytes(
        table.read_bytes().replace(b'minimum_body_longitude', b'maximum_body_longitude', 1)
    )
    position = table_field_names(bundle, BODIES).index('minimum_body_longitude') + 1
    expected = Finding(
        BODIES,
        CheckName.TABLE,
        HEADER,
        f"the header line names field {position} 'maximum_body_longitude', but the label "
        "names it 'minimum_body_longitude'",
    )
    assert expected in _layout(bundle)


def test_a_record_holding_other_than_a_comma_after_a_field_is_found(
    plain_bundle: Path, tmp_path: Path
) -> None:
    """A cell one byte wider than its field puts a digit where the comma after it goes."""
    bundle = copy_bundle(plain_bundle, tmp_path)
    location, _, stop = table_field(bundle, BODIES, 'minimum_body_longitude')
    _overwrite(bundle, 1, stop, b'7')
    expected = Finding(
        BODIES, CheckName.TABLE, location, "record 1 holds '7' after the field, not a comma"
    )
    assert expected in _layout(bundle)


def test_a_field_apart_from_the_one_before_it_is_found(plain_bundle: Path, tmp_path: Path) -> None:
    """A field moved on a byte, and a byte shorter, lies two bytes past the one before."""
    bundle = copy_bundle(plain_bundle, tmp_path)
    location, start, stop = table_field(bundle, BODIES, 'body_name')
    substitute_once(
        bundle / BODIES,
        r'<field_location unit="byte">\d+</field_location>',
        f'<field_location unit="byte">{start + 2}</field_location>',
        within='body_name',
    )
    substitute_once(
        bundle / BODIES,
        r'<field_length unit="byte">\d+</field_length>',
        f'<field_length unit="byte">{stop - start - 1}</field_length>',
        within='body_name',
    )
    expected = Finding(
        BODIES,
        CheckName.TABLE,
        location,
        'begins 2 byte(s) after the field before it, pds:logical_identifier, where a comma '
        'alone lies between two fields',
    )
    assert expected in _layout(bundle)


def test_a_last_field_short_of_the_record_delimiter_is_found(
    plain_bundle: Path, tmp_path: Path
) -> None:
    """A last field one byte shorter leaves a byte between it and the record delimiter."""
    bundle = copy_bundle(plain_bundle, tmp_path)
    last = table_field_names(bundle, BODIES)[-1]
    location, start, stop = table_field(bundle, BODIES, last)
    substitute_once(
        bundle / BODIES,
        r'<field_length unit="byte">\d+</field_length>',
        f'<field_length unit="byte">{stop - start - 1}</field_length>',
        within=last,
    )
    expected = Finding(
        BODIES,
        CheckName.TABLE,
        location,
        'ends 1 byte(s) before the record delimiter, where the last field ends at it',
    )
    assert expected in _layout(bundle)


def test_a_record_of_a_product_the_tree_does_not_hold_is_found(
    plain_bundle: Path, tmp_path: Path
) -> None:
    """A product removed from the tree and its inventories leaves an orphan record."""
    bundle = copy_bundle(plain_bundle, tmp_path)
    location, _, _ = table_field(bundle, BODIES, 'pds:logical_identifier')
    lid = _cell(bundle, 1, 'pds:logical_identifier')
    name = lid.rsplit(':', 1)[-1]
    for path in sorted(bundle.rglob(f'{name}_*')):
        path.unlink()
    for inventory in ('data/collection_data.csv', 'browse/collection_browse.csv'):
        path = bundle / inventory
        lines = path.read_text(encoding='ascii').splitlines(keepends=True)
        path.write_text(
            ''.join(line for line in lines if f':{name}::' not in line), encoding='ascii'
        )
    expected = Finding(
        BODIES,
        CheckName.INTEGRITY,
        location,
        f'record 1 names {lid}, which no label of the tree declares',
    )
    assert expected in _check(bundle)


def test_a_record_whose_start_differs_from_its_products_is_found(
    plain_bundle: Path, tmp_path: Path
) -> None:
    """A record's start time a millisecond off its data label's is found."""
    bundle = copy_bundle(plain_bundle, tmp_path)
    location, start, _ = table_field(bundle, BODIES, 'pds:start_date_time')
    value = _cell(bundle, 1, 'pds:start_date_time')
    moved = f'{value[:-2]}{(int(value[-2]) + 1) % 10}Z'
    label = _cell(bundle, 1, 'file_spec')
    _overwrite(bundle, 1, start, moved.encode('ascii'))
    expected = Finding(
        BODIES,
        CheckName.INTEGRITY,
        location,
        f'record 1 gives {moved!r}, but {label} states {value!r}',
    )
    assert expected in _check(bundle)


def test_a_record_naming_another_label_than_its_products_is_found(
    plain_bundle: Path, tmp_path: Path
) -> None:
    """A record whose ``file_spec`` is not its product's label is found."""
    bundle = copy_bundle(plain_bundle, tmp_path)
    location, start, _ = table_field(bundle, BODIES, 'file_spec')
    label = _cell(bundle, 1, 'file_spec')
    other = label.replace('.lblx', '.lbly')
    lid = _cell(bundle, 1, 'pds:logical_identifier')
    _overwrite(bundle, 1, start, other.encode('ascii'))
    expected = Finding(
        BODIES,
        CheckName.INTEGRITY,
        location,
        f'record 1 gives {other!r}, but the label declaring {lid} is {label}',
    )
    assert expected in _check(bundle)


def test_a_product_without_a_record_its_supplemental_file_calls_for_is_found(
    plain_bundle: Path, tmp_path: Path
) -> None:
    """A bodies table that lost its second record holds none for that record's body."""
    bundle = copy_bundle(plain_bundle, tmp_path)
    lid = _cell(bundle, 2, 'pds:logical_identifier')
    body = _cell(bundle, 2, 'body_name')
    header_length, records = table_records(bundle, BODIES)
    table = (bundle / BODIES).with_suffix('.tab')
    table.write_bytes(table.read_bytes()[: header_length + len(records[0])])
    expected = Finding(
        BODIES,
        CheckName.INTEGRITY,
        TABLE,
        f'holds 0 record(s) for {body} in {lid}, where its supplemental file calls for 1',
    )
    assert expected in _check(bundle)
