"""The global index labels the Cassini ISS Saturn bundle ships, and the tables beside them.

What each index label says of its table: its ``Header`` is the table's header line and its
``Table_Character`` begins where that line ends; it counts the table's rows as its
records and its columns as its fields; and each ``Field_Character`` lands on the column it
names in every record.  Those are asked of the cohort's bundle.  Two more questions need a
configuration the test chooses, and are asked of the shipped templates over plumbing
inputs: that a plane added to the configuration adds a column to the table and a
``Field_Character`` to the label, and that a missing statistic's cell holds the constant
its field declares, in the spelling the label declares it.  The plumbing is tested over
stand-in templates in ``test_global_index.py``.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import pytest
from filecache import FCPath
from tests.mini_nav_results.cohort import WrittenCohorts
from tests.mini_nav_results.cohort_cassini import LIMB_STUB, RINGS_STUB, CohortCassiniISSSaturn

from spindoctor.cli.pds4.global_index import generate_global_index_files
from spindoctor.config import DEFAULT_CONFIG, MAIN_LOGGER

from .conftest import (
    index_entry,
    make_bundle_env,
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
        location: The byte its field begins at in a record, counted from 1.
        length: Its field's length in bytes.
        missing_constant: The missing constant it declares, or an empty string.
    """

    name: str
    location: int
    length: int
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
            location=int(_text(field, 'pds:field_location')),
            length=int(_text(field, 'pds:field_length')),
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

    The fields a label states, in order, name the columns the header line names.  Laid
    end to end with a comma between them, the fields' bytes rebuild every record exactly,
    so each field begins where its column's value begins and ends where it ends, and
    every record is the ``record_length`` the label states.
    """
    products = _cohort_index_products(cassini_cohort, tmp_path)
    described = {name: _field_characters(product.label) for name, product in products.items()}
    names = {name: [field.name for field in fields] for name, fields in described.items()}
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
    assert rebuilt == {name: product.records for name, product in products.items()}
    assert lengths == record_lengths


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


def _bodies_index_over(
    tmp_path: Path, templates: dict[str, str], bodies: list[dict[str, Any]]
) -> _IndexProduct:
    """Write one image's bodies table and its label, under the given body planes.

    The image's one body has a latitude statistic and no other, so any other plane the
    configuration declares has no statistic for it.

    Parameters:
        tmp_path: The directory the bundle environment is built in.
        templates: The index label templates to render, by name.
        bodies: The ``backplanes.bodies`` entries the configuration declares.

    Returns:
        The bodies table and its label.
    """
    env = make_bundle_env(tmp_path, template_contents=templates, bodies=bodies)
    touch_label(env.bundle_dir / 'data', 'shard0/1234567890w')
    write_supplemental(env.bundle_dir / 'data', 'shard0/1234567890w', bodies=LATITUDE_ONLY)
    generate_global_index_files(
        FCPath(env.bundle_results_root), env.dataset.as_dataset(), MAIN_LOGGER
    )
    return _index_product(env.bundle_dir / 'miscellaneous', 'global_bodies_index')


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
    before = _bodies_index_over(tmp_path / 'before', templates, [latitude])
    after = _bodies_index_over(tmp_path / 'after', templates, [latitude, resolution])
    added = ['minimum_resolution', 'maximum_resolution']
    fields_before = [field.name for field in _field_characters(before.label)]
    fields_after = [field.name for field in _field_characters(after.label)]
    assert after.header_names == before.header_names + added
    assert fields_after == fields_before + added


def test_a_missing_statistic_s_cell_holds_the_constant_its_field_declares(
    cassini_cohort: CohortCassiniISSSaturn, tmp_path: Path
) -> None:
    """A cell whose plane has no statistic holds its field's missing constant, as declared.

    The body has no resolution statistic, so both of its resolution cells hold the
    configured masked value.  Each of those fields declares that value as its missing
    constant in the spelling its cells have, which is the text the NASA PDS ``validate``
    tool compares a field's trimmed value to.
    """
    product = _bodies_index_over(
        tmp_path,
        _shipped_index_templates(cassini_cohort),
        [index_entry('latitude', 'rad'), index_entry('resolution', 'km/pixel')],
    )
    fields = {field.name: field for field in _field_characters(product.label)}
    missing = [fields['minimum_resolution'], fields['maximum_resolution']]
    cells = [field.value(product.records[0]).strip() for field in missing]
    declared = [field.missing_constant for field in missing]
    masked_value = float(DEFAULT_CONFIG.backplanes.masked_value)
    assert cells == declared
    assert [float(constant) for constant in declared] == [masked_value, masked_value]
