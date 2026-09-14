"""The global index tables of a PDS4 bundle, and the miscellaneous collection that holds them.

The summary pass reads every supplemental file the labels pass wrote once, here, and
builds two tables from them: one row for each body with geometry in each image the
data collection holds, and one row for each such image with ring backplanes, each giving
the image's exposure start and stop and the minimum and maximum every configured plane
spans.  The
same read takes the range of the products' epochs, which the data collection label and
the bundle label state, and the targets the products name, which the data collection,
bundle and metakernel labels name and the context inventory lists.

Each table is fixed width, as the reference bundle's index tables are: a header line
naming the columns, separated by commas, and then the rows, each field padded to the
longest value written in its column, with a comma between fields.  Where an image has
no statistic for a plane, its two cells hold the configured masked value,
``backplanes.masked_value``, written in the column's format.

Each table's label describes it as the reference bundle's index labels do: a ``Header``
over the header line, then a ``Table_Character`` over the records with one
``Field_Character`` per column, at the location and length the table was laid out with.
The columns a configured plane gives both are built from its configuration entry, which
names each column, its data type and its description; its unit is the unit the plane's
statistic is in, and its missing constant the masked value as the column writes it.
"""

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Self

import pdstemplate
from filecache import FCPath
from pdslogger import PdsLogger

from spindoctor.cli.backplanes.statistics import statistics_units
from spindoctor.cli.pds4.bundle_products import clear_bundle_products, secondary_members
from spindoctor.cli.pds4.collections import (
    clear_collection_products,
    data_directory,
    data_products,
    supplemental_files,
    write_collection,
)
from spindoctor.cli.pds4.epochs import EpochRange, EpochRangeScan, exposure_times
from spindoctor.cli.pds4.labels import write_label
from spindoctor.cli.pds4.statistic_checks import unindexable_statistic
from spindoctor.cli.pds4.targets import Pds4Target, TargetScan, has_geometry, target_table
from spindoctor.dataset.dataset import DataSet, pds4_label_name


@dataclass(frozen=True)
class IndexValueFormat:
    """How a minimum or maximum is written into a global index table.

    Exactly one of the two fields is set.  ``decimals`` writes every value to
    that many decimals, so three decimals write ``1.235`` and ``-89.999``.
    ``significant`` writes every value to that many significant figures,
    positionally: the number of decimals is chosen per value from its
    magnitude, so five figures write ``0.00060000``, ``6.1343``, ``4200.0``,
    ``70853`` and ``123456``.  A value is never written with an exponent or a
    trailing point, and its integer part is never rounded away, so a value with
    more integer digits than figures is written to all of them.  Neither field
    fixes a column's width: values under one format differ in length, so the
    width of a column is the widest value written in it, known only once the
    column is.

    Attributes:
        decimals: The number of decimals every value is written to, or None
            when ``significant`` is set.
        significant: The number of significant figures every value is written
            to, or None when ``decimals`` is set.
    """

    decimals: int | None = None
    significant: int | None = None

    def __post_init__(self) -> None:
        """Refuse a format that sets both fields or neither.

        Raises:
            ValueError: If ``decimals`` and ``significant`` are both set or both
                None.
        """
        if (self.decimals is None) == (self.significant is None):
            raise ValueError(
                'An index value format sets exactly one of decimals and significant; '
                f'got decimals={self.decimals!r}, significant={self.significant!r}'
            )

    def render(self, value: float) -> str:
        """Write one value the way this format says.

        Parameters:
            value: The statistic to write.

        Returns:
            The value as a plain decimal number to a fixed number of decimals,
            with no exponent and no trailing point.

        Raises:
            ValueError: If ``value`` is NaN or infinite, which has no decimal
                form.  The message names the value.
        """
        if not math.isfinite(value):
            raise ValueError(f'An index value has to be a finite number; got {value!r}')
        if self.significant is None:
            return format(value, f'.{self.decimals}f')
        magnitude = 0 if value == 0 else math.floor(math.log10(abs(value)))
        decimals = max(0, self.significant - 1 - magnitude)
        return format(value, f'.{decimals}f')


INDEX_VALUE_FORMATS: dict[str, IndexValueFormat] = {
    'deg': IndexValueFormat(decimals=3),
    'km': IndexValueFormat(decimals=1),
    'deg/pixel': IndexValueFormat(decimals=8),
    'km/pixel': IndexValueFormat(significant=5),
}
"""The format each min and max in the global index tables is written in, by unit.

The key is the unit the statistic is in: for a plane in radians, the degrees unit
:func:`~spindoctor.cli.backplanes.statistics.statistics_units` gives it.  Each format
prints about what one pixel resolves, within the roughly seven significant digits a
float32 backplane array carries.  ``km/pixel`` values span orders of magnitude, so
that format is five significant figures rather than a fixed number of decimals.  No
value is written with an exponent.  The bodies and rings tables share the mapping.
"""


def index_value_format(units: str) -> IndexValueFormat:
    """Return the format a statistic of a plane in these units is written in.

    Parameters:
        units: The unit the plane's values carry, as the configuration declares
            it.  An angular plane's statistic is in degrees whatever the plane
            is in, so what is looked up is the statistic's unit rather than
            this one.

    Returns:
        The format, from :data:`INDEX_VALUE_FORMATS`.

    Raises:
        KeyError: If the statistic's unit has no format in the table.
    """
    return INDEX_VALUE_FORMATS[statistics_units(units)]


MISCELLANEOUS_COLLECTION = 'miscellaneous'
"""The collection the global index tables are in, and the bundle directory it has."""

BODIES_INDEX = 'global_bodies_index'
"""The bodies table's name: the stem of its file and its label, and its LID's last part."""

RINGS_INDEX = 'global_rings_index'
"""The rings table's name: the stem of its file and its label, and its LID's last part."""

INDEX_VERSION = '1.0'
"""The version of each index product, which its label states and its inventory line names."""

_NO_INDEX_PRODUCT = (
    'neither index table was written with its label, so the collection holds no product of its own'
)
"""Why the miscellaneous collection is not written when neither index product is labeled."""


def index_lid(bundle_name: str, index_name: str) -> str:
    """Return the LID of one global index product, built from the bundle's name.

    Parameters:
        bundle_name: The bundle's name, the last part of its own LID.
        index_name: The table's name, :data:`BODIES_INDEX` or :data:`RINGS_INDEX`.

    Returns:
        ``urn:nasa:pds:<bundle_name>:miscellaneous:<index_name>``.
    """
    return f'urn:nasa:pds:{bundle_name}:{MISCELLANEOUS_COLLECTION}:{index_name}'


@dataclass(frozen=True)
class IndexColumn:
    """One column of a global index table: how its values are written, and its label.

    Attributes:
        name: The field's name, as the header line and the label give it: a PDS4
            dictionary attribute's name with the dictionary's prefix, as in
            ``pds:logical_identifier``, where the column holds that attribute, and a
            name of its own otherwise.
        data_type: The PDS4 data type of its values, as in ``ASCII_Real``.
        description: What its values are.
        unit: The unit of its values, or None for a column that has none.
        value_format: The format each statistic in the column is written in, or None
            for a column of text.  A statistic is right-justified in its field, and
            text left-justified.
        missing_constant: What a cell of the column holds where an image has no
            statistic for its plane, spelled as the cell spells it, or None for a column
            no cell of which is ever missing.
    """

    name: str
    data_type: str
    description: str
    unit: str | None = None
    value_format: IndexValueFormat | None = None
    missing_constant: str | None = None


_LID_COLUMN = IndexColumn(
    name='pds:logical_identifier',
    data_type='ASCII_LID',
    description='The logical identifier of the data product whose statistics the row gives.',
)
"""The first column of both tables: the data product each row is about."""

_BODY_COLUMN = IndexColumn(
    name='body_name',
    data_type='ASCII_String',
    description='The body whose statistics the row gives, by the name the backplanes use.',
)
"""The bodies table's second column: the body each row is about."""

_FILE_COLUMN = IndexColumn(
    name='file_spec',
    data_type='ASCII_String',
    description="The path of the data product's label, relative to the bundle's directory.",
)
"""The column naming each row's data label, under the reference bundle's name for it."""

_START_COLUMN = IndexColumn(
    name='pds:start_date_time',
    data_type='ASCII_Date_Time_YMD_UTC',
    description=(
        "When the image's exposure began, in UTC to the millisecond, as the data product's "
        'label states it.'
    ),
)
"""The column giving each row's exposure start, under the PDS4 attribute's name."""

_STOP_COLUMN = IndexColumn(
    name='pds:stop_date_time',
    data_type='ASCII_Date_Time_YMD_UTC',
    description=(
        "When the image's exposure ended, in UTC to the millisecond, as the data product's "
        'label states it.'
    ),
)
"""The column giving each row's exposure stop, under the PDS4 attribute's name."""


@dataclass(frozen=True)
class _IndexPlane:
    """One configured plane, and the two columns of an index table its statistic fills.

    Attributes:
        name: The plane's name, under which a supplemental file records its statistic.
        value_format: The format its statistic is written in.
        missing: What each of its two cells holds where an image has no statistic for
            it: the masked value, written in ``value_format``.
        minimum: The column of its least value.
        maximum: The column of its greatest value.
    """

    name: str
    value_format: IndexValueFormat
    missing: str
    minimum: IndexColumn
    maximum: IndexColumn

    @classmethod
    def from_entry(cls, entry: Mapping[str, Any], *, masked_value: float) -> Self:
        """Return a configured plane and its two columns, as its entry describes them.

        Each column takes its name and its description from the entry's ``index``
        block, and the data type the block gives both.  Its unit is the unit the plane's
        statistic is in, the entry's ``units`` restated through
        :func:`~spindoctor.cli.backplanes.statistics.statistics_units`, and its format
        the one :data:`INDEX_VALUE_FORMATS` gives that unit, so that a label states the
        unit and the format its column's values are written in.  Its missing constant
        is the masked value written in that format.

        Parameters:
            entry: The plane's configuration entry, from ``backplanes.bodies`` or
                ``backplanes.rings``.
            masked_value: The configured masked value, ``backplanes.masked_value``.

        Returns:
            The plane.

        Raises:
            KeyError: If the plane's statistic is in a unit the index has no format for.
        """
        value_format = index_value_format(entry['units'])
        unit = statistics_units(entry['units'])
        # The value every masked pixel of the arrays holds, in the column's own format,
        # so every value of a column, a missing one included, is written in one form
        # and the label declares it in that form (#601).
        missing = value_format.render(masked_value)
        index = entry['index']

        def column(end: str) -> IndexColumn:
            """Return the column of the plane's least or greatest value.

            Parameters:
                end: ``minimum`` or ``maximum``, the key of the column in the entry's
                    ``index`` block.

            Returns:
                The column.
            """
            return IndexColumn(
                name=index[end]['name'],
                data_type=index['data_type'],
                description=index[end]['description'],
                unit=unit,
                value_format=value_format,
                missing_constant=missing,
            )

        return cls(
            name=entry['name'],
            value_format=value_format,
            missing=missing,
            minimum=column('minimum'),
            maximum=column('maximum'),
        )

    def cells(self, statistic: Mapping[str, Any] | None) -> list[str]:
        """Write the plane's statistic as the two cells an index row gives it.

        Parameters:
            statistic: The plane's statistic as a supplemental file records it, its
                minimum and maximum already checked as finite numbers, or None when
                the file records none for the plane.

        Returns:
            The minimum and the maximum, rendered, or :attr:`missing` twice when there
            is no statistic, which is what a plane that measured nothing leaves.

        Raises:
            ValueError: If the minimum or the maximum is not a finite number.
        """
        if statistic is None:
            return [self.missing, self.missing]
        return [
            self.value_format.render(statistic['min']),
            self.value_format.render(statistic['max']),
        ]


def _statistic_columns(planes: Sequence[_IndexPlane]) -> list[IndexColumn]:
    """Return the columns the configured planes give an index table, in order.

    Parameters:
        planes: The configured planes, in the configuration's order.

    Returns:
        Each plane's minimum column and then its maximum column.
    """
    return [column for plane in planes for column in (plane.minimum, plane.maximum)]


@dataclass(frozen=True)
class IndexField:
    """Where one column lies in every record of a fixed-width global index table.

    Attributes:
        column: The column the field holds.
        number: The field's position in a record, counted from 1.
        location: The byte a record's field begins at, counted from 1.
        length: The field's length in bytes, which is the length of the longest value
            written in the column.
    """

    column: IndexColumn
    number: int
    location: int
    length: int


@dataclass(frozen=True)
class IndexTable:
    """A global index table laid out fixed width, as it is written and described.

    Attributes:
        header: The header line: the columns' names, in order, separated by commas,
            ending in a line feed.
        records: The records, one per row: each field padded to its length, the fields
            separated by commas, ending in a line feed.
        fields: Where each column lies in a record, in the columns' order.
    """

    header: str
    records: tuple[str, ...]
    fields: tuple[IndexField, ...]

    @property
    def record_length(self) -> int:
        """The length of every record in bytes, its commas and line feed included."""
        return sum(field.length for field in self.fields) + len(self.fields)


def lay_out_table(columns: Sequence[IndexColumn], rows: Sequence[Sequence[str]]) -> IndexTable:
    """Lay a global index table out fixed width, each field as long as its longest value.

    Values under one format differ in length -- ``1.000`` and ``-12.500`` are both three
    decimals -- so a field's length is the length of the longest value written in its
    column rather than a length its format implies, and is known only once the column
    is.  Each field is padded to that length with spaces, a statistic on the left and
    text on the right, and the fields are separated by commas.  The header line names
    the columns, unpadded.

    Parameters:
        columns: The table's columns, in order.
        rows: The table's rows, each holding one rendered cell per column.  There is at
            least one, since a table with no row is not written.

    Returns:
        The table, its header line, its records and where each field lies in them.
    """
    lengths = [max(len(row[number]) for row in rows) for number in range(len(columns))]
    fields: list[IndexField] = []
    location = 1
    for number, (column, length) in enumerate(zip(columns, lengths, strict=True), start=1):
        fields.append(IndexField(column=column, number=number, location=location, length=length))
        # A comma follows every field but the last, which the line feed follows.
        location += length + 1
    records = tuple(
        ','.join(
            cell.rjust(field.length)
            if field.column.value_format is not None
            else cell.ljust(field.length)
            for cell, field in zip(row, fields, strict=True)
        )
        + '\n'
        for row in rows
    )
    header = ','.join(column.name for column in columns) + '\n'
    return IndexTable(header=header, records=records, fields=tuple(fields))


class IndexWritten(Enum):
    """What writing one global index product came to.

    Attributes:
        LABELED: The table and its label are both on disk.
        UNLABELED: The table is on disk, and its label failed to render.
        OMITTED: Neither is on disk, because no image gives the table a row.
    """

    LABELED = 'labeled'
    UNLABELED = 'unlabeled'
    OMITTED = 'omitted'


def _write_index(
    table: FCPath,
    label: FCPath,
    columns: list[IndexColumn],
    rows: list[list[str]],
    *,
    lid: str,
    template: pdstemplate.PdsTemplate,
    logger: PdsLogger,
) -> IndexWritten:
    """Write one index table and then its label, or neither when the table has no row.

    A table's label states the table's records, and PDS4 requires at least one
    (``records`` has a minimum of 1 in ``PDS4_PDS_1O00.xsd``), so a table with no row
    cannot be described: neither it nor its label is written.  That is no failure, since
    a bundle can hold no image with ring backplanes, and the log says so at info level.
    Otherwise the table is laid out by :func:`lay_out_table` and written as ASCII, and
    then its label is rendered from ``template``, handed:

    - ``INDEX_LID``, the product's LID;
    - ``INDEX_TABLE_PATH``, the table's path, from which the label reads its size,
      checksum, time and number of lines;
    - ``HEADER_LENGTH``, the header line's length in bytes, its line feed included,
      which is where the records begin;
    - ``RECORD_LENGTH``, every record's length in bytes, its line feed included;
    - ``FIELDS``, one :class:`IndexField` per column, in order, laid out from the same
      columns the table was written from, so that the label cannot describe a table
      other than the one beside it.

    The table stays whether or not the label renders.

    Parameters:
        table: Where the table goes.
        label: Where its label goes.
        columns: The table's columns, in order.
        rows: The table's rows, each holding one rendered cell per column.
        lid: The product's LID.
        template: The parsed template the label renders from.
        logger: Logger for diagnostic messages.

    Returns:
        What came of it: both on disk, the table alone, or neither.
    """
    if len(rows) == 0:
        logger.info(
            'The index table %s was not written, nor its label: no image the data '
            'collection holds gives it a row, and a table label has to state at least '
            'one record',
            table,
        )
        return IndexWritten.OMITTED
    laid_out = lay_out_table(columns, rows)
    with table.open('w', encoding='ascii', newline='') as f:
        f.write(laid_out.header)
        f.writelines(laid_out.records)
    logger.info('Generated "%s" with %d rows', table.name, len(rows))
    template_vars = {
        'INDEX_LID': lid,
        'INDEX_TABLE_PATH': table.as_posix(),
        'HEADER_LENGTH': len(laid_out.header),
        'RECORD_LENGTH': laid_out.record_length,
        'FIELDS': laid_out.fields,
    }
    if not write_label(template, template_vars, label, logger=logger):
        return IndexWritten.UNLABELED
    logger.info('Generated "%s"', label.name)
    return IndexWritten.LABELED


@dataclass(frozen=True)
class GlobalIndexOutcome:
    """What generating the global index files came to.

    Attributes:
        failed_labels: The number of labels not written: each index label that could not
            be rendered, and the miscellaneous collection's label when it could not be
            rendered or the collection could not be written.
        epochs: The earliest exposure start and the latest exposure stop over the
            images the data collection holds, the ones the tables' rows are of, as
            their supplemental files record them, which the data collection label
            states, or None when no such image has a supplemental file.
        targets: Every target the backplane metadata of those same images names, each
            once, in the targets table's order, which the data collection, bundle and
            metakernel labels name and the context inventory lists; none when no such
            image has a supplemental file.
    """

    failed_labels: int
    epochs: EpochRange | None
    targets: tuple[Pds4Target, ...]


def generate_global_index_files(
    bundle_results_root: str | Path | FCPath,
    dataset: DataSet,
    logger: PdsLogger,
) -> GlobalIndexOutcome:
    """Generate global index files for bodies and rings.

    Both index labels are attempted, whichever of them fail, and an index table is
    written whether or not the label that describes it renders.  A table that no image
    gives a row is not written, nor its label: a table's label states its records, and
    PDS4 requires at least one.  That is not a failure, since a bundle can hold no image
    with ring backplanes; the log says so at info level.  With neither table written,
    though, the miscellaneous collection holds nothing of its own, and that counts
    (below).

    The tables index exactly the images the data inventory lists, the data labels in
    the data tree that :func:`~spindoctor.cli.pds4.collections.data_products` names,
    each with the rows its supplemental file gives: a row in the bodies table for each
    body its backplane document names that has geometry, a statistic at least, as
    :func:`~spindoctor.cli.pds4.targets.has_geometry` decides, and a row in the rings
    table when it has ring statistics.  A supplemental file with no data
    label beside it adds no row, so the
    tables and the inventory cannot disagree about what the bundle holds, and its
    epochs are not taken into the range; its statistics are still checked, as every
    supplemental file's are.

    Each row gives, after its data product's LID and, in the bodies table, the body, the
    path of the data label and the start and stop of the image's exposure, which
    :func:`~spindoctor.cli.pds4.epochs.exposure_times` writes from the epochs the
    supplemental file records, to the millisecond, as the data label states them.

    Each label describes the table beside it: a ``Header`` over the header line, and a
    ``Table_Character`` over the records, with one ``Field_Character`` per column at the
    location and length the table was laid out with.  A configured plane gives each
    table two columns, its least and its greatest value, whose names, data type and
    descriptions come from the plane's configuration entry, whose unit is the unit its
    statistic is in, and whose missing constant is the masked value in the column's
    format, the text a cell holds where an image has no statistic for the plane.

    The miscellaneous collection is written after the tables, its inventory
    ``collection_miscellaneous.csv`` and its label beside them: a ``P`` line for each
    index product whose label is on disk, by its LID and :data:`INDEX_VERSION`, and then
    an ``S`` line for each secondary member the template directory's document inventory
    cites, through :func:`~spindoctor.cli.pds4.bundle_products.secondary_members`.  It
    takes its members from the index labels, as the data collection takes its members
    from the data labels, so with neither index product labeled -- no image gives either
    table a row, or neither label renders -- it is not written at all, whatever it would
    cite, and counts once as a label not written; the bundle label, which declares it,
    is then not written either.  It is written through
    :func:`~spindoctor.cli.pds4.collections.write_collection`.

    Its read of the supplemental files is the one the summary pass makes, so the
    range of the products' epochs is taken in the same read, through an
    :class:`~spindoctor.cli.pds4.epochs.EpochRangeScan`, over the same images as the
    rows, and returned for the labels that state it; and so are the targets those images'
    backplane metadata names, through a :class:`~spindoctor.cli.pds4.targets.TargetScan`
    over the configuration's targets table, for the labels that name them and the context
    inventory that lists them.

    Both index tables and both index labels, and the miscellaneous collection's
    inventory and label, are cleared before any supplemental file is read, as
    :func:`~spindoctor.cli.pds4.labels.write_label` clears a label before it renders,
    and so are the data and browse collections' inventories and labels, which
    :func:`~spindoctor.cli.pds4.collections.generate_collection_files` writes after the
    index, and every run-level product
    :func:`~spindoctor.cli.pds4.bundle_products.generate_bundle_products` writes last.
    A run refused over what a supplemental file holds therefore leaves no product of
    the summary pass, neither this run's nor an earlier run's: no index still
    describing the bundle as it was, no inventory beside no index, and no bundle label
    declaring collections that are not there.

    The three templates the dataset declares for these products, both index labels'
    and ``collection_miscellaneous.lblx``, are required.  The caller is expected to
    have checked them before processing anything, so one that is missing here raises
    rather than being passed over.

    Parameters:
        bundle_results_root: Root directory of the bundle, a local path or a URL.  The
            bundle's data directory is scanned for its supplemental files and its data
            labels.
        dataset: The dataset instance for bundle-specific methods.
        logger: Logger for diagnostic messages.

    Returns:
        The number of labels not written -- each index label that could not be
        rendered, and the miscellaneous collection's label when the collection could
        not be written or its label could not be rendered -- the range of the
        epochs of the images the data collection holds, or None when none of them has
        a supplemental file, and the targets those images name.

    Raises:
        FileNotFoundError: If the bundle has no data directory to scan, which is
            checked before any product of the pass is cleared or written; if an index
            template or ``collection_miscellaneous.lblx`` is not in the dataset's
            template directory; or if the template directory holds no document
            inventory to take the miscellaneous collection's secondary members from.
        KeyError: If a configured plane's statistic is in a unit the index has no
            column format for, or if the targets table has no entry for a target the
            backplane metadata of an image the data collection holds names; the message
            names the target.  Both are raised before either table is opened.
        ValueError: If a supplemental file holds a statistic no column can: one in
            a unit other than the one the configuration gives its plane, or with a
            minimum or maximum that is NaN or infinite.  The message names the
            file and the plane, what the file records there, and what to
            regenerate.  Every supplemental file is read, and every value in both
            tables rendered, before either table is opened, so none of these
            leaves a table half-written.
    """

    bundle_results_root = FCPath(bundle_results_root)
    bundle_name = dataset.pds4_bundle_name()
    template_dir = FCPath(dataset.pds4_bundle_template_dir())
    bundle_root = bundle_results_root / bundle_name
    config = dataset.config

    # Each configured plane and the two columns its statistic fills, from its
    # configuration entry: the cells are written by these and the labels describe these,
    # so a change to the configuration moves a table and its label together.
    masked_value = float(config.backplanes.masked_value)
    body_planes = [
        _IndexPlane.from_entry(entry, masked_value=masked_value)
        for entry in config.backplanes.bodies
    ]
    ring_planes = [
        _IndexPlane.from_entry(entry, masked_value=masked_value)
        for entry in config.backplanes.rings
    ]

    # A bundle with no data directory is not one a labels pass wrote.  The summary
    # pass runs this generator first, so the check is made here, before any product
    # of the pass is cleared or written into a bundle that is not there.
    data_dir = data_directory(bundle_root)

    # Cleared before any supplemental file is read, by the rule write_label keeps
    # for a label that what is on disk is what this run wrote: the index's own
    # products, and the collection files and run-level products the pass writes after
    # it, so a refusal over one cannot leave an earlier run's index describing the
    # bundle as it was, nor an earlier run's inventory and its label beside no index.
    miscellaneous_dir = bundle_root / MISCELLANEOUS_COLLECTION
    bodies_tab = miscellaneous_dir / f'{BODIES_INDEX}.tab'
    bodies_label = miscellaneous_dir / pds4_label_name(bodies_tab.name)
    rings_tab = miscellaneous_dir / f'{RINGS_INDEX}.tab'
    rings_label = miscellaneous_dir / pds4_label_name(rings_tab.name)
    collection_inventory = miscellaneous_dir / f'collection_{MISCELLANEOUS_COLLECTION}.csv'
    collection_label = miscellaneous_dir / pds4_label_name(collection_inventory.name)
    index_products = (bodies_tab, bodies_label, rings_tab, rings_label)
    for index_product in (*index_products, collection_inventory, collection_label):
        index_product.unlink(missing_ok=True)
    clear_collection_products(bundle_root)
    clear_bundle_products(bundle_root, dataset)

    # Every supplemental file, in the order of the products' names, the last part of
    # each LID
    supplementals = supplemental_files(data_dir)
    members = data_products(data_dir)
    logger.info('Found %d supplemental files and %d data labels', len(supplementals), len(members))

    # Collect body and ring statistics, every cell already rendered: both
    # tables are opened only once every value in them has been written out, so
    # nothing a render can raise leaves a table half-written.
    body_index_rows: list[list[str]] = []
    ring_index_rows: list[list[str]] = []
    # The range of the epochs of the images the data collection holds, the images the
    # rows are of, and the targets those images name, taken in this same read of their
    # files.
    epochs = EpochRangeScan()
    targets = TargetScan(target_table(config))

    for pds4_path_stub, suppl_file in supplementals.items():
        metadata = json.loads(suppl_file.read_text())
        backplanes = metadata.get('backplanes', {})
        # A supplemental file holds the backplane document the labels pass
        # read.  Every index column is in its plane's configured unit and holds
        # only finite numbers, so a file with a statistic the index cannot hold
        # refuses the run here, before either table exists.
        unindexable = unindexable_statistic(backplanes, config)
        if unindexable is not None:
            raise ValueError(
                f'Supplemental file {suppl_file} {unindexable.description}; '
                f'{unindexable.reason}. The file carries a copy of the backplane document a '
                'labels pass read: regenerate the backplanes, then the bundle into an empty '
                'directory, where the labels pass fails any image whose document still '
                'records such a statistic'
            )
        # The rows are the data inventory's members, so the tables and the inventory
        # cannot disagree about what the bundle holds: a supplemental file with no data
        # label beside it, which the collection generator reports as an image whose
        # products disagree, adds none (#602), and nor does it widen the range of the
        # epochs the data collection label states for the images the collection holds.
        if pds4_path_stub not in members:
            continue
        epochs.include(metadata['navigation'])
        targets.include(backplanes)
        start, stop = exposure_times(metadata['navigation'])
        bodies = backplanes.get('bodies', {})
        rings = backplanes.get('rings', {})

        # The stub's last part is the product's name, the last part of its LID
        image_name = dataset.pds4_lid_part_to_image_name(pds4_path_stub.rsplit('/', 1)[-1])
        lid = dataset.pds4_image_name_to_data_lid(image_name)
        # The data label, by its path relative to the bundle's own directory
        path_to_image = members[pds4_path_stub].relative_to(bundle_root).as_posix()

        # Body index: one line per image per body with geometry; a body the image's
        # inventory found that shows at no pixel has no statistic, and no row
        for body_name, body_data in bodies.items():
            if not has_geometry(body_data):
                continue
            body_backplanes = body_data.get('backplanes', {})
            body_row: list[str] = [lid, body_name, path_to_image, start, stop]
            for plane in body_planes:
                body_row.extend(plane.cells(body_backplanes.get(plane.name)))
            body_index_rows.append(body_row)

        # Ring index: one line per image
        ring_backplanes = rings.get('backplanes', {})
        if ring_backplanes:
            ring_row: list[str] = [lid, path_to_image, start, stop]
            for plane in ring_planes:
                ring_row.extend(plane.cells(ring_backplanes.get(plane.name)))
            ring_index_rows.append(ring_row)

    # Each label renders from the template of its own name.  Both are parsed before
    # either table is written, whether or not the table has a row, so a template
    # missing from the tree raises rather than being passed over.
    bodies_template = pdstemplate.PdsTemplate((template_dir / bodies_label.name).as_posix())
    rings_template = pdstemplate.PdsTemplate((template_dir / rings_label.name).as_posix())
    collection_template = pdstemplate.PdsTemplate((template_dir / collection_label.name).as_posix())

    # The bodies table: the data product, the body, the data label and the exposure's
    # start and stop, then the least and the greatest value of each configured body plane
    bodies_written = _write_index(
        bodies_tab,
        bodies_label,
        [
            _LID_COLUMN,
            _BODY_COLUMN,
            _FILE_COLUMN,
            _START_COLUMN,
            _STOP_COLUMN,
            *_statistic_columns(body_planes),
        ],
        body_index_rows,
        lid=index_lid(bundle_name, BODIES_INDEX),
        template=bodies_template,
        logger=logger,
    )

    # The rings table: the data product, the data label and the exposure's start and
    # stop, then the least and the greatest value of each configured ring plane
    # TODO Add planet name to rings table
    rings_written = _write_index(
        rings_tab,
        rings_label,
        [
            _LID_COLUMN,
            _FILE_COLUMN,
            _START_COLUMN,
            _STOP_COLUMN,
            *_statistic_columns(ring_planes),
        ],
        ring_index_rows,
        lid=index_lid(bundle_name, RINGS_INDEX),
        template=rings_template,
        logger=logger,
    )
    failed_labels = [bodies_written, rings_written].count(IndexWritten.UNLABELED)

    # The miscellaneous collection, written after the tables it lists, as the data
    # inventory is written after the labels it lists.  Its primary members are the index
    # products whose labels are on disk; its secondary members are the ones the document
    # inventory the template directory ships cites, taken from there so that the two
    # inventories cannot disagree about them.  A collection takes its members from the
    # labels of its own kind, so with neither index product labeled it holds nothing of
    # its own and is not written, whatever it would cite: it counts, and the bundle
    # label, which declares it, goes with it.
    primaries = [
        f'{index_lid(bundle_name, name)}::{INDEX_VERSION}'
        for name, written in ((BODIES_INDEX, bodies_written), (RINGS_INDEX, rings_written))
        if written is IndexWritten.LABELED
    ]
    if not write_collection(
        MISCELLANEOUS_COLLECTION,
        collection_inventory,
        collection_label,
        primaries=primaries,
        secondaries=secondary_members(template_dir),
        template=collection_template,
        template_vars={'COLLECTION_MISCELLANEOUS_CSV_PATH': collection_inventory.as_posix()},
        reasons_not_written=[] if primaries else [_NO_INDEX_PRODUCT],
        logger=logger,
    ):
        failed_labels += 1

    logger.info(
        'Generated global index files: %d body rows, %d ring rows',
        len(body_index_rows),
        len(ring_index_rows),
    )
    return GlobalIndexOutcome(
        failed_labels=failed_labels, epochs=epochs.result(), targets=targets.result()
    )
