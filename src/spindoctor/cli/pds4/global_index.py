"""The global index tables of a PDS4 bundle, and the formats their values are written in.

The summary pass reads every supplemental file the labels pass wrote once, here, and
builds two tables from them: one row for each body of each image the data collection
holds, and one row for each such image with ring backplanes, each giving the minimum and
maximum every configured plane spans.  The same read takes the range of the products'
epochs, which the data collection label and the bundle label state.

Each table is fixed width, as the reference bundle's index tables are: a header line
naming the columns, separated by commas, and then the rows, each field padded to the
longest value written in its column, with a comma between fields.  Where an image has
no statistic for a plane, its two cells hold the configured masked value,
``backplanes.masked_value``, written in the column's format.
"""

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import pdstemplate
from filecache import FCPath
from pdslogger import PdsLogger

from spindoctor.cli.backplanes.statistics import statistics_units
from spindoctor.cli.pds4.bundle_products import clear_bundle_products
from spindoctor.cli.pds4.collections import (
    clear_collection_products,
    data_directory,
    data_products,
    supplemental_files,
)
from spindoctor.cli.pds4.epochs import EpochRange, EpochRangeScan
from spindoctor.cli.pds4.labels import write_label
from spindoctor.cli.pds4.statistic_checks import unindexable_statistic
from spindoctor.dataset.dataset import DataSet


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


def _index_cells(
    statistic: dict[str, Any] | None, value_format: IndexValueFormat, missing: str
) -> list[str]:
    """Write one plane's minimum and maximum as the two cells an index row gives it.

    Parameters:
        statistic: The plane's statistic as a supplemental file records it, its
            minimum and maximum already checked as finite numbers, or None when
            the file records none for the plane.
        value_format: The format the plane's column is written in.
        missing: What each of the two cells holds when there is no statistic: the
            masked value, written in ``value_format``.

    Returns:
        The minimum and the maximum, rendered, or ``missing`` twice when there is no
        statistic, which is what a plane that measured nothing leaves.

    Raises:
        ValueError: If the minimum or the maximum is not a finite number.
    """
    if statistic is None:
        return [missing, missing]
    return [value_format.render(statistic['min']), value_format.render(statistic['max'])]


MISCELLANEOUS_COLLECTION = 'miscellaneous'
"""The collection the global index tables are in, and the bundle directory it has."""

BODIES_INDEX = 'global_bodies_index'
"""The bodies table's name: the stem of its file and its label, and its LID's last part."""

RINGS_INDEX = 'global_rings_index'
"""The rings table's name: the stem of its file and its label, and its LID's last part."""


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
    """One column of a global index table.

    Attributes:
        name: The column's name, as the header line gives it.
        value_format: The format each statistic in the column is written in, or None
            for a column of text.  A statistic is right-justified in its field, and
            text left-justified.
    """

    name: str
    value_format: IndexValueFormat | None = None


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
    template: pdstemplate.PdsTemplate,
    template_vars: dict[str, Any],
    logger: PdsLogger,
) -> IndexWritten:
    """Write one index table and then its label, or neither when the table has no row.

    A table's label states the table's records, and PDS4 requires at least one
    (``records`` has a minimum of 1 in ``PDS4_PDS_1O00.xsd``), so a table with no row
    cannot be described: neither it nor its label is written.  That is no failure, since
    a bundle can hold no image with ring backplanes, and the log says so at info level.
    Otherwise the table is laid out by :func:`lay_out_table` and written as ASCII, and
    then its label is rendered, which reads the table's size, checksum and record count
    from the file; the table stays whether or not the label renders.

    Parameters:
        table: Where the table goes.
        label: Where its label goes.
        columns: The table's columns, in order.
        rows: The table's rows, each holding one rendered cell per column.
        template: The parsed template the label renders from.
        template_vars: The variables the label's template resolves against.
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
    if not write_label(template, template_vars, label, logger=logger):
        return IndexWritten.UNLABELED
    logger.info('Generated "%s"', label.name)
    return IndexWritten.LABELED


@dataclass(frozen=True)
class GlobalIndexOutcome:
    """What generating the global index files came to.

    Attributes:
        failed_labels: The number of index labels that could not be rendered.
        epochs: The earliest exposure start and the latest exposure stop over the
            supplemental files the index was built from, which the data collection
            label states, or None when there were none.
    """

    failed_labels: int
    epochs: EpochRange | None


def generate_global_index_files(
    bundle_results_root: FCPath,
    dataset: DataSet,
    logger: PdsLogger,
) -> GlobalIndexOutcome:
    """Generate global index files for bodies and rings.

    Both index labels are attempted, whichever of them fail, and an index table is
    written whether or not the label that describes it renders.  A table that no image
    gives a row is not written, nor its label: a table's label states its records, and
    PDS4 requires at least one.  That is not a failure, since a bundle can hold no image
    with ring backplanes; the log says so at info level.

    The tables index exactly the images the data inventory lists, the data labels in
    the data tree that :func:`~spindoctor.cli.pds4.collections.data_products` names,
    each with the rows its supplemental file gives: a row in the bodies table for each
    body it has statistics for, and a row in the rings table when it has ring
    statistics.  A supplemental file with no data label beside it adds no row, so the
    tables and the inventory cannot disagree about what the bundle holds.  Its
    statistics are still checked and its epochs still taken, as every supplemental
    file's are.

    Its read of the supplemental files is the one the summary pass makes, so the
    range of the products' epochs is taken in the same read, through an
    :class:`~spindoctor.cli.pds4.epochs.EpochRangeScan`, and returned for the labels
    that state it.

    Both index tables and both index labels are cleared before any supplemental
    file is read, as :func:`~spindoctor.cli.pds4.labels.write_label` clears a
    label before it renders, and so are the two collection inventories and two
    collection labels :func:`~spindoctor.cli.pds4.collections.generate_collection_files`
    writes after the index, and
    every run-level product
    :func:`~spindoctor.cli.pds4.bundle_products.generate_bundle_products` writes last.
    A run refused over what a supplemental file holds therefore leaves no product of
    the summary pass, neither this run's nor an earlier run's: no index still
    describing the bundle as it was, no inventory beside no index, and no bundle label
    declaring collections that are not there.

    Both index templates the dataset declares are required.  The caller is
    expected to have checked them before processing anything, so one that is
    missing here raises rather than being passed over.

    Parameters:
        bundle_results_root: Root directory of the bundle. The bundle data directory
            will be scanned for all supplemental text files.
        dataset: The dataset instance for bundle-specific methods.
        logger: Logger for diagnostic messages.

    Returns:
        The number of index labels that could not be rendered, and the range of the
        products' epochs over every supplemental file read, or None when there is no
        supplemental file.

    Raises:
        FileNotFoundError: If the bundle has no data directory to scan, which is
            checked before any product of the pass is cleared or written, or an
            index template is not in the dataset's template directory.
        KeyError: If a configured plane's statistic is in a unit the index has no
            column format for.
        ValueError: If a supplemental file holds a statistic no column can: one in
            a unit other than the one the configuration gives its plane, or with a
            minimum or maximum that is NaN or infinite.  The message names the
            file and the plane, what the file records there, and what to
            regenerate.  Every supplemental file is read, and every value in both
            tables rendered, before either table is opened, so none of these
            leaves a table half-written.
    """

    bundle_name = dataset.pds4_bundle_name()
    template_dir = dataset.pds4_bundle_template_dir()
    bundle_root = bundle_results_root / bundle_name
    config = dataset.config

    # Get configured backplane types from config
    bodies_cfg = config.backplanes.bodies
    body_backplane_types = [bp['name'] for bp in bodies_cfg]
    rings_cfg = config.backplanes.rings
    ring_backplane_types = [bp['name'] for bp in rings_cfg]
    body_formats = {bp['name']: index_value_format(bp['units']) for bp in bodies_cfg}
    ring_formats = {bp['name']: index_value_format(bp['units']) for bp in rings_cfg}
    # What a cell holds where an image has no statistic for the plane: the value every
    # masked pixel of the arrays holds, written in the column's own format, so every
    # value of a column, a missing one included, is written in one form (#601).
    masked_value = float(config.backplanes.masked_value)
    body_missing = {name: fmt.render(masked_value) for name, fmt in body_formats.items()}
    ring_missing = {name: fmt.render(masked_value) for name, fmt in ring_formats.items()}

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
    bodies_label = miscellaneous_dir / f'{BODIES_INDEX}.lblx'
    rings_tab = miscellaneous_dir / f'{RINGS_INDEX}.tab'
    rings_label = miscellaneous_dir / f'{RINGS_INDEX}.lblx'
    for index_product in (bodies_tab, bodies_label, rings_tab, rings_label):
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
    # The range of the products' epochs, taken in this same read of the files.
    epochs = EpochRangeScan()

    for pds4_path_stub, suppl_file in supplementals.items():
        metadata = json.loads(suppl_file.read_text())
        epochs.include(metadata['navigation'])
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
        # products disagree, adds none (#602).
        if pds4_path_stub not in members:
            continue
        bodies = backplanes.get('bodies', {})
        rings = backplanes.get('rings', {})

        # The stub's last part is the product's name, the last part of its LID
        image_name = dataset.pds4_lid_part_to_image_name(pds4_path_stub.rsplit('/', 1)[-1])
        lid = dataset.pds4_image_name_to_data_lid(image_name)
        # The data label, by its path relative to the bundle's own directory
        path_to_image = members[pds4_path_stub].relative_to(bundle_root).as_posix()

        # Body index: one line per image per body
        for body_name, body_data in bodies.items():
            body_row: list[str] = [lid, body_name, path_to_image]
            body_backplanes = body_data.get('backplanes', {})
            # Add min/max columns for each configured backplane type
            for bp_type in body_backplane_types:
                body_row.extend(
                    _index_cells(
                        body_backplanes.get(bp_type), body_formats[bp_type], body_missing[bp_type]
                    )
                )
            body_index_rows.append(body_row)

        # Ring index: one line per image
        ring_backplanes = rings.get('backplanes', {})
        if ring_backplanes:
            ring_row: list[str] = [lid, path_to_image]
            # Add min/max columns for each configured ring backplane type
            for ring_type in ring_backplane_types:
                ring_row.extend(
                    _index_cells(
                        ring_backplanes.get(ring_type),
                        ring_formats[ring_type],
                        ring_missing[ring_type],
                    )
                )
            ring_index_rows.append(ring_row)

    # Each label renders from the template of its own name.  Both are parsed before
    # either table is written, whether or not the table has a row, so a template
    # missing from the tree raises rather than being passed over.
    template_base = Path(template_dir)
    bodies_template = pdstemplate.PdsTemplate(str(template_base / bodies_label.name))
    rings_template = pdstemplate.PdsTemplate(str(template_base / rings_label.name))

    # The bodies table: LID, body_name, path_to_image_file, then min/max for each
    # backplane type
    bodies_columns = [
        IndexColumn('LID'),
        IndexColumn('body_name'),
        IndexColumn('path_to_image_file'),
    ]
    for bp_type in body_backplane_types:
        bodies_columns.append(IndexColumn(f'{bp_type}_min', body_formats[bp_type]))
        bodies_columns.append(IndexColumn(f'{bp_type}_max', body_formats[bp_type]))
    bodies_written = _write_index(
        bodies_tab,
        bodies_label,
        bodies_columns,
        body_index_rows,
        template=bodies_template,
        template_vars={
            'INDEX_LID': index_lid(bundle_name, BODIES_INDEX),
            'FILE_RECORDS': len(body_index_rows),
        },
        logger=logger,
    )

    # The rings table: LID, path_to_image_file, then min/max for each ring type
    rings_columns = [IndexColumn('LID'), IndexColumn('path_to_image_file')]
    # TODO Add planet name to rings table
    for ring_type in ring_backplane_types:
        rings_columns.append(IndexColumn(f'{ring_type}_min', ring_formats[ring_type]))
        rings_columns.append(IndexColumn(f'{ring_type}_max', ring_formats[ring_type]))
    rings_written = _write_index(
        rings_tab,
        rings_label,
        rings_columns,
        ring_index_rows,
        template=rings_template,
        template_vars={
            'INDEX_LID': index_lid(bundle_name, RINGS_INDEX),
            'FILE_RECORDS': len(ring_index_rows),
        },
        logger=logger,
    )
    failed_labels = [bodies_written, rings_written].count(IndexWritten.UNLABELED)

    logger.info(
        'Generated global index files: %d body rows, %d ring rows',
        len(body_index_rows),
        len(ring_index_rows),
    )
    return GlobalIndexOutcome(failed_labels=failed_labels, epochs=epochs.result())
