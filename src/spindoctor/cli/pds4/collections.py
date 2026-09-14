import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self, cast

import pdstemplate
from filecache import FCPath
from pdslogger import PdsLogger

from spindoctor.cli.backplanes.statistics import statistics_units
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


def _index_cells(statistic: dict[str, Any] | None, value_format: IndexValueFormat) -> list[str]:
    """Write one plane's minimum and maximum as the two cells an index row gives it.

    Parameters:
        statistic: The plane's statistic as a supplemental file records it, its
            minimum and maximum already checked as finite numbers, or None when
            the file records none for the plane.
        value_format: The format the plane's column is written in.

    Returns:
        The minimum and the maximum, rendered, or two blanks when there is no
        statistic, which is what a plane that measured nothing leaves.

    Raises:
        ValueError: If the minimum or the maximum is not a finite number.
    """
    if statistic is None:
        # TODO Need an appropriate sentinel value for missing data
        return ['', '']
    return [value_format.render(statistic['min']), value_format.render(statistic['max'])]


def _data_dir(bundle_root: FCPath) -> FCPath:
    """Return the bundle's data directory, which both summary generators scan.

    Parameters:
        bundle_root: The bundle's own directory.

    Returns:
        ``<bundle_root>/data``.

    Raises:
        FileNotFoundError: If the bundle has none, which makes it a bundle no labels
            pass wrote.  The message names the directory.
    """
    data_dir = bundle_root / 'data'
    if not data_dir.exists():
        raise FileNotFoundError(f'Data directory does not exist: {data_dir}')
    return data_dir


_SUPPLEMENTAL_SUFFIX = '_supplemental.txt'
"""What follows a product's path stub in the name of its supplemental file."""

_DATA_LABEL_SUFFIX = '_backplanes.lblx'
"""What follows a product's path stub in the name of its data label."""


def _product_stub(path: FCPath, data_dir: FCPath) -> str:
    """Return a product's path stub, read off its supplemental file.

    Parameters:
        path: The product's supplemental file, under ``data_dir``.
        data_dir: The bundle's data directory.

    Returns:
        The file's path relative to ``data_dir``, in POSIX form, less
        ``_supplemental.txt``, as in ``shard0/1234567890w``.
    """
    return path.relative_to(data_dir).as_posix().removesuffix(_SUPPLEMENTAL_SUFFIX)


@dataclass(frozen=True)
class _CollectionProducts:
    """Where in a bundle the collection inventories and labels the summary pass writes go.

    Attributes:
        data_inventory: The data collection's inventory, ``data/collection_data.csv``.
        data_label: Its label, ``data/collection_data.lblx``.
        browse_inventory: The browse collection's inventory,
            ``browse/collection_browse.csv``.
        browse_label: Its label, ``browse/collection_browse.lblx``.
    """

    data_inventory: FCPath
    data_label: FCPath
    browse_inventory: FCPath
    browse_label: FCPath

    @classmethod
    def in_bundle(cls, bundle_root: FCPath) -> Self:
        """Return where the four go in one bundle.

        Parameters:
            bundle_root: The bundle's own directory.

        Returns:
            The four paths, under the bundle's ``data`` and ``browse`` directories.
        """
        return cls(
            data_inventory=bundle_root / 'data' / 'collection_data.csv',
            data_label=bundle_root / 'data' / 'collection_data.lblx',
            browse_inventory=bundle_root / 'browse' / 'collection_browse.csv',
            browse_label=bundle_root / 'browse' / 'collection_browse.lblx',
        )

    def paths(self) -> tuple[FCPath, ...]:
        """Return all four paths.

        Returns:
            The data collection's inventory and label, then the browse collection's.
        """
        return (self.data_inventory, self.data_label, self.browse_inventory, self.browse_label)


def _write_inventory(inventory: FCPath, lidvids: list[str]) -> None:
    """Write a collection inventory listing each LIDVID as a primary member.

    The inventory is comma-separated with no header: one ``P,<lidvid>`` line per
    member, in the order given, each line ending in a line feed alone, the last
    included.  Its number of lines is therefore its number of members, which is
    what the collection label states as its records.

    Parameters:
        inventory: Where the inventory goes.
        lidvids: The members' LIDVIDs, in the order they are listed.
    """
    local_path = cast(Path, inventory.get_local_path())
    with local_path.open('w', newline='', encoding='utf-8') as f:
        csv.writer(f, lineterminator='\n').writerows(['P', lidvid] for lidvid in lidvids)
    inventory.upload()


_NO_MEMBER = (
    'the data tree holds no data label, so the collection has no member, and its label '
    'has to state at least one record'
)
"""Why a collection is not written when the data tree holds no data label.

Both generated collections take their members from the data labels, and the PDS4 schema
requires a collection inventory to hold at least one record.
"""

_NO_RANGE = (
    'the data tree holds no supplemental file, so there is no time range for its label to state'
)
"""Why the data collection is not written when the data tree holds no supplemental file."""


def _write_collection(
    name: str,
    inventory: FCPath,
    label: FCPath,
    *,
    lidvids: list[str],
    template: pdstemplate.PdsTemplate,
    template_vars: dict[str, Any],
    reasons_not_written: list[str],
    logger: PdsLogger,
) -> bool:
    """Write one collection's inventory and then its label, or neither of them.

    A collection with a reason not to be written is not written at all: whatever is at
    either path is removed, so that a collection on disk is always one this run wrote,
    and one error names the collection, both paths and every reason.  Otherwise the
    inventory is written, and then the label, which reads the inventory's size, checksum
    and record count from the file; the inventory stays whether or not the label
    renders.

    Parameters:
        name: The collection's name, as the error gives it: ``data`` or ``browse``.
        inventory: Where the collection's inventory goes.
        label: Where the collection's label goes.
        lidvids: The members' LIDVIDs, in the order the inventory lists them.
        template: The parsed template the label renders from.
        template_vars: The variables the label's template resolves against.
        reasons_not_written: Why the collection cannot be written, one phrase each, or
            an empty list when it can be.
        logger: Logger for diagnostic messages.

    Returns:
        True if the collection's label is on disk, False if it is not.
    """
    if len(reasons_not_written) > 0:
        inventory.unlink(missing_ok=True)
        label.unlink(missing_ok=True)
        logger.error(
            'The %s collection was not written, neither its inventory %s nor its label %s: %s',
            name,
            inventory,
            label,
            '; and '.join(reasons_not_written),
        )
        return False
    _write_inventory(inventory, lidvids)
    logger.info('Generated "%s": %s', inventory.name, inventory)
    if not write_label(template, template_vars, label, logger=logger):
        return False
    logger.info('Generated "%s"', label.name)
    return True


def generate_collection_files(
    bundle_results_root: FCPath,
    dataset: DataSet,
    logger: PdsLogger,
    *,
    epochs: EpochRange | None,
) -> int:
    """Generate the data and browse collection inventories and their labels.

    Each inventory lists one product per line, ``P,<lidvid>``, in the order of the
    images' names: ``data/collection_data.csv`` the data products and
    ``browse/collection_browse.csv`` the browse products, both found by the data
    labels in the data tree.  An inventory has no header, and every line, the last
    included, ends in a line feed alone, so the records its label counts are its
    products.

    The data collection label states the time range of the products the collection
    holds, which is ``epochs``: the range :func:`generate_global_index_files` takes in
    its read of the supplemental files, which is why the summary pass runs that
    first.  The label writes it to whole seconds, the start rounded down and the stop
    up.

    A collection is written only when its label can state what PDS4 requires of it:
    at least one record, and, for the data collection, the time range.  One that
    cannot is not written at all, neither its inventory nor its label, and counts once
    as a label not written, with one error naming the collection and every reason:
    both collections when the data tree holds no data label, since both take their
    members from the data labels, and the data collection when ``epochs`` is None.
    Whatever an earlier run left at either of its paths is removed, so a collection on
    disk is always one this run wrote.

    Every collection label that can be written is attempted, whichever of them fail: a
    broken data collection template must not hide a broken browse collection one.  An
    inventory is written before its label, which reads the inventory's size, checksum
    and record count, and stays whether or not the label renders.

    Every collection template the dataset declares is required.  The caller is
    expected to have checked them before processing anything, so one that is
    missing here raises rather than being passed over, whether or not its collection
    can be written.

    Parameters:
        bundle_results_root: Root directory of the bundle. The bundle data directory
            will be scanned for all backplane label files.
        dataset: The dataset instance for bundle-specific methods.
        logger: Logger for diagnostic messages.
        epochs: The earliest start and the latest stop of the products' exposures,
            or None when the data tree holds no supplemental file.

    Returns:
        The number of collection labels not written: each label that could not be
        rendered, and each collection that could not be written, counted once whatever
        its number of reasons.

    Raises:
        FileNotFoundError: If the bundle has no data directory to scan, or a
            collection template is not in the dataset's template directory.
    """

    bundle_name = dataset.pds4_bundle_name()
    template_dir = dataset.pds4_bundle_template_dir()
    bundle_root = bundle_results_root / bundle_name
    products = _CollectionProducts.in_bundle(bundle_root)
    failed_labels = 0

    # Every product in the data directory, found by its data label
    data_dir = _data_dir(bundle_root)
    label_files = list(data_dir.rglob(f'*{_DATA_LABEL_SUFFIX}'))

    # Sort by image name (extracted from filename)
    def get_image_name_from_label(path: FCPath) -> str:
        # Extract image name from filename
        # (e.g., "1234567890w_backplanes.lblx" -> "1234567890w")
        name = path.stem
        if '_backplanes' in name:
            return name.split('_backplanes')[0]
        return name

    label_files.sort(key=get_image_name_from_label)
    logger.info('Found %d label files in bundle', len(label_files))
    image_names = [
        dataset.pds4_lid_part_to_image_name(label_file.stem.replace('_backplanes', ''))
        for label_file in label_files
    ]

    # A collection label states at least one record, as the PDS4 schema requires of an
    # inventory, so a collection with no member is not written.  Both collections take
    # their members from the data labels, so they have members or lack them together.
    no_member = [] if len(image_names) > 0 else [_NO_MEMBER]
    template_base = Path(template_dir)

    # The data collection, whose label also states the range of the products' epochs,
    # so with no range it is not written rather than labeled with empty dates.  Each
    # template is parsed whether or not its collection is written, so one missing from
    # the tree raises rather than being passed over.
    data_template = pdstemplate.PdsTemplate(str(template_base / 'collection_data.lblx'))
    data_vars: dict[str, Any] = {'COLLECTION_DATA_CSV_PATH': str(products.data_inventory)}
    data_reasons = list(no_member)
    if epochs is None:
        data_reasons.append(_NO_RANGE)
    else:
        data_vars |= epochs.template_variables()
    if not _write_collection(
        'data',
        products.data_inventory,
        products.data_label,
        lidvids=[dataset.pds4_image_name_to_data_lidvid(name) for name in image_names],
        template=data_template,
        template_vars=data_vars,
        reasons_not_written=data_reasons,
        logger=logger,
    ):
        failed_labels += 1

    # The browse collection
    browse_template = pdstemplate.PdsTemplate(str(template_base / 'collection_browse.lblx'))
    if not _write_collection(
        'browse',
        products.browse_inventory,
        products.browse_label,
        lidvids=[dataset.pds4_image_name_to_browse_lidvid(name) for name in image_names],
        template=browse_template,
        template_vars={'COLLECTION_BROWSE_CSV_PATH': str(products.browse_inventory)},
        reasons_not_written=no_member,
        logger=logger,
    ):
        failed_labels += 1

    logger.info('Generated collection files: %d products', len(label_files))
    return failed_labels


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

    Both index labels are attempted, whichever of them fail, and the index
    tables are written whether or not the labels that describe them render.

    Its read of the supplemental files is the one the summary pass makes, so the
    range of the products' epochs is taken in the same read, through an
    :class:`~spindoctor.cli.pds4.epochs.EpochRangeScan`, and returned for the labels
    that state it.

    Both index tables and both index labels are cleared before any supplemental
    file is read, as :func:`~spindoctor.cli.pds4.labels.write_label` clears a
    label before it renders, and so are the two collection inventories and two
    collection labels :func:`generate_collection_files` writes after the index.
    A run refused over what a supplemental file holds therefore leaves no product of
    the summary pass, neither this run's nor an earlier run's: no index still
    describing the bundle as it was, and no inventory beside no index.

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
    failed_labels = 0

    # Get configured backplane types from config
    bodies_cfg = config.backplanes.bodies
    body_backplane_types = [bp['name'] for bp in bodies_cfg]
    rings_cfg = config.backplanes.rings
    ring_backplane_types = [bp['name'] for bp in rings_cfg]
    body_formats = {bp['name']: index_value_format(bp['units']) for bp in bodies_cfg}
    ring_formats = {bp['name']: index_value_format(bp['units']) for bp in rings_cfg}

    # A bundle with no data directory is not one a labels pass wrote.  The summary
    # pass runs this generator first, so the check is made here, before any product
    # of the pass is cleared or written into a bundle that is not there.
    data_dir = _data_dir(bundle_root)

    # Cleared before any supplemental file is read, by the rule write_label keeps
    # for a label that what is on disk is what this run wrote: the index's own
    # products, and the collection files the pass writes after it, so a refusal
    # over one cannot leave an earlier run's index describing the bundle as it was,
    # nor an earlier run's inventory and its label beside no index.
    supplemental_dir = bundle_root / 'document' / 'supplemental'
    bodies_tab = supplemental_dir / 'global_index_bodies.tab'
    bodies_label = supplemental_dir / 'global_index_bodies.lblx'
    rings_tab = supplemental_dir / 'global_index_rings.tab'
    rings_label = supplemental_dir / 'global_index_rings.lblx'
    index_products = (bodies_tab, bodies_label, rings_tab, rings_label)
    collection_products = _CollectionProducts.in_bundle(bundle_root).paths()
    for summary_product in (*index_products, *collection_products):
        summary_product.unlink(missing_ok=True)

    # Scan for all supplemental files
    supplemental_files: list[FCPath] = []
    for suppl_file in data_dir.rglob(f'*{_SUPPLEMENTAL_SUFFIX}'):
        supplemental_files.append(suppl_file)

    # Sort by image name (extracted from filename)
    def get_image_name_from_supplemental(path: FCPath) -> str:
        # Extract image name from filename
        # (e.g., "1234567890w_supplemental.txt" -> "1234567890w")
        return path.name.replace('_supplemental.txt', '')

    supplemental_files.sort(key=get_image_name_from_supplemental)
    logger.info('Found %d supplemental files', len(supplemental_files))

    # Collect body and ring statistics, every cell already rendered: both
    # tables are opened only once every value in them has been written out, so
    # nothing a render can raise leaves a table half-written.
    body_index_rows: list[list[str]] = []
    ring_index_rows: list[list[str]] = []
    # The range of the products' epochs, taken in this same read of the files.
    epochs = EpochRangeScan()

    for suppl_file in supplemental_files:
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
        bodies = backplanes.get('bodies', {})
        rings = backplanes.get('rings', {})

        # Derive pds4_path_stub from supplemental file path
        # Supplemental file is at: bundle_root/data/<pds4_path_stub>_supplemental.txt
        pds4_path_stub = _product_stub(suppl_file, data_dir)

        lid_part = suppl_file.stem.replace('_supplemental', '')
        image_name = dataset.pds4_lid_part_to_image_name(lid_part)
        lid = dataset.pds4_image_name_to_data_lid(image_name)
        # pds4_path_stub includes path and filename prefix
        # Path relative to data directory
        path_to_image = f'data/{pds4_path_stub}_backplanes.lblx'

        # Body index: one line per image per body
        for body_name, body_data in bodies.items():
            body_row: list[str] = [lid, body_name, path_to_image]
            body_backplanes = body_data.get('backplanes', {})
            # Add min/max columns for each configured backplane type
            for bp_type in body_backplane_types:
                body_row.extend(_index_cells(body_backplanes.get(bp_type), body_formats[bp_type]))
            body_index_rows.append(body_row)

        # Ring index: one line per image
        ring_backplanes = rings.get('backplanes', {})
        if ring_backplanes:
            ring_row: list[str] = [lid, path_to_image]
            # Add min/max columns for each configured ring backplane type
            for ring_type in ring_backplane_types:
                ring_row.extend(
                    _index_cells(ring_backplanes.get(ring_type), ring_formats[ring_type])
                )
            ring_index_rows.append(ring_row)

    # Generate global_index_bodies.tab
    bodies_tab_local = cast(Path, bodies_tab.get_local_path())
    with bodies_tab_local.open('w', newline='') as f:
        writer = csv.writer(f)
        # Build header: LID, body_name, path_to_image_file, then min/max for each backplane type
        header = ['LID', 'body_name', 'path_to_image_file']
        for bp_type in body_backplane_types:
            header.append(f'{bp_type}_min')
            header.append(f'{bp_type}_max')
        writer.writerow(header)
        writer.writerows(body_index_rows)
    bodies_tab.upload()
    logger.info('Generated global_index_bodies.tab with %d rows', len(body_index_rows))

    # Generate global_index_rings.tab
    rings_tab_local = cast(Path, rings_tab.get_local_path())
    # No explicit parent mkdir: get_local_path() creates parents (matching the
    # bodies index above), so an extra mkdir here was redundant and asymmetric.
    with rings_tab_local.open('w', newline='') as f:
        writer = csv.writer(f)
        # Build header: LID, path_to_image_file, then min/max for each ring type
        header = ['LID', 'path_to_image_file']
        # TODO Add planet name to rings table
        for ring_type in ring_backplane_types:
            header.append(f'{ring_type}_min')
            header.append(f'{ring_type}_max')
        writer.writerow(header)
        writer.writerows(ring_index_rows)
    rings_tab.upload()
    logger.info('Generated global_index_rings.tab with %d rows', len(ring_index_rows))

    # Generate label files using templates
    template_base = Path(template_dir)

    # Global index bodies label
    bodies_template = template_base / 'global_index_bodies.lblx'
    template = pdstemplate.PdsTemplate(str(bodies_template))
    template_vars = {
        'FILE_RECORDS': len(body_index_rows),
    }
    if write_label(template, template_vars, bodies_label, logger=logger):
        logger.info('Generated global_index_bodies.lblx')
    else:
        failed_labels += 1

    # Global index rings label
    rings_template = template_base / 'global_index_rings.lblx'
    template = pdstemplate.PdsTemplate(str(rings_template))
    template_vars = {
        'FILE_RECORDS': len(ring_index_rows),
    }
    if write_label(template, template_vars, rings_label, logger=logger):
        logger.info('Generated global_index_rings.lblx')
    else:
        failed_labels += 1

    logger.info(
        'Generated global index files: %d body rows, %d ring rows',
        len(body_index_rows),
        len(ring_index_rows),
    )
    return GlobalIndexOutcome(failed_labels=failed_labels, epochs=epochs.result())
