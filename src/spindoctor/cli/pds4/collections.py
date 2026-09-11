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
from spindoctor.cli.pds4.epochs import EpochRange, EpochRangeScan, NoEpochRange
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

The key is the unit the statistic is in, which for an angular plane is the
degrees restatement of the unit the configuration declares.  Two constraints
decide the formats.  The backplane arrays are float32, allocated so by both
per-source stages and cast to it by the writer, so no statistic carries more
than seven significant digits and a format printing more than that prints
noise.  Within that ceiling the geometry sets what is usable.  An angle in
degrees gets three decimals, since one pixel spans about 0.0003 degrees on the
sky for a narrow-field camera and 0.003 for a wide-field one; it writes
``1.235`` and ``-89.999``.  A ring radius in kilometers gets one, since the
radii run from 7e4 to 5e5 km, where float32 spacing is 0.008 to 0.03 km; it
writes ``74658.0`` and ``136780.0``.  A resolution in degrees per pixel gets
eight decimals and writes ``0.00015470`` and ``0.80386227``.  The largest such
value on the real frames tried was 0.80, on an edge-on wide-field ring frame,
and at that end the eighth decimal sits at the edge of what a float32 plane
carries, whose spacing there is 6e-8.  A resolution in kilometers per pixel runs
from 6e-4 a hundred kilometers off a small moon to 7e4 at the grazing limb of a
wide-field frame, eight orders of magnitude that no fixed decimal count fits, so
it gets five significant figures, written positionally: ``0.00060000``,
``6.1343``, ``4200.0`` and ``70853``.

Every format writes a plain decimal number, never one with an exponent or a
trailing point, since the tables are read by people.  Written positionally, a
value never has its integer part rounded away, so from 1e7 up the significant
figures format writes more than seven figures, past what a float32 statistic
carries; no real statistic has come near, the largest seen being 70853.  A
value that rounds up to the next power of ten gains a figure, so ``9.99996``
writes ``10.0000``.  What no format fixes is a column's width: values under
one format differ in length, so the width of a column is the widest value
written in it and is not derivable from the format alone.

The bodies table and the rings table share the mapping so that a value cannot
mean one thing in one and something else in the other, and it is public so that
a label describing a table can say how the column was written.
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


_REGENERATE_REMEDY = (
    "The labels pass writes each product's supplemental file and data label: regenerate the "
    'bundle into an empty directory'
)
"""What a refusal over a product the index cannot take says to do about it."""


def _json_kind(value: Any) -> str:
    """Name the kind of JSON value a supplemental file holds in place of an object.

    Parameters:
        value: What the file's JSON parsed to, anything but an object.

    Returns:
        ``an array``, ``a string``, ``a boolean``, ``null`` or ``a number``, as a
        message says what the file holds.
    """
    if isinstance(value, list):
        return 'an array'
    if isinstance(value, str):
        return 'a string'
    # Python counts a boolean as an integer, so it is named before a number is.
    if isinstance(value, bool):
        return 'a boolean'
    if value is None:
        return 'null'
    return 'a number'


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


def _data_labels(data_dir: FCPath) -> list[FCPath]:
    """Return every data label in the data tree, which is how a product is found.

    The collection inventory lists a product by its data label, and the global index
    holds every supplemental file to having one beside it, so both list them here.

    Parameters:
        data_dir: The bundle's data directory.

    Returns:
        Each ``<stub>_backplanes.lblx`` under it, in the order the listing gives.
    """
    return list(data_dir.rglob(f'*{_DATA_LABEL_SUFFIX}'))


def _product_stub(path: FCPath, data_dir: FCPath, suffix: str) -> str:
    """Return a product's path stub, read off one of its files.

    Parameters:
        path: The product's supplemental file or data label, under ``data_dir``.
        data_dir: The bundle's data directory.
        suffix: What follows the stub in the file's name.

    Returns:
        The file's path relative to ``data_dir``, in POSIX form, less ``suffix``: the
        stub both of a product's files share, as in ``shard0/1234567890w``.
    """
    return path.relative_to(data_dir).as_posix().removesuffix(suffix)


@dataclass(frozen=True)
class _CollectionProducts:
    """Where in a bundle the collection tables and labels the summary pass writes go.

    Attributes:
        data_table: The data collection's inventory, ``data/collection_data.tab``.
        data_label: Its label, ``data/collection_data.lblx``.
        browse_table: The browse collection's inventory, ``browse/collection_browse.tab``.
        browse_label: Its label, ``browse/collection_browse.lblx``.
    """

    data_table: FCPath
    data_label: FCPath
    browse_table: FCPath
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
            data_table=bundle_root / 'data' / 'collection_data.tab',
            data_label=bundle_root / 'data' / 'collection_data.lblx',
            browse_table=bundle_root / 'browse' / 'collection_browse.tab',
            browse_label=bundle_root / 'browse' / 'collection_browse.lblx',
        )

    def paths(self) -> tuple[FCPath, ...]:
        """Return all four paths.

        Returns:
            The data collection's table and label, then the browse collection's.
        """
        return (self.data_table, self.data_label, self.browse_table, self.browse_label)


def generate_collection_files(
    bundle_results_root: FCPath,
    dataset: DataSet,
    logger: PdsLogger,
    *,
    epochs: EpochRange | NoEpochRange,
) -> int:
    """Generate collection CSV and label files for the bundle.

    Every collection label is attempted, whichever of them fail: a broken data
    collection template must not hide a broken browse collection one.  The
    inventory tables are written whether or not the labels that describe them
    render.

    The data collection label states the time range of the products the collection
    holds, which is ``epochs``: the range :func:`generate_global_index_files` takes in
    its read of the supplemental files, which is why the summary pass runs that
    first.  The label writes it to whole seconds, the start rounded down and the stop
    up, so the range contains every product's own start and stop.  With no range to
    state -- no supplemental file at all, or one whose epochs could not be had -- the
    data collection label is counted as not written, with an error saying why, rather
    than rendered with empty dates, which PDS4 does not accept; whatever an earlier
    run left at its path is removed, so a label on disk is always one this run wrote.

    Every collection template the dataset declares is required.  The caller is
    expected to have checked them before processing anything, so one that is
    missing here raises rather than being passed over.

    Parameters:
        bundle_results_root: Root directory of the bundle. The bundle data directory
            will be scanned for all backplane label files.
        dataset: The dataset instance for bundle-specific methods.
        logger: Logger for diagnostic messages.
        epochs: The earliest start and the latest stop of the products' exposures,
            or why there is no such range.

    Returns:
        The number of collection labels that could not be rendered, the data
        collection label counted among them when there is no range for it to state.

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
    label_files = _data_labels(data_dir)

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

    # Generate collection_data.tab
    collection_data_csv = products.data_table
    collection_data_local = cast(Path, collection_data_csv.get_local_path())
    collection_data_local.parent.mkdir(parents=True, exist_ok=True)
    with collection_data_local.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Member Status', 'LIDVID_LID'])
        for label_file in label_files:
            lid_part = label_file.stem.replace('_backplanes', '')
            image_name = dataset.pds4_lid_part_to_image_name(lid_part)
            lidvid = dataset.pds4_image_name_to_data_lidvid(image_name)
            writer.writerow(['P', lidvid])
    collection_data_csv.upload()
    logger.info('Generated "collection_data.tab": %s', collection_data_csv)

    # Generate collection label files using template
    template_base = Path(template_dir)

    # Collection data label.  It states the range of the products' epochs, so a
    # collection with no range is a label not written rather than one stating
    # empty dates.  The path is cleared either way, as write_label clears it, so
    # an earlier run's label never describes this run's inventory.
    collection_data_template = template_base / 'collection_data.lblx'
    template = pdstemplate.PdsTemplate(str(collection_data_template))
    collection_data_label = products.data_label
    if isinstance(epochs, NoEpochRange):
        collection_data_label.unlink(missing_ok=True)
        logger.error(
            'The data collection label %s was not written: it states the time range of '
            'the products the collection holds, and %s',
            collection_data_label,
            epochs.reason,
        )
        failed_labels += 1
    else:
        template_vars = {
            'COLLECTION_DATA_CSV_PATH': str(collection_data_csv),
            **epochs.template_variables(),
        }
        if write_label(template, template_vars, collection_data_label, logger=logger):
            logger.info('Generated "collection_data.lblx"')
        else:
            failed_labels += 1

    # Generate collection_browse.tab (must be written before collection_browse.lblx)
    collection_browse_csv = products.browse_table
    collection_browse_local = cast(Path, collection_browse_csv.get_local_path())
    collection_browse_local.parent.mkdir(parents=True, exist_ok=True)
    with collection_browse_local.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Member Status', 'LIDVID_LID'])
        for label_file in label_files:
            lid_part = label_file.stem.replace('_backplanes', '')
            image_name = dataset.pds4_lid_part_to_image_name(lid_part)
            lidvid = dataset.pds4_image_name_to_browse_lidvid(image_name)
            writer.writerow(['P', lidvid])
    collection_browse_csv.upload()
    logger.info('Generated "collection_browse.tab": %s', collection_browse_csv)

    # Collection browse label
    collection_browse_template = template_base / 'collection_browse.lblx'
    template = pdstemplate.PdsTemplate(str(collection_browse_template))
    collection_browse_label = products.browse_label
    template_vars = {
        'COLLECTION_BROWSE_CSV_PATH': str(collection_browse_csv),
    }
    if write_label(template, template_vars, collection_browse_label, logger=logger):
        logger.info('Generated "collection_browse.lblx"')
    else:
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
            label states, or why there is no such range.
    """

    failed_labels: int
    epochs: EpochRange | NoEpochRange


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
    that state it.  A supplemental file whose navigation document records no epochs
    a label can state leaves no range, since a range taken over the rest could leave
    its product outside, and is indexed all the same.  One that cannot be read, or
    does not hold a JSON object, refuses the run, since left out of the index it would
    still be listed in the collection's inventory.  So does a supplemental file with no
    data label beside it, or a data label with no supplemental file: the inventory
    finds a product by its data label, and the index and the range by its supplemental
    file, so the two would disagree about it.  Both are listed once, and compared
    before any file is read.

    Both index tables and both index labels are cleared before any supplemental
    file is read, as :func:`~spindoctor.cli.pds4.labels.write_label` clears a
    label before it renders, and so are the two collection tables and two
    collection labels :func:`generate_collection_files` writes after the index.
    A run refused over what the data tree holds therefore leaves no product
    of the summary pass, neither this run's nor an earlier run's: no index still
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
        products' epochs over every supplemental file read, or why there is none:
        there is no supplemental file, or one records no epochs a label can state.

    Raises:
        FileNotFoundError: If the bundle has no data directory to scan, which is
            checked before any product of the pass is cleared or written, or an
            index template is not in the dataset's template directory.
        KeyError: If a configured plane's statistic is in a unit the index has no
            column format for.
        ValueError: If a supplemental file cannot be read or does not hold a JSON
            object, the message naming the file and the reason or what it holds
            instead; if a supplemental file has no data label beside it, or a data
            label no supplemental file, the message naming both; or if one holds a
            statistic no column can -- one in a unit other than the one the
            configuration gives its plane, or with a minimum or maximum that is NaN
            or infinite -- the message naming the file and the plane, what the file
            records there, and what to regenerate.  Every supplemental file is read,
            and every value in both tables rendered, before either table is opened,
            so none of these leaves a table half-written.
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
    # Every plane's format is looked up before any supplemental file is read,
    # so a plane declared in a unit the table cannot size fails the run here
    # rather than after half a table has been written.
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

    # The labels pass writes a product's supplemental file and then renders its data
    # label, and the collection inventory finds a product by its data label, so one of
    # the two without the other is a product the index and the inventory would disagree
    # about: indexed and inside the range but not listed, or listed but neither.  Both
    # lists are taken once and compared before any file is read.
    supplemental_stubs = {
        _product_stub(path, data_dir, _SUPPLEMENTAL_SUFFIX) for path in supplemental_files
    }
    label_stubs = {
        _product_stub(path, data_dir, _DATA_LABEL_SUFFIX) for path in _data_labels(data_dir)
    }
    unlabeled = sorted(supplemental_stubs - label_stubs)
    if len(unlabeled) > 0:
        raise ValueError(
            f'Supplemental file {data_dir / (unlabeled[0] + _SUPPLEMENTAL_SUFFIX)} has no '
            f'data label beside it: {data_dir / (unlabeled[0] + _DATA_LABEL_SUFFIX)} is not '
            'there, and the collection inventory, which finds products by their data '
            f'labels, would leave out a product the index holds. {_REGENERATE_REMEDY}'
        )
    unindexed = sorted(label_stubs - supplemental_stubs)
    if len(unindexed) > 0:
        raise ValueError(
            f'Data label {data_dir / (unindexed[0] + _DATA_LABEL_SUFFIX)} has no supplemental '
            f'file beside it: {data_dir / (unindexed[0] + _SUPPLEMENTAL_SUFFIX)} is not '
            'there, and the index and the range of epochs, which read the supplemental '
            f'files, would leave out a product the collection inventory lists. '
            f'{_REGENERATE_REMEDY}'
        )

    # Collect body and ring statistics, every cell already rendered: both
    # tables are opened only once every value in them has been written out, so
    # nothing a render can raise leaves a table half-written.
    body_index_rows: list[list[str]] = []
    ring_index_rows: list[list[str]] = []
    # The range of the products' epochs, taken in this same read of the files.
    epochs = EpochRangeScan()

    for suppl_file in supplemental_files:
        # The labels pass writes every supplemental file, and writes it as a JSON
        # object, so one that cannot be read, or holds anything else, is a broken tree
        # rather than a product to pass over: left out of the index it would still be
        # listed in the collection's inventory, with no epochs for the range.  The run
        # is refused here, before either table exists, as it is for a statistic no
        # column can hold.
        try:
            suppl_text = suppl_file.read_text()
            metadata = json.loads(suppl_text)
        except (OSError, ValueError) as exc:
            raise ValueError(
                f'Supplemental file {suppl_file} could not be read: {exc}. {_REGENERATE_REMEDY}'
            ) from exc
        if not isinstance(metadata, dict):
            raise ValueError(
                f'Supplemental file {suppl_file} holds {_json_kind(metadata)} where a '
                f'supplemental document is a JSON object. {_REGENERATE_REMEDY}'
            )

        epochs.include(f'supplemental file {suppl_file}', metadata.get('navigation'))
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
        pds4_path_stub = _product_stub(suppl_file, data_dir, _SUPPLEMENTAL_SUFFIX)

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
