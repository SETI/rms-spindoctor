"""The global index tables of a PDS4 bundle, and the formats their values are written in.

The summary pass reads every supplemental file the labels pass wrote once, here, and
builds two tables from them: one row for each body of each image the data collection
holds, and one row for each such image with ring backplanes, each giving the minimum and
maximum every configured plane spans.  The same read takes the range of the products'
epochs, which the data collection label and the bundle label state.
"""

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

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
    data_dir = data_directory(bundle_root)

    # Cleared before any supplemental file is read, by the rule write_label keeps
    # for a label that what is on disk is what this run wrote: the index's own
    # products, and the collection files and run-level products the pass writes after
    # it, so a refusal over one cannot leave an earlier run's index describing the
    # bundle as it was, nor an earlier run's inventory and its label beside no index.
    supplemental_dir = bundle_root / 'document' / 'supplemental'
    bodies_tab = supplemental_dir / 'global_index_bodies.tab'
    bodies_label = supplemental_dir / 'global_index_bodies.lblx'
    rings_tab = supplemental_dir / 'global_index_rings.tab'
    rings_label = supplemental_dir / 'global_index_rings.lblx'
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
