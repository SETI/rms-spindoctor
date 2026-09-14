"""The data and browse collections of a PDS4 bundle, and the products they are made of.

The summary pass inventories each collection from the labels of its own kind on disk: the
data collection from the data labels in the bundle's ``data/`` tree, and the browse
collection from the browse labels in its ``browse/`` tree.
:func:`generate_collection_files` writes each inventory and renders its label, and holds
each image's products against each other.  The functions naming the bundle's data
directory and the supplemental files in it, and the one clearing what this module writes,
are shared with the global index generator in :mod:`~spindoctor.cli.pds4.global_index`,
which the summary pass runs first.
"""

import csv
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import pdstemplate
from filecache import FCPath
from pdslogger import PdsLogger

from spindoctor.cli.pds4.epochs import EpochRange
from spindoctor.cli.pds4.labels import write_label
from spindoctor.cli.pds4.targets import Pds4Target
from spindoctor.dataset.dataset import DataSet


def data_directory(bundle_root: FCPath) -> FCPath:
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

_BROWSE_LABEL_SUFFIX = '_summary.lblx'
"""What follows a product's path stub in the name of its browse label."""


def _product_stub(path: FCPath, tree: FCPath, suffix: str) -> str:
    """Return a product's path stub, read off one of its files.

    Parameters:
        path: The file, under ``tree``.
        tree: The bundle directory the file is under, ``data`` or ``browse``.
        suffix: What follows the stub in the file's name.

    Returns:
        The file's path relative to ``tree``, in POSIX form, less ``suffix``, as in
        ``shard0/1234567890w``.
    """
    return path.relative_to(tree).as_posix().removesuffix(suffix)


def _products_by_stub(tree: FCPath, suffix: str) -> dict[str, FCPath]:
    """Return the files of one kind in a bundle directory, keyed by path stub.

    Parameters:
        tree: The bundle directory to search, ``data`` or ``browse``.  One that is
            not there holds none.
        suffix: What follows a product's path stub in the name of the files sought.

    Returns:
        Every file under ``tree`` whose name ends in ``suffix``, keyed by its path
        stub, in the order of the products' names -- each file's name less ``suffix``,
        which is the last part of the product's LID -- whatever directory each is in.
    """
    files = sorted(tree.rglob(f'*{suffix}'), key=lambda path: path.name.removesuffix(suffix))
    return {_product_stub(path, tree, suffix): path for path in files}


def _image_name(stub: str, dataset: DataSet) -> str:
    """Return the name of the image a product's path stub belongs to.

    Parameters:
        stub: The product's path stub, as in ``shard0/1234567890w``.
        dataset: The dataset whose LID scheme the stub's last part follows.

    Returns:
        The image name the dataset gives the stub's last part, which is a LID part.
    """
    return dataset.pds4_lid_part_to_image_name(stub.rsplit('/', 1)[-1])


def supplemental_files(data_dir: FCPath) -> dict[str, FCPath]:
    """Return the supplemental files in a bundle's data tree, keyed by path stub.

    Parameters:
        data_dir: The bundle's data directory, as :func:`data_directory` names it.

    Returns:
        Every ``<stub>_supplemental.txt`` under ``data_dir``, keyed by its path stub, as
        in ``shard0/1234567890w``, in the order of the products' names -- each file's name
        less the suffix, which is the last part of the product's LID -- whatever
        directory each is in.
    """
    return _products_by_stub(data_dir, _SUPPLEMENTAL_SUFFIX)


def data_products(data_dir: FCPath) -> dict[str, FCPath]:
    """Return the data collection's members: the data labels in a bundle's data tree.

    The data inventory lists these, and the global index tables index these, so that
    the two cannot disagree about what the bundle holds.

    Parameters:
        data_dir: The bundle's data directory, as :func:`data_directory` names it.

    Returns:
        Every ``<stub>_backplanes.lblx`` under ``data_dir``, keyed by its path stub, in
        the order the data inventory lists them: the order of the products' names,
        whatever directory each is in.
    """
    return _products_by_stub(data_dir, _DATA_LABEL_SUFFIX)


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


def clear_collection_products(bundle_root: FCPath) -> None:
    """Remove every file :func:`generate_collection_files` can write into one bundle.

    The global index generator, which the summary pass runs first, calls this once it
    has found the bundle's data directory and before it reads any supplemental file, so
    that a collection file on disk after a summary pass is one that pass wrote.

    Parameters:
        bundle_root: The bundle's own directory.
    """
    for path in _CollectionProducts.in_bundle(bundle_root).paths():
        path.unlink(missing_ok=True)


def _write_inventory(inventory: FCPath, primaries: list[str], secondaries: list[str]) -> None:
    """Write a collection inventory: its primary members, then its secondary members.

    The inventory is comma-separated with no header: one ``P,<lidvid>`` line per
    primary member and then one ``S,<lidvid>`` line per secondary member, each in the
    order given, and every line ending in a line feed alone, the last included.  Its
    number of lines is therefore its number of members, which is what the collection
    label states as its records.

    Parameters:
        inventory: Where the inventory goes.
        primaries: The primary members' LIDVIDs, the products the collection holds.
        secondaries: The secondary members' LIDVIDs, the products it cites.
    """
    with inventory.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(['P', lidvid] for lidvid in primaries)
        writer.writerows(['S', lidvid] for lidvid in secondaries)


_NO_MEMBER = 'the collection has no member, and its label has to state at least one record'
"""Why a collection with no member is not written.

The PDS4 schema requires a collection inventory to hold at least one record.
"""

_NO_RANGE = (
    'no data label in the data tree has a supplemental file beside it, so there is no time '
    'range for its label to state'
)
"""Why the data collection is not written when no member's supplemental file gives it a range."""


def write_collection(
    name: str,
    inventory: FCPath,
    label: FCPath,
    *,
    primaries: list[str],
    secondaries: list[str],
    template: pdstemplate.PdsTemplate,
    template_vars: dict[str, Any],
    reasons_not_written: list[str],
    logger: PdsLogger,
) -> bool:
    """Write one generated collection's inventory and then its label, or neither of them.

    A collection is written only when its label can state what PDS4 requires of it.  One
    with no member cannot, since its label has to state at least one record, and nor can
    one the caller gives another reason for.  Such a collection is not written at all:
    whatever is at either path is removed, so that a collection on disk is always one
    this run wrote, and one error names the collection, both paths and every reason.
    Otherwise the inventory is written, and then the label, which reads the inventory's
    size, checksum and record count from the file; the inventory stays whether or not
    the label renders.

    Parameters:
        name: The collection's name, as the error gives it: ``data``, ``browse`` or
            ``miscellaneous``.
        inventory: Where the collection's inventory goes.
        label: Where the collection's label goes.
        primaries: The LIDVIDs of the products the collection holds, in the order its
            inventory lists them.
        secondaries: The LIDVIDs of the products it cites, listed after the primaries.
        template: The parsed template the label renders from.
        template_vars: The variables the label's template resolves against.
        reasons_not_written: Why else the collection cannot be written, one phrase each,
            or an empty list when nothing else stands in its way.
        logger: Logger for diagnostic messages.

    Returns:
        True if the collection's label is on disk, False if it is not.
    """
    has_member = len(primaries) + len(secondaries) > 0
    reasons = ([] if has_member else [_NO_MEMBER]) + reasons_not_written
    if len(reasons) > 0:
        inventory.unlink(missing_ok=True)
        label.unlink(missing_ok=True)
        logger.error(
            'The %s collection was not written, neither its inventory %s nor its label %s: %s',
            name,
            inventory,
            label,
            '; and '.join(reasons),
        )
        return False
    _write_inventory(inventory, primaries, secondaries)
    logger.info('Generated "%s": %s', inventory.name, inventory)
    if not write_label(template, template_vars, label, logger=logger):
        return False
    logger.info('Generated "%s"', label.name)
    return True


@dataclass(frozen=True)
class CollectionOutcome:
    """What generating the collection files came to.

    Attributes:
        failed_labels: The number of collection labels not written: each that could not
            be rendered, and each collection that could not be written, counted once
            whatever its number of reasons.
        disagreeing_images: The number of images whose products disagree: an image with
            a data label and no browse label, or with a browse label or a supplemental
            file and no data label.  Each is counted once, whatever it lacks.
    """

    failed_labels: int
    disagreeing_images: int


def _disagreeing_images(
    data_labels: dict[str, FCPath],
    browse_labels: dict[str, FCPath],
    supplemental_files: dict[str, FCPath],
    *,
    bundle_root: FCPath,
    dataset: DataSet,
    logger: PdsLogger,
) -> int:
    """Log each image whose products disagree, and return how many there are.

    Every data product has a browse product, so an image's products agree when it has
    both a data label and a browse label, and disagree when it has one of them without
    the other, or a supplemental file and no data label.  A data label with no
    supplemental file is not checked, since the labels pass writes an image's
    supplemental file before its data label.

    Each image whose products disagree gets one error, naming the image, the files of
    it that are there, and the label it lacks, at the path where that label belongs.

    Parameters:
        data_labels: The data labels in the data tree, keyed by path stub.
        browse_labels: The browse labels in the browse tree, keyed by path stub.
        supplemental_files: The supplemental files in the data tree, keyed by path stub.
        bundle_root: The bundle's own directory, under which a missing label belongs.
        dataset: The dataset whose LID scheme names each image.
        logger: Logger for the errors.

    Returns:
        The number of images whose products disagree, each counted once.
    """
    stubs = data_labels.keys() | browse_labels.keys() | supplemental_files.keys()
    disagreeing = 0
    for stub in sorted(stubs, key=lambda stub: stub.rsplit('/', 1)[-1]):
        if stub in data_labels:
            if stub in browse_labels:
                continue
            lacks = 'browse label'
            missing = bundle_root / 'browse' / f'{stub}{_BROWSE_LABEL_SUFFIX}'
            held = [data_labels[stub]]
        else:
            lacks = 'data label'
            missing = bundle_root / 'data' / f'{stub}{_DATA_LABEL_SUFFIX}'
            held = [files[stub] for files in (browse_labels, supplemental_files) if stub in files]
        logger.error(
            'The products of image %s disagree: it has %s, and no %s at %s',
            _image_name(stub, dataset),
            ' and '.join(str(path) for path in held),
            lacks,
            missing,
        )
        disagreeing += 1
    return disagreeing


def generate_collection_files(
    bundle_results_root: FCPath,
    dataset: DataSet,
    logger: PdsLogger,
    *,
    epochs: EpochRange | None,
    targets: Sequence[Pds4Target],
) -> CollectionOutcome:
    """Inventory and label the data and browse collections, checking the products agree.

    The collections describe what is on disk, and what they cannot describe is counted
    against the run.  Each inventory lists one product per line, ``P,<lidvid>``, in the
    order of the products' names, the last part of each member's LID, whatever
    directory each is in: ``data/collection_data.csv`` the data products, found
    by the data labels in the data tree, and ``browse/collection_browse.csv`` the
    browse products, found by the browse labels in the browse tree.  An inventory has
    no header, and every line, the last included, ends in a line feed alone, so the
    records its label counts are its products.

    The data collection label states the time range of the products the collection
    holds, which is ``epochs``: the range
    :func:`~spindoctor.cli.pds4.global_index.generate_global_index_files` takes in
    its read of the supplemental files, which is why the summary pass runs that
    first.  The label writes it to whole seconds, the start rounded down and the stop
    up.  It names every target the collection's members name, ``targets``, which that
    generator takes in the same read, as a Mission Science Data collection's label has to.

    Every data product has a browse product, so each image's products are checked
    against each other.  An image with a data label and no browse label, or with a
    browse label or a supplemental file and no data label, disagrees: it gets one error
    naming the image, the files of it that are there and the label it lacks, and it is
    counted once.  It is still listed in whichever inventory holds a product of it.  A
    data label with no supplemental file is not checked, since the labels pass writes
    an image's supplemental file before its data label.

    A collection is written only when its label can state what PDS4 requires of it:
    at least one record, and, for the data collection, the time range.  One that
    cannot is not written at all, neither its inventory nor its label, and counts once
    as a label not written, with one error naming the collection and every reason: a
    collection with no label of its kind on disk, and the data collection when
    ``epochs`` is None.  Whatever an earlier run left at either of its paths is
    removed, so a collection on disk is always one this run wrote.  The check and this
    rule are one rule at two scales, each image holding all its products and each
    collection at least one member: an empty collection is refused by this rule even
    over a bundle with no image, where the check has nothing to count.

    Every collection label that can be written is attempted, whichever of them fail: a
    broken data collection template must not hide a broken browse collection one.  An
    inventory is written before its label, which reads the inventory's size, checksum
    and record count, and stays whether or not the label renders.

    Every collection template the dataset declares is required.  The caller is
    expected to have checked them before processing anything, so one that is
    missing here raises rather than being passed over, whether or not its collection
    can be written.

    Parameters:
        bundle_results_root: Root directory of the bundle.  The bundle's data and
            browse directories are scanned for their labels.
        dataset: The dataset instance for bundle-specific methods.
        logger: Logger for diagnostic messages.
        epochs: The earliest start and the latest stop of the products' exposures,
            or None when no data label in the data tree has a supplemental file
            beside it.
        targets: Every target the products name, in the targets table's order, which
            the data collection label names, handed to its template as ``TARGETS``.

    Returns:
        The number of collection labels not written, each label that could not be
        rendered and each collection that could not be written counted once whatever
        its number of reasons, and the number of images whose products disagree.

    Raises:
        FileNotFoundError: If the bundle has no data directory to scan, or a
            collection template is not in the dataset's template directory.
    """

    bundle_name = dataset.pds4_bundle_name()
    template_dir = dataset.pds4_bundle_template_dir()
    bundle_root = bundle_results_root / bundle_name
    products = _CollectionProducts.in_bundle(bundle_root)
    failed_labels = 0

    # Each collection's products, found by their labels: the data products by the data
    # labels in the data tree and the browse products by the browse labels in the
    # browse tree, so that each inventory lists what is on disk.
    data_dir = data_directory(bundle_root)
    data_labels = data_products(data_dir)
    browse_labels = _products_by_stub(bundle_root / 'browse', _BROWSE_LABEL_SUFFIX)
    logger.info(
        'Found %d data labels and %d browse labels in bundle', len(data_labels), len(browse_labels)
    )
    data_names = [_image_name(stub, dataset) for stub in data_labels]
    browse_names = [_image_name(stub, dataset) for stub in browse_labels]
    template_base = Path(template_dir)

    # Every data product has a browse product (the operator's ruling on #602), and a
    # supplemental file belongs to a data product, so each image's products are held
    # against each other.
    disagreeing_images = _disagreeing_images(
        data_labels,
        browse_labels,
        supplemental_files(data_dir),
        bundle_root=bundle_root,
        dataset=dataset,
        logger=logger,
    )

    # The data collection, whose label states at least one record, as the PDS4 schema
    # requires of an inventory, and the range of the products' epochs, so with no member
    # or no range it is not written rather than labeled with no record or empty dates.
    # Each template is parsed whether or not its collection is written, so one missing
    # from the tree raises rather than being passed over.
    data_template = pdstemplate.PdsTemplate(str(template_base / 'collection_data.lblx'))
    data_vars: dict[str, Any] = {
        'COLLECTION_DATA_CSV_PATH': products.data_inventory.as_posix(),
        'TARGETS': targets,
    }
    data_reasons: list[str] = []
    if epochs is None:
        data_reasons.append(_NO_RANGE)
    else:
        data_vars |= epochs.template_variables()
    if not write_collection(
        'data',
        products.data_inventory,
        products.data_label,
        primaries=[dataset.pds4_image_name_to_data_lidvid(name) for name in data_names],
        secondaries=[],
        template=data_template,
        template_vars=data_vars,
        reasons_not_written=data_reasons,
        logger=logger,
    ):
        failed_labels += 1

    # The browse collection, whose label states at least one record
    browse_template = pdstemplate.PdsTemplate(str(template_base / 'collection_browse.lblx'))
    if not write_collection(
        'browse',
        products.browse_inventory,
        products.browse_label,
        primaries=[dataset.pds4_image_name_to_browse_lidvid(name) for name in browse_names],
        secondaries=[],
        template=browse_template,
        template_vars={'COLLECTION_BROWSE_CSV_PATH': products.browse_inventory.as_posix()},
        reasons_not_written=[],
        logger=logger,
    ):
        failed_labels += 1

    logger.info(
        'Generated collection files: %d data products, %d browse products',
        len(data_labels),
        len(browse_labels),
    )
    return CollectionOutcome(failed_labels=failed_labels, disagreeing_images=disagreeing_images)
