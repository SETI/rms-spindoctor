"""The run-level products of a PDS4 bundle: its label, its readme and its static collections.

The summary pass writes these last, once the global index and the data and browse
collections are on disk, since the bundle label declares every collection the bundle
holds and is kept only over a bundle that holds them all.  Each is rendered from a
template in the dataset's template directory or copied from it:

- ``readme.txt``, copied to the bundle's own directory;
- the user guide, a PDF the template directory holds or does not, copied into
  ``document/user_guide/`` with its label rendered beside it when it is there;
- the metakernel ``kernels.ker``, copied into ``spice_kernels/``, with its label
  ``kernels.lblx`` rendered beside it;
- the context, document, SPICE kernel and XML schema collections, each an inventory the
  template directory ships, copied into the collection's directory, with a collection
  label rendered over it;
- ``bundle.lblx``, rendered last.
"""

from dataclasses import dataclass
from typing import Any
from xml.etree import ElementTree

import pdstemplate
from filecache import FCPath
from pdslogger import PdsLogger

from spindoctor.cli.pds4.epochs import EpochRange
from spindoctor.cli.pds4.labels import write_label
from spindoctor.dataset.dataset import DataSet

PDS4_NAMESPACES = {'pds': 'http://pds.nasa.gov/pds4/pds/v1'}
"""The PDS4 common dictionary's namespace, under the prefix the element paths use."""

_STATIC_COLLECTIONS = ('context', 'document', 'spice_kernels', 'xml_schema')
"""The collections whose inventories the template directory ships, in the order written.

Each is written into the bundle directory of its name: its inventory
``collection_<name>.csv`` and its label ``collection_<name>.lblx``, from the files of the
same names in the template directory.
"""

_README = 'readme.txt'
"""The readme's name, in the template directory and at the bundle's root."""

_BUNDLE_LABEL = 'bundle.lblx'
"""The bundle label's name, as a template and at the bundle's root."""

_METAKERNEL = 'kernels.ker'
"""The metakernel's name, in the template directory and in ``spice_kernels/``."""

_LABEL_SUFFIX = '.lblx'
"""What a label's name ends in, beside the file of the same stem that it describes."""

_PRIMARY_MEMBER = b'P,'
"""How an inventory line naming a primary member begins."""


def _inventory(bundle_root: FCPath, collection: str) -> FCPath:
    """Return where a static collection's inventory goes in a bundle.

    Parameters:
        bundle_root: The bundle's own directory.
        collection: The collection's name, which is its directory's.

    Returns:
        ``<bundle_root>/<collection>/collection_<collection>.csv``.
    """
    return bundle_root / collection / f'collection_{collection}.csv'


def _user_guide(bundle_root: FCPath, dataset: DataSet) -> FCPath:
    """Return where the user guide goes in a bundle.

    Parameters:
        bundle_root: The bundle's own directory.
        dataset: The dataset whose user guide it is.

    Returns:
        ``<bundle_root>/document/user_guide/<the guide's file name>``.
    """
    return bundle_root / 'document' / 'user_guide' / dataset.pds4_user_guide_file_name()


def _label_beside(path: FCPath) -> FCPath:
    """Return where the label of a file goes: beside it, under the same stem.

    Parameters:
        path: The file the label describes.

    Returns:
        The path with its suffix replaced by ``.lblx``.
    """
    return path.with_suffix(_LABEL_SUFFIX)


def bundle_product_paths(bundle_root: FCPath, dataset: DataSet) -> tuple[FCPath, ...]:
    """Return every file :func:`generate_bundle_products` can write into one bundle.

    Parameters:
        bundle_root: The bundle's own directory.
        dataset: The dataset whose user guide the bundle can hold.

    Returns:
        The bundle label and the readme; each static collection's inventory and label;
        the metakernel and its label; and the user guide and its label.
    """
    inventories = [_inventory(bundle_root, name) for name in _STATIC_COLLECTIONS]
    metakernel = bundle_root / 'spice_kernels' / _METAKERNEL
    user_guide = _user_guide(bundle_root, dataset)
    return (
        bundle_root / _BUNDLE_LABEL,
        bundle_root / _README,
        *(path for inventory in inventories for path in (inventory, _label_beside(inventory))),
        metakernel,
        _label_beside(metakernel),
        user_guide,
        _label_beside(user_guide),
    )


@dataclass(frozen=True)
class BundleProductsOutcome:
    """What generating the run-level products came to.

    Attributes:
        failed_labels: The number of run-level labels not written: each that could not be
            rendered, and the bundle label when there is no time range for it to state or
            the bundle holds no label for a collection it declares.
    """

    failed_labels: int


def _render(
    template_dir: FCPath,
    label: FCPath,
    template_vars: dict[str, Any],
    *,
    logger: PdsLogger,
) -> bool:
    """Render the template of a label's own name into that label.

    Parameters:
        template_dir: The dataset's template directory, which holds the template under
            the label's name.
        label: Where the label goes.
        template_vars: The variables the template resolves against.
        logger: Logger for diagnostic messages.

    Returns:
        True if the label is on disk, False if it is not.

    Raises:
        FileNotFoundError: If the template is not in the template directory.
    """
    template = pdstemplate.PdsTemplate((template_dir / label.name).as_posix())
    if not write_label(template, template_vars, label, logger=logger):
        return False
    logger.info('Generated "%s": %s', label.name, label)
    return True


def _copy(source: FCPath, destination: FCPath, *, logger: PdsLogger) -> None:
    """Copy one file of the template directory into the bundle, as it is.

    Parameters:
        source: The file in the template directory.
        destination: Where it goes in the bundle; its directory is made if it is not
            there.
        logger: Logger for diagnostic messages.
    """
    destination.write_bytes(source.read_bytes())
    logger.info('Copied "%s": %s', destination.name, destination)


def _without_primary_members(inventory: bytes) -> bytes:
    """Return an inventory less the lines that name a primary member.

    Parameters:
        inventory: The inventory's bytes, one member per line.

    Returns:
        Every line that does not begin ``P,``, in order, each with its line feed.
    """
    lines = inventory.splitlines(keepends=True)
    return b''.join(line for line in lines if not line.startswith(_PRIMARY_MEMBER))


def _logical_identifier(label: FCPath) -> str:
    """Return the logical identifier a label declares.

    Parameters:
        label: The label.

    Returns:
        The text of its ``Identification_Area/logical_identifier``, stripped, or an
        empty string when it declares none.
    """
    root = ElementTree.fromstring(label.read_bytes())
    path = 'pds:Identification_Area/pds:logical_identifier'
    return root.findtext(path, default='', namespaces=PDS4_NAMESPACES).strip()


def _members_not_held(bundle_label: FCPath, bundle_root: FCPath) -> list[str]:
    """Return each collection a bundle label declares that the bundle holds no label for.

    Parameters:
        bundle_label: The rendered bundle label.
        bundle_root: The bundle's own directory, whose collection directories are
            searched for collection labels.

    Returns:
        The LID of each ``Bundle_Member_Entry`` no ``collection_*.lblx`` one directory
        below the bundle's root declares as its logical identifier, in the order the
        label declares them.
    """
    entries = ElementTree.fromstring(bundle_label.read_bytes()).iterfind(
        'pds:Bundle_Member_Entry', PDS4_NAMESPACES
    )
    declared = [
        entry.findtext('pds:lid_reference', default='', namespaces=PDS4_NAMESPACES).strip()
        for entry in entries
    ]
    held = {_logical_identifier(label) for label in bundle_root.glob('*/collection_*.lblx')}
    return [lid for lid in declared if lid not in held]


def _write_bundle_label(
    template_dir: FCPath,
    bundle_root: FCPath,
    *,
    bundle_name: str,
    epochs: EpochRange | None,
    logger: PdsLogger,
) -> bool:
    """Write the bundle label, but only over a bundle holding every collection it declares.

    The label states the range of the products' epochs, so with no range it is not
    rendered.  Rendered, it is kept only when every ``Bundle_Member_Entry`` names a
    collection whose label is in the bundle, one declaring that LID as its logical
    identifier; otherwise it is removed.  Either way whatever an earlier run left at the
    path is gone, and one error names the label and the reason.

    Parameters:
        template_dir: The dataset's template directory.
        bundle_root: The bundle's own directory.
        bundle_name: The bundle's name, the last part of its LID.
        epochs: The earliest start and the latest stop of the products' exposures, or
            None when the data tree holds no supplemental file.
        logger: Logger for diagnostic messages.

    Returns:
        True if the bundle label is on disk, False if it is not.
    """
    label = bundle_root / _BUNDLE_LABEL
    if epochs is None:
        label.unlink(missing_ok=True)
        logger.error(
            'The bundle label %s was not written: the data tree holds no supplemental '
            'file, so there is no time range for it to state',
            label,
        )
        return False
    template_vars = {'BUNDLE_LID': f'urn:nasa:pds:{bundle_name}'} | epochs.template_variables()
    template = pdstemplate.PdsTemplate((template_dir / _BUNDLE_LABEL).as_posix())
    if not write_label(template, template_vars, label, logger=logger):
        return False
    # The bundle label names every collection the bundle holds, so it is not left
    # naming one that is not there (a collection whose label failed or was never
    # written, or one the template declares before any pass writes it).
    missing = _members_not_held(label, bundle_root)
    if len(missing) > 0:
        label.unlink()
        logger.error(
            'The bundle label %s was not written: it declares %s, and the bundle holds no '
            'collection label declaring that logical identifier',
            label,
            ', '.join(missing),
        )
        return False
    logger.info('Generated "%s": %s', label.name, label)
    return True


def generate_bundle_products(
    bundle_results_root: FCPath,
    dataset: DataSet,
    logger: PdsLogger,
    *,
    epochs: EpochRange | None,
) -> BundleProductsOutcome:
    """Write the readme, the static collections, the user guide and the bundle label.

    Each product goes into the bundle's own directory,
    ``<bundle_results_root>/<pds4_bundle_name()>``, from the dataset's template
    directory:

    - ``readme.txt`` is copied to the bundle's root.
    - The user guide, the PDF
      :meth:`~spindoctor.dataset.dataset.DataSet.pds4_user_guide_file_name` names, is
      copied into ``document/user_guide/`` when the template directory holds it, and
      its label is rendered beside it from the template of the same stem ending in
      ``.lblx``, handed the copy's path as ``USER_GUIDE_PATH``.  When the template
      directory does not hold it, neither is written and one warning names the file.
    - ``kernels.ker`` is copied into ``spice_kernels/``, and ``kernels.lblx`` rendered
      beside it, handed the copy's path as ``METAKERNEL_PATH``.
    - For each of the context, document, SPICE kernel and XML schema collections, the
      inventory ``collection_<name>.csv`` is written into the collection's directory
      and the label ``collection_<name>.lblx`` rendered beside it, handed the
      inventory's path as ``COLLECTION_<NAME>_CSV_PATH``.  Each inventory is the
      template directory's, as it is, except the document inventory when the template
      directory holds no user guide: its primary members are the documents the bundle
      holds, so it is written without its ``P`` lines.
    - ``bundle.lblx`` is rendered last, at the bundle's root, handed ``BUNDLE_LID``,
      ``urn:nasa:pds:<bundle name>``, and the range of the products' epochs as the data
      collection label states it.  With no range it is not rendered.  Rendered, it is
      kept only when every collection it declares in a ``Bundle_Member_Entry`` has a
      label one directory below the bundle's root, ``collection_*.lblx``, declaring
      that LID as its logical identifier; otherwise it is removed.  Either way one error
      names it and the reason, and it counts as a label not written.

    Every label is attempted, whichever of them fail, and a label that fails to render
    is counted; a file copied or an inventory written stays whether or not its label
    renders.  The summary pass clears every path this writes before it reads anything,
    through :func:`bundle_product_paths`, so a product on disk is always one this run
    wrote.

    Parameters:
        bundle_results_root: Root directory of the bundle.
        dataset: The dataset whose template directory, bundle name and user guide the
            products come from.
        logger: Logger for diagnostic messages.
        epochs: The earliest start and the latest stop of the products' exposures, as
            :func:`~spindoctor.cli.pds4.collections.generate_global_index_files` took them,
            or None when the data tree holds no supplemental file.

    Returns:
        The number of run-level labels not written.

    Raises:
        FileNotFoundError: If a template or a file to copy, other than the user guide, is
            not in the dataset's template directory.
    """
    template_dir = FCPath(dataset.pds4_bundle_template_dir())
    bundle_name = dataset.pds4_bundle_name()
    bundle_root = bundle_results_root / bundle_name
    failed_labels = 0

    _copy(template_dir / _README, bundle_root / _README, logger=logger)

    # The user guide is an operator deliverable the template directory may not hold yet;
    # a bundle without it holds no guide and its document inventory lists none.
    user_guide = _user_guide(bundle_root, dataset)
    guide_source = template_dir / user_guide.name
    try:
        guide = guide_source.read_bytes()
    except FileNotFoundError:
        guide = None
        logger.warning(
            'The user guide %s is not in the template directory: the bundle holds no user '
            'guide, and its document inventory lists none',
            guide_source,
        )
    if guide is not None:
        user_guide.write_bytes(guide)
        logger.info('Copied "%s": %s', user_guide.name, user_guide)
        guide_vars = {'USER_GUIDE_PATH': str(user_guide)}
        if not _render(template_dir, _label_beside(user_guide), guide_vars, logger=logger):
            failed_labels += 1

    metakernel = bundle_root / 'spice_kernels' / _METAKERNEL
    _copy(template_dir / _METAKERNEL, metakernel, logger=logger)
    metakernel_vars = {'METAKERNEL_PATH': str(metakernel)}
    if not _render(template_dir, _label_beside(metakernel), metakernel_vars, logger=logger):
        failed_labels += 1

    # Each collection's members are on disk before its inventory lists them, as the data
    # inventory follows the data labels.
    inventories = {
        name: bytes((template_dir / _inventory(bundle_root, name).name).read_bytes())
        for name in _STATIC_COLLECTIONS
    }
    if guide is None:
        inventories['document'] = _without_primary_members(inventories['document'])
    for name, content in inventories.items():
        inventory = _inventory(bundle_root, name)
        inventory.write_bytes(content)
        logger.info('Generated "%s": %s', inventory.name, inventory)
        inventory_vars = {f'COLLECTION_{name.upper()}_CSV_PATH': str(inventory)}
        if not _render(template_dir, _label_beside(inventory), inventory_vars, logger=logger):
            failed_labels += 1

    if not _write_bundle_label(
        template_dir, bundle_root, bundle_name=bundle_name, epochs=epochs, logger=logger
    ):
        failed_labels += 1

    return BundleProductsOutcome(failed_labels=failed_labels)
