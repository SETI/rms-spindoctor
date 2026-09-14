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

import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import pdstemplate
from filecache import FCPath
from pdslogger import PdsLogger

from spindoctor.cli.pds4.epochs import EpochRange
from spindoctor.cli.pds4.labels import write_label
from spindoctor.dataset.dataset import DataSet, pds4_label_name

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

_PRIMARY_MEMBER = b'P,'
"""How an inventory line naming a primary member begins."""

_SECONDARY_MEMBER = b'S,'
"""How an inventory line naming a secondary member begins."""


def _inventory_name(collection: str) -> str:
    """Return the name of a collection's inventory, in the template directory and a bundle.

    Parameters:
        collection: The collection's name.

    Returns:
        ``collection_<collection>.csv``.
    """
    return f'collection_{collection}.csv'


def _inventory(bundle_root: FCPath, collection: str) -> FCPath:
    """Return where a static collection's inventory goes in a bundle.

    Parameters:
        bundle_root: The bundle's own directory.
        collection: The collection's name, which is its directory's.

    Returns:
        ``<bundle_root>/<collection>/collection_<collection>.csv``.
    """
    return bundle_root / collection / _inventory_name(collection)


def secondary_members(template_dir: FCPath) -> list[str]:
    """Return the secondary members the bundle's collections cite, as LIDVIDs.

    The document inventory the template directory ships lists them beside the user
    guide: the external documents and the context products the bundle cites, each at
    its version.  The miscellaneous collection cites the same ones and takes them from
    here, so that the two inventories cannot disagree about them.

    Parameters:
        template_dir: The dataset's template directory.

    Returns:
        The LIDVID of each line of the template directory's ``collection_document.csv``
        that names a secondary member, beginning ``S,``, in the order it lists them.

    Raises:
        FileNotFoundError: If the template directory holds no document inventory.
    """
    inventory = (template_dir / _inventory_name('document')).read_bytes()
    return [
        line.removeprefix(_SECONDARY_MEMBER).decode('ascii')
        for line in inventory.splitlines()
        if line.startswith(_SECONDARY_MEMBER)
    ]


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
        The path in the same directory named by
        :func:`~spindoctor.dataset.dataset.pds4_label_name`.
    """
    return path.parent / pds4_label_name(path.name)


def _bundle_product_paths(bundle_root: FCPath, dataset: DataSet) -> tuple[FCPath, ...]:
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


def clear_bundle_products(bundle_root: str | Path | FCPath, dataset: DataSet) -> None:
    """Remove every file :func:`generate_bundle_products` can write into one bundle.

    The global index generator, which the summary pass runs first, calls this once it
    has found the bundle's data directory and before it reads any supplemental file, so
    that a run-level product on disk after a summary pass is one that pass wrote.
    When the user guide's directory, in a bundle on the local file system, is left holding
    nothing, it is removed as well, so a run over a template directory without the guide
    leaves no ``document/user_guide/``.  A remote store holds no directory apart from the
    files in it, so there is none to remove there.

    Parameters:
        bundle_root: The bundle's own directory, a local path or a URL.
        dataset: The dataset whose user guide the bundle can hold.
    """
    bundle_root = FCPath(bundle_root)
    for path in _bundle_product_paths(bundle_root, dataset):
        path.unlink(missing_ok=True)
    user_guide_dir = _user_guide(bundle_root, dataset).parent
    if not user_guide_dir.is_local():
        # FCPath refuses rmdir on a remote path, which has no directory to remove.
        return
    # glob rather than iterdir, which refuses a bundle root given relative to the working
    # directory; a bundle that never held the guide has no such directory to remove.
    with contextlib.suppress(FileNotFoundError):
        if next(iter(user_guide_dir.glob('*')), None) is None:
            user_guide_dir.rmdir()


@dataclass(frozen=True)
class BundleProductsOutcome:
    """What generating the run-level products came to.

    Attributes:
        failed_labels: The number of run-level labels not written: each that could not be
            rendered; each static collection left with no member, whose inventory and
            label are not written; and the bundle label when there is no time range for
            it to state or the bundle holds no label for a collection it declares.
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
    template_vars = {
        'BUNDLE_LID': f'urn:nasa:pds:{bundle_name}',
        'README_PATH': (bundle_root / _README).as_posix(),
    } | epochs.template_variables()
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
    bundle_results_root: str | Path | FCPath,
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
      template directory's, as it is, except that its primary members, the products of
      this bundle it lists as ``P`` -- the user guide in the document collection, the
      metakernel in the SPICE kernel collection -- are listed only when their labels
      are in the bundle.  A collection left with no member is not written at all,
      neither its inventory nor its label, whatever is at either path is removed, and
      it counts once as a label not written, with an error naming it: the SPICE kernel
      collection, whose one member is the metakernel, is not written when the
      metakernel's label is not.
    - ``bundle.lblx`` is rendered last, at the bundle's root, handed ``BUNDLE_LID``,
      ``urn:nasa:pds:<bundle name>``, ``README_PATH``, and the range of the products'
      epochs as the data collection label states it.  With no range it is not
      rendered.  Rendered, it is kept only when every collection it declares in a
      ``Bundle_Member_Entry`` has a label one directory below the bundle's root,
      ``collection_*.lblx``, declaring that LID as its logical identifier; otherwise it
      is removed.  Either way one error names it and the reason, and it counts as a
      label not written.

    Every label is attempted, whichever of them fail, and a label that fails to render
    is counted; a file copied stays whether or not its label renders.  What this removes
    itself is each label just before it renders it, as ``write_label`` does for every
    label, and what is said above of a collection left with no member and of the bundle
    label.  The summary pass clears the rest of an earlier run's products through
    :func:`clear_bundle_products`, which the global index generator, the first the pass
    runs, calls once it has found the bundle's data directory; so called on its own over
    an earlier run's bundle this leaves whatever of that run's products it neither
    writes nor removes, such as a user guide the template directory no longer holds.

    Parameters:
        bundle_results_root: Root directory of the bundle, a local path or a URL.
        dataset: The dataset whose template directory, bundle name and user guide the
            products come from.
        logger: Logger for diagnostic messages.
        epochs: The earliest start and the latest stop of the products' exposures, as
            :func:`~spindoctor.cli.pds4.global_index.generate_global_index_files` took them,
            or None when the data tree holds no supplemental file.

    Returns:
        The number of run-level labels not written.

    Raises:
        FileNotFoundError: If a template or a file to copy, other than the user guide, is
            not in the dataset's template directory.
    """
    bundle_results_root = FCPath(bundle_results_root)
    template_dir = FCPath(dataset.pds4_bundle_template_dir())
    bundle_name = dataset.pds4_bundle_name()
    bundle_root = bundle_results_root / bundle_name
    failed_labels = 0

    _copy(template_dir / _README, bundle_root / _README, logger=logger)

    # The user guide is an operator deliverable the template directory may not hold yet;
    # a bundle without it holds no guide and its document inventory lists none.
    user_guide = _user_guide(bundle_root, dataset)
    guide_source = template_dir / user_guide.name
    guide_labeled = False
    try:
        guide = guide_source.read_bytes()
    except FileNotFoundError:
        logger.warning(
            'The user guide %s is not in the template directory: the bundle holds no user '
            'guide, and its document inventory lists none',
            guide_source,
        )
    else:
        user_guide.write_bytes(guide)
        logger.info('Copied "%s": %s', user_guide.name, user_guide)
        guide_vars = {'USER_GUIDE_PATH': user_guide.as_posix()}
        guide_labeled = _render(template_dir, _label_beside(user_guide), guide_vars, logger=logger)
        if not guide_labeled:
            failed_labels += 1

    metakernel = bundle_root / 'spice_kernels' / _METAKERNEL
    _copy(template_dir / _METAKERNEL, metakernel, logger=logger)
    metakernel_vars = {'METAKERNEL_PATH': metakernel.as_posix()}
    metakernel_labeled = _render(
        template_dir, _label_beside(metakernel), metakernel_vars, logger=logger
    )
    if not metakernel_labeled:
        failed_labels += 1

    # Each collection's members are on disk before its inventory lists them, as the data
    # inventory follows the data labels.  A static inventory's primary members are
    # products of this bundle, the document collection's the user guide and the SPICE
    # kernel collection's the metakernel, and it lists one only when that product's label
    # is in the bundle; the context and XML schema inventories list no primary member.
    keeps_primaries = {
        'context': True,
        'document': guide_labeled,
        'spice_kernels': metakernel_labeled,
        'xml_schema': True,
    }
    for name in _STATIC_COLLECTIONS:
        inventory = _inventory(bundle_root, name)
        label = _label_beside(inventory)
        content = bytes((template_dir / inventory.name).read_bytes())
        if not keeps_primaries[name]:
            content = _without_primary_members(content)
        if len(content) == 0:
            # As for the data and browse collections, a collection with no member gets
            # neither an inventory nor a label: the label would state no record, which
            # the schema refuses.
            inventory.unlink(missing_ok=True)
            label.unlink(missing_ok=True)
            logger.error(
                'The %s collection was not written, neither its inventory %s nor its '
                'label %s: none of its members has a label in the bundle, and a '
                'collection label has to state at least one record',
                name,
                inventory,
                label,
            )
            failed_labels += 1
            continue
        inventory.write_bytes(content)
        logger.info('Generated "%s": %s', inventory.name, inventory)
        inventory_vars = {f'COLLECTION_{name.upper()}_CSV_PATH': inventory.as_posix()}
        if not _render(template_dir, label, inventory_vars, logger=logger):
            failed_labels += 1

    if not _write_bundle_label(
        template_dir, bundle_root, bundle_name=bundle_name, epochs=epochs, logger=logger
    ):
        failed_labels += 1

    return BundleProductsOutcome(failed_labels=failed_labels)
