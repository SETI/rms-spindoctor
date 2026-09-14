"""The targets a PDS4 bundle's labels name, and the table they are read from.

A label names a target by its PDS4 context product: the product's logical identifier,
with the name and the type the product gives the target, and an inventory lists the
product at its version.  ``backplanes.target_lids`` in the configuration holds one entry
per target, keyed by the name the backplane metadata gives it: a body by the name the
backplane stage looks for it under, and the rings by the ring target their backplanes are
computed for.

An image's targets are every body its backplane metadata names and, when the metadata
holds ring statistics, its ring target.  Every list of targets here is in the table's
order, whatever order the metadata names them in, so that the labels of a bundle list
them alike.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from spindoctor.config import Config

__all__ = ['Pds4Target', 'TargetScan', 'image_targets', 'target_keys', 'target_table']


@dataclass(frozen=True)
class Pds4Target:
    """One target a label names, as its PDS4 context product identifies it.

    Attributes:
        lid: The logical identifier of the target's context product.
        version: The version of that product the PDS registry holds.
        name: The target's name, as the context product gives it.
        target_type: The target's type, as the context product gives it, such as
            ``Planet``, ``Satellite`` or ``Ring``.
    """

    lid: str
    version: str
    name: str
    target_type: str

    @property
    def lidvid(self) -> str:
        """The context product's LIDVID, ``<lid>::<version>``."""
        return f'{self.lid}::{self.version}'


def target_table(config: Config) -> dict[str, Pds4Target]:
    """Return the configured targets, keyed by the name the backplane metadata gives each.

    Parameters:
        config: The configuration whose ``backplanes.target_lids`` is read.

    Returns:
        One target per entry, each from the entry's ``lid``, ``version``, ``name`` and
        ``type``, in the configuration's order.
    """
    return {
        key: Pds4Target(
            lid=entry['lid'],
            version=str(entry['version']),
            name=entry['name'],
            target_type=entry['type'],
        )
        for key, entry in config.backplanes.target_lids.items()
    }


def target_keys(backplane_metadata: Mapping[str, Any]) -> list[str]:
    """Return the names an image's backplane metadata gives the targets it covers.

    Parameters:
        backplane_metadata: The image's backplane metadata, as the backplane stage writes
            it: a ``bodies`` block keyed by body name, and a ``rings`` block naming its
            ring target as ``target`` beside the ring statistics in ``backplanes``.

    Returns:
        Each body the ``bodies`` block names, in the block's order, whether or not the
        body has a statistic, and then the ring target when the ``rings`` block holds at
        least one ring statistic.
    """
    keys = list(backplane_metadata.get('bodies', {}))
    rings = backplane_metadata.get('rings', {})
    if len(rings.get('backplanes', {})) > 0:
        keys.append(rings['target'])
    return keys


def _in_table_order(keys: Iterable[str], table: Mapping[str, Pds4Target]) -> tuple[Pds4Target, ...]:
    """Return the targets of some names, each once, in the table's order.

    Parameters:
        keys: The names, in any order, any of them repeated.
        table: The targets, keyed by name.

    Returns:
        The target of each name, in the order of the table's entries.

    Raises:
        KeyError: If the table has no entry for a name.  The message names every such
            name.
    """
    wanted = set(keys)
    unknown = sorted(wanted - table.keys())
    if len(unknown) > 0:
        raise KeyError(
            f'backplanes.target_lids has no entry for {", ".join(unknown)}, which the '
            'backplane metadata names'
        )
    return tuple(target for key, target in table.items() if key in wanted)


def image_targets(
    backplane_metadata: Mapping[str, Any], table: Mapping[str, Pds4Target]
) -> tuple[Pds4Target, ...]:
    """Return the targets one image's backplane metadata names, as its data label names them.

    Parameters:
        backplane_metadata: The image's backplane metadata.
        table: The targets, keyed by the name the backplane metadata gives each, as
            :func:`target_table` returns them.

    Returns:
        The target of each name :func:`target_keys` returns for the metadata, in the
        table's order.  An image whose metadata names no body and holds no ring statistic
        has none.

    Raises:
        KeyError: If the table has no entry for a name the metadata gives a target.  The
            message names it.
    """
    return _in_table_order(target_keys(backplane_metadata), table)


class TargetScan:
    """The targets over the products one scan takes in, taken as it reads.

    The summary pass reads every supplemental file once, to build the global index, and
    the targets the data collection's members name are taken in that same read: the
    backplane metadata of each image the collection holds is handed to :meth:`include`
    as its file is read.  The result is in the table's order, so the order the files are
    read in cannot change it.
    """

    def __init__(self, table: Mapping[str, Pds4Target]) -> None:
        """Start a scan that has read nothing.

        Parameters:
            table: The targets, keyed by the name the backplane metadata gives each, as
                :func:`target_table` returns them.
        """
        self._table = table
        self._keys: set[str] = set()

    def include(self, backplane_metadata: Mapping[str, Any]) -> None:
        """Take one product's targets into the scan.

        Parameters:
            backplane_metadata: The product's backplane metadata.

        Raises:
            KeyError: If the table has no entry for a name the metadata gives a target.
                The message names it.
        """
        keys = target_keys(backplane_metadata)
        # Looked up as the product is read, so that a name the table has no entry for is
        # refused at the file that names it.
        _in_table_order(keys, self._table)
        self._keys.update(keys)

    def result(self) -> tuple[Pds4Target, ...]:
        """Return every target the products included name, each once.

        Returns:
            The targets, in the table's order, or none when no product was included.
        """
        return _in_table_order(self._keys, self._table)
