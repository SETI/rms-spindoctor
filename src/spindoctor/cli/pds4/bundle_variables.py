"""The variables every template of a PDS4 bundle resolves against.

Every template the bundle stage renders from the dataset's template directory -- each
label, and the readme and the inventories the directory ships -- is handed these beside
its own variables.  They are what names the bundle as a whole, so each is set in one
place, the dataset's configuration, and read through the dataset's ``pds4_*`` hooks.
"""

from typing import Any

from spindoctor.dataset.dataset import DataSet


def bundle_variables(dataset: DataSet) -> dict[str, Any]:
    """Return the variables every template of a dataset's bundle resolves against.

    - ``BUNDLE_LID``: the bundle's LID, ``urn:nasa:pds:<bundle name>``, under the name
      :meth:`~spindoctor.dataset.dataset.DataSet.pds4_bundle_name` gives, which a
      template extends into the LID of each collection and product of the bundle, as in
      ``$BUNDLE_LID$:browse``.
    - ``BUNDLE_VERSION``: the bundle's version,
      :meth:`~spindoctor.dataset.dataset.DataSet.pds4_bundle_version`, which the bundle,
      each of its collections and each product it writes states as its ``version_id``,
      and which every LIDVID naming one of them carries.

    Parameters:
        dataset: The dataset whose bundle the templates describe.

    Returns:
        The variables, by name.
    """
    return {
        'BUNDLE_LID': f'urn:nasa:pds:{dataset.pds4_bundle_name()}',
        'BUNDLE_VERSION': dataset.pds4_bundle_version(),
    }
