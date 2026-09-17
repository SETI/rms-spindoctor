"""The variables every template of a PDS4 bundle resolves against.

Every template the bundle stage renders from the dataset's template directory -- each
label, and the readme and the inventories the directory ships -- is handed these beside
its own variables.  They are what names and versions the bundle as a whole, and the
information model and the dictionary schemas its labels are written against, so each is
set in one place, the dataset's configuration, and read through the dataset's
``pds4_*`` hooks.
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
    - ``INFORMATION_MODEL_VERSION``: the version of the PDS4 information model the labels
      are written against, which each states,
      :meth:`~spindoctor.dataset.dataset.DataSet.pds4_information_model_version`.
    - ``PDS4_<PREFIX>_SCHEMA`` and ``PDS4_<PREFIX>_SCHEMA_XSD``: for each dictionary
      :meth:`~spindoctor.dataset.dataset.DataSet.pds4_schemas` gives, the URL of its
      Schematron and of its XML schema, which a label declaring the dictionary's
      namespace names in its ``xml-model`` instruction and in its ``xsi:schemaLocation``.
      ``<PREFIX>`` is the prefix the namespace takes in a label, in upper case, as in
      ``PDS4_RINGS_SCHEMA_XSD``.
    - ``XML_SCHEMA_LIDVIDS``: the LIDVID of each dictionary's schema product, in the order
      :meth:`~spindoctor.dataset.dataset.DataSet.pds4_schemas` gives them, which the XML
      schema collection's inventory lists.

    Parameters:
        dataset: The dataset whose bundle the templates describe.

    Returns:
        The variables, by name.
    """
    schemas = dataset.pds4_schemas()
    return {
        'BUNDLE_LID': f'urn:nasa:pds:{dataset.pds4_bundle_name()}',
        'BUNDLE_VERSION': dataset.pds4_bundle_version(),
        'INFORMATION_MODEL_VERSION': dataset.pds4_information_model_version(),
        'XML_SCHEMA_LIDVIDS': [schema.lidvid for schema in schemas.values()],
    } | {
        f'PDS4_{prefix.upper()}_SCHEMA{suffix}': url
        for prefix, schema in schemas.items()
        for suffix, url in (('', schema.sch), ('_XSD', schema.xsd))
    }
