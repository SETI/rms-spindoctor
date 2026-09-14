"""The PDS4 schemas the package ships, and each label checked against the XML schemas it declares.

The package ships the XML schema and the Schematron of each dictionary a bundle's labels
declare, in :data:`SCHEMA_DIRECTORY`, and the bundle check resolves every schema from
there: nothing is ever fetched.  A label declares its XML schemas by the
``xsi:schemaLocation`` attribute of its root element and its Schematron by ``xml-model``
instructions, each by URL, and each URL is mapped to the shipped copy of the same file
name.  A URL with no shipped copy is a finding that names it.

A dictionary's XML schema imports the schemas of the dictionaries it builds on, each by
the namespace it defines and a URL.  Those imports are resolved through the shipped
directory as a catalog: each shipped XML schema is offered for the namespace it defines,
so that an import resolves to the shipped schema of its namespace, whichever version its
URL names.  Every warning raised while a set of schemas is built is a finding, so that an
import that fails cannot pass silently.
"""

import functools
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import xmlschema
from lxml import etree

from spindoctor.cli.pds4.check.elements import element_path
from spindoctor.cli.pds4.check.findings import CheckName, Finding

SCHEMA_DIRECTORY = Path(__file__).resolve().parent.parent / 'schemas'
"""The directory the package ships the PDS4 schemas in, as the PDS publishes them."""

XSI_SCHEMA_LOCATION = '{http://www.w3.org/2001/XMLSchema-instance}schemaLocation'
"""The attribute a label declares its XML schemas by, each namespace followed by a URL."""


def shipped_copy(url: str) -> Path | None:
    """Return the shipped copy of a schema a label names by URL.

    Parameters:
        url: The URL of an XML schema or a Schematron.

    Returns:
        The file in :data:`SCHEMA_DIRECTORY` with the URL's file name, or None when the
        package ships no file of that name.
    """
    name = url.rsplit('/', 1)[-1]
    path = SCHEMA_DIRECTORY / name
    if name == '' or not path.is_file():
        return None
    return path


@functools.cache
def _catalog(directory: Path) -> Mapping[str, str]:
    """Return each XML schema in a directory by the namespace it defines.

    Parameters:
        directory: The directory of schemas.

    Returns:
        Each schema's path, keyed by its ``targetNamespace``.
    """
    return MappingProxyType(
        {
            str(etree.parse(str(path)).getroot().get('targetNamespace')): str(path)
            for path in sorted(directory.glob('*.xsd'))
        }
    )


@functools.cache
def _loader_class(directory: Path) -> type[xmlschema.SchemaLoader]:
    """Return an xmlschema loader offering each schema in a directory for its namespace.

    xmlschema tries an import's own URL first and then the locations its loader offers
    for the import's namespace, so the schemas here are used for a namespace only when an
    import asks for it.

    Parameters:
        directory: The directory of schemas.

    Returns:
        The loader class.
    """
    offered: dict[str, Any] = {**xmlschema.SchemaLoader.fallback_locations, **_catalog(directory)}

    class ShippedSchemaLoader(xmlschema.SchemaLoader):
        """xmlschema's loader, offering each shipped XML schema for its own namespace."""

        fallback_locations = MappingProxyType(offered)

    return ShippedSchemaLoader


@dataclass(frozen=True)
class _SchemaSet:
    """A set of XML schemas built from shipped copies, and what building it warned of.

    Attributes:
        schema: The set, built.
        warnings: The text of each warning raised while it was built.
    """

    schema: xmlschema.XMLSchema
    warnings: tuple[str, ...]


@functools.cache
def _schema_set(directory: Path, main: str, others: tuple[tuple[str, str], ...]) -> _SchemaSet:
    """Build the set of XML schemas a label declares, from their shipped copies, once.

    Parameters:
        directory: The directory the imports are resolved through.
        main: The shipped copy of the schema of the label's root element's namespace.
        others: Each other namespace the label declares, with its schema's shipped copy.

    Returns:
        The set, and every warning raised while it was built.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        schema = xmlschema.XMLSchema(
            main,
            locations=list(others),
            allow='local',
            loader_class=_loader_class(directory),
        )
    return _SchemaSet(schema=schema, warnings=tuple(str(warning.message) for warning in caught))


@dataclass(frozen=True)
class LabelSchema:
    """The XML schemas one label declares, built from their shipped copies.

    Attributes:
        schema: The set built from the shipped copy of every XML schema the label
            declares, and of every schema those import, or None when a schema it declares
            has no shipped copy or it declares none for its root element's namespace.
        findings: What resolving the declaration found: each URL with no shipped copy, a
            root element whose namespace the label declares no XML schema for, and each
            warning raised while the set was built.
    """

    schema: xmlschema.XMLSchema | None
    findings: tuple[Finding, ...]


def label_schema(file: str, document: Any) -> LabelSchema:
    """Resolve the XML schemas a label declares to their shipped copies, and build them.

    The set is built once for each distinct declaration, and every label declaring the
    same schemas is checked against the same set.

    Parameters:
        file: The label's path relative to the bundle's directory, which the findings name.
        document: The label, parsed by lxml.

    Returns:
        The set, and what resolving the declaration found.
    """
    root = document.getroot()
    words = str(root.get(XSI_SCHEMA_LOCATION, '')).split()
    located: dict[str, str] = {}
    findings: list[Finding] = []
    for namespace, url in zip(words[0::2], words[1::2], strict=False):
        copy = shipped_copy(url)
        if copy is None:
            findings.append(
                Finding(
                    file,
                    CheckName.XSD,
                    '',
                    f'declares the XML schema {url}, of which the package ships no copy',
                )
            )
        else:
            located[namespace] = str(copy)
    if len(findings) > 0:
        return LabelSchema(schema=None, findings=tuple(findings))
    namespace = etree.QName(root).namespace
    if namespace not in located:
        finding = Finding(
            file,
            CheckName.XSD,
            '',
            f'declares no XML schema for the namespace of its root element, {namespace}',
        )
        return LabelSchema(schema=None, findings=(finding,))
    others = tuple((other, path) for other, path in located.items() if other != namespace)
    schema_set = _schema_set(SCHEMA_DIRECTORY, located[namespace], others)
    return LabelSchema(
        schema=schema_set.schema,
        findings=tuple(
            Finding(file, CheckName.XSD, '', f'building its XML schemas warned: {warning}')
            for warning in schema_set.warnings
        ),
    )


def xsd_findings(file: str, document: Any, schema: xmlschema.XMLSchema) -> list[Finding]:
    """Validate a label against the XML schemas it declares.

    Parameters:
        file: The label's path relative to the bundle's directory, which the findings name.
        document: The label, parsed by lxml.
        schema: The set :func:`label_schema` built for it.

    Returns:
        One finding for each error, at the element it is about, with the line the label
        holds that element on.  An error xmlschema reports twice over one element is one
        finding.
    """
    found: list[Finding] = []
    for error in schema.iter_errors(document, use_location_hints=False):
        element = error.elem
        if isinstance(element, etree._Element):
            location = element_path(element)
        else:
            location = str(error.path or '')
        reason = error.reason if error.reason is not None else error.message
        line = error.sourceline
        message = reason if line is None else f'{reason} (line {line})'
        found.append(Finding(file, CheckName.XSD, location, message))
    return list(dict.fromkeys(found))
