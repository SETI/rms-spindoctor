"""Each label held to the XML schemas it names, fetched by URL or read from a directory.

A label declares its XML schemas by the ``xsi:schemaLocation`` attribute of its root
element and its Schematron by ``xml-model`` instructions, each by URL, and a dictionary's
XML schema imports the schemas of the dictionaries it builds on, each by a URL of its
own.  Every one of those URLs is resolved one way, by :class:`SchemaSource`:

- by default it is fetched, through a file cache that keeps each download, so that a
  later check fetches nothing it already has.  The cache is the directory
  ``_filecache_spindoctor_pds4_schemas`` under ``$FILECACHE_CACHE_ROOT`` when that is
  set, and otherwise under the user's own cache directory, ``$XDG_CACHE_HOME`` or
  ``~/.cache`` (:func:`schema_cache_root`);
- given a directory, it is the file of the URL's name in that directory, and nothing is
  fetched.

No namespace is named in code: an import resolves to the schema at its own URL, as the
PDS ``validate`` tool resolves it.  xmlschema reads each namespace once in a set, though:
where a label itself declares one build of a dictionary and another schema it declares
imports a second build of it, the label's build serves the import and the second is not
read, where ``validate`` reads both.  xmlschema is allowed only local files, so
everything it reads has come through the one rule, as an absolute path.

A URL a label names that cannot be resolved -- one that cannot be fetched, or with no file
of its name in the directory -- is a finding naming it, and the label is not held to its
XML schemas.  An import that cannot be resolved makes xmlschema warn, naming the import's
URL, and each of xmlschema's own warnings raised while a set of schemas is built is a
finding; any other warning raised meanwhile is left to the caller's filters, untouched.  A
set that cannot be built at all -- a URL paired with a namespace its file does not define,
say -- is one finding, and the label is checked without it.
"""

import functools
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

import xmlschema
from filecache import FCPath, FileCache
from lxml import etree
from xmlschema.exceptions import XMLSchemaWarning

from spindoctor.cli.pds4.check.elements import element_path
from spindoctor.cli.pds4.check.findings import CheckName, Finding

SCHEMA_CACHE_NAME = 'spindoctor_pds4_schemas'
"""The name of the file cache fetched schemas are kept in, under the file cache's root."""

XSI_SCHEMA_LOCATION = '{http://www.w3.org/2001/XMLSchema-instance}schemaLocation'
"""The attribute a label declares its XML schemas by, each namespace followed by a URL."""


SCHEMA_CACHE_ROOTS = ('FILECACHE_CACHE_ROOT', 'XDG_CACHE_HOME')
"""The variables that name the schema cache's root, the first set winning."""


def schema_cache_root() -> Path:
    """Return the directory the schema cache is made in.

    Returns:
        ``$FILECACHE_CACHE_ROOT`` when it is set, as for every file cache; otherwise the
        user's own cache directory, ``$XDG_CACHE_HOME`` when that is set and ``~/.cache``
        when it is not, where no other user of the machine can leave a schema the check
        would then read.
    """
    for variable in SCHEMA_CACHE_ROOTS:
        value = os.environ.get(variable, '')
        if value != '':
            return Path(value)
    return Path.home() / '.cache'


@functools.cache
def schema_cache() -> FileCache:
    """Return the file cache fetched schemas are kept in, made when it is first needed.

    Returns:
        The cache named :data:`SCHEMA_CACHE_NAME` under :func:`schema_cache_root`, which
        outlives the process.
    """
    return FileCache(SCHEMA_CACHE_NAME, cache_root=schema_cache_root())


def _fetch(url: str) -> Path:
    """Fetch a schema through the schema cache, or find it there already.

    Parameters:
        url: The schema's URL.

    Returns:
        The local file the cache holds the schema in.

    Raises:
        FileNotFoundError: If the URL cannot be fetched, or the cache cannot hold it.
    """
    try:
        local = FCPath(url, filecache=schema_cache()).retrieve()
    except OSError as exc:
        # FileNotFoundError when the URL cannot be fetched, and PermissionError when the
        # cache cannot be written: either is one finding for the URL, and the check goes on.
        raise FileNotFoundError(f'it cannot be fetched ({exc})') from exc
    if not isinstance(local, Path):
        raise FileNotFoundError(f'it cannot be fetched ({local})')
    return local


@dataclass(frozen=True)
class SchemaSource:
    """Where the check takes the schemas a label names by URL from.

    Attributes:
        directory: A directory holding each schema under the name its URL ends in, from
            which every URL is read and nothing is fetched; or None, when every URL is
            fetched through :func:`schema_cache`.
    """

    directory: Path | None = None

    def locate(self, url: str) -> Path:
        """Return the local file a schema's URL resolves to.

        Parameters:
            url: The URL of an XML schema or a Schematron.

        Returns:
            The file of the URL's name in :attr:`directory`, as an absolute path whether
            or not the directory is given as one, or, with no directory, the file the
            schema cache holds the URL's download in.

        Raises:
            FileNotFoundError: If the directory holds no file of the URL's name, or the
                URL cannot be fetched; the message says which.
        """
        if self.directory is None:
            return _fetch(url)
        name = url.rsplit('/', 1)[-1]
        # Absolute, since xmlschema reads a relative path against the schema importing it.
        path = (self.directory / name).resolve()
        if name == '' or not path.is_file():
            raise FileNotFoundError(f'no file of its name is in {self.directory}')
        return path

    def mapped(self, url: str) -> str:
        """Return the local file xmlschema is to read for a URL it is about to read.

        Parameters:
            url: The URL, a label's or one a schema imports.

        Returns:
            The local file the URL resolves to, or the URL itself when it cannot be
            resolved, which xmlschema, allowed only local files, then refuses with a
            warning naming it.
        """
        try:
            return str(self.locate(url))
        except FileNotFoundError:
            return url


@dataclass(frozen=True)
class _SchemaSet:
    """A set of XML schemas built from where a source takes them, and what came of it.

    Attributes:
        schema: The set, built, or None when it could not be.
        warnings: The text of each of xmlschema's warnings raised while it was built.
        error: Why it could not be built, or None when it was.
    """

    schema: xmlschema.XMLSchema | None
    warnings: tuple[str, ...]
    error: str | None


@functools.cache
def _schema_set(source: SchemaSource, main: str, others: tuple[tuple[str, str], ...]) -> _SchemaSet:
    """Build the set of XML schemas a label declares, from where a source takes them, once.

    Every URL xmlschema reads, the label's and each one a schema imports, goes through
    :meth:`SchemaSource.mapped`.

    Only xmlschema's own warnings are the schemas' concern.  Any other warning raised
    while the set is built -- a ``ResourceWarning`` the garbage collector raises for
    something else in the process, say -- meets the caller's filters, and is shown,
    raised or ignored as if nothing here were listening.

    Parameters:
        source: Where the schemas are taken from.
        main: The URL of the schema of the label's root element's namespace.
        others: Each other namespace the label declares, with its schema's URL.

    Returns:
        The set, and each of xmlschema's warnings raised while it was built; or, when it
        cannot be built, the first line of the reason xmlschema gives.
    """
    schema: xmlschema.XMLSchema | None = None
    error: str | None = None
    found: list[str] = []
    with warnings.catch_warnings():
        shown = warnings.showwarning

        def show(
            message: Warning | str,
            category: type[Warning],
            filename: str,
            lineno: int,
            file: TextIO | None = None,
            line: str | None = None,
        ) -> None:
            """Keep a warning of xmlschema's, and show any other as it would have been.

            Parameters:
                message: The warning.
                category: Its class.
                filename: The file it was raised from.
                lineno: The line it was raised from.
                file: Where it is to be written.
                line: The text of that line.
            """
            if issubclass(category, XMLSchemaWarning):
                found.append(str(message))
            else:
                shown(message, category, filename, lineno, file, line)

        # xmlschema's warnings are kept whatever the caller's filters say, one for each;
        # every other warning still meets those filters, so one they raise is raised.
        warnings.filterwarnings('always', category=XMLSchemaWarning)
        warnings.showwarning = show
        try:
            schema = xmlschema.XMLSchema(
                main,
                locations=list(others),
                allow='local',
                uri_mapper=source.mapped,
            )
        except xmlschema.XMLSchemaException as exc:
            # A set that cannot be built is one finding for the label, which is then
            # checked without it, rather than a traceback that ends the whole check.
            reason = getattr(exc, 'message', None) or str(exc)
            error = str(reason).splitlines()[0]
    return _SchemaSet(schema=schema, warnings=tuple(found), error=error)


@dataclass(frozen=True)
class LabelSchema:
    """The XML schemas one label declares, built from where a source takes them.

    Attributes:
        schema: The set built from every XML schema the label declares, and every schema
            those import, or None when a URL it declares cannot be resolved, it declares
            no schema for its root element's namespace, or the set cannot be built.
        findings: What resolving the declaration found: each URL that cannot be
            resolved, a root element whose namespace the label declares no XML schema
            for, each of xmlschema's warnings raised while the set was built, and why
            it cannot be built.
    """

    schema: xmlschema.XMLSchema | None
    findings: tuple[Finding, ...]


def label_schema(file: str, document: Any, source: SchemaSource) -> LabelSchema:
    """Resolve the XML schemas a label declares, and build them.

    The set is built once for each distinct declaration and source, and every label
    declaring the same schemas is checked against the same set.

    Parameters:
        file: The label's path relative to the bundle's directory, which findings name.
        document: The label, parsed by lxml.
        source: Where the schemas are taken from.

    Returns:
        The set, and what resolving the declaration found.
    """
    root = document.getroot()
    words = str(root.get(XSI_SCHEMA_LOCATION, '')).split()
    located: dict[str, str] = {}
    findings: list[Finding] = []
    for namespace, url in zip(words[0::2], words[1::2], strict=False):
        try:
            source.locate(url)
        except FileNotFoundError as exc:
            message = f'declares the XML schema {url}, but {exc}'
            findings.append(Finding(file, CheckName.XSD, '', message))
        else:
            located[namespace] = url
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
    others = tuple((other, url) for other, url in located.items() if other != namespace)
    schema_set = _schema_set(source, located[namespace], others)
    findings.extend(
        Finding(file, CheckName.XSD, '', f'building its XML schemas warned: {warning}')
        for warning in schema_set.warnings
    )
    if schema_set.error is not None:
        findings.append(
            Finding(
                file,
                CheckName.XSD,
                '',
                'its XML schemas cannot be built, so it is not validated against them: '
                f'{schema_set.error}',
            )
        )
    return LabelSchema(schema=schema_set.schema, findings=tuple(findings))


def xsd_findings(file: str, document: Any, schema: xmlschema.XMLSchema) -> list[Finding]:
    """Validate a label against the XML schemas it declares.

    Parameters:
        file: The label's path relative to the bundle's directory, which findings name.
        document: The label, parsed by lxml.
        schema: The set :func:`label_schema` built for it.

    Returns:
        One finding for each error, at the element it is about, with the line the label
        holds that element on, and with its kind: the class of the xmlschema validator
        that failed, which a release rewording xmlschema's messages leaves as it is.  An
        error xmlschema reports twice over one element is one finding.
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
        validator = error.validator
        kind = type(error).__name__ if validator is None else type(validator).__name__
        found.append(Finding(file, CheckName.XSD, location, message, kind=kind))
    return list(dict.fromkeys(found))
