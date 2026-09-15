"""The check fetching a schema by its URL, from an HTTP server the test runs on this machine.

No test here reaches the network: the server serves a directory of the test's own, and
the file cache the fetches go through is the test's own too.
"""

import functools
import http.server
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from filecache import FileCache

from spindoctor.cli.pds4.check import schemas
from spindoctor.cli.pds4.check.schemas import SchemaSource, label_schema

from .controls import parse

NAMESPACE = 'urn:example:fetch'
"""The namespace of the schema the server serves."""

SCHEMA = f"""<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema"
    targetNamespace="{NAMESPACE}" elementFormDefault="qualified">
  <xs:element name="a" type="xs:string"/>
</xs:schema>
"""
"""A schema of one element, which imports nothing, so that building it fetches nothing else."""


@dataclass(frozen=True)
class _Server:
    """An HTTP server on this machine, and each path it has been asked for.

    Attributes:
        url: Where it serves from.
        requests: Each path asked for, in order.
    """

    url: str
    requests: list[str]


@pytest.fixture
def server(tmp_path: Path) -> Iterator[_Server]:
    """Serve a directory holding one XML schema over HTTP on this machine, for one test.

    Parameters:
        tmp_path: The test's directory, below which the served directory is made.

    Yields:
        The server.
    """
    served = tmp_path / 'served'
    served.mkdir()
    (served / 'example.xsd').write_text(SCHEMA, encoding='utf-8')
    requests: list[str] = []

    class _Handler(http.server.SimpleHTTPRequestHandler):
        """A handler that records each path it is asked for."""

        def do_GET(self) -> None:
            """Record the path, and serve it."""
            requests.append(self.path)
            super().do_GET()

    httpd = http.server.ThreadingHTTPServer(
        ('127.0.0.1', 0), functools.partial(_Handler, directory=str(served))
    )
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield _Server(url=f'http://127.0.0.1:{httpd.server_address[1]}', requests=requests)
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join()


@pytest.fixture
def own_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Send the check's fetches through a file cache of the test's own.

    Parameters:
        tmp_path: The test's directory, below which the cache is made.
        monkeypatch: Fixture the cache is installed through.
    """
    root = tmp_path / 'cache'
    root.mkdir()
    cache = FileCache('schemas', cache_root=root)
    monkeypatch.setattr(schemas, 'schema_cache', lambda: cache)


def _label(tmp_path: Path, url: str) -> Any:
    """Write a label declaring the schema at a URL, and parse it.

    Parameters:
        tmp_path: The test's directory.
        url: The schema's URL.

    Returns:
        The label, parsed.
    """
    label = tmp_path / 'label.lblx'
    label.write_text(
        f'<a xmlns="{NAMESPACE}" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
        f'xsi:schemaLocation="{NAMESPACE} {url}">text</a>\n',
        encoding='utf-8',
    )
    return parse(label)


def test_a_schema_is_fetched_once_and_then_read_from_the_cache(
    server: _Server, own_cache: None, tmp_path: Path
) -> None:
    """A schema is fetched for the first check, and a later check finds it in the cache."""
    url = f'{server.url}/example.xsd'
    label_schema('label.lblx', _label(tmp_path, url), SchemaSource())
    SchemaSource().locate(url)
    assert server.requests == ['/example.xsd']


def test_a_label_is_held_to_a_schema_it_fetched(
    server: _Server, own_cache: None, tmp_path: Path
) -> None:
    """A fetched schema builds, and the label that declares it is held to it."""
    url = f'{server.url}/example.xsd'
    resolved = label_schema('label.lblx', _label(tmp_path, url), SchemaSource())
    assert resolved.schema is not None


def test_a_schema_that_cannot_be_fetched_is_one_finding_that_names_it(
    server: _Server, own_cache: None, tmp_path: Path
) -> None:
    """A URL the server has no file for is one finding naming it, and nothing is built."""
    url = f'{server.url}/missing.xsd'
    resolved = label_schema('label.lblx', _label(tmp_path, url), SchemaSource())
    messages = [finding.message.split(' (', 1)[0] for finding in resolved.findings]
    assert messages == [f'declares the XML schema {url}, but it cannot be fetched']
