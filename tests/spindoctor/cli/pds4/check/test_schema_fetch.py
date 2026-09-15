"""The check fetching a schema by its URL, from an HTTP server the test runs on this machine.

No test here reaches the network: the server serves a directory of the test's own, and
the file cache the fetches go through is the test's own too.
"""

import argparse
import functools
import http.server
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from filecache import FileCache
from tests.spindoctor.cli.sd_create_bundle_helpers import BUNDLE_NAME, stub_dataset

from spindoctor.cli import sd_create_bundle
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

SECOND_NAMESPACE = 'urn:example:second'
"""The namespace of the second schema the server serves."""


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
    second = SCHEMA.replace(NAMESPACE, SECOND_NAMESPACE)
    (served / 'second.xsd').write_text(second, encoding='utf-8')
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


def test_a_cache_that_cannot_be_written_is_a_finding_for_each_url_and_no_traceback(
    server: _Server,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """With a read-only cache, each URL a label names is one finding, and the run exits 1."""
    root = tmp_path / 'cache'
    root.mkdir()
    cache = FileCache('schemas', cache_root=root)
    monkeypatch.setattr(schemas, 'schema_cache', lambda: cache)
    urls = [f'{server.url}/example.xsd', f'{server.url}/second.xsd']
    bundle_dir = tmp_path / BUNDLE_NAME
    bundle_dir.mkdir()
    (bundle_dir / 'label.lblx').write_text(
        f'<a xmlns="{NAMESPACE}" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
        f'xsi:schemaLocation="{NAMESPACE} {urls[0]} {SECOND_NAMESPACE} {urls[1]}">text</a>\n',
        encoding='utf-8',
    )
    monkeypatch.setattr(
        sd_create_bundle,
        'parse_args_check',
        lambda _: argparse.Namespace(dataset_name='stub', schema_dir=None),
    )
    monkeypatch.setattr(sd_create_bundle, 'load_default_and_user_config', lambda *a: None)
    monkeypatch.setattr(sd_create_bundle, 'get_pds4_bundle_results_root', lambda *a: str(tmp_path))
    dataset = stub_dataset(tmp_path)
    monkeypatch.setattr(sd_create_bundle, 'dataset_name_to_class', lambda _: lambda: dataset)
    locked = [root, *root.iterdir()]
    for directory in locked:
        directory.chmod(0o500)
    try:
        with pytest.raises(SystemExit) as excinfo:
            sd_create_bundle.main_check()
    finally:
        for directory in locked:
            directory.chmod(0o700)
    lines = capsys.readouterr().out.splitlines()
    fetched = [line.split(' (', 1)[0] for line in lines if 'cannot be fetched' in line]
    expected = [
        f'label.lblx: error [xsd] declares the XML schema {url}, but it cannot be fetched'
        for url in urls
    ]
    assert (excinfo.value.code, fetched, 'Traceback' in '\n'.join(lines)) == (1, expected, False)


@pytest.mark.parametrize(
    ('variables', 'expected'),
    [
        pytest.param({}, 'home/.cache', id='home'),
        pytest.param({'XDG_CACHE_HOME': 'xdg'}, 'xdg', id='xdg'),
        pytest.param(
            {'XDG_CACHE_HOME': 'xdg', 'FILECACHE_CACHE_ROOT': 'named'}, 'named', id='named'
        ),
    ],
)
def test_the_schema_cache_is_the_users_own_unless_a_root_is_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, variables: dict[str, str], expected: str
) -> None:
    """The cache's root is the user's cache directory, or the file cache's root if named.

    Parameters:
        tmp_path: The directory the home, the XDG cache and the named root are in.
        monkeypatch: Fixture the environment is set through.
        variables: Each variable set, with the directory it names below ``tmp_path``.
        expected: The root the cache is made in, below ``tmp_path``.
    """
    monkeypatch.setenv('HOME', str(tmp_path / 'home'))
    for variable in schemas.SCHEMA_CACHE_ROOTS:
        monkeypatch.delenv(variable, raising=False)
    for variable, directory in variables.items():
        monkeypatch.setenv(variable, str(tmp_path / directory))
    assert schemas.schema_cache_root() == tmp_path / expected
