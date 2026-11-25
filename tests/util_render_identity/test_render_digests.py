"""Tests for the render digest check.

The check lives in ``util/render_identity/render_digests.py``, outside the importable
package, and is loaded from its path.
"""

import dataclasses
import hashlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / 'util' / 'render_identity' / 'render_digests.py'
)


def _load_script() -> ModuleType:
    """Load ``util/render_identity/render_digests.py`` as a module."""
    spec = importlib.util.spec_from_file_location('render_digests', _SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


render_digests = _load_script()


def _digest(value: Any) -> str:
    """The hex digest of one value as the check feeds it."""
    digest = hashlib.sha256()
    render_digests._feed(digest, value)
    return digest.hexdigest()


def test_integer_tuples_with_the_same_digits_digest_apart() -> None:
    """Two tuples whose digits run together the same way have different digests."""
    assert _digest((3, 131)) != _digest((31, 31))


@dataclasses.dataclass
class _Point:
    """A dataclass to nest."""

    x: int


@dataclasses.dataclass
class _Wrapper:
    """A dataclass holding one value."""

    inner: Any


def test_a_nested_dataclass_digests_apart_from_a_dict() -> None:
    """A dataclass inside a dataclass keeps its type, so it differs from its dict."""
    assert _digest(_Wrapper(_Point(1))) != _digest(_Wrapper({'x': 1}))


def test_a_missing_selected_render_fails_the_comparison() -> None:
    """A base render the run selected but did not make gives status 1."""
    base = {'catalog/a#nav': ['1', '2'], 'catalog/b#nav': ['3', '4']}
    results = {'catalog/a#nav': ['1', '2']}
    assert render_digests._compare(base, results, only='catalog/') == 1


def test_a_prefix_limited_run_passes_without_the_other_renders() -> None:
    """Base renders outside the run's prefix are not expected in it."""
    base = {'catalog/a#nav': ['1', '2'], 'sweep/s/1': ['3', '4']}
    results = {'catalog/a#nav': ['1', '2']}
    assert render_digests._compare(base, results, only='catalog/') == 0


def test_a_run_matching_no_render_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A prefix no render name starts with gives status 1 and says so."""
    out = tmp_path / 'digests.json'
    monkeypatch.setattr(sys, 'argv', ['render_digests.py', str(out), '--only', 'nothing/'])
    assert render_digests.main() == 1
    assert "no render name starts with 'nothing/'" in capsys.readouterr().out
