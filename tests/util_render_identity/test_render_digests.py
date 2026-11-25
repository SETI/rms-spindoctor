"""Tests for the render digest check.

The check lives in ``util/render_identity/render_digests.py``, outside the importable
package, and is loaded from its path.
"""

import hashlib
import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

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
