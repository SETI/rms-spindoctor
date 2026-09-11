"""Retrieving a kernel from under ``OOPS_RESOURCES`` for a test module, or skipping it.

``OOPS_RESOURCES`` names the tree the integration tests' kernels come from, which may be
a local directory or a URL.  Read as a local path, a URL makes every kernel under it look
missing, and a module that skips on a missing kernel then skips silently wherever the
tree is remote.  Here the kernel is retrieved through ``FCPath``, which fetches a URL's
file into the local cache, so a module is skipped only when the variable is unset or the
retrieval itself fails, and the message names the kernel and why.
"""

import os
from pathlib import Path
from typing import cast

import pytest
from filecache import FCPath


def retrieved_kernel(relative: str, *, what: str, tests: str) -> Path:
    """Retrieve a kernel from under ``OOPS_RESOURCES``, or skip the calling module.

    Meant for a test module's top level, where ``pytest.skip`` skips the whole module.
    A kernel that is not there skips it (``FileNotFoundError``), and so does a root that
    cannot be reached (``ConnectionError``).

    Parameters:
        relative: The kernel's path under ``OOPS_RESOURCES``, as in
            ``SPICE/General/LSK/naif0012.tls``.
        what: What the kernel is, for the message, as in ``the leapseconds kernel``.
        tests: What goes unrun without it, for the message.

    Returns:
        The kernel's local path: the file itself under a local tree, the cached copy
        under a URL.  ``cspyce.furnsh`` takes it as a string.
    """
    resources = os.environ.get('OOPS_RESOURCES', '')
    if len(resources) == 0:
        pytest.skip(
            f'OOPS_RESOURCES is not set, so {what} cannot be retrieved; skipping {tests}',
            allow_module_level=True,
        )
    kernel = FCPath(resources.rstrip('/')) / relative
    try:
        return cast(Path, kernel.retrieve())
    except (FileNotFoundError, ConnectionError) as exc:
        pytest.skip(
            f'{what} {kernel.as_posix()} could not be retrieved ({exc}); skipping {tests}',
            allow_module_level=True,
        )
