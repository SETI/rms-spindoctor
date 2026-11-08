"""Refuse to run a ``tests.integration`` script against another checkout's code.

A script such as ``python -m tests.integration.update_sim_baselines`` imports
``spindoctor`` from whatever the environment resolves.  In a worktree whose
environment has another checkout installed, that is the other checkout's code, and
the script would silently write baselines, figures, or summaries computed by it into
this checkout.  Every runnable script in this package calls
:func:`require_checkout_package` from its ``__main__`` block before it does any work,
so it either runs this checkout's code or stops and names both paths.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ['foreign_package_message', 'require_checkout_package']


def foreign_package_message(checkout_root: Path, package_dir: Path) -> str | None:
    """Say why an imported package doesn't belong to a checkout, or None when it does.

    The package belongs to the checkout when its directory lies under the
    checkout's ``src`` directory, after both paths are resolved.

    Parameters:
        checkout_root: The checkout's top-level directory.
        package_dir: The directory of the imported ``spindoctor`` package.

    Returns:
        None when the package lies under ``checkout_root / 'src'``; otherwise a
        message naming the package's directory, the checkout, and the
        ``PYTHONPATH`` that selects the checkout's code.
    """
    checkout_src = (checkout_root / 'src').resolve()
    package = package_dir.resolve()
    if package.is_relative_to(checkout_src):
        return None
    return (
        f'spindoctor was imported from {package}, which is outside this checkout '
        f'({checkout_root.resolve()}); run with PYTHONPATH={checkout_src} or '
        'install this checkout'
    )


def require_checkout_package(script_path: str) -> None:
    """Stop a script whose imported ``spindoctor`` lies outside the script's checkout.

    The checkout is the directory two levels above the script's own directory,
    which is where ``tests/integration/<script>.py`` sits in a checkout.

    Parameters:
        script_path: The calling script's ``__file__``.

    Raises:
        SystemExit: With the message from :func:`foreign_package_message` when the
            imported package lies outside the checkout.
    """
    # Imported here so a script that defers its spindoctor imports (as
    # update_baselines does) keeps importing nothing heavy until it runs.
    import spindoctor

    checkout_root = Path(script_path).resolve().parents[2]
    message = foreign_package_message(checkout_root, Path(spindoctor.__file__).parent)
    if message is not None:
        raise SystemExit(message)
