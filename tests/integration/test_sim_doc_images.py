"""Staleness guard for the committed documentation-gallery images.

The gallery PNGs under ``docs/dev_guide/_sim_images/`` and
``docs/simulator_report/_scene_images/`` are committed Sphinx assets written
by ``python -m tests.integration.sim_doc_images``; any change that alters
rendered output must regenerate them in the same change.  This module
enforces that rule mechanically: it regenerates every gallery image into a
temporary directory and asserts each committed PNG is byte-identical to its
fresh counterpart, failing with the list of stale files and the regeneration
command to run.

Regenerating both galleries renders a few dozen small frames, so the module
is ``@pytest.mark.integration`` -- the deliberate tier, not the fast unit
suite.  Everything renders in-process (no holdings or SPICE), so it is also
``@pytest.mark.sim_integration``, which CI runs.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.integration import sim_doc_images

pytestmark = [pytest.mark.integration, pytest.mark.sim_integration]

_DOCS = Path(__file__).parent.parent.parent / 'docs'
# Committed gallery directory per gallery key; the keys index the freshly
# regenerated directories the fixture returns.
_COMMITTED_DIRS: dict[str, Path] = {
    'dev_guide': _DOCS / 'dev_guide' / '_sim_images',
    'report': _DOCS / 'simulator_report' / '_scene_images',
}
_REGEN_COMMAND = 'python -m tests.integration.sim_doc_images'


@pytest.fixture(scope='module')
def regenerated_dirs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Regenerate both galleries once into temporary directories."""
    root = tmp_path_factory.mktemp('sim_doc_images')
    dirs = {'dev_guide': root / 'dev_guide', 'report': root / 'report'}
    sim_doc_images.generate(gui_dir=dirs['dev_guide'], report_dir=dirs['report'])
    return dirs


def test_committed_gallery_images_are_current(regenerated_dirs: dict[str, Path]) -> None:
    """Every committed gallery PNG is byte-identical to a fresh render.

    A mismatch means a rendering change landed without regenerating the
    committed galleries; rerun the generator and review the image diff.
    """
    stale: list[str] = []
    # Counted per gallery rather than in total: a total covers for a gallery
    # that produced nothing, and every committed image in that gallery would
    # then go unchallenged rather than being reported stale.
    empty: list[str] = []
    for gallery, fresh_dir in regenerated_dirs.items():
        committed_dir = _COMMITTED_DIRS[gallery]
        fresh_images = sorted(fresh_dir.glob('*.png'))
        if not fresh_images:
            empty.append(gallery)
        for fresh in fresh_images:
            committed = committed_dir / fresh.name
            if not committed.is_file() or committed.read_bytes() != fresh.read_bytes():
                stale.append(str(committed.relative_to(_DOCS.parent)))
    assert not empty, f'the generator produced no images for gallery/galleries: {empty}'
    assert not stale, (
        f'{len(stale)} committed doc image(s) no longer match a fresh render: '
        f'{stale}; regenerate with `{_REGEN_COMMAND}` and review the diff before committing'
    )


def test_no_orphaned_committed_gallery_images(regenerated_dirs: dict[str, Path]) -> None:
    """No committed gallery PNG has lost its generator definition.

    A committed PNG the generator no longer writes would sit stale forever;
    delete it (or restore its gallery entry) when this fails.
    """
    orphans: list[str] = []
    for gallery, fresh_dir in regenerated_dirs.items():
        fresh_names = {p.name for p in fresh_dir.glob('*.png')}
        for committed in sorted(_COMMITTED_DIRS[gallery].glob('*.png')):
            if committed.name not in fresh_names:
                orphans.append(str(committed.relative_to(_DOCS.parent)))
    assert not orphans, (
        f'committed doc image(s) with no generator definition: {orphans}; '
        f'delete them or restore their entries in tests/integration/sim_doc_images.py'
    )
