"""The eight navigation documents the statistics fixture tree holds.

The statistics ingest and the report regression both run over
``data/results_tree``, and the frozen report output under ``data/golden`` is
what that tree produces.  A document there therefore has to be one the
pipeline could have written: anything else lets the ingest and every report
section be measured against key sets, vocabularies and value shapes no writer
emits, and the frozen output then freezes whatever those produced.

So the documents are built here, through the writer itself.  Seven of them are
the return value of
:func:`~spindoctor.navigate_image_files.build_metadata_from_result` over a real
:class:`~spindoctor.nav_orchestrator.nav_result.NavResult`; the eighth, whose
image never loads, is the return value of
:func:`~spindoctor.navigate_image_files.navigate_image_files` itself.  All eight
are serialized by the writer's own
:func:`~spindoctor.support.file.json_as_string`.  What is chosen here is only
what a navigation run reads from its image and its configuration: the geometry,
the scores, the epochs, the clock.

Three values a run takes from the machine it runs on are pinned instead, since a
stored document cannot hold one: the ``timing`` block is built by the writer's
own :func:`~spindoctor.navigate_image_files.build_timing_section` from fixed
moments, its ``peak_memory_bytes`` is written over with a fixed count because the
writer reads that one out of the running process, and
``provenance.pipeline_run_iso8601`` carries a fixed stamp in the spelling the
orchestrator produces.  Each document is given a different peak, so that the
report's minimum, maximum, mean, median and standard deviation over them are
five different numbers and a wrong one shows.

The builders are one module per host -- ``cassini``, ``voyager`` and
``simulated`` -- over the constants and writer wrappers in ``shared``.  This
module is the whole public surface.

Run the package as a script, from the repository root, to write the tree
again::

    PYTHONPATH=src python -m tests.spindoctor.cli.stats.results_tree_documents

The stored tree is then what the writer emits, and the frozen report output has
to be re-ratified against it.  ``test_results_tree_documents.py`` holds the two
against each other, so a writer change is reported here rather than being
absorbed silently by a tree nothing checks.

What the tree covers
--------------------

Every document earns its place, and regenerating one must not cost the report a
section or a column:

- **Three instruments**: two with SPICE camera frames (``coiss``, ``vgiss``)
  and the simulated scene, which is the one host that correctly records no
  attitude and no exposure times.
- **Two subtrees and a bare stub**: ``COISS_2001`` and ``VGISS_5101`` name a
  subtree; the simulated scene's basename names none, which is the case a stub
  with no separator produces.
- **Three outcomes**: five successes, two failed navigations, and one image
  whose load failed before an observation existed.  The last records no epoch,
  no image shape and no navigation result at all, so its date cells are empty
  and its reason is a ``status_error`` rather than a ``status_reason``.
- **Two failure reasons over two instruments**, each consistent with its own
  inventory: every feature gated on the Cassini failure, no feature at all on
  the Voyager one.
- **Four feature sources**: a body, a second body, a ring system and a star
  catalog, with gated features under two of them.
- **Four camera and image-size groups**, so the offset tables have more than
  one row to order.
- **A BOTSIM pair**: ``N1294561202`` and ``W1294561202`` share a shutter,
  a spacecraft clock and an epoch, and both carry ``shutter_mode`` of
  ``BOTSIM``.  The other Cassini images carry ``NACONLY``; Voyager and the
  simulated scene carry none, as their hosts read none.
- **A suspect offset**: one Cassini image's fused offset reaches the search
  limit for its size, and one Voyager size has no configured limit at all.
- **A spurious technique and an ensemble exclusion**, on separate techniques,
  since the ensemble drops a spurious result before consensus selection and can
  never report one as excluded.
- **Images with two contributing techniques**, which is what the
  cross-technique agreement and confidence-calibration sections measure.
- **A distinct elapsed time per image**, including the shortest on the image
  that failed to load.
- **Clock readings that span their own exposures**: every spacecraft clock
  triple is counted from the host's reading at shutter open, so the interval it
  spans is the interval its own epochs span, to within the tick that clock
  counts in.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from spindoctor.support.file import json_as_string

from .cassini import (
    LOAD_ERROR_STUB,
    cassini_all_features_gated,
    cassini_load_error,
    cassini_ring_edges,
    cassini_star_and_limb,
    cassini_suspect_offset,
)
from .shared import COISS_SUBTREE, VGISS_SUBTREE
from .simulated import simulated_scene
from .voyager import voyager_no_features, voyager_ring_edges

__all__ = [
    'RESULTS_TREE',
    'results_tree_documents',
    'stored_documents',
    'write_results_tree',
]

RESULTS_TREE = Path(__file__).resolve().parent.parent / 'data' / 'results_tree'
"""Where the stored tree lives."""


def results_tree_documents() -> dict[str, dict[str, Any]]:
    """Return every document of the fixture tree, keyed by its results path stub.

    Returns:
        Stub to document, in the order the tree is written.
    """
    return {
        f'{COISS_SUBTREE}/N1294561202_1_CALIB': cassini_star_and_limb(),
        f'{COISS_SUBTREE}/N1294562000_1_CALIB': cassini_all_features_gated(),
        LOAD_ERROR_STUB: cassini_load_error(),
        f'{COISS_SUBTREE}/N1294564000_1_CALIB': cassini_suspect_offset(),
        f'{COISS_SUBTREE}/W1294561202_1_CALIB': cassini_ring_edges(),
        f'{VGISS_SUBTREE}/C1385455_GEOMED': voyager_ring_edges(),
        f'{VGISS_SUBTREE}/C1385460_GEOMED': voyager_no_features(),
        'sim_scene_000042': simulated_scene(),
    }


def write_results_tree(root: Path) -> list[Path]:
    """Write every document into a results root, as the writer serializes them.

    Parameters:
        root: The results root to write under.

    Returns:
        The paths written, in tree order.
    """
    written: list[Path] = []
    for stub, document in results_tree_documents().items():
        path = root / f'{stub}_metadata.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json_as_string(document), encoding='utf-8')
        written.append(path)
    return written


def stored_documents() -> dict[str, dict[str, Any]]:
    """Return every document the stored tree holds, keyed by its stub.

    Returns:
        Stub to parsed document.
    """
    found: dict[str, dict[str, Any]] = {}
    for path in sorted(RESULTS_TREE.rglob('*_metadata.json')):
        stub = path.relative_to(RESULTS_TREE).as_posix().removesuffix('_metadata.json')
        parsed = json.loads(path.read_text(encoding='utf-8'))
        assert isinstance(parsed, dict)
        found[stub] = parsed
    return found
