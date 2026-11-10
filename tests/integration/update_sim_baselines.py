"""Regenerate the simulator regression baselines.

Invoke from a project checkout::

    python -m tests.integration.update_sim_baselines

For each scene in the catalog (``tests/integration/sim_scenes/``) this renders,
navigates, and writes the outcome, offsets and confidence unrounded, to
``tests/integration/sim_baselines/<scene_name>.json``.  Run it whenever a
deliberate change to the sim or the navigator shifts a scene's outcome, then
review the diff before committing -- the baselines are tripwires, so an
unexpected diff is a regression to investigate, not to bless blindly.
"""

from __future__ import annotations

import sys
from pathlib import Path

from spindoctor.sim.scene import iter_scene_paths, load_sim_scene
from tests.integration.checkout_guard import require_checkout_package
from tests.integration.sim_baseline import baseline_for_scene, sim_baseline_path

_SCENES_ROOT = Path(__file__).parent / 'sim_scenes'
_BASELINES_DIR = Path(__file__).parent / 'sim_baselines'


def main() -> int:
    """Render + navigate every catalog scene and write its baseline."""
    _BASELINES_DIR.mkdir(exist_ok=True)
    scene_paths = iter_scene_paths(_SCENES_ROOT)
    if not scene_paths:
        print('No scenes found; nothing to do.', file=sys.stderr)
        return 1
    for scene_path in scene_paths:
        scene = load_sim_scene(scene_path)
        baseline = baseline_for_scene(scene)
        scene_name = scene['scene_name']
        out_path = sim_baseline_path(_BASELINES_DIR, scene_name)
        out_path.write_text(baseline.to_json())
        offset = (
            f'({baseline.offset_dv_px}, {baseline.offset_du_px})'
            if baseline.offset_dv_px is not None
            else 'none'
        )
        print(
            f'{scene_name}: {baseline.status} ({baseline.confidence_rank}) '
            f'offset={offset} conf={baseline.confidence}'
        )
    print(f'Wrote {len(scene_paths)} baseline(s) to {_BASELINES_DIR}')
    return 0


if __name__ == '__main__':
    require_checkout_package(__file__)
    raise SystemExit(main())
