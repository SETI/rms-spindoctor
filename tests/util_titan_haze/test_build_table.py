"""The committed Titan haze table is what ``util/titan_haze/build_table.py`` builds.

The builder lives outside the importable package, so it is loaded from its path.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

from spindoctor.sim.render.bodies import titan_haze_table

_BUILDER_PATH = Path(__file__).resolve().parents[2] / 'util' / 'titan_haze' / 'build_table.py'


def _load_builder() -> ModuleType:
    """Load ``util/titan_haze/build_table.py`` as a uniquely named module."""
    spec = importlib.util.spec_from_file_location('titan_haze_build_table', _BUILDER_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_committed_table_is_the_built_table() -> None:
    """Rebuilding the table from the DISR data reproduces the committed module."""
    builder = _load_builder()
    rebuilt = builder.render_module(builder.build_rows())
    assert rebuilt == Path(titan_haze_table.__file__).read_text()
