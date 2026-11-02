"""The guard that keeps ``np.hypot`` out of the simulator.

``np.hypot`` calls the C library's ``hypot``, whose last bit differs between
CPUs, so the committed renders would only reproduce on some machines.  The
simulator uses :func:`spindoctor.support.stable_math.stable_hypot` instead.
"""

import ast
from pathlib import Path

import pytest

_SIM_ROOT = Path(__file__).resolve().parents[3] / 'src' / 'spindoctor' / 'sim'


def _hypot_calls(path: Path) -> list[int]:
    """Return the line of each ``np.hypot`` or ``numpy.hypot`` reference in a module."""
    tree = ast.parse(path.read_text(), filename=str(path))
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and node.attr == 'hypot'
        and isinstance(node.value, ast.Name)
        and node.value.id in {'np', 'numpy'}
    ]


@pytest.mark.parametrize(
    'module', sorted(_SIM_ROOT.rglob('*.py')), ids=lambda p: str(p.relative_to(_SIM_ROOT))
)
def test_the_simulator_uses_stable_hypot_in_place_of_numpy_hypot(module: Path) -> None:
    """No simulator module references ``np.hypot``; it uses ``stable_hypot``."""
    assert _hypot_calls(module) == []
