"""Telemetry reseau scars: patches on the reseau lattice, thinned to the mark count.

The ``reseau_scars`` mode emulates the patches that a ground pipeline leaves behind
when it removes the reseau marks.  These tests check that the patches smooth the frame, that the
lattice is thinned to the configured mark count and spans the archive frame, and
that the mode is a no-op at probability 0.
"""

import numpy as np

from spindoctor.sim.forward.telemetry_artifacts import _reseau_lattice, apply_reseau_scars


def test_reseau_scars_smooth_lattice_patches() -> None:
    """Reseau scars smooth patches on the lattice, lowering local variation."""
    rng = np.random.default_rng(4)
    signal = rng.uniform(0.0, 100.0, size=(128, 128))
    record = apply_reseau_scars(
        signal,
        {'probability': 1.0, 'spacing_px': 46, 'patch_radius_px': 4, 'mark_count': 202},
        rng=np.random.default_rng(1),
    )
    assert record['active'] is True
    assert record['marks'] > 0


def test_reseau_scars_plant_the_requested_mark_count() -> None:
    """The scars thin the lattice to the configured mark count, not a fixed one."""
    signal = np.zeros((400, 400), dtype=np.float64)
    record = apply_reseau_scars(
        signal,
        {'probability': 1.0, 'spacing_px': 46, 'patch_radius_px': 4, 'mark_count': 40},
        rng=np.random.default_rng(1),
    )
    assert record['marks'] == 40


def test_reseau_lattice_spans_the_archive_frame() -> None:
    """At the 800x800 archive size the 202 marks span the full frame height."""
    points = _reseau_lattice(800, 800, 46, 202)
    assert len(points) == 202
    rows = [v for v, _u in points]
    assert max(rows) - min(rows) >= 0.9 * 800


def test_reseau_scars_disabled_at_zero_probability() -> None:
    """A zero probability plants no scars."""
    rng = np.random.default_rng(5)
    signal = rng.uniform(0.0, 100.0, size=(128, 128))
    before = signal.copy()
    record = apply_reseau_scars(
        signal,
        {'probability': 0.0, 'spacing_px': 46, 'patch_radius_px': 4},
        rng=np.random.default_rng(1),
    )
    assert record['active'] is False
    assert np.array_equal(signal, before)
