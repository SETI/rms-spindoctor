"""Unit tests for the normal albedo the realism match measures from a real disc."""

from __future__ import annotations

import math

import numpy as np
import pytest

from spindoctor.sim.realism.albedo import disc_normal_albedo
from spindoctor.sim.render.bodies.photometry import shade_surface

_PHASE = math.radians(30.0)


def _cosines(n: int) -> tuple[np.ndarray, np.ndarray]:
    """``n`` pixels' incidence and emission cosines, all at least 0.3."""
    rng = np.random.default_rng(3)
    return rng.uniform(0.3, 1.0, n), rng.uniform(0.3, 1.0, n)


def test_a_lommel_seeliger_disc_gives_its_normal_albedo() -> None:
    """A disc drawn at 0.42 times the Lommel-Seeliger law measures 0.42."""
    mu0, mu = _cosines(200)
    disc = 0.42 * shade_surface(mu0, mu, law='lommel_seeliger')
    albedo = disc_normal_albedo(disc, mu0, mu, law='lommel_seeliger', phase_angle=_PHASE)
    assert albedo == pytest.approx(0.42, rel=1e-12)


def test_pixels_near_the_terminator_or_the_limb_do_not_count() -> None:
    """Pixels with either cosine below 0.3 leave the albedo as it was.

    Each kind of excluded pixel outnumbers the disc pixels, so counting either kind
    would move the median.
    """
    mu0, mu = _cosines(200)
    disc = 0.42 * shade_surface(mu0, mu, law='lommel_seeliger')
    low = np.full(250, 0.2)
    high = np.full(250, 0.9)
    albedo = disc_normal_albedo(
        np.concatenate([disc, np.full(500, 5.0)]),
        np.concatenate([mu0, low, high]),
        np.concatenate([mu, high, low]),
        law='lommel_seeliger',
        phase_angle=_PHASE,
    )
    assert albedo == pytest.approx(0.42, rel=1e-12)


def test_too_few_pixels_give_no_albedo() -> None:
    """49 pixels that count are too few."""
    mu0, mu = _cosines(49)
    disc = 0.42 * shade_surface(mu0, mu, law='lommel_seeliger')
    assert disc_normal_albedo(disc, mu0, mu, law='lommel_seeliger', phase_angle=_PHASE) is None
