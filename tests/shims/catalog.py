"""Fake star catalogs and ``MutableStar`` records.

The real star catalogs (UCAC4, Tycho-2 via SPICE, YBSC) require
multi-GB on-disk binaries, network access, or both.  This module
provides:

- :class:`FakeStar` — minimal ``MutableStar``-protocol-compatible
  record that the catalog reduction can mutate freely.
- :class:`FakeStarCatalog` — implements the
  ``find_stars(*, ra_min, ra_max, dec_min, dec_max, vmag_min,
  vmag_max, **kwargs)`` API the real catalogs expose.
- :func:`install_fake_catalogs` — pytest helper that monkeypatches
  the lazy catalog getters in
  :mod:`spindoctor.nav_model.stars.catalog` so the production code reads from
  fake catalogs instead.

The shim is generic — tests can plug in any list of stars and any
combination of catalogs (e.g. just UCAC4, or all three).
"""

from __future__ import annotations

import copy
import math
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    import pytest

__all__ = [
    'FakeStar',
    'FakeStarCatalog',
    'install_fake_catalogs',
    'make_star',
]


@dataclass
class FakeStar:
    """Mutable star record satisfying the ``MutableStar`` protocol surface.

    The fields cover what the catalog reduction reads or writes during
    a single navigation pass: identity, photometry, RA/DEC, proper
    motion, image-space placement, and conflict tagging.  Defaults are
    chosen so that a minimal star (``FakeStar(ra=..., dec=...,
    vmag=...)``) is a complete record once the reduction populates the
    derived fields.

    Parameters:
        unique_number: Catalog ID.  The catalog reduction copies this
            into ``pretty_name`` when ``name`` is empty.
        catalog_name: Catalog string (set by the reduction).
        pretty_name: Human-readable name (set by the reduction).
        name: Optional canonical name from the catalog.
        ra: Catalog right ascension in radians.
        dec: Catalog declination in radians.
        vmag: Catalog V-band magnitude.
        b_v: Catalog ``B-V`` colour, or ``None``.
        johnson_mag_v: Johnson V magnitude (set by the reduction when
            absent).
        johnson_mag_b: Johnson B magnitude (set by the reduction).
        johnson_mag_faked: ``True`` when the reduction supplied
            ``johnson_mag_b`` from the spectral-class colour table
            instead of the catalog.
        photometry_corrected: ``True`` when the bright-end saturation
            pass replaced ``vmag`` with a YBSC or Tycho-2 reference value.
        photometry_saturated: ``True`` when the record is bright per the
            catalog but has no reference match, so its magnitude is
            unreliable and potentially too faint.
        spectral_class: MK spectral class, e.g. ``'G0'``.  The
            reduction defaults to ``stars.default_star_class`` when
            absent.
        temperature: Surface temperature (K).  The reduction supplies
            a default when absent.
        temperature_faked: ``True`` when the reduction supplied a
            default temperature.
        ra_pm: Proper-motion-corrected RA at obs midtime.
        dec_pm: Proper-motion-corrected DEC at obs midtime.
        psf_size: Per-star PSF size in pixels (set by the reduction).
        u: Sub-pixel U coordinate (set by the FOV projection).
        v: Sub-pixel V coordinate (set by the FOV projection).
        move_u: Per-exposure smear amplitude along U (set by the FOV
            projection).
        move_v: Per-exposure smear amplitude along V.
        dn: Catalog flux-to-DN scaling (set by the reduction).
        conflicts: Conflict-marking tag.  Empty string when the star
            is usable; ``'STAR'`` / ``'BODY: <body>'`` /
            ``'RING: <planet>'`` otherwise.
        diff_u: Per-star delta-U used by some technique paths.
        diff_v: Per-star delta-V used by some technique paths.
    """

    unique_number: int | None = None
    catalog_name: str = ''
    pretty_name: str = ''
    name: str = ''
    ra: float | None = None
    dec: float | None = None
    vmag: float | None = None
    b_v: float | None = None
    johnson_mag_v: float | None = None
    johnson_mag_b: float | None = None
    johnson_mag_faked: bool = False
    photometry_corrected: bool = False
    photometry_saturated: bool = False
    spectral_class: str | None = None
    temperature: float | None = None
    temperature_faked: bool = False
    ra_pm: float = 0.0
    dec_pm: float = 0.0
    psf_size: tuple[int, int] = (5, 5)
    u: float = 0.0
    v: float = 0.0
    move_u: float = 0.0
    move_v: float = 0.0
    dn: float = 0.0
    conflicts: str = ''
    diff_u: float = 0.0
    diff_v: float = 0.0

    def __post_init__(self) -> None:
        """Mirror the proper-motion fields onto the catalog RA/DEC by default."""
        if self.ra is not None and self.ra_pm == 0.0:
            self.ra_pm = self.ra
        if self.dec is not None and self.dec_pm == 0.0:
            self.dec_pm = self.dec

    def ra_dec_with_pm(self, tdb: float) -> tuple[float, float]:
        """Return the proper-motion-corrected ``(ra, dec)`` at ``tdb``.

        The shim ignores ``tdb`` and returns whatever the test set on
        ``ra_pm`` / ``dec_pm`` (or the catalog RA/DEC when those are
        zero — see ``__post_init__``).

        Parameters:
            tdb: Target ephemeris time (ignored by the shim).

        Returns:
            ``(ra_pm, dec_pm)`` in radians.
        """
        return self.ra_pm, self.dec_pm


def make_star(**kwargs: object) -> FakeStar:
    """Build a :class:`FakeStar` with sensible defaults overridden by kwargs.

    The default star has ``ra = dec = 0``, ``vmag = 5``, spectral class
    ``G0``, no name, and a 5x5 PSF.  Override any field via keyword.

    Parameters:
        **kwargs: Field overrides forwarded to :class:`FakeStar`.

    Returns:
        Configured :class:`FakeStar`.
    """
    defaults: dict[str, object] = {
        'unique_number': 1,
        'ra': 0.0,
        'dec': 0.0,
        'vmag': 5.0,
        'spectral_class': 'G0',
        'temperature': 5800.0,
        'b_v': 0.65,
    }
    defaults.update(kwargs)
    return FakeStar(**defaults)  # type: ignore[arg-type]


@dataclass
class FakeStarCatalog:
    """Stand-in for ``UCAC4StarCatalog`` / ``SpiceStarCatalog`` / ``YBSCStarCatalog``.

    Stores a list of :class:`FakeStar` records and surfaces them through
    the same ``find_stars`` API the production code calls.  All the
    catalog-specific kwargs (``allow_double``, ``allow_galaxy``) are
    accepted and ignored.

    Parameters:
        stars: Stars the catalog "knows about".
    """

    stars: list[FakeStar] = field(default_factory=list)

    def find_stars(
        self,
        *,
        ra_min: float,
        ra_max: float,
        dec_min: float,
        dec_max: float,
        vmag_min: float,
        vmag_max: float,
        **_: object,
    ) -> Iterator[FakeStar]:
        """Yield deep copies of every star inside the given RA / DEC / mag box.

        Deep-copying matches the production code's expectation: the
        reduction mutates the returned records freely.  Records with a
        ``None`` ``ra`` / ``dec`` / ``vmag`` are still yielded so that
        the production code's defensive None-checks are reachable in
        tests; only records with concrete values participate in the
        RA / DEC / mag window filter.

        Parameters:
            ra_min, ra_max: Right-ascension window in radians.
            dec_min, dec_max: Declination window in radians.
            vmag_min, vmag_max: V-band magnitude window.

        Yields:
            Deep copies of any matching :class:`FakeStar`.
        """
        for star in self.stars:
            if star.ra is None or star.dec is None or star.vmag is None:
                yield copy.deepcopy(star)
                continue
            if not ra_min <= star.ra <= ra_max:
                continue
            if not dec_min <= star.dec <= dec_max:
                continue
            if math.isnan(star.vmag):
                continue
            if not vmag_min <= star.vmag <= vmag_max:
                continue
            yield copy.deepcopy(star)


def install_fake_catalogs(
    monkeypatch: pytest.MonkeyPatch,
    *,
    ucac4: Iterable[FakeStar] | None = None,
    tycho2: Iterable[FakeStar] | None = None,
    ybsc: Iterable[FakeStar] | None = None,
) -> dict[str, FakeStarCatalog]:
    """Swap the lazy catalog getters for test-local fakes.

    The patches install via the pytest ``monkeypatch`` fixture, which
    records the originals and restores them on test teardown.  Each
    call patches **only the calling test's process and only for the
    lifetime of that test** — pytest-xdist workers run in separate
    processes so module-level state in one worker cannot leak into
    another, and the per-test ``monkeypatch`` teardown means subsequent
    tests in the same worker see the unpatched getters again.

    The module-level catalog cache (``_STAR_CATALOG_*``) is not
    touched: the helper replaces the *getter functions* so the cache
    is bypassed entirely while the patches are in effect.  Any catalog
    instance previously cached by an earlier test that imported the
    real getter is undisturbed, and that earlier cache cannot leak the
    fake catalog into a later test because the fake never reaches the
    cache.

    Tests that omit a catalog (e.g. only pass ``ucac4=...``) get an
    empty :class:`FakeStarCatalog` for the other two catalogs, which
    surfaces no stars to the catalog reduction.

    Parameters:
        monkeypatch: Pytest fixture; required so the swap is bound to
            the calling test's lifetime.
        ucac4: Stars exposed by ``get_ucac4_catalog()``.
        tycho2: Stars exposed by ``get_tycho2_catalog()``.
        ybsc: Stars exposed by ``get_ybsc_catalog()``.

    Returns:
        Mapping ``{'ucac4', 'tycho2', 'ybsc'}`` -> the constructed
        :class:`FakeStarCatalog` instances; tests can inspect or mutate
        them after installation.
    """
    cat_map = {
        'ucac4': FakeStarCatalog(stars=list(ucac4) if ucac4 is not None else []),
        'tycho2': FakeStarCatalog(stars=list(tycho2) if tycho2 is not None else []),
        'ybsc': FakeStarCatalog(stars=list(ybsc) if ybsc is not None else []),
    }
    for name in cat_map:
        monkeypatch.setattr(
            f'spindoctor.nav_model.stars.catalog.get_{name}_catalog',
            lambda _name=name: cat_map[_name],
        )
    return cat_map
