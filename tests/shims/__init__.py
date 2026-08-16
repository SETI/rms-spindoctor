"""Reusable shims for nav-pipeline unit tests.

The shims here let tests drive the navigation pipeline end-to-end
without a real ``oops.Observation`` / ``Backplane``, real SPICE
kernels, or real star catalogs.  They are intentionally small and
table-driven; tests construct one with the data the code under test
will read and ignore everything else.

Modules:

    ``backplane``
        :class:`FakeBackplane`, :class:`BodyBackplaneData`,
        :class:`RingBackplaneData`, plus the
        :func:`plant_circular_body` factory.  Backplane methods return
        real ``polymath.Scalar`` instances built from the configured
        per-pixel arrays.
    ``catalog``
        :class:`FakeStar`, :class:`FakeStarCatalog`,
        :func:`install_fake_catalogs`, and a
        :func:`make_star` convenience factory.  ``install_fake_catalogs``
        is per-test-scoped via the pytest ``monkeypatch`` fixture and
        is safe under parallel test execution.
    ``obs``
        :class:`FakeObs` plus support classes
        :class:`FakePSF`, :class:`FakeFOV`,
        :class:`FakeMeshgrid`, :class:`FakeUV`.
    ``context``
        :func:`bare_nav_context`, a minimal ``NavContext`` factory for
        feature-emission tests.
"""

from tests.shims.backplane import (
    BodyBackplaneData,
    FakeBackplane,
    RingBackplaneData,
    plant_circular_body,
)
from tests.shims.catalog import (
    FakeStar,
    FakeStarCatalog,
    install_fake_catalogs,
    make_star,
)
from tests.shims.context import bare_nav_context
from tests.shims.obs import (
    FakeFOV,
    FakeMeshgrid,
    FakeObs,
    FakePSF,
    FakeUV,
    probe_grid_vu,
)

__all__ = [
    'BodyBackplaneData',
    'FakeBackplane',
    'FakeFOV',
    'FakeMeshgrid',
    'FakeObs',
    'FakePSF',
    'FakeStar',
    'FakeStarCatalog',
    'FakeUV',
    'RingBackplaneData',
    'bare_nav_context',
    'install_fake_catalogs',
    'make_star',
    'plant_circular_body',
    'probe_grid_vu',
]
