"""Integration test: Saturn's limb is placed at the configured altitude on a real frame.

Runs the real geometry layer (oops ``Meshgrid`` / ``Backplane`` limb altitude) on a
Cassini ISS wide angle frame of the rev 201 spoke movie, so the ``(u, v)`` order and
pixel conventions of the placement are checked against SPICE rather than a stand-in.

Gated by ``pytestmark = pytest.mark.integration`` and skipped when
``PDS3_HOLDINGS_DIR`` is not set.
"""

from __future__ import annotations

import os
from typing import Any, cast

import numpy as np
import pytest

pytestmark = pytest.mark.integration

if 'PDS3_HOLDINGS_DIR' not in os.environ:
    pytest.skip(
        'PDS3_HOLDINGS_DIR is not set; skipping the visible-limb integration test',
        allow_module_level=True,
    )

from filecache import FCPath  # noqa: E402  (guarded import)

from spindoctor.config import DEFAULT_CONFIG  # noqa: E402
from spindoctor.config.visible_limb import visible_limb_model_for  # noqa: E402
from spindoctor.feature.geometry import LimbPolyline  # noqa: E402
from spindoctor.nav_model.limb_placement import limb_altitudes_km  # noqa: E402
from spindoctor.nav_model.nav_model_body import NavModelBody  # noqa: E402
from spindoctor.obs import ObsCassiniISS  # noqa: E402
from spindoctor.support.constants import PIXEL_CENTER_TO_CORNER_PX  # noqa: E402
from tests.shims import bare_nav_context  # noqa: E402

_IMAGE = 'calibrated/COISS_2xxx/COISS_2088/data/1770788381_1770923800/W1770872554_1_CALIB.IMG'


def test_saturn_limb_sits_at_the_configured_altitude() -> None:
    """Every placed limb vertex lies on the line of sight at the table's altitude."""
    obs = ObsCassiniISS.from_file(FCPath(os.environ['PDS3_HOLDINGS_DIR']) / _IMAGE)
    model = NavModelBody('body:SATURN', obs, 'SATURN')
    model.create_model()
    geometry = next(
        f.geometry
        for f in model.to_features(bare_nav_context(cast(Any, obs)))
        if isinstance(f.geometry, LimbPolyline)
    )
    assert isinstance(geometry, LimbPolyline)
    vertices = geometry.vertices_vu
    uv = np.stack(
        [
            vertices[:, 1] - obs.extfov_margin_u + PIXEL_CENTER_TO_CORNER_PX,
            vertices[:, 0] - obs.extfov_margin_v + PIXEL_CENTER_TO_CORNER_PX,
        ],
        axis=-1,
    )
    entry = visible_limb_model_for(DEFAULT_CONFIG.category('cassini_iss_calib')['wac'], 'SATURN')
    assert entry is not None
    phase = float(model.metadata['phase_angle_deg'])
    expected = entry.altitude_for_image_km(phase, ['CL1', 'CL2'])
    altitude = limb_altitudes_km(obs, 'SATURN', uv)
    assert float(np.nanmax(np.abs(altitude - expected))) < 1.0
