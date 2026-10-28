"""Ring-edge precession: ``ring_epoch`` flows to both sides of the simulator.

A mode-1 eccentric edge with nonzero ``rate_peri_deg_per_day`` precesses by
that rate times the days from ``ring_epoch`` to ``time`` (both TDB seconds).
These tests render a precessing ring_system ringlet through ``ObsSim`` and
check end to end that (a) the navigator-side predicted edges
(``NavModelRingsSimulated``, which reads ``time`` and ``ring_epoch`` from the
filtered ``obs.nav_params`` and places edges through the shared orbit model)
land on the rendered annulus boundary for a nonzero epoch -- both sides
precess together -- and (b) changing ``ring_epoch`` alone moves both the
prediction and the render, so the epoch is not silently defaulted away on
either side.
"""

from typing import Any, cast

import numpy as np

from spindoctor.feature.geometry import RingEdgePolyline
from spindoctor.nav_model.nav_model_rings_simulated import NavModelRingsSimulated
from spindoctor.nav_orchestrator.nav_context import NavContext
from spindoctor.obs.obs_inst_sim import ObsSim
from spindoctor.sim.scene import validate_sim_params
from spindoctor.support.types import NDArrayBoolType

_SIZE = 160
_CENTER = _SIZE / 2.0
_TIME = 864000.0  # 10 days in seconds
_EPOCH = 345600.0  # 4 days in seconds; 6 days of precession vs 10 at epoch 0
_RATE_PERI = 15.0  # deg/day: epochs 0 and _EPOCH differ by 60 deg of pericenter


def _scene(ring_epoch: float) -> dict[str, Any]:
    """A noise-free (calibrated) scene with one precessing eccentric ringlet."""
    scene: dict[str, Any] = {
        'schema_version': 2,
        'scene_name': 'ring_epoch_probe',
        'instrument': 'coiss_calib_nac',
        'size_v': _SIZE,
        'size_u': _SIZE,
        'random_seed': 5,
        'time': _TIME,
        'ring_epoch': ring_epoch,
        'ring_system': {
            'geometry': {
                'center_v': _CENTER,
                'center_u': _CENTER,
                'opening_deg_obs': 90.0,
                'opening_deg_sun': 90.0,
                'node_deg': 0.0,
            },
            'features': [
                {
                    'name': 'PRECESSING',
                    'kind': 'ringlet',
                    'tau': 2.0,
                    'width_px': 15.0,
                    'navigable': True,
                    'orbit': {
                        'a_px': 40.0,
                        'ae_px': 8.0,
                        'long_peri_deg': 20.0,
                        'rate_peri_deg_per_day': _RATE_PERI,
                    },
                }
            ],
        },
    }
    return validate_sim_params(scene, source='ring_epoch_probe')


def _obs(ring_epoch: float) -> ObsSim:
    """Render the scene through ObsSim (nav_params carries time and epoch)."""
    return ObsSim.from_file('/tmp/ring_epoch_probe.yaml', sim_params=_scene(ring_epoch))


def _predicted_edge_vertices(obs: ObsSim) -> dict[str, Any]:
    """The navigator's predicted inner/outer edge vertices, in data coords."""
    models = NavModelRingsSimulated.instances_for_obs(obs)
    assert len(models) == 1
    model = models[0]
    model.create_model()
    out: dict[str, Any] = {}
    for feature in model.to_features(cast(NavContext, None)):
        if feature.feature_type.name != 'RING_EDGE':
            continue
        geometry = feature.geometry
        assert isinstance(geometry, RingEdgePolyline)
        vertices = geometry.vertices_vu.copy()
        vertices[:, 0] -= obs.extfov_margin_v
        vertices[:, 1] -= obs.extfov_margin_u
        out[feature.feature_id.rsplit(':', 1)[-1]] = vertices
    return out


def _rendered_coverage(obs: ObsSim) -> NDArrayBoolType:
    """The rendered annulus coverage mask (noise-free calibrated render).

    The background is the frame's minimum, the calibrated sky residual.
    """
    data = np.asarray(obs.data)
    coverage: NDArrayBoolType = data > float(data.min()) + 1e-9
    return coverage


def test_predicted_edges_sit_on_the_rendered_boundary() -> None:
    """With a nonzero epoch, the predicted edges land on the rendered edges.

    Every navigator-predicted edge vertex must sit on a coverage transition
    of the rendered annulus (a 3x3 neighborhood containing both ring and
    background pixels).  If ``ring_epoch`` reached only one side, the
    60-degree pericenter mismatch would displace the eccentric edge by
    several pixels at some longitudes and the check would fail.
    """
    obs = _obs(_EPOCH)
    coverage = _rendered_coverage(obs)
    edges = _predicted_edge_vertices(obs)
    assert set(edges) == {'inner', 'outer'}
    for vertices in edges.values():
        assert vertices.shape[0] > 0
        for v, u in vertices.astype(int):
            window = coverage[max(v - 1, 0) : v + 2, max(u - 1, 0) : u + 2]
            assert window.any()
            assert not window.all()


def test_ring_epoch_changes_the_prediction() -> None:
    """The same scene at epochs 0 and _EPOCH predicts different edges."""
    edges_at_epoch = _predicted_edge_vertices(_obs(_EPOCH))
    edges_at_zero = _predicted_edge_vertices(_obs(0.0))
    vertices_at_epoch = {tuple(vu) for vu in edges_at_epoch['inner'].astype(int).tolist()}
    vertices_at_zero = {tuple(vu) for vu in edges_at_zero['inner'].astype(int).tolist()}
    assert vertices_at_epoch != vertices_at_zero


def test_ring_epoch_changes_the_render() -> None:
    """The renderer precesses too: the two epochs produce different images."""
    img_at_epoch = np.asarray(_obs(_EPOCH).data)
    img_at_zero = np.asarray(_obs(0.0).data)
    assert not np.array_equal(img_at_epoch, img_at_zero)
