"""One stated scene position draws, and predicts, every kind of object on one pixel.

A scene states every position as a pixel corner, so ``(60.5, 60.5)`` is the
center of array pixel ``(60, 60)``.  A star, a lit body, a Titan with its haze,
and a face-on ringlet each stated there must render centered on that pixel, and
each simulated navigation model must predict its object there too: the body's
disc and blob payloads, the Titan limb payload, and the ring annulus payload,
all in pixel centric extfov coordinates less the margin.  The body disc and
ring annulus templates must also be painted there: each template's masked
centroid, placed at its bbox origin, lands on the rendered centroid.  The rendered
centroids are the anchor, so a renderer or a model that read the scene's corner
coordinate pixel centric fails by the half pixel between the two systems.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from tests.shims import bare_nav_context

from spindoctor.feature.feature import NavFeature
from spindoctor.feature.feature_type import NavFeatureType
from spindoctor.nav_model.nav_model import NavModel
from spindoctor.nav_model.nav_model_body_simulated import NavModelBodySimulated
from spindoctor.nav_model.nav_model_rings_simulated import NavModelRingsSimulated
from spindoctor.nav_model.nav_model_titan_simulated import NavModelTitanSimulated
from spindoctor.obs.obs_inst_sim import ObsSim
from spindoctor.sim.render import render_combined_model
from spindoctor.support.types import NDArrayFloatType

_SIZE = 128
_STATED = 60.5
_PIXEL = 60.0
_TOLERANCE_PX = 0.01


def _scene(**content: Any) -> dict[str, Any]:
    """A noiseless coiss_nac scene holding ``content``."""
    return {
        'instrument': 'coiss_nac',
        'size_v': _SIZE,
        'size_u': _SIZE,
        'random_seed': 1,
        'exposure_sec': 1.0,
        **content,
    }


def _sphere(name: str = 'BODY', **extra: Any) -> dict[str, Any]:
    """A head-on, fully lit sphere stated at the probe position."""
    return {
        'name': name,
        'center_v': _STATED,
        'center_u': _STATED,
        'axis1': 20.0,
        'axis2': 20.0,
        'axis3': 20.0,
        'phase_angle': 0.0,
        'illumination_angle': 0.0,
        'range_km': 1.0e6,
        **extra,
    }


def _titan() -> dict[str, Any]:
    """A Titan with a haze layer, stated at the probe position."""
    return _sphere(
        'TITAN',
        km_per_pixel=20.0,
        atmosphere={'scale_height_px': 4.0, 'tau_ref': 1.0, 'g': 0.0},
    )


def _ring_system() -> dict[str, Any]:
    """A face-on circular ringlet centered on the probe position."""
    return {
        'geometry': {
            'center_v': _STATED,
            'center_u': _STATED,
            'opening_deg_obs': 90.0,
            'opening_deg_sun': 90.0,
            'node_deg': 0.0,
        },
        'features': [
            {
                'name': 'RINGLET',
                'kind': 'ringlet',
                'tau': 1.0,
                'width_px': 4.0,
                'navigable': True,
                'orbit': {'a_px': 20.0},
            }
        ],
    }


_SCENES: dict[str, Callable[[], dict[str, Any]]] = {
    'star': lambda: _scene(
        stars=[{'name': 'S', 'v': _STATED, 'u': _STATED, 'vmag': 2.0, 'psf_sigma': 2.0}]
    ),
    'body': lambda: _scene(bodies=[_sphere()]),
    'titan': lambda: _scene(bodies=[_titan()]),
    'ringlet': lambda: _scene(ring_system=_ring_system()),
}


def _centroid(image: NDArrayFloatType) -> tuple[float, float]:
    """The light-weighted centroid of everything above the frame's median, in array indices."""
    return _centroid_of_weights(np.clip(np.asarray(image) - float(np.median(image)), 0.0, None))


def _centroid_of_weights(weights: NDArrayFloatType) -> tuple[float, float]:
    """The ``weights``-weighted centroid of an array, in its own indices."""
    weights = np.asarray(weights, dtype=np.float64)
    vs, us = np.indices(weights.shape, dtype=np.float64)
    total = float(weights.sum())
    return float((vs * weights).sum() / total), float((us * weights).sum() / total)


@pytest.mark.parametrize('kind', list(_SCENES), ids=list(_SCENES))
def test_a_stated_corner_position_renders_on_its_pixel(kind: str) -> None:
    """Each kind of object stated at ``(60.5, 60.5)`` renders centered on pixel ``(60, 60)``."""
    image, _meta = render_combined_model(_SCENES[kind]())
    centroid_v, centroid_u = _centroid(image)
    assert centroid_v == pytest.approx(_PIXEL, abs=_TOLERANCE_PX)
    assert centroid_u == pytest.approx(_PIXEL, abs=_TOLERANCE_PX)


def _models(kind: str, obs: ObsSim) -> list[NavModel]:
    """The simulated navigation models a scene of ``kind`` builds."""
    if kind == 'body':
        return [NavModelBodySimulated('body', obs, 'BODY', obs.nav_params['bodies'][0])]
    if kind == 'titan':
        return NavModelTitanSimulated.instances_for_obs(obs)
    return NavModelRingsSimulated.instances_for_obs(obs)


_PAYLOADS: list[tuple[str, NavFeatureType]] = [
    ('body', NavFeatureType.BODY_DISC),
    ('body', NavFeatureType.BODY_BLOB),
    ('titan', NavFeatureType.TITAN_LIMB),
    ('ringlet', NavFeatureType.RING_ANNULUS),
]


def _payloads(kind: str, feature_type: NavFeatureType) -> tuple[list[NavFeature], int, int]:
    """The ``feature_type`` features a scene's models emit, and the extfov margins."""
    obs = ObsSim.from_file('/tmp/position_contract.json', sim_params=_SCENES[kind]())
    margin_v = int(obs.extfov_margin_v)
    margin_u = int(obs.extfov_margin_u)
    image_ext = np.zeros(obs.extdata_shape_vu, dtype=np.float64)
    image_ext[margin_v : margin_v + _SIZE, margin_u : margin_u + _SIZE] = np.asarray(obs.data)
    features = []
    for model in _models(kind, obs):
        model.create_model()
        features += [
            f
            for f in model.to_features(bare_nav_context(obs, image_ext))
            if f.feature_type == feature_type
        ]
    return features, margin_v, margin_u


def _rendered_centroid(kind: str) -> tuple[float, float]:
    """The rendered centroid of a scene of ``kind``, in array indices."""
    image, _meta = render_combined_model(_SCENES[kind]())
    return _centroid(image)


@pytest.mark.parametrize(
    ('kind', 'feature_type'), _PAYLOADS, ids=[f'{k}-{t.name}' for k, t in _PAYLOADS]
)
def test_each_simulated_model_predicts_its_object_where_it_renders(
    kind: str, feature_type: NavFeatureType
) -> None:
    """A model's predicted center, less the extfov margin, is the rendered centroid."""
    features, margin_v, margin_u = _payloads(kind, feature_type)
    rendered_v, rendered_u = _rendered_centroid(kind)
    centers = []
    for feature in features:
        center_v, center_u = feature.geometry.predicted_center_vu  # type: ignore[union-attr]
        centers.append((center_v - margin_v, center_u - margin_u))
    assert centers == [
        (pytest.approx(rendered_v, abs=_TOLERANCE_PX), pytest.approx(rendered_u, abs=_TOLERANCE_PX))
    ]


_TEMPLATE_PAYLOADS: list[tuple[str, NavFeatureType]] = [
    ('body', NavFeatureType.BODY_DISC),
    ('ringlet', NavFeatureType.RING_ANNULUS),
]


@pytest.mark.parametrize(
    ('kind', 'feature_type'),
    _TEMPLATE_PAYLOADS,
    ids=[f'{k}-{t.name}' for k, t in _TEMPLATE_PAYLOADS],
)
def test_each_painted_template_lands_where_its_object_renders(
    kind: str, feature_type: NavFeatureType
) -> None:
    """A template's masked centroid, placed at its bbox and less the margin, is the render's."""
    features, margin_v, margin_u = _payloads(kind, feature_type)
    rendered_v, rendered_u = _rendered_centroid(kind)
    centroids = []
    for feature in features:
        assert feature.template_img is not None
        assert feature.template_mask is not None
        weights = np.where(feature.template_mask, feature.template_img, 0.0)
        centroid_v, centroid_u = _centroid_of_weights(weights)
        bbox_v, bbox_u = feature.geometry.bbox_extfov_vu[:2]
        centroids.append((centroid_v + bbox_v - margin_v, centroid_u + bbox_u - margin_u))
    assert len(centroids) == 1
    assert centroids[0] == pytest.approx((rendered_v, rendered_u), abs=_TOLERANCE_PX)
