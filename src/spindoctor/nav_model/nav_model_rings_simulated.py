"""Simulated-rings NavModel.

Predicts ring features for a simulated observation from the idealized
``ring_system`` view the information boundary exposes
(``obs.nav_params['ring_system']``): the shared projection geometry and the
navigable features' catalog orbits, shapes, and declared orbit
uncertainties.  Non-navigable features never reach this model (the boundary
filter drops them), and the planted per-feature ``orbit_error`` is a truth
key this model cannot see -- the navigator predicts catalog positions and
must absorb the planted misplacement honestly.

One model instance per navigable feature.  Each banded feature (ringlet)
emits a ``RING_ANNULUS`` template for the correlation path; every feature
emits one ``RING_EDGE`` polyline per predicted catalog boundary for the
distance-transform fit.  Rendering is delegated to
:mod:`spindoctor.nav_model.sim_ring`, whose projection and orbit math are
shared with the image-side renderer by design.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
import oops

from spindoctor.annotation import Annotations
from spindoctor.config import Config
from spindoctor.feature.feature import NavFeature, NavReliabilityBreakdown
from spindoctor.feature.feature_type import NavFeatureType
from spindoctor.feature.flags import RingAnnulusFlags, RingEdgeFlags
from spindoctor.feature.geometry import RingAnnulusGeometry, RingEdgePolyline
from spindoctor.nav_model.nav_model import NavModel
from spindoctor.nav_model.nav_model_rings_base import NavModelRingsBase
from spindoctor.nav_model.sim_ring import (
    PredictedRingEdge,
    PredictedRingFeature,
    predict_ring_feature,
)
from spindoctor.support.constants import PIXEL_CENTER_TO_CORNER_PX
from spindoctor.support.filters import NavFilterKind, NavFilterSpec
from spindoctor.support.time import now_dt
from spindoctor.support.types import NDArrayFloatType

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    from spindoctor.nav_orchestrator.nav_context import NavContext

__all__ = ['NavModelRingsSimulated']

# Base per-vertex ring-edge uncertainty for a simulated ring: the rendered
# edge is sharp and noise-free, so the floor reflects the one-pixel polyline
# sampling resolution.  A feature's declared_orbit_sigma raises the radial
# sigma above this floor (the navigator is entitled to its catalog error
# bars, never to the drawn orbit-error values).
_RING_EDGE_SIGMA_RADIAL_PX: float = 1.0
_RING_EDGE_SIGMA_ALONG_PX: float = 0.5
# A polyline whose max deviation from its best-fit line is below this is treated
# as straight (rank-1 constraint); a curved ring arc exceeds it and constrains
# the offset in both axes.
_RING_EDGE_FLAT_CURVATURE_PX: float = 1.0


def _ring_edge_is_straight(vertices_vu: NDArrayFloatType) -> bool:
    """Return True when the polyline's deviation from a line is below threshold.

    Computed by SVD of the centered vertices: the smaller singular direction's
    spread is the max perpendicular deviation from the best-fit line.
    """
    if vertices_vu.shape[0] < 3:
        return True
    centered = vertices_vu - vertices_vu.mean(axis=0, keepdims=True)
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    deviations = centered @ vt[1]
    return bool(float(np.max(np.abs(deviations))) <= _RING_EDGE_FLAT_CURVATURE_PX)


class NavModelRingsSimulated(NavModelRingsBase):
    """Ring NavModel predicted from a scene's idealized ring_system view.

    Parameters:
        name: Name of this model instance.
        obs: Observation containing image geometry.
        feature_name: The feature's name, used in metadata and labels.
        feature_params: The feature's idealized mapping from
            ``nav_params['ring_system']['features']`` (kind, shape keys,
            catalog ``orbit``, ``tau``, ``declared_orbit_sigma``).
        ring_system: The idealized ``ring_system`` block (shared
            ``geometry``, ``range_km``, ``km_per_pixel``, ``phase_deg``).
        config: Optional ``Config`` override.
    """

    def __init__(
        self,
        name: str,
        obs: oops.Observation,
        feature_name: str,
        feature_params: dict[str, Any],
        ring_system: dict[str, Any],
        *,
        config: Config | None = None,
    ) -> None:
        """Bind one navigable scene feature to a model instance.

        Parameters:
            name: Model instance name.
            obs: Observation carrying the filtered scene view.
            feature_name: The feature's name, upper-cased for labels.
            feature_params: The feature's idealized mapping from
                ``nav_params['ring_system']['features']``.
            ring_system: The idealized ``ring_system`` block.
            config: Optional ``Config`` override; ``None`` uses
                ``DEFAULT_CONFIG``.
        """
        super().__init__(name, obs, config=config)
        self._feature_name = feature_name.upper()
        self._feature_params: dict[str, Any] = dict(feature_params)
        self._ring_system: dict[str, Any] = dict(ring_system)
        # Scene time and ring epoch are idealized (catalog) values read from
        # the filtered view; obs stand-ins without nav_params get the same
        # defaults the renderer uses.
        nav_params = getattr(obs, 'nav_params', None) or {}
        self._time: float = float(nav_params.get('time', 0.0))
        self._epoch: float = float(nav_params.get('ring_epoch', 0.0))
        self._prediction: PredictedRingFeature | None = None
        self._predicted_center_vu: tuple[float, float] = (0.0, 0.0)
        self._subject_range_km: float = float(ring_system.get('range_km') or float('inf'))
        self._bbox_extfov_vu: tuple[int, int, int, int] = (0, 0, 0, 0)

    @classmethod
    def instances_for_obs(
        cls, obs: oops.Observation, *, config: Config | None = None
    ) -> list[NavModel]:
        """Build one simulated ring model per navigable ring_system feature.

        Reads the filtered idealized view (``obs.nav_params['ring_system']``),
        whose feature list carries exactly the navigable subset; returns an
        empty list for a real obs so the SPICE-backed ``NavModelRings``
        handles those instead.

        Parameters:
            obs: Observation snapshot.
            config: Configuration passed to the constructed instances.  None
                uses ``DEFAULT_CONFIG``.

        Returns:
            One ``NavModelRingsSimulated`` per navigable feature.
        """
        if not getattr(obs, 'is_simulated', False):
            return []
        nav_params = getattr(obs, 'nav_params', None)
        if not isinstance(nav_params, dict):
            return []
        ring_system = nav_params.get('ring_system')
        if not isinstance(ring_system, dict):
            return []
        out: list[NavModel] = []
        for index, feature in enumerate(ring_system.get('features') or []):
            if not isinstance(feature, dict):
                continue
            feature_name = str(feature.get('name', f'RING-FEATURE-{index + 1}'))
            out.append(
                cls(
                    f'rings_sim:{feature_name}',
                    obs,
                    feature_name,
                    feature,
                    ring_system,
                    config=config,
                )
            )
        return out

    def create_model(self) -> None:
        """Render the predicted feature and populate masks, annotations, metadata."""
        metadata: dict[str, Any] = {}
        start_time = now_dt()
        metadata['start_time'] = start_time.isoformat()
        metadata['end_time'] = None
        metadata['elapsed_time_sec'] = None
        self._metadata.clear()
        self._metadata.update(metadata)
        with self.log_section(f'CREATE SIMULATED RINGS MODEL FOR: {self._feature_name}'):
            self._render()
        end_time = now_dt()
        self._metadata['end_time'] = end_time.isoformat()
        self._metadata['elapsed_time_sec'] = (end_time - start_time).total_seconds()

    def _render(self) -> None:
        """Predict the feature on the extended FOV grid."""
        obs = self.obs
        data_size_v = int(obs.data_shape_v)
        data_size_u = int(obs.data_shape_u)
        ext_margin_v = int(obs.extfov_margin_v)
        ext_margin_u = int(obs.extfov_margin_u)
        geometry = self._ring_system.get('geometry') or {}
        center_v = float(geometry.get('center_v', data_size_v / 2.0)) + ext_margin_v
        center_u = float(geometry.get('center_u', data_size_u / 2.0)) + ext_margin_u
        opening_deg_obs = float(geometry.get('opening_deg_obs', 90.0))
        node_deg = float(geometry.get('node_deg', 0.0))
        shape = (data_size_v + 2 * ext_margin_v, data_size_u + 2 * ext_margin_u)
        self._logger.debug(
            'Simulated rings: predicting %r (kind=%s) at extfov center (%.2f, %.2f), '
            'B=%.2f deg, node=%.2f deg',
            self._feature_name,
            self._feature_params.get('kind'),
            center_v,
            center_u,
            opening_deg_obs,
            node_deg,
        )
        prediction = predict_ring_feature(
            shape,
            self._feature_params,
            center_v=center_v,
            center_u=center_u,
            opening_deg_obs=opening_deg_obs,
            node_deg=node_deg,
            time=self._time,
            epoch=self._epoch,
        )
        self._logger.debug(
            'Simulated rings: prediction complete, %d covered pixels, %d edges',
            int(np.count_nonzero(prediction.mask)),
            len(prediction.edges),
        )
        self._prediction = prediction
        # The scene states the ring system's center in pixel corner
        # coordinates, which is what ``predict_ring_feature`` renders against
        # (it puts the center of pixel i at i + 0.5).  The payload is pixel
        # centric, so the half pixel comes off here.
        self._predicted_center_vu = (
            center_v - PIXEL_CENTER_TO_CORNER_PX,
            center_u - PIXEL_CENTER_TO_CORNER_PX,
        )
        self._bbox_extfov_vu = (
            ext_margin_v,
            ext_margin_u,
            ext_margin_v + data_size_v,
            ext_margin_u + data_size_u,
        )

    def _declared_sigmas(self) -> NDArrayFloatType:
        """The declared 1-sigma orbit uncertainties of this feature.

        Returns:
            ``(sigma_a_px, sigma_ae_px, sigma_long_peri)`` with the pericenter
            sigma in radians, ordered like a predicted edge's
            ``orbit_partials_px`` columns.  A key the scene omits or sets to
            null contributes zero.
        """
        sigma = self._feature_params.get('declared_orbit_sigma') or {}
        return np.array(
            [
                float(sigma.get('sigma_a_px') or 0.0),
                float(sigma.get('sigma_ae_px') or 0.0),
                math.radians(float(sigma.get('sigma_long_peri_deg') or 0.0)),
            ],
            dtype=np.float64,
        )

    def _declared_orbit_sigma_px(self, edges: list[PredictedRingEdge]) -> float:
        """The declared 1-sigma radial orbit uncertainty of this feature, in image pixels.

        The semimajor-axis, radial-amplitude, and pericenter-longitude errors are
        independent, so a vertex's 1-sigma radial displacement is the quadrature
        sum of each sigma times the vertex's partial for that element.  For
        ``r ~ a - ae cos(theta - peri)`` on a face-on ring that variance is
        ``sigma_a**2 + cos**2 sigma_ae**2 + sin**2 (ae sigma_peri)**2``.  The
        value returned is its maximum over the feature's predicted vertices, the
        1-sigma envelope of a coherent orbit error over the visible edges.
        Adding the terms linearly would overstate that envelope by up to
        ``sqrt(2)``, and the value is squared straight into a tier-gating
        covariance.

        Parameters:
            edges: The feature's predicted edges.

        Returns:
            The declared sigma in image pixels; ``0.0`` when the scene declares
            no orbit uncertainty or no edge has vertices.
        """
        partials = [edge.orbit_partials_px for edge in edges if edge.vertices_vu.shape[0] > 0]
        if not partials:
            return 0.0
        displacement = np.concatenate(partials, axis=0) * self._declared_sigmas()
        return float(np.sqrt(np.sum(displacement * displacement, axis=1)).max())

    def _declared_radial_sigma_px(self, edge: PredictedRingEdge) -> NDArrayFloatType:
        """The per-vertex radial sigma of one edge from the declared orbit uncertainty.

        Each vertex adds, LINEARLY, the image-plane distance each declared
        1-sigma element error moves it: ``|dr/da| sigma_a + |dr/d(ae)| sigma_ae +
        |dr/dperi| sigma_peri``, about ``sigma_a + |cos(theta - peri)| sigma_ae +
        ae |sin(theta - peri)| sigma_peri`` on a face-on ring.  The sum is floored
        at the one-pixel polyline sampling resolution.

        This deliberately does NOT share the quadrature combine of
        :meth:`_declared_orbit_sigma_px`, which is the coherent displacement
        bound.  This value is the per-vertex prior the robust fit weights by
        ``1 / sigma**2``, so it only sets how the vertices weigh against one
        another; the quadrature value is the one squared into a tier gate.

        Parameters:
            edge: The predicted edge whose vertices get a sigma.

        Returns:
            One radial sigma per vertex, in pixels.
        """
        declared = np.sum(np.abs(edge.orbit_partials_px) * self._declared_sigmas(), axis=1)
        result: NDArrayFloatType = np.maximum(_RING_EDGE_SIGMA_RADIAL_PX, declared)
        return result

    def to_features(self, context: NavContext) -> list[NavFeature]:
        """Emit the ring features.

        Emits a ``RING_ANNULUS`` carrying the predicted coverage template
        (for the correlation path) whenever the feature is a banded kind
        with coverage on the ext-FOV.  The template payload convention
        (``compose_template_features``) is a postage stamp local to
        ``bbox_extfov_vu``, so the ext-FOV-sized prediction is cropped to
        the mask's tight bbox before emission.  Also emits one ``RING_EDGE``
        per predicted catalog boundary -- a per-vertex polyline with outward
        radial normals that ``RingEdgeNav`` fits against the image-edge
        distance transform, so a curved ring arc recovers the planted
        offset in both axes.
        """
        prediction = self._prediction
        if prediction is None:
            return []
        features: list[NavFeature] = []
        sigma_orbit_px = self._declared_orbit_sigma_px(prediction.edges)
        if prediction.template is not None and prediction.mask.any():
            mask_vs, mask_us = np.where(prediction.mask)
            bbox = (
                int(mask_vs.min()),
                int(mask_us.min()),
                int(mask_vs.max()) + 1,
                int(mask_us.max()) + 1,
            )
            template_img = prediction.template[bbox[0] : bbox[2], bbox[1] : bbox[3]].copy()
            template_mask = prediction.mask[bbox[0] : bbox[2], bbox[1] : bbox[3]].copy()
            # The predicted edges are the annulus fit's own radial geometry:
            # RingAnnulusNav derives the translation its NCC absorbs a coherent
            # radial orbit error into from their outward normals.  All edges of
            # a simulated feature share its single declared orbit sigma, so the
            # effective sigma is that scalar and the normals are their concat.
            annulus_normals = [
                edge.normals_vu for edge in prediction.edges if edge.vertices_vu.shape[0] > 0
            ]
            orbit_normals_vu = (
                np.concatenate(annulus_normals, axis=0)
                if annulus_normals
                else np.zeros((0, 2), dtype=np.float64)
            )
            features.append(
                NavFeature(
                    feature_id=f'ring_annulus:{self._feature_name}',
                    feature_type=NavFeatureType.RING_ANNULUS,
                    source_model=self.name,
                    geometry=RingAnnulusGeometry(
                        bbox_extfov_vu=bbox,
                        predicted_center_vu=self._predicted_center_vu,
                        orbit_normals_vu=orbit_normals_vu,
                        sigma_orbit_radial_px=sigma_orbit_px,
                    ),
                    subject_range_km=self._subject_range_km,
                    position_cov_px=None,
                    intensity_sigma_rel=0.0,
                    preferred_filter=NavFilterSpec(kind=NavFilterKind.NONE),
                    reliability=1.0,
                    reliability_reasons=NavReliabilityBreakdown(visible_arc_fraction=1.0),
                    usable_types=frozenset({NavFeatureType.RING_ANNULUS}),
                    flags=RingAnnulusFlags(
                        planet_name=self._feature_name,
                        constituent_edge_count=len(prediction.edges),
                    ),
                    template_img=template_img,
                    template_mask=template_mask,
                )
            )
        for edge in prediction.edges:
            n = edge.vertices_vu.shape[0]
            if n == 0:
                continue
            is_straight = _ring_edge_is_straight(edge.vertices_vu)
            features.append(
                NavFeature(
                    feature_id=f'ring_edge:{self._feature_name}:{edge.edge_type}',
                    feature_type=NavFeatureType.RING_EDGE,
                    source_model=self.name,
                    geometry=RingEdgePolyline(
                        vertices_vu=edge.vertices_vu,
                        normals_vu=edge.normals_vu,
                        sigma_radial_per_vertex_px=self._declared_radial_sigma_px(edge),
                        sigma_along_edge_per_vertex_px=np.full(
                            n, _RING_EDGE_SIGMA_ALONG_PX, dtype=np.float64
                        ),
                        is_straight_line=is_straight,
                        bbox_extfov_vu=self._bbox_extfov_vu,
                        sigma_orbit_radial_px=sigma_orbit_px,
                    ),
                    subject_range_km=self._subject_range_km,
                    position_cov_px=None,
                    intensity_sigma_rel=0.0,
                    preferred_filter=NavFilterSpec(kind=NavFilterKind.NONE),
                    reliability=1.0,
                    reliability_reasons=NavReliabilityBreakdown(visible_arc_fraction=1.0),
                    usable_types=frozenset({NavFeatureType.RING_EDGE}),
                    flags=RingEdgeFlags(
                        is_straight_line=is_straight,
                        edge_name=f'{self._feature_name}:{edge.edge_type}',
                        planet_name=self._feature_name,
                    ),
                )
            )
        return features

    def to_annotations(self, context: NavContext) -> Annotations:
        """Emit ring-edge polyline + label annotations."""
        prediction = self._prediction
        if prediction is None:
            return Annotations()
        edge_info_list = []
        for edge in prediction.edges:
            if not edge.mask.any():
                continue
            qualifier = '' if edge.edge_type == 'edge' else f'{edge.edge_type.upper()} '
            edge_info_list.append(
                (edge.mask, f'{self._feature_name} {qualifier}EDGE', edge.edge_type)
            )
        return self._create_edge_annotations(self.obs, edge_info_list, prediction.mask)
