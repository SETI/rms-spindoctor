"""Catalog-driven body NavModel.

Renders one body's predicted appearance from SPICE, classifies it
against the design's emission rules (limb arc vs. body disc vs. blob vs.
terminator), and emits one ``NavFeature`` per surviving feature type.

The pipeline:

1. Builds an oversampled meshgrid around the predicted body bounding
   box so the limb silhouette is anti-aliased.
2. Extracts the limb and terminator polylines from the discrete
   silhouette masks.
3. Looks up the per-body shape parameters via
   :func:`spindoctor.nav_model.body_shape.load_body_shape`, which merges the
   operator-curated ``config_220_body_shape.yaml`` over the hard-coded
   ``BODY_SHAPE_TABLE`` profiles.
4. Decides which features to emit by computing
   ``limb_uncertainty_px`` and the ``visible_lit_fraction`` /
   ``overflow_fraction`` quantities the design specifies.

The feature-by-feature emission rules:

- ``LIMB_ARC`` is emitted when ``limb_uncertainty_px <=
  LIMB_ARC_MAX_UNCERTAINTY_PX`` and there are surviving limb vertices.
- ``BODY_BLOB`` is emitted when the predicted disc diameter is at
  least ``max(BODY_BLOB_MIN_DIAMETER_PX, shape.min_blob_diameter_px)``
  *and* the limb uncertainty is too high for ``LIMB_ARC`` *and* the
  rendered silhouette contains at least one lit pixel.  A body whose
  silhouette is entirely in shadow has zero photometric signal to
  centroid, so it emits no blob (only the geometric features, per the
  dev guide's Restrictions).  The per-body shape floor can override
  the global default upward but not downward.
- ``BODY_DISC`` is emitted alongside ``LIMB_ARC`` when the body fits
  inside the FOV with at least ``BODY_DISC_MIN_VISIBLE_LIT_FRACTION``
  of its lit side visible and ``overflow_fraction`` below
  ``BODY_DISC_MAX_OVERFLOW_FRACTION``.
- ``TERMINATOR_ARC`` is emitted when the terminator polyline has at
  least ``TERMINATOR_MIN_VERTICES`` surviving vertices and the
  phase-angle factor (``sin(phase_angle)``) is above
  ``TERMINATOR_MIN_PHASE_FACTOR``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import polymath
from oops import Meshgrid
from oops.backplane import Backplane

from spindoctor.annotation import Annotations
from spindoctor.config import DEFAULT_CONFIG, IMAGE_LOGGER, Config
from spindoctor.feature.constants import (
    INCIDENCE_FACTOR_ANGLE_CAP_DEG,
    INCIDENCE_FACTOR_CLIP_DEG,
    MAX_INCIDENCE_FACTOR_CAP,
)
from spindoctor.feature.feature import NavFeature, NavReliabilityBreakdown
from spindoctor.feature.feature_type import NavFeatureType
from spindoctor.feature.flags import (
    BodyDiscFlags,
    LimbArcFlags,
    TerminatorArcFlags,
)
from spindoctor.feature.geometry import (
    BodyDiscGeometry,
    LimbPolyline,
    TerminatorPolyline,
)
from spindoctor.nav_model.body_shape import BodyShape, load_body_shape
from spindoctor.nav_model.nav_model import NavModel
from spindoctor.nav_model.nav_model_body_base import (
    BODY_BLOB_MIN_DIAMETER_PX,
    NavModelBodyBase,
    _sigmoid,
)
from spindoctor.nav_model.silhouette_probe import refine_polyline_vertices
from spindoctor.nav_model.stars.predicted_snr import psf_sigma_px
from spindoctor.obs import ObsSnapshot
from spindoctor.support.constants import HALFPI
from spindoctor.support.image import filter_downsample, shift_array
from spindoctor.support.time import now_dt
from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    from oops import Observation

    from spindoctor.nav_orchestrator.nav_context import NavContext

from spindoctor.support.filters import NavFilterKind, NavFilterSpec

__all__ = [
    'BODY_BLOB_MIN_DIAMETER_PX',
    'BODY_DISC_MAX_OVERFLOW_FRACTION',
    'BODY_DISC_MIN_VISIBLE_LIT_FRACTION',
    'BODY_POSITION_SLOP_FRAC',
    'LIMB_ARC_MAX_UNCERTAINTY_PX',
    'TERMINATOR_MIN_PHASE_FACTOR',
    'TERMINATOR_MIN_VERTICES',
    'TITAN_BODY_NAME',
    'NavModelBody',
    'bodies_in_extfov',
    'limb_reliability',
    'occluder_mask_for_body',
    'shape_features_suppressed',
    'terminator_reliability',
]


BODY_POSITION_SLOP_FRAC: float = 0.05
"""Inflation factor for the body bbox before clipping.

The ``oops.inventory`` bounding box is sometimes a half-pixel too small.
Inflating it by 5% before clipping into the extfov keeps anti-aliased
limb pixels from being lost on the boundary.
"""


LIMB_ARC_MAX_UNCERTAINTY_PX: float = 3.0
"""Cap on the limb normal-sigma at which LIMB_ARC remains useful.

Above this value the per-vertex normal uncertainty is too large for
the DT-based limb fit; the extractor switches to ``BODY_BLOB`` so the
brightness-weighted-centroid technique still has something to work
with.  The numeric value is a config default pending calibration
against the operator-curated image library.
"""


LIMB_ARC_MIN_VERTICES: int = 30
"""Minimum limb-polyline vertices to emit a LIMB_ARC.

Matches ``BodyLimbNav``'s ``min_arc_vertices`` feasibility floor
(``config_510_techniques.yaml``): an arc the technique is guaranteed to
reject cannot constrain any fit, so emitting it only starves the body of
its BODY_BLOB feature.  The uncertainty cap above cannot stand in for
this check -- the per-vertex sigma scales with the shape residual in
*pixels*, so a distant small body (a few-km residual mapped through a
long range) passes the uncertainty test precisely because it is tiny,
which is when the limb fit is least usable.  Below this floor the
extractor falls through to ``BODY_BLOB``.
"""


BODY_DISC_MIN_VISIBLE_LIT_FRACTION: float = 0.4
"""Minimum lit-and-in-FOV fraction for BODY_DISC emission.

Below 40% of the lit hemisphere visible, the disc match is too
asymmetric to be useful; BODY_BLOB or LIMB_ARC carries the load.
"""


BODY_DISC_MAX_OVERFLOW_FRACTION: float = 0.3
"""Maximum overflow fraction for BODY_DISC emission.

A body whose disc is more than 30% off-frame loses too much template
support for the correlation peak to be sharp.
"""


TERMINATOR_MIN_VERTICES: int = 8
"""Minimum surviving vertices for TERMINATOR_ARC emission."""


TERMINATOR_MIN_PHASE_FACTOR: float = 0.05
"""Minimum ``sin(phase_angle)`` for TERMINATOR_ARC emission.

Below sin(phase) ~= 0.05 (phase < 3 deg) the terminator is too close to
the limb to be photometrically distinguishable.
"""


TERMINATOR_PHOTOMETRIC_SOFTNESS_COEFF: float = 0.5
"""Photometric-softness coefficient in the terminator normal-sigma.

Scales the per-vertex limb-softness length (``psf_sigma_px * km/px``)
into an additional position-uncertainty term for terminator arcs, where
the gradual brightness roll-off broadens the apparent edge beyond the
geometric terminator.  The numeric value is a config default pending
calibration against the operator-curated image library (tracked with the
other body thresholds by issue #118 / CODE-NAV-MODEL-002).
"""


TITAN_BODY_NAME: str = 'TITAN'
"""SPICE name of the one body handled as a special opaque-atmosphere case.

Titan's thick haze hides the surface and its visible limb is the haze top,
wavelength-dependent and hundreds of km above the ground, so ellipsoid
limb / terminator / disc navigation is systematically wrong rather than
merely noisy; at high phase Titan is not even a circle.  Titan builds no
shape-based ``NavModelBody`` --
:class:`~spindoctor.nav_model.nav_model_titan.NavModelTitan` emits a haze
envelope feature instead, navigated by solar symmetry rather than by shape.
Titan's atmosphere is unique (transparent in some wavelengths), so it is
handled as a deliberate special case; its handling does not generalize to
other thick-atmosphere bodies such as Venus.
"""


def bodies_in_extfov(
    obs: Observation, *, config: Config | None = None
) -> list[tuple[str, dict[str, Any]]]:
    """Return ``(body_name, inventory_entry)`` for each body inside the extfov.

    Queries ``obs.inventory`` once with the planet plus its configured
    satellites and keeps every body whose ``inventory_body_in_extfov``
    predicate fires.  Shared by the shape-based body model and the Titan
    model so both select from the same in-FOV body set.

    Parameters:
        obs: Observation snapshot.
        config: Configuration whose satellite catalog decides which bodies are
            considered; ``None`` uses ``DEFAULT_CONFIG``.

    Returns:
        List of ``(body_name, inventory_entry)`` pairs in planet-then-satellite
        order; empty when the observation exposes no usable inventory.
    """
    cfg = config if config is not None else DEFAULT_CONFIG
    planet = getattr(obs, 'closest_planet', None)
    if planet is None:
        return []
    body_list: list[str] = [planet, *list(cfg.satellites(planet))]
    inventory_method = getattr(obs, 'inventory', None)
    if not callable(inventory_method):
        return []
    try:
        inv = inventory_method(body_list, return_type='full')
    except ValueError:
        # oops raises ValueError for a body it cannot resolve in the scene;
        # that is a recoverable "nothing in the extfov" outcome.  Let
        # TypeError / AttributeError propagate -- they signal a malformed obs
        # or a genuine bug, not an empty FOV.
        return []
    in_extfov = getattr(obs, 'inventory_body_in_extfov', None)
    if not callable(in_extfov):
        return []
    out: list[tuple[str, dict[str, Any]]] = []
    for body_name in body_list:
        entry = inv.get(body_name)
        if entry is None:
            continue
        if not in_extfov(entry):
            continue
        out.append((body_name, entry))
    return out


def occluder_mask_for_body(
    restr_bp: Backplane,
    body_name: str,
    siblings: list[tuple[str, float]],
    subject_range_km: float,
    *,
    oversample_v: int,
    oversample_u: int,
) -> NDArrayBoolType | None:
    """Union silhouette of nearer sibling bodies over a body's bbox grid.

    A sibling occludes the subject body only when its inventory center range
    is strictly nearer; the per-pixel depth test in
    :meth:`oops.backplane.Backplane.where_in_front` then decides which pixels
    the nearer body actually hides.  The result is downsampled to the same
    discrete grid the caller's masks live on.

    A backplane failure on any occluder degrades to no occlusion for that
    occluder (the caller's arc / template / mask stays untrimmed) rather than
    aborting the render.

    Shared by the shape-based body model and the haze model so both derive
    body-body occlusion from one implementation; the caller supplies the
    backplane it already built, which is what keeps the two callers' results
    identical to what each would compute alone.

    Parameters:
        restr_bp: Backplane over the subject body's oversampled bbox
            meshgrid.
        body_name: SPICE name of the subject body.
        siblings: ``(body_name, range_km)`` for the other bodies in the FOV.
        subject_range_km: Center range of the subject body in kilometers;
            only strictly nearer siblings are considered.
        oversample_v: Vertical oversample factor of ``restr_bp``'s meshgrid.
        oversample_u: Horizontal oversample factor of that meshgrid.

    Returns:
        The bbox-local boolean occluder mask, or ``None`` when no sibling is
        nearer or none hides any pixel (the common case costs nothing).
    """
    nearer = [name for name, rng in siblings if rng < subject_range_km]
    if not nearer:
        return None
    occluder: NDArrayBoolType | None = None
    for sibling_name in nearer:
        try:
            hidden = restr_bp.where_in_front(sibling_name, body_name)
            hidden_over = hidden.mvals.filled(False).astype(bool)
        except Exception:
            IMAGE_LOGGER.warning(
                'Body %s: failed to compute occlusion by %s; arc / template left untrimmed',
                body_name,
                sibling_name,
                exc_info=True,
            )
            continue
        hidden_local = (
            filter_downsample(hidden_over.astype(np.float64), oversample_v, oversample_u) >= 0.5
        )
        occluder = hidden_local if occluder is None else (occluder | hidden_local)
    if occluder is None or not occluder.any():
        return None
    return occluder


@dataclass(frozen=True)
class _PolylineSampler:
    """Bundle of sampled limb / terminator polyline data.

    Encapsulates the per-vertex outputs of the silhouette extraction so
    multiple feature emitters can share the same data without
    re-running the discrete-mask traversal.
    """

    vertices_vu: NDArrayFloatType
    normals_vu: NDArrayFloatType
    incidence_rad: NDArrayFloatType
    km_per_pixel: NDArrayFloatType
    total_vertices: int
    """Ridge vertices found before per-vertex drops (resolution / future shadow).

    ``vertices_vu`` holds only the survivors; ``total_vertices`` is the count
    of the original ridge so :func:`_visible_arc_fraction` can report the
    surviving fraction rather than a constant.
    """


class NavModelBody(NavModelBodyBase):
    """Catalog-driven body NavModel.

    Parameters:
        name: Model instance name (e.g. ``'body:MIMAS'``).
        obs: Observation snapshot.
        body_name: SPICE body name.
        inventory: Optional pre-computed inventory entry; pulled from
            ``obs.inventory`` on demand otherwise.
        siblings: ``(body_name, range_km)`` for the other bodies in the FOV.
            A sibling whose center range is strictly nearer occludes this
            body's predicted limb / terminator arcs and disc template;
            ``None`` (the common single-body case) means no sibling occludes.
        config: Optional ``Config`` override.
    """

    _abstract = False

    def __init__(
        self,
        name: str,
        obs: Observation,
        body_name: str,
        *,
        inventory: dict[str, Any] | None = None,
        siblings: list[tuple[str, float]] | None = None,
        config: Config | None = None,
    ) -> None:
        super().__init__(name, obs, config=config)
        self._body_name = body_name.upper()
        self._inventory = inventory
        self._siblings: list[tuple[str, float]] = list(siblings or [])
        self._occluder_mask: NDArrayBoolType | None = None
        self._model_img: NDArrayFloatType | None = None
        self._body_mask: NDArrayBoolType | None = None
        self._limb_mask: NDArrayBoolType | None = None
        self._terminator_mask: NDArrayBoolType | None = None
        self._limb_sampler: _PolylineSampler | None = None
        self._terminator_sampler: _PolylineSampler | None = None
        self._km_per_pixel_at_limb: float = 0.0
        self._predicted_diameter_px: float = 0.0
        self._predicted_center_vu: tuple[float, float] = (0.0, 0.0)
        self._bbox_extfov_vu: tuple[int, int, int, int] = (0, 0, 0, 0)
        self._subject_range_km: float = float('inf')
        self._visible_lit_fraction: float = 0.0
        self._lit_pixel_count: int = 0
        self._overflow_fraction: float = 1.0
        self._phase_angle_factor: float = 0.0

    @classmethod
    def instances_for_obs(cls, obs: Observation, *, config: Config | None = None) -> list[NavModel]:
        """Return one NavModelBody per body whose bbox lies inside extfov.

        Selects every in-FOV body via :func:`bodies_in_extfov` and constructs
        a NavModel for each.  Titan is excluded: its opaque haze hides the
        surface, so ellipsoid-shape navigation is systematically wrong, and
        :class:`~spindoctor.nav_model.nav_model_titan.NavModelTitan` handles
        it instead.

        Parameters:
            obs: Observation snapshot.
            config: Configuration whose satellite catalog decides which bodies
                are considered; also passed to the constructed instances.  None
                uses ``DEFAULT_CONFIG``.

        Returns:
            One ``NavModelBody`` per non-Titan body present in the extfov.
        """
        # Simulated obs use the sim-params-driven NavModelBodySimulated instead.
        if getattr(obs, 'is_simulated', False):
            return []
        if config is None:
            config = DEFAULT_CONFIG
        in_extfov = bodies_in_extfov(obs, config=config)
        ranges = {
            name.upper(): float(entry.get('range', float('inf'))) for name, entry in in_extfov
        }
        out: list[NavModel] = []
        for body_name, entry in in_extfov:
            if body_name.upper() == TITAN_BODY_NAME:
                continue
            # Every other in-FOV body is a candidate occluder (Titan included:
            # it is not navigated as a shape, but a nearer Titan still hides a
            # moon behind it).  The per-body render decides per pixel which are
            # actually nearer via the backplane depth test.
            siblings = [
                (other.upper(), ranges[other.upper()])
                for other, _ in in_extfov
                if other.upper() != body_name.upper()
            ]
            out.append(
                cls(
                    f'body:{body_name}',
                    obs,
                    body_name,
                    inventory=entry,
                    siblings=siblings,
                    config=config,
                )
            )
        return out

    def create_model(self) -> None:
        """Render the silhouette, masks, and polylines used by ``to_features``."""
        start_time = now_dt()
        self._metadata.clear()
        self._metadata['start_time'] = start_time.isoformat()
        self._metadata['end_time'] = None
        self._metadata['elapsed_time_sec'] = None
        self._metadata['body_name'] = self._body_name
        with self.log_section(f'CREATE BODY MODEL FOR: {self._body_name}'):
            self._render()
            self._log_geometry_summary()
            end_time = now_dt()
            self._metadata['end_time'] = end_time.isoformat()
            self._metadata['elapsed_time_sec'] = (end_time - start_time).total_seconds()
            self._logger.info('Model created in %.3f s', self._metadata['elapsed_time_sec'])

    def _log_geometry_summary(self) -> None:
        """Emit INFO-level geometry summary plus DEBUG-level bbox detail."""
        meta = self._metadata
        self._logger.info(
            'Subsolar (lon, lat) = (%.2f, %.2f) deg; '
            'subobserver (lon, lat) = (%.2f, %.2f) deg; phase = %.2f deg',
            meta.get('sub_solar_lon_deg', float('nan')),
            meta.get('sub_solar_lat_deg', float('nan')),
            meta.get('sub_observer_lon_deg', float('nan')),
            meta.get('sub_observer_lat_deg', float('nan')),
            meta.get('phase_angle_deg', float('nan')),
        )
        self._logger.info(
            'Subject range = %.0f km; predicted diameter = %.2f px; km/px at limb = %.4f',
            self._subject_range_km,
            self._predicted_diameter_px,
            self._km_per_pixel_at_limb,
        )
        self._logger.info(
            'Visible-lit fraction = %.3f; silhouette overflow = %.3f; guaranteed-visible = %s',
            self._visible_lit_fraction,
            self._overflow_fraction,
            meta.get('guaranteed_visible_in_fov', False),
        )
        self._logger.debug(
            'Bbox (extfov vu) = %s; bbox area = %.1f px; size_ok = %s',
            self._bbox_extfov_vu,
            meta.get('bbox_area_px', 0.0),
            meta.get('size_ok', False),
        )

    def _render(self) -> None:
        """Populate masks, polyline samplers, and metadata."""
        obs = self.obs
        ext_bp = obs.ext_bp
        body_name = self._body_name
        body_config = self._config.bodies
        if self._inventory is None:
            self._inventory = obs.inventory([body_name], return_type='full')[body_name]
        inventory = self._inventory

        sub_solar_lon = float(np.degrees(ext_bp.sub_solar_longitude(body_name).vals))
        sub_solar_lat = float(np.degrees(ext_bp.sub_solar_latitude(body_name).vals))
        sub_observer_lon = float(np.degrees(ext_bp.sub_observer_longitude(body_name).vals))
        sub_observer_lat = float(np.degrees(ext_bp.sub_observer_latitude(body_name).vals))
        phase_angle_deg = float(np.degrees(ext_bp.center_phase_angle(body_name).vals))
        self._metadata['sub_solar_lon_deg'] = sub_solar_lon
        self._metadata['sub_solar_lat_deg'] = sub_solar_lat
        self._metadata['sub_observer_lon_deg'] = sub_observer_lon
        self._metadata['sub_observer_lat_deg'] = sub_observer_lat
        self._metadata['phase_angle_deg'] = phase_angle_deg
        self._phase_angle_factor = float(np.sin(np.radians(phase_angle_deg)))
        self._subject_range_km = float(inventory.get('range', float('inf')))

        bb_area = float(inventory['u_pixel_size'] * inventory['v_pixel_size'])
        self._metadata['bbox_area_px'] = bb_area
        self._metadata['size_ok'] = bool(bb_area >= body_config.min_bounding_box_area)

        u_min_unc = int(inventory['u_min_unclipped'])
        u_max_unc = int(inventory['u_max_unclipped'])
        v_min_unc = int(inventory['v_min_unclipped'])
        v_max_unc = int(inventory['v_max_unclipped'])
        u_slop = int((u_max_unc - u_min_unc) * BODY_POSITION_SLOP_FRAC)
        v_slop = int((v_max_unc - v_min_unc) * BODY_POSITION_SLOP_FRAC)
        u_min = u_min_unc - u_slop
        u_max = u_max_unc + u_slop
        v_min = v_min_unc - v_slop
        v_max = v_max_unc + v_slop
        u_min, v_min = obs.clip_extfov(u_min, v_min)
        u_max, v_max = obs.clip_extfov(u_max, v_max)
        if u_min == u_max == obs.extfov_u_max:
            u_min -= 1
        if u_min == u_max == obs.extfov_u_min:
            u_max += 1
        if v_min == v_max == obs.extfov_v_max:
            v_min -= 1
        if v_min == v_max == obs.extfov_v_min:
            v_max += 1

        guaranteed_visible = (
            u_min >= obs.extfov_margin_u
            and u_max <= obs.data_shape_u - 1 - obs.extfov_margin_u
            and v_min >= obs.extfov_margin_v
            and v_max <= obs.data_shape_v - 1 - obs.extfov_margin_v
        )
        self._metadata['guaranteed_visible_in_fov'] = guaranteed_visible

        model_img, limb_mask, terminator_mask, body_mask, sampler_data = (
            self._build_backplane_model(u_min=u_min, u_max=u_max, v_min=v_min, v_max=v_max)
        )
        self._model_img = model_img
        self._body_mask = body_mask
        self._limb_mask = limb_mask
        self._terminator_mask = terminator_mask

        u_center_data = (u_min_unc + u_max_unc) / 2.0
        v_center_data = (v_min_unc + v_max_unc) / 2.0
        self._predicted_center_vu = (
            float(v_center_data + obs.extfov_margin_v),
            float(u_center_data + obs.extfov_margin_u),
        )
        diameter = max(
            float(inventory['u_pixel_size']),
            float(inventory['v_pixel_size']),
        )
        self._predicted_diameter_px = diameter
        self._bbox_extfov_vu = (
            int(v_min + obs.extfov_margin_v),
            int(u_min + obs.extfov_margin_u),
            int(v_max + obs.extfov_margin_v + 1),
            int(u_max + obs.extfov_margin_u + 1),
        )

        self._km_per_pixel_at_limb = sampler_data['km_per_pixel_mean']
        self._limb_sampler = sampler_data['limb_sampler']
        self._terminator_sampler = sampler_data['terminator_sampler']
        self._visible_lit_fraction = float(sampler_data['visible_lit_fraction'])
        self._lit_pixel_count = int(sampler_data['lit_pixel_count'])
        self._overflow_fraction = float(sampler_data['overflow_fraction'])
        self._metadata['km_per_pixel_at_limb'] = self._km_per_pixel_at_limb
        self._metadata['predicted_diameter_px'] = self._predicted_diameter_px
        self._metadata['visible_lit_fraction'] = self._visible_lit_fraction
        self._metadata['overflow_fraction'] = self._overflow_fraction

    def _build_backplane_model(
        self,
        *,
        u_min: int,
        u_max: int,
        v_min: int,
        v_max: int,
    ) -> tuple[
        NDArrayFloatType,
        NDArrayBoolType,
        NDArrayBoolType,
        NDArrayBoolType,
        dict[str, Any],
    ]:
        """Build the silhouette, limb / terminator masks, and samplers.

        Renders an oversampled Lambert silhouette over the body bbox,
        downsamples to the extfov grid, extracts the discrete limb and
        terminator masks, and samples per-vertex polyline data.

        Parameters:
            u_min, u_max, v_min, v_max: Body bounding box in extfov
                coordinates (already clipped).

        Returns:
            ``(model_img, limb_mask, terminator_mask, body_mask, info)``
            where ``info`` carries the polyline samplers, the mean
            km/pixel at the limb, and the visibility / overflow
            fractions.
        """
        obs = self.obs
        body_name = self._body_name
        body_config = self._config.bodies
        inventory = self._inventory
        assert inventory is not None  # populated in _render

        oversample_u = max(
            int(
                np.floor(
                    body_config.oversample_edge_limit / max(np.ceil(inventory['u_pixel_size']), 1)
                )
            ),
            1,
        )
        oversample_v = max(
            int(
                np.floor(
                    body_config.oversample_edge_limit / max(np.ceil(inventory['v_pixel_size']), 1)
                )
            ),
            1,
        )
        oversample_u = min(oversample_u, body_config.oversample_maximum)
        oversample_v = min(oversample_v, body_config.oversample_maximum)
        self._logger.debug(
            'Body %s: backplane oversample (u, v) = (%d, %d); edge_limit = %s, max = %s',
            self._body_name,
            oversample_u,
            oversample_v,
            body_config.oversample_edge_limit,
            body_config.oversample_maximum,
        )
        restr_u_min = u_min + 1.0 / (2 * oversample_u)
        restr_u_max = u_max + 1 - 1.0 / (2 * oversample_u)
        restr_v_min = v_min + 1.0 / (2 * oversample_v)
        restr_v_max = v_max + 1 - 1.0 / (2 * oversample_v)
        restr_meshgrid = Meshgrid.for_fov(
            obs.fov,
            origin=(restr_u_min, restr_v_min),
            limit=(restr_u_max, restr_v_max),
            oversample=(oversample_u, oversample_v),
            swap=True,
        )
        restr_bp = Backplane(obs, meshgrid=restr_meshgrid)

        oversampled_incidence_mvals = restr_bp.incidence_angle(body_name).mvals
        downsampled_incidence_mvals = filter_downsample(
            oversampled_incidence_mvals, oversample_v, oversample_u
        )
        incidence_scalar = polymath.Scalar(downsampled_incidence_mvals)

        body_mask_invalid = incidence_scalar.expand_mask().mask
        body_mask_valid = ~body_mask_invalid
        limb_mask_neighbor = (
            shift_array(body_mask_invalid, (-1, 0))
            | shift_array(body_mask_invalid, (1, 0))
            | shift_array(body_mask_invalid, (0, -1))
            | shift_array(body_mask_invalid, (0, 1))
        )
        # Terminator mask: pixels whose incidence crosses 90 deg
        incidence_vals = incidence_scalar.vals
        is_lit = (incidence_vals < HALFPI) & body_mask_valid
        is_dark = (incidence_vals >= HALFPI) & body_mask_valid
        # The geometric limb is the silhouette boundary (lit + unlit).
        # The DT-fit LIMB_ARC technique can only see the *lit* limb in
        # the image — the unlit limb merges into dark space and has no
        # gradient.  Including unlit-side vertices in the polyline gives
        # the LM no useful signal there, but the high sigma assigned to
        # those vertices widens their Tukey inlier band so they happily
        # lock onto the terminator (a strong DT minimum) when the LM
        # shifts.  Filtering to lit-side vertices removes that hazard
        # at the source.  See Cassini Tethys N1574928113 for the
        # calibration case.
        geometric_limb_mask: NDArrayBoolType = body_mask_valid & limb_mask_neighbor
        limb_mask_local: NDArrayBoolType = geometric_limb_mask & is_lit
        # A pixel is on the terminator if it is lit and any neighbour is dark.
        terminator_local: NDArrayBoolType = is_lit & (
            shift_array(is_dark, (-1, 0))
            | shift_array(is_dark, (1, 0))
            | shift_array(is_dark, (0, -1))
            | shift_array(is_dark, (0, 1))
        )

        if not body_mask_valid.any() or incidence_vals[body_mask_valid].min() >= HALFPI:
            local_model: NDArrayFloatType = np.zeros_like(body_mask_valid, dtype=np.float64)
            local_model[body_mask_valid] = 0.01
        else:
            if body_config.use_lambert:
                lambert_oversampled = restr_bp.lambert_law(body_name).mvals.filled(0.0)
                local_model = filter_downsample(lambert_oversampled, oversample_v, oversample_u)
                local_model = local_model + 0.05
                local_model[body_mask_invalid] = 0.0
            else:
                local_model = body_mask_valid.astype(np.float64)
            if body_config.use_albedo and body_name in body_config.geometric_albedo:
                albedo = float(body_config.geometric_albedo[body_name])
                local_model = local_model * albedo

        # Promote to extfov-shaped arrays.
        ext_v0 = v_min + obs.extfov_margin_v
        ext_u0 = u_min + obs.extfov_margin_u
        v_size = local_model.shape[0]
        u_size = local_model.shape[1]
        model_img = obs.make_extfov_zeros()
        limb_mask = obs.make_extfov_false()
        body_mask = obs.make_extfov_false()
        terminator_mask = obs.make_extfov_false()
        v_slice = slice(ext_v0, ext_v0 + v_size)
        u_slice = slice(ext_u0, ext_u0 + u_size)
        model_img[v_slice, u_slice] = local_model
        limb_mask[v_slice, u_slice] = limb_mask_local
        terminator_mask[v_slice, u_slice] = terminator_local
        body_mask[v_slice, u_slice] = body_mask_valid

        # km/pixel is queried at the oversampled grid and downsampled to
        # match the masks.  When the predicted silhouette is empty the
        # backplane query is skipped (it would return masked values
        # anyway) and the downsampled-shape zeros array is used
        # directly — feeding it back through ``filter_downsample`` would
        # try to downsample an already-downsampled array, asserting on
        # the (downsampled-)shape vs oversample divisibility.
        if body_mask_valid.any():
            km_per_pixel_arr = restr_bp.resolution(body_name).mvals.filled(0.0)
            km_per_pixel_local = filter_downsample(km_per_pixel_arr, oversample_v, oversample_u)
        else:
            km_per_pixel_local = np.zeros_like(body_mask_valid, dtype=np.float64)

        # Body-body occlusion: every predicted pixel a nearer sibling body
        # hides is dropped from the limb / terminator polylines (so the DT fit
        # never chases an arc the image does not show) and, promoted to extfov
        # coordinates, is trimmed out of the disc template (so the correlator
        # does not score against disc brightness that is not there).
        occluder_local = occluder_mask_for_body(
            restr_bp,
            body_name,
            self._siblings,
            self._subject_range_km,
            oversample_v=oversample_v,
            oversample_u=oversample_u,
        )
        occluder_ext = obs.make_extfov_false()
        if occluder_local is not None:
            occluder_ext[v_slice, u_slice] = occluder_local & body_mask_valid
        self._occluder_mask = occluder_ext

        limb_sampler = _build_polyline_sampler(
            local_mask=limb_mask_local,
            region_mask=body_mask_valid,
            incidence_local=incidence_vals,
            km_per_pixel_local=km_per_pixel_local,
            occluder_local=occluder_local,
            ext_v0=ext_v0,
            ext_u0=ext_u0,
        )
        terminator_sampler = _build_polyline_sampler(
            local_mask=terminator_local,
            region_mask=is_lit,
            incidence_local=incidence_vals,
            km_per_pixel_local=km_per_pixel_local,
            occluder_local=occluder_local,
            ext_v0=ext_v0,
            ext_u0=ext_u0,
        )
        # Move each ridge vertex from its integer pixel center onto the probed
        # sub-pixel boundary along its outward normal.  Without this the
        # polyline is quantized to the pixel grid, so a sub-pixel pointing
        # change re-rasterizes the ridge instead of translating it and the
        # DT-based limb / terminator fits inherit a sub-pixel-phase-dependent
        # bias of up to half a pixel that breaks shift equivariance (#447).
        limb_sampler = _refine_sampler_to_boundary(
            obs, body_name, limb_sampler, region='silhouette'
        )
        terminator_sampler = _refine_sampler_to_boundary(
            obs, body_name, terminator_sampler, region='lit'
        )

        sensor_v0 = obs.extfov_margin_v
        sensor_v1 = obs.extfov_margin_v + obs.data_shape_v
        sensor_u0 = obs.extfov_margin_u
        sensor_u1 = obs.extfov_margin_u + obs.data_shape_u
        in_sensor = np.zeros_like(body_mask, dtype=bool)
        in_sensor[sensor_v0:sensor_v1, sensor_u0:sensor_u1] = True
        lit_mask = body_mask.copy()
        if local_model.size > 0:
            lit_arr = obs.make_extfov_false()
            lit_arr[v_slice, u_slice] = is_lit
            lit_mask = lit_arr
        body_total = int(np.count_nonzero(body_mask))
        body_visible = int(np.count_nonzero(body_mask & in_sensor))
        # ``visible_lit_fraction`` measures the fraction of the *whole
        # predicted disc* (lit + dark together) whose cos(incidence) >= 0
        # AND which lies inside the sensor FOV — not the lit hemisphere
        # alone, which would always score ~1.0 for a fully-in-frame
        # body and lose discriminating power for the BODY_DISC gate.
        # NOTE: the name is narrower than the quantity — the denominator is
        # the whole disc, so this deliberately falls with phase (a thin
        # high-phase crescent scores low even when fully framed) as well as
        # with partial framing.  Both regimes make the disc-correlation
        # template poor, which is exactly what the 0.4 gate screens out;
        # the phase coupling is intentional, not a normalisation bug.
        # A lit pixel a nearer sibling hides is not usable disc-template
        # support, so it leaves the numerator while the denominator stays the
        # whole predicted disc: at deep overlap ``visible_lit_fraction`` falls
        # toward the surviving crescent, which is what the BODY_DISC gate and
        # reliability consume.
        lit_visible_in_fov = int(np.count_nonzero(lit_mask & in_sensor & ~occluder_ext))
        visible_lit_fraction = lit_visible_in_fov / max(body_total, 1)
        overflow_fraction = 1.0 - (body_visible / max(body_total, 1))

        if limb_mask_local.any() and km_per_pixel_local[limb_mask_local].size:
            km_per_pixel_mean = float(np.mean(km_per_pixel_local[limb_mask_local]))
        else:
            km_per_pixel_mean = 0.0

        info: dict[str, Any] = {
            'limb_sampler': limb_sampler,
            'terminator_sampler': terminator_sampler,
            'km_per_pixel_mean': km_per_pixel_mean,
            'visible_lit_fraction': visible_lit_fraction,
            'lit_pixel_count': int(np.count_nonzero(lit_mask)),
            'overflow_fraction': overflow_fraction,
        }
        return model_img, limb_mask, terminator_mask, body_mask, info

    def to_features(self, context: NavContext) -> list[NavFeature]:
        """Emit the body's NavFeatures per the design's gate rules."""
        with self.log_section(f'EMIT BODY FEATURES: {self._body_name}'):
            if self._body_mask is None:
                self._logger.debug('body_mask is None — emitting no features')
                return []
            shape = load_body_shape(self._body_name, config=self._config)
            features: list[NavFeature] = []
            limb_uncertainty_px = self._limb_uncertainty_px(shape)
            limb_vertices = (
                int(self._limb_sampler.vertices_vu.shape[0])
                if self._limb_sampler is not None
                else 0
            )
            self._logger.debug(
                'ellipsoid_rms_residual = %.3f km; limb_uncertainty = %.3f px '
                '(arc max %.3f); limb vertices = %d (min %d); shape_class = %s',
                shape.ellipsoid_rms_residual_km,
                limb_uncertainty_px,
                LIMB_ARC_MAX_UNCERTAINTY_PX,
                limb_vertices,
                LIMB_ARC_MIN_VERTICES,
                shape.shape_class_hint,
            )
            suppress_shape_features = shape_features_suppressed(
                shape, self._predicted_diameter_px, config=self._config
            )
            if suppress_shape_features:
                self._logger.info(
                    'Body %s is highly_irregular and resolved (predicted diameter '
                    '%.2f px > %.2f px): suppressing LIMB_ARC / TERMINATOR_ARC / '
                    'BODY_DISC; BODY_BLOB may still emit',
                    self._body_name,
                    self._predicted_diameter_px,
                    math.sqrt(float(self._config.bodies.min_bounding_box_area)),
                )
            limb_arc_emitted = False
            blob_min_px = max(BODY_BLOB_MIN_DIAMETER_PX, shape.min_blob_diameter_px)
            blob_gate_ok = self._predicted_diameter_px >= blob_min_px and self._lit_pixel_count > 0
            if (
                not suppress_shape_features
                and limb_vertices >= LIMB_ARC_MIN_VERTICES
                and limb_uncertainty_px <= LIMB_ARC_MAX_UNCERTAINTY_PX
            ):
                assert self._limb_sampler is not None
                features.append(
                    _build_limb_arc(
                        body_name=self._body_name,
                        sampler=self._limb_sampler,
                        shape=shape,
                        bbox=self._bbox_extfov_vu,
                        subject_range_km=self._subject_range_km,
                        psf_sigma_px=psf_sigma_px(self.obs.star_psf()),
                        source_model=self.name,
                    )
                )
                limb_arc_emitted = True
            elif blob_gate_ok:
                features.append(self._build_blob_feature(shape, context=context))
            else:
                self._logger.debug(
                    'No limb/blob feature emitted: limb unusable (uncertainty %.3f vs '
                    'max %.3f, vertices %d vs min %d) and blob gate failed '
                    '(predicted_diameter %.3f vs min %.3f, lit pixels %d)',
                    limb_uncertainty_px,
                    LIMB_ARC_MAX_UNCERTAINTY_PX,
                    limb_vertices,
                    LIMB_ARC_MIN_VERTICES,
                    self._predicted_diameter_px,
                    blob_min_px,
                    self._lit_pixel_count,
                )

            # Witness BODY_BLOB alongside a geometric limb (the disc, when
            # emitted, only accompanies the limb).  The pose-free brightness
            # centroid is a cross-check the ensemble body-witness veto reads to
            # catch a geometric technique locked onto a mismatched shape.  It is
            # superseded in the fuse (BodyBlobNav.runs_as_witness), so it does
            # not dilute the geometric offset; only the veto consumes it.
            if limb_arc_emitted and blob_gate_ok:
                features.append(self._build_blob_feature(shape, context=context))

            if not suppress_shape_features and self._should_emit_disc(limb_arc_emitted):
                features.append(self._build_disc_feature(shape))

            terminator_feature = (
                None if suppress_shape_features else self._maybe_build_terminator(shape)
            )
            if terminator_feature is not None:
                features.append(terminator_feature)

            kinds = ', '.join(sorted({f.feature_type.name for f in features})) or 'none'
            self._logger.info('Emitted %d feature(s) [%s]', len(features), kinds)
            return features

    def to_annotations(self, context: NavContext) -> Annotations:
        """Reuse the shared body annotation helper."""
        del context
        if self._model_img is None or self._body_mask is None or self._limb_mask is None:
            return Annotations()
        v_center, u_center = self._predicted_center_vu
        return self._create_annotations(
            round(u_center - self.obs.extfov_margin_u),
            round(v_center - self.obs.extfov_margin_v),
            self._model_img,
            self._limb_mask,
            self._body_mask,
        )

    def _limb_uncertainty_px(self, shape: BodyShape) -> float:
        """Return the design's ``limb_uncertainty_px`` for this body."""
        if self._km_per_pixel_at_limb <= 0.0:
            return float('inf')
        return shape.ellipsoid_rms_residual_km / self._km_per_pixel_at_limb

    def _should_emit_disc(self, limb_arc_emitted: bool) -> bool:
        """Return True when BODY_DISC should be emitted alongside other features."""
        if not limb_arc_emitted:
            return False
        if self._visible_lit_fraction < BODY_DISC_MIN_VISIBLE_LIT_FRACTION:
            return False
        return not self._overflow_fraction > BODY_DISC_MAX_OVERFLOW_FRACTION

    def _build_disc_feature(self, shape: BodyShape) -> NavFeature:
        """Construct the BODY_DISC feature (template + geometry + flags)."""
        assert self._model_img is not None
        assert self._body_mask is not None
        # ``compose_template_features`` expects the template payload to be a
        # postage-stamp sized to ``bbox_extfov_vu``.  ``self._model_img`` is
        # an extfov-shaped buffer with non-zero values only inside the body
        # bbox, so crop here to the body's bbox.  The .copy() detaches from
        # the parent so the feature can freeze its own array safely.
        v_min, u_min, v_max, u_max = self._bbox_extfov_vu
        template_img = self._model_img[v_min:v_max, u_min:u_max].copy()
        template_mask = self._body_mask[v_min:v_max, u_min:u_max].copy()
        # A nearer sibling body hides part of this disc; the image shows the
        # occluder there, not this body, so drop the occluded pixels from the
        # template.  Correlating against disc brightness that is not present
        # would be a coherent mismatch the correlator's robust machinery
        # absorbs rather than discounts.
        if self._occluder_mask is not None:
            occluded_bbox = self._occluder_mask[v_min:v_max, u_min:u_max]
            template_img[occluded_bbox] = 0.0
            template_mask[occluded_bbox] = False
        return NavFeature(
            feature_id=f'body_disc:{self._body_name}',
            feature_type=NavFeatureType.BODY_DISC,
            source_model=self.name,
            geometry=BodyDiscGeometry(
                bbox_extfov_vu=self._bbox_extfov_vu,
                predicted_center_vu=self._predicted_center_vu,
                overflow_fraction=self._overflow_fraction,
            ),
            subject_range_km=self._subject_range_km,
            position_cov_px=None,
            intensity_sigma_rel=float(min(0.5, shape.albedo_variation)),
            preferred_filter=NavFilterSpec(kind=NavFilterKind.NONE),
            reliability=_disc_reliability(
                visible_lit_fraction=self._visible_lit_fraction,
                overflow_fraction=self._overflow_fraction,
                diameter_px=self._predicted_diameter_px,
            ),
            reliability_reasons=NavReliabilityBreakdown(
                visible_lit_fraction=self._visible_lit_fraction,
                overflow_fraction=self._overflow_fraction,
            ),
            usable_types=frozenset({NavFeatureType.BODY_DISC}),
            flags=BodyDiscFlags(
                body_name=self._body_name,
                overflow_fov_fraction=self._overflow_fraction,
            ),
            template_img=template_img,
            template_mask=template_mask,
        )

    def _maybe_build_terminator(self, shape: BodyShape) -> NavFeature | None:
        """Build TERMINATOR_ARC when the design's terminator gates pass."""
        sampler = self._terminator_sampler
        if sampler is None or sampler.vertices_vu.shape[0] < TERMINATOR_MIN_VERTICES:
            return None
        if self._phase_angle_factor < TERMINATOR_MIN_PHASE_FACTOR:
            return None
        sigma_normal_per_vertex_px = _sigma_normal_per_vertex(
            sampler=sampler,
            shape=shape,
            psf_sigma_px=psf_sigma_px(self.obs.star_psf()),
            include_albedo=True,
        )
        sigma_tangent_per_vertex_px = np.full_like(sigma_normal_per_vertex_px, 0.5)
        return NavFeature(
            feature_id=f'terminator_arc:{self._body_name}',
            feature_type=NavFeatureType.TERMINATOR_ARC,
            source_model=self.name,
            geometry=TerminatorPolyline(
                vertices_vu=sampler.vertices_vu,
                normals_vu=sampler.normals_vu,
                sigma_normal_per_vertex_px=sigma_normal_per_vertex_px,
                sigma_tangent_per_vertex_px=sigma_tangent_per_vertex_px,
                bbox_extfov_vu=self._bbox_extfov_vu,
            ),
            subject_range_km=self._subject_range_km,
            position_cov_px=None,
            intensity_sigma_rel=0.0,
            preferred_filter=NavFilterSpec(kind=NavFilterKind.NONE),
            reliability=terminator_reliability(
                visible_arc_fraction=_visible_arc_fraction(sampler),
                albedo_variation=shape.albedo_variation,
                phase_factor=self._phase_angle_factor,
            ),
            reliability_reasons=NavReliabilityBreakdown(
                visible_arc_fraction=_visible_arc_fraction(sampler),
                albedo_penalty=min(1.0, shape.albedo_variation),
            ),
            usable_types=frozenset({NavFeatureType.TERMINATOR_ARC}),
            flags=TerminatorArcFlags(
                body_name=self._body_name,
                visible_arc_fraction=_visible_arc_fraction(sampler),
                phase_angle_factor=min(1.0, self._phase_angle_factor),
            ),
        )


def shape_features_suppressed(
    shape: BodyShape, predicted_diameter_px: float, *, config: Config
) -> bool:
    """Whether a body's shape features (limb / terminator / disc) are suppressed.

    Highly-irregular bodies (chaotic rotators, small potato moons) have no
    usable ellipsoid.  Once such a body is resolved beyond a few pixels the
    rendered limb / terminator / disc silhouette does not match the real body,
    so those shape features are suppressed; a point-like BODY_BLOB still
    navigates the centroid.  The 'resolved' threshold reuses the bodies-config
    ``min_bounding_box_area`` -- its square root is the equivalent linear pixel
    extent (default 9 px^2 -> 3 px).  Bodies tagged merely 'irregular' are left
    untouched: the continuous ellipsoid-residual widening in the sigma budget
    already degrades their limb reliability without dropping the feature.

    This is the shared emission policy: the SPICE-backed model applies it to
    all three shape features, and the simulated body model applies it to its
    TERMINATOR_ARC so the two models cannot desync on which bodies terminate.

    Parameters:
        shape: The body's catalog shape profile.
        predicted_diameter_px: Predicted silhouette diameter in pixels.
        config: Configuration supplying ``bodies.min_bounding_box_area``.

    Returns:
        True when the body's shape features must not be emitted.
    """
    resolved_diameter_px = math.sqrt(float(config.bodies.min_bounding_box_area))
    return (
        shape.shape_class_hint == 'highly_irregular'
        and predicted_diameter_px > resolved_diameter_px
    )


def _refine_sampler_to_boundary(
    obs: ObsSnapshot,
    body_name: str,
    sampler: _PolylineSampler,
    *,
    region: Literal['silhouette', 'lit'],
) -> _PolylineSampler:
    """Return ``sampler`` with its vertices probed onto the sub-pixel boundary.

    Thin wrapper over
    :func:`spindoctor.nav_model.silhouette_probe.refine_polyline_vertices`
    that rebuilds the frozen sampler dataclass around the refined vertex
    positions.  The per-vertex incidence and km/px samples keep their
    ridge-pixel values: they vary smoothly at the pixel scale, so re-sampling
    them at the refined position would change nothing measurable.

    Parameters:
        obs: The observation whose FOV defines the probe lines of sight.
        body_name: SPICE body name whose boundary is probed.
        sampler: The discrete-ridge sampler to refine.
        region: ``'silhouette'`` for the limb, ``'lit'`` for the terminator.

    Returns:
        A sampler with refined ``vertices_vu``; the input sampler when it
        holds no vertices.
    """
    if sampler.vertices_vu.shape[0] == 0:
        return sampler
    # Pass this module's Meshgrid / Backplane names so the hermetic render
    # tests' monkeypatched doubles intercept the probe exactly as they
    # intercept the silhouette render.
    refined = refine_polyline_vertices(
        obs,
        body_name,
        sampler.vertices_vu,
        sampler.normals_vu,
        region=region,
        meshgrid_cls=Meshgrid,
        backplane_cls=Backplane,
    )
    return replace(sampler, vertices_vu=refined)


def _build_limb_arc(
    *,
    body_name: str,
    sampler: _PolylineSampler,
    shape: BodyShape,
    bbox: tuple[int, int, int, int],
    subject_range_km: float,
    psf_sigma_px: float,
    source_model: str,
) -> NavFeature:
    """Construct the LIMB_ARC NavFeature for one body."""
    sigma_normal_per_vertex_px = _sigma_normal_per_vertex(
        sampler=sampler, shape=shape, psf_sigma_px=psf_sigma_px, include_albedo=False
    )
    sigma_tangent_per_vertex_px = np.full_like(sigma_normal_per_vertex_px, 0.5)
    visible_arc_fraction = _visible_arc_fraction(sampler)
    return NavFeature(
        feature_id=f'limb_arc:{body_name}',
        feature_type=NavFeatureType.LIMB_ARC,
        source_model=source_model,
        geometry=LimbPolyline(
            vertices_vu=sampler.vertices_vu,
            normals_vu=sampler.normals_vu,
            sigma_normal_per_vertex_px=sigma_normal_per_vertex_px,
            sigma_tangent_per_vertex_px=sigma_tangent_per_vertex_px,
            bbox_extfov_vu=bbox,
        ),
        subject_range_km=subject_range_km,
        position_cov_px=None,
        intensity_sigma_rel=0.0,
        preferred_filter=NavFilterSpec(kind=NavFilterKind.NONE),
        reliability=limb_reliability(
            visible_arc_fraction=visible_arc_fraction,
            visible_arc_px=float(sampler.vertices_vu.shape[0]),
        ),
        reliability_reasons=NavReliabilityBreakdown(
            visible_arc_fraction=visible_arc_fraction,
        ),
        usable_types=frozenset({NavFeatureType.LIMB_ARC}),
        flags=LimbArcFlags(
            body_name=body_name,
            visible_arc_fraction=visible_arc_fraction,
        ),
    )


def _build_polyline_sampler(
    *,
    local_mask: NDArrayBoolType,
    region_mask: NDArrayBoolType,
    incidence_local: NDArrayFloatType,
    km_per_pixel_local: NDArrayFloatType,
    occluder_local: NDArrayBoolType | None = None,
    ext_v0: int,
    ext_u0: int,
) -> _PolylineSampler:
    """Sample a polyline along the True pixels of ``local_mask``.

    The local mask is a 1-pixel-wide ridge along a feature boundary; we
    return a parallel array of vertex coordinates in extfov space, the
    outward-pointing normal at each vertex, the per-vertex incidence
    angle, and the per-vertex km/px scale.

    The outward normal is the discrete gradient of ``region_mask`` (the
    body silhouette for the limb, or the lit mask for the terminator),
    not of the ridge itself: a one-pixel ridge has no consistent
    interior/exterior orientation, so its gradient sign depends on the
    diagonal orientation of the ridge rather than on which side is the
    body interior.  With the body-side True / space-side False
    convention, ``n_v = region[v-1, u] - region[v+1, u]`` and
    ``n_u = region[v, u-1] - region[v, u+1]`` point from inside to
    outside.  Out-of-image neighbours are treated as space (False).

    Parameters:
        local_mask: 1-pixel-wide ridge whose True pixels become vertices.
        region_mask: Body-side / lit-side mask whose discrete gradient
            defines the outward normal.  Same shape as ``local_mask``.
        incidence_local: Per-pixel incidence angle (radians).
        km_per_pixel_local: Per-pixel km/px scale.
        occluder_local: Optional body-body occlusion mask on the same local
            grid; ridge vertices a nearer body hides are dropped from the
            polyline while ``total_vertices`` keeps the full ridge, so the
            visible-arc fraction reports the occlusion loss.
        ext_v0: Extfov v-offset of the local grid origin.
        ext_u0: Extfov u-offset of the local grid origin.
    """
    vs, us = np.where(local_mask)
    total_vertices = int(vs.size)
    if vs.size:
        # Drop zero-resolution ridge vertices (km/px <= 0): the resolution
        # backplane is masked / filled with 0.0 off the resolved body, so
        # such a vertex carries no usable scale.  Keeping it would make the
        # per-vertex sigma NaN, which ``_sigma_normal_per_vertex`` silently
        # floors to a finite fallback -- letting an off-body vertex pollute
        # the LM fit instead of being excluded.  Dropping it here keeps the
        # sampler's parallel arrays consistent and the downstream sigmas
        # finite and positive (which ``lm_subpixel_refine`` requires).
        resolved = km_per_pixel_local[vs, us] > 0.0
        if occluder_local is not None:
            # A ridge vertex a nearer body hides has no counterpart edge in
            # the image; keep it in ``total_vertices`` but drop it from the
            # fitted polyline so the visible-arc fraction reports the loss.
            resolved = resolved & ~occluder_local[vs, us]
        vs, us = vs[resolved], us[resolved]
    if vs.size == 0:
        empty: NDArrayFloatType = np.empty((0, 2), dtype=np.float64)
        return _PolylineSampler(
            vertices_vu=empty,
            normals_vu=empty,
            incidence_rad=np.empty(0, dtype=np.float64),
            km_per_pixel=np.empty(0, dtype=np.float64),
            total_vertices=total_vertices,
        )
    vertices_vu: NDArrayFloatType = np.stack(
        [vs.astype(np.float64) + ext_v0, us.astype(np.float64) + ext_u0], axis=1
    )
    rows, cols = region_mask.shape
    region = region_mask.astype(np.float64)
    normals_vu = np.zeros_like(vertices_vu)
    for i, (v, u) in enumerate(zip(vs, us, strict=True)):
        # Outward normal: discrete gradient of the body-side / lit-side
        # mask, pointing from inside (True) to outside (False).  Out-of-
        # image neighbours count as space (0.0).
        up = region[v - 1, u] if v > 0 else 0.0
        down = region[v + 1, u] if v < rows - 1 else 0.0
        left = region[v, u - 1] if u > 0 else 0.0
        right = region[v, u + 1] if u < cols - 1 else 0.0
        v_dir = up - down
        u_dir = left - right
        norm = math.hypot(v_dir, u_dir) or 1.0
        normals_vu[i, 0] = v_dir / norm
        normals_vu[i, 1] = u_dir / norm
    incidence_rad = incidence_local[vs, us].astype(np.float64)
    km_per_pixel = km_per_pixel_local[vs, us].astype(np.float64)
    return _PolylineSampler(
        vertices_vu=vertices_vu,
        normals_vu=normals_vu,
        incidence_rad=incidence_rad,
        km_per_pixel=km_per_pixel,
        total_vertices=total_vertices,
    )


def _incidence_factor_array(incidence_rad: NDArrayFloatType) -> NDArrayFloatType:
    """Return the design's ``incidence_factor`` array, capped per the constants."""
    deg = np.degrees(incidence_rad)
    deg_clipped = np.clip(deg, 0.0, INCIDENCE_FACTOR_CLIP_DEG)
    factor = 1.0 / np.cos(np.radians(np.minimum(deg_clipped, INCIDENCE_FACTOR_ANGLE_CAP_DEG))) - 1.0
    factor = np.clip(factor, 0.0, MAX_INCIDENCE_FACTOR_CAP)
    out: NDArrayFloatType = factor.astype(np.float64)
    return out


def _sigma_normal_per_vertex(
    *,
    sampler: _PolylineSampler,
    shape: BodyShape,
    psf_sigma_px: float,
    include_albedo: bool,
) -> NDArrayFloatType:
    """Compute the per-vertex normal-sigma per the design.

    Implements the formula from Part 1's "Position covariance per
    feature type" section, including the limb-softness term that uses
    the per-vertex km/px scale and the optional albedo / photometric
    contribution for terminator arcs.
    """
    incidence_factor = _incidence_factor_array(sampler.incidence_rad)
    km_per_pixel = np.where(sampler.km_per_pixel > 0.0, sampler.km_per_pixel, np.nan)
    limb_softness_km = psf_sigma_px * km_per_pixel
    base = (
        shape.ellipsoid_rms_residual_km**2
        + shape.crater_scale_km**2
        + (incidence_factor * limb_softness_km) ** 2
        + shape.spice_orbital_residual_km**2
    )
    if include_albedo:
        albedo_term = (shape.albedo_variation * limb_softness_km) ** 2
        photometric_term = (limb_softness_km * TERMINATOR_PHOTOMETRIC_SOFTNESS_COEFF) ** 2
        base = base + albedo_term + photometric_term
    sigma_km = np.sqrt(np.maximum(base, 0.0))
    sigma_px = sigma_km / km_per_pixel
    return np.nan_to_num(sigma_px, nan=LIMB_ARC_MAX_UNCERTAINTY_PX, posinf=1e3, neginf=1e3)


def _visible_arc_fraction(sampler: _PolylineSampler) -> float:
    """Fraction of the predicted ridge vertices that survived to the fit.

    ``sampler.total_vertices`` is the ridge length found before per-vertex
    drops (zero-resolution / off-body vertices and vertices a nearer sibling
    body occludes), and ``vertices_vu`` holds only the survivors, so the
    visible-arc fraction is ``survivors / total``.  Returns ``0.0`` when no
    ridge vertices were found at all.
    """
    total = sampler.total_vertices
    if total <= 0:
        return 0.0
    return float(sampler.vertices_vu.shape[0]) / float(total)


def limb_reliability(*, visible_arc_fraction: float, visible_arc_px: float) -> float:
    """Sigmoid-of-sum reliability for LIMB_ARC features.

    Shared emission policy: the SPICE-backed model scores its limb sampler
    with this, and the simulated body model feeds it the same quantities
    computed from its own render (arc fraction net of frame clipping and
    body-body occlusion), so the two models cannot desync on how a limb's
    reliability responds to arc visibility and length.

    The score answers a feature-existence question: is this limb arc a
    target a downstream technique should bother running on?  Per-vertex
    geometric softness (high incidence at the terminator-adjacent end
    of the limb) lives in :func:`_sigma_normal_per_vertex`, where the
    LM fit weights individual vertices by their normal sigma; folding
    it into the reliability scalar as well would double-count the same
    physics and, because ``incidence_factor`` saturates near the cap
    on every fully-lit body, would penalize the cleanest possible
    geometries the hardest.
    """
    z = -1.0 + 1.5 * visible_arc_fraction + 1.0 * _sigmoid(visible_arc_px / 50.0)
    return float(_sigmoid(z))


def terminator_reliability(
    *, visible_arc_fraction: float, albedo_variation: float, phase_factor: float
) -> float:
    """Reliability of TERMINATOR_ARC mirroring the design's formula.

    Shared emission policy: the SPICE-backed model scores its terminator
    sampler with this, and the simulated body model feeds it the same
    quantities computed from its own render, so the two models cannot
    desync on how a terminator's reliability responds to arc visibility,
    surface albedo variation, and phase.

    Parameters:
        visible_arc_fraction: Fraction of the predicted terminator ridge
            that is visible / usable for the fit.
        albedo_variation: The body's catalog albedo-variation figure.
        phase_factor: ``sin(phase_angle)``; capped at 1.0.

    Returns:
        The [0, 1] reliability score.
    """
    base = _sigmoid(-1.0 + 1.5 * visible_arc_fraction - 1.5 * albedo_variation)
    return float(base * min(1.0, phase_factor))


def _disc_reliability(
    *, visible_lit_fraction: float, overflow_fraction: float, diameter_px: float
) -> float:
    """Reliability of BODY_DISC per the design (no scoring alpha coefficients yet)."""
    sigmoid_term = _sigmoid(diameter_px / 30.0 - 1.0)
    return float(visible_lit_fraction * (1.0 - overflow_fraction) * sigmoid_term)
