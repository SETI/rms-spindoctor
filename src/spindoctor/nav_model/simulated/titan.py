"""Titan NavModel rendered from operator-supplied simulation parameters.

The catalog-driven :class:`~spindoctor.nav_model.nav_model_titan.NavModelTitan`
is already almost entirely a pure function of one frozen dataclass --
:class:`~spindoctor.nav_model.titan_geometry.TitanGeometryInputs` -- with
``oops`` confined to the one entry point that builds it.  This model
therefore builds the SAME dataclass from a simulated scene's idealized body
parameters and inherits everything downstream: the reliability formula, the
hard-zero conditions, the emitted ``TITAN_LIMB`` payload, and the overlay.  A
simulated haze frame consequently exercises the shipped emission rules rather
than a parallel implementation of them, which is the whole point of grading a
navigator against the simulator.

Information boundary.  Every quantity below comes from the navigator view
(``nav_params``): the body's center, its axes, its pixel scale, its phase,
and its illumination direction -- catalog geometry a real pipeline reads from
SPICE.  The ``atmosphere`` block that gives the rendered body its soft haze
limb is truth and is never read here; the envelope radius comes from the same
configured atmosphere height the real model uses, so the predicted envelope is
a deliberate approximation of the rendered haze exactly as it is on a real
frame.  Sim inventory contract: this model reads operator parameters
directly, so the simulated inventory needs no ``center_uv`` key.

Unconfigured scenes.  A body named TITAN that does not carry the parameters
this model needs (its center, its axes, and the pixel scale that turns the
configured atmosphere height into an envelope radius) yields NO model at all,
and, because the simulated body model excludes TITAN unconditionally, no body
model either.  The frame then resolves through the standard generic status
reasons for a scene with nothing to navigate rather than through a crash or a
silently degenerate feature.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from spindoctor.config import DEFAULT_CONFIG, IMAGE_LOGGER, Config
from spindoctor.nav_model.nav_model import NavModel
from spindoctor.nav_model.nav_model_body import TITAN_BODY_NAME
from spindoctor.nav_model.nav_model_titan import NavModelTitan
from spindoctor.nav_model.simulated.predicted_body import (
    predicted_silhouette,
    predicted_silhouette_bbox,
)
from spindoctor.nav_model.titan_geometry import (
    TitanGeometryInputs,
    occluded_disc_fraction,
    paint_disc,
)
from spindoctor.obs import ObsSnapshotInst
from spindoctor.support.constants import PIXEL_CENTER_TO_CORNER_PX
from spindoctor.support.types import NDArrayBoolType

__all__ = ['REQUIRED_SIM_PARAMS', 'NavModelTitanSimulated']


REQUIRED_SIM_PARAMS: tuple[str, ...] = (
    'center_v',
    'center_u',
    'semi_a_px',
    'semi_b_px',
    'km_per_pixel',
)
"""Body parameters a simulated Titan scene must supply for a model to build.

The center and the two image-plane semi-axes give the predicted disc; the pixel
scale is what turns the configured atmosphere height (kilometers) into an
envelope radius (pixels), and without it the envelope -- the outer bound of
everything the fit samples -- would have to be invented.  A scene missing any
of them gets no haze model, which is a legible absence rather than a
degenerate feature.
"""


def _mean_semi_axis_px(body_params: dict[str, Any]) -> float:
    """Mean image-plane semi-axis of a simulated body, in pixels.

    The haze fit works from one circular radius, so the two image-plane
    semi-axes are averaged the way the real model averages the two per-axis
    center resolutions.

    Parameters:
        body_params: One idealized body entry.

    Returns:
        The mean semi-axis in pixels.
    """
    return 0.5 * (float(body_params['semi_a_px']) + float(body_params['semi_b_px']))


class NavModelTitanSimulated(NavModelTitan):
    """Haze-envelope NavModel for a simulated Titan.

    Subclasses the catalog-driven model and replaces exactly one thing: how
    the geometry dataclass is obtained.  Feature emission, reliability, the
    hard-zero conditions, and the overlay are inherited unchanged, so a
    simulated frame and a real frame cannot diverge in what a haze feature
    means.

    Parameters:
        name: Model instance name (``'titan_sim:TITAN'``).
        obs: Simulated observation snapshot.
        body_params: The idealized parameter dict of the simulated Titan,
            from the filtered ``nav_params['bodies']`` list.
        sibling_bodies: Idealized parameter dicts of the OTHER bodies in the
            same scene.  A sibling with an explicitly nearer ``range_km``
            occludes the envelope and counts toward the occluded fraction;
            every sibling, near or far, contributes its bounding box to the
            contaminant mask, because a moon beside the limb sits in the
            symmetry annulus whether it hides anything or not.
        star_records: Idealized star entries of the same scene; those
            brighter than the configured mask limit contribute a masked
            disc, mirroring the real model's catalog queries.
        config: Optional ``Config`` override.
    """

    inputs = 'scene_view'

    def __init__(
        self,
        name: str,
        obs: ObsSnapshotInst,
        body_params: dict[str, Any],
        *,
        sibling_bodies: list[dict[str, Any]] | None = None,
        star_records: list[dict[str, Any]] | None = None,
        config: Config | None = None,
    ) -> None:
        super().__init__(name, obs, config=config)
        self._body_params: dict[str, Any] = dict(body_params)
        self._sibling_bodies: list[dict[str, Any]] = [dict(s) for s in sibling_bodies or []]
        self._star_records: list[dict[str, Any]] = [dict(s) for s in star_records or []]

    @classmethod
    def instances_for_obs(
        cls, obs: ObsSnapshotInst, *, config: Config | None = None
    ) -> list[NavModel]:
        """Return one instance per configured simulated Titan, else none.

        Reads the navigator view (``obs.nav_params``) and never the
        full scene, whose truth keys -- the haze among them -- stay behind
        the information boundary.

        Parameters:
            obs: Observation snapshot.
            config: Configuration passed to the constructed instances.
                None uses ``DEFAULT_CONFIG``.

        Returns:
            One ``NavModelTitanSimulated`` per adequately configured
            simulated Titan; an empty list otherwise.
        """
        nav_params = getattr(obs, 'nav_params', None)
        if not isinstance(nav_params, dict):
            return []
        bodies = [bp for bp in nav_params.get('bodies', []) or [] if isinstance(bp, dict)]
        stars = [sp for sp in nav_params.get('stars', []) or [] if isinstance(sp, dict)]
        out: list[NavModel] = []
        for index, body_params in enumerate(bodies):
            if str(body_params.get('name', '')).upper() != TITAN_BODY_NAME:
                continue
            missing = [key for key in REQUIRED_SIM_PARAMS if body_params.get(key) is None]
            if missing:
                IMAGE_LOGGER.warning(
                    'Simulated TITAN scene omits %s; building no haze model for it',
                    ', '.join(missing),
                )
                continue
            siblings = [dict(bp) for j, bp in enumerate(bodies) if j != index]
            out.append(
                cls(
                    'titan_sim:TITAN',
                    obs,
                    dict(body_params),
                    sibling_bodies=siblings,
                    star_records=stars,
                    config=config or DEFAULT_CONFIG,
                )
            )
        return out

    @property
    def geometry_inputs(self) -> TitanGeometryInputs:
        """The haze geometry, built from operator parameters on first access."""
        if self._geometry is None:
            self._geometry = self._geometry_from_params()
        return self._geometry

    def _geometry_from_params(self) -> TitanGeometryInputs:
        """Build the geometry dataclass from this scene's idealized parameters.

        The mapping is deliberately the sim analog of each real-frame
        quantity: the disc center is the operator's center shifted into
        extfov coordinates (the pipeline-wide convention for a predicted
        position), the solid radius is the mean image-plane semi-axis, the
        envelope adds the configured atmosphere height through the scene's
        own pixel scale, and the symmetry axis is the illumination direction
        expressed in the fitting library's ``theta`` convention.

        Returns:
            The fully-populated :class:`TitanGeometryInputs`.
        """
        params = self._body_params
        nav_config = self._config.titan['navigation']
        margin_v = int(self.obs.extfov_margin_v)
        margin_u = int(self.obs.extfov_margin_u)
        extfov_shape_vu = (int(self.obs.extdata_shape_vu[0]), int(self.obs.extdata_shape_vu[1]))
        window_px = float(max(margin_v, margin_u))
        center_vu = (
            float(params['center_v']) - PIXEL_CENTER_TO_CORNER_PX + margin_v,
            float(params['center_u']) - PIXEL_CENTER_TO_CORNER_PX + margin_u,
        )
        km_per_px = float(params['km_per_pixel'])
        r_solid_px = _mean_semi_axis_px(params)
        r_env_px = r_solid_px + float(self._config.titan['atmosphere_height']) / km_per_px
        phase_deg = float(params.get('phase_angle', 0.0))
        theta_rad, axis_degenerate = self._symmetry_axis(
            params,
            r_solid_px=r_solid_px,
            phase_deg=phase_deg,
            axis_min_offset_px=float(nav_config['axis_min_offset_px']),
        )
        contaminant, occluded_fraction = self._contaminant_mask(
            extfov_shape_vu=extfov_shape_vu,
            margin_vu=(margin_v, margin_u),
            center_vu=center_vu,
            r_env_px=r_env_px,
        )
        reach = math.ceil(r_env_px)
        return TitanGeometryInputs(
            predicted_center_vu=center_vu,
            r_solid_px=r_solid_px,
            r_env_px=r_env_px,
            km_per_px=km_per_px,
            phase_deg=phase_deg,
            theta_rad=theta_rad,
            axis_degenerate=axis_degenerate,
            occluded_fraction=occluded_fraction,
            contaminant_mask=contaminant,
            extfov_shape_vu=extfov_shape_vu,
            window_px=window_px,
            extfov_margin_vu=(float(margin_v), float(margin_u)),
            bbox_extfov_vu=(
                math.floor(center_vu[0]) - reach,
                math.floor(center_vu[1]) - reach,
                math.ceil(center_vu[0]) + reach + 1,
                math.ceil(center_vu[1]) + reach + 1,
            ),
            subject_range_km=float(params.get('range_km', float('inf'))),
            filters=(),
        )

    @staticmethod
    def _symmetry_axis(
        params: dict[str, Any],
        *,
        r_solid_px: float,
        phase_deg: float,
        axis_min_offset_px: float,
    ) -> tuple[float, bool]:
        """Return ``(theta_rad, axis_degenerate)`` for a simulated body.

        The renderer states the in-plane light direction directly, so the
        sub-solar image direction needs no backplane: the scene's
        ``illumination_angle`` runs from the top of the image clockwise
        toward the right, while the fitting library's ``theta`` has
        ``a_hat = (sin theta, cos theta)`` pointing sunward, which is the
        same direction a quarter turn earlier.

        Degeneracy mirrors the real model's test rather than restating it:
        the sub-solar point of a sphere at phase ``p`` projects
        ``R sin(p)`` from the disc center, so a phase near zero puts it
        inside the same ``axis_min_offset_px`` the backplane search uses,
        on a disc that is rotationally symmetric anyway.

        Parameters:
            params: The body's idealized parameters.
            r_solid_px: The solid-body radius in pixels.
            phase_deg: Phase angle in degrees.
            axis_min_offset_px: Offset below which the axis is degenerate.

        Returns:
            ``(theta_rad, axis_degenerate)``.
        """
        offset_px = r_solid_px * abs(math.sin(math.radians(phase_deg)))
        if offset_px < axis_min_offset_px:
            return 0.0, True
        illumination_rad = math.radians(float(params.get('illumination_angle', 0.0)))
        return illumination_rad - 0.5 * math.pi, False

    def _contaminant_mask(
        self,
        *,
        extfov_shape_vu: tuple[int, int],
        margin_vu: tuple[int, int],
        center_vu: tuple[float, float],
        r_env_px: float,
    ) -> tuple[NDArrayBoolType | None, float]:
        """Build the contaminant mask and the occluded fraction for a sim scene.

        Three of the real model's four components have sim analogs and are
        built here: nearer-sibling occlusion (the same explicit-range rule
        the simulated body model applies), the bounding boxes of every other
        body in the scene, and discs over stars
        brighter than the configured mask limit.  Ring occlusion has no
        analog -- the simulated ring system is a separate scene element with
        no depth relation to a body -- so it contributes nothing, and a
        simulated frame simply cannot exercise that path.  Each sibling's
        silhouette and box come from the predicted silhouette the simulated
        body model draws for the same body, in
        :mod:`~spindoctor.nav_model.simulated.predicted_body`.

        Parameters:
            extfov_shape_vu: ``(rows, columns)`` of the extended frame.
            margin_vu: ``(margin_v, margin_u)`` extfov margins.
            center_vu: Envelope center in extfov coordinates.
            r_env_px: Envelope radius in pixels.

        Returns:
            ``(mask, occluded_fraction)``; the mask is None when nothing is
            masked.
        """
        nav_config = self._config.titan['navigation']
        data_shape = (int(self.obs.data_shape_v), int(self.obs.data_shape_u))
        occluder_ext: NDArrayBoolType = np.zeros(extfov_shape_vu, dtype=bool)
        contaminant_ext: NDArrayBoolType = np.zeros(extfov_shape_vu, dtype=bool)
        own_range = self._body_params.get('range_km')
        for sibling in self._sibling_bodies:
            sibling_range = sibling.get('range_km')
            nearer = (
                own_range is not None
                and sibling_range is not None
                and float(sibling_range) < float(own_range)
            )
            if nearer:
                self._embed_data_mask(
                    occluder_ext, predicted_silhouette(sibling, data_shape), margin_vu
                )
            self._paint_sibling_box(
                contaminant_ext,
                predicted_silhouette_bbox(sibling, data_shape, margin_vu=margin_vu),
                margin_vu,
            )
        contaminant_ext |= occluder_ext
        vmag_limit = float(nav_config['star_mask_vmag_limit'])
        radius_px = float(nav_config['star_mask_radius_px'])
        for star in self._star_records:
            if float(star.get('vmag', 99.0)) > vmag_limit:
                continue
            # A scene states a star's position as a pixel corner and the disc is
            # painted against pixel centric grids, so the half pixel comes off
            # before the extfov margins go on -- the same conversion the
            # catalog-driven model makes for its own star discs.
            paint_disc(
                contaminant_ext,
                (
                    float(star['v']) - PIXEL_CENTER_TO_CORNER_PX + margin_vu[0],
                    float(star['u']) - PIXEL_CENTER_TO_CORNER_PX + margin_vu[1],
                ),
                radius_px,
            )
        fraction = occluded_disc_fraction(occluder_ext, center_vu, r_env_px)
        return (contaminant_ext if contaminant_ext.any() else None), fraction

    @staticmethod
    def _embed_data_mask(
        target: NDArrayBoolType, local: NDArrayBoolType, margin_vu: tuple[int, int]
    ) -> None:
        """OR a data-coordinate mask into an extfov-shaped array at the margin."""
        rows, cols = target.shape
        v_hi = min(rows, margin_vu[0] + local.shape[0])
        u_hi = min(cols, margin_vu[1] + local.shape[1])
        target[margin_vu[0] : v_hi, margin_vu[1] : u_hi] |= local[
            : v_hi - margin_vu[0], : u_hi - margin_vu[1]
        ]

    @staticmethod
    def _paint_sibling_box(
        mask: NDArrayBoolType,
        bbox_vu: tuple[int, int, int, int] | None,
        margin_vu: tuple[int, int],
    ) -> None:
        """Paint one sibling body's bounding box into the contaminant mask.

        Range order is deliberately ignored, exactly as on a real frame: a
        moon behind Titan occludes nothing, but its visible sliver beside
        the limb sits squarely in the symmetry annulus and in the arc rays.
        The box is the bounding box of the part of the sibling's predicted
        silhouette that falls on the extended frame, so it holds the body at
        whatever center, shape, and rotation that prediction gives it, and
        reaches into the extfov margin where the body runs past the frame
        edge.

        Parameters:
            mask: Extfov-shaped contaminant mask, modified in place.
            bbox_vu: ``(v_min, v_max, u_min, u_max)``, the inclusive
                frame-pixel bounds of the sibling's predicted silhouette on
                the extended frame, or None when it covers no pixel there.
            margin_vu: ``(margin_v, margin_u)`` extfov margins.
        """
        if bbox_vu is None:
            return
        v_min, v_max, u_min, u_max = bbox_vu
        mask[
            v_min + margin_vu[0] : v_max + margin_vu[0] + 1,
            u_min + margin_vu[1] : u_max + margin_vu[1] + 1,
        ] = True
