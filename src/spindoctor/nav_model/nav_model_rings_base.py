"""Base class for ring navigation models.

This module provides the annotation creation helper shared by the real ring model
(``NavModelRings``) and the simulated ring model (``NavModelRingsSimulated``).

Anti-aliasing is implemented in
``spindoctor.nav_model.rings.ring_math.compute_antialiasing`` and is invoked from
``RingFeature._render_full_ringlet()``. Annotation helpers live here because they
need observation metadata (image shape, config font settings) that belongs with
``NavModel``, unlike the pure math in ``ring_math``.
"""

import numpy as np
from scipy import ndimage

from spindoctor.annotation import (
    TEXTINFO_BOTTOM_ARROW,
    TEXTINFO_LEFT_ARROW,
    TEXTINFO_RIGHT_ARROW,
    TEXTINFO_TOP_ARROW,
    Annotation,
    Annotations,
    AnnotationTextInfo,
    TextLocInfo,
)
from spindoctor.obs import ObsSnapshotInst
from spindoctor.support.types import NDArrayBoolType

from .nav_model import NavModel


class NavModelRingsBase(NavModel):
    """Base class for ring navigation models.

    Provides the ``_create_edge_annotations`` helper for creating annotations
    consistent between the real and simulated ring model implementations.
    Anti-aliasing math is in ``spindoctor.nav_model.rings.ring_math``.
    """

    _abstract = True

    def _create_edge_annotations(
        self,
        obs: ObsSnapshotInst,
        edge_info_list: list[tuple[NDArrayBoolType, str, str]],
        model_mask: NDArrayBoolType,
    ) -> Annotations:
        """Create annotation objects for ring edges.

        Parameters:
            obs: The observation object.
            edge_info_list: List of (edge_mask, label_text, edge_type) tuples where
                edge_mask is a boolean array indicating edge pixels in extended FOV.
            model_mask: Model mask array.

        Returns:
            Annotations object containing all ring edge annotations.
        """

        annotations = Annotations()
        rings_config = self._config.rings

        # Get annotation configuration (use body defaults if not specified)
        label_font = rings_config.label_font
        label_font_size = rings_config.label_font_size
        label_font_color = rings_config.label_font_color
        label_limb_color = rings_config.label_limb_color
        label_horiz_gap = rings_config.label_horiz_gap
        label_vert_gap = rings_config.label_vert_gap
        label_mask_enlarge = rings_config.label_mask_enlarge

        if not edge_info_list:
            return annotations

        # Create annotations for each edge
        for edge_mask, label_text, _edge_type in edge_info_list:
            if not np.any(edge_mask):
                continue

            # Find candidate text locations along edge
            edge_v, edge_u = np.where(edge_mask)
            if len(edge_v) == 0:
                continue

            # Create text location candidates with offsets from edge
            text_loc: list[TextLocInfo] = []

            # Sample more edge points (up to 50, spaced evenly)
            num_samples = min(50, len(edge_v))
            step = max(1, len(edge_v) // num_samples)
            sampled_indices = range(0, len(edge_v), step)

            # Get image center for determining label side
            u_center = model_mask.shape[1] // 2
            v_center = model_mask.shape[0] // 2

            for idx in sampled_indices:
                v = edge_v[idx]
                u = edge_u[idx]

                # Calculate tangent direction by looking at neighboring edge points
                # Find nearby edge points within a small radius
                search_radius = 5.0  # pixels
                distances = np.sqrt((edge_v - v) ** 2 + (edge_u - u) ** 2)
                nearby_mask = (distances > 0.5) & (distances <= search_radius)
                nearby_indices = np.where(nearby_mask)[0]

                if len(nearby_indices) < 2:
                    # Not enough nearby points, skip this candidate
                    continue

                # Get nearby points relative to current point
                nearby_v_rel = edge_v[nearby_indices] - v
                nearby_u_rel = edge_u[nearby_indices] - u

                # Compute direction by looking at which component has more variation
                # This avoids cancellation when points are on both sides
                # Weight by inverse distance (closer points matter more)
                weights = 1.0 / (distances[nearby_indices] + 0.1)

                # Compute weighted standard deviation (variation) in each direction
                # This tells us which direction the edge runs
                du_var = np.average(nearby_u_rel**2, weights=weights)
                dv_var = np.average(nearby_v_rel**2, weights=weights)

                # Use the direction with more variation as the tangent direction
                if dv_var > du_var:
                    # Edge runs more vertically (v varies more)
                    du_norm = 0.0
                    dv_norm = 1.0 if np.average(nearby_v_rel, weights=weights) >= 0 else -1.0
                else:
                    # Edge runs more horizontally (u varies more)
                    du_norm = 1.0 if np.average(nearby_u_rel, weights=weights) >= 0 else -1.0
                    dv_norm = 0.0

                # Check if tangent is more horizontal or vertical
                # du_norm and dv_norm are the normalized direction components
                # If |du| > |dv|, tangent is more horizontal (edge runs left-right) -> label up/down
                # If |dv| > |du|, tangent is more vertical (edge runs up-down) -> label left/right
                abs_du = abs(du_norm)
                abs_dv = abs(dv_norm)

                if abs_dv > abs_du:
                    # Tangent is more vertical (edge runs vertically, up-down)
                    # Place label perpendicular to edge: left or right
                    if u < u_center:
                        # Left side of image, place label to the left
                        u_offset = max(0, u - label_horiz_gap)
                        text_loc.append(TextLocInfo(TEXTINFO_LEFT_ARROW, v, u_offset))
                    else:
                        # Right side of image, place label to the right
                        u_offset = min(model_mask.shape[1] - 1, u + label_horiz_gap)
                        text_loc.append(TextLocInfo(TEXTINFO_RIGHT_ARROW, v, u_offset))
                else:
                    # Tangent is more horizontal (edge runs horizontally, left-right)
                    # Place label perpendicular to edge: up or down
                    if v < v_center:
                        # Top side of image, place label above
                        v_offset = max(0, v - label_vert_gap)
                        text_loc.append(TextLocInfo(TEXTINFO_TOP_ARROW, v_offset, u))
                    else:
                        # Bottom side of image, place label below
                        v_offset = min(model_mask.shape[0] - 1, v + label_vert_gap)
                        text_loc.append(TextLocInfo(TEXTINFO_BOTTOM_ARROW, v_offset, u))

            if not text_loc:
                continue

            # Create text info
            text_info = AnnotationTextInfo(
                label_text,
                text_loc=text_loc,
                ref_vu=None,
                font=label_font,
                font_size=label_font_size,
                color=label_font_color,
            )

            # Create avoid mask from model_mask to avoid placing text on rings
            text_avoid_mask = ndimage.maximum_filter(
                model_mask.astype(np.float32), int(label_mask_enlarge)
            ).astype(bool)

            # Create annotation
            annotation = Annotation(
                obs,
                edge_mask,
                label_limb_color,
                thicken_overlay=0,
                avoid_mask=text_avoid_mask,
                text_info=text_info,
                config=self._config,
            )

            annotations.add_annotations(annotation)

        return annotations
