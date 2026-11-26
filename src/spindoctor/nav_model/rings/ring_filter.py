"""Ring feature filter pipeline.

This module implements a four-pass filter that decides which ring features (and which
edges within each feature) are included in the final render. It is deliberately
separated from the rendering logic in ``ring_feature.py`` for two reasons:

1. **Single responsibility**: A feature object knows how to render itself; the filter
   knows which features are *worth* rendering for a given observation. These are
   distinct concerns -- a feature's physics do not change based on whether it is
   currently visible.

2. **Testability**: The filter is a pure function of its inputs (features + observation
   parameters). It can be tested independently of backplane computation. Rendering
   involves complex oops backplane calls that require mocking; the filter only needs
   simple numeric comparisons.

Pass 4 (fade conflict) checks individual edges, not whole features, because:

- A GAP feature has two fade edges that shade independently in opposite directions.
  The outer edge shading outward may be clear of conflicts while the inner edge
  shading inward is blocked -- the outer edge is still useful for navigation.
- Excluding the whole feature because one edge is blocked would silently remove
  valid navigation signals.

**Pipeline order rationale**:

1. Date (cheapest: pure arithmetic) eliminates features not valid for this image.
2. Radius eliminates features outside the current field of view.
3. Resolvability eliminates two-edge features too narrow to see as a width.
4. Fade conflict (most expensive: needs all surviving edge radii) eliminates or trims
   individual fade-using edges that are squeezed by a neighbor.

Processing in this order means expensive operations run only on the smaller set of
features that survive the cheaper passes.
"""

from collections.abc import Callable, Sequence
from typing import Any, Literal

from .ring_feature import RingFeature
from .ring_types import RingFeatureType


class RingFeatureFilter:
    """Four-pass filter deciding which ring features and edges to include in a render.

    Instantiate with observation-specific parameters, then call ``filter()`` with the
    full feature list. The filter is stateless between calls; the same instance can
    be reused for the same observation parameters.

    Pipeline passes:
        1. Date: exclude features not valid at ``obs_time_et``.
        2. Radius: exclude features with no edge inside ``[min_radius, max_radius]``.
        3. Resolvability: exclude two-edge features narrower than
           ``min_feature_pixels * min_res`` km.
        4. Fade conflict: exclude or trim individual fade-using edges whose
           conflict-adjusted fade width falls below ``min_allowed_fade_width_pix * min_res``.
    """

    def __init__(
        self,
        *,
        obs_time_et: float,
        min_radius: float,
        max_radius: float,
        min_res_at_radius: Callable[[float], float | None],
        fade_width_pix: float,
        min_allowed_fade_width_pix: float,
        min_feature_pixels: float,
        logger: Any,
    ) -> None:
        """Initialize the filter with observation-specific parameters.

        Parameters:
            obs_time_et: Observation time in TDB seconds (from ``utc_to_et``).
            min_radius: Minimum visible ring radius in km.
            max_radius: Maximum visible ring radius in km.
            min_res_at_radius: Callable mapping radius (km) to minimum radial
                resolution (km/pixel) at that radius, or None if not resolvable.
                Typically derived from ``obs.ext_bp.ring_radial_resolution``.
            fade_width_pix: Desired fade extent in pixels, from the planet's ring
                catalog, which the configuration checks when it loads.
            min_allowed_fade_width_pix: Minimum allowed fade width in pixels after
                conflict reduction, no more than ``fade_width_pix``. Edges adjusted
                below this threshold are excluded.
            min_feature_pixels: Minimum resolvable feature width in pixels. Two-edge
                features narrower than ``min_feature_pixels * min_res`` km are excluded.
            logger: ``PdsLogger`` from the ring ``NavModel`` (same as
                ``NavModelRings._logger``) for filter pass debug output.

        Raises:
            TypeError: If an observation parameter has an invalid type.
            ValueError: If ``logger`` is None or ``min_radius`` exceeds ``max_radius``.
        """
        if isinstance(obs_time_et, bool) or not isinstance(obs_time_et, (int, float)):
            raise TypeError(f'obs_time_et must be int or float, not {type(obs_time_et).__name__}')
        if isinstance(min_radius, bool) or not isinstance(min_radius, (int, float)):
            raise TypeError(f'min_radius must be int or float, not {type(min_radius).__name__}')
        if isinstance(max_radius, bool) or not isinstance(max_radius, (int, float)):
            raise TypeError(f'max_radius must be int or float, not {type(max_radius).__name__}')
        if min_radius > max_radius:
            raise ValueError(
                f'min_radius must be <= max_radius, got min_radius={min_radius}, '
                f'max_radius={max_radius}'
            )
        if not callable(min_res_at_radius):
            raise TypeError('min_res_at_radius must be callable')
        if logger is None:
            raise ValueError('logger must not be None')

        self._obs_time_et = obs_time_et
        self._min_radius = min_radius
        self._max_radius = max_radius
        self._min_res_at_radius = min_res_at_radius
        self._fade_width_pix = fade_width_pix
        self._min_allowed_fade_width_pix = min_allowed_fade_width_pix
        self._min_feature_pixels = min_feature_pixels
        self._logger = logger

    def filter(self, features: Sequence[RingFeature]) -> list[RingFeature]:
        """Run the four-pass filter and return surviving features.

        Features that fail a pass are excluded entirely. Features that partially fail
        pass 4 (one edge of a GAP excluded) are returned with the failing edge set to
        None.

        Parameters:
            features: Ring features retrieved from configuration for the planet.

        Returns:
            Filtered list of features, possibly with some edges trimmed to None.
        """
        feature_seq = list(features)
        self._logger.debug('RingFeatureFilter: starting with %d feature(s)', len(feature_seq))

        # Pass 1: date
        after_date = [f for f in feature_seq if self._passes_date(f)]
        self._logger.debug(
            'RingFeatureFilter: after date pass, %d / %d feature(s)',
            len(after_date),
            len(feature_seq),
        )

        # Pass 2: radius
        after_radius = [f for f in after_date if self._passes_radius(f)]
        self._logger.debug(
            'RingFeatureFilter: after radius pass, %d / %d feature(s)',
            len(after_radius),
            len(after_date),
        )

        # Pass 3: resolvability (two-edge features only)
        after_res = [f for f in after_radius if self._passes_resolvability(f)]
        self._logger.debug(
            'RingFeatureFilter: after resolvability pass, %d / %d feature(s)',
            len(after_res),
            len(after_radius),
        )

        # Build all_edge_radii from pass-2 survivors for conflict detection in pass 4.
        # Using pass-2 survivors (not pass-3) so that a narrowly excluded two-edge
        # feature still contributes its edge radii for conflict detection: if a ringlet
        # is too narrow to render but physically present, its edges can still conflict
        # with neighboring fades.
        all_edge_radii: list[tuple[float, str]] = []
        for feat in after_radius:
            all_edge_radii.extend(feat.all_base_radii())
        all_edge_radii.sort(key=lambda x: x[0])

        # Pass 4: fade conflict (per-edge check; may trim features)
        result: list[RingFeature] = []
        for feat in after_res:
            trimmed = self._apply_fade_filter(feat, all_edge_radii)
            if trimmed is not None:
                result.append(trimmed)

        # Pass 4 outermost-preservation: at very low resolution every
        # neighboring fade overlaps every other and the per-edge check
        # can drop *every* outer-side edge, leaving only an
        # innermost-region feature (e.g. a C-ring gap) that is far less
        # useful for navigation than the outer A-ring edge would have
        # been.  If the outermost in-range edge that survived pass 3
        # was excluded by pass 4, restore the feature carrying it: the
        # outer edge of the ring system is the most useful single
        # navigation reference, and even a narrow-fade rendering of
        # it is better than nothing.  Out-of-range edges are excluded
        # from the comparison so the partial-visibility trim's
        # trimmed-but-still-valid feature is not falsely restored.
        if after_res:
            after_res_in_range = [
                (r, label, feat)
                for feat in after_res
                for r, label in feat.all_base_radii()
                if self._min_radius <= r <= self._max_radius
            ]
            if after_res_in_range:
                outermost_radius, outermost_label, outermost_feature = max(
                    after_res_in_range, key=lambda triple: triple[0]
                )
                survives_outermost = any(
                    feat.key == outermost_feature.key
                    and (outermost_radius, outermost_label) in feat.all_base_radii()
                    for feat in result
                )
                if not survives_outermost:
                    self._logger.debug(
                        'RingFeatureFilter: pass 4 dropped the outermost feature %r '
                        '(largest in-range edge radius %.1f km); restoring it for navigation',
                        outermost_feature.key,
                        outermost_radius,
                    )
                    # If the trimmed-but-incomplete version of this feature
                    # is already in ``result`` (same key, but lacking the
                    # outermost edge), replace it in place rather than
                    # appending a second entry with the same key.  This can
                    # arise when ``_apply_fade_filter`` returns a trimmed
                    # version that drops the outer edge — preserving the
                    # original (untrimmed) feature is what restores the
                    # navigation-useful outer reference.
                    existing_index = next(
                        (
                            idx
                            for idx, feat in enumerate(result)
                            if feat.key == outermost_feature.key
                        ),
                        None,
                    )
                    if existing_index is None:
                        result.append(outermost_feature)
                    else:
                        result[existing_index] = outermost_feature

        self._logger.debug(
            'RingFeatureFilter: after fade pass, %d / %d feature(s)',
            len(result),
            len(after_res),
        )
        return result

    # ------------------------------------------------------------------
    # Pass implementations
    # ------------------------------------------------------------------

    def _passes_date(self, feature: RingFeature) -> bool:
        """Return True if the feature is valid at the observation time."""
        if not feature.is_visible_at(self._obs_time_et):
            self._logger.debug(
                'Pass 1 (date): excluding %r -- not active at observation time',
                feature.key,
            )
            return False
        return True

    def _passes_radius(self, feature: RingFeature) -> bool:
        """Return True if at least one feature edge is within the visible radius range."""
        if not feature.is_in_radius_range(self._min_radius, self._max_radius):
            self._logger.debug(
                'Pass 2 (radius): excluding %r -- no edges in range [%.1f, %.1f] km',
                feature.key,
                self._min_radius,
                self._max_radius,
            )
            return False
        return True

    def _passes_resolvability(self, feature: RingFeature) -> bool:
        """Return True if a two-edge feature is wide enough to resolve.

        Single-edge features and partially visible features (one edge outside the
        visible radius range) always pass: there is no width to check.

        For fully visible two-edge features, the width is
        ``outer_a - inner_a``. The threshold is
        ``min_feature_pixels * min(min_res_at_inner, min_res_at_outer)``. Using the
        minimum resolution along the feature (i.e., the finest resolution in the field
        of view at that feature location) is optimistic: if the feature is resolvable
        anywhere in the image, keep it.
        """
        inner = feature.inner_edge
        outer = feature.outer_edge
        if inner is None or outer is None:
            return True

        # Check whether both edges are within the visible radius range
        inner_in_range = self._min_radius <= inner.base_radius <= self._max_radius
        outer_in_range = self._min_radius <= outer.base_radius <= self._max_radius
        if not (inner_in_range and outer_in_range):
            # Partially visible: skip resolvability check
            return True

        width_km = outer.base_radius - inner.base_radius

        inner_res = self._min_res_at_radius(inner.base_radius)
        outer_res = self._min_res_at_radius(outer.base_radius)

        if inner_res is None or inner_res == 0.0:
            inner_res = outer_res
        if outer_res is None or outer_res == 0.0:
            outer_res = inner_res

        # Fallback above copies a valid resolution when one side is None or 0.0, so
        # both sides stay aligned. The only ambiguous case left is still-missing
        # inner_res, or inner_res == 0.0 (outer_res cannot be 0.0 unless inner_res is
        # too); exclude and log like other pass-3 skips.
        if inner_res is None or inner_res == 0.0 or outer_res is None:
            self._logger.debug(
                'Pass 3 (resolvability): excluding %r -- could not determine resolution',
                feature.key,
            )
            return False

        min_res: float = min(inner_res, outer_res)
        threshold_km = self._min_feature_pixels * min_res

        if width_km < threshold_km:
            self._logger.debug(
                'Pass 3 (resolvability): excluding %r -- width %.1f km < '
                '%.1f px * %.2f km/px = %.1f km threshold',
                feature.key,
                width_km,
                self._min_feature_pixels,
                min_res,
                threshold_km,
            )
            return False

        return True

    def _apply_fade_filter(
        self,
        feature: RingFeature,
        all_edge_radii: list[tuple[float, str]],
    ) -> RingFeature | None:
        """Apply pass 4 fade conflict check to a feature.

        Before running the fade conflict check, trim any out-of-range edge from a
        RINGLET that has both edges structurally present. This implements the
        partial-visibility handling promised by ``RingFeature.uses_fade_for_edge``'s
        docstring: the in-range edge is then treated as a single-edge (fade-using)
        feature and receives correct conflict detection, while ``render()`` naturally
        takes the single-edge fade path instead of producing a solid band that
        extends to an off-screen edge.

        For each edge that uses fade rendering (after any partial-visibility trim),
        compute the adjusted fade width after conflict reduction and compare against
        ``min_allowed_fade_width_pix``. Excluded edges are set to None. If all edges
        are excluded, return None.

        Parameters:
            feature: Feature to check.
            all_edge_radii: Sorted (radius, label) pairs from all pass-2 surviving
                features. Used for conflict detection.

        Returns:
            The (possibly trimmed) feature, or None if all edges were excluded.
        """
        # Partial-visibility trim: for a RINGLET with both edges, remove any edge
        # whose base radius is outside the visible range. Pass 2 guarantees at least
        # one edge is in range, so the trimmed feature always has one valid edge.
        if (
            feature.feature_type is RingFeatureType.RINGLET
            and feature.inner_edge is not None
            and feature.outer_edge is not None
        ):
            inner_in_range = self._min_radius <= feature.inner_edge.base_radius <= self._max_radius
            outer_in_range = self._min_radius <= feature.outer_edge.base_radius <= self._max_radius
            if not (inner_in_range and outer_in_range):
                out_of_range_side = 'inner' if not inner_in_range else 'outer'
                self._logger.debug(
                    'Pass 4 (partial visibility): trimming %r -- %s edge (%.1f km) '
                    'outside visible range [%.1f, %.1f] km',
                    feature.key,
                    out_of_range_side,
                    (
                        feature.inner_edge.base_radius
                        if not inner_in_range
                        else feature.outer_edge.base_radius
                    ),
                    self._min_radius,
                    self._max_radius,
                )
                feature = RingFeature(
                    key=feature.key,
                    name=feature.name,
                    feature_type=feature.feature_type,
                    inner_edge=feature.inner_edge if inner_in_range else None,
                    outer_edge=feature.outer_edge if outer_in_range else None,
                    start_date=feature.start_date,
                    end_date=feature.end_date,
                )

        keep_inner = self._edge_passes_fade(feature, 'inner', all_edge_radii)
        keep_outer = self._edge_passes_fade(feature, 'outer', all_edge_radii)

        if keep_inner and keep_outer:
            return feature

        # Compute what the trimmed feature would look like
        new_inner = feature.inner_edge if keep_inner else None
        new_outer = feature.outer_edge if keep_outer else None

        # If both resulting edges are None (either explicitly excluded or were already
        # None on the original feature), exclude the feature entirely.
        if new_inner is None and new_outer is None:
            self._logger.debug(
                'Pass 4 (fade conflict): excluding %r -- all fade edges excluded',
                feature.key,
            )
            return None

        dropped_edges: list[str] = []
        if feature.inner_edge is not None and not keep_inner:
            dropped_edges.append('inner')
        if feature.outer_edge is not None and not keep_outer:
            dropped_edges.append('outer')
        if dropped_edges:
            self._logger.debug(
                'Pass 4 (fade conflict): trimming %r -- dropping %s edge(s) (fade too '
                'narrow after neighbor conflict)',
                feature.key,
                ' and '.join(dropped_edges),
            )

        # One real edge excluded: return a trimmed feature.
        # RingFeature is frozen, so reconstruct with the excluded edge set to None.
        return RingFeature(
            key=feature.key,
            name=feature.name,
            feature_type=feature.feature_type,
            inner_edge=new_inner,
            outer_edge=new_outer,
            start_date=feature.start_date,
            end_date=feature.end_date,
        )

    def _edge_passes_fade(
        self,
        feature: RingFeature,
        edge_type: Literal['inner', 'outer'],
        all_edge_radii: list[tuple[float, str]],
    ) -> bool:
        """Return True if this edge passes the fade conflict check.

        Non-fade edges (inner and outer edges of full ringlets where both edges are
        present) always pass because they are rendered solid and do not use fade.

        For fade-using edges:
        - Compute ``fade_width_km = fade_width_pix * min_res`` at the edge.
        - Reduce ``fade_width_km`` to ``half_dist`` when a neighbor is in the shade
          direction within the original fade zone.
        - Exclude if the reduced width < ``min_allowed_fade_width_pix * min_res``.

        The shade direction is ``shade_above`` (True = outward, False = inward):
        - RINGLET inner: shade_above=True (fade outward)
        - RINGLET outer: shade_above=False (fade inward)
        - GAP inner: shade_above=False (fade inward / into gap)
        - GAP outer: shade_above=True (fade outward / into gap)

        Parameters:
            feature: The feature whose edge is being checked.
            edge_type: 'inner' or 'outer'.
            all_edge_radii: All surviving edge radii for conflict lookup.

        Returns:
            True if the edge should be kept.
        """
        edge_data = feature.inner_edge if edge_type == 'inner' else feature.outer_edge
        if edge_data is None:
            return True

        if not feature.uses_fade_for_edge(edge_type):
            return True

        edge_a = edge_data.base_radius

        # Determine shade direction
        if feature.feature_type is RingFeatureType.RINGLET:
            shade_above = edge_type == 'inner'
        else:
            shade_above = edge_type == 'outer'
        shade_sign = 1 if shade_above else -1

        min_res = self._min_res_at_radius(edge_a)
        if min_res is None or min_res == 0.0:
            self._logger.debug(
                'Pass 4 (fade conflict): excluding %r %s edge -- resolution unavailable',
                feature.key,
                edge_type,
            )
            return False

        fade_width_km = self._fade_width_pix * min_res

        # Conflict detection: find tightest neighbor in the shade direction
        for other_a, _ in all_edge_radii:
            if other_a == edge_a:
                continue
            signed_dist = shade_sign * (other_a - edge_a)
            if signed_dist > 0:
                half_dist = signed_dist / 2.0
                if half_dist < fade_width_km:
                    fade_width_km = half_dist

        min_allowed_km = self._min_allowed_fade_width_pix * min_res
        if fade_width_km < min_allowed_km:
            self._logger.debug(
                'Pass 4 (fade conflict): excluding %r %s edge at %.1f km -- '
                'adjusted fade %.2f km < min %.2f km',
                feature.key,
                edge_type,
                edge_a,
                fade_width_km,
                min_allowed_km,
            )
            return False

        return True
