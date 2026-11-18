"""Ring feature domain object and cross-feature date-overlap validation.

This module is the core of the ring domain model. It defines:

- ``RingFeature``: An immutable domain object representing a single ring gap
  or ringlet. It owns backplane-based rendering via ``render(context)`` and
  provides query methods used by the filter and orchestrator.

- ``validate_no_date_overlaps()``: A cross-feature validation function that
  detects authoring errors in the YAML config where two features cover the
  same radial region with overlapping date ranges.

**Validation philosophy**: ``from_config()`` raises ``ValueError`` on any
malformed per-feature data (bad types, missing fields, out-of-range values).
``validate_no_date_overlaps()`` raises ``ValueError`` on cross-feature date
conflicts. Both are hard errors because bad config is an authoring mistake that
should be caught immediately, not silently degraded at render time. The
``RingFeatureFilter`` handles *valid* features that are not relevant to a
particular observation.

**Immutability**: ``RingFeature`` is a frozen dataclass. All attributes are set
at construction and never mutated. Derived cached fields (``_start_et``,
``_end_et``) are set in ``__post_init__`` using ``object.__setattr__()``, the
standard Python pattern for frozen dataclasses that need computed cached
values -- ``__post_init__`` is the only place this bypass is appropriate. After
construction completes, all fields are truly frozen.

**Rendering dispatch**: ``render()`` calls ``_compute_edge_radii()`` which uses
``RingEdgeData.parsed_modes_for_backplane()`` to get the mode tuples for the
``oops.ext_bp.radial_mode()`` backplane calls. The rendering path branches based
on feature type and edge availability:

- RINGLET with both edges -> ``_render_full_ringlet()`` -> one ``RingRenderResult``
- GAP with either or both edges, or single-edge RINGLET -> ``_render_single_edge()``
  per present edge -> one ``RingRenderResult`` per edge
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from spindoctor.support.time import utc_to_et
from spindoctor.support.types import NDArrayBoolType

from .ring_math import compute_antialiasing, compute_edge_fade
from .ring_render_context import RingsRenderContext
from .ring_render_result import RingRenderResult
from .ring_types import RingBaseOrbitMode, RingEdgeData, RingFeatureType, RingPerturbationMode

# ``compute_antialiasing`` varies only for |radius - edge| <= 0.5 * resolution;
# beyond that it is clamped to 0 or 1. Applying those plateaus on every
# ``not_solid`` pixel would paint a half-plane of constant model and a nearly
# all-True mask; we keep contributions only in the edge band.
_AA_EDGE_BAND_HALF_WIDTH = 0.5


@dataclass(frozen=True)
class RingFeature:
    """A single ring feature (gap or ringlet) with backplane rendering capability.

    Frozen dataclass: all attributes are set at construction and never mutated.
    ``render()`` takes context and returns results without modifying feature state.
    This immutability guarantees that feature data loaded from YAML config remains
    consistent throughout the pipeline: multiple filter passes, render calls, and
    annotation creation all see the same data.

    Owns backplane-based rendering for real observations.  (Simulated
    observations never build RingFeatures: ``NavModelRingsSimulated``
    predicts scene ring_system features through
    ``spindoctor.nav_model.simulated.predict_ring`` instead.)
    """

    key: str
    name: str | None
    feature_type: RingFeatureType
    inner_edge: RingEdgeData | None
    outer_edge: RingEdgeData | None
    start_date: str | None = None
    end_date: str | None = None
    # Cached ET values computed in __post_init__ via object.__setattr__
    _start_et: float | None = field(init=False, repr=False, default=None)
    _end_et: float | None = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        """Validate and cache date conversions.

        Uses ``object.__setattr__()`` to set derived fields on a frozen
        dataclass. This is the standard Python pattern for frozen dataclasses
        that need computed cached values -- ``__post_init__`` is the only place
        where this bypass is appropriate. After construction completes, all
        fields are truly frozen.

        Raises:
            ValueError: If both inner_edge and outer_edge are None, or if both
                dates convert to ET and ``start_date`` does not strictly precede
                ``end_date``.
        """
        if self.inner_edge is None and self.outer_edge is None:
            raise ValueError(
                f'RingFeature {self.key!r}: at least one edge (inner or outer) is required'
            )

        start_et: float | None = None
        end_et: float | None = None
        if self.start_date is not None:
            start_et = utc_to_et(self.start_date)
        if self.end_date is not None:
            end_et = utc_to_et(self.end_date)
        if start_et is not None and end_et is not None and start_et >= end_et:
            raise ValueError(
                f'RingFeature {self.key!r}: start_date must be strictly before end_date '
                f'([start_date, end_date) is half-open)'
            )
        object.__setattr__(self, '_start_et', start_et)
        object.__setattr__(self, '_end_et', end_et)

    # ------------------------------------------------------------------
    # Query methods (used by orchestrator and filter)
    # ------------------------------------------------------------------

    def is_visible_at(self, obs_time_et: float) -> bool:
        """Return True if this feature is valid at the given observation time.

        A feature with no date range is always visible. If only ``start_date``
        is set, the feature is visible at and after that date. If only
        ``end_date`` is set, the feature is visible before that date.
        The range is half-open: [start_et, end_et).

        Parameters:
            obs_time_et: Observation time in TDB seconds (from ``utc_to_et``).

        Returns:
            True if the feature is active at ``obs_time_et``.
        """
        if self._start_et is not None and obs_time_et < self._start_et:
            return False
        return not (self._end_et is not None and obs_time_et >= self._end_et)

    def is_in_radius_range(self, min_r: float, max_r: float) -> bool:
        """Return True if at least one edge is within the given radius range.

        A feature is kept if ANY of its edges falls in ``[min_r, max_r]``.
        This enables partial visibility: a ringlet with one edge in range
        and one out of range is rendered as a single-edge feature by
        ``render()``.

        Parameters:
            min_r: Minimum ring radius in km (inclusive).
            max_r: Maximum ring radius in km (inclusive).

        Returns:
            True if at least one edge base radius is in ``[min_r, max_r]``.
        """
        if self.inner_edge is not None:
            a = self.inner_edge.base_radius
            if min_r <= a <= max_r:
                return True
        if self.outer_edge is not None:
            a = self.outer_edge.base_radius
            if min_r <= a <= max_r:
                return True
        return False

    @property
    def max_extent_radius(self) -> float:
        """Maximum possible radius (a + ae) across all present edges.

        Returns the outermost radius the feature could occupy at any longitude,
        accounting for eccentricity. Used for a fast pre-filter check: if the
        minimum observed ring radius in the FOV exceeds this value, none of the
        feature can appear in the image regardless of orientation.

        Returns:
            Maximum of (base_orbit.a + base_orbit.ae) for all present edges.

        Raises:
            ValueError: If both ``inner_edge`` and ``outer_edge`` are ``None``,
                so ``max_extent_radius`` cannot be computed.
        """
        extents: list[float] = []
        if self.inner_edge is not None:
            bo = self.inner_edge.base_orbit
            extents.append(bo.a + bo.ae)
        if self.outer_edge is not None:
            bo = self.outer_edge.base_orbit
            extents.append(bo.a + bo.ae)
        if len(extents) == 0:
            raise ValueError(
                'RingFeature has no edges; cannot compute max_extent_radius '
                '(both inner_edge and outer_edge are None)'
            )
        return max(extents)

    def all_base_radii(self) -> list[tuple[float, str]]:
        """Return (radius_km, edge_label) pairs for all present edges.

        Edge labels follow the convention:
        - RINGLET inner edge: 'IER' (Inner Edge Ringlet)
        - RINGLET outer edge: 'OER' (Outer Edge Ringlet)
        - GAP inner edge: 'IEG' (Inner Edge Gap)
        - GAP outer edge: 'OEG' (Outer Edge Gap)

        Returns:
            List of (radius_km, label) tuples for present edges.
        """
        result: list[tuple[float, str]] = []
        labels = self.edge_labels
        if self.inner_edge is not None:
            result.append((self.inner_edge.base_radius, labels['inner']))
        if self.outer_edge is not None:
            result.append((self.outer_edge.base_radius, labels['outer']))
        return result

    def uses_fade_for_edge(self, edge_type: str) -> bool:
        """Return True if the given edge uses fade rendering by structure.

        Structural check based on feature configuration:

        - GAP edges always use fade (gaps render a fading gradient from each
          known edge; they do not fill solid between edges).
        - RINGLET edges use fade only when the feature has a single edge (the
          other is None).

        This is a structural check. The ``RingFeatureFilter`` augments this with
        partial-visibility awareness: if a RINGLET has both edges but one is
        out of the visible radius range, the filter treats the in-range edge as
        fade-using even though this method returns False.

        Parameters:
            edge_type: 'inner' or 'outer'.

        Returns:
            True if this edge uses fade rendering by structure.

        Raises:
            ValueError: If ``edge_type`` is not exactly ``'inner'`` or ``'outer'``.
        """
        if edge_type not in ('inner', 'outer'):
            raise ValueError(
                f"edge_type must be 'inner' or 'outer' (case-sensitive), got {edge_type!r}"
            )
        if self.feature_type is RingFeatureType.GAP:
            return True
        # RINGLET: uses fade only when this is a single-edge feature
        if edge_type == 'inner':
            return self.outer_edge is None
        return self.inner_edge is None

    @property
    def uncertainty(self) -> float:
        """Maximum RMS across all present edges (km).

        The maximum (rather than minimum or average) is conservative: the
        overall uncertainty of a feature is dominated by its least
        well-characterized edge.

        Returns:
            Max of inner and outer edge RMS values, or the single edge RMS
            if only one edge is present.
        """
        inner_rms = self.inner_edge.rms if self.inner_edge is not None else None
        outer_rms = self.outer_edge.rms if self.outer_edge is not None else None
        if inner_rms is not None and outer_rms is not None:
            return max(inner_rms, outer_rms)
        if inner_rms is not None:
            return inner_rms
        if outer_rms is not None:
            return outer_rms
        return 0.0  # unreachable: __post_init__ ensures at least one edge

    def edge_uncertainty(self, edge_type: str) -> float:
        """RMS (km) of one named edge's own orbit solution.

        Unlike :attr:`uncertainty` (the max across the feature's edges, a
        deliberately conservative per-feature scale for the robust fit's
        per-vertex weighting), this is the requested edge's own value.  A
        radial displacement bound belongs to the edge whose orbit solution it
        describes: on the Keeler-A Ring OE ringlet the inner edge is fitted to
        1.006 km while the outer is 10.18 km, and charging the inner edge the
        outer's quality would price a well-determined edge at its sibling's.

        Parameters:
            edge_type: The side name (``'inner'`` / ``'outer'``) or this
                feature's catalog edge label for that side (``'IER'`` /
                ``'OER'`` for a ringlet, ``'IEG'`` / ``'OEG'`` for a gap).
                Both spellings are accepted because the render pipeline
                carries the catalog label on its edge tuples while callers
                working from the feature structure use the side name.

        Returns:
            That edge's RMS in km, or ``0.0`` when the named edge is absent
            (a single-edge feature asked for the missing side).

        Raises:
            ValueError: If ``edge_type`` names neither a side nor one of this
                feature's edge labels.
        """
        side = edge_type
        if side not in ('inner', 'outer'):
            label_to_side = {label: name for name, label in self.edge_labels.items()}
            if side not in label_to_side:
                raise ValueError(
                    f"edge_type must be 'inner' / 'outer' or one of this feature's edge "
                    f'labels {sorted(label_to_side)}; got {edge_type!r}'
                )
            side = label_to_side[side]
        edge = self.inner_edge if side == 'inner' else self.outer_edge
        return 0.0 if edge is None else float(edge.rms)

    @property
    def edge_labels(self) -> dict[str, str]:
        """Map of 'inner'/'outer' to edge label string.

        Returns:
            Dict with keys 'inner' and 'outer' mapping to label strings
            ('IER'/'OER' for ringlets, 'IEG'/'OEG' for gaps).
        """
        if self.feature_type is RingFeatureType.GAP:
            return {'inner': 'IEG', 'outer': 'OEG'}
        return {'inner': 'IER', 'outer': 'OER'}

    # ------------------------------------------------------------------
    # Backplane rendering
    # ------------------------------------------------------------------

    def render(self, context: RingsRenderContext) -> list[RingRenderResult]:
        """Render this feature using backplane data.

        Dispatch logic:

        - RINGLET with both edges visible -> ``_render_full_ringlet()`` -> one result
        - GAP with any edges, or single-edge RINGLET -> ``_render_single_edge()``
          per edge -> one result per edge

        For RINGLETs with one edge out of the visible radius range (partial
        visibility), ``RingFeatureFilter`` trims the out-of-range edge to ``None``
        before this method is called, so ``render()`` naturally takes the
        single-edge path for the remaining in-range edge (fade rendering).

        Parameters:
            context: Immutable rendering context with obs, ring_target, epoch,
                per-pixel resolutions, fade config, and all_edge_radii.

        Returns:
            List of ``RingRenderResult`` objects. Typically one result for full
            ringlets and one per edge for gaps and single-edge features.
        """
        inner_radii_bp = None
        outer_radii_bp = None

        if self.inner_edge is not None:
            inner_radii_bp = self._compute_edge_radii(context, self.inner_edge)
        if self.outer_edge is not None:
            outer_radii_bp = self._compute_edge_radii(context, self.outer_edge)

        labels = self.edge_labels
        results: list[RingRenderResult] = []

        if (
            self.feature_type is RingFeatureType.RINGLET
            and inner_radii_bp is not None
            and outer_radii_bp is not None
        ):
            context.logger.debug(
                'RingFeature %r: rendering full ringlet (solid band + edge anti-aliasing)',
                self.key,
            )
            result = self._render_full_ringlet(context, inner_radii_bp, outer_radii_bp, labels)
            if result is not None:
                results.append(result)
        else:
            context.logger.debug(
                'RingFeature %r: rendering per-edge (GAP or partial/single-edge RINGLET)',
                self.key,
            )
            # GAP or single-edge: one result per edge
            for edge_data, radii_bp, edge_type in [
                (self.inner_edge, inner_radii_bp, 'inner'),
                (self.outer_edge, outer_radii_bp, 'outer'),
            ]:
                if edge_data is None or radii_bp is None:
                    continue
                result = self._render_single_edge(context, edge_data, radii_bp, edge_type, labels)
                if result is not None:
                    results.append(result)

        return results

    def _compute_edge_radii(self, context: RingsRenderContext, edge: RingEdgeData) -> Any:
        """Compute multi-mode edge radius backplane.

        Uses ``RingEdgeData.parsed_modes_for_backplane()`` to get the mode
        tuples, then applies them sequentially via
        ``context.obs.ext_bp.radial_mode()``.

        The first mode (mode 1, base orbit) calls ``radial_mode`` with the
        backplane key, mode number, epoch, ``ae``, longitude and rate of
        periapsis (radians / rad/s), plus ``a0=<semi-major axis>`` as a keyword
        argument (six positional values plus ``a0``). Subsequent modes use the
        four-argument perturbation form.

        Parameters:
            context: Current rendering context.
            edge: Edge data including base orbit and perturbations.

        Returns:
            Backplane object with computed per-pixel radii.
        """
        parsed = edge.parsed_modes_for_backplane()
        radii_bp = context.obs.ext_bp.ring_radius(context.ring_target)

        for mode_info in parsed:
            if len(mode_info) == 5:
                mode, a, ae, long_peri_rad, rate_peri_rad_per_sec = mode_info
                radii_bp = context.obs.ext_bp.radial_mode(
                    context.obs.ext_bp.standardize_backplane_key(radii_bp),
                    mode,
                    context.epoch,
                    ae,
                    long_peri_rad,
                    rate_peri_rad_per_sec,
                    a0=a,
                )
            else:
                mode, amplitude, phase_rad, speed_rad_per_sec = mode_info
                radii_bp = context.obs.ext_bp.radial_mode(
                    context.obs.ext_bp.standardize_backplane_key(radii_bp),
                    mode,
                    context.epoch,
                    amplitude,
                    phase_rad,
                    speed_rad_per_sec,
                )

        return radii_bp

    def _render_full_ringlet(
        self,
        context: RingsRenderContext,
        inner_radii_bp: Any,
        outer_radii_bp: Any,
        labels: dict[str, str],
    ) -> RingRenderResult | None:
        """Render a complete ringlet with solid fill and anti-aliasing.

        Fills the region between inner and outer edges with value 1.0, then
        applies anti-aliasing at both edges. ``compute_antialiasing`` is clipped
        to 0 or 1 outside half a resolution of each edge; only the transitional
        band contributes on ``not_solid`` pixels so the model does not fill a
        half-plane with a constant plateau. The returned mask matches pixels
        with positive model value.

        Edge info for annotations is computed here using the already-computed
        backplanes, avoiding a second backplane computation pass.

        Parameters:
            context: Rendering context.
            inner_radii_bp: Pre-computed inner edge radius backplane.
            outer_radii_bp: Pre-computed outer edge radius backplane.
            labels: Edge label dict from ``edge_labels``.

        Returns:
            ``RingRenderResult`` or None if edge radii could not be determined.
        """
        if self.inner_edge is None:
            raise ValueError(f'RingFeature {self.key!r}: _render_full_ringlet requires inner_edge')
        if self.outer_edge is None:
            raise ValueError(f'RingFeature {self.key!r}: _render_full_ringlet requires outer_edge')

        inner_a = self.inner_edge.base_radius
        outer_a = self.outer_edge.base_radius
        resolutions = context.resolutions
        context.logger.debug(
            'RingFeature %r: full ringlet inner_a=%.3f km outer_a=%.3f km (solid + AA)',
            self.key,
            inner_a,
            outer_a,
        )

        inner_radii = inner_radii_bp.mvals.filled(0.0)
        outer_radii = outer_radii_bp.mvals.filled(0.0)

        shape = resolutions.shape
        model = np.zeros(shape, dtype=np.float64)
        mask = np.zeros(shape, dtype=bool)

        # Solid region between edges
        inner_above = inner_radii - resolutions / 2.0 >= inner_a
        outer_below = outer_radii + resolutions / 2.0 <= outer_a
        solid = np.logical_and(inner_above, outer_below)
        if hasattr(solid, 'filled'):
            solid = solid.filled(False)
        solid = np.asarray(solid, dtype=bool)
        model[solid] += 1.0
        mask[solid] = True

        # Anti-aliasing at inner and outer edges
        inner_shade = compute_antialiasing(
            radii=inner_radii,
            edge_radius=inner_a,
            shade_above=False,
            resolutions=resolutions,
        )
        outer_shade = compute_antialiasing(
            radii=outer_radii,
            edge_radius=outer_a,
            shade_above=True,
            resolutions=resolutions,
        )
        half_w = _AA_EDGE_BAND_HALF_WIDTH * resolutions
        inner_band = np.abs(inner_radii - inner_a) <= half_w + 1e-12
        outer_band = np.abs(outer_radii - outer_a) <= half_w + 1e-12
        inner_shade = np.where(inner_band, inner_shade, 0.0)
        outer_shade = np.where(outer_band, outer_shade, 0.0)

        not_solid = ~solid
        model[not_solid] += inner_shade[not_solid]
        model[not_solid] += outer_shade[not_solid]
        mask[not_solid] |= inner_shade[not_solid] > 0.0
        mask[not_solid] |= outer_shade[not_solid] > 0.0

        model = np.maximum(model, 0.0)
        mask = model > 0.0
        model = np.where(mask, model, 0.0)

        # Build annotation edge masks using already-computed backplanes
        edge_info_list: list[tuple[NDArrayBoolType, str, str]] = []
        for edge_bp, a, edge_type in [
            (inner_radii_bp, inner_a, 'inner'),
            (outer_radii_bp, outer_a, 'outer'),
        ]:
            label = labels[edge_type]
            feature_name = self.name or 'UNNAMED'
            edge_key = context.obs.ext_bp.standardize_backplane_key(edge_bp)
            edge_mask = (
                context.obs.ext_bp.border_atop(edge_key, a).mvals.astype('bool').filled(False)
            )
            edge_info_list.append((edge_mask, f'{feature_name} {label}', label))

        return RingRenderResult(
            model_img=model,
            model_mask=mask,
            uncertainty=self.uncertainty,
            edge_info_list=edge_info_list,
        )

    def _render_single_edge(
        self,
        context: RingsRenderContext,
        edge_data: RingEdgeData,
        radii_bp: Any,
        edge_type: str,
        labels: dict[str, str],
    ) -> RingRenderResult | None:
        """Render a single edge with a linear fade gradient.

        Used for:
        - GAP features (all edges use fade)
        - RINGLET features with only one edge defined
        - RINGLET features where the other edge is out of the visible radius range

        The fade direction is determined by feature type and edge type:
        - RINGLET inner edge: shade_above=True (fade away from planet)
        - RINGLET outer edge: shade_above=False (fade toward planet)
        - GAP inner edge: shade_above=False (fade toward planet / into gap)
        - GAP outer edge: shade_above=True (fade away from planet / into gap)

        Parameters:
            context: Rendering context.
            edge_data: Data for this specific edge.
            radii_bp: Pre-computed radius backplane for this edge.
            edge_type: 'inner' or 'outer'.
            labels: Edge label dict.

        Returns:
            ``RingRenderResult`` or None if rendering failed.
        """
        edge_a = edge_data.base_radius
        edge_radii = radii_bp.mvals.filled(0.0)
        resolutions = context.resolutions
        shape = resolutions.shape

        # Shade direction: ringlet inner fades outward; gap outer fades outward
        if self.feature_type is RingFeatureType.RINGLET:
            shade_above = edge_type == 'inner'
        else:
            shade_above = edge_type == 'outer'

        context.logger.debug(
            'RingFeature %r: single-edge %s fade (edge_a=%.3f km, shade_above=%s, '
            'fade_width_pix=%.2f)',
            self.key,
            edge_type,
            edge_a,
            shade_above,
            context.fade_width_pix,
        )

        model = np.zeros(shape, dtype=np.float64)
        new_model = compute_edge_fade(
            model=model,
            radii=edge_radii,
            edge_radius=edge_a,
            shade_above=shade_above,
            fade_width_pix=context.fade_width_pix,
            resolutions=resolutions,
            all_edge_radii=context.all_edge_radii,
            logger=context.logger,
        )

        fade_mask = (new_model - model) > 0.0
        mask = np.zeros(shape, dtype=bool)
        mask[fade_mask] = True

        label = labels[edge_type]
        feature_name = self.name or 'UNNAMED'
        radii_key = context.obs.ext_bp.standardize_backplane_key(radii_bp)
        edge_mask: NDArrayBoolType = (
            context.obs.ext_bp.border_atop(radii_key, edge_a).mvals.astype('bool').filled(False)
        )
        edge_info_list: list[tuple[NDArrayBoolType, str, str]] = [
            (edge_mask, f'{feature_name} {label}', label)
        ]

        return RingRenderResult(
            model_img=new_model,
            model_mask=mask,
            uncertainty=self.uncertainty,
            edge_info_list=edge_info_list,
        )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, key: str, data: dict[str, Any]) -> 'RingFeature':
        """Construct a RingFeature from a YAML feature dictionary.

        Validates all fields at construction time. This follows the principle
        that bad config is an authoring error that should fail loudly and
        immediately, not silently degrade at render time.

        Parameters:
            key: Feature key (YAML dict key used as identifier).
            data: Feature dictionary with keys:
                - ``feature_type``: 'GAP' or 'RINGLET' (required)
                - ``name``: Human-readable name (optional)
                - ``inner_data``: List of mode dicts (optional)
                - ``outer_data``: List of mode dicts (optional)
                - ``start_date``: ISO date string (optional)
                - ``end_date``: ISO date string (optional)

        Returns:
            Constructed ``RingFeature`` instance.

        Raises:
            TypeError: If ``key`` is not a ``str`` or ``data`` is not a ``dict``.
            ValueError: On any structural or value error:
                - feature_type not 'GAP' or 'RINGLET'
                - neither inner_data nor outer_data present
                - mode data is not a non-empty list
                - mode-1 data missing or has non-positive 'a'
                - rms < 0
                - perturbation mode missing required fields
        """
        if not isinstance(key, str):
            raise TypeError(f'Feature key must be str, got {type(key).__name__}')
        if not isinstance(data, dict):
            raise TypeError(f'Feature {key!r}: data must be a dict, got {type(data).__name__}')

        # Validate feature_type
        raw_type = data.get('feature_type')
        try:
            feature_type = RingFeatureType(raw_type)
        except ValueError as exc:
            raise ValueError(
                f'Feature {key!r}: invalid feature_type {raw_type!r}; expected "GAP" or "RINGLET"'
            ) from exc

        name: str | None = data.get('name')
        start_date: str | None = data.get('start_date')
        end_date: str | None = data.get('end_date')

        raw_inner = data.get('inner_data')
        raw_outer = data.get('outer_data')

        if raw_inner is None and raw_outer is None:
            raise ValueError(
                f'Feature {key!r}: at least one edge (inner_data or outer_data) is required'
            )

        inner_edge: RingEdgeData | None = None
        outer_edge: RingEdgeData | None = None

        if raw_inner is not None:
            inner_edge = _parse_edge_data(key, 'inner', raw_inner)
        if raw_outer is not None:
            outer_edge = _parse_edge_data(key, 'outer', raw_outer)

        return cls(
            key=key,
            name=name,
            feature_type=feature_type,
            inner_edge=inner_edge,
            outer_edge=outer_edge,
            start_date=start_date,
            end_date=end_date,
        )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _integral_mode_num(feature_key: str, edge_type: str, index: int, mode_num: Any) -> int:
    """Parse YAML ``mode`` as ``int``; reject ``bool`` and non-integral floats."""
    if isinstance(mode_num, bool):
        raise ValueError(
            f'Feature {feature_key!r} {edge_type}_data[{index}]: '
            f'mode_num must be int (not bool), got {mode_num!r}'
        )
    if isinstance(mode_num, int):
        return mode_num
    if isinstance(mode_num, float) and math.isfinite(mode_num) and mode_num == math.floor(mode_num):
        return int(mode_num)
    raise ValueError(
        f'Feature {feature_key!r} {edge_type}_data[{index}]: '
        f'mode_num must be an integer, got {mode_num!r}'
    )


def _parse_edge_data(feature_key: str, edge_type: str, mode_list: Any) -> RingEdgeData:
    """Parse a list of mode dicts into a RingEdgeData.

    Parameters:
        feature_key: Feature key for error messages.
        edge_type: 'inner' or 'outer' for error messages.
        mode_list: Should be a non-empty list of mode dicts.

    Returns:
        Parsed ``RingEdgeData``.

    Raises:
        ValueError: On any structural or value error, including perturbation mode
            dicts that omit ``amplitude``, ``phase``, or ``pattern_speed``.
    """
    if not isinstance(mode_list, list) or len(mode_list) == 0:
        raise ValueError(
            f'Feature {feature_key!r} {edge_type}_data: '
            f'expected a non-empty list of mode dicts, got {type(mode_list).__name__!r}'
        )

    base_orbit: RingBaseOrbitMode | None = None
    perturbations: list[RingPerturbationMode] = []

    for i, mode in enumerate(mode_list):
        if not isinstance(mode, dict):
            raise ValueError(
                f'Feature {feature_key!r} {edge_type}_data[{i}]: '
                f'expected dict, got {type(mode).__name__!r}'
            )

        mode_num = mode.get('mode')
        if mode_num is None:
            raise ValueError(f'Feature {feature_key!r} {edge_type}_data[{i}]: missing "mode" field')

        if 'a' in mode:
            # Base-orbit entry: must be mode 1 (semi-major axis and related fields).
            mode_n_base = _integral_mode_num(feature_key, edge_type, i, mode_num)
            if mode_n_base != 1:
                raise ValueError(
                    f'Feature {feature_key!r} {edge_type}_data[{i}]: entry with "a" must use '
                    f'mode 1 (base orbit), got mode {mode_num!r}'
                )
            if base_orbit is not None:
                raise ValueError(
                    f'Feature {feature_key!r} {edge_type}_data[{i}]: duplicate base-orbit entry '
                    f'(only one mode dict with "a" allowed per edge); '
                    f'earlier entry already defined the mode-1 base orbit'
                )
            base_orbit = RingBaseOrbitMode(
                a=mode['a'],
                ae=mode.get('ae', 0.0),
                long_peri=mode.get('long_peri', 0.0),
                rate_peri=mode.get('rate_peri', 0.0),
                rms=mode.get('rms', 0.0),
            )
        else:
            # Perturbation mode: require explicit YAML keys; validation by
            # ``RingPerturbationMode.__post_init__`` (no silent 0.0 defaults).
            mode_n = _integral_mode_num(feature_key, edge_type, i, mode_num)
            for field_name in ('amplitude', 'phase', 'pattern_speed'):
                if field_name not in mode:
                    raise ValueError(
                        f'Feature {feature_key!r} {edge_type}_data[{i}]: '
                        f'perturbation mode dict missing required key {field_name!r}'
                    )
            perturbations.append(
                RingPerturbationMode(
                    mode_num=mode_n,
                    amplitude=mode['amplitude'],
                    phase=mode['phase'],
                    pattern_speed=mode['pattern_speed'],
                )
            )

    if base_orbit is None:
        raise ValueError(
            f'Feature {feature_key!r} {edge_type}_data: missing mode 1 (base orbit with "a" field)'
        )

    return RingEdgeData(base_orbit=base_orbit, perturbations=tuple(perturbations))


# ---------------------------------------------------------------------------
# Cross-feature date-overlap validation
# ---------------------------------------------------------------------------


def validate_no_date_overlaps(features: Sequence[RingFeature]) -> None:
    """Cross-feature validation: detect date-range overlaps in the same radial region.

    Two features "overlap" if their date ranges intersect AND their radial
    extents intersect. This catches authoring errors in the YAML config where
    the same ring edge is defined twice with overlapping validity periods.

    This function runs after all features are loaded via ``from_config()`` and
    before the runtime filter. It is a hard error (``ValueError``) because
    overlapping dates for the same radial region is a config authoring mistake,
    not an observation-dependent condition. The filter handles *valid* features
    that are not relevant for a particular observation.

    Parameters:
        features: All features loaded from one planet's config.

    Raises:
        ValueError: If any pair of features has overlapping dates AND overlapping
            radial extents.
    """
    feature_list = list(features)
    for i, feat_a in enumerate(feature_list):
        for feat_b in feature_list[i + 1 :]:
            if _radial_extents_overlap(feat_a, feat_b) and _dates_overlap(feat_a, feat_b):
                raise ValueError(
                    f'Ring config error: features {feat_a.key!r} and {feat_b.key!r} '
                    f'have overlapping date ranges and overlapping radial extents. '
                    f'Check the YAML config -- features should have non-overlapping '
                    f'date ranges if they cover the same radial region.'
                )


def _radial_extents_overlap(a: RingFeature, b: RingFeature) -> bool:
    """Return True if the radial extents of two features overlap."""
    radii_a = [r for r, _ in a.all_base_radii()]
    radii_b = [r for r, _ in b.all_base_radii()]
    if not radii_a or not radii_b:
        return False
    min_a, max_a = min(radii_a), max(radii_a)
    min_b, max_b = min(radii_b), max(radii_b)
    # Two intervals [min_a, max_a] and [min_b, max_b] overlap iff
    # one doesn't entirely lie below the other
    return max_a >= min_b and max_b >= min_a


def _dates_overlap(a: RingFeature, b: RingFeature) -> bool:
    """Return True if two features with explicit date ranges overlap.

    Only checks features that BOTH have explicit date ranges. If either feature
    has no dates (always active), it is not considered a date-range conflict --
    having a timeless feature alongside a dated feature is intentional
    (e.g., a feature that exists throughout the mission alongside a dated
    variant that replaces it for a specific period would be caught by the
    radial overlap check alone, but both-active is a normal pattern in config).

    The half-open interval check: [a_start, a_end) and [b_start, b_end) overlap
    if a_start < b_end AND b_start < a_end.
    """
    # If either feature has no dates, skip the overlap check
    if a._start_et is None and a._end_et is None:
        return False
    if b._start_et is None and b._end_et is None:
        return False

    # Both have at least one explicit date; treat None as infinite extent
    a_start = a._start_et if a._start_et is not None else float('-inf')
    a_end = a._end_et if a._end_et is not None else float('inf')
    b_start = b._start_et if b._start_et is not None else float('-inf')
    b_end = b._end_et if b._end_et is not None else float('inf')

    return a_start < b_end and b_start < a_end
