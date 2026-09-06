"""Catalog-driven ring NavModel.

The orchestrator iterates one ``NavModelRings`` per planet whose ring
system has any radius inside the extfov.  Per ring feature surviving
the four-pass ``RingFeatureFilter``, the model emits either:

- one ``RING_EDGE`` ``NavFeature`` carrying a per-vertex ``RingEdgePolyline``
  with its own sigma_radial / sigma_along_edge derived from the catalog ``rms``,
  or
- one ``RING_ANNULUS`` ``NavFeature`` carrying the rendered annulus
  template when the surviving edge polyline compresses radially below
  the resolvability threshold.

The per-edge polylines come from sampling the rendered model + edge
mask: each ``True`` pixel in the edge mask contributes one polyline
vertex, and the gradient direction across the edge gives the radial
normal.  Curvature is measured by the maximum deviation of the polyline
from its best-fit straight line.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
import oops
from oops import Meshgrid
from oops.backplane import Backplane

from spindoctor.annotation import Annotations
from spindoctor.config import DEFAULT_CONFIG, Config
from spindoctor.feature.feature import NavFeature
from spindoctor.nav_model.nav_model import NavModel
from spindoctor.nav_model.nav_model_rings_base import NavModelRingsBase
from spindoctor.nav_model.ring_emission import (
    RING_EDGE_DEFAULT_RELIABILITY,
    RING_EDGE_SIGMA_ALONG_PX,
    _aggregate_annulus_orbit_terms,
    _build_annulus_feature,
    _build_edge_feature,
)
from spindoctor.nav_model.ring_polyline import (
    FLAT_CURVATURE_THRESHOLD_PX,
    _composite_ring_renderings,
    _is_straight_line,
    _mask_bbox,
    _median_radial_scale,
    _polyline_from_edge_mask,
    _radial_extent_px,
)
from spindoctor.nav_model.rings import (
    RingFeature,
    RingFeatureFilter,
    RingsRenderContext,
    validate_no_date_overlaps,
)
from spindoctor.support.time import now_dt, utc_to_et
from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    from spindoctor.nav_orchestrator.nav_context import NavContext


__all__ = [
    'FLAT_CURVATURE_THRESHOLD_PX',
    'RING_EDGE_DEFAULT_RELIABILITY',
    'RING_EDGE_SIGMA_ALONG_PX',
    'NavModelRings',
]


SPARSE_VISIBILITY_GRID_SIZE: int = 16
"""Side length of the sparse-grid backplane used by the cheap ring pre-check.

A 16x16 ring-radius backplane evaluates ~256 SPICE ray casts (vs ~1M+
for a 1024x1024 ext-FOV) and runs in well under 10 ms; it is sufficient
to detect the two common skip cases — no ring-plane intersection in
the FOV at all, and a visible radial range entirely beyond the
catalog's outermost feature.
"""


# All ring-annulus emission tunables live in
# ``config_files/config_510_techniques.yaml`` under
# ``feature_emission.ring_annulus`` with per-planet overrides.  No
# Python-level fallback; missing-key access in
# ``_ring_annulus_emission_params`` is a KeyError so a config typo
# fails fast at process startup.


def _ring_annulus_emission_params(config: Config, planet: str) -> tuple[float, float]:
    """Return ``(max_radial_px, kmpp_threshold)`` for ``planet``.

    Reads
    ``config.feature_emission.ring_annulus.planets[planet]`` first,
    then falls back to ``config.feature_emission.ring_annulus.default``.
    Both fields (``max_radial_px`` and ``kmpp_threshold``) are
    independently looked up, so a planet block may set only one and
    inherit the other from default.  A missing default-block field is
    a ``KeyError`` so a config typo fails fast at process startup.

    Parameters:
        config: Active :class:`~spindoctor.config.Config`.
        planet: SPICE planet name (uppercase), e.g. ``'SATURN'``.

    Returns:
        ``(max_radial_px, kmpp_threshold)`` in pixels and km/px.

    Raises:
        KeyError: If neither the per-planet block nor the default
            block defines a required field.
    """
    block = dict(config.category('feature_emission').get('ring_annulus', {}))
    default_block = block.get('default', {}) or {}
    planets_block = block.get('planets', {}) or {}
    planet_block = planets_block.get(planet, {}) or {}

    def _lookup(field: str) -> float:
        """Read one emission field, preferring the planet block over the default.

        Parameters:
            field: Field name to read.

        Returns:
            The field's value as a float.

        Raises:
            KeyError: If neither the planet block nor the default block
                defines the field, so a config typo fails at startup.
        """
        if field in planet_block:
            return float(planet_block[field])
        if field in default_block:
            return float(default_block[field])
        raise KeyError(
            f'feature_emission.ring_annulus: missing field {field!r} for planet '
            f'{planet!r} (no per-planet override and no default block entry)'
        )

    return _lookup('max_radial_px'), _lookup('kmpp_threshold')


def _require_positive_finite_planet_scalar(
    planet: str, key: str, planet_config: dict[str, Any]
) -> float:
    """Return a positive finite float from ``planet_config[key]``.

    Performs explicit type and finiteness checks so misconfigured YAML
    fails at config-read, not in the math.

    Parameters:
        planet: Planet name (used in error messages).
        key: Config key to read.
        planet_config: The per-planet config dict.

    Returns:
        A positive finite float.

    Raises:
        ValueError: If ``key`` is missing or the value is not a positive
            finite float.
    """
    if key not in planet_config:
        raise ValueError(f'Missing required ring configuration key {key!r} for planet {planet}')
    raw: Any = planet_config[key]
    if isinstance(raw, bool):
        raise ValueError(
            f'Invalid {key} for planet {planet}: expected a finite numeric value, got bool'
        )
    try:
        v = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f'Invalid {key} for planet {planet}: expected a finite numeric value, got {raw!r}'
        ) from exc
    if not math.isfinite(v):
        raise ValueError(f'Invalid {key} {v} for planet {planet} (must be finite)')
    if v <= 0.0:
        raise ValueError(f'Invalid {key} {v} for planet {planet}')
    return v


class NavModelRings(NavModelRingsBase):
    """Catalog-driven ring NavModel for one planet.

    Parameters:
        name: Model instance name (e.g. ``'rings:SATURN'``).
        obs: Observation snapshot.
        config: Optional ``Config`` override.
    """

    _abstract = False

    def __init__(
        self,
        name: str,
        obs: oops.Observation,
        *,
        config: Config | None = None,
    ) -> None:
        """Build the per-planet ring model with empty render state.

        Parameters:
            name: Model instance name (e.g. ``'rings:SATURN'``).
            obs: Observation snapshot the model renders against.
            config: Optional ``Config`` override; ``None`` uses
                ``DEFAULT_CONFIG``.
        """
        super().__init__(name, obs, config=config)
        self._planet: str | None = None
        self._render_results: list[
            tuple[
                RingFeature,
                NDArrayFloatType,
                NDArrayBoolType,
                float,
                list[tuple[NDArrayBoolType, str, str]],
            ]
        ] = []
        self._km_per_pixel_radial: float = 0.0
        self._radial_resolution_ext: NDArrayFloatType | None = None
        self._ring_radius_ext: NDArrayFloatType | None = None
        self._ring_occluded_ext: NDArrayBoolType | None = None
        self._extfov_v_size: int = 0
        self._extfov_u_size: int = 0
        self._predicted_center_vu: tuple[float, float] = (0.0, 0.0)
        self._subject_range_km: float = float('inf')

    @classmethod
    def instances_for_obs(
        cls, obs: oops.Observation, *, config: Config | None = None
    ) -> list[NavModel]:
        """Return one NavModelRings per planet whose ring catalog touches the FOV.

        Parameters:
            obs: Observation snapshot.
            config: Configuration whose ring catalog decides which planets have
                navigable rings; also passed to the constructed instances.  None
                uses ``DEFAULT_CONFIG``.

        Returns:
            ``[NavModelRings]`` for the closest planet when its ring catalog is
            configured and the obs exposes the extfov surface, else ``[]``.
        """
        # Simulated obs use the sim-params-driven NavModelRingsSimulated instead.
        if getattr(obs, 'is_simulated', False):
            return []
        planet = getattr(obs, 'closest_planet', None)
        if planet is None:
            return []
        # The model also needs the extfov shape and the ext_bp surface; obs
        # stand-ins that don't expose those are not applicable.
        if not hasattr(obs, 'extdata_shape_vu') or not hasattr(obs, 'ext_bp'):
            return []
        rings_config = (config if config is not None else DEFAULT_CONFIG).rings
        ring_features_dict = getattr(rings_config, 'ring_features', None)
        if not ring_features_dict or planet not in ring_features_dict:
            return []
        return [cls(f'rings:{planet}', obs, config=config)]

    def create_model(self) -> None:
        """Run the four-pass filter, render surviving features, populate metadata."""
        start_time = now_dt()
        self._metadata.clear()
        self._metadata['start_time'] = start_time.isoformat()
        self._metadata['end_time'] = None
        self._metadata['elapsed_time_sec'] = None
        with self.log_section('CREATE RINGS MODEL'):
            self._render()
            self._log_render_summary()
            end_time = now_dt()
            self._metadata['end_time'] = end_time.isoformat()
            self._metadata['elapsed_time_sec'] = (end_time - start_time).total_seconds()
            self._logger.info('Model created in %.3f s', self._metadata['elapsed_time_sec'])

    def _log_render_summary(self) -> None:
        """Emit INFO summary of the surviving ring catalogue + DEBUG detail."""
        meta = self._metadata
        planet = meta.get('planet')
        if not planet:
            self._logger.info('No planet identified — model is empty')
            return
        feature_count = meta.get('feature_count', 0)
        if feature_count == 0:
            self._logger.info('Planet = %s, surviving features = 0 — model is empty', planet)
            return
        self._logger.info(
            'Planet = %s, surviving features = %d, km/px radial = %.4f, subject range = %.0f km',
            planet,
            feature_count,
            self._km_per_pixel_radial,
            self._subject_range_km,
        )
        names = [entry['name'] for entry in meta.get('features', [])]
        self._logger.debug('Surviving feature names = %s', names)
        self._logger.debug(
            'Extfov shape (vu) = (%d, %d); predicted_center = %s',
            self._extfov_v_size,
            self._extfov_u_size,
            self._predicted_center_vu,
        )

    def _render(self) -> None:
        """Filter + render every surviving ring feature for this observation."""
        obs = self.obs
        planet = obs.closest_planet
        if planet is None:
            self._logger.warning('No closest planet found -- cannot create ring model')
            return
        self._planet = planet
        self._metadata['planet'] = planet
        self._extfov_v_size = obs.extdata_shape_vu[0]
        self._extfov_u_size = obs.extdata_shape_vu[1]
        self._predicted_center_vu = (
            float(self._extfov_v_size) / 2.0,
            float(self._extfov_u_size) / 2.0,
        )

        rings_config = self._config.rings
        ring_features_dict = getattr(rings_config, 'ring_features', {})
        if planet not in ring_features_dict:
            self._logger.info('No ring features configured for planet %s', planet)
            return
        planet_config = ring_features_dict[planet]
        if not isinstance(planet_config, dict):
            raise ValueError(
                f'Ring config error: rings.ring_features entry for planet {planet!r} must be '
                f'a dict (got {type(planet_config).__name__!r})'
            )
        epoch_str = planet_config.get('epoch')
        if epoch_str is None:
            raise ValueError(f'No epoch configured for planet {planet}')
        if not isinstance(epoch_str, str):
            raise ValueError(
                f'Ring config error: epoch for planet {planet!r} must be a string, '
                f'got {type(epoch_str).__name__!r}'
            )
        try:
            epoch = utc_to_et(epoch_str)
        except ValueError as exc:
            raise ValueError(
                f'Ring config error: epoch for planet {planet!r} is not a valid UTC '
                f'string ({epoch_str!r}): {exc}'
            ) from exc
        fade_width_pix = _require_positive_finite_planet_scalar(
            planet, 'fade_width_pix', planet_config
        )
        min_allowed_fade_width_pix = _require_positive_finite_planet_scalar(
            planet, 'min_allowed_fade_width_pix', planet_config
        )
        min_feature_pixels = _require_positive_finite_planet_scalar(
            planet, 'min_feature_pixels', planet_config
        )
        if 'features' not in planet_config:
            raise ValueError(
                f'Missing required ring configuration key "features" for planet {planet}'
            )
        features_raw = planet_config['features']
        if not isinstance(features_raw, dict):
            raise ValueError(
                f'Ring config error: "features" for planet {planet!r} must be a dict '
                f'(got {type(features_raw).__name__!r})'
            )
        if not features_raw:
            self._logger.info('Empty ring features dict for planet %s', planet)
            return
        features: list[RingFeature] = []
        for key, data in features_raw.items():
            if not isinstance(data, dict):
                raise ValueError(
                    f'Ring config error: planet {planet!r} features.{key!r} must be a dict '
                    f'(got {type(data).__name__!r})'
                )
            features.append(RingFeature.from_config(key, data))
        validate_no_date_overlaps(features)
        max_feature_extent = max(f.max_extent_radius for f in features)
        ring_target = f'{planet.lower()}:ring'
        # Cheap 16x16 visibility pre-check: rules out the two common skip
        # cases (no ring-plane intersection in the FOV at all; visible
        # radial range entirely beyond the catalog's outermost feature)
        # without paying for the dense ext-FOV backplane.  Saves ~5-7 s
        # per image when it fires.
        if self._sparse_visibility_skip(ring_target, max_feature_extent):
            return
        bp_radii = obs.ext_bp.ring_radius(ring_target)
        if bp_radii.is_all_masked():
            self._logger.info('No rings visible in observation')
            return
        # Ask the backplane to name the key rather than reading bp_radii.key:
        # oops attaches that attribute from outside polymath, so polymath drops
        # it from any array it clones, and no array that was computed from this
        # one carries it at all.  Resolved once because naming a key can cost a
        # scan of every registered backplane.
        bp_radii_key = obs.ext_bp.standardize_backplane_key(bp_radii)
        min_radius = float(bp_radii.min().vals)
        max_radius = float(bp_radii.max().vals)
        self._logger.info(
            'Ring plane radial range visible in image: [%.0f, %.0f] km',
            min_radius,
            max_radius,
        )
        if min_radius > max_feature_extent:
            self._logger.info(
                'Visible ring radial range [%.0f, %.0f] km lies outside catalog '
                'max extent of %.0f km for planet %s',
                min_radius,
                max_radius,
                max_feature_extent,
                planet,
            )
            return
        resolutions: NDArrayFloatType = obs.ext_bp.ring_radial_resolution(ring_target).vals
        self._km_per_pixel_radial = float(np.mean(resolutions[np.isfinite(resolutions)]))
        # Retained per pixel so a physical km displacement can be converted at
        # the edge that carries it: the radial scale varies materially across a
        # wide field (a frame spanning the C ring to the A ring), and the
        # whole-image mean above is only appropriate for the per-vertex
        # statistical sigma.
        self._radial_resolution_ext = resolutions
        # Retained so the emitted edge normals can be signed radially outward
        # (see _polyline_from_edge_mask): the mask-neighbour test alone cannot
        # tell the high-radius side from the low-radius side.
        self._ring_radius_ext = np.asarray(bp_radii.mvals.filled(np.nan), dtype=np.float64)

        def min_res_at_radius(a: float) -> float | None:
            """Finest radial resolution along the ring-plane circle of radius ``a``.

            Parameters:
                a: Ring-plane radius in km.

            Returns:
                The minimum km-per-pixel radial resolution over the pixels
                that circle crosses, or ``None`` when it crosses none of the
                extended FOV.
            """
            border_arr: NDArrayBoolType = (
                obs.ext_bp.border_atop(bp_radii_key, a).mvals.astype('bool').filled(False)
            )
            res_at_edge = resolutions[border_arr]
            res_ma = np.ma.asarray(res_at_edge)
            if res_ma.count() == 0:
                return None
            vals = np.asarray(res_ma.compressed(), dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                return None
            min_val = float(np.min(vals))
            return min_val if min_val > 0.0 else None

        feature_filter = RingFeatureFilter(
            obs_time_et=obs.midtime,
            min_radius=min_radius,
            max_radius=max_radius,
            min_res_at_radius=min_res_at_radius,
            fade_width_pix=fade_width_pix,
            min_allowed_fade_width_pix=min_allowed_fade_width_pix,
            min_feature_pixels=min_feature_pixels,
            logger=self._logger,
        )
        surviving = feature_filter.filter(features)
        if not surviving:
            self._logger.info(
                'All %d ring features filtered out for planet %s', len(features), planet
            )
            return
        all_edge_radii: list[tuple[float, str]] = []
        for feat in surviving:
            all_edge_radii.extend(feat.all_base_radii())
        all_edge_radii.sort(key=lambda x: x[0])
        bp_distance = obs.ext_bp.distance(ring_target, direction='dep')
        distance_arr = bp_distance.mvals.filled(math.inf)
        # Subject range — closest visible ring radius
        if np.any(np.isfinite(distance_arr)):
            self._subject_range_km = float(distance_arr[np.isfinite(distance_arr)].min())
        else:
            self._subject_range_km = float('inf')

        shadow_mask: NDArrayBoolType | None = None
        if rings_config.get('remove_planet_shadow', False):
            try:
                raw_shadow = obs.ext_bp.where_inside_shadow(ring_target, planet.lower())
                shadow_mask = raw_shadow.mvals.filled(False).astype(bool)
            except Exception:
                self._logger.exception('Failed to compute planet shadow for %s', planet)
                raise

        # Ring points hidden behind the planet globe enter neither the emitted
        # ring-edge features nor the summary overlay.  ``where_in_front(planet,
        # ring)`` is True at every ext-FOV pixel where the planet body is
        # intercepted and nearer to the observer than the ring plane along the
        # same line of sight -- exactly the pixels where the disc hides the
        # rings.  The mask is ANDed into every edge mask below, before the
        # per-edge polylines are sampled, so an edge segment behind the disc is
        # never sampled as a polyline vertex and never painted across the
        # planet.  A backplane failure fails the image: an edge painted across
        # the planet is a model that says the rings are somewhere they are not,
        # which navigates to a wrong offset rather than to none.
        self._ring_occluded_ext = None
        try:
            occluded = obs.ext_bp.where_in_front(planet.lower(), ring_target)
            self._ring_occluded_ext = occluded.mvals.filled(False).astype(bool)
        except Exception:
            self._logger.exception('Failed to compute planet occlusion of rings for %s', planet)
            raise

        render_context = RingsRenderContext(
            obs=obs,
            ring_target=ring_target,
            epoch=epoch,
            resolutions=resolutions,
            fade_width_pix=fade_width_pix,
            all_edge_radii=tuple(all_edge_radii),
            logger=self._logger,
        )
        rendered: list[
            tuple[
                RingFeature,
                NDArrayFloatType,
                NDArrayBoolType,
                float,
                list[tuple[NDArrayBoolType, str, str]],
            ]
        ] = []
        for feature in surviving:
            for render_result in feature.render(render_context):
                feat_model = render_result.model_img
                feat_mask = render_result.model_mask
                if shadow_mask is not None:
                    feat_model = np.where(shadow_mask, 0.0, feat_model)
                    feat_mask = feat_mask & ~shadow_mask
                rendered.append(
                    (
                        feature,
                        feat_model,
                        feat_mask,
                        float(render_result.uncertainty),
                        self._visible_edge_info(render_result.edge_info_list),
                    )
                )
        self._render_results = rendered
        self._metadata['planet'] = planet
        self._metadata['epoch'] = epoch_str
        self._metadata['feature_count'] = len(surviving)
        self._metadata['features'] = [
            {'name': f.name, 'type': f.feature_type.value} for f in surviving
        ]

    def _sparse_visibility_skip(self, ring_target: str, max_feature_extent: float) -> bool:
        """Decide whether a sparse 16x16 ring-radius backplane rules out the dense path.

        Builds a :class:`~oops.Meshgrid` covering the ext-FOV at
        :data:`SPARSE_VISIBILITY_GRID_SIZE` samples per axis and runs a
        single ``ring_radius`` evaluation against it.  Returns ``True``
        when either:

        * No sparse sample's ray intersects the ring plane — the FOV
          looks entirely off the ring plane and the dense path would
          report ``is_all_masked()`` after a much costlier
          backplane evaluation.
        * Every sparse sample's ring radius exceeds
          ``max_feature_extent`` — the visible ring plane is at radii
          beyond every catalog feature, so the dense ``min_radius >
          max_feature_extent`` short-circuit would also fire.

        Returns ``False`` when the dense path must run (some sparse
        sample sees a ring radius inside the catalog, or the sparse
        grid was uninformative).

        Parameters:
            ring_target: ``'<planet>:ring'`` target string.
            max_feature_extent: Largest ``RingFeature.max_extent_radius``
                across the surviving catalog entries.

        Returns:
            ``True`` to skip the dense backplane and emit no ring
            features for this image; ``False`` to proceed.
        """
        obs = self.obs
        u_extent = obs.extfov_u_max - obs.extfov_u_min + 1
        v_extent = obs.extfov_v_max - obs.extfov_v_min + 1
        u_step = max(1, u_extent // SPARSE_VISIBILITY_GRID_SIZE)
        v_step = max(1, v_extent // SPARSE_VISIBILITY_GRID_SIZE)
        sparse_meshgrid = Meshgrid.for_fov(
            obs.fov,
            origin=(obs.extfov_u_min + 0.5, obs.extfov_v_min + 0.5),
            limit=(obs.extfov_u_max + 0.5, obs.extfov_v_max + 0.5),
            undersample=(u_step, v_step),
            swap=True,
        )
        sparse_bp = Backplane(obs, meshgrid=sparse_meshgrid)
        sparse_radii = sparse_bp.ring_radius(ring_target)
        if sparse_radii.is_all_masked():
            self._logger.info(
                'Sparse %dx%d visibility pre-check: no ring-plane intersection in FOV',
                SPARSE_VISIBILITY_GRID_SIZE,
                SPARSE_VISIBILITY_GRID_SIZE,
            )
            return True
        sparse_min = float(sparse_radii.min().vals)
        if sparse_min > max_feature_extent:
            sparse_max = float(sparse_radii.max().vals)
            self._logger.info(
                'Sparse %dx%d visibility pre-check: visible ring radial range '
                '[%.0f, %.0f] km lies outside catalog max extent of %.0f km',
                SPARSE_VISIBILITY_GRID_SIZE,
                SPARSE_VISIBILITY_GRID_SIZE,
                sparse_min,
                sparse_max,
                max_feature_extent,
            )
            return True
        return False

    def to_features(self, context: NavContext) -> list[NavFeature]:
        """Emit RING_EDGE / RING_ANNULUS features per surviving render result."""
        del context
        with self.log_section('EMIT RINGS FEATURES'):
            features: list[NavFeature] = []
            edge_count = 0
            straight_count = 0
            max_radial_px, kmpp_threshold = _ring_annulus_emission_params(
                self._config, self._planet or ''
            )
            # Default radial orbit uncertainty for a catalog feature that
            # carries no fitted rms; an edge's own rms takes precedence.  See
            # config_050_rings.yaml for the value's provenance.
            #
            # CONTESTED ASSUMPTION, deliberately conservative: the catalog rms
            # is consumed as a fully-correlated whole-edge displacement, but it
            # is the residual of the edge's own orbit fit and therefore mixes a
            # genuinely coherent radius error with longitude-varying resonant
            # wander that does NOT displace the edge coherently.  The catalog
            # shows the mixture directly -- the same A-ring outer edge is
            # fitted to rms 10.18 km with no m-modes and 1.78 km with nine
            # (config_310_saturn_rings.yaml), a 5.7x reduction attributable to
            # mode structure alone -- so on the resonant edges that carry the
            # largest rms this term is over-severe by roughly that factor.  The
            # principled fix decomposes the catalog modes and prices only the
            # m=0 part coherently, which needs per-mode amplitudes this channel
            # does not consume today; until then the geometry partially
            # compensates (wide-arc coverage shrinks the absorbed sensitivity
            # in RingEdgeNav) and the over-severity is accepted and stated
            # rather than modeled away.
            default_orbit_sigma_km = float(self._config.rings['default_orbit_radial_sigma_km'])
            # Operator-tunable severity for the contested assumption below:
            # the fraction of the catalog rms treated as coherent.  1.0 is
            # today's fully-correlated behavior; lowering it is the reversible
            # lever pending the mode-aware decomposition.
            correlated_fraction = float(
                self._config.rings['orbit_radial_sigma_correlated_fraction']
            )
            if not 0.0 <= correlated_fraction <= 1.0:
                raise ValueError(
                    'rings.orbit_radial_sigma_correlated_fraction must be in [0, 1]; '
                    f'got {correlated_fraction!r}'
                )
            # System-level annulus gate: at or above the per-planet
            # km/px threshold the whole ring system routes to the annulus
            # composite and every surviving feature is emitted as part
            # of it.  The threshold values and their rationale (for
            # Saturn, a measured head-to-head in which the annulus fit
            # was wrong on zero accepted answers in every regime it
            # was measured) live with the
            # ``feature_emission.ring_annulus`` block in
            # config_510_techniques.yaml.
            force_annulus = self._km_per_pixel_radial >= kmpp_threshold
            if force_annulus:
                self._logger.debug(
                    'System-level annulus gate fires for planet %s: '
                    'km_per_pixel_radial = %.2f >= %.2f',
                    self._planet,
                    self._km_per_pixel_radial,
                    kmpp_threshold,
                )
            # Annulus-eligible per-ring renderings accumulate here and
            # collapse into a single composite RING_ANNULUS at the end
            # of the loop.  The design specifies one RING_ANNULUS per
            # planet per scene; emitting one feature per surviving ring
            # would set ``constituent_count=1`` on each and the
            # reliability formula would gate them all out
            # (min(1, 1/5) * 0.7 = 0.14 < 0.30 threshold).
            annulus_renderings: list[tuple[NDArrayFloatType, NDArrayBoolType, str, float]] = []
            # Per-annulus-edge orbit terms: (normals, vertex_count, orbit_sigma_px).
            # The composite RING_ANNULUS carries the concatenated normals and the
            # vertex-weighted effective orbit sigma so RingAnnulusNav can widen
            # its covariance by the same coherent-radial-error channel RingEdgeNav
            # applies per edge.
            annulus_orbit_terms: list[tuple[NDArrayFloatType, int, float]] = []
            for (
                ring_feat,
                model_img,
                model_mask,
                uncertainty_km,
                edge_info_list,
            ) in self._render_results:
                if not edge_info_list:
                    continue
                # The annulus correlation template is built from these full-feature
                # renders, so it must be cleared of ring brightness hidden behind
                # the planet globe just as the per-edge masks already are (see
                # _visible_edge_info).  Trim the render against the same occlusion
                # mask before it can enter annulus_renderings; a backplane failure
                # left the mask None and degrades to no trimming.
                if self._ring_occluded_ext is not None:
                    model_img = np.where(self._ring_occluded_ext, 0.0, model_img)
                    model_mask = model_mask & ~self._ring_occluded_ext
                for edge_mask, label_text, edge_label in edge_info_list:
                    edge_orbit_rms_km = ring_feat.edge_uncertainty(edge_label)
                    vertices_vu, normals_vu = _polyline_from_edge_mask(
                        edge_mask, self._ring_radius_ext
                    )
                    if vertices_vu.shape[0] == 0:
                        continue
                    radial_extent_px = _radial_extent_px(vertices_vu, normals_vu)
                    straight = _is_straight_line(vertices_vu)
                    # The coherent displacement bound is a property of THIS edge's
                    # own orbit solution, not the feature-level max the per-vertex
                    # sigma uses; a physical km displacement converts at the edge's
                    # own unfloored radial scale (the per-vertex floor belongs to
                    # the statistical sigma and would understate a coherent
                    # displacement on sub-km/px frames).  Computed once here so the
                    # edge and annulus paths price the same edge identically.
                    orbit_sigma_km = correlated_fraction * (
                        edge_orbit_rms_km if edge_orbit_rms_km > 0.0 else default_orbit_sigma_km
                    )
                    orbit_km_per_pixel_radial = _median_radial_scale(
                        self._radial_resolution_ext,
                        edge_mask,
                        self._km_per_pixel_radial,
                    )
                    orbit_sigma_px = orbit_sigma_km / orbit_km_per_pixel_radial
                    use_annulus = force_annulus or (
                        radial_extent_px <= max_radial_px and not straight
                    )
                    if use_annulus:
                        annulus_renderings.append(
                            (model_img, model_mask, label_text, radial_extent_px)
                        )
                        annulus_orbit_terms.append(
                            (normals_vu, int(vertices_vu.shape[0]), orbit_sigma_px)
                        )
                        self._logger.debug(
                            'ANNULUS %r -> vertices=%d, radial_extent=%.2f px, '
                            'straight=%s, uncertainty=%.3f km',
                            label_text,
                            int(vertices_vu.shape[0]),
                            radial_extent_px,
                            straight,
                            uncertainty_km,
                        )
                    else:
                        edge_count += 1
                        if straight:
                            straight_count += 1
                        features.append(
                            _build_edge_feature(
                                ring_feat=ring_feat,
                                edge_label=edge_label,
                                label_text=label_text,
                                planet=self._planet or '',
                                vertices_vu=vertices_vu,
                                normals_vu=normals_vu,
                                uncertainty_km=uncertainty_km,
                                orbit_sigma_km=orbit_sigma_km,
                                km_per_pixel_radial=max(self._km_per_pixel_radial, 1.0),
                                orbit_km_per_pixel_radial=orbit_km_per_pixel_radial,
                                is_straight_line=straight,
                                bbox=_mask_bbox(edge_mask),
                                subject_range_km=self._subject_range_km,
                                source_model=self.name,
                            )
                        )
                        self._logger.debug(
                            'EDGE %r -> vertices=%d, radial_extent=%.2f px, '
                            'straight=%s, uncertainty=%.3f km',
                            label_text,
                            int(vertices_vu.shape[0]),
                            radial_extent_px,
                            straight,
                            uncertainty_km,
                        )
            annulus_count = 1 if annulus_renderings else 0
            if annulus_renderings:
                composite_img, composite_mask = _composite_ring_renderings(
                    annulus_renderings,
                    extfov_shape=(self._extfov_v_size, self._extfov_u_size),
                )
                composite_bbox = _mask_bbox(composite_mask)
                composite_radial_extent_px = float(composite_bbox[2] - composite_bbox[0])
                joint_label = ', '.join(sorted({label for _, _, label, _ in annulus_renderings}))
                orbit_normals_vu, sigma_orbit_radial_px = _aggregate_annulus_orbit_terms(
                    annulus_orbit_terms
                )
                # The template payload convention (``compose_template_features``)
                # is a postage stamp local to ``bbox_extfov_vu``; crop the
                # ext-FOV-sized composite down to the bbox.  Passing the full
                # ext-FOV image displaces the painted annulus by the bbox
                # origin when the consumer re-composes it, and the annulus
                # NCC then recovers a garbage offset.
                features.append(
                    _build_annulus_feature(
                        ring_name=joint_label,
                        planet=self._planet or '',
                        model_img=composite_img[
                            composite_bbox[0] : composite_bbox[2],
                            composite_bbox[1] : composite_bbox[3],
                        ],
                        model_mask=composite_mask[
                            composite_bbox[0] : composite_bbox[2],
                            composite_bbox[1] : composite_bbox[3],
                        ],
                        bbox=composite_bbox,
                        predicted_center_vu=self._predicted_center_vu,
                        orbit_normals_vu=orbit_normals_vu,
                        sigma_orbit_radial_px=sigma_orbit_radial_px,
                        subject_range_km=self._subject_range_km,
                        constituent_count=len(annulus_renderings),
                        source_model=self.name,
                    )
                )
                self._logger.debug(
                    'ANNULUS composite for planet %s: %d constituent ring(s), '
                    'composite bbox %r, composite radial_extent=%.2f px',
                    self._planet,
                    len(annulus_renderings),
                    composite_bbox,
                    composite_radial_extent_px,
                )
            self._logger.info(
                'Emitted %d feature(s) — %d edge (%d straight), %d annulus',
                len(features),
                edge_count,
                straight_count,
                annulus_count,
            )
            return features

    def to_annotations(self, context: NavContext) -> Annotations:
        """Render per-edge polyline overlays + ring labels.

        The edge masks stored on ``_render_results`` are already free of the
        pixels the planet globe occludes -- ``_render`` trims each mask against
        the occlusion backplane before storing it, so a Saturn-facing frame that
        captures the far side of the rings behind the disc neither emits nor
        draws ring edges across the planet.  See ``_render`` for the mask.
        """
        del context
        out = Annotations()
        for _ring_feat, _model_img, model_mask, _u, edge_info_list in self._render_results:
            if not edge_info_list:
                continue
            out.add_annotations(self._create_edge_annotations(self.obs, edge_info_list, model_mask))
        return out

    def _visible_edge_info(
        self, edge_info_list: list[tuple[NDArrayBoolType, str, str]]
    ) -> list[tuple[NDArrayBoolType, str, str]]:
        """Drop planet-occluded pixels from each ring-edge mask.

        Applied once in ``_render`` before the trimmed masks are stored on
        ``_render_results``, so every downstream consumer -- the emitted
        ring-edge features (``to_features`` samples polylines from these masks)
        and the summary overlay (``to_annotations``) -- sees only the visible
        ring segments.

        Parameters:
            edge_info_list: ``(edge_mask, label_text, edge_label)`` tuples
                whose masks are in ext-FOV coordinates.

        Returns:
            The same tuples with every edge mask ANDed against the visible
            (non-occluded) region; edges left with no visible pixel are
            dropped entirely.  When no occlusion mask is available (a backplane
            failure degraded it to ``None``) the input is returned unchanged.
        """
        occluded = self._ring_occluded_ext
        if occluded is None:
            return edge_info_list
        visible: list[tuple[NDArrayBoolType, str, str]] = []
        for edge_mask, label_text, edge_label in edge_info_list:
            trimmed = edge_mask & ~occluded
            if np.any(trimmed):
                visible.append((trimmed, label_text, edge_label))
        return visible
