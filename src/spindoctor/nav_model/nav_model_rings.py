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
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
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
from spindoctor.nav_model.ring_visibility import ring_disc_view
from spindoctor.nav_model.rings import (
    RingFeature,
    RingFeatureFilter,
    RingsRenderContext,
)
from spindoctor.obs import ObsSnapshotInst
from spindoctor.support.constants import PIXEL_CENTER_TO_CORNER_PX
from spindoctor.support.memory import release_transient_memory
from spindoctor.support.time import now_dt, utc_to_et
from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    from spindoctor.nav_orchestrator.nav_context import NavContext


__all__ = [
    'BACKPLANE_STRIP_ROWS',
    'FLAT_CURVATURE_THRESHOLD_PX',
    'RING_EDGE_DEFAULT_RELIABILITY',
    'RING_EDGE_SIGMA_ALONG_PX',
    'NavModelRings',
]


BACKPLANE_STRIP_ROWS: int = 128
"""How many extended-frame rows one whole-frame backplane is evaluated over.

``oops`` evaluates a backplane by materializing intermediates over the whole
meshgrid at once, and those intermediates are what a navigation's memory is
mostly made of: measured on a 1.47 Mpx Cassini frame (2026-09-05), one
``where_inside_shadow`` call cost 1.67 GB, ``distance`` 0.91 GB and
``where_in_front`` 0.91 GB, against final arrays of 12 MB apiece -- roughly a
kilobyte of transient per pixel per call. The cost therefore scales with the
extended frame, which the per-instrument search margin sets: Voyager's 400-pixel
margin on a 1000-pixel image gives a 1800x1800 frame, 2.2 times Cassini's,
and Voyager frames measured up to 26 GB against Cassini's 9.9.

A backplane is a per-pixel function of the ray through that pixel apart from one
coupling, so evaluating one over a horizontal strip and concatenating gives the
whole-frame array to within that coupling. ``oops`` solves the photon path by
iteration and stops when the largest light-time change anywhere on the meshgrid
falls under its precision goal, so how far every pixel's solution is refined
depends on which other pixels shared the call. The goal bounds the light time to
3e-07 s, about 90 m of travel, and that is the bound on a strip's answer; the
differences measured are far inside it. On a Cassini frame ``where_in_front``
and ``where_inside_shadow`` come back bit-identical, ``distance`` agrees to
2.5e-13 of its value, and ``ring_radius`` -- which the haze model's strips ask
for, though this one does not -- to 5.6e-12 of its own. A boolean backplane is a
comparison, so it is bit-identical only while no pixel sits nearer the threshold
than that; a frame with one flips that pixel. Lowering the precision goal, as a
reprojection does for the whole process, widens the bound in proportion.
Striping bounds the transient by the strip rather than by the frame, at the cost
of one meshgrid and one Backplane per strip.

128 rows keeps the transient to about a quarter of a gigabyte on the widest
frame while leaving the per-strip fixed cost small against the work.

Not everything is striped. The ring radius, the ring radial resolution and the
``border_atop`` and ``radial_mode`` backplanes are still evaluated over the
whole extended frame through ``obs.ext_bp``: the ring model renders from the
whole-frame radius array, and the edge backplanes are derived inside ``oops``
from that cached array. The ring model's transient floor is therefore still set
by those calls, and striping bounds only the three quantities named here -- the
ring-plane distance, the planet shadow and the planet's occlusion of the rings.
"""


@dataclass(frozen=True)
class _StripQuantity:
    """One whole-frame backplane quantity, as ``_striped_backplanes`` evaluates it.

    Parameters:
        key: Name the assembled array comes back under, distinct within one
            call. Quantities are named rather than positional because which
            of them are asked for varies with the configuration, so a
            position means different things on different frames.
        evaluate: Called with a strip's :class:`~oops.backplane.Backplane` and
            returning the quantity over that strip.
        fill: Value substituted where the strip's result is masked.
        dtype: Element type of the assembled array.
        failure: What the quantity is called in the one-line error logged when
            its evaluation raises, e.g. ``'Planet shadow for SATURN'``.
    """

    key: str
    evaluate: Callable[[Backplane], Any]
    fill: Any
    dtype: Any
    failure: str


class NavModelRings(NavModelRingsBase):
    """Catalog-driven ring NavModel for one planet.

    Parameters:
        name: Model instance name (e.g. ``'rings:SATURN'``).
        obs: Observation snapshot.
        config: Optional ``Config`` override.
    """

    _abstract = False
    inputs = 'spice'

    def __init__(
        self,
        name: str,
        obs: ObsSnapshotInst,
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
        cls, obs: ObsSnapshotInst, *, config: Config | None = None
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
        planet = getattr(obs, 'closest_planet', None)
        if planet is None:
            return []
        # The model also needs the extfov shape and the ext_bp surface; obs
        # stand-ins that don't expose those are not applicable.
        if not hasattr(obs, 'extdata_shape_vu') or not hasattr(obs, 'ext_bp'):
            return []
        rings_config = (config if config is not None else DEFAULT_CONFIG).rings
        if planet not in rings_config.ring_features:
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
        """Emit INFO summary of the surviving ring catalog + DEBUG detail."""
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
        # The center of the extended frame, in the pixel centric coordinates
        # the payload is stated in: a whole number falls at a pixel's center
        # there, so a size-wide frame runs from 0 to size - 1 and its center
        # is (size - 1) / 2.
        self._predicted_center_vu = (
            (float(self._extfov_v_size) - 1.0) / 2.0,
            (float(self._extfov_u_size) - 1.0) / 2.0,
        )

        rings_config = self._config.rings
        catalog = rings_config.ring_features.get(planet)
        if catalog is None:
            self._logger.info('No ring features configured for planet %s', planet)
            return
        epoch_str = catalog.epoch
        epoch = utc_to_et(epoch_str)
        if not catalog.features:
            self._logger.info('Empty ring features dict for planet %s', planet)
            return
        features = [RingFeature.from_entry(key, entry) for key, entry in catalog.features.items()]
        max_feature_extent = max(f.max_extent_radius for f in features)
        ring_target = f'{planet.lower()}:ring'
        # Cheap visibility pre-check: skips the dense extended-frame
        # backplane, several seconds per frame, when the ring plane out to
        # the catalog's outermost feature provably lies outside the frame.
        # It reads the frame boundary rather than a sparse grid, because a
        # grid can step over a distant or edge-on ring system (#591).
        if self._ring_disc_outside_frame(ring_target, planet, max_feature_extent):
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
        # (see _polyline_from_edge_mask): the mask-neighbor test alone cannot
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
            fade_width_pix=catalog.fade_width_pix,
            min_allowed_fade_width_pix=catalog.min_allowed_fade_width_pix,
            min_feature_pixels=catalog.min_feature_pixels,
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
        # The whole-frame backplanes are evaluated together, a strip of rows at
        # a time, so each strip's ring-plane intercept is solved once for all
        # of them.  A failure in any one fails the image: the helper logs one
        # line naming the quantity, and the orchestrator logs the traceback.
        quantities = [
            _StripQuantity(
                key='ring_plane_distance',
                evaluate=lambda bp: bp.distance(ring_target, direction='dep'),
                fill=math.inf,
                dtype=np.float64,
                failure=f'Ring-plane distance for {planet}',
            )
        ]
        remove_planet_shadow = rings_config.remove_planet_shadow
        if remove_planet_shadow:
            quantities.append(
                _StripQuantity(
                    key='planet_shadow',
                    evaluate=lambda bp: bp.where_inside_shadow(ring_target, planet.lower()),
                    fill=False,
                    dtype=bool,
                    failure=f'Planet shadow for {planet}',
                )
            )
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
        quantities.append(
            _StripQuantity(
                key='ring_occlusion',
                evaluate=lambda bp: bp.where_in_front(planet.lower(), ring_target),
                fill=False,
                dtype=bool,
                failure=f'Planet occlusion of rings for {planet}',
            )
        )
        assembled = self._striped_backplanes(quantities)
        # Each array already has the dtype it was asked for; ``asarray`` with
        # that dtype returns it unchanged and only names the type here.
        distance_arr: NDArrayFloatType = np.asarray(
            assembled['ring_plane_distance'], dtype=np.float64
        )
        shadow_mask: NDArrayBoolType | None = None
        if remove_planet_shadow:
            shadow_mask = np.asarray(assembled['planet_shadow'], dtype=bool)
        self._ring_occluded_ext = np.asarray(assembled['ring_occlusion'], dtype=bool)
        # Subject range -- closest visible ring radius
        if np.any(np.isfinite(distance_arr)):
            self._subject_range_km = float(distance_arr[np.isfinite(distance_arr)].min())
        else:
            self._subject_range_km = float('inf')

        render_context = RingsRenderContext(
            obs=obs,
            ring_target=ring_target,
            epoch=epoch,
            resolutions=resolutions,
            fade_width_pix=catalog.fade_width_pix,
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

    def _striped_backplanes(
        self, quantities: Sequence[_StripQuantity]
    ) -> dict[str, np.ndarray[Any, np.dtype[Any]]]:
        """Evaluate whole-frame backplane quantities a strip of rows at a time.

        The array returned per quantity is what a single whole-frame evaluation
        of it returns, to within the photon solver's convergence tolerance.
        What differs is the peak memory: ``oops`` materializes its
        intermediates over the meshgrid it is given, so a frame-sized meshgrid
        costs frame-sized intermediates, and a strip-sized one costs
        strip-sized intermediates. See :data:`BACKPLANE_STRIP_ROWS`.

        One backplane per strip answers every quantity, so the surface
        intercept the backplane caches is solved once per strip rather than
        once per strip per quantity, and the strip's fixed cost -- the
        meshgrid, the backplane, the release afterwards -- is paid once.

        Parameters:
            quantities: What to evaluate on each strip's backplane, each under
                a ``key`` distinct from the others'.

        Returns:
            One extended-frame array per quantity, row-major and of that
            quantity's ``dtype``, under that quantity's ``key``.

        Raises:
            ValueError: If two quantities share a ``key``, which would drop
                one of them silently.
            Exception: Whatever a quantity's evaluation raised, after one line
                naming the quantity by its ``failure`` text has been logged;
                the orchestrator logs the traceback when it fails the image.
        """
        obs = self.obs
        rows, cols = obs.extdata_shape_vu
        assembled: dict[str, np.ndarray[Any, np.dtype[Any]]] = {
            quantity.key: np.empty((rows, cols), dtype=quantity.dtype) for quantity in quantities
        }
        if len(assembled) != len(quantities):
            raise ValueError(
                'two strip quantities share a key: '
                + ', '.join(quantity.key for quantity in quantities)
            )
        v_min = obs.extfov_v_min
        for start in range(0, rows, BACKPLANE_STRIP_ROWS):
            stop = min(start + BACKPLANE_STRIP_ROWS, rows)
            # The strip is named by array rows; the meshgrid reads the
            # geometry layer's pixel corner coordinates, so the half pixel
            # goes on here.
            meshgrid = Meshgrid.for_fov(
                obs.fov,
                origin=(
                    obs.extfov_u_min + PIXEL_CENTER_TO_CORNER_PX,
                    v_min + start + PIXEL_CENTER_TO_CORNER_PX,
                ),
                limit=(
                    obs.extfov_u_max + PIXEL_CENTER_TO_CORNER_PX,
                    v_min + stop - 1 + PIXEL_CENTER_TO_CORNER_PX,
                ),
                swap=True,
            )
            backplane = Backplane(obs, meshgrid=meshgrid)
            for quantity in quantities:
                try:
                    strip = quantity.evaluate(backplane)
                except Exception as exc:
                    self._logger.error('%s could not be evaluated: %s', quantity.failure, exc)
                    raise
                assembled[quantity.key][start:stop, :] = np.asarray(
                    strip.mvals.filled(quantity.fill), dtype=quantity.dtype
                )
                del strip
            # The meshgrid is the other half of the strip and has to go with
            # it: bound to a name, it would outlive the release.
            del backplane, meshgrid
            release_transient_memory()
        return assembled

    def _ring_disc_outside_frame(
        self, ring_target: str, planet: str, max_feature_extent: float
    ) -> bool:
        """Decide whether the ring plane out to the catalog's edge lies outside the frame.

        Asks :func:`~spindoctor.nav_model.ring_visibility.ring_disc_view` for the
        extended frame's boundary traced on the ring plane and the planet center's place
        in the frame, and logs what it established when it decides to skip.  The answer
        is ``True`` only when the planet center projects outside the frame and the
        boundary's trace stays farther than ``max_feature_extent`` from the planet, in
        which case no radius inside the catalog can appear anywhere in the extended
        frame.

        Parameters:
            ring_target: ``'<planet>:ring'`` target string.
            planet: SPICE name of the planet at the ring plane's center.
            max_feature_extent: Largest ``RingFeature.max_extent_radius`` across the
                catalog entries, in km.

        Returns:
            ``True`` to skip the dense backplane and emit no ring features for this
            image; ``False`` to proceed.
        """
        view = ring_disc_view(self.obs, ring_target, planet, max_feature_extent)
        if not view.disc_outside_frame:
            return False
        if not view.edge_meets_plane:
            self._logger.info(
                'Ring visibility pre-check: no pixel on the extended frame boundary looks '
                'onto the ring plane and the planet center is outside the frame, so the '
                'frame does not see the ring plane'
            )
            return True
        self._logger.info(
            'Ring visibility pre-check: the ring plane out to the catalog max extent of '
            '%.0f km lies wholly outside the extended frame (the frame boundary traced on '
            'the ring plane comes no closer than %.0f km to the planet, radii at its pixel '
            'centers span [%.0f, %.0f] km, and the planet center is outside the frame)',
            max_feature_extent,
            view.edge_closest_km,
            float(view.edge_radii_km.min()),
            float(view.edge_radii_km.max()),
        )
        return True

    def to_features(self, context: NavContext) -> list[NavFeature]:
        """Emit RING_EDGE / RING_ANNULUS features per surviving render result."""
        with self.log_section('EMIT RINGS FEATURES'):
            features: list[NavFeature] = []
            edge_count = 0
            straight_count = 0
            emission = self._config.feature_emission.ring_annulus.for_planet(self._planet or '')
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
            default_orbit_sigma_km = self._config.rings.default_orbit_radial_sigma_km
            # Operator-tunable severity for the contested assumption below:
            # the fraction of the catalog rms treated as coherent.  1.0 is
            # today's fully-correlated behavior; lowering it is the reversible
            # lever pending the mode-aware decomposition.
            correlated_fraction = self._config.rings.orbit_radial_sigma_correlated_fraction
            # System-level annulus gate: at or above the per-planet
            # km/px threshold the whole ring system routes to the annulus
            # composite and every surviving feature is emitted as part
            # of it.  The threshold values and their rationale (for
            # Saturn, a measured head-to-head in which the annulus fit
            # was wrong on zero accepted answers in every regime it
            # was measured) live with the
            # ``feature_emission.ring_annulus`` block in
            # config_510_techniques.yaml.
            force_annulus = self._km_per_pixel_radial >= emission.kmpp_threshold
            if force_annulus:
                self._logger.debug(
                    'System-level annulus gate fires for planet %s: '
                    'km_per_pixel_radial = %.2f >= %.2f',
                    self._planet,
                    self._km_per_pixel_radial,
                    emission.kmpp_threshold,
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
                # mask before it can enter annulus_renderings.  The mask was
                # assigned earlier in this method; the guard satisfies the
                # attribute's optional type.
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
                        radial_extent_px <= emission.max_radial_px and not straight
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
            dropped entirely.  The occlusion mask is assigned in ``_render``
            before this is called; the ``None`` branch satisfies the
            attribute's optional type and returns the input unchanged.
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
