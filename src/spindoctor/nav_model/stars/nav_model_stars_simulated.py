"""Simulated-scene star NavModel.

Emits STAR ``NavFeature`` instances for a simulated frame exactly the way
:class:`~spindoctor.nav_model.stars.nav_model_stars.NavModelStars` does for a real
frame -- same predicted-SNR, covariance, reliability, and annotation
machinery -- but builds its star list from the scene's catalog entries in
the filtered idealized view (``obs.nav_params``) rather than reducing real
catalogs.  The renderer's output star records never cross the information
boundary: the navigator knows the catalog, not what was drawn.

The scene renders each star at its catalog ``(v, u)`` shifted by the planted
offset; this model predicts the unshifted catalog position, so a star
technique that detects the shifted peak recovers the planted offset -- the
same prediction/observation split a real navigation has, which is why the
recovery transfers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from spindoctor.nav_model.nav_model import NavModel
from spindoctor.nav_model.stars.nav_model_stars import (
    NavModelStars,
    _star_short_info,
    _star_summary,
)
from spindoctor.obs import ObsSnapshotInst
from spindoctor.sim.star_records import star_record_from_params
from spindoctor.support.time import now_dt

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    from spindoctor.config import Config
    from spindoctor.support.types import MutableStar

__all__ = ['NavModelStarsSimulated']


class NavModelStarsSimulated(NavModelStars):
    """Star NavModel populated from the scene's idealized star catalog.

    Inherits feature emission (``to_features``), annotations, and the
    extfov-coordinate plumbing from :class:`NavModelStars`; only the model
    construction differs -- the stars come from the scene's ``nav_params``
    star entries rather than from a catalog reduction.

    Parameters:
        name: Model name (typically ``'stars'``).
        obs: Simulated observation snapshot carrying ``nav_params``.
        config: Optional ``Config`` override.
    """

    @classmethod
    def instances_for_obs(
        cls, obs: ObsSnapshotInst, *, config: Config | None = None
    ) -> list[NavModel]:
        """Return one star model for a simulated obs whose scene has stars.

        Returns an empty list for a real obs (the catalog-driven
        :class:`NavModelStars` handles those) and for a simulated obs whose
        scene lists no stars (so the orchestrator builds no empty star model).

        Parameters:
            obs: Observation snapshot.
            config: Configuration passed to the constructed instance.  None
                uses ``DEFAULT_CONFIG``.

        Returns:
            ``[NavModelStarsSimulated('stars', obs)]`` for a simulated obs
            with at least one scene star, else ``[]``.
        """
        if not getattr(obs, 'is_simulated', False):
            return []
        nav_params = getattr(obs, 'nav_params', None)
        if not isinstance(nav_params, dict) or not nav_params.get('stars'):
            return []
        return [cls('stars', obs, config=config)]

    def create_model(self) -> None:
        """Build the star list from the scene's idealized catalog entries.

        Each ``nav_params`` star entry becomes a catalog record at its
        unshifted position, through the same builder the renderer uses, so
        prediction and render share one set of defaults while exchanging no
        rendered values.  Smear is per-entry catalog data (zero by default)
        and there are no body/ring occlusion conflicts to mark -- a
        simulated star field is clean by construction.
        """
        start_time = now_dt()
        self._metadata.clear()
        self._metadata['start_time'] = start_time.isoformat()
        self._metadata['end_time'] = None
        self._metadata['elapsed_time_sec'] = None
        with self.log_section('CREATE SIMULATED STARS MODEL'):
            nav_params = getattr(self.obs, 'nav_params', None) or {}
            default_v = float(self.obs.data_shape_v) / 2.0
            default_u = float(self.obs.data_shape_u) / 2.0
            stars: list[MutableStar] = [
                star_record_from_params(
                    star_params, index=i, default_v=default_v, default_u=default_u
                )
                for i, star_params in enumerate(nav_params.get('stars') or [])
            ]
            self._stars = stars
            self._smear_vu = (0.0, 0.0)
            self._metadata['star_count'] = len(stars)
            self._metadata['stars'] = [_star_summary(star) for star in stars]
            self._logger.info('Using %d simulated star(s) from the scene catalog', len(stars))
            for star in stars:
                self._logger.info('  %s', _star_short_info(star))
            end_time = now_dt()
            self._metadata['end_time'] = end_time.isoformat()
            self._metadata['elapsed_time_sec'] = (end_time - start_time).total_seconds()
