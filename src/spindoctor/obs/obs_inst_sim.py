"""Observation snapshot backed by a simulated-image scene.

:class:`ObsSim` wraps a rendered sim scene as an oops ``Snapshot`` with a flat
FOV and dummy geometry, so the standard navigation pipeline runs on it exactly
as on a real frame.  The rendered image and the full scene (truth included)
stay on the snapshot for renderer-side consumers; the navigator-side models
see only the filtered idealized view exposed as ``obs.nav_params`` (the
information boundary -- see :func:`spindoctor.sim.scene.build_nav_params`).
"""

import math
from pathlib import Path
from typing import Any, cast

import oops
from filecache import FCPath
from oops.observation.snapshot import Snapshot

from spindoctor.config import DEFAULT_CONFIG, IMAGE_LOGGER, Config, logged_section
from spindoctor.obs.obs_snapshot_inst import ObsSnapshotInst
from spindoctor.sim.instruments import resolve_extfov_margin, resolve_sim_inst_config
from spindoctor.sim.render import render_combined_model
from spindoctor.sim.scene import build_nav_params, load_sim_scene
from spindoctor.support.types import PathLike


class ObsSim(ObsSnapshotInst):
    """Observation backed by a description of simulated bodies and stars."""

    def __init__(self, snapshot: Snapshot, **kwargs: Any) -> None:
        super().__init__(snapshot, **kwargs)

    @staticmethod
    @logged_section('obs', 'LOAD IMAGE')
    def from_file(
        path: PathLike,
        *,
        config: Config | None = None,
        extfov_margin_vu: tuple[int, int] | None = None,
        **kwargs: Any,
    ) -> 'ObsSim':
        """Creates an ObsSim from a YAML scene file.

        Parameters:
            path: Path or URL of the YAML scene file.  When ``sim_params`` is
                supplied no file is read and this is only the name the
                observation carries, so it may name nothing that exists.
            config: Navigation configuration. If None, uses defaults.
            extfov_margin_vu: Optional extended FOV margins (v,u) to add around the image.
            **kwargs: Additional keyword arguments.
                sim_params: Flat sim-params mapping. If present, this overrides the
                scene file.

        Returns:
            The :class:`ObsSim` wrapping the rendered scene: the rendered
            image as ``data``, the full scene and renderer truth metadata on
            the snapshot, and the filtered idealized view as ``nav_params``.
            Its ``abspath`` is the local file the scene was read from, or,
            for a scene supplied in memory, ``path`` rendered absolute
            without any storage being touched.
        """

        config = config or DEFAULT_CONFIG
        logger = IMAGE_LOGGER

        provided_sim_params = kwargs.get('sim_params')
        scene_path = FCPath(path)
        abspath: Path | FCPath
        if provided_sim_params is None:
            logger.debug(f'Reading simulated image scene {scene_path}')
            # Parsing the scene needs it as a file on local storage, which is
            # what get_local_path provides: it fetches a remote scene into the
            # cache and creates the local directories that hold it.
            abspath = cast(Path, scene_path.get_local_path()).absolute()
            sim_params = load_sim_scene(abspath)
        else:
            # A scene supplied in memory is never read back, so its path is only
            # a label carried into the logs and the metadata.  absolute()
            # renders it without reaching storage, where get_local_path would
            # create the directories of a file that is never opened.
            abspath = scene_path.absolute()
            sim_params = provided_sim_params
            logger.debug('Using provided sim_params')

        # Required fields
        try:
            size_v = int(sim_params['size_v'])
            size_u = int(sim_params['size_u'])
            offset_v = float(sim_params.get('offset_v', 0.0))
            offset_u = float(sim_params.get('offset_u', 0.0))
        except (KeyError, TypeError, ValueError) as e:
            if provided_sim_params is None:
                raise ValueError(
                    'Invalid or missing size/offset field in simulated image '
                    f'scene file "{scene_path}": {e}'
                ) from e
            else:
                raise ValueError(
                    'Invalid or missing size/offset field in provided '
                    'sim_params for simulated image scene file '
                    f'"{scene_path}": {e}'
                ) from e

        # Build a basic Snapshot with a flat FOV and dummy geometry
        fov = oops.fov.FlatFOV((1.0, 1.0), (size_u, size_v))
        snapshot = oops.observation.snapshot.Snapshot(
            axes=('v', 'u'),
            tstart=0.0,
            texp=1.0,
            fov=fov,
            path='SSB',
            frame='J2000',
        )
        # Store data and the full sim-params dictionary for future use
        snapshot.image_url = str(scene_path.absolute())
        snapshot.abspath = abspath

        # The full scene (truth included) stays on the snapshot for the
        # renderer-side consumers: the test harnesses that score recovery
        # against planted truth and the backplane writer that consumes the
        # rendered masks.  The navigator-side models consume ONLY the
        # filtered idealized view built below.
        snapshot.sim_params = sim_params
        snapshot.sim_offset_v = offset_v
        snapshot.sim_offset_u = offset_u
        snapshot.sim_time = float(sim_params.get('time', 0.0))

        # The information boundary: nav_params is sim_params with every
        # truth key stripped and per-body nav_override overlaid (the
        # navigator sees the geometry it believes, never the true values
        # underneath).  See spindoctor.sim.scene.build_nav_params.
        snapshot.nav_params = build_nav_params(sim_params)

        # Render combined model
        logger.debug('Rendering combined simulated model')
        img_rendered, meta = render_combined_model(sim_params)
        snapshot.insert_subfield('data', img_rendered)
        # Renderer output metadata (truth side; never read by NavModels).
        snapshot.sim_body_models = meta.get('bodies', {})
        snapshot.sim_inventory = meta.get('inventory', {})
        snapshot.sim_body_order_near_to_far = meta.get('order_near_to_far', [])
        snapshot.sim_body_index_map = meta.get('body_index_map')
        snapshot.sim_body_mask_map = meta.get('body_mask_map', {})

        # Select the per-instrument config block the sim is emulating (or the
        # generic sim block when no instrument is specified), so the
        # orchestrator sees the right units / noise / saturation settings.
        sim_block = config.category('sim')
        inst_config = resolve_sim_inst_config(
            config, sim_params.get('instrument'), sim_params.get('instrument_config')
        )
        if extfov_margin_vu is None:
            extfov_margin_vu = resolve_extfov_margin(inst_config, sim_block, size_v)

        # A scene may toggle camera-rotation fitting independently of the camera
        # it emulates.  In reality ``fit_camera_rotation`` is a per-camera
        # property, but a sim scene is a navigation fixture: it can ask the
        # navigator to solve for a planted roll on top of any instrument's optics
        # (e.g. Cassini NAC's sharp PSF) without pretending to be a camera that
        # happens to fit rotation.  The resolved block is shared live config, so
        # copy before overriding.
        fit_rotation_override = sim_params.get('fit_camera_rotation')
        if fit_rotation_override is not None:
            inst_config = {**inst_config, 'fit_camera_rotation': bool(fit_rotation_override)}

        snapshot._closest_planet = sim_params.get('closest_planet')
        new_obs = ObsSim(snapshot, config=config, extfov_margin_vu=extfov_margin_vu, simulated=True)
        new_obs._inst_config = cast(dict[str, Any], inst_config)

        new_obs.spice_kernels = ['fake_kernel1.txt', 'fake_kernel2.txt']

        return new_obs

    def star_min_usable_vmag(self) -> float:
        return 0.0

    def star_max_usable_vmag(self) -> float:
        """Returns the maximum usable magnitude for stars in this observation.

        Derived from the emulated instrument's PUBLISHED detector model, never
        from the scene's truth-side ``noise`` block: the navigator may know only
        what a real pipeline could know about the camera.  The renderer
        flux-normalizes a star (its total signal is
        ``star_flux_dn_per_s_vmag0 * 10**(-0.4 * vmag) * exposure``, see
        ``spindoctor.sim.forward``); a Gaussian core of the published
        ``star_psf_sigma`` then puts a fraction ``1 / (2*pi*sigma**2)`` of that
        total in the peak pixel.  The limiting magnitude is where that peak falls
        to twice the effective per-pixel noise sigma -- the matched-filter
        detection boundary.  The published DN zero point is the camera's
        electron zero point over its standard gain state, so a real navigator
        could know it; the scene's truth-side noise block is deliberately not
        consulted, so a scene that plants noise different from the published
        values produces an honestly-wrong detection limit, which is desired model
        error, not a defect.  Keeping this physical matters beyond the faint-star
        gate: the star NavModel synthesises each STAR feature's predicted SNR
        (and from it the CRLB position covariance and reliability score) from how
        far the star sits above this limit, so an arbitrarily permissive
        placeholder inflates every simulated star's SNR by tens of orders of
        magnitude and collapses its covariance to zero.

        The exposure the flux formula scales by is the scene's idealized
        ``exposure_sec``, read from the navigator-visible ``nav_params`` view:
        exposure is commanded, published information a real pipeline always
        has, and the renderer multiplies every star's deposited flux by it, so
        the detection limit must move by ``2.5 * log10(exposure)`` alongside
        the flux or a long exposure's faint stars are gated out (and a short
        exposure's noise floor is overstated).  The dummy Snapshot's ``texp``
        deliberately stays at the 1-second reference: ``texp`` feeds the
        observation's timing (``midtime = tstart + texp / 2`` anchors every
        time-dependent oops computation and the reported exposure metadata),
        and repurposing it would move the sim epoch as a side effect of a
        photometric knob, so the limit reads the exposure directly instead.

        Returns:
            The maximum usable magnitude for stars in this observation.  For
            calibrated-unit sim instruments the published block carries no DN
            zero point to anchor a matched-filter limit, so the navigator uses a
            generous constant.
        """
        inst_config = self._inst_config or {}
        inst_noise = inst_config.get('noise') or {}
        if inst_config.get('data_units', 'raw_dn') != 'raw_dn':
            # calibrated_if: no published DN zero point to anchor the limit, so
            # the navigator-side detection limit is a generous constant.
            return 30.0
        # Published values only: the resolved per-instrument block (which already
        # carries any idealized instrument_config overrides), with the generic
        # sim block supplying any key the instrument block leaves unset.  The
        # scene's truth-side noise block is deliberately not consulted.
        sim_noise = self.config.category('sim')['noise']
        zero_point_dn = float(
            inst_noise.get('star_flux_dn_per_s_vmag0', sim_noise['star_flux_dn_per_s_vmag0'])
        )
        star_psf_sigma = float(
            inst_config.get('star_psf_sigma', self.config.category('sim')['star_psf_sigma'])
        )
        read_noise_dn = float(inst_noise['read_noise_dn'])
        # The scene's idealized exposure (navigator-visible; see the docstring
        # for why the Snapshot's reference texp is not consulted here).
        exposure = float(self.nav_params.get('exposure_sec', 1.0))
        # Peak DN of a magnitude-0 star: its total over the Gaussian PSF core.
        peak0_dn = zero_point_dn * exposure / (2.0 * math.pi * star_psf_sigma**2)
        # Poisson shot noise on the star's own counts keeps the effective sigma
        # above ~1 DN even on a read-noise-free frame.
        sigma_eff = max(read_noise_dn, 1.0)
        return 2.5 * math.log10(peak0_dn / (2.0 * sigma_eff))

    @property
    def camera(self) -> str:
        """The camera that took this observation.

        Returns:
            Always ``'SIM'``; a simulated scene has one synthetic camera.
        """
        return 'SIM'

    def get_public_metadata(self) -> dict[str, Any]:
        """Return the facts the simulated host publishes about the scene's image.

        A simulated image has no spacecraft and no clock, so it publishes none of the
        times, the exposure time or the filters a spacecraft host does.

        Returns:
            The image's path and name, ``sim`` as both PDS4 context identifiers, the image
            shape as ``(x, y)``, the camera, and a description saying the image was
            simulated from a scene file.
        """
        return {
            'image_path': self.image_url,
            'image_name': self.abspath.name,
            'instrument_host_lid': 'sim',
            'instrument_lid': 'sim',
            'image_shape_xy': self.data_shape_uv,
            'camera': self.camera,
            'description': 'Simulated observation from YAML scene',
        }
