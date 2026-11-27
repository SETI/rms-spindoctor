"""Observation snapshot backed by a simulated-image scene.

:class:`ObsSim` wraps a rendered sim scene as an oops ``Snapshot`` with a flat
FOV and dummy geometry, so the standard navigation pipeline runs on it exactly
as on a real frame.  The rendered image and the full scene (truth included)
stay on the snapshot for renderer-side consumers; the navigator-side models
see only the navigator view exposed as ``obs.nav_params`` (the
information boundary -- see :func:`spindoctor.sim.scene.boundary.build_nav_params`).
"""

import dataclasses
import math
from pathlib import Path
from typing import Any, cast

import oops
from filecache import FCPath
from oops.observation import Snapshot

from spindoctor.config import DEFAULT_CONFIG, IMAGE_LOGGER, Config, logged_section
from spindoctor.obs.obs_snapshot_inst import ObsSnapshotInst
from spindoctor.sim.instruments.navigator_link import resolve_extfov_margin, scene_camera_block
from spindoctor.sim.render.driver import render_scene
from spindoctor.sim.scene.boundary import build_nav_params
from spindoctor.sim.scene.io import load_sim_scene
from spindoctor.support.types import NDArrayAnyType, NDArrayBoolType, PathLike


class ObsSim(ObsSnapshotInst):
    """Observation backed by a description of simulated bodies and stars."""

    model_source = 'scene_view'
    """Its navigation models build from the scene's navigator view, ``nav_params``."""

    sim_params: dict[str, Any]
    """The full scene, truth included, for the renderer-side consumers."""

    sim_offset_v: float
    """The planted pointing offset along v, in pixels."""

    sim_offset_u: float
    """The planted pointing offset along u, in pixels."""

    sim_time: float
    """The scene's time, in seconds."""

    nav_params: dict[str, Any]
    """The scene with every truth key stripped: all the navigator may read."""

    sim_body_models: dict[str, Any]
    """The renderer's per-body models, keyed by body name."""

    sim_inventory: dict[str, Any]
    """The renderer's per-body inventory, keyed by body name."""

    sim_body_order_near_to_far: list[str]
    """The rendered bodies' names, nearest first."""

    sim_body_index_map: NDArrayAnyType | None
    """Which body each pixel shows, or None when the scene has no bodies."""

    sim_body_mask_map: dict[str, NDArrayBoolType]
    """Each rendered body's full-frame mask, keyed by body name."""

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
                scene file.  It is validated before it renders, exactly as a
                scene file is when it loads.

        Returns:
            The :class:`ObsSim` wrapping the rendered scene: the rendered
            image as ``data``, the full scene and renderer truth metadata on
            the snapshot, and the navigator view as ``nav_params``.
            Its ``abspath`` is the local file the scene was read from, or,
            for a scene supplied in memory, ``path`` rendered absolute
            without any storage being touched.

        Raises:
            ~spindoctor.sim.scene.errors.SimSceneValidationError: If the scene fails
                validation.
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

        # Rendering validates the scene first, so a scene supplied in memory is
        # held to the rules a scene file meets when it loads, and the size and
        # offset fields read below are known to be well formed.  A scene file was
        # checked when it loaded, under its local path, and the render uses that
        # label too, so it reuses the check instead of running it again.  A scene
        # supplied in memory keeps the label its caller gave.
        label = str(abspath) if provided_sim_params is None else str(scene_path)
        logger.debug('Rendering combined simulated model')
        img_rendered, meta = render_scene(sim_params, source=label)
        size_v = int(sim_params['size_v'])
        size_u = int(sim_params['size_u'])
        offset_v = float(sim_params.get('offset_v', 0.0))
        offset_u = float(sim_params.get('offset_u', 0.0))

        # Build a basic Snapshot with a flat FOV and dummy geometry
        fov = oops.fov.FlatFOV((1.0, 1.0), (size_u, size_v))
        snapshot = Snapshot(
            axes=('v', 'u'),
            cadence=oops.cadence.SnapCadence(0.0, 1.0),
            fov=fov,
            path='SSB',
            frame='J2000',
        )
        snapshot.insert_subfield('data', img_rendered)

        # Select the navigator block of the camera the sim is emulating, so the
        # orchestrator sees the right units / noise / saturation settings.
        # Validation requires every scene to set its instrument.
        inst_config = scene_camera_block(config, sim_params)
        if extfov_margin_vu is None:
            extfov_margin_vu = resolve_extfov_margin(
                inst_config.extfov_margin_vu, config.camera_block('sim').extfov_margin_vu, size_v
            )

        # A scene may toggle camera-rotation fitting independently of the camera
        # it emulates.  In reality ``fit_camera_rotation`` is a per-camera
        # property, but a sim scene is a navigation fixture: it can ask the
        # navigator to solve for a planted roll on top of any instrument's optics
        # (e.g. Cassini NAC's sharp PSF) without pretending to be a camera that
        # happens to fit rotation.
        fit_rotation_override = sim_params.get('fit_camera_rotation')
        if fit_rotation_override is not None:
            inst_config = dataclasses.replace(
                inst_config, fit_camera_rotation=bool(fit_rotation_override)
            )

        new_obs = ObsSim(
            snapshot,
            config=config,
            extfov_margin_vu=extfov_margin_vu,
            closest_planet=sim_params.get('closest_planet'),
            simulated=True,
        )
        new_obs._camera_block = inst_config
        new_obs.image_url = str(scene_path.absolute())
        new_obs.abspath = abspath

        # The full scene (truth included) stays on the observation for the
        # renderer-side consumers: the test harnesses that score recovery
        # against planted truth and the backplane writer that consumes the
        # rendered masks.  The navigator-side models consume ONLY the
        # navigator view built below.
        new_obs.sim_params = sim_params
        new_obs.sim_offset_v = offset_v
        new_obs.sim_offset_u = offset_u
        new_obs.sim_time = float(sim_params.get('time', 0.0))

        # The information boundary: nav_params is sim_params with every
        # truth key stripped and per-body nav_override overlaid (the
        # navigator sees the geometry it believes, never the true values
        # underneath).  See spindoctor.sim.scene.boundary.build_nav_params.
        new_obs.nav_params = build_nav_params(sim_params)

        # Renderer output metadata (truth side; never read by NavModels).
        new_obs.sim_body_models = meta.get('bodies', {})
        new_obs.sim_inventory = meta.get('inventory', {})
        new_obs.sim_body_order_near_to_far = meta.get('order_near_to_far', [])
        new_obs.sim_body_index_map = meta.get('body_index_map')
        new_obs.sim_body_mask_map = meta.get('body_mask_map', {})

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
        ``spindoctor.sim.render``); a Gaussian core of the published
        ``star_psf_sigma`` then puts a fraction ``1 / (2*pi*sigma**2)`` of that
        total in the peak pixel.  The limiting magnitude is where that peak falls
        to twice the effective per-pixel noise sigma -- the matched-filter
        detection boundary.  The peak pixel never holds more than the star's
        total, so a core narrower than a pixel counts as one pixel, as it does in
        the renderer's faint cutoff for the background sky.  The published DN zero
        point is the camera's electron zero point over its standard gain state, so
        a real navigator could know it; the scene's truth-side noise block is
        deliberately not consulted, so a scene that plants noise different from the
        published values gets a detection limit that does not match the scene's
        noise.  That is desired model error, not a defect.  Keeping this physical
        matters beyond the faint-star gate: the star NavModel synthesizes each STAR
        feature's predicted SNR (and from it the CRLB position covariance and
        reliability score) from how far the star sits above this limit, so an
        arbitrarily permissive placeholder inflates every simulated star's SNR by
        tens of orders of magnitude and collapses its covariance to zero.

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

        The readout binning is read from ``nav_params`` too: a real frame's label
        states its readout mode, so a real pipeline always knows it.  On a frame binned
        by ``b``, the zero point and the read noise come from the block's
        ``noise.by_binning[b]`` entry, which holds the navigator's own values for that
        readout mode: the zero point in DN per second, and the read noise of one binned
        pixel in DN.  If the block has no entry for the readout mode, the frame reads
        the unbinned values.
        ``star_psf_sigma`` is in native pixels, so the core is ``star_psf_sigma / b``
        frame pixels.  An unbinned frame uses the unbinned values.

        Returns:
            The maximum usable magnitude for stars in this observation.  For
            calibrated-unit sim instruments the published block carries no DN
            zero point to anchor a matched-filter limit, so the navigator uses a
            generous constant.

        Raises:
            ValueError: If the observation has no instrument configuration.
        """
        inst_config = self.inst_config
        if inst_config.data_units != 'raw_dn':
            # calibrated_if: no published DN zero point to anchor the limit, so
            # the navigator-side detection limit is a generous constant.
            return 30.0
        # Published values only: the navigator block, with any idealized
        # instrument_config overrides already merged over it.  The scene's truth-side
        # noise block is deliberately not consulted.  A raw_dn block sets both noise
        # values.
        inst_noise = inst_config.noise
        zero_point_dn = cast(float, inst_noise.star_flux_dn_per_s_vmag0)
        read_noise_dn = cast(float, inst_noise.read_noise_dn)
        star_psf_sigma = inst_config.star_psf_sigma
        # A binned frame reads its readout mode's values from the navigator block, or
        # the unbinned values when the block has no entry for the readout mode, and its
        # pixels are b native pixels on a side (#993).
        binning = int(self.nav_params['binning'])
        if binning != 1:
            mode = (inst_noise.by_binning or {}).get(binning)
            if mode is not None:
                zero_point_dn = mode.star_flux_dn_per_s_vmag0
                read_noise_dn = mode.read_noise_dn
            star_psf_sigma /= binning
        # The scene's idealized exposure (navigator-visible; see the docstring
        # for why the Snapshot's reference texp is not consulted here).
        exposure = float(self.nav_params.get('exposure_sec', 1.0))
        # Peak DN of a magnitude-0 star: its total over the Gaussian PSF core, which
        # counts as at least one pixel.
        core_area_px = max(2.0 * math.pi * star_psf_sigma**2, 1.0)
        peak0_dn = zero_point_dn * exposure / core_area_px
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
            'image_name': self.image_name,
            'instrument_host_lid': 'sim',
            'instrument_lid': 'sim',
            'image_shape_xy': self.data_shape_uv,
            'camera': self.camera,
            'description': 'Simulated observation from YAML scene',
        }
