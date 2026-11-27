import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast

from psfmodel import PSF, GaussianPSF
from starcat import Star

from spindoctor.config import Config
from spindoctor.config.sections_camera import CameraBlock, StarPhotometry, as_camera_block
from spindoctor.support.types import PathLike

if TYPE_CHECKING:
    from spindoctor.obs import Obs

ModelSource = Literal['scene_view', 'spice']
"""The two kinds of input a navigation model can predict the scene from.

``'spice'`` is SPICE geometry with the star, body, and ring catalogs, which is what a
real camera's image provides.  ``'scene_view'`` is the navigator's view of a simulated
scene.  An observation provides one of them, and each navigation model class uses one.
"""


class ObsInst(ABC):
    """Mix-in class for instrument models representing spacecraft cameras.

    This class provides default functionality for methods related to instruments
    and abstract methods for instrument-specific functionality.
    """

    model_source: ClassVar[ModelSource]
    """The input this observation provides to its navigation models.

    Only the model classes whose ``inputs`` matches it are built for the image.
    """

    _abstract: ClassVar[bool] = True
    """Set to True in the body of a class that is only a shared base, so that it need
    not set ``model_source``."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Require every concrete observation class to set ``model_source``.

        Raises:
            TypeError: If a concrete subclass does not set ``model_source`` in its own
                class body.  Inheriting it does not count, so each instrument's
                choice is written where the instrument is defined.
        """
        super().__init_subclass__(**kwargs)
        if cls.__dict__.get('_abstract', False):
            return
        if 'model_source' not in cls.__dict__:
            raise TypeError(
                f"{cls.__qualname__} must set model_source ('spice' or 'scene_view') "
                'in its class body'
            )

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self._camera_block: CameraBlock | None = None
        # The simulated observation holds its camera block as a mapping, with its
        # scene's overrides merged over it, until the simulator types it (#1029).
        self._inst_config: dict[str, Any] | None = None
        self._parsed_inst_config: tuple[dict[str, Any], CameraBlock] | None = None
        self.is_simulated: bool = kwargs.get('simulated', False)

    @property
    def inst_config(self) -> CameraBlock:
        """The block of the camera that took this observation.

        An observation class that holds its block as a mapping has it parsed into the
        same type (see :func:`~spindoctor.config.sections_camera.as_camera_block`) when
        it is first read.

        Raises:
            ValueError: If the observation has no camera block.
            ~spindoctor.schema.SchemaError: If a block held as a mapping breaks the
                camera block's schema.
        """
        if self._camera_block is not None:
            return self._camera_block
        if self._inst_config is None:
            raise ValueError('Instrument configuration not set')
        parsed = self._parsed_inst_config
        if parsed is None or parsed[0] is not self._inst_config:
            block = as_camera_block(self._inst_config, source='the observation camera block')
            parsed = (self._inst_config, block)
            self._parsed_inst_config = parsed
        return parsed[1]

    @property
    @abstractmethod
    def camera(self) -> str:
        """The camera that took this observation.

        Instruments with more than one camera distinguish them here (Cassini
        ISS and Voyager ISS return ``'NAC'`` or ``'WAC'``); single-camera
        instruments return their one camera's name.  Pointing error is a
        property of the camera, not of the instrument, so statistics are
        grouped by this value.

        Returns:
            The camera name.
        """
        ...

    @property
    def shutter_mode(self) -> str | None:
        """The shutter mode this observation was taken in.

        Instruments that can expose more than one camera at once name the
        mode here, because two cameras exposed simultaneously share one
        spacecraft attitude and a consumer cannot tell that from the times
        alone.  Instruments whose labels carry no such field return None.

        Returns:
            The instrument's shutter mode string, or None when the host
            exposes none.
        """
        return None

    @staticmethod
    @abstractmethod
    def from_file(
        path: PathLike,
        *,
        config: Config | None = None,
        extfov_margin_vu: tuple[int, int] | None = None,
        **kwargs: Any,
    ) -> 'Obs':
        """Creates an instrument instance from an image file.

        Parameters:
            path: Path to the image file.
            config: Configuration object to use. If None, uses DEFAULT_CONFIG.
            extfov_margin_vu: Optional tuple specifying the extended field of view margins
                in (vertical, horizontal) pixels.
            **kwargs: Additional keyword arguments to pass to the instrument constructor.

        Returns:
            An Obs object containing the image data and metadata.
        """
        ...

    def star_psf(self) -> PSF:
        """Returns the point spread function (PSF) model appropriate for stars observed
        by this instrument.

        The PSF is a Gaussian whose sigma is :meth:`star_psf_sigma`.

        Returns:
            A PSF model appropriate for stars observed by this instrument.

        Raises:
            ValueError: If the observation has no camera block.
        """

        return GaussianPSF(sigma=self.star_psf_sigma())

    def star_psf_sigma(self) -> float:
        """Returns the Gaussian sigma, in pixels, of a star in this observation.

        This generic implementation reads the camera block's ``star_psf_sigma``.  An
        instrument whose star width depends on the camera or the readout mode overrides
        it.

        Returns:
            The sigma in pixels of this observation's image.

        Raises:
            ValueError: If the observation has no camera block.
        """

        return self.inst_config.star_psf_sigma

    def _star_photometry(self, where: str) -> StarPhotometry:
        """Return the camera block's ``star_photometry``, refusing a block without one.

        Parameters:
            where: The key path of the camera block's ``star_photometry``, which a message
                gives.

        Returns:
            The camera's star photometry.

        Raises:
            ValueError: If the observation has no camera block, or its block has no
                ``star_photometry``.
        """
        photometry = self.inst_config.star_photometry
        if photometry is None:
            raise ValueError(f'{where} is not set; the star limit and star width read it')
        return photometry

    def _photometric_star_limit(
        self,
        *,
        texp: float,
        where: str,
        gain_state: str,
        mode: str,
        filter_name: str | None,
    ) -> float:
        """Return the faintest V magnitude of a catalog star, from the star photometry.

        A star of V magnitude ``m`` deposits ``flux_per_s_vmag0 * 10**(-0.4 * (m +
        offset)) * texp`` in the unit of ``flux_unit``, divided by the gain state's entry
        for electrons or multiplied by it for DN, where ``offset`` is the filter's
        magnitude offset (0 when ``filter_name`` is None).  The dimmest visible star
        deposits ``min_detectable_dn[mode]`` DN, and the limit is its magnitude plus
        ``search_depth_mag`` (see :class:`~spindoctor.config.sections_camera.StarPhotometry`).

        Parameters:
            texp: The exposure in seconds; positive.
            where: The key path of the camera block's ``star_photometry``, which a message
                gives.
            gain_state: The image's gain state, as the ``gain`` table spells it.
            mode: The image's readout mode, as the mode tables spell it.
            filter_name: The image's filter, as ``filter_mag_offset`` spells it, or None
                for a camera whose reader does not read a filter.

        Returns:
            The limiting magnitude.

        Raises:
            ValueError: If the block has no ``star_photometry``, or its tables leave out the
                image's readout mode, filter, or gain state, or it has no
                ``filter_mag_offset`` and ``filter_name`` is not None.
        """
        photometry = self._star_photometry(where)
        if mode not in photometry.min_detectable_dn:
            raise ValueError(f'{where}.min_detectable_dn has no entry for the mode {mode!r}')
        offset = 0.0
        if filter_name is not None:
            offsets = photometry.filter_mag_offset
            if offsets is None:
                raise ValueError(f'{where} has no filter_mag_offset, which this camera reads')
            if filter_name not in offsets:
                raise ValueError(
                    f'{where}.filter_mag_offset has no entry for the filter {filter_name!r}'
                )
            offset = offsets[filter_name]
        if gain_state not in photometry.gain:
            raise ValueError(f'{where}.gain has no entry for the gain state {gain_state!r}')
        gain = photometry.gain[gain_state]
        signal = photometry.flux_per_s_vmag0 * texp
        dn_vmag0 = signal / gain if photometry.flux_unit == 'electrons' else signal * gain
        dimmest_visible = 2.5 * math.log10(dn_vmag0 / photometry.min_detectable_dn[mode])
        return dimmest_visible - offset + photometry.search_depth_mag

    def _photometric_star_sigma(self, *, where: str, mode: str) -> float:
        """Return the Gaussian sigma of a star, in pixels, from the star photometry.

        Parameters:
            where: The key path of the camera block's ``star_photometry``, which a message
                gives.
            mode: The image's readout mode, as the mode tables spell it.

        Returns:
            The ``psf_sigma`` entry of the mode.

        Raises:
            ValueError: If the block has no ``star_photometry``, or its ``psf_sigma`` leaves
                out the mode.
        """
        widths = self._star_photometry(where).psf_sigma
        if mode not in widths:
            raise ValueError(f'{where}.psf_sigma has no entry for the mode {mode!r}')
        return widths[mode]

    def star_psf_size(self, star: Star) -> tuple[int, int]:
        """Returns the size of the point spread function (PSF) to use for a star.

        This generic implementation uses the camera's ``star_psf_sizes``: the star takes
        the first bin, in order of magnitude, whose magnitude it is brighter than, and a
        star fainter than every bin takes the last.

        Parameters:
            star: The star to get the PSF size for.

        Returns:
            A tuple of the PSF size (v, u) in pixels.

        Raises:
            ValueError: If the observation has no camera block.
        """

        star_psf_sizes = self.inst_config.star_psf_sizes
        keys = sorted(star_psf_sizes)
        # A star reaches the PSF model only once it has a magnitude.
        vmag = cast(float, star.vmag)
        selected_mag = next((mag for mag in keys if vmag < mag), keys[-1])
        return star_psf_sizes[selected_mag]

    @abstractmethod
    def star_min_usable_vmag(self) -> float:
        """Returns the minimum usable magnitude for stars in this observation.

        Returns:
            The minimum usable magnitude for stars in this observation.
        """
        raise NotImplementedError

    @abstractmethod
    def star_max_usable_vmag(self) -> float:
        """Returns the maximum usable magnitude for stars in this observation.

        Returns:
            The maximum usable magnitude for stars in this observation.
        """
        raise NotImplementedError

    @abstractmethod
    def get_public_metadata(self) -> dict[str, Any]:
        """Return the facts this instrument's host publishes about the image.

        The navigation document's ``observation`` block records each of them after the
        image's identity, except a fact the block already states, and the summary PNG's
        caption reads the image name, filters and exposure time from them.

        Returns:
            The facts, keyed by name in the host's own order: the image's path and name,
            the PDS4 context identifiers of the spacecraft and the instrument, the image
            shape as ``(x, y)``, the camera, and, where the host knows them, the start,
            midtime and end of the exposure, the exposure time, the filters and whatever
            else the host states about the image.
        """
        ...
