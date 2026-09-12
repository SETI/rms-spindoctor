from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from psfmodel import PSF, GaussianPSF
from starcat import Star

from spindoctor.config import Config
from spindoctor.support.types import PathLike

if TYPE_CHECKING:
    from spindoctor.obs import Obs


class ObsInst(ABC):
    """Mix-in class for instrument models representing spacecraft cameras.

    This class provides default functionality for methods related to instruments
    and abstract methods for instrument-specific functionality.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self._inst_config: dict[str, Any] | None = None
        self.is_simulated: bool = kwargs.get('simulated', False)

    @property
    def inst_config(self) -> dict[str, Any] | None:
        """Returns the instrument configuration."""
        return self._inst_config

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

        This generic implementation uses the "star_psf_sigma" configuration value and
        creates a Gaussian PSF with that sigma.

        Returns:
            A PSF model appropriate for stars observed by this instrument.
        """

        if self._inst_config is None:
            raise ValueError('Instrument configuration not set')

        sigma = self._inst_config['star_psf_sigma']
        return GaussianPSF(sigma=sigma)

    def star_psf_size(self, star: Star) -> tuple[int, int]:
        """Returns the size of the point spread function (PSF) to use for a star.

        This generic implementation uses the "star_psf_sizes" configuration value and
        returns the appropriate value for the star's magnitude.

        Parameters:
            star: The star to get the PSF size for.

        Returns:
            A tuple of the PSF size (v, u) in pixels.
        """

        if self._inst_config is None:
            raise ValueError('Instrument configuration not set')

        star_psf_sizes = self._inst_config['star_psf_sizes']
        keys = sorted(star_psf_sizes)
        if not keys:
            raise ValueError('star_psf_sizes is empty')

        default_mag = max(keys)
        selected_mag = default_mag
        for mag in keys:
            if star.vmag < mag:
                selected_mag = mag
                break

        seq = star_psf_sizes[selected_mag]
        assert len(seq) == 2, f'star_psf_sizes[{selected_mag}] must have 2 elements, got {seq!r}'
        return (int(seq[0]), int(seq[1]))

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
