from typing import Any

from oops.observation.snapshot import Snapshot
from psfmodel import PSF, GaussianPSF

from .obs_inst import ObsInst
from .obs_snapshot import ObsSnapshot


class ObsSnapshotInst(ObsSnapshot, ObsInst):
    """Mix-in of ObsSnapshot and ObsInst."""

    def __init__(self, snapshot: Snapshot, **kwargs: Any) -> None:
        """Initializes a new ObsSnapshotInst instance.

        Parameters:
            snapshot: The Snapshot object to wrap.
            **kwargs: Additional keyword arguments used by subclasses.
        """
        super().__init__(snapshot, **kwargs)

    def star_sharpness_psf(self) -> PSF | None:
        """Return the narrowest point-spread function a star in this image can have.

        The instrument configuration's optional ``star_sharpness_psf_sigma`` is the
        Gaussian sigma, in pixels, of the narrowest star the camera records: one value, or
        a mapping from the image's row count to a value, since summing pixels narrows a
        star in pixel units.  A detection more concentrated than this point-spread
        function allows a star is a cosmic-ray hit.  A mapping lists only the row counts
        at which the narrowest stars have been measured.

        Returns:
            A Gaussian PSF of that sigma; None when the instrument configuration is unset,
            gives no ``star_sharpness_psf_sigma``, or maps none to the image's row count.
        """
        if self._inst_config is None or 'star_sharpness_psf_sigma' not in self._inst_config:
            return None
        entry = self._inst_config['star_sharpness_psf_sigma']
        if isinstance(entry, dict):
            rows = self.data_shape_vu[0]
            if rows not in entry:
                return None
            entry = entry[rows]
        return GaussianPSF(sigma=float(entry))
