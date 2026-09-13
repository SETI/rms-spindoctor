from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any, cast

import numpy as np
from filecache import FCPath

from spindoctor.config import DEFAULT_CONFIG, IMAGE_LOGGER, Config, logged_section
from spindoctor.support.sclk import exposure_counts, fractional_count
from spindoctor.support.time import et_to_utc
from spindoctor.support.types import PathLike

from .obs_snapshot_inst import ObsSnapshotInst

# SCLK01_MODULI_77 and SCLK01_OFFSETS_77 of the Galileo spacecraft clock kernel,
# $OOPS_RESOURCES/SPICE/Galileo/SCLK/mk00062a.tsc: the RIM count, then its 91 MOD91
# counts, each of 10 MOD10 counts, each of 8 MOD8 counts.
_SCLK_MODULI = (16777215, 91, 10, 8)
_SCLK_OFFSETS = (0, 0, 0, 0)

_SCLK_FIELDS = ('RIM', 'MOD91', 'MOD10', 'MOD8')
"""The VICAR label items holding the four fields of an image's frame count."""


def _sclk_count(fields: Sequence[int]) -> Fraction:
    """Return a Galileo spacecraft clock count as a number of RIM counts.

    Parameters:
        fields: The count's four fields: RIM, MOD91, MOD10 and MOD8.

    Returns:
        The count in RIM counts, ``RIM + MOD91 / 91 + MOD10 / 910 + MOD8 / 7280``.
    """
    return fractional_count(fields, _SCLK_MODULI, _SCLK_OFFSETS)


def _published_sclk(label: Mapping[str, Any]) -> dict[str, float | None]:
    """Return the spacecraft clock counts Galileo SSI publishes for one image.

    A Galileo SSI label records one count, the image's frame count, as the VICAR items
    RIM, MOD91, MOD10 and MOD8; it comes a few seconds before the exposure.  The label
    records no count at the end of the image.

    Parameters:
        label: The image's VICAR label.

    Returns:
        ``start_time_sclk``, the frame count in RIM counts, or None when the label lacks
        any of the four items; ``midtime_sclk`` and ``end_time_sclk``, always None.
    """
    fields: list[Any] = [label.get(name, None) for name in _SCLK_FIELDS]
    if None in fields:
        return exposure_counts(None, None, bracketed=False)
    return exposure_counts(_sclk_count(fields), None, bracketed=False)


class ObsGalileoSSI(ObsSnapshotInst):
    """Implements an observation of a Galileo SSI image.

    This class provides specialized functionality for accessing and analyzing Galileo
    SSI image data.
    """

    @staticmethod
    @logged_section('obs', 'LOAD IMAGE')
    def from_file(
        path: PathLike,
        *,
        config: Config | None = None,
        extfov_margin_vu: tuple[int, int] | None = None,
        **_kwargs: Any,
    ) -> 'ObsGalileoSSI':
        """Creates an ObsGalileoSSI from a Galileo SSI image file.

        Parameters:
            path: Path to the Galileo SSI image file.
            config: Configuration object to use. If None, uses the default configuration.
            extfov_margin_vu: Optional tuple that overrides the extended field of view margins
                found in the config.
            **_kwargs: Additional keyword arguments (none for this instrument).

        Returns:
            An ObsGalileoSSI object containing the image data and metadata.
        """

        import oops.hosts.galileo.ssi

        config = config or DEFAULT_CONFIG
        logger = IMAGE_LOGGER

        logger.debug(f'Reading Galileo SSI image {path}')
        # Galileo SSI navigates in raw DN: there is no I/F-calibrated SSI
        # product and there never will be.  The navigation pipeline treats
        # image brightness scale-invariantly (NCC correlation, image-derived
        # MAD noise thresholds, magnitude-based star gate), so no photometric
        # calibration is required here.
        obs = oops.hosts.galileo.ssi.from_file(path, full_fov=True)
        fc_path = FCPath(path)
        obs.abspath = cast(Path, fc_path.get_local_path()).absolute()
        obs.image_url = str(fc_path.absolute())

        inst_config = config.category('galileo_ssi')
        if extfov_margin_vu is None:
            if isinstance(inst_config.extfov_margin_vu, dict):
                # TODO Do this a better way
                extfov_margin_vu = inst_config.extfov_margin_vu[obs.data.shape[0]]
            else:
                extfov_margin_vu = inst_config.extfov_margin_vu
        logger.debug(f'  Data shape: {obs.data.shape}')
        logger.debug(f'  Extfov margin vu: {extfov_margin_vu}')
        logger.debug(f'  Data min: {np.min(obs.data)}, max: {np.max(obs.data)}')

        new_obs = ObsGalileoSSI(obs, config=config, extfov_margin_vu=extfov_margin_vu)
        new_obs._inst_config = inst_config
        return new_obs

    def star_min_usable_vmag(self) -> float:
        """Returns the minimum usable magnitude for stars in this observation.

        Mirrors the Cassini ISS reference implementation, which imposes no
        bright-end cutoff (saturation of bright stars is handled elsewhere).

        Returns:
            The minimum usable magnitude for stars in this observation.
        """
        return 0.0

    def star_max_usable_vmag(self) -> float:
        """Returns the maximum usable magnitude for stars in this observation.

        The limiting magnitude follows the Cassini Pogson-ratio form,

            star_max_usable_vmag(texp) = anchor + log(texp) / log(2.512)

        where ``anchor`` is the limiting magnitude at a 1 s exposure (each
        2.512x increase in exposure buys +1 mag of depth).

        The anchor is scaled from the Cassini NAC anchor (10.5 mag at 1 s,
        aperture D = 0.19 m) by collecting-area.  Galileo SSI uses a CCD, so
        no detector-sensitivity penalty is applied.  These are nominal optics
        values; the term is approximate and pending calibration against real
        Galileo star fields.

            anchor = 10.5 + 5*log10(0.176/0.19) (CCD) ~= 10.3

        Returns:
            The maximum usable magnitude for stars in this observation.
        """

        # Anchor (limiting mag at texp = 1 s) derived above; rounded to 0.1.
        anchor = 10.3
        if self.texp <= 0:
            return anchor
        return cast(float, anchor + np.log(self.texp) / np.log(2.512))

    @property
    def camera(self) -> str:
        """The camera that took this observation.

        Returns:
            Always ``'SSI'``; Galileo carries a single camera.
        """
        return 'SSI'

    def get_public_metadata(self) -> dict[str, Any]:
        """Returns the public metadata for Galileo SSI.

        The spacecraft clock count is the VICAR label's frame count, in RIM counts, which
        comes a few seconds before the exposure.  The label records no count at the end
        of the image, so the midtime and end counts are None.

        Returns:
            A dictionary containing the public metadata for Galileo SSI.
        """

        return {
            'image_path': self.image_url,
            'image_name': self.abspath.name,
            'instrument_host_lid': 'urn:nasa:pds:context:instrument_host:spacecraft.go',
            'instrument_lid': 'urn:nasa:pds:context:instrument:go.ssi',
            'start_time_utc': et_to_utc(self.time[0]),
            'midtime_utc': et_to_utc(self.midtime),
            'end_time_utc': et_to_utc(self.time[1]),
            'start_time_et': self.time[0],
            'midtime_et': self.midtime,
            'end_time_et': self.time[1],
            **_published_sclk(self.dict),
            'image_shape_xy': self.data_shape_uv,
            'camera': self.camera,
            'exposure_time': self.texp,
            'filters': [self.filter],
        }
