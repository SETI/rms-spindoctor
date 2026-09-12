from pathlib import Path
from typing import Any, cast

import numpy as np
from filecache import FCPath

from spindoctor.config import DEFAULT_CONFIG, IMAGE_LOGGER, Config, logged_section
from spindoctor.support.sclk import exposure_counts, fractional_count, pds3_label_clock_counts
from spindoctor.support.time import et_to_utc
from spindoctor.support.types import PathLike

from .obs_snapshot_inst import ObsSnapshotInst

# SCLK01_MODULI_98 and SCLK01_OFFSETS_98 of the New Horizons spacecraft clock kernel,
# $OOPS_RESOURCES/SPICE/New-Horizons/SCLK/new_horizons_2132.tsc: whole seconds, then
# 1/50000-second ticks.
_SCLK_MODULI = (4294967296, 50000)
_SCLK_OFFSETS = (0, 0)


def _sclk_count(count: str) -> float:
    """Return a New Horizons spacecraft clock count as seconds, with its ticks as a fraction.

    A count is ``SECONDS:TICKS``, after an optional partition and ``/``: whole seconds,
    then the 1/50000-second ticks past them, so ``0031650238:48850`` is
    ``31650238 + 48850 / 50000``, or ``31650238.977``.

    Parameters:
        count: The count as text.

    Returns:
        The count in seconds of the clock.
    """
    _, _, reading = count.strip().rpartition('/')
    seconds, _, ticks = reading.partition(':')
    return fractional_count((int(seconds), int(ticks)), _SCLK_MODULI, _SCLK_OFFSETS)


def _published_sclk(start: str | None, stop: str | None) -> dict[str, float | None]:
    """Return the spacecraft clock counts New Horizons LORRI publishes for one exposure.

    Parameters:
        start: The label's ``SPACECRAFT_CLOCK_START_COUNT``, or None when it carries none.
        stop: The label's ``SPACECRAFT_CLOCK_STOP_COUNT``, or None when it carries none.

    Returns:
        ``start_time_sclk`` and ``end_time_sclk``, the two counts in seconds of the clock,
        and ``midtime_sclk``, their exact mean; a count the label does not carry is None,
        and so is the mean when either count is.
    """
    return exposure_counts(
        None if start is None else _sclk_count(start), None if stop is None else _sclk_count(stop)
    )


class ObsNewHorizonsLORRI(ObsSnapshotInst):
    """Implements an observation of a New Horizons LORRI image.

    This class provides specialized functionality for accessing and analyzing New
    Horizons LORRI image data.
    """

    @staticmethod
    @logged_section('obs', 'LOAD IMAGE')
    def from_file(
        path: PathLike,
        *,
        config: Config | None = None,
        extfov_margin_vu: tuple[int, int] | None = None,
        **_kwargs: Any,
    ) -> 'ObsNewHorizonsLORRI':
        """Creates an ObsNewHorizonsLORRI from a New Horizons LORRI image file.

        Parameters:
            path: Path to the New Horizons LORRI image file.
            config: Configuration object to use. If None, uses the default configuration.
            extfov_margin_vu: Optional tuple that overrides the extended field of view margins
                found in the config.
            **_kwargs: Additional keyword arguments (none for this instrument).

        Returns:
            An ObsNewHorizonsLORRI object containing the image data and metadata.
        """

        import oops.hosts.newhorizons.lorri

        config = config or DEFAULT_CONFIG
        logger = IMAGE_LOGGER

        logger.debug(f'Reading New Horizons LORRI image {path}')
        # calibration=False reads the raw DN image.  LORRI navigates in DN:
        # the calibrated LORRI products are themselves in DN (not I/F), and the
        # navigation pipeline treats image brightness scale-invariantly (NCC
        # correlation, image-derived MAD noise thresholds, magnitude-based star
        # gate), so no I/F conversion is required or expected here.
        obs = oops.hosts.newhorizons.lorri.from_file(path, calibration=False)
        fc_path = FCPath(path)
        obs.abspath = cast(Path, fc_path.get_local_path()).absolute()
        obs.image_url = str(fc_path.absolute())

        inst_config = config.category('newhorizons_lorri')

        if extfov_margin_vu is None:
            if isinstance(inst_config.extfov_margin_vu, dict):
                extfov_margin_vu = inst_config.extfov_margin_vu[obs.data.shape[0]]
            else:
                extfov_margin_vu = inst_config.extfov_margin_vu
        logger.debug(f'  Data shape: {obs.data.shape}')
        logger.debug(f'  Extfov margin vu: {extfov_margin_vu}')
        logger.debug(f'  Data min: {np.min(obs.data)}, max: {np.max(obs.data)}')

        new_obs = ObsNewHorizonsLORRI(obs, config=config, extfov_margin_vu=extfov_margin_vu)
        new_obs._inst_config = inst_config
        # The spacecraft clock counts are in the PDS3 label; the FITS header the
        # observation is read from carries the start count only.
        new_obs._label_clock_counts = pds3_label_clock_counts(fc_path)
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
        aperture D = 0.19 m) by collecting-area.  New Horizons LORRI uses a
        CCD (no detector-sensitivity penalty) and is panchromatic with no
        filter, so a bandpass term of +1.0 mag is added to account for the
        wider passband collecting more flux.  These are nominal optics
        values; the terms are approximate and pending calibration against
        real LORRI star fields.

            anchor = 10.5 + 5*log10(0.208/0.19) (CCD) + 1.0 (panchromatic) ~= 11.7

        Returns:
            The maximum usable magnitude for stars in this observation.
        """

        # Anchor (limiting mag at texp = 1 s) derived above; rounded to 0.1.
        anchor = 11.7
        if self.texp <= 0:
            return anchor
        return cast(float, anchor + np.log(self.texp) / np.log(2.512))

    @property
    def camera(self) -> str:
        """The camera that took this observation.

        Returns:
            Always ``'LORRI'``; New Horizons LORRI is a single camera.
        """
        return 'LORRI'

    def get_public_metadata(self) -> dict[str, Any]:
        """Returns the public metadata for New Horizons LORRI.

        The spacecraft clock counts are those of the PDS3 label beside the image, read when
        the image was loaded, in seconds of the clock; each is None when the label carries
        none.

        Returns:
            A dictionary containing the public metadata for New Horizons LORRI.
        """

        return {
            'image_path': self.image_url,
            'image_name': self.abspath.name,
            'instrument_host_lid': 'urn:nasa:pds:context:instrument_host:spacecraft.nh',
            'instrument_lid': 'urn:nasa:pds:context:instrument:nh.lorri',
            'start_time_utc': et_to_utc(self.time[0]),
            'midtime_utc': et_to_utc(self.midtime),
            'end_time_utc': et_to_utc(self.time[1]),
            'start_time_et': self.time[0],
            'midtime_et': self.midtime,
            'end_time_et': self.time[1],
            **_published_sclk(*self._label_clock_counts),
            'image_shape_xy': self.data_shape_uv,
            'camera': self.camera,
            'exposure_time': self.texp,
            'filters': [],
        }
