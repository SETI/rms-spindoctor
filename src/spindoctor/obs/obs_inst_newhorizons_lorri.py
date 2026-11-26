from fractions import Fraction
from typing import Any, cast

import host_new_horizons
import numpy as np

from spindoctor.config import DEFAULT_CONFIG, IMAGE_LOGGER, Config, logged_section
from spindoctor.support.sclk import exposure_counts, fractional_count
from spindoctor.support.time import et_to_utc
from spindoctor.support.types import NDArrayAnyType, PathLike

from .obs_snapshot_inst import ObsSnapshotInst, host_input

# SCLK01_MODULI_98 and SCLK01_OFFSETS_98 of the New Horizons spacecraft clock kernel,
# $OOPS_RESOURCES/SPICE/New-Horizons/SCLK/new_horizons_2132.tsc: whole seconds, then
# 1/50000-second ticks.
_SCLK_MODULI = (4294967296, 50000)
_SCLK_OFFSETS = (0, 0)


def _sclk_count(count: str) -> Fraction:
    """Return a New Horizons clock count as seconds, with its ticks as a fraction.

    A count is ``SECONDS:TICKS``: whole seconds, then the 1/50000-second ticks past them,
    so ``0031650238:48850`` is ``31650238 + 48850 / 50000``, or ``31650238.977``.

    Parameters:
        count: The count as text.

    Returns:
        The count in seconds of the clock, exactly.
    """
    seconds, _, ticks = count.strip().partition(':')
    return fractional_count((int(seconds), int(ticks)), _SCLK_MODULI, _SCLK_OFFSETS)


def _published_sclk(start: str | None, stop: str | None) -> dict[str, float | None]:
    """Return the spacecraft clock counts New Horizons LORRI publishes for one exposure.

    A LORRI label's two counts mark the start and the end of the exposure, so the count
    halfway between them is its middle.

    Parameters:
        start: The label's ``SPACECRAFT_CLOCK_START_COUNT``, or None when it carries none.
        stop: The label's ``SPACECRAFT_CLOCK_STOP_COUNT``, or None when it carries none.

    Returns:
        ``start_time_sclk`` and ``end_time_sclk``, the two counts in seconds of the
        clock, and ``midtime_sclk``, their exact mean; a count the label does not carry
        is None, and so is the mean when either count is.
    """
    return exposure_counts(
        None if start is None else _sclk_count(start),
        None if stop is None else _sclk_count(stop),
        bracketed=True,
    )


class ObsNewHorizonsLORRI(ObsSnapshotInst):
    """Implements an observation of a New Horizons LORRI image.

    This class provides specialized functionality for accessing and analyzing New
    Horizons LORRI image data.
    """

    model_source = 'spice'

    @staticmethod
    @logged_section('obs', 'LOAD IMAGE')
    def from_file(
        path: PathLike,
        *,
        config: Config | None = None,
        extfov_margin_vu: tuple[int, int] | None = None,
        label: PathLike | None = None,
        **_kwargs: Any,
    ) -> 'ObsNewHorizonsLORRI':
        """Creates an ObsNewHorizonsLORRI from a New Horizons LORRI image file.

        The host reads the image's PDS3 label together with the FITS file (see
        :func:`~spindoctor.obs.obs_snapshot_inst.host_input`).

        Parameters:
            path: Path to the New Horizons LORRI image file.
            config: Configuration object to use. If None, uses the default configuration.
            extfov_margin_vu: Optional tuple that overrides the extended field of view margins
                found in the config.
            label: The image's PDS3 label, if known; otherwise the label beside the image.
            **_kwargs: Additional keyword arguments (none for this instrument).

        Returns:
            An ObsNewHorizonsLORRI object containing the image data and metadata.

        Raises:
            OSError, RuntimeError: If the SPICE kernels lack data the image needs, such as
                its pointing, which cspyce reports when the observation is first used.
            oops.host.HostError: If the host cannot read the image or its label.
        """

        config = config or DEFAULT_CONFIG
        logger = IMAGE_LOGGER

        logger.debug(f'Reading New Horizons LORRI image {path}')
        # The host gives the data as the FITS file stores them, in DN, and where it
        # can, attaches the conversion to I/F as a calibration, which is not applied:
        # the navigation pipeline treats image brightness scale-invariantly (NCC
        # correlation, image-derived MAD noise thresholds, magnitude-based star gate).
        obs = host_new_horizons.NewHorizonsLORRI.from_file(host_input(path, label=label))
        # The host inserts the image as the data subfield, which the Snapshot it returns
        # does not declare.
        data: NDArrayAnyType = obs.subfields['data']

        inst_config = config.category('newhorizons_lorri')

        if extfov_margin_vu is None:
            if isinstance(inst_config.extfov_margin_vu, dict):
                extfov_margin_vu = inst_config.extfov_margin_vu[data.shape[0]]
            else:
                extfov_margin_vu = inst_config.extfov_margin_vu
        logger.debug(f'  Data shape: {data.shape}')
        logger.debug(f'  Extfov margin vu: {extfov_margin_vu}')
        logger.debug(f'  Data min: {np.min(data)}, max: {np.max(data)}')

        new_obs = ObsNewHorizonsLORRI(obs, config=config, extfov_margin_vu=extfov_margin_vu)
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

        The spacecraft clock counts are the host's ``spacecraft_clock_start_count`` and
        ``spacecraft_clock_stop_count``, in seconds of the clock; each is None when no
        label gives it.

        Returns:
            A dictionary containing the public metadata for New Horizons LORRI.
        """

        return {
            'image_path': self.image_url,
            'image_name': self.image_name,
            'instrument_host_lid': 'urn:nasa:pds:context:instrument_host:spacecraft.nh',
            'instrument_lid': 'urn:nasa:pds:context:instrument:nh.lorri',
            'start_time_utc': et_to_utc(self.time[0]),
            'midtime_utc': et_to_utc(self.midtime),
            'end_time_utc': et_to_utc(self.time[1]),
            'start_time_et': self.time[0],
            'midtime_et': self.midtime,
            'end_time_et': self.time[1],
            **_published_sclk(
                self.subfields.get('spacecraft_clock_start_count'),
                self.subfields.get('spacecraft_clock_stop_count'),
            ),
            'image_shape_xy': self.data_shape_uv,
            'camera': self.camera,
            'exposure_time': self.texp,
            'filters': [],
        }
