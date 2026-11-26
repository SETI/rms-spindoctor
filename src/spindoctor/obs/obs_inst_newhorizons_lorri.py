import math
from fractions import Fraction
from typing import Any

import numpy as np
from filecache import FCPath

from spindoctor.config import DEFAULT_CONFIG, IMAGE_LOGGER, Config, logged_section
from spindoctor.support.sclk import exposure_counts, fractional_count, pds3_label_clock_counts
from spindoctor.support.time import et_to_utc
from spindoctor.support.types import PathLike

from .obs_snapshot_inst import ObsSnapshotInst
from .star_photometry_newhorizons_lorri import NewHorizonsLORRIBinning

_STAR_PHOTOMETRY_SECTION = 'newhorizons_lorri_star_photometry'
"""The configuration section holding the camera's star photometry."""

_BINNING_BY_ROWS: dict[int, NewHorizonsLORRIBinning] = {1024: '1X1', 256: '4X4'}
"""The binning mode of a frame of each row count, as the image host reads it."""

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

    _label_clock_counts: tuple[str | None, str | None]
    """The start and stop clock counts of the PDS3 label beside the image."""

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
        # The host module publishes no type information for from_file.
        obs = oops.hosts.newhorizons.lorri.from_file(  # type: ignore[no-untyped-call]
            path, calibration=False
        )
        fc_path = FCPath(path)

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

    @property
    def binning(self) -> NewHorizonsLORRIBinning:
        """The binning mode this observation was read out in.

        Returns:
            ``'1X1'`` for a 1024-row frame or ``'4X4'`` for a 256-row frame, the rule
            the image host applies.

        Raises:
            ValueError: If the frame has any other row count.
        """
        rows = int(self.data_shape_vu[0])
        if rows not in _BINNING_BY_ROWS:
            raise ValueError(f'LORRI frame has {rows} rows; a binning mode needs 1024 or 256')
        return _BINNING_BY_ROWS[rows]

    def star_max_usable_vmag(self) -> float:
        """Returns the faintest V magnitude of a catalog star the star model uses.

        The limit is photometric.  A star of V magnitude ``m`` and the default spectral
        class deposits ``rate * 10**(-0.4 * m) * texp / gain`` DN in the image, where
        ``rate`` is the camera's electron rate for a V = 0 star, ``texp`` is the exposure
        in seconds and ``gain`` is the electrons per DN of the frame's binning mode.  The
        dimmest visible star deposits the binning mode's ``min_detectable_dn``, and the
        limit is that star's magnitude plus ``search_depth_mag``.  Every quantity but
        the exposure comes from the ``newhorizons_lorri_star_photometry`` configuration
        section.

        Returns:
            The limiting magnitude, or ``-inf`` for a frame whose exposure is not
            positive.

        Raises:
            ValueError: If the frame has neither 1024 nor 256 rows, or the section lacks
                values for its binning mode.
        """
        if self.texp <= 0.0:
            return -math.inf
        photometry = self.config.newhorizons_lorri_star_photometry.lorri
        binning = self.binning
        if binning not in photometry.electrons_per_dn:
            raise ValueError(
                f'{_STAR_PHOTOMETRY_SECTION} has no values for the binning mode {binning!r}'
            )
        dn_vmag0 = (
            photometry.flux_electrons_per_s_vmag0 * self.texp / photometry.electrons_per_dn[binning]
        )
        dimmest_visible = 2.5 * math.log10(dn_vmag0 / photometry.min_detectable_dn[binning])
        return dimmest_visible + photometry.search_depth_mag

    def star_psf_sigma(self) -> float:
        """Returns the Gaussian sigma, in pixels, of a star in this observation.

        Returns:
            The ``star_psf_sigma`` of the ``newhorizons_lorri_star_photometry``
            configuration section for the frame's binning mode.

        Raises:
            ValueError: If the frame has neither 1024 nor 256 rows, or the section lacks a
                width for its binning mode.
        """
        widths = self.config.newhorizons_lorri_star_photometry.lorri.star_psf_sigma
        binning = self.binning
        if binning not in widths:
            raise ValueError(
                f'{_STAR_PHOTOMETRY_SECTION} has no star_psf_sigma for the binning mode {binning!r}'
            )
        return widths[binning]

    @property
    def camera(self) -> str:
        """The camera that took this observation.

        Returns:
            Always ``'LORRI'``; New Horizons LORRI is a single camera.
        """
        return 'LORRI'

    def get_public_metadata(self) -> dict[str, Any]:
        """Returns the public metadata for New Horizons LORRI.

        The spacecraft clock counts are those of the PDS3 label beside the image, read
        when the image was loaded, in seconds of the clock; each is None when the label
        carries none.

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
            **_published_sclk(*self._label_clock_counts),
            'image_shape_xy': self.data_shape_uv,
            'camera': self.camera,
            'exposure_time': self.texp,
            'filters': [],
        }
