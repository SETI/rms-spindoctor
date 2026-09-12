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

# SCLK01_MODULI_82 and SCLK01_OFFSETS_82 of the Cassini spacecraft clock kernel,
# $OOPS_RESOURCES/SPICE/Cassini/SCLK/cas00172.tsc: whole seconds, then 1/256-second ticks.
_SCLK_MODULI = (4294967296, 256)
_SCLK_OFFSETS = (0, 0)

_SCLK_TICK_DIGITS = 3
"""Digits in the tick field of a spacecraft clock count as an image label writes it."""


def _sclk_count(count: str) -> Fraction:
    """Return a Cassini spacecraft clock count as seconds, with its ticks as a fraction.

    A count is ``SECONDS.TICKS``: whole seconds, then the 1/256-second ticks past them,
    written as three digits, so ``1459229915.075`` is ``1459229915 + 75 / 256``, or
    ``1459229915.29296875``.  A tick field with fewer digits has lost its trailing zeros,
    as an index table's copy of a count can, and is padded back on the right:
    ``1347929382.11`` is ``1347929382.110``.

    Parameters:
        count: The count as text.

    Returns:
        The count in seconds of the clock, exactly.
    """
    seconds, _, ticks = count.strip().partition('.')
    return fractional_count(
        (int(seconds), int(ticks.ljust(_SCLK_TICK_DIGITS, '0'))), _SCLK_MODULI, _SCLK_OFFSETS
    )


def _published_sclk(start: str | None, stop: str | None) -> dict[str, float | None]:
    """Return the spacecraft clock counts Cassini ISS publishes for one exposure.

    A Cassini ISS label's two counts mark the start and the end of the exposure, so the
    count halfway between them is its middle.

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


class ObsCassiniISS(ObsSnapshotInst):
    """Implements an observation of a Cassini ISS image.

    This class provides specialized functionality for accessing and analyzing Cassini
    ISS image data.
    """

    @staticmethod
    @logged_section('obs', 'LOAD IMAGE')
    def from_file(
        path: PathLike,
        *,
        config: Config | None = None,
        extfov_margin_vu: tuple[int, int] | None = None,
        **kwargs: Any,
    ) -> 'ObsCassiniISS':
        """Creates an ObsCassiniISS from a Cassini ISS image file.

        Parameters:
            path: Path to the Cassini ISS image file.
            config: Configuration object to use. If None, uses the default configuration.
            extfov_margin_vu: Optional tuple that overrides the extended field of view margins
                found in the config.
            **kwargs: Additional keyword arguments:

                - fast_distortion: Whether to use a fast distortion model.

                - return_all_planets: Whether to return all planets.

        Returns:
            An ObsCassiniISS object containing the image data and metadata.
        """

        import oops.hosts.cassini.iss

        config = config or DEFAULT_CONFIG
        logger = IMAGE_LOGGER

        fast_distortion = kwargs.get('fast_distortion', True)
        return_all_planets = kwargs.get('return_all_planets', True)

        logger.debug(f'Reading Cassini ISS image {path}')
        logger.debug(f'  Fast distortion: {fast_distortion}')
        logger.debug(f'  Return all planets: {return_all_planets}')
        obs = oops.hosts.cassini.iss.from_file(
            path, fast_distortion=fast_distortion, return_all_planets=return_all_planets
        )
        fc_path = FCPath(path)
        obs.abspath = cast(Path, fc_path.get_local_path()).absolute()
        obs.image_url = str(fc_path.absolute())

        detector = obs.detector.lower()
        # Cassini ISS CALIB products carry pixels in I/F units; the RAW
        # and CALIB pipelines have different blank / saturation /
        # noisy thresholds.  ``_CALIB.IMG`` in the filename selects the
        # ``cassini_iss_calib`` config block instead of ``cassini_iss``.
        is_calibrated = '_CALIB' in fc_path.name.upper()
        inst_section = 'cassini_iss_calib' if is_calibrated else 'cassini_iss'
        category_dict = config.category(inst_section)
        if not category_dict:
            raise ValueError(
                f'Cassini ISS config section {inst_section!r} is missing or empty; '
                f'expected for detector {detector!r}'
            )
        if detector not in category_dict:
            raise ValueError(
                f'Cassini ISS config section {inst_section!r} has no entry for '
                f'detector {detector!r}; available detectors: {sorted(category_dict)}'
            )
        inst_config = category_dict[detector]

        if extfov_margin_vu is None:
            extfov_margin_vu_entry = inst_config['extfov_margin_vu']
            if isinstance(extfov_margin_vu_entry, dict):
                extfov_margin_vu = extfov_margin_vu_entry[obs.data.shape[0]]
            else:
                extfov_margin_vu = extfov_margin_vu_entry
        logger.debug(f'  Data shape: {obs.data.shape}')
        logger.debug(f'  Extfov margin vu: {extfov_margin_vu}')
        # TODO This is slow when debug turned off
        logger.debug(f'  Data min: {np.min(obs.data)}, max: {np.max(obs.data)}')

        new_obs = ObsCassiniISS(obs, config=config, extfov_margin_vu=extfov_margin_vu)
        new_obs._inst_config = inst_config
        return new_obs

    def star_min_usable_vmag(self) -> float:
        """Returns the minimum usable magnitude for stars in this observation.

        Returns:
            The minimum usable magnitude for stars in this observation.
        """

        if self.detector == 'WAC':
            return 0.0

        return 0.0

    def star_max_usable_vmag(self) -> float:
        """Returns the maximum usable magnitude for stars in this observation.

        Returns:
            The maximum usable magnitude for stars in this observation.
        """

        # A non-positive exposure time is invalid (np.log would give -inf/nan);
        # fall back to the reference-exposure magnitude in that case.
        if self.detector == 'WAC':
            # This is based on star field image W1580760393 with texp 26 and clear filter.
            # This image was not useful beyond mag 10.7.
            # We don't try to compensate for non-clear filters.
            if self.texp <= 0.0:
                return 10.7
            return cast(float, 10.7 + np.log(self.texp / 26) / np.log(2.512))

        # This is based on star field image N1521881358 with texp 1 and clear filter.
        # This image was not useful beyond mag 10.7.
        # We don't try to compensate for non-clear filters.
        if self.texp <= 0.0:
            return 10.5
        return cast(float, 10.5 + np.log(self.texp) / np.log(2.512))

    @property
    def camera(self) -> str:
        """The camera that took this observation.

        Returns:
            The oops detector name: ``'NAC'`` or ``'WAC'``.
        """
        return str(self.detector)

    @property
    def shutter_mode(self) -> str | None:
        """The shutter mode this observation was taken in.

        Cassini ISS can expose both cameras at once; the label records that
        as ``'BOTSIM'``, against ``'NACONLY'`` or ``'WACONLY'`` for a single
        camera.  Two BOTSIM frames share one spacecraft attitude, so a
        consumer correcting that attitude can honor only one of them.

        Returns:
            The ``SHUTTER_MODE_ID`` label value, or None when the label
            carries none or carries a null.

        Raises:
            ValueError: if the label value is not text.  ``str()`` would
                serialize any object without complaint, and the result would
                pass downstream as a legible shutter mode.
        """
        if 'SHUTTER_MODE_ID' not in self.dict:
            return None
        value = self.dict['SHUTTER_MODE_ID']
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError(f'SHUTTER_MODE_ID is not text: {value!r}')
        return value

    def get_public_metadata(self) -> dict[str, Any]:
        """Returns the public metadata for Cassini ISS.

        The spacecraft clock counts are the label's start and stop counts, in seconds of
        the clock with the ticks as a fraction, and their exact mean; each is None when the
        label carries no counts.

        Returns:
            A dictionary containing the public metadata for Cassini ISS.
        """

        # The instrument LID encodes the camera as iss{n,w}a; guard against an
        # unexpected detector so a malformed LID never reaches a PDS4 label.
        if self.detector not in ('NAC', 'WAC'):
            raise ValueError(
                f"unexpected Cassini ISS detector {self.detector!r}; expected 'NAC' or 'WAC'"
            )

        return {
            'image_path': self.image_url,
            'image_name': self.abspath.name,
            'instrument_host_lid': 'urn:nasa:pds:context:instrument_host:spacecraft.co',
            'instrument_lid': f'urn:nasa:pds:context:instrument:iss{self.detector[0].lower()}a.co',
            'start_time_utc': et_to_utc(self.time[0]),
            'midtime_utc': et_to_utc(self.midtime),
            'end_time_utc': et_to_utc(self.time[1]),
            'start_time_et': self.time[0],
            'midtime_et': self.midtime,
            'end_time_et': self.time[1],
            **_published_sclk(
                self.dict.get('SPACECRAFT_CLOCK_START_COUNT', None),
                self.dict.get('SPACECRAFT_CLOCK_STOP_COUNT', None),
            ),
            'image_shape_xy': self.data_shape_uv,
            'camera': self.camera,
            'exposure_time': self.texp,
            'filters': [self.filter1, self.filter2],
            'sampling': self.sampling,
            'gain_mode': self.gain_mode,
            'description': self.dict.get('DESCRIPTION', None),
            'observation_id': self.dict.get('OBSERVATION_ID', None),
        }
