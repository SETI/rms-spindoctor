import math
import re
from fractions import Fraction
from typing import Any

import numpy as np
from filecache import FCPath

from spindoctor.config import DEFAULT_CONFIG, IMAGE_LOGGER, Config, logged_section
from spindoctor.support.sclk import exposure_counts, fractional_count, pds3_label_clock_counts
from spindoctor.support.time import et_to_utc
from spindoctor.support.types import PathLike

from .obs_snapshot_inst import HostLabel, ObsSnapshotInst

_READOUT_MODE = 'ALL'
"""The one readout mode of the star photometry's mode tables: a Voyager camera's star
limit and star width do not depend on the readout."""

_LAB03_GAIN_RE = re.compile(r'\b(?P<gain>LO|HI) GAIN\b')
"""The gain state in the VICAR label's ``LAB03`` record, as in ``LO GAIN``."""

_GAIN_STATES = {'LO': 'LOW', 'HI': 'HIGH'}
"""The gain state for each ``LAB03`` spelling, spelled the way the PDS3 label's
``GAIN_MODE_ID`` spells it."""

# SCLK01_MODULI_31/_32 and SCLK01_OFFSETS_31/_32 of the Voyager 1 and 2 spacecraft clock
# kernels, $OOPS_RESOURCES/SPICE/Voyager/SCLK/vg100042.tsc and vg200041.tsc, which give both
# clocks the same three fields: a leading count of 48-minute units, the first five digits
# of an image number; the 60 frames of 48 seconds in one of them; and the 800 lines of
# 60 milliseconds in a frame, counted from 1.
_SCLK_MODULI = (65536, 60, 800)
_SCLK_OFFSETS = (0, 0, 1)


def _sclk_count(count: str) -> Fraction:
    """Return a Voyager spacecraft clock count as a number of the clock's leading units.

    A count is ``LEADING:FRAME:LINE``: the leading field counts 48-minute units, the frame
    field the 60 frames of 48 seconds in one, and the line field the 800 lines of 60
    milliseconds in a frame, counted from 1.  So ``34461:39:672`` is
    ``34461 + 39 / 60 + (672 - 1) / (60 * 800)``.

    Parameters:
        count: The count as text.

    Returns:
        The count in the clock's leading units, exactly, the frame and the line as a
        fraction of one.
    """
    return fractional_count(
        tuple(int(field) for field in count.strip().split(':')), _SCLK_MODULI, _SCLK_OFFSETS
    )


def _published_sclk(start: str | None, stop: str | None) -> dict[str, float | None]:
    """Return the spacecraft clock counts Voyager ISS publishes for one image.

    A Voyager label's start count lies near the shutter opening, but its stop count is
    the count of the frame the image was read out in, with the line field 001, which can
    come minutes after the shutter closed.  The two do not bracket the exposure, so no
    midtime count is published.

    Parameters:
        start: The label's ``SPACECRAFT_CLOCK_START_COUNT``, or None when it carries none.
        stop: The label's ``SPACECRAFT_CLOCK_STOP_COUNT``, or None when it carries none.

    Returns:
        ``start_time_sclk`` and ``end_time_sclk``, the two counts in the clock's leading
        units, each None when the label does not carry it, and ``midtime_sclk``, always
        None.
    """
    return exposure_counts(
        None if start is None else _sclk_count(start),
        None if stop is None else _sclk_count(stop),
        bracketed=False,
    )


# The Voyager 1 SEDR/PDS3 calibration pipeline always computed I/F as if the
# image had been taken at Jupiter's heliocentric distance, so V1 @ Saturn
# images come out too dim by (~9.54 / 5.20)**2.  Multiply by this factor to
# recover Saturn-correct I/F.  V1 only visited Jupiter and Saturn, and V2 was
# calibrated correctly at each of its encounters, so this is the only case.
_V1_SATURN_IF_CORRECTION = 3.345

# The I/F scaling factor lives in the GEOMED VICAR LABEL3 string as a
# trailing numeric value after this fixed phrase.
_IF_LABEL3_PHRASE = 'FOR (I/F)*10000., MULTIPLY DN VALUE BY'


def _voyager_spacecraft_digit(lab02: Any) -> str:
    """Extract and validate the Voyager spacecraft digit from a LAB02 record.

    The Voyager VICAR ``LAB02`` record is a fixed-format string whose 5th
    character (index 4) is the spacecraft digit, ``'1'`` for Voyager 1 or
    ``'2'`` for Voyager 2.

    Parameters:
        lab02: The raw ``LAB02`` value from the image label.

    Returns:
        The spacecraft digit, either ``'1'`` or ``'2'``.

    Raises:
        ValueError: If ``lab02`` is not a string of length at least 5, or
            its 5th character is not ``'1'`` or ``'2'``.
    """

    if not isinstance(lab02, str) or len(lab02) < 5:
        raise ValueError(f'Unexpected Voyager LAB02 format: {lab02!r}')
    digit = lab02[4]
    if digit not in ('1', '2'):
        raise ValueError(
            f'Unexpected Voyager spacecraft id in LAB02[4]: {digit!r} (expected 1 or 2)'
        )
    return digit


def _camera_name(spacecraft: str, detector: str) -> str:
    """Return the name of a Voyager camera's block inside the ``voyager_iss`` section.

    Parameters:
        spacecraft: The spacecraft digit, ``'1'`` or ``'2'``.
        detector: The oops detector name, ``'NAC'`` or ``'WAC'``.

    Returns:
        ``vgiss_1_nac``, ``vgiss_1_wac``, ``vgiss_2_nac``, or ``vgiss_2_wac``.
    """
    return f'vgiss_{spacecraft}_{detector.lower()}'


def _voyager_if_factor(label3: Any) -> float:
    """Parse the I/F scaling factor from a Voyager GEOMED LABEL3 string.

    Parameters:
        label3: The raw ``LABEL3`` value from the image label.

    Returns:
        The numeric I/F scaling factor.

    Raises:
        ValueError: If ``label3`` is not a string, does not contain the
            expected fixed phrase, or its numeric remainder cannot be parsed.
    """

    if not isinstance(label3, str) or _IF_LABEL3_PHRASE not in label3:
        raise ValueError(f'Unexpected Voyager LABEL3 format: {label3!r}')
    remainder = label3.replace(_IF_LABEL3_PHRASE, '').strip()
    try:
        return float(remainder)
    except ValueError:
        raise ValueError(f'Unexpected Voyager LABEL3 format: {label3!r}') from None


class ObsVoyagerISS(ObsSnapshotInst, HostLabel):
    """Implements an observation of a Voyager ISS image.

    This class provides specialized functionality for accessing and analyzing Voyager
    ISS image data.
    """

    model_source = 'spice'

    detector: str
    """The camera, ``'NAC'`` or ``'WAC'``."""

    filter: str
    """The filter, as the label names it."""

    planet: str
    """The planet of the encounter the image was taken in."""

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
    ) -> 'ObsVoyagerISS':
        """Creates an ObsVoyagerISS from a Voyager ISS image file.

        Parameters:
            path: Path to the Voyager ISS image file.
            config: Configuration object to use. If None, uses the default configuration.
            extfov_margin_vu: Optional tuple that overrides the extended field of view margins
                found in the config.
            **_kwargs: Additional keyword arguments (none for this instrument).

        Returns:
            An ObsVoyagerISS object containing the image data and metadata.
        """

        import oops.hosts.voyager.iss

        config = config or DEFAULT_CONFIG
        logger = IMAGE_LOGGER

        logger.debug(f'Reading Voyager ISS image {path}')
        # The host module publishes no type information for from_file.
        obs = oops.hosts.voyager.iss.from_file(path)  # type: ignore[no-untyped-call]
        fc_path = FCPath(path)

        factor = _voyager_if_factor(obs.dict.get('LABEL3', None))
        obs.data = obs.data * factor / 10000

        spacecraft = _voyager_spacecraft_digit(obs.dict.get('LAB02', None))
        camera = config.camera_block('voyager_iss', _camera_name(spacecraft, str(obs.detector)))
        if spacecraft == '1' and obs.planet.upper() == 'SATURN':
            obs.data = obs.data * _V1_SATURN_IF_CORRECTION
            logger.debug(
                f'  Applied Voyager 1 @ Saturn I/F correction: {_V1_SATURN_IF_CORRECTION:.4f}x'
            )
        # TODO Calibrate once oops.hosts is fixed.

        if extfov_margin_vu is None:
            extfov_margin_vu = camera.extfov_margin_for(obs.data.shape[0])
        logger.debug(f'  Data shape: {obs.data.shape}')
        logger.debug(f'  Extfov margin vu: {extfov_margin_vu}')
        logger.debug(f'  Data min: {np.min(obs.data)}, max: {np.max(obs.data)}')

        new_obs = ObsVoyagerISS(obs, config=config, extfov_margin_vu=extfov_margin_vu)
        new_obs._camera_block = camera
        # The spacecraft clock counts are in the PDS3 label, which the VICAR label the
        # observation keeps does not carry.
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
    def _star_photometry_path(self) -> str:
        """The key path of this camera's star photometry, which a message gives."""
        return (
            f'voyager_iss.{_camera_name(self.spacecraft_digit, str(self.detector))}.star_photometry'
        )

    def star_max_usable_vmag(self) -> float:
        """Returns the faintest camera magnitude of a catalog star the star model uses.

        The limit is photometric, from the camera block's ``star_photometry`` (see
        :class:`~spindoctor.config.sections_camera.StarPhotometry`), whose rate is in
        raw DN, because the Voyager calibration works in DN and does not convert to
        electrons.
        The filter is the observation's filter, the gain state is read from the VICAR
        label's ``LAB03`` record (``LO GAIN`` or ``HI GAIN``, read as ``LOW`` or
        ``HIGH``), and the mode tables have the one mode ``ALL``.

        Returns:
            The limiting magnitude, or ``-inf`` for a frame whose exposure is not
            positive.

        Raises:
            ValueError: If the label does not give a gain state, or the block has no
                ``star_photometry``, or its tables leave out the frame's filter or gain
                state.
        """
        if self.texp <= 0.0:
            return -math.inf
        match = _LAB03_GAIN_RE.search(str(self.dict.get('LAB03', '')))
        if match is None:
            raise ValueError(f'{self.detector} label does not give a gain state in LAB03')
        return self._photometric_star_limit(
            texp=self.texp,
            where=self._star_photometry_path,
            gain_state=_GAIN_STATES[match.group('gain')],
            mode=_READOUT_MODE,
            filter_name=self.filter,
        )

    def star_psf_sigma(self) -> float:
        """Returns the Gaussian sigma, in pixels, of a star in this observation.

        Returns:
            The camera block's ``star_photometry.psf_sigma`` entry ``ALL``.

        Raises:
            ValueError: If the block has no ``star_photometry``, or its ``psf_sigma`` has no
                entry ``ALL``.
        """
        return self._photometric_star_sigma(where=self._star_photometry_path, mode=_READOUT_MODE)

    def star_color_term(self) -> float:
        """Returns how a star's color changes its magnitude in this camera's band.

        Returns:
            The camera block's ``star_photometry.color_term``, zero when the block leaves it
            out.

        Raises:
            ValueError: If the block has no ``star_photometry``.
        """
        return self._star_photometry(self._star_photometry_path).color_term

    @property
    def camera(self) -> str:
        """The camera that took this observation.

        Returns:
            The oops detector name: ``'NAC'`` or ``'WAC'``.
        """
        return str(self.detector)

    @property
    def spacecraft_digit(self) -> str:
        """Which Voyager took this observation.

        Read from the VICAR ``LAB02`` record, whose 5th character is the
        spacecraft digit.  SPICE frame, kernel and clock identifiers are all
        keyed per spacecraft, so consumers that must name one derive it here.

        Returns:
            ``'1'`` for Voyager 1 or ``'2'`` for Voyager 2.

        Raises:
            ValueError: If ``LAB02`` is missing, too short, or names neither
                spacecraft.
        """
        return _voyager_spacecraft_digit(self.dict.get('LAB02', None))

    def get_public_metadata(self) -> dict[str, Any]:
        """Returns the public metadata for Voyager ISS.

        The spacecraft clock counts are those of the PDS3 label beside the image, read
        when the image was loaded, in the clock's leading units; each is None when the
        label carries none, and the midtime count is always None.

        Returns:
            A dictionary containing the public metadata for Voyager ISS.
        """

        spacecraft = self.spacecraft_digit

        # The instrument LID and `camera` field encode the camera as iss{n,w};
        # guard against an unexpected detector so a malformed LID never reaches
        # a PDS4 label.
        if self.detector not in ('NAC', 'WAC'):
            raise ValueError(
                f"unexpected Voyager ISS detector {self.detector!r}; expected 'NAC' or 'WAC'"
            )

        return {
            'image_path': self.image_url,
            'image_name': self.image_name,
            'instrument_host_lid': (
                f'urn:nasa:pds:context:instrument_host:spacecraft.vg{spacecraft}'
            ),
            'instrument_lid': (
                f'urn:nasa:pds:context:instrument:vg{spacecraft}.iss{self.detector[0].lower()}'
            ),
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
            'filters': [self.filter],
        }
