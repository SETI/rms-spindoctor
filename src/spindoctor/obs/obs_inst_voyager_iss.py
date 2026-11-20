from collections.abc import Mapping
from fractions import Fraction
from typing import Any, cast

import host_voyager
import numpy as np

from spindoctor.config import DEFAULT_CONFIG, IMAGE_LOGGER, Config, logged_section
from spindoctor.support.sclk import exposure_counts, fractional_count
from spindoctor.support.time import et_to_utc
from spindoctor.support.types import NDArrayAnyType, NDArrayFloatType, PathLike

from .obs_snapshot_inst import HostLabel, ObsSnapshotInst, host_input

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


def _if_data(data: NDArrayAnyType, subfields: Mapping[str, Any]) -> NDArrayFloatType:
    """Return a calibrated Voyager image's data in units of I/F.

    The host gives the factor that converts a calibrated product's data to I/F as the
    ``if_scale`` subfield, from the VICAR header.  A Voyager 1 image of Saturn is then
    brightened by ``_V1_SATURN_IF_CORRECTION``.

    Parameters:
        data: The image's data, as the host read it.
        subfields: The host's subfields of the image: ``if_scale``, ``spacecraft`` (1 or
            2) and ``planet``, the planet of the encounter.

    Returns:
        The data in units of I/F.

    Raises:
        ValueError: If the host gives no ``if_scale``, as for a RAW or CLEANED product,
            whose VICAR header states no I/F factor.
    """
    if 'if_scale' not in subfields:
        raise ValueError(
            'the VICAR header gives no I/F factor; Voyager ISS images are navigated from '
            'calibrated GEOMED products'
        )
    if_data: NDArrayFloatType = data * subfields['if_scale']
    if subfields['spacecraft'] == 1 and subfields['planet'] == 'SATURN':
        if_data = if_data * _V1_SATURN_IF_CORRECTION
    return if_data


class ObsVoyagerISS(ObsSnapshotInst, HostLabel):
    """Implements an observation of a Voyager ISS image.

    This class provides specialized functionality for accessing and analyzing Voyager
    ISS image data.
    """

    model_source = 'spice'

    detector: str
    """The camera, ``'NAC'`` or ``'WAC'``."""

    filter: str
    """The filter, as the PDS3 label names it; absent when no label gives it."""

    planet: str
    """The planet of the encounter the image was taken in, such as ``'SATURN'``."""

    spacecraft: int
    """Which Voyager took the image: 1 or 2."""

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

        The host reads the PDS3 label beside the image together with the image file (see
        :func:`~spindoctor.obs.obs_snapshot_inst.host_input`), and the data are converted
        to I/F.

        Parameters:
            path: Path to the Voyager ISS image file.
            config: Configuration object to use. If None, uses the default configuration.
            extfov_margin_vu: Optional tuple that overrides the extended field of view margins
                found in the config.
            **_kwargs: Additional keyword arguments (none for this instrument).

        Returns:
            An ObsVoyagerISS object containing the image data and metadata.

        Raises:
            oops.OopsSpiceError: If the SPICE kernels cannot supply the image's pointing.
            oops.host.HostError: If the host cannot read the image or its label.
            ValueError: If the image is not a calibrated product, so that its VICAR
                header gives no I/F factor.
        """

        config = config or DEFAULT_CONFIG
        logger = IMAGE_LOGGER

        logger.debug(f'Reading Voyager ISS image {path}')
        obs = host_voyager.VoyagerISS.from_file(host_input(path))

        inst_config = config.category('voyager_iss')

        data = _if_data(obs.subfields['data'], obs.subfields)
        obs.insert_subfield('data', data)
        logger.debug(
            f'  Voyager {obs.subfields["spacecraft"]} at {obs.subfields["planet"]}, '
            f'I/F factor {obs.subfields["if_scale"]}'
        )

        if extfov_margin_vu is None:
            if isinstance(inst_config.extfov_margin_vu, dict):
                extfov_margin_vu = inst_config.extfov_margin_vu[data.shape[0]]
            else:
                extfov_margin_vu = inst_config.extfov_margin_vu
        logger.debug(f'  Data shape: {data.shape}')
        logger.debug(f'  Extfov margin vu: {extfov_margin_vu}')
        logger.debug(f'  Data min: {np.min(data)}, max: {np.max(data)}')

        new_obs = ObsVoyagerISS(obs, config=config, extfov_margin_vu=extfov_margin_vu)
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
        aperture D = 0.19 m) by collecting-area, with a detector-sensitivity
        term for the Voyager vidicon (roughly 2 mag less sensitive than a
        CCD).  These are nominal optics values; the terms are approximate and
        pending calibration against real Voyager star fields.

            anchor = 10.5 + 5*log10(D / 0.19) + detector_term

        Per camera (apertures from the Voyager ISS optics; vidicon detector):

            NAC: 10.5 + 5*log10(0.176/0.19) - 2.0 (vidicon) ~= 8.3
            WAC: 10.5 + 5*log10(0.057/0.19) - 2.0 (vidicon) ~= 5.9

        Returns:
            The maximum usable magnitude for stars in this observation.
        """

        # Anchor (limiting mag at texp = 1 s) derived above; rounded to 0.1.
        anchor = 5.9 if self.detector == 'WAC' else 8.3
        if self.texp <= 0:
            return anchor
        return cast(float, anchor + np.log(self.texp) / np.log(2.512))

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

        The host's ``spacecraft`` subfield, which every image has.  SPICE frame, kernel
        and clock identifiers are all keyed per spacecraft, so consumers that must name
        one derive it here.

        Returns:
            ``'1'`` for Voyager 1 or ``'2'`` for Voyager 2.
        """
        return str(self.spacecraft)

    def get_public_metadata(self) -> dict[str, Any]:
        """Returns the public metadata for Voyager ISS.

        The spacecraft clock counts are the host's ``spacecraft_clock_start_count`` and
        ``spacecraft_clock_stop_count``, in the clock's leading units.  Each is None when
        no label gives it, and the midtime count is always None.  The filters are the
        host's ``filter``, or none when no label gives it.

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
            **_published_sclk(
                self.subfields.get('spacecraft_clock_start_count'),
                self.subfields.get('spacecraft_clock_stop_count'),
            ),
            'image_shape_xy': self.data_shape_uv,
            'camera': self.camera,
            'exposure_time': self.texp,
            'filters': [self.subfields['filter']] if 'filter' in self.subfields else [],
        }
