import math
from collections.abc import Mapping, Sequence
from fractions import Fraction
from typing import Any, Literal

import numpy as np

from spindoctor.config import DEFAULT_CONFIG, IMAGE_LOGGER, Config, logged_section
from spindoctor.support.sclk import exposure_counts, fractional_count
from spindoctor.support.time import et_to_utc
from spindoctor.support.types import NDArrayAnyType, PathLike

from .obs_snapshot_inst import HostLabel, ObsSnapshotInst

# SCLK01_MODULI_77 and SCLK01_OFFSETS_77 of the Galileo spacecraft clock kernel,
# $OOPS_RESOURCES/SPICE/Galileo/SCLK/mk00062a.tsc: the RIM count, then its 91 MOD91
# counts, each of 10 MOD10 counts, each of 8 MOD8 counts.
_SCLK_MODULI = (16777215, 91, 10, 8)
_SCLK_OFFSETS = (0, 0, 0, 0)

_SCLK_FIELDS = ('RIM', 'MOD91', 'MOD10', 'MOD8')
"""The VICAR label items holding the four fields of an image's frame count."""

_STAR_PHOTOMETRY = 'galileo_ssi.star_photometry'
"""The key path of the camera block's star photometry."""

GalileoSSIReadoutMode = Literal['FULL', 'SUMMED']
"""A Galileo SSI readout mode, as the star photometry's mode tables key it: ``FULL`` for a
full 800 x 800 frame, ``SUMMED`` for a frame read out in 2x2 summation, 400 x 400."""

_GAIN_STATES = {1: '400K', 2: '100K', 3: '40K', 4: '10K'}
"""The gain state for each value of the VICAR label's ``GAIN`` item, spelled the way the
PDS3 label's ``GAIN_MODE_ID`` spells it."""

_READOUT_MODES: dict[str, GalileoSSIReadoutMode] = {
    'AI8': 'SUMMED',
    'HIS': 'SUMMED',
    'HCA': 'FULL',
    'HCJ': 'FULL',
    'HCM': 'FULL',
    'HIM': 'FULL',
    'HMA': 'FULL',
    'IM4': 'FULL',
    'IM8': 'FULL',
    'XCM': 'FULL',
}
"""The readout mode of each telemetry format the VICAR label's ``TLMFMT`` item gives: every
format the Galileo SSI volumes use.  The image host gives the two summed formats, AI8 and
HIS, a 400 x 400 field of view and every other format the full 800 x 800 one."""


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


class ObsGalileoSSI(ObsSnapshotInst, HostLabel):
    """Implements an observation of a Galileo SSI image.

    This class provides specialized functionality for accessing and analyzing Galileo
    SSI image data.
    """

    model_source = 'spice'

    filter: str
    """The filter, as the label names it."""

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
        # Galileo SSI is read through oops.hosts rather than rms-host-galileo, whose
        # GalileoSSI fails on a VICAR label whose observation ID (the PA item) does not begin with
        # an orbit number (GO_0003's OPNAV01A frames); read through the detached
        # .LBL instead, it would publish the label as the image path
        # (SETI/rms-host-galileo#2).
        obs = oops.hosts.galileo.ssi.from_file(path, full_fov=True)
        # The host inserts the image as the data subfield, which the Snapshot it returns
        # does not declare.
        data: NDArrayAnyType = obs.subfields['data']

        camera = config.camera_block('galileo_ssi')
        if extfov_margin_vu is None:
            extfov_margin_vu = camera.extfov_margin_for(data.shape[0])
        logger.debug(f'  Data shape: {data.shape}')
        logger.debug(f'  Extfov margin vu: {extfov_margin_vu}')
        logger.debug(f'  Data min: {np.min(data)}, max: {np.max(data)}')

        new_obs = ObsGalileoSSI(obs, config=config, extfov_margin_vu=extfov_margin_vu)
        new_obs._camera_block = camera
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
        """Returns the faintest V magnitude of a catalog star the star model uses.

        The limit is photometric, from the camera block's ``star_photometry`` (see
        :class:`~spindoctor.config.sections_camera.StarPhotometry`), whose rate counts
        electrons.  The filter is the label's ``FILTER_NAME``, the gain state the VICAR
        label's ``GAIN`` item read as ``400K``, ``100K``, ``40K``, or ``10K``, and the
        readout mode :attr:`readout_mode`.

        Returns:
            The limiting magnitude, or ``-inf`` for a frame whose exposure is not
            positive.

        Raises:
            ValueError: If the label does not give a gain state or telemetry format the
                camera has, or the block has no ``star_photometry``, or its tables leave
                out the frame's readout mode, filter, or gain state.
        """
        if self.texp <= 0.0:
            return -math.inf
        mode = self.readout_mode
        gain_item = self.dict.get('GAIN', None)
        if gain_item not in _GAIN_STATES:
            raise ValueError(f'SSI label does not give a known gain state: GAIN is {gain_item!r}')
        return self._photometric_star_limit(
            texp=self.texp,
            where=_STAR_PHOTOMETRY,
            gain_state=_GAIN_STATES[gain_item],
            mode=mode,
            filter_name=self.filter,
        )

    @property
    def readout_mode(self) -> GalileoSSIReadoutMode:
        """The readout mode this observation was read out in.

        Returns:
            ``'SUMMED'`` for a frame read out in 2x2 summation, ``'FULL'`` for a full
            frame, from the telemetry format the VICAR label's ``TLMFMT`` item gives.

        Raises:
            ValueError: If the label does not give a telemetry format the camera has.
        """
        format_item = self.dict.get('TLMFMT', None)
        if format_item not in _READOUT_MODES:
            raise ValueError(
                f'SSI label does not give a known telemetry format: TLMFMT is {format_item!r}'
            )
        return _READOUT_MODES[format_item]

    def star_psf_sigma(self) -> float:
        """Returns the Gaussian sigma, in pixels, of a star in this observation.

        Returns:
            The camera block's ``star_photometry.psf_sigma`` for the frame's readout mode.

        Raises:
            ValueError: If the label does not give a telemetry format the camera has, or
                the block has no ``star_photometry``, or its ``psf_sigma`` leaves out the
                frame's readout mode.
        """
        return self._photometric_star_sigma(where=_STAR_PHOTOMETRY, mode=self.readout_mode)

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
            'image_name': self.image_name,
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
