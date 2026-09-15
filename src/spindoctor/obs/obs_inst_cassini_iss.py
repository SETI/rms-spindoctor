from collections.abc import Mapping
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

# Seven of the dictionary's seventy ISS_Specific_Attributes are left out of this table
# (#684).  Six are already in the observation block, as the label states them:
# limitations is DESCRIPTION (description), filter_name_1 and filter_name_2 are
# FILTER_NAME (filters), instrument_mode_id is INSTRUMENT_MODE_ID (sampling),
# observation_id is OBSERVATION_ID, and shutter_mode_id is SHUTTER_MODE_ID
# (shutter_mode).  pre-pds_version_number is in the file name and not in the label.
# image_number is IMAGE_NUMBER, the seconds of the clock at shutter close, as the
# archive's own PDS4 labels have it; the dictionary's wording, a value obtained from the
# start count, differs from that for any exposure that spans a second.
_LABEL_FACTS: tuple[tuple[str, str | tuple[str, ...]], ...] = (
    ('MISSION_PHASE_NAME', 'mission_phase_name'),
    ('SPACECRAFT_CLOCK_CNT_PARTITION', 'spacecraft_clock_count_partition'),
    ('SPACECRAFT_CLOCK_START_COUNT', 'spacecraft_clock_start_count'),
    ('SPACECRAFT_CLOCK_STOP_COUNT', 'spacecraft_clock_stop_count'),
    ('ANTIBLOOMING_STATE_FLAG', 'antiblooming_state_flag'),
    ('BIAS_STRIP_MEAN', 'bias_strip_mean'),
    ('CALIBRATION_LAMP_STATE_FLAG', 'calibration_lamp_state_flag'),
    ('COMMAND_FILE_NAME', 'command_file_name'),
    ('COMMAND_SEQUENCE_NUMBER', 'command_sequence_number'),
    ('DARK_STRIP_MEAN', 'dark_strip_mean'),
    ('DATA_CONVERSION_TYPE', 'data_conversion_type'),
    ('DELAYED_READOUT_FLAG', 'delayed_readout_flag'),
    ('DETECTOR_TEMPERATURE', 'detector_temperature'),
    ('ELECTRONICS_BIAS', 'electronics_bias'),
    ('EARTH_RECEIVED_START_TIME', 'earth_received_start_time'),
    ('EARTH_RECEIVED_STOP_TIME', 'earth_received_stop_time'),
    ('EXPECTED_MAXIMUM', ('expected_maximum_full_well', 'expected_maximum_DN_sat')),
    ('EXPECTED_PACKETS', 'expected_packets'),
    ('EXPOSURE_DURATION', 'exposure_duration'),
    ('FILTER_TEMPERATURE', 'filter_temperature'),
    ('FLIGHT_SOFTWARE_VERSION_ID', 'flight_software_version_id'),
    ('GAIN_MODE_ID', 'gain_mode_id'),
    ('SOFTWARE_VERSION_ID', 'ground_software_version_id'),
    ('IMAGE_MID_TIME', 'image_mid_time'),
    ('IMAGE_NUMBER', 'image_number'),
    ('IMAGE_TIME', 'image_time'),
    ('IMAGE_OBSERVATION_TYPE', 'image_observation_type'),
    ('INSTRUMENT_DATA_RATE', 'instrument_data_rate'),
    ('INST_CMPRS_TYPE', 'inst_cmprs_type'),
    (
        'INST_CMPRS_PARAM',
        (
            'inst_cmprs_param_malgo',
            'inst_cmprs_param_tb',
            'inst_cmprs_param_blocks',
            'inst_cmprs_param_quant',
        ),
    ),
    ('INST_CMPRS_RATE', ('inst_cmprs_rate_expected_bits', 'inst_cmprs_rate_actual_bits')),
    ('INST_CMPRS_RATIO', 'inst_cmprs_ratio'),
    ('LIGHT_FLOOD_STATE_FLAG', 'light_flood_state_flag'),
    ('METHOD_DESC', 'method_description'),
    ('MISSING_LINES', 'missing_lines'),
    ('MISSING_PACKET_FLAG', 'missing_packet_flag'),
    ('OPTICS_TEMPERATURE', ('optics_temperature_front', 'optics_temperature_back')),
    ('ORDER_NUMBER', 'order_number'),
    ('PARALLEL_CLOCK_VOLTAGE_INDEX', 'parallel_clock_voltage_index'),
    ('PRODUCT_CREATION_TIME', 'pds3_product_creation_time'),
    ('PRODUCT_VERSION_TYPE', 'pds3_product_version_type'),
    ('TARGET_DESC', 'pds3_target_desc'),
    ('TARGET_LIST', 'pds3_target_list'),
    ('TARGET_NAME', 'pds3_target_name'),
    ('PREPARE_CYCLE_INDEX', 'prepare_cycle_index'),
    ('READOUT_CYCLE_INDEX', 'readout_cycle_index'),
    ('RECEIVED_PACKETS', 'received_packets'),
    ('SENSOR_HEAD_ELEC_TEMPERATURE', 'sensor_head_electronics_temperature'),
    ('SEQUENCE_ID', 'sequence_id'),
    ('SEQUENCE_NUMBER', 'sequence_number'),
    ('SEQUENCE_TITLE', 'sequence_title'),
    ('SHUTTER_STATE_ID', 'shutter_state_id'),
    ('START_TIME', 'start_time_doy'),
    ('STOP_TIME', 'stop_time_doy'),
    ('TELEMETRY_FORMAT_ID', 'telemetry_format_id'),
    ('VALID_MAXIMUM', ('valid_maximum_full_well', 'valid_maximum_DN_sat')),
)
"""The label facts the host publishes: each keyword, with the name it is published under.

A keyword whose value is a sequence carries the names of its elements, in order.  Each
name is the attribute of the Cassini PDS4 dictionary's ``ISS_Specific_Attributes``
(``PDS4_CASSINI_1O00_1800``) that the value is, with ``-`` written as ``_``, and the
table is in the dictionary's order.
"""


def _label_facts(label: Mapping[str, Any]) -> dict[str, Any]:
    """Return the facts an image's label states, under their Cassini dictionary names.

    Each value is the label's own, as the label states it: a number stays a number, text
    stays text, and a time is the label's own spelling of it.  A keyword whose value is a
    sequence is split into the attributes its elements are, in order.  A keyword the label
    lacks is published as None, and so is each element of a sequence it lacks.

    Parameters:
        label: The image's VICAR label items.

    Returns:
        The facts ``_LABEL_FACTS`` names, keyed by attribute name, in its order.

    Raises:
        ValueError: If a sequence keyword holds a different number of elements than the
            attributes it is split into.
    """
    facts: dict[str, Any] = {}
    for keyword, names in _LABEL_FACTS:
        value = label.get(keyword, None)
        if isinstance(names, str):
            facts[names] = value
        elif value is None:
            facts.update(dict.fromkeys(names))
        else:
            facts.update(zip(names, value, strict=True))
    return facts


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
        the clock with the ticks as a fraction, and their exact mean; each is None when
        the label carries no counts.

        After these come the facts the image's label states that an attribute of the
        Cassini PDS4 dictionary's ``ISS_Specific_Attributes`` names and the observation
        block does not already hold, each under that attribute's name and as the label
        states it: a number stays a number, text stays text, and a time is the label's own
        spelling.  A keyword whose value is a sequence (``EXPECTED_MAXIMUM``,
        ``INST_CMPRS_PARAM``, ``INST_CMPRS_RATE``, ``OPTICS_TEMPERATURE``,
        ``VALID_MAXIMUM``) is split into the attributes its elements are, in order, and a
        keyword the label lacks is published as None.  ``_LABEL_FACTS`` lists each keyword
        and the name it is published under.

        Returns:
            A dictionary containing the public metadata for Cassini ISS.

        Raises:
            ValueError: If the detector is neither ``NAC`` nor ``WAC``, or if a sequence
                keyword holds a different number of elements than the attributes it is
                split into.
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
            **_label_facts(self.dict),
        }
