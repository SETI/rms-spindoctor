import re
from collections.abc import Mapping
from datetime import datetime
from fractions import Fraction
from typing import Any, cast

import host_cassini
import numpy as np
from filecache import FCPath
from oops.host import is_pds3_null
from oops.observation import Snapshot

from spindoctor.config import DEFAULT_CONFIG, IMAGE_LOGGER, Config, logged_section
from spindoctor.support.sclk import exposure_counts, fractional_count
from spindoctor.support.time import et_to_utc
from spindoctor.support.types import NDArrayAnyType, PathLike

from .obs_snapshot_inst import HostLabel, ObsSnapshotInst, host_input

# SCLK01_MODULI_82 and SCLK01_OFFSETS_82 of the Cassini spacecraft clock kernel,
# $OOPS_RESOURCES/SPICE/Cassini/SCLK/cas00172.tsc: whole seconds, then 1/256-second ticks.
_SCLK_MODULI = (4294967296, 256)
_SCLK_OFFSETS = (0, 0)

_SCLK_TICK_DIGITS = 3
"""Digits in the tick field of a spacecraft clock count as an image label writes it."""

_METADATA_PREFIX = 'cassini:'
"""The namespace prefix every published attribute carries."""

_LABEL_METADATA: tuple[tuple[str, str | None, int | None], ...] = (
    ('mission_phase_name', 'MISSION_PHASE_NAME', None),
    ('spacecraft_clock_count_partition', 'SPACECRAFT_CLOCK_CNT_PARTITION', None),
    ('spacecraft_clock_start_count', 'SPACECRAFT_CLOCK_START_COUNT', None),
    ('spacecraft_clock_stop_count', 'SPACECRAFT_CLOCK_STOP_COUNT', None),
    ('limitations', 'DESCRIPTION', None),
    ('antiblooming_state_flag', 'ANTIBLOOMING_STATE_FLAG', None),
    ('bias_strip_mean', 'BIAS_STRIP_MEAN', None),
    ('calibration_lamp_state_flag', 'CALIBRATION_LAMP_STATE_FLAG', None),
    ('command_file_name', 'COMMAND_FILE_NAME', None),
    ('command_sequence_number', 'COMMAND_SEQUENCE_NUMBER', None),
    ('dark_strip_mean', 'DARK_STRIP_MEAN', None),
    ('data_conversion_type', 'DATA_CONVERSION_TYPE', None),
    ('delayed_readout_flag', 'DELAYED_READOUT_FLAG', None),
    ('detector_temperature', 'DETECTOR_TEMPERATURE', None),
    ('electronics_bias', 'ELECTRONICS_BIAS', None),
    ('earth_received_start_time', 'EARTH_RECEIVED_START_TIME', None),
    ('earth_received_stop_time', 'EARTH_RECEIVED_STOP_TIME', None),
    ('expected_maximum_full_well', 'EXPECTED_MAXIMUM', 0),
    ('expected_maximum_DN_sat', 'EXPECTED_MAXIMUM', 1),
    ('expected_packets', 'EXPECTED_PACKETS', None),
    ('exposure_duration', 'EXPOSURE_DURATION', None),
    ('filter_name_1', 'FILTER_NAME', 0),
    ('filter_name_2', 'FILTER_NAME', 1),
    ('filter_temperature', 'FILTER_TEMPERATURE', None),
    ('flight_software_version_id', 'FLIGHT_SOFTWARE_VERSION_ID', None),
    ('gain_mode_id', 'GAIN_MODE_ID', None),
    ('ground_software_version_id', 'SOFTWARE_VERSION_ID', None),
    ('image_mid_time', 'IMAGE_MID_TIME', None),
    ('image_number', 'IMAGE_NUMBER', None),
    ('image_time', 'IMAGE_TIME', None),
    ('image_observation_type', 'IMAGE_OBSERVATION_TYPE', None),
    ('instrument_data_rate', 'INSTRUMENT_DATA_RATE', None),
    ('instrument_mode_id', 'INSTRUMENT_MODE_ID', None),
    ('inst_cmprs_type', 'INST_CMPRS_TYPE', None),
    ('inst_cmprs_param_malgo', 'INST_CMPRS_PARAM', 0),
    ('inst_cmprs_param_tb', 'INST_CMPRS_PARAM', 1),
    ('inst_cmprs_param_blocks', 'INST_CMPRS_PARAM', 2),
    ('inst_cmprs_param_quant', 'INST_CMPRS_PARAM', 3),
    ('inst_cmprs_rate_expected_bits', 'INST_CMPRS_RATE', 0),
    ('inst_cmprs_rate_actual_bits', 'INST_CMPRS_RATE', 1),
    ('inst_cmprs_ratio', 'INST_CMPRS_RATIO', None),
    ('light_flood_state_flag', 'LIGHT_FLOOD_STATE_FLAG', None),
    ('method_description', 'METHOD_DESC', None),
    ('missing_lines', 'MISSING_LINES', None),
    ('missing_packet_flag', 'MISSING_PACKET_FLAG', None),
    ('observation_id', 'OBSERVATION_ID', None),
    ('optics_temperature_front', 'OPTICS_TEMPERATURE', 0),
    ('optics_temperature_back', 'OPTICS_TEMPERATURE', 1),
    ('order_number', 'ORDER_NUMBER', None),
    ('parallel_clock_voltage_index', 'PARALLEL_CLOCK_VOLTAGE_INDEX', None),
    ('pds3_product_creation_time', 'PRODUCT_CREATION_TIME', None),
    ('pds3_product_version_type', 'PRODUCT_VERSION_TYPE', None),
    ('pds3_target_desc', 'TARGET_DESC', None),
    ('pds3_target_list', 'TARGET_LIST', None),
    ('pds3_target_name', 'TARGET_NAME', None),
    ('pre-pds_version_number', None, None),
    ('prepare_cycle_index', 'PREPARE_CYCLE_INDEX', None),
    ('readout_cycle_index', 'READOUT_CYCLE_INDEX', None),
    ('received_packets', 'RECEIVED_PACKETS', None),
    ('sensor_head_electronics_temperature', 'SENSOR_HEAD_ELEC_TEMPERATURE', None),
    ('sequence_id', 'SEQUENCE_ID', None),
    ('sequence_number', 'SEQUENCE_NUMBER', None),
    ('sequence_title', 'SEQUENCE_TITLE', None),
    ('shutter_mode_id', 'SHUTTER_MODE_ID', None),
    ('shutter_state_id', 'SHUTTER_STATE_ID', None),
    ('start_time_doy', 'START_TIME', None),
    ('stop_time_doy', 'STOP_TIME', None),
    ('telemetry_format_id', 'TELEMETRY_FORMAT_ID', None),
    ('valid_maximum_full_well', 'VALID_MAXIMUM', 0),
    ('valid_maximum_DN_sat', 'VALID_MAXIMUM', 1),
)
"""What the host publishes about a Cassini ISS image, and where each value comes from.

Each entry is an attribute of the Cassini data dictionary's ``ISS_Specific_Attributes``,
the PDS3 label keyword that states it, and, for a keyword holding several values, which
of them.  The attributes are the dictionary's whole set, in the order the dictionary
declares them, and each is published under that name behind ``_METADATA_PREFIX``.

The names are the dictionary's rather than the label's so that one navigation document
says the same thing whether it was built from a PDS3 volume or from a PDS4 source
bundle, and whether the image file was raw or calibrated.

The values are read from the host's ``dict``, which merges the PDS3 label and the VICAR
header under the PDS3 names, except the two raw-product keywords of
``_RAW_PRODUCT_KEYWORDS``, which are read from the header.  The host gives each value the
type the label parser gives it, and the two that a metadata document cannot hold are
written as text and a list (see :func:`_document_value`).  A keyword filling several
attributes is then split by position: ``EXPECTED_MAXIMUM``, ``FILTER_NAME``,
``INST_CMPRS_PARAM``, ``INST_CMPRS_RATE``, ``OPTICS_TEMPERATURE`` and ``VALID_MAXIMUM``
give their values in the label's own order, and each is published as the host gives it.
A keyword given as one value fills the first of its attributes and leaves the rest None
(see :func:`_element`).  Everything else keeps the form the host gives: a number stays a
number and text stays text.

``pre-pds_version_number`` is stated by no keyword, which is what its ``None`` keyword
records; the image's file name states it instead (see :func:`_version_number`).  Every
other attribute is a keyword the archive's PDS3 labels state.  The host leaves out a
value that no label gives, or that a label writes as a placeholder for a missing value,
such as ``N/A``, or ``-999`` for a temperature.  Such an attribute is published as None,
so that the published set is the same for every image.
"""

_RAW_PRODUCT_KEYWORDS = frozenset({'DESCRIPTION', 'PRODUCT_CREATION_TIME'})
"""The keywords whose attributes describe the raw image product, read from its header.

The data dictionary's ``limitations`` and ``pds3_product_creation_time`` are the raw
product's.  The VICAR header of a calibrated image keeps the raw product's values, while
its PDS3 label gives the calibrated product's own: the calibration's history and the
date it ran, such as 2019-06-09.  The host's merged ``dict`` takes the label's, so these
two are read from ``header_dict``, which the host keeps verbatim.
"""

_IMAGE_NAME_VERSION = re.compile(r'^[NW]\d{10}_(\d+)(?:_|\.|$)', re.IGNORECASE)
"""The version number in an image's file name: the segment after the image number.

``N1521598221_1.IMG`` and ``N1521598221_1_CALIB.IMG`` both state ``1``, so a raw file and
its calibrated counterpart publish the same version.
"""


def _version_number(image_name: str) -> int | None:
    """Return the internal version number an image's file name states.

    The PDS3 file name carries it and no label keyword does, which is why it is read
    from the name.  A name that does not carry one states no version.

    Parameters:
        image_name: Basename of the image file, such as ``N1521598221_1_CALIB.IMG``.

    Returns:
        The version number, or None when the name does not state one.
    """
    match = _IMAGE_NAME_VERSION.match(image_name)
    return None if match is None else int(match.group(1))


def _element(value: Any, index: int) -> Any:
    """Return the value one position of a label keyword states.

    A keyword stating several values fills the attribute at each position from the value
    written there.  A keyword stating a single value fills the first attribute with it
    and leaves the later ones None, because one value is a reading the label has and the
    later positions are readings it does not: ``OPTICS_TEMPERATURE`` given as one number
    is a front optics temperature and no rear one, which is what the two-value form says
    as well when its rear reading is missing, as for a camera with no rear sensor.

    Parameters:
        value: What the host gives for the keyword, or None when it gives nothing.
        index: Which position to take, counting from zero.

    Returns:
        The value at that position, as the host gives it, or None when nothing reaches
        that position.
    """
    if not isinstance(value, list | tuple):
        return value if index == 0 else None
    return value[index] if index < len(value) else None


def _document_value(label: Mapping[str, Any], keyword: str) -> Any:
    """Return a keyword's value from the host's metadata in a form a document can hold.

    The host gives each value the type the label parser gives it.  Two of those types
    have no form in a metadata document, and the parser keeps the label's own form of
    each beside the value: a time is a :class:`~datetime.datetime`, written as the
    label's text, which the parser keeps under the keyword with ``_fmt`` appended; and a
    set is written as a list in the label's order, which the parser keeps under the
    keyword with ``_list`` appended.  A time or a set that the host took from the VICAR
    header has no such text, because the header was not read by the label parser: a time
    is then written as the header writes it, ``YYYY-DDDTHH:MM:SS.sss`` without the final
    ``Z``, and a set as a sorted list.  Every other value is returned as it is.

    Parameters:
        label: The image's metadata, the host's ``dict``.
        keyword: The keyword to read.

    Returns:
        The value, or None when the metadata does not give the keyword.
    """
    value = label.get(keyword, None)
    if isinstance(value, datetime):
        return label.get(
            f'{keyword}_fmt', f'{value:%Y-%jT%H:%M:%S}.{value.microsecond // 1000:03d}'
        )
    if isinstance(value, set | frozenset):
        return label.get(f'{keyword}_list', sorted(value))
    return value


def _label_metadata(
    label: Mapping[str, Any], image_name: str, *, header: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Return the dictionary attributes an image states, under their own names.

    Every attribute ``_LABEL_METADATA`` names is published, so the block has the same
    keys for every image; one whose keyword the metadata lacks is None.  The keywords of
    ``_RAW_PRODUCT_KEYWORDS`` are read from the VICAR header when it gives them (see
    there), and every other keyword from the merged metadata.

    Parameters:
        label: The image's metadata, the host's ``dict``.
        image_name: Basename of the image file, which states the version number.
        header: The image's VICAR header, the host's ``header_dict``, or None when the
            host did not read it.

    Returns:
        The value of each attribute, keyed by that attribute behind
        ``_METADATA_PREFIX``, in the dictionary's own order.
    """
    metadata: dict[str, Any] = {}
    for attribute, keyword, element in _LABEL_METADATA:
        if keyword is None:
            value = _version_number(image_name)
        elif keyword in _RAW_PRODUCT_KEYWORDS and header is not None and keyword in header:
            # header_dict is verbatim, so it keeps the placeholders the host leaves out.
            value = None if is_pds3_null(header[keyword]) else header[keyword]
        elif element is None:
            value = _document_value(label, keyword)
        else:
            value = _element(_document_value(label, keyword), element)
        metadata[f'{_METADATA_PREFIX}{attribute}'] = value
    return metadata


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


class ObsCassiniISS(ObsSnapshotInst, HostLabel):
    """Implements an observation of a Cassini ISS image.

    This class provides specialized functionality for accessing and analyzing Cassini
    ISS image data.
    """

    model_source = 'spice'

    detector: str
    """The camera, ``'NAC'`` or ``'WAC'``."""

    filter1: str
    """The filter in the first filter wheel, as the label names it.

    Absent, like ``filter2``, unless the labels give both filters.
    """

    filter2: str
    """The filter in the second filter wheel, as the label names it."""

    gain_mode: int
    """The gain state, 0 to 3 for 215, 95, 29 and 12 electrons per DN; absent if unknown."""

    @staticmethod
    @logged_section('obs', 'LOAD IMAGE')
    def from_file(
        path: PathLike,
        *,
        config: Config | None = None,
        extfov_margin_vu: tuple[int, int] | None = None,
        label: PathLike | None = None,
        **kwargs: Any,
    ) -> 'ObsCassiniISS':
        """Creates an ObsCassiniISS from a Cassini ISS image file.

        The host reads the image's PDS3 label together with the image file (see
        :func:`~spindoctor.obs.obs_snapshot_inst.host_input`).

        Parameters:
            path: Path to the Cassini ISS image file.
            config: Configuration object to use. If None, uses the default configuration.
            extfov_margin_vu: Optional tuple that overrides the extended field of view margins
                found in the config.
            label: The image's PDS3 label, if known; otherwise the label beside the image.
            **kwargs: Additional keyword arguments:

                - fast_distortion: Whether to use a fast distortion model.

                - return_all_planets: Whether the host lists the kernels of every planet,
                  not only Jupiter or Saturn.

        Returns:
            An ObsCassiniISS object containing the image data and metadata.

        Raises:
            OSError, RuntimeError: If the SPICE kernels lack data the image needs, such as
                its pointing, which cspyce reports when the observation is first used.
            oops.host.HostError: If the host cannot read the image or its label.
            ValueError: If the configuration has no section for the image's camera.
        """

        config = config or DEFAULT_CONFIG
        logger = IMAGE_LOGGER

        fast_distortion = kwargs.get('fast_distortion', True)
        return_all_planets = kwargs.get('return_all_planets', True)

        logger.debug(f'Reading Cassini ISS image {path}')
        logger.debug(f'  Fast distortion: {fast_distortion}')
        logger.debug(f'  Return all planets: {return_all_planets}')
        # The host's stub types every result as a Snapshot or a tuple of Snapshots,
        # whatever select is; a call without select returns the one Snapshot.
        obs = cast(
            Snapshot,
            host_cassini.CassiniISS.from_file(
                host_input(path, label=label),
                fast_distortion=fast_distortion,
                return_all_planets=return_all_planets,
            ),
        )
        # The host inserts the camera and the image as subfields, which the Snapshot it
        # returns does not declare.
        detector = str(obs.subfields['detector']).lower()
        data: NDArrayAnyType = obs.subfields['data']
        fc_path = FCPath(path)
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
                extfov_margin_vu = extfov_margin_vu_entry[data.shape[0]]
            else:
                extfov_margin_vu = extfov_margin_vu_entry
        logger.debug(f'  Data shape: {data.shape}')
        logger.debug(f'  Extfov margin vu: {extfov_margin_vu}')
        # TODO This is slow when debug turned off
        logger.debug(f'  Data min: {np.min(data)}, max: {np.max(data)}')

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
            The ``SHUTTER_MODE_ID`` label value, or None when no label gives one.

        Raises:
            ValueError: if the label value is not text.  ``str()`` would
                serialize any object without complaint, and the result would
                pass downstream as a legible shutter mode.
        """
        if 'SHUTTER_MODE_ID' not in self.dict:
            return None
        value = self.dict['SHUTTER_MODE_ID']
        if not isinstance(value, str):
            raise ValueError(f'SHUTTER_MODE_ID is not text: {value!r}')
        return value

    def get_public_metadata(self) -> dict[str, Any]:
        """Returns the public metadata for Cassini ISS.

        The spacecraft clock counts are the host's ``spacecraft_clock_start_count`` and
        ``spacecraft_clock_stop_count``, in seconds of the clock with the ticks as a
        fraction, and their exact mean; each is None when no label gives the counts.  The
        filters are the host's ``filter1`` and ``filter2``, or none when the labels do not
        give both.

        ``label_metadata`` holds what the image's labels state about the exposure, under
        the names the Cassini data dictionary gives those quantities, so that the block
        reads the same whether the document was built from a PDS3 volume or a PDS4
        source bundle.  Every attribute is present on every image, and one the labels
        do not state is None.  ``_LABEL_METADATA`` lists them.

        Returns:
            A dictionary containing the public metadata for Cassini ISS.

        Raises:
            ValueError: If the detector is neither ``NAC`` nor ``WAC``.
        """

        # The instrument identifier encodes the camera as iss{n,w}a; guard against
        # an unexpected detector so a malformed identifier is never published.
        if self.detector not in ('NAC', 'WAC'):
            raise ValueError(
                f"unexpected Cassini ISS detector {self.detector!r}; expected 'NAC' or 'WAC'"
            )

        image_name = self.image_name
        return {
            'image_path': self.image_url,
            'image_name': image_name,
            'instrument_host_lid': 'urn:nasa:pds:context:instrument_host:spacecraft.co',
            'instrument_lid': f'urn:nasa:pds:context:instrument:iss{self.detector[0].lower()}a.co',
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
            'filters': (
                [self.subfields['filter1'], self.subfields['filter2']]
                if 'filter1' in self.subfields
                else []
            ),
            'label_metadata': _label_metadata(
                self.dict, image_name, header=self.subfields.get('header_dict')
            ),
        }
