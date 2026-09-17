import argparse
from collections.abc import Generator
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar, cast

from filecache import FCPath, FileCache

from spindoctor.config import Config
from spindoctor.support.misc import safe_lstrip_zero
from spindoctor.support.time import et_to_pds4_utc, pds4_utc_midpoint

from .dataset import ImageFile, ImageFiles, Pds4Pass
from .dataset_pds3 import DataSetPDS3

_PDS4_TIME_DIGITS = 3
"""The decimals of a second a data label writes its exposure's times to.

A millisecond, the precision a Cassini image's start and stop are recorded to in its
PDS3 label and index.  Whole seconds, which the reference bundle writes for its mosaics,
would state an exposure of a few milliseconds as a window of one or two seconds.
"""


class DataSetPDS3CassiniISS(DataSetPDS3):
    """Implements dataset access for PDS3 Cassini ISS (Imaging Science Subsystem) data.

    This class provides specialized functionality for accessing and parsing Cassini
    ISS image data stored in PDS3 format. Includes all volumes (1001-1009 and 2001-2116).
    """

    _MIN_1xxx_VOL = 1001
    _MAX_1xxx_VOL = 1009
    _MIN_2xxx_VOL = 2001
    _MAX_2xxx_VOL = 2116
    _ALL_VOLUME_NAMES = tuple(
        f'COISS_{x:04d}'
        for x in list(range(_MIN_1xxx_VOL, _MAX_1xxx_VOL + 1))
        + list(range(_MIN_2xxx_VOL, _MAX_2xxx_VOL + 1))
    )
    _INDEX_COLUMNS = ('FILE_SPECIFICATION_NAME',)
    _INDEX_CAMERA_COLUMNS = ('INSTRUMENT_ID',)
    _INDEX_CAMERA_MAP: ClassVar[dict[str, str]] = {'ISSNA': 'NAC', 'ISSWA': 'WAC'}
    _VOLUMES_DIR_NAME = 'calibrated'

    # Methods inherited from DataSetPDS3

    @staticmethod
    def _get_label_filespec_from_index(row: dict[str, Any]) -> str:
        """Extracts the label file specification from a row from an index table.

        Parameters:
            row: Dictionary containing PDS3 index table row data.

        Returns:
            The file specification string from the row.
        """

        filespec = cast(str, row['FILE_SPECIFICATION_NAME'])
        # Intentionally uppercase only
        if not filespec.endswith('.IMG'):
            raise ValueError(f'Bad Primary File Spec "{filespec}" - expected ".IMG"')
        # Intentionally uppercase only
        return filespec.replace('.IMG', '_CALIB.LBL')

    @staticmethod
    def _get_image_filespec_from_label_filespec(label_filespec: str) -> str:
        """Extracts the image file specification from a label file specification.

        Parameters:
            label_filespec: The label file specification string to parse.

        Returns:
            The image file specification string.
        """
        # Intentionally uppercase only
        return label_filespec.replace('.LBL', '.IMG')

    @staticmethod
    def _get_img_name_from_label_filespec(filespec: str) -> str | None:
        """Extract the image name (with no extension) from a file specification.

        Parameters:
            filespec: The file specification string to parse.

        Returns:
            The image name if valid. None if the name is valid but should not be
            processed.

        Raises:
            ValueError: If the file specification format is invalid.
        """

        parts = filespec.split('/')
        if parts[0].upper().startswith('COISS_'):  # Volume name
            parts = parts[1:]
        if len(parts) != 3:
            raise ValueError(f'Bad Primary File Spec "{filespec}" - expected 3 directory levels')
        if parts[0].upper() != 'DATA':
            raise ValueError(f'Bad Primary File Spec "{filespec}" - expected "DATA"')
        range_dir = parts[1]
        img_name = parts[2]
        if len(range_dir) != 21 or range_dir[10] != '_':
            raise ValueError(
                f'Bad Primary File Spec "{filespec}" - expected "DATA/dddddddddd_dddddddddd"'
            )
        img_name = img_name.rsplit('.')[0].rsplit('_')[0]
        return img_name

    @staticmethod
    def _img_name_valid(img_name: str) -> bool:
        """True if an image name is valid for this instrument.

        Parameters:
            img_name: The name of the image.

        Returns:
            True if the image name is valid for this instrument, False otherwise.
        """

        img_name = img_name.upper()
        img_name = img_name.replace('_CALIB', '')
        img_name = img_name.replace('.IMG', '')
        # [NW]dddddddddd[_d[d]]
        if img_name[0] not in 'NW':
            return False
        if len(img_name) == 11:
            try:
                _ = int(img_name[1:])
                return True
            except ValueError:
                return False
        if 13 <= len(img_name) <= 14:
            if img_name[11] != '_':
                return False
            try:
                _ = int(img_name[1:11])
                _ = int(img_name[12:])
                return True
            except ValueError:
                return False
        return False

    @staticmethod
    def _extract_img_number(img_name: str) -> int:
        """Extract the image number from an image name.

        Parameters:
            img_name: The name of the image. Can be just the image name, the image filename,
                or the full file spec.

        Returns:
            The image number.

        Raises:
            ValueError: If the image name format is invalid.
        """

        if not DataSetPDS3CassiniISS._img_name_valid(img_name):
            raise ValueError(f'Invalid image name "{img_name}"')

        return int(safe_lstrip_zero(img_name[1:11]))

    @staticmethod
    def _volset_and_volume(volume: str) -> str:
        """Get the volset and volume name.

        Parameters:
            volume: The volume name.
        """
        return f'COISS_{volume[6]}xxx/{volume}'

    @staticmethod
    def _volume_to_index(volume: str) -> str:
        """Get the index file name for a volume.

        Parameters:
            volume: The volume name.
        """
        # Intentionally lowercase only
        return f'COISS_{volume[6]}xxx/{volume}/{volume}_index.lbl'

    @staticmethod
    def _results_path_stub(volume: str, filespec: str) -> str:
        """Get the results path stub for an image filespec.

        Parameters:
            volume: The volume name.
            filespec: The filespec of the image.

        Returns:
            The results path stub. This is {volume}/{filespec} without the .IMG extension.
        """

        return str(Path(f'{volume}/{filespec}').with_suffix(''))

    def _check_additional_image_selection_criteria(
        self,
        _img_path: str,
        img_name: str,
        _img_num: int,
        arguments: argparse.Namespace | None = None,
    ) -> bool:
        """Check additional image selection criteria.

        Parameters:
            _img_path: The path to the image.
            img_name: The name of the image.
            _img_num: The number of the image.
            arguments: The parsed arguments.

        Returns:
            True if the image should be processed, False otherwise.
        """

        if arguments is None or arguments.camera is None:
            return True
        if arguments.camera.lower() == 'nac':
            return img_name[0] == 'N'
        if arguments.camera.lower() == 'wac':
            return img_name[0] == 'W'
        raise ValueError(f'Invalid camera type "{arguments.camera}"')

    # Public methods

    def __init__(
        self,
        pds3_holdings_root: str | Path | FCPath | None = None,
        *,
        index_filecache: FileCache | None = None,
        pds3_holdings_filecache: FileCache | None = None,
        config: Config | None = None,
    ) -> None:
        """Initializes a Cassini ISS dataset handler.

        Parameters:
            pds3_holdings_root: Path to the PDS3 holdings directory, which may be a URL
                accepted by FCPath. If None, the root is resolved when it is first needed,
                from the command line, the configuration, or the environment, in that
                order; see DataSetPDS3.pds3_holdings_root.
            index_filecache: FileCache object to use for index files. If None, creates a new one.
            pds3_holdings_filecache: FileCache object to use for PDS3 holdings files. If None,
                creates a new one.
            config: Configuration object to use. If None, uses DEFAULT_CONFIG.
        """
        super().__init__(
            pds3_holdings_root=pds3_holdings_root,
            index_filecache=index_filecache,
            pds3_holdings_filecache=pds3_holdings_filecache,
            config=config,
        )

    @staticmethod
    def add_selection_arguments(
        cmdparser: argparse.ArgumentParser,
        group: argparse._ArgumentGroup | None = None,
    ) -> None:
        """Adds Cassini ISS-specific command-line arguments for image selection.

        Parameters:
            cmdparser: The argument parser to add arguments to.
            group: Optional argument group to add arguments to. If None, creates a new group.
        """

        # First add the PDS3 arguments
        DataSetPDS3.add_selection_arguments(cmdparser, group)

        if group is None:
            group = cmdparser.add_argument_group('Image selection (Cassini ISS-specific)')
        group.add_argument(
            '--camera',
            type=str,
            default=None,
            choices=['nac', 'NAC', 'wac', 'WAC'],
            help='Only process images with the given camera type',
        )

    # Maximum difference in IMAGE_TIME (seconds) allowed between the NAC and WAC frames
    # of a BOTSIM pair. BOTSIM ("both simultaneous") commands fire both shutters at the
    # same time, so the two frames are exposed within a fraction of a second of each
    # other; the observed offset between paired NAC/WAC IMAGE_TIME values is well under
    # one second. A 2.0 s tolerance comfortably accommodates that small offset while
    # remaining far below the gap to any neighboring (non-partner) observation.
    _BOTSIM_MAX_TIME_DELTA_SEC = 2.0

    @staticmethod
    def _parse_image_time(image_time: str) -> datetime | None:
        """Parse a Cassini ISS IMAGE_TIME value (``YYYY-DOYThh:mm:ss.sss``).

        Parameters:
            image_time: The IMAGE_TIME string from the index row.

        Returns:
            A timezone-aware UTC ``datetime``, or None if the value cannot be parsed.
        """

        try:
            parsed = datetime.strptime(image_time, '%Y-%jT%H:%M:%S.%f')
        except (ValueError, TypeError):
            return None
        return parsed.replace(tzinfo=UTC)

    @classmethod
    def _is_botsim_pair(cls, first: ImageFile, second: ImageFile) -> bool:
        """True if two frames form a genuine BOTSIM NAC/WAC pair.

        A BOTSIM pair requires both frames to be flagged ``SHUTTER_MODE_ID == 'BOTSIM'``,
        to come from opposite cameras (one NAC, one WAC), to share the same
        ``OBSERVATION_ID``, and to have ``IMAGE_TIME`` values within
        ``_BOTSIM_MAX_TIME_DELTA_SEC`` of each other. IMAGE_NUMBER is an SCLK-derived
        counter (not a wall-clock time), so it is deliberately not used as the pairing
        key; IMAGE_TIME provides the actual acquisition time.

        Parameters:
            first: The first candidate frame.
            second: The second candidate frame.

        Returns:
            True if the two frames are a confident BOTSIM pair, False otherwise.
        """

        first_row = first.index_file_row
        second_row = second.index_file_row
        if first_row['SHUTTER_MODE_ID'] != 'BOTSIM' or second_row['SHUTTER_MODE_ID'] != 'BOTSIM':
            return False
        # Opposite cameras (one N, one W)
        if {first.image_file_name[0], second.image_file_name[0]} != {'N', 'W'}:
            return False
        # Same observation
        if first_row.get('OBSERVATION_ID') != second_row.get('OBSERVATION_ID'):
            return False
        # Acquired at essentially the same time
        first_time = cls._parse_image_time(str(first_row.get('IMAGE_TIME', '')))
        second_time = cls._parse_image_time(str(second_row.get('IMAGE_TIME', '')))
        if first_time is None or second_time is None:
            return False
        return abs((first_time - second_time).total_seconds()) <= cls._BOTSIM_MAX_TIME_DELTA_SEC

    def yield_image_files_index(self, **kwargs: Any) -> Generator[ImageFiles, None, None]:
        """Yield filenames given search criteria using index files.

        Parameters:
            group: How to group the results. Only 'botsim' is supported.
            **kwargs: Arbitrary keyword arguments, usually used to restrict the search.

        Yields:
            ImageFiles objects containing information about groups of selected
            image files.
        """

        group = kwargs.pop('group', None)

        if group is None:
            # closing(): see DataSetPDS3.yield_image_files_index.
            with closing(self._yield_image_files_index(**kwargs)) as imagefiles:
                for imagefile in imagefiles:
                    yield ImageFiles(image_files=[imagefile])
            return

        if group != 'botsim':
            raise ValueError(f'Invalid group type "{group}"')

        # Combine confirmed BOTSIM NAC/WAC pairs into a single ImageFiles group; every
        # other frame is yielded on its own. A frame is held back for one iteration only
        # to test it against its immediate successor, so no frame is ever dropped: an
        # unpaired held frame is always yielded before moving on.
        last_imagefile: ImageFile | None = None
        # closing(): see DataSetPDS3.yield_image_files_index.
        paired = closing(
            self._yield_image_files_index(
                additional_index_columns=(
                    'SHUTTER_MODE_ID',
                    'IMAGE_NUMBER',
                    'OBSERVATION_ID',
                    'IMAGE_TIME',
                ),
                **kwargs,
            )
        )
        with paired as imagefiles:
            for imagefile in imagefiles:
                if last_imagefile is None:
                    last_imagefile = imagefile
                    continue
                if self._is_botsim_pair(last_imagefile, imagefile):
                    if last_imagefile.image_file_name[0] == 'N':
                        nac_imagefile = last_imagefile
                        wac_imagefile = imagefile
                    else:
                        wac_imagefile = last_imagefile
                        nac_imagefile = imagefile
                    yield ImageFiles(image_files=[nac_imagefile, wac_imagefile])
                    last_imagefile = None
                else:
                    # Not a pair: emit the held frame and hold the current one. The held
                    # frame is never silently discarded.
                    yield ImageFiles(image_files=[last_imagefile])
                    last_imagefile = imagefile

        if last_imagefile is not None:
            yield ImageFiles(image_files=[last_imagefile])

    @staticmethod
    def supported_grouping() -> list[str]:
        """Returns the list of supported grouping types.

        Returns:
            The list of supported grouping types. For Cassini ISS, only 'botsim'
            is supported.
        """
        return ['botsim']

    def pds4_bundle_template_dir(self) -> str:
        """Returns absolute path to template directory for PDS4 bundle generation.

        Returns:
            Absolute path to template directory.
        """
        # Check config first
        template_dir = None
        dataset_config = self.config.pds4.get(self._dataset_name_for_pds4_config(), {})
        if 'template_dir' in dataset_config:
            template_dir = str(dataset_config['template_dir'])

        if template_dir is None:
            template_dir = self._default_pds4_template_dir()

        # If it's an absolute path, return as-is
        if Path(template_dir).is_absolute():
            return template_dir

        # Otherwise, make it relative to pds4/templates
        pds4_templates_dir = Path(__file__).resolve().parent.parent / 'cli' / 'pds4' / 'templates'
        return str(pds4_templates_dir / template_dir)

    def pds4_bundle_name(self) -> str:
        """Returns bundle name for PDS4 bundle generation.

        Returns:
            Bundle name.
        """
        # Check config first
        dataset_config = self.config.pds4.get(self._dataset_name_for_pds4_config(), {})
        if 'bundle_name' in dataset_config:
            return str(dataset_config['bundle_name'])
        # Default
        return self._default_pds4_bundle_name()

    def pds4_required_templates(self, pds4_pass: Pds4Pass) -> list[str]:
        """Returns the template filenames one bundle pass must find for this dataset.

        Parameters:
            pds4_pass: Which pass's templates to name: ``labels`` for the
                per-image pass, ``summary`` for the collection and index pass.

        Returns:
            The filenames, relative to the template directory.
        """
        if pds4_pass == 'labels':
            return ['data.lblx', 'browse.lblx']
        return [
            'collection_data.lblx',
            'collection_browse.lblx',
            'global_index_bodies.lblx',
            'global_index_rings.lblx',
        ]

    @staticmethod
    def pds4_bundle_path_for_image(image_name: str) -> str:
        """Maps image name to bundle directory path.

        Parameters:
            image_name: The image name to map (e.g., "N1234567890").

        Returns:
            Bundle directory path relative to bundle root
            (e.g., "1234xxxxxx/123456xxxx").
        """
        # Extract image number from name (skip first character N/W).  A valid
        # Cassini image name is always >= 11 chars here; a shorter name is a
        # programming error, so fail loudly rather than returning an empty
        # string that callers would concatenate into a malformed path.
        if len(image_name) < 11:
            raise ValueError(
                f'invalid Cassini image name {image_name!r}: expected >= 11 characters'
            )
        img_num_str = image_name[1:11]
        img_num = int(img_num_str)

        # Format: 1234xxxxxx/123456xxxx
        first_part = f'{img_num // 1000000:04d}xxxxxx'
        second_part = f'{img_num // 10000:06d}xxxx'
        return f'{first_part}/{second_part}/'

    def pds4_path_stub(self, image_file: ImageFile) -> str:
        """Returns PDS4 path stub for bundle directory structure.

        Parameters:
            image_file: The image file to generate path stub for.

        Returns:
            Path stub relative to bundle root (e.g., "1234xxxxxx/123456xxxx/1234567890w").
        """
        image_name = image_file.image_file_name.split('_', 1)[0].split('.', 1)[0]
        image_lid_part = image_name[1:] + image_name[0].lower()
        # pds4_bundle_path_for_image now raises on an invalid name rather than
        # returning '', so the bundle path is always present here.
        bundle_path = self.pds4_bundle_path_for_image(image_name)
        return f'{bundle_path.rstrip("/")}/{image_lid_part}'

    def pds4_lid_part_to_image_name(self, lid_part: str) -> str:
        """Returns the image name for the given LID part.

        Inverts the transform applied by :meth:`pds4_path_stub` and the
        ``pds4_image_name_to_*`` builders, which move the leading camera letter
        (``N``/``W``) to a lowercase suffix (``N1454725799`` ->
        ``1454725799n``). The inverse moves the trailing lowercase letter back
        to the front as an uppercase prefix (``1454725799n`` -> ``N1454725799``).

        Parameters:
            lid_part: The LID part (an on-disk product filename stem).

        Returns:
            The image name that produced the given LID part.
        """
        if len(lid_part) < 2:
            raise ValueError(f'invalid Cassini LID part {lid_part!r}: expected >= 2 characters')
        return lid_part[-1].upper() + lid_part[:-1]

    def pds4_image_name_to_browse_lid(self, image_name: str) -> str:
        """Returns the browse LID for the given image name.

        Parameters:
            image_name: The image name to convert to a browse LID.

        Returns:
            The browse LID.
        """
        image_name = image_name.split('_', 1)[0].split('.', 1)[0]
        image_lid_part = image_name[1:] + image_name[0].lower()
        return f'urn:nasa:pds:{self.pds4_bundle_name()}:browse:{image_lid_part}'

    def pds4_image_name_to_browse_lidvid(self, image_name: str) -> str:
        """Returns the browse LIDVID for the given image name.

        Parameters:
            image_name: The image name to convert to a browse LID.

        Returns:
            The browse LIDVID.
        """
        image_name = image_name.split('_', 1)[0].split('.', 1)[0]
        image_lid_part = image_name[1:] + image_name[0].lower()
        return f'urn:nasa:pds:{self.pds4_bundle_name()}:browse:{image_lid_part}::1.0'

    def pds4_image_name_to_data_lid(self, image_name: str) -> str:
        """Returns the data LID for the given image name.

        Parameters:
            image_name: The image name to convert to a data LID.

        Returns:
            The data LID.
        """
        image_name = image_name.split('_', 1)[0].split('.', 1)[0]
        image_lid_part = image_name[1:] + image_name[0].lower()
        return f'urn:nasa:pds:{self.pds4_bundle_name()}:data:{image_lid_part}'

    def pds4_image_name_to_data_lidvid(self, image_name: str) -> str:
        """Returns the data LIDVID for the given image name.

        Parameters:
            image_name: The image name to convert to a data LID.

        Returns:
            The data LIDVID.
        """
        image_name = image_name.split('_', 1)[0].split('.', 1)[0]
        image_lid_part = image_name[1:] + image_name[0].lower()
        return f'urn:nasa:pds:{self.pds4_bundle_name()}:data:{image_lid_part}::1.0'

    def pds4_template_variables(
        self,
        *,
        image_file: ImageFile,
        nav_metadata: dict[str, Any],
        backplane_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Returns template variables for PDS4 label generation.

        ``START_DATE_TIME`` and ``STOP_DATE_TIME`` are the exposure's start and stop,
        read from the epochs the navigation recorded under ``navigation_result.times``
        and written the way a PDS4 label writes a UTC time, to the millisecond, with a
        trailing ``Z``, each rounded to the nearest millisecond.  An image's start and
        stop are recorded to the millisecond in its PDS3 label and index, and the
        epochs are computed from those values, so each epoch lies within a few
        nanoseconds of a millisecond, on one side of it or the other: the nearest
        millisecond is the time recorded, where rounding a start down or a stop up
        would move it a whole millisecond whenever the epoch lands on the far side.
        ``IMAGE_MID_TIME`` is the midpoint of the two as written, a half millisecond
        rounding up, which is PDS3's ``IMAGE_MID_TIME``: an exposure an odd number of
        milliseconds long has its midtime on a half millisecond, where the recorded
        midtime epoch lands a few nanoseconds to either side of it.

        Parameters:
            image_file: The image file being processed.
            nav_metadata: Navigation metadata dictionary: a success document recording
                the exposure's epochs under ``navigation_result.times``.
            backplane_metadata: Backplane metadata dictionary.

        Returns:
            Dictionary mapping variable names to values for template
            substitution.

        Raises:
            KeyError: If ``nav_metadata`` records no ``navigation_result.times``, as
                a navigation that recorded no pointing leaves it.  The labels pass
                fails such an image before it asks for these variables.
        """
        vars_dict: dict[str, Any] = {}

        # Camera information from image name
        img_name = image_file.image_file_name.upper()
        if img_name.startswith('N'):
            vars_dict['CAMERA_WIDTH'] = 'Narrow'
            vars_dict['CAMERA_WN_UC'] = 'N'
            vars_dict['CAMERA_WN_LC'] = 'n'
        elif img_name.startswith('W'):
            vars_dict['CAMERA_WIDTH'] = 'Wide'
            vars_dict['CAMERA_WN_UC'] = 'W'
            vars_dict['CAMERA_WN_LC'] = 'w'
        else:
            vars_dict['CAMERA_WIDTH'] = 'Unknown'
            vars_dict['CAMERA_WN_UC'] = ''
            vars_dict['CAMERA_WN_LC'] = ''

        # The exposure's start and stop, from the epochs its navigation recorded, each
        # at the nearest millisecond: the epochs are computed from times recorded to the
        # millisecond, so the nearest is the one recorded, where a floor or a ceiling
        # would lose it whenever the float lands a few nanoseconds on its far side.  The
        # midtime is their midpoint as written, a half rounding up as PDS3's does; the
        # midtime epoch of an odd-millisecond exposure sits on the half, either side.
        times = nav_metadata['navigation_result']['times']
        vars_dict['START_DATE_TIME'] = et_to_pds4_utc(
            times['start_et'], digits=_PDS4_TIME_DIGITS, rounding='nearest'
        )
        vars_dict['STOP_DATE_TIME'] = et_to_pds4_utc(
            times['stop_et'], digits=_PDS4_TIME_DIGITS, rounding='nearest'
        )
        vars_dict['IMAGE_MID_TIME'] = pds4_utc_midpoint(
            vars_dict['START_DATE_TIME'], vars_dict['STOP_DATE_TIME']
        )

        # Placeholder values for required template variables
        pds4_bundle_name = self.pds4_bundle_name()
        image_name = image_file.image_file_name.split('_', 1)[0].split('.', 1)[0]
        image_lid_part = image_name[1:] + image_name[0].lower()
        vars_dict['BUNDLE_LID'] = f'urn:nasa:pds:{pds4_bundle_name}'
        vars_dict['BUNDLE_LIDVID'] = f'urn:nasa:pds:{pds4_bundle_name}::1.0'
        vars_dict['BROWSE_LID'] = f'urn:nasa:pds:{pds4_bundle_name}:browse:{image_lid_part}'
        vars_dict['BROWSE_LIDVID'] = f'urn:nasa:pds:{pds4_bundle_name}:browse:{image_lid_part}::1.0'
        vars_dict['DATA_LID'] = f'urn:nasa:pds:{pds4_bundle_name}:data:{image_lid_part}'
        vars_dict['DATA_LIDVID'] = f'urn:nasa:pds:{pds4_bundle_name}:data:{image_lid_part}::1.0'
        vars_dict['TITLE'] = f'Backplanes for {image_file.image_file_name}'
        vars_dict['DESCRIPTION'] = f'Backplanes for navigated image {image_file.image_file_name}'
        vars_dict['COMMENT'] = 'Generated from navigated image data'
        vars_dict['SOURCE_IMAGE_LIDVID'] = (
            f'urn:nasa:pds:{pds4_bundle_name}:data:{image_lid_part}::1.0'
        )

        # Extract from index_file_row if available
        index_row = image_file.index_file_row
        if index_row:
            # Map PDS3 index columns to PDS4 cassini: namespace variables
            # Based on COISS index file structure from
            # https://pds-rings.seti.org/holdings/metadata/COISS_2xxx/COISS_2001/COISS_2001_index.lbl
            vars_dict['cassini:mission_phase_name'] = index_row.get('MISSION_PHASE_NAME', '')
            vars_dict['cassini:spacecraft_clock_count_partition'] = index_row.get(
                'SPACECRAFT_CLOCK_COUNT_PARTITION', ''
            )
            vars_dict['cassini:spacecraft_clock_start_count'] = index_row.get(
                'SPACECRAFT_CLOCK_START_COUNT', ''
            )
            vars_dict['cassini:spacecraft_clock_stop_count'] = index_row.get(
                'SPACECRAFT_CLOCK_STOP_COUNT', ''
            )
            vars_dict['cassini:limitations'] = index_row.get('LIMITATIONS', 'N/A')
            vars_dict['cassini:antiblooming_state_flag'] = index_row.get(
                'ANTIBLOOMING_STATE_FLAG', ''
            )
            vars_dict['cassini:bias_strip_mean'] = index_row.get('BIAS_STRIP_MEAN', '')
            vars_dict['cassini:calibration_lamp_state_flag'] = index_row.get(
                'CALIBRATION_LAMP_STATE_FLAG', ''
            )
            vars_dict['cassini:command_file_name'] = index_row.get('COMMAND_FILE_NAME', '')
            vars_dict['cassini:command_sequence_number'] = index_row.get(
                'COMMAND_SEQUENCE_NUMBER', ''
            )
            vars_dict['cassini:dark_strip_mean'] = index_row.get('DARK_STRIP_MEAN', '')
            vars_dict['cassini:data_conversion_type'] = index_row.get('DATA_CONVERSION_TYPE', '')
            vars_dict['cassini:delayed_readout_flag'] = index_row.get('DELAYED_READOUT_FLAG', '')
            vars_dict['cassini:detector_temperature'] = index_row.get('DETECTOR_TEMPERATURE', '')
            vars_dict['cassini:electronics_bias'] = index_row.get('ELECTRONICS_BIAS', '')
            vars_dict['cassini:earth_received_start_time'] = index_row.get(
                'EARTH_RECEIVED_START_TIME', ''
            )
            vars_dict['cassini:earth_received_stop_time'] = index_row.get(
                'EARTH_RECEIVED_STOP_TIME', ''
            )
            vars_dict['cassini:expected_maximum_full_well'] = index_row.get(
                'EXPECTED_MAXIMUM', '-1'
            )
            vars_dict['cassini:expected_maximum_DN_sat'] = index_row.get(
                'EXPECTED_MAXIMUM_DN_SAT', '-1'
            )
            vars_dict['cassini:expected_packets'] = index_row.get('EXPECTED_PACKETS', '-1')
            vars_dict['cassini:exposure_duration'] = index_row.get('EXPOSURE_DURATION', '')
            vars_dict['cassini:filter_name_1'] = index_row.get('FILTER1', '')
            vars_dict['cassini:filter_name_2'] = index_row.get('FILTER2', '')
            vars_dict['cassini:filter_temperature'] = index_row.get('FILTER_TEMPERATURE', '')
            vars_dict['cassini:flight_software_version_id'] = index_row.get(
                'FLIGHT_SOFTWARE_VERSION_ID', ''
            )
            vars_dict['cassini:gain_mode_id'] = index_row.get('GAIN_MODE_ID', '')
            vars_dict['cassini:ground_software_version_id'] = index_row.get(
                'GROUND_SOFTWARE_VERSION_ID', ''
            )
            vars_dict['cassini:image_mid_time'] = index_row.get('IMAGE_TIME', '')
            vars_dict['cassini:image_number'] = index_row.get('IMAGE_NUMBER', '')
            vars_dict['cassini:image_time'] = index_row.get('IMAGE_TIME', '')
            vars_dict['cassini:image_observation_type'] = index_row.get(
                'IMAGE_OBSERVATION_TYPE', ''
            )
            vars_dict['cassini:instrument_data_rate'] = index_row.get('INSTRUMENT_DATA_RATE', '')
            vars_dict['cassini:instrument_mode_id'] = index_row.get('INSTRUMENT_MODE_ID', '')
            vars_dict['cassini:inst_cmprs_type'] = index_row.get('INST_CMPRS_TYPE', '')
            vars_dict['cassini:inst_cmprs_param_malgo'] = '999'
            vars_dict['cassini:inst_cmprs_param_tb'] = '999'
            vars_dict['cassini:inst_cmprs_param_blocks'] = '999'
            vars_dict['cassini:inst_cmprs_param_quant'] = '999'
            vars_dict['cassini:inst_cmprs_rate_expected_bits'] = index_row.get(
                'INST_CMPRS_RATE_EXPECTED_BITS', '-1.'
            )
            vars_dict['cassini:inst_cmprs_rate_actual_bits'] = index_row.get(
                'INST_CMPRS_RATE_ACTUAL_BITS', '-1.'
            )
            vars_dict['cassini:inst_cmprs_ratio'] = index_row.get('INST_CMPRS_RATIO', '')
            vars_dict['cassini:light_flood_state_flag'] = index_row.get(
                'LIGHT_FLOOD_STATE_FLAG', ''
            )
            vars_dict['cassini:method_description'] = index_row.get('METHOD_DESC', '')
            vars_dict['cassini:missing_lines'] = index_row.get('MISSING_LINES', '0')
            vars_dict['cassini:missing_packet_flag'] = index_row.get('MISSING_PACKET_FLAG', 'NO')
            vars_dict['cassini:observation_id'] = index_row.get('OBSERVATION_ID', '')
            vars_dict['cassini:optics_temperature_front'] = index_row.get('OPTICS_TEMPERATURE', '')
            vars_dict['cassini:optics_temperature_back'] = index_row.get(
                'OPTICS_TEMPERATURE_BACK', '-999.'
            )
            vars_dict['cassini:order_number'] = index_row.get('ORDER_NUMBER', '-999')
            vars_dict['cassini:parallel_clock_voltage_index'] = index_row.get(
                'PARALLEL_CLOCK_VOLTAGE_INDEX', ''
            )
            vars_dict['cassini:pds3_product_creation_time'] = index_row.get(
                'PRODUCT_CREATION_TIME', ''
            )
            vars_dict['cassini:pds3_product_version_type'] = index_row.get(
                'PRODUCT_VERSION_TYPE', 'FINAL'
            )
            vars_dict['cassini:pds3_target_desc'] = index_row.get('TARGET_DESC', '')
            vars_dict['cassini:pds3_target_list'] = index_row.get('TARGET_LIST', 'N/A')
            vars_dict['cassini:pds3_target_name'] = index_row.get('TARGET_NAME', '')
            vars_dict['cassini:pre-pds_version_number'] = index_row.get(
                'PRE_PDS_VERSION_NUMBER', '2'
            )
            vars_dict['cassini:prepare_cycle_index'] = index_row.get('PREPARE_CYCLE_INDEX', '')
            vars_dict['cassini:readout_cycle_index'] = index_row.get('READOUT_CYCLE_INDEX', '')
            vars_dict['cassini:received_packets'] = index_row.get('RECEIVED_PACKETS', '-1')
            vars_dict['cassini:sensor_head_electronics_temperature'] = index_row.get(
                'SENSOR_HEAD_ELEC_TEMPERATURE', ''
            )
            vars_dict['cassini:sequence_id'] = index_row.get('SEQUENCE_ID', '')
            vars_dict['cassini:sequence_number'] = index_row.get('SEQUENCE_NUMBER', '-1')
            vars_dict['cassini:sequence_title'] = index_row.get('SEQUENCE_TITLE', '')
            vars_dict['cassini:shutter_mode_id'] = index_row.get('SHUTTER_MODE_ID', '')
            vars_dict['cassini:shutter_state_id'] = index_row.get('SHUTTER_STATE_ID', '')
            vars_dict['cassini:start_time_doy'] = index_row.get('START_TIME_DOY', '')
            vars_dict['cassini:stop_time_doy'] = index_row.get('STOP_TIME_DOY', '')
            vars_dict['cassini:telemetry_format_id'] = index_row.get('TELEMETRY_FORMAT_ID', '')
            vars_dict['cassini:valid_maximum_full_well'] = index_row.get(
                'VALID_MAXIMUM_FULL_WELL', '-1'
            )
            vars_dict['cassini:valid_maximum_DN_sat'] = index_row.get('VALID_MAXIMUM_DN_SAT', '-1')

        return vars_dict

    def _dataset_name_for_pds4_config(self) -> str:
        """Returns the dataset name for config lookup under the PDS4 section.

        Returns:
            Dataset name for config (e.g., "coiss_cruise" or "coiss_saturn").
        """
        raise NotImplementedError('PDS4 bundle generation not supported for this dataset')

    def _default_pds4_template_dir(self) -> str:
        """Returns the default template directory name.

        Returns:
            Default template directory name.
        """
        raise NotImplementedError('PDS4 bundle generation not supported for this dataset')

    def _default_pds4_bundle_name(self) -> str:
        """Returns the default bundle name.

        Returns:
            Default bundle name.
        """
        raise NotImplementedError('PDS4 bundle generation not supported for this dataset')


class DataSetPDS3CassiniISSCruise(DataSetPDS3CassiniISS):
    """Implements dataset access for PDS3 Cassini ISS Cruise data (volumes 1001-1009)."""

    _MIN_1xxx_VOL = 1001
    _MAX_1xxx_VOL = 1009
    _ALL_VOLUME_NAMES = tuple(f'COISS_{x:04d}' for x in range(_MIN_1xxx_VOL, _MAX_1xxx_VOL + 1))

    def _dataset_name_for_pds4_config(self) -> str:
        return 'coiss_cruise'

    def _default_pds4_template_dir(self) -> str:
        return 'cassini_iss_cruise_1.0'

    def _default_pds4_bundle_name(self) -> str:
        return 'cassini_iss_cruise_backplanes_rsfrench2027'


class DataSetPDS3CassiniISSSaturn(DataSetPDS3CassiniISS):
    """Implements dataset access for PDS3 Cassini ISS Saturn data (volumes 2001-2116)."""

    _MIN_2xxx_VOL = 2001
    _MAX_2xxx_VOL = 2116
    _ALL_VOLUME_NAMES = tuple(f'COISS_{x:04d}' for x in range(_MIN_2xxx_VOL, _MAX_2xxx_VOL + 1))

    def _dataset_name_for_pds4_config(self) -> str:
        return 'coiss_saturn'

    def _default_pds4_template_dir(self) -> str:
        return 'cassini_iss_saturn_1.0'

    def _default_pds4_bundle_name(self) -> str:
        return 'cassini_iss_saturn_backplanes_rsfrench2027'
