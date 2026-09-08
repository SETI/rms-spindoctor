import argparse
import csv
import os
import random
import re
from abc import abstractmethod
from collections.abc import Generator, Iterator
from contextlib import closing
from functools import lru_cache
from pathlib import Path
from typing import Any, ClassVar, cast

from filecache import FCPath, FileCache
from pdstable import PdsTable

from spindoctor.config import (
    Config,
    get_nav_results_root,
    get_results_index_db_url,
    get_results_tree_tuning,
)
from spindoctor.support.misc import flatten_list

from .dataset import DataSet, ImageFile, ImageFiles
from .results_filter import RESULTS_FILTER_BATCH_SIZE, ResultsFilter, SelectionError

# A PDS3 ^IMAGE pointer naming the data file that holds the image object, e.g.
#   ^IMAGE = ("N1454725799_1.IMG",4)
#   ^IMAGE                          = ("C3250013_GEOMED.IMG", 2)
#   ^IMAGE = "LOR_0003103486_0X630_SCI.FIT"
# Matches only the top-level IMAGE pointer, not IMAGE_HEADER / EXTENSION_*_IMAGE.
_LABEL_IMAGE_POINTER_RE = re.compile(
    r'^\s*\^IMAGE\s*=\s*\(?\s*"?(?P<name>[^"\',()\s]+)"?',
    re.MULTILINE,
)


def _positive_int(value: str) -> int:
    """Parses an argparse value as a strictly positive integer.

    Parameters:
        value: The raw command-line string.

    Returns:
        The parsed integer.

    Raises:
        argparse.ArgumentTypeError: If the value is not a positive integer.
    """
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f'must be a positive integer, got {value!r}')
    return ivalue


class DataSetPDS3(DataSet):
    """Parent class for PDS3 datasets.

    This class provides functionality common to all PDS3 datasets.
    """

    # Data definitions overriden by subclasses
    _ALL_VOLUME_NAMES: tuple[str, ...] = ()
    _INDEX_COLUMNS: tuple[str, ...] = ()
    # Index columns naming the camera that took the image, in preference order;
    # the first one present and non-null in a row wins.  These are always read
    # and land in ``ImageFile.index_file_row``, so an image's camera is known
    # from the index alone -- no SPICE, and no need to load the image.  See
    # :meth:`camera_from_index_row`.
    _INDEX_CAMERA_COLUMNS: tuple[str, ...] = ()
    # Maps the raw index value (upper-cased, stripped) to the camera name the
    # rest of the system uses -- the same names ``ObsInst.camera`` returns, so
    # the two sources are interchangeable.
    _INDEX_CAMERA_MAP: ClassVar[dict[str, str]] = {}
    _VOLUMES_DIR_NAME: str = ''
    # True when every image number in a volume is greater than every image number in
    # all earlier volumes, letting a sequential scan stop once a whole volume is past
    # the end of the requested number range. False for datasets whose image counter
    # resets between volumes (Voyager FDS counts restart per spacecraft/encounter and
    # VG2 Neptune rolls over below VG1 Jupiter), where every volume must be scanned.
    _IMG_NUM_MONOTONIC_ACROSS_VOLUMES: bool = True

    def __init__(
        self,
        pds3_holdings_root: str | Path | FCPath | None = None,
        *,
        index_filecache: FileCache | None = None,
        pds3_holdings_filecache: FileCache | None = None,
        config: Config | None = None,
    ) -> None:
        """Initializes a PDS3 dataset with directory and cache settings.

        Parameters:
            pds3_holdings_root: Path to PDS3 holdings directory. If None, uses PDS3_HOLDINGS_DIR
                environment variable. May be a URL accepted by FCPath.
            index_filecache: FileCache object to use for index files. If None, creates a new one.
            pds3_holdings_filecache: FileCache object to use for PDS3 holdings files. If None,
                creates a new one.
            config: Configuration object to use. If None, uses DEFAULT_CONFIG.

        Raises:
            ValueError: If pds3_holdings_root is None and PDS3_HOLDINGS_DIR environment variable
                is not set.
        """

        super().__init__(config=config)

        if index_filecache is None:
            self._index_filecache = FileCache('nav_pds3_index')  # Index shared; MP safe
        else:
            self._index_filecache = index_filecache

        if pds3_holdings_filecache is None:
            self._pds3_holdings_filecache = FileCache(None)  # Data not shared
        else:
            self._pds3_holdings_filecache = pds3_holdings_filecache

        if pds3_holdings_root is not None:
            self._pds3_holdings_root: FCPath | None = self._pds3_holdings_filecache.new_path(
                pds3_holdings_root
            )
        else:
            self._pds3_holdings_root = None

    @property
    def pds3_holdings_root(self) -> FCPath:
        """The PDS3 holdings directory; may be a URL."""
        if self._pds3_holdings_root is not None:
            return self._pds3_holdings_root

        pds3_holdings_root = None
        try:
            pds3_holdings_root = self.config.environment.pds3_holdings_root
        except AttributeError:
            pass
        if pds3_holdings_root is None:
            pds3_holdings_root = os.getenv('PDS3_HOLDINGS_DIR')
        if pds3_holdings_root is None:
            raise ValueError(
                'One of configuration variable "pds3_holdings_root" or '
                'PDS3_HOLDINGS_DIR environment variable must be set'
            )
        self._pds3_holdings_root = self._pds3_holdings_filecache.new_path(pds3_holdings_root)

        return self._pds3_holdings_root

    def __str__(self) -> str:
        return f'DataSetPDS3(pds3_holdings_root={self._pds3_holdings_root})'

    def __repr__(self) -> str:
        return self.__str__()

    @classmethod
    def camera_from_index_row(cls, index_row: dict[str, Any]) -> str | None:
        """The camera that took the image, read from its PDS3 index row.

        This needs neither SPICE nor the image itself, so an image whose
        navigation died for want of a SPICE kernel still names its camera.
        The result uses the same names as ``ObsInst.camera``, so an
        observation's camera and this one are interchangeable.

        Parameters:
            index_row: An ``ImageFile.index_file_row``; an empty dict (an
                image not enumerated from an index) yields None.

        Returns:
            The camera name, or None when the row carries no recognized
            camera value.  An unrecognized value yields None rather than
            itself: a name the rest of the system does not know would
            silently split the statistics it feeds.
        """
        for column in cls._INDEX_CAMERA_COLUMNS:
            value = index_row.get(column)
            # PdsTable reports a null cell via a companion ``<column>_mask``.
            if value is None or index_row.get(f'{column}_mask'):
                continue
            camera = cls._INDEX_CAMERA_MAP.get(str(value).strip().upper())
            if camera is not None:
                return camera
        return None

    @staticmethod
    @abstractmethod
    def _get_label_filespec_from_index(row: dict[str, Any]) -> str:
        """Extracts the label file specification from a row from an index table.

        Parameters:
            row: Dictionary containing PDS3 index table row data.

        Returns:
            The file specification string from the row.
        """
        ...

    @staticmethod
    @abstractmethod
    def _get_image_filespec_from_label_filespec(label_filespec: str) -> str:
        """Extracts the image file specification from a label file specification.

        Parameters:
            label_filespec: The label file specification string to parse.

        Returns:
            The image file specification string.
        """
        ...

    @staticmethod
    @abstractmethod
    def _get_img_name_from_label_filespec(filespec: str) -> str | None:
        """Extracts the image name (with no extension) from a file specification.

        Parameters:
            filespec: The file specification string to parse.

        Returns:
            The image name if valid. None if the name is valid but should not be
            processed.

        Raises:
            ValueError: If the file specification format is invalid.
        """
        ...

    @staticmethod
    @abstractmethod
    def _img_name_valid(img_name: str) -> bool:
        """True if an image name is valid for this instrument.

        Parameters:
            img_name: The name of the image.

        Returns:
            True if the image name is valid for this instrument, False otherwise.
        """
        ...

    @staticmethod
    @abstractmethod
    def _extract_img_number(img_name: str) -> int:
        """Extract the image number from an image name.

        Parameters:
            img_name: The name of the image. Can be just the image name, the image
                filename, or the full file spec.

        Returns:
            The image number.

        Raises:
            ValueError: If the image name format is invalid.
        """
        ...

    @staticmethod
    @abstractmethod
    def _volset_and_volume(volume: str) -> str:
        """Get the volset and volume name.

        Parameters:
            volume: The volume name.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _volume_to_index(volume: str) -> str:
        """Get the index file name for a volume.

        Parameters:
            volume: The volume name.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _results_path_stub(volume: str, filespec: str) -> str:
        """Get the results path stub for an image filespec."""
        raise NotImplementedError

    @staticmethod
    def add_selection_arguments(
        cmdparser: argparse.ArgumentParser,
        group: argparse._ArgumentGroup | None = None,
    ) -> None:
        """Adds PDS3-specific command-line arguments for image selection.

        Parameters:
            cmdparser: The argument parser to add arguments to.
            group: Optional argument group to add arguments to. If None, creates a new group.
        """

        if group is None:
            group = cmdparser.add_argument_group('Image selection (PDS3-specific)')
        group.add_argument(
            'img_name',
            action='append',
            nargs='*',
            type=str,
            help='Specific image name(s) to process',
        )
        group.add_argument(
            '--first-image-num',
            type=int,
            default=None,
            metavar='IMAGE_NUM',
            help="""The starting image number; only images with this number or greater will be
            processed""",
        )
        group.add_argument(
            '--last-image-num',
            type=int,
            default=None,
            metavar='IMAGE_NUM',
            help="""The ending image number; only images with this number or less will be
            processed""",
        )
        group.add_argument(
            '--volumes',
            action='append',
            help="""One or more entire PDS3 volume names; only images in these
            volumes or volume subdirectories will be processed. Can accept multiple values
            separated by commas or multiple arguments.""",
        )
        group.add_argument(
            '--first-volume',
            type=str,
            default=None,
            metavar='VOL_NAME',
            help="""The starting PDS3 volume name; only images in this volume or chronologically
            later will be processed""",
        )
        group.add_argument(
            '--last-volume',
            type=str,
            default=None,
            metavar='VOL_NAME',
            help="""The ending PDS3 volume name; only images in this volume or chronologically
            earlier will be processed""",
        )
        group.add_argument(
            '--image-filespec-csv',
            action='append',
            help="""A CSV file that contains filespecs of images to process; a header row
            is required and must contain a column named 'Primary File Spec' or 'primaryfilespec'.
            The list is still subject to other selection criteria.""",
        )
        group.add_argument(
            '--image-file-list',
            action='append',
            help="""A file that contains filespecs or names of images to process;
            the list is still subject to other selection criteria.""",
        )
        group.add_argument(
            '--has-offset-file',
            action='store_true',
            default=False,
            help='Only process images that already have an offset metadata file',
        )
        group.add_argument(
            '--has-no-offset-file',
            action='store_true',
            default=False,
            help="Only process images that don't already have an offset metadata file",
        )
        group.add_argument(
            '--has-offset-error',
            action='store_true',
            default=False,
            help="""Only process images if the offset metadata file exists and
            indicates a fatal error""",
        )
        group.add_argument(
            '--has-no-offset-error',
            action='store_true',
            default=False,
            help="""Only process images if the offset metadata file exists and
            records a status other than the fatal one, which for the documents
            this pipeline writes is the images whose navigation ran to a
            result""",
        )
        group.add_argument(
            '--has-offset-spice-error',
            action='store_true',
            default=False,
            help="""Only process images if the offset metadata file exists and
            indicates a fatal error from missing SPICE data""",
        )
        group.add_argument(
            '--has-offset-nonspice-error',
            action='store_true',
            default=False,
            help="""Only process images if the offset metadata file exists and
            indicates a fatal error other than missing SPICE data""",
        )
        # group.add_argument(
        #     '--selection-expr', type=str, metavar='EXPR',
        #     help='Expression to evaluate to decide whether to reprocess an offset')
        group.add_argument(
            '--choose-random-images',
            type=_positive_int,
            default=None,
            metavar='N',
            help='Choose N random images to process within other constraints (N must be positive)',
        )
        # group.add_argument(
        #     '--show-image-list-only', action='store_true', default=False,
        #     help="""Just show a list of files that would be processed without doing
        #             any actual processing"""
        # )

    def _validate_selection_arguments(self, arguments: argparse.Namespace) -> None:
        """Validates user arguments that can't be checked during initial parsing.

        Parameters:
            arguments: The parsed arguments to validate.
        """

        # TODO This method is currently unused and should be used
        # For some reason mypy can't see the img_name field
        for img_name in flatten_list(arguments.img_name):
            if not self._img_name_valid(img_name):
                raise argparse.ArgumentTypeError(f'Invalid image name {img_name}')

    def yield_image_files_from_arguments(
        self, arguments: argparse.Namespace
    ) -> Iterator[ImageFiles]:
        """Given parsed arguments, yield all selected filenames.

        Parameters:
            arguments: The parsed arguments structure.

        Yields:
            ImageFiles objects containing information about groups of selected
            image files.
        """

        # Start with wanting all images
        img_name_list: list[str] = []
        img_filespec_list: list[str] = []
        # TODO It's a problem that these are image filespecs but elsewhere we are expecting label

        # Limit to the user-specific list of images, if any
        if arguments.img_name is not None and flatten_list(arguments.img_name):
            img_name_list = [x.upper() for x in flatten_list(arguments.img_name)]

        # Also limit to the list of images in the FileSpec CSV file, if any
        if arguments.image_filespec_csv:
            for csv_source in arguments.image_filespec_csv:
                # Read via FCPath so the CSV may live behind an http(s)
                # holdings path, consistent with the rest of the dataset layer.
                local_csv = cast(Path, FCPath(csv_source).get_local_path())
                with open(local_csv, encoding='utf-8') as csvfile:
                    csvreader = csv.reader(csvfile)
                    header = next(csvreader)
                    for colnum in range(len(header)):
                        if (
                            header[colnum] == 'Primary File Spec'
                            or header[colnum] == 'primaryfilespec'
                        ):
                            break
                    else:
                        raise ValueError(
                            f'Badly formatted CSV file "{csv_source}" - no Primary File Spec header'
                        )
                    # Header is row 1; data rows start at 2.
                    for rownum, row in enumerate(csvreader, start=2):
                        if colnum >= len(row):
                            # Ragged / short row: skip it rather than aborting
                            # the whole batch on an IndexError.
                            self.logger.warning(
                                'Skipping malformed row %d in CSV "%s": %d column(s), '
                                'need at least %d',
                                rownum,
                                csv_source,
                                len(row),
                                colnum + 1,
                            )
                            continue
                        img_filespec_list.append(row[colnum])

        # Also limit to the list of images in the filelist file, if any
        if arguments.image_file_list:
            for list_source in arguments.image_file_list:
                local_list = cast(Path, FCPath(list_source).get_local_path())
                with open(local_list, encoding='utf-8') as fp:
                    for line in fp:
                        line = line.strip()
                        if len(line) == 0 or line[0] == '#':
                            continue
                        # Ignore anything after the filename token.
                        token = line.split(' ')[0]
                        if not self._img_name_valid(token):
                            raise ValueError(
                                f'Bad filename in filelist file "{list_source}": {line!r}'
                            )
                        img_name_list.append(token)

        first_image_number = arguments.first_image_num
        last_image_number = arguments.last_image_num
        first_volume_number = arguments.first_volume
        last_volume_number = arguments.last_volume
        volumes = None
        if arguments.volumes:
            volumes = [x for y in arguments.volumes for x in y.split(',')]

        yield from self.yield_image_files_index(
            img_start_num=first_image_number,
            img_end_num=last_image_number,
            vol_start=first_volume_number,
            vol_end=last_volume_number,
            volumes=volumes,
            img_name_list=img_name_list,
            img_filespec_list=img_filespec_list,
            has_offset_file=arguments.has_offset_file,
            has_no_offset_file=arguments.has_no_offset_file,
            has_offset_error=arguments.has_offset_error,
            has_no_offset_error=arguments.has_no_offset_error,
            has_offset_spice_error=arguments.has_offset_spice_error,
            has_offset_nonspice_error=arguments.has_offset_nonspice_error,
            # TODO selection_expr=arguments.selection_expr,
            choose_random_images=arguments.choose_random_images,
            arguments=arguments,
        )

    @staticmethod
    def _image_filename_from_label(label_path: Path) -> str | None:
        """Extracts the image data filename from a PDS3 label's ^IMAGE pointer.

        Parameters:
            label_path: Local path to the retrieved PDS3 label file.

        Returns:
            The filename named by the label's ``^IMAGE`` pointer, or None when the
            label has no such pointer or the pointer does not name a file (an
            attached-label record offset like ``^IMAGE = 3``).
        """
        try:
            label_text = label_path.read_text(encoding='utf-8', errors='replace')
        except OSError:
            return None
        match = _LABEL_IMAGE_POINTER_RE.search(label_text)
        if match is None:
            return None
        name = match.group('name')
        # An attached-label pointer gives a record offset, not a filename
        if '.' not in name:
            return None
        return name

    def _image_url_from_label(self, image_url: FCPath, label_path: Path) -> FCPath | None:
        """Resolves the definitive image URL from the label's ^IMAGE pointer.

        Used as the :class:`ImageFile` ``image_url_resolver``: the index-time image
        filespec is an extension-swap guess derived from the label filespec, and
        this callback corrects it against what the label actually points to.

        The comparison is case-insensitive because archive labels do not reliably
        record the on-disk filename case (NH LORRI labels name the ``.fit`` file in
        uppercase while the holdings store it in lowercase).  When the pointer
        names a genuinely different file, the label filename's case convention is
        applied to the pointer name for mono-case names.

        Parameters:
            image_url: The provisional image URL derived by extension swap.
            label_path: Local path to the retrieved PDS3 label file.

        Returns:
            The corrected image URL, or None when the guess already matches the
            label pointer (or the label has no usable pointer).
        """
        name = self._image_filename_from_label(label_path)
        if name is None or name.lower() == image_url.name.lower():
            return None
        label_name = label_path.name
        if name.isupper() and label_name.islower():
            name = name.lower()
        elif name.islower() and label_name.isupper():
            name = name.upper()
        self._logger.debug(
            'Image filespec %s corrected to %s from label %s',
            image_url.name,
            name,
            label_name,
        )
        return image_url.parent / name

    @lru_cache(maxsize=3)  # noqa: B019 # small cache; dataset instances are long-lived
    def _read_pds_table(self, fn: str, columns: tuple[str, ...] | None = None) -> PdsTable:
        """Reads a PDS table file with caching.

        Parameters:
            fn: Path to the PDS table file.
            columns: Optional tuple of column names to read. If None, all columns
                     are read. This is useful for improving performance.

        Returns:
            The parsed PdsTable object.
        """

        return PdsTable(fn, columns=columns, label_method='fast')

    def _yield_image_files_index(self, **kwargs: Any) -> Generator[ImageFile, None, None]:
        """Yield filenames given search criteria using index files.

        This function assumes that the dataset is in a set of PDS3 volumes laid out like
        the PDS Ring-Moon Systems Node archive:

            $(PDS3_HOLDINGS_DIR)/volumes/{volume_set}/{volume}/
            {sub_dirs}/{image}.[IMG,LBL]

        and that the index files are laid out as:

            $(PDS3_HOLDINGS_DIR)/metadata/(volume set)/(volume)/{volume}_index.[lbl,tab]

        Parameters:
            img_start_num: Optional[int] = None,
            img_end_num: Optional[int] = None,
            vol_start: Optional[str] = None,
            vol_end: Optional[str] = None,
            volumes: Optional[list[str]] = None,
            camera: Optional[str] = None,
            img_name_list: Optional[list[str]] = None,
            img_filespec_list: Optional[list[str]] = None,
                Label filespecs; each is resolved to an image base name into local
                ``img_name_filter_list`` (unresolvable entries skipped).
            has_offset_file: bool = False,
            has_no_offset_file: bool = False,
            has_offset_error: bool = False,
            has_no_offset_error: bool = False,
            has_offset_spice_error: bool = False,
            has_offset_nonspice_error: bool = False,
                Results-based filters matching the same-named command-line options;
                see :class:`spindoctor.dataset.results_filter.ResultsFilter`.
            nav_results_root: str | Path | FCPath | None = None,
                Results root for the filters above.  None resolves via the
                arguments, configuration, or NAV_RESULTS_ROOT environment variable.
            results_index_db_url: str | None = None,
                Results index answering the filters above, so that an
                enumeration reads rows instead of walking the results tree and
                reading its documents.  None resolves via the arguments,
                configuration, or NAV_RESULTS_INDEX_DB environment variable when the
                arguments carry a ``results_index_db`` attribute, which is what a
                program that declares ``--results-index-db`` supplies; a caller whose
                arguments carry no such attribute reads the results tree, and
                so does one whose resolved value is the literal ``none``.  A
                level that names the index with an empty value raises
                :class:`spindoctor.dataset.results_filter.SelectionError`.
            choose_random_images: int | None = None,
                When set, a positive count of images to sample uniformly at
                random across the selected volumes.  Must be a positive
                integer; non-positive values raise ValueError.
            max_filenames: Optional[int] = None,
            suffix: Optional[str] = None,
            planets: Optional[str] = None

        Yields:
            ImageFile objects containing information about selected image files.
        """

        kwargs = kwargs.copy()
        img_start_num: int | None = kwargs.pop('img_start_num', None)
        img_end_num: int | None = kwargs.pop('img_end_num', None)
        vol_start: str | None = kwargs.pop('vol_start', None)
        vol_end: str | None = kwargs.pop('vol_end', None)
        volumes: list[str] | None = kwargs.pop('volumes', None)
        camera: str | None = kwargs.pop('camera', None)
        img_name_list: list[str] | None = kwargs.pop('img_name_list', None)
        img_name_filter_list: list[str] | None = kwargs.pop('img_filespec_list', None)
        has_offset_file: bool = kwargs.pop('has_offset_file', False)
        has_no_offset_file: bool = kwargs.pop('has_no_offset_file', False)
        has_offset_error: bool = kwargs.pop('has_offset_error', False)
        has_no_offset_error: bool = kwargs.pop('has_no_offset_error', False)
        has_offset_spice_error: bool = kwargs.pop('has_offset_spice_error', False)
        has_offset_nonspice_error: bool = kwargs.pop('has_offset_nonspice_error', False)
        nav_results_root: str | Path | FCPath | None = kwargs.pop('nav_results_root', None)
        results_index_db_url: str | None = kwargs.pop('results_index_db_url', None)
        choose_random_images: int | None = kwargs.pop('choose_random_images', None)
        if choose_random_images is not None and choose_random_images <= 0:
            raise ValueError(
                f'choose_random_images must be a positive integer, got {choose_random_images}'
            )
        max_filenames: int | None = kwargs.pop('max_filenames', None)
        arguments: argparse.Namespace | None = kwargs.pop('arguments', None)
        additional_index_columns: tuple[str, ...] = kwargs.pop('additional_index_columns', ())

        if len(kwargs) > 0:
            raise ValueError(f'Unexpected keyword arguments: {kwargs}')

        logger = self._logger

        logger.info(f'*** Image number range: {img_start_num} - {img_end_num}')
        logger.info(f'*** Volume range:       {vol_start} - {vol_end}')
        logger.info(f'*** Camera:             {camera}')
        results_filter_flags = {
            'has_offset_file': has_offset_file,
            'has_no_offset_file': has_no_offset_file,
            'has_offset_error': has_offset_error,
            'has_no_offset_error': has_no_offset_error,
            'has_offset_spice_error': has_offset_spice_error,
            'has_offset_nonspice_error': has_offset_nonspice_error,
        }
        active_filter_flags = [name for name, value in results_filter_flags.items() if value]
        if active_filter_flags:
            logger.info('*** Results filters:    %s', ', '.join(active_filter_flags))
        if img_name_list:
            logger.info('*** Explicit image names:')
            for explicit_img_name in img_name_list:
                logger.info(f'        {explicit_img_name}')
        if img_name_filter_list:
            logger.info('*** Explicit image filespecs (resolved to name filters):')
            for explicit_img_filespec in img_name_filter_list:
                logger.info(f'        {explicit_img_filespec}')
        if volumes is not None and volumes != []:
            logger.info('*** Images restricted to volumes:')
            for volume in volumes:
                for vol in volume.split(','):
                    logger.info(f'        {vol}')

        all_volume_names = self._ALL_VOLUME_NAMES
        index_columns = tuple(
            dict.fromkeys(
                self._INDEX_COLUMNS + self._INDEX_CAMERA_COLUMNS + additional_index_columns
            )
        )
        volumes_dir_name = self._VOLUMES_DIR_NAME

        # Restrict volumes to given "volumes" argument
        if volumes is not None:
            for vol in volumes:
                if vol not in all_volume_names:
                    raise ValueError(f'Illegal volume name: {vol}')
            # This keeps the order of the provided volumes
            valid_volumes = [v for v in volumes if v in all_volume_names]
        else:
            valid_volumes = list(all_volume_names)

        # Restrict volumes to given "vol_start" and "vol_end" arguments
        # keeping original order
        vol_start_idx: int | None = None
        vol_end_idx: int | None = None
        if vol_start is not None:
            if vol_start not in all_volume_names:
                raise ValueError(f'Illegal volume name: {vol_start}')
            vol_start_idx = all_volume_names.index(vol_start)
        if vol_end is not None:
            if vol_end not in all_volume_names:
                raise ValueError(f'Illegal volume name: {vol_end}')
            vol_end_idx = all_volume_names.index(vol_end)
        if vol_start_idx is not None and vol_end_idx is not None and vol_start_idx > vol_end_idx:
            raise ValueError(f'vol_start ({vol_start!r}) must not be after vol_end ({vol_end!r})')
        valid_volumes = [
            v
            for v in valid_volumes
            if (
                (vol_start_idx is None or vol_start_idx <= all_volume_names.index(v))
                and (vol_end_idx is None or all_volume_names.index(v) <= vol_end_idx)
            )
        ]

        # Build the results-based filter, if any of its flags is active. A run
        # selecting images that have a document lists the selected volumes once,
        # at construction, so testing an image afterwards is a set lookup; a run
        # selecting images that have none, and an error filter, ask about a batch
        # of candidates instead.
        results_filter: ResultsFilter | None = None
        if any(results_filter_flags.values()):
            resolved_arguments = arguments if arguments is not None else argparse.Namespace()
            if nav_results_root is None:
                nav_results_root = get_nav_results_root(resolved_arguments, self.config)
            if results_index_db_url is None and 'results_index_db' in vars(resolved_arguments):
                # Only a program that declares --results-index-db reads an index, and
                # the presence of the argument is that declaration. The URL
                # resolves from the configuration and the environment as every
                # root does, so an operator who exports one gets it wherever it
                # applies -- but a program whose selection is meant to read
                # files never becomes index-backed because a variable was
                # exported for another one, and a caller that names no argument
                # at all is asking for the tree.
                try:
                    results_index_db_url = get_results_index_db_url(resolved_arguments, self.config)
                except ValueError as exc:
                    # A level that named the index with an empty value is a run
                    # that was configured wrong, and its message already says
                    # which level and what to write; re-raised in the family a
                    # program reporting a refused selection catches, so it reads
                    # as the setting it is rather than as a defect in the walk.
                    raise SelectionError(str(exc)) from exc
            # ResultsFilter accepts the str | Path | FCPath union and normalizes
            # at its boundary.
            results_filter = ResultsFilter(
                valid_volumes,
                nav_results_root,
                logger=logger,
                results_index_db_url=results_index_db_url,
                tuning=get_results_tree_tuning(self.config),
                **results_filter_flags,
            )

        # Closed when the enumeration is done with it, however it ends: an error
        # filter holds its storage open between batches, and a caller that walks
        # away part way through a generator must not leak it.
        try:
            # URLs to the volume raw directory and index directory
            volume_raw_dir_url = self.pds3_holdings_root / volumes_dir_name
            index_dir_url = self.pds3_holdings_root / 'metadata'

            # Validate the image_name_list and img_name_filter_list (from img_filespec_list kwarg)
            if img_name_list:
                for explicit_img_name in img_name_list:
                    if not self._img_name_valid(explicit_img_name):
                        raise ValueError(f'Invalid image name "{explicit_img_name}"')
            if img_name_filter_list:
                new_img_name_filter_list: list[str] = []
                for explicit_img_filespec in img_name_filter_list:
                    try:
                        new_img_name = self._get_img_name_from_label_filespec(explicit_img_filespec)
                    except ValueError as exc:
                        logger.warning(
                            'Skipping explicit image filespec %r: %s',
                            explicit_img_filespec,
                            exc,
                        )
                        continue
                    if new_img_name is None:
                        continue
                    new_img_name_filter_list.append(new_img_name)
                img_name_filter_list = new_img_name_filter_list

            # Optimize the first and last image number based on image_name_list
            # and img_name_filter_list. This is just to improve performance
            if img_name_list:
                img_start_num = max(
                    0 if img_start_num is None else img_start_num,
                    min([self._extract_img_number(x) for x in img_name_list]),
                )
                img_end_num = min(
                    999999999999 if img_end_num is None else img_end_num,
                    max([self._extract_img_number(x) for x in img_name_list]),
                )

            if img_name_filter_list:
                img_start_num = max(
                    0 if img_start_num is None else img_start_num,
                    min([self._extract_img_number(x) for x in img_name_filter_list]),
                )
                img_end_num = min(
                    999999999999 if img_end_num is None else img_end_num,
                    max([self._extract_img_number(x) for x in img_name_filter_list]),
                )

            # Limit the number of returned yields from this method if necessary
            limit_yields = choose_random_images if choose_random_images else None
            if max_filenames is not None:
                if limit_yields is None:
                    limit_yields = max_filenames
                else:
                    limit_yields = min(limit_yields, max_filenames)

            def _read_index_rows(search_vol: str) -> tuple[list[dict[str, Any]], FCPath]:
                """Retrieve and read the index table for a volume.

                Returns:
                    The list of index rows and the index ``.tab`` URL (used only for
                    error messages).
                """
                index_label_url = index_dir_url / self._volume_to_index(search_vol)
                index_tab_url = index_label_url.with_suffix('.tab')
                # This will raise a FileNotFoundError if the index file label or table
                # can't be found
                # TODO Implement actual error handling
                # We have to convert the FCPaths to Posix strings here so that
                # FileCache.retrieve() can use them. Note that if for some reason there was a
                # specific FileCache given for pds3_holdings_root, it will be overriden by
                # self._index_filecache.
                # TODO Needs to return exceptions instead of a single FileNotFoundError
                # so we can tell the user what's actually going on.
                ret = self._index_filecache.retrieve(
                    [index_label_url.as_posix(), index_tab_url.as_posix()]
                )
                index_label_localpath, _ = cast(list[Path], ret)
                index_tab = self._read_pds_table(index_label_localpath, columns=index_columns)
                return index_tab.dicts_by_row(), index_tab_url

            def _row_to_imagefile(
                row: dict[str, Any], search_vol: str, index_tab_url: FCPath
            ) -> tuple[ImageFile | None, bool]:
                """Apply all active filters to one index row.

                Returns:
                    A tuple ``(imagefile, past_end)``. ``imagefile`` is the constructed
                    ``ImageFile`` if the row passes every filter, or None if it is filtered
                    out. ``past_end`` is True if ``img_num`` exceeds ``img_end_num``. Index
                    rows are time-ordered but not strictly monotonic in image number:
                    measured over the full COISS archive (2026-07-12), 16 of 126 volumes
                    contain a few local inversions -- mostly simultaneous NAC/WAC exposure
                    pairs where the wide-angle row appears one count after its narrow-angle
                    partner, plus out-of-order runs of up to ~500 counts in the early
                    cruise volumes (COISS_1001-1003) -- so a True value means only that
                    *this row* is past the end of the range, not that the scan as a whole
                    can stop.
                """
                label_filespec = self._get_label_filespec_from_index(row)
                img_filespec = self._get_image_filespec_from_label_filespec(label_filespec)

                # Get the image name
                try:
                    img_name = self._get_img_name_from_label_filespec(label_filespec)
                except ValueError:
                    logger.error(
                        'IMGNAME: Index file "%s" contains bad Primary File Spec "%s"',
                        index_tab_url,
                        label_filespec,
                    )
                    return None, False
                if img_name is None:
                    return None, False  # Not a name we should process

                # Get the image number and test that it's in range. This runs before the
                # explicit-list filters so a row rejected by those lists still reports
                # past_end, letting the caller stop scanning volumes past the range.
                try:
                    img_num = self._extract_img_number(img_name)
                except ValueError as err:
                    raise ValueError(
                        f'IMGNUM: Index file "{index_tab_url}" contains bad path "{label_filespec}"'
                    ) from err
                if img_end_num is not None and img_num > img_end_num:
                    return None, True
                if img_start_num is not None and img_num < img_start_num:
                    return None, False

                # Check that the image filespec is in the requested list
                if img_name_filter_list and img_name not in img_name_filter_list:
                    return None, False

                # Check that the image name is in the requested list
                if img_name_list:
                    for restrict_name in img_name_list:
                        if img_name.lower().startswith(restrict_name.lower()):
                            break
                    else:
                        return None, False

                # Check that the image meets any additional selection criteria specific
                # to this dataset
                volume_dir_url = volume_raw_dir_url / self._volset_and_volume(search_vol)
                label_url = volume_dir_url / label_filespec
                img_url = volume_dir_url / img_filespec
                if not self._check_additional_image_selection_criteria(
                    img_url.as_posix(), img_name, img_num, arguments
                ):
                    return None, False

                # Check the filters the construction listing settles (a set lookup,
                # no round trips)
                results_path_stub = self._results_path_stub(search_vol, label_filespec)
                if results_filter is not None and not results_filter.passes(results_path_stub):
                    return None, False

                imagefile = ImageFile(
                    image_file_url=img_url,
                    label_file_url=label_url,
                    index_file_row=row,
                    camera=self.camera_from_index_row(row),
                    results_path_stub=results_path_stub,
                    image_url_resolver=self._image_url_from_label,
                )
                return imagefile, False

            if choose_random_images:
                # Uniform random sampling across every selected volume: collect every row
                # passing the cheap filters (index-derived criteria plus the listed
                # results sets) into one pool, shuffle it, then rejection-sample through
                # the batched filter until the requested count is reached. Reading
                # every volume's index is required for cross-volume uniformity (the
                # indexes are cached locally, so repeat runs are cheap); the pool holds
                # one ImageFile per qualifying image, so an unconstrained sample costs
                # memory proportional to the archive's image count.
                pool: list[ImageFile] = []
                for search_vol in valid_volumes:
                    rows, index_tab_url = _read_index_rows(search_vol)
                    all_rows_past_end = bool(rows)
                    for row in rows:
                        imagefile, past_end = _row_to_imagefile(row, search_vol, index_tab_url)
                        if not past_end:
                            all_rows_past_end = False
                        if imagefile is not None:
                            pool.append(imagefile)
                    if all_rows_past_end and self._IMG_NUM_MONOTONIC_ACROSS_VOLUMES:
                        break
                random.shuffle(pool)
                num_yields = 0
                for batch_start in range(0, len(pool), RESULTS_FILTER_BATCH_SIZE):
                    batch = pool[batch_start : batch_start + RESULTS_FILTER_BATCH_SIZE]
                    if results_filter is not None:
                        batch = results_filter.filter_batch(batch)
                    for imagefile in batch:
                        yield imagefile
                        num_yields += 1
                        if limit_yields is not None and num_yields >= limit_yields:
                            return
                return

            # Sequential scanning over the requested volumes in order. Index rows are
            # time-ordered but not strictly monotonic in image number (COISS volumes
            # carry rare local inversions: simultaneous NAC/WAC exposure pairs whose
            # wide-angle row sorts one count late, and out-of-order runs in the early
            # cruise volumes), so a single past-the-end row must not stop the scan;
            # every row is range-filtered individually. When image numbers are
            # monotonic across volumes, the scan stops after the first volume in which
            # every row is past the end of the requested range; otherwise (Voyager,
            # whose FDS counts roll over between encounter volume sets) no
            # image-number-based stop applies and every requested volume is scanned,
            # though a requested result limit can still end the scan early. Accepted
            # images pass through the batched filter in buffered chunks (so one
            # question covers many candidates) while preserving enumeration order; with
            # nothing left to ask per batch the buffer flushes immediately.
            num_yields = 0
            pending: list[ImageFile] = []
            pending_flush_size = (
                RESULTS_FILTER_BATCH_SIZE
                if results_filter is not None and results_filter.needs_batch_filtering
                else 1
            )

            def _flush_pending() -> list[ImageFile]:
                """Run the pending buffer through the filter's batched question.

                Returns:
                    The images of the buffer the filter keeps, in the order the
                    enumeration accepted them.
                """
                nonlocal pending
                batch, pending = pending, []
                if results_filter is not None:
                    batch = results_filter.filter_batch(batch)
                return batch

            for search_vol in valid_volumes:
                rows, index_tab_url = _read_index_rows(search_vol)
                all_rows_past_end = bool(rows)
                for row in rows:
                    imagefile, past_end = _row_to_imagefile(row, search_vol, index_tab_url)
                    if not past_end:
                        all_rows_past_end = False
                    if imagefile is None:
                        continue
                    pending.append(imagefile)
                    if len(pending) < pending_flush_size:
                        continue
                    for filtered in _flush_pending():
                        yield filtered
                        num_yields += 1
                        if limit_yields is not None and num_yields >= limit_yields:
                            return
                if all_rows_past_end and self._IMG_NUM_MONOTONIC_ACROSS_VOLUMES:
                    break
            for filtered in _flush_pending():
                yield filtered
                num_yields += 1
                if limit_yields is not None and num_yields >= limit_yields:
                    return
        finally:
            if results_filter is not None:
                results_filter.close()

    def _check_additional_image_selection_criteria(
        self,
        img_path: str,
        img_name: str,
        img_num: int,
        arguments: argparse.Namespace | None = None,
    ) -> bool:
        """Check additional image selection criteria. Overridden by subclasses.

        Parameters:
            img_path: The path to the image.
            img_name: The name of the image.
            img_num: The number of the image.
            arguments: The parsed arguments.

        Returns:
            True if the image should be processed, False otherwise.
        """

        return True

    def yield_image_files_index(self, **kwargs: Any) -> Generator[ImageFiles, None, None]:
        """Yield filenames given search criteria using index files. Overridden by subclasses.

        Parameters:
            **kwargs: Arbitrary keyword arguments, usually used to restrict the search.

        Yields:
            ImageFiles objects containing information about groups of selected
            image files.
        """

        # closing(): the inner generator holds the results filter, and a caller
        # that stops reading part way leaves this one to be collected rather
        # than closed, so the storage the filter holds open would be released
        # whenever the interpreter got round to it.
        with closing(self._yield_image_files_index(**kwargs)) as imagefiles:
            for imagefile in imagefiles:
                yield ImageFiles(image_files=[imagefile])

    @staticmethod
    def supported_grouping() -> list[str]:
        """Returns the list of supported grouping types.

        Returns:
            The list of supported grouping types.
        """
        return []
