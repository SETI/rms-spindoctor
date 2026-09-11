from pathlib import Path
from typing import Any, ClassVar, cast

from filecache import FCPath, FileCache

from spindoctor.config import Config
from spindoctor.support.misc import safe_lstrip_zero

from .dataset import ImageFile
from .dataset_pds3 import DataSetPDS3


class DataSetPDS3VoyagerISS(DataSetPDS3):
    """Implements dataset access for PDS3 Voyager ISS (Imaging Science Subsystem) data.

    This class provides specialized functionality for accessing and parsing Voyager
    ISS image data stored in PDS3 format.
    """

    _MIN_5xxx_VOL1 = 5101
    _MAX_5xxx_VOL1 = 5120
    _MIN_5xxx_VOL2 = 5201
    _MAX_5xxx_VOL2 = 5214
    _MIN_6xxx_VOL1 = 6101
    _MAX_6xxx_VOL1 = 6121
    _MIN_6xxx_VOL2 = 6201
    _MAX_6xxx_VOL2 = 6215
    _MIN_7xxx_VOL2 = 7201
    _MAX_7xxx_VOL2 = 7207
    _MIN_8xxx_VOL2 = 8201
    _MAX_8xxx_VOL2 = 8210

    _ALL_VOLUME_NAMES = tuple(
        f'VGISS_{x:04d}'
        for x in list(range(_MIN_5xxx_VOL1, _MAX_5xxx_VOL1 + 1))
        + list(range(_MIN_5xxx_VOL2, _MAX_5xxx_VOL2 + 1))
        + list(range(_MIN_6xxx_VOL1, _MAX_6xxx_VOL1 + 1))
        + list(range(_MIN_6xxx_VOL2, _MAX_6xxx_VOL2 + 1))
        + list(range(_MIN_7xxx_VOL2, _MAX_7xxx_VOL2 + 1))
        + list(range(_MIN_8xxx_VOL2, _MAX_8xxx_VOL2 + 1))
    )
    _INDEX_COLUMNS = ('FILE_SPECIFICATION_NAME',)
    # Voyager indexes carry no INSTRUMENT_ID; the name spells the camera out.
    _INDEX_CAMERA_COLUMNS = ('INSTRUMENT_NAME',)
    _INDEX_CAMERA_MAP: ClassVar[dict[str, str]] = {
        'NARROW ANGLE CAMERA': 'NAC',
        'WIDE ANGLE CAMERA': 'WAC',
    }
    _VOLUMES_DIR_NAME = 'volumes'
    # FDS counts restart per spacecraft/encounter (the volume order interleaves VG1
    # and VG2, and VG2's Neptune counts roll over below VG1's Jupiter counts), so an
    # image-number range can match frames in any volume and no volume-level early
    # exit is possible.
    _IMG_NUM_MONOTONIC_ACROSS_VOLUMES = False

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
        if not filespec.endswith('.LBL'):
            raise ValueError(f'Bad Primary File Spec "{filespec}" - expected ".LBL"')
        return filespec

    @staticmethod
    def _get_image_filespec_from_label_filespec(label_filespec: str) -> str:
        """Extracts the image file specification from a label file specification.

        Parameters:
            label_filespec: The label file specification string to parse.

        Returns:
            The image file specification string.
        """
        return label_filespec.replace('.LBL', '.IMG')

    @staticmethod
    def _get_img_name_from_label_filespec(filespec: str) -> str | None:
        """Extract the image name (with no extension) from a file specification.

        Parameters:
            filespec: The file specification string to parse.

        Raises:
            ValueError: If the file specification format is invalid.
        """

        parts = filespec.split('/')
        if len(parts) != 3:
            raise ValueError(f'Bad Primary File Spec "{filespec}" - expected 3 directory levels')
        if parts[0].upper() != 'DATA':
            raise ValueError(f'Bad Primary File Spec "{filespec}" - expected "DATA"')
        range_dir = parts[1]
        img_name = parts[2]
        if len(range_dir) != 8 or range_dir[0] != 'C':
            raise ValueError(f'Bad Primary File Spec "{filespec}" - expected "Cddddddd"')
        # Only the geometrically-corrected (_GEOMED) products are navigated;
        # _CALIB / _RAW products return None (skipped) by design.
        if not img_name.endswith('_GEOMED.LBL'):
            return None
        return img_name.rsplit('_GEOMED')[0]

    @staticmethod
    def _img_name_valid(img_name: str) -> bool:
        """True if an image name is valid for this instrument.

        Parameters:
            img_name: The name of the image.

        Returns:
            True if the image name is valid for this instrument, False otherwise.
        """

        img_name = img_name.upper()

        # Accept the canonical short name ``Cddddddd`` as well as the product
        # file names users naturally list (e.g. ``C1234567_GEOMED``,
        # ``C1234567_CALIB``, ``C1234567_GEOMED.IMG``): strip any extension and
        # product suffix, then validate the ``Cddddddd`` core.
        core = img_name.split('.', 1)[0].split('_', 1)[0]
        if len(core) != 8 or core[0] != 'C':
            return False
        try:
            _ = int(safe_lstrip_zero(core[1:]))
        except ValueError:
            return False

        return True

    @staticmethod
    def _extract_img_number(img_name: str) -> int:
        """Extract the image number from an image name.

        Parameters:
            img_name: The name of the image.

        Returns:
            The image number.

        Raises:
            ValueError: If the image name format is invalid.
        """

        if not DataSetPDS3VoyagerISS._img_name_valid(img_name):
            raise ValueError(f'Invalid image name "{img_name}"')

        return int(safe_lstrip_zero(img_name[1:]))

    @staticmethod
    def _volset_and_volume(volume: str) -> str:
        """Get the volset and volume name.

        Parameters:
            volume: The volume name.
        """
        return f'VGISS_{volume[6]}xxx/{volume}'

    @staticmethod
    def _volume_to_index(volume: str) -> str:
        """Get the index file name for a volume.

        Parameters:
            volume: The volume name.
        """
        return f'VGISS_{volume[6]}xxx/{volume}/{volume}_index.lbl'

    @staticmethod
    def _results_path_stub(volume: str, filespec: str) -> str:
        """Get the results path stub for an image filespec.

        Parameters:
            volume: The volume name.
            filespec: The filespec of the image.
        """
        return str(Path(f'{volume}/{filespec}').with_suffix(''))

    def __init__(
        self,
        pds3_holdings_root: str | Path | FCPath | None = None,
        *,
        index_filecache: FileCache | None = None,
        pds3_holdings_filecache: FileCache | None = None,
        config: Config | None = None,
    ) -> None:
        """Initializes a Voyager ISS dataset handler.

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

    def pds4_bundle_template_dir(self) -> str:
        """Returns absolute path to template directory for PDS4 bundle generation."""
        template_dir = None
        vgiss_config = self.config.pds4.get('vgiss', {})
        if 'template_dir' in vgiss_config:
            template_dir = str(vgiss_config['template_dir'])

        if template_dir is None:
            template_dir = 'voyager_iss_1.0'

        if Path(template_dir).is_absolute():
            return template_dir

        pds4_templates_dir = Path(__file__).resolve().parent.parent / 'cli' / 'pds4' / 'templates'
        return str(pds4_templates_dir / template_dir)

    def pds4_bundle_name(self) -> str:
        """Returns bundle name for PDS4 bundle generation."""
        vgiss_config = self.config.pds4.get('vgiss', {})
        if 'bundle_name' in vgiss_config:
            return str(vgiss_config['bundle_name'])
        return 'voyager_iss_backplanes_rsfrench2027'

    @staticmethod
    def pds4_bundle_path_for_image(image_name: str) -> str:
        """Maps image name to bundle directory path."""
        raise NotImplementedError

    def pds4_path_stub(self, image_file: ImageFile) -> str:
        """Returns PDS4 path stub for bundle directory structure."""
        raise NotImplementedError

    def pds4_lid_part_to_image_name(self, lid_part: str) -> str:
        """Returns the image name for the given LID part.

        Parameters:
            lid_part: The LID part (an on-disk product filename stem).

        Returns:
            The image name that produced the given LID part.
        """
        raise NotImplementedError

    def pds4_image_name_to_browse_lid(self, image_name: str) -> str:
        """Returns the browse LID for the given image name.

        Parameters:
            image_name: The image name to convert to a browse LID.

        Returns:
            The browse LID.
        """
        raise NotImplementedError

    def pds4_image_name_to_browse_lidvid(self, image_name: str) -> str:
        """Returns the browse LIDVID for the given image name.

        Parameters:
            image_name: The image name to convert to a browse LIDVID.

        Returns:
            The browse LIDVID.
        """
        raise NotImplementedError

    def pds4_image_name_to_data_lid(self, image_name: str) -> str:
        """Returns the data LID for the given image name.

        Parameters:
            image_name: The image name to convert to a data LID.

        Returns:
            The data LID.
        """
        raise NotImplementedError

    def pds4_image_name_to_data_lidvid(self, image_name: str) -> str:
        """Returns the data LIDVID for the given image name.

        Parameters:
            image_name: The image name to convert to a data LIDVID.

        Returns:
            The data LIDVID.
        """
        raise NotImplementedError

    def pds4_template_variables(
        self,
        *,
        image_file: ImageFile,
        nav_metadata: dict[str, Any],
        backplane_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Returns template variables for PDS4 label generation."""
        raise NotImplementedError
