from pathlib import Path
from typing import Any, ClassVar, cast

from filecache import FCPath, FileCache

from spindoctor.config import Config
from spindoctor.support.misc import safe_lstrip_zero

from .dataset import ImageFile
from .dataset_pds3 import DataSetPDS3


class DataSetPDS3GalileoSSI(DataSetPDS3):
    """Implements dataset access for PDS3 Galileo SSI (Solid State Imager) data.

    This class provides specialized functionality for accessing and parsing Galileo
    SSI image data stored in PDS3 format.
    """

    _MIN_VOL = 2
    _MAX_VOL = 23
    _ALL_VOLUME_NAMES = tuple(f'GO_{x:04d}' for x in range(_MIN_VOL, _MAX_VOL + 1))
    _INDEX_COLUMNS = ('FILE_SPECIFICATION_NAME',)
    _INDEX_CAMERA_COLUMNS = ('INSTRUMENT_ID',)
    _INDEX_CAMERA_MAP: ClassVar[dict[str, str]] = {'SSI': 'SSI'}
    _VOLUMES_DIR_NAME = 'volumes'

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
        if not filespec.endswith('.LBL'):
            raise ValueError(f'Bad Primary File Spec "{filespec}" - expected ".LBL"')
        # Intentionally uppercase only
        return filespec

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
            The image name if the filespec names an image this dataset
            navigates. None if the filespec is well formed and names something
            else, which the caller drops without comment.

        Raises:
            ValueError: If the file specification format is invalid.
        """

        parts = filespec.split('/')
        if parts[0].upper().startswith('GO_'):
            parts = parts[1:]
        if not (2 <= len(parts) <= 4):
            raise ValueError(
                f'Bad Primary File Spec "{filespec}" - expected 2 to 4 directory levels'
            )
        if parts[0].upper() == 'REDO':
            parts = parts[1:]
        if parts[0].upper() in (
            'RAW_CAL',
            'VENUS',
            'EARTH',
            'MOON',
            'GASPRA',
            'IDA',
            'SL9',
            'EMCONJ',
            'GOPEX',
        ):
            img_name = parts[1]
        elif parts[0].upper() in (
            'C3',
            'C9',
            'C10',
            'C20',
            'C21',
            'C22',
            'C30',
            'E4',
            'E6',
            'E11',
            'E12',
            'E14',
            'E15',
            'E17',
            'E18',
            'E19',
            'E26',
            'G1',
            'G2',
            'G7',
            'G8',
            'G28',
            'G29',
            'I24',
            'I25',
            'I27',
            'I31',
            'I32',
            'I33',
            'J0',
        ):
            img_name = parts[2]
        else:
            raise ValueError(f'Bad Primary File Spec "{filespec}" - bad target directory')
        # Intentionally uppercase only
        if not img_name.endswith('.LBL'):
            return None
        img_name = img_name.rsplit('.', maxsplit=1)[0]
        if not DataSetPDS3GalileoSSI._img_name_valid(img_name):
            # Only the R and S products are navigated, and a name outside that
            # rule carries no image number for the enumeration to read.  The
            # archive holds such names: GO_0016's SL9 directory stores a G
            # product beside the R product of each of its thirteen images, a
            # second representation of an image the enumeration already has.
            return None
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

        # Cdddddddddd[RS]
        if len(img_name) != 12 or img_name[0] != 'C' or img_name[11] not in 'RS':
            return False
        try:
            _ = int(safe_lstrip_zero(img_name[1:11]))
        except ValueError:
            return False

        return True

    @staticmethod
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

        if not DataSetPDS3GalileoSSI._img_name_valid(img_name):
            raise ValueError(f'Invalid image name "{img_name}"')

        return int(safe_lstrip_zero(img_name[1:11]))

    @staticmethod
    def _volset_and_volume(volume: str) -> str:
        """Get the volset and volume name.

        Parameters:
            volume: The volume name.

        Returns:
            The volset and volume name.
        """
        return f'GO_0xxx/{volume}'

    @staticmethod
    def _volume_to_index(volume: str) -> str:
        """Get the index file name for a volume.

        Parameters:
            volume: The volume name.

        Returns:
            The index file name.
        """
        # Intentionally lowercase only
        return f'GO_0xxx/{volume}/{volume}_index.lbl'

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

    # Public methods

    def __init__(
        self,
        pds3_holdings_root: str | Path | FCPath | None = None,
        *,
        index_filecache: FileCache | None = None,
        pds3_holdings_filecache: FileCache | None = None,
        config: Config | None = None,
    ) -> None:
        """Initializes a Galileo SSI dataset handler.

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
        gossi_config = self.config.pds4.get('gossi', {})
        if 'template_dir' in gossi_config:
            template_dir = str(gossi_config['template_dir'])

        if template_dir is None:
            template_dir = 'galileo_ssi_1.0'

        if Path(template_dir).is_absolute():
            return template_dir

        pds4_templates_dir = Path(__file__).resolve().parent.parent / 'cli' / 'pds4' / 'templates'
        return str(pds4_templates_dir / template_dir)

    def pds4_bundle_name(self) -> str:
        """Returns bundle name for PDS4 bundle generation."""
        gossi_config = self.config.pds4.get('gossi', {})
        if 'bundle_name' in gossi_config:
            return str(gossi_config['bundle_name'])
        return 'galileo_ssi_backplanes_rsfrench2027'

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
