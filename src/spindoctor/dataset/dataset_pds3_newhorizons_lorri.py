from pathlib import Path
from typing import Any, ClassVar, cast

from filecache import FCPath, FileCache

from spindoctor.config import Config
from spindoctor.support.misc import safe_lstrip_zero

from .dataset import ImageFile
from .dataset_pds3 import DataSetPDS3


class DataSetPDS3NewHorizonsLORRI(DataSetPDS3):
    """Implements dataset access for PDS3 New Horizons LORRI (Low-Resolution Imaging
    Experiment) data.

    This class provides specialized functionality for accessing and parsing New
    Horizons LORRI image data stored in PDS3 format.
    """

    _ALL_VOLUME_NAMES = (
        'NHLALO_2001',
        'NHJULO_2001',
        'NHPCLO_2001',
        'NHPELO_2001',
        'NHKCLO_2001',
        'NHKELO_2001',
        'NHK2LO_2001',
    )
    _INDEX_COLUMNS = ('FILE_SPECIFICATION_NAME',)
    _INDEX_CAMERA_COLUMNS = ('INSTRUMENT_ID',)
    _INDEX_CAMERA_MAP: ClassVar[dict[str, str]] = {'LORRI': 'LORRI'}
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
        # Intentionally lowercase only
        if not filespec.endswith(('_sci.lbl', '_eng.lbl')):
            raise ValueError(
                f'Bad Primary File Spec "{filespec}" - expected "_sci.lbl" or "_eng.lbl"'
            )
        return filespec

    @staticmethod
    def _get_image_filespec_from_label_filespec(label_filespec: str) -> str:
        """Extracts the image file specification from a label file specification.

        Parameters:
            label_filespec: The label file specification string to parse.

        Returns:
            The image file specification string.
        """
        # Intentionally lowercase only
        return label_filespec.replace('.lbl', '.fit')

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
        # The range directory is "ddddddd_ddddddd" (two 7-digit request IDs
        # joined by '_'), so it is exactly 15 chars with '_' at index 8.
        if len(range_dir) != 15 or range_dir[8] != '_':
            raise ValueError(
                f'Bad Primary File Spec "{filespec}" - expected "DATA/ddddddd_ddddddd"'
            )
        # Intentionally lowercase only
        if not img_name.endswith(('_sci.lbl', '_eng.lbl')):
            raise ValueError(
                f'Bad Primary File Spec "{filespec}" - expected "_sci.lbl" or "_eng.lbl"'
            )
        return img_name[:14]

    @staticmethod
    def _img_name_valid(img_name: str) -> bool:
        """True if an image name is valid for this instrument.

        Parameters:
            img_name: The name of the image. Can be just the image name, the image filename,
                or the full file spec.

        Returns:
            True if the image name is valid for this instrument, False otherwise.
        """

        img_name = img_name.upper()

        # lor_0297859350
        if len(img_name) != 14 or not img_name.startswith('LOR_'):
            return False
        try:
            _ = int(safe_lstrip_zero(img_name[4:14]))
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

        if not DataSetPDS3NewHorizonsLORRI._img_name_valid(img_name):
            raise ValueError(f'Invalid image name "{img_name}"')

        return int(safe_lstrip_zero(img_name[4:14]))

    @staticmethod
    def _volset_and_volume(volume: str) -> str:
        """Get the volset and volume name.

        Parameters:
            volume: The volume name.
        """
        return f'NHxxLO_xxxx/{volume}'

    @staticmethod
    def _volume_to_index(volume: str) -> str:
        """Get the index file name for a volume.

        Parameters:
            volume: The volume name.
        """
        return f'NHxxLO_xxxx/{volume}/{volume}_index.lbl'

    @staticmethod
    def _results_path_stub(volume: str, filespec: str) -> str:
        """Get the results path stub for an image filespec.

        Parameters:
            volume: The volume name.
            filespec: The filespec of the image.
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
        """Initializes a New Horizons LORRI dataset handler.

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
        nhlorri_config = self.config.pds4.get('nhlorri', {})
        if 'template_dir' in nhlorri_config:
            template_dir = str(nhlorri_config['template_dir'])

        if template_dir is None:
            template_dir = 'newhorizons_lorri_1.0'

        if Path(template_dir).is_absolute():
            return template_dir

        pds4_templates_dir = Path(__file__).resolve().parent.parent / 'cli' / 'pds4' / 'templates'
        return str(pds4_templates_dir / template_dir)

    def pds4_bundle_name(self) -> str:
        """Returns bundle name for PDS4 bundle generation."""
        nhlorri_config = self.config.pds4.get('nhlorri', {})
        if 'bundle_name' in nhlorri_config:
            return str(nhlorri_config['bundle_name'])
        return 'newhorizons_lorri_backplanes_rsfrench2027'

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
