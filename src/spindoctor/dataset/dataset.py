import argparse
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal, cast

from filecache import FCPath

from spindoctor.config import MAIN_LOGGER, Config, LogRole
from spindoctor.support.nav_base import NavBase

Pds4Pass = Literal['labels', 'summary']
"""Which pass of PDS4 bundle generation a set of templates belongs to:
``labels`` for the per-image pass, ``summary`` for the collection and index
pass."""


@dataclass
class ImageFile:
    """Represents a single image file with its metadata and lazy-loaded paths.

    Attributes:
        image_file_url: Remote URL for the image file.
        label_file_url: Remote URL for the label file.
        results_path_stub: Local path stub for storing results.
        index_file_row: Optional metadata from index files.
        camera: Optional name of the camera that took the image, read from
            the index row when the file was enumerated.  Known without SPICE
            and without opening the image, and uses the same names as
            ``ObsInst.camera``.  None when the image was not enumerated
            from an index, or its index row names no recognized camera.
        nav_record: The navigation record already read for this image while the
            run's selection was being made, in the shape the navigator writes
            it -- the same shape a pointing source hands back.  A run that
            selected its images on what their navigation documents record has
            read and parsed each kept image's document to decide, so the record
            travels with the image and the per-image stage reads no document of
            its own.  None whenever nothing has read one: a run whose selection
            asked nothing of the navigation results, one answered out of a
            results index (where a record is rebuilt from a column set of its
            consumer's own), and an image built from a cloud task file, which
            carries an image's two URLs, its stub and its index row and no
            record.  A consumer that meets None reads the record from its own
            storage.
        extra_params: Optional extra parameters that will be passed to the observation
            class's from_file method when the file is read.
        image_url_resolver: Optional callable ``(image_file_url, label_file_path) ->
            FCPath | None`` that determines the definitive image URL from the label
            contents.  When set, it is invoked (at most once) before the image file
            is first retrieved; a non-None return value replaces ``image_file_url``.
            ``image_file_url`` is then a provisional guess until
            :meth:`resolve_image_url` has run.
    """

    image_file_url: FCPath
    label_file_url: FCPath
    results_path_stub: str
    index_file_row: dict[str, Any] = field(default_factory=dict)
    camera: str | None = None
    nav_record: dict[str, Any] | None = None
    extra_params: dict[str, Any] = field(default_factory=dict)
    image_url_resolver: Callable[[FCPath, Path], FCPath | None] | None = None
    _image_file_path: Path | None = None
    _label_file_path: Path | None = None

    @property
    def image_file_name(self) -> str:
        return self.image_file_url.name

    @property
    def label_file_name(self) -> str:
        return self.label_file_url.name

    @property
    def image_file_path(self) -> Path:
        """Local path to the image file, downloading and memoizing on first use.

        Not thread-safe: a single ``ImageFile`` must be used by one thread at a
        time.  The lazy memoization is unsynchronized, so concurrent first
        access from two threads can race and download twice.  Enumerate-then-
        dispatch one ``ImageFile`` per worker thread.
        """
        if self._image_file_path is None:
            self._image_file_path = cast(Path, self.resolve_image_url().get_local_path())
        return self._image_file_path

    def resolve_image_url(self) -> FCPath:
        """The definitive image URL, consulting the label when a resolver is set.

        When ``image_url_resolver`` is set, the label file is retrieved and the
        resolver maps the label contents to the correct image URL, which replaces
        ``image_file_url``.  The resolver runs at most once; subsequent calls
        return the memoized URL.

        A resolver failure (typically an unretrievable label) falls back to the
        current ``image_file_url`` guess with a logged warning: resolution runs
        before the pipeline's per-image error boundary is open, and the guess is
        usable whenever the image file itself is retrievable.

        Returns:
            The image file URL, corrected from the label contents when a resolver
            is set and reports a different filename.
        """
        if self.image_url_resolver is not None:
            resolver, self.image_url_resolver = self.image_url_resolver, None
            try:
                resolved = resolver(self.image_file_url, self.label_file_path)
            except OSError as exc:
                MAIN_LOGGER.warning(
                    'Image URL resolution from label %s failed (%s); keeping %s',
                    self.label_file_url.as_posix(),
                    exc,
                    self.image_file_url.as_posix(),
                )
                resolved = None
            if resolved is not None:
                self.image_file_url = resolved
        return self.image_file_url

    def retrieve_image_file(self) -> Path:
        return cast(Path, self.resolve_image_url().retrieve())

    @property
    def label_file_path(self) -> Path:
        """Local path to the label file, downloading and memoizing on first use.

        Not thread-safe; see :attr:`image_file_path` for the single-thread-per-
        ``ImageFile`` expectation.
        """
        if self._label_file_path is None:
            self._label_file_path = cast(Path, self.label_file_url.get_local_path())
        return self._label_file_path

    def retrieve_label_file(self) -> Path:
        return cast(Path, self.label_file_url.retrieve())


@dataclass
class ImageFiles:
    """A collection of ImageFile objects that behaves like a sequence.

    Supports iteration, indexing, and length operations on the wrapped image files.
    """

    image_files: list[ImageFile]

    def __iter__(self) -> Iterator[ImageFile]:
        return iter(self.image_files)

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> ImageFile:
        return self.image_files[idx]


class DataSet(ABC, NavBase):
    """Enumerates the images a run will process.

    Class attributes:
        log_role: ``LogRole.MAIN``.  Enumeration spans the whole run rather
            than belonging to any one image, so its records go to the run's
            log; routing them to the image logger would file them under
            whichever image happened to be open, or none at all.
    """

    log_role: ClassVar[LogRole] = LogRole.MAIN

    def __init__(self, *, config: Config | None = None) -> None:
        """Initializes a dataset.

        Parameters:
            config: Configuration object to use. If None, uses DEFAULT_CONFIG.
        """
        super().__init__(config=config)

    @staticmethod
    @abstractmethod
    def _img_name_valid(img_name: str) -> bool:
        """Validates whether the provided image name follows the dataset's naming convention.

        Parameters:
            img_name: The image name to validate.

        Returns:
            True if the image name is valid for this dataset, False otherwise.
        """
        ...

    @staticmethod
    @abstractmethod
    def add_selection_arguments(
        cmdparser: argparse.ArgumentParser,
        group: argparse._ArgumentGroup | None = None,
    ) -> None:
        """Adds dataset-specific command-line arguments for image selection.

        Parameters:
            cmdparser: The argument parser to add arguments to.
            group: Optional argument group to add arguments to. If None, creates a new group.
        """
        ...

    @abstractmethod
    def yield_image_files_from_arguments(
        self, arguments: argparse.Namespace
    ) -> Iterator[ImageFiles]:
        """Yields image filenames based on provided command-line arguments.

        Parameters:
            arguments: The parsed arguments structure.

        Yields:
            Information about the selected files in groups as ImageFiles objects.
        """
        ...

    @abstractmethod
    def yield_image_files_index(self, **kwargs: Any) -> Iterator[ImageFiles]:
        """Yields image filenames based on index information.

        Parameters:
            **kwargs: Arbitrary keyword arguments, usually used to restrict the search.

        Yields:
            Information about the selected files in groups as ImageFiles objects.
        """
        ...

    @staticmethod
    @abstractmethod
    def supported_grouping() -> list[str]:
        """Returns the list of supported grouping types.

        Returns:
            The list of supported grouping types.
        """
        ...

    def pds4_bundle_template_dir(self) -> str:
        """Returns absolute path to template directory for PDS4 bundle generation.

        Checks config section pds4.{dataset_name}.template_dir first, then allows override.
        If just a name is given, it is relative to the pds4/templates directory.
        If a full path is given, it will be left as absolute.

        Returns:
            Absolute path to template directory
            (e.g., "/path/to/pds4/templates/cassini_iss_saturn_1.0").
        """
        # We don't make PDS4 methods as @abstractmethod because it's possible to make
        # a DataSet that doesn't support PDS4 bundle generation
        raise NotImplementedError

    def pds4_bundle_name(self) -> str:
        """Returns bundle name for PDS4 bundle generation.

        Checks config section pds4.{dataset_name}.bundle_name first, then allows override.

        Returns:
            Bundle name (e.g., "cassini_iss_saturn_backplanes_rsfrench2027").
        """
        # We don't make PDS4 methods as @abstractmethod because it's possible to make
        # a DataSet that doesn't support PDS4 bundle generation
        raise NotImplementedError

    def pds4_required_templates(self, pds4_pass: Pds4Pass) -> list[str]:
        """Returns the template filenames one bundle pass must find for this dataset.

        Each pass checks these before it processes anything, in the directory
        :meth:`pds4_bundle_template_dir` names, and refuses to run when one of
        them is not there.  Every product of a pass renders from the same
        directory, so a template that is missing is missing for every image, and
        saying so once is the whole of the report.

        Parameters:
            pds4_pass: Which pass's templates to name: ``labels`` for the
                per-image pass, ``summary`` for the collection and index pass.

        Returns:
            The filenames, relative to the template directory.
        """
        # We don't make PDS4 methods as @abstractmethod because it's possible to make
        # a DataSet that doesn't support PDS4 bundle generation
        raise NotImplementedError

    @staticmethod
    def pds4_bundle_path_for_image(image_name: str) -> str:
        """Maps image name to bundle directory path.

        Parameters:
            image_name: The image name to map.

        Returns:
            Bundle directory path relative to bundle root (e.g., "1234xxxxxx/123456xxxx").
        """
        # We don't make PDS4 methods as @abstractmethod because it's possible to make
        # a DataSet that doesn't support PDS4 bundle generation
        raise NotImplementedError

    def pds4_path_stub(self, image_file: ImageFile) -> str:
        """Returns PDS4 path stub for bundle directory structure.

        Parameters:
            image_file: The image file to generate path stub for.

        Returns:
            Path stub relative to bundle root (e.g., "1234xxxxxx/123456xxxx/1234567890w").
        """
        # We don't make PDS4 methods as @abstractmethod because it's possible to make
        # a DataSet that doesn't support PDS4 bundle generation
        raise NotImplementedError

    def pds4_lid_part_to_image_name(self, lid_part: str) -> str:
        """Returns the image name for the given LID part.

        Inverse of the image-name transformation baked into
        :meth:`pds4_path_stub` and the ``pds4_image_name_to_*`` builders. The
        bundle stores each product under a filename whose stem is the LID part
        (e.g. ``1234567890w``); recovering the original image name lets bundle
        scanners round-trip that stem back through the canonical LID builders
        instead of re-applying the transform.

        Parameters:
            lid_part: The LID part (an on-disk product filename stem).

        Returns:
            The image name that produced the given LID part.
        """
        # We don't make PDS4 methods as @abstractmethod because it's possible to make
        # a DataSet that doesn't support PDS4 bundle generation
        raise NotImplementedError

    def pds4_image_name_to_browse_lid(self, image_name: str) -> str:
        """Returns the browse LID for the given image name.

        Parameters:
            image_name: The image name to convert to a browse LID.

        Returns:
            The browse LID.
        """
        # We don't make PDS4 methods as @abstractmethod because it's possible to make
        # a DataSet that doesn't support PDS4 bundle generation
        raise NotImplementedError

    def pds4_image_name_to_browse_lidvid(self, image_name: str) -> str:
        """Returns the browse LIDVID for the given image name.

        Parameters:
            image_name: The image name to convert to a browse LID.

        Returns:
            The browse LID.
        """
        # We don't make PDS4 methods as @abstractmethod because it's possible to make
        # a DataSet that doesn't support PDS4 bundle generation
        raise NotImplementedError

    def pds4_image_name_to_data_lid(self, image_name: str) -> str:
        """Returns the data LID for the given image name.

        Parameters:
            image_name: The image name to convert to a data LID.

        Returns:
            The data LID.
        """
        # We don't make PDS4 methods as @abstractmethod because it's possible to make
        # a DataSet that doesn't support PDS4 bundle generation
        raise NotImplementedError

    def pds4_image_name_to_data_lidvid(self, image_name: str) -> str:
        """Returns the data LIDVID for the given image name.

        Parameters:
            image_name: The image name to convert to a data LID.

        Returns:
            The data LID.
        """
        # We don't make PDS4 methods as @abstractmethod because it's possible to make
        # a DataSet that doesn't support PDS4 bundle generation
        raise NotImplementedError

    def pds4_template_variables(
        self,
        *,
        image_file: ImageFile,
        nav_metadata: dict[str, Any],
        backplane_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Returns template variables for PDS4 label generation.

        Parameters:
            image_file: The image file being processed.
            nav_metadata: Navigation metadata dictionary (as read from offset_metadata
                JSON file).
            backplane_metadata: Backplane metadata dictionary (created from backplane FITS
                file).

        Returns:
            Dictionary mapping variable names to values for template substitution.
        """
        # We don't make PDS4 methods as @abstractmethod because it's possible to make
        # a DataSet that doesn't support PDS4 bundle generation
        raise NotImplementedError
