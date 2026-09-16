"""The files the labels pass reads for one image, and what a selected image has of them.

The labels pass reads four files for each image, each at the image's results path stub:
its navigation document and its summary PNG under the navigation results root, and its
backplane FITS and its backplane metadata under the backplane results root.
:func:`image_inputs` is where the readers of the bundle passes take those paths from:
the navigation document's from the navigation records' own rule,
:func:`~spindoctor.nav_records.document.document_path`, and the other three from the
rules the navigation and the backplane stages write them by, which those stages state
where they write and expose no function for.  ``sd_create_bundle labels --check-only``
reports, through :func:`report_image_inputs`, whether each selected image has all four
and whether its navigation succeeded, so that a cohort can be chosen before a label is
written.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from filecache import FCPath

from spindoctor.dataset.dataset import ImageFile
from spindoctor.nav_records import document_path, read_document

NAVIGATION_SUCCESS = 'success'
"""The ``status`` a navigation document records for a navigation that succeeded."""


@dataclass(frozen=True)
class ImageInputs:
    """Where the four files the labels pass reads for one image are.

    Attributes:
        navigation_document: The record of the image's navigation, where
            :func:`~spindoctor.nav_records.document.document_path` puts it:
            ``<navigation results root>/<stub>_metadata.json``.
        summary_png: ``<navigation results root>/<stub>_summary.png``, which the image's
            browse product copies.
        backplane_fits: ``<backplane results root>/<stub>_backplanes.fits``, which the
            image's data product copies.
        backplane_metadata: ``<backplane results root>/<stub>_backplane_metadata.json``,
            the statistics of the image's backplanes.
    """

    navigation_document: FCPath
    summary_png: FCPath
    backplane_fits: FCPath
    backplane_metadata: FCPath


def image_inputs(
    results_path_stub: str,
    *,
    nav_results_root: str | Path | FCPath,
    backplane_results_root: str | Path | FCPath,
) -> ImageInputs:
    """Return where the four files the labels pass reads for one image are.

    Parameters:
        results_path_stub: The image's results path stub.
        nav_results_root: The root holding the navigation documents and summary PNGs.
        backplane_results_root: The root holding the backplane FITS files and metadata.

    Returns:
        The four paths.
    """
    nav_root = FCPath(nav_results_root)
    backplane_root = FCPath(backplane_results_root)
    return ImageInputs(
        navigation_document=document_path(nav_root, results_path_stub),
        summary_png=nav_root / f'{results_path_stub}_summary.png',
        backplane_fits=backplane_root / f'{results_path_stub}_backplanes.fits',
        backplane_metadata=backplane_root / f'{results_path_stub}_backplane_metadata.json',
    )


def navigation_record(image_file: ImageFile, inputs: ImageInputs) -> dict[str, Any] | None:
    """Return the record of an image's navigation.

    The record the enumeration already read, when there is one: a run whose selection
    names an error filter has read and parsed the document of every image it kept, and
    the enumeration hands that record on with the image.  Otherwise the navigation
    document, read.

    Parameters:
        image_file: The image.
        inputs: Where the files the labels pass reads for it are.

    Returns:
        The record, or None when there is no navigation document, which means the image
        was never navigated.

    Raises:
        ValueError: If the navigation document does not hold a JSON object, as
            :func:`~spindoctor.nav_records.document.read_document` refuses it.
    """
    if image_file.nav_record is not None:
        return image_file.nav_record
    try:
        return read_document(inputs.navigation_document)
    except FileNotFoundError:
        return None


def navigation_succeeded(record: Mapping[str, Any]) -> bool:
    """Say whether a navigation record records a navigation that succeeded.

    Parameters:
        record: The record.

    Returns:
        True when its ``status`` is :data:`NAVIGATION_SUCCESS`.
    """
    return record.get('status') == NAVIGATION_SUCCESS


@dataclass(frozen=True)
class InputsReport:
    """What one selected image has of the files the labels pass reads.

    Attributes:
        results_path_stub: The image, by its results path stub.
        navigation_document: Whether its navigation document is there.
        summary_png: Whether its summary PNG is there.
        backplane_fits: Whether its backplane FITS is there.
        backplane_metadata: Whether its backplane metadata is there.
        navigation_succeeded: Whether its navigation record records a navigation that
            succeeded.
        navigation_status: The ``status`` its navigation record gives, or None when it
            has no record.
    """

    results_path_stub: str
    navigation_document: bool
    summary_png: bool
    backplane_fits: bool
    backplane_metadata: bool
    navigation_succeeded: bool
    navigation_status: str | None

    @property
    def complete(self) -> bool:
        """Whether the image has all four files and its navigation succeeded."""
        return all(
            (
                self.navigation_document,
                self.summary_png,
                self.backplane_fits,
                self.backplane_metadata,
                self.navigation_succeeded,
            )
        )

    def line(self) -> str:
        """Return the report as the one line ``--check-only`` prints for the image.

        Returns:
            The image's stub, whether each file is present or absent, what its navigation
            came to, and whether it is complete, as in ``<stub>: navigation document
            present, summary PNG present, backplane FITS present, backplane metadata
            present, navigation succeeded: complete``.
        """
        files = ', '.join(
            f'{name} {"present" if there else "absent"}'
            for name, there in (
                ('navigation document', self.navigation_document),
                ('summary PNG', self.summary_png),
                ('backplane FITS', self.backplane_fits),
                ('backplane metadata', self.backplane_metadata),
            )
        )
        if self.navigation_succeeded:
            navigation = 'navigation succeeded'
        elif self.navigation_status is None:
            navigation = 'no navigation recorded'
        else:
            navigation = f'navigation did not succeed (status {self.navigation_status})'
        state = 'complete' if self.complete else 'incomplete'
        return f'{self.results_path_stub}: {files}, {navigation}: {state}'


def _exists(path: FCPath) -> bool:
    """Say whether one file is there.

    Parameters:
        path: The file, a single path.

    Returns:
        True when it is there; ``FCPath.exists`` answers a single path with one boolean.
    """
    return path.exists() is True


def report_image_inputs(
    image_file: ImageFile,
    *,
    nav_results_root: str | Path | FCPath,
    backplane_results_root: str | Path | FCPath,
) -> InputsReport:
    """Report what one selected image has of the files the labels pass reads.

    Reads the image's navigation record, as the labels pass does, and looks for each
    file; it writes no file.

    Parameters:
        image_file: The image.
        nav_results_root: The root holding the navigation documents and summary PNGs.
        backplane_results_root: The root holding the backplane FITS files and metadata.

    Returns:
        The report.
    """
    nav_root = FCPath(nav_results_root)
    backplane_root = FCPath(backplane_results_root)
    inputs = image_inputs(
        image_file.results_path_stub,
        nav_results_root=nav_root,
        backplane_results_root=backplane_root,
    )
    record = navigation_record(image_file, inputs)
    return InputsReport(
        results_path_stub=image_file.results_path_stub,
        navigation_document=_exists(inputs.navigation_document),
        summary_png=_exists(inputs.summary_png),
        backplane_fits=_exists(inputs.backplane_fits),
        backplane_metadata=_exists(inputs.backplane_metadata),
        navigation_succeeded=record is not None and navigation_succeeded(record),
        navigation_status=None if record is None else str(record.get('status')),
    )
