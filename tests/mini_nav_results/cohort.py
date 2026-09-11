"""The cohort a bundle's tests run over, and what writing one to disk shares.

A cohort is what a navigation run and the backplane stage after it leave behind for a
few of one bundle's images -- the navigation documents and summary PNGs under a
navigation root, the backplane FITS files and their metadata documents under a
backplane root -- written wherever the caller points it and torn down with it.
Nothing here is checked in: the builders are the artifact, not their output, so a
cohort costs the repository nothing.

Each bundle has its own cohort, a subclass of :class:`Cohort` in a module named for
the bundle.  The subclass supplies what is the bundle's: its images, the holdings
layout they sit in, how an image's camera is read from its index row, the range each
backplane plane spans, and the registered dataset the bundle is built with.  This
module supplies what every cohort shares: writing the two roots and the holdings
directory, the images in the shape an enumeration hands them on, and the one-image
batches the per-image stages take.  Adding a bundle's cohort is one such module and
one entry in the package's ``COHORTS`` registry.

A test takes a written cohort from the session-scoped ``mini_nav_cohorts`` fixture,
which writes each cohort once, the first time a test asks for it, rather than per
test, since a dozen label tests should not each rewrite a FITS.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Self, TypeVar, cast

from filecache import FCPath

from spindoctor.config import DEFAULT_CONFIG
from spindoctor.dataset.dataset import DataSet, ImageFile, ImageFiles
from spindoctor.support.file import json_as_string

from .backplanes import CohortBody, write_backplanes
from .browse import write_summary_png

NAV_RESULTS_DIR_NAME = 'nav_results'
"""The directory under a cohort root standing in for ``nav_results_root``."""

BACKPLANE_RESULTS_DIR_NAME = 'backplane_results'
"""The directory under a cohort root standing in for ``backplane_results_root``."""

HOLDINGS_DIR_NAME = 'holdings'
"""The directory under a cohort root the images' own URLs name.

It is created and left empty.  No image is written there and no consumer of the
cohort opens one: what a bundle describes is what a navigation run left behind,
not the image it read.  What the directory is for is being somewhere -- a
dataset is constructed over it, and a documented directory that is not there is
a diagnostic about the wrong thing for whoever first resolves a path under it.

Below it, an image sits where an enumeration would find it, under the subtree its
cohort's holdings layout names, which is the layout each document records as
well, under the holdings root the run that wrote it was given.
"""


@dataclass(frozen=True)
class CohortImage:
    """One image of a cohort, and everything a run leaves on disk for it.

    Attributes:
        stub: Where its results sit under a results root.
        image_name: The calibrated image's name, without its extension.
        camera: The camera that took it.
        midtime_et: The exposure midtime, which is its epoch and the one input
            every other value here is derived from.
        document: The navigation document a run wrote for it.
        index_file_row: The index row an enumeration hands on with it.
        bodies: The bodies its backplanes cover, empty when it has none.
        rings: Whether its backplanes cover the ring system.
    """

    stub: str
    image_name: str
    camera: str
    midtime_et: float
    document: dict[str, Any]
    index_file_row: dict[str, Any]
    bodies: tuple[CohortBody, ...]
    rings: bool

    @property
    def navigated(self) -> bool:
        """Whether the navigation succeeded, which is what the bundle stage reads."""
        return bool(self.document.get('status') == 'success')


@dataclass(frozen=True)
class Cohort(ABC):
    """A written cohort, and where each half of it went.

    The base of every bundle's cohort.  A subclass sets the three class attributes
    and implements :meth:`images`, :meth:`camera_of` and :meth:`dataset`; this class
    writes the cohort from them, with :meth:`write`.

    Attributes:
        NAME: The name the cohort is registered and chosen under.
        HOLDINGS_SUBTREE: Where the bundle's images sit under a holdings root,
            which is where an enumeration finds them.
        PLANE_BOUNDS: What one plane of each configured name spans, in the units
            the configuration declares; the backplane products ramp between them.
        root: The directory everything below sits under.
        nav_results_root: Where the navigation documents and browse images are.
        backplane_results_root: Where the backplane FITS files and their
            metadata documents are.
        holdings_root: Where the images' own URLs point, created and empty.
        image_files: Every image, in the shape an enumeration hands them on --
            two URLs, a results path stub, an index row and a camera.
        written: Every file written, in the order it was written.
    """

    NAME: ClassVar[str]
    HOLDINGS_SUBTREE: ClassVar[str]
    PLANE_BOUNDS: ClassVar[Mapping[str, tuple[float, float]]]

    root: Path
    nav_results_root: Path
    backplane_results_root: Path
    holdings_root: Path
    image_files: ImageFiles
    written: tuple[Path, ...]

    @classmethod
    @abstractmethod
    def images(cls) -> tuple[CohortImage, ...]:
        """Return every image of the cohort, in the order a run would have processed them.

        Returns:
            The images, each with its document, its index row and its backplanes.
        """

    @classmethod
    @abstractmethod
    def camera_of(cls, index_file_row: dict[str, Any]) -> str | None:
        """Return the camera that took an image, read from its index row.

        Parameters:
            index_file_row: The index row an enumeration hands on with the image.

        Returns:
            The camera, as the bundle's dataset names it, or None when the row
            names none.
        """

    @abstractmethod
    def dataset(self) -> DataSet:
        """Return the registered dataset the cohort's bundle is built with.

        Returns:
            The dataset, constructed over :attr:`holdings_root`.
        """

    @classmethod
    def documents(cls) -> dict[str, dict[str, Any]]:
        """Return every image's navigation document, keyed by its results path stub.

        Returns:
            Stub to document, in the order a run would have written them.
        """
        return {image.stub: image.document for image in cls.images()}

    @classmethod
    def write(cls, root: Path) -> Self:
        """Write every product of the cohort under a root.

        The shipped configuration decides which planes the backplane products
        carry, which is what the run they stand in for would have read.

        Parameters:
            root: The directory to write under; created if it is not there.

        Returns:
            The written cohort: where each half went, and every file written.
        """
        DEFAULT_CONFIG.ensure_loaded()
        nav_results_root = root / NAV_RESULTS_DIR_NAME
        backplane_results_root = root / BACKPLANE_RESULTS_DIR_NAME
        holdings_root = root / HOLDINGS_DIR_NAME
        holdings_root.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        image_files: list[ImageFile] = []

        for image in cls.images():
            document_path = nav_results_root / f'{image.stub}_metadata.json'
            document_path.parent.mkdir(parents=True, exist_ok=True)
            document_path.write_text(json_as_string(image.document), encoding='utf-8')
            written.append(document_path)

            if image.navigated:
                png_path = nav_results_root / f'{image.stub}_summary.png'
                write_summary_png(FCPath(png_path))
                written.append(png_path)

                fits_path = backplane_results_root / f'{image.stub}_backplanes.fits'
                fits_path.parent.mkdir(parents=True, exist_ok=True)
                write_backplanes(
                    FCPath(fits_path),
                    bodies=image.bodies,
                    rings=image.rings,
                    plane_bounds=cls.PLANE_BOUNDS,
                    config=DEFAULT_CONFIG,
                )
                written.append(fits_path)
                written.append(backplane_results_root / f'{image.stub}_backplane_metadata.json')

            image_url = holdings_root / cls.HOLDINGS_SUBTREE / f'{image.stub}.IMG'
            image_files.append(
                ImageFile(
                    image_file_url=FCPath(image_url),
                    label_file_url=FCPath(image_url.with_suffix('.LBL')),
                    results_path_stub=image.stub,
                    index_file_row=image.index_file_row,
                    camera=cls.camera_of(image.index_file_row),
                )
            )

        return cls(
            root=root,
            nav_results_root=nav_results_root,
            backplane_results_root=backplane_results_root,
            holdings_root=holdings_root,
            image_files=ImageFiles(image_files=image_files),
            written=tuple(written),
        )

    def batch(self, results_path_stub: str) -> ImageFiles:
        """Return the one-image batch the per-image stages take.

        Parameters:
            results_path_stub: Which image, by its results path stub.

        Returns:
            A batch holding that image alone.

        Raises:
            KeyError: If the cohort holds no such image.
        """
        for image_file in self.image_files:
            if image_file.results_path_stub == results_path_stub:
                return ImageFiles(image_files=[image_file])
        raise KeyError(f'the cohort holds no image at {results_path_stub!r}')


CohortT = TypeVar('CohortT', bound=Cohort)
"""Any one bundle's cohort class, so that what is written is typed as that class."""


class WrittenCohorts:
    """Each cohort a session asks for, written the first time it is asked for.

    A session writes one bundle's cohort once and hands the same one to every test
    that asks for it, and writes none that no test asks for.
    """

    def __init__(self, make_root: Callable[[str], Path]) -> None:
        """Prepare to write cohorts under directories the given factory makes.

        Parameters:
            make_root: Makes a fresh directory from a name, as pytest's temporary
                path factory does.
        """
        self._make_root = make_root
        self._written: dict[type[Cohort], Cohort] = {}

    def __call__(self, cohort_class: type[CohortT]) -> CohortT:
        """Return the cohort of this class, writing it on the first request.

        Parameters:
            cohort_class: Which bundle's cohort.

        Returns:
            The written cohort.
        """
        if cohort_class not in self._written:
            self._written[cohort_class] = cohort_class.write(self._make_root(cohort_class.NAME))
        return cast(CohortT, self._written[cohort_class])
