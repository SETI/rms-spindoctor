"""The cohort on disk: a navigation root and a backplane root under one root.

What a navigation run and the backplane stage after it leave behind for three
Cassini images, written wherever the caller points them and torn down with it.
Nothing here is checked in.  The builders are the artifact, not their output,
which is what keeps a second instrument's cohort from costing the repository
anything: adding one is a module beside ``cohort_cassini``, and the FITS, the
browse images and the documents it implies exist only while a test is running.

A test takes the whole thing as a session-scoped fixture rather than building
it per test, since a dozen label tests should not each rewrite a FITS.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from filecache import FCPath

from spindoctor.config import DEFAULT_CONFIG
from spindoctor.dataset.dataset import ImageFile, ImageFiles
from spindoctor.dataset.dataset_pds3_cassini_iss import DataSetPDS3CassiniISS
from spindoctor.support.file import json_as_string

from .backplanes import write_backplanes
from .browse import write_summary_png
from .cohort_cassini import HOLDINGS_SUBTREE, cohort_images

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

Below it, an image sits where an enumeration would find it: under the
calibrated products of its volume set and volume, which is the layout each
document records as well, under the holdings root the run that wrote it was
given.
"""


@dataclass(frozen=True)
class Cohort:
    """A written cohort, and where each half of it went.

    Attributes:
        root: The directory everything below sits under.
        nav_results_root: Where the navigation documents and browse images are.
        backplane_results_root: Where the backplane FITS files and their
            metadata documents are.
        holdings_root: Where the images' own URLs point, created and empty.
        image_files: Every image, in the shape an enumeration hands them on --
            two URLs, a results path stub, an index row and a camera.
        written: Every file written, in the order it was written.
    """

    root: Path
    nav_results_root: Path
    backplane_results_root: Path
    holdings_root: Path
    image_files: ImageFiles
    written: tuple[Path, ...]

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


def write_cohort(root: Path) -> Cohort:
    """Write every cohort product under a root.

    The shipped configuration decides which planes the backplane products
    carry, which is what the run they stand in for would have read.

    Parameters:
        root: The directory to write under; created if it is not there.

    Returns:
        Where each half went, and every file written.
    """
    DEFAULT_CONFIG.ensure_loaded()
    nav_results_root = root / NAV_RESULTS_DIR_NAME
    backplane_results_root = root / BACKPLANE_RESULTS_DIR_NAME
    holdings_root = root / HOLDINGS_DIR_NAME
    holdings_root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    image_files: list[ImageFile] = []

    for image in cohort_images():
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
                config=DEFAULT_CONFIG,
            )
            written.append(fits_path)
            written.append(backplane_results_root / f'{image.stub}_backplane_metadata.json')

        image_url = holdings_root / HOLDINGS_SUBTREE / f'{image.stub}.IMG'
        image_files.append(
            ImageFile(
                image_file_url=FCPath(image_url),
                label_file_url=FCPath(image_url.with_suffix('.LBL')),
                results_path_stub=image.stub,
                index_file_row=image.index_file_row,
                camera=DataSetPDS3CassiniISS.camera_from_index_row(image.index_file_row),
            )
        )

    return Cohort(
        root=root,
        nav_results_root=nav_results_root,
        backplane_results_root=backplane_results_root,
        holdings_root=holdings_root,
        image_files=ImageFiles(image_files=image_files),
        written=tuple(written),
    )
