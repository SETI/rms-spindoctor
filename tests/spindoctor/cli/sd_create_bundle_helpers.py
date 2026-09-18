"""Shared stand-ins for the ``sd_create_bundle`` driver tests.

The driver is exercised from a test file for each thing it does -- the labels pass and
the cloud-task worker, the summary pass, and the check -- and each stands it up on the
same stub dataset, which serves the ``pds4_*`` hooks the driver calls over a template
directory holding what it declares, and enumerates chosen batches of images.  It lives
here, in a plain module, so no test file imports another.
"""

import argparse
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NoReturn

from filecache import FCPath

from spindoctor.config import DEFAULT_CONFIG
from spindoctor.dataset.dataset import ImageFile, ImageFiles, Pds4Pass, Pds4Schema


def image_file(name: str, *, base_dir: Path | None = None) -> ImageFile:
    """Build one hermetic image file whose URLs are never retrieved.

    Parameters:
        name: Bare image name.
        base_dir: Directory the URLs live in; the non-writable ``/hermetic``
            when None.  Only a call path that resolves ``image_file_path``,
            which creates the URL's parent directory, needs a real one.

    Returns:
        The constructed image file.
    """
    base = str(base_dir) if base_dir is not None else '/hermetic'
    return ImageFile(
        image_file_url=FCPath(f'{base}/{name}.img'),
        label_file_url=FCPath(f'{base}/{name}.lbl'),
        results_path_stub=f'res/{name}',
    )


def batch_image_name(batch: int, index: int) -> str:
    """Name the image at one position of one enumerated batch.

    Parameters:
        batch: Which batch the image is in.
        index: The image's position within that batch.

    Returns:
        The bare image name.
    """
    return f'12345678{batch}{index}w'


BUNDLE_NAME = 'fake_bundle'
"""The bundle the stub dataset names, and so the directory a run writes into."""


def refuse(*args: Any, **kwargs: Any) -> NoReturn:
    """Stand in for something a run must not call.

    Parameters:
        *args: Whatever the run passed.
        **kwargs: Whatever the run passed by keyword.

    Raises:
        AssertionError: Always, naming what it was passed.
    """
    raise AssertionError(f'called with {args!r} and {kwargs!r}, which the run must not do')


STAND_IN_INFORMATION_MODEL_VERSION = '1.0.0.0'
"""The information model version the stub datasets give their labels."""

STAND_IN_SCHEMAS = {
    'fake': Pds4Schema(
        location='https://example.invalid/fake/v1/PDS4_FAKE',
        lidvid='urn:nasa:pds:system_bundle:xml_schema:fake-xml_schema::1.0',
    )
}
"""The dictionary schemas the stub datasets give their labels: one neutral stand-in."""

REQUIRED_TEMPLATES: dict[Pds4Pass, list[str]] = {
    'labels': ['data.lblx', 'browse.lblx'],
    'summary': [
        'collection_data.lblx',
        'collection_browse.lblx',
        'global_bodies_index.lblx',
        'global_rings_index.lblx',
    ],
}
"""What the stub dataset declares each pass must find: the templates that pass renders."""


class StubDataset:
    """A dataset serving the pds4_* hooks the drivers call, over chosen batches.

    It carries a configuration declaring no backplanes and no targets, as every dataset
    carries one, for the summary pass's index generator to read.
    """

    def __init__(
        self,
        template_dir: Path,
        *,
        image_count: int = 1,
        batch_count: int = 1,
        base_dir: Path | None = None,
    ) -> None:
        """Prepare an enumeration of batches holding that many images each.

        Parameters:
            template_dir: Directory served as the dataset's template directory.
            image_count: How many images each batch holds.
            batch_count: How many batches the enumeration yields.
            base_dir: Directory the enumerated images live in; only a run that
                reaches the real generation needs a real one.
        """
        self._template_dir = template_dir
        self._image_count = image_count
        self._batch_count = batch_count
        self._base_dir = base_dir
        self.config = SimpleNamespace(
            backplanes=SimpleNamespace(
                bodies=[],
                rings=[],
                ring_incidence_angle=DEFAULT_CONFIG.backplanes.ring_incidence_angle,
                masked_value=DEFAULT_CONFIG.backplanes.masked_value,
                target_lids={},
            )
        )

    def pds4_bundle_name(self) -> str:
        """Return the bundle name whose directory the run writes into."""
        return BUNDLE_NAME

    def pds4_bundle_version(self) -> str:
        """Return the bundle's version, which each of its products carries."""
        return '1.0'

    def pds4_information_model_version(self) -> str:
        """Return the information model version the labels are written against."""
        return STAND_IN_INFORMATION_MODEL_VERSION

    def pds4_schemas(self) -> dict[str, Pds4Schema]:
        """Return the dictionary schemas the labels declare."""
        return dict(STAND_IN_SCHEMAS)

    def pds4_bundle_template_dir(self) -> str:
        """Return the template directory the declared templates are looked for in."""
        return str(self._template_dir)

    def pds4_required_templates(self, pds4_pass: Pds4Pass) -> list[str]:
        """Return the template filenames the given pass must find.

        Parameters:
            pds4_pass: Which pass's templates to name.

        Returns:
            The template filenames the pass must find.
        """
        return REQUIRED_TEMPLATES[pds4_pass]

    def yield_image_files_from_arguments(
        self, arguments: argparse.Namespace
    ) -> Iterator[ImageFiles]:
        """Yield the batches the run will process.

        Parameters:
            arguments: The parsed command line, unused.

        Yields:
            Each batch, holding the configured number of images.
        """
        for batch in range(self._batch_count):
            yield ImageFiles(
                image_files=[
                    image_file(batch_image_name(batch, n), base_dir=self._base_dir)
                    for n in range(self._image_count)
                ]
            )


def stub_dataset(
    tmp_path: Path,
    *,
    image_count: int = 1,
    batch_count: int = 1,
    base_dir: Path | None = None,
) -> StubDataset:
    """Build the stub dataset over a template directory holding every template.

    Parameters:
        tmp_path: Base temporary directory the template directory lives under.
        image_count: How many images each enumerated batch holds.
        batch_count: How many batches the enumeration yields.
        base_dir: Directory the enumerated images live in.

    Returns:
        The stub dataset, whose template directory is already populated.
    """
    template_dir = tmp_path / 'templates'
    template_dir.mkdir(exist_ok=True)
    for names in REQUIRED_TEMPLATES.values():
        for name in names:
            (template_dir / name).write_text('<Product/>\n', encoding='utf-8')
    return StubDataset(
        template_dir,
        image_count=image_count,
        batch_count=batch_count,
        base_dir=base_dir,
    )
