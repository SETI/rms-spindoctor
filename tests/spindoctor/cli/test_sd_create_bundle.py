"""Tests that a bundle run which did not write every label says so in its result.

``pdstemplate`` reports a label it could not render through its return value
rather than by raising, so a run that dropped one looks exactly like a run that
did not, unless the drivers act on what came back to them.  These pin that: what
each of the three entry points -- the two subcommands and the cloud-task worker
-- does with a product it failed to write, and what it does with none.

Each entry point is stood up on stubs, because what is under test is the
counting and the report, not the configuration loading, the logging setup or the
generation they read.
"""

import argparse
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pdstemplate
import pytest
from cloud_tasks.worker import WorkerData
from filecache import FCPath

from spindoctor.cli import sd_create_bundle, sd_create_bundle_cloud_tasks
from spindoctor.cli.pds4.bundle_data import BundleDataOutcome
from spindoctor.dataset.dataset import ImageFile, ImageFiles, Pds4Pass
from spindoctor.dataset.dataset_sim import DataSetSim


def _image_file(name: str, *, base_dir: Path | None = None) -> ImageFile:
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


def _batch_image_name(batch: int, index: int) -> str:
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

REQUIRED_TEMPLATES: dict[Pds4Pass, list[str]] = {
    'labels': ['data.lblx', 'browse.lblx'],
    'summary': [
        'collection_data.lblx',
        'collection_browse.lblx',
        'global_index_bodies.lblx',
        'global_index_rings.lblx',
    ],
}
"""What the stub dataset declares each pass must find, as Cassini declares it."""


class _StubDataset:
    """A dataset serving the pds4_* hooks the drivers call, over chosen batches.

    It carries a configuration declaring the backplanes it is given, as every
    dataset carries one, since the drivers check the configured units before
    they reach a hook.
    """

    def __init__(
        self,
        template_dir: Path,
        *,
        image_count: int = 1,
        batch_count: int = 1,
        base_dir: Path | None = None,
        bodies: list[dict[str, Any]] | None = None,
        rings: list[dict[str, Any]] | None = None,
    ) -> None:
        """Prepare an enumeration of batches holding that many images each.

        Parameters:
            template_dir: Directory served as the dataset's template directory.
            image_count: How many images each batch holds.
            batch_count: How many batches the enumeration yields.
            base_dir: Directory the enumerated images live in; only a run that
                reaches the real generation needs a real one.
            bodies: ``config.backplanes.bodies`` entries; none when None.
            rings: ``config.backplanes.rings`` entries; none when None.
        """
        self._template_dir = template_dir
        self._image_count = image_count
        self._batch_count = batch_count
        self._base_dir = base_dir
        self.config = SimpleNamespace(
            backplanes=SimpleNamespace(
                bodies=bodies if bodies is not None else [],
                rings=rings if rings is not None else [],
            )
        )

    def pds4_bundle_name(self) -> str:
        """Return the bundle name whose directory the run writes into."""
        return BUNDLE_NAME

    def pds4_bundle_template_dir(self) -> str:
        """Return the template directory the declared templates are looked for in."""
        return str(self._template_dir)

    def pds4_required_templates(self, pds4_pass: Pds4Pass) -> list[str]:
        """Return the template filenames the given pass must find.

        Parameters:
            pds4_pass: Which pass's templates to name.
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
                    _image_file(_batch_image_name(batch, n), base_dir=self._base_dir)
                    for n in range(self._image_count)
                ]
            )


def _stub_dataset(
    tmp_path: Path,
    *,
    image_count: int = 1,
    batch_count: int = 1,
    base_dir: Path | None = None,
    bodies: list[dict[str, Any]] | None = None,
    rings: list[dict[str, Any]] | None = None,
) -> _StubDataset:
    """Build the stub dataset over a template directory holding every template.

    Parameters:
        tmp_path: Base temporary directory the template directory lives under.
        image_count: How many images each enumerated batch holds.
        batch_count: How many batches the enumeration yields.
        base_dir: Directory the enumerated images live in.
        bodies: ``config.backplanes.bodies`` entries the dataset's configuration declares.
        rings: ``config.backplanes.rings`` entries the dataset's configuration declares.

    Returns:
        The stub dataset, whose template directory is already populated.
    """
    template_dir = tmp_path / 'templates'
    template_dir.mkdir(exist_ok=True)
    for names in REQUIRED_TEMPLATES.values():
        for name in names:
            (template_dir / name).write_text('<Product/>\n', encoding='utf-8')
    return _StubDataset(
        template_dir,
        image_count=image_count,
        batch_count=batch_count,
        base_dir=base_dir,
        bodies=bodies,
        rings=rings,
    )


def _dataset_with_unusable_units(tmp_path: Path) -> _StubDataset:
    """Build the stub dataset declaring one plane in a unit the index cannot size and one in none.

    Parameters:
        tmp_path: Base temporary directory the template directory lives under.

    Returns:
        The stub dataset, whose configuration both passes refuse.
    """
    return _stub_dataset(
        tmp_path,
        bodies=[{'name': 'body_tilt', 'units': 'mrad'}],
        rings=[{'name': 'ring_radius', 'units': None}],
    )


@pytest.fixture
def labels_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Stand the labels subcommand up on stubs, leaving only its counting live.

    Parameters:
        tmp_path: Base temporary directory served as every results root.
        monkeypatch: Fixture the stand-ins are installed through.
    """
    monkeypatch.setattr(
        sd_create_bundle, 'parse_args_labels', lambda _: argparse.Namespace(dry_run=False)
    )
    monkeypatch.setattr(sd_create_bundle, 'load_default_and_user_config', lambda *a: None)
    monkeypatch.setattr(sd_create_bundle, 'build_run_logging', lambda *a: None)
    monkeypatch.setattr(sd_create_bundle, 'get_nav_results_root', lambda *a: str(tmp_path))
    monkeypatch.setattr(sd_create_bundle, 'get_backplane_results_root', lambda *a: str(tmp_path))
    monkeypatch.setattr(sd_create_bundle, 'get_pds4_bundle_results_root', lambda *a: str(tmp_path))
    monkeypatch.setattr(pdstemplate.PdsTemplate, 'set_logger', staticmethod(lambda *a: None))
    monkeypatch.setattr(sd_create_bundle, 'DATASET', _stub_dataset(tmp_path))


@pytest.fixture
def summary_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Stand the summary subcommand up on stubs, leaving only its counting live.

    Parameters:
        tmp_path: Base temporary directory served as the bundle results root.
        monkeypatch: Fixture the stand-ins are installed through.
    """
    monkeypatch.setattr(
        sd_create_bundle,
        'parse_args_summary',
        lambda _: argparse.Namespace(dataset_name='coiss_saturn'),
    )
    monkeypatch.setattr(sd_create_bundle, 'load_default_and_user_config', lambda *a: None)
    monkeypatch.setattr(sd_create_bundle, 'build_run_logging', lambda *a: None)
    monkeypatch.setattr(sd_create_bundle, 'get_pds4_bundle_results_root', lambda *a: str(tmp_path))
    monkeypatch.setattr(pdstemplate.PdsTemplate, 'set_logger', staticmethod(lambda *a: None))
    dataset = _stub_dataset(tmp_path)
    monkeypatch.setattr(sd_create_bundle, 'dataset_name_to_class', lambda _: lambda: dataset)


def _labels_outcome(monkeypatch: pytest.MonkeyPatch, outcome: BundleDataOutcome) -> None:
    """Make every product of the labels run come to the given outcome.

    Parameters:
        monkeypatch: Fixture the stand-in is installed through.
        outcome: What generating one image's data files is to report.
    """
    monkeypatch.setattr(sd_create_bundle, 'generate_bundle_data_files', lambda **kwargs: outcome)


def _summary_counts(monkeypatch: pytest.MonkeyPatch, collections: int, index: int) -> None:
    """Make the two summary generators report the given failed-label counts.

    Parameters:
        monkeypatch: Fixture the stand-ins are installed through.
        collections: Failed collection labels to report.
        index: Failed global index labels to report.
    """
    monkeypatch.setattr(sd_create_bundle, 'generate_collection_files', lambda **kwargs: collections)
    monkeypatch.setattr(sd_create_bundle, 'generate_global_index_files', lambda **kwargs: index)


def _dry_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """Turn the labels run into a dry run.

    Parameters:
        monkeypatch: Fixture the parsed command line is installed through.
    """
    monkeypatch.setattr(
        sd_create_bundle, 'parse_args_labels', lambda _: argparse.Namespace(dry_run=True)
    )


def _record_generation(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Record every image the labels run generates products for.

    Parameters:
        monkeypatch: Fixture the recording stand-in is installed through.

    Returns:
        The list the calls are appended to, one entry per image.
    """
    calls: list[dict[str, Any]] = []

    def _generate(**kwargs: Any) -> BundleDataOutcome:
        """Record one image's generation and report it written.

        Parameters:
            **kwargs: What the driver passed for this image.

        Returns:
            WRITTEN, so the run counts the image as labeled.
        """
        calls.append(kwargs)
        return BundleDataOutcome.WRITTEN

    monkeypatch.setattr(sd_create_bundle, 'generate_bundle_data_files', _generate)
    return calls


def test_main_labels_refuses_a_bundle_root_that_holds_files(
    labels_run: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A bundle goes into an empty directory, so a populated one ends the run at once.

    The run writes nothing rather than assembling one bundle out of two runs,
    and it will not clear the directory itself, so the report has to say what
    the operator is to do about it.
    """
    bundle_root = tmp_path / BUNDLE_NAME
    (bundle_root / 'data').mkdir(parents=True)
    calls = _record_generation(monkeypatch)
    with pytest.raises(SystemExit) as excinfo:
        sd_create_bundle.main_labels()
    assert excinfo.value.code == 1
    assert calls == []
    out = capsys.readouterr().out
    assert f'The bundle root {bundle_root} already holds files' in out
    assert 'Clear it, or name another bundle results root' in out


@pytest.mark.parametrize('root_exists', [True, False], ids=['empty root', 'no root at all'])
def test_main_labels_writes_into_a_bundle_root_with_nothing_in_it(
    labels_run: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    root_exists: bool,
) -> None:
    """A bundle root with nothing in it is what the run asks for, made or not.

    A first run makes the directory itself, and an operator who cleared one by
    hand leaves it there, so both are the empty directory the precondition
    names.

    Parameters:
        labels_run: Fixture standing the subcommand up on stubs.
        monkeypatch: Fixture the recording generation is installed through.
        tmp_path: Base temporary directory served as the bundle results root.
        capsys: Fixture the closing report is read from.
        root_exists: Whether the empty bundle root is already on disk.
    """
    if root_exists:
        (tmp_path / BUNDLE_NAME).mkdir()
    calls = _record_generation(monkeypatch)
    sd_create_bundle.main_labels()
    assert len(calls) == 1
    assert 'Label generation complete: 1 image(s) labeled, 0 skipped' in capsys.readouterr().out


def test_main_labels_refuses_a_template_the_dataset_does_not_have(
    labels_run: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A template the dataset declares and does not have ends the run at once.

    Every image of the pass renders from the same template directory, so a
    template that is not there is not there for any of them; the run names it
    once, before it has processed an image, rather than failing identically
    thousands of times.
    """
    missing = tmp_path / 'templates' / 'data.lblx'
    missing.unlink()
    calls = _record_generation(monkeypatch)
    with pytest.raises(SystemExit) as excinfo:
        sd_create_bundle.main_labels()
    assert excinfo.value.code == 1
    assert calls == []
    assert f'PDS4 template not found: {missing}' in capsys.readouterr().out


def test_main_labels_refuses_a_unit_the_bundle_cannot_use(
    labels_run: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A configured backplane in a unit the bundle cannot use ends the labels run at once.

    The pass holds every document to the configured unit, so a unit it cannot
    use, one the index has no format for or none at all, fails every image
    the same way; the run names each such entry once, before it has read an
    image, and leaves the bundle root as it found it.
    """
    monkeypatch.setattr(sd_create_bundle, 'DATASET', _dataset_with_unusable_units(tmp_path))
    calls = _record_generation(monkeypatch)
    with pytest.raises(SystemExit) as excinfo:
        sd_create_bundle.main_labels()
    assert excinfo.value.code == 1
    assert calls == []
    assert not (tmp_path / BUNDLE_NAME).exists()
    out = capsys.readouterr().out
    assert 'Backplane body_tilt declares a unit the bundle cannot use' in out
    assert 'Backplane ring_radius declares a unit the bundle cannot use' in out


def test_main_labels_exits_non_zero_when_a_product_fails(
    labels_run: None, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An image whose label could not be written ends the run non-zero.

    The report counts images and labels, not products: an image whose data
    label failed still has a browse label, a summary PNG and a supplemental
    file in the bundle, so calling it an unwritten product would be untrue.
    """
    _labels_outcome(monkeypatch, BundleDataOutcome.FAILED)
    with pytest.raises(SystemExit) as excinfo:
        sd_create_bundle.main_labels()
    assert excinfo.value.code == 1
    expected = (
        'Label generation incomplete: 0 image(s) labeled, 0 skipped, '
        '1 whose labels were not written'
    )
    assert expected in capsys.readouterr().out


def test_main_labels_exits_non_zero_when_a_batch_is_not_one_image(
    labels_run: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A batch of any other size is a product the run did not write.

    Parameters:
        labels_run: Fixture standing the subcommand up on stubs.
        monkeypatch: Fixture the two-image enumeration is installed through.
        tmp_path: Base temporary directory the stub dataset is built under.
    """
    monkeypatch.setattr(sd_create_bundle, 'DATASET', _stub_dataset(tmp_path, image_count=2))
    _labels_outcome(monkeypatch, BundleDataOutcome.WRITTEN)
    with pytest.raises(SystemExit) as excinfo:
        sd_create_bundle.main_labels()
    assert excinfo.value.code == 1


def test_a_malformed_batch_counts_every_image_it_holds(
    labels_run: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Two images in one batch are two images the run did not label.

    The closing line is the account of what the bundle covers, so a batch of
    two counted as one image would understate what is missing from it.

    Parameters:
        labels_run: Fixture standing the subcommand up on stubs.
        monkeypatch: Fixture the two-image enumeration is installed through.
        tmp_path: Base temporary directory the stub dataset is built under.
        capsys: Fixture the closing line is read from.
    """
    monkeypatch.setattr(sd_create_bundle, 'DATASET', _stub_dataset(tmp_path, image_count=2))
    _labels_outcome(monkeypatch, BundleDataOutcome.WRITTEN)
    with pytest.raises(SystemExit):
        sd_create_bundle.main_labels()
    assert '2 whose labels were not written' in capsys.readouterr().out


def test_an_empty_batch_fails_the_run_it_adds_no_images_to(
    labels_run: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A batch holding nothing is still a run that did not do what it was asked.

    It contributes no images, so counting it among them would be a wrong
    number; leaving it out of the exit status would be a wrong answer.

    Parameters:
        labels_run: Fixture standing the subcommand up on stubs.
        monkeypatch: Fixture the empty enumeration is installed through.
        tmp_path: Base temporary directory the stub dataset is built under.
        capsys: Fixture the closing line is read from.
    """
    monkeypatch.setattr(sd_create_bundle, 'DATASET', _stub_dataset(tmp_path, image_count=0))
    _labels_outcome(monkeypatch, BundleDataOutcome.WRITTEN)
    with pytest.raises(SystemExit) as excinfo:
        sd_create_bundle.main_labels()
    assert excinfo.value.code == 1
    assert '0 whose labels were not written, 1 empty batch(es)' in capsys.readouterr().out


def test_main_labels_exits_zero_when_every_product_is_written(
    labels_run: None, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A run that wrote every label says how many, and does not end the process."""
    _labels_outcome(monkeypatch, BundleDataOutcome.WRITTEN)
    sd_create_bundle.main_labels()
    assert 'Label generation complete: 1 image(s) labeled, 0 skipped' in capsys.readouterr().out


def test_main_labels_does_not_fail_a_run_over_a_skipped_image(
    labels_run: None, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An image the bundle has nothing to describe is not a label the run failed to write.

    A selection made by volume names far more images than have been navigated
    and backplaned, so a run over one is mostly skips and still exits zero.
    """
    _labels_outcome(monkeypatch, BundleDataOutcome.SKIPPED)
    sd_create_bundle.main_labels()
    assert 'Label generation complete: 0 image(s) labeled, 1 skipped' in capsys.readouterr().out


def test_main_labels_reports_a_selection_that_matched_no_images(
    labels_run: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A selection matching nothing reports zero rather than an unqualified completion.

    Image-name selection is by prefix, so a mistyped name matches nothing and is
    otherwise indistinguishable from a healthy run; the closing line has to make
    zero visible as zero.

    Parameters:
        labels_run: Fixture standing the subcommand up on stubs.
        monkeypatch: Fixture the empty enumeration is installed through.
        tmp_path: Base temporary directory the stub dataset is built under.
        capsys: Fixture the closing report is read from.
    """
    monkeypatch.setattr(sd_create_bundle, 'DATASET', _stub_dataset(tmp_path, batch_count=0))
    _labels_outcome(monkeypatch, BundleDataOutcome.WRITTEN)
    sd_create_bundle.main_labels()
    assert 'Label generation complete: 0 image(s) labeled, 0 skipped' in capsys.readouterr().out


def test_a_dry_run_reports_what_it_would_have_processed(
    labels_run: None, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A dry run says how many images it would process, and never claims to have.

    Nothing is written, so a closing line about labels generated would say
    something that did not happen.
    """
    _dry_run(monkeypatch)
    _labels_outcome(monkeypatch, BundleDataOutcome.WRITTEN)
    sd_create_bundle.main_labels()
    out = capsys.readouterr().out
    assert 'Dry run complete: 1 image(s) would be processed' in out
    assert 'Label generation complete' not in out


def test_a_dry_run_over_a_malformed_batch_reports_it_and_exits_zero(
    labels_run: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A dry run reports a batch it could not have processed without failing on it.

    Reporting the batch is what the dry run is for; counting a label it never
    set out to write is not.

    Parameters:
        labels_run: Fixture standing the subcommand up on stubs.
        monkeypatch: Fixture the two-image enumeration is installed through.
        tmp_path: Base temporary directory the stub dataset is built under.
        capsys: Fixture the report is read from.
    """
    _dry_run(monkeypatch)
    monkeypatch.setattr(sd_create_bundle, 'DATASET', _stub_dataset(tmp_path, image_count=2))
    _labels_outcome(monkeypatch, BundleDataOutcome.WRITTEN)
    sd_create_bundle.main_labels()
    assert 'Expected 1 image file, got 2' in capsys.readouterr().out


def test_main_labels_carries_on_past_an_image_it_could_not_read(
    labels_run: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """One unreadable image costs its own label and no other image's.

    The middle image's navigation document is on disk and will not parse, which
    is a defect in that document rather than an image the bundle has nothing to
    say about, so the generation raises through to the run.  The images on
    either side of it are still processed and the run still closes with a count
    saying so.
    """
    monkeypatch.setattr(
        sd_create_bundle, 'DATASET', _stub_dataset(tmp_path, batch_count=3, base_dir=tmp_path)
    )
    middle = _batch_image_name(1, 0)
    unparseable = tmp_path / 'res' / f'{middle}_metadata.json'
    unparseable.parent.mkdir(parents=True, exist_ok=True)
    unparseable.write_text('not json at all', encoding='utf-8')
    with pytest.raises(SystemExit) as excinfo:
        sd_create_bundle.main_labels()
    assert excinfo.value.code == 1
    out = capsys.readouterr().out
    assert 'Failed to generate bundle data files for' in out
    expected = (
        'Label generation incomplete: 0 image(s) labeled, 2 skipped, '
        '1 whose labels were not written'
    )
    assert expected in out


def test_main_summary_refuses_a_template_the_dataset_does_not_have(
    summary_run: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The summary pass checks its own templates up front, as the labels pass does.

    The two passes render different templates, so each looks for the ones it
    renders; a summary run whose collection template is not there stops on it
    even though the labels pass before it found everything it needed.
    """
    missing = tmp_path / 'templates' / 'collection_data.lblx'
    missing.unlink()
    _summary_counts(monkeypatch, collections=0, index=0)
    with pytest.raises(SystemExit) as excinfo:
        sd_create_bundle.main_summary()
    assert excinfo.value.code == 1
    assert f'PDS4 template not found: {missing}' in capsys.readouterr().out


def test_main_summary_refuses_a_unit_the_bundle_cannot_use(
    summary_run: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The summary pass refuses the same units up front, before it reads the data tree.

    The collection files are written before the index tables, so a check left
    to the index writer would leave them on disk over a run that then failed;
    this one runs before either generator, and the bundle's data directory is
    there so that a run past it would have written a collection file.
    """
    dataset = _dataset_with_unusable_units(tmp_path)
    monkeypatch.setattr(sd_create_bundle, 'dataset_name_to_class', lambda _: lambda: dataset)
    (tmp_path / BUNDLE_NAME / 'data').mkdir(parents=True)
    with pytest.raises(SystemExit) as excinfo:
        sd_create_bundle.main_summary()
    assert excinfo.value.code == 1
    assert not (tmp_path / BUNDLE_NAME / 'data' / 'collection_data.tab').exists()
    out = capsys.readouterr().out
    assert 'Backplane body_tilt declares a unit the bundle cannot use' in out
    assert 'Backplane ring_radius declares a unit the bundle cannot use' in out


@pytest.mark.parametrize(
    ('collections', 'index'), [(1, 0), (0, 1)], ids=['collection label', 'index label']
)
def test_main_summary_exits_non_zero_when_a_label_is_not_written(
    summary_run: None, monkeypatch: pytest.MonkeyPatch, collections: int, index: int
) -> None:
    """A label either generator could not write ends the run non-zero.

    Both counts are cases of their own because the run has to add them: a
    summary that reported only the second generator's failures would still pass
    a test that never gave the first any.

    Parameters:
        summary_run: Fixture standing the subcommand up on stubs.
        monkeypatch: Fixture the failed-label counts are installed through.
        collections: Failed collection labels for this case.
        index: Failed global index labels for this case.
    """
    _summary_counts(monkeypatch, collections=collections, index=index)
    with pytest.raises(SystemExit) as excinfo:
        sd_create_bundle.main_summary()
    assert excinfo.value.code == 1


def test_main_summary_reports_why_the_index_could_not_be_generated(
    summary_run: None, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An index generator that raises ends the run with the reason in the log.

    A supplemental file in another unit is refused with a message naming the
    file and both units, and the frames of a traceback do not carry it.
    """
    monkeypatch.setattr(sd_create_bundle, 'generate_collection_files', lambda **kwargs: 0)

    def _refuse(**kwargs: Any) -> int:
        """Refuse the index the way a supplemental file in another unit is refused.

        Parameters:
            **kwargs: What the driver passed, unused.

        Raises:
            ValueError: Always, naming a file and both units.
        """
        raise ValueError(
            'Supplemental file X records the tilt statistic in rad where the '
            'configuration expects deg'
        )

    monkeypatch.setattr(sd_create_bundle, 'generate_global_index_files', _refuse)
    with pytest.raises(SystemExit) as excinfo:
        sd_create_bundle.main_summary()
    assert excinfo.value.code == 1
    out = capsys.readouterr().out
    assert 'Supplemental file X records the tilt statistic in rad' in out


def test_main_summary_exits_zero_when_every_label_is_written(
    summary_run: None, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A summary that wrote every label reports completion and does not end the process."""
    _summary_counts(monkeypatch, collections=0, index=0)
    sd_create_bundle.main_summary()
    assert 'Summary generation complete' in capsys.readouterr().out


@pytest.fixture
def cloud_task_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Stand the cloud-task worker up on stubs, leaving only its reporting live.

    Parameters:
        tmp_path: Base temporary directory served as every results root.
        monkeypatch: Fixture the stand-ins are installed through.
    """
    module = sd_create_bundle_cloud_tasks
    monkeypatch.setattr(module, 'load_default_and_user_config', lambda *a: None)
    monkeypatch.setattr(module, 'get_nav_results_root', lambda *a: str(tmp_path))
    monkeypatch.setattr(module, 'get_backplane_results_root', lambda *a: str(tmp_path))
    monkeypatch.setattr(module, 'get_pds4_bundle_results_root', lambda *a: str(tmp_path))
    monkeypatch.setattr(module, 'dataset_name_to_class', lambda _: object)


def _run_cloud_task(
    monkeypatch: pytest.MonkeyPatch, outcome: BundleDataOutcome
) -> tuple[bool, Any]:
    """Run one cloud task whose generation comes to the given outcome.

    Parameters:
        monkeypatch: Fixture the generation stand-in is installed through.
        outcome: What generating the task's one image is to report.

    Returns:
        The worker's retry flag and result.
    """
    monkeypatch.setattr(
        sd_create_bundle_cloud_tasks, 'generate_bundle_data_files', lambda **kwargs: outcome
    )
    task_data = {
        'dataset_name': 'coiss_saturn',
        'files': [
            {
                'image_file_url': '/hermetic/1234567890w.img',
                'label_file_url': '/hermetic/1234567890w.lbl',
                'results_path_stub': 'res/1234567890w',
            }
        ],
    }
    worker_data = cast(WorkerData, SimpleNamespace(args=argparse.Namespace(config_file=None)))
    return sd_create_bundle_cloud_tasks.process_task('task', task_data, worker_data)


def test_a_cloud_task_reports_a_product_it_could_not_write(
    cloud_task_run: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A label the template could not render comes back as an error result."""
    retry, result = _run_cloud_task(monkeypatch, BundleDataOutcome.FAILED)
    assert result == {'status': 'error', 'status_error': 'label_not_written'}
    assert retry is False


def test_a_cloud_task_reports_a_product_it_wrote(
    cloud_task_run: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A product whose labels are on disk still comes back a success."""
    _, result = _run_cloud_task(monkeypatch, BundleDataOutcome.WRITTEN)
    assert result == {'status': 'success'}


def test_the_labels_parser_builds_a_dataset_that_reads_no_holdings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dataset that is not PDS3 goes through the labels parser like any other.

    The selection arguments are whatever the dataset class declares, and only
    that class reads them, so the parser has nothing of its own to read off the
    namespace.  A parser that read a PDS3 option itself would fail on the first
    dataset declaring none, which nothing else drives through it.
    """
    monkeypatch.setattr(sd_create_bundle, 'DATASET', None)
    monkeypatch.setattr(sd_create_bundle, 'DATASET_NAME', None)
    sd_create_bundle.parse_args_labels(['sim'])
    assert isinstance(sd_create_bundle.DATASET, DataSetSim)
