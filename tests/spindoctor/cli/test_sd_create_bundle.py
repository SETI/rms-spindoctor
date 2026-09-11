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
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import pdstemplate
import pytest
from cloud_tasks.worker import WorkerData
from filecache import FCPath
from tests.spindoctor.cli.pds4.conftest import (
    make_bundle_env,
    navigated_document,
    touch_label,
    write_supplemental,
)

from spindoctor.cli import sd_create_bundle, sd_create_bundle_cloud_tasks
from spindoctor.cli.pds4.bundle_data import BundleDataOutcome
from spindoctor.cli.pds4.collections import GlobalIndexOutcome
from spindoctor.cli.pds4.epochs import EpochRange, NoEpochRange
from spindoctor.config import DEFAULT_CONFIG
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

SUMMARY_PRODUCTS = (
    'data/collection_data.tab',
    'data/collection_data.lblx',
    'browse/collection_browse.tab',
    'browse/collection_browse.lblx',
    'document/supplemental/global_index_bodies.tab',
    'document/supplemental/global_index_bodies.lblx',
    'document/supplemental/global_index_rings.tab',
    'document/supplemental/global_index_rings.lblx',
)
"""Every file the summary pass writes, relative to the bundle's directory."""


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
                masked_value=DEFAULT_CONFIG.backplanes.masked_value,
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
    """Build the stub dataset declaring planes in units neither pass can use.

    One plane is in a unit the index cannot size, one in a null unit, and one
    has no units key at all.

    Parameters:
        tmp_path: Base temporary directory the template directory lives under.

    Returns:
        The stub dataset, whose configuration both passes refuse.
    """
    return _stub_dataset(
        tmp_path,
        bodies=[{'name': 'body_tilt', 'units': 'mrad'}],
        rings=[{'name': 'ring_radius', 'units': None}, {'name': 'ring_tilt'}],
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
    outcome = GlobalIndexOutcome(failed_labels=index, epochs=NoEpochRange('not taken here'))
    monkeypatch.setattr(sd_create_bundle, 'generate_collection_files', lambda **kwargs: collections)
    monkeypatch.setattr(sd_create_bundle, 'generate_global_index_files', lambda **kwargs: outcome)


def _dry_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """Turn the labels run into a dry run.

    Parameters:
        monkeypatch: Fixture the parsed command line is installed through.
    """
    monkeypatch.setattr(
        sd_create_bundle, 'parse_args_labels', lambda _: argparse.Namespace(dry_run=True)
    )


def _record_generation(
    monkeypatch: pytest.MonkeyPatch, *, module: ModuleType = sd_create_bundle
) -> list[dict[str, Any]]:
    """Record every image a bundle driver generates products for.

    Parameters:
        monkeypatch: Fixture the recording stand-in is installed through.
        module: The driver whose generation is replaced; the labels
            subcommand's when not given.

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

    monkeypatch.setattr(module, 'generate_bundle_data_files', _generate)
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


def test_main_labels_refuses_a_masked_value_no_float_plane_can_hold(
    labels_run: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A configured masked value no float plane can hold ends the labels run at once.

    Every float array of every data label declares it as its missing constant, so it is
    unusable for every image; the run says so once, before it has read an image, and
    leaves the bundle root as it found it.
    """
    dataset = _stub_dataset(tmp_path)
    dataset.config.backplanes.masked_value = -999.1
    monkeypatch.setattr(sd_create_bundle, 'DATASET', dataset)
    calls = _record_generation(monkeypatch)
    with pytest.raises(SystemExit) as excinfo:
        sd_create_bundle.main_labels()
    assert excinfo.value.code == 1
    assert calls == []
    assert not (tmp_path / BUNDLE_NAME).exists()
    expected = 'the configured masked value -999.1 is not one a 32-bit float holds exactly'
    assert expected in capsys.readouterr().out


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
    saying so.  The log gives the parser's reason, which the frames of a
    traceback do not carry.
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
    assert 'Expecting value: line 1 column 1 (char 0)' in out
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

    Every unusable backplane is named, where the index writer, left to it, would
    stop on the first plane it could not format; and neither generator is called,
    the check coming before both, so nothing in the bundle is written or cleared.
    """
    dataset = _dataset_with_unusable_units(tmp_path)
    monkeypatch.setattr(sd_create_bundle, 'dataset_name_to_class', lambda _: lambda: dataset)
    called: list[str] = []
    monkeypatch.setattr(
        sd_create_bundle, 'generate_global_index_files', lambda **kwargs: called.append('index')
    )
    monkeypatch.setattr(
        sd_create_bundle,
        'generate_collection_files',
        lambda **kwargs: called.append('collections'),
    )
    with pytest.raises(SystemExit) as excinfo:
        sd_create_bundle.main_summary()
    assert excinfo.value.code == 1
    assert called == []
    out = capsys.readouterr().out
    assert 'Backplane body_tilt declares a unit the bundle cannot use' in out
    assert 'Backplane ring_radius declares a unit the bundle cannot use' in out
    assert 'Backplane ring_tilt declares no units' in out


def _latitude_in(units: str) -> dict[str, Any]:
    """Return one body's latitude statistic, recorded in the given unit.

    Parameters:
        units: The unit the statistic records.  The configuration the test gives its
            dataset declares the latitude plane in radians, whose statistics are
            taken in degrees.

    Returns:
        The ``backplanes.bodies`` payload of a supplemental file.
    """
    return {'MIMAS': {'backplanes': {'latitude': {'min': -1.2, 'max': 1.4, 'units': units}}}}


@pytest.mark.parametrize(
    ('bodies', 'raw_text', 'reason'),
    [
        (
            _latitude_in('rad'),
            None,
            'records the latitude statistic in rad where the configuration expects deg',
        ),
        (None, 'not json', 'could not be read: Expecting value: line 1 column 1 (char 0)'),
    ],
    ids=['a statistic in another unit', 'a file that is not JSON'],
)
def test_a_refused_summary_leaves_no_product_an_earlier_summary_wrote(
    summary_run: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    bodies: dict[str, Any] | None,
    raw_text: str | None,
    reason: str,
) -> None:
    """A run refused over a supplemental file leaves no file of the pass behind.

    The first run writes every file of the pass; the second is refused over a second
    supplemental file, one recording its statistic in another unit or one that is not
    JSON.  Both generators are the real ones, so an earlier run's inventory or
    collection label left beside no index is reported here, as is an earlier run's
    index; and the log gives the reason the run was refused.

    Parameters:
        summary_run: Fixture standing the subcommand up on stubs.
        monkeypatch: Fixture the dataset and the bundle root are installed through.
        tmp_path: Base temporary directory.
        capsys: Fixture capturing the log.
        bodies: The second file's body statistics, when it holds a document.
        raw_text: What the second file holds in place of a document, or None.
        reason: What the log says of the second file.
    """
    env = make_bundle_env(tmp_path / 'env', bodies=[{'name': 'latitude', 'units': 'rad'}])
    dataset = env.dataset.as_dataset()
    monkeypatch.setattr(sd_create_bundle, 'dataset_name_to_class', lambda _: lambda: dataset)
    monkeypatch.setattr(
        sd_create_bundle, 'get_pds4_bundle_results_root', lambda *a: str(env.bundle_results_root)
    )
    data_dir = env.bundle_dir / 'data'
    touch_label(data_dir, 'shard0/1111111111n')
    write_supplemental(
        data_dir,
        'shard0/1111111111n',
        bodies=_latitude_in('deg'),
        navigation=navigated_document(),
    )
    sd_create_bundle.main_summary()
    products = [env.bundle_dir / name for name in SUMMARY_PRODUCTS]
    assert [product for product in products if not product.exists()] == []
    write_supplemental(
        data_dir,
        'shard0/2222222222w',
        bodies=bodies,
        navigation=navigated_document(),
        raw_text=raw_text,
    )
    with pytest.raises(SystemExit) as excinfo:
        sd_create_bundle.main_summary()
    assert excinfo.value.code == 1
    assert [product for product in products if product.exists()] == []
    assert reason in capsys.readouterr().out


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


def test_main_summary_reports_why_the_collection_files_could_not_be_generated(
    summary_run: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A collection generator that raises ends the run with the reason in the log.

    The bundle here has no data directory, and the collection generator refuses it
    naming the directory it looked for.  The index generator, which runs first and
    refuses such a bundle the same way, is stood in for, so that the refusal the log
    reports is the collection generator's.  The frames of a traceback show the
    statement that raised but not the path it interpolated, so the path is what says
    the reason reached the log.
    """
    outcome = GlobalIndexOutcome(failed_labels=0, epochs=NoEpochRange('not taken here'))
    monkeypatch.setattr(sd_create_bundle, 'generate_global_index_files', lambda **kwargs: outcome)
    with pytest.raises(SystemExit) as excinfo:
        sd_create_bundle.main_summary()
    assert excinfo.value.code == 1
    missing = tmp_path / BUNDLE_NAME / 'data'
    expected = f'Failed to generate collection files: Data directory does not exist: {missing}'
    assert expected in capsys.readouterr().out


def test_main_summary_hands_the_index_s_range_to_the_collection_files(
    summary_run: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The index runs first, and the collection files are handed the range its scan took.

    The scan of the supplemental files is the pass's one read of them, so the range
    the data collection label states can come from nowhere else.
    """
    calls: list[str] = []
    handed: list[Any] = []
    epochs = EpochRange(start_et=100.0, stop_et=900.0)

    def _index(**kwargs: Any) -> GlobalIndexOutcome:
        """Record the call, and report the range.

        Parameters:
            **kwargs: What the driver passed, unused.

        Returns:
            An outcome carrying the range and no failed label.
        """
        calls.append('index')
        return GlobalIndexOutcome(failed_labels=0, epochs=epochs)

    def _collections(**kwargs: Any) -> int:
        """Record the call and the range it is handed, and report no failed label.

        Parameters:
            **kwargs: What the driver passed; ``epochs`` is recorded.

        Returns:
            Zero.
        """
        calls.append('collections')
        handed.append(kwargs['epochs'])
        return 0

    monkeypatch.setattr(sd_create_bundle, 'generate_global_index_files', _index)
    monkeypatch.setattr(sd_create_bundle, 'generate_collection_files', _collections)
    sd_create_bundle.main_summary()
    assert calls == ['index', 'collections']
    assert handed == [epochs]


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

    The dataset is the stub, whose configuration declares no backplanes and so
    none in a unit the worker refuses.

    Parameters:
        tmp_path: Base temporary directory served as every results root.
        monkeypatch: Fixture the stand-ins are installed through.
    """
    module = sd_create_bundle_cloud_tasks
    monkeypatch.setattr(module, 'load_default_and_user_config', lambda *a: None)
    monkeypatch.setattr(module, 'get_nav_results_root', lambda *a: str(tmp_path))
    monkeypatch.setattr(module, 'get_backplane_results_root', lambda *a: str(tmp_path))
    monkeypatch.setattr(module, 'get_pds4_bundle_results_root', lambda *a: str(tmp_path))
    dataset = _stub_dataset(tmp_path)
    monkeypatch.setattr(module, 'dataset_name_to_class', lambda _: lambda: dataset)


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
    return _process_cloud_task()


def _process_cloud_task() -> tuple[bool, Any]:
    """Run one cloud task over one image, through whatever generation is installed.

    Returns:
        The worker's retry flag and result.
    """
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


def test_a_cloud_task_refuses_a_unit_the_bundle_cannot_use(
    cloud_task_run: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A task under a configuration no index column can format generates nothing.

    The check each document gets covers only the planes that document holds, so
    a task over images whose documents hold none of these planes would otherwise
    write labels the summary pass then refuses to index.  Every unusable entry is
    named in the one result, as the two passes name each in their log.
    """
    dataset = _dataset_with_unusable_units(tmp_path)
    monkeypatch.setattr(
        sd_create_bundle_cloud_tasks, 'dataset_name_to_class', lambda _: lambda: dataset
    )
    calls = _record_generation(monkeypatch, module=sd_create_bundle_cloud_tasks)
    retry, result = _process_cloud_task()
    assert calls == []
    assert retry is False
    status = {key: value for key, value in result.items() if key != 'status_exception'}
    assert status == {'status': 'error', 'status_error': 'unusable_unit'}
    reported = result['status_exception']
    assert 'Backplane body_tilt declares a unit the bundle cannot use' in reported
    assert 'Backplane ring_radius declares a unit the bundle cannot use' in reported
    assert 'Backplane ring_tilt declares no units' in reported


def test_a_cloud_task_refuses_a_masked_value_no_float_plane_can_hold(
    cloud_task_run: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A task under a masked value no float plane can hold generates nothing.

    Every float array of every data label declares the value as its missing constant,
    so a task is refused as the labels pass is, and asks for no retry, since the
    configuration will not change on a second attempt.
    """
    dataset = _stub_dataset(tmp_path)
    dataset.config.backplanes.masked_value = float('nan')
    monkeypatch.setattr(
        sd_create_bundle_cloud_tasks, 'dataset_name_to_class', lambda _: lambda: dataset
    )
    calls = _record_generation(monkeypatch, module=sd_create_bundle_cloud_tasks)
    retry, result = _process_cloud_task()
    assert calls == []
    assert retry is False
    assert result == {
        'status': 'error',
        'status_error': 'unusable_masked_value',
        'status_exception': 'the configured masked value nan is not a finite number',
    }


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
