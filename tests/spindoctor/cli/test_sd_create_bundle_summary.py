"""Tests that a summary run which did not write every label says so in its result.

``pdstemplate`` reports a label it could not render through its return value
rather than by raising, so a run that dropped one looks exactly like a run that
did not, unless the driver acts on what came back to it.  These pin that for the
summary subcommand: what it does with a collection, index or run-level label it
failed to write, with an image whose products disagree, and with a generator that
raises, and the order it runs its generators in.  The labels subcommand's are in
``test_sd_create_bundle.py``.

The subcommand is stood up on stubs, because what is under test is the counting
and the report, not the configuration loading, the logging setup or the
generation it reads; the tests that need the generators' own output run the real
ones over a hermetic bundle.
"""

import argparse
from pathlib import Path
from typing import Any

import pdstemplate
import pytest
from tests.spindoctor.cli.pds4.conftest import (
    RUN_LEVEL_PRODUCTS,
    make_bundle_env,
    touch_browse_label,
    touch_label,
    write_supplemental,
)
from tests.spindoctor.cli.sd_create_bundle_helpers import BUNDLE_NAME, stub_dataset

from spindoctor.cli import sd_create_bundle
from spindoctor.cli.pds4.bundle_products import BundleProductsOutcome
from spindoctor.cli.pds4.collections import CollectionOutcome
from spindoctor.cli.pds4.epochs import EpochRange
from spindoctor.cli.pds4.global_index import GlobalIndexOutcome

SUMMARY_PRODUCTS = (
    'data/collection_data.csv',
    'data/collection_data.lblx',
    'browse/collection_browse.csv',
    'browse/collection_browse.lblx',
    'miscellaneous/global_bodies_index.tab',
    'miscellaneous/global_bodies_index.lblx',
    'miscellaneous/global_rings_index.tab',
    'miscellaneous/global_rings_index.lblx',
    *RUN_LEVEL_PRODUCTS,
)
"""Every file the summary pass writes over a template directory holding the user guide."""


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
        lambda _: argparse.Namespace(dataset_name='stub'),
    )
    monkeypatch.setattr(sd_create_bundle, 'load_default_and_user_config', lambda *a: None)
    monkeypatch.setattr(sd_create_bundle, 'build_run_logging', lambda *a: None)
    monkeypatch.setattr(sd_create_bundle, 'get_pds4_bundle_results_root', lambda *a: str(tmp_path))
    monkeypatch.setattr(pdstemplate.PdsTemplate, 'set_logger', staticmethod(lambda *a: None))
    dataset = stub_dataset(tmp_path)
    monkeypatch.setattr(sd_create_bundle, 'dataset_name_to_class', lambda _: lambda: dataset)


def _summary_counts(
    monkeypatch: pytest.MonkeyPatch,
    collections: int,
    index: int,
    *,
    disagreeing: int = 0,
    bundle: int = 0,
) -> None:
    """Make the three summary generators report the given counts.

    Parameters:
        monkeypatch: Fixture the stand-ins are installed through.
        collections: Failed collection labels to report.
        index: Failed global index labels to report.
        disagreeing: Images whose products disagree, for the collection generator to
            report.
        bundle: Failed run-level labels to report.
    """
    index_outcome = GlobalIndexOutcome(failed_labels=index, epochs=None)
    collection_outcome = CollectionOutcome(
        failed_labels=collections, disagreeing_images=disagreeing
    )
    bundle_outcome = BundleProductsOutcome(failed_labels=bundle)
    monkeypatch.setattr(
        sd_create_bundle, 'generate_collection_files', lambda **kwargs: collection_outcome
    )
    monkeypatch.setattr(
        sd_create_bundle, 'generate_global_index_files', lambda **kwargs: index_outcome
    )
    monkeypatch.setattr(
        sd_create_bundle, 'generate_bundle_products', lambda **kwargs: bundle_outcome
    )


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


def _latitude_in(units: str) -> dict[str, Any]:
    """Return one body's latitude statistic, recorded in the given unit.

    Parameters:
        units: The unit the statistic records.  The configuration the test gives its
            dataset declares the latitude plane in radians, whose statistics are
            taken in degrees.

    Returns:
        The ``backplanes.bodies`` payload of a supplemental file.
    """
    return {'MOON': {'backplanes': {'latitude': {'min': -1.2, 'max': 1.4, 'units': units}}}}


def test_a_refused_summary_leaves_no_product_an_earlier_summary_wrote(
    summary_run: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A run refused over a supplemental file leaves no file of the pass behind.

    The first run writes every file of the pass; the second is refused over a second
    supplemental file, one recording its statistic in another unit.  The generators
    are the real ones, so an earlier run's inventory or collection label left beside
    no index is reported here, as is an earlier run's index or run-level product; and
    the log gives the reason the run was refused.
    """
    env = make_bundle_env(tmp_path / 'env', bodies=[{'name': 'latitude', 'units': 'rad'}])
    dataset = env.dataset.as_dataset()
    monkeypatch.setattr(sd_create_bundle, 'dataset_name_to_class', lambda _: lambda: dataset)
    monkeypatch.setattr(
        sd_create_bundle, 'get_pds4_bundle_results_root', lambda *a: str(env.bundle_results_root)
    )
    data_dir = env.bundle_dir / 'data'
    touch_label(data_dir, 'shard0/1111111111n')
    touch_browse_label(env.bundle_dir / 'browse', 'shard0/1111111111n')
    write_supplemental(data_dir, 'shard0/1111111111n', bodies=_latitude_in('deg'))
    sd_create_bundle.main_summary()
    products = [env.bundle_dir / name for name in SUMMARY_PRODUCTS]
    assert [product for product in products if not product.exists()] == []
    write_supplemental(data_dir, 'shard0/2222222222w', bodies=_latitude_in('rad'))
    with pytest.raises(SystemExit) as excinfo:
        sd_create_bundle.main_summary()
    assert excinfo.value.code == 1
    assert [product for product in products if product.exists()] == []
    expected = 'records the latitude statistic in rad where the configuration expects deg'
    assert expected in capsys.readouterr().out


def test_a_summary_over_no_data_label_exits_one_and_writes_neither_collection(
    summary_run: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Supplemental files and no data label end the summary non-zero, with no collection.

    This is the tree a labels pass leaves when every data label fails to render, since
    it writes an image's supplemental file before the image's data label.  The index
    then has a range to hand on, but neither collection has a member, and a collection
    label states at least one record, so neither collection is written, each counts
    among the labels not written, and the log names each; so does the bundle label,
    which declares both.  The image, a supplemental file with no data label, is one
    whose products disagree, and the closing line counts it beside the labels.  The
    generators are the real ones.
    """
    env = make_bundle_env(tmp_path / 'env')
    dataset = env.dataset.as_dataset()
    monkeypatch.setattr(sd_create_bundle, 'dataset_name_to_class', lambda _: lambda: dataset)
    monkeypatch.setattr(
        sd_create_bundle, 'get_pds4_bundle_results_root', lambda *a: str(env.bundle_results_root)
    )
    write_supplemental(env.bundle_dir / 'data', 'shard0/1111111111n')
    with pytest.raises(SystemExit) as excinfo:
        sd_create_bundle.main_summary()
    assert excinfo.value.code == 1
    not_written = ('data/collection_', 'browse/collection_', 'bundle.lblx')
    unwritten = [env.bundle_dir / name for name in SUMMARY_PRODUCTS if name.startswith(not_written)]
    assert [product for product in unwritten if product.exists()] == []
    out = capsys.readouterr().out
    closing = (
        'Summary generation incomplete: 3 label(s) were not written, '
        '1 image(s) whose products disagree'
    )
    assert closing in out
    assert 'The data collection was not written' in out
    assert 'The browse collection was not written' in out


@pytest.mark.parametrize(
    ('collections', 'index', 'disagreeing', 'bundle'),
    [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)],
    ids=['collection label', 'index label', 'disagreeing image', 'run-level label'],
)
def test_main_summary_exits_non_zero_when_a_count_is_not_zero(
    summary_run: None,
    monkeypatch: pytest.MonkeyPatch,
    collections: int,
    index: int,
    disagreeing: int,
    bundle: int,
) -> None:
    """A label not written, or an image whose products disagree, ends the run non-zero.

    Each count is a case of its own because the run has to act on each: a summary
    that reported only one generator's failed labels, or only failed labels and no
    disagreeing image, would still pass a test that never gave it the other.

    Parameters:
        summary_run: Fixture standing the subcommand up on stubs.
        monkeypatch: Fixture the counts are installed through.
        collections: Failed collection labels for this case.
        index: Failed global index labels for this case.
        disagreeing: Images whose products disagree, for this case.
        bundle: Failed run-level labels for this case.
    """
    _summary_counts(
        monkeypatch, collections=collections, index=index, disagreeing=disagreeing, bundle=bundle
    )
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
    unreached = CollectionOutcome(failed_labels=0, disagreeing_images=0)
    monkeypatch.setattr(sd_create_bundle, 'generate_collection_files', lambda **kwargs: unreached)

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
    outcome = GlobalIndexOutcome(failed_labels=0, epochs=None)
    monkeypatch.setattr(sd_create_bundle, 'generate_global_index_files', lambda **kwargs: outcome)
    with pytest.raises(SystemExit) as excinfo:
        sd_create_bundle.main_summary()
    assert excinfo.value.code == 1
    missing = tmp_path / BUNDLE_NAME / 'data'
    expected = f'Failed to generate collection files: Data directory does not exist: {missing}'
    assert expected in capsys.readouterr().out


def test_main_summary_reports_why_the_bundle_products_could_not_be_generated(
    summary_run: None, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A run-level generator that raises ends the run with the reason in the log.

    A file the template directory does not hold is refused naming it, which the frames
    of a traceback do not carry.
    """
    _summary_counts(monkeypatch, collections=0, index=0)

    def _refuse(**kwargs: Any) -> BundleProductsOutcome:
        """Refuse the way a file missing from the template directory is refused.

        Parameters:
            **kwargs: What the driver passed, unused.

        Raises:
            FileNotFoundError: Always, naming the file.
        """
        raise FileNotFoundError('No such file: templates/readme.txt')

    monkeypatch.setattr(sd_create_bundle, 'generate_bundle_products', _refuse)
    with pytest.raises(SystemExit) as excinfo:
        sd_create_bundle.main_summary()
    assert excinfo.value.code == 1
    expected = 'Failed to generate the bundle products: No such file: templates/readme.txt'
    assert expected in capsys.readouterr().out


def test_main_summary_hands_the_index_s_range_to_the_generators_after_it(
    summary_run: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The index runs first, and the collections and then the run-level products get its range.

    The scan of the supplemental files is the pass's one read of them, so the range
    the data collection label and the bundle label state can come from nowhere else.
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

    def _collections(**kwargs: Any) -> CollectionOutcome:
        """Record the call and the range it is handed, and report nothing counted.

        Parameters:
            **kwargs: What the driver passed; ``epochs`` is recorded.

        Returns:
            An outcome with no failed label and no disagreeing image.
        """
        calls.append('collections')
        handed.append(kwargs['epochs'])
        return CollectionOutcome(failed_labels=0, disagreeing_images=0)

    def _bundle(**kwargs: Any) -> BundleProductsOutcome:
        """Record the call and the range it is handed, and report nothing counted.

        Parameters:
            **kwargs: What the driver passed; ``epochs`` is recorded.

        Returns:
            An outcome with no failed label.
        """
        calls.append('bundle')
        handed.append(kwargs['epochs'])
        return BundleProductsOutcome(failed_labels=0)

    monkeypatch.setattr(sd_create_bundle, 'generate_global_index_files', _index)
    monkeypatch.setattr(sd_create_bundle, 'generate_collection_files', _collections)
    monkeypatch.setattr(sd_create_bundle, 'generate_bundle_products', _bundle)
    sd_create_bundle.main_summary()
    assert calls == ['index', 'collections', 'bundle']
    assert handed == [epochs, epochs]


def test_main_summary_exits_zero_when_every_label_is_written(
    summary_run: None, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A summary that wrote every label reports completion and does not end the process."""
    _summary_counts(monkeypatch, collections=0, index=0)
    sd_create_bundle.main_summary()
    assert 'Summary generation complete' in capsys.readouterr().out
