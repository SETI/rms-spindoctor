"""Spec-first tests for the bundle's run-level products (phase 6).

Contract under test (docs/dev_guide/dev_guide_pds4.rst "The bundle's run-level products"
and "Exit status"): ``generate_bundle_products`` copies the readme, the user guide when
the template directory holds it, the metakernel and the static collections' inventories
from the template directory into the bundle, renders a label beside each, and renders the
bundle label last, keeping it only over a bundle holding a label for every collection it
declares.  An inventory lists a product of the bundle only when that product's label is in
the bundle, and a collection left with no member is not written.  A bundle without the
user guide is one warning naming the file rather than a label not written.  Every label is
attempted, and each one not written is counted.  The summary pass clears an earlier run's
products before it writes, and, in a bundle on the local file system, the user guide's
directory with them when it is left empty.

What the shipped Cassini templates say is tested over the cohort in
``test_bundle_products_cassini_iss_saturn.py``.
"""

import shutil
from collections.abc import Sequence
from pathlib import Path

import pytest
from filecache import FCPath
from filecache import file_cache as file_cache_module
from filecache.file_cache_source import FileCacheSourceFake

from spindoctor.cli.pds4.bundle_products import clear_bundle_products, generate_bundle_products
from spindoctor.cli.pds4.epochs import EpochRange
from spindoctor.cli.pds4.global_index import generate_global_index_files
from spindoctor.cli.pds4.targets import Pds4Target, target_table
from spindoctor.config import MAIN_LOGGER

from .conftest import (
    A_RANGE,
    BROKEN_TEMPLATE,
    DEFAULT_BUNDLE_NAME,
    RUN_LEVEL_FILES,
    RUN_LEVEL_PRODUCTS,
    USER_GUIDE_NAME,
    USER_GUIDE_PDF,
    BundleEnv,
    make_bundle_env,
    read_csv_rows,
    run_collections,
    touch_browse_label,
    touch_label,
    write_supplemental,
)


def _bundle_env(
    tmp_path: Path, *, template_contents: dict[str, str] | None = None, browse: bool = True
) -> BundleEnv:
    """Build a bundle whose index and data and browse collections the pass has written.

    The index generator runs first, as in the summary pass, and writes the miscellaneous
    collection, which the bundle label declares; then the data and browse collections
    are written.  The one image has a data label and a supplemental file naming one
    body, so the bodies table has a row, which the miscellaneous collection holds.

    Parameters:
        tmp_path: Base temporary directory.
        template_contents: Template files written over the default set.
        browse: Whether the one image has a browse label, and so whether the browse
            collection is written.

    Returns:
        The environment, its collection files written.
    """
    env = make_bundle_env(tmp_path, template_contents=template_contents)
    touch_label(env.bundle_dir / 'data', 'shard0/1234567890w')
    write_supplemental(
        env.bundle_dir / 'data', 'shard0/1234567890w', bodies={'MOON': {'backplanes': {}}}
    )
    if browse:
        touch_browse_label(env.bundle_dir / 'browse', 'shard0/1234567890w')
    generate_global_index_files(
        FCPath(env.bundle_results_root), env.dataset.as_dataset(), MAIN_LOGGER
    )
    run_collections(env)
    return env


def _run(
    env: BundleEnv,
    *,
    epochs: EpochRange | None = A_RANGE,
    targets: Sequence[Pds4Target] = (),
) -> int:
    """Run generate_bundle_products over the environment's bundle.

    Parameters:
        env: The environment to process.
        epochs: The range of the products' epochs to hand the generator.
        targets: The targets the products name, to hand the generator.

    Returns:
        The number of run-level labels not written.
    """
    return generate_bundle_products(
        FCPath(env.bundle_results_root),
        env.dataset.as_dataset(),
        MAIN_LOGGER,
        epochs=epochs,
        targets=targets,
    ).failed_labels


def _lines_holding(capsys: pytest.CaptureFixture[str], text: str) -> list[str]:
    """Return the lines of the captured output that hold a text.

    Parameters:
        capsys: The fixture the output was captured by.
        text: The text sought.

    Returns:
        Each line holding ``text``, in order.
    """
    return [line for line in capsys.readouterr().out.splitlines() if text in line]


COPIES = {
    'readme.txt': 'readme.txt',
    'spice_kernels/kernels.ker': 'kernels.ker',
    'document/collection_document.csv': 'collection_document.csv',
    'spice_kernels/collection_spice_kernels.csv': 'collection_spice_kernels.csv',
    'xml_schema/collection_xml_schema.csv': 'collection_xml_schema.csv',
}
"""Each file the bundle takes from the template directory as it is, and its name there."""


def test_the_copied_products_are_the_template_directory_s_files(tmp_path: Path) -> None:
    """The readme, the metakernel and each static inventory are the template directory's.

    The template directory holds the user guide, so the document inventory is copied as
    it is too.
    """
    env = _bundle_env(tmp_path)
    _run(env)
    differing = [
        path
        for path, name in COPIES.items()
        if (env.bundle_dir / path).read_text(encoding='utf-8') != RUN_LEVEL_FILES[name]
    ]
    assert differing == []


def test_the_context_inventory_lists_each_target_after_the_template_directory_s_members(
    tmp_path: Path,
) -> None:
    """The context inventory is the template directory's lines, then one line per target.

    Each target is a secondary member at its context product's version, in the order it
    is handed, which is the targets table's.
    """
    env = _bundle_env(tmp_path)
    table = target_table(env.dataset.as_dataset().config)
    _run(env, targets=(table['PLANET'], table['MOON']))
    rows = read_csv_rows(env.bundle_dir / 'context' / 'collection_context.csv')
    assert rows == [
        ['S', 'urn:nasa:pds:context:instrument:fake::1.0'],
        ['S', 'urn:nasa:pds:context:target:fake.planet::1.0'],
        ['S', 'urn:nasa:pds:context:target:fake.moon::1.0'],
    ]


def test_a_user_guide_the_template_directory_holds_is_copied_labeled_and_listed(
    tmp_path: Path,
) -> None:
    """The guide is copied into document/user_guide/, labeled beside it, and listed P."""
    env = _bundle_env(tmp_path)
    failed = _run(env)
    assert failed == 0
    guide = env.bundle_dir / 'document' / 'user_guide' / USER_GUIDE_NAME
    assert guide.read_text(encoding='utf-8') == USER_GUIDE_PDF
    assert FCPath(guide).as_posix() in guide.with_suffix('.lblx').read_text(encoding='utf-8')
    rows = read_csv_rows(env.bundle_dir / 'document' / 'collection_document.csv')
    primaries = [row for row in rows if row[0] == 'P']
    lidvid = f'urn:nasa:pds:{DEFAULT_BUNDLE_NAME}:document:fake-user-guide::1.0'
    assert primaries == [['P', lidvid]]


def test_a_user_guide_the_template_directory_lacks_is_neither_copied_nor_listed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Without the guide, nothing goes into document/user_guide/ and no P line is listed.

    That is not a label the run failed to write but one warning, naming the file.  The
    document inventory's other lines are still listed.
    """
    env = _bundle_env(tmp_path)
    source = Path(env.dataset.pds4_bundle_template_dir()) / USER_GUIDE_NAME
    source.unlink()
    failed = _run(env)
    assert failed == 0
    assert not (env.bundle_dir / 'document' / 'user_guide').exists()
    rows = read_csv_rows(env.bundle_dir / 'document' / 'collection_document.csv')
    assert rows == [['S', 'urn:nasa:pds:context:instrument:fake::1.0']]
    warnings = _lines_holding(capsys, 'is not in the template directory')
    assert len(warnings) == 1
    assert '| WARNING |' in warnings[0]
    assert FCPath(source).as_posix() in warnings[0]


def test_a_user_guide_whose_label_was_not_written_is_not_listed(tmp_path: Path) -> None:
    """A guide whose label failed to render is left out of the document inventory.

    An inventory lists a product of the bundle only when the product's label is in the
    bundle, so the guide's line goes, as when the template directory holds no guide, and
    the inventory's other members stay.
    """
    env = _bundle_env(tmp_path, template_contents={'fake-user-guide.lblx': BROKEN_TEMPLATE})
    _run(env)
    rows = read_csv_rows(env.bundle_dir / 'document' / 'collection_document.csv')
    assert rows == [['S', 'urn:nasa:pds:context:instrument:fake::1.0']]


def test_a_spice_kernel_collection_whose_metakernel_label_was_not_written_is_not_written(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """With no metakernel label, the SPICE kernel collection has no member and no files.

    Its one member is the metakernel, so its inventory would list nothing: neither the
    inventory nor the label is written, what an earlier run left at either path is
    removed, and one error names the collection.  That it counts, and costs the bundle
    label, is the metakernel case of the failed-label test.
    """
    env = _bundle_env(tmp_path, template_contents={'kernels.lblx': BROKEN_TEMPLATE})
    earlier = [
        env.bundle_dir / 'spice_kernels' / name
        for name in ('collection_spice_kernels.csv', 'collection_spice_kernels.lblx')
    ]
    earlier[0].parent.mkdir(parents=True)
    for path in earlier:
        path.write_text('an earlier run\n', encoding='utf-8')
    _run(env)
    assert [path for path in earlier if path.exists()] == []
    errors = _lines_holding(capsys, 'The spice_kernels collection was not written')
    assert len(errors) == 1
    assert '| ERROR |' in errors[0]


def test_a_rerun_without_the_user_guide_leaves_no_user_guide_directory(tmp_path: Path) -> None:
    """A rerun over a template directory that no longer holds the guide leaves no directory.

    The summary pass clears an earlier run's guide and its label before it writes, and
    the directory they were in, left empty, goes with them.
    """
    env = _bundle_env(tmp_path)
    _run(env)
    (Path(env.dataset.pds4_bundle_template_dir()) / USER_GUIDE_NAME).unlink()
    generate_global_index_files(
        FCPath(env.bundle_results_root), env.dataset.as_dataset(), MAIN_LOGGER
    )
    _run(env)
    assert not (env.bundle_dir / 'document' / 'user_guide').exists()


def test_clearing_takes_a_bundle_root_relative_to_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The run-level products clear under a bundle root named relative to where it runs.

    A bundle results root can be given relative to the working directory, and the user
    guide's directory, emptied, is removed under one too.
    """
    env = _bundle_env(tmp_path)
    _run(env)
    monkeypatch.chdir(env.bundle_results_root)
    clear_bundle_products(FCPath(DEFAULT_BUNDLE_NAME), env.dataset.as_dataset())
    assert not (env.bundle_dir / 'document' / 'user_guide').exists()


def test_clearing_takes_a_bundle_root_in_a_remote_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The run-level products clear under a bundle root that is not on the local disk.

    A ``fake://`` URL is served out of a local directory but answers as a remote store
    does, so the clearing takes the branch a bucket takes: every run-level product is
    removed, and no directory is, a remote store holding none.  The storage directory is
    installed through ``monkeypatch``, with the storage layer's memory of the backends it
    has built, since a backend keeps the directory it was built with.
    """
    env = _bundle_env(tmp_path)
    _run(env)
    storage = tmp_path / 'remote'
    monkeypatch.setattr(FileCacheSourceFake, '_DEFAULT_STORAGE_DIR', storage)
    monkeypatch.setattr(file_cache_module, '_SOURCE_CACHE', {})
    shutil.copytree(env.bundle_results_root, storage / 'bucket' / 'bundles')
    remote_root = FCPath(f'fake://bucket/bundles/{DEFAULT_BUNDLE_NAME}')
    clear_bundle_products(remote_root, env.dataset.as_dataset())
    backing = storage / 'bucket' / 'bundles' / DEFAULT_BUNDLE_NAME
    assert [product for product in RUN_LEVEL_PRODUCTS if (backing / product).exists()] == []


def test_clearing_takes_a_bundle_root_given_as_a_string(tmp_path: Path) -> None:
    """The run-level products clear under a bundle root given as a plain string."""
    env = _bundle_env(tmp_path)
    _run(env)
    clear_bundle_products(str(env.bundle_dir), env.dataset.as_dataset())
    assert [product for product in RUN_LEVEL_PRODUCTS if (env.bundle_dir / product).exists()] == []


def test_the_bundle_label_states_the_bundle_s_lid_and_the_range_it_is_handed(
    tmp_path: Path,
) -> None:
    """Over a bundle holding every collection it declares, the bundle label is written.

    It states the bundle's LID and the range of the products' epochs in whole seconds,
    as the data collection label does.
    """
    env = _bundle_env(tmp_path)
    _run(env)
    text = (env.bundle_dir / 'bundle.lblx').read_text(encoding='utf-8')
    assert f'<lid>urn:nasa:pds:{DEFAULT_BUNDLE_NAME}</lid>' in text
    assert '<start>2004-02-07T04:25:35Z</start>' in text
    assert '<stop>2004-02-22T05:32:17Z</stop>' in text


def test_a_bundle_label_declaring_a_collection_not_written_is_not_written(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A collection the bundle label declares and the bundle lacks keeps the label off disk.

    The image has no browse label, so the browse collection was not written; the bundle
    label, which declares it, is removed once rendered and counted, and an error names
    the collection it lacks.
    """
    env = _bundle_env(tmp_path, browse=False)
    failed = _run(env)
    assert failed == 1
    assert not (env.bundle_dir / 'bundle.lblx').exists()
    missing = f'it declares urn:nasa:pds:{DEFAULT_BUNDLE_NAME}:browse, and the bundle holds no'
    errors = _lines_holding(capsys, missing)
    assert len(errors) == 1
    assert '| ERROR |' in errors[0]


def test_a_bundle_label_with_no_range_to_state_is_not_written(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """With no time range the bundle label is not rendered, and an earlier one is removed."""
    env = _bundle_env(tmp_path)
    earlier = env.bundle_dir / 'bundle.lblx'
    earlier.write_text('an earlier run\n', encoding='utf-8')
    failed = _run(env, epochs=None)
    assert failed == 1
    assert not earlier.exists()
    errors = _lines_holding(capsys, 'so there is no time range for it to state')
    assert len(errors) == 1
    assert '| ERROR |' in errors[0]


@pytest.mark.parametrize(
    ('broken', 'absent', 'failed'),
    [
        ('fake-user-guide.lblx', {'document/user_guide/fake-user-guide.lblx'}, 1),
        (
            'kernels.lblx',
            {
                'spice_kernels/kernels.lblx',
                'spice_kernels/collection_spice_kernels.csv',
                'spice_kernels/collection_spice_kernels.lblx',
                'bundle.lblx',
            },
            3,
        ),
        ('collection_context.lblx', {'context/collection_context.lblx', 'bundle.lblx'}, 2),
        ('bundle.lblx', {'bundle.lblx'}, 1),
    ],
    ids=['user guide', 'metakernel', 'static collection', 'bundle'],
)
def test_a_run_level_label_that_fails_to_render_is_counted(
    tmp_path: Path, broken: str, absent: set[str], failed: int
) -> None:
    """A run-level label that fails is counted, and every other run-level product is written.

    Each is a case of its own because each is counted by a statement of its own.  Every
    label is attempted whichever fail, so each case holds every run-level product but the
    failed label and what its failure takes with it.  The metakernel's takes the SPICE
    kernel collection, which then has no member; a static collection's leaves that
    collection without its label; and either takes the bundle label, which declares the
    collection.  Each of those counts.

    Parameters:
        tmp_path: Base temporary directory.
        broken: The template whose render errors in this case.
        absent: The run-level products not on disk afterwards.
        failed: The labels not written.
    """
    env = _bundle_env(tmp_path, template_contents={broken: BROKEN_TEMPLATE})
    assert _run(env) == failed
    on_disk = {product for product in RUN_LEVEL_PRODUCTS if (env.bundle_dir / product).exists()}
    assert on_disk == set(RUN_LEVEL_PRODUCTS) - absent
