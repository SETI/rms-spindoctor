"""Spec-first tests for the bundle's run-level products (phase 6).

Contract under test (docs/dev_guide/dev_guide_pds4.rst "Pipeline overview" and "Exit
status"): ``generate_bundle_products`` copies the readme, the user guide when the template
directory holds it, the metakernel and the static collections' inventories from the
template directory into the bundle, renders a label beside each, and renders the bundle
label last, keeping it only over a bundle holding a label for every collection it declares.
The document inventory lists the user guide only when the bundle holds it, and a bundle
without it is one warning naming the file rather than a label not written.  Every label is
attempted, and each one not written is counted.

What the shipped Cassini templates say is tested over the cohort in
``test_bundle_products_cassini_iss_saturn.py``.
"""

from pathlib import Path

import pytest
from filecache import FCPath

from spindoctor.cli.pds4.bundle_products import generate_bundle_products
from spindoctor.cli.pds4.epochs import EpochRange
from spindoctor.config import MAIN_LOGGER

from .conftest import (
    A_RANGE,
    BROKEN_TEMPLATE,
    DEFAULT_BUNDLE_NAME,
    RUN_LEVEL_FILES,
    USER_GUIDE_NAME,
    USER_GUIDE_PDF,
    BundleEnv,
    make_bundle_env,
    read_csv_rows,
    run_collections,
    touch_browse_label,
    touch_label,
)


def _bundle_env(
    tmp_path: Path, *, template_contents: dict[str, str] | None = None, browse: bool = True
) -> BundleEnv:
    """Build a bundle whose data and browse collections the summary pass has written.

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
    if browse:
        touch_browse_label(env.bundle_dir / 'browse', 'shard0/1234567890w')
    run_collections(env)
    return env


def _run(env: BundleEnv, *, epochs: EpochRange | None = A_RANGE) -> int:
    """Run generate_bundle_products over the environment's bundle.

    Parameters:
        env: The environment to process.
        epochs: The range of the products' epochs to hand the generator.

    Returns:
        The number of run-level labels not written.
    """
    return generate_bundle_products(
        FCPath(env.bundle_results_root), env.dataset.as_dataset(), MAIN_LOGGER, epochs=epochs
    ).failed_labels


COPIES = {
    'readme.txt': 'readme.txt',
    'spice_kernels/kernels.ker': 'kernels.ker',
    'context/collection_context.csv': 'collection_context.csv',
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


def test_a_user_guide_the_template_directory_holds_is_copied_labeled_and_listed(
    tmp_path: Path,
) -> None:
    """The guide is copied into document/user_guide/, labeled beside it, and listed P."""
    env = _bundle_env(tmp_path)
    failed = _run(env)
    assert failed == 0
    guide = env.bundle_dir / 'document' / 'user_guide' / USER_GUIDE_NAME
    assert guide.read_text(encoding='utf-8') == USER_GUIDE_PDF
    assert str(FCPath(guide)) in guide.with_suffix('.lblx').read_text(encoding='utf-8')
    rows = read_csv_rows(env.bundle_dir / 'document' / 'collection_document.csv')
    assert rows[0] == ['P', f'urn:nasa:pds:{DEFAULT_BUNDLE_NAME}:document:fake-user-guide::1.0']


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
    warnings = [
        line
        for line in capsys.readouterr().out.splitlines()
        if 'is not in the template directory' in line
    ]
    assert len(warnings) == 1
    assert '| WARNING |' in warnings[0]
    assert str(FCPath(source)) in warnings[0]


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
    label, which declares it, is removed once rendered and counted, and the error names
    the collection it lacks.
    """
    env = _bundle_env(tmp_path, browse=False)
    failed = _run(env)
    assert failed == 1
    assert not (env.bundle_dir / 'bundle.lblx').exists()
    missing = f'it declares urn:nasa:pds:{DEFAULT_BUNDLE_NAME}:browse, and the bundle holds no'
    assert missing in capsys.readouterr().out


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
    assert 'so there is no time range for it to state' in capsys.readouterr().out


@pytest.mark.parametrize(
    ('broken', 'label', 'kept', 'failed'),
    [
        (
            'fake-user-guide.lblx',
            'document/user_guide/fake-user-guide.lblx',
            'document/user_guide/fake-user-guide.pdf',
            1,
        ),
        ('kernels.lblx', 'spice_kernels/kernels.lblx', 'spice_kernels/kernels.ker', 1),
        (
            'collection_context.lblx',
            'context/collection_context.lblx',
            'context/collection_context.csv',
            2,
        ),
        ('bundle.lblx', 'bundle.lblx', 'readme.txt', 1),
    ],
    ids=['user guide', 'metakernel', 'static collection', 'bundle'],
)
def test_a_run_level_label_that_fails_to_render_is_counted(
    tmp_path: Path, broken: str, label: str, kept: str, failed: int
) -> None:
    """A run-level label that fails to render is off disk and counted; what it describes stays.

    Each is a case of its own because each is counted by a statement of its own.  A static
    collection whose label fails is one the bundle label declares and the bundle holds no
    label for, so the bundle label is not kept either, and that case counts two.

    Parameters:
        tmp_path: Base temporary directory.
        broken: The template whose render errors in this case.
        label: Where the label it renders goes.
        kept: The file that label describes, which stays.
        failed: The labels not written.
    """
    env = _bundle_env(tmp_path, template_contents={broken: BROKEN_TEMPLATE})
    assert _run(env) == failed
    assert not (env.bundle_dir / label).exists()
    assert (env.bundle_dir / kept).is_file()
