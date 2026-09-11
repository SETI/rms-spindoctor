"""Spec-first tests for PDS4 collection inventories and global index files (phase 2).

Contract under test (docs/user_guide/user_guide_pds4_bundle.rst "Summary Pass" /
"Summary Pass Outputs" and docs/dev_guide/dev_guide_pds4.rst "Pipeline overview"):
``generate_collection_files`` scans the bundle's ``data/`` tree for
``*_backplanes.lblx`` labels, sorts them by image name, and writes the
``collection_data.tab`` / ``collection_browse.tab`` inventories (``Member
Status`` + ``LIDVID_LID`` columns, one ``P`` row per product, LIDVIDs from the
dataset's ``pds4_image_name_to_*_lidvid`` builders) plus the matching
``.lblx`` labels.  ``generate_global_index_files``
scans ``data/`` for ``*_supplemental.txt`` files and writes
``document/supplemental/global_index_bodies.tab`` (one row per image/body) and
``global_index_rings.tab`` (one row per image with ring backplanes), with
min/max columns for each configured backplane type, each written in the format
its unit calls for, plus their labels.  Every template a generator renders is required: the
drivers check the ones their dataset declares before processing anything, so one
that is missing raises here rather than being passed over.

Both generators recover the original image name from each on-disk product stem
via ``DataSet.pds4_lid_part_to_image_name`` before building LIDs, so the stem
(which is already in LID-part form) is not fed back through the image-name
transform.  This is what keeps the inventory LIDVIDs and global-index LIDs
matching the product labels' DATA_LID (regression coverage for #139 and #256).
"""

import math
from pathlib import Path
from typing import Any

import pytest
from filecache import FCPath

from spindoctor.cli.pds4 import collections as collections_module
from spindoctor.cli.pds4.collections import (
    IndexValueFormat,
    generate_collection_files,
    generate_global_index_files,
)
from spindoctor.config import MAIN_LOGGER
from spindoctor.dataset.dataset_pds3_cassini_iss import DataSetPDS3CassiniISSSaturn

from .conftest import (
    COLLECTION_BROWSE_TEMPLATE,
    COLLECTION_DATA_TEMPLATE,
    GLOBAL_INDEX_TEMPLATE,
    BundleEnv,
    make_bundle_env,
    read_tab,
    touch_label,
    write_supplemental,
    write_templates,
)

BODY_STATS = {'MIMAS': {'backplanes': {'latitude': {'min': 1.234567891, 'max': 2, 'units': 'deg'}}}}
"""A body's statistics, in the unit the default configuration's latitude plane takes."""
RING_STATS = {'backplanes': {'radius': {'min': 74500.0, 'max': 136800.987654, 'units': 'km'}}}
"""Ring statistics, in the unit the default configuration's radius plane takes."""
BROKEN_TEMPLATE = '<Broken>$COMPLETELY_UNSET_VARIABLE$</Broken>\n'
"""A template naming a variable no caller defines, so the render errors."""
COLLECTION_LABELS = {
    'collection_data.lblx': ('data', COLLECTION_DATA_TEMPLATE),
    'collection_browse.lblx': ('browse', COLLECTION_BROWSE_TEMPLATE),
}
"""Each collection label's bundle subdirectory and its intact template body."""


def _run_collections(env: BundleEnv) -> int:
    """Run generate_collection_files against the environment's bundle root.

    Parameters:
        env: The hermetic bundle environment to process.

    Returns:
        The number of collection labels that could not be rendered.
    """
    return generate_collection_files(
        FCPath(env.bundle_results_root), env.dataset.as_dataset(), MAIN_LOGGER
    )


def _run_global_index(env: BundleEnv) -> int:
    """Run generate_global_index_files against the environment's bundle root.

    Parameters:
        env: The hermetic bundle environment to process.

    Returns:
        The number of index labels that could not be rendered.
    """
    return generate_global_index_files(
        FCPath(env.bundle_results_root), env.dataset.as_dataset(), MAIN_LOGGER
    )


def _index_env(
    tmp_path: Path,
    *,
    bodies: list[dict[str, Any]] | None = None,
    rings: list[dict[str, Any]] | None = None,
) -> BundleEnv:
    """Build an environment with body/ring backplane types configured.

    Parameters:
        tmp_path: Base temporary directory.
        bodies: ``config.backplanes.bodies`` entries, each with a ``name`` and
            the ``units`` its plane is declared in.  When None, a 'latitude' in
            radians and a 'resolution' in kilometers per pixel.
        rings: ``config.backplanes.rings`` entries on the same terms.  When
            None, a 'radius' in kilometers.

    Returns:
        A :class:`BundleEnv` whose config lists those backplanes.
    """
    if bodies is None:
        bodies = [
            {'name': 'latitude', 'units': 'rad'},
            {'name': 'resolution', 'units': 'km/pixel'},
        ]
    if rings is None:
        rings = [{'name': 'radius', 'units': 'km'}]
    return make_bundle_env(tmp_path, bodies=bodies, rings=rings)


# ---------------------------------------------------------------------------
# generate_collection_files: inventories
# ---------------------------------------------------------------------------


def test_missing_data_dir_raises(tmp_path: Path) -> None:
    """A bundle without a data directory is rejected with FileNotFoundError."""
    env = make_bundle_env(tmp_path)
    with pytest.raises(FileNotFoundError, match='Data directory does not exist'):
        _run_collections(env)


def test_empty_data_dir_writes_header_only_inventories(tmp_path: Path) -> None:
    """An empty data tree yields inventories with only the header row."""
    env = make_bundle_env(tmp_path)
    (env.bundle_dir / 'data').mkdir(parents=True)
    _run_collections(env)
    data_rows = read_tab(env.bundle_dir / 'data' / 'collection_data.tab')
    assert data_rows == [['Member Status', 'LIDVID_LID']]
    browse_rows = read_tab(env.bundle_dir / 'browse' / 'collection_browse.tab')
    assert browse_rows == [['Member Status', 'LIDVID_LID']]


def test_data_inventory_row_per_label_with_primary_status(tmp_path: Path) -> None:
    """Each *_backplanes.lblx yields one P row with the dataset's data LIDVID."""
    env = make_bundle_env(tmp_path)
    touch_label(env.bundle_dir / 'data', 'shard0/1234567890w')
    _run_collections(env)
    rows = read_tab(env.bundle_dir / 'data' / 'collection_data.tab')
    assert len(rows) == 2
    assert rows[1] == ['P', 'urn:nasa:pds:fake_bundle:data:1234567890w::1.0']


def test_browse_inventory_uses_browse_lidvids(tmp_path: Path) -> None:
    """The browse inventory lists the same images with browse LIDVIDs."""
    env = make_bundle_env(tmp_path)
    touch_label(env.bundle_dir / 'data', 'shard0/1234567890w')
    _run_collections(env)
    rows = read_tab(env.bundle_dir / 'browse' / 'collection_browse.tab')
    assert rows[1] == ['P', 'urn:nasa:pds:fake_bundle:browse:1234567890w::1.0']


def test_inventory_rows_sorted_by_image_name_not_path(tmp_path: Path) -> None:
    """Inventory rows sort by extracted image name, ignoring shard directories."""
    env = make_bundle_env(tmp_path)
    touch_label(env.bundle_dir / 'data', 'zz9/1111111111n')
    touch_label(env.bundle_dir / 'data', 'aa0/2222222222w')
    _run_collections(env)
    rows = read_tab(env.bundle_dir / 'data' / 'collection_data.tab')
    assert rows[1][1] == 'urn:nasa:pds:fake_bundle:data:1111111111n::1.0'
    assert rows[2][1] == 'urn:nasa:pds:fake_bundle:data:2222222222w::1.0'


def test_duplicate_image_names_produce_duplicate_rows(tmp_path: Path) -> None:
    """Characterization: the same image name in two shards is listed twice."""
    env = make_bundle_env(tmp_path)
    touch_label(env.bundle_dir / 'data', 'shard0/1234567890w')
    touch_label(env.bundle_dir / 'data', 'shard1/1234567890w')
    _run_collections(env)
    rows = read_tab(env.bundle_dir / 'data' / 'collection_data.tab')
    assert len(rows) == 3
    assert rows[1] == rows[2]


def test_non_backplane_label_files_ignored(tmp_path: Path) -> None:
    """Only *_backplanes.lblx files are inventoried from the data tree."""
    env = make_bundle_env(tmp_path)
    other = env.bundle_dir / 'data' / 'shard0' / '1234567890w_other.lblx'
    other.parent.mkdir(parents=True)
    other.write_text('<x/>\n', encoding='utf-8')
    _run_collections(env)
    rows = read_tab(env.bundle_dir / 'data' / 'collection_data.tab')
    assert rows == [['Member Status', 'LIDVID_LID']]


# ---------------------------------------------------------------------------
# generate_collection_files: labels
# ---------------------------------------------------------------------------


def test_collection_labels_rendered_when_templates_exist(tmp_path: Path) -> None:
    """Collection labels render with the CSV path and (empty) date-range variables."""
    env = make_bundle_env(
        tmp_path,
        template_contents={
            'collection_data.lblx': COLLECTION_DATA_TEMPLATE,
            'collection_browse.lblx': COLLECTION_BROWSE_TEMPLATE,
        },
    )
    touch_label(env.bundle_dir / 'data', 'shard0/1234567890w')
    failed = _run_collections(env)
    assert failed == 0
    data_label = env.bundle_dir / 'data' / 'collection_data.lblx'
    text = data_label.read_text(encoding='utf-8')
    assert str(FCPath(env.bundle_dir) / 'data' / 'collection_data.tab') in text
    assert '<start></start>' in text
    assert '<stop></stop>' in text
    browse_label = env.bundle_dir / 'browse' / 'collection_browse.lblx'
    browse_text = browse_label.read_text(encoding='utf-8')
    assert str(FCPath(env.bundle_dir) / 'browse' / 'collection_browse.tab') in browse_text


@pytest.mark.parametrize(
    ('label', 'subdir'),
    [('collection_data.lblx', 'data'), ('collection_browse.lblx', 'browse')],
    ids=['data collection', 'browse collection'],
)
def test_a_collection_label_is_written_to_the_bundle_not_to_a_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, label: str, subdir: str
) -> None:
    """Each collection label writer is handed the bundle's own path for the label.

    The location an ``FCPath`` names and the location of its local cache copy
    are the same file only while the bundle root is local.  On a cloud bundle
    root they are two different files, so a caller that hands over the cache
    path writes the label into the cache rather than into the bundle, and names
    in its error report a path that is nowhere in the bundle.  A local bundle
    root cannot tell those apart by the file that appears, so what the writer is
    handed is what says which one it was.  Each label is a case of its own
    because each is handed over by a statement of its own.

    Parameters:
        tmp_path: Base temporary directory.
        monkeypatch: Fixture the recording label writer is installed through.
        label: The collection label this case checks.
        subdir: The bundle subdirectory that label belongs in.
    """
    handed_over: dict[str, Any] = {}

    def _record(
        template: Any, template_vars: dict[str, Any], label_path: Any, *, logger: Any
    ) -> bool:
        """Record the path handed over and report the label as written.

        Parameters:
            template: The parsed template, unused.
            template_vars: The template variables, unused.
            label_path: The path whose type and value are under test.
            logger: The logger the caller passed, unused.

        Returns:
            True, so the caller counts the label as written.
        """
        handed_over[FCPath(label_path).name] = label_path
        return True

    env = make_bundle_env(
        tmp_path,
        template_contents={
            'collection_data.lblx': COLLECTION_DATA_TEMPLATE,
            'collection_browse.lblx': COLLECTION_BROWSE_TEMPLATE,
        },
    )
    touch_label(env.bundle_dir / 'data', 'shard0/1234567890w')
    monkeypatch.setattr(collections_module, 'write_label', _record)
    _run_collections(env)
    assert isinstance(handed_over[label], FCPath)
    assert handed_over[label] == FCPath(env.bundle_dir) / subdir / label


@pytest.mark.parametrize(
    ('broken', 'intact'),
    [
        ('collection_data.lblx', 'collection_browse.lblx'),
        ('collection_browse.lblx', 'collection_data.lblx'),
    ],
    ids=['data collection', 'browse collection'],
)
def test_a_broken_collection_template_is_counted_and_leaves_the_other(
    tmp_path: Path, broken: str, intact: str
) -> None:
    """One unrenderable collection label is counted; the other still gets written.

    Failing on the first would hide the second, so a run reports every broken
    collection template rather than one per run.  Each label is a case of its
    own because each is counted by a statement of its own.

    Parameters:
        tmp_path: Base temporary directory.
        broken: Template whose render errors in this case.
        intact: Template that still renders in this case.
    """
    broken_dir, _ = COLLECTION_LABELS[broken]
    intact_dir, intact_body = COLLECTION_LABELS[intact]
    env = make_bundle_env(
        tmp_path, template_contents={broken: BROKEN_TEMPLATE, intact: intact_body}
    )
    touch_label(env.bundle_dir / 'data', 'shard0/1234567890w')
    failed = _run_collections(env)
    assert failed == 1
    assert not (env.bundle_dir / broken_dir / broken).exists()
    assert (env.bundle_dir / intact_dir / intact).is_file()


def test_a_missing_collection_template_raises(tmp_path: Path) -> None:
    """A collection template the dataset declares and does not have ends the run.

    The driver checks every declared template before it processes anything, so
    one that is missing this far in is a template tree that does not carry what
    its dataset says it does.  Passing over it would leave the bundle with an
    inventory no label describes, and nothing saying so.
    """
    env = make_bundle_env(tmp_path)
    (Path(env.dataset.pds4_bundle_template_dir()) / 'collection_data.lblx').unlink()
    touch_label(env.bundle_dir / 'data', 'shard0/1234567890w')
    with pytest.raises(FileNotFoundError, match=r'collection_data\.lblx'):
        _run_collections(env)


# ---------------------------------------------------------------------------
# generate_global_index_files: tables
# ---------------------------------------------------------------------------


def test_bodies_index_header_from_configured_backplane_types(tmp_path: Path) -> None:
    """The bodies index header lists min/max columns per configured body backplane."""
    env = _index_env(tmp_path)
    write_supplemental(env.bundle_dir / 'data', 'shard0/1234567890w', bodies=BODY_STATS)
    _run_global_index(env)
    rows = read_tab(env.bundle_dir / 'document' / 'supplemental' / 'global_index_bodies.tab')
    assert rows[0] == [
        'LID',
        'body_name',
        'path_to_image_file',
        'latitude_min',
        'latitude_max',
        'resolution_min',
        'resolution_max',
    ]


def test_bodies_index_one_row_per_image_body(tmp_path: Path) -> None:
    """The bodies index has one row per (image, body) pair."""
    env = _index_env(tmp_path)
    two_bodies: dict[str, Any] = {
        'MIMAS': {'backplanes': {'latitude': {'min': 1.0, 'max': 2.0, 'units': 'deg'}}},
        'ENCELADUS': {'backplanes': {'latitude': {'min': 3.0, 'max': 4.0, 'units': 'deg'}}},
    }
    write_supplemental(env.bundle_dir / 'data', 'shard0/1111111111n', bodies=two_bodies)
    write_supplemental(env.bundle_dir / 'data', 'shard0/2222222222w', bodies=BODY_STATS)
    _run_global_index(env)
    rows = read_tab(env.bundle_dir / 'document' / 'supplemental' / 'global_index_bodies.tab')
    assert len(rows) == 4
    body_names = [row[1] for row in rows[1:]]
    assert body_names == ['MIMAS', 'ENCELADUS', 'MIMAS']


def test_a_degrees_column_is_written_to_three_decimals(tmp_path: Path) -> None:
    """A statistic in degrees is written to a thousandth of a degree.

    One pixel is three ten-thousandths of a degree on the sky for the
    narrow-angle camera and ten times that for the wide-angle one, so the third
    decimal is the last one a pixel resolves.  The plane is declared in radians
    and the column is in degrees, so the format is found by the unit the
    statistic is in rather than the one the plane was declared in.
    """
    env = _index_env(tmp_path)
    write_supplemental(env.bundle_dir / 'data', 'shard0/1234567890w', bodies=BODY_STATS)
    _run_global_index(env)
    rows = read_tab(env.bundle_dir / 'document' / 'supplemental' / 'global_index_bodies.tab')
    assert rows[1][3] == '1.235'
    assert rows[1][4] == '2.000'


def test_a_kilometers_column_is_written_to_one_decimal(tmp_path: Path) -> None:
    """A ring radius in kilometers is written to a tenth of a kilometer.

    The radii run from 7e4 to 5e5 km, where a float32 plane's spacing is
    hundredths of a kilometer, so a second decimal would print noise.
    """
    env = _index_env(tmp_path)
    radii = {'backplanes': {'radius': {'min': 74500.04, 'max': 136800.96, 'units': 'km'}}}
    write_supplemental(env.bundle_dir / 'data', 'shard0/1234567890w', rings=radii)
    _run_global_index(env)
    rows = read_tab(env.bundle_dir / 'document' / 'supplemental' / 'global_index_rings.tab')
    assert rows[1][2] == '74500.0'
    assert rows[1][3] == '136801.0'


def test_a_degrees_per_pixel_column_keeps_a_value_far_smaller_than_one(tmp_path: Path) -> None:
    """A longitudinal resolution of order a ten-thousandth of a degree per pixel survives.

    This is the column the eight-decimal width exists for: every value in it is
    below one, so eight decimals stays within the seven significant digits a
    float32 plane carries, where a narrower fixed-point format rounds a value
    this small to one significant figure or to zero and the table then reports
    a measurement it did not make.  Both the minimum and the maximum are
    checked, since a format applied to one and not the other is the way a table
    half-rounds.
    """
    env = _index_env(tmp_path, rings=[{'name': 'longitudinal_resolution', 'units': 'rad/pixel'}])
    fine = {
        'backplanes': {
            'longitudinal_resolution': {'min': 0.00015470, 'max': 0.00080214, 'units': 'deg/pixel'}
        }
    }
    write_supplemental(env.bundle_dir / 'data', 'shard0/1234567890w', rings=fine)
    _run_global_index(env)
    rows = read_tab(env.bundle_dir / 'document' / 'supplemental' / 'global_index_rings.tab')
    assert rows[1][2] == '0.00015470'
    assert rows[1][3] == '0.00080214'


def test_a_kilometers_per_pixel_column_keeps_five_figures_without_an_exponent(
    tmp_path: Path,
) -> None:
    """A resolution in kilometers per pixel keeps five figures at either end of its range.

    The column runs from 6e-4 km per pixel a hundred kilometers off Enceladus
    to 7e4 at the grazing limb of a wide-angle frame, eight orders of magnitude
    that no fixed decimal count fits: eight decimals would print the large end
    to twelve digits of noise, and a width fit to the large end would print the
    small end as zero.  Five significant figures write both, with trailing
    zeros kept so that every value shows the same number of them.  They are
    written positionally, so a value of five or six integer digits is written
    as the integer it is rather than with a trailing point or an exponent,
    which are what a general format writes there and what a person reading
    the table would have to decode.  A zero has no magnitude to count figures
    from, and is written with the four decimals a value of one gets.
    """
    env = _index_env(tmp_path)
    resolutions: dict[str, Any] = {
        'MIMAS': {
            'backplanes': {'resolution': {'min': 0.0006, 'max': 4200.0, 'units': 'km/pixel'}}
        },
        'SATURN': {
            'backplanes': {'resolution': {'min': 70853.2, 'max': 123456.0, 'units': 'km/pixel'}}
        },
        'PAN': {'backplanes': {'resolution': {'min': 0.0, 'max': 1.0, 'units': 'km/pixel'}}},
    }
    write_supplemental(env.bundle_dir / 'data', 'shard0/1234567890w', bodies=resolutions)
    _run_global_index(env)
    rows = read_tab(env.bundle_dir / 'document' / 'supplemental' / 'global_index_bodies.tab')
    assert rows[1][5] == '0.00060000'
    assert rows[1][6] == '4200.0'
    assert rows[2][5] == '70853'
    assert rows[2][6] == '123456'
    assert rows[3][5] == '0.0000'


@pytest.mark.parametrize(
    ('decimals', 'significant'), [(3, 5), (None, None)], ids=['both set', 'neither set']
)
def test_an_index_value_format_sets_exactly_one_of_its_fields(
    decimals: int | None, significant: int | None
) -> None:
    """A format is a number of decimals or of significant figures, never both or neither.

    Parameters:
        decimals: The decimals field for this case.
        significant: The significant-figures field for this case.
    """
    with pytest.raises(ValueError, match='exactly one of decimals and significant'):
        IndexValueFormat(decimals=decimals, significant=significant)


@pytest.mark.parametrize(
    'value_format',
    [IndexValueFormat(decimals=3), IndexValueFormat(significant=5)],
    ids=['decimals', 'significant figures'],
)
def test_an_index_value_format_refuses_a_value_that_is_not_a_finite_number(
    value_format: IndexValueFormat,
) -> None:
    """NaN has no decimal form, so it is refused by name rather than written as ``nan``.

    Each way of writing a value is a case of its own, since each reaches the
    value by a path of its own.

    Parameters:
        value_format: The format asked to write NaN.
    """
    with pytest.raises(ValueError, match='got nan'):
        value_format.render(math.nan)


def test_a_plane_in_a_unit_the_index_cannot_size_is_refused_before_any_table(
    tmp_path: Path,
) -> None:
    """A configured unit with no column format fails the run with nothing written.

    The formats are looked up for every configured plane before a supplemental
    file is read, so the refusal names the unit and leaves no half-written
    table behind it.  The unit is on a ring plane and the supplemental file
    holds a body row, so a lookup deferred until the rings table is written
    would leave the bodies table on disk.
    """
    env = _index_env(tmp_path, rings=[{'name': 'tilt', 'units': 'mrad'}])
    write_supplemental(env.bundle_dir / 'data', 'shard0/1234567890w', bodies=BODY_STATS)
    with pytest.raises(ValueError, match="'mrad'"):
        _run_global_index(env)
    assert not (env.bundle_dir / 'document').exists()


def _ring_resolution_env(tmp_path: Path) -> BundleEnv:
    """Build an environment declaring, beside the default bodies, one ring plane in rad/pixel.

    Parameters:
        tmp_path: Base temporary directory.

    Returns:
        The environment, whose ring statistic is expected in degrees per pixel.
    """
    return _index_env(tmp_path, rings=[{'name': 'longitudinal_resolution', 'units': 'rad/pixel'}])


def _ring_resolution_stats(units: str | None) -> dict[str, Any]:
    """Build ring statistics holding one longitudinal resolution in the given unit.

    Parameters:
        units: The unit the statistic records; None records no unit at all.

    Returns:
        The ``backplanes.rings`` payload of a supplemental file.
    """
    statistic: dict[str, Any] = {'min': 1.4e-05, 'max': 3.9e-05}
    if units is not None:
        statistic['units'] = units
    return {'backplanes': {'longitudinal_resolution': statistic}}


def test_a_supplemental_file_in_another_unit_is_refused_with_nothing_written(
    tmp_path: Path,
) -> None:
    """A supplemental file recording a plane in a unit its configuration does not give ends the run.

    The labels pass holds every document to its configured unit, but a bundle
    tree can hold supplemental files a labels pass wrote before it did, and
    indexing one would put a column in two units.  The error names the file,
    the plane and both units and says what to regenerate, and no table is
    written.
    """
    env = _ring_resolution_env(tmp_path)
    write_supplemental(
        env.bundle_dir / 'data', 'shard0/1234567890w', rings=_ring_resolution_stats('rad/pixel')
    )
    with pytest.raises(ValueError) as excinfo:
        _run_global_index(env)
    message = str(excinfo.value)
    assert '1234567890w_supplemental.txt' in message
    assert 'the longitudinal_resolution statistic' in message
    assert 'in rad/pixel' in message
    assert 'expects deg/pixel' in message
    assert 'regenerate the backplanes, then the bundle into an empty directory' in message
    assert not (env.bundle_dir / 'document').exists()


def test_a_supplemental_file_recording_no_unit_is_refused_with_nothing_written(
    tmp_path: Path,
) -> None:
    """A statistic with no units key predates the unit being recorded, and ends the run too."""
    env = _ring_resolution_env(tmp_path)
    write_supplemental(
        env.bundle_dir / 'data', 'shard0/1234567890w', rings=_ring_resolution_stats(None)
    )
    with pytest.raises(ValueError) as excinfo:
        _run_global_index(env)
    message = str(excinfo.value)
    assert '1234567890w_supplemental.txt' in message
    assert 'the longitudinal_resolution statistic' in message
    assert 'in no unit at all' in message
    assert 'expects deg/pixel' in message
    assert not (env.bundle_dir / 'document').exists()


def test_a_supplemental_file_with_a_body_statistic_in_another_unit_is_refused(
    tmp_path: Path,
) -> None:
    """A body plane is held to its unit in the summary pass, as a ring plane is.

    The bodies are read from a member of their own, one entry per body, so a
    check that read only the rings would index this file.
    """
    env = _index_env(tmp_path)
    radians = {'MIMAS': {'backplanes': {'latitude': {'min': -1.2, 'max': 1.4, 'units': 'rad'}}}}
    write_supplemental(env.bundle_dir / 'data', 'shard0/1234567890w', bodies=radians)
    with pytest.raises(ValueError) as excinfo:
        _run_global_index(env)
    assert 'the latitude statistic in rad where the configuration expects deg' in str(excinfo.value)


def test_a_disagreeing_supplemental_file_anywhere_is_refused_before_the_first_table(
    tmp_path: Path,
) -> None:
    """A disagreement in any supplemental file stops the run before the bodies table exists.

    Every file is read and its rows accumulated before either table is
    written, so a file that disagrees after others that agree still leaves
    nothing behind: the first file here holds a body row the bodies table
    would carry, and the second is the one refused.
    """
    env = _ring_resolution_env(tmp_path)
    write_supplemental(env.bundle_dir / 'data', 'shard0/1111111111n', bodies=BODY_STATS)
    write_supplemental(
        env.bundle_dir / 'data', 'shard0/2222222222w', rings=_ring_resolution_stats('rad/pixel')
    )
    with pytest.raises(ValueError, match='2222222222w_supplemental'):
        _run_global_index(env)
    assert not (env.bundle_dir / 'document').exists()


@pytest.mark.parametrize(
    ('maximum', 'recorded'),
    [
        (math.inf, 'records a resolution maximum of inf'),
        (10**400, 'records a resolution maximum of an integer 401 digits long'),
    ],
    ids=['infinity', 'integer too large for a float'],
)
def test_a_supplemental_file_holding_a_maximum_no_column_can_is_refused_with_nothing_written(
    tmp_path: Path, maximum: float, recorded: str
) -> None:
    """A maximum no column can hold ends the run, naming file and plane, with no table.

    The JSON reader returns an infinity for the token the writer writes for one,
    and an integer for an integer literal of any length, so a supplemental file
    can carry either.  Every index format writes through a float, which holds
    neither.

    Parameters:
        tmp_path: Base temporary directory.
        maximum: The resolution maximum the supplemental file records.
        recorded: What the refusal says the file records.
    """
    env = _index_env(tmp_path)
    unholdable = {
        'SATURN': {'backplanes': {'resolution': {'min': 60.0, 'max': maximum, 'units': 'km/pixel'}}}
    }
    write_supplemental(env.bundle_dir / 'data', 'shard0/1234567890w', bodies=unholdable)
    with pytest.raises(ValueError) as excinfo:
        _run_global_index(env)
    message = str(excinfo.value)
    assert '1234567890w_supplemental.txt' in message
    assert recorded in message
    assert not (env.bundle_dir / 'document').exists()


def test_a_render_that_fails_leaves_no_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every cell is rendered before either table opens, so a failed render leaves none.

    The checks refuse every value they know no column can hold, but they are not
    the only way a render can fail, and a table opened before its cells exist is
    left half-written by whichever failure comes.
    """

    def _refuse(self: IndexValueFormat, value: float) -> str:
        """Fail every render, as a value no check anticipated would.

        Parameters:
            self: The format asked to render, unused.
            value: The value asked for, named in the message.

        Raises:
            ValueError: Always.
        """
        raise ValueError(f'cannot render {value!r}')

    env = _index_env(tmp_path)
    write_supplemental(env.bundle_dir / 'data', 'shard0/1234567890w', bodies=BODY_STATS)
    monkeypatch.setattr(IndexValueFormat, 'render', _refuse)
    with pytest.raises(ValueError, match='cannot render'):
        _run_global_index(env)
    assert not (env.bundle_dir / 'document').exists()


def test_a_refused_run_leaves_none_of_the_index_products_an_earlier_run_wrote(
    tmp_path: Path,
) -> None:
    """A refused run leaves none of the index tables and labels an earlier run wrote.

    Left in place, they would sit beside the collection files the run has
    rewritten, still indexing the refused file as it was.
    """
    products = [
        'global_index_bodies.tab',
        'global_index_bodies.lblx',
        'global_index_rings.tab',
        'global_index_rings.lblx',
    ]
    env = _ring_resolution_env(tmp_path)
    supplemental_dir = env.bundle_dir / 'document' / 'supplemental'
    supplemental_dir.mkdir(parents=True)
    for product in products:
        (supplemental_dir / product).write_text('an earlier run\n', encoding='utf-8')
    write_supplemental(
        env.bundle_dir / 'data', 'shard0/1234567890w', rings=_ring_resolution_stats('rad/pixel')
    )
    with pytest.raises(ValueError, match='1234567890w_supplemental'):
        _run_global_index(env)
    assert [product for product in products if (supplemental_dir / product).exists()] == []


def test_bodies_index_missing_backplane_values_blank(tmp_path: Path) -> None:
    """Backplane types absent from a body's stats produce empty columns."""
    env = _index_env(tmp_path)
    write_supplemental(env.bundle_dir / 'data', 'shard0/1234567890w', bodies=BODY_STATS)
    _run_global_index(env)
    rows = read_tab(env.bundle_dir / 'document' / 'supplemental' / 'global_index_bodies.tab')
    assert rows[1][5] == ''
    assert rows[1][6] == ''


def test_index_path_to_image_file_is_data_relative(tmp_path: Path) -> None:
    """The path_to_image_file column points at data/<stub>_backplanes.lblx."""
    env = _index_env(tmp_path)
    write_supplemental(
        env.bundle_dir / 'data', 'shard0/1234567890w', bodies=BODY_STATS, rings=RING_STATS
    )
    _run_global_index(env)
    bodies_rows = read_tab(env.bundle_dir / 'document' / 'supplemental' / 'global_index_bodies.tab')
    assert bodies_rows[1][2] == 'data/shard0/1234567890w_backplanes.lblx'
    rings_rows = read_tab(env.bundle_dir / 'document' / 'supplemental' / 'global_index_rings.tab')
    assert rings_rows[1][1] == 'data/shard0/1234567890w_backplanes.lblx'


def test_rings_index_row_only_for_images_with_ring_backplanes(tmp_path: Path) -> None:
    """Images without ring backplanes are omitted from the rings index."""
    env = _index_env(tmp_path)
    write_supplemental(env.bundle_dir / 'data', 'shard0/1111111111n', bodies=BODY_STATS)
    write_supplemental(env.bundle_dir / 'data', 'shard0/2222222222w', rings=RING_STATS)
    _run_global_index(env)
    rows = read_tab(env.bundle_dir / 'document' / 'supplemental' / 'global_index_rings.tab')
    assert rows[0] == ['LID', 'path_to_image_file', 'radius_min', 'radius_max']
    assert len(rows) == 2
    assert rows[1][1] == 'data/shard0/2222222222w_backplanes.lblx'
    assert rows[1][2] == '74500.0'
    assert rows[1][3] == '136801.0'


def test_no_supplemental_files_writes_header_only_indexes(tmp_path: Path) -> None:
    """With no supplemental files (even no data dir), header-only tables are written."""
    env = _index_env(tmp_path)
    _run_global_index(env)
    bodies_rows = read_tab(env.bundle_dir / 'document' / 'supplemental' / 'global_index_bodies.tab')
    assert len(bodies_rows) == 1
    rings_rows = read_tab(env.bundle_dir / 'document' / 'supplemental' / 'global_index_rings.tab')
    assert len(rings_rows) == 1


def test_unreadable_supplemental_skipped_with_logged_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A malformed supplemental file is skipped; other images are still indexed.

    The log gives the parser's reason, which the frames of a traceback do not
    carry.
    """
    env = _index_env(tmp_path)
    write_supplemental(env.bundle_dir / 'data', 'shard0/1111111111n', raw_text='not json')
    write_supplemental(env.bundle_dir / 'data', 'shard0/2222222222w', bodies=BODY_STATS)
    _run_global_index(env)
    rows = read_tab(env.bundle_dir / 'document' / 'supplemental' / 'global_index_bodies.tab')
    assert len(rows) == 2
    assert '2222222222w' in rows[1][0]
    out = capsys.readouterr().out
    assert 'Error reading supplemental file' in out
    assert 'Expecting value: line 1 column 1 (char 0)' in out


def test_global_index_labels_rendered_with_file_records(tmp_path: Path) -> None:
    """Global index labels render with FILE_RECORDS set to the row counts."""
    env = _index_env(tmp_path)
    write_templates(
        Path(env.dataset.pds4_bundle_template_dir()),
        {
            'global_index_bodies.lblx': GLOBAL_INDEX_TEMPLATE,
            'global_index_rings.lblx': GLOBAL_INDEX_TEMPLATE,
        },
    )
    two_bodies: dict[str, Any] = {
        'MIMAS': {'backplanes': {'latitude': {'min': 1.0, 'max': 2.0, 'units': 'deg'}}},
        'ENCELADUS': {'backplanes': {'latitude': {'min': 3.0, 'max': 4.0, 'units': 'deg'}}},
    }
    write_supplemental(
        env.bundle_dir / 'data', 'shard0/1234567890w', bodies=two_bodies, rings=RING_STATS
    )
    failed = _run_global_index(env)
    assert failed == 0
    supplemental_dir = env.bundle_dir / 'document' / 'supplemental'
    bodies_text = (supplemental_dir / 'global_index_bodies.lblx').read_text(encoding='utf-8')
    assert '<records>2</records>' in bodies_text
    rings_text = (supplemental_dir / 'global_index_rings.lblx').read_text(encoding='utf-8')
    assert '<records>1</records>' in rings_text


@pytest.mark.parametrize(
    ('broken', 'intact'),
    [
        ('global_index_bodies.lblx', 'global_index_rings.lblx'),
        ('global_index_rings.lblx', 'global_index_bodies.lblx'),
    ],
    ids=['bodies index', 'rings index'],
)
def test_a_broken_index_template_is_counted_and_leaves_the_other(
    tmp_path: Path, broken: str, intact: str
) -> None:
    """One unrenderable index label is counted; the other still gets written.

    Parameters:
        tmp_path: Base temporary directory.
        broken: Template whose render errors in this case.
        intact: Template that still renders in this case.
    """
    env = _index_env(tmp_path)
    write_templates(
        Path(env.dataset.pds4_bundle_template_dir()),
        {broken: BROKEN_TEMPLATE, intact: GLOBAL_INDEX_TEMPLATE},
    )
    write_supplemental(
        env.bundle_dir / 'data', 'shard0/1234567890w', bodies=BODY_STATS, rings=RING_STATS
    )
    failed = _run_global_index(env)
    assert failed == 1
    supplemental_dir = env.bundle_dir / 'document' / 'supplemental'
    assert not (supplemental_dir / broken).exists()
    assert (supplemental_dir / intact).is_file()


def test_a_missing_index_template_raises(tmp_path: Path) -> None:
    """An index template the dataset declares and does not have ends the run.

    As with the collection labels: the template tree does not carry what its
    dataset says it does, and an index table with no label describing it is
    worse than a run that stops.
    """
    env = _index_env(tmp_path)
    (Path(env.dataset.pds4_bundle_template_dir()) / 'global_index_bodies.lblx').unlink()
    write_supplemental(env.bundle_dir / 'data', 'shard0/1234567890w', bodies=BODY_STATS)
    with pytest.raises(FileNotFoundError, match=r'global_index_bodies\.lblx'):
        _run_global_index(env)


# ---------------------------------------------------------------------------
# LID cross-referencing (known bug #139 and LID round trips)
# ---------------------------------------------------------------------------


def _cross_reference_env(tmp_path: Path) -> BundleEnv:
    """Build a one-image bundle with a label and a supplemental file in place.

    Parameters:
        tmp_path: Base temporary directory.

    Returns:
        The environment, ready for both phase-2 generators.
    """
    env = _index_env(tmp_path)
    touch_label(env.bundle_dir / 'data', 'shard0/1234567890w')
    write_supplemental(
        env.bundle_dir / 'data', 'shard0/1234567890w', bodies=BODY_STATS, rings=RING_STATS
    )
    return env


def test_global_index_bodies_lid_matches_collection_inventory(tmp_path: Path) -> None:
    """#139 round trip: the bodies-index LID equals the collection inventory LID."""
    env = _cross_reference_env(tmp_path)
    _run_collections(env)
    _run_global_index(env)
    inventory_rows = read_tab(env.bundle_dir / 'data' / 'collection_data.tab')
    inventory_lid = inventory_rows[1][1].split('::')[0]
    bodies_rows = read_tab(env.bundle_dir / 'document' / 'supplemental' / 'global_index_bodies.tab')
    assert bodies_rows[1][0] == inventory_lid


def test_global_index_rings_lid_matches_collection_inventory(tmp_path: Path) -> None:
    """#139 round trip: the rings-index LID equals the collection inventory LID."""
    env = _cross_reference_env(tmp_path)
    _run_collections(env)
    _run_global_index(env)
    inventory_rows = read_tab(env.bundle_dir / 'data' / 'collection_data.tab')
    inventory_lid = inventory_rows[1][1].split('::')[0]
    rings_rows = read_tab(env.bundle_dir / 'document' / 'supplemental' / 'global_index_rings.tab')
    assert rings_rows[1][0] == inventory_lid


def test_inventory_lidvid_round_trips_with_canonical_builders(tmp_path: Path) -> None:
    """With canonical LID builders, the inventory LIDVID is the data LID plus ::1.0."""
    env = _cross_reference_env(tmp_path)
    _run_collections(env)
    rows = read_tab(env.bundle_dir / 'data' / 'collection_data.tab')
    expected = env.dataset.pds4_image_name_to_data_lid('1234567890w') + '::1.0'
    assert rows[1][1] == expected


def test_cassini_inventory_lidvid_matches_label_lid(tmp_path: Path) -> None:
    """The Cassini collection inventory LIDVID matches the label's DATA_LID."""
    dataset = DataSetPDS3CassiniISSSaturn(tmp_path / 'holdings')
    bundle_results_root = tmp_path / 'bundle'
    bundle_dir = bundle_results_root / dataset.pds4_bundle_name()
    touch_label(bundle_dir / 'data', '1454xxxxxx/145472xxxx/1454725799n')
    generate_collection_files(FCPath(bundle_results_root), dataset, MAIN_LOGGER)
    rows = read_tab(bundle_dir / 'data' / 'collection_data.tab')
    inventory_lid = rows[1][1].split('::')[0]
    label_lid = dataset.pds4_image_name_to_data_lid('N1454725799')
    assert inventory_lid == label_lid
