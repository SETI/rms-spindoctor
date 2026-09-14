"""Spec-first tests for the PDS4 global index files and their LIDs (phase 2).

Contract under test (docs/user_guide/user_guide_pds4_bundle.rst "Summary Pass" /
"Summary Pass Outputs" and docs/dev_guide/dev_guide_pds4.rst "Pipeline overview"):
``generate_global_index_files`` scans ``data/`` for ``*_supplemental.txt`` files
and writes ``miscellaneous/global_bodies_index.tab`` (one row per
image/body) and ``global_rings_index.tab`` (one row per image with ring
backplanes), with min/max columns for each configured backplane type, each
written in the format its unit calls for, plus their labels.  Every template a
generator renders is required: the drivers check the ones their dataset declares
before processing anything, so one that is missing raises here rather than being
passed over.

Both generators recover the original image name from each on-disk product stem
via ``DataSet.pds4_lid_part_to_image_name`` before building LIDs, so the stem
(which is already in LID-part form) is not fed back through the image-name
transform.  This is what keeps the inventory LIDVIDs and global-index LIDs
matching the product labels' DATA_LID.
"""

import math
from pathlib import Path
from typing import Any

import pytest
from filecache import FCPath

from spindoctor.cli.pds4.global_index import IndexValueFormat, generate_global_index_files
from spindoctor.config import MAIN_LOGGER

from .conftest import (
    BROKEN_TEMPLATE,
    GLOBAL_INDEX_TEMPLATE,
    BundleEnv,
    make_bundle_env,
    read_csv_rows,
    read_index_rows,
    run_collections,
    touch_label,
    write_supplemental,
    write_templates,
)

BODY_STATS = {
    'MOON_A': {'backplanes': {'latitude': {'min': 1.234567891, 'max': 2, 'units': 'deg'}}}
}
"""A body's statistics, in the unit the default configuration's latitude plane takes."""
RING_STATS = {'backplanes': {'radius': {'min': 81000.0, 'max': 125000.987654, 'units': 'km'}}}
"""Ring statistics, in the unit the default configuration's radius plane takes."""


def _run_global_index(env: BundleEnv) -> int:
    """Run generate_global_index_files against the environment's bundle root.

    Parameters:
        env: The hermetic bundle environment to process.

    Returns:
        The number of index labels that could not be rendered.
    """
    return generate_global_index_files(
        FCPath(env.bundle_results_root), env.dataset.as_dataset(), MAIN_LOGGER
    ).failed_labels


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


def _write_image(
    data_dir: Path,
    stub: str,
    *,
    bodies: dict[str, Any] | None = None,
    rings: dict[str, Any] | None = None,
) -> Path:
    """Write what the labels pass leaves for one image the index is to have rows for.

    The tables index the images the data inventory lists, which are the data labels in
    the data tree, so the image gets a data label beside its supplemental file.

    Parameters:
        data_dir: The bundle's ``data`` directory.
        stub: The image's path stub.
        bodies: ``backplanes.bodies`` payload keyed by body name.
        rings: ``backplanes.rings`` payload.

    Returns:
        The path of the supplemental file.
    """
    touch_label(data_dir, stub)
    return write_supplemental(data_dir, stub, bodies=bodies, rings=rings)


# ---------------------------------------------------------------------------
# generate_global_index_files: tables
# ---------------------------------------------------------------------------


def test_bodies_index_header_from_configured_backplane_types(tmp_path: Path) -> None:
    """The bodies index header lists min/max columns per configured body backplane."""
    env = _index_env(tmp_path)
    _write_image(env.bundle_dir / 'data', 'shard0/1234567890w', bodies=BODY_STATS)
    _run_global_index(env)
    rows = read_index_rows(env.bundle_dir / 'miscellaneous' / 'global_bodies_index.tab')
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
        'MOON_A': {'backplanes': {'latitude': {'min': 1.0, 'max': 2.0, 'units': 'deg'}}},
        'MOON_B': {'backplanes': {'latitude': {'min': 3.0, 'max': 4.0, 'units': 'deg'}}},
    }
    _write_image(env.bundle_dir / 'data', 'shard0/1111111111n', bodies=two_bodies)
    _write_image(env.bundle_dir / 'data', 'shard0/2222222222w', bodies=BODY_STATS)
    _run_global_index(env)
    rows = read_index_rows(env.bundle_dir / 'miscellaneous' / 'global_bodies_index.tab')
    assert len(rows) == 4
    body_names = [row[1] for row in rows[1:]]
    assert body_names == ['MOON_A', 'MOON_B', 'MOON_A']


def test_each_field_is_as_long_as_the_longest_value_in_its_column(tmp_path: Path) -> None:
    """The table is fixed width, each field padded to the longest value in its column.

    Values under one format differ in length -- ``1.000`` and ``-12.500`` are both three
    decimals -- so a field is as long as the longest value its column holds rather than
    a length its format implies.  A statistic is right-justified in its field and text
    left-justified, a comma separates the fields, and every line ends in a line feed.
    The header line names the columns, unpadded.
    """
    env = _index_env(tmp_path, bodies=[{'name': 'latitude', 'units': 'rad'}], rings=[])
    two_bodies: dict[str, Any] = {
        'A': {'backplanes': {'latitude': {'min': 1.0, 'max': 2.0, 'units': 'deg'}}},
        'MOON_B': {'backplanes': {'latitude': {'min': -12.5, 'max': 45.25, 'units': 'deg'}}},
    }
    _write_image(env.bundle_dir / 'data', 'shard0/1111111111n', bodies=two_bodies)
    _run_global_index(env)
    table = env.bundle_dir / 'miscellaneous' / 'global_bodies_index.tab'
    lid = 'urn:nasa:pds:fake_bundle:data:1111111111n'
    path = 'data/shard0/1111111111n_backplanes.lblx'
    assert table.read_bytes().decode('ascii').splitlines(keepends=True) == [
        'LID,body_name,path_to_image_file,latitude_min,latitude_max\n',
        f'{lid},A     ,{path},  1.000, 2.000\n',
        f'{lid},MOON_B,{path},-12.500,45.250\n',
    ]


def test_a_degrees_column_is_written_to_three_decimals(tmp_path: Path) -> None:
    """A statistic in degrees is written to a thousandth of a degree.

    The third decimal is about the last one a pixel resolves.  The plane is
    declared in radians and the column is in degrees, so the format is found by
    the unit the statistic is in rather than the one the plane was declared in.
    """
    env = _index_env(tmp_path)
    _write_image(env.bundle_dir / 'data', 'shard0/1234567890w', bodies=BODY_STATS)
    _run_global_index(env)
    rows = read_index_rows(env.bundle_dir / 'miscellaneous' / 'global_bodies_index.tab')
    assert rows[1][3] == '1.235'
    assert rows[1][4] == '2.000'


def test_a_kilometers_column_is_written_to_one_decimal(tmp_path: Path) -> None:
    """A ring radius in kilometers is written to a tenth of a kilometer.

    At radii of order a hundred thousand kilometers a float32 plane's spacing is
    hundredths of a kilometer, so a second decimal would print noise.
    """
    env = _index_env(tmp_path)
    radii = {'backplanes': {'radius': {'min': 81000.04, 'max': 125000.96, 'units': 'km'}}}
    _write_image(env.bundle_dir / 'data', 'shard0/1234567890w', rings=radii)
    _run_global_index(env)
    rows = read_index_rows(env.bundle_dir / 'miscellaneous' / 'global_rings_index.tab')
    assert rows[1][2] == '81000.0'
    assert rows[1][3] == '125001.0'


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
    _write_image(env.bundle_dir / 'data', 'shard0/1234567890w', rings=fine)
    _run_global_index(env)
    rows = read_index_rows(env.bundle_dir / 'miscellaneous' / 'global_rings_index.tab')
    assert rows[1][2] == '0.00015470'
    assert rows[1][3] == '0.00080214'


def test_a_kilometers_per_pixel_column_keeps_five_figures_without_an_exponent(
    tmp_path: Path,
) -> None:
    """A resolution in kilometers per pixel keeps five figures at either end of its range.

    A column can run from under a meter per pixel close to a small body to tens
    of thousands of kilometers at a grazing limb, eight orders of magnitude that
    no fixed decimal count fits: eight decimals would print the large end
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
        'MOON_A': {
            'backplanes': {'resolution': {'min': 0.0006, 'max': 4200.0, 'units': 'km/pixel'}}
        },
        'PLANET': {
            'backplanes': {'resolution': {'min': 70853.2, 'max': 123456.0, 'units': 'km/pixel'}}
        },
        'MOON_C': {'backplanes': {'resolution': {'min': 0.0, 'max': 1.0, 'units': 'km/pixel'}}},
    }
    _write_image(env.bundle_dir / 'data', 'shard0/1234567890w', bodies=resolutions)
    _run_global_index(env)
    rows = read_index_rows(env.bundle_dir / 'miscellaneous' / 'global_bodies_index.tab')
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


def _ring_resolution_env(tmp_path: Path) -> BundleEnv:
    """Build an environment declaring, beside the default bodies, one ring plane in rad/pixel.

    Parameters:
        tmp_path: Base temporary directory.

    Returns:
        The environment, whose ring statistic is expected in degrees per pixel.
    """
    return _index_env(tmp_path, rings=[{'name': 'longitudinal_resolution', 'units': 'rad/pixel'}])


def _ring_resolution_stats(units: str) -> dict[str, Any]:
    """Build ring statistics holding one longitudinal resolution in the given unit.

    Parameters:
        units: The unit the statistic records.

    Returns:
        The ``backplanes.rings`` payload of a supplemental file.
    """
    statistic: dict[str, Any] = {'min': 1.4e-05, 'max': 3.9e-05, 'units': units}
    return {'backplanes': {'longitudinal_resolution': statistic}}


def test_a_supplemental_file_in_another_unit_is_refused_with_nothing_written(
    tmp_path: Path,
) -> None:
    """A supplemental file recording a plane in a unit its configuration does not give ends the run.

    Indexing it would put a column in two units.  The error names the file, the
    plane and both units and says what to regenerate, and no table is written.
    """
    env = _ring_resolution_env(tmp_path)
    _write_image(
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
    assert not (env.bundle_dir / 'miscellaneous').exists()


def test_a_supplemental_file_with_a_body_statistic_in_another_unit_is_refused(
    tmp_path: Path,
) -> None:
    """A body plane is held to its unit in the summary pass, as a ring plane is.

    The bodies are read from a member of their own, one entry per body, so a
    check that read only the rings would index this file.
    """
    env = _index_env(tmp_path)
    radians = {'MOON_A': {'backplanes': {'latitude': {'min': -1.2, 'max': 1.4, 'units': 'rad'}}}}
    _write_image(env.bundle_dir / 'data', 'shard0/1234567890w', bodies=radians)
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
    _write_image(env.bundle_dir / 'data', 'shard0/1111111111n', bodies=BODY_STATS)
    _write_image(
        env.bundle_dir / 'data', 'shard0/2222222222w', rings=_ring_resolution_stats('rad/pixel')
    )
    with pytest.raises(ValueError, match='2222222222w_supplemental'):
        _run_global_index(env)
    assert not (env.bundle_dir / 'miscellaneous').exists()


def test_a_supplemental_file_holding_an_infinite_maximum_is_refused_with_nothing_written(
    tmp_path: Path,
) -> None:
    """An infinite maximum ends the run, naming the file and the plane, with no table.

    The JSON reader returns an infinity for the token the writer writes for one, and
    no index column can hold it.
    """
    env = _index_env(tmp_path)
    unholdable = {
        'PLANET': {
            'backplanes': {'resolution': {'min': 60.0, 'max': math.inf, 'units': 'km/pixel'}}
        }
    }
    _write_image(env.bundle_dir / 'data', 'shard0/1234567890w', bodies=unholdable)
    with pytest.raises(ValueError) as excinfo:
        _run_global_index(env)
    message = str(excinfo.value)
    assert '1234567890w_supplemental.txt' in message
    assert 'records a resolution maximum of inf' in message
    assert not (env.bundle_dir / 'miscellaneous').exists()


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
    _write_image(env.bundle_dir / 'data', 'shard0/1234567890w', bodies=BODY_STATS)
    monkeypatch.setattr(IndexValueFormat, 'render', _refuse)
    with pytest.raises(ValueError, match='cannot render'):
        _run_global_index(env)
    assert not (env.bundle_dir / 'miscellaneous').exists()


def test_a_refused_run_leaves_none_of_the_index_products_an_earlier_run_wrote(
    tmp_path: Path,
) -> None:
    """A refused run leaves none of the index tables and labels an earlier run wrote.

    Left in place, they would sit beside the collection files the run has
    rewritten, still indexing the refused file as it was.
    """
    products = [
        'global_bodies_index.tab',
        'global_bodies_index.lblx',
        'global_rings_index.tab',
        'global_rings_index.lblx',
    ]
    env = _ring_resolution_env(tmp_path)
    index_dir = env.bundle_dir / 'miscellaneous'
    index_dir.mkdir(parents=True)
    for product in products:
        (index_dir / product).write_text('an earlier run\n', encoding='utf-8')
    _write_image(
        env.bundle_dir / 'data', 'shard0/1234567890w', rings=_ring_resolution_stats('rad/pixel')
    )
    with pytest.raises(ValueError, match='1234567890w_supplemental'):
        _run_global_index(env)
    assert [product for product in products if (index_dir / product).exists()] == []


def test_bodies_index_missing_backplane_values_blank(tmp_path: Path) -> None:
    """Backplane types absent from a body's stats produce empty columns."""
    env = _index_env(tmp_path)
    _write_image(env.bundle_dir / 'data', 'shard0/1234567890w', bodies=BODY_STATS)
    _run_global_index(env)
    rows = read_index_rows(env.bundle_dir / 'miscellaneous' / 'global_bodies_index.tab')
    assert rows[1][5] == ''
    assert rows[1][6] == ''


def test_index_path_to_image_file_is_data_relative(tmp_path: Path) -> None:
    """The path_to_image_file column points at data/<stub>_backplanes.lblx."""
    env = _index_env(tmp_path)
    _write_image(env.bundle_dir / 'data', 'shard0/1234567890w', bodies=BODY_STATS, rings=RING_STATS)
    _run_global_index(env)
    bodies_rows = read_index_rows(env.bundle_dir / 'miscellaneous' / 'global_bodies_index.tab')
    assert bodies_rows[1][2] == 'data/shard0/1234567890w_backplanes.lblx'
    rings_rows = read_index_rows(env.bundle_dir / 'miscellaneous' / 'global_rings_index.tab')
    assert rings_rows[1][1] == 'data/shard0/1234567890w_backplanes.lblx'


def test_the_rows_are_the_data_inventory_s_members(tmp_path: Path) -> None:
    """Each table indexes exactly the images the data inventory lists, and no other.

    The third image has a supplemental file whose statistics would give it a row in
    each table, and no data label, which is what a labels pass leaves when an image's
    data label fails to render.  It adds no row, so the tables and the inventory cannot
    disagree about what the bundle holds.
    """
    env = _index_env(tmp_path)
    data_dir = env.bundle_dir / 'data'
    _write_image(data_dir, 'shard0/1111111111n', bodies=BODY_STATS, rings=RING_STATS)
    _write_image(data_dir, 'shard1/2222222222w', bodies=BODY_STATS, rings=RING_STATS)
    write_supplemental(data_dir, 'shard0/3333333333n', bodies=BODY_STATS, rings=RING_STATS)
    _run_global_index(env)
    run_collections(env)
    inventory = read_csv_rows(data_dir / 'collection_data.csv')
    members = [lidvid.split('::')[0] for _, lidvid in inventory]
    tables = env.bundle_dir / 'miscellaneous'
    bodies = read_index_rows(tables / 'global_bodies_index.tab')
    rings = read_index_rows(tables / 'global_rings_index.tab')
    assert [row[0] for row in bodies[1:]] == members
    assert [row[0] for row in rings[1:]] == members


def test_rings_index_row_only_for_images_with_ring_backplanes(tmp_path: Path) -> None:
    """Images without ring backplanes are omitted from the rings index."""
    env = _index_env(tmp_path)
    _write_image(env.bundle_dir / 'data', 'shard0/1111111111n', bodies=BODY_STATS)
    _write_image(env.bundle_dir / 'data', 'shard0/2222222222w', rings=RING_STATS)
    _run_global_index(env)
    rows = read_index_rows(env.bundle_dir / 'miscellaneous' / 'global_rings_index.tab')
    assert rows[0] == ['LID', 'path_to_image_file', 'radius_min', 'radius_max']
    assert len(rows) == 2
    assert rows[1][1] == 'data/shard0/2222222222w_backplanes.lblx'
    assert rows[1][2] == '81000.0'
    assert rows[1][3] == '125001.0'


def test_an_index_table_no_image_gives_a_row_is_left_out(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A table with no row is not written, nor its label, and nothing is counted against the run.

    A table's label states its records, and the PDS4 schema requires at least one, so no
    label can describe an empty table.  A bundle holding no image with ring backplanes is
    a real state rather than a fault, so the run says so at info level and fails nothing.
    The bodies table, which has a row, is written with its label.
    """
    env = _index_env(tmp_path)
    _write_image(env.bundle_dir / 'data', 'shard0/1234567890w', bodies=BODY_STATS)
    failed = _run_global_index(env)
    assert failed == 0
    written = sorted(path.name for path in (env.bundle_dir / 'miscellaneous').iterdir())
    assert written == ['global_bodies_index.lblx', 'global_bodies_index.tab']
    reports = [
        line
        for line in capsys.readouterr().out.splitlines()
        if 'global_rings_index.tab was not written' in line
    ]
    assert len(reports) == 1
    assert '| INFO |' in reports[0]


def test_global_index_labels_rendered_with_file_records(tmp_path: Path) -> None:
    """Each index label is handed its table's row count and its LID, built from the bundle name.

    The LIDs are ``urn:nasa:pds:<bundle>:miscellaneous:global_bodies_index`` and
    ``...:global_rings_index``.
    """
    env = _index_env(tmp_path)
    write_templates(
        Path(env.dataset.pds4_bundle_template_dir()),
        {
            'global_bodies_index.lblx': GLOBAL_INDEX_TEMPLATE,
            'global_rings_index.lblx': GLOBAL_INDEX_TEMPLATE,
        },
    )
    two_bodies: dict[str, Any] = {
        'MOON_A': {'backplanes': {'latitude': {'min': 1.0, 'max': 2.0, 'units': 'deg'}}},
        'MOON_B': {'backplanes': {'latitude': {'min': 3.0, 'max': 4.0, 'units': 'deg'}}},
    }
    _write_image(env.bundle_dir / 'data', 'shard0/1234567890w', bodies=two_bodies, rings=RING_STATS)
    failed = _run_global_index(env)
    assert failed == 0
    index_dir = env.bundle_dir / 'miscellaneous'
    bodies_text = (index_dir / 'global_bodies_index.lblx').read_text(encoding='utf-8')
    assert '<records>2</records>' in bodies_text
    rings_text = (index_dir / 'global_rings_index.lblx').read_text(encoding='utf-8')
    assert '<records>1</records>' in rings_text
    bodies_lid = 'urn:nasa:pds:fake_bundle:miscellaneous:global_bodies_index'
    assert f'<lid>{bodies_lid}</lid>' in bodies_text
    rings_lid = 'urn:nasa:pds:fake_bundle:miscellaneous:global_rings_index'
    assert f'<lid>{rings_lid}</lid>' in rings_text


@pytest.mark.parametrize(
    ('broken', 'intact'),
    [
        ('global_bodies_index.lblx', 'global_rings_index.lblx'),
        ('global_rings_index.lblx', 'global_bodies_index.lblx'),
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
    _write_image(env.bundle_dir / 'data', 'shard0/1234567890w', bodies=BODY_STATS, rings=RING_STATS)
    failed = _run_global_index(env)
    assert failed == 1
    index_dir = env.bundle_dir / 'miscellaneous'
    assert not (index_dir / broken).exists()
    assert (index_dir / intact).is_file()


def test_a_missing_index_template_raises(tmp_path: Path) -> None:
    """An index template the dataset declares and does not have ends the run.

    As with the collection labels: the template tree does not carry what its
    dataset says it does, and an index table with no label describing it is
    worse than a run that stops.
    """
    env = _index_env(tmp_path)
    (Path(env.dataset.pds4_bundle_template_dir()) / 'global_bodies_index.lblx').unlink()
    _write_image(env.bundle_dir / 'data', 'shard0/1234567890w', bodies=BODY_STATS)
    with pytest.raises(FileNotFoundError, match=r'global_bodies_index\.lblx'):
        _run_global_index(env)


# ---------------------------------------------------------------------------
# LID cross-referencing and round trips: regression coverage for #139 and #256
# ---------------------------------------------------------------------------


def _cross_reference_env(tmp_path: Path) -> BundleEnv:
    """Build a one-image bundle with a label and a supplemental file in place.

    Parameters:
        tmp_path: Base temporary directory.

    Returns:
        The environment, ready for both phase-2 generators.
    """
    env = _index_env(tmp_path)
    _write_image(env.bundle_dir / 'data', 'shard0/1234567890w', bodies=BODY_STATS, rings=RING_STATS)
    return env


def test_global_bodies_index_lid_matches_collection_inventory(tmp_path: Path) -> None:
    """Round trip: the bodies-index LID equals the collection inventory LID."""
    env = _cross_reference_env(tmp_path)
    _run_global_index(env)
    run_collections(env)
    inventory_rows = read_csv_rows(env.bundle_dir / 'data' / 'collection_data.csv')
    inventory_lid = inventory_rows[0][1].split('::')[0]
    bodies_rows = read_index_rows(env.bundle_dir / 'miscellaneous' / 'global_bodies_index.tab')
    assert bodies_rows[1][0] == inventory_lid


def test_global_rings_index_lid_matches_collection_inventory(tmp_path: Path) -> None:
    """Round trip: the rings-index LID equals the collection inventory LID."""
    env = _cross_reference_env(tmp_path)
    _run_global_index(env)
    run_collections(env)
    inventory_rows = read_csv_rows(env.bundle_dir / 'data' / 'collection_data.csv')
    inventory_lid = inventory_rows[0][1].split('::')[0]
    rings_rows = read_index_rows(env.bundle_dir / 'miscellaneous' / 'global_rings_index.tab')
    assert rings_rows[1][0] == inventory_lid


def test_inventory_lidvid_round_trips_with_canonical_builders(tmp_path: Path) -> None:
    """With canonical LID builders, the inventory LIDVID is the data LID plus ::1.0."""
    env = _cross_reference_env(tmp_path)
    run_collections(env)
    rows = read_csv_rows(env.bundle_dir / 'data' / 'collection_data.csv')
    expected = env.dataset.pds4_image_name_to_data_lid('1234567890w') + '::1.0'
    assert rows[0][1] == expected
