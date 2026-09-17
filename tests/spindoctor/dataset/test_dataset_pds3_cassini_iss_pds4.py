"""Hermetic tests of the Cassini Saturn dataset's PDS4 hooks and shipped template tree.

The dataset is built on an empty holdings root under ``tmp_path``.  The LID and LIDVID
builders, the LID-part inverse, the bundle path and path stub, the configuration
lookups, the template variables and the template directory read nothing from the
holdings, so these tests run without ``PDS3_HOLDINGS_DIR``, which every test in
``test_dataset_pds3_cassini_iss.py`` requires.
"""

import re
from pathlib import Path

import julian
import pytest
from tests.spindoctor.cli.pds4.conftest import make_image_file, navigated_document

from spindoctor.config import Config
from spindoctor.dataset.dataset_pds3_cassini_iss import DataSetPDS3CassiniISSSaturn

# ---------------------------------------------------------------------------
# Real Cassini dataset: LID/LIDVID construction and template tree
# ---------------------------------------------------------------------------


def _cassini_dataset(
    tmp_path: Path, *, config: Config | None = None
) -> DataSetPDS3CassiniISSSaturn:
    """Construct the reference Cassini Saturn dataset on a local holdings root.

    Parameters:
        tmp_path: Base temporary directory used as the (empty) holdings root.
        config: Optional Config override; DEFAULT_CONFIG when None.
    """
    return DataSetPDS3CassiniISSSaturn(tmp_path / 'holdings', config=config)


def test_cassini_data_lid_canonical_form(tmp_path: Path) -> None:
    """The Cassini data LID rotates the camera letter to a lowercase suffix."""
    dataset = _cassini_dataset(tmp_path)
    lid = dataset.pds4_image_name_to_data_lid('N1454725799')
    assert lid == 'urn:nasa:pds:cassini_iss_saturn_backplanes_rsfrench2027:data:1454725799n'


def test_cassini_lidvid_version_field(tmp_path: Path) -> None:
    """LIDVIDs append a ::1.0 version to the corresponding LID."""
    dataset = _cassini_dataset(tmp_path)
    lid = dataset.pds4_image_name_to_data_lid('N1454725799')
    lidvid = dataset.pds4_image_name_to_data_lidvid('N1454725799')
    assert lidvid == f'{lid}::1.0'


def test_cassini_browse_lid_uses_browse_collection(tmp_path: Path) -> None:
    """Browse LIDs differ from data LIDs only in the collection segment."""
    dataset = _cassini_dataset(tmp_path)
    lid = dataset.pds4_image_name_to_browse_lid('N1454725799')
    assert lid == 'urn:nasa:pds:cassini_iss_saturn_backplanes_rsfrench2027:browse:1454725799n'


def test_cassini_lid_part_to_image_name_inverts_rotation(tmp_path: Path) -> None:
    """The LID-part inverse moves the trailing letter back to an uppercase prefix."""
    dataset = _cassini_dataset(tmp_path)
    assert dataset.pds4_lid_part_to_image_name('1454725799n') == 'N1454725799'


def test_cassini_lid_part_round_trips_through_data_lid(tmp_path: Path) -> None:
    """An on-disk LID part recovers the image name whose data LID embeds it."""
    dataset = _cassini_dataset(tmp_path)
    image_name = dataset.pds4_lid_part_to_image_name('1454725799n')
    lid = dataset.pds4_image_name_to_data_lid(image_name)
    assert lid.endswith(':data:1454725799n')


def test_cassini_lid_part_to_image_name_rejects_too_short(tmp_path: Path) -> None:
    """A LID part shorter than two characters is a programming error."""
    dataset = _cassini_dataset(tmp_path)
    with pytest.raises(ValueError, match='invalid Cassini LID part'):
        dataset.pds4_lid_part_to_image_name('n')


def test_cassini_lid_strips_version_suffix_and_extension(tmp_path: Path) -> None:
    """Image-name version suffixes and extensions do not leak into the LID."""
    dataset = _cassini_dataset(tmp_path)
    plain = dataset.pds4_image_name_to_data_lid('N1454725799')
    suffixed = dataset.pds4_image_name_to_data_lid('N1454725799_1.IMG')
    assert suffixed == plain


@pytest.mark.parametrize(
    'template_name',
    [
        'bundle.lblx',
        'readme.txt',
        'data.lblx',
        'browse.lblx',
        'collection_data.lblx',
        'collection_browse.lblx',
        'collection_context.lblx',
        'collection_context.csv',
        'collection_document.lblx',
        'collection_document.csv',
        'collection_xml_schema.lblx',
        'collection_xml_schema.csv',
        'global_index_bodies.lblx',
        'global_index_rings.lblx',
        'cassini-iss-saturn-backplanes-user-guide.lblx',
    ],
)
def test_cassini_template_tree_ships_documented_files(tmp_path: Path, template_name: str) -> None:
    """Every file in the dev guide's reference template tree ships as package data."""
    dataset = _cassini_dataset(tmp_path)
    template_dir = Path(dataset.pds4_bundle_template_dir())
    assert (template_dir / template_name).is_file()


def test_cassini_declares_only_templates_it_ships(tmp_path: Path) -> None:
    """Every template the Cassini dataset declares required is in its shipped tree.

    Each pass refuses to run when a template it declares is not there, so a
    declaration naming a file the package does not ship would stop every run of
    that pass rather than one product of it.
    """
    dataset = _cassini_dataset(tmp_path)
    template_dir = Path(dataset.pds4_bundle_template_dir())
    declared = dataset.pds4_required_templates('labels') + dataset.pds4_required_templates(
        'summary'
    )
    assert [name for name in declared if not (template_dir / name).is_file()] == []


# ---------------------------------------------------------------------------
# Real Cassini dataset: bundle paths, configuration and template variables
# ---------------------------------------------------------------------------


def test_cassini_bundle_path_for_image_shards_by_image_number(tmp_path: Path) -> None:
    """Cassini image names shard into 1234xxxxxx/123456xxxx/ directories."""
    dataset = _cassini_dataset(tmp_path)
    assert dataset.pds4_bundle_path_for_image('N1454725799') == '1454xxxxxx/145472xxxx/'


def test_cassini_bundle_path_rejects_short_image_name(tmp_path: Path) -> None:
    """A too-short Cassini image name raises instead of building a malformed path."""
    dataset = _cassini_dataset(tmp_path)
    with pytest.raises(ValueError, match='invalid Cassini image name'):
        dataset.pds4_bundle_path_for_image('N123')


def test_cassini_path_stub_appends_lid_part(tmp_path: Path) -> None:
    """The path stub is the shard path plus the rotated lowercase image LID part."""
    dataset = _cassini_dataset(tmp_path)
    image_file = make_image_file('N1454725799_1')
    assert dataset.pds4_path_stub(image_file) == '1454xxxxxx/145472xxxx/1454725799n'


def test_cassini_default_bundle_name_from_config(tmp_path: Path) -> None:
    """The bundle name comes from the shipped pds4.coiss_saturn config block."""
    dataset = _cassini_dataset(tmp_path)
    assert dataset.pds4_bundle_name() == 'cassini_iss_saturn_backplanes_rsfrench2027'


def test_cassini_default_template_dir_is_shipped_package_data(tmp_path: Path) -> None:
    """The default template dir resolves inside the shipped templates package data."""
    dataset = _cassini_dataset(tmp_path)
    template_dir = Path(dataset.pds4_bundle_template_dir())
    assert template_dir.name == 'cassini_iss_saturn_1.0'
    assert template_dir.parent.name == 'templates'
    assert (template_dir / 'data.lblx').is_file()
    assert (template_dir / 'browse.lblx').is_file()
    assert (template_dir / 'collection_data.lblx').is_file()
    assert (template_dir / 'global_index_bodies.lblx').is_file()


def test_cassini_config_overrides_template_dir_and_bundle_name(tmp_path: Path) -> None:
    """config pds4.<dataset>.template_dir/bundle_name override the defaults."""
    override = tmp_path / 'override.yaml'
    override.write_text(
        'pds4:\n'
        '  coiss_saturn:\n'
        '    template_dir: /absolute/custom/templates\n'
        '    bundle_name: custom_bundle_name\n',
        encoding='utf-8',
    )
    config = Config()
    config.update_config(override)
    dataset = _cassini_dataset(tmp_path, config=config)
    assert dataset.pds4_bundle_template_dir() == '/absolute/custom/templates'
    assert dataset.pds4_bundle_name() == 'custom_bundle_name'


def test_cassini_relative_template_dir_override_resolves_under_templates(
    tmp_path: Path,
) -> None:
    """A bare-name template_dir override resolves under the packaged templates dir."""
    override = tmp_path / 'override.yaml'
    override.write_text(
        'pds4:\n  coiss_saturn:\n    template_dir: my_custom_set\n', encoding='utf-8'
    )
    config = Config()
    config.update_config(override)
    dataset = _cassini_dataset(tmp_path, config=config)
    template_dir = Path(dataset.pds4_bundle_template_dir())
    assert template_dir.name == 'my_custom_set'
    assert template_dir.parent.name == 'templates'


def test_cassini_data_label_lid_matches_dataset_builder(tmp_path: Path) -> None:
    """The DATA_LID template variable equals pds4_image_name_to_data_lid's output."""
    dataset = _cassini_dataset(tmp_path)
    image_file = make_image_file('N1454725799_1')
    variables = dataset.pds4_template_variables(
        image_file=image_file, nav_metadata=navigated_document(), backplane_metadata={}
    )
    assert variables['DATA_LID'] == dataset.pds4_image_name_to_data_lid('N1454725799_1')
    assert variables['BROWSE_LID'] == dataset.pds4_image_name_to_browse_lid('N1454725799_1')


def test_cassini_camera_variables_from_image_name(tmp_path: Path) -> None:
    """The camera template variables derive from the image name's leading letter."""
    dataset = _cassini_dataset(tmp_path)
    variables = dataset.pds4_template_variables(
        image_file=make_image_file('W1454725799_1'),
        nav_metadata=navigated_document(),
        backplane_metadata={},
    )
    assert variables['CAMERA_WIDTH'] == 'Wide'
    assert variables['CAMERA_WN_UC'] == 'W'
    assert variables['CAMERA_WN_LC'] == 'w'


def test_a_start_nanoseconds_short_of_its_millisecond_is_written_as_pds3_states_it(
    tmp_path: Path,
) -> None:
    """A start computed from times recorded to the millisecond is written as recorded.

    W1630770594's PDS3 label records an ``IMAGE_TIME`` of ``2009-247T15:07:30.812``
    and an ``EXPOSURE_DURATION`` of 50 ms, with a ``START_TIME`` of
    ``2009-247T15:07:30.762``, a ``STOP_TIME`` of ``2009-247T15:07:30.812`` and an
    ``IMAGE_MID_TIME`` of ``2009-247T15:07:30.787``.  The epochs are built as oops
    builds them, the stop from ``IMAGE_TIME``, the start as the stop less the
    exposure and the midtime halfway between, and are the ones the image's
    navigation document records.  SPICE's ``et2utc`` writes the start at nine
    decimals as ``15:07:30.761999965``, so rounded down it would be written a
    millisecond before the time PDS3 states; the midtime is the midpoint of the start
    and stop as written.  The expected strings are the PDS3 label's three times, in
    the PDS4 spelling.
    """
    stop_et = float(julian.tdb_from_tai(julian.tai_from_iso('2009-247T15:07:30.812')))
    start_et = stop_et - 50.0 / 1000.0
    times = {'start_et': start_et, 'stop_et': stop_et, 'midtime_et': (start_et + stop_et) / 2}
    variables = _cassini_dataset(tmp_path).pds4_template_variables(
        image_file=make_image_file('W1630770594_1'),
        nav_metadata={'status': 'success', 'navigation_result': {'times': times}},
        backplane_metadata={},
    )
    assert variables['START_DATE_TIME'] == '2009-09-04T15:07:30.762Z'
    assert variables['STOP_DATE_TIME'] == '2009-09-04T15:07:30.812Z'
    assert variables['IMAGE_MID_TIME'] == '2009-09-04T15:07:30.787Z'


def test_an_odd_millisecond_exposure_s_midtime_is_its_half_millisecond_taken_up(
    tmp_path: Path,
) -> None:
    """An exposure an odd number of milliseconds long has its midtime rounded up.

    W1629783475's PDS3 label records an ``IMAGE_TIME`` of ``2009-236T04:55:38.829``
    and an ``EXPOSURE_DURATION`` of 5 ms, with a ``START_TIME`` of
    ``2009-236T04:55:38.824`` and an ``IMAGE_MID_TIME`` of ``2009-236T04:55:38.827``:
    the midtime is half a millisecond past ``.826``, taken up.  The epochs are built as
    oops builds them, the stop from ``IMAGE_TIME``, the start as the stop less the
    exposure and the midtime halfway between; SPICE's ``et2utc`` writes that midtime
    epoch at nine decimals as ``04:55:38.826499999``, whose nearest millisecond is
    ``.826``.  The expected string is the PDS3 label's ``IMAGE_MID_TIME``, in the PDS4
    spelling.
    """
    stop_et = float(julian.tdb_from_tai(julian.tai_from_iso('2009-236T04:55:38.829')))
    start_et = stop_et - 5.0 / 1000.0
    times = {'start_et': start_et, 'stop_et': stop_et, 'midtime_et': (start_et + stop_et) / 2}
    variables = _cassini_dataset(tmp_path).pds4_template_variables(
        image_file=make_image_file('W1629783475_1'),
        nav_metadata={'status': 'success', 'navigation_result': {'times': times}},
        backplane_metadata={},
    )
    assert variables['IMAGE_MID_TIME'] == '2009-08-24T04:55:38.827Z'


def test_cassini_lid_charset_is_pds4_legal(tmp_path: Path) -> None:
    """Cassini LIDs are lowercase urn:nasa:pds identifiers with a legal charset."""
    dataset = _cassini_dataset(tmp_path)
    lid = dataset.pds4_image_name_to_data_lid('N1454725799_1.IMG')
    assert lid == lid.lower()
    assert re.fullmatch(r'urn:nasa:pds(:[a-z0-9_-]+)+', lid) is not None
