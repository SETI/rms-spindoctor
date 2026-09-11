"""Hermetic tests of the Cassini Saturn dataset's PDS4 hooks and shipped template tree.

The dataset is built on an empty holdings root under ``tmp_path``.  The LID and LIDVID
builders, the LID-part inverse and the template directory read nothing from the
holdings, so these tests run without ``PDS3_HOLDINGS_DIR``, which every test in
``test_dataset_pds3_cassini_iss.py`` requires.
"""

from pathlib import Path

import pytest

from spindoctor.dataset.dataset_pds3_cassini_iss import DataSetPDS3CassiniISSSaturn

# ---------------------------------------------------------------------------
# Real Cassini dataset: LID/LIDVID construction and template tree
# ---------------------------------------------------------------------------


def _cassini_dataset(tmp_path: Path) -> DataSetPDS3CassiniISSSaturn:
    """Construct the reference Cassini Saturn dataset on a local holdings root.

    Parameters:
        tmp_path: Base temporary directory used as the (empty) holdings root.
    """
    return DataSetPDS3CassiniISSSaturn(tmp_path / 'holdings')


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
