"""Tests of the targets the Cassini ISS Saturn bundle's labels name.

The shipped configuration's targets table is the one place a target's context product is
identified, so every body and ring target the backplane stage can produce for an image of
Saturn has to have an entry there: an image naming one without could not be labeled.
These hold the shipped table to the stage's own rules for the bodies and the ring target
it looks for, and hold the data labels the shipped templates render over the cohort to
the targets their images' backplanes cover.  What the labels pass does with the targets is
tested over stand-in templates in ``test_bundle_data.py``.
"""

import json
from pathlib import Path
from xml.etree import ElementTree

import pytest
from filecache import FCPath
from tests.mini_nav_results.cohort import Cohort, WrittenCohorts
from tests.mini_nav_results.cohort_cassini import LIMB_STUB, RINGS_STUB, CohortCassiniISSSaturn

from spindoctor.cli.backplanes.backplanes_bodies import backplane_body_names
from spindoctor.cli.backplanes.backplanes_rings import ring_target
from spindoctor.cli.pds4.bundle_data import generate_bundle_data_files
from spindoctor.cli.pds4.targets import target_table
from spindoctor.config import DEFAULT_CONFIG, MAIN_LOGGER

from .conftest import (
    label_cohort_images,
    make_cohort_bundle_env,
    read_csv_rows,
    write_cohort_bundle,
)

PDS4_NAMESPACES = {'pds': 'http://pds.nasa.gov/pds4/pds/v1'}
"""The PDS4 common dictionary's namespace, under the prefix the paths below use."""

ENCELADUS = ('Enceladus', 'Satellite', 'urn:nasa:pds:context:target:satellite.saturn.enceladus')
"""The limb image's body, as its context product gives it: name, type and LID."""

SATURN = ('Saturn', 'Planet', 'urn:nasa:pds:context:target:planet.saturn')
"""The ring image's body, as its context product gives it."""

SATURN_RINGS = ('Saturn Rings', 'Ring', 'urn:nasa:pds:context:target:ring.saturn.rings')
"""The ring image's ring target, as its context product gives it."""


@pytest.fixture
def cassini_cohort(mini_nav_cohorts: WrittenCohorts) -> CohortCassiniISSSaturn:
    """Return the Cassini ISS Saturn cohort, as the session wrote it.

    Parameters:
        mini_nav_cohorts: What the session's cohorts are written by.

    Returns:
        The written cohort.
    """
    return mini_nav_cohorts(CohortCassiniISSSaturn)


def _targets_named(label: Path, area: str) -> list[tuple[str | None, ...]]:
    """Return every target a label names, in the order it names them.

    Parameters:
        label: The label.
        area: The area its targets are in: ``Observation_Area`` in a data label, and
            ``Context_Area`` in a bundle, collection or SPICE kernel label.

    Returns:
        Each ``Target_Identification``'s name, type, LID reference and reference type.
    """
    root = ElementTree.parse(label).getroot()
    return [
        (
            target.findtext('pds:name', namespaces=PDS4_NAMESPACES),
            target.findtext('pds:type', namespaces=PDS4_NAMESPACES),
            target.findtext('pds:Internal_Reference/pds:lid_reference', namespaces=PDS4_NAMESPACES),
            target.findtext(
                'pds:Internal_Reference/pds:reference_type', namespaces=PDS4_NAMESPACES
            ),
        )
        for target in root.iterfind(f'pds:{area}/pds:Target_Identification', PDS4_NAMESPACES)
    ]


def _data_label(bundle_dir: Path) -> Path:
    """Return the one data label a bundle holds.

    Parameters:
        bundle_dir: The bundle's own directory, into which one image was labeled.

    Returns:
        The data label.
    """
    (label,) = (bundle_dir / 'data').rglob('*_backplanes.lblx')
    return label


def test_every_body_the_backplane_stage_looks_for_in_a_saturn_image_has_a_target() -> None:
    """Saturn and every satellite the configuration lists for it have an entry."""
    table = target_table(DEFAULT_CONFIG)
    bodies = backplane_body_names('SATURN', DEFAULT_CONFIG)
    assert [body for body in bodies if body not in table] == []


def test_the_ring_target_of_a_saturn_image_has_a_target() -> None:
    """The ring target a Saturn image's ring backplanes are computed for has an entry."""
    assert ring_target('SATURN') in target_table(DEFAULT_CONFIG)


@pytest.mark.parametrize(
    ('stub', 'expected'),
    [(LIMB_STUB, [ENCELADUS]), (RINGS_STUB, [SATURN, SATURN_RINGS])],
    ids=['limb image', 'ring image'],
)
def test_a_cohort_data_label_names_each_target_its_backplanes_cover(
    cassini_cohort: Cohort, tmp_path: Path, stub: str, expected: list[tuple[str, str, str]]
) -> None:
    """The limb image names Enceladus, and the ring image Saturn and Saturn's rings.

    Each is named as its context product gives it, by its LID, with the reference type a
    data product's target takes.  The limb image's backplane metadata names Saturn's ring
    target beside no ring statistic, which names no target.
    """
    env = make_cohort_bundle_env(cassini_cohort, tmp_path)
    label_cohort_images(env, [stub])
    named = _targets_named(_data_label(env.bundle_dir), 'Observation_Area')
    assert named == [(*target, 'data_to_target') for target in expected]


def test_the_data_label_of_an_image_with_two_bodies_names_both(
    cassini_cohort: Cohort, tmp_path: Path
) -> None:
    """An image whose backplanes cover two bodies has a Target_Identification for each.

    The limb image's backplane metadata is given a second body, Mimas, holding
    Enceladus's statistics, under a backplane root of the test's own beside the image's
    FITS.  The two are named in the targets table's order.
    """
    source = cassini_cohort.backplane_results_root
    backplane_root = tmp_path / 'backplanes'
    metadata_name = f'{LIMB_STUB}_backplane_metadata.json'
    metadata = json.loads((source / metadata_name).read_text(encoding='utf-8'))
    metadata['bodies']['MIMAS'] = metadata['bodies']['ENCELADUS']
    (backplane_root / metadata_name).parent.mkdir(parents=True)
    (backplane_root / metadata_name).write_text(json.dumps(metadata), encoding='utf-8')
    fits_name = f'{LIMB_STUB}_backplanes.fits'
    (backplane_root / fits_name).write_bytes((source / fits_name).read_bytes())
    env = make_cohort_bundle_env(cassini_cohort, tmp_path)
    generate_bundle_data_files(
        env.dataset,
        cassini_cohort.batch(LIMB_STUB),
        nav_results_root=FCPath(cassini_cohort.nav_results_root),
        backplane_results_root=FCPath(backplane_root),
        bundle_results_root=FCPath(env.bundle_results_root),
        logger=MAIN_LOGGER,
    )
    named = _targets_named(_data_label(env.bundle_dir), 'Observation_Area')
    assert [name for name, *_ in named] == ['Enceladus', 'Mimas']


@pytest.mark.parametrize(
    ('label', 'reference_type'),
    [
        ('bundle.lblx', 'bundle_to_target'),
        ('data/collection_data.lblx', 'collection_to_target'),
        ('spice_kernels/kernels.lblx', 'data_to_target'),
        ('spice_kernels/collection_spice_kernels.lblx', 'collection_to_target'),
    ],
    ids=['bundle', 'data collection', 'metakernel', 'kernel collection'],
)
def test_a_run_level_label_names_every_target_the_data_labels_name(
    cassini_cohort: Cohort, tmp_path: Path, label: str, reference_type: str
) -> None:
    """The bundle, the data and SPICE kernel collections and the metakernel name the targets.

    Every target a data label names, each once, in the targets table's order, with the
    reference type the Schematron allows under that kind of product's context area.
    """
    env = write_cohort_bundle(cassini_cohort, tmp_path, (LIMB_STUB, RINGS_STUB))
    named = _targets_named(env.bundle_dir / label, 'Context_Area')
    assert named == [(*target, reference_type) for target in (SATURN, ENCELADUS, SATURN_RINGS)]


def test_no_inventory_but_the_context_inventory_lists_a_target(
    cassini_cohort: Cohort, tmp_path: Path
) -> None:
    """The context inventory lists the targets, and no other inventory of the bundle does.

    No document or miscellaneous label names a target, and the data and SPICE kernel
    inventories list their own products alone.
    """
    env = write_cohort_bundle(cassini_cohort, tmp_path, (LIMB_STUB, RINGS_STUB))
    listing = {
        inventory.relative_to(env.bundle_dir).as_posix()
        for inventory in env.bundle_dir.rglob('collection_*.csv')
        if any(':context:target:' in lidvid for _, lidvid in read_csv_rows(inventory))
    }
    assert listing == {'context/collection_context.csv'}


def test_every_target_a_data_label_names_is_in_the_context_inventory(
    cassini_cohort: Cohort, tmp_path: Path
) -> None:
    """Each target LID a cohort data label references is a member of the context collection."""
    env = write_cohort_bundle(cassini_cohort, tmp_path, (LIMB_STUB, RINGS_STUB))
    rows = read_csv_rows(env.bundle_dir / 'context' / 'collection_context.csv')
    listed = [lidvid.partition('::')[0] for _, lidvid in rows]
    named = [
        lid
        for label in (env.bundle_dir / 'data').rglob('*_backplanes.lblx')
        for _, _, lid, _ in _targets_named(label, 'Observation_Area')
    ]
    assert [lid for lid in named if lid not in listed] == []


def test_the_context_inventory_lists_each_target_at_its_registered_version(
    cassini_cohort: Cohort, tmp_path: Path
) -> None:
    """After the mission, the spacecraft and the two cameras, each target at its version.

    The versions are the ones the PDS registry held for the targets' context products.
    """
    env = write_cohort_bundle(cassini_cohort, tmp_path, (LIMB_STUB, RINGS_STUB))
    rows = read_csv_rows(env.bundle_dir / 'context' / 'collection_context.csv')
    assert rows[4:] == [
        ['S', 'urn:nasa:pds:context:target:planet.saturn::1.4'],
        ['S', 'urn:nasa:pds:context:target:satellite.saturn.enceladus::1.2'],
        ['S', 'urn:nasa:pds:context:target:ring.saturn.rings::1.1'],
    ]
