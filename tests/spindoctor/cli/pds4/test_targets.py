"""Tests for the targets a bundle's labels name, over plumbing backplane metadata.

What the shipped targets table holds, and what the shipped templates make of it, is
tested in ``test_targets_cassini_iss_saturn.py``.
"""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from filecache import FCPath

from spindoctor.cli.pds4.global_index import generate_global_index_files
from spindoctor.cli.pds4.targets import Pds4Target, TargetScan, image_targets, target_table
from spindoctor.config import MAIN_LOGGER, Config

from .conftest import (
    PLUMBING_RING_TARGET,
    make_bundle_env,
    ring_metadata,
    touch_label,
    write_supplemental,
)


def _target(key: str) -> Pds4Target:
    """Return a stand-in target for a name.

    Parameters:
        key: The name the backplane metadata gives the target.

    Returns:
        A satellite at version 1.0 whose LID and name are made from the key.
    """
    return Pds4Target(
        lid=f'urn:nasa:pds:context:target:fake.{key.lower()}',
        version='1.0',
        name=key.title(),
        target_type='Satellite',
    )


TABLE = {key: _target(key) for key in ('PLANET', 'MOON_A', 'MOON_B', 'PLANET_RINGS')}
"""A targets table, in an order that is neither alphabetical nor the metadata's below."""


def _metadata(*bodies: str, rings: bool = False) -> dict[str, Any]:
    """Return backplane metadata naming some bodies and the ring target.

    Parameters:
        *bodies: The bodies the ``bodies`` block names, in order, none with a statistic.
        rings: Whether the ``rings`` block holds a ring statistic.

    Returns:
        The metadata, in the shape the backplane stage writes it.
    """
    ring_statistics = {'ring_radius': {'min': 1.0, 'max': 2.0, 'units': 'km'}} if rings else {}
    return {
        'bodies': {body: {'backplanes': {}} for body in bodies},
        'rings': {
            'target': 'PLANET_RINGS',
            'incidence_angle': {'value': 45.0, 'units': 'deg'},
            'backplanes': ring_statistics,
        },
    }


def test_an_image_s_targets_are_its_bodies_and_its_rings_in_the_table_s_order() -> None:
    """Every body the metadata names, and the ring target, each in the table's order.

    The metadata names the bodies in an order other than the table's, and neither body
    has a statistic, which does not keep a body from being a target.
    """
    targets = image_targets(_metadata('MOON_B', 'PLANET', rings=True), TABLE)
    assert targets == (TABLE['PLANET'], TABLE['MOON_B'], TABLE['PLANET_RINGS'])


def test_rings_with_no_ring_statistic_are_no_target_of_an_image() -> None:
    """The ring target is named beside every ring result, and is a target only with a statistic.

    The backplane stage names the ring target for an image whose rings are out of the
    field too, and such an image has no ring backplanes for a label to describe.
    """
    assert image_targets(_metadata('MOON_A'), TABLE) == (TABLE['MOON_A'],)


def test_a_target_with_no_table_entry_is_refused_by_name() -> None:
    """A name the table has no entry for is refused, the message naming it."""
    with pytest.raises(KeyError, match=r'backplanes\.target_lids has no entry for MOON_C'):
        image_targets(_metadata('MOON_C', 'MOON_A'), TABLE)


def test_a_scan_s_targets_are_every_product_s_each_once_in_the_table_s_order() -> None:
    """Over two products naming one body alike, each target is there once, in order."""
    scan = TargetScan(TABLE)
    scan.include(_metadata('MOON_B', rings=True))
    scan.include(_metadata('MOON_B', 'PLANET'))
    assert scan.result() == (TABLE['PLANET'], TABLE['MOON_B'], TABLE['PLANET_RINGS'])


def test_a_scan_refuses_a_target_with_no_table_entry_at_the_product_naming_it() -> None:
    """A product naming a target the table has no entry for is refused as it is taken in."""
    scan = TargetScan(TABLE)
    with pytest.raises(KeyError, match='MOON_C'):
        scan.include(_metadata('MOON_C'))


def test_the_index_takes_the_targets_the_data_collection_s_members_name(tmp_path: Path) -> None:
    """The summary pass's read takes the targets of the images the data collection holds.

    The first image has a data label beside its supplemental file and names a body and
    the rings.  The second has a supplemental file and no data label, which is not a
    member, and names a body the first does not, which is not among the targets.
    """
    env = make_bundle_env(tmp_path)
    data_dir = env.bundle_dir / 'data'
    touch_label(data_dir, 'shard0/1111111111n')
    radii = ring_metadata({'radius': {'min': 81000.0, 'max': 125000.0, 'units': 'km'}})
    write_supplemental(
        data_dir, 'shard0/1111111111n', bodies={'MOON_B': {'backplanes': {}}}, rings=radii
    )
    write_supplemental(data_dir, 'shard0/2222222222w', bodies={'MOON_A': {'backplanes': {}}})
    dataset = env.dataset.as_dataset()
    index = generate_global_index_files(FCPath(env.bundle_results_root), dataset, MAIN_LOGGER)
    table = target_table(dataset.config)
    assert index.targets == (table['MOON_B'], table[PLUMBING_RING_TARGET])


def test_the_table_s_entries_become_targets_in_order_each_version_as_text() -> None:
    """Each entry is one target, in the configuration's order, its version as text.

    A version YAML reads as a number is still stated as the text an inventory line gives
    it.
    """
    target_lids = {
        'MOON_B': {'lid': 'urn:b', 'version': 1.2, 'name': 'B', 'type': 'Satellite'},
        'MOON_A': {'lid': 'urn:a', 'version': '1.10', 'name': 'A', 'type': 'Satellite'},
    }
    config = cast(Config, SimpleNamespace(backplanes=SimpleNamespace(target_lids=target_lids)))
    assert list(target_table(config).values()) == [
        Pds4Target(lid='urn:b', version='1.2', name='B', target_type='Satellite'),
        Pds4Target(lid='urn:a', version='1.10', name='A', target_type='Satellite'),
    ]
