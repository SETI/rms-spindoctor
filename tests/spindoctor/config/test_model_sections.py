"""Tests for the typed configuration sections the navigation models read.

The schema core checks each key's type and range, and is tested in its own suite.
What is tested here is that each packaged section parses, each section's cross-field
rule, the per-planet lookup of the ring emission table, and that the loader refuses a
bad override at load.
"""

import re
from pathlib import Path

import pytest

from spindoctor.config import Config
from spindoctor.config.sections_models import (
    BodiesSection,
    FeatureEmissionSection,
    RingAnnulusEmission,
    RingAnnulusEmissionOverride,
    RingAnnulusEmissionTable,
    StarsSection,
    TitanSection,
)
from spindoctor.schema import SchemaError


def test_packaged_bodies_section_parses() -> None:
    """The packaged ``bodies`` section parses."""
    assert isinstance(Config().bodies, BodiesSection)


def test_packaged_stars_section_parses() -> None:
    """The packaged ``stars`` section parses."""
    assert isinstance(Config().stars, StarsSection)


def test_packaged_titan_section_parses() -> None:
    """The packaged ``titan`` section parses."""
    assert isinstance(Config().titan, TitanSection)


def test_packaged_satellites_section_parses() -> None:
    """The packaged ``satellites`` section parses, and a planet it leaves out has none."""
    config = Config()
    assert 'TITAN' in config.satellites('saturn')
    assert config.satellites('NOT_A_PLANET') == []


def test_packaged_feature_emission_section_parses() -> None:
    """The packaged ``feature_emission`` section parses."""
    assert isinstance(Config().feature_emission, FeatureEmissionSection)


def test_a_planet_entry_keeps_the_default_for_a_key_it_leaves_out() -> None:
    """A planet entry replaces only the keys it sets; a planet with none gets the default."""
    table = RingAnnulusEmissionTable(
        default=RingAnnulusEmission(max_radial_px=5.0, kmpp_threshold=1000.0),
        planets={
            'SATURN': RingAnnulusEmissionOverride(kmpp_threshold=25.0),
            'URANUS': RingAnnulusEmissionOverride(max_radial_px=3.0),
        },
    )
    saturn = RingAnnulusEmission(max_radial_px=5.0, kmpp_threshold=25.0)
    uranus = RingAnnulusEmission(max_radial_px=3.0, kmpp_threshold=1000.0)
    assert table.for_planet('SATURN') == saturn
    assert table.for_planet('URANUS') == uranus
    assert table.for_planet('PLUTO') is table.default


@pytest.mark.parametrize(
    ('text', 'expected'),
    [
        (
            'stars:\n  ring_occlusion_radii_km:\n    SATURN: [[100.0, 100.0]]\n',
            'stars.ring_occlusion_radii_km.SATURN[0]: the inner radius (100.0 km) must be '
            'below the outer radius (100.0 km)',
        ),
        (
            'titan:\n  navigation:\n    ring_occlusion_radii_km: [200.0, 100.0]\n',
            'titan.navigation.ring_occlusion_radii_km: the inner radius (200.0 km) must be '
            'below the outer radius (100.0 km)',
        ),
    ],
    ids=['stars', 'titan'],
)
def test_an_annulus_whose_inner_radius_is_not_below_its_outer_is_refused(
    tmp_path: Path, text: str, expected: str
) -> None:
    """An opaque annulus must have its inner radius below its outer radius.

    Parameters:
        tmp_path: A directory for the override file.
        text: The override file's text.
        expected: The message's key path and problem.
    """
    user_path = tmp_path / 'user_config.yaml'
    user_path.write_text(text, encoding='utf-8')
    with pytest.raises(SchemaError, match=re.escape(f'{user_path}: {expected}')):
        Config().update_config(user_path)


def test_an_unknown_key_is_refused_and_changes_nothing(tmp_path: Path) -> None:
    """An unknown nested key is refused at load, naming the file and the key path."""
    config = Config()
    section = config.titan
    user_path = tmp_path / 'user_config.yaml'
    user_path.write_text('titan:\n  navigation:\n    arc:\n      min_ray: 3\n', encoding='utf-8')
    expected = f"{user_path}: titan.navigation.arc: unknown keys: ['min_ray']"
    with pytest.raises(SchemaError, match=re.escape(expected)):
        config.update_config(user_path)
    assert config.titan is section
