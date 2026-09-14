"""Tests of the targets the Cassini ISS Saturn bundle's labels name.

The shipped configuration's targets table is the one place a target's context product is
identified, so every body and ring target the backplane stage can produce for an image of
Saturn has to have an entry there: an image naming one without could not be labeled.
These hold the shipped table to the stage's own rules for the bodies and the ring target
it looks for.
"""

from spindoctor.cli.backplanes.backplanes_bodies import backplane_body_names
from spindoctor.cli.backplanes.backplanes_rings import ring_target
from spindoctor.cli.pds4.targets import target_table
from spindoctor.config import DEFAULT_CONFIG


def test_every_body_the_backplane_stage_looks_for_in_a_saturn_image_has_a_target() -> None:
    """Saturn and every satellite the configuration lists for it have an entry."""
    table = target_table(DEFAULT_CONFIG)
    bodies = backplane_body_names('SATURN', DEFAULT_CONFIG)
    assert [body for body in bodies if body not in table] == []


def test_the_ring_target_of_a_saturn_image_has_a_target() -> None:
    """The ring target a Saturn image's ring backplanes are computed for has an entry."""
    assert ring_target('SATURN') in target_table(DEFAULT_CONFIG)
