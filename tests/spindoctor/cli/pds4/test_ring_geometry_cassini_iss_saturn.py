"""Tests of the ring geometry the Cassini ISS Saturn bundle's data labels state.

Over the cohort and the shipped templates: the ring image's data label states
``rings:Reprojection_Geometry``, filled from its ring backplane statistics and the
incidence angle its backplane metadata records, and the limb image's, whose backplanes
cover no rings, states none.  The geometry is built over stand-in metadata in
``test_ring_geometry.py``.
"""

import json
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import pytest
from tests.mini_nav_results.cohort import Cohort, WrittenCohorts
from tests.mini_nav_results.cohort_cassini import LIMB_STUB, RINGS_STUB, CohortCassiniISSSaturn

from spindoctor.cli.pds4.global_index import INDEX_VALUE_FORMATS
from spindoctor.support.time import pds4_utc_midpoint

from .conftest import label_cohort_images, make_cohort_bundle_env

NAMESPACES = {
    'pds': 'http://pds.nasa.gov/pds4/pds/v1',
    'rings': 'http://pds.nasa.gov/pds4/rings/v1',
}
"""The PDS4 common and rings dictionaries' namespaces, under the prefixes the paths use."""

RING_SYSTEMS = 'pds:Observation_Area/pds:Discipline_Area/rings:Ring_Moon_Systems'
"""Where a data label states the ring geometry, below its root."""

GEOMETRY = f'{RING_SYSTEMS}/rings:Ring_Reprojection/rings:Reprojection_Geometry'
"""The class holding an image's ring geometry, below a data label's root."""


@pytest.fixture
def cassini_cohort(mini_nav_cohorts: WrittenCohorts) -> CohortCassiniISSSaturn:
    """Return the Cassini ISS Saturn cohort, as the session wrote it.

    Parameters:
        mini_nav_cohorts: What the session's cohorts are written by.

    Returns:
        The written cohort.
    """
    return mini_nav_cohorts(CohortCassiniISSSaturn)


def _labelled(cohort: Cohort, tmp_path: Path, stub: str) -> ElementTree.Element:
    """Label one cohort image with the shipped templates and return its data label's root.

    Parameters:
        cohort: The session's cohort.
        tmp_path: Base temporary directory for this test's bundle.
        stub: Which image, by its results path stub.

    Returns:
        The root element of the image's data label.
    """
    env = make_cohort_bundle_env(cohort, tmp_path)
    label_cohort_images(env, [stub])
    (label,) = (env.bundle_dir / 'data').rglob('*_backplanes.lblx')
    return ElementTree.parse(label).getroot()


def _stated(element: ElementTree.Element) -> list[tuple[str, str | None, str | None]]:
    """Return what each child of an element states: its name, its unit and its value.

    Parameters:
        element: The element whose children are read.

    Returns:
        Each child's local name, ``unit`` attribute and stripped text, in order.
    """
    return [
        (child.tag.rsplit('}', 1)[-1], child.get('unit'), (child.text or '').strip())
        for child in element
    ]


def _written(statistic: dict[str, Any], end: str) -> str:
    """Return one end of a ring statistic as the global index tables write it.

    Parameters:
        statistic: The statistic, as the backplane metadata records it.
        end: ``min`` or ``max``.

    Returns:
        The value, in the format its unit takes.
    """
    return INDEX_VALUE_FORMATS[statistic['units']].render(statistic[end])


@pytest.mark.parametrize(
    ('stub', 'count'), [(LIMB_STUB, 0), (RINGS_STUB, 1)], ids=['limb image', 'ring image']
)
def test_only_the_ring_image_s_data_label_states_the_ring_geometry(
    cassini_cohort: Cohort, tmp_path: Path, stub: str, count: int
) -> None:
    """The image with ring backplanes states the ring geometry, and the other none."""
    root = _labelled(cassini_cohort, tmp_path, stub)
    assert len(root.findall(RING_SYSTEMS, NAMESPACES)) == count


def test_the_ring_geometry_states_each_ring_statistic_and_the_incidence_angle(
    cassini_cohort: Cohort, tmp_path: Path
) -> None:
    """Each statistic in its attribute, and the incidence angle's mean and range, in order.

    The values are the image's backplane metadata's, written as the global index tables
    write them, each stated in the unit its attribute takes: a resolution per pixel in the
    length or the angle a pixel spans.  The description and the three values every
    image's geometry states lead.
    """
    metadata_path = cassini_cohort.backplane_results_root / f'{RINGS_STUB}_backplane_metadata.json'
    rings = json.loads(metadata_path.read_text(encoding='utf-8'))['rings']
    statistics = rings['backplanes']
    incidence = rings['incidence_angle']
    geometry = _labelled(cassini_cohort, tmp_path, RINGS_STUB).find(GEOMETRY, NAMESPACES)
    assert geometry is not None
    stated = _stated(geometry)

    def ends(
        plane: str, attribute: str, unit: str, keys: tuple[str, str] = ('min', 'max')
    ) -> list[tuple[str, str, str]]:
        """Return the two attributes one plane's statistic is stated as.

        Parameters:
            plane: The plane's configured name.
            attribute: The attribute, less its ``minimum_`` or ``maximum_``.
            unit: The unit the label states it in.
            keys: The statistic's keys for the minimum and the maximum.

        Returns:
            Its minimum and its maximum.
        """
        minimum, maximum = keys
        return [
            (f'minimum_{attribute}', unit, _written(statistics[plane], minimum)),
            (f'maximum_{attribute}', unit, _written(statistics[plane], maximum)),
        ]

    assert stated[4:-1] == [
        *ends('ring_phase_angle', 'phase_angle', 'deg'),
        ('mean_incidence_angle', 'deg', _written(incidence, 'mean')),
        ('minimum_incidence_angle', 'deg', _written(incidence, 'min')),
        ('maximum_incidence_angle', 'deg', _written(incidence, 'max')),
        *ends('ring_emission_angle', 'emission_angle', 'deg'),
        *ends('ring_longitude', 'inertial_ring_longitude', 'deg', ('wrapped_min', 'wrapped_max')),
        *ends('ring_radius', 'ring_radius', 'km'),
    ]
    grid = geometry.find('rings:Reprojection_Grid_Parameters', NAMESPACES)
    assert grid is not None
    assert _stated(grid) == [
        *ends('ring_radial_resolution', 'radial_resolution', 'km'),
        *ends('ring_longitudinal_resolution', 'longitudinal_resolution', 'deg'),
    ]


def test_the_ring_geometry_is_in_the_plane_of_saturn_s_equator_at_the_image_s_midtime(
    cassini_cohort: Cohort, tmp_path: Path
) -> None:
    """The geometry is no co-rotating frame's, in the equator's plane, its epoch the midtime.

    The basis epoch is the midpoint of the start and the stop the label states, as the
    label's midtime is written.
    """
    root = _labelled(cassini_cohort, tmp_path, RINGS_STUB)
    time_coordinates = 'pds:Observation_Area/pds:Time_Coordinates'
    start = root.findtext(f'{time_coordinates}/pds:start_date_time', namespaces=NAMESPACES)
    stop = root.findtext(f'{time_coordinates}/pds:stop_date_time', namespaces=NAMESPACES)
    assert start is not None
    assert stop is not None
    geometry = root.find(GEOMETRY, NAMESPACES)
    assert geometry is not None
    assert _stated(geometry)[1:4] == [
        ('epoch_reprojection_basis_utc', None, pds4_utc_midpoint(start, stop)),
        ('reprojection_plane', None, 'Equator'),
        ('corotating_flag', None, 'N'),
    ]
