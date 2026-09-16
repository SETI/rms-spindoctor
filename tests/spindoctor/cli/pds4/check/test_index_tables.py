"""The index records a supplemental file calls for, over a file written here.

A supplemental file the check reads holds a JSON object, whose ``backplanes`` holds a
``bodies`` object and a ``rings`` object.  One holding anything else on that path is
reported, and the check goes on, rather than ending the whole check with a traceback.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from spindoctor.cli.pds4.check.findings import CheckName, Finding
from spindoctor.cli.pds4.check.index_tables import INDEX_LABELS, index_row_findings

from .controls import parse

LID = 'urn:nasa:pds:fake_bundle:data:image'
"""The logical identifier the label of the product declares."""

SUPPLEMENTAL = 'image_backplane_metadata.json'
"""The supplemental file the label names, beside it in the bundle's data directory."""

LABEL = f"""<Product_Observational xmlns="http://pds.nasa.gov/pds4/pds/v1">
  <Identification_Area>
    <logical_identifier>{LID}</logical_identifier>
  </Identification_Area>
  <File_Area_Observational_Supplemental>
    <File>
      <file_name>{SUPPLEMENTAL}</file_name>
    </File>
  </File_Area_Observational_Supplemental>
</Product_Observational>
"""
"""A data label naming its supplemental file, which the records are read through."""

WELL_FORMED = {
    'backplanes': {
        'bodies': {'SATURN': {'backplanes': {'minimum_body_longitude': 0.0}}},
        'rings': {'backplanes': {'minimum_ring_radius': 0.0}},
    }
}
"""What the backplane stage writes: one body with geometry, and ring statistics."""


def _findings(bundle_dir: Path, metadata: Any) -> list[Finding]:
    """Read the index records a supplemental file holding the given metadata calls for.

    Parameters:
        bundle_dir: The bundle's directory, holding the one label and its file.
        metadata: What the supplemental file holds, written to it as JSON.

    Returns:
        The findings of holding the tree's index tables to its supplemental files.
    """
    label = bundle_dir / 'data' / 'image.lblx'
    label.parent.mkdir()
    label.write_text(LABEL, encoding='utf-8')
    (label.parent / SUPPLEMENTAL).write_text(json.dumps(metadata), encoding='utf-8')
    return index_row_findings(bundle_dir, {'data/image.lblx': parse(label)}, {})


@pytest.mark.parametrize(
    ('metadata', 'held'),
    [
        pytest.param([1, 2], 'an array', id='top'),
        pytest.param({'backplanes': [1, 2]}, 'an array at backplanes', id='backplanes'),
        pytest.param(
            {'backplanes': {'bodies': [1, 2]}}, 'an array at backplanes.bodies', id='bodies'
        ),
        pytest.param(
            {'backplanes': {'bodies': {'SATURN': 7}}},
            'a number at backplanes.bodies.SATURN',
            id='body',
        ),
        pytest.param({'backplanes': {'rings': 'yes'}}, 'a string at backplanes.rings', id='rings'),
    ],
)
def test_a_supplemental_file_holding_no_json_object_is_one_finding(
    tmp_path: Path, metadata: Any, held: str
) -> None:
    """A place the records are read from holding anything else is reported, not raised.

    Parameters:
        tmp_path: The bundle's directory.
        metadata: What the supplemental file holds.
        held: What the finding says the file holds, and where.
    """
    expected = Finding(
        f'data/{SUPPLEMENTAL}',
        CheckName.INTEGRITY,
        '',
        f'holds {held}, not a JSON object, so the index records it calls for are not known',
    )
    assert _findings(tmp_path, metadata) == [expected]


def test_a_well_formed_supplemental_file_calls_for_one_record_of_each_table(
    tmp_path: Path,
) -> None:
    """A body with geometry and ring statistics call for one record of each table."""
    expected = [
        f'the tree holds no {file}, but its supplemental files call for 1 record(s) of {name}'
        for file, name in INDEX_LABELS.items()
    ]
    assert [finding.message for finding in _findings(tmp_path, WELL_FORMED)] == expected
