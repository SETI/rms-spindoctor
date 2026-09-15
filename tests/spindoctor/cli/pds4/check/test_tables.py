"""The table reader over a label written here: a file area it leaves alone."""

from pathlib import Path

from spindoctor.cli.pds4.check.tables import table_findings

from .controls import parse

MIXED_LABEL = """<Product_Ancillary xmlns="http://pds.nasa.gov/pds4/pds/v1">
  <File_Area_Ancillary>
    <File>
      <file_name>table.dat</file_name>
    </File>
    <Table_Character>
      <offset unit="byte">0</offset>
      <records>1</records>
      <record_delimiter>Line-Feed</record_delimiter>
      <Record_Character>
        <fields>1</fields>
        <groups>0</groups>
        <record_length unit="byte">4</record_length>
        <Field_Character>
          <name>count</name>
          <field_number>1</field_number>
          <field_location unit="byte">1</field_location>
          <data_type>ASCII_Integer</data_type>
          <field_length unit="byte">3</field_length>
        </Field_Character>
      </Record_Character>
    </Table_Character>
    <Array_1D>
      <offset unit="byte">8</offset>
    </Array_1D>
  </File_Area_Ancillary>
</Product_Ancillary>
"""
"""A label whose file area holds a table of one 4-byte record and, at byte 8, an array."""


def test_a_file_area_holding_an_object_of_another_class_is_left_alone(tmp_path: Path) -> None:
    """A table beside an array is not read, since the array's extent is not known.

    A reader that took the table to run to the array's offset would find 8 bytes where the
    label places 4.
    """
    (tmp_path / 'table.dat').write_bytes(b'123\n' + bytes(8))
    label = tmp_path / 'label.lblx'
    label.write_text(MIXED_LABEL, encoding='ascii')
    assert table_findings('label.lblx', label, parse(label), None) == []
