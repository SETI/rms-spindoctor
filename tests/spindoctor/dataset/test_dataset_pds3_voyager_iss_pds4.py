"""Hermetic tests of the Voyager ISS dataset's PDS4 hooks, which it leaves unimplemented.

The dataset is built on an empty holdings root under ``tmp_path``, so these run
without ``PDS3_HOLDINGS_DIR``, which every test in ``test_dataset_pds3_voyager_iss.py``
requires.
"""

from pathlib import Path

import pytest
from tests.spindoctor.cli.pds4.conftest import make_image_file

from spindoctor.dataset.dataset_pds3_voyager_iss import DataSetPDS3VoyagerISS


def test_voyager_pds4_hooks_not_implemented(tmp_path: Path) -> None:
    """The Voyager dataset's per-image PDS4 hooks are NotImplementedError walls.

    The base-class walls raise a bare NotImplementedError, so the assertions
    pin the empty message: a messaged NotImplementedError escaping from deeper
    code would fail them.
    """
    dataset = DataSetPDS3VoyagerISS(tmp_path / 'holdings')
    image_file = make_image_file('C1234567')
    with pytest.raises(NotImplementedError) as stub_exc:
        dataset.pds4_path_stub(image_file)
    assert str(stub_exc.value) == ''
    with pytest.raises(NotImplementedError) as lidvid_exc:
        dataset.pds4_image_name_to_data_lidvid('C1234567')
    assert str(lidvid_exc.value) == ''
