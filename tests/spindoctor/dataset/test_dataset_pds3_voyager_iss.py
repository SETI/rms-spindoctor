import pytest
from tests.config import REQUIRES_PDS3_HOLDINGS

import spindoctor.dataset.dataset_pds3_voyager_iss as dsvgiss

pytestmark = REQUIRES_PDS3_HOLDINGS


@pytest.fixture
def ds_voyager_iss() -> dsvgiss.DataSetPDS3VoyagerISS:
    """Voyager ISS dataset fixture for testing."""
    return dsvgiss.DataSetPDS3VoyagerISS()


def test_voyager_iss_yield_basic(ds_voyager_iss: dsvgiss.DataSetPDS3VoyagerISS) -> None:
    ret = list(ds_voyager_iss.yield_image_files_index(max_filenames=2))
    assert len(ret) == 2
    assert (
        ret[0]
        .image_files[0]
        .label_file_url.as_posix()
        .endswith('volumes/VGISS_5xxx/VGISS_5101/DATA/C13854XX/C1385455_GEOMED.LBL')
    )
    assert (
        ret[1]
        .image_files[0]
        .label_file_url.as_posix()
        .endswith('volumes/VGISS_5xxx/VGISS_5101/DATA/C13894XX/C1389407_GEOMED.LBL')
    )


def test_voyager_iss_yield_vol_start(
    ds_voyager_iss: dsvgiss.DataSetPDS3VoyagerISS,
) -> None:
    ret = list(ds_voyager_iss.yield_image_files_index(max_filenames=1, vol_start='VGISS_8201'))
    assert len(ret) == 1
    assert (
        ret[0]
        .image_files[0]
        .label_file_url.as_posix()
        .endswith('volumes/VGISS_8xxx/VGISS_8201/DATA/C08966XX/C0896631_GEOMED.LBL')
    )


def test_voyager_iss_yield_vol_end(
    ds_voyager_iss: dsvgiss.DataSetPDS3VoyagerISS,
) -> None:
    ret = list(ds_voyager_iss.yield_image_files_index(vol_end='VGISS_5102'))
    assert len(ret) == 928
    assert (
        ret[-1]
        .image_files[0]
        .label_file_url.as_posix()
        .endswith('volumes/VGISS_5xxx/VGISS_5102/DATA/C15199XX/C1519931_GEOMED.LBL')
    )


def test_voyager_iss_yield_img_start_num(
    ds_voyager_iss: dsvgiss.DataSetPDS3VoyagerISS,
) -> None:
    ret = list(ds_voyager_iss.yield_image_files_index(max_filenames=2, img_start_num=1469548))
    assert len(ret) == 2
    assert ret[0].image_files[0].label_file_url.as_posix().endswith('C1469548_GEOMED.LBL')
    assert ret[1].image_files[0].label_file_url.as_posix().endswith('C1469550_GEOMED.LBL')


def test_voyager_iss_yield_img_end_num(
    ds_voyager_iss: dsvgiss.DataSetPDS3VoyagerISS,
) -> None:
    # FDS counts are not monotonic across volumes: VG2's Neptune counts (VGISS_8xxx)
    # roll over below VG1's Jupiter counts, so a number range alone does not bound
    # the scan for Voyager. This range matches 117 VG1 Jupiter frames in VGISS_5101
    # and every one of VGISS_8201's 792 Neptune frames. Two volumes keep the index
    # download small; every volume's index together is about 360 MB (#820).
    ret = list(
        ds_voyager_iss.yield_image_files_index(
            img_end_num=1469550, volumes=['VGISS_5101', 'VGISS_8201']
        )
    )
    assert len(ret) == 909
    specs = [g.image_files[0].label_file_url.as_posix() for g in ret]
    vg1_jupiter = [s for s in specs if '/VGISS_5xxx/' in s]
    assert len(vg1_jupiter) == 117
    assert vg1_jupiter[-2].endswith(
        'volumes/VGISS_5xxx/VGISS_5101/DATA/C14695XX/C1469548_GEOMED.LBL'
    )
    assert vg1_jupiter[-1].endswith(
        'volumes/VGISS_5xxx/VGISS_5101/DATA/C14695XX/C1469550_GEOMED.LBL'
    )
    vg2_neptune = [s for s in specs if '/VGISS_8xxx/' in s]
    assert len(vg2_neptune) == 792


def test_voyager_iss_yield_volumes(
    ds_voyager_iss: dsvgiss.DataSetPDS3VoyagerISS,
) -> None:
    ret = list(ds_voyager_iss.yield_image_files_index(volumes=['VGISS_5101', 'VGISS_8201']))
    assert len(ret) == 1403
    ret2 = [x.image_files[0].label_file_url.as_posix() for x in ret]
    ret3 = [x for x in ret2 if 'VGISS_5101' not in x and 'VGISS_8201' not in x]
    assert len(ret3) == 0


# def test_voyager_iss_camera():  # TODO: Figure this out
#     ret = ds_voyager_iss.yield_filenames_index(max_filenames=1, volumes=['COISS_1001'],
#                                           camera='W',
#     ret = [x[0].as_posix() for x in ret]
#     assert len(ret) == 1
#     assert ret[0].endswith('W1294561143_1.LBL')
#     ret = ds_voyager_iss.yield_filenames_index(max_filenames=1, volumes=['COISS_1001'],
#                                           camera='N')
#     ret = [x[0].as_posix() for x in ret]
#     assert len(ret) == 1
#     assert ret[0].endswith('N1294562651_1.LBL')
