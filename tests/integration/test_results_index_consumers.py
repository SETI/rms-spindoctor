"""Backplanes and reprojections built from an index against ones built from files.

The acceptance evidence for the index-backed readers: on real library frames,
navigated for real and ingested by the real ingest, the products a consumer
writes are identical whichever storage its navigation records came from.  The
comparison is on the written product rather than on the selection, because the
selection is only interesting insofar as it decides the geometry, and a
disagreement anywhere between the row and the pointed observation shows up here.

Identity is asserted exactly.  The two runs pass the same recorded values to the
same code, so nothing between them is allowed to be approximately equal: a
recorded attitude that survived storage to fifteen places and not sixteen is the
failure this is looking for, and a tolerance would hide it.

Both a frame whose record carries a corrected attitude and one whose navigation
fitted a camera rotation are run, since the two take different mechanisms and
the second is the case that distinguishes a missing pointing block from a
missing corrected attitude.
"""

import math
import os
import uuid
from pathlib import Path
from typing import Any, cast

import numpy as np
import pdslogger
import pytest
import sqlalchemy

pytestmark = pytest.mark.integration

_RESOURCES = os.environ.get('OOPS_RESOURCES', '')
_SPICE_ROOT = Path(_RESOURCES) / 'SPICE'

if (
    len(_RESOURCES) == 0
    or not (_SPICE_ROOT / 'Cassini' / 'CK-reconstructed').is_dir()
    or 'PDS3_HOLDINGS_DIR' not in os.environ
):
    pytest.skip(
        'the index-backed reader comparison needs local binary kernels and the holdings; set '
        'OOPS_RESOURCES to a local SPICE tree and PDS3_HOLDINGS_DIR to the holdings',
        allow_module_level=True,
    )

from astropy.io import fits  # noqa: E402  (guarded import)
from filecache import FCPath  # noqa: E402  (guarded import)

from spindoctor.cli.backplanes.backplanes import (  # noqa: E402  (guarded import)
    generate_backplanes_image_files,
)
from spindoctor.cli.reproj.offsets import (  # noqa: E402  (guarded import)
    PointingMechanism,
    apply_pointing_to_obs,
)
from spindoctor.cli.reproj.pointing_source import (  # noqa: E402  (guarded import)
    FilePointingSource,
    IndexPointingSource,
    PointingSource,
)
from spindoctor.cli.results_index import ingest_metadata_files  # noqa: E402  (guarded import)
from spindoctor.dataset.dataset import ImageFile, ImageFiles  # noqa: E402  (guarded import)
from spindoctor.nav_records import TreeTuning  # noqa: E402  (guarded import)
from spindoctor.navigate_image_files import (  # noqa: E402  (guarded import)
    navigate_image_files,
)
from spindoctor.obs import (  # noqa: E402  (guarded import)
    ObsCassiniISS,
    ObsGalileoSSI,
    ObsSnapshotInst,
)
from spindoctor.reproj.rings import (  # noqa: E402  (guarded import)
    RingMosaic,
    RingMosaicData,
    RingReprojResult,
)
from spindoctor.results_index import (  # noqa: E402  (guarded import)
    IMAGES,
    normalize_root_url,
    open_index,
)
from tests.integration.ck_round_trip import (  # noqa: E402  (guarded import)
    holdings_url,
    sidecar_for,
)

# A Cassini NAC frame of the main rings, navigated by the ring-edge technique:
# it carries a recorded corrected attitude and has ring content to reproject.
_RING_FRAME = 'N1863267861_1_CALIB'

# A Galileo SSI frame the library records as navigating successfully.  Galileo
# fits a camera rotation, so its record carries a baseline and no corrected
# attitude, which is the row shape one column alone cannot tell from a record
# with no pointing block at all.
_GALILEO_FRAME = 'C0059894800R'

_MISSION_TO_OBS_CLASS: dict[str, type[ObsSnapshotInst]] = {
    'COISS': ObsCassiniISS,
    'GOSSI': ObsGalileoSSI,
}

# The reprojection window: the main rings at a resolution coarse enough to run
# quickly and fine enough that a pointing difference of a pixel would move
# brightness between bins.
_RING_RADIUS_INNER_KM = 74658.0
_RING_RADIUS_OUTER_KM = 136780.0
_RING_RADIUS_RESOLUTION_KM = 25.0
_RING_LONGITUDE_RESOLUTION_RAD = 0.1 * math.pi / 180.0


def _image_files_for(image_id: str) -> tuple[type[ObsSnapshotInst], ImageFiles]:
    """Resolve one library image into its observation class and file batch.

    Parameters:
        image_id: The library's id for the image.

    Returns:
        The instrument's observation class and the one-image batch.
    """
    sidecar = sidecar_for(image_id)
    obs_class = _MISSION_TO_OBS_CLASS[sidecar.mission]
    url = holdings_url(sidecar)
    return obs_class, ImageFiles(
        image_files=[ImageFile(image_file_url=url, label_file_url=url, results_path_stub=image_id)]
    )


@pytest.fixture(scope='module')
def navigated() -> dict[str, Path]:
    """Hold the results root each frame was navigated into, so it navigates once.

    Returns:
        The cache, empty to begin with.
    """
    # Module-scoped; --dist=loadfile keeps the whole file on one worker, so the
    # cache is built once there and any other distribution only repeats work.
    return {}


def _nav_root(
    image_id: str, cache: dict[str, Path], tmp_path_factory: pytest.TempPathFactory
) -> Path:
    """Navigate one frame into a results root of its own, or return the one it has.

    Parameters:
        image_id: The library's id for the image.
        cache: The roots of the frames already navigated.
        tmp_path_factory: Where the navigation may write.

    Returns:
        The results root holding that frame's navigation document.
    """
    if image_id not in cache:
        obs_class, image_files = _image_files_for(image_id)
        root = tmp_path_factory.mktemp(f'nav_{image_id}')
        success, metadata = navigate_image_files(
            obs_class, image_files, FCPath(str(root)), write_output_files=True
        )
        assert success, f'{image_id} did not navigate; the comparison has nothing to read'
        assert metadata.get('status') == 'success'
        cache[image_id] = root
    return cache[image_id]


def _quiet_logger() -> pdslogger.PdsLogger:
    """Return a logger that keeps ingest chatter out of the test output.

    Returns:
        A logger of its own, named uniquely for the life of the process.
    """
    logger = pdslogger.PdsLogger(f'index_consumers_{uuid.uuid4().hex}')
    logger.set_level('ERROR')
    return logger


def _index_source(root: Path, database: Path) -> IndexPointingSource:
    """Ingest a results root and return an index-backed source over it.

    Parameters:
        root: The results root to walk.
        database: Path of the SQLite index to create.

    Returns:
        The source, which the caller closes.
    """
    url = f'sqlite:///{database.as_posix()}'
    engine = open_index(url, create=True)
    try:
        ingest_metadata_files(
            engine, [root.as_posix()], logger=_quiet_logger(), tuning=TreeTuning()
        )
    finally:
        engine.dispose()
    return IndexPointingSource(open_index(url), normalize_root_url(root))


def _backplane_planes(
    image_id: str, source: PointingSource, work: Path
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Run the backplane stage end to end through one source.

    Parameters:
        image_id: The library's id for the image.
        source: Where the stage reads the navigation record from.
        work: A fresh directory for the backplane root.

    Returns:
        The stage's result and the product's planes by HDU name.
    """
    obs_class, image_files = _image_files_for(image_id)
    backplane_root = FCPath(str(work)) / 'bp'
    result = generate_backplanes_image_files(
        obs_class,
        image_files,
        pointing_source=source,
        backplane_results_root=backplane_root,
        write_output_files=True,
    )
    planes: dict[str, np.ndarray] = {}
    product = backplane_root / f'{image_id}_backplanes.fits'
    with fits.open(cast(Path, product.retrieve())) as hdul:
        for hdu in hdul[1:]:
            planes[str(hdu.name)] = np.asarray(hdu.data, np.float64)
    return result, planes


@pytest.fixture(scope='module')
def ring_backplanes(
    navigated: dict[str, Path], tmp_path_factory: pytest.TempPathFactory
) -> dict[str, Any]:
    """Build the ring frame's backplanes through both sources.

    Parameters:
        navigated: The cache of navigated results roots.
        tmp_path_factory: Where the runs may write.

    Returns:
        The two results and the two sets of planes.
    """
    root = _nav_root(_RING_FRAME, navigated, tmp_path_factory)
    work = tmp_path_factory.mktemp('bp_ring')
    source = _index_source(root, work / 'index.sqlite3')
    try:
        file_result, file_planes = _backplane_planes(
            _RING_FRAME, FilePointingSource(FCPath(str(root))), work / 'files'
        )
        index_result, index_planes = _backplane_planes(_RING_FRAME, source, work / 'index')
    finally:
        source.close()
    return {
        'file_result': file_result,
        'index_result': index_result,
        'file_planes': file_planes,
        'index_planes': index_planes,
    }


@pytest.fixture(scope='module')
def galileo_selections(
    navigated: dict[str, Path], tmp_path_factory: pytest.TempPathFactory
) -> dict[str, Any]:
    """Classify the fitted-rotation frame's real record through both sources.

    The frame is an Earth-system image, for which no ring backplane exists, so
    what is compared is the classification and the values a product would be
    built from rather than a product.  What it contributes is the row shape:
    a real record from a host whose navigation fits a camera rotation, stored
    by the real ingest, is the case a NULL ``cmatrix`` alone cannot tell from a
    record that carried no pointing block.

    Parameters:
        navigated: The cache of navigated results roots.
        tmp_path_factory: Where the index is written.

    Returns:
        The two selections and the row the index holds.
    """
    root = _nav_root(_GALILEO_FRAME, navigated, tmp_path_factory)
    work = tmp_path_factory.mktemp('sel_galileo')
    _obs_class, image_files = _image_files_for(_GALILEO_FRAME)
    image_file = image_files.image_files[0]
    database = work / 'index.sqlite3'
    source = _index_source(root, database)
    try:
        selections = {
            'file': FilePointingSource(FCPath(str(root))).load_pointing(image_file),
            'index': source.load_pointing(image_file),
        }
    finally:
        source.close()
    engine = open_index(f'sqlite:///{database.as_posix()}')
    try:
        with engine.connect() as connection:
            row = connection.execute(
                sqlalchemy.select(IMAGES.c.cmatrix, IMAGES.c.cmatrix_original).where(
                    IMAGES.c.root_url == normalize_root_url(root),
                    IMAGES.c.results_path_stub == _GALILEO_FRAME,
                )
            ).one()
    finally:
        engine.dispose()
    return {'file': selections['file'], 'index': selections['index'], 'row': row}


def _identical(first: dict[str, np.ndarray], second: dict[str, np.ndarray]) -> list[str]:
    """Return the names of the planes the two products do not agree on exactly.

    Parameters:
        first: One product's planes by HDU name.
        second: The other product's planes by HDU name.

    Returns:
        The names that differ, treating NaN as equal to NaN: a backplane leaves
        an unclaimed pixel NaN, and two products that both leave it so agree.
    """
    return sorted(
        name
        for name in set(first) | set(second)
        if name not in first
        or name not in second
        or not np.array_equal(first[name], second[name], equal_nan=True)
    )


def test_the_ring_frame_takes_the_cmatrix_through_the_index(
    ring_backplanes: dict[str, Any],
) -> None:
    """The index-backed run applies the recorded corrected attitude."""
    assert ring_backplanes['index_result']['pointing_source'] == 'cmatrix'


def test_the_ring_frame_takes_the_same_mechanism_from_files(
    ring_backplanes: dict[str, Any],
) -> None:
    """As the file-backed run does, which is what makes the products comparable."""
    assert ring_backplanes['file_result']['pointing_source'] == 'cmatrix'


def test_the_ring_backplanes_are_identical(ring_backplanes: dict[str, Any]) -> None:
    """Every plane of the product agrees exactly, storage notwithstanding."""
    assert _identical(ring_backplanes['file_planes'], ring_backplanes['index_planes']) == []


def test_the_ring_backplanes_are_not_empty(ring_backplanes: dict[str, Any]) -> None:
    """A guard on the comparison above, which two empty products would satisfy."""
    assert len(ring_backplanes['file_planes']) > 0


def test_a_real_fitted_rotation_record_stores_no_corrected_attitude(
    galileo_selections: dict[str, Any],
) -> None:
    """The row that shape produces, from a real navigation of a real frame."""
    assert galileo_selections['row'].cmatrix is None


def test_a_real_fitted_rotation_record_stores_its_baseline(
    galileo_selections: dict[str, Any],
) -> None:
    """Which is what separates it from a record that had no pointing block.

    Both leave ``cmatrix`` NULL; only this one leaves the baseline behind.
    """
    assert galileo_selections['row'].cmatrix_original is not None


def test_the_fitted_rotation_frame_falls_back_to_the_offset_through_the_index(
    galileo_selections: dict[str, Any],
) -> None:
    """A record with a baseline and no corrected attitude takes the offset path."""
    assert galileo_selections['index'].mechanism is PointingMechanism.OFFSET


def test_the_fitted_rotation_frame_takes_the_same_mechanism_from_files(
    galileo_selections: dict[str, Any],
) -> None:
    """As reading its document does."""
    assert galileo_selections['file'].mechanism is PointingMechanism.OFFSET


def test_the_fitted_rotation_frame_names_the_same_reason_either_way(
    galileo_selections: dict[str, Any],
) -> None:
    """And is reported under the reason that says why, not a bare absence."""
    assert galileo_selections['index'].reason == galileo_selections['file'].reason


def test_the_fitted_rotation_reason_names_the_fitted_rotation(
    galileo_selections: dict[str, Any],
) -> None:
    """A guard on the comparison above, which two wrong reasons would satisfy."""
    assert galileo_selections['index'].reason == 'no_cmatrix_rotation_fitted'


def test_the_fitted_rotation_frame_carries_the_same_offset_either_way(
    galileo_selections: dict[str, Any],
) -> None:
    """The value a product would be built from survives storage exactly."""
    assert galileo_selections['index'].offset == galileo_selections['file'].offset


def test_the_fitted_rotation_frame_has_an_offset_to_carry(
    galileo_selections: dict[str, Any],
) -> None:
    """A guard on the comparison above, which two absent offsets would satisfy."""
    assert galileo_selections['file'].offset is not None


def _reprojected(image_id: str, source: PointingSource) -> RingReprojResult:
    """Reproject one frame's rings through one source's pointing.

    Parameters:
        image_id: The library's id for the image.
        source: Where the navigation record is read from.

    Returns:
        The reprojection result.
    """
    obs_class, image_files = _image_files_for(image_id)
    image_file = image_files.image_files[0]
    obs = obs_class.from_file(image_file.image_file_path.absolute(), extfov_margin_vu=(0, 0))
    assert isinstance(obs, ObsSnapshotInst)
    selection = source.load_pointing(image_file)
    apply_pointing_to_obs(obs, selection, subject=image_id)
    mosaic = RingMosaic(
        'SATURN',
        _RING_RADIUS_INNER_KM,
        _RING_RADIUS_OUTER_KM,
        longitude_resolution=_RING_LONGITUDE_RESOLUTION_RAD,
        radius_resolution=_RING_RADIUS_RESOLUTION_KM,
    )
    return mosaic.reproject(obs, image_name=image_id)


@pytest.fixture(scope='module')
def ring_reprojections(
    navigated: dict[str, Path], tmp_path_factory: pytest.TempPathFactory
) -> dict[str, Any]:
    """Reproject the ring frame through both sources.

    Parameters:
        navigated: The cache of navigated results roots.
        tmp_path_factory: Where the index is written.

    Returns:
        The two reprojection results.
    """
    root = _nav_root(_RING_FRAME, navigated, tmp_path_factory)
    work = tmp_path_factory.mktemp('reproj_ring')
    source = _index_source(root, work / 'index.sqlite3')
    try:
        return {
            'file': _reprojected(_RING_FRAME, FilePointingSource(FCPath(str(root)))),
            'index': _reprojected(_RING_FRAME, source),
        }
    finally:
        source.close()


def test_the_ring_reprojections_populate_the_same_longitudes(
    ring_reprojections: dict[str, Any],
) -> None:
    """The two runs put the ring plane into the same longitude columns.

    A pointing that differed by a fraction of a pixel would move the ring's
    edges between columns, so this is the coarsest thing the comparison can
    catch and the first thing it would.
    """
    assert np.array_equal(
        ring_reprojections['file'].longitude_antimask,
        ring_reprojections['index'].longitude_antimask,
    )


def test_the_ring_reprojections_claim_the_same_cells(
    ring_reprojections: dict[str, Any],
) -> None:
    """And leave the same cells unclaimed inside those columns."""
    assert np.array_equal(
        np.ma.getmaskarray(ring_reprojections['file'].img),
        np.ma.getmaskarray(ring_reprojections['index'].img),
    )


def test_the_ring_reprojections_hold_the_same_brightness(
    ring_reprojections: dict[str, Any],
) -> None:
    """And the same brightness in every cell they claim."""
    assert np.array_equal(
        ring_reprojections['file'].img.filled(np.nan),
        ring_reprojections['index'].img.filled(np.nan),
        equal_nan=True,
    )


def test_the_ring_reprojection_populated_something(
    ring_reprojections: dict[str, Any],
) -> None:
    """A guard on the comparisons above, which two empty grids would satisfy."""
    assert int(np.count_nonzero(~np.ma.getmaskarray(ring_reprojections['file'].img))) > 0


def _mosaicked(reprojection: RingReprojResult) -> RingMosaic:
    """Accumulate one reprojection into a mosaic of its own.

    A mosaic is what a run of ``sd_mosaic`` writes, and it is an accumulation
    of reprojections rather than a second computation, so the comparison is
    made on the assembled product as well as on what went into it.

    Parameters:
        reprojection: The reprojection to accumulate.

    Returns:
        The mosaic holding it.
    """
    mosaic = RingMosaic(
        'SATURN',
        _RING_RADIUS_INNER_KM,
        _RING_RADIUS_OUTER_KM,
        longitude_resolution=_RING_LONGITUDE_RESOLUTION_RAD,
        radius_resolution=_RING_RADIUS_RESOLUTION_KM,
    )
    mosaic.add(reprojection)
    return mosaic


@pytest.fixture(scope='module')
def ring_mosaics(ring_reprojections: dict[str, Any]) -> dict[str, RingMosaicData]:
    """Assemble a mosaic from each source's reprojection, as a run would save it.

    Parameters:
        ring_reprojections: The two reprojections.

    Returns:
        The two mosaics, in the form ``sd_mosaic`` writes.
    """
    return {mode: _mosaicked(ring_reprojections[mode]).to_sparse() for mode in ('file', 'index')}


def test_the_mosaics_cover_the_same_longitudes(ring_mosaics: dict[str, RingMosaicData]) -> None:
    """The assembled products claim the same longitude columns."""
    assert np.array_equal(
        ring_mosaics['file'].longitude_antimask, ring_mosaics['index'].longitude_antimask
    )


def test_the_mosaics_hold_the_same_brightness(ring_mosaics: dict[str, RingMosaicData]) -> None:
    """And the same value in every cell of them."""
    assert np.array_equal(
        ring_mosaics['file'].img.filled(np.nan),
        ring_mosaics['index'].img.filled(np.nan),
        equal_nan=True,
    )


def test_the_mosaics_claim_the_same_cells(ring_mosaics: dict[str, RingMosaicData]) -> None:
    """And leave the same ones unclaimed."""
    assert np.array_equal(
        np.ma.getmaskarray(ring_mosaics['file'].img),
        np.ma.getmaskarray(ring_mosaics['index'].img),
    )


def test_the_mosaics_name_the_same_contributing_images(
    ring_mosaics: dict[str, RingMosaicData],
) -> None:
    """And record the same image behind the product."""
    assert (
        ring_mosaics['file'].contributing_image_names
        == ring_mosaics['index'].contributing_image_names
    )


def test_the_mosaics_are_not_empty(ring_mosaics: dict[str, RingMosaicData]) -> None:
    """A guard on the comparisons above, which two empty mosaics satisfy."""
    assert int(np.count_nonzero(~np.ma.getmaskarray(ring_mosaics['file'].img))) > 0
