"""The C-matrix readers against the offset path, on real library frames.

This is the acceptance evidence for the reading half of the corrected-pointing
work: on real frames of all four kernel-eligible instruments plus the
body-navigated WAC frame, the geometry built on the recorded ``cmatrix`` (a
frame replacement) agrees with the geometry built on the recorded ``offset``
(an ``OffsetFOV``) to within the derived rotation-versus-shift bound -- and
the comparison runs not only over a raw line-of-sight grid but through the
consumers the readers actually serve: one ring reprojection, one body
reprojection, and one end-to-end backplane run.

The agreement bound is measured in pixel space, per frame, in the test's own
metric: ``K_inst`` is the worst pixel-space residual between an exact rigid
rotation and the uniform pixel shift it approximates, over a 17x17 grid and
eight offset directions at 50 px of total displacement on the frame's own
FOV; the expected residual at the frame's navigated offset is then
``B = K_inst * |offset| / 50`` (the residual is linear in the offset), and
the derived tolerance is ``2 B + 0.005 px`` -- the factor of two covering the
navigated direction falling between the swept directions and grid placement,
the floor covering numerical noise and the distorted-FOV zero-offset
structure.  Each residual is additionally pinned at measured-plus-margin: a
residual above a pin is a defect to diagnose, never a tolerance to raise.

Every navigation here runs in-process (nothing furnishes kernels between
runs, unlike the round trip), and the per-frame work is cached module-wide;
run this file under ``--dist=loadfile`` so the cache is built once.
"""

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

pytestmark = pytest.mark.integration

_RESOURCES = os.environ.get('OOPS_RESOURCES', '')
_SPICE_ROOT = Path(_RESOURCES) / 'SPICE'

if (
    len(_RESOURCES) == 0
    or not (_SPICE_ROOT / 'Cassini' / 'CK-reconstructed').is_dir()
    or 'PDS3_HOLDINGS_DIR' not in os.environ
):
    pytest.skip(
        'the reader comparison needs local binary kernels and the holdings; set OOPS_RESOURCES '
        'to a local SPICE tree and PDS3_HOLDINGS_DIR to the holdings',
        allow_module_level=True,
    )

import oops  # noqa: E402  (guarded import)
from astropy.io import fits  # noqa: E402  (guarded import)
from filecache import FCPath  # noqa: E402  (guarded import)

from spindoctor.cli.backplanes.backplanes import (  # noqa: E402  (guarded import)
    generate_backplanes_image_files,
)
from spindoctor.cli.reproj.offsets import (  # noqa: E402  (guarded import)
    PointingMechanism,
    apply_pointing_to_obs,
    select_pointing,
)
from spindoctor.cli.reproj.pointing_source import (  # noqa: E402  (guarded import)
    FilePointingSource,
)
from spindoctor.dataset.dataset import ImageFile, ImageFiles  # noqa: E402  (guarded import)
from spindoctor.navigate_image_files import navigate_image_files  # noqa: E402  (guarded import)
from spindoctor.obs import ObsSnapshotInst  # noqa: E402  (guarded import)
from spindoctor.reproj.bodies import BodyMosaic  # noqa: E402  (guarded import)
from spindoctor.reproj.rings import RingMosaic  # noqa: E402  (guarded import)
from spindoctor.support.cmatrix import _oops_correction_matrix  # noqa: E402  (guarded import)
from tests.cmatrix_helpers import observation_attitude  # noqa: E402  (guarded import)
from tests.integration.ck_round_trip import (  # noqa: E402  (guarded import)
    _MISSION_TO_OBS_CLASS,
    holdings_url,
    sidecar_for,
)

# The round-trip library cohort: star-navigated frames of the four
# kernel-eligible instruments, plus the body-navigated Cassini WAC frame.
_CASSINI_NAC = 'N1461997416_1_CALIB'
_CASSINI_WAC = 'W1580760393_1_CALIB'
_VOYAGER_NAC = 'C1205021_GEOMED'
_LORRI = 'lor_0030713591_0x633_sci'
_CASSINI_WAC_BODY = 'W1637520502_1_CALIB'

_COHORT = (_CASSINI_NAC, _CASSINI_WAC, _VOYAGER_NAC, _LORRI, _CASSINI_WAC_BODY)

# A Galileo SSI frame the library records as navigating successfully.  Galileo
# fits a camera rotation, so its record carries no cmatrix and it pins the
# offset-path fallback rather than the comparison.
_GALILEO_SSI = 'C0059894800R'

# The ring-content frame the ring-reprojection comparison runs on: a Cassini
# NAC frame of the main rings from the image library's ring_only_flat class,
# navigated successfully by RingEdgeNav.
_RING_FRAME = 'N1863267861_1_CALIB'

# The body the body-navigated WAC frame resolves, and its mean radius in km,
# which converts a surface distance into the angle it subtends at the center.
_BODY_NAME = 'RHEA'
_BODY_RADIUS_KM = 764.3

# The in-metric measurement: a 17x17 grid across the full frame, eight offset
# directions, 50 px of total boresight displacement.
_GRID_N = 17
_REFERENCE_OFFSET_PX = 50.0
_N_DIRECTIONS = 8

# The derived-bound floor, covering numerical noise and the distorted-FOV
# zero-offset structure (measured at 9.1e-4 px near the WAC center).
_BOUND_FLOOR_PX = 0.005

# How far the two paths may sit apart at the boresight, where they agree by
# construction to floating point.
_BORESIGHT_TOL_PX = 1e-3

# Per-frame pins on the worst LOS-grid pixel-space residual, in px: the value
# measured 2026-08-07 with this module's own _worst_grid_residual_px (17x17
# grid, both paths inverted through the offset path's uv_from_los and
# differenced) at each frame's navigated offset, pinned at roughly twice the
# measured value.  A residual above its pin is a defect to diagnose, never a
# tolerance to raise.
_LOS_RESIDUAL_PIN_PX = {
    _CASSINI_NAC: 2.5e-4,  # measured 1.21e-4
    _CASSINI_WAC: 1.8e-2,  # measured 8.93e-3
    _VOYAGER_NAC: 2.4e-3,  # measured 1.19e-3
    _LORRI: 5.0e-5,  # measured 2.41e-5
    _CASSINI_WAC_BODY: 5.5e-3,  # measured 2.66e-3
    _RING_FRAME: 1.1e-3,  # measured 5.35e-4
}


def _grid_uv(fov: Any) -> list[tuple[float, float]]:
    """Return the 17x17 pixel grid across one FOV.

    Parameters:
        fov: The oops FOV.

    Returns:
        The (u, v) grid points, half a pixel inside the edges.
    """
    size_u, size_v = float(fov.uv_shape.vals[0]), float(fov.uv_shape.vals[1])
    us = np.linspace(0.5, size_u - 0.5, _GRID_N)
    vs = np.linspace(0.5, size_v - 0.5, _GRID_N)
    return [(float(u), float(v)) for u in us for v in vs]


def _residual_px(
    offset_fov: Any, relative: np.ndarray, los_frame: np.ndarray, uv: tuple[float, float]
) -> float:
    """Measure the two paths' pixel-space separation at one pixel.

    Both lines of sight are inverted through the same offset-path mapping and
    differenced, rather than one being inverted and compared against ``uv``
    directly: the distorted-FOV inverse map is itself only approximate (the
    WAC's ``uv_from_los``/``los_from_uv`` round trip misses by up to 0.057 px
    at the frame corners), and differencing two inversions through the same
    map cancels that systematic error, leaving the genuine path
    disagreement.

    Parameters:
        offset_fov: The offset path's FOV (an ``OffsetFOV``).
        relative: The rotation from the C-matrix path's frame coordinates
            into the offset path's frame coordinates.
        los_frame: The unit line of sight the C-matrix path assigns to
            ``uv``, in the C-matrix path's frame.
        uv: The pixel both paths were asked about.

    Returns:
        The pixel-space distance between where the offset path sees the two
        paths' lines of sight.
    """
    los_cm_in_off = relative @ los_frame
    uv_cm = offset_fov.uv_from_los(oops.Vector3(tuple(los_cm_in_off)))
    los_off = offset_fov.los_from_uv(oops.Pair(uv)).unit()
    uv_off = offset_fov.uv_from_los(los_off)
    return math.hypot(
        float(uv_cm.vals[0]) - float(uv_off.vals[0]),
        float(uv_cm.vals[1]) - float(uv_off.vals[1]),
    )


def _worst_grid_residual_px(
    fov: Any, offset_dv_du: tuple[float, float], *, relative: np.ndarray | None = None
) -> tuple[float, float]:
    """Measure the rotation-versus-shift disagreement in pixel space.

    Parameters:
        fov: The frame's unmodified FOV.
        offset_dv_du: The ``(dv, du)`` offset both mechanisms express.
        relative: The rotation carrying C-matrix-path frame coordinates onto
            offset-path frame coordinates, when the caller has real frames in
            hand; None derives it from the offset on the FOV alone.

    Returns:
        Tuple of the worst grid-pixel residual and the boresight residual,
        both in px.
    """
    dv, du = offset_dv_du
    if relative is None:
        # With no real frames, the relative rotation is the transpose of the
        # correction the offset implies: the corrected frame is M . C, so a
        # corrected-frame vector arrives in original-frame coordinates
        # through M^T.
        relative = _oops_correction_matrix(fov, offset_dv_du).T
    offset_fov = oops.fov.OffsetFOV(fov, uv_offset=(du, dv))
    worst = 0.0
    for uv in _grid_uv(fov):
        los = np.asarray(fov.los_from_uv(oops.Pair(uv)).unit().vals, np.float64)
        worst = max(worst, _residual_px(offset_fov, relative, los, uv))
    uv_los = (float(fov.uv_los.vals[0]), float(fov.uv_los.vals[1]))
    boresight_los = np.asarray(fov.los_from_uv(fov.uv_los).unit().vals, np.float64)
    boresight = _residual_px(offset_fov, relative, boresight_los, uv_los)
    return worst, boresight


def _measure_k_inst(fov: Any) -> float:
    """Measure ``K_inst`` on one FOV, in the test's own pixel-space metric.

    Parameters:
        fov: The frame's unmodified FOV.

    Returns:
        The worst pixel-space residual over the grid, swept over eight offset
        directions at 50 px of total displacement.
    """
    worst = 0.0
    for at in range(_N_DIRECTIONS):
        angle = 2.0 * math.pi * at / _N_DIRECTIONS
        offset = (_REFERENCE_OFFSET_PX * math.sin(angle), _REFERENCE_OFFSET_PX * math.cos(angle))
        worst = max(worst, _worst_grid_residual_px(fov, offset)[0])
    return worst


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


def _load_obs(image_id: str) -> ObsSnapshotInst:
    """Load one library image fresh from the holdings.

    Parameters:
        image_id: The library's id for the image.

    Returns:
        The observation.
    """
    obs_class, image_files = _image_files_for(image_id)
    path = image_files.image_files[0].image_file_path.absolute()
    obs = obs_class.from_file(path, extfov_margin_vu=(0, 0))
    assert isinstance(obs, ObsSnapshotInst)
    return obs


def _without_pointing(metadata: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of one record with its pointing block removed.

    Selecting the copy takes the offset mechanism, which is how the offset
    path is produced for comparison without touching the reader's precedence.

    Parameters:
        metadata: The navigation record.

    Returns:
        The copy.
    """
    out = json.loads(json.dumps(metadata))
    out.get('navigation_result', {}).pop('pointing', None)
    return dict(out)


@pytest.fixture(scope='module')
def navigations() -> dict[str, dict[str, Any]]:
    """Hold each frame's navigation metadata, so a frame navigates once.

    Returns:
        The cache, empty to begin with.
    """
    # Module-scoped; --dist=loadfile keeps the whole file on one worker, so
    # the cache is built once there and any other distribution only repeats
    # work, never changes a result.
    return {}


def _navigated(
    image_id: str, cache: dict[str, dict[str, Any]], tmp_path_factory: pytest.TempPathFactory
) -> dict[str, Any]:
    """Navigate one frame in-process, or return what it already measured.

    Parameters:
        image_id: The library's id for the image.
        cache: The findings of the frames already navigated.
        tmp_path_factory: Where the navigation may write its outputs.

    Returns:
        The image's navigation metadata.
    """
    if image_id not in cache:
        obs_class, image_files = _image_files_for(image_id)
        work = tmp_path_factory.mktemp(f'nav_{image_id}')
        _success, metadata = navigate_image_files(
            obs_class, image_files, FCPath(str(work)), write_output_files=False
        )
        assert metadata.get('status') == 'success', (
            f'{image_id} did not navigate; the comparison has no measurement to read'
        )
        cache[image_id] = metadata
    return cache[image_id]


def _both_paths(image_id: str, metadata: dict[str, Any]) -> tuple[ObsSnapshotInst, ObsSnapshotInst]:
    """Load one frame twice and apply the metadata once through each path.

    Parameters:
        image_id: The library's id for the image.
        metadata: The image's navigation record.

    Returns:
        Tuple of the C-matrix-pointed observation and the offset-pointed one.
    """
    obs_cm = _load_obs(image_id)
    obs_off = _load_obs(image_id)
    selection = select_pointing(metadata, subject=image_id)
    assert selection.mechanism is PointingMechanism.CMATRIX
    applied = apply_pointing_to_obs(obs_cm, selection, subject=image_id)
    assert applied.source == 'cmatrix'
    offset_selection = select_pointing(_without_pointing(metadata), subject=image_id)
    assert offset_selection.mechanism is PointingMechanism.OFFSET
    applied_off = apply_pointing_to_obs(obs_off, offset_selection, subject=image_id)
    assert applied_off.source == 'offset'
    return obs_cm, obs_off


@dataclass(frozen=True)
class FrameComparison:
    """One frame's measured two-path agreement and its derived bound.

    Parameters:
        image_id: The library's id for the frame.
        k_inst_px: The frame's measured ``K_inst``, px at 50 px displacement.
        offset_total_px: The navigated offset's total displacement, px.
        bound_px: The derived tolerance ``2 K |offset| / 50 + floor``.
        worst_px: The worst grid-pixel residual between the two paths.
        boresight_px: The boresight residual between the two paths.
    """

    image_id: str
    k_inst_px: float
    offset_total_px: float
    bound_px: float
    worst_px: float
    boresight_px: float


def _compare_frame(image_id: str, metadata: dict[str, Any]) -> FrameComparison:
    """Run the LOS-grid comparison for one navigated frame.

    Parameters:
        image_id: The library's id for the frame.
        metadata: The image's navigation record.

    Returns:
        The measurements.
    """
    obs_cm, obs_off = _both_paths(image_id, metadata)
    fov = obs_cm.fov
    k_inst = _measure_k_inst(fov)
    offset = metadata['offset']
    offset_total = math.hypot(float(offset[0]), float(offset[1]))
    midtime = float(obs_cm.midtime)
    attitude_cm = observation_attitude(obs_cm, midtime)
    attitude_off = observation_attitude(obs_off, midtime)
    relative = attitude_off @ attitude_cm.T
    worst, boresight = _worst_grid_residual_px(
        fov, (float(offset[0]), float(offset[1])), relative=relative
    )
    bound = 2.0 * k_inst * offset_total / _REFERENCE_OFFSET_PX + _BOUND_FLOOR_PX
    return FrameComparison(
        image_id=image_id,
        k_inst_px=k_inst,
        offset_total_px=offset_total,
        bound_px=bound,
        worst_px=worst,
        boresight_px=boresight,
    )


@pytest.fixture(scope='module')
def frame_comparisons() -> dict[str, FrameComparison]:
    """Hold each frame's comparison, so a frame is compared once.

    Returns:
        The cache, empty to begin with.
    """
    return {}


@pytest.fixture(scope='module', params=(*_COHORT, _RING_FRAME))
def frame_comparison(
    request: pytest.FixtureRequest,
    navigations: dict[str, dict[str, Any]],
    frame_comparisons: dict[str, FrameComparison],
    tmp_path_factory: pytest.TempPathFactory,
) -> FrameComparison:
    """Return the LOS-grid comparison for one frame of the cohort.

    Parameters:
        request: The parametrization, naming the image.
        navigations: The cache of navigation metadata.
        frame_comparisons: The cache of comparisons.
        tmp_path_factory: Where navigations may write.

    Returns:
        The frame's measurements.
    """
    image_id = str(request.param)
    if image_id not in frame_comparisons:
        metadata = _navigated(image_id, navigations, tmp_path_factory)
        frame_comparisons[image_id] = _compare_frame(image_id, metadata)
    return frame_comparisons[image_id]


def test_the_boresight_agrees_to_floating_point(frame_comparison: FrameComparison) -> None:
    """At the boresight the two paths agree by construction.

    The correction was built as the minimal rotation carrying the OffsetFOV
    boresight line of sight onto the unmodified one, so any disagreement here
    is a defect in the reader, not a rotation-versus-shift term.
    """
    assert frame_comparison.boresight_px <= _BORESIGHT_TOL_PX


def test_the_worst_grid_pixel_is_within_the_derived_bound(
    frame_comparison: FrameComparison,
) -> None:
    """Away from the boresight the two paths differ at second order, bounded.

    The bound is the frame's own in-metric ``K_inst`` scaled to its navigated
    offset, doubled, plus the floor.  Every directional error this comparison
    guards against -- sign, transpose, skipped conjugation -- displaces the
    geometry by roughly twice the offset instead, several pixels on every
    frame.
    """
    assert frame_comparison.worst_px <= frame_comparison.bound_px


def test_the_worst_grid_pixel_is_within_its_pin(frame_comparison: FrameComparison) -> None:
    """The measured residual stays pinned at measured-plus-margin.

    The derived bound scales with the offset; the pin does not, so a
    regression that grows the residual while staying under the elastic bound
    still fails here.
    """
    assert frame_comparison.worst_px <= _LOS_RESIDUAL_PIN_PX[frame_comparison.image_id]


# ---------------------------------------------------------------------------
# The consumers the readers serve
# ---------------------------------------------------------------------------

# Saturn's main rings, coarsely gridded: enough cells to expose a shifted
# reprojection, small enough to keep one comparison affordable.
_RING_RADIUS_INNER_KM = 74658.0
_RING_RADIUS_OUTER_KM = 136780.0
_RING_RADIUS_RESOLUTION_KM = 25.0
_RING_LONGITUDE_RESOLUTION_RAD = 0.1 * math.pi / 180.0

# Pins on the consumer-level agreement, measured 2026-08-07 with this module
# on the local holdings and pinned at measured-plus-margin (the measured
# values are in the comments and the PR).  The two ring products came out
# bit-identical (every geometry difference is meters against 25 km cells, so
# no pixel changed bins); the pins leave room for a handful of bin-boundary
# flips under numerical drift without letting a directional error -- which
# shifts the whole pattern by roughly twice the offset -- pass.
_RING_MAX_DISAGREEING_FRACTION = 1.0e-3  # measured 0.0 (bit-identical products)
_RING_MAX_DISPUTED_CELL_FRACTION = 1.0e-3  # measured 0.0
_BODY_MAX_DISAGREEING_FRACTION = 5.0e-4  # measured 2.28e-4 (limb boundary cells)
_BODY_MAX_DISPUTED_CELL_FRACTION = 6.5e-4  # measured 3.16e-4
_BACKPLANE_MAX_ANGLE_RATIO = 0.5  # measured 0.148 of the per-pixel allowance
_BACKPLANE_MAX_DISPUTED_PIXEL_FRACTION = 5.0e-5  # measured 5.1e-6


# A cell "disagrees" when its two values differ by more than this fraction of
# the product's own 99th-percentile amplitude.  A sub-bound pointing shift
# moves each cell's sampling by a small fraction of a cell, which changes the
# value negligibly everywhere except across a brightness boundary (a limb or
# a sharp ringlet edge), where a nearest-sample flip swings the cell by the
# full local contrast however small the shift; so the counted quantity is the
# size of that boundary population, not the worst single boundary cell.
_CELL_DISAGREEMENT_FRACTION = 0.05


@dataclass(frozen=True)
class GridAgreement:
    """How two reprojections of one frame agree, cell by cell.

    Parameters:
        max_common_diff: The largest absolute difference over cells valid in
            both products.
        amplitude: The 99th-percentile absolute value of the offset-path
            product over the common cells, which scales the disagreement
            threshold to the data.
        disagreeing_fraction: The fraction of common cells whose values
            differ by more than ``_CELL_DISAGREEMENT_FRACTION`` of the
            amplitude.
        disputed_fraction: The fraction of cells valid in exactly one
            product, over cells valid in either.
    """

    max_common_diff: float
    amplitude: float
    disagreeing_fraction: float
    disputed_fraction: float


def _masked_agreement(a: Any, b: Any) -> GridAgreement:
    """Compare two aligned masked arrays cell by cell.

    Parameters:
        a: The C-matrix-path masked array.
        b: The offset-path masked array, aligned to the first.

    Returns:
        The agreement measurements.
    """
    valid_a = ~np.ma.getmaskarray(a)
    valid_b = ~np.ma.getmaskarray(b)
    both = valid_a & valid_b
    either = valid_a | valid_b
    assert int(np.sum(both)) > 0
    diff = np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))[both]
    amplitude = float(np.percentile(np.abs(np.asarray(b, np.float64)[both]), 99.0))
    assert amplitude > 0.0
    return GridAgreement(
        max_common_diff=float(np.max(diff)),
        amplitude=amplitude,
        disagreeing_fraction=float(np.sum(diff > _CELL_DISAGREEMENT_FRACTION * amplitude))
        / float(np.sum(both)),
        disputed_fraction=float(np.sum(either & ~both)) / float(np.sum(either)),
    )


def _dense_ring_image(result: Any, n_full_lon: int) -> Any:
    """Spread one sparse ring reprojection over the full longitude axis.

    Parameters:
        result: The RingReprojResult.
        n_full_lon: The full number of longitude bins.

    Returns:
        A masked array of shape (n_radius, n_full_lon), masked where the
        result holds no column.
    """
    n_radius = result.img.shape[0]
    dense = np.ma.MaskedArray(
        np.zeros((n_radius, n_full_lon), dtype=np.float64),
        mask=np.ones((n_radius, n_full_lon), dtype=bool),
    )
    dense[:, np.nonzero(result.longitude_antimask)[0]] = result.img
    return dense


def _ring_agreement_for(metadata: dict[str, Any]) -> GridAgreement:
    """Reproject the ring frame through both paths and compare the products.

    This is what exercises ``obs.ext_bp``, the ``Event`` built directly from
    the replaced frame inside ``_reduced_oops_precision``, and the sparse
    binning -- none of which the LOS grid touches.

    Parameters:
        metadata: The ring frame's navigation record.

    Returns:
        The agreement between the two RingReprojResults.
    """
    obs_cm, obs_off = _both_paths(_RING_FRAME, metadata)
    mosaic = RingMosaic(
        'SATURN',
        _RING_RADIUS_INNER_KM,
        _RING_RADIUS_OUTER_KM,
        longitude_resolution=_RING_LONGITUDE_RESOLUTION_RAD,
        radius_resolution=_RING_RADIUS_RESOLUTION_KM,
    )
    result_cm = mosaic.reproject(obs_cm, image_name='cmatrix')
    result_off = mosaic.reproject(obs_off, image_name='offset')
    n_full = int(result_cm.longitude_antimask.shape[0])
    return _masked_agreement(
        _dense_ring_image(result_cm, n_full), _dense_ring_image(result_off, n_full)
    )


@pytest.fixture(scope='module')
def ring_agreement(
    navigations: dict[str, dict[str, Any]], tmp_path_factory: pytest.TempPathFactory
) -> GridAgreement:
    """Return the ring frame's two-path reprojection agreement.

    Parameters:
        navigations: The cache of navigation metadata.
        tmp_path_factory: Where the navigation may write.

    Returns:
        The agreement.
    """
    return _ring_agreement_for(_navigated(_RING_FRAME, navigations, tmp_path_factory))


def test_the_ring_reprojections_populate_the_same_cells(ring_agreement: GridAgreement) -> None:
    """The two paths bin the ring into (radius, longitude) cells alike.

    A sub-bound pointing difference can flip only boundary cells, a thin edge
    of the populated region.
    """
    assert ring_agreement.disputed_fraction <= _RING_MAX_DISPUTED_CELL_FRACTION


def test_the_ring_reprojections_agree_on_common_cells(ring_agreement: GridAgreement) -> None:
    """Where both paths populate a cell, they put the same brightness in it.

    The derived pixel-space bound carried through the projection's local
    scale moves each source pixel by meters against 25 km cells, so away
    from a brightness boundary the binned values cannot move, and at a
    boundary only a bin-flip-thin population can; the counted quantity is
    that population's size against the product's own amplitude.
    """
    assert ring_agreement.disagreeing_fraction <= _RING_MAX_DISAGREEING_FRACTION


def _body_agreement_for(metadata: dict[str, Any]) -> GridAgreement:
    """Reproject the body frame through both paths and compare the products.

    This is what exercises ``uv_from_coords`` and the body lat/lon binning
    through the replaced frame.

    Parameters:
        metadata: The body frame's navigation record.

    Returns:
        The agreement between the two BodyReprojResults, aligned on their
        latitude/longitude index ranges.
    """
    obs_cm, obs_off = _both_paths(_CASSINI_WAC_BODY, metadata)
    mosaic = BodyMosaic(body_name=_BODY_NAME)
    result_cm = mosaic.reproject(obs_cm, image_name='cmatrix')
    result_off = mosaic.reproject(obs_off, image_name='offset')
    lat_lo = min(result_cm.lat_idx_range[0], result_off.lat_idx_range[0])
    lat_hi = max(result_cm.lat_idx_range[1], result_off.lat_idx_range[1])
    lon_lo = min(result_cm.lon_idx_range[0], result_off.lon_idx_range[0])
    lon_hi = max(result_cm.lon_idx_range[1], result_off.lon_idx_range[1])

    def _dense(result: Any) -> Any:
        """Spread one body reprojection onto the union lat/lon index grid.

        Parameters:
            result: The BodyReprojResult.

        Returns:
            A masked array covering the union of both results' index ranges,
            masked where this result holds no cell.
        """
        shape = (lat_hi - lat_lo + 1, lon_hi - lon_lo + 1)
        dense = np.ma.MaskedArray(
            np.zeros(shape, dtype=np.float64), mask=np.ones(shape, dtype=bool)
        )
        lat_at = result.lat_idx_range[0] - lat_lo
        lon_at = result.lon_idx_range[0] - lon_lo
        dense[lat_at : lat_at + result.img.shape[0], lon_at : lon_at + result.img.shape[1]] = (
            result.img
        )
        return dense

    return _masked_agreement(_dense(result_cm), _dense(result_off))


@pytest.fixture(scope='module')
def body_agreement(
    navigations: dict[str, dict[str, Any]], tmp_path_factory: pytest.TempPathFactory
) -> GridAgreement:
    """Return the body frame's two-path reprojection agreement.

    Parameters:
        navigations: The cache of navigation metadata.
        tmp_path_factory: Where the navigation may write.

    Returns:
        The agreement.
    """
    return _body_agreement_for(_navigated(_CASSINI_WAC_BODY, navigations, tmp_path_factory))


def test_the_body_reprojections_populate_the_same_cells(body_agreement: GridAgreement) -> None:
    """The two paths put the body's surface into the same lat/lon cells."""
    assert body_agreement.disputed_fraction <= _BODY_MAX_DISPUTED_CELL_FRACTION


def test_the_body_reprojections_agree_on_common_cells(body_agreement: GridAgreement) -> None:
    """Where both paths populate a cell, they put the same brightness in it.

    The exception the pin allows is the limb boundary, where a sub-bound
    sampling shift can flip a nearest-sample cell by the full local contrast
    however exact the pointing; the counted quantity is that boundary
    population's size against the product's own amplitude.
    """
    assert body_agreement.disagreeing_fraction <= _BODY_MAX_DISAGREEING_FRACTION


@dataclass(frozen=True)
class BackplaneAgreement:
    """How the two paths' end-to-end backplane products agree.

    Parameters:
        worst_angle_ratio: The worst per-pixel angular disagreement between
            the latitude/longitude planes (longitude scaled by cos latitude),
            as a fraction of that pixel's own allowance -- the derived
            pixel-space bound carried through the pixel's surface resolution.
        disputed_fraction: The fraction of pixels claimed by exactly one
            product, over pixels claimed by either.
        cmatrix_source: The ``pointing_source`` the C-matrix run reported.
        offset_source: The ``pointing_source`` the offset run reported.
    """

    worst_angle_ratio: float
    disputed_fraction: float
    cmatrix_source: str
    offset_source: str


def _run_backplanes(
    image_id: str, metadata: dict[str, Any], work: Path
) -> tuple[str, dict[str, np.ndarray]]:
    """Run the backplane driver end to end on one record.

    Parameters:
        image_id: The library's id for the frame.
        metadata: The navigation record to feed the driver.
        work: A fresh directory for the nav and backplane roots.

    Returns:
        The reported ``pointing_source`` and the product's planes by HDU
        name.
    """
    obs_class, image_files = _image_files_for(image_id)
    nav_root = FCPath(str(work)) / 'nav'
    bp_root = FCPath(str(work)) / 'bp'
    (nav_root / f'{image_id}_metadata.json').write_text(json.dumps(metadata))
    result = generate_backplanes_image_files(
        obs_class,
        image_files,
        pointing_source=FilePointingSource(nav_root),
        backplane_results_root=bp_root,
        write_output_files=True,
    )
    assert result['status'] == 'success'
    product = bp_root / f'{image_id}_backplanes.fits'
    planes: dict[str, np.ndarray] = {}
    with fits.open(cast(Path, product.retrieve())) as hdul:
        for hdu in hdul[1:]:
            planes[str(hdu.name)] = np.asarray(hdu.data, np.float64)
    return str(result['pointing_source']), planes


def _backplane_agreement_for(
    metadata: dict[str, Any], comparison: FrameComparison, work_cm: Path, work_off: Path
) -> BackplaneAgreement:
    """Generate backplanes for the body frame through both paths and compare.

    The product's per-pixel planes are body latitude and longitude (the
    backplane set carries no sky coordinates), so the comparison is angular
    position on the body's surface: the derived pixel-space bound converts
    through each pixel's own surface resolution into the angle one bound's
    worth of pixels subtends at the body center.

    Parameters:
        metadata: The body frame's navigation record.
        comparison: The body frame's LOS-grid comparison, whose derived bound
            sizes the per-pixel allowance.
        work_cm: A fresh directory for the C-matrix run.
        work_off: A fresh directory for the offset run.

    Returns:
        The agreement between the two FITS products.
    """
    source_cm, planes_cm = _run_backplanes(_CASSINI_WAC_BODY, metadata, work_cm)
    source_off, planes_off = _run_backplanes(
        _CASSINI_WAC_BODY, _without_pointing(metadata), work_off
    )
    valid_cm = planes_cm['BODY_ID_MAP'] != 0
    valid_off = planes_off['BODY_ID_MAP'] != 0
    both = valid_cm & valid_off
    either = valid_cm | valid_off
    lat_cm = planes_cm['BODY_LATITUDE']
    lat_off = planes_off['BODY_LATITUDE']
    lon_cm = planes_cm['BODY_LONGITUDE']
    lon_off = planes_off['BODY_LONGITUDE']
    # Angular disagreement on the surface, in radians at the body center:
    # latitude difference plus longitude difference scaled to arc length.
    # The longitude planes span 0..2pi, so their difference is wrapped onto
    # (-pi, pi] first; a pixel straddling the seam otherwise scores as ~2pi.
    lon_diff = np.abs(np.angle(np.exp(1j * (lon_cm - lon_off))))
    angle = np.abs(lat_cm - lat_off) + lon_diff * np.abs(np.cos(0.5 * (lat_cm + lat_off)))
    # The allowance per pixel: the derived bound in image pixels, through the
    # pixel's own surface resolution, as an angle at the body center.  The
    # coarser of the two products' resolutions is used, since the shift can
    # land on either.
    resolution = np.maximum(
        planes_cm['BODY_COARSEST_RESOLUTION'], planes_off['BODY_COARSEST_RESOLUTION']
    )
    allowance = comparison.bound_px * resolution / _BODY_RADIUS_KM
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(both & (allowance > 0.0), angle / allowance, 0.0)
    return BackplaneAgreement(
        worst_angle_ratio=float(np.max(ratio)),
        disputed_fraction=float(np.sum(either & ~both)) / float(np.sum(either)),
        cmatrix_source=source_cm,
        offset_source=source_off,
    )


@pytest.fixture(scope='module')
def backplane_agreement(
    navigations: dict[str, dict[str, Any]],
    frame_comparisons: dict[str, FrameComparison],
    tmp_path_factory: pytest.TempPathFactory,
) -> BackplaneAgreement:
    """Return the body frame's two-path end-to-end backplane agreement.

    Parameters:
        navigations: The cache of navigation metadata.
        frame_comparisons: The cache of comparisons; the body frame's entry
            is reused when present and computed into the cache when absent,
            never recomputed beside it.
        tmp_path_factory: Where the navigation and the two runs write.

    Returns:
        The agreement.
    """
    metadata = _navigated(_CASSINI_WAC_BODY, navigations, tmp_path_factory)
    if _CASSINI_WAC_BODY not in frame_comparisons:
        frame_comparisons[_CASSINI_WAC_BODY] = _compare_frame(_CASSINI_WAC_BODY, metadata)
    return _backplane_agreement_for(
        metadata,
        frame_comparisons[_CASSINI_WAC_BODY],
        tmp_path_factory.mktemp('bp_cmatrix'),
        tmp_path_factory.mktemp('bp_offset'),
    )


def test_the_backplane_run_reports_the_cmatrix_source(
    backplane_agreement: BackplaneAgreement,
) -> None:
    """The end-to-end driver applied the C-matrix, and said so."""
    assert backplane_agreement.cmatrix_source == 'cmatrix'


def test_the_stripped_record_reports_the_offset_source(
    backplane_agreement: BackplaneAgreement,
) -> None:
    """The comparison record really did exercise the offset path."""
    assert backplane_agreement.offset_source == 'offset'


def test_the_backplane_products_claim_the_same_pixels(
    backplane_agreement: BackplaneAgreement,
) -> None:
    """The two products place the body over the same pixels, bar the limb edge."""
    assert backplane_agreement.disputed_fraction <= _BACKPLANE_MAX_DISPUTED_PIXEL_FRACTION


def test_the_backplane_geometry_agrees_within_the_bound(
    backplane_agreement: BackplaneAgreement,
) -> None:
    """Per pixel, the surface geometry sits within the converted bound.

    A ratio of one means a pixel disagreed by exactly its allowance: the
    derived pixel-space bound expressed through that pixel's own surface
    resolution as an angle at the body center.
    """
    assert backplane_agreement.worst_angle_ratio <= _BACKPLANE_MAX_ANGLE_RATIO


# ---------------------------------------------------------------------------
# The offset fallback on a real fitted-rotation record
# ---------------------------------------------------------------------------


@pytest.fixture(scope='module')
def galileo_metadata(
    navigations: dict[str, dict[str, Any]], tmp_path_factory: pytest.TempPathFactory
) -> dict[str, Any]:
    """Navigate the Galileo frame, whose record carries no corrected attitude.

    Returns:
        The image's navigation metadata.
    """
    return _navigated(_GALILEO_SSI, navigations, tmp_path_factory)


def test_a_galileo_record_selects_the_offset_path(galileo_metadata: dict[str, Any]) -> None:
    """A fitted-rotation record takes the offset mechanism, with its reason."""
    selection = select_pointing(galileo_metadata, subject=_GALILEO_SSI)
    assert selection.mechanism is PointingMechanism.OFFSET
    assert selection.reason == 'no_cmatrix_rotation_fitted'


def test_a_galileo_product_is_the_offset_product(galileo_metadata: dict[str, Any]) -> None:
    """The applied pointing is exactly the OffsetFOV the offset path always built.

    The reader must not change a fitted-rotation product: the wrapped FOV
    carries precisely the recorded offset and the frame is untouched.
    """
    obs = _load_obs(_GALILEO_SSI)
    frame_before = obs.frame
    selection = select_pointing(galileo_metadata, subject=_GALILEO_SSI)
    applied = apply_pointing_to_obs(obs, selection, subject=_GALILEO_SSI)
    assert applied.source == 'offset'
    assert isinstance(obs.fov, oops.fov.OffsetFOV)
    offset = galileo_metadata['offset']
    assert float(obs.fov.uv_offset.vals[0]) == float(offset[1])
    assert float(obs.fov.uv_offset.vals[1]) == float(offset[0])
    assert obs.frame is frame_before
