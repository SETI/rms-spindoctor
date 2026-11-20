"""The corrected-pointing round trip, on real images against real kernels.

This is the acceptance test of the C-kernel work: navigate a real image, write
the corrected kernel its measurement implies, furnish that kernel in a fresh
process, and navigate the same image again.  If everything from the C-matrix
convention through the segment writer to the baseline pairing is right, the
second navigation finds nothing left to correct.

The three steps run as three subprocesses, because oops caches frames and
manages its own kernel pool; :mod:`tests.integration.ck_round_trip` is the
program each subprocess runs, and this module reads what the three of them
wrote and does the deciding.

What is asserted, and why in these units:

- **The pointing actually changed.**  Read straight from the kernel pool at the
  exposure midtime, the camera attitude has moved from the one navigation
  recorded by exactly the offset navigation measured.  Without this, a kernel
  that was furnished underneath the originals and silently ignored would leave
  the re-navigation measuring the same offset as the first run and, on a frame
  whose offset happened to be small, look like a pass.
- **The record epochs read back exactly.**  A segment carries records at the
  exposure start, midtime and stop; those are the epochs this plan claims, and
  the readback is compared against what the segment says rather than against
  what it was meant to say.  Interior epochs are deliberately not asserted: the
  record scheme does not bound them, and a longer exposure's 1 s cadence
  records are reproduced just as exactly but exist only on long exposures.
- **The re-navigated offset is zero.**  This is the end-to-end statement, and
  the only one whose residual is not pure floating-point noise: the pointing
  chain reproduces to 1e-15 radians, while a second navigation of a corrected
  frame lands within a few thousandths of a pixel of zero because the
  techniques re-measure rather than recompute.  A convention error anywhere in
  the chain would leave roughly twice the original offset here, which is
  several pixels on every frame in the cohort.
- **Both runs committed the same techniques, and the same ones carried
  weight.**  The comparison above is only meaningful when the same measurement
  was repeated; a run that committed a different set, or in which a technique's
  confidence collapsed, has not been shown to agree or to disagree, so the test
  fails as inconclusive rather than passing.

The cohort is one star-navigated Cassini NAC frame, one Cassini WAC frame, one
Voyager frame and one New Horizons LORRI frame, and a fifth frame -- a Cassini
WAC view of a resolved body -- carries the pointing assertions but not the
offset one, because on that frame the offset measures the navigation techniques
rather than the pointing.  Galileo SSI cannot take part at all and is tested for
exactly that: its navigation fits a camera rotation, whose pivot no result
records, so it records no corrected attitude to write.
"""

import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

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
        'the round trip needs local binary kernels and the holdings; set OOPS_RESOURCES to a '
        'local SPICE tree and PDS3_HOLDINGS_DIR to the holdings',
        allow_module_level=True,
    )

import oops  # noqa: E402  (guarded import)

from spindoctor.cli.ck.assignment import rotation_angle_rad  # noqa: E402  (guarded import)
from spindoctor.cli.ck.images import (  # noqa: E402  (guarded import)
    ImageEntry,
    OmissionReason,
)
from tests.integration.ck_round_trip import (  # noqa: E402  (guarded import)
    GENERATE,
    NAVIGATE,
    RECORD_LABELS,
    RENAVIGATE,
    angle_to_pixels,
    metadata_path,
    pixel_scales,
    read_json,
    step_path,
)

# The frames the round trip runs on.  The two Cassini frames are star
# navigated, which is the best-constrained truth the library holds, and the WAC
# one is small-offset on purpose: the rotation-versus-shift difference between
# an exact rigid rotation and a uniform pixel shift is 9.89e-2 px at a 50 px
# total offset on a WAC and is linear in the offset, so a 4.8 px frame spends
# about a hundredth of the budget on it.
_CASSINI_NAC = 'N1461997416_1_CALIB'
_CASSINI_WAC = 'W1580760393_1_CALIB'
_VOYAGER_NAC = 'C1205021_GEOMED'
_LORRI = 'lor_0030713591_0x633_sci'

COHORT = (_CASSINI_NAC, _CASSINI_WAC, _VOYAGER_NAC, _LORRI)

# A Cassini WAC frame of a resolved body, whose ensemble is carried by the
# correlation and distance-transform body techniques rather than by star
# centroids.  Its pointing chain is asserted exactly like the cohort's, and its
# re-navigated offset deliberately is not: the techniques are not exactly
# shift-equivariant, and re-measuring this frame after the correction leaves
# 0.1022 px on dv, most of it BodyLimbNav's answers differing by 1.720 px
# where the correction moved the model by 1.858.  Pinning that number would
# pin a property of the navigation techniques in a test about pointing; the
# frame is here because the part this plan owns is exact on it -- the record
# epochs read back to 2.6e-17 rad, the midtime to 8.8e-16 rad, and the pool's
# pointing moves by the measured offset to within 5.7e-5 px -- and a
# convention error would still leave several pixels.
_CASSINI_WAC_BODY = 'W1637520502_1_CALIB'

# A Galileo SSI frame the library records as navigating successfully.  Galileo
# is the one instrument configured to fit a camera rotation, so it stands in
# for the whole mission here.
_GALILEO_SSI = 'C0059894800R'

# How far the re-navigated offset may sit from zero, per axis, in pixels.
# Measured across the cohort the largest residual is 0.0029 px (LORRI); the
# others are 0.0006 / 0.0014 (NAC), 0.0009 / 0.0024 (WAC) and 0.0002 / 0.0001
# (Voyager).  The pin is seven times the largest of those and five times inside
# the 0.1 px per-axis target, which leaves room for a technique's subpixel
# refinement to change without loosening what this test is for: a convention
# error in the C-matrix, the conjugation or the correction's direction leaves
# about twice the original offset here, between 3.7 and 98 px on these frames.
OFFSET_TOL_PX = 0.02

# How far a corrected attitude read back from the kernel pool may sit from the
# attitude it should be, in radians.  Measured over eleven real frames, the
# disagreements are 0 to 5.6e-17 rad for a record read back against what the
# segment says and at most 1.5e-15 rad for the midtime read back against the
# recorded corrected C-matrix -- both floating-point noise.  The pin is nearly
# three orders above the largest of them and still six orders inside a
# thousandth of a Cassini NAC pixel.
ATTITUDE_TOL_RAD = 1e-12

# Each step gets its own generous ceiling so a hung navigation fails the test
# rather than the suite.  A step takes seconds; this is not a performance bound.
_STEP_TIMEOUT_S = 900.0

# How much of a failed step's output to quote back.
_OUTPUT_TAIL_CHARS = 4000

_REPO_ROOT = Path(__file__).resolve().parents[2]


class _StubFov:
    """A field of view whose every pixel looks in one direction.

    Stands in for a degenerate FOV, which no real instrument has and which no
    constructor will build: its pixel scale is zero, so every angle-to-pixel
    conversion made through it would divide by zero.
    """

    def __init__(self) -> None:
        """Report the boresight at the middle of a one-pixel detector."""
        self.uv_los = oops.Pair((0.5, 0.5))

    def xy_from_uv(self, uv: Any) -> Any:
        """Return the tangent-plane origin whatever pixel is asked about.

        Parameters:
            uv: The pixel, ignored.

        Returns:
            The origin.
        """
        return oops.Pair((0.0, 0.0))

    def los_from_xy(self, xy: Any) -> Any:
        """Return the boresight whatever tangent-plane point is asked about.

        Parameters:
            xy: The tangent-plane point, ignored.

        Returns:
            The boresight direction.
        """
        return oops.Vector3((0.0, 0.0, 1.0))


def _run_step(step: str, image_id: str, work: Path) -> None:
    """Run one step of the round trip in its own process.

    Parameters:
        step: Name of the step.
        image_id: The library's id for the image.
        work: The working directory the three steps share.

    Raises:
        AssertionError: if the step fails, quoting the end of its output.
    """
    # The step imports spindoctor in a process of its own, where an editable
    # install resolves the package through a finder that outranks the working
    # directory.  Run from a worktree, it would therefore navigate with the
    # source the install points at rather than the source under test, and every
    # assertion below would describe the wrong tree.  Naming this checkout's
    # src first is what keeps the subprocess measuring the code the test was
    # collected from.
    env = dict(os.environ)
    src = str(_REPO_ROOT / 'src')
    existing = env.get('PYTHONPATH')
    env['PYTHONPATH'] = f'{src}{os.pathsep}{existing}' if existing else src
    completed = subprocess.run(
        [sys.executable, '-m', 'tests.integration.ck_round_trip', step, image_id, str(work)],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
        timeout=_STEP_TIMEOUT_S,
        env=env,
    )
    if completed.returncode != 0:
        raise AssertionError(
            f'the {step} step failed for {image_id} with exit status {completed.returncode}\n'
            f'stdout:\n{completed.stdout[-_OUTPUT_TAIL_CHARS:]}\n'
            f'stderr:\n{completed.stderr[-_OUTPUT_TAIL_CHARS:]}'
        )


class RoundTrip:
    """What the three steps found for one image.

    Parameters:
        image_id: The library's id for the image.
        work: The working directory the three steps shared.
    """

    def __init__(self, image_id: str, work: Path) -> None:
        """Read the three steps' findings."""
        self.image_id = image_id
        self.work = work
        self.navigated = read_json(metadata_path(work, image_id))['navigation_result']
        self.generated = read_json(step_path(work, image_id, GENERATE))
        renavigated = read_json(step_path(work, image_id, RENAVIGATE))
        self.renavigated = renavigated['navigation_result']
        self.measured = renavigated['measured']
        # The smaller of the two axis scales, so an attitude residual converts
        # to the larger number of pixels; on these frames the two differ by
        # about a hundredth of a percent.
        self.scale_rad_px = min(
            float(renavigated['scale_u_rad_px']), float(renavigated['scale_v_rad_px'])
        )

    @property
    def cmatrix(self) -> Any:
        """The corrected attitude the first navigation recorded."""
        return self.navigated['pointing']['cmatrix']

    @property
    def cmatrix_original(self) -> Any:
        """The uncorrected attitude the first navigation recorded."""
        return self.navigated['pointing']['cmatrix_original']

    def claimed(self, label: str) -> Any:
        """Return the camera attitude the written segment claims at one record.

        Parameters:
            label: Which record: ``start``, ``midtime`` or ``stop``.

        Returns:
            The 3x3 attitude, as nine row-major floats.
        """
        return self.generated['claimed'][label]['cmatrix']

    def pixels(self, angle_rad: float) -> float:
        """Express an angle in pixels of this image's own FOV.

        Parameters:
            angle_rad: The angle in radians.

        Returns:
            The angle in pixels.
        """
        return angle_to_pixels(angle_rad, self.scale_rad_px)


def committed_techniques(block: dict[str, Any]) -> set[str]:
    """Return the techniques whose measurement a navigation committed to.

    Parameters:
        block: One run's ``navigation_result`` metadata block.

    Returns:
        The techniques that ran, less those the ensemble excluded, which is
        the set that decided the offset.
    """
    return set(block['techniques_used']) - set(block['excluded_from_consensus'])


def contributing_techniques(block: dict[str, Any]) -> set[str]:
    """Return the techniques that carried any weight in a navigation.

    A technique that ran and answered with no confidence is in the committed
    set above but contributed nothing to the offset, so the two sets are not
    the same statement: a run where a technique's confidence collapsed
    measured something different even though it ran the same techniques.

    Parameters:
        block: One run's ``navigation_result`` metadata block.

    Returns:
        The committed techniques whose confidence is above zero.
    """
    committed = committed_techniques(block)
    return {
        entry['technique_name']
        for entry in block['per_technique']
        if entry['technique_name'] in committed and float(entry['confidence']) > 0.0
    }


@pytest.fixture(scope='module')
def round_trips() -> dict[str, RoundTrip]:
    """Hold each frame's findings, so a frame two fixtures ask for runs once.

    Returns:
        The cache, empty to begin with.
    """
    # Module-scoped, so under xdist every worker that runs tests from this file
    # rebuilds the cache from scratch.  --dist=loadfile (the project's mandated
    # mode) keeps the whole file on one worker, so this costs nothing there;
    # any other distribution would only repeat work, never change a result.
    return {}


def _round_trip(
    image_id: str, cache: dict[str, RoundTrip], tmp_path_factory: pytest.TempPathFactory
) -> RoundTrip:
    """Run the three steps for one frame, or return what they already found.

    Parameters:
        image_id: The library's id for the image.
        cache: The findings of the frames already run.
        tmp_path_factory: Where the three steps share their working directory.

    Returns:
        The round trip's findings.
    """
    if image_id not in cache:
        work = tmp_path_factory.mktemp(image_id)
        for step in (NAVIGATE, GENERATE, RENAVIGATE):
            _run_step(step, image_id, work)
        cache[image_id] = RoundTrip(image_id, work)
    return cache[image_id]


@pytest.fixture(scope='module', params=COHORT)
def round_trip(
    request: pytest.FixtureRequest,
    round_trips: dict[str, RoundTrip],
    tmp_path_factory: pytest.TempPathFactory,
) -> RoundTrip:
    """Return the findings for one frame whose re-navigated offset is pinned.

    Parameters:
        request: The parametrization, naming the image.
        round_trips: The cache of findings.
        tmp_path_factory: Where the three steps share their working directory.

    Returns:
        The round trip's findings.
    """
    return _round_trip(str(request.param), round_trips, tmp_path_factory)


@pytest.fixture(scope='module', params=(*COHORT, _CASSINI_WAC_BODY))
def chain_round_trip(
    request: pytest.FixtureRequest,
    round_trips: dict[str, RoundTrip],
    tmp_path_factory: pytest.TempPathFactory,
) -> RoundTrip:
    """Return the findings for one frame whose pointing chain is asserted.

    This is the cohort plus the body-navigated frame, whose chain is exact
    although its re-navigated offset is a measurement of the navigation
    techniques rather than of the pointing.

    Parameters:
        request: The parametrization, naming the image.
        round_trips: The cache of findings.
        tmp_path_factory: Where the three steps share their working directory.

    Returns:
        The round trip's findings.
    """
    return _round_trip(str(request.param), round_trips, tmp_path_factory)


@pytest.fixture(scope='module')
def galileo_navigation(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Navigate one Galileo SSI frame and return its metadata.

    Only the first step runs: the point is that this frame never reaches the
    second one.

    Parameters:
        tmp_path_factory: Where the step writes its findings.

    Returns:
        The image's full navigation metadata.
    """
    work = tmp_path_factory.mktemp(_GALILEO_SSI)
    _run_step(NAVIGATE, _GALILEO_SSI, work)
    return read_json(metadata_path(work, _GALILEO_SSI))


def test_the_first_navigation_succeeded(chain_round_trip: RoundTrip) -> None:
    """The frame navigates, which is the premise of everything below."""
    assert chain_round_trip.navigated['status'] == 'success'


def test_the_re_navigation_succeeded(chain_round_trip: RoundTrip) -> None:
    """Correcting the pointing does not stop the frame from navigating."""
    assert chain_round_trip.renavigated['status'] == 'success'


def test_both_runs_committed_the_same_techniques(round_trip: RoundTrip) -> None:
    """The two runs repeated one measurement rather than making two.

    Comparing offsets across runs that committed different techniques would
    compare two different measurements, so a mismatch is reported as
    inconclusive rather than as agreement or disagreement.  Both which
    techniques ran and which of them carried any weight have to match: a
    technique whose confidence collapsed between the runs contributed to one
    offset and not to the other.
    """
    first = committed_techniques(round_trip.navigated)
    second = committed_techniques(round_trip.renavigated)
    assert second == first, (
        f'inconclusive-mismatch for {round_trip.image_id}: the first run committed '
        f'{sorted(first)} and the re-navigation committed {sorted(second)}, so the two offsets '
        f'are not measurements of the same thing'
    )
    weighted_first = contributing_techniques(round_trip.navigated)
    weighted_second = contributing_techniques(round_trip.renavigated)
    assert weighted_second == weighted_first, (
        f'inconclusive-mismatch for {round_trip.image_id}: {sorted(weighted_first)} carried '
        f'weight in the first run and {sorted(weighted_second)} in the re-navigation, so the two '
        f'offsets are not measurements of the same thing'
    )


def test_the_kernel_moved_the_pointing_by_the_measured_offset(chain_round_trip: RoundTrip) -> None:
    """The furnished pool answers a pointing that has moved, and by how much.

    Read at the exposure midtime, the camera attitude has left the one the
    navigation recorded by the number of pixels the navigation measured.  A
    corrected kernel that was furnished but had no effect -- buried under the
    originals, or written for the wrong object -- leaves this at zero.
    """
    moved_rad = rotation_angle_rad(
        np.asarray(chain_round_trip.measured['midtime']).reshape(3, 3),
        np.asarray(chain_round_trip.cmatrix_original).reshape(3, 3),
    )
    offset = chain_round_trip.navigated['offset_px']
    expected_px = math.hypot(float(offset[0]), float(offset[1]))
    assert chain_round_trip.pixels(moved_rad) == pytest.approx(expected_px, abs=OFFSET_TOL_PX)


def test_the_pool_answers_the_recorded_corrected_attitude(chain_round_trip: RoundTrip) -> None:
    """At the midtime, the corrected kernel gives back the recorded C-matrix.

    This is the whole claim of the metadata field, made against SPICE rather
    than against the code that wrote it.
    """
    residual_rad = rotation_angle_rad(
        np.asarray(chain_round_trip.measured['midtime']).reshape(3, 3),
        np.asarray(chain_round_trip.cmatrix).reshape(3, 3),
    )
    assert residual_rad <= ATTITUDE_TOL_RAD


@pytest.mark.parametrize('label', RECORD_LABELS)
def test_a_record_epoch_reads_back_what_the_segment_claims(
    chain_round_trip: RoundTrip, label: str
) -> None:
    """Each of the three record epochs answers exactly what was written.

    The comparison is against the segment's own quaternions, so a segment that
    wrote something other than the corrected attitude fails here rather than
    being compared against the intention it was written with.

    Parameters:
        label: Which record: the exposure start, midtime or stop.
    """
    residual_rad = rotation_angle_rad(
        np.asarray(chain_round_trip.measured[label]).reshape(3, 3),
        np.asarray(chain_round_trip.claimed(label)).reshape(3, 3),
    )
    assert residual_rad <= ATTITUDE_TOL_RAD


def _expect_the_voyager_kernel_unread(
    request: pytest.FixtureRequest, round_trip: RoundTrip
) -> None:
    """Mark the Voyager frame's re-navigated offset as an expected failure.

    The Voyager host reads an image's scan platform pointing at the image's own clock
    count, within 20 ticks, and the corrected segment stamps its records at the exposure
    start, midtime and stop, so the host reads the original record and the corrected
    kernel leaves the pointing as it was: the second run measures the first run's offset
    again.  The
    mark is strict, so that the fix shows up here.

    Parameters:
        request: The running test, which takes the mark.
        round_trip: The frame's findings.
    """
    if round_trip.image_id == _VOYAGER_NAC:
        # #1085: the CK writer has to stamp a record at the image's clock count
        request.applymarker(
            pytest.mark.xfail(
                strict=True,
                reason='the Voyager host does not read the corrected segment',
            )
        )


def test_the_re_navigated_offset_is_zero_along_v(
    request: pytest.FixtureRequest, round_trip: RoundTrip
) -> None:
    """Nothing is left to correct along ``v``."""
    _expect_the_voyager_kernel_unread(request, round_trip)
    assert abs(float(round_trip.renavigated['offset_px'][0])) <= OFFSET_TOL_PX


def test_the_re_navigated_offset_is_zero_along_u(
    request: pytest.FixtureRequest, round_trip: RoundTrip
) -> None:
    """Nothing is left to correct along ``u``."""
    _expect_the_voyager_kernel_unread(request, round_trip)
    assert abs(float(round_trip.renavigated['offset_px'][1])) <= OFFSET_TOL_PX


def test_the_re_navigated_cmatrix_matches_the_first(round_trip: RoundTrip) -> None:
    """The second run's corrected attitude is the first run's.

    The second run computes its own C-matrix from its own (near zero) offset on
    top of the corrected baseline, so this is the same statement as the offset
    above expressed as a rotation -- and it is the form acceptance asks for.
    """
    residual_rad = rotation_angle_rad(
        np.asarray(round_trip.renavigated['pointing']['cmatrix']).reshape(3, 3),
        np.asarray(round_trip.cmatrix).reshape(3, 3),
    )
    assert round_trip.pixels(residual_rad) <= OFFSET_TOL_PX


def test_a_galileo_frame_fits_no_camera_rotation(galileo_navigation: dict[str, Any]) -> None:
    """Galileo fits no rotation, so its twist reaches the reported translation.

    The twist is real -- this frame measured -0.432 deg when the instrument
    fitted one -- and it is deliberately absorbed rather than fitted, because a
    rotation each technique measures about its own pivot is not a quantity the
    ensemble can fuse or the attitude can carry.
    """
    assert 'rotation_deg' not in galileo_navigation['navigation_result']


def test_a_galileo_frame_records_a_corrected_attitude(
    galileo_navigation: dict[str, Any],
) -> None:
    """With no fitted rotation the corrected C-matrix is built and recorded."""
    assert 'cmatrix' in galileo_navigation['navigation_result']['pointing']


def test_a_galileo_frame_still_records_its_uncorrected_attitude(
    galileo_navigation: dict[str, Any],
) -> None:
    """Everything but the correction is recorded, exactly as section 2.3 says."""
    assert 'cmatrix_original' in galileo_navigation['navigation_result']['pointing']


def test_a_galileo_frame_is_not_omitted_as_rotation_unsupported(
    galileo_navigation: dict[str, Any],
) -> None:
    """Galileo frames reach the corrected kernels the mission had none of."""
    entry = ImageEntry.from_metadata(galileo_navigation)
    assert entry.ineligibility_reason is not OmissionReason.ROTATION_UNSUPPORTED


def test_pixel_scales_measures_a_flat_fov() -> None:
    """A FOV's measured scale is the scale it was built with."""
    fov = oops.fov.FlatFOV(uv_scale=(1.0e-5, 2.0e-5), uv_shape=(64, 64))
    scale_u, scale_v = pixel_scales(fov)
    assert scale_u == pytest.approx(1.0e-5, rel=1e-6)
    assert scale_v == pytest.approx(2.0e-5, rel=1e-6)


def test_pixel_scales_refuses_a_fov_with_no_scale() -> None:
    """A FOV every pixel of which looks the same way has no scale to measure."""
    with pytest.raises(ValueError, match='not a positive angle'):
        pixel_scales(_StubFov())
