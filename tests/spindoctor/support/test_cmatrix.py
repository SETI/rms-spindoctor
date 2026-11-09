"""Hermetic tests for ``spindoctor.support.cmatrix``.

Every test here builds its own FOV and its own attitudes, so nothing depends
on SPICE kernels or on a real observation.  The conventions under test are
the ones a hermetic test can silently get wrong in both directions at once:
the sign of the tangent-plane offset, the direction of the rotation
``cspyce.axisar`` builds, and the conjugation of the oops-frame correction
into the SPICE frame.  Each is pinned by recovering the planted offset back
out of the recorded C-matrix with an independent inverse.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, cast

import numpy as np
import oops
import pytest
from tests.cmatrix_helpers import (
    offset_from_correction,
    some_attitude,
    synthetic_baseline,
    synthetic_fov,
)

from spindoctor.obs import ObsSnapshotInst
from spindoctor.spice_ids import CK_OBJECT_SCLK_ID
from spindoctor.support.cmatrix import (
    _CASSINI_CK_FRAME_ID,
    PointingSolution,
    _attitude_baseline,
    _build_pointing_solution,
    _check_flip,
    _ck_object_sclk_id,
    _declared_camera_frame,
    _FrameIdentity,
    _oops_correction_matrix,
    _pxform,
    _sclk_id,
    _sclk_string,
    _spice_cmatrix,
    _validate_rotation,
    compute_pointing,
)
from spindoctor.support.exceptions import NavPointingError

# A planted offset with two distinct, non-zero, opposite-signed components,
# so a sign flip or an axis swap cannot go unnoticed.
_PLANTED_OFFSET = (8.68, -17.37)

# The 180-degree flip oops applies on top of the SPICE Cassini ISS frames.
_CASSINI_FLIP = np.diag([-1.0, -1.0, 1.0])

# Voyager 1's scan-platform CK object, which is the wrong one for a Voyager 2
# image: ``ckmeta`` answers -31 for it, the other spacecraft's clock.
_VOYAGER1_CK_FRAME_ID = -31100
_VOYAGER2_SCLK_ID = -32

# A CK object no mission owns, used to pin what ``ckmeta`` does with one.
_NONEXISTENT_CK_FRAME_ID = -999999

# A spacecraft clock no furnished kernel can ever describe.
_NONEXISTENT_SCLK_ID = -999

_IDENTITY = np.eye(3)


def _outcome_of(check: Callable[[], None]) -> str:
    """Report whether a validator accepted or refused its input.

    The validators under test signal acceptance by returning and refusal by
    raising, so a test that only calls one asserts nothing.  This turns the
    two outcomes into a value a test can assert on.
    """
    try:
        check()
    except NavPointingError:
        return 'refused'
    return 'accepted'


def test_correction_maps_the_uncorrected_boresight_onto_the_corrected_one() -> None:
    """``M . d`` is the direction the unmodified FOV assigns to the boresight."""
    fov = synthetic_fov()
    correction = _oops_correction_matrix(fov, _PLANTED_OFFSET)
    xy_los = fov.xy_from_uv(fov.uv_los)
    los_u, los_v = np.asarray(fov.uv_los.vals, np.float64)
    xy_offset = fov.xy_from_uv(oops.Pair((los_u + _PLANTED_OFFSET[1], los_v + _PLANTED_OFFSET[0])))
    uncorrected = np.asarray(fov.los_from_xy(xy_los - xy_offset).unit().vals, np.float64)
    corrected = np.asarray(fov.los_from_xy(xy_los).unit().vals, np.float64)
    assert correction @ uncorrected == pytest.approx(corrected, abs=1e-15)


def test_correction_is_a_proper_rotation() -> None:
    """The correction matrix has unit determinant."""
    correction = _oops_correction_matrix(synthetic_fov(), _PLANTED_OFFSET)
    assert float(np.linalg.det(correction)) == pytest.approx(1.0, abs=1e-14)


def test_planted_offset_is_recovered_from_the_recorded_cmatrix() -> None:
    """A planted offset survives the round trip through the recorded C-matrix.

    With no flip between the oops and SPICE frames, inverting the recorded
    correction must return the planted ``(dv, du)``.  A flipped ``xy_offset``
    sign, or a transposed correction, returns the negated offset instead.
    """
    fov = synthetic_fov()
    original = some_attitude()
    solution = _build_pointing_solution(
        synthetic_baseline(original), fov, offset_px=_PLANTED_OFFSET, rotation_fitted=False
    )
    assert solution.cmatrix is not None
    correction = np.asarray(solution.cmatrix, np.float64) @ original.T
    recovered = offset_from_correction(fov, correction)
    assert recovered[0] == pytest.approx(_PLANTED_OFFSET[0], abs=1e-9)
    assert recovered[1] == pytest.approx(_PLANTED_OFFSET[1], abs=1e-9)


def test_zero_offset_leaves_the_cmatrix_identical_to_the_original() -> None:
    """A zero offset takes the identity guard and changes nothing at all."""
    original = some_attitude()
    solution = _build_pointing_solution(
        synthetic_baseline(original), synthetic_fov(), offset_px=(0.0, 0.0), rotation_fitted=False
    )
    assert solution.cmatrix is not None
    assert np.array_equal(solution.cmatrix, solution.baseline.cmatrix_original)


def test_zero_offset_correction_is_exactly_the_identity() -> None:
    """The correction for a zero offset is the identity matrix, bit for bit."""
    assert np.array_equal(_oops_correction_matrix(synthetic_fov(), (0.0, 0.0)), _IDENTITY)


def test_cassini_style_flip_reproduces_the_offset_in_the_spice_convention() -> None:
    """The recorded C-matrix carries the planted offset through a 180-degree flip.

    With ``R = diag(-1, -1, 1)`` between the oops observation frame and the
    SPICE camera frame, converting both recorded attitudes back into the oops
    convention must yield exactly the correction the offset implies.  Dropping
    the ``R`` conjugation leaves a correction of the right magnitude whose
    tangent-plane components are both negated.
    """
    fov = synthetic_fov()
    original = some_attitude()
    solution = _build_pointing_solution(
        synthetic_baseline(original, oops_from_spice=_CASSINI_FLIP),
        fov,
        offset_px=_PLANTED_OFFSET,
        rotation_fitted=False,
    )
    assert solution.cmatrix is not None
    corrected_oops = _CASSINI_FLIP @ np.asarray(solution.cmatrix, np.float64)
    original_oops = _CASSINI_FLIP @ original
    recovered = offset_from_correction(fov, corrected_oops @ original_oops.T)
    assert recovered[0] == pytest.approx(_PLANTED_OFFSET[0], abs=1e-9)
    assert recovered[1] == pytest.approx(_PLANTED_OFFSET[1], abs=1e-9)


def test_flip_conjugation_changes_the_recorded_cmatrix() -> None:
    """Conjugating through the flip is not a no-op for a Cassini-style flip.

    Guards the test above: if ``R^T M R`` happened to equal ``M`` here, that
    test would pass with the conjugation dropped.
    """
    fov = synthetic_fov()
    original = some_attitude()
    correction = _oops_correction_matrix(fov, _PLANTED_OFFSET)
    conjugated = _spice_cmatrix(original, correction, _CASSINI_FLIP)
    unconjugated = correction @ original
    assert float(np.max(np.abs(conjugated - unconjugated))) > 1e-6


def test_fitted_rotation_records_the_original_but_no_corrected_cmatrix() -> None:
    """A result carrying a fitted camera rotation gets no corrected attitude."""
    original = some_attitude()
    solution = _build_pointing_solution(
        synthetic_baseline(original),
        synthetic_fov(),
        offset_px=_PLANTED_OFFSET,
        rotation_fitted=True,
    )
    assert solution.cmatrix is None
    assert np.array_equal(solution.baseline.cmatrix_original, original)


def test_missing_offset_records_the_original_but_no_corrected_cmatrix() -> None:
    """A result with no offset gets no corrected attitude."""
    original = some_attitude()
    solution = _build_pointing_solution(
        synthetic_baseline(original), synthetic_fov(), offset_px=None, rotation_fitted=False
    )
    assert solution.cmatrix is None
    assert np.array_equal(solution.baseline.cmatrix_original, original)


def test_baseline_with_a_bad_determinant_raises() -> None:
    """A baseline attitude that is not a proper rotation is refused."""
    reflected = some_attitude()
    reflected[0] = -reflected[0]
    with pytest.raises(NavPointingError, match='cmatrix_original is not a proper rotation'):
        _build_pointing_solution(
            synthetic_baseline(reflected),
            synthetic_fov(),
            offset_px=_PLANTED_OFFSET,
            rotation_fitted=False,
        )


def test_baseline_that_is_not_orthonormal_raises() -> None:
    """A unit-determinant baseline that is not orthonormal is refused."""
    sheared = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    with pytest.raises(NavPointingError, match='cmatrix_original is not orthonormal'):
        _build_pointing_solution(
            synthetic_baseline(sheared),
            synthetic_fov(),
            offset_px=_PLANTED_OFFSET,
            rotation_fitted=False,
        )


def test_baseline_with_a_bad_determinant_raises_without_an_offset() -> None:
    """The baseline is checked even when no corrected attitude is produced."""
    reflected = some_attitude()
    reflected[0] = -reflected[0]
    with pytest.raises(NavPointingError, match='cmatrix_original is not a proper rotation'):
        _build_pointing_solution(
            synthetic_baseline(reflected), synthetic_fov(), offset_px=None, rotation_fitted=False
        )


def test_corrected_cmatrix_that_is_not_a_rotation_raises() -> None:
    """A correction that is not a rotation is refused by the result check."""
    with pytest.raises(NavPointingError, match='cmatrix is not a proper rotation'):
        _spice_cmatrix(some_attitude(), np.diag([1.0, 1.0, 2.0]), _IDENTITY)


def test_attitude_baseline_rejects_a_matrix_of_the_wrong_shape() -> None:
    """A baseline built from a non-3x3 matrix is refused at construction."""
    with pytest.raises(NavPointingError, match='expected a 3x3 matrix'):
        synthetic_baseline(np.eye(4))


def test_recorded_matrices_are_read_only() -> None:
    """Recorded C-matrices cannot be mutated through the returned solution."""
    solution = _build_pointing_solution(
        synthetic_baseline(some_attitude()),
        synthetic_fov(),
        offset_px=_PLANTED_OFFSET,
        rotation_fitted=False,
    )
    assert solution.cmatrix is not None
    with pytest.raises(ValueError, match='read-only'):
        solution.cmatrix[0, 0] = 0.0


def test_a_sub_pixel_offset_still_produces_a_real_correction() -> None:
    """A small but genuine offset is corrected, not swallowed by the guard.

    Pins the degenerate-axis threshold from below: raising it far enough to
    absorb a real sub-pixel offset would silently record every such image as
    needing no correction at all.
    """
    fov = synthetic_fov()
    small = (0.05, -0.02)
    correction = _oops_correction_matrix(fov, small)
    assert not np.array_equal(correction, _IDENTITY)
    recovered = offset_from_correction(fov, correction)
    assert recovered[0] == pytest.approx(small[0], abs=1e-9)
    assert recovered[1] == pytest.approx(small[1], abs=1e-9)


def _quarter_turn_about_z() -> np.ndarray:
    """A 90-degree rotation about Z: a proper rotation that is not involutory."""
    return np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def test_conjugation_direction_is_pinned_by_a_non_involutory_flip() -> None:
    """The conjugation is ``R^T M R``, not ``R M R^T``.

    Every flip in the instrument table is diagonal and therefore its own
    inverse, so the two orders agree on real data and no real-frame test can
    tell them apart.  A quarter turn about Z can.
    """
    fov = synthetic_fov()
    flip = _quarter_turn_about_z()
    original = some_attitude()
    solution = _build_pointing_solution(
        synthetic_baseline(original, oops_from_spice=flip),
        fov,
        offset_px=_PLANTED_OFFSET,
        rotation_fitted=False,
    )
    assert solution.cmatrix is not None
    recovered = offset_from_correction(
        fov, (flip @ np.asarray(solution.cmatrix, np.float64)) @ (flip @ original).T
    )
    assert recovered[0] == pytest.approx(_PLANTED_OFFSET[0], abs=1e-9)
    assert recovered[1] == pytest.approx(_PLANTED_OFFSET[1], abs=1e-9)


def test_the_two_conjugation_orders_differ_under_a_non_involutory_flip() -> None:
    """Guards the test above: the quarter turn really does tell them apart."""
    flip = _quarter_turn_about_z()
    correction = _oops_correction_matrix(synthetic_fov(), _PLANTED_OFFSET)
    forward = flip.T @ correction @ flip
    reversed_order = flip @ correction @ flip.T
    assert float(np.max(np.abs(forward - reversed_order))) > 1e-6


def test_a_nan_baseline_is_rejected() -> None:
    """A NaN attitude is refused rather than serialized as an invalid JSON token."""
    corrupted = some_attitude()
    corrupted[1, 1] = np.nan
    with pytest.raises(NavPointingError, match='cmatrix_original holds a non-finite value'):
        _build_pointing_solution(
            synthetic_baseline(corrupted),
            synthetic_fov(),
            offset_px=_PLANTED_OFFSET,
            rotation_fitted=False,
        )


def test_an_infinite_baseline_is_rejected() -> None:
    """An infinite attitude entry is refused for the same reason."""
    corrupted = some_attitude()
    corrupted[0, 2] = np.inf
    with pytest.raises(NavPointingError, match='cmatrix_original holds a non-finite value'):
        _build_pointing_solution(
            synthetic_baseline(corrupted), synthetic_fov(), offset_px=None, rotation_fitted=False
        )


def test_a_nan_correction_is_rejected_by_the_result_check() -> None:
    """A NaN reaching the corrected matrix is refused too."""
    corrupted = np.eye(3) * np.nan
    with pytest.raises(NavPointingError, match='cmatrix holds a non-finite value'):
        _spice_cmatrix(some_attitude(), corrupted, _IDENTITY)


def _cassini_identity() -> _FrameIdentity:
    """The Cassini NAC frame identity, for the flip-check and clock tests."""
    return _FrameIdentity(
        ck_frame_id=_CASSINI_CK_FRAME_ID,
        sclk_id=CK_OBJECT_SCLK_ID[_CASSINI_CK_FRAME_ID],
        frozen_oops_attitude=False,
    )


def test_a_mismatched_flip_is_refused() -> None:
    """A measured flip that is not the instrument's constant raises."""
    with pytest.raises(NavPointingError, match='the host declares'):
        _check_flip(_IDENTITY, _CASSINI_FLIP, 'CASSINI_ISS_NAC', 0.0)


def test_the_mismatched_flip_message_names_the_camera_frame() -> None:
    """The refusal says which frame disagreed, so it is attributable."""
    with pytest.raises(NavPointingError, match='CASSINI_ISS_NAC'):
        _check_flip(_IDENTITY, _CASSINI_FLIP, 'CASSINI_ISS_NAC', 0.0)


def test_a_flip_within_tolerance_is_accepted() -> None:
    """A flip differing by less than the tolerance is accepted silently."""
    perturbed = _CASSINI_FLIP + np.full((3, 3), 1e-12)
    outcome = _outcome_of(lambda: _check_flip(perturbed, _CASSINI_FLIP, 'CASSINI_ISS_NAC', 0.0))
    assert outcome == 'accepted'


def test_a_flip_just_outside_tolerance_is_refused() -> None:
    """A flip differing by more than the tolerance raises."""
    perturbed = _CASSINI_FLIP.copy()
    perturbed[0, 1] = 1e-8
    with pytest.raises(NavPointingError, match='the host declares'):
        _check_flip(perturbed, _CASSINI_FLIP, 'CASSINI_ISS_NAC', 0.0)


def test_the_missions_own_clock_resolves_and_is_accepted() -> None:
    """The clock SPICE resolves for a mission's CK object is the recorded one.

    Pins the recorded clock against what ``ckmeta`` actually computes: a typo
    in either the CK object or the clock id fails here rather than producing
    time strings from a plausible-looking wrong clock.
    """
    resolved = _sclk_id(_cassini_identity(), 'CASSINI_ISS_NAC')
    assert resolved == CK_OBJECT_SCLK_ID[_CASSINI_CK_FRAME_ID]


def test_a_ck_object_with_no_recorded_clock_is_a_pointing_failure() -> None:
    """An object outside the recorded set is refused rather than looked up."""
    with pytest.raises(NavPointingError, match='has no recorded spacecraft clock'):
        _ck_object_sclk_id(_NONEXISTENT_CK_FRAME_ID)


def _voyager2_identity_with_the_voyager1_ck_object() -> _FrameIdentity:
    """A Voyager 2 identity carrying Voyager 1's CK object.

    One Voyager instrument key serves two spacecraft, so naming the wrong
    spacecraft's CK object is the realistic way this mismatch arises.
    """
    return _FrameIdentity(
        ck_frame_id=_VOYAGER1_CK_FRAME_ID,
        sclk_id=_VOYAGER2_SCLK_ID,
        frozen_oops_attitude=True,
    )


def test_a_clock_that_is_not_the_missions_is_refused() -> None:
    """A CK object resolving to another spacecraft's clock raises."""
    with pytest.raises(NavPointingError, match='resolves to spacecraft clock -31'):
        _sclk_id(_voyager2_identity_with_the_voyager1_ck_object(), 'VG2_ISSNA')


def test_the_refused_clock_message_names_the_expected_clock() -> None:
    """The refusal says which clock was expected, so it is attributable."""
    with pytest.raises(NavPointingError, match='not the -32'):
        _sclk_id(_voyager2_identity_with_the_voyager1_ck_object(), 'VG2_ISSNA')


def test_a_frame_the_kernel_pool_does_not_know_is_a_pointing_failure() -> None:
    """A rotation SPICE cannot supply is reported as the computation's own failure.

    The original SPICE exception survives the conversion as the cause, so the
    failure stays debuggable.
    """
    with pytest.raises(NavPointingError, match='cannot supply the J2000 to NO_SUCH_FRAME') as info:
        _pxform('NO_SUCH_FRAME', 0.0)
    assert isinstance(info.value.__cause__, LookupError)


class _StubDeclaringObs:
    """Observation stub carrying only the two frame subfields a host declares."""

    def __init__(self, name: str, frame_id: int) -> None:
        """Build the stub around one declared frame.

        Parameters:
            name: The SPICE frame name the host declares.
            frame_id: The SPICE frame id the host declares.
        """
        self.spice_frame_name = name
        self.spice_frame_id = frame_id


def test_a_frame_name_with_no_spice_id_is_a_pointing_failure() -> None:
    """A declared frame name that resolves to no id is reported, not left to surface later."""
    obs = cast(ObsSnapshotInst, _StubDeclaringObs('NO_SUCH_FRAME', -999999))
    with pytest.raises(NavPointingError, match='NO_SUCH_FRAME has no SPICE id'):
        _declared_camera_frame(obs)


def test_a_clock_that_cannot_encode_an_epoch_is_a_pointing_failure() -> None:
    """A clock with no furnished SCLK kernel is reported rather than raising raw."""
    with pytest.raises(NavPointingError, match='cannot encode et'):
        _sclk_string(_NONEXISTENT_SCLK_ID, 0.0)


def test_a_host_with_no_spice_camera_frame_records_nothing() -> None:
    """An observation whose instrument has no frame mapping yields no solution.

    A simulated image has no spacecraft and no furnished camera frame; it is
    returned as "nothing to record" rather than raising, so a simulated
    navigation runs unchanged.
    """
    unmapped = cast(ObsSnapshotInst, object())
    assert compute_pointing(unmapped, offset_px=_PLANTED_OFFSET, rotation_fitted=False) is None


def test_a_determinant_just_inside_tolerance_is_accepted() -> None:
    """A matrix whose determinant is within the tolerance of 1 is accepted.

    Pins the rotation tolerance from below.  Scaling a rotation by
    ``1 + 1e-11`` moves its determinant by about ``3e-11``, comfortably
    inside the documented ``1e-9``.
    """
    nearly = some_attitude() * (1.0 + 1e-11)
    assert _outcome_of(lambda: _validate_rotation(nearly, 'test matrix')) == 'accepted'


def test_a_determinant_just_outside_tolerance_is_refused() -> None:
    """A determinant further than the tolerance from 1 raises.

    Pins the rotation tolerance from above.  Scaling by ``1 + 1e-8`` moves
    the determinant by about ``3e-8``, an order of magnitude outside the
    documented ``1e-9``, so widening the tolerance fails this test.
    """
    with pytest.raises(NavPointingError, match='is not a proper rotation'):
        _validate_rotation(some_attitude() * (1.0 + 1e-8), 'test matrix')


def test_a_shear_just_inside_tolerance_is_accepted() -> None:
    """A unit-determinant shear smaller than the tolerance is accepted.

    A shear of ``s`` leaves the determinant exactly 1 and moves
    ``max|C C^T - I|`` by about ``s``, so it exercises the orthonormality
    bound alone.
    """
    sheared = np.array([[1.0, 1e-11, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    assert _outcome_of(lambda: _validate_rotation(sheared, 'test matrix')) == 'accepted'


def test_a_shear_just_outside_tolerance_is_refused() -> None:
    """A unit-determinant shear larger than the tolerance raises."""
    sheared = np.array([[1.0, 1e-8, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    with pytest.raises(NavPointingError, match='is not orthonormal'):
        _validate_rotation(sheared, 'test matrix')


@pytest.mark.parametrize('offset', [(), (1.0, 2.0, 3.0)], ids=['empty', 'triple'])
def test_an_offset_that_is_not_a_pair_is_refused(offset: tuple[float, ...]) -> None:
    """An offset without exactly two components is refused before anything else.

    The guard runs before the instrument lookup, so even an observation with
    no SPICE camera frame has its offset checked: a malformed offset is a
    defect in the caller, never "nothing to record".
    """
    unmapped = cast(ObsSnapshotInst, object())
    with pytest.raises(ValueError, match=r'must hold exactly \(dv, du\)'):
        compute_pointing(unmapped, offset_px=cast(Any, offset), rotation_fitted=False)


@pytest.mark.parametrize('offset', [(math.nan, 0.0), (0.0, math.inf)], ids=['nan', 'inf'])
def test_a_non_finite_offset_is_refused(offset: tuple[float, float]) -> None:
    """A NaN or infinite offset component is a ValueError, not a pointing failure.

    A caller that absorbs ``NavPointingError`` per image must not absorb a
    regressed technique emitting NaN offsets for a whole batch, so the guard
    deliberately raises the un-absorbed type.
    """
    unmapped = cast(ObsSnapshotInst, object())
    with pytest.raises(ValueError, match='holds a non-finite value'):
        compute_pointing(unmapped, offset_px=offset, rotation_fitted=False)


class _StubTimedObs:
    """Times-only observation stub for the baseline's finiteness guard.

    The guard runs before any SPICE lookup, so nothing beyond the four time
    attributes is ever read.
    """

    def __init__(
        self,
        *,
        start: float = 100.0,
        stop: float = 100.5,
        midtime: float = 100.25,
        texp: float = 0.5,
    ) -> None:
        """Build the stub with any one value poisoned as the test requires."""
        self.time = (start, stop)
        self.midtime = midtime
        self.texp = texp
        self.spice_to_frame = oops.Matrix3(_CASSINI_FLIP)
        self.spice_frame_name = 'CASSINI_ISS_NAC'
        self.spice_frame_id = -82360


@pytest.mark.parametrize(
    ('poisoned', 'label'),
    [
        ({'start': math.nan}, 'start'),
        ({'stop': math.nan}, 'stop'),
        ({'midtime': math.nan}, 'midtime'),
        ({'texp': math.nan}, 'exposure duration'),
        ({'texp': math.inf}, 'exposure duration'),
    ],
    ids=['nan-start', 'nan-stop', 'nan-midtime', 'nan-exposure', 'inf-exposure'],
)
def test_a_non_finite_observation_time_is_refused(poisoned: dict[str, float], label: str) -> None:
    """A non-finite epoch or exposure never reaches SPICE or the metadata.

    The exposure duration especially: the epochs would be refused by the
    spacecraft clock conversion anyway, but ``texp`` passes through nothing
    before serialization, so this guard is its only defense.
    """
    obs = cast(ObsSnapshotInst, _StubTimedObs(**poisoned))
    with pytest.raises(NavPointingError, match=f'records a non-finite {label}'):
        _attitude_baseline(obs, _cassini_identity())


def test_attitude_baseline_refuses_a_nan_matrix_at_construction() -> None:
    """Construction itself enforces finiteness rather than trusting callers."""
    with pytest.raises(NavPointingError, match='cmatrix_original holds a non-finite value'):
        synthetic_baseline(np.full((3, 3), np.nan))


def test_attitude_baseline_refuses_a_non_rotation_flip_at_construction() -> None:
    """The flip matrix is validated at construction alongside the attitude."""
    with pytest.raises(NavPointingError, match='oops_from_spice is not a proper rotation'):
        synthetic_baseline(some_attitude(), oops_from_spice=np.diag([1.0, 1.0, 2.0]))


def test_pointing_solution_refuses_a_nan_corrected_matrix_at_construction() -> None:
    """A NaN corrected matrix is refused when the solution is built."""
    with pytest.raises(NavPointingError, match='cmatrix holds a non-finite value'):
        PointingSolution(
            baseline=synthetic_baseline(some_attitude()), cmatrix=np.full((3, 3), np.nan)
        )


def test_pointing_solution_refuses_a_non_rotation_corrected_matrix_at_construction() -> None:
    """A corrected matrix that is not a proper rotation is refused at construction."""
    with pytest.raises(NavPointingError, match='cmatrix is not a proper rotation'):
        PointingSolution(
            baseline=synthetic_baseline(some_attitude()), cmatrix=np.diag([1.0, 1.0, 2.0])
        )
