"""Hermetic tests for ``apply_cmatrix_to_obs``, the reading half of the C-matrix work.

Every test builds its own FOV, attitudes and observation stub, so nothing here
depends on SPICE kernels or holdings.  Each directional test is named for the
mutation it pins: a change that flips a sign, transposes a matrix, skips the
conjugation, or removes the identity short-circuit must fail the named test.
The instrument table lookup is injected via ``_frame_identity`` so a synthetic
flip -- including the non-involutory quarter turn no real instrument has -- can
pin the conjugation's direction.
"""

from __future__ import annotations

import math
from typing import Any, cast

import cspyce
import numpy as np
import oops
import pytest
from oops.frame import Frame
from oops.frame.cmatrix import Cmatrix
from tests.cmatrix_helpers import (
    SYNTHETIC_MIDTIME_ET,
    observation_attitude,
    some_attitude,
    synthetic_baseline,
    synthetic_fov,
    synthetic_frame_identity,
)

import spindoctor.support.cmatrix as cmatrix_module
from spindoctor.obs import ObsSnapshotInst
from spindoctor.support.cmatrix import (
    CMATRIX_BASELINE_MISMATCH,
    CMATRIX_FOREIGN_MIDTIME,
    MALFORMED_POINTING,
    CmatrixApplication,
    _build_pointing_solution,
    apply_cmatrix_to_obs,
)
from spindoctor.support.exceptions import NavPointingError

# The planted offset shared by every test that navigates: two distinct,
# non-zero, opposite-signed components so a sign flip or an axis swap cannot
# go unnoticed.  Each application runs at SYNTHETIC_MIDTIME_ET, the epoch
# the synthetic baseline records.
_PLANTED_OFFSET = (8.68, -17.37)

# The 180-degree flip oops applies on top of the SPICE Cassini ISS frames.
_CASSINI_FLIP = np.diag([-1.0, -1.0, 1.0])


# The flip the injected host declares for the stub observations a test builds.
# ``_inject_identity`` sets it, so a test that injects a synthetic instrument
# gets that instrument's convention on both the identity and the observation.
_DECLARED_FLIP: np.ndarray = np.eye(3)


class _FrameOnlyObs:
    """Observation stub carrying exactly what the reader touches.

    The reader reads ``midtime``, reads and replaces ``frame``, and hands the
    observation to the instrument table lookup, which these tests inject.
    """

    # The reader now reaches the observation through the oops Observation API,
    # so the stub borrows the two methods it calls rather than reimplementing
    # them; whatever oops does to a frame is what these tests exercise.
    set_frame = oops.obs.Observation.set_frame
    set_spice_cmatrix = oops.obs.Observation.set_spice_cmatrix

    @property
    def spice_to_frame(self) -> Any:
        """The oops-from-SPICE rotation the injected host declares.

        Read at access time rather than at construction, so a test may build
        the observation before it injects the instrument whose convention it
        wants gated against.
        """
        return oops.Matrix3(_DECLARED_FLIP if self._flip is None else self._flip)

    # The reader also reads the SPICE camera frame off the observation, as the
    # host declares it.
    spice_frame_name = 'TEST_CAMERA'
    spice_frame_id = -999999

    def __init__(
        self,
        c_oops: np.ndarray,
        *,
        midtime: float = SYNTHETIC_MIDTIME_ET,
        flip: np.ndarray | None = None,
    ) -> None:
        """Build the stub around one constant observation-frame attitude.

        Parameters:
            c_oops: The J2000-to-observation-frame rotation.
            midtime: The exposure midtime, TDB seconds past J2000.
            flip: The oops-from-SPICE rotation the host would declare; None
                for the identity.
        """
        self.frame = oops.frame.Cmatrix(c_oops)
        self.midtime = midtime
        self._flip = flip


def _inject_identity(monkeypatch: pytest.MonkeyPatch, flip: np.ndarray) -> None:
    """Make the reader see a synthetic instrument whose flip is ``flip``.

    Parameters:
        monkeypatch: The patcher.
        flip: The flip the injected instrument table reports.
    """
    identity = synthetic_frame_identity(flip)
    monkeypatch.setattr(cmatrix_module, '_frame_identity', lambda obs: identity)
    monkeypatch.setitem(globals(), '_DECLARED_FLIP', np.asarray(flip, dtype=np.float64))


def _navigated_record(
    flip: np.ndarray, *, offset_px: tuple[float, float] = _PLANTED_OFFSET
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Produce one navigated image's recorded pair and its observation attitude.

    Runs the writing half on a synthetic baseline under the given flip, exactly
    as a real navigation would, so the pair the reader is handed means what a
    recorded pair means.

    Parameters:
        flip: The rotation ``R`` between the oops and SPICE frames.
        offset_px: The navigated ``(dv, du)`` offset.

    Returns:
        Tuple of the corrected ``cmatrix``, the ``cmatrix_original`` baseline,
        and the oops observation-frame attitude ``R . cmatrix_original``.
    """
    original = some_attitude()
    solution = _build_pointing_solution(
        synthetic_baseline(original, oops_from_spice=flip),
        synthetic_fov(),
        offset_px=offset_px,
        rotation_fitted=False,
    )
    assert solution.cmatrix is not None
    c_oops: np.ndarray = np.asarray(flip, np.float64) @ original
    return np.asarray(solution.cmatrix, np.float64), original, c_oops


def _boresight_j2000(attitude: np.ndarray, fov: oops.fov.FlatFOV) -> np.ndarray:
    """Return the J2000 line of sight of a FOV's boresight under one attitude.

    Parameters:
        attitude: The J2000-to-frame rotation the FOV is held in.
        fov: The field of view.

    Returns:
        The unit line of sight in J2000.
    """
    los_frame = np.asarray(fov.los_from_uv(fov.uv_los).unit().vals, np.float64)
    los_j2000: np.ndarray = attitude.T @ los_frame
    return los_j2000


def test_the_reader_reproduces_the_offset_boresight(monkeypatch: pytest.MonkeyPatch) -> None:
    """The corrected frame's boresight equals the OffsetFOV line of sight.

    The recorded pair is produced by the writing half from a planted offset;
    applying it through the reader must point the unmodified FOV's boresight
    exactly where the offset path points it.  Any sign flip or reversed
    composition moves it roughly twice the offset away instead.
    """
    fov = synthetic_fov()
    cmatrix, original, c_oops = _navigated_record(_CASSINI_FLIP)
    obs = _FrameOnlyObs(c_oops)
    _inject_identity(monkeypatch, _CASSINI_FLIP)
    outcome = apply_cmatrix_to_obs(
        cast(ObsSnapshotInst, obs), cmatrix, original, SYNTHETIC_MIDTIME_ET
    )
    assert outcome is CmatrixApplication.FRAME_REPLACED
    corrected = observation_attitude(cast(ObsSnapshotInst, obs), SYNTHETIC_MIDTIME_ET)
    dv, du = _PLANTED_OFFSET
    offset_fov = oops.fov.OffsetFOV(fov, uv_offset=(du, dv))
    offset_los = c_oops.T @ np.asarray(offset_fov.los_from_uv(fov.uv_los).unit().vals, np.float64)
    assert _boresight_j2000(np.asarray(corrected, np.float64), fov) == pytest.approx(
        offset_los, abs=1e-13
    )


def _quarter_turn_about_z() -> np.ndarray:
    """A 90-degree rotation about Z: a proper rotation that is not involutory."""
    return np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def test_skipping_the_conjugation_points_the_wrong_way(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Composing the recorded pair without ``R_hat`` misses the recorded offset.

    Under a non-involutory flip the reader still lands each pixel on the
    OffsetFOV line of sight, while composing ``cmatrix`` without ``R_hat`` (or
    with its transpose) lands somewhere else entirely.  Every real flip is
    diagonal and self-inverse, so only a synthetic frame can pin the
    direction; and the quarter turn about Z fixes the boresight itself, so
    the discriminating comparison is at an off-axis pixel.
    """
    fov = synthetic_fov()
    flip = _quarter_turn_about_z()
    cmatrix, original, c_oops = _navigated_record(flip)
    obs = _FrameOnlyObs(c_oops)
    _inject_identity(monkeypatch, flip)
    apply_cmatrix_to_obs(cast(ObsSnapshotInst, obs), cmatrix, original, SYNTHETIC_MIDTIME_ET)
    corrected = np.asarray(
        observation_attitude(cast(ObsSnapshotInst, obs), SYNTHETIC_MIDTIME_ET), np.float64
    )
    dv, du = _PLANTED_OFFSET
    offset_fov = oops.fov.OffsetFOV(fov, uv_offset=(du, dv))
    off_axis_uv = oops.Pair((100.0, 900.0))
    los_frame = np.asarray(fov.los_from_uv(off_axis_uv).unit().vals, np.float64)
    offset_los = c_oops.T @ np.asarray(offset_fov.los_from_uv(off_axis_uv).unit().vals, np.float64)
    # Off the boresight the rotation and the shift differ at second order in
    # field angle -- nanoradians here -- while every wrong composition misses
    # by the field angle itself, five orders of magnitude more.
    assert corrected.T @ los_frame == pytest.approx(offset_los, abs=1e-8)
    r_hat = c_oops @ original.T
    for wrong in (cmatrix, r_hat.T @ cmatrix):
        wrong_los = np.asarray(wrong, np.float64).T @ los_frame
        assert float(np.max(np.abs(wrong_los - offset_los))) > 1e-4


def test_a_zero_correction_reproduces_the_observation_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An identity correction reproduces the midtime attitude exactly.

    ``cmatrix`` and ``cmatrix_original`` equal as arrays must take the
    short-circuit: two float64 matrix products do not cancel to bit precision,
    so without it "no correction means no change" would be false at the 1e-16
    level and this test fails.
    """
    original = some_attitude()
    c_oops = _CASSINI_FLIP @ original
    obs = _FrameOnlyObs(c_oops)
    _inject_identity(monkeypatch, _CASSINI_FLIP)
    outcome = apply_cmatrix_to_obs(
        cast(ObsSnapshotInst, obs), original.copy(), original.copy(), SYNTHETIC_MIDTIME_ET
    )
    assert outcome is CmatrixApplication.FRAME_REPLACED
    corrected = observation_attitude(cast(ObsSnapshotInst, obs), SYNTHETIC_MIDTIME_ET)
    assert np.array_equal(np.asarray(corrected, np.float64), c_oops)


@pytest.mark.parametrize('which', ['cmatrix_original', 'record'])
def test_a_transposed_record_fails_the_flip_gate(
    monkeypatch: pytest.MonkeyPatch, which: str
) -> None:
    """A transposed recording -- still a proper rotation -- trips the gate.

    Transposing ``cmatrix_original``, or the whole record as a column-major
    serialization defect would, displaces ``R_hat`` by far more than the flip
    tolerance, so validation alone (which any proper rotation passes) is not
    what stands between the reader and a wrong product.

    Parameters:
        which: What is transposed: the baseline matrix alone, or both matrices
            of the record together.
    """
    cmatrix, original, c_oops = _navigated_record(_CASSINI_FLIP)
    obs = _FrameOnlyObs(c_oops)
    _inject_identity(monkeypatch, _CASSINI_FLIP)
    if which == 'record':
        cmatrix = cmatrix.T.copy()
    with pytest.raises(NavPointingError, match='the host declares') as info:
        apply_cmatrix_to_obs(
            cast(ObsSnapshotInst, obs), cmatrix, original.T.copy(), SYNTHETIC_MIDTIME_ET
        )
    assert info.value.reason == CMATRIX_BASELINE_MISMATCH


def test_a_drifted_baseline_fails_the_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    """An observation frame that has left the recorded baseline is refused.

    A 1e-5 rad perturbation -- a changed kernel pool -- is four orders of
    magnitude beyond the flip tolerance, and it is not the corrected pool, so
    the outcome is the mismatch refusal rather than a silent application.
    """
    drift = np.asarray(cspyce.axisar([1.0, 0.0, 0.0], 1.0e-5), np.float64)
    cmatrix, original, c_oops = _navigated_record(_CASSINI_FLIP)
    obs = _FrameOnlyObs(drift @ c_oops)
    frame_before = obs.frame
    _inject_identity(monkeypatch, _CASSINI_FLIP)
    with pytest.raises(NavPointingError, match='the host declares') as info:
        apply_cmatrix_to_obs(cast(ObsSnapshotInst, obs), cmatrix, original, SYNTHETIC_MIDTIME_ET)
    assert info.value.reason == CMATRIX_BASELINE_MISMATCH
    # The reader promises the observation is never mutated on a raise.
    assert obs.frame is frame_before


def test_an_already_corrected_pool_is_left_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    """A pool already answering the corrected attitude gets no second correction.

    When the observation frame is ``R . cmatrix`` -- corrected kernels
    furnished at load time -- the flip gate fires, the probe recognizes the
    state, the distinguished outcome is returned, and the frame is not
    replaced: the observation is already right, and either fallback would
    corrupt it.
    """
    cmatrix, original, _ = _navigated_record(_CASSINI_FLIP)
    obs = _FrameOnlyObs(_CASSINI_FLIP @ cmatrix)
    frame_before = obs.frame
    _inject_identity(monkeypatch, _CASSINI_FLIP)
    outcome = apply_cmatrix_to_obs(
        cast(ObsSnapshotInst, obs), cmatrix, original, SYNTHETIC_MIDTIME_ET
    )
    assert outcome is CmatrixApplication.POOL_ALREADY_CORRECTED
    assert obs.frame is frame_before


def test_a_foreign_midtime_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """A record whose midtime is another observation's is refused by the gate."""
    cmatrix, original, c_oops = _navigated_record(_CASSINI_FLIP)
    obs = _FrameOnlyObs(c_oops)
    _inject_identity(monkeypatch, _CASSINI_FLIP)
    with pytest.raises(NavPointingError, match='belongs to a different observation') as info:
        apply_cmatrix_to_obs(
            cast(ObsSnapshotInst, obs), cmatrix, original, SYNTHETIC_MIDTIME_ET + 1.0
        )
    assert info.value.reason == CMATRIX_FOREIGN_MIDTIME


def test_an_unmapped_host_is_refused() -> None:
    """An observation whose instrument the frame table does not know is refused.

    The real table lookup runs here: the stub is no registered instrument, so
    no expected flip exists to gate against and applying anything would be a
    guess.
    """
    cmatrix, original, c_oops = _navigated_record(_CASSINI_FLIP)
    obs = _FrameOnlyObs(c_oops)
    with pytest.raises(NavPointingError, match='no SPICE camera frame') as info:
        apply_cmatrix_to_obs(cast(ObsSnapshotInst, obs), cmatrix, original, SYNTHETIC_MIDTIME_ET)
    assert info.value.reason == cmatrix_module.CMATRIX_UNKNOWN_HOST


@pytest.mark.parametrize(
    ('mutate', 'match'),
    [
        ('nan_cmatrix', 'cmatrix holds a non-finite value'),
        ('nan_original', 'cmatrix_original holds a non-finite value'),
        ('wrong_shape', 'expected a 3x3 matrix'),
        ('wrong_rank', 'expected a 3x3 matrix'),
        ('non_rotation', 'cmatrix is not a proper rotation'),
        ('bool_elements', 'cmatrix holds values that are not real numbers'),
        ('absent_original', 'cmatrix_original holds values that are not real numbers'),
    ],
    ids=[
        'nan-cmatrix',
        'nan-original',
        'wrong-shape',
        'wrong-rank',
        'non-rotation',
        'bool-elements',
        'absent-original',
    ],
)
def test_a_malformed_record_is_refused(
    monkeypatch: pytest.MonkeyPatch, mutate: str, match: str
) -> None:
    """Each malformed-record class is refused rather than coerced.

    NaN defeats every comparison, ``reshape`` accepts nine values of any rank,
    and booleans convert to float64 without complaint, so each class is probed
    as its own input domain.

    Parameters:
        mutate: Which malformation to plant.
        match: The refusal message expected for it.
    """
    cmatrix, original, c_oops = _navigated_record(_CASSINI_FLIP)
    obs = _FrameOnlyObs(c_oops)
    _inject_identity(monkeypatch, _CASSINI_FLIP)
    bad_cmatrix: Any = cmatrix
    bad_original: Any = original
    if mutate == 'nan_cmatrix':
        bad_cmatrix = cmatrix.copy()
        bad_cmatrix[1, 1] = np.nan
    elif mutate == 'nan_original':
        bad_original = original.copy()
        bad_original[0, 2] = np.nan
    elif mutate == 'wrong_shape':
        bad_cmatrix = np.eye(4)
    elif mutate == 'wrong_rank':
        bad_cmatrix = cmatrix.reshape(9)
    elif mutate == 'non_rotation':
        bad_cmatrix = np.diag([1.0, 1.0, 2.0])
    elif mutate == 'bool_elements':
        bad_cmatrix = np.eye(3, dtype=np.bool_)
    elif mutate == 'absent_original':
        bad_original = None
    with pytest.raises(NavPointingError, match=match) as info:
        apply_cmatrix_to_obs(
            cast(ObsSnapshotInst, obs), bad_cmatrix, bad_original, SYNTHETIC_MIDTIME_ET
        )
    assert info.value.reason == MALFORMED_POINTING


@pytest.mark.parametrize(
    ('bad_midtime', 'match'),
    [
        (math.nan, 'midtime_et is not finite'),
        (None, 'midtime_et is not a real number'),
        (True, 'midtime_et is not a real number'),
    ],
    ids=['nan', 'none', 'bool'],
)
def test_a_malformed_midtime_is_refused(
    monkeypatch: pytest.MonkeyPatch, bad_midtime: Any, match: str
) -> None:
    """A midtime the gate cannot compare against is refused, never waved through.

    A NaN ``midtime_et`` makes the gate's inequality false both ways, which
    would silently defeat the one check that ties the record to this
    observation; it is refused as malformed before the gate runs.

    Each case carries the refusal it must produce, because the three reach two
    different gates: a NaN is a real number and fails only the finiteness
    check.  Matching on the field name alone would accept a reader that had
    dropped that check while keeping the type check.

    Parameters:
        bad_midtime: The unusable midtime to plant.
        match: The refusal message expected for it.
    """
    cmatrix, original, c_oops = _navigated_record(_CASSINI_FLIP)
    obs = _FrameOnlyObs(c_oops)
    _inject_identity(monkeypatch, _CASSINI_FLIP)
    with pytest.raises(NavPointingError, match=match) as info:
        apply_cmatrix_to_obs(cast(ObsSnapshotInst, obs), cmatrix, original, bad_midtime)
    assert info.value.reason == MALFORMED_POINTING


def test_frame_replacement_registers_no_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """The replacement frame is registered under no ID a later lookup can find."""
    cmatrix, original, c_oops = _navigated_record(_CASSINI_FLIP)
    _inject_identity(monkeypatch, _CASSINI_FLIP)
    frame_registry_before = dict(Frame._FRAME_REGISTRY)  # type: ignore[attr-defined]  # private to oops; its stub omits it
    for _ in range(2):
        obs = _FrameOnlyObs(c_oops)
        apply_cmatrix_to_obs(cast(ObsSnapshotInst, obs), cmatrix, original, SYNTHETIC_MIDTIME_ET)
    assert frame_registry_before == Frame._FRAME_REGISTRY  # type: ignore[attr-defined]  # private to oops; its stub omits it


def test_each_distinct_replacement_is_retained(monkeypatch: pytest.MonkeyPatch) -> None:
    """A replacement is cached by its matrix and never released.

    oops keys every Cmatrix by its own matrix, so each image a batch re-points
    to a distinct attitude leaves one wayframe and two frame-cache entries
    behind for the life of the process.  This pins the growth rate, so a
    change that bounds it is visible here rather than as a memory report from
    a full-catalog run.
    """
    _inject_identity(monkeypatch, _CASSINI_FLIP)
    # The stub's own frame is a Cmatrix too, and on a cold cache the first one
    # built would be counted alongside the replacements.  Build it once before
    # the baseline is taken, so what the deltas measure is the replacements.
    _, _, warm_c_oops = _navigated_record(_CASSINI_FLIP)
    _FrameOnlyObs(warm_c_oops)
    wayframes_before = len(Cmatrix._WAYFRAMES)
    cache_before = len(Frame._FRAME_CACHE)  # type: ignore[attr-defined]  # private to oops; its stub omits it
    replacements = 3
    for i in range(replacements):
        cmatrix, original, c_oops = _navigated_record(_CASSINI_FLIP, offset_px=(1.0 + i, -2.0 - i))
        obs = _FrameOnlyObs(c_oops)
        apply_cmatrix_to_obs(cast(ObsSnapshotInst, obs), cmatrix, original, SYNTHETIC_MIDTIME_ET)
    assert len(Cmatrix._WAYFRAMES) == wayframes_before + replacements
    assert len(Frame._FRAME_CACHE) == cache_before + 2 * replacements  # type: ignore[attr-defined]  # private to oops; its stub omits it
