"""Corrected-attitude C-matrices for a navigated observation.

A C-matrix is the rotation taking a vector expressed in J2000 to the same vector
expressed in a camera frame::

    v_frame = C . v_J2000

which is what ``cspyce.pxform('J2000', frame_name, et)`` returns.  Navigation
measures a pixel offset; this module converts that offset into the corrected
attitude the camera actually had, so the same measurement can later be written
into a C-kernel and consumed by any SPICE-aware tool without applying a pixel
offset by hand.

Two matrices are produced per navigated image, both in the SPICE camera frame
convention and both at the exposure midtime:

``cmatrix_original``
    The attitude the furnished kernels gave at navigation time, before any
    correction.
``cmatrix``
    The corrected attitude, the one a kernel should carry.

The oops observation frame is **not** the SPICE camera frame.  Each host states
the relation between them: it inserts the constant rotation ``R`` satisfying
``C_oops = R . C_spice`` as the observation's ``spice_to_frame`` subfield, and
names the SPICE camera frame in ``spice_frame_name`` and ``spice_frame_id``.
This module reads all three rather than keeping a table of its own, so the frame
an attitude is recorded in and the frame oops built the observation on cannot
name different cameras -- the confusion that puts a wide-angle attitude in the
narrow-angle frame.

A correction built in the oops frame and composed onto a SPICE-convention matrix
without accounting for ``R`` is a proper rotation of the right magnitude
pointing the wrong way, so the two conversions go through
``Observation.get_spice_cmatrix`` and ``Observation.set_spice_cmatrix``, which
apply it in both directions from the one declaration.

What a host declares is a claim, not a measurement, so it is checked: at the
exposure start, midtime and stop the rotation between the observation frame and
``pxform`` of the named camera frame must equal the declared ``R``.  A frame
that is right at the midtime and wrong at the edges is a moving frame
misdeclared as a constant one, and the correction is written to a kernel as a
single body-fixed rotation.

Voyager is the exception to the check, not to the convention: oops freezes the
observation frame from a tolerance-snapped ``ckgp`` lookup rather than
evaluating a frame chain, so ``pxform`` cannot evaluate that camera frame at
all.  The accessor answers correctly there regardless, so only the cross-check
is skipped.

What stays here is what oops does not do: turning a navigated pixel offset into
a rotation, refusing a malformed record, naming the C-kernel object and the
spacecraft clock a correction is written against, and gating a recorded
attitude before it is applied.
"""

import enum
import math
from dataclasses import dataclass

import cspyce
import numpy as np
import oops

from spindoctor.obs import (
    ObsCassiniISS,
    ObsGalileoSSI,
    ObsNewHorizonsLORRI,
    ObsSnapshotInst,
    ObsVoyagerISS,
)
from spindoctor.spice_ids import CK_OBJECT_SCLK_ID, VOYAGER_CK_OBJECT_ID
from spindoctor.support.exceptions import NavPointingError

# Which array dtypes count as recorded numbers is a property of a metadata
# record rather than of the C-matrix conventions, and the module that reads
# recorded values owns it, so this validator and the record reader cannot
# come to disagree about what nine ``True`` values are.
from spindoctor.support.nav_record import REAL_NUMBER_DTYPE_KINDS
from spindoctor.support.types import NDArrayAnyType, NDArrayFloatType

# The offset-to-attitude conversion and its inverse are deliberately behind
# two public functions -- plus the record validator the readers share -- so
# that when oops grows its own corrected-attitude API this module's body is
# replaced and only ``compute_pointing``, ``apply_cmatrix_to_obs`` and
# ``validated_record_rotation`` have to survive the swap.  The helpers below
# them are private for that reason, not because they are trivial.
__all__ = [
    'CMATRIX_BASELINE_MISMATCH',
    'CMATRIX_FOREIGN_MIDTIME',
    'CMATRIX_UNKNOWN_HOST',
    'MALFORMED_POINTING',
    'AttitudeBaseline',
    'CmatrixApplication',
    'PointingSolution',
    'apply_cmatrix_to_obs',
    'compute_pointing',
    'validated_record_rotation',
]

# The machine-readable classifications ``apply_cmatrix_to_obs`` stamps onto
# the ``NavPointingError`` it raises, so a caller that degrades per image can
# tally its degradations per reason without parsing messages.
MALFORMED_POINTING = 'malformed_pointing'
CMATRIX_FOREIGN_MIDTIME = 'cmatrix_foreign_midtime'
CMATRIX_BASELINE_MISMATCH = 'cmatrix_baseline_mismatch'
CMATRIX_UNKNOWN_HOST = 'cmatrix_unknown_host'


class CmatrixApplication(enum.Enum):
    """What ``apply_cmatrix_to_obs`` did to the observation.

    ``FRAME_REPLACED`` is the ordinary outcome: the observation's frame now
    carries the corrected attitude.  ``POOL_ALREADY_CORRECTED`` is the
    distinguished no-op: the furnished kernel pool already answered the
    corrected attitude, so the observation was left untouched -- it is
    already right, and applying the correction again (or falling back to the
    pixel offset) would corrupt it by roughly twice the navigated offset.
    The member values are the short reason strings run-level accounting uses.
    """

    FRAME_REPLACED = 'cmatrix'
    POOL_ALREADY_CORRECTED = 'pool_already_corrected'


# The object each mission's C-kernels describe.  The correction is measured at
# the camera but written at the bus or platform the existing kernels already
# cover, so a corrected kernel targets one of these.  Voyager needs one per
# spacecraft under a single instrument key, and its pair is read from
# ``spindoctor.spice_ids`` because the kernel writer needs the same two ids to
# recognize a frozen-attitude object.
_CASSINI_CK_FRAME_ID = -82000
_GALILEO_CK_FRAME_ID = -77001
_LORRI_CK_FRAME_ID = -98000

# What cspyce raises when the furnished kernels cannot answer: a missing frame
# or an unresolvable clock arrives as a LookupError, an unreadable kernel as
# an OSError, and a SPICE error as a RuntimeError or a ValueError.  Each is an
# environment failure rather than a defect, so it is converted to a
# NavPointingError at the call site that made it.
_SPICE_FAILURES = (LookupError, OSError, RuntimeError, ValueError)

_IDENTITY: NDArrayFloatType = np.eye(3)
_IDENTITY.setflags(write=False)

# Angular tolerance, in matrix-element terms, for the measured flip matching
# its expected constant value and for that value being epoch-independent.
_FLIP_TOL = 1e-9

# A C-matrix must be a proper rotation to this tolerance; anything looser is a
# defect in the frame chain rather than something to orthonormalize away.
_ROTATION_TOL = 1e-9

# How far a recorded midtime may sit from the observation's own before the
# record is judged to belong to a different observation.  The two are the
# same float64 computation serialized unrounded, so they agree exactly; a
# microsecond leaves room for nothing but representation noise.
_MIDTIME_TOL_S = 1e-6

# Below this cross-product norm the corrected and uncorrected boresights are
# the same direction to sub-nanoradian precision and the correction is exactly
# the identity.
_DEGENERATE_AXIS_NORM = 1e-12


@dataclass(frozen=True)
class _FrameIdentity:
    """Per-instrument frame facts needed to place a correction in SPICE terms.

    Parameters:
        ck_frame_id: SPICE id of the object a corrected C-kernel targets.
        sclk_id: SPICE id of the spacecraft clock that object's time tags are
            encoded against.
        frozen_oops_attitude: True when oops freezes the observation frame
            from a tolerance-snapped pointing lookup, so ``pxform`` cannot
            evaluate the camera frame at all and the declared flip has nothing
            to be cross-checked against.
    """

    ck_frame_id: int
    sclk_id: int
    frozen_oops_attitude: bool


@dataclass(frozen=True)
class AttitudeBaseline:
    """The SPICE-derived facts about one observation, independent of any offset.

    Parameters:
        cmatrix_original: Uncorrected J2000-to-camera rotation at the exposure
            midtime, in the SPICE camera frame convention.
        oops_from_spice: The measured constant rotation ``R`` satisfying
            ``C_oops = R . C_spice``.
        camera_frame: SPICE name of the camera frame.
        camera_frame_id: SPICE id of the camera frame.
        ck_frame_id: SPICE id of the object a corrected C-kernel targets.
        start_et: Exposure start, TDB seconds past J2000.
        stop_et: Exposure stop, TDB seconds past J2000.
        midtime_et: Exposure midtime, TDB seconds past J2000.
        exposure_s: Exposure duration in seconds.
        sclk_start: Spacecraft clock string at ``start_et``.
        sclk_midtime: Spacecraft clock string at ``midtime_et``.
        sclk_stop: Spacecraft clock string at ``stop_et``.
    """

    cmatrix_original: NDArrayFloatType
    oops_from_spice: NDArrayFloatType
    camera_frame: str
    camera_frame_id: int
    ck_frame_id: int
    start_et: float
    stop_et: float
    midtime_et: float
    exposure_s: float
    sclk_start: str
    sclk_midtime: str
    sclk_stop: str

    def __post_init__(self) -> None:
        """Store both matrices as read-only 3x3 float arrays and validate them.

        Raises:
            NavPointingError: if either matrix is not a proper orthonormal
                rotation.  The class is public, so construction itself
                enforces what the docstring promises rather than trusting
                every caller to have validated first.
        """
        object.__setattr__(self, 'cmatrix_original', _as_readonly_3x3(self.cmatrix_original))
        object.__setattr__(self, 'oops_from_spice', _as_readonly_3x3(self.oops_from_spice))
        _validate_rotation(self.cmatrix_original, 'cmatrix_original')
        _validate_rotation(self.oops_from_spice, 'oops_from_spice')


@dataclass(frozen=True)
class PointingSolution:
    """A navigated image's baseline attitude plus its corrected attitude.

    Parameters:
        baseline: The uncorrected attitude, frame identities, and times.
        cmatrix: Corrected J2000-to-camera rotation at the exposure midtime,
            in the SPICE camera frame convention.  ``None`` when the result
            carried no offset, or carried a fitted camera rotation whose pivot
            is not recorded and therefore cannot be expressed as an attitude.
    """

    baseline: AttitudeBaseline
    cmatrix: NDArrayFloatType | None

    def __post_init__(self) -> None:
        """Store the corrected matrix as a read-only 3x3 float array and validate it.

        Raises:
            NavPointingError: if the corrected matrix is present and is not a
                proper orthonormal rotation.
        """
        if self.cmatrix is not None:
            object.__setattr__(self, 'cmatrix', _as_readonly_3x3(self.cmatrix))
            _validate_rotation(self.cmatrix, 'cmatrix')


def _as_readonly_3x3(matrix: NDArrayFloatType) -> NDArrayFloatType:
    """Copy ``matrix`` into a read-only 3x3 float64 array.

    Parameters:
        matrix: Any 3x3 array-like of numbers.

    Returns:
        A read-only float64 copy.

    Raises:
        NavPointingError: if the input is not 3x3.
    """
    out = np.array(matrix, dtype=np.float64)
    if out.shape != (3, 3):
        raise NavPointingError(f'expected a 3x3 matrix; got shape {out.shape}')
    out.setflags(write=False)
    return out


def _validate_rotation(matrix: NDArrayFloatType, label: str) -> None:
    """Raise unless ``matrix`` is a proper rotation to ``_ROTATION_TOL``.

    Parameters:
        matrix: The 3x3 matrix to check.
        label: Name used in the exception message.

    Raises:
        NavPointingError: if the matrix holds a non-finite value, or if its
            determinant differs from 1, or it is not orthonormal, by more than
            the tolerance.
    """
    # NaN fails every inequality below, so both tolerance guards would pass a
    # NaN matrix silently; and a non-finite value serialized into the metadata
    # is written as a bare NaN / Infinity token that strict JSON parsers
    # reject.  Reject it here, where the defect is still attributable.
    if not bool(np.all(np.isfinite(matrix))):
        raise NavPointingError(f'{label} holds a non-finite value: {np.asarray(matrix).tolist()!r}')
    det = float(np.linalg.det(matrix))
    if abs(det - 1.0) > _ROTATION_TOL:
        raise NavPointingError(
            f'{label} is not a proper rotation: determinant {det!r} differs from 1 by more '
            f'than {_ROTATION_TOL}'
        )
    residual = float(np.max(np.abs(matrix @ matrix.T - np.eye(3))))
    if residual > _ROTATION_TOL:
        raise NavPointingError(
            f'{label} is not orthonormal: max|C C^T - I| = {residual!r} exceeds {_ROTATION_TOL}'
        )


def _oops_correction_matrix(fov: oops.fov.FOV, offset_px: tuple[float, float]) -> NDArrayFloatType:
    """Build the pointing correction a navigated offset implies, in oops terms.

    The offset is applied downstream as ``oops.fov.OffsetFOV(fov,
    uv_offset=(du, dv))``, which maps pixels to tangent-plane coordinates as
    ``fov.xy_from_uv(uv) - xy_offset`` with ``xy_offset =
    fov.xy_from_uv(fov.uv_los + (du, dv))``.  Under the corrected pointing the
    true direction seen by pixel ``uv`` in the *original* frame is therefore
    ``fov.los_from_xy(fov.xy_from_uv(uv) - xy_offset)``, and the corrected
    frame is the one in which the unmodified FOV holds.  Evaluating that at
    the boresight gives the returned rotation ``M``, the minimal rotation
    taking the uncorrected boresight direction to the corrected one.

    ``M`` is an active vector rotation: ``M . d`` is the corrected direction
    ``b``, where ``d`` is the direction the uncorrected FOV assigns to the
    boresight pixel.  It is exact by construction; nothing is orthonormalized.
    An exact rigid rotation is not exactly a uniform tangent-plane shift, so
    away from the boresight the rotation and the pixel offset differ at second
    order in field angle.

    Parameters:
        fov: The observation's unmodified oops FOV.
        offset_px: The navigated ``(dv, du)`` offset in pixels.

    Returns:
        The 3x3 rotation ``M`` in oops observation frame coordinates.  Exactly
        the identity when the offset moves the boresight by less than about a
        picoradian.

    Raises:
        ValueError: if the offset is antipodal -- the two boresight directions
            oppose each other -- where the rotation axis is undefined.  A
            navigated offset of half the sky is a defect in the caller's data,
            not a pointing this module can express.
    """
    dv, du = float(offset_px[0]), float(offset_px[1])
    # Every concrete FOV states its line of sight; the FOV base class declares none.
    uv_los: oops.Pair = fov.uv_los  # type: ignore[attr-defined]
    los_u, los_v = np.asarray(uv_los.vals, dtype=np.float64)
    xy_los = fov.xy_from_uv(uv_los)
    xy_offset = fov.xy_from_uv(oops.Pair((los_u + du, los_v + dv)))
    uncorrected = np.asarray(fov.los_from_xy(xy_los - xy_offset).unit().vals, dtype=np.float64)
    corrected = np.asarray(fov.los_from_xy(xy_los).unit().vals, dtype=np.float64)
    axis = np.cross(uncorrected, corrected)
    axis_norm = float(np.linalg.norm(axis))
    dot = float(np.dot(uncorrected, corrected))
    if axis_norm < _DEGENERATE_AXIS_NORM:
        # A vanishing cross product is two directions aligned or two directions
        # opposed.  Only the aligned case is a no-op correction; the opposed
        # one has no minimal rotation at all, and treating it as the identity
        # would silently claim an uncorrected attitude for a wildly wrong one.
        if dot < 0.0:
            raise ValueError(
                f'the offset {(dv, du)!r} px turns the boresight antipodal; no minimal '
                f'rotation expresses it'
            )
        return _IDENTITY
    # arctan2(|d x b|, d . b) is the well-conditioned form of arccos(d . b)
    # for unit vectors: arccos loses relative precision as the angle goes to
    # zero, which is exactly the regime a sub-pixel offset lives in.
    angle = float(np.arctan2(axis_norm, dot))
    correction: NDArrayFloatType = np.asarray(
        cspyce.axisar(axis / axis_norm, angle), dtype=np.float64
    )
    return correction


def _spice_cmatrix(
    cmatrix_original: NDArrayFloatType,
    correction: NDArrayFloatType,
    oops_from_spice: NDArrayFloatType,
) -> NDArrayFloatType:
    """Apply an oops-frame correction to a SPICE-convention C-matrix.

    With ``R`` the constant rotation satisfying ``C_oops = R . C_spice`` and
    ``M`` the correction expressed in oops observation frame coordinates, the
    corrected attitude in the SPICE convention is ``(R^T . M . R) . C``.
    Skipping the conjugation yields a proper rotation of the right magnitude
    pointing the wrong way.

    Parameters:
        cmatrix_original: Uncorrected J2000-to-camera rotation, SPICE
            convention.
        correction: The oops-frame correction ``M``.
        oops_from_spice: The constant rotation ``R``.

    Returns:
        The corrected 3x3 J2000-to-camera rotation in the SPICE convention.
        Bit-identical to ``cmatrix_original`` when the correction is exactly
        the identity, since no correction must mean no change.

    Raises:
        NavPointingError: if ``cmatrix_original`` or the corrected result is
            not a proper orthonormal rotation.
    """
    original = np.asarray(cmatrix_original, dtype=np.float64)
    _validate_rotation(original, 'cmatrix_original')
    if np.array_equal(correction, np.eye(3)):
        return original
    rotation = np.asarray(oops_from_spice, dtype=np.float64)
    conjugated = rotation.T @ np.asarray(correction, dtype=np.float64) @ rotation
    corrected: NDArrayFloatType = conjugated @ original
    _validate_rotation(corrected, 'cmatrix')
    return corrected


def _build_pointing_solution(
    baseline: AttitudeBaseline,
    fov: oops.fov.FOV,
    *,
    offset_px: tuple[float, float] | None,
    rotation_fitted: bool,
) -> PointingSolution:
    """Combine a SPICE baseline and a navigated offset into a PointingSolution.

    The baseline is always carried through.  A corrected ``cmatrix`` is
    produced only when the navigation yielded an offset and did not fit a
    camera rotation: a fitted rotation turns about a per-technique pivot that
    no result records, so the correction it implies is not expressible from
    the recorded data and no corrected attitude is claimed.

    Parameters:
        baseline: The observation's uncorrected attitude, frames, and times.
        fov: The observation's unmodified oops FOV.
        offset_px: The navigated ``(dv, du)`` offset, or ``None`` when the
            navigation produced no offset.
        rotation_fitted: True when the result carries a fitted camera
            rotation.

    Returns:
        A PointingSolution whose ``cmatrix`` is ``None`` unless an offset was
        supplied with no fitted rotation.

    Raises:
        NavPointingError: if the baseline or the corrected matrix is not a
            proper orthonormal rotation.
    """
    if offset_px is None or rotation_fitted:
        _validate_rotation(np.asarray(baseline.cmatrix_original, np.float64), 'cmatrix_original')
        return PointingSolution(baseline=baseline, cmatrix=None)
    correction = _oops_correction_matrix(fov, offset_px)
    cmatrix = _spice_cmatrix(baseline.cmatrix_original, correction, baseline.oops_from_spice)
    return PointingSolution(baseline=baseline, cmatrix=cmatrix)


def compute_pointing(
    obs: ObsSnapshotInst,
    *,
    offset_px: tuple[float, float] | None,
    rotation_fitted: bool,
) -> PointingSolution | None:
    """Compute the recorded and corrected attitudes for one navigated image.

    This is the single entry point for the offset-to-attitude conversion.  It
    reads the observation's own frame and FOV, measures the constant rotation
    between the oops observation frame and the SPICE camera frame, checks that
    rotation against the constant expected for the instrument, and returns
    both attitudes in the SPICE convention along with the frame identities and
    the exposure times a C-kernel writer needs.

    Every expected failure is reported as a ``NavPointingError``, so a caller
    that must survive an attitude the environment cannot supply absorbs that
    one exception and nothing else: any other exception escaping this function
    is a defect in it.

    Parameters:
        obs: The navigated observation.
        offset_px: The navigated ``(dv, du)`` offset, or ``None``.
        rotation_fitted: True when the result carries a fitted camera
            rotation.

    Returns:
        A PointingSolution, or ``None`` when the observation's host is not one
        whose SPICE frames this module knows (a simulated image, for example).

    Raises:
        NavPointingError: if the furnished kernels cannot supply the attitude,
            the camera frame or the spacecraft clock; if the resolved
            spacecraft clock is not the one expected for the mission; if the
            measured flip between the oops and SPICE frames differs from the
            constant expected for the instrument or varies across the
            exposure; or if either C-matrix is not a proper orthonormal
            rotation.
        ValueError: if ``offset_px`` is malformed -- not exactly two values,
            or holding a non-finite one.  A malformed offset is a defect in
            the caller, not an attitude the environment cannot supply, so it
            is deliberately not a ``NavPointingError``: a caller absorbing
            those per image must not absorb a regressed technique emitting
            NaN offsets for a whole batch.  Also if a Voyager observation's
            label names a spacecraft that is neither Voyager, which the host
            refuses when it reads the label.
    """
    if offset_px is not None:
        if len(offset_px) != 2:
            raise ValueError(f'offset_px must hold exactly (dv, du); got {offset_px!r}')
        if not all(math.isfinite(float(value)) for value in offset_px):
            raise ValueError(f'offset_px holds a non-finite value: {offset_px!r}')
    identity = _frame_identity(obs)
    if identity is None:
        return None
    baseline = _attitude_baseline(obs, identity)
    return _build_pointing_solution(
        baseline, obs.fov, offset_px=offset_px, rotation_fitted=rotation_fitted
    )


def validated_record_rotation(matrix: NDArrayAnyType, label: str) -> NDArrayFloatType:
    """Validate one recorded C-matrix for the reader, refusing rather than coercing.

    The parameter is deliberately an array of any dtype, not a float array:
    refusing the dtypes a metadata record can carry -- booleans, integers,
    text -- is exactly this function's job, so a caller hands it whatever the
    record held and lets it judge.  Narrowing the annotation to a float array
    would make the callers cast, which is the coercion this refuses to do.

    Parameters:
        matrix: The recorded 3x3 rotation, of any dtype.
        label: Name used in refusal messages.

    Returns:
        A read-only float64 copy.

    Raises:
        NavPointingError: with reason ``malformed_pointing`` if the value is
            not a 3x3 array of real numbers forming a proper orthonormal
            rotation.  Booleans are refused although they convert to float64
            without complaint, since nine ``True`` values would otherwise
            pass as an identity rotation.
    """
    given = np.asarray(matrix)
    if given.dtype.kind not in REAL_NUMBER_DTYPE_KINDS:
        raise NavPointingError(
            f'{label} holds values that are not real numbers (dtype {given.dtype})',
            reason=MALFORMED_POINTING,
        )
    try:
        out = _as_readonly_3x3(given)
        _validate_rotation(out, label)
    except NavPointingError as exc:
        raise NavPointingError(str(exc), reason=MALFORMED_POINTING) from exc
    return out


def apply_cmatrix_to_obs(
    obs: ObsSnapshotInst,
    cmatrix: NDArrayAnyType,
    cmatrix_original: NDArrayAnyType,
    midtime_et: float,
) -> CmatrixApplication:
    """Point an observation at its recorded corrected attitude.

    This is the reading half of :func:`compute_pointing`: it replaces the
    observation's frame with one whose midtime attitude is the recorded
    ``cmatrix``, while the field of view is left untouched.  The replacement
    goes through ``Observation.set_spice_cmatrix``, which replaces the
    recorded matrix with the nearest exact rotation (a change of order 1e-16
    for a matrix that passed the checks below) and composes the
    ``spice_to_frame`` rotation the host declares onto it.

    ``R_hat = C_oops . cmatrix_original^T``, with ``C_oops`` the observation
    frame's own midtime attitude, is measured for the gate below and is not
    what gets applied.  The two agree: the gate passes only when ``R_hat``
    equals the declared rotation to ``_FLIP_TOL``.

    A record whose correction is the identity (``cmatrix`` equal to
    ``cmatrix_original`` as arrays) is the one case that does not go through
    the setter.  It installs ``C_oops`` itself, so no correction means exactly
    no change; composing the declared rotation onto the recorded baseline
    instead would be the same attitude but not the same float64.

    Before anything is applied, the record is gated:

    1. Both matrices must be proper rotations of real numbers and
       ``midtime_et`` finite.
    2. ``midtime_et`` must equal the observation's own midtime to a
       microsecond: the recorded attitude is a midtime attitude, so a record
       from another observation is refused rather than applied.
    3. ``R_hat`` must equal the ``spice_to_frame`` rotation the host declares,
       to the writer's own tolerance.  Because ``R_hat`` mixes the observation's
       *current* attitude with the *recorded* baseline, this one inequality
       fails on a changed kernel pool, a transposed ``cmatrix_original`` or
       whole record, and a changed host convention alike.  The one sub-case
       it cannot see is a transposed ``cmatrix`` alone, which no
       single-serializer defect produces: the inequality contains only
       ``cmatrix_original``.
    4. When that gate fails, one probe distinguishes the known non-defect
       state: a pool that already answers the corrected attitude (corrected
       kernels furnished at load time).  There the observation is already
       right and nothing is applied.

    The replacement frame is built unregistered, so batch loops pollute no
    global oops frame state, and it carries zero angular velocity where the
    original carried the spacecraft's -- no switched consumer reads frame
    omega, but a future velocity-aware one must not consume it from the
    replaced frame.  The observation's cached geometry is NOT cleared here:
    a caller that has touched any derived geometry (a ``Backplane``, the
    center scan ``from_file`` itself performs) must call ``obs.reset_all()``
    after a replacement, or rebuild products on the stale cache --
    :func:`spindoctor.cli.reproj.offsets.apply_pointing_to_obs` does exactly
    that and is the entry point the pipeline's consumers use.

    Parameters:
        obs: The observation to point, mutated in place.
        cmatrix: The recorded corrected J2000-to-camera rotation, SPICE
            convention, at the exposure midtime.  Annotated as an array of any
            dtype for the reason :func:`validated_record_rotation` gives:
            refusing the dtypes a record can carry is this function's job, so
            a caller hands it what the record held rather than casting first.
        cmatrix_original: The recorded uncorrected rotation, same convention
            and epoch, on the same terms.
        midtime_et: The recorded exposure midtime, TDB seconds past J2000.

    Returns:
        ``CmatrixApplication.FRAME_REPLACED`` when the observation's frame
        was replaced with the corrected attitude, or
        ``CmatrixApplication.POOL_ALREADY_CORRECTED`` when the furnished pool
        already answered the corrected attitude and the observation was
        deliberately left untouched -- the caller must then apply nothing
        else, in particular not the pixel offset, which would double-correct.

    Raises:
        NavPointingError: for every expected failure, carrying a
            machine-readable ``reason``: ``malformed_pointing`` when either
            matrix or the midtime is unusable, ``cmatrix_unknown_host`` when
            the observation's instrument has no frame mapping to gate
            against, ``cmatrix_foreign_midtime`` when the record belongs to
            another observation, and ``cmatrix_baseline_mismatch`` when
            ``R_hat`` fails its gate for no known non-defect reason.  The
            observation is never mutated on a raise.
    """
    corrected = validated_record_rotation(cmatrix, 'cmatrix')
    original = validated_record_rotation(cmatrix_original, 'cmatrix_original')
    if isinstance(midtime_et, bool) or not isinstance(midtime_et, int | float):
        raise NavPointingError(
            f'the recorded midtime_et is not a real number: {midtime_et!r}',
            reason=MALFORMED_POINTING,
        )
    if not math.isfinite(float(midtime_et)):
        # NaN in particular: it makes the midtime gate's inequality false in
        # both directions, which would wave a foreign record through the one
        # check that ties it to this observation.
        raise NavPointingError(
            f'the recorded midtime_et is not finite: {midtime_et!r}',
            reason=MALFORMED_POINTING,
        )
    identity = _frame_identity(obs)
    if identity is None:
        raise NavPointingError(
            f'the observation host {type(obs).__name__} has no SPICE camera frame mapping, '
            f'so no expected oops-from-SPICE flip exists to gate the recorded attitude against',
            reason=CMATRIX_UNKNOWN_HOST,
        )
    obs_midtime = float(obs.midtime)
    if abs(obs_midtime - float(midtime_et)) > _MIDTIME_TOL_S:
        raise NavPointingError(
            f'the recorded midtime_et {float(midtime_et)!r} belongs to a different observation: '
            f'this one exposes at midtime {obs_midtime!r}',
            reason=CMATRIX_FOREIGN_MIDTIME,
        )
    expected = _declared_flip(obs)
    camera_frame = _declared_camera_frame_name(obs)
    c_oops = _observation_attitude(obs, obs_midtime)
    if np.array_equal(corrected, original):
        # Two float64 matrix products do not cancel to bit precision, so
        # without this short-circuit "no correction means no change" would be
        # false at the 1e-16 level; it mirrors the writer's identity guard.
        # Going through the oops setter would compose the declared flip onto
        # the recorded baseline instead, which is the same attitude but not
        # the same float64.
        obs.set_frame(oops.frame.Cmatrix(np.asarray(c_oops, dtype=np.float64)))
        return CmatrixApplication.FRAME_REPLACED
    r_hat = np.asarray(c_oops, dtype=np.float64) @ original.T
    if not np.allclose(r_hat, expected, rtol=0.0, atol=_FLIP_TOL):
        if np.allclose(c_oops, expected @ corrected, rtol=0.0, atol=_FLIP_TOL):
            # The pool already answers the corrected attitude -- corrected
            # kernels furnished at load time.  The observation is already
            # right; applying the correction again, or the offset, would
            # move it by roughly twice the navigated offset.
            return CmatrixApplication.POOL_ALREADY_CORRECTED
        raise NavPointingError(
            f'the rotation between the observation frame and the recorded '
            f'{camera_frame} baseline is {r_hat.tolist()!r}, which differs from the '
            f'{np.asarray(expected).tolist()!r} the host declares by up to '
            f'{float(np.max(np.abs(r_hat - expected)))!r}; the kernel pool, the record, or '
            f'the host convention has changed since navigation',
            reason=CMATRIX_BASELINE_MISMATCH,
        )
    obs.set_spice_cmatrix(corrected)
    return CmatrixApplication.FRAME_REPLACED


def _ck_object_sclk_id(ck_frame_id: int) -> int:
    """Return the spacecraft clock a CK object's time tags are encoded against.

    The value is read from the shared mapping the C-kernel writer reads too,
    so that the clock a correction is timed with and the clock it is written
    with cannot disagree.

    Parameters:
        ck_frame_id: SPICE id of the object a corrected C-kernel targets.

    Returns:
        The spacecraft clock id recorded for that object.

    Raises:
        NavPointingError: if no clock is recorded for the object.
    """
    if ck_frame_id not in CK_OBJECT_SCLK_ID:
        raise NavPointingError(
            f'CK object {ck_frame_id} has no recorded spacecraft clock; expected one of '
            f'{sorted(CK_OBJECT_SCLK_ID)}'
        )
    return CK_OBJECT_SCLK_ID[ck_frame_id]


def _frame_identity(obs: ObsSnapshotInst) -> _FrameIdentity | None:
    """Return the SPICE frame facts for an observation's instrument.

    Parameters:
        obs: The observation to identify.

    Returns:
        The instrument's ``_FrameIdentity``, or ``None`` for a host with no
        SPICE camera frame this module knows.

    Raises:
        NavPointingError: if the instrument's CK object has no recorded
            spacecraft clock.
        ValueError: if a Voyager observation's label names a spacecraft that
            is neither Voyager, which the host refuses when it reads the
            label, so reaching it here means that stopped being true.
    """
    if isinstance(obs, ObsCassiniISS):
        return _FrameIdentity(
            ck_frame_id=_CASSINI_CK_FRAME_ID,
            sclk_id=_ck_object_sclk_id(_CASSINI_CK_FRAME_ID),
            frozen_oops_attitude=False,
        )
    if isinstance(obs, ObsGalileoSSI):
        return _FrameIdentity(
            ck_frame_id=_GALILEO_CK_FRAME_ID,
            sclk_id=_ck_object_sclk_id(_GALILEO_CK_FRAME_ID),
            frozen_oops_attitude=False,
        )
    if isinstance(obs, ObsNewHorizonsLORRI):
        return _FrameIdentity(
            ck_frame_id=_LORRI_CK_FRAME_ID,
            sclk_id=_ck_object_sclk_id(_LORRI_CK_FRAME_ID),
            frozen_oops_attitude=False,
        )
    if isinstance(obs, ObsVoyagerISS):
        digit = obs.spacecraft_digit
        # One instrument key serves two spacecraft, so the CK object is the one
        # recorded for this spacecraft; its clock then follows from the object
        # rather than from the same digit, so a wrong pairing is refused
        # instead of producing a self-consistent wrong one.  The host validates
        # the digit when it reads the label, so a key error here would mean
        # that stopped being true.
        ck_frame_id = VOYAGER_CK_OBJECT_ID[digit]
        return _FrameIdentity(
            ck_frame_id=ck_frame_id,
            sclk_id=_ck_object_sclk_id(ck_frame_id),
            frozen_oops_attitude=True,
        )
    return None


def _spice_attitude(obs: ObsSnapshotInst, et: float) -> NDArrayFloatType:
    """Return the observation's attitude at one epoch in the SPICE convention.

    This is ``oops.Observation.get_spice_cmatrix``, which composes the
    observation frame's own attitude with the inverse of the host's declared
    ``spice_to_frame`` rotation.  It answers uniformly for a frame oops
    evaluates from a chain and for one it froze from a pointing lookup, so no
    caller here has to distinguish the two.

    Parameters:
        obs: The observation whose attitude is read.
        et: TDB seconds past J2000.

    Returns:
        The 3x3 J2000-to-camera rotation in the SPICE convention.

    Raises:
        NavPointingError: if the furnished kernels cannot place the frame at
            this epoch.
    """
    try:
        matrix = obs.get_spice_cmatrix(time=et)
    except _SPICE_FAILURES as exc:
        raise NavPointingError(
            f'the observation frame has no attitude at et {et!r}: {exc}'
        ) from exc
    return _as_readonly_3x3(np.asarray(matrix.vals, dtype=np.float64))


def _declared_camera_frame_name(obs: ObsSnapshotInst) -> str:
    """Return the name of the SPICE camera frame the host declares.

    Reads the ``spice_frame_name`` subfield and nothing else, so a caller that
    only has to name the frame -- in a refusal message, say -- needs no kernel
    pool to do it.

    Parameters:
        obs: The observation to read.

    Returns:
        The SPICE name of the frame the observation was built on.
    """
    return str(obs.spice_frame_name)


def _declared_camera_frame(obs: ObsSnapshotInst) -> tuple[str, int]:
    """Return the SPICE camera frame the host declares for this observation.

    oops inserts the name and the id as the ``spice_frame_name`` and
    ``spice_frame_id`` subfields.  Reading them rather than spelling the name
    here means the frame this module evaluates and the frame oops built the
    observation on cannot name different cameras -- the confusion that puts a
    wide-angle attitude in the narrow-angle frame.

    The two subfields are cross-checked against the kernel pool, because they
    are one fact stated twice and a table can be edited on one side only.

    Parameters:
        obs: The observation to read.

    Returns:
        The frame's SPICE name and its id.

    Raises:
        NavPointingError: if the name resolves to no id in the furnished pool,
            or if the id it resolves to is not the one declared.
    """
    name = _declared_camera_frame_name(obs)
    declared = int(obs.spice_frame_id)
    try:
        resolved = int(cspyce.namfrm(name))
    except _SPICE_FAILURES as exc:
        raise NavPointingError(f'the frame {name} has no SPICE id: {exc}') from exc
    if resolved != declared:
        raise NavPointingError(
            f'the host declares frame {name} as id {declared}, but the furnished kernels '
            f'resolve that name to {resolved}; the two disagree about which frame the '
            f'observation was built on'
        )
    return name, declared


def _declared_flip(obs: ObsSnapshotInst) -> NDArrayFloatType:
    """Return the constant rotation ``R`` the host declares for this observation.

    oops inserts it as the ``spice_to_frame`` subfield, the rotation from the
    SPICE frame convention of the instrument to the oops convention, and reads
    the same subfield in ``Observation.get_spice_cmatrix``.  Taking it from
    there rather than from a table here means the attitude this module records
    and the frame oops built cannot disagree about the convention.

    Parameters:
        obs: The observation to read.

    Returns:
        The 3x3 rotation as a read-only float64 array.

    Raises:
        NavPointingError: if the declared rotation is not a proper orthonormal
            rotation.
    """
    flip = _as_readonly_3x3(np.asarray(obs.spice_to_frame.vals, dtype=np.float64))
    _validate_rotation(flip, 'spice_to_frame')
    return flip


def _observation_attitude(obs: ObsSnapshotInst, et: float) -> NDArrayFloatType:
    """Return the observation frame's J2000-to-frame rotation at one epoch.

    Parameters:
        obs: The observation whose frame is evaluated.
        et: TDB seconds past J2000.

    Returns:
        The 3x3 rotation in the oops observation frame convention.

    Raises:
        NavPointingError: if the furnished kernels cannot place the frame at
            this epoch.
    """
    try:
        transform = obs.frame.wrt(oops.frame.Frame.J2000).transform_at_time(et)
    except _SPICE_FAILURES as exc:
        raise NavPointingError(
            f'the observation frame has no attitude at et {et!r}: {exc}'
        ) from exc
    # A shape check rather than a reshape: a reshape would also accept a flat
    # nine-element array of any rank that a changed oops return could supply.
    return _as_readonly_3x3(np.asarray(transform.matrix.vals))


def _attitude_baseline(obs: ObsSnapshotInst, identity: _FrameIdentity) -> AttitudeBaseline:
    """Read the uncorrected attitude, frame ids, and exposure times from SPICE.

    Parameters:
        obs: The observation to read.
        identity: The instrument's frame facts.

    Returns:
        The observation's AttitudeBaseline.

    Raises:
        NavPointingError: if the furnished kernels cannot supply the attitude,
            the camera frame id or the clock strings; if the resolved
            spacecraft clock is not the instrument's; or if the measured flip
            between the oops observation frame and the SPICE camera frame
            differs from the instrument's expected constant or varies across
            the exposure.
    """
    start_et = float(obs.time[0])
    stop_et = float(obs.time[1])
    midtime_et = float(obs.midtime)
    exposure_s = float(obs.texp)
    # The epochs pass through the spacecraft clock conversion, which refuses a
    # non-finite value; the exposure duration passes through nothing before it
    # is serialized, and a bare NaN token in the metadata fails every strict
    # JSON reader long after the defect is attributable.  Refuse all four here,
    # where the observation that carried them is still in hand.
    for label, value in (
        ('start', start_et),
        ('stop', stop_et),
        ('midtime', midtime_et),
        ('exposure duration', exposure_s),
    ):
        if not math.isfinite(value):
            raise NavPointingError(f'the observation records a non-finite {label}: {value!r}')
    oops_from_spice = _declared_flip(obs)
    camera_frame, camera_frame_id = _declared_camera_frame(obs)
    cmatrix_original = _spice_attitude(obs, midtime_et)
    if not identity.frozen_oops_attitude:
        # The declared flip is oops's claim about its own frame, not a
        # measurement, so it is checked against the frame chain SPICE actually
        # builds -- at every epoch of the exposure, since a flip that is right
        # at the midtime and wrong at the edges is a moving frame misdeclared
        # as a constant one.
        for et in (start_et, midtime_et, stop_et):
            measured = _observation_attitude(obs, et) @ _pxform(camera_frame, et).T
            _check_flip(measured, oops_from_spice, camera_frame, et)
    sclk_id = _sclk_id(identity, camera_frame)
    return AttitudeBaseline(
        cmatrix_original=cmatrix_original,
        oops_from_spice=oops_from_spice,
        camera_frame=camera_frame,
        camera_frame_id=camera_frame_id,
        ck_frame_id=identity.ck_frame_id,
        start_et=start_et,
        stop_et=stop_et,
        midtime_et=midtime_et,
        exposure_s=exposure_s,
        sclk_start=_sclk_string(sclk_id, start_et),
        sclk_midtime=_sclk_string(sclk_id, midtime_et),
        sclk_stop=_sclk_string(sclk_id, stop_et),
    )


def _sclk_id(identity: _FrameIdentity, camera_frame: str) -> int:
    """Resolve the spacecraft clock for a CK object and check it is the right one.

    ``cspyce.ckmeta`` computes a clock id from a CK object id rather than
    validating it: it answers ``-999`` for the nonexistent object ``-999999``
    and ``-12`` for ``-12345``, raising for neither.  A CK object that is
    wrong for any reason therefore yields a plausible clock id, and every
    clock string built from it encodes the wrong spacecraft's time while
    ``sce2s`` reports success.  The resolved id is checked against the one the
    mission actually uses before any clock string is built.

    Parameters:
        identity: The instrument's frame facts, carrying both the CK object
            and the spacecraft clock expected for it.
        camera_frame: SPICE name of the camera frame, for the message.

    Returns:
        The resolved spacecraft clock id, equal to ``identity.sclk_id``.

    Raises:
        NavPointingError: if no clock resolves for the CK object, or if the
            one that resolves is not the instrument's.
    """
    try:
        resolved = int(cspyce.ckmeta(identity.ck_frame_id, 'SCLK'))
    except _SPICE_FAILURES as exc:
        raise NavPointingError(
            f'no spacecraft clock resolves for CK object {identity.ck_frame_id}: {exc}'
        ) from exc
    if resolved != identity.sclk_id:
        raise NavPointingError(
            f'CK object {identity.ck_frame_id} resolves to spacecraft clock {resolved}, not the '
            f'{identity.sclk_id} the {camera_frame} camera is tagged against; every '
            f'clock string built from it would encode the wrong spacecraft'
        )
    # The recorded id is returned, not the one ``ckmeta`` computed, even though
    # the check above has just proved them equal.  ``ckmeta`` answers for
    # objects that do not exist, so it is a cross-check here and never the
    # source: if this check is ever weakened, the clock strings still come from
    # the recorded table rather than from whatever ``ckmeta`` returned.
    return identity.sclk_id


def _sclk_string(sclk_id: int, et: float) -> str:
    """Encode one epoch as a spacecraft clock string.

    Parameters:
        sclk_id: SPICE id of the spacecraft clock.
        et: TDB seconds past J2000.

    Returns:
        The clock string ``sce2s`` produces for that epoch.

    Raises:
        NavPointingError: if the furnished kernels cannot encode the epoch.
    """
    try:
        return str(cspyce.sce2s(sclk_id, et))
    except _SPICE_FAILURES as exc:
        raise NavPointingError(
            f'spacecraft clock {sclk_id} cannot encode et {et!r}: {exc}'
        ) from exc


def _pxform(camera_frame: str, et: float) -> NDArrayFloatType:
    """Evaluate the J2000-to-camera rotation from the furnished kernels.

    Parameters:
        camera_frame: SPICE name of the camera frame.
        et: TDB seconds past J2000.

    Returns:
        The 3x3 rotation ``pxform('J2000', camera_frame, et)`` as float64.

    Raises:
        NavPointingError: if the furnished kernels cannot supply the rotation
            at this epoch.
    """
    try:
        matrix = cspyce.pxform('J2000', camera_frame, et)
    except _SPICE_FAILURES as exc:
        raise NavPointingError(
            f'the furnished kernels cannot supply the J2000 to {camera_frame} rotation at '
            f'et {et!r}: {exc}'
        ) from exc
    return np.asarray(matrix, dtype=np.float64)


def _check_flip(
    measured: NDArrayFloatType,
    expected: NDArrayFloatType,
    camera_frame: str,
    et: float,
) -> None:
    """Raise unless the measured oops-to-SPICE flip is the declared constant.

    Parameters:
        measured: The rotation ``R`` measured from the observation and the
            furnished kernels at ``et``.
        expected: The rotation the host declared as ``spice_to_frame``.
        camera_frame: SPICE name of the camera frame, for the message.
        et: The epoch the measurement was made at, for the message.

    Raises:
        NavPointingError: if the two differ by more than ``_FLIP_TOL``.
    """
    if not np.allclose(measured, expected, rtol=0.0, atol=_FLIP_TOL):
        raise NavPointingError(
            f'at et {et!r} the rotation between the oops observation frame and the SPICE '
            f'{camera_frame} frame is {np.asarray(measured).tolist()!r}, which differs from '
            f'the {np.asarray(expected).tolist()!r} the host declares by up to '
            f'{float(np.max(np.abs(measured - expected)))!r}'
        )
