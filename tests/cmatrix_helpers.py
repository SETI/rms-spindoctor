"""Independent inverse helpers shared by the C-matrix tests.

The hermetic and the real-frame C-matrix tests both check the recorded
attitude by recovering the planted offset back out of it.  The recovery is
written from the offset model itself, using only oops FOV primitives, and
imports nothing from the forward path under test: that independence is what
lets it catch a sign, conjugation or composition error.  A helper that
called the implementation would agree with it whatever it did.
"""

from __future__ import annotations

import numpy as np
import oops

from spindoctor.obs import ObsSnapshotInst
from spindoctor.support.cmatrix import AttitudeBaseline, _FrameIdentity


def offset_from_correction(fov: oops.fov.FOV, correction: np.ndarray) -> tuple[float, float]:
    """Recover the ``(dv, du)`` offset an oops-frame correction rotation encodes.

    Maps the corrected boresight direction back through the rotation, reads
    off the tangent-plane point that direction came from, and converts the
    implied ``xy_offset`` back into pixels.

    Parameters:
        fov: The observation's unmodified oops FOV.
        correction: The 3x3 correction rotation in oops observation frame
            coordinates.

    Returns:
        The ``(dv, du)`` offset in pixels that produces this correction.
    """
    # Every concrete FOV states its line of sight; the FOV base class declares none.
    uv_los: oops.Pair = fov.uv_los  # type: ignore[attr-defined]
    los_u, los_v = np.asarray(uv_los.vals, np.float64)
    xy_los = fov.xy_from_uv(uv_los)
    corrected = np.asarray(fov.los_from_xy(xy_los).unit().vals, np.float64)
    uncorrected = np.asarray(correction, np.float64).T @ corrected
    xy_uncorrected = oops.Pair((uncorrected[0] / uncorrected[2], uncorrected[1] / uncorrected[2]))
    uv_u, uv_v = np.asarray(fov.uv_from_xy(xy_los - xy_uncorrected).vals, np.float64)
    return (float(uv_v - los_v), float(uv_u - los_u))


def observation_attitude(obs: ObsSnapshotInst, et: float) -> np.ndarray:
    """Read the oops observation frame's J2000-to-frame rotation at one epoch.

    Parameters:
        obs: The observation whose frame is evaluated.
        et: TDB seconds past J2000.

    Returns:
        The 3x3 rotation in the oops observation frame convention.
    """
    transform = obs.frame.wrt(oops.frame.Frame.J2000).transform_at_time(et)
    matrix = np.asarray(transform.matrix.vals, np.float64)
    # A shape assertion rather than a reshape: reshape would silently accept a
    # flat nine-element array of any rank that a changed oops return could
    # supply, and the helper must fail loudly instead.
    assert matrix.shape == (3, 3)
    return matrix


# A Cassini-NAC-like square camera: 1024 pixels at 6 microradians each.  A
# FlatFOV maps its boresight pixel to xy exactly (0, 0), which is what makes
# the zero-offset identity guard exercisable.
PIXEL_RAD = 6.0e-6
SHAPE = (1024, 1024)

# The synthetic exposure window every baseline built here records.  Tests
# that hand the reader an epoch import these rather than repeating the
# literals, so the record and the gate cannot drift apart.
SYNTHETIC_START_ET = 100.0
SYNTHETIC_MIDTIME_ET = 100.25
SYNTHETIC_STOP_ET = 100.5


def synthetic_fov() -> oops.fov.FlatFOV:
    """Build the synthetic camera the hermetic C-matrix tests point.

    Returns:
        A flat square FOV of :data:`SHAPE` pixels at :data:`PIXEL_RAD` radians
        each, whose boresight pixel maps to xy exactly ``(0, 0)``.
    """
    return oops.fov.FlatFOV((PIXEL_RAD, PIXEL_RAD), SHAPE)


def some_attitude() -> np.ndarray:
    """Return an arbitrary, deliberately non-axis-aligned J2000-to-camera rotation.

    Returns:
        A 3x3 proper rotation, fixed across calls, whose every element is
        nonzero so a dropped or transposed axis cannot pass unnoticed.
    """
    ra, dec, twist = 0.7, -0.4, 1.9
    rot_z = np.array(
        [[np.cos(ra), np.sin(ra), 0.0], [-np.sin(ra), np.cos(ra), 0.0], [0.0, 0.0, 1.0]]
    )
    rot_y = np.array(
        [[np.cos(dec), 0.0, -np.sin(dec)], [0.0, 1.0, 0.0], [np.sin(dec), 0.0, np.cos(dec)]]
    )
    rot_x = np.array(
        [[1.0, 0.0, 0.0], [0.0, np.cos(twist), np.sin(twist)], [0.0, -np.sin(twist), np.cos(twist)]]
    )
    attitude: np.ndarray = rot_x @ rot_y @ rot_z
    return attitude


def synthetic_baseline(
    cmatrix_original: np.ndarray, *, oops_from_spice: np.ndarray | None = None
) -> AttitudeBaseline:
    """Build a synthetic AttitudeBaseline around a given attitude and flip.

    Parameters:
        cmatrix_original: The uncorrected J2000-to-camera rotation.
        oops_from_spice: The flip between the oops and SPICE frames; None for
            the identity.

    Returns:
        The baseline, at a fixed synthetic epoch.
    """
    return AttitudeBaseline(
        camera_frame='TEST_CAMERA',
        cmatrix_original=cmatrix_original,
        oops_from_spice=oops_from_spice if oops_from_spice is not None else np.eye(3),
        camera_frame_id=-999999,
        ck_frame_id=-999000,
        start_et=SYNTHETIC_START_ET,
        stop_et=SYNTHETIC_STOP_ET,
        midtime_et=SYNTHETIC_MIDTIME_ET,
        exposure_s=SYNTHETIC_STOP_ET - SYNTHETIC_START_ET,
        sclk_start='1/100.000',
        sclk_midtime='1/100.250',
        sclk_stop='1/100.500',
    )


def synthetic_frame_identity(flip: np.ndarray) -> _FrameIdentity:
    """Build an instrument identity for a synthetic instrument.

    Injected in place of the real instrument table so hermetic tests can gate
    against any flip, including the non-involutory ones no real instrument
    has.  The flip itself now travels on the observation, as the host declares
    it, so this takes it only to keep one call shape for the tests.

    Parameters:
        flip: Unused; the observation carries the flip.

    Returns:
        The identity, with placeholder frame and clock ids.
    """
    del flip
    return _FrameIdentity(
        ck_frame_id=-999000,
        sclk_id=-999,
        frozen_oops_attitude=False,
    )
