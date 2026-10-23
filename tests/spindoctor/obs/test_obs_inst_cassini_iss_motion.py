"""Tests for ``spindoctor.obs.obs_inst_cassini_iss_motion``.

The hermetic tests write a predicted C-kernel for the Cassini bus and give the
observation a reconstructed attitude history of its own, turning about another axis at
another rate, so a motion taken from the wrong one cannot pass.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import cspyce
import julian
import numpy as np
import polymath
import pytest
import spicedb
from tests.config import (
    REQUIRES_EXTERNAL_DATA,
    URL_CASSINI_ISS_CRUISE_02,
    URL_CASSINI_ISS_RHEA_01,
)
from tests.kernel_pool import isolated_kernel_pool
from tests.spindoctor.cli.ck.ck_helpers import ET0, TICKS_PER_SECOND, write_support_kernels

from spindoctor.obs.obs_inst_cassini_iss import ObsCassiniISS
from spindoctor.obs.obs_inst_cassini_iss_motion import (
    PREDICTED_SOURCE_PREFIX,
    RECONSTRUCTED_RATE_SOURCE,
    cassini_exposure_motion,
    predicted_ck_paths,
)
from spindoctor.obs.obs_snapshot import ObsSnapshot
from spindoctor.support.exposure_motion import motion_from_attitudes, motion_from_rate
from spindoctor.support.types import NDArrayFloatType

_BUS = -82000
"""The Cassini bus, whose attitude the predicted kernels give."""

_SETTINGS = {
    'spacecraft_id': -82,
    'ck_object_id': _BUS,
    'predicted_ck_name': 'CAS-CK-PREDICTED',
    'predicted_ck_version': 'V02',
    'predicted_max_streak_to_predicted': 1.5,
    'reconstructed_half_window_s': 8.0,
}
"""The configuration section, as packaged."""

_EXPOSURE = (ET0 - 0.6, ET0 + 0.6)
"""The test exposure, TDB seconds past J2000."""


def _rotation(axis: tuple[float, float, float], angle: float) -> NDArrayFloatType:
    """Return the rotation of ``angle`` radians about ``axis``."""
    unit = np.asarray(axis, dtype=np.float64) / float(np.linalg.norm(axis))
    return np.asarray(cspyce.axisar(unit, angle), dtype=np.float64)


def _predicted(et: float) -> NDArrayFloatType:
    """Return the as-flown attitude: a steady 8 microradians per second."""
    return _rotation((0.2, -0.6, 0.77), 8.0e-6 * (et - ET0)) @ _rotation((1.0, 0.4, -0.3), 0.9)


def _reconstructed(et: float) -> NDArrayFloatType:
    """Return the reconstructed attitude: another axis, another rate, another orientation."""
    return _rotation((0.9, 0.1, 0.3), 2.0e-5 * (et - ET0)) @ _rotation((1.0, 0.4, -0.3), 0.9001)


def _write_predicted_ck(
    path: Path, start_et: float, stop_et: float, attitude: Any = _predicted
) -> None:
    """Write a predicted C-kernel for the bus with a record every 6 s."""
    epochs = np.arange(start_et, stop_et + 1e-9, 6.0)
    handle = cspyce.ckopn(str(path), 'predicted', 0)
    try:
        cspyce.ckw03(
            handle,
            epochs[0] * TICKS_PER_SECOND,
            epochs[-1] * TICKS_PER_SECOND,
            _BUS,
            'J2000',
            False,
            'predicted',
            epochs * TICKS_PER_SECOND,
            np.array([cspyce.m2q(attitude(float(et))) for et in epochs]),
            np.zeros((len(epochs), 3)),
            1,
            [epochs[0] * TICKS_PER_SECOND],
        )
    finally:
        cspyce.ckcls(handle)


@dataclass
class _Matrix:
    """What ``get_spice_cmatrix`` returns: an object carrying ``vals``."""

    vals: NDArrayFloatType


class _FakeObs:
    """Observation stand-in exposing the exposure times and the reconstructed attitude.

    Parameters:
        attitude: The J2000-to-bus rotation at an epoch, or None for an observation
            whose pointing cannot be evaluated away from the exposure.
    """

    def __init__(self, attitude: Any, exposure: tuple[float, float] = _EXPOSURE) -> None:
        self.time = exposure
        self.midtime = 0.5 * (exposure[0] + exposure[1])
        self._attitude = attitude

    def get_spice_cmatrix(self, *, time: float) -> _Matrix:
        """Return the reconstructed attitude, or fail outside the exposure without one."""
        if self._attitude is None:
            raise OSError('SPICE(CKINSUFFDATA) -- insufficient C kernel data')
        return _Matrix(self._attitude(time))


@pytest.fixture
def clock(tmp_path: Path) -> Iterator[None]:
    """Furnish the hermetic leapseconds and clock kernels in an emptied pool."""
    with isolated_kernel_pool():
        paths = write_support_kernels(tmp_path / 'support')
        for path in paths:
            cspyce.furnsh(str(path))
        try:
            yield
        finally:
            for path in reversed(paths):
                cspyce.unload(str(path))


def _obs(attitude: Any, exposure: tuple[float, float] = _EXPOSURE) -> ObsSnapshot:
    """Return the stand-in typed as the observation it stands in for."""
    return cast(ObsSnapshot, _FakeObs(attitude, exposure))


def test_predicted_kernel_covering_the_exposure_supplies_the_motion(
    tmp_path: Path, clock: None
) -> None:
    """A predicted kernel covering the exposure gives the motion between its attitudes."""
    kernel = tmp_path / 'covering.bc'
    _write_predicted_ck(kernel, ET0 - 60.0, ET0 + 60.0)
    motion = cassini_exposure_motion(_obs(_reconstructed), _SETTINGS, kernel_paths=[kernel])
    expected = motion_from_attitudes(
        _predicted(_EXPOSURE[0]), _predicted(ET0), _predicted(_EXPOSURE[1]), source=''
    )
    assert motion is not None
    assert motion.source == PREDICTED_SOURCE_PREFIX + 'covering.bc'
    assert motion.max_streak_to_predicted == 1.5
    np.testing.assert_allclose(motion.end_rotation, expected.end_rotation, rtol=0.0, atol=1e-14)


def test_reconstructed_rate_supplies_the_motion_where_no_kernel_covers(
    tmp_path: Path, clock: None
) -> None:
    """Without a covering predicted kernel the motion is the pointing's rate over 16 s."""
    kernel = tmp_path / 'elsewhere.bc'
    _write_predicted_ck(kernel, ET0 + 100.0, ET0 + 200.0)
    motion = cassini_exposure_motion(_obs(_reconstructed), _SETTINGS, kernel_paths=[kernel])
    expected = motion_from_rate(
        _reconstructed(ET0 - 8.0),
        _reconstructed(ET0),
        _reconstructed(ET0 + 8.0),
        window_s=16.0,
        exposure_s=_EXPOSURE[1] - _EXPOSURE[0],
        source='',
    )
    assert motion is not None
    assert motion.source == RECONSTRUCTED_RATE_SOURCE.format(window=16.0)
    assert motion.max_streak_to_predicted is None
    np.testing.assert_allclose(motion.end_rotation, expected.end_rotation, rtol=0.0, atol=1e-14)


def test_kernel_covering_only_part_of_the_exposure_does_not_supply_it(
    tmp_path: Path, clock: None
) -> None:
    """A predicted kernel ending before the exposure does falls back to the reconstructed rate."""
    kernel = tmp_path / 'partial.bc'
    _write_predicted_ck(kernel, ET0 - 60.0, ET0 + 0.3)
    motion = cassini_exposure_motion(_obs(_reconstructed), _SETTINGS, kernel_paths=[kernel])
    assert motion is not None
    assert motion.source == RECONSTRUCTED_RATE_SOURCE.format(window=16.0)


def test_later_predicted_kernel_takes_precedence(tmp_path: Path, clock: None) -> None:
    """Of two predicted kernels covering the exposure, the later one supplies the motion."""
    earlier, later = tmp_path / 'earlier.bc', tmp_path / 'later.bc'
    _write_predicted_ck(earlier, ET0 - 60.0, ET0 + 60.0, attitude=_reconstructed)
    _write_predicted_ck(later, ET0 - 60.0, ET0 + 60.0)
    motion = cassini_exposure_motion(_obs(_reconstructed), _SETTINGS, kernel_paths=[earlier, later])
    expected = motion_from_attitudes(
        _predicted(_EXPOSURE[0]), _predicted(ET0), _predicted(_EXPOSURE[1]), source=''
    )
    assert motion is not None
    assert motion.source == PREDICTED_SOURCE_PREFIX + 'later.bc'
    np.testing.assert_allclose(motion.end_rotation, expected.end_rotation, rtol=0.0, atol=1e-14)


def test_exposure_as_long_as_the_window_uses_its_own_pointing(clock: None) -> None:
    """An uncovered exposure of 16 s or more spans many records, so no motion is supplied."""
    exposure = (ET0 - 8.0, ET0 + 8.0)
    assert (
        cassini_exposure_motion(_obs(_reconstructed, exposure), _SETTINGS, kernel_paths=[]) is None
    )


def test_no_motion_where_neither_kernel_set_covers(clock: None) -> None:
    """Without a predicted kernel or a reconstructed window there is no motion to supply."""
    assert cassini_exposure_motion(_obs(None), _SETTINGS, kernel_paths=[]) is None


def test_reading_the_predicted_kernel_leaves_the_pointing_unchanged(
    tmp_path: Path, clock: None
) -> None:
    """The pool's attitude is bit-identical after the motion is read from another kernel."""
    reconstructed = tmp_path / 'reconstructed.bc'
    handle = cspyce.ckopn(str(reconstructed), 'reconstructed', 0)
    epochs = np.arange(ET0 - 60.0, ET0 + 60.5, 4.0)
    try:
        cspyce.ckw03(
            handle,
            epochs[0] * TICKS_PER_SECOND,
            epochs[-1] * TICKS_PER_SECOND,
            _BUS,
            'J2000',
            False,
            'reconstructed',
            epochs * TICKS_PER_SECOND,
            np.array([cspyce.m2q(_reconstructed(float(et))) for et in epochs]),
            np.zeros((len(epochs), 3)),
            1,
            [epochs[0] * TICKS_PER_SECOND],
        )
    finally:
        cspyce.ckcls(handle)
    predicted = tmp_path / 'predicted.bc'
    _write_predicted_ck(predicted, ET0 - 60.0, ET0 + 60.0)
    cspyce.furnsh(str(reconstructed))
    try:
        ticks = [et * TICKS_PER_SECOND for et in (*_EXPOSURE, ET0)]
        before = [np.asarray(cspyce.ckgp(_BUS, tick, 0.0, 'J2000')[0]) for tick in ticks]
        count = cspyce.ktotal('ALL')
        motion = cassini_exposure_motion(_obs(_reconstructed), _SETTINGS, kernel_paths=[predicted])
        after = [np.asarray(cspyce.ckgp(_BUS, tick, 0.0, 'J2000')[0]) for tick in ticks]
        assert cspyce.ktotal('ALL') == count
        assert all(np.array_equal(a, b) for a, b in zip(after, before, strict=True))
        assert motion is not None
        assert motion.source == PREDICTED_SOURCE_PREFIX + 'predicted.bc'
    finally:
        cspyce.unload(str(reconstructed))


@dataclass
class _Kernel:
    """A SPICE database row: the release and the file of one kernel."""

    kernel_version: str
    filespec: str


class _Prefix:
    """A file cache prefix whose ``missing`` files cannot be retrieved."""

    def __init__(self, root: Path, missing: set[str]) -> None:
        self._root = root
        self._missing = missing

    def retrieve(self, filespec: str, *, exception_on_fail: bool) -> Path | Exception:
        """Return the local path, or the failure for a missing file."""
        if filespec in self._missing:
            return FileNotFoundError(filespec)
        return self._root / filespec


def test_predicted_ck_paths_selects_each_epoch_of_one_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each epoch is selected on its own, other releases and unretrievable files left out."""
    listed = {
        1.0: [_Kernel('V01', 'old/a.bc'), _Kernel('V02', 'new/a.bc')],
        2.0: [_Kernel('V02', 'new/a.bc'), _Kernel('V02', 'new/gone.bc')],
        3.0: [_Kernel('V02', 'new/b.bc')],
    }

    def select_ck(_spacecraft: int, *, name: str, time: tuple[float, float]) -> list[_Kernel]:
        if time[0] not in listed:
            raise ValueError('no results found matching query')
        return listed[time[0]]

    monkeypatch.setattr(spicedb, 'db_is_open', lambda: True)
    monkeypatch.setattr(spicedb, 'open_db', lambda: None)
    monkeypatch.setattr(spicedb, 'select_ck', select_ck)
    monkeypatch.setattr(
        spicedb,
        'get_spice_filecache_prefix',
        lambda: _Prefix(tmp_path, {'new/gone.bc'}),
    )
    monkeypatch.setattr(julian, 'tai_from_tdb', lambda et: et)
    paths = predicted_ck_paths(
        [1.0, 2.0, 3.0, 4.0],
        spacecraft_id=-82,
        kernel_name='CAS-CK-PREDICTED',
        kernel_version='V02',
    )
    assert paths == [tmp_path / 'new/a.bc', tmp_path / 'new/b.bc']


def _pointing(obs: ObsCassiniISS) -> list[NDArrayFloatType]:
    """Return the observation's attitude at its exposure's start, middle, and end."""
    return [
        np.asarray(obs.get_spice_cmatrix(time=et).vals)
        for et in (obs.time[0], obs.midtime, obs.time[1])
    ]


@REQUIRES_EXTERNAL_DATA
def test_tour_image_motion_comes_from_the_predicted_kernel_and_keeps_its_pointing() -> None:
    """A tour image reads the as-flown kernel and keeps its attitude and star positions."""
    obs = ObsCassiniISS.from_file(URL_CASSINI_ISS_RHEA_01)
    before = _pointing(obs)
    center_ra, center_dec = obs.center_ra_dec(apparent=True)
    uv_before = obs.uv_from_ra_and_dec(polymath.Scalar(center_ra), polymath.Scalar(center_dec))
    motion = obs.exposure_motion()
    uv_after = obs.uv_from_ra_and_dec(polymath.Scalar(center_ra), polymath.Scalar(center_dec))
    assert motion is not None
    assert motion.source.startswith(PREDICTED_SOURCE_PREFIX)
    assert all(np.array_equal(a, b) for a, b in zip(_pointing(obs), before, strict=True))
    assert np.array_equal(np.asarray(uv_after.vals), np.asarray(uv_before.vals))


@REQUIRES_EXTERNAL_DATA
def test_late_cruise_image_motion_comes_from_the_reconstructed_rate() -> None:
    """An image from before the as-flown kernels begin falls back to the reconstructed rate."""
    obs = ObsCassiniISS.from_file(URL_CASSINI_ISS_CRUISE_02)
    motion = obs.exposure_motion()
    assert motion is not None
    assert motion.source == RECONSTRUCTED_RATE_SOURCE.format(window=16.0)
