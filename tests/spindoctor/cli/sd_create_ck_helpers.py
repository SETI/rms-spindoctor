"""Shared builders and drivers for the ``sd_create_ck`` end-to-end tests.

The driver is exercised from two test files -- one asserting the products of a
clean run, one asserting its refusals and exit statuses -- and both build the
same runs: a kernel directory holding the hermetic support kernels and two
baselines, a results root of per-image metadata documents, and a command line
pointed at them.  Those builders live here, in a plain module, so neither test
file imports the other; the fixtures that hand out prepared run trees live in
the package's ``conftest.py``.
"""

import csv
import json
from pathlib import Path
from typing import Any

import cspyce
import numpy as np
import pytest
from tests.spindoctor.cli.ck.ck_helpers import (
    CASSINI_CAMERA_FRAME,
    CASSINI_CK_FRAME_ID,
    ET0,
    baseline_angular_velocity,
    baseline_attitude,
    image_metadata,
    write_baseline_ck,
    write_support_kernels,
)

from spindoctor.cli import sd_create_ck
from spindoctor.support.types import NDArrayFloatType

# The two exposures are far enough apart that no original kernel covers both,
# so each image has exactly one candidate and the two land in different files.
IMAGE_A_ET = ET0
IMAGE_B_ET = ET0 + 1000.0
EXPOSURE_S = 2.0

CASSINI_SCLK_ID = -82

BASELINE_A = 'orig_a.bc'
BASELINE_B = 'orig_b.bc'
KERNEL_NAMES = ('test.tf', 'test.tls', 'test.tsc', BASELINE_A, BASELINE_B)


def camera_attitude(et: float) -> NDArrayFloatType:
    """Return the camera attitude the baseline kernels give at one epoch.

    Parameters:
        et: TDB seconds past J2000.

    Returns:
        The 3x3 J2000-to-camera rotation, which is what the metadata records
        as the uncorrected attitude.
    """
    attitude: NDArrayFloatType = np.asarray(
        cspyce.pxform('J2000', CASSINI_CAMERA_FRAME, et), dtype=np.float64
    )
    return attitude


def corrected(attitude: NDArrayFloatType) -> NDArrayFloatType:
    """Return an attitude a small correction away from another.

    Parameters:
        attitude: The uncorrected 3x3 rotation.

    Returns:
        The corrected rotation, turned by a milliradian about a fixed axis so
        that a segment built from it differs measurably from its baseline.
    """
    axis = np.array([0.2, 0.5, -0.84])
    turn = np.asarray(cspyce.axisar(axis / np.linalg.norm(axis), 1.0e-3), dtype=np.float64)
    turned: NDArrayFloatType = turn @ np.asarray(attitude, dtype=np.float64)
    return turned


def write_metadata(root: Path, stub: str, metadata: dict[str, Any]) -> Path:
    """Write one per-image metadata document under a results root.

    Parameters:
        root: The navigation results root.
        stub: The image's results path stub.
        metadata: The document.

    Returns:
        The file written.
    """
    path = root / f'{stub}_metadata.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    return path


def image_document(
    *,
    image_name: str,
    midtime: float,
    cmatrix_original: NDArrayFloatType,
    cmatrix: NDArrayFloatType | None,
    status: str = 'success',
) -> dict[str, Any]:
    """Build one Cassini image's metadata document.

    Parameters:
        image_name: Basename recorded for the image.
        midtime: Exposure midtime, TDB seconds past J2000.
        cmatrix_original: The uncorrected attitude recorded for it.
        cmatrix: The corrected attitude, or None for an image without one.
        status: The navigation status.

    Returns:
        The document.
    """
    return image_metadata(
        image_name=image_name,
        cmatrix=cmatrix,
        cmatrix_original=cmatrix_original,
        camera_frame=CASSINI_CAMERA_FRAME,
        ck_frame_id=CASSINI_CK_FRAME_ID,
        start_et=midtime - EXPOSURE_S / 2.0,
        stop_et=midtime + EXPOSURE_S / 2.0,
        status=status,
        instrument='coiss',
        camera='NAC',
        shutter_mode='NACONLY',
        kernels=KERNEL_NAMES,
        sclk_midtime='1/1484573295.118',
        offset=(-3.25, 1.125),
        sigma_px=(0.0625, 0.0313),
        confidence=0.8125,
        confidence_rank='high',
        status_reason='ensemble_agreement',
    )


def write_kernels(
    kernels: Path, *, angular_velocity_in_b: bool = True
) -> tuple[NDArrayFloatType, NDArrayFloatType]:
    """Write the hermetic kernels and the two baselines, and read what they give.

    The kernels have to be furnished to build the baselines and to read the
    attitudes the metadata records, and they are unloaded again before the
    driver runs: the driver refuses to identify a clock kernel while another
    already defines that clock, which is the whole point of that refusal.

    Parameters:
        kernels: The kernel directory to write into.
        angular_velocity_in_b: Whether the second baseline carries angular
            velocity.  Without it, an image assigned to that baseline
            reproduces its attitude and then cannot be built into a segment.

    Returns:
        The uncorrected camera attitudes at the two exposure midtimes.
    """
    support = write_support_kernels(kernels)
    for path in support:
        cspyce.furnsh(str(path))
    baselines = []
    try:
        for name, center, with_av in (
            (BASELINE_A, IMAGE_A_ET, True),
            (BASELINE_B, IMAGE_B_ET, angular_velocity_in_b),
        ):
            path = kernels / name
            write_baseline_ck(
                path,
                ck_frame_id=CASSINI_CK_FRAME_ID,
                sclk_id=CASSINI_SCLK_ID,
                epochs=[center - 10.0, center, center + 10.0],
                attitude=baseline_attitude,
                angular_velocity=baseline_angular_velocity if with_av else None,
            )
            baselines.append(path)
        cspyce.furnsh(str(baselines[0]))
        cspyce.furnsh(str(baselines[1]))
        return camera_attitude(IMAGE_A_ET), camera_attitude(IMAGE_B_ET)
    finally:
        for path in reversed([*support, *baselines]):
            cspyce.unload(str(path))


def driver_argv(tree: dict[str, Path], *extra: str) -> list[str]:
    """Return the command line that points the driver at a prepared tree.

    Parameters:
        tree: The directories a fixture built.
        extra: Additional arguments.

    Returns:
        The argv list, program name first.
    """
    return [
        'sd_create_ck',
        'coiss',
        '--nav-results-root',
        str(tree['results']),
        '--kernel-dir',
        str(tree['kernels']),
        '--output-dir',
        str(tree['output']),
        '--log-root',
        str(tree['output'] / 'logs'),
        *extra,
    ]


def run_driver(
    tree: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    *extra: str,
    expected_exit: int = 0,
) -> None:
    """Run the driver over a prepared tree.

    Parameters:
        tree: The directories the fixture built.
        monkeypatch: Used to set the command line.
        extra: Additional arguments.
        expected_exit: The exit status the run should end with.
    """
    monkeypatch.setattr('sys.argv', driver_argv(tree, *extra))
    with pytest.raises(SystemExit) as exit_info:
        sd_create_ck.main()
    assert exit_info.value.code == expected_exit


def run_stopped(
    tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, message: str, *extra: str
) -> str:
    """Run the driver over a tree it must refuse, and hold it to the reason.

    Parameters:
        tree: The directories the fixture built.
        monkeypatch: Used to set the command line.
        message: Text the refusal must name.
        extra: Additional command-line arguments.

    Returns:
        The whole refusal, for a caller asserting on more of it.
    """
    monkeypatch.setattr('sys.argv', driver_argv(tree, *extra))
    with pytest.raises(ValueError, match=message) as refusal:
        sd_create_ck.main()
    return str(refusal.value)


def utc_of(tree: dict[str, Path], et: float) -> str:
    """Return one epoch as a UTC string, with the leapseconds kernel furnished.

    The run tree deliberately leaves nothing furnished, so a test that needs to
    express an epoch in UTC furnishes the leapseconds kernel for that one call.

    Parameters:
        tree: The directories the fixture built.
        et: TDB seconds past J2000.

    Returns:
        The epoch as an ISO calendar UTC string.
    """
    lsk = str(tree['kernels'] / 'test.tls')
    cspyce.furnsh(lsk)
    try:
        return str(cspyce.et2utc(et, 'ISOC', 3))
    finally:
        cspyce.unload(lsk)


def run_log(tree: dict[str, Path]) -> str:
    """Return what the run wrote to its main log.

    Parameters:
        tree: The directories the fixture built.

    Returns:
        The main log's text.
    """
    logs = list((tree['output'] / 'logs' / 'sd_create_ck').glob('main_*.log'))
    assert len(logs) == 1
    return logs[0].read_text()


def report_rows(tree: dict[str, Path]) -> dict[str, dict[str, str]]:
    """Read the report the run wrote, keyed by image name.

    Parameters:
        tree: The directories the fixture built.

    Returns:
        One entry per row.
    """
    with (tree['output'] / 'coiss_ck_report.csv').open() as stream:
        return {row['image_name']: row for row in csv.DictReader(stream)}
