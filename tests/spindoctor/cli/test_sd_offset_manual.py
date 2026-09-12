"""Tests that the ``sd_offset --manual`` pass writes the observation's published facts.

The manual pass loads one selected image, has the operator pick its offset in the
manual-navigation dialog, and writes the same navigation document the autonomous pipeline
writes, whose ``observation`` block records what the observation's host publishes about
the image.  The dialog is stood in for, so no Qt is needed.
"""

import argparse
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from filecache import FCPath

from spindoctor.cli import sd_offset
from spindoctor.config.logging_config import LogLevels, LogSinks, RunLogging
from spindoctor.dataset.dataset import ImageFile, ImageFiles
from spindoctor.nav_orchestrator.image_classifier_result import NavImageClassifierResult
from spindoctor.nav_orchestrator.nav_result import NavResult
from spindoctor.nav_orchestrator.provenance import Provenance
from spindoctor.support.status_reason import NavStatusReason


class _Snapshot:
    """An observation of one small image whose host publishes an exposure time."""

    def __init__(self) -> None:
        """Build the snapshot around a small ramp image."""
        self.data = np.arange(32 * 48, dtype=np.float64).reshape(32, 48)
        self.camera = 'CAM'
        self.shutter_mode = None

    def get_public_metadata(self) -> dict[str, Any]:
        """Publish the image's name and its exposure time.

        Returns:
            The published facts.
        """
        return {'image_name': 'image_0001.img', 'exposure_time': 0.5}


class _ObsClass:
    """The observation class the manual pass loads the one image through."""

    @classmethod
    def from_file(cls, path: Any, **kwargs: Any) -> _Snapshot:
        """Return the snapshot whatever the path.

        Parameters:
            path: The image path the pass resolved; unread.
            kwargs: Further loader options; unread.

        Returns:
            The snapshot.
        """
        return _Snapshot()


class _OneImageDataset:
    """A dataset whose selection names exactly one image."""

    def __init__(self, image_file: ImageFile) -> None:
        """Hold the one image the selection names.

        Parameters:
            image_file: The image.
        """
        self._image_file = image_file

    def yield_image_files_from_arguments(
        self, arguments: argparse.Namespace
    ) -> Iterator[ImageFiles]:
        """Yield the one image as a one-image batch.

        Parameters:
            arguments: The parsed command line, unused.

        Yields:
            The batch.
        """
        yield ImageFiles(image_files=[self._image_file])


def _accepted(obs: Any, *, config: Any) -> NavResult:
    """Stand in for the dialog: the operator accepts an offset.

    Parameters:
        obs: The loaded observation; unread.
        config: The configuration the pass hands the dialog; unread.

    Returns:
        A successful result carrying the accepted offset.
    """
    return NavResult.success(
        offset_px=(1.5, -2.0),
        covariance_px2=np.eye(2),
        confidence=1.0,
        confidence_rank='high',
        status_reason=NavStatusReason.OK,
        per_technique=[],
        feature_inventory=[],
        image_classifier=NavImageClassifierResult(
            image_class='clean', saturation_frac=0.0, missing_frac=0.0, noise_sigma=1.0, max_dn=1.0
        ),
        provenance=Provenance(
            spindoctor_version='0.0.0',
            image_et=1234.8178901,
            pipeline_run_iso8601='2026-09-11T00:00:00Z',
        ),
    )


def test_the_manual_pass_records_the_published_facts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The manual pass's document records what the observation's host publishes."""
    image = tmp_path / 'image_0001.img'
    image.write_bytes(b'\x00')
    image_file = ImageFile(
        image_file_url=FCPath(image),
        label_file_url=FCPath(image.with_suffix('.lbl')),
        results_path_stub='image_0001',
    )
    monkeypatch.setattr(sd_offset, 'DATASET', _OneImageDataset(image_file))
    monkeypatch.setattr(sd_offset, 'run_manual_nav', _accepted)
    run_logging = RunLogging(
        levels=LogLevels(),
        sinks=LogSinks(log_root=FCPath(tmp_path / 'logs'), image_file=False),
        timestamp='2026-09-11T00-00-00',
        main_log_path=None,
    )
    sd_offset._run_manual_pass(
        _ObsClass,  # type: ignore[arg-type]
        argparse.Namespace(),
        FCPath(tmp_path / 'results'),
        run_logging,
        write_output_files=True,
    )
    document = json.loads((tmp_path / 'results' / 'image_0001_metadata.json').read_text())
    assert document['observation']['exposure_time'] == 0.5
