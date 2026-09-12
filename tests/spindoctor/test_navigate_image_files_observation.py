"""Tests that a navigation document records the facts the observation's host publishes.

:func:`~spindoctor.navigate_image_files.build_metadata_from_result` writes the facts a
host publishes through :meth:`~spindoctor.obs.obs_inst.ObsInst.get_public_metadata` into
the document's ``observation`` block, after the image's identity, and
:func:`~spindoctor.navigate_image_files.navigate_image_files` supplies them for every
image whose navigation ran to a result, successful or failed.  A load-error or
internal-error document carries none of them.

The writer treats every host alike, so the host here is a made-up one.
"""

from pathlib import Path
from typing import Any

import numpy as np
from filecache import FCPath

from spindoctor.dataset.dataset import ImageFile, ImageFiles
from spindoctor.nav_orchestrator.image_classifier_result import NavImageClassifierResult
from spindoctor.nav_orchestrator.nav_result import NavResult
from spindoctor.nav_orchestrator.provenance import Provenance
from spindoctor.navigate_image_files import build_metadata_from_result, navigate_image_files
from spindoctor.support.status_reason import NavStatusReason

_PUBLISHED: dict[str, Any] = {
    'image_path': '/cache/image_0001.img',
    'image_name': 'image_0001.img',
    'start_time_et': 1234.5678901,
    'midtime_et': 1234.8178901,
    'end_time_et': 1235.0678901,
    'image_shape_xy': (48, 32),
    'exposure_time': 0.5,
    'filters': ['F1', 'F2'],
    'note': None,
}
"""What the made-up host publishes about its one image.

The path is spelled differently from the one the writer is given, so a test can tell
which of the two the document kept, and the note is null.
"""


class _BlankSnapshot:
    """An observation whose image holds no data, publishing the made-up host's facts."""

    def __init__(self) -> None:
        """Build the snapshot around an all-zero image, which carries no data."""
        self.data = np.zeros((32, 48), np.float64)
        self.extdata = self.data
        self.midtime = 1234.8178901
        self.camera = 'CAM'
        self.shutter_mode = None

    def reset_all(self) -> None:
        """Drop cached geometry, of which this stand-in holds none."""

    def extfov_data_sensor_mask(self) -> np.ndarray:
        """Report every pixel as live sensor.

        Returns:
            An all-true mask the shape of the data.
        """
        return np.ones(self.data.shape, bool)

    def get_public_metadata(self) -> dict[str, Any]:
        """Publish the made-up host's facts about the image.

        Returns:
            The published facts.
        """
        return _PUBLISHED


class _BlankObsClass:
    """The observation class the driver loads the one image through."""

    @classmethod
    def from_file(cls, path: Any, **kwargs: Any) -> _BlankSnapshot:
        """Return the blank snapshot whatever the path.

        Parameters:
            path: The image path the driver resolved; unread.
            kwargs: Further loader options; unread.

        Returns:
            The snapshot.
        """
        return _BlankSnapshot()


def test_the_observation_block_records_the_published_facts_without_a_pointing() -> None:
    """Every fact the host publishes reaches the observation block, a null one included.

    The result carries no pointing, as one does whose attitude could not be computed.
    The block keeps its own path rather than the host's, and leaves out the host's x/y
    shape, which ``image_shape`` already states.
    """
    result = NavResult.success(
        offset_px=(1.5, -2.0),
        covariance_px2=np.eye(2),
        confidence=0.5,
        confidence_rank='medium',
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
    metadata = build_metadata_from_result(
        result,
        Path('/holdings/image_0001.img'),
        'image_0001.img',
        instrument='fake',
        camera='CAM',
        image_shape=(32, 48),
        public_metadata=_PUBLISHED,
    )
    assert 'pointing' not in metadata['navigation_result']
    assert metadata['observation'] == {
        'image_path': '/holdings/image_0001.img',
        'image_name': 'image_0001.img',
        'instrument': 'fake',
        'camera': 'CAM',
        'image_shape': [32, 48],
        'start_time_et': 1234.5678901,
        'midtime_et': 1234.8178901,
        'end_time_et': 1235.0678901,
        'exposure_time': 0.5,
        'filters': ['F1', 'F2'],
        'note': None,
    }


def test_a_failed_navigation_records_the_published_facts(
    tmp_path: Path, fakes_report_as_simulated: None
) -> None:
    """A navigation that fails still records what the observation's host publishes."""
    image = tmp_path / 'image_0001.img'
    image.write_bytes(b'\x00')
    image_files = ImageFiles(
        image_files=[
            ImageFile(
                image_file_url=FCPath(str(image)),
                label_file_url=FCPath(str(image.with_suffix('.lbl'))),
                results_path_stub='image_0001',
            )
        ]
    )
    _success, metadata = navigate_image_files(
        _BlankObsClass,  # type: ignore[arg-type]
        image_files,
        FCPath(str(tmp_path / 'results')),
        nav_models=['!*'],
        write_output_files=False,
    )
    assert metadata['status'] == 'failed'
    assert metadata['observation']['midtime_et'] == 1234.8178901
    assert metadata['observation']['filters'] == ['F1', 'F2']
