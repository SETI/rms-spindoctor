"""Tests that a navigation document records what the observation's host publishes.

``build_metadata_from_result`` writes the facts an instrument host publishes through
``ObsInst.get_public_metadata`` into the document's ``observation`` block, after the
image's identity, and ``navigate_image_files`` supplies them for every image that
loaded, whatever became of its navigation.
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

_CASSINI_PUBLIC_METADATA: dict[str, Any] = {
    'image_path': '/cache/N1635282917_1_CALIB.IMG',
    'image_name': 'N1635282917_1_CALIB.IMG',
    'instrument_host_lid': 'urn:nasa:pds:context:instrument_host:spacecraft.co',
    'instrument_lid': 'urn:nasa:pds:context:instrument:issna.co',
    'start_time_utc': '2009-10-26T20:32:22.024',
    'midtime_utc': '2009-10-26T20:32:22.134',
    'end_time_utc': '2009-10-26T20:32:22.244',
    'start_time_et': 309861208.2064568,
    'midtime_et': 309861208.3164568,
    'end_time_et': 309861208.4264568,
    'start_time_scet': 1635282917.063,
    'midtime_scet': 1635282917.0904999,
    'end_time_scet': 1635282917.118,
    'image_shape_xy': (1024, 1024),
    'camera': 'NAC',
    'exposure_time': 0.22,
    'filters': ['CL1', 'CL2'],
    'sampling': 'FULL',
    'gain_mode': 2,
    'description': 'N/A',
    'observation_id': 'ISS_120RH_MUTUALEVE001_PRIME',
}
"""What Cassini ISS publishes for N1635282917_1_CALIB.IMG, in its own order and types.

The path is spelled differently from the one the writer is given, so a test can tell
which of the two the document kept.
"""


class _BlankSnapshot:
    """An observation whose image holds no data, publishing Cassini's facts."""

    def __init__(self) -> None:
        """Build the snapshot around an all-zero image, which carries no data."""
        self.data = np.zeros((32, 32), np.float64)
        self.extdata = self.data
        self.midtime = 309861208.3164568
        self.camera = 'NAC'
        self.shutter_mode = 'NACONLY'

    def reset_all(self) -> None:
        """Drop cached geometry, of which this stand-in holds none."""

    def extfov_data_sensor_mask(self) -> np.ndarray:
        """Report every pixel as live sensor.

        Returns:
            An all-true mask the shape of the data.
        """
        return np.ones(self.data.shape, bool)

    def get_public_metadata(self) -> dict[str, Any]:
        """Publish what Cassini ISS publishes about N1635282917_1_CALIB.IMG.

        Returns:
            The published facts.
        """
        return _CASSINI_PUBLIC_METADATA


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
    """A Cassini host's published facts, its times among them, reach the observation block.

    The result carries no pointing, as one does whose attitude could not be computed.
    The block keeps its own path, name and camera rather than the host's, and leaves
    out the host's x/y shape, which ``image_shape`` already states.
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
            image_et=309861208.316457,
            pipeline_run_iso8601='2026-09-11T00:00:00Z',
        ),
    )
    metadata = build_metadata_from_result(
        result,
        Path('/holdings/N1635282917_1_CALIB.IMG'),
        'N1635282917_1_CALIB.IMG',
        instrument='coiss',
        camera='NAC',
        shutter_mode='NACONLY',
        image_shape=(1024, 1024),
        public_metadata=_CASSINI_PUBLIC_METADATA,
    )
    assert 'pointing' not in metadata['navigation_result']
    assert metadata['observation'] == {
        'image_path': '/holdings/N1635282917_1_CALIB.IMG',
        'image_name': 'N1635282917_1_CALIB.IMG',
        'instrument': 'coiss',
        'camera': 'NAC',
        'shutter_mode': 'NACONLY',
        'image_shape': [1024, 1024],
        'instrument_host_lid': 'urn:nasa:pds:context:instrument_host:spacecraft.co',
        'instrument_lid': 'urn:nasa:pds:context:instrument:issna.co',
        'start_time_utc': '2009-10-26T20:32:22.024',
        'midtime_utc': '2009-10-26T20:32:22.134',
        'end_time_utc': '2009-10-26T20:32:22.244',
        'start_time_et': 309861208.2064568,
        'midtime_et': 309861208.3164568,
        'end_time_et': 309861208.4264568,
        'start_time_scet': 1635282917.063,
        'midtime_scet': 1635282917.0904999,
        'end_time_scet': 1635282917.118,
        'exposure_time': 0.22,
        'filters': ['CL1', 'CL2'],
        'sampling': 'FULL',
        'gain_mode': 2,
        'description': 'N/A',
        'observation_id': 'ISS_120RH_MUTUALEVE001_PRIME',
    }


def test_a_failed_navigation_records_the_published_facts(
    tmp_path: Path, fakes_report_as_simulated: None
) -> None:
    """A navigation that fails still records what the observation's host publishes."""
    image = tmp_path / 'N1635282917_1_CALIB.IMG'
    image.write_bytes(b'\x00')
    image_files = ImageFiles(
        image_files=[
            ImageFile(
                image_file_url=FCPath(str(image)),
                label_file_url=FCPath(str(image.with_suffix('.LBL'))),
                results_path_stub='N1635282917_1_CALIB',
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
    assert metadata['observation']['midtime_et'] == 309861208.3164568
    assert metadata['observation']['filters'] == ['CL1', 'CL2']
