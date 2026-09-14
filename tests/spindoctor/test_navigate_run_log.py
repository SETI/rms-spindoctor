"""Tests that each image's answer reaches the run's log as well as its own.

The detail of how an offset was reached belongs to the image and stays in the
image's log. What the answer *was* belongs to both: it is what an operator
watching a batch is waiting for, and following a run should not mean opening
a file per image to find out whether any of them worked.
"""

from pathlib import Path
from typing import Any, cast

import numpy as np
import pdslogger
import pytest
from filecache import FCPath

from spindoctor.config import (
    IMAGE_LOGGER,
    MAIN_LOGGER,
    LogLevels,
    LogSinks,
    build_image_log_handlers,
    build_main_logger,
    set_log_levels,
)
from spindoctor.config.program_names import SD_OFFSET
from spindoctor.dataset.dataset import ImageFile, ImageFiles
from spindoctor.navigate_image_files import log_final_result_to_run, navigate_image_files
from spindoctor.support.exceptions import NavPointingError

_STAMP = '2026-07-29T12-00-00'


class _StubResult:
    """Carries only what the run-log line reads off a navigation result."""

    def __init__(
        self,
        *,
        offset_px: tuple[float, float] | None,
        status: str,
        confidence: float = 0.75,
        confidence_rank: str = 'medium',
        status_reason: str = 'no_features_extracted',
    ) -> None:
        """Build a stand-in result.

        Parameters:
            offset_px: The offset, or None for an image with no answer.
            status: The navigation status.
            confidence: The reported confidence.
            confidence_rank: The confidence rank.
            status_reason: Why the navigation ended as it did.
        """
        self.offset_px = offset_px
        self.status = status
        self.confidence = confidence
        self.confidence_rank = confidence_rank
        self.status_reason = status_reason


@pytest.fixture(autouse=True)
def _fakes_report_as_simulated(fakes_report_as_simulated: None) -> None:
    """Apply the shared simulated-instrument report to every test in this module."""


def _main_log_of(tmp_path: Path, report: Any) -> str:
    """Return the run log written while reporting one result.

    Parameters:
        tmp_path: Directory used as the log root.
        report: Callable invoked with the main logger configured.

    Returns:
        The text of the main log file.
    """
    levels = LogLevels()
    set_log_levels(levels)
    path = build_main_logger(
        MAIN_LOGGER,
        SD_OFFSET,
        LogSinks(log_root=FCPath(tmp_path), main_console=False),
        levels,
        timestamp=_STAMP,
    )
    try:
        report()
    finally:
        for handler in list(MAIN_LOGGER.handlers):
            if handler is not pdslogger.NULL_HANDLER:
                # Removed before closing: a closed handler left attached would
                # swallow later tests' output through the module-level logger.
                MAIN_LOGGER.remove_handler(handler)
                handler.close()
    assert path is not None
    with path.open('r') as stream:
        return str(stream.read())


def _line_for(tmp_path: Path, result: _StubResult) -> str:
    """Return the run-log text for one result.

    Parameters:
        tmp_path: Directory used as the log root.
        result: The result to report.

    Returns:
        The text of the main log file.
    """
    return _main_log_of(
        tmp_path, lambda: log_final_result_to_run('N1234567890_1.IMG', cast(Any, result))
    )


def test_the_run_log_reports_the_offset(tmp_path: Path) -> None:
    """The offset is the thing an operator watching a run is waiting for."""
    result = _StubResult(offset_px=(1.5, -2.5), status='success')
    assert '(1.500, -2.500)' in _line_for(tmp_path, result)


def test_the_run_log_reports_the_status(tmp_path: Path) -> None:
    """Alongside whether the answer is one to act on."""
    result = _StubResult(offset_px=(1.5, -2.5), status='success')
    assert 'status=success' in _line_for(tmp_path, result)


def test_the_run_log_reports_the_confidence(tmp_path: Path) -> None:
    """An offset without its confidence is not an answer yet."""
    result = _StubResult(offset_px=(1.5, -2.5), status='success', confidence=0.75)
    assert '0.750' in _line_for(tmp_path, result)


def test_the_run_log_names_the_image(tmp_path: Path) -> None:
    """A line about one image says which, since a batch has many."""
    result = _StubResult(offset_px=(1.5, -2.5), status='success')
    assert 'N1234567890_1.IMG' in _line_for(tmp_path, result)


def test_the_run_log_reports_an_image_with_no_offset(tmp_path: Path) -> None:
    """An image that produced no answer is reported rather than passed over."""
    result = _StubResult(offset_px=None, status='failed')
    assert 'no offset' in _line_for(tmp_path, result)


def test_the_run_log_gives_the_reason_there_is_no_offset(tmp_path: Path) -> None:
    """And says why, which is the whole of what the run's log can say."""
    result = _StubResult(offset_px=None, status='failed', status_reason='no_signal_in_image')
    assert 'no_signal_in_image' in _line_for(tmp_path, result)


def test_it_is_one_line_per_image(tmp_path: Path) -> None:
    """One line, so a batch of thousands stays readable.

    The detail is in the image's own log; repeating it here would undo the
    reason it was moved there.
    """
    result = _StubResult(offset_px=(1.5, -2.5), status='success')
    reported = [
        line for line in _line_for(tmp_path, result).splitlines() if 'N1234567890_1.IMG' in line
    ]
    assert len(reported) == 1


# ---------------------------------------------------------------------------
# Wired into the driver, and taking nothing out of the image log
# ---------------------------------------------------------------------------


class _FakeSnapshot:
    """Minimal stand-in for an observation the driver can navigate."""

    def __init__(self) -> None:
        """Build a snapshot with a plain noise field."""
        rng = np.random.default_rng(seed=99)
        self.data = rng.standard_normal(size=(32, 32)) + 100.0
        self.extdata = self.data
        self.midtime = 100.0
        self.camera = 'NAC'
        self.shutter_mode: str | None = None

    def extfov_data_sensor_mask(self) -> np.ndarray:
        """Return the sensor mask.

        Returns:
            An all-true mask the shape of the data.
        """
        return np.ones(self.data.shape, bool)

    def get_public_metadata(self) -> dict[str, Any]:
        """Publish no fact about the image beyond what the driver records itself.

        Returns:
            An empty mapping.
        """
        return {}


class _FakeObsClass:
    """Observation class whose images always load."""

    @classmethod
    def from_file(cls, path: Any, **kwargs: Any) -> _FakeSnapshot:
        """Return a snapshot regardless of the path.

        Parameters:
            path: Ignored.
            **kwargs: Ignored.

        Returns:
            The snapshot.
        """
        return _FakeSnapshot()


def _image_files(tmp_path: Path) -> ImageFiles:
    """Build a one-image batch pointing at placeholder files.

    Parameters:
        tmp_path: Directory to write the placeholders into.

    Returns:
        The batch.
    """
    image = tmp_path / 'fake_image.IMG'
    image.write_bytes(b'\x00')
    label = tmp_path / 'fake_image.LBL'
    label.write_bytes(b'\x00')
    return ImageFiles(
        image_files=[
            ImageFile(
                image_file_url=FCPath(str(image)),
                label_file_url=FCPath(str(label)),
                results_path_stub='fake_image',
            )
        ]
    )


def test_the_driver_reports_each_image_to_the_run_log(tmp_path: Path) -> None:
    """Navigating an image writes its answer to the run's log.

    Driven through the real function rather than the helper alone, so the
    wiring is what is under test and not just the formatting.
    """
    text = _main_log_of(
        tmp_path,
        lambda: navigate_image_files(
            cast(Any, _FakeObsClass),
            _image_files(tmp_path),
            FCPath(str(tmp_path / 'results')),
            write_output_files=False,
        ),
    )
    assert 'fake_image.IMG: status=' in text


def test_a_failed_pointing_computation_is_reported_to_the_run_log_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one-line pointing failure lands in the run's own log file.

    The orchestrator promises an operator watching a batch one line per
    affected image; reading the file pins the routing itself, not merely that
    the message was emitted somewhere.
    """

    def _boom(obs: Any, *, offset_px: Any, rotation_fitted: bool) -> None:
        raise NavPointingError('the furnished kernels cannot supply the rotation')

    monkeypatch.setattr('spindoctor.nav_orchestrator.orchestrator.compute_pointing', _boom)
    text = _main_log_of(
        tmp_path,
        lambda: navigate_image_files(
            cast(Any, _FakeObsClass),
            _image_files(tmp_path),
            FCPath(str(tmp_path / 'results')),
            write_output_files=False,
        ),
    )
    assert 'Corrected pointing not recorded for this sim image' in text


def test_a_missing_frame_mapping_warning_is_reported_to_the_run_log_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The missing-frame-mapping warning lands in the run's own log file.

    A registered instrument navigating with no SPICE frame mapping produces no
    corrected attitude for any of its images, so the warning must be visible
    from the run log rather than only from each per-image log.
    """
    monkeypatch.setattr(
        'spindoctor.nav_orchestrator.orchestrator.obs_class_to_inst_name', lambda cls: 'newinst'
    )
    text = _main_log_of(
        tmp_path,
        lambda: navigate_image_files(
            cast(Any, _FakeObsClass),
            _image_files(tmp_path),
            FCPath(str(tmp_path / 'results')),
            write_output_files=False,
        ),
    )
    assert 'No SPICE camera frame is mapped for instrument newinst' in text


def test_the_image_log_keeps_its_own_detail(tmp_path: Path) -> None:
    """Summarizing to the run's log takes nothing out of the image's.

    The orchestrator's own records are what the image log is for, and they
    stay there whether or not the run's log carries a summary.
    """
    levels = LogLevels()
    set_log_levels(levels)
    handlers, path = build_image_log_handlers(
        'nav', 'fake_image', LogSinks(log_root=FCPath(tmp_path)), levels, timestamp=_STAMP
    )
    try:
        with IMAGE_LOGGER.open('IMAGE', handler=handlers, level=levels.image_section_level()):
            navigate_image_files(
                cast(Any, _FakeObsClass),
                _image_files(tmp_path),
                FCPath(str(tmp_path / 'results')),
                write_output_files=False,
            )
    finally:
        for handler in handlers:
            if handler is not pdslogger.NULL_HANDLER:
                handler.close()
    assert path is not None
    with path.open('r') as stream:
        text = stream.read()
    # Both records the run's one-line summary deliberately does not carry,
    # asserted from one navigation rather than two identical ones.
    assert 'Image classifier:' in text
    assert 'Final: status=' in text


def test_a_bad_stub_fails_its_own_image_not_the_batch(tmp_path: Path) -> None:
    """A stub that would put the log outside the log root is a per-image error.

    Raising through the driver would end the batch and discard the images
    already navigated, for one malformed entry.  Both halves of "failed this
    image" are asserted from one navigation: the reason it gives, and that it
    reports failure rather than a navigated image.
    """
    image = tmp_path / 'fake_image.IMG'
    image.write_bytes(b'\x00')
    label = tmp_path / 'fake_image.LBL'
    label.write_bytes(b'\x00')
    batch = ImageFiles(
        image_files=[
            ImageFile(
                image_file_url=FCPath(str(image)),
                label_file_url=FCPath(str(label)),
                results_path_stub='../../escaped/fake_image',
            )
        ]
    )
    success, metadata = navigate_image_files(
        cast(Any, _FakeObsClass),
        batch,
        FCPath(str(tmp_path / 'results')),
        write_output_files=False,
    )
    assert metadata['status_error'] == 'invalid_results_path_stub'
    assert success is False
