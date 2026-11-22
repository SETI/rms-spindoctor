"""Tests for choosing the file a host reads to load an image."""

from pathlib import Path
from typing import Any

import pytest
from filecache import FCPath

from spindoctor.obs.obs_snapshot_inst import host_input


class _Response:
    """The part of an HTTP response that a failed request carries: its status."""

    def __init__(self, status_code: int) -> None:
        self.status_code = status_code


class _RequestError(Exception):
    """A failed web request, carrying the server's response if there was one."""

    def __init__(self, response: _Response | None) -> None:
        super().__init__('request failed')
        self.response = response


class _S3ClientError(Exception):
    """An S3 client error, which carries its response as a mapping."""

    def __init__(self, status: int) -> None:
        super().__init__('An error occurred when calling the GetObject operation')
        self.response = {
            'Error': {'Code': str(status)},
            'ResponseMetadata': {'HTTPStatusCode': status},
        }


class _GSNotFoundError(Exception):
    """A Google Cloud Storage missing-object error, which carries its status as ``code``."""

    code = 404


def _retrieve_failing_inside(monkeypatch: pytest.MonkeyPatch, client_error: Exception) -> None:
    """Make every retrieval fail as filecache fails an S3 or Google Cloud Storage download.

    filecache raises FileNotFoundError inside the handler of the client's error, so the
    client's error is the context of the FileNotFoundError and not its cause.

    Parameters:
        monkeypatch: The test's monkeypatch fixture.
        client_error: The error the storage client raised.
    """

    def retrieve(self: FCPath, *args: Any, **kwargs: Any) -> Path:
        try:
            raise client_error
        except Exception:
            # As filecache does, without "from", so the client's error is only the context
            raise FileNotFoundError(f'Failed to download file: {self}')  # noqa: B904

    monkeypatch.setattr(FCPath, 'retrieve', retrieve)


def _retrieve_failing_with(monkeypatch: pytest.MonkeyPatch, response: _Response | None) -> None:
    """Make every retrieval fail as filecache fails a web request.

    Parameters:
        monkeypatch: The test's monkeypatch fixture.
        response: The response the failed request carries, or None for an unanswered
            request.
    """

    def retrieve(self: FCPath, *args: Any, **kwargs: Any) -> Path:
        raise FileNotFoundError(f'Failed to download file: {self}') from _RequestError(response)

    monkeypatch.setattr(FCPath, 'retrieve', retrieve)


def test_an_image_with_a_label_beside_it_is_read_through_the_label(tmp_path: Path) -> None:
    """The label with the image's name and the suffix .LBL is the file the host reads."""
    image = tmp_path / 'C2062133_GEOMED.IMG'
    image.write_bytes(b'')
    (tmp_path / 'C2062133_GEOMED.LBL').write_text('END\n')
    assert host_input(str(image)) == FCPath(tmp_path / 'C2062133_GEOMED.LBL')


def test_a_lower_case_image_name_has_a_lower_case_label(tmp_path: Path) -> None:
    """An image whose suffix is lower case has its label's suffix in lower case too."""
    image = tmp_path / 'lor_0299147641_0x630_sci.fit'
    image.write_bytes(b'')
    (tmp_path / 'lor_0299147641_0x630_sci.lbl').write_text('END\n')
    assert host_input(FCPath(image)) == FCPath(tmp_path / 'lor_0299147641_0x630_sci.lbl')


def test_an_image_with_no_label_beside_it_is_read_alone(tmp_path: Path) -> None:
    """With no label beside it, the host reads the image file itself."""
    image = tmp_path / 'C0368558239R.IMG'
    image.write_bytes(b'')
    assert host_input(image) == FCPath(image)


def test_a_named_label_is_read_without_looking_beside_the_image(tmp_path: Path) -> None:
    """A label the caller names is the file read, whether or not one is beside the image."""
    image = tmp_path / 'N1454725799_1_CALIB.IMG'
    named = tmp_path / 'elsewhere' / 'N1454725799_1_CALIB.LBL'
    assert host_input(image, label=named) == FCPath(named)


def test_a_label_the_server_says_is_missing_leaves_the_image_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A label the web server answers 404 for does not exist, so the image is read alone."""
    _retrieve_failing_with(monkeypatch, _Response(404))
    image = 'https://example.org/holdings/C3421729_GEOMED.IMG'
    assert host_input(image) == FCPath(image)


def test_a_label_the_server_cannot_serve_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A server error leaves the label's existence unknown, so the load fails."""
    _retrieve_failing_with(monkeypatch, _Response(503))
    with pytest.raises(OSError, match=r'cannot tell whether the label .*C3421729_GEOMED\.LBL'):
        host_input('https://example.org/holdings/C3421729_GEOMED.IMG')


def test_a_label_on_a_server_that_does_not_answer_is_an_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unanswered request, such as a timeout, is not the label's absence."""
    _retrieve_failing_with(monkeypatch, None)
    with pytest.raises(OSError, match=r'cannot tell whether the label .*C3421729_GEOMED\.LBL'):
        host_input('https://example.org/holdings/C3421729_GEOMED.IMG')


def test_a_label_s3_denies_access_to_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """An S3 access denial is not the label's absence, so the load fails."""
    _retrieve_failing_inside(monkeypatch, _S3ClientError(403))
    with pytest.raises(OSError, match=r'cannot tell whether the label .*N1454725799_1\.LBL'):
        host_input('s3://holdings/COISS_2001/N1454725799_1.IMG')


def test_a_label_s3_says_is_missing_leaves_the_image_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A label S3 answers 404 for does not exist, so the image is read alone."""
    _retrieve_failing_inside(monkeypatch, _S3ClientError(404))
    image = 's3://holdings/COISS_2001/N1454725799_1.IMG'
    assert host_input(image) == FCPath(image)


def test_a_label_google_storage_says_is_missing_leaves_the_image_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A label Google Cloud Storage reports missing does not exist; the image is read alone."""
    _retrieve_failing_inside(monkeypatch, _GSNotFoundError())
    image = 'gs://holdings/COISS_2001/N1454725799_1.IMG'
    assert host_input(image) == FCPath(image)
