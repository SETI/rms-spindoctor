from collections.abc import Mapping
from typing import Any

from filecache import FCPath
from oops.observation import Snapshot

from spindoctor.support.types import PathLike

from .obs_inst import ObsInst
from .obs_snapshot import ObsSnapshot

# HTTP statuses that say a file does not exist, as opposed to a server that could not
# answer
_ABSENT_HTTP_STATUSES = frozenset({404, 410})


def _http_status(cause: BaseException) -> int | None:
    """Return the HTTP status a failed remote request carries, if it carries one.

    A web request's error carries a response with a ``status_code``.  An S3 client
    error carries its response as a mapping, with the status under
    ``ResponseMetadata``.  A Google Cloud Storage error carries the status as its
    ``code``.

    Parameters:
        cause: The exception behind a failed retrieval.

    Returns:
        The HTTP status, or None if the exception carries none.
    """
    response = getattr(cause, 'response', None)
    if isinstance(response, Mapping):
        metadata = response.get('ResponseMetadata')
        status = metadata.get('HTTPStatusCode') if isinstance(metadata, Mapping) else None
    else:
        status = getattr(response, 'status_code', None)
    if status is None:
        status = getattr(cause, 'code', None)
    return status if isinstance(status, int) else None


def _is_absence(exc: FileNotFoundError) -> bool:
    """Whether a failure to retrieve a file means that the file does not exist.

    filecache raises FileNotFoundError for a missing file, but for a web server it also
    raises it for any failed request, such as a timeout, a refused connection or a
    server error, with the request's exception as the cause.  For S3 it raises it for
    any client error, such as a denied access, and for Google Cloud Storage for a
    missing object, in both cases inside the handler of the client's exception, which
    is then the error's context rather than its cause.  A failure says the file is not
    there only if nothing caused it, or if the exception behind it carries an HTTP status
    of 404 or 410.

    Parameters:
        exc: The error raised by retrieving the file.

    Returns:
        True if the file does not exist, False if its existence is unknown.
    """
    cause = exc.__cause__ if exc.__cause__ is not None else exc.__context__
    if cause is None:
        return True
    return _http_status(cause) in _ABSENT_HTTP_STATUSES


def host_input(image: PathLike, *, label: PathLike | None = None) -> FCPath:
    """Return the file to give a host to read an image: its PDS3 label, if it has one.

    A host given an image file reads that file alone, so it lacks every value that only
    the PDS3 label gives, such as a clock count the image file does not record.  Given
    the label, the host reads the label and the image file together and merges what the
    two say.  The data are the same either way.

    A label the caller names, as a dataset does from the PDS3 index, is the file read.
    Otherwise the label is the file beside the image with the image's name and the
    suffix ``.LBL``, or ``.lbl`` when the image's own suffix is lower case, as a PDS3
    volume names it, and the image is read alone only if that file does not exist.  A
    label whose existence cannot be determined, such as one on a web server that does
    not answer, is an error: reading the image alone would publish less than the label
    gives, and nothing would say so.

    Parameters:
        image: The image file, local or remote.
        label: The image's PDS3 label, if the caller knows it.

    Returns:
        The label if one is named or exists beside the image, otherwise the image file.

    Raises:
        OSError: If the file beside the image fails with an error that leaves its
            absence unconfirmed.  An error that the storage client raises as its own
            type, such as a denied access to a Google Cloud Storage bucket, is not
            converted and reaches the caller as it was raised.
    """
    if label is not None:
        return FCPath(label)
    image_path = FCPath(image)
    suffix = image_path.suffix
    label_path = image_path.with_suffix('.LBL' if suffix == suffix.upper() else '.lbl')
    try:
        label_path.retrieve()
    except FileNotFoundError as exc:
        if not _is_absence(exc):
            raise OSError(
                f'cannot tell whether the label {label_path.as_posix()} exists: {exc}'
            ) from exc
        return image_path
    return label_path


class HostLabel:
    """Mix-in for an observation whose host keeps the image's metadata on the snapshot.

    The metadata is declared here rather than on the instrument's class, whose methods
    annotate their own results as ``dict`` and would otherwise name this attribute.
    """

    dict: dict[Any, Any]
    """The image's metadata, merged from every label the host read, keyed by keyword.

    A value that no label gives has no key.
    """


class ObsSnapshotInst(ObsSnapshot, ObsInst):
    """Mix-in of ObsSnapshot and ObsInst."""

    _abstract = True

    def __init__(self, snapshot: Snapshot, **kwargs: Any) -> None:
        """Initializes a new ObsSnapshotInst instance.

        Parameters:
            snapshot: The Snapshot object to wrap.
            **kwargs: Additional keyword arguments used by subclasses.
        """
        super().__init__(snapshot, **kwargs)
