"""Turn one navigation metadata document into the per-image facts a consumer reads.

A consumer that summarizes a whole run asks for the same per-image shape
whichever storage answered it, so that shape is built here, once, rather than by
each consumer.  :class:`ImageFacts` is that shape: the image's own values, one
entry per technique that reported, and the aggregated inventory of the features
the models offered.

Each of the three is keyed by results-index column name, and deliberately so.
The shape *is* the row shape, so the index side of the seam hands its rows
straight back with no conversion at all, and there is no second spelling of any
value for the two sides to drift apart on.  A column the index gains is a key
here, and a consumer reads it under one name whichever storage built the
mapping.

The column mapping lives here, apart from the walk and the writer, because it
is the part that has to be read against the document the navigator writes.

Some of these columns are read back by a consumer that classifies the record
they came from, and those are not coerced here at all: their domain belongs to
:mod:`spindoctor.support.nav_record`, which is where the readers get it too, and
this module stores what those functions return and NULL wherever they return
nothing.  The rule is what makes the index faithful rather than merely close: a
value stored means to a reader what the document's own value meant, and a value
the reader could not have used is not stored, so a record rebuilt from a row is
classified exactly as its document is.  Writing a second set of rules here that
agreed today is how the two storages came to disagree about a three-element
offset and a rotation written as a 3x3 nesting.

Two mappings are easy to get wrong and are called out where they are made.  The
offset comes from the document's top-level ``offset`` and is stored as written:
``navigation_result.offset_px`` is the same offset rounded for display, and the
index carries the value consumers apply rather than the one an operator reads.
``status_error`` and ``status_reason`` are different vocabularies -- one is
matched verbatim by a selection filter, the other explains a non-success
outcome -- so they are kept in different columns rather than merged into one.

A file that carries neither an observation image name nor an observation
instrument is not a per-image navigation document at all.  A results tree holds
such files, so this refuses them by raising :class:`MetadataDocumentError`,
which carries the reason apart from the file name; the caller counts them and
goes on.

Every container the document schema declares is checked before it is read.  A
file whose ``observation`` is a string, or whose ``per_technique`` holds
numbers, is a document of some other shape rather than a navigation result, and
the refusal names which field said so.  Reading it without the check would raise
an ``AttributeError`` out of the middle of a run and cost every other file in
the tree, so the shape is part of what "current-schema" means here.  The names
inside a container are checked the same way and for the same reason: a
``per_technique`` list that names one technique twice describes rows the index
cannot hold, and finding that out from the database ends the run instead of the
file.
"""

from dataclasses import dataclass
from typing import Any, NoReturn, cast

from spindoctor.nav_records.derived import date_from_image_et, image_number_from_name
from spindoctor.nav_records.record import NavRecord

# Every column a consumer classifies a rebuilt record from is filled by the
# reader's own function rather than by a rule written here.  A second rule that
# agreed today is exactly what let a document and its row supply different
# pointing.
from spindoctor.support.nav_record import (
    UNKNOWN_STATUS,
    finite_float,
    record_offset,
    record_rotation_matrix,
    record_status,
    record_status_error,
)

__all__ = [
    'NOT_A_NAVIGATION_DOCUMENT',
    'DocumentOrigin',
    'ImageFacts',
    'MetadataDocumentError',
    'facts_from_document',
    'subtree_of',
]

NOT_A_NAVIGATION_DOCUMENT = 'not a current-schema navigation document'
"""Opening of every refusal of a file that is some other kind of document.

A results tree holds hundreds of ``*_metadata.json`` files that were never
navigation results.  The tally an operator reads has to say that in as many
words, because "no observation.instrument" on its own reads as a navigation
result that failed to ingest.
"""


class MetadataDocumentError(ValueError):
    """A file ingest cannot read as a current-schema navigation document.

    Carries the reason apart from the file name so a run that meets several
    hundred such files can tally them by cause rather than printing several
    hundred lines that each have to be read individually.
    """

    def __init__(self, reason: str, *, source_file: str) -> None:
        """Record the reason and the file it applies to.

        Parameters:
            reason: What is wrong with the document, with nothing in it that
                identifies the individual file.
            source_file: Path or URL of the file, for the message.
        """
        super().__init__(f'{source_file}: {reason}')
        self.reason = reason


@dataclass(frozen=True)
class DocumentOrigin:
    """Where one metadata document came from and what the walk saw of it.

    Parameters:
        root_url: Normalized URL of the results root the document lives under.
        results_path_stub: The document's path under that root, without the
            ``_metadata.json`` suffix.
        source_file: Full path or URL of the document, recorded as provenance.
        mtime_ns: Modification time from the listing entry, or None when the
            backend's listing does not report one.
        size_bytes: Size from the listing entry, or None when the backend's
            listing does not report one.
    """

    root_url: str
    results_path_stub: str
    source_file: str
    mtime_ns: int | None
    size_bytes: int | None


@dataclass(frozen=True)
class ImageFacts:
    """What one metadata document says about its image, column by column.

    Each of the three mappings is keyed by results-index column name, so a
    consumer reads the same keys whether the facts were built from a document or
    read back out of the index.  That correspondence is also the bound on them:
    what they hold is what the column set holds, and a field of the document no
    column holds is in neither storage's answer.

    Neither list carries an order a consumer may rely on.  A source reading
    documents yields the techniques in the order the document wrote them and
    the feature sources in key order; a source reading rows yields each list in
    whatever order the server answers a sort on the image key in.  Every entry
    of both carries its own identity -- a technique its name, a feature source
    its type, model and name -- so a consumer that needs an order sorts on that.

    Parameters:
        image: The image's own values, keyed by ``images`` column name.
        techniques: One mapping per technique that reported, keyed by
            ``techniques`` column name.
        feature_sources: One mapping per feature type and source, keyed by
            ``feature_sources`` column name.
        record: The record these facts were read out of, when the storage
            answered from a document and therefore has one to hand back.  None
            when they came from a row: the record a consumer wants there is
            rebuilt from a different set of columns than the facts hold, so it
            stays a read of its own.  A consumer that wants both -- what the
            document records, and the record itself -- therefore reads one
            document once on the tree side, and asks the index twice.
    """

    image: dict[str, Any]
    techniques: list[dict[str, Any]]
    feature_sources: list[dict[str, Any]]
    record: NavRecord | None = None


# ---------------------------------------------------------------------------
# Shapes the document schema declares
# ---------------------------------------------------------------------------


def _refuse(detail: str, source: DocumentOrigin) -> NoReturn:
    """Refuse a file that is not a current-schema navigation document.

    Parameters:
        detail: Which field said so, with nothing file-specific in it.
        source: Where the document came from.

    Raises:
        MetadataDocumentError: Always.
    """
    raise MetadataDocumentError(
        f'{NOT_A_NAVIGATION_DOCUMENT} ({detail})', source_file=source.source_file
    )


def _object(value: Any, name: str, source: DocumentOrigin) -> dict[str, Any]:
    """Read a JSON object the schema declares, or an empty one when it is absent.

    Parameters:
        value: The value as it was parsed.
        name: Dotted path of the field, for the refusal.
        source: Where the document came from.

    Returns:
        The object, or an empty one when the field is absent.

    Raises:
        MetadataDocumentError: If the field is present and is not an object.
    """
    if value is None:
        return {}
    if not isinstance(value, dict):
        _refuse(f'{name} is not an object', source)
    return value


def _array(value: Any, name: str, source: DocumentOrigin) -> list[Any]:
    """Read a JSON array the schema declares, or an empty one when it is absent.

    Parameters:
        value: The value as it was parsed.
        name: Dotted path of the field, for the refusal.
        source: Where the document came from.

    Returns:
        The array, or an empty one when the field is absent.

    Raises:
        MetadataDocumentError: If the field is present and is not an array.
    """
    if value is None:
        return []
    if not isinstance(value, list):
        _refuse(f'{name} is not a list', source)
    return value


def _array_of_objects(value: Any, name: str, source: DocumentOrigin) -> list[dict[str, Any]]:
    """Read a JSON array of objects the schema declares.

    Parameters:
        value: The value as it was parsed.
        name: Dotted path of the field, for the refusal.
        source: Where the document came from.

    Returns:
        The entries, or an empty list when the field is absent.

    Raises:
        MetadataDocumentError: If the field is not an array, or holds an entry
            that is not an object.
    """
    entries = _array(value, name, source)
    if not all(isinstance(entry, dict) for entry in entries):
        _refuse(f'{name} holds an entry that is not an object', source)
    return cast(list[dict[str, Any]], entries)


# ---------------------------------------------------------------------------
# One document into rows
# ---------------------------------------------------------------------------


def _int_or_none(value: Any) -> int | None:
    """Coerce a JSON value to an integer identifier, or None.

    Parameters:
        value: The value as it was parsed.

    Returns:
        The integer, or None when the value is absent or is not one.  A frame
        identity is an integer, so a value recorded as a float is not one and is
        stored as nothing rather than converted into one.  A boolean is refused
        for the same reason: it is an ``int`` in Python and an identifier
        nowhere.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _byte_count_or_none(value: Any) -> int | None:
    """Coerce a JSON value to a count of bytes, or None.

    Parameters:
        value: The value as it was parsed.

    Returns:
        The count, or None when the value is absent or is not one.  A count of
        bytes is a whole number that cannot be negative, so a float, a negative
        and a boolean are each stored as nothing rather than as a number the
        reader would have to distrust.  A run on a kernel publishing no peak
        records none, and nothing is the honest answer for it.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 0:
        return value
    return None


def _str_or_none(value: Any) -> str | None:
    """Coerce a JSON value to a non-empty string, or None.

    An empty string becomes None so that the reason columns hold either a
    reason or nothing.  A ``COALESCE`` over them then falls through in exactly
    the cases the reader expects it to.

    Parameters:
        value: The value as it was parsed.

    Returns:
        The string, or None when it is absent, empty, or not a string.
    """
    if isinstance(value, str) and value:
        return value
    return None


def _text_list_or_none(value: Any) -> list[str] | None:
    """Store a JSON list of text as it stands, or None.

    Nothing is coerced and nothing is dropped.  The consumer of this column
    refuses a provenance block that is not a list of names, and refuses one
    naming nothing where an attitude was recorded, so a column that filtered
    the bad members out or rendered a malformed block as an empty list would
    answer for a document the reader would have refused.

    Parameters:
        value: The value as it was parsed.

    Returns:
        The list, or None when the value is absent or is not a list of strings.
        An empty list is kept, because a run that recorded no kernels is a
        statement about the run and its consumer says so.
    """
    if not isinstance(value, list):
        return None
    if not all(isinstance(member, str) for member in value):
        return None
    return list(value)


def _pair(value: Any) -> tuple[float | None, float | None]:
    """Split a two-element JSON list into a pair of finite floats.

    A sequence of any other length is refused whole rather than truncated: two
    of three recorded numbers are not the pair anybody wrote.

    Parameters:
        value: The value as it was parsed, or None.

    Returns:
        The two values, each None when absent or unusable.
    """
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None, None
    return finite_float(value[0]), finite_float(value[1])


def _image_shape(value: Any) -> tuple[int | None, int | None]:
    """Per-axis ``(v, u)`` pixel dimensions from a metadata image_shape list.

    Parameters:
        value: The recorded ``observation.image_shape``.

    Returns:
        The two dimensions, each None when the value is not a usable pair.
    """
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None, None
    try:
        return int(value[0]), int(value[1])
    except (TypeError, ValueError):
        return None, None


def _covariance_or_none(covariance: Any) -> list[list[float]] | None:
    """Store a recorded covariance matrix whole, or nothing.

    The matrix is kept square and row-major as the document wrote it, so a
    reader gets back every term the fit produced.  A twist-fitted result records
    3x3, and its rotation row and column hold the offset-to-rotation cross terms
    that no per-axis sigma states: reduced to the offset block, or to the square
    roots of its diagonal, those terms are gone and a reader has no way to know
    they were ever recorded.  So the reduction is left to whoever wants one.

    Parameters:
        covariance: The recorded ``covariance_px2``.

    Returns:
        The matrix, each entry a finite float, or None when the value is absent
        or is no square matrix of real numbers.  A ragged matrix, a
        non-square one, and one holding a NaN, an infinity, a string or a
        boolean are each refused whole rather than repaired: a covariance
        missing a term is not the covariance anybody fitted, and half of one is
        worse than none because it looks like a measurement.
    """
    if not isinstance(covariance, list) or len(covariance) == 0:
        return None
    rows: list[list[float]] = []
    for row in covariance:
        if not isinstance(row, list) or len(row) != len(covariance):
            return None
        values = [finite_float(entry) for entry in row]
        if any(value is None for value in values):
            return None
        rows.append(cast(list[float], values))
    return rows


def _cmatrix_or_none(value: Any) -> list[float] | None:
    """Store the nine row-major values a recorded rotation denotes to its readers.

    The matrix is assembled by the readers' own function and nothing else is
    asked of it here.  A column that re-decided what nine real numbers are
    would be a second reader of the record, and the second answer is the one
    that drifts: a rotation written in a nesting the readers assemble and this
    column refused would be applied through a document and be NULL in a row.
    Whether the matrix that survives is a proper rotation is decided by the one
    validator both readers apply to it, not by this column.

    Parameters:
        value: The recorded matrix, row-major or nested, or None.

    Returns:
        The nine values row-major, or None when the readers can make no 3x3
        matrix of finite real numbers from the recorded value.
    """
    matrix = record_rotation_matrix(value)
    if matrix is None:
        return None
    nine: list[float] = matrix.reshape(9).tolist()
    return nine


def _source_names_from_feature_ids(feature_ids: Any) -> list[str]:
    """Body / ring / catalog names parsed from curated feature ids.

    Feature ids follow ``kind:NAME[...]`` (``body_disc:IAPETUS``,
    ``ring_edge:SATURN:feature_135_ieg:IEG``, ``star:UCAC4:10230452``).

    Parameters:
        feature_ids: The technique's recorded feature id list.

    Returns:
        The distinct names, sorted.
    """
    names: set[str] = set()
    if not isinstance(feature_ids, list):
        return []
    for feature_id in feature_ids:
        if not isinstance(feature_id, str):
            continue
        parts = feature_id.split(':')
        if len(parts) >= 2 and parts[1]:
            names.add(parts[1])
    return sorted(names)


def subtree_of(results_path_stub: str) -> str | None:
    """The subtree a stub names, or None when it names none.

    Every table recording a file fills its subtree column from here, so no two
    of them can disagree about which directory a stub is under.

    Parameters:
        results_path_stub: The stub.

    Returns:
        The first path segment, or None for a stub with no separator -- which
        is what the simulated dataset's bare scene basenames produce.
    """
    subtree, separator, _rest = results_path_stub.partition('/')
    return subtree if separator else None


def _technique_rows(
    source: DocumentOrigin, per_technique: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Build the ``techniques`` rows of one document.

    Parameters:
        source: Where the document came from.
        per_technique: The recorded ``navigation_result.per_technique`` list.

    Returns:
        One row per technique that reported, in recorded order.

    Raises:
        MetadataDocumentError: If an entry carries no ``technique_name``, if two
            entries carry the same one, or if an entry's ``diagnostics`` is not
            an object.
    """
    rows: list[dict[str, Any]] = []
    named: set[str] = set()
    for entry in per_technique:
        # The name is the identity of the row -- it is half the primary key of
        # ``techniques``, and every report groups and joins on it -- so an entry
        # with none has no identity and two entries with one name have the same
        # one.  Standing a nameless entry in as "unknown" manufactured the
        # second case out of the first, and either way the database refuses the
        # insert and ends the run.  Numbering the duplicates instead would put a
        # technique nobody ran into the operator's report.
        name = _str_or_none(entry.get('technique_name'))
        if name is None:
            _refuse('navigation_result.per_technique[] carries no technique_name', source)
        if name in named:
            _refuse('navigation_result.per_technique[] names one technique twice', source)
        named.add(name)
        offset_dv, offset_du = _pair(entry.get('offset_px'))
        rows.append(
            {
                'root_url': source.root_url,
                'results_path_stub': source.results_path_stub,
                'technique_name': name,
                'offset_dv': offset_dv,
                'offset_du': offset_du,
                # Whole, as on the image: a reader that wants the per-axis
                # sigmas takes the square roots of the diagonal, and one that
                # wants the correlation between the axes can still have it.
                'covariance_px2': _covariance_or_none(entry.get('covariance_px2')),
                'confidence': finite_float(entry.get('confidence')),
                'spurious': bool(entry.get('spurious')),
                'at_edge': bool(entry.get('at_edge')),
                # An empty list is a statement: this technique named no source.
                'source_names': _source_names_from_feature_ids(entry.get('feature_ids')),
                # As is an empty object: it reported no diagnostics.
                'diagnostics': _object(
                    entry.get('diagnostics'),
                    'navigation_result.per_technique[].diagnostics',
                    source,
                ),
            }
        )
    return rows


def _feature_source_rows(
    source: DocumentOrigin, inventory: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Build the ``feature_sources`` rows of one document.

    The inventory is aggregated by ``(feature_type, source_model, source_name)``;
    per-feature identity and gating detail is not retained.

    Parameters:
        source: Where the document came from.
        inventory: The recorded ``navigation_result.feature_inventory`` list.

    Returns:
        One row per distinct feature type and source, in key order.
    """
    counts: dict[tuple[str, str, str], list[int]] = {}
    for entry in inventory:
        feature_id = str(entry.get('feature_id', ''))
        parts = feature_id.split(':')
        name = parts[1] if len(parts) >= 2 and parts[1] else '(none)'
        key = (
            str(entry.get('feature_type', 'unknown')),
            str(entry.get('source_model', 'unknown')),
            name,
        )
        tally = counts.setdefault(key, [0, 0])
        tally[0] += 1
        tally[1] += int(bool(entry.get('gated')))
    return [
        {
            'root_url': source.root_url,
            'results_path_stub': source.results_path_stub,
            'feature_type': feature_type,
            'source_model': source_model,
            'source_name': source_name,
            'n_features': tally[0],
            'n_gated': tally[1],
        }
        for (feature_type, source_model, source_name), tally in sorted(counts.items())
    ]


def facts_from_document(metadata: dict[str, Any], source: DocumentOrigin) -> ImageFacts:
    """Flatten one metadata document into the facts its consumers read.

    Refuses a file that is not a navigation document, by raising
    :class:`MetadataDocumentError`, and answers for one that is.  The refusals
    are the shapes the document schema declares and nothing else.

    A fault in this code is not turned into a refusal.  A refusal is recorded
    against the file with its modification time and size, so the next pass skips
    it: a defect here written down that way would outlive its own fix, still
    saying the file was never a navigation document, while every run after it
    reported itself clean over a tree an image is missing from.  It is a
    property of this code rather than of the file, and it ends the pass instead
    -- which is the same answer
    :func:`~spindoctor.nav_records.document.document_or_refusal` gives for a
    fault in reading the file at all.

    The offset comes from the document's top-level ``offset`` and is stored as
    written.  ``navigation_result.offset_px`` is the same offset rounded for
    display and is deliberately not what the index carries, because every
    consumer applies the value rather than reading it.

    ``status`` comes from the document's top-level ``status`` and from nowhere
    else, so that the column holds ``'success'`` exactly when the document did.
    A reader that rebuilds a record from the row and classifies it gets the same
    answer as one that reads the document, which it could not if the column
    stood in for a field the document did not have.

    ``status_error`` and ``status_reason`` are different vocabularies and are
    kept in different columns: ``status_error`` is what a selection filter
    matches verbatim, ``status_reason`` is the navigator's explanation of a
    non-success outcome.

    Parameters:
        metadata: Parsed metadata JSON as written by ``navigate_image_files``.
        source: Where the document came from and what the walk saw of it.

    Returns:
        The image's own values and the two lists of child mappings.

    Raises:
        MetadataDocumentError: If the document lacks the observation image name
            or the observation instrument, if any container the schema declares
            holds something of another shape, or if its technique entries do not
            each carry a distinct name.  That is what a file which is not a
            per-image navigation document looks like.
    """
    observation = _object(metadata.get('observation'), 'observation', source)
    image_name = _str_or_none(observation.get('image_name'))
    if image_name is None:
        _refuse('no observation.image_name', source)
    instrument = _str_or_none(observation.get('instrument'))
    if instrument is None:
        _refuse('no observation.instrument', source)
    nav = _object(metadata.get('navigation_result'), 'navigation_result', source)
    provenance = _object(nav.get('provenance'), 'navigation_result.provenance', source)
    classifier = _object(nav.get('image_classifier'), 'navigation_result.image_classifier', source)
    times = _object(nav.get('times'), 'navigation_result.times', source)
    pointing = _object(nav.get('pointing'), 'navigation_result.pointing', source)
    # An image's epoch is its observation's midtime, which the navigator
    # records as provenance.  A document written for an image that never
    # loaded has no provenance and so no epoch, and reads as None here.
    image_et = finite_float(provenance.get('image_et'))
    per_technique = _array_of_objects(
        nav.get('per_technique'), 'navigation_result.per_technique', source
    )
    inventory = _array_of_objects(
        nav.get('feature_inventory'), 'navigation_result.feature_inventory', source
    )
    excluded = _array(
        nav.get('excluded_from_consensus'), 'navigation_result.excluded_from_consensus', source
    )
    if not all(isinstance(name, str) for name in excluded):
        _refuse(
            'navigation_result.excluded_from_consensus holds a name that is not a string', source
        )
    timing = _object(metadata.get('timing'), 'timing', source)
    shape_v, shape_u = _image_shape(observation.get('image_shape'))
    # Read through the readers' own function, so the pair stored is the pair a
    # consumer would apply and nothing is stored where a consumer would apply
    # nothing.  The reason it names is the reader's business, not a column's.
    recorded_offset = record_offset(metadata)
    offset_dv, offset_du = (
        recorded_offset.pair if recorded_offset.pair is not None else (None, None)
    )
    # Likewise read through the consumers' own function.  It answers the word a
    # record naming no error is reported under, and that is a record with
    # nothing to store rather than one whose error is that word.
    recorded_error = record_status_error(metadata)
    status_error = None if recorded_error == UNKNOWN_STATUS else recorded_error
    sigma_dv, sigma_du = _pair(nav.get('sigma_px'))

    image_row: dict[str, Any] = {
        'root_url': source.root_url,
        'results_path_stub': source.results_path_stub,
        'subtree': subtree_of(source.results_path_stub),
        'image_name': image_name,
        'instrument': instrument,
        # Present whenever the dataset index supplied it, including for an
        # image that never loaded; NULL only when the dataset has no camera
        # column at all.  It is never inferred from the image name.
        'camera': _str_or_none(observation.get('camera')),
        'shutter_mode': _str_or_none(observation.get('shutter_mode')),
        'image_path': _str_or_none(observation.get('image_path')),
        # A copy of the epoch the document recorded as provenance, which is
        # what a rebuilt record carries it back as and what the report
        # aggregates into the time span it gives per instrument, with the
        # calendar date a bound compares rendered from it.
        'image_et': image_et,
        'image_date': date_from_image_et(image_et),
        # The document's own top-level field, read by the function every
        # consumer reads it through and never stood in for by the copy inside
        # ``navigation_result``.  Borrowing the nested copy would make the
        # column say ``success`` for a document that never did, and a reader
        # classifying the rebuilt record would then apply a corrected pointing
        # the same record read as a file supplies no pointing at all.
        'status': record_status(metadata),
        # Read through the consumer's own function and stored as NULL exactly
        # where that function reports the record as naming no error, which is
        # what a rebuilt record's absent field then says.  A column deciding
        # for itself which fields name an error would be a second reader of
        # this one, agreeing until one of the two changed.
        'status_error': status_error,
        # A copy of the document's own field, stored whether or not the error
        # beside it was one this code names: the traceback belongs to the
        # failure the document recorded, not to the vocabulary it was
        # classified under.
        'status_traceback': _str_or_none(metadata.get('status_traceback')),
        'status_reason': _str_or_none(nav.get('status_reason')),
        'offset_dv': offset_dv,
        'offset_du': offset_du,
        'sigma_dv': sigma_dv,
        'sigma_du': sigma_du,
        'covariance_px2': _covariance_or_none(nav.get('covariance_px2')),
        'sigma_along_unobservable_px': finite_float(nav.get('sigma_along_unobservable_px')),
        'rotation_deg': finite_float(nav.get('rotation_deg')),
        'sigma_rotation_deg': finite_float(nav.get('sigma_rotation_deg')),
        'confidence': finite_float(metadata.get('confidence')),
        'confidence_rank': _str_or_none(nav.get('confidence_rank')),
        'n_techniques': len(per_technique),
        # In the order the document wrote it.  Nothing here re-orders a
        # recorded list: a consumer comparing this column against the document
        # it came from finds the list that document holds, and one wanting some
        # other order applies it.  An empty list is a statement: the ensemble
        # excluded nothing.
        'excluded_from_consensus': list(excluded),
        'image_class': _str_or_none(classifier.get('class')),
        'noise_sigma': finite_float(classifier.get('noise_sigma')),
        'image_shape_v': shape_v,
        'image_shape_u': shape_u,
        'run_start': _str_or_none(timing.get('start_iso8601')),
        'run_end': _str_or_none(timing.get('end_iso8601')),
        'elapsed_s': finite_float(timing.get('elapsed_s')),
        'peak_memory_bytes': _byte_count_or_none(timing.get('peak_memory_bytes')),
        'config_hash': _str_or_none(provenance.get('config_hash')),
        'git_sha': _str_or_none(provenance.get('spindoctor_git_sha')),
        'pipeline_run': _str_or_none(provenance.get('pipeline_run_iso8601')),
        # Stored only as the list of text the document holds, so that a
        # consumer reading the rebuilt record refuses the same malformed
        # provenance the document would have made it refuse.  A block that is
        # not a list of names is stored as none at all rather than as an empty
        # one, which is a run that named no kernels.
        'spice_kernels': _text_list_or_none(provenance.get('spice_kernels')),
        'image_number': image_number_from_name(image_name),
        'start_et': finite_float(times.get('start_et')),
        'stop_et': finite_float(times.get('stop_et')),
        # Read from the document rather than computed from the shutter epochs
        # beside it: a reader gates this value against the observation's own
        # midtime to a microsecond, so the column has to carry what was
        # recorded rather than a value that reproduces it only as long as one
        # producer's arithmetic stays what it is.
        'midtime_et': finite_float(times.get('midtime_et')),
        'exposure_s': finite_float(times.get('exposure_s')),
        'sclk_start': _str_or_none(times.get('sclk_start')),
        'sclk_midtime': _str_or_none(times.get('sclk_midtime')),
        'sclk_stop': _str_or_none(times.get('sclk_stop')),
        'camera_frame': _str_or_none(pointing.get('camera_frame')),
        'camera_frame_id': _int_or_none(pointing.get('camera_frame_id')),
        'ck_frame_id': _int_or_none(pointing.get('ck_frame_id')),
        'cmatrix': _cmatrix_or_none(pointing.get('cmatrix')),
        'cmatrix_original': _cmatrix_or_none(pointing.get('cmatrix_original')),
        'source_file': source.source_file,
        'mtime_ns': source.mtime_ns,
        'size_bytes': source.size_bytes,
    }
    return ImageFacts(
        image=image_row,
        techniques=_technique_rows(source, per_technique),
        feature_sources=_feature_source_rows(source, inventory),
    )
