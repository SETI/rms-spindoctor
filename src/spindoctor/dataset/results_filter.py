"""Filter image selections against existing navigation result files.

Implements the ``--has-offset-file`` / ``--has-no-offset-file`` /
``--has-offset-error`` / ``--has-no-offset-error`` /
``--has-offset-spice-error`` / ``--has-offset-nonspice-error`` image selection
options shared by the PDS3 datasets.

The error filters, including the negative one, ask what a document records, so
each of them requires the document to exist: an image nothing has been written
for records no error, and is selected by ``--has-no-offset-file`` rather than by
``--has-no-offset-error``.

Every one of them is answered through the record seam, which is what makes one
implementation serve both storages.  Two of the seam's questions cover all six
flags, and they are asked at different moments because they need different
things.

A listing says which images have a navigation document.  It opens no document,
so it is the cheap question over either storage, and it is asked in one of two
ways depending on what the run is selecting.

A run selecting images that *have* a document -- ``--has-offset-file``, and every
error filter, each of which reads what a document records and so requires one to
exist -- lists the selected subtrees once, when the filter is built.  Testing an
image afterwards is then a set lookup, and the same listing is what the run log
reports the root as holding.

A run selecting images that have *none* -- ``--has-no-offset-file`` -- asks about
the candidates the enumeration produced, in batches, naming their stubs.  It has
to be asked that way to be worth asking: the run keeps the images the results
root holds nothing for, so a listing of the whole of every selected volume would
produce a set used only to reject from, and a run whose other constraints name
ten images would pay for fifty thousand entries to answer about ten.  What the
seam does with the stubs it is handed is the seam's own business: it checks the
named files where a check is a syscall and walks their directories where it is a
paid round trip, and where it walks, one walk of a root serves every batch of the
same scan.

What a document records is asked of the per-image facts, in batches, as the
enumeration offers its candidates.  It has to be: an error filter reads a
document, and which documents to read is the set of candidate images, which the
other selection constraints decide and which is not known when the filter is
built.  Asked of the subtrees instead, a run whose other constraints keep one
image in a hundred would read every document under them and discard almost all
of it -- on a cloud results root, one paid download apiece.  So a batch of
candidates names its stubs, and only images that passed the listing are ever
named, which is why nothing here reads a document twice or reads one for an
image no filter could keep.

The facts are the values a results index holds in its columns, so a document is
narrowed on exactly what a row is narrowed on and the two storages agree by
construction rather than by two pieces of code that agree today.  Which storage
answers is settled by
:func:`spindoctor.results_index.open_record_source`: a run naming a results
index reads rows, and a run naming none reads the documents themselves.

A candidate whose file yields no facts satisfies no error filter, the one phrased
in the negative included, because what it records is unknown rather than known to
be an outcome.  A document written to an earlier metadata schema is one of those,
so a run over a results root holding nothing else can select no image at all.
The scan therefore says how many candidates it could read no record out of and
names one of them with the reason, once, when it is closed: counted rather than
reported one at a time because a real results root holds hundreds of them, and
reported at all because a selection that is short for this reason is otherwise
indistinguishable from a root holding no such image.

The listing taken when the filter is built covers the subtrees the enumeration
selected and no others, one at a time.  A subtree the results root does not hold
is an ordinary state -- a volume nobody has navigated yet has no directory under
the results root -- so it contributes no documents rather than ending the run,
while a subtree that is there and will not be read still does.  Asking about one
subtree at a time is what allows that distinction: a listing of several ends at
the first one it cannot read, and the subtrees after it would go unasked.

A scan that names its candidates needs no such restriction, and applies none.
It asks about the images the enumeration accepted, which come from the selected
volumes because that is what the enumeration walked, so naming a volume as well
would narrow nothing.  A volume the results root has no directory for still
costs nothing and ends nothing: the images under it are images the root holds no
document for, which is what the seam answers about a named file whose directory
is not there.

What the index answers differently
----------------------------------

The index holds what one ingest pass could read and record, so the answers below
are bounded by that rather than by the filters.  Each is stated here, in the
plan, in the navigation guide's account of ``--results-index-db``, and in a test of
its own, and one found later is added in the same four places rather than left
to be rediscovered.  The guide is one of them because an operator reading it is
the person a silently short selection is served to: an enumeration a user is
never shown answers nobody's question about the selection they got.

- **A file the pass could not retrieve** has no row at all in the index and
  reads as absent, which is what the absence filters read as "this image was
  never navigated".  Nothing is recorded for it deliberately: a recorded row
  would be skipped for as long as the file did not change, and a download that
  failed once says nothing that will still be true next pass.

  Two other ways a file could go unrecorded are not divergences, because
  neither leaves a completed pass behind it: an ingest that cannot list a
  directory stops there, and one whose writer the database refuses a document's
  rows stops there.  So a root with a completed pass is a root every directory
  of which was listed and every document of which was stored.

- **A document rewritten in place, keeping the length and the modification time
  it had before,** keeps the row the document before it produced, so an error
  filter answers from what that one recorded.  Those two metrics are everything
  a listing supplies about a file, and reading the file to find out whether it
  needs reading is the retrieval the skip exists to avoid, so no number of
  completed passes corrects this one: an ingest told to read every document
  regardless is what puts the row right.  A tree restored by a copy that
  preserves times, a document patched and stamped back from a sibling, and a
  backend reporting one modification time for two writes all produce it; an
  ordinary re-navigation writes a different length at a later time and does not.

The index is also a snapshot: it answers as of the last ingest over the root, so
a document written since is one it does not hold and a document deleted since is
one it still holds.  When that pass finished is reported with the answer,
because outside the members above that is what decides whether the answer is the
answer the tree would give.  Inside them the age decides nothing: each of those
survives a pass that finished a second ago, which is why each is stated here
rather than left to be read off the stamp.
"""

from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType

from filecache import FCPath
from pdslogger import PdsLogger

from spindoctor.nav_records import (
    ImageFacts,
    ListedRecord,
    NavRecord,
    RecordSource,
    Selection,
    TreeTuning,
    UnlistableDirectoryError,
    UnreadableFile,
)
from spindoctor.results_index import open_record_source, snapshot_finish_time

from .dataset import ImageFile

__all__ = [
    'FATAL_STATUS',
    'RESULTS_FILTER_BATCH_SIZE',
    'SPICE_STATUS_ERROR',
    'ResultsFilter',
    'SelectionError',
]

RESULTS_FILTER_BATCH_SIZE = 64
"""How many candidate images one question about their documents covers.

The enumeration buffers this many accepted images before asking what their
documents record, so the cost of asking is paid once for the batch rather than
once for each image.  It bounds a batch of candidates rather than a batch of
downloads: what the seam does with the stubs it is handed, and how many files or
rows it fetches at a time, is the seam's own business.
"""

FATAL_STATUS = 'error'
"""Value of ``status`` that the error filters select on.

An image whose navigation failed outright.  The other statuses describe a run
that finished, whatever it concluded, and no error filter selects one.
"""

SPICE_STATUS_ERROR = 'missing_spice_data'
"""Value of ``status_error`` that the SPICE error filters tell apart.

Matched verbatim, which is what makes ``status_error`` a field of its own: it is
the navigator's machine-readable classification of a fatal error, distinct from
the prose of ``status_reason``.
"""


class SelectionError(ValueError):
    """A selection this run cannot be given, as opposed to one that went wrong.

    Raised for a contradictory combination of selection flags, and for a results index
    that cannot be opened, cannot be read, or holds no completed ingest of the
    results root.  Each is a run that is misconfigured, each already carries a
    message saying what to change, and a program is therefore free to report
    that message instead of tracing back.

    It is a :class:`ValueError` because that is the family every caller of this
    module already catches.  It is a type of its own so that a program reporting
    the message catches these and not every other ``ValueError`` an enumeration
    can raise: a malformed index label, a value the walk could not convert, or
    an outright programming error is a run that went wrong, and its traceback is
    the useful thing about it.
    """


def _named_flags(names: Sequence[str]) -> str:
    """Render the flags a refusal is about, in the spelling the caller passed.

    Parameters:
        names: The flag names, in the order they are declared.

    Returns:
        The names joined into a phrase, so that a message can name every flag
        that made the selection unsatisfiable rather than a category the reader
        has to look up.
    """
    if len(names) == 1:
        return names[0]
    return f'{", ".join(names[:-1])} and {names[-1]}'


def _file_count(count: int) -> str:
    """Render a number of files so that a line reads for one of them as for many.

    Parameters:
        count: How many files.

    Returns:
        The count and the noun agreeing with it.
    """
    return f'{count} file' if count == 1 else f'{count} files'


def _image_count(count: int) -> str:
    """Render a number of images so that a line reads for one of them as for many.

    Parameters:
        count: How many images.

    Returns:
        The count and the noun agreeing with it.
    """
    return f'{count} image' if count == 1 else f'{count} images'


def _elapsed_phrase(seconds: float) -> str:
    """Return an interval in the coarsest unit that has a whole number of it.

    Parameters:
        seconds: The interval.

    Returns:
        The interval named in days, hours or minutes, or as less than a minute.
    """
    for size, unit in ((86400.0, 'day'), (3600.0, 'hour'), (60.0, 'minute')):
        if seconds >= size:
            count = int(seconds // size)
            return f'{count} {unit}' if count == 1 else f'{count} {unit}s'
    return 'less than a minute'


def _snapshot_age(ingested_utc: str | None) -> str:
    """Return how old an index's answer is, in the terms an operator acts on.

    The index answers as of the pass that filled it and detects no change since,
    so the age of that pass is what says whether its answer is the answer the
    tree would give.  Both the stamp and the interval are reported: the stamp
    names the pass to re-run, and the interval is what a reader compares against
    what they know they have navigated.

    Parameters:
        ingested_utc: When the newest pass over the root finished, as the index
            recorded it, or None when it recorded nothing.

    Returns:
        A phrase naming that time and how long ago it was.  A stamp that will
        not parse, or one in the future because two clocks disagree, is reported
        as it stands: a reader can act on a value the index really holds, and an
        interval computed from a stamp nobody can read would be a fiction.  One
        that is empty or nothing but spaces reads as no stamp at all, because
        the phrase it would otherwise be reported as is a blank space in the
        line.  A stamp carrying no offset is read as UTC, which is what every
        pass writes and the only reading under which the interval means
        anything; one written as local time is reported an offset out.
    """
    if ingested_utc is None or not ingested_utc.strip():
        return 'at a time this index does not record'
    try:
        finished = datetime.fromisoformat(ingested_utc)
    except ValueError:
        return ingested_utc
    if finished.tzinfo is None:
        finished = finished.replace(tzinfo=UTC)
    elapsed = (datetime.now(UTC) - finished).total_seconds()
    if elapsed < 0:
        return ingested_utc
    return f'{ingested_utc} ({_elapsed_phrase(elapsed)} ago)'


@contextmanager
def _where_a_subtree_the_root_lacks_holds_nothing() -> Iterator[None]:
    """Let a subtree the results root does not hold contribute no documents.

    A volume nobody has navigated has no directory under the results root, which
    is an ordinary state of a results tree and not a reason to end an
    enumeration.  Every other way a directory refuses to be listed -- this user
    may not read it, the share it lives on has gone away, it stopped being a
    directory -- means the filter cannot see what is under it, and answering
    from what it did see would silently select images it has no evidence about.

    The walk refuses all of them alike, because to a walk they mean one thing:
    there may be documents here it cannot see.  The distinction is the caller's,
    and it is read off the failure the storage layer raised underneath.

    Yields:
        Nothing; this wraps the asking rather than supplying anything to it.

    Raises:
        UnlistableDirectoryError: If the directory is there and would not be
            read, which is the refusal that must reach the operator.
    """
    try:
        yield
    except UnlistableDirectoryError as exc:
        if not isinstance(exc.__cause__, (FileNotFoundError, NotADirectoryError)):
            raise


class ResultsFilter:
    """Filters candidate images against their navigation result files.

    Constructed once per enumeration when any of the results-based selection
    flags is active.  Construction validates the flag combination (the flags
    AND together; a combination carrying a contradictory pair raises, naming
    every contradiction it carries rather than the first one found) and, for a
    run selecting images that have a document, lists the selected subtrees, so
    that testing an image afterwards is a set lookup and costs nothing.

    The filter is applied in two stages, because the questions need different
    things:

    - :meth:`passes` is the per-image test, answered from that listing.  It
      settles ``has_offset_file`` outright, and for an error filter it settles
      the half that requires the document to exist.  A run that lists no
      subtrees keeps every image it is offered here and settles the whole of its
      selection in the stage below.
    - :meth:`filter_batch` asks about a batch of candidates by name: which of
      them the results root holds a document for, when the run selects the ones
      it holds none for, and what their documents record, when an error filter
      is active.  Both are asked of the candidates rather than of the subtrees,
      because the candidates are what a run might still keep and the other
      selection constraints are what decide them.
      :attr:`needs_batch_filtering` says whether it has anything to do.  An
      error filter reading documents hands each kept image the record it read,
      so the stage that goes on to process the image reads no document twice.

    A filter with a second question to ask holds its storage open until it is
    closed, so it is usable as a context manager and an enumeration closes it
    when it is done.
    """

    def __init__(
        self,
        volumes: Iterable[str],
        nav_results_root: str | Path | FCPath,
        *,
        has_offset_file: bool = False,
        has_no_offset_file: bool = False,
        has_offset_error: bool = False,
        has_no_offset_error: bool = False,
        has_offset_spice_error: bool = False,
        has_offset_nonspice_error: bool = False,
        results_index_db_url: str | None = None,
        logger: PdsLogger,
        tuning: TreeTuning | None = None,
    ) -> None:
        """Validates the flag combination and asks what the results root holds.

        Parameters:
            volumes: Volume names selected by the other constraints; only these
                subdirectories of the results root are asked about, whichever
                storage answers.
            nav_results_root: Root of the navigation results tree; may be a
                cloud URL.  A ``str`` or ``Path`` is normalized to an
                :class:`FCPath` at construction, which is the type the seam and
                the run log are both given it as.
            has_offset_file: Only keep images whose offset metadata file exists.
            has_no_offset_file: Only keep images whose offset metadata file does
                not exist.
            has_offset_error: Only keep images whose offset metadata file
                indicates a fatal error (``status == 'error'``).
            has_no_offset_error: Only keep images whose offset metadata file
                records a status other than the fatal one, which for the
                documents this pipeline writes is the images whose navigation
                ran to a result of any kind.  Like every other error filter this
                one asks what a document records, so it keeps only images that
                have one: an image nothing has been written for records no error
                and is selected by ``has_no_offset_file``.  So is a file no
                per-image facts could be read out of, for the reason
                :meth:`_records_a_wanted_error` gives.
            has_offset_spice_error: Only keep images whose offset metadata file
                indicates a fatal error from missing SPICE data.
            has_offset_nonspice_error: Only keep images whose offset metadata
                file indicates a fatal error other than missing SPICE data.
            results_index_db_url: Connection URL of a results index to answer every
                filter from, or None to read the results tree.  A URL that
                cannot be used is an error rather than a reason to fall back
                to the tree.
            logger: Logger for scan statistics, for the candidates no record
                could be read for, and for the one line the seam has to say
                about a directory it declined to descend twice.
            tuning: How much of a pass over the results tree runs at once,
                which the enumeration resolves from its configuration; None is
                the library's own defaults.  Ignored when an index answers.

        Raises:
            SelectionError: If the flag combination is contradictory, or if the
                results index cannot be opened, cannot be read, or holds no
                completed ingest of this results root.
        """
        # Declaration order, so that every message names the flags in the order
        # the options themselves are documented in.  A refusal that names one
        # flag against the others it excludes leads with that one and lists the
        # rest in this order, which is what says which exclusion is claimed.
        reading_a_document = [
            name
            for name, given in (
                ('has_offset_error', has_offset_error),
                ('has_no_offset_error', has_no_offset_error),
                ('has_offset_spice_error', has_offset_spice_error),
                ('has_offset_nonspice_error', has_offset_nonspice_error),
            )
            if given
        ]
        naming_an_error = [name for name in reading_a_document if name != 'has_no_offset_error']
        contradictions: list[str] = []
        if has_offset_file and has_no_offset_file:
            contradictions.append(
                'has_offset_file and has_no_offset_file are mutually exclusive: one image '
                'cannot both have an offset metadata file and have none'
            )
        if has_offset_spice_error and has_offset_nonspice_error:
            contradictions.append(
                'has_offset_spice_error and has_offset_nonspice_error are mutually exclusive: '
                'one fatal error either came from missing SPICE data or did not'
            )
        if has_no_offset_file and reading_a_document:
            # Whichever of the four was given is named, because "the
            # offset-error filters" is a category the reader would have to look
            # up, and one of its members is a flag whose own name says "no
            # error" -- so a message that only named the category would leave a
            # user who typed --has-no-offset-error reading about something else.
            # "Cannot be combined with" rather than "are mutually exclusive":
            # the exclusion runs between this flag and each of the others, and
            # the others are satisfiable together -- has_offset_error with
            # has_offset_spice_error is a row of the table this refusal is
            # tested against.  Named as a set, the message would assert an
            # exclusion between every pair of them, which is false.
            named = _named_flags(reading_a_document)
            asks = 'asks' if len(reading_a_document) == 1 else 'ask'
            contradictions.append(
                f'has_no_offset_file cannot be combined with {named}: {named} {asks} what an '
                'offset metadata file records, which requires the file to exist'
            )
        if has_no_offset_error and naming_an_error:
            # Each of the three names a document that records a fatal error, of
            # any kind or of one kind, and this one names a document that
            # records none: no document is both, so the combination is a
            # selection nothing could ever satisfy rather than a narrow one.
            # The three are not exclusive of each other, so this reads the same
            # way as the refusal above: one flag against the ones it excludes.
            contradictions.append(
                f'has_no_offset_error cannot be combined with {_named_flags(naming_an_error)}: '
                'one document cannot both record a fatal error and record none'
            )
        if contradictions:
            # Every contradiction the flags carry, not the first one found: a
            # selection refused one pair at a time costs the user a run per pair,
            # and a flag left out of the message reads as one the run accepted.
            raise SelectionError('; '.join(contradictions))
        self._has_no_offset_file = has_no_offset_file
        self._has_no_offset_error = has_no_offset_error
        self._has_offset_spice_error = has_offset_spice_error
        self._has_offset_nonspice_error = has_offset_nonspice_error
        # An error filter reads what a document records, so it keeps only images
        # that have one.  Folding it into the presence half is what makes that
        # true and what keeps a batch from ever naming an image with no
        # document: the listing has already excluded it.
        self._needs_metadata_read = bool(reading_a_document)
        self._needs_offset_presence = has_offset_file or self._needs_metadata_read
        # One type for the two things the root is used for: naming the root to
        # the seam, which resolves it to a single spelling, and naming it in the
        # run log.  Nothing is read through it, because the storage the seam
        # opens is the thing that reads.
        self._nav_results_root = (
            nav_results_root if isinstance(nav_results_root, FCPath) else FCPath(nav_results_root)
        )
        self._logger = logger
        self._results_index_db_url = results_index_db_url
        self._tuning = tuning
        # Held open only while there is a second question to ask, since a source
        # over an index holds a connection pool and a run that has nothing left
        # to ask should not.
        self._source: RecordSource | None = None
        # What the scan owes the operator about the candidates it could read no
        # record out of.  Kept as a count and one example rather than reported
        # as they are met: a results root whose documents an error filter can
        # read nothing out of holds hundreds of them, and a line apiece would
        # bury the selection the run was asked for.
        self._unreadable_count = 0
        self._unreadable_example: UnreadableFile | None = None
        # What an absence scan has asked about and how much of it was already
        # navigated, which is what the run log owes an operator whose selection
        # was answered a batch at a time.
        self._candidates = 0
        self._documented = 0
        # None where nothing was listed at construction, which is a filter that
        # keeps every image it is offered here: a run with no results-based flag
        # at all, and one selecting the images this root holds no document for,
        # which asks about its candidates instead.
        self._stubs: frozenset[str] | None = None
        if self._needs_offset_presence:
            # Fixed here rather than left as it came: an iterator is emptied by
            # whichever pass reads it first.
            self._stubs = self._documented_stubs(tuple(volumes))
        elif has_no_offset_file:
            self._source = self._held_open()

    def __enter__(self) -> 'ResultsFilter':
        """Enter an enumeration's use of this filter.

        Returns:
            The filter itself.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Leave an enumeration's use of this filter, closing it.

        Parameters:
            exc_type: The exception's class, when the enumeration is leaving on one.
            exc: The exception, when the enumeration is leaving on one.
            traceback: Its traceback, when the enumeration is leaving on one.
        """
        self.close()

    def close(self) -> None:
        """Report what the scan found and what it read nothing out of, and release it.

        Called when the enumeration is done with the filter, however it ended,
        which is the moment the scan is over and its counts are final.  A filter
        with no second question to ask never held a storage open, and closing
        twice costs nothing and says nothing twice.
        """
        self._report_candidates()
        self._report_unreadable()
        if self._source is not None:
            self._source.close()
            self._source = None

    @property
    def needs_batch_filtering(self) -> bool:
        """Whether :meth:`filter_batch` has anything to do.

        The enumeration reads this to decide whether to buffer accepted images
        into batches, so that one question covers many, or to yield each one as
        it is accepted.

        Returns:
            True when an error filter is active, which needs a document read,
            and when the run selects the images this root holds no document for,
            which asks about its candidates by name.  ``has_offset_file`` alone
            is settled by the listing taken when the filter was built.
        """
        return self._needs_metadata_read or self._has_no_offset_file

    def passes(self, results_path_stub: str) -> bool:
        """True if the image satisfies the filters the construction listing settles.

        A set lookup: the results root was listed once when this filter was
        built, so an enumeration offering a million candidates pays for that
        answer once.  It settles ``has_offset_file`` outright, and for an error
        filter it settles the half that requires the document to exist -- what
        the document records is :meth:`filter_batch`'s.

        Parameters:
            results_path_stub: The image's results path stub (relative to the
                results root, no suffix).

        Returns:
            True if the image is still a candidate.  Every image is when nothing
            was listed at construction: a run with no results-based flag has
            nothing to test one against, and one selecting the images this root
            holds no document for tests them in :meth:`filter_batch`, against
            the answer to a question about the candidates themselves.
        """
        if self._stubs is None:
            return True
        return results_path_stub in self._stubs

    def filter_batch(self, image_files: list[ImageFile]) -> list[ImageFile]:
        """Apply the filters asked of a batch of candidates, in input order.

        One question per batch, naming the candidates' stubs, so what is asked
        about is the images a run might still keep rather than everything under
        the subtrees it enumerated.

        Two filters are asked here, and never both: a run selecting the images
        the results root holds no document for asks which of the batch it holds
        one for, and an error filter asks what the batch's documents record.
        Every stub an error filter names passed :meth:`passes`, so each of them
        has a document and none is read for an image already excluded.

        An error filter answered out of the documents hands each kept image the
        record its own document was read out of, so the stage that processes the
        image reads no document a second time.  It is attached to the images
        that are kept and to no others, and nothing is attached by a filter
        answered out of a results index: a record is rebuilt there from the
        column set its consumer declares, which is not the one an error filter
        reads, so the two reads want different columns and stay separate.

        A record lives exactly as long as the image it is attached to, which
        bounds how many are held at once.  An enumeration that yields its images
        one at a time holds one batch of them, so at most
        :data:`RESULTS_FILTER_BATCH_SIZE` records are live however many images
        the run covers.  One that pools its candidates and samples from the pool
        holds the records of every image it has kept so far, which its own
        requested sample count bounds: it stops at that many, so it holds at most
        that many plus the remainder of the batch the last one came out of.

        Parameters:
            image_files: Candidates that already passed :meth:`passes`.

        Returns:
            Those of them the active filters keep, in input order.  With neither
            asked here the batch is returned as it was given.

        Raises:
            SelectionError: If a results index stops answering while the
                enumeration reads it, which is the same refusal failing to read
                it at the start is.
        """
        if not image_files:
            return image_files
        if self._has_no_offset_file:
            return self._undocumented(image_files)
        if not self._needs_metadata_read:
            return image_files
        matching = self._matching_records(tuple(image.results_path_stub for image in image_files))
        kept: list[ImageFile] = []
        for image in image_files:
            if image.results_path_stub not in matching:
                continue
            record = matching[image.results_path_stub]
            if record is not None:
                image.nav_record = record.metadata
            kept.append(image)
        return kept

    def _undocumented(self, image_files: list[ImageFile]) -> list[ImageFile]:
        """Keep the candidates of one batch the results root holds no document for.

        Parameters:
            image_files: The batch, in the order the enumeration accepted it.

        Returns:
            Those of them nothing has been written for, in input order.

        Raises:
            SelectionError: If a results index stops answering while this reads
                it.
        """
        documented = self._documented_of(tuple(image.results_path_stub for image in image_files))
        self._candidates += len(image_files)
        # One count per image rather than one per document, because the two
        # halves of the reported line are counted against each other: a batch
        # offering one stub twice holds two candidates and one document, and a
        # count of the documents would say that fewer of them were already
        # navigated than were.
        self._documented += sum(1 for image in image_files if image.results_path_stub in documented)
        return [image for image in image_files if image.results_path_stub not in documented]

    def _documented_of(self, stubs: Sequence[str]) -> frozenset[str]:
        """Ask which of one batch of candidates the storage holds a document for.

        Parameters:
            stubs: The candidates' results path stubs.

        Returns:
            Those the storage holds a file for.  A file that is there and says
            nothing readable is one of them, because presence is a question
            about the file and not about what is in it.

        Raises:
            SelectionError: If a results index stops answering while this reads
                it.
        """
        source = self._source
        if source is None:  # pragma: no cover - closed only when nothing asks again
            raise SelectionError('this selection filter has been closed and cannot answer')
        try:
            return frozenset(
                listed.stub for listed in source.listing(Selection(stubs=tuple(stubs)))
            )
        except ValueError as exc:
            raise SelectionError(str(exc)) from exc

    def _open(self) -> RecordSource:
        """Open the storage this run resolved.

        Returns:
            The source, which reads rows when the run named a results index and
            documents when it named none.

        Raises:
            SelectionError: If the index cannot be opened, or holds no completed
                ingest of this results root.
        """
        try:
            return open_record_source(
                [self._nav_results_root],
                results_index_db_url=self._results_index_db_url,
                logger=self._logger,
                tuning=self._tuning,
            )
        except ValueError as exc:
            # Every way the index refuses to answer arrives as a ValueError
            # carrying a message that says what to change; this is the boundary
            # where it becomes the type a program can report on without
            # catching every other ValueError an enumeration raises.
            raise SelectionError(str(exc)) from exc

    def _held_open(self) -> RecordSource:
        """Open the storage a scan will ask about its candidates, and say so.

        The scan has counted nothing yet, so what is reported is what a run over
        an index is owed whatever it goes on to find: how old the answer it is
        about to be given is.

        Returns:
            The source, which the filter holds until it is closed.

        Raises:
            SelectionError: If the index cannot be opened, cannot be read, or
                holds no completed ingest of this results root.  Whatever was
                opened is released before any failure leaves this method, since
                a constructor that raises hands the caller no filter to close.
        """
        source = self._open()
        try:
            self._report(None)
        except BaseException:
            source.close()
            raise
        return source

    def _documented_stubs(self, subtrees: Sequence[str]) -> frozenset[str]:
        """List the selected subtrees and report what they hold.

        Parameters:
            subtrees: The subtrees of the results root to list.

        Returns:
            The stub of every image the results root has a document for.  A file
            that is there and says nothing readable is one of them, because
            presence is a question about the file and not about what is in it.

        Raises:
            SelectionError: If the index cannot be opened, cannot be read, or
                holds no completed ingest of this results root.  Whatever was
                opened is released before any failure leaves this method, since
                a constructor that raises hands the caller no filter to close.
        """
        source = self._open()
        try:
            try:
                # Read inside the same guard as the open, because a source
                # reading rows runs its query as the caller reads the stream: a
                # storage that stops answering surfaces here rather than above.
                stubs = frozenset(listed.stub for listed in self._listed(source, subtrees))
            except ValueError as exc:
                raise SelectionError(str(exc)) from exc
            # Reported while this frame is still the only thing holding the
            # source, and before the filter is given it.  Reporting is a second
            # question of a results index and can refuse the selection; a
            # constructor that raises hands the caller no object, so a source
            # stored on the filter first would be released by nothing.
            self._report(len(stubs))
        except BaseException:
            source.close()
            raise
        if self._needs_metadata_read:
            self._source = source
        else:
            source.close()
        return stubs

    def _matching_records(self, stubs: Sequence[str]) -> dict[str, NavRecord | None]:
        """Ask what one batch of candidates' documents record, and match them.

        Parameters:
            stubs: The candidates' results path stubs.

        Returns:
            One entry per stub whose document satisfies the active error
            filters, mapped to the record the facts were read out of -- which is
            a record when the storage answered from a document, and None when it
            answered from a row and there is none to hand back.  A file no
            per-image facts could be read out of is not among them, for the
            reason :meth:`_records_a_wanted_error` gives; it is counted instead,
            so that :meth:`close` can say how much of the batch went that way.

        Raises:
            SelectionError: If a results index stops answering while this reads
                it.
        """
        source = self._source
        if source is None:  # pragma: no cover - closed only when nothing asks again
            raise SelectionError('this selection filter has been closed and cannot answer')
        matching: dict[str, NavRecord | None] = {}
        try:
            for facts in source.facts(Selection(stubs=tuple(stubs))):
                if isinstance(facts, UnreadableFile):
                    self._note_unreadable(facts)
                    continue
                if self._records_a_wanted_error(facts):
                    matching[str(facts.image['results_path_stub'])] = facts.record
        except ValueError as exc:
            raise SelectionError(str(exc)) from exc
        return matching

    def _note_unreadable(self, unreadable: UnreadableFile) -> None:
        """Count a candidate no record came out of, and keep the first as the example.

        The first rather than the last, so that the file named is the same one
        whichever batch the enumeration stopped in, and so that the count and
        the example are one file's worth of state rather than a list of every
        file a mission-wide scan passed over.

        Parameters:
            unreadable: The file, and why no record came out of it.
        """
        self._unreadable_count += 1
        if self._unreadable_example is None:
            self._unreadable_example = unreadable

    def _report_unreadable(self) -> None:
        """Say how many candidates yielded no record, with one of them and its reason.

        Nothing is said by a scan that read a record out of every candidate it
        was offered, and nothing is said twice: the count and the example are
        cleared once reported, so a filter closed again reports no second scan.
        """
        example = self._unreadable_example
        if example is None:
            return
        self._logger.warning(
            '*** Results scan: %s under %s yielded no navigation record, which no error '
            'filter selects; for example %s: %s',
            _file_count(self._unreadable_count),
            self._nav_results_root,
            example.path,
            example.reason,
        )
        self._unreadable_count = 0
        self._unreadable_example = None

    def _report(self, documents: int | None) -> None:
        """Say what the results root was found to hold, and how current that is.

        Parameters:
            documents: How many offset metadata files the root holds under the
                selected subtrees, or None for a scan that asks about its
                candidates as it meets them and has therefore counted nothing
                yet.  What such a scan found is reported when it is over, by
                :meth:`_report_candidates`.

        Raises:
            SelectionError: If the index cannot be read for the age of its
                answer, which is the same refusal reading it for the answer is.
        """
        if self._results_index_db_url is None:
            if documents is None:
                self._logger.info(
                    '*** Results scan will ask %s about each candidate image',
                    self._nav_results_root,
                )
            else:
                self._logger.info(
                    '*** Results scan found %d offset metadata files under %s',
                    documents,
                    self._nav_results_root,
                )
            return
        try:
            ingested_utc = snapshot_finish_time(self._results_index_db_url, self._nav_results_root)
        except ValueError as exc:
            raise SelectionError(str(exc)) from exc
        if documents is None:
            self._logger.info(
                '*** Results index answers about the images under %s, ingested %s',
                self._nav_results_root,
                _snapshot_age(ingested_utc),
            )
            return
        self._logger.info(
            '*** Results index holds %d offset metadata files under %s, ingested %s',
            documents,
            self._nav_results_root,
            _snapshot_age(ingested_utc),
        )

    def _report_candidates(self) -> None:
        """Say how much of what a scan asked about was already navigated.

        A scan that asks about its candidates never counts the root, so this is
        what stands in the place of the count a construction listing reports:
        how many images the run offered and how many of them already had a
        document.  Nothing is said by a filter that asked about none, and
        nothing is said twice: the counts are cleared once reported.
        """
        if not self._candidates:
            return
        self._logger.info(
            '*** Results scan asked about %s under %s, %d of which had a navigation document',
            _image_count(self._candidates),
            self._nav_results_root,
            self._documented,
        )
        self._candidates = 0
        self._documented = 0

    def _listed(self, source: RecordSource, subtrees: Sequence[str]) -> Iterator[ListedRecord]:
        """Yield what the storage holds under each selected subtree in turn.

        One subtree per call rather than one call naming all of them, so that a
        subtree the results root does not hold costs only itself.  A single call
        ends at the first subtree it cannot read and the ones after it go
        unasked, which for an enumeration over volumes nobody has navigated yet
        would be every volume after the first.

        Parameters:
            source: The storage this run resolved.
            subtrees: The subtrees of the results root to ask about.

        Yields:
            One entry per document, subtree by subtree.
        """
        for subtree in subtrees:
            with _where_a_subtree_the_root_lacks_holds_nothing():
                yield from source.listing(Selection(subtrees=(subtree,)))

    def _records_a_wanted_error(self, facts: ImageFacts) -> bool:
        """True if what a document records satisfies the error filters.

        Read from the per-image facts rather than from the raw document, which
        is what makes the two storages agree: the facts are the values a results
        index holds in its columns, so a document is narrowed on exactly what a
        row is narrowed on.

        A file no facts could be read out of never reaches here.  It is excluded
        from every error filter, the one phrased in the negative included: what
        such a file records is unknown rather than known to be an outcome.

        Parameters:
            facts: What the document says about its image.

        Returns:
            True if the image is selected by the active error filters.
        """
        status = facts.image.get('status')
        if self._has_no_offset_error:
            return status != FATAL_STATUS
        if status != FATAL_STATUS:
            return False
        status_error = facts.image.get('status_error')
        if self._has_offset_spice_error and status_error != SPICE_STATUS_ERROR:
            return False
        return not (self._has_offset_nonspice_error and status_error == SPICE_STATUS_ERROR)
