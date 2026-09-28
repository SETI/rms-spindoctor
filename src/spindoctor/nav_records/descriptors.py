"""How large a retrieval batch the open-file limit allows.

A batch of documents retrieved from a remote root costs its own size in open
file descriptors.  The downloader takes a lock file for every document in the
batch before it downloads any of them, and releases none until the last one has
arrived, so a batch of a thousand holds a thousand descriptors at its peak, on
top of the temporary file and the connection each download thread holds.
Nothing in the downloader's interface says so, and nothing in it bounds the
batch by what the process may open, so a batch larger than the limit fails
partway through with an operating-system error naming a lock file rather than a
setting.

That makes the open-file limit a third bound on ``retrieve_batch_size``,
alongside the two the setting already states: it must be at least
``retrieve_threads`` or the download pool never fills, and it bounds what a
round of retrieval holds in memory.  The third bound is the one an operator
cannot see, because it is not a property of the tree, the link or the
configuration but of the shell the run started from.  A login shell on a
typical Linux system allows 1024 open files, which the configured batch size
alone can exhaust.

Two things are therefore done here.  The soft limit is raised toward the hard
limit, which a process may always do for itself and which on a normal system
lifts the ceiling far above any batch; and the pass is then held to what the
limit actually permits, so that a run under a hard limit too low for the
configured batch retrieves in smaller batches instead of failing.  A pass only
ever loses speed this way, never documents.

Only a pass over a remote root is touched at all.  A local file is read where
it lies and no lock is taken for it, so a local pass spends no descriptor per
document however large the batch, and has neither its tuning held down nor the
process's limit raised on its account.
"""

import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from filecache import FCPath
from pdslogger import PdsLogger

from spindoctor.nav_records.tuning import TreeTuning

__all__ = [
    'RESERVED_DESCRIPTORS',
    'UNBOUNDED_SOFT_LIMIT',
    'raise_open_file_limit',
    'within_open_file_limit',
]


RESERVED_DESCRIPTORS = 256
"""How many open files are left for everything a retrieval does not hold.

A pass holds more open than the locks and downloads of the batch in flight: the
index it writes, the logs it writes, the standard streams, and whatever the
interpreter itself holds open.  The batch and the threads are counted
separately, because how many there are is configured; this covers the rest,
generously, since the cost of reserving more than a pass needs is a smaller
batch and the cost of reserving less is the failure this exists to prevent.
"""

UNBOUNDED_SOFT_LIMIT = 1 << 20
"""What the soft limit is raised to where the hard limit is unbounded.

An unbounded hard limit is permission to raise the soft one as far as is
useful rather than a reason to leave it alone, and the soft one is what a run is
actually held to.  Some systems refuse an unbounded soft limit on open files
outright, so a finite target is the one that works everywhere; this one is far
above any batch a pass retrieves.
"""

_DESCRIPTORS_PER_THREAD = 2
"""How many open files one download thread holds while it works.

The file it is writing and the connection it is reading, both for as long as
the download lasts, which is inside the window where the whole batch's locks
are also held.
"""


if sys.platform == 'win32':  # pragma: no cover - the limit is Unix-only

    def _open_file_limits() -> tuple[int, int] | None:
        """Return the soft and hard limits on open files.

        Returns:
            None, because this platform has no such limit to report.
        """
        return None

    def _set_soft_open_file_limit(soft: int, hard: int) -> None:
        """Set the soft limit on open files, leaving the hard limit alone.

        Parameters:
            soft: What to raise the soft limit to.
            hard: The hard limit, which is written back unchanged.
        """

    _UNLIMITED = -1
    """What an unbounded limit reads as, which this platform never reports."""

else:
    import resource

    def _open_file_limits() -> tuple[int, int] | None:
        """Return the soft and hard limits on open files.

        Returns:
            The pair, soft first.
        """
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        return soft, hard

    def _set_soft_open_file_limit(soft: int, hard: int) -> None:
        """Set the soft limit on open files, leaving the hard limit alone.

        Parameters:
            soft: What to raise the soft limit to.
            hard: The hard limit, which is written back unchanged.

        Raises:
            OSError: If the system refuses the limit.
            ValueError: If the limit is not one this process may ask for.
        """
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))

    _UNLIMITED = resource.RLIM_INFINITY
    """What an unbounded limit reads as."""


def raise_open_file_limit() -> int | None:
    """Raise this process's open-file limit as far as it is allowed to go.

    A process may raise its own soft limit up to its hard limit without
    privilege, and a pass over a remote tree wants far more open files at once
    than a login shell grants by default.  Doing it here rather than asking the
    operator to do it in the shell means a run behaves the same whichever shell
    started it.

    The hard limit being unbounded is a reason to raise the soft one, not to
    leave it alone: it is the soft limit a run is held to, and an unbounded hard
    limit is exactly the case where raising it always succeeds.

    Calling this more than once is harmless: the second call finds the limit
    where the first one put it.

    Returns:
        The soft limit in force afterwards, or None where there is no limit to
        be held to -- a platform that has none, or a soft limit already
        unbounded -- in which case nothing is held back on account of one.
    """
    limits = _open_file_limits()
    if limits is None:
        return None
    soft, hard = limits
    if soft == _UNLIMITED:
        return None
    target = UNBOUNDED_SOFT_LIMIT if hard == _UNLIMITED else hard
    if soft >= target:
        return soft
    try:
        _set_soft_open_file_limit(target, hard)
    except (OSError, ValueError):
        # The limit stays where it was and the pass is sized against it.
        return soft
    return target


def within_open_file_limit(
    tuning: TreeTuning,
    roots: Sequence[str | Path | FCPath],
    *,
    logger: PdsLogger | None = None,
) -> TreeTuning:
    """Return the tuning a pass over these roots can actually run at.

    The tuning unchanged unless the pass reads a remote root and the open-file
    limit cannot cover what the configuration asks for, in which case a copy
    asking for as much as the limit does cover.  A pass peaks at one descriptor
    per document in the batch it is retrieving, plus two per download thread,
    plus what :data:`RESERVED_DESCRIPTORS` covers, so it is the sum of those
    that the limit has to leave room for and both settings are held down to fit.

    A pass only ever loses speed this way.  Every document is still retrieved,
    in more batches or on fewer threads, which is why a limit lower than the
    configuration wants is reported rather than refused: it is the shell the
    run started from, not a mistake in the run.

    Parameters:
        tuning: How much of the pass the configuration asks to run at once.
        roots: The results roots the pass will read.  A pass over local roots
            alone is returned untouched and does not raise the process's limit,
            because a local file is read where it lies and no lock is taken
            for it.
        logger: Where to say that the limit held the pass back, said once for
            the pass rather than once for each batch of it.  None says nothing,
            which is what a caller with no logger of its own needs.

    Returns:
        The tuning to run the pass at, which is the one given whenever the
        limit covers it.
    """
    if all(FCPath(root).is_local() for root in roots):
        return tuning
    limit = raise_open_file_limit()
    if limit is None:
        return tuning
    # What is left for the locks of a batch and the threads' files and
    # connections together.  Threads are given at most a quarter of it, so that
    # a thread count far above what the limit allows cannot leave the batch
    # with nothing.
    spare = limit - RESERVED_DESCRIPTORS
    threads = min(tuning.retrieve_threads, max(1, spare // (2 * _DESCRIPTORS_PER_THREAD)))
    batch = max(threads, min(tuning.retrieve_batch_size, spare - _DESCRIPTORS_PER_THREAD * threads))
    if batch == tuning.retrieve_batch_size and threads == tuning.retrieve_threads:
        return tuning
    if logger is not None:
        logger.info(
            'Open file limit of %d holds this pass to %d document(s) at a time on %d '
            'thread(s), rather than the configured %d on %d; this process has already '
            'raised its soft limit to its hard limit, so lifting it further needs '
            'ulimit -Hn or the container setting',
            limit,
            batch,
            threads,
            tuning.retrieve_batch_size,
            tuning.retrieve_threads,
        )
    return replace(tuning, retrieve_threads=threads, retrieve_batch_size=batch)
