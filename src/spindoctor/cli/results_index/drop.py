"""Dropping a results index from the command line.

The destructive half of the statistics system.  It opens the database the URL
names, finds out whether that database holds a SpinDoctor index at all, says
what of one is in it, asks whoever typed the command whether that is what they
meant, removes those tables and stops.  It never walks a results tree: dropping
is a deliberate act, not the opening move of a long ingest, and a command that
did both would make a mistyped URL expensive twice over.

The account of what is about to go is written to the log, which is where
everything else ``sd_results_index`` does is written.  The one thing this module
puts on standard output itself is the question, which ``input`` writes there: a
prompt is a dialogue rather than a report, and carrying it in the log alone
would leave a run whose log was routed to a file waiting silently for an answer.

**A database has to prove it holds an index before anything is dropped from
it.**  The proof is the index's own stamp table, and the schema that stamp was
found in is the only schema anything is dropped from.  A database holding tables
that merely share these names -- ``images`` above all -- is refused and named,
because nothing can tell somebody else's table from the remains of an index of
ours, and a destructive command must not decide such a thing on its own.

Two questions this answers by reporting rather than by refusing:

**An ingest run that has not finished does not stop a drop.**  Such a run is
either a pass writing the index at this moment or one that died, and nothing in
the index tells the two apart -- there is no heartbeat and no process to ask.
A pass that died is also the commonest reason to want a drop, so refusing on
that evidence would withhold the command from the case that needs it most, in
order to guard a case the confirmation already guards.  It is therefore counted
and said out loud, before the question is asked, so that the person who is about
to shoot a live ingest is told so while there is still an answer to give.  What
a drop under a live pass costs is that pass, which fails on a table that has
gone and leaves nothing behind: an unfinished run already reads as "not
ingested" to every consumer, so no reader is told anything different before and
after.

**Another process holding the database does not stop a drop either.**  Neither
backend can always tell whether another process holds the database: SQLite's readers take
no lock to observe under write-ahead logging, and a PostgreSQL role need not be
allowed to read the server's activity view.  What can be done is to make the
attempt fail rather than hang, which
:data:`~spindoctor.results_index.drop.DROP_LOCK_TIMEOUT_MS` does for the reading
and for the drop alike, and to leave nothing half-finished when it does, which
the transaction around the drop does on both backends.  So the database itself
decides, promptly and per table, instead of a guess deciding beforehand.
"""

from pdslogger import PdsLogger
from sqlalchemy.exc import SQLAlchemyError

from spindoctor.results_index import (
    IndexContents,
    drop_index_tables,
    index_contents,
    masked_url,
    open_database,
)

__all__ = ['AGREEMENT', 'drop_results_index']

AGREEMENT = ('y', 'yes')
"""The answers that mean yes.  Anything else, including nothing, means no.

Compared after the line is stripped and lower-cased, so ``Y``, ``YES`` and a
line with spaces around it are the same answer as ``yes``.
"""

_PROMPT = 'Drop {tables} table(s) and {rows} row(s) from {url}, schema {schema}?{unfinished} [y/N] '
"""What the operator is asked.

The question carries the facts an answer turns on rather than only the verb.
The account above it goes to the log, which is where the whole of what this run
did belongs and which is on the terminal in an ordinary run -- but a run whose
log has been routed to a file still puts this line in front of the person
typing, and a question that read only "are you sure?" would be one they had to
answer out of memory.  The schema is one of those facts: on a server the same
URL reaches several, and which one holds the index is not something the command
line said.

``sd_results_index`` carries a main logger and reports through it; this is a
dialogue rather than a report, and the prompt is written to standard output by
``input`` itself.
"""

_UNFINISHED = ' {count} ingest run(s) have not finished.'
"""The one fact from the account that is repeated into the question.

A pass may be writing the index at this moment, and the drop would end it.  It
is the only thing here that a person answering could not have known from the
command they typed.
"""

_FAILURE_CAUSES = (
    ('55P03', 'Another session is holding a lock on one of these tables.'),
    (
        '2BP01',
        'Another object of this database depends on one of these tables, and has to be '
        'removed, or stop depending on it, first.',
    ),
    (
        '42501',
        'This account does not own one of these tables, and a table is dropped by its owner.',
    ),
    (
        '42P01',
        'One of these tables went between the reading of what this database held and the '
        'drop of it.',
    ),
    ('SQLITE_BUSY', 'Another process is holding the write lock on this SQLite database.'),
    ('SQLITE_READONLY', 'This SQLite database is read-only.'),
    (
        'SQLITE_CONSTRAINT_FOREIGNKEY',
        'A table outside the index carries a reference into one of these tables.',
    ),
)
"""What a database's own failure code says the cause was.

A destructive command that names the wrong cause is worse than one that names
none: it sends whoever reads it to grant a privilege over a lock, or to hunt a
session over a view.  So each code is answered with what it means and nothing
is answered with a guess -- a code not in this table is reported as the database
worded it, with no cause invented for it.

PostgreSQL codes are the five-character SQLSTATE the server returns; SQLite
codes are the result-code names its driver carries, matched by prefix because
SQLite refines several of them into extended forms.
"""


def _failure_code(exc: BaseException) -> str:
    """Return the code a database driver's own exception carries.

    Parameters:
        exc: The exception to read, either a driver exception or the wrapper
            SQLAlchemy raised around one.

    Returns:
        The SQLSTATE or SQLite result-code name, or an empty string for a
        failure that carries neither -- a connection that dropped, or an
        exception SQLAlchemy raised on its own behalf.
    """
    original = getattr(exc, 'orig', exc)
    for attribute in ('sqlstate', 'sqlite_errorname'):
        code = getattr(original, attribute, None)
        if isinstance(code, str) and code:
            return code
    return ''


def _because(exc: SQLAlchemyError) -> str:
    """Return what the database said the cause was, ready to append to a message.

    Parameters:
        exc: The failure to diagnose.

    Returns:
        One sentence with a leading space, or an empty string when the database
        gave no code this recognizes and there is therefore nothing to say
        beyond what it said itself.
    """
    code = _failure_code(exc)
    for prefix, cause in _FAILURE_CAUSES:
        if code.startswith(prefix):
            return f' {cause}'
    return ''


def _summary(safe_url: str, contents: IndexContents) -> list[str]:
    """Return the account of what a drop is about to remove.

    Written for a person deciding whether to answer yes, so it leads with what
    is lost rather than with what is done: the tables by name, how many rows are
    in each of them, and the two facts that change the answer -- a schema
    version that is not the one this code reads, which is the case the drop
    exists for, and an ingest run that nothing has finished.

    Parameters:
        safe_url: The index URL with its credentials already masked.
        contents: What the database holds of the index.

    Returns:
        The lines, in the order they are said.
    """
    lines = [
        f'About to drop the SpinDoctor results index tables from {safe_url}, '
        f'schema {contents.schema}'
    ]
    lines.extend(f'    {table.name}: {table.rows} row(s)' for table in contents.tables)
    lines.append(f'    {len(contents.tables)} table(s), {contents.rows} row(s) in all')
    if contents.schema_version is None:
        lines.append('This database carries no readable schema version stamp.')
    else:
        lines.append(f'This database is stamped with schema version {contents.schema_version}.')
    if contents.unfinished_runs:
        lines.append(
            f'{contents.unfinished_runs} ingest run(s) have begun and not finished. Either a '
            f'pass is writing this index now, which the drop would end, or one died; nothing '
            f'recorded here tells the two apart.'
        )
    lines.append(
        f'Nothing else in schema {contents.schema} is touched, and no other schema of this '
        f'database is looked at. The metadata documents are the source of truth, so a dropped '
        f'index is rebuilt by running sd_results_index again.'
    )
    return lines


def _answer(safe_url: str, contents: IndexContents) -> str | None:
    """Put the question to whoever typed the command.

    Parameters:
        safe_url: The index URL with its credentials already masked.
        contents: What the database holds of the index.

    Returns:
        The line that was typed, or None when there was nobody to type one.
        Three ways of having no standard input are read the same way, because
        the drop's answer to all of them is the same: one at its end, which is
        what a scheduled run redirected from nowhere has; one whose file has
        been closed under it; and none at all, which is what a service manager
        that hands its children no input leaves behind.
    """
    unfinished = (
        _UNFINISHED.format(count=contents.unfinished_runs) if contents.unfinished_runs else ''
    )
    question = _PROMPT.format(
        tables=len(contents.tables),
        rows=contents.rows,
        url=safe_url,
        schema=contents.schema,
        unfinished=unfinished,
    )
    try:
        return input(question)
    except (EOFError, OSError, RuntimeError):
        return None


def _nothing_of_ours(safe_url: str, contents: IndexContents, logger: PdsLogger) -> int:
    """Report a database that proved it holds no index of SpinDoctor's, and drop nothing.

    Two states reach this, and they are not the same answer.  A database with no
    table of these names in it is the state a drop was asked for and exits 0.
    One holding tables of these names with no stamp of ours over them is a URL
    naming something that is not a SpinDoctor index, which is refused: the
    tables are either somebody else's -- ``images`` is not a name anyone owns --
    or what is left of an index whose stamp has gone, and nothing in the
    database says which.

    The first of the two is reported as a fact about this connection rather than
    about the database.  What was asked is what an unqualified name on this
    connection reaches, so an index in a schema outside its search path, or in
    one this account may not look into, answers the same way as one that is not
    there.  The status is 0 all the same, because the state asked for -- no
    index reachable through this URL -- is the state found.

    Parameters:
        safe_url: The index URL with its credentials already masked.
        contents: What the database holds of the index, proving nothing.
        logger: Logger the account is written to.

    Returns:
        The exit status: 0 when there was nothing of these names, 1 when there
        was and none of it could be shown to be the index's.
    """
    if not contents.unproven:
        logger.info(
            'This connection to %s reaches none of the results index tables, so nothing was '
            'dropped. What was looked at is what the connection reaches: an index in a schema '
            'outside its search path, or one this account may not look into, is not reported '
            'here and is not what was dropped. An index that is not there and one that has '
            'been dropped are the same thing to every program that reads one.',
            safe_url,
        )
        return 0
    logger.fatal(
        'Nothing was dropped: %s holds table(s) the results index also uses (%s), but no '
        "schema_meta of SpinDoctor's stands over them, so nothing here says this database "
        'holds a results index. They are either tables somebody else created under names '
        'this index also uses, or what an index whose stamp has gone left behind, and the '
        'database does not say which. Check the URL; remove them by hand if they are an '
        'index of yours.',
        safe_url,
        ', '.join(contents.unproven),
    )
    return 1


def drop_results_index(url: str, *, assume_yes: bool, logger: PdsLogger) -> int:
    """Remove every table of a results index, having said what that costs.

    Which index is being dropped is recorded before it is opened, since a run
    log that says only that a database could not be opened leaves out the one
    thing a reader has to check: which of the three resolution levels supplied
    the URL, and which URL it was.

    No step of this answers an interrupt with a traceback.  Opening the
    database, reading what it holds and dropping the tables can each wait on
    something -- a server, a lock, a scan of a large table -- and Ctrl-C during
    any of them is reported as the refusal it is, naming which step stopped and
    saying that nothing went.

    Parameters:
        url: Connection URL of the index to drop.
        assume_yes: Whether to drop without asking.  A run with nobody at the
            terminal needs this; without it, a standard input that is at its end
            is treated as a refusal rather than as consent.
        logger: Logger the whole account is written to, whether or not anybody
            was asked.

    Returns:
        The exit status: 0 when the tables were dropped and 0 again when the
        database held none of them, since an index that is already gone is the
        state asked for; 1 when the database could not be opened or read, when
        it holds tables of these names that nothing proves are the index's, when
        a table would not drop, when the operator answered anything but yes, and
        when any step of it was interrupted.
    """
    safe_url = masked_url(url)
    logger.info('Results index to drop the tables of: %s', safe_url)
    try:
        engine = open_database(url)
    except ValueError as exc:
        logger.fatal('Cannot open the database to drop the results index from: %s', exc)
        return 1
    except KeyboardInterrupt:
        logger.fatal('Nothing was dropped from %s: opening it was interrupted.', safe_url)
        return 1
    try:
        # Said before rather than after, because reading it counts the rows of
        # every table, which on a production-sized index is a scan apiece: a
        # confirmation that appears minutes after the command was typed is
        # otherwise a command that looks hung.
        logger.info('Reading what %s holds of the results index', safe_url)
        try:
            contents = index_contents(engine)
        except SQLAlchemyError as exc:
            logger.fatal(
                'Nothing was dropped: what %s holds of the results index could not be read '
                '(%s: %s).%s',
                safe_url,
                type(exc).__name__,
                exc,
                _because(exc),
            )
            return 1
        except KeyboardInterrupt:
            # The reading is the step that can run for minutes, so it is the
            # step most likely to be interrupted, and a destructive command owes
            # an interrupt the same line every other refusal gets rather than a
            # traceback.  Nothing had been dropped yet, which is what the line
            # says.
            logger.fatal(
                'Nothing was dropped from %s: the reading of what it holds was interrupted.',
                safe_url,
            )
            return 1
        if contents.schema is None:
            return _nothing_of_ours(safe_url, contents, logger)
        for line in _summary(safe_url, contents):
            logger.info('%s', line)
        if not assume_yes:
            try:
                answer = _answer(safe_url, contents)
            except KeyboardInterrupt:
                # The one refusal a person makes with a key rather than a word.
                # It reaches here as an exception rather than as an answer, and
                # a destructive command owes it the same line every other
                # refusal gets rather than a traceback.
                logger.fatal('Nothing was dropped from %s: the question was interrupted.', safe_url)
                return 1
            if answer is None:
                logger.fatal(
                    'Nothing was dropped from %s: there is nobody to confirm with, since '
                    'standard input is at its end. Pass --yes to drop without being asked.',
                    safe_url,
                )
                return 1
            if answer.strip().lower() not in AGREEMENT:
                logger.info(
                    'Nothing was dropped from %s: the answer was %r rather than yes.',
                    safe_url,
                    answer,
                )
                return 1
        try:
            dropped = drop_index_tables(engine, contents)
        except SQLAlchemyError as exc:
            logger.fatal(
                'The drop of %s did not complete (%s: %s).%s The transaction was taken back, '
                'so that database is exactly as it was.',
                safe_url,
                type(exc).__name__,
                exc,
                _because(exc),
            )
            return 1
        except KeyboardInterrupt:
            # An interrupt while the drop waits on a lock is the one way this
            # step ends without a database error, and it is the moment at which
            # the reassurance matters most: the transaction is taken back on
            # both backends, so nothing of the index has gone.
            logger.fatal(
                'The drop of %s was interrupted. The transaction was taken back, so that '
                'database is exactly as it was.',
                safe_url,
            )
            return 1
        logger.info('Dropped from %s, schema %s: %s', safe_url, contents.schema, ', '.join(dropped))
        logger.info(
            'That index is now what one nobody has ingested into looks like. Run '
            'sd_results_index to build it again.'
        )
    finally:
        engine.dispose()
    return 0
