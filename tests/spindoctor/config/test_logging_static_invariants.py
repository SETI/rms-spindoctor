"""Tests for logging invariants that hold across the source rather than in one run.

Some of what this design guarantees is an absence, and an absence is not
observable by exercising the code: a logger that should not be there does
nothing until the day something logs to it. These read the source instead, so
a conversion cannot be quietly undone.

Reading the source rather than importing the module is what lets these cover
whole packages, so a file added to one of them tomorrow is covered without
being named here, and lets them cover the GUI packages without importing
PyQt6 to do it. The complement is in ``test_log_scope.py``, which imports a
handful of named modules and inspects their attributes: that catches a logger
bound at run time, which no amount of reading imports will find.
"""

import ast
from pathlib import Path

import pytest
from filecache import FCPath

_SRC = FCPath(Path(__file__).resolve().parents[3]) / 'src' / 'spindoctor'

# The statistics report and the GUI programs carry no logger and write to the
# terminal with print(): a report's output is terminal text for a person reading
# it, and a GUI is read as it runs.  Ingest is deliberately not among them --
# it is infrastructure other programs depend on, so a partial or failed pass has
# to appear in a run log rather than only in an exit code -- so the statistics
# package is named by its report modules rather than as a whole.
_PRINT_ONLY = [
    'cli/stats/report.py',
    'cli/stats/report_common.py',
    'cli/stats/report_sections.py',
    'cli/sd_stats_report.py',
    'cli/sd_backplane_viewer.py',
    'cli/sd_create_simulated_image.py',
    'cli/sim_editor',
    'cli/sd_mosaic_display.py',
    'ui/mosaic_viewer',
]

_LOGGER_NAMES = frozenset({'IMAGE_LOGGER', 'MAIN_LOGGER'})

_RECORD_METHODS = frozenset(
    {
        'blankline',
        'critical',
        'debug',
        'dot_underscore',
        'ds_store',
        'error',
        'exception',
        'fatal',
        'hidden',
        'info',
        'invisible',
        'log',
        'normal',
        'summarize',
        'warn',
        'warning',
    }
)
"""The calls that put a line in a log, whatever object they are made on.

Every level the logging library offers, not the four a converted module happened
to use: a line written at a level nobody thought of is still a line in somebody's
run log, and ``warn`` beside ``warning`` is the same call under the older
spelling.
"""

_SECTION_METHODS = frozenset({'open', 'close'})
"""The section calls, which count only against something named as a logger.

``with logger.open(...):`` is the house idiom for a section and puts as much in a
log as an info call does, so it has to be here.  But a source, a cursor and a
file are closed by the same word, so unlike the names above these are matched
against what they are called on rather than against every object.
"""

_LOGGER_RECEIVER = 'logger'
"""What a lent logger is called, which is how a section call is recognized.

A layer that is lent a logger names the parameter for what it is, so the receiver
of a section call reads as ``logger``, ``self._logger`` or the like.  A layer
that hid one under another name would still be caught by any of the record
methods above, which are matched on every object.
"""

_MAY_WRITE_THROUGH_A_LENT_LOGGER = {
    'nav_records/walk.py': (('_claim', 'info'),),
}
"""Every place either data-access layer writes through a logger it was lent.

The walk, and only for the directory it declines to descend a second time: a
root is still wholly listed when that happens, and a run told nothing would read
the decline as documents that were never there.

Keyed by the module's path under the package rather than by its name, so a file
of the same name elsewhere is not exempted by coincidence, and holding the
enclosing function and the call rather than the file, so a second line added to
the same function is caught as readily as one added anywhere else.
"""

# Core packages route through NavBase.logger; the stdlib logging module is
# never imported there.
_NO_STDLIB_LOGGING = [
    'feature',
    'nav_model',
    'nav_orchestrator',
    'nav_technique',
    'support',
]

# The two data-access layers have no voice of their own: what either of them did
# is reported by whichever program called it, in that program's log.  So neither
# may name a program's logger -- that is a voice its caller did not configure --
# and neither may import the stdlib module, which the index layer's database
# library logs through and which a logger here would be the one place to wire
# into this design's ladder.
#
# What they may name is pdslogger, and only to type a logger a caller lends
# them.  The walk over the documents has one thing to say -- that it declined to
# descend a directory it had already listed under another name -- and a run that
# is not told it reads as one that covered the whole root, so the fact goes into
# the log of the program that asked.  The builder of a source takes that logger
# and passes it on, which is why the layer that builds the index-backed one
# names the type too.  Making a logger is a different thing and is refused for
# the whole source by test_the_loggers_are_the_only_two.
_NO_LOGGER_OF_ITS_OWN = [
    'nav_records',
    'results_index',
]


def _python_files(relative: str) -> list[FCPath]:
    """Return the Python files a target names.

    Parameters:
        relative: Path under the package, either a module or a directory.

    Returns:
        The files to inspect.
    """
    target = _SRC / relative
    return sorted(target.rglob('*.py')) if target.is_dir() else [target]


def _imported_names(path: FCPath) -> set[str]:
    """Return every name a module imports.

    Parsed rather than matched textually, so a name in a docstring or a
    comment explaining why it is absent does not count as a use.

    Parameters:
        path: The module to read.

    Returns:
        The imported names, including module names.
    """
    tree = ast.parse(path.read_text())
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            names.update(alias.name for alias in node.names)
            if node.module is not None:
                names.update(_dotted_prefixes(node.module))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.update(_dotted_prefixes(alias.name))
                # An aliased module hides the real name behind the alias, and
                # what it is later used for is an attribute access rather than
                # an import, so the alias is recorded as well.
                if alias.asname is not None:
                    names.add(alias.asname)
        elif isinstance(node, ast.Attribute):
            # "from spindoctor import config" then "config.MAIN_LOGGER" never
            # imports the logger by name; the attribute is where it surfaces.
            names.add(node.attr)
    return names


def _dotted_prefixes(dotted: str) -> set[str]:
    """Return a dotted module name and every package leading to it.

    ``import logging.handlers`` records only the full dotted name, so a check
    for ``logging`` would miss it.

    Parameters:
        dotted: A possibly dotted module name.

    Returns:
        The name and each of its prefixes.
    """
    parts = dotted.split('.')
    return {'.'.join(parts[: index + 1]) for index in range(len(parts))}


@pytest.mark.parametrize('relative', _PRINT_ONLY + _NO_STDLIB_LOGGING + _NO_LOGGER_OF_ITS_OWN)
def test_every_target_of_these_checks_exists(relative: str) -> None:
    """A target that has moved would make its check pass by finding nothing.

    Every assertion in this module is that a search came back empty, so a path
    that names nothing is indistinguishable from a path that is clean.
    """
    assert (_SRC / relative).exists()


def test_a_module_exempted_twice_is_read_in_the_order_the_scan_reports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The exemption is a set of places, not a sequence, and must not be read as one.

    The scan returns a module's sites sorted, so an exemption compared in the
    order it happens to be written in would fail a module that writes exactly
    what it is allowed to -- an exemption that has to be spelled in a particular
    order is a trap for whoever adds the second entry.

    Parameters:
        monkeypatch: Fixture the exemption table is amended through.
    """
    monkeypatch.setitem(
        _MAY_WRITE_THROUGH_A_LENT_LOGGER,
        'nav_records/walk.py',
        (('walk_from', 'warn'), ('walk_from', 'info')),
    )
    assert _exempt_sites(_SRC / 'nav_records' / 'walk.py') == [
        ('walk_from', 'info'),
        ('walk_from', 'warn'),
    ]


@pytest.mark.parametrize('relative', sorted(_MAY_WRITE_THROUGH_A_LENT_LOGGER))
def test_every_exempted_module_exists(relative: str) -> None:
    """An exemption naming nothing exempts nothing, and hides that it is dead.

    Parameters:
        relative: Path, relative to the source root, of the exempted module.
    """
    assert (_SRC / relative).exists()


@pytest.mark.parametrize('relative', _PRINT_ONLY)
def test_a_print_only_program_imports_no_logger(relative: str) -> None:
    """A program that reports through print() holds no logger to drift back to."""
    offenders = [
        f'{path.name}:{sorted(_LOGGER_NAMES & _imported_names(path))}'
        for path in _python_files(relative)
        if _LOGGER_NAMES & _imported_names(path)
    ]
    assert offenders == []


@pytest.mark.parametrize('relative', _PRINT_ONLY)
def test_a_print_only_program_imports_no_pdslogger(relative: str) -> None:
    """Nor does it reach around the loggers to pdslogger itself."""
    offenders = [
        path.name for path in _python_files(relative) if 'pdslogger' in _imported_names(path)
    ]
    assert offenders == []


@pytest.mark.parametrize('relative', _PRINT_ONLY)
def test_a_print_only_program_imports_no_stdlib_logging(relative: str) -> None:
    """Nor the stdlib module, which is the third way back to having a logger.

    Checking only for the two loggers and for pdslogger would let
    ``logging.getLogger(__name__)`` reintroduce exactly what these programs
    were converted away from, configured by nothing this design controls.
    """
    offenders = [
        path.name for path in _python_files(relative) if 'logging' in _imported_names(path)
    ]
    assert offenders == []


@pytest.mark.parametrize('package', _NO_STDLIB_LOGGING)
def test_core_code_does_not_import_stdlib_logging(package: str) -> None:
    """Core navigation code logs through pdslogger, never the stdlib module.

    The two do not share a level ladder or a handler set, so a stdlib logger
    here would be configured by nothing this design controls -- and in a cloud
    task it would reach the worker's terminal by a path isolation never sees.
    """
    offenders = [path.name for path in _python_files(package) if 'logging' in _imported_names(path)]
    assert offenders == []


@pytest.mark.parametrize('package', _NO_LOGGER_OF_ITS_OWN)
def test_a_data_access_layer_holds_no_logger_of_its_own(package: str) -> None:
    """Neither of the two program loggers, and not the stdlib module.

    Reaching for a program's logger would give a library layer a voice its
    caller did not configure, and the stdlib one would additionally turn on
    whatever the database library it sits over decided to say.  A logger a
    caller hands in is neither of those: it is the caller's own, configured by
    the caller's own command line, and it is the only way the one thing these
    layers have to say reaches the run that asked for it.

    Parameters:
        package: Path, relative to the source root, of the package to scan.
    """
    forbidden = _LOGGER_NAMES | {'logging'}
    offenders = [
        f'{path.name}:{sorted(found)}'
        for path in _python_files(package)
        if (found := forbidden & _imported_names(path))
    ]
    assert offenders == []


@pytest.mark.parametrize('package', _NO_LOGGER_OF_ITS_OWN)
def test_a_data_access_layer_writes_no_record_of_its_own(package: str) -> None:
    """Only where it was lent a logger to write one through.

    The import scan above cannot see this: a layer may name pdslogger to type a
    parameter, so a name it was handed is one it could also write through
    anywhere.  What keeps these layers quiet is that each of them writes through
    a lent logger at exactly the places listed here and nowhere else, so a line
    added elsewhere -- reporting a query, a row, a root -- is caught rather than
    discovered in somebody's run log.

    Parameters:
        package: Path, relative to the source root, of the package to scan.
    """
    offenders = sorted(
        f'{_under_source(path)}: writes at {_logging_call_sites(path)}, '
        f'exempt at {_exempt_sites(path)}'
        for path in _python_files(package)
        if _logging_call_sites(path) != _exempt_sites(path)
    )
    assert offenders == []


def test_the_loggers_are_the_only_two() -> None:
    """No third logger that can write a record is constructed anywhere in the package.

    Every record belongs to the run or to one image.  A logger outside those
    two is configured by none of the command line, the configuration, or the
    cloud-task isolation.  A null logger is not one of those and is not counted:
    it holds no handler and writes nowhere, and it is how a layer that must have
    no voice of its own spells the absence of one.
    """
    constructed = sorted(
        path.relative_to(_SRC).as_posix()
        for path in _SRC.rglob('*.py')
        if _constructs_a_logger(path)
    )
    assert constructed == ['config/log_scope.py', 'config/logger.py']


def _under_source(path: FCPath) -> str:
    """Return one module's path under the package, as the exemptions spell it.

    Parameters:
        path: The module.

    Returns:
        The path relative to the source root, with forward separators.
    """
    return path.relative_to(_SRC).as_posix()


def _exempt_sites(path: FCPath) -> list[tuple[str, str]]:
    """Return the sites one module is exempted at, ordered as the scan orders them.

    Sorted rather than taken in declaration order, because the scan returns its
    sites sorted: a module exempted at two places would otherwise fail on the
    order the exemption happens to be written in rather than on anything the
    module does.

    Parameters:
        path: The module.

    Returns:
        The exempt sites, sorted.
    """
    return sorted(_MAY_WRITE_THROUGH_A_LENT_LOGGER.get(_under_source(path), ()))


def _logging_call_sites(path: FCPath) -> list[tuple[str, str]]:
    """Return every call in a module that would put a line in a log, with where it is.

    Each site carries the function it is written in as well as the call, so an
    exemption names a place rather than a file, and a second line added beside an
    exempt one is a second site rather than the same one again.

    Parameters:
        path: The module to read.

    Returns:
        The sites, sorted, one entry per call and repeats kept.
    """
    return sorted(_sites_within(ast.parse(path.read_text()), '<module>'))


def _sites_within(node: ast.AST, enclosing: str) -> list[tuple[str, str]]:
    """Return the log-writing calls under one node, attributed to their function.

    Parameters:
        node: The node to search under.
        enclosing: Name of the function this node is written in.

    Returns:
        One entry per call, each naming the innermost function around it.
    """
    sites: list[tuple[str, str]] = []
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            sites.extend(_sites_within(child, child.name))
            continue
        method = _log_call_method(child)
        if method is not None:
            sites.append((enclosing, method))
        sites.extend(_sites_within(child, enclosing))
    return sites


def _log_call_method(node: ast.AST) -> str | None:
    """Return the log-writing method one node calls, or None when it calls none.

    A record method is matched against any object, because the object a lent
    logger arrives as has whatever name its parameter was given, and nothing
    else offers a method of one of those names.  A section call is matched only
    against something named as a logger, because a source and a cursor are
    closed by the same word.

    Parameters:
        node: The node to inspect.

    Returns:
        The method name, or None.
    """
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
        return None
    if node.func.attr in _RECORD_METHODS:
        return node.func.attr
    if (
        node.func.attr in _SECTION_METHODS
        and _LOGGER_RECEIVER in ast.unparse(node.func.value).lower()
    ):
        return node.func.attr
    return None


_WRITING_LOGGER_CLASSES = frozenset({'CriticalLogger', 'EasyLogger', 'ErrorLogger', 'PdsLogger'})
"""The logging library's classes whose instances can write a record.

``NullLogger`` is deliberately absent: it writes nowhere, so constructing one is
not acquiring a voice.  Every other class the library exports is one, and naming
the set rather than one class is what keeps a second spelling of the same act
from passing.
"""

_LOGGER_FACTORIES = frozenset({'get_logger', 'getLogger'})
"""The factory calls that hand back a logger without naming a constructor.

``PdsLogger.get_logger('x')`` returns a configured logger exactly as the
constructor does, so a check that matched only the constructor would watch one
of two doors.
"""


def _constructs_a_logger(path: FCPath) -> bool:
    """Whether a module makes a logger that can write a record.

    Matched in the AST rather than the text: a docstring mentioning the call
    is not one, and an aliased import (``import pdslogger as _pl``) is.  Both
    ways of making one are matched -- calling a class, and asking a class or the
    module for one by name -- since the two produce the same object.

    Parameters:
        path: The module to read.

    Returns:
        True when the module makes a logger.
    """
    tree = ast.parse(path.read_text())
    # ``import pdslogger as p`` then ``p.PdsLogger(...)``.
    module_aliases = {'pdslogger'} | {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
        if alias.name == 'pdslogger'
    }
    # ``from pdslogger import PdsLogger as P`` then ``P(...)``: the name the
    # class is called by is whatever the import bound it to.
    class_names = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == 'pdslogger'
        for alias in node.names
        if alias.name in _WRITING_LOGGER_CLASSES
    }
    reachable = module_aliases | class_names
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id in class_names:
            return True
        if not isinstance(func, ast.Attribute) or not isinstance(func.value, ast.Name):
            continue
        if func.attr in _WRITING_LOGGER_CLASSES and func.value.id in module_aliases:
            return True
        if func.attr in _LOGGER_FACTORIES and func.value.id in reachable:
            return True
    return False


# ---------------------------------------------------------------------------
# The checks themselves
# ---------------------------------------------------------------------------
#
# Everything above asserts that a search came back empty, and a search that
# cannot find anything comes back empty too.  These hand the two predicates
# source they must object to, so a narrowing of either is caught here rather
# than by the absence it would silently create.


def _module(tmp_path: Path, source: str) -> FCPath:
    """Write one module for a predicate to read.

    Parameters:
        tmp_path: Directory to write it in.
        source: The module's source.

    Returns:
        The file.
    """
    path = tmp_path / 'probe.py'
    path.write_text(source)
    return FCPath(path)


@pytest.mark.parametrize(
    'call',
    [
        pytest.param('logger.info("x")', id='info'),
        pytest.param('logger.warn("x")', id='warn'),
        pytest.param('logger.warning("x")', id='warning'),
        pytest.param('logger.error("x")', id='error'),
        pytest.param('logger.blankline()', id='blankline'),
        pytest.param('logger.normal("x")', id='normal'),
        pytest.param('logger.hidden("x")', id='hidden'),
        pytest.param('logger.invisible("x")', id='invisible'),
        pytest.param('logger.summarize()', id='summarize'),
        pytest.param('logger.open("SECTION")', id='open'),
        pytest.param('logger.close()', id='close'),
        pytest.param('self._logger.open("SECTION")', id='a-held-logger'),
    ],
)
def test_every_way_of_writing_a_line_is_seen(call: str, tmp_path: Path) -> None:
    """Enumerating some of them lets the others in, which is the hole this closes.

    ``with logger.open(...):`` is the house idiom for a section and puts as much
    in a log as an info call does, so a check watching only the four levels
    somebody happened to use would be blind to the most likely addition of all.

    Parameters:
        call: The call to plant.
        tmp_path: Directory the probe module is written in.
    """
    probe = _module(tmp_path, f'def f(logger, self):\n    {call}\n')
    assert _logging_call_sites(probe) == [('f', call.split('(')[0].split('.')[-1])]


@pytest.mark.parametrize(
    'call',
    [
        pytest.param('source.close()', id='a-source'),
        pytest.param('cursor.close()', id='a-cursor'),
        pytest.param('handle.open()', id='a-file'),
    ],
)
def test_closing_something_that_is_not_a_logger_is_not_a_line(call: str, tmp_path: Path) -> None:
    """A source and a cursor are closed by the same word a section is.

    Parameters:
        call: The call to plant.
        tmp_path: Directory the probe module is written in.
    """
    assert _logging_call_sites(_module(tmp_path, f'def f():\n    {call}\n')) == []


def test_a_second_line_in_an_exempted_function_is_a_second_site(tmp_path: Path) -> None:
    """An exemption names a call, not a file, so a line added beside it is caught."""
    probe = _module(tmp_path, 'def f(logger):\n    logger.info("a")\n    logger.info("b")\n')
    assert _logging_call_sites(probe) == [('f', 'info'), ('f', 'info')]


def test_a_line_is_attributed_to_the_function_it_is_written_in(tmp_path: Path) -> None:
    """So an exemption for one function does not cover the module around it."""
    source = 'def f(logger):\n    logger.info("a")\n\n\ndef g(logger):\n    logger.info("b")\n'
    probe = _module(tmp_path, source)
    assert _logging_call_sites(probe) == [('f', 'info'), ('g', 'info')]


@pytest.mark.parametrize(
    'source',
    [
        pytest.param('from pdslogger import PdsLogger\nx = PdsLogger("a")\n', id='the-class'),
        pytest.param('import pdslogger\nx = pdslogger.PdsLogger("a")\n', id='through-the-module'),
        pytest.param('import pdslogger as p\nx = p.PdsLogger("a")\n', id='an-aliased-module'),
        pytest.param('from pdslogger import PdsLogger as P\nx = P("a")\n', id='an-aliased-class'),
        pytest.param(
            'from pdslogger import PdsLogger\nx = PdsLogger.get_logger("a")\n', id='the-factory'
        ),
        pytest.param('import pdslogger\nx = pdslogger.get_logger("a")\n', id='the-modules-factory'),
        pytest.param('from pdslogger import EasyLogger\nx = EasyLogger("a")\n', id='another-class'),
    ],
)
def test_every_way_of_making_a_logger_is_seen(source: str, tmp_path: Path) -> None:
    """A logger asked for by name is the same object as one that was constructed.

    Parameters:
        source: The module source to plant.
        tmp_path: Directory the probe module is written in.
    """
    assert _constructs_a_logger(_module(tmp_path, source)) is True


def test_a_null_logger_is_not_a_logger_of_its_own(tmp_path: Path) -> None:
    """It holds no handler and writes nowhere, so making one acquires no voice.

    It is how a layer that must have none spells the absence of one, and
    counting it would make the check about the word rather than about the
    record.
    """
    source = 'from pdslogger import NullLogger\nx = NullLogger()\n'
    assert _constructs_a_logger(_module(tmp_path, source)) is False
