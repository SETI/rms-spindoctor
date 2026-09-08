"""The navigation path absorbs no exception, stated over the source.

An exception that is swallowed does nothing observable until the day something
raises, and what it produces then is a result rather than a failure -- so no
test that exercises the code can find one.  This reads the source instead, the
way ``test_logging_static_invariants`` does, and for the same reason: it covers
whole packages, so a file added to one of them tomorrow is covered without
being named here.

What it forbids is narrow.  ``except Exception`` is fine where the handler
re-raises: the navigation packages use it to say which stage could not compute
what before letting the exception reach the orchestrator, which fails the image
with ``status_reason=internal_error``.  What is forbidden is a broad clause
that control can leave *without raising* -- one that returns, continues,
passes, or only logs -- because whatever runs after it produces an offset
which looks exactly like a whole one.  No error filter selects such a
document, ``--has-no-offset-file`` passes over it because it exists, and no
later pass corrects it.

The exception types cannot be narrowed to tell the two cases apart: ``oops``
declines with ``ValueError`` and ``LookupError``, and so does a defect in
``oops``, in ``numpy``, or here.  So the rule is about what the handler does,
not what it catches.
"""

import ast
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[2] / 'src' / 'spindoctor'

_NAVIGATION_PACKAGES = [
    'feature',
    'nav_model',
    'nav_orchestrator',
    'nav_technique',
    'obs',
    'support',
]
"""The packages under ``src/spindoctor`` that compute one image's offset.

The CLI, UI and reporting packages are deliberately outside this.  A program's
top level is where a catch-all belongs, and a viewer that declines to draw one
panel is not a navigation that concluded something on half its evidence.
"""

_TOP_LEVEL_MODULES = '.'
"""The modules directly under ``src/spindoctor``, ``navigate_image_files`` among them.

A group of its own because a package name stands for everything beneath it,
and everything beneath ``src/spindoctor`` includes the packages kept out above.
"""

_NAVIGATION_SOURCES = [*_NAVIGATION_PACKAGES, _TOP_LEVEL_MODULES]
"""Every group of modules the rule covers."""


_BROAD_BUILTINS = frozenset({'Exception', 'BaseException'})
"""The two names that catch everything a stage could fail with."""


def _sources_in(group: str) -> list[Path]:
    """Every module in one covered group, in path order.

    Parameters:
        group: A package name under ``src/spindoctor``, or ``_TOP_LEVEL_MODULES``
            for the modules directly under it.

    Returns:
        The paths of the ``.py`` files: all of a package's, however deep, and
        only the top level's own for ``_TOP_LEVEL_MODULES``.
    """
    if group == _TOP_LEVEL_MODULES:
        return sorted(_SRC.glob('*.py'))
    return sorted((_SRC / group).rglob('*.py'))


def _broad_names_in(tree: ast.Module) -> frozenset[str]:
    """Every local name in a module that refers to a catch-everything type.

    A module can spell ``Exception`` under another name -- ``from builtins
    import Exception as Anything``, or ``Anything = Exception`` -- and a checker
    that only knows the two builtins would read the handler as narrow and pass
    a file that catches everything.

    Parameters:
        tree: The parsed module.

    Returns:
        The builtin names plus every local alias bound to one of them, whether
        by import or by assignment, an alias of an alias included.
    """
    names = set(_BROAD_BUILTINS)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name in _BROAD_BUILTINS:
                    names.add(alias.asname or alias.name)
    # An alias of an alias is an alias, and the assignments are walked in no
    # particular order, so they are read again until a pass adds nothing.
    assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign | ast.AnnAssign)]
    while True:
        before = len(names)
        for node in assignments:
            names.update(_aliases_bound_by(node, names))
        if len(names) == before:
            return frozenset(names)


def _aliases_bound_by(node: ast.Assign | ast.AnnAssign, names: set[str]) -> set[str]:
    """The names an assignment binds to one of ``names``.

    Parameters:
        node: A plain or annotated assignment.
        names: The names already known to be broad.

    Returns:
        Every target name whose value is one of ``names``: the target of
        ``A = B`` and of ``A: type = B``, and each target of ``A, C = B, D``
        paired with its own value.
    """
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    pairs: list[tuple[ast.expr, ast.expr | None]] = []
    for target in targets:
        if isinstance(target, ast.Tuple) and isinstance(node.value, ast.Tuple):
            pairs.extend(zip(target.elts, node.value.elts, strict=False))
        else:
            pairs.append((target, node.value))
    return {
        target.id
        for target, value in pairs
        if isinstance(target, ast.Name) and isinstance(value, ast.Name) and value.id in names
    }


def _is_broad(handler: ast.ExceptHandler, broad_names: frozenset[str]) -> bool:
    """Whether an except clause catches everything.

    Parameters:
        handler: The ``except`` clause to judge.
        broad_names: The names this module has for a catch-everything type.

    Returns:
        True for a bare ``except:``, and for one naming a catch-everything
        type -- plainly, under an alias, or qualified as ``builtins.Exception``
        -- alone or among others.
    """
    if handler.type is None:
        return True
    named = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
    for node in named:
        if isinstance(node, ast.Name) and node.id in broad_names:
            return True
        if isinstance(node, ast.Attribute) and node.attr in _BROAD_BUILTINS:
            return True
    return False


def _escapes(handler: ast.ExceptHandler) -> bool:
    """Whether an exception caught here reaches the caller on every path.

    Asking only whether a ``raise`` appears anywhere inside would accept a
    handler that raises on one branch and returns a value on the other, which
    is the shape this file exists to forbid wearing a disguise.  So a handler
    escapes only when control cannot leave it without raising.

    Parameters:
        handler: The ``except`` clause to judge.

    Returns:
        True when the handler cannot be left except by raising.
    """
    return _always_escapes(handler.body)


def _always_escapes(body: list[ast.stmt]) -> bool:
    """Whether a statement list always leaves by raising.

    Conservative in the direction that matters: whatever this does not
    recognize counts as falling through, so an unrecognized shape is reported
    rather than passed over.  It recognizes as raising a ``raise``, an ``if``
    both of whose branches raise, a ``match`` with a wildcard case all of whose
    cases raise, and a ``try`` every part of which raises.  Everything else
    falls through or leaves without raising: a ``with`` even when its body
    raises, since the context manager may suppress the raise; a loop, whose
    body may not run at all; and any statement that can ``return``, ``break``
    or ``continue`` at this level, a ``finally`` among them, since each of
    those leaves without raising and discards an exception in flight.  A
    nested function or class is not this list's control flow.

    Parameters:
        body: The statements to judge, in order.

    Returns:
        True when the list cannot be left except by raising.
    """
    for node in body:
        if _diverts_control([node]):
            return False
        if isinstance(node, ast.Raise):
            return True
        if isinstance(node, ast.If):
            # Without an ``else`` the false branch falls straight through.
            if node.orelse and _always_escapes(node.body) and _always_escapes(node.orelse):
                return True
        elif isinstance(node, ast.Try):
            if _always_escapes(node.finalbody):
                return True
            # A raise in the body is only an escape if no handler catches it,
            # which cannot be known -- so every handler has to escape too.
            if (
                _always_escapes(node.body)
                and (not node.orelse or _always_escapes(node.orelse))
                and all(_always_escapes(inner.body) for inner in node.handlers)
            ):
                return True
        elif isinstance(node, ast.Match):
            cases = node.cases
            wildcard = any(
                isinstance(case.pattern, ast.MatchAs)
                and case.pattern.pattern is None
                and case.guard is None
                for case in cases
            )
            if wildcard and all(_always_escapes(case.body) for case in cases):
                return True
    return False


def _diverts_control(body: list[ast.stmt], *, in_loop: bool = False) -> bool:
    """Whether a statement list can be left by ``return``, ``break`` or ``continue``.

    Each of the three leaves without raising, and in a ``finally`` each discards
    the exception in flight.  A nested function or class is its own control flow
    and is not entered.  A ``break`` or ``continue`` inside a loop that is itself
    inside the list only leaves that loop, so neither counts there; a ``return``
    counts wherever it is.

    Parameters:
        body: The statements to judge.
        in_loop: Whether the statements are inside a loop that is itself inside
            the list judged at the top of the recursion.

    Returns:
        True when one of the three can run, whether or not it always does.
    """
    for node in body:
        if isinstance(node, ast.Return):
            return True
        if isinstance(node, ast.Break | ast.Continue) and not in_loop:
            return True
        if isinstance(node, ast.For | ast.AsyncFor | ast.While):
            # The ``else`` of a loop runs outside it, so a ``break`` there is
            # not that loop's.
            inside = _diverts_control(node.body, in_loop=True)
            if inside or _diverts_control(node.orelse, in_loop=in_loop):
                return True
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            continue
        elif any(_diverts_control(inner, in_loop=in_loop) for inner in _statement_lists(node)):
            return True
    return False


def _statement_lists(node: ast.stmt) -> list[list[ast.stmt]]:
    """The statement lists a compound statement holds directly.

    Parameters:
        node: Any statement; a simple one holds none.

    Returns:
        Every body, ``else``, ``finally``, ``except`` clause and ``case`` of the
        statement, each as a list, in source order.
    """
    if isinstance(node, ast.If | ast.For | ast.AsyncFor | ast.While):
        return [node.body, node.orelse]
    if isinstance(node, ast.With | ast.AsyncWith):
        return [node.body]
    if isinstance(node, ast.Try | ast.TryStar):
        handlers = [handler.body for handler in node.handlers]
        return [node.body, *handlers, node.orelse, node.finalbody]
    if isinstance(node, ast.Match):
        return [case.body for case in node.cases]
    return []


def _swallowing_handlers(path: Path) -> list[int]:
    """Return the line of every broad except clause in a file that absorbs.

    A handler is judged together with the ``finally`` of the statement it
    belongs to: a ``finally`` that returns, breaks or continues discards the
    handler's re-raise, so the handler absorbs whatever its own body does.

    Parameters:
        path: The module to read.

    Returns:
        The line numbers, in source order.
    """
    tree = _parse(path)
    broad_names = _broad_names_in(tree)
    lines: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try | ast.TryStar):
            continue
        discarded = _diverts_control(node.finalbody)
        lines.extend(
            handler.lineno
            for handler in node.handlers
            if _is_broad(handler, broad_names) and (discarded or not _escapes(handler))
        )
    return sorted(lines)


def _parse(path: Path) -> ast.Module:
    """Parse a source file as UTF-8, whatever the process encoding is.

    Source under these packages is not all ASCII, and reading it in whatever
    the environment happens to prefer would fail the run with a decoding error
    before a single handler had been judged.

    Parameters:
        path: The module to read.

    Returns:
        The parsed module.
    """
    return ast.parse(path.read_text(encoding='utf-8'), filename=str(path))


@pytest.mark.parametrize('group', _NAVIGATION_SOURCES)
def test_no_navigation_source_swallows_a_broad_exception(group: str) -> None:
    """A broad clause control can leave without raising turns a fault into an answer.

    Parameters:
        group: A package under ``src/spindoctor``, or its top-level modules.
    """
    found = [
        f'{path.relative_to(_SRC)}:{line}'
        for path in _sources_in(group)
        for line in _swallowing_handlers(path)
    ]
    assert found == []


def test_the_navigation_sources_do_still_catch_broadly() -> None:
    """Or the test above would hold by there being nothing to judge.

    The rule is about what a handler does, not that broad clauses are gone:
    each stage catches everything precisely so it can say which stage could not
    compute what, and then re-raises.
    """
    reraising = [
        path
        for group in _NAVIGATION_SOURCES
        for path in _sources_in(group)
        for node in ast.walk(_parse(path))
        if isinstance(node, ast.ExceptHandler)
        and _is_broad(node, _BROAD_BUILTINS)
        and _escapes(node)
    ]
    assert reraising != []


def _handler(source: str) -> ast.ExceptHandler:
    """Return the first except clause in a snippet.

    Parameters:
        source: Python source containing at least one ``except`` clause.

    Returns:
        The first ``ExceptHandler`` node in it.
    """
    return next(node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.ExceptHandler))


def _broad(source: str) -> bool:
    """Whether the first except clause in a snippet catches everything.

    Parameters:
        source: Python source containing at least one ``except`` clause.

    Returns:
        What :func:`_is_broad` says about it, with the snippet's own aliases.
    """
    return _is_broad(_handler(source), _broad_names_in(ast.parse(source)))


def _swallowed(source: str, directory: Path) -> list[int]:
    """Return the lines :func:`_swallowing_handlers` reports in a snippet.

    Parameters:
        source: Python source to judge as a module.
        directory: Directory the snippet is written under.

    Returns:
        The reported line numbers.
    """
    path = directory / 'snippet.py'
    path.write_text(source, encoding='utf-8')
    return _swallowing_handlers(path)


def test_a_handler_that_only_logs_is_judged_to_swallow() -> None:
    """The checker itself, on the shape it exists to forbid."""
    source = 'try:\n    f()\nexcept Exception:\n    log("failed")\n    return None\n'
    assert _broad(source)
    assert not _escapes(_handler(source))


def test_a_handler_that_logs_and_reraises_is_judged_to_escape() -> None:
    """The complement, so the checker is not passing everything."""
    source = 'try:\n    f()\nexcept Exception:\n    log("failed")\n    raise\n'
    assert _escapes(_handler(source))


def test_a_handler_that_continues_the_loop_is_judged_to_swallow() -> None:
    """Moving on to the next item is leaving without raising."""
    source = (
        'for item in items:\n    try:\n        f()\n'
        '    except Exception:\n        log("failed")\n        continue\n'
    )
    assert not _escapes(_handler(source))


def test_a_raise_inside_a_nested_function_does_not_count_as_escaping() -> None:
    """A closure defined in the handler is not this handler's control flow."""
    source = (
        'try:\n    f()\nexcept Exception:\n    def g():\n        raise ValueError\n    return g\n'
    )
    assert not _escapes(_handler(source))


def test_a_conditional_raise_does_not_count_as_escaping() -> None:
    """Raising on one branch and answering on the other is still answering.

    This is the shape a checker that looks for any ``raise`` anywhere inside
    the handler passes, and it is exactly the forbidden one: the frames that
    take the other branch get an offset out of a fault.
    """
    source = 'try:\n    f()\nexcept Exception:\n    if fatal:\n        raise\n    return None\n'
    assert not _escapes(_handler(source))


def test_a_conditional_return_before_a_raise_does_not_count_as_escaping() -> None:
    """The same shape with the branches the other way round.

    A ``raise`` at the end of the handler is not reached by the frames that
    took the returning branch.
    """
    source = (
        'try:\n    f()\nexcept Exception:\n    if recoverable:\n        return None\n    raise\n'
    )
    assert not _escapes(_handler(source))


def test_both_branches_raising_counts_as_escaping() -> None:
    """A branch is only a hole when one side of it falls through."""
    source = (
        'try:\n    f()\nexcept Exception:\n'
        '    if fatal:\n        raise\n    else:\n        raise RuntimeError\n'
    )
    assert _escapes(_handler(source))


def test_a_raise_caught_by_a_nested_handler_does_not_count_as_escaping() -> None:
    """An inner ``try`` that catches its own raise leaves nothing to the caller."""
    source = (
        'try:\n    f()\nexcept Exception:\n'
        '    try:\n        raise\n    except Exception:\n        pass\n'
        '    return None\n'
    )
    assert not _escapes(_handler(source))


def test_a_raise_in_a_finally_counts_as_escaping() -> None:
    """A ``finally`` runs on every path out of the statement it guards."""
    source = (
        'try:\n    f()\nexcept Exception:\n    try:\n        g()\n    finally:\n        raise\n'
    )
    assert _escapes(_handler(source))


@pytest.mark.parametrize('leave', ['return None', 'break', 'continue'])
def test_a_finally_that_leaves_does_not_count_as_escaping(leave: str) -> None:
    """A ``finally`` that returns, breaks or continues discards the raise in flight.

    Parameters:
        leave: The statement the ``finally`` leaves by.
    """
    source = (
        'for item in items:\n    try:\n        f()\n    except Exception:\n'
        f'        try:\n            raise\n        finally:\n            {leave}\n'
    )
    assert not _escapes(_handler(source))


def test_a_finally_that_leaves_around_a_re_raising_handler_is_reported(tmp_path: Path) -> None:
    """The handler re-raises, and the ``finally`` around it discards the raise."""
    source = (
        'for item in items:\n    try:\n        f()\n    except Exception:\n        raise\n'
        '    finally:\n        continue\n'
    )
    assert _swallowed(source, tmp_path) == [4]


def test_a_finally_that_only_cleans_up_leaves_a_re_raising_handler_alone(tmp_path: Path) -> None:
    """A ``finally`` that neither returns nor leaves a loop lets the raise through."""
    source = 'try:\n    f()\nexcept Exception:\n    raise\nfinally:\n    g()\n'
    assert _swallowed(source, tmp_path) == []


def test_a_break_in_a_loop_inside_a_finally_still_counts_as_escaping() -> None:
    """A ``break`` leaves the loop it is in, not the ``finally`` around that loop."""
    source = (
        'try:\n    f()\nexcept Exception:\n    try:\n        raise\n    finally:\n'
        '        for item in items:\n            break\n'
    )
    assert _escapes(_handler(source))


def test_a_return_in_a_function_inside_a_finally_still_counts_as_escaping() -> None:
    """A ``return`` in a nested function leaves that function, not the ``finally``."""
    source = (
        'try:\n    f()\nexcept Exception:\n    try:\n        raise\n    finally:\n'
        '        def g():\n            return None\n'
    )
    assert _escapes(_handler(source))


def test_a_raise_inside_a_with_does_not_count_as_escaping() -> None:
    """A context manager may suppress what its body raises."""
    source = 'try:\n    f()\nexcept Exception:\n    with suppress(Exception):\n        raise\n'
    assert not _escapes(_handler(source))


def test_a_raise_inside_a_loop_does_not_count_as_escaping() -> None:
    """A loop body may run no times at all."""
    source = (
        'try:\n    f()\nexcept Exception:\n    for item in items:\n        raise\n    return None\n'
    )
    assert not _escapes(_handler(source))


def test_a_bare_except_is_broad() -> None:
    """``except:`` catches more than ``except Exception``, not less."""
    assert _broad('try:\n    f()\nexcept:\n    return None\n')


def test_exception_named_among_others_is_broad() -> None:
    """``except (KeyError, Exception)`` catches everything the second name does."""
    assert _broad('try:\n    f()\nexcept (KeyError, Exception):\n    return None\n')


def test_a_qualified_exception_name_is_broad() -> None:
    """``except builtins.Exception`` catches what ``except Exception`` catches."""
    assert _broad('import builtins\ntry:\n    f()\nexcept builtins.Exception:\n    return None\n')


def test_an_aliased_exception_import_is_broad() -> None:
    """A module may spell the catch-everything type under any name it likes."""
    assert _broad(
        'from builtins import Exception as Anything\n'
        'try:\n    f()\nexcept Anything:\n    return None\n'
    )


def test_an_assigned_exception_alias_is_broad() -> None:
    """An assignment aliases the catch-everything type as well as an import does."""
    assert _broad('Broad = Exception\ntry:\n    f()\nexcept Broad:\n    return None\n')


def test_an_annotated_exception_alias_is_broad() -> None:
    """An annotation on the alias changes nothing about what it names."""
    assert _broad('Broad: type = Exception\ntry:\n    f()\nexcept Broad:\n    return None\n')


def test_an_alias_of_an_alias_is_broad() -> None:
    """A name bound to an alias is bound to what the alias names."""
    assert _broad('A = Exception\nB = A\ntry:\n    f()\nexcept B:\n    return None\n')


def test_a_tuple_assigned_exception_alias_is_broad() -> None:
    """Each target of a tuple assignment takes the value at its own position."""
    assert _broad(
        'Broad, Other = Exception, KeyError\ntry:\n    f()\nexcept Broad:\n    return None\n'
    )


def test_the_other_target_of_a_tuple_assignment_is_not_broad() -> None:
    """A name paired with a narrow type is narrow, whatever its neighbor is."""
    assert not _broad(
        'Broad, Other = Exception, KeyError\ntry:\n    f()\nexcept Other:\n    return None\n'
    )


def test_a_narrow_except_is_not_broad() -> None:
    """A named type is a decision about which failures are expected."""
    assert not _broad('try:\n    f()\nexcept KeyError:\n    return None\n')


def test_a_narrow_qualified_name_is_not_broad() -> None:
    """Qualifying a narrow type does not make it broad."""
    assert not _broad('try:\n    f()\nexcept oops.OopsError:\n    return None\n')
