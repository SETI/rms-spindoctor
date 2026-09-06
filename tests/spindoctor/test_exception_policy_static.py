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
with ``status_reason=internal_error``.  What is forbidden is a broad clause that
*returns a value*, because that produces an offset which looks exactly like a
whole one.  No error filter selects such a document, ``--has-no-offset-file``
passes over it because it exists, and no later pass corrects it.

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
"""The packages one navigation of one image runs through.

The CLI, UI and reporting packages are deliberately outside this.  A program's
top level is where a catch-all belongs, and a viewer that declines to draw one
panel is not a navigation that concluded something on half its evidence.
"""


_BROAD_BUILTINS = frozenset({'Exception', 'BaseException'})
"""The two names that catch everything a stage could fail with."""


def _broad_names_in(tree: ast.Module) -> frozenset[str]:
    """Every local name in a module that refers to a catch-everything type.

    A module can spell ``Exception`` under another name -- ``from builtins
    import Exception as Anything`` -- and a checker that only knows the two
    builtins would read the handler as narrow and pass a file that catches
    everything.

    Parameters:
        tree: The parsed module.

    Returns:
        The builtin names plus every local alias bound to one of them.
    """
    names = set(_BROAD_BUILTINS)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name in _BROAD_BUILTINS:
                    names.add(alias.asname or alias.name)
    return frozenset(names)


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
    escapes only when control cannot reach the end of it.

    Parameters:
        handler: The ``except`` clause to judge.

    Returns:
        True when the handler cannot fall through to the code after it.
    """
    return _always_escapes(handler.body)


def _always_escapes(body: list[ast.stmt]) -> bool:
    """Whether a statement list always leaves by raising.

    Conservative in the direction that matters: a construct this does not
    understand counts as falling through, so an unrecognised shape is reported
    rather than passed over.  A nested function or class is not this list's
    control flow, and a loop body may not run at all, so neither counts.

    Parameters:
        body: The statements to judge, in order.

    Returns:
        True when reaching the end of the list is impossible.
    """
    for node in body:
        if isinstance(node, ast.Raise):
            return True
        if isinstance(node, ast.If):
            # Without an ``else`` the false branch falls straight through.
            if node.orelse and _always_escapes(node.body) and _always_escapes(node.orelse):
                return True
        elif isinstance(node, ast.With):
            if _always_escapes(node.body):
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


def _swallowing_handlers(path: Path) -> list[int]:
    """Return the line of every broad except clause in a file that absorbs.

    Parameters:
        path: The module to read.

    Returns:
        The line numbers, in source order.
    """
    tree = _parse(path)
    broad_names = _broad_names_in(tree)
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.ExceptHandler)
        and _is_broad(node, broad_names)
        and not _escapes(node)
    ]


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


@pytest.mark.parametrize('package', _NAVIGATION_PACKAGES)
def test_no_navigation_package_swallows_a_broad_exception(package: str) -> None:
    """A broad clause that returns a value turns a fault into an answer.

    Parameters:
        package: The package under ``src/spindoctor`` to read.
    """
    found = [
        f'{path.relative_to(_SRC)}:{line}'
        for path in sorted((_SRC / package).rglob('*.py'))
        for line in _swallowing_handlers(path)
    ]
    assert found == []


def test_the_navigation_packages_do_still_catch_broadly() -> None:
    """Or the test above would hold by there being nothing to judge.

    The rule is about what a handler does, not that broad clauses are gone:
    each stage catches everything precisely so it can say which stage could not
    compute what, and then re-raises.
    """
    reraising = [
        path
        for package in _NAVIGATION_PACKAGES
        for path in sorted((_SRC / package).rglob('*.py'))
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


def test_a_handler_that_only_logs_is_judged_to_swallow() -> None:
    """The checker itself, on the shape it exists to forbid."""
    source = 'try:\n    f()\nexcept Exception:\n    log("failed")\n    return None\n'
    assert _broad(source)
    assert not _escapes(_handler(source))


def test_a_handler_that_logs_and_reraises_is_judged_to_escape() -> None:
    """The complement, so the checker is not passing everything."""
    source = 'try:\n    f()\nexcept Exception:\n    log("failed")\n    raise\n'
    assert _escapes(_handler(source))


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


def test_a_narrow_except_is_not_broad() -> None:
    """A named type is a decision about which failures are expected."""
    assert not _broad('try:\n    f()\nexcept KeyError:\n    return None\n')


def test_a_narrow_qualified_name_is_not_broad() -> None:
    """Qualifying a narrow type does not make it broad."""
    assert not _broad('try:\n    f()\nexcept oops.OopsError:\n    return None\n')
