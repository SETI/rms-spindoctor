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


def _is_broad(handler: ast.ExceptHandler) -> bool:
    """Whether an except clause catches everything.

    Parameters:
        handler: The ``except`` clause to judge.

    Returns:
        True for a bare ``except:`` and for one naming ``Exception`` or
        ``BaseException``, alone or among others.
    """
    if handler.type is None:
        return True
    named = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
    return any(isinstance(n, ast.Name) and n.id in ('Exception', 'BaseException') for n in named)


def _escapes(handler: ast.ExceptHandler) -> bool:
    """Whether an exception caught here still reaches the caller.

    A handler that re-raises, or raises something else, has reported rather
    than absorbed.  A nested function or class defined inside it is not part of
    this handler's control flow, so a ``raise`` in one does not count.

    Parameters:
        handler: The ``except`` clause to judge.

    Returns:
        True when the handler cannot fall through to the code after it.
    """
    nested = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
    pending: list[ast.AST] = list(handler.body)
    while pending:
        node = pending.pop()
        if isinstance(node, ast.Raise):
            return True
        # Descend into everything except a definition, whose body runs when it
        # is called rather than when this handler does.
        if not isinstance(node, nested):
            pending.extend(ast.iter_child_nodes(node))
    return False


def _swallowing_handlers(path: Path) -> list[int]:
    """Return the line of every broad except clause in a file that absorbs.

    Parameters:
        path: The module to read.

    Returns:
        The line numbers, in source order.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.ExceptHandler) and _is_broad(node) and not _escapes(node)
    ]


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
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path)))
        if isinstance(node, ast.ExceptHandler) and _is_broad(node) and _escapes(node)
    ]
    assert reraising != []


def test_a_handler_that_only_logs_is_judged_to_swallow() -> None:
    """The checker itself, on the shape it exists to forbid."""
    source = 'try:\n    f()\nexcept Exception:\n    log("failed")\n    return None\n'
    handler = next(
        node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.ExceptHandler)
    )
    assert _is_broad(handler)
    assert not _escapes(handler)


def test_a_handler_that_logs_and_reraises_is_judged_to_escape() -> None:
    """The complement, so the checker is not passing everything."""
    source = 'try:\n    f()\nexcept Exception:\n    log("failed")\n    raise\n'
    handler = next(
        node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.ExceptHandler)
    )
    assert _escapes(handler)


def test_a_raise_inside_a_nested_function_does_not_count_as_escaping() -> None:
    """A closure defined in the handler is not this handler's control flow."""
    source = (
        'try:\n    f()\nexcept Exception:\n    def g():\n        raise ValueError\n    return g\n'
    )
    handler = next(
        node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.ExceptHandler)
    )
    assert not _escapes(handler)


def test_a_bare_except_is_broad() -> None:
    """``except:`` catches more than ``except Exception``, not less."""
    source = 'try:\n    f()\nexcept:\n    return None\n'
    handler = next(
        node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.ExceptHandler)
    )
    assert _is_broad(handler)


def test_exception_named_among_others_is_broad() -> None:
    """``except (KeyError, Exception)`` catches everything the second name does."""
    source = 'try:\n    f()\nexcept (KeyError, Exception):\n    return None\n'
    handler = next(
        node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.ExceptHandler)
    )
    assert _is_broad(handler)


def test_a_narrow_except_is_not_broad() -> None:
    """A named type is a decision about which failures are expected."""
    source = 'try:\n    f()\nexcept KeyError:\n    return None\n'
    handler = next(
        node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.ExceptHandler)
    )
    assert not _is_broad(handler)
