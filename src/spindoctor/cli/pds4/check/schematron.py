"""The Schematron rules a label declares, evaluated as the ISO Schematron skeleton does.

A label names each Schematron it is held to by an ``xml-model`` processing instruction
before its root element, by URL, which is mapped to the shipped copy of the same file name
(see :mod:`~spindoctor.cli.pds4.check.schemas`).  The rules are evaluated with
elementpath's XPath 2.0 engine and matched the way the skeleton's XSLT matches them:

- a node matches a rule when it is in ``//(context)`` evaluated from the document node, so
  a context of several steps, as in ``pds:Inventory/pds:offset``, matches every node it
  selects anywhere in the label.  A context that is a union of path expressions joined
  by ``|``, as every context of the shipped Schematron is, is selected as ``//``
  followed by each branch, a branch that is an absolute path kept as it is, which
  selects the same nodes without evaluating the context afresh at every node of the
  label; any other context is selected as ``//(context)`` itself (see
  :func:`match_expression`);
- within a pattern, only the first rule a node matches fires for that node;
- the schema's variables and each pattern's are evaluated at the document node, and a
  rule's at the node it matched, each in the order they are declared.

An assert fails when its test is false, and a report fires when its test is true.  Each
is a finding, with the message its rule writes, its ``value-of`` and ``name`` parts
evaluated at the node.  It is a warning when the assert or report carries a ``role`` of
``warning`` or ``warn``, in any case, or carries none and its rule does -- the spellings
the shipped Schematron use, which the PDS ``validate`` tool reports as warnings -- and an
error otherwise.  Strings are compared by the Unicode code point collation, XPath's
default, whatever locale the process runs under.
"""

import functools
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from elementpath import AttributeNode, ElementNode, XPath2Parser, XPathContext, XPathToken
from elementpath.collations import UNICODE_CODEPOINT_COLLATION
from elementpath.tree_builders import get_node_tree
from lxml import etree

from spindoctor.cli.pds4.check.elements import element_path, local_name
from spindoctor.cli.pds4.check.findings import CheckName, Finding, Severity
from spindoctor.cli.pds4.check.schemas import shipped_copy

SCHEMATRON_NAMESPACE = 'http://purl.oclc.org/dsdl/schematron'
"""ISO Schematron's namespace, which an ``xml-model`` instruction names as its type."""

WARNING_ROLES = frozenset({'warning', 'warn'})
"""The ``role`` values, in lower case, that make an assert or a report a warning."""

_NOT_A_PATH_UNION = re.compile(r'\(:|\b(?:union|intersect|except)\b')
"""What a context split at ``|`` cannot hold: a comment, or a set operator's keyword."""

_S = f'{{{SCHEMATRON_NAMESPACE}}}'
"""The Schematron namespace as the prefix of a tag."""

_Bindings = tuple[tuple[str, XPathToken], ...]
"""Variables as a Schematron declares them: each name with its parsed expression."""


@dataclass(frozen=True)
class _Check:
    """One assert or report of a rule.

    Attributes:
        is_assert: True for an assert, which fails when its test is false; False for a
            report, which fires when its test is true.
        test: The test, parsed as its effective boolean value.
        title: The check's title, or the empty string when it has none.
        message: The message: literal text, and parsed ``value-of`` and ``name`` parts.
        severity: Whether a failed assert or a fired report is an error or a warning.
    """

    is_assert: bool
    test: XPathToken
    title: str
    message: tuple[str | XPathToken, ...]
    severity: Severity


@dataclass(frozen=True)
class _Rule:
    """One rule of a pattern.

    Attributes:
        context: The rule's context, as the Schematron writes it.
        match: The expression selecting the nodes the context matches, as
            :func:`match_expression` writes it, parsed.
        lets: The rule's variables.
        checks: Its asserts and reports, in order.
    """

    context: str
    match: XPathToken
    lets: _Bindings
    checks: tuple[_Check, ...]


@dataclass(frozen=True)
class _Pattern:
    """One pattern of a Schematron.

    Attributes:
        lets: The pattern's variables.
        rules: Its rules, in order, the first a node matches being the one that fires.
    """

    lets: _Bindings
    rules: tuple[_Rule, ...]


@dataclass(frozen=True)
class _Schematron:
    """One Schematron with every expression in it parsed.

    Attributes:
        name: Its file name.
        namespaces: The prefixes it declares, each with its namespace.
        lets: Its schema-level variables.
        patterns: Its patterns, in order.
    """

    name: str
    namespaces: tuple[tuple[str, str], ...]
    lets: _Bindings
    patterns: tuple[_Pattern, ...]


def declared_schematron(document: Any) -> list[str]:
    """Return the URL of each Schematron a label declares.

    Parameters:
        document: The label, parsed by lxml.

    Returns:
        The ``href`` of each ``xml-model`` instruction before the root element whose
        ``schematypens`` is ISO Schematron's namespace, in the order the label gives them.
    """
    hrefs: list[str] = []
    node = document.getroot().getprevious()
    while node is not None:
        if (
            isinstance(node, etree._ProcessingInstruction)
            and node.target == 'xml-model'
            and node.get('schematypens') == SCHEMATRON_NAMESPACE
            and node.get('href') is not None
        ):
            hrefs.append(str(node.get('href')))
        node = node.getprevious()
    return list(reversed(hrefs))


def _bindings(parser: XPath2Parser, element: Any) -> _Bindings:
    """Parse the variables an element of a Schematron declares.

    Parameters:
        parser: The parser, knowing the Schematron's prefixes.
        element: The schema, pattern or rule element.

    Returns:
        Each ``let`` child's name and parsed value, in order.
    """
    return tuple(
        (str(let.get('name')), parser.parse(str(let.get('value'))))
        for let in element.iterfind(f'{_S}let')
    )


def _message(parser: XPath2Parser, element: Any) -> tuple[str | XPathToken, ...]:
    """Parse the message of an assert or report.

    Parameters:
        parser: The parser, knowing the Schematron's prefixes.
        element: The assert or report.

    Returns:
        Its text, with each ``value-of`` parsed as the space-separated string values of
        what it selects and each ``name`` as the name of its path, or of the node the rule
        matched; a ``title`` is left out, and any other element is kept as written.
    """
    parts: list[str | XPathToken] = [element.text or '']
    for part in element:
        if part.tag == f'{_S}value-of':
            select = part.get('select')
            parts.append(parser.parse(f"string-join(for $v in ({select}) return string($v), ' ')"))
        elif part.tag == f'{_S}name':
            path = part.get('path')
            parts.append(parser.parse('name()' if path is None else f'name({path})'))
        elif isinstance(part.tag, str) and local_name(part) != 'title':
            parts.append(etree.tostring(part, encoding='unicode', with_tail=False))
        parts.append(part.tail or '')
    return tuple(parts)


def _severity(element: Any) -> Severity:
    """Say whether an assert or a report is a warning or an error.

    Parameters:
        element: The assert or report.

    Returns:
        A warning when its ``role``, or its rule's when it carries none, is one of
        :data:`WARNING_ROLES`, in any case and with white space trimmed; an error
        otherwise.
    """
    role = element.get('role')
    if role is None:
        role = element.getparent().get('role')
    if role is not None and str(role).strip().lower() in WARNING_ROLES:
        return Severity.WARNING
    return Severity.ERROR


def _check(parser: XPath2Parser, element: Any) -> _Check:
    """Parse one assert or report.

    Parameters:
        parser: The parser, knowing the Schematron's prefixes.
        element: The assert or report.

    Returns:
        The parsed check.
    """
    titles = [
        str(part.text or '').strip()
        for part in element
        if isinstance(part.tag, str) and local_name(part) == 'title'
    ]
    return _Check(
        is_assert=element.tag == f'{_S}assert',
        test=parser.parse(f'boolean(({element.get("test")}))'),
        title=titles[0] if len(titles) > 0 else '',
        message=_message(parser, element),
        severity=_severity(element),
    )


def _branches(context: str) -> list[str]:
    """Split a rule's context at the unions outside its predicates, groups and literals.

    Parameters:
        context: The context, as the Schematron writes it.

    Returns:
        Each branch of its union, stripped; the context alone when it holds no union.
    """
    branches: list[str] = []
    depth = 0
    quote: str | None = None
    start = 0
    for index, character in enumerate(context):
        if quote is not None:
            if character == quote:
                quote = None
        elif character in '"\'':
            quote = character
        elif character in '[(':
            depth += 1
        elif character in '])':
            depth -= 1
        elif character == '|' and depth == 0:
            branches.append(context[start:index].strip())
            start = index + 1
    branches.append(context[start:].strip())
    return branches


def match_expression(context: str) -> str:
    """Return what selects, from the document node, every node a context matches.

    A node matches a context when it is in ``//(context)``.  When the context is a union
    of path expressions joined by ``|``, ``//`` before each branch selects the same
    nodes, since ``//`` is ``/descendant-or-self::node()/`` and the step after it is
    evaluated at every node; a branch that is an absolute path selects the same nodes
    from any node, so it is kept as it is.  A context holding a comment, or the
    ``union``, ``intersect`` or ``except`` keyword, is not split, since a ``|`` in it may
    lie in the comment and a keyword binds operands a split would part: it is selected
    as ``//(context)`` itself.

    Parameters:
        context: A rule's context, as the Schematron writes it.

    Returns:
        The expression, as in ``//pds:Inventory/pds:offset`` for
        ``pds:Inventory/pds:offset``, and ``//(x:b union x:c)`` for ``x:b union x:c``.
    """
    if _NOT_A_PATH_UNION.search(context) is not None:
        return f'//({context})'
    return ' | '.join(
        branch if branch.startswith('/') else f'//{branch}' for branch in _branches(context)
    )


def _rule(parser: XPath2Parser, element: Any) -> _Rule:
    """Parse one rule.

    Parameters:
        parser: The parser, knowing the Schematron's prefixes.
        element: The rule.

    Returns:
        The parsed rule.
    """
    context = str(element.get('context'))
    return _Rule(
        context=context,
        match=parser.parse(match_expression(context)),
        lets=_bindings(parser, element),
        checks=tuple(
            _check(parser, part) for part in element if part.tag in (f'{_S}assert', f'{_S}report')
        ),
    )


@functools.cache
def _compiled(path: Path) -> _Schematron:
    """Parse every expression of one Schematron, once.

    Parameters:
        path: The Schematron's file.

    Returns:
        The parsed Schematron.
    """
    root = etree.parse(str(path)).getroot()
    namespaces = {str(ns.get('prefix')): str(ns.get('uri')) for ns in root.iterfind(f'{_S}ns')}
    # A parser given no collation takes the process's at construction, and a UTF-8 locale
    # makes contains() and starts-with() compare collation keys, which fails the LID
    # prefix rules; a Qt application sets the locale from the environment.  XPath's own
    # default, and the ISO skeleton's, is the code point collation.
    parser = XPath2Parser(namespaces=namespaces, default_collation=UNICODE_CODEPOINT_COLLATION)
    return _Schematron(
        name=path.name,
        namespaces=tuple(namespaces.items()),
        lets=_bindings(parser, root),
        patterns=tuple(
            _Pattern(
                lets=_bindings(parser, pattern),
                rules=tuple(_rule(parser, rule) for rule in pattern.iterfind(f'{_S}rule')),
            )
            for pattern in root.iterfind(f'{_S}pattern')
        ),
    )


def _context(
    tree: Any, namespaces: dict[str, str], item: Any, variables: dict[str, Any]
) -> XPathContext:
    """Return a context to evaluate one expression of a Schematron in.

    Parameters:
        tree: The label's node tree.
        namespaces: The Schematron's prefixes.
        item: The context item, or None for the document node.
        variables: The variables in scope.

    Returns:
        The context.
    """
    return XPathContext(root=tree, namespaces=namespaces, item=item, variables=variables)


def _bind(
    bindings: _Bindings,
    tree: Any,
    namespaces: dict[str, str],
    item: Any,
    variables: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate variables in order, each seeing those before it.

    Parameters:
        bindings: The variables to evaluate.
        tree: The label's node tree.
        namespaces: The Schematron's prefixes.
        item: The node they are evaluated at, or None for the document node.
        variables: The variables already in scope.

    Returns:
        The variables in scope, these added.
    """
    bound = dict(variables)
    for name, token in bindings:
        bound[name] = token.evaluate(_context(tree, namespaces, item, bound))
    return bound


def _node_path(node: Any) -> str:
    """Return where a node a rule matched sits in the label.

    Parameters:
        node: The node, an element or an attribute.

    Returns:
        The element's path, an attribute's as its element's path and ``/@name``, or ``/``
        for the document node.
    """
    if isinstance(node, AttributeNode):
        return f'{_node_path(node.parent)}/@{node.name}'
    if isinstance(node, ElementNode):
        return element_path(node.value)
    return '/'


def _evaluate(file: str, tree: Any, schematron: _Schematron) -> list[Finding]:
    """Evaluate every rule of one Schematron over a label.

    Parameters:
        file: The label's path relative to the bundle's directory, which findings name.
        tree: The label's node tree.
        schematron: The parsed Schematron.

    Returns:
        One finding for each assert that failed and each report that fired, at the node
        its rule matched.
    """
    namespaces = dict(schematron.namespaces)
    variables = _bind(schematron.lets, tree, namespaces, None, {})
    findings: list[Finding] = []
    for pattern in schematron.patterns:
        pattern_variables = _bind(pattern.lets, tree, namespaces, None, variables)
        fired: set[int] = set()
        for rule in pattern.rules:
            matched = rule.match.evaluate(_context(tree, namespaces, None, pattern_variables))
            nodes = matched if isinstance(matched, list) else [matched]
            for node in nodes:
                if node is None or id(node) in fired:
                    continue
                fired.add(id(node))
                rule_variables = _bind(rule.lets, tree, namespaces, node, pattern_variables)
                for check in rule.checks:
                    test_context = _context(tree, namespaces, node, rule_variables)
                    holds = bool(check.test.evaluate(test_context))
                    if holds == check.is_assert:
                        continue
                    text = ''.join(
                        part
                        if isinstance(part, str)
                        else str(part.evaluate(_context(tree, namespaces, node, rule_variables)))
                        for part in check.message
                    )
                    text = ' '.join(text.split())
                    if check.title != '':
                        text = f'{check.title}: {text}'
                    kind = 'assert' if check.is_assert else 'report'
                    findings.append(
                        Finding(
                            file,
                            CheckName.SCHEMATRON,
                            _node_path(node),
                            f'{text} ({kind} of the rule on {rule.context} in {schematron.name})',
                            check.severity,
                        )
                    )
    return findings


def schematron_findings(file: str, document: Any) -> list[Finding]:
    """Evaluate the rules of every Schematron a label declares over it.

    Parameters:
        file: The label's path relative to the bundle's directory, which findings name.
        document: The label, parsed by lxml.

    Returns:
        One finding for each Schematron it declares by a URL of which the package ships no
        copy, and one for each assert that failed and each report that fired.
    """
    findings: list[Finding] = []
    tree: Any = None
    for href in declared_schematron(document):
        copy = shipped_copy(href)
        if copy is None:
            findings.append(
                Finding(
                    file,
                    CheckName.SCHEMATRON,
                    '',
                    f'declares the Schematron {href}, of which the package ships no copy',
                )
            )
            continue
        if tree is None:
            tree = get_node_tree(root=document)
        findings.extend(_evaluate(file, tree, _compiled(copy)))
    return findings
