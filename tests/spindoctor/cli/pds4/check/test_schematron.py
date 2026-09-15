"""The Schematron evaluator's match semantics, over a Schematron and a label written here.

Each test writes a small Schematron into its own directory, which stands in for the
directory of shipped schemas, and a label declaring it, so that the one rule of the
evaluator a test is about is the only thing that decides its outcome.
"""

import locale
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from spindoctor.cli.pds4.check import schemas
from spindoctor.cli.pds4.check.findings import CheckName, Finding, Severity
from spindoctor.cli.pds4.check.schematron import match_expression, schematron_findings

from .controls import parse

SCHEMATRON_NAMESPACE = 'http://purl.oclc.org/dsdl/schematron'
"""ISO Schematron's namespace."""

NAMESPACE = 'urn:example:check'
"""The namespace of the labels written here, under the prefix ``x``."""

SCHEMATRON_NAME = 'CHECK.sch'
"""The file name of the Schematron written here."""


def _evaluate(directory: Path, monkeypatch: pytest.MonkeyPatch, rules: str, body: str) -> Any:
    """Write a Schematron and a label declaring it, and evaluate the one over the other.

    Parameters:
        directory: Where both are written; it stands in for the shipped schemas.
        monkeypatch: Fixture the stand-in directory is installed through.
        rules: The Schematron's content inside its ``schema`` element, after its ``ns``.
        body: The content of the label's root element, ``x:a``.

    Returns:
        The findings.
    """
    (directory / SCHEMATRON_NAME).write_text(
        f'<sch:schema xmlns:sch="{SCHEMATRON_NAMESPACE}" queryBinding="xslt2">\n'
        f'  <sch:ns prefix="x" uri="{NAMESPACE}"/>\n{rules}\n</sch:schema>\n',
        encoding='utf-8',
    )
    monkeypatch.setattr(schemas, 'SCHEMA_DIRECTORY', directory)
    label = directory / 'label.lblx'
    label.write_text(
        f'<?xml-model href="https://example.invalid/check/v1/{SCHEMATRON_NAME}" '
        f'schematypens="{SCHEMATRON_NAMESPACE}"?>\n'
        f'<x:a xmlns:x="{NAMESPACE}">{body}</x:a>\n',
        encoding='utf-8',
    )
    return schematron_findings('label.lblx', parse(label))


def test_a_rule_of_several_steps_fires_at_the_node_it_selects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A two-step context matches its node below the root; its message is evaluated there."""
    rules = (
        '<sch:pattern><sch:rule context="x:c/x:b">'
        '<sch:assert test="@ok = \'yes\'"><title>x:c/x:b</title>b is <sch:value-of '
        'select="@ok"/></sch:assert></sch:rule></sch:pattern>'
    )
    findings = _evaluate(tmp_path, monkeypatch, rules, '<x:c><x:b ok="no"/></x:c>')
    expected = Finding(
        'label.lblx',
        CheckName.SCHEMATRON,
        '/x:a/x:c/x:b',
        f'x:c/x:b: b is no (assert of the rule on x:c/x:b in {SCHEMATRON_NAME})',
    )
    assert findings == [expected]


def test_only_the_first_rule_a_node_matches_in_a_pattern_fires(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A node the first rule of a pattern matches is not held to a later rule of it."""
    rules = (
        '<sch:pattern>'
        '<sch:rule context="x:b"><sch:assert test="true()">first</sch:assert></sch:rule>'
        '<sch:rule context="x:a/x:b"><sch:assert test="false()">second</sch:assert></sch:rule>'
        '</sch:pattern>'
    )
    assert _evaluate(tmp_path, monkeypatch, rules, '<x:b/>') == []


def test_a_schema_variable_is_evaluated_at_the_document_node(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A schema-level ``let`` sees the document node, which has no name."""
    rules = (
        '<sch:let name="where" value="local-name(.)"/>'
        '<sch:pattern><sch:rule context="x:b">'
        '<sch:assert test="$where = \'\'">where</sch:assert></sch:rule></sch:pattern>'
    )
    assert _evaluate(tmp_path, monkeypatch, rules, '<x:b/>') == []


def test_a_pattern_variable_is_evaluated_at_the_document_node(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pattern-level ``let`` sees the document node, which has no name."""
    rules = (
        '<sch:pattern><sch:let name="where" value="local-name(.)"/><sch:rule context="x:b">'
        '<sch:assert test="$where = \'\'">where</sch:assert></sch:rule></sch:pattern>'
    )
    assert _evaluate(tmp_path, monkeypatch, rules, '<x:b/>') == []


def test_a_rule_variable_is_evaluated_at_the_node_the_rule_matched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rule-level ``let`` sees the node its rule matched."""
    rules = (
        '<sch:pattern><sch:rule context="x:b"><sch:let name="here" value="local-name(.)"/>'
        '<sch:assert test="$here = \'b\'">here</sch:assert></sch:rule></sch:pattern>'
    )
    assert _evaluate(tmp_path, monkeypatch, rules, '<x:b/>') == []


def test_a_report_fires_when_its_test_holds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A report whose test is true is a finding, as an assert whose test is false is."""
    rules = (
        '<sch:pattern><sch:rule context="x:b">'
        '<sch:report test="@ok = \'no\'">b reports</sch:report></sch:rule></sch:pattern>'
    )
    findings = _evaluate(tmp_path, monkeypatch, rules, '<x:b ok="no"/>')
    expected = Finding(
        'label.lblx',
        CheckName.SCHEMATRON,
        '/x:a/x:b',
        f'b reports (report of the rule on x:b in {SCHEMATRON_NAME})',
    )
    assert findings == [expected]


def test_a_label_declaring_a_schematron_the_package_does_not_ship_is_a_finding(
    tmp_path: Path,
) -> None:
    """A Schematron URL with no shipped copy is a finding that names it."""
    url = 'https://pds.nasa.gov/pds4/pds/v1/PDS4_PDS_1N00.sch'
    label = tmp_path / 'label.lblx'
    label.write_text(
        f'<?xml-model href="{url}" schematypens="{SCHEMATRON_NAMESPACE}"?>\n'
        '<Product_Bundle xmlns="http://pds.nasa.gov/pds4/pds/v1"/>\n',
        encoding='utf-8',
    )
    expected = Finding(
        'label.lblx',
        CheckName.SCHEMATRON,
        '',
        f'declares the Schematron {url}, of which the package ships no copy',
    )
    assert schematron_findings('label.lblx', parse(label)) == [expected]


@pytest.fixture
def collating_locale() -> Iterator[None]:
    """Collate strings by a language's rules for one test, as a Qt application leaves a process.

    Yields:
        Nothing; the collation in force before the test is restored after it.
    """
    before = locale.setlocale(locale.LC_COLLATE)
    try:
        locale.setlocale(locale.LC_COLLATE, 'en_US.UTF-8')
    except locale.Error:
        pytest.skip('this machine has no en_US.UTF-8 locale to collate strings by')
    try:
        yield
    finally:
        locale.setlocale(locale.LC_COLLATE, before)


def test_a_substring_test_compares_code_points_whatever_the_locale(
    collating_locale: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``contains`` and ``starts-with`` compare code points, XPath's default, not by the locale."""
    rules = (
        '<sch:pattern><sch:rule context="x:b">'
        '<sch:let name="prefix" value="\'urn:nasa:pds:\'"/>'
        '<sch:assert test="contains(\'(urn:nasa:pds:, urn:esa:psa:)\', $prefix)">in</sch:assert>'
        '<sch:assert test="starts-with(@lid, $prefix)">starts</sch:assert>'
        '</sch:rule></sch:pattern>'
    )
    assert _evaluate(tmp_path, monkeypatch, rules, '<x:b lid="urn:nasa:pds:bundle"/>') == []


@pytest.mark.parametrize(
    ('context', 'expression'),
    [
        pytest.param('pds:Inventory/pds:offset', '//pds:Inventory/pds:offset', id='relative'),
        pytest.param('/pds:Product_Bundle', '/pds:Product_Bundle', id='absolute'),
        pytest.param(
            "pds:a[@b = 'x|y'] | (pds:c | pds:d)/pds:e",
            "//pds:a[@b = 'x|y'] | //(pds:c | pds:d)/pds:e",
            id='union',
        ),
    ],
)
def test_a_context_is_matched_from_the_document_node(context: str, expression: str) -> None:
    """Each branch of a context's union is sought below the document node, unless absolute.

    Parameters:
        context: A rule's context.
        expression: The expression selecting the nodes it matches.
    """
    assert match_expression(context) == expression


def test_a_rule_whose_role_marks_a_warning_warns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed assert of a rule whose ``role`` is ``warning`` is a warning."""
    rules = (
        '<sch:pattern><sch:rule context="x:b" role="warning">'
        '<sch:assert test="false()">b warns</sch:assert></sch:rule></sch:pattern>'
    )
    expected = Finding(
        'label.lblx',
        CheckName.SCHEMATRON,
        '/x:a/x:b',
        f'b warns (assert of the rule on x:b in {SCHEMATRON_NAME})',
        Severity.WARNING,
    )
    assert _evaluate(tmp_path, monkeypatch, rules, '<x:b/>') == [expected]


def test_an_assert_whose_role_marks_a_warning_warns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed assert whose own ``role`` is ``WARN``, in a rule with none, is a warning."""
    rules = (
        '<sch:pattern><sch:rule context="x:b">'
        '<sch:assert test="false()" role="WARN">b warns</sch:assert></sch:rule></sch:pattern>'
    )
    expected = Finding(
        'label.lblx',
        CheckName.SCHEMATRON,
        '/x:a/x:b',
        f'b warns (assert of the rule on x:b in {SCHEMATRON_NAME})',
        Severity.WARNING,
    )
    assert _evaluate(tmp_path, monkeypatch, rules, '<x:b/>') == [expected]


@pytest.mark.parametrize(
    ('context', 'locations'),
    [
        pytest.param('x:b union x:c', ['/x:a/x:b', '/x:a/x:c'], id='union'),
        pytest.param('x:c union x:b', ['/x:a/x:b', '/x:a/x:c'], id='union-reversed'),
        pytest.param('x:b intersect x:b', ['/x:a/x:b'], id='intersect'),
        pytest.param("x:b (: it's the first :) | x:c", ['/x:a/x:b', '/x:a/x:c'], id='comment'),
    ],
)
def test_a_context_other_than_a_union_of_paths_matches_what_it_selects(
    context: str, locations: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A context holding a set operator's keyword or a comment matches as ``//(context)`` does.

    Parameters:
        context: A rule's context.
        locations: Where the nodes it matches lie in the label.
        tmp_path: The directory the Schematron and the label are written in.
        monkeypatch: Fixture the stand-in directory is installed through.
    """
    rules = (
        f'<sch:pattern><sch:rule context="{context}">'
        '<sch:assert test="false()">here</sch:assert></sch:rule></sch:pattern>'
    )
    findings = _evaluate(tmp_path, monkeypatch, rules, '<x:b/><x:c/>')
    assert sorted(finding.location for finding in findings) == locations
