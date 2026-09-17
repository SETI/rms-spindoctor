"""Tests that the four statements of the index-versus-tree enumeration agree.

The answers a results index gives differently from the results tree are stated
in four places: the module docstring of
:mod:`spindoctor.dataset.results_filter`, the navigation guide's account of
``--results-index-db``, and the plan twice, in its Phase 5 entry and in acceptance
criterion 1.  The guide is one of them because an operator reading it is the
person a silently short selection is served to, and the list has twice been
lost from there while every other copy of it stayed intact.

So the four are read out of their files and compared against each other here.
What is under test is documentation rather than behavior, which is unusual for a
test module and is the point: each member has a behavioral test of its own among
the selection filters' tests, and none of those notices when the sentence that
tells an operator about the member is deleted.
"""

import re
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest
from filecache import FCPath

from spindoctor.dataset import results_filter

ENUMERATION_MEMBERS = {
    'a file with no row': 'has no row at all',
    'a document rewritten in place': 'rewritten in place',
}
"""The answers the index gives differently from the tree, and what names each.

Every place that states the enumeration states each member in its own words,
which is why the member is identified by a phrase it carries rather than by its
position or by the number of members: a list compared by length agrees with a
list that dropped one member and gained another, and agrees with itself after a
member is deleted and an unrelated paragraph is emphasized in its place.

The phrase is chosen to be the one wording every statement of that member shares
and no other member's carries.  Adding a member means adding it here, to the
module docstring, to the navigation guide and to the plan's two lists, in one
commit; so does removing one, which is what a member that has stopped being a
divergence asks for.  A member stops being one when the two storages are made to
answer it alike, and the test that proved the difference then moves to the
parity tests and asserts the agreement.

What this can and cannot check
------------------------------

Each phrase is bound to the entry that carries it and to that entry's place in
the list, in :func:`_members_by_entry`, so what is compared is which member each
entry states rather than which phrases the region holds somewhere.  That is what
makes a deletion fail: an entry removed, an entry that states two members
because a neighbor absorbed the deleted one's phrase, an entry that states
none, and a list whose members come in another order are all a mismatch.

It cannot check that an entry tells the truth about the member it states.  These
are text matches, and a paragraph that keeps its identifying phrase while
claiming the opposite of what the member says passes here and is wrong where it
counts.  Nothing that matches text can close that; what closes it is reading the
paragraph, which is why each member also has a behavioral test of its own and
why this file's guard is described as binding the lists together rather than as
validating any of them.
"""

PLANS = FCPath(Path(__file__).resolve().parents[3]) / 'plans'
PLAN = PLANS / 'archive' / 'RESULTS_INDEX_PLAN_2026-08-04.md'
"""The plan, which states the enumeration twice: in Phase 5 and in criterion 1.

The plan is archived, its work having shipped, and the archive holds it under a
dated name.  Being archived does not free it from this check: it is one of the
four statements the check binds together, and a member added to or removed from
the enumeration is edited into all four or into none.
"""

NAVIGATION_GUIDE = (
    FCPath(Path(__file__).resolve().parents[3])
    / 'docs'
    / 'user_guide'
    / 'user_guide_navigation.rst'
)
"""The guide, which states the enumeration where an operator will meet it.

The member the guide is most easily written without is the one that costs an
operator most: a selection answered from an index is short by every document
the ingest refused, and nothing in the run says so.
"""


def _plan_lines() -> list[str]:
    """Return the plan's lines, skipping the test only where no plan is packaged.

    The plan is a repository document rather than a packaged one, so a checkout
    always has it and an installed tree never does.  What is skipped on is
    therefore the whole ``plans`` directory rather than the one file: a checkout
    that holds the directory and not this file is naming a plan that has moved,
    which is a stale path here and not a tree the check does not apply to.
    Skipping on the file alone makes a renamed plan indistinguishable from an
    installed tree, and this check compares four lists that are meant to move
    together -- so the rename it would go quiet for is exactly the event it
    exists to catch.

    Returns:
        The lines of the plan file.

    Raises:
        FileNotFoundError: If the plans directory is there and this plan is not.
    """
    if not PLANS.is_dir():
        pytest.skip(f'{PLANS.as_posix()} is not in this tree')
    with PLAN.open('r', encoding='utf-8') as plan:
        text: str = plan.read()
    return text.splitlines()


def _normalized(text: str) -> str:
    """Return one entry with the markup the four lists spell differently removed.

    Parameters:
        text: The entry as its list writes it.

    Returns:
        The text with emphasis and literal markers dropped, its line breaks
        collapsed to single spaces, and its case folded, so that one phrase
        identifies a member whether the list around it is Python, ``.rst`` or
        Markdown, and whether the phrase is wrapped across two lines.
    """
    stripped = text.replace('*', '').replace('`', '')
    return re.sub(r'\s+', ' ', stripped).strip().casefold()


def _lead_paragraphs(
    lines: Sequence[str], lead: re.Pattern[str], *, opens: str | None = None, closes: str | None
) -> list[str]:
    """Return each paragraph of one region that a member's own lead opens.

    An entry is its lead line and the lines wrapped under it, up to the blank
    line that closes the paragraph: the identity of a member belongs where a
    reader meets it, which is the paragraph that announces it, and not in a
    sub-list or an aside further down.

    Parameters:
        lines: The lines of the file, or of the docstring.
        lead: Pattern matching the opening line of an entry.
        opens: Text identifying the line the region starts after, or None to
            start at the first line.
        closes: Prefix of the line that ends the region, or None to read to the
            end.

    Returns:
        One entry per lead, each as a single line of text.
    """
    inside = opens is None
    collected: list[list[str]] = []
    open_paragraph = False
    for line in lines:
        if not inside:
            inside = opens is not None and opens in line
            continue
        if closes is not None and line.startswith(closes):
            break
        if lead.match(line):
            collected.append([line.strip()])
            open_paragraph = True
        elif not line.strip():
            open_paragraph = False
        elif open_paragraph:
            collected[-1].append(line.strip())
    return [' '.join(parts) for parts in collected]


def _phase_five_members() -> list[str]:
    """Return the enumeration as the Phase 5 entry states it.

    Returns:
        One entry per member.
    """
    return _lead_paragraphs(
        _plan_lines(),
        re.compile(r'^  \d+\. '),
        opens='**What the index answers differently',
        closes='- **The answer says how old',
    )


def _criterion_one_members() -> list[str]:
    """Return the enumeration as acceptance criterion 1 restates it.

    Returns:
        One entry per member.
    """
    return _lead_paragraphs(
        _plan_lines(),
        re.compile(r'^   \d+\. '),
        opens='## 5. Acceptance criteria',
        closes='2. No pipeline',
    )


def _docstring_members() -> list[str]:
    """Return the enumeration as the module docstring states it.

    Returns:
        One entry per member.
    """
    docstring = results_filter.__doc__ or ''
    return _lead_paragraphs(docstring.splitlines(), re.compile(r'^- \*\*'), closes=None)


def _navigation_guide_members() -> list[str]:
    """Return the enumeration as the navigation guide states it.

    The guide's members are the bold-led paragraphs of its account of
    ``--results-index-db``, which runs from the sentence introducing that option's
    answers to the end of the selection section.

    Returns:
        One entry per member.
    """
    try:
        with NAVIGATION_GUIDE.open('r', encoding='utf-8') as guide:
            text: str = guide.read()
        lines = text.splitlines()
    except FileNotFoundError:
        pytest.skip(f'{NAVIGATION_GUIDE.as_posix()} is not in this tree')
    return _lead_paragraphs(
        lines,
        re.compile(r'^\*\*'),
        opens='Given ``--results-index-db``',
        closes='Miscellaneous',
    )


def _members_by_entry(entries: Sequence[str]) -> list[list[str]]:
    """Return what each entry of one list states, entry by entry.

    The nesting is the whole point: a member is bound to the entry carrying its
    phrase, so a phrase found in a neighboring entry does not stand in for the
    entry that was deleted.  Flattened, the two are the same multiset, and a
    list that lost a member and gained the phrase elsewhere reads as unchanged.

    Parameters:
        entries: The list's entries, in the order the list states them.

    Returns:
        One list per entry, naming the members that entry carries the
        identifying phrase of: empty for an entry stating none, and two names
        long for an entry that absorbed another member's phrase.
    """
    return [
        [
            name
            for name, phrase in ENUMERATION_MEMBERS.items()
            if _normalized(phrase) in _normalized(entry)
        ]
        for entry in entries
    ]


ENUMERATION_LISTS = [
    pytest.param(_docstring_members, id='module-docstring'),
    pytest.param(_navigation_guide_members, id='navigation-guide'),
    pytest.param(_phase_five_members, id='plan-phase-5'),
    pytest.param(_criterion_one_members, id='plan-criterion-1'),
]
"""Every place a member of the enumeration has to be stated, and how to read it.

The guide is one of them because an operator reading it is the person a silently
short selection is served to.  The plan states the enumeration twice, and both
are here, because acceptance criterion 1 restates it as the list rather than as
a pointer at one.
"""


@pytest.mark.parametrize('members', ENUMERATION_LISTS)
def test_every_list_states_each_member_in_an_entry_of_its_own(
    members: Callable[[], list[str]],
) -> None:
    """Each member bound to its own entry, in the order the enumeration declares.

    A selection answered from an index differs from the same selection answered
    from the tree, silently and by however many documents the ingest refused, so
    the operator choosing between them is told which members of this list apply
    to their root.  The way that account has twice been lost is deletion: the
    paragraph goes, and some neighbor ends up carrying the words.  Bound entry
    by entry, that is a list one entry short whose surviving entries no longer
    line up with the members, however the phrases are distributed.

    What passes here and should not is an entry that keeps its phrase and says
    the opposite of what the member says; the constant's docstring says so, and
    the behavioral test of each member is what covers it.
    """
    assert _members_by_entry(members()) == [[name] for name in ENUMERATION_MEMBERS]


@pytest.mark.parametrize('members', ENUMERATION_LISTS)
def test_no_entry_of_a_list_states_anything_outside_the_enumeration(
    members: Callable[[], list[str]],
) -> None:
    """An entry naming no member is a member added to one list and not the rest.

    It is also what a paragraph promoted into the enumeration by emphasis looks
    like: the snapshot, which the docstring states after the list precisely
    because the age of the pass decides nothing about the members, reads as one
    of them the moment its lead is emphasized like theirs.
    """
    entries = members()
    stated = _members_by_entry(entries)
    assert [entry for entry, names in zip(entries, stated, strict=True) if not names] == []
