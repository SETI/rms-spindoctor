"""Tests for the one place a PDS4 label is written.

``pdstemplate`` reports an unresolved variable or a failed expression through the
``(errors, warnings)`` pair its ``write`` returns rather than by raising, and in
repair mode it declines to save a label that drew errors.  ``write_label`` is
what turns that pair into a report and an answer, and what makes sure a label an
earlier run left behind does not stand in for one this run could not write.
"""

from pathlib import Path
from typing import Any

import pdstemplate
import pytest
from filecache import FCPath

from spindoctor.cli.pds4.labels import write_label
from spindoctor.config import MAIN_LOGGER

BROKEN_TEMPLATE = '<Product>$COMPLETELY_UNSET_VARIABLE$</Product>\n'
"""A template whose only expression names a variable no caller defines."""

CLEAN_TEMPLATE = '<Product>nothing to resolve</Product>\n'
"""A template with no expressions at all, so a render of it cannot error."""


class _CountingTemplate:
    """A stand-in template that reports fixed error and warning counts.

    ``pdstemplate`` will not emit a warning without an error on any template this
    project renders, so the warning path needs a template that simply says what
    it drew.  What it does with the label mirrors repair mode: written when
    nothing errored, left alone when something did.
    """

    def __init__(self, *, errors: int, warnings: int) -> None:
        """Prepare a template that will report the given counts.

        Parameters:
            errors: Error count to report from :meth:`write`.
            warnings: Warning count to report from :meth:`write`.
        """
        self._errors = errors
        self._warnings = warnings

    def write(
        self, dictionary: dict[str, Any], label_path: Any, *, mode: str = 'save'
    ) -> tuple[int, int]:
        """Write the label unless something errored, and report the counts.

        Parameters:
            dictionary: The template variables, unused.
            label_path: Where the label goes.
            mode: The save mode the caller asked for, unused.

        Returns:
            The configured error and warning counts.
        """
        if self._errors == 0:
            FCPath(label_path).write_text('<Product/>\n')
        return (self._errors, self._warnings)


def test_warnings_without_errors_still_write_the_label(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A render that only drew warnings produces a label and reports the warnings.

    The level is part of the report: a warning that reached the log as an info
    or an error line would say the wrong thing about a label that is on disk.
    """
    label_path = FCPath(tmp_path) / 'product.lblx'
    template = _CountingTemplate(errors=0, warnings=2)
    written = write_label(template, {}, label_path, logger=MAIN_LOGGER)
    assert written is True
    expected = f'WARNING | Rendering PDS4 label {label_path} drew 2 warning(s)'
    assert expected in capsys.readouterr().out


def test_the_error_report_names_the_label_and_the_count(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A render that drew errors is reported at error level, naming label and count.

    This is the line the user guide sends an operator searching for, so its
    level, the path it names and the count it gives are all part of it.
    """
    label_path = FCPath(tmp_path) / 'product.lblx'
    template = _CountingTemplate(errors=1, warnings=0)
    written = write_label(template, {}, label_path, logger=MAIN_LOGGER)
    assert written is False
    expected = f'ERROR | Rendering PDS4 label {label_path} drew 1 error(s); it was not written'
    assert expected in capsys.readouterr().out


def test_a_stale_label_is_removed_when_the_write_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A label an earlier run left behind does not outlive a run that failed.

    A render that errors writes nothing at all, so the earlier run's label would
    otherwise survive untouched and stand in the bundle for the one this run
    could not write.
    """
    template_path = tmp_path / 'broken.lblx'
    template_path.write_text(BROKEN_TEMPLATE, encoding='utf-8')
    label_path = tmp_path / 'product.lblx'
    label_path.write_text('<Product>from an earlier run</Product>\n', encoding='utf-8')
    template = pdstemplate.PdsTemplate(str(template_path))
    written = write_label(template, {}, FCPath(label_path), logger=MAIN_LOGGER)
    assert written is False
    assert not label_path.exists()
    assert f'Removed the PDS4 label an earlier run left at {label_path}' in capsys.readouterr().out


def test_a_label_that_vanished_before_the_removal_is_not_an_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A label removed by something else between the render and the unlink is fine.

    The removal is what the failed render asks for, so another writer having
    done it first is the outcome, not a fault to raise through the caller.

    Parameters:
        tmp_path: Base temporary directory.
        monkeypatch: Fixture the vanishing is installed through.
    """

    def _vanished(self: FCPath, *args: Any, **kwargs: Any) -> None:
        """Behave as an unlink whose file was removed by something else first.

        Parameters:
            self: The path being unlinked.
            *args: The sub-path the caller passed, unused.
            **kwargs: The removal options the caller passed, unused.
        """
        raise FileNotFoundError(str(self))

    template_path = tmp_path / 'broken.lblx'
    template_path.write_text(BROKEN_TEMPLATE, encoding='utf-8')
    label_path = tmp_path / 'product.lblx'
    label_path.write_text('<Product>from an earlier run</Product>\n', encoding='utf-8')
    template = pdstemplate.PdsTemplate(str(template_path))
    monkeypatch.setattr(FCPath, 'unlink', _vanished)
    written = write_label(template, {}, FCPath(label_path), logger=MAIN_LOGGER)
    assert written is False


def test_a_directory_at_the_label_path_is_reported_not_raised(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Something at the label path that cannot be removed is a label not written.

    The answer the function owes its caller is whether the label is on disk, and
    a directory sitting where the label goes means it is not.
    """
    template_path = tmp_path / 'broken.lblx'
    template_path.write_text(BROKEN_TEMPLATE, encoding='utf-8')
    label_path = tmp_path / 'product.lblx'
    label_path.mkdir()
    template = pdstemplate.PdsTemplate(str(template_path))
    written = write_label(template, {}, FCPath(label_path), logger=MAIN_LOGGER)
    assert written is False
    assert f'Could not remove the file at PDS4 label path {label_path}' in capsys.readouterr().out


def test_an_undecodable_file_at_the_label_path_is_removed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A file a killed run left at the label path does not end the run.

    Repair mode reads back whatever is already there to compare it with what it
    rendered, so bytes that are not text end the render.  Removing them leaves
    the operator's re-run somewhere to write.
    """
    template_path = tmp_path / 'clean.lblx'
    template_path.write_text(CLEAN_TEMPLATE, encoding='utf-8')
    label_path = tmp_path / 'product.lblx'
    label_path.write_bytes(b'\xff\xfe')
    template = pdstemplate.PdsTemplate(str(template_path))
    written = write_label(template, {}, FCPath(label_path), logger=MAIN_LOGGER)
    assert written is False
    assert not label_path.exists()
    assert f'The file at PDS4 label path {label_path} is not readable' in capsys.readouterr().out
