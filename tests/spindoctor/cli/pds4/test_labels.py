"""Tests for the one place a PDS4 label is written.

``pdstemplate`` reports an unresolved variable or a failed expression through the
``(errors, warnings)`` pair its ``write`` returns rather than by raising, and in
repair mode it declines to save a label that drew errors.  ``write_label`` is
what turns that pair into a report and an answer.
"""

from pathlib import Path
from typing import Any

import pytest
from filecache import FCPath

from spindoctor.cli.pds4.labels import write_label
from spindoctor.config import MAIN_LOGGER


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


def test_a_failed_render_does_not_leave_an_earlier_label(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A label that is already there does not survive a render that errors.

    The summary pass rewrites an inventory table and then renders the label
    describing it, and it can be run again over a bundle it has already
    summarized.  Repair mode declines to save a label that drew errors, so
    without this the first run's label would stay beside the second run's
    table and describe data that is no longer there.
    """
    label_path = FCPath(tmp_path) / 'collection_data.lblx'
    label_path.write_text('<Product>from an earlier run</Product>\n')
    template = _CountingTemplate(errors=1, warnings=0)
    written = write_label(template, {}, label_path, logger=MAIN_LOGGER)
    assert written is False
    assert not label_path.exists()
    capsys.readouterr()


def test_a_successful_render_replaces_an_earlier_label(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Clearing the path first does not cost the ordinary re-render its label."""
    label_path = FCPath(tmp_path) / 'collection_data.lblx'
    label_path.write_text('<Product>from an earlier run</Product>\n')
    template = _CountingTemplate(errors=0, warnings=0)
    written = write_label(template, {}, label_path, logger=MAIN_LOGGER)
    assert written is True
    assert label_path.read_text() == '<Product/>\n'
    capsys.readouterr()
