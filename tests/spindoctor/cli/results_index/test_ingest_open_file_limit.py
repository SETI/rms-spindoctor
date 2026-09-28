"""Tests that every ingest pass is bounded by the open-file limit before it reads.

A batch of documents retrieved from a cloud root costs an open file apiece
while the batch is in flight, and
:func:`spindoctor.nav_records.within_open_file_limit` is what holds a pass down
to what the limit covers.  The bound is only worth anything where it is
actually applied, so what is pinned here is that each of the three passes that
retrieves documents applies it, to the roots that pass will read, before it
reads any of them.

The bound itself -- what it computes, and when it leaves a pass alone -- is
tested beside it, in ``tests/spindoctor/nav_records/test_descriptors.py``.
"""

from typing import Any, cast

import pytest

from spindoctor.cli.results_index import driver, tasks
from spindoctor.config import MAIN_LOGGER
from spindoctor.nav_records import TreeTuning

ROOT = 'gs://a-bucket/a-results-root'
"""A root whose documents are downloaded, and so cost descriptors."""


class _Bounded(Exception):  # noqa: N818
    """Raised in place of the bound, to stop the pass as soon as it has been applied."""


def _recording(seen: list[list[str]]) -> Any:
    """Build a stand-in for the bound that records its roots and stops the pass.

    Parameters:
        seen: List each call's roots are appended to.

    Returns:
        The stand-in.
    """

    def recording(tuning: TreeTuning, roots: Any, **_: Any) -> TreeTuning:
        seen.append([str(root) for root in roots])
        raise _Bounded

    return recording


def test_an_ingest_is_bounded_before_it_walks_a_root(monkeypatch: pytest.MonkeyPatch) -> None:
    """The pass that reported the failure this exists to prevent.

    Parameters:
        monkeypatch: Fixture the bound is replaced through.
    """
    seen: list[list[str]] = []
    monkeypatch.setattr(driver, 'within_open_file_limit', _recording(seen))
    with pytest.raises(_Bounded):
        driver.ingest_metadata_files(
            cast(Any, None), [ROOT], logger=MAIN_LOGGER, tuning=TreeTuning()
        )
    assert seen == [[ROOT]]


def test_a_fan_out_is_bounded_once_for_every_root_it_divides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Once for the fan-out, not once per root, which is what listing each would say.

    Parameters:
        monkeypatch: Fixture the bound is replaced through.
    """
    seen: list[list[str]] = []
    monkeypatch.setattr(tasks, 'within_open_file_limit', _recording(seen))
    with pytest.raises(_Bounded):
        tasks.fan_out_ingest_tasks(
            cast(Any, None), [ROOT, ROOT + '-two'], logger=MAIN_LOGGER, tuning=TreeTuning()
        )
    assert seen == [[ROOT, ROOT + '-two']]


def test_a_task_share_is_bounded_by_its_own_root(monkeypatch: pytest.MonkeyPatch) -> None:
    """A cloud worker retrieves the same way, and under the same limit, as the driver.

    Parameters:
        monkeypatch: Fixture the bound and the share parser are replaced through.
    """
    seen: list[list[str]] = []
    monkeypatch.setattr(tasks, 'within_open_file_limit', _recording(seen))
    monkeypatch.setattr(
        tasks,
        '_share_from_task',
        lambda data: tasks._Share(
            run_id=1, root_url=ROOT, force=False, has_file_metrics=True, files=[]
        ),
    )
    with pytest.raises(_Bounded):
        tasks.ingest_task_share(cast(Any, None), {}, logger=MAIN_LOGGER, tuning=TreeTuning())
    assert seen == [[ROOT]]
