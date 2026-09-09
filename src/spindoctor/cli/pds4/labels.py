"""Writing one PDS4 label, and reporting the write that did not happen.

``pdstemplate`` reports an unresolved template variable, a failed expression or a
template validation error through the ``(errors, warnings)`` pair its ``write``
returns rather than by raising, so a caller that discards that pair cannot tell a
product that got a label from one that did not.  Every label the bundle stage
writes goes through :func:`write_label`, which is what turns that pair into a
report on the log and an answer to the caller.
"""

from typing import Any, cast

import pdstemplate
from filecache import FCPath
from pdslogger import PdsLogger


def _remove_label_file(label_path: FCPath, *, logger: PdsLogger) -> None:
    """Remove whatever sits at a label path, reporting what happened.

    A path with nothing at it is the outcome asked for, so it is not reported.
    Anything the filesystem refuses to remove is reported at error level naming
    the path, because the caller is about to say the label is not on disk and
    something at that path is.

    Parameters:
        label_path: The label path to clear.
        logger: Logger for the report.
    """
    try:
        label_path.unlink()
    except FileNotFoundError:
        return
    except OSError as exc:
        logger.error('Could not remove the file at PDS4 label path %s: %s', label_path, exc)
        return
    logger.warning('Removed the PDS4 label an earlier run left at %s', label_path)


def write_label(
    template: pdstemplate.PdsTemplate,
    template_vars: dict[str, Any],
    label_path: FCPath,
    *,
    logger: PdsLogger,
) -> bool:
    """Render one template into one label, reporting whatever went wrong.

    The label is written in ``pdstemplate``'s repair mode, which saves a label
    that drew warnings but never one that drew errors.  A render that draws
    errors writes nothing at all, so a label an earlier run left at
    ``label_path`` would otherwise survive untouched and stand in the bundle for
    the label this run could not write.  It is removed instead: a bundle missing
    a label says so, and a bundle assembled out of two runs does not.

    Repair mode reads back whatever is already at ``label_path`` in order to
    compare it with what it rendered, so a file left there by a killed run that
    is not readable as text ends the render.  That file is removed too, and the
    label counts as not written, which leaves the operator's re-run a clear path
    to write on.

    Parameters:
        template: The parsed template to render.
        template_vars: The variables the template's expressions resolve against.
        label_path: Where the label goes.
        logger: Logger for the warning and error reports.

    Returns:
        True if the label is on disk at ``label_path``, False if it is not.
    """
    try:
        error_count, warning_count = cast(
            tuple[int, int], template.write(template_vars, label_path, mode='repair')
        )
    except UnicodeDecodeError:
        logger.error(
            'The file at PDS4 label path %s is not readable as text; no label was written',
            label_path,
        )
        _remove_label_file(label_path, logger=logger)
        return False

    if warning_count > 0:
        logger.warning('Rendering PDS4 label %s drew %d warning(s)', label_path, warning_count)
    if error_count == 0:
        return True

    logger.error(
        'Rendering PDS4 label %s drew %d error(s); it was not written',
        label_path,
        error_count,
    )
    _remove_label_file(label_path, logger=logger)
    return False
