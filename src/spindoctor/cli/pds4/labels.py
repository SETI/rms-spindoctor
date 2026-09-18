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
    errors therefore writes nothing at all, so whatever was at ``label_path``
    before would be what is there after.  For the labels pass that is nothing,
    because a bundle is written into an empty directory; for a summary pass run
    a second time over a bundle it has already summarized it is the first run's
    label, which would then sit beside the inventory table this run has already
    rewritten and describe data that is no longer there.  The path is therefore
    cleared before the render rather than after it, so that a label on disk is
    always one this run wrote.

    Parameters:
        template: The parsed template to render.
        template_vars: The variables the template's expressions resolve against.
        label_path: Where the label goes.
        logger: Logger for the warning and error reports.

    Returns:
        True if the label is on disk at ``label_path``, False if it is not.
    """
    # Ahead of the render, because a render that draws errors writes nothing
    # and would otherwise leave an earlier run's label describing this run's
    # data.  A label that is not there is the outcome this reports as False.
    label_path.unlink(missing_ok=True)

    error_count, warning_count = cast(
        tuple[int, int], template.write(template_vars, label_path, mode='repair')
    )

    if warning_count > 0:
        logger.warning('Rendering PDS4 label %s drew %d warning(s)', label_path, warning_count)
    if error_count == 0:
        return True

    logger.error(
        'Rendering PDS4 label %s drew %d error(s); it was not written',
        label_path,
        error_count,
    )
    return False
