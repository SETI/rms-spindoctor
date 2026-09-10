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
    errors therefore writes nothing at all, and a bundle is written into an
    empty directory, so nothing is left at ``label_path`` for that label to be
    confused with.

    Parameters:
        template: The parsed template to render.
        template_vars: The variables the template's expressions resolve against.
        label_path: Where the label goes.
        logger: Logger for the warning and error reports.

    Returns:
        True if the label is on disk at ``label_path``, False if it is not.
    """
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
