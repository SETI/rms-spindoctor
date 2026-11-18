"""The navigated frame's verdict, rendered off the UI thread.

The editor's picture is offset-free, but the status bar reports the navigator's
verdict on the frame the navigator reads, with the planted offset applied.  That
frame needs a full render of its own, so :class:`VerdictRunner` runs it on a worker
thread, as a :class:`~spindoctor.ui.sim_editor.worker.LatestRequestRunner`.  The
editor asks for it at the same moment as the full render of the picture, once the
user has stopped editing, and the verdict line fills in when it finishes.
"""

from typing import Any

from PyQt6.QtCore import QObject

from spindoctor.config import DEFAULT_CONFIG
from spindoctor.sim.instruments.navigator_link import frame_units_summary, resolve_sim_inst_config
from spindoctor.sim.render.driver import render_combined_model
from spindoctor.ui.sim_editor.worker import LatestRequestRunner

__all__ = ['VerdictRunner', 'navigated_frame_summary']


def navigated_frame_summary(sim_params: dict[str, Any]) -> str:
    """Render the frame the navigator reads and describe it for the status bar.

    The frame carries the planted offset and roll (``ignore_offset=False``).  The
    scene is validated before it renders, as it is for every render, so this
    function raises for a scene that fails validation.  The text is
    :func:`~spindoctor.sim.instruments.navigator_link.frame_units_summary` of that frame
    under the scene's resolved instrument block.

    Parameters:
        sim_params: The scene mapping.  The caller owns it; it is only read.

    Returns:
        One line of text naming the navigated frame's units, range, marker share,
        and the navigator's verdict when it would not call the frame clean.

    Raises:
        SimSceneValidationError: If the scene fails validation.
    """
    image, _ = render_combined_model(sim_params, ignore_offset=False)
    inst_config = resolve_sim_inst_config(
        DEFAULT_CONFIG,
        sim_params.get('instrument'),
        sim_params.get('instrument_config'),
    )
    return frame_units_summary(image, inst_config)


def _summarize(sim_params: dict[str, Any]) -> str:
    """Call :func:`navigated_frame_summary` for the worker.

    The runner holds this wrapper rather than the function itself, so a test can
    replace :func:`navigated_frame_summary` in this module and the worker calls the
    replacement.

    Parameters:
        sim_params: The worker's copy of the scene.

    Returns:
        The navigated frame's status text.
    """
    return navigated_frame_summary(sim_params)


class VerdictRunner(LatestRequestRunner):
    """Runs the navigated-frame verdict on a worker thread, latest request wins.

    ``ready`` carries the request's scene and the status text.
    """

    def __init__(self, parent: QObject | None = None) -> None:
        """Create an idle runner.

        Parameters:
            parent: The owning Qt object (the editor window).
        """
        super().__init__(parent, work=_summarize, description='Navigated-frame verdict render')
