"""The full render of the scene on screen, run off the UI thread.

While the user edits, the editor shows the simulator's fast preview, which draws in a
fraction of a second.  Once the user has stopped editing for a moment, the editor asks
:class:`FullRenderRunner` for the full render of the same scene, the image every test
and saved product uses.  It runs on a worker thread, as a
:class:`~spindoctor.cli.sim_editor.worker.LatestRequestRunner`, and replaces the
preview when it finishes, provided the scene has not changed in the meantime.
"""

from dataclasses import dataclass
from typing import Any

from PyQt6.QtCore import QObject

from spindoctor.cli.sim_editor.worker import LatestRequestRunner
from spindoctor.config import DEFAULT_CONFIG
from spindoctor.sim.instruments.navigator_link import frame_units_summary, resolve_sim_inst_config
from spindoctor.sim.render import render_combined_model
from spindoctor.support.types import NDArrayFloatType

__all__ = ['FullRender', 'FullRenderRunner', 'render_full_frame']


@dataclass(frozen=True)
class FullRender:
    """The full, offset-free render of one scene.

    Parameters:
        image: The rendered image.
        meta: The renderer's output metadata.
        units_summary: The status bar's line for the image: its units, value range,
            marker share, and the navigator's verdict when it would not call the
            frame clean.
    """

    image: NDArrayFloatType
    meta: dict[str, Any]
    units_summary: str


def render_full_frame(sim_params: dict[str, Any]) -> FullRender:
    """Render the full, offset-free image of a scene, as the editor displays it.

    The status bar's line for the image is computed here too, so that on the worker
    thread an error in it is reported as a failed full render, as an error in the
    render itself is.

    Parameters:
        sim_params: The scene mapping.  The caller owns it; it is only read.

    Returns:
        The image and metadata of
        :func:`~spindoctor.sim.render.render_combined_model` with
        ``ignore_offset=True``, and
        :func:`~spindoctor.sim.instruments.navigator_link.frame_units_summary` of the
        image under the scene's resolved instrument block.

    Raises:
        SimSceneValidationError: If the scene fails validation.
    """
    image, meta = render_combined_model(sim_params, ignore_offset=True)
    inst_config = resolve_sim_inst_config(
        DEFAULT_CONFIG,
        sim_params.get('instrument'),
        sim_params.get('instrument_config'),
    )
    return FullRender(image=image, meta=meta, units_summary=frame_units_summary(image, inst_config))


def _render(sim_params: dict[str, Any]) -> FullRender:
    """Call :func:`render_full_frame` for the worker.

    The runner holds this wrapper rather than the function itself, so a test can
    replace :func:`render_full_frame` in this module and the worker calls the
    replacement.

    Parameters:
        sim_params: The worker's copy of the scene.

    Returns:
        The full render.
    """
    return render_full_frame(sim_params)


class FullRenderRunner(LatestRequestRunner):
    """Runs the full render on a worker thread, latest request wins.

    ``ready`` carries the request's scene and its :class:`FullRender`.
    """

    def __init__(self, parent: QObject | None = None) -> None:
        """Create an idle runner.

        Parameters:
            parent: The owning Qt object (the editor window).
        """
        super().__init__(parent, work=_render, description='Full render')
