"""Widgets for the ``sd_create_simulated_image`` scene editor.

The editor is a single ``QMainWindow`` (:class:`CreateSimulatedImageModel`)
composed from one mixin per schema block, so each block's controls -- and any
control tab a later realism phase adds -- live in their own module rather than
in one monolith.  :func:`main` is the console-script entry point.
"""

from spindoctor.ui.sim_editor.main_window import CreateSimulatedImageModel, main

__all__ = ['CreateSimulatedImageModel', 'main']
