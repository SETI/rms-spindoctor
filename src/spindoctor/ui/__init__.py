"""PyQt6 and matplotlib UI (mosaics, manual offset, the scene editor, shared widgets).

This directory is a regular package so ``spindoctor.ui`` and ``spindoctor.ui.mosaic_viewer`` share
the same import layout.

Modules:

    ``common``
        Shared UI helpers (e.g. stretch controls, zoom/pan) used across dialogs and
        the mosaic viewer.
    ``manual_nav_dialog``
        Matplotlib-based manual navigation dialog.

Subpackages:

    ``mosaic_viewer``
        Ring and body mosaic windows, projections, graticule, and tiled image display.
    ``sim_editor``
        The ``sd_create_simulated_image`` scene editor: one mixin module per scene
        block, assembled into a single main window.
"""

__all__: list[str] = []
