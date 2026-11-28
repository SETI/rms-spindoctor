# Developer-guide simulator gallery

These PNGs illustrate the simulator's scene ingredients for
`dev_guide_simulator.rst`. They are rendered in-process by
`util/sim_reports/sim_doc_images.py` from hand-built `sim_params` dicts (one
feature per image) and committed as Sphinx assets.

Regenerate (and review the diff) after any change that alters rendering:

    python -m util.sim_reports.sim_doc_images

Each image uses `spindoctor.sim.png_export.render_scene_png`, which stretches detector
counts to visible grayscale with a percentile clip plus a per-image gamma (dim
features such as a crescent or a faint star field use a higher gamma). The scene
definitions live in `_GUI_GALLERY` in the generator; edit there to change a
panel.
