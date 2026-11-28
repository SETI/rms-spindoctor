# Simulator-report scene gallery

These PNGs are the actual catalog scenes the sensitivity report measures,
rendered from their YAML so a reader sees the frame behind each result. They are
produced in-process by `util/sim_reports/sim_doc_images.py` and committed as
Sphinx assets.

Regenerate (and review the diff) after a scene's geometry changes:

    python -m util.sim_reports.sim_doc_images

The mapping from PNG to scene file is `_REPORT_SCENES` in the generator. Each
image is rendered with `spindoctor.sim.png_export.render_scene_png` (percentile stretch
plus a per-image gamma).
