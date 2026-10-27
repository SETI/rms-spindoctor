"""Render the documentation example images for the simulator chapters.

Run as ``python -m tests.integration.sim_doc_images`` (no holdings or SPICE
needed -- everything is rendered in-process).  It writes two galleries of
viewable PNGs:

* ``docs/dev_guide/_sim_images/`` -- illustrative scenes for the simulator
  developer-guide chapter (one per scene ingredient: ellipsoid vs mesh body,
  craters, crescent, rings, stars, multi-body, detector noise, stray light, and
  a composite frame).  These are hand-built ``sim_params`` dicts chosen to show
  one feature each.

* ``docs/simulator_report/_scene_images/`` -- the *actual* catalog scenes the
  sensitivity report discusses (rendered from their YAML), so a reader sees the
  frames behind each measurement.

Each gallery also gets a ``NOTES.md`` describing how to regenerate it, so the
images can be rebuilt if the renderer changes.  The PNGs are committed assets
(Sphinx embeds them directly); rerun this module and review the diff after any
change that alters rendering.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

from spindoctor.sim.png_export import render_scene_png
from spindoctor.sim.scene import load_sim_scene, validate_sim_params

_DOCS = Path(__file__).parent.parent.parent / 'docs'
_GUI_DIR = _DOCS / 'dev_guide' / '_sim_images'
_SCENES_ROOT = Path(__file__).parent / 'sim_scenes'
_REPORT_DIR = _DOCS / 'simulator_report' / '_scene_images'

_COISS = 'coiss_nac'


def _scene(bodies: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    """Build a minimal coiss_nac sim_params scene around a body list."""
    params: dict[str, Any] = {
        'size_v': 220,
        'size_u': 220,
        'instrument': _COISS,
        'random_seed': 7,
        'offset_v': 0.0,
        'offset_u': 0.0,
        'noise': {'poisson': True, 'read_noise_dn': 3.0},
        'bodies': bodies,
    }
    params.update(extra)
    return params


def _ellipsoid(**over: Any) -> dict[str, Any]:
    body = {
        'name': 'BODY',
        'center_v': 110.0,
        'center_u': 110.0,
        'axis1': 160.0,
        'axis2': 120.0,
        'axis3': 110.0,
        'illumination_angle': 25.0,
        'phase_angle': 35.0,
    }
    body.update(over)
    return body


def _mesh(**over: Any) -> dict[str, Any]:
    body = _ellipsoid()
    body.update(
        {
            'shape_model': 'polyhedral_mesh',
            'mesh_lumpiness': 0.4,
            'mesh_seed': 3,
            # Fine enough that flat-shaded facets do not show at gallery size.
            'mesh_n_lat': 64,
            'mesh_n_lon': 128,
            'pose_euler_deg': [10.0, 35.0, 0.0],
        }
    )
    body.update(over)
    return body


def _scatter_stars(
    count: int,
    seed: int,
    *,
    size: int = 220,
    vmag_lo: float = 3.0,
    vmag_hi: float = 6.0,
    margin: float = 8.0,
) -> list[dict[str, Any]]:
    """Return a deterministic spread of explicit bright catalog stars.

    Under the flux-normalized star model a faint sky-count field sits at the
    read-noise floor, so a legible star-field panel is built from explicit
    bright stars (a few hundred DN each) scattered across the frame rather than
    a dense faint sky.  The draw is seeded, so the panel is reproducible.

    Parameters:
        count: Number of stars to scatter.
        seed: Seed for the reproducible position / magnitude draw.
        size: Frame edge length in pixels (stars land inside a margin of it).
        vmag_lo: Bright end of the magnitude draw.
        vmag_hi: Faint end of the magnitude draw.
        margin: Keep-out border in pixels so no star clips the frame edge.

    Returns:
        A list of ``{name, v, u, vmag}`` star dicts.
    """
    rng = np.random.default_rng(seed)
    stars: list[dict[str, Any]] = []
    for i in range(count):
        stars.append(
            {
                'name': f'S{i}',
                'v': round(float(rng.uniform(margin, size - margin)), 1),
                'u': round(float(rng.uniform(margin, size - margin)), 1),
                'vmag': round(float(rng.uniform(vmag_lo, vmag_hi)), 2),
            }
        )
    return stars


# (filename, sim_params, render kwargs) for the developer-guide gallery.
_GUI_GALLERY: list[tuple[str, dict[str, Any], dict[str, Any]]] = [
    ('ellipsoid_body', _scene([_ellipsoid()]), {'gamma': 1.1}),
    ('mesh_body', _scene([_mesh()]), {'gamma': 1.1}),
    # Gouraud shading on the coarse default tessellation: it smooths the facets
    # inside the silhouette, while the silhouette itself stays a polygon.
    (
        'mesh_body_gouraud',
        _scene([_mesh(shading='gouraud', mesh_n_lat=16, mesh_n_lon=32)]),
        {'gamma': 1.1},
    ),
    (
        'body_craters',
        _scene([_ellipsoid(axis2=150.0, crater_fill=0.5, crater_max_radius=0.3)]),
        {'gamma': 1.1},
    ),
    ('crescent_body', _scene([_mesh(phase_angle=130.0)]), {'gamma': 1.4}),
    (
        'topographic_limb',
        _scene([_ellipsoid(limb_relief_rms=0.03, limb_relief_corr_deg=12.0)]),
        {'gamma': 1.1},
    ),
    (
        'ragged_terminator',
        _scene([_ellipsoid(phase_angle=125.0, limb_relief_rms=0.035, limb_relief_corr_deg=10.0)]),
        {'gamma': 1.4},
    ),
    (
        'haze_limb_body',
        _scene(
            [
                _ellipsoid(
                    axis1=170.0,
                    axis2=170.0,
                    axis3=170.0,
                    illumination_angle=90.0,
                    phase_angle=40.0,
                    atmosphere={'scale_height_px': 8.0, 'tau_ref': 3.0, 'g': 0.6},
                )
            ]
        ),
        {'gamma': 1.2},
    ),
    (
        'haze_crescent_horns',
        _scene(
            [
                _ellipsoid(
                    axis1=170.0,
                    axis2=170.0,
                    axis3=170.0,
                    illumination_angle=90.0,
                    phase_angle=150.0,
                    atmosphere={'scale_height_px': 8.0, 'tau_ref': 3.0, 'g': 0.75},
                )
            ]
        ),
        {'gamma': 1.5},
    ),
    (
        'banded_transit',
        _scene(
            [
                _ellipsoid(
                    axis1=180.0,
                    axis2=180.0,
                    axis3=180.0,
                    illumination_angle=20.0,
                    phase_angle=25.0,
                    disc_texture={
                        'band_amplitude': 0.22,
                        'band_wavenumber': 8.0,
                        'storms': [
                            {
                                'lat_deg': -25.0,
                                'lon_deg': 90.0,
                                'radius_deg': 8.0,
                                'albedo_factor': 1.35,
                            }
                        ],
                    },
                    transits=[
                        {
                            'moon': {
                                'dv_px': -28.0,
                                'du_px': 36.0,
                                'radius_px': 11.0,
                                'albedo_factor': 1.3,
                            }
                        },
                        {
                            'shadow': {
                                'dv_px': 10.0,
                                'du_px': -32.0,
                                'radius_px': 9.0,
                                'darkness': 0.85,
                            }
                        },
                    ],
                )
            ]
        ),
        {'gamma': 1.1},
    ),
    (
        'mutual_event',
        _scene(
            [
                _ellipsoid(
                    name='FAR',
                    center_v=110.0,
                    center_u=95.0,
                    axis1=110.0,
                    axis2=110.0,
                    axis3=110.0,
                    illumination_angle=20.0,
                    phase_angle=30.0,
                    range_km=700000.0,
                ),
                _ellipsoid(
                    name='NEAR',
                    center_v=110.0,
                    center_u=125.0,
                    axis1=120.0,
                    axis2=120.0,
                    axis3=120.0,
                    illumination_angle=20.0,
                    phase_angle=30.0,
                    range_km=500000.0,
                ),
            ]
        ),
        {'gamma': 1.1},
    ),
    (
        'rings',
        _scene(
            [],
            ring_system={
                'geometry': {
                    'center_v': 110.0,
                    'center_u': 110.0,
                    'opening_deg_obs': 90.0,
                    'opening_deg_sun': 90.0,
                    'node_deg': 0.0,
                },
                'features': [
                    {
                        'name': 'RINGLET_INNER',
                        'kind': 'ringlet',
                        'tau': 2.0,
                        'width': 10.0,
                        'orbit': {'a': 56.0},
                    },
                    {
                        'name': 'RINGLET_OUTER',
                        'kind': 'ringlet',
                        'tau': 0.7,
                        'width': 8.0,
                        'orbit': {'a': 90.0, 'ae': 6.0},
                    },
                ],
            },
        ),
        {'gamma': 1.2},
    ),
    # The four panels below mirror the ring_system catalog scenes of the same
    # names (tests/integration/sim_scenes/ring_system/), minus the planted
    # offsets and expected blocks the gallery does not exercise.
    (
        'ring_edge_wave_gap',
        _scene(
            [],
            random_seed=11,
            ring_system={
                'geometry': {
                    'center_v': 110.0,
                    'center_u': 110.0,
                    'opening_deg_obs': 50.0,
                    'opening_deg_sun': 50.0,
                    'node_deg': 30.0,
                },
                'features': [
                    {
                        'name': 'SHEET',
                        'kind': 'edge',
                        'side': 'in',
                        'tau': 1.2,
                        'orbit': {'a': 95.0},
                    },
                    {
                        'name': 'WAVYGAP',
                        'kind': 'gap',
                        'tau': 1.2,
                        'width': 6.0,
                        'navigable': True,
                        'orbit': {
                            'a': 60.0,
                            'edge_wave': {
                                'amp': 2.0,
                                'wavelength': 10.0,
                                'damp': 0.6,
                                'lam0': 45.0,
                            },
                        },
                    },
                ],
            },
        ),
        {'gamma': 1.2},
    ),
    (
        'ring_mmode',
        _scene(
            [],
            random_seed=12,
            ring_system={
                'geometry': {
                    'center_v': 110.0,
                    'center_u': 110.0,
                    'opening_deg_obs': 65.0,
                    'opening_deg_sun': 65.0,
                    'node_deg': 15.0,
                },
                'features': [
                    {
                        'name': 'BRINGEDGE',
                        'kind': 'ringlet',
                        'tau': 2.0,
                        'width': 12.0,
                        'navigable': True,
                        'orbit': {'a': 70.0, 'modes': [{'m': 2, 'amp': 5.0, 'peri': 40.0}]},
                    },
                ],
            },
        ),
        {'gamma': 1.2},
    ),
    (
        'ring_spokes',
        _scene(
            [],
            random_seed=13,
            ring_system={
                'geometry': {
                    'center_v': 110.0,
                    'center_u': 110.0,
                    'opening_deg_obs': 90.0,
                    'opening_deg_sun': 90.0,
                    'node_deg': 0.0,
                },
                'features': [
                    {
                        'name': 'BSHEET',
                        'kind': 'ringlet',
                        'tau': 1.5,
                        'width': 40.0,
                        'navigable': True,
                        'orbit': {'a': 45.0},
                    },
                ],
                'azimuthal': {
                    'spokes': {
                        'count': 6,
                        'r_inner': 50.0,
                        'r_outer': 80.0,
                        'contrast': -0.5,
                        'width_deg': 14.0,
                    },
                },
            },
        ),
        {'gamma': 1.2},
    ),
    (
        'ring_moonlet_propeller',
        _scene(
            [],
            random_seed=14,
            ring_system={
                'geometry': {
                    'center_v': 110.0,
                    'center_u': 110.0,
                    'opening_deg_obs': 90.0,
                    'opening_deg_sun': 90.0,
                    'node_deg': 0.0,
                },
                'features': [
                    {
                        'name': 'ASHEET',
                        'kind': 'ringlet',
                        'tau': 1.8,
                        'width': 45.0,
                        'navigable': True,
                        'orbit': {'a': 40.0},
                    },
                    {
                        'name': 'ENCKE',
                        'kind': 'gap',
                        'tau': 1.8,
                        'width': 8.0,
                        'orbit': {'a': 58.0},
                    },
                ],
                'moonlets': [
                    {
                        'a': 62.0,
                        'lam_deg': 130.0,
                        'radius_px': 1.5,
                        'amplitude': 0.5,
                        'propeller': {'length_deg': 25.0, 'width_px': 3.0, 'contrast': -0.7},
                    },
                ],
            },
        ),
        {'gamma': 1.2},
    ),
    (
        'star_field',
        _scene(
            [],
            optics={'psf': {'match_navigator': True}},
            stars=_scatter_stars(45, 11, vmag_hi=6.5),
            sky_counts={'a': -1.5, 'b': 0.34, 'density_factor': 1.0},
            noise={'poisson': True, 'read_noise_dn': 1.5},
        ),
        {'gamma': 1.6, 'high_percentile': 99.9},
    ),
    (
        'body_and_stars',
        _scene(
            [_ellipsoid(axis1=120.0, axis2=95.0, axis3=85.0)],
            optics={'psf': {'match_navigator': True}},
            stars=_scatter_stars(28, 5),
            sky_counts={'a': -1.8, 'b': 0.34, 'density_factor': 1.0},
            noise={'poisson': True, 'read_noise_dn': 2.0},
        ),
        {'gamma': 1.6, 'high_percentile': 99.9},
    ),
    (
        'multi_body',
        _scene(
            [
                _ellipsoid(
                    name='A', center_v=70.0, center_u=70.0, axis1=90.0, axis2=80.0, axis3=75.0
                ),
                _mesh(name='B', center_v=150.0, center_u=150.0, axis1=70.0, axis2=55.0, axis3=50.0),
                _ellipsoid(
                    name='C', center_v=60.0, center_u=160.0, axis1=40.0, axis2=36.0, axis3=34.0
                ),
            ]
        ),
        {'gamma': 1.2},
    ),
    (
        'detector_noise',
        _scene(
            [_ellipsoid(axis1=120.0, axis2=95.0, axis3=85.0)],
            noise={
                'poisson': True,
                'read_noise_dn': 8.0,
                'cosmic_ray_rate_per_sec': 0.0006,
                'missing_data_rate': 0.01,
            },
            exposure_sec=2.0,
        ),
        {'gamma': 1.3, 'high_percentile': 99.7},
    ),
    (
        'telemetry_loss',
        _scene(
            [
                _ellipsoid(
                    axis1=150.0,
                    axis2=150.0,
                    axis3=150.0,
                    illumination_angle=25.0,
                    phase_angle=30.0,
                )
            ],
            artifacts={
                'missing_lines': {'expected_events': 10.0},
                'partial_lines': {'expected_events': 10.0},
            },
        ),
        {'gamma': 1.15},
    ),
    (
        'stray_light_gradient',
        _scene(
            [_ellipsoid(axis1=120.0, axis2=95.0, axis3=85.0)],
            optics={'stray_light': {'amplitude': 0.5, 'direction_deg': 35.0, 'model': 'linear'}},
        ),
        {'gamma': 1.2},
    ),
    (
        'composite_scene',
        _scene(
            [
                _mesh(
                    name='MOON',
                    center_v=140.0,
                    center_u=95.0,
                    axis1=90.0,
                    axis2=72.0,
                    axis3=66.0,
                    range_km=500000.0,
                )
            ],
            optics={'psf': {'match_navigator': True}},
            stars=_scatter_stars(22, 3),
            sky_counts={'a': -1.8, 'b': 0.34, 'density_factor': 1.0},
            ring_system={
                'geometry': {
                    'center_v': 110.0,
                    'center_u': 110.0,
                    'opening_deg_obs': 90.0,
                    'opening_deg_sun': 90.0,
                    'node_deg': 0.0,
                },
                # The moon sits in front of the ring (physical depths).
                'range_km': 1000000.0,
                'km_per_pixel': 1000.0,
                'features': [
                    {
                        'name': 'RINGLET',
                        'kind': 'ringlet',
                        'tau': 1.2,
                        'width': 20.0,
                        'orbit': {'a': 150.0, 'ae': 8.0},
                    },
                ],
            },
            noise={'poisson': True, 'read_noise_dn': 4.0},
        ),
        {'gamma': 1.5, 'high_percentile': 99.9},
    ),
]

# (filename, scene-relative path) for the report scene gallery: the actual
# catalog scenes the report measures, rendered from their YAML.
_REPORT_SCENES: list[tuple[str, str, dict[str, Any]]] = [
    ('regular_sphere_base', 'phase_sweep_regular_body/regular_sphere_base.yaml', {'gamma': 1.1}),
    ('disc', 'algorithmic_invariants/planted_offset_disc.yaml', {'gamma': 1.1}),
    ('blob_crescent', 'algorithmic_invariants/planted_offset_blob_crescent.yaml', {'gamma': 1.4}),
    ('mesh_disc', 'algorithmic_invariants/planted_offset_irregular.yaml', {'gamma': 1.1}),
    ('limb_mesh', 'algorithmic_invariants/planted_offset_limb_mesh.yaml', {'gamma': 1.1}),
    ('shape_mismatch', 'algorithmic_invariants/planted_offset_shapemismatch.yaml', {'gamma': 1.1}),
    (
        'mesh_crescent',
        'algorithmic_invariants/planted_offset_blob_mesh_crescent.yaml',
        {'gamma': 1.4},
    ),
    ('pose_disagree', 'phase_sweep_irregular_body/hyperion_pose_disagree.yaml', {'gamma': 1.1}),
    ('ring', 'algorithmic_invariants/planted_offset_ring.yaml', {'gamma': 1.3}),
    (
        'star_field',
        'algorithmic_invariants/planted_offset_star_field.yaml',
        {'gamma': 1.9, 'high_percentile': 99.9},
    ),
]

_GUI_NOTES = """\
# Developer-guide simulator gallery

These PNGs illustrate the simulator's scene ingredients for
`dev_guide_simulator.rst`. They are rendered in-process by
`tests/integration/sim_doc_images.py` from hand-built `sim_params` dicts (one
feature per image) and committed as Sphinx assets.

Regenerate (and review the diff) after any change that alters rendering:

    python -m tests.integration.sim_doc_images

Each image uses `spindoctor.sim.png_export.render_scene_png`, which stretches detector
counts to visible grayscale with a percentile clip plus a per-image gamma (dim
features such as a crescent or a faint star field use a higher gamma). The scene
definitions live in `_GUI_GALLERY` in the generator; edit there to change a
panel.
"""

_REPORT_NOTES = """\
# Simulator-report scene gallery

These PNGs are the actual catalog scenes the sensitivity report measures,
rendered from their YAML so a reader sees the frame behind each result. They are
produced in-process by `tests/integration/sim_doc_images.py` and committed as
Sphinx assets.

Regenerate (and review the diff) after a scene's geometry changes:

    python -m tests.integration.sim_doc_images

The mapping from PNG to scene file is `_REPORT_SCENES` in the generator. Each
image is rendered with `spindoctor.sim.png_export.render_scene_png` (percentile stretch
plus a per-image gamma).
"""


def generate(*, gui_dir: Path | None = None, report_dir: Path | None = None) -> list[Path]:
    """Render both galleries and write their NOTES files; return all paths.

    Parameters:
        gui_dir: Output directory for the developer-guide gallery; the
            committed ``docs/dev_guide/_sim_images/`` when None.  The
            staleness test points this at a temporary directory to compare
            a fresh render against the committed PNGs.
        report_dir: Output directory for the report scene gallery; the
            committed ``docs/simulator_report/_scene_images/`` when None.

    Returns:
        The written paths (the gallery PNGs; the NOTES files are written
        alongside but not returned).
    """
    gui_dir = _GUI_DIR if gui_dir is None else gui_dir
    report_dir = _REPORT_DIR if report_dir is None else report_dir
    written: list[Path] = []
    gui_dir.mkdir(parents=True, exist_ok=True)
    for name, params, kwargs in _GUI_GALLERY:
        validate_sim_params(params, source=f'_GUI_GALLERY[{name}]')
        written.append(render_scene_png(params, gui_dir / f'{name}.png', upscale=2, **kwargs))
    (gui_dir / 'NOTES.md').write_text(_GUI_NOTES)

    report_dir.mkdir(parents=True, exist_ok=True)
    for name, rel_path, kwargs in _REPORT_SCENES:
        scene_path = _SCENES_ROOT / rel_path
        if not scene_path.is_file():
            print(f'skip {name}: {scene_path} not found', file=sys.stderr)
            continue
        params = load_sim_scene(scene_path)
        written.append(
            render_scene_png(
                params, report_dir / f'{name}.png', ignore_offset=False, upscale=2, **kwargs
            )
        )
    (report_dir / 'NOTES.md').write_text(_REPORT_NOTES)
    return written


def main() -> int:
    """Entry point: render every documentation image."""
    paths = generate()
    print(f'Wrote {len(paths)} documentation image(s):')
    for path in paths:
        print(f'  {path.relative_to(_DOCS.parent)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
