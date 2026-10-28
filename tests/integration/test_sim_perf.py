"""Single-core cold-render performance budget for the forward model.

A 512x512 scene with a whole-scene PSF plus the full detector stack at
oversample 4 must render in under 2 s single-core, and a 1024x1024
Cassini-class scene in under 8 s -- for a star-field frame (the optics +
detector stack), for a frame dominated by a large lit body with limb
relief (the topographic body renderer's split-resolution path plus the
terminator shadow march), and for a twelve-feature ring system (the
annulus-bounded ring stage's scaling with feature count).  The budget is a
*cold-render* budget: the render caches are cleared so the timed render pays
the kernel-build and compile costs a first render pays.  One-time costs that are not render
cost -- the lazy YAML config load, module imports -- are paid by an untimed
warm-up render (whose caches are cleared again) before the timers start.

The harness enforces single-core itself: it pins the process to one CPU
with an affinity mask and caps every BLAS/OpenMP pool to one thread via
``threadpoolctl`` for the duration of the timed render, so an unpinned
numpy FFT cannot silently multithread and fake the budget.  (Thread-count
*environment variables* would be inert here: OpenBLAS sizes its pool when
numpy is first imported, long before this test runs.)  This is
integration-marked so it runs under ``pytest -m ''`` (and the deliberate
integration layer) but not the fast unit suite.

The assertion reads the render's process CPU time on the pinned core, not
wall time.  CPU time is far less load-sensitive than wall time (it does not
charge the render for time slices consumed by unrelated processes), but it
is not immune to load: cache and memory-bandwidth contention from a busy
host makes each instruction cost more cycles, inflating CPU time by roughly
10-25% under heavy neighbors -- and by 40% or more, sustained for the whole
run, when a parallel test battery saturates every core.  The budget check
takes the best of up to three cold attempts, passing as soon as one attempt
meets the budget; that absorbs transient contention, but not the sustained
kind, which is why ``scripts/run-all-checks.sh`` excludes this file from its
parallel pytest run and executes it as a dedicated serial step afterwards.
Run this file on its own (not under a concurrent battery) when measuring.
A breach across all attempts on an otherwise-quiet host is a real
regression to investigate, not to bless by raising the budget.  Wall time
is measured and reported alongside CPU time in the failure message for
context.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest
from threadpoolctl import threadpool_limits

from spindoctor.sim import render

pytestmark = pytest.mark.integration

# Cold attempts per budget check: pass on the first attempt that meets the
# budget, fail only when all of them breach it.
_MAX_ATTEMPTS = 3


@contextmanager
def _single_core() -> Iterator[None]:
    """Pin the process to one CPU and cap BLAS/OpenMP pools to one thread."""
    saved_affinity: set[int] | None = None
    if hasattr(os, 'sched_getaffinity'):
        saved_affinity = set(os.sched_getaffinity(0))
        os.sched_setaffinity(0, {min(saved_affinity)})
    try:
        # threadpoolctl talks to the already-loaded BLAS/OpenMP runtimes
        # directly, so the cap works even though numpy sized its pools at
        # import time.
        with threadpool_limits(limits=1):
            yield
    finally:
        if saved_affinity is not None:
            os.sched_setaffinity(0, saved_affinity)


def _psf_detector_scene(size: int) -> dict[str, Any]:
    """A size x size Cassini-class star field with a PSF and the full detector stack.

    A body-free star-navigation frame at oversample 4 exercises the whole-scene
    PSF convolution on the oversampled grid and the electron detector stack
    (shot + read noise, dark, hot pixels, banding, bias structure, cosmic rays,
    bloom, quantization).
    """
    return {
        'size_v': size,
        'size_u': size,
        'random_seed': 1,
        'instrument': 'coiss_nac',
        'exposure_sec': 1.0,
        'artifacts': {'instrument_defaults': True},
        'optics': {'psf': {'sigma_v': 0.55, 'sigma_u': 0.55, 'w': 0.025, 'r0': 2.0, 'n': 3.0}},
        'noise': {
            'poisson': True,
            'read_noise_dn': 4.0,
            'cosmic_ray_events_per_pixel_per_sec': 0.001,
            'bloom_length': 2,
        },
        'sky_counts': {'density_factor': 200.0},
    }


def _body_psf_detector_scene(size: int) -> dict[str, Any]:
    """A size x size body-dominated frame with relief, a PSF, and the detector stack.

    A large lit body with limb relief at oversample 4 exercises the topographic
    body renderer under the same budgets as the star field: detector-resolution
    shading upsampled to the oversampled grid, the relief-perturbed silhouette,
    and the capped terminator shadow march, followed by the PSF convolution and
    the detector stack.  The high phase angle puts the terminator across the
    disc so the march band is fully active.
    """
    scene = _psf_detector_scene(size)
    del scene['sky_counts']
    scene['bodies'] = [
        {
            'name': 'BUDGET-BODY',
            'center_v': size / 2.0,
            'center_u': size / 2.0,
            'axis1': size * 0.7,
            'axis2': size * 0.7,
            'axis3': size * 0.7,
            'illumination_angle': 30.0,
            'phase_angle': 70.0,
            'limb_relief_rms': 0.01,
            'limb_relief_corr_deg': 15.0,
        }
    ]
    return scene


def _haze_body_psf_detector_scene(size: int) -> dict[str, Any]:
    """A body-dominated frame whose body also carries an atmospheric haze layer.

    The lit relief body of the plain budget scene at reduced semi-axes, plus
    an ``atmosphere`` block with a detached shell at oversample 4: the haze
    evaluation and its halo transmission screen run on top of the
    topographic render, the PSF convolution, and the detector stack.  The
    haze cost scales with the bounding box of the body plus its halo (all
    coordinate grids and per-pixel terms are built only inside that box),
    not with the frame, so this holds the same 2 s (512) and 8 s (1024)
    single-core budgets as the plain body scene.
    """
    scene = _body_psf_detector_scene(size)
    # The body shrinks from the plain scene's 0.7 frame widths to 0.55: at
    # 0.7 the thin-haze 1024 render measured 7.0-7.7 s CPU against the 8 s
    # budget on quiet serial runs of the reference host (as little as ~4%
    # headroom on the slower runs -- flake territory); at 0.55 it measures
    # 6.0-6.3 s best-of-3 across quiet serial runs on the same host (21-25%
    # headroom), while the disc still dominates the frame and the full haze
    # stack stays exercised.
    for axis_key in ('axis1', 'axis2', 'axis3'):
        scene['bodies'][0][axis_key] = size * 0.55
    # A realistically thin haze (scale height a few pixels) relative to the
    # large body, so the halo adds only a narrow margin to the body's own
    # bounding box.  A thick variant (scale_height_px 20, detached_px 60)
    # measures within the 1024 budget too, but with under 5% headroom --
    # not enough margin for a loaded host -- so the committed budget scene
    # keeps the thin geometry.
    scene['bodies'][0]['atmosphere'] = {
        'scale_height_px': 4.0,
        'tau_ref': 3.0,
        'ref_altitude_px': 0.0,
        'g': 0.6,
        'detached_px': 12.0,
    }
    return scene


def _ring_psf_detector_scene(size: int) -> dict[str, Any]:
    """A size x size ring-system frame with a PSF and the full detector stack.

    Twelve radial features spanning the frame -- interior sheets (one-sided
    edges), ringlets, gaps, and a density-wave train, every one on an
    eccentric orbit carrying an m-mode -- exercise the per-feature
    annulus-bounded ring stage at oversample 4 under the same budget as the
    star-field and body frames.  Realism scenes compose tens of features,
    so the ring stage's scaling with feature count is what this budget
    holds.
    """
    scene = _psf_detector_scene(size)
    del scene['sky_counts']
    scale = size / 1024.0
    specs: list[tuple[str, float, float]] = [
        ('edge', 90.0, 0.0),
        ('ringlet', 140.0, 18.0),
        ('gap', 150.0, 6.0),
        ('ringlet', 200.0, 30.0),
        ('edge', 260.0, 0.0),
        ('ringlet', 300.0, 12.0),
        ('gap', 305.0, 4.0),
        ('ringlet', 360.0, 25.0),
        ('wave', 400.0, 0.0),
        ('ringlet', 430.0, 15.0),
        ('gap', 460.0, 8.0),
        ('ringlet', 500.0, 20.0),
    ]
    features: list[dict[str, Any]] = []
    for index, (kind, a, width) in enumerate(specs):
        feature: dict[str, Any] = {
            'name': f'BUDGET-RING-{index}',
            'kind': kind,
            'tau': 0.8,
            'orbit': {
                'a_px': a * scale,
                'ae_px': 1.0 * scale,
                'long_peri_deg': 10.0 * index,
                'rate_peri_deg_per_day': 0.0,
                'modes': [{'m': 3, 'amp_px': 1.5 * scale, 'peri_deg': 20.0}],
            },
        }
        if kind in ('ringlet', 'gap'):
            feature['width_px'] = width * scale
        if kind == 'edge':
            feature['side'] = 'in'
        if kind == 'wave':
            feature['wavelength_px'] = 12.0 * scale
            feature['density_wave_damping_px'] = 30.0 * scale
        features.append(feature)
    scene['ring_system'] = {
        'geometry': {
            'center_v': size / 2.0,
            'center_u': size / 2.0,
            'opening_deg_obs': 40.0,
            'opening_deg_sun': 30.0,
            'node_deg': 30.0,
        },
        'phase_deg': 60.0,
        'features': features,
    }
    return scene


def _warm_non_render_costs() -> None:
    """Pay every one-time non-render cost before the timers start.

    A trivial warm-up render forces the lazy config-YAML load and any
    deferred module imports; the render caches are then cleared again so the
    timed render stays genuinely cold on the render paths themselves.
    """
    render.render_combined_model(
        {
            'size_v': 8,
            'size_u': 8,
            'random_seed': 1,
            'instrument': 'coiss_nac',
            'noise': {'read_noise_dn': 1.0},
        }
    )
    render.clear_render_caches()


def _cold_render_seconds(scene: dict[str, Any]) -> tuple[float, float]:
    """Time one render from cold caches, single-core.

    Returns:
        (cpu_seconds, wall_seconds) for the render.  The budget assertion
        reads the CPU time; the wall time is reported for context.
    """
    with _single_core():
        render.clear_render_caches()
        cpu_start = time.process_time()
        wall_start = time.perf_counter()
        render.render_combined_model(scene)
        return time.process_time() - cpu_start, time.perf_counter() - wall_start


def _assert_cold_render_budget(scene: dict[str, Any], budget_s: float, label: str) -> None:
    """Best-of-``_MAX_ATTEMPTS`` cold-render budget check.

    Each attempt clears the render caches, so every attempt is genuinely
    cold.  Passes as soon as one attempt's CPU time meets the budget (one
    clean attempt proves the code meets it; contention inflation is
    transient); fails with every attempt's CPU and wall time when all of
    them breach it (a persistent breach is a real regression).

    Parameters:
        scene: The scene to render.
        budget_s: The single-core CPU-seconds budget.
        label: Scene label for the failure message.
    """
    _warm_non_render_costs()
    attempts: list[tuple[float, float]] = []
    for _ in range(_MAX_ATTEMPTS):
        cpu, wall = _cold_render_seconds(scene)
        attempts.append((cpu, wall))
        if cpu < budget_s:
            return
    detail = ', '.join(f'{cpu:.2f}s CPU / {wall:.2f}s wall' for cpu, wall in attempts)
    pytest.fail(
        f'{label} cold render exceeded the {budget_s:.1f}s CPU budget on all '
        f'{_MAX_ATTEMPTS} attempts: {detail}'
    )


def test_512_psf_detector_render_under_2s() -> None:
    """A 512x512 PSF + detector scene renders in under 2 s single-core (cold)."""
    _assert_cold_render_budget(_psf_detector_scene(512), 2.0, '512x512')


def test_1024_cassini_render_under_8s() -> None:
    """A 1024x1024 Cassini-class scene renders in under 8 s single-core (cold)."""
    _assert_cold_render_budget(_psf_detector_scene(1024), 8.0, '1024x1024')


# Body-bearing budget scenes (#290): the previous body renderer's
# per-subsample shading overran these budgets ~4x at oversample 4; the
# topographic renderer's split-resolution path must hold them.


def test_512_body_render_under_2s() -> None:
    """A 512x512 lit-body scene with relief renders in under 2 s single-core."""
    _assert_cold_render_budget(_body_psf_detector_scene(512), 2.0, '512x512 body')


def test_1024_body_render_under_8s() -> None:
    """A 1024x1024 lit-body scene with relief renders in under 8 s single-core."""
    _assert_cold_render_budget(_body_psf_detector_scene(1024), 8.0, '1024x1024 body')


def test_512_haze_body_render_under_2s() -> None:
    """A 512x512 haze-body scene renders in under 2 s single-core (cold)."""
    _assert_cold_render_budget(_haze_body_psf_detector_scene(512), 2.0, '512x512 haze body')


def test_1024_haze_body_render_under_8s() -> None:
    """A 1024x1024 haze-body scene renders in under 8 s single-core (cold)."""
    _assert_cold_render_budget(_haze_body_psf_detector_scene(1024), 8.0, '1024x1024 haze body')


def test_1024_ring_system_render_under_8s() -> None:
    """A 1024x1024 twelve-feature ring scene renders in under 8 s single-core."""
    _assert_cold_render_budget(_ring_psf_detector_scene(1024), 8.0, '1024x1024 ring')
