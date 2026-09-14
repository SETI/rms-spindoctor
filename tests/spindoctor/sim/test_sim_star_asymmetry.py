"""Image-side star information-asymmetry: catalog error, scatter, binaries, variables.

A scene renders every star -- displaced off its catalog position by an explicit
per-star error or the seeded scene-level scatter, at a variable brightness, and
with an unresolved companion -- while the navigator is told only the catalog
values.  These tests exercise the renderer directly (the truth records and the
deposited image), independent of any navigation run: the rendered star lands
where the planted error puts it, the truth metadata reports the realized delta,
and the existing star deposit is byte-identical when the new keys default off.
"""

from typing import Any

import numpy as np

from spindoctor.sim.render import clear_render_caches, render_combined_model

_QUIET_NOISE: dict[str, Any] = {'poisson': False, 'read_noise_dn': 0.0, 'bias_dn': 0.0}


def _scene(stars: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    """A quiet single-frame star scene with a navigator-matched PSF."""
    scene: dict[str, Any] = {
        'size_v': 128,
        'size_u': 128,
        'random_seed': 7,
        'instrument': 'coiss_nac',
        'exposure_sec': 1.0,
        'optics': {'psf': {'match_navigator': True}},
        'noise': dict(_QUIET_NOISE),
        'stars': stars,
    }
    scene.update(extra)
    return scene


def _centroid(img: np.ndarray, center_v: int, center_u: int, radius: int) -> tuple[float, float]:
    """Flux-weighted centroid of a cutout centered on ``(center_v, center_u)``."""
    win = img[center_v - radius : center_v + radius + 1, center_u - radius : center_u + radius + 1]
    vv, uu = np.mgrid[0 : win.shape[0], 0 : win.shape[1]].astype(np.float64)
    total = float(win.sum())
    mv = float((win * vv).sum()) / total
    mu = float((win * uu).sum()) / total
    return mv + center_v - radius, mu + center_u - radius


def test_catalog_error_displaces_the_rendered_star() -> None:
    """An explicit catalog_error moves the rendered star off its catalog position."""
    scene = _scene(
        [
            {
                'name': 'A',
                'v': 64.5,
                'u': 64.5,
                'vmag': 4.0,
                'catalog_error_v': 2.0,
                'catalog_error_u': -1.5,
            }
        ]
    )
    img, _ = render_combined_model(scene)
    cv, cu = _centroid(img, 66, 62, 6)
    assert abs(cv - 66.0) < 0.1
    assert abs(cu - 62.5) < 0.1


def test_catalog_error_recorded_in_truth() -> None:
    """The truth metadata carries the realized rendered-vs-catalog position delta."""
    scene = _scene(
        [
            {
                'name': 'A',
                'v': 64.5,
                'u': 64.5,
                'vmag': 4.0,
                'catalog_error_v': 2.0,
                'catalog_error_u': -1.5,
            }
        ]
    )
    _, meta = render_combined_model(scene)
    info = meta['star_info'][0]
    assert abs(info['catalog_error_v'] - 2.0) < 1e-6
    assert abs(info['catalog_error_u'] - (-1.5)) < 1e-6


def test_scene_scatter_adds_to_explicit_error() -> None:
    """The scene scatter draw adds to any explicit per-star catalog error."""
    scene = _scene(
        [{'name': 'A', 'v': 64.5, 'u': 64.5, 'vmag': 4.0, 'catalog_error_v': 2.0}],
        star_catalog_scatter_px=3.0,
    )
    _, meta = render_combined_model(scene)
    # The realized v error is the explicit 2.0 plus a nonzero scatter draw.
    assert meta['star_info'][0]['catalog_error_v'] != 2.0


def test_scene_scatter_is_deterministic() -> None:
    """The same seed reproduces the same scatter realization bit-for-bit."""
    scene = _scene([{'name': 'A', 'v': 64.5, 'u': 64.5, 'vmag': 4.0}], star_catalog_scatter_px=4.0)
    clear_render_caches()
    first, _ = render_combined_model(scene)
    clear_render_caches()
    second, _ = render_combined_model(scene)
    assert np.array_equal(first, second)


def test_scene_scatter_survives_differential_star_smear() -> None:
    """A stars-class smear keeps the scattered star displacement.

    Differential smear replaces the point-source plane with the per-class star
    layer, so the layer must carry the same seeded scatter draw as the primary
    deposit: the smeared render's star centroid stays at the scattered position
    (catalog plus the realized draw), never snapping back to the catalog one.
    """
    base = _scene(
        [{'name': 'A', 'v': 64.5, 'u': 64.5, 'vmag': 4.0}],
        star_catalog_scatter_px=5.0,
    )
    smeared = _scene(
        [{'name': 'A', 'v': 64.5, 'u': 64.5, 'vmag': 4.0}],
        star_catalog_scatter_px=5.0,
    )
    smeared['optics'] = {
        'psf': {'match_navigator': True},
        'smear': [{'dv_px': 0.0, 'du_px': 4.0, 'object_class': 'stars'}],
    }
    img_plain, meta_plain = render_combined_model(base)
    img_smeared, meta_smeared = render_combined_model(smeared)
    # Both renders realize the same seeded draw; it is large enough to detect.
    err_v = float(meta_plain['star_info'][0]['catalog_error_v'])
    err_u = float(meta_plain['star_info'][0]['catalog_error_u'])
    assert meta_smeared['star_info'][0]['catalog_error_v'] == err_v
    assert (err_v**2 + err_u**2) ** 0.5 > 0.5
    center_v = round(64.0 + err_v)
    center_u = round(64.0 + err_u)
    cv_plain, cu_plain = _centroid(img_plain, center_v, center_u, 10)
    cv_smeared, cu_smeared = _centroid(img_smeared, center_v, center_u, 10)
    # The unsmeared render sits at the scattered position...
    assert abs(cv_plain - (64.0 + err_v)) < 0.1
    assert abs(cu_plain - (64.0 + err_u)) < 0.1
    # ...and the centred smear kernel preserves that centroid, so the smeared
    # render differs from the catalog position by the same realized draw.
    assert abs(cv_smeared - cv_plain) < 0.1
    assert abs(cu_smeared - cu_plain) < 0.1


def test_variable_star_renders_fainter() -> None:
    """A positive delta_mag renders the star fainter than its catalog vmag."""
    catalog = _scene([{'name': 'A', 'v': 64.5, 'u': 64.5, 'vmag': 4.0}])
    variable = _scene([{'name': 'A', 'v': 64.5, 'u': 64.5, 'vmag': 4.0, 'delta_mag': 2.5}])
    img_c, _ = render_combined_model(catalog)
    img_v, _ = render_combined_model(variable)
    sum_c = float(img_c[54:75, 54:75].sum())
    sum_v = float(img_v[54:75, 54:75].sum())
    # 2.5 magnitudes fainter is one tenth the flux.
    assert abs(sum_v - sum_c / 10.0) < 0.02 * sum_c


def test_companion_pulls_the_photocenter() -> None:
    """An unresolved companion shifts the blended photocenter toward it."""
    single = _scene([{'name': 'A', 'v': 64.5, 'u': 64.5, 'vmag': 4.0}])
    binary = _scene(
        [
            {
                'name': 'A',
                'v': 64.5,
                'u': 64.5,
                'vmag': 4.0,
                'companion': {'sep_px': 4.0, 'delta_mag': 0.5, 'angle_deg': 0.0},
            }
        ]
    )
    img_s, _ = render_combined_model(single)
    img_b, meta = render_combined_model(binary)
    cv_s, _ = _centroid(img_s, 64, 64, 8)
    cv_b, _ = _centroid(img_b, 65, 64, 8)
    # angle_deg 0 places the companion along +v, so the photocenter moves in +v.
    assert cv_b > cv_s + 0.5
    assert meta['star_info'][0]['has_companion'] is True


def test_navigable_flag_recorded_but_star_still_renders() -> None:
    """A non-navigable star renders and its flag is recorded in the truth."""
    scene = _scene(
        [
            {'name': 'KNOWN', 'v': 40.5, 'u': 40.5, 'vmag': 4.0, 'navigable': True},
            {'name': 'CONF', 'v': 88.5, 'u': 88.5, 'vmag': 4.0, 'navigable': False},
        ]
    )
    img, meta = render_combined_model(scene)
    flags = {info['name']: info['navigable'] for info in meta['star_info']}
    assert flags == {'KNOWN': True, 'CONF': False}
    # The non-navigable confounder is genuinely on the detector.
    assert float(img[84:93, 84:93].max()) > 1.0


def test_defaulted_keys_render_byte_identical() -> None:
    """A scene without the new keys renders identically to one that sets their defaults."""
    plain = _scene([{'name': 'A', 'v': 50.5, 'u': 70.5, 'vmag': 3.5}])
    explicit = _scene(
        [
            {
                'name': 'A',
                'v': 50.5,
                'u': 70.5,
                'vmag': 3.5,
                'catalog_error_v': 0.0,
                'catalog_error_u': 0.0,
                'delta_mag': 0.0,
                'navigable': True,
            }
        ]
    )
    img_plain, _ = render_combined_model(plain)
    img_explicit, _ = render_combined_model(explicit)
    assert np.array_equal(img_plain, img_explicit)
