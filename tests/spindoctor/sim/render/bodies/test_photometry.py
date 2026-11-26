"""Analytic point checks of the renderer's photometric laws.

Each law is asserted against its closed form at known (mu0, mu, k) values,
the lunar-Lambert blend against its endpoints (pure Lommel-Seeliger at
phase 0, pure Lambert once the McEwen cubic reaches 0), Lommel-Seeliger's
flat disc at phase 0 against Lambert's limb darkening, and the opposition
surge against its documented normalized-exponential form.  A scene body that
leaves the law out shades with lunar-Lambert.
"""

import math

import numpy as np
import pytest
from tests.spindoctor.sim.scenes import disc_body, frame_scene, quiet_noise

from spindoctor.sim.render.bodies.photometry import (
    MINNAERT_CEILING,
    incidence_cosines,
    shade_surface,
    surge_factor,
)
from spindoctor.sim.render.driver import render_scene


def _one(value: float) -> np.ndarray:
    """A single-element float64 array."""
    return np.array([value], dtype=np.float64)


def test_lambert_is_the_incidence_cosine() -> None:
    """Lambert: I = mu0."""
    out = shade_surface(_one(0.37), _one(0.9), law='lambert')
    assert float(out[0]) == pytest.approx(0.37, abs=1e-12)


def test_lommel_seeliger_point_value() -> None:
    """Lommel-Seeliger: I = 2 * mu0 / (mu0 + mu)."""
    out = shade_surface(_one(0.6), _one(0.8), law='lommel_seeliger')
    assert float(out[0]) == pytest.approx(2.0 * 0.6 / 1.4, abs=1e-12)


def test_lommel_seeliger_passes_one_toward_the_sunlit_limb() -> None:
    """Lommel-Seeliger is carried above 1 where the sun is higher than the camera."""
    out = shade_surface(_one(0.9), _one(0.1), law='lommel_seeliger')
    assert float(out[0]) == pytest.approx(1.8, abs=1e-12)


def test_lommel_seeliger_is_flat_at_zero_phase() -> None:
    """At phase 0 (mu0 = mu) Lommel-Seeliger is 1 across the disc; Lambert darkens.

    This is the limb-darkening profile difference the law mismatch plants:
    the Lambert disc falls as mu toward the limb while the Lommel-Seeliger
    disc stays flat.
    """
    mu = np.linspace(0.05, 1.0, 20)
    ls = shade_surface(mu, mu, law='lommel_seeliger')
    lambert = shade_surface(mu, mu, law='lambert')
    np.testing.assert_allclose(ls, np.ones_like(mu), atol=1e-12)
    assert float(lambert[0]) < float(lambert[-1])


def test_minnaert_point_value() -> None:
    """Minnaert: I = mu0**k * mu**(k - 1) at known mu0, mu, k."""
    out = shade_surface(_one(0.6), _one(0.8), law='minnaert', minnaert_k=0.5)
    assert float(out[0]) == pytest.approx(math.sqrt(0.6) / math.sqrt(0.8), abs=1e-12)


def test_minnaert_k_one_is_lambert() -> None:
    """Minnaert with k = 1 degenerates to Lambert exactly."""
    mu0 = np.linspace(0.02, 0.9, 15)
    mu = np.linspace(0.9, 0.1, 15)
    np.testing.assert_allclose(
        shade_surface(mu0, mu, law='minnaert', minnaert_k=1.0),
        shade_surface(mu0, mu, law='lambert'),
        atol=1e-12,
    )


def test_minnaert_limb_divergence_is_capped() -> None:
    """The k < 1 limb divergence is capped at the Lommel-Seeliger ceiling of 2."""
    out = shade_surface(_one(0.5), _one(1e-6), law='minnaert', minnaert_k=0.5)
    assert float(out[0]) == MINNAERT_CEILING


def test_minnaert_below_the_ceiling_passes_one() -> None:
    """A k < 1 Minnaert value between 1 and the ceiling is carried uncapped."""
    out = shade_surface(_one(0.8), _one(0.25), law='minnaert', minnaert_k=0.5)
    assert float(out[0]) == pytest.approx(math.sqrt(0.8) / math.sqrt(0.25), abs=1e-12)


def test_lunar_lambert_zero_phase_endpoint_is_lommel_seeliger() -> None:
    """At phase 0 the McEwen blend is 1: pure Lommel-Seeliger."""
    mu0 = np.linspace(0.05, 1.0, 10)
    mu = np.linspace(1.0, 0.1, 10)
    np.testing.assert_allclose(
        shade_surface(mu0, mu, law='lunar_lambert', phase_angle=0.0),
        shade_surface(mu0, mu, law='lommel_seeliger'),
        atol=1e-12,
    )


def test_lunar_lambert_high_phase_endpoint_is_lambert() -> None:
    """Past the cubic's zero (~119 deg) the blend is 0: pure Lambert."""
    mu0 = np.linspace(0.05, 1.0, 10)
    mu = np.linspace(1.0, 0.1, 10)
    np.testing.assert_allclose(
        shade_surface(mu0, mu, law='lunar_lambert', phase_angle=math.radians(150.0)),
        shade_surface(mu0, mu, law='lambert'),
        atol=1e-12,
    )


def test_lunar_lambert_mid_phase_is_the_documented_cubic() -> None:
    """At 60 deg phase the blend equals the McEwen cubic value."""
    blend = 1.0 - 0.019 * 60.0 + 2.42e-4 * 60.0**2 - 1.46e-6 * 60.0**3
    mu0, mu = 0.6, 0.8
    expected = blend * (2.0 * mu0 / (mu0 + mu)) + (1.0 - blend) * mu0
    out = shade_surface(_one(mu0), _one(mu), law='lunar_lambert', phase_angle=math.radians(60.0))
    assert float(out[0]) == pytest.approx(expected, abs=1e-12)


def test_surge_factor_form() -> None:
    """The surge is (1 + A * exp(-alpha / w)) / (1 + A): 1 at opposition."""
    assert surge_factor(0.0, amplitude=0.8, width_deg=5.0) == pytest.approx(1.0, abs=1e-12)
    alpha = math.radians(30.0)
    expected = (1.0 + 0.8 * math.exp(-30.0 / 5.0)) / 1.8
    assert surge_factor(alpha, amplitude=0.8, width_deg=5.0) == pytest.approx(expected, abs=1e-12)
    assert surge_factor(alpha, amplitude=0.0, width_deg=5.0) == 1.0


def test_surge_scales_the_law() -> None:
    """shade_surface multiplies the law by the surge factor."""
    alpha = math.radians(40.0)
    plain = shade_surface(_one(0.5), _one(0.9), law='lambert', phase_angle=alpha)
    surged = shade_surface(
        _one(0.5),
        _one(0.9),
        law='lambert',
        phase_angle=alpha,
        surge_amplitude=1.0,
        surge_width_deg=10.0,
    )
    factor = surge_factor(alpha, amplitude=1.0, width_deg=10.0)
    assert float(surged[0]) == pytest.approx(float(plain[0]) * factor, abs=1e-12)


@pytest.mark.parametrize('law', ['lambert', 'lommel_seeliger', 'minnaert', 'lunar_lambert'])
def test_night_side_is_black(law: str) -> None:
    """Geometric night on the visible disc renders 0 under every law."""
    out = shade_surface(_one(-0.3), _one(0.5), law=law, phase_angle=math.radians(60.0))
    assert float(out[0]) == 0.0


def test_far_hemisphere_is_black() -> None:
    """A sunlit point on the far hemisphere renders 0."""
    out = shade_surface(_one(0.6), _one(-0.5), law='lambert')
    assert float(out[0]) == 0.0


def test_unknown_law_is_rejected() -> None:
    """An unknown law name raises with the vocabulary in the message."""
    with pytest.raises(ValueError, match='unknown photometric law'):
        shade_surface(_one(0.5), _one(0.5), law='hapke')


def test_incidence_cosines_match_the_lambert_renderer() -> None:
    """The cos(incidence) helper reproduces lambert_from_normals' interior values.

    Inside (0, 1) the Lambert renderer's output IS the incidence cosine, so the
    two paths must agree exactly.
    """
    from spindoctor.sim.shared.ellipsoid import lambert_from_normals

    rng = np.random.default_rng(4)
    normal_v = rng.uniform(-0.4, 0.4, 32)
    normal_u = rng.uniform(-0.4, 0.4, 32)
    normal_z = np.sqrt(1.0 - normal_v**2 - normal_u**2)
    kwargs = {'illumination_angle': 0.7, 'phase_angle': 0.9}
    cos_i = incidence_cosines(normal_v, normal_u, normal_z, **kwargs)
    lambert = lambert_from_normals(normal_v, normal_u, normal_z, **kwargs)
    interior = (cos_i > 0.02) & (cos_i < 0.99)
    assert interior.any()
    np.testing.assert_array_equal(cos_i[interior], lambert[interior])


def test_a_body_without_a_law_shades_lunar_lambert() -> None:
    """A scene body that leaves the law out renders exactly as a lunar-Lambert body."""
    body = disc_body(32.0, 32.0, 20.0, illumination_angle=40.0, phase_angle=30.0)
    stated = {**body, 'photometric_law': 'lunar_lambert'}
    absent_image, _ = render_scene(frame_scene(noise=quiet_noise(), bodies=[body]))
    stated_image, _ = render_scene(frame_scene(noise=quiet_noise(), bodies=[stated]))
    np.testing.assert_array_equal(absent_image, stated_image)
