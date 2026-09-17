"""Shared fixtures and helpers for ``spindoctor.nav_technique`` test files.

Each technique end-to-end test needs the same scaffolding: a fake observation
(``FakeObs``), a synthetic image, a populated ``NavContext``, and per-feature
polyline / feature factories.  Centralising these here removes the per-file
duplication and makes the technique tests focus on their assertions instead of
their setup.

Fixtures are exposed as **factory fixtures** — each yields a small builder
function, so each test calls the builder with its own per-test parameters.

What these fixtures can and cannot establish
--------------------------------------------

Every image factory draws its scene about a ``center_vu`` the test chooses,
over a grid built with ``np.arange``, so the scene is stated in the pixel
centric coordinates the array itself uses.  Every geometry factory then states
its polyline vertices and its predicted positions about that same number.  A
technique test therefore recovers the offset the test planted between the two
whatever coordinate system the model layer works in, because both sides of the
comparison were written here from one number.  That is what a technique test is
for -- it measures a fit, not a convention -- but it means nothing in this
directory can fail on a pixel coordinate convention, and a tolerance here is a
statement about the fitter's convergence rather than about a coordinate system.
The tests that pin where a model's emitted vertices actually sit live beside
the models, in ``tests/spindoctor/nav_model``, and anchor on something outside
the model: a frame's own symmetry, a fixture's stated sphere or ramp, or the
optical axis ``oops`` reports.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from spindoctor.feature.feature import NavFeature, NavReliabilityBreakdown
from spindoctor.feature.feature_type import NavFeatureType
from spindoctor.feature.flags import (
    LimbArcFlags,
    RingEdgeFlags,
    StarFlags,
    TerminatorArcFlags,
    TitanHazeFlags,
)
from spindoctor.feature.geometry import (
    LimbPolyline,
    RingEdgePolyline,
    StarGeometry,
    TerminatorPolyline,
    TitanHazeGeometry,
)
from spindoctor.nav_orchestrator.image_classifier_result import NavImageClassifierResult
from spindoctor.nav_orchestrator.image_derivatives import (
    DEFAULT_IMAGE_GRADIENT_SIGMA_PX,
    build_image_edge_dt,
    compute_image_gradient_vu,
)
from spindoctor.nav_orchestrator.nav_context import NavContext
from spindoctor.nav_orchestrator.provenance import Provenance
from spindoctor.support.filters import NavFilterKind, NavFilterSpec

# Public type aliases for the factory fixtures defined below.  Test
# functions annotate their fixture arguments using these so the test
# tree stays mypy-strict-clean.

DiscImageFactory = Callable[[tuple[int, int], tuple[float, float], float], np.ndarray]
"""Signature of the ``disc_image`` factory fixture."""

HorizontalStepImageFactory = Callable[[tuple[int, int], float], np.ndarray]
"""Signature of the ``horizontal_step_image`` factory fixture."""

HazeDiscImageFactory = Callable[[tuple[int, int], tuple[float, float], float, float], np.ndarray]
"""Signature of the ``haze_disc_image`` factory fixture."""

CirclePolylineFactory = Callable[[tuple[float, float], float, int], tuple[np.ndarray, np.ndarray]]
"""Signature of the ``circle_polyline`` factory fixture."""

ArcPolylineFactory = Callable[
    [tuple[float, float], float, int, float, float], tuple[np.ndarray, np.ndarray]
]
"""Signature of the ``arc_polyline`` factory fixture."""

FlatPolylineFactory = Callable[[float, float, float, int], tuple[np.ndarray, np.ndarray]]
"""Signature of the ``flat_polyline`` factory fixture."""

NavContextFactory = Callable[..., NavContext]
"""Signature of the ``make_nav_context`` factory fixture (kwargs-flexible)."""

NavFeatureFactory = Callable[..., NavFeature]
"""Signature of the ``make_*_feature`` factory fixtures (kwargs-flexible)."""

DrawGaussianStarFactory = Callable[..., None]
"""Signature of the ``draw_gaussian_star`` factory fixture (kwargs-flexible)."""


class FakeObs:
    """Minimal observation stand-in matching the obs.* attribute access used by
    the DT techniques.

    The technique implementations only read ``obs.extfov_margin_vu`` (via the
    shared :func:`spindoctor.nav_technique.nav_technique.search_window_for_obs`
    helper); the rest of the obs surface is irrelevant once a
    fully-populated ``NavContext`` is supplied directly.

    Parameters:
        extfov_margin_vu: ``(margin_v, margin_u)`` extfov margin tuple
            returned to callers; defaults to ``(32, 32)``.
    """

    def __init__(self, extfov_margin_vu: tuple[int, int] = (32, 32)) -> None:
        self.extfov_margin_vu = extfov_margin_vu


# ---------------------------------------------------------------------------
# Image factories
# ---------------------------------------------------------------------------


def _render_disc_image(
    shape: tuple[int, int], center_vu: tuple[float, float], radius: float
) -> np.ndarray:
    """Anti-aliased bright disc on a dark background.

    A 1-pixel-wide brightness ramp at ``radius`` puts the gradient peak at
    exactly the geometric edge so the DT-based fitter can converge below
    0.05 px on noise-free fixtures without integer-grid bias.
    """
    vs, us = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    rr = np.hypot(vs - center_vu[0], us - center_vu[1])
    inside = rr <= radius - 0.5
    outside = rr >= radius + 0.5
    ramp = np.clip(radius + 0.5 - rr, 0.0, 1.0)
    image = np.where(inside, 100.0, np.where(outside, 0.0, 100.0 * ramp))
    return image.astype(np.float64)


def _render_haze_disc_image(
    shape: tuple[int, int],
    center_vu: tuple[float, float],
    r_limb_px: float,
    theta_rad: float,
    peak_dn: float = 1000.0,
) -> np.ndarray:
    """Mirror-symmetric hazy disc with a logistic limb and an along-axis ramp.

    The brightness is ``peak_dn`` times a logistic falloff through
    ``r_limb_px`` with a one-pixel scale length, modulated by a linear
    brightening toward the sub-solar side.  The ramp runs ALONG the symmetry
    axis, so it preserves the mirror symmetry the technique measures while
    still giving the scene a sunward direction.
    """
    vs, us = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    dv = vs - center_vu[0]
    du = us - center_vu[1]
    rho = np.hypot(dv, du)
    along = dv * np.sin(theta_rad) + du * np.cos(theta_rad)
    modulation = 1.0 + 0.25 * along / r_limb_px
    image = peak_dn * modulation / (1.0 + np.exp((rho - r_limb_px) / 1.0))
    rendered: np.ndarray = image.astype(np.float64)
    return rendered


def _render_horizontal_step_image(shape: tuple[int, int], step_v: float) -> np.ndarray:
    """Vertical step: bright above row ``step_v``, dark below.

    The Sobel-of-Gaussian gradient peaks one pixel wide at ``step_v``, giving
    the LM a clean single minimum to converge to.
    """
    vs, _ = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    image = np.where(vs <= step_v, 100.0, 0.0)
    return image.astype(np.float64)


def _draw_gaussian_star(
    image: np.ndarray, center_vu: tuple[float, float], peak_dn: float, sigma: float = 1.0
) -> None:
    """Add a 2-D Gaussian point source to ``image`` in place."""
    h, w = image.shape
    half = max(1, int(np.ceil(3.0 * sigma)))
    cv, cu = center_vu
    v_lo = max(0, int(np.floor(cv - half)))
    v_hi = min(h, int(np.ceil(cv + half)) + 1)
    u_lo = max(0, int(np.floor(cu - half)))
    u_hi = min(w, int(np.ceil(cu + half)) + 1)
    vs = np.arange(v_lo, v_hi, dtype=np.float64)
    us = np.arange(u_lo, u_hi, dtype=np.float64)
    vv, uu = np.meshgrid(vs, us, indexing='ij')
    image[v_lo:v_hi, u_lo:u_hi] += peak_dn * np.exp(
        -((vv - cv) ** 2 + (uu - cu) ** 2) / (2.0 * sigma * sigma)
    )


@pytest.fixture
def disc_image() -> DiscImageFactory:
    """Factory fixture producing anti-aliased bright-disc images."""
    return _render_disc_image


@pytest.fixture
def haze_disc_image() -> HazeDiscImageFactory:
    """Factory fixture producing mirror-symmetric hazy-disc images."""
    return _render_haze_disc_image


@pytest.fixture
def horizontal_step_image() -> HorizontalStepImageFactory:
    """Factory fixture producing horizontal-step images for flat-edge tests."""
    return _render_horizontal_step_image


@pytest.fixture
def draw_gaussian_star() -> DrawGaussianStarFactory:
    """Factory fixture for stamping a 2-D Gaussian PSF onto an image in place.

    Signature: ``draw_gaussian_star(image, center_vu, peak_dn, sigma=1.0)``.
    Mutates ``image`` and returns ``None``.
    """
    return _draw_gaussian_star


# ---------------------------------------------------------------------------
# Polyline factories
# ---------------------------------------------------------------------------


def _circle_polyline(
    center_vu: tuple[float, float], radius: float, n_vertices: int
) -> tuple[np.ndarray, np.ndarray]:
    """Full closed circle: ``(N, 2)`` vertices and ``(N, 2)`` outward normals."""
    angles = np.linspace(0.0, 2.0 * np.pi, n_vertices, endpoint=False)
    vs = center_vu[0] + radius * np.sin(angles)
    us = center_vu[1] + radius * np.cos(angles)
    nv = np.sin(angles)
    nu = np.cos(angles)
    return (
        np.stack([vs, us], axis=-1).astype(np.float64),
        np.stack([nv, nu], axis=-1).astype(np.float64),
    )


def _arc_polyline(
    center_vu: tuple[float, float],
    radius: float,
    n_vertices: int,
    angle_start: float,
    angle_end: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Partial arc between ``angle_start`` and ``angle_end`` radians.

    Useful for half-visible limbs (``[-pi/2, pi/2]``) or terminator arcs on a
    crescent geometry.
    """
    angles = np.linspace(angle_start, angle_end, n_vertices)
    vs = center_vu[0] + radius * np.sin(angles)
    us = center_vu[1] + radius * np.cos(angles)
    nv = np.sin(angles)
    nu = np.cos(angles)
    return (
        np.stack([vs, us], axis=-1).astype(np.float64),
        np.stack([nv, nu], axis=-1).astype(np.float64),
    )


def _flat_polyline(
    v_value: float, u_start: float, u_end: float, n_vertices: int
) -> tuple[np.ndarray, np.ndarray]:
    """Horizontal polyline at row ``v_value`` plus outward radial normal ``+v``."""
    us = np.linspace(u_start, u_end, n_vertices)
    vs = np.full_like(us, v_value)
    nv = np.ones_like(us)
    nu = np.zeros_like(us)
    return (
        np.stack([vs, us], axis=-1).astype(np.float64),
        np.stack([nv, nu], axis=-1).astype(np.float64),
    )


@pytest.fixture
def circle_polyline() -> CirclePolylineFactory:
    """Factory fixture producing full-circle polylines."""
    return _circle_polyline


@pytest.fixture
def arc_polyline() -> ArcPolylineFactory:
    """Factory fixture producing partial-arc polylines."""
    return _arc_polyline


@pytest.fixture
def flat_polyline() -> FlatPolylineFactory:
    """Factory fixture producing horizontal flat-edge polylines."""
    return _flat_polyline


# ---------------------------------------------------------------------------
# NavContext factory
# ---------------------------------------------------------------------------


def _make_nav_context(
    image: np.ndarray,
    *,
    extfov_margin_vu: tuple[int, int] = (32, 32),
    technique_names: tuple[str, ...] = (),
    fit_camera_rotation: bool = False,
    max_rotation_deg: float = 5.0,
) -> NavContext:
    """Build a fully-populated ``NavContext`` from ``image``.

    All masks are trivially populated (full-sensor, no saturation, no cosmic
    rays); the image classifier reports a clean image; provenance is filled
    with deterministic values.  Image gradient / DT derivatives are computed
    via :func:`build_image_edge_dt` + :func:`compute_image_gradient_vu` so
    the techniques' ``image_edge_dt_ext`` / ``image_gradient_vu_ext`` reads
    behave exactly as they would under the orchestrator.  Pass
    ``fit_camera_rotation=True`` to exercise techniques' 3-DoF code path.
    """
    sensor_mask = np.ones(image.shape, dtype=bool)
    saturation_mask = np.zeros(image.shape, dtype=bool)
    cosmic_ray_mask = np.zeros(image.shape, dtype=bool)
    classifier_result = NavImageClassifierResult(
        image_class='clean',
        saturation_frac=0.0,
        missing_frac=0.0,
        noise_sigma=1.0,
        max_dn=float(image.max()),
        flags=[],
    )
    provenance = Provenance(
        spindoctor_version='0.0.0',
        image_et=0.0,
        pipeline_run_iso8601='2026-04-27T00:00:00Z',
        technique_names=technique_names,
        extractor_names=(),
    )
    gradient, edge_dt = build_image_edge_dt(image, image_noise_sigma=1.0)
    gradient_vu = compute_image_gradient_vu(image, sigma_px=DEFAULT_IMAGE_GRADIENT_SIGMA_PX)
    return NavContext(
        obs=FakeObs(extfov_margin_vu=extfov_margin_vu),
        image_ext=image,
        sensor_mask_ext=sensor_mask,
        image_noise_sigma=1.0,
        saturation_mask_ext=saturation_mask,
        cosmic_ray_mask_ext=cosmic_ray_mask,
        image_classifier=classifier_result,
        provenance=provenance,
        image_gradient_ext=gradient,
        image_gradient_vu_ext=gradient_vu,
        image_edge_dt_ext=edge_dt,
        fit_camera_rotation=fit_camera_rotation,
        max_rotation_deg=max_rotation_deg,
    )


@pytest.fixture
def make_nav_context() -> NavContextFactory:
    """Factory fixture returning fully-populated ``NavContext`` objects."""
    return _make_nav_context


# ---------------------------------------------------------------------------
# Feature factories (per-feature-type)
# ---------------------------------------------------------------------------


def _bbox_for_vertices(vertices: np.ndarray) -> tuple[int, int, int, int]:
    """Return a ``(v_min, u_min, v_max, u_max)`` bounding box for a polyline."""
    n = vertices.shape[0]
    if n == 0:
        return (0, 0, 0, 0)
    vmin = int(vertices[:, 0].min())
    vmax = int(vertices[:, 0].max()) + 1
    umin = int(vertices[:, 1].min())
    umax = int(vertices[:, 1].max()) + 1
    return (vmin, umin, vmax, umax)


def _make_limb_feature(
    body_name: str,
    *,
    vertices: np.ndarray,
    outward_normals: np.ndarray,
    sigma_normal_px: float = 0.5,
    visible_arc_fraction: float = 1.0,
) -> NavFeature:
    """Build a ``LIMB_ARC`` ``NavFeature`` from a vertex / outward-normal pair."""
    n = vertices.shape[0]
    return NavFeature(
        feature_id=f'limb_arc:{body_name}',
        feature_type=NavFeatureType.LIMB_ARC,
        source_model='body',
        geometry=LimbPolyline(
            vertices_vu=vertices,
            normals_vu=outward_normals,
            sigma_normal_per_vertex_px=np.full(n, sigma_normal_px, dtype=np.float64),
            sigma_tangent_per_vertex_px=np.full(n, 0.5, dtype=np.float64),
            bbox_extfov_vu=_bbox_for_vertices(vertices),
        ),
        subject_range_km=1.0e6,
        position_cov_px=None,
        intensity_sigma_rel=0.0,
        preferred_filter=NavFilterSpec(kind=NavFilterKind.NONE),
        reliability=0.9,
        reliability_reasons=NavReliabilityBreakdown(visible_arc_fraction=visible_arc_fraction),
        usable_types=frozenset({NavFeatureType.LIMB_ARC}),
        flags=LimbArcFlags(body_name=body_name, visible_arc_fraction=visible_arc_fraction),
    )


def _make_terminator_feature(
    body_name: str,
    *,
    vertices: np.ndarray,
    outward_normals: np.ndarray,
    sigma_normal_px: float = 1.0,
    visible_arc_fraction: float = 1.0,
    phase_angle_factor: float = 0.7,
    albedo_penalty: float = 0.1,
) -> NavFeature:
    """Build a ``TERMINATOR_ARC`` ``NavFeature``."""
    n = vertices.shape[0]
    return NavFeature(
        feature_id=f'terminator_arc:{body_name}',
        feature_type=NavFeatureType.TERMINATOR_ARC,
        source_model='body',
        geometry=TerminatorPolyline(
            vertices_vu=vertices,
            normals_vu=outward_normals,
            sigma_normal_per_vertex_px=np.full(n, sigma_normal_px, dtype=np.float64),
            sigma_tangent_per_vertex_px=np.full(n, 0.5, dtype=np.float64),
            bbox_extfov_vu=_bbox_for_vertices(vertices),
        ),
        subject_range_km=1.0e6,
        position_cov_px=None,
        intensity_sigma_rel=0.0,
        preferred_filter=NavFilterSpec(kind=NavFilterKind.NONE),
        reliability=0.7,
        reliability_reasons=NavReliabilityBreakdown(
            visible_arc_fraction=visible_arc_fraction,
            albedo_penalty=albedo_penalty,
        ),
        usable_types=frozenset({NavFeatureType.TERMINATOR_ARC}),
        flags=TerminatorArcFlags(
            body_name=body_name,
            visible_arc_fraction=visible_arc_fraction,
            phase_angle_factor=phase_angle_factor,
        ),
    )


def _make_ring_feature(
    name: str,
    *,
    vertices: np.ndarray,
    outward_normals: np.ndarray,
    is_straight_line: bool,
    sigma_radial_px: float = 0.5,
    sigma_orbit_radial_px: float = 0.0,
    planet_name: str = 'SATURN',
) -> NavFeature:
    """Build a ``RING_EDGE`` ``NavFeature``."""
    n = vertices.shape[0]
    return NavFeature(
        feature_id=f'ring_edge:{name}',
        feature_type=NavFeatureType.RING_EDGE,
        source_model='rings',
        geometry=RingEdgePolyline(
            vertices_vu=vertices,
            normals_vu=outward_normals,
            sigma_radial_per_vertex_px=np.full(n, sigma_radial_px, dtype=np.float64),
            sigma_along_edge_per_vertex_px=np.full(n, 0.5, dtype=np.float64),
            is_straight_line=is_straight_line,
            bbox_extfov_vu=_bbox_for_vertices(vertices),
            sigma_orbit_radial_px=sigma_orbit_radial_px,
        ),
        subject_range_km=1.0e6,
        position_cov_px=None,
        intensity_sigma_rel=0.0,
        preferred_filter=NavFilterSpec(kind=NavFilterKind.NONE),
        reliability=0.8,
        reliability_reasons=NavReliabilityBreakdown(visible_arc_fraction=1.0),
        usable_types=frozenset({NavFeatureType.RING_EDGE}),
        flags=RingEdgeFlags(
            is_straight_line=is_straight_line,
            polarity_predictable=False,
            edge_name=name,
            planet_name=planet_name,
        ),
    )


def _make_star_feature(
    feature_id: str,
    *,
    predicted_vu: tuple[float, float],
    predicted_snr: float,
    bbox_pad: int = 6,
    in_body_silhouette: bool = False,
    in_saturation_or_cosmic_mask: bool = False,
    vmag: float | None = 5.0,
) -> NavFeature:
    """Build a STAR ``NavFeature`` with the supplied prediction + brightness."""
    pv, pu = predicted_vu
    bbox = (
        int(np.floor(pv - bbox_pad)),
        int(np.floor(pu - bbox_pad)),
        int(np.ceil(pv + bbox_pad)),
        int(np.ceil(pu + bbox_pad)),
    )
    sigma = 0.5
    cov = (sigma * sigma) * np.eye(2, dtype=np.float64)
    return NavFeature(
        feature_id=feature_id,
        feature_type=NavFeatureType.STAR,
        source_model='stars',
        geometry=StarGeometry(
            predicted_vu=predicted_vu,
            catalog_vu=predicted_vu,
            bbox_extfov_vu=bbox,
        ),
        subject_range_km=float('inf'),
        position_cov_px=cov,
        intensity_sigma_rel=0.0,
        preferred_filter=NavFilterSpec(kind=NavFilterKind.NONE),
        reliability=0.95,
        reliability_reasons=NavReliabilityBreakdown(
            predicted_snr=1.0,
            in_body_silhouette=in_body_silhouette,
            in_saturation_or_cosmic=in_saturation_or_cosmic_mask,
            smear_length_ok=True,
        ),
        usable_types=frozenset({NavFeatureType.STAR}),
        flags=StarFlags(
            saturated=False,
            smear_length_px=0.0,
            in_body_silhouette=in_body_silhouette,
            in_saturation_or_cosmic_mask=in_saturation_or_cosmic_mask,
            predicted_snr=predicted_snr,
            vmag=vmag,
        ),
    )


def _make_titan_feature(
    body_name: str = 'TITAN',
    *,
    predicted_center_vu: tuple[float, float],
    r_solid_px: float,
    r_env_px: float,
    sun_angle_rad: float = 0.0,
    axis_degenerate: bool = False,
    phase_deg: float = 30.0,
    km_per_px: float = 20.0,
    contaminant_mask: np.ndarray | None = None,
    filters: tuple[str, ...] = ('CL1', 'CL2'),
    reliability: float = 0.9,
) -> NavFeature:
    """Build a ``TITAN_LIMB`` ``NavFeature`` for a hazy body."""
    pad = int(np.ceil(r_env_px))
    bbox = (
        int(np.floor(predicted_center_vu[0])) - pad,
        int(np.floor(predicted_center_vu[1])) - pad,
        int(np.ceil(predicted_center_vu[0])) + pad + 1,
        int(np.ceil(predicted_center_vu[1])) + pad + 1,
    )
    return NavFeature(
        feature_id=f'titan_limb:{body_name}',
        feature_type=NavFeatureType.TITAN_LIMB,
        source_model=f'titan:{body_name}',
        geometry=TitanHazeGeometry(
            predicted_center_vu=predicted_center_vu,
            sun_angle_rad=sun_angle_rad,
            axis_degenerate=axis_degenerate,
            phase_deg=phase_deg,
            r_solid_px=r_solid_px,
            r_env_px=r_env_px,
            km_per_px=km_per_px,
            contaminant_mask=contaminant_mask,
            filters=filters,
            bbox_extfov_vu=bbox,
        ),
        subject_range_km=1.2e6,
        position_cov_px=None,
        intensity_sigma_rel=0.0,
        preferred_filter=NavFilterSpec(kind=NavFilterKind.NONE),
        reliability=reliability,
        reliability_reasons=NavReliabilityBreakdown(
            titan_envelope_diameter_px=2.0 * r_env_px,
            titan_occluded_fraction=0.0,
        ),
        usable_types=frozenset({NavFeatureType.TITAN_LIMB}),
        flags=TitanHazeFlags(body_name=body_name),
    )


@pytest.fixture
def make_titan_feature() -> NavFeatureFactory:
    """Factory fixture producing ``TITAN_LIMB`` ``NavFeature`` instances."""
    return _make_titan_feature


@pytest.fixture
def make_limb_feature() -> NavFeatureFactory:
    """Factory fixture producing ``LIMB_ARC`` ``NavFeature`` instances."""
    return _make_limb_feature


@pytest.fixture
def make_terminator_feature() -> NavFeatureFactory:
    """Factory fixture producing ``TERMINATOR_ARC`` ``NavFeature`` instances."""
    return _make_terminator_feature


@pytest.fixture
def make_ring_feature() -> NavFeatureFactory:
    """Factory fixture producing ``RING_EDGE`` ``NavFeature`` instances."""
    return _make_ring_feature


@pytest.fixture
def make_star_feature() -> NavFeatureFactory:
    """Factory fixture producing ``STAR`` ``NavFeature`` instances."""
    return _make_star_feature
