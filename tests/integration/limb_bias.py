"""Instrumented harness for measuring the body-limb navigation bias.

The limb navigator (``BodyLimbNav``) recovers a translational offset by
fitting a body's predicted silhouette polyline to the image edge distance
transform.  A systematic sub-pixel error in that fit does not average out
across frames, so it sets the accuracy floor for every limb-navigated
image.  This module measures that error under controlled conditions.

Two independent ground-truth channels are supported:

* Simulator planted truth.  A sim scene renders a body at
  ``center + planted_offset``; the simulated body NavModel predicts the
  unshifted geometry, so the navigator should recover ``planted_offset``
  exactly.  The simulator sets the spacecraft position, the body ephemeris,
  and the pointing by construction, so a residual limb-fit error on a sim
  scene carries no spacecraft-position and no body-ephemeris component.
  What it does carry is the limb fit read against the scene description the
  renderer and the model both work from: it is a difference between two
  sides of one description, so a coordinate convention the two apply alike
  cancels exactly and cannot show up here.
  :func:`measure_sim_limb_bias` returns the signed per-axis error against
  that planted truth.

* Real-frame star navigation.  On a real frame that also carries several
  navigable stars, the star techniques provide an independent, usually more
  precise, offset.  The limb-minus-star gap on a real frame mixes the
  limb-fit residual with any spacecraft-position or body-ephemeris error,
  so it must be read together with the sim measurement above.
  :func:`measure_real_limb_vs_star` returns both offsets and their signed
  gap.

A third function, :func:`renderer_centroid_offset`, measures the simulator's
own body renderer against the center it was asked for, which anchors the
renderer side on its own and needs no agreement from the model side.  It
calls the renderer directly and never touches the navigation code path.

Offset convention (matches the rest of the pipeline): a predicted position
``(v, u)`` means the actual position is ``(v + dv, u + du)``, so the signed
limb-fit error is ``recovered_offset - planted_offset``.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from spindoctor.nav_model import build_models_for_obs
from spindoctor.nav_orchestrator import NavOrchestrator
from spindoctor.obs.obs_inst_sim import ObsSim
from spindoctor.sim.forward.body import create_simulated_body

__all__ = [
    'LimbBiasSample',
    'RealLimbStarSample',
    'RendererCentroidCheck',
    'build_body_scene',
    'measure_real_limb_vs_star',
    'measure_sim_limb_bias',
    'renderer_centroid_offset',
    'ridge_inset_phase_zero',
]

# Star technique names used as the independent real-frame ground truth,
# most precise first.  The first one that produces a non-spurious result on a
# frame is taken as that frame's star reference offset.
_STAR_TECHNIQUES: tuple[str, ...] = (
    'StarRefineNav',
    'StarUniqueMatchNav',
    'StarFieldFromCatalogNav',
)


@dataclass(frozen=True)
class LimbBiasSample:
    """One signed limb-fit error measurement against sim planted truth.

    Attributes:
        planted_offset_vu: The planted ``(dv, du)`` the navigator should
            recover.
        recovered_offset_vu: The ``BodyLimbNav`` offset, or ``None`` when the
            technique produced no usable result.
        error_vu: Signed ``recovered - planted`` per axis, or ``None``.
        spurious: Whether the technique flagged its own result spurious.
    """

    planted_offset_vu: tuple[float, float]
    recovered_offset_vu: tuple[float, float] | None
    error_vu: tuple[float, float] | None
    spurious: bool

    @property
    def error_mag_px(self) -> float | None:
        """Euclidean magnitude of the signed error, or ``None``."""
        if self.error_vu is None:
            return None
        return math.hypot(self.error_vu[0], self.error_vu[1])


@dataclass(frozen=True)
class RealLimbStarSample:
    """Limb-versus-star offset comparison on a single real frame.

    Attributes:
        image_id: Short image identifier.
        limb_offset_vu: ``BodyLimbNav`` offset, or ``None``.
        star_offset_vu: Independent star-technique offset, or ``None``.
        star_technique: Name of the star technique that supplied the
            reference, or ``None``.
        gap_vu: Signed ``limb - star`` per axis, or ``None`` when either
            offset is missing.
    """

    image_id: str
    limb_offset_vu: tuple[float, float] | None
    star_offset_vu: tuple[float, float] | None
    star_technique: str | None
    gap_vu: tuple[float, float] | None

    @property
    def gap_mag_px(self) -> float | None:
        """Euclidean magnitude of the limb-minus-star gap, or ``None``."""
        if self.gap_vu is None:
            return None
        return math.hypot(self.gap_vu[0], self.gap_vu[1])


@dataclass(frozen=True)
class RendererCentroidCheck:
    """Result of the simulator body-renderer geometry validation.

    Attributes:
        requested_center_vu: Sub-pixel body center requested from the
            renderer, in the renderer's pixel corner coordinates.
        geometric_center_vu: The same center in the pixel centric coordinates
            the navigator measures in (``requested - 0.5`` on each axis).
        measured_centroid_vu: The intensity-weighted centroid of the
            rendered body, pixel centric.
        centroid_error_vu: Signed ``measured - geometric`` per axis.
    """

    requested_center_vu: tuple[float, float]
    geometric_center_vu: tuple[float, float]
    measured_centroid_vu: tuple[float, float]
    centroid_error_vu: tuple[float, float]

    @property
    def centroid_error_mag_px(self) -> float:
        """Euclidean magnitude of the centroid error."""
        return math.hypot(self.centroid_error_vu[0], self.centroid_error_vu[1])


def build_body_scene(
    *,
    diameter_px: float,
    phase_deg: float,
    illumination_deg: float,
    offset_vu: tuple[float, float],
    size_px: int = 260,
    noise: bool = False,
) -> dict[str, Any]:
    """Build a single-sphere sim scene as a flat ``sim_params`` mapping.

    The scene is a well-resolved, centered sphere on black sky with a planted
    offset, sized with room to spare so the whole limb stays on the frame
    under the offset.  Noise is off by default so the measured limb-fit error
    is deterministic rather than a per-frame noise draw.

    Parameters:
        diameter_px: Full body diameter in pixels (all three axes equal, so
            the silhouette is a circle).
        phase_deg: Phase angle in degrees; 0 is fully lit, higher values
            leave a lit crescent and a soft terminator.
        illumination_deg: In-plane light direction in degrees; 0 lights the
            top of the image, 90 the right.
        offset_vu: Planted ``(dv, du)`` pointing offset the navigator should
            recover.
        size_px: Square image side in pixels.
        noise: When ``True`` apply the default detector noise; when ``False``
            render a noise-free frame.

    Returns:
        A flat ``sim_params`` mapping ready for :class:`ObsSim`.
    """
    center = size_px / 2.0
    scene: dict[str, Any] = {
        'schema_version': 2,
        'scene_name': 'limb_bias_probe',
        'instrument': 'coiss_nac',
        'size_v': size_px,
        'size_u': size_px,
        'random_seed': 7,
        'exposure_sec': 1.0,
        'bodies': [
            {
                'name': 'RHEA',
                'shape_model': 'ellipsoid',
                'center_v': center,
                'center_u': center,
                'axis1': diameter_px,
                'axis2': diameter_px,
                'axis3': diameter_px,
                'illumination_angle': illumination_deg,
                'phase_angle': phase_deg,
            }
        ],
        'offset_v': offset_vu[0],
        'offset_u': offset_vu[1],
        'offset_rotation_deg': 0.0,
    }
    scene['noise'] = (
        {'poisson': True, 'read_noise_dn': 4.0}
        if noise
        else {'poisson': False, 'read_noise_dn': 0.0}
    )
    return scene


def _navigate(sim_params: dict[str, Any], only_techniques: str) -> Any:
    """Navigate an in-memory sim scene, returning the fused ``NavResult``."""
    obs = ObsSim.from_file('sim://limb_bias', sim_params=sim_params)
    orchestrator = NavOrchestrator(
        build_models_for_obs(obs), only_models='*', only_techniques=only_techniques
    )
    return orchestrator.navigate(obs)


def _technique_offset(result: Any, technique: str) -> tuple[float, float] | None:
    """Return the non-spurious offset for ``technique``, or ``None``."""
    for entry in result.per_technique:
        if entry.technique_name == technique and not entry.spurious:
            return (float(entry.offset_px[0]), float(entry.offset_px[1]))
    return None


def _technique_is_spurious(result: Any, technique: str) -> bool:
    """Return whether ``technique`` produced a result and flagged it spurious."""
    for entry in result.per_technique:
        if entry.technique_name == technique:
            return bool(entry.spurious)
    return False


def measure_sim_limb_bias(sim_params: dict[str, Any]) -> LimbBiasSample:
    """Navigate one sim scene with ``BodyLimbNav`` and record the signed error.

    Parameters:
        sim_params: A flat sim scene mapping, e.g. from
            :func:`build_body_scene`.

    Returns:
        A :class:`LimbBiasSample` with the signed per-axis error against the
        scene's planted offset.
    """
    planted = (float(sim_params.get('offset_v', 0.0)), float(sim_params.get('offset_u', 0.0)))
    result = _navigate(sim_params, 'BodyLimbNav')
    recovered = _technique_offset(result, 'BodyLimbNav')
    spurious = _technique_is_spurious(result, 'BodyLimbNav')
    if recovered is None:
        return LimbBiasSample(
            planted_offset_vu=planted,
            recovered_offset_vu=None,
            error_vu=None,
            spurious=spurious,
        )
    error = (recovered[0] - planted[0], recovered[1] - planted[1])
    return LimbBiasSample(
        planted_offset_vu=planted,
        recovered_offset_vu=recovered,
        error_vu=error,
        spurious=spurious,
    )


def measure_real_limb_vs_star(obs: Any, image_id: str) -> RealLimbStarSample:
    """Compare the limb and star offsets on one navigated real observation.

    Runs the full technique ensemble once and reads the ``BodyLimbNav``
    offset and the most precise available star-technique offset from the same
    navigation, so both are measured against the identical image geometry.

    Parameters:
        obs: A real :class:`ObsSnapshotInst` already built from an image.
        image_id: Short identifier for reporting.

    Returns:
        A :class:`RealLimbStarSample` carrying both offsets and their signed
        gap.
    """
    result = NavOrchestrator(
        build_models_for_obs(obs), only_models='*', only_techniques='*'
    ).navigate(obs)
    limb = _technique_offset(result, 'BodyLimbNav')
    star: tuple[float, float] | None = None
    star_name: str | None = None
    for name in _STAR_TECHNIQUES:
        star = _technique_offset(result, name)
        if star is not None:
            star_name = name
            break
    gap: tuple[float, float] | None = None
    if limb is not None and star is not None:
        gap = (limb[0] - star[0], limb[1] - star[1])
    return RealLimbStarSample(
        image_id=image_id,
        limb_offset_vu=limb,
        star_offset_vu=star,
        star_technique=star_name,
        gap_vu=gap,
    )


def renderer_centroid_offset(
    *,
    center_vu: tuple[float, float],
    diameter_px: float,
    size_px: int = 200,
) -> RendererCentroidCheck:
    """Validate the sim body renderer's sub-pixel placement, nav-code-free.

    Renders a fully-lit (phase 0) sphere directly through
    :func:`create_simulated_body` at a requested sub-pixel center and measures
    the intensity-weighted centroid of the result.  A phase-0 sphere is
    radially symmetric, so its brightness centroid must coincide with its
    geometric center; any offset would be a positional bias baked into the
    renderer itself.  The renderer works in pixel corner coordinates, where a
    pixel's center sits half a pixel past its array row, so a requested center
    ``c`` puts the geometric center at ``c - 0.5`` pixel centric; the check
    compares against that.

    Parameters:
        center_vu: Requested body center in the renderer's pixel corner
            coordinates.
        diameter_px: Body diameter in pixels.
        size_px: Square image side in pixels.

    Returns:
        A :class:`RendererCentroidCheck` with the signed centroid error in
        pixel centric coordinates.
    """
    img = create_simulated_body(
        size=(size_px, size_px),
        center=center_vu,
        axis1=diameter_px,
        axis2=diameter_px,
        axis3=diameter_px,
        illumination_angle=0.0,
        phase_angle=0.0,
        anti_aliasing=1,
    )
    total = float(img.sum())
    vs, us = np.mgrid[0:size_px, 0:size_px].astype(np.float64)
    centroid_v = float((vs * img).sum() / total)
    centroid_u = float((us * img).sum() / total)
    geometric = (center_vu[0] - 0.5, center_vu[1] - 0.5)
    error = (centroid_v - geometric[0], centroid_u - geometric[1])
    return RendererCentroidCheck(
        requested_center_vu=center_vu,
        geometric_center_vu=geometric,
        measured_centroid_vu=(centroid_v, centroid_u),
        centroid_error_vu=error,
    )


def ridge_inset_phase_zero(*, diameter_px: float, size_px: int = 260) -> float:
    """Measure how far the brightness gradient ridge sits inside the limb.

    Renders a fully-lit sphere and, along a horizontal scan through the
    center, finds the radius of the peak brightness gradient magnitude and
    compares it to the geometric limb radius.  A positive return value means
    the steepest-slope point (which the edge distance transform localizes)
    lies inside the true silhouette boundary -- the photometric roll-off
    signature that biases the limb fit.  This is a diagnostic measurement, not
    a navigation call.

    Parameters:
        diameter_px: Body diameter in pixels.
        size_px: Square image side in pixels.

    Returns:
        The inset in pixels (geometric radius minus ridge radius) on the lit
        limb.
    """
    center = size_px / 2.0
    img = create_simulated_body(
        size=(size_px, size_px),
        center=(center, center),
        axis1=diameter_px,
        axis2=diameter_px,
        axis3=diameter_px,
        illumination_angle=90.0,
        phase_angle=0.0,
        anti_aliasing=1,
    )
    row = img[round(center - 0.5)].astype(np.float64)
    grad = np.abs(np.gradient(row))
    center_centric = center - 0.5
    right = np.arange(row.shape[0]) > center_centric
    ridge_idx = int(np.argmax(np.where(right, grad, 0.0)))
    ridge_radius = float(ridge_idx) - center_centric
    geometric_radius = diameter_px / 2.0
    return geometric_radius - ridge_radius


def sweep_scenes(
    axis: str,
    values: list[float],
    *,
    base: dict[str, Any],
) -> list[tuple[float, dict[str, Any]]]:
    """Return ``(value, sim_params)`` pairs varying one axis of a base scene.

    Parameters:
        axis: One of ``'phase'``, ``'diameter'``, ``'illumination'``,
            ``'offset_v'``, or ``'offset_u'``.
        values: The values to sweep.
        base: A base scene from :func:`build_body_scene`.

    Returns:
        One ``(value, sim_params)`` pair per value.

    Raises:
        ValueError: If ``axis`` is not a recognized sweep axis.
    """
    out: list[tuple[float, dict[str, Any]]] = []
    for value in values:
        scene = copy.deepcopy(base)
        body = scene['bodies'][0]
        if axis == 'phase':
            body['phase_angle'] = value
        elif axis == 'diameter':
            body['axis1'] = body['axis2'] = body['axis3'] = value
        elif axis == 'illumination':
            body['illumination_angle'] = value
        elif axis == 'offset_v':
            scene['offset_v'] = value
        elif axis == 'offset_u':
            scene['offset_u'] = value
        else:
            raise ValueError(f'unknown sweep axis: {axis!r}')
        out.append((value, scene))
    return out
