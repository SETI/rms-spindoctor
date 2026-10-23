"""The LM's normal-projected covariance: a vertex informs only its own normal axis."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.ndimage import distance_transform_edt

from spindoctor.nav_technique.dt_fitting import LMRefineResult, lm_subpixel_refine


def _diagonal_line(angle_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A rasterized straight edge whose normal makes ``angle_deg`` with +v.

    Returns the edge's distance transform, model vertices sampled every pixel along the
    true line, and the line's unit normal at each vertex.  The mask is the one-pixel
    rasterization of the line, so its distance transform is a staircase.
    """
    shape = (200, 200)
    normal = np.array([math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))])
    tangent = np.array([-normal[1], normal[0]])
    vs, us = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    distance = (vs - 100.0) * normal[0] + (us - 100.0) * normal[1]
    mask = np.abs(distance) <= 0.5
    edge_dt = distance_transform_edt(~mask).astype(np.float64)
    steps = np.arange(-60.0, 60.0, 1.0)
    vertices = np.array([100.0, 100.0]) + steps[:, None] * tangent[None, :]
    normals = np.tile(normal, (steps.shape[0], 1))
    return edge_dt, vertices, normals


def _eigenvalue_ratio(covariance: np.ndarray) -> float:
    """Smallest over largest eigenvalue of the translation block."""
    eigvals = np.linalg.eigvalsh(covariance[:2, :2])
    return float(eigvals[0] / eigvals[-1])


def _refine(
    edge_dt: np.ndarray, vertices: np.ndarray, normals: np.ndarray, *, project: bool
) -> LMRefineResult:
    """Refine a model laid 0.3 px off the line, with or without the projection."""
    return lm_subpixel_refine(
        vertices_vu=vertices + 0.3 * normals,
        normals_vu=normals,
        sigma_normal_per_vertex_px=np.full(vertices.shape[0], 0.5),
        image_edge_dt=edge_dt,
        use_polarity=False,
        covariance_normals_vu=normals if project else None,
    )


def test_projected_covariance_leaves_a_straight_staircase_edge_rank_1() -> None:
    """Projected onto the line normal, a staircase edge gives no along-edge information."""
    edge_dt, vertices, normals = _diagonal_line(30.0)
    result = _refine(edge_dt, vertices, normals, project=True)
    assert _eigenvalue_ratio(result.covariance) < 1.0e-8


def test_projected_covariance_leaves_the_fit_unchanged() -> None:
    """The projection changes only the reported covariance, never the converged pose."""
    edge_dt, vertices, normals = _diagonal_line(30.0)
    projected = _refine(edge_dt, vertices, normals, project=True)
    plain = _refine(edge_dt, vertices, normals, project=False)
    assert projected.offset_vu == plain.offset_vu


def test_projected_covariance_rejects_mismatched_normals() -> None:
    """A normal array that does not match the vertices fails before fitting."""
    edge_dt, vertices, normals = _diagonal_line(30.0)
    with pytest.raises(ValueError, match='covariance_normals_vu must match vertices_vu shape'):
        lm_subpixel_refine(
            vertices_vu=vertices,
            normals_vu=normals,
            sigma_normal_per_vertex_px=np.full(vertices.shape[0], 0.5),
            image_edge_dt=edge_dt,
            use_polarity=False,
            covariance_normals_vu=normals[:-1],
        )
