"""Unit tests for the limb gradient-peak measurement used by the realism match."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import erfc

from spindoctor.sim.realism import limb_peak
from spindoctor.sim.realism.limb_peak import fit_limb_peak_offset, gradient_peak_positions

_PSF_SIGMA_PX = 0.8


def _vertical_edge(edge_u: float, *, size: int = 40) -> np.ndarray:
    """A bright half-plane left of ``edge_u`` blurred by a Gaussian of 0.8 px."""
    uu = np.broadcast_to(np.arange(size, dtype=np.float64), (size, size))
    return np.asarray(0.5 * erfc((uu - edge_u) / (np.sqrt(2.0) * _PSF_SIGMA_PX)))


def _disc_normals(n: int, start_deg: float, stop_deg: float) -> np.ndarray:
    """Unit outward normals at ``n`` angles spread evenly over an arc."""
    angles = np.radians(np.linspace(start_deg, stop_deg, n))
    return np.column_stack([np.sin(angles), np.cos(angles)])


def test_a_blurred_step_on_a_pixel_center_peaks_at_its_edge() -> None:
    """A blurred step centered on a pixel peaks exactly at the step."""
    image = _vertical_edge(20.0)
    points = np.array([[v, 20.0] for v in range(8, 32, 4)], dtype=np.float64)
    normals = np.tile([0.0, 1.0], (points.shape[0], 1))
    peaks = gradient_peak_positions(image, points, normals)
    assert np.max(np.abs(peaks)) < 1e-6


def test_the_sub_pixel_ripple_averages_to_zero() -> None:
    """Over the edge's sub-pixel phases the peak lands on the step on average."""
    edges = np.arange(20.0, 21.0, 0.05)
    peaks = [
        gradient_peak_positions(_vertical_edge(e), np.array([[20.0, e]]), np.array([[0.0, 1.0]]))[0]
        for e in edges
    ]
    assert float(np.mean(peaks)) == pytest.approx(0.0, abs=1e-3)


def test_a_peak_is_measured_from_the_silhouette_point() -> None:
    """A silhouette point 0.62 px outside the edge puts the peak 0.62 px inward.

    The distance falls between two samples, so the parabola through the samples
    around the steepest one has to place it.
    """
    image = _vertical_edge(20.0)
    points = np.array([[20.0, 20.62]])
    peaks = gradient_peak_positions(image, points, np.array([[0.0, 1.0]]))
    assert peaks[0] == pytest.approx(-0.62, abs=1e-3)


def test_a_track_leaving_the_image_has_no_peak() -> None:
    """A vertex whose sampling track runs off the frame gets NaN."""
    image = _vertical_edge(20.3)
    peaks = gradient_peak_positions(image, np.array([[20.0, 37.0]]), np.array([[0.0, 1.0]]))
    assert np.isnan(peaks[0])


def test_a_frame_with_a_nan_pixel_is_refused() -> None:
    """A NaN pixel far from every track would spread through the spline; it is refused."""
    image = _vertical_edge(20.3)
    image[2, 2] = np.nan
    with pytest.raises(ValueError, match='NaN or infinite pixels'):
        gradient_peak_positions(image, np.array([[20.0, 20.3]]), np.array([[0.0, 1.0]]))


def test_an_inward_brightening_edge_has_no_peak() -> None:
    """A normal pointing from the sky into the disc finds no limb."""
    image = _vertical_edge(20.3)
    peaks = gradient_peak_positions(image, np.array([[20.0, 20.3]]), np.array([[0.0, -1.0]]))
    assert np.isnan(peaks[0])


def test_the_fit_separates_the_radial_offset_from_a_translation() -> None:
    """Peaks made of an offset and a translation give both back."""
    normals = _disc_normals(100, -90.0, 90.0)
    peaks = -0.4 + normals @ np.array([0.7, -1.1])
    fit = fit_limb_peak_offset(peaks, normals)
    assert fit is not None
    assert fit.offset_px == pytest.approx(-0.4, abs=1e-9)
    assert fit.translation_vu == pytest.approx((0.7, -1.1), abs=1e-9)


def test_a_limb_whose_normals_all_point_one_way_gives_no_fit() -> None:
    """Forty peaks along one normal cannot tell the offset from the translation."""
    normals = np.tile([1.0, 0.0], (40, 1))
    assert fit_limb_peak_offset(np.ones(40), normals) is None


def test_the_fit_drops_outlying_peaks() -> None:
    """A few peaks on a crater rim leave the offset where the rest put it."""
    rng = np.random.default_rng(3)
    normals = _disc_normals(120, -90.0, 90.0)
    peaks = np.asarray(rng.normal(-0.25, 0.05, normals.shape[0]), dtype=np.float64)
    peaks[::15] += 2.5
    fit = fit_limb_peak_offset(peaks, normals)
    assert fit is not None
    assert fit.offset_px == pytest.approx(-0.25, abs=0.03)
    assert fit.n_kept == normals.shape[0] - 8


def test_the_fit_needs_twenty_measured_peaks() -> None:
    """Nineteen measured peaks are too few to fit."""
    normals = _disc_normals(30, -90.0, 90.0)
    peaks = np.zeros(30)
    peaks[19:] = np.nan
    assert fit_limb_peak_offset(peaks, normals) is None


def test_the_peak_is_looked_for_up_to_3_px_from_the_silhouette() -> None:
    """An edge 2.5 px inside the silhouette point is found there."""
    image = _vertical_edge(20.0)
    peaks = gradient_peak_positions(image, np.array([[20.0, 22.5]]), np.array([[0.0, 1.0]]))
    assert peaks[0] == pytest.approx(-2.5, abs=1e-3)


def test_a_steepest_fall_at_the_end_of_the_search_range_is_no_peak() -> None:
    """An edge 3.5 px inside the silhouette point, beyond the search, gives NaN."""
    image = _vertical_edge(20.0)
    peaks = gradient_peak_positions(image, np.array([[20.0, 23.5]]), np.array([[0.0, 1.0]]))
    assert np.isnan(peaks[0])


def test_the_fit_needs_twenty_kept_peaks() -> None:
    """Thirty measured peaks of which the robust fit drops twelve are too few."""
    normals = _disc_normals(30, -90.0, 90.0)
    peaks = np.full(30, -0.2)
    peaks[::5] += 3.0
    peaks[1::5] -= 3.0
    peaks[2::5] += 0.01
    assert fit_limb_peak_offset(peaks, normals) is None


def test_the_offset_comes_from_the_final_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    """After one reweighting the offset is the reweighted fit's, below the plain mean.

    Of 1000 peaks around a full circle, 324 lie at 1 px and the rest at 0; the plain
    least-squares offset is their mean, 0.324, and the biweight then weights the peaks
    at 1 px less.
    """
    monkeypatch.setattr(limb_peak, '_TUKEY_ITERATIONS', 1)
    normals = _disc_normals(1000, 0.0, 359.64)
    peaks = np.where(np.arange(1000) % 1000 * 0.324 % 1.0 < 0.324, 1.0, 0.0)
    fit = fit_limb_peak_offset(peaks, normals)
    assert fit is not None
    assert fit.offset_px < 0.31
