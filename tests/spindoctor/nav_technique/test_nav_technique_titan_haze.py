"""Tests for ``spindoctor.nav_technique.nav_technique_titan_haze.TitanHazeNav``.

The technique is a thin wrapper around the pure fitting library, so these
tests exercise the wrapping: feasibility, the config-driven tuning load, the
assembled offset and its anisotropic covariance, the gate-to-spurious
mapping, and the result-level at-edge rule that per-pass flags cannot
express.  Recovery bounds match the fitting library's noisy case (0.15 px
cross-track, 1.5 px along-track) because the rendered scene is the same one.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from tests.spindoctor.nav_technique.conftest import (
    HazeDiscImageFactory,
    NavContextFactory,
    NavFeatureFactory,
)

from spindoctor.feature.feature_type import NavFeatureType
from spindoctor.nav_technique.diagnostics import TitanHazeDiagnostics
from spindoctor.nav_technique.nav_technique import NavTechnique
from spindoctor.nav_technique.nav_technique_titan_haze import (
    TitanHazeNav,
    _arc_gate_rows,
    _symmetry_gate_rows,
)
from spindoctor.nav_technique.technique_result import NavTechniqueResult
from spindoctor.nav_technique.titan_fitting import (
    ArcFitParams,
    ArcFitResult,
    SymmetryFitParams,
    SymmetryFitResult,
    axis_vectors,
)

SHAPE_VU = (170, 170)
CENTER_VU = (85.0, 85.0)
R_LIMB_PX = 44.0
R_SOLID_PX = 40.0
R_ENV_PX = 46.0
WINDOW_PX = 10
CASE2_CROSS_PX = 0.15
CASE2_ALONG_PX = 1.50


def _displaced(offset_vu: tuple[float, float]) -> tuple[float, float]:
    """Return the true disc center for a planted offset from the prediction."""
    return (CENTER_VU[0] + offset_vu[0], CENTER_VU[1] + offset_vu[1])


def _run_technique(
    haze_disc_image: HazeDiscImageFactory,
    make_nav_context: NavContextFactory,
    make_titan_feature: NavFeatureFactory,
    *,
    offset_vu: tuple[float, float],
    theta_rad: float = 0.0,
    window_px: int = WINDOW_PX,
) -> NavTechniqueResult:
    """Render a displaced haze disc and navigate it through the full interface."""
    image = haze_disc_image(SHAPE_VU, _displaced(offset_vu), R_LIMB_PX, theta_rad)
    context = make_nav_context(image, extfov_margin_vu=(window_px, window_px))
    feature = make_titan_feature(
        predicted_center_vu=CENTER_VU,
        r_solid_px=R_SOLID_PX,
        r_env_px=R_ENV_PX,
        sun_angle_rad=theta_rad,
    )
    return TitanHazeNav().navigate([feature], context)


# ---------------------------------------------------------------------------
# Registration and configuration
# ---------------------------------------------------------------------------


def test_technique_is_registered() -> None:
    """The technique self-registers so the orchestrator can discover it."""
    assert TitanHazeNav in NavTechnique._registry


def test_technique_accepts_only_titan_limb() -> None:
    """The technique consumes TITAN_LIMB features and nothing else."""
    assert TitanHazeNav.accepts_feature_types == frozenset({NavFeatureType.TITAN_LIMB})


def test_technique_is_primary_tier() -> None:
    """A hazy body has no second estimator, so the technique is primary."""
    assert TitanHazeNav.tier == 'primary'


def test_technique_runs_without_a_prior() -> None:
    """The technique runs in pass 1."""
    assert TitanHazeNav.requires_prior is False


def test_confidence_spec_loads_from_config() -> None:
    """The confidence spec comes from config_510, not from hardcoded terms."""
    technique = TitanHazeNav()
    assert technique.confidence_spec is not None


def test_confidence_terms_are_declared_attributes() -> None:
    """Every confidence term names an attribute the technique declares."""
    technique = TitanHazeNav()
    assert technique.confidence_spec is not None
    referenced = {term.feature for term in technique.confidence_spec.terms}
    assert referenced <= TitanHazeNav.confidence_attributes


def test_model_error_floor_is_loaded() -> None:
    """The covariance model-error floor is a configured tunable."""
    assert TitanHazeNav.tuning['model_error_floor_px'] > 0.0


# ---------------------------------------------------------------------------
# Feasibility
# ---------------------------------------------------------------------------


def test_is_infeasible_without_titan_features(make_star_feature: NavFeatureFactory) -> None:
    """A frame with no haze feature is infeasible."""
    technique = TitanHazeNav()
    star = make_star_feature('star:test:1', predicted_vu=(10.0, 10.0), predicted_snr=20.0)
    assert technique.is_feasible([star]).feasible is False


def test_infeasibility_reason_names_the_missing_type(
    make_star_feature: NavFeatureFactory,
) -> None:
    """The infeasibility reason names the missing feature type."""
    technique = TitanHazeNav()
    star = make_star_feature('star:test:1', predicted_vu=(10.0, 10.0), predicted_snr=20.0)
    assert technique.is_feasible([star]).reason == 'no TITAN_LIMB features'


def test_is_feasible_with_one_titan_feature(make_titan_feature: NavFeatureFactory) -> None:
    """One haze feature makes the technique feasible."""
    technique = TitanHazeNav()
    feature = make_titan_feature(
        predicted_center_vu=CENTER_VU, r_solid_px=R_SOLID_PX, r_env_px=R_ENV_PX
    )
    assert technique.is_feasible([feature]).feasible is True


def test_feasibility_reports_the_consumed_count(make_titan_feature: NavFeatureFactory) -> None:
    """The feasibility report counts the features it would consume."""
    technique = TitanHazeNav()
    feature = make_titan_feature(
        predicted_center_vu=CENTER_VU, r_solid_px=R_SOLID_PX, r_env_px=R_ENV_PX
    )
    assert technique.is_feasible([feature]).consumed_feature_count == 1


# ---------------------------------------------------------------------------
# End-to-end recovery through the NavTechnique interface
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ('offset_vu', 'theta_rad'),
    [
        ((0.3, -0.4), 0.0),
        ((-1.2, 0.8), 0.7),
        ((2.5, 1.5), 1.9),
    ],
)
def test_planted_offset_recovered_cross_track(
    haze_disc_image: HazeDiscImageFactory,
    make_nav_context: NavContextFactory,
    make_titan_feature: NavFeatureFactory,
    offset_vu: tuple[float, float],
    theta_rad: float,
) -> None:
    """The cross-track component is recovered within the fitting library's bound."""
    result = _run_technique(
        haze_disc_image,
        make_nav_context,
        make_titan_feature,
        offset_vu=offset_vu,
        theta_rad=theta_rad,
    )
    c_hat, _ = axis_vectors(theta_rad)
    error = np.asarray(result.offset_px) - np.asarray(offset_vu)
    assert abs(float(error @ c_hat)) <= CASE2_CROSS_PX


@pytest.mark.parametrize(
    ('offset_vu', 'theta_rad'),
    [
        ((0.3, -0.4), 0.0),
        ((-1.2, 0.8), 0.7),
        ((2.5, 1.5), 1.9),
    ],
)
def test_planted_offset_recovered_along_track(
    haze_disc_image: HazeDiscImageFactory,
    make_nav_context: NavContextFactory,
    make_titan_feature: NavFeatureFactory,
    offset_vu: tuple[float, float],
    theta_rad: float,
) -> None:
    """The along-track component is recovered within the fitting library's bound."""
    result = _run_technique(
        haze_disc_image,
        make_nav_context,
        make_titan_feature,
        offset_vu=offset_vu,
        theta_rad=theta_rad,
    )
    _, a_hat = axis_vectors(theta_rad)
    error = np.asarray(result.offset_px) - np.asarray(offset_vu)
    assert abs(float(error @ a_hat)) <= CASE2_ALONG_PX


def test_clean_scene_is_not_spurious(
    haze_disc_image: HazeDiscImageFactory,
    make_nav_context: NavContextFactory,
    make_titan_feature: NavFeatureFactory,
) -> None:
    """A clean rendered haze disc navigates rather than reporting spurious."""
    result = _run_technique(
        haze_disc_image, make_nav_context, make_titan_feature, offset_vu=(0.3, -0.4)
    )
    assert result.spurious is False


def test_clean_scene_reports_no_failed_gate(
    haze_disc_image: HazeDiscImageFactory,
    make_nav_context: NavContextFactory,
    make_titan_feature: NavFeatureFactory,
) -> None:
    """No gate fires on a clean scene."""
    result = _run_technique(
        haze_disc_image, make_nav_context, make_titan_feature, offset_vu=(0.3, -0.4)
    )
    assert isinstance(result.diagnostics, TitanHazeDiagnostics)
    assert result.diagnostics.gate_failed is None


def test_result_names_the_source_body(
    haze_disc_image: HazeDiscImageFactory,
    make_nav_context: NavContextFactory,
    make_titan_feature: NavFeatureFactory,
) -> None:
    """The result attributes itself to the hazy body it consumed."""
    result = _run_technique(
        haze_disc_image, make_nav_context, make_titan_feature, offset_vu=(0.3, -0.4)
    )
    assert result.source_bodies == frozenset({'TITAN'})


def test_result_reports_no_rotation(
    haze_disc_image: HazeDiscImageFactory,
    make_nav_context: NavContextFactory,
    make_titan_feature: NavFeatureFactory,
) -> None:
    """Rotation is unobservable from one quasi-circular feature."""
    result = _run_technique(
        haze_disc_image, make_nav_context, make_titan_feature, offset_vu=(0.3, -0.4)
    )
    assert result.rotation_rad is None


def test_rotation_fitting_instrument_gets_a_rank_deficient_covariance(
    haze_disc_image: HazeDiscImageFactory,
    make_nav_context: NavContextFactory,
    make_titan_feature: NavFeatureFactory,
) -> None:
    """On a rotation-fitting instrument the covariance is the 3x3 unobservable form.

    Every result in an ensemble group must share the same degrees of
    freedom, so a technique that carries no rotation evidence declares that
    through the sentinel rather than by shipping a smaller matrix.
    """
    image = haze_disc_image(SHAPE_VU, _displaced((0.3, -0.4)), R_LIMB_PX, 0.0)
    context = make_nav_context(
        image, extfov_margin_vu=(WINDOW_PX, WINDOW_PX), fit_camera_rotation=True
    )
    feature = make_titan_feature(
        predicted_center_vu=CENTER_VU, r_solid_px=R_SOLID_PX, r_env_px=R_ENV_PX
    )
    result = TitanHazeNav().navigate([feature], context)
    assert result.covariance_px2.shape == (3, 3)


# ---------------------------------------------------------------------------
# Covariance orientation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('theta_rad', [0.0, 0.7, 1.9, -2.4])
def test_covariance_major_axis_follows_the_symmetry_axis(
    haze_disc_image: HazeDiscImageFactory,
    make_nav_context: NavContextFactory,
    make_titan_feature: NavFeatureFactory,
    theta_rad: float,
) -> None:
    """The largest-variance direction lies along the symmetry axis.

    The mirror-symmetry scan localizes across the axis far better than the
    limb-arc fit localizes along it, so the error ellipse must be elongated
    along ``a_hat`` -- within 5 degrees of it.
    """
    result = _run_technique(
        haze_disc_image,
        make_nav_context,
        make_titan_feature,
        offset_vu=(0.3, -0.4),
        theta_rad=theta_rad,
    )
    eigenvalues, eigenvectors = np.linalg.eigh(np.asarray(result.covariance_px2)[:2, :2])
    major = eigenvectors[:, int(np.argmax(eigenvalues))]
    _, a_hat = axis_vectors(theta_rad)
    cosine = abs(float(major @ a_hat))
    assert math.degrees(math.acos(min(cosine, 1.0))) <= 5.0


def test_covariance_is_anisotropic(
    haze_disc_image: HazeDiscImageFactory,
    make_nav_context: NavContextFactory,
    make_titan_feature: NavFeatureFactory,
) -> None:
    """Along-track uncertainty genuinely exceeds cross-track uncertainty."""
    result = _run_technique(
        haze_disc_image, make_nav_context, make_titan_feature, offset_vu=(0.3, -0.4)
    )
    eigenvalues = np.linalg.eigvalsh(np.asarray(result.covariance_px2)[:2, :2])
    assert float(eigenvalues[1]) > float(eigenvalues[0])


# ---------------------------------------------------------------------------
# Gates and edge flags
# ---------------------------------------------------------------------------


def test_noise_only_scene_is_spurious(
    make_nav_context: NavContextFactory,
    make_titan_feature: NavFeatureFactory,
) -> None:
    """A frame with no hazy body at all is rejected, not fitted."""
    rng = np.random.default_rng(seed=7)
    image = rng.standard_normal(SHAPE_VU) * 10.0 + 100.0
    context = make_nav_context(image, extfov_margin_vu=(WINDOW_PX, WINDOW_PX))
    feature = make_titan_feature(
        predicted_center_vu=CENTER_VU, r_solid_px=R_SOLID_PX, r_env_px=R_ENV_PX
    )
    result = TitanHazeNav().navigate([feature], context)
    assert result.spurious is True


def test_noise_only_scene_names_a_gate(
    make_nav_context: NavContextFactory,
    make_titan_feature: NavFeatureFactory,
) -> None:
    """The rejection is attributed to a specific named fit gate."""
    rng = np.random.default_rng(seed=7)
    image = rng.standard_normal(SHAPE_VU) * 10.0 + 100.0
    context = make_nav_context(image, extfov_margin_vu=(WINDOW_PX, WINDOW_PX))
    feature = make_titan_feature(
        predicted_center_vu=CENTER_VU, r_solid_px=R_SOLID_PX, r_env_px=R_ENV_PX
    )
    result = TitanHazeNav().navigate([feature], context)
    assert isinstance(result.diagnostics, TitanHazeDiagnostics)
    assert result.diagnostics.gate_failed in {
        'valid_fraction',
        'peak_score',
        'second_peak',
        'ray_yield',
        'arc_inliers',
        'arc_radius',
        'arc_residual',
    }


def test_spurious_result_carries_zero_confidence(
    make_nav_context: NavContextFactory,
    make_titan_feature: NavFeatureFactory,
) -> None:
    """A rejected fit contributes no confidence to the ensemble."""
    rng = np.random.default_rng(seed=7)
    image = rng.standard_normal(SHAPE_VU) * 10.0 + 100.0
    context = make_nav_context(image, extfov_margin_vu=(WINDOW_PX, WINDOW_PX))
    feature = make_titan_feature(
        predicted_center_vu=CENTER_VU, r_solid_px=R_SOLID_PX, r_env_px=R_ENV_PX
    )
    result = TitanHazeNav().navigate([feature], context)
    assert result.confidence == 0.0


_AT_EDGE_WINDOW_PX = 8
"""Search half-window for the at-edge cases."""

_AT_EDGE_OVERSHOOT = 1.3
"""Multiple of the search window the at-edge cases plant the body at.

Comfortably past the bound rather than exactly on it, so the assertion
cannot be satisfied by a fraction of a pixel of fit overshoot.
"""


def _at_edge_result(
    haze_disc_image: HazeDiscImageFactory,
    make_nav_context: NavContextFactory,
    make_titan_feature: NavFeatureFactory,
) -> NavTechniqueResult:
    """Navigate a body planted well beyond the search window along the axis."""
    _, a_hat = axis_vectors(0.0)
    reach = _AT_EDGE_OVERSHOOT * _AT_EDGE_WINDOW_PX
    offset_vu = (float(reach * a_hat[0]), float(reach * a_hat[1]))
    return _run_technique(
        haze_disc_image,
        make_nav_context,
        make_titan_feature,
        offset_vu=offset_vu,
        window_px=_AT_EDGE_WINDOW_PX,
    )


def test_offset_beyond_the_window_sets_at_edge(
    haze_disc_image: HazeDiscImageFactory,
    make_nav_context: NavContextFactory,
    make_titan_feature: NavFeatureFactory,
) -> None:
    """An assembled component past the search window flags the result.

    Each pass gates its own component, so a total beyond the declared
    search bound has to be flagged at the result level for the ensemble's
    conservative at-edge handling to apply.
    """
    assert _at_edge_result(haze_disc_image, make_nav_context, make_titan_feature).at_edge is True


def test_at_edge_result_carries_zero_confidence(
    haze_disc_image: HazeDiscImageFactory,
    make_nav_context: NavContextFactory,
    make_titan_feature: NavFeatureFactory,
) -> None:
    """An at-edge fit contributes no confidence, matching every other technique.

    The offset may be reporting the search bound rather than the body, so
    the confidence spec's hard-zero gate refuses to price it at all.
    """
    result = _at_edge_result(haze_disc_image, make_nav_context, make_titan_feature)
    assert result.confidence == 0.0


def test_at_edge_result_is_still_committed(
    haze_disc_image: HazeDiscImageFactory,
    make_nav_context: NavContextFactory,
    make_titan_feature: NavFeatureFactory,
) -> None:
    """An at-edge fit is flagged, not discarded: the ensemble decides its fate."""
    assert _at_edge_result(haze_disc_image, make_nav_context, make_titan_feature).spurious is False


# ---------------------------------------------------------------------------
# Diagnostics payload
# ---------------------------------------------------------------------------


def test_diagnostics_record_the_fitted_haze_radius(
    haze_disc_image: HazeDiscImageFactory,
    make_nav_context: NavContextFactory,
    make_titan_feature: NavFeatureFactory,
) -> None:
    """The fitted radius is reported in kilometers for haze-table accumulation.

    The feature's scale is 20 km/px and the rendered limb sits at 44 px, so
    the fitted radius should land near 880 km.
    """
    result = _run_technique(
        haze_disc_image, make_nav_context, make_titan_feature, offset_vu=(0.3, -0.4)
    )
    assert isinstance(result.diagnostics, TitanHazeDiagnostics)
    assert result.diagnostics.fitted_haze_radius_km == pytest.approx(880.0, abs=40.0)


def test_diagnostics_record_the_envelope_diameter(
    haze_disc_image: HazeDiscImageFactory,
    make_nav_context: NavContextFactory,
    make_titan_feature: NavFeatureFactory,
) -> None:
    """The envelope diameter travels from the feature into the diagnostics."""
    result = _run_technique(
        haze_disc_image, make_nav_context, make_titan_feature, offset_vu=(0.3, -0.4)
    )
    assert isinstance(result.diagnostics, TitanHazeDiagnostics)
    assert result.diagnostics.envelope_diameter_px == pytest.approx(2.0 * R_ENV_PX)


def test_diagnostics_record_the_filters(
    haze_disc_image: HazeDiscImageFactory,
    make_nav_context: NavContextFactory,
    make_titan_feature: NavFeatureFactory,
) -> None:
    """Filter names ride into the diagnostics for filter-dependent analysis."""
    result = _run_technique(
        haze_disc_image, make_nav_context, make_titan_feature, offset_vu=(0.3, -0.4)
    )
    assert isinstance(result.diagnostics, TitanHazeDiagnostics)
    assert result.diagnostics.filters == ('CL1', 'CL2')


def test_diagnostics_split_the_offset_onto_the_axis(
    haze_disc_image: HazeDiscImageFactory,
    make_nav_context: NavContextFactory,
    make_titan_feature: NavFeatureFactory,
) -> None:
    """The reported cross / along components reconstruct the offset."""
    theta = 0.7
    result = _run_technique(
        haze_disc_image,
        make_nav_context,
        make_titan_feature,
        offset_vu=(-1.2, 0.8),
        theta_rad=theta,
    )
    assert isinstance(result.diagnostics, TitanHazeDiagnostics)
    c_hat, a_hat = axis_vectors(math.radians(result.diagnostics.sun_angle_deg))
    rebuilt = result.diagnostics.cross_track_px * c_hat + result.diagnostics.along_track_px * a_hat
    assert float(np.max(np.abs(rebuilt - np.asarray(result.offset_px)))) < 1.0e-9


def test_degenerate_axis_suppresses_angle_refinement(
    haze_disc_image: HazeDiscImageFactory,
    make_nav_context: NavContextFactory,
    make_titan_feature: NavFeatureFactory,
) -> None:
    """A degenerate axis leaves the reported angle exactly where the model put it.

    Any axis is equally valid on a rotationally symmetric disc, so refining
    it would only chase noise.
    """
    image = haze_disc_image(SHAPE_VU, _displaced((0.3, -0.4)), R_LIMB_PX, 0.0)
    context = make_nav_context(image, extfov_margin_vu=(WINDOW_PX, WINDOW_PX))
    feature = make_titan_feature(
        predicted_center_vu=CENTER_VU,
        r_solid_px=R_SOLID_PX,
        r_env_px=R_ENV_PX,
        sun_angle_rad=0.0,
        axis_degenerate=True,
    )
    result = TitanHazeNav().navigate([feature], context)
    assert isinstance(result.diagnostics, TitanHazeDiagnostics)
    assert result.diagnostics.theta_refined_deg == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Gate table
# ---------------------------------------------------------------------------


_GATE_NAMES = (
    'valid_fraction',
    'peak_score',
    'second_peak',
    'cross_at_edge',
    'ray_yield',
    'arc_inliers',
    'arc_radius',
    'arc_residual',
    'along_at_edge',
)
"""Every gate the technique reports on, in the order the fits evaluate them."""


def _gate_table_rows(captured: str) -> dict[str, list[str]]:
    """Return the gate table's rows, split into fields and keyed by gate name.

    A pdslogger line carries its timestamp and level ahead of the message,
    so the message is what follows the last separator; a table row is a
    message whose first field is a gate name.
    """
    rows: dict[str, list[str]] = {}
    for line in captured.splitlines():
        fields = line.split('|')[-1].split()
        if fields and fields[0] in _GATE_NAMES:
            rows[fields[0]] = fields
    return rows


def _navigate_clean(
    haze_disc_image: HazeDiscImageFactory,
    make_nav_context: NavContextFactory,
    make_titan_feature: NavFeatureFactory,
) -> None:
    """Navigate a clean rendered scene, discarding the result."""
    _run_technique(haze_disc_image, make_nav_context, make_titan_feature, offset_vu=(0.3, -0.4))


def test_the_predicted_centre_is_logged_in_the_image_frame(
    haze_disc_image: HazeDiscImageFactory,
    make_nav_context: NavContextFactory,
    make_titan_feature: NavFeatureFactory,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The logged prediction names a pixel of the image, not of the padded array.

    The disc is drawn on the center of row 85 and column 85 of the padded
    array, which sits ``WINDOW_PX`` rows and columns of padding in from the
    image's own first row and column, so it is the center of image row 75.  A
    position stated to a person names a row's center by that row's number plus
    a half, so the line has to read 75.50.
    """
    _run_technique(haze_disc_image, make_nav_context, make_titan_feature, offset_vu=(0.0, 0.0))
    assert 'predicted center (75.50, 75.50)' in capsys.readouterr().out


def test_gate_table_is_logged_inside_the_technique_section(
    haze_disc_image: HazeDiscImageFactory,
    make_nav_context: NavContextFactory,
    make_titan_feature: NavFeatureFactory,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The table is emitted, so a per-image log explains every accept or reject."""
    _navigate_clean(haze_disc_image, make_nav_context, make_titan_feature)
    assert 'Gate table (final pass):' in capsys.readouterr().out


def test_gate_table_is_inside_the_named_technique_section(
    haze_disc_image: HazeDiscImageFactory,
    make_nav_context: NavContextFactory,
    make_titan_feature: NavFeatureFactory,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The table belongs to this technique's own log section."""
    _navigate_clean(haze_disc_image, make_nav_context, make_titan_feature)
    assert 'TECHNIQUE: TitanHazeNav' in capsys.readouterr().out


@pytest.mark.parametrize('gate_name', _GATE_NAMES)
def test_gate_table_reports_every_gate(
    haze_disc_image: HazeDiscImageFactory,
    make_nav_context: NavContextFactory,
    make_titan_feature: NavFeatureFactory,
    capsys: pytest.CaptureFixture[str],
    gate_name: str,
) -> None:
    """Every Section-2.2 and Section-2.3 gate gets its own line."""
    _navigate_clean(haze_disc_image, make_nav_context, make_titan_feature)
    assert gate_name in _gate_table_rows(capsys.readouterr().out)


def test_gate_table_reports_the_configured_threshold(
    haze_disc_image: HazeDiscImageFactory,
    make_nav_context: NavContextFactory,
    make_titan_feature: NavFeatureFactory,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Each row carries the threshold its measurement was compared against."""
    _navigate_clean(haze_disc_image, make_nav_context, make_titan_feature)
    row = _gate_table_rows(capsys.readouterr().out)['peak_score']
    assert row[2:4] == ['>=', '0.6000']


def test_gate_table_reports_the_measured_value(
    haze_disc_image: HazeDiscImageFactory,
    make_nav_context: NavContextFactory,
    make_titan_feature: NavFeatureFactory,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Each row carries the measurement, not just the verdict.

    A clean rendered disc correlates almost perfectly with its mirror, so
    the peak score is printed at four decimals somewhere above 0.9.
    """
    _navigate_clean(haze_disc_image, make_nav_context, make_titan_feature)
    row = _gate_table_rows(capsys.readouterr().out)['peak_score']
    assert float(row[1]) > 0.9


def test_gate_table_passes_every_gate_on_a_clean_scene(
    haze_disc_image: HazeDiscImageFactory,
    make_nav_context: NavContextFactory,
    make_titan_feature: NavFeatureFactory,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """No row reports a failure when the fit succeeds."""
    _navigate_clean(haze_disc_image, make_nav_context, make_titan_feature)
    rows = _gate_table_rows(capsys.readouterr().out)
    assert [name for name, row in rows.items() if row[-1] != 'PASS'] == []


def test_gate_table_marks_the_gate_that_rejected_the_frame(
    make_nav_context: NavContextFactory,
    make_titan_feature: NavFeatureFactory,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The gate named in the diagnostics is the row marked FAIL."""
    rng = np.random.default_rng(seed=7)
    image = rng.standard_normal(SHAPE_VU) * 10.0 + 100.0
    context = make_nav_context(image, extfov_margin_vu=(WINDOW_PX, WINDOW_PX))
    feature = make_titan_feature(
        predicted_center_vu=CENTER_VU, r_solid_px=R_SOLID_PX, r_env_px=R_ENV_PX
    )
    result = TitanHazeNav().navigate([feature], context)
    assert isinstance(result.diagnostics, TitanHazeDiagnostics)
    assert result.diagnostics.gate_failed is not None
    rows = _gate_table_rows(capsys.readouterr().out)
    assert rows[result.diagnostics.gate_failed][-1] == 'FAIL'


def test_gate_table_marks_arc_gates_the_fit_never_reached(
    make_nav_context: NavContextFactory,
    make_titan_feature: NavFeatureFactory,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Gates behind an early arc return are skipped, not reported as passing.

    A featureless frame yields no usable limb rays at all, so the arc fit
    returns before it has an inlier count, a radius, or a residual; printing
    those defaults as verdicts would invent evidence.
    """
    context = make_nav_context(np.zeros(SHAPE_VU), extfov_margin_vu=(WINDOW_PX, WINDOW_PX))
    feature = make_titan_feature(
        predicted_center_vu=CENTER_VU, r_solid_px=R_SOLID_PX, r_env_px=R_ENV_PX
    )
    result = TitanHazeNav().navigate([feature], context)
    assert isinstance(result.diagnostics, TitanHazeDiagnostics)
    rows = _gate_table_rows(capsys.readouterr().out)
    assert rows['arc_radius'][-1] == 'SKIP'


# ---------------------------------------------------------------------------
# Gate-table rows (no image, no oops)
# ---------------------------------------------------------------------------


_ROW_SYM_PARAMS = SymmetryFitParams(
    annulus_inner_fraction=0.55,
    annulus_outer_pad_px=6.0,
    angle_refine_deg=5.0,
    angle_refine_step_deg=0.5,
    angle_refine_min_gain=0.02,
    min_peak_score=0.60,
    min_valid_fraction=0.50,
    max_second_peak_ratio=0.90,
    cross_sigma_scale=1.0,
    sigma_floor_cross_px=0.30,
)
"""Cross-track tuning the row tests compare against, pinned independently of config."""

_ROW_ARC_PARAMS = ArcFitParams(
    sector_half_angle_deg=60.0,
    ray_step_deg=2.0,
    radial_step_px=0.5,
    radial_inner_fraction=0.80,
    radial_outer_pad_px=6.0,
    median_filter_samples=5,
    min_gradient_snr=4.0,
    min_rays=20,
    min_inlier_fraction=0.50,
    max_residual_rms_px=2.0,
    tukey_c=4.685,
    along_sigma_scale=1.0,
    sigma_floor_along_px=1.00,
)
"""Along-track tuning the row tests compare against, pinned independently of config."""


def _symmetry_result(**overrides: object) -> SymmetryFitResult:
    """Build a cross-track result that passes every gate, then apply overrides."""
    fields: dict[str, object] = {
        'cross_track_px': 0.3,
        'sigma_cross_px': 0.3,
        'theta_rad': 0.0,
        'peak_score': 0.99,
        'valid_fraction': 0.99,
        'second_peak_ratio': 0.0,
        'at_edge': False,
        'gate_failed': None,
    }
    fields.update(overrides)
    return SymmetryFitResult(**fields)  # type: ignore[arg-type]


def _arc_result(**overrides: object) -> ArcFitResult:
    """Build an along-track result that passes every gate, then apply overrides."""
    fields: dict[str, object] = {
        'along_track_px': 0.4,
        'sigma_along_px': 1.0,
        'radius_px': 44.0,
        'n_rays_total': 59,
        'n_rays_inlier': 59,
        'residual_rms_px': 0.01,
        'at_edge': False,
        'gate_failed': None,
    }
    fields.update(overrides)
    return ArcFitResult(**fields)  # type: ignore[arg-type]


def _verdicts(symmetry: SymmetryFitResult, arc: ArcFitResult) -> dict[str, str]:
    """Return every gate-table row's verdict, keyed by gate name."""
    rows = _symmetry_gate_rows(symmetry, params=_ROW_SYM_PARAMS, window_px=float(WINDOW_PX))
    rows += _arc_gate_rows(
        arc,
        params=_ROW_ARC_PARAMS,
        r_solid_px=R_SOLID_PX,
        r_env_px=R_ENV_PX,
        window_px=float(WINDOW_PX),
    )
    return {name: verdict for name, _measured, _threshold, verdict in rows}


_FAILING_FITS: list[tuple[str, SymmetryFitResult, ArcFitResult]] = [
    (
        'valid_fraction',
        _symmetry_result(valid_fraction=0.40, gate_failed='valid_fraction'),
        _arc_result(),
    ),
    ('peak_score', _symmetry_result(peak_score=0.50, gate_failed='peak_score'), _arc_result()),
    (
        'second_peak',
        _symmetry_result(second_peak_ratio=0.95, gate_failed='second_peak'),
        _arc_result(),
    ),
    (
        'ray_yield',
        _symmetry_result(),
        _arc_result(
            n_rays_total=5,
            n_rays_inlier=0,
            radius_px=0.0,
            residual_rms_px=0.0,
            sigma_along_px=float(WINDOW_PX),
            along_track_px=0.0,
            gate_failed='ray_yield',
        ),
    ),
    ('arc_inliers', _symmetry_result(), _arc_result(n_rays_inlier=15, gate_failed='arc_inliers')),
    ('arc_radius', _symmetry_result(), _arc_result(radius_px=5.0, gate_failed='arc_radius')),
    (
        'arc_residual',
        _symmetry_result(),
        _arc_result(residual_rms_px=4.0, gate_failed='arc_residual'),
    ),
]
"""One fit per named gate, tripping exactly that gate the way the library does.

The pair carries the ``gate_failed`` the fitting library would report, so
each case also pins the table's row for that name to ``FAIL`` -- the
property Phase E triage reads the table for.
"""

_NAN_FITS: list[tuple[str, SymmetryFitResult, ArcFitResult]] = [
    ('peak_score', _symmetry_result(peak_score=float('nan')), _arc_result()),
    ('arc_residual', _symmetry_result(), _arc_result(residual_rms_px=float('nan'))),
]
"""Fits whose measurement is undefined; the gates negate their comparisons
so an unmeasurable quantity fails rather than slipping past, and the table
must land on the same side.
"""


@pytest.mark.parametrize(
    ('gate_name', 'symmetry', 'arc'), _FAILING_FITS, ids=[case[0] for case in _FAILING_FITS]
)
def test_gate_row_marks_the_tripped_gate_failed(
    gate_name: str, symmetry: SymmetryFitResult, arc: ArcFitResult
) -> None:
    """The row the fitting library names as the failure reads FAIL."""
    assert _verdicts(symmetry, arc)[gate_name] == 'FAIL'


@pytest.mark.parametrize(
    ('gate_name', 'symmetry', 'arc'), _FAILING_FITS, ids=[case[0] for case in _FAILING_FITS]
)
def test_gate_row_marks_only_the_tripped_gate_failed(
    gate_name: str, symmetry: SymmetryFitResult, arc: ArcFitResult
) -> None:
    """No other row reads FAIL, so a flipped comparison cannot hide behind one.

    A swapped threshold or an inverted predicate on any row would show up
    here as a second failure on a fit that only tripped one gate.
    """
    verdicts = _verdicts(symmetry, arc)
    assert sorted(name for name, verdict in verdicts.items() if verdict == 'FAIL') == [gate_name]


@pytest.mark.parametrize(
    ('gate_name', 'symmetry', 'arc'), _NAN_FITS, ids=[f'{case[0]}_nan' for case in _NAN_FITS]
)
def test_gate_row_fails_an_unmeasurable_quantity(
    gate_name: str, symmetry: SymmetryFitResult, arc: ArcFitResult
) -> None:
    """A NaN measurement fails its gate rather than passing it."""
    assert _verdicts(symmetry, arc)[gate_name] == 'FAIL'


def test_gate_rows_all_pass_for_a_clean_fit() -> None:
    """A fit that tripped nothing reads PASS on every row."""
    verdicts = _verdicts(_symmetry_result(), _arc_result())
    assert sorted(set(verdicts.values())) == ['PASS']


def test_gate_rows_cover_every_named_gate() -> None:
    """The table reports every Section-2.2 and Section-2.3 gate, and no others."""
    assert sorted(_verdicts(_symmetry_result(), _arc_result())) == sorted(_GATE_NAMES)


def test_cross_at_edge_row_reports_edge() -> None:
    """A cross-track peak on the window boundary is flagged, not failed."""
    verdicts = _verdicts(_symmetry_result(at_edge=True), _arc_result())
    assert verdicts['cross_at_edge'] == 'EDGE'


def test_cross_at_edge_is_not_a_gate_failure() -> None:
    """An at-edge cross-track fit leaves every row's verdict non-FAIL."""
    verdicts = _verdicts(_symmetry_result(at_edge=True), _arc_result())
    assert [name for name, verdict in verdicts.items() if verdict == 'FAIL'] == []


def test_along_at_edge_row_reports_edge() -> None:
    """An along-track shift reaching the window bound is flagged, not failed."""
    verdicts = _verdicts(_symmetry_result(), _arc_result(at_edge=True))
    assert verdicts['along_at_edge'] == 'EDGE'


def test_along_at_edge_is_not_a_gate_failure() -> None:
    """An at-edge along-track fit leaves every row's verdict non-FAIL."""
    verdicts = _verdicts(_symmetry_result(), _arc_result(at_edge=True))
    assert [name for name, verdict in verdicts.items() if verdict == 'FAIL'] == []


@pytest.mark.parametrize(
    'gate_name', ['arc_inliers', 'arc_radius', 'arc_residual', 'along_at_edge']
)
def test_gate_rows_after_an_early_arc_return_are_skipped(gate_name: str) -> None:
    """Gates behind the ray-yield return report SKIP, not a defaulted verdict."""
    _name, symmetry, arc = _FAILING_FITS[3]
    assert _verdicts(symmetry, arc)[gate_name] == 'SKIP'
