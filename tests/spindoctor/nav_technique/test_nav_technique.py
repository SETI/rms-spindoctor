"""Tests for ``spindoctor.nav_technique.nav_technique.NavTechnique``."""

from typing import Any

import numpy as np
import pytest

# Side-effect imports — register the shipped techniques in
# ``NavTechnique._registry`` so the validator has a non-empty input.
import spindoctor.nav_technique.nav_technique_body_limb
import spindoctor.nav_technique.nav_technique_body_terminator
import spindoctor.nav_technique.nav_technique_ring_edge  # noqa: F401
from spindoctor.feature.feature import NavFeature
from spindoctor.feature.feature_type import NavFeatureType
from spindoctor.nav_technique.confidence import ConfidenceSpec, ConfidenceTerm
from spindoctor.nav_technique.diagnostics import BodyLimbDiagnostics
from spindoctor.nav_technique.feasibility import NavFeasibilityReport
from spindoctor.nav_technique.nav_technique import (
    NavTechnique,
    add_model_error_floor,
    add_size_scaled_model_error,
    filter_technique_names,
    load_model_error_floor,
    load_ncc_covariance_tuning,
    validate_registered_confidence_specs,
)
from spindoctor.nav_technique.technique_result import NavTechniqueResult


class _ConcreteTechniqueForTest(NavTechnique):
    """Concrete subclass for direct-invocation testing.

    ``_abstract = True`` keeps it out of the process-wide registry so it
    cannot leak into full-ensemble navigations run later in the same
    worker; the registration behavior itself is exercised by
    ``test_navtechnique_registry_records_subclass`` with a scoped probe.
    """

    name = '_ConcreteTechniqueForTest'
    _abstract = True
    accepts_feature_types = frozenset({NavFeatureType.STAR})

    def is_feasible(self, features: list[NavFeature]) -> NavFeasibilityReport:
        return NavFeasibilityReport(feasible=False, reason='test_only')

    def navigate(self, features: list[NavFeature], context: Any) -> NavTechniqueResult:
        return NavTechniqueResult(
            technique_name=self.name,
            feature_ids=(),
            offset_px=(0.0, 0.0),
            covariance_px2=np.eye(2, dtype=np.float64),
            confidence=0.0,
            spurious=False,
            at_edge=False,
            diagnostics=BodyLimbDiagnostics(),
        )


def test_navtechnique_registry_records_subclass() -> None:
    """Concrete subclasses self-register via __init_subclass__."""

    class _RegistrationProbe(_ConcreteTechniqueForTest):
        name = '_RegistrationProbe'
        _abstract = False

    try:
        assert _RegistrationProbe in NavTechnique._registry
    finally:
        NavTechnique._registry.remove(_RegistrationProbe)


def test_navtechnique_abstract_subclass_stays_out_of_registry() -> None:
    """``_abstract = True`` opts a subclass out of the registry entirely."""
    assert _ConcreteTechniqueForTest not in NavTechnique._registry


def test_registry_contains_only_shipped_techniques() -> None:
    """No test-defined technique leaks into the process-wide registry.

    A leaked fake runs inside every later full-ensemble navigation in
    the same process and fuses its hardcoded offset into real results —
    a worker-colocation heisenbug under pytest-xdist.  Test fakes must
    set ``_abstract = True`` and register through a scoped fixture.
    """
    for cls in NavTechnique._registry:
        assert cls.__module__.startswith('spindoctor.'), (
            f'{cls.__module__}.{cls.__qualname__} is registered in '
            f'NavTechnique._registry but is not a shipped technique; test fakes '
            f'must set _abstract = True and use a scoped registration fixture'
        )


def test_navtechnique_can_invoke_navigate() -> None:
    """A concrete subclass's navigate returns a NavTechniqueResult."""
    technique = _ConcreteTechniqueForTest()
    result = technique.navigate([], context=None)
    assert isinstance(result, NavTechniqueResult)


def test_filter_technique_names_inclusion() -> None:
    """Inclusion patterns admit matching names only."""
    names = ['BodyLimbNav', 'BodyDiscCorrelateNav', 'StarRefineNav']
    out = filter_technique_names(names, ['Body*'])
    assert out == ['BodyLimbNav', 'BodyDiscCorrelateNav']


def test_filter_technique_names_exclusion() -> None:
    """Leading-bang patterns exclude matches."""
    names = ['BodyLimbNav', 'BodyDiscCorrelateNav', 'StarRefineNav']
    out = filter_technique_names(names, ['*', '!Body*'])
    assert out == ['StarRefineNav']


def test_filter_technique_names_exclusion_only_implies_star_include() -> None:
    """A pure-exclusion pattern list implies '*' as the include."""
    names = ['BodyLimbNav', 'StarRefineNav']
    out = filter_technique_names(names, ['!StarRefineNav'])
    assert out == ['BodyLimbNav']


def test_filter_technique_names_default_star() -> None:
    """Default '*' pattern keeps every name."""
    names = ['A', 'B', 'C']
    out = filter_technique_names(names, '*')
    assert out == ['A', 'B', 'C']


def test_filter_technique_names_rejects_empty_patterns() -> None:
    """An empty list raises ValueError."""
    with pytest.raises(ValueError, match='at least one'):
        filter_technique_names(['A'], [])


def test_validate_registered_confidence_specs_passes_for_shipped_techniques() -> None:
    """Every shipped technique's spec only references declared attributes."""
    validate_registered_confidence_specs()


def test_validate_registered_confidence_specs_rejects_unknown_attribute() -> None:
    """A spec referencing an attribute outside confidence_attributes raises."""

    class _BadConfidenceTechnique(NavTechnique):
        name = '_BadConfidenceTechnique'
        accepts_feature_types = frozenset({NavFeatureType.STAR})
        confidence_spec = ConfidenceSpec(
            alpha0=0.0,
            terms=(ConfidenceTerm(feature='nope_undeclared', alpha=1.0),),
        )
        confidence_attributes = frozenset({'at_edge'})

        def is_feasible(self, features: list[NavFeature]) -> NavFeasibilityReport:
            return NavFeasibilityReport(feasible=False, reason='test_only')

        def navigate(self, features: list[NavFeature], context: Any) -> NavTechniqueResult:
            return NavTechniqueResult(
                technique_name=self.name,
                feature_ids=(),
                offset_px=(0.0, 0.0),
                covariance_px2=np.eye(2, dtype=np.float64),
                confidence=0.0,
                spurious=False,
                at_edge=False,
                diagnostics=BodyLimbDiagnostics(),
            )

    try:
        with pytest.raises(ValueError, match='nope_undeclared'):
            validate_registered_confidence_specs()
    finally:
        NavTechnique._registry.remove(_BadConfidenceTechnique)


# --- model-error floor helpers (#210) ---


def test_load_model_error_floor_defaults_to_disabled() -> None:
    """A tuning block without the key yields a disabled (0.0) floor."""
    assert load_model_error_floor({}, 'SomeNav') == 0.0


def test_load_model_error_floor_accepts_positive_value() -> None:
    """A configured positive floor is returned unchanged."""
    assert load_model_error_floor({'model_error_floor_px': 0.92}, 'SomeNav') == 0.92


@pytest.mark.parametrize('bad', [-0.5, float('nan'), float('inf')])
def test_load_model_error_floor_rejects_invalid_values(bad: float) -> None:
    """Negative and non-finite floors raise instead of silently misbehaving."""
    with pytest.raises(ValueError, match='model_error_floor_px must be finite'):
        load_model_error_floor({'model_error_floor_px': bad}, 'SomeNav')


def test_add_model_error_floor_adds_in_quadrature() -> None:
    """The floor squares onto the translation diagonal without mutating the input."""
    cov = np.array([[1.0, 0.1], [0.1, 4.0]])
    out = add_model_error_floor(cov, 2.0)
    assert out[0, 0] == pytest.approx(5.0)
    assert out[1, 1] == pytest.approx(8.0)
    assert out[0, 1] == pytest.approx(0.1)
    assert cov[0, 0] == pytest.approx(1.0)


def test_add_model_error_floor_zero_is_identity() -> None:
    """A disabled floor returns the input covariance object unchanged."""
    cov = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert add_model_error_floor(cov, 0.0) is cov


def test_add_size_scaled_model_error_adds_size_and_floor_in_quadrature() -> None:
    """The size term and floor both square onto the translation diagonal."""
    cov = np.array([[1.0, 0.1], [0.1, 4.0]])
    out = add_size_scaled_model_error(cov, size_px=100.0, size_frac=0.02, floor_px=0.5)
    extra = (0.02 * 100.0) ** 2 + 0.5**2  # 4.0 + 0.25 = 4.25
    assert out[0, 0] == pytest.approx(1.0 + extra)
    assert out[1, 1] == pytest.approx(4.0 + extra)
    assert out[0, 1] == pytest.approx(0.1)


def test_add_size_scaled_model_error_zero_terms_is_identity() -> None:
    """Both terms disabled returns the input covariance object unchanged."""
    cov = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert add_size_scaled_model_error(cov, size_px=100.0, size_frac=0.0, floor_px=0.0) is cov


def test_load_ncc_covariance_tuning_reads_all_terms() -> None:
    """All three covariance terms are read from the tuning mapping."""
    tuning = {
        'localization_uncertainty_scale': 1.0,
        'model_error_size_frac': 0.005,
        'model_error_floor_px': 0.05,
    }
    out = load_ncc_covariance_tuning(tuning, 'SomeNav')
    assert out.localization_uncertainty_scale == pytest.approx(1.0)
    assert out.model_error_size_frac == pytest.approx(0.005)
    assert out.model_error_floor_px == pytest.approx(0.05)


def test_load_ncc_covariance_tuning_defaults_disabled() -> None:
    """Absent keys default every covariance term to disabled (0.0)."""
    out = load_ncc_covariance_tuning({}, 'SomeNav')
    assert out.localization_uncertainty_scale == 0.0
    assert out.model_error_size_frac == 0.0
    assert out.model_error_floor_px == 0.0


@pytest.mark.parametrize('bad', [-1.0, float('nan')])
def test_load_ncc_covariance_tuning_rejects_invalid_size_frac(bad: float) -> None:
    """A negative or non-finite size fraction is rejected."""
    with pytest.raises(ValueError, match='model_error_size_frac'):
        load_ncc_covariance_tuning({'model_error_size_frac': bad}, 'SomeNav')
