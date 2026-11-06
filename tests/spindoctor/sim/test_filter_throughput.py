"""The scene ``detector.filter`` key: one measured throughput sets the signal scale.

A filter combination's catalog throughput is the electrons per second one native
pixel collects at I/F 1.0.  Naming it sets the raw DN of an extended source, the
calibrated product's DN-to-I/F divisor, and the star brightness (relative to the
camera's clear filter) together, so a calibrated product renders the same I/F
through any filter.  A scene without a filter keeps its catalog scale, and a
filter the camera does not catalog is refused at validation and at render.
"""

from typing import Any

import numpy as np
import pytest

from spindoctor.sim.forward.detector.params import (
    DetectorParamError,
    calibration_map,
    resolve_detector_params,
)
from spindoctor.sim.forward.filter_catalog import (
    FILTER_THROUGHPUT_E_PER_S_PER_IF,
    FilterCatalogError,
    canonical_filter,
    filter_star_flux_scale,
    resolve_filter_throughput,
)
from spindoctor.sim.render import render_combined_model
from spindoctor.sim.scene import validate_sim_params
from spindoctor.sim.scene_schema import SimSceneValidationError

_NAC_CLEAR = FILTER_THROUGHPUT_E_PER_S_PER_IF['coiss_nac']['CL1+CL2']
_NAC_GREEN = FILTER_THROUGHPUT_E_PER_S_PER_IF['coiss_nac']['CL1+GRN']
_QUIET = {'poisson': False, 'read_noise_dn': 0.0, 'bias_dn': 0.0}


def _scene(
    instrument: str, *, detector: dict[str, Any] | None = None, **extra: Any
) -> dict[str, Any]:
    """A noise-free 64 x 64 scene on ``instrument``."""
    scene: dict[str, Any] = {
        'size_v': 64,
        'size_u': 64,
        'random_seed': 5,
        'instrument': instrument,
        'exposure_sec': 0.01,
        'noise': dict(_QUIET),
    }
    if detector is not None:
        scene['detector'] = detector
    scene.update(extra)
    return scene


def _body(**overrides: Any) -> dict[str, Any]:
    """A full-phase sphere filling half the frame."""
    body = {
        'name': 'MIMAS',
        'center_v': 32.0,
        'center_u': 32.0,
        'axis1': 32.0,
        'axis2': 32.0,
        'axis3': 32.0,
        'illumination_angle': 0.0,
        'phase_angle': 0.0,
    }
    body.update(overrides)
    return body


# ---- The catalog ----


@pytest.mark.parametrize(
    ('instrument', 'written', 'canonical'),
    [
        ('coiss_nac', 'CL1+CL2', 'CL1+CL2'),
        ('coiss_nac', ' p120 + grn ', 'CL1+GRN'),
        ('coiss_nac', 'IRP0+CB3', 'CL1+CB3'),
        ('coiss_wac', 'CB2+IRP90', 'CB2+CL2'),
        ('coiss_calib_wac', 'MT2+IRP0', 'MT2+CL2'),
    ],
    ids=['clear', 'nac-polarizer-lowercase', 'nac-ir-polarizer', 'wac-polarizer', 'calib-alias'],
)
def test_a_polarizer_reads_as_the_clear_element_of_its_wheel(
    instrument: str, written: str, canonical: str
) -> None:
    """Spelling is folded to upper case and a polarizer to its wheel's clear element."""
    assert canonical_filter(instrument, written) == canonical


def test_a_calibrated_camera_shares_its_raw_camera_table() -> None:
    """The CALIB product's throughput is the raw camera's."""
    assert resolve_filter_throughput('coiss_calib_nac', 'CL1+GRN') == _NAC_GREEN


def test_an_uncataloged_filter_is_refused_with_the_available_ones() -> None:
    """A combination outside the table names the table's combinations."""
    with pytest.raises(FilterCatalogError, match=r"filter 'CL1\+XYZ' is not cataloged.*CL1\+CL2"):
        resolve_filter_throughput('coiss_nac', 'CL1+XYZ')


def test_a_camera_without_a_table_refuses_every_filter() -> None:
    """LORRI has no filter wheel, so it has no table."""
    with pytest.raises(FilterCatalogError, match=r"'nhlorri' has no filter table"):
        resolve_filter_throughput('nhlorri', 'CLEAR')


@pytest.mark.parametrize(
    ('instrument', 'filter_name', 'scale'),
    [
        ('coiss_nac', None, 1.0),
        ('coiss_nac', 'CL1+CL2', 1.0),
        ('coiss_nac', 'CL1+GRN', _NAC_GREEN / _NAC_CLEAR),
        ('coiss_wac', 'CL1+RED', 5.76e6 / 1.35e7),
    ],
    ids=['unnamed', 'clear', 'green', 'wac-red'],
)
def test_the_star_scale_is_the_throughput_over_the_clear_throughput(
    instrument: str, filter_name: str | None, scale: float
) -> None:
    """The star zero point refers to the clear filter."""
    assert filter_star_flux_scale(instrument, filter_name) == pytest.approx(scale)


# ---- The detector parameters ----


def test_an_unnamed_filter_keeps_the_catalog_reference_exposure() -> None:
    """A scene without a filter resolves the scale it always had."""
    assert resolve_detector_params(_scene('coiss_nac')).exposure_ref_sec == 1.0


@pytest.mark.parametrize(
    ('instrument', 'filter_name', 'binning'),
    [
        ('coiss_calib_nac', 'CL1+GRN', 1),
        ('coiss_calib_nac', 'CL1+GRN', 2),
        ('coiss_calib_wac', 'CL1+VIO', 1),
    ],
    ids=['nac', 'nac-sum2', 'wac'],
)
def test_a_filter_sets_the_calibrated_divisor_from_its_throughput(
    instrument: str, filter_name: str, binning: int
) -> None:
    """DN per I/F is throughput / gain x exposure x the binned pixel count."""
    dp = resolve_detector_params(
        _scene(instrument, detector={'filter': filter_name, 'binning': binning})
    )
    cal = calibration_map(dp)
    assert cal is not None
    throughput = resolve_filter_throughput(instrument, filter_name)
    expected = throughput / dp.gain_e_per_dn * 0.01 * binning**2
    assert cal.dn_per_if == pytest.approx(expected)


def test_a_filter_and_a_reference_exposure_are_refused_together_at_render() -> None:
    """Both keys set the signal scale, so the resolver refuses the pair."""
    scene = _scene('coiss_nac', detector={'filter': 'CL1+CL2', 'exposure_ref_sec': 1.0})
    with pytest.raises(DetectorParamError, match='set one of them'):
        resolve_detector_params(scene)


def test_a_filter_with_the_vidicon_model_is_refused_at_render() -> None:
    """The vidicon DN path has no reference exposure for a filter to set."""
    scene = _scene('coiss_nac', detector={'filter': 'CL1+CL2', 'detector_model': 'vidicon'})
    with pytest.raises(DetectorParamError, match='vidicon DN path'):
        resolve_detector_params(scene)


def test_an_uncataloged_filter_is_refused_at_render() -> None:
    """The resolver refuses a filter the camera does not catalog."""
    with pytest.raises(DetectorParamError, match="'gossi' has no filter table"):
        resolve_detector_params(_scene('gossi', detector={'filter': 'GRN'}))


# ---- Validation ----


@pytest.mark.parametrize(
    ('instrument', 'detector', 'message'),
    [
        ('coiss_nac', {'filter': 'CL1+XYZ'}, r"detector\.filter: filter 'CL1\+XYZ' is not"),
        ('vgiss', {'filter': 'CLEAR'}, r"detector\.filter: sim instrument 'vgiss' has no"),
        ('coiss_nac', {'filter': 3}, r'detector\.filter must be a string'),
        (
            'coiss_nac',
            {'filter': 'CL1+CL2', 'exposure_ref_sec': 2.0},
            r'detector\.filter and detector\.exposure_ref_sec both set',
        ),
        (
            'coiss_nac',
            {'filter': 'CL1+CL2', 'detector_model': 'vidicon'},
            r'detector\.filter sets the CCD signal scale',
        ),
    ],
    ids=['uncataloged', 'no-table', 'not-a-string', 'with-exposure-ref', 'with-vidicon'],
)
def test_the_validator_refuses_a_bad_filter(
    instrument: str, detector: dict[str, Any], message: str
) -> None:
    """Each bad filter fails validation with a message naming the problem."""
    with pytest.raises(SimSceneValidationError, match=message):
        validate_sim_params(_scene(instrument, detector=detector))


def test_the_validator_accepts_a_cataloged_filter() -> None:
    """A cataloged combination, in any spelling the catalog folds, validates."""
    validate_sim_params(_scene('coiss_calib_wac', detector={'filter': 'cl1+irp0'}))


# ---- The render ----


def test_raw_dn_scale_with_the_filter_throughput() -> None:
    """An unsaturated body's total raw DN scales by the throughput ratio."""
    clear, _ = render_combined_model(
        _scene('coiss_nac', detector={'filter': 'CL1+CL2'}, bodies=[_body()])
    )
    green, _ = render_combined_model(
        _scene('coiss_nac', detector={'filter': 'CL1+GRN'}, bodies=[_body()])
    )
    assert float(clear.max()) < 4000.0
    ratio = float(green.sum()) / float(clear.sum())
    assert ratio == pytest.approx(_NAC_GREEN / _NAC_CLEAR, rel=0.01)


def test_a_calibrated_product_renders_the_same_if_through_any_filter() -> None:
    """The divisor follows the throughput, so I/F agrees to the quantization step."""
    clear, _ = render_combined_model(
        _scene('coiss_calib_nac', detector={'filter': 'CL1+CL2'}, bodies=[_body()])
    )
    green, _ = render_combined_model(
        _scene('coiss_calib_nac', detector={'filter': 'CL1+GRN'}, bodies=[_body()])
    )
    # One green-filter DN in I/F bounds the two products' rounding.
    step_if = 30.0 / (_NAC_GREEN * 0.01)
    assert float(np.max(np.abs(green - clear))) <= step_if


def test_a_star_dims_by_the_throughput_ratio() -> None:
    """A star's total DN through green is the clear total times the throughput ratio."""
    star = [{'name': 'S', 'v': 32.0, 'u': 32.0, 'vmag': 4.0}]
    clear, _ = render_combined_model(
        _scene('coiss_nac', detector={'filter': 'CL1+CL2'}, stars=star, exposure_sec=0.1)
    )
    green, _ = render_combined_model(
        _scene('coiss_nac', detector={'filter': 'CL1+GRN'}, stars=star, exposure_sec=0.1)
    )
    ratio = float(green.sum()) / float(clear.sum())
    assert ratio == pytest.approx(_NAC_GREEN / _NAC_CLEAR, rel=0.01)


def test_a_clear_filter_star_matches_an_unnamed_filter_star() -> None:
    """The zero point refers to the clear filter, so naming it leaves stars unchanged."""
    star = [{'name': 'S', 'v': 32.0, 'u': 32.0, 'vmag': 4.0}]
    unnamed, _ = render_combined_model(_scene('coiss_nac', stars=star, exposure_sec=0.1))
    clear, _ = render_combined_model(
        _scene('coiss_nac', detector={'filter': 'CL1+CL2'}, stars=star, exposure_sec=0.1)
    )
    assert np.array_equal(unnamed, clear)
