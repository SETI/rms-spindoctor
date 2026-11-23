"""The quantization sub-modes and the companding tables ``sqrt_lut`` decodes through."""

import numpy as np
import pytest
from tests.spindoctor.sim.scenes import disc_scene

from spindoctor.sim.instruments.resolvers import resolve_downlink_table
from spindoctor.sim.instruments.schema import DownlinkTable
from spindoctor.sim.render.detector.params import resolve_detector_params
from spindoctor.sim.render.detector.quantization import quantize_dn
from spindoctor.sim.render.detector.sqrt_lut import LUTSqrt, resolve_sqrt_lut
from spindoctor.sim.render.driver import render_scene

_CASSINI_LUT = resolve_sqrt_lut(resolve_downlink_table('coiss_nac', None), saturation_dn=4095.0)
_CASSINI_DN = _CASSINI_LUT.levels()


def test_exact_quantization_rounds_to_integer() -> None:
    """The exact sub-mode rounds DN to the nearest integer."""
    dn = np.array([[10.4, 10.6]], dtype=np.float64)
    out = quantize_dn(dn, mode='exact', saturation_dn=4095.0)
    assert np.array_equal(out, np.array([[10.0, 11.0]]))


def test_8bit_quantization_rounds_and_clips_at_255() -> None:
    """The 8-bit sub-mode rounds to integer DN and clips at the 255 code ceiling."""
    dn = np.array([[10.4, 254.6, 300.0]], dtype=np.float64)
    out = quantize_dn(dn, mode='8bit', saturation_dn=4095.0)
    assert np.array_equal(out, np.array([[10.0, 255.0, 255.0]]))


def test_8bit_ceiling_is_independent_of_saturation_dn() -> None:
    """A deeper detector's saturation_dn does not raise the 8-bit code ceiling."""
    dn = np.array([[1000.0]], dtype=np.float64)
    out = quantize_dn(dn, mode='8bit', saturation_dn=65535.0)
    assert float(out[0, 0]) == 255.0


def test_uneven_12bit_snaps_to_power_of_two_boundaries() -> None:
    """The uneven-12bit sub-mode concentrates codes at power-of-two boundaries."""
    dn = np.array([[255.0, 256.0, 257.0]], dtype=np.float64)
    out = quantize_dn(dn, mode='uneven_12bit', saturation_dn=4095.0)
    assert np.all(out == 256.0)


def test_quantize_ls8b_wraps_modulo_256() -> None:
    """LS8B keeps the low 8 bits: 256 -> 0, 300 -> 44, 4095 -> 255."""
    out = quantize_dn(np.array([256.0, 300.0, 4095.0]), mode='ls8b', saturation_dn=4095.0)
    assert out.tolist() == [0.0, 44.0, 255.0]


def test_quantize_contour_8bit_posterizes_to_step_multiples() -> None:
    """contour_8bit posterizes the 8-bit word to DN multiples of the step."""
    dn = np.arange(0, 256, dtype=np.float64)
    out = quantize_dn(dn, mode='contour_8bit', saturation_dn=255.0, contour_step=8)
    levels = set(out.tolist())
    for level in levels:
        assert level % 8 == 0 or level == 255.0
    assert 8.0 in levels
    assert 248.0 in levels


def test_a_scene_sets_the_contour_step() -> None:
    """A contour_8bit frame holds only multiples of the scene's step."""
    img, _ = render_scene(
        disc_scene(
            'gossi',
            noise={'read_noise_dn': 5.0},
            detector={'quantization': 'contour_8bit', 'contour_step': 16},
        )
    )
    levels = set(np.unique(img).tolist())
    assert len(levels) > 1
    assert {level % 16 for level in levels - {255.0}} == {0.0}


def test_sqrt_lut_leaves_a_signal_dependent_residual() -> None:
    """The sqrt-LUT companding round-trip leaves a residual that grows with signal."""
    dn = np.array([[100.0, 3000.0]], dtype=np.float64)
    out = quantize_dn(dn, mode='sqrt_lut', saturation_dn=4095.0)
    residual = np.abs(out - dn)
    assert float(residual[0, 1]) > float(residual[0, 0])


@pytest.mark.parametrize(
    'instrument', ['coiss_nac', 'coiss_wac', 'coiss_calib_nac', 'coiss_calib_wac']
)
def test_sqrt_lut_decodes_through_the_cassini_table(instrument: str) -> None:
    """Each Cassini camera's sqrt_lut quantization reads its measured table."""
    dp = resolve_detector_params(disc_scene(instrument, detector={'quantization': 'sqrt_lut'}))
    assert dp.sqrt_lut == LUTSqrt(codes=256, linear_dn=2.0, quadratic_dn=0.055, whole_dn=True)


@pytest.mark.parametrize(
    ('code', 'step'),
    [(10, 3.0), (25, 5.0), (55, 8.0), (100, 13.0), (200, 24.0)],
    ids=['code-10', 'code-25', 'code-55', 'code-100', 'code-200'],
)
def test_the_cassini_table_steps_grow_with_the_code(code: int, step: float) -> None:
    """The table's decode steps match the steps measured on real TABLE frames."""
    assert float(_CASSINI_DN[code + 1] - _CASSINI_DN[code]) == step


def test_the_cassini_table_decodes_256_whole_dn_up_to_4086() -> None:
    """Every code decodes to its own whole DN, the top code to 4086."""
    assert len(_CASSINI_DN) == 256
    assert np.array_equal(_CASSINI_DN, np.round(_CASSINI_DN))
    assert float(_CASSINI_DN[-1]) == 4086.0


def test_the_sqrt_lut_snaps_each_value_to_the_nearest_decoded_dn() -> None:
    """Values between two decoded DN go to the nearer one; above the table, to its top."""
    lower, upper = float(_CASSINI_DN[40]), float(_CASSINI_DN[41])
    dn = np.array([[lower + 0.4 * (upper - lower), lower + 0.6 * (upper - lower), 5000.0]])
    out = quantize_dn(dn, mode='sqrt_lut', saturation_dn=4095.0, sqrt_lut=_CASSINI_LUT)
    assert out.tolist() == [[lower, upper, float(_CASSINI_DN[-1])]]


def test_the_sqrt_lut_keeps_a_nan_pixel_nan() -> None:
    """An invalid pixel stays invalid instead of decoding to the table's top."""
    out = quantize_dn(
        np.array([[np.nan, 100.0]]), mode='sqrt_lut', saturation_dn=4095.0, sqrt_lut=_CASSINI_LUT
    )
    assert bool(np.isnan(out[0, 0]))


def test_a_camera_without_a_table_gets_the_plain_square_root_curve() -> None:
    """Without a catalog entry the curve is a pure square root topping out at the ADC ceiling."""
    curve = resolve_sqrt_lut(None, saturation_dn=4095.0)
    assert curve == LUTSqrt(
        codes=256, linear_dn=0.0, quadratic_dn=4095.0 / 255.0**2, whole_dn=False
    )
    assert float(curve.levels()[-1]) == pytest.approx(4095.0)


def test_a_table_entry_sets_its_own_code_count() -> None:
    """A 10-bit table decodes 1024 levels, its top code at the curve's value for 1023."""
    table = DownlinkTable(codes=1024, linear_dn=0.0, quadratic_dn=0.01, whole_dn=False)
    levels = resolve_sqrt_lut(table, saturation_dn=16383.0).levels()
    assert len(levels) == 1024
    assert float(levels[-1]) == pytest.approx(0.01 * 1023.0**2)


def test_the_plain_curve_on_a_zero_ceiling_sends_every_value_to_zero() -> None:
    """The plain curve never decodes above the ADC ceiling, even a zero one."""
    dn = np.array([[0.4, 37.0, 2500.0]])
    out = quantize_dn(dn, mode='sqrt_lut', saturation_dn=0.0)
    assert out.tolist() == [[0.0, 0.0, 0.0]]


def test_the_sqrt_lut_mode_defaults_to_the_plain_curve() -> None:
    """A call without a curve quantizes through the plain square-root curve."""
    dn = np.array([[0.4, 37.0, 2500.0]])
    plain = resolve_sqrt_lut(None, saturation_dn=4095.0)
    default = quantize_dn(dn, mode='sqrt_lut', saturation_dn=4095.0)
    explicit = quantize_dn(dn, mode='sqrt_lut', saturation_dn=4095.0, sqrt_lut=plain)
    assert np.array_equal(default, explicit)
