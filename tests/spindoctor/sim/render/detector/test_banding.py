"""The banding effect: its bands, its dark step, and the cameras that show it."""

from typing import Any

import numpy as np
import pytest
from tests.spindoctor.sim.scenes import disc_scene, frame_scene, quiet_noise

from spindoctor.sim.render.detector.banding import add_banding
from spindoctor.sim.render.detector.params import resolve_detector_params
from spindoctor.sim.render.driver import render_scene
from spindoctor.support.types import NDArrayFloatType

_SIZE = 64


def _draw(
    plane: NDArrayFloatType,
    *,
    amplitude: float = 0.0,
    period_px: float = 0.0,
    orientation: str = 'horizontal',
    horizontal_pattern: str = 'periodic',
    freq_step_factor: float = 1.0,
    dark_step_dn: float = 0.0,
    ringing_amplitude: float = 0.0,
    ringing_period_px: float = 0.0,
    ringing_decay_px: float = 0.0,
    binning: int = 1,
    gain_e_per_dn: float = 1.0,
    seed: int = 0,
) -> dict[str, Any]:
    """Draw banding on a plane with a seeded generator; a key left out draws nothing.

    Parameters:
        plane: The plane, modified in place.
        amplitude: The bands' amplitude.
        period_px: The periodic bands' period, in frame pixels.
        orientation: 'horizontal', 'vertical', or 'both'.
        horizontal_pattern: 'periodic' or 'random'.
        freq_step_factor: The periodic horizontal bands' frequency factor below the
            mid-line.
        dark_step_dn: The dark step below the mid-line, in DN.
        ringing_amplitude: The column ringing's amplitude at the first column.
        ringing_period_px: The column ringing's period, in native columns.
        ringing_decay_px: The column ringing's e-folding distance, in native columns.
        binning: Native columns per frame column.
        gain_e_per_dn: Plane units per DN.
        seed: The generator's seed.

    Returns:
        The banding's truth record.
    """
    return add_banding(
        plane,
        amplitude=amplitude,
        period_px=period_px,
        orientation=orientation,
        horizontal_pattern=horizontal_pattern,
        freq_step_factor=freq_step_factor,
        dark_step_dn=dark_step_dn,
        ringing_amplitude=ringing_amplitude,
        ringing_period_px=ringing_period_px,
        ringing_decay_px=ringing_decay_px,
        binning=binning,
        gain_e_per_dn=gain_e_per_dn,
        rng=np.random.default_rng(seed),
    )


def _random_rows(*, period_px: float = 16.0) -> NDArrayFloatType:
    """Random horizontal stripes of 100 units, drawn on an empty plane."""
    plane = np.zeros((_SIZE, _SIZE), dtype=np.float64)
    _draw(
        plane,
        amplitude=100.0,
        period_px=period_px,
        orientation='horizontal',
        horizontal_pattern='random',
        seed=5,
    )
    return plane


def test_random_stripes_are_constant_along_a_row() -> None:
    """Each random stripe is one offset across its whole row."""
    assert float(np.ptp(_random_rows(), axis=1).max()) == 0.0


def test_random_stripes_stay_within_the_amplitude() -> None:
    """No random stripe's offset is larger than the amplitude."""
    assert float(np.abs(_random_rows()).max()) <= 100.0


def test_random_stripes_spread_across_the_amplitude() -> None:
    """The random stripes reach out toward both ends of the amplitude."""
    assert float(np.ptp(_random_rows())) > 150.0


def test_random_stripes_need_no_period() -> None:
    """Random stripes are drawn with a zero period, which only periodic bands need."""
    assert float(np.ptp(_random_rows(period_px=0.0))) > 150.0


def _ringing(*, binning: int = 1, seed: int = 0) -> NDArrayFloatType:
    """Column ringing alone, drawn on an empty plane."""
    plane = np.zeros((_SIZE, _SIZE), dtype=np.float64)
    _draw(
        plane,
        ringing_amplitude=10.0,
        ringing_period_px=20.0,
        ringing_decay_px=15.0,
        binning=binning,
        seed=seed,
    )
    return plane


def _native_wave(size: int) -> NDArrayFloatType:
    """The ringing's damped sine at each of ``size`` native columns."""
    cols = np.arange(size, dtype=np.float64)
    wave: NDArrayFloatType = 10.0 * np.exp(-cols / 15.0) * np.sin(2.0 * np.pi * cols / 20.0)
    return wave


def test_ringing_is_a_damped_sine_from_the_first_column() -> None:
    """Every row of the ringing is the damped sine that starts at column 0."""
    assert np.allclose(_ringing(), _native_wave(_SIZE)[None, :], rtol=0.0, atol=1e-12)


def test_ringing_is_the_same_in_every_frame() -> None:
    """The ringing is fixed to the line start, so a different generator draws it the same."""
    assert np.array_equal(_ringing(seed=1), _ringing(seed=2))


def test_a_binned_column_rings_with_its_native_columns_mean() -> None:
    """At 4x4 a frame column takes the mean of the ringing over its four native columns."""
    expected = _native_wave(4 * _SIZE).reshape(_SIZE, 4).mean(axis=1)
    assert np.allclose(_ringing(binning=4)[0], expected, rtol=0.0, atol=1e-12)


def test_ringing_alone_turns_the_banding_on() -> None:
    """Ringing with no bands still draws, and the truth record says the banding is active."""
    record = _draw(
        np.zeros((_SIZE, _SIZE), dtype=np.float64),
        ringing_amplitude=10.0,
        ringing_period_px=20.0,
        ringing_decay_px=15.0,
    )
    assert record['active'] is True


def test_the_chain_rings_over_binned_columns() -> None:
    """A 4x4 LORRI frame records its ringing's period and decay in binned columns."""
    banding = {
        'probability': 1.0,
        'amplitude': 0.0,
        'ringing_amplitude': 10.0,
        'ringing_period_px': 20.0,
        'ringing_decay_px': 15.0,
    }
    scene = frame_scene(
        instrument='nhlorri',
        exposure_sec=0.1,
        noise=quiet_noise(),
        detector={'quantization': 'exact', 'binning': 4},
        artifacts={'banding': banding},
    )
    _img, truth = render_scene(scene)
    assert truth['artifacts']['banding']['ringing'] == {'period_px': 5.0, 'decay_px': 3.75}


def test_banding_horizontal_is_row_correlated() -> None:
    """Horizontal banding is constant along a row and varies between rows."""
    electrons = np.zeros((_SIZE, _SIZE), dtype=np.float64)
    _draw(
        electrons,
        amplitude=100.0,
        period_px=16.0,
        orientation='horizontal',
        freq_step_factor=1.0,
        dark_step_dn=0.0,
        gain_e_per_dn=1.0,
        seed=5,
    )
    assert float(np.ptp(electrons, axis=1).max()) < 1e-9
    assert float(np.ptp(electrons, axis=0).max()) > 1.0


def test_banding_vertical_is_column_correlated() -> None:
    """Vertical banding is constant down a column and varies between columns."""
    electrons = np.zeros((_SIZE, _SIZE), dtype=np.float64)
    _draw(
        electrons,
        amplitude=100.0,
        period_px=8.0,
        orientation='vertical',
        freq_step_factor=1.0,
        dark_step_dn=0.0,
        gain_e_per_dn=1.0,
        seed=5,
    )
    assert float(np.ptp(electrons, axis=0).max()) < 1e-9
    assert float(np.ptp(electrons, axis=1).max()) > 1.0


def _bands(orientation: str, *, freq_step_factor: float = 1.0) -> NDArrayFloatType:
    """Bands of 100 units every 16 pixels, drawn on an empty plane."""
    plane = np.zeros((_SIZE, _SIZE), dtype=np.float64)
    _draw(
        plane,
        amplitude=100.0,
        period_px=16.0,
        orientation=orientation,
        freq_step_factor=freq_step_factor,
        dark_step_dn=0.0,
        gain_e_per_dn=1.0,
        seed=5,
    )
    return plane


def test_both_orientations_vary_along_a_row() -> None:
    """Bands drawn in both orientations include the vertical set, which varies along a row."""
    assert float(np.ptp(_bands('both'), axis=1).min()) > 1.0


def test_both_orientations_vary_down_a_column() -> None:
    """Bands drawn in both orientations include the horizontal set, which varies down a column."""
    assert float(np.ptp(_bands('both'), axis=0).min()) > 1.0


def test_the_frequency_step_leaves_the_upper_half_alone() -> None:
    """The bands above the mid-line are the same with and without the frequency step."""
    stepped = _bands('horizontal', freq_step_factor=2.0)
    assert np.array_equal(stepped[: _SIZE // 2], _bands('horizontal')[: _SIZE // 2])


def test_the_frequency_step_changes_the_lower_half() -> None:
    """The bands below the mid-line change with the frequency step."""
    stepped = _bands('horizontal', freq_step_factor=2.0)
    assert not np.array_equal(stepped[_SIZE // 2 :], _bands('horizontal')[_SIZE // 2 :])


def _dark_step_frame(instrument: str, noise: dict[str, Any]) -> dict[str, Any]:
    """A quiet empty frame whose bands are negligible and whose dark step is 10 DN."""
    banding = {'probability': 1.0, 'amplitude': 1.0e-6, 'period_px': 16.0, 'dark_step_dn': 10.0}
    return frame_scene(
        instrument=instrument,
        exposure_sec=0.1,
        noise=noise,
        detector={'quantization': 'exact'},
        artifacts={'banding': banding},
    )


def test_the_ccd_chain_converts_the_dark_step_through_the_gain() -> None:
    """On the CCD chain the lower half of the frame sits 10 DN above the upper half."""
    img, _ = render_scene(_dark_step_frame('coiss_nac', quiet_noise()))
    step = float(img[_SIZE // 2 :].mean() - img[: _SIZE // 2].mean())
    assert step == pytest.approx(10.0)


def test_the_vidicon_chain_adds_the_dark_step_in_dn() -> None:
    """On the vidicon chain the dark step is 10 DN, read through the calibration."""
    quiet = {'vidicon': {'read_noise_line_dn': 0.0, 'read_noise_pixel_dn': 0.0}}
    scene = _dark_step_frame('vgiss_1_nac', quiet)
    img, _ = render_scene(scene)
    step_if = 10.0 / (resolve_detector_params(scene).calibration_scale_dn_per_s_per_if * 0.1)
    step = float(img[_SIZE // 2 :].mean() - img[: _SIZE // 2].mean())
    assert step == pytest.approx(step_if)


def test_the_ccd_chain_records_its_bands() -> None:
    """The frame truth records the orientation, the period, and the dark-step line."""
    _img, truth = render_scene(_dark_step_frame('coiss_nac', quiet_noise()))
    assert truth['artifacts']['banding'] == {
        'active': True,
        'orientation': 'horizontal',
        'horizontal_pattern': 'periodic',
        'period_px': 16.0,
        'dark_step_line': _SIZE // 2,
        'ringing': None,
    }


def test_banding_dark_step_lifts_the_lower_half() -> None:
    """A dark-level step raises the mean of the rows below the mid-line."""
    electrons = np.zeros((_SIZE, _SIZE), dtype=np.float64)
    _draw(
        electrons,
        amplitude=10.0,
        period_px=16.0,
        orientation='horizontal',
        freq_step_factor=1.0,
        dark_step_dn=50.0,
        gain_e_per_dn=2.0,
        seed=1,
    )
    top = float(electrons[: _SIZE // 2].mean())
    bottom = float(electrons[_SIZE // 2 :].mean())
    assert bottom - top == pytest.approx(100.0, abs=5.0)


def test_banding_disabled_at_zero_amplitude() -> None:
    """A zero banding amplitude is a no-op."""
    electrons = np.zeros((_SIZE, _SIZE), dtype=np.float64)
    record = _draw(
        electrons,
        amplitude=0.0,
        period_px=16.0,
        orientation='both',
        freq_step_factor=1.0,
        dark_step_dn=0.0,
        gain_e_per_dn=1.0,
        seed=5,
    )
    assert record['active'] is False
    assert np.all(electrons == 0.0)


def test_the_calibrated_cassini_products_show_no_banding() -> None:
    """The instrument defaults of a calibrated Cassini camera leave banding off."""
    dp = resolve_detector_params(
        disc_scene('coiss_calib_nac', artifacts={'instrument_defaults': True})
    )
    assert 'banding' not in dp.detector_modes


def test_the_vidicon_bands_are_vertical() -> None:
    """The vidicon chain draws its camera's bands down the columns, in DN."""
    quiet = {'vidicon': {'read_noise_line_dn': 0.0, 'read_noise_pixel_dn': 0.0}}
    blocks: dict[str, Any] = {'noise': quiet, 'detector': {'quantization': 'exact'}}
    on, _ = render_scene(
        disc_scene('vgiss_1_nac', artifacts={'banding': {'probability': 1.0}}, **blocks)
    )
    strong, truth = render_scene(
        disc_scene(
            'vgiss_1_nac', artifacts={'banding': {'probability': 1.0, 'amplitude': 20.0}}, **blocks
        )
    )
    assert truth['artifacts']['banding']['orientation'] == 'vertical'
    sky = strong[:8]
    assert float(np.ptp(sky, axis=0).max()) == 0.0
    assert float(np.ptp(sky)) > 10.0 * float(np.ptp(on[:8]))
