"""The cosmic-ray rate: one unit, one name, and honored on every detector path.

``noise.cosmic_ray_events_per_pixel_per_sec`` is the expected number of
cosmic-ray events per pixel per second of exposure: a frame's Poisson mean is
the rate times the exposure times the pixel count, with no other factor.  The
CCD electron chain and the vidicon DN chain both deposit the hits, with that
same meaning, and the validator accepts a rate on every scene.  On the vidicon
the hits land on the stored target image: after the prior frame's residual
image, which this frame's hits don't reach, and before the readout.  Under
``artifacts.instrument_defaults`` the catalog's rate applies, and below that the
configuration's ``sim.noise`` floor.  An explicit null ``detector.detector_model``
means the instrument's default model.
"""

from typing import Any, cast

import numpy as np
import pytest

from spindoctor.config import DEFAULT_CONFIG
from spindoctor.sim.forward.artifacts_catalog import DETECTOR_DEFAULTS, resolve_detector_defaults
from spindoctor.sim.forward.detector import apply_detector, chain
from spindoctor.sim.forward.detector.noise_stages import add_cosmic_rays
from spindoctor.sim.forward.detector.params import resolve_detector_params
from spindoctor.sim.forward.stages import SimFrame
from spindoctor.sim.instruments import SIM_INSTRUMENTS
from spindoctor.sim.scene import validate_sim_params
from spindoctor.sim.seeds import derive_effect_seed

_KEY = 'cosmic_ray_events_per_pixel_per_sec'
# The detector block a property case uses: absent, an explicit null (which the
# validator and the renderer must both read as the instrument's default), or a
# named model.
_ABSENT = 'absent'
_INSTRUMENTS = (*sorted(SIM_INSTRUMENTS), 'generic')


class _RecordingRng:
    """A seeded generator that records every Poisson mean it is asked to draw."""

    def __init__(self, seed: int) -> None:
        """Wrap ``np.random.default_rng(seed)``."""
        self._rng = np.random.default_rng(seed)
        self.poisson_means: list[float] = []

    def poisson(self, lam: float) -> Any:
        """Record ``lam`` and draw from the wrapped generator."""
        self.poisson_means.append(float(lam))
        return self._rng.poisson(lam)

    def __getattr__(self, name: str) -> Any:
        """Delegate every other draw to the wrapped generator."""
        return getattr(self._rng, name)


def _scene(instrument: str, noise: dict[str, Any], **extra: Any) -> dict[str, Any]:
    """A minimal valid scene on ``instrument`` carrying ``noise``."""
    scene: dict[str, Any] = {
        'size_v': 32,
        'size_u': 32,
        'random_seed': 4,
        'instrument': instrument,
        'exposure_sec': 1.0,
        'noise': noise,
    }
    scene.update(extra)
    return scene


def _detector_dn(scene: dict[str, Any]) -> np.ndarray:
    """Run the detector stage over a dark (zero-signal) frame of ``scene``'s size.

    A dark frame keeps every instrument below its ADC ceiling, so a cosmic-ray
    hit always changes the DN it lands on.
    """
    shape = (int(scene['size_v']), int(scene['size_u']))
    frame = SimFrame(
        signal=np.zeros(shape, dtype=np.float64),
        point_e=np.zeros(shape, dtype=np.float64),
    )
    rng = np.random.default_rng(derive_effect_seed(int(scene['random_seed']), 'detector'))
    apply_detector(frame, params=scene, rng=rng)
    return frame.signal.copy()


# ---- The unit ----


@pytest.mark.parametrize(
    ('size_v', 'size_u', 'exposure_sec', 'rate'),
    [(100, 100, 1.0, 1.0e-3), (64, 200, 2.5, 4.0e-4), (1, 1, 10.0, 0.05)],
    ids=['square-1s', 'wide-2.5s', 'one-pixel-10s'],
)
def test_frame_poisson_mean_is_rate_times_exposure_times_pixel_count(
    size_v: int, size_u: int, exposure_sec: float, rate: float
) -> None:
    """The per-pixel reading: the event count's mean is rate * exposure * pixels."""
    rng = _RecordingRng(9)
    add_cosmic_rays(
        np.zeros((size_v, size_u), dtype=np.float64),
        events_per_pixel_per_sec=rate,
        exposure_sec=exposure_sec,
        amplitude_e=1.0e5,
        rng=cast(np.random.Generator, rng),
    )
    assert rng.poisson_means[0] == pytest.approx(rate * exposure_sec * size_v * size_u)


def test_scene_rate_reaches_the_stage_unscaled() -> None:
    """The scene's rate resolves to the stage's rate with no area or unit factor."""
    resolved = resolve_detector_params(_scene('coiss_nac', {_KEY: 3.5e-4}))
    assert resolved.cosmic_ray_events_per_pixel_per_sec == pytest.approx(3.5e-4)


# ---- The configuration floor ----


def test_sim_config_floor_uses_the_per_pixel_key() -> None:
    """The packaged sim config's noise floor is the per-pixel key, at zero."""
    assert DEFAULT_CONFIG.category('sim')['noise'][_KEY] == 0.0


def test_config_floor_rate_reaches_a_scene_that_leaves_it_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-zero sim.noise floor is the rate of a scene that sets none."""
    monkeypatch.setitem(DEFAULT_CONFIG.category('sim')['noise'], _KEY, 6.0e-4)
    resolved = resolve_detector_params(_scene('coiss_nac', {}))
    assert resolved.cosmic_ray_events_per_pixel_per_sec == pytest.approx(6.0e-4)


def test_config_floor_rate_gives_way_to_the_scene(monkeypatch: pytest.MonkeyPatch) -> None:
    """A scene's own rate wins over a non-zero sim.noise floor."""
    monkeypatch.setitem(DEFAULT_CONFIG.category('sim')['noise'], _KEY, 6.0e-4)
    resolved = resolve_detector_params(_scene('coiss_nac', {_KEY: 1.0e-5}))
    assert resolved.cosmic_ray_events_per_pixel_per_sec == pytest.approx(1.0e-5)


# ---- Every detector path ----


@pytest.mark.parametrize('instrument', _INSTRUMENTS)
@pytest.mark.parametrize(
    'model', [_ABSENT, None, 'ccd', 'vidicon'], ids=['default', 'null', 'ccd', 'vidicon']
)
def test_a_rate_deposits_hits_on_every_detector_path(instrument: str, model: str | None) -> None:
    """A positive rate changes the detector output on every instrument and model."""
    extra = {} if model == _ABSENT else {'detector': {'detector_model': model}}
    without = _detector_dn(_scene(instrument, {_KEY: 0.0}, **extra))
    with_rate = _detector_dn(_scene(instrument, {_KEY: 0.05}, **extra))
    assert not np.array_equal(without, with_rate)


@pytest.mark.parametrize('instrument', _INSTRUMENTS)
@pytest.mark.parametrize(
    'model', [_ABSENT, None, 'ccd', 'vidicon'], ids=['default', 'null', 'ccd', 'vidicon']
)
def test_the_validator_accepts_a_rate_on_every_path(instrument: str, model: str | None) -> None:
    """A positive rate validates on every instrument and detector model."""
    extra = {} if model == _ABSENT else {'detector': {'detector_model': model}}
    scene = _scene(instrument, {_KEY: 0.05}, **extra)
    assert validate_sim_params(scene) is scene


@pytest.mark.parametrize('detector_model', ['ccd', 'vidicon'])
def test_both_chains_give_the_stage_the_rate_and_exposure(
    monkeypatch: pytest.MonkeyPatch, detector_model: str
) -> None:
    """The CCD and vidicon chains call the stage with the scene's rate and exposure."""
    calls: list[tuple[float, float]] = []

    def _record(image: np.ndarray, **kwargs: Any) -> None:
        calls.append((kwargs['events_per_pixel_per_sec'], kwargs['exposure_sec']))

    monkeypatch.setattr(chain, 'add_cosmic_rays', _record)
    scene = _scene('vgiss', {_KEY: 3.0e-4}, detector={'detector_model': detector_model})
    scene['exposure_sec'] = 2.5
    _detector_dn(scene)
    assert calls == [(3.0e-4, 2.5)]


def _vgiss_dn(noise: dict[str, Any], **extra: Any) -> np.ndarray:
    """A dark 64 x 64 vgiss frame in raw DN (its unit class pinned to raw_dn)."""
    scene = _scene('vgiss', noise, instrument_config={'data_units': 'raw_dn'}, **extra)
    scene.update(size_v=64, size_u=64)
    return _detector_dn(scene)


def test_a_vidicon_hit_digitizes_at_the_adc_ceiling() -> None:
    """Most vidicon hit pixels read 255 DN, the 8-bit ceiling, and none reads more.

    A hit's core deposits at least the DN full scale; a splatter's neighbors take
    a fraction of it, so a few hit pixels sit below the ceiling.
    """
    without = _vgiss_dn({_KEY: 0.0})
    with_rate = _vgiss_dn({_KEY: 0.01})
    hits = with_rate[with_rate != without]
    assert float(hits.max()) == 255.0
    assert float((hits == 255.0).mean()) > 0.5


def test_the_vidicon_hit_count_follows_the_rate() -> None:
    """A dark vidicon frame takes about rate * exposure * pixels events.

    Each event marks at least one pixel and the morphology adds a few, so the
    count of hit pixels sits between the event count's 4-sigma floor and four
    times its 4-sigma ceiling.
    """
    rate = 2.0e-3
    expected = rate * 1.0 * 64 * 64
    hit = int((_vgiss_dn({_KEY: rate}) != _vgiss_dn({_KEY: 0.0})).sum())
    assert expected - 4.0 * expected**0.5 < hit < 4.0 * (expected + 4.0 * expected**0.5)


def test_this_frames_hits_stay_out_of_the_residual_image() -> None:
    """The residual image ghosts the prior frame, so a hit gets no ghost at the offset.

    The hits change a dark frame by the same amount with and without the
    residual image; a hit the residual image copied would add a faint ghost five
    pixels down and right.
    """
    residual = {'residual_image': {'probability': 1.0, 'amplitude': 0.05, 'offset_px': [5, 5]}}
    plain = _vgiss_dn({_KEY: 0.01}) - _vgiss_dn({_KEY: 0.0})
    ghosted = _vgiss_dn({_KEY: 0.01}, artifacts=residual) - _vgiss_dn(
        {_KEY: 0.0}, artifacts=residual
    )
    assert int(np.count_nonzero(plain)) > 0
    assert np.array_equal(ghosted, plain)


@pytest.mark.parametrize('instrument', _INSTRUMENTS)
def test_null_detector_model_resolves_to_the_catalog_model(instrument: str) -> None:
    """An explicit null detector model renders through the instrument's default model."""
    scene = _scene(instrument, {}, detector={'detector_model': None})
    expected = resolve_detector_defaults(instrument)['detector_model']
    assert resolve_detector_params(scene).detector_model == expected


def test_null_detector_keys_resolve_like_absent_ones() -> None:
    """Every detector key set to null resolves exactly as if the block omitted it."""
    nulls = dict.fromkeys(('gain_state', 'detector_model', 'exposure_ref_sec', 'quantization'))
    scene = _scene('coiss_nac', {}, detector=nulls)
    assert validate_sim_params(scene) is scene
    assert resolve_detector_params(scene) == resolve_detector_params(_scene('coiss_nac', {}))


# ---- instrument_defaults ----


def test_instrument_defaults_applies_the_catalog_rate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Under instrument_defaults the catalog's rate applies (every entry holds 0.0 today)."""
    monkeypatch.setitem(DETECTOR_DEFAULTS['coiss_nac'], _KEY, 2.0e-4)
    scene = _scene('coiss_nac', {}, artifacts={'instrument_defaults': True})
    resolved = resolve_detector_params(scene)
    assert resolved.cosmic_ray_events_per_pixel_per_sec == pytest.approx(2.0e-4)


def test_catalog_rate_stays_off_without_instrument_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without instrument_defaults the catalog's rate does not apply."""
    monkeypatch.setitem(DETECTOR_DEFAULTS['coiss_nac'], _KEY, 2.0e-4)
    resolved = resolve_detector_params(_scene('coiss_nac', {}))
    assert resolved.cosmic_ray_events_per_pixel_per_sec == 0.0
