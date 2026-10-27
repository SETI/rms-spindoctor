"""The cosmic-ray rate: one unit, one name, and refused where the chain ignores it.

``noise.cosmic_ray_events_per_pixel_per_sec`` is the expected number of
cosmic-ray events per pixel per second of exposure: a frame's Poisson mean is
the rate times the exposure times the pixel count, with no other factor.  The
retired ``cosmic_ray_rate_per_sec`` and ``pixel_area_cm2`` keys are refused with
a message naming the replacement or the removal.  The cosmic-ray stage exists
only in the CCD electron chain, so the validator refuses a non-zero rate on a
scene whose detector model resolves to the vidicon, and on no other scene.
Under ``artifacts.instrument_defaults`` the catalog's rate applies.  An explicit
null ``detector.detector_model`` means the instrument's default model, to the
renderer as to the validator.
"""

import re
from typing import Any, cast

import numpy as np
import pytest

from spindoctor.config import DEFAULT_CONFIG
from spindoctor.sim.forward.artifacts_catalog import DETECTOR_DEFAULTS, resolve_detector_defaults
from spindoctor.sim.forward.detector import apply_detector
from spindoctor.sim.forward.detector.noise_stages import add_cosmic_rays
from spindoctor.sim.forward.detector.params import resolve_detector_params
from spindoctor.sim.forward.stages import SimFrame
from spindoctor.sim.instruments import SIM_INSTRUMENTS
from spindoctor.sim.scene import SimSceneValidationError, validate_sim_params
from spindoctor.sim.scene_checks import RENAMED_NOISE_KEYS
from spindoctor.sim.seeds import derive_effect_seed

_KEY = 'cosmic_ray_events_per_pixel_per_sec'
_RETIRED = ('cosmic_ray_rate_per_sec', 'pixel_area_cm2')
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


# ---- The retired keys ----


def test_renamed_noise_keys_table() -> None:
    """The table maps the old rate to the per-pixel key and records the area's removal."""
    assert {old: new for old, (new, _) in RENAMED_NOISE_KEYS.items()} == {
        'cosmic_ray_rate_per_sec': _KEY,
        'pixel_area_cm2': None,
    }


@pytest.mark.parametrize(
    ('old_key', 'message'),
    [
        (
            'cosmic_ray_rate_per_sec',
            "noise: 'cosmic_ray_rate_per_sec' was renamed; use "
            "'cosmic_ray_events_per_pixel_per_sec' "
            '(expected events per pixel per second of exposure)',
        ),
        (
            'pixel_area_cm2',
            "noise: 'pixel_area_cm2' was removed; the cosmic-ray rate "
            "'cosmic_ray_events_per_pixel_per_sec' is already per pixel",
        ),
    ],
)
def test_retired_noise_key_is_refused_naming_its_fate(old_key: str, message: str) -> None:
    """Each retired key fails validation with a message naming its replacement or removal."""
    with pytest.raises(SimSceneValidationError, match=re.escape(f'scene: {message}')):
        validate_sim_params(_scene('coiss_nac', {old_key: 1.0}), source='scene')


@pytest.mark.parametrize('old_key', _RETIRED)
def test_retired_key_is_absent_from_the_sim_config(old_key: str) -> None:
    """The packaged sim config's noise section carries neither retired key."""
    assert old_key not in DEFAULT_CONFIG.category('sim')['noise']


def test_sim_config_floor_uses_the_per_pixel_key() -> None:
    """The packaged sim config's noise floor is the per-pixel key, at zero."""
    assert DEFAULT_CONFIG.category('sim')['noise'][_KEY] == 0.0


@pytest.mark.parametrize('entry', sorted(DETECTOR_DEFAULTS))
def test_catalog_entries_carry_only_the_per_pixel_key(entry: str) -> None:
    """No catalog entry carries a retired key."""
    assert set(DETECTOR_DEFAULTS[entry]) & set(_RETIRED) == set()


# ---- The vidicon path ----


@pytest.mark.parametrize(
    ('instrument', 'detector', 'via'),
    [
        ('vgiss', None, ''),
        ('coiss_nac', {'detector_model': 'vidicon'}, ' via detector.detector_model'),
        ('generic', {'detector_model': 'vidicon'}, ' via detector.detector_model'),
    ],
    ids=['vgiss-default', 'coiss-nac-vidicon', 'generic-vidicon'],
)
def test_nonzero_rate_on_the_vidicon_is_refused(
    instrument: str, detector: dict[str, str] | None, via: str
) -> None:
    """A positive rate on a vidicon-path scene fails, saying the chain would ignore it."""
    extra = {} if detector is None else {'detector': detector}
    scene = _scene(instrument, {_KEY: 0.01}, **extra)
    message = (
        f'scene: noise.{_KEY} is 0.01, but instrument {instrument!r} renders through '
        f'the vidicon detector{via}, whose DN chain lacks a cosmic-ray stage, so the '
        'rate would be silently ignored; set it to 0 or remove it'
    )
    with pytest.raises(SimSceneValidationError, match=re.escape(message)):
        validate_sim_params(scene, source='scene')


def test_zero_rate_on_vgiss_is_accepted() -> None:
    """A zero rate on the vidicon is the floor and passes."""
    scene = _scene('vgiss', {_KEY: 0.0})
    assert validate_sim_params(scene) is scene


@pytest.mark.parametrize('instrument', _INSTRUMENTS)
@pytest.mark.parametrize(
    'model', [_ABSENT, None, 'ccd', 'vidicon'], ids=['default', 'null', 'ccd', 'vidicon']
)
def test_validator_refuses_exactly_the_paths_where_the_rate_is_inert(
    instrument: str, model: str | None
) -> None:
    """The validator refuses a rate if and only if the detector output ignores it."""
    extra = {} if model == _ABSENT else {'detector': {'detector_model': model}}
    without = _detector_dn(_scene(instrument, {_KEY: 0.0}, **extra))
    with_rate = _detector_dn(_scene(instrument, {_KEY: 0.05}, **extra))
    inert = bool(np.array_equal(without, with_rate))
    try:
        validate_sim_params(_scene(instrument, {_KEY: 0.05}, **extra))
    except SimSceneValidationError:
        refused = True
    else:
        refused = False
    assert refused == inert


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
