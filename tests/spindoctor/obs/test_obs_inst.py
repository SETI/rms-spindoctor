"""Unit tests for the ``ObsInst`` instrument mix-in helpers."""

from typing import Any

import pytest
from starcat import Star

from spindoctor.obs.obs_inst import ObsInst


class _ConcreteObsInst(ObsInst):
    """Minimal concrete ``ObsInst`` for exercising the generic helpers.

    The abstract methods are stubbed so the class can be instantiated
    without a real ``Obs`` / ``oops.Observation`` backing it.
    """

    model_source = 'spice'

    @staticmethod
    def from_file(path: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    @property
    def camera(self) -> str:
        return 'TEST'

    def star_min_usable_vmag(self) -> float:
        return 0.0

    def star_max_usable_vmag(self) -> float:
        return 20.0

    def get_public_metadata(self) -> dict[str, Any]:
        return {}


def _make_star(vmag: float) -> Star:
    """Build a ``Star`` carrying the given visual magnitude."""
    star = Star()
    star.vmag = vmag
    return star


def test_star_psf_size_selects_threshold() -> None:
    """A star below a threshold returns that threshold's size."""
    inst = _ConcreteObsInst()
    inst._inst_config = {'star_psf_sizes': {5: [3, 3], 10: [7, 7]}}
    assert inst.star_psf_size(_make_star(4.0)) == (3, 3)


def test_star_psf_size_fainter_than_all_returns_largest() -> None:
    """A star fainter than every threshold returns the largest size."""
    inst = _ConcreteObsInst()
    inst._inst_config = {'star_psf_sizes': {5: [3, 3], 10: [7, 7]}}
    assert inst.star_psf_size(_make_star(99.0)) == (7, 7)


def test_star_psf_size_empty_raises_value_error() -> None:
    """An empty ``star_psf_sizes`` raises ValueError, not UnboundLocalError."""
    inst = _ConcreteObsInst()
    inst._inst_config = {'star_psf_sizes': {}}
    with pytest.raises(ValueError, match='star_psf_sizes is empty'):
        inst.star_psf_size(_make_star(4.0))


def test_star_psf_size_returns_int_tuple() -> None:
    """The selected size is coerced to a 2-element int tuple."""
    inst = _ConcreteObsInst()
    inst._inst_config = {'star_psf_sizes': {10: [7.0, 7.0]}}
    result = inst.star_psf_size(_make_star(4.0))
    assert result == (7, 7)


def test_star_psf_size_wrong_length_raises() -> None:
    """A non-2-element size entry trips the length assertion."""
    inst = _ConcreteObsInst()
    inst._inst_config = {'star_psf_sizes': {10: [7, 7, 7]}}
    with pytest.raises(AssertionError, match='must have 2 elements'):
        inst.star_psf_size(_make_star(4.0))


def test_shutter_mode_defaults_to_none() -> None:
    """An instrument whose host exposes no shutter mode reports None."""
    assert _ConcreteObsInst().shutter_mode is None


def test_concrete_observation_must_set_its_own_model_source() -> None:
    """A concrete class that only inherits ``model_source`` is refused at definition.

    Without the check, an instrument that forgot it would load its images and then fail
    every one of them when the models are built.
    """
    with pytest.raises(TypeError, match='_Forgetful must set model_source'):

        class _Forgetful(_ConcreteObsInst):
            pass
