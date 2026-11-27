"""Unit tests for the ``ObsInst`` instrument mix-in helpers."""

from typing import Any

import pytest
from starcat import Star
from tests.camera_sections import camera_mapping, raw_camera

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


@pytest.mark.parametrize(
    ('vmag', 'size'), [(4.0, (3, 3)), (99.0, (7, 7))], ids=['brighter', 'fainter-than-all']
)
def test_star_psf_size_follows_the_magnitude_bins(vmag: float, size: tuple[int, int]) -> None:
    """A star takes the first bin it is brighter than, or the last when fainter than all."""
    inst = _ConcreteObsInst()
    inst._camera_block = raw_camera(star_psf_sizes={5: (3, 3), 10: (7, 7)})
    assert inst.star_psf_size(_make_star(vmag)) == size


def test_a_camera_block_held_as_a_mapping_is_parsed() -> None:
    """An observation that holds its camera block as a mapping reads it as the type."""
    inst = _ConcreteObsInst()
    inst._inst_config = camera_mapping(raw_camera())
    assert inst.inst_config == raw_camera()


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
