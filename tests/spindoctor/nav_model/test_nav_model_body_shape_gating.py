"""Tests for the highly-irregular resolved-body shape-feature gate.

A body tagged ``highly_irregular`` in ``config_220_body_shape.yaml`` has no
usable ellipsoid.  Once it is resolved beyond a few pixels the model must
suppress the shape-based features (LIMB_ARC / TERMINATOR_ARC / BODY_DISC) and
fall back to the point-like BODY_BLOB.  Bodies tagged merely ``irregular`` are
unchanged.

These tests drive ``NavModelBody.to_features`` directly with hand-populated
render state so they do not need a live ``oops`` backplane.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np

from spindoctor.config import Config
from spindoctor.feature.feature_type import NavFeatureType
from spindoctor.nav_model.nav_model_body import NavModelBody, _PolylineSampler


class _FakePSF:
    """PSF stand-in exposing the per-axis sigma ``psf_sigma_px`` reads."""

    sigma_x: float = 1.0
    sigma_y: float = 1.0


class _FakeObs:
    """Observation stand-in for the body-model feature emitters."""

    extfov_margin_v: int = 0
    extfov_margin_u: int = 0

    def star_psf(self) -> _FakePSF:
        """Return a fixed isotropic PSF."""
        return _FakePSF()


def _fake_context() -> Any:
    """Return a NavContext-like namespace the blob emitter can read."""
    rng = np.random.default_rng(1234)
    image = 100.0 + rng.standard_normal((20, 20))
    return SimpleNamespace(
        image_ext=image,
        sensor_mask_ext=np.ones((20, 20), dtype=bool),
        saturation_mask_ext=np.zeros((20, 20), dtype=bool),
        cosmic_ray_mask_ext=np.zeros((20, 20), dtype=bool),
        image_noise_sigma=1.0,
    )


def _limb_sampler(n: int) -> _PolylineSampler:
    """Return a limb sampler of ``n`` well-resolved vertices."""
    return _PolylineSampler(
        vertices_vu=np.zeros((n, 2), dtype=np.float64),
        normals_vu=np.tile(np.array([1.0, 0.0]), (n, 1)),
        incidence_rad=np.zeros(n, dtype=np.float64),
        km_per_pixel=np.full(n, 10.0, dtype=np.float64),
        total_vertices=n,
    )


def _make_body_model(*, body_name: str, diameter_px: float) -> NavModelBody:
    """Build a NavModelBody with render state populated for ``to_features``.

    The limb sampler carries 40 well-resolved vertices so a LIMB_ARC would be
    emitted whenever the shape gate allows it; the disc and terminator gates
    are held closed (zero visible-lit fraction, zero phase factor) so an
    unsuppressed body emits exactly one LIMB_ARC and the test isolates the
    shape gate.
    """
    obs: Any = _FakeObs()
    model = NavModelBody(f'body:{body_name}', obs, body_name, config=Config())
    body_mask = np.zeros((20, 20), dtype=bool)
    vv, uu = np.indices((20, 20))
    body_mask[np.hypot(vv - 10.0, uu - 10.0) <= 6.0] = True
    model_img = np.where(body_mask, 1.0, 0.0).astype(np.float64)
    model._body_mask = body_mask
    model._model_img = model_img
    model._lit_pixel_count = int(np.count_nonzero(body_mask))
    model._limb_sampler = _limb_sampler(40)
    model._terminator_sampler = _limb_sampler(0)
    model._km_per_pixel_at_limb = 10.0
    model._predicted_diameter_px = diameter_px
    model._predicted_center_vu = (10.0, 10.0)
    model._bbox_extfov_vu = (4, 4, 16, 16)
    model._subject_range_km = 1000.0
    model._visible_lit_fraction = 0.0
    model._overflow_fraction = 0.0
    model._phase_angle_factor = 0.0
    model._metadata['phase_angle_deg'] = 0.0
    return model


def _emitted_types(model: NavModelBody) -> set[NavFeatureType]:
    """Return the set of feature types emitted by ``to_features``."""
    features = model.to_features(_fake_context())
    return {f.feature_type for f in features}


def test_highly_irregular_resolved_suppresses_limb() -> None:
    """A resolved Hyperion emits no LIMB_ARC (highly_irregular shape gate)."""
    model = _make_body_model(body_name='HYPERION', diameter_px=40.0)
    assert NavFeatureType.LIMB_ARC not in _emitted_types(model)


def test_highly_irregular_resolved_suppresses_terminator_and_disc() -> None:
    """A resolved Hyperion emits neither TERMINATOR_ARC nor BODY_DISC."""
    model = _make_body_model(body_name='HYPERION', diameter_px=40.0)
    types = _emitted_types(model)
    assert NavFeatureType.TERMINATOR_ARC not in types
    assert NavFeatureType.BODY_DISC not in types


def test_highly_irregular_resolved_still_emits_blob() -> None:
    """A resolved Hyperion still navigates as a point-like BODY_BLOB."""
    model = _make_body_model(body_name='HYPERION', diameter_px=40.0)
    assert NavFeatureType.BODY_BLOB in _emitted_types(model)


def test_highly_irregular_below_threshold_still_emits_limb() -> None:
    """An unresolved Hyperion (below the resolved threshold) is not suppressed."""
    model = _make_body_model(body_name='HYPERION', diameter_px=2.5)
    assert NavFeatureType.LIMB_ARC in _emitted_types(model)


def test_irregular_body_emits_limb_when_resolved() -> None:
    """A merely-irregular Phoebe keeps its LIMB_ARC when resolved (unchanged path)."""
    model = _make_body_model(body_name='PHOEBE', diameter_px=40.0)
    assert NavFeatureType.LIMB_ARC in _emitted_types(model)


def test_irregular_resolved_emits_witness_blob_with_limb() -> None:
    """A resolved Phoebe emits its LIMB_ARC and a witness BODY_BLOB alongside it."""
    model = _make_body_model(body_name='PHOEBE', diameter_px=40.0)
    types = _emitted_types(model)
    assert NavFeatureType.LIMB_ARC in types


def test_irregular_resolved_co_emits_blob_witness() -> None:
    """The witness blob is co-emitted so the body-witness veto is reachable."""
    model = _make_body_model(body_name='PHOEBE', diameter_px=40.0)
    assert NavFeatureType.BODY_BLOB in _emitted_types(model)
