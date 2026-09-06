"""Integration-style tests for ``NavModelStars`` using stub-based obs.

These tests exercise the end-to-end ``create_model`` /
``to_features`` / ``to_annotations`` contract by injecting a fake
catalog reduction plus fake mask context, so the path that builds
``NavFeature`` instances, computes the CRLB covariance, and emits
annotations is covered without requiring real SPICE / catalog data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast
from unittest import mock

import numpy as np
import pytest

from spindoctor.annotation import Annotations
from spindoctor.feature.constants import MIN_ANISOTROPIC_SMEAR_PX
from spindoctor.feature.feature_type import NavFeatureType
from spindoctor.feature.flags import StarFlags
from spindoctor.feature.geometry import StarGeometry
from spindoctor.nav_model.stars.nav_model_stars import (
    SNR_REF,
    NavModelStars,
    _crlb_covariance,
    _reliability_from_snr,
    _safe_mask_lookup,
    _snr_reason_score,
    _star_feature_id,
)
from spindoctor.nav_model.stars.smeared_psf import compute_smear_vector_px
from spindoctor.nav_orchestrator.nav_context import NavContext
from spindoctor.support.types import MutableStar


@dataclass
class _FakeMutableStar:
    """Minimal star record exposing the fields ``NavModelStars`` reads."""

    catalog_name: str = 'UCAC4'
    unique_number: int | None = 12345
    pretty_name: str = '12345'
    name: str = ''
    vmag: float | None = 5.0
    photometry_corrected: bool = False
    photometry_saturated: bool = False
    spectral_class: str = 'G0'
    psf_size: tuple[int, int] = (5, 5)
    u: float = 50.0
    v: float = 60.0
    move_u: float = 0.0
    move_v: float = 0.0
    ra_pm: float = 0.0
    dec_pm: float = 0.0
    conflicts: str = ''
    dn: float = 1.0


class _FakePSF:
    """PSF stand-in returning a Gaussian sigma."""

    def __init__(self, sigma: float = 1.0) -> None:
        self.sigma = sigma


class _FakeStarsConfig:
    """Stand-in for ``config.stars`` exposing only the fields we need."""

    label_font: str = 'Liberation Mono'
    label_font_size: int = 10
    label_font_color: tuple[int, int, int] = (255, 0, 0)
    label_star_color: tuple[int, int, int] = (255, 255, 0)
    max_smear: float = 100.0
    min_predicted_snr: float = 0.0


@dataclass
class _FakeContext:
    """Per-image NavContext stand-in carrying the masks NavModelStars consults."""

    image_noise_sigma: float = 0.5
    saturation_mask_ext: np.ndarray = field(
        default_factory=lambda: np.zeros((128, 128), dtype=bool)
    )
    cosmic_ray_mask_ext: np.ndarray = field(
        default_factory=lambda: np.zeros((128, 128), dtype=bool)
    )


class _FakeObs:
    """Observation stand-in covering the methods ``NavModelStars`` reads."""

    def __init__(self, *, extfov_margin: int = 14, star_max_vmag: float = 12.0) -> None:
        self.extfov_margin_v = extfov_margin
        self.extfov_margin_u = extfov_margin
        self.extdata_shape_vu = (128, 128)
        self.data_shape_v = 100
        self.data_shape_u = 100
        self._psf = _FakePSF(sigma=1.0)
        self._star_max_vmag = star_max_vmag

    def star_psf(self) -> _FakePSF:
        """Return the fake PSF shared by every test star."""
        return self._psf

    def star_max_usable_vmag(self) -> float:
        """Return the limiting magnitude used by the star gate."""
        return self._star_max_vmag

    def make_extfov_false(self) -> np.ndarray:
        """Return a False-filled boolean array of the extfov shape."""
        return np.zeros(self.extdata_shape_vu, dtype=bool)

    def make_extfov_zeros(self) -> np.ndarray:
        """Return a zero-filled float64 array of the extfov shape."""
        return np.zeros(self.extdata_shape_vu, dtype=np.float64)

    def clip_extfov(self, u: int, v: int) -> tuple[int, int]:
        """Clip ``(u, v)`` inside the extfov rectangle."""
        return (
            int(np.clip(u, 0, self.extdata_shape_vu[1] - 1)),
            int(np.clip(v, 0, self.extdata_shape_vu[0] - 1)),
        )


def _make_model(*, star_max_vmag: float = 12.0) -> tuple[NavModelStars, _FakeObs]:
    """Build a NavModelStars instance with the fake obs and stars config."""
    obs = _FakeObs(star_max_vmag=star_max_vmag)
    config = mock.Mock()
    config.stars = _FakeStarsConfig()
    model = NavModelStars('stars', cast(Any, obs), config=config)
    # Bypass the catalog reduction in tests by injecting our own list.
    return model, obs


def test_star_feature_id_uses_catalog_and_unique_number() -> None:
    """``feature_id`` is ``star:<CATALOG>:<unique_number>``."""
    star = _FakeMutableStar(catalog_name='ybsc', unique_number=42, pretty_name='Sirius')
    assert _star_feature_id(cast(MutableStar, star)) == 'star:YBSC:42'


def test_star_feature_id_falls_back_to_pretty_name() -> None:
    """When ``unique_number`` is ``None`` the helper uses ``pretty_name``."""
    star = _FakeMutableStar(catalog_name='ucac4', unique_number=None, pretty_name='Vega')
    assert _star_feature_id(cast(MutableStar, star)) == 'star:UCAC4:Vega'


def test_crlb_covariance_isotropic_below_smear_threshold() -> None:
    """Below the anisotropy threshold the covariance is ``sigma_psf^2 / SNR * I``."""
    cov = _crlb_covariance(snr=10.0, sigma_psf=1.0, move_v=0.0, move_u=0.0)
    expected_sigma2 = 1.0 / 10.0
    expected = expected_sigma2 * np.eye(2)
    assert np.allclose(cov, expected, atol=1e-12)


def test_crlb_covariance_anisotropic_along_smear() -> None:
    """The major axis of the covariance lies along the smear vector."""
    cov = _crlb_covariance(snr=10.0, sigma_psf=1.0, move_v=4.0, move_u=0.0)
    # Eigenvalues of cov; the larger one is along the smear axis.
    eigvals, eigvecs = np.linalg.eigh(cov)
    largest_axis = eigvecs[:, np.argmax(eigvals)]
    # Major axis should lie along v (since smear is purely along v).
    assert abs(largest_axis[0]) > abs(largest_axis[1])


def test_crlb_covariance_smear_threshold_boundary() -> None:
    """Smear exactly at MIN_ANISOTROPIC_SMEAR_PX is still treated isotropically.

    The threshold check is strict ``<``, so a smear of MIN-eps falls
    into the isotropic branch.
    """
    smear = MIN_ANISOTROPIC_SMEAR_PX - 0.001
    cov = _crlb_covariance(snr=10.0, sigma_psf=1.0, move_v=smear, move_u=0.0)
    assert np.allclose(cov, cov[0, 0] * np.eye(2), atol=1e-12)


def test_crlb_covariance_zero_snr_returns_huge() -> None:
    """A zero or negative SNR returns a huge isotropic covariance."""
    cov = _crlb_covariance(snr=0.0, sigma_psf=1.0, move_v=0.0, move_u=0.0)
    assert cov[0, 0] == pytest.approx(1e6)


def test_crlb_covariance_rejects_zero_psf_sigma() -> None:
    """A zero or negative PSF sigma raises ``ValueError`` naming the bad value."""
    with pytest.raises(ValueError, match='sigma_psf must be > 0'):
        _crlb_covariance(snr=5.0, sigma_psf=0.0, move_v=0.0, move_u=0.0)


def test_reliability_from_snr_zero_when_in_body() -> None:
    """A body-occluded star has zero reliability."""
    out = _reliability_from_snr(snr=20.0, in_body=True, in_ring=False, in_saturation=False)
    assert out == 0.0


def test_reliability_from_snr_zero_when_in_ring() -> None:
    """A ring-occluded star has zero reliability."""
    out = _reliability_from_snr(snr=20.0, in_body=False, in_ring=True, in_saturation=False)
    assert out == 0.0


def test_reliability_from_snr_zero_when_in_saturation() -> None:
    """A saturated / cosmic-ray pixel produces zero reliability."""
    out = _reliability_from_snr(snr=20.0, in_body=False, in_ring=False, in_saturation=True)
    assert out == 0.0


def test_reliability_from_snr_monotone_in_snr() -> None:
    """For valid stars the reliability is monotone increasing in SNR."""
    low = _reliability_from_snr(snr=5.0, in_body=False, in_ring=False, in_saturation=False)
    high = _reliability_from_snr(snr=20.0, in_body=False, in_ring=False, in_saturation=False)
    assert high > low


def test_snr_reason_score_uses_min_snr_when_set() -> None:
    """When a configured floor is non-zero the score is ``snr/min_snr``."""
    assert _snr_reason_score(snr=4.0, min_snr=8.0) == pytest.approx(0.5)


def test_snr_reason_score_caps_at_one() -> None:
    """The breakdown score saturates at 1.0."""
    assert _snr_reason_score(snr=200.0, min_snr=8.0) == 1.0


def test_snr_reason_score_default_centre_50() -> None:
    """When no floor is configured the score saturates at SNR=50."""
    assert _snr_reason_score(snr=50.0, min_snr=0.0) == 1.0
    assert _snr_reason_score(snr=25.0, min_snr=0.0) == pytest.approx(0.5)


def test_to_features_empty_when_no_stars() -> None:
    """``to_features`` returns ``[]`` when the reduced star list is empty."""
    model, _obs = _make_model()
    model._stars = []
    out = model.to_features(cast(NavContext, _FakeContext()))
    assert out == []


def test_to_features_emits_one_feature_per_star() -> None:
    """One STAR feature is emitted per reduced star above the SNR floor."""
    model, _obs = _make_model()
    star = _FakeMutableStar(vmag=4.0, u=50.0, v=60.0)
    model._stars = [cast(MutableStar, star)]
    features = model.to_features(cast(NavContext, _FakeContext()))
    assert len(features) == 1
    feat = features[0]
    assert feat.feature_type is NavFeatureType.STAR
    assert isinstance(feat.geometry, StarGeometry)
    assert isinstance(feat.flags, StarFlags)
    assert feat.feature_id == 'star:UCAC4:12345'


def test_to_features_skips_star_fainter_than_limit() -> None:
    """A star fainter than ``obs.star_max_usable_vmag()`` is dropped."""
    model, _obs = _make_model(star_max_vmag=8.0)
    faint = _FakeMutableStar(vmag=9.0)
    model._stars = [cast(MutableStar, faint)]
    assert model.to_features(cast(NavContext, _FakeContext())) == []


def test_to_features_keeps_star_brighter_than_limit() -> None:
    """A star brighter than the limiting magnitude is kept."""
    model, _obs = _make_model(star_max_vmag=8.0)
    bright = _FakeMutableStar(vmag=4.0)
    model._stars = [cast(MutableStar, bright)]
    assert len(model.to_features(cast(NavContext, _FakeContext()))) == 1


def test_to_features_skips_star_without_vmag() -> None:
    """A star with no catalog magnitude is dropped by the magnitude gate."""
    model, _obs = _make_model(star_max_vmag=8.0)
    no_mag = _FakeMutableStar(vmag=None)
    model._stars = [cast(MutableStar, no_mag)]
    assert model.to_features(cast(NavContext, _FakeContext())) == []


def test_to_features_star_at_limit_gets_snr_ref() -> None:
    """A star exactly at the limiting magnitude gets ``snr_eff == SNR_REF``."""
    model, _obs = _make_model(star_max_vmag=8.0)
    at_limit = _FakeMutableStar(vmag=8.0)
    model._stars = [cast(MutableStar, at_limit)]
    feat = model.to_features(cast(NavContext, _FakeContext()))[0]
    assert isinstance(feat.flags, StarFlags)
    assert feat.flags.predicted_snr == pytest.approx(SNR_REF)


def test_to_features_brighter_star_has_higher_effective_snr() -> None:
    """The effective SNR (and reliability) rises with magnitude margin.

    Replaces the old DN-based SNR-floor skip test: detectability now comes
    from how far below the limiting magnitude the star sits, not from a
    DN-derived photometric SNR.
    """
    model, _obs = _make_model(star_max_vmag=8.0)
    dim = _FakeMutableStar(unique_number=1, vmag=7.0)
    bright = _FakeMutableStar(unique_number=2, vmag=3.0, u=70.0, v=80.0)
    model._stars = [cast(MutableStar, dim), cast(MutableStar, bright)]
    feats = model.to_features(cast(NavContext, _FakeContext()))
    by_id = {f.feature_id: f for f in feats}
    dim_feat = by_id['star:UCAC4:1']
    bright_feat = by_id['star:UCAC4:2']
    assert isinstance(dim_feat.flags, StarFlags)
    assert isinstance(bright_feat.flags, StarFlags)
    assert bright_feat.flags.predicted_snr > dim_feat.flags.predicted_snr
    assert bright_feat.reliability > dim_feat.reliability
    # The CRLB covariance is derived from the same effective SNR, so the
    # brighter star's position is tighter (smaller variance).
    bright_cov = bright_feat.position_cov_px
    dim_cov = dim_feat.position_cov_px
    assert bright_cov is not None
    assert dim_cov is not None
    assert bright_cov[0, 0] < dim_cov[0, 0]


def test_to_features_returns_empty_when_noise_zero() -> None:
    """A zero or negative noise sigma triggers an empty feature list."""
    model, _obs = _make_model()
    model._stars = [cast(MutableStar, _FakeMutableStar())]
    ctx = _FakeContext(image_noise_sigma=0.0)
    assert model.to_features(cast(NavContext, ctx)) == []


def test_to_features_skips_when_smear_exceeds_max() -> None:
    """A star whose smear exceeds ``stars.max_smear`` is dropped from the feature list."""
    model, _obs = _make_model()
    cfg = mock.Mock()
    cfg.stars = _FakeStarsConfig()
    cfg.stars.max_smear = 1.0
    model._stars_config = cfg.stars
    star = _FakeMutableStar(move_v=10.0, move_u=10.0)
    model._stars = [cast(MutableStar, star)]
    assert model.to_features(cast(NavContext, _FakeContext())) == []


def test_to_features_marks_in_body_when_conflict_set() -> None:
    """A star tagged with a BODY conflict is emitted with ``in_body_silhouette=True``."""
    model, _obs = _make_model()
    star = _FakeMutableStar(conflicts='BODY: MIMAS')
    model._stars = [cast(MutableStar, star)]
    feat = model.to_features(cast(NavContext, _FakeContext()))[0]
    assert isinstance(feat.flags, StarFlags)
    assert feat.flags.in_body_silhouette is True
    assert feat.reliability == 0.0


def test_to_features_keeps_saturated_star_fainter_than_limit() -> None:
    """A saturated star fainter than the limit is kept, not dropped.

    Its recorded magnitude is an untrusted, too-faint saturated reading, so
    the faint gate must not reject it.
    """
    model, _obs = _make_model(star_max_vmag=8.0)
    saturated = _FakeMutableStar(vmag=9.0, photometry_saturated=True)
    model._stars = [cast(MutableStar, saturated)]
    assert len(model.to_features(cast(NavContext, _FakeContext()))) == 1


def test_to_features_saturated_star_gets_at_least_snr_ref() -> None:
    """A saturated star fainter than the limit is treated as at-limit bright."""
    model, _obs = _make_model(star_max_vmag=8.0)
    saturated = _FakeMutableStar(vmag=9.0, photometry_saturated=True)
    model._stars = [cast(MutableStar, saturated)]
    feat = model.to_features(cast(NavContext, _FakeContext()))[0]
    assert isinstance(feat.flags, StarFlags)
    assert feat.flags.predicted_snr == pytest.approx(SNR_REF)


def test_to_features_surfaces_photometry_saturated_flag() -> None:
    """The saturated-photometry flag is carried onto the emitted feature."""
    model, _obs = _make_model(star_max_vmag=8.0)
    saturated = _FakeMutableStar(vmag=6.5, photometry_saturated=True)
    model._stars = [cast(MutableStar, saturated)]
    feat = model.to_features(cast(NavContext, _FakeContext()))[0]
    assert isinstance(feat.flags, StarFlags)
    assert feat.flags.photometry_saturated is True


def test_to_annotations_empty_when_no_stars() -> None:
    """``to_annotations`` returns an empty collection when no stars are reduced."""
    model, _obs = _make_model()
    model._stars = []
    annotations = model.to_annotations(cast(NavContext, _FakeContext()))
    assert isinstance(annotations, Annotations)
    assert len(annotations.annotations) == 0


def test_instances_for_obs_returns_single_stars_model() -> None:
    """``NavModelStars.instances_for_obs`` returns one model per observation."""
    obs = _FakeObs()
    instances = NavModelStars.instances_for_obs(cast(Any, obs))
    assert len(instances) == 1
    assert instances[0].name == 'stars'


def test_create_model_populates_metadata_and_star_count(monkeypatch: pytest.MonkeyPatch) -> None:
    """``create_model`` records start / end times, elapsed, and star count.

    The catalog reduction and conflict marking are monkeypatched to
    return a controlled list so the metadata path is exercised without
    needing real catalogs or SPICE.
    """
    model, _obs = _make_model()
    star = _FakeMutableStar()
    monkeypatch.setattr(
        'spindoctor.nav_model.stars.nav_model_stars.reduce_catalogs',
        lambda _obs, _config: [cast(MutableStar, star)],
    )
    monkeypatch.setattr(
        'spindoctor.nav_model.stars.nav_model_stars.mark_body_and_ring_conflicts',
        lambda _obs, _config, _stars: None,
    )
    monkeypatch.setattr(
        'spindoctor.nav_model.stars.nav_model_stars.compute_smear_vector_px',
        lambda _obs: (0.0, 0.0),
    )
    model.create_model()
    assert model.metadata['star_count'] == 1
    assert model.metadata['elapsed_time_sec'] >= 0.0
    assert isinstance(model.metadata['stars'], list)
    assert model.metadata['stars'][0]['catalog_name'] == 'UCAC4'
    assert len(model.stars) == 1


def test_to_annotations_renders_overlay_for_usable_stars() -> None:
    """``to_annotations`` paints star-box overlays + label entries for usable stars.

    Exercises the ``_build_annotations`` path (overlay drawing, avoid
    mask, label-position generation, and ``Annotation`` construction).
    """
    model, _obs = _make_model()
    star = _FakeMutableStar(unique_number=1, vmag=4.0, u=50.0, v=60.0)
    model._stars = [cast(MutableStar, star)]
    annotations = model.to_annotations(cast(NavContext, _FakeContext()))
    # One ``Annotation`` whose text_info carries one entry per star.
    assert len(annotations.annotations) == 1
    annotation = annotations.annotations[0]
    assert len(annotation.text_info_list) == 1


def test_to_annotations_skips_stars_blocked_by_body_or_ring() -> None:
    """A star tagged with a BODY or RING conflict is excluded from the overlay."""
    model, _obs = _make_model()
    star_ok = _FakeMutableStar(unique_number=1, vmag=4.0, u=50.0, v=60.0)
    star_blocked = _FakeMutableStar(
        unique_number=2,
        vmag=5.0,
        u=60.0,
        v=70.0,
        conflicts='BODY: MIMAS',
    )
    model._stars = [cast(MutableStar, star_ok), cast(MutableStar, star_blocked)]
    annotations = model.to_annotations(cast(NavContext, _FakeContext()))
    annotation = annotations.annotations[0]
    # Only the unblocked star contributes a label.
    assert len(annotation.text_info_list) == 1


def test_to_annotations_keeps_stars_with_star_only_conflict() -> None:
    """A star tagged with the ``'STAR'`` conflict is still labelled."""
    model, _obs = _make_model()
    star = _FakeMutableStar(unique_number=1, vmag=4.0, u=50.0, v=60.0, conflicts='STAR')
    model._stars = [cast(MutableStar, star)]
    annotations = model.to_annotations(cast(NavContext, _FakeContext()))
    assert len(annotations.annotations[0].text_info_list) == 1


def test_smear_from_an_obs_that_cannot_report_its_centre_is_not_silently_zero() -> None:
    """A zero smear vector is a measurement, not a missing one.

    It feeds the star covariance, so an obs whose centre lookup is broken must
    fail rather than report that the camera never moved.
    """

    class _NoCentreObs:
        """Minimal obs stand-in lacking ``center_ra_dec``."""

    with pytest.raises(AttributeError):
        compute_smear_vector_px(cast(Any, _NoCentreObs()))


def test_safe_mask_lookup_returns_false_for_none_mask() -> None:
    """``_safe_mask_lookup`` returns False when ``mask`` is None."""
    assert _safe_mask_lookup(None, 5.0, 5.0) is False


def test_safe_mask_lookup_returns_false_for_empty_mask() -> None:
    """``_safe_mask_lookup`` returns False for a 0-element mask."""
    assert _safe_mask_lookup(np.zeros(0), 5.0, 5.0) is False


def test_safe_mask_lookup_clamps_indices_to_bounds() -> None:
    """Out-of-range coordinates are clamped to the mask's shape."""
    mask = np.zeros((10, 10), dtype=bool)
    mask[9, 9] = True
    assert _safe_mask_lookup(mask, 100.0, 100.0) is True
    mask2 = np.zeros((10, 10), dtype=bool)
    mask2[0, 0] = True
    assert _safe_mask_lookup(mask2, -100.0, -100.0) is True
