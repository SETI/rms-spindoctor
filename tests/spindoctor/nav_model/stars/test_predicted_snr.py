"""Tests for ``spindoctor.nav_model.stars.predicted_snr``."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import pytest
from psfmodel import GaussianPSF
from starcat import SCLASS_TO_B_MINUS_V as CANONICAL_SCLASS_TO_B_MINUS_V

from spindoctor.nav_model.stars.predicted_snr import (
    SCLASS_TO_B_MINUS_V,
    integrated_signal_dn,
    predicted_snr,
    psf_aperture_pixels,
    psf_sigma_px,
)


@dataclass
class _FakeStar:
    """Minimal star stand-in for predicted_snr tests."""

    vmag: float | None


class _FakeSigmaPSF:
    """Minimal PSF stand-in exposing only a single ``sigma`` attribute."""

    def __init__(self, sigma: float) -> None:
        self.sigma = sigma


class _FakeAxisPSF:
    """PSF stand-in mimicking ``psfmodel.GaussianPSF``'s per-axis attrs."""

    def __init__(self, sigma_x: float, sigma_y: float) -> None:
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y


class _FakeFwhmPSF:
    """PSF stand-in that exposes ``fwhm()`` but no sigma attributes."""

    def __init__(self, fwhm: float) -> None:
        self._fwhm = fwhm

    def fwhm(self) -> float:
        """Return the constant FWHM passed at construction."""
        return self._fwhm


def test_psf_sigma_px_reads_axis_sigmas() -> None:
    """``psf_sigma_px`` averages ``sigma_x`` and ``sigma_y`` when both are present."""
    assert psf_sigma_px(_FakeAxisPSF(sigma_x=1.0, sigma_y=2.0)) == 1.5  # type: ignore[arg-type]


def test_psf_sigma_px_supports_real_gaussian_psf() -> None:
    """``psfmodel.GaussianPSF(sigma=...)`` is read via ``sigma_x`` / ``sigma_y``.

    Regression: the legacy implementation looked for a ``sigma`` attribute
    on every PSF, but the shipped ``GaussianPSF`` exposes only per-axis
    sigmas, so every Cassini star-feature extraction raised
    ``AttributeError`` before the helper handled both shapes.
    """
    assert psf_sigma_px(GaussianPSF(sigma=0.54)) == pytest.approx(0.54)
    assert psf_sigma_px(GaussianPSF(sigma=0.77)) == pytest.approx(0.77)


def test_psf_sigma_px_reads_sigma_attribute() -> None:
    """``psf_sigma_px`` falls back to a single ``sigma`` attribute."""
    assert psf_sigma_px(_FakeSigmaPSF(sigma=1.25)) == 1.25  # type: ignore[arg-type]


def test_psf_sigma_px_falls_back_to_fwhm() -> None:
    """When no sigma attrs are present, the helper converts FWHM via the Gaussian factor."""
    psf = _FakeFwhmPSF(fwhm=2.3548200450309493)
    assert math.isclose(psf_sigma_px(psf), 1.0, rel_tol=1e-12)  # type: ignore[arg-type]


def test_psf_sigma_px_raises_when_neither_sigma_nor_fwhm_present() -> None:
    """The helper raises ``AttributeError`` for an unsupported PSF subclass."""

    class _NoApiPSF:
        pass

    with pytest.raises(AttributeError, match='exposes neither sigma_x/sigma_y'):
        psf_sigma_px(_NoApiPSF())  # type: ignore[arg-type]


def test_psf_aperture_pixels_returns_4_pi_sigma_squared() -> None:
    """The aperture matches the Gaussian noise-equivalent area formula."""
    expected = 4.0 * math.pi * 0.7 * 0.7
    assert math.isclose(psf_aperture_pixels(0.7), expected, rel_tol=1e-12)


def test_psf_aperture_pixels_rejects_non_positive_sigma() -> None:
    """A zero or negative sigma raises ``ValueError`` with the offending value."""
    with pytest.raises(ValueError, match='sigma_px_value must be > 0'):
        psf_aperture_pixels(0.0)


def test_integrated_signal_dn_uses_canonical_anchor() -> None:
    """The flux-to-DN scaling matches ``2.512 ** -(vmag - 4)`` at vmag=4."""
    star: Any = _FakeStar(vmag=4.0)
    assert math.isclose(integrated_signal_dn(star, mag_offset=0.0), 1.0, rel_tol=1e-12)


def test_integrated_signal_dn_applies_mag_offset() -> None:
    """The mag offset shifts the catalog magnitude before the flux conversion."""
    star: Any = _FakeStar(vmag=4.0)
    expected = 2.512 ** -(4.0 + 0.5 - 4.0)
    assert math.isclose(integrated_signal_dn(star, mag_offset=0.5), expected, rel_tol=1e-12)


def test_integrated_signal_dn_returns_zero_when_vmag_missing() -> None:
    """Stars without a catalog vmag produce zero predicted DN."""
    star: Any = _FakeStar(vmag=None)
    assert integrated_signal_dn(star, mag_offset=0.0) == 0.0


def test_predicted_snr_matches_design_formula() -> None:
    """Predicted SNR matches the design's matched-filter form for known inputs."""
    star: Any = _FakeStar(vmag=4.0)
    psf: Any = _FakeSigmaPSF(sigma=1.0)
    image_noise_sigma = 0.5
    sig = 1.0
    aperture = 4.0 * math.pi
    expected = sig / math.sqrt(sig + image_noise_sigma**2 * aperture)
    out = predicted_snr(star, psf=psf, image_noise_sigma=image_noise_sigma)
    assert math.isclose(out, expected, rel_tol=1e-12)


def test_predicted_snr_returns_zero_for_missing_vmag() -> None:
    """A star with no vmag produces zero SNR (no signal)."""
    star: Any = _FakeStar(vmag=None)
    psf: Any = _FakeSigmaPSF(sigma=1.0)
    assert predicted_snr(star, psf=psf, image_noise_sigma=0.5) == 0.0


def test_predicted_snr_raises_for_non_positive_noise_sigma() -> None:
    """A non-positive noise sigma raises ``ValueError`` naming the bad value."""
    star: Any = _FakeStar(vmag=4.0)
    psf: Any = _FakeSigmaPSF(sigma=1.0)
    with pytest.raises(ValueError, match='image_noise_sigma must be > 0'):
        predicted_snr(star, psf=psf, image_noise_sigma=0.0)


def test_predicted_snr_unity_signal_scale_is_no_op() -> None:
    """signal_dn_to_image_unit_scale=1.0 reproduces the default behavior."""
    star: Any = _FakeStar(vmag=4.0)
    psf: Any = _FakeSigmaPSF(sigma=1.0)
    base = predicted_snr(star, psf=psf, image_noise_sigma=0.5)
    explicit = predicted_snr(
        star, psf=psf, image_noise_sigma=0.5, signal_dn_to_image_unit_scale=1.0
    )
    assert math.isclose(base, explicit, rel_tol=1e-12)


def test_predicted_snr_calibrated_if_scale_recovers_dn_equivalent_snr() -> None:
    """Scaling both the noise sigma and scale by k leaves the SNR unchanged.

    This is the sanity check from the unit-handling fix: an IF-units
    image has a tiny ``image_noise_sigma`` and an equally tiny
    ``signal_dn_to_image_unit_scale``; their ratio is the DN-equivalent
    background sigma, so the SNR matches the raw-DN result on the same
    underlying observation.
    """
    star: Any = _FakeStar(vmag=4.0)
    psf: Any = _FakeSigmaPSF(sigma=1.0)
    raw_dn_snr = predicted_snr(star, psf=psf, image_noise_sigma=0.5)
    scale = 5.0e-7
    if_snr = predicted_snr(
        star,
        psf=psf,
        image_noise_sigma=0.5 * scale,
        signal_dn_to_image_unit_scale=scale,
    )
    assert math.isclose(raw_dn_snr, if_snr, rel_tol=1e-12)


def test_predicted_snr_pleiades_recovers_above_gate_on_calibrated_if() -> None:
    """A bright star on a calibrated_if image clears the STAR reliability gate.

    Worked example from Phase 10 / Part F: a Pleiades-class star
    (``vmag~3``, ``mag_offset=-10`` after Phase 10 calibration) on a
    Cassini ISS NAC CALIB image (``image_noise_sigma~1e-7`` I/F,
    ``signal_dn_to_image_unit_scale~5e-7``) must produce an SNR well
    above the implicit ``snr=5`` floor or the reliability gate drops
    every catalog star.  Without the unit conversion the SNR would
    collapse to ``sqrt(signal_dn)`` and most stars would be gated out.
    """
    star: Any = _FakeStar(vmag=3.0)
    psf: Any = _FakeSigmaPSF(sigma=0.54)
    snr = predicted_snr(
        star,
        psf=psf,
        image_noise_sigma=1.0e-7,
        mag_offset=-10.0,
        signal_dn_to_image_unit_scale=5.0e-7,
    )
    assert snr > 5.0


def test_predicted_snr_raises_for_non_positive_signal_scale() -> None:
    """A non-positive signal scale raises ``ValueError``."""
    star: Any = _FakeStar(vmag=4.0)
    psf: Any = _FakeSigmaPSF(sigma=1.0)
    with pytest.raises(ValueError, match='signal_dn_to_image_unit_scale'):
        predicted_snr(
            star,
            psf=psf,
            image_noise_sigma=0.5,
            signal_dn_to_image_unit_scale=0.0,
        )


def test_sclass_to_b_minus_v_re_export_matches_starcat() -> None:
    """The re-exported lookup table is the same dict as ``starcat``'s."""
    assert SCLASS_TO_B_MINUS_V is CANONICAL_SCLASS_TO_B_MINUS_V
