"""Tests for ``spindoctor.nav_orchestrator.image_classifier_result.NavImageClassifierResult``."""

from spindoctor.nav_orchestrator.image_classifier_result import NavImageClassifierResult


def test_navimageclassifierresult_clean_default_flags_empty() -> None:
    """A clean image has an empty flags list by default."""
    result = NavImageClassifierResult(
        image_class='clean',
        saturation_frac=0.0,
        missing_frac=0.0,
        unread_line_frac=0.0,
        noise_sigma=1.0,
        max_dn=4000.0,
    )
    assert result.flags == []


def test_navimageclassifierresult_noisy_clean_image() -> None:
    """``noisy`` is a flag, not a class — clean+noisy is the canonical form."""
    result = NavImageClassifierResult(
        image_class='clean',
        saturation_frac=0.0,
        missing_frac=0.0,
        unread_line_frac=0.0,
        noise_sigma=15.0,
        max_dn=4000.0,
        flags=['noisy'],
    )
    assert result.image_class == 'clean'
    assert 'noisy' in result.flags


def test_navimageclassifierresult_corrupt_image() -> None:
    """A corrupt image classifier result has its own class."""
    result = NavImageClassifierResult(
        image_class='corrupt',
        saturation_frac=0.0,
        missing_frac=0.0,
        unread_line_frac=0.0,
        noise_sigma=0.0,
        max_dn=0.0,
    )
    assert result.image_class == 'corrupt'
