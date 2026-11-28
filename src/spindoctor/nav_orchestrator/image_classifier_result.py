"""NavImageClassifierResult — output of the image-quality classifier.

Carried on every ``NavResult`` (success or failure) so downstream consumers
can see at a glance which class the input image fell into and which optional
flags were set.
"""

from dataclasses import dataclass, field
from typing import Literal

__all__ = ['NavImageClassifierResult']


ImageClass = Literal[
    'clean',
    'blank',
    'fully_overexposed',
    'mostly_missing_data',
    'corrupt',
]
"""Closed set of image classes the classifier may report.

``clean``, ``blank``, ``fully_overexposed``, and ``mostly_missing_data``
are emitted by ``NavImageClassifier``; ``corrupt`` is set by the caller
when image-file parsing raises.  A truncated readout is not a class of its
own: its unread lines are measured by ``unread_line_frac``, and the image is
``clean`` when enough lines were read out.  Additional classes (e.g.
alternating-lines, ccd-bloom-dominant) are not yet emitted; their detection
paths are tracked separately.
"""

ImageFlag = Literal[
    'partial_dropout',
    'noisy',
]
"""Advisory flags that may appear on a classifier result.

``partial_dropout`` is raised on a clean image when the missing-data fraction,
unread lines included, exceeds the ``partial_dropout_min_frac`` advisory
threshold.  ``noisy`` is raised when the MAD-based
noise sigma exceeds the per-instrument threshold.
"""


@dataclass(frozen=True)
class NavImageClassifierResult:
    """The image-quality classifier's verdict for one image.

    ``noisy`` is a flag, not a class — a noisy-but-clean image is
    ``image_class='clean', flags=['noisy']``.  Images whose file fails
    to read get ``image_class='corrupt'``.

    Parameters:
        image_class: One of the ``ImageClass`` literal values.
        saturation_frac: Fraction of pixels at or above the saturation DN.
        missing_frac: Fraction of pixels equal to the missing-data marker.
        unread_line_frac: Fraction of the image's lines on which every pixel is
            missing.
        noise_sigma: Per-image MAD-based noise sigma (DN units).
        max_dn: Maximum DN observed in the image.
        background_gradient_score: Dimensionless score of the low-order
            brightness ramp across the sensor (affine-plane peak-to-peak over
            residual MAD-sigma; see
            :func:`spindoctor.support.background_gradient.background_gradient_score`).
            A flat field scores near zero; a scattered-light veiling gradient
            scores well above five.  A non-constant image whose downsample fits
            an affine plane with exactly zero residual (a perfectly noiseless
            ramp) saturates to
            :data:`~spindoctor.support.background_gradient.SATURATED_GRADIENT_SCORE`,
            a large finite sentinel.  ``None`` when the image is too small for
            the measure or the downsample is perfectly constant.
        flags: Additional flags caveats (independent of ``image_class``).
    """

    image_class: ImageClass
    saturation_frac: float
    missing_frac: float
    unread_line_frac: float
    noise_sigma: float
    max_dn: float
    background_gradient_score: float | None = None
    flags: list[ImageFlag] = field(default_factory=list)
