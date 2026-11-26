"""The tuning keys of the correlation techniques.

``BodyDiscCorrelateNav``, ``BodyBlobNav``, and ``RingAnnulusNav`` fold model-error terms
into the covariance of a correlation or centroid fit.  :class:`NCCCovarianceTuning`
declares the keys of those terms, :class:`BodyDiscTuning` extends it with the keys only
the body-disc technique reads, and :class:`BodyBlobTuning` declares the blob technique's
keys.  The schema core parses each technique's ``tuning`` block into its type when the
configuration loads.
"""

from dataclasses import dataclass

from spindoctor.schema import SCHEMA_CONFIG, NonNeg, Positive, Unit

__all__ = ['BodyBlobTuning', 'BodyDiscTuning', 'NCCCovarianceTuning']


@dataclass(frozen=True)
class NCCCovarianceTuning:
    """The tuning keys of the model-error terms folded into an NCC covariance.

    The bare peak-curvature (matched-filter) covariance measures only the
    statistical precision at the correlation peak.  On a well-matched
    template that precision is far smaller than the true registration error,
    which is dominated by silhouette / photometric model error and by
    localization ambiguity.  Three quadrature terms restore calibration:

    - ``localization_uncertainty_scale`` multiplies the inter-pyramid-level
      peak migration (applied inside the correlator, carried here for
      completeness of the covariance model).
    - ``model_error_size_frac`` scales a body / template size in pixels into
      a size-proportional silhouette-error sigma (a coherent shape mismatch
      displaces the peak by a roughly fixed *fraction* of the extent).
    - ``model_error_floor_px`` is a size-independent absolute floor
      (pointing, camera distortion, sub-pixel interpolation).

    The schema core parses these keys out of a correlation technique's ``tuning``
    block when the configuration loads; a technique whose tuning holds more keys
    extends this type.

    Parameters:
        localization_uncertainty_scale: Multiplier on the correlator
            ``consistency`` (px); ``0.0`` disables it.
        model_error_size_frac: Fraction of the template extent (px) taken as
            the silhouette-error sigma; ``0.0`` disables it.
        model_error_floor_px: Absolute floor (px); ``0.0`` disables it.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    localization_uncertainty_scale: NonNeg
    model_error_size_frac: NonNeg
    model_error_floor_px: NonNeg


@dataclass(frozen=True)
class BodyDiscTuning(NCCCovarianceTuning):
    """The tuning keys of ``BodyDiscCorrelateNav``, beyond its covariance terms.

    Parameters:
        rotation_at_edge_fraction: The fraction of the largest allowed rotation at
            which the best rotation candidate trips the at-edge flag.
        consistency_max_fraction_of_diameter: The inter-pyramid peak migration
            allowed per pixel of the largest body's extent before the result is
            spurious.
        consistency_max_px: The smallest migration cap in pixels, whatever the body's
            size.
        refine_lowpass_sigma_px: The sigma in pixels of the Gaussian low-pass applied
            to both surfaces in the full-resolution sub-pixel refinement; 0 disables
            it.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    rotation_at_edge_fraction: Unit
    consistency_max_fraction_of_diameter: NonNeg
    consistency_max_px: NonNeg
    refine_lowpass_sigma_px: NonNeg


@dataclass(frozen=True)
class BodyBlobTuning:
    """The tuning keys of ``BodyBlobNav``.

    Parameters:
        at_edge_tolerance_px: Pixels of slack around the search-window bounds for the
            at-edge test.
        model_error_size_frac: The bias of the brightness-weighted centroid (lit
            hemisphere, irregular shape), which the photon-only per-blob CRLB cannot
            see, as a fraction of the body radius.
        model_error_floor_px: The model-error floor in pixels, added in quadrature to
            the joint translation covariance.
        max_missing_shift_frac: A blob is dropped when leaving its missing pixels out
            would move the centroid of the body's predicted lit silhouette by more than
            this fraction of the body radius.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    at_edge_tolerance_px: NonNeg
    model_error_size_frac: NonNeg
    model_error_floor_px: NonNeg
    max_missing_shift_frac: Positive
