"""The tuning keys of the star techniques.

``StarUniqueMatchNav``, ``StarRefineNav``, and ``StarFieldFromCatalogNav`` find a star's
source the same way: a matched-filter kernel picks the detection, a box around its peak
centroids it, and a threshold on the image noise decides whether it is a source at all.
:class:`StarTuning` declares the keys that tune that detection and judge the result's
position against the search window.  Each technique's tuning type extends it with the
keys only that technique reads, and the schema core parses the technique's ``tuning``
block into that type when the configuration loads.
"""

from dataclasses import dataclass
from typing import Annotated

from annotated_types import Ge, Gt
from pydantic import AfterValidator

from spindoctor.schema import (
    SCHEMA_CONFIG,
    Bool,
    Float,
    Int,
    NonNeg,
    NonNegInt,
    Positive,
    PositiveInt,
    RuleError,
    Unit,
)
from spindoctor.schema.errors import RangeError

__all__ = [
    'StarFieldTuning',
    'StarRefineTuning',
    'StarTuning',
    'StarUniqueMatchTuning',
]

#: A ratio of one brightness to another that a rule requires to be at least 1.
_AtLeastOne = Annotated[Float, Ge(1.0)]


def _at_least_three(value: int) -> int:
    """Return ``value`` if it is at least 3; raise the schema's range error if not."""
    if value < 3:
        raise RangeError('an integer of at least 3')
    return value


#: A count of stars or inliers that cannot be fewer than three, the size of the
#: smallest pattern that cross-checks a rigid translation.
_AtLeastThree = Annotated[Int, AfterValidator(_at_least_three)]


def _check_isolation_radii(inner_px: int, outer_px: int) -> None:
    """Refuse an isolation annulus whose inner radius exceeds its outer radius.

    Parameters:
        inner_px: The value of ``source_isolation_inner_px``.
        outer_px: The value of ``source_isolation_outer_px``.

    Raises:
        RuleError: If the inner radius is larger than the outer radius.
    """
    if inner_px > outer_px:
        raise RuleError(
            'source_isolation_inner_px must not exceed source_isolation_outer_px, '
            f'got {inner_px} and {outer_px}',
            at='source_isolation_inner_px',
            reads=('source_isolation_inner_px', 'source_isolation_outer_px'),
        )


@dataclass(frozen=True)
class StarTuning:
    """The tuning keys shared by the star techniques.

    Parameters:
        psf_sigma_px: The Gaussian sigma in pixels of the matched-filter kernel that
            picks a detection; the strongest local maximum of the response wins, not
            the brightest single pixel.
        centroid_box_half_px: The half-width in pixels of the centroid box around a
            detection's peak, and of the matched-filter kernel.
        detection_sigma: A detection whose brightest pixel stands less than this many
            noise sigmas above the median of its centroid box is noise.
        at_edge_tolerance_px: Pixels of slack around the search-window bounds for the
            at-edge test.
        rotation_at_edge_fraction: The fraction of the largest allowed rotation at
            which a fitted rotation trips the at-edge flag.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    psf_sigma_px: Positive
    centroid_box_half_px: PositiveInt
    detection_sigma: NonNeg
    at_edge_tolerance_px: NonNeg
    rotation_at_edge_fraction: Unit


@dataclass(frozen=True)
class StarUniqueMatchTuning(StarTuning):
    """The tuning keys of ``StarUniqueMatchNav``, beyond those all star techniques share.

    Parameters:
        brightness_margin_to_next_catalog_star_mag: The magnitude difference within
            which another usable star could be a lone star's source; also the floor of
            the brightness-margin confidence term.
        max_peak_sharpness: The largest ratio of a detection's brightest-pixel excess to
            the mean excess of its four edge neighbors, beyond the noise allowance, for
            the detection to be a star rather than a cosmic ray or hot pixel.
        source_isolation_inner_px: The inner radius in pixels of the annulus the
            source-isolation test examines.
        source_isolation_outer_px: The outer radius in pixels of that annulus.
        source_isolation_max_ratio: A detection is kept only when the brightest
            response in the annulus exceeds the annulus mean by less than this fraction
            of the detection's own height above that mean.
        max_residual_px: The largest residual in pixels of a pair in the 2-star path.
        pair_search_stars: The number of brightest usable stars whose candidates the
            2-star path pairs.
        pair_search_sources: The number of strongest isolated sources in each searched
            star's window that are its candidates.
        pair_contradiction_stars: A pair's offset is refuted when at least this many
            other searched stars are each located elsewhere and none is found at the
            offset.
        pair_absent_stars: A pair's offset is refuted when at least this many other
            stars at least as bright as its fainter star show nothing on clear sky where
            the pair puts them.
        pair_vote_peak_factor: The factor within which a source's peak must match the
            peak the pair predicts for it to vote for the pair; it must exceed 1.
        one_star_min_peak_ratio: The smallest ratio of a lone star's source's peak to that
            of every other source in its window.
        one_star_max_residual_px: The largest prediction-to-detection distance in pixels
            the 1-star path accepts when no rival source is in its window.
        one_star_max_peak_sharpness: The largest peak sharpness of a lone star, with no
            allowance for noise.
        one_star_absent_stars: A lone star's offset is refuted when at least this many
            other usable stars show nothing on clear sky where it puts them.
        one_star_confidence_cap: The post-sigmoid confidence cap of the 1-star path.
        two_star_confidence_cap: The post-sigmoid confidence cap of the 2-star path.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    brightness_margin_to_next_catalog_star_mag: NonNeg
    max_peak_sharpness: _AtLeastOne
    source_isolation_inner_px: NonNegInt
    source_isolation_outer_px: PositiveInt
    source_isolation_max_ratio: NonNeg
    max_residual_px: Positive
    pair_search_stars: NonNegInt
    pair_search_sources: PositiveInt
    pair_contradiction_stars: NonNegInt
    pair_absent_stars: PositiveInt
    pair_vote_peak_factor: Annotated[Float, Gt(1.0)]
    one_star_min_peak_ratio: _AtLeastOne
    one_star_max_residual_px: Positive
    one_star_max_peak_sharpness: _AtLeastOne
    one_star_absent_stars: PositiveInt
    one_star_confidence_cap: Unit
    two_star_confidence_cap: Unit

    def __post_init__(self) -> None:
        """Refuse an inverted isolation annulus."""
        _check_isolation_radii(self.source_isolation_inner_px, self.source_isolation_outer_px)


@dataclass(frozen=True)
class StarRefineTuning(StarTuning):
    """The tuning keys of ``StarRefineNav``, beyond those all star techniques share.

    Parameters:
        refine_window_px: The half-width in pixels of the per-star refine window around
            the prior-shifted prediction.
        max_peak_sharpness: The largest peak sharpness of a star, as
            :attr:`StarUniqueMatchTuning.max_peak_sharpness`.
        max_per_star_residual_px: A star whose detection lies more than this many pixels
            from the prior-shifted prediction is dropped.
        min_inliers: Fewer surviving inliers than this makes the result spurious.
        single_inlier_confidence_cap: The post-sigmoid confidence cap of a refine with
            one inlier.
        model_error_floor_px: The model-error floor in pixels, added in quadrature to
            the translation covariance.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    refine_window_px: Positive
    max_peak_sharpness: _AtLeastOne
    max_per_star_residual_px: Positive
    min_inliers: PositiveInt
    single_inlier_confidence_cap: Unit
    model_error_floor_px: NonNeg


@dataclass(frozen=True)
class StarFieldTuning(StarTuning):
    """The tuning keys of ``StarFieldFromCatalogNav``, beyond those all star techniques share.

    Parameters:
        max_sources: The number of brightest detected sources, and of brightest catalog
            stars, fed to the matcher; at least 3, one triplet per side.
        source_isolation_inner_px: The inner radius in pixels of the annulus the
            source-isolation test examines.
        source_isolation_outer_px: The outer radius in pixels of that annulus.
        source_isolation_max_ratio: A detection is kept only when the brightest
            response in the annulus exceeds the annulus mean by less than this fraction
            of the detection's own height above that mean.
        hash_match_tolerance: The match radius in the weighted triplet-hash space.
        hash_ratio_weight: The weight of the two side ratios in the hash distance.
        hash_angle_weight: The weight of the angle in the hash distance.
        inlier_tolerance_px: The largest residual in pixels of a detection-to-catalog
            correspondence that counts as an inlier.
        pattern_match_min_inliers: The fewest inliers the triplet matcher accepts; at
            least 3, one triplet per side.
        wide_offset_enabled: Whether the wide-offset asterism fallback runs when the
            triplet matcher finds too few inliers.
        wide_offset_min_inliers: The fewest inliers of a wide-offset lock; at least 3,
            since two stars cannot cross-check a rigid translation.
        wide_offset_anchor_catalog: The number of brightest photometry-trusted catalog
            stars used as seed anchors.
        wide_offset_anchor_detections: The number of brightest detections paired with
            each catalog anchor.
        wide_offset_false_lock_budget: The largest expected number of chance locks a
            wide-offset candidate may have.
        wide_offset_max_residual_rms_px: The largest inlier residual RMS in pixels of a
            wide-offset lock.
        wide_offset_confidence_cap: The post-formula confidence cap of a wide-offset
            lock.
        wide_offset_one_star_enabled: Whether the single-bright-star wide-offset lock
            runs when no other match locks.
        wide_offset_one_star_catalog_margin_mag: The smallest magnitude margin of the
            catalog anchor over the next-brightest predictable star for a one-star lock.
        wide_offset_one_star_detection_margin_ratio: The smallest ratio of the brightest
            detection's peak to the next-brightest for a one-star lock.
        wide_offset_one_star_confidence_cap: The post-formula confidence cap of a
            one-star wide-offset lock.
        rotation_separability_floor_deg: A fitted rotation smaller than this, in
            degrees, cannot be told from a translation and is reported as unobservable.
        model_error_floor_px: The model-error floor in pixels, added in quadrature to
            the translation covariance.
        psf_refine_enabled: Whether each matched inlier is re-centroided by a fit of the
            instrument's point-spread function.
        psf_refine_box_px: The side in pixels of the square box for that fit; odd.
        psf_refine_search_limit_px: The farthest in pixels the fit may move from the
            moment centroid.
        psf_refine_snr_max: A detection whose integrated SNR exceeds this keeps its
            moment centroid.

    Raises:
        RuleError: If ``psf_refine_box_px`` is even.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    max_sources: _AtLeastThree
    source_isolation_inner_px: NonNegInt
    source_isolation_outer_px: PositiveInt
    source_isolation_max_ratio: NonNeg
    hash_match_tolerance: NonNeg
    hash_ratio_weight: NonNeg
    hash_angle_weight: NonNeg
    inlier_tolerance_px: Positive
    pattern_match_min_inliers: _AtLeastThree
    wide_offset_enabled: Bool
    wide_offset_min_inliers: _AtLeastThree
    wide_offset_anchor_catalog: PositiveInt
    wide_offset_anchor_detections: PositiveInt
    wide_offset_false_lock_budget: Positive
    wide_offset_max_residual_rms_px: NonNeg
    wide_offset_confidence_cap: Unit
    wide_offset_one_star_enabled: Bool
    wide_offset_one_star_catalog_margin_mag: Positive
    wide_offset_one_star_detection_margin_ratio: _AtLeastOne
    wide_offset_one_star_confidence_cap: Unit
    rotation_separability_floor_deg: NonNeg
    model_error_floor_px: NonNeg
    psf_refine_enabled: Bool
    psf_refine_box_px: PositiveInt
    psf_refine_search_limit_px: Positive
    psf_refine_snr_max: NonNeg

    def __post_init__(self) -> None:
        """Refuse an even PSF-fit box and an inverted isolation annulus."""
        _check_isolation_radii(self.source_isolation_inner_px, self.source_isolation_outer_px)
        if self.psf_refine_box_px % 2 == 0:
            raise RuleError('expected an odd number of pixels', at='psf_refine_box_px')
