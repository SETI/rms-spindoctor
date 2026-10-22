==========================================================
Star Unique Match (StarUniqueMatchNav)
==========================================================

Overview
========

:class:`~spindoctor.nav_technique.nav_technique_star_unique_match.StarUniqueMatchNav` is
the pass-1 1-star or 2-star unique-match technique. It searches a window around the
prediction of each of the brightest predictable stars, takes the strongest detection above
a noise threshold, and reports the translation implied by a cross-checked pair of stars
(2-star path) or, when no pair cross-checks, by the brightest star alone (1-star path);
the 1-star path additionally requires the star to be a configured magnitude margin
brighter than every other predictable star within reach of the image. It runs without a
prior — its job is to seed the pass-2
:class:`~spindoctor.nav_technique.nav_technique_star_refine.StarRefineNav` on scenes where
the SPICE pointing error is small enough that the brightest predictable star sits inside
the per-instrument search window.

Feasibility passes when at least one usable
:data:`~spindoctor.feature.feature_type.NavFeatureType.STAR` feature is in the input set;
feasibility fails when none are usable (every predictable star is occluded by a body
silhouette or ring annulus).

Theory
======

The technique exploits a structural fact: when one predictable star dominates the field's
predicted SNR by a comfortable margin, the strongest detection inside a small search
window around that star's prediction is almost certainly the matched detection — there is
no other predictable source bright enough to be confused with it. Two stars dominating the
field similarly support a 2-star unique-match path that recovers translation plus rotation
by Procrustes alignment.

Brightness-margin gate
----------------------

The matcher computes a magnitude difference between the brightest predictable star within
reach of the image and the next-brightest one, using the standard relation
:math:`\Delta m = 2.5 \log_{10}(\mathrm{SNR}_{1} / \mathrm{SNR}_{2})`; the 2-star path
section below says which stars are within reach. The margin floor
(``brightness_margin_to_next_catalog_star_mag``) gates only the 1-star path: a star whose
brightness margin falls below it is not "uniquely bright" and the 1-star path fails. The
2-star path is attempted first whenever at least two usable stars exist, regardless of the
margin; on that path the margin (to the next-brightest *unmatched* star) is computed as a
diagnostic only.

Local centroid
--------------

For each candidate match, the technique cuts a square window around the predicted
position, matched-filters it, takes the strongest response peak inside it as the
detection, and fits a brightness-weighted moment over a
``centroid_box_half_px``-half-width box around it to recover the sub-pixel centroid. The
shared helper ``local_centroid`` in ``_star_helpers`` does this for
:class:`~spindoctor.nav_technique.nav_technique_star_refine.StarRefineNav` as well.

- **The window** is centered on the pixel that holds the prediction and reaches
  ``floor(search_window_px)`` whole pixels either side of it
  (``search_window_bounds``), so it is always
  ``2 * floor(search_window_px) + 1`` pixels on a side and its center is never more than
  half a pixel from the prediction. A window whose edges are rounded outward from the
  prediction instead is a pixel wider for a non-integer prediction and reaches farther on
  one side than the other, with the longer side flipping as the prediction crosses a
  pixel center, so a sub-pixel change of prediction admits or drops a whole row of
  candidates. The peak-ratio gate below uses the same window.
- **A detection is a peak inside the window**: a pixel whose response is the largest
  within the kernel's footprint, judged with the image outside the window in view, the
  local-maximum rule ``StarFieldFromCatalogNav`` uses. A window pixel on the flank of a
  source outside the window is no peak, so a star whose own peak lies outside the window
  is not measured. Taking the brightest pixel of the window instead measures such a star
  through the light its flank spreads across the window's edge, with a centroid box that
  reaches outside the window, and takes the edge of a ring or a limb where the window cuts
  it: on ``ISS_006RI_LPHRLFMOV001_PRIME`` such cut edges at the corners of two stars'
  windows formed a consistent pair and produced an answer 65 px wrong.
- **The detection statistic** is the response to the Gaussian matched filter
  ``StarFieldFromCatalogNav`` detects with (``psf_sigma_px``, side
  ``2 * centroid_box_half_px + 1``). The response weighs a source by the light it spreads
  over a PSF footprint and is zero over a uniform region. On ``N1530185128_1`` a
  single-pixel cosmic ray 1.8 times the brighter star's peak sits 30.7 rows and 24
  columns from that star's prediction; taking the brightest pixel measured the ray, while
  the star, smeared over 7 rows, gives 1.35 times the ray's response even when the ray is
  inside the window. Preferring the qualifying peak nearest the prediction would not help
  there, nor wherever the prior is tens of pixels off: noise peaks that clear the
  threshold sit closer to the prediction than the star does.
- **A peak sharper than a star is passed over** and the next-strongest peak taken. The
  response protects a star narrower than the kernel much less than a broad one: against a
  Cassini NAC star (PSF sigma 0.54 px) a single-pixel spike wins once it exceeds about 1.7
  times the star's peak when the star is centered on a pixel, and 2.5 times when it sits
  on a pixel corner. So the brightest pixel of each candidate's box is compared with the
  mean of its four edge neighbors, both above the box's median: a candidate whose excess
  exceeds ``max_peak_sharpness`` times its neighbors' excess by more than
  ``detection_sigma`` times the noise of that difference is a cosmic ray or hot pixel. Of
  6322 sources at catalog stars' positions under the published F ring bundle's pointing,
  in 2500 frames of the movies set aside for tuning, the default of 8 passes over two: a
  hot pixel whose neighbors hold none of its light, and a star of ratio 8.4. A pixel on
  the image's edge has no complete set of neighbors and is never passed over.
- **The threshold** applies to the brightest pixel of the detection's centroid box, which
  must stand ``detection_sigma * image_noise_sigma`` above the box's median. A noise
  bump on the floor of a saturated ring or body, which the response can make a peak,
  stands nowhere near that far above its own surroundings and is no detection.
- **The centroid** is the background-subtracted moment of the box, with the box
  re-centered on its own centroid until it stops moving (at most five times). The
  response peaks at the edge of a saturated core, where a box first straddles it; the
  re-centering moves the box over the core. A core longer than the box is measured
  wherever the box comes to rest inside it. A box that moves more than
  ``centroid_box_half_px`` from the candidate, or whose center leaves the window, has
  walked onto another source, often a brighter one beyond the window, and the window holds
  no detection.
- **The detection must be isolated**, by the test
  :class:`~spindoctor.nav_technique.nav_technique_star_field.StarFieldFromCatalogNav`
  applies to its sources (``source_isolation_inner_px``, ``source_isolation_outer_px``,
  and ``source_isolation_max_ratio``, over the same matched filter). A ridge of a ring or
  a cosmic-ray track is a peak of the response but runs through the surrounding annulus.
  On the black-sky Galileo library frame ``C0164958500R`` two tracks at the same offset
  from two predictions formed a pair that passed its cross-check. Without the test, 3 more
  of the 1030 frames of ``ISS_006RI_LPHRLFMOV001_PRIME`` navigate, none wrong. Both radii
  move out by half the star's expected smear length: on the library frame
  ``N1530185128_1`` the spacecraft's motion streaks every star over 8.5 px, each streak
  crosses an unsmeared star's annulus, and no pair would pass.
- **A window outshone from outside** is refused on the refinement path only: when the
  window's brightest pixel lies at its edge, belongs to no peak inside the window, and is
  brighter than the detection's own peak, the light of a source outside the window
  outshines anything chosen inside it.
  :class:`~spindoctor.nav_technique.nav_technique_star_refine.StarRefineNav`'s window is
  6 px around a prediction already corrected by the pass-1 offset, so there it means
  the star has left the window; without the refusal, a 2-degree roll of the simulated
  star field had refinement accept noise peaks beside each displaced star and report a
  zero roll. This technique's windows often cut a ring, whose flank would refuse a
  correct window: with a 30 px window, refusing here lost 26 correct solutions in the
  1030 frames of an F ring movie.

A detection that fails the noise threshold or the isolation test collapses the candidate.
On the 2-star path a best-assignment residual exceeding ``max_residual_px`` rejects the
pair, and on the 1-star path a no-rival detection whose prediction-to-detection distance
exceeds ``one_star_max_residual_px`` marks the result spurious.

1-star path
-----------

The 1-star path reports the per-star residual (observed centroid minus predicted position)
as the translation. The reported covariance is the per-feature Cramer-Rao lower bound from
:attr:`~spindoctor.feature.feature.NavFeature.position_cov_px`, inflated by ``residual_px**2``
on the diagonal so a noisy match honestly reports its scatter rather than the noise-free
CRLB floor. Rotation is unobservable on a single point match.

The path rests on the premise that the search window holds a *single unambiguous*
detection, and the absolute ``detection_sigma`` threshold alone cannot enforce that: the
maximum of the window's several thousand noise pixels sits at 3.5-4.5 sigma, so with a
marginal star the brightest pixel is routinely a noise spike that clears the threshold.
The detection therefore also passes a peak-to-runner-up ambiguity gate
(``one_star_min_peak_ratio``): the detection's own background-subtracted peak must exceed
the brightest pixel of the rest of the window (outside the detection's own centroid box)
by the configured ratio, or the result is spurious with an ``ambiguous_detection`` reason.
Because the detection is chosen by matched-filter response, the rest of the window can
hold a single pixel brighter than the detection's peak; the ratio is then below 1 and the
one-star match is refused, the safe answer for a match with no second star to check it. A
real star detection towers over the window's noise order statistics while a matched spike
sits within a few percent of its runner-up; the ratio is unit-free, so no DN-scale or
photometric calibration enters the gate. The measured ratio is surfaced as the
:attr:`~spindoctor.nav_technique.diagnostics.StarUniqueMatchDiagnostics.detection_peak_ratio`
diagnostic.

The peak-ratio and brightness-margin gates both report an ``inf`` sentinel when no rival
exists (no runner-up detection above the window background, no other predictable catalog
star). The sentinels pass the ratio checks by design so a genuinely unique bright star
stays matchable — but an infinite peak ratio also means the ambiguity gate measured
nothing: on a real noisy background the runner-up is always finite, so the sentinel is the
signature of a lone hot pixel or artifact on a flat or quantized frame. In that vacuous
case the detection must additionally sit within ``one_star_max_residual_px`` of the
prediction. The residual on this path is the claimed offset itself, and with no rival
statistics to lean on, a no-rival detection far from the prediction is far more likely an
unrelated artifact than a large pointing error; the default bound of 10 px is one third of
the 30 px window it was set against, the ~1-sigma core of the pointing envelope that
window was sized to bracket. A finite peak ratio leaves acceptance with the measured
ambiguity gate: genuine one-star matches with offsets up to ~24 px exist in the
operator-verified library, so the 10 px bound cannot hold every match. Every 1-star match
is held instead to ``inner_search_window_px`` (30 px), above those offsets.

2-star path
-----------

When at least two usable stars exist, the technique first attempts the 2-star path. Every
pair among the ``pair_search_stars`` brightest usable stars is tried, brightest pairs
first, in two windows: ``inner_search_window_px`` and then ``search_window_px``. For each
pair both detection-to-prediction assignments are tried and the one with the smaller
residual kept; the residual is half the difference between the two stars' offsets, and a
pair passes only when it is within ``max_residual_px``. The first passing pair is the
answer, unless another passing pair puts the offset more than twice ``max_residual_px``
from it. Two cross-checked answers that disagree mean at least one is a coincidence, and
which one cannot be told, so the technique reports spurious with an ``ambiguous_pairs``
reason. The first passing pair is refused in the same way, with a ``pair_contradicted``
reason, when the other searched stars have at least ``pair_contradiction_stars`` distinct
isolated detections in the pair's window and every one of those detections lies more than
twice ``max_residual_px`` from the pair's offset. Overlapping windows can return one
source for several stars, so detections within 1 px of one another count once. Two stars
can agree by coincidence; three more stars that are in the image and found elsewhere say
they did. Each other star
is first sought where the pair's offset puts it, in a window of half-width twice
``max_residual_px`` around its moved prediction. An isolated point source found there
votes for the pair, whatever else the star's own window holds, when it is as bright as
the star should be. The pair's two stars set the scale: each one's peak above its local
background, divided by its predicted SNR, and the two quotients' geometric mean, times
the other star's predicted SNR, is the peak that star should show. A source whose peak
lies within a factor of ``pair_vote_peak_factor`` of that votes for the pair. A faint
field star or a two-pixel cosmic-ray hit that lies there by chance is a small fraction
of it, and the star is then sought in its own window as if nothing were found at the
offset. Only a star that is absent at the pair's offset can vote against it. A frame
taken in high cosmic-ray flux holds hits brighter than many of its stars: on F ring
frame N1799563905 the brightest source in each of three other stars' windows is such a
hit, and the stars at the pair's offset, with peaks 0.68 to 2.04 times what the pair
predicts, keep the pair, 0.12 px from the answer four other stars confirm. A simulated
six-star field whose catalog positions each scatter by 6 or 8 px has two stars agreeing to
0.8 and 1.1 px, reported 5 and 7 px wrong without the refusal. Of the 6098 two-star
answers in the F ring movies set aside for tuning, none has three such stars and 32 have
two, all of them correct.

The path applies no per-window ambiguity gate: a window won by something other than its
star moves that star's offset away from the other's and the residual declines the pair,
while the peak ratio would refuse correct pairs. On ``N1530185128_1`` the fainter star's
window holds a single-pixel spike within 0.4 percent of the star's own peak, while the
pair's residual is under half a pixel.

Trying more than the brightest two matters when the brightest predicted star is not in the
image. A star predicted inside the extended field of view may fall off the detector once
the pointing error is applied, or under a ring's glare; with only the brightest two tried,
its window holds no star, the pair fails, and the frame falls through to the 1-star path,
whose brightness margin a field of comparable stars never clears. On
``ISS_006RI_LPHRLFMOV001_PRIME`` searching pairs among the brightest 3, 5, and 8 stars
navigates 710, 771, and 777 of the 1030 frames, every one within 2 px of the published F
ring bundle. The narrower window is searched as well because the strongest detection of
the wider one is more often something brighter than the star; with no narrower window the
technique navigates 750 frames, none wrong.

Both paths draw only on stars whose search window holds at least one pixel of the image.
The extended field of view pads the image, and the star catalog predicts stars across all
of it; the Cassini NAC's padding is 140 px across samples, wider than its 50 px window. A
star predicted farther outside the image than the window reaches cannot be in the image at
any offset the window allows, so it can neither be detected nor be confused with another
star's detection. A photometric star limit deep enough to predict faint stars across the
padding fills the brightest ranks with them. Under such a limit, the second through fifth
brightest stars of F ring frame ``N1862698331`` all lie beyond the window's reach, no pair
of the five has both stars in the image, and the frame is refused; with them set aside, a
pair of stars in the image answers 0.15 px from the published F ring bundle. A star set
aside this way is not the 1-star path's candidate either, and it does not count
against that candidate's brightness margin. A frame whose every usable star is out of
reach reports a spurious result.

.. Setting aside stars out of the window's reach: #763.

The residual tolerance is measured. With it opened to 4 px, the 704 two-star matches on
that observation that lie within 2 px of the bundle have residuals of median 0.10 px and
95th percentile 0.30 px, and no star is paired with the wrong source. At 1.2 px, 701 of
the 704 pass; 1.2 rather than 1.0 admits the correct pair of library frame
``N1550270436``, whose residual is 1.04 px and whose fallback without it is a body
centroid 15 px wrong.

For each pair the fit is a Procrustes / similarity-transform fit on the two
correspondences that reports the recovered translation plus rotation. The rotation
variance is derived analytically as

.. math::

    \sigma_{\theta}^{2} = \frac{2 \,(\sigma_{v}^{2} + \sigma_{u}^{2})}{L^{2}}

where :math:`L` is the catalog-pair angular separation and the per-axis sigmas are the
average of the two stars' CRLB floors. When no pair passes, the technique falls back to
the 1-star path.


Per-mode confidence cap
-----------------------

The 1-star path cannot self-check (a single observation has nothing to validate against), so
the technique applies a post-sigmoid cap of ``one_star_confidence_cap`` (default 0.7). The
2-star path can validate the assignment via the per-correspondence residual, so its cap is
``two_star_confidence_cap`` (default 0.8). The caps are applied by the technique itself,
not by the YAML spec's :attr:`~spindoctor.nav_technique.confidence.ConfidenceSpec.hard_cap`, so the
two paths can share a single spec.

Restrictions and assumptions
----------------------------

- The brightness-margin gate is driven by the catalog visual magnitudes: each star's
  ``predicted_snr`` is the magnitude-margin effective SNR derived from how far the star
  sits below the per-observation limiting magnitude, so the SNR ratio between two stars
  recovers their catalog magnitude difference. A wrong catalog magnitude shifts the
  brightness-margin computation and may erroneously accept or reject the match.
- The search window assumes the SPICE pointing error is bounded by ``search_window_px``
  (default 50 px). When the pointing error exceeds this width, the strongest detection in
  the window is not the matched detection. On ``ISS_006RI_LPHRLFMOV001_PRIME`` the Cassini
  NAC pointing error reaches 43 px, and a 30 px window puts the stars of 191 of the 1030
  frames out of reach. A 1-star match is accepted only within ``inner_search_window_px``
  (30 px) of its prediction, while the whole window is still searched so that a brighter
  source anywhere in it vetoes: nothing cross-checks a single detection, and the wider the
  region it may come from, the likelier an artifact is the brightest thing in it. On the
  black-sky Galileo library frame ``C0164392700R`` a lone artifact lies 40 px from the one
  predicted star. On ``ISS_006RI``, windows of 30, 40, 50, and
  60 px navigate 753, 774, 771, and 762 frames, none wrong; the narrower ones leave the
  stars of the frames whose error exceeds them out of reach, and the other gates refuse
  those frames. Beyond the window is the regime where the multi-star
  :class:`~spindoctor.nav_technique.nav_technique_star_field.StarFieldFromCatalogNav` triplet
  matcher takes over.
- Both paths assume the matched detection is unsaturated. A saturated star whose centroid
  has been clipped will produce a biased centroid; in practice the brightest catalog stars
  are held off by the magnitude gate. The predicted-position saturation / cosmic-ray mask is
  not consulted to reject stars (a star's predicted pixel is not special once an offset is
  present), so among the predictable stars occlusion is the only thing that makes one
  unusable.

Sources of uncertainty
----------------------

The 1-star covariance is the per-feature CRLB floor from
:attr:`~spindoctor.feature.feature.NavFeature.position_cov_px` inflated by ``residual_px**2``
on the diagonal. The 2-star translation covariance is the average of the two stars'
residual-inflated matrices, plus the analytic rotation variance derived above when rotation
is fitted.
When the converged offset sits within the at-edge tolerance of any axis bound, or when the
2-star rotation parameter is at the configured fraction of its cap, the result is flagged
:attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.at_edge` and the hard-zero
gate forces confidence to zero. When a no-rival 1-star detection's
prediction-to-detection distance exceeds ``one_star_max_residual_px`` the result is
flagged :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.spurious`.

Configuration
=============

All numeric tunables for this technique live in ``techniques.StarUniqueMatchNav.tuning`` in
``src/spindoctor/config_files/config_510_techniques.yaml``.

- ``brightness_margin_to_next_catalog_star_mag`` — float, default ``1.5`` (mag). Minimum
  magnitude difference between the brightest predictable star and the next-brightest for
  the 1-star path to fire.
- ``search_window_px`` — float, default ``50.0`` px. Half-width of the search window
  around each prediction. Must bracket the SPICE pointing error: a star whose own peak
  lies outside the window is not measured. The window is
  ``2 * floor(search_window_px) + 1`` pixels on a side, centered on the pixel holding the
  prediction.
- ``inner_search_window_px`` — float, default ``30.0`` px. A second, narrower window in
  which the 2-star path also searches each pair. A 1-star detection farther than this from
  its prediction is refused with a ``one_star_detection_beyond_inner_window`` reason.

  Pointing error belongs to a camera, so both values are defaults: a camera whose instrument
  configuration names ``star_search_window_px`` and ``star_one_star_radius_px`` uses its
  own (:func:`~spindoctor.nav_technique.nav_technique.star_search_px_for_obs`). Cassini's
  cameras name theirs from their pointing error against the published F ring bundle, on the
  movies set aside for tuning: the NAC's 50 and 30 px hold 98.7% of its 10362 star frames on
  each axis and 90.9% radially, and the WAC's 13 and 9 px hold the same fractions of its 794,
  whose error reaches 14 px. With the NAC's values the WAC navigates 307 of those frames
  within 2 px of the bundle; with its own, 386.

- ``psf_sigma_px`` — float, default ``1.0`` px. Gaussian sigma of the matched-filter
  kernel that chooses the detection in each window and measures its isolation; the same
  value ``StarFieldFromCatalogNav`` detects with.
- ``max_peak_sharpness`` — float, default ``8.0``. Largest ratio of a detection's
  brightest-pixel excess to the mean excess of its four edge neighbors; a sharper peak is
  passed over as a cosmic ray or hot pixel.
- ``source_isolation_inner_px``, ``source_isolation_outer_px`` — int, defaults ``4`` and
  ``6`` px. Radii of the annulus over which a detection's isolation is measured.
- ``source_isolation_max_ratio`` — float, default ``0.5``. Largest allowed excess of the
  annulus maximum over the annulus mean, as a fraction of the detection's own height above
  that mean.
- ``pair_search_stars`` — int, default ``5``. The 2-star path tries every pair among this
  many brightest usable stars.
- ``pair_contradiction_stars`` — int, default ``3``. The 2-star path refuses its first
  passing pair when at least this many distinct sources are detected for the other
  searched stars, all away from the pair's offset, and none is found where the pair's
  offset puts it.
- ``pair_vote_peak_factor`` — float, default ``4.0`` (dimensionless), finite and above
  1. A source found where the pair's offset puts another star counts as that star only
  when its peak above the local background lies within this factor of the peak the
  pair's stars predict for it. Among 93 two-star answers in the F ring movies set aside
  for tuning, the 492 other stars at least a quarter as bright as the pair's fainter star
  showed peaks 0.35 to 2.8 times the prediction (2nd to 99th percentile), and 4 sets
  aside 9 of them.
- ``centroid_box_half_px`` — int, default ``3`` px. Half-width of the brightness-weighted
  moment box around the detection, and of the matched-filter kernel.
- ``max_residual_px`` — float, default ``1.2`` px. Maximum allowed best-assignment
  residual for a pair to pass the 2-star cross-check. Two passing pairs farther apart than
  twice this make the result spurious.
- ``detection_sigma`` — float, default ``4.0`` (dimensionless). Detection-threshold
  multiplier on the per-image noise sigma; a detection's brightest pixel must stand this
  many noise sigmas above the median of its centroid box.
- ``one_star_confidence_cap`` — float, default ``0.7`` (dimensionless). Post-sigmoid
  confidence cap when the 1-star path fires. A single match cannot self-check.
- ``two_star_confidence_cap`` — float, default ``0.8`` (dimensionless). Post-sigmoid
  confidence cap when the 2-star path fires. Per-correspondence residual cross-checks the
  assignment.
- ``one_star_min_peak_ratio`` — float, default ``1.5`` (dimensionless). Minimum
  background-subtracted peak-to-runner-up ratio for a 1-star detection; below it the
  result is spurious with an ``ambiguous_detection`` reason. Values below 1 are invalid:
  a ratio below 1 means the window holds a pixel brighter than the detection's own peak.
- ``one_star_max_residual_px`` — float, default ``10.0`` px. Maximum
  prediction-to-detection distance accepted by the 1-star path when the peak-ratio gate
  is vacuous (its no-rival ``inf`` sentinel); beyond it the result is spurious with a
  ``no_rival_detection`` reason. Default is one third of the 30 px
  ``inner_search_window_px`` it was set against (the ~1-sigma core of the pointing
  envelope that window brackets).
  Not consulted when the peak ratio is finite. Values at or below 0 are invalid.
- ``at_edge_tolerance_px`` — float, default ``1.0`` px. A converged offset whose absolute
  distance from any search-window axis bound falls within this tolerance is flagged
  :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.at_edge`.
- ``rotation_at_edge_fraction`` — float, default ``0.95`` (dimensionless). Fraction of
  :attr:`~spindoctor.nav_orchestrator.nav_context.NavContext.max_rotation_deg` at which the
  converged 2-star rotation magnitude trips
  :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.at_edge`. Only the 2-star
  Procrustes path uses it; the 1-star path always reports rotation as unobservable.

Per-instrument overrides
------------------------

Per-instrument YAML files in ``src/spindoctor/config_files/config_4N0_inst_*.yaml`` do not
override any of these knobs.

Confidence formula
------------------

The technique reports a calibrated confidence in :math:`[0, 1]` produced by the shared
sigmoid combination; see :doc:`dev_guide_techniques_confidence`. Spec is
``techniques.StarUniqueMatchNav``; consumes attributes off
:class:`~spindoctor.nav_technique.diagnostics.StarUniqueMatchDiagnostics` plus
:attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.at_edge` and
:attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.spurious`.

- :attr:`~spindoctor.nav_technique.diagnostics.StarUniqueMatchDiagnostics.predicted_snr` —
  alpha = 0.809, offset = 0.0, divisor = 250.0, cap at 1.0. Predicted SNR of the brightest
  catalog star. Higher SNR shrinks the centroid CRLB; saturates at SNR=250 (calibration
  campaign raw p5/p50/p95 = 15/85/252 — the sim's flux-normalized star deposit brightens
  the healthy range, so the divisor spans it).
- :attr:`~spindoctor.nav_technique.diagnostics.StarUniqueMatchDiagnostics.brightness_margin_mag`
  — alpha = 1.002, offset = 1.5, divisor = 1.5, cap at 1.0. Magnitude margin to the
  next-brightest predictable star in the extfov. Below the 1.5 mag floor the 1-star path
  short-circuits before reaching the formula; above the floor, additional margin earns
  confidence up to a 1.5 mag span.
- :attr:`~spindoctor.nav_technique.diagnostics.StarUniqueMatchDiagnostics.residual_px` —
  alpha = -0.112, offset = 0.0, divisor = 2.0, no cap. Detection-vs-prediction residual.
  Larger residuals pull confidence down.

Hard-zero gate: :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.at_edge` and
:attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.spurious` either firing forces
confidence to zero before the sigmoid evaluates. The constant baseline is
:math:`\alpha_{0} = 0.781`. No post-sigmoid ``hard_cap`` is applied at the spec level; the
per-mode caps above are applied by the technique itself.

Implementation
==============

Source files:

- ``src/spindoctor/nav_technique/nav_technique_star_unique_match.py`` —
  :class:`~spindoctor.nav_technique.nav_technique_star_unique_match.StarUniqueMatchNav` and the
  brightness-margin / 1-star / 2-star helpers.
- ``src/spindoctor/nav_technique/_star_unique_two_star.py`` — package-private pair geometry
  of the 2-star path: the assignment solve (``solve_two_star_assignment``,
  ``similarity_fit_assignment``) and the match covariances (``build_star_covariance``,
  ``two_star_covariance``, ``two_star_covariance_3dof``).
- ``src/spindoctor/nav_technique/_star_helpers.py`` — package-private helpers ``usable_stars``,
  ``search_window_bounds``, ``local_centroid``, ``detection_peak_ratio``,
  ``is_isolated_point_source``, ``isolated_response``, ``gaussian_psf_kernel``,
  ``predicted_snr``, ``predicted_vu``, ``brightness_margin_mag``, and
  ``similarity_transform_fit``.
- ``src/spindoctor/nav_technique/confidence.py`` — sigmoid-combination evaluator; documented at
  :doc:`dev_guide_techniques_confidence`.
- ``src/spindoctor/nav_technique/diagnostics.py`` —
  :class:`~spindoctor.nav_technique.diagnostics.StarUniqueMatchDiagnostics`; documented at
  :doc:`dev_guide_techniques_diagnostics`.

Public class :class:`~spindoctor.nav_technique.nav_technique_star_unique_match.StarUniqueMatchNav`,
base :class:`~spindoctor.nav_technique.nav_technique.NavTechnique`. Self-registers via
``__init_subclass__``.

Class attributes:

- :attr:`~spindoctor.nav_technique.nav_technique.NavTechnique.name` — ``'StarUniqueMatchNav'``.
- :attr:`~spindoctor.nav_technique.nav_technique.NavTechnique.accepts_feature_types` —
  ``frozenset({STAR})``.
- :attr:`~spindoctor.nav_technique.nav_technique.NavTechnique.requires_prior` — ``False``.
- :attr:`~spindoctor.nav_technique.nav_technique.NavTechnique.confidence_attributes` —
  ``{'at_edge', 'spurious', 'predicted_snr', 'brightness_margin_mag', 'residual_px'}``.

Public methods (autodocumented at :doc:`/api_reference/api_nav_technique`):
:meth:`~spindoctor.nav_technique.nav_technique_star_unique_match.StarUniqueMatchNav.is_feasible`
and
:meth:`~spindoctor.nav_technique.nav_technique_star_unique_match.StarUniqueMatchNav.navigate`.

Diagnostics
-----------

:class:`~spindoctor.nav_technique.diagnostics.StarUniqueMatchDiagnostics`:

- :attr:`~spindoctor.nav_technique.diagnostics.StarUniqueMatchDiagnostics.mode` —
  ``'one_star'`` or ``'two_star'``. Diagnostic only; the confidence formula does not
  consume strings, but the per-mode cap branches inside
  :meth:`~spindoctor.nav_technique.nav_technique_star_unique_match.StarUniqueMatchNav.navigate`
  read it.
- :attr:`~spindoctor.nav_technique.diagnostics.StarUniqueMatchDiagnostics.predicted_snr` —
  predicted SNR of the brightest catalog star. Consumed by the confidence formula.
- :attr:`~spindoctor.nav_technique.diagnostics.StarUniqueMatchDiagnostics.brightness_margin_mag`
  — magnitude difference to the next-brightest unmatched catalog source predictable in
  extfov; ``+inf`` when no unmatched star exists. Consumed by the confidence formula.
- :attr:`~spindoctor.nav_technique.diagnostics.StarUniqueMatchDiagnostics.residual_px` —
  detection-vs-prediction residual. Consumed by the confidence formula and by the
  spurious-detection gate.
- :attr:`~spindoctor.nav_technique.diagnostics.StarUniqueMatchDiagnostics.detection_peak_ratio`
  — background-subtracted peak over window runner-up for the 1-star path (``0.0`` when
  not measured; ``inf`` when the runner-up sits at or below the background). Diagnostic
  only; the ambiguity gate consumes it before the confidence formula runs.

Call path
---------

Call path traced through
:meth:`~spindoctor.nav_technique.nav_technique_star_unique_match.StarUniqueMatchNav.navigate`:

1. Open a logged section. Filter the offered features down to ``usable_stars``
   (predictable and not occluded by a body silhouette or ring annulus), then to
   ``stars_in_reach`` (those whose search window overlaps the image).
2. Sort the usable cohort by predicted SNR (brightest first) and compute the brightness
   margin between consecutive entries via ``brightness_margin_mag``.
3. Branch on the usable-star count:

   - **2-star path.** Attempted first whenever at least two usable stars exist (no
     brightness-margin precondition). For every pair among the ``pair_search_stars``
     brightest stars, in each of the two windows, slice the window around each prediction
     in :attr:`~spindoctor.nav_orchestrator.nav_context.NavContext.image_ext` and call
     ``local_centroid`` to extract the sub-pixel detection. When both detections clear the
     noise threshold and are isolated point sources, run a Procrustes / similarity refit
     via ``similarity_transform_fit`` to recover translation and rotation, and keep the
     pair when its residual is within ``max_residual_px``. Report the first passing pair,
     or a spurious result when passing pairs disagree; when none passes, fall back to the
     1-star path.
   - **1-star path.**  When fewer than two stars are usable (or no pair passed),
     require the brightest to clear the brightness-margin floor, slice a window
     around its prediction, extract the sub-pixel
     centroid, refuse it when it is not an isolated point source or lies outside
     ``inner_search_window_px``, pass the peak-to-runner-up ambiguity gate (plus the
     ``one_star_max_residual_px`` residual gate when that ratio is the no-rival
     sentinel), and report the per-star residual as the translation.

4. Result-shape branches on the chosen path and
   :attr:`~spindoctor.nav_orchestrator.nav_context.NavContext.fit_camera_rotation`:

   - **1-star, no rotation.**  (2, 2) covariance: the CRLB floor from
     :attr:`~spindoctor.feature.feature.NavFeature.position_cov_px` inflated by
     ``residual_px**2`` on the diagonal;
     :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.rotation_rad` and
     :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.sigma_rotation_rad` are
     ``None``.
   - **1-star, rotation fit.**  Rank-deficient (3, 3) covariance via
     :func:`~spindoctor.nav_technique.nav_technique.embed_rotation_unobservable`;
     :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.rotation_rad` is
     ``0.0`` and the sigma is the rotation-unobservable sentinel.
   - **2-star, no rotation.**  (2, 2) covariance from the average of the two stars'
     residual-inflated
     :attr:`~spindoctor.feature.feature.NavFeature.position_cov_px` matrices.
   - **2-star, rotation fit.**  Full (3, 3) covariance: the (2, 2) translation block is
     the average of the two residual-inflated per-star matrices, the rotation diagonal is
     the analytic
     :math:`2 (\sigma_{v}^{2} + \sigma_{u}^{2}) / L^{2}` form derived from the catalog-pair
     separation :math:`L`.
     :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.rotation_rad` is the
     Procrustes-fit angle and the sigma is the square root of the rotation diagonal.

5. Apply the at-edge test against the search-window axis bounds and the rotation cap.
6. Build a :class:`~spindoctor.nav_technique.diagnostics.StarUniqueMatchDiagnostics`, evaluate the
   confidence spec via :func:`~spindoctor.nav_technique.confidence.evaluate_sigmoid_combination`,
   apply the per-mode cap (``one_star_confidence_cap`` or ``two_star_confidence_cap``)
   inside the technique itself, log the breakdown, and assemble the
   :class:`~spindoctor.nav_technique.technique_result.NavTechniqueResult`.

The :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.feature_ids` field
preserves every consumed
:attr:`~spindoctor.feature.feature.NavFeature.feature_id` so the orchestrator's curator can
attribute each match at audit time.

Examples
========

``one_bright_star_no_body`` (Cassini ISS WAC, image ``W1449079117_1``)
    Single star (Vega) in an otherwise empty FOV. The stars model emits a single ``STAR``
    feature with a brightness margin of ``+inf`` to the (nonexistent) next-brightest.
    :class:`~spindoctor.nav_technique.nav_technique_star_unique_match.StarUniqueMatchNav` runs the
    1-star path: a small search window around Vega's prediction returns the brightest peak
    above the noise threshold, the brightness-weighted moment recovers the centroid, and
    the per-star residual is reported as the translation. The 1-star post-sigmoid cap of
    0.7 holds the technique's confidence below 0.7 even when every term saturates.
    Operator-verified offset is :math:`(\Delta v, \Delta u) = (3.06, -0.02)` px. The
    pass-2 :class:`~spindoctor.nav_technique.nav_technique_star_refine.StarRefineNav` consumes
    this prior and polishes it; see :doc:`dev_guide_techniques_star_refine`.

``two_bright_stars_no_body`` (Cassini ISS / NHLORRI star-calibration scene class)
    Catalog yields exactly two unambiguous stars in the FOV, each at least 1.5 magnitudes
    brighter than its next-brightest competing candidate. The stars model emits two
    ``STAR`` features; the technique falls into the 2-star path, runs the per-star local
    centroid against each prediction, and — when both detections clear the per-mode
    photometric and at-image-edge gates — invokes
    ``similarity_transform_fit`` on the catalog-vs-detection point pair to recover a
    translation plus rotation simultaneously. The reported covariance is the
    closed-form 3x3 from the similarity-transform residual scatter; the post-sigmoid
    cap is the 2-star ``two_star_confidence_cap`` (default 0.8), allowing higher
    final confidence than the 1-star path. The
    :attr:`~spindoctor.nav_technique.diagnostics.StarUniqueMatchDiagnostics.mode` field on the
    diagnostics carries ``'two_star'`` so the curator surfaces which path fired.

``faint_stars`` (Galileo SSI / Voyager outer-leg scene class)
    Every catalog star in the FOV is fainter than the per-observation limiting magnitude
    ``obs.star_max_usable_vmag()``. The stars model emits no ``STAR`` features that clear
    the magnitude gate, so the orchestrator does not offer any usable star feature to
    :meth:`~spindoctor.nav_technique.nav_technique_star_unique_match.StarUniqueMatchNav.is_feasible`.
    Feasibility fails with reason ``no_usable_star_features`` and the technique skips its
    :meth:`~spindoctor.nav_technique.nav_technique_star_unique_match.StarUniqueMatchNav.navigate`
    pass entirely. The orchestrator surfaces the infeasibility on the per-image
    :class:`~spindoctor.nav_orchestrator.nav_result.NavResult` via the per-technique
    :attr:`~spindoctor.nav_technique.feasibility.NavFeasibilityReport.reason` so the curator
    records which gate fired without the technique having to allocate a result
    record.
