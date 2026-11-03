==========================================================
Star Unique Match (StarUniqueMatchNav)
==========================================================

Overview
========

:class:`~spindoctor.nav_technique.nav_technique_star_unique_match.StarUniqueMatchNav` is
the pass-1 1-star or 2-star unique-match technique. It searches the instrument's offset
window around the prediction of each of the brightest predictable stars for isolated
point sources, and reports the translation that the most stars agree on through a
cross-checked pair (2-star path) or, when no pair cross-checks, the one implied by a
single star whose source no other star contradicts and nothing else in its window rivals
(1-star path). It runs without a
prior — its job is to seed the pass-2
:class:`~spindoctor.nav_technique.nav_technique_star_refine.StarRefineNav` on scenes where
the SPICE pointing error is small enough that the predictable stars sit inside the
per-instrument offset window.

Feasibility passes when at least one usable
:data:`~spindoctor.feature.feature_type.NavFeatureType.STAR` feature is in the input set;
feasibility fails when none are usable (every predictable star is occluded by a body
silhouette or ring annulus).

Theory
======

The technique exploits two structural facts. Two stars whose candidate sources agree on
one offset are unlikely to agree by chance, so a cross-checked pair recovers translation
plus rotation by Procrustes alignment. A single star whose window holds one source clearly
brighter than every other, at an offset where no other star should have been seen, is
best explained by that source; nothing cross-checks it, so its confidence is capped lower.

Brightness margin
-----------------

The matcher computes a magnitude difference between two stars from their predicted SNRs,
using the standard relation
:math:`\Delta m = 2.5 \log_{10}(\mathrm{SNR}_{1} / \mathrm{SNR}_{2})`. On the 1-star path,
another usable star within ``brightness_margin_to_next_catalog_star_mag`` of the matched
star could be the matched source as well when its window holds it, and the match is
refused (see below). On both paths the margin from the matched star (the brighter of a
pair) to the brightest other usable star is reported as a diagnostic that the confidence
formula consumes.

Local centroid
--------------

For each star, the technique cuts a window around the predicted position, matched-filters
it, takes the strongest response peak inside it as the detection, and fits a
brightness-weighted moment over a ``centroid_box_half_px``-half-width box around it to
recover the sub-pixel centroid. The shared helper ``local_centroid`` in ``_star_helpers``
does this for
:class:`~spindoctor.nav_technique.nav_technique_star_refine.StarRefineNav`, and for this
technique wherever it looks for a star at the position a solution puts it. Both paths
measure each of several peaks of a star's window the same way (``window_sources``; see
below).

- **The window** is centered on the pixel that holds the prediction and reaches
  ``floor(half)`` whole pixels either side of it along each axis
  (``search_window_bounds``), so each side is always ``2 * floor(half) + 1`` pixels long
  and its center is never more than half a pixel from the prediction.
  :class:`~spindoctor.nav_technique.nav_technique_star_unique_match.StarUniqueMatchNav`
  takes its half-widths from the instrument's offset window (``extfov_margin_vu``, read
  through :func:`~spindoctor.nav_technique.nav_technique.search_window_for_obs`), axis by
  axis, as every other technique searches it: pointing error is a property of the camera,
  so stars get no window of their own. A window whose edges are rounded outward from the
  prediction instead is a pixel wider for a non-integer prediction and reaches farther on
  one side than the other, with the longer side flipping as the prediction crosses a
  pixel center, so a sub-pixel change of prediction admits or drops a whole row of
  candidates. The peak-to-runner-up ratio test below uses the same window.
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
  must stand ``detection_sigma * image_noise_sigma`` above the box's median and above the
  box's plane background (below). A noise bump on the floor of a saturated ring or body,
  which the response can make a peak, stands nowhere near that far above its own
  surroundings and is no detection. On a sloping glare the upper edge of a box stands
  half the slope's rise across the box above the box's median, but no higher than the
  noise above the plane.
- **The centroid** is the moment of the box above a plane background, with the box
  re-centered on its own centroid until it stops moving (at most five times). The plane's
  slope is fitted to the ring of pixels 2 px wide around the box, less its pixels more
  than three robust sigmas from the ring's median, and its level is the median of the box
  less that slope. The box median alone leaves the upper half of a sloping glare in the
  moment's weights and pulls the centroid uphill: on F ring frame ``N1571454567`` a V 9.9
  star on ring glare was measured 1.7 px from where a high-passed image puts it. The
  response peaks at the edge of a saturated core, where a box first straddles it; the
  re-centering moves the box over the core. A box that moves more than
  ``centroid_box_half_px`` from the candidate, or whose center leaves the window, has
  walked onto another source, often a brighter one beyond the window, and the window holds
  no detection.
- **A clipped core as long as the box** is measured whole. A clipped core and its
  column bleed are flat. A box lying wholly on them is centered on its own centroid
  wherever it lies, so re-centering stops a half-width in from the end it started at. On
  the simulated ``saturated_star`` scene a vmag -1 star bled over 11 rows was measured
  1.8 px off that way, and ``StarRefineNav``, which weights stars by predicted SNR,
  followed it 1.6 px off. The flat top is the 4-connected set of pixels at or above the
  brightest pixel of the candidate's box less ``detection_sigma * image_noise_sigma``.
  It is a clipped core when four things hold: it is at least as long as the box along
  one axis; it is no wider than the box along the other; the candidate lies on it or
  beside it; and the median of the pixels bordering it lies at least 10 tolerances below
  it. A ring's crest or a limb is flat within the tolerance over long stretches, but its
  sides slope away gently. On Cassini NAC frame ``N1737165983`` a crest measured whole
  pulled a lone star's centroid 1.95 px onto the ring, and its bordering pixels lie only
  4.3 tolerances below it. The moment is taken over the core padded by
  ``centroid_box_half_px`` on every side; the scene's fused offset then lies 0.03 px from
  the planted one. A core that reaches the edge of the window padded by
  ``centroid_box_half_px``, or whose moment lies outside the window, is no point source
  of the window. A clipped core in a calibrated (I/F) image is divided by the flat field
  and is no longer flat within the tolerance; the re-centered box measures it.
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

A peak that fails the noise threshold or the isolation test is passed over and the next
one measured. On the 2-star path a residual exceeding ``max_residual_px`` rejects a
pair, and on the 1-star path a no-rival detection whose prediction-to-detection distance
exceeds ``one_star_max_residual_px`` marks the result spurious.

1-star path
-----------

The 1-star path runs when no two stars agree on an offset, or when only one star is
usable. It searches the whole offset window, as the 2-star path does, and a window that
wide often holds isolated point sources that are not the star: a cosmic-ray hit, a field
star below the catalog's limit, or a small moon. Taking the window's strongest peak, and
nothing when that peak is not isolated, refuses frames whose lone star is plainly in the
image because the strongest peak is a ring's flank, and accepts cosmic rays and faint field
stars tens of pixels from the answer when the star itself is not in the image.

So each of the ``pair_search_stars`` brightest stars, paired with each of the candidates
the pair search found for it, is a hypothesis about the offset (``lone_star_hypotheses`` in
``_star_unique_one_star``), and the other stars judge it:

- A source sharper than a star, allowing nothing for noise
  (``one_star_max_peak_sharpness``) is no lone star. The ``max_peak_sharpness`` test that
  decides what counts as a source allows for noise, and a cosmic-ray hit a little broader
  than a hot pixel passes it. None of 3656 stars found at the published bundles' pointing
  on the F ring and spokes star frames is sharper than 6.96; with a limit of 8, five lone
  sources between 7.03 and 7.97, each 30 to 100 px from the bundle's answer, are taken.
- An offset is refuted when ``one_star_absent_stars`` other usable stars that it puts
  inside the image, on clear sky, show nothing there (``absent_stars``, the test the
  2-star path applies). The lone source's peak per unit of predicted SNR says how bright
  each should be, and a star whose peak there is below that divided by
  ``pair_vote_peak_factor`` is absent. A star the offset puts in the padding, behind a ring
  or a limb, or too faint to stand above the noise at that scale cannot be absent. One
  absent star suffices, where a pair needs two, because a pair is two stars' agreement and
  a lone star has none. On 978 one-star frames of the F ring and spokes movies, requiring
  two answers 13 fewer frames within 1 px of the bundle's answer and one more 39 px from
  it.

Of the hypotheses left, the one brightest per unit of its star's predicted SNR is the
answer (``choose_lone_star``), as the 2-star path chooses among equally supported pairs.
A star whose window holds only sources far too faint for it so yields to a fainter star
whose source matches it. The match is refused as ambiguous when, more than
``2 * max_residual_px`` from the answer:

- another source of the same star's window, whether or not it is star-like or refuted, has
  a peak above the answer's divided by ``one_star_min_peak_ratio``
  (``ambiguous_detection``). The star need not be the brightest source of a wide window,
  but then nothing says which of the two it is. The sources are the window's isolated
  point sources, so a hot pixel, a ring's flank, or a noise spike does not compete. The
  brightest pixel of a wide window is often one of those: 162 of the 361 one-star answers
  on F ring NAC frames have a pixel brighter than the star in the window. A ratio of 2.0
  refuses 14 lone stars within 1 px of the bundle's answer that 1.5 keeps, and answers
  none that 1.5 refuses.
- another star's surviving hypothesis is within ``pair_vote_peak_factor`` of the answer's
  brightness per unit of predicted SNR (``ambiguous_lone_stars``): two stars each explain a
  source, and nothing contradicts either.

The answer is then refused when another usable star within
``brightness_margin_to_next_catalog_star_mag`` of the matched star has the source inside
its own window (``ambiguous_identity``), since nothing in the image tells the two stars
apart. On F ring frame ``N1492081691`` one source lies in the windows of two stars 0.2 mag
apart, and taking it for the brighter gives an answer 7 px from the bundle's. A star whose
window cannot reach the source does not compete. Refusing whenever any star of the extended
field of view is that close in brightness refuses lone stars whenever a comparable star is
predicted far in the padding.

The answer must lie inside the lone-star ellipse, whose semi-axes are the offset window's
half-widths times the camera's ``one_star_ellipse_fraction`` (0.6 for the Cassini NAC: 30
px for a window half-width of 50 px), or it is refused with a
``one_star_detection_outside_ellipse`` reason. Every camera sets its own fraction; see
`Per-instrument overrides`_.

The path also keeps a no-rival residual bound. The source's background-subtracted peak
over the brightest pixel of the rest of the window (outside the source's centroid box) is
reported as the
:attr:`~spindoctor.nav_technique.diagnostics.StarUniqueMatchDiagnostics.detection_peak_ratio`
diagnostic. It is ``inf`` when no pixel of the rest of the window clears the background. On
a real noisy background that pixel is always above it, so the sentinel is the signature of
a lone hot pixel or artifact on a flat or quantized frame. In that case the source must
additionally sit within ``one_star_max_residual_px`` of the prediction. The residual on
this path is the claimed offset itself, and when nothing else in the window compares, a lone
detection far from the prediction is far more likely an unrelated artifact than a large
pointing error. The default bound of 10 px is one third of the 30 px window it was set
against.

The path reports the per-star residual (observed centroid minus predicted position) as the
translation. The reported covariance is the per-feature Cramer-Rao lower bound from
:attr:`~spindoctor.feature.feature.NavFeature.position_cov_px`, inflated by
``residual_px**2`` on the diagonal so a noisy match honestly reports its scatter rather
than the noise-free CRLB floor. Rotation is unobservable on a single point match.

A star's predicted SNR is synthesized from how far its catalog magnitude sits below the
frame's limiting magnitude, so it ranks the stars of a frame but says how bright a star
appears only loosely, and differently for each camera. The peak above the local
background per unit of predicted SNR of Cassini NAC stars found at the published bundles'
pointing on the F ring and spokes star frames has a median of 10 noise sigmas and ranges
from 0.7 to 126. So no lower bound applies, and the only absolute test is an upper bound a
camera may set in its instrument configuration (``star_max_peak_per_predicted_snr``, in
noise sigmas): a source brighter than the bound is not taken for a lone star, though it
still competes as a source of its window. Each bound is measured from the stars found on
the F ring and spokes star frames, and nothing yet derives it from the camera's
photometry. A camera that sets none has no bound; the instrument chapters give the
values.

2-star path
-----------

When at least two usable stars exist, the technique first attempts the 2-star path. Each
of the ``pair_search_stars`` brightest usable stars is searched for over the whole offset
window, and the window yields the star's candidates: its ``pair_search_sources`` strongest
isolated point sources (``window_sources`` in ``_star_helpers``). Each is a
matched-filter peak no sharper than a star, above the noise threshold, re-centered
without walking away, and isolated; a peak that fails any test is passed over and the
next one taken. The isolation test for a whole window is built once
(``isolation_lookup``), since a star smeared over tens of pixels widens its annulus enough
to make a test per peak cost seconds. A wide window often holds something stronger than
its star: the flank of a ring, a brighter star, or a cosmic-ray hit. Taking only each
window's strongest peak, and nothing when that peak was not isolated, refused frames whose
stars were plainly in the image once stars searched the Cassini NAC's (50, 140) px window:
222 of 1986 frames of the F ring and spokes movies that the search below answers, 215 of
them within 2 px of the published bundle's pointing and the other 7 where stars confirm
the answer over the bundle's, and library frame ``N1550270436``, whose body-centroid
fallback is then 15 px wrong. Where a star's source is found at all, it is the strongest,
second, or third isolated source of the star's window in all but 2 of 3108 Cassini NAC
star windows on F ring frames (2835, 254, and 17 times; fourth twice), so three candidates
per star hold it. Every further candidate is one more chance of a coincidence: with four
or five, two sources of F ring frame ``N1615481899`` ranked first and fourth agree 128 px
from the bundle's answer, within a factor of 4 of the right pair's brightness, and the
frame is refused as ambiguous.

Two catalog entries predicted within 1 px of each other (a star listed in two catalogs, or
listed twice) are one source in the image, and only the brighter is searched
(``distinct_stars``); counted twice, one star would back a pair twice, or one hidden star
would count as two absent ones. On four frames of one WAC observation a star listed in
both the bright-star and Tycho-2 catalogs, as its own rival, kept the 1-star path from
answering.

Every two candidates, belonging to two different stars, make a pair
(``pair_hypotheses`` in ``_star_unique_two_star``): each candidate is assigned to its own
star, the residual is half the difference between the two stars' offsets, and the pair
passes only when it is within ``max_residual_px``. A candidate in two stars' windows is
not paired with itself. A passing pair is backed by every searched star with a candidate
within twice ``max_residual_px`` of where the pair's solution puts it, its own two stars
among them; with a rotation fitted, the solution places each star with the fitted rotation
as well as the translation, so a star far from a short pair is placed where the roll
carries it. The true offset gathers every star whose source was found, while two
unrelated sources that happen to agree gather only themselves.

Pairs are judged best supported first. Of the pairs a set of agreeing stars forms, the one
nearest all of them is judged first: a star 1.8 px off where the others put it still pairs
with each of them, and a pair that includes it reports an offset 0.9 px off (on F ring
frame ``N1546714862`` the brightest star is such a star). The other searched stars refute a
pair's offset (``refute_pair``) when:

- they contradict it. The pair's two stars set a brightness
  scale: the geometric mean of their two quotients, times another star's predicted SNR, is
  the peak that star should show. Each other star is first sought where the pair's
  solution puts it, in a window of half-width twice ``max_residual_px``; an isolated point
  source found there votes for the pair, whatever else the star's own window holds, when
  its peak lies within a factor of ``pair_vote_peak_factor`` of the peak the star should
  show. A faint field star or a two-pixel cosmic-ray hit that lies there by chance is a
  small fraction of it and does not vote. A frame taken in high cosmic-ray flux holds hits
  brighter than many of its stars: on F ring frame N1799563905 the brightest source in
  each of three other stars' windows is such a hit, and the stars at the pair's offset,
  with peaks 0.68 to 2.04 times what the pair predicts, keep the pair, 0.12 px from the
  answer four other stars confirm. A star not found at the offset votes against the pair
  when exactly one of its candidates is as bright as it should be (``star_like_source``),
  and that one lies elsewhere and is not one of the pair's own two sources, which a
  fainter star predicted near a star of the pair can hold in its window; with no scale (a
  star of the pair whose catalog magnitude is only a saturated lower bound) every
  candidate qualifies, so only a window holding one candidate votes. A window that holds
  several sources the star could be does not locate it at all: on the Galileo black-sky
  frame ``C0164958745R`` every window holds saturated cosmic-ray hits as bright as its
  star. Overlapping windows can return one source for several stars, so sources within 1
  px of one another count once. With no vote for the pair, its offset is refuted when at
  least ``pair_contradiction_stars`` distinct sources vote against it. A simulated
  six-star field whose catalog positions each scatter by 6 or 8 px has two stars agreeing
  to 0.8 and 1.1 px, reported 5 and 7 px wrong without the refusal.
- more of the other searched stars are absent where the pair's solution puts them than are
  found there, and at least ``pair_absent_stars`` of them (``absent_stars``), or one when
  both of the pair's stars are predicted in the padding, where only a pointing error that
  carries both into the image puts them in it. A star is
  absent when it is at least as bright as the pair's fainter star, so it should be seen at
  least as well as a star already found; the solution puts it inside the image, not the
  padding; the pixels around that position are clear sky, their robust spread at most
  twice the image noise, so no ring, limb, or bright neighbor can hide it; and the
  brightest pixel there stands above its local background by less than its expected peak
  divided by ``pair_vote_peak_factor``. On the library frame ``N1598093741`` every star is
  smeared over 19 px, so no star is an isolated point source, and two sources that are not
  stars agree 45 px from the body techniques' answer; the two brightest stars, 4.5 and 2
  times as bright as the pair's brighter star, lie on clear sky there and show nothing. One
  absent star is not enough: on a dozen frames of ``ISS_006RI_LPHRLFMOV001_PRIME`` one
  catalog star brighter than the pair shows nothing at the right offset. On F ring frame
  ``N1755782055`` two stars predicted just past the image's last row pair two unrelated
  sources 85 px from the bundle's answer, and the brightest star, inside the image and on
  clear sky there, shows nothing; at the right offset it stands 3000 noise sigmas high.
  Three faint sources that agree with one another outnumber a bright pair, and the two
  bright stars, absent where the faint three put them, refute that offset.

A refuted offset sets aside every pair within twice ``max_residual_px`` of it
(``choose_pair``), since the votes and absences speak to the offset, not to the pair. The
answer is the best supported pair not set aside. When several pairs backed by equally many
stars survive more than twice ``max_residual_px`` apart, the stars are the brightest of
them, judged by the pair's peak per unit of predicted SNR: a ghost image of each star or a
coincidence among faint sources shows far less light. On F ring frame ``N1600240376`` each
star has a copy 15.5 rows and 11 columns away, 35 times fainter, and the copies form a
second pair backed by as many stars. Another surviving pair within a factor of
``pair_vote_peak_factor`` of the brightest one's peak per SNR leaves two answers the stars
support equally, one of them a coincidence that cannot be told from the other, and the
technique reports spurious with an ``ambiguous_pairs`` reason. A frame whose every pair is
ruled out reports spurious with a ``pair_contradicted`` reason and does not fall back to
the 1-star path.

The path does not apply the peak-to-runner-up ratio test to its stars' windows: the star
need not be the strongest source in its window, and a source that is not the star moves
that star's offset away from the other's, where the residual declines the pair, while the
peak ratio would refuse correct pairs. On ``N1530185128_1`` the fainter star's window
holds a single-pixel spike within 0.4 percent of the star's own peak, while the pair's
residual is under half a pixel.

Trying more than the brightest two matters when the brightest predicted star is not in the
image. A star predicted inside the extended field of view may fall off the detector once
the pointing error is applied, or under a ring's glare; with only the brightest two tried,
its window holds no star, the pair fails, and the frame falls through to the 1-star path,
where a field of comparable stars rarely leaves one source that only one of them could
be. On
``ISS_006RI_LPHRLFMOV001_PRIME`` searching pairs among the brightest 3, 5, and 8 stars
navigates 710, 771, and 777 of the 1030 frames, every one within 2 px of the published F
ring bundle (measured with each window's strongest detection in a 50 px window).

Every star is predicted inside the extended field of view, and its padding is the offset
window, so every star's window holds at least one pixel of the image. A star predicted
far into the padding is in the image only when the pointing error nearly fills the
window; until then it still takes a place among the brightest stars the pair search draws
on.

.. A star search no wider than the offset window has no star out of reach: #763.

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

- The brightness comparisons are driven by the catalog visual magnitudes: each star's
  ``predicted_snr`` is the magnitude-margin effective SNR derived from how far the star
  sits below the per-observation limiting magnitude, so the SNR ratio between two stars
  recovers their catalog magnitude difference. A wrong catalog magnitude shifts which
  star a lone source is taken for, how bright another star should be, and the brightness
  margin, and may erroneously accept or reject the match.
- The search window assumes the SPICE pointing error is bounded by the instrument's offset
  window. When the pointing error exceeds it, the star is not in its window. On
  ``ISS_006RI_LPHRLFMOV001_PRIME`` the Cassini NAC pointing error reaches 43 px, and a
  30 px window puts the stars of 191 of the 1030 frames out of reach. A 1-star match is
  accepted only inside the lone-star ellipse (the offset window's half-widths times the
  camera's ``one_star_ellipse_fraction``) around its prediction, while the whole window is
  still searched so that a source nearly as bright anywhere in it makes the match
  ambiguous: nothing cross-checks a single detection, and the wider the region it may come
  from, the likelier an artifact is the brightest thing in it. Replayed on the F ring and
  spokes star frames, the Cassini NAC's 334 one-star answers within 2 px of the published
  bundles' reach 0.55 of the window, and the 61 farther from them lie between 0.01 and 0.56
  of it, so a narrower ellipse would not separate the two. A camera whose pointing error
  reaches the edge of its window sets a wider fraction; the instrument chapters give the
  evidence. On the black-sky
  Galileo library frame
  ``C0164392700R`` a lone artifact lies 40 px from the one predicted star; the Galileo
  offset window of 350 px puts it inside the 210 px ellipse, so the ellipse does not refuse
  it there until the camera's window is set from its pointing error. Beyond the window is
  the regime where the multi-star
  :class:`~spindoctor.nav_technique.nav_technique_star_field.StarFieldFromCatalogNav` triplet
  matcher takes over.
- A lone star is judged against every source of the whole offset window, so a source
  nearly as bright anywhere in it refuses the match even where a window sized to the
  star's own reach would hold the star alone. On the 1986 F ring and spokes star frames,
  against star windows of 50 px (NAC) and 13 px (WAC) with lone stars believed within
  30 and 9 px, 17 NAC lone stars and 1 WAC lone star that those found are refused: on 15
  of the NAC frames a comparable source lies in the whole window outside the smaller one,
  or the offset lies outside the lone-star ellipse, and on the WAC frame another star
  explains another source nearly as well. In turn the whole window answers 310 NAC and 38
  WAC frames within 1 px of the bundle that those windows refuse.
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

- ``brightness_margin_to_next_catalog_star_mag`` — float, default ``1.5`` (mag). Another
  usable star within this many magnitudes of a lone star, whose window holds the lone
  star's source, could be that source as well, and the 1-star path refuses the match.
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
- ``pair_search_stars`` — int, default ``5``. The 2-star path searches this many
  brightest usable stars and pairs their candidates.
- ``pair_search_sources`` — int, default ``3``, at least 1. Number of strongest isolated
  point sources in each searched star's window that are its candidates. The star's own
  source ranked first, second, or third in all but 2 of 3108 star windows measured on F
  ring frames.
- ``pair_contradiction_stars`` — int, default ``3``. A pair's offset is refuted when at
  least this many distinct sources, each the one source in another searched star's window
  as bright as that star should be, lie away from the offset, and none of the other stars
  is found where the pair puts it.
- ``pair_absent_stars`` — int, default ``2``, at least 1. A pair's offset is refuted
  when at least this many other searched stars, each at least as bright as
  the pair's fainter star, lie on clear sky in the image at the pair's offset and show
  nothing there, and they outnumber the other stars found there. One absent star suffices
  when both of the pair's stars are predicted in the padding.
- ``pair_vote_peak_factor`` — float, default ``4.0`` (dimensionless), finite and above
  1. A source found where the pair's offset puts another star counts as that star only
  when its peak above the local background lies within this factor of the peak the
  pair's stars predict for it. Among 93 two-star answers in the F ring movies set aside
  for tuning, the 492 other stars at least a quarter as bright as the pair's fainter star
  showed peaks 0.35 to 2.8 times the prediction (2nd to 99th percentile), and 4 sets
  aside 9 of them. Of equally supported pairs, one within this factor of the brightest
  one's peak per unit of predicted SNR makes the choice ambiguous.
- ``centroid_box_half_px`` — int, default ``3`` px. Half-width of the brightness-weighted
  moment box around the detection, and of the matched-filter kernel.
- ``max_residual_px`` — float, default ``1.2`` px. Maximum allowed
  residual for a pair to pass the 2-star cross-check. A star is found at a pair's offset
  within twice this, and pairs farther apart than twice this are different answers.
- ``detection_sigma`` — float, default ``4.0`` (dimensionless). Detection-threshold
  multiplier on the per-image noise sigma; a detection's brightest pixel must stand this
  many noise sigmas above the median of its centroid box.
- ``one_star_confidence_cap`` — float, default ``0.7`` (dimensionless). Post-sigmoid
  confidence cap when the 1-star path fires. A single match cannot self-check.
- ``two_star_confidence_cap`` — float, default ``0.8`` (dimensionless). Post-sigmoid
  confidence cap when the 2-star path fires. Per-correspondence residual cross-checks the
  assignment.
- ``one_star_min_peak_ratio`` — float, default ``1.5`` (dimensionless). Minimum ratio of
  a lone star's source's peak above its local background to that of every other isolated
  point source of the star's window; below it the result is spurious with an
  ``ambiguous_detection`` reason. Values below 1 are invalid.
- ``one_star_max_peak_sharpness`` — float, default ``7.0``, finite and at least 1. Largest
  ratio of a lone star's brightest-pixel excess to the mean excess of its four edge
  neighbors, allowing nothing for noise; a sharper source is no lone star.
- ``one_star_absent_stars`` — int, default ``1``, at least 1. A lone star's offset is
  refuted when at least this many other usable stars lie on clear sky in the image where
  it puts them and show nothing there.
- ``one_star_max_residual_px`` — float, default ``10.0`` px. Maximum
  prediction-to-detection distance accepted by the 1-star path when the peak-ratio gate
  is vacuous (its no-rival ``inf`` sentinel); beyond it the result is spurious with a
  ``no_rival_detection`` reason. Default is one third of the 30 px window half-width it
  was set against (the ~1-sigma core of the pointing envelope that window brackets).
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
override any of these knobs. Two settings of the 1-star path are properties of the camera
and live only there; the technique's own configuration sets neither:

- ``one_star_ellipse_fraction`` — float, required of every camera block (every mapping
  that declares ``data_units``, the simulator's included). A 1-star detection outside the
  ellipse whose semi-axes are the offset window's half-widths times it is refused with a
  ``one_star_detection_outside_ellipse`` reason. The ellipse is taken in place of a single
  radius because the offset window is per axis: it is a circle for a square window and
  never reaches past the window on either axis. Loading the configuration refuses a
  camera block that sets none, or one outside ``(0, 1]``, and the technique checks the
  value again when it reads it, since a simulated scene may override the camera's
  configuration. The instrument chapters give each camera's value and the configuration
  carries its evidence.
- ``star_max_peak_per_predicted_snr`` — float, optional. Bounds a lone star's source's peak
  per unit of predicted SNR in noise sigmas (see the 1-star path above); a camera that
  sets none has no bound.

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
  — alpha = 1.002, offset = 1.5, divisor = 1.5, cap at 1.0. Magnitude margin from the
  matched star to the brightest other usable star in the extfov. A margin below the
  1.5 mag offset, negative when a brighter star is not seen, contributes nothing (the
  term is clamped at 0); above it, additional margin earns confidence up to a 1.5 mag
  span.
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
- ``src/spindoctor/nav_technique/_star_unique_two_star.py`` — package-private pair
  arithmetic of the 2-star path: the pair search and its support (``pair_hypotheses``,
  ``solve_assigned_pair``, ``similarity_fit_assignment``), the judgment of the pairs
  (``choose_pair``, ``pair_brightness``, ``refute_pair``) and the other stars' votes it
  rests on (``third_star_votes``, ``star_like_source``, ``absent_stars``, ``peak_per_snr``,
  ``source_peak``), and the match covariances (``build_star_covariance``,
  ``two_star_covariance``, ``two_star_covariance_3dof``).
- ``src/spindoctor/nav_technique/_star_helpers.py`` — package-private helpers ``usable_stars``,
  ``distinct_stars``, ``search_window_bounds``, ``local_centroid``, ``window_sources``,
  ``detection_peak_ratio``, ``too_sharp``, ``gaussian_psf_kernel``, ``predicted_snr``,
  ``predicted_vu``, ``brightness_margin_mag``, and ``similarity_transform_fit``.
- ``src/spindoctor/nav_technique/_star_background.py`` — package-private plane background
  under a centroid box (``box_background``, ``peak_excess``, ``region_centroid``).
- ``src/spindoctor/nav_technique/_star_unique_one_star.py`` — package-private lone-star
  search of the 1-star path: ``lone_star_hypotheses`` (each searched star paired with
  each of its candidates, judged by sharpness and by absent stars) and
  ``choose_lone_star``.
- ``src/spindoctor/nav_technique/_star_isolation.py`` — package-private source isolation
  test shared with the star field: ``isolation_radii``, ``isolated_response``,
  ``is_isolated_point_source``, and ``isolation_lookup``.
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
  predicted SNR of the matched catalog star (the brighter of a pair). Consumed by the
  confidence formula.
- :attr:`~spindoctor.nav_technique.diagnostics.StarUniqueMatchDiagnostics.brightness_margin_mag`
  — magnitude difference to the next-brightest unmatched catalog source predictable in
  extfov; ``+inf`` when no unmatched star exists. Consumed by the confidence formula.
- :attr:`~spindoctor.nav_technique.diagnostics.StarUniqueMatchDiagnostics.residual_px` —
  detection-vs-prediction residual. Consumed by the confidence formula and by the
  spurious-detection gate.
- :attr:`~spindoctor.nav_technique.diagnostics.StarUniqueMatchDiagnostics.detection_peak_ratio`
  — background-subtracted peak of the 1-star path's source over the brightest pixel of
  the rest of its window (``0.0`` when not measured; ``inf`` when that pixel sits at or
  below the background). Diagnostic; the no-rival residual bound applies when it is
  ``inf``.

Call path
---------

Call path traced through
:meth:`~spindoctor.nav_technique.nav_technique_star_unique_match.StarUniqueMatchNav.navigate`:

1. Open a logged section. Filter the offered features down to ``usable_stars``
   (predictable and not occluded by a body silhouette or ring annulus).
2. Sort the usable cohort by predicted SNR (brightest first) and compute the brightness
   margin between consecutive entries via ``brightness_margin_mag``.
3. Branch on the usable-star count:

   - **2-star path.** Attempted first whenever at least two usable stars exist (no
     brightness-margin precondition). For each of the ``pair_search_stars`` brightest
     stars, call ``window_sources`` on its offset window in
     :attr:`~spindoctor.nav_orchestrator.nav_context.NavContext.image_ext` for its
     candidates. Pass them to ``pair_hypotheses``, which solves every pair of candidates
     of two different stars (with ``similarity_transform_fit`` when rotation is fitted),
     keeps the pairs whose residual is within ``max_residual_px``, and orders them by
     support. ``choose_pair`` judges them best supported first, sets aside every pair near
     an offset ``refute_pair`` refutes, and chooses the brightest of the equally supported
     survivors. Report it, or a spurious result when another survivor is within
     ``pair_vote_peak_factor`` of its brightness or every pair is refuted; when no pair
     passes the residual cross-check, fall back to the 1-star path.
   - **1-star path.**  When fewer than two stars are usable (or no pair passed), pair
     each searched star with each of its candidates (``lone_star_hypotheses``), judge
     each by its sharpness and by the other stars absent where it puts them, and choose
     the survivor brightest per unit of predicted SNR (``choose_lone_star``). Refuse it
     when another source of its window is within ``one_star_min_peak_ratio`` of its peak,
     another star's survivor is within ``pair_vote_peak_factor`` of its brightness,
     another comparable star's window holds its source, it lies outside the lone-star
     ellipse, or, when the window's pixel-level peak ratio is the no-rival sentinel, it
     lies beyond ``one_star_max_residual_px`` of the prediction; otherwise report the
     per-star residual as the translation.

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
    1-star path: Vega's offset window holds one isolated source far brighter than any
    other, the brightness-weighted moment recovers its centroid, and the per-star residual
    is reported as the translation. The 1-star post-sigmoid cap of
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
