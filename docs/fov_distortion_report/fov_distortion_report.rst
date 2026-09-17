=========================================
FOV Distortion and Camera-Twist Report
=========================================

This chapter reports how much residual camera-pointing error survives, per
instrument, after the SpinDoctor navigation pipeline applies the geometry it
already knows. It measures two quantities directly from star fields: the
rotational error of the field of view (the camera twist) and the lateral
distortion that remains after the known distortion model is removed. It is a
standalone chapter, separate from the user and developer guides, and is
regenerable on demand from the star-frame cohorts and the analysis driver.

The star field is the one scene with per-pixel ground truth good to a fraction
of a pixel across the whole frame, so it is the natural instrument for both
measurements: every catalog star is a point whose true image position is known
independently of the spacecraft pointing.

.. contents::
   :local:
   :depth: 2

Purpose and scope
=================

The oops geometry library implements the per-instrument distortion model for
every supported camera and applies it wherever pixel coordinates and look
vectors are interconverted. Navigation therefore already corrects the *known*
distortion. What remains unmeasured is the **residual**: the field-position
dependent error left after that model is applied, and the frame-to-frame
rotation of the field that a translation-only pointing solution never removes.
This report quantifies both, per instrument, and states what each result means
for two consumers.

**Camera twist.** A single rigid rotation of the field about the optical center.
Its value on one frame is one measurement; the property that matters is whether,
across many frames of an instrument, the twist is a single common value or
scatters:

* A common value is a static camera-frame alignment error. It can be folded into
  a corrected pointing kernel once and removed for every image, after which
  per-frame rotation fitting during navigation buys nothing.
* A scatter that is large in its own right is genuine per-frame attitude error.
  No static kernel can remove it, so navigation must fit the rotation per frame
  where accuracy at the field edge matters.

The pipeline can fit a per-frame camera rotation, but the fit is slow, so it is
enabled per instrument only where it earns its cost. This report produces a
per-instrument recommendation for that setting grounded in the measured twist
scatter.

**Lateral residual distortion.** The displacement of each star that remains after
the pointing translation and the twist are removed. Its radial part is fitted
with the same low-order polynomial the simulator distortion stage plants, so the
measured coefficients feed the simulator's per-instrument defaults directly; its
non-radial part is reported as a scalar amplitude.

Methodology
===========

Per-frame measurement
---------------------

For one image the analysis:

1. Loads the observation and runs star-only navigation to recover the
   translation prior -- the residual spacecraft pointing offset ``(dv, du)``.
2. Predicts every catalog star the star model would use and, shifting each
   prediction by the pointing offset, fits a sub-pixel Gaussian centroid to the
   image at that location.
3. Rejects stars with no detectable peak, a centroid that ran to the search
   limit, or a post-fit residual far larger than the rest.

The surviving predicted / detected pairs are the per-star residual field. The
navigation offset is used only to place the centroid search windows; the twist
and distortion are fitted from the pairs themselves, so the measurement does not
depend on navigation having fitted a rotation. The rotation fit is forced on for
the measurement regardless of the instrument's shipped rotation-fitting setting,
because that setting is an output of this study rather than an input.

Decomposition
-------------

The total predicted-to-detected displacement is separated into three physically
distinct parts:

* A **global translation** -- the residual pointing offset that plain navigation
  already removes. It is not a camera defect and is discarded.
* A **rigid rotation about the optical center** -- the twist -- fitted by a
  weighted orthogonal-Procrustes (Kabsch) solution for the proper rotation and
  translation that best map the catalog cloud onto the detected cloud.
* The **residual after translation and rotation are removed** -- the lateral
  distortion. Its radial component (the projection onto the outward line from the
  optical center) is fitted against a low-order polynomial in the normalized
  field radius; its tangential component's RMS is reported as the non-radial
  wander.

A radial distortion field and a rigid rotation are not quite orthogonal -- a
strong radial term biases a single-pass rotation fit and vice versa -- so the
two fits are alternated until they converge, which takes a couple of iterations.

The radial model is matched to the simulator distortion warp
``source = center + (p - center) * (1 + k1 * rho_n**2 + k2 * rho_n**4)``, where
``rho_n`` is the field radius normalized by half the image diagonal. Under that
warp the radial displacement in pixels is
``rho_ref * (k1 * rho_n**3 + k2 * rho_n**5)``, so fitting the radial residual on
the ``rho_n**3`` and ``rho_n**5`` basis recovers ``k1`` and ``k2`` directly.

The distortion stage works in pixel-centric coordinates on the render grid, and
``center`` and ``p`` are positions there, while a scene states its optical
center as ``optics.distortion.center_v`` / ``center_u`` in pixel-corner
coordinates on the detector grid and the stage converts it. See
:ref:`coordinate-systems`.

Per-instrument aggregation
--------------------------

The per-frame twists are combined into an inverse-variance-weighted mean and a
frame-to-frame scatter. The verdict keys on the scatter expressed as its
displacement at the field corner in pixels -- the quantity navigation actually
cares about -- rather than on the formal chi-square. A very precise per-frame
twist fit drives the reduced chi-square large on a scatter that is operationally
negligible, so the chi-square is reported as a diagnostic but does not decide the
verdict. When the corner-displacement scatter is below a small fraction of a
pixel the twist counts as a single common value; otherwise it is called
frame-varying. The per-star radial residuals of every frame are pooled and fitted
once to give the instrument's aggregate radial distortion model.

The recommendation follows from the verdict: a frame-varying twist recommends
per-frame rotation fitting; a consistent twist large enough to displace the field
corner meaningfully recommends a fixed kernel correction with rotation fitting
left off; a consistent and negligible twist recommends neither.

Separating distortion from centroiding noise
--------------------------------------------

Two effects both grow with field radius and must not be conflated. Star-catalog
astrometric error is field-position-independent and averages down across a frame.
Point-spread-function centroiding error, in contrast, grows toward the corners,
exactly where the distortion is largest, because a star's image there is sampled
by fewer well-behaved pixels. The report presents the post-fit residual RMS
against field radius so the centroiding floor is visible as a separate curve; the
centroiding contribution is modeled explicitly and not assumed separable from the
distortion.

Per-instrument results
======================

.. note::

   The tables and figures in this section are regenerated by the analysis
   driver over the committed star-frame cohorts. See :ref:`fov-running` for the
   command. The measurements carry sub-millipixel numerical jitter across runs;
   treat the twist and distortion values as the stable quantity and the final
   digit as indicative.

Camera twist
------------

The twist is expressed both as an angle and as the displacement it produces at
the field corner (half the image diagonal), which is the quantity navigation
sees. "Frames locked" is the number of cohort frames on which star navigation
locked and enough stars survived the centroid and residual gates.

.. list-table:: Per-instrument camera twist
   :header-rows: 1
   :widths: 26 12 20 12 12 18

   * - Instrument
     - Frames locked
     - Mean twist (deg)
     - Corner (px)
     - Scatter (px)
     - Verdict
   * - Cassini ISS NAC
     - 50 / 50
     - +0.0115 +/- 0.0002
     - +0.15
     - 0.039
     - consistent
   * - Cassini ISS WAC
     - 46 / 50
     - -0.0115 +/- 0.0002
     - -0.15
     - 0.037
     - consistent
   * - Galileo SSI
     - 7 / 18
     - -0.0526 +/- 0.0017
     - -0.52
     - 0.092
     - consistent
   * - New Horizons LORRI (pre-KE)
     - 16 / 48
     - +0.1912 +/- 0.0006
     - +2.42
     - 0.027
     - consistent
   * - New Horizons LORRI (post-KE)
     - 0 / 50
     - unmeasured
     - --
     - --
     - no pointing-kernel coverage
   * - Voyager 1 ISS NAC
     - 0 / 1
     - unmeasured
     - --
     - --
     - no locked frame
   * - Voyager 2 ISS NAC
     - 5 / 27
     - +0.030 +/- 0.005
     - +0.37
     - 0.283
     - frame-varying
   * - Voyager 2 ISS WAC
     - 14 / 45
     - +0.358 +/- 0.005
     - +4.42
     - 0.275
     - frame-varying

The three regimes appear directly in the sample figures. A well-aligned camera
(Cassini ISS NAC) shows a residual field with no coherent rotation once the
translation is removed:

.. figure:: _figures/coiss_nac_sample.png
   :width: 100%
   :alt: Cassini ISS NAC residual field, twist removed, and radial profile.

   Cassini ISS NAC. Left: the per-star residual field with the mean translation
   removed, arrows magnified. Middle: the residual after the fitted twist is
   removed. Right: the radial residual against field radius with the fitted
   model. The twist is negligible and the distortion is a few hundredths of a
   pixel.

.. figure:: _figures/coiss_nac_twist.png
   :width: 100%
   :alt: Cassini ISS NAC per-frame twist across the cohort.

   Cassini ISS NAC per-frame twist. The frame-to-frame scatter (0.039 px at the
   corner) is far below the threshold, so the small mean twist is one common
   value.

A camera with a real static twist (New Horizons LORRI, pre-KE) shows a coherent
rotation in the residual field that the same value removes on every frame:

.. figure:: _figures/nhlorri_pre_ke_sample.png
   :width: 100%
   :alt: New Horizons LORRI residual field, twist removed, and radial profile.

   New Horizons LORRI (pre-KE). The residual field carries a coherent rotation
   (a +0.19 deg twist, +2.4 px at the corner) that the fit removes cleanly,
   leaving a small distortion residual.

.. figure:: _figures/nhlorri_pre_ke_twist.png
   :width: 100%
   :alt: New Horizons LORRI per-frame twist across the cohort.

   New Horizons LORRI (pre-KE) per-frame twist. The twist sits tightly on a
   single non-zero value (corner scatter 0.027 px): a static camera-frame
   alignment error.

A camera whose twist varies frame to frame (Voyager 2 ISS WAC) shows a large
rotation whose value changes from image to image:

.. figure:: _figures/vg2iss_wac_sample.png
   :width: 100%
   :alt: Voyager 2 ISS WAC residual field, twist removed, and radial profile.

   Voyager 2 ISS WAC. The residual field carries a large rotation, and a large
   distortion residual, reflecting the resampled vidicon geometry.

.. figure:: _figures/vg2iss_wac_twist.png
   :width: 100%
   :alt: Voyager 2 ISS WAC per-frame twist across the cohort.

   Voyager 2 ISS WAC per-frame twist. The twist scatters by 0.28 px at the
   corner, well above the threshold: genuine per-frame attitude error that no
   static kernel removes.

The remaining locked instruments are shown below for completeness.

.. figure:: _figures/coiss_wac_sample.png
   :width: 100%
   :alt: Cassini ISS WAC residual field, twist removed, and radial profile.

   Cassini ISS WAC. A negligible common twist, opposite in sign to the NAC, and
   a small distortion.

.. figure:: _figures/gossi_sample.png
   :width: 100%
   :alt: Galileo SSI residual field, twist removed, and radial profile.

   Galileo SSI. A consistent -0.05 deg twist across the frames that lock.

.. figure:: _figures/vg2iss_nac_sample.png
   :width: 100%
   :alt: Voyager 2 ISS NAC residual field, twist removed, and radial profile.

   Voyager 2 ISS NAC. A frame-varying twist and a large distortion residual, as
   for the WAC.

Lateral residual distortion
---------------------------

The radial distortion coefficients are reported in the simulator distortion
stage's convention (``k1`` on ``rho_n**2``, ``k2`` on ``rho_n**4``, with
``rho_n`` the field radius over half the image diagonal). The radial and
non-radial RMS are the amplitudes of the distortion signal; the residual floor
is the median per-frame RMS after the radial model is removed, set by centroid
and star-catalog noise.

.. list-table:: Per-instrument residual distortion
   :header-rows: 1
   :widths: 26 14 14 16 16 14

   * - Instrument
     - k1
     - k2
     - Radial RMS (px)
     - Non-radial RMS (px)
     - Floor (px)
   * - Cassini ISS NAC
     - +3.2e-04
     - -3.5e-04
     - 0.048
     - 0.068
     - 0.081
   * - Cassini ISS WAC
     - +9.0e-06
     - +5.5e-05
     - 0.086
     - 0.073
     - 0.053
   * - Galileo SSI
     - -3.5e-04
     - +1.7e-03
     - 0.155
     - 0.100
     - 0.115
   * - New Horizons LORRI (pre-KE)
     - +8.1e-04
     - -1.1e-03
     - 0.078
     - 0.055
     - 0.082
   * - Voyager 2 ISS NAC
     - -5.6e-03
     - +8.2e-03
     - 0.936
     - 0.268
     - 0.255
   * - Voyager 2 ISS WAC
     - -6.9e-03
     - +1.5e-02
     - 0.345
     - 0.260
     - 0.336

The pooled radial profile and the residual-floor curve are shown for a
well-behaved camera and for the worst case:

.. figure:: _figures/coiss_nac_radial.png
   :width: 100%
   :alt: Cassini ISS NAC pooled radial distortion and residual floor.

   Cassini ISS NAC. Left: the pooled per-star radial residual against field
   radius with the aggregate model. Right: the post-fit residual RMS against
   radius, the centroid-plus-astrometric floor. The distortion is a few
   hundredths of a pixel and the floor rises modestly toward the edge.

Between those extremes, Galileo SSI carries the most significant radial
distortion of the well-behaved cameras. The pooled radial residual climbs
steadily toward the edge and reaches about half a pixel at the field corner, a
``k2``-dominated (pincushion) shape well above the ~0.1 px floor: a real
distortion the navigator does not remove.

.. figure:: _figures/gossi_radial.png
   :width: 100%
   :alt: Galileo SSI pooled radial distortion and residual floor.

   Galileo SSI. The radial residual rises to ~0.5 px at the field corner,
   dominated by the ``k2`` term (``k1`` -3.5e-04, ``k2`` +1.7e-03). Pooled over
   the frames that lock, the trend is clear despite the small cohort.

New Horizons LORRI carries only a small radial distortion: a low hump peaking
near 0.06 px at mid-field, comparable to its own centroid-plus-astrometric
floor, so the radial term is barely above the noise. LORRI's significant
signature is the twist, not the distortion.

.. figure:: _figures/nhlorri_pre_ke_radial.png
   :width: 100%
   :alt: New Horizons LORRI pooled radial distortion and residual floor.

   New Horizons LORRI (pre-KE). The radial residual is a ~0.06 px mid-field hump
   (``k1`` +8.1e-04, ``k2`` -1.1e-03) sitting close to the floor; the downturn
   past the outermost radius is the low-order fit extrapolating beyond the last
   stars.

.. figure:: _figures/vg2iss_wac_radial.png
   :width: 100%
   :alt: Voyager 2 ISS WAC pooled radial distortion and residual floor.

   Voyager 2 ISS WAC. The radial distortion reaches several tenths of a pixel
   and the floor is high, both consequences of the resampled vidicon geometry.

Non-rotational distortion field
-------------------------------

The radial model captures only the rotationally symmetric part of the
distortion. The full non-rotational distortion -- everything left after the
translation and the twist are removed -- also carries a non-radial part:
tangential distortion and optical decentering, which a radial polynomial cannot
represent. Pooling every star's post-twist residual and averaging it into a grid
over the field of view maps that structure directly. The left panel of each
figure is the full non-rotational field; the right panel is the non-radial
component alone, with each residual's radial projection removed.

.. figure:: _figures/coiss_nac_distortion_map.png
   :width: 100%
   :alt: Cassini ISS NAC non-rotational distortion field and non-radial component.

   Cassini ISS NAC. The full field (left) is a few hundredths of a pixel with
   little coherent structure, and the non-radial component (right) is at the
   noise floor: the residual distortion is small and close to radially
   symmetric.

.. figure:: _figures/vg2iss_wac_distortion_map.png
   :width: 100%
   :alt: Voyager 2 ISS WAC non-rotational distortion field and non-radial component.

   Voyager 2 ISS WAC. The full field (left) reaches several tenths of a pixel,
   and the non-radial component (right) is a substantial fraction of it: the
   resampled vidicon geometry carries coherent tangential distortion, not just a
   radial term.

Interpretation and recommendations
==================================

Rotation fitting during navigation
----------------------------------

* **Cassini ISS NAC and WAC.** The twist is one common value near +/-0.01 deg,
  which displaces the field corner by ~0.15 px, and its frame-to-frame scatter
  is below 0.04 px. Per-frame rotation fitting is unnecessary and no kernel
  correction is warranted; the shipped setting with rotation fitting off is
  correct. The residual distortion is a few hundredths of a pixel, and the
  star-technique accuracy floor is 0.05 to 0.08 px.
* **New Horizons LORRI (pre-KE).** The twist is a clean, common +0.191 deg
  (+2.4 px at the corner) with a corner scatter of only 0.027 px: a static
  camera-frame alignment error. It is a strong candidate for a fixed pointing
  kernel correction, after which per-frame rotation fitting is not needed. The
  post-KE epochs fall outside the loaded pointing-kernel coverage and are
  unmeasured; the residual there stands on the adopted literature distortion
  model.
* **Galileo SSI.** The twist is a common -0.053 deg (-0.5 px at the corner)
  across the frames that lock, which points to a static kernel correction rather
  than per-frame fitting. The locked cohort is small (seven frames), so this is
  provisional pending a larger star-frame set.
* **Voyager 2 ISS NAC and WAC.** The twist is frame-varying: the corner scatter
  is 0.28 px, well above the threshold, and for the WAC the mean twist is large
  (+0.36 deg, +4.4 px at the corner). No static kernel removes a frame-varying
  twist, so navigation must fit the rotation per frame for these cameras. This
  is the measured basis for enabling per-frame rotation fitting on Voyager, and
  the timing below shows that fit adds negligible time on star fields. The
  residual distortion is large, consistent with the resampled vidicon geometry,
  and the residual floor is high (0.25 to 0.34 px), so the Voyager numbers carry
  lower confidence than their residual floor alone would suggest.

Feeding the simulator and the pointing kernels
----------------------------------------------

The per-instrument radial coefficients ``k1`` and ``k2`` are reported in the
simulator distortion stage's convention and populate that stage's per-instrument
residual-distortion defaults. The simulator's non-radial wander amplitude is set
separately: it ships at 0.0 px for Cassini ISS NAC, Cassini ISS WAC, Galileo SSI
and New Horizons LORRI, so the non-radial term is off for those four, and at
0.2 px for the Voyager entry. The radial amplitudes differ sharply between
cameras: Cassini and New Horizons LORRI
sit at or near their noise floors (a few hundredths of a pixel), while Galileo
SSI reaches ~0.5 px at the field corner and Voyager several tenths of a pixel, so
those two set meaningfully larger distortion defaults. The simulator keys
Voyager under a single camera, so it adopts the Voyager 2 wide-angle block until
per-camera keys exist; the confidence calibration and the operator-curated image
library do not use the instrument-default distortion, so updating it leaves them
untouched. The
consistent twists (New Horizons LORRI, Galileo SSI) are the static corrections a
future instrument or frame kernel would absorb; they are recorded here with
their uncertainties so a kernel build can adopt them. The frame-varying Voyager
twists are not kernel-correctable and are recorded instead as the justification
for per-frame rotation fitting.

Cost of per-frame rotation fitting
----------------------------------

Because the Voyager twist is frame-varying, navigation must fit the rotation per
frame -- but only if that fit is affordable. Running the main navigation pipeline
over the locked Voyager 2 frames twice, with rotation fitting off and on,
measures that cost directly. On these star fields the fit is essentially free:

.. list-table:: Main-pipeline navigation time on locked Voyager 2 frames
   :header-rows: 1
   :widths: 30 22 22 22

   * - Cohort
     - Rotation off (s/frame)
     - Rotation on (s/frame)
     - Added by rotation
   * - Voyager 2 ISS WAC + NAC (19 frames)
     - 4.61
     - 4.61
     - -0.02 (x0.99)

The star-technique rotation fit is a small similarity refit on the matched
stars, not a search, so it adds a negligible fraction of the per-frame time
(median 4.61 s either way), which is dominated by image loading and the shared
per-image derivatives. All nineteen locked frames navigate successfully with
rotation fitting on and recover a per-frame twist of 0.02 to 0.51 deg. The cost
objection to rotation fitting does not hold for star-field navigation; it
would need separate measurement on frames whose navigation rests on the
distance-transform body and ring techniques, where the rotation enters a
higher-dimensional fit.

Confidence and yield
--------------------

The Cassini cohorts lock nearly every frame and give the most trustworthy
numbers. The other instruments lock a fraction of their star-frame cohorts, so
their aggregate values rest on smaller samples: Galileo and Voyager 2 NAC in
particular carry single-digit frame counts and are provisional. Where an
instrument's residual floor approaches its measured distortion amplitude
(Voyager), the distortion is an upper bound rather than a precise value; the
twist verdict remains robust because the frame-to-frame scatter exceeds the
per-frame twist uncertainty by several fold.

Frames that do not lock
=======================

A frame contributes a measurement only if star-only navigation locks and enough
stars survive the centroid and residual gates. Star navigation locks a fraction
of each star-frame cohort: the Cassini cohorts lock nearly every frame, while
Galileo, New Horizons LORRI, and Voyager lock a smaller subset, and the single
Voyager 1 NAC frame does not lock at all. Two causes dominate the misses --
frames whose star field is too sparse or faint for a confident lock, and frames
whose epoch falls outside the loaded pointing-kernel coverage (the New Horizons
post-Kuiper-extension frames, and the Voyager 1 frames whose kernels end in
1980). The per-frame CSV records the status of every attempted frame, and the
locked count per instrument appears in the tables above. Where an instrument
yields too few locked frames to characterize its twist and distortion, the
residual is documented as under-measured rather than estimated, and the adopted
literature distortion model stands.

Each instrument also carries a larger candidate frame list. Running those
candidate lists adds very little: Voyager 2 NAC locks 5 of 255 and Voyager 2 WAC
15 of 575 (the same handful the
vetted lists find, at the same twists), while the Voyager 1 NAC and WAC
candidate lists lock none of 234 and none of 57 -- their frames either fail
navigation on sparse, faint fields or fall outside kernel coverage. The
candidate lists therefore do not materially extend the Voyager measurement.

The Voyager entries also under-represent the instrument spread: Voyager 1 and
Voyager 2 flew separate cameras, and each mission's narrow- and wide-angle
cameras have distinct optics. Each is a separate cohort here, but only Voyager 2
locks enough frames to characterize, and its narrow- and wide-angle results
differ (the wide-angle twist and distortion are the larger). A full Voyager
characterization needs a per-camera locked set for all four cameras.

.. _fov-running:

Regenerating this report
========================

The analysis lives in ``util/fov_distortion/``. After configuring the navigation
environment, from the repository root:

.. code-block:: bash

   python util/fov_distortion/run.py util/fov_distortion/configs/coiss_nac.yaml --report-figures

Each cohort YAML in ``util/fov_distortion/configs/`` names one instrument and
camera's star frames. The driver writes a per-frame CSV, a JSON summary, and the
figures embedded above; ``--report-figures`` writes the figures into this
chapter's ``_figures`` directory.
