==========================================================
Body Blob Centroid (BodyBlobNav)
==========================================================

Overview
========

:class:`~spindoctor.nav_technique.nav_technique_body_blob.BodyBlobNav` recovers a single
translation from one or more body brightness centroids. For each offered ``BODY_BLOB``
feature the technique computes a brightness-weighted moment inside the predicted bounding
box, compares the observed centroid to the predicted lit-weighted centroid carried on the
feature, and runs a covariance-weighted joint fit across every consumed body to recover
the per-image translation. With ``N >= 2`` blobs the fit is over-determined, and blobs
that disagree by more than their covariances allow widen the reported covariance; an
outlier blob is not rejected.

Feasibility passes when at least one offered ``BODY_BLOB`` carries a non-zero predicted
diameter; feasibility fails when every blob has degenerate geometry (a sub-pixel body that
collapses the brightness-weighted moment).

The technique reports a confidence intrinsically capped at 0.4 via the spec's
:attr:`~spindoctor.nav_technique.confidence.ConfidenceSpec.hard_cap` — a brightness-weighted
centroid is much weaker evidence than a limb fit, so even an ideal blob match cannot dominate
the ensemble combine.

Theory
======

The technique fits a per-image translation by minimizing the covariance-weighted
squared residual between the per-blob observed and predicted centroids.

Coarse acquisition (lit-shape matched filter)
---------------------------------------------

A brightness-weighted moment only sees the body when it already sits inside the predicted
bounding box, so the bare centroid's capture range is just the box -- a few pixels of
per-body slop. Once the SPICE pointing error exceeds that slop the body drifts out of the
box, the moment is taken over a clipped fragment, and the technique reports a *silently*
biased centroid (no spurious or at-edge flag fires). To extend the capture range to the full
extended-FOV search window, each blob first runs a coarse acquisition that re-centers its
bounding box on the body before the centroid is taken:

- If a pass-1 prior offset is installed on the context (another technique already located the
  body), the box is shifted by the rounded prior. The prior is a measured offset, so it
  applies regardless of phase.
- Otherwise the technique correlates a matched-filter template of the predicted *lit
  silhouette* against the lit-signal image (background subtracted, clipped at zero,
  sky-masked) over ``predicted_center +/- margin``. The response peaks where a body of that
  shape is best centered; the integer peak offset re-centers the box. The template depends on
  phase (:data:`~spindoctor.nav_technique.nav_technique_body_blob._COARSE_CORRELATION_MAX_PHASE_DEG`,
  90 deg):

  - **At or below half phase** the lit silhouette is a near-full disc, so the kernel is a
    filled disc of the predicted body radius.
  - **Above half phase** the sunlit region is a thin crescent whose bright pixels sit a
    fraction of a radius off the body center; a disc kernel would lock onto the crescent arc
    rather than the center. The kernel is instead a *synthesized crescent* -- a Lambertian
    ``max(0, cos(incidence))`` rendering of a sphere of the predicted radius at the body's
    phase, lit from the sub-solar direction the ``BODY_BLOB`` feature carries
    (``sub_solar_dir_vu``, the projected body-to-Sun direction; see
    :doc:`dev_guide_navigation_models_body`). Correlating the crescent puts the template
    *center* on the body center instead of the bright arc.

  The kernel is flipped before the FFT so the operation is a cross-correlation, and the
  template's own brightness-centroid offset is added back to the peak: the feature carries the
  body's *lit* centroid (which on a crescent sits off the geometric center), so the recovered
  shift is expressed in lit-centroid terms and matches the residual the centroid step forms.

Both templates clamp their radius to the frame's half-diagonal
(``_clamped_kernel_radius``): a template
larger than the frame adds no localization information, while the kernel array and its FFT
convolution allocate memory quadratically in the predicted diameter — a mostly off-frame gas
giant predicts tens of thousands of pixels, enough to exhaust RAM without the clamp.

The crescent template needs the sub-solar direction. It is undefined near full phase (the lit
and geometric centroids coincide), where the disc kernel is used anyway, and is reported as
``(0, 0)`` then. If a body is past half phase yet carries no direction (its illumination
geometry was not populated), the coarse stage makes no relocation and keeps the predicted box
(an installed prior still applies). The coarse offset is integer; the sub-pixel precision
comes entirely from the brightness-weighted moment below, computed inside the re-centered box,
so the recovered ``observed - predicted`` residual already includes the coarse shift.

Per-blob centroid
-----------------

For each consumed body, the technique computes the brightness-weighted moment over the
(coarse-re-centered) predicted bounding box:

.. math::

    \bar{x}_{\mathrm{obs}} =
        \frac{\sum_{(v, u) \in \mathrm{bbox}} I(v, u) \,(v, u)}
             {\sum_{(v, u) \in \mathrm{bbox}} I(v, u)}.

The predicted centroid carried on the feature is the lit-weighted moment of the rendered
model (see :doc:`dev_guide_navigation_models_body`); the per-blob residual is

.. math::

    r_{i} = \bar{x}_{\mathrm{obs},\,i} - \bar{x}_{\mathrm{pred},\,i}.

Per-blob covariance
-------------------

Each blob's centroid carries photon noise and a model bias. The photon part is the
standard CRLB scaling for a uniform-brightness disc:

.. math::

    \sigma_{\mathrm{photon}} \approx
        \frac{D_{\mathrm{px}}}{2 \, \sqrt{N_{\mathrm{lit}}} \, \mathrm{SNR}}

where :math:`D_{\mathrm{px}}` is the predicted disc diameter in pixels,
:math:`N_{\mathrm{lit}}` is the number of lit pixels inside the predicted bounding box, and
SNR is the per-pixel signal-to-noise ratio.

The bias is what the photon term cannot see. The measured brightness-weighted centroid is
compared with the rendered lit centroid, and the rendering's photometric law, albedo, and
shape are not the real body's, nor is the body exactly where its ephemeris puts it. With
:math:`R = D_{\mathrm{px}} / 2`, the capped radius :math:`R_c = \min(R,
\mathtt{bias\_radius\_cap\_px})`, the blob's
:attr:`~spindoctor.feature.flags.BodyBlobFlags.phase_irregularity_factor` :math:`k` (the
centroid bias of a shape irregular beyond its ellipsoid, which grows as less of the body
is lit; see the phase-and-irregularity coupling in
:doc:`dev_guide_navigation_models_body`), the blob's
:attr:`~spindoctor.feature.flags.BodyBlobFlags.ephemeris_sigma_px`
:math:`\sigma_{\mathrm{eph}}`, and the photocenter offset of a Lambertian sphere at phase
angle :math:`\alpha`,

.. math::

    g(\alpha) = \frac{3\pi}{16}\,
    \frac{\sin\alpha\,(1 + \cos\alpha)}{\sin\alpha + (\pi - \alpha)\cos\alpha}

(0.20 at 30 degrees, 0.59 at 90, 0.80 at 135, and 0.88 in the limit of a vanishing
crescent), each blob's 2x2 bias covariance is

.. math::

    \sigma_{\mathrm{iso}}^{2} =
        (f_{\mathrm{size}} R)^{2} + (k R_c)^{2} + \sigma_{\mathrm{eph}}^{2},

    B = \bigl(\sigma_{\mathrm{iso}}^{2} + (a\, g R_c)^{2}\bigr)\, \hat{s}\hat{s}^{T}
      + \bigl(\sigma_{\mathrm{iso}}^{2} + (f_{\mathrm{across}}\, g R_c)^{2}\bigr)\,
        \hat{n}\hat{n}^{T}

and its whole covariance is :math:`C = \sigma_{\mathrm{photon}}^{2} I + B`. Here
:math:`f_{\mathrm{size}}` is ``model_error_size_frac``, :math:`\hat{s}` is the unit image
direction toward the Sun the feature carries (``sub_solar_dir_vu``), :math:`\hat{n}` is
the direction across it, :math:`f_{\mathrm{across}}` is ``across_sun_frac``, and :math:`a`
is the blob's along-Sun fraction. A regular body (factor :math:`k` at most
``regular_irregularity_max``) takes ``along_sun_frac_regular``; an irregular one (at least
``irregular_irregularity_min``) takes ``along_sun_frac_irregular``; between the two the
fraction is interpolated linearly. A blob whose direction toward the Sun is unknown (a
vector shorter than ``_MIN_SUN_DIRECTION_NORM`` in ``body_blob_covariance.py``, 0.5, which
the body model reports at low phase) splits the two phase terms evenly over both axes.

The ephemeris term is the body's SPK position uncertainty,
:attr:`~spindoctor.nav_model.body_shape.BodyShape.spice_orbital_residual_km`, divided by
the km per pixel at its limb; the body model puts it on the feature. An ephemeris error
moves the whole blob, so it enters both axes. For Prometheus and Pandora it is 5 km: their
blob offsets against star-confirmed pointing on Cassini F ring bundle frames are nearly
constant within one observation and differ between observations (a median 9 km per
observation), and a maximum-likelihood fit of the blob covariance gives 4.8 to 4.9 km per
axis. A body without its own value takes its shape class's (2 km by default; see
:doc:`dev_guide_navigation_models_body`).

The predicted center is already the rendered body's lit-weighted centroid, so the phase
shift itself is modeled; the phase term sizes what the model leaves uncertain, and that
error lands along the direction toward the Sun. The measured centroids of the regular icy
moons sit sunward of the rendered ones (a median 1.59 px for Tethys, Dione, and Rhea seen
by the Cassini WAC at 110 to 150 degrees phase, a few pixels across, against
star-corrected pointing), so a regular body takes the larger along-Sun fraction. An
irregular moon's bias is mostly in its irregularity term, so it takes the smaller one. The
phase and irregularity terms stop growing at the radius cap: over 29 library blobs 50 to
870 px across, against truth independent of the blob, the error is a median 4.5 px, far
less than a bias proportional to the full radius would put it. The size term is not
capped.

Joint translation fit
---------------------

Part of the bias is common to every blob in a frame: the bodies share the Sun direction,
the rendering's photometric law, and an ephemeris solution, so their biases do not average
away. The shared part is isotropic, :math:`K = \rho\, m\, I`, where :math:`\rho` is
``bias_correlation`` and :math:`m` is the smallest eigenvalue of any blob's bias
covariance :math:`B_i`, so every blob's own part :math:`C_i - K` stays
positive-semidefinite. The :math:`2N \times 2N` covariance :math:`\mathcal{C}` of the
stacked offsets has the blocks :math:`C_i` on its diagonal and :math:`K` off it. With
:math:`H` the stack of :math:`N` 2x2 identities and :math:`o` the stacked
observed-minus-predicted offsets :math:`r_i`, the blobs combine by generalized least
squares over that full covariance:

.. math::

    (\Delta v, \Delta u)^{*} = \bigl(H^{T} \mathcal{C}^{-1} H\bigr)^{-1} H^{T}
        \mathcal{C}^{-1} o,
    \qquad C_{\mathrm{joint}} = \bigl(H^{T} \mathcal{C}^{-1} H\bigr)^{-1}

:math:`C_{\mathrm{joint}}` is the blobs' own parts combined by their inverses, plus
:math:`K`: a common error moves every blob alike, so it neither enters the weights nor
averages away, and the joint offset always lies between the blobs' own offsets. A single
blob reports its own covariance.

Joint covariance
----------------

Because the combination uses the full covariance, :math:`C_{\mathrm{joint}}` is the
covariance of the answer actually formed. With two or more blobs the residual chi-square
of the stacked offsets over :math:`2N - 2` degrees of freedom tests whether the blobs
agree as well as the covariance says; when they agree worse, :math:`C_{\mathrm{joint}}` is
multiplied by :math:`\chi^2 / (2N - 2)` (it is never scaled down).
``model_error_floor_px`` is then added in quadrature to both axes as the size-independent
floor.

The bias terms were fitted by maximum likelihood on the joint blob errors of one half of
906 Cassini F ring bundle and library frames and checked on the other half. The truth is a
star technique's answer on 497 of them, else the bundle pointing (the WAC bundle shifted
by its measured (0.43, 0.69) px offset), else the library's operator truth; lock failures
and at-edge results are excluded. On the held-out half the median Mahalanobis distance of
the reported answer and the share beyond 2 are 0.88 and 0% for NAC two-blob answers, 0.62
and 5% for single NAC blobs under 10 px, 0.75 and 14% for single NAC blobs 10 to 30 px
across, and 1.04 and 5% for WAC blobs; a correctly sized two-axis Gaussian gives 1.18 and
13.5%. Regular moons more than 100 px across at more than 90 degrees phase can err by 15
to 75 px along the direction toward the Sun, beyond their reported sigma; on those frames
a limb or disc fit of the same body supersedes the blob.

Dropping a blob locked on something larger
------------------------------------------

The coarse stage searches the whole window for a body of the predicted size, and on a ring
or crowded frame it can land on a ring arc or a larger body's limb instead. Such a lock
lights its whole box. The centroid step therefore also measures the lit region's gyration
radius (the signal-weighted RMS distance of the lit pixels from their centroid) and the
fraction of the lit signal on the box's outermost rows and columns. A blob whose gyration
radius exceeds ``max_gyration_radius_frac`` of the predicted radius **and** whose edge
fraction exceeds ``max_edge_signal_frac`` is dropped before the joint fit; a uniform disc
spreads 0.71 of its radius. Either test alone would also drop a real body: one that fills
an unpadded box, or a limb-bright rim. When every blob is dropped the result is spurious.

Over 1,242 Cassini F ring bundle and library blobs, the 954 within 5 px of truth reach a
gyration ratio of at most 0.84 and never exceed both thresholds; 157 of the 219 blobs more
than a diameter (and at least 10 px) from truth exceed both. Of the 62 that do not, 49 are
crescents past 110 degrees phase and 11 are between 94 and 108 degrees, whose lit arcs
spread like a correct crescent's; they are kept.

Restrictions and assumptions
----------------------------

- Per-blob centroids assume the (coarse-re-centered) bounding box truly contains the body's
  flux. When a cosmic-ray hit, an in-band stellar source, or a neighboring body's halo lands
  inside the box, the moment skews and the technique reports a wrong centroid. The upstream
  ``BODY_BLOB`` emission gates filter pathological cases (see
  :doc:`dev_guide_navigation_models_body`).
- The coarse lit-shape acquisition extends the capture range from the bounding box to the full
  search window at any phase: a disc template at or below half phase, a synthesized crescent
  above it. The only residual gap is a body past half phase whose illumination geometry was
  not populated (no ``sub_solar_dir_vu``), where the crescent cannot be oriented; such a body
  is then recovered only via the bounding-box centroid (small offsets) or an installed prior.
- A vanishing total flux (an entirely-in-shadow body whose predicted bounding box happens to
  cover the right part of the FOV) collapses the moment; the technique drops such blobs
  before the joint fit and reports a no-signal failure when every blob is dropped.
- **Undetected bodies are gated out; small detected ones are not.** The ``BODY_BLOB``
  feature's reliability is driven by a *measured* detection SNR: the model computes the
  flux-weighted effective count ``N`` of its *observable* lit pixels (the Kish effective
  sample size of the rendered brightnesses over on-sensor, unsaturated pixels -- a Lambert
  crescent's near-terminator tail and any off-sensor overhang do not inflate it), takes the
  median of the ``N`` brightest valid pixels in the search window
  (the predicted bbox expanded by the extfov margins, since the pointing error is unknown),
  and subtracts the level pure noise's top-``N`` order statistics would produce. The SNR
  sigmoid is centered at the technique's own 3-sigma lit-pixel threshold. A window with no
  body-scale signal above the noise floor therefore sits decisively below the keep
  threshold at any size, while a bright body only a little above the 5 px emission floor
  is admitted. The extent term applies a mild near-floor discount rather than a second
  size threshold: the reliability crosses the 0.20 keep threshold at detection SNR
  3.1-3.3 across the emitted size range. Whether the centroid is then *precise* is the
  per-blob covariance's and the confidence formula's job, not the reliability threshold's.
- The technique carries no rotation evidence — a brightness-weighted centroid is rotation-
  invariant about itself. When the per-instrument
  :attr:`~spindoctor.nav_orchestrator.nav_context.NavContext.fit_camera_rotation` is true, the
  technique returns the rank-deficient 3x3 covariance from
  :func:`~spindoctor.nav_technique.nav_technique.embed_rotation_unobservable` and reports
  :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.rotation_rad` as zero with
  the rotation-unobservable sentinel sigma.

Sources of uncertainty
----------------------

The reported covariance is the generalized-least-squares covariance over the full
covariance of the blobs described above (their own parts and the part they share), scaled
by the residual chi-square when the blobs disagree, plus the absolute floor. The centroid
bias of a body whose true shape or orientation differs from the rendered ellipsoid, and of
a phase shift the rendered photometry gets wrong, is carried as uncertainty, sized by the
:attr:`~spindoctor.feature.flags.BodyBlobFlags.phase_irregularity_factor` and the phase
terms, and so is the body's ephemeris error, sized by
:attr:`~spindoctor.feature.flags.BodyBlobFlags.ephemeris_sigma_px`; the centroid itself is
not corrected for either. When the converged offset sits within
:math:`\mathtt{at\_edge\_tolerance\_px}` of any axis bound, the result is flagged
:attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.at_edge` and the
hard-zero gate forces confidence to zero.

Configuration
=============

All numeric tunables for this technique live in ``techniques.BodyBlobNav.tuning`` in
``src/spindoctor/config_files/config_510_techniques.yaml``.

- ``at_edge_tolerance_px`` — float, default ``1.0`` px. A converged offset whose absolute
  distance from any search-window axis bound falls within this tolerance is flagged
  :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.at_edge`.
- ``model_error_size_frac`` — float, default ``0.05`` (fraction of the body radius). The
  isotropic bias every blob carries (see "Per-blob covariance"). ``0.0`` disables it.
- ``model_error_floor_px`` — float, default ``0.1`` px. Absolute size-independent floor added
  in quadrature to the joint covariance. ``0.0`` disables it.
- ``along_sun_frac_regular`` — float, default ``0.75``. Phase bias along the direction
  toward the Sun, as a fraction of the Lambertian photocenter offset, for a regular body.
- ``along_sun_frac_irregular`` — float, default ``0.15``. The same fraction for an
  irregular body.
- ``across_sun_frac`` — float, default ``0.2``. Phase bias across the direction toward the
  Sun, as a fraction of the photocenter offset, for every blob.
- ``bias_correlation`` — float, default ``0.85``. Fraction of the smallest bias variance
  among a frame's blobs that every blob shares (see "Joint translation fit"). Must lie in
  ``[0, 1)``.
- ``regular_irregularity_max`` — float, default ``0.02``. Phase-irregularity factor at and
  below which a body counts as regular.
- ``irregular_irregularity_min`` — float, default ``0.05``. Factor at and above which a
  body counts as irregular; between the two bounds the along-Sun fraction is interpolated.
- ``bias_radius_cap_px`` — float, default ``10.0`` px. Radius beyond which the phase and
  irregularity terms stop growing.
- ``max_gyration_radius_frac`` — float, default ``0.88``. Gyration radius, as a fraction
  of the predicted radius, past which a blob may be a lock on something larger (see
  "Dropping a blob locked on something larger").
- ``max_edge_signal_frac`` — float, default ``0.2``. Fraction of the lit signal on the box
  edge past which the same blob is dropped; both must be exceeded.

The bias terms must be finite and at least zero, ``irregular_irregularity_min`` must
exceed ``regular_irregularity_max``, and ``bias_radius_cap_px``,
``max_gyration_radius_frac``, and ``max_edge_signal_frac`` must be positive; construction
raises otherwise.

The remaining numeric thresholds (the per-blob CRLB scaling constants, the noise-floor
detection threshold) are derived from the per-image
:attr:`~spindoctor.nav_orchestrator.nav_context.NavContext.image_noise_sigma` and the per-blob
geometry; no YAML knob is exposed.

Per-instrument overrides
------------------------

The ``at_edge_tolerance_px`` knob is global; per-instrument YAML files in
``src/spindoctor/config_files/config_4N0_inst_*.yaml`` do not override it. The
search-window margin used by the at-edge test comes from the per-instrument
:class:`~spindoctor.nav_orchestrator.instrument_config.InstrumentSettings`.

Confidence formula
------------------

The technique reports a calibrated confidence in :math:`[0, 1]` produced by the shared
sigmoid combination; see :doc:`dev_guide_techniques_confidence` for the per-term arithmetic.
The formula spec is ``techniques.BodyBlobNav`` in the same YAML file and consumes attributes
off :class:`~spindoctor.nav_technique.diagnostics.BodyBlobDiagnostics` plus
:attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.at_edge`.

- :attr:`~spindoctor.nav_technique.diagnostics.BodyBlobDiagnostics.body_snr_inside_predicted_bbox`
  — alpha = 0.895, offset = 0.0, divisor = 600.0, cap at 1.0. Per-image SNR inside the
  predicted bounding box. Brightness-weighted centroid uncertainty shrinks with SNR; the
  calibration campaign's raw p5/p50/p95 is 18/61/517 (a heavy tail).
- :attr:`~spindoctor.nav_technique.diagnostics.BodyBlobDiagnostics.body_extent_px` —
  alpha = -1.063, offset = 8.0, divisor = 130.0, cap at 1.0. Predicted body's longer-axis
  extent in pixels. The alpha is *negative*, reversing the design's more-signal prior: the
  lit-weighted-centroid model error grows with apparent size, so the absolute probability
  of recovering within 1 px falls as the blob gets bigger. Small blobs remain the
  technique's tightest regime in absolute pixels. Campaign raw p5/p50/p95 is 8/25/69.
- :attr:`~spindoctor.nav_technique.diagnostics.BodyBlobDiagnostics.blob_count` — alpha = 0.4,
  offset = 0.0, divisor = 3.0, cap at 1.0. Number of ``BODY_BLOB`` features fused.
  Multi-body geometry over-determines the joint translation up to a 3-blob saturation.
  Not identifiable in the single-body sim campaign; retained at the design prior.
- :attr:`~spindoctor.nav_technique.diagnostics.BodyBlobDiagnostics.max_phase_irregularity_factor`
  — alpha = -0.545, offset = 0.0, divisor = 0.35, cap at 1.0. Maximum phase-and-irregularity
  factor across the consumed blobs (see :doc:`dev_guide_navigation_models_body` for the
  formula); the divisor spans the campaign's raw p5/p50/p95 of 0.003/0.019/0.369. With
  surface texture and relief in the rendered cohort the irregularity penalty carries
  substantial weight: an irregular body's unpredictable shadowing is model error the
  ellipsoidal render cannot correct.

Hard-zero gate: :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.at_edge`
firing forces confidence to zero before the sigmoid evaluates. The constant baseline is
:math:`\alpha_{0} = 1.465`. A post-sigmoid ``hard_cap`` of ``0.4`` clamps the result: a
brightness-weighted centroid cannot drive the ensemble past 0.4 confidence even when every
term saturates.

Implementation
==============

Source files:

- ``src/spindoctor/nav_technique/nav_technique_body_blob.py`` —
  :class:`~spindoctor.nav_technique.nav_technique_body_blob.BodyBlobNav`, the per-blob residual
  collector, and the joint-translation helper.
- ``src/spindoctor/nav_technique/confidence.py`` — shared sigmoid-combination evaluator;
  documented at :doc:`dev_guide_techniques_confidence`.
- ``src/spindoctor/nav_technique/diagnostics.py`` —
  :class:`~spindoctor.nav_technique.diagnostics.BodyBlobDiagnostics`; documented at
  :doc:`dev_guide_techniques_diagnostics`.

Public class :class:`~spindoctor.nav_technique.nav_technique_body_blob.BodyBlobNav`, base
:class:`~spindoctor.nav_technique.nav_technique.NavTechnique`. Self-registers via
``__init_subclass__`` so ``NavTechnique._registry`` discovers it.

Class attributes:

- :attr:`~spindoctor.nav_technique.nav_technique_body_blob.BodyBlobNav.name` — ``'BodyBlobNav'``.
- :attr:`~spindoctor.nav_technique.nav_technique_body_blob.BodyBlobNav.accepts_feature_types` —
  ``frozenset({BODY_BLOB})``.
- :attr:`~spindoctor.nav_technique.nav_technique_body_blob.BodyBlobNav.requires_prior` — ``False``.
  Runs in pass 1 of the orchestrator's two-pass pipeline.
- :attr:`~spindoctor.nav_technique.nav_technique_body_blob.BodyBlobNav.confidence_attributes` —
  ``{'at_edge', 'body_snr_inside_predicted_bbox', 'body_extent_px', 'blob_count',
  'residual_px', 'max_phase_angle_deg', 'max_phase_irregularity_factor'}``.

Public methods (autodocumented at :doc:`/api_reference/api_nav_technique`):
:meth:`~spindoctor.nav_technique.nav_technique_body_blob.BodyBlobNav.is_feasible` and
:meth:`~spindoctor.nav_technique.nav_technique_body_blob.BodyBlobNav.navigate`.

Diagnostics
-----------

:class:`~spindoctor.nav_technique.diagnostics.BodyBlobDiagnostics`:

- :attr:`~spindoctor.nav_technique.diagnostics.BodyBlobDiagnostics.body_snr_inside_predicted_bbox`
  — per-image SNR inside the predicted bounding box. Consumed by the confidence formula.
- :attr:`~spindoctor.nav_technique.diagnostics.BodyBlobDiagnostics.body_extent_px` — predicted
  body's longer-axis extent in pixels. Consumed by the confidence formula.
- :attr:`~spindoctor.nav_technique.diagnostics.BodyBlobDiagnostics.blob_count` — number of
  ``BODY_BLOB`` features fused. Consumed by the confidence formula.
- :attr:`~spindoctor.nav_technique.diagnostics.BodyBlobDiagnostics.residual_px` — joint-fit RMS
  residual after solving the precision-weighted mean. Diagnostic only; not in the formula.
- :attr:`~spindoctor.nav_technique.diagnostics.BodyBlobDiagnostics.max_phase_angle_deg` — maximum
  raw phase angle across consumed blobs. Diagnostic only; the formula consumes
  ``max_phase_irregularity_factor`` instead because raw phase understates the centroid
  uncertainty for an irregular body.
- :attr:`~spindoctor.nav_technique.diagnostics.BodyBlobDiagnostics.max_phase_irregularity_factor` —
  maximum
  :math:`(\sigma_{\mathrm{ellipsoid}} / R_{\mathrm{body}}) \cdot (1 + 2 \sin^{2}(\phi/2))`
  across consumed blobs, where :math:`\sigma_{\mathrm{ellipsoid}}` is the body's ellipsoid
  RMS shape residual in km and :math:`R_{\mathrm{body}}` its radius in km. Consumed by the
  confidence formula.

Call path
---------

Call path traced through
:meth:`~spindoctor.nav_technique.nav_technique_body_blob.BodyBlobNav.navigate`:

1. Open a logged section. Filter the offered features down to ``BODY_BLOB`` entries with a
   non-zero predicted diameter via the private eligibility helper.
2. Read the search-window margin off the observation via
   :func:`~spindoctor.nav_technique.nav_technique.search_window_for_obs`, the extfov image off
   :attr:`~spindoctor.nav_orchestrator.nav_context.NavContext.image_ext`, and the per-image noise
   sigma off :attr:`~spindoctor.nav_orchestrator.nav_context.NavContext.image_noise_sigma`
   (clamped at a tiny floor so the noise-floor test stays well-defined on near-blank
   inputs).
3. For each eligible blob, the private residual collector slices the predicted bounding
   box from the extfov image, evaluates the brightness-weighted-moment centroid, drops the
   blob when total flux falls below the noise floor or when its lit region is a lock on
   something larger (see "Dropping a blob locked on something larger"), computes the
   per-blob residual against the predicted lit-weighted centroid, and keeps the blob's
   photon variance.

   - **No surviving blobs.**  The technique returns a spurious zero-confidence result via
     the private fail helper, with the corresponding
     :class:`~spindoctor.nav_technique.diagnostics.BodyBlobDiagnostics`.

4. The private joint-fit helper builds each blob's 2x2 bias covariance (the size,
   irregularity, ephemeris, and along- and across-Sun phase terms), adds the shared part
   to form the full covariance of all blobs, combines the blobs by generalized least
   squares over it, scales the joint covariance by the residual chi-square when the blobs
   disagree, and adds the absolute floor; the covariance helpers live in
   ``body_blob_covariance.py``.
5. Apply the at-edge test against the search-window axis bounds.
6. Result-shape branches on
   :attr:`~spindoctor.nav_orchestrator.nav_context.NavContext.fit_camera_rotation`:

   - **No rotation fit.**
     :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.covariance_px2` is the
     (2, 2) translation block.
     :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.rotation_rad` and
     :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.sigma_rotation_rad` are
     ``None``.
   - **Rotation fit.**  The technique embeds the (2, 2) translation block in a (3, 3)
     covariance via
     :func:`~spindoctor.nav_technique.nav_technique.embed_rotation_unobservable`, sets
     :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.rotation_rad` to ``0.0``,
     and reports
     :func:`~spindoctor.nav_technique.nav_technique.rotation_unobservable_sigma_rad` as the
     :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.sigma_rotation_rad`.
     A brightness-weighted centroid is rotation-invariant about itself, so the technique
     carries no rotation evidence.

7. Build a :class:`~spindoctor.nav_technique.diagnostics.BodyBlobDiagnostics` from the per-blob
   residuals (max-SNR, max-extent, blob-count, RMS residual, max raw phase, max
   phase-irregularity factor), evaluate the confidence spec via
   :func:`~spindoctor.nav_technique.confidence.evaluate_sigmoid_combination`, log the per-term
   breakdown via :func:`~spindoctor.nav_technique.nav_technique.log_confidence_breakdown`, and
   assemble the :class:`~spindoctor.nav_technique.technique_result.NavTechniqueResult`.

The :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.feature_ids` field
preserves every consumed
:attr:`~spindoctor.feature.feature.NavFeature.feature_id` so the orchestrator's curator can
attribute each contribution at audit time.

Examples
========

``below_resolution_body`` (Cassini ISS NAC, image ``N1777325846_1``)
    Mimas approximately 20 px in diameter in the lower left, at phase angle 72 degrees. The
    body model emits a single ``BODY_BLOB`` feature (the per-pixel ellipsoid uncertainty exceeds
    :data:`~spindoctor.nav_model.nav_model_body.LIMB_ARC_MAX_UNCERTAINTY_PX` so ``LIMB_ARC`` is
    suppressed in favor of the centroid path).
    :class:`~spindoctor.nav_technique.nav_technique_body_blob.BodyBlobNav` consumes the blob and
    converges within ~1 px of the operator-verified offset
    :math:`(\Delta v, \Delta u) = (6.08, -1.53)` px. The post-sigmoid hard cap of 0.4
    keeps the technique from outranking a hypothetical limb fit on a similar but
    well-resolved scene.

``multi_body`` (Cassini ISS NAC, image ``N1487595731_1``)
    Dione and Rhea both visible at phase angle approximately 90 degrees. When the body
    model emits ``BODY_BLOB`` features for both bodies (or one body's limb fails the uncertainty
    gate), :class:`~spindoctor.nav_technique.nav_technique_body_blob.BodyBlobNav` fuses the two
    centroids into a joint translation. The 3-blob saturation in the confidence formula is
    not reached, but the multi-body
    :attr:`~spindoctor.nav_technique.diagnostics.BodyBlobDiagnostics.blob_count` term still
    contributes a positive offset. Operator-verified offset is
    :math:`(\Delta v, \Delta u) = (7.03, -18.42)` px.
