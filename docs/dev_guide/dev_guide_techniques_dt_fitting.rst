==============================================
DT Fitting (Shared Polyline-vs-Image Fitter)
==============================================

Overview
========

DT fitting is the shared algorithmic core that aligns a polyline of model vertices against an
image's edge-distance-transform. It is reused by every distance-transform-based technique in
the pipeline:
:class:`~spindoctor.nav_technique.nav_technique_body_limb.BodyLimbNav`,
:class:`~spindoctor.nav_technique.nav_technique_body_terminator.BodyTerminatorNav`, and
:class:`~spindoctor.nav_technique.nav_technique_ring_edge.RingEdgeNav`. The module exposes pure
numerical helpers — coarse integer cross-correlation, polarity filtering, Tukey biweight
weights, Levenberg-Marquardt sub-pixel refinement, and an information-matrix-to-covariance
helper — and lets each technique assemble its own vertex / normal / sigma arrays before calling
in.

Because this page documents algorithmic infrastructure rather than a directly-configurable
component, the Configuration section is short: the only knobs are module-level default
constants exposed in the module's public surface. Every consuming technique reads
technique-specific tunables from its own YAML stanza and may override these defaults at call
time.

Theory
======

The shared fitter solves a single problem: given a collection of model vertices, each with an
outward normal and a prior position sigma along that normal, find the rigid transformation that
maps the vertices closest to the image's edge pixels. "Closest" is measured against a
truncated, signed-magnitude distance transform of the thresholded image gradient, so the
problem reduces to a weighted nonlinear least-squares fit whose residuals are bilinearly
sampled DT values at the transformed vertex positions.

Stage 1 — coarse integer cross-correlation
------------------------------------------

The first stage rasterizes the polyline into a binary mask aligned with the image, thresholds
the truncated DT into an edge mask, and evaluates the per-vertex match fraction

.. math::

    f(\Delta v, \Delta u) =
        \frac{\sum_{v, u}
            \mathrm{polyline}[v, u] \, \mathrm{edge}[v + \Delta v, u + \Delta u]}
             {N_{\mathrm{in\,bounds}}(\Delta v, \Delta u)}

over every integer offset in :math:`[-m_{v}, m_{v}] \times [-m_{u}, m_{u}]`, where
:math:`(m_{v}, m_{u})` is the per-instrument SPICE pointing-error margin and
:math:`N_{\mathrm{in\,bounds}}` is the number of polyline vertices that remain inside the
image after the shift. Both inputs are binary; dividing the raw overlap count by the
in-bounds vertex count removes the bias of a raw overlap count toward shifts that simply
keep more vertices in bounds. This per-vertex match fraction is deliberately *not* the
binary normalized cross-correlation (a binary NCC normalizes by
:math:`\sqrt{N_{\mathrm{in\,bounds}}}`, not :math:`N_{\mathrm{in\,bounds}}`, so its argmax
can differ): the plain fraction fully cancels the in-bounds-count advantage, at the cost of
over-rewarding shifts with very few surviving vertices. That failure mode is closed by a
minimum-support guard — a shift whose in-bounds vertex fraction falls below
:data:`~spindoctor.nav_technique.dt_fitting.DEFAULT_COARSE_MIN_SUPPORT_FRACTION` (0.5) is
ineligible, so a perfect score on a handful of edge-pixel stragglers cannot beat a dense
well-supported match.

Ties are broken by Manhattan distance from the origin (and lexicographic order among offsets at
equal distance) so that on perfectly-flat inputs the nearest-to-origin shift wins
deterministically. An empty polyline mask short-circuits to the zero offset.

The brute-force scan is intentionally not FFT-based. The polyline support is sparse on the
image grid (a typical limb is ~200 pixels on a 1024² image), so iterating over the polyline
indices for each shift is faster than the full-image FFT for the search-window sizes the
pipeline actually uses.

Stage 1b — polarity-weighted coarse acquisition
-----------------------------------------------

The binary match fraction above is *polarity-blind*: a model vertex counts as a match whenever
it lands on any detected edge pixel, regardless of that edge's orientation. In a cluttered
scene — a ring ansa behind a limb, the terminator on a lit disc, a second moon in the field —
a dense population of unrelated edges can out-score the short true feature arc, seeding the LM
in the wrong basin where it converges to a confident-but-wrong offset that the downstream
spurious gates do not always catch.

:func:`~spindoctor.nav_technique.dt_fitting.coarse_polarity_search_scored` adds the
discriminator the mask-overlap count discards: the *direction* of the image gradient under
each vertex. A vertex still has to land on a detected edge pixel (the same density-neutral
binary signal), but its contribution is then weighted by :math:`\max(0, \cos\theta_{i})`,
where :math:`\theta_{i}` is the angle between the model's per-vertex outward normal and the
image gradient vector sampled at the shifted vertex:

.. math::

    f_{\mathrm{pol}}(\Delta v, \Delta u) =
        \frac{\sum_{i \in \mathrm{bounds}}
            \mathrm{edge}[x_{i} + \Delta x] \,
            \max\!\left(0, \hat{n}_{i} \cdot \hat{g}[x_{i} + \Delta x]\right)}
             {N_{\mathrm{in\,bounds}}(\Delta v, \Delta u)}

Both vectors are unit-normalised before the dot product, so only gradient *direction* enters:
a faint-but-correctly-oriented true edge is never out-weighted by a bright high-contrast
clutter edge. A vertex whose local edge runs the wrong way (gradient anti-parallel to the
model normal — a terminator, a far-side ring edge, a crater rim) contributes nothing; an
unrelated clutter edge at a random orientation contributes on average only
:math:`1/\pi \approx 0.32` of its overlap; the true feature, whose edges run along the
predicted normals by construction, keeps nearly full weight. Where every edge gradient is
aligned with its normal (:math:`\cos\theta = 1`), each match reverts to a hard 0/1 and the
score becomes the plain per-vertex edge-overlap fraction, so the *seed* is unchanged on clean,
uncluttered frames — the search only diverges from the overlap argmax where polarity actually
discriminates. (The absolute score still differs slightly from Stage 1's even in the aligned
case: Stage 1b divides by the raw in-bounds vertex count, Stage 1 by the count of distinct
rasterized pixels; only the argmax matters for seeding, and it coincides.) The minimum-support
guard and the Manhattan tie-break are identical to Stage 1.

This form is used exactly where its polarity signal is trustworthy: the same techniques that
run the LM with ``use_polarity=True`` (``BodyLimbNav`` and ``BodyTerminatorNav``), whose model
normals carry a predictable expected gradient direction. ``RingEdgeNav`` runs
``use_polarity=False`` — a ring edge may be bright-inside or dark-inside depending on the
lighting and gap-versus-ringlet context the catalog does not encode — so it keeps the
polarity-blind Stage 1 search; making its coarse seed robust is separate work.

Stage 2 — sub-pixel Levenberg-Marquardt refinement
--------------------------------------------------

Starting from the integer seed, the refiner minimizes

.. math::

    C(\Delta v, \Delta u, \theta) = \sum_{i} w_{i}\,
        \mathrm{DT}\!\left[ R(\theta)\,(x_{i} - x_{p}) + x_{p} + (\Delta v, \Delta u) \right]^{2}

where :math:`x_{i}` are the input vertices, :math:`x_{p}` is the rotation pivot (the centroid
of the input vertices, or a caller-supplied pivot), :math:`R(\theta)` is the in-plane
rotation, and :math:`\mathrm{DT}` is the bilinearly sampled image-edge distance transform.
The per-vertex weight is the product of the prior precision :math:`1 / \sigma_{i}^{2}` and a
Tukey biweight evaluated at the scaled residual :math:`\mathrm{DT}_{i} / \sigma_{i}`. When
the caller disables rotation, the parameter vector collapses to :math:`(\Delta v, \Delta u)`.

Each LM iteration:

1. Samples the DT and finite-difference Jacobian at the current pose. Translation derivatives
   use a 0.25-pixel central-difference step; the rotation derivative uses a small
   :math:`10^{-3}`-radian step.
2. Computes the Tukey biweight weights from the sigma-scaled residuals.
3. Solves the LM-damped normal equations :math:`(J^{\top} W J + \lambda \, \mathrm{diag}(J^{\top} W J))\, \delta = -J^{\top} W r`
   via :func:`numpy.linalg.solve`, falling back to the symmetric pseudoinverse on a singular
   step.
4. Before the trial cost is even evaluated, rejects any trial pose whose translation has
   walked more than ``trust_region_px`` from the integer coarse seed (the damping is doubled
   and the iteration retried); this denies the joint LM + Tukey IRLS the runway to drift the
   polyline onto unrelated DT minima. Otherwise accepts the trial pose when the
   weighted-cost decreases (and shrinks the LM damping by a
   factor of two), or rejects it (and grows the damping by a factor of two).
5. After an accepted step, recomputes residuals / weights / Jacobian *at* the accepted pose
   so the cached state used by the final-iteration covariance reflects the committed
   parameters. The convergence test uses the step-norm in pixel-equivalent units —
   translation magnitudes contribute directly; the rotation step is multiplied by the
   pivot-to-image-centre distance so the same threshold applies to translation-only and
   translation-plus-rotation fits.

The iteration terminates when the step-norm drops below the configured tolerance, when the
damping grows past :math:`10^{6}` (the trust region collapsed), or when the iteration cap is
reached.

Stage 3 — optional continuous gradient-ridge refinement
-------------------------------------------------------

``gradient_ridge_refine`` is a final, sub-pixel polish that fits directly against the
*continuous* gradient-magnitude field rather than the integer-quantized DT. For each vertex at
its current pose it bilinearly samples :math:`|\nabla|` along the outward normal across
:math:`t \in [-3, +3]` px and parabola-interpolates the sub-pixel signed distance :math:`t^{*}`
to the gradient peak; the residual is :math:`r_{i} = t^{*}_{i}`, driven to zero by
Gauss-Newton. Translating a vertex by :math:`\delta` changes the residual by
:math:`-(n_{i} \cdot \delta)`, so the translation Jacobian rows are :math:`[-n_{v}, -n_{u}]`;
the rotation column (when fit) is central-differenced. The same Tukey reweighting and weighted
normal equations as Stage 2 apply, and a displacement cap discards the result if it walks more
than ~1.5 px from the DT-LM pose. When the stage runs, the reported residuals / weights / RMS /
covariance are recomputed against the DT at the refined pose so the spurious gates stay on the
same footing as without it.

The per-technique ``gradient_ridge_refine`` tuning flag is **on (1) for
RingEdgeNav and off (0) for the limb and terminator fits**. It is correct for *symmetric*
edges only:

- The ring edge benefits on dense real ring scenes: the binary edge mask quantises detected
  edges to the integer pixel grid, so a large fraction of model vertices land exactly on
  edge pixels (a DT of zero and zero gradient) and the DT-LM step stalls at the integer
  coarse-NCC seed. The continuous pass refines against the un-thresholded gradient
  magnitude, recovering the sub-pixel offset the quantised DT discards. A ring edge is a
  symmetric transition whose gradient peak coincides with the geometric edge, so refining
  onto the peak is unbiased. (On clean sim scenes the DT-LM already reaches its ~0.016 px
  floor, so the refine is a no-op there.)
- The body limb is *worse* with it on. A limb is a one-sided transition, so its gradient peak
  sits ~0.1 px inside the geometric silhouette the model predicts; the DT-LM's quantization
  partially cancels that offset, and converging precisely onto the gradient peak removes the
  cancellation and sharpens the bias. The limb floor is therefore a model-vs-image edge offset,
  not a DT-construction artifact; the genuine remedy is a model-side prediction of the limb
  at the gradient-peak location. See :doc:`dev_guide_techniques_body_limb`.

Polarity filtering
------------------

When the caller enables polarity, the refiner classifies each vertex *once*, at the coarse
integer seed, before the LM iteration starts: it samples the per-pixel image gradient vector
at the seed-shifted vertex position, dot-products it with the model's outward normal, and
marks the vertex a *polarity match* when the dot product is strictly positive (and a
*polarity mismatch* otherwise). Mismatched vertices are excluded by zeroing their weight
directly — the per-iteration weight is the product
``inv_sigma_sq * tukey_w * polarity_mask`` — so their exclusion is independent of the
per-vertex sigma. (A near-infinite synthetic residual is recorded for them in the raw
residual array only to keep it numerically well-defined; with a zero weight it contributes
nothing to the cost or the normal equations.) This keeps a body-limb fit from latching onto
the body's interior crater rims
(whose gradient direction is opposite the limb's), and a ring-edge fit from latching onto a
neighboring ring's edge (whose gradient sign reverses).

Robustness via Tukey biweight
-----------------------------

The Holland-Welsch biweight is the redescender used by the IRLS step:

.. math::

    w_{i} = \begin{cases}
        \bigl(1 - (r_{i} / c)^{2}\bigr)^{2}, & |r_{i}| \le c \\
        0, & \text{otherwise}
    \end{cases}

with :math:`c = 4.685` (the default), corresponding to ~95 % asymptotic efficiency on Gaussian
residuals when the residuals are pre-scaled by the desired robust scale. The per-technique
caller divides each raw DT residual by the per-vertex prior sigma before the biweight is
applied, so the cutoff is interpreted in sigma units rather than raw pixels. Vertices whose
scaled residuals exceed the cutoff are dropped completely. Recomputing the weights after each
accepted LM step (iteratively-reweighted least squares) makes the Tukey-rejected vertices
truly drop out of the information matrix and keeps the converged covariance an honest
expression of the surviving inliers' precision.

.. _dt-uncertainty:

From cost surface to information matrix
---------------------------------------

The reported parameter covariance is the local quadratic approximation of the cost surface at
convergence. Writing the residual at iteration's end as :math:`r_{i}(\hat p) = \mathrm{DT}_{i}`
and the per-vertex weight as :math:`w_{i} = (1 / \sigma_{i}^{2}) \cdot w_{i}^{\mathrm{Tukey}}`,
the cost is

.. math::

    C(p) = \sum_{i} w_{i} \, r_{i}(p)^{2}.

At the converged estimate :math:`\hat p` the gradient

.. math::

    \nabla C(\hat p) = 2 \sum_{i} w_{i} \, r_{i}(\hat p) \, \nabla r_{i}(\hat p)

vanishes by construction, and the local Hessian reduces under the Gauss-Newton approximation
— dropping the second-order :math:`r_{i} \, \nabla^{2} r_{i}` term, which integrates to zero
against zero-mean residuals at convergence — to

.. math::

    H \approx 2 \sum_{i} w_{i} \, \nabla r_{i}(\hat p) \, \nabla r_{i}(\hat p)^{\top}
        = 2 \, J^{\top} W J,

with :math:`J` the residual Jacobian (rows are :math:`\nabla r_{i}(\hat p)^{\top}`) and
:math:`W = \mathrm{diag}(w_{i})`. Under the M-estimator interpretation the parameter
covariance is the inverse of the information matrix :math:`I = J^{\top} W J`; the factor of
two cancels in the standard derivation against the residual variance the Tukey weights have
already absorbed. The dropped :math:`r \, \nabla^{2} r` term is exactly the variance
contribution that becomes negligible at convergence under low-residual conditions.

Pseudoinverse to covariance
---------------------------

After convergence, the M-estimator information matrix :math:`I = J^{\top} W J` is mapped to a
parameter covariance via :func:`scipy.linalg.pinvh` (the Hermitian Moore-Penrose pseudoinverse).
The pseudoinverse handles rank-deficient inputs gracefully: when the model polyline is
geometrically rank-deficient (every edge in a flat-ring scene is parallel and the along-ring
axis is unobservable) :math:`I` is singular, and replacing the inverse with :math:`I^{+}` lets
the unobservable direction remain unobservable in the returned covariance: the corresponding
eigenvalue is zero, the marginal variance along that axis is also zero in :math:`I^{+}`, and
the per-axis sigma reported by the ensemble correctly flags the rank deficiency rather than
silently inverting floating-point noise.

A single project-wide cutoff
(:data:`~spindoctor.nav_technique.dt_fitting.DEFAULT_PINVH_RCOND`, ``1e-9``) keeps the threshold
consistent across every technique that consumes this helper and across the orchestrator's
ensemble combine. The value is calibrated so that a tighter cutoff would silently treat
near-rank-deficient matrices as full-rank, producing garbage inverse entries; a looser cutoff
would prematurely zero observable directions. ``1e-9`` is liberal enough to fold near-singular
directions into the null space and conservative enough to preserve genuine 2-D / 3-D
constraints typical of body-limb / ring-edge fits.

Three "no information" cases are guarded so the returned covariance does not falsely advertise
perfect certainty: an empty Jacobian, a zero-inlier population, and a zero-total-weight pose.
Each yields an all-infinity covariance that the orchestrator's ensemble combine treats as
fully unconstrained.

Combining with a prior in the ensemble
--------------------------------------

The per-technique covariance returned to the orchestrator is the M-estimator pseudoinverse
:math:`I^{+}`. The orchestrator's ensemble combines per-technique results in *information
form*: invert each covariance, sum, invert again. :func:`scipy.linalg.pinvh` is used at every
inversion and the same ``rtol`` propagates throughout, so a single rank-1 ring constraint plus
any other rank-2 result yields a fully-resolved 2-D answer; a single rank-1 ring constraint
alone yields a rank-1 final covariance with the unobservable axis honestly flagged. This is
why the per-technique result must report :math:`I^{+}` rather than a regularized dense
inverse: the ensemble's rank-deficiency-aware fusion depends on the unobservable directions
being explicit zeros in the per-technique information matrix, not finite values driven by an
arbitrary regulariser.

Restrictions and assumptions
----------------------------

- The refiner is purely local; it converges to the basin of attraction selected by the coarse
  integer seed. When the seed is wrong (a multi-body scene where the coarse peak picks
  a competing edge population rather than the technique's true feature arc), LM converges to a
  wrong minimum. The polarity-weighted coarse acquisition (Stage 1b) reduces this for the
  polarity techniques by down-weighting wrong-direction edges before the seed is chosen; a
  seed that still lands in a wrong basin is caught by the per-technique spurious gates.
  ``RingEdgeNav``, which cannot use polarity, relies on the seed and the gates alone.
- Bilinear DT sampling assumes the DT is differentiable almost everywhere. Pixel-grid
  pathologies (a perfectly flat plateau across the entire search window) collapse the
  Jacobian and the LM step has no preferred direction; the refiner returns the seed unchanged
  with the Tukey-zero-weight covariance sentinel.
- The polarity filter assumes the gradient vector image is the signed
  :math:`(g_{v}, g_{u})` Sobel-of-Gaussian readout produced by the orchestrator's
  :func:`~spindoctor.nav_orchestrator.image_derivatives.compute_image_gradient_vu`. Magnitude-only
  gradient images cannot drive the polarity test and the caller must opt out via
  ``use_polarity=False``.
- The convergence test uses a single step-norm tolerance for both translation-only and
  translation-plus-rotation fits. Callers that fit rotation must provide a positive
  pivot-to-image-centre distance so the rotation step can be converted to pixel-equivalent
  units; missing or zero distance raises :exc:`ValueError`.

Sources of uncertainty
----------------------

The covariance returned on the result reflects the cost-surface curvature at convergence under
the surviving Tukey weights. It does not capture systematic biases (a per-vertex sigma scaled
incorrectly by the model upstream propagates straight into the covariance), and it does not
capture the bias introduced by polarity-rejected vertices. When the caller disables
polarity filtering the covariance reverts to a pure prior-precision M-estimator on the full
polyline.

Configuration
=============

DT fitting carries no YAML configuration of its own. Every numeric default exposed in the
module's public API is a module-level constant; consuming techniques pass technique-specific
overrides at call time. The defaults below are exposed in the module's ``__all__`` so
tests, downstream tools, and consuming techniques can reference the canonical values without
re-reading the source.

- :data:`~spindoctor.nav_technique.dt_fitting.DEFAULT_TUKEY_C` — float, default ``4.685`` sigma.
  Holland-Welsch biweight cutoff. Sigma-scaled residuals beyond this magnitude are dropped
  completely; 4.685 corresponds to ~95 % asymptotic Gaussian efficiency.
- :data:`~spindoctor.nav_technique.dt_fitting.DEFAULT_LM_DAMPING` — float, default ``1.0e-3``
  (dimensionless). Initial Levenberg-Marquardt damping :math:`\lambda`. Multiplied by 0.5
  on accepted steps and by 2 on rejected steps; bracketed in :math:`[10^{-12}, 10^{6}]`.
- :data:`~spindoctor.nav_technique.dt_fitting.DEFAULT_LM_MAX_ITERATIONS` — int, default ``30``
  (count). Maximum LM iterations before the refiner terminates without convergence.
- :data:`~spindoctor.nav_technique.dt_fitting.DEFAULT_LM_STEP_TOLERANCE` — float, default
  ``1.0e-3`` px. Step-norm threshold below which the iteration terminates with
  ``converged=True``. Combines translation magnitude and rotation step times pivot distance
  into a single pixel-equivalent number.
- :data:`~spindoctor.nav_technique.dt_fitting.DEFAULT_PINVH_RCOND` — float, default ``1.0e-9``
  (dimensionless). Cutoff passed to :func:`scipy.linalg.pinvh` for the
  information-to-covariance map. Eigenvalues smaller than this fraction of the largest are
  treated as null.
- :data:`~spindoctor.nav_technique.dt_fitting.DEFAULT_COARSE_MIN_SUPPORT_FRACTION` — float,
  default ``0.5`` (dimensionless). Minimum in-bounds vertex fraction for a coarse-search
  candidate shift to be eligible; guards the per-vertex match fraction against
  over-rewarding shifts that clip most of the polyline off-frame. The zero shift keeps
  every vertex in bounds, so at least one candidate is always eligible.
- :data:`~spindoctor.nav_technique.dt_fitting.DEFAULT_RIDGE_HALF_WIDTH_PX` — float, default
  ``3.0`` px. Half-width of the gradient-ridge sub-pixel search window along each vertex
  normal; wide enough to cover the residual a clean DT-LM convergence leaves, narrow
  enough that the sampled profile contains a single edge ridge.
- :data:`~spindoctor.nav_technique.dt_fitting.DEFAULT_RIDGE_SAMPLE_STEP_PX` — float, default
  ``0.5`` px. Spacing of the gradient-ridge sample points along each normal; the peak is
  located by a three-point parabola fit around the discrete argmax.
- :data:`~spindoctor.nav_technique.dt_fitting.DEFAULT_RIDGE_MAX_ITERATIONS` — int, default
  ``10`` (count). Maximum gradient-ridge Gauss-Newton iterations; the stage starts from
  the DT-LM optimum so a handful suffices, and the cap is a safety net.
- :data:`~spindoctor.nav_technique.dt_fitting.DEFAULT_RIDGE_MAX_TOTAL_DISPLACEMENT_PX` —
  float, default ``1.5`` px. Cap on how far the gradient-ridge stage may move the DT-LM
  offset; beyond it the ridge result is discarded and the DT-LM offset kept (a defensive
  guard against a pathological ridge walk onto an unrelated gradient feature).

Implementation
==============

Source files:

- ``src/spindoctor/nav_technique/dt_fitting.py`` — the shared helpers and the
  :class:`~spindoctor.nav_technique.dt_fitting.LMRefineResult` dataclass.
- ``src/spindoctor/support/distance_transform.py`` —
  :func:`~spindoctor.support.distance_transform.sample_dt_bilinear`, the bilinear DT sampler the
  refiner calls per iteration.
- ``src/spindoctor/nav_orchestrator/image_derivatives.py`` — the image-side derivatives
  documented at :doc:`dev_guide_techniques`; the orchestrator computes them once per image
  and the DT fitter consumes the products attached to
  :class:`~spindoctor.nav_orchestrator.nav_context.NavContext`.

Public surface (autodocumented at :doc:`/api_reference/api_nav_technique`):

- :func:`~spindoctor.nav_technique.dt_fitting.coarse_ncc_search` — the integer-shift binary
  cross-correlation (offset-only convenience form).
- :func:`~spindoctor.nav_technique.dt_fitting.coarse_ncc_search_scored` — the same search
  returning a :class:`~spindoctor.nav_technique.dt_fitting.CoarseSearchResult`, whose
  ``score`` is the winning shift's per-vertex edge-pixel match fraction — the
  acquisition-quality signal the fit-quality gates consume. ``RingEdgeNav`` calls this form.
- :func:`~spindoctor.nav_technique.dt_fitting.coarse_polarity_search_scored` — the
  polarity-weighted variant (Stage 1b): the same integer search and
  :class:`~spindoctor.nav_technique.dt_fitting.CoarseSearchResult` output, but each match is
  weighted by the agreement between the model's per-vertex normal and the sampled image
  gradient direction, so a competing wrong-polarity edge population cannot mis-seed the fit.
  The polarity techniques (``BodyLimbNav``, ``BodyTerminatorNav``) call this form.
- :func:`~spindoctor.nav_technique.dt_fitting.polarity_filter` — per-vertex polarity acceptance from
  the gradient vector image.
- :func:`~spindoctor.nav_technique.dt_fitting.tukey_biweight_weights` — Holland-Welsch redescender
  evaluated at scaled residuals.
- :func:`~spindoctor.nav_technique.dt_fitting.lm_subpixel_refine` — translation (or translation +
  rotation) LM refinement with Tukey reweighting.
- :func:`~spindoctor.nav_technique.dt_fitting.information_matrix_to_covariance` — Hessian to
  covariance via the symmetric pseudoinverse.
- :func:`~spindoctor.nav_technique.dt_fitting.find_secondary_dt_minimum` — the basin
  second-opinion: scans the search window for a competing DT-cost minimum away from the
  converged offset, so a consuming technique can flag a non-unimodal fit as spurious
  (:class:`~spindoctor.nav_technique.nav_technique_body_terminator.BodyTerminatorNav`
  uses it; see :doc:`dev_guide_techniques_body_terminator`).
- :class:`~spindoctor.nav_technique.dt_fitting.LMRefineResult` — frozen result dataclass exposed by
  :func:`~spindoctor.nav_technique.dt_fitting.lm_subpixel_refine`.
- :class:`~spindoctor.nav_technique.dt_fitting.SecondaryBasin` — frozen result dataclass exposed
  by :func:`~spindoctor.nav_technique.dt_fitting.find_secondary_dt_minimum`.

Consuming techniques follow the same eight-step pattern (the body-limb fit is the canonical
example documented at :doc:`dev_guide_techniques_body_limb`):

1. Aggregate the per-feature vertex / normal / sigma arrays.
2. Threshold the truncated DT into the image edge mask.
3. Read the search-window margin via
   :func:`~spindoctor.nav_technique.nav_technique.search_window_for_obs`.
4. Call the coarse search to obtain the integer seed and its acquisition score:
   :func:`~spindoctor.nav_technique.dt_fitting.coarse_polarity_search_scored` for the polarity
   techniques (``BodyLimbNav``, ``BodyTerminatorNav``), or
   :func:`~spindoctor.nav_technique.dt_fitting.coarse_ncc_search_scored` for ``RingEdgeNav``.
5. Decide whether to fit camera rotation; when rotation is fit, set the rotation pivot to the
   vertex centroid and read the pivot-to-image-centre distance via
   :func:`~spindoctor.nav_technique.nav_technique.rotation_pivot_distance_px`.
6. Call :func:`~spindoctor.nav_technique.dt_fitting.lm_subpixel_refine` with the polyline, the
   per-vertex sigmas, the integer seed, and the rotation options.
7. Apply the per-technique at-edge and spurious tests against the converged
   :class:`~spindoctor.nav_technique.dt_fitting.LMRefineResult`, plus the shared fit-quality
   gates (below).
8. Build the per-technique diagnostics dataclass and evaluate the confidence formula
   (capped by the gates' convergence demotion when it applies).

The :class:`~spindoctor.nav_technique.dt_fitting.LMRefineResult` exposes the converged offset
:attr:`~spindoctor.nav_technique.dt_fitting.LMRefineResult.offset_vu`, the rotation
:attr:`~spindoctor.nav_technique.dt_fitting.LMRefineResult.rotation_rad` (zero when rotation was not
fit), the parameter :attr:`~spindoctor.nav_technique.dt_fitting.LMRefineResult.covariance`, the
per-vertex :attr:`~spindoctor.nav_technique.dt_fitting.LMRefineResult.residuals_px` at the final
estimate, the per-vertex final :attr:`~spindoctor.nav_technique.dt_fitting.LMRefineResult.weights`,
the weighted :attr:`~spindoctor.nav_technique.dt_fitting.LMRefineResult.rms_px`, the unweighted
:attr:`~spindoctor.nav_technique.dt_fitting.LMRefineResult.raw_rms_px` (well-defined even when the
weighted RMS collapses to zero; averaged over the polarity-accepted vertices only, so a single
polarity-rejected vertex's near-infinite synthetic residual cannot inflate it — the mis-convergence
gate that consumes it stays meaningful in multi-body scenes where a secondary body contributes a
few wrong-polarity vertices), the
:attr:`~spindoctor.nav_technique.dt_fitting.LMRefineResult.iterations` count, the
:attr:`~spindoctor.nav_technique.dt_fitting.LMRefineResult.converged` flag (true when either
the DT-LM met its step tolerance or the final gradient-ridge stage applied and met its own —
a quantized-DT stall that burns the LM budget at the integer seed is routine on dense edge
scenes; note this certifies LOCAL optimality only, since the ridge is seeded from the DT-LM
pose and displacement-capped, so it polishes the same lock rather than re-checking the
acquisition), the
:attr:`~spindoctor.nav_technique.dt_fitting.LMRefineResult.degenerate` flag (set when no vertex
survives reweighting), the
:attr:`~spindoctor.nav_technique.dt_fitting.LMRefineResult.inlier_count`, and the
:attr:`~spindoctor.nav_technique.dt_fitting.LMRefineResult.polarity_rejected_count` (the
polarity-fraction numerator the fit-quality gates consume).

Shared fit-quality gates
========================

``src/spindoctor/nav_technique/dt_fit_gates.py`` is the shared verdict layer over the
fitting machinery's health signals; every DT technique calls
:func:`~spindoctor.nav_technique.dt_fit_gates.evaluate_dt_fit_gates` with its
:class:`~spindoctor.nav_technique.dt_fitting.LMRefineResult`, its coarse acquisition score,
and its per-technique thresholds
(:class:`~spindoctor.nav_technique.dt_fit_gates.DTFitGateConfig`, read from the technique's
``tuning`` block). Three signals feed the verdict:

- **LM convergence** (``lm_unconverged_confidence_cap``). A fit that reached no local
  optimum in either stage is unverified, not necessarily wrong: formal convergence flickers
  on healthy dense-edge scenes, so the consequence is a demotion, not a reject — the
  post-sigmoid confidence is capped just below the high tier's boundary, denying the fit
  solo high-tier standing while leaving genuine cross-technique corroboration able to
  rescue the frame. Because convergence certifies only local optimality of the pose the
  acquisition selected, a wrong lock that converges is not caught here; that is what the
  combined polarity gate below is for.
- **Polarity-rejection fraction** (``spurious_max_polarity_rejection_fraction``,
  ``spurious_unconverged_polarity_rejection_fraction``). The fraction of model vertices
  whose local gradient direction disagrees with the model normal at the seed. Healthy
  multi-body frames run around 11 % (a small secondary body on the ansa contributes
  wrong-polarity vertices while the dominant limb fits cleanly), so the standalone hard
  gate sits far above that band and catches only grossly wrong locks. The discriminating
  form is the *combined* gate: an unconverged LM whose polarity rejection is also elevated
  is the signature of a coarse-seed mis-lock the LM could neither escape nor verify — a
  Cassini body-mostly-offscreen investigation measured 12.7 % rejection at the iteration
  cap on a seed 6.8 px off, while the healthy frames at comparable rejection all converge.
  Both polarity gates are inert for the polarity-free ring-edge fit.
- **Coarse acquisition quality** (``spurious_min_coarse_peak_fraction``). The winning
  coarse shift's edge-pixel match fraction. A seed whose best shift leaves most of the
  model off every detected edge never had a lock to refine — the LM then polishes noise.
  The shipped threshold is deliberately conservative plumbing; deriving a
  library-calibrated lock is separate calibration work.

A firing hard gate ORs into the technique's own spurious decision (the reasons are
logged); the convergence demotion caps the calibrated confidence after the sigmoid.
The measured quantities land on the technique diagnostics (``lm_converged``,
``polarity_rejection_fraction``, ``coarse_peak_fraction``) so every gate decision is
auditable per frame.

Examples
========

The shared helpers operate on numpy arrays rather than on observations, so the worked examples
below are numerical illustrations rather than image-library scenes.

**Coarse NCC scan size.**  A typical body limb produces 200 vertices on a 1024 × 1024
extended-FOV image; the per-instrument search window is around 30 pixels in each axis. The
coarse scan evaluates the binary cross-correlation at
:math:`(2 \cdot 30 + 1)^{2} = 3721` integer offsets, summing approximately 200 polyline
indices per offset for a total of order :math:`7.4 \times 10^{5}` boolean accumulations on
edge-mask values. Memory traffic is dominated by the polyline-index gather; the FFT
alternative would touch :math:`1024^{2}` pixels regardless of polyline support and is slower
for windows this size.

**LM iteration count.**  An on-axis body limb with sub-pixel SPICE pointing usually converges
in 4-8 iterations; the default :data:`~spindoctor.nav_technique.dt_fitting.DEFAULT_LM_MAX_ITERATIONS`
of 30 is a safety net for pathological inputs. When the polyline lands inside a flat DT
plateau the refiner exits via the
:math:`\lambda \ge 10^{6}` damping bound rather than reaching the iteration cap.

**Tukey weight on a clean limb.**  With per-vertex prior sigma 1.0 px and a converged DT RMS
of 0.5 px, the scaled residuals lie well inside the cutoff
(:math:`0.5 / 1.0 \approx 0.5 \ll 4.685`); every vertex retains a near-unit weight and the
fit reduces to a weighted-least-squares problem with the prior precision dominating. When
the same fit lands a vertex 6 sigma away from any image edge, the biweight zeroes that
vertex's weight on the first reweighting and the LM step ignores it for every subsequent
iteration.

**Rank-deficient covariance.**  A perfectly straight ring-edge polyline produces an
information matrix of rank one — only the radial direction is constrained; the along-edge
direction is unobservable. The pseudoinverse via
:data:`~spindoctor.nav_technique.dt_fitting.DEFAULT_PINVH_RCOND` maps the null information
eigenvalue to a *zero* covariance eigenvalue: the unobservable direction lies in the null
space of both the information matrix and its pseudoinverse, so the marginal variance along
the tangent is exactly zero (the same zero-variance / null-space convention the ring-edge
rank-1 projection relies on). The orchestrator's ensemble combine, which uses the
Moore-Penrose pseudoinverse to fuse covariances, recognizes the null direction as a
zero-information contribution rather than a perfectly measured axis, treats the technique's
result as a one-dimensional radial constraint in the joint solve, and — when no other
technique constrains the tangent — surfaces the unobservable axis through the
``sigma_along_unobservable_px`` sentinel downstream instead of a numeric per-axis sigma.
