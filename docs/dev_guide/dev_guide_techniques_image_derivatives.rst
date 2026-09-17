==================================================
Image Derivatives (Shared Gradient and Edge DT)
==================================================

Overview
========

Image derivatives are the shared image-side products that every distance-transform technique
samples to align its model polylines against the observed image. Three quantities are produced
once per navigation by the orchestrator and attached to the per-image
:class:`~spindoctor.nav_orchestrator.nav_context.NavContext`: a smoothed gradient magnitude image, a
signed per-pixel gradient vector image, and a thresholded, non-maximum-suppressed,
truncated-distance-transform of the gradient ridge. Computing them once keeps the per-image
cost bounded regardless of how many DT-based techniques run later.

A combined entry point shares the heavy Gaussian + Sobel pass across all three products; two
additional entry points produce the gradient-only and DT-only subsets when a caller does not
need the full bundle. The DT-based techniques
(:class:`~spindoctor.nav_technique.nav_technique_body_limb.BodyLimbNav`,
:class:`~spindoctor.nav_technique.nav_technique_body_terminator.BodyTerminatorNav`,
:class:`~spindoctor.nav_technique.nav_technique_ring_edge.RingEdgeNav`) consume the DT and gradient
vector images directly; see :doc:`dev_guide_techniques_dt_fitting` for the fitter that
operates on these products.

Theory
======

The image-derivatives pass turns the raw extended-FOV pixel array into three intermediate
products that downstream polyline fitters can sample cheaply.

Gaussian smoothing then Sobel
-----------------------------

The raw image is first smoothed by an isotropic Gaussian with a per-pixel standard deviation
matched to the typical instrument PSF. The smoothing serves two purposes: it removes
single-pixel noise spikes that would otherwise dominate the gradient, and it rounds the
discrete edge profile so the bilinear-DT samples taken later by the LM refiner are
differentiable.

After smoothing, the image is differentiated by separable Sobel filters along the v and u
axes, producing a signed pair :math:`(g_{v}, g_{u})` at every pixel. The pair is preserved
unchanged in the gradient-vector image. Its Euclidean length

.. math::

    G(v, u) = \sqrt{g_{v}(v, u)^{2} + g_{u}(v, u)^{2}}

is the gradient-magnitude image.

Edge thresholding and Canny-style non-maximum suppression
---------------------------------------------------------

The truncated DT is built from a thinned edge mask rather than the raw gradient magnitude.
Two steps:

- **Threshold.**  Pixels whose gradient magnitude exceeds the threshold

  .. math::

      \tau = k \cdot \sigma_{\mathrm{noise}}

  are kept as edge candidates, where :math:`\sigma_{\mathrm{noise}}` is the MAD-derived
  per-image noise sigma and :math:`k` is the per-image threshold multiplier. A 4-sigma
  default keeps single-pixel noise excursions out of the DT input while admitting limb,
  terminator, and ring edges with margin.

- **Directional non-maximum suppression.**  Each candidate pixel is kept only if its
  magnitude is at least as large as both of its neighbors along the local gradient
  direction. The gradient direction is quantized to four 45-degree sectors (boundaries at
  22.5, 67.5, 112.5, and 157.5 degrees from the u-axis) so the lookup reduces to a small
  fixed set of 3 × 3 shifts. The standard Canny rule keeps the full edge length intact while
  thinning the gradient ridge to one pixel wide — the right input for both the integer
  cross-correlation and the distance transform.

A naive 3 × 3 NMS would discard most pixels along a smooth ridge; the directional check
preserves edge length by comparing each candidate only against the two neighbors along its
own gradient direction.

Truncated distance transform
----------------------------

The thinned edge mask is fed to a distance transform with a documented saturation cap.
Pixels farther than the cap from any edge pixel saturate at the cap value instead of growing
without bound; the cap bounds the LM cost contribution from polyline vertices that fall in
empty regions of the image and bounds the DT array's working range to a documented maximum.

The thresholding intentionally treats *every* pixel above the threshold as an edge candidate;
the per-technique polarity filter rejects vertex matches that disagree on the sign of the
gradient at that location. An entirely empty thresholded mask falls back to a saturated DT
(every pixel at the cap), so downstream consumers always see a fully defined array even when
the input image carries no signal above noise.

Restrictions and assumptions
----------------------------

- The pass is feature-agnostic. Every gradient is taken with the same isotropic Gaussian
  and the same threshold multiplier; per-feature softening is the model's job, not the
  derivative pass's.
- The Gaussian sigma is matched to the typical instrument PSF in pixels. Below the PSF the
  gradient is dominated by noise; well above the PSF, sharp limbs blur out and the DT loses
  contrast against the background.
- The threshold is expressed as a multiple of the per-image noise sigma. An over-confident
  noise estimate (too small) lets noise spikes through; an under-confident one suppresses
  real edges. The orchestrator always computes the noise sigma via
  ``estimate_image_noise_sigma(image, sensor_mask)`` on the *filtered* working image (not
  the raw classifier sigma), so the estimate stays self-consistent with the gradients the
  DT techniques actually fit.
- The input image must be 2-D and contain only finite values. NaN or +/-inf pixels would
  propagate through the Gaussian and Sobel passes and poison every downstream consumer; the
  pass raises rather than silently degrading. The orchestrator sanitizes the per-instrument
  missing-data marker (including the calibrated-IF ``NaN`` marker) to a finite fill before
  invoking this pass, so calibrated frames with NaN dropout markers reach the derivative
  kernels as finite data.

Sources of uncertainty
----------------------

The derivatives are deterministic given the input image and the configuration, so they
contribute no uncertainty in the statistical sense. The product they feed into — the LM
fitter — does carry uncertainty; see :doc:`dev_guide_techniques_dt_fitting` for the
covariance treatment. What the derivatives *do* shape is the scale of the LM cost surface:
the DT cap sets the maximum per-vertex cost contribution, and the threshold multiplier sets
the noise floor below which the DT contains no edge information.

Configuration
=============

Image derivatives carry no YAML configuration of their own. Every numeric default is a
module-level constant exposed through the
:class:`~spindoctor.nav_orchestrator.image_derivatives.ImageDerivativesConfig` dataclass; the
orchestrator's
:class:`~spindoctor.nav_orchestrator.orchestrator.NavOrchestrator` constructor accepts an
``image_derivatives_config`` override and otherwise uses the documented defaults.

- :data:`~spindoctor.nav_orchestrator.image_derivatives.DEFAULT_IMAGE_GRADIENT_SIGMA_PX` — float,
  default ``1.2`` px. Gaussian sigma used to smooth the image before the Sobel operator.
  Matches the typical instrument PSF.
- :data:`~spindoctor.nav_orchestrator.image_derivatives.DEFAULT_EDGE_THRESHOLD_K_SIGMA` — float,
  default ``4.0`` (dimensionless). Multiples of ``image_noise_sigma`` used to threshold the
  gradient magnitude into a binary edge mask. Pixels at or below this threshold are
  discarded regardless of NMS outcome.
- :data:`~spindoctor.nav_orchestrator.image_derivatives.DEFAULT_DT_HALF_WIDTH_PX` — float, default
  ``64.0`` px. Maximum distance returned by the truncated distance transform. Pixels
  farther than this from any thresholded gradient pixel saturate at this value.

The :class:`~spindoctor.nav_orchestrator.image_derivatives.ImageDerivativesConfig` fields:

- :attr:`~spindoctor.nav_orchestrator.image_derivatives.ImageDerivativesConfig.image_gradient_sigma_px`
  — float, default ``DEFAULT_IMAGE_GRADIENT_SIGMA_PX`` px. Per-axis Gaussian sigma; both
  axes share the same value (anisotropic blur is intentionally not exposed because the
  image-side computation must be feature-agnostic).
- :attr:`~spindoctor.nav_orchestrator.image_derivatives.ImageDerivativesConfig.edge_threshold_k_sigma`
  — float, default ``DEFAULT_EDGE_THRESHOLD_K_SIGMA`` (dimensionless). Threshold multiplier
  fed into the gradient-magnitude thresholding step.
- :attr:`~spindoctor.nav_orchestrator.image_derivatives.ImageDerivativesConfig.dt_half_width_px` —
  float, default ``DEFAULT_DT_HALF_WIDTH_PX`` px. Cap on the DT distance.

The dataclass's ``__post_init__`` rejects any non-positive or non-finite field with
:exc:`ValueError`; a malformed override fails fast at construction rather than mid-image.

Implementation
==============

Source files:

- ``src/spindoctor/nav_orchestrator/image_derivatives.py`` —
  :class:`~spindoctor.nav_orchestrator.image_derivatives.ImageDerivativesConfig`,
  :func:`~spindoctor.nav_orchestrator.image_derivatives.build_image_edge_dt`,
  :func:`~spindoctor.nav_orchestrator.image_derivatives.compute_image_gradient_vu`, and
  :func:`~spindoctor.nav_orchestrator.image_derivatives.compute_all_image_derivatives` plus the
  three ``DEFAULT_*`` module constants.
- ``src/spindoctor/support/filters.py`` — the
  :class:`~spindoctor.support.filters.NavFilterSpec` /
  :class:`~spindoctor.support.filters.NavFilterKind` machinery the DT step delegates into for the
  truncated distance transform.

Public surface (autodocumented at :doc:`/api_reference/api_nav_orchestrator`):

- :func:`~spindoctor.nav_orchestrator.image_derivatives.compute_all_image_derivatives` — combined
  entry point that returns the gradient magnitude, edge DT, and gradient-vector products in
  a single Gaussian + Sobel pass. The orchestrator's per-image setup uses this entry point.
- :func:`~spindoctor.nav_orchestrator.image_derivatives.build_image_edge_dt` — returns the
  gradient-magnitude and edge-DT pair only (omits the signed gradient vector).
- :func:`~spindoctor.nav_orchestrator.image_derivatives.compute_image_gradient_vu` — returns the
  signed ``(g_v, g_u)`` gradient-vector image only (omits the DT).
- :class:`~spindoctor.nav_orchestrator.image_derivatives.ImageDerivativesConfig` — frozen dataclass
  carrying the three configurable parameters.

Each public function validates its inputs (finiteness of the input image, positivity of the
configured sigmas, non-negativity of the noise sigma) and raises :exc:`TypeError` or
:exc:`ValueError` on a violation.

Call path
---------

The combined :func:`~spindoctor.nav_orchestrator.image_derivatives.compute_all_image_derivatives`
entry point follows three steps:

1. Validate ``image_noise_sigma`` is finite and non-negative; pick the supplied
   :class:`~spindoctor.nav_orchestrator.image_derivatives.ImageDerivativesConfig` or fall back to
   the documented defaults.
2. Run the shared private smooth-and-Sobel helper once, producing ``(g_v, g_u)``. The
   helper validates that the input image is 2-D and finite and that the smoothing sigma is
   strictly positive.
3. Pass the gradient pair through the shared private edge-DT helper, which computes the
   gradient magnitude, applies the directional Canny-style non-maximum suppression, and
   builds the truncated distance transform via
   :func:`~spindoctor.support.filters.apply_filter` with
   :attr:`~spindoctor.support.filters.NavFilterKind.DISTANCE_TRANSFORM`. Stack the gradient pair
   into the ``(H, W, 2)`` gradient-vector image and return the three products.

The stand-alone :func:`~spindoctor.nav_orchestrator.image_derivatives.build_image_edge_dt` and
:func:`~spindoctor.nav_orchestrator.image_derivatives.compute_image_gradient_vu` entry points each
call the shared smooth-and-Sobel helper exactly once and produce only the subset they
declare. Calling both stand-alone helpers on the same image runs the heavy Gaussian + Sobel
pass twice; prefer the combined entry point when both products are needed.

Layout of the gradient-vector image: the last axis stacks ``g_v`` (index 0) and ``g_u``
(index 1). The polarity filter in :doc:`dev_guide_techniques_dt_fitting` samples this image
at each polyline vertex's shifted position and dot-products the sampled vector against the
model's outward normal.

Examples
========

The image-derivatives helpers operate on numpy arrays rather than on observations; the
worked examples below are numerical illustrations rather than image-library scenes.

**One-pass cost on a typical extended-FOV image.**  A 1024 × 1024 extended-FOV image at the
default ``image_gradient_sigma_px = 1.2`` runs one separable Gaussian (truncated by SciPy's
default at four sigma, radius ``int(4.0 * 1.2 + 0.5) = 5``, an 11 × 11 effective kernel)
plus two separable Sobel passes — three
passes over the image, no full-image FFT. Reusing the
:func:`~spindoctor.nav_orchestrator.image_derivatives.compute_all_image_derivatives` combined entry
point keeps it at three passes; calling
:func:`~spindoctor.nav_orchestrator.image_derivatives.build_image_edge_dt` and
:func:`~spindoctor.nav_orchestrator.image_derivatives.compute_image_gradient_vu` separately on the
same image runs the Gaussian + Sobel pass twice (six passes total) for the same products.

**Threshold scaling on a clean ISS NAC image.**  A typical Cassini ISS NAC frame has
``image_noise_sigma`` near 5 DN. At the default
:data:`~spindoctor.nav_orchestrator.image_derivatives.DEFAULT_EDGE_THRESHOLD_K_SIGMA` of 4.0 the
gradient threshold is :math:`\tau = 20` DN/px. A bright limb on a dark background produces
gradient magnitudes well above 100 DN/px and survives the threshold; isolated single-pixel
noise of order 5 DN/px is rejected.

**DT cap on an empty region.**  A polyline vertex that lands 100 pixels from any edge pixel
is clamped to the saturation cap
:data:`~spindoctor.nav_orchestrator.image_derivatives.DEFAULT_DT_HALF_WIDTH_PX` = 64 px instead of
contributing a 100 px DT residual to the LM cost. The Tukey biweight
(see :doc:`dev_guide_techniques_dt_fitting`) zeroes the vertex's weight on the first
reweighting step regardless, but the cap keeps the linear-system right-hand side from
diverging numerically before that happens.

**Empty edge mask fallback.**  A blank or fully-overexposed image produces no pixels above
the threshold. Rather than emit an empty DT, the helper falls back to a fully saturated DT
(every pixel at the cap) so downstream consumers always see a well-formed array. In
practice the orchestrator's image classifier short-circuits before any technique reaches the
DT — see the ``blank`` and ``fully_overexposed`` classes documented under the orchestrator
pages — so this fallback fires only on pathological inputs.
