==========================================================
Haze Symmetry (TitanHazeNav)
==========================================================

Overview
========

:class:`~spindoctor.nav_technique.nav_technique_titan_haze.TitanHazeNav` recovers a single
translation from the solar symmetry of a body whose atmosphere hides its surface. It consumes
the one :data:`~spindoctor.feature.feature_type.NavFeatureType.TITAN_LIMB` feature the haze
model emits (see :doc:`dev_guide_navigation_models_titan`) and measures two independent
components in a rotated frame: the displacement perpendicular to the body-center-to-sub-solar
line, from the shift that maximizes mirror symmetry, and the displacement along that line,
from a free-radius circle fit to the sunward limb arc. The method is published as Hanson,
French, Waugh, Barth and Anderson (2025), *Geophysical Research Letters*,
doi:10.1029/2024GL113415.

Feasibility passes when at least one ``TITAN_LIMB`` feature carrying a
:class:`~spindoctor.feature.geometry.TitanHazeGeometry` payload survives the reliability gate,
and fails with reason ``'no TITAN_LIMB features'`` otherwise. A frame carries at most one hazy
body, so the technique consumes exactly one feature.

The technique's :attr:`~spindoctor.nav_technique.nav_technique.NavTechnique.tier` is
``'primary'``. That is a statement of exclusivity, not of superiority: a hazy body has no
second estimator, so the fallback tier's "superseded when a primary covers the same body"
semantics would never fire and would only mislead a reader.

Two properties of the reported answer matter downstream. Its covariance is strongly
anisotropic -- the mirror-correlation scan localizes the cross-track direction far more tightly
than the circle fit localizes the along-track one -- and its rotation is unobservable, because
a single quasi-circular feature carries no roll evidence.

Theory
======

Absent clouds or visible surface features, a hazy atmosphere is mirror-symmetric about the
image-plane line through the body center and the sub-solar point. Two consequences carry the
method:

- **Cross-track.** The image displacement perpendicular to that line is the shift at which the
  observed brightness is most nearly mirror-symmetric about it. This holds whatever the haze
  altitude is and whatever the filter sees, because it is a statement about a symmetry, not
  about a radius.
- **Along-track.** The limb region facing the sub-solar point is close to circular, so a circle
  fit with a FREE radius to that arc pins the displacement along the line without assuming a
  haze altitude. The free radius is what makes the fit filter-independent: a wavelength-
  dependent haze top changes the fitted radius, not the fitted center.

Symbols used throughout: :math:`\theta` is the symmetry-axis angle;
:math:`\hat{a} = (\sin\theta, \cos\theta)` in :math:`(v, u)` is the unit vector along the axis
pointing toward the sub-solar side; :math:`\hat{c} = (\cos\theta, -\sin\theta)` is its
perpendicular; :math:`r_{\mathrm{solid}}` and :math:`r_{\mathrm{env}}` are the apparent solid
and haze-envelope radii in pixels; and :math:`W` is the search half-window, the scalar maximum
of the observation's two extended-FOV margins.

Cross-track: mirror correlation
-------------------------------

The extended image is resampled onto a grid rotated to the symmetry axis and centered on the
predicted disc center, with cubic interpolation. Both grid axes span
:math:`\pm(r_{\mathrm{env}} + \mathtt{annulus\_outer\_pad\_px} + W)`, so the whole disc stays
in-grid wherever the body actually sits inside the pointing window.

For each integer candidate shift :math:`c` in :math:`[-W, +W]`, the scan forms the mirror pairs
:math:`\bigl(G(c + q, t),\; G(c - q, t)\bigr)` over the annulus domain and scores them with a
**Pearson correlation**:

.. math::

    \mathrm{score}(c) = \mathrm{corr}\bigl(G(c + q, t),\; G(c - q, t)\bigr),
    \qquad q > 0 .

Pearson is load-bearing rather than incidental. It is invariant to an affine brightness
relation between the two halves, so a hemispheric north-south brightness difference -- whose
boundary runs roughly along the sun axis, which is exactly what the mirror maps onto itself --
costs the score nothing as long as it scales or offsets one side uniformly. Only structural,
non-affine asymmetry costs correlation. A sum-of-squared-differences or unnormalized
correlation score would read the hemispheric difference as an asymmetry and bias the peak.

The domain is an annulus, not the full disc: with the inner edge at
``annulus_inner_fraction`` of :math:`r_{\mathrm{env}}` and the outer edge one pad beyond it,
structured content in the disc interior (surface windows, cloud fields) cannot bias the
symmetry estimate. In the first pass the along-track position is still unknown, so the annulus
is dilated along the axis by :math:`W` into a capsule -- without that, a small body with a
large true along-track error can miss a :math:`t = 0`-centered annulus entirely and produce no
signal at any :math:`c`. Both shapes are symmetric in :math:`s` about the candidate axis, so
neither can bias the peak position.

The integer peak is refined by fitting a parabola through the three points around it, and the
cross-track sigma comes from that parabola's curvature and its peak height:

.. math::

    \sigma_{\mathrm{cross}} = \mathtt{cross\_sigma\_scale}
        \sqrt{\frac{1 - s_{\mathrm{pk}}}{2 \, |a|}}

clamped to :math:`[\mathtt{sigma\_floor\_cross\_px},\, W]`. It is a noise-deficit heuristic, and
``cross_sigma_scale`` exists to make the resulting z-scores unit-normal against planted truth.

An optional angle refinement repeats the scan over :math:`\theta` offsets within
``angle_refine_deg`` and adopts a refined angle only when its peak beats the SPICE angle's peak
by more than ``angle_refine_min_gain``. A hazy atmosphere's symmetry axis is known to tilt a
few degrees from the spin axis; this absorbs the tilt without trusting a noisy fit. Refinement
is skipped when the model reports a degenerate axis.

Along-track: sunward limb-arc circle fit
----------------------------------------

With the cross-track shift fixed, rays are cast from the shifted center over
:math:`\pm\mathtt{sector\_half\_angle\_deg}` about the symmetry axis at ``ray_step_deg``
spacing. Each ray samples a radial brightness profile, median-filters it, and takes the most
negative outward gradient -- the steepest falloff into sky -- as its limb radius
:math:`\rho_\phi`, refined by parabolic interpolation on the three gradient samples around the
minimum. The search window's width, not a haze-altitude assumption, is what bounds where the
limb may sit.

Two ray-drop rules police the result. A ray is dropped unless its gradient minimum clears
``min_gradient_snr`` times the median absolute deviation of the gradient over the window, and
it is dropped when the minimum lands on the first or last sample of that window. The second
rule is the guard against a specific confident-wrong failure: a body displaced past the window
returns a cluster of rays pinned at exactly the window bound, whose mutual agreement then wins
the robust fit and produces a gate-passing, floor-sigma answer wrong by the whole excess. The
radius gate cannot see that case, because the saturation radius is inside the gate band by
construction.

The surviving ray endpoints are fitted with a circle whose center is constrained to the
symmetry axis, minimizing a Tukey biweight loss over the scalar along-track shift :math:`d` and
the free radius :math:`R`:

.. math::

    \min_{d,\, R} \sum_\phi \rho_{\mathrm{Tukey}}
        \left( \frac{e_\phi}{s_{\mathrm{MAD}}} \right),
    \qquad
    e_\phi = \bigl\| x_\phi - (p_1 + d\,\hat{a}) \bigr\| - R .

The solve is iteratively reweighted least squares: each outer iteration recomputes
:math:`s_{\mathrm{MAD}} = 1.4826 \, \mathrm{MAD}(e_\phi)` over the current inliers, reweights
with :func:`~spindoctor.nav_technique.dt_fitting.tukey_biweight_weights`, and takes one
full Gauss-Newton step on :math:`(d, R)`. The along-track sigma is the corresponding entry of
the weighted covariance :math:`s^2 (J^\top \mathrm{diag}(w) J)^{-1}`, scaled by
``along_sigma_scale`` and clamped to :math:`[\mathtt{sigma\_floor\_along\_px},\, W]`.

Two passes
----------

When the first pass's :math:`|d|` exceeds ``recenter_threshold_px``, the whole sequence repeats
once with the grid centered on the shifted position. One repeat is enough: the first pass
bounds the residual along-track error to the fit-noise scale, so the second pass's annulus and
ray windows are well placed. Gates are evaluated on the FINAL pass only, so a pass-1 score
diluted by the capsule annulus cannot kill a frame the recenter pass exists to rescue.

The assembled offset takes the final pass's cross-track component and the SUM of the
along-track shifts over both passes:

.. math::

    (\Delta v, \Delta u) = c_{\mathrm{sub}}^{(\mathrm{final})} \, \hat{c}
        + \left( \sum_{\mathrm{passes}} d \right) \hat{a} .

Each pass's symmetry scan re-measures the full cross-track offset (the recenter moves the grid
along :math:`\hat{a}` only), so the second pass's cross-track estimate replaces the first's;
only the along-track contributions accumulate.

Contaminant masking rides the hypothesis
----------------------------------------

The haze model ships an undilated mask of the pixels the fits must ignore: nearer bodies, ring
occlusion, in-frame sibling bodies, and bright catalog stars. A pointing error is a scene-wide
translation -- it displaces the body, its moons, the rings, and the stars identically -- so
every relative position in that mask is exact, and the technique applies the mask SHIFTED by
the current center hypothesis rather than statically. The residual uncertainty is absorbed by
dilating the mask along the axis that is not yet solved: by :math:`W` in the first pass, and by
``recenter_threshold_px`` in the recenter pass.

Faint stars, cosmic rays, and hot pixels are deliberately unmasked. Faint point sources are a
few low-amplitude pixels against thousands of mirror pairs in the Pearson score and a handful
of samples in a median-filtered radial profile; cosmic rays and hot pixels additionally have no
predicted position to shift with the hypothesis, so a static mask for them would be wrong as
soon as the pointing error is non-zero.

Covariance and rotation
-----------------------

The two per-axis sigmas are rotated into image axes,

.. math::

    \Sigma_{vu} = M \,
        \mathrm{diag}\bigl(\sigma_{\mathrm{cross}}^2,\ \sigma_{\mathrm{along}}^2\bigr) \,
        M^\top ,

with :math:`M`'s columns :math:`\hat{c}` and :math:`\hat{a}` expressed in :math:`(v, u)`, and
the configured ``model_error_floor_px`` is then added in quadrature by
:func:`~spindoctor.nav_technique.nav_technique.add_model_error_floor`. The anisotropy is the
physical content of the result; an isotropic covariance would hand the ensemble a wrong error
ellipse in both directions at once.

On an instrument that does not fit camera rotation the result carries a (2, 2) covariance and a
``None`` rotation. Where the per-instrument
:attr:`~spindoctor.nav_orchestrator.nav_context.NavContext.fit_camera_rotation` is true, the
result instead carries the rank-deficient (3, 3) form from
:func:`~spindoctor.nav_technique.nav_technique.embed_rotation_unobservable` with a zero
rotation and the unobservable sentinel sigma, matching
:class:`~spindoctor.nav_technique.nav_technique_body_blob.BodyBlobNav`'s equally rotation-blind
centroid. The physical claim is the same either way -- no rotation evidence -- but matching the
fleet's degrees-of-freedom convention keeps a hazy-body-plus-stars frame fusable.

Restrictions and assumptions
----------------------------

- **The haze is mirror-symmetric about the sun axis.** Clouds, seasonal hemispheric structure,
  and a symmetry axis tilted off the SPICE sun direction all violate it. The annulus
  restriction, the Pearson score's affine invariance, and the Tukey weighting are the designed
  mitigations, and the simulator's haze-structure keys (see :ref:`sim-atmosphere`) exist to
  attack each of them separately.
- **The sunward limb arc is circular.** A real haze limb is not exactly a circle, which is why
  the inlier residual RMS is gated and why the reported covariance carries a model-error floor
  the statistical fit cannot see.
- **The center shift and the free radius are partly degenerate.** The sunward-sector fit
  determines the limb position :math:`d + R` far better than it separates :math:`d` from
  :math:`R`. Widening the sector is what buys the separation: over the real Cassini cohort the
  correlation between the two fell from 0.984 at a 60-degree half-sector to 0.942 at 80
  degrees, a factor 3.5 in the variance the degeneracy adds to :math:`d`.
- **A small disc at high phase is the weak regime.** The sunward arc has its least support
  there, and it is where the planted-truth campaign's entire along-track tail lives: bodies at
  or above a 40 px solid radius give an along-track P95 of 0.72 px across all families and
  phases, while bodies below 40 px above 60 degrees phase give P95 3.01 px.
- **Radial sampling reaches** :math:`r_{\mathrm{env}} + \mathtt{radial\_outer\_pad\_px} + W`,
  and :math:`W` is 140 px on a Cassini NAC, so a large well-framed body can lose whole rays to
  out-of-frame outer samples. The ray-drop rule is correct; sizing the reach by the full search
  window rather than by where the limb can actually be is what costs those rays.
- **Rotation is unobservable** from a single quasi-circular feature, as described above.

Sources of uncertainty
----------------------

The cross-track sigma is a noise-deficit heuristic on the correlation peak's curvature; the
along-track sigma is the robust circle fit's own Gauss-Newton covariance. Both measure
statistical precision only. Model error -- the haze envelope's departure from a circle, and the
wavelength-dependent haze top the free radius absorbs into :math:`R` rather than into
:math:`d` -- is priced by the per-axis sigma floors and the covariance model-error floor, all
three held on real-frame evidence rather than on the simulator's tighter numbers.

One consequence of the along-track floor is worth stating plainly: ``hypot(1.00, 0.20)`` is
1.02 px against the ``high`` confidence tier's ``max_sigma_px`` of 0.5 in
``config_540_orchestrator.yaml``, so a frame whose only content is a hazy body caps at the
``medium`` tier however good the fit. For a single quasi-circular feature whose along-track
coordinate is degenerate with a free radius, that is the honest statement.

Configuration
=============

Unlike the other techniques, the haze fit's numeric tunables live with the body they describe,
in ``titan.navigation`` in ``src/spindoctor/config_files/config_060_titan.yaml``; only the
covariance floor and the confidence coefficients live in
``techniques.TitanHazeNav`` in ``config_510_techniques.yaml``. The
:doc:`model page <dev_guide_navigation_models_titan>` documents the emission-side keys
(``titan.navigation`` scalars plus ``titan.atmosphere_height``); the two fit blocks are here.

``titan.navigation.symmetry`` -- the cross-track scan:

.. list-table::
   :header-rows: 1
   :widths: 34 12 54

   * - Key
     - Default
     - Effect
   * - ``annulus_inner_fraction``
     - ``0.55``
     - Inner edge of the scoring annulus as a fraction of the envelope radius. Raising it makes
       the score more limb-dominated and less sensitive to disc-interior structure.
   * - ``annulus_outer_pad_px``
     - ``6.0``
     - Pixels of annulus beyond the envelope radius, and the pad the rotated grid extends by.
   * - ``angle_refine_deg``
     - ``5.0``
     - Half-range of the symmetry-angle search. Set to zero internally on a degenerate axis.
   * - ``angle_refine_step_deg``
     - ``0.5``
     - Angular spacing of that search.
   * - ``angle_refine_min_gain``
     - ``0.02``
     - Peak-score improvement a refined angle must beat before it replaces the SPICE angle.
   * - ``min_peak_score``
     - ``0.60``
     - ``peak_score`` gate: minimum Pearson score at the winning shift.
   * - ``min_valid_fraction``
     - ``0.50``
     - ``valid_fraction`` gate: minimum fraction of annulus mirror pairs usable at that shift.
   * - ``max_second_peak_ratio``
     - ``0.90``
     - ``second_peak`` gate: maximum normalized height of the strongest competing peak at least
       3 px away.
   * - ``cross_sigma_scale``
     - ``0.10``
     - Multiplier on the noise-deficit cross-track sigma, set so planted-truth cross-track
       z-scores are unit-normal on the rows the floor does not clamp.
   * - ``sigma_floor_cross_px``
     - ``0.30``
     - Lower clamp on the reported cross-track sigma.

``titan.navigation.arc`` -- the along-track circle fit:

.. list-table::
   :header-rows: 1
   :widths: 34 12 54

   * - Key
     - Default
     - Effect
   * - ``sector_half_angle_deg``
     - ``80.0``
     - Half-width of the sunward ray sector. Wider separates the center shift from the free
       radius better; narrower trades random error for bias when limb sharpness varies with ray
       angle.
   * - ``ray_step_deg``
     - ``2.0``
     - Angular spacing of the rays, so the sector offers 81 of them at the default width.
   * - ``radial_step_px``
     - ``0.5``
     - Sampling step along each ray.
   * - ``radial_inner_fraction``
     - ``0.80``
     - Inner end of each radial profile as a fraction of the SOLID radius (the symmetry
       annulus's inner fraction scales the ENVELOPE radius instead).
   * - ``radial_outer_pad_px``
     - ``6.0``
     - Pixels sampled beyond the envelope radius. Keep it well clear of ~1.5 px, or the rule
       that drops a ray whose steepest falloff lands on its window bound has no margin outside
       the window to detect.
   * - ``median_filter_samples``
     - ``5``
     - Taps of the per-ray median filter.
   * - ``min_gradient_snr``
     - ``8.0``
     - Ratio of the gradient minimum to the gradient's own median absolute deviation below
       which a ray is dropped.
   * - ``min_rays``
     - ``20``
     - ``ray_yield`` and ``arc_inliers`` gates: minimum surviving and minimum inlier ray counts.
   * - ``min_inlier_fraction``
     - ``0.50``
     - ``arc_inliers`` gate: minimum inlier share of the surviving rays.
   * - ``max_residual_rms_px``
     - ``2.0``
     - ``arc_residual`` gate: maximum inlier residual RMS. On the real cohort the frames between
       2 and 3 px of residual lock measurably wrong, so the apparent gap above this cap is not
       a safe place to raise it to.
   * - ``tukey_c``
     - ``4.685``
     - Tukey biweight tuning constant, in units of the robust scale.
   * - ``along_sigma_scale``
     - ``1.0``
     - Multiplier on the circle fit's own along-track sigma.
   * - ``sigma_floor_along_px``
     - ``1.00``
     - Lower clamp on the reported along-track sigma. This is the term that caps a hazy-body-only
       frame at the ``medium`` confidence tier.

Per-instrument overrides
------------------------

None of these keys is overridden per instrument. The search half-window the fit uses comes from
the per-instrument extended-FOV margins on
:class:`~spindoctor.nav_orchestrator.instrument_config.InstrumentSettings`, read through
:func:`~spindoctor.nav_technique.nav_technique.search_window_for_obs`.

Confidence formula
------------------

The technique reports a calibrated confidence in :math:`[0, 1]` produced by the shared sigmoid
combination; see :doc:`dev_guide_techniques_confidence` for the per-term arithmetic. The spec is
``techniques.TitanHazeNav`` in ``config_510_techniques.yaml`` and consumes attributes off
:class:`~spindoctor.nav_technique.diagnostics.TitanHazeDiagnostics` plus
:attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.at_edge`.

- :attr:`~spindoctor.nav_technique.diagnostics.TitanHazeDiagnostics.envelope_diameter_px` --
  alpha = 5.55, divisor = 160.0, cap at 1.0. The dominant term, because the failure mode the
  anchors have to separate is entirely a small-disc one: every planted-truth row wrong by more
  than twice its axis bound has an envelope under 140 px. Real Cassini frames run 200-1075 px,
  so on them this term sits at its cap and the quality terms carry the discrimination.
- :attr:`~spindoctor.nav_technique.diagnostics.TitanHazeDiagnostics.symmetry_peak_score` --
  alpha = 2.75, divisor = 1.0, cap at 1.0. Pearson score at the winning cross-track shift,
  already in :math:`[0, 1]`.
- :attr:`~spindoctor.nav_technique.diagnostics.TitanHazeDiagnostics.symmetry_valid_fraction` --
  alpha = 1.75, divisor = 1.0, cap at 1.0. Usable share of the annulus mirror pairs; contaminant
  masking and frame edges are what reduce it.
- :attr:`~spindoctor.nav_technique.diagnostics.TitanHazeDiagnostics.arc_inlier_fraction` --
  alpha = 1.50, divisor = 1.0, cap at 1.0. Share of the surviving limb rays the robust circle
  fit kept.
- :attr:`~spindoctor.nav_technique.diagnostics.TitanHazeDiagnostics.arc_residual_rms_px` --
  alpha = -2.50, divisor = 3.0, cap at 1.0. NEGATIVE: a large inlier residual means the sunward
  limb is not the circle the method assumes. The coefficient is bounded at -2.5 in the fit's own
  configuration rather than left at the -15.12 an unconstrained solve returns, which would be a
  near-hard gate at 0.5 px of residual -- harmless in a simulator whose limb is a perfect
  circle, ruinous on real frames whose median residual is 1.1 px.

Hard-zero gate: :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.at_edge`
forces confidence to zero before the sigmoid evaluates, because a fit whose assembled offset
reaches the search window may be reporting the window bound rather than the body. The constant
baseline is :math:`\alpha_0 = -9.30`. The anchors are fitted and verified by
``util/titan_truth/fit_confidence.py``; confidence stays under the program-wide
``confidence_provisional`` marker until a real-anchored recalibration lands.

Implementation
==============

Source files:

- ``src/spindoctor/nav_technique/nav_technique_titan_haze.py`` --
  :class:`~spindoctor.nav_technique.nav_technique_titan_haze.TitanHazeNav`, the tuning-block
  readers, the covariance rotation, and the gate table.
- :mod:`spindoctor.nav_technique.titan_fitting` -- the pure fitting library: rotated-grid
  resampling (``grid``), the mirror-correlation scan (``symmetry``), the radial profiles and
  constrained robust circle fit (``arc``), and the two-pass driver (``driver``). It imports
  nothing from the package beyond the shared array types and two
  :mod:`spindoctor.nav_technique.dt_fitting` weighting helpers, so the whole algorithm is
  exercisable on synthetic arrays with no observation and no SPICE kernel.
- ``src/spindoctor/nav_technique/diagnostics.py`` --
  :class:`~spindoctor.nav_technique.diagnostics.TitanHazeDiagnostics`; documented at
  :doc:`dev_guide_techniques_diagnostics`.

Public class :class:`~spindoctor.nav_technique.nav_technique_titan_haze.TitanHazeNav`, base
:class:`~spindoctor.nav_technique.nav_technique.NavTechnique`. Self-registers via
``__init_subclass__`` so ``NavTechnique._registry`` discovers it.

Class attributes:

- :attr:`~spindoctor.nav_technique.nav_technique_titan_haze.TitanHazeNav.name` --
  ``'TitanHazeNav'``.
- :attr:`~spindoctor.nav_technique.nav_technique_titan_haze.TitanHazeNav.accepts_feature_types`
  -- ``frozenset({TITAN_LIMB})``.
- :attr:`~spindoctor.nav_technique.nav_technique_titan_haze.TitanHazeNav.requires_prior` --
  ``False``. Runs in pass 1 of the orchestrator's two-pass pipeline.
- :attr:`~spindoctor.nav_technique.nav_technique_titan_haze.TitanHazeNav.tier` -- ``'primary'``.
- :attr:`~spindoctor.nav_technique.nav_technique_titan_haze.TitanHazeNav.confidence_attributes`
  -- ``{'at_edge', 'symmetry_peak_score', 'symmetry_valid_fraction', 'arc_inlier_fraction',
  'arc_residual_rms_px', 'envelope_diameter_px'}``.

Public methods (autodocumented at :doc:`/api_reference/api_nav_technique`):
:meth:`~spindoctor.nav_technique.nav_technique_titan_haze.TitanHazeNav.is_feasible` and
:meth:`~spindoctor.nav_technique.nav_technique_titan_haze.TitanHazeNav.navigate`.

Fitting-library surface
-----------------------

:mod:`spindoctor.nav_technique.titan_fitting` re-exports the whole algorithm as pure functions
taking arrays and frozen parameter dataclasses and returning frozen result dataclasses:

- :func:`~spindoctor.nav_technique.titan_fitting.resample_rotated_grid` -- the rotated
  :math:`(s, t)` resample of an image and its validity mask.
- :func:`~spindoctor.nav_technique.titan_fitting.symmetry_scan` -- the mirror-correlation scan
  including angle refinement, returning a
  :class:`~spindoctor.nav_technique.titan_fitting.SymmetryFitResult`.
- :func:`~spindoctor.nav_technique.titan_fitting.radial_profiles` and
  :func:`~spindoctor.nav_technique.titan_fitting.limb_radii_from_profiles` -- per-ray sampling
  and limb extraction.
- :func:`~spindoctor.nav_technique.titan_fitting.constrained_circle_fit` -- the axis-constrained
  robust circle fit, returning an
  :class:`~spindoctor.nav_technique.titan_fitting.ArcFitResult`.
- :func:`~spindoctor.nav_technique.titan_fitting.fit_titan_center` -- the two-pass driver the
  technique wraps.

Gates
-----

Every gate is evaluated on the final pass. A failure yields a spurious
:class:`~spindoctor.nav_technique.technique_result.NavTechniqueResult` naming the gate in
:attr:`~spindoctor.nav_technique.diagnostics.TitanHazeDiagnostics.gate_failed`; the frame then
resolves through the standard generic statuses, so a hazy-body-only frame whose fit fails ends
``all_techniques_spurious``.

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Gate name
     - Fails when
   * - ``valid_fraction``
     - The usable share of annulus mirror pairs at the winning shift is below
       ``min_valid_fraction``.
   * - ``peak_score``
     - The Pearson score at that shift is below ``min_peak_score``.
   * - ``second_peak``
     - A competing local maximum at least 3 px away exceeds ``max_second_peak_ratio`` of the
       normalized peak height.
   * - ``ray_yield``
     - Fewer than ``min_rays`` rays survive profile extraction.
   * - ``arc_inliers``
     - The robust fit keeps fewer than ``min_rays`` rays, or fewer than
       ``min_inlier_fraction`` of the surviving ones.
   * - ``arc_radius``
     - The fitted radius falls outside
       :math:`[0.98\,r_{\mathrm{solid}},\; 1.05\,(r_{\mathrm{env}} + W)]`.
   * - ``arc_residual``
     - The inlier residual RMS exceeds ``max_residual_rms_px``.

Two further conditions are flags rather than rejections: a cross-track or along-track component
that reaches the search window sets
:attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.at_edge`, which the
ensemble treats conservatively and which forces confidence to zero.

Every gate is logged, whether it fired or not. Inside the ``TECHNIQUE: TitanHazeNav`` log
section the technique prints one line per gate with its measurement, its threshold, and a
verdict of ``PASS``, ``FAIL``, ``EDGE`` (an at-edge flag), or ``SKIP`` (a gate the fit returned
before reaching), so an operator reading a per-image log sees why a frame was accepted or
rejected without re-running anything. The band the ``arc_radius`` row quotes comes from
:data:`~spindoctor.nav_technique.titan_fitting.arc.ARC_RADIUS_MIN_FRACTION` and
:data:`~spindoctor.nav_technique.titan_fitting.arc.ARC_RADIUS_MAX_FRACTION`, the same constants
the gate itself tests.

Diagnostics
-----------

:class:`~spindoctor.nav_technique.diagnostics.TitanHazeDiagnostics` records every gate input plus
the geometry the fit worked from:

- :attr:`~spindoctor.nav_technique.diagnostics.TitanHazeDiagnostics.sun_angle_deg`,
  :attr:`~spindoctor.nav_technique.diagnostics.TitanHazeDiagnostics.axis_degenerate`,
  :attr:`~spindoctor.nav_technique.diagnostics.TitanHazeDiagnostics.theta_refined_deg` -- the
  symmetry axis the final pass used, whether it was localizable at all, and how far refinement
  moved it from the SPICE direction.
- :attr:`~spindoctor.nav_technique.diagnostics.TitanHazeDiagnostics.phase_deg` and
  :attr:`~spindoctor.nav_technique.diagnostics.TitanHazeDiagnostics.envelope_diameter_px` -- the
  regime the frame sits in. The diameter is a confidence term.
- :attr:`~spindoctor.nav_technique.diagnostics.TitanHazeDiagnostics.cross_track_px` and
  :attr:`~spindoctor.nav_technique.diagnostics.TitanHazeDiagnostics.along_track_px` -- the
  assembled offset resolved back onto the axes the covariance describes.
- :attr:`~spindoctor.nav_technique.diagnostics.TitanHazeDiagnostics.symmetry_peak_score`,
  :attr:`~spindoctor.nav_technique.diagnostics.TitanHazeDiagnostics.symmetry_valid_fraction`,
  :attr:`~spindoctor.nav_technique.diagnostics.TitanHazeDiagnostics.symmetry_second_peak_ratio`
  -- the three cross-track gate inputs. The peak score is ``None`` when no candidate shift had
  enough usable signal to correlate at all, which is a different statement from a measured zero.
- :attr:`~spindoctor.nav_technique.diagnostics.TitanHazeDiagnostics.arc_rays_total`,
  :attr:`~spindoctor.nav_technique.diagnostics.TitanHazeDiagnostics.arc_rays_inlier`,
  :attr:`~spindoctor.nav_technique.diagnostics.TitanHazeDiagnostics.arc_inlier_fraction`,
  :attr:`~spindoctor.nav_technique.diagnostics.TitanHazeDiagnostics.arc_residual_rms_px` -- the
  along-track gate inputs. The residual is ``None`` when the robust fit rejected every ray,
  reported as absent rather than as a perfect zero that a falling confidence sigmoid would read
  as maximally good.
- :attr:`~spindoctor.nav_technique.diagnostics.TitanHazeDiagnostics.fitted_haze_radius_km` and
  :attr:`~spindoctor.nav_technique.diagnostics.TitanHazeDiagnostics.filters` -- the fitted radius
  in kilometers and the filters it was measured through, recorded so a haze-radius table per
  instrument, filter, and phase bin can be accumulated from production output.
- :attr:`~spindoctor.nav_technique.diagnostics.TitanHazeDiagnostics.recentered` and
  :attr:`~spindoctor.nav_technique.diagnostics.TitanHazeDiagnostics.gate_failed` -- whether the
  second pass ran, and which gate rejected the fit.

Call path
---------

Call path traced through
:meth:`~spindoctor.nav_technique.nav_technique_titan_haze.TitanHazeNav.navigate`:

1. Open a logged section and select the single eligible ``TITAN_LIMB`` feature.
2. Read the search-window margins off the observation via
   :func:`~spindoctor.nav_technique.nav_technique.search_window_for_obs` and take their scalar
   maximum: the two fit components live in the rotated frame, where per-axis image margins
   cannot be applied cleanly.
3. Build the two frozen parameter dataclasses from ``titan.navigation.symmetry`` and
   ``titan.navigation.arc``, zeroing the angle-refinement range when the feature reports a
   degenerate axis.
4. Run :func:`~spindoctor.nav_technique.titan_fitting.fit_titan_center` over
   :attr:`~spindoctor.nav_orchestrator.nav_context.NavContext.image_ext` -- the raw extended
   image, not the gradient or distance-transform planes the shape-fitting techniques consume,
   because this method reads brightness symmetry and a radial brightness falloff directly.
5. Log the gate table.
6. Resolve the assembled offset back onto :math:`\hat{c}` and :math:`\hat{a}`, populate the
   diagnostics, and short-circuit to a spurious result when either half of the fit named a
   failed gate.
7. Otherwise rotate the per-axis sigmas into image axes, add the model-error floor, branch the
   result shape on
   :attr:`~spindoctor.nav_orchestrator.nav_context.NavContext.fit_camera_rotation`, evaluate the
   confidence spec via
   :func:`~spindoctor.nav_technique.confidence.evaluate_sigmoid_combination`, and log the
   per-term breakdown via
   :func:`~spindoctor.nav_technique.nav_technique.log_confidence_breakdown`.

Examples
========

``titan_haze`` (simulated Cassini ISS NAC, 360 px frame)
    The standing base scene for the offset sweeps: a 120 px-radius hazy body at 60 degrees
    phase, rendered with a soft haze limb a few pixels above the solid radius so the free-radius
    arc fit has a genuine altitude mismatch to absorb. Across the dense sub-pixel sweep the
    total recovery error stays between 0.11 and 0.29 px, and across the wide sweep it holds
    flat out to a planted 45 px -- the extended-FOV search margin, not a limit of the method.
    See :doc:`/simulator_report/simulator_report`.

``W1822132529_1`` (Cassini ISS WAC)
    An unoccluded, well-resolved Titan; the integration-marked end-to-end case, exercising the
    real bright-star mask queries against the YBSC and Tycho-2 catalogs alongside the fit
    itself.

Cross-filter and star-anchored cohort frames
    Over the 82-frame Cassini cohort in ``util/titan_cohort``, frames where a star technique
    independently locks give a per-frame absolute anchor. Against those anchors the haze fit
    disagrees by 0.99 px rms cross-track and 1.50 px rms along-track over nine pairs, implying
    about 0.70 and 1.06 px of single-frame error. Repeat frames of one target through one filter
    agree to 0.34 px cross-track, 0.33 px along-track, and 4 km of fitted haze radius. The full
    record is ``util/titan_cohort/CAMPAIGN_20260726.md``.

Frames sharing the field of view with another body
    A co-visible moon navigated by its own shape features measures the same scene-wide pointing
    offset, so it anchors the haze fit exactly the way a star lock does. Across the cohort
    frames where both commit,
    :class:`~spindoctor.nav_technique.nav_technique_body_limb.BodyLimbNav` on the companion
    body agrees with the haze fit at 2-sigma on 11 of 12 opportunities -- the second-strongest
    real-frame corroboration after the star anchors.
    :class:`~spindoctor.nav_technique.nav_technique_body_disc.BodyDiscCorrelateNav` (3 of 7)
    and :class:`~spindoctor.nav_technique.nav_technique_body_blob.BodyBlobNav` (6 of 12) agree
    less often, which measures those witnesses rather than the haze fit: the companion moons
    are small in these frames, and the blob centroid's own scatter dominates its disagreements
    (its worst reading sits 120 px from every other technique on its frame).
