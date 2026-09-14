==========================
Camera-rotation correction
==========================

This page documents how the autonomous-navigation pipeline fits an
in-plane camera rotation alongside the per-image translation when the
mission's reconstructed attitude carries a rotation residual large
enough to be observable per-image.

Why per-instrument
==================

An instrument whose reconstructed attitude is essentially free of
rotation residual is navigated in 2 degrees of freedom (``dv``, ``du``),
and one whose attitude carries an image-to-image residual too
inconsistent to be calibrated out mission-wide needs a small rotation
fitted as part of each image's solution, which is what
``fit_camera_rotation`` enables.

Three things decide the setting for a given instrument, and only the
first is about the measurement: whether the residual is frame-varying
rather than one common value a fixed frame-kernel correction would
remove; what the per-frame fit costs, since it turns every technique's
2x2 covariance into a 3x3 one; and whether the instrument's corrected
pointing is wanted as a C-kernel, because a fitted rotation turns about
a per-technique pivot the result does not record and therefore cannot be
expressed as an attitude at all. An instrument with the flag on writes
no corrected C-kernels.

The pivot is the deeper of those. Rotation is fitted per technique, and
each technique turns about its own center -- a vertex centroid for the
DT techniques, a composite centroid for the disc, a weighted catalog
centroid for the star fits. A rigid rotation by theta about pivot P and
the same rotation about pivot Q differ by the pure translation
``(I - R(theta))(P - Q)``, so two techniques describing one physical
twist report *different translations*, which the ensemble then fuses as
though they were comparable. For a fraction of a degree across a couple
of hundred pixels of pivot separation that disagreement is of order a
pixel. A fitted rotation is therefore not a well-defined quantity while
the pivot varies, which is a stronger objection than the kernel one: it
is wrong rather than merely unusable. Fitting every technique about the
FOV center removes it and needs no pivot field recorded anywhere; that
is the direction of the standing rotation design.

The measured per-instrument twist and its frame-to-frame scatter are in
:doc:`/fov_distortion_report/fov_distortion_report`. What each
instrument's flag is actually set to, and why, is stated in its chapter
under :doc:`instruments/instruments` -- one chapter, one setting, one
reason -- rather than enumerated here, where it would drift out of step
with the configuration files.

The flag
========

Each per-camera config block under ``config_4N0_inst_*.yaml`` carries:

.. code-block:: yaml

    fit_camera_rotation: false       # on for exactly one instrument
    max_rotation_deg: 5.0            # bound on the rotation magnitude

The orchestrator reads both fields from
:class:`~spindoctor.nav_orchestrator.instrument_config.InstrumentSettings` and
plumbs them onto :class:`~spindoctor.nav_orchestrator.nav_context.NavContext`
as ``fit_camera_rotation`` and ``max_rotation_deg``. Every technique
reads from the context — never from the obs directly — so the flag is
an image-side property that travels with the navigation, not a
technique-side opt-in.

Parameter vector
================

When ``fit_camera_rotation`` is True, every technique works in a 3-DoF
parameter space. The parameter vector is ``(dv, du, theta)`` with
``theta`` in radians, bounded by ``±deg_to_rad(max_rotation_deg)``.
Each technique's covariance grows from 2x2 to 3x3; the
:class:`~spindoctor.nav_orchestrator.ensemble._CombinedEstimate` wrapping the
ensemble's combined output carries an optional ``rotation_rad`` field
that is populated only on 3-DoF runs.

The rotation pivot is the natural geometric center for each technique:

* :class:`~spindoctor.nav_technique.nav_technique_body_limb.BodyLimbNav`,
  :class:`~spindoctor.nav_technique.nav_technique_body_terminator.BodyTerminatorNav`,
  :class:`~spindoctor.nav_technique.nav_technique_ring_edge.RingEdgeNav`: the
  centroid of the polyline vertices.
* :class:`~spindoctor.nav_technique.nav_technique_body_disc.BodyDiscCorrelateNav`,
  :class:`~spindoctor.nav_technique.nav_technique_ring_annulus.RingAnnulusNav`:
  the centroid of the predicted body / planet centers carried on the
  template payloads (the 3-D NCC pyramid pre-rotates each level's
  template about that pivot).
* :class:`~spindoctor.nav_technique.nav_technique_star_field.StarFieldFromCatalogNav`:
  the centroid of the inlier matched-point set.
* :class:`~spindoctor.nav_technique.nav_technique_star_unique_match.StarUniqueMatchNav`
  in 2-star mode: the centroid of the two predicted positions.

Per-technique strategy
======================

DT-based techniques (limb, terminator, ring edge) fit rotation as the
third Levenberg-Marquardt parameter. The shared
:func:`~spindoctor.nav_technique.dt_fitting.lm_subpixel_refine` helper accepts
``fit_rotation=True`` plus a ``pivot_vu`` and ``pivot_distance_px``
(used to convert rotation steps into pixel-equivalent magnitudes for
the convergence test). The Jacobian against ``theta`` is computed by
central differences on the rotated-vertex DT samples; the M-estimator
information matrix at convergence is inverted via ``pinvh`` to produce
the 3x3 covariance.

Template-NCC techniques run a 3-D NCC pyramid that augments the
existing translation pyramid with a rotation-sample schedule per
:doc:`dev_guide_techniques`:

* Level 0 (coarsest): 11 rotation samples spanning
  ``±max_rotation_deg`` in 1° steps.
* Level 1: 5 samples in 0.5° steps centered on the level-0 winner.
* Level 2: 3 samples in 0.25° steps centered on the level-1 winner.
* Level 3 (full resolution): one sample at the level-2 winner; sub-deg
  refinement falls out of the per-level NCC peak interpolation.

Each level pre-rotates the composite template about the technique's
pivot; the NCC kernel itself is unchanged. The technique builds a 3x3
covariance from the NCC peak's local curvature in ``(dv, du, theta)``;
the translation block derives from the existing 2-D peak curvature and
the rotation block is the second-difference along the rotation axis at
the converged estimate.

Star techniques fit a 2-D similarity transform (rotation + translation,
no scale).
:class:`~spindoctor.nav_technique.nav_technique_star_unique_match.StarUniqueMatchNav`
in 2-star mode rotates the catalog pair onto the detected pair via
``atan2(cross, dot)`` of the centroid-relative vectors.
:class:`~spindoctor.nav_technique.nav_technique_star_field.StarFieldFromCatalogNav`
runs the orthogonal-Procrustes (Kabsch) SVD on the inlier set; the
``det(U @ Vt)`` correction column keeps the result a proper rotation
(no reflection).
:class:`~spindoctor.nav_technique.nav_technique_star_refine.StarRefineNav`
runs the same Procrustes on the per-star residuals when at least two
inliers survive. The 1-star path always reports rotation as
unobservable.

Rank-deficient rotation
=======================

A few technique / scene combinations carry no rotation evidence. The
canonical example is
:class:`~spindoctor.nav_technique.nav_technique_body_blob.BodyBlobNav`: the
brightness-weighted centroid is rotation-invariant about itself, so a
rotation parameter is unobservable from a blob alone. The same
applies to :class:`~spindoctor.nav_technique.nav_technique_star_unique_match.StarUniqueMatchNav`
in 1-star mode, to flat-ring-only scenes from
:class:`~spindoctor.nav_technique.nav_technique_ring_edge.RingEdgeNav`, and to
:class:`~spindoctor.nav_technique.nav_technique_titan_haze.TitanHazeNav`, whose
single quasi-circular haze envelope is rotation-invariant about its own center.

To honor the parameter-vector contract (every technique on the same
image emits the same DoF) without inventing rotation evidence, those
techniques call
:func:`~spindoctor.nav_technique.nav_technique.embed_rotation_unobservable`
to promote the 2x2 translation covariance to a 3x3 with the rotation
diagonal carrying
:data:`~spindoctor.nav_technique.nav_technique.ROTATION_UNOBSERVABLE_VARIANCE`
(``1.0e15`` px², the finite-sentinel substitute for ``+inf`` that
``np.linalg.eigvalsh`` cannot represent). The ensemble's
``pinvh``-based combine sees the rotation eigenvalue as null and
gracefully drops the technique's rotation contribution while still
fusing its translation constraint.

Ensemble combine in 3-D
=======================

:func:`~spindoctor.nav_orchestrator.ensemble.ensemble` operates uniformly on
2-DoF and 3-DoF inputs; it just picks the parameter-vector dimension
that matches the inputs' covariance shape. Mixed-DoF inputs (one 2x2
covariance and one 3x3 covariance in the same image) raise
``ValueError`` — the orchestrator pins the DoF per image via
``context.fit_camera_rotation`` so this assertion should never fire in
production but catches programmer errors in technique implementations.

Single-link clustering by Mahalanobis distance, the precision-weighted
information-form merge, and the rank-deficiency check all extend
naturally to 3-D — see
:doc:`dev_guide_orchestrator` for the underlying math.

JSON output
===========

The metadata curator converts ``rotation_rad`` to ``rotation_deg`` and
``sigma_rotation_rad`` to ``sigma_rotation_deg`` for JSON output
(:func:`~spindoctor.nav_orchestrator.curator.build_metadata_dict`); both
fields are omitted entirely when ``fit_camera_rotation`` is False, so
2-DoF runs do not litter the JSON with null fields. The ``rank``
derivation only consults the translation sigma — ``max_sigma_px``
compares ``max(sigma_dv, sigma_du)`` — so a low rotation sigma can
never inflate a tier above what the translation accuracy supports.

at_edge for rotation
====================

The ``at_edge`` flag fires when the converged rotation magnitude
crosses ``rotation_at_edge_fraction * max_rotation_deg``, where
``rotation_at_edge_fraction`` is read from each technique's tuning
block in ``config_510_techniques.yaml`` (default ``0.95``;
:data:`~spindoctor.nav_technique.nav_technique.ROTATION_AT_EDGE_FRACTION`
is the canonical value that ships in the YAML). This is a separate
condition from translation ``at_edge`` (which checks proximity to
the search-window margin); both are OR-ed together onto the
:class:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.at_edge`
field. A separate INFO log line surfaces the rotation magnitude and
its sigma whenever ``fit_camera_rotation`` is on, with an explicit
``AT_EDGE`` annotation when the rotation cap is the trigger.
