==========================================================
Star Navigation Model
==========================================================

Overview
========

:class:`~spindoctor.nav_model.stars.nav_model_stars.NavModelStars` is the catalog-driven star
navigation model. For each observation the model reduces the configured catalog set into a
deduplicated star list, gates each star by catalog magnitude against the per-observation
limiting magnitude :meth:`obs.star_max_usable_vmag() <spindoctor.obs.obs_inst.ObsInst.star_max_usable_vmag>`,
flags stars whose predicted positions overlap a body silhouette or
ring annulus, and emits one
:data:`~spindoctor.feature.feature_type.NavFeatureType.STAR` :class:`~spindoctor.feature.feature.NavFeature`
per star that is at least as bright as the limiting magnitude and whose conflict flags
allow it.

The orchestrator builds exactly one instance of this model per observation regardless of
scene content; star navigation is the universal fallback when other navigation paths fail.

Theory
======

The star model orchestrates four cooperating steps: catalog reduction, conflict marking,
the magnitude detectability gate, and per-star feature emission.

Catalog reduction
-----------------

Multiple catalogs are merged into a single deduplicated star list, processed in priority
order (most-precise first). When two stars from different catalogs match within a
configurable angular tolerance and visual-magnitude tolerance, the higher-priority catalog's
entry survives. Pairs of stars whose magnitudes are too close together to disambiguate
visually are dropped from both catalogs (the autonomous match would be unable to attribute a
detection to one or the other).

Pixel datum
-----------

A star record states its position in ``oops`` uv: ``u`` and ``v`` are what
``ObsSnapshot.uv_from_ra_and_dec`` returns, and
``oops`` indexes a pixel by its corner, so the centre of pixel index ``i`` is at uv
``i + 0.5``. Every other predicted position in the pipeline -- body centres, limb and
terminator vertices, ring edges -- is a pixel index, and so is every centroid a star
technique measures, because each one builds its coordinate array with ``np.arange``
over the slice it indexes the image with.

The half pixel between the two is
:data:`~spindoctor.support.types.STAR_UV_DATUM_PX`. It comes off wherever a record's
position is used to index an array:
``NavModelStars._extfov_indices``
before a predicted position reaches a feature, and the Titan contaminant mask before a
star disc is painted. It stays on where the consumer wants uv: the conflict-check
meshgrid, which ``oops`` builds from FOV uv, and the smeared-PSF stamp, whose
``eval_rect`` offset is measured from the pixel's lower edge. A simulated scene states
a star's position as the pixel index it is drawn on, and the record builder shared by
the renderer and the simulated star model adds the datum at that boundary.

Bright-end saturation correction
--------------------------------

UCAC4 aperture V-magnitudes saturate at the bright end: for a star brighter than roughly
its documented reliable range the reported magnitude reads systematically too faint, by a
few tenths of a magnitude near :math:`V \approx 8` and climbing to several magnitudes by
:math:`V \approx 3`. Because the detectability gate and the navigable-content screen both
read the catalog magnitude, an uncorrected field of two genuinely bright anchor stars plus
a faint remainder is mistaken for a scatter of comparably faint sources: the two dominant
anchors are dragged down into the faint population and the pipeline cannot tell they are
there. In the Pleiades, UCAC4 reports Eta Tau (true :math:`V \approx 2.9`) and 27 Tau (true
:math:`V \approx 3.6`) near :math:`V \approx 6.7`.

After the catalog merge, :func:`~spindoctor.nav_model.stars.saturation.correct_star_photometry`
cross-references the merged list against a trusted-photometry reference built from the full
in-field YBSC and Tycho-2 sets. YBSC (the Yale Bright Star Catalogue) carries real Johnson V
and B photometry but is complete only to about :math:`V \approx 6.5`. Tycho-2's star-mapper
photometry does not saturate at the bright end and is complete to about :math:`V \approx 11`,
so it reaches the :math:`V \approx 6.5` to :math:`8` stars YBSC misses. The reference is the
in-field union of the two, with YBSC preferred wherever it covers a star (it supplies a
self-consistent Johnson pair) and Tycho-2 supplying the rest. Only UCAC4 saturates: records
whose catalog appears in
:data:`~spindoctor.nav_model.stars.saturation.REFERENCE_CATALOGS` (``ybsc``, ``tycho2``)
are the reference and are never correction candidates, so a reference is never corrected
against itself.

For each candidate record brighter than
:data:`~spindoctor.nav_model.stars.saturation.UCAC4_SATURATION_VMAG_LIMIT` (8.0):

- If it positionally matches a reference star and disagrees with it by more than
  :data:`~spindoctor.nav_model.stars.saturation.SATURATION_CORRECTION_MIN_MAG` (0.5 mag),
  it adopts the reference magnitude and a recomputed ``dn`` while keeping its own astrometry,
  and is flagged ``photometry_corrected``. A match against YBSC propagates YBSC's Johnson
  pair; a match against Tycho-2 adopts Tycho-2's V and fakes the colour from the candidate's
  spectral class (this pipeline discards Tycho-2's own colour), setting ``johnson_mag_faked``.
- If it matches a reference star but already agrees to within the tolerance, it is not
  saturated and is left untouched and unflagged. This is what keeps the flag honest: a
  genuine :math:`V \approx 7` to :math:`8` star beyond YBSC completeness matches its Tycho-2
  reference, agrees with it, and is not mistaken for saturation.
- If it has no reference in either catalog -- now genuinely rare -- it is flagged
  ``photometry_saturated``, so downstream consumers treat its magnitude as an untrusted
  lower bound on brightness rather than a trustworthy reading.

Empirically, Tycho-2 agrees with YBSC to a few hundredths of a magnitude through the bright
regime (over the Pleiades and Hyades: Eta Tau :math:`V = 2.84` vs YBSC :math:`2.87`,
Theta-2 Tau :math:`3.39` vs :math:`3.40`, Theta-1 Tau :math:`3.84` vs :math:`3.84`) where
UCAC4 reads the same stars near :math:`V \approx 6.7`, so extending the reference to Tycho-2
leaves the accurate bright end untouched while filling in the stars YBSC cannot reach.

Correcting a saturated magnitude then exposes a duplicate the merge missed. The merge only
deduplicates two catalogs' records of one physical star when their magnitudes agree, so a
star that UCAC4 saturated by more than the duplicate-magnitude tolerance survived alongside
its accurate Tycho-2 or YBSC twin. Once corrected the two magnitudes agree, and a final pass
collapses the revealed duplicate, keeping the twin that never needed correcting: the same
saturation that corrupts UCAC4's photometry also displaces its astrometry, so the catalog
that read the star at its true magnitude also placed it most accurately.

**Magnitude-aware position match.** That astrometric displacement is what a fixed-radius
match misses: a badly saturated UCAC4 record can sit farther from its true-magnitude twin
than the base match radius, so both the reference match and the duplicate collapse widen the
radius in proportion to how much fainter the record reads --
:data:`~spindoctor.nav_model.stars.saturation.SATURATION_MATCH_WIDEN_PER_MAG` base radii per
magnitude of gap, capped at
:data:`~spindoctor.nav_model.stars.saturation.SATURATION_MATCH_MAX_WIDEN_RADII` base radii so
the widened match stays inside the arc-minute spacing of bright stars. A larger saturation
error implies a larger displacement, so the widening scales with the gap. Each reference star
is consumed by at most one corrected record, so one true-magnitude star cannot correct or
absorb two distinct saturated records.

Among the references that fall inside their own widened reach the match takes the brightest,
not the merely nearest, breaking a brightness tie on separation. A bright unrelated reference
earns a wide reach of its own from its large magnitude gap to the saturated candidate, so a
pure nearest-neighbour rule could let it capture the candidate when it happens to sit closer
than the true twin. Preferring the brightest qualifying reference blocks that: UCAC4
saturation drives a bright star's reading systematically faint, so a faint reading is best
explained by the brightest reference that can account for it. The failure mode this guards
needs two bright stars within the widened reach, which the sparse fields the pipeline
navigates do not present, so in practice the two rules agree; the brightness preference only
diverges on a crowded bright field.

**Detectability consumes the flag.** The detectability model exempts a ``photometry_saturated``
star from the faint-magnitude gate -- its recorded magnitude is a too-faint saturated
reading, so it must not be rejected on that reading -- and treats the star as at least as
bright as the limiting magnitude when computing its effective SNR, without fabricating a
specific brighter value. The flag is carried onto the emitted
:class:`~spindoctor.feature.flags.StarFlags` so downstream techniques and the star metadata
summary see it.

Conflict marking
----------------

Each predicted star is checked against:

- Body silhouettes — when a star's predicted position lies inside any body's predicted
  silhouette mask, the star is flagged ``in_body_silhouette``.
- Ring annuli — when a star's predicted position lies inside any planet's ring system
  (defined by the per-planet radial bounds in the YAML config), the star is flagged
  ``in_body_silhouette`` as well (a ring conflict occludes the star the same way).

The model does **not** gate a star on the saturation or cosmic-ray mask at its predicted
position: once a pointing offset is present the predicted position carries no special
significance, and a sharp stellar peak self-triggers the cosmic-ray detector, so consulting
the mask there would suppress good stars. The ``saturated`` and
``in_saturation_or_cosmic_mask`` flags therefore stay clear. Star usability is an
occlusion-only gate.

Conflict-flagged stars stay in the model's list (so the curator surfaces them in the
sidecar) but are excluded from the autonomous matching path by the upstream
``usable_stars`` filter consulted by every star technique.

Magnitude detectability gate
----------------------------

A star is detectable when its catalog visual magnitude is at least as bright as the
per-observation limiting magnitude :meth:`obs.star_max_usable_vmag()
<spindoctor.obs.obs_inst.ObsInst.star_max_usable_vmag>`. The gate is purely
magnitude-based: a star with no catalog magnitude, or with
:math:`V_{\mathrm{mag}} > m_{\mathrm{limit}}`, is skipped, and no per-pixel DN photometry
or DN-to-image-unit scaling enters the decision. The catalog reduction applies the same
magnitude cap, so the gate in the model is mostly a self-contained re-check that also
handles the missing-magnitude case.

Each per-instrument :meth:`star_max_usable_vmag()
<spindoctor.obs.obs_inst.ObsInst.star_max_usable_vmag>` follows the form

.. math::

    m_{\mathrm{limit}}(t_{\mathrm{exp}}) =
        \mathrm{anchor} + \frac{\log t_{\mathrm{exp}}}{\log 2.512},

where ``anchor`` is the limiting magnitude at a one-second exposure and each Pogson factor
of exposure time buys one more magnitude of depth.

Magnitude-margin effective SNR
------------------------------

The CRLB covariance and the reliability sigmoid still want an SNR-like quantity, so the
model synthesises one from how far below the limiting magnitude the star sits rather than
from any photometric DN measurement:

.. math::

    \mathrm{SNR}_{\mathrm{eff}} =
        \mathrm{SNR}_{\mathrm{REF}} \cdot 2.512^{\,(m_{\mathrm{limit}} - V_{\mathrm{mag}})},

floored at the module constant ``SNR_FLOOR`` so it stays strictly positive. A star exactly
at the limit gets ``SNR_REF`` (``8.0``, just below the reliability sigmoid centre); each
magnitude of headroom multiplies the effective SNR by one Pogson ratio (``2.512``),
matching the flux ratio per magnitude. ``SNR_FLOOR`` (``0.1``) keeps the covariance away
from the zero-SNR huge-variance branch.

When the per-image smear length exceeds the anisotropic-smear threshold the PSF is
replaced by a smear-aware kernel built from the per-image smear vector via
:func:`~spindoctor.nav_model.stars.smeared_psf.compute_smear_vector_px`; the kernel has elongated
support, and the smear vector also rotates the CRLB covariance along the smear direction.
Stars whose smear length exceeds the per-instrument ``max_smear`` cap are dropped from the
emission set, as are stars fainter than the limiting magnitude.

:mod:`spindoctor.nav_model.stars.predicted_snr` retains a separate raw-DN ``predicted_snr``
photometry helper for diagnostics; it is not the gate consulted by the navigator's star
path.

Per-feature uncertainty
-----------------------

Each emitted ``STAR`` feature carries a Cramer-Rao lower bound covariance derived from the
magnitude-margin effective SNR and the per-instrument PSF. For a 2-D Gaussian PSF the CRLB
on each position component reduces to

.. math::

    \sigma_{\mathrm{pos}} = \frac{\sigma_{\mathrm{PSF}}}{\mathrm{SNR}},

so stars well above the limiting magnitude have sub-pixel position sigma and stars near
the limit have several-pixel sigma. The per-axis variances populate the diagonal of the
:attr:`~spindoctor.feature.feature.NavFeature.position_cov_px` covariance; the off-diagonal is
zero except when the smear-aware kernel is anisotropic, in which case the cross-axis
correlation reflects the smear orientation.

Restrictions and assumptions
----------------------------

- The detectability gate depends on the per-instrument limiting-magnitude model. When
  :meth:`star_max_usable_vmag() <spindoctor.obs.obs_inst.ObsInst.star_max_usable_vmag>`
  is miscalibrated for an instrument or exposure, the emission gate may include or exclude
  the wrong stars.
- Catalog reduction assumes the per-catalog sky-coverage epoch is recent enough that proper
  motion brings the star to within the duplicate-detection tolerance of its true epoch
  position. Stars with very high proper motion may move out of catalog match between
  catalogs.
- Ring-occlusion checks consult the per-planet radial bounds in the YAML; non-listed
  planets contribute no ring-occlusion flag.

Sources of uncertainty
----------------------

The per-star CRLB covariance reflects the magnitude-margin centroid uncertainty. It
does not capture systematic biases from a misaligned camera distortion model, from
unmodelled background flux, or from PSF smear that exceeds the per-instrument
``max_smear`` cap (stars above that cap are dropped from the emission set rather than
emitted with unreliable predictions). Star usability is gated only on body / ring occlusion;
the saturation and cosmic-ray masks are not consulted at the predicted position, so the
:class:`~spindoctor.feature.flags.StarFlags` saturation flags stay clear.

Configuration
=============

The model's runtime knobs live in ``stars`` in
``src/spindoctor/config_files/config_030_stars.yaml``.

- ``catalogs`` — list[str], default ``[ucac4, tycho2, ybsc]``. Catalog priority order
  (most-precise first). Catalog files are loaded from the configured catalog roots. YBSC and
  Tycho-2 also serve as the photometric reference for the bright-end saturation correction
  regardless of their list position; dropping both from the list disables that correction.
- ``body_conflict_margin`` — int, default ``5`` px. Margin around body silhouettes for
  the body-conflict flag.
- ``default_star_class`` — str, default ``'G0'``. Spectral class assigned to catalog stars
  with no per-entry class.
- ``stellar_aberration`` — bool, default ``true``. Apply stellar aberration to predicted
  positions.
- ``proper_motion`` — bool, default ``true``. Apply proper motion to predicted positions.
- ``max_stars`` — int, default ``100``. Maximum number of predicted stars retained per
  observation.
- ``max_movement_steps`` — int, default ``50``. Maximum sub-exposure steps used by the
  smear-vector computation.
- ``label_font`` — str. Font used for star labels in the summary PNG.
- ``label_font_size`` — int, default ``18`` px. Label font size.
- ``label_font_color`` — list[int], default ``[255, 0, 0]`` (RGB). Label font color.
- ``label_star_color`` — list[int], default ``[255, 0, 0]`` (RGB). Color of the star marker
  drawn on the summary PNG.
- ``duplicate_ra_dec_threshold_arcsec`` — float, default ``5`` arcsec. Tolerance for the
  per-catalog duplicate detection step.
- ``duplicate_vmag_threshold`` — float, default ``3`` mag. Magnitude tolerance for the
  duplicate detection step.
- ``overlapping_vmag_threshold`` — float, default ``2`` mag. Below this magnitude
  difference, two visually-overlapping stars are dropped from both catalogs.
- ``calibrated_data`` — bool, default ``true``. Whether the per-image data is in
  calibrated I/F units (vs. raw DN).
- ``float_psf_sigma`` — bool, default ``false``. When true, the per-instrument PSF sigma is
  treated as a fit parameter; when false, the per-instrument value is used verbatim.
- ``search_multipliers`` — list[float], default ``[0.25, 0.5, 0.75, 1.0]``. Multipliers on
  the per-instrument SPICE pointing-error envelope used by the star matcher's coarse
  search.
- ``perform_photometry`` — bool, default ``true``. Whether to run per-star photometric
  validation.
- ``try_without_photometry`` — bool, default ``false``. Whether to attempt a fallback
  match path with photometry disabled.
- ``min_stars_low_confidence`` — list (count, confidence), default ``[3, 0.75]``. Minimum
  star count and confidence level for the low-confidence match path.
- ``min_stars_high_confidence`` — list (count, confidence), default ``[6, 1.0]``. Minimum
  star count and confidence level for the high-confidence match path.
- ``min_confidence`` — float, default ``0.9``. Minimum confidence level for the match to
  succeed.
- ``psf_gain`` — list (DN, gain), default ``[5000, 4]``. PSF integrated gain mapping for
  flux estimation.
- ``max_smear`` — float, default ``100`` (dimensionless). Maximum smear length above which
  a star is dropped from the emission set.
- ``min_vmag`` — float, default ``5.0`` mag. Minimum visual magnitude (i.e. brightest)
  considered.
- ``max_vmag`` — float, default ``15.0`` mag. Maximum visual magnitude (i.e. dimmest)
  considered for prediction.
- ``vmag_increment`` — float, default ``0.5`` mag. Magnitude bin width for the per-bin
  star-list build.
- ``max_star_dn`` — float, default ``100000.0`` DN. Above this, stars are too bright to
  use (saturation regime).
- ``min_dn_force_one_star`` — float, default ``25000.0`` DN. Below this DN, the
  unique-bright single-star path will not fire even on a uniquely bright catalog star.
- ``star_body_conflict_margin`` — int, default ``3`` px. Smaller-than-body conflict margin
  used when the per-star centroid is close to the body silhouette boundary.
- ``too_bright_dn`` — float, default ``1000`` DN. Threshold above which a star is "very
  bright" for the per-star photometric tests.
- ``too_bright_factor`` — float, default ``1`` (dimensionless). Multiplier on
  ``too_bright_dn`` (reserved tuning slot).
- ``ring_occlusion_enabled`` — bool, default ``true``. Whether to flag stars whose
  predicted positions lie inside a planet's ring system.
- ``ring_occlusion_min_opaque_fraction`` — float, default ``0.25``. Fraction of the
  star's conflict-window pixels that must fall inside an opaque annulus before the
  star is flagged ring-occluded. Membership is tested per pixel, so a star whose
  window straddles a ringlet edge or a gap boundary is judged by how much of the
  window is actually over opaque material.
- ``ring_occlusion_radii_km`` — dict[str, list[list[float]]]. Per-planet ring annular
  bounds (km) used by the occlusion check. Saturn entry covers C, B, and A rings; Uranus
  covers the main ring system; Neptune covers Galle through Adams rings.

Module-level emission constants
-------------------------------

The detection-side thresholds and the catalog magnitude-bin grid are Python module-level
constants in the :mod:`spindoctor.nav_model.stars` subpackage and are not exposed as YAML knobs.
Tests and downstream tools read the canonical values via these symbols.

- :data:`~spindoctor.nav_model.stars.detection.DAOPHOT_DEFAULT_DETECTION_SIGMA` — float,
  ``4.0`` (in ``image_noise_sigma``). Threshold for matched-filter peaks. Below 4 sigma
  the cosmic-ray-driven false-positive rate dominates real detections; the autonomous
  detector rejects peaks below this level.
- :data:`~spindoctor.nav_model.stars.detection.DAOPHOT_DEFAULT_SHARPNESS_MIN` — float, ``0.2``
  (dimensionless). Minimum DAOPHOT sharpness for a real star. Sharpness below 0.2 is
  dominated by single-pixel hot spikes whose wing contribution is too small to be a
  star.
- :data:`~spindoctor.nav_model.stars.detection.DAOPHOT_DEFAULT_SHARPNESS_MAX` — float, ``1.0``
  (dimensionless). Maximum DAOPHOT sharpness for a real star. Sharpness above 1.0
  indicates an extended source (galaxy / blended pair) whose central pixel does not
  dominate.
- :data:`~spindoctor.nav_model.stars.detection.DAOPHOT_DEFAULT_ROUNDNESS_BOUND` — float,
  ``1.0`` (dimensionless). Maximum :math:`|\mathrm{roundness}|` for a real star,
  computed as the per-axis Gaussian-marginal asymmetry; values outside
  :math:`[-1, 1]` point at a CCD bloom or one-axis trail rather than a smear-oriented
  PSF.
- :data:`~spindoctor.nav_model.stars.catalog.CATALOG_MAGNITUDE_BINS` — tuple[float, ...]. The
  coarse magnitude grid the catalog reducer pulls stars by. Stars are pulled bin by bin
  until the configured ``max_stars`` budget is hit, which avoids the worst case of
  pulling every dim star in a degree-square chunk of UCAC4.
- :data:`~spindoctor.nav_model.stars.saturation.UCAC4_SATURATION_VMAG_LIMIT` — float, ``8.0``.
  V-magnitude brighter than which UCAC4 aperture photometry is untrusted and a candidate
  record becomes a saturation-correction candidate. Chosen from the UCAC4-vs-YBSC residual
  and UCAC4's documented reliable aperture range (roughly V9 to V14).
- :data:`~spindoctor.nav_model.stars.saturation.SATURATION_CORRECTION_MIN_MAG` — float,
  ``0.5``. Smallest candidate-versus-reference magnitude gap treated as real saturation; a
  smaller gap means the record already agrees with the reference and is left untouched. This
  no-op band both preserves already-accurate bright-star photometry and leaves a genuine
  :math:`V \approx 7` to :math:`8` star that agrees with its Tycho-2 reference unflagged.
- :data:`~spindoctor.nav_model.stars.saturation.SATURATION_MATCH_WIDEN_PER_MAG` — float,
  ``0.5``. Base radii of extra position-match reach per magnitude of saturation gap, so a
  displaced saturated record still matches its true-magnitude reference.
- :data:`~spindoctor.nav_model.stars.saturation.SATURATION_MATCH_MAX_WIDEN_RADII` — float,
  ``3.0``. Cap on the widened match radius, in base radii, so the match never reaches an
  unrelated field star.

Per-instrument overrides
------------------------

Per-instrument YAML files in ``src/spindoctor/config_files/config_4N0_inst_*.yaml`` may override
catalog selection or photometric parameters; see the per-instrument source for the full
list.

Implementation
==============

Source files:

- ``src/spindoctor/nav_model/stars/nav_model_stars.py`` —
  :class:`~spindoctor.nav_model.stars.nav_model_stars.NavModelStars` plus the per-star
  CRLB-covariance helper, the smear-resolver, and the per-summary-PNG annotation
  builder.
- ``src/spindoctor/nav_model/stars/catalog.py`` —
  :mod:`spindoctor.nav_model.stars.catalog` multi-catalog reduction helpers and
  :data:`~spindoctor.nav_model.stars.catalog.CATALOG_MAGNITUDE_BINS`.
- ``src/spindoctor/nav_model/stars/saturation.py`` —
  :mod:`spindoctor.nav_model.stars.saturation` UCAC4 bright-end saturation correction against
  a YBSC and Tycho-2 reference, exposed as
  :func:`~spindoctor.nav_model.stars.saturation.correct_star_photometry`
  (the in-place record pass used by the reduction) and
  :func:`~spindoctor.nav_model.stars.saturation.correct_saturated_vmags` (the plain-magnitude
  helper the navigable-content screen shares).
- ``src/spindoctor/nav_model/stars/conflicts.py`` —
  :mod:`spindoctor.nav_model.stars.conflicts` body / ring conflict marking.
- ``src/spindoctor/nav_model/stars/predicted_snr.py`` —
  :func:`~spindoctor.nav_model.stars.predicted_snr.psf_sigma_px` (used by the model and the
  detection helpers) and the ``SCLASS_TO_B_MINUS_V`` spectral-class-to-colour table.
  :func:`~spindoctor.nav_model.stars.predicted_snr.predicted_snr` is a raw-DN photometry
  diagnostic and is not the model's detectability gate.
- ``src/spindoctor/nav_model/stars/smeared_psf.py`` —
  :func:`~spindoctor.nav_model.stars.smeared_psf.compute_smear_vector_px` and
  :func:`~spindoctor.nav_model.stars.smeared_psf.smear_length_px`.
- ``src/spindoctor/nav_model/stars/detection.py`` —
  :mod:`spindoctor.nav_model.stars.detection` matched-filter source detection plus the four
  :data:`~spindoctor.nav_model.stars.detection.DAOPHOT_DEFAULT_DETECTION_SIGMA`,
  :data:`~spindoctor.nav_model.stars.detection.DAOPHOT_DEFAULT_SHARPNESS_MIN`,
  :data:`~spindoctor.nav_model.stars.detection.DAOPHOT_DEFAULT_SHARPNESS_MAX`, and
  :data:`~spindoctor.nav_model.stars.detection.DAOPHOT_DEFAULT_ROUNDNESS_BOUND` constants
  documented above. Consumed by the star techniques
  rather than by the model itself; documented here because its constants set the
  detection floor every star technique applies.

Public class :class:`~spindoctor.nav_model.stars.nav_model_stars.NavModelStars`, base
:class:`~spindoctor.nav_model.nav_model.NavModel`. Self-registers via ``__init_subclass__``.

Public methods (autodocumented at :doc:`/api_reference/api_nav_model`):

- :meth:`~spindoctor.nav_model.stars.nav_model_stars.NavModelStars.instances_for_obs` — always
  returns ``[NavModelStars('stars', obs)]`` (one instance per observation).
- :meth:`~spindoctor.nav_model.stars.nav_model_stars.NavModelStars.create_model` — runs catalog
  reduction, computes the smear vector, marks conflicts, applies the magnitude
  detectability gate, and populates the per-star CRLB covariance.
- :meth:`~spindoctor.nav_model.stars.nav_model_stars.NavModelStars.to_features` — emits one
  :data:`~spindoctor.feature.feature_type.NavFeatureType.STAR` feature per surviving catalog star.
- :meth:`~spindoctor.nav_model.stars.nav_model_stars.NavModelStars.to_annotations` — emits
  per-star markers and labels for the summary PNG.

Inherited read-only properties on :class:`~spindoctor.nav_model.nav_model.NavModel`:
:attr:`~spindoctor.nav_model.nav_model.NavModel.name`,
:attr:`~spindoctor.nav_model.nav_model.NavModel.obs`,
:attr:`~spindoctor.nav_model.nav_model.NavModel.metadata`.

Annotation helpers
------------------

:class:`~spindoctor.nav_model.stars.nav_model_stars.NavModelStars` derives directly from
:class:`~spindoctor.nav_model.nav_model.NavModel` and carries its own annotation helpers
(unlike the body and ring families, which share helpers via an abstract base).

- ``_build_annotations`` — builds the per-star
  :class:`~spindoctor.annotation.annotations.Annotations` collection: a rectangle outline at
  every predicted star position sized by the star's ``psf_size`` (assigned
  from :meth:`~spindoctor.obs.obs_inst.ObsInst.star_psf_size`), plus per-star labels
  carrying the
  catalog name and visual magnitude. Stars flagged with a body / ring conflict are
  skipped (they are surfaced in the per-image metadata for reviewer awareness but not
  drawn). Consumes the ``label_*`` and ``label_star_color`` keys documented above.
- ``_extfov_indices`` — converts a star's recorded oops uv to an extfov-frame pixel
  index (see `Pixel datum`_), for the rectangle drawer and for the emitted feature.
- The per-star label string is built by the module-level ``_star_label`` helper, which
  picks one of the four arrow directions
  (:data:`~spindoctor.annotation.annotation_text_info.TEXTINFO_TOP_ARROW` /
  :data:`~spindoctor.annotation.annotation_text_info.TEXTINFO_BOTTOM_ARROW` /
  :data:`~spindoctor.annotation.annotation_text_info.TEXTINFO_LEFT_ARROW` /
  :data:`~spindoctor.annotation.annotation_text_info.TEXTINFO_RIGHT_ARROW`) based on the
  star's distance from the frame edge.

Per-image metadata
------------------

:meth:`~spindoctor.nav_model.stars.nav_model_stars.NavModelStars.create_model` populates
:attr:`~spindoctor.nav_model.nav_model.NavModel.metadata` with the following entries for the
curator to surface in the per-image JSON sidecar:

- ``start_time`` / ``end_time`` / ``elapsed_time_sec`` — wall-clock timing for the model
  build.
- ``star_count`` — int, number of stars that survived catalog reduction, conflict
  marking, and the magnitude detectability gate.
- ``stars`` — list[dict], one entry per surviving star. Each entry carries
  ``catalog_name``, ``unique_number``, ``pretty_name``, ``ra_deg``, ``dec_deg``,
  ``vmag``, ``photometry_corrected``, ``photometry_saturated``, ``u``, ``v``, ``move_u``,
  ``move_v``, ``spectral_class``, and ``conflicts`` (the comma-separated body- /
  ring-occlusion flag string built from the per-star conflict marking step). ``u`` and
  ``v`` are the pixel index the star lands on in the nominal (unpadded) frame, the
  same convention every other recorded position uses; the matching STAR feature's
  ``predicted_vu`` is that index plus the extended-FOV margin, and the record the
  entry is built from holds the same point as oops uv, half a pixel higher on each
  axis (see `Pixel datum`_). The two
  ``photometry_*`` booleans are the bright-end saturation provenance:
  ``photometry_corrected`` marks a record whose magnitude was replaced by a YBSC or Tycho-2
  reference value, and ``photometry_saturated`` marks a bright record with no reference in
  either catalog whose magnitude is an untrusted lower bound.

Conflict-flagged stars stay in the metadata so a reviewer can see which stars were
predicted but excluded; the upstream ``usable_stars`` filter consulted by every star
technique drops them from the autonomous matching path.

Call path
---------

Call path traced through
:meth:`~spindoctor.nav_model.stars.nav_model_stars.NavModelStars.create_model`:

1. Open a logged section. Read the per-image observation epoch and configured catalog
   list.
2. Compute the per-image motion smear vector via
   :func:`~spindoctor.nav_model.stars.smeared_psf.compute_smear_vector_px`. Stars whose smear
   length exceeds ``max_smear`` are dropped from the emission set at feature-build time.
3. Reduce the configured catalogs into a deduplicated star list via
   :func:`~spindoctor.nav_model.stars.catalog.reduce_catalogs`. The function returns a list of
   mutable star records carrying RA/Dec, magnitude, spectral class, parallax, proper
   motion, and per-catalog provenance.
4. Mark body / ring conflicts on each star via
   :func:`~spindoctor.nav_model.stars.conflicts.mark_body_and_ring_conflicts`. The function
   consults the per-image inventory and the YAML's ``ring_occlusion_radii_km`` and
   records each star's conflict; at feature-emission time both body and ring conflicts
   set the ``in_body_silhouette`` flag (a ring conflict occludes the star the same way).
5. Resolve the per-observation limiting magnitude from :meth:`obs.star_max_usable_vmag()
   <spindoctor.obs.obs_inst.ObsInst.star_max_usable_vmag>` and synthesise the
   magnitude-margin effective SNR for each star (used by the CRLB covariance and the
   reliability sigmoid).
6. Drop stars fainter than the limiting magnitude (or with no catalog magnitude) and
   stars whose smear length exceeds the per-instrument cap; record the survivors on
   ``self._stars``.

Call path traced through
:meth:`~spindoctor.nav_model.stars.nav_model_stars.NavModelStars.to_features`:

1. For each surviving star, build a :class:`~spindoctor.feature.geometry.StarGeometry` from the
   predicted position and the per-feature CRLB covariance.
2. Build a :class:`~spindoctor.feature.flags.StarFlags` carrying the magnitude-margin effective
   SNR, the catalog magnitude, and the body / ring conflict flags. The saturation /
   cosmic-ray-mask flags are left clear -- star usability is an occlusion-only gate.
3. Construct one :data:`~spindoctor.feature.feature_type.NavFeatureType.STAR`
   :class:`~spindoctor.feature.feature.NavFeature` per star and return the list.

Examples
========

``one_bright_star_no_body`` (Cassini ISS WAC, image ``W1449079117_1``)
    Single bright star (Vega) in an otherwise empty FOV. The star model emits one STAR
    feature for Vega, which is far brighter than the limiting magnitude and carries no
    body / ring conflict flags. The
    pass-1 :class:`~spindoctor.nav_technique.nav_technique_star_unique_match.StarUniqueMatchNav`
    consumes the feature in its 1-star path and reports the operator-verified offset
    :math:`(\Delta v, \Delta u) = (3.06, -0.02)` px. Other catalog stars in the WAC's
    extended FOV are below the magnitude floor and produce no STAR feature.

``star_dominated`` (Cassini ISS WAC, image ``W1580760393_1``)
    Dense star field with no body in FOV. The star model emits dozens of STAR features
    above the magnitude floor. The pass-1
    :class:`~spindoctor.nav_technique.nav_technique_star_field.StarFieldFromCatalogNav` consumes
    the cohort and reports the operator-verified offset
    :math:`(\Delta v, \Delta u) = (-2.68, -3.68)` px via the triplet-hash matcher.

``below_resolution_body`` (Cassini ISS NAC, image ``N1777325846_1``)
    Mimas in the lower left at ~20 px diameter at phase 72 degrees. The star model emits
    STAR features for catalog stars in the FOV; stars whose predicted positions fall
    inside the predicted Mimas silhouette are flagged ``in_body_silhouette`` and the
    upstream ``usable_stars`` filter drops them so
    the star techniques do not consume them.
