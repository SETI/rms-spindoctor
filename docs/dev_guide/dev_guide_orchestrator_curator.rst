==========================================================
JSON Curation (build_metadata_dict)
==========================================================

Overview
========

The curator turns a :class:`~spindoctor.nav_orchestrator.nav_result.NavResult` into a JSON-friendly
metadata dict consumed by downstream readers. The complete key-by-key specification of
the file this block is written into, with annotated examples of every document shape,
is the user guide's :doc:`/user_guide/user_guide_metadata` chapter; this page covers the
conversion mechanism. Two functions form the public surface:
:func:`~spindoctor.nav_orchestrator.curator.build_metadata_dict` does the conversion, and
:func:`~spindoctor.nav_orchestrator.curator.assert_diagnostic_fields_present` runs at startup to
enforce the per-technique ``CURATOR_FIELDS`` allow-list discipline so a new diagnostic
field cannot silently disappear from the JSON output.

Theory
======

The curator picks JSON-friendly fields from a
:class:`~spindoctor.nav_orchestrator.nav_result.NavResult`, rounds floats to documented
precision, substitutes the ``JSON_INF_SENTINEL`` finite sentinel for non-finite floats
(including NaN), and emits the ``navigation_result`` block consumed by downstream readers.

Float rounding policy
---------------------

Three precision constants govern the rounding:

- ``PIXEL_DECIMALS = 4`` — pixel-domain quantities (offsets, sigmas, covariance
  entries).
- ``CONFIDENCE_DECIMALS = 3`` — confidence scores in :math:`[0, 1]`.
- ``ET_DECIMALS = 6`` — ET timestamps (seconds past J2000 TDB).

The constants are chosen tighter than the per-image tolerance budget so the JSON output
is byte-identical across runs of the same input — a regression-baseline comparator can
diff the JSON directly.

Corrected pointing
------------------

When the orchestrator could determine the observation's attitude, the
:class:`~spindoctor.nav_orchestrator.nav_result.NavResult` carries a
:class:`~spindoctor.support.cmatrix.PointingSolution` and the curator emits two further
blocks under ``navigation_result``:

- ``pointing`` — ``cmatrix_original`` (the uncorrected J2000-to-camera rotation the
  furnished kernels gave at navigation time), ``cmatrix`` (the same rotation corrected
  by the navigated offset), ``camera_frame``, ``camera_frame_id`` and ``ck_frame_id``.
  Both matrices are nine row-major floats in the SPICE camera-frame convention at the
  exposure midtime. ``cmatrix`` is present only when the navigation produced an offset
  and fitted no camera rotation.
- ``times`` — ``start_et``, ``stop_et``, ``midtime_et``, ``exposure_s`` and the three
  spacecraft-clock strings ``sclk_start``, ``sclk_midtime`` and ``sclk_stop``.

Both blocks are written unrounded, against the policy above. A consumer identifies the
kernel an image navigated against by reproducing ``cmatrix_original`` to within a
nanoradian and defines a segment interval from the exact exposure epochs, and rounding
would put the recorded values outside those bounds.

The observation block
---------------------

:func:`~spindoctor.navigate_image_files.build_metadata_from_result` wraps this block into
the per-image document beside an ``observation`` block. That block holds the image's
identity -- its path, name, registered instrument, camera, shutter mode and image shape
-- followed by everything the observation's instrument host publishes through
:meth:`~spindoctor.obs.obs_inst.ObsInst.get_public_metadata`: the exposure's start,
midtime and end in UTC and ET, the label's spacecraft clock counts, the exposure time,
the filters, the PDS4 context identifiers, and each host's own descriptive facts. Each
spacecraft host parses its own clock's label format, converts a count to an exact
fraction of its clock's leading unit with the clock's moduli and offsets through
:func:`~spindoctor.support.sclk.fractional_count`, and publishes the label's start and
stop counts through :func:`~spindoctor.support.sclk.exposure_counts`, which writes each
count, and the mean of the two, as the float nearest its exact value. A label's counts
mark different moments
on different instruments, so each host says whether its two counts bracket the exposure.
Cassini ISS's and New Horizons LORRI's do, and their midtime count is the exact mean of
the two. Voyager ISS's stop count is that of the frame the image was read out in, and a
Galileo SSI label has no stop count, so their midtime count is None. A count the label
does not carry is None. A published key the block already holds keeps the block's value,
and ``image_shape_xy`` is left out because it is ``image_shape`` in the other axis order.
Both drivers that write a navigated document -- the autonomous pipeline and the
``sd_offset --manual`` pass -- supply the published facts, so they are recorded for every
image whose navigation ran to a result, successful or failed, whether or not a
``pointing`` block was recorded. A load-error or internal-error document carries none of
them. They are copied as the host states them, unrounded.

Allow-list discipline
---------------------

Every per-technique diagnostic field that ships in the JSON appears in the technique's
``CURATOR_FIELDS`` class attribute (a mapping of dataclass-field name to JSON-key name, or
``None`` to skip). The curator walks ``CURATOR_FIELDS`` rather than the dataclass's
``fields()`` directly, so a new field added to a diagnostics dataclass without an entry
in the mapping does not silently leak into the JSON.
:func:`~spindoctor.nav_orchestrator.curator.assert_diagnostic_fields_present` runs at startup
(or in CI) and fails the build with :exc:`AssertionError` when any dataclass field is
missing from its ``CURATOR_FIELDS``.

Restrictions and assumptions
----------------------------

- The curator does not handle nested dataclasses generically. Per-technique diagnostic
  classes are flat (every public field is a Python primitive or numpy scalar); when a
  future diagnostic dataclass needs nested structure the curator will need a recursive
  variant.
- Non-finite floats (``+inf``, ``-inf``, ``nan``) are mapped: ``+inf`` becomes
  ``JSON_INF_SENTINEL``, ``-inf`` becomes ``-JSON_INF_SENTINEL``, and ``nan`` becomes the
  *positive* ``JSON_INF_SENTINEL``. The NaN mapping is deliberate: the only path that
  produces a NaN here is a degenerate covariance/variance entry, and rendering that as
  ``0.0`` would read as a zero-variance, infinitely confident value — the opposite of
  the truth. The huge sentinel instead conveys unbounded uncertainty.
  The sentinel is a documented finite value the JSON schema reserves for "unbounded".
- The curator does not include the per-image image array in the JSON (it would balloon
  the sidecar to multi-megabyte sizes). An external image-export step writes the image
  alongside the JSON when needed.

Sources of uncertainty
----------------------

The curator reports no uncertainty. Every output is a deterministic projection of the
input :class:`~spindoctor.nav_orchestrator.nav_result.NavResult`.

Configuration
=============

The curator carries no YAML configuration of its own. The three rounding constants
(``PIXEL_DECIMALS``, ``CONFIDENCE_DECIMALS``, ``ET_DECIMALS``) and the
``JSON_INF_SENTINEL`` live as module-level constants; ``JSON_INF_SENTINEL`` lives in
:mod:`spindoctor.feature.constants` and is shared with other JSON producers.

Implementation
==============

Source file: ``src/spindoctor/nav_orchestrator/curator.py`` —
:func:`~spindoctor.nav_orchestrator.curator.build_metadata_dict`,
:func:`~spindoctor.nav_orchestrator.curator.assert_diagnostic_fields_present`, plus the private
``_round_float`` / ``_round_pair`` / ``_round_matrix`` rounding helpers.

Public surface (autodocumented at :doc:`/api_reference/api_nav_orchestrator`):

- :func:`~spindoctor.nav_orchestrator.curator.build_metadata_dict` — turns a
  :class:`~spindoctor.nav_orchestrator.nav_result.NavResult` into a JSON-friendly nested dict.
  Public entry point for the per-image-sidecar writer.
- :func:`~spindoctor.nav_orchestrator.curator.assert_diagnostic_fields_present` — verifies every
  per-technique diagnostic field has a ``CURATOR_FIELDS`` entry. Run at config-load
  time; raises :exc:`AssertionError` on the first unmapped field.

Per-technique diagnostics dataclasses (documented at
:doc:`dev_guide_techniques_diagnostics`) declare their own ``CURATOR_FIELDS`` class
attributes; the curator picks fields from each via the dataclass's
``CURATOR_FIELDS`` rather than from
:func:`dataclasses.fields` directly.

Examples
========

**Per-image JSON sidecar shape.**  After a successful
:class:`~spindoctor.nav_technique.nav_technique_body_limb.BodyLimbNav` fit on
``body_partial_overflow``, the curator emits::

    {
      "navigation_result": {
        "status": "success",
        "offset_px": [11.06, 30.53],
        "sigma_px": [2.613, 2.6128],
        "confidence_rank": "low",
        "confidence": 0.675,
        "confidence_provisional": true,
        "status_reason": "ok",
        "covariance_px2": [[6.8277, 0.0017], [0.0017, 6.8269]],
        "per_technique": [
          {
            "technique_name": "BodyLimbNav",
            "feature_ids": ["limb_arc:RHEA"],
            "offset_px": [12.06, 30.53],
            "covariance_px2": [[6.8277, 0.0017], [0.0017, 6.8269]],
            "confidence": 0.675,
            "spurious": false,
            "at_edge": false,
            "diagnostics": {
              "visible_limb_arc_fraction": 0.85,
              "visible_arc_px": 120.0,
              "dt_fit_rms_px": 0.4,
              "lm_iterations": 5,
              "tukey_inlier_count": 118
            }
          }
        ],
        ...
      }
    }

Every per-technique diagnostic key under ``"diagnostics"`` corresponds to a non-``None``
entry in the diagnostics dataclass's ``CURATOR_FIELDS``.

The literal ``"confidence_provisional": true`` marker flags that the confidence values
and ``confidence_rank`` tiers are calibrated against simulated planted-truth recovery
only (sim-anchored) and must not be read as probabilities of real-image accuracy; the
curator emits it unconditionally.  A developer who recalibrates against real-image
error measurements should retire the marker at that point.

**Allow-list catches a missed field.**  An operator adds a new field
``mean_polarity_score`` to
:class:`~spindoctor.nav_technique.diagnostics.BodyLimbDiagnostics` without updating
``CURATOR_FIELDS``. At startup
:func:`~spindoctor.nav_orchestrator.curator.assert_diagnostic_fields_present` runs over the
:class:`~spindoctor.nav_orchestrator.nav_result.NavResult` returned by the smoke test, walks the
diagnostic's :func:`dataclasses.fields`, and raises::

    AssertionError: BodyLimbDiagnostics has unmapped fields ['mean_polarity_score'];
    add them to CURATOR_FIELDS or set value to None to skip

The build fails before the new field can silently disappear from the JSON sidecar.

**Non-finite handling.**  A pathological technique reports ``rotation_rad = +inf`` (a
genuine cost-collapse case the LM refiner mapped to the rotation-unobservable sentinel).
The curator emits ``JSON_INF_SENTINEL`` (``1.0e9``) in the JSON instead of ``+inf``,
keeping the file JSON-spec-compliant; downstream readers treat the sentinel as
"unbounded / no information". A NaN (a degenerate covariance entry) maps to the same
positive sentinel, so it also reads as unbounded uncertainty rather than a spuriously
confident ``0.0``.
