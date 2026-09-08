==========================
Config and Static Data
==========================

Every SpinDoctor subsystem reads its tunables from a single
:class:`~spindoctor.config.config.Config` object loaded from a stack of YAML files. The
files split cleanly into two kinds: **runtime configuration** (knobs an operator
might want to tune per run — search ranges, emission thresholds, label fonts) and
**static data** (per-body shape tables, per-ring catalogues, per-instrument
calibration constants curated from external publications). Both kinds load through
the same loader; they differ in how they are reviewed and updated.

This chapter documents the loader, the section structure, the file layout, and the
static-data citation discipline.

The Config object
=================

:class:`~spindoctor.config.config.Config` lazily loads its YAML stack on first attribute
access. The stack is, in order (later files override earlier ones for the same key):

1. The bundled ``src/spindoctor/config_files/*.yaml`` files, sorted by filename. The
   3-digit numeric prefix is the merge order; the file groups are documented
   under :ref:`config-file-layout` below.
2. Exactly one of: any files passed via ``--config-file PATH`` on the CLI
   (repeatable; each merges in order), or -- only when no ``--config-file`` is
   given -- ``nav_default_config.yaml`` in the current working directory (if
   present), for per-checkout personal defaults. Passing ``--config-file``
   replaces the personal-defaults file rather than merging on top of it.
3. The handful of CLI flags that map to config keys (``--pds3-holdings-root``,
   ``--nav-results-root``, etc.) override the corresponding ``environment``
   block entry.

Direct programmatic use:

.. code-block:: python

   from spindoctor.config import Config, DEFAULT_CONFIG

   cfg = Config()                            # lazy; reads the stack on first access
   cfg.update_config('custom.yaml')          # merge in an override file
   print(cfg.offset.correlation_fft_upsample_factor)
   print(cfg.environment.pds3_holdings_root)

The module-level :data:`~spindoctor.config.config.DEFAULT_CONFIG` singleton is what
most subsystems read when no explicit ``config=`` keyword is supplied.
Per-class ``config`` properties on
:class:`~spindoctor.support.nav_base.NavBase` subclasses surface the same singleton
through dependency injection.

Sections
--------

Top-level YAML keys are exposed as
:class:`~spindoctor.support.attrdict.AttrDict` properties so code can write
``cfg.bodies.use_lambert`` instead of ``cfg['bodies']['use_lambert']``. That
convenience has a cost: an :class:`~spindoctor.support.attrdict.AttrDict` is its
own instance dictionary, so
anything that sets an attribute on a section adds a key to it. ``oops`` caches
its mutability verdict by setting an attribute on every object it walks, and it
walks everything reachable from an observation, so building a backplane reaches
the shared configuration and writes a key into every section of it, which
:func:`~spindoctor.config.logging_keys.validate_logging_config` then rejects.
The class carries ``oops``'s own opt-out marker to prevent that; it is a
workaround for one package rather than a general convention, and is tracked for
removal. The shipping sections:

- ``general`` — global settings shared across programs.
- ``logging`` — logger defaults, per-module levels, and per-program
  overrides.
- ``offset`` — correlation and star-refinement parameters.
- ``stars`` — star-model and ring-occlusion parameters; see
  :doc:`dev_guide_navigation_models_star`.
- ``bodies`` — body rendering parameters; see
  :doc:`dev_guide_navigation_models_body`.
- ``rings`` — ring-model parameters (planet shadow removal, fade widths,
  per-planet ``ring_features``); see
  :doc:`dev_guide_navigation_models_ring`.
- ``titan`` — Titan-specific parameters: the haze envelope height, the emission-side
  reliability floors and mask settings, and the two blocks of fit tunables the haze
  technique reads (``titan.navigation.symmetry`` and ``titan.navigation.arc``). See
  :doc:`dev_guide_navigation_models_titan` and :doc:`dev_guide_techniques_titan_haze`.
- ``bootstrap`` — bootstrap navigation parameters.
- ``backplanes`` — the list of body and ring backplanes to generate; see
  :doc:`dev_guide_backplanes`.
- ``pds4`` — per-dataset PDS4 template directories and bundle names; see
  :doc:`dev_guide_pds4`.
- ``environment`` — deployment locations (``pds3_holdings_root``,
  ``nav_results_root``, ``backplane_results_root``, ``bundle_results_root``,
  ``results_index_db``).
- ``results_tree`` — how much of a pass over a navigation results tree runs at
  once, read into a :class:`~spindoctor.nav_records.TreeTuning` by
  :func:`~spindoctor.config.get_results_tree_tuning`; see
  :doc:`dev_guide_results_index`.
- ``body_shape`` — static per-body shape catalogue (see
  :ref:`static-data-citations` below).
- ``coiss`` / ``vgiss`` / ``gossi`` / ``nhlorri`` — per-camera blocks
  (``noise``, ``mag_offset``, ``image_quality_thresholds``,
  ``source_image_filter``, etc.).
- ``techniques`` — per-:class:`~spindoctor.nav_technique.nav_technique.NavTechnique`
  tunables and confidence-formula
  coefficients; see :doc:`dev_guide_techniques`.
- ``satellites`` — per-planet satellite lists used by the body
  :class:`~spindoctor.nav_model.nav_model.NavModel`'s inventory query.
- ``feature_emission`` — per-planet
  :attr:`~spindoctor.feature.feature_type.NavFeatureType.RING_EDGE` vs
  :attr:`~spindoctor.feature.feature_type.NavFeatureType.RING_ANNULUS` gates; see
  :doc:`dev_guide_techniques_ring_annulus`.

The user-facing tour at :doc:`/introduction_configuration` covers how operators
override these defaults with their own files; this chapter is the reference for
what ships and where.

.. _config-file-layout:

File layout
===========

The numeric prefix encodes the load order and hints at the section's role. The
ranges are conventional, not enforced by the loader:

.. list-table::
   :header-rows: 1
   :widths: 18 35 47

   * - Prefix
     - Group
     - Files
   * - ``0xx``
     - Global / model-shared
     - ``config_010_general``, ``config_015_logging``, ``config_020_offset``,
       ``config_030_stars``,
       ``config_040_bodies``, ``config_050_rings``, ``config_060_titan``,
       ``config_070_bootstrap``
   * - ``1xx``
     - Catalogues
     - ``config_100_satellites``
   * - ``2xx``
     - Per-target tables (static data)
     - ``config_220_body_shape``
   * - ``3xx``
     - Per-planet ring catalogues (static data)
     - ``config_300_jupiter_rings``, ``config_310_saturn_rings``,
       ``config_320_uranus_rings``, ``config_330_neptune_rings``
   * - ``4xx``
     - Per-instrument camera blocks (mixed runtime + static)
     - ``config_400_inst_coiss``, ``config_410_inst_gossi``,
       ``config_420_inst_nhlorri``, ``config_430_inst_vgiss``,
       ``config_440_sim``
   * - ``5xx``
     - Per-technique tunables
     - ``config_510_techniques``
   * - ``9xx``
     - Downstream-product settings
     - ``config_900_backplanes``, ``config_950_pds4``,
       ``config_960_results_tree``

Per-file contents
-----------------

Each shipping file and what it holds:

- ``config_010_general`` — general settings shared across programs.
- ``config_015_logging`` — logging defaults, per-module levels, and
  per-program overrides.
- ``config_020_offset`` — offset-finding and star-refinement parameters.
- ``config_030_stars`` — star-model and ring-occlusion parameters.
- ``config_040_bodies`` — body (planet / moon) rendering parameters.
- ``config_050_rings`` — ring-model parameters.
- ``config_060_titan`` — Titan-specific navigation parameters.
- ``config_070_bootstrap`` — bootstrap navigation parameters (angles in degrees).
- ``config_100_satellites`` — satellite definitions for each planet.
- ``config_220_body_shape`` — per-body shape table (radii, ellipsoid residual,
  crater scale, albedo) consumed by the body NavModel and feature extractors;
  see :ref:`static-data-citations`.
- ``config_300_jupiter_rings`` / ``config_310_saturn_rings`` /
  ``config_320_uranus_rings`` / ``config_330_neptune_rings`` — per-planet ring
  system parameters.
- ``config_400_inst_coiss`` — Cassini ISS instrument-specific settings.
- ``config_410_inst_gossi`` — Galileo SSI instrument-specific settings.
- ``config_420_inst_nhlorri`` — New Horizons LORRI instrument-specific settings.
- ``config_430_inst_vgiss`` — Voyager ISS instrument-specific settings.
- ``config_440_sim`` — simulated-image settings.
- ``config_510_techniques`` — per-NavTechnique confidence-formula coefficients and
  runtime tunables (spurious-detection thresholds, at-edge tolerances, minimum
  arc lengths) plus the planet-specific ``feature_emission.ring_annulus`` block
  that decides RING_EDGE vs RING_ANNULUS emission.
- ``config_900_backplanes`` — backplane-generation settings.
- ``config_950_pds4`` — PDS4 metadata and export settings for generated products,
  PDS4 label-template overrides, and the mapping of internal fields to PDS4 keys.
- ``config_960_results_tree`` — how much of a pass over a navigation results
  tree runs at once; see :doc:`dev_guide_results_index`.

Loader rules
------------

- YAML mappings are deep-merged, not overwritten. A user override that sets
  ``bodies.use_lambert: false`` does not unset every other key under
  ``bodies``.
- Lists are overwritten wholesale. A user override of ``stars.catalogs``
  overwrites the bundled list rather than appending to it.
- Mapping keys whose name starts with ``_`` are stripped at load time. This
  is the strip-rule that lets static-data files carry ``_sources`` blocks
  alongside their numeric values without bloating the parsed
  :class:`~spindoctor.config.config.Config` object;
  see :ref:`static-data-citations`.

Numeric values are typed by YAML; downstream consumers convert with explicit
casts where the section schema is mixed (e.g. the per-instrument
``image_quality_thresholds`` block constructs a frozen
:class:`~spindoctor.nav_orchestrator.image_classifier.ImageQualityThresholds`
dataclass).

Path resolution
---------------

The ``environment`` block carries the PDS3 holdings read root, three downstream
output roots, and the results index URL:

- ``pds3_holdings_root`` — read root for PDS3 holdings (default
  ``$PDS3_HOLDINGS_DIR``, falling back to
  ``https://pds-rings.seti.org/holdings``).
- ``nav_results_root`` — write root for ``_metadata.json`` and ``_summary.png``
  files produced by ``sd_offset``.
- ``backplane_results_root`` — write root for backplane FITS / NumPy products
  produced by ``sd_backplanes``.
- ``bundle_results_root`` — write root for PDS4 bundles produced by
  ``sd_create_bundle``.
- ``results_index_db`` — connection URL of the results index, a database derived
  from the navigation results tree by a separate ingest step. It is not
  authoritative: the per-image ``_metadata.json`` documents are, and the index
  can be deleted and rebuilt from them. A ``sqlite:`` URL names a local
  filesystem path and nothing else: it carries no query string, because the
  driver would then open a file named after the query rather than the file named
  in the URL. A ``postgresql+psycopg:`` URL names a server. Leaving it unset
  means "no index", which is the default mode of every program, and the literal
  value ``none`` says so explicitly, overriding a URL set elsewhere. A value
  that is empty, or nothing but spaces, is neither a URL nor that word, and is
  refused at whichever level carries it rather than read as either.

Each root may be a local path or a URL; ``filecache``-aware consumers handle
both. ``results_index_db`` is the exception: it is a database connection URL,
not a location ``filecache`` resolves. Environment-variable overrides
(``PDS3_HOLDINGS_DIR``, ``NAV_RESULTS_ROOT``, ``BACKPLANE_RESULTS_ROOT``,
``BUNDLE_RESULTS_ROOT``, ``NAV_RESULTS_INDEX_DB``) take precedence over the YAML
defaults; CLI flags take precedence over the env vars.

The whole ``environment`` block is left out of the provenance configuration
digest recorded with each navigation result. It says where a deployment keeps
its files, never what the pipeline computes, so two results that differ only in
it were produced by the same configuration.

.. _static-data-citations:

Static data: catalogues and citation discipline
================================================

The pipeline treats a small set of YAML files as **static data**: per-body
shape parameters, per-ring radial uncertainties, and per-instrument
photometric / noise constants. These tables substitute for cross-image
statistical learning — no run depends on the result of any other run, but
every run benefits from values that astronomers and instrument teams have
already calibrated.

Static-data files
-----------------

- ``config_220_body_shape.yaml`` populates ``config.body_shape`` — per-body
  radii, ellipsoid residuals, albedo, crater scale; consumed by
  :func:`~spindoctor.nav_model.body_shape.load_body_shape` and from there by every
  body :class:`~spindoctor.nav_technique.nav_technique.NavTechnique`'s covariance
  and reliability formula. See
  :doc:`dev_guide_navigation_models_body`.
- ``config_3N0_*_rings.yaml`` populate ``config.rings.<planet>.ring_features``
  — per-ring-edge radii, eccentricities, RMS radial precision; consumed by
  the ring-edge extractor to derive per-edge ``sigma_radial``. See
  :doc:`dev_guide_navigation_models_ring`.
- ``config_4N0_inst_*.yaml`` populate ``config.<camera>`` — per-camera
  ``noise``, ``mag_offset``, ``image_quality_thresholds``, and
  ``source_image_filter`` blocks; consumed by the orchestrator preflight, the
  star photometry helper, and the per-instrument PSF model.

Citation requirement
--------------------

Every numeric value in ``config_220_body_shape.yaml`` and any new value added
to a ``config_4N0_inst_*.yaml`` ``noise:`` / ``mag_offset:`` block **requires
an accurate, non-fabricated citation**. The reasoning:

- Navigation trust is downstream-safety-critical. An invented
  ``ellipsoid_rms_residual_km`` propagates silently into every per-feature
  uncertainty estimate for that body for every image forever.
- The runtime has no cross-image cross-check that would catch a wrong value;
  the orchestrator trusts the static data and feeds it directly into
  reliability scores and technique covariances.

Schema
------

Each body block in ``config_220_body_shape.yaml`` is wrapped in a top-level
``body_shape:`` mapping; each entry is keyed by upper-case SPICE body name and
carries an optional sibling ``_sources`` mapping. Keys beginning with ``_``
are stripped at config-load time so the documentation does not bloat the
parsed ``Config`` — the citation lives in the file for human review only.

.. code-block:: yaml

    body_shape:
      MIMAS:
        radii_km: [207.8, 196.7, 190.6]
        ellipsoid_rms_residual_km: 0.74
        crater_scale_km: 1.0
        albedo_mean: null
        albedo_variation: 0.06
        shape_class_hint: regular
        _sources:
          radii_km: 'Archinal et al. 2011, CMDA 109, Table 5 (per Thomas
                     2010, Icarus 208); PDF fetched in-session 2026-07-10'
          ellipsoid_rms_residual_km: '...'
          # ... and so on for every numeric field.

Body-shape table: sources
-------------------------

The table carries entries for the bodies the four supported missions
navigate: the Saturn system including the irregulars, the Galileans plus
Amalthea, the Uranian majors, Triton and Proteus, and the Pluto system.
Two documents carry nearly all of the measured values:

- **Thomas et al. (2007), Icarus 190, 573–584** — *Shapes of the
  saturnian icy satellites and their significance*.  The source of the
  ellipsoid RMS limb-fit residuals for the six classical Saturn moons and
  Phoebe: the paper defines the roughness as "the root-mean-square (rms)
  of the radial residual of each limb point from the best-fit ellipsoid"
  (section 2.3), quotes Mimas at 0.74 km (section 3.2), and plots the
  full set in its Fig. 8 (km and fraction-of-radius panels).  Its
  section 4 also gives the **2.5–8 %-of-mean-radius roughness class** for
  small satellites and asteroids (citing Thomas 1989), which is the
  stated basis for the ESTIMATE residuals of Hyperion, Janus,
  Epimetheus, Prometheus, and Pandora — bodies with no directly-published
  per-body residual.
- **Archinal et al. (2011), Celest Mech Dyn Astr 109, 101–135** — the IAU
  WGCCRE 2009 report.  Table 5 supplies every satellite's radii (the
  Saturn rows reproduce Thomas 2010) and the RMS-deviation-from-ellipsoid
  values for Titan (0.26 km), Europa (0.32), Callisto (0.6), Amalthea
  (3.2), the Uranian majors, and Proteus (7.9); Table 4 supplies the
  1-bar planet ellipsoids.

Pluto and Charon radii cite **Nimmo et al. (2017), Icarus 287**
(arXiv:1603.00821).  Io's and Ganymede's RMS deviations cite
**Archinal et al. (2018), CMDA 130:22** Table 5; their ``_sources``
entries record that the citation was made from a search-result summary
rather than the document itself and flag them for reviewer spot-check.

Fields that are *estimates by design*: ``crater_scale_km`` (characteristic
limb topographic roughness beyond the ellipsoid) and ``albedo_variation``
have no standard published per-body scalar; entries carry
``'ESTIMATE — <physical basis>'`` sources and feed sigma/reliability
terms only.  ``albedo_mean`` is ``null`` throughout because it has no
runtime consumer; a developer adding a consumer should populate it with
cited values at that point, following the procedure below.

Downstream, ``ellipsoid_rms_residual_km`` drives the LIMB_ARC
normal-sigma quadrature and the ``max_phase_irregularity_factor``
confidence term of :class:`~spindoctor.nav_technique.nav_technique_body_blob.BodyBlobNav` (see
:doc:`dev_guide_navigation_models_body`).  The calibration tooling in
``util/calibration/`` renders its simulated bodies at relief amplitudes
derived from these same residual-over-radius ratios, so the sim-anchored
confidence coefficients and this table form one system: a developer who
revises a residual here should re-run that calibration (see
``util/calibration/README.md``).

Anti-hallucination procedure
----------------------------

AI agents drafting body-shape entries:

1. **Cite only documents fetched in-session.**  Every citation must be
   traceable to a ``WebFetch`` / ``WebSearch`` lookup performed in the same
   session, or to an ``oops``-package data file read directly. No citing
   from training-data memory.  A citation made from a search-result
   *summary* (rather than the fetched document itself) is permitted only
   when its ``_sources`` entry says so explicitly and flags the value for
   reviewer spot-check, as the Io and Ganymede residual entries do.
2. If a value cannot be sourced from a fetched document, leave it as ``null``
   and write
   ``'PLACEHOLDER — no source found, calibration pending'`` as the
   ``_sources`` entry. The runtime fallback (10 % radius default plus a
   reliability cap of 0.3) handles ``null`` values.
3. DOIs and paper titles must verify against a real
   ``https://doi.org/<DOI>`` lookup; agents do not invent identifiers.
4. Any draft PR that lists a citation an AI agent invented (caught in human
   review) is reverted in full and re-drafted by a different process.

Human review
------------

Every PR touching ``config_220_body_shape.yaml`` requires a reviewer to
spot-check **at least 5 randomly-selected citations** by opening the cited
document and verifying the value appears at the cited location. PRs are
merged only after the reviewer marks the PR with the
``cited-values-spot-checked`` label.

Validation tests
----------------

``tests/spindoctor/config_files/test_body_shape_citations.py`` enforces:

- Every body declares a ``_sources`` mapping.
- Every required numeric / list field on a body has a corresponding
  ``_sources`` entry that is a non-empty string.
- No ``_sources`` value contains the substrings ``TODO`` / ``FIXME`` /
  ``XXX`` (case-insensitive).
- ``PLACEHOLDER`` is allowed only when the value itself is ``null``.

The same validation pattern extends to per-camera ``noise`` /
``mag_offset`` blocks in ``config_4N0_inst_*.yaml`` and to any new entries
added to ``config_3N0_*_rings.yaml``. Existing ring-catalogue values are
grandfathered (they were curated by orbit-fitting astronomers and the
catalogues document their pedigree in the file header) — only *new*
additions need explicit ``_sources`` entries.

Strip-rule guarantee
--------------------

``Config._load_yaml`` strips every mapping key whose
name starts with ``_`` before merging, so ``_sources`` blocks never appear in
the parsed :class:`~spindoctor.config.config.Config` object. The runtime accessors
(``config.body_shape``, ``config.<camera>.mag_offset``, etc.) see only the
value-bearing fields. Tests assert this behaviour explicitly so the strip
rule cannot regress silently.

Adding a new tunable
====================

When a new YAML knob is added:

1. Pick the file whose section it belongs to (e.g. a new ``bodies`` key goes
   in ``config_040_bodies.yaml``). If no existing file fits, allocate a new
   numeric prefix per the layout table above.
2. Add the key with a sensible default and a one-line YAML comment naming
   the consumer.
3. Document the key on the consumer's dev-guide page (for example,
   :doc:`dev_guide_navigation_models_body` lists every key under
   ``bodies``). The page lists name, type, default, units, and consumer.
4. If the key represents static data (a measured constant rather than a
   knob), add the ``_sources`` entry per the citation discipline above.
5. Add or extend a unit test under ``tests/spindoctor/config_files/`` that asserts
   the loader exposes the key with the expected default.
