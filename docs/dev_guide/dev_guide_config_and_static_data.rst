==========================
Config and Static Data
==========================

Every SpinDoctor subsystem reads its tunables from a single
:class:`~spindoctor.config.config.Config` object loaded from a stack of YAML files. The
files split cleanly into two kinds: **runtime configuration** (knobs an operator
might want to tune per run — search ranges, emission thresholds, label fonts) and
**static data** (per-body shape tables, per-ring catalogs, per-instrument
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
   block entry. ``--pds3-holdings-root`` is declared by
   :class:`~spindoctor.dataset.dataset_pds3.DataSetPDS3` rather than by each
   program, so a program offers it exactly when the dataset it was asked for
   enumerates a PDS3 holdings tree.

.. _config-merge:

How the files merge
-------------------

Each file is read with :func:`~spindoctor.schema.layers.read_layer`, and the files are
merged in the order above by :func:`~spindoctor.schema.layers.merge`, the same merge the
simulator's files use (see :doc:`dev_guide_schema`):

- A mapping merges key by key. An override that sets ``bodies.use_lambert: false``
  keeps every other key under ``bodies``, and the same holds for a key set deeper in a
  section.
- Any other value replaces the earlier one whole. A list replaces the earlier list, so
  an override of ``stars.catalogs`` takes the place of the packaged list instead of
  adding to it.
- A null is refused, in every file, under
  :attr:`~spindoctor.schema.layers.NullPolicy.REFUSE`. A key that is not set is left
  out of the file, and every reader treats an absent key as not set. A user override
  can therefore change a packaged value but cannot unset it.
- An override that gives a whole section, such as ``bodies``, as something other than
  a mapping is refused too.
- Every mapping key that begins with an underscore is dropped as its file is read, at
  any depth. That is what lets a static-data file carry ``_sources`` citations beside
  its values (see :ref:`static-data-citations`): no reader of the configuration ever
  sees them, and tests hold the loader to that.

A file the merge refuses raises :class:`~spindoctor.schema.errors.SchemaError`, a
:class:`ValueError` whose message has one line per problem, in the form
``file: key.path: message``. An override is given by the path it was loaded from, and
a packaged file by its name. The merged result's typed sections are then parsed (see
:ref:`config-typed-sections`). An override refused at either step leaves the
configuration as it was, so a later override still loads.
:attr:`~spindoctor.config.config.Config.override_paths` lists the overrides applied, in
order, for provenance.

Numeric values are typed by YAML. A consumer of a section that is not typed converts a
value with an explicit cast.

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

.. _config-typed-sections:

Typed sections
--------------

Some sections are parsed into frozen dataclasses through the schema core (see
:doc:`dev_guide_schema`), every time a configuration is installed: once when the
packaged files are read, and again for each override. The typed sections are the fields
of :class:`~spindoctor.config.sections.TypedSections`. Each one is named after its
top-level key, and the matching :class:`~spindoctor.config.config.Config` property
returns the typed object:

- ``planets`` — the SPICE names of the planets the navigator knows, read through
  :attr:`~spindoctor.config.config.Config.planets`.
- ``orchestrator`` — :class:`~spindoctor.config.sections.OrchestratorSection`, holding
  ``ensemble`` (an :class:`~spindoctor.nav_orchestrator.ensemble.EnsembleConfig`; see
  :doc:`dev_guide_orchestrator_ensemble`) and ``reliability_gate`` (a
  :class:`~spindoctor.feature.reliability.FeatureReliabilityGate`).
- ``bodies`` — :class:`~spindoctor.config.sections_models.BodiesSection`. A key of
  ``geometric_albedo`` is a body's SPICE name and must be in upper case.
- ``stars`` — :class:`~spindoctor.config.sections_models.StarsSection`. An opaque ring
  annulus in ``ring_occlusion_radii_km`` must have its inner radius below its outer
  radius.
- ``titan`` — :class:`~spindoctor.config.sections_models.TitanSection`. Its
  ``navigation.symmetry`` and ``navigation.arc`` blocks parse straight into the haze
  fit's :class:`~spindoctor.nav_technique.titan_fitting.SymmetryFitParams` and
  :class:`~spindoctor.nav_technique.titan_fitting.ArcFitParams`, and its opaque ring
  annulus follows the same rule as the star one.
- ``satellites`` — a list of satellites for each planet, keyed by the planet's SPICE
  name and read through :meth:`~spindoctor.config.config.Config.satellites`. Each
  satellite is its SPICE name and must be in upper case.
- ``feature_emission`` — :class:`~spindoctor.config.sections_models.FeatureEmissionSection`.
  A planet's entry under ``ring_annulus.planets`` may leave a key out, and that key
  keeps the ``default`` entry's value.
- ``rings`` — :class:`~spindoctor.config.sections_rings.RingsSection`, holding the ring
  model's settings and, under ``ring_features``, each planet's ring catalog as a
  :class:`~spindoctor.config.sections_rings.RingCatalog`. The rules a catalog follows
  are listed in :doc:`dev_guide_navigation_models_ring`.
- ``general`` — :class:`~spindoctor.config.sections.GeneralSection`: the directory of
  TrueType fonts that annotation text and summary labels are drawn in
  (``truetype_font_dir``).
- ``offset`` — :class:`~spindoctor.config.sections.OffsetSection`: the FFT upsample
  factor of the image correlation (``correlation_fft_upsample_factor``), from 1 to
  :data:`~spindoctor.config.sections.MAX_UPSAMPLE_FACTOR`. The disc correlation, the ring
  annulus correlation, and the manual navigation dialog read it from here.
- ``results_tree`` — a :class:`~spindoctor.nav_records.TreeTuning`: how much of a pass
  over a navigation results tree runs at once; see :doc:`dev_guide_results_index`. It
  refuses a round of work smaller than the thread pool that the round feeds.
- ``environment`` — :class:`~spindoctor.config.sections.EnvironmentSection`: deployment
  locations (``pds3_holdings_root``, ``nav_results_root``, ``backplane_results_root``,
  ``bundle_results_root``, ``log_root``, and ``results_index_db``). Every key is
  optional, and a key left out reads as None. The packaged file
  (``config_005_environment.yaml``) leaves every key unset and lists each as a
  commented-out line. The schema does not decide whether the command line, this section,
  or an environment variable wins. Each reader applies its own order, and the helpers in
  :mod:`spindoctor.config.config_helper` define the usual one.
- ``body_shape`` — the body-shape table: one
  :class:`~spindoctor.config.sections_body_shape.BodyShapeEntry` per body, keyed by the
  body's SPICE name. Every key of an entry is optional, and a key that the entry leaves
  out takes its value from the next tier (see :ref:`static-data-citations`).
- ``logging`` — :class:`~spindoctor.config.logging_keys.LoggingSection`, holding the
  levels and terminal settings of the two loggers, with each program's overrides under
  ``programs`` as a :class:`~spindoctor.config.logging_keys.ProgramLogging`. Its
  module keys come from the technique and model registries, so the set of keys it
  accepts is known only when the program runs; see :doc:`dev_guide_logging`.
- ``techniques`` — one block per technique the package ships: its confidence formula
  and its tuning (see :ref:`config-techniques-section`).
- ``backplanes`` — :class:`~spindoctor.config.sections_products.BackplanesSection`,
  holding the body and ring planes, the ring incidence angle, the masked value, and
  ``target_lids``; see :doc:`dev_guide_backplanes`. ``target_lids`` maps each body
  and ring system to its PDS4 target. Its keys are SPICE names in upper case, and the
  loader refuses a key that is not.
- ``pds4`` — one :class:`~spindoctor.config.sections_products.Pds4Bundle` per bundle.
  As with the ``techniques`` section, its type is built from a registry. Each
  registered dataset class that sets ``PDS4_CONFIG_KEY`` adds one optional field under
  that key, and the loader refuses an entry under any other key.
  :meth:`~spindoctor.config.sections_products.Pds4Section.bundle` returns one bundle's
  entry; see :doc:`dev_guide_pds4`.
- the camera sections — each camera's block, a
  :class:`~spindoctor.config.sections_camera.CameraBlock`, under the section the files
  give it; see :ref:`config-camera-blocks`.

Four mappings are keyed by planet: ``satellites``, ``stars.ring_occlusion_radii_km``,
``feature_emission.ring_annulus.planets``, and ``rings.ring_features``. Each of their
keys must be a planet of ``planets``, written in upper case as SPICE spells it. Every
reader looks a planet up by its upper-case name, and would never read a key that is
misspelled or in lower case, so such a key is refused at load. That rule reads two
sections, so :class:`~spindoctor.config.sections.TypedSections` checks it. The message
gives the key and the file that last set either it or ``planets``.

Two mappings are keyed by body: ``bodies.geometric_albedo`` and ``body_shape``. For the
same reason, each of their keys must be in upper case, as SPICE spells it, and the loader
refuses a key that is not. One rule in :class:`~spindoctor.config.sections.TypedSections`
checks both mappings, so both give the same message.

Keys come in two kinds, and each kind has its own spelling. A key for a setting,
such as ``use_albedo`` or ``body_limb``, is a name the package chose, and it is in
snake_case. A key for something outside the package is spelled the way its source
spells it: a body or planet as SPICE spells it, such as ``SATURN`` or ``TITAN``, and a
feature type by its name, such as ``STAR`` or ``LIMB_ARC``. Body names stay in upper case
because SPICE, the metadata documents, the backplanes, the logs, and the simulator scenes
all spell them that way. A lower-case key would show the user a second spelling of the
same body, and every lookup would have to convert the SPICE name before reading it.

.. _config-camera-blocks:

Camera blocks
^^^^^^^^^^^^^

A camera block holds what the navigator knows about one camera: the units its images
are read in (``data_units``), the margin its offset search may reach
(``extfov_margin_vu``), its star PSF (``star_psf_sigma`` and ``star_psf_sizes``), the
share of the offset window in which a lone star is believed
(``one_star_ellipse_fraction``), the rotation settings (``fit_camera_rotation`` and
``max_rotation_deg``), and its ``noise``, ``image_quality_thresholds``,
``source_image_filter``, and ``mag_offset`` blocks. Each camera's block parses into
:class:`~spindoctor.config.sections_camera.CameraBlock`. Each of its blocks has a type of
its own: :class:`~spindoctor.config.sections_camera.CameraNoise`, with a
:class:`~spindoctor.config.sections_camera.BinnedNoise` for each binned readout mode;
:class:`~spindoctor.config.sections_camera.QualityThresholds`;
:class:`~spindoctor.config.sections_camera.MagOffset`; and a source-image filter chosen
by its ``kind``, either a :class:`~spindoctor.config.sections_camera.BandpassFilter` or a
:class:`~spindoctor.config.sections_camera.BlurFilter`. Each field's type is also
exported under a name of its own, such as
:data:`~spindoctor.config.sections_camera.ReadNoiseDn`, so code that states some of the
same keys can type them with the same ranges.

The loader finds the camera blocks by the key that every one of them sets,
``data_units`` (:func:`~spindoctor.config.sections_camera.camera_sections`). A top-level
section that sets it is one camera's block. A section whose entries set it holds
several cameras, keyed by camera name, and parses into a mapping of camera name to
block. Each camera section becomes a field of the typed sections under its own name, so
the key path of a problem starts at the section, followed by the camera in a section
that holds several. :meth:`~spindoctor.config.config.Config.camera_block` returns one
camera's block,
:meth:`~spindoctor.config.config.Config.camera_names` gives the names of a section's
cameras, and an observation carries its camera's block as
:attr:`~spindoctor.obs.obs_inst.ObsInst.inst_config`.

Which keys a camera sets follows its ``data_units``:

- A ``raw_dn`` camera sets ``noise.saturation_dn``, ``noise.read_noise_dn``, and
  ``noise.star_flux_dn_per_s_vmag0``, and the DN limits of its quality thresholds:
  ``blank_max_dn``, ``saturation_threshold_dn``, and ``noisy_threshold_dn``. It may also
  set ``noise.by_binning``, keyed by binning factors of 2 or more. Each entry of that
  table sets ``read_noise_dn`` and ``star_flux_dn_per_s_vmag0``. If the table has no
  entry for a binned mode, that mode reads the unbinned values.
- A ``calibrated_if`` camera sets the I/F limits, ``blank_max_if`` and
  ``noisy_threshold_if``, and leaves out every DN value of the two blocks. The same
  full-well count maps to a different I/F for every exposure, filter, and gain, so such
  a camera has no saturation limit.
- Every other key is either set by every camera or optional by nature. A camera that
  leaves out ``star_max_peak_per_predicted_snr`` puts no bound on a lone star's peak.

A key that the camera's units do not read is refused at load, and so is a missing key
that they do read. Two more rules tie keys together: the magnitude offset's
``fallback_combo`` must have an entry in ``mag_offset_table``, and a bandpass filter's
heavy blur (``lo_sigma_px``) must be wider than its light one (``hi_sigma_px``).

The simulator's own camera section is parsed into the same type as the others when the
configuration loads. A simulated scene's ``instrument_config`` block is merged over its
camera's block, and the merged block is parsed into the same type again (see
:ref:`sim-instrument-config`).

.. _config-techniques-section:

The techniques section
^^^^^^^^^^^^^^^^^^^^^^

The ``techniques`` section's type is built from the technique registry
(:mod:`spindoctor.nav_technique.confidence_config`), so no list of techniques is
written twice. :func:`~spindoctor.config.sections.typed_sections_type` returns the type
the loader parses: :class:`~spindoctor.config.sections.TypedSections` with one field
per shipped technique. Techniques that tests register are not shipped, have no block,
and set their own confidence formula.

A block holds the technique's confidence formula (``alpha0``, ``terms``,
``hard_zero_if``, and ``hard_cap``; see :doc:`dev_guide_techniques_confidence`) and its
``tuning`` mapping:

- Every technique declares ``Tuning``, a frozen schema dataclass of its tuning keys.
  The loader parses the block's ``tuning`` into that type, and the technique reads
  ``self.tuning.key``. A technique family that shares keys declares them once on a base
  type:

  - The DT techniques' types share
    :class:`~spindoctor.nav_technique.dt_tuning.DTTuning`, which in turn holds the keys
    of the shared fit-quality gates
    (:class:`~spindoctor.nav_technique.dt_fit_gates.DTFitGateConfig`).
  - :class:`~spindoctor.nav_technique.correlation_tuning.NCCCovarianceTuning` holds the
    model-error terms of a correlation technique's covariance. The ring-annulus
    technique's tuning is that type, and the body-disc technique's type extends it.
  - The star techniques' types share
    :class:`~spindoctor.nav_technique.star_tuning.StarTuning`, which holds the keys of
    the detection kernel, the centroid box, the detection threshold, and the at-edge
    tests.

- A technique whose covariance carries a model-error floor declares
  ``model_error_floor_px`` in its tuning, and passes it to
  :func:`~spindoctor.nav_technique.nav_technique.add_model_error_floor`. The floor is one
  number for every camera.
- Each block is keyed by the technique's snake_case key, the one ``logging.techniques``
  uses (:func:`~spindoctor.config.logging_keys.log_key_for`), such as ``body_limb``,
  ``star_field_from_catalog``, or ``titan_haze``. Results and metadata still identify
  the technique by its ``name``.
- The ``terms`` and the ``hard_zero_if`` keys of a block may refer only to the
  diagnostic attributes its technique declares in ``confidence_attributes``. Any other
  attribute is refused at load with the file and key path, as in
  ``techniques.body_limb.terms[0].feature``.

Each time a configuration is installed, the loader assigns every shipped technique
class its ``confidence_spec`` and ``tuning`` from the parsed section. An override that
is refused assigns nothing. A block under a name the section does not use, such as a
technique's class name, is refused as an unknown key.

The rules for a typed section:

- A tunable's value lives in the YAML only. Its field has no default, so the packaged
  file must set it and no copy of it exists in code.
- A key that is optional by nature is typed ``X | None = None``, so leaving it out reads
  as None. The low confidence tier's ``max_sigma_px`` is such a key: the tier sets no
  sigma requirement.
- An unknown key, a value of the wrong type, and a value out of range are refused when
  the configuration loads, and so is a broken rule that ties two keys together. The
  message has one line per problem, in the form ``file: key.path: message``. The file
  is the one that set the value, and the key path starts at the section's name, as in
  ``orchestrator.ensemble.min_confidence``.
- A reader takes the typed object from the configuration and never repeats a check the
  type makes. A test that needs different values builds a variant of the packaged section
  with :func:`dataclasses.replace`.

Sections
--------

Top-level YAML keys other than the typed sections, such as each camera's block, are read
through :meth:`~spindoctor.config.Config.category`, which returns an
:class:`~spindoctor.support.attrdict.AttrDict` so code can write
``cfg.category('galileo_ssi').star_psf_sigma`` instead of
``cfg.category('galileo_ssi')['star_psf_sigma']``. That
convenience has a cost: an :class:`~spindoctor.support.attrdict.AttrDict` is its
own instance dictionary, so
anything that sets an attribute on a section adds a key to it. ``oops`` caches
its mutability verdict by setting an attribute on every object it walks, and it
walks everything reachable from an observation, so building a backplane reaches
the shared configuration and writes a key into every section of it, so a section
read back holds a key that no configuration file declares.
The class carries ``oops``'s own opt-out marker to prevent that; it is a
workaround for one package rather than a general convention, and is tracked for
removal. The shipping sections:

- ``general`` — global settings shared across programs; a typed section (see
  :ref:`config-typed-sections`).
- ``planets`` — the planets the navigator knows, by SPICE name; a typed section.
- ``logging`` — logger defaults, per-module levels, and per-program
  overrides; a typed section (see :ref:`config-typed-sections`).
- ``offset`` — the FFT upsample factor of the image correlation
  (``correlation_fft_upsample_factor``); a typed section.
- ``stars`` — star-model and ring-occlusion parameters; a typed section (see
  :ref:`config-typed-sections`) described in :doc:`dev_guide_navigation_models_star`.
- ``bodies`` — body rendering parameters; a typed section described in
  :doc:`dev_guide_navigation_models_body`.
- ``rings`` — ring-model parameters (planet shadow removal, fade widths,
  per-planet ``ring_features``); a typed section described in
  :doc:`dev_guide_navigation_models_ring`.
- ``titan`` — Titan-specific parameters: the haze envelope height, the emission-side
  reliability floors and mask settings, and the two blocks of fit tunables the haze
  technique reads (``titan.navigation.symmetry`` and ``titan.navigation.arc``); a typed
  section. See :doc:`dev_guide_navigation_models_titan` and
  :doc:`dev_guide_techniques_titan_haze`.
- ``backplanes`` — the body and ring backplanes to generate, and the PDS4 target of
  each body and ring system; a typed section (see :ref:`config-typed-sections`)
  described in :doc:`dev_guide_backplanes`.
- ``pds4`` — each bundle's template directory, name, version, and schemas; a typed
  section (see :ref:`config-typed-sections`) described in :doc:`dev_guide_pds4`.
- ``body_shape`` — static per-body shape catalog; a typed section (see
  :ref:`static-data-citations` below).
- the camera sections — each camera's block (``noise``, ``mag_offset``,
  ``image_quality_thresholds``, ``source_image_filter``, and the rest); typed (see
  :ref:`config-camera-blocks`).
- ``techniques`` — per-:class:`~spindoctor.nav_technique.nav_technique.NavTechnique`
  tunables and confidence-formula
  coefficients; see :doc:`dev_guide_techniques`.
- ``orchestrator`` — the ensemble combine's tunables and confidence tiers, and the
  per-feature-type reliability gate; a typed section (see :ref:`config-typed-sections`).
- ``satellites`` — per-planet satellite lists used by the body
  :class:`~spindoctor.nav_model.nav_model.NavModel`'s inventory query, by the star
  model's body conflicts, and by the backplanes; a typed section.
- ``feature_emission`` — per-planet
  :attr:`~spindoctor.feature.feature_type.NavFeatureType.RING_EDGE` vs
  :attr:`~spindoctor.feature.feature_type.NavFeatureType.RING_ANNULUS` gates; a typed
  section. See :doc:`dev_guide_techniques_ring_annulus`.

The user-facing tour at :doc:`/user_guide/user_guide_configuration` covers how operators
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
       ``config_040_bodies``, ``config_050_rings``, ``config_060_titan``
   * - ``1xx``
     - Catalogs
     - ``config_100_satellites``
   * - ``2xx``
     - Per-target tables (static data)
     - ``config_220_body_shape``
   * - ``3xx``
     - Per-planet ring catalogs (static data)
     - ``config_300_jupiter_rings``, ``config_310_saturn_rings``,
       ``config_320_uranus_rings``, ``config_330_neptune_rings``
   * - ``4xx``
     - Per-instrument camera blocks (mixed runtime + static)
     - ``config_400_inst_coiss``, ``config_410_inst_gossi``,
       ``config_420_inst_nhlorri``, ``config_430_inst_vgiss``,
       ``config_440_sim``
   * - ``5xx``
     - Per-technique and orchestrator tunables
     - ``config_510_techniques``, ``config_540_orchestrator``
   * - ``9xx``
     - Downstream-product settings
     - ``config_900_backplanes``, ``config_951_pds4_coiss_saturn``,
       ``config_960_results_tree``

Per-file contents
-----------------

Each shipping file and what it holds:

- ``config_010_general`` — general settings shared across programs.
- ``config_015_logging`` — logging defaults, per-module levels, and
  per-program overrides.
- ``config_020_offset`` — the image correlation's FFT upsample factor.
- ``config_030_stars`` — star-model and ring-occlusion parameters.
- ``config_040_bodies`` — body (planet / moon) rendering parameters.
- ``config_050_rings`` — ring-model parameters.
- ``config_060_titan`` — Titan-specific navigation parameters.
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
- ``config_510_techniques`` — the ``techniques`` section, each technique's
  confidence-formula coefficients and runtime tunables (spurious-detection thresholds,
  at-edge tolerances, minimum arc lengths), plus the planet-specific
  ``feature_emission.ring_annulus`` block that decides RING_EDGE vs RING_ANNULUS
  emission.
- ``config_540_orchestrator`` — the ``orchestrator`` section: the ensemble combine's
  tunables and confidence tiers, and the per-feature-type reliability gate.
- ``config_900_backplanes`` — backplane-generation settings.
- ``config_951_pds4_coiss_saturn`` — the ``coiss_saturn`` dataset's PDS4 bundle
  settings, ``pds4.coiss_saturn``: its template directory, the bundle's name and
  version (``bundle_name``, ``bundle_version``), and the information model version and
  the dictionary schemas its labels are written against (``information_model_version``,
  ``schemas``).  Each dataset that bundles has such a file, named for its mission and
  target; like every ``9xx`` file, none is hashed as static data (see
  :doc:`dev_guide_pds4`).
- ``config_960_results_tree`` — how much of a pass over a navigation results
  tree runs at once; see :doc:`dev_guide_results_index`.

Path resolution
---------------

The ``environment`` block carries the PDS3 holdings read root, three downstream
output roots, and the results index URL:

- ``pds3_holdings_root`` — read root for PDS3 holdings. No value ships, so a
  run that sets neither this nor ``PDS3_HOLDINGS_DIR`` nor
  ``--pds3-holdings-root`` stops and says so.
- ``nav_results_root`` — write root for ``_metadata.json`` and ``_summary.png``
  files produced by ``sd_offset``.
- ``backplane_results_root`` — write root for backplane FITS / NumPy products
  produced by ``sd_backplanes``.
- ``bundle_results_root`` — write root for PDS4 bundles produced by
  ``sd_create_bundle``.
- ``results_index_db`` — connection URL of the results index, a database derived
  from the navigation results tree by a separate ingest step. It is not
  authoritative: the per-image ``_metadata.json`` documents are, and the index
  can be deleted and rebuilt from them. A ``sqlite:`` URL points to a local
  filesystem path and nothing else: it carries no query string, because the
  driver would then open a file named after the query rather than the file named
  in the URL. A ``postgresql+psycopg:`` URL points to a server. Leaving it unset
  means "no index", which is the default mode of every program, and the literal
  value ``none`` says so explicitly, overriding a URL set elsewhere. A value
  that is empty, or nothing but spaces, is neither a URL nor that word, and is
  refused at whichever level carries it rather than read as either.

Each root may be a local path or a URL; ``filecache``-aware consumers handle
both. ``results_index_db`` is the exception: it is a database connection URL,
not a location ``filecache`` resolves. Every one of them resolves from the first
level that sets it: the CLI flag, then this ``environment`` block, then the
environment variable (``PDS3_HOLDINGS_DIR``, ``NAV_RESULTS_ROOT``,
``NAV_BACKPLANE_RESULTS_ROOT``, ``NAV_BUNDLE_RESULTS_ROOT``,
``NAV_RESULTS_INDEX_DB``). A key written into a configuration file therefore
wins over the same value exported into the environment, and a flag typed on the
command line wins over both.

The whole ``environment`` block is left out of the provenance configuration
digest recorded with each navigation result. It says where a deployment keeps
its files, never what the pipeline computes, so two results that differ only in
it were produced by the same configuration.

.. _static-data-citations:

Static data: catalogs and citation discipline
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
- ``config_4N0_inst_*.yaml`` populate the camera sections (see
  :ref:`config-camera-blocks`) — each camera's ``noise``, ``mag_offset``,
  ``image_quality_thresholds``, and ``source_image_filter`` blocks; consumed by the
  orchestrator preflight, the star photometry helper, and the per-instrument PSF model.

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
carries an optional sibling ``_sources`` mapping. The loader drops it (see
:ref:`config-merge`), so the citation lives in the file for human review only.

The loader then parses the entry into a
:class:`~spindoctor.config.sections_body_shape.BodyShapeEntry`, whose fields give each
key's type and range. Every key is optional. A key that the entry leaves out is read as
None, and :func:`~spindoctor.nav_model.body_shape.load_body_shape` takes that value from
the body's built-in profile. The loader refuses a key that the type does not declare,
and a body name that is not in upper case.

.. code-block:: yaml

    body_shape:
      MIMAS:
        radii_km: [207.8, 196.7, 190.6]
        ellipsoid_rms_residual_km: 0.74
        crater_scale_km: 1.0
        albedo_variation: 0.06
        shape_class_hint: regular
        _sources:
          radii_km: 'Archinal et al. 2011, CMDA 109, Table 5 (per Thomas
                     2010, Icarus 208); PDF fetched in-session 2026-07-10'
          ellipsoid_rms_residual_km: '...'
          albedo_mean: 'PLACEHOLDER — no in-session source fetched; no runtime consumer'
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
terms only.  ``albedo_mean`` is left out of every entry because it has no
runtime consumer, and each entry's ``_sources`` says so.
:class:`~spindoctor.config.sections_body_shape.BodyShapeEntry` does not declare it
either, so the loader refuses a value written for it.  A developer adding a consumer
should add the key to the type and populate it with cited values at that point,
following the procedure below.

Downstream, ``ellipsoid_rms_residual_km`` drives the LIMB_ARC
normal-sigma quadrature and the ``max_phase_irregularity_factor``
confidence term of :class:`~spindoctor.nav_technique.nav_technique_body_blob.BodyBlobNav` (see
:doc:`dev_guide_navigation_models_body`).  The calibration tooling in
``util/calibration/`` renders its simulated irregular moons as lumpy meshes at relief
amplitudes derived from these same residual-over-radius ratios, and its regular moons
as smooth ellipsoids, so the ``max_phase_irregularity_factor`` term is fitted on every
blob row, with the regular moons' factors sitting near zero.  The sim-anchored confidence coefficients and this table form
one system: a developer who revises a residual here should re-run that calibration
(see ``util/calibration/README.md``).

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
2. If a value cannot be sourced from a fetched document, leave the key out
   of the entry and write
   ``'PLACEHOLDER — no source found, calibration pending'`` as its
   ``_sources`` entry. The body's built-in profile takes the place of a value
   that is left out, and a body without a built-in profile uses the generic one.
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
- ``PLACEHOLDER`` is allowed only for a field the entry leaves out.
- The packaged table parses into its type, with an entry for every body in the
  file.

The same validation pattern extends to per-camera ``noise`` /
``mag_offset`` blocks in ``config_4N0_inst_*.yaml`` and to any new entries
added to ``config_3N0_*_rings.yaml``. Existing ring-catalog values are
grandfathered (they were curated by orbit-fitting astronomers and the
catalogs document their pedigree in the file header) — only *new*
additions need explicit ``_sources`` entries.

Adding a new tunable
====================

When a new YAML knob is added:

1. Pick the file whose section it belongs to (e.g. a new ``bodies`` key goes
   in ``config_040_bodies.yaml``). If no existing file fits, allocate a new
   numeric prefix per the layout table above.
2. Add the key with a sensible default and a one-line YAML comment that
   identifies the consumer.
3. Document the key on the consumer's dev-guide page (for example,
   :doc:`dev_guide_navigation_models_body` lists every key under
   ``bodies``). The page lists name, type, default, units, and consumer.
4. If the key represents static data (a measured constant rather than a
   knob), add the ``_sources`` entry per the citation discipline above.
5. Add or extend a unit test under ``tests/spindoctor/config_files/`` that asserts
   the loader exposes the key with the expected default.
