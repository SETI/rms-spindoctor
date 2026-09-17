==========================================================
Reproducibility Envelope (Provenance)
==========================================================

Overview
========

:class:`~spindoctor.nav_orchestrator.provenance.Provenance` is the frozen dataclass attached to
every :class:`~spindoctor.nav_orchestrator.nav_result.NavResult` to record the exact runtime
state under which a navigation produced its outputs. Two navigations with identical
inputs produce byte-identical
:class:`~spindoctor.nav_orchestrator.provenance.Provenance` *except* for
:attr:`~spindoctor.nav_orchestrator.provenance.Provenance.pipeline_run_iso8601`, which is
wall-clock by construction; regression-baseline comparison strips that field before
comparing.

Theory
======

The provenance envelope captures three independent kinds of state:

- **Code state.**  ``spindoctor_version`` and ``spindoctor_git_sha`` together identify the exact
  source code that ran.
- **External-data state.**  ``spice_kernels`` lists every SPICE kernel actually loaded;
  ``static_data_hashes`` sha256-hashes every YAML in
  ``src/spindoctor/config_files`` whose filename matches one of the
  ``_STATIC_DATA_PREFIXES`` (``config_220_`` for the body shape catalog, ``config_3``
  for ring catalogs, ``config_4`` for per-instrument blocks); ``star_catalogs``
  records each configured star catalog's name and its resolved path or URL (the same
  roots the catalog constructors read: ``UCAC4_PATH``, ``YBSC_PATH``, and
  ``SPICE_PATH``/``OOPS_RESOURCES`` for Tycho-2).  Catalog version numbers are not
  recorded because the catalog data carries none.
- **Configuration state.**  ``config_hash`` is a sha256 digest of the *resolved*
  config content (bundled defaults deep-merged with every applied override,
  serialized deterministically with sorted keys), and ``config_overrides`` lists the
  user/CLI override files (``nav_default_config.yaml`` or ``--config-file`` paths) in
  application order.  Together they pin exactly which configuration shaped the run.
- **Pipeline state.**  ``technique_names`` and ``extractor_names`` enumerate every
  registered :class:`~spindoctor.nav_technique.nav_technique.NavTechnique` and
  :class:`~spindoctor.nav_model.nav_model.NavModel` under the current process — so a regression
  run
  pinned to an old code revision but with a new technique registered records the
  difference in its provenance even when the outputs are otherwise byte-identical.

Restrictions and assumptions
----------------------------

- The static-data hash list is built once per ``navigate`` call by
  :func:`~spindoctor.nav_orchestrator.provenance.collect_provenance_metadata`. Comments and
  whitespace are included in the hashed bytes, so a YAML edit that only adds a comment
  changes the hash.
- The SPICE-kernel list is read live from ``spiceypy.ktotal`` /
  ``spiceypy.kdata``; the orchestrator does not coerce or sort the list itself
  (the dataclass sorts it for byte-identical output).
- The git SHA is read from ``git rev-parse HEAD`` plus a ``--is-dirty`` check; the
  reported value is ``'dirty'`` when the working tree has uncommitted changes and
  ``None`` when neither git nor a recorded SHA is available.
- The dataclass is frozen; the
  :attr:`~spindoctor.nav_orchestrator.provenance.Provenance.spice_kernel_count` derived field is
  populated in ``__post_init__`` from the kernel list length.

Sources of uncertainty
----------------------

The envelope reports no uncertainty. Every field is a deterministic readout of runtime
state at navigate time.

Configuration
=============

The envelope carries no YAML configuration of its own. The list of filename prefixes
counted as static data lives in module-level
``_STATIC_DATA_PREFIXES``; downstream callers that want a different set of YAML files
hashed must extend the list at module level.

Implementation
==============

Source file: ``src/spindoctor/nav_orchestrator/provenance.py`` —
:class:`~spindoctor.nav_orchestrator.provenance.Provenance`,
:class:`~spindoctor.nav_orchestrator.provenance.ProvenanceMetadata`, and
:func:`~spindoctor.nav_orchestrator.provenance.collect_provenance_metadata`.

Public surface (autodocumented at :doc:`/api_reference/api_nav_orchestrator`):

- :class:`~spindoctor.nav_orchestrator.provenance.Provenance` — frozen dataclass. Public fields:

  - :attr:`~spindoctor.nav_orchestrator.provenance.Provenance.spindoctor_version` — version string
    (e.g. ``'0.5.2'``).
  - :attr:`~spindoctor.nav_orchestrator.provenance.Provenance.image_et` — observation midtime ET
    (TDB seconds past J2000).
  - :attr:`~spindoctor.nav_orchestrator.provenance.Provenance.pipeline_run_iso8601` — UTC timestamp
    when the run began; excluded from regression-baseline comparison.
  - :attr:`~spindoctor.nav_orchestrator.provenance.Provenance.spindoctor_git_sha` — short git SHA,
    ``'dirty'``, or ``None``.
  - :attr:`~spindoctor.nav_orchestrator.provenance.Provenance.spice_kernels` — sorted tuple of
    SPICE kernel filenames.
  - :attr:`~spindoctor.nav_orchestrator.provenance.Provenance.static_data_hashes` — read-only
    mapping of YAML filename to sha256 hex digest.
  - :attr:`~spindoctor.nav_orchestrator.provenance.Provenance.technique_names` — sorted tuple of
    registered technique class names.
  - :attr:`~spindoctor.nav_orchestrator.provenance.Provenance.extractor_names` — sorted tuple of
    registered extractor class names.
  - :attr:`~spindoctor.nav_orchestrator.provenance.Provenance.config_hash` — sha256 hex digest
    of the fully-resolved config content, or ``None`` when unavailable.
  - :attr:`~spindoctor.nav_orchestrator.provenance.Provenance.config_overrides` — applied
    user/CLI override config paths in application order (not sorted; the merge is
    order-sensitive).
  - :attr:`~spindoctor.nav_orchestrator.provenance.Provenance.star_catalogs` — read-only
    mapping of configured star-catalog name to its resolved path or URL (``''`` when
    unresolvable).
  - :attr:`~spindoctor.nav_orchestrator.provenance.Provenance.spice_kernel_count` — derived;
    populated from ``len(spice_kernels)`` in ``__post_init__``.

- :class:`~spindoctor.nav_orchestrator.provenance.ProvenanceMetadata` — internal dataclass
  returned by
  :func:`~spindoctor.nav_orchestrator.provenance.collect_provenance_metadata` carrying the
  freshly-read git SHA, kernel list, static-data hash dict, resolved-config hash, applied
  config-override paths, and star-catalog paths.

- :func:`~spindoctor.nav_orchestrator.provenance.collect_provenance_metadata` — runs the live
  readouts. Called once per
  :meth:`~spindoctor.nav_orchestrator.orchestrator.NavOrchestrator.navigate`.

The dataclass enforces invariants in ``__post_init__``: every collection input is coerced
to its read-only / sorted form so two
:class:`~spindoctor.nav_orchestrator.provenance.Provenance` instances with the same inputs are
byte-identical for hash / serialization.

Examples
========

**Two navigations of the same image with the same SPICE kernels.**  An operator runs a
batch over a Cassini ISS image at two different wall-clock times. Both runs produce
:class:`~spindoctor.nav_orchestrator.provenance.Provenance` instances that differ only in
:attr:`~spindoctor.nav_orchestrator.provenance.Provenance.pipeline_run_iso8601`; every other
field is byte-identical. A regression-baseline comparator strips that field and confirms
the two outputs match.

**Code change that surfaces in provenance.**  An operator pulls a new commit that touches
:mod:`spindoctor.nav_technique.dt_fitting` and reruns navigation. The new
:class:`~spindoctor.nav_orchestrator.provenance.Provenance` carries a different
:attr:`~spindoctor.nav_orchestrator.provenance.Provenance.spindoctor_git_sha`; the rest of the
envelope is unchanged unless the per-technique result changed. The reviewer can correlate
output diffs with the SHA delta directly.

**Static-data change.**  An operator edits ``config_220_body_shape.yaml`` to refine
Mimas's ellipsoid residual. The next
:class:`~spindoctor.nav_orchestrator.provenance.Provenance` carries a different
:attr:`~spindoctor.nav_orchestrator.provenance.Provenance.static_data_hashes` entry for that
file; downstream regression baselines pinned to the old hash flag the difference and
require re-baselining.
