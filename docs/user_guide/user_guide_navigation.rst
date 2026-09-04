================
Image Navigation
================

Introduction
============

SpinDoctor is a spacecraft image navigation system designed to analyze images from various space missions and determine precise positional offsets. This guide explains how to use the primary command-line interface exposed by the ``sd_offset`` script to navigate images and generate results, and how to invoke the cloud-tasks variant for queue-driven processing.

Purpose of the System
---------------------

The primary purpose of SpinDoctor is to determine the precise pointing of spacecraft instruments by comparing the observed images with theoretical models of what should appear in the field of view. This process, known as "navigation," is crucial for:

1. Validating and correcting spacecraft pointing information
2. Ensuring accurate scientific interpretations of the imagery
3. Creating properly annotated and labeled images for analysis
4. Supporting mission planning and operations

The system works by:

1. Reading spacecraft imagery and metadata
2. Generating theoretical models of stars, planets, moons, and rings
3. Correlating the observed features with the theoretical models
4. Calculating the offset between the expected and actual pointing
5. Producing annotated images and data files with the results

Supported Missions
------------------

SpinDoctor supports multiple instruments, organized by dataset names you will pass on the command line. Dataset names are case-insensitive and map to instrument-specific handlers. The complete set is:

* ``coiss`` and ``coiss_pds3`` — Cassini Imaging Science Subsystem (all volumes) — :doc:`instruments/cassini_iss`
* ``coiss_cruise`` and ``coiss_cruise_pds3`` — Cassini Imaging Science Subsystem (Cruise volumes 1001-1009) — :doc:`instruments/cassini_iss`
* ``coiss_saturn`` and ``coiss_saturn_pds3`` — Cassini Imaging Science Subsystem (Saturn volumes 2001-2116) — :doc:`instruments/cassini_iss`
* ``gossi`` and ``gossi_pds3`` — Galileo Solid State Imager — :doc:`instruments/galileo_ssi`
* ``nhlorri`` and ``nhlorri_pds3`` — New Horizons Long Range Reconnaissance Imager — :doc:`instruments/newhorizons_lorri`
* ``vgiss`` and ``vgiss_pds3`` — Voyager Imaging Science Subsystem — :doc:`instruments/voyager_iss`
* ``sim`` — simulated images (see :doc:`user_guide_simulated_images`)

Each instrument's chapter carries the volumes it covers, which product is navigated, the image-name forms accepted, the units and thresholds it is judged against, and everything else that is true of that instrument and not of another. The shared chapters describe the mechanisms; the instrument chapters carry the values.

Installation and Setup
======================

See :doc:`/introduction_overview` for package installation with ``pip`` or
``pipx``.

Environment Setup
-----------------

In addition to installing the package, the following external resources are
needed at runtime.

**SPICE kernels.**  Download the SPICE kernels required for your mission and
set ``SPICE_PATH`` to the directory that contains them:

.. code-block:: bash

   export SPICE_PATH=/path/to/your/spice/kernels

**PDS3 holdings.**  For PDS3 datasets (all currently supported missions), set
``PDS3_HOLDINGS_DIR`` to the root of a PDS3 holdings tree (or pass
``--pds3-holdings-root`` on the command line):

.. code-block:: bash

   export PDS3_HOLDINGS_DIR=/path/to/your/pds3/data

The holdings tree follows the layout used by the PDS Ring-Moon Systems Node::

   $PDS3_HOLDINGS_DIR/
       volumes/
           <volume_set>/
               <volume>/
                   <data directories>/
       metadata/
           <volume_set>/
               <volume>/
                   <volume>_index.lbl
                   <volume>_index.tab

Remote holdings are supported: ``PDS3_HOLDINGS_DIR`` and
``--pds3-holdings-root`` accept any URL understood by ``filecache.FCPath``
(for example ``https://pds-rings.seti.org/holdings``).

Configuration System
====================

SpinDoctor uses a hierarchical YAML-based configuration system. For detailed
information about the configuration system, including its structure, default
YAML files, and how to override settings using configuration files and
command-line options, see :doc:`/introduction_configuration`.

Command-Line Interface
======================

Basic Usage
-----------

The main entry point for SpinDoctor is the ``sd_offset`` script installed via ``pyproject.toml``. The basic syntax is:

.. code-block:: bash

   sd_offset DATASET_NAME [options]

Where ``DATASET_NAME`` is one of the supported names listed in the "Supported Missions" section. Names are case-insensitive (for example, ``COISS`` and ``coiss`` are equivalent).

Command-Line Arguments
----------------------

The command-line interface groups options by purpose. Environment options control configuration sources and output roots. Navigation options select which models or techniques to run. Output options determine whether to write artifacts locally or to produce a cloud-tasks description instead of processing. Dataset selection options are provided by each dataset type: PDS3 datasets expose volume and image filters. A single profiling toggle is available for performance analysis.

Environment options
^^^^^^^^^^^^^^^^^^^

* ``--config-file PATH`` (repeatable): one or more configuration file paths to
  override defaults. See :doc:`/introduction_configuration` for details.

* ``--pds3-holdings-root PATH``: root directory or URL for PDS3 holdings,
  overriding both the ``PDS3_HOLDINGS_DIR`` environment variable and any
  corresponding configuration setting.

* ``--nav-results-root PATH``: root directory or URL where navigation results
  will be written, overriding both the ``NAV_RESULTS_ROOT`` environment variable
  and any corresponding configuration setting.

* ``--results-index-db URL``: connection URL of a results index (a ``sqlite:``
  URL naming a local path, or a ``postgresql+psycopg:`` URL naming a server),
  overriding both the ``NAV_RESULTS_INDEX_DB`` environment variable and any
  corresponding configuration setting. The results-file selection filters below
  are then answered from the index's rows, and the results tree is not read.
  Pass ``--results-index-db none`` to name no index, and so read the tree, even
  when a URL is set in
  the environment or a configuration file; the opt-out is that word exactly, in
  lower case, with any surrounding spaces ignored, since any other non-empty
  value is read as the URL of an index. A value that is empty, or nothing but
  spaces, is refused: it is neither a connection URL nor the way to name no
  index, so the run stops and names the setting that carries it.

Navigation options
^^^^^^^^^^^^^^^^^^

* ``--nav-models LIST``: a comma-separated glob-pattern list selecting which
  ``NavModel`` instances run.  Names follow the ``stars`` /
  ``body:NAME`` / ``rings:PLANET`` convention.  Defaults to ``*``.  See
  :ref:`selecting-models-and-techniques` for the full syntax (globs,
  ``!`` exclusion, prefix-only shorthand).

* ``--nav-techniques LIST``: a comma-separated glob-pattern list selecting
  which registered ``NavTechnique`` subclasses run.  Defaults to ``*``.
  See :ref:`selecting-models-and-techniques` for the full syntax and the
  list of shipping technique class names.

Output options
^^^^^^^^^^^^^^

* ``--output-cloud-tasks-file PATH``: write a JSON file describing tasks for all selected images suitable for a cloud-tasks queue, and exit without performing navigation.
* ``--dry-run``: print the images that would be processed without performing navigation.
* ``--no-write-output-files``: perform navigation but do not write any output files.

Dataset selection (PDS3 datasets)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For PDS3 datasets (``coiss``, ``coiss_pds3``, ``coiss_cruise``, ``coiss_cruise_pds3``, ``coiss_saturn``, ``coiss_saturn_pds3``, ``gossi``, ``gossi_pds3``, ``nhlorri``, ``nhlorri_pds3``, ``vgiss``, ``vgiss_pds3``), the following options control which images are selected. All filters combine with logical AND, and explicit lists restrict the search domain before range filters to improve performance.

* ``img_name`` (positional, repeatable): specific image name(s) to process.
* ``--first-image-num N``: minimum image number (inclusive).
* ``--last-image-num N``: maximum image number (inclusive). Voyager Flight
  Data Subsystem (FDS) counts restart per spacecraft/encounter, so for Voyager
  a number range can match frames from more than one encounter; combine it
  with the volume options to bound the selection.
* ``--volumes NAME[,NAME...]`` (repeatable): one or more complete PDS3 volume names; you may pass comma-separated values or specify the option multiple times.
* ``--first-volume NAME``: starting PDS3 volume; only that volume and chronologically later ones are processed.
* ``--last-volume NAME``: ending PDS3 volume; only that volume and chronologically earlier ones are processed.
* ``--image-filespec-csv FILE`` (repeatable): CSV file(s) containing PDS3 file specifications; files must include a header column named ``Primary File Spec`` or ``primaryfilespec``.
* ``--image-file-list FILE`` (repeatable): file(s) containing file specifications or names, one per line; lines beginning with ``#`` are ignored.
* ``--choose-random-images N``: choose a random subset of N images, uniformly
  distributed across all images (in every selected volume) that meet the other
  criteria; the selected images are yielded in random order.
* ``--has-offset-file`` / ``--has-no-offset-file``: only images whose offset
  metadata file (``*_metadata.json`` under the navigation results root) already
  exists / does not exist.
* ``--has-offset-error`` / ``--has-no-offset-error``: only images whose offset
  metadata file exists and records / does not record a fatal error (``status``
  of ``error``). Every error filter asks what a document records, so each one
  requires the document to exist: an image nothing has been written for records
  no error, and ``--has-no-offset-file`` is what selects it. So
  ``--has-offset-file --has-no-offset-error`` is how a run asks for the images
  whose navigation reached a result rather than dying before it could read the
  image. What it selects is stated exactly below, along with the answers a
  results index is known to give differently from the results tree.
* ``--has-offset-spice-error`` / ``--has-offset-nonspice-error``: like
  ``--has-offset-error``, but restricted to fatal errors caused by / not caused
  by missing SPICE data.

An error filter asks what a document records about its image, and reads it from
the per-image facts the document yields: the values a results index holds in its
columns, so a document is narrowed on exactly what a row is narrowed on. An
offset metadata file that cannot be read, does not parse as JSON, does not parse
to a JSON object, or parses to a JSON object that is not a navigation document of
the current metadata schema yields no such facts, and satisfies no error filter,
positive or negative: what it records is unknown rather than known to be an
outcome. A document written to an earlier metadata schema is one of those, so a
results root holding nothing else answers every error filter, including
``--has-offset-spice-error``, with no image at all; re-navigating those images
rewrites their documents to the current schema. Such a file is still a file that
exists, so the presence filters count it, ``--has-offset-file`` selects its image
and ``--has-no-offset-file`` passes it over. When a run ends, its log says how
many of the candidates it read yielded no facts and names one of them with the
reason, so a selection short for this reason says so rather than only coming back
smaller than expected.

Two questions cover all six flags, and they are asked at different moments
because they need different things.

Which images the results root holds a document for opens no document at all, and
a run selecting images that *have* one --- ``--has-offset-file``, and every error
filter, since each of those reads what a document records --- asks it once, when
the enumeration starts, by listing the volumes it selected. That costs one
listing per directory rather than one read per image, and testing a candidate
image against the answer afterwards is a set lookup that costs nothing.

A run selecting images that have *none* --- ``--has-no-offset-file`` --- asks the
same question about the candidate images themselves, in batches, as the
enumeration offers them. It keeps the images the results root holds nothing for,
so every entry a listing of a whole volume produced would be an entry to reject
from: a run whose other constraints name ten images would pay for fifty thousand
of them to answer about ten. Asking about the candidates instead costs a check
per candidate on a local results root, where a check is a system call, and one
listing per directory on a cloud one, where it is a paid round trip --- and on a
cloud root one listing serves every batch of the run rather than being taken
again for each.

What each document records is read from the per-image facts, in batches, as the
enumeration offers its candidates. An error filter has to read a document, and
which documents to read is the set of candidate images, which the other
selection constraints decide and which is not known when the enumeration starts.
So a batch of candidates names its images and asks about those. A run whose
other constraints keep one image in a hundred therefore reads a hundredth of the
documents, rather than every document under the volumes it selected --- on a
cloud results root, one paid download apiece. Only images that have a document
are ever asked about, since the listing has already excluded the rest, which is
also what makes every error filter keep only images that have one.

A document read to answer an error filter is not read a second time. Every
program that processes an image reads one navigation record for it, and a run of
``sd_backplanes`` or ``sd_mosaic`` that names an error filter has already read
that record while it was deciding what to select: the record travels with the
image, and the per-image stage uses it rather than opening the document again.
On a cloud results root that is a download saved for every image the run goes on
to process, not merely a second look at a local copy --- the enumeration and the
per-image stage keep separate download caches, so neither ever sees the other's
files.

Two consequences follow, and both are properties of the run rather than of any
one image. What such a run applies is what the document said when the selection
was made, so a document rewritten or deleted while the run is going is not
noticed for an image already selected; the same is true of any run for the
window between its listing and its per-image stage, and this narrows that window
rather than opening a new one. And the reasons that describe failing to read a
document --- reported as ``no_metadata`` for one that is not there, and as
``unreadable_metadata``, ``invalid_json`` or ``metadata_not_an_object`` for one
that will not read --- are not reported for an image whose record was carried,
because a record is carried only for a document that was read whole. Such an
image is reported under what its record records, or under nothing at all.

Given ``--results-index-db`` nothing is carried, and this saves nothing: the
filter narrows on columns and the per-image stage reads the columns it needs,
which are a different set, so the two stay separate reads of the index.

The listing taken when the enumeration starts covers the volumes the enumeration
selected and no others, and they are asked about one at a time. Reading the
results tree, a selected volume the results root has no directory for
contributes nothing: a volume nobody has navigated yet has no directory under
the results root, which is an ordinary state of a results tree rather than a
reason to stop. A directory that is there and will not be listed -- this user may
not read it, the share it lives on has gone away -- ends the run instead, because
a filter answering from what it could see would silently select images it has no
evidence about. Asking one volume at a time is what tells those apart: a single
request covering all of them would end at the first volume it could not read, and
every volume after it would go unasked.

A run that asks about its candidates needs no such restriction and applies none.
The images it is offered come from the volumes the enumeration walked, so naming
those volumes as well would narrow nothing, and a volume the results root has no
directory for still costs nothing and ends nothing: the images under it are
images the root holds no document for.

Given ``--results-index-db``, whichever of the two questions the flags call for
is answered from the index's rows, and the results tree is not read at all. A
results root for which the index has no completed ingest is refused rather than
answered, because absence of a row would otherwise read as "this image was never
navigated". The paragraphs below are the answers an index is known to give
differently from the tree; each of them is a property of what an ingest pass
could read and record rather than of the storage.

**A file the index has no row at all for reads as absent**, and
``--has-no-offset-file`` reads absence as "this image was never navigated", so
it selects that image again. One pass ends that way: one that could not retrieve
the file. Nothing is recorded for it deliberately -- a recorded row would be
skipped for as long as the file did not change, and a download that failed once
says nothing that will still be true next pass. Two other ways a file could go
unrecorded are not this, because neither leaves a completed pass behind it: an
ingest that cannot list a directory stops there, and one whose document the
index would not store stops there. A root the index holds a completed pass over
is a root every directory of which was listed and every document of which was
stored.

**A document rewritten in place, keeping the length and the modification time
it had before,** is one the ingest skips, because those two metrics are
everything a listing supplies about a file. Its row goes on recording what the
document before it said, so an error filter answers from that one however
recently the last pass finished. A tree restored by a copy that preserves times,
a document patched and stamped back from a sibling, and a backend reporting one
modification time for two writes all produce it; an ordinary re-navigation
writes a different length at a later time and does not. Running
``sd_results_index ingest --force`` over the root re-reads every document and is
what puts such a row right.

An index is also a snapshot of the tree as of the last ingest over that root,
with no staleness detection: an image navigated since is one the index does not
hold, so ``--has-no-offset-file`` selects it again, and a result file deleted
since is one the index still holds, so ``--has-offset-file`` selects an image
whose metadata file is gone. The run log says when the pass that filled the
index finished and how long ago that was, which is what says whether either
applies to this run. Run ``sd_results_index ingest`` to bring the index up to
date, or
pass ``--results-index-db none``, which names no index, for a run that must read
the tree. That age is
what decides the answer outside the paragraphs above; inside them it decides
nothing, since each of them survives a pass that finished a second ago.

An ingest that meets a directory it cannot list stops there, reports it as an
error, and completes no root from that point on, so no index a consumer reads
holds a root that was only partly walked. A results root that cannot be listed
at all is the other case and does not stop the pass: that root alone is left
unfinished, which every consumer already refuses, and the ingest goes on to the
next root. Fix what stopped the walk -- a directory permission, a share that was
not answering -- and run ``sd_results_index ingest`` again.

Miscellaneous
^^^^^^^^^^^^^

* ``--profile`` / ``--no-profile``: enable or disable runtime profiling (default is disabled).

Logging options
^^^^^^^^^^^^^^^

``sd_offset`` writes a main log reporting what the run is doing, and one log
per image carrying the detail of navigating it:

.. code-block:: text

   {log_root}/sd_offset/main_{timestamp}.log
   {log_root}/nav/{results_path_stub}_{timestamp}.log

``--log-root`` says where those go, defaulting to a ``logs`` directory under
the navigation results root. The main log goes to the terminal as well as a
file; image logs go to a file only, so the per-technique detail is on disk
rather than on screen unless ``--log-image-to-console`` asks for it.

The level of any one component can be raised or lowered on its own, which is
the usual way to investigate a single technique across many images:

.. code-block:: bash

   sd_offset coiss_saturn --volumes COISS_2001 \
       --log-level WARNING --log-level titan_haze=DEBUG

The full set of options, the component names, the configuration-file
equivalents and the precedence between them are in :doc:`user_guide_logging`.

Example Commands
----------------

To process a single Cassini image by specifying its name explicitly and using the default navigation technique:

.. code-block:: bash

   sd_offset coiss N1234567890

To process Voyager images within a single PDS3 volume:

.. code-block:: bash

   sd_offset vgiss --volumes VGISS_5101

To process a New Horizons image list found in a CSV from PDS, restricting the
run to the body-limb and ring-edge DT techniques:

.. code-block:: bash

   sd_offset nhlorri --image-filespec-csv /path/to/nhlorri.csv \
       --nav-techniques 'BodyLimbNav,RingEdgeNav'

To choose ten random Cassini images between two volumes and perform a dry run:

.. code-block:: bash

   sd_offset coiss --first-volume COISS_2001 --last-volume COISS_2010 --choose-random-images 10 --dry-run

To generate a cloud-tasks JSON file for images across two Voyager volumes without processing:

.. code-block:: bash

   sd_offset vgiss --volumes VGISS_5101 --volumes VGISS_5102 --output-cloud-tasks-file tasks.json

Cloud-tasks entry point
-----------------------

Queue-driven processing is supported by ``sd_offset_cloud_tasks``. This variant reads tasks from a queue and processes each batch of files described by the task payload. It accepts the same environment options used to derive configuration and results roots and does not include dataset selection flags because the task provides the list of files. Invoke it with:

.. code-block:: bash

   sd_offset_cloud_tasks [--config-file PATH] [--nav-results-root PATH]

Cloud-tasks JSON schema
^^^^^^^^^^^^^^^^^^^^^^^

The file produced by ``--output-cloud-tasks-file`` is a JSON array of task
objects. Each task is:

.. code-block:: json

    {
        "task_id": "<dataset_name>-<label_file_name>-<index>",
        "data": {
            "dataset_name": "<dataset_name>",
            "arguments": {
                "nav_models": ["body:*", "rings", "stars"],
                "nav_techniques": ["*"]
            },
            "files": [
                {
                    "image_file_url": "<path or URL to image file>",
                    "label_file_url": "<path or URL to label file>",
                    "results_path_stub": "<relative stub used to name outputs>",
                    "index_file_row": {"<column>": "<value>", "...": "..."},
                    "extra_params": {"<key>": "<value>"}
                }
            ]
        }
    }

Fields:

* ``task_id``: unique string identifier built from the dataset name, the
  first image's label filename, and the enumeration index.
* ``data.dataset_name``: one of the supported dataset names.
* ``data.arguments``: an object with optional keys ``nav_models`` and
  ``nav_techniques`` (each a list of strings, or ``null``).
* ``data.files``: one or more file descriptors with required fields
  ``image_file_url``, ``label_file_url``, and ``results_path_stub``, and
  optional ``index_file_row`` (metadata from the source index file, may be
  ``null``) and ``extra_params`` (arbitrary key/value dictionary forwarded
  to the task implementation; optional, may be ``null`` or omitted).

Whole-mission task files
^^^^^^^^^^^^^^^^^^^^^^^^

A mission is more than one ``sd_offset`` invocation to enumerate: Cassini ISS holds far more images than one queue should carry, and Voyager ISS is run one planetary encounter at a time. ``cloud_support/scripts/`` holds a generator per instrument that makes the selections and writes the files a queue is loaded from -- Galileo SSI and New Horizons LORRI as a single file each, Voyager ISS as one file per encounter, and Cassini ISS as consecutive groups of whole volumes holding roughly fifty thousand images apiece. Each generator requires the holdings root the cloud workers will read, because the image and label URLs a task carries are absolute and are fixed when the task is written.

These scripts are part of the repository rather than of the installed package. ``cloud_support/README.md`` describes them together with the compute-instance startup script and the job configuration they are used with.

.. _selecting-models-and-techniques:

Selecting models and techniques
===============================

``sd_offset`` runs every applicable navigation model and every feasible
navigation technique by default.  Two glob-pattern filters narrow that
set: ``--nav-models`` selects which :class:`~spindoctor.nav_model.nav_model.NavModel`
instances run, ``--nav-techniques`` selects which
:class:`~spindoctor.nav_technique.nav_technique.NavTechnique` subclasses run.
The same syntax applies in three places:

* ``sd_offset --nav-models LIST --nav-techniques LIST`` on the CLI.
* ``sd_offset_cloud_tasks`` task JSON, under
  ``data.arguments.nav_models`` and ``data.arguments.nav_techniques``
  (each a list of strings).
* :class:`~spindoctor.nav_orchestrator.orchestrator.NavOrchestrator` programmatic
  use, via the ``only_models=`` and ``only_techniques=`` keyword arguments.

The two filters share their pattern syntax; only the *names* they match
differ.  Filtering is purely additive over the existing registry — it does
not register new models or techniques, so an entry that does not exist on
this build of ``rms-spindoctor`` simply does not match.

Pattern syntax
--------------

Patterns are gitignore-style fnmatch globs evaluated against the
candidate name.  A single string or a comma-separated list (CLI) /
list-of-strings (JSON, Python) is accepted; the orchestrator splits on
commas and trims whitespace.

Inclusion patterns
^^^^^^^^^^^^^^^^^^

* A literal name matches that name only:
  ``BodyLimbNav`` matches the technique class
  :class:`~spindoctor.nav_technique.nav_technique_body_limb.BodyLimbNav` and
  nothing else.
* ``*`` matches any sequence of characters; ``?`` matches a single
  character; ``[abc]`` matches any character from the set.  Standard
  Python ``fnmatch`` semantics apply.
* The default ``'*'`` matches every candidate.

Exclusion patterns
^^^^^^^^^^^^^^^^^^

* A leading ``!`` marks an *exclusion* pattern: matches against the
  remaining glob are removed from the result.
  ``--nav-techniques '!StarFieldFromCatalogNav'`` runs every registered
  technique except that one.
* When every pattern in the list begins with ``!`` (a pure-exclusion
  list), an implicit ``'*'`` inclusion is added so the result is
  "everything except the excluded names".  ``--nav-models '!body:MIMAS'``
  is therefore equivalent to ``--nav-models '*,!body:MIMAS'``.
* When at least one inclusion pattern is present, only the listed
  inclusions plus their non-excluded matches survive.
  ``'body:*,!body:MIMAS'`` runs every body model except Mimas.

Multiple patterns
^^^^^^^^^^^^^^^^^

* On the CLI, comma-separate patterns inside a single argument:
  ``--nav-models 'body:MIMAS,rings:SATURN,stars'``.
* In JSON or Python, supply a list of strings:
  ``["body:MIMAS", "rings:SATURN", "stars"]``.
* The list is order-independent: a candidate name is kept iff it
  matches at least one inclusion pattern and no exclusion pattern.

Model names
-----------

The catalog-driven models register under these per-instance names:

* ``stars`` — :class:`~spindoctor.nav_model.stars.nav_model_stars.NavModelStars`
  (one instance per observation; no namespace).
* ``body:NAME`` —
  :class:`~spindoctor.nav_model.nav_model_body.NavModelBody` (one instance per
  body whose bounding box overlaps the extended FOV).  The ``NAME``
  portion is the upper-case SPICE body name
  (``body:MIMAS``, ``body:DIONE``, ``body:SATURN``).
* ``rings:PLANET`` —
  :class:`~spindoctor.nav_model.nav_model_rings.NavModelRings` (at most one
  instance, for the planet returned by ``obs.closest_planet``, and only
  when that planet has an entry in the ``rings.ring_features`` catalog;
  only Saturn's catalog is populated, so ``rings:SATURN`` is the only
  instance in practice).
* ``titan:TITAN`` —
  :class:`~spindoctor.nav_model.nav_model_titan.NavModelTitan` (one
  instance whenever Titan is inside the extended FOV).  Titan's opaque
  haze hides the surface, so instead of shape features the model emits the
  haze-envelope geometry that
  :class:`~spindoctor.nav_technique.nav_technique_titan_haze.TitanHazeNav`
  navigates from.  On a simulated image the equivalent model registers as
  ``titan_sim:TITAN``.

Two convenience normalizations apply to model patterns:

* The ``VALUE`` part of ``prefix:VALUE`` is upper-cased automatically,
  so ``body:saturn`` matches ``body:SATURN``.
* A bare prefix without a colon and without glob characters
  (``body``, ``rings``, ``titan``) is auto-expanded to ``prefix:*``, matching
  every namespaced model under that prefix.  ``stars`` (which has no
  namespace) continues to match itself directly.

Both normalizations preserve the leading ``!`` exclusion marker.
``--nav-models 'body'`` is therefore shorthand for "every body model";
``--nav-models '!body'`` excludes every body model.

Technique names
---------------

Techniques register under their class name.  The shipping concrete
techniques are:

* Body family —
  :class:`~spindoctor.nav_technique.nav_technique_body_disc.BodyDiscCorrelateNav`,
  :class:`~spindoctor.nav_technique.nav_technique_body_blob.BodyBlobNav`,
  :class:`~spindoctor.nav_technique.nav_technique_body_limb.BodyLimbNav`,
  :class:`~spindoctor.nav_technique.nav_technique_body_terminator.BodyTerminatorNav`.
* Ring family —
  :class:`~spindoctor.nav_technique.nav_technique_ring_annulus.RingAnnulusNav`,
  :class:`~spindoctor.nav_technique.nav_technique_ring_edge.RingEdgeNav`.
* Star family —
  :class:`~spindoctor.nav_technique.nav_technique_star_field.StarFieldFromCatalogNav`,
  :class:`~spindoctor.nav_technique.nav_technique_star_unique_match.StarUniqueMatchNav`,
  :class:`~spindoctor.nav_technique.nav_technique_star_refine.StarRefineNav`.
* Titan family —
  :class:`~spindoctor.nav_technique.nav_technique_titan_haze.TitanHazeNav`.

The star field matcher re-centroids each matched star with a point-spread-function fit
when the star is faint, and keeps the simpler brightness-weighted centroid when the star
is bright enough that its noise has already fallen below the PSF fit's residual bias.
This makes the star field the most accurate technique on a well-exposed field. The
brightness at which it switches is the configurable
``techniques.StarFieldFromCatalogNav.tuning.psf_refine_snr_max`` knob in
``config_510_techniques.yaml`` (set the whole step off with ``psf_refine_enabled: 0``).

:class:`~spindoctor.nav_technique.nav_technique_manual.NavTechniqueManual` is
the interactive driver and is not part of the autonomous registry; it
cannot be invoked by ``--nav-techniques``.

Multiple feasible techniques run in parallel and the orchestrator
combines their results via the ensemble step; ``--nav-techniques`` is
not a "pick one technique" knob — it restricts the candidate set the
orchestrator considers.

Examples
--------

.. code-block:: bash

   # Run every model and every technique (the default).
   sd_offset coiss N1234567890

   # Mimas only — drop every other body and the ring/star models.
   sd_offset coiss N1234567890 --nav-models 'body:MIMAS'

   # Every body, plus rings, but no stars.
   sd_offset coiss N1234567890 --nav-models 'body:*,rings'

   # Every model except Mimas (auto-expanded ``'*'`` inclusion).
   sd_offset coiss N1234567890 --nav-models '!body:MIMAS'

   # Two specific DT-based techniques only.
   sd_offset nhlorri LOR_0034851733 \
       --nav-techniques 'BodyLimbNav,RingEdgeNav'

   # Every technique except the catalog star matcher.
   sd_offset coiss N1234567890 \
       --nav-techniques '!StarFieldFromCatalogNav'

   # Body and ring families only (every body / ring technique, no stars).
   sd_offset coiss N1234567890 \
       --nav-techniques 'Body*,Ring*'

Inputs and Outputs
==================

Input Files
-----------

The primary input to SpinDoctor is spacecraft imagery. The system supports:

* PDS3 formatted image files (.IMG)
* Associated metadata (labels, SPICE kernels)

The system requires access to:

1. The raw image data
2. SPICE kernels for the appropriate mission and time period
3. Configuration settings (optional, defaults are provided)

Output Files
------------

SpinDoctor generates two types of output files:

Metadata Files (``*_metadata.json``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These JSON files contain the navigation results. The complete key-by-key
specification -- every key, its type, its presence rules, the rounding
policy, and one annotated example per document shape -- is the
:doc:`user_guide_metadata` chapter; in summary they include:

* ``observation`` — the image's identity: name, path, instrument, and
  ``camera`` (the camera that took it, e.g. ``NAC``). An image that fails to
  load has no observation to ask, so the navigator falls back to what the
  PDS3 index told it when the image was enumerated, recording ``camera``
  there; that needs no SPICE and never opens the image, so a frame whose
  navigation dies for want of a kernel is still attributed to its camera. An
  image navigated by explicit path rather than enumerated from an index has
  none. ``shutter_mode`` records
  the mode the image was taken in for an instrument whose label carries one
  (Cassini ISS reports ``BOTSIM`` when both cameras were exposed at once);
  instruments whose labels carry no such field omit it.
* The calculated pointing offset (dv, du)
* Uncertainty estimates (sigma_v, sigma_u)
* Confidence scores
* Metadata about the navigation process
* Status information (success, error, etc.)
* Technique-specific metadata (one ``per_technique`` entry per technique run,
  with each technique's offset, covariance, confidence, spurious / at-edge
  flags, and diagnostics)
* ``excluded_from_consensus`` — technique names the ensemble left out of the
  reported combine (outliers rejected against a multi-technique consensus, or
  the runner-up alternative on a conflicted result)
* ``pointing`` — the image's attitude as a C-matrix: ``cmatrix_original``, the
  uncorrected J2000-to-camera rotation the furnished kernels gave, and
  ``cmatrix``, the same rotation corrected by the navigated offset, alongside
  the SPICE ``camera_frame``, ``camera_frame_id``, and the ``ck_frame_id`` of
  the object a corrected C-kernel targets. Both matrices are nine row-major
  floats at the exposure midtime. ``cmatrix`` is present only when the
  navigation produced an offset and fitted no camera rotation
* ``times`` — the exposure window the attitude belongs to: ``start_et``,
  ``stop_et``, ``midtime_et``, ``exposure_s``, and the spacecraft-clock
  strings ``sclk_start``, ``sclk_midtime`` and ``sclk_stop``
* Timestamps

.. note::

   The ``confidence`` values and ``confidence_rank`` tiers are
   calibrated against *simulated* planted-truth recovery only
   (sim-anchored): on real images they carry
   the simulator's realism as an unquantified assumption and must not
   be read as probabilities of real-image accuracy.  The
   ``confidence_provisional: true`` field in every ``_metadata.json``
   that carries a navigation result marks this sim-anchored basis
   (image-load-error metadata has no navigation result block and
   therefore no such field).  The tiers additionally price statistical
   error, not unmodeled systematic error: a coherent model error the
   diagnostics cannot see -- a ring feature whose true orbit sits a few
   pixels off the catalog orbit, or a high-phase haze crescent biasing
   a centroid -- can be absorbed into a confident, gate-passing wrong
   offset, so a high tier is not evidence against that kind of error
   (see the ensemble chapter's confident-wrong section in the developer
   guide).

These files are also the input to the run-statistics tooling
(``sd_results_index`` / ``sd_stats_report``), which aggregates them into
success/failure, technique-usage, offset, and cross-technique-agreement
reports; see :doc:`user_guide_statistics`.

Summary PNG Files (``*_summary.png``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A navigation that reached a result writes a ``*_summary.png`` beside its
metadata document: one annotated picture showing what the navigator saw and
where it placed its model. The source image is composited with the merged model
overlay at the fitted offset, so a glance tells you whether the predicted
features land on the real ones. An image whose data could not be loaded at all
-- a frame outside the SPICE kernels' coverage, most often -- writes the
metadata document, with a ``status`` of ``error``, and no picture: nothing was
read to draw one from.

The base layer is the source image rendered in grayscale with a quantile
contrast stretch. The black point sits at a low quantile; the white point adapts
to how many bright pixels the frame carries, so a sparse star field or a small
body against dark sky is not blown out by a handful of saturated pixels. The
model overlay is drawn on top, shifted by the navigated ``(dv, du)`` offset so
each prediction sits where the fit says the real feature is.

The overlay carries one set of annotations per contributing model:

* **Stars** -- each predicted catalog star is boxed and labelled with its name,
  magnitude, and (when known) spectral class. Every star box is additionally
  contrast-stretched against its own local minimum and maximum, so a faint star
  only a few DN above a bright background stays visible inside its box even where
  the whole-frame stretch would bury it.
* **Bodies** -- each body in the field of view contributes its lit-limb outline,
  with the body name labelled by an arrow pointing to the limb.
* **Rings** -- each catalog ring edge is drawn as a polyline following the edge
  across the frame and labelled with the edge name. Ring points hidden behind
  the planet globe are dropped, so an edge stops at the planet limb rather than
  being painted across the disc.

A metadata text block is placed in one corner. It gives the image name, filter,
and exposure, the navigation status (and, on success, the techniques that
contributed to the consensus offset), and the fused confidence value with its
tier. The corner is chosen to avoid overlapping the other annotation labels,
breaking ties toward the darkest corner; a long technique list wraps within the
block, and the block is omitted on a frame too small to hold it. The summary
carries no scale bar or coordinate grid -- it is a visual check of the fit, not
a measurement product; the numeric offset and geometry live in the metadata JSON
and the backplanes.

.. figure:: _images/summary_png_example.png
   :width: 80%
   :align: center

   Summary PNG for the real navigated Cassini ISS frame ``N1484688342``, showing
   every annotation family at once. A crescent Mimas carries its lit-limb outline
   and a ``MIMAS`` label; catalog ring edges (Encke and Keeler) are drawn as
   labelled polylines across the bright ring band; roughly a dozen predicted
   stars are boxed and labelled with catalog name, magnitude, and spectral class,
   each box locally contrast-stretched so the faint stars stay visible; and the
   lower-left metadata block reports a successful fit at confidence 0.660.

The overlay assembly is described in
:doc:`/dev_guide/dev_guide_annotations`, and the ring-edge planet-occlusion trim
in :doc:`/dev_guide/dev_guide_navigation_models_ring`.

Interpreting Results
--------------------

The key information in the results is:

1. **Offset Values**: The u,v pixel offsets that should be applied to the nominal pointing to match the observed features
2. **Correlation Quality**: How well the models matched the observed features
3. **Annotations**: Identifications of specific features in the image
4. **Status**: Whether the navigation was successful, and if not, why

Simulated Images
================

SpinDoctor includes an image simulator used to test and validate the navigation
pipeline. It is not needed for navigating real data, but a simulated frame can be
navigated through the same pipeline by passing the ``sim`` dataset name and a path
to a YAML scene file:

.. code-block:: bash

   sd_offset sim /path/to/scene.yaml

The simulator, its scene formats, and the ``sd_create_simulated_image`` GUI are
documented for developers in the :doc:`/dev_guide/dev_guide_simulator` chapter.
See also :doc:`user_guide_simulated_images`.

Navigation Techniques
=====================

The autonomous-navigation pipeline runs every registered ``NavTechnique``
whose feasibility check passes on the surviving feature set, then combines
the per-technique offsets via the orchestrator's precision-weighted
ensemble.  Use ``--nav-techniques`` to restrict which techniques run; the
default ``*`` runs all of them.

The algorithmic detail (DT pipeline, Levenberg-Marquardt refinement,
information-matrix covariance) lives in
:doc:`/dev_guide/dev_guide_techniques` and
:doc:`/dev_guide/dev_guide_techniques_dt_fitting`; this page summarises
what each technique does and which scenes it applies to.

Implemented techniques
----------------------

``BodyLimbNav``
^^^^^^^^^^^^^^^

Translation fit on a body's lit limb.  Consumes every ``LIMB_ARC`` feature
emitted by ``NavModelBody``, concatenates their per-vertex polylines, and
runs a coarse-NCC plus Levenberg-Marquardt refinement against the image
edge-distance transform.  Tukey biweight reweighting rejects outlier
vertices; the M-estimator information matrix at the converged solution
yields the result's covariance.  Multi-body inputs sharpen the fit by
``sqrt(N_bodies)``.

Best for: scenes where one or more bodies show a visible limb arc
(typical Cassini ISS Mimas / Enceladus / Tethys / Dione / Rhea encounter
images).  Feasibility threshold: at least one limb arc with at least
30 surviving polyline vertices.

At very low phase (below about 15 degrees) the lit arc spans almost the
whole silhouette and the across-limb gradient that constrains the fit is
weak, so the fit can lock onto the wrong basin.  The technique detects that
mis-lock and marks the result spurious rather than emit a confident
multi-pixel offset, so a near-fully-lit single body is navigated by the disc
or other techniques instead of by a misleading limb fit.

``BodyTerminatorNav``
^^^^^^^^^^^^^^^^^^^^^

Same shape as ``BodyLimbNav`` on ``TERMINATOR_ARC`` features, with two
differences: each body's per-vertex sigmas collapse to one per-body scalar
(the body's mean sigma), and the confidence formula carries
phase-angle-factor and albedo-penalty terms.  Best for crescent
geometries where the terminator runs through bright, nearly-uniform
hemispheres.

``RingEdgeNav``
^^^^^^^^^^^^^^^

DT-based fit on every ``RING_EDGE`` feature.  Polarity prediction is
intentionally disabled today (the ring catalog does not yet flag
polarity_predictable, deferred work).  When every input edge is
straight-line the technique reports ``is_rank_1=True`` and returns an
honest rank-deficient covariance; the ensemble combine fuses it with any
orthogonal-axis result (a star, body limb, body blob) before declaring a
final answer.

Best for: close-range ring scenes.  For Saturn the rings model emits
per-edge features only below 25 km/px radial resolution and routes
everything coarser to ``RingAnnulusNav``, which a 131-frame
operator-audited head-to-head measured wrong on zero accepted answers
in every resolution band it was measured.  At fine resolution the
ring's many similar concentric ringlet edges resolve individually and
a shape-only edge fit can lock onto the wrong one, and below 25 km/px
neither ring technique is yet validated at scale, so a sub-25 km/px
ring-edge answer that no other technique corroborates warrants care.

``RingAnnulusNav``
^^^^^^^^^^^^^^^^^^

Pyramid-NCC fit on every ``RING_ANNULUS`` feature.  ``RING_ANNULUS``
features are emitted by the rings model in two regimes: when a curved
(non-straight) ring edge compresses radially to at most the per-planet
``feature_emission.ring_annulus.max_radial_px`` threshold in
``config_510_techniques.yaml`` (individual edges no longer separable;
a straight-line compressed edge stays a rank-1 ``RING_EDGE`` instead),
and when the scene's radial resolution is at or above the per-planet
km/px threshold -- 25 km/px for Saturn, so the whole Saturn system is
annulus-class at that resolution and coarser.  Under the system-level
gate every surviving ring collapses into a single composite annulus
per planet; below it a scene can emit a mix, with ``RING_EDGE``
features alongside one composite for the edges that compressed below
``max_radial_px``.  Multi-planet scenes
(rare) emit one ``RING_ANNULUS`` per ring system; the technique fuses
them via Z-buffer paint and runs one joint NCC.
``use_gradient='auto'`` self-selects raw vs gradient mode per image.

Best for: ring scenes at or above the per-planet km/px threshold: a
131-frame operator-audited head-to-head measured it wrong on zero
accepted answers at every resolution band, and its
rendered-brightness template disambiguates similar concentric edges
that a shape-only fit can confuse (distant Cassini ring views;
potential NHLORRI Pluto/Charon ring geometries).

``NavTechniqueManual``
^^^^^^^^^^^^^^^^^^^^^^

Interactive PyQt6 dialog that composes every template-bearing feature
into a single ext-FOV overlay and lets the operator pick the offset by
hand.  Not part of the autonomous registry; opt into it from the normal
``sd_offset`` driver with the ``--manual`` flag, which requires the
selection to resolve to exactly one image:

.. code-block:: bash

   echo W1521598221_1_CALIB > /tmp/img_list.txt
   sd_offset coiss --manual --image-file-list /tmp/img_list.txt

The driver loads the image, runs the orchestrator's ``prepare`` step
(image classifier + NavModels + features + reliability gate), opens the
dialog, and prints the chosen ``offset_dv_px`` / ``offset_du_px`` to
stdout.  Exit code is ``2`` if the dialog is cancelled or no
template-bearing features are available.  The dialog's **Save as
Library Entry...** button is the recommended path for adding a sidecar
to the operator-curated test image library; see
:doc:`/dev_guide/dev_guide_image_library`.

Programmatic equivalent (one obs in, ``NavTechniqueResult`` out):

.. code-block:: python

   from spindoctor.nav_technique import run_manual_nav

   result = run_manual_nav(obs)

Filtering examples
------------------

Run only the ring-edge technique:

.. code-block:: bash

   sd_offset coiss N1234567890 --nav-techniques RingEdgeNav

Run every technique except ``BodyTerminatorNav``:

.. code-block:: bash

   sd_offset coiss N1234567890 --nav-techniques '!BodyTerminatorNav'

Run both DT body techniques together:

.. code-block:: bash

   sd_offset coiss N1234567890 --nav-techniques 'BodyLimbNav,BodyTerminatorNav'

Output
------

Every technique that runs contributes one entry to
``NavResult.per_technique`` carrying the per-technique offset, 2x2
covariance, calibrated confidence, and a typed ``*Diagnostics``
dataclass.  The orchestrator's ensemble combine reconciles those
entries into a single ``NavResult.offset_px`` and ``confidence_rank``;
both numbers land in the per-image ``_metadata.json``.

Navigation Models
=================

A *navigation model* is SpinDoctor's prediction of what the image *should*
look like at the spacecraft's nominal pointing.  Four model families
ship out of the box: stars, planetary bodies, planetary rings, and
Titan's haze envelope.
Each contributes one or more *features* (typed predictions with their
own per-feature uncertainty) to the navigator.  You can restrict which
families run by passing ``--nav-models`` on the command line; valid
entries are ``stars``, ``rings``, ``titan`` (equivalently
``titan:TITAN``), and body-specific entries of the form ``body:NAME``
(glob patterns are allowed).

Star Navigation Model
---------------------

The star model builds a deduplicated catalog of stars expected to fall
inside the field of view, applies stellar aberration and proper motion
to bring each catalog position into the spacecraft frame at observation
time, and emits one feature per usable star.

**Catalog precedence.**  Catalogs are searched in the order configured
in ``config_030_stars.yaml`` under ``stars.catalogs`` (default
``[ucac4, tycho2, ybsc]``).  Stars present in more than one catalog are
deduplicated using the RA / DEC and V-magnitude thresholds in the same
file.

**Per-star detectability.**  Each star is gated by its catalog visual
magnitude against the per-observation limiting magnitude
``obs.star_max_usable_vmag()``, which depends on the per-instrument
sensitivity and the exposure time.  Stars fainter than the limiting
magnitude (or with no catalog magnitude) are dropped.

**Smear.**  When the spacecraft attitude rate is non-zero during the
exposure, stars smear into trails.  The model computes the per-image
smear vector from the SPICE pointing brackets and uses
``psfmodel.eval_rect(movement=...)`` to render a smear-aware kernel
when a downstream technique needs one.  Stars whose smear length
exceeds ``stars.max_smear`` are dropped (the centroid is unfittable).

**Body and ring conflicts.**  Each star's predicted pixel is checked
against an ``oops`` body intercept and a per-planet opaque ring
annulus (configured under ``stars.ring_occlusion_radii_km``).  Stars
that fall behind a body or inside an opaque ring annulus are tagged
with a ``BODY:`` or ``RING:`` conflict string and excluded from
matching.  Body intercepts win over ring intercepts.

**Configuration.**  Most user-tunable parameters live in
``config_030_stars.yaml``:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Key
     - Effect
   * - ``stars.catalogs``
     - Catalog search order; default ``[ucac4, tycho2, ybsc]``.
   * - ``stars.max_stars``
     - Maximum number of stars retained per image (default 100).
   * - ``stars.max_smear``
     - Smear length in pixels above which a star is dropped (default
       100).
   * - ``stars.min_vmag`` / ``stars.max_vmag``
     - Magnitude window applied to the per-instrument
       ``star_min_usable_vmag`` / ``star_max_usable_vmag`` floor.
   * - ``stars.proper_motion``
     - Apply proper motion at ``obs.midtime`` (default true).
   * - ``stars.stellar_aberration``
     - Apply stellar aberration (default true).
   * - ``stars.ring_occlusion_enabled``
     - Toggle the ring-annulus occlusion check (default true).
   * - ``stars.ring_occlusion_radii_km``
     - Per-planet list of opaque ``[inner_km, outer_km]`` annuli.

Body Navigation Model
---------------------

For every body whose predicted bounding box overlaps the extended
field of view, the body model renders an oversampled Lambert-shaded
silhouette, extracts the limb and terminator polylines, and emits a
mix of feature types depending on resolution, lighting, and shape
quality:

- ``LIMB_ARC`` — emitted when the limb position is well-determined
  (per-vertex normal sigma below the ``LIMB_ARC_MAX_UNCERTAINTY_PX``
  cap).  Carries a polyline of vertex coordinates and per-vertex
  anisotropic sigmas.
- ``BODY_BLOB`` — emitted instead of ``LIMB_ARC`` when the limb is too
  uncertain to fit but the predicted body diameter is above the
  body-specific blob threshold.  Carries only a centroid and bounding
  box.
- ``BODY_DISC`` — emitted alongside ``LIMB_ARC`` when the body fits
  well inside the FOV (overflow below 30 %, lit-and-visible fraction
  at least 40 %).  Carries the rendered template for full-disc
  correlation.
- ``TERMINATOR_ARC`` — emitted when the terminator polyline has at
  least 8 vertices and the phase-angle factor (``sin(phase_angle)``)
  is above 0.05.

**Per-body shape data.**  ``ellipsoid_residual_km``, ``crater_scale_km``,
``albedo_variation``, ``spice_orbital_residual_km``, and
``min_blob_diameter_px`` come from the static body-shape table.  These
quantities drive the per-vertex polyline sigmas and the BODY_BLOB
emission threshold.  For bodies absent from the table a conservative
generic-icy-moon profile is used.

**Configuration.**  ``config_040_bodies.yaml`` exposes:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Key
     - Effect
   * - ``bodies.min_bounding_box_area``
     - Minimum predicted body bbox area (px squared) below which
       silhouette rendering is skipped.
   * - ``bodies.oversample_edge_limit``
     - Anti-aliasing oversample limit for the silhouette render.
   * - ``bodies.oversample_maximum``
     - Hard cap on the per-axis oversample factor.
   * - ``bodies.use_lambert``
     - Use Lambert shading (default true) vs. flat-disc rendering.
   * - ``bodies.use_albedo`` / ``bodies.geometric_albedo``
     - Apply per-body geometric albedo when computing brightness.

The bodies considered for navigation are the planet returned by
``obs.closest_planet`` plus the satellites configured under the
top-level ``satellites.<PLANET>`` mapping in
``config_100_satellites.yaml``.

Ring Navigation Model
---------------------

The ring navigation model generates theoretical brightness profiles
for planetary ring edges and emits one feature per surviving edge.
Two top-level options in ``config_050_rings.yaml`` control whether
ring pixels in shadow are excluded from the model before navigation.

For each surviving ring feature the model emits one of:

- ``RING_EDGE`` — a per-vertex polyline of edge coordinates with
  per-vertex radial sigma derived from the catalog ``rms`` divided by
  the radial km-per-pixel scale.  When the polyline is straight
  (deviation from a best-fit line below 1 px) the ``is_straight_line``
  flag is set so techniques can handle the rank-1 covariance.
- ``RING_ANNULUS`` — a multi-edge composite template emitted when the
  surviving polyline compresses radially below 5 px (the edges are
  not separable at the image scale).

Per-edge feature definitions live in the per-planet ring files
(``config_300_jupiter_rings.yaml``, ``config_310_saturn_rings.yaml``,
``config_320_uranus_rings.yaml``, ``config_330_neptune_rings.yaml``)
under ``rings.ring_features.<PLANET>.features``.  Only the Saturn file
carries features today; the Jupiter, Uranus, and Neptune files are
empty placeholders.  See "Ring YAML configuration" in the developer
guide for the full schema.

Planet shadow removal
^^^^^^^^^^^^^^^^^^^^^

When a planet casts a shadow across part of its own ring system, those ring
arcs appear dark in the image. If the model still shows those arcs as bright,
the navigator will try to align a bright model against a dark image region,
which introduces a systematic pointing error.

The ``rings.remove_planet_shadow`` option (default ``true``) instructs the
ring model to zero out all ring pixels that fall inside the planet's own
shadow:

.. code-block:: yaml

   rings:
     remove_planet_shadow: true   # default

When active, the ring model logs the number of masked pixels at ``INFO`` level:

.. code-block::

   Planet shadow removal: 1284 pixel(s) inside SATURN shadow will be masked

If the shadow geometry cannot be computed for a particular observation (for
example, because the illumination geometry is degenerate), a warning is logged
and the full unmasked ring model is used instead. Navigation proceeds
normally; no output files are suppressed.

To disable shadow removal entirely -- for example, to compare navigation
quality with and without the mask -- set the option to ``false`` in a
``--config-file`` override:

.. code-block:: yaml

   rings:
     remove_planet_shadow: false

Body shadow removal (future)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``rings.remove_body_shadows`` option (default ``false``) is reserved for a
future enhancement that will remove ring pixels shadowed by moons. Setting it
to ``true`` has no effect in the current release.

.. code-block:: yaml

   rings:
     remove_body_shadows: false   # default; not yet implemented

Titan Navigation Model
----------------------

Titan's atmospheric haze is opaque at most wavelengths, so what a camera
sees is not the solid surface but the haze top: hundreds of kilometres
up, at an altitude that varies with wavelength, latitude, season, and
phase.  Fitting an ellipsoid limb to that edge is systematically wrong
rather than merely noisy, so Titan is navigated from a property of the
haze itself.

Absent clouds or visible surface features, a hazy atmosphere is
mirror-symmetric about the image-plane line through the body centre and
the sub-solar point.  The image shift perpendicular to that line
("cross-track") is the shift that maximises mirror symmetry; and because
the limb arc facing the Sun is close to circular, a circle fit with a
*free* radius to that arc gives the shift along the line
("along-track") without assuming any haze altitude.  The free radius is
what makes the method filter-independent: a haze top that sits higher in
blue than in red changes the fitted radius, not the fitted centre.  The
method is published as Hanson, French, Waugh, Barth and Anderson (2025),
*Geophysical Research Letters*, doi:10.1029/2024GL113415.

**What it produces.**  Whenever Titan is inside the extended field of
view the model emits a single ``TITAN_LIMB`` feature and
:class:`~spindoctor.nav_technique.nav_technique_titan_haze.TitanHazeNav`
measures the offset from it, on any instrument and any filter, with no
per-filter or per-phase training data.  The reported
uncertainty is deliberately *anisotropic*: the mirror-symmetry scan
localises the cross-track direction far more tightly than the circle fit
localises the along-track one, and the ensemble consumes that ellipse
rather than an averaged circle.

**Accuracy.**  Single-frame accuracy is **1 px or better cross-track and
3 px or better along-track**.  That bound comes from planted-truth
simulation (the 95th percentile of recovery error on the clean-scene
family of a 700-scene randomised campaign is 0.17 px cross-track and
0.82 px along-track; families with injected artifacts run wider) and is
confirmed
on real frames by an independent witness: over the Cassini validation
cohort, frames where a star technique locks independently give an
absolute per-frame anchor, and the haze fit disagrees with it by 0.99 px
rms cross-track and 1.50 px rms along-track over nine such pairs --
about 0.70 and 1.06 px of single-frame error once the anchor's own
uncertainty is removed.  A second anchor class corroborates the first:
when another moon shares the field of view, its own limb navigation
measures the same scene-wide offset, and it agrees with the haze fit at
2-sigma on 11 of the 12 cohort frames where both commit.  Repeat frames
of one target through one filter agree to 0.34 px cross-track and
0.33 px along-track.

Two consequences of the along-track figure are worth planning around.
An image whose only navigable content is Titan reports at most the
``medium`` confidence tier, because the honest along-track uncertainty
of a single quasi-circular feature exceeds the ``high`` tier's sigma
budget; adding a star field or a resolved moon to the frame is what
lifts it.  And a *small* Titan at *high* phase is the method's weak
regime: the sunward arc has its least support there, and it is where
essentially all of the along-track error lives.  Bodies whose apparent
solid radius is 40 px or more recover to 0.72 px along-track at the 95th
percentile across every phase; below 40 px above 60 degrees of phase the
same percentile is 3.0 px.

**Marginal frames are refused, not guessed.**  Three conditions score
the feature's reliability at exactly zero, after which the standard
per-feature-type gate removes it before any fit runs:

* the haze envelope, allowing for the full pointing search window, does
  not fit inside the detector;
* more than ``max_occluded_fraction`` of the envelope is hidden by a
  nearer moon or by the rings (the main rings are treated as opaque, so
  a Titan seen through the C ring or a gap is refused rather than fitted
  through ring stripes);
* the envelope is smaller than ``min_envelope_diameter_px`` across.

A frame refused this way ends with ``status_reason``
``all_features_gated``, and the per-image ``_metadata.json`` records the
measured envelope diameter and occluded fraction that produced the
refusal, so the cause is readable without re-running anything.

Frames that clear the gate can still be refused by the fit itself.  Each
of the seven fit gates below rejects a frame with its name recorded in
the technique's diagnostics, and a Titan-only frame whose fit is
rejected ends ``all_techniques_spurious``:

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Gate
     - Rejects the frame when
   * - ``valid_fraction``
     - Too much of the symmetry annulus is masked or off-frame to
       correlate.
   * - ``peak_score``
     - The best mirror-symmetry score is too low -- the haze is not
       symmetric enough to measure.
   * - ``second_peak``
     - A rival symmetry peak comes too close to the winner, so the axis
       could lock onto the wrong one.
   * - ``ray_yield``
     - Too few limb rays survive; the sunward limb is not detectable
       along enough of the arc.
   * - ``arc_inliers``
     - The robust circle fit rejected too many of the rays it was given.
   * - ``arc_radius``
     - The fitted radius is implausible for the body's known size.
   * - ``arc_residual``
     - The sunward limb departs too far from a circle.

Whichever way a frame ends, it is attributable: a committed offset, a
named fit gate, or a gated feature whose reliability breakdown says why.
Nothing produces a silent empty failure.

**Overlay.**  The summary PNG draws the predicted haze envelope circle,
the symmetry axis, the sunward arc sector, and a centre cross.  Because
annotations are composited at the navigated offset, the drawn circle
lands on the fitted position on a committed frame and stays at the SPICE
prediction when nothing was committed.  A feature below the reliability
gate is drawn dotted and labelled ``TITAN (low reliability)``.

**Configuration.**  ``config_060_titan.yaml`` exposes:

.. list-table::
   :header-rows: 1
   :widths: 42 58

   * - Key
     - Effect
   * - ``titan.atmosphere_height``
     - Haze envelope above the solid radius in km (default 700).  Bounds
       the search annulus and the ray windows; the fit itself assumes no
       haze altitude.
   * - ``titan.navigation.min_envelope_diameter_px``
     - Envelope diameter below which reliability is forced to zero.
   * - ``titan.navigation.max_occluded_fraction``
     - Occluded share of the envelope above which reliability is forced
       to zero.
   * - ``titan.navigation.ring_occlusion_radii_km``
     - ``[inner_km, outer_km]`` ring-plane range treated as opaque.
   * - ``titan.navigation.axis_min_offset_px``
     - Below this predicted-centre-to-sub-solar distance the disc is
       treated as rotationally symmetric and the axis search is skipped.
   * - ``titan.navigation.recenter_threshold_px``
     - Along-track shift above which the fit runs a second, recentred
       pass.
   * - ``titan.navigation.star_mask_vmag_limit`` /
       ``titan.navigation.star_mask_radius_px``
     - Brightness above which a catalog star is masked out of the fit,
       and the radius of each masked disc.
   * - ``titan.navigation.reliability_diameter_midpoint_px`` /
       ``titan.navigation.reliability_diameter_scale_px``
     - Midpoint and width of the sigmoid that turns apparent size into
       feature reliability.
   * - ``titan.navigation.surface_window_filters``
     - Filters that see through the haze to the surface.  Recorded as a
       diagnostic flag; the fit does not branch on it.
   * - ``titan.navigation.high_phase_deg``
     - Phase angle above which the emitted feature is flagged
       ``high_phase``, marking frames whose sunward arc carries its
       least support.
   * - ``titan.annotation.*``
     - Overlay styling: the dot spacing that marks a below-gate feature
       and the size of the center cross.
   * - ``titan.navigation.symmetry.*``
     - Cross-track scan: annulus extent, symmetry-angle refinement, the
       ``valid_fraction`` / ``peak_score`` / ``second_peak`` gate
       thresholds, and the reported cross-track sigma's scale and floor.
   * - ``titan.navigation.arc.*``
     - Along-track circle fit: sunward sector width and ray spacing,
       radial sampling, the limb-gradient signal-to-noise cut, the
       ``ray_yield`` / ``arc_inliers`` / ``arc_residual`` gate
       thresholds, the robust-fit tuning constant, and the reported
       along-track sigma's scale and floor.

The covariance model-error floor and the confidence-formula
coefficients live with the other techniques, under
``techniques.TitanHazeNav`` in ``config_510_techniques.yaml``.  The
developer guide documents every key's default and its measured
justification: see :doc:`/dev_guide/dev_guide_navigation_models_titan`
and :doc:`/dev_guide/dev_guide_techniques_titan_haze`.

Consolidating Navigation Outputs
================================

``sd_offset`` writes each image's results (the ``*_metadata.json`` offset file
and the ``*_summary.png`` preview) under the navigation results root, mirroring
the per-volume directory hierarchy of the input holdings. Browsing results
across many volumes therefore means descending a deep path tree. The
``sd_consolidate_metadata`` program copies the results for a selected set of
images into a single flat directory so they are easy to review or hand off.

It selects images with the same dataset arguments as ``sd_offset`` (positional
image names, ``--volumes``, image-number and file-list filters, and so on), and
for each selected image copies the requested product(s) out of the navigation
results root into the destination directory:

* ``--dest-dir PATH``: destination directory; every copied file lands directly
  here with no subdirectories, and missing parents are created on first write.
* ``--copy-metadata``: copy the per-image ``*_metadata.json`` files.
* ``--copy-png``: copy the per-image ``*_summary.png`` files.
* ``--copy-both``: copy both (equivalent to ``--copy-metadata --copy-png``).
* ``--index-prefix``: prefix each destination filename with a six-digit
  increasing index so the flat listing matches the iteration order.
* ``--overwrite``: overwrite destination files that already exist.
* ``--dry-run``: report what would be copied without copying anything.

It reads (and never modifies) the navigation results; the source root comes
from ``--nav-results-root``, the ``NAV_RESULTS_ROOT`` environment variable, or
the ``nav_results_root`` configuration value, and holdings and configuration
are resolved exactly as for ``sd_offset``.

For example, to gather the summary PNGs for one Cassini volume into a single
directory for a quick visual pass::

   sd_consolidate_metadata coiss --volumes COISS_2xxx/COISS_2116 \
       --copy-png --index-prefix --dest-dir /tmp/coiss_2116_summaries

Troubleshooting
===============

Common Issues
-------------

If SPICE kernels are missing, ensure that all required kernels are available and that environment variables and configuration files point to valid paths. For PDS3 inputs, verify the files conform to expected formats. In cases where no features are found or correlations are weak, check image quality, adjust the selected models or techniques, or limit processing to images known to contain suitable features. Use ``--dry-run`` to validate selection criteria without performing full processing.

Getting Help
------------

If you encounter persistent issues:

Review logs for detailed errors, consult the developer documentation for architectural context, and provide the command line, log snippets, and representative input data when asking for support.
