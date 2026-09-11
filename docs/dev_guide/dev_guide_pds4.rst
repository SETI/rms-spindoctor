=======================
PDS4 Bundle Generation
=======================

The :mod:`pds4` package builds PDS4-compliant bundles from SpinDoctor's per-image
navigation metadata and per-pixel backplanes. A bundle is the deliverable the
Ring-Moon Systems node ships to PDS for archive: one collection of data labels
(one per image), one collection of browse PNGs, plus the auxiliary collections
(context, document, xml_schema) and the bundle-level label that wires them
together. This chapter covers the bundle-generation driver, the per-dataset
extension points, the templated label workflow, and the output layout.

The user-facing CLI walkthrough lives at :doc:`/user_guide/user_guide_pds4_bundle`; this
chapter is the developer's reference.

Pipeline overview
=================

Bundle generation is a two-phase process driven by ``sd_create_bundle``:

1. **Per-image data labels.**  For each image in the input batch,
   :func:`~spindoctor.cli.pds4.bundle_data.generate_bundle_data_files` reads the
   ``_metadata.json`` produced by ``sd_offset`` and the
   ``_backplane_metadata.json`` produced by ``sd_backplanes``, populates a
   ``pdstemplate`` rendering context with per-image template variables, and
   writes the matching ``<image>_backplanes.lblx`` file (plus a copy of the
   browse PNG into the bundle's ``browse/`` tree). The backplane FITS file
   itself is copied (or symlinked, depending on the dataset's preference) from
   the backplane root into the bundle's ``data/`` tree.

2. **Collections + bundle assembly.**  After every per-image data label is in
   place, :func:`~spindoctor.cli.pds4.collections.generate_collection_files` walks the
   bundle's ``data/`` tree, collects every ``_backplanes.lblx`` it finds,
   sorts them by image name, writes the
   ``collection_data.csv`` inventory and the matching
   ``collection_data.lblx`` label, and renders the bundle's other collection
   labels (context, browse, document, xml_schema) plus the top-level
   ``bundle.lblx``.  :func:`~spindoctor.cli.pds4.collections.generate_global_index_files`
   writes the per-bundle ``global_index_bodies.lblx`` and
   ``global_index_rings.lblx`` summary tables.

The driver runs phase 1 once per image (fan-out friendly — each image is
independent) and phase 2 once at the end (sequential — needs every per-image
label in place before it can build the inventory).

Driver: ``sd_create_bundle``
=============================

``sd_create_bundle`` (``src/spindoctor/cli/sd_create_bundle.py``) is the per-image
phase-1 entry point. Like the other CLIs it takes a ``DATASET_NAME``, the
selection flags from the matching :class:`~spindoctor.dataset.dataset.DataSet`
subclass (``--pds3-holdings-root`` among them, for a PDS3 dataset), the standard
environment options (``--config-file``, ``--bundle-results-root``,
``--nav-results-root``, ``--backplane-results-root``), and walks every
selected image.

A separate ``--collections`` flag triggers phase 2 (collection + bundle
labels) without re-rendering per-image data labels. Operators typically run
``sd_create_bundle DATASET --image-list FOO --no-collections`` in parallel
across many shards, then once with ``--collections`` to assemble the bundle.

Cloud-tasks variant ``sd_create_bundle_cloud_tasks`` reads the same task JSON
schema as ``sd_offset_cloud_tasks`` (see :doc:`/user_guide/user_guide_navigation`) so the
same task queue can drive offset + backplane + bundle in three queue passes.

Per-dataset extension points
============================

PDS4 bundle generation is parameterized by the
:class:`~spindoctor.dataset.dataset.DataSet` subclass. The base class declares the
extension points as non-abstract methods that raise ``NotImplementedError`` —
a dataset that does not need PDS4 support can simply not override them, and
the bundle drivers refuse to run.

The full extension-point set:

- :meth:`~spindoctor.dataset.dataset.DataSet.pds4_bundle_template_dir` — absolute
  path to the directory of ``pdstemplate`` ``.lblx`` files this dataset uses.
  Lookups consult ``config.pds4.<dataset_name>.template_dir`` first; relative
  paths resolve under ``src/spindoctor/cli/pds4/templates/``. The reference Cassini ISS
  Saturn dataset uses ``cassini_iss_saturn_1.0/``.
- :meth:`~spindoctor.dataset.dataset.DataSet.pds4_bundle_name` — the bundle's
  external name (for example
  ``cassini_iss_saturn_backplanes_rsfrench2027``). The bundle root is
  ``<bundle_results_root>/<bundle_name>/``. Lookups consult
  ``config.pds4.<dataset_name>.bundle_name``.
- :meth:`~spindoctor.dataset.dataset.DataSet.pds4_bundle_path_for_image` — maps an
  image name to its position in the bundle's ``data/`` directory tree
  (typically a sharded path like ``1234xxxxxx/123456xxxx`` to keep per-leaf
  cardinality manageable on filesystems that struggle with very wide
  directories).
- :meth:`~spindoctor.dataset.dataset.DataSet.pds4_path_stub` — full per-image stub
  including the image name (e.g.
  ``1234xxxxxx/123456xxxx/1234567890w``). Builds the per-file paths under
  ``data/`` and ``browse/``.
- :meth:`~spindoctor.dataset.dataset.DataSet.pds4_lid_part_to_image_name` —
  inverse of the image-name transform baked into ``pds4_path_stub`` and the
  ``pds4_image_name_to_*`` builders. Each product is stored on disk under a
  filename whose stem is the LID part (e.g. ``1234567890w``); this hook lets
  the collection and global-index scanners recover the original image name from
  that stem and round-trip it back through the canonical LID builders instead
  of re-applying the transform.
- :meth:`~spindoctor.dataset.dataset.DataSet.pds4_image_name_to_browse_lid` /
  :meth:`~spindoctor.dataset.dataset.DataSet.pds4_image_name_to_browse_lidvid`
  — emit the browse-product Logical Identifier (LID) and LID + version
  (LIDVID) for the given image name. LIDs follow the PDS4 namespace
  convention ``urn:nasa:pds:<bundle>:browse:<image>``.
- :meth:`~spindoctor.dataset.dataset.DataSet.pds4_image_name_to_data_lid` /
  :meth:`~spindoctor.dataset.dataset.DataSet.pds4_image_name_to_data_lidvid` —
  same, for the data product (the backplane ``.lblx`` + ``.fits`` pair).
- :meth:`~spindoctor.dataset.dataset.DataSet.pds4_template_variables` — returns a
  dict of template variables consumed by the per-image
  ``data.lblx`` / ``browse.lblx`` templates. Inputs are the
  :class:`~spindoctor.dataset.dataset.ImageFile`, the navigation metadata dict
  parsed from ``<image>_metadata.json``, and the backplane metadata dict
  parsed from ``<image>_backplane_metadata.json``. The dataset is free to
  derive any per-image quantity the templates reference (target body,
  observer, mid-time, exposure, filters, navigation offset and confidence,
  per-backplane min/max/units, and so on).

Reference implementation:
:class:`~spindoctor.dataset.dataset_pds3_cassini_iss.DataSetPDS3CassiniISS`
overrides every PDS4 hook above and serves as the canonical worked example.
Voyager ISS (:class:`~spindoctor.dataset.dataset_pds3_voyager_iss.DataSetPDS3VoyagerISS`)
mirrors the same shape for an instrument with different image-naming
conventions.

The ``pds4`` config block
-------------------------

``src/spindoctor/config_files/config_950_pds4.yaml`` populates ``config.pds4`` with
per-dataset bundle metadata: the bundle name, the template directory name,
the LID namespace prefix, and any per-bundle template defaults the
``pds4_template_variables`` hook draws from. See
:doc:`dev_guide_config_and_static_data` for the loader contract; the file
is loaded by the standard numeric-prefix order at the ``9xx`` "downstream
products" tier.

Templated label workflow
========================

Labels are rendered via the ``pdstemplate`` library — a Python expression
language embedded in PDS4 ``.lblx`` files (XML). Each template carries
expressions that resolve against a dictionary of variables; the
``pds4_template_variables`` hook is the contract that connects per-image
metadata to the templates.

A typical render looks like:

.. code-block:: python

   import pdstemplate

   template_path = template_dir / 'data.lblx'
   variables = dataset.pds4_template_variables(
       image_file=image_file,
       nav_metadata=nav_metadata,
       backplane_metadata=backplane_metadata,
   )
   template = pdstemplate.Template(str(template_path))
   rendered = template.generate(variables)
   destination.write_text(rendered)

The ``pdstemplate`` library handles the XML escaping, the expression syntax,
and the per-template error reporting; consumers only supply the variable
dictionary and the destination path.

Template tree
-------------

Each dataset's template directory under ``src/spindoctor/cli/pds4/templates/`` contains the
shipping ``.lblx`` files. The Cassini-ISS-Saturn-1.0 set is the reference
layout:

::

   src/spindoctor/cli/pds4/templates/cassini_iss_saturn_1.0/
     bundle.lblx                              # top-level bundle label
     readme.txt                               # bundle-level README
     data.lblx                                # per-image backplane data label
     browse.lblx                              # per-image browse-product label
     collection_data.lblx                     # data-collection label (CSV inventory)
     collection_browse.lblx                   # browse-collection label
     collection_context.lblx                  # context-collection label
     collection_context.csv                   # context inventory (static)
     collection_document.lblx                 # document-collection label
     collection_document.csv                  # document inventory (static)
     collection_xml_schema.lblx               # schema-collection label
     collection_xml_schema.csv                # schema inventory (static)
     global_index_bodies.lblx                 # per-bundle bodies summary
     global_index_rings.lblx                  # per-bundle rings summary
     cassini-iss-saturn-backplanes-user-guide.lblx  # bundle user-guide doc

The static inventory CSVs are copied verbatim into the bundle; the
per-image and per-bundle ``.lblx`` files are rendered fresh on every run.

Output layout
=============

A finished bundle has the standard PDS4 directory shape:

::

   <bundle_results_root>/<bundle_name>/
     bundle.lblx
     readme.txt
     data/
       <pds4_bundle_path_for_image>/
         <image>_backplanes.lblx
         <image>_backplanes.fits             # copied from backplane_results_root
     browse/
       <pds4_bundle_path_for_image>/
         <image>_browse.lblx
         <image>_browse.png                  # copied from nav_results_root
     collection/
       data/
         collection_data.lblx
         collection_data.csv
       browse/
         collection_browse.lblx
         collection_browse.csv
       context/
         collection_context.lblx
         collection_context.csv              # static
       document/
         collection_document.lblx
         collection_document.csv             # static
         <user-guide doc>.lblx
       xml_schema/
         collection_xml_schema.lblx
         collection_xml_schema.csv           # static
     index/
       global_index_bodies.lblx
       global_index_bodies.csv
       global_index_rings.lblx
       global_index_rings.csv

Adding PDS4 support to a new dataset
====================================

The end-to-end checklist:

1. Override every ``pds4_*`` method on the new
   :class:`~spindoctor.dataset.dataset.DataSet` subclass. Use
   :class:`~spindoctor.dataset.dataset_pds3_cassini_iss.DataSetPDS3CassiniISS` as
   the reference implementation. The methods that absolutely must work
   are :meth:`~spindoctor.dataset.dataset.DataSet.pds4_bundle_template_dir`,
   :meth:`~spindoctor.dataset.dataset.DataSet.pds4_bundle_name`,
   :meth:`~spindoctor.dataset.dataset.DataSet.pds4_path_stub`, the four
   ``pds4_image_name_to_*_lid[vid]`` methods,
   :meth:`~spindoctor.dataset.dataset.DataSet.pds4_lid_part_to_image_name`
   (the inverse of ``pds4_path_stub``'s image-name transform), and
   :meth:`~spindoctor.dataset.dataset.DataSet.pds4_template_variables`.
2. Drop a per-dataset template directory under
   ``src/spindoctor/cli/pds4/templates/<dataset>_<version>/`` containing the ``.lblx``
   files and the static inventory CSVs. Copy from
   ``cassini_iss_saturn_1.0/`` and adapt the field set.
3. Add an entry under ``pds4.<dataset_name>:`` in
   ``config_950_pds4.yaml`` that points at the new template directory and
   sets the bundle name plus any per-bundle defaults the
   ``pds4_template_variables`` hook draws from.
4. Add an integration smoke test that renders one image through
   ``sd_create_bundle`` and asserts the resulting ``data.lblx`` validates
   against the PDS4 schema. The Cassini ISS test under
   ``tests/integration/`` is the pattern to follow.

API reference
=============

The :mod:`pds4` package has no autogenerated entry under
:doc:`/api_reference`; the module's public surface is the three
phase-1 / phase-2 entry points listed below, plus the
:class:`~spindoctor.dataset.dataset.DataSet` ``pds4_*`` extension hooks
documented above.

- :func:`~spindoctor.cli.pds4.bundle_data.generate_bundle_data_files` — phase 1, one image.
- :func:`~spindoctor.cli.pds4.collections.generate_collection_files` — phase 2, collection
  + bundle assembly.
- :func:`~spindoctor.cli.pds4.collections.generate_global_index_files` — per-bundle bodies
  / rings global indexes.
