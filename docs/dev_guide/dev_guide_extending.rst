====================
Extending the System
====================

Adding a new dataset
--------------------

To add a dataset, create a class in ``src/spindoctor/dataset/`` that inherits from
:class:`~spindoctor.dataset.dataset.DataSet` (or from
:class:`~spindoctor.dataset.dataset_pds3.DataSetPDS3` for archives). Implement
:meth:`~spindoctor.dataset.dataset.DataSet._img_name_valid`, the file-yielding
methods, and
:meth:`~spindoctor.dataset.dataset.DataSet.add_selection_arguments` to expose CLI
selection flags. Register the dataset in ``src/spindoctor/dataset/__init__.py`` so
it becomes available to the CLI.

Example:

.. code-block:: python

   from spindoctor.dataset.dataset_pds3 import DataSetPDS3

   class DataSetNewInstrument(DataSetPDS3):
       def __init__(self, *, config=None):
           super().__init__(config=config)

       @staticmethod
       def _img_name_valid(name: str) -> bool:
           return name.startswith("NEW") and name.endswith(".IMG")

The dataset will automatically be available to the CLI once registered.

PDS3 datasets additionally implement the static index-parsing hooks (private
methods, so they are shown here rather than in the API reference). A minimal
skeleton, using the Cassini ISS implementation
(``src/spindoctor/dataset/dataset_pds3_cassini_iss.py``) as the reference:

.. code-block:: python

   from pathlib import Path

   from spindoctor.dataset.dataset_pds3 import DataSetPDS3


   class DataSetPDS3NewInstrument(DataSetPDS3):
       _ALL_VOLUME_NAMES = tuple(f'NEWI_{n:04d}' for n in range(1, 12))
       _INDEX_COLUMNS = ('FILE_SPECIFICATION_NAME',)
       # Index columns naming the camera that took the image, in preference
       # order.  Read for every enumerated image and landing on
       # ImageFile.camera.  This needs no SPICE and never opens the image,
       # so an image whose load fails for want of a kernel is still
       # attributed to its camera.
       _INDEX_CAMERA_COLUMNS = ('INSTRUMENT_ID',)
       # Raw index value (upper-cased, stripped) -> the camera name the rest
       # of the system uses, i.e. the same name ObsInst.camera returns.  An
       # unmapped value is reported as unknown rather than passed through.
       _INDEX_CAMERA_MAP = {'NEWICAM': 'NEWICAM'}
       _VOLUMES_DIR_NAME = 'volumes'

       @staticmethod
       def _get_label_filespec_from_index(row):
           # Index row -> label filespec (the primary filespec).
           return row['FILE_SPECIFICATION_NAME'].replace('.IMG', '.LBL')

       @staticmethod
       def _get_image_filespec_from_label_filespec(label_filespec):
           # Index-time guess; see the label-pointer note below.
           return label_filespec.replace('.LBL', '.IMG')

       @staticmethod
       def _get_img_name_from_label_filespec(filespec):
           # 'data/<range>/NEW1234567890.LBL' -> 'NEW1234567890';
           # None = valid but not processed; ValueError = malformed.
           return filespec.rsplit('/', 1)[-1].split('.', 1)[0]

       @staticmethod
       def _img_name_valid(img_name):
           return img_name.upper().startswith('NEW')

       @staticmethod
       def _extract_img_number(img_name):
           return int(img_name[3:13])

       @staticmethod
       def _volset_and_volume(volume):
           return f'NEWI_xxxx/{volume}'

       @staticmethod
       def _volume_to_index(volume):
           return f'NEWI_xxxx/{volume}/{volume}_index.lbl'

       @staticmethod
       def _results_path_stub(volume, filespec):
           return str(Path(f'{volume}/{filespec}').with_suffix(''))

Two behaviors deserve attention:

* ``_get_image_filespec_from_label_filespec`` is an index-time *guess* (typically an
  extension swap). The definitive image filename is resolved lazily from the label's
  ``^IMAGE`` pointer when the image is first retrieved, via the
  :attr:`~spindoctor.dataset.dataset.ImageFile.image_url_resolver` installed on every
  yielded :class:`~spindoctor.dataset.dataset.ImageFile`; the guess only needs to be
  right often enough to serve as a display name and manifest entry.
* ``_IMG_NUM_MONOTONIC_ACROSS_VOLUMES`` (class attribute, default ``True``) tells the
  index scanner whether every image number in a volume exceeds every image number in
  all earlier volumes, which permits stopping a ``--last-image-num`` scan after the
  first volume that is entirely past the range. Set it to ``False`` when the
  instrument's image counter resets between volumes (Voyager Flight Data Subsystem
  (FDS) counts restart per spacecraft/encounter), at the cost of scanning every
  requested volume.

Implementing PDS4 bundle generation methods
-------------------------------------------

To support PDS4 bundle generation, datasets override the PDS4 hook methods on
:class:`~spindoctor.dataset.dataset.DataSet`. The base implementations are
non-abstract; each raises :exc:`NotImplementedError` so that a dataset that
cannot be packaged as a PDS4 bundle simply leaves them unimplemented.

* :meth:`~spindoctor.dataset.dataset.DataSet.pds4_bundle_template_dir`: returns the
  absolute path to the template directory for PDS4 label generation. If a
  relative name is provided in config, it should be resolved relative to
  ``src/spindoctor/cli/pds4/templates/``.
* :meth:`~spindoctor.dataset.dataset.DataSet.pds4_bundle_name`: returns the bundle
  name (e.g. ``"<instrument_name>_backplanes_rsfrench2027"``).
* :meth:`~spindoctor.dataset.dataset.DataSet.pds4_bundle_version`: returns the bundle's
  version (e.g. ``"1.0"``), which every product of the bundle carries.
* :meth:`~spindoctor.dataset.dataset.DataSet.pds4_information_model_version` and
  :meth:`~spindoctor.dataset.dataset.DataSet.pds4_schemas`: return the PDS4 information
  model version and the schema of each dictionary the bundle's labels declare.
* :meth:`~spindoctor.dataset.dataset.DataSet.pds4_bundle_path_for_image`: maps an
  image name to a bundle directory path (e.g.
  ``"1234xxxxxx/123456xxxx"``). This is a static method.
* :meth:`~spindoctor.dataset.dataset.DataSet.pds4_path_stub`: returns the full path
  stub including directory and filename prefix (e.g.
  ``"1234xxxxxx/123456xxxx/1234567890w"``).
* :meth:`~spindoctor.dataset.dataset.DataSet.pds4_template_variables`: returns a
  dictionary mapping template variable names to values for PDS4 label
  generation. This should extract values from navigation metadata, backplane
  metadata, and PDS3 index rows (if available).
* :meth:`~spindoctor.dataset.dataset.DataSet.pds4_image_name_to_data_lid`: converts
  an image name to a data product LID. Returns a full LID string (e.g.
  ``"urn:nasa:pds:<bundle_name>:data:<image_name>"``).
* :meth:`~spindoctor.dataset.dataset.DataSet.pds4_image_name_to_data_lidvid`:
  converts an image name to a data product LIDVID.
* :meth:`~spindoctor.dataset.dataset.DataSet.pds4_image_name_to_browse_lid`:
  converts an image name to a browse product LID.
* :meth:`~spindoctor.dataset.dataset.DataSet.pds4_image_name_to_browse_lidvid`:
  converts an image name to a browse product LIDVID.

For datasets that do not support PDS4 bundle generation, leaving these
methods unimplemented (so they inherit the base ``raise
NotImplementedError``) is the supported pattern. See
:class:`~spindoctor.dataset.dataset_pds3_cassini_iss.DataSetPDS3CassiniISS` for a
complete implementation example.

Adding a new instrument
-----------------------

To add an instrument, implement a subclass of
:class:`~spindoctor.obs.obs_snapshot_inst.ObsSnapshotInst` in ``src/spindoctor/obs/`` that
provides :meth:`~spindoctor.obs.obs_snapshot_inst.ObsSnapshotInst.from_file` and
any instrument-specific helpers. Update the instrument registry in
``src/spindoctor/obs/__init__.py`` so datasets can resolve the instrument class.

.. code-block:: python

   from spindoctor.obs.obs_snapshot_inst import ObsSnapshotInst
   from spindoctor.support.types import PathLike

   class ObsNewInstrument(ObsSnapshotInst):
       def __init__(self, obs, *, config=None, **kwargs):
           super().__init__(obs, config=config, **kwargs)

       @property
       def camera(self) -> str:
           # The camera that took this observation.  Return the oops
           # detector when the instrument has more than one camera; a
           # single-camera instrument returns its one name.  ObsInst
           # declares this abstract, so a subclass without it cannot be
           # instantiated.
           return 'NEWICAM'

       @classmethod
       def from_file(
           cls,
           path: PathLike,
           *,
           config=None,
           extfov_margin_vu=None,
           **kwargs,
       ):
           ...

An instrument also needs, in the same change:

1. **A configuration block** in
   ``src/spindoctor/config_files/config_4N0_inst_*.yaml``, carrying the
   instrument's units, saturation ceiling, missing-pixel marker, image-quality
   thresholds, extended-FOV margins, PSF parameters and rotation-fitting flag.
2. **A SPICE frame identity** in ``_frame_identity``
   (:mod:`spindoctor.support.cmatrix`) and a clock row in
   :data:`~spindoctor.spice_ids.CK_OBJECT_SCLK_ID`, without which no image of
   the instrument records a corrected attitude and none can reach a C-kernel.
   See :doc:`dev_guide_ck_kernels` for the full list of what a mission needs
   there.
3. **A chapter in each guide**: ``docs/user_guide/instruments/<name>.rst`` and
   ``docs/dev_guide/instruments/<name>.rst``, each copied from the
   ``_template.rst`` beside it and carrying every section that template
   declares. The chapter file name is the instrument's observation module name
   with the ``obs_inst_`` prefix removed, so ``obs_inst_new_instrument.py``
   requires ``new_instrument.rst``. Both indexes discover chapters by glob, so
   nothing else has to be edited -- and
   ``tests/spindoctor/test_instrument_chapters.py`` fails on a registered
   instrument with no chapter, or a chapter missing a template section.

Adding a new NavModel
---------------------

:class:`~spindoctor.nav_model.nav_model.NavModel` subclasses self-register via
``__init_subclass__``. To add a new predicted-scene generator:

1. Create a class in ``src/spindoctor/nav_model/`` inheriting from
   :class:`~spindoctor.nav_model.nav_model.NavModel` (or from one of the abstract
   bases such as
   :class:`~spindoctor.nav_model.nav_model_body_base.NavModelBodyBase`).
2. Implement :meth:`~spindoctor.nav_model.nav_model.NavModel.create_model`,
   :meth:`~spindoctor.nav_model.nav_model.NavModel.to_features`, and
   :meth:`~spindoctor.nav_model.nav_model.NavModel.to_annotations`.
3. Override :meth:`~spindoctor.nav_model.nav_model.NavModel.instances_for_obs` if
   the model auto-instantiates per-observation (one instance per body in
   FOV, one per planet with visible rings, one stars model). Subclasses
   that require operator parameters (simulated models populated from GUI
   JSON) inherit the empty default; the caller constructs them directly.

.. code-block:: python

   from spindoctor.nav_model.nav_model import NavModel

   class NavModelNewFeature(NavModel):
       def __init__(self, name, obs, *, config=None):
           super().__init__(name, obs, config=config)

       def create_model(self) -> None:
           ...

       def to_features(self, context) -> list[NavFeature]:
           ...

       def to_annotations(self, context) -> Annotations:
           ...

Adding a new NavTechnique
-------------------------

:class:`~spindoctor.nav_technique.nav_technique.NavTechnique` subclasses also
self-register via ``__init_subclass__``.

1. Create a class in ``src/spindoctor/nav_technique/`` inheriting from
   :class:`~spindoctor.nav_technique.nav_technique.NavTechnique`.
2. Set the class attributes
   :attr:`~spindoctor.nav_technique.nav_technique.NavTechnique.name`,
   :attr:`~spindoctor.nav_technique.nav_technique.NavTechnique.accepts_feature_types`,
   and (if relevant)
   :attr:`~spindoctor.nav_technique.nav_technique.NavTechnique.requires_prior`.
3. Implement :meth:`~spindoctor.nav_technique.nav_technique.NavTechnique.is_feasible`
   (must read feature metadata only, no pixels) and
   :meth:`~spindoctor.nav_technique.nav_technique.NavTechnique.navigate`.

.. code-block:: python

   from spindoctor.feature.feature_type import NavFeatureType
   from spindoctor.nav_technique.feasibility import NavFeasibilityReport
   from spindoctor.nav_technique.nav_technique import NavTechnique
   from spindoctor.nav_technique.technique_result import NavTechniqueResult

   class NavTechniqueNewMethod(NavTechnique):
       name = 'NavTechniqueNewMethod'
       accepts_feature_types = frozenset({NavFeatureType.STAR})

       def is_feasible(self, features):
           return NavFeasibilityReport(feasible=len(features) >= 1, reason='ok')

       def navigate(self, features, context) -> NavTechniqueResult:
           ...

The technique becomes visible to
:class:`~spindoctor.nav_orchestrator.orchestrator.NavOrchestrator` as soon as the
module is imported. Glob filters on the orchestrator (``only_techniques``)
let you exclude or single out a technique without modifying the registry.

Adding to the image library via the manual-nav dialog
-----------------------------------------------------

The interactive
:class:`~spindoctor.ui.manual_nav_dialog.ManualNavDialog` (built by
:class:`~spindoctor.nav_technique.nav_technique_manual.NavTechniqueManual`)
exposes a **Save as Library Entry...** button alongside the OK / Cancel
controls. Clicking it captures the current dv/du and writes a sidecar
seeded with the auto-fillable fields plus ``TODO_REPLACE_*``
placeholders for the operator-curated ones (scene_tags,
``primary_technique``, notes). The YAML helper lives in
:mod:`spindoctor.ui.library_entry`.

Operator workflow:

1. Open the manual-nav dialog on the candidate image and pick the
   offset by hand (or accept **Auto**).
2. Click **Save as Library Entry...**. The save-file dialog suggests
   ``<image_id>.yaml`` as the filename. Point it at the appropriate
   scene-class directory under
   ``tests/integration/image_library/images/<class>/``. A companion
   ``<image_id>.png`` capturing the red-image / green-model overlay
   at the chosen ``(dv, du)`` is dropped next to the YAML so future
   reviewers can see the scene at a glance; it is not consumed by any
   test.
3. Open the saved YAML and fill in every ``TODO_REPLACE_*`` value. An
   unedited template trips
   :func:`tests.integration.sidecar.load_sidecar` so CI fails loudly if
   you forget.
4. Run the structural-invariants test
   (``pytest tests/integration/test_image_library.py``); the per-image
   regression test (``test_autonomous_nav.py``) follows once
   ``PDS3_HOLDINGS_DIR`` is set.

See :doc:`dev_guide_image_library` for the sidecar schema and
the deeper rationale behind the curation policy.
