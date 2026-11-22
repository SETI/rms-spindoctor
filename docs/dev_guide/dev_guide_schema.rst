=========================
Schemas for YAML Files
=========================

SpinDoctor reads structured YAML files in three families: the navigator's
configuration, the simulator's instrument catalog, and simulated scenes. The
:mod:`spindoctor.schema` package is the one place that checks such a file and turns it
into typed Python objects. It has two halves:

- The **schema core** (:mod:`spindoctor.schema.core`) checks one mapping against a
  schema and returns an instance of the schema's class. A schema is a plain frozen
  dataclass, and pydantic does the checking.
- The **layer merge** (:mod:`spindoctor.schema.layers`) reads YAML files and merges
  them in order into the one mapping the core checks. It applies the file family's
  rule for null, and it records which file set each value.

The navigator's configuration loads through :class:`~spindoctor.config.config.Config`
(see :doc:`dev_guide_config_and_static_data`), and the simulator's instrument catalog
and scenes are checked by their own modules.

The package imports nothing else from ``spindoctor``. Its only outside imports are
pydantic, ``annotated_types``, ``ruamel.yaml``, and ``filecache``, and a test reads its
source to hold it to that. Both the simulator and the configuration code can
therefore build on it without importing each other.

Declaring a schema
==================

A schema is a frozen dataclass that sets ``__pydantic_config__ = SCHEMA_CONFIG``. Each
field's annotation gives its type and range, and a field with a default is optional:

.. code-block:: python

   from dataclasses import dataclass, field
   from typing import Annotated, Literal

   from spindoctor.schema import (
       SCHEMA_CONFIG, DigitInt, Float, NonNeg, Positive, Str, Unit, tagged
   )


   @dataclass(frozen=True)
   class BinnedNoise:
       __pydantic_config__ = SCHEMA_CONFIG
       read_noise_dn: NonNeg


   @dataclass(frozen=True)
   class Gaussian:
       __pydantic_config__ = SCHEMA_CONFIG
       kind: Literal['gaussian']
       sigma_px: Positive


   @dataclass(frozen=True)
   class Airy:
       __pydantic_config__ = SCHEMA_CONFIG
       kind: Literal['airy']
       first_zero_px: Positive


   @dataclass(frozen=True)
   class Detector:
       __pydantic_config__ = SCHEMA_CONFIG
       name: Str
       saturation_dn: Positive
       full_well_dn: Positive
       bloom_fraction: Unit = 0.0
       by_binning: dict[DigitInt, BinnedNoise] = field(default_factory=dict)
       psf: Annotated[Gaussian | Airy, tagged('kind')] = Gaussian('gaussian', 1.0)
       gain: Float = field(default=1.0, metadata={'boundary': 'truth'})

       def __post_init__(self) -> None:
           if self.saturation_dn > self.full_well_dn:
               raise ValueError('saturation_dn is above full_well_dn')

The rules behind this example:

- **Scalars use the package's aliases.** :data:`~spindoctor.schema.core.Float`,
  :data:`~spindoctor.schema.core.Int`, :data:`~spindoctor.schema.core.Bool`, and
  :data:`~spindoctor.schema.core.Str` are strict. A number written in quotes is a
  string and fails where a number belongs. A boolean is never a number, and a
  float must be finite. A YAML integer is accepted where a float belongs.
- **Ranges use the range aliases.** :data:`~spindoctor.schema.core.Positive`,
  :data:`~spindoctor.schema.core.NonNeg`, and :data:`~spindoctor.schema.core.Unit`
  are floats above zero, at or above zero, and in [0, 1].
  :data:`~spindoctor.schema.core.PositiveInt` and
  :data:`~spindoctor.schema.core.NonNegInt` are the integer versions. Any other range is
  an ``annotated_types`` constraint, such as ``Annotated[Float, Le(90)]``.
- **Integer keys use** :data:`~spindoctor.schema.core.DigitInt`. A scene that passes
  through JSON comes back with string keys, and ``DigitInt`` reads ``'2'`` as ``2``.
- **A path or URL uses** :data:`~spindoctor.schema.core.FCPath`.
- **A rule that ties fields together is** ``__post_init__`` on the class that owns
  them. It raises ``ValueError``, and its message becomes the problem's message.
- **An entry that comes in kinds is a tagged union.** Each member declares a
  ``Literal`` tag field, and :func:`~spindoctor.schema.core.tagged` marks the union.
  The tag picks the member, so a problem is reported against that member alone.
- **Markers for other code go in the field's metadata.** The core never reads them.
  :func:`~spindoctor.schema.metadata.walk_metadata` collects them for the whole schema,
  keyed by field path, such as ``by_binning.*.read_noise_dn``. The simulator can use
  them to sort keys by its information boundary, and the scene editor can read hints
  for its controls.

Unknown keys are always refused, so a misspelled key is an error. Strictness lives
on the fields, never on the dataclass, because a strict pydantic dataclass accepts
only an instance of itself and refuses a mapping. No class subclasses pydantic's
``BaseModel``; a test reads the source to keep it that way.

:func:`~spindoctor.schema.core.check_schema` refuses a schema that breaks these rules:
a dataclass that is not frozen or does not use ``SCHEMA_CONFIG``, a bare ``float``,
``int``, ``bool``, or ``str``, or an annotation outside the supported forms that the
core module's docstring lists. :func:`~spindoctor.schema.core.parse` runs the check
the first time it sees a class. A new form or convention needs a reason, written in
that docstring.

Parsing and error messages
==========================

:func:`~spindoctor.schema.core.parse` checks a mapping and returns the instance. When
the mapping breaks the schema it raises one
:class:`~spindoctor.schema.core.SchemaError`, which lists every problem at once, one
line each, as ``file: key.path: message``:

.. code-block:: text

   generic.yaml, coiss.yaml: missing required keys: ['name']
   coiss.yaml: bloom_fraction: expected a number, got str '0.8' (a number written so YAML reads it as a string)
   coiss.yaml: by_binning.2.read_noise_dn: expected a non-negative number, got -1
   coiss.yaml: psf.kind: expected one of 'gaussian', 'airy', got 'moffat'
   coiss.yaml: unknown keys: ['gian']

Missing keys and unknown keys of one mapping share a line. A position in a list is
written ``[n]``, as in ``bodies[2].radius``. Each issue is also available on the
error's ``issues`` attribute.

The file on each line comes from the ``origins`` that the layer merge records. A line
about a value gives the file that set it, and so does a line about an unknown key. A
line about a key that is absent, or about a whole mapping, gives every file that set
a value in that mapping. When ``parse`` is given no origins, every line gives its
``source`` argument.

Merging layers and nulls
========================

A :class:`~spindoctor.schema.layers.Layer` is one mapping and its label: a file name,
or a file name and an entry such as ``coiss.yaml: coiss_nac``.
:func:`~spindoctor.schema.layers.read_layer` reads a YAML file as a layer and drops
every key that begins with an underscore, which is how a file keeps notes such as
citations beside its values. :func:`~spindoctor.schema.layers.merge` applies layers
in order:

- Mappings merge key by key, so a later layer that sets one key keeps its siblings.
- Any other value replaces the earlier one. A list replaces the earlier list whole.
- A null follows the merge's :class:`~spindoctor.schema.layers.NullPolicy`.

.. list-table::
   :header-rows: 1
   :widths: 15 25 60

   * - Policy
     - Family
     - What a null does
   * - ``REMOVE``
     - Instrument catalog
     - Deletes the value an earlier layer set. A null with nothing beneath it to
       delete is an error.
   * - ``REFUSE``
     - Simulated scenes
     - Is always an error. A scene leaves a key out to take its default.

Under both policies, a null inside a list, or inside a mapping that no earlier layer
has, is an error. A merge collects every refused null in every layer before it
raises.

The navigator's configuration treats a null as a value: some keys use it to mean
"no limit". The package has no policy for that, so the navigator's loader keeps its
own merge.

The merge returns a :class:`~spindoctor.schema.layers.Merged` with the merged ``data``
and the ``origins`` of each value. Pass both to ``parse``:

.. code-block:: python

   from pathlib import Path

   from spindoctor.schema import NullPolicy, merge, parse, read_layer

   layers = [read_layer(Path('defaults.yaml')), read_layer(Path('user.yaml'))]
   merged = merge(layers, nulls=NullPolicy.REMOVE)
   detector = parse(Detector, merged.data, source='user.yaml', origins=merged.origins)

Inheriting from another entry
=============================

A file of named entries can let one entry inherit another, as the instrument catalog
does. :func:`~spindoctor.schema.layers.split_entries` turns a file layer into one
layer per entry, labeled with the file and the entry.
:func:`~spindoctor.schema.layers.resolve_bases` then merges each entry over the
chain of entries it inherits from. The far end of the chain merges first, so the
entry's own values win. The key that holds the base's name is dropped from the
result. Keys listed in ``not_inherited`` reach an entry only from its own layer. An
unknown base, a base that is not a string, and a chain that loops are each reported
with the entry's file and name.

API reference
=============

See :doc:`/api_reference/api_schema`.
