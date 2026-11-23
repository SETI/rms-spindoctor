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
(see :doc:`dev_guide_config_and_static_data`). The simulator's instrument catalog loads
through this package: its files merge under the ``REMOVE`` rule for null, and each
entry parses into a typed dataclass (see :ref:`sim-instrument-files`). So does each
artifact mode's map, in a scene or in the catalog. The rest of a scene is checked by the
simulator's own scene validator.

The package imports nothing else from ``spindoctor``. Beyond the standard library it
may import only pydantic, ``annotated_types``, ``ruamel.yaml``, and ``filecache``, and a
test reads its source to hold it to that. Both the simulator and the configuration code can
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
  :data:`~spindoctor.schema.core.NonNegInt` are the integer versions. Each states its
  own range when it refuses a value: "expected a number in [0, 1], got 2". Any other
  range is an ``annotated_types`` constraint, such as ``Annotated[Float, Le(90)]``,
  and its message gives the bound: "expected a number of at most 90, got 91".
- **Integer keys use** :data:`~spindoctor.schema.core.DigitInt`. A scene that passes
  through JSON comes back with string keys, and ``DigitInt`` reads ``'2'`` as ``2``.
  A key with a leading zero, such as ``'02'``, is refused, and so is a mapping that
  writes one key both as ``2`` and as ``'2'``, so no two keys can stand for one
  integer.
- **A path or URL uses** :data:`~spindoctor.schema.core.FCPath`.
- **A choice is a** ``Literal`` **or an** ``Enum`` **of strings.** A value outside it
  is refused as ``'cmos' is not one of ['ccd', 'vidicon']``. A choice of numbers is
  refused, because it would also accept ``True`` for ``1``.
- **A rule that ties fields together is** ``__post_init__`` on the class that owns
  them. It raises ``ValueError``, and its message becomes the problem's message, at
  the class's key path. To point at the key at fault instead, the rule raises
  :class:`~spindoctor.schema.errors.RuleError` with that key's path relative to the
  class, such as ``RuleError('saturation_dn is above full_well_dn',
  at='saturation_dn')``. The line then gives that key and the file that set it, so a
  mistake that several entries inherit from one base reads the same in each. A rule
  that compares keys passes all of them as ``reads``, such as
  ``reads=('saturation_dn', 'full_well_dn')``. Either key may come from either file, and
  the line gives the file that set or removed one of them last, since that is the file
  that broke the rule. Any other exception is treated as a bug in the rule and is raised
  as it is.
- **An alias is only for a key that is a Python keyword.** A file may write a key such
  as ``from``, which can't be a field name. The field takes another name and declares
  the key as its alias: ``from_: Annotated[Str, Field(alias='from')]``.
- **An entry that comes in kinds is a tagged union.** Each member declares a
  ``Literal`` tag field, and :func:`~spindoctor.schema.core.tagged` marks the union.
  The tag picks the member, so a problem is reported against that member alone. A
  union of two or more dataclasses must be tagged, and every member must declare the
  tag field as a ``Literal``.
- **Markers for other code go in the field's metadata.** The core never reads them.
  :func:`~spindoctor.schema.metadata.walk_metadata` collects them for the whole schema,
  keyed by field path, such as ``by_binning.*.read_noise_dn``. The simulator can use
  them to sort keys by its information boundary, and the scene editor can read hints
  for its controls. Two types at one path, such as two members of a tagged union,
  must mark the fields they share the same way.
- **A schema is built once, at import.** ``parse`` keeps the checked pydantic adapter
  of every class it has seen for the life of the process, so a class built on the fly
  for each call would be kept again on each call.

Unknown keys are always refused, so a misspelled key is an error. Strictness lives
on the fields, never on the dataclass, because a strict pydantic dataclass accepts
only an instance of itself and refuses a mapping. No class subclasses pydantic's
``BaseModel``; a test reads the source to keep it that way.

:func:`~spindoctor.schema.core.check_schema` refuses a schema that breaks these rules.
That covers a dataclass that is not frozen or does not use ``SCHEMA_CONFIG``, an untagged
union of dataclasses, and a choice that is not all strings. It also covers anything
that would read a value leniently: a bare ``float``, ``int``, ``bool``, ``str``, or
``FCPath``, a ``Strict(False)`` or ``AllowInfNan(True)`` marker, and a validator that
runs before the type check, on any type. A default that doesn't fit its field's
annotation, an ``InitVar``, and an annotation outside the supported forms that the core
module's docstring lists are refused too. :func:`~spindoctor.schema.core.parse` runs the
check the first time it sees a class. A new form or convention needs a reason, written
in that docstring.

Parsing and error messages
==========================

:func:`~spindoctor.schema.core.parse` checks a mapping and returns the instance. When
the mapping breaks the schema it raises one
:class:`~spindoctor.schema.errors.SchemaError`, which lists every problem at once, one
line each, as ``file: key.path: message``:

.. code-block:: text

   generic.yaml, coiss.yaml: missing required keys: ['name']
   coiss.yaml: bloom_fraction: expected a number, got str '0.8' (a number written so YAML reads it as a string)
   coiss.yaml: by_binning.2.read_noise_dn: expected a non-negative number, got -1
   coiss.yaml: psf.kind: 'moffat' is not one of ['airy', 'gaussian']
   coiss.yaml: unknown keys: ['gian']

Missing keys and unknown keys of one mapping share a line. A bad mapping key and a bad
value under it are two problems, on two lines. A position in a list is written
``[n]``, as in ``bodies[2].radius``. A list of the wrong length reads "expected a
list of exactly 2 items" when the schema allows one length, and "expected a list of at
most 3 items" when it allows a range. When a value fails every member of a union, the
members' expectations share a line, unless one member got further into the value. For
example, when a key takes either ``'all'`` or a list of names and the file gives a list
with one bad name, only the bad name is reported. Each issue is also available on the
error's ``issues`` attribute, as an :class:`~spindoctor.schema.errors.Issue`.

The file on each line comes from the ``origins`` that the layer merge records. A line
about a value gives the file that set it, and so does a line about an unknown key. A
line about a required key that a null removed gives the file with the null. A line
about any other absent key, or about a whole mapping, gives every file that set or
removed a value in that mapping. When ``parse`` is given no origins, every line gives
its ``source`` argument.

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
and the ``origins`` of each value. The origins also hold each key a null removed,
with the file that removed it. They are in the order of the layers. Pass both to
``parse``:

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
with the entry's file and name. The entry's other null problems are reported beside
a bad base, except a null that removes nothing, since the base that could not be read
may have set that key.

API reference
=============

See :doc:`/api_reference/api_schema`.
