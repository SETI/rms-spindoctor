"""The typed ``backplanes`` and ``pds4`` sections of the navigator configuration.

Both sections describe products the pipeline writes after navigation: the backplanes
each image gets, and the PDS4 bundle that publishes them.  Each is a frozen dataclass
parsed through the schema core (:func:`spindoctor.schema.parse`) each time a
configuration is installed, and every value lives in the packaged YAML.

The ``pds4`` section holds one entry per bundle, keyed by the name a dataset class gives
its bundle (:attr:`spindoctor.dataset.dataset.DataSet.PDS4_CONFIG_KEY`).  Its fields come
from the dataset registry, so the type to parse is :func:`pds4_section_type`, which adds
them.

The module imports nothing from the rest of ``spindoctor`` but the schema core, so the
dataset and product modules can import its types.
"""

import dataclasses
import functools
from dataclasses import dataclass

from spindoctor.schema import SCHEMA_CONFIG, Float, RuleError, Str

__all__ = [
    'BackplanesSection',
    'IncidenceEntry',
    'IncidenceIndex',
    'IndexColumnEntry',
    'Pds4Bundle',
    'Pds4Schema',
    'Pds4Section',
    'PlaneEntry',
    'PlaneIndex',
    'TargetEntry',
    'pds4_section_type',
]


@dataclass(frozen=True)
class IndexColumnEntry:
    """One column a global index table gives a statistic: its field name and description.

    Parameters:
        name: The field's name, as the table's header line and its label give it: a
            PDS4 dictionary attribute's name with the dictionary's prefix, as in
            ``geom:minimum_latitude``, where the column holds the quantity that attribute
            defines, and a name of its own otherwise.
        description: What the column's values are.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    name: Str
    description: Str


@dataclass(frozen=True)
class PlaneIndex:
    """The columns a global index table gives one plane's statistic.

    Every plane gets the columns of its least and its greatest value.  A plane of
    longitudes, whose statistic also records its range wrapped at zero, gets two more,
    where that range starts and where it ends.

    Parameters:
        data_type: The PDS4 data type of every column's values, as in ``ASCII_Real``.
        minimum: The column of the plane's least value.
        maximum: The column of the plane's greatest value.
        wrapped_minimum: The column of where the plane's range wrapped at zero starts.
        wrapped_maximum: The column of where the plane's range wrapped at zero ends.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    data_type: Str
    minimum: IndexColumnEntry
    maximum: IndexColumnEntry
    wrapped_minimum: IndexColumnEntry | None = None
    wrapped_maximum: IndexColumnEntry | None = None

    def __post_init__(self) -> None:
        """Refuse a wrapped range given only one end.

        A wrapped range is written as a pair of columns, so the two keys are set
        together or not at all.
        """
        if (self.wrapped_minimum is None) != (self.wrapped_maximum is None):
            given, absent = (
                ('wrapped_minimum', 'wrapped_maximum')
                if self.wrapped_maximum is None
                else ('wrapped_maximum', 'wrapped_minimum')
            )
            raise RuleError(
                f'{given} is set and {absent} is not; a wrapped range takes both columns',
                at=given,
                reads=('wrapped_minimum', 'wrapped_maximum'),
            )

    @property
    def wrapped(self) -> tuple[IndexColumnEntry, ...]:
        """The columns of the plane's range wrapped at zero, start and end, if it has them.

        Returns:
            ``wrapped_minimum`` and ``wrapped_maximum``, or nothing when neither is set.
        """
        if self.wrapped_minimum is None or self.wrapped_maximum is None:
            return ()
        return (self.wrapped_minimum, self.wrapped_maximum)


@dataclass(frozen=True)
class PlaneEntry:
    """One backplane every image gets: how it is computed and how it is indexed.

    Parameters:
        name: The plane's name, under which the backplane metadata records its
            statistic, and in upper case its FITS extension's name.
        method: The oops backplane method the plane is evaluated with.
        units: The units the plane's array carries.  The plane's statistic is in the
            same units, except that an angular plane's is in degrees.
        index: The columns the global index tables give the plane's statistic.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    name: Str
    method: Str
    units: Str
    index: PlaneIndex


@dataclass(frozen=True)
class IncidenceIndex:
    """The columns the rings index table gives the incidence angle of sunlight on the rings.

    Parameters:
        data_type: The PDS4 data type of every column's values, as in ``ASCII_Real``.
        minimum: The column of the least angle over the ring pixels.
        maximum: The column of the greatest angle over the ring pixels.
        mean: The column of the mean angle over the ring pixels.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    data_type: Str
    minimum: IndexColumnEntry
    maximum: IndexColumnEntry
    mean: IndexColumnEntry


@dataclass(frozen=True)
class IncidenceEntry:
    """The incidence angle of sunlight on the ring plane, recorded once per image.

    Parameters:
        units: The units the backplane metadata records the angle in.
        index: The columns the rings index table gives it.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    units: Str
    index: IncidenceIndex


@dataclass(frozen=True)
class TargetEntry:
    """The PDS4 context product that identifies one target in a bundle's labels.

    Parameters:
        lid: The logical identifier of the target's context product.
        version: The version of that product the PDS registry holds.  It is text, since
            a version such as 1.10 is not 1.1.
        name: The target's name, as the context product gives it.
        type: The target's type, as the context product gives it, such as ``Planet``,
            ``Satellite``, or ``Ring``.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    lid: Str
    version: Str
    name: Str
    type: Str


@dataclass(frozen=True)
class BackplanesSection:
    """The ``backplanes`` section: the planes each image gets, and how a bundle names them.

    Parameters:
        bodies: The planes each body in view gets, in the order the products give them.
        rings: The planes the rings get, in the order the products give them.
        ring_incidence_angle: The incidence angle of sunlight on the ring plane, which
            is recorded once per image rather than as a plane.
        masked_value: The value a plane holds where it has no valid measurement.
        target_lids: The PDS4 target each body and ring system is identified as, keyed
            by the name the backplane metadata gives it.  A target is keyed by its SPICE
            name in upper case.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    bodies: tuple[PlaneEntry, ...]
    rings: tuple[PlaneEntry, ...]
    ring_incidence_angle: IncidenceEntry
    masked_value: Float
    target_lids: dict[Str, TargetEntry]

    def __post_init__(self) -> None:
        """Refuse a ``target_lids`` key that is not in upper case, as SPICE spells it."""
        for key in self.target_lids:
            if key != key.upper():
                raise RuleError(
                    f'{key!r} is not in upper case; a target is written in upper case, as '
                    'SPICE and the PDS spell it',
                    at=('target_lids', key),
                )


@dataclass(frozen=True)
class Pds4Schema:
    """The schema of one PDS4 dictionary a bundle's labels declare.

    A dictionary is published as an XML schema and a Schematron at one location, the two
    differing only in their extensions, and registered as a product of the PDS system
    bundle, which the bundle's XML schema collection lists.

    Parameters:
        location: The URL of the dictionary's XML schema and Schematron less their
            extension, as in ``https://pds.nasa.gov/pds4/geom/v1/PDS4_GEOM_1O00_19B0``.
        lidvid: The LIDVID of the dictionary's schema product, as the XML schema
            collection lists it.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    location: Str
    lidvid: Str

    @property
    def xsd(self) -> str:
        """The URL of the dictionary's XML schema, its location ending in ``.xsd``."""
        return f'{self.location}.xsd'

    @property
    def sch(self) -> str:
        """The URL of the dictionary's Schematron, its location ending in ``.sch``."""
        return f'{self.location}.sch'


@dataclass(frozen=True)
class Pds4Bundle:
    """One bundle's entry in the ``pds4`` section.

    Parameters:
        template_dir: The directory of the bundle's label templates: an absolute path,
            or a name under the packaged templates directory.
        bundle_name: The bundle's name, which its logical identifier ends in.
        bundle_version: The version the bundle, each of its collections, and each of its
            products carry.  It is text, since a version such as 1.10 is not 1.1.
        information_model_version: The PDS4 information model the labels are written
            against.
        schemas: The schema of each dictionary the labels declare, keyed by the prefix
            its namespace takes in a label, in the order the XML schema collection
            lists them.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    template_dir: Str
    bundle_name: Str
    bundle_version: Str
    information_model_version: Str
    schemas: dict[Str, Pds4Schema]


class Pds4BundleEntryError(LookupError):
    """The ``pds4`` section holds no entry for a bundle key a dataset class names."""


@dataclass(frozen=True)
class Pds4Section:
    """The base of the ``pds4`` section's type.

    :func:`pds4_section_type` adds one optional field per bundle a dataset class names,
    so a key that names no dataset's bundle is refused.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    def bundle(self, key: str) -> Pds4Bundle:
        """Return one bundle's entry.

        Parameters:
            key: The name a dataset class gives its bundle.

        Returns:
            The bundle's entry.

        Raises:
            Pds4BundleEntryError: If the section holds no entry for the bundle.
        """
        entry = getattr(self, key, None)
        if not isinstance(entry, Pds4Bundle):
            raise Pds4BundleEntryError(f'the pds4 section holds no entry for the bundle {key!r}')
        return entry


@functools.cache
def pds4_section_type(keys: tuple[str, ...]) -> type[Pds4Section]:
    """Return the type of the ``pds4`` section for a set of bundles.

    Parameters:
        keys: The names the dataset classes give their bundles.

    Returns:
        A frozen schema dataclass with one optional field of :class:`Pds4Bundle` per
        bundle.  A bundle the files give no entry reads as None.
    """
    section_type: type[Pds4Section] = dataclasses.make_dataclass(
        'Pds4Section',
        [(key, Pds4Bundle | None, dataclasses.field(default=None)) for key in keys],
        bases=(Pds4Section,),
        frozen=True,
        namespace={'__pydantic_config__': SCHEMA_CONFIG},
    )
    return section_type
