"""Shared hermetic fixtures for the PDS4 bundle generation test suite.

Two environments, for two different questions.

The PDS4 backend (``spindoctor.cli.pds4``) is driven entirely by a ``DataSet``'s
``pds4_*`` hooks plus plain files on disk, so these helpers provide a duck-typed
stand-in dataset that implements only the ``pds4_*`` hook surface, tiny
``pdstemplate`` ``.lblx`` templates whose substitution behavior the tests fully
control, and writers for the navigation / backplane metadata files the bundle
stage consumes.  That is :class:`BundleEnv`, and it is how the plumbing is
tested: which file goes where, which variable reaches which template, what a
render that errors leaves behind.

:class:`CohortBundleEnv` is the other half.  It runs the registered dataset a
cohort's bundle is built with over the templates that dataset ships, on the
products the cohort writes, and it is how a question about what a label *says*
is asked -- an epoch, a target, a described data object.  Neither
answers the other's question: a label rendered from a template the test wrote
says whatever the test put there, and a plumbing failure inside the shipped
template set is a needle in three hundred lines of XML.

Nothing here touches SPICE, PDS holdings, or the network; all inputs and
outputs live under ``tmp_path``.
"""

import csv
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
from astropy.io import fits
from filecache import FCPath
from tests.mini_nav_results.cohort import Cohort

from spindoctor.cli.pds4.bundle_data import generate_bundle_data_files
from spindoctor.cli.pds4.bundle_products import generate_bundle_products
from spindoctor.cli.pds4.collections import CollectionOutcome, generate_collection_files
from spindoctor.cli.pds4.epochs import EpochRange
from spindoctor.cli.pds4.global_index import generate_global_index_files
from spindoctor.cli.pds4.targets import Pds4Target
from spindoctor.config import DEFAULT_CONFIG, MAIN_LOGGER
from spindoctor.dataset.dataset import DataSet, ImageFile, ImageFiles, Pds4Pass

DEFAULT_BUNDLE_NAME = 'fake_bundle'
DEFAULT_SHARD = 'shard0'

PDS4_NAMESPACE = 'http://pds.nasa.gov/pds4/pds/v1'
"""The PDS4 common dictionary's namespace, which the bundle label is read in."""


def collection_template(collection: str, body: str) -> str:
    """Return a stand-in collection label template declaring the collection's LID.

    The bundle label is kept only over a bundle holding a label for each collection it
    declares, found by the logical identifier the collection label declares, so every
    stand-in collection label declares one.

    Parameters:
        collection: The collection's name, the last part of its LID.
        body: The elements that follow the identification area.

    Returns:
        The template.
    """
    return (
        f'<Product_Collection xmlns="{PDS4_NAMESPACE}">\n'
        '  <Identification_Area><logical_identifier>'
        f'urn:nasa:pds:{DEFAULT_BUNDLE_NAME}:{collection}'
        '</logical_identifier></Identification_Area>\n'
        f'{body}</Product_Collection>\n'
    )


BUNDLE_COLLECTIONS = (
    'browse',
    'context',
    'data',
    'document',
    'miscellaneous',
    'spice_kernels',
    'xml_schema',
)
"""The collections the summary pass writes, which the stand-in bundle label declares."""


def bundle_template(collections: Sequence[str]) -> str:
    """Return a stand-in bundle label template declaring the given collections.

    Parameters:
        collections: The collections it declares, by the last part of each LID.

    Returns:
        The template: the bundle LID and time range it is handed, and one
        ``Bundle_Member_Entry`` per collection.
    """
    entries = ''.join(
        '  <Bundle_Member_Entry><lid_reference>'
        f'urn:nasa:pds:{DEFAULT_BUNDLE_NAME}:{name}'
        '</lid_reference></Bundle_Member_Entry>\n'
        for name in collections
    )
    return (
        f'<Product_Bundle xmlns="{PDS4_NAMESPACE}">\n'
        '  <lid>$BUNDLE_LID$</lid>\n'
        '  <start>$EARLIEST_START_DATE_TIME$</start>\n'
        '  <stop>$LATEST_STOP_DATE_TIME$</stop>\n'
        f'{entries}</Product_Bundle>\n'
    )


# Minimal pdstemplate templates.  Each references only variables the module under
# test injects itself (BACKPLANE_*/BROWSE_FULL_*/COLLECTION_*/FILE_RECORDS and the
# run-level products' paths) plus the LID variables served by
# FakePds4DataSet.pds4_template_variables, so the tests exercise the substitution
# plumbing without depending on the shipped draft template content.
DATA_TEMPLATE = (
    '<Product_Observational>\n'
    '  <lid>$DATA_LID$</lid>\n'
    '  <fits>$BACKPLANE_FILENAME$</fits>\n'
    '  <suppl>$BACKPLANE_SUPPL_FILENAME$</suppl>\n'
    '</Product_Observational>\n'
)
BROWSE_TEMPLATE = (
    '<Product_Browse>\n'
    '  <lid>$BROWSE_LID$</lid>\n'
    '  <png>$BROWSE_FULL_FILENAME$</png>\n'
    '</Product_Browse>\n'
)
COLLECTION_DATA_TEMPLATE = collection_template(
    'data',
    '  <csv>$COLLECTION_DATA_CSV_PATH$</csv>\n'
    '  <start>$EARLIEST_START_DATE_TIME$</start>\n'
    '  <stop>$LATEST_STOP_DATE_TIME$</stop>\n',
)
COLLECTION_BROWSE_TEMPLATE = collection_template(
    'browse', '  <csv>$COLLECTION_BROWSE_CSV_PATH$</csv>\n'
)
GLOBAL_INDEX_TEMPLATE = (
    '<Index>\n  <lid>$INDEX_LID$</lid>\n'
    '  <records>$FILE_RECORDS(INDEX_TABLE_PATH)-1$</records>\n</Index>\n'
)
COLLECTION_MISCELLANEOUS_TEMPLATE = collection_template(
    'miscellaneous', '  <csv>$COLLECTION_MISCELLANEOUS_CSV_PATH$</csv>\n'
)
BROKEN_TEMPLATE = '<Broken>$COMPLETELY_UNSET_VARIABLE$</Broken>\n'
"""A template naming a variable no caller defines, so the render errors."""

USER_GUIDE_NAME = 'fake-user-guide.pdf'
"""The user guide the fake dataset names, in its template directory and in a bundle."""

USER_GUIDE_PDF = '%PDF-1.4 a stand-in user guide\n'
"""The stand-in user guide's content."""

RUN_LEVEL_FILES = {
    'bundle.lblx': bundle_template(BUNDLE_COLLECTIONS),
    'readme.txt': 'A stand-in readme.\n',
    'collection_context.csv': 'S,urn:nasa:pds:context:instrument:fake::1.0\n',
    'collection_context.lblx': collection_template(
        'context', '  <csv>$COLLECTION_CONTEXT_CSV_PATH$</csv>\n'
    ),
    'collection_document.csv': (
        'S,urn:nasa:pds:context:instrument:fake::1.0\n'
        f'P,urn:nasa:pds:{DEFAULT_BUNDLE_NAME}:document:fake-user-guide::1.0\n'
    ),
    'collection_document.lblx': collection_template(
        'document', '  <csv>$COLLECTION_DOCUMENT_CSV_PATH$</csv>\n'
    ),
    'collection_spice_kernels.csv': (
        f'P,urn:nasa:pds:{DEFAULT_BUNDLE_NAME}:spice_kernels:kernels::1.0\n'
    ),
    'collection_spice_kernels.lblx': collection_template(
        'spice_kernels', '  <csv>$COLLECTION_SPICE_KERNELS_CSV_PATH$</csv>\n'
    ),
    'collection_xml_schema.csv': 'S,urn:nasa:pds:system_bundle:xml_schema:fake::1.0\n',
    'collection_xml_schema.lblx': collection_template(
        'xml_schema', '  <csv>$COLLECTION_XML_SCHEMA_CSV_PATH$</csv>\n'
    ),
    'kernels.ker': 'KPL/MK\n',
    'kernels.lblx': '<Product_SPICE_Kernel>$METAKERNEL_PATH$</Product_SPICE_Kernel>\n',
    'fake-user-guide.lblx': '<Product_Document>$USER_GUIDE_PATH$</Product_Document>\n',
}
"""The files the summary pass takes from the template directory for the run-level products.

The templates it renders and the files it copies, and their stand-in content.  The user
guide itself is not among them, since a bundle is written without it when the template
directory does not hold it.
"""

RUN_LEVEL_PRODUCTS = (
    'bundle.lblx',
    'readme.txt',
    'context/collection_context.csv',
    'context/collection_context.lblx',
    'document/collection_document.csv',
    'document/collection_document.lblx',
    'document/user_guide/fake-user-guide.pdf',
    'document/user_guide/fake-user-guide.lblx',
    'spice_kernels/collection_spice_kernels.csv',
    'spice_kernels/collection_spice_kernels.lblx',
    'spice_kernels/kernels.ker',
    'spice_kernels/kernels.lblx',
    'xml_schema/collection_xml_schema.csv',
    'xml_schema/collection_xml_schema.lblx',
)
"""Every run-level product the summary pass writes into a stand-in bundle, relative to it.

The template directory :func:`make_bundle_env` writes holds the user guide, so its copy
and label are among them.
"""

LABELS_TEMPLATES = {'data.lblx': DATA_TEMPLATE, 'browse.lblx': BROWSE_TEMPLATE}
"""The templates the per-image labels pass renders, and their fake bodies."""

SUMMARY_TEMPLATES = {
    'collection_data.lblx': COLLECTION_DATA_TEMPLATE,
    'collection_browse.lblx': COLLECTION_BROWSE_TEMPLATE,
    'global_bodies_index.lblx': GLOBAL_INDEX_TEMPLATE,
    'global_rings_index.lblx': GLOBAL_INDEX_TEMPLATE,
    'collection_miscellaneous.lblx': COLLECTION_MISCELLANEOUS_TEMPLATE,
    **RUN_LEVEL_FILES,
}
"""The files the summary pass takes from the template directory, and their fake content."""

DEFAULT_TEMPLATES = LABELS_TEMPLATES | SUMMARY_TEMPLATES
"""Every file the fake dataset declares, which is every one it is given."""

PLUMBING_RING_TARGET = 'PLANET_RINGS'
"""The ring target the plumbing backplane metadata names beside its ring statistics."""

TARGET_LIDS: dict[str, dict[str, str]] = {
    name: {
        'lid': f'urn:nasa:pds:context:target:fake.{name.lower()}',
        'version': '1.0',
        'name': name.title(),
        'type': target_type,
    }
    for name, target_type in (
        ('PLANET', 'Planet'),
        ('A', 'Satellite'),
        ('MOON', 'Satellite'),
        ('MOON_A', 'Satellite'),
        ('MOON_B', 'Satellite'),
        (PLUMBING_RING_TARGET, 'Ring'),
    )
}
"""The stand-in targets table the plumbing datasets carry, as ``backplanes.target_lids``.

It has an entry for each body and the ring target the plumbing backplane metadata names,
so that every plumbing image has a target its data label can name.
"""


def ring_metadata(statistics: dict[str, Any]) -> dict[str, Any]:
    """Return the rings block of backplane metadata holding these ring statistics.

    Parameters:
        statistics: The ring statistics, keyed by plane name, or an empty mapping for an
            image with no ring backplanes.

    Returns:
        The block in the shape the backplane writer leaves on disk: the ring target
        :data:`PLUMBING_RING_TARGET`, the incidence angle of sunlight on its plane, and
        the statistics under ``backplanes``.
    """
    return {
        'target': PLUMBING_RING_TARGET,
        'incidence_angle': {'value': 64.6, 'units': 'deg'},
        'backplanes': statistics,
    }


class FakePds4DataSet:
    """Duck-typed ``DataSet`` exposing only the ``pds4_*`` hooks the bundle stage calls.

    The LID/LIDVID builders emit the canonical
    ``urn:nasa:pds:<bundle>:<collection>:<image>`` form with a ``::1.0`` version
    suffix, using the image name verbatim as the LID part so tests can round-trip
    names without instrument-specific transformations.  Calls to
    ``pds4_template_variables`` are recorded on ``template_variables_calls``, and
    the same ``template_variables`` dict object is returned each time so tests can
    observe the file-path variables the bundle stage injects into it.
    """

    def __init__(
        self,
        template_dir: Path,
        *,
        bundle_name: str = DEFAULT_BUNDLE_NAME,
        shard: str = DEFAULT_SHARD,
        template_variables: dict[str, Any] | None = None,
        bodies: list[dict[str, Any]] | None = None,
        rings: list[dict[str, Any]] | None = None,
    ) -> None:
        """Build the fake dataset.

        Parameters:
            template_dir: Directory served by :meth:`pds4_bundle_template_dir`.
            bundle_name: Bundle name served by :meth:`pds4_bundle_name`.
            shard: Directory prefix used by :meth:`pds4_path_stub` for every image.
            template_variables: Dict returned (by reference) from
                :meth:`pds4_template_variables`; empty dict when None.
            bodies: ``config.backplanes.bodies`` entries (dicts with a ``name`` key).
            rings: ``config.backplanes.rings`` entries (dicts with a ``name`` key).
        """
        self._template_dir = template_dir
        self._bundle_name = bundle_name
        self._shard = shard
        self.template_variables: dict[str, Any] = (
            template_variables if template_variables is not None else {}
        )
        self.template_variables_calls: list[dict[str, Any]] = []
        self.config = SimpleNamespace(
            backplanes=SimpleNamespace(
                bodies=bodies if bodies is not None else [],
                rings=rings if rings is not None else [],
                masked_value=DEFAULT_CONFIG.backplanes.masked_value,
                target_lids=TARGET_LIDS,
            )
        )

    def as_dataset(self) -> DataSet:
        """Return self cast to ``DataSet`` for passing into typed call sites."""
        return cast(DataSet, self)

    def pds4_bundle_template_dir(self) -> str:
        """Return the configured template directory as a string path."""
        return str(self._template_dir)

    def pds4_bundle_name(self) -> str:
        """Return the configured bundle name."""
        return self._bundle_name

    def pds4_required_templates(self, pds4_pass: Pds4Pass) -> list[str]:
        """Return the template filenames the given pass must find.

        These are the templates :func:`make_bundle_env` writes into the
        template directory, so what the fake declares is what its tree carries.

        Parameters:
            pds4_pass: Which pass's templates to name.
        """
        if pds4_pass == 'labels':
            return list(LABELS_TEMPLATES)
        return list(SUMMARY_TEMPLATES)

    def pds4_user_guide_file_name(self) -> str:
        """Return the user guide's file name, which :func:`make_bundle_env` writes."""
        return USER_GUIDE_NAME

    def pds4_path_stub(self, image_file: ImageFile) -> str:
        """Return ``<shard>/<image name>`` as the per-image bundle path stub.

        Parameters:
            image_file: The image file to generate the path stub for.
        """
        return f'{self._shard}/{image_file.image_file_url.stem}'

    def pds4_lid_part_to_image_name(self, lid_part: str) -> str:
        """Return the image name for the given LID part (identity round trip).

        The builders here use the image name verbatim as the LID part, so the
        inverse is the identity.

        Parameters:
            lid_part: The LID part (an on-disk product filename stem).
        """
        return lid_part

    def pds4_image_name_to_data_lid(self, image_name: str) -> str:
        """Return the canonical data LID for the given image name.

        Parameters:
            image_name: The image name, used verbatim as the LID part.
        """
        return f'urn:nasa:pds:{self._bundle_name}:data:{image_name}'

    def pds4_image_name_to_data_lidvid(self, image_name: str) -> str:
        """Return the canonical data LIDVID for the given image name.

        Parameters:
            image_name: The image name, used verbatim as the LID part.
        """
        return f'{self.pds4_image_name_to_data_lid(image_name)}::1.0'

    def pds4_image_name_to_browse_lid(self, image_name: str) -> str:
        """Return the canonical browse LID for the given image name.

        Parameters:
            image_name: The image name, used verbatim as the LID part.
        """
        return f'urn:nasa:pds:{self._bundle_name}:browse:{image_name}'

    def pds4_image_name_to_browse_lidvid(self, image_name: str) -> str:
        """Return the canonical browse LIDVID for the given image name.

        Parameters:
            image_name: The image name, used verbatim as the LID part.
        """
        return f'{self.pds4_image_name_to_browse_lid(image_name)}::1.0'

    def pds4_template_variables(
        self,
        *,
        image_file: ImageFile,
        nav_metadata: dict[str, Any],
        backplane_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Record the call and return the shared template-variables dict.

        Parameters:
            image_file: The image file being processed.
            nav_metadata: Navigation metadata dictionary.
            backplane_metadata: Backplane metadata dictionary.
        """
        self.template_variables_calls.append(
            {
                'image_file': image_file,
                'nav_metadata': nav_metadata,
                'backplane_metadata': backplane_metadata,
            }
        )
        return self.template_variables


class NoPds4DataSet:
    """Duck-typed dataset whose ``pds4_*`` hooks all raise ``NotImplementedError``.

    Mirrors the ``DataSet`` base-class contract for datasets that do not support
    PDS4 bundle generation (dev_guide_pds4.rst "Per-dataset extension points").
    It carries a configuration declaring no backplanes and the stand-in targets table,
    as every dataset carries one, so that what the bundle stage reads before it reaches a
    hook is there.
    """

    def __init__(self) -> None:
        """Build the dataset with a configuration declaring no backplanes."""
        self.config = SimpleNamespace(
            backplanes=SimpleNamespace(
                bodies=[],
                rings=[],
                masked_value=DEFAULT_CONFIG.backplanes.masked_value,
                target_lids=TARGET_LIDS,
            )
        )

    def as_dataset(self) -> DataSet:
        """Return self cast to ``DataSet`` for passing into typed call sites."""
        return cast(DataSet, self)

    def __getattr__(self, name: str) -> Any:
        """Raise ``NotImplementedError`` from any ``pds4_*`` hook lookup.

        Parameters:
            name: Attribute being looked up.
        """
        if name.startswith('pds4_'):

            def _unsupported(*args: Any, **kwargs: Any) -> Any:
                raise NotImplementedError('PDS4 bundle generation not supported for this dataset')

            return _unsupported
        raise AttributeError(name)


def make_image_file(
    name: str = '1234567890w',
    *,
    results_path_stub: str | None = None,
    base_dir: Path | None = None,
) -> ImageFile:
    """Build a hermetic ``ImageFile`` with local (never-retrieved) URLs.

    Parameters:
        name: Bare image name; the image URL becomes ``<base>/<name>.img``.
        results_path_stub: Results path stub; defaults to ``res/<name>``.
        base_dir: Existing directory for the image/label URLs.  Required (only)
            by call paths that resolve ``image_file_path``, which creates the
            URL's parent directory; defaults to the non-writable ``/hermetic``.

    Returns:
        The constructed ``ImageFile``.
    """
    stub = results_path_stub if results_path_stub is not None else f'res/{name}'
    base = str(base_dir) if base_dir is not None else '/hermetic'
    return ImageFile(
        image_file_url=FCPath(f'{base}/{name}.img'),
        label_file_url=FCPath(f'{base}/{name}.lbl'),
        results_path_stub=stub,
    )


def write_templates(template_dir: Path, contents: dict[str, str]) -> None:
    """Write template files into a directory, creating it if needed.

    Parameters:
        template_dir: Directory to hold the ``.lblx`` template files.
        contents: Mapping from template filename to file content.
    """
    template_dir.mkdir(parents=True, exist_ok=True)
    for filename, content in contents.items():
        (template_dir / filename).write_text(content, encoding='utf-8')


def index_entry(name: str, units: str) -> dict[str, Any]:
    """Return a backplane configuration entry, with the index block the index tables read.

    Parameters:
        name: The plane's name.
        units: The unit its array carries.

    Returns:
        The entry: its name and units, and an ``index`` block naming its two columns
        ``minimum_<name>`` and ``maximum_<name>``, each ``ASCII_Real`` and described.
    """
    return {
        'name': name,
        'units': units,
        'index': {
            'data_type': 'ASCII_Real',
            'minimum': {'name': f'minimum_{name}', 'description': f'The least {name}.'},
            'maximum': {'name': f'maximum_{name}', 'description': f'The greatest {name}.'},
        },
    }


@dataclass
class BundleEnv:
    """A complete hermetic environment for one bundle-generation call.

    Attributes:
        dataset: The fake dataset serving the ``pds4_*`` hooks.
        image_file: The single input image.
        image_files: The one-image batch wrapping ``image_file``.
        nav_root: Navigation results root directory.
        backplane_root: Backplane results root directory.
        bundle_results_root: Bundle results root directory.
        bundle_dir: ``bundle_results_root / <bundle name>``.
        results_path_stub: The image's results path stub.
        pds4_path_stub: The image's PDS4 bundle path stub.
    """

    dataset: FakePds4DataSet
    image_file: ImageFile
    image_files: ImageFiles
    nav_root: Path
    backplane_root: Path
    bundle_results_root: Path
    bundle_dir: Path
    results_path_stub: str
    pds4_path_stub: str


def make_bundle_env(
    tmp_path: Path,
    *,
    image_name: str = '1234567890w',
    results_path_stub: str | None = None,
    template_contents: dict[str, str] | None = None,
    template_variables: dict[str, Any] | None = None,
    bodies: list[dict[str, Any]] | None = None,
    rings: list[dict[str, Any]] | None = None,
) -> BundleEnv:
    """Build the standard single-image bundle environment under ``tmp_path``.

    Parameters:
        tmp_path: Base temporary directory.
        image_name: Bare image name for the single input image.
        results_path_stub: The image's results path stub; ``res/<image_name>``
            when None.
        template_contents: Template files written over the default set, which
            holds every file the fake dataset declares and its user guide.  A test
            naming one replaces that one and keeps the rest, because a dataset is
            required to carry all of them.
        template_variables: Variables served by the fake dataset's
            ``pds4_template_variables`` hook; defaults to DATA_LID / BROWSE_LID
            entries matching ``image_name``.
        bodies: ``config.backplanes.bodies`` entries for the fake dataset.
        rings: ``config.backplanes.rings`` entries for the fake dataset.

    Returns:
        The populated :class:`BundleEnv`.
    """
    template_dir = tmp_path / 'templates'
    user_guide = {USER_GUIDE_NAME: USER_GUIDE_PDF}
    write_templates(template_dir, DEFAULT_TEMPLATES | user_guide | (template_contents or {}))

    if template_variables is None:
        template_variables = {
            'DATA_LID': f'urn:nasa:pds:{DEFAULT_BUNDLE_NAME}:data:{image_name}',
            'BROWSE_LID': f'urn:nasa:pds:{DEFAULT_BUNDLE_NAME}:browse:{image_name}',
        }
    dataset = FakePds4DataSet(
        template_dir,
        template_variables=template_variables,
        bodies=bodies,
        rings=rings,
    )

    image_file = make_image_file(image_name, results_path_stub=results_path_stub, base_dir=tmp_path)
    nav_root = tmp_path / 'nav'
    backplane_root = tmp_path / 'backplanes'
    bundle_results_root = tmp_path / 'bundle'
    nav_root.mkdir()
    backplane_root.mkdir()
    bundle_results_root.mkdir()
    return BundleEnv(
        dataset=dataset,
        image_file=image_file,
        image_files=ImageFiles(image_files=[image_file]),
        nav_root=nav_root,
        backplane_root=backplane_root,
        bundle_results_root=bundle_results_root,
        bundle_dir=bundle_results_root / DEFAULT_BUNDLE_NAME,
        results_path_stub=image_file.results_path_stub,
        pds4_path_stub=f'{DEFAULT_SHARD}/{image_name}',
    )


NAVIGATED_EXPOSURE: dict[str, float] = {
    'start_time_et': 129399999.77,
    'end_time_et': 129400000.23,
}
"""The exposure a success document's ``observation`` block records, in TDB seconds.

The navigation records an image's exposure in the ``observation`` block of its
document, pointing or not; both passes read the start and end from there, the summary
pass from every supplemental file, so every navigated document and every supplemental
file the plumbing tests write records these.
"""


def navigated_document(**extra: Any) -> dict[str, Any]:
    """Return a success navigation document recording an exposure.

    Parameters:
        **extra: Keys merged into the document, over the two it holds otherwise.

    Returns:
        A document with ``status`` ``success`` and an ``observation`` block recording
        :data:`NAVIGATED_EXPOSURE`, with ``extra`` merged in.
    """
    return {'status': 'success', 'observation': dict(NAVIGATED_EXPOSURE), **extra}


A_RANGE = EpochRange(start_et=129399999.77, stop_et=130700000.54)
"""A range for the collection generator where what the label states is not the question.

SPICE's ``et2utc`` writes the two epochs as ``2004-02-07T04:25:35.585`` and
``2004-02-22T05:32:16.355``, so a label states the range as ``2004-02-07T04:25:35Z``
to ``2004-02-22T05:32:17Z``.
"""


def run_collections(
    env: BundleEnv,
    *,
    epochs: EpochRange | None = A_RANGE,
    targets: Sequence[Pds4Target] = (),
) -> CollectionOutcome:
    """Run generate_collection_files against the environment's bundle root.

    Parameters:
        env: The hermetic bundle environment to process.
        epochs: The range of the products' epochs to hand the generator.
        targets: The targets the products name, to hand the generator.

    Returns:
        What the generation came to: the collection labels not written, and the images
        whose products disagree.
    """
    return generate_collection_files(
        FCPath(env.bundle_results_root),
        env.dataset.as_dataset(),
        MAIN_LOGGER,
        epochs=epochs,
        targets=targets,
    )


def write_nav_inputs(
    env: BundleEnv,
    *,
    status: str | None = 'success',
    nav_extra: dict[str, Any] | None = None,
    backplane_metadata: dict[str, Any] | None = None,
    summary_png: bytes | None = b'\x89PNG fake bytes',
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Write the navigation and backplane input files for the environment's image.

    The navigation document is :func:`navigated_document`'s, whose ``observation`` block
    records an exposure, so that a success document is one the labels pass can label.

    Parameters:
        env: The bundle environment to populate.
        status: Navigation ``status`` value; None omits the key entirely.
        nav_extra: Extra keys merged into the navigation metadata dict, over the
            ``status`` and ``observation`` it holds otherwise.
        backplane_metadata: Backplane metadata dict; when None, one naming the body
            ``MOON`` with no statistic, and no ring backplanes, so that its data label
            has a target to name.
        summary_png: Bytes for the ``_summary.png`` file; None writes no PNG.

    Returns:
        The navigation metadata dict and the backplane metadata dict as written.
    """
    nav_metadata = navigated_document()
    if status is None:
        del nav_metadata['status']
    else:
        nav_metadata['status'] = status
    if nav_extra:
        nav_metadata.update(nav_extra)
    if backplane_metadata is None:
        backplane_metadata = {'bodies': {'MOON': {'backplanes': {}}}, 'rings': ring_metadata({})}

    nav_file = env.nav_root / f'{env.results_path_stub}_metadata.json'
    nav_file.parent.mkdir(parents=True, exist_ok=True)
    nav_file.write_text(json.dumps(nav_metadata), encoding='utf-8')

    bp_file = env.backplane_root / f'{env.results_path_stub}_backplane_metadata.json'
    bp_file.parent.mkdir(parents=True, exist_ok=True)
    bp_file.write_text(json.dumps(backplane_metadata), encoding='utf-8')

    if summary_png is not None:
        png_file = env.nav_root / f'{env.results_path_stub}_summary.png'
        png_file.write_bytes(summary_png)
    write_backplane_fits(env.backplane_root / f'{env.results_path_stub}_backplanes.fits')
    return nav_metadata, backplane_metadata


def write_backplane_fits(path: Path, *, shape: tuple[int, int] = (2, 2)) -> None:
    """Write a small backplane FITS: an empty primary HDU and one float plane.

    The labels pass copies an image's FITS into the bundle beside its data label, so
    every image the plumbing tests label needs one beside its backplane metadata.  It
    is a real FITS, as the backplane stage writes one, rather than a few bytes
    standing in for one, so that it is the kind of file the pass copies.

    Parameters:
        path: Where the FITS goes; its directory is created if it is not there.
        shape: The plane's lines and samples.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    plane = fits.ImageHDU(data=np.zeros(shape, dtype=np.float32), name='BODY_LATITUDE')
    plane.header['BUNIT'] = 'rad'
    fits.HDUList([fits.PrimaryHDU(), plane]).writeto(path)


def write_supplemental(
    data_dir: Path,
    stub: str,
    *,
    bodies: dict[str, Any] | None = None,
    rings: dict[str, Any] | None = None,
    navigation: dict[str, Any] | None = None,
) -> Path:
    """Write a ``<stub>_supplemental.txt`` file in the bundle data tree.

    Parameters:
        data_dir: The bundle's ``data`` directory.
        stub: Path stub (may include shard subdirectories) for the image.
        bodies: ``backplanes.bodies`` payload keyed by body name.
        rings: ``backplanes.rings`` payload, as :func:`ring_metadata` builds it.
        navigation: The ``navigation`` document; :func:`navigated_document`'s when
            None.

    Returns:
        The path of the written supplemental file.
    """
    path = data_dir / f'{stub}_supplemental.txt'
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'navigation': navigation if navigation is not None else navigated_document(),
        'backplanes': {
            'bodies': bodies if bodies is not None else {},
            'rings': rings if rings is not None else {},
        },
    }
    path.write_text(json.dumps(payload), encoding='utf-8')
    return path


def _touch_placeholder(path: Path) -> Path:
    """Create a placeholder label at a path, and the directories above it.

    Parameters:
        path: Where the placeholder goes.

    Returns:
        ``path``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('<placeholder/>\n', encoding='utf-8')
    return path


def touch_label(data_dir: Path, stub: str) -> Path:
    """Create a ``<stub>_backplanes.lblx`` placeholder in the bundle data tree.

    Parameters:
        data_dir: The bundle's ``data`` directory.
        stub: Path stub (may include shard subdirectories) for the image.

    Returns:
        The path of the created label file.
    """
    return _touch_placeholder(data_dir / f'{stub}_backplanes.lblx')


def touch_browse_label(browse_dir: Path, stub: str) -> Path:
    """Create a ``<stub>_summary.lblx`` placeholder in the bundle browse tree.

    Parameters:
        browse_dir: The bundle's ``browse`` directory.
        stub: Path stub (may include shard subdirectories) for the image.

    Returns:
        The path of the created label file.
    """
    return _touch_placeholder(browse_dir / f'{stub}_summary.lblx')


def read_csv_rows(path: Path) -> list[list[str]]:
    """Read a collection inventory as comma-separated rows.

    Parameters:
        path: The inventory to read.

    Returns:
        Every row, in file order, as lists of strings.
    """
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.reader(f))


def read_index_rows(path: Path) -> list[list[str]]:
    """Read a global index table as its header line and its rows, each cell unpadded.

    Every field of a record is padded to its column's length and the fields are
    separated by commas, which no value holds, so a line's cells are its parts between
    commas with the padding stripped.

    Parameters:
        path: The table to read.

    Returns:
        The header line's names, then each record's values, in file order.
    """
    lines = path.read_bytes().decode('ascii').splitlines()
    return [[cell.strip() for cell in line.split(',')] for line in lines]


@dataclass
class CohortBundleEnv:
    """A bundle-generation environment over a cohort and its bundle's shipped templates.

    Where :class:`BundleEnv` controls every variable so that the substitution
    plumbing can be asserted on, this one controls none of them: the dataset is
    the registered one the cohort's bundle is built with, the templates are the
    ones that dataset ships, and the inputs are the documents and products a
    navigation run and the backplane stage leave behind.  What it is for is
    asserting what a label says, which nothing built out of stand-ins can
    answer.

    Attributes:
        dataset: The registered dataset the cohort's bundle is built with,
            serving its own ``pds4_*`` hooks and its own template directory.
        cohort: The written cohort, holding both input roots and every image.
        bundle_results_root: Where this test's bundle goes.
        bundle_dir: ``bundle_results_root / <the dataset's bundle name>``.
    """

    dataset: DataSet
    cohort: Cohort
    bundle_results_root: Path
    bundle_dir: Path


def make_cohort_bundle_env(cohort: Cohort, tmp_path: Path) -> CohortBundleEnv:
    """Build a bundle environment over the cohort, writing into ``tmp_path``.

    The cohort is read-only and shared by the session; only the bundle the run
    writes belongs to one test.

    Parameters:
        cohort: The session's cohort, holding the navigation and backplane
            roots the run reads.
        tmp_path: Base temporary directory for this test's bundle output.

    Returns:
        The populated :class:`CohortBundleEnv`.
    """
    dataset = cohort.dataset()
    bundle_results_root = tmp_path / 'bundle'
    bundle_results_root.mkdir(parents=True, exist_ok=True)
    return CohortBundleEnv(
        dataset=dataset,
        cohort=cohort,
        bundle_results_root=bundle_results_root,
        bundle_dir=bundle_results_root / dataset.pds4_bundle_name(),
    )


def label_cohort_images(env: CohortBundleEnv, stubs: Sequence[str]) -> None:
    """Run the labels pass over some of the environment's cohort images, in turn.

    Parameters:
        env: The environment whose cohort the images are from and whose bundle the
            labels go into.
        stubs: The images to label, by results path stub.
    """
    for stub in stubs:
        generate_bundle_data_files(
            env.dataset,
            env.cohort.batch(stub),
            nav_results_root=FCPath(env.cohort.nav_results_root),
            backplane_results_root=FCPath(env.cohort.backplane_results_root),
            bundle_results_root=FCPath(env.bundle_results_root),
            logger=MAIN_LOGGER,
        )


def summarize_bundle(env: CohortBundleEnv) -> None:
    """Run the summary pass's three generators over the environment's bundle.

    They run in the order the pass runs them: the global index first, whose range of
    the products' epochs and whose targets the collection generator and the run-level
    products are handed, and the run-level products last.

    Parameters:
        env: The environment whose bundle is summarized.
    """
    bundle_results_root = FCPath(env.bundle_results_root)
    index = generate_global_index_files(bundle_results_root, env.dataset, MAIN_LOGGER)
    generate_collection_files(
        bundle_results_root, env.dataset, MAIN_LOGGER, epochs=index.epochs, targets=index.targets
    )
    generate_bundle_products(
        bundle_results_root, env.dataset, MAIN_LOGGER, epochs=index.epochs, targets=index.targets
    )


def write_cohort_bundle(cohort: Cohort, tmp_path: Path, stubs: Sequence[str]) -> CohortBundleEnv:
    """Build a bundle over some of a cohort's images, running both passes.

    Parameters:
        cohort: The session's cohort, holding the navigation and backplane roots the
            run reads.
        tmp_path: Base temporary directory for this test's bundle output.
        stubs: The images to bundle, by results path stub.

    Returns:
        The environment the bundle was written into.
    """
    env = make_cohort_bundle_env(cohort, tmp_path)
    label_cohort_images(env, stubs)
    summarize_bundle(env)
    return env
