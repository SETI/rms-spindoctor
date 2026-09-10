"""Shared hermetic fixtures for the PDS4 bundle generation test suite.

The PDS4 backend (``spindoctor.cli.pds4``) is driven entirely by a ``DataSet``'s
``pds4_*`` hooks plus plain files on disk, so these helpers provide a duck-typed
stand-in dataset that implements only the ``pds4_*`` hook surface, tiny
``pdstemplate`` ``.lblx`` templates whose substitution behavior the tests fully
control, and writers for the navigation / backplane metadata files the bundle
stage consumes.  Nothing here touches SPICE, PDS holdings, or the network; all
inputs and outputs live under ``tmp_path``.
"""

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from filecache import FCPath

from spindoctor.dataset.dataset import DataSet, ImageFile, ImageFiles, Pds4Pass

# Minimal pdstemplate templates.  Each references only variables the module under
# test injects itself (BACKPLANE_*/BROWSE_FULL_*/COLLECTION_*/FILE_RECORDS) plus
# the LID variables served by FakePds4DataSet.pds4_template_variables, so the
# tests exercise the substitution plumbing without depending on the shipped
# draft template content.
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
COLLECTION_DATA_TEMPLATE = (
    '<Collection_Data>\n'
    '  <csv>$COLLECTION_DATA_CSV_PATH$</csv>\n'
    '  <start>$EARLIEST_START_DATE_TIME$</start>\n'
    '  <stop>$LATEST_STOP_DATE_TIME$</stop>\n'
    '</Collection_Data>\n'
)
COLLECTION_BROWSE_TEMPLATE = (
    '<Collection_Browse>\n  <csv>$COLLECTION_BROWSE_CSV_PATH$</csv>\n</Collection_Browse>\n'
)
GLOBAL_INDEX_TEMPLATE = '<Index>\n  <records>$FILE_RECORDS$</records>\n</Index>\n'

LABELS_TEMPLATES = {'data.lblx': DATA_TEMPLATE, 'browse.lblx': BROWSE_TEMPLATE}
"""The templates the per-image labels pass renders, and their fake bodies."""

SUMMARY_TEMPLATES = {
    'collection_data.lblx': COLLECTION_DATA_TEMPLATE,
    'collection_browse.lblx': COLLECTION_BROWSE_TEMPLATE,
    'global_index_bodies.lblx': GLOBAL_INDEX_TEMPLATE,
    'global_index_rings.lblx': GLOBAL_INDEX_TEMPLATE,
}
"""The templates the summary pass renders, and their fake bodies."""

DEFAULT_TEMPLATES = LABELS_TEMPLATES | SUMMARY_TEMPLATES
"""Every template the fake dataset declares, which is every one it is given."""

DEFAULT_BUNDLE_NAME = 'fake_bundle'
DEFAULT_SHARD = 'shard0'


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
    """

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
    template_contents: dict[str, str] | None = None,
    template_variables: dict[str, Any] | None = None,
    bodies: list[dict[str, Any]] | None = None,
    rings: list[dict[str, Any]] | None = None,
) -> BundleEnv:
    """Build the standard single-image bundle environment under ``tmp_path``.

    Parameters:
        tmp_path: Base temporary directory.
        image_name: Bare image name for the single input image.
        template_contents: Template files written over the default set, which
            holds every template the fake dataset declares.  A test naming one
            replaces that one and keeps the rest, because a dataset is required
            to carry all of them.
        template_variables: Variables served by the fake dataset's
            ``pds4_template_variables`` hook; defaults to DATA_LID / BROWSE_LID
            entries matching ``image_name``.
        bodies: ``config.backplanes.bodies`` entries for the fake dataset.
        rings: ``config.backplanes.rings`` entries for the fake dataset.

    Returns:
        The populated :class:`BundleEnv`.
    """
    template_dir = tmp_path / 'templates'
    write_templates(template_dir, DEFAULT_TEMPLATES | (template_contents or {}))

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

    image_file = make_image_file(image_name, base_dir=tmp_path)
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


def write_nav_inputs(
    env: BundleEnv,
    *,
    status: str | None = 'success',
    nav_extra: dict[str, Any] | None = None,
    backplane_metadata: dict[str, Any] | None = None,
    summary_png: bytes | None = b'\x89PNG fake bytes',
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Write the navigation and backplane input files for the environment's image.

    Parameters:
        env: The bundle environment to populate.
        status: Navigation ``status`` value; None omits the key entirely.
        nav_extra: Extra keys merged into the navigation metadata dict.
        backplane_metadata: Backplane metadata dict; a small default when None.
        summary_png: Bytes for the ``_summary.png`` file; None writes no PNG.

    Returns:
        The navigation metadata dict and the backplane metadata dict as written.
    """
    nav_metadata: dict[str, Any] = {}
    if status is not None:
        nav_metadata['status'] = status
    if nav_extra:
        nav_metadata.update(nav_extra)
    if backplane_metadata is None:
        backplane_metadata = {'bodies': {}, 'rings': {}}

    nav_file = env.nav_root / f'{env.results_path_stub}_metadata.json'
    nav_file.parent.mkdir(parents=True, exist_ok=True)
    nav_file.write_text(json.dumps(nav_metadata), encoding='utf-8')

    bp_file = env.backplane_root / f'{env.results_path_stub}_backplane_metadata.json'
    bp_file.parent.mkdir(parents=True, exist_ok=True)
    bp_file.write_text(json.dumps(backplane_metadata), encoding='utf-8')

    if summary_png is not None:
        png_file = env.nav_root / f'{env.results_path_stub}_summary.png'
        png_file.write_bytes(summary_png)
    return nav_metadata, backplane_metadata


def write_supplemental(
    data_dir: Path,
    stub: str,
    *,
    bodies: dict[str, Any] | None = None,
    rings: dict[str, Any] | None = None,
    raw_text: str | None = None,
) -> Path:
    """Write a ``<stub>_supplemental.txt`` file in the bundle data tree.

    Parameters:
        data_dir: The bundle's ``data`` directory.
        stub: Path stub (may include shard subdirectories) for the image.
        bodies: ``backplanes.bodies`` payload keyed by body name.
        rings: ``backplanes.rings`` payload (``{'backplanes': {...}}``).
        raw_text: Literal file content overriding the JSON payload entirely.

    Returns:
        The path of the written supplemental file.
    """
    path = data_dir / f'{stub}_supplemental.txt'
    path.parent.mkdir(parents=True, exist_ok=True)
    if raw_text is not None:
        path.write_text(raw_text, encoding='utf-8')
        return path
    payload = {
        'navigation': {},
        'backplanes': {
            'bodies': bodies if bodies is not None else {},
            'rings': rings if rings is not None else {},
        },
    }
    path.write_text(json.dumps(payload), encoding='utf-8')
    return path


def touch_label(data_dir: Path, stub: str) -> Path:
    """Create a ``<stub>_backplanes.lblx`` placeholder in the bundle data tree.

    Parameters:
        data_dir: The bundle's ``data`` directory.
        stub: Path stub (may include shard subdirectories) for the image.

    Returns:
        The path of the created label file.
    """
    path = data_dir / f'{stub}_backplanes.lblx'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('<placeholder/>\n', encoding='utf-8')
    return path


def read_tab(path: Path) -> list[list[str]]:
    """Read a collection or index ``.tab`` file as CSV rows.

    Parameters:
        path: The ``.tab`` file to read.

    Returns:
        All rows, header first, as lists of strings.
    """
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.reader(f))
