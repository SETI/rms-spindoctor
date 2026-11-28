"""Static guard: navigator-side code never references a truth attribute name.

The information boundary's filtered channel (``obs.nav_params``) is structural,
but the full scene truth still rides the observation object as plain
attributes (``obs.sim_params``, the planted ``sim_offset_v`` / ``sim_offset_u``,
and the renderer's output records) -- one attribute access away from any
navigator-side model's ``self.obs``.  This module is the mechanical consumer
guard: it walks the AST of every module under ``spindoctor.nav_model``,
``spindoctor.nav_technique``, and ``spindoctor.nav_orchestrator`` and fails if
any of them references a truth-bearing attribute name, either as an attribute
access (``obs.sim_params``) or as a bare string literal (``getattr(obs,
'sim_params')``, ``vars(obs)['sim_params']``).

The deny-list is every ``sim_``-prefixed attribute ``ObsSim.from_file`` stores
on the snapshot, plus the historical ``sim_star_list`` name so it cannot
return; a completeness test parses ``obs_inst_sim.py`` so a new truth
attribute added there without a deny-list extension fails here, not silently.

A second scan covers the ``atmosphere`` block's key names.  That block is a
single atomic entry of the schema's truth inventory, so the structural
boundary test can only prove the block does not cross; it cannot prove that
no navigator-side module reaches for a key inside it by name through some
other channel.  Scanning for the key names closes that gap, and it is the
haze navigator that makes it worth closing: a simulated Titan model reads the
body's idealized geometry and must never read the haze that geometry is a
deliberate mismatch for.

A third scan makes the boundary follow the package layout.  The neutral simulator
modules are ``spindoctor.sim.shared`` and ``spindoctor.sim.scene.fields``.  A module of
a navigator-side package imports nothing else from ``spindoctor.sim`` or from the scene
editor (``spindoctor.ui.sim_editor``): not the renderer, not the scene rules, and not
the simulated instrument catalog or the artifact-mode registry, which describe how the
image side renders a frame (a navigator-side model reads what the navigator would know
from its own config).  Neither does a module of
``sim/shared/``, so the shared conventions never pull in the renderer, the scene rules,
or the editor.

A fourth scan covers the simulated observation (``obs/obs_inst_sim.py``), the bridge
that renders the image and builds ``nav_params``.  It may import the renderer, but its
star limit is navigator-side, so it imports neither ``spindoctor.sim.instruments`` nor
``spindoctor.sim.artifact_modes`` except as ``_ALLOWED_IMAGE_SIDE_IMPORTS`` sanctions.

A fifth scan covers the renderer (``sim/render/``), the image side.  It takes a camera's
physics from the instrument entries, so it doesn't import ``spindoctor.config``, and from
``navigator_link`` it imports only the matches of ``_NAVIGATOR_MATCHES``: the values a
frame has to share with the navigator.

A sixth scan covers the scene package (``sim/scene/``).  The scene's types own every
key, and the renderer reads them, so no scene module imports the renderer
(``spindoctor.sim.render``).

Legitimate exceptions, should one ever exist, go in ``_ALLOWED_REFERENCES``:
a mapping from ``(module path relative to the spindoctor package, name)`` to a
one-line justification.  The allowlist is currently empty -- no navigator-side
module has a sanctioned reason to name a truth attribute.
"""

import ast
from pathlib import Path

import pytest

import spindoctor
from spindoctor.sim.scene.truth_atmosphere import ATMOSPHERE_KEYS, HAZE_STRUCTURE_KEYS

# Truth-bearing attribute names stored on the simulated observation by
# ObsSim.from_file (src/spindoctor/obs/obs_inst_sim.py), plus the historical
# sim_star_list attribute (removed from the observation; denied so a future
# change cannot quietly reintroduce a consumer).
TRUTH_ATTRIBUTE_DENYLIST: frozenset[str] = frozenset(
    {
        'sim_params',
        'sim_offset_v',
        'sim_offset_u',
        'sim_time',
        'sim_body_models',
        'sim_inventory',
        'sim_body_order_near_to_far',
        'sim_body_index_map',
        'sim_body_mask_map',
        'sim_star_list',
    }
)

# Key names of the body ``atmosphere`` truth block, minus the single-letter
# Henyey-Greenstein key: one character is not a distinctive enough token to
# scan for (any ``.g`` attribute or ``'g'`` literal anywhere in navigator code
# would trip it), and the multi-character keys beside it already make an
# atmosphere-reading module impossible to write without a hit.
ATMOSPHERE_TRUTH_KEY_NAMES: frozenset[str] = frozenset(
    key for key in ATMOSPHERE_KEYS if len(key) > 1
)

# Navigator-side packages the guard scans, relative to the spindoctor package.
_NAVIGATOR_PACKAGES: tuple[str, ...] = ('nav_model', 'nav_technique', 'nav_orchestrator')

# Sanctioned references: (module path relative to the spindoctor package,
# denied name) -> one-line justification.  Empty by design; add an entry only
# with a justification a reviewer can check.
_ALLOWED_REFERENCES: dict[tuple[str, str], str] = {}

_PACKAGE_ROOT = Path(spindoctor.__file__).parent

# The image-side simulator modules the simulated observation may not import: the
# instrument catalog package, and the artifact-mode registry.
_IMAGE_SIDE_MODULES: tuple[str, ...] = (
    'spindoctor.sim.instruments',
    'spindoctor.sim.artifact_modes',
)

# The simulator's packages, the scene editor among them.
_SIMULATOR_PACKAGES: tuple[str, ...] = ('spindoctor.sim', 'spindoctor.ui.sim_editor')
# The simulator modules both sides may import.  The modules of the navigator packages
# and of the shared directory below (relative to the spindoctor package) may import
# nothing else from the simulator's packages.
_NEUTRAL_SIM_MODULES: tuple[str, ...] = ('spindoctor.sim.shared', 'spindoctor.sim.scene.fields')
_SHARED_DIR = 'sim/shared'

# Navigator-side modules outside the navigator packages whose image-side imports the
# fourth scan covers, relative to the spindoctor package.
_NAVIGATOR_MODULES: tuple[str, ...] = ('obs/obs_inst_sim.py',)

# Sanctioned image-side imports: (module path relative to the spindoctor package,
# imported module) -> one-line justification.  An entry covers the names imported
# from that module too.
_ALLOWED_IMAGE_SIDE_IMPORTS: dict[tuple[str, str], str] = {
    ('obs/obs_inst_sim.py', 'spindoctor.sim.instruments.navigator_link'): (
        'it finds which navigator config section and detector to load for the instrument'
    ),
}

# What the renderer may import from navigator_link, with the module that reads the
# navigator's configuration on its behalf.
_NAVIGATOR_MATCHES: frozenset[str] = frozenset({'navigator_marker', 'navigator_matched_psf'})
_NAVIGATOR_LINK = 'spindoctor.sim.instruments.navigator_link'
_RENDER_DIR = 'sim/render'

# The scene package, relative to the spindoctor package.
_SCENE_DIR = 'sim/scene'


def _within(module: str, packages: tuple[str, ...]) -> bool:
    """Whether ``module`` is one of ``packages`` or lies inside one of them."""
    return any(module == package or module.startswith(f'{package}.') for package in packages)


def _module_package(path: Path) -> str:
    """The dotted package a module file under the spindoctor package belongs to."""
    parent = path.parent.relative_to(_PACKAGE_ROOT.parent)
    return '.'.join(parent.parts)


def _imported_module(node: ast.ImportFrom, *, package: str) -> str:
    """The absolute module a ``from ... import`` statement reads from.

    A relative import climbs ``level - 1`` packages above ``package`` before it
    appends the module it states.
    """
    if node.level == 0:
        return node.module or ''
    parts = package.split('.')
    base = parts[: len(parts) - (node.level - 1)]
    return '.'.join([*base, node.module] if node.module else base)


def _image_side_imports(source: str, *, package: str) -> list[tuple[int, str]]:
    """Return each import of an image-side simulator module in ``source``.

    Parameters:
        source: Python source text to scan.
        package: The dotted package the source's module belongs to, against which
            a relative import resolves.

    Returns:
        List of ``(line number, module)`` tuples, one per imported module.
    """
    imports: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            module = _imported_module(node, package=package)
            # ``from spindoctor.sim import instruments`` imports the package itself.
            modules = [module, *(f'{module}.{alias.name}' for alias in node.names)]
        else:
            continue
        imports.extend(
            (node.lineno, module) for module in modules if _within(module, _IMAGE_SIDE_MODULES)
        )
    return imports


def _non_neutral_sim_imports(source: str, *, package: str) -> list[tuple[int, str]]:
    """Return each import of a simulator or editor module other than the neutral modules.

    A ``from`` import passes when the module it reads from is neutral, or when the name
    it imports is a neutral module, as in ``from spindoctor.sim.scene import fields``.

    Parameters:
        source: Python source text to scan.
        package: The dotted package the source's module belongs to, against which
            a relative import resolves.

    Returns:
        List of ``(line number, module)`` tuples, one per imported name; a name read
        from a module appears as ``<module>.<name>``.
    """
    imports: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            targets = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            module = _imported_module(node, package=package)
            if _within(module, _NEUTRAL_SIM_MODULES):
                continue
            targets = [f'{module}.{alias.name}' for alias in node.names]
        else:
            continue
        imports.extend(
            (node.lineno, target)
            for target in targets
            if _within(target, _SIMULATOR_PACKAGES) and not _within(target, _NEUTRAL_SIM_MODULES)
        )
    return imports


def _truth_references(
    source: str, *, filename: str, denied: frozenset[str] = TRUTH_ATTRIBUTE_DENYLIST
) -> list[tuple[str, int, str]]:
    """Return every reference to a denied name in ``source``.

    A reference is an attribute access whose attribute name is denied, or a
    string literal exactly equal to a denied name (the ``getattr`` /
    ``vars()`` evasion, and the form a scene-dictionary lookup takes).  Bare
    identifiers (parameter names, local variables) are deliberately not
    flagged: the leak surface is the namespace a value is read out of, not
    the word itself.

    Parameters:
        source: Python source text to scan.
        filename: Label used in the returned reference records.
        denied: Names to flag; defaults to the observation-attribute
            deny-list.

    Returns:
        List of ``(filename, line number, denied name)`` tuples.
    """
    references: list[tuple[str, int, str]] = []
    for node in ast.walk(ast.parse(source, filename=filename)):
        if isinstance(node, ast.Attribute) and node.attr in denied:
            references.append((filename, node.lineno, node.attr))
        elif (
            isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in denied
        ):
            references.append((filename, node.lineno, node.value))
    return references


def _package_violations(
    package: str,
    *,
    denied: frozenset[str] = TRUTH_ATTRIBUTE_DENYLIST,
    label: str = 'truth attribute',
) -> list[str]:
    """Return formatted, non-allowlisted denied-name references in ``package``."""
    violations: list[str] = []
    for path in sorted((_PACKAGE_ROOT / package).rglob('*.py')):
        rel = path.relative_to(_PACKAGE_ROOT).as_posix()
        for filename, lineno, name in _truth_references(
            path.read_text(encoding='utf-8'), filename=rel, denied=denied
        ):
            if (rel, name) in _ALLOWED_REFERENCES:
                continue
            violations.append(f'{filename}:{lineno}: references {label} {name!r}')
    return violations


def test_denylist_covers_every_obs_sim_truth_attribute() -> None:
    """Every sim_* attribute ObsSim stores on the snapshot is on the deny-list.

    Parses ``obs_inst_sim.py`` and collects the attribute names assigned onto
    the snapshot with the ``sim_`` prefix, so adding a truth attribute there
    without extending the deny-list fails here.
    """
    source = (_PACKAGE_ROOT / 'obs' / 'obs_inst_sim.py').read_text(encoding='utf-8')
    stored: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Attribute) and target.attr.startswith('sim_'):
                stored.add(target.attr)
    assert stored
    missing = stored - TRUTH_ATTRIBUTE_DENYLIST
    assert not missing, (
        f'ObsSim stores truth attribute(s) {sorted(missing)} not covered by the '
        f'static guard deny-list; extend TRUTH_ATTRIBUTE_DENYLIST in the same change'
    )


@pytest.mark.parametrize('package', _NAVIGATOR_PACKAGES)
def test_navigator_package_references_no_truth_attribute(package: str) -> None:
    """No module in the navigator-side package references a truth attribute."""
    violations = _package_violations(package)
    assert not violations, (
        'navigator-side code references simulated-scene truth attributes '
        '(read obs.nav_params instead, or add an _ALLOWED_REFERENCES entry '
        'with a justification):\n' + '\n'.join(violations)
    )


@pytest.mark.parametrize('package', _NAVIGATOR_PACKAGES)
def test_navigator_package_mentions_no_atmosphere_truth_key(package: str) -> None:
    """No navigator-side module mentions a key of the haze truth block.

    The structural boundary filter drops the whole ``atmosphere`` mapping,
    so a navigator-side read of one of its keys can only come from some
    other channel -- and would be a leak the atomic-block test cannot see.
    """
    violations = _package_violations(
        package, denied=ATMOSPHERE_TRUTH_KEY_NAMES, label='atmosphere truth key'
    )
    assert not violations, (
        'navigator-side code mentions simulated-scene haze truth keys (the haze '
        'is nature, and the predicted geometry is a deliberate mismatch for '
        'it):\n' + '\n'.join(violations)
    )


def _is_allowed_import(rel: str, module: str) -> bool:
    """Whether an allowlist entry sanctions ``module``, or the module a name comes from."""
    return any(
        rel == allowed_rel and (module == allowed or module.startswith(f'{allowed}.'))
        for allowed_rel, allowed in _ALLOWED_IMAGE_SIDE_IMPORTS
    )


def _image_side_import_violations(paths: list[Path]) -> list[str]:
    """Return each non-allowlisted image-side import in ``paths``, formatted."""
    violations: list[str] = []
    for path in paths:
        rel = path.relative_to(_PACKAGE_ROOT).as_posix()
        violations.extend(
            f'{rel}:{lineno}: imports {module}'
            for lineno, module in _image_side_imports(
                path.read_text(encoding='utf-8'), package=_module_package(path)
            )
            if not _is_allowed_import(rel, module)
        )
    return violations


@pytest.mark.parametrize('module', _NAVIGATOR_MODULES)
def test_navigator_module_imports_only_sanctioned_image_side_modules(module: str) -> None:
    """A navigator-side module outside the packages imports only allowlisted modules."""
    violations = _image_side_import_violations([_PACKAGE_ROOT / module])
    assert not violations, (
        'navigator-side code imports an image-side simulator module (add an '
        '_ALLOWED_IMAGE_SIDE_IMPORTS entry with a justification if it is sanctioned):\n'
        + '\n'.join(violations)
    )


def test_import_guard_flags_a_module_import() -> None:
    """The import scanner detects ``from spindoctor.sim import instruments``."""
    imports = _image_side_imports(
        'from spindoctor.sim import instruments\n', package='spindoctor.nav_model'
    )
    assert imports == [(1, 'spindoctor.sim.instruments')]


def test_import_guard_flags_a_relative_import() -> None:
    """The import scanner resolves ``from ..sim import artifact_modes`` in a navigator module."""
    imports = _image_side_imports(
        'from ..sim import artifact_modes\n', package='spindoctor.nav_model'
    )
    assert imports == [(1, 'spindoctor.sim.artifact_modes')]


def _non_neutral_violations(directory: str) -> list[str]:
    """Return each non-neutral simulator import of the modules in ``directory``."""
    violations: list[str] = []
    for path in sorted((_PACKAGE_ROOT / directory).rglob('*.py')):
        rel = path.relative_to(_PACKAGE_ROOT).as_posix()
        violations.extend(
            f'{rel}:{lineno}: imports {module}'
            for lineno, module in _non_neutral_sim_imports(
                path.read_text(encoding='utf-8'), package=_module_package(path)
            )
        )
    return violations


@pytest.mark.parametrize('package', _NAVIGATOR_PACKAGES)
def test_navigator_package_imports_only_neutral_simulator_modules(package: str) -> None:
    """A navigator-side module imports only sim.shared and sim.scene.fields."""
    violations = _non_neutral_violations(package)
    assert not violations, (
        'navigator-side code imports a simulator or scene-editor module other than '
        'spindoctor.sim.shared and spindoctor.sim.scene.fields:\n' + '\n'.join(violations)
    )


def test_shared_modules_import_only_neutral_simulator_modules() -> None:
    """A module of sim/shared imports only sim.shared and sim.scene.fields."""
    violations = _non_neutral_violations(_SHARED_DIR)
    assert not violations, (
        'a sim/shared module imports a simulator or scene-editor module other than '
        'spindoctor.sim.shared and spindoctor.sim.scene.fields:\n' + '\n'.join(violations)
    )


def test_neutral_guard_flags_a_render_import_in_a_simulated_model() -> None:
    """The neutral-import scanner flags a simulated model importing the driver."""
    imports = _non_neutral_sim_imports(
        'from spindoctor.sim.render import driver\n', package='spindoctor.nav_model.simulated'
    )
    assert imports == [(1, 'spindoctor.sim.render.driver')]


def test_neutral_guard_flags_a_scene_rules_import_in_a_shared_module() -> None:
    """The neutral-import scanner flags a shared module importing scene rules."""
    imports = _non_neutral_sim_imports(
        'from ..scene.truth_atmosphere import ATMOSPHERE_KEYS\n', package='spindoctor.sim.shared'
    )
    assert imports == [(1, 'spindoctor.sim.scene.truth_atmosphere.ATMOSPHERE_KEYS')]


def test_neutral_guard_flags_an_editor_import_in_a_shared_module() -> None:
    """The neutral-import scanner flags a shared module importing the scene editor."""
    imports = _non_neutral_sim_imports(
        'from spindoctor.ui import sim_editor\n', package='spindoctor.sim.shared'
    )
    assert imports == [(1, 'spindoctor.ui.sim_editor')]


def test_atmosphere_guard_covers_the_structure_keys() -> None:
    """The scanned name set includes every symmetry-breaking structure key.

    The single-letter phase-asymmetry key is deliberately excluded (see the
    deny-set comment); every key that describes structure is long enough to
    scan for, so none may be dropped silently.
    """
    assert HAZE_STRUCTURE_KEYS <= ATMOSPHERE_TRUTH_KEY_NAMES


def test_guard_flags_an_attribute_access() -> None:
    """The scanner detects a direct truth-attribute access."""
    references = _truth_references('def f(obs):\n    return obs.sim_params\n', filename='probe.py')
    assert references == [('probe.py', 2, 'sim_params')]


def test_guard_flags_a_string_literal_evasion() -> None:
    """The scanner detects the getattr string-literal form."""
    references = _truth_references(
        "def f(obs):\n    return getattr(obs, 'sim_offset_v')\n", filename='probe.py'
    )
    assert references == [('probe.py', 2, 'sim_offset_v')]


def test_allowlist_entries_name_existing_modules() -> None:
    """Every allowlist entry points at a module that still exists."""
    for rel, _name in [*_ALLOWED_REFERENCES, *_ALLOWED_IMAGE_SIDE_IMPORTS]:
        assert (_PACKAGE_ROOT / rel).is_file(), f'stale allowlist entry: {rel}'


def _navigator_config_imports(source: str, *, package: str) -> list[tuple[int, str]]:
    """Return each import that lets a renderer module read the navigator's configuration.

    Parameters:
        source: Python source text to scan.
        package: The dotted package the source's module belongs to, against which
            a relative import resolves.

    Returns:
        List of ``(line number, module)`` tuples, one per imported name of
        ``spindoctor.config``, and of ``navigator_link`` other than the matches.
    """
    allowed = {f'{_NAVIGATOR_LINK}.{name}' for name in _NAVIGATOR_MATCHES}
    imports: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            targets = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            module = _imported_module(node, package=package)
            targets = [f'{module}.{alias.name}' for alias in node.names]
        else:
            continue
        imports.extend(
            (node.lineno, target)
            for target in targets
            if _within(target, ('spindoctor.config', _NAVIGATOR_LINK)) and target not in allowed
        )
    return imports


def test_the_renderer_reads_the_navigator_config_only_through_the_matches() -> None:
    """No renderer module imports the navigator's config, except through the matches."""
    violations = [
        f'{path.relative_to(_PACKAGE_ROOT).as_posix()}:{lineno}: imports {module}'
        for path in sorted((_PACKAGE_ROOT / _RENDER_DIR).rglob('*.py'))
        for lineno, module in _navigator_config_imports(
            path.read_text(encoding='utf-8'), package=_module_package(path)
        )
    ]
    assert violations == []


def test_the_navigator_config_scan_flags_the_block_and_passes_a_match() -> None:
    """The scanner flags the block resolver and the config, and passes a match."""
    source = (
        f'from {_NAVIGATOR_LINK} import navigator_marker, resolve_sim_inst_config\n'
        'from spindoctor.config import DEFAULT_CONFIG\n'
    )
    assert _navigator_config_imports(source, package='spindoctor.sim.render') == [
        (1, f'{_NAVIGATOR_LINK}.resolve_sim_inst_config'),
        (2, 'spindoctor.config.DEFAULT_CONFIG'),
    ]


def _render_imports(source: str, *, package: str) -> list[tuple[int, str]]:
    """Return each import of a renderer module in ``source``.

    Parameters:
        source: Python source text to scan.
        package: The dotted package the source's module belongs to, against which
            a relative import resolves.

    Returns:
        List of ``(line number, module)`` tuples, one per imported renderer name.
    """
    imports: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            targets = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            module = _imported_module(node, package=package)
            targets = [f'{module}.{alias.name}' for alias in node.names]
        else:
            continue
        imports.extend(
            (node.lineno, target)
            for target in targets
            if _within(target, ('spindoctor.sim.render',))
        )
    return imports


def test_no_scene_module_imports_the_renderer() -> None:
    """No module of the scene package imports the renderer."""
    importing = [
        path.relative_to(_PACKAGE_ROOT).as_posix()
        for path in sorted((_PACKAGE_ROOT / _SCENE_DIR).rglob('*.py'))
        if _render_imports(path.read_text(encoding='utf-8'), package=_module_package(path))
    ]
    assert importing == []


def test_the_renderer_import_scan_flags_a_relative_import() -> None:
    """The scanner flags a scene module reading the renderer through a relative import."""
    imports = _render_imports(
        'from ..render.detector import sqrt_lut\n', package='spindoctor.sim.scene'
    )
    assert imports == [(1, 'spindoctor.sim.render.detector.sqrt_lut')]
