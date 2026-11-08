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

A third scan covers imports.  The simulated instrument catalog and the artifact-mode
registry describe how the image side renders a frame, so no navigator-side module
imports ``spindoctor.sim.instrument_*`` or ``spindoctor.sim.artifact_modes``.  A
navigator-side model reads what the navigator would know from its own config.

Legitimate exceptions, should one ever exist, go in ``_ALLOWED_REFERENCES``:
a mapping from ``(module path relative to the spindoctor package, name)`` to a
one-line justification.  The allowlist is currently empty -- no navigator-side
module has a sanctioned reason to name a truth attribute.
"""

import ast
from pathlib import Path

import pytest

import spindoctor
from spindoctor.sim.scene import ATMOSPHERE_KEYS

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

# The image-side simulator modules no navigator-side module may import: the
# instrument catalog and its readers, and the artifact-mode registry.
_IMAGE_SIDE_MODULE_PREFIX = 'spindoctor.sim.instrument_'
_IMAGE_SIDE_MODULES: frozenset[str] = frozenset({'spindoctor.sim.artifact_modes'})


def _image_side_imports(source: str) -> list[tuple[int, str]]:
    """Return each import of an image-side simulator module in ``source``.

    Parameters:
        source: Python source text to scan.

    Returns:
        List of ``(line number, module)`` tuples, one per imported module.
    """
    imports: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            # ``from spindoctor.sim import instrument_catalog`` imports the module itself.
            modules = [node.module, *(f'{node.module}.{alias.name}' for alias in node.names)]
        else:
            continue
        imports.extend(
            (node.lineno, module)
            for module in modules
            if module.startswith(_IMAGE_SIDE_MODULE_PREFIX) or module in _IMAGE_SIDE_MODULES
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
def test_navigator_package_names_no_atmosphere_truth_key(package: str) -> None:
    """No navigator-side module names a key of the haze truth block.

    The structural boundary filter drops the whole ``atmosphere`` mapping,
    so a navigator-side read of one of its keys can only come from some
    other channel -- and would be a leak the atomic-block test cannot see.
    """
    violations = _package_violations(
        package, denied=ATMOSPHERE_TRUTH_KEY_NAMES, label='atmosphere truth key'
    )
    assert not violations, (
        'navigator-side code names simulated-scene haze truth keys (the haze '
        'is nature, and the predicted geometry is a deliberate mismatch for '
        'it):\n' + '\n'.join(violations)
    )


@pytest.mark.parametrize('package', _NAVIGATOR_PACKAGES)
def test_navigator_package_imports_no_image_side_simulator_module(package: str) -> None:
    """No navigator-side module imports the instrument catalog or the mode registry."""
    violations = [
        f'{path.relative_to(_PACKAGE_ROOT).as_posix()}:{lineno}: imports {module}'
        for path in sorted((_PACKAGE_ROOT / package).rglob('*.py'))
        for lineno, module in _image_side_imports(path.read_text(encoding='utf-8'))
    ]
    assert not violations, (
        'navigator-side code imports an image-side simulator module:\n' + '\n'.join(violations)
    )


def test_import_guard_flags_a_module_import() -> None:
    """The import scanner detects ``from spindoctor.sim import instrument_catalog``."""
    imports = _image_side_imports('from spindoctor.sim import instrument_catalog\n')
    assert imports == [(1, 'spindoctor.sim.instrument_catalog')]


def test_atmosphere_guard_covers_the_structure_keys() -> None:
    """The scanned name set includes every symmetry-breaking structure key.

    The single-letter phase-asymmetry key is deliberately excluded (see the
    deny-set comment); every key that names structure is long enough to
    scan for, so none may be dropped silently.
    """
    from spindoctor.sim.forward.haze_structure import HAZE_STRUCTURE_KEYS

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
    for rel, _name in _ALLOWED_REFERENCES:
        assert (_PACKAGE_ROOT / rel).is_file(), f'stale allowlist entry: {rel}'
