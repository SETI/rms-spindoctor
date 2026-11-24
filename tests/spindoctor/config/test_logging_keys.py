"""Tests for the logging configuration key namespace and its section type.

The schema core checks each value's type and is tested in its own suite.  What is
tested here is that the packaged section parses, the checks the section adds to the
core (module keys from the registries, program names, and level names), the merge of a
program's block, and that the loader refuses a bad override at load.
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

import spindoctor
from spindoctor.config.config import Config
from spindoctor.config.config_helper import load_default_and_user_config
from spindoctor.config.logging_config import BACKEND_NAMES
from spindoctor.config.logging_keys import (
    OTHER_LOG_KEYS,
    LoggingSection,
    log_key_for,
    model_log_keys,
    normalize_level,
    technique_log_keys,
)
from spindoctor.config.program_names import (
    PROGRAM_NAMES,
    SD_MOSAIC,
    SD_OFFSET,
    SD_RESULTS_INDEX,
)
from spindoctor.schema import SchemaError
from spindoctor.support.nav_base import NavBase


def _config_with_logging(tmp_path: Path, body: str) -> Config:
    """Build a Config whose logging section is overridden by ``body``.

    Parameters:
        tmp_path: Directory to write the override file into.
        body: YAML text placed under a ``logging:`` key.

    Returns:
        A loaded Config carrying the shipped defaults plus ``body``.
    """
    override = tmp_path / 'override.yaml'
    override.write_text(f'logging:\n{body}')
    config = Config()
    config.read_config()
    config.update_config(str(override))
    return config


# ---------------------------------------------------------------------------
# log_key_for
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ('class_name', 'expected'),
    [
        ('TitanHazeNav', 'titan_haze'),
        ('BodyDiscCorrelateNav', 'body_disc_correlate'),
        ('StarFieldFromCatalogNav', 'star_field_from_catalog'),
        ('NavTechniqueManual', 'manual'),
        ('NavModelStars', 'stars'),
        ('NavModelBody', 'body'),
    ],
)
def test_log_key_derived_from_class_name(class_name: str, expected: str) -> None:
    """A technique or model class derives the documented snake_case key."""
    cls = type(class_name, (), {})
    assert log_key_for(cls) == expected


def test_log_key_of_a_real_model_is_its_family() -> None:
    """A model class derives the bare family name."""
    assert log_key_for(type('NavModelRings', (), {})) == 'rings'


def test_log_key_of_a_simulated_model_is_the_same_family() -> None:
    """A simulated model collapses onto its real sibling's key, not a variant of it."""
    assert log_key_for(type('NavModelRingsSimulated', (), {})) == 'rings'


def test_log_key_strips_both_suffixes_in_either_order() -> None:
    """A name carrying Simulated and Nav reduces fully, whichever order they appear in."""
    assert log_key_for(type('FooSimulatedNav', (), {})) == 'foo'


def test_log_key_honors_a_declared_override() -> None:
    """An explicit log_key class attribute wins over the derived value."""
    cls = type('SomethingAwkwardNav', (), {'log_key': 'chosen_name'})
    assert log_key_for(cls) == 'chosen_name'


def test_log_key_honors_an_inherited_override() -> None:
    """A family sharing one key declares it once on their base."""
    parent = type('ParentNav', (), {'log_key': 'parent_key'})
    child = type('BodyLimbNav', (parent,), {})
    assert log_key_for(child) == 'parent_key'


def test_the_config_key_matches_what_the_component_reads() -> None:
    """The configuration namespace and the runtime lookup cannot disagree.

    A key honored at run time but not here would name a component the
    configuration rejects, while the key it did accept governed nothing.
    """
    parent = type('FamilyBaseNav', (NavBase,), {'log_key': 'family'})
    child = type('MemberOneNav', (parent,), {})
    assert log_key_for(child) == child().resolved_log_key


# ---------------------------------------------------------------------------
# Registry-derived key sets
# ---------------------------------------------------------------------------


def test_technique_keys_include_a_known_technique() -> None:
    """A registered autonomous technique appears in the technique key set."""
    assert 'titan_haze' in technique_log_keys()


def test_technique_keys_include_the_interactive_technique() -> None:
    """The interactive technique gets a key despite being outside the autonomous registry."""
    assert 'manual' in technique_log_keys()


def test_technique_keys_are_all_snake_case() -> None:
    """No derived technique key carries an upper-case character."""
    offenders = [key for key in technique_log_keys() if key != key.lower()]
    assert offenders == []


def test_model_keys_include_the_four_families() -> None:
    """The four shipped model families are present in the model key set."""
    assert model_log_keys() >= frozenset({'stars', 'body', 'rings', 'titan'})


def test_shipped_model_classes_map_to_exactly_four_families() -> None:
    """The shipped model classes collapse onto exactly four family keys."""
    # Scoped to shipped classes because the registry is process-global and test
    # modules elsewhere in the suite define their own NavModel subclasses, which
    # self-register and would otherwise make this assertion order-dependent.
    from spindoctor.nav_model.nav_model import NavModel

    shipped = {
        log_key_for(cls) for cls in NavModel._registry if cls.__module__.startswith('spindoctor.')
    }
    assert shipped == frozenset({'stars', 'body', 'rings', 'titan'})


def test_no_backend_key_leaked_into_other() -> None:
    """A per-image backend is not addressable as an "other" module key."""
    assert OTHER_LOG_KEYS.isdisjoint(BACKEND_NAMES)


# ---------------------------------------------------------------------------
# The section type
# ---------------------------------------------------------------------------


def test_packaged_logging_section_parses() -> None:
    """The packaged section parses, and names no program of its own."""
    section = Config().logging
    assert isinstance(section, LoggingSection)
    assert section.programs == {}


def test_shipped_defaults_preserve_the_annotation_level() -> None:
    """The shipped annotation level matches the behavior it replaces."""
    assert Config().logging.other['annotate'] == 'ERROR'


def test_shipped_defaults_disable_strict_scope() -> None:
    """Strict scope is off by default, so production warns rather than raising."""
    assert Config().logging.strict_scope is False


@pytest.mark.parametrize(
    ('category', 'key'),
    [('techniques', 'titan_haze'), ('models', 'rings'), ('other', 'correlate')],
)
def test_a_registered_module_key_is_accepted(tmp_path: Path, category: str, key: str) -> None:
    """Each category accepts the keys of its registry.

    Parameters:
        tmp_path: Directory to write the override file into.
        category: The category the override sets a level in.
        key: A module key the category's registry holds.
    """
    config = _config_with_logging(tmp_path, f'  {category}:\n    {key}: DEBUG\n')
    assert config.logging.categories()[category][key] == 'DEBUG'


def test_a_program_block_merges_over_the_top_level_settings(tmp_path: Path) -> None:
    """A program block that sets every key wins key by key; categories merge."""
    config = _config_with_logging(
        tmp_path,
        '  main: INFO\n  image: INFO\n  main_console: false\n  image_console: false\n'
        '  techniques:\n    body_limb: ERROR\n    titan_haze: ERROR\n'
        '  models:\n    rings: ERROR\n    stars: ERROR\n'
        '  other:\n    obs: ERROR\n'
        f'  programs:\n    {SD_MOSAIC}:\n      main: DEBUG\n      image: WARNING\n'
        '      main_console: true\n      image_console: true\n'
        '      techniques:\n        titan_haze: DEBUG\n'
        '      models:\n        stars: DEBUG\n'
        '      other:\n        annotate: DEBUG\n',
    )
    section = config.logging.for_program(SD_MOSAIC)
    assert section.main == 'DEBUG'
    assert section.image == 'WARNING'
    assert section.main_console is True
    assert section.image_console is True
    assert dict(section.techniques) == {'body_limb': 'ERROR', 'titan_haze': 'DEBUG'}
    assert dict(section.models) == {'rings': 'ERROR', 'stars': 'DEBUG'}
    assert dict(section.other) == {'obs': 'ERROR', 'annotate': 'DEBUG'}
    assert dict(section.programs) == {}
    assert section.strict_scope is config.logging.strict_scope


@pytest.mark.parametrize(
    ('body', 'expected'),
    [
        (
            '  techniques:\n    titan_hazey: DEBUG\n',
            "logging.techniques.titan_hazey: 'titan_hazey' is not a known techniques key",
        ),
        (
            '  models:\n    ringz: DEBUG\n',
            "logging.models.ringz: 'ringz' is not a known models key",
        ),
        (
            '  other:\n    reproj: DEBUG\n',
            "logging.other.reproj: 'reproj' is not a known other key",
        ),
        ('  main: CHATTY\n', "logging.main: 'CHATTY' is not one of"),
        (
            '  programs:\n    sd_nonesuch:\n      main: INFO\n',
            "logging.programs.sd_nonesuch: 'sd_nonesuch' is not a known program",
        ),
        (
            f'  programs:\n    {SD_OFFSET}:\n      strict_scope: true\n',
            f"logging.programs.{SD_OFFSET}: unknown keys: ['strict_scope']",
        ),
    ],
    ids=[
        'unknown-technique',
        'unknown-model',
        'backend-as-other',
        'unknown-level',
        'unknown-program',
        'strict-scope-per-program',
    ],
)
def test_a_bad_logging_override_is_refused_at_load(
    tmp_path: Path, body: str, expected: str
) -> None:
    """A key or level the section does not accept is refused with the file and key path.

    Parameters:
        tmp_path: Directory to write the override file into.
        body: YAML text placed under a ``logging:`` key.
        expected: The message's key path and problem.
    """
    override = tmp_path / 'override.yaml'
    with pytest.raises(SchemaError, match=re.escape(f'{override}: {expected}')):
        _config_with_logging(tmp_path, body)


def test_the_parse_reads_complete_registries_on_its_own(tmp_path: Path) -> None:
    """A process that has imported nothing but the configuration accepts every module key.

    The module keys come from the technique and model registries, which are filled as
    those packages are imported.  The configuration imports them itself when it
    checks a key, so a program need not import them before its configuration loads.
    """
    override = tmp_path / 'override.yaml'
    override.write_text(
        'logging:\n  techniques:\n    titan_haze: DEBUG\n  models:\n    titan: DEBUG\n'
    )
    script = (
        'import sys\n'
        'from spindoctor.config import Config\n'
        "assert 'spindoctor.nav_technique' not in sys.modules\n"
        'config = Config()\n'
        'config.update_config(sys.argv[1])\n'
        'print(sorted(config.logging.techniques), sorted(config.logging.models))\n'
    )
    package_root = str(Path(spindoctor.__file__).resolve().parent.parent)
    completed = subprocess.run(
        [sys.executable, '-c', script, str(override)],
        env={**os.environ, 'PYTHONPATH': package_root},
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "['titan_haze'] ['titan']"


# ---------------------------------------------------------------------------
# Level canonicalization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ('raw', 'expected'),
    [('debug', 'DEBUG'), ('  info  ', 'INFO'), ('WaRnInG', 'WARNING'), ('NONE', 'NONE')],
)
def test_normalize_level(raw: str, expected: str) -> None:
    """Level names canonicalize to upper case with surrounding whitespace removed."""
    assert normalize_level(raw) == expected


# ---------------------------------------------------------------------------
# Other-category keys name real components
# ---------------------------------------------------------------------------


def test_other_keys_name_no_dead_module() -> None:
    """The other category carries no key naming a module that does not exist."""
    assert 'nav_correlate_all' not in OTHER_LOG_KEYS


def test_other_keys_include_the_annotation_module() -> None:
    """Annotation opens its own section, so it is addressable."""
    assert 'annotate' in OTHER_LOG_KEYS


def test_every_other_key_names_a_component_that_opens_a_section() -> None:
    """A key naming a component that never opens a section would do nothing.

    Guards the invariant directly rather than restating a list: each key must
    appear either in a ``logged_section`` call or as a declared ``log_key``,
    which are the two ways a component gets a section for its level to be
    applied at.
    """
    import spindoctor

    root = Path(spindoctor.__file__).parent
    sources = '\n'.join(f.read_text() for f in root.rglob('*.py'))
    missing = [
        key
        for key in sorted(OTHER_LOG_KEYS)
        if f"logged_section('{key}'" not in sources
        and f"log_key: ClassVar[str] = '{key}'" not in sources
    ]
    assert missing == []


# ---------------------------------------------------------------------------
# Startup wiring
# ---------------------------------------------------------------------------


def test_startup_load_names_the_section_for_an_empty_override(tmp_path: Path) -> None:
    """A section written with no body is reported by name, not as an AttributeError."""
    override = tmp_path / 'empty_section.yaml'
    override.write_text('general:\n')
    arguments = argparse.Namespace(config_file=[str(override)])
    with pytest.raises(ValueError, match='general'):
        load_default_and_user_config(arguments, Config())


def test_startup_load_tolerates_a_namespace_without_config_file() -> None:
    """A bare Namespace is accepted, since callers legitimately omit config_file."""
    config = Config()
    load_default_and_user_config(argparse.Namespace(), config)
    assert config.is_loaded


def test_startup_load_accepts_a_good_logging_section(tmp_path: Path) -> None:
    """A valid override survives the startup load and reaches the merged config."""
    override = tmp_path / 'good.yaml'
    override.write_text('logging:\n  techniques:\n    titan_haze: DEBUG\n')
    arguments = argparse.Namespace(config_file=[str(override)])
    config = Config()
    load_default_and_user_config(arguments, config)
    assert config.logging.techniques['titan_haze'] == 'DEBUG'


# ---------------------------------------------------------------------------
# Program identities
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ('module_name', 'expected'),
    [
        ('sd_offset', SD_OFFSET),
        ('sd_offset_cloud_tasks', SD_OFFSET),
        ('sd_backplanes', 'sd_backplanes'),
        ('sd_backplanes_cloud_tasks', 'sd_backplanes'),
        ('sd_mosaic', SD_MOSAIC),
        ('sd_mosaic_cloud_tasks', SD_MOSAIC),
        ('sd_create_bundle', 'sd_create_bundle'),
        ('sd_create_ck', 'sd_create_ck'),
        ('sd_consolidate_metadata', 'sd_consolidate_metadata'),
        ('sd_results_index', SD_RESULTS_INDEX),
        ('sd_results_index_cloud_tasks', SD_RESULTS_INDEX),
    ],
)
def test_dispatch_module_declares_its_program_identity(module_name: str, expected: str) -> None:
    """Each dispatch module declares the program identity it shares logs and config with."""
    module = __import__(f'spindoctor.cli.{module_name}', fromlist=['PROGRAM_NAME'])
    declared = module.PROGRAM_NAME
    assert declared == expected


@pytest.mark.parametrize(
    'module_name',
    [
        'sd_offset',
        'sd_offset_cloud_tasks',
        'sd_backplanes',
        'sd_backplanes_cloud_tasks',
        'sd_mosaic',
        'sd_mosaic_cloud_tasks',
        'sd_create_bundle',
        'sd_create_ck',
        'sd_consolidate_metadata',
        'sd_results_index',
        'sd_results_index_cloud_tasks',
    ],
)
def test_declared_program_identity_is_registered(module_name: str) -> None:
    """Every declared identity is one the logging section accepts as a program key."""
    module = __import__(f'spindoctor.cli.{module_name}', fromlist=['PROGRAM_NAME'])
    assert module.PROGRAM_NAME in PROGRAM_NAMES
