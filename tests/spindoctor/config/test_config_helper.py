"""Tests for :mod:`spindoctor.config.config_helper`.

Covers the documented resolution order of the getters (explicit argument, then
``config.environment``, then environment variable) and the user-config loading
behavior of ``load_default_and_user_config`` (bundled defaults always; explicit
``--config-file`` paths when given; otherwise ``nav_default_config.yaml`` from
the current directory).

The results-root getters and the results-index getter share that order and part
ways at the end of it: an unresolved root is an error, while an unresolved index
URL is an answer -- no index was resolved -- and each caller decides for itself
whether it can proceed without one. ``sd_results_index`` writes an index and
refuses without one; every other consumer reads the results tree instead. A
value that is empty, or nothing but spaces, is the exception to that answer: it
is neither a URL nor the ``none`` that names no index, and is refused at
whichever level carries it.

The tests are hermetic: every test clears the results-root and results-index
environment variables via ``monkeypatch``, seeds ``Config`` instances
directly (never the ``DEFAULT_CONFIG`` singleton), and changes into
``tmp_path`` before exercising the relative ``nav_default_config.yaml``
lookup so the repository's real user-config file is never picked up.
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Callable
from pathlib import Path

import pytest

from spindoctor.config import Config
from spindoctor.config.config_helper import (
    RESULTS_INDEX_DB_NONE,
    get_backplane_results_root,
    get_nav_results_root,
    get_pds4_bundle_results_root,
    get_results_index_db_url,
    load_default_and_user_config,
)

RootGetter = Callable[[argparse.Namespace, Config], str]

ALL_ENV_VARS = (
    'NAV_RESULTS_ROOT',
    'NAV_BACKPLANE_RESULTS_ROOT',
    'NAV_BUNDLE_RESULTS_ROOT',
    'BUNDLE_RESULTS_ROOT',
    'NAV_RESULTS_INDEX_DB',
)

SQLITE_URL = 'sqlite:////data/nav-results/index.sqlite3'

POSTGRES_URL = 'postgresql+psycopg://user@host/spindoctor'

# (getter, argparse attribute, environment-section key, env var, CLI flag)
GETTER_CASES = [
    pytest.param(
        get_nav_results_root,
        'nav_results_root',
        'nav_results_root',
        'NAV_RESULTS_ROOT',
        '--nav-results-root',
        id='nav',
    ),
    pytest.param(
        get_backplane_results_root,
        'backplane_results_root',
        'backplane_results_root',
        'NAV_BACKPLANE_RESULTS_ROOT',
        '--backplane-results-root',
        id='backplane',
    ),
    pytest.param(
        get_pds4_bundle_results_root,
        'bundle_results_root',
        'bundle_results_root',
        'NAV_BUNDLE_RESULTS_ROOT',
        '--bundle-results-root',
        id='bundle',
    ),
]


@pytest.fixture(autouse=True)
def _clear_root_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove every results-root and results-index environment variable.

    Parameters:
        monkeypatch: Pytest fixture used to delete the environment variables
            for the duration of the test.
    """

    for env_var in ALL_ENV_VARS:
        monkeypatch.delenv(env_var, raising=False)


def _config_with_environment(environment: dict[str, str | None] | None) -> Config:
    """Build a ``Config`` whose ``environment`` section holds the given keys.

    The internal config dict is seeded directly so ``read_config`` never
    loads the bundled defaults, keeping the getter tests fast and hermetic.

    Parameters:
        environment: Mapping to expose as ``config.environment``, or None for
            an empty section.

    Returns:
        A ``Config`` instance with only the requested ``environment`` section.
    """

    config = Config()
    config._config_dict = {'environment': dict(environment or {})}
    config._update_attrdicts()
    return config


# ---------------------------------------------------------------------------
# Root getters: resolution order
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(('getter', 'arg_attr', 'config_key', 'env_var', 'cli_flag'), GETTER_CASES)
def test_getter_argument_wins_over_config_and_env(
    getter: RootGetter,
    arg_attr: str,
    config_key: str,
    env_var: str,
    cli_flag: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The parsed-argument value takes precedence over config and env var.

    Parameters:
        getter: The root getter under test.
        arg_attr: Name of the argparse attribute the getter reads.
        config_key: Key in ``config.environment`` the getter reads.
        env_var: Environment variable the getter reads.
        cli_flag: CLI flag named in the getter's error message (unused here).
        monkeypatch: Pytest fixture for environment isolation.
    """

    monkeypatch.setenv(env_var, '/from/env')
    config = _config_with_environment({config_key: '/from/config'})
    arguments = argparse.Namespace(**{arg_attr: '/from/args'})
    assert getter(arguments, config) == '/from/args'


@pytest.mark.parametrize(('getter', 'arg_attr', 'config_key', 'env_var', 'cli_flag'), GETTER_CASES)
def test_getter_config_wins_over_env_when_argument_is_none(
    getter: RootGetter,
    arg_attr: str,
    config_key: str,
    env_var: str,
    cli_flag: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the argument None, ``config.environment`` beats the env var.

    Parameters:
        getter: The root getter under test.
        arg_attr: Name of the argparse attribute the getter reads.
        config_key: Key in ``config.environment`` the getter reads.
        env_var: Environment variable the getter reads.
        cli_flag: CLI flag named in the getter's error message (unused here).
        monkeypatch: Pytest fixture for environment isolation.
    """

    monkeypatch.setenv(env_var, '/from/env')
    config = _config_with_environment({config_key: '/from/config'})
    arguments = argparse.Namespace(**{arg_attr: None})
    assert getter(arguments, config) == '/from/config'


@pytest.mark.parametrize(('getter', 'arg_attr', 'config_key', 'env_var', 'cli_flag'), GETTER_CASES)
def test_getter_missing_argument_attribute_uses_config(
    getter: RootGetter,
    arg_attr: str,
    config_key: str,
    env_var: str,
    cli_flag: str,
) -> None:
    """A namespace without the attribute at all falls through to the config.

    Parameters:
        getter: The root getter under test.
        arg_attr: Name of the argparse attribute the getter reads (unused here).
        config_key: Key in ``config.environment`` the getter reads.
        env_var: Environment variable the getter reads (unused here).
        cli_flag: CLI flag named in the getter's error message (unused here).
    """

    config = _config_with_environment({config_key: '/from/config'})
    assert getter(argparse.Namespace(), config) == '/from/config'


@pytest.mark.parametrize(('getter', 'arg_attr', 'config_key', 'env_var', 'cli_flag'), GETTER_CASES)
def test_getter_env_var_used_when_argument_and_config_unset(
    getter: RootGetter,
    arg_attr: str,
    config_key: str,
    env_var: str,
    cli_flag: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The env var is the last documented fallback before the error.

    Parameters:
        getter: The root getter under test.
        arg_attr: Name of the argparse attribute the getter reads (unused here).
        config_key: Key in ``config.environment`` the getter reads (unused here).
        env_var: Environment variable the getter reads.
        cli_flag: CLI flag named in the getter's error message (unused here).
        monkeypatch: Pytest fixture for environment isolation.
    """

    monkeypatch.setenv(env_var, '/from/env')
    config = _config_with_environment(None)
    assert getter(argparse.Namespace(), config) == '/from/env'


@pytest.mark.parametrize(('getter', 'arg_attr', 'config_key', 'env_var', 'cli_flag'), GETTER_CASES)
def test_getter_config_key_none_falls_through_to_env(
    getter: RootGetter,
    arg_attr: str,
    config_key: str,
    env_var: str,
    cli_flag: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A config key explicitly set to null is treated as unset.

    Parameters:
        getter: The root getter under test.
        arg_attr: Name of the argparse attribute the getter reads (unused here).
        config_key: Key in ``config.environment`` the getter reads.
        env_var: Environment variable the getter reads.
        cli_flag: CLI flag named in the getter's error message (unused here).
        monkeypatch: Pytest fixture for environment isolation.
    """

    monkeypatch.setenv(env_var, '/from/env')
    config = _config_with_environment({config_key: None})
    assert getter(argparse.Namespace(), config) == '/from/env'


@pytest.mark.parametrize(('getter', 'arg_attr', 'config_key', 'env_var', 'cli_flag'), GETTER_CASES)
def test_getter_raises_when_nothing_is_set(
    getter: RootGetter,
    arg_attr: str,
    config_key: str,
    env_var: str,
    cli_flag: str,
) -> None:
    """With no argument, config key, or env var the getter raises ValueError.

    The error message names the CLI flag, the configuration variable, and the
    environment variable so the operator knows every way to fix it.

    Parameters:
        getter: The root getter under test.
        arg_attr: Name of the argparse attribute the getter reads (unused here).
        config_key: Key in ``config.environment`` the getter reads (unused here).
        env_var: Environment variable named in the error message.
        cli_flag: CLI flag named in the error message.
    """

    config = _config_with_environment(None)
    with pytest.raises(ValueError, match=f'{cli_flag}.*{env_var}.*must be set'):
        getter(argparse.Namespace(**{arg_attr: None}), config)


# ---------------------------------------------------------------------------
# Root getters: corner cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(('getter', 'arg_attr', 'config_key', 'env_var', 'cli_flag'), GETTER_CASES)
def test_getter_empty_string_env_var_counts_as_set(
    getter: RootGetter,
    arg_attr: str,
    config_key: str,
    env_var: str,
    cli_flag: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty-string env var is returned verbatim, not treated as unset.

    The docstrings only distinguish set from unset (None); an exported but
    empty variable therefore satisfies the lookup and yields ''.

    Parameters:
        getter: The root getter under test.
        arg_attr: Name of the argparse attribute the getter reads (unused here).
        config_key: Key in ``config.environment`` the getter reads (unused here).
        env_var: Environment variable the getter reads.
        cli_flag: CLI flag named in the getter's error message (unused here).
        monkeypatch: Pytest fixture for environment isolation.
    """

    monkeypatch.setenv(env_var, '')
    config = _config_with_environment(None)
    assert getter(argparse.Namespace(), config) == ''


@pytest.mark.parametrize(('getter', 'arg_attr', 'config_key', 'env_var', 'cli_flag'), GETTER_CASES)
def test_getter_empty_string_argument_wins(
    getter: RootGetter,
    arg_attr: str,
    config_key: str,
    env_var: str,
    cli_flag: str,
) -> None:
    """An empty-string argument still outranks a configured value.

    Parameters:
        getter: The root getter under test.
        arg_attr: Name of the argparse attribute the getter reads.
        config_key: Key in ``config.environment`` the getter reads.
        env_var: Environment variable the getter reads (unused here).
        cli_flag: CLI flag named in the getter's error message (unused here).
    """

    config = _config_with_environment({config_key: '/from/config'})
    arguments = argparse.Namespace(**{arg_attr: ''})
    assert getter(arguments, config) == ''


@pytest.mark.parametrize('root_value', ['relative/results/dir', 'gs://bucket/nav-results'])
@pytest.mark.parametrize(('getter', 'arg_attr', 'config_key', 'env_var', 'cli_flag'), GETTER_CASES)
def test_getter_returns_value_verbatim(
    getter: RootGetter,
    arg_attr: str,
    config_key: str,
    env_var: str,
    cli_flag: str,
    root_value: str,
) -> None:
    """Relative paths and cloud URLs pass through without normalization.

    The getters promise only to return the string that was found; no
    absolutization or URL handling is documented or performed.

    Parameters:
        getter: The root getter under test.
        arg_attr: Name of the argparse attribute the getter reads.
        config_key: Key in ``config.environment`` the getter reads (unused here).
        env_var: Environment variable the getter reads (unused here).
        cli_flag: CLI flag named in the getter's error message (unused here).
        root_value: The relative-path or cloud-URL value to round-trip.
    """

    config = _config_with_environment(None)
    arguments = argparse.Namespace(**{arg_attr: root_value})
    assert getter(arguments, config) == root_value


def test_bundle_getter_env_var_is_nav_prefixed(monkeypatch: pytest.MonkeyPatch) -> None:
    """The bundle getter reads NAV_BUNDLE_RESULTS_ROOT, not BUNDLE_RESULTS_ROOT.

    The fallback environment variable carries the same ``NAV_`` prefix as the
    sibling getters; a value exported under the un-prefixed name is ignored.
    """

    monkeypatch.setenv('NAV_BUNDLE_RESULTS_ROOT', '/from/nav/env')
    config = _config_with_environment(None)
    result = get_pds4_bundle_results_root(argparse.Namespace(), config)
    assert result == '/from/nav/env'


def test_bundle_getter_ignores_unprefixed_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """A value exported under the un-prefixed name does not resolve the root."""

    monkeypatch.setenv('BUNDLE_RESULTS_ROOT', '/from/documented/env')
    config = _config_with_environment(None)
    with pytest.raises(ValueError, match='NAV_BUNDLE_RESULTS_ROOT'):
        get_pds4_bundle_results_root(argparse.Namespace(), config)


# ---------------------------------------------------------------------------
# get_results_index_db_url
# ---------------------------------------------------------------------------


def test_results_index_db_argument_wins_over_config_and_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The parsed-argument value takes precedence over config and env var."""

    monkeypatch.setenv('NAV_RESULTS_INDEX_DB', POSTGRES_URL)
    config = _config_with_environment({'results_index_db': 'sqlite:///from-config.sqlite3'})
    arguments = argparse.Namespace(results_index_db=SQLITE_URL)
    assert get_results_index_db_url(arguments, config) == SQLITE_URL


def test_results_index_db_config_wins_over_env_when_argument_is_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the argument None, ``config.environment`` beats the env var."""

    monkeypatch.setenv('NAV_RESULTS_INDEX_DB', POSTGRES_URL)
    config = _config_with_environment({'results_index_db': SQLITE_URL})
    assert get_results_index_db_url(argparse.Namespace(results_index_db=None), config) == SQLITE_URL


def test_results_index_db_missing_argument_attribute_uses_config() -> None:
    """A namespace without the attribute at all falls through to the config.

    A program that has not yet grown the flag still honors the configuration key.
    """

    config = _config_with_environment({'results_index_db': SQLITE_URL})
    assert get_results_index_db_url(argparse.Namespace(), config) == SQLITE_URL


def test_results_index_db_env_var_used_when_argument_and_config_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The env var is the last place looked before answering "no index"."""

    monkeypatch.setenv('NAV_RESULTS_INDEX_DB', POSTGRES_URL)
    config = _config_with_environment(None)
    assert get_results_index_db_url(argparse.Namespace(), config) == POSTGRES_URL


def test_results_index_db_config_key_none_falls_through_to_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A config key explicitly set to null is treated as unset, not as opted out."""

    monkeypatch.setenv('NAV_RESULTS_INDEX_DB', POSTGRES_URL)
    config = _config_with_environment({'results_index_db': None})
    assert get_results_index_db_url(argparse.Namespace(), config) == POSTGRES_URL


def test_results_index_db_unset_everywhere_is_not_an_error() -> None:
    """Absence is an answer here: no index was resolved.

    This is the one way this getter differs from the results-root getters, which
    raise when nothing is set. What a caller does with the answer is the
    caller's: the statistics programs refuse, and the pipeline programs read
    files.
    """

    config = _config_with_environment(None)
    assert get_results_index_db_url(argparse.Namespace(results_index_db=None), config) is None


def test_results_index_db_sentinel_on_the_command_line_overrides_the_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without the sentinel an exported variable would make file mode impossible."""

    monkeypatch.setenv('NAV_RESULTS_INDEX_DB', POSTGRES_URL)
    config = _config_with_environment(None)
    arguments = argparse.Namespace(results_index_db=RESULTS_INDEX_DB_NONE)
    assert get_results_index_db_url(arguments, config) is None


def test_results_index_db_sentinel_on_the_command_line_overrides_the_config_key() -> None:
    """The sentinel outranks a configured URL for the same reason."""

    config = _config_with_environment({'results_index_db': SQLITE_URL})
    arguments = argparse.Namespace(results_index_db=RESULTS_INDEX_DB_NONE)
    assert get_results_index_db_url(arguments, config) is None


def test_results_index_db_sentinel_in_the_config_opts_out(monkeypatch: pytest.MonkeyPatch) -> None:
    """The sentinel is honored wherever the value came from, not only on argv."""

    monkeypatch.setenv('NAV_RESULTS_INDEX_DB', POSTGRES_URL)
    config = _config_with_environment({'results_index_db': RESULTS_INDEX_DB_NONE})
    assert get_results_index_db_url(argparse.Namespace(), config) is None


def test_results_index_db_sentinel_in_the_env_var_opts_out(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exporting the sentinel is how a machine turns the index off by default."""

    monkeypatch.setenv('NAV_RESULTS_INDEX_DB', RESULTS_INDEX_DB_NONE)
    config = _config_with_environment(None)
    assert get_results_index_db_url(argparse.Namespace(), config) is None


def test_results_index_db_sentinel_is_the_exact_literal() -> None:
    """A URL that merely contains the word is a URL, not an opt-out."""

    config = _config_with_environment(None)
    arguments = argparse.Namespace(results_index_db='sqlite:///none.sqlite3')
    assert get_results_index_db_url(arguments, config) == 'sqlite:///none.sqlite3'


@pytest.mark.parametrize('empty', ['', '   '])
def test_results_index_db_set_to_nothing_is_refused(empty: str) -> None:
    """An empty value is never a value anyone meant to write.

    The sentinel is how a level says "no index" on purpose, so an empty one is a
    typo, a script that computed nothing, or a variable half unset.

    Parameters:
        empty: A value carrying no URL.
    """

    config = _config_with_environment(None)
    with pytest.raises(ValueError, match='empty value'):
        get_results_index_db_url(argparse.Namespace(results_index_db=empty), config)


@pytest.mark.parametrize('empty', ['', '   '])
def test_results_index_db_set_to_nothing_does_not_fall_through_to_the_next_level(
    empty: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The level that set it said something, so a lower level does not answer for it.

    An operator who writes an empty option is not asking for whatever the
    machine exports, and the refusal names the option rather than the variable.

    Parameters:
        empty: A value carrying no URL.
        monkeypatch: Fixture the exported variable is set through.
    """

    monkeypatch.setenv('NAV_RESULTS_INDEX_DB', POSTGRES_URL)
    config = _config_with_environment(None)
    with pytest.raises(ValueError, match='--results-index-db'):
        get_results_index_db_url(argparse.Namespace(results_index_db=empty), config)


def test_results_index_db_set_to_nothing_names_the_environment_variable_that_set_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A refusal that does not say which level carries the value is a hunt.

    Parameters:
        monkeypatch: Fixture the exported variable is set through.
    """

    monkeypatch.setenv('NAV_RESULTS_INDEX_DB', '')
    config = _config_with_environment(None)
    with pytest.raises(ValueError, match='NAV_RESULTS_INDEX_DB'):
        get_results_index_db_url(argparse.Namespace(), config)


def test_results_index_db_set_to_nothing_names_the_configuration_variable_that_set_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The middle level is named the same way, since one unset does not fix it.

    A URL is exported below it so that a resolver falling through to the
    environment would answer with that URL instead of refusing, which is the
    regression this would otherwise pass over.

    Parameters:
        monkeypatch: Fixture the exported variable is set through.
    """

    monkeypatch.setenv('NAV_RESULTS_INDEX_DB', POSTGRES_URL)
    config = _config_with_environment({'results_index_db': ''})
    with pytest.raises(ValueError, match=re.escape('environment.results_index_db')):
        get_results_index_db_url(argparse.Namespace(), config)


def test_results_index_db_set_to_nothing_says_how_to_ask_for_no_index() -> None:
    """The sentinel is what an operator writes to mean this on purpose."""

    config = _config_with_environment({'results_index_db': ''})
    with pytest.raises(ValueError, match=RESULTS_INDEX_DB_NONE):
        get_results_index_db_url(argparse.Namespace(), config)


def test_results_index_db_set_to_nothing_is_reported_by_the_caller_and_nowhere_else(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A refusal that also logged itself would print twice, once out of order.

    Parameters:
        capsys: Fixture the main log would be read back from.
    """

    config = _config_with_environment({'results_index_db': ''})
    with pytest.raises(ValueError, match='empty value'):
        get_results_index_db_url(argparse.Namespace(), config)
    assert capsys.readouterr().out == ''


def test_results_index_db_set_to_nothing_leaves_what_follows_to_the_caller() -> None:
    """Two of the three programs that resolve a URL have no file-reading mode.

    One resolver serves all of them, so a message saying what this run will do
    next is false for the two that refuse.
    """

    config = _config_with_environment({'results_index_db': ''})
    with pytest.raises(ValueError) as refusal:
        get_results_index_db_url(argparse.Namespace(), config)
    assert 'reads the navigation results files' not in str(refusal.value)


@pytest.mark.parametrize('spaced', [' none', 'none ', '  none  '])
def test_results_index_db_sentinel_written_with_spaces_around_it_opts_out(spaced: str) -> None:
    """The spaces a shell or a configuration file leaves behind are not the value.

    The value is read through the spaces around it, so the sentinel is recognized
    rather than passed on as a URL that cannot be parsed.

    Parameters:
        spaced: The sentinel as some level wrote it.
    """

    config = _config_with_environment(None)
    assert get_results_index_db_url(argparse.Namespace(results_index_db=spaced), config) is None


@pytest.mark.parametrize('url', [SQLITE_URL, POSTGRES_URL])
def test_results_index_db_returns_the_url_verbatim(url: str) -> None:
    """Both URL forms pass through unchanged; the opener parses them, not this.

    Parameters:
        url: The URL to round-trip.
    """

    config = _config_with_environment(None)
    assert get_results_index_db_url(argparse.Namespace(results_index_db=url), config) == url


# ---------------------------------------------------------------------------
# load_default_and_user_config
# ---------------------------------------------------------------------------


def test_load_bundled_defaults_without_user_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing ./nav_default_config.yaml is tolerated and defaults load.

    Parameters:
        tmp_path: Empty directory used as the current working directory.
        monkeypatch: Pytest fixture used to change the working directory.
    """

    monkeypatch.chdir(tmp_path)
    config = Config()
    load_default_and_user_config(argparse.Namespace(), config)
    assert config.is_loaded is True
    assert config.override_paths == ()


def test_load_applies_user_default_config_from_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """nav_default_config.yaml in the current directory overrides defaults.

    Parameters:
        tmp_path: Directory holding the generated user config file.
        monkeypatch: Pytest fixture used to change the working directory.
    """

    monkeypatch.chdir(tmp_path)
    (tmp_path / 'nav_default_config.yaml').write_text(
        'environment:\n  nav_results_root: /user/results\n', encoding='utf-8'
    )
    config = Config()
    load_default_and_user_config(argparse.Namespace(), config)
    assert config.environment.nav_results_root == '/user/results'
    assert config.override_paths == ('nav_default_config.yaml',)


def test_load_user_default_preserves_bundled_siblings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """User overrides deep-merge: untouched bundled keys survive.

    Parameters:
        tmp_path: Directory holding the generated user config file.
        monkeypatch: Pytest fixture used to change the working directory.
    """

    monkeypatch.chdir(tmp_path)
    (tmp_path / 'nav_default_config.yaml').write_text(
        'general:\n  truetype_font_dir: /user/fonts\n', encoding='utf-8'
    )
    config = Config()
    load_default_and_user_config(argparse.Namespace(), config)
    assert config.general.truetype_font_dir == '/user/fonts'
    assert config.logging.other['annotate'] == 'ERROR'


def test_load_explicit_config_files_apply_in_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every --config-file path is applied, later files winning on conflicts.

    Parameters:
        tmp_path: Directory holding the generated override files.
        monkeypatch: Pytest fixture used to change the working directory.
    """

    monkeypatch.chdir(tmp_path)
    first = tmp_path / 'first.yaml'
    first.write_text('environment:\n  nav_results_root: /first\n', encoding='utf-8')
    second = tmp_path / 'second.yaml'
    second.write_text('environment:\n  nav_results_root: /second\n', encoding='utf-8')
    config = Config()
    arguments = argparse.Namespace(config_file=[str(first), str(second)])
    load_default_and_user_config(arguments, config)
    assert config.environment.nav_results_root == '/second'
    assert config.override_paths == (str(first), str(second))


def test_load_explicit_config_files_skip_user_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When --config-file is given, nav_default_config.yaml is not read.

    Parameters:
        tmp_path: Directory holding both the explicit override and the
            user-default file that must be ignored.
        monkeypatch: Pytest fixture used to change the working directory.
    """

    monkeypatch.chdir(tmp_path)
    explicit = tmp_path / 'explicit.yaml'
    explicit.write_text('environment:\n  nav_results_root: /explicit\n', encoding='utf-8')
    (tmp_path / 'nav_default_config.yaml').write_text(
        'environment:\n  nav_results_root: /user\n  backplane_results_root: /user-bp\n',
        encoding='utf-8',
    )
    config = Config()
    load_default_and_user_config(argparse.Namespace(config_file=[str(explicit)]), config)
    assert config.environment.nav_results_root == '/explicit'
    assert 'backplane_results_root' not in config.environment


def test_load_missing_explicit_config_file_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the implicit user default is optional; explicit paths must exist.

    Parameters:
        tmp_path: Empty directory used as the current working directory.
        monkeypatch: Pytest fixture used to change the working directory.
    """

    monkeypatch.chdir(tmp_path)
    config = Config()
    missing = str(tmp_path / 'no_such_config.yaml')
    arguments = argparse.Namespace(config_file=[missing])
    with pytest.raises(FileNotFoundError, match=r'no_such_config\.yaml'):
        load_default_and_user_config(arguments, config)


def test_load_malformed_user_default_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A user config that is not a YAML mapping raises ValueError.

    The docstring is silent on malformed files; only FileNotFoundError is
    suppressed, so the non-mapping diagnostic from Config propagates.

    Parameters:
        tmp_path: Directory holding the malformed user config file.
        monkeypatch: Pytest fixture used to change the working directory.
    """

    monkeypatch.chdir(tmp_path)
    (tmp_path / 'nav_default_config.yaml').write_text('- not\n- a\n- mapping\n', encoding='utf-8')
    config = Config()
    with pytest.raises(ValueError, match='expected a mapping at the top level'):
        load_default_and_user_config(argparse.Namespace(), config)


def test_load_config_file_none_still_loads_user_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """config_file=None (flag defined but not passed) must load the user default.

    Every sd_* CLI defines --config-file with default=None and documents 'If
    not provided, attempts to load ./nav_default_config.yaml if present'
    (CLAUDE.md agrees: user overrides come from nav_default_config.yaml). The
    implementation's early return fires whenever the attribute exists, even
    when it is None/empty, so real CLI runs never load the user file.

    Parameters:
        tmp_path: Directory holding the user config file that should load.
        monkeypatch: Pytest fixture used to change the working directory.
    """

    monkeypatch.chdir(tmp_path)
    (tmp_path / 'nav_default_config.yaml').write_text(
        'environment:\n  nav_results_root: /user/results\n', encoding='utf-8'
    )
    config = Config()
    load_default_and_user_config(argparse.Namespace(config_file=None), config)
    assert config.environment.nav_results_root == '/user/results'


def test_loaded_user_config_feeds_getter_before_env_var(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End to end: a user-config root outranks the environment variable.

    Parameters:
        tmp_path: Directory holding the generated user config file.
        monkeypatch: Pytest fixture for cwd and environment isolation.
    """

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('NAV_RESULTS_ROOT', '/from/env')
    (tmp_path / 'nav_default_config.yaml').write_text(
        'environment:\n  nav_results_root: /from/user/config\n', encoding='utf-8'
    )
    config = Config()
    load_default_and_user_config(argparse.Namespace(), config)
    result = get_nav_results_root(argparse.Namespace(nav_results_root=None), config)
    assert result == '/from/user/config'
