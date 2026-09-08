import argparse
import os
from dataclasses import fields

from filecache import FCPath

from spindoctor.nav_records.tuning import TreeTuning

from .config import Config
from .logging_keys import validate_logging_config

RESULTS_INDEX_DB_NONE = 'none'
"""Value of the results index URL that makes :func:`get_results_index_db_url` answer None.

An exported NAV_RESULTS_INDEX_DB would otherwise reach every program on the machine, and
one that resolves a URL never falls back to reading files.  What "no index" then
means belongs to each caller: a program with a file-reading path takes it, and one
without a file-reading path refuses.
"""


def get_backplane_results_root(arguments: argparse.Namespace, config: Config) -> str:
    """Get the backplane results root from the arguments, configuration, or environment.

    First look in arguments.backplane_results_root, then in
    config.environment.backplane_results_root, then in the environment variable
    NAV_BACKPLANE_RESULTS_ROOT.

    Parameters:
        arguments: The parsed arguments. config: The configuration possibly containing the
        environment section.

    Returns:
        The backplane results root.

    Raises:
        ValueError: If the backplane results root cannot be determined.
    """
    backplane_results_root_str = None
    try:
        backplane_results_root_str = arguments.backplane_results_root
    except AttributeError:
        pass
    if backplane_results_root_str is None:
        try:
            backplane_results_root_str = config.environment.backplane_results_root
        except AttributeError:
            pass
    if backplane_results_root_str is None:
        backplane_results_root_str = os.getenv('NAV_BACKPLANE_RESULTS_ROOT')
    if backplane_results_root_str is None:
        raise ValueError(
            'One of --backplane-results-root, the configuration variable '
            '"environment.backplane_results_root", or the NAV_BACKPLANE_RESULTS_ROOT '
            'environment variable must be set'
        )
    return backplane_results_root_str


def get_nav_results_root(arguments: argparse.Namespace, config: Config) -> str:
    """Get the navigation root from the arguments, configuration, or environment.

    First look in arguments.nav_results_root, then in config.environment.nav_results_root,
    then in the environment variable NAV_RESULTS_ROOT.

    Parameters:
        arguments: The parsed arguments. config: The configuration possibly containing the
        environment section.

    Returns:
        The navigation results root.

    Raises:
        ValueError: If the navigation results root cannot be determined.
    """
    nav_results_root_str = None
    try:
        nav_results_root_str = arguments.nav_results_root
    except AttributeError:
        pass
    if nav_results_root_str is None:
        try:
            nav_results_root_str = config.environment.nav_results_root
        except AttributeError:
            pass
    if nav_results_root_str is None:
        nav_results_root_str = os.getenv('NAV_RESULTS_ROOT')
    if nav_results_root_str is None:
        raise ValueError(
            'One of --nav-results-root, the configuration variable '
            '"environment.nav_results_root", or the NAV_RESULTS_ROOT '
            'environment variable must be set'
        )
    return nav_results_root_str


def get_log_root(arguments: argparse.Namespace, config: Config) -> str:
    """Get the log root from the arguments, configuration, or environment.

    First look in ``arguments.log_root``, then in ``config.environment.log_root``,
    then in the environment variable ``NAV_LOG_ROOT``.  Unlike the other roots
    this one has a fallback rather than an error: logs belong under the
    navigation results root by default, so a run that has not been told where to
    put them still puts them somewhere predictable.

    Parameters:
        arguments: The parsed arguments.
        config: The configuration possibly containing the environment section.

    Returns:
        The log root.

    Raises:
        ValueError: If neither a log root nor a navigation results root can be
            determined.
    """
    log_root_str = None
    try:
        log_root_str = arguments.log_root
    except AttributeError:
        pass
    if log_root_str is None:
        try:
            log_root_str = config.environment.log_root
        except AttributeError:
            pass
    if log_root_str is None:
        log_root_str = os.getenv('NAV_LOG_ROOT')
    if log_root_str is None:
        # FCPath rather than os.path.join: a results root is routinely a cloud
        # URL, and joining those must not depend on the local path separator.
        log_root_str = (FCPath(get_nav_results_root(arguments, config)) / 'logs').as_posix()
    return str(log_root_str)


def get_pds4_bundle_results_root(arguments: argparse.Namespace, config: Config) -> str:
    """Get the PDS4 bundle root from the arguments, configuration, or environment.

    First look in arguments.bundle_results_root, then in
    config.environment.bundle_results_root, then in the environment variable
    NAV_BUNDLE_RESULTS_ROOT.

    Parameters:
        arguments: The parsed arguments. config: The configuration possibly containing the
        environment section.

    Returns:
        The PDS4 bundle root.

    Raises:
        ValueError: If the PDS4 bundle root cannot be determined.
    """
    pds4_bundle_root_str = None
    try:
        pds4_bundle_root_str = arguments.bundle_results_root
    except AttributeError:
        pass
    if pds4_bundle_root_str is None:
        try:
            pds4_bundle_root_str = config.environment.bundle_results_root
        except AttributeError:
            pass
    if pds4_bundle_root_str is None:
        pds4_bundle_root_str = os.getenv('NAV_BUNDLE_RESULTS_ROOT')
    if pds4_bundle_root_str is None:
        raise ValueError(
            'One of --bundle-results-root, the configuration variable '
            '"environment.bundle_results_root", or the NAV_BUNDLE_RESULTS_ROOT '
            'environment variable must be set'
        )
    return pds4_bundle_root_str


def get_results_index_db_url(arguments: argparse.Namespace, config: Config) -> str | None:
    """Get the results index URL from the arguments, configuration, or environment.

    First look in arguments.results_index_db, then in
    config.environment.results_index_db, then in the environment variable
    NAV_RESULTS_INDEX_DB.

    Unlike the results roots, absence is not an error: it means no index was
    resolved, and each caller decides whether it can proceed without one.  The
    literal value ``none`` resolves to the same answer, so a run on a machine that
    exports NAV_RESULTS_INDEX_DB can still be told to read files by passing
    ``--results-index-db none``.  The sentinel is honored wherever the value came
    from, so a configuration file can opt out of an exported variable in the same
    way; surrounding spaces are not part of it, and it is otherwise matched as the
    exact string, so a URL that merely contains the word is still a URL.

    A value that is empty, or nothing but spaces, is refused rather than read as
    naming no index.  ``none`` is the deliberate spelling of "no index" and is
    honored at all three levels, so a machine that means to run without one
    already has a way to say so; an empty value is a typo, a script that computed
    nothing, or a variable half unset.  Answering it with a warning instead would
    put one line in a batch log and then read a different source than the operator
    believes for as long as the run lasts, which on a cloud root is hours.
    Refusing fails every run on a machine configured that way, which is the point:
    one unset fixes it, and it is found on the first run rather than after a long
    batch has quietly read the tree.  The refusal names the level that supplied
    the value and says what to write instead, so it is read once and fixed once.

    Parameters:
        arguments: The parsed arguments.
        config: The configuration possibly containing the environment section.

    Returns:
        The results index connection URL, or None when no index was named.

    Raises:
        ValueError: If a level supplied a value that is empty or nothing but
            spaces.
    """
    # Absence is the ordinary case at both levels -- most programs define no
    # --results-index-db argument, and most configurations name no index -- so each is
    # asked for the key rather than made to raise for it, which would also hide an
    # AttributeError raised by something other than the lookup.
    named_by = '--results-index-db'
    results_index_db_str = vars(arguments).get('results_index_db')
    if results_index_db_str is None:
        named_by = 'the environment.results_index_db configuration variable'
        results_index_db_str = config.environment.get('results_index_db')
    if results_index_db_str is None:
        named_by = 'the NAV_RESULTS_INDEX_DB environment variable'
        results_index_db_str = os.getenv('NAV_RESULTS_INDEX_DB')
    if results_index_db_str is None:
        return None
    url = str(results_index_db_str)
    if not url.strip():
        raise ValueError(
            f'{named_by} is set to an empty value, which is neither a connection URL '
            f'nor the way to name no index. Write {RESULTS_INDEX_DB_NONE} to name no index, '
            f'or a connection URL to name one.'
        )
    if url.strip() == RESULTS_INDEX_DB_NONE:
        return None
    return url


def get_results_tree_tuning(config: Config) -> TreeTuning:
    """Get how much of a pass over a results tree runs at once from the configuration.

    Read from ``config.results_tree`` alone: no command-line option or
    environment variable names these, because they describe a machine rather
    than a run.  A program resolves them once, here, and passes the result to
    whatever reads the tree.

    Parameters:
        config: The configuration, whose ``results_tree`` section is read.

    Returns:
        The tuning, with every setting the section omits at its default.

    Raises:
        ValueError: If the section names a setting that does not exist, or a
            value no pass can run at -- one that is not a positive integer,
            ``null`` included, or a round of work smaller than the pool it
            feeds.  The message names the section and the setting.
    """
    section = config.results_tree
    known = [field.name for field in fields(TreeTuning)]
    # Sorted as text: a YAML key can be a number, and a number does not sort
    # against a string.
    unknown = sorted(str(name) for name in section if name not in known)
    if unknown:
        raise ValueError(
            f'configuration section results_tree names no setting called {unknown[0]!r}; '
            f'the settings are {", ".join(known)}'
        )
    try:
        return TreeTuning(**section)
    except ValueError as exc:
        raise ValueError(f'configuration section results_tree: {exc}') from exc


def load_default_and_user_config(arguments: argparse.Namespace, config: Config) -> None:
    """Load the default and user configuration (if any).

    The merged result's ``logging`` and ``results_tree`` sections are validated
    before returning, so a misspelled module key, program name, level name or
    tuning setting fails here rather than having no effect at the point it was
    meant to apply, and a value no pass can run at fails before a run has
    written anything.

    A named file that cannot be read is not skipped in favor of the defaults;
    ``Config``'s own diagnostic propagates, so a missing file still raises
    ``FileNotFoundError`` and a file that is not a mapping still raises the
    ``ValueError`` naming it.  Only the implicit user default is optional.

    Parameters:
        arguments: The parsed arguments, which may carry a ``config_file``
            attribute.  Callers that construct a bare ``Namespace`` need not
            supply it.
        config: The configuration to update.

    Raises:
        ValueError: If the merged ``logging`` or ``results_tree`` section is
            not valid.
    """
    config.read_config()
    # If the user specified one or more config files, load them; if they didn't,
    # load the default config file.  getattr rather than attribute access
    # because callers legitimately pass a Namespace with no config_file at all,
    # and rather than try/except so an error raised deeper in the load cannot be
    # mistaken for the argument simply being absent.
    config_files = getattr(arguments, 'config_file', None)
    if config_files:
        for config_file in config_files:
            config.update_config(config_file)
    else:
        try:
            config.update_config('nav_default_config.yaml')
        except FileNotFoundError:
            pass
    validate_logging_config(config)
    # Built for the check alone and discarded.  The loader runs before a
    # program has written anything, so an unusable value is refused here.
    get_results_tree_tuning(config)
