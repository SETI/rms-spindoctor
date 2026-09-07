"""The logging command-line surface shared by every program that has a logger.

One helper adds the same arguments everywhere, so a user who learns them for
one program knows them for the rest.  A program that processes images
individually accepts the image-logger flags as well; one that does not rejects
them by name rather than accepting and ignoring them, which would leave
someone believing they had changed something.

The flags select which sinks each logger writes to and what level it writes
at.  Both sinks of a logger always share a level, so there is no per-sink
level to set; see :mod:`spindoctor.config.logging_config` for how a level is
resolved from these arguments and the configuration together.
"""

import argparse
import sys
from collections.abc import Iterator
from contextlib import contextmanager

from spindoctor.config.logging_keys import LOG_LEVEL_NAMES

__all__ = ['add_logging_arguments', 'reporting_configuration_errors']


@contextmanager
def reporting_configuration_errors() -> Iterator[None]:
    """Report a bad configuration setting as a message rather than a traceback.

    Every refusal the configuration surface raises names the offending key --
    an unknown component or an unusable tuning value in a configuration file,
    an unknown level on the command line -- so all of them should be presented
    the same way.  Without this the configuration-file half reaches the
    terminal as a stack trace, which reads as a crash rather than as the thing
    the operator typed.

    Wrap only the calls that load the configuration and read logging settings.
    Deriving a results root also raises ValueError, and reporting that as a
    configuration problem would send the reader to the wrong place.

    Yields:
        None.
    """
    try:
        yield
    except (TypeError, ValueError) as exc:
        print(f'Invalid configuration: {exc}', file=sys.stderr)
        sys.exit(1)


_LEVEL_CHOICES = ', '.join(sorted(LOG_LEVEL_NAMES))


def add_logging_arguments(
    parser: argparse.ArgumentParser, *, has_image_logger: bool = True
) -> None:
    """Add the shared logging arguments to ``parser``.

    Parameters:
        parser: The parser to add the arguments to.
        has_image_logger: Whether this program processes images individually.
            False omits the image-logger flags, so passing one is an error
            naming the flag rather than a silently ignored request.

    """
    group = parser.add_argument_group('Logging')

    group.add_argument(
        '--log-root',
        type=str,
        default=None,
        metavar='PATH',
        help="""Root directory for this run's log files; overrides the
        environment.log_root configuration variable and the NAV_LOG_ROOT
        environment variable. Defaults to a "logs" directory under the
        navigation results root.""",
    )
    group.add_argument(
        '--log-main-to-console',
        action=argparse.BooleanOptionalAction,
        default=None,
        help="""Write the main log to the terminal (default: yes). The main log
        reports what the program is doing at the top level.""",
    )
    group.add_argument(
        '--log-main-to-file',
        action=argparse.BooleanOptionalAction,
        default=None,
        help='Write the main log to a file under the log root (default: yes).',
    )
    group.add_argument(
        '--log-level',
        action='append',
        default=None,
        metavar='LEVEL|MODULE=LEVEL',
        help=f"""Log level. Given alone it sets the default for both loggers;
        given as MODULE=LEVEL it sets one module, for example
        "--log-level titan_haze=DEBUG". May be repeated, so "--log-level DEBUG
        --log-level titan_haze=INFO" means everything at DEBUG except that
        technique. Levels are {_LEVEL_CHOICES}.""",
    )
    group.add_argument(
        '--log-level-main',
        type=str,
        default=None,
        metavar='LEVEL',
        help='Level for the main logger, overriding a bare --log-level.',
    )

    if has_image_logger:
        group.add_argument(
            '--log-image-to-console',
            action=argparse.BooleanOptionalAction,
            default=None,
            help="""Write per-image logs to the terminal (default: no). Each
            image's log carries the detail of processing that one image, which
            is usually wanted in a file rather than on screen.""",
        )
        group.add_argument(
            '--log-image-to-file',
            action=argparse.BooleanOptionalAction,
            default=None,
            help='Write a log file per image under the log root (default: yes).',
        )
        group.add_argument(
            '--log-level-image',
            type=str,
            default=None,
            metavar='LEVEL',
            help='Level for the image loggers, overriding a bare --log-level.',
        )
