import math
import socket
import subprocess
from typing import Any, cast

import numpy as np
import oops
import pdslogger

from spindoctor.support.command_line import masked_command_line
from spindoctor.support.types import NDArrayFloatType


def ra_rad_to_hms(ra: float) -> str:
    """Converts right ascension in radians to a formatted string in hours, minutes, and seconds.

    Parameters:
        ra: Right ascension value in radians.

    Returns:
        Formatted string in the form "HHhMMmSS.SSSs".

    Raises:
        ValueError: If the right ascension value is negative.
    """

    if ra < 0:
        raise ValueError(f'ra cannot be negative, got {ra}')

    ra = ra % math.tau
    ra_deg = ra * oops.DPR / 15  # In hours
    hh = int(ra_deg)
    mm = int((ra_deg - hh) * 60)
    ss = int((ra_deg - hh - mm / 60.0) * 3600 * 1000 + 0.5) / 1000
    if ss >= 60:
        mm += 1
        ss -= 60
    if mm >= 60:
        hh += 1
        mm -= 60
    if hh >= 24:
        hh -= 24

    return f'{hh:02d}h{mm:02d}m{ss:06.3f}s'


def dec_rad_to_dms(dec: float) -> str:
    """Converts declination in radians to a formatted string in degrees, minutes, and seconds.

    Parameters:
        dec: Declination value in radians.

    Returns:
        Formatted string in the form "+/-DDDdMMmSS.SSSs".
    """

    dec_deg = dec * oops.DPR  # In degrees
    is_neg = False
    if dec_deg < 0:
        is_neg = True
        dec_deg = -dec_deg
    dd = int(dec_deg)
    mm = int((dec_deg - dd) * 60)
    ss = int((dec_deg - dd - mm / 60.0) * 3600 * 1000 + 0.5) / 1000
    if ss >= 60:
        mm += 1
        ss -= 60
    if mm >= 60:
        dd += 1
        mm -= 60
    # TODO Check this - does this make sense for both dec and rad?
    if dd >= 180:
        dd -= 360
    elif dd <= -180:
        dd += 360
    if dd < 0:
        is_neg = not is_neg
        dd = -dd
    neg = '-' if is_neg else '+'

    return f'{neg}{dd:03d}d{mm:02d}m{ss:06.3f}s'


def flatten_list(lst: list[Any]) -> list[Any]:
    """Flattens a list of lists into a single list.

    Parameters:
        lst: The list to flatten.

    Returns:
        A flattened list.
    """

    return [x for sublist in lst for x in sublist]


def safe_lstrip_zero(s: str) -> str:
    """Strips leading zeros from a string but leaves one zero behind if that's all there is.

    Parameters:
        s: The string to strip leading zeros from.

    Returns:
        The string with leading zeros stripped.
    """

    if not s:
        return s

    ret = s.lstrip('0')
    if ret == '':
        ret = '0'
    return ret


def mad_std(a: NDArrayFloatType | list[float]) -> float:
    """Median absolute deviation (MAD) standard deviation.

    Parameters:
        a: Sample values; must contain at least one finite element.

    Returns:
        The robust standard-deviation estimate ``1.4826 * MAD``.

    Raises:
        ValueError: If ``a`` is empty or contains no finite values, which would
            otherwise return a silent ``NaN`` that poisons every downstream
            covariance / confidence computation.
    """
    a_array = cast(NDArrayFloatType, np.asarray(a, dtype=np.float64))
    finite = a_array[np.isfinite(a_array)]
    if finite.size == 0:
        raise ValueError('mad_std requires at least one finite value; got none')
    m = np.median(finite)
    return 1.4826 * float(np.median(np.abs(finite - m)))


_GIT_VERSION_CACHE: str | None = None


def current_git_version() -> str:
    """Return the git version of the current repo, caching the result.

    The result is cached because the git version cannot change during a single
    program run.

    Returns:
        The git describe string, or 'GIT DESCRIBE FAILED' if the command fails.
    """
    global _GIT_VERSION_CACHE
    if _GIT_VERSION_CACHE is not None:
        return _GIT_VERSION_CACHE
    # The ways git can decline are enumerated rather than caught wholesale: not
    # installed, not a repository, no tag to describe from. Each records itself
    # in the returned string, so the failure is visible in the run log instead
    # of being absorbed. Anything else -- a decode fault, a defect here -- is a
    # bug and propagates.
    try:
        ret = subprocess.check_output(
            ['git', 'describe', '--all', '--long', '--dirty', '--abbrev=40', '--tags']
        ).strip()
        _GIT_VERSION_CACHE = ret.decode('ascii')
    except (OSError, subprocess.SubprocessError):
        _GIT_VERSION_CACHE = 'GIT DESCRIBE FAILED'
    return _GIT_VERSION_CACHE


_LOCAL_HOST_NAME_CACHE: str | None = None


def get_local_host_name() -> str:
    """Return this machine's fully qualified domain name as a string.

    The value is obtained from ``socket.getfqdn()`` on the first call and stored
    in the module-level ``_LOCAL_HOST_NAME_CACHE`` so later calls return the same
    string without calling ``getfqdn`` again.

    Returns:
        The FQDN string on success, or the literal ``'LOCAL HOST NAME FAILED'`` if
        ``socket.getfqdn()`` raises any exception.

    Side effects:
        On the first successful call, sets ``_LOCAL_HOST_NAME_CACHE`` to the FQDN.
        On the first failing call, sets ``_LOCAL_HOST_NAME_CACHE`` to
        ``'LOCAL HOST NAME FAILED'``. Subsequent calls return the cached value and do
        not call ``socket.getfqdn()`` again.
    """
    global _LOCAL_HOST_NAME_CACHE
    if _LOCAL_HOST_NAME_CACHE is not None:
        return _LOCAL_HOST_NAME_CACHE
    # As above: a name service that will not answer is the expected way this
    # fails, and it records itself in the value. Anything else propagates.
    try:
        ret = socket.getfqdn()
        _LOCAL_HOST_NAME_CACHE = ret
    except OSError:
        _LOCAL_HOST_NAME_CACHE = 'LOCAL HOST NAME FAILED'
    return _LOCAL_HOST_NAME_CACHE


def log_run_environment(logger: pdslogger.PdsLogger, command_list: list[str]) -> None:
    """Log host, git, and command-line context to the given logger.

    Call once at process startup on the main logger, after the run's logging
    has been configured. Per-image loggers may omit this to avoid duplicating
    the same block on the console when handlers mirror output to
    ``MAIN_LOGGER``.

    The command line is recorded through
    :func:`~spindoctor.support.command_line.masked_command_line`, because a run
    log reaches a console, a log file and whoever is sent one, and one word of
    a command line can be a database password.

    Parameters:
        logger: The logger to write to.
        command_list: The command-line arguments for the current run
            (typically ``sys.argv[1:]``).
    """
    with logger.open('RUN-TIME ENVIRONMENT'):
        logger.info('*' * 40)
        logger.info('Host Local Name: %s', get_local_host_name())
        # if spindoctor.aws.AWS_ON_EC2_INSTANCE:
        #     logger.info('Host Public Name: %s (%s) in %s', spindoctor.aws.AWS_HOST_PUBLIC_NAME,
        #                  spindoctor.aws.AWS_HOST_PUBLIC_IPV4, spindoctor.aws.AWS_HOST_ZONE)
        #     logger.info('Host AMI ID: %s', spindoctor.aws.AWS_HOST_AMI_ID)
        #     logger.info('Host Instance Type: %s', spindoctor.aws.AWS_HOST_INSTANCE_TYPE)
        #     logger.info('Host Instance ID: %s', spindoctor.aws.AWS_HOST_INSTANCE_ID)
        logger.info('GIT Status:      %s', current_git_version())
        logger.info('Command line:    %s', ' '.join(masked_command_line(command_list)))
        logger.info('*' * 40)
