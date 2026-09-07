#!/usr/bin/env python3
################################################################################
# sd_create_ck.py
#
# Turn the corrected pointing a navigation run recorded into SPICE C-kernels.
# One mission per invocation: every navigation record under the navigation
# results root is read, each eligible image is paired with the
# original kernel it navigated against, and one corrected kernel is written per
# original, mirroring its name with "_nav" before the extension.  Beside them go
# a meta-kernel that furnishes the set in the order that makes a correction take
# precedence, and a CSV report saying what became of every image considered.
################################################################################

import argparse
import os
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import cspyce
import pdslogger
from filecache import FCPath, FileCache

# Make CLI runnable from source tree with
#    python src/package
package_source_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, package_source_path)

from spindoctor import __version__
from spindoctor.cli.ck.assignment import Assignment, assign_images, group_for_output
from spindoctor.cli.ck.comments import CommentArea, build_comment_lines
from spindoctor.cli.ck.images import ImageEntry, OmissionReason
from spindoctor.cli.ck.index import CkFile, build_ck_index
from spindoctor.cli.ck.inputs import (
    LSK_SUFFIXES,
    RECORD_COLUMNS,
    clock_kernels,
    furnish_frame_kernels,
    furnish_supporting_kernels,
    kernel_paths,
    read_whole_mission,
    recorded_basenames,
    resolve_one,
    select_by_time,
)
from spindoctor.cli.ck.kernel_file import check_ck_file, check_output_paths, write_ck_file
from spindoctor.cli.ck.metakernel import write_meta_kernel
from spindoctor.cli.ck.pointing import ImagePointing
from spindoctor.cli.ck.pool import furnished
from spindoctor.cli.ck.report import (
    ImageReportFacts,
    ReportRow,
    read_image_report_facts,
    write_report,
)
from spindoctor.cli.ck.segment import (
    BaselineCoverageGapError,
    CkSegment,
    build_segment,
    resolve_sclk_id,
)
from spindoctor.cli.logging_args import add_logging_arguments, reporting_configuration_errors
from spindoctor.config import (
    DEFAULT_CONFIG,
    IMAGE_LOGGER,
    MAIN_LOGGER,
    RunLogging,
    build_image_log_handlers,
    build_run_logging,
    get_nav_results_root,
    get_results_index_db_url,
    get_results_tree_tuning,
    load_default_and_user_config,
)
from spindoctor.config.program_names import SD_CREATE_CK
from spindoctor.nav_records import NavRecord, Selection, UnlistableDirectoryError
from spindoctor.results_index import open_record_source
from spindoctor.support.misc import log_run_environment

PROGRAM_NAME = SD_CREATE_CK
"""Program identity: names the main log directory and the
``logging.programs`` configuration block for this program."""

BACKEND = 'ck'
"""The per-image log subtree this program writes under."""

MISSIONS = ('coiss', 'gossi', 'nhlorri', 'vgiss')
"""The instrument identities whose images have a C-kernel object to correct.

Spelled here rather than read from the observation registry, which is the
authority on the names: the registry would drag every host class into a
program that needs only their names.  A mission whose images carry no
corrected attitude at all -- simulated images -- is deliberately not offered.
"""

REPORT_SUFFIX = '_ck_report.csv'
META_KERNEL_SUFFIX = '_nav.tm'


################################################################################
#
# ARGUMENT PARSING
#
################################################################################


def parse_args(command_list: list[str]) -> argparse.Namespace:
    """Build the parser and parse one command line.

    Parameters:
        command_list: The arguments, the mission first.

    Returns:
        The parsed arguments.
    """
    cmdparser = argparse.ArgumentParser(
        description='Write corrected-pointing C-kernels from navigated images.',
        epilog="""Each corrected kernel mirrors one original and carries a segment per
                navigated exposure; the originals stay required, since the corrections
                cover only the exposures that were navigated.""",
    )

    cmdparser.add_argument(
        'mission',
        type=str.lower,
        choices=MISSIONS,
        help='The mission whose navigated images to write kernels for',
    )

    environment_group = cmdparser.add_argument_group('Environment')
    environment_group.add_argument(
        '--config-file',
        action='append',
        default=None,
        help="""The configuration file(s) to use to override default settings;
        may be specified multiple times. If not provided, attempts to load
        ./nav_default_config.yaml if present.""",
    )
    environment_group.add_argument(
        '--nav-results-root',
        type=str,
        default=None,
        help="""The root directory of the navigation results to read metadata from;
        overrides the NAV_RESULTS_ROOT environment variable and the nav_results_root
        configuration variable""",
    )
    environment_group.add_argument(
        '--results-index-db',
        type=str,
        default=None,
        metavar='URL',
        help="""Connection URL of the results index written by sd_results_index (a
        sqlite: URL naming a local path, or a postgresql+psycopg: URL naming a
        server); overrides the environment.results_index_db configuration variable
        and NAV_RESULTS_INDEX_DB. The run then reads the whole mission's navigation
        records in bulk instead of one file per image. Pass --results-index-db none to
        read the files even where an index is configured. Without an index the
        navigation results tree is read directly, which is the default.""",
    )
    environment_group.add_argument(
        '--kernel-dir',
        action='append',
        required=True,
        metavar='DIR',
        help="""A directory of SPICE kernels; may be specified multiple times, and at
        least one is required. Every directory is scanned for C-kernels to pair images
        against, and all of them together resolve the kernel basenames each image's
        provenance records, so the leapseconds, frame and spacecraft clock kernels
        navigation used must be among them. Directories are not searched recursively.""",
    )

    selection_group = cmdparser.add_argument_group('Image selection')
    selection_group.add_argument(
        '--start-time',
        type=str,
        default=None,
        metavar='UTC',
        help="""Ignore images whose exposure midtime is before this UTC time. An image
        that recorded no midtime is ignored whenever either bound is given, since it
        cannot be placed in time.""",
    )
    selection_group.add_argument(
        '--stop-time',
        type=str,
        default=None,
        metavar='UTC',
        help='Ignore images whose exposure midtime is after this UTC time.',
    )

    output_group = cmdparser.add_argument_group('Output')
    output_group.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help="""Directory the corrected kernels, the meta-kernel and the report are
        written to. It is created if it does not exist.""",
    )

    add_logging_arguments(cmdparser)

    return cmdparser.parse_args(command_list)


################################################################################
#
# BUILDING, THEN WRITING
#
################################################################################


@dataclass(frozen=True)
class PreparedFile:
    """One corrected C-kernel, built in memory and not yet written.

    Parameters:
        baseline: The original C-kernel this file mirrors.
        output: Where the corrected file goes.
        segments: Its segments, in the order the images were assigned.
        comment_lines: The comment area to attach to it.
    """

    baseline: CkFile
    output: FCPath
    segments: tuple[CkSegment, ...]
    comment_lines: tuple[str, ...]


@dataclass(frozen=True)
class BuiltOutputs:
    """Everything a run will write, and the images building it left out.

    Parameters:
        files: The corrected kernels, ready to write, ordered by name.
        omissions: Why each image the build could not use receives no segment,
            keyed by image name.  An image is in one of the files or in here,
            never in both and never in neither.
    """

    files: tuple[PreparedFile, ...]
    omissions: Mapping[str, OmissionReason]


def image_report_facts(records: Sequence[NavRecord]) -> dict[str, ImageReportFacts]:
    """Read the facts the report and the comment areas both carry, by image name.

    Parameters:
        records: The records the run considered.

    Returns:
        One entry per image, keyed by the name recorded for it.

    Raises:
        ValueError: if a record holds a value the report cannot render, or if two
            records name the same image, whose facts could not then be told apart
            and one of which would silently replace the other in whichever
            kernel's comment area carried it.
    """
    facts_by_name: dict[str, ImageReportFacts] = {}
    for record in records:
        facts = read_image_report_facts(record.metadata)
        if facts.image_name in facts_by_name:
            raise ValueError(
                f'two records name the image {facts.image_name!r}; their reported facts cannot '
                f'be told apart'
            )
        facts_by_name[facts.image_name] = facts
    return facts_by_name


def report_rows(
    assignments: Sequence[Assignment], facts_by_name: Mapping[str, ImageReportFacts]
) -> list[ReportRow]:
    """Build the report, one row per image the run considered.

    Parameters:
        assignments: What became of each image the run considered, in report
            order.
        facts_by_name: The reported facts of each image, by name.

    Returns:
        The rows.

    Raises:
        KeyError: if an assignment names an image the facts do not hold, which
            means the two were read from different sets of records.
    """
    return [
        ReportRow(
            facts=facts_by_name[assignment.image_name],
            source_bc=assignment.output_name,
            omission_reason=assignment.omission_reason,
        )
        for assignment in assignments
    ]


def build_output_files(
    assignments: Sequence[Assignment],
    facts_by_name: Mapping[str, ImageReportFacts],
    output_dir: FCPath,
    *,
    sclk_basenames: Mapping[int, str],
    configuration_hash: str,
) -> BuiltOutputs:
    """Build every corrected kernel a run will write, and write none of them.

    Nothing here reaches the filesystem, so a failure part way through leaves
    the output directory as it found it.  That is the point of separating the
    two phases: the alternative is a run that dies with some kernels written,
    no meta-kernel, and no report -- the one artifact that says what is in the
    files that did get written.

    An image whose baseline supplies no pointing at one of its record epochs is
    omitted here rather than ending the run, and an output file every one of
    whose images was omitted is not built at all.

    Parameters:
        assignments: What became of each image the run considered.
        facts_by_name: The reported facts of each image, by name.
        output_dir: Where the corrected kernels will go.
        sclk_basenames: The clock kernel each spacecraft clock is encoded with.
        configuration_hash: Digest of the configuration this run used.

    Returns:
        The files to write, ordered by name, and the images left out.

    Raises:
        ValueError: if a segment cannot be built: a baseline supplying angular
            velocity at only some of an exposure's records, an exposure needing
            more records than a segment holds, or an image carrying no
            pointing.
        KeyError: if a frame the correction is expressed between is not defined
            by the furnished kernels.
        OSError: if a baseline kernel cannot be furnished or read.
    """
    # Every segment of the run is resident at once, which is what buys the
    # separation.  Measured on this writer's own segments, one costs 678 bytes
    # for the three records an ordinary exposure carries and 1826 bytes for the
    # 21 a twenty-second one does, so the 50,000-image batch this tool is sized
    # for holds about 34 MB of segments -- a few percent of the per-image
    # metadata such a batch already has resident, and flat in the number of
    # output files.  A batch of exposures long enough to hit the record cap
    # would be a different story: 50,000 segments of 10,000 records each is
    # 32 GB, and no instrument commands such an exposure.
    prepared: list[PreparedFile] = []
    omissions: dict[str, OmissionReason] = {}
    for group in group_for_output(assignments):
        segments: list[CkSegment] = []
        kept: list[Assignment] = []
        with furnished(group.baseline.path):
            for assignment in group.assignments:
                segment = _segment_for(assignment)
                if segment is None:
                    omissions[assignment.image_name] = OmissionReason.BASELINE_COVERAGE_GAP
                    continue
                segments.append(segment)
                kept.append(assignment)
        if len(kept) == 0:
            MAIN_LOGGER.warning(
                'No image is left to correct %s; no corrected kernel is written for it',
                group.baseline.basename,
            )
            continue
        area = CommentArea(
            generator_version=__version__,
            configuration_hash=configuration_hash,
            baseline_basenames=(group.baseline.basename,),
            sclk_basename=sclk_basenames[_clock_of(kept[0])],
            images=tuple(facts_by_name[assignment.image_name] for assignment in kept),
        )
        prepared.append(
            PreparedFile(
                baseline=group.baseline,
                output=output_dir / group.name,
                segments=tuple(segments),
                comment_lines=tuple(build_comment_lines(area)),
            )
        )
    return BuiltOutputs(files=tuple(prepared), omissions=omissions)


def apply_build_omissions(
    assignments: Sequence[Assignment], omissions: Mapping[str, OmissionReason]
) -> tuple[Assignment, ...]:
    """Report the images the build could not use as omitted.

    An image the assignment step paired with a baseline can still end up with
    no segment, because the pairing is made at the exposure midtime alone and
    the baseline has to answer at every record epoch.  The report is written
    from the assignments, so those images are given the reason here in place of
    the baseline, which is what puts each of them in the report exactly once
    with it.

    Parameters:
        assignments: The assignments as the assignment step made them.
        omissions: Why each omitted image receives no segment, by image name.

    Returns:
        The assignments, in the order given, with the omitted images revised.

    Raises:
        ValueError: if an omission names an image these assignments do not
            carry a baseline for -- an image of some other run, or one already
            omitted for another reason -- since the reason would then reach no
            row of the report and the image would be reported as corrected.
    """
    corrected = {
        assignment.image_name for assignment in assignments if assignment.baseline is not None
    }
    unplaced = sorted(set(omissions) - corrected)
    if len(unplaced) > 0:
        raise ValueError(
            f'the build omitted {unplaced}, which these assignments do not hold as corrected '
            f'images; those omission reasons would reach no row of the report'
        )
    return tuple(
        Assignment(
            entry=assignment.entry, baseline=None, omission_reason=omissions[assignment.image_name]
        )
        if assignment.image_name in omissions
        else assignment
        for assignment in assignments
    )


def write_output_files(files: Sequence[PreparedFile]) -> list[tuple[FCPath, FCPath]]:
    """Write the corrected kernels a run has already built, all of them or none.

    The whole set is judged before the first file is opened: every destination
    together, then each file's own contents.  A run whose third output path is
    already occupied therefore writes none of the three.  Judging as it goes
    instead would leave the first two behind -- a partial set, with no
    meta-kernel and no report to say what is in it, and one the refusal to
    overwrite an existing corrected kernel then blocks the rerun on.

    That is not atomicity, and it is not claimed.  What it establishes is that
    nothing knowable in advance stops the writing part way through.  A residual
    window remains and cannot be closed by any check made beforehand: the
    device filling up, a path or a permission changing between the check and
    the write, and a record set SPICE refuses once the file is open.  A file
    that fails that way is removed, but the files written before it are not,
    and this call raises with them on disk.

    Parameters:
        files: The built files, in the order they are to be written.

    Returns:
        One entry per file written, pairing the original with the correction.

    Raises:
        ValueError: if any output path cannot be written -- naming every one
            that cannot, and why -- if any file's records or comment area is
            one SPICE cannot store, or if SPICE refuses a segment once a file
            is open.
        RuntimeError: if SPICE cannot create a file.
        OSError: if the output directory cannot be created, or if SPICE
            refuses a write for an operating-system reason.
    """
    paths = [local_output_path(prepared.output) for prepared in files]
    check_output_paths(paths)
    for prepared, path in zip(files, paths, strict=True):
        check_ck_file(path, prepared.segments, prepared.comment_lines)
    written: list[tuple[FCPath, FCPath]] = []
    for prepared, path in zip(files, paths, strict=True):
        write_ck_file(path, prepared.segments, prepared.comment_lines)
        MAIN_LOGGER.info(
            'Wrote %s: %d segment(s) correcting %s',
            prepared.output.as_posix(),
            len(prepared.segments),
            prepared.baseline.basename,
        )
        written.append((prepared.baseline.path, prepared.output))
    return written


def absolute_directory(directory: str) -> str:
    """Return one directory named on the command line, resolved to absolute.

    A remote directory is passed through: it is already absolute, and there is
    no local working directory to resolve it against.

    Parameters:
        directory: The directory as it was typed.

    Returns:
        The same directory, absolute if it is local.
    """
    if '://' in directory:
        return directory
    return str(Path(directory).resolve())


def local_output_path(path: FCPath) -> Path:
    """Return the local path SPICE writes a kernel to, creating its directory.

    The directory is created here rather than at the write, so that a caller
    judging a whole set of destinations before it writes any of them has a
    directory to judge.

    Parameters:
        path: The output path.

    Returns:
        The same path as a local one.

    Raises:
        ValueError: if it is not local, since SPICE creates a file by name on
            the local filesystem and cannot write to a remote root.
        OSError: if the directory does not exist and cannot be created.
    """
    local = Path(path.as_posix())
    if '://' in path.as_posix():
        raise ValueError(
            f'{path.as_posix()!r} is not a local directory; SPICE creates a kernel by name on the '
            f'local filesystem, so a corrected kernel cannot be written to a remote root'
        )
    local.parent.mkdir(parents=True, exist_ok=True)
    return local


def _segment_for(assignment: Assignment) -> CkSegment | None:
    """Build the corrected segment of one assigned image.

    Parameters:
        assignment: The image and the baseline it corrects.

    Returns:
        The segment, or ``None`` when the baseline supplies no pointing at one
        of the record epochs, which omits this one image and lets the run
        continue.

    Raises:
        ValueError: if the image carries no pointing, which an assignment with
            a baseline cannot; if the baseline supplies angular velocity at
            only some of the record epochs, which no segment can express; or if
            the recorded exposure would need more records than a segment holds.
        KeyError: if a frame the correction is expressed between is not defined
            by the furnished kernels.
        OSError: if the baseline kernel cannot be read.
    """
    pointing = pointing_of(assignment)
    try:
        return build_segment(pointing)
    except BaselineCoverageGapError as exc:
        # Said here as well as by the reporting pass, which names the reason
        # but not the epoch: the reason says the baseline could not answer
        # somewhere inside the exposure, and this says where.  The run log
        # only, for the same reason as below -- no image scope is open here.
        MAIN_LOGGER.warning('%s: %s', assignment.image_name, exc)
        return None
    except (OSError, KeyError, ValueError) as exc:
        # The run log only, deliberately.  This stops the run, and the image
        # log it would otherwise be written to is opened by the reporting pass
        # that never gets to run; logging through the image logger with no
        # image scope open is a defect rather than a fallback, and under
        # strict_scope it would replace this error with a scope error and
        # demote the real one to a context.  It stops the run because the
        # reason set the report may use has no entry for an image whose
        # baseline reproduced its attitude and then supplied angular velocity
        # at only some of its records, and inventing one would be a schema
        # change for every consumer of the report.
        MAIN_LOGGER.exception(
            '%s: could not build the corrected segment: %s', assignment.image_name, exc
        )
        raise


def _clock_of(assignment: Assignment) -> int:
    """Return the spacecraft clock an assigned image's time tags are encoded with.

    Parameters:
        assignment: The image.

    Returns:
        The clock id.

    Raises:
        ValueError: if the image carries no pointing.
    """
    return resolve_sclk_id(pointing_of(assignment).ck_frame_id)


def pointing_of(assignment: Assignment) -> ImagePointing:
    """Return the recorded pointing of an image that is getting a segment.

    Parameters:
        assignment: The image and the baseline it corrects.

    Returns:
        Its recorded pointing.

    Raises:
        ValueError: if the image carries none.  An assignment that names a
            baseline cannot, so this is the guard on a caller that passed one
            that names none instead.
    """
    pointing = assignment.entry.pointing
    if pointing is None:
        raise ValueError(f'{assignment.image_name} has no pointing to write')
    return pointing


################################################################################
#
# REPORTING WHAT HAPPENED
#
################################################################################


def log_dispositions(
    records: Sequence[NavRecord], assignments: Sequence[Assignment], run_logging: RunLogging
) -> Counter[str]:
    """Report what became of each image, to that image's log and to the run's.

    An operator watching a batch must not have to open a per-image log to learn
    that corrections stopped being written, so every omission is one line in
    the run log as well as the detail in the image's own, and the counts are
    reported once at the end.

    Parameters:
        records: The records the run considered.
        assignments: What became of each of them.
        run_logging: The run's logging configuration.

    Returns:
        How many images each omission reason accounted for, with the images
        that received a segment counted under ``'corrected'``.

    Raises:
        ValueError: if the two sequences are of different lengths, which means
            they are not the same images.
    """
    totals: Counter[str] = Counter()
    for record, assignment in zip(records, assignments, strict=True):
        totals[_disposition_of(assignment)] += 1
        _log_one_disposition(record, assignment, run_logging)
    return totals


def _disposition_of(assignment: Assignment) -> str:
    """Return the one word summarizing what became of one image.

    Parameters:
        assignment: The image's assignment.

    Returns:
        The omission reason's value, or ``'corrected'``.
    """
    if assignment.omission_reason is None:
        return 'corrected'
    return assignment.omission_reason.value


def _log_one_disposition(
    record: NavRecord, assignment: Assignment, run_logging: RunLogging
) -> None:
    """Write one image's disposition to its own log, and any omission to the run's.

    Parameters:
        record: The image's navigation record.
        assignment: What became of it.
        run_logging: The run's logging configuration.
    """
    handlers, log_path = build_image_log_handlers(
        BACKEND,
        record.stub,
        run_logging.sinks,
        run_logging.levels,
        timestamp=run_logging.timestamp,
    )
    try:
        with IMAGE_LOGGER.open(
            record.path.as_posix(),
            handler=handlers,
            level=run_logging.levels.image_section_level(),
        ):
            if assignment.omission_reason is None:
                # A corrected assignment always names its baseline; the field
                # is optional only for the omitted case in the other branch.
                assert assignment.baseline is not None
                IMAGE_LOGGER.info(
                    'Corrected in %s, from the baseline %s',
                    assignment.output_name,
                    assignment.baseline.basename,
                )
            else:
                IMAGE_LOGGER.warning(
                    'No corrected segment written: %s', assignment.omission_reason.value
                )
    finally:
        for handler in handlers:
            if handler is not pdslogger.NULL_HANDLER:
                handler.close()
    if assignment.omission_reason is not None:
        MAIN_LOGGER.warning(
            '%s: no corrected segment written (%s)%s',
            assignment.image_name,
            assignment.omission_reason.value,
            '' if log_path is None else f'; see {log_path}',
        )


def log_totals(totals: Counter[str]) -> None:
    """Report how many images each disposition accounted for.

    Parameters:
        totals: The counts, keyed by disposition.
    """
    MAIN_LOGGER.info('Images corrected %d', totals.get('corrected', 0))
    for reason in OmissionReason:
        MAIN_LOGGER.info('Images omitted, %s: %d', reason.value, totals.get(reason.value, 0))


################################################################################
#
# MAIN
#
################################################################################


def main() -> None:
    """Write one mission's corrected C-kernels, meta-kernel and report."""
    command_list = sys.argv[1:]
    arguments = parse_args(command_list)

    with reporting_configuration_errors():
        load_default_and_user_config(arguments, DEFAULT_CONFIG)

    nav_results_root = FileCache(None).new_path(get_nav_results_root(arguments, DEFAULT_CONFIG))
    results_index_db_url = get_results_index_db_url(arguments, DEFAULT_CONFIG)
    tuning = get_results_tree_tuning(DEFAULT_CONFIG)
    # Resolved to absolute, both of them, because the meta-kernel names the
    # kernels it furnishes by these paths and SPICE resolves a relative name
    # against the *consumer's* working directory.  A meta-kernel written with
    # relative names works only from the directory that generated it, and
    # elsewhere fails on the first correction -- after the originals have
    # already loaded, so the pool is left uncorrected rather than empty.
    output_dir = FileCache(None).new_path(absolute_directory(arguments.output_dir))
    kernel_dirs = [absolute_directory(directory) for directory in arguments.kernel_dir]

    with reporting_configuration_errors():
        run_logging = build_run_logging(PROGRAM_NAME, arguments, DEFAULT_CONFIG)

    start_time = time.time()
    MAIN_LOGGER.info('*************************************')
    MAIN_LOGGER.info('*** BEGINNING C-KERNEL GENERATION ***')
    MAIN_LOGGER.info('*************************************')
    MAIN_LOGGER.info('')
    log_run_environment(MAIN_LOGGER, command_list)

    # Opened before anything else is read, and closed at the end of the run:
    # an index that cannot be opened, or that has no completed ingest of this
    # root, is a misconfigured run rather than a reason to walk the tree
    # instead.
    with open_record_source(
        [nav_results_root],
        results_index_db_url=results_index_db_url,
        columns=RECORD_COLUMNS,
        logger=MAIN_LOGGER,
        tuning=tuning,
    ) as source:
        try:
            stream = source.records(Selection(instrument=arguments.mission))
            records, unreadable = read_whole_mission(stream)
        except UnlistableDirectoryError as exc:
            # The one failure a read of the tree stops for rather than charging
            # to a file.  A directory nobody listed holds documents nobody read,
            # so a kernel set built around the gap would silently cover less
            # than the tree and go on being trusted; and a console entry point
            # owes its caller a message and a status for that rather than a
            # traceback.  Nothing has been written at this point, so there is
            # nothing to undo.
            MAIN_LOGGER.fatal(
                'C-kernel generation stopped: %s. No kernel, meta-kernel or report has '
                'been written. Make that directory readable and run again.',
                exc,
            )
            sys.exit(1)
        storage = source.describe()
    MAIN_LOGGER.info(
        'Read %d %s navigation record(s) from %s',
        len(records),
        arguments.mission,
        storage,
    )
    for entry in unreadable:
        MAIN_LOGGER.error('Could not read %s: %s', entry.path.as_posix(), entry.reason)
    if len(unreadable) > 0:
        MAIN_LOGGER.error('Could not read %d metadata file(s)', len(unreadable))

    paths = kernel_paths(kernel_dirs)
    # The leapseconds kernel is furnished before the time filter, because the
    # filter's own bounds are UTC and there is nothing to convert them with
    # until it is.  Its candidates are therefore the whole mission's, not the
    # selection's; that is safe where a clock kernel would not be, since
    # leap seconds are the same fact in every version of the kernel that
    # states them.
    lsk = furnish_supporting_kernels(recorded_basenames(records), paths, LSK_SUFFIXES)
    MAIN_LOGGER.info('Furnished leapseconds kernel(s): %s', ', '.join(lsk))

    records, undated = select_by_time(
        records,
        _selection_et('--start-time', arguments.start_time),
        _selection_et('--stop-time', arguments.stop_time),
    )
    if undated > 0:
        MAIN_LOGGER.warning(
            'Ignored %d image(s) that recorded no exposure midtime to place in time', undated
        )
    if len(records) == 0:
        MAIN_LOGGER.warning('No images selected; nothing to write')
        # The unreadable count still decides the exit status: a run whose
        # selection is empty only because its metadata could not be read must
        # not report itself clean.
        sys.exit(1 if len(unreadable) > 0 else 0)
    MAIN_LOGGER.info('Selected %d image(s)', len(records))

    # Read from the selected images only.  A run restricted to a time range
    # must not be refused for a disagreement between two kernel versions that
    # only images outside that range recorded.
    basenames = recorded_basenames(records)
    # The entries are read before any frame kernel is furnished, because they
    # need no SPICE and they name the frames the frame kernels are checked
    # against.
    entries = [_entry_for(record) for record in records]
    frames = furnish_frame_kernels(entries, basenames, paths)
    MAIN_LOGGER.info('Furnished frame kernel(s): %s', ', '.join(frames))

    sclk_basenames = clock_kernels(entries, basenames, paths)
    for sclk_id, basename in sorted(sclk_basenames.items()):
        MAIN_LOGGER.info('Spacecraft clock %d is encoded with %s', sclk_id, basename)
        cspyce.furnsh(str(cast(Path, resolve_one(basename, paths).retrieve())))

    index = build_ck_index(kernel_dirs)
    MAIN_LOGGER.info('Indexed %d candidate C-kernel(s)', len(index.files))
    skipped = sorted(index.unreadable_objects)
    if len(skipped) > 0:
        # A real kernel can describe an object no furnished clock kernel can
        # convert to TDB -- a merged New Horizons pointing file names object -1
        # beside the spacecraft -- and the scan indexes the rest of the file
        # rather than stopping.  The run says so once here; an image that
        # actually corrects such an object is refused by the assignment step.
        MAIN_LOGGER.warning(
            'Skipped the coverage of CK object(s) %s: no furnished kernel defines their '
            'spacecraft clock, so no indexed file offers them as a baseline',
            ', '.join(str(ck_frame_id) for ck_frame_id in skipped),
        )

    assignments = assign_images(entries, index)
    facts_by_name = image_report_facts(records)

    # Everything is built before anything is written.  A run that dies part way
    # through the build then leaves the output directory untouched, rather than
    # some corrected kernels, no meta-kernel, and no report saying what is in
    # them.
    built = build_output_files(
        assignments,
        facts_by_name,
        output_dir,
        sclk_basenames=sclk_basenames,
        configuration_hash=DEFAULT_CONFIG.resolved_config_hash(),
    )
    assignments = apply_build_omissions(assignments, built.omissions)
    rows = report_rows(assignments, facts_by_name)
    written = write_output_files(built.files)

    if len(written) > 0:
        meta_kernel = output_dir / f'{arguments.mission}{META_KERNEL_SUFFIX}'
        write_meta_kernel(
            meta_kernel,
            originals=[original for original, _correction in written],
            corrections=[correction for _original, correction in written],
        )
        MAIN_LOGGER.info('Wrote meta-kernel %s', meta_kernel.as_posix())
    else:
        MAIN_LOGGER.warning('No image received a corrected segment; no meta-kernel written')

    report = output_dir / f'{arguments.mission}{REPORT_SUFFIX}'
    write_report(report, rows)
    MAIN_LOGGER.info('Wrote report %s with %d row(s)', report.as_posix(), len(rows))

    log_totals(log_dispositions(records, assignments, run_logging))
    MAIN_LOGGER.info('Total elapsed time %.2f sec', time.time() - start_time)
    # Non-zero when anything the run was pointed at could not be read, so a
    # batch wrapper can tell a clean run from one that silently skipped its
    # input.  An image the run considered and omitted for a reason is not that:
    # it is in the report, which is the answer, and it exits zero.
    sys.exit(1 if len(unreadable) > 0 else 0)


def _selection_et(flag: str, value: str | None) -> float | None:
    """Convert one selection bound to ET, naming its flag when refused.

    Parameters:
        flag: The command-line flag the value arrived on.
        value: The UTC time string, or ``None`` when the flag was not given.

    Returns:
        TDB seconds past J2000, or ``None`` for an absent bound.

    Raises:
        ValueError: if SPICE cannot read the value as a UTC time.  Named for
            the flag, so a mistyped bound reads as the argument refusal it is
            rather than as a defect inside the run.
    """
    if value is None:
        return None
    try:
        return float(cspyce.utc2et(value))
    except ValueError as exc:
        raise ValueError(f'{flag} {value!r} is not a UTC time SPICE can read: {exc}') from exc


def _entry_for(record: NavRecord) -> ImageEntry:
    """Read one record's generator entry, naming the image if it cannot be read.

    Parameters:
        record: The record.

    Returns:
        The entry.

    Raises:
        TypeError: if the record holds a value of the wrong kind.
        ValueError: if a field the generator needs is absent or unusable.
    """
    try:
        return ImageEntry.from_metadata(record.metadata)
    except (TypeError, ValueError) as exc:
        # The run log only, for the same reason the segment failure reports
        # there: this stops the run, and no image scope is open to write to.
        MAIN_LOGGER.exception(
            '%s: cannot be read as a navigated image: %s', record.path.as_posix(), exc
        )
        raise


if __name__ == '__main__':
    main()
