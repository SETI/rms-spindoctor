"""C-kernel generation from the corrected pointing navigation recorded.

Navigation measures where a camera was actually pointing and records that
measurement as a C-matrix in the image's metadata.  This package turns those
recorded matrices into SPICE C-kernels, so the same measurement loads into any
SPICE-aware tool with a ``furnsh`` instead of having to be applied as a pixel
offset by hand.

The package consumes the recorded C-matrix and performs no offset-to-rotation
conversion of its own.  It writes kernels through ``cspyce``.

The table of which spacecraft clock each CK object's time tags are encoded
against lives in ``spindoctor.spice_ids``, a constants module importing only the
standard library.  That table is the check against ``ckmeta`` computing a clock
id rather than validating one.

This package is part of the ``spindoctor.cli`` tree, which holds the command
line programs and the helpers only they use; the importable library API is the
rest of ``spindoctor``.  The ``__all__`` below is therefore the surface the
kernel-generating program imports from its own package -- what the modules
here offer each other and the driver, gathered in one place so that a reader
can see the shape of the writer without opening five files.  It is not
published library API: nothing outside ``spindoctor.cli`` imports it, it
carries no stability promise, and it has no page in the API reference, which
covers the library packages.  Every other ``spindoctor.cli`` subpackage stands
the same way.

Where a run gets the records it writes kernels from is not this package's.  It
reads them through :mod:`spindoctor.results_index.record_source`, the one seam
every program reads navigation records through, so that a results tree and an
ingested index are read here by the same code that reads them for the
reprojection, backplane and statistics programs.

One global to respect: ``cspyce.use_errors()`` / ``cspyce.use_flags()`` is
process-wide and shared with oops.  This package assumes the exceptions regime
(``use_errors``, the package default) and never flips it.
"""

from spindoctor.cli.ck.assignment import (
    Assignment,
    OutputGroup,
    assign_images,
    attitudes_reproduce,
    baseline_attitudes,
    group_for_output,
    output_basename,
    reproduces_baseline,
    rotation_angle_rad,
)
from spindoctor.cli.ck.clocks import clock_is_defined, select_sclk_kernel
from spindoctor.cli.ck.comments import (
    CommentArea,
    build_comment_lines,
    read_comment_area,
    reserved_comment_chars,
    write_comment_area,
)
from spindoctor.cli.ck.frames import (
    FK_SUFFIXES,
    camera_frame_is_defined,
    frames_defined_by,
    object_frame_is_defined,
    require_one_frame_kernel_per_frame,
)
from spindoctor.cli.ck.images import ImageEntry, OmissionReason, botsim_losers
from spindoctor.cli.ck.index import (
    CkFile,
    CkIndex,
    CoverageInterval,
    KernelClass,
    build_ck_index,
    kernel_class_for_basename,
)
from spindoctor.cli.ck.inputs import (
    clock_kernels,
    furnish_frame_kernels,
    furnish_supporting_kernels,
    kernel_paths,
    recorded_basenames,
    resolve_one,
    select_by_time,
)
from spindoctor.cli.ck.kernel_file import (
    check_ck_file,
    check_output_paths,
    first_data_record,
    write_ck_file,
)
from spindoctor.cli.ck.metakernel import build_meta_kernel_lines, write_meta_kernel
from spindoctor.cli.ck.pointing import ImagePointing
from spindoctor.cli.ck.pool import furnished
from spindoctor.cli.ck.report import (
    REPORT_COLUMNS,
    ImageReportFacts,
    ReportRow,
    read_image_report_facts,
    report_text,
    utc_for_et,
    write_report,
)
from spindoctor.cli.ck.segment import (
    BaselineCoverageGapError,
    CkSegment,
    build_segment,
    resolve_sclk_id,
    write_segment,
)

# The package's own surface, not published API: see the module docstring.
__all__ = [
    'FK_SUFFIXES',
    'REPORT_COLUMNS',
    'Assignment',
    'BaselineCoverageGapError',
    'CkFile',
    'CkIndex',
    'CkSegment',
    'CommentArea',
    'CoverageInterval',
    'ImageEntry',
    'ImagePointing',
    'ImageReportFacts',
    'KernelClass',
    'OmissionReason',
    'OutputGroup',
    'ReportRow',
    'assign_images',
    'attitudes_reproduce',
    'baseline_attitudes',
    'botsim_losers',
    'build_ck_index',
    'build_comment_lines',
    'build_meta_kernel_lines',
    'build_segment',
    'camera_frame_is_defined',
    'check_ck_file',
    'check_output_paths',
    'clock_is_defined',
    'clock_kernels',
    'first_data_record',
    'frames_defined_by',
    'furnish_frame_kernels',
    'furnish_supporting_kernels',
    'furnished',
    'group_for_output',
    'kernel_class_for_basename',
    'kernel_paths',
    'object_frame_is_defined',
    'output_basename',
    'read_comment_area',
    'read_image_report_facts',
    'recorded_basenames',
    'report_text',
    'reproduces_baseline',
    'require_one_frame_kernel_per_frame',
    'reserved_comment_chars',
    'resolve_one',
    'resolve_sclk_id',
    'rotation_angle_rad',
    'select_by_time',
    'select_sclk_kernel',
    'utc_for_et',
    'write_ck_file',
    'write_comment_area',
    'write_meta_kernel',
    'write_report',
    'write_segment',
]
