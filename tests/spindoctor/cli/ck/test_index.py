"""Hermetic tests for ``spindoctor.cli.ck.index``.

Every kernel here is written by the suite into directories named the way the
holdings name theirs, so the classification, the coverage read-back and the
candidate filter are exercised against real SPICE files without touching the
holdings.
"""

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cspyce
import numpy as np
import pytest
from filecache import FCPath
from tests.spindoctor.cli.ck.ck_helpers import (
    CASSINI_CK_FRAME_ID,
    ET0,
    VOYAGER_CK_FRAME_ID,
    KernelPool,
    baseline_attitude,
    baseline_segment,
    write_baseline_ck,
    write_ck,
    write_type1_ck,
)

from spindoctor.cli.ck import index as index_module
from spindoctor.cli.ck.index import (
    SNAPPED_LOOKUP_TOL_TICKS,
    CkFile,
    CkIndex,
    CoverageInterval,
    KernelClass,
    build_ck_index,
    kernel_class_for_basename,
)
from spindoctor.cli.ck.segment import CkSegment
from spindoctor.support.types import NDArrayFloatType

# The clocks the two test objects are tagged against, matching the test SCLK.
_CASSINI_SCLK_ID = -82
_VOYAGER_SCLK_ID = -31

# The baseline kernels the index reads cover four seconds from ET0, at
# half-second records.
_COVERAGE_S = 4.0
_RECORD_STEP_S = 0.5

# Real reconstructed and gapfill names from the Cassini holdings, so the
# index is exercised on the shapes it will actually meet.
_RECONSTRUCTED_NAME = '03236_04002ra.bc'
_GAPFILL_NAME = '03001_04001pa_gapfill_v01.bc'
_PREDICTED_NAME = '04009_04051px.bc'

# An object whose clock id SPICE computes as 0, which no SCLK kernel defines.
# A real merged New Horizons pointing file names this object beside the
# spacecraft, so its coverage cannot be expressed in TDB.
_CLOCKLESS_OBJECT_ID = -1

# A name the Galileo holdings really hold, which declares no class.
_UNCLASSIFIED_NAME = 'ckjabv3_plt.bc'


def _write_ck(
    directory: Path,
    name: str,
    *,
    ck_frame_id: int = CASSINI_CK_FRAME_ID,
    sclk_id: int = _CASSINI_SCLK_ID,
    start_et: float = ET0,
    attitude: Callable[[float], NDArrayFloatType] = baseline_attitude,
) -> Path:
    """Write one baseline C-kernel into a directory, creating the directory.

    Parameters:
        directory: Directory to write into.
        name: Basename of the kernel.
        ck_frame_id: SPICE id of the object the kernel describes.
        sclk_id: The spacecraft clock its time tags are encoded against.
        start_et: First record epoch, TDB seconds past J2000.
        attitude: The J2000-to-CK-object rotation at an epoch.

    Returns:
        The path written.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    records = int(_COVERAGE_S / _RECORD_STEP_S) + 1
    write_baseline_ck(
        path,
        ck_frame_id=ck_frame_id,
        sclk_id=sclk_id,
        epochs=[start_et + step * _RECORD_STEP_S for step in range(records)],
        attitude=attitude,
        angular_velocity=None,
    )
    return path


def test_build_ck_index_records_the_object_and_its_coverage(
    pool: KernelPool, tmp_path: Path
) -> None:
    """A scanned kernel reports which object it describes and over what epochs."""
    root = tmp_path / 'CK-reconstructed'
    _write_ck(root, _RECONSTRUCTED_NAME)
    index = build_ck_index([root])
    assert len(index.files) == 1
    assert index.files[0].coverage[0].ck_frame_id == CASSINI_CK_FRAME_ID
    assert index.files[0].coverage[0].start_et == pytest.approx(ET0, abs=1e-6)
    assert index.files[0].coverage[0].stop_et == pytest.approx(ET0 + _COVERAGE_S, abs=1e-6)


def test_build_ck_index_takes_the_class_from_the_basename(pool: KernelPool, tmp_path: Path) -> None:
    """Each file is classified by its own name, whatever directory it sits in.

    All three kernels share one directory whose name declares nothing, so only
    the basenames can be supplying the three different classes.
    """
    root = tmp_path / 'CK'
    for name in (_RECONSTRUCTED_NAME, _GAPFILL_NAME, _PREDICTED_NAME):
        _write_ck(root, name)
    index = build_ck_index([root])
    classes = {ck_file.basename: ck_file.kernel_class for ck_file in index.files}
    assert classes[_RECONSTRUCTED_NAME] is KernelClass.RECONSTRUCTED
    assert classes[_GAPFILL_NAME] is KernelClass.GAPFILL
    assert classes[_PREDICTED_NAME] is KernelClass.PREDICTED


def test_build_ck_index_ignores_a_directory_naming_another_class(
    pool: KernelPool, tmp_path: Path
) -> None:
    """A directory whose name contradicts the file in it does not decide the class."""
    root = tmp_path / 'CK-predicted'
    _write_ck(root, _RECONSTRUCTED_NAME)
    index = build_ck_index([root])
    assert index.files[0].kernel_class is KernelClass.RECONSTRUCTED


def test_build_ck_index_indexes_the_other_kernel_extension(
    pool: KernelPool, tmp_path: Path
) -> None:
    """A C-kernel stored under the .ck extension is indexed like a .bc one."""
    root = tmp_path / 'CK'
    _write_ck(root, 'V1SAT_VERSION2_TYPE3_UVS_SEDR.ck')
    index = build_ck_index([root])
    assert index.files[0].basename == 'V1SAT_VERSION2_TYPE3_UVS_SEDR.ck'


def test_build_ck_index_skips_files_that_are_not_kernels(pool: KernelPool, tmp_path: Path) -> None:
    """The label files sitting beside the binaries are not C-kernels."""
    root = tmp_path / 'CK-reconstructed'
    _write_ck(root, _RECONSTRUCTED_NAME)
    (root / '03236_04002ra.lbl').write_text('PDS_VERSION_ID = PDS3\n')
    index = build_ck_index([root])
    assert [ck_file.basename for ck_file in index.files] == [_RECONSTRUCTED_NAME]


def test_candidates_keep_a_file_covering_the_midtime(pool: KernelPool, tmp_path: Path) -> None:
    """A kernel named by the metadata that covers the epoch is a candidate."""
    root = tmp_path / 'CK-reconstructed'
    _write_ck(root, _RECONSTRUCTED_NAME)
    index = build_ck_index([root])
    candidates = index.candidates(
        basenames=[_RECONSTRUCTED_NAME], ck_frame_id=CASSINI_CK_FRAME_ID, et=ET0 + 2.0
    )
    assert [ck_file.basename for ck_file in candidates] == [_RECONSTRUCTED_NAME]


def test_candidates_drop_a_file_that_does_not_cover_the_midtime(
    pool: KernelPool, tmp_path: Path
) -> None:
    """Coverage is checked before any kernel is furnished."""
    root = tmp_path / 'CK-reconstructed'
    _write_ck(root, _RECONSTRUCTED_NAME)
    index = build_ck_index([root])
    candidates = index.candidates(
        basenames=[_RECONSTRUCTED_NAME],
        ck_frame_id=CASSINI_CK_FRAME_ID,
        et=ET0 + _COVERAGE_S + 100.0,
    )
    assert candidates == ()


def test_candidates_drop_a_file_describing_another_object(pool: KernelPool, tmp_path: Path) -> None:
    """A kernel for a different spacecraft cannot have supplied the baseline."""
    root = tmp_path / 'CK'
    _write_ck(root, 'vgr1_super.bc', ck_frame_id=VOYAGER_CK_FRAME_ID, sclk_id=_VOYAGER_SCLK_ID)
    index = build_ck_index([root])
    candidates = index.candidates(
        basenames=['vgr1_super.bc'], ck_frame_id=CASSINI_CK_FRAME_ID, et=ET0 + 2.0
    )
    assert candidates == ()


def test_candidates_ignore_a_basename_the_metadata_never_named(
    pool: KernelPool, tmp_path: Path
) -> None:
    """The metadata's kernel list is what limits the candidates."""
    root = tmp_path / 'CK-reconstructed'
    _write_ck(root, _RECONSTRUCTED_NAME)
    index = build_ck_index([root])
    candidates = index.candidates(
        basenames=['04002_04009ra.bc'], ck_frame_id=CASSINI_CK_FRAME_ID, et=ET0 + 2.0
    )
    assert candidates == ()


def test_candidates_offer_each_directory_holding_the_same_basename(
    pool: KernelPool, tmp_path: Path
) -> None:
    """One name in two directories is two candidates; only reproduction tells them apart."""
    first = tmp_path / 'CK-reconstructed'
    second = tmp_path / 'CK-gapfill'
    _write_ck(first, _RECONSTRUCTED_NAME)
    _write_ck(second, _RECONSTRUCTED_NAME)
    index = build_ck_index([first, second])
    candidates = index.candidates(
        basenames=[_RECONSTRUCTED_NAME], ck_frame_id=CASSINI_CK_FRAME_ID, et=ET0 + 2.0
    )
    assert [ck_file.path.as_posix() for ck_file in candidates] == [
        (first / _RECONSTRUCTED_NAME).as_posix(),
        (second / _RECONSTRUCTED_NAME).as_posix(),
    ]


def test_candidates_are_ordered_by_kernel_class(pool: KernelPool, tmp_path: Path) -> None:
    """Reconstructed pointing is offered before gapfill before predicted.

    One directory holds all three, so the order can only come from the names.
    Sorted by name alone the gapfill kernel would come last, not second.
    """
    root = tmp_path / 'CK'
    for name in (_PREDICTED_NAME, _GAPFILL_NAME, _RECONSTRUCTED_NAME):
        _write_ck(root, name)
    index = build_ck_index([root])
    candidates = index.candidates(
        basenames=[_PREDICTED_NAME, _GAPFILL_NAME, _RECONSTRUCTED_NAME],
        ck_frame_id=CASSINI_CK_FRAME_ID,
        et=ET0 + 2.0,
    )
    assert [ck_file.basename for ck_file in candidates] == [
        _RECONSTRUCTED_NAME,
        _GAPFILL_NAME,
        _PREDICTED_NAME,
    ]


def test_candidates_of_one_class_are_ordered_by_greatest_basename(
    pool: KernelPool, tmp_path: Path
) -> None:
    """Within a class the lexicographically greatest name comes first."""
    root = tmp_path / 'CK-reconstructed'
    _write_ck(root, '04002_04009ra.bc')
    _write_ck(root, '03236_04002ra.bc')
    index = build_ck_index([root])
    candidates = index.candidates(
        basenames=['03236_04002ra.bc', '04002_04009ra.bc'],
        ck_frame_id=CASSINI_CK_FRAME_ID,
        et=ET0 + 2.0,
    )
    assert [ck_file.basename for ck_file in candidates] == ['04002_04009ra.bc', '03236_04002ra.bc']


def test_frozen_object_coverage_admits_an_epoch_the_snapped_lookup_reaches(
    pool: KernelPool, tmp_path: Path
) -> None:
    """A scan platform's coverage is widened by the tolerance its lookup used.

    An image whose midtime falls outside a Voyager segment can still have
    navigated against it, because the pointing lookup that froze its frame
    answers with a record up to its tolerance away.
    """
    root = tmp_path / 'CK'
    _write_ck(
        root,
        'vg1_sat_version1_type1_iss_sedr.bc',
        ck_frame_id=VOYAGER_CK_FRAME_ID,
        sclk_id=_VOYAGER_SCLK_ID,
        start_et=ET0 + 100.0,
    )
    index = build_ck_index([root])
    assert index.files[0].covers(VOYAGER_CK_FRAME_ID, ET0) is True


def test_evaluated_object_coverage_stops_at_the_segment_window(
    pool: KernelPool, tmp_path: Path
) -> None:
    """An object read by evaluating a frame chain is covered only where it is covered."""
    root = tmp_path / 'CK-reconstructed'
    _write_ck(root, _RECONSTRUCTED_NAME, start_et=ET0 + 100.0)
    index = build_ck_index([root])
    assert index.files[0].covers(CASSINI_CK_FRAME_ID, ET0) is False


@pytest.mark.parametrize(
    'et', [float('nan'), float('inf'), float('-inf')], ids=['nan', 'inf', 'negative-inf']
)
def test_coverage_excludes_a_non_finite_epoch(et: float) -> None:
    """No interval contains an epoch that is not a number.

    A NaN answers every comparison with False, so it would fall outside by
    accident rather than by test; an infinity compares equal to itself and
    would sit inside any window reaching that far.

    Parameters:
        et: The non-finite epoch asked about.
    """
    interval = CoverageInterval(ck_frame_id=CASSINI_CK_FRAME_ID, start_et=-1e300, stop_et=1e300)
    assert interval.contains(CASSINI_CK_FRAME_ID, et) is False


@pytest.mark.parametrize('et', [1.0, 2.0], ids=['start', 'stop'])
def test_coverage_includes_its_endpoints(et: float) -> None:
    """An epoch exactly at either end of a window is covered.

    Parameters:
        et: The endpoint asked about.
    """
    interval = CoverageInterval(ck_frame_id=CASSINI_CK_FRAME_ID, start_et=1.0, stop_et=2.0)
    assert interval.contains(CASSINI_CK_FRAME_ID, et) is True


@pytest.mark.parametrize(
    'endpoints',
    [(np.nan, 1.0), (1.0, np.nan), (-np.inf, 1.0), (1.0, np.inf)],
    ids=['nan-start', 'nan-stop', 'infinite-start', 'infinite-stop'],
)
def test_coverage_refuses_a_non_finite_window(endpoints: tuple[float, float]) -> None:
    """A window that is not a pair of epochs is refused where it is built.

    Parameters:
        endpoints: The window's start and stop.
    """
    with pytest.raises(ValueError, match='is not finite'):
        CoverageInterval(
            ck_frame_id=CASSINI_CK_FRAME_ID, start_et=endpoints[0], stop_et=endpoints[1]
        )


def test_coverage_refuses_a_window_that_ends_before_it_starts() -> None:
    """A backwards window would cover nothing while looking like a window."""
    with pytest.raises(ValueError, match=r'ends at 1\.0 before it starts at 2\.0'):
        CoverageInterval(ck_frame_id=CASSINI_CK_FRAME_ID, start_et=2.0, stop_et=1.0)


@pytest.mark.parametrize(
    ('basename', 'expected'),
    [
        ('03236_04002ra.bc', KernelClass.RECONSTRUCTED),
        ('04059_04066rb.bc', KernelClass.RECONSTRUCTED),
        ('00001_00092rc.bc', KernelClass.RECONSTRUCTED),
        ('001001_001004ra.bc', KernelClass.RECONSTRUCTED),
        ('010109_010114rb.bc', KernelClass.RECONSTRUCTED),
        ('001105_001108.bc', KernelClass.RECONSTRUCTED),
        ('03001_04001pa_gapfill_v01.bc', KernelClass.GAPFILL),
        ('07001_08001pa_gapfill_v14.bc', KernelClass.GAPFILL),
        ('04009_04051px.bc', KernelClass.PREDICTED),
        ('04051_04092ph_psiv2.bc', KernelClass.PREDICTED),
        ('04135_04171pd_fsiv.bc', KernelClass.PREDICTED),
        ('04009_04051py_as_flown.bc', KernelClass.PREDICTED),
        ('05099_05134pg_fsiv_lmb.bc', KernelClass.PREDICTED),
        ('nh_scispi_2015_recon.bc', KernelClass.RECONSTRUCTED),
        ('nh_scispi_2015_pred.bc', KernelClass.PREDICTED),
        ('merged_nhpc_2007_01_v006.bc', KernelClass.UNCLASSIFIED),
        ('nhpc_haz_2015.bc', KernelClass.UNCLASSIFIED),
        ('vg2_nep_version1_type1_iss_sedr.bc', KernelClass.UNCLASSIFIED),
        ('vgr1_super.bc', KernelClass.UNCLASSIFIED),
        ('V1SAT_VERSION2_TYPE3_UVS_SEDR.ck', KernelClass.UNCLASSIFIED),
        ('ckjabv3_plt.bc', KernelClass.UNCLASSIFIED),
        ('gll_plt_pre_1990_v00.bc', KernelClass.UNCLASSIFIED),
    ],
    ids=[
        'cassini-tour-ra',
        'cassini-tour-rb',
        'cassini-cruise-rc',
        'cassini-jupiter-ra',
        'cassini-jupiter-rb',
        'cassini-jupiter-no-release-code',
        'cassini-gapfill-v01',
        'cassini-gapfill-v14',
        'cassini-predicted-bare',
        'cassini-predicted-psiv',
        'cassini-predicted-fsiv',
        'cassini-as-flown',
        'cassini-predicted-two-suffixes',
        'new-horizons-recon',
        'new-horizons-pred',
        'new-horizons-merged',
        'new-horizons-hazard',
        'voyager-sedr',
        'voyager-bus',
        'voyager-upper-case',
        'galileo-platform',
        'galileo-predicted-platform',
    ],
)
def test_kernel_class_for_basename(basename: str, expected: KernelClass) -> None:
    """Real holdings basenames classify as their own names declare.

    Parameters:
        basename: The kernel basename, taken from the real holdings.
        expected: The class that name declares.
    """
    assert kernel_class_for_basename(basename) is expected


def test_kernel_class_reads_a_cassini_name_whatever_its_case() -> None:
    """Case carries no meaning, and the holdings do not spell every name alike."""
    assert kernel_class_for_basename('03236_04002RA.BC') is KernelClass.RECONSTRUCTED


def _put_rules_in_force(
    monkeypatch: pytest.MonkeyPatch, rules: tuple[tuple[re.Pattern[str], KernelClass], ...]
) -> None:
    """Make one replacement mission table the only one in force.

    Parameters:
        monkeypatch: The fixture that restores the shipped table afterwards.
        rules: The patterns the single mission in force declares.
    """
    monkeypatch.setattr(
        index_module,
        '_MISSION_NAME_RULES',
        (index_module._MissionNameRules(mission='Cassini', rules=rules),),
    )


def test_kernel_class_puts_gapfill_ahead_of_predicted_without_relying_on_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A gapfill name is a ``pa`` name, and the predicted pattern excludes it itself.

    Reversing the order the patterns are tested in must not be able to turn
    this kernel into a predicted one.

    Parameters:
        monkeypatch: The fixture that puts the reversed table in force.
    """
    _put_rules_in_force(monkeypatch, tuple(reversed(index_module._CASSINI_NAME_RULES)))
    assert kernel_class_for_basename(_GAPFILL_NAME) is KernelClass.GAPFILL


def test_kernel_class_refuses_a_name_declaring_two_classes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A name matching two classes is ambiguous, not resolved by test order.

    The shipped patterns are mutually exclusive, so the guard is reached here
    by putting a deliberately overlapping table in force: it exists to catch a
    later edit that makes two patterns overlap, which would otherwise be
    settled silently by whichever was tested first.

    Parameters:
        monkeypatch: The fixture that puts the overlapping table in force.
    """
    _put_rules_in_force(
        monkeypatch,
        (
            (re.compile(r'\d{5}_\d{5}r[a-z]\.bc'), KernelClass.RECONSTRUCTED),
            (re.compile(r'\d{5}_\d{5}.*\.bc'), KernelClass.PREDICTED),
        ),
    )
    with pytest.raises(ValueError, match='declares more than one kernel class'):
        kernel_class_for_basename(_RECONSTRUCTED_NAME)


def test_kernel_class_accepts_two_patterns_of_one_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two patterns agreeing on the class is not ambiguity.

    Cassini's two date conventions both mean reconstructed, so matching more
    than one pattern is only a refusal when the classes differ.

    Parameters:
        monkeypatch: The fixture that puts the agreeing table in force.
    """
    _put_rules_in_force(
        monkeypatch,
        (
            (re.compile(r'\d{5}_\d{5}r[a-z]\.bc'), KernelClass.RECONSTRUCTED),
            (re.compile(r'\d{5}_\d{5}.*\.bc'), KernelClass.RECONSTRUCTED),
        ),
    )
    assert kernel_class_for_basename(_RECONSTRUCTED_NAME) is KernelClass.RECONSTRUCTED


@pytest.mark.parametrize(
    'name',
    [
        '03236_04002ra.bc',
        '00001_00092rc.bc',
        '001105_001108.bc',
        '03001_04001pa_gapfill_v01.bc',
        '04009_04051px.bc',
        '04009_04051py_as_flown.bc',
        'nh_scispi_2015_recon.bc',
        'nh_scispi_2015_pred.bc',
    ],
    ids=[
        'cassini-tour-ra',
        'cassini-cruise-rc',
        'cassini-jupiter-no-release-code',
        'cassini-gapfill',
        'cassini-predicted',
        'cassini-as-flown',
        'new-horizons-recon',
        'new-horizons-pred',
    ],
)
def test_no_shipped_pattern_pair_overlaps_on_a_real_basename(name: str) -> None:
    """The shipped patterns are mutually exclusive on every name they classify.

    Mutual exclusivity is what lets the two-class refusal stay unreached, so
    it is asserted here rather than assumed: every real name that any rule
    matches is matched by rules of exactly one class.

    Parameters:
        name: A real holdings basename that some shipped rule matches.
    """
    matched = {
        kernel_class
        for mission_rules in index_module._MISSION_NAME_RULES
        for pattern, kernel_class in mission_rules.rules
        if pattern.fullmatch(name.lower()) is not None
    }
    assert len(matched) == 1


@pytest.mark.parametrize(
    'basename',
    ['', '.', '..', '/', 'CK-reconstructed/03236_04002ra.bc', '/holdings/03236_04002ra.bc'],
    ids=['empty', 'dot', 'dot-dot', 'separator', 'relative-path', 'absolute-path'],
)
def test_kernel_class_refuses_something_that_is_not_a_basename(basename: str) -> None:
    """A path is not a file's own name, and would classify as declaring nothing.

    Parameters:
        basename: A value that names no single file.
    """
    with pytest.raises(ValueError, match='is not a C-kernel basename'):
        kernel_class_for_basename(basename)


@pytest.mark.parametrize(
    'basename',
    ['03236_04002ra_nav.bc', '04009_04051px_nav.bc', '03236_04002RA_NAV.BC', 'X_NAV.bc'],
    ids=[
        'reconstructed-corrected',
        'predicted-corrected',
        'upper-case-marker',
        'upper-case-marker-lower-extension',
    ],
)
def test_kernel_class_refuses_a_corrected_kernel(basename: str) -> None:
    """A corrected kernel is this program's output and never a baseline candidate.

    Left to the patterns the first two names would answer differently -- the
    predicted pattern ends in a wildcard and would accept the marker, the
    reconstructed one would not -- so classifying either is refused outright.
    The marker is read case-blind: an upper-cased copy of a corrected kernel
    is still a corrected kernel.

    Parameters:
        basename: A corrected kernel's name, in either case.
    """
    with pytest.raises(ValueError, match='names a corrected kernel'):
        kernel_class_for_basename(basename)


@pytest.mark.parametrize(
    'basename',
    ['03236_04002ra', '03236_04002ra.ck', '03236_04002ra.bc.lbl', ' 03236_04002ra.bc'],
    ids=['no-extension', 'other-extension', 'label', 'leading-space'],
)
def test_kernel_class_declines_to_guess_at_a_name_off_the_convention(basename: str) -> None:
    """A name that is not the convention declares nothing rather than nearly matching.

    The conventions include the extension, so a stem, a label and a name under
    another mission's extension all fall through -- as does a name that differs
    from a real one only by a character the filesystem keeps, since that is a
    different file.

    Parameters:
        basename: A name close to a real one but not the convention.
    """
    assert kernel_class_for_basename(basename) is KernelClass.UNCLASSIFIED


def test_build_ck_index_refuses_no_directories() -> None:
    """Scanning nothing would report every image as unreproducible."""
    with pytest.raises(ValueError, match='no kernel directory to scan'):
        build_ck_index([])


def test_build_ck_index_refuses_a_repeated_directory(tmp_path: Path) -> None:
    """One directory named twice would offer every file in it twice."""
    root = tmp_path / 'CK-reconstructed'
    root.mkdir()
    with pytest.raises(ValueError, match='named more than once'):
        build_ck_index([root, root])


def test_build_ck_index_refuses_a_missing_directory(tmp_path: Path) -> None:
    """A mistyped kernel root is refused rather than silently indexed as empty."""
    with pytest.raises(ValueError, match='does not exist or is not a directory'):
        build_ck_index([tmp_path / 'CK-reconstructed'])


def test_build_ck_index_refuses_a_file_named_as_a_directory(tmp_path: Path) -> None:
    """A path that is not a directory cannot be scanned."""
    path = tmp_path / 'CK-reconstructed'
    path.write_text('not a directory')
    with pytest.raises(ValueError, match='does not exist or is not a directory'):
        build_ck_index([path])


def test_build_ck_index_refuses_finding_no_kernels(tmp_path: Path) -> None:
    """An empty index is indistinguishable from every baseline having drifted."""
    root = tmp_path / 'CK-reconstructed'
    root.mkdir()
    with pytest.raises(ValueError, match='no C-kernel found under'):
        build_ck_index([root])


def test_ck_index_refuses_the_same_path_twice() -> None:
    """One file offered twice would be tried twice and could tie with itself."""
    ck_file = CkFile(
        path=FCPath('/holdings/CK-reconstructed') / _RECONSTRUCTED_NAME,
        kernel_class=KernelClass.RECONSTRUCTED,
        coverage=(),
    )
    with pytest.raises(ValueError, match='the same path more than once'):
        CkIndex(files=(ck_file, ck_file))


def test_build_ck_index_skips_a_corrected_kernel(pool: KernelPool, tmp_path: Path) -> None:
    """A correction written back beside its original is not a candidate for the next run.

    A corrected kernel reproduces its own baseline exactly wherever the
    correction was the identity, and its name sorts after the original's, so
    indexing one would let a correction win the tie-break and be corrected
    again.
    """
    root = tmp_path / 'CK-reconstructed'
    _write_ck(root, _RECONSTRUCTED_NAME)
    _write_ck(root, '03236_04002ra_nav.bc')
    index = build_ck_index([root])
    assert [ck_file.basename for ck_file in index.files] == [_RECONSTRUCTED_NAME]


def test_build_ck_index_skips_an_upper_cased_corrected_kernel(
    pool: KernelPool, tmp_path: Path
) -> None:
    """The corrected-kernel marker is read case-blind by the scan too.

    An upper-cased copy of a corrected kernel is still a corrected kernel, and
    one that slipped past the scan would then be refused by the case-blind
    classifier -- or worse, offered as a baseline candidate.
    """
    root = tmp_path / 'CK-reconstructed'
    _write_ck(root, _RECONSTRUCTED_NAME)
    _write_ck(root, '03236_04002RA_NAV.BC')
    index = build_ck_index([root])
    assert [ck_file.basename for ck_file in index.files] == [_RECONSTRUCTED_NAME]


def test_build_ck_index_refuses_a_file_that_is_not_a_kernel(
    pool: KernelPool, tmp_path: Path
) -> None:
    """A file wearing the extension and not the format stops the scan by name.

    A text file named ``.bc`` is a corrupted or mislabeled holding, and
    skipping it silently would drop a directory's kernels from the index with
    nothing to say so.
    """
    root = tmp_path / 'CK-reconstructed'
    _write_ck(root, _RECONSTRUCTED_NAME)
    (root / 'x.bc').write_text('this is not a kernel\n')
    with pytest.raises(OSError, match='must be a binary CK file'):
        build_ck_index([root])


def test_build_ck_index_refuses_a_directory_named_twice_by_different_paths(
    pool: KernelPool, tmp_path: Path
) -> None:
    """One directory spelled two ways is still one directory."""
    root = tmp_path / 'CK-reconstructed'
    _write_ck(root, _RECONSTRUCTED_NAME)
    with pytest.raises(ValueError, match='named more than once'):
        build_ck_index([root, tmp_path / 'CK-reconstructed' / '..' / 'CK-reconstructed'])


def test_build_ck_index_refuses_a_symlinked_duplicate(pool: KernelPool, tmp_path: Path) -> None:
    """A symbolic link to a scanned directory would index every file in it twice."""
    root = tmp_path / 'CK-reconstructed'
    _write_ck(root, _RECONSTRUCTED_NAME)
    link = tmp_path / 'CK-reconstructed-link'
    link.symlink_to(root, target_is_directory=True)
    with pytest.raises(ValueError, match='named more than once'):
        build_ck_index([root, link])


def test_build_ck_index_classifies_a_symlinked_directory_by_the_basenames(
    pool: KernelPool, tmp_path: Path
) -> None:
    """A link named for one class holds whatever its files say they are.

    The duplicate test resolves the roots it is given; the classification never
    looks at them at all, so neither spelling of the directory can change what
    a kernel is.
    """
    actual = tmp_path / 'ck_files'
    _write_ck(actual, _GAPFILL_NAME)
    link = tmp_path / 'CK-reconstructed'
    link.symlink_to(actual, target_is_directory=True)
    index = build_ck_index([link])
    assert index.files[0].kernel_class is KernelClass.GAPFILL


def test_a_kernel_naming_no_class_is_offered_last(pool: KernelPool, tmp_path: Path) -> None:
    """A kernel that says nothing about its own pointing is the last resort.

    The Galileo holdings name no class anywhere, and such a kernel must not
    outrank one that does -- not even a predicted one, whose name sorts before
    it and which would therefore win on the basename key alone.
    """
    root = tmp_path / 'CK'
    _write_ck(root, _PREDICTED_NAME)
    _write_ck(root, _UNCLASSIFIED_NAME)
    index = build_ck_index([root])
    candidates = index.candidates(
        basenames=[_PREDICTED_NAME, _UNCLASSIFIED_NAME],
        ck_frame_id=CASSINI_CK_FRAME_ID,
        et=ET0 + 2.0,
    )
    assert [ck_file.basename for ck_file in candidates] == [_PREDICTED_NAME, _UNCLASSIFIED_NAME]


def test_one_basename_in_two_directories_of_a_class_is_ordered_by_path(
    pool: KernelPool, tmp_path: Path
) -> None:
    """The path is the last resort of the tie-break, and it is never a tie.

    Two directories of one class can hold the same basename, which leaves the
    class and the name alike; the ordering then has to come from somewhere
    that cannot repeat, or it would follow whatever order the roots were
    given in.  They are given here in the order the path key reverses, so
    that following the roots and following the key are different answers.
    """
    first = tmp_path / 'a-CK-reconstructed'
    second = tmp_path / 'b-CK-reconstructed'
    _write_ck(first, _RECONSTRUCTED_NAME)
    _write_ck(second, _RECONSTRUCTED_NAME)
    index = build_ck_index([first, second])
    candidates = index.candidates(
        basenames=[_RECONSTRUCTED_NAME], ck_frame_id=CASSINI_CK_FRAME_ID, et=ET0 + 2.0
    )
    assert [ck_file.path.as_posix() for ck_file in candidates] == [
        (second / _RECONSTRUCTED_NAME).as_posix(),
        (first / _RECONSTRUCTED_NAME).as_posix(),
    ]


@pytest.mark.parametrize('as_type', [str, Path, FCPath], ids=['str', 'path', 'fcpath'])
def test_build_ck_index_normalizes_the_root_it_is_given(
    pool: KernelPool, tmp_path: Path, as_type: Callable[[str], Any]
) -> None:
    """A root may arrive as text, a local path, or a path that may be remote.

    Parameters:
        as_type: How the caller spells the directory.
    """
    root = tmp_path / 'CK-reconstructed'
    _write_ck(root, _RECONSTRUCTED_NAME)
    index = build_ck_index([as_type(str(root))])
    assert index.files[0].basename == _RECONSTRUCTED_NAME


def test_an_indexed_file_keeps_a_path_that_can_be_remote(pool: KernelPool, tmp_path: Path) -> None:
    """The index stores the kind of path that survives a remote kernel tree.

    Casting to a local path would discard where a remote kernel came from, and
    the scan would then be the only thing that could ever have read it.
    """
    root = tmp_path / 'CK-reconstructed'
    _write_ck(root, _RECONSTRUCTED_NAME)
    index = build_ck_index([root])
    assert isinstance(index.files[0].path, FCPath)


def test_the_coverage_filter_admits_every_epoch_the_snapped_lookup_reaches(
    pool: KernelPool, tmp_path: Path
) -> None:
    """The widening and the lookup are one tolerance, and this is why they must be.

    A discrete baseline is written with one record.  An exposure the widest
    snapped lookup can still serve -- exactly that tolerance away from the
    record -- has to survive the coverage filter, or the only candidate that
    reproduces it is dropped before anything is furnished and the image is
    reported as having no baseline at all.
    """
    root = tmp_path / 'CK'
    root.mkdir()
    path = root / 'vg1_sat_version1_type1_iss_sedr.bc'
    record_et = ET0 + 1.0
    tick = float(cspyce.sce2c(_VOYAGER_SCLK_ID, record_et))
    write_type1_ck(
        path,
        ck_frame_id=VOYAGER_CK_FRAME_ID,
        ticks=[tick],
        attitude=baseline_attitude,
        sclk_id=_VOYAGER_SCLK_ID,
    )
    index = build_ck_index([root])
    furthest_et = float(cspyce.sct2e(_VOYAGER_SCLK_ID, tick + SNAPPED_LOOKUP_TOL_TICKS))
    assert index.files[0].covers(VOYAGER_CK_FRAME_ID, furthest_et) is True


def _write_ck_naming_a_clockless_object(directory: Path, name: str) -> Path:
    """Write a kernel describing the test bus and an object with no clock.

    This is the shape a real merged New Horizons pointing file has: the
    spacecraft, whose clock every mission kernel set furnishes, and a second
    object whose clock id SPICE computes as 0, which no SCLK kernel defines.

    Parameters:
        directory: Directory to write into.
        name: Basename of the kernel.

    Returns:
        The path written.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    epochs = [ET0 + step * _RECORD_STEP_S for step in range(3)]
    ticks = [float(cspyce.sce2c(_CASSINI_SCLK_ID, et)) for et in epochs]
    quats = np.vstack(
        [np.asarray(cspyce.m2q(baseline_attitude(et)), dtype=np.float64) for et in epochs]
    )
    write_ck(
        path,
        [
            baseline_segment(
                ck_frame_id=CASSINI_CK_FRAME_ID,
                sclk_id=_CASSINI_SCLK_ID,
                epochs=epochs,
                attitude=baseline_attitude,
                angular_velocity=None,
            ),
            CkSegment(
                ck_frame_id=_CLOCKLESS_OBJECT_ID,
                segid='clockless',
                sclkdp=np.asarray(ticks, dtype=np.float64),
                quats=quats,
                avvs=None,
            ),
        ],
    )
    return path


def test_an_object_with_no_clock_is_recorded_rather_than_stopping_the_scan(
    pool: KernelPool, tmp_path: Path
) -> None:
    """A file naming an object no clock kernel describes still indexes.

    The New Horizons holdings hold such a file, so refusing the scan over it
    would make the whole mission unindexable for the sake of an object no
    image ever asks about.
    """
    root = tmp_path / 'CK-reconstructed'
    _write_ck_naming_a_clockless_object(root, _RECONSTRUCTED_NAME)
    index = build_ck_index([root])
    assert index.files[0].unreadable_objects == (_CLOCKLESS_OBJECT_ID,)


def test_the_readable_object_of_that_file_is_still_covered(
    pool: KernelPool, tmp_path: Path
) -> None:
    """The objects whose clock is furnished keep their coverage."""
    root = tmp_path / 'CK-reconstructed'
    _write_ck_naming_a_clockless_object(root, _RECONSTRUCTED_NAME)
    index = build_ck_index([root])
    covered = {interval.ck_frame_id for interval in index.files[0].coverage}
    assert covered == {CASSINI_CK_FRAME_ID}


def test_an_unreadable_object_is_offered_as_no_candidate(pool: KernelPool, tmp_path: Path) -> None:
    """A file offers no coverage for an object whose window it could not read."""
    root = tmp_path / 'CK-reconstructed'
    _write_ck_naming_a_clockless_object(root, _RECONSTRUCTED_NAME)
    index = build_ck_index([root])
    candidates = index.candidates(
        basenames=[_RECONSTRUCTED_NAME], ck_frame_id=_CLOCKLESS_OBJECT_ID, et=ET0
    )
    assert candidates == ()


def test_the_index_gathers_every_unreadable_object_it_met(pool: KernelPool, tmp_path: Path) -> None:
    """The index reports the unreadable objects of all its files together."""
    root = tmp_path / 'CK-reconstructed'
    _write_ck_naming_a_clockless_object(root, _RECONSTRUCTED_NAME)
    _write_ck(root, _GAPFILL_NAME)
    index = build_ck_index([root])
    assert index.unreadable_objects == frozenset({_CLOCKLESS_OBJECT_ID})


def test_an_index_of_readable_files_reports_no_unreadable_object(
    pool: KernelPool, tmp_path: Path
) -> None:
    """Nothing is reported unreadable when every object's clock is furnished."""
    root = tmp_path / 'CK-reconstructed'
    _write_ck(root, _RECONSTRUCTED_NAME)
    index = build_ck_index([root])
    assert index.unreadable_objects == frozenset()


def _write_ck_of_only_a_clockless_object(directory: Path, name: str) -> Path:
    """Write a kernel describing nothing but an object with no clock.

    Time tags for such an object are never decoded -- reading its coverage in
    TDB is exactly what fails -- so arbitrary increasing ticks are enough.

    Parameters:
        directory: Directory to write into.
        name: Basename of the kernel.

    Returns:
        The path written.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    ticks = np.asarray([0.0, 1.0e6, 2.0e6], dtype=np.float64)
    quats = np.tile(np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64), (3, 1))
    write_ck(
        path,
        [
            CkSegment(
                ck_frame_id=_CLOCKLESS_OBJECT_ID,
                segid='clockless only',
                sclkdp=ticks,
                quats=quats,
                avvs=None,
            )
        ],
    )
    return path


def test_a_file_of_only_unclockable_objects_does_not_stop_the_scan(
    pool: KernelPool, tmp_path: Path
) -> None:
    """A kernel whose every object lacks a clock indexes beside its neighbors."""
    root = tmp_path / 'CK-reconstructed'
    _write_ck(root, _RECONSTRUCTED_NAME)
    _write_ck_of_only_a_clockless_object(root, 'merged_clockless_v001.bc')
    index = build_ck_index([root])
    found = sorted(ck_file.basename for ck_file in index.files)
    assert found == [_RECONSTRUCTED_NAME, 'merged_clockless_v001.bc']


def test_a_file_of_only_unclockable_objects_contributes_no_coverage(
    pool: KernelPool, tmp_path: Path
) -> None:
    """It stays in the index but offers nothing, so it is never a baseline."""
    root = tmp_path / 'CK-reconstructed'
    _write_ck_of_only_a_clockless_object(root, 'merged_clockless_v001.bc')
    index = build_ck_index([root])
    assert index.files[0].coverage == ()
    assert index.files[0].unreadable_objects == (_CLOCKLESS_OBJECT_ID,)
