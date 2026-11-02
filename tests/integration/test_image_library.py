"""Structural invariants for the image-library directory tree (Phase 4 §A).

These tests run in the fast suite without holdings access; they enforce the
shape of ``tests/integration/image_library/images/<class>/<image_id>.yaml``:

- Every subdirectory of ``images/`` is a member of the declared scene-class
  list (typo guard).
- Every ``.yaml`` sidecar parses + validates against the
  :class:`~tests.integration.sidecar.Sidecar` schema.
- The primary scene-tag in each sidecar matches the containing directory.
- ``image_id`` is unique across the whole library.

The library-wide coverage checks (at least two sidecars per class, every
technique the primary technique of some sidecar) are in
``test_image_library_coverage``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.integration.sidecar import (
    DECLARED_SCENE_CLASSES,
    LibraryRoot,
    Sidecar,
    SidecarValidationError,
    load_sidecar,
)


@pytest.fixture(scope='module')
def library() -> LibraryRoot:
    """Return the library root tied to this test file's location."""
    return LibraryRoot()


def test_images_root_exists(library: LibraryRoot) -> None:
    """The library's ``images/`` directory must exist (CI guards drift)."""
    assert library.images.is_dir(), f'expected library images dir at {library.images}'


def test_class_subdirectories_are_subset_of_declared(library: LibraryRoot) -> None:
    """Every subdirectory under ``images/`` must be a known scene class.

    This catches typos like ``body_overflow`` vs ``body_partial_overflow``
    immediately, before any per-sidecar test runs with a confusing error.
    The structural test allows a *subset* of declared classes to be present,
    so an empty class doesn't fail it; ``test_image_library_coverage`` checks
    that every class holds enough sidecars.
    """
    actual = {p.name for p in library.discover_class_directories()}
    unknown = actual - DECLARED_SCENE_CLASSES
    assert not unknown, (
        f'unknown scene-class directories under {library.images}: {sorted(unknown)}; '
        f'allowed names are {sorted(DECLARED_SCENE_CLASSES)}'
    )


def test_every_sidecar_validates(library: LibraryRoot) -> None:
    """Every ``.yaml`` sidecar must parse + validate against the schema."""
    for sidecar_path in library.discover_sidecar_paths():
        try:
            load_sidecar(sidecar_path)
        except SidecarValidationError as exc:
            pytest.fail(str(exc))


def test_primary_scene_tag_matches_directory(library: LibraryRoot) -> None:
    """``scene_tags[0]`` must equal the parent directory's basename."""
    for sidecar_path in library.discover_sidecar_paths():
        sidecar = load_sidecar(sidecar_path)
        directory = sidecar_path.parent.name
        assert sidecar.primary_scene_tag == directory, (
            f'{sidecar_path}: primary scene_tag {sidecar.primary_scene_tag!r} '
            f'does not match containing directory {directory!r}'
        )


def test_image_ids_are_unique(library: LibraryRoot) -> None:
    """No two sidecars may share an ``image_id``."""
    by_id: dict[str, Path] = {}
    for sidecar_path in library.discover_sidecar_paths():
        sidecar = load_sidecar(sidecar_path)
        prior = by_id.get(sidecar.image_id)
        if prior is not None:
            pytest.fail(
                f'duplicate image_id {sidecar.image_id!r} in {sidecar_path} (also at {prior})'
            )
        by_id[sidecar.image_id] = sidecar_path


def test_sidecar_filename_matches_image_id(library: LibraryRoot) -> None:
    """``<image_id>.yaml`` filename must match the ``image_id`` field."""
    for sidecar_path in library.discover_sidecar_paths():
        sidecar = load_sidecar(sidecar_path)
        expected = f'{sidecar.image_id}.yaml'
        assert sidecar_path.name == expected, (
            f'{sidecar_path}: filename does not match image_id={sidecar.image_id!r}; '
            f'expected {expected!r}'
        )


# ---------------------------------------------------------------------------
# Sidecar schema unit tests (do not require the on-disk library)
# ---------------------------------------------------------------------------

_VALID_SIDECAR_TEXT = """\
schema_version: 1
image_id: TEST_IMG_0001
mission: COISS
camera: NAC
filter_combo: 'CL+CL'
image_url: 'pds3://volumes/COISS_2xxx/COISS_2021/data/.../TEST.IMG'
scene_tags: [body_mostly_offscreen, mimas]
ground_truth:
  offset_dv_px: 12.5
  offset_du_px: -3.25
  offset_uncertainty_px: 1.5
  source: operator_verified
  operator: rfrench
  verified_date: 2026-04-28
  ui_version: 'spindoctor 0.0.0'
  notes: 'Hand-picked for the schema test.'
expected:
  status: success
  confidence_tier: high
  primary_technique: BodyLimbNav
  techniques_must_run: [BodyLimbNav]
  techniques_must_skip: [StarFieldFromCatalogNav]
"""


def test_load_sidecar_accepts_valid_yaml(tmp_path: Path) -> None:
    """The reference YAML round-trips through ``load_sidecar`` cleanly."""
    p = tmp_path / 'TEST_IMG_0001.yaml'
    p.write_text(_VALID_SIDECAR_TEXT)
    sidecar = load_sidecar(p)
    assert isinstance(sidecar, Sidecar)
    assert sidecar.image_id == 'TEST_IMG_0001'
    assert sidecar.mission == 'COISS'
    assert sidecar.scene_tags == ('body_mostly_offscreen', 'mimas')
    assert sidecar.ground_truth.offset_dv_px == 12.5
    assert sidecar.expected.primary_technique == 'BodyLimbNav'


def test_load_sidecar_rejects_unknown_mission(tmp_path: Path) -> None:
    """``mission`` must be one of the declared mission codes."""
    p = tmp_path / 'BAD.yaml'
    p.write_text(_VALID_SIDECAR_TEXT.replace('mission: COISS', 'mission: APOLLO'))
    with pytest.raises(SidecarValidationError, match=r'mission'):
        load_sidecar(p)


def test_load_sidecar_rejects_unknown_primary_scene_tag(tmp_path: Path) -> None:
    """The primary ``scene_tag`` must be a declared scene class."""
    bad = _VALID_SIDECAR_TEXT.replace(
        'scene_tags: [body_mostly_offscreen, mimas]',
        'scene_tags: [body_overflow, mimas]',
    )
    p = tmp_path / 'BAD.yaml'
    p.write_text(bad)
    with pytest.raises(SidecarValidationError, match=r'primary scene_tag'):
        load_sidecar(p)


def test_load_sidecar_rejects_zero_uncertainty(tmp_path: Path) -> None:
    """Tolerance must be strictly positive."""
    bad = _VALID_SIDECAR_TEXT.replace('offset_uncertainty_px: 1.5', 'offset_uncertainty_px: 0.0')
    p = tmp_path / 'BAD.yaml'
    p.write_text(bad)
    with pytest.raises(SidecarValidationError, match=r'offset_uncertainty_px'):
        load_sidecar(p)


def test_load_sidecar_rejects_inconsistent_failed_status(tmp_path: Path) -> None:
    """``status=failed`` requires ``confidence_tier=failed`` and vice versa."""
    bad = _VALID_SIDECAR_TEXT.replace('status: success', 'status: failed')
    p = tmp_path / 'BAD.yaml'
    p.write_text(bad)
    with pytest.raises(SidecarValidationError, match=r'expected\.status'):
        load_sidecar(p)


def test_load_sidecar_rejects_inconsistent_conflicted_status(tmp_path: Path) -> None:
    """``status=conflicted`` requires ``confidence_tier=conflicted`` and vice versa."""
    bad = _VALID_SIDECAR_TEXT.replace('status: success', 'status: conflicted')
    p = tmp_path / 'BAD.yaml'
    p.write_text(bad)
    with pytest.raises(SidecarValidationError, match=r'expected\.status=conflicted requires'):
        load_sidecar(p)


def test_load_sidecar_accepts_consistent_conflicted_pair(tmp_path: Path) -> None:
    """A sidecar with both ``status=conflicted`` and ``confidence_tier=conflicted`` validates."""
    good = _VALID_SIDECAR_TEXT.replace('status: success', 'status: conflicted').replace(
        'confidence_tier: high', 'confidence_tier: conflicted'
    )
    p = tmp_path / 'GOOD.yaml'
    p.write_text(good)
    sidecar = load_sidecar(p)
    assert sidecar.expected.status == 'conflicted'
    assert sidecar.expected.confidence_tier == 'conflicted'


def test_load_sidecar_rejects_must_run_skip_overlap(tmp_path: Path) -> None:
    """A technique cannot be in both ``techniques_must_run`` and ``..._skip``."""
    bad = _VALID_SIDECAR_TEXT.replace(
        'techniques_must_skip: [StarFieldFromCatalogNav]',
        'techniques_must_skip: [BodyLimbNav]',
    )
    p = tmp_path / 'BAD.yaml'
    p.write_text(bad)
    with pytest.raises(SidecarValidationError, match=r'overlap'):
        load_sidecar(p)


def test_load_sidecar_rejects_unknown_schema_version(tmp_path: Path) -> None:
    """Sidecars from a future schema must fail loudly."""
    bad = _VALID_SIDECAR_TEXT.replace('schema_version: 1', 'schema_version: 99')
    p = tmp_path / 'BAD.yaml'
    p.write_text(bad)
    with pytest.raises(SidecarValidationError, match=r'schema_version'):
        load_sidecar(p)


# 12.5 * cos(60 deg) + (-3.25) * sin(60 deg) = 3.435
_RANK1_CONSTRAINT_BLOCK = """\
  constraint:
    type: rank_1
    normal_angle_deg: 60.0
    offset_along_normal_px: 3.435
    uncertainty_px: 0.4
"""


def _with_constraint(text: str, block: str = _RANK1_CONSTRAINT_BLOCK) -> str:
    """Insert a ``ground_truth.constraint`` block into the reference YAML."""
    return text.replace(
        "  notes: 'Hand-picked for the schema test.'\n",
        "  notes: 'Hand-picked for the schema test.'\n" + block,
    )


def test_load_sidecar_accepts_rank_1_constraint(tmp_path: Path) -> None:
    """A consistent ``ground_truth.constraint`` block round-trips."""
    p = tmp_path / 'TEST_IMG_0001.yaml'
    p.write_text(_with_constraint(_VALID_SIDECAR_TEXT))
    sidecar = load_sidecar(p)
    constraint = sidecar.ground_truth.constraint
    assert constraint is not None
    assert constraint.type == 'rank_1'
    assert constraint.normal_angle_deg == pytest.approx(60.0)
    assert constraint.offset_along_normal_px == pytest.approx(3.435)
    assert constraint.uncertainty_px == pytest.approx(0.4)


def test_load_sidecar_constraint_absent_is_none(tmp_path: Path) -> None:
    """A sidecar without a constraint block parses with ``constraint=None``."""
    p = tmp_path / 'TEST_IMG_0001.yaml'
    p.write_text(_VALID_SIDECAR_TEXT)
    assert load_sidecar(p).ground_truth.constraint is None


def test_load_sidecar_rejects_constraint_off_line_representative(tmp_path: Path) -> None:
    """The 2-D offset must project onto the declared normal component."""
    bad = _with_constraint(
        _VALID_SIDECAR_TEXT,
        _RANK1_CONSTRAINT_BLOCK.replace(
            'offset_along_normal_px: 3.435', 'offset_along_normal_px: 5.0'
        ),
    )
    p = tmp_path / 'BAD.yaml'
    p.write_text(bad)
    with pytest.raises(SidecarValidationError, match=r'constraint line'):
        load_sidecar(p)


def test_load_sidecar_rejects_unknown_constraint_type(tmp_path: Path) -> None:
    """``constraint.type`` must be a declared constraint type."""
    bad = _with_constraint(
        _VALID_SIDECAR_TEXT, _RANK1_CONSTRAINT_BLOCK.replace('type: rank_1', 'type: rank_2')
    )
    p = tmp_path / 'BAD.yaml'
    p.write_text(bad)
    with pytest.raises(SidecarValidationError, match=r'type'):
        load_sidecar(p)


def test_load_sidecar_rejects_nonpositive_constraint_uncertainty(tmp_path: Path) -> None:
    """``constraint.uncertainty_px`` must be strictly positive."""
    bad = _with_constraint(
        _VALID_SIDECAR_TEXT,
        _RANK1_CONSTRAINT_BLOCK.replace('uncertainty_px: 0.4', 'uncertainty_px: 0.0'),
    )
    p = tmp_path / 'BAD.yaml'
    p.write_text(bad)
    with pytest.raises(SidecarValidationError, match=r'uncertainty_px'):
        load_sidecar(p)


def test_load_sidecar_accepts_status_reason(tmp_path: Path) -> None:
    """An ``expected.status_reason`` from the NavStatusReason enum validates."""
    good = _VALID_SIDECAR_TEXT.replace(
        'primary_technique: BodyLimbNav',
        'primary_technique: BodyLimbNav\n  status_reason: rank_1_only',
    )
    p = tmp_path / 'TEST_IMG_0001.yaml'
    p.write_text(good)
    assert load_sidecar(p).expected.status_reason == 'rank_1_only'


def test_load_sidecar_rejects_unknown_status_reason(tmp_path: Path) -> None:
    """``expected.status_reason`` outside the NavStatusReason enum fails."""
    bad = _VALID_SIDECAR_TEXT.replace(
        'primary_technique: BodyLimbNav',
        'primary_technique: BodyLimbNav\n  status_reason: half_rank',
    )
    p = tmp_path / 'BAD.yaml'
    p.write_text(bad)
    with pytest.raises(SidecarValidationError, match=r'status_reason'):
        load_sidecar(p)


def test_load_sidecar_accepts_pending_issue(tmp_path: Path) -> None:
    """An ``expected.pending_issue`` of the form ``#<number>`` validates."""
    good = _VALID_SIDECAR_TEXT.replace(
        'primary_technique: BodyLimbNav',
        'primary_technique: BodyLimbNav\n  pending_issue: "#285"',
    )
    p = tmp_path / 'TEST_IMG_0002.yaml'
    p.write_text(good)
    assert load_sidecar(p).expected.pending_issue == '#285'


def test_load_sidecar_rejects_malformed_pending_issue(tmp_path: Path) -> None:
    """``expected.pending_issue`` that is not a ``#<number>`` reference fails."""
    bad = _VALID_SIDECAR_TEXT.replace(
        'primary_technique: BodyLimbNav',
        'primary_technique: BodyLimbNav\n  pending_issue: soon',
    )
    p = tmp_path / 'BAD.yaml'
    p.write_text(bad)
    with pytest.raises(SidecarValidationError, match=r'pending_issue'):
        load_sidecar(p)
