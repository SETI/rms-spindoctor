"""Sidecar schema for the operator-curated image library (Part 10 §"Per-image
sidecar schema").

A sidecar is a YAML file at
``tests/integration/image_library/images/<class>/<image_id>.yaml`` carrying
operator-verified ground-truth offset, expected-status / expected-confidence-tier
targets, and provenance.  The schema is the single source of truth: ``test_image_library``
exercises its structural invariants; ``test_autonomous_nav`` consumes the parsed
fields to score real-holdings runs.

The validator is intentionally hand-rolled (pydantic is not a project dependency).
Every field that the orchestrator or the regression test reads is checked here so
malformed entries break collection rather than producing a confusing per-image
failure later.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML

from spindoctor.support.status_reason import NavStatusReason

# ---------------------------------------------------------------------------
# Library-wide enumerations
# ---------------------------------------------------------------------------

# All scene classes declared in Part 10 §"Library structure".  The structural
# test asserts that every subdirectory under ``images/`` is a member of this set
# so typos like ``body_overflow`` vs ``body_partial_overflow`` fail loudly.
DECLARED_SCENE_CLASSES: frozenset[str] = frozenset(
    {
        'star_dominated',
        'body_full_fov',
        'body_partial_overflow',
        'body_mostly_offscreen',
        'body_irregular',
        'multi_body',
        'ring_only_curved',
        'ring_only_flat',
        'ring_plus_body',
        'stars_plus_body',
        'one_bright_star_no_body',
        'two_bright_stars_no_body',
        'faint_stars',
        'scattered_light',
        'high_phase_terminator',
        'below_resolution_body',
        'negative_cases',
    }
)

ALLOWED_MISSIONS: frozenset[str] = frozenset(
    {
        'COISS',
        'VGISS',
        'GOSSI',
        'NHLORRI',
    }
)
"""Mission codes accepted by ``Sidecar.mission``.

Match the dataset names registered in :mod:`spindoctor.dataset` upper-cased
(``coiss`` / ``vgiss`` / ``gossi`` / ``nhlorri``) so the sidecar's
``mission`` is unambiguous against a CLI invocation like
``sd_offset --dataset coiss``.
"""

ALLOWED_CAMERAS: frozenset[str] = frozenset(
    {
        'NAC',
        'WAC',
        'SSI',
        'NA',
        'WA',
        'LORRI',
    }
)

ALLOWED_STATUSES: frozenset[str] = frozenset({'success', 'failed', 'conflicted'})
# 'conflicted' mirrors :data:`spindoctor.nav_orchestrator.nav_result.ConfidenceRank` —
# the orchestrator hard-sets the rank to 'conflicted' whenever it returns a
# conflicted NavResult, so the sidecar schema has to accept it as an
# expected.confidence_tier value alongside the four tier names.
ALLOWED_TIERS: frozenset[str] = frozenset({'high', 'medium', 'low', 'failed', 'conflicted'})
ALLOWED_GT_SOURCES: frozenset[str] = frozenset({'operator_verified'})
ALLOWED_CONSTRAINT_TYPES: frozenset[str] = frozenset({'rank_1'})
ALLOWED_STATUS_REASONS: frozenset[str] = frozenset({reason.value for reason in NavStatusReason})
_PENDING_ISSUE_RE: re.Pattern[str] = re.compile(r'#\d+')

CURRENT_SCHEMA_VERSION: int = 1

# Maximum disagreement (px) tolerated between the 2-D representative offset
# projected onto the constraint normal and the declared
# ``offset_along_normal_px``.  Both come from the same manual-nav save, so any
# larger gap means one of them was hand-edited inconsistently.
_CONSTRAINT_PROJECTION_TOLERANCE_PX: float = 0.01


# ---------------------------------------------------------------------------
# Dataclasses (one per nested block)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GroundTruthConstraint:
    """Single-axis (rank-1) observability constraint on a ground-truth offset.

    For a straight (flat) ring edge only the offset component along the edge
    normal is observable; the along-edge component is physically
    unconstrained.  ``normal_angle_deg`` is the edge-normal direction in the
    ``(v, u)`` frame, measured from ``+v`` toward ``+u`` — the unit normal is
    ``(cos(angle), sin(angle))`` and a 2-D offset's observable component is
    ``dv * cos(angle) + du * sin(angle)``.
    """

    type: str
    normal_angle_deg: float
    offset_along_normal_px: float
    uncertainty_px: float


@dataclass(frozen=True)
class GroundTruth:
    """Operator-verified offset for one image.

    When ``constraint`` is present the 2-D ``offset_dv_px`` /
    ``offset_du_px`` pair is a *representative point* on the constraint line
    (whatever the operator saved from the manual-nav drag); only the
    component along the constraint normal is operator-verified, and the
    regression comparison projects onto that normal instead of comparing per
    axis.
    """

    offset_dv_px: float
    offset_du_px: float
    offset_uncertainty_px: float
    source: str
    operator: str
    verified_date: date
    ui_version: str
    notes: str | None = None
    constraint: GroundTruthConstraint | None = None


@dataclass(frozen=True)
class Expected:
    """Expected-outcome targets the regression test compares against.

    ``status_reason`` is optional: when present the regression test also
    asserts the orchestrator's discrete ``NavStatusReason`` (e.g.
    ``rank_1_only`` for a flat-ring frame whose only observable axis is the
    edge normal); when absent only the status / tier are checked.

    ``pending_issue`` marks a fixture whose pinned outcome is provisional:
    the frame is believed navigable in principle but the current pipeline
    cannot yet navigate it, and the referenced issue (format ``#<number>``)
    tracks closing that gap. The regression test still asserts the pinned
    outcome, so when the issue is fixed and behavior changes the test
    fails loudly and prompts the fixture to be re-verified. It is a
    machine-readable flag distinguishing "provisionally failing, revisit"
    from a genuinely-unnavigable negative case.
    """

    status: str
    confidence_tier: str
    primary_technique: str
    techniques_must_run: tuple[str, ...] = ()
    techniques_must_skip: tuple[str, ...] = ()
    status_reason: str | None = None
    pending_issue: str | None = None


@dataclass(frozen=True)
class CameraRotationExpected:
    """Optional expected camera-rotation for fit_camera_rotation runs."""

    rotation_deg: float | None
    uncertainty_deg: float | None


@dataclass(frozen=True)
class Sidecar:
    """A fully validated library sidecar.

    Use :func:`load_sidecar` to parse + validate one YAML file.  Direct
    instantiation is also supported in tests.
    """

    path: Path
    schema_version: int
    image_id: str
    mission: str
    camera: str
    filter_combo: str
    image_url: str
    scene_tags: tuple[str, ...]
    ground_truth: GroundTruth
    expected: Expected
    exposure_time_sec: float | None = None
    image_datetime_utc: str | None = None
    camera_rotation_expected: CameraRotationExpected | None = None

    @property
    def primary_scene_tag(self) -> str:
        """First entry of ``scene_tags`` — must equal the containing directory."""
        return self.scene_tags[0]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class SidecarValidationError(ValueError):
    """Raised when a sidecar fails schema validation."""


def load_sidecar(path: Path) -> Sidecar:
    """Parse ``path`` as a sidecar YAML and validate every field.

    Parameters:
        path: Filesystem path to a ``<image_id>.yaml`` sidecar.

    Returns:
        A frozen :class:`Sidecar` ready for downstream use.

    Raises:
        SidecarValidationError: If any required field is missing, has the
            wrong type, or violates an enum / cross-field constraint.
    """
    yaml = YAML(typ='safe')
    try:
        raw = yaml.load(path.read_text())
    except Exception as exc:  # ruamel may raise several exception types
        raise SidecarValidationError(f'{path}: cannot parse YAML: {exc}') from exc
    if not isinstance(raw, dict):
        raise SidecarValidationError(f'{path}: top-level YAML must be a mapping')
    return _validate_sidecar(raw, path=path)


def _validate_sidecar(raw: dict[str, Any], *, path: Path) -> Sidecar:
    schema_version = _require_int(raw, 'schema_version', path=path)
    if schema_version != CURRENT_SCHEMA_VERSION:
        raise SidecarValidationError(
            f'{path}: schema_version must be {CURRENT_SCHEMA_VERSION}, got {schema_version}'
        )

    image_id = _require_str(raw, 'image_id', path=path)
    mission = _require_enum(raw, 'mission', ALLOWED_MISSIONS, path=path)
    camera = _require_enum(raw, 'camera', ALLOWED_CAMERAS, path=path)
    filter_combo = _require_str(raw, 'filter_combo', path=path)
    image_url = _require_str(raw, 'image_url', path=path)

    scene_tags = _require_str_list(raw, 'scene_tags', path=path, min_len=1)
    if len(scene_tags) != len(set(scene_tags)):
        raise SidecarValidationError(f'{path}: scene_tags contains duplicates: {scene_tags!r}')
    if scene_tags[0] not in DECLARED_SCENE_CLASSES:
        raise SidecarValidationError(
            f'{path}: primary scene_tag {scene_tags[0]!r} is not a declared '
            f'scene class (must be one of {sorted(DECLARED_SCENE_CLASSES)})'
        )

    ground_truth = _validate_ground_truth(
        _require_mapping(raw, 'ground_truth', path=path), path=path
    )
    expected = _validate_expected(_require_mapping(raw, 'expected', path=path), path=path)
    camera_rotation_expected = _validate_camera_rotation(
        raw.get('camera_rotation_expected'), path=path
    )
    exposure_time_sec = _validate_optional_exposure(raw.get('exposure_time_sec'), path=path)
    image_datetime_utc = _validate_optional_datetime(raw.get('image_datetime_utc'), path=path)

    return Sidecar(
        path=path,
        schema_version=schema_version,
        image_id=image_id,
        mission=mission,
        camera=camera,
        filter_combo=filter_combo,
        image_url=image_url,
        scene_tags=tuple(scene_tags),
        ground_truth=ground_truth,
        expected=expected,
        exposure_time_sec=exposure_time_sec,
        image_datetime_utc=image_datetime_utc,
        camera_rotation_expected=camera_rotation_expected,
    )


def _validate_optional_datetime(raw: Any, *, path: Path) -> str | None:
    """Validate the optional ``image_datetime_utc`` field.

    Permitted values: missing / None (legacy sidecars predating the
    field), or a non-empty string.  No format check beyond non-empty:
    the field is informational metadata for future cross-referencing
    against PDS labels, not a navigation input.
    """
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise SidecarValidationError(
            f'{path}: image_datetime_utc must be a string, got {type(raw).__name__}'
        )
    if not raw.strip():
        raise SidecarValidationError(
            f'{path}: image_datetime_utc must be a non-empty string when present'
        )
    return raw


def _validate_optional_exposure(raw: Any, *, path: Path) -> float | None:
    """Validate the optional ``exposure_time_sec`` field.

    Permitted values: missing / None (legacy sidecars predating Phase 10
    metadata expansion), or a finite positive float.  Anything else is
    a hard error so a typo or unit mistake fails loudly at load time.
    """
    if raw is None:
        return None
    if isinstance(raw, bool):
        raise SidecarValidationError(
            f'{path}: exposure_time_sec must be a positive number, got bool {raw!r}'
        )
    if not isinstance(raw, (int, float)):
        raise SidecarValidationError(
            f'{path}: exposure_time_sec must be a number, got {type(raw).__name__}'
        )
    coerced = float(raw)
    if not math.isfinite(coerced) or coerced <= 0.0:
        raise SidecarValidationError(
            f'{path}: exposure_time_sec must be a finite positive number, got {raw!r}'
        )
    return coerced


def _validate_ground_truth(raw: dict[str, Any], *, path: Path) -> GroundTruth:
    offset_dv_px = _require_float(raw, 'offset_dv_px', path=path)
    offset_du_px = _require_float(raw, 'offset_du_px', path=path)
    offset_uncertainty_px = _require_float(raw, 'offset_uncertainty_px', path=path)
    if offset_uncertainty_px <= 0.0:
        raise SidecarValidationError(
            f'{path}: ground_truth.offset_uncertainty_px must be > 0, got {offset_uncertainty_px}'
        )
    source = _require_enum(raw, 'source', ALLOWED_GT_SOURCES, path=path)
    operator = _require_str(raw, 'operator', path=path)
    verified_date = _require_date(raw, 'verified_date', path=path)
    ui_version = _require_str(raw, 'ui_version', path=path)
    notes = raw.get('notes')
    if notes is not None and not isinstance(notes, str):
        raise SidecarValidationError(f'{path}: ground_truth.notes must be a string when present')
    constraint = _validate_constraint(raw.get('constraint'), path=path)
    if constraint is not None:
        projected = offset_dv_px * math.cos(
            math.radians(constraint.normal_angle_deg)
        ) + offset_du_px * math.sin(math.radians(constraint.normal_angle_deg))
        if abs(projected - constraint.offset_along_normal_px) > _CONSTRAINT_PROJECTION_TOLERANCE_PX:
            raise SidecarValidationError(
                f'{path}: ground_truth (offset_dv_px, offset_du_px) projected onto the '
                f'constraint normal is {projected:.4f} px but '
                f'constraint.offset_along_normal_px is '
                f'{constraint.offset_along_normal_px:.4f} px; the representative point '
                f'must lie on the constraint line'
            )
    return GroundTruth(
        offset_dv_px=offset_dv_px,
        offset_du_px=offset_du_px,
        offset_uncertainty_px=offset_uncertainty_px,
        source=source,
        operator=operator,
        verified_date=verified_date,
        ui_version=ui_version,
        notes=notes,
        constraint=constraint,
    )


def _validate_constraint(raw: Any, *, path: Path) -> GroundTruthConstraint | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise SidecarValidationError(
            f'{path}: ground_truth.constraint must be a mapping when present'
        )
    constraint_type = _require_enum(raw, 'type', ALLOWED_CONSTRAINT_TYPES, path=path)
    normal_angle_deg = _require_float(raw, 'normal_angle_deg', path=path)
    offset_along_normal_px = _require_float(raw, 'offset_along_normal_px', path=path)
    uncertainty_px = _require_float(raw, 'uncertainty_px', path=path)
    if uncertainty_px <= 0.0:
        raise SidecarValidationError(
            f'{path}: ground_truth.constraint.uncertainty_px must be > 0, got {uncertainty_px}'
        )
    return GroundTruthConstraint(
        type=constraint_type,
        normal_angle_deg=normal_angle_deg,
        offset_along_normal_px=offset_along_normal_px,
        uncertainty_px=uncertainty_px,
    )


def _validate_expected(raw: dict[str, Any], *, path: Path) -> Expected:
    status = _require_enum(raw, 'status', ALLOWED_STATUSES, path=path)
    confidence_tier = _require_enum(raw, 'confidence_tier', ALLOWED_TIERS, path=path)
    if status == 'failed' and confidence_tier != 'failed':
        raise SidecarValidationError(
            f'{path}: expected.status=failed requires expected.confidence_tier=failed'
        )
    if status != 'failed' and confidence_tier == 'failed':
        raise SidecarValidationError(
            f'{path}: expected.confidence_tier=failed requires expected.status=failed'
        )
    if status == 'conflicted' and confidence_tier != 'conflicted':
        raise SidecarValidationError(
            f'{path}: expected.status=conflicted requires expected.confidence_tier=conflicted'
        )
    if status != 'conflicted' and confidence_tier == 'conflicted':
        raise SidecarValidationError(
            f'{path}: expected.confidence_tier=conflicted requires expected.status=conflicted'
        )
    primary_technique = _require_str(raw, 'primary_technique', path=path)
    techniques_must_run = _optional_str_tuple(
        raw.get('techniques_must_run'), label='techniques_must_run', path=path
    )
    techniques_must_skip = _optional_str_tuple(
        raw.get('techniques_must_skip'), label='techniques_must_skip', path=path
    )
    overlap = set(techniques_must_run) & set(techniques_must_skip)
    if overlap:
        raise SidecarValidationError(
            f'{path}: techniques_must_run and techniques_must_skip overlap: {sorted(overlap)}'
        )
    status_reason = raw.get('status_reason')
    if status_reason is not None and (
        not isinstance(status_reason, str) or status_reason not in ALLOWED_STATUS_REASONS
    ):
        raise SidecarValidationError(
            f'{path}: expected.status_reason must be one of '
            f'{sorted(ALLOWED_STATUS_REASONS)} when present, got {status_reason!r}'
        )
    pending_issue = raw.get('pending_issue')
    if pending_issue is not None and (
        not isinstance(pending_issue, str) or not _PENDING_ISSUE_RE.fullmatch(pending_issue)
    ):
        raise SidecarValidationError(
            f'{path}: expected.pending_issue must be an issue reference of the '
            f'form "#<number>" when present, got {pending_issue!r}'
        )
    return Expected(
        status=status,
        confidence_tier=confidence_tier,
        primary_technique=primary_technique,
        techniques_must_run=techniques_must_run,
        techniques_must_skip=techniques_must_skip,
        status_reason=status_reason,
        pending_issue=pending_issue,
    )


def _validate_camera_rotation(raw: Any, *, path: Path) -> CameraRotationExpected | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise SidecarValidationError(
            f'{path}: camera_rotation_expected must be a mapping when present'
        )
    rotation_deg = raw.get('rotation_deg')
    uncertainty_deg = raw.get('uncertainty_deg')
    for label, value in (('rotation_deg', rotation_deg), ('uncertainty_deg', uncertainty_deg)):
        if value is None:
            continue
        if isinstance(value, bool):
            raise SidecarValidationError(
                f'{path}: camera_rotation_expected.{label} must be a number or null, '
                f'got bool {value!r}'
            )
        if not isinstance(value, (int, float)):
            raise SidecarValidationError(
                f'{path}: camera_rotation_expected.{label} must be a number or null'
            )
        if not math.isfinite(float(value)):
            raise SidecarValidationError(
                f'{path}: camera_rotation_expected.{label} must be a finite number or null, '
                f'got {value!r}'
            )
    return CameraRotationExpected(
        rotation_deg=None if rotation_deg is None else float(rotation_deg),
        uncertainty_deg=None if uncertainty_deg is None else float(uncertainty_deg),
    )


# ---------------------------------------------------------------------------
# Discovery / cross-sidecar invariants
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LibraryRoot:
    """Root of the on-disk library; computed from this file's location."""

    root: Path = field(default_factory=lambda: Path(__file__).resolve().parent / 'image_library')

    @property
    def images(self) -> Path:
        """Path to ``image_library/images``."""
        return self.root / 'images'

    @property
    def baselines(self) -> Path:
        """Path to ``tests/integration/baselines``."""
        return self.root.parent / 'baselines'

    def discover_sidecar_paths(self) -> list[Path]:
        """Return every ``<class>/<image_id>.yaml`` under ``images/`` (sorted)."""
        if not self.images.is_dir():
            return []
        return sorted(self.images.glob('*/*.yaml'))

    def discover_class_directories(self) -> list[Path]:
        """Return every direct subdirectory of ``images/`` (sorted)."""
        if not self.images.is_dir():
            return []
        return sorted(p for p in self.images.iterdir() if p.is_dir())


def load_all_sidecars(root: LibraryRoot | None = None) -> list[Sidecar]:
    """Discover and validate every sidecar under ``image_library/images``."""
    paths = (root or LibraryRoot()).discover_sidecar_paths()
    return [load_sidecar(p) for p in paths]


# ---------------------------------------------------------------------------
# Field-validation helpers
# ---------------------------------------------------------------------------


def _missing(path: Path, key: str) -> SidecarValidationError:
    return SidecarValidationError(f'{path}: missing required field {key!r}')


def _optional_str_tuple(value: Any, *, label: str, path: Path) -> tuple[str, ...]:
    """Validate an optional list-of-strings field; ``None`` becomes ``()``."""
    if value is None:
        return ()
    if not isinstance(value, list):
        raise SidecarValidationError(
            f'{path}: expected.{label} must be a list of strings, got {type(value).__name__}'
        )
    out: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise SidecarValidationError(
                f'{path}: expected.{label} must be a list of strings; got non-string entry {item!r}'
            )
        out.append(item)
    return tuple(out)


def _require_str(raw: dict[str, Any], key: str, *, path: Path) -> str:
    if key not in raw:
        raise _missing(path, key)
    value = raw[key]
    if not isinstance(value, str) or not value:
        raise SidecarValidationError(f'{path}: {key!r} must be a non-empty string, got {value!r}')
    return value


def _require_int(raw: dict[str, Any], key: str, *, path: Path) -> int:
    if key not in raw:
        raise _missing(path, key)
    value = raw[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise SidecarValidationError(f'{path}: {key!r} must be an int, got {value!r}')
    return int(value)


def _require_float(raw: dict[str, Any], key: str, *, path: Path) -> float:
    if key not in raw:
        raise _missing(path, key)
    value = raw[key]
    if isinstance(value, bool):
        raise SidecarValidationError(f'{path}: {key!r} must be a number, got bool {value!r}')
    if not isinstance(value, (int, float)):
        raise SidecarValidationError(f'{path}: {key!r} must be a number, got {value!r}')
    coerced = float(value)
    if not math.isfinite(coerced):
        raise SidecarValidationError(f'{path}: {key!r} must be a finite number, got {value!r}')
    return coerced


def _require_enum(raw: dict[str, Any], key: str, allowed: frozenset[str], *, path: Path) -> str:
    value = _require_str(raw, key, path=path)
    if value not in allowed:
        raise SidecarValidationError(
            f'{path}: {key!r} must be one of {sorted(allowed)}, got {value!r}'
        )
    return value


def _require_date(raw: dict[str, Any], key: str, *, path: Path) -> date:
    if key not in raw:
        raise _missing(path, key)
    value = raw[key]
    if isinstance(value, date):
        return value
    raise SidecarValidationError(f'{path}: {key!r} must be a date (YYYY-MM-DD), got {value!r}')


def _require_str_list(raw: dict[str, Any], key: str, *, path: Path, min_len: int = 0) -> list[str]:
    if key not in raw:
        raise _missing(path, key)
    value = raw[key]
    if not isinstance(value, list):
        raise SidecarValidationError(
            f'{path}: {key!r} must be a list of strings, got {type(value).__name__}'
        )
    out: list[str] = []
    for i, item in enumerate(value):
        if not isinstance(item, str):
            raise SidecarValidationError(f'{path}: {key!r}[{i}] must be a string, got {item!r}')
        out.append(item)
    if len(out) < min_len:
        raise SidecarValidationError(
            f'{path}: {key!r} must have at least {min_len} entries, got {len(out)}'
        )
    return out


def _require_mapping(raw: dict[str, Any], key: str, *, path: Path) -> dict[str, Any]:
    if key not in raw:
        raise _missing(path, key)
    value = raw[key]
    if not isinstance(value, dict):
        raise SidecarValidationError(
            f'{path}: {key!r} must be a mapping, got {type(value).__name__}'
        )
    return value
