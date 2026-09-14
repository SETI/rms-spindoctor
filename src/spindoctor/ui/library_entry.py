"""Save-as-library-entry helper for the manual-navigation dialog.

The manual-nav dialog drives library curation: an operator picks an
offset by hand, then clicks "Save as Library Entry..." to drop a sidecar
into ``tests/integration/image_library/images/<class>/<image_id>.yaml``.
This module owns the YAML template and the obs-introspection helper so
the dialog (which is otherwise PyQt-only) keeps a thin UI surface.

The emitted YAML carries ``TODO``-style placeholders for every field the
dialog cannot infer from the observation (scene class, expected
technique, etc.).  The library's own structural-invariants test
(:mod:`tests.integration.test_image_library`) refuses to load a sidecar
that still has placeholder enum values, so the operator must edit the
file before committing it.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from datetime import date
from typing import Any

__all__ = [
    'LibraryEntryDraft',
    'build_sidecar_yaml',
    'compute_image_url',
    'infer_obs_metadata',
]


# Mission-string mapping — keys are class names of the supported obs subclasses.
# Mission codes match the dataset names in
# :mod:`spindoctor.dataset` (``coiss`` / ``vgiss`` / ``gossi`` / ``nhlorri``)
# upper-cased, so a sidecar's ``mission`` is unambiguous against a CLI
# invocation like ``sd_offset --dataset coiss``.  Add a new entry here
# when adding a new instrument; keep the value in
# :data:`tests.integration.sidecar.ALLOWED_MISSIONS` in sync.
_OBS_CLASS_TO_MISSION: dict[str, str] = {
    'ObsCassiniISS': 'COISS',
    'ObsVoyagerISS': 'VGISS',
    'ObsGalileoSSI': 'GOSSI',
    'ObsNewHorizonsLORRI': 'NHLORRI',
}


@dataclass(frozen=True)
class LibraryEntryDraft:
    """Auto-inferred fields the dialog seeds into a fresh sidecar."""

    image_id: str
    mission: str
    camera: str
    filter_combo: str
    exposure_time_sec: float | None
    image_datetime_utc: str | None


def infer_obs_metadata(obs: Any) -> LibraryEntryDraft:
    """Pull the sidecar's auto-fillable fields off an observation snapshot.

    Missing or unknown fields are returned as empty strings (or ``None``
    for ``exposure_time_sec`` / ``image_datetime_utc``) so the operator
    sees them in the YAML and can decide what to write; an empty
    mission/camera trips :func:`tests.integration.sidecar.load_sidecar`
    on validation, which is the right error.
    """
    image_id = ''
    abspath = getattr(obs, 'abspath', None)
    if abspath is not None and getattr(abspath, 'stem', None):
        image_id = str(abspath.stem)

    mission = _OBS_CLASS_TO_MISSION.get(type(obs).__name__, '')

    camera = ''
    detector = getattr(obs, 'detector', None)
    if detector:
        camera = str(detector)

    f1 = getattr(obs, 'filter1', None)
    f2 = getattr(obs, 'filter2', None)
    filters = [str(f) for f in (f1, f2) if f]
    filter_combo = '+'.join(sorted(filters)) if filters else ''

    exposure_time_sec: float | None = None
    raw_texp = getattr(obs, 'texp', None)
    if raw_texp is not None:
        try:
            value = float(raw_texp)
        except (TypeError, ValueError):
            value = float('nan')
        if value > 0.0 and value == value:  # finite & positive
            exposure_time_sec = value

    image_datetime_utc: str | None = None
    raw_midtime = getattr(obs, 'midtime', None)
    if raw_midtime is not None:
        try:
            from spindoctor.support.time import et_to_utc

            image_datetime_utc = et_to_utc(float(raw_midtime), digits=3)
        except Exception:
            image_datetime_utc = None

    return LibraryEntryDraft(
        image_id=image_id,
        mission=mission,
        camera=camera,
        filter_combo=filter_combo,
        exposure_time_sec=exposure_time_sec,
        image_datetime_utc=image_datetime_utc,
    )


def compute_image_url(obs: Any, config: Any = None) -> str:
    """Return the best ``image_url`` for the sidecar.

    When the obs's absolute path lives under a known PDS3 holdings root
    (resolved from ``PDS3_HOLDINGS_DIR`` first, then
    ``config.environment.pds3_holdings_root``), the URL is rewritten to
    the opaque ``pds3://<relative-path>`` form the regression test
    re-resolves at load time.  Otherwise the raw absolute path is
    returned so the operator at least sees what was navigated.

    Parameters:
        obs: Observation snapshot; ``obs.abspath`` is the file path.
        config: Optional ``Config`` override for the
            ``environment.pds3_holdings_root`` fallback.

    Returns:
        Either ``'pds3://<relative-path>'`` or the raw absolute path
        as a string.  Empty string when ``obs`` carries no ``abspath``.
    """
    abspath = getattr(obs, 'abspath', None)
    if abspath is None:
        return ''
    abspath_str = str(abspath)
    holdings_root = _resolve_pds3_holdings_root(config)
    if holdings_root:
        rel = _relative_to_holdings(abspath_str, holdings_root)
        if rel is not None:
            return f'pds3://{rel}'
    return abspath_str


def _resolve_pds3_holdings_root(config: Any) -> str:
    """Return the configured PDS3 holdings root or ``''`` if unknown.

    Order matches :class:`spindoctor.dataset.dataset_pds3.DataSetPDS3`:
    ``PDS3_HOLDINGS_DIR`` env var, then
    ``config.environment.pds3_holdings_root``.  Missing or empty
    configurations yield ``''``.
    """
    env_root = os.environ.get('PDS3_HOLDINGS_DIR')
    if env_root:
        return env_root.rstrip('/')
    if config is None:
        try:
            from spindoctor.config import DEFAULT_CONFIG

            config = DEFAULT_CONFIG
        except Exception:  # pragma: no cover - bootstrap sandbox
            return ''
    try:
        environment = config.environment
    except Exception:  # pragma: no cover - partial-config sandbox
        return ''
    cfg_root = getattr(environment, 'pds3_holdings_root', None)
    if not cfg_root:
        return ''
    return str(cfg_root).rstrip('/')


def _relative_to_holdings(abspath_str: str, holdings_root: str) -> str | None:
    """Return the holdings-relative path or ``None`` if abspath is outside.

    Tolerates trailing slashes and case-sensitive comparisons; does not
    walk the filesystem.  ``holdings_root`` is expected to already be
    normalized by :func:`_resolve_pds3_holdings_root` (no trailing
    slash).
    """
    if not abspath_str.startswith(holdings_root):
        return None
    remainder = abspath_str[len(holdings_root) :]
    return remainder.lstrip('/')


def build_sidecar_yaml(
    *,
    draft: LibraryEntryDraft,
    image_url: str,
    offset_dv_px: float,
    offset_du_px: float,
    ui_version: str,
    operator: str | None = None,
    today: date | None = None,
    constraint_normal_angle_deg: float | None = None,
) -> str:
    """Render the sidecar as a YAML string with placeholders for the rest.

    The output is deliberately not an opaque blob: every placeholder is
    a clear ``TODO_REPLACE_*`` string so the operator knows exactly what
    to edit before committing.  Fields the dialog can infer
    (``offset_*_px``, ``mission``, ``camera``, ``filter_combo``,
    ``exposure_time_sec``, ``image_id``) are filled in directly.

    When ``constraint_normal_angle_deg`` is set (the operator navigated
    with the constrained-drag mode on a rank-1 scene), the ground truth
    carries a ``constraint`` block — ``offset_along_normal_px`` computed
    from the saved 2-D offset so the two stay consistent by construction —
    and ``expected`` pins ``status_reason: rank_1_only``.
    """
    op_name = operator or os.environ.get('USER') or 'unknown'
    on_date = (today or date.today()).isoformat()
    if draft.exposure_time_sec is None:
        exposure_block = '# seconds; from obs.texp\nexposure_time_sec: TODO_REPLACE_EXPOSURE\n'
    else:
        exposure_block = f'exposure_time_sec: {draft.exposure_time_sec:.4f}\n'
    if draft.image_datetime_utc is None:
        datetime_block = (
            '# UTC ISO 8601; from et_to_utc(obs.midtime)\n'
            "image_datetime_utc: 'TODO_REPLACE_DATETIME'\n"
        )
    else:
        datetime_block = f"image_datetime_utc: '{draft.image_datetime_utc}'\n"
    constraint_block = ''
    status_reason_block = ''
    if constraint_normal_angle_deg is not None:
        angle_rad = math.radians(constraint_normal_angle_deg)
        along_normal = offset_dv_px * math.cos(angle_rad) + offset_du_px * math.sin(angle_rad)
        constraint_block = (
            '  # Rank-1 scene: only the offset component along this normal is\n'
            '  # observable; (offset_dv_px, offset_du_px) is a representative\n'
            '  # point on the constraint line.\n'
            '  constraint:\n'
            '    type: rank_1\n'
            f'    normal_angle_deg: {constraint_normal_angle_deg:.4f}\n'
            f'    offset_along_normal_px: {along_normal:.4f}\n'
            '    # 1sigma along the normal; tighten for sharp edges.\n'
            '    uncertainty_px: 1.0\n'
        )
        status_reason_block = '  status_reason: rank_1_only\n'
    return (
        'schema_version: 1\n'
        '# COISS | VGISS | GOSSI | NHLORRI\n'
        f'mission: {draft.mission or "TODO_REPLACE_MISSION"}\n'
        '# NAC | WAC | SSI | NA | WA | LORRI\n'
        f'camera: {draft.camera or "TODO_REPLACE_CAMERA"}\n'
        f'image_id: {draft.image_id or "TODO_REPLACE_IMAGE_ID"}\n'
        + datetime_block
        + exposure_block
        + "# canonicalized: filters sorted, '+'-joined\n"
        + f"filter_combo: '{draft.filter_combo}'\n"
        + f"image_url: '{image_url}'\n"
        '\n'
        '# First tag is the primary class; must match the directory the\n'
        '# sidecar lives in.\n'
        'scene_tags:\n'
        '  - TODO_REPLACE_PRIMARY_CLASS\n'
        '\n'
        'ground_truth:\n'
        f'  offset_dv_px: {offset_dv_px:.4f}\n'
        f'  offset_du_px: {offset_du_px:.4f}\n'
        '  # 1sigma marginal; tighten for bright stars / sharp limbs.\n'
        '  offset_uncertainty_px: 1.0\n' + constraint_block + '  source: operator_verified\n'
        f'  operator: {op_name}\n'
        f'  verified_date: {on_date}\n'
        f"  ui_version: 'spindoctor {ui_version}'\n"
        '  notes: |\n'
        '    TODO: describe the scene and any caveats.\n'
        '\n'
        'expected:\n'
        '  # success | failed | conflicted\n'
        '  status: success\n'
        '  # high | medium | low | failed\n'
        '  confidence_tier: high\n' + status_reason_block + '  # e.g. BodyLimbNav\n'
        '  primary_technique: TODO_REPLACE_TECHNIQUE\n'
        '  techniques_must_run: []\n'
        '  techniques_must_skip: []\n'
    )
