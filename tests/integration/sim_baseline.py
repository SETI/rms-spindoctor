"""Regression-baseline schema for the simulator scene catalog.

A sim baseline is a tiny JSON file at
``tests/integration/sim_baselines/<scene_name>.json`` recording the outcome the
orchestrator produced for a catalog scene:

```json
{
  "confidence": 0.3661834217129642,
  "confidence_rank": "medium",
  "offset_du_px": -1.796875,
  "offset_dv_px": 3.500421107761235,
  "scene_name": "planted_offset_disc",
  "status": "success"
}
```

Unlike the real-image baseline (which always navigates), a sim scene may fail;
``status`` is recorded and the offsets are ``null`` for a non-success run.

The offsets and the confidence are stored unrounded and compared within a
tolerance; the status and the confidence rank are compared exactly.  A rounded
value compared exactly fails whenever a tiny move crosses a rounding boundary: the
disc correlation's answer moves in steps of 1/128 px, one step of its upsampled
grid, and a last-bit change upstream can move it one step.  A per-axis tolerance
of :data:`OFFSET_TOLERANCE_PX` (0.02 px, about two and a half grid steps) absorbs
that and still catches a real regression.  The confidence is held to
:data:`CONFIDENCE_TOLERANCE`, a provisional value: how far a last-bit change can move
a scene's confidence has not been measured.  The rank comes from thresholds on the
confidence and on the offset's uncertainty, so a scene that sits near one of them can
still change rank, and fail, after a change far smaller than either tolerance.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from spindoctor.nav_model import build_models_for_obs
from spindoctor.nav_orchestrator import NavOrchestrator
from spindoctor.obs.obs_inst_sim import ObsSim

OFFSET_TOLERANCE_PX: float = 0.02
"""The largest per-axis offset difference, in pixels, that still matches a baseline."""

CONFIDENCE_TOLERANCE: float = 0.02
"""The largest confidence difference that still matches a baseline.

A provisional value: how far a last-bit change can move a confidence is unmeasured.
"""


@dataclass(frozen=True)
class SimBaseline:
    """A snapshot of one sim scene's navigation outcome, offsets and confidence unrounded."""

    scene_name: str
    status: str
    confidence_rank: str
    offset_dv_px: float | None
    offset_du_px: float | None
    confidence: float

    @classmethod
    def from_run(
        cls,
        *,
        scene_name: str,
        status: str,
        confidence_rank: str,
        offset_px: tuple[float, float] | None,
        confidence: float,
    ) -> SimBaseline:
        """Build a baseline from live navigation outputs, keeping every value as computed."""
        return cls(
            scene_name=scene_name,
            status=status,
            confidence_rank=confidence_rank,
            offset_dv_px=float(offset_px[0]) if offset_px is not None else None,
            offset_du_px=float(offset_px[1]) if offset_px is not None else None,
            confidence=float(confidence),
        )

    def to_json(self) -> str:
        """Serialize to deterministic JSON (sorted keys, no NaN, floats round-trip exactly)."""
        return (
            json.dumps(
                {
                    'scene_name': self.scene_name,
                    'status': self.status,
                    'confidence_rank': self.confidence_rank,
                    'offset_dv_px': self.offset_dv_px,
                    'offset_du_px': self.offset_du_px,
                    'confidence': self.confidence,
                },
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + '\n'
        )


def baseline_mismatches(expected: SimBaseline, actual: SimBaseline) -> list[str]:
    """Describe every way a fresh outcome fails to match its baseline.

    The scene name, the status, and the confidence rank must be equal.  An offset
    must be present in both or absent in both, and when present each axis must lie
    within :data:`OFFSET_TOLERANCE_PX` of the baseline's.  The confidence must lie
    within :data:`CONFIDENCE_TOLERANCE`.

    Parameters:
        expected: The committed baseline.
        actual: The outcome of a fresh render and navigation.

    Returns:
        One line per mismatch, giving the field and both values; empty when the
        outcome matches.
    """
    problems = [
        f'{name}: expected {want!r}, got {got!r}'
        for name, want, got in (
            ('scene_name', expected.scene_name, actual.scene_name),
            ('status', expected.status, actual.status),
            ('confidence_rank', expected.confidence_rank, actual.confidence_rank),
        )
        if want != got
    ]
    for name, want_px, got_px in (
        ('offset_dv_px', expected.offset_dv_px, actual.offset_dv_px),
        ('offset_du_px', expected.offset_du_px, actual.offset_du_px),
    ):
        if want_px is None or got_px is None:
            if want_px is not got_px:
                problems.append(f'{name}: expected {want_px!r}, got {got_px!r}')
        elif abs(got_px - want_px) > OFFSET_TOLERANCE_PX:
            problems.append(
                f'{name}: expected {want_px!r} within {OFFSET_TOLERANCE_PX} px, got {got_px!r}'
            )
    if abs(actual.confidence - expected.confidence) > CONFIDENCE_TOLERANCE:
        problems.append(
            f'confidence: expected {expected.confidence!r} within {CONFIDENCE_TOLERANCE}, '
            f'got {actual.confidence!r}'
        )
    return problems


_REQUIRED_KEYS: tuple[str, ...] = (
    'scene_name',
    'status',
    'confidence_rank',
    'offset_dv_px',
    'offset_du_px',
    'confidence',
)


def load_sim_baseline(path: Path) -> SimBaseline:
    """Parse one sim baseline JSON file with strict schema validation."""
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f'{path}: cannot parse JSON: {exc}') from exc
    if not isinstance(raw, dict):
        raise ValueError(f'{path}: top-level JSON must be a mapping, got {type(raw).__name__}')
    missing = [k for k in _REQUIRED_KEYS if k not in raw]
    if missing:
        raise KeyError(f'{path}: missing required key(s): {missing}')
    return SimBaseline(
        scene_name=_require_str(raw, 'scene_name', path),
        status=_require_str(raw, 'status', path),
        confidence_rank=_require_str(raw, 'confidence_rank', path),
        offset_dv_px=_optional_float(raw['offset_dv_px'], 'offset_dv_px', path),
        offset_du_px=_optional_float(raw['offset_du_px'], 'offset_du_px', path),
        confidence=_require_float(raw['confidence'], 'confidence', path),
    )


def baseline_for_scene(scene: dict[str, Any]) -> SimBaseline:
    """Render and navigate the scene ``sim_params``, returning its baseline."""
    obs = ObsSim.from_file('/tmp/sim_baseline.yaml', sim_params=scene)
    result = NavOrchestrator(
        build_models_for_obs(obs), only_models='*', only_techniques='*'
    ).navigate(obs)
    return SimBaseline.from_run(
        scene_name=scene['scene_name'],
        status=str(result.status),
        confidence_rank=str(result.confidence_rank),
        offset_px=result.offset_px,
        confidence=float(result.confidence),
    )


def sim_baseline_path(baselines_dir: Path, scene_name: str) -> Path:
    """Return the canonical baseline path for a scene."""
    return baselines_dir / f'{scene_name}.json'


def discover_sim_baseline_paths(baselines_dir: Path) -> list[Path]:
    """List every ``<scene_name>.json`` under ``sim_baselines/`` (sorted)."""
    if not baselines_dir.is_dir():
        return []
    return sorted(baselines_dir.glob('*.json'))


def _require_str(raw: dict[str, Any], key: str, path: Path) -> str:
    value = raw[key]
    if not isinstance(value, str):
        raise TypeError(f'{path}: {key!r} must be a string, got {type(value).__name__}')
    return value


def _require_float(value: Any, key: str, path: Path) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f'{path}: {key!r} must be a number, got {type(value).__name__}')
    return float(value)


def _optional_float(value: Any, key: str, path: Path) -> float | None:
    if value is None:
        return None
    return _require_float(value, key, path)
