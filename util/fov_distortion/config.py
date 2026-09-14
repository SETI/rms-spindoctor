"""Per-instrument analysis configuration.

Each instrument-and-camera cohort is a YAML sidecar in ``configs/`` naming the
star frames to analyze and the detection parameters to use.  Frame lists are
carried verbatim (with ``${PDS3_HOLDINGS_DIR}`` and similar tokens) so the same
files drive the analysis on any holdings mirror.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML
from util.fov_distortion.measure import MeasureParams

__all__ = ['AnalysisConfig', 'load_config']


@dataclass(frozen=True)
class AnalysisConfig:
    """One instrument-and-camera analysis cohort.

    Parameters:
        inst_id: Instrument id (``coiss`` / ``vgiss`` / ``nhlorri`` / ``gossi``).
        label: Human-readable instrument / camera label for figures and tables.
        camera: Camera name (``NAC`` / ``WAC`` / ...), or ``None`` for a
            single-camera instrument.
        images: Frame URLs, possibly with unexpanded environment tokens.
        params: Detection and rejection parameters.
    """

    inst_id: str
    label: str
    camera: str | None
    images: list[str]
    params: MeasureParams


def load_config(path: str | Path) -> AnalysisConfig:
    """Load an analysis cohort from a YAML sidecar.

    Parameters:
        path: Path to the YAML file.

    Returns:
        The parsed :class:`AnalysisConfig`.

    Raises:
        ValueError: if required keys are missing or a parameter is unknown.
    """
    yaml = YAML(typ='safe')
    with open(path, encoding='utf-8') as handle:
        raw: dict[str, Any] = yaml.load(handle) or {}

    for key in ('inst_id', 'label', 'images'):
        if key not in raw:
            raise ValueError(f'config {path} is missing required key {key!r}')

    param_fields = {f.name for f in dataclasses.fields(MeasureParams)}
    raw_params = raw.get('params') or {}
    unknown = set(raw_params) - param_fields
    if unknown:
        raise ValueError(f'config {path} has unknown params: {sorted(unknown)}')
    if 'radial_powers' in raw_params:
        raw_params = {**raw_params, 'radial_powers': tuple(raw_params['radial_powers'])}
    params = MeasureParams(**raw_params)

    return AnalysisConfig(
        inst_id=str(raw['inst_id']),
        label=str(raw['label']),
        camera=raw.get('camera'),
        images=[str(u) for u in raw['images']],
        params=params,
    )
