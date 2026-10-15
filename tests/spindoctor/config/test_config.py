"""Tests for the load-time validation in :class:`spindoctor.config.Config`."""

from __future__ import annotations

from pathlib import Path

import pytest

from spindoctor.config import Config


@pytest.mark.parametrize(
    'value', ['.nan', '-1.0', '2.0', 'abc'], ids=['nan', 'below_zero', 'above_one', 'text']
)
def test_malformed_blob_max_overflow_fraction_is_refused(tmp_path: Path, value: str) -> None:
    """An overflow limit that is not a finite fraction fails the load, naming the key."""
    override = tmp_path / 'override.yaml'
    override.write_text(f'bodies:\n  blob_max_overflow_fraction: {value}\n', encoding='utf-8')
    config = Config()
    with pytest.raises(ValueError, match=r'bodies\.blob_max_overflow_fraction must be finite'):
        config.update_config(override)
