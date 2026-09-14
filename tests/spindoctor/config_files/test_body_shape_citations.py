"""Validation tests for ``config_220_body_shape.yaml`` citations.

Per Part 0 §74 of the design plan: every body in the body-shape catalog
must carry a complete ``_sources`` mapping; every non-``null`` numeric
field must have a non-empty citation; ``PLACEHOLDER`` is allowed only
when paired with a ``null`` value; the strings ``TODO`` / ``FIXME`` /
``XXX`` are forbidden anywhere in the citation text.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from ruamel.yaml import YAML

_BODY_SHAPE_PATH = (
    Path(__file__).resolve().parents[3]
    / 'src'
    / 'spindoctor'
    / 'config_files'
    / 'config_220_body_shape.yaml'
)
"""Path to the shipped body-shape catalog YAML."""


_FIELDS_REQUIRING_CITATION = (
    'radii_km',
    'ellipsoid_rms_residual_km',
    'crater_scale_km',
    'albedo_mean',
    'albedo_variation',
    'shape_class_hint',
)
"""Fields whose ``_sources`` entry must be present and non-empty."""


_FORBIDDEN_TOKENS = ('TODO', 'FIXME', 'XXX')
"""Substrings that must not appear in any citation (case-insensitive)."""


def _load_raw_yaml() -> dict[str, Any]:
    """Return the body-shape catalog parsed *with* ``_sources`` preserved."""
    yaml = YAML(typ='safe')
    with open(_BODY_SHAPE_PATH, encoding='utf-8') as f:
        loaded = yaml.load(f) or {}
    body_shape = loaded.get('body_shape')
    assert isinstance(body_shape, dict), 'config_220_body_shape.yaml must contain body_shape:'
    return body_shape


def test_every_body_has_sources_mapping() -> None:
    """Every body declares a ``_sources`` mapping covering every required field."""
    bodies = _load_raw_yaml()
    for body_name, entry in bodies.items():
        assert isinstance(entry, dict), f'{body_name} entry must be a mapping'
        sources = entry.get('_sources')
        assert isinstance(sources, dict), (
            f'{body_name} missing _sources mapping (every body requires one)'
        )


def test_every_required_field_has_non_empty_citation() -> None:
    """Each required field is paired with a non-empty citation string."""
    bodies = _load_raw_yaml()
    for body_name, entry in bodies.items():
        sources = entry['_sources']
        for field in _FIELDS_REQUIRING_CITATION:
            citation = sources.get(field)
            assert isinstance(citation, str), f'{body_name}._sources[{field!r}] must be a string'
            assert citation.strip(), f'{body_name}._sources[{field!r}] must be non-empty'


def test_no_forbidden_tokens_in_citations() -> None:
    """No citation string contains TODO / FIXME / XXX (case-insensitive)."""
    bodies = _load_raw_yaml()
    for body_name, entry in bodies.items():
        sources = entry['_sources']
        for field, citation in sources.items():
            text = str(citation).upper()
            for token in _FORBIDDEN_TOKENS:
                assert token not in text, (
                    f'{body_name}._sources[{field!r}] contains forbidden token '
                    f'{token!r}: {citation!r}'
                )


def test_placeholder_only_paired_with_null_value() -> None:
    """``PLACEHOLDER`` citations are allowed only when the field's value is null."""
    bodies = _load_raw_yaml()
    for body_name, entry in bodies.items():
        sources = entry['_sources']
        for field in _FIELDS_REQUIRING_CITATION:
            if field not in sources:
                continue
            citation = sources[field]
            value = entry.get(field)
            if 'PLACEHOLDER' in str(citation).upper():
                assert value is None, (
                    f'{body_name}.{field}: PLACEHOLDER citation is only allowed '
                    f'when the value is null; got {value!r}'
                )


def test_loader_strips_underscore_keys() -> None:
    """The runtime ``Config`` loader drops every ``_sources`` block."""
    pytest.importorskip('spindoctor.config')
    from spindoctor.config import Config

    config = Config()
    config.read_config()
    body_shape = config.body_shape
    assert 'MIMAS' in body_shape
    assert '_sources' not in body_shape['MIMAS']
