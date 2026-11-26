"""Validation tests for ``config_220_body_shape.yaml`` citations.

Per Part 0 §74 of the design plan: every body in the body-shape catalog
must carry a complete ``_sources`` mapping; every field, set or left out,
must have a non-empty citation; ``PLACEHOLDER`` is allowed only for a field
the entry leaves out; the strings ``TODO`` / ``FIXME`` / ``XXX`` are
forbidden anywhere in the citation text.  The packaged table also parses
into its type once the citations are dropped.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ruamel.yaml import YAML

from spindoctor.config import Config

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


def test_placeholder_only_for_a_field_left_out() -> None:
    """``PLACEHOLDER`` citations are allowed only for a field the entry leaves out."""
    bodies = _load_raw_yaml()
    for body_name, entry in bodies.items():
        sources = entry['_sources']
        for field in _FIELDS_REQUIRING_CITATION:
            if field not in sources:
                continue
            citation = sources[field]
            if 'PLACEHOLDER' in str(citation).upper():
                assert field not in entry, (
                    f'{body_name}.{field}: PLACEHOLDER citation is only allowed '
                    f'when the field is left out; got {entry[field]!r}'
                )


def test_packaged_body_shape_section_parses() -> None:
    """The packaged table parses, with an entry for every body in the file.

    An entry refuses a key its type does not declare, so the table parses only because
    the loader drops every ``_sources`` block first.
    """
    assert sorted(Config().body_shape) == sorted(_load_raw_yaml())
