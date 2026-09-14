"""Tests for ``spindoctor.nav_orchestrator.provenance.Provenance``."""

from pathlib import Path

import pytest

from spindoctor.config import Config
from spindoctor.nav_orchestrator.provenance import (
    Provenance,
    ProvenanceMetadata,
    collect_provenance_metadata,
)


def test_provenance_minimal_construction() -> None:
    """The minimal required fields are version + ET + ISO timestamp."""
    prov = Provenance(
        spindoctor_version='0.5.2',
        image_et=414504000.0,
        pipeline_run_iso8601='2026-04-26T12:00:00Z',
    )
    assert prov.spindoctor_version == '0.5.2'
    assert prov.image_et == 414504000.0
    assert prov.spice_kernels == ()
    assert prov.spice_kernel_count == 0
    assert prov.static_data_hashes == {}


def test_provenance_carries_kernels_and_hashes() -> None:
    """Optional kernel list and hash dict are stored verbatim."""
    prov = Provenance(
        spindoctor_version='0.5.2',
        image_et=1.0,
        pipeline_run_iso8601='2026-04-26T12:00:00Z',
        spice_kernels=('a.bsp', 'b.tpc'),
        static_data_hashes={'config_220_body_shape.yaml': 'deadbeef'},
    )
    assert prov.spice_kernels == ('a.bsp', 'b.tpc')
    assert prov.spice_kernel_count == 2
    assert prov.static_data_hashes['config_220_body_shape.yaml'] == 'deadbeef'


def test_provenance_static_data_hashes_is_immutable() -> None:
    """``static_data_hashes`` is wrapped as ``MappingProxyType``."""
    import pytest

    prov = Provenance(
        spindoctor_version='0.5.2',
        image_et=1.0,
        pipeline_run_iso8601='2026-04-26T12:00:00Z',
        static_data_hashes={'a.yaml': 'aa'},
    )
    with pytest.raises(TypeError) as exc_info:
        prov.static_data_hashes['b.yaml'] = 'bb'  # type: ignore[index]
    # ``MappingProxyType`` raises ``TypeError`` with "does not support
    # item assignment" when assignment is attempted.
    assert 'item assignment' in str(exc_info.value)


def test_collect_provenance_metadata_returns_dataclass() -> None:
    """``collect_provenance_metadata`` returns a populated dataclass."""
    from collections.abc import Mapping
    from types import MappingProxyType

    meta = collect_provenance_metadata()
    assert isinstance(meta, ProvenanceMetadata)
    # The git SHA may be ``None`` if the working tree isn't a repo, but
    # the field must exist.
    assert isinstance(meta.git_sha, str | type(None))
    assert isinstance(meta.spice_kernels, tuple)
    # ``static_data_hashes`` is exposed as ``Mapping[str, str]`` and the
    # implementation returns a ``MappingProxyType``; pin both shape and
    # contents so a future refactor that swaps the wrapping type still
    # has to satisfy the read-only-mapping contract.
    assert isinstance(meta.static_data_hashes, Mapping)
    assert isinstance(meta.static_data_hashes, MappingProxyType)
    for key, value in meta.static_data_hashes.items():
        assert isinstance(key, str)
        assert isinstance(value, str)


def test_collect_provenance_metadata_hashes_static_data_yamls() -> None:
    """The hash dict covers config_220_body_shape.yaml and the inst configs."""
    meta = collect_provenance_metadata()
    names = set(meta.static_data_hashes.keys())
    # Every shipped 4N0 instrument block plus the body-shape catalog
    # must appear; ring catalogs (3N0) are also static data.
    assert 'config_220_body_shape.yaml' in names
    assert 'config_400_inst_coiss.yaml' in names
    assert 'config_310_saturn_rings.yaml' in names
    # SHA-256 hex digest is 64 chars.
    for digest in meta.static_data_hashes.values():
        assert len(digest) == 64
        assert all(c in '0123456789abcdef' for c in digest)


def test_collect_provenance_metadata_is_byte_identical_across_calls() -> None:
    """Two consecutive calls produce identical hashes (modulo wall-clock)."""
    a = collect_provenance_metadata()
    b = collect_provenance_metadata()
    assert dict(a.static_data_hashes) == dict(b.static_data_hashes)
    assert a.spice_kernels == b.spice_kernels


def test_provenance_carries_config_and_catalog_fields() -> None:
    """The config hash, override list, and star catalogs are stored."""
    prov = Provenance(
        spindoctor_version='0.5.2',
        image_et=1.0,
        pipeline_run_iso8601='2026-04-26T12:00:00Z',
        config_hash='ab' * 32,
        config_overrides=('/z_first.yaml', '/a_second.yaml'),
        star_catalogs={'ybsc': 'gs://bucket/YBSC', 'ucac4': 'gs://bucket/UCAC4'},
    )
    assert prov.config_hash == 'ab' * 32
    # Override order is application order; it must NOT be sorted.
    assert prov.config_overrides == ('/z_first.yaml', '/a_second.yaml')
    assert dict(prov.star_catalogs) == {
        'ucac4': 'gs://bucket/UCAC4',
        'ybsc': 'gs://bucket/YBSC',
    }


def test_provenance_star_catalogs_is_immutable() -> None:
    """``star_catalogs`` is wrapped as ``MappingProxyType``."""
    prov = Provenance(
        spindoctor_version='0.5.2',
        image_et=1.0,
        pipeline_run_iso8601='2026-04-26T12:00:00Z',
        star_catalogs={'ucac4': 'gs://bucket/UCAC4'},
    )
    with pytest.raises(TypeError, match='item assignment'):
        prov.star_catalogs['ybsc'] = 'x'  # type: ignore[index]


def test_collect_provenance_metadata_includes_config_hash() -> None:
    """The metadata carries a 64-char sha256 of the resolved config."""
    meta = collect_provenance_metadata()
    assert meta.config_hash is not None
    assert len(meta.config_hash) == 64
    assert all(c in '0123456789abcdef' for c in meta.config_hash)


def test_collect_provenance_metadata_lists_configured_star_catalogs() -> None:
    """Every configured catalog name appears in the star-catalog mapping."""
    meta = collect_provenance_metadata()
    names = set(meta.star_catalogs.keys())
    assert 'ucac4' in names
    assert 'tycho2' in names
    assert 'ybsc' in names


def test_star_catalog_paths_follow_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Catalog paths mirror the env vars the catalog constructors read."""
    monkeypatch.setenv('UCAC4_PATH', 'gs://bucket/UCAC4')
    monkeypatch.setenv('YBSC_PATH', 'gs://bucket/YBSC')
    monkeypatch.setenv('SPICE_PATH', '/kernels')
    meta = collect_provenance_metadata()
    assert meta.star_catalogs['ucac4'] == 'gs://bucket/UCAC4'
    assert meta.star_catalogs['ybsc'] == 'gs://bucket/YBSC'
    assert meta.star_catalogs['tycho2'] == '/kernels/Stars'


def test_star_catalog_paths_empty_when_unresolvable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset catalog env vars record an empty path, never a fabricated one."""
    monkeypatch.delenv('UCAC4_PATH', raising=False)
    monkeypatch.delenv('YBSC_PATH', raising=False)
    monkeypatch.delenv('SPICE_PATH', raising=False)
    monkeypatch.delenv('OOPS_RESOURCES', raising=False)
    meta = collect_provenance_metadata()
    assert meta.star_catalogs['ucac4'] == ''
    assert meta.star_catalogs['ybsc'] == ''
    assert meta.star_catalogs['tycho2'] == ''


def test_resolved_config_hash_identical_for_identical_inputs() -> None:
    """Two configs loaded from the same bundled defaults hash identically."""
    a = Config()
    b = Config()
    assert a.resolved_config_hash() == b.resolved_config_hash()


def test_resolved_config_hash_changes_with_override_content(tmp_path: Path) -> None:
    """An override that changes resolved content changes the hash."""
    base = Config()
    override = tmp_path / 'override.yaml'
    override.write_text('stars:\n  max_stars: 12345\n', encoding='utf-8')
    modified = Config()
    modified.update_config(override)
    assert modified.resolved_config_hash() != base.resolved_config_hash()


def test_resolved_config_hash_ignores_override_provenance_not_content(tmp_path: Path) -> None:
    """Two override files with identical content produce identical hashes."""
    text = 'stars:\n  max_stars: 12345\n'
    first = tmp_path / 'first.yaml'
    second = tmp_path / 'second.yaml'
    first.write_text(text, encoding='utf-8')
    second.write_text(text, encoding='utf-8')
    a = Config()
    a.update_config(first)
    b = Config()
    b.update_config(second)
    assert a.resolved_config_hash() == b.resolved_config_hash()


def test_override_paths_recorded_in_application_order(tmp_path: Path) -> None:
    """User override files are recorded in the order they were applied."""
    first = tmp_path / 'z_first.yaml'
    second = tmp_path / 'a_second.yaml'
    first.write_text('stars:\n  max_stars: 111\n', encoding='utf-8')
    second.write_text('stars:\n  max_stars: 222\n', encoding='utf-8')
    config = Config()
    config.update_config(first)
    config.update_config(second)
    assert config.override_paths == (str(first), str(second))


def test_override_paths_empty_for_bundled_defaults() -> None:
    """A plain bundled-defaults load records no overrides."""
    config = Config()
    config.read_config()
    assert config.override_paths == ()


def test_collect_provenance_metadata_records_overrides(tmp_path: Path) -> None:
    """The applied override list flows into the collected metadata."""
    override = tmp_path / 'override.yaml'
    override.write_text('stars:\n  max_stars: 777\n', encoding='utf-8')
    config = Config()
    config.update_config(override)
    meta = collect_provenance_metadata(config)
    assert meta.config_overrides == (str(override),)


def test_static_data_hashes_skips_unreadable_files(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A ``read_bytes`` failure is logged at WARNING and the file is skipped.

    Provenance metadata is best-effort: a per-file I/O error must not
    abort the navigation run.
    """
    import pathlib

    from spindoctor.nav_orchestrator import provenance as provenance_mod

    real_read_bytes = pathlib.Path.read_bytes

    def fake_read_bytes(self: pathlib.Path) -> bytes:
        if self.name == 'config_400_inst_coiss.yaml':
            raise PermissionError(f'simulated permission denied for {self.name}')
        return real_read_bytes(self)

    monkeypatch.setattr(pathlib.Path, 'read_bytes', fake_read_bytes)
    # The resolver is process-memoized; clear the cache so this call recomputes
    # under the mocked read_bytes, and clear it again afterwards so the
    # mock-derived result does not leak into other tests.
    provenance_mod._resolve_static_data_hashes.cache_clear()
    try:
        hashes = provenance_mod._resolve_static_data_hashes()
        assert 'config_400_inst_coiss.yaml' not in hashes
        # Other static-data files still hash successfully.
        assert 'config_220_body_shape.yaml' in hashes
        out = capsys.readouterr().out
        assert 'config_400_inst_coiss.yaml' in out
        assert 'simulated permission denied' in out
    finally:
        provenance_mod._resolve_static_data_hashes.cache_clear()
