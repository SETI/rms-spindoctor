"""Tests for ``spindoctor.ui.library_entry`` — the YAML helper behind the dialog's
"Save as Library Entry..." button.

These tests do not need PyQt; they exercise the pure-Python helpers in
isolation so the (heavier) dialog tests stay focused on UI wiring.
"""

from __future__ import annotations

import math
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from tests.integration.sidecar import SidecarValidationError, load_sidecar

from spindoctor.ui.library_entry import (
    LibraryEntryDraft,
    build_sidecar_yaml,
    compute_image_url,
    infer_obs_metadata,
)


def _fake_obs(
    *,
    cls_name: str = 'ObsCassiniISS',
    abspath: Path | None = None,
    detector: str | None = 'NAC',
    filter1: str | None = 'CL1',
    filter2: str | None = 'CL2',
    texp: float | None = 0.46,
    midtime: float | None = None,
) -> Any:
    """Build a tiny class with the named class name and the given attributes."""
    cls = type(
        cls_name,
        (object,),
        {
            'abspath': abspath,
            'detector': detector,
            'filter1': filter1,
            'filter2': filter2,
            'texp': texp,
            'midtime': midtime,
        },
    )
    return cls()


def test_infer_obs_metadata_reads_cassini_iss_fields(tmp_path: Path) -> None:
    """A Cassini-ISS-shaped obs yields the right mission / camera / filter combo."""
    img_path = tmp_path / 'W1521598221_1_CALIB.IMG'
    img_path.write_bytes(b'')
    draft = infer_obs_metadata(_fake_obs(abspath=img_path, texp=0.46))
    assert draft.image_id == 'W1521598221_1_CALIB'
    assert draft.mission == 'COISS'
    assert draft.camera == 'NAC'
    assert draft.filter_combo == 'CL1+CL2'  # sorted, '+'-joined
    assert draft.exposure_time_sec == pytest.approx(0.46)


def test_infer_obs_metadata_drops_non_positive_texp(tmp_path: Path) -> None:
    """A non-positive or non-finite ``obs.texp`` yields ``None`` rather than a bad value."""
    img_path = tmp_path / 'X.IMG'
    img_path.write_bytes(b'')
    for bad in (0.0, -1.0, float('nan'), 'huh'):
        draft = infer_obs_metadata(_fake_obs(abspath=img_path, texp=bad))  # type: ignore[arg-type]
        assert draft.exposure_time_sec is None


def test_infer_obs_metadata_missing_texp_attribute(tmp_path: Path) -> None:
    """``obs`` without a ``texp`` attribute drops cleanly to ``None``."""
    img_path = tmp_path / 'X.IMG'
    img_path.write_bytes(b'')
    obs = _fake_obs(abspath=img_path)
    delattr(type(obs), 'texp')
    draft = infer_obs_metadata(obs)
    assert draft.exposure_time_sec is None


def test_infer_obs_metadata_extracts_image_datetime_utc(tmp_path: Path) -> None:
    """``obs.midtime`` (ET seconds) is converted to a UTC ISO 8601 string.

    The exact formatted string is timekernel-dependent, so only the
    coarse ISO 8601 shape is asserted (year-month-dayT hour:min:sec).
    """
    img_path = tmp_path / 'X.IMG'
    img_path.write_bytes(b'')
    # ET=0 maps to 2000-01-01 (J2000) plus a few minutes of offset.
    draft = infer_obs_metadata(_fake_obs(abspath=img_path, midtime=0.0))
    assert draft.image_datetime_utc is not None
    assert draft.image_datetime_utc.startswith('2000-01-01T')


def test_infer_obs_metadata_missing_midtime_attribute(tmp_path: Path) -> None:
    """``obs`` without a ``midtime`` attribute drops cleanly to ``None``."""
    img_path = tmp_path / 'X.IMG'
    img_path.write_bytes(b'')
    obs = _fake_obs(abspath=img_path, midtime=None)
    draft = infer_obs_metadata(obs)
    assert draft.image_datetime_utc is None


def test_infer_obs_metadata_returns_blanks_for_unknown_obs() -> None:
    """An unrecognized obs class defaults to empty mission / camera fields."""
    obs = _fake_obs(cls_name='ObsMystery', detector=None, filter1=None, filter2=None)
    draft = infer_obs_metadata(obs)
    assert draft.mission == ''
    assert draft.camera == ''
    assert draft.filter_combo == ''


def test_infer_obs_metadata_canonicalizes_filter_order(tmp_path: Path) -> None:
    """``filter_combo`` is sorted lexically so ``CL+IR`` and ``IR+CL`` are identical."""
    img_path = tmp_path / 'X.IMG'
    img_path.write_bytes(b'')
    a = infer_obs_metadata(_fake_obs(abspath=img_path, filter1='IR', filter2='CL'))
    b = infer_obs_metadata(_fake_obs(abspath=img_path, filter1='CL', filter2='IR'))
    assert a.filter_combo == b.filter_combo == 'CL+IR'


def test_compute_image_url_rewrites_to_pds3_when_under_holdings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A path under ``PDS3_HOLDINGS_DIR`` is rewritten with the ``pds3://`` prefix."""
    monkeypatch.setenv('PDS3_HOLDINGS_DIR', '/mnt/ganymede/PDS/holdings')
    obs = _fake_obs(
        abspath=Path('/mnt/ganymede/PDS/holdings/calibrated/COISS_2xxx/x.IMG'),
    )
    assert compute_image_url(obs) == 'pds3://calibrated/COISS_2xxx/x.IMG'


def test_compute_image_url_tolerates_trailing_slash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A trailing slash on ``PDS3_HOLDINGS_DIR`` does not double-up the separator."""
    monkeypatch.setenv('PDS3_HOLDINGS_DIR', '/mnt/ganymede/PDS/holdings/')
    obs = _fake_obs(abspath=Path('/mnt/ganymede/PDS/holdings/volumes/X.IMG'))
    assert compute_image_url(obs) == 'pds3://volumes/X.IMG'


def test_compute_image_url_falls_through_when_outside_holdings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A path that is not under the holdings root is returned verbatim."""
    monkeypatch.setenv('PDS3_HOLDINGS_DIR', '/mnt/ganymede/PDS/holdings')
    obs = _fake_obs(abspath=Path('/tmp/somewhere/else/X.IMG'))
    assert compute_image_url(obs) == '/tmp/somewhere/else/X.IMG'


def test_compute_image_url_falls_through_when_holdings_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No env / no config root => raw path passes through."""
    monkeypatch.delenv('PDS3_HOLDINGS_DIR', raising=False)
    # Drop the global config's pds3_holdings_root if present so the
    # fallback path does not silently route through it.
    config_stub = type('C', (), {'environment': type('E', (), {'pds3_holdings_root': None})()})()
    obs = _fake_obs(abspath=Path('/tmp/x.IMG'))
    assert compute_image_url(obs, config_stub) == '/tmp/x.IMG'


def test_compute_image_url_uses_config_when_env_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the env var is unset, ``config.environment.pds3_holdings_root`` wins."""
    monkeypatch.delenv('PDS3_HOLDINGS_DIR', raising=False)
    config_stub = type(
        'C',
        (),
        {'environment': type('E', (), {'pds3_holdings_root': '/srv/pds_holdings'})()},
    )()
    obs = _fake_obs(abspath=Path('/srv/pds_holdings/calibrated/COISS/y.IMG'))
    assert compute_image_url(obs, config_stub) == 'pds3://calibrated/COISS/y.IMG'


def test_compute_image_url_returns_blank_when_obs_has_no_path() -> None:
    """An obs without ``abspath`` yields an empty string."""
    obs = _fake_obs(abspath=None)
    assert compute_image_url(obs) == ''


def test_build_sidecar_yaml_round_trips_through_validator(tmp_path: Path) -> None:
    """A sidecar with TODO placeholders fails validation but with a clear error.

    The operator workflow is: save -> edit TODOs -> commit.  This test
    exercises the *post-edit* state by replacing every placeholder with
    a valid value.
    """
    draft = LibraryEntryDraft(
        image_id='W1521598221_1_CALIB',
        mission='COISS',
        camera='NAC',
        filter_combo='CL1+CL2',
        exposure_time_sec=0.46,
        image_datetime_utc='2006-04-28T12:34:56.789Z',
    )
    yaml_text = build_sidecar_yaml(
        draft=draft,
        image_url='pds3://volumes/COISS_2xxx/COISS_2021/.../W1521598221_1_CALIB.IMG',
        offset_dv_px=12.345,
        offset_du_px=-6.789,
        ui_version='0.1.dev0',
        operator='rfrench',
        today=date(2026, 4, 28),
    )
    edited = (
        yaml_text.replace('TODO_REPLACE_PRIMARY_CLASS', 'body_mostly_offscreen')
        .replace(
            'TODO_REPLACE_TECHNIQUE',
            'BodyLimbNav',
        )
        .replace(
            'TODO: describe the scene and any caveats.',
            'Limb fit verified by overlay.',
        )
    )
    p = tmp_path / 'W1521598221_1_CALIB.yaml'
    p.write_text(edited)
    sidecar = load_sidecar(p)
    assert sidecar.image_id == 'W1521598221_1_CALIB'
    assert sidecar.mission == 'COISS'
    assert sidecar.camera == 'NAC'
    assert sidecar.filter_combo == 'CL1+CL2'
    assert sidecar.scene_tags == ('body_mostly_offscreen',)
    assert sidecar.ground_truth.offset_dv_px == pytest.approx(12.345)
    assert sidecar.ground_truth.offset_du_px == pytest.approx(-6.789)
    assert sidecar.ground_truth.operator == 'rfrench'
    assert sidecar.ground_truth.verified_date == date(2026, 4, 28)
    assert sidecar.expected.primary_technique == 'BodyLimbNav'
    assert sidecar.exposure_time_sec == pytest.approx(0.46)
    assert sidecar.image_datetime_utc == '2006-04-28T12:34:56.789Z'


def test_build_sidecar_yaml_unedited_fails_validation(tmp_path: Path) -> None:
    """The unedited template trips the validator with a useful message.

    This is the safety net: an operator who forgets to edit TODOs gets a
    loud error from CI rather than a silently-broken library entry.
    """
    draft = LibraryEntryDraft(
        image_id='X',
        mission='COISS',
        camera='NAC',
        filter_combo='CL+CL',
        exposure_time_sec=None,
        image_datetime_utc=None,
    )
    yaml_text = build_sidecar_yaml(
        draft=draft,
        image_url='pds3://x/X.IMG',
        offset_dv_px=0.0,
        offset_du_px=0.0,
        ui_version='0.0.0',
        today=date(2026, 4, 28),
    )
    p = tmp_path / 'X.yaml'
    p.write_text(yaml_text)
    with pytest.raises(
        SidecarValidationError, match=r'TODO_REPLACE_PRIMARY_CLASS|primary scene_tag'
    ):
        load_sidecar(p)


def test_build_sidecar_yaml_constraint_round_trips_through_validator(tmp_path: Path) -> None:
    """A constrained (rank-1) save emits a constraint block that validates.

    ``offset_along_normal_px`` is computed from the same 2-D offset the
    YAML carries, so the validator's on-the-line consistency check passes
    by construction, and the ``expected`` block pins ``rank_1_only``.
    """
    draft = LibraryEntryDraft(
        image_id='N1863267799_1_CALIB',
        mission='COISS',
        camera='NAC',
        filter_combo='CL1+CL2',
        exposure_time_sec=0.68,
        image_datetime_utc='2016-12-18T01:23:45.678Z',
    )
    yaml_text = build_sidecar_yaml(
        draft=draft,
        image_url='pds3://volumes/COISS_2xxx/COISS_2116/.../N1863267799_1_CALIB.IMG',
        offset_dv_px=47.2772,
        offset_du_px=-73.4648,
        ui_version='0.1.dev0',
        operator='rfrench',
        today=date(2026, 7, 10),
        constraint_normal_angle_deg=45.0,
    )
    edited = (
        yaml_text.replace('TODO_REPLACE_PRIMARY_CLASS', 'ring_only_flat')
        .replace('TODO_REPLACE_TECHNIQUE', 'RingEdgeNav')
        .replace(
            'TODO: describe the scene and any caveats.',
            'Straight Keeler-gap edges; normal component only.',
        )
    )
    p = tmp_path / 'N1863267799_1_CALIB.yaml'
    p.write_text(edited)
    sidecar = load_sidecar(p)
    constraint = sidecar.ground_truth.constraint
    assert constraint is not None
    assert constraint.type == 'rank_1'
    assert constraint.normal_angle_deg == pytest.approx(45.0)
    expected_along = 47.2772 * math.cos(math.radians(45.0)) - 73.4648 * math.sin(math.radians(45.0))
    assert constraint.offset_along_normal_px == pytest.approx(expected_along, abs=1e-3)
    assert sidecar.expected.status_reason == 'rank_1_only'


def test_build_sidecar_yaml_without_constraint_has_no_block() -> None:
    """An unconstrained save emits neither constraint nor status_reason."""
    draft = LibraryEntryDraft(
        image_id='X',
        mission='COISS',
        camera='NAC',
        filter_combo='CL+CL',
        exposure_time_sec=None,
        image_datetime_utc=None,
    )
    yaml_text = build_sidecar_yaml(
        draft=draft,
        image_url='pds3://x/X.IMG',
        offset_dv_px=0.0,
        offset_du_px=0.0,
        ui_version='0.0.0',
        today=date(2026, 7, 10),
    )
    assert 'constraint:' not in yaml_text
    assert 'status_reason' not in yaml_text
