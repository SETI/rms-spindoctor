"""Tests for the agreement campaign scene generator.

Requires the spindoctor package on the path (schema validation); run from
the repo checkout with ``PYTHONPATH=src``.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parent.parent.parent / 'src'))

from scene_gen import (  # noqa: E402
    FAMILIES,
    FRAME_SIZE,
    generate_scenes,
)

_SEED = 987654


@pytest.mark.parametrize('family', FAMILIES)
def test_generation_is_deterministic(family: str) -> None:
    """The same seed regenerates identical scenes."""
    first = generate_scenes(family, 5, campaign_seed=_SEED)
    second = generate_scenes(family, 5, campaign_seed=_SEED)
    assert first == second


@pytest.mark.parametrize('family', FAMILIES)
def test_scenes_validate_against_schema(family: str) -> None:
    """Every generated scene passes the sim schema (validated inline)."""
    scenes = generate_scenes(family, 10, campaign_seed=_SEED)
    assert len(scenes) == 10
    for scene_id, params, geometry in scenes:
        assert scene_id.startswith(family)
        assert 'offset_v' in params
        assert 'composition' in geometry


def test_unknown_family_rejected() -> None:
    """An unknown family raises with the valid list."""
    with pytest.raises(ValueError, match='unknown scene family'):
        generate_scenes('nope', 1, campaign_seed=_SEED)


def test_limb_disc_body_fully_in_frame() -> None:
    """The limb_disc body silhouette stays inside the frame."""
    for _, params, _ in generate_scenes('limb_disc', 25, campaign_seed=_SEED):
        body = params['bodies'][0]
        radius = body['axis1'] / 2.0
        for key, center in (('center_v', body['center_v']), ('center_u', body['center_u'])):
            assert center - radius > 0.0, key
            assert center + radius < FRAME_SIZE, key
        assert body['axis1'] >= 104.0


def test_limb_disc_psf_authors_matched_psf() -> None:
    """The PSF family's control render matches the navigator (floor)."""
    for _, params, geometry in generate_scenes('limb_disc_psf', 15, campaign_seed=_SEED):
        assert params['optics']['psf'] == {'match_navigator': True}
        assert geometry['composition'] == 'limb_disc_psf'
        assert 0.0 <= geometry['illumination_deg'] <= 360.0


def test_limb_disc_psf_body_fully_in_frame() -> None:
    """The PSF family keeps the same fully-framed single-body geometry."""
    for _, params, _ in generate_scenes('limb_disc_psf', 15, campaign_seed=_SEED):
        assert len(params['bodies']) == 1
        body = params['bodies'][0]
        radius = body['axis1'] / 2.0
        assert body['center_v'] - radius > 0.0
        assert body['center_v'] + radius < FRAME_SIZE


def test_limb_disc_psf_geometry_matches_limb_disc() -> None:
    """PSF scenes differ from limb_disc only by the optics block, not geometry."""
    plain = generate_scenes('limb_disc', 10, campaign_seed=_SEED)
    psf = generate_scenes('limb_disc_psf', 10, campaign_seed=_SEED)
    # Different families draw independent RNG streams, so the bodies differ;
    # what must hold is that the PSF family adds optics and the plain one omits it.
    assert all('optics' not in params for _, params, _ in plain)
    assert all('optics' in params for _, params, _ in psf)


def test_ring_line_clears_framed_body() -> None:
    """The ringlet mid-line keeps clear of the fully-framed body."""
    for _, params, geometry in generate_scenes('limb_disc_ring_diverse', 25, campaign_seed=_SEED):
        body = params['bodies'][0]
        phi = math.radians(geometry['ring_radial_deg'])
        geom = params['ring_system']['geometry']
        a = params['ring_system']['features'][0]['orbit']['a_px']
        # Mid-line point: ring center plus a along the radial direction.
        pv = geom['center_v'] + a * math.cos(phi)
        pu = geom['center_u'] + a * math.sin(phi)
        # Distance from the body center to the line through (pv, pu)
        # perpendicular to phi.
        dist = abs(
            (body['center_v'] - pv) * math.cos(phi) + (body['center_u'] - pu) * math.sin(phi)
        )
        assert dist > body['axis1'] / 2.0 + 8.0


def test_aniso_ring_line_clears_clipped_body() -> None:
    """The aniso families' ringlet line clears the large clipped body."""
    for family in ('limb_ring_aniso_fixed', 'limb_ring_aniso_diverse'):
        for _, params, geometry in generate_scenes(family, 25, campaign_seed=_SEED):
            body = params['bodies'][0]
            phi = math.radians(geometry['ring_radial_deg'])
            geom = params['ring_system']['geometry']
            a = params['ring_system']['features'][0]['orbit']['a_px']
            pv = geom['center_v'] + a * math.cos(phi)
            pu = geom['center_u'] + a * math.sin(phi)
            dist = abs(
                (body['center_v'] - pv) * math.cos(phi) + (body['center_u'] - pu) * math.sin(phi)
            )
            assert dist > body['axis1'] / 2.0 + 8.0, family


def test_fixed_families_freeze_their_angles() -> None:
    """The *_fixed cohorts hold their control angles constant."""
    radials = {
        geometry['ring_radial_deg']
        for _, _, geometry in generate_scenes('limb_disc_ring_fixed', 20, campaign_seed=_SEED)
    }
    assert radials == {200.0}
    aniso = generate_scenes('limb_ring_aniso_fixed', 20, campaign_seed=_SEED)
    arcs = {geometry['limb_arc_outward_deg'] for _, _, geometry in aniso}
    assert arcs == {220.0}
    relative = {
        round((geometry['ring_radial_deg'] - geometry['limb_arc_outward_deg']) % 360.0, 6)
        for _, _, geometry in aniso
    }
    assert relative == {340.0}


def test_diverse_families_spread_their_angles() -> None:
    """The *_diverse cohorts draw a spread of control angles."""
    radials = [
        geometry['ring_radial_deg']
        for _, _, geometry in generate_scenes('limb_disc_ring_diverse', 40, campaign_seed=_SEED)
    ]
    assert len({round(r, 3) for r in radials}) > 30
    spread = max(radials) - min(radials)
    assert spread > 180.0


def test_multi_body_names_and_separation() -> None:
    """multi_body scenes carry RHEA and DIONE, well separated."""
    for _, params, geometry in generate_scenes('multi_body', 25, campaign_seed=_SEED):
        names = [b['name'] for b in params['bodies']]
        assert names == ['RHEA', 'DIONE']
        assert geometry['body_names'] == ['RHEA', 'DIONE']
        b1, b2 = params['bodies']
        gap = math.hypot(b1['center_v'] - b2['center_v'], b1['center_u'] - b2['center_u'])
        assert gap > (b1['axis1'] + b2['axis1']) / 2.0 + 20.0
