"""Self-tests for the ``tests.shims`` package.

These verify that the shims expose the methods and shapes the
navigation pipeline expects, independent of any consumer.  They also
exercise the convenience factories.
"""

from __future__ import annotations

import math
from typing import Any, cast

import numpy as np
import polymath
import pytest

from spindoctor.nav_model.stars import catalog as nav_catalog
from tests.shims import (
    BodyBackplaneData,
    FakeBackplane,
    FakeObs,
    FakeStarCatalog,
    RingBackplaneData,
    install_fake_catalogs,
    make_star,
    plant_circular_body,
)


def test_plant_circular_body_paints_disc_of_correct_radius() -> None:
    """A 5-px-radius circle paints exactly the pixels with centre distance <= radius."""
    shape = (40, 40)
    centre_vu = (20.0, 20.0)
    radius_px = 5.0
    data = plant_circular_body(
        shape=shape, centre_vu=centre_vu, radius_px=radius_px, resolution_km_px=2.0
    )
    vv, uu = np.meshgrid(
        np.arange(shape[0], dtype=np.float64),
        np.arange(shape[1], dtype=np.float64),
        indexing='ij',
    )
    expected_pixels = int(
        np.count_nonzero((vv - centre_vu[0]) ** 2 + (uu - centre_vu[1]) ** 2 <= radius_px**2)
    )
    assert int(np.count_nonzero(data.body_mask)) == expected_pixels


def test_plant_circular_body_incidence_increases_with_radius() -> None:
    """The synthetic incidence ramps from 0 at centre to pi/2 at limb."""
    data = plant_circular_body(
        shape=(40, 40), centre_vu=(20.0, 20.0), radius_px=10.0, resolution_km_px=1.0
    )
    centre_inc = float(data.incidence_rad[20, 20])
    near_limb = float(data.incidence_rad[20, 30])
    assert centre_inc == pytest.approx(0.0, abs=0.05)
    assert near_limb == pytest.approx(math.pi / 2.0, abs=0.05)


def test_fake_backplane_returns_real_polymath_scalar() -> None:
    """Backplane methods return real ``polymath.Scalar`` instances."""
    body = plant_circular_body(shape=(20, 20), centre_vu=(10.0, 10.0), radius_px=5.0)
    bp = FakeBackplane(per_body={'MIMAS': body})
    incidence = bp.incidence_angle('MIMAS')
    assert isinstance(incidence, polymath.Scalar)


def test_fake_backplane_body_methods_round_trip() -> None:
    """Body backplane methods return the configured per-pixel arrays."""
    body = plant_circular_body(
        shape=(20, 20),
        centre_vu=(10.0, 10.0),
        radius_px=5.0,
        sub_solar_lon_deg=30.0,
        phase_angle_deg=45.0,
    )
    bp = FakeBackplane(per_body={'MIMAS': body})
    incidence = bp.incidence_angle('MIMAS')
    assert incidence.mvals.shape == (20, 20)
    assert bp.sub_solar_longitude('MIMAS').vals == pytest.approx(math.radians(30.0))
    assert bp.center_phase_angle('MIMAS').vals == pytest.approx(math.radians(45.0))
    intercepted = bp.where_intercepted('MIMAS')
    assert bool(intercepted.any()) is True


def test_fake_backplane_body_lambert_default_is_cosine_of_incidence() -> None:
    """Without an explicit ``lambert`` array the shim returns ``cos(incidence)``.

    On-body pixels carry ``cos(incidence)`` clipped to ``[0, inf)``; off-body
    pixels are zero.
    """
    body = plant_circular_body(shape=(10, 10), centre_vu=(5.0, 5.0), radius_px=3.0)
    bp = FakeBackplane(per_body={'MIMAS': body})
    lambert = bp.lambert_law('MIMAS')
    incidence = bp.incidence_angle('MIMAS')
    expected = np.where(
        body.body_mask, np.clip(np.cos(incidence.mvals.filled(0.0)), 0.0, None), 0.0
    )
    np.testing.assert_array_equal(lambert.mvals.filled(0.0), expected)
    assert (lambert.mvals.filled(0.0)[~body.body_mask] == 0.0).all()


def test_fake_backplane_unknown_body_raises_lookup_error() -> None:
    """Looking up a body that wasn't configured raises ``LookupError``."""
    bp = FakeBackplane(per_body={})
    with pytest.raises(LookupError, match="no entry for body 'TITAN'"):
        bp.incidence_angle('TITAN')


def test_fake_backplane_unknown_ring_raises_lookup_error() -> None:
    """Looking up a ring target that wasn't configured raises ``LookupError``."""
    bp = FakeBackplane(per_ring={})
    with pytest.raises(LookupError, match="no entry for ring target 'jupiter:ring'"):
        bp.ring_radius('jupiter:ring')


def test_fake_backplane_where_in_front_answers_for_a_hidden_body() -> None:
    """``occluder_mask_for_body`` hides a body behind a body, not behind a ring.

    It calls ``where_in_front(sibling_name, body_name)``, so the hidden target
    is a body name.  A stand-in that looks every hidden target up among the
    rings raises ``LookupError`` for that whole caller.
    """
    body = plant_circular_body(shape=(12, 9), centre_vu=(6.0, 4.0), radius_px=3.0)
    bp = FakeBackplane(per_body={'MIMAS': body})
    hidden = bp.where_in_front('ENCELADUS', 'MIMAS')
    assert hidden.vals.shape == (12, 9)
    assert not bool(np.asarray(hidden.vals).any())


def test_fake_backplane_where_in_front_answers_for_a_hidden_ring() -> None:
    """The rings model hides a ring behind the planet, so ring targets still work."""
    ring = RingBackplaneData(
        ring_radius_km=np.linspace(70_000.0, 140_000.0, 42).reshape(6, 7),
        ring_mask=np.ones((6, 7), dtype=bool),
    )
    bp = FakeBackplane(per_ring={'saturn:ring': ring})
    hidden = bp.where_in_front('saturn', 'saturn:ring')
    assert hidden.vals.shape == (6, 7)
    assert not bool(np.asarray(hidden.vals).any())


def test_fake_backplane_where_in_front_refuses_an_unregistered_target() -> None:
    """A target in neither registry is an incomplete scene, and says so."""
    bp = FakeBackplane()
    with pytest.raises(LookupError, match="no entry for ring target 'jupiter:ring'"):
        bp.where_in_front('jupiter', 'jupiter:ring')


def test_fake_backplane_ring_methods_round_trip() -> None:
    """Ring backplane methods return the configured per-pixel arrays."""
    radius = np.linspace(70_000.0, 140_000.0, 100).reshape(10, 10)
    ring = RingBackplaneData(
        ring_radius_km=radius,
        ring_mask=np.ones(radius.shape, dtype=bool),
        default_radial_resolution_km_px=200.0,
    )
    bp = FakeBackplane(per_ring={'saturn:ring': ring})
    radii_scalar = bp.ring_radius('saturn:ring')
    assert not radii_scalar.is_all_masked()
    assert radii_scalar.min(builtins=True) == pytest.approx(70_000.0)
    assert radii_scalar.max(builtins=True) == pytest.approx(140_000.0)
    res = bp.ring_radial_resolution('saturn:ring')
    assert np.all(np.asarray(res.vals) == 200.0)


def test_fake_backplane_ring_radius_carries_border_atop_key() -> None:
    """``ring_radius`` exposes the ``key`` the production code reads back."""
    ring = RingBackplaneData(
        ring_radius_km=np.zeros((2, 2)),
        ring_mask=np.ones((2, 2), dtype=bool),
    )
    bp = FakeBackplane(per_ring={'saturn:ring': ring})
    radii = bp.ring_radius('saturn:ring')
    assert radii.key == ('ring_radius', 'saturn:ring')


def test_fake_backplane_names_the_key_of_a_ring_radius_array() -> None:
    """``standardize_backplane_key`` answers for an array it handed out."""
    ring = RingBackplaneData(
        ring_radius_km=np.zeros((2, 2)),
        ring_mask=np.ones((2, 2), dtype=bool),
    )
    bp = FakeBackplane(per_ring={'saturn:ring': ring})
    radii = bp.ring_radius('saturn:ring')
    assert bp.standardize_backplane_key(radii) == ('ring_radius', 'saturn:ring')


def test_fake_backplane_names_the_key_without_the_attribute() -> None:
    """An array carrying no ``key`` is found in the registry by identity."""
    ring = RingBackplaneData(
        ring_radius_km=np.zeros((2, 2)),
        ring_mask=np.ones((2, 2), dtype=bool),
    )
    bp = FakeBackplane(per_ring={'saturn:ring': ring})
    radii = bp.ring_radius('saturn:ring')
    del radii.key
    assert bp.standardize_backplane_key(radii) == ('ring_radius', 'saturn:ring')


def test_fake_backplane_refuses_to_name_an_unregistered_array() -> None:
    """An array this backplane never handed out has no key here."""
    ring = RingBackplaneData(
        ring_radius_km=np.zeros((2, 2)),
        ring_mask=np.ones((2, 2), dtype=bool),
    )
    bp = FakeBackplane(per_ring={'saturn:ring': ring})
    with pytest.raises(ValueError) as exc_info:
        bp.standardize_backplane_key(polymath.Scalar(np.zeros((2, 2))))
    assert 'illegal backplane key type' in str(exc_info.value)


def test_fake_backplane_border_atop_uses_default_threshold() -> None:
    """``border_atop`` returns pixels whose ring radius is within 0.5 km of ``a``."""
    radius = np.array([[100_000.0, 100_000.4], [200_000.0, 100_000.8]])
    ring = RingBackplaneData(ring_radius_km=radius, ring_mask=np.ones(radius.shape, dtype=bool))
    bp = FakeBackplane(per_ring={'saturn:ring': ring})
    radii = bp.ring_radius('saturn:ring')
    assert radii.key is not None
    edge = bp.border_atop(radii.key, 100_000.3)
    # |100_000.0 - 100_000.3| = 0.3 -> True; |100_000.4 - 100_000.3| = 0.1 -> True;
    # |200_000.0 - 100_000.3| = ~100K -> False; |100_000.8 - 100_000.3| = 0.5 -> False (strict).
    assert np.asarray(edge.vals).tolist() == [[True, True], [False, False]]


def test_fake_backplane_border_atop_rejects_wrong_key() -> None:
    """A key whose head is not ``'ring_radius'`` raises ``LookupError``."""
    ring = RingBackplaneData(
        ring_radius_km=np.zeros((2, 2)),
        ring_mask=np.ones((2, 2), dtype=bool),
    )
    bp = FakeBackplane(per_ring={'saturn:ring': ring})
    with pytest.raises(LookupError, match='expected a ring_radius key'):
        bp.border_atop(('something_else', 'saturn:ring'), 100.0)


def test_fake_obs_extdata_padded_around_data() -> None:
    """``extdata`` matches ``data`` size plus 2 * margin."""
    data = np.full((20, 30), 5.0)
    obs = FakeObs(data=data, extfov_margin_vu=(2, 4))
    assert obs.extdata.shape == (24, 38)
    # Sensor area equals the input.
    assert (obs.extdata[2:22, 4:34] == 5.0).all()
    # Extfov padding is zero.
    assert (obs.extdata[:2, :] == 0.0).all()


def test_fake_obs_extract_offset_array_without_offset() -> None:
    """``extract_offset_array`` with no offset returns the sensor-area window."""
    obs = FakeObs(data=np.zeros((6, 6)), extfov_margin_vu=(2, 2))
    array = np.arange(100, dtype=np.float64).reshape((10, 10))
    out = obs.extract_offset_array(array, (0.0, 0.0))
    assert np.array_equal(out, array[2:8, 2:8])


def test_fake_obs_extract_offset_array_shifts_by_the_offset() -> None:
    """A positive offset extracts the window that much earlier in the array."""
    obs = FakeObs(data=np.zeros((6, 6)), extfov_margin_vu=(2, 2))
    array = np.arange(100, dtype=np.float64).reshape((10, 10))
    out = obs.extract_offset_array(array, (1.0, -1.0))
    assert np.array_equal(out, array[1:7, 3:9])


def test_fake_obs_extract_offset_array_zero_fills_beyond_the_extfov() -> None:
    """An offset past the margin zero-fills the part outside the extfov."""
    obs = FakeObs(data=np.zeros((6, 6)), extfov_margin_vu=(2, 2))
    array = np.ones((10, 10), dtype=np.float64)
    out = obs.extract_offset_array(array, (5.0, 0.0))
    assert float(out[0, 0]) == 0.0


def test_fake_obs_extract_offset_array_rejects_a_wrong_shape() -> None:
    """An array that is not extfov-shaped is a programming error."""
    obs = FakeObs(data=np.zeros((6, 6)), extfov_margin_vu=(2, 2))
    with pytest.raises(ValueError, match='must equal extdata shape'):
        obs.extract_offset_array(np.zeros((6, 6)), (0.0, 0.0))


def test_fake_obs_extract_offset_array_rejects_a_three_dimensional_array() -> None:
    """A 3-D array is rejected here exactly as the real method rejects it."""
    obs = FakeObs(data=np.zeros((6, 6)), extfov_margin_vu=(2, 2))
    with pytest.raises(ValueError, match='must equal extdata shape'):
        obs.extract_offset_array(np.zeros((10, 10, 3)), (0.0, 0.0))


def test_fake_obs_inventory_returns_only_requested_bodies() -> None:
    """``inventory`` returns only entries for the requested bodies."""
    inv = {
        'MIMAS': {
            'u_min_unclipped': 10,
            'u_max_unclipped': 30,
            'v_min_unclipped': 10,
            'v_max_unclipped': 30,
            'u_pixel_size': 20.0,
            'v_pixel_size': 20.0,
            'range': 1e5,
            'center_uv': (20.0, 20.0),
        }
    }
    obs = FakeObs(data=np.zeros((40, 40)), inventory_records=inv)
    out = obs.inventory(['MIMAS', 'TETHYS'], return_type='full')
    assert list(out.keys()) == ['MIMAS']
    assert out['MIMAS']['range'] == 1e5


def test_fake_obs_inventory_body_in_extfov_predicate() -> None:
    """The ``inventory_body_in_extfov`` predicate matches the real obs's contract."""
    obs = FakeObs(data=np.zeros((10, 10)), extfov_margin_vu=(2, 2))
    inside = {
        'u_min_unclipped': -1,
        'u_max_unclipped': 5,
        'v_min_unclipped': -1,
        'v_max_unclipped': 5,
    }
    outside = {
        'u_min_unclipped': 100,
        'u_max_unclipped': 110,
        'v_min_unclipped': 100,
        'v_max_unclipped': 110,
    }
    assert obs.inventory_body_in_extfov(inside) is True
    assert obs.inventory_body_in_extfov(outside) is False


def test_fake_obs_uv_from_ra_and_dec_brackets_diverge_with_tfrac() -> None:
    """The shim's UV result depends on ``tfrac`` so smear bracketing works."""
    obs = FakeObs(data=np.zeros((10, 10)))
    uv0 = obs.uv_from_ra_and_dec(np.zeros(1), np.zeros(1), tfrac=0.0, apparent=True)
    uv1 = obs.uv_from_ra_and_dec(np.zeros(1), np.zeros(1), tfrac=1.0, apparent=True)
    u0, v0 = uv0.to_scalars()
    u1, v1 = uv1.to_scalars()
    assert isinstance(u0, polymath.Scalar)
    assert np.asarray(u1.vals).tolist() != np.asarray(u0.vals).tolist()
    assert np.asarray(v1.vals).tolist() != np.asarray(v0.vals).tolist()


def test_make_star_applies_kwargs() -> None:
    """``make_star`` returns a default star with overridden fields."""
    star = make_star(unique_number=42, vmag=3.0, name='Sirius')
    assert star.unique_number == 42
    assert star.vmag == 3.0
    assert star.name == 'Sirius'
    assert star.spectral_class == 'G0'  # default unchanged


def test_make_star_post_init_mirrors_ra_into_ra_pm() -> None:
    """``__post_init__`` mirrors catalog RA / DEC into the proper-motion fields."""
    star = make_star(ra=1.5, dec=-0.5)
    assert star.ra_pm == 1.5
    assert star.dec_pm == -0.5


def test_fake_star_catalog_filters_by_box() -> None:
    """``find_stars`` returns only stars inside the RA/DEC/mag box."""
    inside = make_star(unique_number=1, ra=0.05, dec=0.05, vmag=5.0)
    outside_ra = make_star(unique_number=2, ra=1.0, dec=0.05, vmag=5.0)
    outside_mag = make_star(unique_number=3, ra=0.05, dec=0.05, vmag=20.0)
    cat = FakeStarCatalog(stars=[inside, outside_ra, outside_mag])
    matches = list(
        cat.find_stars(
            ra_min=0.0,
            ra_max=0.1,
            dec_min=0.0,
            dec_max=0.1,
            vmag_min=0.0,
            vmag_max=10.0,
        )
    )
    assert [s.unique_number for s in matches] == [1]


def test_install_fake_catalogs_is_test_local(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patches installed via ``monkeypatch`` unwind on test teardown.

    Within the test, the production getters return our fakes; pytest's
    teardown then restores the originals so subsequent tests in the
    same worker see the unpatched module.  Across pytest-xdist workers
    (separate processes) the patches cannot leak by construction.
    """
    star = make_star(unique_number=7, ra=0.05, dec=0.05, vmag=5.0)
    cat_map = install_fake_catalogs(monkeypatch, ucac4=[star])
    # The catalog module is imported at the top; monkeypatch swaps
    # the attribute on that module, so dynamic attribute lookup picks
    # up the patched version.  Mypy types the getters to return their
    # real catalog classes; cast for the identity check against the
    # FakeStarCatalog instances.
    assert cast(Any, nav_catalog.get_ucac4_catalog()) is cat_map['ucac4']
    assert cast(Any, nav_catalog.get_tycho2_catalog()) is cat_map['tycho2']
    assert cast(Any, nav_catalog.get_ybsc_catalog()) is cat_map['ybsc']
    assert len(cat_map['ucac4'].stars) == 1
    assert len(cat_map['tycho2'].stars) == 0


def test_install_fake_catalogs_does_not_persist_across_tests() -> None:
    """A test that did not call ``install_fake_catalogs`` sees the real getters.

    Verifies the per-test scoping promise: pytest's ``monkeypatch``
    fixture rolled back any swap from the previous test, so this test
    looking up the lazy getter via dynamic attribute access on the
    catalog module sees the production function again.
    """
    # The unpatched function is the one defined in
    # ``spindoctor.nav_model.stars.catalog``; a leftover lambda from an
    # earlier test would have ``__module__`` set to ``tests.shims.catalog``.
    assert nav_catalog.get_ucac4_catalog.__module__ == 'spindoctor.nav_model.stars.catalog'


def test_body_backplane_data_default_resolution_fills_array() -> None:
    """``resolution_array`` returns a constant fill when not configured."""
    data = BodyBackplaneData(
        body_mask=np.ones((4, 4), dtype=bool),
        incidence_rad=np.zeros((4, 4)),
        default_resolution_km_px=3.0,
    )
    assert (data.resolution_array() == 3.0).all()


def test_ring_backplane_data_default_distance_fills_array() -> None:
    """``distance_array`` returns a constant fill when not configured."""
    data = RingBackplaneData(
        ring_radius_km=np.zeros((4, 4)),
        ring_mask=np.ones((4, 4), dtype=bool),
        default_distance_km=2.5e9,
    )
    assert (data.distance_array() == 2.5e9).all()
