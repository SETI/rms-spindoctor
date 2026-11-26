"""The real moon discs of the photometric-law study, measured on each calibrated frame.

:func:`collect_real_discs` looks at every moon in a calibrated real frame, whatever the
frame's scene class.  A moon is measured when its shape is regular, no atmosphere hides
its surface, its disc is not clipped at the camera's full scale, at least a quarter of
its lit disc lies in the frame, and enough lit pixels lie clear of the limb and the
terminator.  Every moon in view that fails one of these is listed with its reason.

The frame's lighting and viewing angles come from its SPICE geometry, computed at each
pixel's center after moving the frame by its verified offset, so they line up with the
image to a fraction of a pixel.  The geometry traces each body alone, so a moon's disc
leaves out the pixels a nearer moon or the planet covers, the pixels behind the planet's
main rings (:data:`~tests.integration.sim_realism_body_facts.OCCULTING_RINGS`), and the
pixels in the planet's shadow, each widened by the disc clearance for the blur.  A ring
other than the main rings, such as a faint dusty ring, is not taken as hiding a moon.

The disc's pixels are selected and fitted with :mod:`spindoctor.sim.realism.disc_profile`:
Minnaert's exponent ``k``, the lunar-Lambert blend ``L``, and the brightness profile
along the photometric equator.  The selection is kept with the disc, so each twin is
fitted over the same pixels.  A moon whose surface holds two terrains of different
albedo (:data:`~tests.integration.sim_realism_body_facts.TWO_TONE_TERRAINS`) is measured
one terrain at a time, over the cap drawn inside each, and the rest of its disc is left
out.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import scipy.ndimage

from spindoctor.config import DEFAULT_CONFIG
from spindoctor.nav_model.body_shape import load_body_shape
from spindoctor.nav_model.nav_model_body import TITAN_BODY_NAME, bodies_in_extfov
from spindoctor.sim.realism.disc_profile import (
    DISC_CLEARANCE_PX,
    MIN_FIT_PIXELS,
    equator_profile,
    fit_lunar_lambert,
    fit_minnaert,
    select_disc_pixels,
)
from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType
from tests.integration.sim_realism_body import EDGE_CLEARANCE_PX, MIN_CLIPPED_FRACTION
from tests.integration.sim_realism_body_facts import (
    OCCULTING_RINGS,
    TWO_TONE_TERRAINS,
    TerrainCap,
    unit_vectors,
)
from tests.integration.sim_realism_laws import (
    ExcludedDisc,
    LawStudyInputs,
    RealDisc,
    StudyFrame,
    crop_box,
    crop_image,
    decimate,
)

__all__ = [
    'MIN_DIAMETER_PX',
    'MIN_LIT_FRACTION',
    'box_in_fov_at_offset',
    'cap_lit_area_px',
    'collect_real_discs',
    'disc_lit_area_px',
    'light_direction_deg',
    'lit_share',
]

MIN_DIAMETER_PX = 10.0
"""A moon in view smaller than this is listed by its size and not measured."""

MIN_LIT_FRACTION = 0.25
"""The least share of a disc's lit area, or a terrain's, that must lie in the frame.

A strip along the limb alone cannot tell the laws apart.
"""

# The step of the latitude-longitude grid a terrain cap's lit area is summed over.
_CAP_GRID_STEP_DEG = 0.5


def collect_real_discs(
    inputs: LawStudyInputs,
    obs: Any,
    frame: StudyFrame,
    *,
    backplane: Any,
    clipped: NDArrayBoolType | None,
) -> None:
    """Measure every moon in one calibrated real frame for the study.

    A moon counts as in view when its predicted box, moved by the frame's verified
    offset, overlaps the frame.

    Parameters:
        inputs: Where the measured discs and the moons left out are added.
        obs: The real observation; its data are I/F.
        frame: The frame's facts.
        backplane: The frame's geometry at each image pixel's center, moved by the
            verified offset (:func:`~tests.integration.sim_realism_body.offset_backplane`).
        clipped: Which of the image's pixels read the camera's full scale, or None.
    """
    planet = getattr(obs, 'closest_planet', None)
    data = np.asarray(obs.data, dtype=np.float64)
    full_scale = (
        np.zeros(data.shape, dtype=bool) if clipped is None else np.asarray(clipped, dtype=bool)
    )
    usable = np.isfinite(data) & (data > 0.0) & ~full_scale
    in_view = [
        (body, entry)
        for body, entry in bodies_in_extfov(obs, config=DEFAULT_CONFIG)
        if box_in_fov_at_offset(obs, entry, frame.offset_vu)
    ]
    for body, entry in in_view:
        if body == planet:
            continue
        diameter = float(max(entry['u_pixel_size'], entry['v_pixel_size']))
        if diameter < 1.0:
            reason: str | None = 'under 1 px across'
        elif diameter < MIN_DIAMETER_PX:
            reason = f'about {diameter:.0f} px across'
        else:
            reason = _exclusion_by_kind(body)
        if reason is not None:
            inputs.excluded.append(ExcludedDisc(frame.image_id, body, reason))
            continue
        _measure_moon(
            inputs,
            backplane,
            frame,
            body,
            diameter_px=diameter,
            center_uv=(float(entry['center_uv'][0]), float(entry['center_uv'][1])),
            data=data,
            usable=usable,
            full_scale=full_scale,
            hidden=_hidden_pixels(backplane, body, entry, in_view, planet=planet),
        )


def box_in_fov_at_offset(
    obs: Any, entry: Mapping[str, Any], offset_vu: tuple[float, float]
) -> bool:
    """Whether a body's predicted box, moved by the verified offset, overlaps the frame.

    Parameters:
        obs: The real observation.
        entry: The body's inventory entry, at its predicted position.
        offset_vu: The frame's verified offset ``(dv, du)``.

    Returns:
        True when the moved box overlaps the frame's nominal field of view.
    """
    moved = dict(entry)
    for axis, shift in (('v', offset_vu[0]), ('u', offset_vu[1])):
        moved[f'{axis}_min_unclipped'] = entry[f'{axis}_min_unclipped'] + shift
        moved[f'{axis}_max_unclipped'] = entry[f'{axis}_max_unclipped'] + shift
    return bool(obs.inventory_body_in_fov(moved))


def _boxes_overlap(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    """Whether two inventory entries' pixel boxes overlap."""
    return bool(
        a['u_min'] <= b['u_max']
        and b['u_min'] <= a['u_max']
        and a['v_min'] <= b['v_max']
        and b['v_min'] <= a['v_max']
    )


def _hidden_pixels(
    backplane: Any,
    body: str,
    entry: Mapping[str, Any],
    in_view: Sequence[tuple[str, Mapping[str, Any]]],
    *,
    planet: str | None,
) -> NDArrayBoolType:
    """The pixels of a moon's disc that something else hides, widened by the blur.

    The frame's geometry traces each body alone, so the moon's backplanes still mark as
    its own the pixels where a nearer body passes in front of it, where the planet's
    main rings lie in front of it, and where it lies in the planet's shadow.  Those
    pixels are left out of its disc, with a margin of the disc clearance for the blur.

    Parameters:
        backplane: The frame's offset backplane.
        body: The moon.
        entry: The moon's inventory entry.
        in_view: Every body in view with its inventory entry.
        planet: The frame's planet, or None.

    Returns:
        A mask in the image's shape.
    """
    shape = np.shape(backplane.where_intercepted(body).vals)
    cover = np.zeros(shape, dtype=bool)
    for other, other_entry in in_view:
        if other_entry is entry or other_entry['range'] >= entry['range']:
            continue
        if not _boxes_overlap(entry, other_entry):
            continue
        cover |= np.asarray(backplane.where_intercepted(other).vals, dtype=bool)
    if planet is not None:
        ring = OCCULTING_RINGS.get(planet)
        if ring is not None:
            cover |= np.asarray(backplane.where_in_back(body, ring).vals, dtype=bool)
        cover |= np.asarray(backplane.where_inside_shadow(body, planet).vals, dtype=bool)
    if cover.any():
        cover = scipy.ndimage.binary_dilation(cover, iterations=math.ceil(DISC_CLEARANCE_PX))
    return np.asarray(cover, dtype=bool)


def _exclusion_by_kind(body: str) -> str | None:
    """Why a moon is left out whatever the frame, or None."""
    if body == TITAN_BODY_NAME:
        return 'its atmosphere hides the surface'
    # Each twin is a sphere, so only a moon known to be regular can be paired with one.
    shape = load_body_shape(body).shape_class_hint
    if shape == 'unknown':
        return 'its shape is not known, so a spherical twin may not match it'
    if shape != 'regular':
        return f'{shape.replace("_", " ")} shape, which a spherical twin cannot match'
    return None


def _plane(values: Any, fill: float) -> NDArrayFloatType:
    """A backplane's values as a float array, with its masked pixels set to ``fill``."""
    return np.asarray(np.ma.filled(values.mvals, fill), dtype=np.float64)


def _scalar(values: Any) -> float:
    """The one value a backplane of an observation-wide quantity holds."""
    return float(np.asarray(values.vals, dtype=np.float64).ravel()[0])


def _terrain_caps(
    backplane: Any, body: str
) -> list[tuple[TerrainCap | None, NDArrayBoolType | None]]:
    """The parts a moon is measured in: its whole disc, or the cap of each terrain."""
    caps = TWO_TONE_TERRAINS.get(body)
    if caps is None:
        return [(None, None)]
    lat_deg = np.degrees(_plane(backplane.latitude(body, lat_type='centric'), np.nan))
    west_deg = np.degrees(
        _plane(backplane.longitude(body, reference='iau', direction='west'), np.nan)
    )
    return [(cap, cap.contains(lat_deg, west_deg)) for cap in caps]


def disc_lit_area_px(radius_px: float, phase_deg: float) -> float:
    """The projected area of a sphere's lit disc, in pixels.

    Parameters:
        radius_px: The sphere's apparent radius.
        phase_deg: The phase angle in degrees.

    Returns:
        ``pi * r**2 * (1 + cos(phase)) / 2``.
    """
    return math.pi * radius_px**2 * (1.0 + math.cos(math.radians(phase_deg))) / 2.0


def cap_lit_area_px(
    cap: TerrainCap,
    radius_px: float,
    *,
    sub_observer_deg: tuple[float, float],
    sub_solar_deg: tuple[float, float],
) -> float:
    """The projected area of the lit, visible part of a terrain cap, in pixels.

    The cap is summed over a grid of latitude and longitude on a sphere of the moon's
    apparent radius: each lit, visible cell adds its area times the cosine of its
    emission angle.

    Parameters:
        cap: The terrain cap.
        radius_px: The moon's apparent radius.
        sub_observer_deg: The sub-observer point's planetocentric latitude and west
            longitude, in degrees.
        sub_solar_deg: The sub-solar point's, likewise.

    Returns:
        The area the cap's lit part would cover if the whole disc lay in the frame.
    """
    step = _CAP_GRID_STEP_DEG
    lat, lon = np.meshgrid(
        np.arange(-90.0 + step / 2.0, 90.0, step),
        np.arange(step / 2.0, 360.0, step),
        indexing='ij',
    )
    normals = unit_vectors(lat, lon)
    observer = unit_vectors(np.asarray(sub_observer_deg[0]), np.asarray(sub_observer_deg[1]))
    sun = unit_vectors(np.asarray(sub_solar_deg[0]), np.asarray(sub_solar_deg[1]))
    mu = np.sum(normals * observer, axis=-1)
    mu0 = np.sum(normals * sun, axis=-1)
    counted = cap.contains(lat, lon) & (mu > 0.0) & (mu0 > 0.0)
    cell_sr = np.cos(np.radians(lat)) * math.radians(step) ** 2
    return float(radius_px**2 * np.sum(np.where(counted, mu * cell_sr, 0.0)))


def lit_share(lit_in_frame: NDArrayBoolType, expected_px: float) -> float:
    """The share of a lit area that lies in the frame.

    Parameters:
        lit_in_frame: The lit pixels of the disc, or of the terrain, in the image.
        expected_px: The area they would cover if the whole disc lay in the frame.

    Returns:
        The pixel count over the expected area, at most 1.
    """
    return min(1.0, float(np.count_nonzero(lit_in_frame)) / max(expected_px, 1.0))


def _measure_moon(
    inputs: LawStudyInputs,
    backplane: Any,
    frame: StudyFrame,
    body: str,
    *,
    diameter_px: float,
    center_uv: tuple[float, float],
    data: NDArrayFloatType,
    usable: NDArrayBoolType,
    full_scale: NDArrayBoolType,
    hidden: NDArrayBoolType,
) -> None:
    """Measure one moon, or each of its terrains, and add it to the inputs.

    The test of a clipped disc reads only the interior pixels nothing hides, so a
    clipped body in front of the moon does not count against it.
    """
    on_body = np.asarray(backplane.where_intercepted(body).vals, dtype=bool)
    if not on_body.any():
        inputs.excluded.append(ExcludedDisc(frame.image_id, body, 'its disc is off the image'))
        return
    interior = scipy.ndimage.binary_erosion(on_body, iterations=EDGE_CLEARANCE_PX) & ~hidden
    if interior.any() and float(np.mean(full_scale[interior])) >= MIN_CLIPPED_FRACTION:
        inputs.excluded.append(
            ExcludedDisc(frame.image_id, body, "its disc is clipped at the camera's full scale")
        )
        return
    mu0 = np.cos(_plane(backplane.incidence_angle(body), np.pi))
    mu = np.cos(_plane(backplane.emission_angle(body), np.pi))
    phase_deg_map = np.degrees(_plane(backplane.phase_angle(body), np.nan))
    radius = diameter_px / 2.0
    offset = frame.offset_vu
    center_vu = (center_uv[1] + offset[0], center_uv[0] + offset[1])
    lit = on_body & (mu0 > 0.0)
    disc_selection = select_disc_pixels(on_body, mu0, mu, usable=usable & ~hidden)
    bounds = crop_box(on_body)
    for cap, cap_mask in _terrain_caps(backplane, body):
        label = body if cap is None else f'{body} ({cap.name})'
        selected = disc_selection if cap_mask is None else disc_selection & cap_mask
        lit_part = lit if cap_mask is None else lit & cap_mask
        if not lit_part.any():
            inputs.excluded.append(
                ExcludedDisc(frame.image_id, label, 'none of its lit area lies in the frame')
            )
            continue
        phase_deg = float(np.nanmedian(phase_deg_map[selected if selected.any() else lit_part]))
        if cap is None:
            expected = disc_lit_area_px(radius, phase_deg)
        else:
            expected = cap_lit_area_px(
                cap,
                radius,
                sub_observer_deg=(
                    math.degrees(_scalar(backplane.sub_observer_latitude(body))),
                    math.degrees(_scalar(backplane.sub_observer_longitude(body))),
                ),
                sub_solar_deg=(
                    math.degrees(_scalar(backplane.sub_solar_latitude(body))),
                    math.degrees(_scalar(backplane.sub_solar_longitude(body))),
                ),
            )
        share = lit_share(lit_part, expected)
        if share < MIN_LIT_FRACTION:
            inputs.excluded.append(
                ExcludedDisc(
                    frame.image_id,
                    label,
                    f'only {100.0 * share:.0f}% of its lit area lies in the frame',
                )
            )
            continue
        kept = decimate(selected)
        minnaert = (
            fit_minnaert(data[kept], mu0[kept], mu[kept])
            if int(selected.sum()) >= MIN_FIT_PIXELS
            else None
        )
        if minnaert is None:
            inputs.excluded.append(
                ExcludedDisc(
                    frame.image_id,
                    label,
                    f'only {int(selected.sum())} lit pixels lie {DISC_CLEARANCE_PX:g} px '
                    'clear of the limb and the terminator',
                )
            )
            continue
        inputs.discs.append(
            RealDisc(
                frame=frame,
                body=body,
                part=None if cap is None else cap.name,
                phase_deg=phase_deg,
                diameter_px=diameter_px,
                center_vu=center_vu,
                illumination_deg=light_direction_deg(on_body, mu0, center_vu, radius),
                intensity=data[kept],
                mu0=mu0[kept],
                mu=mu[kept],
                minnaert=minnaert,
                lunar_lambert=fit_lunar_lambert(data[kept], mu0[kept], mu[kept]),
                profile=equator_profile(data[kept], mu0[kept], mu[kept], math.radians(phase_deg)),
                crop=crop_image(data, bounds),
                crop_bounds=bounds,
                selection=kept,
            )
        )


def light_direction_deg(
    on_body: NDArrayBoolType,
    mu0: NDArrayFloatType,
    center_vu: tuple[float, float],
    radius_px: float,
) -> float:
    """The in-image light direction that best explains a disc's incidence cosines.

    Each lit pixel's cosine is the dot product of its sphere normal with the direction
    to the Sun, so a least-squares fit over the lit pixels gives that direction.

    Parameters:
        on_body: Which pixels the moon covers.
        mu0: The cosine of the incidence angle at each pixel.
        center_vu: The moon's center, as a pixel-corner position.
        radius_px: The moon's apparent radius.

    Returns:
        The direction as a scene's ``illumination_angle``: 0 when the light comes from
        the top of the image, 90 from the right.
    """
    rows, cols = np.nonzero(on_body & (mu0 > 0.05))
    normal_v = (rows + 0.5 - center_vu[0]) / radius_px
    normal_u = (cols + 0.5 - center_vu[1]) / radius_px
    normal_z = np.sqrt(np.clip(1.0 - normal_v**2 - normal_u**2, 0.0, None))
    design = np.column_stack([normal_v, normal_u, normal_z])
    sun, *_ = np.linalg.lstsq(design, mu0[rows, cols], rcond=None)
    return float(math.degrees(math.atan2(sun[1], -sun[0])) % 360.0)
