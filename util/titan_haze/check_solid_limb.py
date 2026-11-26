"""Check the rendered haze across a body's solid limb against a line-of-sight integration.

Usage::

    python util/titan_haze/check_solid_limb.py

A white Lambert sphere carries a haze, and one pixel is put just inside and then just
outside the solid limb, on the sunward and the anti-sunward side.  Beside the brightness
:func:`spindoctor.sim.render.bodies.atmosphere.apply_atmosphere` renders there, the
script prints a reference that uses none of the renderer's formulas: the single
scattering of the same exponential haze, integrated numerically along each line of
sight through the sphere's shell, with the sunlight reaching every point integrated
through the same haze and cut off where the sphere shadows it, plus the lit surface seen
through the column where the line of sight meets it::

    I/F = sum over the path of (w / 4) P beta exp(-tau_observer - tau_sun) dl
          + A mu0 exp(-tau_observer - tau_sun)        (at the surface)

The extinction falls off as ``exp(-h / H)`` with altitude ``h``, scaled so the
integrated tangent column at the surface equals the haze's surface tangent depth.  The
reference leaves out the light scattered more than once, which the renderer includes, so
the two brightness levels differ; the step across the limb, as a fraction of the
brighter side, is what they share.  The hazes are the guide's generic crescent haze (a
scale height of 8 px and a tangent depth of 3 at the surface over an 85 px body) and
Titan's haze at 650 nm at the scale of the ``titan_crescent_horns`` scene.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

from spindoctor.sim.render.bodies.atmosphere import (  # noqa: E402
    AtmosphereSpec,
    apply_atmosphere,
)
from spindoctor.support.constants import PIXEL_CENTER_TO_CORNER_PX  # noqa: E402

# The render grid, with the sphere near its center.
_SIZE = 300
_CENTER = 150.0

# How far inside and outside the limb the reference samples, and the renderer's pixel.
_REFERENCE_OFFSET_PX = 0.01
_RENDER_OFFSET_PX = 1e-6


@dataclass(frozen=True)
class Case:
    """One haze, body, and phase to check.

    Parameters:
        label: A short name for the printout.
        scale_height_px: The haze scale height ``H`` in pixels.
        tau_ref: The tangent optical depth at ``ref_altitude_px``.
        ref_altitude_px: The altitude of ``tau_ref`` above the surface, in pixels.
        albedo: The single-scattering albedo.
        g: The first Henyey-Greenstein term's asymmetry parameter.
        g2: The second term's asymmetry parameter.
        g2_fraction: The share of the scattered light in the second term.
        radius_px: The sphere's radius in pixels.
        phase_deg: The phase angle in degrees; the sun lies along +u.
    """

    label: str
    scale_height_px: float
    tau_ref: float
    ref_altitude_px: float
    albedo: float
    g: float
    g2: float
    g2_fraction: float
    radius_px: float
    phase_deg: float


def _hg(g: float, cos_phase: float) -> float:
    """The Henyey-Greenstein phase function at a phase angle's cosine."""
    return (1.0 - g * g) / (1.0 + g * g + 2.0 * g * cos_phase) ** 1.5


class LineOfSightReference:
    """Single scattering integrated along lines of sight through a spherical haze shell.

    Parameters:
        case: The haze, body, and phase.
    """

    def __init__(self, case: Case) -> None:
        self.radius = case.radius_px
        self.height = case.scale_height_px
        self.albedo = case.albedo
        self.tau_surface = case.tau_ref * math.exp(case.ref_altitude_px / case.scale_height_px)
        self.top = case.ref_altitude_px + case.scale_height_px * math.log(
            max(case.tau_ref, 1e-30) / 1e-9
        )
        phase = math.radians(case.phase_deg)
        self.phase_function = (1.0 - case.g2_fraction) * _hg(
            case.g, math.cos(phase)
        ) + case.g2_fraction * _hg(case.g2, math.cos(phase))
        self.sun_u = math.sin(phase)
        self.sun_z = math.cos(phase)
        self.step = case.scale_height_px / 40.0
        x = np.arange(-self._half_chord(0.0), self._half_chord(0.0), self.step / 4.0)
        altitude = np.sqrt(self.radius**2 + x * x) - self.radius
        column = float(np.exp(-altitude / self.height).sum()) * self.step / 4.0
        self.beta0 = self.tau_surface / column

    def _half_chord(self, d: float) -> float:
        """Half the chord the top of the haze cuts on a line of sight at distance ``d``."""
        top = self.radius + self.top
        return math.sqrt(max(top * top - d * d, 0.0))

    def _beta(self, radius: np.ndarray) -> np.ndarray:
        """The extinction coefficient at each radius."""
        return self.beta0 * np.exp(-np.maximum(radius - self.radius, 0.0) / self.height)

    def _tau_sun(self, u: np.ndarray, z: np.ndarray) -> np.ndarray:
        """The optical depth from each point toward the sun; infinite in the shadow."""
        out = np.empty(u.shape)
        t = np.arange(0.0, 2.0 * (self.radius + self.top), self.step)
        for start in range(0, u.size, 256):
            pu = u[start : start + 256, None] + t[None, :] * self.sun_u
            pz = z[start : start + 256, None] + t[None, :] * self.sun_z
            radius = np.sqrt(pu * pu + pz * pz)
            shadow = (radius < self.radius - 1e-9).any(axis=1)
            column = self._beta(radius).sum(axis=1) * self.step
            out[start : start + 256] = np.where(shadow, np.inf, column)
        return out

    def brightness(self, d: float) -> float:
        """The I/F of the line of sight at signed distance ``d`` from the center along u.

        Parameters:
            d: The image-plane distance from the body center along the sun's direction,
                in pixels; negative on the anti-sunward side.

        Returns:
            The single-scattered haze light plus the lit surface seen through it.
        """
        z_top = self._half_chord(d)
        hits = abs(d) < self.radius
        z_end = math.sqrt(self.radius**2 - d * d) if hits else -z_top
        n = max(math.ceil((z_top - z_end) / self.step), 2)
        z = np.linspace(z_top, z_end, n + 1)
        dz = (z_top - z_end) / n
        beta = self._beta(np.sqrt(d * d + z * z))
        tau_observer = np.concatenate([[0.0], np.cumsum(0.5 * (beta[1:] + beta[:-1]) * dz)])
        tau_sun = self._tau_sun(np.full(z.shape, d), z)
        haze = 0.25 * self.albedo * self.phase_function * float(
            np.trapezoid(beta * np.exp(-tau_observer - tau_sun), dx=dz)
        )
        surface = 0.0
        if hits:
            mu0 = (d * self.sun_u + z_end * self.sun_z) / self.radius
            if mu0 > 0.0:
                surface = mu0 * math.exp(-tau_observer[-1] - tau_sun[-1])
        return haze + surface


def _rendered(case: Case, *, side: int, inside: bool) -> float:
    """The renderer's brightness at a pixel just inside or outside the solid limb.

    The sphere, a white Lambert surface under the haze, is moved so the pixel's center
    sits ``_RENDER_OFFSET_PX`` inside or outside its limb.
    """
    radius = case.radius_px
    pixel = (int(_CENTER), int(_CENTER + side * radius) - (1 if side < 0 else 0))
    pixel_u = pixel[1] + PIXEL_CENTER_TO_CORNER_PX - _CENTER
    target = radius - _RENDER_OFFSET_PX if inside else radius + _RENDER_OFFSET_PX
    center_u = _CENTER + pixel_u - side * math.sqrt(target**2 - PIXEL_CENTER_TO_CORNER_PX**2)
    offsets_v = np.arange(_SIZE) + (PIXEL_CENTER_TO_CORNER_PX - _CENTER)
    offsets_u = np.arange(_SIZE) + (PIXEL_CENTER_TO_CORNER_PX - center_u)
    radius2 = offsets_v[:, None] ** 2 + offsets_u[None, :] ** 2
    outline = (radius2 < radius * radius).astype(np.float64)
    phase = math.radians(case.phase_deg)
    normal_z = np.sqrt(np.maximum(radius * radius - radius2, 0.0))
    cos_incidence = (offsets_u[None, :] * math.sin(phase) + normal_z * math.cos(phase)) / radius
    layers = apply_atmosphere(
        outline * np.maximum(cos_incidence, 0.0),
        AtmosphereSpec(
            scale_height_px=case.scale_height_px,
            tau_ref=case.tau_ref,
            ref_altitude_px=case.ref_altitude_px,
            g=case.g,
            g2=case.g2,
            g2_fraction=case.g2_fraction,
            albedo=case.albedo,
        ),
        center_v=_CENTER,
        center_u=center_u,
        semi_a=radius,
        semi_b=radius,
        semi_c=radius,
        rotation_z=0.0,
        sub_observer_lat=0.0,
        illumination_angle=math.radians(90.0),
        phase_angle=phase,
        silhouette=outline,
    )
    return float(layers.disc[pixel] if inside else layers.halo.emission[pixel])


def _step(inside: float, outside: float) -> float:
    """The step from inside to outside as a fraction of the brighter side."""
    return (outside - inside) / max(inside, outside, 1e-12)


def main() -> None:
    """Print the renderer's and the reference's brightness across the solid limb."""
    doc = {'scale_height_px': 8.0, 'tau_ref': 3.0, 'ref_altitude_px': 0.0, 'albedo': 0.6}
    doc_phase = {'g': 0.75, 'g2': 0.0, 'g2_fraction': 0.0, 'radius_px': 85.0}
    titan = {
        'scale_height_px': 1.485,
        'tau_ref': 15.055,
        'ref_altitude_px': 4.621,
        'albedo': 0.917,
        'g': 0.886,
        'g2': -0.151,
        'g2_fraction': 0.099,
        'radius_px': 85.0,
    }
    cases = [
        Case('generic', phase_deg=150.0, **doc, **doc_phase),
        Case('generic', phase_deg=40.0, **doc, **doc_phase),
        Case('titan', phase_deg=155.0, **titan),
        Case('titan', phase_deg=60.0, **titan),
        Case('titan', phase_deg=40.0, **titan),
    ]
    print('haze     phase side   renderer in / out (step)      reference in / out (step)')
    for case in cases:
        reference = LineOfSightReference(case)
        for side, name in ((1, 'sun '), (-1, 'anti')):
            r_in = _rendered(case, side=side, inside=True)
            r_out = _rendered(case, side=side, inside=False)
            x_in = reference.brightness(side * (case.radius_px - _REFERENCE_OFFSET_PX))
            x_out = reference.brightness(side * (case.radius_px + _REFERENCE_OFFSET_PX))
            print(
                f'{case.label:8s} {case.phase_deg:5.0f} {name}   '
                f'{r_in:.4f} / {r_out:.4f} ({_step(r_in, r_out):+.3f})     '
                f'{x_in:.4f} / {x_out:.4f} ({_step(x_in, x_out):+.3f})',
                flush=True,
            )


if __name__ == '__main__':
    main()
