# Body-limb navigation bias: diagnosis

Date: 2026-07-14
Scope: measurement and attribution only. No navigation code was changed.
Refs: issues #150 (measured bias) and #128 (proposed redesign).

## Summary

`BodyLimbNav` fits a body's predicted silhouette polyline to the image edge
distance transform. It carries a systematic sub-pixel error that does not
average out across frames, so it sets the accuracy floor for limb-navigated
images.

This diagnosis reproduces the bias in the simulator, where the spacecraft
position, body ephemeris, and pointing are exact by construction, so any
residual limb-fit error is purely algorithmic. Findings:

- The genuine algorithmic limb bias is roughly **0.05 to 0.14 px** in the
  regime where the limb technique is actually used (moderate phase, a
  well-resolved disc). It is **directional**: it points from the sunlit limb
  toward the body interior, i.e. opposite the illumination direction.
- The bias **direction tracks the illumination direction** and is largely
  independent of body size. This is the signature of a **limb-darkening /
  photometric roll-off** effect, not a fixed frame convention and not pure
  interpolation. It is the dominant component.
- A smaller, secondary **sub-pixel interpolation ripple** (about 0.05 px,
  roughly one-pixel period in the sub-pixel offset phase) rides on top of the
  directional bias.
- The **simulator's own body renderer is clean**: its intensity-weighted
  centroid matches the requested geometric center to better than 2e-5 px, so
  the sim can be trusted as ground truth. The bias is therefore in the
  navigation code path, not the fixture.
- On real frames the **limb-minus-star gap is 0.5 to 1.8 px**, far larger than
  the ~0.1 px algorithmic bias. Almost all of the real-frame gap is
  spacecraft-position / body-ephemeris error; the algorithmic limb bias
  explains only about 0.1 px of it.

## How the two error sources were separated

The operator's constraint is that an apparent limb error on a real frame could
be either a genuine limb-fit bias or a wrong spacecraft position / moon-orbit
ephemeris. Only the simulator can isolate the algorithmic part, because there
the geometry is exact.

- **Sim scenes** (planted truth): the scene renders a body at
  `center + planted_offset`; the simulated body NavModel predicts the unshifted
  geometry, so the navigator should recover `planted_offset` exactly. Any
  residual is the algorithmic / optical-model bias with no geometry component.
- **Real frames** (star ground truth): on a frame that also carries navigable
  stars, the star techniques give an independent, usually more precise offset.
  The limb-minus-star gap mixes the algorithmic limb bias with any
  position/ephemeris error.

The sim isolates the algorithmic component (~0.1 px). The real-frame gap
(0.5 to 1.8 px) minus that algorithmic component bounds the ephemeris /
position error, which dominates on the real frames measured here.

## Harness

- `tests/integration/limb_bias.py` -- importable measurement functions
  (signed per-axis error against sim planted truth; real-frame limb-vs-star
  gap; the renderer geometry validation).
- `tests/integration/limb_bias_runner.py` -- `python -m` driver that writes the
  CSV tables in this directory.
- `tests/integration/test_limb_bias.py` -- renderer-validation and smoke tests.
- `make_plots.py` -- renders the two PNGs in this directory from the CSVs.

All sim measurements are noise-free so the reported error is the deterministic
algorithmic bias, not a per-frame noise draw. Probe defaults: 160 px diameter,
30 deg phase, 25 deg illumination, planted offset (0.3, 0.3), 260 px frame.

## Block 1: simulator renderer validation (constraint 3)

Rendered a fully-lit (phase 0) sphere directly through the renderer at a range
of sub-pixel centers and measured its intensity-weighted centroid. A phase-0
sphere is radially symmetric, so its brightness centroid must equal its
geometric center; any offset would be a renderer-baked positional bias. The
renderer places pixel index `i` at coordinate `i + 0.5`, so a requested center
`c` lands the geometric center at pixel index `c - 0.5`.

Result: worst-case centroid error **1.4e-5 px** across all sub-pixel centers
tested (`renderer_validation.csv`). The renderer is clean to far better than
0.1 px; the sim is a trustworthy ground truth.

Separately, along a scan through a phase-0 limb, the brightness **gradient
ridge** (the steepest-slope point that the edge distance transform localizes)
sits about **0.5 px inside** the geometric limb. This is the mechanism, not a
renderer bug: the limb-darkened brightness already rolls off approaching the
geometric limb, so the steepest slope is inboard of the true silhouette
boundary. When only one side of the disc is sunlit, that inboard displacement
becomes a net translational bias pointing away from the sunlit limb.

## Block 2: sim planted-truth bias sweeps

Signed per-axis limb-fit error against planted truth (`sim_sweeps.csv`).

### Illumination direction (the cause discriminator)

The bias vector rotates with the illumination direction and points from the
sunlit limb toward the body interior. See `bias_vs_illumination.png`.

| illum (deg) | err_v (px) | err_u (px) | mag (px) |
|-------------|-----------|-----------|---------|
| 0   | +0.034 | +0.046 | 0.057 |
| 45  | +0.007 | -0.114 | 0.114 |
| 90  | -0.016 | -0.104 | 0.105 |
| 135 | -0.070 | -0.070 | 0.100 |
| 180 | -0.104 | -0.016 | 0.105 |
| 225 | -0.114 | +0.007 | 0.114 |
| 270 | +0.046 | +0.034 | 0.057 |
| 315 | +0.096 | +0.095 | 0.136 |

At illumination 90 deg (light from +u, sunlit limb on the +u side) the bias
points -u, back toward the body center; at 180 deg (light from +v) it points
-v. The direction tracking illumination is the defining signature of a
photometric / limb-darkening edge-model mismatch. A fixed component of about
0.04 px (tied to the fixed 0.3, 0.3 sub-pixel offset) rides on top, which is
why the magnitudes are not perfectly constant around the circle.

### Sub-pixel offset phase (interpolation ripple)

Sweeping the planted sub-pixel offset shows a roughly one-pixel-period ripple
of about 0.05 px on top of the directional bias. See `bias_vs_subpixel.png`.

| offset_u (px) | err_v (px) | err_u (px) | mag (px) |
|---------------|-----------|-----------|---------|
| 0.0 | +0.141 | -0.002 | 0.141 |
| 0.2 | +0.076 | -0.005 | 0.076 |
| 0.4 | +0.016 | -0.043 | 0.045 |
| 0.6 | +0.081 | +0.046 | 0.093 |
| 0.8 | +0.136 | +0.055 | 0.147 |

The error is non-zero even at integer offset (no interpolation of the observed
image), so the directional component is not an interpolation artifact; the
ripple is the added interpolation / edge-localization phase dependence.

### Phase angle

| phase (deg) | mag (px) | note |
|-------------|---------|------|
| 0   | 0.48 | soft all-around limb, poorly conditioned |
| 5   | 0.24 | poorly conditioned |
| 10  | 2.71 | mis-convergence (see below) |
| 20  | 0.09 | stable |
| 30  | 0.06 | stable |
| 40  | 0.07 | stable |
| 50  | 0.08 | stable |
| 60  | 0.08 | stable |

In the operating regime (20 to 60 deg phase) the bias is a stable ~0.06 to
0.09 px. Below ~15 deg the whole silhouette is a soft, symmetric limb-darkened
roll-off with weak gradient, the fit is poorly conditioned, and it can
mis-converge -- the 2.71 px error at 10 deg is a real mis-convergence that the
spurious gate did not catch. This is a separate robustness issue from the
systematic bias.

### Body diameter (also the resolution axis)

For an analytic sim body, apparent size and resolution (limb curvature per
pixel) are the same knob.

| diameter (px) | mag (px) |
|---------------|---------|
| 110 | 0.035 |
| 140 | 0.064 |
| 170 | 0.070 |
| 200 | 0.098 |
| 230 | 0.084 |

The bias is roughly constant to mildly increasing with size (about 0.03 to
0.10 px). It does not fall off as 1/radius, which argues against a pure
curvature effect and for a fixed directional edge inset that a longer arc pins
more tightly.

## Block 3: real-frame limb-vs-star gap

Navigated the operator-curated `stars_plus_body` frames; recorded the signed
gap between the `BodyLimbNav` offset and the most precise available
star-technique offset on the same frame (`real_limb_vs_star.csv`).

| image | limb (v, u) | star (v, u) | gap mag (px) |
|-------|-------------|-------------|-------------|
| N1488823805 (Dione, phase 50) | (-0.93, 4.99) | (-1.42, 4.73) | 0.55 |
| N1686349893 | (5.85, -31.25) | (4.65, -32.53) | 1.76 |
| N1806609736 | (1.50, 12.83) | (1.47, 11.40) | 1.43 |

(Three of six frames produced both a limb and a star offset; the others
yielded no navigable limb.)

The real-frame gaps (0.5 to 1.8 px) are far larger than the ~0.1 px
sim-isolated algorithmic bias. Interpreted through the separation above: the
algorithmic limb bias contributes only about 0.1 px, and the remaining
0.4 to 1.7 px is spacecraft-position / body-ephemeris error (plus the star
techniques' own ~0.1 px scatter -- two star techniques on the Dione frame
disagreed by about 0.13 px). The star channel is the more precise reference on
these frames, but the limb-vs-star gap on real frames is geometry-dominated and
cannot by itself measure the algorithmic bias -- which is exactly why the sim
isolation was required.

## Attribution

1. Magnitude and direction of the genuine algorithmic bias: **0.05 to 0.14 px**,
   directional, pointing from the sunlit limb toward the body interior.
2. What it varies with: strongly with **illumination direction** (the bias
   direction tracks it); with **sub-pixel offset phase** (a ~0.05 px ripple);
   weakly with **phase angle** in the operating range and unstable below
   ~15 deg; roughly flat to mildly rising with **body size**.
3. Sim renderer: **clean** to < 2e-5 px centroid error.
4. Real-frame limb-vs-star gap: **0.5 to 1.8 px**, of which the algorithmic
   limb bias explains ~0.1 px and the remainder is position / ephemeris error.

The evidence points at the **limb-darkening / photometric roll-off model** as
the dominant cause: the edge distance transform localizes the steepest-gradient
ridge of a photometrically rolled-off limb, which sits inside the geometric
silhouette by an amount that depends on which limb is sunlit, producing a
directional translational bias.

## Recommendation for the #128 redesign, ranked by evidence

1. **Fit a photometric limb, not a geometric edge (strongest evidence).**
   Predict the expected limb brightness profile -- a limb-darkened disc
   convolved with the instrument PSF -- and recover position by matching that
   profile to the image, instead of aligning a geometric silhouette polyline to
   the max-gradient ridge. This removes the illumination-tracking directional
   bias, which is the dominant component. Equivalently, apply an
   illumination-dependent sub-pixel correction that moves the predicted limb
   inward to where the gradient ridge of the modeled profile actually falls.

2. **Reduce the sub-pixel interpolation ripple (medium evidence).** The
   gradient-ridge sub-pixel localization shows a ~0.05 px, roughly one-pixel
   period ripple with offset phase. A higher-order sub-pixel edge estimator
   (parabolic or Gaussian peak fit on the gradient profile, or a matched-filter
   localization) would cut this second-order term.

3. **Gate or constrain low-phase limb fits (robustness, weaker evidence).**
   Below ~15 deg phase the limb is a soft symmetric roll-off and the fit can
   mis-converge (a 2.71 px jump was seen at 10 deg). Consider a phase floor for
   limb navigation, or a stronger prior / constraint there, and tighten the
   spurious gate so a multi-pixel mis-convergence is caught.

4. **Audit the pixel-centre convention across model outputs (hygiene).** The
   simulated body's `BODY_DISC` predicted-centre metadata is recorded at
   `center` while the renderer places the geometric center at `center - 0.5`.
   The limb path reads the rendered mask directly and is not affected, but the
   half-pixel convention should be standardized across every model output as
   part of the redesign so a future consumer does not inherit the discrepancy.
