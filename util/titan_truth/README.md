# Titan planted-truth campaign

A randomized simulated-scene campaign that measures how accurately
`TitanHazeNav` recovers a known pointing offset from a hazy body, how honest
the uncertainty it reports alongside that offset is, and which of its gates
refuse which frames.

The standing per-technique sweeps under `tests/integration/sim_sweeps/`
(`titan_offset_fine`, `titan_offset_wide`) vary one parameter at a time from
one base scene and feed the simulator report's response curves. This campaign
is the other half: many scenes, every parameter drawn at once, aimed at the
statistics — percentiles, z-scores, gate rates — that a stated accuracy bound
and a calibrated sigma rest on.

## Running it

```bash
source /seti/newnav/setup.sh
python util/titan_truth/collect.py --per-family 100 --workers 10 \
    --out _work/titan_truth/rows.jsonl
python util/titan_truth/analyze.py _work/titan_truth/rows.jsonl
```

600 scenes take about a minute on ten workers. `--seed` draws a different
campaign; the default seed is recorded in every run's manifest line, and the
per-scene seed derives from it, so any single scene can be regenerated and
inspected on its own.

## What the scenes contain

Every scene renders Titan at its published radius, at a randomly drawn range
(28-78 px apparent solid radius), phase (10-140 deg), sun direction, read
noise (1.5-16 DN), and planted pointing offset (0-40 px, uniform in
direction). The families differ only in what else is on the frame:

| family | what it adds |
|---|---|
| `clean` | nothing — the estimator's mirror-symmetry assumption holds |
| `clouds` | 1-4 Gaussian clouds on the disc |
| `asymmetry` | tilted haze axis, hemispheric falloff and brightness differences, an interior ramp, a limb sharpness gradient |
| `stars` | 3-12 stars, straddling the mask magnitude limit |
| `artifacts` | cosmic rays and hot pixels at deliberate STRESS incidence |
| `artifacts_nominal` | the same contamination class at the instrument's shipped, realism-matched incidence |
| `combined` | all of the above together |

`clean` is the family the published accuracy bound is stated on; the others
measure how far each broken assumption moves it. Everything the families add
is a truth-side scene key, invisible to the navigator through `nav_params`.

The two artifact families are a matched pair and must not be confused. The
stress one draws hot-pixel incidence an order of magnitude wide and forces a
nonzero cosmic-ray rate the realism recalibration deliberately retained at
zero; it bounds the regime. `artifacts_nominal` overrides nothing and is the
family an operational prediction reads from. `_artifact_blocks` states the
provenance of every stress range.

The sun direction is drawn uniformly over the circle, and the body's in-plane
roll follows it at `rotation_z = illumination_angle - 90`. That roll is what
puts the line of the hemispheric split the structure keys use along the sun
axis, which is what makes `ns_asymmetry_amplitude` the affine control the
design intends: the mirror about the sun axis has to map one hemisphere onto
the other for a pure brightness scaling to be the invariance being tested. The
control is affine only poleward of the simulator's blend band, 10 degrees of
latitude either side of the split. Inside the band the scaling fades
continuously to half its value on the mirror line, where the two halves meet,
so mirror pairs there differ by less than the full factor. No continuous blend
can avoid that, and a switch would draw a seam. The band lies along the sun
axis, over the sunward limb the technique scores on. Poleward of the band the
control is a pure scaling only where the scaled half stays below saturation:
the simulator clips the signal at 1, so a positive amplitude flattens the
bright crescent. Over 26 of the 40 clean scenes that navigate both with and
without
it, an amplitude of +0.35 moves the recovered cross-track position by 0.063
pixels on average with a switch at the equator and 0.065 pixels with the band,
about four times what -0.35 costs with the band. On the structureless spheres
every family renders, an in-plane roll changes no rendered pixel, so carrying
it costs the campaign no angular coverage.

## What the report says

- **Recovery error percentiles per family**, resolved onto the technique's own
  symmetry axis. Cross-track (the mirror-correlation axis) and along-track
  (the limb-arc axis) have different bounds and must not be averaged together.
- **Commit rate by phase bin.** A haze disc near full illumination is close to
  rotationally symmetric, so the mirror-correlation scan grows side lobes and
  the competing-peak gate refuses the frame. The table is where that shows up.
- **Reported-sigma calibration.** The z-score `error / sigma` per axis, both
  over every committed row (what the ensemble consumes) and over the rows whose
  sigma was not clamped to its configured per-axis floor (what
  `cross_sigma_scale` / `along_sigma_scale` actually control). When the
  clamped fraction is large, the floor and not the scale is what sets the
  reported uncertainty, and the printed z-versus-scale curve says whether any
  scale can reach the target band at all.
- **The no-confident-wrong check**: the error distribution restricted to
  results the technique called confident. Provisional while the confidence
  spec carries placeholder anchors.

## Reading a failure

A `FAIL` on a clean-scene bound is a real regression in the estimator. A
`FAIL` on a z-score band with a high clamped fraction is a statement about the
sigma floor, not about the scale — check the z-versus-scale curve printed
beneath it before changing anything. Gate counts moving without an error
percentile moving is a gate-threshold question, not an accuracy one.
