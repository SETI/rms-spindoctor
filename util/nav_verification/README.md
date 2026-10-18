# Checking a navigation pass against something outside it

A navigation run reports its own confidence, and that number is a statement about
the fit rather than about the answer. A technique that fits two stars tightly
reports a tight sigma whatever it is pointing at, so a pass cannot tell you which
of its confident answers are wrong. Deciding that needs evidence the pass did not
produce, and there are two kinds available.

## Layout

| Module | Role |
| --- | --- |
| `boresight.py` | Where a C-matrix points, the angle between two answers in pixels of the camera that saw them, and that difference resolved onto the camera's axes. No mission constants, no spindoctor imports. |
| `bundle_cassini_fring.py` | The published Cassini F ring bundle read as an independent pointing answer: where its per-frame supplementary files live, what they say, and the NAC plate scale. |
| `compare_pointing.py` | Per-frame comparison of a results root against that bundle. |
| `measure_core_radius.py` | Where the ring core lands in a co-rotating mosaic, which needs no external answer at all. |
| `tests/` | Unit tests on the arithmetic. No holdings and no SPICE needed. |

## Comparing against an independently navigated answer

The F ring bundle records, beside every reprojected product, the attitude that
project navigated the frame to and how it navigated it. It is an answer rather
than truth, and its own file says which kind: a frame labeled `Stars` was fixed
against a catalog and is worth believing to well under a pixel, while
`Ring and/or Satellite Models` and `Manual` are weaker, so a disagreement with one
of those is a disagreement rather than an error.

```bash
source /seti/newnav/setup.sh
python util/nav_verification/compare_pointing.py \
    --nav-results-root /data/nav-run \
    --observation ISS_006RI_LPHRLFMOV001_PRIME \
    --output /tmp/iss_006ri_verification.json
```

The report gives, per frame, the angle to the bundle's answer in pixels of that
frame (the camera's plate scale times its summing factor) beside the quantities
that might have predicted a bad one -- the reported sigma, the spread between
the techniques that formed the answer, how many star features the models
emitted, and how many features the most confident technique consumed -- and then
lists every frame more than 1 px off. A frame within 1 px agrees, one more than
2 px off is wrong, and one in between is marginal; `--agree-px` and `--wrong-px`
move the two bounds. Those columns are the point: they are what tells you
whether a pass's own numbers separate its good answers from its bad ones, and on
the observations checked so far they do not.

### A difference every frame shares

A C-matrix is an absolute attitude, so a difference between two of them is a
difference in where the camera pointed, whether it varies from frame to frame or
is the same on all of them. Every number the report prints is the difference as
measured, with nothing taken out.

A difference shared by every frame is what a coordinate defect inside one of the
two pipelines looks like, so the report measures it on its own: the median signed
difference over the frames that agree, resolved onto the camera's axes. One over
0.1 px in either axis is called out as a defect.

The F ring bundle showed one. Our attitudes differed from its by `+0.494, +0.491`
px on ISS_006RI and within a hundredth of that on four other observations spanning
the mission. The defect was ours: the star model predicted each star's position
with whole numbers on pixel corners, as the geometry library does, while the star
techniques measured each centroid with whole numbers on pixel centers, as an array
index does, and the offset is the difference of the two. A perfectly pointed frame
came out shifted by half a pixel in each axis, and the attitude written from that
offset was rotated by the same amount. With the star model's positions converted
before the comparison, the shared difference on the same frames is `+0.005,
-0.013` px and the scatter about it is unchanged.

## Measuring the mosaic instead

A mosaic built in a co-rotating frame is built on a model that says where the ring
core is, so the core should land at radius offset zero in every column.

```bash
python util/nav_verification/measure_core_radius.py mosaic.fits
```

A displacement that is the same all the way round is the navigation, the orbit
model, or the epoch being off together. A displacement that steps between adjacent
longitudes is two neighboring frames navigated differently from each other, because
real ring structure varies smoothly with longitude and a pointing error does not.
The step's longitude says which frame to look at, and the check works on
observations nobody else has navigated.

Adjacency is in longitude, not in storage. A sparse mosaic keeps only the longitudes
it holds data for, so two columns side by side in the file can be tens of degrees
apart on the ring -- ISS_241RF has sixteen such gaps, the widest 74.9 degrees -- and
over that distance the ring really does change. Pairs straddling a gap are counted
and reported as gaps, never as steps.

The median filter that removes single-column spikes before the steps are counted
keeps to the same boundary: it runs within each stretch of columns one bin apart, so
a column beside a gap is never smoothed against columns tens of degrees away. On
ISS_241RF that is the difference between 132 reported steps and 146; the fourteen it
adds are real changes at the ends of stretches that the values across the gap had
been averaging away.

## Running the tests

From the repository root, and with `python -m` rather than a bare `pytest`, because
`util` is a directory on the path rather than an installed package:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -m pytest util/nav_verification/tests -q
```

`pyproject.toml` sets `testpaths = ["tests"]` and `scripts/run-all-checks.sh` lints
and type-checks `src tests`, so nothing under `util/` runs in the gate. Run the
checks by hand when changing this package:

```bash
python -m ruff check util/nav_verification
python -m ruff format --check util/nav_verification
MYPYPATH=src python -m mypy util/nav_verification --explicit-package-bases
```
