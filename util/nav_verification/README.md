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

The report gives, per frame, the angle to the bundle's answer in NAC pixels beside
the quantities that might have predicted a bad one -- the reported sigma, the
spread between the techniques that formed the answer, how many star features the
models emitted, and how many features the most confident technique consumed -- and
then lists every frame worse than the tolerance. Those columns are the point: they
are what tells you whether a pass's own numbers separate its good answers from its
bad ones, and on the observations checked so far they do not.

### The constant comes out first

Two pipelines can disagree by the same small vector on every frame, which is not a
disagreement about pointing at all but about which corner of a pixel its coordinate
names. On this bundle that constant is real and it is large enough to matter:
`+0.494, +0.491` px on ISS_006RI, and within a hundredth of that on four other
observations spanning the mission. Left in, it sits inside every number the report
prints and makes a good pass look ten times worse than it is.

So the difference is resolved onto the camera's axes, the part common to the run is
measured as a median over the frames that already agree -- where a handful of badly
navigated frames cannot move it -- and what is left is the per-frame disagreement.
Both are printed. On ISS_006RI the raw median is 0.70 px and the median after the
constant is **0.07 px**, with 243 of 417 frames inside a tenth of a pixel.

The constant itself is worth chasing rather than discarding: a half-pixel in both
axes is what a pixel-center-versus-corner convention difference looks like, and only
one of the two pipelines can be right.

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
