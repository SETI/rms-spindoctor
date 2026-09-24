# Navigation Memory Sweep

**Date:** 2026-09-05
**Baseline:** `rf_nav_memory_and_errors` @ 512b944d (the Titan backplane
bounds, the parallel results-index walk, and the results-index log-root
fix, on top of `rf_cloud_support`). Every measurement below ran this one
commit; the exception-policy work of the same night was deliberately kept
in a separate worktree so it could not change the code mid-run.
**Question:** what per-task memory limit does navigation need, what does
the tail of the distribution look like, and can the images that need the
most be predicted before they are navigated?
**Method:** 1,800 images drawn at random across the four supported
instruments -- 1,200 Cassini ISS (Saturn), 300 Voyager ISS, 200 Galileo
SSI, 100 New Horizons LORRI -- each navigated in its own process with
`--no-write-output-files`, its peak RSS taken from
`getrusage(RUSAGE_CHILDREN)`. One frame was then re-run alone with a
0.5-second RSS trace joined to its section headers, to attribute the
memory to a pipeline stage. 1,613 clean measurements were obtained; the
shortfall is accounted for below. This is a frozen snapshot; nothing in
this file is maintained.

---

## Verdict

There is no single per-task memory limit that is both safe and
economical. The four instruments span a factor of six in peak memory --
4.43 GB for LORRI against 25.95 GB for Voyager -- and the ordering is not
an accident of which images were drawn. It follows the **extended
field-of-view area**, which each instrument's configured search margin
sets, and which differs between instruments by more than a factor of two.

The 8 GB that a `max_memory_allowed_per_task` of 8 would enforce fails
6.3% of all images and 32.7% of Voyager images.

Grouping the heavy frames by volume or by planetary encounter predicts
them well, but that is a symptom rather than a cause: volumes correlate
with encounters, encounters correlate with how much of a ring system and
how many satellites fall in one frame, and it is the *scene* that costs
the memory. A single frame profiled to the section shows where it goes,
and it is not where the frame's contents would suggest.

## What was measured

| dataset | n | median | p75 | p90 | p95 | p99 | max | >8 GB | >12 GB | >16 GB |
|---|---|---|---|---|---|---|---|---|---|---|
| coiss_saturn | 1175 | 1.83 | 5.11 | 5.75 | 6.97 | 8.62 | 9.92 | 2.0% | 0.0% | 0.0% |
| vgiss | 147 | 5.25 | 12.76 | 16.78 | 17.39 | 25.28 | 25.95 | 32.7% | 27.2% | 12.9% |
| gossi | 198 | 3.75 | 6.89 | 9.35 | 10.72 | 12.49 | 12.58 | 15.2% | 1.0% | 0.0% |
| nhlorri | 93 | 0.24 | 0.28 | 0.76 | 0.95 | 4.43 | 4.43 | 0.0% | 0.0% | 0.0% |
| **all** | **1613** | **1.77** | **5.17** | **6.91** | **8.89** | **16.49** | **25.95** | **6.3%** | **2.6%** | **1.2%** |

The body of the distribution is small and the tail is long. Half of every
instrument's images fit in under 6 GB; the cost is concentrated in a few
percent of frames, and those few percent are 4 to 14 times the median.

## The first-order driver: extended field-of-view area

Each instrument configures a search margin, and the extended frame is the
detector plus twice that margin on each axis. The resulting areas order
exactly with the observed peaks:

| instrument | detector | margin (v, u) | extended frame | Mpx | x detector | max GB |
|---|---|---|---|---|---|---|
| nhlorri | 1024x1024 | 60, 60 | 1144x1144 | 1.31 | 1.25 | 4.43 |
| coiss NAC | 1024x1024 | 50, 140 | 1124x1304 | 1.47 | 1.40 | 9.92 |
| gossi | 800x800 | 350, 350 | 1500x1500 | 2.25 | 3.52 | 12.58 |
| vgiss | 1000x1000 | 400, 400 | 1800x1800 | 3.24 | 3.24 | 25.95 |

Voyager's margin is 400 pixels on a 1000-pixel detector, so its extended
frame holds 3.24 times the detector's area and 2.2 times Cassini's
extended frame. Galileo's margin inflates its frame the most in relative
terms, 3.52x. LORRI's barely inflates at all, and LORRI is the one
instrument with no image over 8 GB.

This is why the smaller detectors cost more memory than the larger ones,
which is otherwise the wrong way round: an 800-pixel Galileo frame is
worked over a 1500-pixel grid, while a 1024-pixel LORRI frame is worked
over an 1144-pixel one.

The relationship is not proportional -- LORRI uses less than its area
share, Voyager more -- so area is a first-order term and not the whole
story.

## Where the memory actually goes

`C3466614` (VGISS_6110, Voyager 2 at Saturn) peaked at 26.29 GB when
re-run alone. Its RSS trace joined to its section headers:

```
 0.12 ->  1.92 GB   image load, derivatives, stars, and 16 body models
 1.92 -> 16.22 GB   CREATE RINGS MODEL     +14.3 GB over 154 s
12.40 -> 26.29 GB   TITAN MODEL            +13.9 GB over 239 s   <== peak
```

Two models account for 24 of the 26 GB, in roughly equal shares.

**Sixteen body models cost 0.05 GB between them.** Saturn, Atlas,
Calypso, Daphnis, Dione, Enceladus, Epimetheus, Helene, Janus, Mimas,
Pan, Pandora, Prometheus, Rhea, Telesto and Tethys were all built for
this frame, and their combined cost is not visible in the trace. A
crowded satellite inventory is not what makes a frame expensive.

What the two expensive models have in common is that each reasons about
something whose extent is unbounded by the frame:

- The rings model logged `Ring plane radial range visible in image:
  [505, 3104680] km`. Saturn's main rings end near 140,000 km. This is a
  distant approach viewing the ring plane at a shallow angle, so the
  plane sweeps out to 3.1 million km across the field -- a radial extent
  22 times the rings themselves.
- The Titan model logged `envelope diameter = 8.02 px`. Titan is eight
  pixels across in this frame and cost 13.9 GB. The envelope-box bounds
  added in this same commit never engaged (the stride line appears zero
  times in the log) because an eight-pixel body's box is trivially small.
  The cost is therefore elsewhere in that model -- the contaminant mask
  is the candidate, since it tests occlusion against each sibling body in
  turn over a mask box, and this frame has sixteen siblings.

The eight-pixel body costing 13.9 GB is the clearest signal in the sweep
that something is disproportionate rather than merely large.

## What the volume and encounter grouping is

Both group well, and both are downstream of the same thing.

| Voyager encounter | n | median | max | >8 GB |
|---|---|---|---|---|
| Jupiter | 67 | 5.61 | 22.27 | 31.3% |
| Saturn | 50 | 13.86 | 25.95 | 54.0% |
| Uranus | 14 | 4.20 | 6.43 | 0.0% |
| Neptune | 16 | 2.88 | 5.25 | 0.0% |

| Cassini volumes | n | median | max | >8 GB |
|---|---|---|---|---|
| COISS_2001-2003 | 27 | 6.24 | 8.58 | 3.7% |
| COISS_2004-2008 | 57 | 5.05 | 9.92 | 19.3% |
| COISS_2009-2020 | 99 | 4.10 | 8.70 | 4.0% |
| COISS_2021+ | 992 | 1.41 | 8.62 | 0.8% |

Uranus and Neptune produced no image over 8 GB at all. Jupiter and Saturn
carry the whole Voyager cost, and the early Cassini volumes carry a
disproportionate share of the Cassini cost. Both facts are consistent
with the profile above: those are the frames that view an extended ring
system, and the ring model is one of the two expensive stages.

Grouping by volume therefore predicts cost, and is knowable before any
image is navigated, but it explains nothing. Two frames in the same
volume can differ by a factor of ten.

## Runtime as a predictor

Peak memory rises monotonically with wall time across all instruments:

| wall | n | median GB | max GB |
|---|---|---|---|
| 0-20 s | 854 | 0.67 | 5.11 |
| 20-60 s | 507 | 5.08 | 13.65 |
| 60-120 s | 191 | 6.08 | 18.03 |
| 120-300 s | 38 | 6.87 | 22.27 |
| 300+ s | 23 | 15.77 | 25.95 |

Nothing under 20 seconds exceeded 5.11 GB. This is the strongest
correlate found, and it is useless as a predictor because it is only
known after the fact -- but it does mean a `max_runtime` bound
incidentally bounds memory, which may make it the more robust of the two
levers.

The correlation is a tendency and not a law: the 22.27 GB frame ran in
163 seconds, while several 400-second frames stayed near 14 GB.

## What the numbers imply for a limit

On this evidence, and with no fix applied:

| dataset | observed max | suggested limit |
|---|---|---|
| nhlorri | 4.43 | 6 GB |
| coiss_saturn | 9.92 | 12 GB |
| gossi | 12.58 | 16 GB |
| vgiss | 25.95 | 32 GB |

A single limit for all four would be 32 GB, set entirely by Voyager, and
would waste 26 GB per task on the LORRI group. The task files are already
generated per instrument, and Voyager's are already split per encounter,
so a per-group limit costs nothing to adopt.

These are the numbers *before* any attempt to reduce the memory. The
profile above suggests they are not intrinsic.

## What is not known

- The 26 GB frame is one profile. Whether the rings model and the Titan
  contaminant mask dominate every heavy frame, or only Voyager Saturn
  frames, is untested.
- Why the Titan model spends 13.9 GB on an eight-pixel body is not
  established. The contaminant mask is a hypothesis from reading the
  code, not a measurement.
- Whether the memory is *held* or merely *allocated and released* was not
  distinguished. Peak RSS cannot tell a live 14 GB structure from a
  sequence of transient allocations that were never returned to the
  allocator, and the two have different fixes.
- Cassini's distribution is settled: its maximum was set at image 250 of
  1,175 and never moved. Voyager's is not: its maximum rose through
  13.9, 16.8, 17.8, 22.3 and 26.0 GB as the sample grew, on 147 images.
  A larger Voyager sample may find more.
- Simulated observations were not sampled.

## Faults in the measurement itself

Recorded because they bound how far the numbers can be trusted, and
because two of them were mine.

**25 measurements were contaminated by concurrent work.** All 25 images
that recorded `internal_error` fall in a single two-minute window during
which a four-worker `pytest` run shared the same `capped-run.slice` as
the sweep. They are real `MemoryError`s under real memory pressure, and
they are excluded from every figure above. That they were reported at all
rather than degrading silently is a consequence of the exception-policy
work of the same night: under the previous code each would have been
absorbed into a plausible-looking navigation result.

**The sweep was killed once by `systemd-oomd` and its cause was initially
missed.** Three concurrent navigations at up to 9.9 GB drove the 31 GB
cgroup close enough to its ceiling for oomd to reap the entire scope on
sustained pressure. `journalctl -k` showed nothing, because oomd kills
appear only in the user journal; the first diagnosis therefore concluded
the machine was healthy. The run was restarted at lower concurrency and
under a ceiling sized for the workload rather than for a runaway.

**A verification run tested the wrong tree.** `setup.sh` begins with `cd
/seti/newnav/rms-nav`, so `cd <worktree> && source setup.sh && pytest`
silently ran the main checkout. One reported "12198 passed" came from
unmodified code and was worthless.

**Three predictive boundaries were proposed and retracted.** An
image-number window, a "no Voyager frame above 8.1 GB is anything but
Saturn" claim, and a Galileo encounter boundary were each fitted to a
dozen points and broken by the next batch. The groupings reported above
are the ones that survived the full sample.

**14 images were never measured.** Four failed on a defect in the
harness's image-name extraction (Cassini version numbers are not always
`_1_`), and the rest were in flight when the sweep was stopped and
restarted.

**The Voyager sample was trimmed from 300 to 147.** At roughly three
minutes per image running alone, the full sample would have taken ten
further hours. The trim was taken deliberately; it is why Voyager's
figures rest on the smallest sample of the four and why its tail is the
least settled.
