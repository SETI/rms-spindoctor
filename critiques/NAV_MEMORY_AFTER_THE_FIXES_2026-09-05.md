# Navigation Memory After the Fixes

**Date:** 2026-09-05
**Measured at:** `rf_nav_memory_stats` @ 5f90d9cb -- the strip-level
release, the correlation spectra reordering, the body-fills-frame decline,
and the peak-memory recording, on top of the striping work.
**Predecessor:** `NAV_MEMORY_SWEEP_2026-09-05.md`, which measured the same
pipeline before any of it and asked what per-task limit navigation needs.
**Question:** what does the distribution look like now, does 8 GB hold, and
which of the predecessor's open questions can be closed.
**Method:** the 244 images the predecessor's sweep found heaviest -- every
Cassini, Galileo and Voyager frame it measured above roughly 6 GB -- each
re-navigated in its own process with `--no-write-output-files`, peak RSS
from `getrusage(RUSAGE_CHILDREN)`, two at a time. Twelve New Horizons
frames were measured separately, by the same method, chosen as the twelve
heaviest of the ninety-three that sweep saw, since none of them was inside
the 244. This is a frozen snapshot; nothing in this file is maintained.

---

## Verdict

8 GB now holds for 238 of the 244 heaviest frames, and for every Cassini,
Galileo and New Horizons frame among them. The six that exceed it are all
Voyager, and four of the six are within 0.3 GB of the line, which is about
three times the run-to-run spread.

The instrument ordering the predecessor found is unchanged and still
follows extended field-of-view area, but the spread has collapsed. Galileo
and New Horizons now sit at a fifth of the limit, Cassini at half, and only
Voyager approaches it.

| dataset | n | baseline max | now: median | p90 | max | over 8 GB |
|---|---|---|---|---|---|---|
| coiss_saturn | 105 | 9.92 | 2.97 | 3.83 | 4.37 | 0 |
| gossi | 65 | 12.58 | 1.50 | 1.73 | 1.77 | 0 |
| vgiss | 74 | 25.95 | 3.81 | 7.81 | 9.82 | 6 |
| nhlorri | 12 | 4.43 | 0.76 | 0.76 | 1.69 | 0 |

These are the heavy tail, not a random sample, so the percentages here are
not comparable with the predecessor's. Read them as "of the frames that
were the problem", not "of the archive". The archive is better than this
table, not worse.

New Horizons is a smaller sample than the rest and was drawn differently.
No LORRI frame was inside the 244, because none of them reached the roughly
6 GB the selection took, so its twelve heaviest frames were measured on
their own. Its whole distribution sat below the limit before any of this
work and it was never the question; it is here because the same fixes reach
it, and because what it does at the bottom of the range says something the
other three cannot.

## The six that remain

| image | baseline | now | over by |
|---|---|---|---|
| C3488207 | 14.30 | 9.82 | +1.82 |
| C4376436 | 14.94 | 8.77 | +0.77 |
| C4343938 | 14.35 | 8.29 | +0.29 |
| C4338535 | 13.99 | 8.25 | +0.25 |
| C4347906 | 13.86 | 8.23 | +0.23 |
| C4330620 | 13.85 | 8.02 | +0.02 |

Their baselines cluster in 13.85-14.94 GB, and every frame whose baseline
exceeded 15.5 GB now clears the limit. That inversion is the shape of the
result: the frames that were worst were worst because of the stages that
were fixed, and what is left is a cost those stages never dominated.

The band is a correlate and not a cause. Thirteen frames of this sweep have
baselines inside it and only four of them exceed the limit; the others land
at 7.28 to 7.48 GB and one at 2.05 GB. Nothing here identifies what
separates them, and the honest position is that C3488207 in particular is
unexplained.

## What the fixes were, and what each was worth

Measured on Voyager Saturn frames, whose offsets were byte-identical before
and after every change:

| change | effect |
|---|---|
| strip-level release | about 3 GB per frame |
| correlation spectra reordering | about 0.9 GB per frame |

The first of them is the consequence of the first finding. Evaluating a
backplane a strip of rows at a time bounds the live heap, which is what
striping was built to do, and does not bound resident size, which is what an
out-of-memory kill reads. The intermediates sit in reference cycles, so
nothing is freed until a collection runs; and once freed, the C allocator
keeps the arenas. A striped pass therefore grew resident size by the *sum* of
its strips rather than the largest of them, and its striping could not be
observed from outside.  Measured over one striped ring pass:

| released with | resident growth | wall time |
|---|---|---|
| neither | 2.84 GB | 47.5 s |
| collection only | 1.39 GB | 41.9 s |
| arena release only | 2.45 GB | 45.9 s |
| both | 0.00 GB | 45.3 s |

The second finding is that after that, the remaining peak was not in a model
at all. A sampled trace puts it inside `RingAnnulusNav`, whose masked
correlation transforms the zero-padded extended frame: on a Voyager frame the
six spectra it built up front were the largest arrays in the process, though
their lifetimes barely overlap. Building each where it is first needed and
dropping it at its last use holds three at once. The stage profile of one
frame, after the release and before the reordering:

```
NavModelStars       1.01 -> 1.02
16x NavModelBody    1.02 -> 1.26      (0.24 GB for all sixteen)
NavModelRings       1.26 -> 5.09
>> RingAnnulusNav   5.06 -> peak 8.69 -> 5.07 on exit
```

## What New Horizons shows about the floor

The twelve heaviest LORRI frames, which are the twelve the predecessor found
heaviest out of ninety-three:

| baseline | now | |
|---|---|---|
| 4.43 | 0.76 | |
| 3.13 | 1.65 | |
| 2.42 | 1.69 | |
| 0.97 | 0.51 | |
| 0.95 | 0.50 | |
| 0.95 | 0.52 | |
| 0.87 | 0.76 | |
| 0.76 | 0.76 | unchanged |
| 0.76 | 0.76 | unchanged |
| 0.76 | 0.76 | unchanged |
| 0.75 | 0.75 | unchanged |
| 0.75 | 0.75 | unchanged |

Two things are worth reading out of this. The heavy end moves as much as
anywhere else -- the worst frame falls by a factor of 5.8, and the three
frames above 2 GB all fall sharply -- which says the fixes are not specific
to wide-margin instruments; they follow the work, and LORRI does the same
work on a smaller frame.

The light end does not move at all. Five of the twelve sit at 0.75 to 0.76 GB
before and after, to the hundredth of a gigabyte. That is the cost of a
navigating process that has loaded its interpreter, its libraries and its
kernels, and it is the same number whatever the image. Nothing in this work
reaches below it, and nothing should be expected to: it is not the
fragmentation floor discussed next but a fixed startup cost underneath even
that. It matters for sizing, because it says the marginal cost of the
cheapest possible navigation is not near zero, and a worker running many
small images pays it once per process rather than once per image.

## The floor

A ring render leaves about four gigabytes resident that no part of the
program holds. Dropping the observation's cache of computed backplanes
returns 0.32 GB; dropping the models returns nothing; dropping the
observation returns nothing. What remains is resident but free: live objects
scattered across the allocator's arenas, so whole pages cannot be returned.

This is why Voyager's median is 3.81 GB while Galileo's is 1.50 GB. Most of
what a Voyager Saturn frame now costs is the floor rather than the work.

Two placements of the release were measured against that floor and rejected
-- at each model and technique boundary, and immediately after the whole-frame
ring evaluations -- because both were freeing memory that was already free
and merely unreturnable. Pinning the C library's mmap threshold recovers
about 0.34 GB of it for about 6% more runtime, which is a poor trade. The
floor is tracked as issue 573.

## What the predecessor did not know, and what is now known

**"Whether the rings model and the Titan contaminant mask dominate every
heavy frame is untested."** They do not. On the frame profiled, sixteen body
models cost 0.24 GB between them and the peak was a technique, not a model.

**"Whether the memory is held or merely allocated and released was not
distinguished."** It is allocated and released, and not returned. This was
the central question and it had the central answer: every fix that assumed
the memory was held measured zero.

**"Voyager's distribution is not settled."** Still not settled. This sweep
re-measured the frames the predecessor found heaviest rather than drawing a
new random sample, so it cannot find a Voyager frame the predecessor missed.

**Per-task limits.** On this evidence a limit of 12 GB covers every frame
measured here with room for the run-to-run spread, against the 32 GB the
predecessor's numbers implied. A limit of 8 GB covers everything except six
Voyager frames.

## What is still not known

- What distinguishes C3488207, and the five frames grazing the limit, from
  the frames beside them in the same baseline band. No trace has been run on
  any of them.
- Whether the fragmentation floor can be reduced at all. Forcing large
  arrays to `mmap` recovered a twelfth of it, which says the fragmentation is
  not dominated by the frame-sized arrays, and nothing has been measured
  about what it *is* dominated by.
- Whether any archive frame outside this heavy tail regressed. Nothing here
  measured a frame the predecessor found cheap.
- Simulated observations were not sampled, as they were not in the
  predecessor.

## Faults in the measurement itself

**The first run of this sweep was discarded.** It measured a tree that the
correlation reordering had not yet reached, and it was competing for memory
with the verification of that reordering. Nineteen images had been measured;
they are kept as `results_pretrim.tsv` and are used in this file only as the
"striping, no release" column of the per-change table.

**One frame's numbers come from a tree edited mid-run.** The technique-level
release was moved out of an exception guard while a four-frame verification
was in flight. The edit changes behaviour only on the error path and the
frames in question succeeded, so the numbers stand, but the rule against
editing a tree under measurement was broken and is recorded here rather than
waved past.

**A residue probe reported an impossible number and was believed briefly.**
Walking `gc.get_objects()` for live arrays reported approximately zero bytes
against a 4 GB residue. Numeric NumPy arrays are not tracked by the garbage
collector, so that walk cannot see them. The floor was established instead by
demolition -- dropping each suspected holder in turn and reading settled
resident size -- which does not depend on the collector knowing anything.

**A sampled trace initially missed the peak by 2.7 GB.** Sampling resident
size only at instrumented call sites reported 5.96 GB for a frame whose
`ru_maxrss` was 8.66 GB. The replacement samples from a background thread and
records what was running at the maximum, which is what found `RingAnnulusNav`.

**Three fixes were proposed, implemented, measured at zero, and removed.**
Each assumed the memory was held. They are listed above so that the next
reader does not spend the same hours; the demolition probe that settles the
question takes four minutes and should have been run first.
