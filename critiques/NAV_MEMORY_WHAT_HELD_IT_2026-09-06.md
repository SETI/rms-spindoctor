# What Held a Navigation's Memory

**Date:** 2026-09-06
**Measured at:** `rf_nav_ncc_memory`, on top of the striping, strip-release,
correlation-reordering and body-model work.
**Predecessors:** `NAV_MEMORY_SWEEP_2026-09-05.md` and
`NAV_MEMORY_AFTER_THE_FIXES_2026-09-05.md`. The second left six Voyager frames
above 8 GB and attributed most of what remained to a fragmentation floor
"held by nothing the program can drop".
**Question:** what holds it, and can the six be brought under the limit.
**Method:** the same six frames, each navigated in its own process with
`--no-write-output-files`, peak RSS from `getrusage(RUSAGE_CHILDREN)`, two at a
time -- the predecessor's method exactly, so the numbers are comparable.
Stage-level figures come from `/proc/self/clear_refs` and `VmHWM` rather than
from sampling, so neither costs a measurement. This is a frozen snapshot;
nothing in this file is maintained.

---

## Verdict

The floor is not a floor. The residue a ring render leaves is held, by the
observation's own backplane caches, and dropping it is a two-line change. All
six frames now fit, the heaviest at 7.66 GB against 9.82, with every offset and
status unchanged and about half the runtime.

| image | before | after | status and offset |
|---|---|---|---|
| C3488207 | 9.82 | 7.66 | success, unchanged |
| C4376436 | 8.77 | 6.52 | failed both, unchanged |
| C4343938 | 8.29 | 6.40 | success, unchanged |
| C4338535 | 8.25 | 6.39 | conflicted, unchanged |
| C4347906 | 8.23 | 6.45 | success, unchanged |
| C4330620 | 8.02 | 6.42 | success, unchanged |

## The residue is live

The predecessor established the residue by demolition -- dropping each
suspected holder and reading settled resident size -- and concluded that
nothing held it. Asking the allocators directly says otherwise. On C3488207,
after `gc.collect()` and `malloc_trim(0)` have both run:

| | |
|---|---|
| glibc `fordblks`, free and retained | 0.23 GB |
| glibc `uordblks`, handed out, non-mmap | 3.16 GB |
| glibc `hblkhd`, handed out, 40 mappings | 3.46 GB |
| CPython pymalloc, 99 arenas | 0.10 GB |

`fordblks` is the quantity the fragmentation account predicts should be large.
It is a fifteenth of what is resident.

An `oops` `Backplane` caches every event, intercept and computed surface it has
been asked for, and it is sized by the meshgrid rather than by the answer, so
on the extended frame each entry is megabytes. Emptying the caches one kind at
a time, over every `Backplane` alive in the process:

| cache | entries | returns |
|---|---|---|
| computed backplanes | 237 | 1.89 GB |
| unmasked intercept events | 2 | 1.26 GB |
| observation events, with and without line-of-sight derivatives | 2 | 1.88 GB |

Settled resident size: 6.80 GB to 1.77 GB.

**Why the predecessor's demolition missed it.** It cleared
`obs.ext_bp.backplanes`, and only that, on one of the three `Backplane` objects
alive. `surface_events`, `intercepts` and `obs_events` were never touched. The
0.32 GB that came back was read as the answer, and the conclusion drawn from it
-- that nothing inside the models could lower the floor -- was a bound on the
probe rather than on the program.

`obs_events` is worth naming on its own: `oops` builds both the plain
observation event and the one carrying line-of-sight derivatives eagerly, in
`Backplane._refresh`, whether or not a derivative is ever asked for. On a
1800x1800 meshgrid that is 1.88 GB allocated before any model has asked a
question.

## What the two changes were worth

The models are the only stage that reads a backplane; no technique and nothing
below them in the orchestrator touches one. So the release goes where the model
stage ends. Reading `VmHWM` on each side of that boundary separates the stages:

| tree | model stage | resident at the boundary | technique stage | whole run |
|---|---|---|---|---|
| before | 7.66 | 7.66 -> 7.66 | 9.82 | 9.82 |
| release only | 7.67 | 7.67 -> 1.50 | 5.30 | 7.67 |
| release and correlator | 7.66 | 1.50 -> 1.50 | 3.77 | 7.66 |

The correlator change is three properties of the transforms, none of which
touches the arithmetic. The fields are real, so half a spectrum is redundant.
The result is real, so `np.real` of a complex inverse is a strided *view* that
keeps a complex array of twice its size alive -- six times over, once per
shift-sum surface. And the normalization allocated a dozen full-size
temporaries where it could write into surfaces that had just stopped being
needed. One call at Voyager's padded size:

| surfaces built with | resident growth | wall |
|---|---|---|
| full spectra, real parts as views | 2.16 GB | 18.5 s |
| full spectra, real parts copied out | 1.75 GB | 17.9 s |
| half spectra | 1.07 GB | 8.6 s |

The middle row is bit-identical to the first. The last agrees to 9e-16, marks
the same shifts invalid and puts the peak in the same place.

## What is still not known

- **What sets the model stage.** It is now the whole peak, 7.66 GB on the
  heaviest frame, and 0.34 GB of headroom is thin. Clearing the computed-
  backplane cache between models measures 7.10 GB for no change in wall time
  or offset, which is issue 584; it reaches into an `oops` cache dictionary,
  which is why it is a question rather than a change.
- **Whether a heavier Voyager frame exists.** This sweep, like its
  predecessor, re-measured frames already known to be heavy. Neither drew a
  fresh sample, so neither can find a frame the first sweep missed.
- **Whether the correlation surface needs to be as large as it is.** Peaks
  come only from within the search margin, 5% of the surface, while the
  quality metric's background is drawn from all of it. That is issue 585, and
  it is a question about the metric before it is one about memory.
- Simulated observations were not sampled, as in both predecessors.

## Faults in the measurement itself

**A probe inflated the process it was measuring, and the difference was chased
as a regression.** A background sampler capturing a Python stack at every new
high-water mark reported 12.44 GB on a frame whose `ru_maxrss` is 9.82. Four
navigations were spent bisecting the four branches of an open series for a
2.6 GB regression that does not exist. The check that would have caught it in
one run -- reading `ru_maxrss` and the sampled peak in the same process, which
agree to 0.01 GB -- was run afterwards.

**The same probe reported a growth ladder that stopped 7.6 GB below its own
peak.** It captured a stack only when the mark rose by more than 50 MB, and
raised the mark silently on smaller steps, so memory that climbed in small
increments raised the mark on every sample and was recorded on none of them.

**Gate runs tested the wrong tree.** `setup.sh` begins with
`cd /seti/newnav/rms-nav`, so `cd <worktree> && source setup.sh && pytest` runs
in the main worktree. Several lint and test runs reported on code that was not
the code under change. The give-away was a test count 41 short of the branch's
own.

**A queued job waited forever, and a `pkill` killed the session twice.** A
`pgrep -f` guard matched the shell that had just written the script it was
guarding, because that shell's command line carries the heredoc, pattern
included. Waiting on a predecessor's marker file has neither failure.

**The first synthetic comparison had nothing in it to compare.** Flat random
fields fail the bidirectional NCC's variance test at every shift, so both
surfaces came back entirely `-inf` and the comparison divided by an empty set.
Ring-like fields under noise were needed before any of the correlator numbers
above could be read.
