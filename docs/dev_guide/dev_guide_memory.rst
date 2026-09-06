=================
Bounding a Run
=================

A navigation's memory is dominated by the backplanes it evaluates, and a backplane
is sized by the meshgrid it is handed rather than by the answer it returns. ``oops``
materializes its intermediates over that meshgrid at roughly a kilobyte per pixel,
so a stage that hands it the extended frame pays for the extended frame whatever
the question was. The extended frame is the detector plus twice the instrument's
search margin, which is what makes the wide-margin instruments the expensive ones:
a Voyager frame extends to 1800 x 1800, or 3.24 megapixels, against a 1000 x 1000
detector.

Two mechanisms keep that bounded. They are independent, and a stage needs both.

Striping
========

A backplane is a per-pixel function of the ray through each pixel, so a band of
rows can be evaluated on its own and the bands stacked into the array a whole-frame
evaluation would have returned. Only one band's intermediates exist at a time, so
the live heap is set by the strip height rather than by the frame.

The stacked array is the whole-frame array exactly, not an approximation of it:
strip boundaries fall on whole rows, and no per-pixel quantity depends on a
neighbouring row. Tests assert the identity rather than a tolerance.

Two places stripe:
``NavModelRings._striped_backplane`` for the ring quantities and
``titan_geometry._striped_occlusion`` for both occlusion masks over one set of
strips. Each caps a strip at 128 rows.

A caller must take everything it needs from a strip while that strip is the one in
hand. Asking again afterwards rebuilds the whole box and gives back nothing.

Releasing
=========

Striping bounds the live heap. It does not on its own bound the process's resident
size, and resident size is the number that matters: it is what the kernel's
out-of-memory killer reads and what a recorded peak reports.

Two things sit between the two, and neither is sufficient alone. The intermediates
are held in reference cycles, so dropping the last name bound to a strip frees
nothing until a collection runs, and a collection of the oldest generation is not
otherwise due within the handful of allocations a strip costs. Once freed, the C
allocator keeps the arenas rather than returning them, so the address space is
still charged to the process.

Left alone the two compound, and a run's resident size grows by the *sum* of its
strips rather than by the largest of them -- which is the quantity striping exists
to reduce. A striped pass that does not release is a pass whose striping cannot be
observed from outside.

:func:`~spindoctor.support.memory.release_transient_memory` does both halves.
Measured over one striped ring pass on a Voyager Saturn frame:

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Released with
     - Resident growth
     - Wall time
   * - neither
     - 2.84 GB
     - 47.5 s
   * - collection only
     - 1.39 GB
     - 41.9 s
   * - arena release only
     - 2.45 GB
     - 45.9 s
   * - both
     - 0.00 GB
     - 45.3 s

The results are identical in every row; only the resident size differs. The release
costs nothing worth counting, because a strip is expensive enough that a collection
between strips does not register against it.

Where it is called
------------------

After each strip, in each of the three striped loops, and nowhere else.

Coarser placements were measured and rejected. Releasing at the boundary between
whole models, and again between techniques, changed a Voyager Saturn frame's peak
from 8.70 GB to 8.74 GB and another from 8.60 GB to 8.78 GB: no gain, inside the
run-to-run spread. The reason is that a navigation's peak is set by the largest
single stage rather than by an accumulation across stages, so there is nothing at
a stage boundary for a release to reclaim -- the strips inside that stage have
already given it back.

This is also deliberately not wired into a general allocation path. A collection is
cheap against a backplane evaluation and expensive against a small one, so it
belongs only where a large unit of work has just ended, which is what a strip is.

What a release cannot reach
===========================

A ring render leaves roughly four gigabytes of resident size behind on a Voyager
Saturn frame, and no part of it is held by the program. Dropping the observation's
cache of computed backplanes returns a third of a gigabyte; dropping the models
returns nothing; dropping the observation itself returns nothing. What remains is
resident but free: small live objects are scattered across the allocator's arenas,
so whole pages cannot be handed back however often the release runs.

This is worth knowing before optimizing. Two placements of
:func:`~spindoctor.support.memory.release_transient_memory` were tried against this
floor -- one after the whole-frame ring evaluations, one at each model and technique
boundary -- and both measured no change, because both were trying to free memory
that was already free and merely unreturnable.

Pinning the C library's mmap threshold, so that large arrays are served by mappings
that are returned on free rather than from the heap, recovers about a third of a
gigabyte of it for about six percent more runtime. That is a poor trade and is not
configured anywhere; it is recorded here so the next reader does not have to
rediscover the result.

The practical consequence is a lower bound. Striping and releasing bound what a
stage adds on top of this floor, and nothing available inside the models lowers the
floor itself.

Correlation
===========

The remaining peak on a wide-margin frame is not in a model at all. It is the
masked normalized cross-correlation in
:class:`~spindoctor.nav_technique.nav_technique_ring_annulus.RingAnnulusNav`, which
transforms the extended frame, zero-padded for linear rather than circular
correlation. On a Voyager frame that peak stands several gigabytes above the floor
and is returned in full when the technique exits, which is why releasing between
techniques does nothing for it: it is one allocation spike, not an accumulation.

``_masked_ncc_bidir`` needs six spectra, but their
lifetimes barely overlap: the mask spectrum is finished after the second shift-wise
sum, the image spectrum after the third, the model-mask spectrum after the fourth.
Each is therefore built where it is first needed and dropped at its last use, so
three exist at once rather than six. Every product is the one it always was -- only
the order of allocation changed -- and tests check the result against a
transform-free evaluation of the same sums.

Declining early
===============

The cheapest backplane is the one never built.
:func:`~spindoctor.nav_model.nav_model_rings.NavModelRings._sparse_visibility_skip`
is the idea for the rings: a 16 x 16 evaluation rules out the two common cases
-- no ring-plane intersection anywhere in the frame, and a visible radial range
entirely outside the catalogue's outermost feature -- without paying for a dense
backplane.
