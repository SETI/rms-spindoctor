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
image.

Two mechanisms keep that bounded. They are independent, and a stage needs both.

Striping
========

A backplane is very nearly a per-pixel function of the ray through each pixel, so a
band of rows can be evaluated on its own and the bands stacked into the array a
whole-frame evaluation would have returned. Only one band's intermediates exist at a
time, so the live heap is set by the strip height rather than by the frame. What
the "very nearly" costs is the subject of the paragraphs below, and it is small
enough to be worth this.

Strip boundaries fall on whole rows, so the stacked array is the whole-frame
array up to one coupling between pixels. ``oops`` solves the photon path by
iteration and stops when the largest light-time change *anywhere on the meshgrid*
falls under its precision goal, so how far every pixel's solution is refined
depends on which other pixels shared the call. That goal bounds the light time to
3e-07 s, about 90 m of travel, and that is the bound on a striped answer: not the
last bit, but far tighter than anything a navigation can see.

Measured against whole-frame calls on a Cassini frame:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Backplane
     - Striped against whole-frame
   * - ``where_in_front``
     - bit-identical
   * - ``where_inside_shadow``
     - bit-identical
   * - ``distance``
     - agrees to 2.5e-13 of its value
   * - ``ring_radius``
     - agrees to 5.6e-12 of its value

The boolean backplanes are comparisons, so they are bit-identical only while no
pixel sits nearer its threshold than the difference above; a frame that has one
flips that pixel. Anything that lowers the precision goal -- a reprojection does,
for the whole process -- widens all of this in proportion. The ring radius is the
loosest of the four and is the one the haze model's strips read, where it is
thresholded against the ring annulus.

The tests check the assembly against the whole-frame array through a backplane
stand-in that answers each strip from the same dense arrays, on a frame taller
than one strip. They test the stacking, not the solver: the agreement above is
measured rather than asserted.

Three places stripe:
``NavModelRings._striped_backplanes`` for the ring quantities,
``nav_model_body._body_strips`` for a body's oversampled box, and
``titan_geometry._striped_occlusion`` for both occlusion masks over one set of
strips. Each caps a strip at
:data:`~spindoctor.nav_model.nav_model_rings.BACKPLANE_STRIP_ROWS`,
:data:`~spindoctor.nav_model.nav_model_body.BODY_STRIP_ROWS` and
:data:`~spindoctor.nav_model.titan_geometry.OCCLUDER_STRIP_ROWS` rows
respectively, the same number for the same reason, and each strip's single
backplane answers every quantity asked of it, so the surface intercept is solved
once per strip.

A caller must take everything it needs from a strip while that strip is the one in
hand. Asking again afterwards rebuilds the whole box and gives back nothing.

Striping is not the right bound everywhere. It moves the same work into smaller
pieces, so it bounds memory and nothing else, and the haze model's envelope box is
a case where the work itself is unbounded: it grows with the body's apparent size
and reaches seventy times the frame's area. That box is undersampled rather than
striped, which is the only bound that reaches the cost, and it pays for it in
angular resolution. See :doc:`dev_guide_navigation_models_titan`.

The ring model does not stripe everything, and the ring radius in the table above
is measured because the haze model's strips read it, not because this model does.
The ring radius, the ring radial resolution, ``border_atop`` and ``radial_mode``
backplanes run over the whole extended frame through the observation's extended
backplane, because the model renders from the whole-frame radius array and the
edge backplanes are derived inside ``oops`` from that cached array. Striping
bounds the three quantities it covers -- the ring-plane distance, the planet
shadow and the planet occlusion -- and those whole-frame calls set the ring
model's remaining floor. Lifting that floor means striping inside ``oops``
itself rather than inside each of its consumers, where every consumer would
inherit it; that is asked for upstream rather than worked around here.

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

:func:`~spindoctor.support.memory.release_transient_memory` collects the cycles
and, where the C library exposes ``malloc_trim``, returns the freed arenas too.
Where it does not, the collection still runs, which is the larger of the two
effects, and the resident size falls when the allocator next reuses the space
rather than at once. Measured over one striped ring pass on a Voyager Saturn
frame, with both halves available:

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

After each strip, in each of the two striped loops, and nowhere else.

Coarser placements were measured and rejected. Releasing at the boundary between
whole models, and again between techniques, changed a Voyager Saturn frame's peak
from 8.70 GB to 8.74 GB and another from 8.60 GB to 8.78 GB: no gain, inside the
run-to-run spread. A release reclaims only what nothing refers to any more, and at
a stage boundary the strips inside that stage have already given back what they
can while the observation's backplane caches still hold what they hold. Neither
part is something a release can move; see `What a release cannot reach`_.

This is also deliberately not wired into a general allocation path. A collection is
cheap against a backplane evaluation and expensive against a small one, so it
belongs only where a large unit of work has just ended, which is what a strip is.

What a release cannot reach
===========================

A ring render leaves roughly four gigabytes of resident size behind on a Voyager
Saturn frame. A release cannot move it, and the reason is not fragmentation: asked
directly on such a frame, after a collection and an arena release have both run,
the C library reports 0.23 GB free and retained against 3.16 GB handed out on the
heap and 3.46 GB handed out in mappings, with a further 0.10 GB in the
interpreter's own arenas. Almost all of it is live.

What holds it is the observation's backplanes. An ``oops`` ``Backplane`` caches
every event, intercept and computed surface it has been asked for, and each entry
is sized by the meshgrid rather than by the answer, so on an extended frame the
entries are megabytes and there are hundreds. Emptied one kind at a time on a
Voyager Saturn frame:

.. list-table::
   :header-rows: 1
   :widths: 45 15 40

   * - Cache
     - Entries
     - Returns
   * - computed backplanes
     - 237
     - 1.89 GB
   * - unmasked intercept events
     - 2
     - 1.26 GB
   * - observation events, with and without line-of-sight derivatives
     - 2
     - 1.88 GB

Settled resident size falls from 6.80 GB to 1.77 GB. The observation events alone
are 1.88 GB, and ``oops`` builds them when a ``Backplane`` is constructed whether
or not a derivative is ever asked for.

So the two placements of
:func:`~spindoctor.support.memory.release_transient_memory` that measured nothing
-- one after the whole-frame ring evaluations, one at each model and technique
boundary -- were not freeing memory that was already free. They were asking a
release to reclaim memory the observation still refers to, which no release does.

Pinning the C library's mmap threshold, so that large arrays are served by mappings
that are returned on free rather than from the heap, recovers about a third of a
gigabyte for about six percent more runtime. That is a poor trade and is not
configured anywhere; it is recorded here so the next reader does not have to
rediscover the result.

The direction that does reach it is dropping the caches once nothing will read them
again. The models are the only stage that reads a backplane, so the caches are dead
weight from the end of the model stage onward, and every backplane is a property
that rebuilds itself if anything below ever does read one. Nothing in this pipeline
drops them, which is where the four gigabytes come from.

.. warning::

   A walk of :func:`gc.get_objects` cannot answer any of this. Numeric NumPy arrays
   are not tracked by the collector, so such a walk reports approximately zero live
   array bytes under gigabytes of live data. Measure by clearing a suspected holder
   and reading resident size back, or by asking the allocator what it has handed
   out.

Correlation
===========

The remaining peak on a wide-margin frame is not in a model at all. It is the
masked normalized cross-correlation in
:class:`~spindoctor.nav_technique.nav_technique_ring_annulus.RingAnnulusNav`, which
transforms the extended frame, zero-padded for linear rather than circular
correlation. On a Voyager frame that peak stands several gigabytes above what the
backplane caches are already holding, and is returned in full when the technique
exits, which is why releasing between techniques does nothing for it: it is one
allocation spike, not an accumulation.

``_masked_ncc_bidir`` needs six spectra, but their lifetimes barely overlap: the
mask spectrum is finished after the second shift-wise sum, the image spectrum after
the third, the model-mask spectrum after the fourth. Each is therefore built where
it is first needed and dropped at its last use, so three exist at once rather than
six. The real part of each inverse transform is copied out into its own array,
because numpy's real part is a view, and a view would keep the complex output alive
behind each of the six shift-wise sums for the rest of the function. The live set
is therefore at most three spectra plus the one inverse transform in flight, plus
the product and conjugate temporaries of the statement being evaluated. Every
product is the one it always was -- only the order of allocation changed -- and
tests check the result against a transform-free evaluation of the same sums.

Declining early
===============

The cheapest backplane is the one never built. A body whose disc reaches past all
four corners of the extended frame leaves no sky around it, and if its terminator
is outside the frame as well there is nothing in the image a shape-based technique
could match. :func:`~spindoctor.nav_model.nav_model_body.body_fills_extfov` asks
the first question of the inventory alone -- a disc against the frame corners,
costing no backplane -- and
:func:`~spindoctor.nav_model.nav_model_body.body_edge_in_frame` asks the second of
four one-pixel-wide backplanes along the frame's boundary, which is where any limb
or terminator inside the frame has to show. Only then does the model decline before
building anything. When nothing in front of the body was in view either, the
navigation records ``body_fills_fov`` as its reason, which is what lets a
statistics report omit the image as one that could not have been navigated.

The ring model's pre-check is the same idea: a 16 x 16 evaluation rules out the two
common cases -- no ring-plane intersection anywhere in the frame, and a visible radial range
entirely outside the catalogue's outermost feature -- without paying for a dense
backplane.
