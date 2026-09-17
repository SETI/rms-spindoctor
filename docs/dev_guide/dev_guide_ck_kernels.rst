=================================
Corrected-Pointing C-Kernels
=================================

This chapter is the developer's reference for the two halves of the
corrected-pointing product: the attitude computation that records a C-matrix
beside every navigated offset, and the writer that turns those matrices into
SPICE C-kernels. The consumer's view -- what the kernels claim, how to load
them, the report columns and the ``sd_create_ck`` command line -- lives at
:doc:`/user_guide/user_guide_ck_kernels`.

Overview
========

Navigation measures a pixel offset. That measurement is only usable by code
that knows how to apply it, so the same measurement is also recorded as an
attitude:

1. :mod:`spindoctor.support.cmatrix` converts the navigated offset into a
   corrected camera attitude and records it, with the uncorrected attitude, the
   frame identities and the exposure epochs, into the per-image metadata. This
   half imports ``oops``, because it has to read the observation's own frame
   and field of view.
2. The :mod:`spindoctor.cli.ck` package reads those recorded matrices back out
   of the records and writes type-3 C-kernel segments.

The recorded matrices have a second consumer inside SpinDoctor itself: the
backplane and reprojection stages apply them back onto observations through
:func:`~spindoctor.cli.reproj.offsets.apply_pointing_to_obs`, whose matrix
mechanism is :func:`~spindoctor.support.cmatrix.apply_cmatrix_to_obs` (see
`The readers`_),
so the pipeline's own downstream products are built on the same attitude a
kernel consumer sees.

The two halves share exactly one table -- which spacecraft clock each C-kernel
object's time tags are encoded against -- and it lives in
:mod:`spindoctor.spice_ids`, a constants module importing only the standard
library. Both sides read :data:`~spindoctor.spice_ids.CK_OBJECT_SCLK_ID`;
neither keeps its own copy.

Frames, and the trap this section exists to avoid
=================================================

A C-matrix is the rotation taking a vector expressed in J2000 to the same
vector expressed in a frame::

    v_frame = C . v_J2000

which is what ``cspyce.pxform('J2000', frame_name, et)`` returns. Two are
recorded per navigated image, both in the **SPICE camera frame convention** and
both at the exposure **midtime**: ``cmatrix_original``, the attitude the
furnished kernels gave at navigation time, and ``cmatrix``, the corrected
attitude a kernel should carry. Recording both makes the correction
self-contained -- their difference is the correction -- and gives the writer the
means to verify that the baseline kernels have not changed since navigation.

**The oops observation frame is not the SPICE camera frame.** The navigated
offset lives in the ``oops`` observation frame, which each host relates to the
SPICE camera frame by a constant rotation ``R`` satisfying
``C_oops = R . C_spice``. The host states that rotation, and the camera frame it
applies to, on the observation itself:

.. list-table::
   :header-rows: 1
   :widths: 26 22 26 26

   * - Instrument
     - ``spice_frame_name``
     - ``spice_frame_id``
     - ``spice_to_frame`` (``R``)
   * - Cassini ISS
     - ``CASSINI_ISS_NAC`` / ``_WAC``
     - -82360 / -82361
     - ``diag(-1, -1, +1)``, a 180 degree flip about the boresight
   * - New Horizons LORRI
     - ``NH_LORRI``
     - -98300
     - ``diag(+1, -1, -1)``; the SPICE boresight is -Z
   * - Galileo SSI
     - ``GLL_SCAN_PLATFORM``
     - -77001
     - ``I``; the observation frame is the platform frame
   * - Voyager ISS
     - ``VGn_ISSNA`` / ``VGn_ISSWA``
     - -3n101 / -3n102
     - ``I``; the frozen frame already carries the camera attitude

Those subfields are read rather than restated here, so that the frame an
attitude is recorded in is by construction the frame ``oops`` built the
observation on. A table kept here could name a different camera than the
observation it describes, and ``CASSINI_ISS_NAC`` and ``CASSINI_ISS_WAC`` are
264.8 arcsec apart, or 44.2 NAC pixels, so that confusion is not a rounding
error.

A correction built in the ``oops`` frame and composed onto a SPICE-convention
matrix **without** accounting for ``R`` is a proper rotation of the right
magnitude pointing the wrong way -- for Cassini, with both tangent-plane
components negated. It survives every check a hermetic test can make: the result
is orthonormal, its determinant is 1, and its rotation angle is correct, because
a magnitude is invariant under exactly the error being made. Only real host
frames meeting real kernels can see it. Both conversions therefore go through
:meth:`oops.observation.Observation.get_spice_cmatrix` and
:meth:`~oops.observation.Observation.set_spice_cmatrix`, which apply ``R`` in
both directions from the single declaration.

What a host declares is a claim rather than a measurement, so it is checked. At
the exposure start, midtime and stop, ``C_oops(t) . pxform('J2000',
spice_frame_name, t)^T`` must equal the declared ``R``. A frame that is right at
the midtime and wrong at the edges is a moving frame misdeclared as a constant
one, and the correction reaches a kernel as a single body-fixed rotation. A
violation raises rather than being absorbed.

Voyager is the exception to the check, not to the convention: its frame is
frozen from a tolerance-snapped ``ckgp`` lookup rather than an evaluated chain,
so ``pxform`` cannot place ``VGn_ISSNA`` at all. The accessor still answers
correctly, so only the cross-check is skipped.

Deriving the corrected attitude
===============================

The conversion lives behind one function,
:func:`~spindoctor.support.cmatrix.compute_pointing`, and nothing else computes
it. The helpers beneath it are private for that reason rather than because they
are trivial. The two conversions between the ``oops`` and SPICE conventions are
``oops``'s; what remains here is what ``oops`` does not do -- turning a pixel
offset into a rotation, refusing a malformed record, naming the C-kernel object
and spacecraft clock, and gating a recorded attitude before applying it.

Step 1: the corrected boresight, in the oops frame
--------------------------------------------------

The navigated offset is applied downstream as
``oops.fov.OffsetFOV(fov, uv_offset=(du, dv))``. Note the order: the metadata
``offset`` is ``[dv, du]``, and every consumer constructs the field of view with
``(du, dv)``. That field of view maps pixels to camera tangent-plane
coordinates as::

    xy_from_uv(uv) = fov.xy_from_uv(uv) - xy_offset
    xy_offset      = fov.xy_from_uv(fov.uv_los + (du, dv))

Under the corrected pointing, the true direction seen by pixel ``uv`` in the
*original* frame is ``fov.los_from_xy(fov.xy_from_uv(uv) - xy_offset)``, and the
corrected frame is the one in which the unmodified field of view holds. So the
rotation ``M`` from original to corrected, in ``oops`` frame coordinates,
satisfies::

    M . los_from_xy(xy - xy_offset) = los_from_xy(xy)     for all xy

Evaluating at the boresight -- ``xy = xy_from_uv(uv_los)``, which need not be
zero for a subarray or a distorted field of view -- gives the constraint that
fixes ``M``::

    d = los_from_xy(xy_from_uv(uv_los) - xy_offset)
    b = los_from_xy(xy_from_uv(uv_los))
    M . d = b

Step 2: construct ``M``
-----------------------

``M`` is the minimal rotation carrying ``d`` onto ``b``: axis ``d x b``
normalized, angle ``arctan2(|d x b|, d . b)``, realized with
``cspyce.axisar(axis, angle)``. ``axisar`` is the **active** vector rotation
(``M . d = b``); ``cspyce.rotate`` is the frame rotation and returns the
transpose. When ``|d x b|`` falls below 1e-12 -- a zero or sub-nanoradian offset
-- ``M`` is exactly the identity, and the corrected matrix is then returned
bit-identical to the uncorrected one, so that no correction means no change.
``M`` is exact by construction; nothing is orthonormalized.

The angle is ``arctan2`` and not the mathematically equivalent
``arccos(d . b)``, and the difference matters in exactly the regime a sub-pixel
offset lives in. At a Cassini NAC pixel scale of 6e-6 rad/px: at 0.01 px
``arccos`` returns 5.9605e-08 rad against a true 6.0000e-08, 0.7% low; at
0.001 px it returns **exactly 0.0**, because the cosine of 6e-9 rad rounds to
1.0 in float64 and the correction would be dropped altogether. ``arctan2``
returns the true angle in both cases.

An exact rigid rotation is not exactly a uniform tangent-plane shift; the two
differ at second order in field angle. Measured over a 17x17 pixel grid across
each full frame, worst case over eight offset directions, at **50 pixels of
total boresight displacement**, comparing ``M`` applied to the ``oops``
``OffsetFOV`` line of sight against the unmodified field of view:

.. list-table::
   :header-rows: 1
   :widths: 34 22 22 22

   * - Instrument
     - Worst residual (rad)
     - In tangent-plane px
     - In pixel space
   * - Cassini NAC
     - 6.01e-9
     - 1.00e-3 px
     - 1.24e-3 px
   * - Cassini WAC
     - 5.91e-6
     - 9.89e-2 px
     - 7.86e-2 px
   * - New Horizons LORRI
     - 1.62e-8
     - 8.15e-4 px
     - 1.23e-3 px
   * - Galileo SSI
     - 1.82e-8
     - 1.79e-3 px
     - 1.79e-3 px
   * - Voyager 2 NAC
     - 1.29e-8
     - 1.64e-3 px
     - 1.64e-3 px

The residual is **linear in the offset**, not quadratic: measured across
12.5 / 25 / 50 / 100 px of displacement, each doubling multiplies it by 2.034,
2.067 and 2.129, on the NAC and the WAC identically. The term is second order in
*field angle* and first order in the offset. Two consequences for anyone
changing this code: quoting a residual without the offset it was measured at is
meaningless, and halving an offset buys only half the headroom a quadratic
reading would promise. The wide angle camera is the case to watch -- at a 50 px
offset this term alone is 9.89e-2 px of total displacement, against a round-trip
target stated as 0.1 px per axis.

Step 3: express both attitudes in the SPICE convention
------------------------------------------------------

With ``C_oops`` the observation frame's J2000-to-camera matrix at the midtime,
evaluated from the observation's own frame object::

    cmatrix_original = pxform('J2000', camera_frame, et_mid)
    R                = C_oops . cmatrix_original^T
    cmatrix          = (R^T . M . R) . cmatrix_original

Dropping the conjugation is the error the preceding section describes. The
conjugation's *direction* cannot be pinned by any real frame, because every
``R`` in the table is diagonal and therefore its own inverse; a synthetic
non-involutory ``R`` is what the test suite uses to hold ``R^T M R`` apart from
``R M R^T``.

Voyager takes a different path through the same function. ``oops`` builds the
observation frame as
``P . ckgp(ck_id, sce2t(scid, et_mid), 800 + texp/48, 'J2000')`` with
``P = pxform('VGn_SCAN_PLATFORM', camera_frame, 0)`` -- frozen,
time-independent, and tolerance-snapped, so a ``pxform`` at the midtime does not
reproduce it. For Voyager the observation frame attitude already is the SPICE
camera attitude that was navigated: ``cmatrix_original = C_oops``, ``R`` is the
identity by construction, and the writer has to reproduce the baseline with the
same snapped lookup. The tick conversion is ``sce2t``, not ``sce2c``; the two
differ, and the reproduction step matches the call it reproduces.

The result is checked before it is recorded: ``|det(C) - 1| < 1e-9``,
``max|C C^T - I| < 1e-9``, and every element finite, all three raising on
violation. A non-proper rotation here is a defect, not something to
orthonormalize away. The finiteness check is not redundant with the other two:
``NaN`` fails every inequality, so a ``NaN`` matrix passes both tolerance guards
silently, and the metadata writer emits these matrices unrounded -- bypassing
the helper that maps non-finite floats onto the JSON sentinel -- so an unchecked
``NaN`` would reach the file as a bare token that is not valid JSON.

Where it is computed, and how it is wired
=========================================

:func:`~spindoctor.support.cmatrix.compute_pointing` takes the observation, the
offset and the fitted-rotation flag, and returns a
:class:`~spindoctor.support.cmatrix.PointingSolution`: an
:class:`~spindoctor.support.cmatrix.AttitudeBaseline` carrying the uncorrected
matrix, the measured ``R``, the frame identities and the exposure times, plus
the corrected ``cmatrix`` -- which is ``None`` when the navigation produced no
offset or fitted a camera rotation. It returns ``None`` altogether for a host
whose SPICE frames it does not know, which is what a simulated image is.

:class:`~spindoctor.nav_orchestrator.nav_result.NavResult` is constructed inside
the ensemble, which never sees the observation, and it is a frozen dataclass.
So the wiring is a ``dataclasses.replace`` where the observation is in hand:
the result gains an optional ``pointing`` field, populated by
:meth:`~spindoctor.nav_orchestrator.orchestrator.NavOrchestrator.with_pointing`,
and the curator serializes it (see :doc:`dev_guide_orchestrator_curator` for the
JSON shape).

The stamping site is
:meth:`~spindoctor.nav_orchestrator.orchestrator.NavOrchestrator.navigate`
rather than the pipeline it calls. The pipeline has five early failure returns
and ``navigate`` has two more of its own -- the hard-failure image-class
short-circuit, which never enters the pipeline at all, and the contract-error
path -- so stamping the pipeline's final return alone would leave the
uncorrected matrix and the exposure times off every failed result. All three of
:meth:`~spindoctor.nav_orchestrator.orchestrator.NavOrchestrator.navigate`'s
returns route through
:meth:`~spindoctor.nav_orchestrator.orchestrator.NavOrchestrator.with_pointing`
instead.

That method is public because the manual-navigation driver needs it.
:func:`~spindoctor.nav_technique.nav_technique_manual.run_manual_nav` builds its
:class:`~spindoctor.nav_orchestrator.nav_result.NavResult` directly from the
operator's pick and never calls
:meth:`~spindoctor.nav_orchestrator.orchestrator.NavOrchestrator.navigate`, so
it calls
:meth:`~spindoctor.nav_orchestrator.orchestrator.NavOrchestrator.with_pointing`
itself. Operator-ratified offsets are the highest-quality pointing in the
corpus; leaving them unstamped would make them the one subset excluded from
every generated kernel.

``with_pointing`` absorbs :exc:`~spindoctor.support.exceptions.NavPointingError`
and nothing else. A pointing solution is recorded metadata rather than the
navigation itself, so an attitude the environment cannot supply is reported and
the field is left unset; no wrong C-matrix is ever recorded. The narrow catch is
the point: :exc:`~spindoctor.support.exceptions.NavPointingError` is what the
computation raises for every failure it expects -- its own guards, and the
frame, kernel and clock lookups SPICE cannot answer, each converted at the call
site with ``raise ... from`` so the original traceback survives. Catching the
untyped :exc:`LookupError` / :exc:`OSError` / :exc:`RuntimeError` /
:exc:`ValueError` family
instead would make a defect inside the computation indistinguishable from an
expected SPICE failure, and would quietly drop pointing from a 50,000-image
batch while every image still reported ``status=success``.

Anything that degrades or omits a solution goes to **both** logs: the detail to
the image log, one line to the run log. A registered instrument that reaches
navigation with no entry in the frame table is a build defect and warns to both;
a simulated image, which has no spacecraft and no furnished camera frame, is
expected and logs at debug.

The readers
===========

The metadata readers -- the backplane stage and the mosaic drivers -- consume
the recorded attitude through
:func:`~spindoctor.cli.reproj.offsets.apply_pointing_to_obs`, the consumer
entry point that owns selection, the offset fallback, the pool no-op, reason
reporting and cache clearing. The matrix mechanism beneath it is one public
function beside :func:`~spindoctor.support.cmatrix.compute_pointing`:
:func:`~spindoctor.support.cmatrix.apply_cmatrix_to_obs` -- calling it
directly bypasses everything the consumer surface owns. It inverts Step 3 of
the derivation above rather than re-deriving anything: the corrected frame is,
by the writing half's own construction, *the frame in which the unmodified
field of view holds*, so the reader replaces the observation's frame and
leaves the FOV alone::

    if cmatrix == cmatrix_original (np.array_equal):
        obs.set_frame(oops.frame.Cmatrix(C_oops(mid)))  # short-circuit
    else:
        R_hat = C_oops(mid) . cmatrix_original^T        # measured, gated
        obs.set_spice_cmatrix(cmatrix)

with ``C_oops(mid)`` read from the observation's own frame at the midtime. In
the ordinary case ``oops`` composes the declared ``R`` onto the recorded matrix
itself, which is the same attitude ``R_hat . cmatrix`` would give, to within the
tolerance the gate has just enforced. The ``array_equal`` short-circuit is the
one case that does not go through the setter: it mirrors the writer's identity
guard, and makes an identity correction reproduce the observation's own midtime
attitude exactly. Two float64 matrix products do not cancel to bit precision, so
composing ``R`` onto the recorded baseline instead would make "no correction
means no change" false at the 1e-16 level.

The replacement frame is built with no frame ID, so nothing is registered under
a name a later lookup can collide with. It is not free of consequence, though:
``oops`` caches every ``Cmatrix`` by its matrix, so each image re-pointed to a
distinct attitude leaves one wayframe and two frame-cache entries behind for the
life of the process. That growth is pinned by a test rather than left to be
discovered as a memory report from a full-catalog run.

``R_hat`` is measured for one reason only: the gate. Before anything is
applied, in order:

1. Both matrices must be proper rotations of real numbers, and the recorded
   ``midtime_et`` finite (else ``malformed_pointing``). The observation's
   host must have a frame mapping to gate against (else
   ``cmatrix_unknown_host`` -- unreachable for a record the writer produced,
   since writing a pointing block required the mapping).
2. The midtime gate: ``|obs.midtime - midtime_et| <= 1e-6 s``. A mismatch
   means the record is not this observation's
   (``cmatrix_foreign_midtime``).
3. The flip gate: ``max|R_hat - R_expected| <= 1e-9``, with ``R_expected``
   the ``spice_to_frame`` rotation the host declares. Because ``R_hat`` mixes
   the observation's *current* attitude with the *recorded* baseline, this
   one inequality fails on a changed kernel pool, a transposed
   ``cmatrix_original`` or whole record (a transposed rotation is still a
   proper rotation, so validation alone cannot catch it), and a changed host
   convention alike. The one sub-case it cannot see is a transposed
   ``cmatrix`` alone, which no single-serializer defect produces: the
   inequality contains only ``cmatrix_original``.
4. On flip-gate failure, one cheap probe before concluding corruption:
   ``max|C_oops(mid) - R_expected . cmatrix| <= 1e-9``. If it holds, the
   furnished pool **already answers the corrected attitude** -- corrected
   kernels furnished at load time. The correct action is to apply nothing:
   the observation is already right, and either fallback would corrupt it by
   roughly twice the offset. This is the distinguished
   ``POOL_ALREADY_CORRECTED`` outcome, counted under its own reason.
5. Only when neither explanation fits is it ``cmatrix_baseline_mismatch``:
   the record is never applied, and the caller degrades to the offset path,
   which reproduces the pre-C-matrix product exactly.

The selection and fallback ladder is shared, not duplicated:
:func:`~spindoctor.cli.reproj.offsets.select_pointing` classifies an
already-parsed metadata record into a
:class:`~spindoctor.cli.reproj.offsets.PointingSelection` (the mechanism, the
values, and the per-reason short form), and
:func:`~spindoctor.cli.reproj.offsets.apply_pointing_to_obs` applies it,
returning an :class:`~spindoctor.cli.reproj.offsets.AppliedPointing` naming
what the observation now carries (``cmatrix``, ``pool``, ``offset`` or
``none``). The offset path is the documented mechanism for every record with
no usable C-matrix: a fitted-rotation result
(``no_cmatrix_rotation_fitted`` -- the mechanism, not a mission), a record
with no pointing block (``no_pointing_block``), and a malformed pointing
block (``malformed_pointing``, warned to both logs like the gate refusals).
Every caller treats a record that supplies no pointing the same way: the
product is built on uncorrected pointing, the shortfall is reported, and the
reason is counted.  A caller that refused a record class the others processed
would build one product from a document and another from the index row the
same document was ingested into, since a column pair holds every way an offset
can fail to be a pair.

After either mechanism mutates the observation,
:func:`~spindoctor.cli.reproj.offsets.apply_pointing_to_obs` calls
``obs.reset_all()``. "Apply before any geometry is computed" is not an
invariant :meth:`~spindoctor.obs.obs_inst.ObsInst.from_file`
leaves available: :class:`~spindoctor.obs.obs_snapshot.ObsSnapshot` runs the
closest-planet scan while it is constructed, which builds and caches an
:class:`oops.backplane.Backplane` against the uncorrected frame before any
caller can apply anything, and nothing pins that cache's consumers to
rotation-invariant quantities. The reset clears every cached
:class:`oops.backplane.Backplane` and :class:`oops.meshgrid.Meshgrid` so all
downstream geometry is built on the corrected observation.

Two boundaries worth stating. First, the recorded ``cmatrix`` is a midtime
attitude, so the replacement frame is constant across the exposure where the
original frame was time-varying; every switched consumer is a midtime
evaluation (oops ``Snapshot`` evaluates every ``Backplane`` at the scalar
midtime, and the ring reprojection constructs its events at ``obs.midtime``
explicitly), so nothing changes -- but the replacement frame's transform also
carries **zero angular velocity** where the original carried the
spacecraft's. No switched consumer reads frame omega; a future
velocity-aware backplane (smear planes) must not consume it from the
replaced frame. Second, the two paths agree exactly only at the boresight,
where the correction was constructed; away from it they differ at second
order in field angle and first order in the offset, because the *offset*
path is the approximation. The acceptance evidence
(``tests/integration/test_cmatrix_readers.py``) measures that bound in its
own metric -- ``K_inst``, the worst pixel-space residual via ``uv_from_los``
inversion over a 17x17 grid and eight offset directions at 50 px of
displacement on each frame's own FOV -- and holds the two paths within
``2 K_inst |offset| / 50 + 0.005 px`` at the LOS grid and through one ring
reprojection, one body reprojection, and one end-to-end backplane run, with
every measured residual additionally pinned at measured-plus-margin.

The writer
==========

The writer package is :mod:`spindoctor.cli.ck`, and ``sd_create_ck``
(``src/spindoctor/cli/sd_create_ck.py``) is the driver that sequences it. One
module per question:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Module
     - Responsibility
   * - :mod:`spindoctor.cli.ck.pointing`
     - Reads one image's recorded ``pointing`` and ``times`` blocks into
       :class:`~spindoctor.cli.ck.pointing.ImagePointing`, the single input
       type the segment writer accepts, refusing anything malformed rather
       than coercing it.
   * - :mod:`spindoctor.cli.ck.images`
     - Judges eligibility from an image's own record
       (:class:`~spindoctor.cli.ck.images.ImageEntry`,
       :class:`~spindoctor.cli.ck.images.OmissionReason`) and applies the
       simultaneous-exposure rule that needs more than one image.
   * - :mod:`spindoctor.cli.ck.inputs`
     - Gathers what a run reads before it can write: the metadata documents,
       the kernel directories, the time filter, and the supporting kernels to
       furnish.
   * - :mod:`spindoctor.cli.ck.clocks`
     - Chooses the spacecraft clock kernel a run's time tags are encoded with.
   * - :mod:`spindoctor.cli.ck.frames`
     - Refuses a pool in which two frame kernels define one frame the run's
       images name.
   * - :mod:`spindoctor.cli.ck.index`
     - Scans the kernel directories once per run
       (:func:`~spindoctor.cli.ck.index.build_ck_index`), recording which
       objects each C-kernel describes and over what epochs.
   * - :mod:`spindoctor.cli.ck.assignment`
     - Pairs each image with the original kernel it navigated against, by
       reproduction, and groups the assignments by output file.
   * - :mod:`spindoctor.cli.ck.segment`
     - Builds one type-3 segment
       (:func:`~spindoctor.cli.ck.segment.build_segment`) and writes it.
   * - :mod:`spindoctor.cli.ck.kernel_file`
     - Opens, fills and closes one corrected kernel, comments and all.
   * - :mod:`spindoctor.cli.ck.comments`
     - Renders and attaches the comment area.
   * - :mod:`spindoctor.cli.ck.metakernel`
     - Renders the meta-kernel that furnishes a corrected set in order.
   * - :mod:`spindoctor.cli.ck.report`
     - Renders the per-mission CSV.

A run proceeds in that order: read the documents, filter by time, furnish the
leapseconds, frame and clock kernels the documents name, index the candidate
C-kernels, assign each image to a baseline, build every corrected file, and only
then write them, followed by the meta-kernel and the report.

**Building and writing are separate phases**, and nothing reaches the
filesystem in the first. A refusal while building therefore leaves the output
directory untouched, rather than some kernels written, no meta-kernel and no
report -- the one artifact that says what is in the files that did get written,
and whose absence leaves a partial set that the refusal to overwrite an existing
corrected kernel then blocks a rerun on. The cost is that every segment of the
run is resident at once: 678 bytes for the three records an ordinary exposure
carries, measured on this writer's own segments, so about 34 MB for a
50,000-image batch, against the per-image metadata such a batch already holds.

**The writing phase judges before it writes, and judges the whole set.**
Separating the phases is not enough on its own: a destination the run cannot
write is a property of the output directory, not of anything the build sees, so
a run whose second output path is occupied would build cleanly and then refuse
after the first file was already on disk -- the same partial set, reached a
different way. So :func:`~spindoctor.cli.ck.kernel_file.check_output_paths`
judges every destination together and
:func:`~spindoctor.cli.ck.kernel_file.check_ck_file` judges each file's
contents, both before the first ``ckopn``. The refusal names every path that
failed rather than the first, so a set is cleared in one pass.

The per-file refusals stay where they are.
:func:`~spindoctor.cli.ck.kernel_file.check_ck_file` is what
:func:`~spindoctor.cli.ck.kernel_file.write_ck_file` itself calls, so a direct
caller is held to the same rules and
the two cannot drift; the set-level check adds only what no per-file check can
see -- one path named twice, and a directory judged once for all of them.

**None of this is atomicity, and neither the code nor this guide claims it.**
What the checks establish is that nothing knowable in advance stops the writing
of the corrected kernels part way through. Three things stay outside that, and
no check made beforehand can reach them: space on the device, a path or a
permission that changes between the check and the write, and a record set
``ckw03`` refuses once the file is open. A file failing that way is closed and
removed; the files written before it are not, and the meta-kernel and the report
are never reached. That residual window is recorded in the docstring of
``write_output_files`` in ``sd_create_ck`` and in the user guide,
which tells an operator that this is the failure needing the output directory
cleared before a rerun.

The boundary is drawn at the corrected kernels on purpose. The meta-kernel's own
rules -- no quote, no trailing ``+``, no trailing blank in a path -- are
knowable in advance for the corrections, and are deliberately *not* checked with
the output paths, because the meta-kernel also names every original, whose paths
come from ``--kernel-dir`` and are not in the output set at all. Checking half
of what it will refuse would read as having closed a window that is still open.
So that failure keeps its own shape: it lands after every corrected kernel is
written, leaves a complete and self-consistent set of them, and takes only the
meta-kernel and the report with it. The user guide names it beside the three
above.

Assignment is by reproduction, not by the recorded kernel list
--------------------------------------------------------------

An image's provenance records the SPICE kernels that were furnished when it
navigated, but it records **sorted basenames only**: no load order, no
directory, and in a batch run the list accumulates every kernel earlier images
needed, so it is a superset of the ones this image used. Several of those names
can describe the right object over the right epochs while only one is the file
the navigation actually read.

So the pairing is measured. Each candidate the index offers is furnished on its
own, with nothing else covering the same object, and asked for the attitude the
image navigated against; a candidate whose answer reproduces the recorded
``cmatrix_original`` to within **1e-9 radians** supplied that baseline, and one
whose answer does not, did not. That is a reproduction bound rather than a
navigation bound: the same kernels evaluated the same way agree to
floating-point noise, so anything larger is a different kernel. The angle is
taken through a quaternion rather than through the matrix trace, which loses
half its digits as the angle goes to zero.

The same mechanism is the detector for a kernel set that changed since
navigation ran. An eligible image that no candidate reproduces is reported
``no_reproducing_baseline`` and receives no segment, because a correction
measured against a baseline that no longer exists is worse than no correction.

Three failure modes would otherwise arrive disguised as that verdict, and each
is refused before any candidate is tried, naming what is missing:

* A camera frame or CK object frame the images name that the pool does not
  define. A frame kernel that was never furnished defeats the same lookup for
  every image alike, so a run that forgot it would write nothing, blame the
  holdings, and say so once per image.
* Two frame kernels defining one of those frames, or two versions of one
  spacecraft clock kernel. A text kernel's last assignment wins, so the corpus
  would be reproduced entirely against whichever version sorted last.
* An object whose spacecraft clock no furnished kernel defines. The index
  reports coverage in TDB, which needs that clock; such an object is recorded as
  unreadable and offers no coverage, which would make every image correcting it
  look like drift. Recording it rather than refusing the scan is deliberate: a
  real New Horizons kernel names object -1 beside -98000, and refusing there
  would make the whole mission unindexable for the sake of an object no image
  ever asks about. The driver warns once in its run log, naming the skipped
  objects, so a kernel set missing a clock says so even when no image needs it.

When several candidates reproduce -- which the holdings make ordinary, since
reconstructed, gapfill and predicted sets overlap -- the tie-break prefers
reconstructed over gapfill over predicted (read from the kernel's basename), then
the lexicographically greatest basename, then the greatest path. The reproducing
candidates agree on the attitude by construction, so the tie-break decides only
which output file carries the segment; all it has to be is deterministic.

A corrected kernel is never indexed as a candidate. Writing the corrections back
beside the originals is the natural workflow, and a corrected kernel reproduces
its own baseline exactly wherever the correction was the identity, so indexing
one would offer a correction as the next run's baseline.

Building the segment
--------------------

The correction is measured at the camera; the kernels that exist describe a bus
or a scan platform, and that is where it has to be written. With
``F = pxform(ck_frame, camera_frame, midtime)`` -- always computed, never
assumed, because Cassini's ``F`` is a permutation-like matrix nowhere near the
identity::

    C_ck_corrected(mid) = F^-1 . cmatrix
    delta               = C_ck_corrected(mid) . C_ck_original(mid)^T

with ``C_ck_original`` read from the baseline kernel the caller has furnished.
Across the exposure the correction is held **body-fixed**::

    C_ck_corrected(t) = delta . C_ck_original(t)      for t in [start, stop]

which is the physical model: the spacecraft is pointed slightly wrong and the
error turns with it, so the correction is right at every epoch in the window
even though what the segment *reproduces* between its records is only as good
as the interpolation between them (the measured interior error is in
:doc:`/user_guide/user_guide_ck_kernels`).

Voyager is the exception. Its navigated attitude came from a frozen,
tolerance-snapped lookup rather than a time-varying frame chain, so its segment
carries the single corrected attitude, constant across the window; writing
time-varying pointing there would disagree with what was navigated.

Records go at the exposure start, midtime and stop, plus a one-second cadence
once the window reaches ten seconds, each encoded with ``sce2c``. Both the
decision to add interior records and their count are taken from the same
quantity, the span from start to stop, so the two cannot answer for different
exposures; the count is capped, because the arithmetic that expands a span has
no bound of its own and a recorded span of 1e9 s would exhaust memory before
anything noticed the epochs were not an exposure. Time tags must be
strictly increasing in encoded SCLK; a tag that does not increase is dropped,
and epochs that all encode to one tick yield a single record at the midtime.
Because ``sce2c`` encodes a *fractional* tick, that last path needs three epochs
equal as float64 -- a nanosecond exposure near ET 5e8 -- and is unreachable for
any real exposure; a 1 ms exposure against a 1/256 s clock is 0.256 ticks and
still produces three records. It is a guard, not a case the corpus contains.

Records are quaternions from ``m2q`` with sign continuity enforced: ``m2q``
fixes the scalar component non-negative, which flips the sign between adjacent
records whenever the attitude's rotation angle passes 180 degrees, so each
record is negated as needed to keep a non-negative dot product with its
predecessor. SPICE's own type-3 reader restores the sign when it interpolates,
so no read-back can see the difference and the test that guards this asserts on
the written records. The enforcement stays because the file should say what it
means.

Why angular velocity is copied verbatim, never rotated, and never left out
--------------------------------------------------------------------------

This is the part most likely to be "fixed" into wrongness, because the wrong
treatment is the one that looks thorough.

CK angular velocity is expressed in the segment's **base reference frame**
(J2000 here), per the CK Required Reading -- not in the frame the segment
describes. The corrected frame differs from the original by a constant
body-fixed rotation, and two frames rigidly attached to each other have
identical angular velocity in the base frame. So the corrected records carry the
original's vectors bit-identically, with ``avflag = 1``.

Rotating those vectors through ``delta`` -- the superficially thorough
treatment -- would express them in neither the original frame nor the corrected
one. It writes a vector in no frame at all, and nothing downstream reports it:
the file loads, ``ckgpav`` answers, and the rates are wrong by the size of the
correction. A test pins the correct direction and fails if the rotation is
introduced.

The second trap is the mirror of the first: ``avflag = 0`` looks like the
honest thing to write for a segment whose rate is unknown, and it is the one
value that hides the correction. SPICE does not read such a segment as "attitude
here, rate unknown" -- it skips the segment outright for ``ckgpav`` and for
``sxform`` and answers from the next loaded kernel that does carry a rate for
the same object and epoch, which is the original, with the original's
uncorrected attitude. Measured on a real reconstructed Cassini kernel
(``04002_04009ra.bc``) with a 1.000e-04 rad correction and ``avflag`` the only
variable:

.. list-table::
   :header-rows: 1
   :widths: 34 33 33

   * - Corrected segment
     - ``ckgp`` versus ``ckgpav``
     - ``sxform`` versus corrected
   * - ``avflag = 0``
     - 1.000e-04 rad
     - 1.000e-04 rad
   * - ``avflag = 1``
     - 0.000e+00 rad
     - 0.000e+00 rad

That is decisive because ``oops`` reads pointing through ``sxform``:
``oops/frame/spiceframe.py`` defaults ``omega_type='tabulated'`` and calls
``cspyce.sxform`` in that mode, and no ``oops`` host overrides it. An
``avflag = 0`` corrected segment would deliver its correction to ``ckgp`` and
``pxform`` and withhold it from ``sxform``, with which of the two a consumer saw
depending on what else was in their pool. Exposure across the local baselines:
every Cassini -82000 segment (2645) and every New Horizons -98000 segment (4346)
carries a rate, so those are immune; of Galileo -77001's 150 segments, 38 carry
none; Voyager's nine carry none at all.

So the writer applies **all records or none, and none means refuse**. A baseline
that carries no rate is reported by ``ckgpav`` exactly as pointing that is not
covered at all is -- ``found`` false, one answer for two conditions that mean
opposite things here, since one refuses the run and the other omits an image.
The sampling pass over every record decides; when it comes back empty-handed, a
second pass reads attitude alone, so a genuine coverage gap surfaces as itself
(:exc:`~spindoctor.cli.ck.segment.BaselineCoverageGapError`) and an exposure
that lacked only its rate is refused with a :exc:`ValueError` naming that.
Both lookups are made through the flag-returning form of the SPICE call rather
than the raising one, so "nothing here", which is a question about one exposure,
is told apart from a pool with no C-kernel furnished at all, an undefined
reference frame, or a kernel that cannot be read, each of which still raises and
still stops the run. Choosing the form per call does not touch the process-wide
``use_errors`` regime. Refusal rather than zeros
for the records that lack a rate: the attitude would be right, but a platform
genuinely parks, so an invented zero is indistinguishable from a measured one,
and the overlay would start answering ``sxform`` at epochs where the pool has no
answer at all.

A frozen (Voyager) segment writes zeros with ``avflag = 1``. Its attitude is
constant, so zero is that attitude's true angular velocity -- a measurement
rather than an invention -- and writing it is what makes ``sxform`` return the
corrected attitude. The baseline's own vectors are still not copied there, since
the rigid-attachment argument does not hold for a segment that deliberately
drops the baseline's time variation.

Writing the file
----------------

``ckopn`` reserves the comment area and ``ckw03`` writes segments, but nothing
in the CK interface fills the comments; that happens through the plain DAF
interface (``dafopw`` / ``dafac`` / ``dafcls``) on the file the CK interface has
already closed. :func:`~spindoctor.cli.ck.kernel_file.write_ck_file` runs the
two halves in the one order that works, and three measurements against the
installed toolkit shape what it may write:

* **The comment area is grown when it has to be.** A ``dafac`` that overflows
  what ``ncomch`` reserved neither fails nor truncates: SPICE extends the area
  by shifting every data record in the file, and the comments read back
  complete. So the reservation buys a file that is not rewritten rather than a
  comment that is not lost, and the only observable is the address of the first
  data record -- which is what
  :func:`~spindoctor.cli.ck.kernel_file.first_data_record` exists to let a test
  assert.
* **A comment line longer than 255 characters is stored and then cannot be read
  back at all.** ``dafec`` reads into a 255-character buffer and raises on the
  first line that overflows it, which loses the whole comment area rather than
  one line's tail.
* **Trailing whitespace does not survive, and a character outside printable
  ASCII is refused outright** -- ``dafac`` raises on it after the segments
  have been written, leaving a file without the comment area it was meant to
  have. Python's ``isprintable`` is not the test: an accented letter is
  printable to Python and refused by SPICE all the same. The one free-length
  field, the status reason, is elided to fit its line rather than allowed to
  refuse the file.

All three are refused before a file is opened. A write that fails part way
through is closed with ``dafcls`` rather than ``ckcls`` and the half-written
file is removed: ``ckcls`` on a C-kernel that received no segment raises in its
own right *and* leaves the file open, so it would replace the failure worth
reading and leak the handle as well.

The meta-kernel has its own constraint. A SPICE text kernel holds at most 80
characters per string value and truncates a longer one silently, so every path
-- and every path in a holdings tree is longer than that -- is written through
SPICE's ``+`` continuation as several strings. A path ending in ``+`` cannot be
expressed at all and is refused by name; so is a path ending in a blank, which a
text kernel trims. Joins are chosen to fall on non-blanks, because the holdings
tree holds names with spaces in them and a name that lost a character in the
middle would reach the consumer as a file it never asked for. The joined path
is bounded too: SPICE accepts at most 255 characters in a file name, and the
continuation scheme could spell a longer one into the meta-kernel without
complaint -- which every consumer's ``furnsh`` would then refuse, after the
kernels listed before it had already loaded. A path over that limit is refused
at writing time instead.

Invariants
==========

* **Units and conventions.** Epochs are TDB seconds past J2000 throughout;
  segment time tags are encoded SCLK from ``sce2c``. Both C-matrices are
  J2000-to-camera in the SPICE convention at the exposure midtime. The metadata
  ``offset`` is ``[dv, du]`` and ``oops.fov.OffsetFOV`` takes ``(du, dv)``.
* **Both attitudes are evaluated at the exposure midtime.** A midtime/start
  mix-up in the reproduction step fails every baseline at the 1e-9 rad bound, so
  an unexplained across-the-board ``no_reproducing_baseline`` should be checked
  against this first.
* **The caller owns the kernel pool.** The assignment step refuses to run with
  any C-kernel furnished (``ktotal('CK')``), since a stray one answers the same
  lookups as the candidate under test; the segment builder requires the
  supporting kernels and the one baseline to be furnished before it is called.
* **The spacecraft clock table is the resolver; ``ckmeta`` is the cross-check.**
  ``ckmeta`` computes rather than validates -- it answers -999 for the
  nonexistent object -999999 and raises for neither -- so both call sites return
  the value recorded in :data:`~spindoctor.spice_ids.CK_OBJECT_SCLK_ID` even
  though the check has just proved the two equal. Weakening the check later
  cannot quietly promote ``ckmeta`` back to being the source.
* **The cspyce error regime is process-wide.** ``cspyce.use_errors()`` /
  ``use_flags()`` is shared with ``oops``; the writer assumes the exceptions
  regime, which is the package default, and never flips it.
* **The omission-reason set is closed, and every member is one a run emits.**
  Adding a reason is a schema change for every consumer of the report, and a
  reason no run can produce is worse than a missing one, since it asks every
  consumer to write dead code. So a failure with no reason of its own stops the
  run instead: an unreadable navigated image, a baseline supplying angular
  velocity at only some records, and a window too long for a segment's records.
  Those report to the run log only -- they end the run, and the per-image log
  they would otherwise use is opened by the reporting pass that never runs.
* **A baseline that reproduced and then could not answer is an omission, not a
  refusal.** The pairing is made at the exposure midtime and the segment has
  records at the exposure start and stop as well, so an exposure straddling the
  end of a baseline's coverage is reachable and ordinary; it takes
  ``baseline_coverage_gap``, a reason of its own, and not
  ``no_reproducing_baseline``, whose whole value is that it means the holdings
  changed since navigation ran. The image is left out of its file, the file is
  written without it, and a file that loses every image is not written at all.
* **A run writes every corrected kernel or none of them, for every reason it
  can know in advance.** Adding a refusal that a write could hit means adding
  it to :func:`~spindoctor.cli.ck.kernel_file.check_output_paths` or
  :func:`~spindoctor.cli.ck.kernel_file.check_ck_file` -- never to
  :func:`~spindoctor.cli.ck.kernel_file.write_ck_file` alone, which by then
  has files behind it. The guarantee is
  bounded on purpose: it says nothing about a write that fails for a reason
  only the filesystem or ``ckw03`` knows at the moment of writing.
* **"Is anything already there?" is not ``Path.exists``.** It follows symbolic
  links, so a link with no target answers ``False`` and the write then creates
  the target -- outside the output directory. A link that resolves in a loop
  answers ``False`` for a different reason and behaves the same way. Both are
  pinned as measurements in the writer's tests, so a rewrite back to ``exists``
  fails rather than passes. For the same class of reason, a path holding a
  character outside printable ASCII is refused rather than tested:
  ``os.path.lexists`` answers ``False`` for a name holding a null byte, SPICE
  is handed the name as a C string, and the comment area and meta-kernel that
  must repeat the name back accept only printable ASCII.

Adding a mission
================

A mission needs four things before its images can carry a corrected attitude,
and the first two are the ones most easily missed.

1. **A frame identity** in :mod:`spindoctor.support.cmatrix`: the SPICE camera
   frame name, the CK object a corrected kernel targets, and the constant
   rotation ``R`` between the ``oops`` observation frame and the SPICE camera
   frame. Without it, no image of that instrument records a corrected attitude,
   and the orchestrator warns to both logs when one reaches navigation.
2. **A row in** :data:`~spindoctor.spice_ids.CK_OBJECT_SCLK_ID` pairing that CK
   object with its spacecraft clock. The clock is not derivable from the object
   by arithmetic: ``-31100 // 1000`` is -32 in Python, which is the other
   Voyager.
3. **An entry in the driver's mission list** (``MISSIONS`` in
   ``src/spindoctor/cli/sd_create_ck.py``), spelled there rather than read from
   the observation registry, which would drag every host class into a program
   that needs only their names.
4. **An integration assertion** that the measured ``R`` equals the constant the
   frame table claims, on a real frame. That is the one check hermetic tests
   cannot make, and the flip is the dominant risk in this subsystem.

An instrument whose observation frame is frozen from a tolerance-snapped lookup,
as Voyager's is, needs more: its CK object joins the frozen-attitude set, which
makes its segments carry one constant attitude and a zero rate, widens its
coverage in the index by the snapped lookup's tolerance, and routes its
reproduction through the same snapped lookup rather than through ``pxform``.

Testing
=======

The suite is deliberately layered, because the frame conventions and the
kernel-writing mechanics fail in different ways.

* **Hermetic writer tests** (``tests/spindoctor/cli/ck/``) write their own
  minimal leapseconds, clock and frame kernels as small text files, plus an
  original C-kernel produced by the writer's own primitives, so no holdings are
  needed. They cover the segment records, the angular-velocity policy, the
  quaternion sign repair, the comment area and meta-kernel rendering, the
  assignment tie-break, and the report.
* **Frame integration tests** (``tests/integration/test_cmatrix_frames.py``)
  measure ``R`` on real frames of each instrument and check it against the
  table, and confirm it is constant across the exposure.
* **The round trip** (``tests/integration/test_ck_round_trip.py``, driven by
  ``tests/integration/ck_round_trip.py``) navigates a real image, generates the
  corrected kernel, and re-navigates in a fresh process with that kernel
  furnished. Three separate processes, because ``oops`` caches frames and
  manages its own kernel pool: a ``furnsh`` in the middle of a process that has
  already navigated is not guaranteed to take effect, and a round trip that
  quietly measured the uncorrected pointing twice would pass. The re-navigation
  first asserts that the pointing actually changed, which distinguishes "the
  kernel took effect" from "the kernel was silently buried" -- something the
  re-measured offset alone cannot do.

Two facts from the round trip are worth carrying into any future work on it.
The corrected kernel must be furnished *after* the host's ``from_file`` returns,
since that is when ``oops`` has finished furnishing the originals and SPICE
gives precedence to the C-kernel furnished last. And a host that freezes its
observation frame during ``from_file`` -- Voyager -- cannot see a kernel
furnished after that call at all, so for those hosts the image is loaded a
second time with the correction already furnished.

A convention error and a technique's imprecision look different in the result,
which is what makes the round trip diagnostic rather than merely a pass or a
fail. A sign or conjugation error leaves roughly *twice* the original offset and
a read-back attitude that disagrees with what was recorded. Technique
non-equivariance -- a technique not returning exactly the negative of the shift
it was given -- leaves a fraction of a pixel with a read-back that is exact to
floating point.

API reference
=============

:mod:`spindoctor.support.cmatrix` and :mod:`spindoctor.spice_ids` appear under
:doc:`/api_reference`. The :mod:`spindoctor.cli.ck` package does not: it is part
of the command-line tree rather than the importable library API, it carries no
stability promise, and nothing outside ``spindoctor.cli`` imports it. Every
other ``spindoctor.cli`` subpackage stands the same way. Its entry points are
the modules listed in `The writer`_ above.
