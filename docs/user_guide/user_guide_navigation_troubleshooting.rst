===============
Troubleshooting
===============

A navigation run that finishes without crashing can still leave images unnavigated, and
the reason is almost always recorded. Start with the two places that hold it: the run's
log, which carries one line per image, and that image's own log, which carries the detail
of what every model and technique found. :doc:`/user_guide/user_guide_logging` explains
where both are written. The per-image metadata document records the same verdict in
machine-readable form, under the ``status``, ``status_reason``, ``status_error``, and
``status_exception`` keys described in :doc:`/user_guide/user_guide_metadata`.

The fastest first look is the summary picture the run writes beside each metadata
document. It draws the models over the image, so a model landing nowhere near the data is
visible at a glance. :doc:`/user_guide/user_guide_navigation_outputs` describes it and
says where it is written.

The symptoms below are the ones that come up in practice, each with what to check.

The run stops before it navigates anything
==========================================

The run stops with this message::

   One of --nav-results-root, the configuration variable
   "environment.nav_results_root", or the NAV_RESULTS_ROOT environment variable
   must be set

There is no built-in results root, so every run has to be told where to write. Supply it
by any one of the three the message names; the command-line option wins over the
configuration file, which wins over the environment variable. The backplane and bundle
results roots behave the same way, and their messages name their own three sources. The
log root is the exception: when none of its three is set, it falls back to a ``logs``
directory under the navigation results root. See
:doc:`/user_guide/user_guide_installation`.

An environment variable is set but has no effect
================================================

Check the spelling of the variable. Every results and log root is named ``NAV_..._ROOT``,
while the holdings variable is ``PDS3_HOLDINGS_DIR`` -- ``DIR``, not ``ROOT``, even
though the matching option is ``--pds3-holdings-root``. The full list, with the option
and configuration setting each one pairs with, is in
:doc:`/user_guide/user_guide_installation`.

A misspelled variable is not read and is not reported. Remember too that an exported
variable is consulted last: it does not override a value a configuration file already
sets.

The run stops on a missing PDS3 index label
===========================================

The run stops with a file-not-found error naming a path under ``metadata``, such as::

   File(s) do not exist: .../metadata/COISS_2xxx/COISS_2001/COISS_2001_index.lbl,
   .../metadata/COISS_2xxx/COISS_2001/COISS_2001_index.tab

The holdings tree has its ``volumes`` half but not its ``metadata`` half. Image selection
is answered from each volume's PDS3 index table, which lives under ``metadata``, so a tree
without that half cannot be enumerated at all. Retrieve the ``metadata`` tree for the
volumes you are processing, or point ``--pds3-holdings-root`` at a tree that has both
halves. :doc:`/user_guide/user_guide_installation` shows the layout.

Every image fails and the failure mentions SPICE
================================================

The image's metadata document records ``missing_spice_data``, or the navigation itself
reports ``kernels_unavailable``. SPICE kernels covering the image's observation time were
not available.

Check that ``SPICE_PATH`` names the kernel tree, that the tree is readable from the
machine running the program, and that the kernels there actually cover the mission phase
the image comes from. An image from outside the loaded coverage fails this way even when
every path is correct.

No images are processed at all
==============================

The run reports that no images matched the selection. The dataset name, the volume names,
or the image-name filters selected an empty set.

Run the same selection with ``--dry-run``, which reports which images would be processed
without navigating any of them. :doc:`/user_guide/user_guide_image_selection` describes
the selection options and how they combine. Also confirm that the holdings root is the
one you meant: it comes from ``--pds3-holdings-root``, then from the configuration, and
only then from the ``PDS3_HOLDINGS_DIR`` environment variable. An exported variable does
not override a value a configuration file already sets.

An image cannot be read
=======================

The metadata document records ``image_read_error``, or the navigation reports
``image_corrupt``. The image or label file could not be parsed.

Check that both the image file and its label are present and complete. For PDS3 holdings,
confirm that the volume was retrieved whole, and that the label beside the image is the
label for that image.

No features are found
=====================

The navigation reports ``no_features_extracted``, ``all_features_gated``, or
``body_fills_fov``. There was nothing in the frame the navigator could measure: a blank
sky, a body too small or too dim to show an edge, or a body whose disc covers the whole
field of view and hides everything behind it.

Check the image itself, and check which models were built for it. A frame that genuinely
contains no measurable feature has no answer to give. When the frame does contain
something you expected to be used, widen the model or technique selection with
``--nav-models`` and ``--nav-techniques``; see
:doc:`/user_guide/user_guide_navigation_models`.

Techniques run but nothing is accepted
======================================

The navigation reports ``no_feasible_techniques``, ``all_techniques_spurious``,
``final_confidence_below_threshold``, or ``final_sigma_above_threshold``. Measurements
were made, but none of them earned enough confidence, or the combined answer was too
imprecise to accept.

Read that image's log to see what each technique found and where it stopped. The
acceptance thresholds are configurable; see
:doc:`/user_guide/user_guide_configuration`. Lowering them makes more frames report an
answer and makes those answers less trustworthy, so prefer to understand why the evidence
was weak before changing them.

Techniques disagree
===================

The navigation reports a conflicted result: ``conflicted_techniques``,
``body_shape_lock_suspect``, or ``lone_blob_in_collapsed_regime``. Two or more
measurements pointed at different offsets, or one measurement was contradicted by another
on the same body.

The image's log names the competing measurements and their confidences. A conflict usually
means one technique locked onto the wrong thing. Restricting the run to the techniques you
trust for that kind of frame, with ``--nav-techniques``, will tell you which one.

The offset is reported along only one axis
==========================================

The navigation reports ``rank_1_only``, or refuses the frame with
``unobservable_offset``. The frame's features constrain the pointing in one direction
only. A single straight ring edge with nothing crossing it is the usual case: sliding the
image along the edge changes nothing measurable.

Nothing is wrong with the run. Such a frame needs a second, differently oriented feature
before a full two-axis offset can be recovered.

The instrument is not configured
================================

The navigation reports ``instrument_not_configured``. The camera that took the image has
no settings of its own. See :doc:`/user_guide/instruments/instruments` for the instruments
and cameras that are supported.

An internal error is reported
=============================

The metadata document records ``internal_error`` or ``contract_violation``, with the
exception text in ``status_exception`` and the full traceback in the log. This is a defect
in SpinDoctor rather than a problem with the image or the configuration.

Getting help
============

Report a problem on the issue tracker at https://github.com/SETI/rms-spindoctor/issues.
Include the command line you ran, the relevant part of the run's log, the failing image's
log, and the image's metadata document. Say which holdings root and which SPICE kernels
were in use. Those together are usually enough to reproduce the failure.

The tracker is public, so replace anything secret before you post. A
``--results-index-db`` URL can carry a database password: SpinDoctor hides it wherever it
writes the URL itself, but the command line you copy out of your own shell is not
something it can reach, so hide the password there yourself.
