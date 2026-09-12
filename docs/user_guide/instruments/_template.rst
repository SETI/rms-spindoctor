===========================
Instrument chapter template
===========================

Copy this file to ``<instrument>.rst`` in this directory and replace every
placeholder. The glob toctree in :doc:`instruments` picks the new chapter up
with no other file edited.

Every section below appears on every chapter, in this order. A section that is
empty for an instrument says so -- ``None.`` or ``Not supported.`` -- rather
than being left out, because a reader comparing two instruments needs the same
question answered on both pages. Nothing here is a comparison: a chapter states
what is true of its own instrument and never mentions another.

Do not restate the instrument team's own documentation. Apertures, focal
lengths, pixel scales, angular field extents and filter tables belong to the
official instrument and volume documents, which the References section points
at. What a chapter carries is what SpinDoctor itself decides, configures,
measures, or does for this instrument, described for this instrument alone.

Overview
========

One or two sentences of scope: mission, cameras, planetary system, epochs.
Orientation for a reader who has landed here, not a description of the
instrument.

Pipeline support
================

Navigation, corrected-pointing C-kernels, backplanes, mosaics, PDS4 bundles,
the simulator, and the statistics database: each supported, partial with the
reason, or unsupported.

Datasets and image selection
============================

Dataset names and aliases; supported volumes; the holdings subtree and the
index path; which product is actually navigated; the image-name forms accepted
on the command line and their case sensitivity; image numbering and any
range-selection caveat; cameras; instrument-specific CLI flags; grouping.
Illustrative invocations belong here.

Image data and units
====================

Data units; the saturation ceiling and the saturation policy; the missing-pixel
marker; the blank and noisy thresholds and what they do to classification;
corrections applied at load; which configured values are still provisional.

Field of view and geometry
==========================

Extended-FOV margins and why they vary with image size; whether camera rotation
is fitted and what that changes for the user; the measured twist and residual
distortion.

Metadata fields
===============

Which keys this instrument writes into ``_metadata.json``, and which common
keys it does not.

Corrected-pointing C-kernels
============================

The corrected object and what it physically is; the spacecraft clock; the
camera frames; which kernel directories to supply, with a worked invocation;
the baseline kernel naming and class conventions; angular-velocity availability
in the baselines; the segment shape; which omission reasons this instrument can
produce and why; the interpolation-error characterization.

Known limitations
=================

What a user should expect not to work, or to work less well, and why.

References
==========

The official instrument and volume documentation for this instrument.
