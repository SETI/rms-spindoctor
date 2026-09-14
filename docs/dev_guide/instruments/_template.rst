===========================
Instrument chapter template
===========================

Copy this file to ``<instrument>.rst`` in this directory and replace every
placeholder. The glob toctree in :doc:`instruments` picks the new chapter up
with no other file edited.

Every section below appears on every chapter, in this order. A section that is
empty for an instrument says so -- ``None.`` or ``Not supported.`` -- rather
than being left out. A chapter states what is true of its own instrument and
never mentions another; the shared subsystem chapters carry the mechanism.

Code map
========

The Obs class, the DataSet class or classes, the config block, the sim
instrument key, the statistics key, the log key, and the ``oops`` host module.

Loading the image
=================

The host entry point and the keyword arguments passed to it, with the reason
for each; what ``from_file`` does beyond the host call; how the config block is
selected; how the extended-FOV margin is resolved.

Label and index dependencies
============================

Label fields read and what breaks without them; the index time and camera
columns and the camera map; filespec parsing rules; image-number monotonicity
across volumes; case conventions.

Configuration block
===================

The block's shape (nested per camera, or flat); keys that depart from the
common schema; which values are still placeholders; the ``_sources`` convention
where it is used.

Photometric and PSF calibration
===============================

The limiting-magnitude form and this instrument's anchors with their
derivation; the PSF sigma and the star cutout box sizes; the magnitude-offset
table; how to recalibrate.

Frames, attitude, and rotation fitting
======================================

Camera frame names; the CK object; the spacecraft clock; the oops-from-SPICE
flip; whether the observation frame is evaluated or frozen; per-spacecraft
variation; the rotation-fitting flag, its cost, and its interaction with
C-kernel eligibility.

C-kernel specifics
==================

The baseline kernel structure and segment types; the angular-velocity census;
the kernel-name class rules; deviations in segment construction; the
reproduction path.

Simulator model
===============

Artifact-catalog entries; PSF and distortion parameters; artifact-mode
availability with the exclusion reasons; realism-match status.

Image library and test coverage
===============================

Cohort anchors; integration tests and fixtures.

PDS4 hooks
==========

Template directory, bundle name, LID and LIDVID builders, path stub, template
variables.

Backplanes, mosaics, and statistics
===================================

Describe this instrument alone: its behavior in each of those stages.

Open items
==========

The TODOs carried in the code for this instrument.
