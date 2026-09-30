============
Introduction
============

SpinDoctor determines where a spacecraft camera was really pointing when it took an
image, and turns that answer into archival data products. It reads images from Cassini
ISS, Voyager ISS, Galileo SSI, and New Horizons LORRI, compares each one against a model
of what the sky should have contained, and records the correction needed to make the
model line up with the image.

What SpinDoctor Is For
======================

The pointing recorded for an image by its mission comes from spacecraft telemetry and is
only as good as the attitude reconstruction behind it. It is often wrong by several
pixels, and sometimes by far more. Every measurement that depends on knowing which part
of the sky, or which part of a ring or a moon, a pixel looked at inherits that error.

Determining the true pointing of an image is called *navigation*. A navigated image
supports work that an unnavigated one does not:

* Per-pixel geometry, such as the latitude, longitude, ring radius, and illumination
  angles behind each pixel.
* Measurements tied to a location on a body or in a ring, where an error of a few pixels
  puts the measurement somewhere other than where it was meant to be.
* Reprojection and mosaicing, which cannot line up two images of the same terrain
  without knowing where each one pointed.
* Archival products that carry corrected geometry for other people to use.

How a Single Image Is Navigated
===============================

Navigation is a comparison between an image and a prediction of that image:

1. The image and its label are read, and the spacecraft trajectory and attitude are
   looked up in SPICE kernels.
2. Models are built of everything that should be in the field of view: the star field,
   each body, and each planet's rings.
3. Each model is matched against the image by whichever methods suit it, such as fitting
   a body's illuminated limb, correlating a star field, or fitting a ring edge.
4. The individual answers are reconciled into one pointing correction, with an
   uncertainty and a statement of how much the methods agreed.
5. The results are written out: a metadata document recording the correction and
   everything that led to it, and a summary image showing the models drawn over the
   data.

The correction is expressed two ways. The *offset* says how far the image must be
shifted, in pixels, to agree with the prediction, and is meaningful only alongside the
SPICE kernels that produced that prediction. The *corrected pointing* is the resulting
camera attitude, and is what other programs should use.

The Rest of the Pipeline
========================

Navigation is the first of four processing phases. Each of the three that follow reads
what navigation recorded. C-kernel generation and backplane generation read nothing
else the pipeline produced. Bundle generation reads the backplanes as well:

1. **Navigation.** Every image is compared against models of the stars, planets, moons,
   and rings that should have been in its field of view, and the pointing correction that
   makes the models line up with the image is recorded. Each navigated image gets a
   metadata document of its own, holding the correction, its uncertainty, and the
   corrected pointing. Run by ``sd_offset`` (:doc:`user_guide_navigation_running`).

2. **Corrected-pointing C-kernel generation.** The corrected attitudes are packaged as
   SPICE C kernels, one corrected kernel mirroring each original kernel the images were
   navigated against, so any SPICE-based tool can furnish the improved attitude. Run by
   ``sd_create_ck`` (:doc:`user_guide_ck_kernels`).

3. **Backplane generation.** The per-pixel geometry of a navigated image is computed:
   longitude, latitude, incidence, emission, phase, ring radius, and the rest. Run by
   ``sd_backplanes`` (:doc:`user_guide_backplanes`).

4. **PDS4 bundle generation.** The navigation results and the backplanes are assembled
   into a PDS4 bundle with labels, collections, and browse products, ready for archiving.
   Run by ``sd_create_bundle`` (:doc:`user_guide_pds4_bundle`).

Reprojection and mosaicing are not a phase. They are a set of tools that read navigated
images and build maps of a body's surface or of a planet's rings
(:doc:`user_guide_reprojection`).

Supported Missions
==================

Every program that processes images takes a dataset name as its first argument. The name
says which mission and instrument the images come from, and for Cassini ISS it can also
narrow the run to one part of the archive. Names are case-insensitive, so ``COISS`` and
``coiss`` select the same dataset. Each name also has a form ending in ``_pds3``, which
selects exactly the same images and spells out that they come from a PDS3 archive. Type
the short form; the examples throughout this guide use it.

* Cassini Imaging Science Subsystem: ``coiss`` and ``coiss_pds3`` for all volumes,
  ``coiss_cruise`` and ``coiss_cruise_pds3`` for the cruise volumes 1001 to 1009, and
  ``coiss_saturn`` and ``coiss_saturn_pds3`` for the Saturn volumes 2001 to 2116. See
  :doc:`instruments/cassini_iss`.
* Galileo Solid State Imager: ``gossi`` and ``gossi_pds3``. See
  :doc:`instruments/galileo_ssi`.
* New Horizons Long Range Reconnaissance Imager: ``nhlorri`` and ``nhlorri_pds3``. See
  :doc:`instruments/newhorizons_lorri`.
* Voyager Imaging Science Subsystem: ``vgiss`` and ``vgiss_pds3``. See
  :doc:`instruments/voyager_iss`.
* Simulated images: ``sim``. These are rendered from a scene description rather than
  read from an archive, and are used to check the pipeline against a known answer. See
  :doc:`user_guide_simulated_images`.

Cassini ISS, Voyager ISS, and Galileo SSI images are in VICAR format, and New Horizons
LORRI images are in FITS format. All four instruments are read from PDS3 archives, which
is the organization of volumes, labels, and PDS3 index tables around those image files
rather than a format of its own.

Each instrument has its own chapter, which carries the volumes it covers, which product
is navigated, the image-name forms accepted, the thresholds its results are judged
against, and everything else true of that instrument alone. The shared chapters describe
the mechanisms, and the instrument chapters carry the values.

Where to Start
==============

* :doc:`/quick_start` installs the package and walks through navigating one image.
* :doc:`user_guide_installation` covers installation, external data, and every
  environment variable in full.
* :doc:`user_guide_concepts` introduces what every program has in common, including
  :doc:`user_guide_configuration`, :doc:`user_guide_logging`, and
  :doc:`user_guide_image_selection`.
* :doc:`user_guide_navigation` is the navigation phase itself.
