============
Introduction
============

SpinDoctor is a spacecraft image navigation system designed to analyze images from various space missions and determine precise positional offsets. This guide explains how to use the primary command-line interface exposed by the ``sd_offset`` script to navigate images and generate results, and how to invoke the cloud-tasks variant for queue-driven processing.

Purpose of the System
---------------------

The primary purpose of SpinDoctor is to determine the precise pointing of spacecraft instruments by comparing the observed images with theoretical models of what should appear in the field of view. This process, known as "navigation," is crucial for:

1. Validating and correcting spacecraft pointing information
2. Ensuring accurate scientific interpretations of the imagery
3. Creating properly annotated and labeled images for analysis
4. Supporting mission planning and operations

The system works by:

1. Reading spacecraft imagery and metadata
2. Generating theoretical models of stars, planets, moons, and rings
3. Correlating the observed features with the theoretical models
4. Calculating the offset between the expected and actual pointing
5. Producing annotated images and data files with the results

Supported Missions
------------------

SpinDoctor supports multiple instruments, organized by dataset names you will pass on the command line. Dataset names are case-insensitive and map to instrument-specific handlers. The complete set is:

* ``coiss`` and ``coiss_pds3`` — Cassini Imaging Science Subsystem (all volumes) — :doc:`instruments/cassini_iss`
* ``coiss_cruise`` and ``coiss_cruise_pds3`` — Cassini Imaging Science Subsystem (Cruise volumes 1001-1009) — :doc:`instruments/cassini_iss`
* ``coiss_saturn`` and ``coiss_saturn_pds3`` — Cassini Imaging Science Subsystem (Saturn volumes 2001-2116) — :doc:`instruments/cassini_iss`
* ``gossi`` and ``gossi_pds3`` — Galileo Solid State Imager — :doc:`instruments/galileo_ssi`
* ``nhlorri`` and ``nhlorri_pds3`` — New Horizons Long Range Reconnaissance Imager — :doc:`instruments/newhorizons_lorri`
* ``vgiss`` and ``vgiss_pds3`` — Voyager Imaging Science Subsystem — :doc:`instruments/voyager_iss`
* ``sim`` — simulated images (see :doc:`user_guide_simulated_images`)

Each instrument's chapter carries the volumes it covers, which product is navigated, the image-name forms accepted, the units and thresholds it is judged against, and everything else that is true of that instrument and not of another. The shared chapters describe the mechanisms; the instrument chapters carry the values.


