================
Simulated Images
================

SpinDoctor includes an image simulator that synthesizes spacecraft frames -- stars,
planetary bodies, and rings, with a realistic detector model -- for arbitrary geometry. It
exists to **test and validate the navigation pipeline**: because a simulated frame's true
pointing offset is known by construction, it is the only kind of frame whose navigation
answer can be checked exactly. The simulator backs the algorithmic-invariant tests, the
regression baselines, and the single-variable sensitivity sweeps that verify the navigator
behaves as expected.

You do not need the simulator to navigate real spacecraft images; the results you rely
on are validated with it on your behalf. A simulated frame is navigated by the same
program that navigates a real one, as described below, and a simulated scene cannot be
assembled into a PDS4 bundle.

What a simulated check measures
===============================

What such a check measures depends on how the simulated image is built. When the rendered
scene matches the navigator's own model of the sky, recovering the known offset measures
**reproducibility** -- a self-consistency floor -- rather than accuracy. The numbers speak
to accuracy only to the degree a simulated frame resembles a real one, which is what each
instrument's **realism match** establishes; that match is what makes simulated numbers
credible. Sensitivity is reported through **model-mismatch sweeps**: recovery error
measured as a function of how far the rendered image departs from the navigator's model.

Navigating a simulated frame
============================

A simulated scene is described by a YAML file. The frame it describes is navigated by the
same pipeline that navigates real images: pass ``sim`` as the dataset name to
``sd_offset`` (:doc:`/user_guide/user_guide_navigation_running`), and give the path to the
scene file in place of an image name.

.. code-block:: bash

   sd_offset sim /path/to/scene.yaml

More than one scene file may be named on one command line, and every other ``sd_offset``
option behaves as it does for a real dataset. The ``sim`` dataset takes scene-file paths
directly, so it has none of the holdings-based selection options -- there is no
``--volumes`` and no image-number filtering -- and it does not support PDS4 bundle
generation.

The scene editor
================

``sd_create_simulated_image`` is an interactive editor for scene files. It opens a window
in which a scene is assembled and rendered.

.. code-block:: bash

   sd_create_simulated_image

Give it the path of an existing scene file to open that scene on launch instead of
starting from an empty one.

.. code-block:: bash

   sd_create_simulated_image /path/to/scene.yaml

On a ``polyhedral_mesh`` body the editor disables the controls for the appearance keys a
mesh doesn't support, because the scene file would refuse them: atmosphere, photometric
law, opposition surge, albedo texture, disc texture, transits, and the crater controls.
Each disabled control's tooltip says why. When switching a body's shape makes a value the
scene carries refused, the editor moves that value out of the scene, and the status bar
names it. Switching back puts the value back during the same session. A scene saved in
between doesn't carry it.

Units of a simulated frame
==========================

A scene file's ``instrument`` names the camera a frame imitates, and the camera
decides the frame's units. There are two kinds of frame, and in both a lost pixel
reads 0, as it does in the real products.

* ``coiss_nac``, ``coiss_wac``, ``gossi``, ``nhlorri``, and ``generic`` frames are in
  raw counts (DN). The dark sky sits a little above 0.
* ``coiss_calib_nac``, ``coiss_calib_wac``, and ``vgiss`` frames are in I/F, like the
  calibrated Cassini products and the geometrically corrected Voyager products. The
  dark sky never reads exactly 0: it sits at about 0.012 on a ``vgiss`` frame, and
  at about 0.00004 on a calibrated Cassini frame.

An I/F frame spans roughly 0 to 1, where a DN frame of the same scene can reach
4095, so a display or threshold scaled for counts shows an I/F frame as black.
With ``instrument_defaults`` on, read noise moves the dark sky about. On a
calibrated Cassini frame it scatters around 0, and more than a quarter of the
pixels of a disc scene sit slightly below 0, as on the real frames. On a ``vgiss``
frame it stays above 0 on all but about 7 pixels in 100,000, as the real Voyager
sky stays above 0. The editor's status bar names the units and value range of each
render.

The status bar also warns when the navigator would refuse a render: when more than
30% of its pixels are lost, or when the frame would read as blank, as an empty
calibrated Cassini frame does. It reads the frame the navigator sees, with the scene's
planted offset applied, so its verdict can differ from what the preview shows. When
the scene plants an offset, the editor renders that frame a second time in the
background. The preview appears immediately, the status bar shows the units followed
by "verdict pending", and the verdict fills in a moment later, once the second render
finishes. The editor redraws a moment after you stop changing a value, and each
redraw replaces the pending verdict. Until that redraw, the status bar still describes
the scene as it was before the change.

The random pixel loss (the missing-data rate) and the cosmic-ray rate work on every
camera, the Voyager vidicon included.

Planting defects in a scene
===========================

A scene file's ``artifacts`` section plants the defects a real downlink or detector
leaves in a frame: lost lines, truncated frames, hot pixels, compression blockiness, and
many more. Each defect has its own entry, named after the defect, and each entry turns
on through exactly one key. The key's name says what its number means.

``expected_events``
   The expected number of events per frame, such as lost lines, lost compression
   blocks, or spiked pixels. The simulator draws the actual number at random around this
   value, so ``missing_lines: {expected_events: 3.0}`` loses about three lines. Any
   non-negative number is valid.

``expected_events_per_sec``
   The expected number of events per second, for Cassini's bright and dark pixel pairs
   (``bright_dark_pairs``) and for radiation hits (``radiation_transients``). A frame
   exposed for 0.5 seconds with ``bright_dark_pairs: {expected_events_per_sec: 10}``
   carries about five pairs. Radiation hits also land while the frame waits for readout,
   so their count is the rate times the entry's ``environment_factor`` times the
   exposure plus the ``readout_dwell_sec``. A 0.5 second frame that waits 1.5 seconds,
   with ``radiation_transients: {expected_events_per_sec: 10, readout_dwell_sec: 1.5}``,
   carries about 20 hits.

``probability``
   The chance, from 0 to 1, that a defect affecting the whole frame happens at all, such
   as a truncated frame or a coherent banding pattern. ``1.0`` means every frame carries
   it.

``pixel_fraction``
   For hot pixels only: the fraction of the detector's pixels that are hot, from 0 to 1.

Dead pixels and dead columns can also be given as a fixed ``count`` in place of
``expected_events``. An entry names one of the two keys, never both. Every key defaults
to 0, and an entry whose key, or whose ``count``, is 0 doesn't plant anything.

.. code-block:: yaml

   artifacts:
     missing_lines: {expected_events: 3.0}
     truncated_frame: {probability: 1.0, fraction: 0.25}
     dead_pixels: {count: 10}
     hot_pixels: {pixel_fraction: 0.002}

A key that a defect doesn't take is refused when the scene is loaded. The developer
guide's :doc:`/dev_guide/dev_guide_simulator` chapter lists every defect, the cameras
each one is available on, and the key each one takes.

Rings in a scene
================

A scene file's ``ring_system`` section draws a ring system: a list of features, each a
ringlet, a gap, a one-sided edge, a ramp, or a density wave (``wave``), placed on an
orbit. Each length and angle in the section carries its unit in its name, apart from
the ring center ``center_v`` and ``center_u``, which are positions on the image in
pixels.

* A name ending in ``_px`` is a distance in the ring plane, measured in detector pixels.
  On a face-on ring it is the distance measured on the image. On a tilted ring it
  shrinks across the line of nodes, as the ring does. A moonlet's ``radius_px`` is the
  exception: the moonlet is drawn as a disc on the image, so its radius is in image
  pixels and doesn't shrink.
* Every angle is in degrees, and its name carries ``_deg``. Every longitude is measured
  in the ring plane from the ascending node. An edge wave's damping length
  ``edge_wave_damping_deg`` is in degrees of longitude, at most ``360 / pi``, about
  114.59.

.. code-block:: yaml

   ring_system:
     geometry: {opening_deg_obs: 50.0, opening_deg_sun: 50.0, node_deg: 30.0}
     features:
       - name: SHEET
         kind: edge
         tau: 1.2
         orbit: {a_px: 95.0}
       - name: WAVYGAP
         kind: gap
         tau: 1.2
         width_px: 6.0
         navigable: true
         orbit:
           a_px: 60.0
           edge_wave:
             amp_px: 2.0
             wavelength_px: 10.0
             edge_wave_damping_deg: 34.4
             lam0_deg: 45.0

Every feature needs a ``kind``, an optical depth ``tau``, and an orbit with its radius
``a_px``. Each kind then takes its own shape keys. A ringlet, a gap, and a ramp need a
radial width ``width_px``. An edge and a ramp may name the ``side`` of the orbit that
carries the optical depth, ``in`` or ``out``. A wave needs a radial wavelength
``wavelength_px`` and a radial damping length ``density_wave_damping_px``. A shape key on a
kind that doesn't take it is refused, because it would draw a different feature than the
one intended.

An edge fills every radius on one side of its orbit, by default everything inward of
``a_px``. Two edges therefore can't bound a band. They overlap, and the band shows up only
as the stretch where the optical depth doubles. Draw a band as a ringlet, or cut it into a
sheet as a gap.

An unknown ring key is refused when the scene is loaded. An optional ring key set to
``null`` counts as unset: it takes the value it has when it is left out.
The developer guide's :doc:`/dev_guide/dev_guide_simulator` chapter lists every ring key
and tabulates which kind takes which shape key.

Further reading
===============

For the scene file format, the editor in detail, and how simulated frames are rendered and
then navigated as part of validation, see the developer guide's
:doc:`/dev_guide/dev_guide_simulator` chapter and the
:doc:`/simulator_report/simulator_report`.
