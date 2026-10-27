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

The editor disables a control whose value the scene file would refuse, and the control's
tooltip says why:

* On a ``polyhedral_mesh`` body, the controls for the appearance keys a mesh doesn't
  support: atmosphere, photometric law, opposition surge, albedo texture, disc texture,
  transits, and the crater controls.
* On the Voyager vidicon detector (``vgiss``, or a detector model of ``vidicon``), the
  cosmic-ray rate.
* On a calibrated frame (``coiss_calib_nac``, ``coiss_calib_wac``, or ``vgiss``), the
  missing-data rate.

When switching a body's shape, the camera, or the detector model makes a value the scene
carries refused, the editor moves that value out of the scene, and the status bar names
it. Switching back puts the value back during the same session. A scene saved in between
doesn't carry it.

Units of a simulated frame
==========================

A scene file's ``instrument`` names the camera a frame imitates, and the camera
decides the frame's units and the value that marks a lost pixel. There are three
cases.

* ``coiss_nac``, ``coiss_wac``, ``gossi``, ``nhlorri``, and ``generic`` frames are in
  raw counts (DN). A lost pixel reads 0, and the dark sky sits a little above 0.
* ``coiss_calib_nac`` and ``coiss_calib_wac`` frames are in I/F, like the calibrated
  Cassini products. A lost pixel reads NaN.
* ``vgiss`` frames are in I/F, like the geometrically corrected Voyager products,
  and a lost pixel reads 0, as it does in those products.

An I/F frame spans roughly 0 to 1, where a DN frame of the same scene can reach
4095, so a display or threshold scaled for counts shows an I/F frame as black.
With ``instrument_defaults`` on, read noise takes part of an I/F frame's dark sky
slightly below 0: about 5% of the pixels of a ``vgiss`` disc scene, and more than
a quarter of a ``coiss_calib_nac`` one. The editor's status bar names the units
and value range of each render.

A ``vgiss`` frame's dark sky is exactly 0, the same value that marks a lost pixel.
The navigator therefore treats the dark sky as lost data, and it refuses a frame
when more than 30% of the pixels are exactly 0. The editor's status bar warns
when a render crosses that limit. It counts the pixels of the frame the navigator
sees, with the scene's planted offset applied, so the count can differ from what
the preview shows. It also says when the navigator would call a frame blank, as
it does an empty ``vgiss`` frame. A scene run through ``sd_offset`` gets a warning
in its navigation log instead, when the dark sky is what puts the frame over the
limit. A frame that the scene's planted data loss alone puts over the limit
doesn't get that warning, because the scene asked for that loss. Real Voyager
frames are unaffected, because their dark sky sits slightly above 0.

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
   non-negative number is valid. Radiation hits (``radiation_transients``) are the one
   exception to "per frame": their entry's ``environment_factor`` multiplies the number,
   and a ``readout_dwell_sec`` adds hits in proportion to the extra time the frame waits
   before readout.

``expected_events_per_sec``
   For Cassini's bright and dark pixel pairs (``bright_dark_pairs``) only: the expected
   number of pairs per second of exposure. A frame exposed for 0.5 seconds with
   ``expected_events_per_sec: 10`` carries about five pairs.

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
* A name ending in ``_deg`` is in degrees. Every longitude is measured in the ring plane
  from the ascending node.
* An edge wave's ``damping_rad`` is in radians of longitude.

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
           edge_wave: {amp_px: 2.0, wavelength_px: 10.0, damping_rad: 0.6, lam0_deg: 45.0}

Every feature needs a ``kind``, an optical depth ``tau``, and an orbit with its radius
``a_px``. Each kind then takes its own shape keys. A ringlet, a gap, and a ramp need a
radial width ``width_px``. An edge and a ramp may name the ``side`` of the orbit that
carries the optical depth, ``in`` or ``out``. A wave needs a radial wavelength
``wavelength_px`` and a radial damping length ``damping_px``. A shape key on a kind that
doesn't take it is refused, because it would draw a different feature than the one
intended.

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
