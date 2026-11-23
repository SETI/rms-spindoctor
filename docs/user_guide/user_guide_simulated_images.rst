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

You do not need the simulator to navigate real spacecraft images. The simulator checks
that the navigation techniques behave as designed; it doesn't measure how accurately they
navigate real frames, and the confidence values a navigation reports are provisional,
fitted on simulated frames. A simulated frame is navigated by the same
program that navigates a real one, as described below, and a simulated scene cannot be
assembled into a PDS4 bundle.

What a simulated check measures
===============================

Recovering a simulated frame's known offset measures **reproducibility** -- how tightly
the navigation techniques converge -- rather than accuracy on real frames. On a simulated
frame the navigator predicts the scene with simulated models that share the simulator's
own geometry and shading, so a scene that matches that model is the model's own picture
plus a shift and noise, and its recovery error is a **self-consistency floor**.
Sensitivity is reported through **model-mismatch sweeps**: recovery error measured as a
function of how far the rendered image departs from the navigator's model. Each
instrument's **realism match** compares simulated frames with real ones; a close match is
a precondition for reading any simulated number as a statement about real frames, but it
isn't enough on its own, and today simulated results stand for reproducibility, measured
mostly on the Cassini NAC.

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

Most scene settings are optional. To use a setting's default, leave the setting out of
the file.

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

While you edit, the editor shows a fast preview. It draws the scene in a fraction of a
second, even on a full-size frame, so dragging a body or changing a value feels
immediate. It is fast because it leaves out three things that are hard to see on
screen. It draws on the detector's own pixels instead of a finer grid, it draws each
body's edge without smoothing, and it leaves out the camera's residual lens distortion.
Everything you place is still there at its own brightness: bodies, rings, and stars,
blurred by the camera's point-spread function, with the scene's noise.

About a second after your last change, the editor draws the full render in the
background. This is the same image a test or a saved frame gets. When it finishes, it
replaces the preview. If you change something while the full render is being drawn,
the editor discards that render and goes back to fast previews. The status bar says
which image is on screen: "Fast preview", "Fast preview (full render running)", or
"Full render". Judge noise, blur, and exact positions on the full render only.

A saved PNG image always comes from the full render. If you save an image while the
preview is on screen, the editor draws the full render first, which can take a few
seconds on a full-size frame, and then shows it in place of the preview.

On screen, the status bar's units and verdict wait for the full render too, and so does
the saturation overlay. Until it arrives, the status bar says the verdict is pending,
and the saturation overlay says it is waiting for the full render. If the full render
fails, the preview stays on screen and the status bar says the full render failed, with
the start of the error message. Hold the pointer over that message to read all of it.
The saturation overlay then says the full render failed, and so does the verdict. A
scene that plants an offset is the exception: its verdict comes from a render of its
own.

``--full-render-delay SECONDS`` sets how long the editor waits after your last change
before it starts the full render. The default is 1 second.

.. code-block:: bash

   sd_create_simulated_image --full-render-delay 2.5 /path/to/scene.yaml

On a ``polyhedral_mesh`` body the editor disables the controls for the appearance keys a
mesh doesn't support, because the scene file would refuse them: atmosphere, photometric
law, opposition surge, albedo texture, disc texture, transits, and the crater controls.
Each disabled control's tooltip says why. When you switch a body's ``shape_model`` to
``polyhedral_mesh``, the editor moves the values a mesh refuses out of the scene, and the
status bar lists them. Switching back puts them back during the same session. A scene
saved in between doesn't carry them.

Each control shows the value the simulator draws the frame with. If the scene leaves a
setting out, its control shows the default. Some controls sit in a group that turns on
and off as a whole, such as a body's atmosphere or opposition surge. If the scene leaves
out such a group, the group starts turned off. If the scene does not set a body's name,
the editor shows the name the simulator gives it, such as ``SIM-BODY-1`` for the first
body in the list. If the scene does not set a body's center, the editor shows the center
of the frame.

Units of a simulated frame
==========================

A scene file's ``instrument`` selects the camera a frame imitates, and the camera
decides the frame's units. There are two kinds of frame, and in both a lost pixel
reads 0, as it does in the real products.

* ``coiss_nac``, ``coiss_wac``, ``gossi``, ``nhlorri``, and ``generic`` frames are in
  raw counts (DN). The dark sky sits a little above 0.
* ``coiss_calib_nac``, ``coiss_calib_wac``, and the four ``vgiss`` cameras' frames
  are in I/F, like the calibrated Cassini products and the geometrically corrected
  Voyager products. The dark sky never reads exactly 0: it sits at about 0.012 on a
  Voyager frame, and at about 0.00004 on a calibrated Cassini frame.

An I/F frame spans roughly 0 to 1, where a DN frame of the same scene can reach
4095, so a display or threshold scaled for counts shows an I/F frame as black.
With ``instrument_defaults`` on, read noise moves the dark sky about. On a
calibrated Cassini frame it scatters around 0, and more than a quarter of the
pixels of a disc scene sit slightly below 0, as on the real frames. On a Voyager
frame it stays above 0 on all but about 7 pixels in 100,000, as the real Voyager
sky stays above 0. The editor's status bar shows the units and value range of each
render.

The status bar also warns when the navigator would refuse a render: when more than
30% of its pixels are lost, or when the frame would read as blank, as an empty
calibrated Cassini frame does. It reads the frame the navigator sees, with the scene's
planted offset applied, so its verdict can differ from what the picture shows. When
the scene plants an offset, the editor draws that frame in the background too, at the
same time as the full render. Each preview sets the status bar to the units followed by
"verdict pending", and the verdict fills in once the background renders finish. The
preview redraws a moment after you stop changing a value. Until then, the status bar
still describes the scene as it was before the change.

Before each redraw the editor checks the scene by the same rules it uses to load a
scene file. An edit can leave the scene unfinished for a moment, such as when two
bodies share a name while you retype one, or when a value is out of range. If the scene
would be refused, the last valid picture stays on screen. The status bar shows the
reason in place of a verdict, whether or not the scene plants an offset. Until then
you cannot drag an object on the picture, because it shows the earlier scene;
the object's tab still edits it. Once an edit makes the scene valid again, the preview
redraws as usual.

The random pixel loss (the missing-data rate, in the Artifacts tab's detector override)
and the cosmic-ray rate (on the General tab) work on every camera. The detector override
also sets the image's saturation level. On a camera that delivers raw counts the
navigator keeps using the camera's own level, so a scene can show how navigation copes
with a detector that saturates early. A calibrated frame has no saturation level for the
navigator to use.

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
``expected_events``. An entry sets one of the two keys, never both. Every key defaults
to 0, and an entry whose key, or whose ``count``, is 0 doesn't plant anything.

With ``instrument_defaults`` on, the defects that every frame of the camera shows are
turned on for you: hot pixels, banding, the readout dark, the bias structure, and
column bloom on a raw Cassini camera, for example. An entry such as ``hot_pixels: {}``
then plants the camera's own hot pixels, and a value you set in the entry replaces the
camera's value for that key alone. Setting the entry's key to 0, as in
``banding: {probability: 0}``, turns that defect off.

.. code-block:: yaml

   artifacts:
     missing_lines: {expected_events: 3.0}
     truncated_frame: {probability: 1.0, fraction: 0.25}
     dead_pixels: {count: 10}
     hot_pixels: {pixel_fraction: 0.002}

A key that a defect doesn't take is refused when the scene is loaded. The developer
guide's :doc:`/dev_guide/dev_guide_simulator` chapter lists every defect, the cameras
each one is available on, and the key each one takes.

Some defects arise in one kind of detector only. Hot pixels, column bloom, and
radiation hits, among others, are CCD defects, and the limb bend and the prior-frame
ghost are vidicon defects. If a scene asks for a defect its detector can't produce, the
scene is refused when it is loaded. It is also refused if ``instrument_defaults`` would
turn on one of its camera's defects that the scene's chosen ``detector_model`` can't
produce. In the editor, if the scene's camera doesn't offer a defect or its detector
can't produce it, the defect's entry is disabled, and its tooltip says why. If the
scene already includes that defect, its entry stays enabled so that you can turn it
off. The editor won't save the scene until you do.

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
``a_px``. Each kind then takes its own keys. A ringlet, a gap, and a ramp need a
radial width ``width_px``. An edge and a ramp may name the ``side`` of the orbit that
carries the optical depth, ``in`` or ``out``. A wave needs a radial wavelength
``wavelength_px`` and a radial damping length ``density_wave_damping_px``. A key on a
kind that doesn't take it is refused, because it would draw a different feature than the
one intended.

An edge fills every radius on one side of its orbit, by default everything inward of
``a_px``. Two edges therefore can't bound a band. They overlap, and the band shows up only
as the stretch where the optical depth doubles. Draw a band as a ringlet, or cut it into a
sheet as a gap.

An unknown ring key is refused when the scene is loaded.
The developer guide's :doc:`/dev_guide/dev_guide_simulator` chapter lists every ring key
and tabulates which kind takes which key.

Further reading
===============

For the scene file format, the editor in detail, and how simulated frames are rendered and
then navigated as part of validation, see the developer guide's
:doc:`/dev_guide/dev_guide_simulator` chapter and the
:doc:`/simulator_report/simulator_report`.
