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
4095, so a display or threshold scaled for counts shows an I/F frame as black. The
editor's status bar names the units and value range of each render.

A ``vgiss`` frame's dark sky is exactly 0, the same value that marks a lost pixel.
The navigator therefore treats the dark sky as lost data, and it refuses a frame
when more than 30% of the pixels are exactly 0. The editor's status bar warns
when a render crosses that limit. Real Voyager frames are unaffected, because
their dark sky sits slightly above 0.

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

``probability``
   The chance, from 0 to 1, that a defect affecting the whole frame happens at all, such
   as a truncated frame or a coherent banding pattern. ``1.0`` means every frame carries
   it.

``pixel_fraction``
   For hot pixels only: the fraction of the detector's pixels that are hot, from 0 to 1.

Dead pixels and dead columns can also be given as a fixed ``count`` in place of
``expected_events``. An entry names one of the two keys, never both. Every key defaults
to 0, and an entry whose key is 0 plants nothing.

.. code-block:: yaml

   artifacts:
     missing_lines: {expected_events: 3.0}
     truncated_frame: {probability: 1.0, fraction: 0.25}
     dead_pixels: {count: 10}
     hot_pixels: {pixel_fraction: 0.002}

A scene that still uses the retired ``incidence`` key is refused when it is loaded, and
the message names the key that replaces it. The developer guide's
:doc:`/dev_guide/dev_guide_simulator` chapter lists every defect, the cameras each one
is available on, and the key each one takes.

Further reading
===============

For the scene file format, the editor in detail, and how simulated frames are rendered and
then navigated as part of validation, see the developer guide's
:doc:`/dev_guide/dev_guide_simulator` chapter and the
:doc:`/simulator_report/simulator_report`.
