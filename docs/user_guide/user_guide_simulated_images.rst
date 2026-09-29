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

You do not need the simulator to navigate real spacecraft images, and nothing in the
navigation, C-kernel, backplane, or PDS4 bundle stages reads a simulated frame. The
results you rely on are validated with the simulator on your behalf.

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
in which a scene is assembled and rendered, and it can be given the path of an existing
scene file to open on launch.

.. code-block:: bash

   sd_create_simulated_image [SCENE.yaml]

Further reading
===============

For the scene file format, the editor in detail, and how simulated frames are rendered and
then navigated as part of validation, see the developer guide's
:doc:`/dev_guide/dev_guide_simulator` chapter and the
:doc:`/simulator_report/simulator_report`.
