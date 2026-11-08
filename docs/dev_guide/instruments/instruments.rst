===========
Instruments
===========

One chapter per instrument, for a developer changing or extending that
instrument's support. Each chapter is self-contained: it answers, for its own
instrument, every question the shared subsystem chapters answer in general, and it
never mentions another instrument. The subsystem chapters carry the mechanism;
these carry the values.

The sections, in the order every chapter carries them, are Code map, Loading
the image, Label and index dependencies, Configuration block, Photometric and
PSF calibration, Frames attitude and rotation fitting, C-kernel specifics,
Simulator model, Image library and test coverage, PDS4 hooks, Backplanes
mosaics and statistics, and Open items. A section that is empty for an
instrument still appears, saying ``None.`` or ``Not supported.``

A new instrument gets a chapter by copying ``_template.rst`` in this directory
to ``<instrument>.rst`` and filling it in. The list below is a glob, so nothing
else has to be edited. A registered instrument with no chapter here, or a
chapter missing one of the template's sections, fails
``tests/spindoctor/test_instrument_chapters.py``.

.. toctree::
   :maxdepth: 2
   :glob:

   *
