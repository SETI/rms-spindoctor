===========
Instruments
===========

One chapter per instrument. Each is self-contained: it answers, for its own
instrument, every question the shared chapters answer in general, and it names
no other instrument. Comparing two instruments means opening both chapters and
reading the same section on each, which is what the fixed section order below
is for.

The sections, in the order every chapter carries them, are Overview, Pipeline
support, Datasets and image selection, Image data and units, Field of view and
geometry, Metadata fields, Corrected-pointing C-kernels, Known limitations, and
References. A section that is empty for an instrument still appears, saying
``None.`` or ``Not supported.``, so that a missing answer is never confused
with an unasked question.

These chapters do not restate the instrument teams' own documentation. There
are no apertures, focal lengths, pixel scales, angular field extents or filter
tables here; each chapter's References section points at the official documents
that carry them. What a chapter carries instead is what SpinDoctor decides,
configures, measures, or does for that instrument.

A new instrument gets a chapter by copying ``_template.rst`` in this directory
to ``<instrument>.rst`` and filling it in. The list below is a glob, so nothing
else has to be edited.

.. toctree::
   :maxdepth: 2
   :glob:

   *
