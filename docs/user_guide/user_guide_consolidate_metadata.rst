================================
Consolidating Navigation Outputs
================================

``sd_offset`` writes each image's results (the ``*_metadata.json`` offset file
and the ``*_summary.png`` preview) under the navigation results root, mirroring
the per-volume directory hierarchy of the input holdings. Browsing results
across many volumes therefore means descending a deep path tree. The
``sd_consolidate_metadata`` program copies the results for a selected set of
images into a single flat directory so they are easy to review or hand off.

It selects images with the same dataset arguments as ``sd_offset`` (positional
image names, ``--volumes``, image-number and file-list filters, and so on), and
for each selected image copies the requested product(s) out of the navigation
results root into the destination directory:

* ``--dest-dir PATH``: destination directory; every copied file lands directly
  here with no subdirectories, and missing parents are created on first write.
* ``--copy-metadata``: copy the per-image ``*_metadata.json`` files.
* ``--copy-png``: copy the per-image ``*_summary.png`` files.
* ``--copy-both``: copy both (equivalent to ``--copy-metadata --copy-png``).
* ``--index-prefix``: prefix each destination filename with a six-digit
  increasing index so the flat listing matches the iteration order.
* ``--overwrite``: overwrite destination files that already exist.
* ``--dry-run``: report what would be copied without copying anything.

It reads (and never modifies) the navigation results; the source root comes
from ``--nav-results-root``, the ``NAV_RESULTS_ROOT`` environment variable, or
the ``nav_results_root`` configuration value, and holdings and configuration
are resolved exactly as for ``sd_offset``.

For example, to gather the summary PNGs for one Cassini volume into a single
directory for a quick visual pass::

   sd_consolidate_metadata coiss --volumes COISS_2xxx/COISS_2116 \
       --copy-png --index-prefix --dest-dir /tmp/coiss_2116_summaries


