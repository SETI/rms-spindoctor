================================
Consolidating Navigation Outputs
================================

The navigation results root mirrors the directory hierarchy of the input holdings, so an
image's metadata document and its summary preview lie many directories down, which makes
reviewing fifty frames drawn from a dozen volumes a matter of descending a dozen deep
paths.

``sd_consolidate_metadata`` copies the results for a selected set of images into one flat
directory, named only by each image. Nothing is moved and nothing in the navigation
results root is changed; the program only reads from it.

Basic invocation
================

.. code-block:: bash

   sd_consolidate_metadata DATASET [selection] --dest-dir PATH \
       [--copy-metadata|--copy-png|--copy-all]

Images are selected exactly as they are for ``sd_offset``
(:doc:`/user_guide/user_guide_navigation_running`): the same dataset name, the same
positional image names, the same ``--volumes``, and the same image-number and file-list
filters. :doc:`/user_guide/user_guide_image_selection` describes them in full. Whatever
selection navigated a set of images will consolidate the results for that same set.

At least one of ``--copy-metadata``, ``--copy-png``, or ``--copy-all`` must be given.
Given none of them the program has nothing to do and says so.

What gets copied
================

``--copy-metadata``
    Copy each image's metadata document, the ``*_metadata.json`` file that records the
    frame's corrected pointing and everything navigation concluded about it. See
    :doc:`/user_guide/user_guide_metadata`.

``--copy-png``
    Copy each image's summary preview, the ``*_summary.png`` annotated image.

``--copy-all``
    Copy both. Equivalent to giving ``--copy-metadata`` and ``--copy-png`` together.

Where it goes
=============

``--dest-dir PATH``
    The destination directory. Required. Every copied file lands directly in it and no
    subdirectories are made, while the directory itself and any missing parents are
    created on the first write. The path may be local or a remote URL such as ``gs://`` or
    ``s3://``.

``--add-numerical-prefix``
    Prefix each destination filename with a six-digit increasing number, so that an
    alphabetical listing of the destination directory comes out in the order the images
    were selected. Without it the files sort by image name.

``--overwrite``
    Replace destination files that already exist. Without it an existing file is left
    alone and reported as skipped, which makes a repeated run safe.

``--dry-run``
    Report every copy that would happen, and make none of them. Use it to confirm the
    selection before writing anything.

An image that was never navigated, or whose navigation wrote no preview, simply has
nothing to copy. It is reported as a file that is not present, and the run continues.

Where the results are read from
===============================

The navigation results root comes from ``--nav-results-root``, then the
``environment.nav_results_root`` configuration setting, and only then the
``NAV_RESULTS_ROOT`` environment variable. Configuration files are resolved as they are
for every other program; see :doc:`/user_guide/user_guide_configuration`.

Example
=======

Gather the summary previews for one Cassini volume into a single directory, numbered so
they can be flipped through in selection order:

.. code-block:: bash

   sd_consolidate_metadata coiss --volumes COISS_2xxx/COISS_2116 \
       --copy-png --add-numerical-prefix --dest-dir /tmp/coiss_2116_summaries

The run reports how many files it copied and how many were not present, and writes the
same account to its log; see :doc:`/user_guide/user_guide_logging`.
