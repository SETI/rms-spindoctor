Navigation Statistics
=====================

``sd_stats_report`` turns the results of a navigation run into a deterministic
report: Markdown text plus PNG charts. It is the standing quality check on a
production run -- success and failure rates, which techniques and models carry
the load, offset distributions, how well the techniques agree with one another,
and whether the confidence tiers behave as designed -- and it is cron-friendly,
being a single non-interactive invocation:

.. code-block:: bash

    sd_stats_report --nav-results-root /data/nav-offset-results \
        --output-dir stats_report
    sd_stats_report \
        --results-index-db sqlite:////data/nav-offset-results/index.sqlite3 \
        --instrument coiss --start-date 2005-03-01 --end-date 2005-03-01 \
        --output-dir day_report

**The report reads a navigation results tree or a results index, and over the
records both storages can read it is the same report either way.** Every number
in it comes from one pass over the per-image records, and the two storages hand
that pass the same records, so a report over a tree and a report from an index
built from that tree carry the same text and the same charts.

One count is the exception, and it is the count of files that yielded no record:
a file the storage could not deliver at all is counted from a tree and not from
an index, because the ingest deliberately records no refusal for a retrieval
that failed once. The **Files that yielded no record** entry below says what
that means for the number printed.

An index built by an ingest run with ``--no-prune`` is the other way the two can
differ, and it is a choice somebody made rather than a property of the storages:
that ingest leaves in place the rows of documents that have left the tree, so
the report from it counts images the tree no longer holds and the report over
the tree does not. Running ``sd_results_index ingest`` again without the flag
removes them. :doc:`user_guide_results_index` says what the flag gives up and what it
saves.

The configuration is read the way every other program reads it: the files named
with ``--config-file``, or else ``nav_default_config.yaml`` in the working
directory. Over a tree, the ``results_tree`` section decides how many requests
the report makes at once; :doc:`user_guide_results_index` says what each
setting costs.

An index is optional here exactly as it is everywhere else, and
``--results-index-db`` is resolved the same way: from the command line, then the
``environment.results_index_db`` configuration variable, then the
``NAV_RESULTS_INDEX_DB`` environment variable, with the literal ``none`` at any
level meaning no index. The index is a database built from the navigation
results tree by a separate pass, ``sd_results_index ingest``, and
:doc:`user_guide_results_index` is where it is documented: how it is named and
resolved, how to build and rebuild one, the tables it holds, and how to query it
directly for the questions this report does not answer.

.. code-block:: bash

    sd_results_index ingest --nav-results-root /data/nav-offset-results \
        --results-index-db sqlite:////data/nav-offset-results/index.sqlite3
    sd_stats_report \
        --results-index-db sqlite:////data/nav-offset-results/index.sqlite3

With no index the navigation results tree is read instead. The roots to read
come from ``--nav-results-root`` (repeatable), then the ``nav_results_root``
configuration variable, then ``NAV_RESULTS_ROOT``, as they do for the ingest.
On a machine where an index is configured, ``--results-index-db none`` is how a
single run is made to read the tree anyway:

.. code-block:: bash

    sd_stats_report --nav-results-root /data/nav-offset-results \
        --results-index-db none

**Reading a tree costs one full read of every document under every root, on
every report run.** That is exactly the cost an index exists to remove -- a
Cassini-scale root is several hundred thousand documents, and on a cloud root
each one is a paid round trip. For a local tree and a single report that is
the right trade; for a cloud root, or for a report you will run more than
once, build an index with ``sd_results_index ingest`` first and name it.

A root, or a directory under one, that cannot be listed fails the run, and no
report is written. A report that quietly covered less than the tree could not
be told apart from one that covered all of it, so the pass stops instead.

``sd_stats_report [--config-file PATH] [--results-index-db URL] [--nav-results-root ROOT] [--root ROOT] [--output-dir DIR] [--instrument NAME] [--start-date YYYY-MM-DD] [--end-date YYYY-MM-DD] [--min-image NAME] [--max-image NAME] [--top-n N] [--filelists] [--suspect-fraction F] [--csv]``
writes ``report.md`` and its charts into the output directory. All filters
combine and apply to every section; dates are inclusive UTC image dates, so a
single day's run is ``--start-date D --end-date D``. An image records its epoch
as the midtime of the observation that was loaded for it, so an image whose load
failed has no date and either bound passes it over. ``--min-image`` /
``--max-image`` bound the numeric portion of the image name (the first digit run
in the basename, so ``--min-image N1454725799`` and ``--min-image 1454725799``
are equivalent); both bounds are inclusive and either may be given alone. The
same inputs always produce the same numbers and the same charts.

``--root`` restricts an index-backed report to one ingested navigation-results
root and may be given more than once; with none given the report covers every
root the index holds a completed ingest of. A root whose newest ingest run did
not finish is passed over rather than counted, because no absence under it can
be read as an image nothing navigated; naming such a root outright is an error
rather than an empty report. A report legitimately spans roots where a
per-image lookup never does.

**A report that passed a root over says so.** Its header names the roots it
covered as the narrowing they are, and a ``Roots dropped`` line under them names
the ones it left out; nothing under a dropped root is reported, neither its
images nor its files that yielded no record, because a half-covered root is
worse than an uncovered one. Ingest such a root and the next report covers it.

Because ``--root`` selects among the roots one index holds, it is a different
thing from ``--nav-results-root``, and it is refused when no index is named
rather than read as a second spelling of it.

Three options control drill-down output:

- ``--top-n N`` makes each categorical section (failure reasons, failure
  taxonomy, ensemble exclusions, suspect offsets) list up to N example
  image names per category and instrument, caps the suspect-offset and
  worst-BOTSIM-pair tables at N rows, and lists the N slowest images. The
  default is 0, which turns the example lists, the worst-pair table and the
  slowest-image list off entirely -- but leaves the suspect-offset table
  uncapped, so it prints one row for every suspect image the selection holds.
- ``--filelists`` writes one plain-text file per category and instrument
  (one image name per line, the full list rather than the top N) into the
  ``filelists/`` subdirectory of the output directory, ready to feed back
  into re-runs and triage scripts.
- ``--csv`` writes ``images.csv`` next to ``report.md``: one row per image,
  ``results_path_stub`` first and then every remaining per-image field in
  schema order -- ``root_url`` through ``mtime_ns`` and ``size_bytes`` -- plus
  ``n_technique_rows``, ``n_feature_sources``, ``n_features`` and ``n_gated``
  aggregates, for pandas or spreadsheet analysis. Rows end with a single
  newline on every platform, and a JSON column that holds nothing is an empty
  cell. One index column stays out: ``status_traceback``, whose value carries
  newlines of its own, so a row holding it would span as many lines as the
  traceback has frames and every line-oriented reader of the file would
  miscount it. Query the index for that column. Each row is written as it is read, so the rows are **not sorted**:
  their order is whatever the storage yields them in -- the walk's own order
  for a tree, the server's own order for an index -- and the two do not agree.
  ``results_path_stub`` is the first column so that putting them in order is
  one shell pipeline, with the header line held out of the sort:

  .. code-block:: bash

      { head -n 1 images.csv; tail -n +2 images.csv | sort -t, -k1,1; } > sorted-images.csv

The first two write *image names* rather than file names -- ``N1454725799``
rather than ``N1454725799_1_CALIB.IMG`` -- because that is the token the
datasets' ``--image-filelist`` option selects on. The filelists are
directly consumable by it: one name per line, with a leading ``#`` comment
naming the category.

Every image count in the report carries its percentage -- ``5 (3.2%)``.
Counts are broken down by instrument: a table of counts gets one column per
instrument plus a total column, where an instrument column's percentage is
of that instrument's images and the total column's is of all selected
images, so each column sums to 100% on its own. Tables of *statistics*
rather than counts (offsets, run time, per-body shares, cross-technique
agreement) carry an instrument column instead, a total being meaningless
for a mean or a standard deviation. Bar charts are stacked, one segment per
instrument, with a fixed color per instrument across every chart.

The report contains:

- **Images selected** -- per instrument: how many images, the first and last
  image, and the first and last available date. Image numbers only compare
  within one instrument, so the bounds are never pooled across instruments.
  The date bounds are found independently of the image ordering, so a
  single image with no recorded epoch at either end of the number range
  cannot hide the instrument's real time span.
- **Files that yielded no record** -- how many files named like a navigation
  document no record could be read out of: one that could not be retrieved,
  one that is not JSON, or JSON that is not a navigation document of the
  current schema. The line is printed whether the count is zero or not,
  because a line that vanished at zero could not be told apart from a report
  that never looked for such files at all. **The count covers the whole of
  every selected root and is not narrowed by ``--instrument``,
  ``--start-date`` or ``--min-image``**: a file no record could be read out of
  carries no instrument, no date and no image number, so it cannot obey a
  filter that compares one. For the same reason it is kept out of the
  per-instrument count tables, whose percentages are of images.

  One of those kinds is counted from a tree and not from an index: a file the
  storage could not deliver at all. The ingest deliberately records no refusal
  for it, because a retrieval that failed once is worth trying again on the next
  pass rather than being remembered as a file that will not read, so a
  tree-backed count of unreadable files can exceed an index-backed one over the
  same root by the number of files that would not come back. Every other kind --
  a file that is not JSON, JSON that is not an object, and JSON that is not a
  navigation document of the current schema -- is counted alike by both, under
  the same reason.
- **Success / failure counts** with a breakdown of failure reasons. The
  reason table carries each reason's status, so errors (SPICE-related or
  not) are visible alongside outright navigation failures.
- **Failure taxonomy by image content** -- failed images classified from
  their recorded feature inventory (``stars-only``, ``single-body``,
  ``multi-body``, ``rings-only``, ``body+rings``, ``no-features``), with a
  per-category failure-reason breakdown and a per-body table of how often
  each named body appears in failed versus successful images (a body with
  a high failure share points at a modeling problem for that body).
- **Technique usage** -- the images each technique ran on, plus a per-
  technique, per-instrument detail table of non-spurious runs and mean
  confidence.
- **Model and source usage** -- which bodies, rings, and star catalogs
  appeared, in how many images, and how many of their features survived the
  reliability gate.
- **Offset statistics** -- mean, median, standard deviation, minimum, and
  maximum of the fused V and U offsets over successful images, grouped by
  camera, with one histogram per camera, plus the same statistics grouped
  by (instrument, camera, image size). Distributions are never pooled
  across cameras: one Cassini WAC pixel is ten NAC pixels, so a pooled
  distribution would describe neither camera.
- **Suspect offsets** -- successful images whose fused offset reaches at
  least ``--suspect-fraction`` (default 0.9) of the instrument's per-axis
  maximum expected pointing offset (the configured ``extfov_margin_vu``
  search margin; for Cassini ISS the NAC/WAC margin chosen from the image
  name) on either axis. An offset pinned near the search boundary may be a
  correlation artifact, so these images deserve operator review. When a
  limit cannot be resolved for an image (unknown instrument, no recorded
  image shape), the report says so rather than silently skipping it.
- **BOTSIM pair consistency (Cassini ISS)** -- BOTSIM observations shutter
  the NAC and WAC simultaneously and the image names share one
  spacecraft-clock count. One WAC pixel is ten NAC pixels, so a consistent
  pair satisfies NAC offset = 10 x WAC offset per axis; the section reports
  the count, median, and 95th percentile of the ``NAC - 10 x WAC``
  residuals over pairs where both frames navigated successfully, and (with
  ``--top-n``) the worst pairs. This is an end-to-end accuracy check that
  needs no ground truth.
- **Cross-technique agreement** -- for every technique pair, the median and
  95th-percentile Euclidean distance between their offsets on images where
  both produced non-spurious results.
- **Confidence calibration** -- per confidence tier, the distribution of each
  image's maximum cross-technique disagreement. The tiers always read
  ``high`` / ``medium`` / ``low`` / ``failed`` / ``conflicted``, so a tier
  with no images reads as an explicit zero rather than a missing row.
  Without ground truth,
  agreement between independent techniques is the production proxy for
  accuracy (the calibrated anchor is the simulation campaign; see
  :doc:`/dev_guide/dev_guide_techniques_confidence`): a healthy pipeline shows
  high-tier images agreeing tightly and disagreement growing toward the low
  tier.
- **Ensemble outlier exclusions** -- how often the ensemble excluded a
  technique from the consensus, and which techniques.
- **Run-time statistics** -- per instrument (and pooled, when more than one
  instrument is selected): minimum, maximum, mean, median, standard
  deviation, and total of the per-image wall-clock run times, a run-time
  histogram, and (with ``--top-n``) the slowest images. The section is
  omitted when no selected image carries timing data.
- **Peak-memory statistics** -- the same shape of table over the largest
  resident size each navigation reached, in GiB, with a histogram and (with
  ``--top-n``) the hungriest images. The maximum is the column that sizes a
  worker, since it is the figure an out-of-memory kill is decided against,
  and the distribution says how much of a pass runs nowhere near it. Each
  peak is what the navigating process reached while that image ran, so a pass
  whose images were navigated one per process reads as their individual costs.
  The image count is of images that recorded a peak rather than of images that
  ran, and the section is omitted when none did.
