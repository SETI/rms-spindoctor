"""Navigation statistics system: metadata ingestion and reporting.

Two programs, one of which builds the results index that
:mod:`spindoctor.results_index` defines and one of which reads navigation
records from wherever they are kept:

- ``sd_results_index`` (:mod:`spindoctor.cli.results_index`, with the column
  mapping in :mod:`spindoctor.nav_records.facts`) walks one or more
  navigation-results roots -- local paths or any URL accepted by ``FCPath`` --
  reads every ``*_metadata.json`` written by ``navigate_image_files``, and
  writes one row per image keyed by its root and its results path stub.  A
  document whose recorded size and modification time still match the tree is
  not read again, so a second pass over an unchanged root costs one listing.
- ``sd_stats_report`` (:mod:`spindoctor.cli.stats.report`, with the pass over
  the records in :mod:`spindoctor.cli.stats.report_accumulate`, the section
  builders in :mod:`spindoctor.cli.stats.report_sections`, the chart writers
  in :mod:`spindoctor.cli.stats.report_charts`, and the shared state and
  helpers in :mod:`spindoctor.cli.stats.report_common`) reads every record
  once and emits a deterministic report (Markdown text plus PNG charts):
  success/failure counts with failure reasons, a failure taxonomy by scene
  content with per-body failure shares, technique and model usage, offset
  statistics (overall and by instrument/image size), a suspect-offset screen
  against the configured search limits, Cassini BOTSIM NAC/WAC pair
  consistency, cross-technique agreement, confidence-tier calibration against
  observed agreement, and run-time statistics.  Optional drill-down flags list
  example images per category, write per-category filelists, and export a
  one-row-per-image CSV.

Both are cron-friendly single commands and work over any date range and any
instrument.  Both take the index as a connection URL, so the same pair of
commands works against a local SQLite file and against a PostgreSQL server.
The report needs no index: named one, it reads its rows, and named none it
reads the documents themselves, one file read per image in place of one query
per run.
"""

from spindoctor.cli.stats.report import main_report

__all__ = ['main_report']
