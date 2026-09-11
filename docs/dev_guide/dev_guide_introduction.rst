============
Introduction
============

This is the developer manual for **SpinDoctor**, the spacecraft image-navigation system
distributed as the ``rms-spindoctor`` PyPI package. The user manual lives under
:doc:`/user_guide/user_guide` (one chapter per CLI driver); this manual is for contributors and
maintainers — people who modify, extend, build, test, or release the package.

Package overview
================

SpinDoctor ingests images from Cassini ISS, Voyager 1/2 ISS, Galileo SSI, and New
Horizons LORRI; predicts the per-pixel scene geometry from SPICE kernels; and reports
the offset between the predicted and observed scene that brings the two into
alignment. Downstream the same package generates per-pixel backplanes, PDS4 bundles,
body / ring mosaics, and reprojections.

The runtime is pure Python (3.11+), with NumPy / SciPy for numerics, ``oops`` for
geometry and SPICE access, ``starcat`` / ``psfmodel`` for star navigation, ``filecache``
for transparent local-or-remote file access, and PyQt6 for the optional manual-nav and
mosaic-viewer GUIs. No GPU is required.

Repository layout
=================

The repo is a standard ``src``-layout Python package:

::

   rms-spindoctor/
     pyproject.toml              # PEP 621 metadata + tool config (ruff/mypy/pytest)
     README.md                   # PyPI-facing summary
     CONTRIBUTING.md             # contributor checklist
     scripts/
       run-all-checks.sh         # parallel wrapper: ruff + mypy + pytest + sphinx + pymarkdown
     src/
       nav/                      # importable top-level package (`import spindoctor`)
         config/                 # YAML loader + DEFAULT_CONFIG + logger + path helpers
         config_files/           # bundled YAML defaults (numeric-prefix load order)
         dataset/                # DataSet + per-mission subclasses (file enumeration)
         obs/                    # Obs / ObsSnapshot / ObsInst / per-instrument
         feature/                # NavFeature dataclass + geometry / flags / reliability
         nav_model/              # NavModel + per-family subpackages (stars, rings, ...)
         nav_technique/          # NavTechnique + per-technique subclasses + dt_fitting
         nav_orchestrator/       # NavOrchestrator + ensemble + curator + classifier
         annotation/             # Annotation / Annotations (summary-PNG overlay)
         reproj/                 # BodyMosaic, RingMosaic, cartographic_model
         sim/                    # simulated-image renderer
         support/                # cross-cutting helpers (filters, types, file, time, ...)
         ui/                     # PyQt6 widgets (manual nav dialog, mosaic viewer)
         navigate_image_files.py # top-level driver called by sd_offset
       backplanes/               # per-pixel geometry products (driven by sd_backplanes)
       pds4/                     # PDS4 bundle generation (driven by sd_create_bundle)
       spindoctor.cli.reproj/               # CLI-only helpers shared by mosaic drivers
       main/                     # CLI entry-point dispatch modules
     tests/
       nav/                      # unit tests, mirror src/spindoctor layout
       integration/              # operator-curated image library + regression tests
     docs/
       user_guide/               # user manual chapters
       dev_guide/                # this manual
       api_reference/            # autodoc landing pages
       conf.py                   # Sphinx config

The ``src/spindoctor/`` tree is the importable public package; everything else under
``src/`` is supporting code (``backplanes``, ``pds4``, ``spindoctor.cli.reproj``, ``main``).

Install
=======

.. code-block:: bash

   git clone https://github.com/SETI/rms-spindoctor.git
   cd rms-spindoctor
   python -m venv venv
   source venv/bin/activate                  # Windows: venv\Scripts\activate

Editable install of runtime + dev tools (ruff, mypy, pytest, coverage):

.. code-block:: bash

   pip install -e ".[dev]"

The ``[dev]`` group transitively includes the docs group (Sphinx, ``myst-parser``,
``sphinx-rtd-theme``, ``sphinxcontrib-mermaid``). The runtime-only install
(``pip install rms-spindoctor``) is what end users get from PyPI.

Editable installs + mypy: export
``SETUPTOOLS_ENABLE_FEATURES=legacy-editable`` if mypy cannot find the ``nav``
package.

Environment variables
---------------------

Most runs need at least one of these. Each is the last level consulted: a CLI
flag wins over the ``environment`` block of a configuration file, which wins over
the variable exported here. Each accepts a local path or a URL handled
transparently by ``filecache``:

- ``SPICE_PATH`` — required by every real navigation run; points at the SPICE
  kernels directory.
- ``PDS3_HOLDINGS_DIR`` — root for PDS3 holdings; also namable per run with
  ``--pds3-holdings-root``.
- ``PDS4_HOLDINGS_DIR`` — root for PDS4 holdings.
- ``OOPS_RESOURCES`` — root for the ``oops`` resources bundle.
- ``UCAC4_PATH`` / ``YBSC_PATH`` — star-catalog roots used by
  :class:`~spindoctor.nav_model.stars.nav_model_stars.NavModelStars`.
- ``NAV_RESULTS_ROOT`` — output directory for ``_metadata.json`` and
  ``_summary.png`` files written by ``sd_offset``.

Running the CLI tools
=====================

Every CLI listed in ``[project.scripts]`` is installed onto ``$PATH`` by
``pip install``. The full set:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Script
     - Purpose
   * - ``sd_offset``
     - Run autonomous navigation on one image (or a batch); writes JSON
       metadata + summary PNG. Documented at :doc:`/user_guide/user_guide_navigation`.
   * - ``sd_offset_cloud_tasks``
     - Queue-driven navigation variant; reads task JSON, processes one batch.
   * - ``sd_backplanes`` / ``sd_backplanes_cloud_tasks``
     - Generate per-pixel backplanes from a navigated image. See
       :doc:`/user_guide/user_guide_backplanes` and :doc:`/dev_guide/dev_guide_backplanes`.
   * - ``sd_backplane_viewer``
     - PyQt6 viewer for a backplane file.
   * - ``sd_create_bundle`` / ``sd_create_bundle_cloud_tasks``
     - Build a PDS4 bundle from navigated images + backplanes. See
       :doc:`/user_guide/user_guide_pds4_bundle` and :doc:`/dev_guide/dev_guide_pds4`.
   * - ``sd_mosaic`` / ``sd_mosaic_rings`` / ``sd_mosaic_body``
     - Reprojection drivers (rings, body, dispatcher). See
       :doc:`/user_guide/user_guide_reprojection` and :doc:`/dev_guide/dev_guide_reprojection`.
   * - ``sd_mosaic_display`` / ``sd_mosaic_display_rings`` / ``sd_mosaic_display_body``
     - PyQt6 mosaic viewer.
   * - ``sd_mosaic_cloud_tasks``
     - Queue-driven mosaic builder.
   * - ``sd_create_simulated_image``
     - Render a synthetic test image (operator-supplied bodies / rings / stars).
       See :doc:`dev_guide_simulator`.
   * - ``sd_create_ck``
     - Write the navigated attitudes of a results tree into SPICE C-kernels.
       See :doc:`/user_guide/user_guide_ck_kernels`.
   * - ``sd_results_index`` / ``sd_results_index_cloud_tasks``
     - Read a navigation-results tree into the results index, in one process or
       over a queue of workers. ``sd_results_index drop`` empties the index
       instead. See :doc:`/user_guide/user_guide_results_index`.
   * - ``sd_stats_report``
     - Write the navigation statistics report and its charts, over a
       navigation-results tree or over the index. See
       :doc:`/user_guide/user_guide_statistics`.
   * - ``sd_consolidate_metadata``
     - Copy each image's metadata JSON and summary PNG into one flat directory.
       See :doc:`/user_guide/user_guide_navigation`.

Quick smoke test:

.. code-block:: bash

   sd_offset coiss N1234567890 --dry-run

Running the test suite
======================

The suite has three tiers, separated by the ``integration`` and ``postgres``
markers. ``pyproject.toml`` sets ``addopts = ["-m", "not integration and not
postgres"]``, so a plain ``pytest`` runs the default tier alone; pass ``-m ""``
(empty marker filter) to run every tier, or name one marker to run it by itself.
:doc:`dev_guide_testing` describes what each tier holds and what it needs.

.. code-block:: bash

   pytest                                      # default tier (fast)
   pytest -m ""                                # full suite, every tier
   pytest -m integration                       # only integration
   pytest -m postgres                          # only the postgres tier
   pytest -n auto --dist=loadfile              # parallel (matches CI)
   pytest tests/spindoctor/reproj/test_bodies.py      # one file
   pytest tests/spindoctor/reproj/test_bodies.py::test_foo  # one test
   pytest --cov                                # with coverage

``pytest-xdist`` must run with ``--dist=loadfile`` — the default scheduling
crashes PyQt6 workers when tests from one file split across processes.

The integration suite pulls real images from the holdings URLs and uses the
operator-curated regression library at ``tests/integration/image_library/``.
It needs the env vars listed above (in particular ``PDS3_HOLDINGS_DIR``,
``UCAC4_PATH``, ``YBSC_PATH``, ``OOPS_RESOURCES``) plus a ``SPICE_PATH``.
``scripts/run-all-checks.sh -i`` (or ``--integration``) flips the marker for the
all-checks runner.

The postgres tier re-asks the results-index questions against a real PostgreSQL
server. It reads its connection URL from ``SPINDOCTOR_TEST_POSTGRES_URL`` and
skips itself when that is unset. ``scripts/run-all-checks.sh -P`` (or
``--postgres``) flips its marker.

Test markers and layout
-----------------------

- Unit tests live under ``tests/spindoctor/`` and mirror ``src/spindoctor/`` directory by
  directory (``tests/spindoctor/reproj/test_bodies.py`` ↔ ``src/spindoctor/reproj/bodies.py``).
- Integration tests live under ``tests/integration/`` and carry the
  ``integration`` marker. Two sub-layers: the structural-invariants test
  (fast, no holdings needed) and the per-image regression test (slow, needs
  holdings).
- The image library (operator-curated regression cohort) lives at
  ``tests/integration/image_library/``. See :doc:`dev_guide_image_library`
  for the calibration workflow, sidecar schema, and baseline tooling.

Lint, types, formatter
======================

.. code-block:: bash

   ruff check src tests              # lint
   ruff format --check src tests     # format check (CI uses --check; local: drop --check)
   mypy src tests                    # type-check (strict mode is on)
   pymarkdown scan docs/ .cursor/ README.md CONTRIBUTING.md  # markdown lint

Project conventions: line length 100, single-quoted strings, three-group import
order (stdlib / third-party / local), Google-style docstrings with
``Parameters:`` (not ``Args:``) wrapped at 90 chars, at most three positional
parameters per function (rest keyword-only after ``*``), modules under 1000 lines,
no module-level ``# type: ignore`` without a specific error code. See
:doc:`dev_guide_best_practices` for the full list and the rationale.

The all-in-one wrapper:

.. code-block:: bash

   ./scripts/run-all-checks.sh        # ruff + mypy + pytest + sphinx + pymarkdown, parallel
   ./scripts/run-all-checks.sh -i     # same, but include integration tests

Building the documentation
==========================

Documentation source is reStructuredText under ``docs/`` plus Markdown via
``myst-parser``. Sphinx config lives at ``docs/conf.py``. Build with:

.. code-block:: bash

   pip install -e ".[docs]"           # or [dev]
   sphinx-build -W -b html docs docs/_build

The ``-W`` flag promotes warnings to errors; CI runs the same. The built HTML
opens at ``docs/_build/index.html``. Per-format alternates are available from
the ``docs/`` directory's ``Makefile`` (``make latexpdf``, ``make singlehtml``,
``make epub``).

Mermaid diagrams render via ``sphinxcontrib-mermaid``. Validate complex
diagrams in the Mermaid Live Editor (https://mermaid.live/) before committing.

The autodoc API pages under ``docs/api_reference/`` are populated from
docstrings in the source tree; new modules require a corresponding entry under
``docs/api_reference/`` so they appear in the rendered API surface.

CI / CD pipeline
================

GitHub Actions defines three workflows under ``.github/workflows/``:

- ``run-tests.yml`` — runs on every PR and on pushes to ``main``, plus a weekly
  cron. Three jobs:

  - **Lint** — Python 3.13. ``ruff check``, ``ruff format --check``, ``mypy``.
  - **Test** — matrix across Python 3.11 / 3.12 / 3.13 / 3.14 on Ubuntu. Runs
    ``pytest -m "not integration and not postgres" -n auto --dist=loadfile``
    with the holdings / catalog env vars exported. Uploads coverage to Codecov
    from the 3.13 job on pushes (PRs from forks are excluded for secrets
    reasons).
  - **Docs** — ``sphinx-build -W`` against the ``[docs]`` extra.

  Both opt-in tiers are excluded in CI, and the marker expression restates the
  exclusion that ``addopts`` already carries, because a ``-m`` flag replaces
  that expression rather than adding to it. The integration tier — of which the
  operator-curated image-library regression tests are one part — needs real
  PDS3 holdings, SPICE kernels and the star catalogs, and is slow. The postgres
  tier needs a server named by ``SPINDOCTOR_TEST_POSTGRES_URL``, and CI runs no
  service container for one. Maintainers run both locally before merging
  anything that could plausibly regress navigation accuracy or the index.

- ``publish_to_test_pypi.yml`` — runs on pushes to ``main`` once tests pass.
  Builds a wheel + sdist and uploads to TestPyPI under the dev version stamped
  by ``setuptools_scm``.

- ``publish_to_pypi.yml`` — triggered by publishing a GitHub Release on ``main``.
  Builds via ``python -m build`` and uploads to production PyPI using the
  ``PYPI_API_TOKEN`` repo secret.

Release / deployment
====================

Versions are derived automatically from git tags by ``setuptools_scm``; there
is no version string committed to the repo.

Cutting a release:

1. Land all release content on ``main`` (squash-merged).
2. From a clean checkout of ``main``, draft a GitHub Release and tag it
   ``vX.Y.Z`` (semver). Publishing the release triggers
   ``publish_to_pypi.yml``.
3. Verify the new version appears on https://pypi.org/project/rms-spindoctor/ and
   that ``pip install rms-spindoctor==X.Y.Z`` works in a clean venv.
4. ReadTheDocs builds the new tag automatically from the same Git ref; verify
   the tagged version appears in the version selector at
   https://rms-spindoctor.readthedocs.io/.

Pre-release iteration uses TestPyPI (``publish_to_test_pypi.yml``); install
from there with
``pip install --index-url https://test.pypi.org/simple/ rms-spindoctor``.

Contribution workflow
=====================

The contributor checklist is at :doc:`/contributing`. Highlights:

- Conventional Commits subjects (``feat:``, ``fix:``, ``docs:``, ``refactor:``,
  ``test:``, ``perf:``, ``ci:``, ``chore:``, ``style:``). Subject imperative,
  ≤ 50 chars, no trailing period. One logical change per commit.
- PRs squash-merge to ``main``; the squash message becomes the release notes
  entry.
- Every PR must pass ``./scripts/run-all-checks.sh`` locally and the same
  checks in CI.
- Static-data values (per-body shape, per-instrument calibration) require a
  citation traceable to a fetched document; see :doc:`dev_guide_config_and_static_data`.

Pointers to the rest of this manual
===================================

- :doc:`dev_guide_navigation` — how the navigation pipeline is structured and
  where each subsystem lives.
- :doc:`dev_guide_reprojection` — body / ring mosaicing.
- :doc:`dev_guide_backplanes` — per-pixel backplane generation.
- :doc:`dev_guide_pds4` — PDS4 bundle generation.
- :doc:`dev_guide_support` — config / static data / logging.
- :doc:`dev_guide_extending` — adding a new instrument, model, or technique.
- :doc:`dev_guide_best_practices` — coding conventions.
- :doc:`/api_reference` — autodoc API surface.
- :doc:`/contributing` — contributor checklist.
