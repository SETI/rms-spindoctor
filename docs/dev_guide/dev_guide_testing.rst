=======
Testing
=======

Overview
========

The test suite has three tiers, separated by the ``integration`` and
``postgres`` markers. ``addopts`` in ``pyproject.toml`` carries ``-m "not
integration and not postgres"``, so a plain ``pytest`` runs the default tier
alone:

- The **default tier** runs on a plain ``pytest``. Everything in it is fast and
  self-contained: it needs no spacecraft holdings, no SPICE kernels, and no
  network. Unit tests live here, and so do the in-process simulator tests --
  every simulated frame is rendered and navigated in memory, so the whole
  simulator-driven invariant and structural coverage runs without external data.
- The **integration tier** is opted into with ``-m ""`` or ``-m integration``.
  It holds the slow and the archive-backed tests: the real-image
  regression cohort (which fetches PDS holdings and resolves SPICE geometry) and
  the heavier in-process simulator tests. The simulator tests carry a second
  marker, ``sim_integration``: they need no holdings, SPICE kernels, or star
  catalogs, so CI runs them on every pull request, and so does
  ``./scripts/run-all-checks.sh`` without ``-i``. There are two exceptions. The
  simulator performance budgets measure CPU time and need a quiet host, and the
  realism smoke (``test_sim_realism_smoke.py``) and the realism regression check
  (``test_sim_realism_regression.py``) read real frames.
- The **postgres tier** is opted into with ``-m postgres`` (or ``-m ""``). It
  re-asks the results-index questions against a real PostgreSQL server, because
  SQLite accepts spellings a server rejects: an integer compared against a
  boolean, a single-precision float where a double was meant, a foreign key that
  is only enforced when asked. The tests read their connection URL from
  ``SPINDOCTOR_TEST_POSTGRES_URL`` and skip themselves when it is unset, so the
  tier is runnable on any machine with a server and harmless on any machine
  without one. Each test creates and drops a PostgreSQL schema of its own, so
  repeat runs and parallel workers do not collide. CI has no service container
  for it; run it locally before changing anything in
  :mod:`spindoctor.results_index`
  or the statistics programs.

The simulator (:doc:`dev_guide_simulator`) is the engine behind several tiers: it
lets the suite grow algorithmic-invariant and sensitivity coverage on frames
whose true offset is known by construction, without operator labor and without
real data.

Running the suite
=================

.. code-block:: bash

   pytest                              # default tier (fast, no holdings)
   pytest -m ""                        # full suite, every tier
   pytest -m integration               # only the integration tier
   pytest -m sim_integration           # only its simulator tests
   pytest -m postgres                  # only the postgres tier
   pytest -n 4 --dist=loadfile         # parallel (loadfile avoids PyQt6
                                       #   worker crashes)
   pytest tests/spindoctor/sim/test_sim_noise.py            # one file
   pytest tests/spindoctor/sim/test_sim_noise.py::test_foo  # one test
   pytest --cov                        # with coverage

   ./scripts/run-all-checks.sh         # ruff + mypy + pytest + docs + markdown
   ./scripts/run-all-checks.sh -i      # the same, including integration tests
   ./scripts/run-all-checks.sh -P      # the same, including the postgres tier

The postgres tier additionally needs a server to point at:

.. code-block:: bash

   export SPINDOCTOR_TEST_POSTGRES_URL=postgresql+psycopg://USER@HOST:5432/spindoctor

Supply the credentials the way the server expects them -- a ``~/.pgpass`` entry,
``PGPASSWORD``, or a peer-authenticated local socket -- rather than writing a
password into a shell profile or a checked-in file. A password inside the URL
works, and the tier masks it out of every message, but the URL itself is then a
secret in the environment of every process the shell starts.

``pytest-xdist`` must run with ``--dist=loadfile``; the default scheduling
crashes PyQt6 workers when tests from one file split across processes. Run it at
``-n 4``, never ``-n auto``, and pin the BLAS / OpenMP thread counts first:

.. code-block:: bash

   export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

Integration runs take the same flags.

**The suite opens no results index it was not handed.** An index URL resolves
from an argument, then from the ``environment.results_index_db`` configuration
variable, then from ``NAV_RESULTS_INDEX_DB``. A test of what a program does
without a results index sets none of the three. Both ambient levels are closed
by fixtures in ``tests/conftest.py``, which unset the environment variable and
run from a directory holding no ``nav_default_config.yaml``: once for the whole
session, so that a fixture of any scope is built with both closed, and again
around each test, so that what a test sets up for itself is undone afterwards.
So a suite started in a directory whose configuration sets a live results index
neither reads nor locks it. For a SQLite results index, an open takes a
write-lock probe against a file an ingest may be holding.

A test that wants either level sets it up for itself. These fixtures cannot do two
things for you. First, a subprocess started in a working directory of its own
reads its configuration there, so a test that starts one must choose that
directory deliberately. Second, every test in a worker runs from the same
directory. Anything a test leaves there is removed when the test ends. A test
that leaves a ``nav_default_config.yaml`` there also fails, so that the mistake
is reported at the test that made it: had the file stayed, every later test in
that worker would have read it as its configuration. Move to a directory of your
own with ``monkeypatch.chdir`` before writing anything.

Archive-backed tests additionally require the holdings and catalog environment
(set by CI; see :doc:`dev_guide_introduction`):

.. code-block:: bash

   export PDS3_HOLDINGS_DIR=https://pds-rings.seti.org/holdings
   export PDS4_HOLDINGS_DIR=https://pds-rings.seti.org/pds4
   export OOPS_RESOURCES=https://storage.googleapis.com/rms-node-oops-resources
   export UCAC4_PATH=https://storage.googleapis.com/rms-node-star-catalogs/UCAC4
   export YBSC_PATH=https://storage.googleapis.com/rms-node-star-catalogs/YBSC
   # plus SPICE kernels at $SPICE_PATH for any real navigation run

Test kinds
==========

The suite is layered by what a test proves and what it needs. The simulator-only
tests need none of the archive environment above.

.. list-table::
   :widths: 28 14 16 42
   :header-rows: 1

   * - Kind (path)
     - Tier
     - Requires
     - What it proves
   * - Unit tests (``tests/spindoctor/**``)
     - default
     - nothing
     - One component in isolation (config, feature, dataset, obs, model,
       technique, orchestrator, reproj, support).
   * - Results index on a server
       (``tests/spindoctor/results_index/test_postgres.py``,
       ``tests/spindoctor/results_index/test_drop_postgres.py``,
       ``tests/spindoctor/results_index/test_creating_open_postgres.py``)
     - postgres
     - PostgreSQL
     - The schema, the version gate, and the type discipline against a backend
       that enforces them, rather than against SQLite's permissive typing; the
       questions only a server can be asked of a drop -- that it leaves another
       owner's tables standing, including one with the same name as one of the
       results index's own tables; that a search path crossing schemas cannot
       make one drop span two of them; that an emptied database is the same
       database as one nothing was built in; that a stamp column this account
       may not read costs the version and not the drop; and that a lock somebody
       else holds ends it rather than hanging it. And the questions only a
       server can be asked of a creating open: that the schema it examines is
       the one the index resolves to rather than the database around it, and
       that a table of one of the index's own names in another schema of the
       search path is neither adopted nor built around.
   * - Statistics programs on a server
       (``tests/spindoctor/cli/stats/test_postgres.py``, and the
       ``postgres``-marked tests of ``tests/spindoctor/cli/stats/**``)
     - postgres
     - PostgreSQL
     - That the command lines behave the same way against a server as against a
       file: the same exit statuses for a database holding no index and for one
       that is not there, and a failure named as what the server said it was.
   * - Simulator unit tests (``tests/spindoctor/sim/**``)
     - default
     - nothing
     - The renderer's contracts: determinism, noise, saturation, PSF, stray
       light, instrument coupling, camera roll, irregular-body rendering and the
       ``nav_override`` channel, scene-schema validation, and the information
       boundary (``test_information_boundary.py``: no truth key is reachable
       through ``obs.nav_params``).
   * - GUI smoke (``tests/main/test_create_simulated_image.py``,
       ``tests/main/test_sim_editor_round_trip.py``)
     - default
     - PyQt6
     - Each GUI control wires to the right ``sim_params`` field and a YAML
       scene round-trips through the editor without loss.
   * - Scene structural (``test_sim_scenes.py``)
     - default
     - nothing
     - Every catalog scene validates, sits in a declared class, has a unique
       name, and renders.
   * - Algorithmic invariants (``test_sim_algorithmic_invariants.py``)
     - default
     - nothing
     - Each technique recovers its planted offset / roll on a clean scene --
       correct by construction, so no baseline.
   * - End-to-end sim nav (``test_sim_navigation.py``)
     - default
     - nothing
     - A simulated frame navigates through the full orchestrator and recovers a
       planted offset.
   * - Sim bug regression (``test_sim_regression.py``)
     - default
     - nothing
     - Fast bug-specific scenes guarding defects the sweeps surfaced.
   * - Sim regression baselines (``test_sim_baselines.py``)
     - integration (``sim_integration``)
     - nothing
     - Every catalog scene re-navigates to its recorded outcome, within small
       tolerances on the offset and the confidence (a tripwire). Integration-marked
       because rendering and navigating every scene takes minutes; see
       :ref:`reproducing-sim-baselines`.
   * - Sensitivity sweeps (``test_sim_sweeps.py``)
     - integration (``sim_integration``)
     - nothing
     - Each single-variable sweep responds as expected (a technique transition, a
       degradation to failure, recovery within tolerance). Heavier -- each sweep
       navigates several frames.
   * - Pose behavioral (``test_sim_irregular_pose.py``)
     - integration (``sim_integration``)
     - nothing
     - On a wrong-pose irregular body the limb degrades far off while the
       pose-free blob stays accurate (asserted per technique).
   * - Real-image structural (``test_image_library.py``)
     - default
     - nothing
     - The operator-curated sidecar catalog validates structurally (schema,
       classes, uniqueness) without fetching images.
   * - Real-image regression (``test_autonomous_nav.py``,
       ``test_baselines.py``)
     - integration
     - holdings + SPICE
     - Each curated real image navigates to its expected status / tier / offset
       and matches its recorded baseline -- the calibration tripwire.

The simulator's role across these tiers is summarized in
:doc:`dev_guide_simulator`. The operator-curated real-image cohort is documented
in :doc:`dev_guide_image_library`.

Characterization runners and updaters
=====================================

These are not pytest tests; they are ``python -m`` scripts that produce the
report figures, the example images, and the regression baselines. They render and
navigate in-process and need no holdings.

.. list-table::
   :widths: 42 58
   :header-rows: 1

   * - Command
     - Produces
   * - ``python -m tests.integration.sim_sweep_runner``
     - Per-sweep response-curve JSON under ``sim_sweeps/results/`` (gitignored)
       and the offset / star / roll / mesh figures in the report. Add
       ``--dump-images DIR`` to also write every sweep frame as a PNG.
   * - ``python -m tests.integration.technique_snr_characterization``
     - The per-technique accuracy-vs-SNR and accuracy-vs-offset report figures.
   * - ``python -m tests.integration.star_snr_characterization``
     - The star-field centroiding (moment / PSF / adaptive) report figures.
   * - ``python -m tests.integration.sim_doc_images``
     - The developer-guide scene gallery and the report scene images.
   * - ``python -m tests.integration.update_sim_baselines``
     - Regenerates the simulator regression baselines under ``sim_baselines/``.
   * - ``python -m tests.integration.update_baselines``
     - Regenerates the real-image regression baselines (needs holdings).
   * - ``python -m tests.mini_nav_results results_tree
       tests/spindoctor/cli/stats/data/results_tree``
     - Rewrites the documents of the stored statistics fixture tree through the
       navigation metadata writer. The frozen report output has to be
       re-ratified against whatever it wrote. Documents are all it writes: the
       stored tree also holds an empty browse image beside three of them, which
       nothing reads, so an empty directory this is pointed at gets the
       documents alone.
   * - ``python -m tests.mini_nav_results cohort <bundle> <outdir>``
     - Writes one bundle's cohort, ``<bundle>`` being a name in the package's
       ``COHORTS`` registry -- navigation documents, browse PNGs,
       backplane FITS files and their metadata -- for calling the bundle stage
       over without navigating anything. It is not a holdings tree, so
       ``sd_create_bundle`` cannot enumerate it. Nothing it writes is checked
       in; the suite builds its own copy into a temporary directory.

After a deliberate change that shifts a baseline or a figure, rerun the relevant
updater or runner and review the diff before committing -- the baselines are
tripwires, so an unexpected change is a regression to investigate, not to bless
blindly. A new invariant scene also needs a baseline (``update_sim_baselines``);
verify no existing baseline shifts.

.. _reproducing-sim-baselines:

Reproducing the simulator baselines
===================================

A simulator baseline records a scene's outcome: its status, its confidence rank,
its offset, and its confidence, the last two as computed, without rounding. The
test compares the status and the confidence rank exactly.

It compares each axis of the offset within 0.02 px, because some techniques
answer on a grid. The disc correlation's offset moves in steps of 1/128 px, one
step of its upsampled grid, and a last-bit change upstream can move it one step.
The tolerance is about two and a half of those steps.

It compares the confidence within 0.02, a value chosen by hand. It can be
revisited once runs on other machines show how far a scene's confidence moves.
The rank comes from thresholds on the confidence and on the offset's uncertainty,
so a scene that sits near one of those thresholds can still change rank, and
fail, after a change far smaller than either tolerance.

The render-diff check compares every committed render with a fresh one, pixel
for pixel. It compares the decoded pixels, so a change in the PNG encoder doesn't
fail it. The gallery check compares PNGs byte for byte. All of this works because
a simulated render and its navigation are deterministic. A scene gives the same
result in a fresh process and in a long-lived ``pytest-xdist`` worker, whatever
ran before it.

What can change the result is the CPU. NumPy picks vectorized code paths by CPU
(AVX2, AVX-512, and so on), and the paths differ in the last bits of their
results. That can move a pixel of a render, a gallery image's bytes, or a sweep
check that sits near its threshold. So the test package pins NumPy to the
x86-64-v2 baseline: before NumPy is first imported, ``tests/__init__.py`` sets
``NPY_DISABLE_CPU_FEATURES`` to every dispatch group above it.
``tests/integration/test_simd_pin.py`` fails if the pin didn't take effect. The
pin covers ``pytest`` and the ``python -m tests.integration...`` tools alike.

The pin doesn't reach two other libraries that NumPy calls. OpenBLAS, which runs
``@``, picks its kernel by CPU, and the C library's ``hypot``, which runs
``np.hypot``, differs between CPUs in its last bit. So the simulator's render
path uses neither. The mesh renderer calls
:func:`spindoctor.support.stable_math.stable_matmul3`, which sums its 3-vector
products in a fixed order, in place of ``@``, and the simulator calls
:func:`spindoctor.support.stable_math.stable_hypot` in place of ``np.hypot``.
``tests/spindoctor/sim/test_sim_stable_math_guard.py`` fails if a simulator module
takes a matrix product with ``@``, ``.dot``, or a NumPy matrix-product function, or
calls ``np.hypot``.

A last-bit change in the render path stays small. The PSF's FFT convolution
spreads it to the mean of every pixel, but the detector draws its noise one
pixel at a time. Each pixel takes one uniform value for its shot noise and one
normal value for its read noise, however bright or dark the scene is. A changed
mean can therefore change only its own pixel's noise. It does so only when the
pixel's uniform value sits within that last bit of a step in the Poisson
distribution, a chance of about 1e-13 per pixel, and then by one count. That is
why a noisy render can be compared exactly.

The pin matters for the checks that are exact: the committed renders and the
gallery images, which are compared pixel for pixel or byte for byte, and the
sweep checks, which can sit near a threshold. A last-bit change can move a pixel
wherever a mask edge or a rounding step falls on it.

On x86-64, with NumPy 2.4 or later, and with ``NPY_DISABLE_CPU_FEATURES`` not
already set, every machine then reproduces the same files. They reproduced under
NumPy 2.4 and 2.5, and under SciPy and ``rms-psfmodel`` releases months apart. A
value already in the variable overrides the pin, and the files it regenerates may
then differ. The pin doesn't apply below NumPy 2.4, or on an ARM machine, whose
results may differ.

CI is the reference environment: GitHub's ``ubuntu-latest`` runner, at each
supported Python version. Every pull request runs the simulator checks there, so
a baseline that CI doesn't reproduce fails the pull request.

A NumPy release can change how a random stream turns into values. That moves the
noise in every noisy render, and it also moves star counts, star positions,
cosmic rays, and hot pixels. The render-diff check and the gallery check fail,
and baselines move too. Regenerate the committed renders, the galleries, and the
baselines together then, as you would after a deliberate change to what a scene
renders, and review the sheets, the galleries, and the baseline diff:

.. code-block:: bash

   python -m tests.integration.render_contact_sheet
   python -m tests.integration.sim_doc_images
   python -m tests.integration.update_sim_baselines

When a change moves a baseline, regenerate it in that change and say in the
commit message why it moved. A baseline left stale fails CI and gets blamed on
whatever change is in flight when somebody notices.

Run the updaters against the checkout's own source. ``pytest`` puts ``src`` on
the import path itself, but ``python -m`` imports ``spindoctor`` from the active
environment, which may have a different checkout installed. Every runnable
script in ``tests/integration`` checks this first
(``tests/integration/checkout_guard.py``): when the imported ``spindoctor`` lies
outside the script's own checkout, it stops before writing anything and reports
both paths. Unless this checkout is the one installed, run:

.. code-block:: bash

   PYTHONPATH=src python -m tests.integration.update_sim_baselines

A new runnable script calls ``require_checkout_package(__file__)`` as the first
statement of its ``__main__`` block; ``tests/integration/test_checkout_guard.py``
fails for one that doesn't.
