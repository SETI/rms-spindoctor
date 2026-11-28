"""The realism match: an on-demand comparison of simulated frames with real ones.

Nothing in the test suite or the distributed package runs or imports this program.
Run it from the root of a checkout, with the holdings and star-catalog environment
variables of a real navigation run set::

    python -m util.sim_realism.sim_realism [--instrument NAME] [--max-frames N]
        [--skip-fom7] [--skip-law-study]

It writes its figures to ``docs/simulator_report/_figures/`` and its record to
``util/sim_realism/realism_results/realism_summary.json``.  The simulator developer
guide describes the match, the photometric-law study, and the band tool
(``python -m util.sim_realism.realism_regression``).  The unit tests of the helpers
are in ``util/sim_realism/tests/`` and run with
``pytest util/sim_realism/tests`` from the checkout root.
"""
