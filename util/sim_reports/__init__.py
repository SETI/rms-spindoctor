"""Simulator report, figure, and characterization programs that run on demand.

None of these is a test.  Each renders or navigates simulated frames in-process
(no holdings or SPICE, except where noted) and writes a report, a set of figures,
or a table.  Run each from the root of the checkout, so that its ``src`` and
``tests`` are the ones imported::

    python -m util.sim_reports.sim_doc_images
    python -m util.sim_reports.sim_sweep_runner [--only NAME] [--dump-images DIR]
    python -m util.sim_reports.sim_sweep_plots
    python -m util.sim_reports.technique_snr_characterization
    python -m util.sim_reports.star_snr_characterization
    python -m util.sim_reports.limb_bias_runner

``sim_doc_images`` writes the galleries of the developer guide and the simulator
report.  ``sim_sweep_runner`` runs every parameter sweep of
``tests/integration/sim_sweeps/`` and writes the per-step JSON, then draws the
response curves with ``sim_sweep_plots``.  The two SNR characterizations write
the accuracy-versus-SNR figures of the report.  ``limb_bias_runner`` writes the
limb-bias tables under ``util/calibration/limb_bias/`` and needs
``PDS3_HOLDINGS_DIR`` for its real-frame block.

The tests under ``tests/`` here check what these programs write.  CI does not run
them; run them with ``pytest util/sim_reports/tests -m ""``.
"""

# Importing ``tests`` first applies its NumPy SIMD pin (``NPY_DISABLE_CPU_FEATURES``)
# before any program here imports NumPy, so their renders match the committed images on
# every x86-64 machine.  ``python -m util.sim_reports.<name>`` imports this package
# before the program module.
import tests  # noqa: F401
